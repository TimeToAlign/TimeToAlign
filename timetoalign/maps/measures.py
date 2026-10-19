"""Measure views: conversion maps that read a measure map.

A measure map lists a work's measures in printed order, each with its
position in quarter notes (``qstamp``), its lengths and its number. The four
views read one:

- MeasureRecordMap: quarters -> the measure (record) holding the position
- MeasureNumberMap: quarters -> that measure's number
- FloatingMeasureMap: quarters -> floating measures
- MeasureBeatMap: quarters -> {measure number, beat}

They are the TimeSkeleton format's measure views, and every measure-derived
map of the library is written as one of them. The measure holding a position
is the last measure whose ``qstamp`` lies at or before it; a position before
the first measure is held by the first, one after the last by the last.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from ..core.enums import TimeUnit
from ..core.time import CoordinateValue
from .base import ConversionMap, MeasureMapIds
from .interval import QuartersToFloatingMeasures

if TYPE_CHECKING:
    from typing_extensions import Self

    from timetoalign.timelines.structure import MeasureMap


def _named_measure_map(
    data: Mapping[str, Any],
    measure_maps: Mapping[str, Any],
    target_unit: TimeUnit | None,
) -> MeasureMap:
    """Return the measure map a view's form names.

    Raises:
        ValueError: If the form's units are not the view's (quarters, and
            *target_unit*), or it names no measure map, or one
            *measure_maps* lacks.
    """
    view = data.get("type")
    expected = (
        TimeUnit.quarters.value,
        None if target_unit is None else target_unit.value,
    )
    if (data.get("source_unit"), data.get("target_unit")) != expected:
        raise ValueError(
            f"a {view} reads {expected[0]!r} as {expected[1]!r}, the units of every "
            f"measure view of its kind, not {data.get('source_unit')!r} as "
            f"{data.get('target_unit')!r}"
        )
    if "measure_map" not in data:
        raise ValueError(f"a {view} lacks the key(s) ['measure_map']")
    measure_map_id = data["measure_map"]
    if measure_map_id not in measure_maps:
        raise ValueError(
            f"a {view} names the measure map {measure_map_id!r}, which the "
            f"manifest's measure_maps do not hold. Available measure maps: "
            f"{list(measure_maps)}"
        )
    return measure_maps[measure_map_id]


class _MeasureView(ConversionMap[Any]):
    """A map reading the measure of a measure map that holds a position.

    Subclasses answer something of that measure; none is invertible.
    """

    _view_target_unit: ClassVar[TimeUnit | None] = None

    def __init__(
        self,
        measure_map: MeasureMap,
        *,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a view over *measure_map*.

        Args:
            measure_map: The measure map read; every measure states its
                ``qstamp``.
            uid: Optional explicit ID.
            name: Human-readable name for this map.

        Raises:
            ValueError: If the measure map is empty or a measure states no
                ``qstamp``.
        """
        super().__init__(
            source_unit=TimeUnit.quarters,
            target_unit=self._view_target_unit,
            uid=uid,
            name=name,
        )
        records = tuple(measure_map.measures)
        if not records:
            raise ValueError(f"A {type(self).__name__} requires at least one measure")
        if any(record.qstamp is None for record in records):
            raise ValueError(
                f"A {type(self).__name__} requires every measure to state its qstamp"
            )
        self._measure_map = measure_map
        self._records = records
        self._starts = np.array([float(r.qstamp) for r in records], dtype=np.float64)

    @property
    def measure_map(self) -> MeasureMap:
        """The measure map this view reads."""
        return self._measure_map

    @property
    def is_invertible(self) -> bool:
        """A view is not invertible: many positions lie in one measure."""
        return False

    def inverse(self) -> Self:
        """A view has no inverse.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is not invertible: many positions lie in one measure"
        )

    def _index(self, value: CoordinateValue) -> int:
        """The index of the measure holding *value*, clamped to the measures."""
        index = int(np.searchsorted(self._starts, float(value), side="right")) - 1
        return max(0, min(index, len(self._records) - 1))

    def _indices(self, values: NDArray[Any]) -> NDArray[Any]:
        """The index of the measure holding each of *values*."""
        indices = np.searchsorted(self._starts, values.astype(np.float64), side="right")
        return np.clip(indices - 1, 0, len(self._records) - 1)

    def _number(self, index: int) -> int:
        """The number of the measure at *index*: its count where it states none."""
        record = self._records[index]
        return record.number if record.number is not None else record.count

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the view naming its measure map's id."""
        return self._map_object(
            self._manifest_type,
            {"measure_map": entries.id_of(self._measure_map)},
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> Self:
        """Read the view over the measure map it names."""
        return cls(
            _named_measure_map(data, measure_maps, cls._view_target_unit),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def _measure_maps(self) -> Iterator[Any]:
        yield self._measure_map

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self._id!r}, measures={len(self._records)})"


# region MeasureRecordMap


class MeasureRecordMap(_MeasureView):
    """Map quarters to the measure holding the position.

    A timeline's measure map is written as the measure map this view names.

    Examples:
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows([{"count": 1, "time_signature": "3/4"}, {"count": 3}])
        >>> MeasureRecordMap(bars)(4).id
        '2'
    """

    _manifest_type: ClassVar[str | None] = "MeasureRecordMap"

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> Any:
        """Return the measure holding *value*."""
        return self._records[self._index(value)]

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Return the measure holding each position, as an object array."""
        result = np.empty(values.shape, dtype=object)
        for position, index in np.ndenumerate(self._indices(values)):
            result[position] = self._records[index]
        return result


# endregion


# region MeasureNumberMap


class MeasureNumberMap(_MeasureView):
    """Map quarters to the number of the measure holding the position.

    The number is the measure map row's ``number``, its count where the row
    states none.

    Examples:
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows([{"number": 0, "actual_length": 1, "time_signature": "4/4"}, {}])
        >>> numbers = MeasureNumberMap(bars)
        >>> numbers(0), numbers(1), numbers(9)
        (0, 1, 1)
    """

    _manifest_type: ClassVar[str | None] = "MeasureNumberMap"

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> int:
        """Return the number of the measure holding *value*."""
        return self._number(self._index(value))

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Return the number of the measure holding each position."""
        numbers = np.array(
            [self._number(index) for index in range(len(self._records))], dtype=np.int64
        )
        return numbers[self._indices(values)]


# endregion


# region MeasureBeatMap


class MeasureBeatMap(_MeasureView):
    """Map quarters to ``{"number": <measure number>, "beat": <beat>}``.

    The beat counts quarters from the measure's ``qstamp``, the first beat
    being 1: position ``qstamp + 1/2`` is beat ``3/2``. An exact position
    gives an exact beat; a float is read as the exact value it holds.

    Examples:
        >>> from fractions import Fraction
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows([{"count": 1, "time_signature": "4/4"}, {"count": 2}])
        >>> MeasureBeatMap(bars)(Fraction(11, 2))
        {'number': 2, 'beat': Fraction(5, 2)}
    """

    _manifest_type: ClassVar[str | None] = "MeasureBeatMap"

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> dict[str, Any]:
        """Return the number of the measure holding *value* and the beat in it."""
        index = self._index(value)
        exact = value if isinstance(value, Fraction) else Fraction(float(value))
        beat = exact - self._records[index].qstamp + 1
        return {"number": self._number(index), "beat": beat}

    def _convert_array(
        self, values: NDArray[Any], **kwargs: Any
    ) -> dict[str, NDArray[Any]]:
        """Return the numbers and the beats as float arrays, one per output."""
        indices = self._indices(values)
        numbers = np.array(
            [self._number(index) for index in range(len(self._records))], dtype=np.int64
        )
        return {
            "number": numbers[indices],
            "beat": values.astype(np.float64) - self._starts[indices] + 1.0,
        }


# endregion


# region FloatingMeasureMap


class FloatingMeasureMap(QuartersToFloatingMeasures):
    """The floating-measure lattice of a measure map, remembering the map.

    Converts as :class:`QuartersToFloatingMeasures` built from the same
    measure map, and is written as the format's ``FloatingMeasureMap``
    naming it.

    Examples:
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows([{"count": 1, "time_signature": "4/4"}, {"count": 2}])
        >>> FloatingMeasureMap(bars)(6)
        2.5
    """

    _manifest_type: ClassVar[str | None] = "FloatingMeasureMap"

    def __init__(
        self,
        measure_map: MeasureMap,
        *,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize the lattice of *measure_map*.

        Args:
            measure_map: Measures whose exact starts and lengths define the
                lattice.
            uid: Optional explicit ID.
            name: Human-readable name (default: "quarters_to_measures").

        Raises:
            ValueError: If the measure map has no lattice (see
                :meth:`QuartersToFloatingMeasures.from_measure_map`).
        """
        x_values, y_values = QuartersToFloatingMeasures._lattice(measure_map)
        super().__init__(x_values=x_values, y_values=y_values, uid=uid, name=name)
        self._measure_map = measure_map

    @property
    def measure_map(self) -> MeasureMap:
        """The measure map this lattice reads."""
        return self._measure_map

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``FloatingMeasureMap`` naming its measure map's id."""
        return self._map_object(
            "FloatingMeasureMap",
            {"measure_map": entries.id_of(self._measure_map)},
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> FloatingMeasureMap:
        """Read the lattice of the measure map the form names."""
        return cls(
            _named_measure_map(data, measure_maps, TimeUnit.floating_measures),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def _measure_maps(self) -> Iterator[Any]:
        yield self._measure_map

    def __repr__(self) -> str:
        n = len(self._x) - 1
        return f"FloatingMeasureMap(n_measures={n})"


# endregion
