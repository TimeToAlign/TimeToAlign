"""Composite conversion maps.

This module provides maps that are composed of other maps:
- ChainMap: Applies a sequence of maps (f(g(x)))
- PiecewiseMap: Applies different maps based on the input coordinate
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.enums import NumberType, TimeUnit
from ..core.time import (
    CoordinateValue,
    number_from_json,
    number_to_json,
)
from .base import ConversionMap, MeasureMapIds

if TYPE_CHECKING:
    from typing_extensions import Self


def _read_maps(
    data: Mapping[str, Any], key: str, measure_maps: Mapping[str, Any]
) -> list[ConversionMap[Any]]:
    """Read the array of maps under *key*.

    Raises:
        ValueError: If the value is not an array, or a map is malformed.
    """
    items = data.get(key)
    if not isinstance(items, list):
        raise ValueError(f"a {data.get('type')}'s {key!r} must be an array")
    return [ConversionMap.from_dict(item, measure_maps=measure_maps) for item in items]


def _piecewise_form(
    cmap: ConversionMap[Any],
    breaks: Sequence[CoordinateValue],
    maps: list[dict[str, Any]],
    *,
    minimal: bool,
) -> dict[str, Any]:
    """Write *cmap* as the format's ``PiecewiseMap`` of *breaks* and written *maps*."""
    return cmap._map_object(
        "PiecewiseMap",
        {"breaks": [number_to_json(b) for b in breaks], "maps": maps},
        minimal=minimal,
    )


# region ChainMap


class ChainMap(ConversionMap[CoordinateValue]):
    """Composition of multiple maps: f(g(h(x))).

    The maps are applied in sequence from first to last.
    The output unit of one map must match the input unit of the next
    (if specified).

    The format's ``ChainMap`` type: ``maps`` in the order they apply.

    Attributes:
        maps: The sequence of maps to apply.

    Examples:
        >>> # Ticks -> Seconds -> Milliseconds
        >>> ticks_to_secs = ScalarMap(scalar=1/480, source_unit="ticks", target_unit="seconds")
        >>> secs_to_ms = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        >>> chain = ChainMap([ticks_to_secs, secs_to_ms])
        >>> chain(480)
        1000.0
    """

    _manifest_type: ClassVar[str | None] = "ChainMap"

    def __init__(
        self,
        maps: Sequence[ConversionMap[Any]],
        *,
        uid: str | None = None,
        name: str | None = None,
        output_number_type: NumberType | str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a ChainMap.

        Args:
            maps: Sequence of maps. Must not be empty.
            uid: Optional explicit ID.
            seed: The seed of the minter of the id when *uid* is None.
            name: Human-readable name for this map. Defaults to the map's ID.
            output_number_type: The representation results are written in,
                among those the target unit admits; ``None`` for the target
                unit's default.

        Raises:
            ValueError: If maps is empty.
            ValueError: If units of adjacent maps are incompatible.
        """
        if not maps:
            raise ValueError("ChainMap requires at least one map")

        # Validate unit compatibility
        for i in range(len(maps) - 1):
            curr_map = maps[i]
            next_map = maps[i + 1]
            if (
                curr_map.target_unit is not None
                and next_map.source_unit is not None
                and curr_map.target_unit != next_map.source_unit
            ):
                raise ValueError(
                    f"Incompatible units at index {i}: "
                    f"Map {curr_map.id} outputs {curr_map.target_unit}, "
                    f"but Map {next_map.id} expects {next_map.source_unit}"
                )

        # Set first: whether the chain computes in doubles depends on them.
        self._maps = list(maps)
        super().__init__(
            source_unit=maps[0].source_unit,
            target_unit=maps[-1].target_unit,
            uid=uid,
            seed=seed,
            name=name,
            output_number_type=output_number_type,
        )

    @property
    def maps(self) -> list[ConversionMap[Any]]:
        """The sequence of maps."""
        return list(self._maps)

    @property
    def is_invertible(self) -> bool:
        """Whether the chain is invertible (all sub-maps must be invertible)."""
        return all(m.is_invertible for m in self._maps)

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Apply maps in sequence, each with its own guard and representation."""
        result = value
        for m in self._maps:
            result = m._evaluate_nested(result, **kwargs)
        return result

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Apply maps in sequence to array."""
        result = values
        for m in self._maps:
            result = m._convert_array(result, **kwargs)
        return result

    def inverse(self) -> Self:
        """Return the inverse chain.

        This reverses the order of maps and inverts each one.
        (f(g(x)))^-1 = g^-1(f^-1(x))
        """
        if not self.is_invertible:
            raise NotImplementedError("ChainMap is not invertible")

        inv_maps = [m.inverse() for m in reversed(self._maps)]
        return self.__class__(
            inv_maps,
            uid=f"inv_{self.id}",
            output_number_type=self._inverse_output_number_type(),
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``ChainMap`` with each member's own form."""
        return self._map_object(
            "ChainMap",
            {
                "maps": [
                    m._to_manifest(minimal=minimal, entries=entries) for m in self._maps
                ]
            },
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> ChainMap:
        """Read a ``ChainMap``, its members in the order they apply."""
        return cls(
            maps=_read_maps(data, "maps", measure_maps),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def _nested_maps(self) -> list[ConversionMap[Any]]:
        return self._maps

    def __repr__(self) -> str:
        return f"ChainMap(steps={len(self._maps)})"


# endregion

# region PiecewiseMap


class PiecewiseMap(ConversionMap[CoordinateValue]):
    """Map defined by different maps on different intervals.

    Also known as ConcatenationMap.
    Defined by a set of break points and the maps to use between them.

    Intervals are [start, end).

    Attributes:
        breaks: Sorted list of break points. N breaks define N+1 regions (or N regions).
                Here we strictly define regions between breaks.
                If we have breaks [0, 10, 20], we have regions [0, 10) and [10, 20).
                Anything outside is handled by 'extrapolate' or explicit outer maps.
        maps: Maps for each interval. len(maps) must be len(breaks) - 1.
    """

    _manifest_type: ClassVar[str | None] = "PiecewiseMap"

    def __init__(
        self,
        *,
        breaks: Sequence[CoordinateValue],
        maps: Sequence[ConversionMap[Any]],
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
        output_number_type: NumberType | str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a PiecewiseMap.

        Args:
            breaks: Sorted list of interval boundaries.
            maps: List of maps, one for each interval between breaks.
            source_unit: Input unit.
            target_unit: Output unit.
            uid: Explicit ID.
            seed: The seed of the minter of the id when *uid* is None.
            name: Human-readable name for this map. Defaults to the map's ID.
            output_number_type: The representation results are written in,
                among those the target unit admits; ``None`` for the target
                unit's default.

        Raises:
            ValueError: If lengths mismatch or breaks are not sorted.
        """
        if len(breaks) < 2:
            raise ValueError("PiecewiseMap requires at least 2 break points")
        if len(maps) != len(breaks) - 1:
            raise ValueError(
                f"Number of maps ({len(maps)}) must be len(breaks)-1 ({len(breaks)-1})"
            )

        # Verify sorting
        breaks_float = [float(b) for b in breaks]
        if not all(x < y for x, y in zip(breaks_float, breaks_float[1:])):
            raise ValueError("Breaks must be strictly increasing")

        # Set first: whether the map computes in doubles depends on them.
        self._maps = list(maps)
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            seed=seed,
            name=name,
            output_number_type=output_number_type,
        )

        self._breaks = np.array(breaks_float, dtype=np.float64)
        self._breaks_orig = list(breaks)

    @property
    def breaks(self) -> NDArray[np.floating[Any]]:
        """The interval boundaries."""
        return self._breaks

    @property
    def maps(self) -> list[ConversionMap[Any]]:
        """The maps for each interval."""
        return list(self._maps)

    def _get_map_index(
        self, value: CoordinateValue, allow_upper_bound: bool = False
    ) -> int:
        """Find the index of the interval ``[breaks[i], breaks[i+1])`` holding *value*.

        The comparison runs in the input's own arithmetic: an int or a
        Fraction is compared exactly with the breaks as given, a float with
        the breaks as floats. A break given as a Fraction no float
        represents therefore splits exact inputs where it lies.

        Args:
            value: The input.
            allow_upper_bound: Assign the last break to the last interval
                (used for inversion).

        Raises:
            ValueError: If *value* lies outside ``[breaks[0], breaks[-1])``.
        """
        if isinstance(value, (int, Fraction)):
            breaks: Sequence[CoordinateValue] | NDArray[np.float64] = self._breaks_orig
        else:
            breaks, value = self._breaks, float(value)
        if allow_upper_bound and value == breaks[-1]:
            return len(self._maps) - 1
        if value < breaks[0] or value >= breaks[-1]:
            raise ValueError(f"Value {value} out of bounds [{breaks[0]}, {breaks[-1]})")
        return bisect.bisect_right(breaks, value) - 1

    def _convert_scalar(
        self, value: CoordinateValue, allow_upper_bound: bool = False, **kwargs: Any
    ) -> CoordinateValue:
        """Convert scalar by the map of the interval holding it."""
        m = self._maps[self._get_map_index(value, allow_upper_bound)]
        return m._evaluate_nested(value, **kwargs)

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert array by processing chunks."""
        x = values.astype(np.float64)
        result = np.empty_like(x)

        # Optimize using searchsorted
        indices = np.searchsorted(self._breaks, x, side="right") - 1

        # Clip indices to valid range for map lookup, but mask out-of-bounds later
        map_indices = np.clip(indices, 0, len(self._maps) - 1)

        # Identify valid range
        valid_mask = (x >= self._breaks[0]) & (x < self._breaks[-1])
        if not np.all(valid_mask):
            # Simple policy: raise error for now, could support extrapolation
            raise ValueError("Values outside map bounds")

        # Iterate over unique maps involved
        unique_indices = np.unique(map_indices)

        for idx in unique_indices:
            mask = (map_indices == idx) & valid_mask
            if not np.any(mask):
                continue

            result[mask] = self._maps[idx]._convert_array(values[mask], **kwargs)

        return result

    def inverse(self) -> Self:
        """Return the inverse map.

        Requires all sub-maps to be invertible and the resulting target intervals to be contiguous.
        This is non-trivial if the target intervals don't line up.
        For now, we implement the naive inversion:
        Convert breaks using forward maps to get new breaks, invert maps.
        """
        # Convert breaks. Note: allow mapping the final break point.
        new_breaks = []
        for b in self._breaks_orig:
            new_breaks.append(self._convert_scalar(b, allow_upper_bound=True))

        # Note: Depending on map direction (increasing/decreasing), new_breaks might not be sorted.
        # We assume strictly increasing monotonic maps for simple inversion.

        # Check monotonicity
        if not all(x < y for x, y in zip(new_breaks, new_breaks[1:])):
            raise NotImplementedError("Cannot invert non-monotonic PiecewiseMap")

        inv_maps = [m.inverse() for m in self._maps]

        return self.__class__(
            breaks=new_breaks,
            maps=inv_maps,
            source_unit=self._target_unit,
            target_unit=self._source_unit,
            uid=f"inv_{self.id}",
            output_number_type=self._inverse_output_number_type(),
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``PiecewiseMap``: its breaks and each interval's map."""
        return _piecewise_form(
            self,
            self._breaks_orig,
            [m._to_manifest(minimal=minimal, entries=entries) for m in self._maps],
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> PiecewiseMap:
        """Read a ``PiecewiseMap``, one map per interval between its breaks.

        Raises:
            ValueError: If ``breaks`` or ``maps`` is not an array.
        """
        if not isinstance(data.get("breaks"), list):
            raise ValueError("a PiecewiseMap's 'breaks' must be an array")
        return cls(
            breaks=[number_from_json(b) for b in data["breaks"]],
            maps=_read_maps(data, "maps", measure_maps),
            source_unit=data.get("source_unit"),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def _nested_maps(self) -> list[ConversionMap[Any]]:
        return self._maps

    def __repr__(self) -> str:
        return f"PiecewiseMap(regions={len(self._maps)})"


# endregion
