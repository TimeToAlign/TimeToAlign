"""Base classes and protocols for conversion maps.

This module defines the ConversionMap abstract base class that all
map implementations must inherit from.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from ..core.enums import NumberType, TimeUnit
from ..core.ids import _OpenObject, mint_id, validate_id
from ..core.time import Coordinate, CoordinateValue, express_as

if TYPE_CHECKING:
    from typing_extensions import Self

module_logger = logging.getLogger(__name__)

# Type variable for the output type of a map
T = TypeVar("T")

# Input types that maps can accept
MapInput = Union[CoordinateValue, Coordinate, NDArray[np.floating[Any]]]

# region ConversionMap


class ConversionMap(_OpenObject, ABC, Generic[T]):
    """Abstract base class for coordinate conversion maps.

    A ConversionMap transforms coordinates from one representation to another.
    Maps are callable and support both scalar and array inputs.

    The key methods are:
    - __call__(value): Convert a single value or array
    - inverse(): Get the inverse map (if invertible)

    Attributes:
        id: Unique identifier for this map instance.
        source_unit: The unit of input coordinates (optional).
        target_unit: The unit of output coordinates (optional).

    Examples:
        >>> # Create a map that doubles values
        >>> linear = LinearMap(scalar=2.0)
        >>> linear(5.0)
        10.0

        >>> # Maps are callable
        >>> linear(np.array([1.0, 2.0, 3.0]))
        array([2., 4., 6.])

        >>> # Get the inverse
        >>> inv = linear.inverse()
        >>> inv(10.0)
        5.0
    """

    # Prefix of a minted id (``cmap_1``); subclasses inherit their nearest
    # ancestor's.
    _id_prefix: ClassVar[str] = "cmap"

    # The one optional key every map's dictionary form shares.
    _dict_defaults: ClassVar[dict[str, Any]] = {"name": None}

    # Class-level attributes that subclasses may override
    _default_source_unit: TimeUnit | None = None
    _default_target_unit: TimeUnit | None = None

    # Set by maps whose result is an ordinal (a measure count, a floor index)
    # rather than a position on the target axis, so the axis's own
    # representation does not get imposed on a number that is not one of its
    # coordinates.
    _declared_output_number_type: NumberType | None = None

    # The TimeSkeleton map type a class reads and writes as itself. Only the
    # classes of the format's ten abstract types declare one; every other
    # class writes one of them and is never read back as itself.
    _manifest_type: ClassVar[str | None] = None

    # The classes reading each manifest type, filled as they are defined.
    _registry: ClassVar[dict[str, type["ConversionMap[Any]"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a class that declares its own manifest type."""
        super().__init_subclass__(**kwargs)
        if "_manifest_type" in cls.__dict__ and cls._manifest_type is not None:
            ConversionMap._registry[cls._manifest_type] = cls

    def __init__(
        self,
        *,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ConversionMap.

        Args:
            source_unit: The unit of input coordinates. Defaults to class default.
            target_unit: The unit of output coordinates. Defaults to class default.
            uid: Optional explicit ID, kept verbatim. If None, an id
                ``{prefix}_1`` is minted, which a timeline re-mints should it
                collide in the timeline's manifest.
            name: Human-readable name for this map. Used as field header in
                timestamp tables. Defaults to "source_to_target" if units are
                provided, otherwise uses the map's ID.
        """
        self._id_minted = uid is None
        self._id = mint_id(self._id_prefix, ()) if uid is None else validate_id(uid)

        # Set units
        if source_unit is not None:
            self._source_unit = TimeUnit(source_unit)
        else:
            self._source_unit = self._default_source_unit

        if target_unit is not None:
            self._target_unit = TimeUnit(target_unit)
        else:
            self._target_unit = self._default_target_unit

        # Set name: explicit > "source_to_target" > id. Only a given name is
        # written by to_dict(); the others are derived for display.
        self._name_given = name is not None
        if name is not None:
            self._name = name
        elif self._source_unit and self._target_unit:
            self._name = f"{self._source_unit.value}_to_{self._target_unit.value}"
        else:
            self._name = self._id

        self._logger = module_logger.getChild(self.__class__.__name__)

    @property
    def id(self) -> str:
        """Unique identifier for this map."""
        return self._id

    @property
    def source_unit(self) -> TimeUnit | None:
        """The unit of input coordinates, if specified."""
        return self._source_unit

    @property
    def target_unit(self) -> TimeUnit | None:
        """The unit of output coordinates, if specified."""
        return self._target_unit

    @property
    def name(self) -> str:
        """Human-readable name for this map.

        Used as field header in timestamp tables. Defaults to
        "source_to_target" if units are provided, otherwise the map's ID.
        """
        return self._name

    @property
    def is_invertible(self) -> bool:
        """Whether this map has a well-defined inverse.

        Subclasses should override if they are not invertible.
        """
        return True

    @property
    def output_number_type(self) -> NumberType | None:
        """The representation this map's results are written in, if any.

        The target decides, and it decides alone: this map's own declared
        output representation where it has one, else the target unit's
        default. What the value is converted *from* has no say -- a result
        landing on a float-canonical axis is a float whether the source axis
        was exact or not, and every reader of this map (row, column, typed
        getter, direct call) gets that same answer.

        A map whose answer is an ordinal rather than a position on the target
        axis -- a measure count, a floor index -- declares that through
        ``_declared_output_number_type``. A label or structured map names no
        target unit and so declares nothing.
        """
        if self._declared_output_number_type is not None:
            return self._declared_output_number_type
        if self._target_unit is None:
            return None
        return self._target_unit.default_number_type

    # region Conversion methods

    def __call__(
        self,
        value: MapInput,
        **kwargs: Any,
    ) -> T | NDArray[Any]:
        """Convert a value or array of values.

        This is the primary interface for using a map.

        Args:
            value: A scalar, Coordinate, or numpy array.
            **kwargs: Additional arguments passed to _convert_scalar or _convert_array.

        Returns:
            The converted value(s), matching input shape.

        Raises:
            ValueError: If the input Coordinate has an incompatible unit.
        """
        # Handle Coordinate input
        if isinstance(value, Coordinate):
            if self._source_unit is not None and value.unit != self._source_unit:
                raise ValueError(
                    f"Coordinate unit {value.unit} does not match "
                    f"map source unit {self._source_unit}"
                )
            return self._on_target_axis(self._evaluate(value.value, **kwargs))

        # Handle array input
        if isinstance(value, np.ndarray):
            return self._convert_array(value, **kwargs)

        # Handle scalar input
        return self._on_target_axis(self._evaluate(value, **kwargs))

    def _evaluate(self, value: CoordinateValue, **kwargs: Any) -> T:
        """The map's own result, before an axis decides how it is written.

        Callers that re-express at a boundary of their own -- a stamp
        resolving a unit, a display assembling a conversion row -- take this
        rather than calling the map, so the value is written once instead of
        being degraded on the way out and rebuilt on arrival. Everyone else
        calls the map and gets its target axis's representation.
        """
        return self._convert_scalar(value, **kwargs)

    def _on_target_axis(self, result: T) -> T:
        """Write a public conversion result the way its target axis writes numbers.

        A map's arithmetic runs in whatever representation its own scalars
        happen to have, which says nothing about the axis the result lands
        on: ``ticks``-valued output is an integer count however the ratio was
        computed. Re-expressing here means a caller reading a map directly
        gets the same number a timestamp getter would report for it.

        Label and structured maps declare no target unit and pass through
        untouched, as does the array lane, which stays float64 by contract.
        """
        number_type = self.output_number_type
        if number_type is None or not isinstance(result, (int, float, Fraction)):
            return result
        return express_as(result, number_type)  # type: ignore[return-value]

    @abstractmethod
    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> T:
        """Convert a single scalar value.

        Args:
            value: The numeric value to convert.
            **kwargs: Subclass-specific arguments.

        Returns:
            The converted value.
        """
        ...

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert an array of values.

        Default implementation uses vectorized scalar conversion.
        Subclasses may override for efficiency.

        Args:
            values: Array of values to convert.
            **kwargs: Subclass-specific arguments.

        Returns:
            Array of converted values.
        """
        # Default: vectorize the scalar conversion
        vec_convert = np.vectorize(lambda x: self._convert_scalar(x, **kwargs))
        return vec_convert(values)

    def convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert an array of values.

        This is the public API for array conversion, used by the timestamp
        system and other batch operations. It delegates to _convert_array().

        All ConversionMap subclasses support efficient array operations through
        this method. Linear maps use NumPy broadcasting, TableMaps use np.interp,
        and composite maps delegate to their sub-maps.

        Args:
            values: NumPy array of values to convert.
            **kwargs: Subclass-specific arguments.

        Returns:
            NumPy array of converted values with same shape as input.

        Examples:
            >>> linear = LinearMap(scalar=2.0, offset=1.0)
            >>> linear.convert_array(np.array([0.0, 1.0, 2.0]))
            array([1., 3., 5.])

            >>> tempo_map = TableMap.from_tempo_changes([0, 960], [120, 60], 480)
            >>> tempo_map.convert_array(np.array([0, 480, 960]))
            array([0.  , 0.5 , 1.5])
        """
        return self._convert_array(values, **kwargs)

    # endregion

    # region Inverse

    @abstractmethod
    def inverse(self) -> Self:
        """Return the inverse of this map.

        Returns:
            A new ConversionMap that reverses this transformation.

        Raises:
            NotImplementedError: If the map is not invertible.
        """
        ...

    def matches_selector(self, key: str) -> bool:
        """Return whether a user conversion-map selector addresses this map.

        Subclasses that are addressed by other identifiers (for example,
        InterpolationMap by its source timeline id) should override this
        and extend the match.

        Args:
            key: A selector string from a conversion-map specification.

        Returns:
            True if key equals this map's id or name.
        """
        return key == self._id or key == self._name

    # endregion

    # region Composition

    def then(self, other: ConversionMap[Any]) -> ConversionMap[Any]:
        """Compose this map with another (this first, then other).

        Args:
            other: The map to apply after this one.

        Returns:
            A ChainMap that applies both maps in sequence.
        """
        # Import here to avoid circular imports
        from .composite import ChainMap

        return ChainMap([self, other])

    def __rshift__(self, other: ConversionMap[Any]) -> ConversionMap[Any]:
        """Compose maps using >> operator (self >> other)."""
        return self.then(other)

    # endregion

    # region Serialization

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return the map's TimeSkeleton form: one of the format's map types.

        Every map is written as one of ten abstract types (``LinearMap``,
        ``TableMap``, ``ConstantMap``, ``PiecewiseMap``, ``ChainMap``,
        ``CombinationMap`` and the measure views ``MeasureRecordMap``,
        ``MeasureNumberMap``, ``FloatingMeasureMap``, ``MeasureBeatMap``),
        with that type's fields only; the map read back from it converts
        every coordinate as this map does. Numbers are written in the
        representation they have: an integer, a float, or an exact Fraction
        pair. ``name`` is the name given at construction, or None when the
        map displays a derived one. Keys the library does not know, kept
        when the map was read, are written back.

        A measure view names its measure map by id. Written on its own, it
        names the measure map's own id; written with a timeline, the
        manifest's ``measure_maps`` entry it names (see
        :meth:`Timeline.to_dict <timetoalign.timelines.Timeline.to_dict>`).

        Args:
            minimal: Leave out every key at its default, in nested maps too.

        Returns:
            The map's TimeSkeleton form.

        Raises:
            ValueError: If the map has no TimeSkeleton form (a periodic map,
                a table interpolating to the nearest or the next anchor) or
                cannot be written losslessly; the message names the map and
                the reason.
        """
        return self._to_manifest(minimal=minimal, entries=STANDALONE_MEASURE_MAPS)

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the map's TimeSkeleton form, measure views naming ids of *entries*.

        Raises:
            ValueError: Always, for a map without a TimeSkeleton form.
        """
        raise self._no_form("its type has no TimeSkeleton form")

    def _no_form(self, reason: str) -> ValueError:
        """The error refusing to write this map, naming it and *reason*."""
        return ValueError(
            f"{type(self).__name__} {self._id!r} cannot be written to a "
            f"TimeSkeleton manifest: {reason}"
        )

    def _map_object(
        self,
        manifest_type: str,
        fields: Mapping[str, Any],
        *,
        minimal: bool,
        defaults: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Write the fields every map shares, then *fields*, then unknown keys.

        Args:
            manifest_type: The TimeSkeleton type written.
            fields: The type's own fields.
            minimal: Leave out every key at its default.
            defaults: The type's own optional fields and their defaults.
        """
        return self._dict_form(
            {
                "type": manifest_type,
                "id": self._id,
                "name": self._name if self._name_given else None,
                "source_unit": self._source_unit.name if self._source_unit else None,
                "target_unit": self._target_unit.name if self._target_unit else None,
                **fields,
            },
            minimal=minimal,
            defaults={**ConversionMap._dict_defaults, **(defaults or {})},
        )

    def _measure_maps(self) -> Iterator[Any]:
        """Yield every measure map this map, or a map nested in it, reads."""
        return iter(())

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        measure_maps: Mapping[str, Any] | None = None,
    ) -> ConversionMap[Any]:
        """Read a map from its TimeSkeleton form.

        Dispatches on ``data["type"]``, one of the format's ten map types,
        and builds that type's class: a manifest states the abstract type
        only, so a ``LinearMap`` from ticks to quarters reads as a
        :class:`LinearMap`, never as the ``TicksToQuarters`` it may have
        been written from. A key the type writes is required unless it has
        a default; every key it does not write is kept and written back.

        Args:
            data: The map's TimeSkeleton form.
            measure_maps: The manifest's measure maps by id, which the
                measure views name.

        Returns:
            The map read.

        Raises:
            ValueError: If the type is missing or is none of the ten, a
                required key is missing, a value is malformed, or a measure
                view names a measure map *measure_maps* does not hold.
        """
        if not isinstance(data, dict):
            raise ValueError(
                f"a conversion map must be an object, got {type(data).__name__}"
            )
        map_type = data.get("type")
        if map_type is None:
            raise ValueError("Map dictionary must include 'type'")
        map_cls = cls._registry.get(map_type)
        if map_cls is None:
            known = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown map type: {map_type!r}. Known types: {known}")
        try:
            cmap = map_cls._from_manifest(data, measure_maps or {})
        except KeyError as error:
            raise ValueError(
                f"a {map_type} lacks the key(s) [{error.args[0]!r}]"
            ) from error
        written = cmap.to_dict()
        optional = {**ConversionMap._dict_defaults, **map_cls._dict_defaults}
        missing = [key for key in written if key not in optional and key not in data]
        if missing:
            raise ValueError(f"a {map_type} lacks the key(s) {missing}")
        cmap._keep_unknown_keys(data, known=set(written) | set(optional))
        return cmap

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> ConversionMap[Any]:
        """Build this manifest type's map from its form."""
        raise NotImplementedError(f"{cls.__name__} is read as no manifest type")

    # endregion

    # region Display

    def __repr__(self) -> str:
        parts = [f"id={self._id!r}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        return f"{self.__class__.__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        if self._source_unit and self._target_unit:
            return (
                f"{self.__class__.__name__}({self._source_unit} -> {self._target_unit})"
            )
        return f"{self.__class__.__name__}({self._id})"

    # endregion

    # region Equality

    def __eq__(self, other: object) -> bool:
        """Two maps are equal if they are of one class and hold equal state.

        The id counts, whether it was minted does not; a cached inverse and
        the logger take no part. A map and the abstract type it writes
        convert alike but are not equal: they are of two classes.
        """
        if not isinstance(other, ConversionMap):
            return NotImplemented
        if type(self) is not type(other):
            return False
        mine, theirs = _state(self), _state(other)
        return mine.keys() == theirs.keys() and all(
            _same_value(mine[key], theirs[key]) for key in mine
        )

    def __hash__(self) -> int:
        """Hash based on the map's ID."""
        return hash(self._id)

    # endregion


# endregion


# region Measure-map ids


MEASURE_MAP_ID_PREFIX = "measuremap"
"""The prefix of a minted measure-map id (``measuremap_1``)."""


class MeasureMapIds(Protocol):
    """Where a map being written finds the ids of the measure maps it names.

    A timeline writing its manifest supplies the manifest's measure maps:
    ``own`` is the timeline's measure map, ``id_of`` the id under which a
    measure map is written in the manifest's ``measure_maps``, and
    ``id_of_rows`` that of the measure map compressed Measure Map rows
    describe, added to them.
    """

    own: Any

    def id_of(self, measure_map: Any) -> str:
        """Return the id under which *measure_map* is written."""
        ...

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        """Return the id of the measure map *rows* describe."""
        ...


class _StandaloneMeasureMaps:
    """The ids a map written on its own names: each measure map's own id."""

    own: Any = None

    def id_of(self, measure_map: Any) -> str:
        return measure_map.id

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        return mint_id(MEASURE_MAP_ID_PREFIX, ())


STANDALONE_MEASURE_MAPS: MeasureMapIds = _StandaloneMeasureMaps()
"""The ids of measure maps written outside a manifest."""


# endregion


# region Equality


_STATELESS = frozenset({"_logger", "_inverse", "_id_minted"})
"""Attributes that take no part in a map's equality."""


def _state(cmap: ConversionMap[Any]) -> dict[str, Any]:
    """The attributes of *cmap* its equality compares, unknown keys included."""
    state = {key: value for key, value in vars(cmap).items() if key not in _STATELESS}
    state["_unknown_keys"] = dict(cmap._unknown_keys)
    return state


def _same_value(left: Any, right: Any) -> bool:
    """Whether two attribute values are equal, arrays compared element-wise."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.shape == right.shape
            and bool(np.array_equal(left, right))
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_same_value(a, b) for a, b in zip(left, right))
        )
    return bool(left == right)


# endregion
