"""Base classes and protocols for conversion maps.

This module defines the ConversionMap abstract base class that all
map implementations must inherit from.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, Union

import numpy as np
from numpy.typing import NDArray

from ..core.enums import NumberType, TimeUnit
from ..core.ids import mint_id, validate_id
from ..core.time import Coordinate, CoordinateValue, express_as

if TYPE_CHECKING:
    from typing_extensions import Self

module_logger = logging.getLogger(__name__)

# Type variable for the output type of a map
T = TypeVar("T")

# Input types that maps can accept
MapInput = Union[CoordinateValue, Coordinate, NDArray[np.floating[Any]]]

# region ConversionMap


class ConversionMap(ABC, Generic[T]):
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

    # Class-level attributes that subclasses may override
    _default_source_unit: TimeUnit | None = None
    _default_target_unit: TimeUnit | None = None

    # Set by maps whose result is an ordinal (a measure count, a floor index)
    # rather than a position on the target axis, so the axis's own
    # representation does not get imposed on a number that is not one of its
    # coordinates.
    _declared_output_number_type: NumberType | None = None

    # Registry of concrete map types, keyed by class name, populated
    # automatically as subclasses are defined. Used by from_dict() to
    # dispatch deserialization without a hand-maintained lookup table.
    _registry: ClassVar[dict[str, type["ConversionMap[Any]"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register every concrete subclass under its class name.

        Abstract subclasses (those still carrying unimplemented abstract
        methods) are not registered, since they cannot be instantiated by
        from_dict().
        """
        super().__init_subclass__(**kwargs)
        if not inspect.isabstract(cls):
            ConversionMap._registry[cls.__name__] = cls

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

        # Set name: explicit > "source_to_target" > id
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize the map to a dictionary.

        Subclasses should call super().to_dict() and add their parameters;
        rational parameters go through
        :func:`~timetoalign.core.number_to_json` so that the whole
        dictionary stays JSON-serializable.

        The map's ``name`` is always emitted, and every subclass
        ``from_dict`` passes it back to the constructor, so a custom name
        survives the round trip.

        Returns:
            Dictionary representation of the map.
        """
        return {
            "type": self.__class__.__name__,
            "id": self._id,
            "name": self._name,
            "source_unit": self._source_unit.name if self._source_unit else None,
            "target_unit": self._target_unit.name if self._target_unit else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionMap[Any]:
        """Deserialize a map from a dictionary.

        Dispatches to the concrete subclass named by ``data["type"]`` using
        the self-registering class registry populated by
        ``__init_subclass__``.

        Args:
            data: Dictionary representation.

        Returns:
            A ConversionMap instance.

        Raises:
            ValueError: If the type is missing or unknown.
            TypeError: If the registered class does not override from_dict,
                which would otherwise recurse indefinitely.
        """
        map_type = data.get("type")
        if map_type is None:
            raise ValueError("Map dictionary must include 'type'")

        map_cls = cls._registry.get(map_type)
        if map_cls is None:
            known = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unknown map type: {map_type!r}. Known types: {known}")

        if map_cls.from_dict.__func__ is ConversionMap.from_dict.__func__:
            raise TypeError(
                f"{map_cls.__name__} must override from_dict() to deserialize itself"
            )

        return map_cls.from_dict(data)

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
        """Two maps are equal if they have the same type and parameters."""
        if not isinstance(other, ConversionMap):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        """Hash based on the map's ID."""
        return hash(self._id)

    # endregion


# endregion
