"""Base classes and protocols for conversion maps.

This module defines the ConversionMap abstract base class that all
map implementations must inherit from.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    NamedTuple,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
import timetoshare
from numpy.typing import NDArray

from ..core.enums import NumberType, TimeUnit
from ..core.ids import _OpenObject, _plan_admissions, mint_id, quote_id, validate_id
from ..core.time import Coordinate, CoordinateValue, express_as

if TYPE_CHECKING:
    from typing_extensions import Self

module_logger = logging.getLogger(__name__)

# Type variable for the output type of a map
T = TypeVar("T")

# Input types that maps can accept
MapInput = Union[CoordinateValue, Coordinate, NDArray[np.floating[Any]]]

# A calibration pair: a position on the source axis and the value it reads.
CalibrationPair = tuple[
    Union[CoordinateValue, Coordinate], Union[CoordinateValue, Coordinate]
]


class _Calibration(NamedTuple):
    """Calibration pairs read for a law: what every law starts from."""

    points: list[tuple[int, CoordinateValue, CoordinateValue]]
    """``(index, position, value)`` sorted by position, positions as given."""
    value_unit: TimeUnit
    """The unit of the values, the map's target unit."""
    origin: CoordinateValue
    """Where the map's source axis starts, in the coordinates of the positions."""
    exact: bool
    """Whether every position and value is exact (an ``int`` or a ``Fraction``)."""


def _check_calibration(
    calibration: _Calibration,
    points: Sequence[tuple[int, CoordinateValue, CoordinateValue]],
    reach: Callable[[CoordinateValue], Fraction],
    law: str,
) -> None:
    """Refuse a calibration pair lying off the law beyond the derived bound.

    *reach* answers, exactly on the Fractions of the numbers as given, the
    position at which the law reads a value. The bound is measured where
    the rounding happens: on the positions, which a program computed in
    doubles and printed, as the caller gives them and before *origin* is
    subtracted, while the values are read as given. When every number of
    the calibration is exact, each position must be the one the law
    reaches. Otherwise it may miss that position by the error of a position
    computed in doubles as ``x0 + w·(v − v_first)/(v_last − v_first)``,
    the multiplication and the division in any order, which forward error
    analysis bounds as follows.

    Let ``t = (v − v_first)/(v_last − v_first)``, exact on the values as
    given, and ``u = 2**-53``. An operation rounding to the nearest double
    errs by at most half an ulp of its result, at most ``u`` of its
    magnitude, and not at all when its exact result is a double. The
    product ``w·t`` passes through the multiplication, the division and
    each label subtraction whose exact difference is no double: ``k``
    roundings, each a factor ``(1 + δ)``, ``|δ| ≤ u``, in any order, so it
    misses by at most ``γ_k·|t|·|w|`` with ``γ_k = k·u/(1 − k·u)``. Adding
    ``x0`` rounds once more, by half an ulp of the position itself. One
    position therefore misses ``x0 + w·t`` by at most
    ``E(p, t) = ½·ulp(p) + γ_k·|t|·|w|``.

    The line is drawn through the end pairs as given, which are computed
    positions too: the first at ``t = 0``, where the product is exactly 0,
    the last at ``t = 1``. Their errors move the line at ``t`` by at most
    ``|1 − t|·E(p_first, 0) + |t|·E(p_last, 1)``. ``|w|`` is known through
    the span as given, ``S = |p_last − p_first|``, up to those errors:
    ``|w| ≤ W = (S + ½·ulp(p_first) + ½·ulp(p_last)) / (1 − γ_k)``, with
    ``k`` counted for the last pair. The bound is
    ``E(p, t) + |1 − t|·E(p_first, 0) + |t|·E(p_last, 1)`` with ``W`` for
    ``|w|``, evaluated exactly. A position rounded once from its exact value
    misses it by ``½·ulp(p)`` alone, which the bound contains.

    Args:
        calibration: The calibration the law was built from.
        points: The pairs to check, those the law was not built through.
        reach: The exact position at which the law reads a value.
        law: The law as the message names it.

    Raises:
        ValueError: Naming the first pair beyond the bound.
    """
    _, first_position, first_value = calibration.points[0]
    _, last_position, last_value = calibration.points[-1]
    span = Fraction(last_value) - Fraction(first_value)

    def t_of(value: CoordinateValue) -> Fraction:
        """Return the fraction ``t`` of the calibrated span at which *value* lies."""
        return (Fraction(value) - Fraction(first_value)) / span

    def gamma(value: CoordinateValue) -> Fraction:
        """Return the relative rounding bound ``γ_k`` of the product ``w·t`` for *value*.

        ``k`` counts the roundings of the product ``w·t``.
        """
        roundings = (Fraction(value) - Fraction(first_value), span)
        k = 2 + sum(Fraction(float(d)) != d for d in roundings)
        return Fraction(k, 2**53 - k)

    def half_ulp(position: CoordinateValue) -> Fraction:
        """Return half the unit in the last place of *position* read as a float."""
        return Fraction(math.ulp(float(position))) / 2

    def error(position: CoordinateValue, value: CoordinateValue) -> Fraction:
        """Return the error bound ``E(p, t)`` of the pair (*position*, *value*).

        ``|w|`` is taken at its bound ``W`` (``width``).
        """
        return half_ulp(position) + gamma(value) * abs(t_of(value)) * width

    if not calibration.exact:
        width = (
            abs(Fraction(last_position) - Fraction(first_position))
            + half_ulp(first_position)
            + half_ulp(last_position)
        ) / (1 - gamma(last_value))
        ends = error(first_position, first_value), error(last_position, last_value)
    for index, position, value in points:
        on_law = reach(value)
        miss = abs(Fraction(position) - on_law)
        bound = Fraction(0)
        if not calibration.exact:
            t = t_of(value)
            bound = error(position, value) + abs(1 - t) * ends[0] + abs(t) * ends[1]
        if miss > bound:
            raise ValueError(
                f"Calibration pair {index} ({position}, {value}) lies off the "
                f"{law}, which reaches {value} at "
                f"{on_law if calibration.exact else float(on_law)}."
            )


class OutputNumberTypeError(ValueError):
    """A float handed to a map declared to write fractions.

    Such a map computes exactly, so it takes ints and Fractions only: a float
    result rebuilt as an exact value would state a ratio the input never
    stated. The input is refused as a value, like any other position a map
    cannot answer for.
    """


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

    _manifest_path: ClassVar[tuple[str, ...]] = ("timelines", "conversion_maps")

    # Class-level attributes that subclasses may override
    _default_source_unit: TimeUnit | None = None
    _default_target_unit: TimeUnit | None = None

    # Set by maps whose result is an ordinal (a measure count, a floor index)
    # rather than a position on the target axis, so the axis's own
    # representation does not get imposed on a number that is not one of its
    # coordinates.
    _declared_output_number_type: NumberType | None = None

    # The TimeToShare map type a class reads and writes as itself. Only the
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

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[CalibrationPair],
        *,
        source_unit: TimeUnit | str | None,
        value_unit: TimeUnit | str | None = None,
        origin: CoordinateValue | Coordinate = 0,
    ) -> Self:
        """Build the map a law reads from calibration pairs.

        Each pair holds a position and the value it reads on the target
        axis. Positions are given in coordinates where the map's source axis
        starts at *origin*: the map reads ``position - origin``, while the
        pairs are checked against the law on the positions as given, where
        their rounding happened. Values are coordinates carrying the one
        unit they share, or numbers in *value_unit*. A map class that is a
        law of calibration implements this; the pairs it receives are
        checked against the law, never fitted to it.

        Exactness is a property of the whole calibration. When every
        position and value is exact (an ``int`` or a ``Fraction``), the
        pairs must lie on the law exactly. Otherwise, exact and float
        numbers may mix, and a position may miss the one at which the law
        reads its value by the error a position computed in doubles may
        carry, a bound derived by forward error analysis in ulps of the
        positions and of the span.

        Args:
            pairs: ``(position, value)`` pairs, in any order.
            source_unit: The unit of the positions.
            value_unit: The unit of the values, the map's target unit. Needed
                when a value is a number; a coordinate value must be in it.
            origin: Where the source axis starts, in the coordinates of the
                positions: a child's offset on its parent.

        Returns:
            The map through the pairs.

        Raises:
            NotImplementedError: If the class is no law of calibration.
            TypeError: If a value is a number and no *value_unit* is given.
            ValueError: As the implementing class states.
        """
        raise NotImplementedError(f"{cls.__name__} is built from no calibration pairs")

    @classmethod
    def _calibration_points(
        cls,
        pairs: Sequence[CalibrationPair],
        source_unit: TimeUnit | str | None,
        value_unit: TimeUnit | str | None,
        origin: CoordinateValue | Coordinate,
    ) -> _Calibration:
        """Read *pairs* and *origin*, the positions sorted and kept as given.

        A coordinate is read as its number once its unit is checked against
        *source_unit* (positions and *origin*) or *value_unit*; without
        *value_unit*, the first coordinate value sets the unit every value
        must share. *index* is the pair's place in the given sequence, which
        every message names.

        Raises:
            TypeError: If a value is a number and no *value_unit* is given.
            ValueError: If there are fewer than two pairs, a coordinate is in
                another unit than its axis, or two pairs share a position.
        """
        if len(pairs) < 2:
            raise ValueError(
                f"A calibration needs at least two pairs, got {len(pairs)}."
            )
        units = {
            "position": None if source_unit is None else TimeUnit(source_unit),
            "value": None if value_unit is None else TimeUnit(value_unit),
        }
        if isinstance(origin, Coordinate):
            if units["position"] is not None and origin.unit != units["position"]:
                raise ValueError(
                    f"The calibration origin is in {origin.unit.value}, not in "
                    f"{units['position'].value}."
                )
            origin = origin.value
        points = []
        for index, pair in enumerate(pairs):
            numbers = []
            for role, number in zip(units, pair):
                if isinstance(number, Coordinate):
                    if role == "value" and units[role] is None:
                        units[role] = number.unit
                    if units[role] is not None and number.unit != units[role]:
                        raise ValueError(
                            f"Calibration pair {index} has a {role} in "
                            f"{number.unit.value}, not in {units[role].value}."
                        )
                    number = number.value
                elif role == "value" and value_unit is None:
                    raise TypeError(
                        f"Calibration pair {index} has a value of type "
                        f"{type(number).__name__}, not a Coordinate carrying its "
                        f"unit; give value_unit to read numbers."
                    )
                numbers.append(number)
            points.append((index, numbers[0], numbers[1]))
        points.sort(key=lambda point: point[1])
        for (first, position, _), (second, other, _) in zip(points, points[1:]):
            if position == other:
                raise ValueError(
                    f"Calibration pairs {first} and {second} share the "
                    f"position {position}."
                )
        exact = all(
            isinstance(number, (int, Fraction))
            for _, *numbers in points
            for number in numbers
        )
        return _Calibration(points, units["value"], origin, exact)

    def __init__(
        self,
        *,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
        output_number_type: NumberType | str | None = None,
    ) -> None:
        """Initialize a ConversionMap.

        Args:
            source_unit: The unit of input coordinates. Defaults to class default.
            target_unit: The unit of output coordinates. Defaults to class default.
            uid: Optional explicit ID, kept verbatim. If None, an id
                ``{prefix}_1`` is minted, which a timeline re-mints should it
                collide in the timeline's manifest, and a map nesting this
                one should it collide in that map's tree.
            name: Human-readable name for this map. Used as field header in
                timestamp tables. Defaults to "source_to_target" if units are
                provided, otherwise uses the map's ID.
            output_number_type: The representation the map's results are
                written in, among those the target unit admits. ``None``
                leaves it to the map class or the target unit's default (see
                :attr:`output_number_type`).

        Raises:
            ValueError: If the target unit does not admit
                *output_number_type*, or if it is ``fraction`` and the map
                computes in doubles (see :attr:`output_number_type`).
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

        # A map is immutable once built, its members and parameters set
        # before this runs, so what it states about its own arithmetic and
        # representation is read once here rather than on every value.
        self._float_arithmetic: bool = self._computes_in_doubles()

        # A setting of the library only: the TimeToShare manifest has no key
        # for it, so to_dict() leaves it out and from_dict() reads none.
        self._given_output_number_type: NumberType | None = None
        if output_number_type is not None:
            self._given_output_number_type = (
                NumberType(output_number_type)
                if self._target_unit is None
                else self._target_unit.resolve_number_type(output_number_type)
            )
            if (
                self._float_arithmetic
                and self._given_output_number_type is NumberType.fraction
            ):
                raise ValueError(
                    f"{type(self).__name__} computes in doubles and cannot be "
                    f"declared to write fractions."
                )
        self._own_output_number_type: NumberType | None = (
            self._stated_output_number_type()
        )

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

        The target decides, and it decides alone: the representation given
        at construction where there is one, else the map class's declared
        output representation, else ``float`` for a map that computes in
        doubles, else the target unit's default. What the value is converted
        *from* has no say -- a result landing on a float-canonical axis is a
        float whether the source axis was exact or not, and every reader of
        this map (row, column, typed getter, direct call) gets that same
        answer.

        **A map that computes in doubles answers doubles.** Its results are
        ``float`` wherever the target unit admits one, whatever the unit's
        default: an interpolation into ``quarters`` answers the float it
        computed, never the float rebuilt as a ``Fraction``. For the same
        reason such a map refuses the ``fraction`` declaration. A map
        computes in doubles when its own arithmetic does (an
        :class:`~timetoalign.maps.InterpolationMap`, a map with a float
        parameter such as a float scalar or a float anchor) or when a map
        nested in it answers floats: a chain or a piecewise map with one
        such member is one. On a discrete target, which admits ``int`` only, the result
        is rounded as on any other.

        A map whose answer is an ordinal rather than a position on the target
        axis -- a measure count, a floor index -- declares that through
        ``_declared_output_number_type``. A label or structured map names no
        target unit and so declares nothing.
        """
        own = self._own_output_number_type
        if own is not None:
            return own
        if self._target_unit is None:
            return None
        return self._target_unit.default_number_type

    def _stated_output_number_type(self) -> NumberType | None:
        """The representation the map states itself, before the target unit's default.

        What :attr:`output_number_type` answers from the map alone: the
        given representation, the class's declared one, or ``float`` for a
        map computing in doubles where its target unit admits floats.
        ``None`` leaves it to the target unit. A map nesting this one writes
        its results in this representation, so the default of the member's
        target unit, an intermediate axis, is not imposed. Read once, at
        construction, as ``_own_output_number_type``.
        """
        if self._given_output_number_type is not None:
            return self._given_output_number_type
        if self._declared_output_number_type is not None:
            return self._declared_output_number_type
        if (
            self._target_unit is not None
            and self._float_arithmetic
            and NumberType.float in self._target_unit.allowed_number_types
        ):
            return NumberType.float
        return None

    def _computes_in_doubles(self) -> bool:
        """Whether the map computes in doubles, whatever its input.

        A map does when one of its own parameters is a float (a float scalar
        or offset, a float anchor: :meth:`_parameters`), or when a map nested
        in it answers floats: it computes in doubles itself, or states
        ``float`` as its own representation. Read once, at construction, as
        ``_float_arithmetic``.
        """
        return any(
            isinstance(parameter, (float, np.floating))
            for parameter in self._parameters()
        ) or any(
            nested._float_arithmetic
            or nested._own_output_number_type is NumberType.float
            for nested in self._nested_maps()
        )

    def _parameters(self) -> Sequence[Any]:
        """The numbers the map's own arithmetic runs on.

        A subclass sets them, and the maps :meth:`_nested_maps` answers,
        before :meth:`__init__` of this class runs, which reads whether the
        map computes in doubles from them.
        """
        return ()

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
            OutputNumberTypeError: If the map was given the ``fraction``
                declaration and the scalar input is a float.
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

        A map given the ``fraction`` declaration computes exactly, so it
        takes ints and Fractions only: a float input would leave a float
        result rebuilt as an exact value, which is a ratio the input never
        stated.

        Raises:
            OutputNumberTypeError: If the map was given the ``fraction``
                declaration and *value* is a float.
        """
        if self._given_output_number_type is NumberType.fraction and isinstance(
            value, (float, np.floating)
        ):
            raise OutputNumberTypeError(
                f"Map {self._id} is declared to write fractions and computes "
                f"exactly, so it takes an int or a Fraction, not the float {value!r}."
            )
        return self._convert_scalar(value, **kwargs)

    def _evaluate_nested(self, value: CoordinateValue, **kwargs: Any) -> T:
        """The result a map nesting this one receives from it.

        The member's own guard applies (:meth:`_evaluate`), and its result
        is written in the representation the member states itself
        (``_own_output_number_type``), never in its target unit's default:
        that unit is an intermediate axis, or the nesting map's, which
        writes the final result once.

        Raises:
            OutputNumberTypeError: If the map was given the ``fraction``
                declaration and *value* is a float.
        """
        return self._written_as(
            self._evaluate(value, **kwargs), self._own_output_number_type
        )

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
        return self._written_as(result, self.output_number_type)

    @staticmethod
    def _written_as(result: T, number_type: NumberType | None) -> T:
        """*result* written in *number_type*; ``None`` or a non-number passes through."""
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

    def _inverse_output_number_type(self) -> NumberType | None:
        """The output representation an inverse of this map is given.

        The representation given to this map, where the inverse's target,
        this map's source unit, admits it; else none, so that the inverse
        writes its results as its target unit does by default.
        """
        given = self._given_output_number_type
        if given is None or self._source_unit is None:
            return given
        return given if given in self._source_unit.allowed_number_types else None

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

    def _nested_maps(self) -> Sequence[ConversionMap[Any]]:
        """The maps this map's TimeToShare form nests, in the order written."""
        return ()

    def _map_tree(self) -> Iterator[ConversionMap[Any]]:
        """Yield this map, then every map nested in it at any depth, as written."""
        yield self
        for nested in self._nested_maps():
            yield from nested._map_tree()

    def _measure_maps(self) -> Iterator[Any]:
        """Yield every measure map this map, or a map nested in it, reads."""
        for nested in self._nested_maps():
            yield from nested._measure_maps()

    def _identity_entries(self, owner: str = "") -> list[tuple[str, bool, str, str]]:
        """Return ``(id, minted, prefix, description)`` for this map and every
        map nested in it, in the order of :meth:`_map_tree`.

        Args:
            owner: Who holds this map, ending each description
                (``" of timeline 'clt_1'"``).
        """
        return [
            (
                cmap._id,
                cmap._id_minted,
                cmap._id_prefix,
                f"conversion map {quote_id(cmap._id)}"
                + (
                    ""
                    if cmap is self
                    else f" nested in conversion map {quote_id(self._id)}"
                )
                + owner,
            )
            for cmap in self._map_tree()
        ]

    def _unique_map_tree(self, owner: str = "") -> list[ConversionMap[Any]]:
        """Return :meth:`_map_tree` as a list, refusing a map held twice in it.

        Args:
            owner: Who holds this map, ending the description in the error.

        Raises:
            ValueError: If this tree nests one map in two places, since each
                place is written with an id of its own.
        """
        tree = list(self._map_tree())
        if len({id(cmap) for cmap in tree}) < len(tree):
            raise ValueError(
                f"conversion map {quote_id(self._id)}{owner} nests one map in two "
                "places: give each place a map of its own, since each is written "
                "with its own id"
            )
        return tree

    def _admit_ids(self) -> None:
        """Give this map and every map nested in it ids unique in its tree.

        A composite admits its tree when it is built, so that its own form
        repeats no id: a given id is kept, a minted id another map of the
        tree is given or keeps before it is minted again. A timeline admits
        a map into its manifest (``Timeline._admit_map``).

        Raises:
            ValueError: If this tree nests one map in two places, or two maps
                of it are given one id. Nothing is changed then.
        """
        tree = self._unique_map_tree()
        for index, new_id in _plan_admissions(self._identity_entries(), {}).items():
            tree[index]._id = new_id

    # endregion

    # region Serialization

    def to_dict(self, *, minimal: bool = True) -> dict[str, Any]:
        """Return the map's TimeToShare form: one of the format's map types.

        Every map is written as one of ten abstract types (``LinearMap``,
        ``TableMap``, ``ConstantMap``, ``PiecewiseMap``, ``ChainMap``,
        ``CombinationMap`` and the measure views ``MeasureRecordMap``,
        ``MeasureNumberMap``, ``FloatingMeasureMap``, ``MeasureBeatMap``),
        with that type's fields only; the map read back from it converts
        every coordinate as this map does. Numbers are written in the
        representation they have: an integer, a float, or an exact Fraction
        pair. ``name`` is the name given at construction, or None when the
        map displays a derived one. An output number type given at
        construction is not written: a map of the format is independent of
        any number type, so the map read back writes its results as its
        target unit does by default. Keys the library does not know, kept
        when the map was read, are written back.

        A measure view names its measure map by id. Written on its own, it
        names the measure map's own id; written with a timeline, the
        manifest's ``measure_maps`` entry it names (see
        :meth:`Timeline.to_dict <timetoalign.timelines.Timeline.to_dict>`).

        Args:
            minimal: Write the minimal form (the default): every key at its
                default value left out, in nested maps too. With False, the
                complete form: every key written, its default included.

        Returns:
            The map's TimeToShare form.

        Raises:
            ValueError: If the map has no TimeToShare form (a periodic map,
                a table interpolating to the nearest or the next anchor) or
                cannot be written losslessly; the message names the map and
                the reason.
        """
        return self._to_manifest(minimal=minimal, entries=STANDALONE_MEASURE_MAPS)

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the map's TimeToShare form, measure views naming ids of *entries*.

        Raises:
            ValueError: Always, for a map without a TimeToShare form.
        """
        raise self._no_form("its type has no TimeToShare form")

    def _no_form(self, reason: str) -> ValueError:
        """The error refusing to write this map, naming it and *reason*."""
        return ValueError(
            f"{type(self).__name__} {quote_id(self._id)} cannot be written to a "
            f"TimeToShare manifest: {reason}"
        )

    def _map_object(
        self,
        manifest_type: str,
        fields: Mapping[str, Any],
        *,
        minimal: bool,
    ) -> dict[str, Any]:
        """Write the fields every map shares, then *fields*, then unknown keys.

        Args:
            manifest_type: The TimeToShare type written.
            fields: The type's own fields.
            minimal: Leave out every key at its default.
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
        )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        measure_maps: Mapping[str, Any] | None = None,
    ) -> ConversionMap[Any]:
        """Read a map from its TimeToShare form.

        Dispatches on ``data["type"]``, one of the format's ten map types,
        and builds that type's class: a manifest states the abstract type
        only, so a ``LinearMap`` from ticks to quarters reads as a
        :class:`LinearMap`, never as the ``TicksToQuarters`` it may have
        been written from. A key the type writes is required unless it has
        a default, which TimeToShare supplies (:func:`timetoshare.complete`);
        every key it does not write is kept and written back.

        Args:
            data: The map's TimeToShare form.
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
        completed = cls._completed_at(data)
        try:
            cmap = map_cls._from_manifest(completed, measure_maps or {})
        except KeyError as error:
            raise ValueError(
                f"a {map_type} lacks the key(s) [{error.args[0]!r}]"
            ) from error
        written = cmap.to_dict(minimal=False)
        missing = [key for key in written if key not in completed]
        if missing:
            raise ValueError(f"a {map_type} lacks the key(s) {missing}")
        cmap._keep_unknown_keys(data, known=set(written))
        return cmap

    @classmethod
    def _completed_at(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Return *data*, a map's form, with every key it leaves out at its default.

        The complete form is the one :func:`timetoshare.complete` writes: the
        map is placed as the first entry of each array ``_manifest_path``
        names and taken back out, its nested maps completed with it.
        """
        holder: Any = data
        for key in reversed(cls._manifest_path):
            holder = {key: [holder]}
        result = timetoshare.complete(holder)
        for key in cls._manifest_path:
            result = result[key][0]
        return result

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
        """Return the measure map's own id."""
        return measure_map.id

    def id_of_rows(self, rows: list[dict[str, Any]]) -> str:
        """Return the first measure map id, since a map written alone has no siblings."""
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
