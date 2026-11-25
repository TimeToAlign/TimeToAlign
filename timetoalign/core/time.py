"""Time scalars (``TimeScalar`` / ``Coordinate`` / ``Duration``) and paired fields.

This module is the single home for the four scalar pydantic models that
represent positions and elapsed extents on a timeline, together with
their paired ``SemanticField`` field wrappers.

Class hierarchy (diamond MRO for the Id-variants)::

    TimeScalar(BaseModel)                          # abstract base
    ├── Coordinate(TimeScalar)
    ├── Duration(TimeScalar)
    └── IdTimeScalar(TimeScalar)                   # adds `timeline_id`

    IdCoordinate(Coordinate, IdTimeScalar)         # diamond
    IdDuration(Duration, IdTimeScalar)             # diamond

* ``Coordinate`` denotes a position on a timeline (origin is meaningful).
* ``Duration`` denotes an elapsed extent (signed; may be negative when
  produced by ``Coordinate - Coordinate`` with the right operand later
  than the left).
* ``IdCoordinate`` / ``IdDuration`` carry the additional ``timeline_id``
  identifying the timeline they refer to.

Arithmetic semantics (all operators raise ``TypeError`` on incompatible
types and on cross-timeline-id mixing):

* ``Coordinate + Coordinate`` → ``TypeError`` (subtract to get a Duration)
* ``Coordinate + Duration``   → ``Coordinate``
* ``Coordinate - Coordinate`` → ``Duration`` (the canonical way to get one)
* ``Coordinate - Duration``   → ``Coordinate``
* ``Duration  ± Duration``    → ``Duration``
* ``Duration  ± Coordinate``  → ``TypeError``
* ``{Coord,Dur} * / // <number>`` → same kind back
* ``{Coord,Dur} * / // {Coord,Dur}`` → ``TypeError``

PyArrow storage shape (identical for Coordinate, Duration, IdCoordinate,
IdDuration): the scalar's ``value`` is denormalised into a struct
``{value: float64, numerator: int64, denominator: int64}`` so that
rational precision survives Parquet round-trips.  ``unit`` and
``timeline_id`` live in ``pa.Field.metadata`` (the ``TIMETOALIGN_METADATA_KEY``
JSON blob), NOT in the struct.

Paired-field hierarchy::

    SemanticField (Generic[T])
    └── TimeScalarField[TimeScalarT]  (abstract; _raw_cls = DenominateNumberField;
                          consolidates the from_field / from_table /
                          metadata-resolve / repr plumbing)
        ├── CoordinateField   (TimeScalarField[Coordinate])
        │   └── IdCoordinateField (scalar_cls = IdCoordinate, carries timeline_id)
        └── DurationField     (TimeScalarField[Duration])
            └── IdDurationField   (scalar_cls = IdDuration, carries timeline_id)

The inner raw field is a ``DenominateNumberField`` (a ``RedundantNumberField``
with a single bound ``unit``) carrying the denormalised
``{value, numerator, denominator}`` struct.  ``unit`` lives in the
field's ``TIMETOALIGN_METADATA_KEY`` metadata blob; ``timeline_id`` lives there
too for the Id-variants.
"""

from __future__ import annotations

import json
import math
import operator
import re
from abc import abstractmethod
from collections.abc import Callable, Iterable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NamedTuple,
    Protocol,
    Self,
    TypeGuard,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .enums import Domain, NumberType, TimeUnit
from .fields import (
    TIMETOALIGN_METADATA_KEY,
    DataField,
    ScalarVocabulary,
    SemanticField,
    StructField,
    _compared,
    _ComparisonKernel,
    _filled_mask,
    _is_arrow_column,
    _is_arrow_field,
    _is_dict,
    _is_list,
    _is_ndarray,
    _is_series,
    _is_tuple,
    _member_fields,
    _struct_array,
    _struct_array_from,
    _struct_array_of,
    _struct_child,
    _struct_members,
    _struct_types_match,
    _valid_struct_cell,
    build_struct_array,
    data_shaped,
    drop_from_schema,
    metadata_blob_from_dict,
    parse_metadata_blob,
    register_value_projector,
)

if TYPE_CHECKING:
    from typing_extensions import TypeIs

# ---------------------------------------------------------------------------
# Type aliases & module constants
# ---------------------------------------------------------------------------

# Canonical numeric value type for all four scalars.  Re-exported as
# CoordinateValue / DurationValue for back-compat with maps/ and loader/.
TimeScalarValue = Union[int, float, Fraction]
CoordinateValue = TimeScalarValue
DurationValue = TimeScalarValue

TimeScalarT = TypeVar("TimeScalarT", bound="TimeScalar")


# ═══════════════════════════════════════════════════════════════════════════
# 1. NUMBER STORAGE — one struct, one canonical policy, one builder
# ═══════════════════════════════════════════════════════════════════════════

RATIONAL_STRUCT_TYPE: pa.StructType = pa.struct(
    [
        pa.field("value", pa.float64(), nullable=True),
        pa.field("numerator", pa.int64(), nullable=True),
        pa.field("denominator", pa.int64(), nullable=True),
    ]
)
"""Canonical storage shape for every number the library stores.

Every coordinate-, duration- and rational-valued column in the library uses
this struct — there is no second ``{num, den}`` shape.

The struct is **redundant on purpose**: it carries the same number twice, as
a float64 and as an integer ratio.  On a non-null row **both sides are always
populated**; a null sub-field occurs only where the whole row is null.  Which
of the two sides is authoritative is not encoded in the row — it is declared
once per field, in the field's ``number_type`` metadata (see
:class:`DenominateNumberField`).  Readers therefore never have to guess from
the data which representation to trust, and a float column and an exact
column are told apart by their schema rather than by sampling their rows.
"""

_INT64_MIN: int = -(2**63)
_INT64_MAX: int = 2**63 - 1

# The mirror of a float is its exact dyadic ratio whenever that ratio's two
# components fit int64.  Doubles smaller than about 2**-10 need denominators
# past 2**62, which do not fit; for those the mirror falls back to the nearest
# ratio over this denominator.  See _pair_from_float.
_MIRROR_DENOMINATOR_EXPONENT: int = 62

_FRACTION_RE = re.compile(r"^\s*(-?\d+)\s*/\s*(-?\d+)\s*$")
"""Recogniser for ``"<numerator>/<denominator>"`` strings."""

ROUNDING_MODES: tuple[str, ...] = ("round", "floor", "ceil", "truncate")
"""Ways an inexact value may be made integral.

``"round"`` goes to the nearest integer and settles ties on the even one —
Python's own :func:`round`, and PyArrow's ``half_to_even`` — so that a column
of ``.5`` values does not drift upwards.  ``"floor"`` and ``"ceil"`` go to
−∞ and +∞; ``"truncate"`` goes towards zero.
"""


# Type guards for values of unchecked type: each is one ``isinstance`` on the
# container and claims nothing about its elements.


def _is_chunked(value: object) -> TypeGuard[pa.ChunkedArray[Any]]:
    return isinstance(value, pa.ChunkedArray)


def _is_array(value: object) -> TypeGuard[pa.Array[Any]]:
    return isinstance(value, pa.Array)


def _is_numpy_integer(value: object) -> TypeGuard[np.integer[Any]]:
    return isinstance(value, np.integer)


def _is_numpy_floating(value: object) -> TypeGuard[np.floating[Any]]:
    return isinstance(value, np.floating)


def _fits_int64(n: int) -> bool:
    return _INT64_MIN <= n <= _INT64_MAX


def _require_int64(n: int, label: str, source: Any) -> int:
    """Guard the canonical side against a ratio int64 cannot hold.

    A canonical value is exact or it is an error — best effort belongs to
    mirrors only. A double small enough to need a denominator past ``2**62``
    has no exact ratio this schema can store, so an exact field refuses it
    rather than recording a degraded one, and says where to put it instead.
    """
    if not _fits_int64(n):
        raise ValueError(
            f"exact {label} {n} of {source!r} does not fit in int64, so it "
            f"cannot be stored as an exact value without loss. Use "
            f"number_type float for this field: the double is then kept "
            f"exactly, and only its ratio mirror is approximate."
        )
    return n


def _round_to_int(value: float | Fraction, rounding: str) -> int:
    """Make an inexact value integral under the named rounding mode."""
    if rounding == "round":
        return round(value)
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    if rounding == "truncate":
        return math.trunc(value)
    raise ValueError(
        f"Unknown rounding mode: {rounding!r}. Use one of {', '.join(ROUNDING_MODES)}."
    )


def _pair_from_float(value: float) -> tuple[int, int]:
    """Return the integer ratio mirroring *value*.

    Every finite double is exactly one rational whose denominator is a power
    of two, and that ratio is what this returns whenever both of its
    components fit int64 — the mirror of ``0.1`` is
    ``3602879701896397 / 36028797018963968``, ugly but exact.

    Doubles below roughly ``2**-10`` need a denominator past ``2**62``, which
    int64 cannot hold.  For those the mirror is the nearest ratio over
    ``2**62`` instead.  Only the mirror is affected: the canonical float side
    keeps the double untouched, so nothing a caller reads back as a float is
    ever degraded.  Nothing here approximates a float as a *small* ratio —
    the ratio stays dyadic, and no denominator is ever invented to make a
    number look tidier than it is.
    """
    if not math.isfinite(value):
        raise ValueError(f"cannot store non-finite value {value!r} as a number cell")
    exact = Fraction(value)
    if _fits_int64(exact.numerator) and _fits_int64(exact.denominator):
        return exact.numerator, exact.denominator
    if abs(value) >= 1.0:
        raise ValueError(f"value {value!r} is too large to mirror as an int64 ratio")
    scale = 1 << _MIRROR_DENOMINATOR_EXPONENT
    mirrored = Fraction(round(value * scale), scale)
    return mirrored.numerator, mirrored.denominator


def _pair_from_exact(value: int | Fraction) -> tuple[int, int]:
    """Return the integer ratio of an already-exact value."""
    exact = Fraction(value)
    return (
        _require_int64(exact.numerator, "numerator", value),
        _require_int64(exact.denominator, "denominator", value),
    )


def parse_number(value: object) -> int | float | Fraction:
    """Read one source value as the Python number it represents.

    Accepts ``int``, ``float``, ``Fraction``, and strings — either
    ``"<numerator>/<denominator>"`` or a plain numeric literal.  The result
    keeps the *kind* of number the source expressed: an integer literal comes
    back an ``int``, ``"1/3"`` comes back an exact ``Fraction``, and a decimal
    literal comes back a ``float``.  Nothing is promoted or approximated here;
    choosing a representation is :func:`number_cell`'s job.

    Raises:
        TypeError: If *value* is not a supported type.
        ValueError: If a string cannot be read as a number.
    """
    if isinstance(value, bool):
        raise TypeError("Boolean values are not valid number cells")
    if isinstance(value, (int, float, Fraction)):
        return value
    if _is_numpy_integer(value):
        return int(value)
    if _is_numpy_floating(value):
        return float(value)
    if isinstance(value, str):
        match = _FRACTION_RE.match(value)
        if match is not None:
            numerator, denominator = int(match.group(1)), int(match.group(2))
            if denominator == 0:
                raise ValueError(f"zero denominator in {value!r}")
            return Fraction(numerator, denominator)
        text = value.strip()
        try:
            return int(text)
        except ValueError:
            return float(text)
    raise TypeError(f"cannot read {type(value).__name__} as a number")


def to_canonical(
    value: Any,
    number_type: NumberType,
    *,
    rounding: str = "round",
) -> int | float | Fraction:
    """Coerce one value into a field's declared representation.

    This is the single place where a source number becomes the thing a field
    stores, and the rules are the ones a musician would expect:

    * into an ``int`` field, an integer stays itself, and an inexact float is
      made integral under *rounding*;
    * into a ``fraction`` field, an integer or ratio stays exact, and a float
      becomes its exact dyadic ratio;
    * into a ``float`` field, everything becomes the nearest double.

    An **exact** non-integral value entering an ``int`` field raises. A
    rounding mode licenses discarding the part of a *measurement* that the
    unit cannot express; it does not license discarding a third of a beat that
    someone deliberately wrote down. That is a mistake upstream, and saying so
    is more useful than quietly recording zero.

    A non-finite float (an infinity, NaN) has no integer and no ratio, so
    only a ``float`` field can hold it.

    Raises:
        ValueError: On an exact non-integral value into an ``int`` field, a
            non-finite value into an ``int`` or ``fraction`` field, or an
            unknown rounding mode.
    """
    number = parse_number(value)
    if (
        number_type is not NumberType.float
        and isinstance(number, float)
        and not math.isfinite(number)
    ):
        raise ValueError(
            f"the non-finite value {number} cannot be stored as "
            f"{number_type.name}: only a float holds it"
        )
    if number_type is NumberType.int:
        if isinstance(number, int):
            return number
        if isinstance(number, Fraction):
            if number.denominator != 1:
                raise ValueError(
                    f"exact value {number} cannot be stored in an integer-valued "
                    "field: rounding modes apply to inexact measurements, not to "
                    "a ratio that was written down deliberately"
                )
            return int(number)
        return _round_to_int(number, rounding)
    if number_type is NumberType.fraction:
        return Fraction(number)
    if number_type is NumberType.float:
        return float(number)
    raise ValueError(f"Unknown NumberType: {number_type!r}")


def quantize_to_unit(
    value: Any,
    unit: TimeUnit,
    *,
    rounding: str = "round",
) -> int | float | Fraction:
    """Re-express a value in the representation a bare unit declares.

    :func:`express_as` for callers holding a unit rather than a timeline —
    a unit's declared representation is its default. Writing
    ``Coordinate(Fraction(1, 2), TimeUnit.ticks)`` by hand is still refused,
    because nobody means half a tick; but *converting* a quarter position
    into ticks at a given resolution quantizes by definition, and this is
    where that happens, visibly and in one place.
    """
    return express_as(parse_number(value), unit.default_number_type, rounding=rounding)


def number_cell(
    value: Any,
    number_type: NumberType | None = None,
    *,
    rounding: str = "round",
    preserve_source_float: bool = False,
) -> dict[str, Any]:
    """Encode one number as a storage cell, both sides populated.

    The canonical side is whichever *number_type* names; the other side is
    mirrored from it, so the two never disagree about the value they carry.

    Args:
        value: The number to store — ``int``, ``float``, ``Fraction``, a
            numeric or ``"n/d"`` string, or an already-shaped cell dict.
        number_type: The field's declared representation. Defaults to the one
            the Python value already expresses, which is what a caller
            without a field in hand means.
        rounding: How to make an inexact value integral for an ``int`` field.
        preserve_source_float: For an ``int`` field fed a float, keep the
            incoming float on the float side instead of mirroring the stored
            integer. The cell then records what was measured alongside what
            was stored. Arithmetic does not carry it forward.

    Returns:
        A dict with the ``value`` / ``numerator`` / ``denominator`` keys of
        :data:`RATIONAL_STRUCT_TYPE`.
    """
    if _is_dict(value):
        return _normalise_cell(value)
    number = parse_number(value)
    if number_type is None:
        number_type = NumberType.from_number(number)
    canonical = to_canonical(number, number_type, rounding=rounding)
    # A float field's canonical value is a float, an int or fraction field's
    # an int or a Fraction.
    if isinstance(canonical, float):
        numerator, denominator = _pair_from_float(canonical)
        return {
            "value": canonical,
            "numerator": numerator,
            "denominator": denominator,
        }
    numerator, denominator = _pair_from_exact(canonical)
    mirror = (
        float(number)
        if preserve_source_float and number_type is NumberType.int
        else float(canonical)
    )
    return {"value": mirror, "numerator": numerator, "denominator": denominator}


def _normalise_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Complete and type-check an already-shaped storage cell."""
    if set(cell) != {"value", "numerator", "denominator"}:
        raise ValueError(f"Invalid coordinate dict structure: {cell}")
    numerator = cell.get("numerator")
    denominator = cell.get("denominator")
    if numerator is None or denominator is None:
        value = cell.get("value")
        if value is None:
            return {"value": None, "numerator": None, "denominator": None}
        return number_cell(float(value), NumberType.float)
    numerator = _coerce_ratio_component(numerator, "numerator")
    denominator = _coerce_ratio_component(denominator, "denominator")
    if denominator:
        # One spelling per value in memory: 2/4 is held as 1/2.
        ratio = Fraction(numerator, denominator)
        numerator, denominator = ratio.numerator, ratio.denominator
    return {
        "value": (None if cell["value"] is None else float(cell["value"])),
        "numerator": numerator,
        "denominator": denominator,
    }


def _coerce_ratio_component(component: Any, label: str) -> int:
    """Return one ratio component as an integer."""
    if isinstance(component, bool):
        raise ValueError(f"rational struct {label} must be an integer, got a bool")
    if isinstance(component, int) or _is_numpy_integer(component):
        return int(component)
    if isinstance(component, float):
        if component.is_integer():
            return int(component)
        raise ValueError(
            f"rational struct {label} must be an integer or integer-valued float, "
            f"got non-integral float {component!r}"
        )
    raise ValueError(
        f"rational struct {label} must be an integer or integer-valued float, "
        f"got {component!r}"
    )


def struct_to_rational(struct: dict[str, Any]) -> Fraction:
    """Recover the exact ratio carried by a storage cell.

    Both sides of a cell are always populated, so this is total over every
    non-null cell: the ratio is read straight out, never reconstructed from
    the float side.

    Raises:
        ValueError: If either component is missing or non-integral, or the
            denominator is zero.
    """
    numerator = _coerce_ratio_component(struct.get("numerator"), "numerator")
    denominator = _coerce_ratio_component(struct.get("denominator"), "denominator")
    if denominator == 0:
        raise ValueError("rational struct denominator must be non-zero")
    return Fraction(numerator, denominator)


def struct_to_coordinate(
    struct: dict[str, Any], number_type: NumberType
) -> int | float | Fraction:
    """Decode a storage cell into the requested representation.

    Pass the owning field's declared ``number_type`` to get the canonical
    side back untouched; pass another to read the mirror.

    Args:
        struct: A cell shaped like :data:`RATIONAL_STRUCT_TYPE`.
        number_type: Representation to return.

    Returns:
        The number in the requested representation.
    """
    if number_type is NumberType.float:
        value = struct.get("value")
        if value is not None:
            return float(value)
        return float(struct_to_rational(struct))
    if struct.get("numerator") is None or struct.get("denominator") is None:
        # A well-formed cell always carries both sides, but this decoder also
        # reads cells built by hand and columns written before that was so.
        # The float side is then the only place the number lives, and its own
        # exact ratio is the honest answer -- not a refusal, and not a tidier
        # ratio nearby.
        value = struct.get("value")
        if value is None:
            raise ValueError(f"cell {struct} carries no usable value")
        exact = Fraction(float(value))
    else:
        exact = struct_to_rational(struct)
    if number_type is NumberType.int:
        if exact.denominator != 1:
            raise ValueError(
                f"cell {struct} carries the non-integral value {exact}; it cannot "
                "be read as an integer"
            )
        return exact.numerator
    return exact


def coordinate_to_struct(
    coordinate: int | float | Fraction | dict[str, Any],
) -> dict[str, Any]:
    """Encode one coordinate as a storage cell in its own natural form.

    A convenience spelling of :func:`number_cell` for callers who hold a
    value but no field: the representation the Python value already expresses
    becomes the canonical one, and the mirror is filled in.  An
    already-shaped cell passes through, completed and type-checked.
    """
    return number_cell(coordinate)


def rational_to_struct(value: Any) -> dict[str, Any]:
    """Encode one value as a storage cell whose ratio is authoritative.

    The spelling to use where the source genuinely counts in ratios — a
    score's metrical onsets, a note's notated duration — so that the cell
    declares the exact side canonical no matter what Python type the value
    happened to arrive as.
    """
    return number_cell(value, NumberType.fraction)


# -- JSON number form -------------------------------------------------------


def number_to_json(value: object) -> int | float | list[int]:
    """Encode a Python number using the TimeToShare JSON representation.

    Integers and floats retain their JSON scalar forms. Exact fractions use
    their numerator and strictly positive denominator as a two-item list.

    Args:
        value: Number to encode.

    Returns:
        A JSON integer, JSON number, or exact Fraction pair.

    Raises:
        TypeError: If *value* is not an accepted Python number.
        ValueError: If a number cannot be represented as JSON or int64.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError(f"cannot encode {value!r} as a JSON number")
    if isinstance(value, int):
        return _require_int64(value, "integer", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"cannot encode non-finite JSON number {value!r}")
        return value
    return [
        _require_int64(value.numerator, "numerator", value),
        _require_int64(value.denominator, "denominator", value),
    ]


def _integral_member(component: object) -> int | None:
    """The integer a Fraction member denotes, or None if it denotes none.

    A member is an integer, or a float with no fractional part: a JSON round
    trip through JavaScript may respell ``8`` as ``8.0``, which still denotes
    the integer 8. A boolean denotes no integer.
    """
    if isinstance(component, bool):
        return None
    if isinstance(component, int):
        return component
    if (
        isinstance(component, float)
        and math.isfinite(component)
        and component.is_integer()
    ):
        return int(component)
    return None


def is_fraction_json(value: object) -> bool:
    """Return whether *value* is a valid JSON Fraction pair.

    Both members denote integers (an integral float such as ``8.0`` counts),
    and the denominator is strictly positive.
    """
    if not _is_list(value) or len(value) != 2:
        return False
    numerator, denominator = (_integral_member(component) for component in value)
    return numerator is not None and denominator is not None and denominator > 0


def number_from_json(
    data: Any, number_type: NumberType | None = None
) -> int | float | Fraction:
    """Decode one TimeToShare JSON number.

    Without *number_type* the JSON form decides: an integer is an ``int``, a
    number with a fractional part or exponent is a ``float``, a pair is a
    ``Fraction``. A declared *number_type* decides how the value is read,
    never its lexical form, because a JSON round trip (JavaScript does not
    tell ``2.0`` from ``2``) may respell any number: under ``int`` an
    integral JSON number reads as that integer, under ``float`` an integer
    reads as the float it equals exactly, and under ``fraction`` a pair is
    read. A Fraction member may be an integral float (``[1, 8.0]`` is 1/8)
    whatever the number type. Still refused: a non-integral value under
    ``int``, a non-integral Fraction member, a denominator below 1, a pair
    under ``int`` or ``float``, and a bare number under ``fraction``.

    Args:
        data: JSON scalar or Fraction pair to decode.
        number_type: The number type that decides the reading, or ``None``
            to let the JSON form decide.

    Returns:
        The corresponding Python ``int``, ``float``, or ``Fraction``.

    Raises:
        TypeError: If *data* is not a JSON value type.
        ValueError: If *data* is not a valid JSON number representation, or
            not one *number_type* admits.
    """
    if isinstance(data, bool) or data is None or isinstance(data, (str, dict)):
        raise ValueError(f"invalid JSON number {data!r}")
    if isinstance(data, int):
        value: int | float | Fraction = _require_int64(data, "integer", data)
    elif isinstance(data, float):
        if not math.isfinite(data):
            raise ValueError(f"invalid JSON number {data!r}")
        value = data
    elif _is_list(data):
        if len(data) != 2:
            raise ValueError(f"invalid JSON Fraction {data!r}: expected two items")
        numerator, denominator = (_integral_member(component) for component in data)
        if numerator is None or denominator is None:
            raise ValueError(
                f"invalid JSON Fraction {data!r}: components must be integers"
            )
        if denominator <= 0:
            raise ValueError(
                f"invalid JSON Fraction {data!r}: denominator must be strictly positive"
            )
        value = Fraction(
            _require_int64(numerator, "numerator", data),
            _require_int64(denominator, "denominator", data),
        )
    else:
        raise TypeError(f"invalid JSON number {data!r} of type {type(data).__name__}")
    if number_type is None:
        return value
    if (
        number_type is NumberType.int
        and isinstance(value, float)
        and value.is_integer()
    ):
        return _require_int64(int(value), "integer", data)
    if number_type is NumberType.float and type(value) is int and float(value) == value:
        return float(value)
    expected = {
        NumberType.int: int,
        NumberType.float: float,
        NumberType.fraction: Fraction,
    }[number_type]
    if type(value) is not expected:
        raise ValueError(
            f"JSON number {data!r} is not a {number_type.name} representation"
        )
    return value


# -- the one array builder --------------------------------------------------


def _pairs_from_float_array(
    values: np.ndarray, *, exact_required: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a whole float column as integer ratios, without a Python loop.

    Vectorised counterpart of :func:`_pair_from_float`, and held to the same
    result: every finite double decomposes as ``mantissa * 2**exponent`` with
    the mantissa exact in 53 bits, so the ratio is read straight out of
    :func:`numpy.frexp` and then reduced by its common powers of two.  The
    denominator exponent is capped at :data:`_MIRROR_DENOMINATOR_EXPONENT`,
    which is where the scalar path stops being exact too. That cap is only
    tolerable for a mirror; pass *exact_required* when the result is the
    canonical side, and a value that would need capping raises instead.
    """
    finite = np.isfinite(values)
    if not finite.all():
        offender = values[~finite][0]
        raise ValueError(f"cannot store non-finite value {offender!r} as a number cell")
    if np.any(np.abs(values) >= float(1 << 63)):
        offender = values[np.abs(values) >= float(1 << 63)][0]
        raise ValueError(f"value {offender!r} is too large to mirror as an int64 ratio")

    _, exponent = np.frexp(values)
    wanted = 53 - exponent
    if exact_required:
        # The canonical side of an exact field may not be approximated, so a
        # value whose dyadic denominator overflows int64 is refused here
        # rather than quietly stored as the nearest one that fits.
        overflowing = wanted > _MIRROR_DENOMINATOR_EXPONENT
        if overflowing.any():
            offender = float(values[overflowing][0])
            _pair_from_exact(Fraction(offender))
    shift = np.clip(wanted, 0, _MIRROR_DENOMINATOR_EXPONENT).astype(np.int64)
    denominator = np.int64(1) << shift

    # Scale by the denominator rather than by its exponent.  The denominator is
    # a power of two, so multiplying by it is exact — the same guarantee an
    # exponent-taking scale offers — and the exponent never crosses into a
    # ufunc argument, where numpy would bind it to a C int and find no loop for
    # it wherever that int is narrower than the value it carries.
    numerator = np.rint(values * denominator.astype(np.float64)).astype(np.int64)

    # Reduce: the ratio is dyadic, so its only common factor is a power of two
    # — the numerator's lowest set bit, and never more than the denominator.
    # A zero numerator has no lowest bit and reduces to 0/1.
    lowest_set_bit = numerator & -numerator
    common = np.where(
        numerator == 0, denominator, np.minimum(lowest_set_bit, denominator)
    )
    return numerator // common, denominator // common


def _as_object_list(values: Any) -> list[Any]:
    """Flatten any supported source column into a list of Python values."""
    if _is_chunked(values):
        values = values.combine_chunks()
    if _is_arrow_column(values):
        return values.to_pylist()
    if _is_series(values):
        return values.tolist()
    if _is_ndarray(values):
        return values.tolist()
    return list(values)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return value is pd.NaT


def build_number_struct_array(
    values: Any,
    *,
    number_type: NumberType,
    rounding: str = "round",
    preserve_source_floats: bool = False,
    on_error: str = "raise",
) -> pa.StructArray:
    """Build a whole column of storage cells under one representation.

    This is the only builder of :data:`RATIONAL_STRUCT_TYPE` arrays in the
    library, and it applies exactly the policy :func:`number_cell` applies to
    a single value: the canonical side is *number_type*, the other side
    mirrors it, and both are populated on every non-null row.

    A plain numeric column takes a vectorised path, and so does a column
    already of :data:`RATIONAL_STRUCT_TYPE`, whose complete cells are taken as
    they stand (a cell carries both sides, so *number_type* has nothing to
    choose). Anything mixed, textual or rational falls back to reading values
    one at a time, because no array kernel parses ``"3/8"``.

    Args:
        values: Source column — a PyArrow array, numpy array, pandas Series,
            or any iterable of numbers, ``"n/d"`` strings, or ``None``.
        number_type: The representation the target field declares.
        rounding: How to make inexact values integral for an ``int`` field.
        preserve_source_floats: Keep incoming floats on the float side of an
            ``int`` field instead of mirroring the stored integers.
        on_error: What to do with a cell that is not a number at all -- an
            empty string, a stray label, ``3/0``. ``"raise"`` (default) says
            so; ``"null"`` records it as missing, which is what a tolerant
            reader of a messy file wants.

    Returns:
        A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.

    Raises:
        ValueError: On an unknown rounding mode or error policy, and on an
            unreadable cell when *on_error* is ``"raise"``.
    """
    if on_error not in ("raise", "null"):
        raise ValueError(f"Unknown error policy: {on_error!r}. Use 'raise' or 'null'.")
    if rounding not in ROUNDING_MODES:
        raise ValueError(
            f"Unknown rounding mode: {rounding!r}. "
            f"Use one of {', '.join(ROUNDING_MODES)}."
        )

    vectorised = _build_numeric_struct_array(
        values,
        number_type=number_type,
        rounding=rounding,
        preserve_source_floats=preserve_source_floats,
    )
    if vectorised is None:
        vectorised = _pass_number_struct_array(values)
    if vectorised is not None:
        return vectorised

    rows: list[dict[str, Any] | None] = []
    for raw in _as_object_list(values):
        if _is_missing(raw):
            rows.append(None)
            continue
        try:
            rows.append(
                number_cell(
                    raw,
                    number_type,
                    rounding=rounding,
                    preserve_source_float=preserve_source_floats,
                )
            )
        except (TypeError, ValueError, ZeroDivisionError):
            if on_error == "raise":
                raise
            rows.append(None)
    return _struct_array_of(rows, RATIONAL_STRUCT_TYPE)


def _numeric_source(values: Any) -> npt.NDArray[Any] | None:
    """Return *values* as a plain float64/int64 numpy array, else ``None``."""
    if _is_chunked(values):
        values = values.combine_chunks()
    if _is_array(values):
        if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
            return None
        values = values.to_numpy(zero_copy_only=False)
    elif _is_series(values):
        values = values.to_numpy()
    elif _is_list(values):
        values = np.asarray(values)
    if not _is_ndarray(values):
        return None
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {values.shape}")
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(
        values.dtype, np.floating
    ):
        return values
    return None


def _build_numeric_struct_array(
    values: Any,
    *,
    number_type: NumberType,
    rounding: str,
    preserve_source_floats: bool,
) -> pa.StructArray | None:
    """Vectorised builder for plain numeric columns; ``None`` if inapplicable."""
    array = _numeric_source(values)
    if array is None:
        return None
    if len(array) == 0:
        return _struct_array_of([], RATIONAL_STRUCT_TYPE)

    is_integer_dtype = np.issubdtype(array.dtype, np.integer)
    null_mask = (
        np.zeros(len(array), dtype=bool) if is_integer_dtype else np.isnan(array)
    )
    working = array.astype(np.float64, copy=True)
    working[null_mask] = 0.0

    if number_type is NumberType.int:
        if is_integer_dtype:
            numerator = array.astype(np.int64)
            mirror = numerator.astype(np.float64)
        else:
            numerator = _round_float_array(working, rounding)
            mirror = working if preserve_source_floats else numerator.astype(np.float64)
        denominator = np.ones(len(array), dtype=np.int64)
        value = mirror
    elif number_type is NumberType.fraction:
        if is_integer_dtype:
            numerator = array.astype(np.int64)
            denominator = np.ones(len(array), dtype=np.int64)
        else:
            numerator, denominator = _pairs_from_float_array(
                working, exact_required=True
            )
        value = numerator.astype(np.float64) / denominator.astype(np.float64)
    elif number_type is NumberType.float:
        value = working
        numerator, denominator = _pairs_from_float_array(working)
    else:
        raise ValueError(f"Unknown NumberType: {number_type!r}")

    if null_mask.any():
        return _struct_array_from(
            [
                pa.array(value, type=pa.float64(), mask=null_mask),
                pa.array(numerator, type=pa.int64(), mask=null_mask),
                pa.array(denominator, type=pa.int64(), mask=null_mask),
            ],
            fields=_member_fields(RATIONAL_STRUCT_TYPE),
            mask=pa.array(null_mask),
        )
    return _struct_array_from(
        [
            pa.array(value, type=pa.float64()),
            pa.array(numerator, type=pa.int64()),
            pa.array(denominator, type=pa.int64()),
        ],
        fields=_member_fields(RATIONAL_STRUCT_TYPE),
    )


def _pass_number_struct_array(values: Any) -> pa.StructArray | None:
    """Take a column of complete number structs whole; ``None`` if inapplicable.

    Vectorised counterpart of what :func:`number_cell` does to one cell dict
    with both sides present: the cell is kept, its ratio in lowest terms
    (:func:`in_lowest_terms`), so memory holds one spelling of each value. A
    column of another struct type, or one holding a non-null cell with a
    null numerator or denominator, is left to the per-row reader, which
    knows what such a cell means.
    """
    if _is_chunked(values):
        values = values.combine_chunks()
    if not isinstance(values, pa.StructArray) or [
        (field.name, field.type) for field in _member_fields(values.type)
    ] != [(field.name, field.type) for field in _member_fields(RATIONAL_STRUCT_TYPE)]:
        return None
    valid = values.is_valid()
    # flatten() applies the slice offset and nulls every side of a null cell.
    sides = _struct_members(values)
    incomplete = pc.and_(valid, pc.or_(sides[1].is_null(), sides[2].is_null()))
    if pc.any(incomplete).as_py():
        return None
    return in_lowest_terms(
        _struct_array_from(
            sides, fields=_member_fields(RATIONAL_STRUCT_TYPE), mask=pc.invert(valid)
        )
    )


def _lowest_terms(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each ratio as ``Fraction(n, d)`` holds it: lowest terms, sign on top.

    Vectorized over two int64 columns; a zero denominator is left as it
    stands.
    """
    divisor = np.gcd(numerator, denominator)
    divisor = np.where(denominator < 0, -divisor, divisor)
    divisor = np.where(divisor == 0, 1, divisor)
    return numerator // divisor, denominator // divisor


def in_lowest_terms(numbers: Any) -> pa.StructArray:
    """A column of numbers or coordinate cells, every ratio in lowest terms.

    Memory holds one spelling of each value, the one ``Fraction(n, d)``
    holds: a cell holding ``2/4`` holds ``1/2``, ``1/-2`` is ``-1/2``. The
    other members (the float, a cell's ``unit`` and ``number_type``) are
    kept, and so is a cell missing its numerator or denominator. Every
    entry into memory of a column of number structs reduces through here
    (:func:`build_number_struct_array` and the MatchClaim table's reader),
    so cells compare by value wherever they are compared. Vectorized.

    Args:
        numbers: A struct column holding the members ``numerator`` and
            ``denominator`` (:data:`RATIONAL_STRUCT_TYPE`,
            :data:`COORDINATE_CELL_TYPE`).

    Returns:
        A ``pa.StructArray`` of the type of *numbers*.

    Examples:
        >>> numbers = pa.array(
        ...     [{"value": 0.5, "numerator": 2, "denominator": 4}],
        ...     RATIONAL_STRUCT_TYPE,
        ... )
        >>> in_lowest_terms(numbers).to_pylist()
        [{'value': 0.5, 'numerator': 1, 'denominator': 2}]
    """
    if _is_chunked(numbers):
        numbers = numbers.combine_chunks()
    names = numbers.type.names
    # flatten() applies the slice offset and nulls every member of a null cell.
    members = numbers.flatten()
    at_n, at_d = names.index("numerator"), names.index("denominator")
    numerator: pa.Array[Any] = members[at_n]
    denominator: pa.Array[Any] = members[at_d]
    complete = pc.and_(numerator.is_valid(), denominator.is_valid())
    n, d = _lowest_terms(
        numerator.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64),
        denominator.fill_null(1).to_numpy(zero_copy_only=False).astype(np.int64),
    )
    members[at_n] = pc.if_else(complete, pa.array(n, pa.int64()), numerator)
    members[at_d] = pc.if_else(complete, pa.array(d, pa.int64()), denominator)
    return _struct_array_from(
        members, fields=_member_fields(numbers.type), mask=numbers.is_null()
    )


def _round_float_array(values: np.ndarray, rounding: str) -> np.ndarray:
    """Make a float column integral under the named rounding mode."""
    if rounding == "round":
        # numpy's rint settles ties on the even integer, matching Python's
        # round() and PyArrow's half_to_even.
        return np.rint(values).astype(np.int64)
    if rounding == "floor":
        return np.floor(values).astype(np.int64)
    if rounding == "ceil":
        return np.ceil(values).astype(np.int64)
    if rounding == "truncate":
        return np.trunc(values).astype(np.int64)
    raise ValueError(
        f"Unknown rounding mode: {rounding!r}. Use one of {', '.join(ROUNDING_MODES)}."
    )


def build_coordinate_struct_array(objects: Iterable[Any]) -> pa.StructArray:
    """Build a storage column from scalars exposing a ``.value``.

    Used by the pydantic → Arrow translator for ``Coordinate`` / ``Duration``
    and their Id-variants, whose ``value`` is denormalised into the storage
    struct while ``unit`` and ``timeline_id`` go to field metadata.  Each
    scalar keeps the representation it holds, so a column of exact scalars
    stays exact and a column of floats stays float.
    """
    rows: list[dict[str, Any] | None] = []
    for obj in objects:
        rows.append(None if obj is None else number_cell(obj.value))
    return _struct_array_of(rows, RATIONAL_STRUCT_TYPE)


# ---------------------------------------------------------------------------
# Arithmetic — one engine for scalars and fields alike
# ---------------------------------------------------------------------------


_ADDITIVE_OPERATIONS: frozenset[str] = frozenset({"add", "subtract"})

_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    # ``a / b`` and ``a // b`` are what ``operator.truediv`` and
    # ``operator.floordiv`` compute; typeshed leaves those two unannotated.
    "divide": lambda a, b: a / b,
    "floor_divide": lambda a, b: a // b,
}


def combine_numbers(
    left: Any,
    right: Any,
    operation: str,
    number_type: NumberType,
    *,
    rounding: str = "round",
) -> int | float | Fraction:
    """Combine two numbers in a declared representation.

    Every arithmetic path in the library — scalar with scalar, field with
    scalar, field with field — funnels through here, so a sum behaves the
    same whether it was written on one value or a million.  Both operands are
    first brought into *number_type*, the operation runs there, and the
    result is brought back into it.  Working in the representation the field
    declares is what keeps exactness from leaking: two thirds of a beat added
    to a third of a beat is one beat, not ``0.9999999999999999``.

    Raises:
        ValueError: On an unknown operation or representation.
    """
    apply = _OPERATIONS.get(operation)
    if apply is None:
        raise ValueError(f"Unknown field arithmetic operation: {operation}")
    left_value = to_canonical(left, number_type, rounding=rounding)
    if operation in _ADDITIVE_OPERATIONS:
        # An added or subtracted operand is a quantity in the same unit, so
        # it belongs in the same representation before anything happens.
        right_value = to_canonical(right, number_type, rounding=rounding)
        return to_canonical(
            apply(left_value, right_value), number_type, rounding=rounding
        )

    # A scaling factor is dimensionless, so converting it into the field's
    # representation would be a category error: halving a tick position
    # means halving it, not multiplying by ``round(0.5)``. The rule is
    # therefore **quantize the result, never the operand** — the arithmetic
    # runs in the most exact representation the two sides afford, and
    # rounding happens once, at the end, on the canonical side.
    factor = parse_number(right)
    if number_type is NumberType.float:
        return float(apply(float(left_value), float(factor)))
    return express_as(
        apply(Fraction(left_value), Fraction(factor)), number_type, rounding=rounding
    )


def express_as(
    value: int | float | Fraction,
    number_type: NumberType,
    *,
    rounding: str = "round",
) -> int | float | Fraction:
    """Re-express a value in a declared representation. THE boundary rule.

    Every value that crosses onto a timeline, field or stamp axis is written
    in that axis's declared ``number_type`` — whatever it was before, however
    it was obtained. A fraction-canonical axis yields fractions, always; a
    float-canonical axis yields floats, always.

    That uniformity is the point. A number's *type* says how it is written,
    not where it came from; letting provenance leak into the type would make
    two coordinates on one axis disagree about what they are, and would
    quietly overload a type system to carry a story it cannot be queried for.
    Estimation is recorded where it can be asked about — ``is_interpolated``
    and its siblings — never in the choice of ``int`` versus ``Fraction``.

    Re-expressing a float as its exact dyadic is not fabrication: the value
    is numerically identical, digit for digit. Fabrication is inventing a
    *tidier* ratio than the number really is, and that remains banned.

    Args:
        value: The number to re-express.
        number_type: The representation the axis declares.
        rounding: How to make a value integral for an ``int`` axis. Rounding
            is licensed here — this is a re-expression the library performs,
            not a value a caller wrote down.
    """
    if (
        number_type is NumberType.int
        and isinstance(value, Fraction)
        and value.denominator != 1
    ):
        return _round_to_int(value, rounding)
    return to_canonical(value, number_type, rounding=rounding)


def express_scalar_as(
    scalar: TimeScalarT,
    number_type: NumberType,
    *,
    rounding: str = "round",
) -> TimeScalarT:
    """Rewrite a typed scalar in a declared representation.

    :func:`express_as` for callers holding a ``Coordinate`` or ``Duration``
    rather than a bare number. The scalar's identity survives -- an
    ``IdCoordinate`` stays bound to its timeline -- because the only thing
    the axis gets to decide is how the number is written.
    """
    if scalar.number_type is number_type:
        return scalar
    return scalar.model_copy(
        update={"value": express_as(scalar.value, number_type, rounding=rounding)}
    )


def combine_number_columns(
    left: pa.Array[Any] | pa.ChunkedArray[Any],
    right: pa.Array[Any] | pa.ChunkedArray[Any] | int | float | Fraction,
    operation: str,
    number_type: NumberType,
    *,
    rounding: str = "round",
    row_numbers: Sequence[int] | np.ndarray | None = None,
) -> pa.StructArray:
    """Combine a column of storage cells with another column or a scalar.

    Same rules as :func:`combine_numbers`, applied to whole columns, and held
    to the same result cell for cell: the operation runs on the canonical
    side and the mirror is rebuilt from the result, so the two sides of the
    output never disagree.

    A ``float`` axis runs as PyArrow kernels over its doubles. An ``int`` or
    ``fraction`` axis is computed exactly, on the ratio side, and never on
    doubles: ``n1/d1 + n2/d2`` is ``(n1·d2 + n2·d1) / (d1·d2)``, then reduced
    to lowest terms (an ``int`` axis rounds that ratio once, under
    *rounding*, where :func:`combine_numbers` would). The products of a row
    are computed in int64 wherever a bound on its magnitudes proves they fit,
    which is every row of a table of small-denominator positions; any other
    row is computed with Python's unbounded integers. Either way no
    intermediate can overflow, and only a result whose lowest terms do not
    fit the cell's int64 members is refused.

    Args:
        left: A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.
        right: Another such array, or a single number.
        operation: One of the keys of :data:`_OPERATIONS`.
        number_type: The representation the result declares.
        rounding: How to make inexact results integral for an ``int`` result.
        row_numbers: The caller's number of the row at each position of
            *left*, by which a refused row is named; defaults to the
            positions themselves.

    Returns:
        A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.

    Raises:
        ValueError: On an unknown operation, or, naming the row, its operands
            and its result, on an exact result whose lowest terms do not fit
            int64.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown field arithmetic operation: {operation}")
    if number_type is NumberType.float:
        return _combine_float_columns(left, right, operation)
    return _combine_exact_columns(
        left, right, operation, number_type, rounding=rounding, row_numbers=row_numbers
    )


_COLUMN_KERNELS: dict[str, Callable[[Any, Any], Any]] = {
    "add": pc.add,
    "subtract": pc.subtract,
    "multiply": pc.multiply,
    "divide": pc.divide,
    "floor_divide": lambda a, b: pc.floor(pc.divide(a, b)),
}

_OPERATION_SYMBOLS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "floor_divide": "//",
}


_COMPARISON_KERNELS: dict[str, _ComparisonKernel] = {
    "equal": "equal",
    "less": "less",
    "less_equal": "less_equal",
    "greater_equal": "greater_equal",
}

# Wide enough for an int64 numerator or denominator; the product of two such
# columns (precision 37 + 37 + 1) still fits decimal256's 76 digits.
_CROSS_MULTIPLICATION_TYPE = pa.decimal256(37, 0)


def compare_number_column(
    column: pa.Array[Any] | pa.ChunkedArray[Any],
    number: int | float | Fraction | pa.Array[Any] | pa.ChunkedArray[Any],
    comparison: str,
    number_type: NumberType,
) -> pa.BooleanArray:
    """Compare a column of storage cells with a number, as the column reads.

    An ``int`` or ``fraction`` column is compared exactly with an exact
    number (an ``int`` or a ``Fraction``), on its ratio side: ``n/d``
    against ``p/q`` is ``n·q`` against ``p·d``, cross-multiplied in
    decimals wide enough that no product overflows. A float compares as a
    key: against the float side of the cells, which holds each stored value
    as the nearest double, so ``0.1`` equals a cell holding ``1/10`` and
    ``10.5`` lies between cells holding ``10`` and ``11``. A ``float``
    column is compared on its float side whatever the number, which is what
    comparing two floats means. An infinite float compares as an
    infinite double does (every finite row lies below ``inf``).

    Args:
        column: A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.
        number: The number to compare every row with, or a second column of
            storage cells, compared row for row.
        comparison: ``"equal"``, ``"less"``, ``"less_equal"`` or
            ``"greater_equal"`` (row against *number*).
        number_type: How the column declares its numbers.

    Returns:
        A boolean array, null where a compared row is null.

    Raises:
        ValueError: If *number* is NaN, which names no position.
    """
    column = _struct_array(column)
    valid = column.is_valid()
    kernel = _COMPARISON_KERNELS[comparison]
    # The ratio side of *number*; ``None`` where it compares as a float key.
    ratio: tuple[Any, Any] | None
    if isinstance(number, (pa.Array, pa.ChunkedArray)):
        number = _struct_array(number)
        valid = pc.and_(valid, number.is_valid())
        float_side: Any = _struct_child(number, "value")
        ratio = _struct_child(number, "numerator"), _struct_child(number, "denominator")
    elif isinstance(number, float):
        if math.isnan(number):
            raise ValueError("NaN names no position: it compares with no cell")
        float_side = pa.scalar(number, pa.float64())
        ratio = None
    else:
        exact = Fraction(number)
        float_side = pa.scalar(float(exact))
        ratio = (
            pa.scalar(Decimal(exact.numerator), _CROSS_MULTIPLICATION_TYPE),
            pa.scalar(Decimal(exact.denominator), _CROSS_MULTIPLICATION_TYPE),
        )
    if ratio is None or number_type is NumberType.float:
        compared = _compared(kernel, _struct_child(column, "value"), float_side)
    else:
        numerator, denominator = ratio
        wide = _CROSS_MULTIPLICATION_TYPE
        left = pc.multiply(
            _struct_child(column, "numerator").cast(wide), denominator.cast(wide)
        )
        right = pc.multiply(
            _struct_child(column, "denominator").cast(wide), numerator.cast(wide)
        )
        compared = _compared(kernel, left, right)
    # A struct's own nulls do not reach its members; carry them over.
    return pc.if_else(valid, compared, pa.scalar(None, type=pa.bool_()))


def _combine_float_columns(
    left: pa.Array[Any] | pa.ChunkedArray[Any],
    right: pa.Array[Any] | pa.ChunkedArray[Any] | int | float | Fraction,
    operation: str,
) -> pa.StructArray:
    """The ``float`` lane of :func:`combine_number_columns`: PyArrow kernels."""
    left_values = _float_side(left)
    if isinstance(right, (pa.Array, pa.ChunkedArray)):
        right_values: Any = _float_side(right)
    else:
        right_values = pa.scalar(float(parse_number(right)), type=pa.float64())
    result = _COLUMN_KERNELS[operation](left_values, right_values)
    return build_number_struct_array(result, number_type=NumberType.float)


def _float_side(array: pa.Array[Any] | pa.ChunkedArray[Any]) -> pa.Array[Any]:
    """The float side of a storage column, null where a cell is null."""
    column = _struct_array(array)
    return pc.if_else(
        column.is_valid(),
        _struct_child(column, "value").cast(pa.float64()),
        pa.scalar(None, type=pa.float64()),
    )


# A row whose cross-products stay below this bound (estimated on doubles, so
# a relative error far below the factor of two to ``2**62``) is computed in
# int64; doubling a remainder below it, as half-to-even rounding does, still
# fits.
_INT64_PRODUCT_BOUND = float(2**61)


def _combine_exact_columns(
    left: pa.Array[Any] | pa.ChunkedArray[Any],
    right: pa.Array[Any] | pa.ChunkedArray[Any] | int | float | Fraction,
    operation: str,
    number_type: NumberType,
    *,
    rounding: str,
    row_numbers: Sequence[int] | np.ndarray | None,
) -> pa.StructArray:
    """The ``int`` and ``fraction`` lanes of :func:`combine_number_columns`.

    Each row is ``combine_numbers`` of its two operands. The rows whose
    operands are complete ratios and whose cross-products provably fit int64
    are computed as numpy int64 columns; every other row (a cell carrying
    only its float side, an operand past int64, a zero divisor, products
    past the bound) is computed one at a time, exactly, by
    :func:`combine_numbers` itself.
    """
    left = _struct_array(left)
    left_n, left_d, left_complete = _ratio_sides(left, number_type)
    valid = left.is_valid().to_numpy(zero_copy_only=False)
    right_column = (
        _struct_array(right) if isinstance(right, (pa.Array, pa.ChunkedArray)) else None
    )
    if right_column is not None:
        right_n, right_d, right_complete = _ratio_sides(right_column, number_type)
        valid = valid & right_column.is_valid().to_numpy(zero_copy_only=False)
    else:
        operand = _exact_operand(right, operation, number_type, rounding)
        right_n = np.full(len(left), 0 if operand is None else operand.numerator)
        right_d = np.full(len(left), 1 if operand is None else operand.denominator)
        right_complete = np.full(len(left), operand is not None)
    right_n, right_d = right_n.astype(np.int64), right_d.astype(np.int64)

    fast = valid & left_complete & right_complete
    if operation in ("divide", "floor_divide"):
        fast &= right_n != 0
    magnitude = np.abs(left_n.astype(np.float64)), np.abs(right_n.astype(np.float64))
    if operation in _ADDITIVE_OPERATIONS:
        bounds = (
            magnitude[0] * right_d + magnitude[1] * left_d,
            left_d * 1.0 * right_d,
        )
    elif operation == "multiply":
        bounds = (magnitude[0] * magnitude[1], left_d * 1.0 * right_d)
    else:
        bounds = (magnitude[0] * right_d, left_d * magnitude[1])
    fast &= (bounds[0] < _INT64_PRODUCT_BOUND) & (bounds[1] < _INT64_PRODUCT_BOUND)
    # Neutral operands where a row is not computed here, so that no column
    # operation below divides by zero; those rows are filled in below.
    left_n = np.where(fast, left_n, 0)
    left_d = np.where(fast, left_d, 1)
    right_n = np.where(fast, right_n, 1)
    right_d = np.where(fast, right_d, 1)

    if operation == "add":
        numerator, denominator = left_n * right_d + right_n * left_d, left_d * right_d
    elif operation == "subtract":
        numerator, denominator = left_n * right_d - right_n * left_d, left_d * right_d
    elif operation == "multiply":
        numerator, denominator = left_n * right_n, left_d * right_d
    else:
        numerator, denominator = left_n * right_d, left_d * right_n
        numerator = np.where(denominator < 0, -numerator, numerator)
        denominator = np.abs(denominator)
        if operation == "floor_divide":
            numerator, denominator = numerator // denominator, np.ones_like(denominator)
    if number_type is NumberType.int:
        if np.any(denominator != 1):
            numerator = _round_ratios(numerator, denominator, rounding)
        denominator = np.ones_like(numerator)
        values = numerator.astype(np.float64)
    else:
        numerator, denominator = _lowest_terms(numerator, denominator)
        values = _ratio_values(numerator, denominator)

    for position in np.flatnonzero(valid & ~fast).tolist():
        left_value = struct_to_coordinate(
            _valid_struct_cell(left[position]), number_type
        )
        right_value = (
            struct_to_coordinate(
                _valid_struct_cell(right_column[position]), number_type
            )
            if right_column is not None
            else right
        )
        result = combine_numbers(
            left_value, right_value, operation, number_type, rounding=rounding
        )
        row = position if row_numbers is None else row_numbers[position]
        exact = Fraction(result)
        for side, component in (
            ("numerator", exact.numerator),
            ("denominator", exact.denominator),
        ):
            if not _fits_int64(component):
                raise ValueError(
                    f"row {row}: {left_value} {_OPERATION_SYMBOLS[operation]} "
                    f"{right_value} = {result}, whose {side} {component} in lowest "
                    f"terms does not fit int64, the width of an exact cell; the "
                    f"result cannot be stored exactly. Give the operands exact "
                    f"values (a ratio written as a ratio, not as a decimal), or "
                    f"declare the axis number_type float"
                )
        numerator[position], denominator[position] = exact.numerator, exact.denominator
        values[position] = float(result)

    null = ~valid
    return _struct_array_from(
        [
            pa.array(values, pa.float64(), mask=null),
            pa.array(numerator, pa.int64(), mask=null),
            pa.array(denominator, pa.int64(), mask=null),
        ],
        fields=_member_fields(RATIONAL_STRUCT_TYPE),
        mask=pa.array(null),
    )


def _ratio_sides(
    column: pa.StructArray, number_type: NumberType
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ratio members of a storage column, and where they are complete.

    A row is complete when it holds both members, a positive denominator,
    and, on an ``int`` axis, the denominator 1: only then is the ratio the
    number :func:`struct_to_coordinate` reads. The members of other rows are
    placeholders.
    """
    members: dict[str, pa.Array[Any]] = dict(
        zip(column.type.names, _struct_members(column), strict=True)
    )
    numerator, denominator = members["numerator"], members["denominator"]
    complete = pc.and_(numerator.is_valid(), denominator.is_valid()).to_numpy(
        zero_copy_only=False
    )
    n = numerator.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    d = denominator.fill_null(1).to_numpy(zero_copy_only=False).astype(np.int64)
    complete &= (d == 1) if number_type is NumberType.int else (d > 0)
    return n, d, complete


def _exact_operand(
    value: Any, operation: str, number_type: NumberType, rounding: str
) -> Fraction | None:
    """A scalar operand as :func:`combine_numbers` reads it, if int64 holds it.

    ``None`` when it does not, or when reading it fails; the rows are then
    computed one at a time, which raises what :func:`combine_numbers` raises.
    """
    try:
        if operation in _ADDITIVE_OPERATIONS:
            exact = Fraction(to_canonical(value, number_type, rounding=rounding))
        else:
            exact = Fraction(parse_number(value))
    except (TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
    if _fits_int64(exact.numerator) and _fits_int64(exact.denominator):
        return exact
    return None


def _round_ratios(
    numerator: np.ndarray, denominator: np.ndarray, rounding: str
) -> np.ndarray:
    """Each ``n/d`` (``d > 0``) made integral as :func:`_round_to_int` would.

    Integer arithmetic only, so no ratio passes through a double.
    """
    floor = numerator // denominator
    if rounding == "floor":
        return floor
    remainder = numerator - floor * denominator
    if rounding == "ceil":
        return floor + (remainder != 0)
    if rounding == "truncate":
        return floor + ((remainder != 0) & (numerator < 0))
    if rounding == "round":
        twice = 2 * remainder
        return floor + (
            (twice > denominator) | ((twice == denominator) & (floor % 2 != 0))
        )
    raise ValueError(
        f"Unknown rounding mode: {rounding!r}. Use one of {', '.join(ROUNDING_MODES)}."
    )


def _ratio_values(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Each ratio's double, the one ``float(Fraction(n, d))`` gives (``d > 0``).

    Dividing the two members as doubles rounds once, and so is correctly
    rounded, wherever both are doubles exactly (magnitudes up to ``2**53``)
    or the denominator is a power of two (the numerator's one rounding is
    then only scaled). The rare other rows are divided exactly.
    """
    values = numerator.astype(np.float64) / denominator.astype(np.float64)
    dyadic = (denominator & (denominator - 1)) == 0
    inexact = ~dyadic & (
        (numerator > _EXACT_DOUBLE_INT)
        | (numerator < -_EXACT_DOUBLE_INT)
        | (denominator > _EXACT_DOUBLE_INT)
    )
    for row in np.flatnonzero(inexact):
        values[row] = float(Fraction(int(numerator[row]), int(denominator[row])))
    return values


# ---------------------------------------------------------------------------
# Raw storage fields — the struct primitives
# ---------------------------------------------------------------------------
#
#     NumberField                  abstract struct-shaped DataField
#     └── RedundantNumberField     the mirror struct, representation-neutral
#         └── DenominateNumberField  + bound unit + declared number_type
#
# These are storage primitives, NOT SemanticFields. The semantic wrappers
# (CoordinateField, DurationField and their Id-variants, further down this
# module) compose them through ``SemanticField._raw_cls``.


def infer_number_type(values: Any) -> NumberType:
    """Guess how a source column writes its numbers.

    Used where no field has declared a representation yet — a bare column
    arriving from a file. Integers read as ``int``, ratios and ``"n/d"``
    strings as ``fraction``, decimals as ``float``. A column that mixes them
    takes the most expressive kind present, so nothing is flattened on the
    way in.
    """
    if _is_chunked(values):
        values = values.combine_chunks()
    if _is_array(values):
        if pa.types.is_integer(values.type):
            return NumberType.int
        if pa.types.is_floating(values.type):
            return NumberType.float
    elif _is_ndarray(values):
        if np.issubdtype(values.dtype, np.integer):
            return NumberType.int
        if np.issubdtype(values.dtype, np.floating):
            return NumberType.float

    seen: set[NumberType] = set()
    for raw in _as_object_list(values):
        if _is_missing(raw):
            continue
        try:
            seen.add(NumberType.from_number(parse_number(raw)))
        except (TypeError, ValueError):
            continue
    if NumberType.fraction in seen:
        return NumberType.fraction
    if NumberType.float in seen:
        return NumberType.float
    return NumberType.int


class NumberField(StructField):
    """Abstract: a struct-shaped DataField holding a (composite) number.

    Concrete subclasses pin the inner sub-schema. Today the only descendant
    is :class:`RedundantNumberField`; future shapes (complex numbers,
    intervals) would be siblings.
    """


class RedundantNumberField(NumberField):
    """Raw field for the mirrored number struct.

    Sub-schema (fixed)::

        {value: float64, numerator: int64, denominator: int64}

    The struct carries the same number twice so that exactness survives a
    Parquet round-trip while a float64 projection stays immediately
    consumable by analytics. Which side is authoritative is not this class's
    business — it is neutral about representation, and
    :class:`DenominateNumberField` is where a declaration attaches.
    """

    PA_SCHEMA: ClassVar[pa.StructType] = RATIONAL_STRUCT_TYPE
    _blueprint_pa_type = RATIONAL_STRUCT_TYPE

    def __init__(
        self,
        data: pa.Array[Any] | pa.ChunkedArray[Any] | None = None,
        field: pa.Field[pa.DataType] | None = None,
        *,
        name: str | None = None,
    ) -> None:
        DataField.__init__(self, data, field, name=name)
        if not _struct_types_match(self._field.type, type(self).PA_SCHEMA):
            raise TypeError(
                f"{type(self).__name__} requires the number struct schema "
                f"{type(self).PA_SCHEMA}; got {self._field.type}"
            )

    def cell(self, i: int) -> dict[str, Any] | None:
        """Return the i-th storage cell as a plain dict.

        The unwrapped view onto the struct, for the semantic fields layered
        on top: they decode a cell straight into whatever they pair with, so
        making them go through this class's own scalar would build an object
        only to throw it away.
        """
        return StructField.__getitem__(self, i)

    @classmethod
    def from_field(
        cls,
        source: tuple[
            pa.Array[Any] | pa.ChunkedArray[Any] | None, pa.Field[pa.DataType]
        ],
        **kw: Any,
    ) -> "RedundantNumberField":
        """Create a field from a ``(data, field)`` tuple.

        Args:
            source: A tuple of ``(pa.Array | None, pa.Field)``; the field
                must have the number struct schema.

        Returns:
            A new field of this class.

        Raises:
            TypeError: If the field type is not the number struct schema.
        """
        cls._warn_ignored_keywords(kw)
        data, field = source
        return cls(data, field)

    def from_array(
        self,
        source: pa.Array[Any] | pa.ChunkedArray[Any],
        *,
        name: str | None = None,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
    ) -> "RedundantNumberField":
        """Encode a source column as storage cells and wrap the result.

        Accepts ``"<numerator>/<denominator>"`` strings as well as plain
        numbers. Without an explicit *number_type* the column's own way of
        writing numbers is kept (see :func:`infer_number_type`).
        """
        if not self.is_empty:
            raise TypeError(
                f"{type(self).__name__}.from_array() can only be called on a blueprint "
                f"(empty DataField); this instance has live data"
            )
        out_name = name if name is not None else self.name
        resolved = (
            infer_number_type(source)
            if number_type is None
            else NumberType(number_type)
        )
        struct_arr = build_number_struct_array(
            source, number_type=resolved, rounding=rounding, on_error="null"
        )
        return RedundantNumberField(
            struct_arr, pa.field(out_name, RATIONAL_STRUCT_TYPE)
        )


class DenominateNumber(ScalarVocabulary, BaseModel):
    """A number that knows both how it is written and what it measures.

    The element type of :class:`DenominateNumberField`, and the only scalar
    in the library that carries the storage struct's redundancy into Python:
    it holds the exact ratio *and* the float projection side by side, plus
    the unit they are in and which of the two is authoritative.

    That makes it the right thing to reach for when the question is "what
    does this cell actually contain" — inspecting a column, reporting a
    precision loss, comparing two loaders' readings of the same file.  It is
    deliberately NOT a parent of :class:`Coordinate` or :class:`Duration`:
    those are positions and extents, they hold exactly one value, and giving
    them a second one would put the library right back into guessing which
    to trust.

    Attributes:
        exact: The value as an exact ratio.
        approximate: The value as a double.
        unit: The unit both sides are measured in.
        number_type: Which side is authoritative.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    exact: Fraction
    approximate: float
    unit: TimeUnit
    number_type: NumberType

    @classmethod
    def from_cell(
        cls, cell: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> "DenominateNumber":
        """Build one from a storage cell and its field's declaration."""
        normalised = _normalise_cell(cell)
        return cls(
            exact=struct_to_rational(normalised),
            approximate=float(normalised["value"]),
            unit=unit,
            number_type=number_type,
        )

    @classmethod
    def from_value(
        cls,
        value: Any,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
        *,
        rounding: str = "round",
    ) -> "DenominateNumber":
        """Build one from any number, under a unit's representation rules."""
        resolved_unit = TimeUnit(unit) if isinstance(unit, str) else unit
        resolved_type = resolved_unit.resolve_number_type(number_type)
        cell = number_cell(value, resolved_type, rounding=rounding)
        return cls.from_cell(cell, resolved_unit, resolved_type)

    @property
    def domain(self) -> Domain:
        """The temporal domain implied by the unit."""
        return self.unit.domain

    @property
    def value(self) -> int | float | Fraction:
        """The authoritative side, in the representation the field declares."""
        if self.number_type is NumberType.float:
            return self.approximate
        if self.number_type is NumberType.int:
            return self.exact.numerator
        return self.exact

    @property
    def is_exact(self) -> bool:
        """Whether the authoritative side is an exact one."""
        return self.number_type is not NumberType.float

    def to_cell(self) -> dict[str, Any]:
        """Render back to the storage cell this came from."""
        return {
            "value": self.approximate,
            "numerator": self.exact.numerator,
            "denominator": self.exact.denominator,
        }

    def metadata_dict(self) -> dict[str, str]:
        """Return the discriminator plus the (unit, domain, number_type) triple."""
        return {
            **super().metadata_dict(),
            "unit": self.unit.value,
            "domain": self.domain.value,
            "number_type": self.number_type.name,
        }

    def __repr__(self) -> str:
        return f"DenominateNumber({self.value!r}, {self.unit}, {self.number_type})"

    def __str__(self) -> str:
        return f"{self.value} {self.unit}"


class DenominateNumberField(RedundantNumberField):
    """A :class:`RedundantNumberField` with a unit and a declared representation.

    Both are per-field, NOT per-row: every row is read in the SAME unit and
    the SAME representation, recorded once in the field's
    :data:`TIMETOALIGN_METADATA_KEY` metadata blob. This is what lets the
    storage struct stay redundant without becoming ambiguous — a reader
    consults the field to learn which side of the struct to trust, and never
    has to sniff the rows.

    Pairs with :class:`DenominateNumber`.

    Args:
        data: The underlying struct array, or ``None`` for blueprint use.
        field: The ``pa.Field`` descriptor.
        unit: The unit bound to this field. Required — a
            DenominateNumberField is defined by its bound unit — but it may
            arrive implicitly through the field's metadata blob.
        number_type: The representation bound to this field. Defaults to the
            unit's own (see :attr:`TimeUnit.default_number_type`).
    """

    scalar_cls: ClassVar[type[BaseModel]] = DenominateNumber

    # A bound unit is not optional, so this class cannot stand in for
    # itself while a blueprint is still deferred; SemanticField reads this
    # flag and substitutes a plain StructField placeholder.
    _requires_binding: ClassVar[bool] = True

    def __init__(
        self,
        data: pa.Array[Any] | pa.ChunkedArray[Any] | None = None,
        field: pa.Field[pa.DataType] | None = None,
        *,
        name: str | None = None,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ):
        if field is None:
            if name is None:
                raise TypeError("DenominateNumberField requires either field= or name=")
            field = pa.field(name, type(self)._default_blueprint_pa_type())
        super().__init__(data, field)
        self._unit: TimeUnit = self._resolve_unit(self._field, unit)
        self._number_type: NumberType = self._unit.resolve_number_type(
            self._resolve_number_type(self._field, number_type)
        )

    # -- properties ----------------------------------------------------------

    @property
    def unit(self) -> TimeUnit:
        """The single unit bound to this field."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The representation this field declares authoritative."""
        return self._number_type

    @property
    def domain(self) -> Any:
        """The temporal domain implied by the unit."""
        return self._unit.domain

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _field_metadata(pa_field: pa.Field[pa.DataType]) -> dict[str, Any]:
        raw_meta = pa_field.metadata
        if not raw_meta or TIMETOALIGN_METADATA_KEY not in raw_meta:
            return {}
        try:
            return parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
        except (UnicodeDecodeError, ValueError):
            return {}

    @staticmethod
    def _resolve_unit(pa_field: pa.Field[pa.DataType], override: Any) -> TimeUnit:
        """Resolve the unit from a kwarg override, then from field metadata.

        Raises:
            ValueError: If neither supplies a unit.
        """
        if override is not None:
            return TimeUnit(override)
        unit = DenominateNumberField._field_metadata(pa_field).get("unit")
        if unit is not None:
            return TimeUnit(unit)
        raise ValueError(
            f"DenominateNumberField requires a unit; field {pa_field.name!r} "
            "carries neither a kwarg unit nor a 'unit' entry in its metadata"
        )

    @staticmethod
    def _resolve_number_type(pa_field: pa.Field[pa.DataType], override: Any) -> Any:
        """Resolve the representation from an override, then field metadata."""
        if override is not None:
            return NumberType(override) if isinstance(override, str) else override
        number_type = DenominateNumberField._field_metadata(pa_field).get("number_type")
        return None if number_type is None else NumberType(number_type)

    def __getitem__(self, i: int) -> DenominateNumber | None:
        cell = self.cell(i)
        if cell is None:
            return None
        return DenominateNumber.from_cell(cell, self._unit, self._number_type)

    @classmethod
    def from_field(
        cls,
        source: tuple[
            pa.Array[Any] | pa.ChunkedArray[Any] | None, pa.Field[pa.DataType]
        ],
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        **kw: Any,
    ) -> "DenominateNumberField":
        """Construct from a ``(data, pa.Field)`` tuple."""
        cls._warn_ignored_keywords(kw)
        data, field = source
        if unit is None:
            unit = cls._resolve_unit(field, None)
        return cls(data, field, unit=unit, number_type=number_type)

    def from_array(
        self,
        source: pa.Array[Any] | pa.ChunkedArray[Any],
        *,
        name: str | None = None,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
    ) -> "DenominateNumberField":
        """Encode a source column under this field's unit and representation.

        The emitted field carries a versioned
        :data:`TIMETOALIGN_METADATA_KEY` blob naming
        both, so a reader of the resulting column never has to be told them
        again.
        """
        if not self.is_empty:
            raise TypeError(
                f"{type(self).__name__}.from_array() can only be called on a blueprint "
                f"(empty DataField); this instance has live data"
            )
        out_name = name if name is not None else self.name
        resolved = (
            self._number_type
            if number_type is None
            else self._unit.resolve_number_type(number_type)
        )
        struct_arr = build_number_struct_array(
            source, number_type=resolved, rounding=rounding, on_error="null"
        )
        pa_field = pa.field(out_name, RATIONAL_STRUCT_TYPE).with_metadata(
            {
                TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                    {"unit": self._unit.value, "number_type": resolved.name}
                )
            }
        )
        return DenominateNumberField(
            struct_arr, pa_field, unit=self._unit, number_type=resolved
        )


def format_number(value: int | float | Fraction, *, discrete: bool = False) -> str:
    """Render a number the way this library shows numbers to people.

    The single formatter behind every pretty rendering — scalars, stamps,
    tables. Having one is the point: a display that shows less than the value
    carries is a lie the reader cannot detect, and two formatters drift into
    exactly that. So a third of a quarter reads ``1/3`` wherever it appears,
    and never ``0.333333``.

    * an exact ratio prints as a ratio;
    * anything integral prints without a denominator or a decimal point,
      including a discrete unit's value, which is integral by definition;
    * a float prints as its shortest round-tripping form, expanded out of
      scientific notation so a small number never reads as ``0``.
    """
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return str(value)
    if discrete or isinstance(value, int):
        return str(int(value))
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    rendered = str(value)
    if "e" in rendered.lower():
        rendered = format(Decimal(rendered), "f")
    return rendered


def _settle_under_unit(
    value: TimeScalarValue,
    unit: TimeUnit,
    *,
    rounding: str = "round",
) -> TimeScalarValue:
    """Hold *value* to a representation its unit admits.

    The hard constraint, applied on every construction path including the
    ones that bypass ``__init__``. A value already written in an admissible
    representation is handed back untouched; one whose kind the unit cannot
    express at all is converted to the unit's own. The softer question of
    which admissible representation to *prefer* belongs to
    :func:`construct_scalar_value`.
    """
    if NumberType.from_number(value) in unit.allowed_number_types:
        return value
    return to_canonical(value, unit.default_number_type, rounding=rounding)


def construct_scalar_value(
    value: TimeScalarValue,
    unit: TimeUnit,
    number_type: NumberType | str | None = None,
    *,
    rounding: str = "round",
) -> TimeScalarValue:
    """Decide what representation a newly constructed scalar holds.

    A scalar holds exactly one value, so something has to choose how it is
    written, and that choice is the unit's — not whichever Python type the
    caller's literal happened to have. ``Coordinate(2, quarters)`` and
    ``Coordinate(1.5, quarters)`` are both quarter positions and both come
    out exact, rather than differing because one was typed with a decimal
    point.

    An explicit *number_type* overrides that, validated against what the
    unit admits, and is how a caller deliberately chooses the float lane.

    Coercion toward the default never loses information:

    * into ``fraction``: an integer becomes ``Fraction(v, 1)`` and a float
      becomes its exact dyadic. There is no int64 ceiling here — Python's
      ``Fraction`` is arbitrary-precision, and the storage refusal belongs
      at the field boundary where the ceiling actually exists.
    * into ``float``: an exact ``Fraction`` is **kept**, because silently
      degrading an exact input is the thing this whole scheme exists to
      prevent; degrading it takes an explicit ``number_type=float``. An
      integer widens only if it survives the trip.
    * into ``int`` (the discrete units): an inexact value is rounded, an
      exact non-integral one raises.
    """
    if number_type is not None:
        return to_canonical(
            value, unit.resolve_number_type(number_type), rounding=rounding
        )

    default = unit.default_number_type
    kind = NumberType.from_number(value)
    if kind is default:
        return value
    if default is NumberType.int:
        return to_canonical(value, NumberType.int, rounding=rounding)
    if default is NumberType.fraction:
        return Fraction(value)
    if kind is NumberType.fraction:
        # Never silently degrade an exact value; ask for it explicitly.
        return value
    widened = float(value)
    if widened != value:
        raise ValueError(
            f"{value!r} cannot be held exactly as a float in {unit}; pass "
            f"number_type='fraction' to keep it exact"
        )
    return widened


# ---------------------------------------------------------------------------
# Cell-level helpers for engine code that moves coordinates around
# ---------------------------------------------------------------------------


def exact_coordinate_value(value: Any) -> Fraction | None:
    """Return the exact ratio a coordinate carries, or ``None``.

    Accepts a storage cell, a bare number, or anything else; returns
    ``None`` only when there is genuinely no ratio to be had.
    """
    if _is_dict(value):
        try:
            return struct_to_rational(value)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Fraction(value, 1)
    return None


def coordinate_numeric_value(value: Any) -> Fraction | float:
    """Return the number a coordinate represents, exactly where possible."""
    exact = exact_coordinate_value(value)
    if exact is not None:
        return exact
    if _is_dict(value):
        return struct_to_rational(value)
    if hasattr(value, "value"):
        return value.value
    return float(value)


def shift_number(value: Any, offset: Any, *, subtract: bool) -> Fraction | float:
    """Shift a coordinate by an offset: one step between a parent and a child.

    The step runs on the exact side whenever both operands have one, so it
    stays exact on an exact axis; otherwise it is one float addition or
    subtraction. Either way a zero offset gives back the value unchanged.
    """
    value_exact = exact_coordinate_value(value)
    offset_exact = exact_coordinate_value(offset)
    if value_exact is not None and offset_exact is not None:
        return value_exact - offset_exact if subtract else value_exact + offset_exact
    left = float(coordinate_numeric_value(value))
    right = float(coordinate_numeric_value(offset))
    return left - right if subtract else left + right


def shift_coordinate(value: Any, offset: Any, *, subtract: bool) -> dict[str, Any]:
    """Shift a coordinate by an offset and rebuild its storage cell.

    Both sides of a cell always carry the number, so the shift runs on the
    exact side whenever the operands have one -- which, for a stored
    coordinate, is always. Parent-to-child offsetting therefore composes
    without drifting, however deep the nesting goes.
    """
    return coordinate_to_struct(shift_number(value, offset, subtract=subtract))


def subtract_coordinates(left: Any, right: Any) -> dict[str, Any]:
    """Return the storage cell for ``left - right``."""
    return shift_coordinate(left, right, subtract=True)


def _apply_construction_protocol(
    data: dict[str, Any],
    *,
    number_type: NumberType | str | None,
    rounding: str,
) -> dict[str, Any]:
    """Run :func:`construct_scalar_value` over a scalar's constructor kwargs.

    Only when both a value and a unit are actually present — pydantic owns
    the diagnostics for anything malformed, and stepping in front of it here
    would replace a precise validation error with a vaguer one.
    """
    value = data.get("value")
    unit = data.get("unit")
    if value is None or unit is None:
        return data
    # Anything that is not a real number is pydantic's to reject, and its
    # diagnostic is better than one improvised here. Bools in particular are
    # ints to Python and not numbers to this library.
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        return data
    try:
        resolved_unit = TimeUnit(unit) if isinstance(unit, str) else unit
    except ValueError:
        return data
    if not isinstance(resolved_unit, TimeUnit):
        return data
    return {
        **data,
        "value": construct_scalar_value(
            value, resolved_unit, number_type, rounding=rounding
        ),
    }


# ---------------------------------------------------------------------------
# TimeScalar — abstract base for Coordinate / Duration / Id-variants
# ---------------------------------------------------------------------------


class TimeScalar(ScalarVocabulary, BaseModel):
    """Abstract base for ``Coordinate`` / ``Duration`` and their Id-variants.

    Holds the (value, unit) pair plus all shared mechanics: validators,
    conversions, comparison, status predicates, stringification, and the
    operator-compatibility helper :meth:`_binop_other` that subclasses
    use to centralise type/unit/timeline-id checking.

    ``TimeScalar`` is not constructed directly; instantiate ``Coordinate``,
    ``Duration``, ``IdCoordinate``, or ``IdDuration`` instead.
    """

    model_config = ConfigDict(
        frozen=True,
        # pydantic does NOT know about Fraction natively;
        # arbitrary_types is required so the field validator below can
        # accept Fraction.
        arbitrary_types_allowed=True,
    )

    value: TimeScalarValue
    unit: TimeUnit

    def __init__(
        self,
        value: TimeScalarValue | dict[str, Any] | None = None,
        unit: TimeUnit | str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        """The constructor of ``Coordinate`` and ``Duration``.

        Takes the value and the unit positionally or as keywords, never both,
        and settles the value under the scalar construction protocol
        (:func:`_apply_construction_protocol`).
        """
        if type(self) is TimeScalar:
            raise TypeError(
                "TimeScalar is abstract; construct Coordinate, Duration, "
                "IdCoordinate or IdDuration"
            )
        if value is not None or unit is not None:
            if "value" in data or "unit" in data:
                raise TypeError(
                    f"{type(self).__name__} received conflicting positional "
                    "and keyword arguments"
                )
            data = {"value": value, "unit": unit, **data}
        data = _apply_construction_protocol(
            data, number_type=number_type, rounding=rounding
        )
        super().__init__(**data)

    # -- validators ---------------------------------------------------------

    @model_validator(mode="after")
    def _apply_unit_number_policy(self) -> TimeScalar:
        """Hold the value to what its unit can actually express.

        A unit knows which representations mean anything for it, so this is
        where a value that cannot be written in one of them gets settled --
        once, for every scalar in the hierarchy, rather than per subclass.

        In practice this only bites on the discrete units. A measurement that
        arrives inexact is made integral, because that is what reading a
        tick position off a float clock means. An **exact** non-integral
        value is refused: ``Coordinate(Fraction(5, 24), TimeUnit.ticks)``
        is not a rounding candidate, it is somebody having mixed up
        quarters and ticks, and rounding it to zero would bury the mistake
        somewhere much harder to find.
        """
        settled = _settle_under_unit(self.value, self.unit)
        if settled is not self.value:
            object.__setattr__(self, "value", settled)
        return self

    @field_validator("value", mode="before")
    @classmethod
    def _validate_value(cls, v: object) -> TimeScalarValue:
        if isinstance(v, bool):
            raise TypeError(
                f"Boolean values are not valid {cls.__name__.lower()} values"
            )
        if isinstance(v, (int, float, Fraction)):
            return v
        raise TypeError(
            f"{cls.__name__} value must be int, float, or Fraction, "
            f"got {type(v).__name__}"
        )

    @field_validator("unit", mode="before")
    @classmethod
    def _validate_unit(cls, v: object) -> TimeUnit:
        if isinstance(v, TimeUnit):
            return v
        if isinstance(v, str):
            return TimeUnit(v)
        raise TypeError(
            f"{cls.__name__} unit must be a TimeUnit or string, got {type(v).__name__}"
        )

    # -- conversions --------------------------------------------------------

    @data_shaped
    def to_float(self) -> float:
        """Convert value to ``float``."""
        return float(self.value)

    @data_shaped
    def to_int(self, rounding: str = "round") -> int:
        """Convert value to ``int`` using the given rounding mode.

        Args:
            rounding: One of ``"round"`` (default, nearest, ties to even),
                ``"floor"`` (towards −∞), ``"ceil"`` (towards +∞), or
                ``"truncate"`` (towards zero). One vocabulary and one
                default apply wherever the library makes a value integral.
        """
        return _round_to_int(self.value, rounding)

    def to_fraction(self) -> Fraction:
        """Convert value to ``Fraction``, exactly.

        A float converts to the ratio it actually is, not to a tidier one
        that happens to be nearby: ``0.1`` is
        ``3602879701896397/36028797018963968``. Asking a float for its exact
        value should not invent an exactness it never had.
        """
        if isinstance(self.value, Fraction):
            return self.value
        return Fraction(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    # -- properties ---------------------------------------------------------

    @property
    def number_type(self) -> NumberType:
        """The representation the value is held in, read off its Python type.

        Raises:
            TypeError: If the value is a ``bool`` or not a number.
        """
        v = self.value
        if isinstance(v, bool):
            raise TypeError("Boolean values are not valid time-scalar values")
        if isinstance(v, int):
            return NumberType.int
        if isinstance(v, float):
            return NumberType.float
        if isinstance(v, Fraction):
            return NumberType.fraction
        raise TypeError(f"Unknown number type for {type(v)}")  # pragma: no cover

    @property
    def domain(self) -> Domain:
        """The temporal domain implied by the unit."""
        return self.unit.domain

    @property
    def semantic_type(self) -> str:
        """Canonical name of this scalar's type (overridden by subclasses)."""
        return type(self).__name__

    def metadata_dict(self) -> dict[str, str]:
        """Return the discriminator plus the (unit, domain, number_type) triple."""
        return {
            **super().metadata_dict(),
            "unit": self.unit.value,
            "domain": self.domain.value,
            "number_type": self.number_type.name,
        }

    def to_dict(self) -> dict[str, Any]:
        """Render the numeric value exactly as an Arrow coordinate cell."""
        return coordinate_to_struct(self.value)

    # -- internal helper used by all operators -----------------------------

    def _id_or_none(self) -> str | None:
        """Return ``self.timeline_id`` if present, else ``None``."""
        return getattr(self, "timeline_id", None)

    def _binop_other(
        self, other: object, op_name: str
    ) -> tuple[TimeScalarValue, str | None]:
        """Validate ``other`` for a binary op against ``self``.

        Returns ``(numeric_value_of_other, result_timeline_id)``.  Raises
        ``TypeError`` on incompatible types, unit mismatch, or
        cross-timeline-id mixing of two Id-scalars with different ids.
        """
        if isinstance(other, bool):
            raise TypeError(f"Cannot {op_name} {type(self).__name__} with bool")
        if isinstance(other, (int, float, Fraction)):
            return other, self._id_or_none()
        if isinstance(other, TimeScalar):
            if self.unit != other.unit:
                raise TypeError(
                    f"Cannot {op_name} {type(self).__name__} and "
                    f"{type(other).__name__} with different units: "
                    f"{self.unit} vs {other.unit}"
                )
            self_id = self._id_or_none()
            other_id = other._id_or_none()
            if self_id is not None and other_id is not None and self_id != other_id:
                raise TypeError(
                    f"Cannot {op_name} {type(self).__name__} and "
                    f"{type(other).__name__} with mismatched timeline_id: "
                    f"{self_id!r} vs {other_id!r}"
                )
            result_id = self_id if self_id is not None else other_id
            return other.value, result_id
        raise TypeError(
            f"Cannot {op_name} {type(self).__name__} with {type(other).__name__}"
        )

    def _combine(
        self, other: object, op_name: str, operation: str
    ) -> tuple[TimeScalarValue, str | None]:
        """Validate *other*, then combine it with this scalar's value.

        The arithmetic itself is :func:`combine_numbers`, run in this
        scalar's own representation, so an exact left operand stays exact:
        a third of a beat plus half a beat is five sixths of a beat, not
        ``0.8333333333333333``. The left operand decides, matching how the
        paired fields behave.
        """
        other_value, tl_id = self._binop_other(other, op_name)
        return (
            combine_numbers(self.value, other_value, operation, self.number_type),
            tl_id,
        )

    def _scale(
        self, factor: object, op_name: str, operation: str, symbol: str
    ) -> TimeScalarValue:
        """Validate a bare numeric *factor*, then scale this scalar's value."""
        if isinstance(factor, TimeScalar):
            raise TypeError(
                f"Cannot {op_name} two TimeScalars: "
                f"{type(self).__name__} {symbol} {type(factor).__name__}"
            )
        if isinstance(factor, bool) or not isinstance(factor, (int, float, Fraction)):
            raise TypeError(
                f"Cannot {op_name} {type(self).__name__} by {type(factor).__name__}"
            )
        if operation in ("divide", "floor_divide") and factor == 0:
            raise ZeroDivisionError(f"Cannot divide {type(self).__name__} by zero")
        return combine_numbers(self.value, factor, operation, self.number_type)

    # -- comparisons --------------------------------------------------------
    #
    # NB: pydantic's default ``__eq__`` / ``__hash__`` already compare ALL
    # fields (including ``timeline_id``), which is what we want — do NOT
    # override them.

    def _cmp_value(self, other: object, op_name: str) -> TimeScalarValue:
        if isinstance(other, bool):
            raise TypeError(f"Cannot {op_name} {type(self).__name__} with bool")
        if isinstance(other, (int, float, Fraction)):
            return other
        if isinstance(other, TimeScalar):
            if self.unit != other.unit:
                raise TypeError(
                    f"Cannot {op_name} {type(self).__name__} and "
                    f"{type(other).__name__} with different units: "
                    f"{self.unit} vs {other.unit}"
                )
            self_id = self._id_or_none()
            other_id = other._id_or_none()
            if self_id is not None and other_id is not None and self_id != other_id:
                raise TypeError(
                    f"Cannot {op_name} {type(self).__name__} and "
                    f"{type(other).__name__} with mismatched timeline_id: "
                    f"{self_id!r} vs {other_id!r}"
                )
            return other.value
        raise TypeError(
            f"Cannot {op_name} {type(self).__name__} with {type(other).__name__}"
        )

    @data_shaped
    def __lt__(self, other: object) -> bool:
        return self.value < self._cmp_value(other, "compare")

    @data_shaped
    def __le__(self, other: object) -> bool:
        return self.value <= self._cmp_value(other, "compare")

    @data_shaped
    def __gt__(self, other: object) -> bool:
        return self.value > self._cmp_value(other, "compare")

    @data_shaped
    def __ge__(self, other: object) -> bool:
        return self.value >= self._cmp_value(other, "compare")

    # -- status predicates --------------------------------------------------

    @data_shaped
    def is_zero(self) -> bool:
        """Whether the value equals zero."""
        return self.value == 0

    @data_shaped
    def is_positive(self) -> bool:
        """Whether the value is greater than zero."""
        return self.value > 0

    @data_shaped
    def is_negative(self) -> bool:
        """Whether the value is less than zero."""
        return self.value < 0

    # -- mutators (copy-on-write) ------------------------------------------

    def _rebuild(self, value: TimeScalarValue, unit: TimeUnit) -> TimeScalar:
        """Construct a same-class instance with possibly new value/unit.

        Preserves ``timeline_id`` for Id-variants.
        """
        tl_id = self._id_or_none()
        if isinstance(self, IdTimeScalar) and tl_id is not None:
            return type(self)(value, unit, tl_id)
        return type(self)(value, unit)

    def with_value(self, new_value: TimeScalarValue) -> TimeScalar:
        """Return a same-class instance with a different value."""
        return self._rebuild(new_value, self.unit)

    def with_unit(self, new_unit: TimeUnit) -> TimeScalar:
        """Return a same-class instance with a different unit.

        Warning: this does NOT convert the value — use a ConversionMap
        for that.  Reinterpretation only.
        """
        return self._rebuild(self.value, new_unit)

    # -- stringification helpers -------------------------------------------

    def _format_value(self) -> str:
        """Format ``self.value`` for ``__str__`` via the shared formatter."""
        return format_number(self.value, discrete=self.unit.is_discrete)


# ---------------------------------------------------------------------------
# IdTimeScalar — abstract; contributes only the ``timeline_id`` field
# ---------------------------------------------------------------------------


class IdTimeScalar(TimeScalar):
    """Abstract mixin contributing ``timeline_id`` + its validator.

    Construct ``IdCoordinate`` or ``IdDuration`` instead — instantiating
    ``IdTimeScalar`` directly raises ``TypeError``.
    """

    timeline_id: str

    def __init__(
        self,
        value: TimeScalarValue | None = None,
        unit: TimeUnit | str | None = None,
        timeline_id: str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        """The constructor of ``IdCoordinate`` and ``IdDuration``.

        Takes the value, the unit and the timeline id positionally or as
        keywords, never one of them both ways, and settles the value under
        the scalar construction protocol (:func:`_apply_construction_protocol`).
        """
        if type(self) is IdTimeScalar:
            raise TypeError(
                "IdTimeScalar is abstract; construct IdCoordinate or IdDuration"
            )
        positional = {
            k: v
            for k, v in (
                ("value", value),
                ("unit", unit),
                ("timeline_id", timeline_id),
            )
            if v is not None
        }
        if positional and any(k in data for k in positional):
            raise TypeError(
                f"{type(self).__name__} received conflicting positional and "
                "keyword arguments"
            )
        data = _apply_construction_protocol(
            {**positional, **data}, number_type=number_type, rounding=rounding
        )
        BaseModel.__init__(self, **data)

    @field_validator("timeline_id", mode="before")
    @classmethod
    def _validate_timeline_id(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(f"timeline_id must be a string, got {type(v).__name__}")
        if not v:
            raise ValueError("timeline_id cannot be empty")
        return v


# ---------------------------------------------------------------------------
# Result-class factories
# ---------------------------------------------------------------------------


def _make_coordinate(
    value: TimeScalarValue,
    unit: TimeUnit,
    timeline_id: str | None,
    number_type: NumberType | None = None,
) -> Coordinate:
    """Return ``IdCoordinate`` if a timeline id is given, else ``Coordinate``.

    Arithmetic passes the operand's own representation so a result is not
    re-settled to the unit default: a deliberately float-typed quarters
    coordinate stays float after adding to it.
    """
    if number_type is None:
        number_type = NumberType.from_number(value)
    if timeline_id is not None:
        return IdCoordinate(value, unit, timeline_id, number_type=number_type)
    return Coordinate(value, unit, number_type=number_type)


def _make_duration(
    value: TimeScalarValue,
    unit: TimeUnit,
    timeline_id: str | None,
    number_type: NumberType | None = None,
) -> Duration:
    """Return ``IdDuration`` if a timeline id is given, else ``Duration``."""
    if number_type is None:
        number_type = NumberType.from_number(value)
    if timeline_id is not None:
        return IdDuration(value, unit, timeline_id, number_type=number_type)
    return Duration(value, unit, number_type=number_type)


# ---------------------------------------------------------------------------
# Value projectors — denormalise ``value`` into 3 Arrow fields, drop ``unit``
# and ``timeline_id`` (they live in field metadata).
# ---------------------------------------------------------------------------


def _time_value_projector(
    _model_cls: type[BaseModel], _name: str, _info: object
) -> list[pa.Field[pa.DataType]]:
    """Project ``value`` onto the denormalised storage struct."""
    return list(RATIONAL_STRUCT_TYPE)


register_value_projector(TimeScalar, "value", _time_value_projector)
register_value_projector(TimeScalar, "unit", drop_from_schema)
register_value_projector(IdTimeScalar, "timeline_id", drop_from_schema)


# ═══════════════════════════════════════════════════════════════════════════
# Paired SemanticField classes (TimeScalarField hierarchy)
# ═══════════════════════════════════════════════════════════════════════════


def _scalar_compare_value(other: Any) -> Any:
    """Extract the numeric payload for ``pa.compute`` comparison helpers.

    Accepts a bare number (``int`` / ``float`` / ``Fraction``) or a
    ``TimeScalar`` and returns the underlying numeric value.  Mirrors
    ``TimeScalar._cmp_value`` but without unit / timeline-id checking —
    callers are responsible for ensuring compatibility (Field-level
    arithmetic does the metadata-level invariant check once, not per row).
    """
    if isinstance(other, TimeScalar):
        return float(other.value)
    if isinstance(other, Fraction):
        return float(other)
    if isinstance(other, (int, float)):
        return other
    raise TypeError(f"Cannot compare TimeScalarField with {type(other).__name__}")


def _coord_field_from_value(
    value_arr: pa.StructArray,
    unit: Any,
    number_type: Any,
    timeline_id: str | None,
) -> CoordinateField | IdCoordinateField:
    """Construct a CoordinateField (or IdCoordinateField) from a value column."""
    pa_field = pa.field(
        "value" if timeline_id is None else "id_value",
        value_arr.type,
    )
    raw = StructField(value_arr, pa_field)
    if timeline_id is not None:
        return IdCoordinateField(raw, unit, number_type, timeline_id)
    return CoordinateField(raw, unit, number_type)


def _dur_field_from_value(
    value_arr: pa.StructArray,
    unit: Any,
    number_type: Any,
    timeline_id: str | None,
) -> DurationField | IdDurationField:
    """Construct a DurationField (or IdDurationField) from a value column."""
    pa_field = pa.field(
        "value" if timeline_id is None else "id_value",
        value_arr.type,
    )
    raw = StructField(value_arr, pa_field)
    if timeline_id is not None:
        return IdDurationField(raw, unit, number_type, timeline_id)
    return DurationField(raw, unit, number_type)


_AxisFieldScalarT = TypeVar("_AxisFieldScalarT", bound=BaseModel)
"""The scalar of an :class:`_AxisField`."""


class _AxisField(SemanticField[_AxisFieldScalarT]):
    """A field whose numbers lie on one axis: one unit, one number type.

    The common parent of :class:`TimeScalarField` (one number per row) and
    :class:`SharedAxisField` (several numbers per row).  The axis is a
    property of the field, never of a row.  A field declared without data
    (``source_fields=``) is *unbound*: it names the column it will read, but
    the axis comes from the table it is read from, so until then
    :attr:`unit` and :attr:`domain` are ``None``, the field prints as
    unbound, and :meth:`metadata_dict` states no axis.
    """

    def __repr__(self) -> str:
        if self.unit is None:
            return f"{type(self).__name__}(name={self.name!r}, {self._unbound_description()})"
        return self._repr_bound()

    @property
    @abstractmethod
    def unit(self) -> TimeUnit | None:
        """The unit of every number; ``None`` while the field is unbound."""

    @property
    @abstractmethod
    def number_type(self) -> NumberType:
        """The number type every number is written in."""

    @property
    def domain(self) -> Domain | None:
        """The domain implied by the unit; ``None`` while the field is unbound."""
        unit = self.unit
        return None if unit is None else unit.domain

    def metadata_dict(self) -> dict[str, str]:
        """Return the discriminator plus the axis (unit, domain, number type).

        An unbound field has no axis yet, and states only the discriminator.
        """
        unit = self.unit
        if unit is None:
            return super().metadata_dict()
        return {
            **super().metadata_dict(),
            "unit": unit.value,
            "domain": unit.domain.value,
            "number_type": self.number_type.name,
        }

    def _unbound_description(self) -> str:
        """The one wording of an unbound field, for its repr, card and errors."""
        return f"unbound, source_fields={self._blueprint_source_fields!r}"

    def _unbound_error(self) -> TypeError:
        """The error raised where an unbound field is asked for its rows."""
        return TypeError(
            f"{type(self).__name__} {self.name!r} is {self._unbound_description()}; "
            "it has no axis until it is read from a table"
        )

    def _bound_unit(self) -> TimeUnit:
        """The unit, for a path that needs the field bound.

        Raises:
            TypeError: If the field is unbound.
        """
        unit = self.unit
        if unit is None:
            raise self._unbound_error()
        return unit

    @staticmethod
    def _resolve_metadata(
        pa_field: pa.Field[pa.DataType],
        unit_override: TimeUnit | str | None,
        nt_override: NumberType | str | None,
    ) -> tuple[TimeUnit | None, NumberType]:
        """Extract unit and number_type from a ``pa.Field``'s metadata."""
        meta: dict[str, Any] = {}
        raw_meta = pa_field.metadata
        if raw_meta:
            if TIMETOALIGN_METADATA_KEY in raw_meta:
                meta = parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
            else:
                meta = {
                    k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()
                }

        if unit_override is not None:
            resolved_unit = (
                TimeUnit(unit_override)
                if isinstance(unit_override, str)
                else unit_override
            )
        elif "unit" in meta:
            resolved_unit = TimeUnit(meta["unit"])
        else:
            resolved_unit = None  # caller decides if this is fatal

        if nt_override is not None:
            resolved_nt = (
                NumberType(nt_override) if isinstance(nt_override, str) else nt_override
            )
        elif "number_type" in meta:
            resolved_nt = NumberType(meta["number_type"])
        else:
            resolved_nt = NumberType.float

        return resolved_unit, resolved_nt

    def _repr_bound(self) -> str:
        """The repr of a bound field."""
        return super().__repr__()

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Add the ``Axis`` row to an unbound field's card."""
        rows = super()._repr_rows()
        if self.unit is None:
            import html as _html

            rows.insert(1, ("Axis", _html.escape(self._unbound_description())))
        return rows


class TimeScalarField(_AxisField[TimeScalarT]):
    """Abstract parent for ``Coordinate`` / ``Duration`` (+ Id) semantic fields.

    Provides the shared ``from_field`` / ``from_table`` machinery for
    ``CoordinateField`` and ``DurationField``; the axis accessors and the
    repr come from :class:`_AxisField`. Concrete
    subclasses pin the scalar (``TimeScalarField[Coordinate]``, which drives
    ``scalar_cls`` and ``pa_schema`` caching through
    ``SemanticField.__init_subclass__``), then optionally override
    ``semantic_type``, ``metadata_dict``, and ``__getitem__``.

    The inner raw field is a :class:`DenominateNumberField` (a
    ``RedundantNumberField`` with a single bound unit).  Construction accepts
    either a bare :class:`StructField` (promoted internally to a
    ``DenominateNumberField`` using the supplied ``unit``) or a fully
    formed ``DenominateNumberField``.

    Args:
        raw: A ``StructField`` (will be promoted) or a
            ``DenominateNumberField`` holding the denormalised
            ``{value, numerator, denominator}`` struct.
        unit: The time unit bound to this field.  Required when *raw*
            is a bare ``StructField`` or its unit cannot be inferred
            from metadata.
        number_type: The numeric representation used when materialising
            individual scalars (``NumberType.float`` /
            ``NumberType.fraction`` / ``NumberType.int``).
    """

    # Inner raw field type is fixed for the whole TimeScalarField branch.
    _raw_cls = DenominateNumberField  # type: ignore[assignment]

    # ``scalar_cls`` is intentionally left as ``None`` here — concrete
    # leaves pin it.

    def __init__(
        self,
        raw: object = None,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        *,
        source_fields: "str | dict[str, Any] | None" = None,
    ) -> None:
        if raw is None and source_fields is None:
            raise TypeError(
                f"{type(self).__name__} requires either a raw field (live) or "
                "source_fields= (blueprint)"
            )

        if raw is not None:
            # Promote a bare StructField to a DenominateNumberField; if
            # already a DenominateNumberField, optionally override unit.
            if not isinstance(raw, DenominateNumberField):
                if not isinstance(raw, StructField):
                    raise TypeError(
                        f"{type(self).__name__} requires StructField or "
                        f"DenominateNumberField, got {type(raw).__name__}"
                    )
                resolved_unit = self._require_unit(unit, type(self).__name__)
                raw = DenominateNumberField(raw.data, raw.field, unit=resolved_unit)
            elif unit is not None:
                # Override: rebuild with the supplied unit.
                raw = DenominateNumberField(
                    raw.data,
                    raw.field,
                    unit=TimeUnit(unit) if isinstance(unit, str) else unit,
                )

        super().__init__(raw, source_fields=source_fields)

        # Resolve number_type: explicit override > metadata > NumberType.float.
        if number_type is None and raw is not None:
            _, number_type = self._resolve_metadata(raw.field, None, None)
        elif number_type is None:
            number_type = NumberType.float
        self._number_type: NumberType = (
            NumberType(number_type) if isinstance(number_type, str) else number_type
        )

    # -- forwarded properties from the raw layer ----------------------------

    @property
    def unit(self) -> TimeUnit | None:
        """The time unit bound to this field (via the inner ``DenominateNumberField``).

        ``None`` while the field is unbound: a blueprint holds a plain
        ``StructField`` in its place until it is read from a table.
        """
        raw = self._raw
        return raw.unit if isinstance(raw, DenominateNumberField) else None

    @property
    def _unit(self) -> Any:
        """Back-compat alias for :attr:`unit`.

        Older callers (subclass methods, tests, serialisation helpers)
        reach into ``self._unit`` directly; keep the name available so
        we do not have to rewrite every reference at once.
        """
        return self.unit

    @property
    def number_type(self) -> Any:
        """The numeric representation used for scalar access."""
        return self._number_type

    # -- abstract semantic contract ----------------------------------------

    @property
    def semantic_type(self) -> str:
        """Name of the scalar this field materialises.

        The class name without ``Field``; concrete subclasses override it.
        """
        # Concrete subclasses override with their scalar's name.
        return type(self).__name__.removesuffix("Field")

    # -- materialisation ----------------------------------------------------

    def _materialise_value(self, i: int) -> TimeScalarValue | None:
        """Extract the i-th element's value in this field's representation.

        Raises:
            TypeError: If the field is unbound.
        """
        raw = self._raw
        if not isinstance(raw, DenominateNumberField):
            raise self._unbound_error()
        cell = raw.cell(i)
        if cell is None:
            return None
        return struct_to_coordinate(cell, self._number_type)

    def __getitem__(self, i: int) -> TimeScalarT | None:
        """Materialise the i-th scalar instance.

        Default implementation builds the paired scalar from
        ``(value, unit)``. Id-variants override to thread the field's
        ``_timeline_id`` into the result.
        """
        value = self._materialise_value(i)
        if value is None:
            return None
        scalar = type(self).paired_scalar()
        return scalar(value, self.unit, number_type=self._number_type)

    # -- unit / number_type resolution helpers ------------------------------

    @staticmethod
    def _require_unit(unit: Any, cls_name: str = "TimeScalarField") -> Any:
        if unit is None:
            raise ValueError(
                f"'unit' is required when constructing {cls_name} from a bare "
                "array or StructField"
            )
        return TimeUnit(unit) if isinstance(unit, str) else unit

    @staticmethod
    def _require_number_type(
        number_type: Any, cls_name: str = "TimeScalarField"
    ) -> Any:
        if number_type is None:
            raise ValueError(
                f"'number_type' is required when constructing {cls_name} from a "
                "bare array or StructField"
            )
        return NumberType(number_type) if isinstance(number_type, str) else number_type

    # -- factories ----------------------------------------------------------

    @classmethod
    def _from_field_impl(
        cls,
        source: Any,
        *,
        unit: TimeUnit | str | None,
        number_type: NumberType | str | None,
        name: str,
        extra_resolved: dict[str, Any] | None = None,
    ) -> Self:
        """Construct ``cls`` from one of the supported source shapes.

        ``extra_resolved`` is a hook for Id-subclasses to inject already-
        resolved kwargs (e.g. ``timeline_id``) into the final constructor.
        """
        extra_resolved = extra_resolved or {}

        if _is_tuple(source):
            data, pa_field = source
            resolved_unit, resolved_nt = cls._resolve_metadata(
                pa_field, unit, number_type
            )
            if resolved_unit is None:
                raise ValueError(
                    f"Cannot determine unit for field {pa_field.name!r}: no "
                    "'unit' in metadata and no override supplied"
                )
            struct_field = StructField(data, pa_field)
            return cls(struct_field, resolved_unit, resolved_nt, **extra_resolved)

        if _is_arrow_field(source):
            resolved_unit, resolved_nt = cls._resolve_metadata(
                source, unit, number_type
            )
            if resolved_unit is None:
                raise ValueError(
                    f"Cannot determine unit for field {source.name!r}: no "
                    "'unit' in metadata and no override supplied"
                )
            struct_field = StructField(None, source)
            return cls(struct_field, resolved_unit, resolved_nt, **extra_resolved)

        if isinstance(source, StructField):
            resolved_unit = cls._require_unit(unit, cls.__name__)
            resolved_nt = cls._require_number_type(number_type, cls.__name__)
            return cls(source, resolved_unit, resolved_nt, **extra_resolved)

        if _is_arrow_column(source):
            resolved_unit = cls._require_unit(unit, cls.__name__)
            resolved_nt = cls._require_number_type(number_type, cls.__name__)
            pa_field = pa.field(name, source.type)
            struct_field = StructField(source, pa_field)
            return cls(struct_field, resolved_unit, resolved_nt, **extra_resolved)

        raise TypeError(
            f"Unsupported source type for {cls.__name__}.from_field: "
            f"{type(source).__name__}"
        )

    @classmethod
    def from_field(
        cls,
        source: (
            pa.Array[Any]
            | pa.ChunkedArray[Any]
            | StructField
            | pa.Field[pa.DataType]
            | tuple[pa.Array[Any] | pa.ChunkedArray[Any] | None, pa.Field[pa.DataType]]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str | None = None,
        **kw: Any,
    ) -> Self:
        """Create a field from an Arrow column, a ``pa.Field`` or a ``StructField``.

        A ``pa.Field`` or a ``(data, field)`` tuple supplies the unit and
        number type through the field's metadata; *unit* and *number_type*
        override it. A bare array, chunked array or ``StructField`` carries no
        metadata, so both must be passed. A ``pa.Field`` alone gives a
        schema-only field.

        Args:
            source: A ``(pa.Array | None, pa.Field)`` tuple, a ``pa.Field``, a
                ``StructField``, or a bare ``pa.Array`` / ``pa.ChunkedArray``
                of number structs.
            unit: The time unit; overrides the metadata.
            number_type: The declared representation; overrides the
                metadata. A tuple or ``pa.Field`` source whose metadata names
                none declares ``float``.
            name: Column name given to a bare array; defaults to the class
                name without ``Field``, lower-cased.
            **kw: Unused keywords; ignored, with a warning naming them.

        Returns:
            A new field of this class.

        Raises:
            ValueError: If the unit (or, for a bare source, the number type)
                cannot be determined.
            TypeError: If *source* is none of the supported shapes.
        """
        cls._warn_ignored_keywords(kw)
        if name is None:
            name = cls.__name__.removesuffix("Field").lower() or "value"
        return cls._from_field_impl(
            source, unit=unit, number_type=number_type, name=name
        )

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        field: str | None = None,
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> Self:
        """Create a field from one column of a table.

        Args:
            table: The table holding the column.
            field: Name of the column; may be omitted when exactly one struct
                column carries a timetoalign metadata blob.
            unit: The time unit; overrides the column's metadata.
            number_type: The declared representation; overrides the column's
                metadata.

        Returns:
            A new field of this class over the column.

        Raises:
            ValueError: If *field* is omitted and no single candidate column
                is found, or the unit cannot be determined.
        """
        if field is None:
            schema_fields: list[pa.Field[pa.DataType]] = list(table.schema)
            candidates = [
                f.name
                for f in schema_fields
                if pa.types.is_struct(f.type)
                and f.metadata
                and TIMETOALIGN_METADATA_KEY in f.metadata
            ]
            if len(candidates) == 1:
                field = candidates[0]
            elif len(candidates) == 0:
                raise ValueError(
                    "No struct field carrying a timetoalign metadata blob found in "
                    "table; pass field= explicitly"
                )
            else:
                raise ValueError(
                    f"Multiple candidate fields found: {candidates}; "
                    "pass field= explicitly"
                )
        pa_field = table.field(field)
        data = table.column(field)
        return cls.from_field((data, pa_field), unit=unit, number_type=number_type)

    def with_unit(self, unit: TimeUnit) -> TimeScalarField[TimeScalarT]:
        """Return a same-class field with a different unit (no value conversion)."""
        return type(self)(self._raw, unit, self._number_type)

    # -- vectorized data-shaped mirrors -------------------------------------
    #
    # These mirror the @data_shaped methods on ``TimeScalar`` (declared in
    # scalar form earlier in the module).  Each is a ``pa.compute``
    # expression over the underlying ``value`` sub-field of the
    # denormalised rational struct, NEVER a per-row Python loop.

    def _value_array(self) -> pa.Array[Any]:
        """Return the underlying ``value`` sub-field with outer-struct nulls propagated.

        The inner ``value`` sub-field by itself ignores the outer
        ``{value, numerator, denominator}`` struct's null mask, so a
        naive ``arr.field("value")`` would yield ``0.0`` (or any inner
        value) at positions where the outer struct is null.  We fold
        the outer null mask in via :func:`pc.if_else`, so every
        data-shaped predicate / arithmetic / conversion mirror that
        consumes this method's output sees ``None`` at null positions
        and agrees element-wise with the scalar dispatch.
        """
        arr = _struct_array(self.to_pyarrow())
        value_arr = _struct_child(arr, "value")
        outer_valid = arr.is_valid()
        return pc.if_else(outer_valid, value_arr, pa.scalar(None, type=value_arr.type))

    def to_float(self) -> pa.Array[Any]:
        """Vectorized cast of the ``value`` sub-field to ``float64``."""
        return self._value_array().cast(pa.float64())

    def to_int(self, rounding: str = "round") -> pa.Array[Any]:
        """Vectorized cast to ``int64``; same modes and default as the scalar."""
        v = self._value_array()
        if rounding == "truncate":
            rounded = pc.trunc(v)
        elif rounding == "round":
            rounded = pc.round(v)
        elif rounding == "floor":
            rounded = pc.floor(v)
        elif rounding == "ceil":
            rounded = pc.ceil(v)
        else:
            raise ValueError(
                f"Unknown rounding mode: {rounding!r}. "
                f"Use 'truncate', 'round', 'floor', or 'ceil'."
            )
        return rounded.cast(pa.int64())

    def is_zero(self) -> pa.BooleanArray:
        """Vectorized ``pc.equal(value, 0)`` predicate."""
        return _compared("equal", self._value_array(), pa.scalar(0))

    def is_positive(self) -> pa.BooleanArray:
        """Vectorized ``pc.greater(value, 0)`` predicate."""
        return _compared("greater", self._value_array(), pa.scalar(0))

    def is_negative(self) -> pa.BooleanArray:
        """Vectorized ``pc.less(value, 0)`` predicate."""
        return _compared("less", self._value_array(), pa.scalar(0))

    # -- comparison mirrors -------------------------------------------------
    #
    # NB: Python ``__lt__`` / ``__le__`` / ``__gt__`` / ``__ge__`` MUST
    # return a bool — pa.Array cannot legally be returned from a dunder
    # without breaking ``sorted()``, ``list.sort()``, etc.  We therefore
    # split the surface: the parity-mirror dunders raise (see __init_subclass__
    # warning below), and the vectorized predicates live under explicit
    # names (``less_than``, ``less_equal``, etc.) plus a single ``compare``
    # entry-point.

    def less_than(self, other: TimeScalar | int | float | Fraction) -> pa.BooleanArray:
        """Vectorized ``pc.less(value, other)``."""
        return _compared("less", self._value_array(), _scalar_compare_value(other))

    def less_equal(self, other: TimeScalar | int | float | Fraction) -> pa.BooleanArray:
        """Vectorized ``pc.less_equal(value, other)``."""
        return _compared(
            "less_equal", self._value_array(), _scalar_compare_value(other)
        )

    def greater_than(
        self, other: TimeScalar | int | float | Fraction
    ) -> pa.BooleanArray:
        """Vectorized ``pc.greater(value, other)``."""
        return _compared("greater", self._value_array(), _scalar_compare_value(other))

    def greater_equal(
        self, other: TimeScalar | int | float | Fraction
    ) -> pa.BooleanArray:
        """Vectorized ``pc.greater_equal(value, other)``."""
        return _compared(
            "greater_equal", self._value_array(), _scalar_compare_value(other)
        )

    # Parity-mirror dunders.  Python protocol requires ``bool`` returns;
    # use the explicit ``less_than`` / ``less_equal`` / ``greater_than``
    # / ``greater_equal`` methods for vectorized comparisons.
    def __lt__(self, other: object) -> pa.BooleanArray:
        return self.less_than(other)  # type: ignore[arg-type]

    def __le__(self, other: object) -> pa.BooleanArray:
        return self.less_equal(other)  # type: ignore[arg-type]

    def __gt__(self, other: object) -> pa.BooleanArray:
        return self.greater_than(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> pa.BooleanArray:
        return self.greater_equal(other)  # type: ignore[arg-type]

    def _binop_operand(self, other: Any, op: str) -> tuple[Any, str | None]:
        """Extract the numeric payload and result timeline_id for arithmetic.

        Performs the metadata-level invariant check exactly once (unit match,
        cross-timeline-id mismatch) at the entry point of a Field arithmetic
        operation — NEVER per row.  Returns ``(numeric_other, result_tl_id)``.
        """
        self_tl = getattr(self, "_timeline_id", None)
        if isinstance(other, TimeScalar):
            if self._bound_unit() != other.unit:
                raise TypeError(
                    f"Cannot {op} {type(self).__name__} and "
                    f"{type(other).__name__} with different units: "
                    f"{self.unit} vs {other.unit}"
                )
            other_tl = getattr(other, "timeline_id", None)
            if self_tl is not None and other_tl is not None and self_tl != other_tl:
                raise TypeError(
                    f"Cannot {op} {type(self).__name__} and "
                    f"{type(other).__name__} with mismatched timeline_id: "
                    f"{self_tl!r} vs {other_tl!r}"
                )
            return other.value, self_tl or other_tl
        if _is_time_scalar_field(other):
            if self._bound_unit() != other._bound_unit():
                raise TypeError(
                    f"Cannot {op} {type(self).__name__} and "
                    f"{type(other).__name__} with different units: "
                    f"{self.unit} vs {other.unit}"
                )
            other_tl = getattr(other, "_timeline_id", None)
            if self_tl is not None and other_tl is not None and self_tl != other_tl:
                raise TypeError(
                    f"Cannot {op} {type(self).__name__} and "
                    f"{type(other).__name__} with mismatched timeline_id: "
                    f"{self_tl!r} vs {other_tl!r}"
                )
            return other._value_array(), self_tl or other_tl
        if isinstance(other, Fraction):
            return other, self_tl
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return other, self_tl
        raise TypeError(
            f"Cannot {op} {type(self).__name__} with {type(other).__name__}"
        )

    def _binop(
        self, other: Any, op_name: str, operation: str
    ) -> tuple[pa.StructArray, str | None]:
        """Check the metadata invariants once, then combine the columns.

        Unit and timeline-id compatibility are properties of the two *fields*,
        so they are settled here, before any data is touched — never re-tested
        per row. The arithmetic then runs in the left operand's declared
        representation, which is what decides the result's.
        """
        _, tl_id = self._binop_operand(other, op_name)
        if _is_time_scalar_field(other):
            operand: Any = other.to_pyarrow()
        elif isinstance(other, TimeScalar):
            operand = other.value
        else:
            operand = other
        return (
            combine_number_columns(
                self.to_pyarrow(), operand, operation, self._number_type
            ),
            tl_id,
        )

    def _repr_bound(self) -> str:
        length = len(self) if not self.is_empty else 0
        return (
            f"{type(self).__name__}(name={self.name!r}, unit={self._unit}, "
            f"number_type={self._number_type}, len={length})"
        )


def _is_time_scalar_field(value: object) -> TypeGuard[TimeScalarField[Any]]:
    return isinstance(value, TimeScalarField)


# ---------------------------------------------------------------------------
# Coordinate
# ---------------------------------------------------------------------------


class Coordinate(TimeScalar):
    """A position on a timeline, defined by a value and unit.

    Coordinates are immutable and support arithmetic operations when
    units match.  They preserve the native numeric type (int, float, or
    Fraction) for precision.

    Attributes:
        value: The numeric position (int, float, or Fraction).
        unit: The time unit (e.g., seconds, quarters, pixels).

    Examples:
        >>> c1 = Coordinate(120, TimeUnit.ticks)
        >>> c2 = Coordinate(240, TimeUnit.ticks)
        >>> c2 - c1
        Duration(120, ticks)
    """

    # Declared here, and passed on to the base, so that type checkers and
    # the API reference see this signature rather than the keyword-only one
    # derived for a pydantic model.
    def __init__(
        self,
        value: TimeScalarValue | dict[str, Any] | None = None,
        unit: TimeUnit | str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        super().__init__(
            value, unit, number_type=number_type, rounding=rounding, **data
        )

    def __index__(self) -> int:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return self.value
        raise TypeError(
            f"Cannot use Coordinate with {type(self.value).__name__} value as index. "
            f"Only integer Coordinates can be used as indices."
        )

    @property
    def semantic_type(self) -> str:
        """Canonical name of this type: ``"Coordinate"``."""
        return "Coordinate"

    # -- arithmetic ---------------------------------------------------------

    @data_shaped
    def __add__(self, other: object) -> Coordinate:
        if isinstance(other, Coordinate):
            raise TypeError(
                "Cannot add two Coordinates; subtract them to obtain a Duration"
            )
        value, tl_id = self._combine(other, "add", "add")
        return _make_coordinate(value, self.unit, tl_id)

    @overload
    def __sub__(self, other: Coordinate) -> Duration: ...

    @overload
    def __sub__(self, other: Duration | int | float | Fraction) -> Coordinate: ...

    @data_shaped
    def __sub__(self, other: object) -> TimeScalar:
        value, tl_id = self._combine(other, "subtract", "subtract")
        # Two positions differ by an extent; everything else stays a position.
        if isinstance(other, Coordinate):
            return _make_duration(value, self.unit, tl_id)
        return _make_coordinate(value, self.unit, tl_id)

    @data_shaped
    def __mul__(self, scalar: object) -> Coordinate:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return _make_coordinate(
            self._scale(scalar, "multiply", "multiply", "*"),
            self.unit,
            self._id_or_none(),
        )

    @data_shaped
    def __rmul__(self, scalar: object) -> Coordinate:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return self.__mul__(scalar)

    @data_shaped
    def __truediv__(self, scalar: object) -> Coordinate:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return _make_coordinate(
            self._scale(scalar, "divide", "divide", "/"),
            self.unit,
            self._id_or_none(),
        )

    @data_shaped
    def __floordiv__(self, scalar: object) -> Coordinate:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return _make_coordinate(
            self._scale(scalar, "floor-divide", "floor_divide", "//"),
            self.unit,
            self._id_or_none(),
        )

    # -- copy-on-write ------------------------------------------------------

    def with_timeline(self, timeline_id: str) -> IdCoordinate:
        """Return an ``IdCoordinate`` carrying the given timeline id."""
        return IdCoordinate(self.value, self.unit, timeline_id)

    # -- formatting ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"Coordinate({self.value!r}, {self.unit})"

    def __str__(self) -> str:
        return f"{self._format_value()} {self.unit}"


# ---------------------------------------------------------------------------
# CoordinateField
# ---------------------------------------------------------------------------


class CoordinateField(TimeScalarField[Coordinate]):
    """Semantic field for coordinate fields.

    Wraps a ``StructField`` containing the coordinate struct
    ``{value: float64, numerator: int64, denominator: int64}`` and adds
    semantic identity: unit, domain, number_type.

    Pairs with :class:`Coordinate`.

    Examples:
        >>> import pyarrow as pa
        >>> from timetoalign.core.enums import TimeUnit, NumberType
        >>> from timetoalign.core.time import CoordinateField
        >>> arr = pa.array(
        ...     [{"value": 1.5, "numerator": 3, "denominator": 2}],
        ...     type=pa.struct([
        ...         pa.field("value", pa.float64()),
        ...         pa.field("numerator", pa.int64()),
        ...         pa.field("denominator", pa.int64()),
        ...     ]),
        ... )
        >>> cf = CoordinateField.from_field(arr, unit=TimeUnit.seconds, number_type=NumberType.float)
        >>> cf[0]
        Coordinate(1.5, seconds)
    """

    @property
    def semantic_type(self) -> str:
        """Name of the scalar this field materialises: ``"Coordinate"``."""
        return "Coordinate"

    def __getitem__(self, i: int) -> Coordinate | None:
        value = self._materialise_value(i)
        if value is None:
            return None
        return Coordinate(value, self._unit, number_type=self._number_type)

    # -- vectorized arithmetic mirrors --------------------------------------
    #
    # Mirrors ``Coordinate.__add__`` etc.  Per-row ``pa.compute`` over the
    # ``value`` sub-field; metadata-level invariants (unit match, timeline
    # id) are checked once per call, never per row.

    def __add__(self, other: object) -> CoordinateField | DurationField:
        if isinstance(other, (Coordinate, CoordinateField)):
            raise TypeError(
                "Cannot add two Coordinate fields; subtract them to obtain a Duration"
            )
        struct, tl = self._binop(other, "add", "add")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    @overload
    def __sub__(self, other: Coordinate | CoordinateField) -> DurationField: ...

    @overload
    def __sub__(self, other: object) -> CoordinateField | DurationField: ...

    def __sub__(self, other: object) -> CoordinateField | DurationField:
        struct, tl = self._binop(other, "subtract", "subtract")
        # Two positions differ by an extent; everything else stays a position.
        if isinstance(other, (Coordinate, CoordinateField)):
            return _dur_field_from_value(struct, self._unit, self._number_type, tl)
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __mul__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = self._binop(scalar, "multiply", "multiply")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __rmul__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = self._binop(scalar, "divide", "divide")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __floordiv__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = self._binop(scalar, "floor_divide", "floor_divide")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field[pa.DataType]) -> bool:
        """Reject ``IdCoordinateField`` shapes — they live in a sibling class.

        The base ``SemanticField.matches_pa_field`` is shape-OR-metadata.
        Since ``IdCoordinateField`` has the same struct shape, this method
        explicitly rejects ``pa.Field`` whose metadata advertises
        ``"IdCoordinateField"``; structural shape matches still resolve to
        ``CoordinateField`` for backward compatibility with non-Id fields.
        """
        # Inspect metadata first.  If the metadata blob explicitly says
        # this is an IdCoordinateField, this class does NOT match (its
        # sibling will).
        if (
            pa_field.metadata is not None
            and TIMETOALIGN_METADATA_KEY in pa_field.metadata
        ):
            try:
                meta = parse_metadata_blob(pa_field.metadata[TIMETOALIGN_METADATA_KEY])
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                meta = {}
            if meta:
                ft = meta.get("field_type")
                if ft == "IdCoordinateField":
                    return False
                if ft == cls.__name__:
                    return True
        return super().matches_pa_field(pa_field)


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


class Duration(TimeScalar):
    """An elapsed extent on a timeline.

    Same physical storage as :class:`Coordinate` (denormalised
    ``{value, numerator, denominator}``) — the distinction is semantic.
    Durations may be negative when produced by ``Coordinate - Coordinate``
    where the second operand precedes the first; status predicates
    (:meth:`is_negative`, :meth:`is_positive`) make the sign queryable.

    Attributes:
        value: The numeric duration (int, float, or Fraction; any sign).
        unit: The time unit (e.g., seconds, quarters, ticks).
    """

    # Declared here, and passed on to the base, so that type checkers and
    # the API reference see this signature rather than the keyword-only one
    # derived for a pydantic model.
    def __init__(
        self,
        value: TimeScalarValue | dict[str, Any] | None = None,
        unit: TimeUnit | str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        super().__init__(
            value, unit, number_type=number_type, rounding=rounding, **data
        )

    @property
    def semantic_type(self) -> str:
        """Canonical name of this type: ``"Duration"``."""
        return "Duration"

    # -- arithmetic ---------------------------------------------------------

    @data_shaped
    def __add__(self, other: object) -> Duration:
        if isinstance(other, Coordinate):
            raise TypeError(
                "Cannot add a Coordinate to a Duration; use 'coord + dur' instead"
            )
        value, tl_id = self._combine(other, "add", "add")
        return _make_duration(value, self.unit, tl_id)

    @data_shaped
    def __sub__(self, other: object) -> Duration:
        if isinstance(other, Coordinate):
            raise TypeError("Cannot subtract a Coordinate from a Duration")
        value, tl_id = self._combine(other, "subtract", "subtract")
        return _make_duration(value, self.unit, tl_id)

    @data_shaped
    def __mul__(self, scalar: object) -> Duration:
        return _make_duration(
            self._scale(scalar, "multiply", "multiply", "*"),
            self.unit,
            self._id_or_none(),
        )

    @data_shaped
    def __rmul__(self, scalar: object) -> Duration:
        return self.__mul__(scalar)

    @data_shaped
    def __truediv__(self, scalar: object) -> Duration:
        return _make_duration(
            self._scale(scalar, "divide", "divide", "/"),
            self.unit,
            self._id_or_none(),
        )

    @data_shaped
    def __floordiv__(self, scalar: object) -> Duration:
        return _make_duration(
            self._scale(scalar, "floor-divide", "floor_divide", "//"),
            self.unit,
            self._id_or_none(),
        )

    # -- copy-on-write ------------------------------------------------------

    def with_timeline(self, timeline_id: str) -> IdDuration:
        """Return an ``IdDuration`` carrying the given timeline id."""
        return IdDuration(self.value, self.unit, timeline_id)

    # -- formatting ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"Duration({self.value!r}, {self.unit})"

    def __str__(self) -> str:
        return f"{self._format_value()} {self.unit}"


# ---------------------------------------------------------------------------
# DurationField
# ---------------------------------------------------------------------------


class DurationField(TimeScalarField[Duration]):
    """Semantic field for duration fields.

    Uses the same coordinate struct ``{value, numerator, denominator}``
    as :class:`CoordinateField`; the distinction is semantic.
    Pairs with :class:`Duration`.
    """

    @property
    def semantic_type(self) -> str:
        """Name of the scalar this field materialises: ``"Duration"``."""
        return "Duration"

    def __getitem__(self, i: int) -> Duration | None:
        value = self._materialise_value(i)
        if value is None:
            return None
        return Duration(value, self._unit, number_type=self._number_type)

    # -- vectorized arithmetic mirrors --------------------------------------

    def __add__(self, other: object) -> DurationField:
        if isinstance(other, (Coordinate, CoordinateField)):
            raise TypeError(
                "Cannot add a Coordinate to a Duration field; "
                "use 'coord + dur' instead"
            )
        struct, tl = self._binop(other, "add", "add")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __sub__(self, other: object) -> DurationField:
        if isinstance(other, (Coordinate, CoordinateField)):
            raise TypeError("Cannot subtract a Coordinate from a Duration field")
        struct, tl = self._binop(other, "subtract", "subtract")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __mul__(self, scalar: object) -> DurationField:
        struct, tl = self._binop(scalar, "multiply", "multiply")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __rmul__(self, scalar: object) -> DurationField:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> DurationField:
        struct, tl = self._binop(scalar, "divide", "divide")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __floordiv__(self, scalar: object) -> DurationField:
        struct, tl = self._binop(scalar, "floor_divide", "floor_divide")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field[pa.DataType]) -> bool:
        """Reject ``IdDurationField`` shapes; otherwise defer to the base."""
        if (
            pa_field.metadata is not None
            and TIMETOALIGN_METADATA_KEY in pa_field.metadata
        ):
            try:
                meta = parse_metadata_blob(pa_field.metadata[TIMETOALIGN_METADATA_KEY])
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                meta = {}
            if meta:
                ft = meta.get("field_type")
                if ft == "IdDurationField":
                    return False
                if ft == cls.__name__:
                    return True
        return super().matches_pa_field(pa_field)


# ---------------------------------------------------------------------------
# IdCoordinate — diamond MRO: IdCoordinate -> Coordinate -> IdTimeScalar
#                            -> TimeScalar -> BaseModel
# ---------------------------------------------------------------------------


class IdCoordinate(Coordinate, IdTimeScalar):
    """A ``Coordinate`` that carries the ID of the timeline it belongs to.

    Extends ``Coordinate`` with a ``timeline_id`` field, providing the
    most specific form of coordinate specification.  Operator results
    inherit the timeline_id (``IdCoordinate + Duration → IdCoordinate``;
    ``IdCoordinate - IdCoordinate → IdDuration`` when the ids match).
    """

    # An IdCoordinate is stored by CoordinateField (the timeline id lives
    # in field metadata, not in the row), so the discriminator is the
    # plain coordinate one rather than the derived "IdCoordinateField".
    field_type_name = "CoordinateField"

    # Passed on to IdTimeScalar by name: the plain base precedes it in the
    # MRO and would otherwise supply its constructor without a timeline id.
    def __init__(
        self,
        value: TimeScalarValue | None = None,
        unit: TimeUnit | str | None = None,
        timeline_id: str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        IdTimeScalar.__init__(
            self,
            value,
            unit,
            timeline_id,
            number_type=number_type,
            rounding=rounding,
            **data,
        )

    @classmethod
    def from_coordinate(cls, coord: Coordinate, timeline_id: str) -> IdCoordinate:
        """Bind *coord*'s value and unit to the timeline *timeline_id*."""
        return cls(coord.value, coord.unit, timeline_id)

    def to_coordinate(self) -> Coordinate:
        """Return this value and unit as a plain :class:`Coordinate`."""
        return Coordinate(self.value, self.unit)

    def with_timeline(self, timeline_id: str) -> IdCoordinate:
        """Return this value and unit bound to the timeline *timeline_id* instead."""
        return IdCoordinate(self.value, self.unit, timeline_id)

    def __repr__(self) -> str:
        return f"IdCoordinate({self.value!r}, {self.unit}, {self.timeline_id!r})"

    def __str__(self) -> str:
        return f"{self._format_value()} {self.unit} @{self.timeline_id}"


# ---------------------------------------------------------------------------
# IdCoordinateField — carries timeline_id, materialises IdCoordinate
# ---------------------------------------------------------------------------


class IdCoordinateField(CoordinateField):
    """Coordinate field annotated with a ``timeline_id``.

    On-disk struct shape is identical to :class:`CoordinateField`; the
    timeline id lives in field metadata (the ``TIMETOALIGN_METADATA_KEY`` blob)
    and on the live instance.  Materialised scalars are
    :class:`IdCoordinate`.
    """

    scalar_cls = IdCoordinate

    def __init__(
        self,
        raw: StructField,
        unit: Any,
        number_type: Any,
        timeline_id: object,
    ) -> None:
        super().__init__(raw, unit, number_type)
        if not isinstance(timeline_id, str) or not timeline_id:
            raise ValueError(
                f"IdCoordinateField requires a non-empty timeline_id string; "
                f"got {timeline_id!r}"
            )
        self._timeline_id = timeline_id

    @property
    def timeline_id(self) -> str:
        """The ID of the timeline every element of this field belongs to."""
        return self._timeline_id

    @property
    def semantic_type(self) -> str:
        """Name of the scalar this field materialises: ``"IdCoordinate"``."""
        return "IdCoordinate"

    def metadata_dict(self) -> dict[str, str]:
        """Extend the inherited payload with this field's bound timeline id."""
        return {**super().metadata_dict(), "timeline_id": self._timeline_id}

    def __getitem__(self, i: int) -> IdCoordinate | None:
        value = self._materialise_value(i)
        if value is None:
            return None
        return IdCoordinate(
            value, self._unit, self._timeline_id, number_type=self._number_type
        )

    @classmethod
    def from_field(
        cls,
        source: (
            pa.Array[Any]
            | pa.ChunkedArray[Any]
            | StructField
            | pa.Field[pa.DataType]
            | tuple[pa.Array[Any] | pa.ChunkedArray[Any] | None, pa.Field[pa.DataType]]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        timeline_id: str | None = None,
        name: str | None = None,
        **kw: Any,
    ) -> IdCoordinateField:
        """Create a field bound to one timeline; see :meth:`TimeScalarField.from_field`.

        Args:
            source: As for :meth:`TimeScalarField.from_field`.
            unit: The time unit; overrides the metadata.
            number_type: The declared representation; overrides the metadata.
            timeline_id: The timeline the field belongs to; overrides the
                ``timeline_id`` in the field's timetoalign metadata blob.
            name: Column name given to a bare array; defaults to
                ``"id_coordinate"``.
            **kw: Unused keywords; ignored, with a warning naming them.

        Returns:
            A new IdCoordinateField.

        Raises:
            ValueError: If no timeline id is passed or found in the metadata,
                or the unit (or number type) cannot be determined.
        """
        cls._warn_ignored_keywords(kw)
        if name is None:
            name = "id_coordinate"
        resolved_tl_id = _resolve_timeline_id(source, timeline_id)
        return cls._from_field_impl(
            source,
            unit=unit,
            number_type=number_type,
            name=name,
            extra_resolved={"timeline_id": resolved_tl_id},
        )

    def with_unit(self, unit: TimeUnit) -> IdCoordinateField:
        """Return a field over the same data with *unit* (no value conversion)."""
        return IdCoordinateField(self._raw, unit, self._number_type, self._timeline_id)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field[pa.DataType]) -> bool:
        """Require ``field_type == "IdCoordinateField"`` in the metadata blob.

        Id-variants share their struct shape with their non-Id parent, so
        pure structural matching cannot distinguish them.  The
        ``TIMETOALIGN_METADATA_KEY`` JSON blob's ``field_type`` key is the
        authoritative discriminator (already injected by ``metadata_dict``
        at the SemanticField boundary).
        """
        if (
            pa_field.metadata is None
            or TIMETOALIGN_METADATA_KEY not in pa_field.metadata
        ):
            return False
        try:
            meta = parse_metadata_blob(pa_field.metadata[TIMETOALIGN_METADATA_KEY])
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return False
        return meta.get("field_type") == "IdCoordinateField"


# ---------------------------------------------------------------------------
# IdDuration — diamond MRO: IdDuration -> Duration -> IdTimeScalar
#                          -> TimeScalar -> BaseModel
# ---------------------------------------------------------------------------


class IdDuration(Duration, IdTimeScalar):
    """A ``Duration`` that carries the ID of the timeline it belongs to.

    Produced by ``Coordinate - Coordinate`` (with at least one operand
    being an ``IdCoordinate``) and ``IdDuration ± Duration``.
    """

    # Mirrors IdCoordinate: stored by DurationField, with the timeline id
    # carried in field metadata rather than derived into the name.
    field_type_name = "DurationField"

    # Passed on to IdTimeScalar by name: the plain base precedes it in the
    # MRO and would otherwise supply its constructor without a timeline id.
    def __init__(
        self,
        value: TimeScalarValue | None = None,
        unit: TimeUnit | str | None = None,
        timeline_id: str | None = None,
        /,
        *,
        number_type: NumberType | str | None = None,
        rounding: str = "round",
        **data: Any,
    ) -> None:
        IdTimeScalar.__init__(
            self,
            value,
            unit,
            timeline_id,
            number_type=number_type,
            rounding=rounding,
            **data,
        )

    @classmethod
    def from_duration(cls, dur: Duration, timeline_id: str) -> IdDuration:
        """Bind *dur*'s value and unit to the timeline *timeline_id*."""
        return cls(dur.value, dur.unit, timeline_id)

    def to_duration(self) -> Duration:
        """Return this value and unit as a plain :class:`Duration`."""
        return Duration(self.value, self.unit)

    def with_timeline(self, timeline_id: str) -> IdDuration:
        """Return this value and unit bound to the timeline *timeline_id* instead."""
        return IdDuration(self.value, self.unit, timeline_id)

    def __repr__(self) -> str:
        return f"IdDuration({self.value!r}, {self.unit}, {self.timeline_id!r})"

    def __str__(self) -> str:
        return f"{self._format_value()} {self.unit} @{self.timeline_id}"


# ---------------------------------------------------------------------------
# IdDurationField — carries timeline_id, materialises IdDuration
# ---------------------------------------------------------------------------


class IdDurationField(DurationField):
    """Duration field annotated with a ``timeline_id``.

    On-disk struct shape is identical to :class:`DurationField`; the
    timeline id lives in field metadata.  Materialised scalars are
    :class:`IdDuration`.
    """

    scalar_cls = IdDuration

    def __init__(
        self,
        raw: StructField,
        unit: Any,
        number_type: Any,
        timeline_id: object,
    ) -> None:
        super().__init__(raw, unit, number_type)
        if not isinstance(timeline_id, str) or not timeline_id:
            raise ValueError(
                f"IdDurationField requires a non-empty timeline_id string; "
                f"got {timeline_id!r}"
            )
        self._timeline_id = timeline_id

    @property
    def timeline_id(self) -> str:
        """The ID of the timeline every element of this field belongs to."""
        return self._timeline_id

    @property
    def semantic_type(self) -> str:
        """Name of the scalar this field materialises: ``"IdDuration"``."""
        return "IdDuration"

    def metadata_dict(self) -> dict[str, str]:
        """Extend the inherited payload with this field's bound timeline id."""
        return {**super().metadata_dict(), "timeline_id": self._timeline_id}

    def __getitem__(self, i: int) -> IdDuration | None:
        value = self._materialise_value(i)
        if value is None:
            return None
        return IdDuration(
            value, self._unit, self._timeline_id, number_type=self._number_type
        )

    @classmethod
    def from_field(
        cls,
        source: (
            pa.Array[Any]
            | pa.ChunkedArray[Any]
            | StructField
            | pa.Field[pa.DataType]
            | tuple[pa.Array[Any] | pa.ChunkedArray[Any] | None, pa.Field[pa.DataType]]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        timeline_id: str | None = None,
        name: str | None = None,
        **kw: Any,
    ) -> IdDurationField:
        """Create a field bound to one timeline; see :meth:`TimeScalarField.from_field`.

        Args:
            source: As for :meth:`TimeScalarField.from_field`.
            unit: The time unit; overrides the metadata.
            number_type: The declared representation; overrides the metadata.
            timeline_id: The timeline the field belongs to; overrides the
                ``timeline_id`` in the field's timetoalign metadata blob.
            name: Column name given to a bare array; defaults to
                ``"id_duration"``.
            **kw: Unused keywords; ignored, with a warning naming them.

        Returns:
            A new IdDurationField.

        Raises:
            ValueError: If no timeline id is passed or found in the metadata,
                or the unit (or number type) cannot be determined.
        """
        cls._warn_ignored_keywords(kw)
        if name is None:
            name = "id_duration"
        resolved_tl_id = _resolve_timeline_id(source, timeline_id)
        return cls._from_field_impl(
            source,
            unit=unit,
            number_type=number_type,
            name=name,
            extra_resolved={"timeline_id": resolved_tl_id},
        )

    def with_unit(self, unit: TimeUnit) -> IdDurationField:
        """Return a field over the same data with *unit* (no value conversion)."""
        return IdDurationField(self._raw, unit, self._number_type, self._timeline_id)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field[pa.DataType]) -> bool:
        """Require ``field_type == "IdDurationField"`` in the metadata blob."""
        if (
            pa_field.metadata is None
            or TIMETOALIGN_METADATA_KEY not in pa_field.metadata
        ):
            return False
        try:
            meta = parse_metadata_blob(pa_field.metadata[TIMETOALIGN_METADATA_KEY])
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return False
        return meta.get("field_type") == "IdDurationField"


# ---------------------------------------------------------------------------
# Type aliases used by callers
# ---------------------------------------------------------------------------

# CoordinateSpec: any form of coordinate specification.
CoordinateSpec = Union[int, float, Fraction, Coordinate, IdCoordinate]


class ResolvedCoordinate(NamedTuple):
    """Decomposed form of a CoordinateSpec input."""

    value: int | float | Fraction
    timeline_id: str | None
    unit: TimeUnit | None


def resolve_coordinate_spec(
    spec: CoordinateSpec, *, timeline_id: str | None = None
) -> ResolvedCoordinate:
    """Decompose a coordinate specification without converting its unit.

    Args:
        spec: Coordinate specification to decompose.
        timeline_id: Optional timeline ID for otherwise unqualified input.

    Returns:
        The numeric value and any timeline or unit metadata.

    Raises:
        ValueError: If an explicit timeline ID conflicts with an IdCoordinate.
        TypeError: If spec is not a supported coordinate specification.
    """
    if isinstance(spec, IdCoordinate):
        if timeline_id is not None and timeline_id != spec.timeline_id:
            raise ValueError(
                f"Timeline ID '{timeline_id}' conflicts with coordinate timeline ID "
                f"'{spec.timeline_id}'"
            )
        return ResolvedCoordinate(spec.value, spec.timeline_id, spec.unit)
    if isinstance(spec, Coordinate):
        return ResolvedCoordinate(spec.value, timeline_id, spec.unit)
    if not isinstance(spec, bool) and isinstance(spec, (int, float, Fraction)):
        return ResolvedCoordinate(spec, timeline_id, None)
    raise TypeError(f"Unsupported coordinate specification type: {type(spec).__name__}")


# Optional coordinate (common pattern).
OptionalCoordinate = Union[Coordinate, None]


# ---------------------------------------------------------------------------
# Coordinate cell: a number with its own unit and number type
# ---------------------------------------------------------------------------

COORDINATE_CELL_TYPE: pa.StructType = pa.struct(
    [
        *_member_fields(RATIONAL_STRUCT_TYPE),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("number_type", pa.string(), nullable=True),
    ]
)
"""One coordinate (or duration) as a table cell that names its own axis.

The number struct of :data:`RATIONAL_STRUCT_TYPE`, both sides populated on
every non-null cell (a float mirrored by its exact ratio), followed by the
``unit`` (the :class:`TimeUnit` value) and the ``number_type`` (``"int"``,
``"float"`` or ``"fraction"``), which says which side is authoritative. It
is the cell of a table whose rows may lie on different timelines, such as
the MatchClaim table and the Projections table in memory, where no field
can record one axis for all rows. A package's files write the number once
instead (:data:`PACKAGE_COORDINATE_CELL_TYPE`, :data:`PACKAGE_NUMBER_TYPE`).
"""


def coordinate_cell_projector(
    _model_cls: type[BaseModel], name: str, _info: object
) -> list[pa.Field[pa.DataType]]:
    """Store a scalar's coordinate field as one :data:`COORDINATE_CELL_TYPE` cell.

    The value projector (``core.fields.register_value_projector``) of every
    coordinate that names its own axis in a table whose rows may lie on
    different timelines: the coordinates of a MatchClaim's anchors, a
    Projection's ``start`` and ``duration``.
    """
    return [pa.field(name, COORDINATE_CELL_TYPE, nullable=True)]


def axis_coordinate_cells(
    values: Any,
    *,
    unit: TimeUnit | str,
    number_type: NumberType | str | None = None,
    rounding: str = "round",
) -> pa.StructArray:
    """Build the coordinate cells of a column of numbers on one axis.

    The numbers go through :func:`build_number_struct_array` (vectorized for
    a plain numeric column and for a column of number structs), written in
    *number_type*; every non-null cell names *unit* and *number_type*.

    Args:
        values: One number per row, ``None`` for a null cell (any column
            :func:`build_number_struct_array` reads).
        unit: The axis's unit.
        number_type: The axis's number type; the unit's default when omitted.
        rounding: How an inexact value is made integral on an ``int`` axis.

    Returns:
        A ``pa.StructArray`` of :data:`COORDINATE_CELL_TYPE`.

    Examples:
        >>> axis_coordinate_cells([3, None], unit="samples").to_pylist()[0]
        {'value': 3.0, 'numerator': 3, 'denominator': 1, 'unit': 'samples', 'number_type': 'int'}
    """
    unit = TimeUnit(unit)
    number_type = unit.resolve_number_type(
        None if number_type is None else NumberType(number_type)
    )
    numbers = build_number_struct_array(
        values, number_type=number_type, rounding=rounding
    )
    valid = numbers.is_valid()
    n = len(numbers)
    return _struct_array_from(
        [
            _struct_child(numbers, "value"),
            _struct_child(numbers, "numerator"),
            _struct_child(numbers, "denominator"),
            pc.if_else(valid, pa.scalar(unit.value), pa.nulls(n, pa.string())),
            pc.if_else(valid, pa.scalar(number_type.name), pa.nulls(n, pa.string())),
        ],
        fields=_member_fields(COORDINATE_CELL_TYPE),
        mask=pc.invert(valid),
    )


def coordinate_cells(coordinates: Iterable[TimeScalar | None]) -> pa.StructArray:
    """Build the coordinate cells of *coordinates*, each on its own axis.

    Each coordinate keeps its unit and number type; the rows of one axis are
    built together by :func:`axis_coordinate_cells`, and the row order is
    kept. A :class:`Duration` makes a cell like a :class:`Coordinate`.

    Args:
        coordinates: One coordinate (or duration) per row, ``None`` for a
            null cell.

    Returns:
        A ``pa.StructArray`` of :data:`COORDINATE_CELL_TYPE`.

    Examples:
        >>> cells = coordinate_cells(
        ...     [Coordinate(Fraction(1, 3), "quarters"), None, Coordinate(0.5, "seconds")]
        ... )
        >>> [cell and (cell["numerator"], cell["denominator"]) for cell in cells.to_pylist()]
        [(1, 3), None, (1, 2)]
    """
    items = list(coordinates)
    rows_of_axis: dict[tuple[TimeUnit, NumberType], list[tuple[int, TimeScalar]]] = {}
    for index, coordinate in enumerate(items):
        if coordinate is not None:
            axis = (coordinate.unit, coordinate.number_type)
            rows_of_axis.setdefault(axis, []).append((index, coordinate))
    order = [index for rows in rows_of_axis.values() for index, _ in rows]
    nulls = [index for index, coordinate in enumerate(items) if coordinate is None]
    parts = [
        axis_coordinate_cells(
            [coordinate.value for _, coordinate in rows],
            unit=unit,
            number_type=number_type,
        )
        for (unit, number_type), rows in rows_of_axis.items()
    ]
    parts.append(pa.nulls(len(nulls), COORDINATE_CELL_TYPE))
    cells = pa.concat_arrays(parts)
    position = np.empty(len(items), dtype=np.int64)
    position[np.asarray(order + nulls, dtype=np.int64)] = np.arange(
        len(items), dtype=np.int64
    )
    return cells.take(position)


@overload
def cell_coordinate(cell: dict[str, Any]) -> Coordinate: ...


@overload
def cell_coordinate(cell: dict[str, Any], scalar: type[TimeScalarT]) -> TimeScalarT: ...


@overload
def cell_coordinate(cell: dict[str, Any] | None) -> Coordinate | None: ...


@overload
def cell_coordinate(
    cell: dict[str, Any] | None, scalar: type[TimeScalarT]
) -> TimeScalarT | None: ...


def cell_coordinate(
    cell: dict[str, Any] | None,
    scalar: type[TimeScalar] = Coordinate,
) -> TimeScalar | None:
    """Read the coordinate a coordinate cell holds, on the axis it names.

    The side the cell's ``number_type`` names is read, so an exact value
    stays exact and a float stays the float written.

    Args:
        cell: A cell of :data:`COORDINATE_CELL_TYPE` as a dict, or ``None``.
        scalar: The scalar to make, :class:`Coordinate` or :class:`Duration`.

    Returns:
        The scalar, or ``None`` for a null cell.

    Raises:
        ValueError: If the cell names no unit or number type, or its number
            cannot be read in its number type.

    Examples:
        >>> cell_coordinate(
        ...     {"value": 0.5, "numerator": 1, "denominator": 2,
        ...      "unit": "quarters", "number_type": "fraction"}
        ... )
        Coordinate(Fraction(1, 2), quarters)
    """
    if cell is None:
        return None
    if cell.get("unit") is None or cell.get("number_type") is None:
        raise ValueError(f"The coordinate cell {cell} names no unit or number type")
    number_type = NumberType(cell["number_type"])
    return scalar(
        struct_to_coordinate(cell, number_type),
        TimeUnit(cell["unit"]),
        number_type=number_type,
    )


# ---------------------------------------------------------------------------
# The package form: a number written once, by its number type
# ---------------------------------------------------------------------------

PACKAGE_NUMBER_TYPE: pa.StructType = pa.struct(
    [
        pa.field("numerator", pa.int64(), nullable=True),
        pa.field("denominator", pa.int64(), nullable=True),
        pa.field("value", pa.float64(), nullable=True),
    ]
)
"""A number as a package's tables write it: once, never mirrored.

The member its number type names holds it, the others are null:

============  =============  ===============  =========
number type   ``numerator``  ``denominator``  ``value``
============  =============  ===============  =========
``int``       n              null             null
``fraction``  n              d ≥ 1            null
``float``     null           null             x (finite)
============  =============  ===============  =========

The number type is not in the struct: a Projections table states it once
per row, a MatchClaim table's coordinate cell beside the number
(:data:`PACKAGE_COORDINATE_CELL_TYPE`). In memory every number keeps both
sides (:data:`RATIONAL_STRUCT_TYPE`); :func:`package_numbers` and
:func:`numbers_from_package` convert at the file boundary, and nowhere else.
"""

PACKAGE_COORDINATE_CELL_TYPE: pa.StructType = pa.struct(
    [
        *_member_fields(PACKAGE_NUMBER_TYPE),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("number_type", pa.string(), nullable=True),
    ]
)
"""A coordinate cell as a package's MatchClaim table writes it.

The number of :data:`PACKAGE_NUMBER_TYPE`, then the ``unit`` and the
``number_type`` of the coordinate: :data:`COORDINATE_CELL_TYPE` in the
package form (:func:`package_cells`, :func:`cells_from_package`).
"""

_NUMBER_MEMBERS = ("numerator", "denominator", "value")


def _members(array: Any, members: Sequence[str], what: str) -> dict[str, pa.Array[Any]]:
    """The members of a struct column by name, nulls of a null cell applied.

    Raises:
        ValueError: If *array* is not a struct holding exactly *members*.
    """
    if _is_chunked(array):
        array = array.combine_chunks()
    names = list(array.type.names) if pa.types.is_struct(array.type) else []
    if sorted(names) != sorted(members):
        raise ValueError(
            f"{what} is not a struct of the members {list(members)}: its type is "
            f"{array.type}"
        )
    # flatten() applies the slice offset and nulls every member of a null cell.
    return dict(zip(names, array.flatten(), strict=True))


#: The largest magnitude below which every integer is a double exactly.
_EXACT_DOUBLE_INT = 2**53


def _is(number_types: pa.Array[Any], name: str) -> pa.BooleanArray:
    """Mask: the rows whose number type is *name* (a null type is no type)."""
    return _filled_mask(
        _compared("equal", number_types, pa.scalar(name)), pa.scalar(False)
    )


def package_numbers(numbers: Any, number_types: Any) -> pa.StructArray:
    """Write a column of numbers in the package form, each by its number type.

    The encoder of the pair (:func:`numbers_from_package` reads): of each
    cell's two sides, the one its number type names is written, the other
    left null, so a float is never also written as a ratio, and an int has
    no denominator. A fraction is written as memory holds it.

    Args:
        numbers: A column of :data:`RATIONAL_STRUCT_TYPE` (both sides filled).
        number_types: Per row ``"int"``, ``"fraction"`` or ``"float"``, as a
            string column; a null cell takes a null type.

    Returns:
        A ``pa.StructArray`` of :data:`PACKAGE_NUMBER_TYPE`, null where
        *numbers* is null.

    Raises:
        ValueError: If a non-null number names no number type of the three.

    Examples:
        >>> numbers = build_number_struct_array([0.5], number_type=NumberType.float)
        >>> package_numbers(numbers, pa.array(["float"])).to_pylist()
        [{'numerator': None, 'denominator': None, 'value': 0.5}]
    """
    sides = _members(numbers, _NUMBER_MEMBERS, "A number column")
    if _is_chunked(number_types):
        number_types = number_types.combine_chunks()
    valid = _struct_array(numbers).is_valid()
    is_int = _is(number_types, "int")
    is_fraction = _is(number_types, "fraction")
    is_float = _is(number_types, "float")
    typeless = pc.and_(valid, pc.invert(pc.or_(is_int, pc.or_(is_fraction, is_float))))
    if pc.any(typeless).as_py():
        row = typeless.index(True).as_py()
        raise ValueError(
            f"Row {row} holds a number of the number type "
            f"{number_types[row].as_py()!r}, not 'int', 'fraction' or 'float'"
        )
    return _struct_array_from(
        [
            pc.if_else(is_float, pa.scalar(None, pa.int64()), sides["numerator"]),
            pc.if_else(is_fraction, sides["denominator"], pa.scalar(None, pa.int64())),
            pc.if_else(is_float, sides["value"], pa.scalar(None, pa.float64())),
        ],
        fields=_member_fields(PACKAGE_NUMBER_TYPE),
        mask=pc.invert(valid),
    )


def numbers_from_package(
    numbers: Any, number_types: Any, *, where: str = "A package number"
) -> pa.StructArray:
    """Read a column of numbers in the package form, each by its number type.

    The decoder of the pair (:func:`package_numbers` writes): every cell must
    hold its number as its number type says (an int ``(n, null, null)``, a
    fraction ``(n, d ≥ 1, null)``, a float ``(null, null, x)`` with *x*
    finite), and the other side is filled again as memory holds it: an int's
    denominator 1 and its float, a fraction's ratio as ``Fraction(n, d)``
    holds it (``(2, 4, null)`` reads as ``1/2``, the same value) and its float
    ``n / d`` (correctly rounded, as ``float(Fraction(n, d))``), a float's
    exact ratio (:func:`build_number_struct_array`'s mirrors). Vectorized.

    Args:
        numbers: A column of :data:`PACKAGE_NUMBER_TYPE` (its members read by
            name, in any order).
        number_types: Per row the number type, as a string column.
        where: What the column is, for a message (``"MatchClaim table
            start_anchor.coordinate_a"``); the row follows.

    Returns:
        A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`, null where
        *numbers* is null.

    Raises:
        ValueError: If the column is not of the three members, or a
            non-null cell does not hold its number as its number type says
            (an unknown or null number type included), naming the first such
            row and what it holds.

    Examples:
        >>> numbers = pa.array(
        ...     [{"numerator": 3, "denominator": None, "value": None}],
        ...     PACKAGE_NUMBER_TYPE,
        ... )
        >>> numbers_from_package(numbers, pa.array(["int"])).to_pylist()
        [{'value': 3.0, 'numerator': 3, 'denominator': 1}]
    """
    sides = _members(numbers, _NUMBER_MEMBERS, where)
    if _is_chunked(number_types):
        number_types = number_types.combine_chunks()
    valid = _struct_array(numbers).is_valid()
    numerator, denominator, value = (sides[name] for name in _NUMBER_MEMBERS)
    has_numerator = numerator.is_valid()
    has_denominator = denominator.is_valid()
    has_value = value.is_valid()
    is_int = _is(number_types, "int")
    is_fraction = _is(number_types, "fraction")
    is_float = _is(number_types, "float")
    held = pc.or_(
        pc.and_(
            is_int,
            pc.and_(has_numerator, pc.invert(pc.or_(has_denominator, has_value))),
        ),
        pc.or_(
            pc.and_(
                is_fraction,
                pc.and_(
                    pc.and_(has_numerator, pc.invert(has_value)),
                    _filled_mask(
                        _compared("greater_equal", denominator, pa.scalar(1)),
                        pa.scalar(False),
                    ),
                ),
            ),
            pc.and_(
                is_float,
                pc.and_(
                    pc.invert(pc.or_(has_numerator, has_denominator)),
                    _filled_mask(pc.is_finite(value), pa.scalar(False)),
                ),
            ),
        ),
    )
    wrong = pc.and_(valid, pc.invert(held))
    if pc.any(wrong).as_py():
        row = wrong.index(True).as_py()
        number_type = number_types[row].as_py()
        cell = (numerator[row].as_py(), denominator[row].as_py(), value[row].as_py())
        spelled = "(" + ", ".join("null" if x is None else repr(x) for x in cell) + ")"
        if number_type not in ("int", "fraction", "float"):
            problem = (
                f"its number type is {number_type!r}, not 'int', 'fraction' or 'float'"
            )
        else:
            problem = (
                f"the {number_type} cell (numerator, denominator, value) holds "
                f"{spelled}, where an int is (n, null, null), a fraction "
                f"(n, d >= 1, null) and a float (null, null, x) with x finite"
            )
        raise ValueError(f"{where} row {row}: {problem}")

    int_rows = is_int.to_numpy(zero_copy_only=False)
    float_rows = is_float.to_numpy(zero_copy_only=False)
    n = numerator.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    d = denominator.fill_null(1).to_numpy(zero_copy_only=False).astype(np.int64)
    x = value.fill_null(0.0).to_numpy(zero_copy_only=False).astype(np.float64)
    d[int_rows] = 1
    n, d = _lowest_terms(n, d)
    mirror_n, mirror_d = _pairs_from_float_array(np.where(float_rows, x, 0.0))
    out_n = np.where(float_rows, mirror_n, n)
    out_d = np.where(float_rows, mirror_d, d)
    # The correctly rounded quotient a Fraction gives in memory, so a position
    # read from a file and the same position built in memory share their
    # float key.
    out_x = np.where(float_rows, x, _ratio_values(n, d))
    null = pc.invert(valid).to_numpy(zero_copy_only=False)
    return _struct_array_from(
        [
            pa.array(out_x, pa.float64(), mask=null),
            pa.array(out_n, pa.int64(), mask=null),
            pa.array(out_d, pa.int64(), mask=null),
        ],
        fields=_member_fields(RATIONAL_STRUCT_TYPE),
        mask=pa.array(null),
    )


def package_cells(cells: Any) -> pa.StructArray:
    """Write a column of coordinate cells in the package form.

    Each cell's number by :func:`package_numbers`, in its own number type;
    ``unit`` and ``number_type`` kept.

    Args:
        cells: A column of :data:`COORDINATE_CELL_TYPE`.

    Returns:
        A ``pa.StructArray`` of :data:`PACKAGE_COORDINATE_CELL_TYPE`.
    """
    members = _members(
        cells, list(COORDINATE_CELL_TYPE.names), "A coordinate cell column"
    )
    valid = _struct_array(cells).is_valid()
    numbers = _struct_array_from(
        [members[name] for name in RATIONAL_STRUCT_TYPE.names],
        fields=_member_fields(RATIONAL_STRUCT_TYPE),
        mask=pc.invert(valid),
    )
    written = package_numbers(numbers, members["number_type"])
    return _struct_array_from(
        [*_struct_members(written), members["unit"], members["number_type"]],
        fields=_member_fields(PACKAGE_COORDINATE_CELL_TYPE),
        mask=pc.invert(valid),
    )


def cells_from_package(
    cells: Any, *, where: str = "A coordinate cell"
) -> pa.StructArray:
    """Read a column of coordinate cells in the package form.

    Each cell's number by :func:`numbers_from_package`, in the number type
    the cell names; a non-null cell must name its unit.

    Args:
        cells: A column of :data:`PACKAGE_COORDINATE_CELL_TYPE` (members
            read by name, in any order).
        where: What the column is, for a message; the row follows.

    Returns:
        A ``pa.StructArray`` of :data:`COORDINATE_CELL_TYPE`.

    Raises:
        ValueError: As :func:`numbers_from_package`, or if a non-null cell
            names no unit.
    """
    members = _members(cells, list(PACKAGE_COORDINATE_CELL_TYPE.names), where)
    valid = _struct_array(cells).is_valid()
    unitless = pc.and_(valid, members["unit"].is_null())
    if pc.any(unitless).as_py():
        raise ValueError(
            f"{where} row {unitless.index(True).as_py()}: it names no unit"
        )
    numbers = _struct_array_from(
        [members[name] for name in _NUMBER_MEMBERS],
        fields=_member_fields(PACKAGE_NUMBER_TYPE),
        mask=pc.invert(valid),
    )
    read = numbers_from_package(numbers, members["number_type"], where=where)
    return _struct_array_from(
        [*_struct_members(read), members["unit"], members["number_type"]],
        fields=_member_fields(COORDINATE_CELL_TYPE),
        mask=pc.invert(valid),
    )


# ---------------------------------------------------------------------------
# SharedAxisField — fields whose numbers share one axis
# ---------------------------------------------------------------------------

AxisScalarT = TypeVar("AxisScalarT", bound=BaseModel)
"""The scalar of a :class:`SharedAxisField`."""


class AxisScalarLike(Protocol):
    """A scalar that reads itself from a row whose numbers lie on one axis.

    A :class:`SharedAxisField` stores each number as the number struct alone
    and the axis once, in its metadata; a scalar of this shape puts the axis
    back as it reads a row, and the field materialises its rows through it.
    Not every shared-axis scalar has this shape: :class:`Interval` has no
    ``from_row``, and :class:`IntervalField` reads its rows itself.
    """

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any] | None,
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None,
    ) -> Self | None:
        """Construct from *row* on the axis given; ``None`` for a null cell."""
        ...


def _reads_axis_rows(scalar: type) -> TypeIs[type[AxisScalarLike]]:
    """Whether *scalar* reads its rows on one axis (:class:`AxisScalarLike`)."""
    return hasattr(scalar, "from_row")


def _as_unit(unit: TimeUnit | str | None) -> TimeUnit | None:
    """*unit* as a :class:`TimeUnit`; ``None`` stays ``None``."""
    return TimeUnit(unit) if isinstance(unit, str) else unit


def _as_number_type(number_type: NumberType | str | None) -> NumberType | None:
    """*number_type* as a :class:`NumberType`; ``None`` stays ``None``."""
    return NumberType(number_type) if isinstance(number_type, str) else number_type


class SharedAxisField(_AxisField[AxisScalarT]):
    """A field whose every number lies on one axis, declared once.

    The scalar holds several numbers (an interval's two ends, a shape's
    coordinates, a box's interval and shape), all in one unit and one
    number type. Each is stored as the number struct alone
    (:data:`RATIONAL_STRUCT_TYPE`, both sides filled); the unit and the
    number type are written once, in the field's metadata, and read back
    from there.

    Args:
        raw: The struct field holding the numbers.
        unit: The unit of every number; read from the metadata when omitted.
        number_type: The number type of every number; read from the metadata
            when omitted.
        source_fields: Optional blueprint source-field declaration.

    Raises:
        TypeError: If neither *raw* nor *source_fields* is given.
        ValueError: If *raw* is given and no unit is found.
    """

    def __init__(
        self,
        raw: StructField | None = None,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        *,
        source_fields: str | dict[str, Any] | None = None,
    ) -> None:
        name = type(self).__name__
        if raw is None and source_fields is None:
            raise TypeError(f"{name} requires either a raw field or source_fields=")
        super().__init__(raw, source_fields=source_fields)
        self._unit: TimeUnit | None
        if raw is None:
            self._unit = None
            self._number_type = NumberType.float
            return
        resolved_unit, resolved_number_type = self._resolve_metadata(
            raw.field, unit, number_type
        )
        if resolved_unit is None:
            raise ValueError(f"{name} requires unit metadata or an explicit unit")
        self._unit = resolved_unit
        self._number_type = resolved_number_type

    @property
    def unit(self) -> TimeUnit | None:
        """Return the unit every number is in; ``None`` while unbound."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """Return the number type every number is written in."""
        return self._number_type

    @property
    def semantic_type(self) -> str:
        """Return the paired scalar's name."""
        scalar = type(self).scalar_cls
        if scalar is None:
            # An unpaired field (ShapeField itself) has no semantic type; as an
            # AttributeError, ``hasattr`` and runtime protocol checks answer False.
            raise AttributeError(f"{type(self).__name__} has no paired scalar")
        return scalar.__name__

    @classmethod
    def _shared_axis(
        cls,
        values: Sequence[Any],
        unit: TimeUnit | str | None,
        number_type: NumberType | str | None,
    ) -> tuple[TimeUnit, NumberType]:
        """The one axis of *values* (each with ``unit`` and ``number_type``).

        An explicit *unit* or *number_type* must agree with the values; an
        empty or all-null input needs both.

        Raises:
            ValueError: If the values lie on two axes, an explicit axis
                contradicts them, or an empty input names none.
        """
        name = cls.__name__
        unit, number_type = _as_unit(unit), _as_number_type(number_type)
        present = [value for value in values if value is not None]
        if not present:
            if unit is None or number_type is None:
                raise ValueError(f"Empty {name} input requires unit and number_type")
            return unit, number_type
        inferred_unit = present[0].unit
        inferred_number_type = present[0].number_type
        if any(
            value.unit != inferred_unit or value.number_type != inferred_number_type
            for value in present
        ):
            raise ValueError(f"{name} values must share one unit and number_type")
        if unit is not None and unit != inferred_unit:
            raise ValueError(f"Explicit {name} unit conflicts with values")
        if number_type is not None and number_type != inferred_number_type:
            raise ValueError(f"Explicit {name} number_type conflicts with values")
        return inferred_unit, inferred_number_type

    @classmethod
    def _of_struct_array(
        cls, array: pa.StructArray, name: str, unit: TimeUnit, number_type: NumberType
    ) -> Self:
        """The field of *array*, named *name*, on the axis given."""
        return cls(StructField(array, pa.field(name, array.type)), unit, number_type)

    @classmethod
    def _from_scalars(
        cls,
        items: Sequence[AxisScalarT | None],
        unit: TimeUnit | str | None,
        number_type: NumberType | str | None,
        name: str | None,
    ) -> Self:
        """The field of the scalars *items* (``None`` for a null row), in order.

        The one body of the ``from_*`` builders of fields whose scalar's
        derived struct holds its numbers as number structs: the axis is
        :meth:`_shared_axis`'s, the struct :func:`build_struct_array`'s, the
        name *name* or the class's default.

        Raises:
            ValueError: If the scalars lie on two axes, or an explicit axis
                contradicts them.
        """
        values = list(items)
        unit, number_type = cls._shared_axis(values, unit, number_type)
        return cls._of_struct_array(
            build_struct_array(cls.paired_scalar(), values),
            name if name is not None else cls._default_name(),
            unit,
            number_type,
        )

    @classmethod
    def from_field(
        cls,
        source: (
            pa.Array[Any]
            | pa.ChunkedArray[Any]
            | StructField
            | pa.Field[pa.DataType]
            | tuple[pa.Array[Any] | None, pa.Field[pa.DataType]]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str | None = None,
        **kw: Any,
    ) -> Self:
        """Build the field from an Arrow or raw field source.

        Args:
            source: Supported Arrow field representation.
            unit: Unit override; else the metadata's.
            number_type: Number type override; else the metadata's.
            name: Field name for a bare array; the class's default name
                when omitted.
            **kw: Keywords this method does not use; ignored, with a
                warning naming them.

        Returns:
            A live or blueprint field.
        """
        cls._warn_ignored_keywords(kw)
        if _is_tuple(source):
            data, field = source
            return cls(StructField(data, field), unit, number_type)
        if _is_arrow_field(source):
            return cls(StructField(None, source), unit, number_type)
        if isinstance(source, StructField):
            return cls(source, unit, number_type)
        if _is_arrow_column(source):
            field_name = name if name is not None else cls._default_name()
            return cls(
                StructField(source, pa.field(field_name, source.type)),
                unit,
                number_type,
            )
        raise TypeError(
            f"Unsupported source type for {cls.__name__}.from_field: "
            f"{type(source).__name__}"
        )

    @classmethod
    def _default_name(cls) -> str:
        """The name of a field built without one: the scalar's, in snake case
        (``interval``, ``bounding_box``)."""
        return re.sub(
            r"(?<!^)(?=[A-Z])", "_", cls.__name__.removesuffix("Field")
        ).lower()

    def __getitem__(self, i: int) -> AxisScalarT | None:
        """Materialize one scalar on the field's axis, or ``None``.

        Raises:
            TypeError: If the paired scalar does not read its rows on one
                axis (:class:`AxisScalarLike`); the field of such a scalar
                reads its rows itself, as :class:`IntervalField` does.
        """
        cell = self._raw[i]
        if cell is None:
            return None
        scalar = self.paired_scalar()
        if not _reads_axis_rows(scalar):
            raise TypeError(
                f"{type(self).__name__} cannot materialise {scalar.__name__}: "
                "it has no from_row on one axis"
            )
        return scalar.from_row(
            cell, unit=self._bound_unit(), number_type=self.number_type
        )


# ---------------------------------------------------------------------------
# Interval / IntervalField
# ---------------------------------------------------------------------------


class Interval(BaseModel):
    """A non-decreasing interval between two plain coordinates.

    Args:
        start: Inclusive interval start.
        end: Exclusive interval end.

    Raises:
        TypeError: If either endpoint is not a plain :class:`Coordinate`.
        ValueError: If endpoint units or number types differ, or if the end
            precedes the start.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    start: Coordinate
    end: Coordinate

    def __init__(
        self,
        start: Coordinate | TimeScalarValue | None = None,
        end: Coordinate | TimeScalarValue | None = None,
        unit: TimeUnit | str | None = None,
        /,
        **data: Any,
    ) -> None:
        if start is not None or end is not None or unit is not None:
            if "start" in data or "end" in data:
                raise TypeError(
                    f"{type(self).__name__} received conflicting positional "
                    "and keyword arguments"
                )
            resolved_unit = TimeUnit(unit) if unit is not None else None
            endpoints: list[Coordinate] = []
            for label, value in (("start", start), ("end", end)):
                if isinstance(value, Coordinate):
                    if resolved_unit is not None and value.unit != resolved_unit:
                        raise ValueError(
                            f"Interval {label} unit {value.unit} does not match "
                            f"requested unit {resolved_unit}"
                        )
                    endpoints.append(value)
                    continue
                if value is None:
                    raise TypeError(f"Interval {label} must be provided")
                if resolved_unit is None:
                    raise TypeError(
                        "Raw Interval endpoints require a TimeUnit or abbreviation"
                    )
                endpoints.append(
                    Coordinate(
                        value,
                        resolved_unit,
                        number_type=resolved_unit.default_number_type,
                    )
                )
            data = {"start": endpoints[0], "end": endpoints[1], **data}
        super().__init__(**data)

    @field_validator("start", "end", mode="before")
    @classmethod
    def _validate_plain_coordinate(cls, value: object) -> Coordinate:
        if type(value) is not Coordinate:
            raise TypeError(
                "Interval endpoints must be plain Coordinate values; "
                f"got {type(value).__name__}"
            )
        return Coordinate(
            value.value,
            value.unit,
            number_type=value.number_type,
        )

    @model_validator(mode="after")
    def _validate_interval(self) -> Interval:
        if self.start.unit != self.end.unit:
            raise ValueError(
                "Interval endpoints must use the same unit: "
                f"{self.start.unit} != {self.end.unit}"
            )
        if self.start.number_type != self.end.number_type:
            raise ValueError(
                "Interval endpoints must use the same number_type: "
                f"{self.start.number_type} != {self.end.number_type}"
            )
        if self.start.value > self.end.value:
            raise ValueError("Interval start must not exceed interval end")
        return self

    @property
    def unit(self) -> TimeUnit:
        """Return the common endpoint unit."""
        return self.start.unit

    @property
    def number_type(self) -> NumberType:
        """Return the common endpoint number type."""
        return self.start.number_type

    @property
    def duration(self) -> Duration:
        """Return the exact non-negative interval duration."""
        return Duration(
            self.end.value - self.start.value,
            self.unit,
            number_type=self.number_type,
        )

    # -- formatting ---------------------------------------------------------

    def __str__(self) -> str:
        """Render the half-open span, naming the shared unit once.

        Both endpoints go through the same formatter the scalars use, so an
        interval can never show less of a value than a bare
        :class:`Coordinate` would: ``[1/2, 3/2) quarters``, not
        ``[0.5, 1.5)``. The unit is stated once because the endpoints are
        validated to share it.
        """
        return f"[{self.start._format_value()}, {self.end._format_value()}) {self.unit}"


class IntervalField(SharedAxisField[Interval]):
    """Semantic field storing intervals with shared axis metadata.

    Args:
        raw: Struct field containing ``start`` and ``end`` coordinate structs.
        unit: Common endpoint unit.
        number_type: Common endpoint number type.
        source_fields: Optional blueprint source-field declaration.
    """

    @classmethod
    def from_intervals(
        cls,
        intervals: Sequence[Interval | None],
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str = "interval",
    ) -> IntervalField:
        """Build an interval field from scalar intervals.

        Args:
            intervals: Interval values, optionally containing null rows.
            unit: Shared unit override, required for an empty/all-null input.
            number_type: Shared number type override, required for an
                empty/all-null input.
            name: Arrow field name.

        Returns:
            A live interval field.

        Raises:
            ValueError: If non-null intervals do not share one metadata triple.
        """
        resolved_unit, resolved_number_type = cls._shared_axis(
            intervals, unit, number_type
        )
        rows = [
            (
                None
                if value is None
                else {
                    "start": coordinate_to_struct(value.start.value),
                    "end": coordinate_to_struct(value.end.value),
                }
            )
            for value in intervals
        ]
        interval_type = pa.struct(
            [
                pa.field("start", RATIONAL_STRUCT_TYPE),
                pa.field("end", RATIONAL_STRUCT_TYPE),
            ]
        )
        array = _struct_array_of(rows, interval_type)
        return cls._of_struct_array(array, name, resolved_unit, resolved_number_type)

    def __getitem__(self, i: int) -> Interval | None:
        """Materialize one interval scalar or null."""
        cell = self._raw[i]
        if cell is None:
            return None
        return Interval(
            start=Coordinate(
                struct_to_coordinate(cell["start"], self.number_type),
                self.unit,
                number_type=self.number_type,
            ),
            end=Coordinate(
                struct_to_coordinate(cell["end"], self.number_type),
                self.unit,
                number_type=self.number_type,
            ),
        )

    def _declares_interval(self) -> bool:
        """Whether the declared struct holds ``start`` and ``end`` as number structs."""
        declared = self.pa_type
        if not isinstance(declared, pa.StructType):
            return False
        members = _member_fields(declared)
        return all(
            declared.get_field_index(member) >= 0
            and members[declared.get_field_index(member)].type == RATIONAL_STRUCT_TYPE
            for member in ("start", "end")
        )

    def _endpoint(self, name: str) -> CoordinateField | None:
        """The field of the endpoint *name* on the interval's axis.

        ``None`` while the field is unbound; a field carrying no data of
        its own when the interval field carries none.

        Raises:
            TypeError: If the field carries no data and its declared struct
                is not an interval's, or a member is not a number struct.
            KeyError: If the rows' struct has no member *name*.
        """
        unit = self.unit
        if unit is None:
            return None
        if self.is_empty and self._declares_interval():
            return CoordinateField.from_field(
                pa.field(name, RATIONAL_STRUCT_TYPE),
                unit=unit,
                number_type=self.number_type,
            )
        array = _struct_array(self.to_pyarrow())
        child = _struct_child(array, name)
        outer_valid = array.is_valid()
        child = pc.if_else(outer_valid, child, pa.scalar(None, type=child.type))
        return CoordinateField.from_field(
            child,
            unit=unit,
            number_type=self.number_type,
            name=name,
        )

    @property
    def start(self) -> CoordinateField | None:
        """Return the vectorized start-coordinate field; ``None`` while unbound."""
        return self._endpoint("start")

    @property
    def end(self) -> CoordinateField | None:
        """Return the vectorized end-coordinate field; ``None`` while unbound."""
        return self._endpoint("end")

    @property
    def duration(self) -> DurationField | None:
        """Return the vectorized duration field; ``None`` while unbound.

        A field carrying no data answers a duration field carrying none.
        """
        end, start = self.end, self.start
        if end is None or start is None:
            return None
        if self.is_empty:
            return DurationField.from_field(
                pa.field("value", RATIONAL_STRUCT_TYPE),
                unit=end.unit,
                number_type=self.number_type,
            )
        return end - start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_timeline_id(source: Any, override: object) -> str:
    """Resolve ``timeline_id`` from a kwarg override or field metadata."""
    if override is not None:
        if not isinstance(override, str) or not override:
            raise ValueError(
                f"timeline_id must be a non-empty string, got {override!r}"
            )
        return override

    pa_field: pa.Field[pa.DataType] | None = None
    if _is_tuple(source) and len(source) == 2:
        pa_field = source[1]
    elif _is_arrow_field(source):
        pa_field = source
    elif isinstance(source, StructField):
        pa_field = source.field

    if pa_field is not None and pa_field.metadata:
        if TIMETOALIGN_METADATA_KEY in pa_field.metadata:
            try:
                meta = parse_metadata_blob(pa_field.metadata[TIMETOALIGN_METADATA_KEY])
            except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
                meta = {}
            tl_id = meta.get("timeline_id")
            if isinstance(tl_id, str) and tl_id:
                return tl_id

    raise ValueError(
        "timeline_id is required for Id-variant fields; pass it explicitly "
        "or store it in the field's timetoalign metadata blob."
    )
