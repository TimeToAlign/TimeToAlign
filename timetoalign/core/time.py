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
    └── TimeScalarField  (abstract; _raw_cls = DenominateNumberField;
                          consolidates the from_field / from_table /
                          metadata-resolve / repr plumbing)
        ├── CoordinateField   (scalar_cls = Coordinate)
        │   └── IdCoordinateField (scalar_cls = IdCoordinate, carries timeline_id)
        └── DurationField     (scalar_cls = Duration)
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
from collections.abc import Iterable, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any, ClassVar, NamedTuple, TypeVar, Union

import numpy as np
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
    _struct_types_match,
    data_shaped,
    metadata_blob_from_dict,
    parse_metadata_blob,
    register_value_projector,
)

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


def parse_number(value: Any) -> int | float | Fraction:
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
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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

    Raises:
        ValueError: On an exact non-integral value into an ``int`` field, or
            an unknown rounding mode.
    """
    number = parse_number(value)
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
    if isinstance(value, dict):
        return _normalise_cell(value)
    number = parse_number(value)
    if number_type is None:
        number_type = NumberType.from_number(number)
    canonical = to_canonical(number, number_type, rounding=rounding)
    if number_type is NumberType.float:
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
    return {
        "value": (None if cell["value"] is None else float(cell["value"])),
        "numerator": _coerce_ratio_component(numerator, "numerator"),
        "denominator": _coerce_ratio_component(denominator, "denominator"),
    }


def _coerce_ratio_component(component: Any, label: str) -> int:
    """Return one ratio component as an integer."""
    if isinstance(component, bool):
        raise ValueError(f"rational struct {label} must be an integer, got a bool")
    if isinstance(component, (int, np.integer)):
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


def number_to_json(value: int | float | Fraction) -> int | float | list[int]:
    """Encode a Python number using the TimeSkeleton JSON representation.

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
    if not isinstance(value, list) or len(value) != 2:
        return False
    members = [_integral_member(component) for component in value]
    return None not in members and members[1] > 0


def number_from_json(
    data: Any, number_type: NumberType | None = None
) -> int | float | Fraction:
    """Decode one TimeSkeleton JSON number.

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
    elif isinstance(data, list):
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
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values.to_pylist()
    if isinstance(values, pd.Series):
        return values.tolist()
    if isinstance(values, np.ndarray):
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

    A plain numeric column takes a vectorised path; anything mixed, textual
    or rational falls back to reading values one at a time, because no array
    kernel parses ``"3/8"``.

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
    return pa.array(rows, type=RATIONAL_STRUCT_TYPE)


def _numeric_source(values: Any) -> np.ndarray | None:
    """Return *values* as a plain float64/int64 numpy array, else ``None``."""
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array):
        if not (pa.types.is_integer(values.type) or pa.types.is_floating(values.type)):
            return None
        values = values.to_numpy(zero_copy_only=False)
    elif isinstance(values, pd.Series):
        values = values.to_numpy()
    elif isinstance(values, list):
        values = np.asarray(values)
    if not isinstance(values, np.ndarray):
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
        return pa.array([], type=RATIONAL_STRUCT_TYPE)

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
        return pa.StructArray.from_arrays(
            [
                pa.array(value, type=pa.float64(), mask=null_mask),
                pa.array(numerator, type=pa.int64(), mask=null_mask),
                pa.array(denominator, type=pa.int64(), mask=null_mask),
            ],
            fields=list(RATIONAL_STRUCT_TYPE),
            mask=pa.array(null_mask),
        )
    return pa.StructArray.from_arrays(
        [
            pa.array(value, type=pa.float64()),
            pa.array(numerator, type=pa.int64()),
            pa.array(denominator, type=pa.int64()),
        ],
        fields=list(RATIONAL_STRUCT_TYPE),
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
    return pa.array(rows, type=RATIONAL_STRUCT_TYPE)


# ---------------------------------------------------------------------------
# Arithmetic — one engine for scalars and fields alike
# ---------------------------------------------------------------------------


_ADDITIVE_OPERATIONS: frozenset[str] = frozenset({"add", "subtract"})

_OPERATIONS: dict[str, Any] = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
    "floor_divide": operator.floordiv,
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
    left: pa.Array,
    right: pa.Array | int | float | Fraction,
    operation: str,
    number_type: NumberType,
    *,
    rounding: str = "round",
) -> pa.StructArray:
    """Combine a column of storage cells with another column or a scalar.

    Same rules as :func:`combine_numbers`, applied to whole columns: the
    operation runs on the canonical side and the mirror is rebuilt from the
    result, so the two sides of the output never disagree.

    Vectorisation is honest about what it can do. The ``int`` and ``float``
    lanes run as PyArrow kernels over the canonical sub-field. The
    ``fraction`` lane reads rows into Python, because exact ratio arithmetic
    over int64 columns would overflow on the very denominators that make
    exactness worth having.

    Args:
        left: A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.
        right: Another such array, or a single number.
        operation: One of the keys of :data:`_OPERATIONS`.
        number_type: The representation the result declares.
        rounding: How to make inexact results integral for an ``int`` result.

    Returns:
        A ``pa.StructArray`` of :data:`RATIONAL_STRUCT_TYPE`.
    """
    if operation not in _OPERATIONS:
        raise ValueError(f"Unknown field arithmetic operation: {operation}")

    if number_type is NumberType.fraction:
        return _combine_exact_columns(left, right, operation, rounding=rounding)

    left_values = _canonical_column(left, number_type)
    if isinstance(right, (pa.Array, pa.ChunkedArray)):
        right_values = _canonical_column(right, number_type)
    else:
        right_values = pa.scalar(
            to_canonical(right, number_type, rounding=rounding),
            type=left_values.type,
        )

    result = _COLUMN_KERNELS[operation](left_values, right_values)
    return build_number_struct_array(result, number_type=number_type, rounding=rounding)


_COLUMN_KERNELS: dict[str, Any] = {
    "add": pc.add,
    "subtract": pc.subtract,
    "multiply": pc.multiply,
    "divide": pc.divide,
    "floor_divide": lambda a, b: pc.floor(pc.divide(a, b)),
}


def _canonical_column(array: pa.Array, number_type: NumberType) -> pa.Array:
    """Extract the canonical sub-field of a storage column, nulls preserved."""
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    valid = array.is_valid()
    if number_type is NumberType.int:
        canonical = array.field("numerator")
        canonical = pc.cast(canonical, pa.float64())
    else:
        canonical = pc.cast(array.field("value"), pa.float64())
    return pc.if_else(valid, canonical, pa.scalar(None, type=pa.float64()))


def _combine_exact_columns(
    left: pa.Array,
    right: pa.Array | int | float | Fraction,
    operation: str,
    *,
    rounding: str,
) -> pa.StructArray:
    """Row-wise exact arithmetic for the ``fraction`` lane."""
    if isinstance(left, pa.ChunkedArray):
        left = left.combine_chunks()
    left_rows = left.to_pylist()
    if isinstance(right, (pa.Array, pa.ChunkedArray)):
        if isinstance(right, pa.ChunkedArray):
            right = right.combine_chunks()
        right_rows: list[Any] = right.to_pylist()
    else:
        right_rows = [right] * len(left_rows)

    rows: list[dict[str, Any] | None] = []
    for left_row, right_row in zip(left_rows, right_rows):
        if left_row is None or right_row is None:
            rows.append(None)
            continue
        # Decode through struct_to_coordinate, not struct_to_rational: a
        # hand-built or legacy cell may carry the number on the float side
        # only, and arithmetic should read it rather than refuse it.
        left_value = struct_to_coordinate(left_row, NumberType.fraction)
        right_value = (
            struct_to_coordinate(right_row, NumberType.fraction)
            if isinstance(right_row, dict)
            else right_row
        )
        rows.append(
            number_cell(
                combine_numbers(
                    left_value,
                    right_value,
                    operation,
                    NumberType.fraction,
                    rounding=rounding,
                ),
                NumberType.fraction,
            )
        )
    return pa.array(rows, type=RATIONAL_STRUCT_TYPE)


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
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(values, pa.Array):
        if pa.types.is_integer(values.type):
            return NumberType.int
        if pa.types.is_floating(values.type):
            return NumberType.float
    elif isinstance(values, np.ndarray):
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
    _blueprint_pa_type: ClassVar[pa.DataType] = RATIONAL_STRUCT_TYPE

    def __init__(
        self,
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
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
        cls, source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field], **kw: Any
    ) -> "RedundantNumberField":
        data, field = source
        return cls(data, field)

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
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
        data: pa.Array | pa.ChunkedArray | None = None,
        field: pa.Field | None = None,
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
    def _field_metadata(pa_field: pa.Field) -> dict[str, Any]:
        raw_meta = pa_field.metadata
        if not raw_meta or TIMETOALIGN_METADATA_KEY not in raw_meta:
            return {}
        try:
            return parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
        except (UnicodeDecodeError, ValueError):
            return {}

    @staticmethod
    def _resolve_unit(pa_field: pa.Field, override: Any) -> TimeUnit:
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
    def _resolve_number_type(pa_field: pa.Field, override: Any) -> Any:
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
        source: tuple[pa.Array | pa.ChunkedArray | None, pa.Field],
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        **kw: Any,
    ) -> "DenominateNumberField":
        """Construct from a ``(data, pa.Field)`` tuple."""
        data, field = source
        if unit is None:
            unit = cls._resolve_unit(field, None)
        return cls(data, field, unit=unit, number_type=number_type)

    def from_array(
        self,
        source: pa.Array | pa.ChunkedArray,
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
    if isinstance(value, dict):
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
    if isinstance(value, dict):
        return struct_to_rational(value)
    if hasattr(value, "value"):
        return value.value
    return float(value)


def shift_coordinate(value: Any, offset: Any, *, subtract: bool) -> dict[str, Any]:
    """Shift a coordinate by an offset and rebuild its storage cell.

    Both sides of a cell always carry the number, so the shift runs on the
    exact side whenever the operands have one -- which, for a stored
    coordinate, is always. Parent-to-child offsetting therefore composes
    without drifting, however deep the nesting goes.
    """
    value_exact = exact_coordinate_value(value)
    offset_exact = exact_coordinate_value(offset)
    if value_exact is not None and offset_exact is not None:
        result = value_exact - offset_exact if subtract else value_exact + offset_exact
        return coordinate_to_struct(result)

    left = float(coordinate_numeric_value(value))
    right = float(coordinate_numeric_value(offset))
    return coordinate_to_struct(left - right if subtract else left + right)


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
        return self.value == 0

    @data_shaped
    def is_positive(self) -> bool:
        return self.value > 0

    @data_shaped
    def is_negative(self) -> bool:
        return self.value < 0

    # -- mutators (copy-on-write) ------------------------------------------

    def _rebuild(self, value: TimeScalarValue, unit: TimeUnit) -> TimeScalar:
        """Construct a same-class instance with possibly new value/unit.

        Preserves ``timeline_id`` for Id-variants.
        """
        tl_id = self._id_or_none()
        if tl_id is not None:
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

    def __init__(self, *args: Any, **data: Any) -> None:
        if type(self) is IdTimeScalar:
            raise TypeError(
                "IdTimeScalar is abstract; construct IdCoordinate or IdDuration"
            )
        super().__init__(*args, **data)

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
) -> list[pa.Field]:
    """Project ``value`` onto the denormalised storage struct."""
    return list(RATIONAL_STRUCT_TYPE)


def _drop_field_projector(
    _model_cls: type[BaseModel], _name: str, _info: object
) -> list[pa.Field]:
    """Drop a pydantic field from the Arrow representation — lives in metadata."""
    return []


register_value_projector(TimeScalar, "value", _time_value_projector)
register_value_projector(TimeScalar, "unit", _drop_field_projector)
register_value_projector(IdTimeScalar, "timeline_id", _drop_field_projector)


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


def _scalar_binop_value_and_id(
    self: TimeScalarField, other: Any, op: str
) -> tuple[Any, str | None]:
    """Extract the numeric payload and result timeline_id for arithmetic.

    Performs the metadata-level invariant check exactly once (unit match,
    cross-timeline-id mismatch) at the entry point of a Field arithmetic
    operation — NEVER per row.  Returns ``(numeric_other, result_tl_id)``.
    """
    self_tl = getattr(self, "_timeline_id", None)
    if isinstance(other, TimeScalar):
        if self.unit != other.unit:
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
    if isinstance(other, TimeScalarField):
        if self.unit != other.unit:
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
    raise TypeError(f"Cannot {op} {type(self).__name__} with {type(other).__name__}")


def _field_binop(
    left: "TimeScalarField",
    other: Any,
    op_name: str,
    operation: str,
) -> tuple[pa.StructArray, str | None]:
    """Check the metadata invariants once, then combine the columns.

    Unit and timeline-id compatibility are properties of the two *fields*,
    so they are settled here, before any data is touched — never re-tested
    per row. The arithmetic then runs in the left operand's declared
    representation, which is what decides the result's.
    """
    _, tl_id = _scalar_binop_value_and_id(left, other, op_name)
    if isinstance(other, TimeScalarField):
        operand: Any = other.to_pyarrow()
    elif isinstance(other, TimeScalar):
        operand = other.value
    else:
        operand = other
    return (
        combine_number_columns(
            left.to_pyarrow(), operand, operation, left._number_type
        ),
        tl_id,
    )


class TimeScalarField(SemanticField):
    """Abstract parent for ``Coordinate`` / ``Duration`` (+ Id) semantic fields.

    Provides the shared ``from_field`` / ``from_table`` / ``__repr__``
    machinery for ``CoordinateField`` and ``DurationField``. Concrete
    subclasses set ``scalar_cls`` (which drives ``pa_schema`` caching through
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
    # leaves set it.

    def __init__(
        self,
        raw: StructField | DenominateNumberField | None = None,
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
        self._number_type = (
            NumberType(number_type) if isinstance(number_type, str) else number_type
        )

    # -- forwarded properties from the raw layer ----------------------------

    @property
    def unit(self) -> Any:
        """The time unit bound to this field (via the inner ``DenominateNumberField``)."""
        return None if self._raw is None else self._raw.unit

    @property
    def _unit(self) -> Any:
        """Back-compat alias for :attr:`unit`.

        Older callers (subclass methods, tests, serialisation helpers)
        reach into ``self._unit`` directly; keep the name available so
        we do not have to rewrite every reference at once.
        """
        return self.unit

    @property
    def domain(self) -> Any:
        """The temporal domain implied by the unit."""
        return None if self._raw is None else self._raw.domain

    @property
    def number_type(self) -> Any:
        """The numeric representation used for scalar access."""
        return self._number_type

    # -- abstract semantic contract ----------------------------------------

    @property
    def semantic_type(self) -> str:
        # Concrete subclasses override with their scalar's name.
        return type(self).__name__.removesuffix("Field")

    def metadata_dict(self) -> dict[str, str]:
        """Return the discriminator plus the (unit, domain, number_type) triple."""
        return {
            **super().metadata_dict(),
            "unit": self.unit.value,
            "domain": self.domain.value,
            "number_type": self._number_type.name,
        }

    # -- materialisation ----------------------------------------------------

    def _materialise_value(self, i: int) -> TimeScalarValue | None:
        """Extract the i-th element's value in this field's representation."""
        cell = self._raw.cell(i)
        if cell is None:
            return None
        return struct_to_coordinate(cell, self._number_type)

    def __getitem__(self, i: int):
        """Materialise the i-th scalar instance.

        Default implementation builds ``cls.scalar_cls(value, unit)``.
        Id-variants override to thread the field's ``_timeline_id``
        into the result.
        """
        value = self._materialise_value(i)
        if value is None:
            return None
        cls = type(self).scalar_cls
        if cls is None:  # pragma: no cover - guarded by __init_subclass__
            raise TypeError(
                f"{type(self).__name__} has no scalar_cls; cannot materialise scalar"
            )
        return cls(value, self.unit, number_type=self._number_type)

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

    @staticmethod
    def _resolve_metadata(
        pa_field: pa.Field,
        unit_override: Any,
        nt_override: Any,
    ) -> tuple[Any, Any]:
        """Extract unit and number_type from a ``pa.Field``'s metadata."""
        meta: dict[str, Any] = {}
        raw_meta = pa_field.metadata
        if raw_meta:
            if TIMETOALIGN_METADATA_KEY in raw_meta:
                meta = parse_metadata_blob(raw_meta[TIMETOALIGN_METADATA_KEY])
            else:
                meta = {
                    (k.decode("utf-8") if isinstance(k, bytes) else k): (
                        v.decode("utf-8") if isinstance(v, bytes) else v
                    )
                    for k, v in raw_meta.items()
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
    ) -> TimeScalarField:
        """Construct ``cls`` from one of the supported source shapes.

        ``extra_resolved`` is a hook for Id-subclasses to inject already-
        resolved kwargs (e.g. ``timeline_id``) into the final constructor.
        """
        extra_resolved = extra_resolved or {}

        if isinstance(source, tuple):
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

        if isinstance(source, pa.Field):
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

        if isinstance(source, (pa.Array, pa.ChunkedArray)):
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
            pa.Array
            | pa.ChunkedArray
            | StructField
            | pa.Field
            | tuple[pa.Array | None, pa.Field]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str | None = None,
    ) -> TimeScalarField:
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
    ) -> TimeScalarField:
        if field is None:
            candidates = [
                f.name
                for f in table.schema
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
        pa_field = table.schema.field(field)
        data = table.column(field)
        return cls.from_field((data, pa_field), unit=unit, number_type=number_type)

    def with_unit(self, unit: TimeUnit) -> TimeScalarField:
        """Return a same-class field with a different unit (no value conversion)."""
        return type(self)(self._raw, unit, self._number_type)

    # -- vectorized data-shaped mirrors -------------------------------------
    #
    # These mirror the @data_shaped methods on ``TimeScalar`` (declared in
    # scalar form earlier in the module).  Each is a ``pa.compute``
    # expression over the underlying ``value`` sub-field of the
    # denormalised rational struct, NEVER a per-row Python loop.

    def _value_array(self) -> pa.Array:
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
        arr = self.to_pyarrow()
        value_arr = arr.field("value")
        outer_valid = arr.is_valid()
        return pc.if_else(outer_valid, value_arr, pa.scalar(None, type=value_arr.type))

    def to_float(self) -> pa.Array:
        """Vectorized cast of the ``value`` sub-field to ``float64``."""
        return pc.cast(self._value_array(), pa.float64())

    def to_int(self, rounding: str = "round") -> pa.Array:
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
        return pc.cast(rounded, pa.int64())

    def is_zero(self) -> pa.Array:
        """Vectorized ``pc.equal(value, 0)`` predicate."""
        return pc.equal(self._value_array(), 0)

    def is_positive(self) -> pa.Array:
        """Vectorized ``pc.greater(value, 0)`` predicate."""
        return pc.greater(self._value_array(), 0)

    def is_negative(self) -> pa.Array:
        """Vectorized ``pc.less(value, 0)`` predicate."""
        return pc.less(self._value_array(), 0)

    # -- comparison mirrors -------------------------------------------------
    #
    # NB: Python ``__lt__`` / ``__le__`` / ``__gt__`` / ``__ge__`` MUST
    # return a bool — pa.Array cannot legally be returned from a dunder
    # without breaking ``sorted()``, ``list.sort()``, etc.  We therefore
    # split the surface: the parity-mirror dunders raise (see __init_subclass__
    # warning below), and the vectorized predicates live under explicit
    # names (``less_than``, ``less_equal``, etc.) plus a single ``compare``
    # entry-point.

    def less_than(self, other: TimeScalar | int | float | Fraction) -> pa.Array:
        """Vectorized ``pc.less(value, other)``."""
        return pc.less(self._value_array(), _scalar_compare_value(other))

    def less_equal(self, other: TimeScalar | int | float | Fraction) -> pa.Array:
        """Vectorized ``pc.less_equal(value, other)``."""
        return pc.less_equal(self._value_array(), _scalar_compare_value(other))

    def greater_than(self, other: TimeScalar | int | float | Fraction) -> pa.Array:
        """Vectorized ``pc.greater(value, other)``."""
        return pc.greater(self._value_array(), _scalar_compare_value(other))

    def greater_equal(self, other: TimeScalar | int | float | Fraction) -> pa.Array:
        """Vectorized ``pc.greater_equal(value, other)``."""
        return pc.greater_equal(self._value_array(), _scalar_compare_value(other))

    # Parity-mirror dunders.  Python protocol requires ``bool`` returns;
    # use the explicit ``less_than`` / ``less_equal`` / ``greater_than``
    # / ``greater_equal`` methods for vectorized comparisons.
    def __lt__(self, other: object) -> pa.Array:
        return self.less_than(other)  # type: ignore[arg-type]

    def __le__(self, other: object) -> pa.Array:
        return self.less_equal(other)  # type: ignore[arg-type]

    def __gt__(self, other: object) -> pa.Array:
        return self.greater_than(other)  # type: ignore[arg-type]

    def __ge__(self, other: object) -> pa.Array:
        return self.greater_equal(other)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        length = len(self) if not self.is_empty else 0
        return (
            f"{type(self).__name__}(name={self.name!r}, unit={self._unit}, "
            f"number_type={self._number_type}, len={length})"
        )


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

    def __index__(self) -> int:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return self.value
        raise TypeError(
            f"Cannot use Coordinate with {type(self.value).__name__} value as index. "
            f"Only integer Coordinates can be used as indices."
        )

    @property
    def semantic_type(self) -> str:
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


class CoordinateField(TimeScalarField):
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

    scalar_cls = Coordinate

    @property
    def semantic_type(self) -> str:
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
        struct, tl = _field_binop(self, other, "add", "add")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __sub__(self, other: object) -> CoordinateField | DurationField:
        struct, tl = _field_binop(self, other, "subtract", "subtract")
        # Two positions differ by an extent; everything else stays a position.
        if isinstance(other, (Coordinate, CoordinateField)):
            return _dur_field_from_value(struct, self._unit, self._number_type, tl)
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __mul__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = _field_binop(self, scalar, "multiply", "multiply")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __rmul__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = _field_binop(self, scalar, "divide", "divide")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    def __floordiv__(self, scalar: object) -> CoordinateField:
        """Scales a *position* — see Duration for tempo-style scaling of *extents*."""
        struct, tl = _field_binop(self, scalar, "floor_divide", "floor_divide")
        return _coord_field_from_value(struct, self._unit, self._number_type, tl)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
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

    @property
    def semantic_type(self) -> str:
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


class DurationField(TimeScalarField):
    """Semantic field for duration fields.

    Uses the same coordinate struct ``{value, numerator, denominator}``
    as :class:`CoordinateField`; the distinction is semantic.
    Pairs with :class:`Duration`.
    """

    scalar_cls = Duration

    @property
    def semantic_type(self) -> str:
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
        struct, tl = _field_binop(self, other, "add", "add")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __sub__(self, other: object) -> DurationField:
        if isinstance(other, (Coordinate, CoordinateField)):
            raise TypeError("Cannot subtract a Coordinate from a Duration field")
        struct, tl = _field_binop(self, other, "subtract", "subtract")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __mul__(self, scalar: object) -> DurationField:
        struct, tl = _field_binop(self, scalar, "multiply", "multiply")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __rmul__(self, scalar: object) -> DurationField:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: object) -> DurationField:
        struct, tl = _field_binop(self, scalar, "divide", "divide")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    def __floordiv__(self, scalar: object) -> DurationField:
        struct, tl = _field_binop(self, scalar, "floor_divide", "floor_divide")
        return _dur_field_from_value(struct, self._unit, self._number_type, tl)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
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
    field_type_name: ClassVar[str] = "CoordinateField"

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
                "IdCoordinate received conflicting positional and keyword arguments"
            )
        data = _apply_construction_protocol(
            {**positional, **data}, number_type=number_type, rounding=rounding
        )
        BaseModel.__init__(self, **data)

    @classmethod
    def from_coordinate(cls, coord: Coordinate, timeline_id: str) -> IdCoordinate:
        return cls(coord.value, coord.unit, timeline_id)

    def to_coordinate(self) -> Coordinate:
        return Coordinate(self.value, self.unit)

    def with_timeline(self, timeline_id: str) -> IdCoordinate:
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
        timeline_id: str,
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
        return self._timeline_id

    @property
    def semantic_type(self) -> str:
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
            pa.Array
            | pa.ChunkedArray
            | StructField
            | pa.Field
            | tuple[pa.Array | None, pa.Field]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        timeline_id: str | None = None,
        name: str | None = None,
    ) -> IdCoordinateField:
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
        return IdCoordinateField(self._raw, unit, self._number_type, self._timeline_id)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
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
    field_type_name: ClassVar[str] = "DurationField"

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
                "IdDuration received conflicting positional and keyword arguments"
            )
        data = _apply_construction_protocol(
            {**positional, **data}, number_type=number_type, rounding=rounding
        )
        BaseModel.__init__(self, **data)

    @classmethod
    def from_duration(cls, dur: Duration, timeline_id: str) -> IdDuration:
        return cls(dur.value, dur.unit, timeline_id)

    def to_duration(self) -> Duration:
        return Duration(self.value, self.unit)

    def with_timeline(self, timeline_id: str) -> IdDuration:
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
        timeline_id: str,
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
        return self._timeline_id

    @property
    def semantic_type(self) -> str:
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
            pa.Array
            | pa.ChunkedArray
            | StructField
            | pa.Field
            | tuple[pa.Array | None, pa.Field]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        timeline_id: str | None = None,
        name: str | None = None,
    ) -> IdDurationField:
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
        return IdDurationField(self._raw, unit, self._number_type, self._timeline_id)

    @classmethod
    def matches_pa_field(cls, pa_field: pa.Field) -> bool:
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


class IntervalField(SemanticField[Interval]):
    """Semantic field storing intervals with shared axis metadata.

    Args:
        raw: Struct field containing ``start`` and ``end`` coordinate structs.
        unit: Common endpoint unit.
        number_type: Common endpoint number type.
        source_fields: Optional blueprint source-field declaration.
    """

    scalar_cls = Interval

    def __init__(
        self,
        raw: StructField | None = None,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        *,
        source_fields: str | dict[str, Any] | None = None,
    ) -> None:
        if raw is None and source_fields is None:
            raise TypeError(
                "IntervalField requires either a raw field or source_fields="
            )
        super().__init__(raw, source_fields=source_fields)
        if raw is None:
            self._unit = None
            self._number_type = NumberType.float
            return
        resolved_unit, resolved_number_type = TimeScalarField._resolve_metadata(
            raw.field, unit, number_type
        )
        if resolved_unit is None:
            raise ValueError("IntervalField requires unit metadata or an explicit unit")
        self._unit = resolved_unit
        self._number_type = resolved_number_type

    @property
    def unit(self) -> TimeUnit:
        """Return the common endpoint unit."""
        return self._unit

    @property
    def domain(self) -> Domain:
        """Return the temporal domain implied by the unit."""
        return self._unit.domain

    @property
    def number_type(self) -> NumberType:
        """Return the common endpoint number type."""
        return self._number_type

    @property
    def semantic_type(self) -> str:
        """Return the paired scalar semantic type."""
        return "Interval"

    def metadata_dict(self) -> dict[str, str]:
        """Return semantic type and shared axis metadata."""
        return {
            **super().metadata_dict(),
            "unit": self.unit.value,
            "domain": self.domain.value,
            "number_type": self.number_type.name,
        }

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
        present = [value for value in intervals if value is not None]
        if present:
            inferred_unit = present[0].unit
            inferred_number_type = present[0].number_type
            if any(
                value.unit != inferred_unit or value.number_type != inferred_number_type
                for value in present
            ):
                raise ValueError(
                    "IntervalField values must share one unit and number_type"
                )
            resolved_unit = TimeUnit(unit) if isinstance(unit, str) else unit
            resolved_number_type = (
                NumberType(number_type) if isinstance(number_type, str) else number_type
            )
            if resolved_unit is not None and resolved_unit != inferred_unit:
                raise ValueError("Explicit IntervalField unit conflicts with values")
            if (
                resolved_number_type is not None
                and resolved_number_type != inferred_number_type
            ):
                raise ValueError(
                    "Explicit IntervalField number_type conflicts with values"
                )
            resolved_unit = inferred_unit
            resolved_number_type = inferred_number_type
        else:
            if unit is None or number_type is None:
                raise ValueError(
                    "Empty IntervalField input requires unit and number_type"
                )
            resolved_unit = TimeUnit(unit) if isinstance(unit, str) else unit
            resolved_number_type = (
                NumberType(number_type) if isinstance(number_type, str) else number_type
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
        array = pa.array(rows, type=interval_type)
        field = pa.field(name, interval_type)
        return cls(StructField(array, field), resolved_unit, resolved_number_type)

    @classmethod
    def from_field(
        cls,
        source: (
            pa.Array
            | pa.ChunkedArray
            | StructField
            | pa.Field
            | tuple[pa.Array | None, pa.Field]
        ),
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str = "interval",
    ) -> IntervalField:
        """Build an interval field from an Arrow or raw field source.

        Args:
            source: Supported Arrow field representation.
            unit: Common endpoint unit override.
            number_type: Common endpoint number type override.
            name: Field name for a bare array.

        Returns:
            A live or blueprint interval field.
        """
        if isinstance(source, tuple):
            data, field = source
            return cls(StructField(data, field), unit, number_type)
        if isinstance(source, pa.Field):
            return cls(StructField(None, source), unit, number_type)
        if isinstance(source, StructField):
            return cls(source, unit, number_type)
        if isinstance(source, (pa.Array, pa.ChunkedArray)):
            return cls(
                StructField(source, pa.field(name, source.type)), unit, number_type
            )
        raise TypeError(
            f"Unsupported source type for IntervalField.from_field: "
            f"{type(source).__name__}"
        )

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

    def _endpoint(self, name: str) -> CoordinateField:
        array = self.to_pyarrow()
        if isinstance(array, pa.ChunkedArray):
            array = array.combine_chunks()
        child = array.field(name)
        outer_valid = array.is_valid()
        child = pc.if_else(outer_valid, child, pa.scalar(None, type=child.type))
        return CoordinateField.from_field(
            child,
            unit=self.unit,
            number_type=self.number_type,
            name=name,
        )

    @property
    def start(self) -> CoordinateField:
        """Return the vectorized start-coordinate field."""
        return self._endpoint("start")

    @property
    def end(self) -> CoordinateField:
        """Return the vectorized end-coordinate field."""
        return self._endpoint("end")

    @property
    def duration(self) -> DurationField:
        """Return the vectorized duration field."""
        return self.end - self.start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_timeline_id(source: Any, override: str | None) -> str:
    """Resolve ``timeline_id`` from a kwarg override or field metadata."""
    if override is not None:
        if not isinstance(override, str) or not override:
            raise ValueError(
                f"timeline_id must be a non-empty string, got {override!r}"
            )
        return override

    pa_field: pa.Field | None = None
    if isinstance(source, tuple) and len(source) == 2:
        pa_field = source[1]
    elif isinstance(source, pa.Field):
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
