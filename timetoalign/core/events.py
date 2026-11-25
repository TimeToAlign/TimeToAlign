"""Event-level scalars and their paired SemanticField wrappers.

This is the catch-all module for TTA scalar types that are not the
fundamental Coordinate / Duration pair.  Each scalar is followed
immediately by its ``XField(SemanticField[X])`` class in the same file
— the paired Object + ObjectField is the unit of code organisation
and is never split across files (see Contributing → §2.4
Architectural decision log).

Sections (in file order):

* Graphical scalars + paired Fields (``Point``, the abstract ``Shape`` and
  its ``BoundingBox`` and ``Quadrilateral``)
* Pitch scalars + paired Fields (7 pairs)
* Harmony scalars + paired Fields (5 pairs) — plus the DCML import
  schemas (``DcmlStorageSchema`` etc.) and the ``Inversion`` enum.
* Event scalars + paired Fields (Note and its ``ScoreNote`` subclass,
  Measure, MeasureNumber, Id, EventReference, and the ``Event`` base of
  Projection and Box)
* MIDI event scalars + paired Fields (``MidiEvent`` base +
  ``ScoreMidiEvent`` subclass).
"""

from __future__ import annotations

import math
import re
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from fractions import Fraction
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import pyarrow as pa
import pyarrow.compute as pc
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)

from .enums import NumberType, TimeUnit
from .fields import (
    ScalarVocabulary,
    SemanticField,
    _dictionary_chunked_array,
    _is_dict,
    _joined_strings,
    _member_fields,
    _struct_array_from,
    _struct_if_else,
    blob_metadata,
    build_struct_array,
    data_shaped,
    drop_from_schema,
    register_value_projector,
)
from .ids import _OpenObject, quote_id, validate_id
from .protocols import TwelveTETPitchMixin
from .time import (
    Coordinate,
    Duration,
    SharedAxisField,
    TimeScalar,
    cell_coordinate,
    combine_number_columns,
    compare_number_column,
    coordinate_cell_projector,
    coordinate_cells,
    format_number,
    number_from_json,
    number_to_json,
    struct_to_coordinate,
    struct_to_rational,
)

if TYPE_CHECKING:
    import pandas as pd

#: A position or an extent: the two scalars a number of a shape or an event is.
_PositionOrExtentT = TypeVar("_PositionOrExtentT", Coordinate, Duration)


def _pc_mod(arr: pa.Array[Any], n: int) -> pa.Array[Any]:
    """Vectorized Python-semantics integer modulo (``a mod n``, n > 0).

    PyArrow does not expose a ``compute.mod`` function as of v23, and
    ``pc.divide`` on integers truncates toward zero (C-style) rather
    than flooring (Python-style).  To match Python's ``a % n``
    semantics on negative inputs:

    1. Truncated remainder: ``r = a - trunc(a / n) * n``
    2. Adjust to non-negative: ``((r + n) % n)`` via one more subtract.
       Equivalent: ``r_pos = r if r >= 0 else r + n``.

    Implemented as: ``r + n*(r < 0)`` via ``pc.add`` + ``pc.if_else``.
    """
    # TODO(pyarrow): replace with pc.mod once it lands — tracked upstream at
    # https://github.com/apache/arrow/issues/46901 (open as of pyarrow v23).
    quot = pc.divide(arr, n)
    r = pc.subtract(arr, pc.multiply(quot, n))
    # Adjust to non-negative when r is negative.
    return pc.if_else(pc.less(r, pa.scalar(0)), pc.add(r, n), r)


def _member(field: SemanticField[Any], name: str) -> pa.Array[Any]:
    """The struct member *name* of *field*, null on every null row.

    ``StructArray.field`` hands back the child array as stored, and a null
    row's children may hold anything (a loader writes ``""`` and ``0``
    there); ``pc.struct_field`` applies the row's validity.
    """
    return pc.struct_field(field.to_pyarrow(), name)


def _step_lookup(steps: pa.Array[Any], table: dict[str, int]) -> pa.Array[Any]:
    """Vectorized ``step name -> table[step]`` (int64); a null step stays null."""
    names = pa.array(list(table), type=pa.string())
    values = pa.array(list(table.values()), type=pa.int64())
    return pc.take(values, pc.index_in(steps, value_set=names))


def _struct_like(
    field: SemanticField[Any],
    arrays: list[pa.Array[Any]],
    fields: list[pa.Field[pa.DataType]],
) -> pa.StructArray:
    """A struct of *arrays* holding a null row wherever *field* holds one."""
    return _struct_array_from(arrays, fields, mask=field.to_pyarrow().is_null())


# ═══════════════════════════════════════════════════════════════════════════
# 1. PITCH — scalars and paired Fields
# ═══════════════════════════════════════════════════════════════════════════

# Step-name to semitone offset from C (C=0, D=2, E=4, F=5, G=7, A=9, B=11)
_STEP_TO_SEMITONE: dict[str, int] = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# Pitch-class to step name (for reverse mapping — natural notes only)
_PC_TO_STEP: dict[int, str] = {v: k for k, v in _STEP_TO_SEMITONE.items()}

# All 12 chromatic pitch class labels (for EPC display)
_PC_TO_LABEL: tuple[str, ...] = (
    "C",
    "C♯/D♭",
    "D",
    "D♯/E♭",
    "E",
    "F",
    "F♯/G♭",
    "G",
    "G♯/A♭",
    "A",
    "A♯/B♭",
    "B",
)

# Step-name to diatonic index (C=0, D=1, E=2, F=3, G=4, A=5, B=6)
_STEP_TO_GPC: dict[str, int] = {
    "C": 0,
    "D": 1,
    "E": 2,
    "F": 3,
    "G": 4,
    "A": 5,
    "B": 6,
}

# Base fifths for each step (unaltered)
_BASE_FIFTHS: dict[str, int] = {
    "F": -1,
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
}

# Regex for parsing pitch labels like "C#4", "Bb3", "D♯5", "F♭-1"
_PITCH_LABEL_RE = re.compile(
    r"^([A-Ga-g])"  # step letter
    r"([#♯]*|[b♭]*)"  # accidentals
    r"(-?\d+)?$"  # optional octave
)

_SPECIFIC_PITCH_STEPS = ("C", "D", "E", "F", "G", "A", "B")
_StepLiteral = Literal["C", "D", "E", "F", "G", "A", "B"]


def _is_step(value: str) -> TypeGuard[_StepLiteral]:
    """Whether *value* is one of the seven step letters, in upper case."""
    return value in _SPECIFIC_PITCH_STEPS


def _step_alter_to_fifths(step: str, alter: int) -> int:
    """Compute line-of-fifths position from step and alter."""
    return _BASE_FIFTHS.get(step, 0) + (7 * alter)


def _parse_pitch_label(label: str) -> tuple[_StepLiteral, int, int | None]:
    """Parse a pitch label string into (step, alter, octave_or_None)."""
    m = _PITCH_LABEL_RE.match(label.strip())
    if m is None or not _is_step(step := m.group(1).upper()):
        raise ValueError(f"Cannot parse pitch label: {label!r}")
    acc_str = m.group(2)
    octave_str = m.group(3)

    if not acc_str:
        alter = 0
    elif acc_str[0] in ("#", "♯"):
        alter = len(acc_str)
    else:
        alter = -len(acc_str)

    octave = int(octave_str) if octave_str is not None else None
    return step, alter, octave


def canonical_pitch_name(name: object) -> str:
    """Respell the accidentals of a pitch name with ``♯`` and ``♭``.

    ``"Bb3"`` becomes ``"B♭3"``, ``"F##4"`` ``"F♯♯4"`` and ``"C#"`` ``"C♯"``,
    the spelling :meth:`SpecificPitch.get` writes. The name is read as
    :meth:`SpecificPitch.from_string` reads it: whitespace around it is
    ignored and the rest must be one pitch name as a whole. Only the signs
    after the step letter change: the step, the octave and the surrounding
    whitespace stay as written (``"C-1"`` is C in octave -1 and stays,
    ``" Bb3"`` becomes ``" B♭3"``), and a string that is no pitch name
    (``""``, ``"rest"``, ``"G3, Bb4"``) is returned unchanged.

    Args:
        name: A pitch name with or without an octave.

    Returns:
        The name with every accidental written ``♯`` or ``♭``.

    Raises:
        TypeError: If ``name`` is not a string.
    """
    if not isinstance(name, str):
        raise TypeError(
            f"A pitch name is a string, not {type(name).__name__}: {name!r}"
        )
    core = name.strip()
    match = _PITCH_LABEL_RE.fullmatch(core)
    if match is None or not match.group(2):
        return name
    step, signs, octave = match.groups()
    sign = "♯" if signs[0] in ("#", "♯") else "♭"
    start, end = len(name) - len(name.lstrip()), len(name.rstrip())
    leading, trailing = name[:start], name[end:]
    return f"{leading}{step}{sign * len(signs)}{octave or ''}{trailing}"


def _respelled_cell(value: Any, sep: str | None) -> Any:
    """Respell one cell of a name column; a cell holding no string is kept."""
    if not isinstance(value, str):
        return value
    if sep is None:
        return canonical_pitch_name(value)
    return sep.join(canonical_pitch_name(part) for part in value.split(sep))


def canonical_pitch_names(names: pd.Series, *, sep: str | None = None) -> pd.Series:
    """Respell a column of pitch names with :func:`canonical_pitch_name`.

    Each distinct string is respelled once and the column is rewritten by
    looking its values up, so the cost follows the number of distinct names,
    not of rows. A null stays null, and the column keeps its type:

    * ``object``, ``str`` and the other string types: the same type, each
      name replaced by its respelling;
    * ``category``: a categorical of the respelled categories, in their
      order; two categories that respell to one name become one category;
    * an Arrow dictionary type: the same type, its dictionary respelled and
      merged in the same way.

    Args:
        names: The column of pitch names.
        sep: Where a cell lists several names (``"G3,D4,Bb4"`` with
            ``sep=","``), the separator; each listed name is respelled.

    Returns:
        A new column, the input left unchanged.
    """
    import pandas as pd

    dtype = names.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return _respelled_categorical(names, sep)
    if isinstance(dtype, pd.ArrowDtype) and pa.types.is_dictionary(dtype.pyarrow_dtype):
        return _respelled_dictionary(names, dtype.pyarrow_dtype, sep)
    respelled: dict[str, str] = {}
    for value in names.dropna().unique():
        new = _respelled_cell(value, sep)
        if isinstance(value, str) and new != value:
            respelled[value] = new
    result = names.copy()
    if respelled:
        changed = names.isin(list(respelled))
        result[changed] = names[changed].map(respelled)
    return result


def _respelled_categorical(names: pd.Series, sep: str | None) -> pd.Series:
    """Respell the categories of a categorical column, merging equal results."""
    import numpy as np
    import pandas as pd

    categories = names.cat.categories
    new = pd.Index(
        [_respelled_cell(value, sep) for value in categories], dtype=categories.dtype
    )
    merged = new.unique()
    positions = merged.get_indexer(new)
    codes = names.cat.codes.to_numpy()
    new_codes = np.where(codes >= 0, positions[codes] if len(positions) else codes, -1)
    respelled = pd.Categorical.from_codes(
        new_codes, dtype=pd.CategoricalDtype(merged, ordered=names.cat.ordered)
    )
    return pd.Series(respelled, index=names.index, name=names.name)


def _respelled_dictionary(
    names: pd.Series, dictionary_type: pa.DictionaryType[Any, Any], sep: str | None
) -> pd.Series:
    """Respell the dictionary of an Arrow dictionary column, merging equal results."""
    import pandas as pd

    arrow: pd.arrays.ArrowExtensionArray = names.array
    column = arrow.__arrow_array__()
    source = column.chunks
    if not _are_dictionary_chunks(source, column.type):
        raise TypeError(f"Expected a dictionary column, got {column.type}")
    value_type: pa.DataType = dictionary_type.value_type
    chunks: list[pa.DictionaryArray[Any, Any]] = []
    for chunk in source:
        new = [_respelled_cell(value, sep) for value in chunk.dictionary.to_pylist()]
        merged = list(dict.fromkeys(new))
        position = {value: i for i, value in enumerate(merged)}
        index_type: pa.DataType = chunk.indices.type
        lookup: pa.Array[Any] = pa.array(
            [position[value] for value in new], type=index_type
        )
        chunks.append(
            pa.DictionaryArray.from_arrays(
                pc.take(lookup, chunk.indices),
                pa.array(merged, type=value_type),
                ordered=dictionary_type.ordered,
            )
        )
    respelled = _dictionary_chunked_array(chunks, dictionary_type)
    return pd.Series(
        pd.arrays.ArrowExtensionArray(respelled), index=names.index, name=names.name
    )


def _are_dictionary_chunks(
    chunks: list[pa.Array[Any]], column_type: pa.DataType
) -> TypeGuard[list[pa.DictionaryArray[Any, Any]]]:
    """Whether *chunks*, the chunks of a column of *column_type*, are
    dictionary arrays: they are exactly when the column's type is a
    dictionary type. No chunk is inspected."""
    return pa.types.is_dictionary(column_type)


def _parse_octave_from_sp(sp: str | None, step: str) -> int:
    """Extract octave number from a specific-pitch string like ``"C4"``."""
    if not sp:
        return 4
    try:
        idx = len(sp)
        while idx > 0 and (sp[idx - 1].isdigit() or sp[idx - 1] == "-"):
            idx -= 1
        if idx < len(sp):
            return int(sp[idx:])
    except (ValueError, IndexError):
        pass
    return 4


# ---------------------------------------------------------------------------
# Point, Shape, BoundingBox, Quadrilateral + paired Fields
# ---------------------------------------------------------------------------

#: A number as a source writes it in text: an integer or a decimal.
_DECIMAL_RE = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")
_INTEGER_RE = re.compile(r"[+-]?\d+")


def _unwrapped(annotation: Any) -> Any:
    """*annotation* without an ``Optional`` wrapper."""
    if get_origin(annotation) in (Union, UnionType):
        arms = [arm for arm in get_args(annotation) if arm is not type(None)]
        if len(arms) == 1:
            return arms[0]
    return annotation


class _OnOneAxis:
    """A scalar whose numbers all lie on one axis, read from a memory row.

    The scalar's field stores each number as the number struct alone and
    the axis once, in its metadata (:class:`~timetoalign.core.time.SharedAxisField`);
    :meth:`from_row` puts the axis back.
    """

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any] | None,
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None,
    ) -> Any:
        """Construct from a memory row whose numbers lie on one axis.

        Each ``Coordinate`` or ``Duration`` property is read from its number
        struct in *number_type*, each nested scalar of one axis from its own
        struct, every other property as it stands. Keys the scalar does not
        have (an event table's ``event_type``) are not read.

        Args:
            row: The row, or ``None`` for a null cell.
            unit: The unit of every number.
            number_type: The number type every number is written in.

        Returns:
            The scalar, or ``None`` for a null cell.
        """
        if row is None:
            return None
        axis = TimeUnit(unit)
        written = axis.resolve_number_type(number_type)
        values: dict[str, Any] = {}
        for name, info in cls.model_fields.items():  # type: ignore[attr-defined]
            if name not in row:
                continue
            values[name] = _axis_value(
                _unwrapped(info.annotation), row[name], axis, written
            )
        return cls._built(**values)

    @classmethod
    def _built(cls, **values: Any) -> Any:
        """The scalar of *values*, as every constructor of a shape (and of a
        Box) makes it.

        A refusal of the scalar's validators is raised as the validator
        composed it, a ``ValueError`` holding the message alone, never inside
        pydantic's validation report; an input pydantic refuses before any
        validator runs (a missing or unknown property) keeps pydantic's.

        Raises:
            ValueError: The validator's refusal.
            ValidationError: Pydantic's refusal of the input.
        """
        try:
            return cls(**values)
        except ValidationError as error:
            for detail in error.errors():
                cause = detail.get("ctx", {}).get("error")
                if isinstance(cause, ValueError):
                    raise ValueError(*cause.args) from None
            raise


def _axis_value(kind: Any, cell: Any, unit: TimeUnit, number_type: NumberType) -> Any:
    """The property *kind* of one memory cell on the axis given."""
    if cell is None:
        return None
    if kind is Coordinate or kind is Duration:
        return kind(
            struct_to_coordinate(cell, number_type), unit, number_type=number_type
        )
    if isinstance(kind, type) and issubclass(kind, _OnOneAxis):
        return kind.from_row(cell, unit=unit, number_type=number_type)
    return cell


def _on_axis(
    value: Any,
    kind: type[_PositionOrExtentT],
    *,
    what: str,
    unit: TimeUnit | str | None,
    number_type: NumberType | str | None,
    owner: str = "A shape",
) -> _PositionOrExtentT:
    """*value* as a ``Coordinate`` or ``Duration`` (*kind*) of a shape.

    The one entry of every number into a shape (and of a Box's origin). A
    raw number takes *unit*; a scalar keeps its own, which *unit*, if given,
    must name. *number_type*, if given, decides how the number is written;
    else a scalar keeps its own and a raw number takes the unit's default.

    Geometry is finite and never rounded (:func:`_held_exactly`): NaN and
    infinities are refused on every axis; on an ``int`` axis a float is taken
    only when it is a whole number (``100.0`` is 100) and a ratio only when
    its denominator is 1; on a ``fraction`` axis a float is taken only when
    it is a whole number; on a ``float`` axis a ratio is taken only when it
    is a whole number, a float as given.

    Raises:
        TypeError: If a raw number comes without a unit.
        ValueError: If a scalar is in another unit than *unit*, or the axis
            cannot hold the value exactly.
    """
    if isinstance(value, TimeScalar):
        if unit is not None and value.unit != TimeUnit(unit):
            raise ValueError(
                f"{owner}'s {what} is in {value.unit}, not in {TimeUnit(unit)}: "
                "a shape is never converted."
            )
        axis = value.unit
        written = value.number_type if number_type is None else number_type
        number = value.value
    else:
        if unit is None:
            raise TypeError(
                f"{owner}'s {what} is the bare number {value!r}, which names no "
                "unit: pass unit=."
            )
        axis, written, number = TimeUnit(unit), number_type, value
    target = axis.resolve_number_type(written)
    held = _held_exactly(number, axis, target, f"{owner}'s {what}")
    return kind(held, axis, number_type=target)


def _held_exactly(
    number: Any, unit: TimeUnit, number_type: NumberType, what: str
) -> Any:
    """*number* as the axis *unit*, *number_type* holds it, never rounded.

    The strict entry of geometry, which every number of a shape and a Box's
    origin pass: a whole float or a whole ratio is read as its integer, so
    the scalar's construction protocol never rounds a shape's number nor
    turns it into another number type; any value the axis cannot hold as
    given is refused here, naming *what*, the value and the axis.

    Args:
        number: The value given.
        unit: The unit of the axis.
        number_type: The axis's number type, resolved.
        what: How the message names the value: ``"A shape's x"``.

    Returns:
        The value the scalar is made of.

    Raises:
        ValueError: If *number* is NaN or infinite (on every axis), a float
            that is no whole number on an ``int`` or ``fraction`` axis, or a
            ratio that is no whole number on an ``int`` or ``float`` axis.
    """
    if isinstance(number, bool):
        return number
    shown = format_number(number) if isinstance(number, (float, Fraction)) else ""
    if isinstance(number, float):
        if not math.isfinite(number):
            raise ValueError(
                f"{what} is {shown}, which is no finite number: geometry holds "
                "finite numbers only."
            )
        if number_type is NumberType.float:
            return number
        if number.is_integer():
            return int(number)
        reason = (
            "geometry is never rounded."
            if number_type is NumberType.int
            else "a float is read as a ratio only when it is a whole number; "
            "give a Fraction."
        )
    elif isinstance(number, Fraction):
        if number.denominator == 1:
            return number.numerator
        if number_type is NumberType.fraction:
            return number
        if number_type is NumberType.float:
            raise ValueError(
                f"{what} is {shown}, a ratio, which {unit} as float does not take: "
                "the declared number type governs and a ratio is read as a float "
                "only when it is a whole number; give a float."
            )
        reason = "geometry is never rounded."
    else:
        return number
    raise ValueError(
        f"{what} is {shown}, which {unit} as {number_type.name} cannot hold "
        f"exactly: {reason}"
    )


def _read_number(token: str, number_type: NumberType) -> int | float | Fraction:
    """A number a source writes in text, read exactly.

    An integer is an ``int``; a decimal is a ``float`` on a float axis and its
    exact ratio otherwise, so the scalar's construction protocol, not this
    reader, decides whether the axis can hold it.

    Raises:
        ValueError: If *token* is no number.
    """
    token = token.strip()
    if _INTEGER_RE.fullmatch(token):
        return int(token)
    if not _DECIMAL_RE.fullmatch(token):
        raise ValueError(f"{token!r} is no number")
    return float(token) if number_type is NumberType.float else Fraction(token)


def _whole_text(value: int | float | Fraction, refusal: str) -> str:
    """*value* written as a whole number, or *refusal* raised with it."""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
    elif isinstance(value, int) or value.is_integer():
        return str(int(value))
    raise ValueError(refusal.format(format_number(value)))


def _decimal_text(value: int | float | Fraction) -> str:
    """*value* written as a decimal number, exactly.

    Raises:
        ValueError: If *value* is a ratio without a finite decimal expansion.
    """
    if not isinstance(value, Fraction):
        return format_number(value)
    if value.denominator == 1:
        return str(value.numerator)
    rest, twos, fives = value.denominator, 0, 0
    while rest % 2 == 0:
        rest, twos = rest // 2, twos + 1
    while rest % 5 == 0:
        rest, fives = rest // 5, fives + 1
    if rest != 1:
        raise ValueError(
            f"SVG points are decimal numbers; {format_number(value)} has no finite "
            "decimal expansion."
        )
    places = max(twos, fives)
    digits = str(abs(value.numerator * 10**places // value.denominator)).rjust(
        places + 1, "0"
    )
    sign = "-" if value < 0 else ""
    return f"{sign}{digits[:-places]}.{digits[-places:]}".rstrip("0")


def _axis_text(scalar: TimeScalar) -> str:
    """How a message names the axis of *scalar*: unit and number type."""
    return f"{scalar.unit} as {scalar.number_type}"


def _held_on_their_axis(
    owner: BaseModel, members: list[tuple[str, TimeScalar]]
) -> None:
    """Refuse a member of *owner* its axis does not hold (:func:`_held_exactly`).

    A scalar made directly, not through a constructor's strict entry, is
    checked by its validator here: a NaN or an infinity is refused.
    """
    for name, scalar in members:
        _held_exactly(
            scalar.value,
            scalar.unit,
            scalar.number_type,
            f"A {type(owner).__name__}'s {name}",
        )


class Point(_OnOneAxis, ScalarVocabulary, BaseModel):
    """A point of an image: its ``x`` and ``y``, on one axis.

    Image space: the origin is the image's top-left corner, x runs right
    and y down. Both coordinates have one unit and one number type.

    Storage struct (derived): ``{x, y}``, each the number struct; the unit
    and number type are the field's (:class:`PointField`).

    Examples:
        >>> Point(x=Coordinate(1221, "pixels"), y=Coordinate(1589, "pixels"))
        Point(x=Coordinate(1221, pixels), y=Coordinate(1589, pixels))
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    x: Coordinate
    y: Coordinate

    @classmethod
    def of(
        cls,
        point: Any,
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> Point:
        """A point given as a ``Point`` or as a pair ``(x, y)``.

        Each member of a pair is a raw number (in *unit*), a ``Coordinate``
        or an ``IdCoordinate``.

        Raises:
            ValueError: If a pair does not have two members, or a member is
                a value the axis cannot hold exactly (never rounded).
        """
        if isinstance(point, Point):
            if unit is None and number_type is None:
                return point
            point = (point.x, point.y)
        x, y = point
        return cls._built(
            x=_on_axis(x, Coordinate, what="x", unit=unit, number_type=number_type),
            y=_on_axis(y, Coordinate, what="y", unit=unit, number_type=number_type),
        )

    @model_validator(mode="after")
    def _on_one_axis(self) -> Point:
        if (self.x.unit, self.x.number_type) != (self.y.unit, self.y.number_type):
            raise ValueError(
                f"A Point lies on one axis: its y is in {_axis_text(self.y)}, its x "
                f"in {_axis_text(self.x)}."
            )
        _held_on_their_axis(self, [("x", self.x), ("y", self.y)])
        return self

    def __repr__(self) -> str:
        return f"Point(x={self.x!r}, y={self.y!r})"

    def _text(self) -> str:
        """The point as a message writes it: ``(1221, 1589)``."""
        return f"({self.x._format_value()}, {self.y._format_value()})"


class PointField(SharedAxisField[Point]):
    """Paired Field for :class:`Point`: ``{x, y}`` on one declared axis."""


class Shape(_OnOneAxis, ScalarVocabulary, BaseModel):
    """An area of an image, on one axis: a :class:`BoundingBox` or a
    :class:`Quadrilateral`.

    The shape is in image space (origin the top-left corner, x right, y
    down), every member in one unit and one number type. Everything a shape
    answers is derived here from :attr:`corners`, its four corners in the
    canonical order ``top_left, top_right, bottom_right, bottom_left`` (the
    order SVG and IIIF selectors draw a polygon in).

    Every number enters a shape as given, finite and never rounded: NaN and
    the infinities are refused on every axis; on an ``int`` axis a float is
    taken only when it is a whole number (``100.0`` is 100) and a ratio only
    when its denominator is 1, on a ``fraction`` axis a float only when it
    is a whole number, on a ``float`` axis a ratio only when it is a whole
    number; anything else is refused, naming the value and the axis. A
    constructor's refusal is a ``ValueError`` holding its message alone.

    A shape is written as a format writes it (:meth:`as_media_fragment`,
    :meth:`as_svg_points`, :meth:`as_mei_zone`) and read back from it; a
    value a format cannot write is refused, never rounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    #: The keys of the shape's dict form, in order.
    _DICT_KEYS: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None,
    ) -> Shape:
        """The shape of a dict form (:meth:`to_dict`), on the axis given.

        Args:
            data: The dict; its keys decide the class.
            unit: The unit of every number.
            number_type: The number type the numbers are read in
                (:func:`~timetoalign.core.time.number_from_json`); the
                unit's default when ``None``.

        Returns:
            A :class:`BoundingBox` or a :class:`Quadrilateral`.

        Raises:
            ValueError: If the keys are neither class's, or name a class
                that is not *cls* or one of its subclasses.
        """
        keys = set(data)
        for kind in (BoundingBox, Quadrilateral):
            if keys == set(kind._DICT_KEYS):
                break
        else:
            raise ValueError(
                f"A shape's dict holds {sorted(keys)}: a BoundingBox's keys are "
                "x, y, width and height, a Quadrilateral's top_left, top_right, "
                "bottom_right and bottom_left."
            )
        if not issubclass(kind, cls):
            raise ValueError(f"The dict holds a {kind.__name__}, not a {cls.__name__}.")
        axis = TimeUnit(unit)
        return kind._from_dict_values(data, axis, axis.resolve_number_type(number_type))

    @classmethod
    def from_mei_zone(
        cls,
        attributes: dict[str, str],
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> Shape:
        """The shape an MEI ``zone`` states.

        With ``points``, a :class:`Quadrilateral` of those points (read by
        :meth:`Quadrilateral.from_svg_points`: four points of a convex
        outline in either direction from any corner, brought into the
        canonical order), whose bounding box must be the one ``ulx`` ...
        ``lry`` state where they are given; without, the
        :class:`BoundingBox` from ``ulx``, ``uly``, ``lrx``, ``lry``. Other
        attributes are not read.

        Args:
            attributes: The zone's attributes, as written.
            unit: The unit of the coordinates.
            number_type: Their number type; the unit's default when ``None``.

        Raises:
            ValueError: If the zone states no complete box and no points, its
                points are not four corners of a convex outline, or its
                points and its box name two areas.
        """
        axis = TimeUnit(unit)
        written = axis.resolve_number_type(number_type)
        names = ("ulx", "uly", "lrx", "lry")
        given = [attributes.get(name) for name in names]
        stated = [value for value in given if value is not None]
        box = None
        if len(stated) == len(names):
            box = BoundingBox.from_corners(
                *(_read_number(value, written) for value in stated),
                unit=axis,
                number_type=written,
            )
        points = attributes.get("points")
        if points is None:
            if box is None:
                missing = [name for name, value in zip(names, given) if value is None]
                raise ValueError(
                    f"An MEI zone without points states ulx, uly, lrx and lry; this "
                    f"one lacks {', '.join(missing)}."
                )
            return box
        quadrilateral = Quadrilateral.from_svg_points(
            points, unit=axis, number_type=written
        )
        if box is not None and box != quadrilateral.bounding_box:
            raise ValueError(
                f"The MEI zone's points {points!r} span "
                f"{quadrilateral.bounding_box.as_mei_zone()}, its ulx, uly, lrx and "
                f"lry {box.as_mei_zone()}: a zone states one area."
            )
        return quadrilateral

    @classmethod
    @abstractmethod
    def _from_dict_values(
        cls, data: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> Shape:
        """The shape of a dict form holding this class's keys."""

    @model_validator(mode="after")
    def _validate_shape(self) -> Shape:
        members = self._members()
        first_name, first = members[0]
        for name, scalar in members[1:]:
            if (scalar.unit, scalar.number_type) != (first.unit, first.number_type):
                raise ValueError(
                    f"A {type(self).__name__} lies on one axis: its {name} is in "
                    f"{_axis_text(scalar)}, its {first_name} in {_axis_text(first)}."
                )
        _held_on_their_axis(self, members)
        self._check()
        return self

    @property
    @abstractmethod
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """The four corners: top-left, top-right, bottom-right, bottom-left."""

    @property
    def unit(self) -> TimeUnit:
        """The unit of every member."""
        return self._members()[0][1].unit

    @property
    def number_type(self) -> NumberType:
        """The number type every member is written in."""
        return self._members()[0][1].number_type

    @property
    def x_min(self) -> Coordinate:
        """The smallest x of the corners."""
        return min(corner.x for corner in self.corners)

    @property
    def x_max(self) -> Coordinate:
        """The largest x of the corners."""
        return max(corner.x for corner in self.corners)

    @property
    def y_min(self) -> Coordinate:
        """The smallest y of the corners."""
        return min(corner.y for corner in self.corners)

    @property
    def y_max(self) -> Coordinate:
        """The largest y of the corners."""
        return max(corner.y for corner in self.corners)

    @property
    def bounding_box(self) -> BoundingBox:
        """The smallest box holding the shape."""
        return BoundingBox(
            x=self.x_min,
            y=self.y_min,
            width=self.x_max - self.x_min,
            height=self.y_max - self.y_min,
        )

    @abstractmethod
    def _members(self) -> list[tuple[str, TimeScalar]]:
        """Every number of the shape, named as a message names it."""

    def _check(self) -> None:  # noqa: B027 - a hook, empty by default
        """Refuse a shape its members do not describe."""

    @abstractmethod
    def _dict_values(self) -> dict[str, Any]:
        """The dict form: each number through ``number_to_json``."""

    def to_dict(self) -> dict[str, Any]:
        """The shape as a dict of bare numbers, in its axis's number type.

        Each number is written by
        :func:`~timetoalign.core.time.number_to_json`: an ``int`` as an
        integer, a ``float`` as a number, a ``Fraction`` as a pair. The
        keys decide the class: ``x, y, width, height`` for a
        :class:`BoundingBox`, the four corners, each ``{x, y}``, for a
        :class:`Quadrilateral`.

        Examples:
            >>> BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels").to_dict()
            {'x': 385, 'y': 46, 'width': 139, 'height': 20}
        """
        return self._dict_values()

    def as_media_fragment(self) -> str:
        """The shape's bounding box as a Media Fragments ``xywh`` selector.

        IIIF's image regions and selectors use the same form.

        Raises:
            ValueError: If a member of the bounding box is not a whole number.

        Examples:
            >>> BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels").as_media_fragment()
            'xywh=385,46,139,20'
        """
        box = self.bounding_box
        refusal = "A media fragment writes whole numbers; this shape's {} is {{}}."
        return "xywh=" + ",".join(
            _whole_text(scalar.value, refusal.format(name))
            for name, scalar in box._members()
        )

    def as_svg_points(self) -> str:
        """The four corners in the syntax of SVG's ``points``.

        ``x,y`` pairs in the canonical order, separated by spaces.

        Raises:
            ValueError: If a value is a ratio with no finite decimal
                expansion.

        Examples:
            >>> BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels").as_svg_points()
            '385,46 524,46 524,66 385,66'
        """
        return " ".join(
            f"{_decimal_text(corner.x.value)},{_decimal_text(corner.y.value)}"
            for corner in self.corners
        )

    def as_mei_zone(self) -> dict[str, str]:
        """The attributes of an MEI ``zone`` holding the shape.

        ``ulx``, ``uly``, ``lrx``, ``lry`` are the bounding box's corners
        (points, not pixel indices: ``lrx`` is ``x + width``); a
        :class:`Quadrilateral` adds its corners as ``points``.

        Raises:
            ValueError: If a corner of the bounding box is not a whole number,
                or a point has no finite decimal expansion.

        Examples:
            >>> BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels").as_mei_zone()
            {'ulx': '385', 'uly': '46', 'lrx': '524', 'lry': '66'}
        """
        refusal = "MEI zone coordinates are whole numbers; this shape's {} is {{}}."
        corners = {
            "ulx": self.x_min,
            "uly": self.y_min,
            "lrx": self.x_max,
            "lry": self.y_max,
        }
        zone = {
            name: _whole_text(scalar.value, refusal.format(name))
            for name, scalar in corners.items()
        }
        if not isinstance(self, BoundingBox):
            zone["points"] = self.as_svg_points()
        return zone


ShapeT = TypeVar("ShapeT", bound=Shape)


class ShapeField(SharedAxisField[ShapeT]):
    """The field of one kind of shape: every number on one declared axis.

    Each number is stored as the number struct alone, both sides filled; the
    unit and the number type are the field's metadata, as an
    :class:`~timetoalign.core.time.IntervalField` holds them.
    """

    @classmethod
    def from_shapes(
        cls,
        shapes: Sequence[ShapeT | None],
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str | None = None,
    ) -> ShapeField[ShapeT]:
        """Build the field of *shapes*, in row order.

        Args:
            shapes: Shapes of the field's class (``None`` for a null row).
            unit: The shapes' unit; required for an empty or all-null input.
            number_type: Their number type; required likewise.
            name: The field's name: ``bounding_box`` or ``quadrilateral`` when
                omitted.

        Returns:
            The field.

        Raises:
            TypeError: If a shape is not of the field's class.
            ValueError: If the shapes lie on two axes, or an explicit axis
                contradicts them.
        """
        return cls._from_scalars(shapes, unit, number_type, name)

    def x_min(self) -> pa.StructArray:
        """Per row the shape's smallest x, a number struct (null on a null row)."""
        return self._extreme(self._x_columns()[0], "less_equal")

    def x_max(self) -> pa.StructArray:
        """Per row the shape's largest x, a number struct (null on a null row)."""
        return self._extreme(self._x_columns()[1], "greater_equal")

    def _x_columns(self) -> tuple[list[pa.StructArray], list[pa.StructArray]]:
        """The columns whose least is ``x_min`` and those whose greatest is
        ``x_max``."""
        raise NotImplementedError(type(self).__name__)

    def _extreme(
        self, columns: list[pa.StructArray], comparison: str
    ) -> pa.StructArray:
        """Per row the cell of *columns* that *comparison* keeps, compared exactly."""
        result = columns[0]
        for column in columns[1:]:
            keep = compare_number_column(result, column, comparison, self.number_type)
            result = _struct_if_else(keep, result, column)
        return result


class BoundingBox(Shape):
    """An axis-aligned box of an image: ``x``, ``y``, ``width``, ``height``.

    The members of the Media Fragments and IIIF ``xywh`` region: the
    top-left corner and the extent, x right and y down. ``width`` and
    ``height`` are never negative (zero is kept). All four are on one axis.

    Storage struct (derived): ``{x, y, width, height}``, each the number
    struct; the axis is the field's (:class:`BoundingBoxField`).

    Examples:
        >>> box = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
        >>> box.x_max
        Coordinate(524, pixels)
    """

    _DICT_KEYS: ClassVar[tuple[str, ...]] = ("x", "y", "width", "height")

    x: Coordinate
    y: Coordinate
    width: Duration
    height: Duration

    @classmethod
    def from_xywh(
        cls,
        x: Any,
        y: Any,
        width: Any,
        height: Any,
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> BoundingBox:
        """The box at ``x``, ``y`` of extent ``width`` by ``height``.

        Args:
            x: The left edge: a raw number, a ``Coordinate`` or an
                ``IdCoordinate``.
            y: The top edge, likewise.
            width: The extent along x: a raw number or a ``Duration``.
            height: The extent along y, likewise.
            unit: The unit of the raw numbers; a scalar carries its own.
            number_type: How the numbers are written; a scalar keeps its
                own and a raw number takes the unit's default when omitted.

        Raises:
            TypeError: If a raw number comes without a unit.
            ValueError: If the members lie on two axes, an extent is
                negative, or a value is one the axis cannot hold exactly
                (``1.5`` on ``pixels``: never rounded).
        """
        return cls._built(
            x=_on_axis(x, Coordinate, what="x", unit=unit, number_type=number_type),
            y=_on_axis(y, Coordinate, what="y", unit=unit, number_type=number_type),
            width=_on_axis(
                width, Duration, what="width", unit=unit, number_type=number_type
            ),
            height=_on_axis(
                height, Duration, what="height", unit=unit, number_type=number_type
            ),
        )

    @classmethod
    def from_corners(
        cls,
        ulx: Any,
        uly: Any,
        lrx: Any,
        lry: Any,
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> BoundingBox:
        """The box from its upper-left and lower-right corners.

        The corners are points (as MEI's ``ulx`` ... ``lry`` and ATON's
        origin plus extent state them): ``width`` is ``lrx - ulx``.

        Raises:
            TypeError: If a raw number comes without a unit.
            ValueError: If the members lie on two axes, a lower-right
                coordinate lies before its upper-left one, or a value is one
                the axis cannot hold exactly (never rounded).
        """
        left = _on_axis(ulx, Coordinate, what="ulx", unit=unit, number_type=number_type)
        top = _on_axis(uly, Coordinate, what="uly", unit=unit, number_type=number_type)
        right = _on_axis(
            lrx, Coordinate, what="lrx", unit=unit, number_type=number_type
        )
        bottom = _on_axis(
            lry, Coordinate, what="lry", unit=unit, number_type=number_type
        )
        return cls._built(x=left, y=top, width=right - left, height=bottom - top)

    @classmethod
    def from_media_fragment(
        cls,
        text: str,
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> BoundingBox:
        """The box a Media Fragments ``xywh`` selector names.

        ``xywh=x,y,w,h`` and ``xywh=pixel:x,y,w,h`` are read (a leading
        ``#`` is allowed); ``xywh=percent:`` is refused, since it needs the
        image's size.

        Raises:
            ValueError: If *text* is no ``xywh`` selector or is a percent one.
        """
        match = re.fullmatch(
            r"#?xywh=(?:(pixel|percent):)?([^,]*),([^,]*),([^,]*),([^,]*)",
            text.strip(),
        )
        if match is None:
            raise ValueError(f"{text!r} is no xywh media fragment: write xywh=x,y,w,h.")
        if match.group(1) == "percent":
            raise ValueError(
                f"{text!r} is a percent media fragment, which needs the image's size: "
                "give the box in pixels (xywh=x,y,w,h)."
            )
        axis = TimeUnit(unit)
        written = axis.resolve_number_type(number_type)
        return cls.from_xywh(
            *(_read_number(token, written) for token in match.groups()[1:]),
            unit=axis,
            number_type=written,
        )

    @classmethod
    def _from_dict_values(
        cls, data: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> BoundingBox:
        return cls.from_xywh(
            *(number_from_json(data[key], number_type) for key in cls._DICT_KEYS),
            unit=unit,
            number_type=number_type,
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(x={self.x!r}, y={self.y!r}, width={self.width!r}, "
            f"height={self.height!r})"
        )

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """The four corners: top-left, top-right, bottom-right, bottom-left."""
        right, bottom = self.x + self.width, self.y + self.height
        return (
            Point(x=self.x, y=self.y),
            Point(x=right, y=self.y),
            Point(x=right, y=bottom),
            Point(x=self.x, y=bottom),
        )

    @property
    def x_min(self) -> Coordinate:
        """``x``."""
        return self.x

    @property
    def x_max(self) -> Coordinate:
        """``x + width``."""
        return self.x + self.width

    @property
    def y_min(self) -> Coordinate:
        """``y``."""
        return self.y

    @property
    def y_max(self) -> Coordinate:
        """``y + height``."""
        return self.y + self.height

    @property
    def bounding_box(self) -> BoundingBox:
        """The box itself."""
        return self

    def _members(self) -> list[tuple[str, TimeScalar]]:
        return [
            ("x", self.x),
            ("y", self.y),
            ("width", self.width),
            ("height", self.height),
        ]

    def _check(self) -> None:
        for name, extent in (("width", self.width), ("height", self.height)):
            if extent.is_negative():
                raise ValueError(
                    f"A BoundingBox's {name} is {extent._format_value()}; its width "
                    "and height are never negative."
                )

    def _dict_values(self) -> dict[str, Any]:
        return {name: number_to_json(scalar.value) for name, scalar in self._members()}


class BoundingBoxField(ShapeField[BoundingBox]):
    """Paired Field for :class:`BoundingBox`: ``{x, y, width, height}``."""

    def _x_columns(self) -> tuple[list[pa.StructArray], list[pa.StructArray]]:
        array = self.to_pyarrow()
        left = pc.struct_field(array, "x")
        right = combine_number_columns(
            left, pc.struct_field(array, "width"), "add", self.number_type
        )
        return [left], [right]


class Quadrilateral(Shape):
    """A four-cornered area of an image, its corners in the canonical order.

    ``top_left``, ``top_right``, ``bottom_right``, ``bottom_left``: the
    cyclic order SVG ``polygon`` and IIIF ``SvgSelector`` draw a box in
    (:data:`CORNER_ORDER`). Its sides may be skewed (a measure of a scanned
    page). All eight numbers are on one axis.

    The corners run clockwise (with y down) around a convex outline,
    starting at the top-left one: an outline whose sides cross is refused,
    so a polygon drawn from the corners is the area they bound. **The
    top-left corner is the corner of least x + y; of two such corners, the
    one of least y** (the upper). On an axis-aligned box it is the corner
    the name says; on a skewed one it is the corner nearest the image's
    origin along the diagonal. A list that runs clockwise but starts at
    another corner is refused by the constructor, saying which corner it
    starts at; :meth:`from_svg_points` reads a list in either direction
    from any corner and turns it into this order.

    For example, the diamond ``0,5 5,10 10,5 5,0`` (left, bottom, right,
    top: counter-clockwise with y down) has two corners of least
    ``x + y`` = 5, (0, 5) and (5, 0); the upper, (5, 0), is the top-left
    corner, so :meth:`from_svg_points` reverses the list and starts it
    there: ``5,0 10,5 5,10 0,5``.

    Storage struct (derived): ``{top_left, top_right, bottom_right,
    bottom_left}``, each ``{x, y}`` of number structs; the axis is the
    field's (:class:`QuadrilateralField`).

    Examples:
        >>> measure = Quadrilateral.from_points(
        ...     [(1221, 1589), (2151, 1589), (1221, 2266), (2161, 2266)],
        ...     order=("top_left", "top_right", "bottom_left", "bottom_right"),
        ...     unit="pixels",
        ... )
        >>> measure.as_svg_points()
        '1221,1589 2151,1589 2161,2266 1221,2266'
        >>> Quadrilateral.from_svg_points("0,5 5,10 10,5 5,0", unit="pixels").as_svg_points()
        '5,0 10,5 5,10 0,5'
    """

    CORNER_ORDER: ClassVar[tuple[str, str, str, str]] = (
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    )
    _DICT_KEYS: ClassVar[tuple[str, ...]] = CORNER_ORDER

    top_left: Point
    top_right: Point
    bottom_right: Point
    bottom_left: Point

    @classmethod
    def from_points(
        cls,
        points: Sequence[Any],
        *,
        order: Sequence[str] = CORNER_ORDER,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
    ) -> Quadrilateral:
        """The quadrilateral of four points, each given the role *order* names.

        The roles must hold: the corners, put in the canonical order, run
        clockwise around a convex outline from the top-left corner (the
        corner of least x + y; of two, the upper).

        Args:
            points: Four points, each a :class:`Point` or a pair ``(x, y)`` of
                raw numbers or ``Coordinate``s.
            order: The corner each point is, in the order given: a
                permutation of :data:`CORNER_ORDER` (DMOS writes a measure
                ``top_left, top_right, bottom_left, bottom_right``).
            unit: The unit of raw numbers; a scalar carries its own.
            number_type: How the numbers are written.

        Raises:
            ValueError: If *order* is not the four corner names, the number
                of points is not four, a value is one the axis cannot hold
                exactly (never rounded), the corners are not clockwise
                around a convex outline, or they are but the one named
                ``top_left`` is not the top-left corner.
        """
        roles = tuple(order)
        if sorted(roles) != sorted(cls.CORNER_ORDER):
            raise ValueError(
                f"The order {list(roles)} names each corner once: top_left, "
                "top_right, bottom_right and bottom_left."
            )
        given = list(points)
        if len(given) != 4:
            raise ValueError(f"A Quadrilateral has four corners; {len(given)} given.")
        return cls._built(
            **{
                role: Point.of(point, unit=unit, number_type=number_type)
                for role, point in zip(roles, given)
            }
        )

    @classmethod
    def from_svg_points(
        cls,
        text: str,
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> Quadrilateral:
        """The quadrilateral of an SVG ``points`` list, in the canonical order.

        SVG ``points`` (and MEI's ``@points``, IIIF's ``SvgSelector``) fix
        neither the corner a polygon starts at nor its direction, so four
        points of a convex outline are read in either direction from any
        corner and brought into the canonical order: a counter-clockwise
        list (y down) is reversed and the list is started at its top-left
        corner, the corner of least x + y (of two, the upper). A closed ring,
        the four points followed by the first again, is the same outline and
        reads as its four points.

        Raises:
            ValueError: If *text* does not hold four points (or a closed ring
                of them), or the points do not bound a convex quadrilateral
                (an outline turning both ways), naming the points where it
                turns.

        Examples:
            A measure listed counter-clockwise from its bottom-right corner:

            >>> Quadrilateral.from_svg_points(
            ...     "2161,2266 2151,1589 1221,1589 1221,2266", unit="pixels"
            ... ).as_svg_points()
            '1221,1589 2151,1589 2161,2266 1221,2266'
        """
        axis = TimeUnit(unit)
        written = axis.resolve_number_type(number_type)
        tokens = [token for token in re.split(r"[\s,]+", text.strip()) if token]
        if len(tokens) % 2:
            raise ValueError(
                f"The SVG points {text!r} hold {len(tokens)} numbers, which are no "
                "list of x,y pairs; a Quadrilateral's are four points."
            )
        refusal = ValueError(
            f"The SVG points {text!r} hold {len(tokens) // 2} points; a "
            "Quadrilateral's are four."
        )
        if len(tokens) not in (8, 10):
            raise refusal
        numbers = [_read_number(token, written) for token in tokens]
        pairs = list(zip(numbers[0::2], numbers[1::2]))
        if len(pairs) == 5:
            if pairs[4] != pairs[0]:
                raise refusal
            del pairs[4]  # a closed ring: the outline back at its first point
        corners = [Point.of(pair, unit=axis, number_type=written) for pair in pairs]
        turns = cls._turns(corners)
        clockwise = [index for index, turn in enumerate(turns) if turn > 0]
        counter = [index for index, turn in enumerate(turns) if turn < 0]
        if clockwise and counter:
            raise ValueError(
                f"The SVG points {text!r} bound no convex quadrilateral: the outline "
                f"turns clockwise at {_point_numbers(clockwise)} and "
                f"counter-clockwise at {_point_numbers(counter)}."
            )
        if counter:
            corners.reverse()
        start = cls._top_left_at(corners)
        return cls._built(
            **dict(zip(cls.CORNER_ORDER, corners[start:] + corners[:start]))
        )

    @classmethod
    def _from_dict_values(
        cls, data: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> Quadrilateral:
        points: list[tuple[int | float | Fraction, int | float | Fraction]] = []
        for name in cls.CORNER_ORDER:
            corner = data[name]
            if set(corner) != {"x", "y"}:
                raise ValueError(
                    f"A Quadrilateral's {name} holds {sorted(corner)}: a corner "
                    "holds x and y."
                )
            points.append(
                (
                    number_from_json(corner["x"], number_type),
                    number_from_json(corner["y"], number_type),
                )
            )
        return cls.from_points(points, unit=unit, number_type=number_type)

    def __repr__(self) -> str:
        return (
            "Quadrilateral("
            + ", ".join(
                f"{name}={corner._text()}"
                for name, corner in zip(self.CORNER_ORDER, self.corners)
            )
            + f", unit={self.unit})"
        )

    @property
    def corners(self) -> tuple[Point, Point, Point, Point]:
        """The four corners: top-left, top-right, bottom-right, bottom-left."""
        return (self.top_left, self.top_right, self.bottom_right, self.bottom_left)

    def _members(self) -> list[tuple[str, TimeScalar]]:
        return [
            (f"{name}.{member}", getattr(getattr(self, name), member))
            for name in self.CORNER_ORDER
            for member in ("x", "y")
        ]

    def _check(self) -> None:
        corners = self.corners
        listed = ", ".join(
            f"{corner_name} {corner._text()}"
            for corner_name, corner in zip(self.CORNER_ORDER, corners)
        )
        for name, turn in zip(self.CORNER_ORDER, self._turns(corners)):
            if turn < 0:
                raise ValueError(
                    f"The corners {listed} do not run clockwise around a convex "
                    f"outline: the turn at {name} is counter-clockwise."
                )
        start = self._top_left_at(corners)
        if corners[start] != corners[0]:
            raise ValueError(
                f"The corners {listed} run clockwise around a convex outline but "
                f"start at its {self.CORNER_ORDER[-start]} corner: its top-left "
                f"corner is {corners[start]._text()}, the corner of least x + y "
                "(of two, the upper)."
            )

    def _dict_values(self) -> dict[str, Any]:
        return {
            name: {
                "x": number_to_json(corner.x.value),
                "y": number_to_json(corner.y.value),
            }
            for name, corner in zip(self.CORNER_ORDER, self.corners)
        }

    @staticmethod
    def _turns(corners: Sequence[Point]) -> list[int | float | Fraction]:
        """At each corner, the turn of the outline drawn through *corners*.

        The cross product of the side arriving at the corner and the side
        leaving it, exact on an exact axis: positive where the outline turns
        clockwise (y down), negative where it turns counter-clockwise, zero
        where it runs straight on.
        """
        turns: list[int | float | Fraction] = []
        for index, here in enumerate(corners):
            before, after = corners[index - 1], corners[(index + 1) % len(corners)]
            turns.append(
                (here.x.value - before.x.value) * (after.y.value - here.y.value)
                - (here.y.value - before.y.value) * (after.x.value - here.x.value)
            )
        return turns

    @staticmethod
    def _top_left_at(corners: Sequence[Point]) -> int:
        """The index of the top-left corner among *corners*.

        The first corner of least ``x + y``; of two such corners, of least
        ``y``. Corners tying on both are one point (a box without width or
        height), any of which is the top-left corner.
        """
        keys = [(corner.x.value + corner.y.value, corner.y.value) for corner in corners]
        return keys.index(min(keys))


def _point_numbers(indices: Sequence[int]) -> str:
    """How a message names the points at *indices*: ``point 3``,
    ``points 1, 2 and 4`` (counted from 1)."""
    numbers = [str(index + 1) for index in indices]
    if len(numbers) == 1:
        return f"point {numbers[0]}"
    return f"points {', '.join(numbers[:-1])} and {numbers[-1]}"


class QuadrilateralField(ShapeField[Quadrilateral]):
    """Paired Field for :class:`Quadrilateral`: four corners of ``{x, y}``."""

    def _x_columns(self) -> tuple[list[pa.StructArray], list[pa.StructArray]]:
        array = self.to_pyarrow()
        columns = [
            pc.struct_field(pc.struct_field(array, name), "x")
            for name in Quadrilateral.CORNER_ORDER
        ]
        return columns, columns


# ---------------------------------------------------------------------------
# EnharmonicPitchClass + EnharmonicPitchClassField
# ---------------------------------------------------------------------------


class EnharmonicPitchClass(TwelveTETPitchMixin, BaseModel):
    """Enharmonic pitch class scalar -- chromatic pitch class (0-11).

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{pitch_class: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "EPC"

    pitch_class: int

    def __init__(
        self,
        pitch_class: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if pitch_class is not None:
            if "pitch_class" in data:
                raise TypeError(
                    "EnharmonicPitchClass received conflicting positional and "
                    "keyword arguments"
                )
            data = {"pitch_class": pitch_class, **data}
        super().__init__(**data)

    @property
    def semantic_type(self) -> str:
        return "EnharmonicPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "epc",
        }

    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "EnharmonicPitchClass":
        if target_type is EnharmonicPitchClass or target_type is type(self):
            return self
        raise TypeError(
            f"Cannot convert EnharmonicPitchClass to {target_type.__name__} "
            f"(octave and/or spelling information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        return str(self.pitch_class)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EnharmonicPitchClass | None:
        pc = row.get("pitch_class")
        if pc is None:
            return None
        return cls(pitch_class=int(pc))

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.pitch_class == other
        if isinstance(other, EnharmonicPitchClass):
            return self.pitch_class == other.pitch_class
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("EnharmonicPitchClass", self.pitch_class))

    def __str__(self) -> str:
        # Pretty label override: ``get()`` returns the bare pitch-class
        # integer (untouched -- its Field-vectorized mirror depends on it),
        # so ``str()`` mirrors the repr's inner token instead.
        if 0 <= self.pitch_class < 12:
            return _PC_TO_LABEL[self.pitch_class]
        return str(self.pitch_class)

    def __repr__(self) -> str:
        if 0 <= self.pitch_class < 12:
            return f"EPC({_PC_TO_LABEL[self.pitch_class]})"
        return f"EPC({self.pitch_class})"


class EnharmonicPitchClassField(SemanticField[EnharmonicPitchClass]):
    """Columnar wrapper for ``EnharmonicPitchClass`` (paired Field)."""

    def _pc_array(self) -> pa.Array[Any]:
        return _member(self, "pitch_class")

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized cast of ``pitch_class`` to string."""
        return self._pc_array().cast(pa.string())


# ---------------------------------------------------------------------------
# GenericPitchClass + GenericPitchClassField
# ---------------------------------------------------------------------------

_GPC_NAMES: tuple[str, ...] = ("C", "D", "E", "F", "G", "A", "B")


class GenericPitchClass(TwelveTETPitchMixin, BaseModel):
    """Generic pitch class scalar -- diatonic step (0-6).

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "GPC"

    step: int

    def __init__(
        self,
        step: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if step is not None:
            if "step" in data:
                raise TypeError(
                    "GenericPitchClass received conflicting positional and "
                    "keyword arguments"
                )
            data = {"step": step, **data}
        super().__init__(**data)

    @property
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.step

    @property
    def semantic_type(self) -> str:
        return "GenericPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "gpc",
        }

    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "GenericPitchClass":
        if target_type is GenericPitchClass or target_type is type(self):
            return self
        raise TypeError(
            f"Cannot convert GenericPitchClass to {target_type.__name__} "
            f"(spelling or octave information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if 0 <= self.step < 7:
            return _GPC_NAMES[self.step]
        return str(self.step)

    def to_dict(self) -> dict[str, object]:
        return {"pitch_class": self.step}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenericPitchClass | None:
        pc = row.get("pitch_class")
        if pc is None:
            pc = row.get("step")
        if pc is None:
            return None
        return cls(step=int(pc))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.step == other
        if isinstance(other, GenericPitchClass):
            return self.step == other.step
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("GenericPitchClass", self.step))


class GenericPitchClassField(SemanticField[GenericPitchClass]):
    """Columnar wrapper for ``GenericPitchClass`` (paired Field)."""

    def _step_array(self) -> pa.Array[Any]:
        return _member(self, "step")

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized step-index → step-name lookup via a take table."""
        step = self._step_array()
        lookup = pa.array(list(_GPC_NAMES), type=pa.string())
        return pc.take(lookup, step.cast(pa.int64()))


# ---------------------------------------------------------------------------
# GenericPitch + GenericPitchField
# ---------------------------------------------------------------------------


class GenericPitch(TwelveTETPitchMixin, BaseModel):
    """Generic pitch scalar -- diatonic step + octave.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: int64, octave: int64}``.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "GP"

    step: int
    octave: int

    def __init__(
        self,
        step: int | None = None,
        octave: int | None = None,
        /,
        **data: Any,
    ) -> None:
        positional = {
            k: v for k, v in (("step", step), ("octave", octave)) if v is not None
        }
        if positional and any(k in data for k in positional):
            raise TypeError(
                "GenericPitch received conflicting positional and keyword arguments"
            )
        data = {**positional, **data}
        super().__init__(**data)

    @property
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.step

    @property
    def semantic_type(self) -> str:
        return "GenericPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "gp",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is GenericPitch or target_type is type(self):
            return self
        if target_type is GenericPitchClass:
            return GenericPitchClass(step=self.step)
        raise TypeError(
            f"Cannot convert GenericPitch to {target_type.__name__} "
            f"(accidental information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if 0 <= self.step < 7:
            return f"{_GPC_NAMES[self.step]}{self.octave}"
        return f"step={self.step}, oct={self.octave}"

    def to_dict(self) -> dict[str, object]:
        return {"pitch_class": self.step, "octave": self.octave}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenericPitch | None:
        pc = row.get("pitch_class")
        if pc is None:
            pc = row.get("step")
        if pc is None:
            return None
        octave = row.get("octave", 4)
        return cls(step=int(pc), octave=int(octave))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericPitch):
            return self.step == other.step and self.octave == other.octave
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("GenericPitch", self.step, self.octave))


class GenericPitchField(SemanticField[GenericPitch]):
    """Columnar wrapper for ``GenericPitch`` (paired Field)."""

    def _step_array(self) -> pa.Array[Any]:
        return _member(self, "step")

    def _octave_array(self) -> pa.Array[Any]:
        return _member(self, "octave")

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``GenericPitch.to(target_scalar_cls)``.

        Supports ``GenericPitch`` (identity) and ``GenericPitchClass``
        (drops the ``octave`` sub-field).
        """
        if target_scalar_cls is GenericPitch:
            return self
        if target_scalar_cls is GenericPitchClass:
            step = self._step_array()
            new_struct = _struct_like(
                self,
                [step.cast(pa.int64())],
                [pa.field("step", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("step", new_struct.type)
            return GenericPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert GenericPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized step-name + octave string concatenation."""
        step = self._step_array()
        octave = self._octave_array()
        lookup = pa.array(list(_GPC_NAMES), type=pa.string())
        step_str = pc.take(lookup, step.cast(pa.int64()))
        octave_str = octave.cast(pa.string())
        return _joined_strings(step_str, octave_str, "")


# ---------------------------------------------------------------------------
# SpecificPitchClass + SpecificPitchClassField
# ---------------------------------------------------------------------------


class SpecificPitchClass(TwelveTETPitchMixin, BaseModel):
    """Pitch class with spelling.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{step: string, alter: int64}``.  ``fifths`` and ``pitch_class`` are
    derived (not stored).
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "SPC"

    step: _StepLiteral
    alter: int = 0

    def __init__(
        self,
        step: str | None = None,
        alter: int | None = None,
        /,
        **data: Any,
    ) -> None:
        positional = {
            k: v for k, v in (("step", step), ("alter", alter)) if v is not None
        }
        if positional and any(k in data for k in positional):
            raise TypeError(
                "SpecificPitchClass received conflicting positional and "
                "keyword arguments"
            )
        data = {**positional, **data}
        super().__init__(**data)

    @field_validator("step", mode="before")
    @classmethod
    def _normalise_step(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(f"step must be a string, got {type(v).__name__}")
        upper = v.upper()
        if upper not in _SPECIFIC_PITCH_STEPS:
            raise ValueError(f"step must be one of {_SPECIFIC_PITCH_STEPS}, got {v!r}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    @data_shaped
    def fifths(self) -> int:
        """Position on the line of fifths (derived from step + alter)."""
        return _step_alter_to_fifths(self.step, self.alter)

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (base + self.alter) % 12

    @property
    def semantic_type(self) -> str:
        return "SpecificPitchClass"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "spc",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "SpecificPitchClass | EnharmonicPitchClass":
        if target_type is SpecificPitchClass or target_type is type(self):
            return self
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        raise TypeError(
            f"Cannot convert SpecificPitchClass to {target_type.__name__} "
            f"(octave information required)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        alter_str = ""
        if self.alter > 0:
            alter_str = "♯" * self.alter
        elif self.alter < 0:
            alter_str = "♭" * abs(self.alter)
        return f"{self.step}{alter_str}"

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "alter": self.alter,
            "pitch_class": self.pitch_class,
            "fifths": self.fifths,
            "label": self.get(),
        }

    @classmethod
    def from_string(cls, value: str) -> SpecificPitchClass:
        """Construct a spelled pitch class from its written string.

        Args:
            value: Pitch-class string such as ``"C♯"`` or ``"D♭"``.

        Returns:
            The parsed specific pitch class.
        """
        step, alter, _ = _parse_pitch_label(value)
        return cls(step=step, alter=alter)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SpecificPitchClass | None:
        step = row.get("step") or row.get("gpc_str")
        if step is None:
            return None
        alter = row.get("alter")
        if alter is None:
            alter = row.get("acc", 0)
        return cls(
            step=str(step),
            alter=int(alter or 0),
        )


class SpecificPitchClassField(SemanticField[SpecificPitchClass]):
    """Columnar wrapper for ``SpecificPitchClass`` (paired Field)."""

    def _step_array(self) -> pa.Array[Any]:
        return _member(self, "step")

    def _alter_array(self) -> pa.Array[Any]:
        return _member(self, "alter")

    def _step_to_semitone(self) -> pa.Array[Any]:
        """Vectorized ``step`` → semitone offset from C (int64), null on a null row."""
        return _step_lookup(self._step_array(), _STEP_TO_SEMITONE)

    def _step_to_base_fifths(self) -> pa.Array[Any]:
        """Vectorized ``step`` → line-of-fifths position of the natural (int64)."""
        return _step_lookup(self._step_array(), _BASE_FIFTHS)

    @property
    def pitch_class(self) -> pa.Array[Any]:
        """Vectorized ``(step_semi + alter) mod 12``."""
        semi = self._step_to_semitone()
        alter = self._alter_array().cast(pa.int64())
        return _pc_mod(pc.add(semi, alter), 12)

    @property
    def fifths(self) -> pa.Array[Any]:
        """Vectorized line-of-fifths position (``base_fifths + 7*alter``)."""
        base = self._step_to_base_fifths()
        alter = self._alter_array().cast(pa.int64())
        return pc.add(base, pc.multiply(alter, 7))

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``SpecificPitchClass.to(target_scalar_cls)``.

        Supports identity, and ``EnharmonicPitchClass`` (collapses spelling
        to the chromatic pitch-class).
        """
        if target_scalar_cls is SpecificPitchClass:
            return self
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = _struct_like(
                self,
                [pc_arr.cast(pa.int64())],
                [pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert SpecificPitchClass field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized step + accidental concatenation.

        Implementation note: per-element accidental construction via
        ``pc.compute`` (no scalar materialisation loop) using
        ``pc.if_else`` over the small fixed alter range observed in
        musical practice (``-2..+2``).
        """
        step = self._step_array()
        alter = self._alter_array().cast(pa.int64())
        # Build accidental string per row: positive → "♯" * alter; negative
        # → "♭" * |alter|; zero → "".  Bounded alter range allows a
        # case-by-case if_else chain.
        out = pa.array([""] * len(step), type=pa.string())
        for n in (-2, -1, 1, 2):
            if n > 0:
                marker = "♯" * n
            else:
                marker = "♭" * abs(n)
            mask = pc.equal(alter, pa.scalar(n))
            out = pc.if_else(mask, pa.scalar(marker, type=pa.string()), out)
        return pc.binary_join_element_wise(step, out, "")


# ---------------------------------------------------------------------------
# EnharmonicPitch + EnharmonicPitchField
# ---------------------------------------------------------------------------

# Note name labels for enharmonic display (sharp spelling for black keys)
_EP_LABELS: tuple[str, ...] = (
    "C",
    "C♯",
    "D",
    "E♭",
    "E",
    "F",
    "F♯",
    "G",
    "A♭",
    "A",
    "B♭",
    "B",
)


class EnharmonicPitch(TwelveTETPitchMixin, BaseModel):
    """Enharmonic pitch scalar -- MIDI pitch with pitch-name display.

    Pydantic v2 ``BaseModel``, frozen.  Storage struct (derived):
    ``{midi_number: int64}``.  ``pitch_class`` and ``octave`` are
    properties, not stored.
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "EP"

    midi_number: int

    def __init__(
        self,
        midi_number: int | None = None,
        /,
        **data: Any,
    ) -> None:
        if midi_number is not None:
            if "midi_number" in data:
                raise TypeError(
                    f"{type(self).__name__} received conflicting positional and "
                    "keyword arguments"
                )
            data = {"midi_number": midi_number, **data}
        super().__init__(**data)

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        return self.midi_number % 12

    @property
    @data_shaped
    def octave(self) -> int:
        return (self.midi_number // 12) - 1

    @property
    def semantic_type(self) -> str:
        return "EnharmonicPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "ep",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is EnharmonicPitch or target_type is type(self):
            return self
        if target_type is MidiPitch:
            return MidiPitch(midi_number=self.midi_number)
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        raise TypeError(
            f"Cannot convert {type(self).__name__} to {target_type.__name__} "
            f"(spelling information required for specific pitch types)"
        )

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format == "midi":
            return str(self.midi_number)
        return f"{_EP_LABELS[self.pitch_class]}{self.octave}"

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EnharmonicPitch | None:
        if "midi_number" in row and row.get("midi_number") is not None:
            return cls(midi_number=int(row["midi_number"]))
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, EnharmonicPitch):
            return self.midi_number == other.midi_number
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self).__name__, self.midi_number))

    def __str__(self) -> str:
        return f"{_PC_TO_LABEL[self.pitch_class]}{self.octave}"

    def __repr__(self) -> str:
        return f"EP({_PC_TO_LABEL[self.pitch_class]}{self.octave})"


class EnharmonicPitchField(SemanticField[EnharmonicPitch]):
    """Columnar wrapper for ``EnharmonicPitch`` (paired Field).

    The on-disk struct shape is the pydantic-derived
    ``{midi_number: int64}``; cached on the class as ``pa_schema``.
    """

    def _midi_array(self) -> pa.Array[Any]:
        return _member(self, "midi_number")

    @property
    def pitch_class(self) -> pa.Array[Any]:
        """Vectorized ``midi_number mod 12``."""
        return _pc_mod(self._midi_array().cast(pa.int64()), 12)

    @property
    def octave(self) -> pa.Array[Any]:
        """Vectorized ``(midi_number // 12) - 1``."""
        return pc.subtract(pc.divide(self._midi_array().cast(pa.int64()), 12), 1)

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``EnharmonicPitch.to(target_scalar_cls)``.

        Supports identity, ``MidiPitch`` (metadata-only retype), and
        ``EnharmonicPitchClass`` (``pc.mod(midi_number, 12)``).
        """
        if target_scalar_cls is EnharmonicPitch:
            return self
        if target_scalar_cls is MidiPitch:
            # Metadata-only retype: same struct, different scalar_cls.
            arr = self.to_pyarrow()
            new_pa_field = pa.field(self.name, arr.type)
            return MidiPitchField.from_field((arr, new_pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = _struct_like(
                self,
                [pc_arr.cast(pa.int64())],
                [pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert EnharmonicPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized label + octave string concatenation.

        Implementation: pitch-class → label via ``pc.take`` from the
        12-element ``_EP_LABELS`` lookup table, then concat with the
        string-cast octave.
        """
        if format == "midi":
            return self._midi_array().cast(pa.string())
        pc_arr = self.pitch_class
        labels = pa.array(list(_EP_LABELS), type=pa.string())
        label_str = pc.take(labels, pc_arr.cast(pa.int64()))
        octave_str = self.octave.cast(pa.string())
        return _joined_strings(label_str, octave_str, "")


# ---------------------------------------------------------------------------
# MidiPitch + MidiPitchField
# ---------------------------------------------------------------------------


class MidiPitch(EnharmonicPitch):
    """Display alias of :class:`EnharmonicPitch` reserved for ``MidiPitchField``.

    Identical data, storage struct, conversions, and protocol conformance.
    Differs only in ``__repr__``, ``semantic_type``, and the default
    ``get()`` format.
    """

    _REPR_ABBR: ClassVar[str] = "MP"

    @property
    def semantic_type(self) -> str:
        return "MidiPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "ep",
        }

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format is None or format == "midi":
            return str(self.midi_number)
        return super().get(format=format)

    def __str__(self) -> str:
        return str(self.midi_number)

    def __repr__(self) -> str:
        return f"MP({self.midi_number})"


class MidiPitchField(SemanticField[MidiPitch]):
    """Columnar wrapper for ``MidiPitch`` (paired Field).

    Same struct as ``EnharmonicPitchField`` (``MidiPitch`` is a subclass
    of ``EnharmonicPitch``), but with distinct ``scalar_cls`` so
    ``__getitem__`` yields ``MidiPitch`` instances.
    """

    def _midi_array(self) -> pa.Array[Any]:
        return _member(self, "midi_number")

    @property
    def pitch_class(self) -> pa.Array[Any]:
        return _pc_mod(self._midi_array().cast(pa.int64()), 12)

    @property
    def octave(self) -> pa.Array[Any]:
        return pc.subtract(pc.divide(self._midi_array().cast(pa.int64()), 12), 1)

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``EnharmonicPitch.to`` for the MidiPitch view."""
        if target_scalar_cls is MidiPitch:
            return self
        if target_scalar_cls is EnharmonicPitch:
            arr = self.to_pyarrow()
            new_pa_field = pa.field(self.name, arr.type)
            return EnharmonicPitchField.from_field((arr, new_pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class
            new_struct = _struct_like(
                self,
                [pc_arr.cast(pa.int64())],
                [pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert MidiPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized cast of ``midi_number`` to string (default format)."""
        if format is None or format == "midi":
            return self._midi_array().cast(pa.string())
        # Delegate to EP-style formatting when an explicit non-midi format is asked
        pc_arr = self.pitch_class
        labels = pa.array(list(_EP_LABELS), type=pa.string())
        label_str = pc.take(labels, pc_arr.cast(pa.int64()))
        octave_str = self.octave.cast(pa.string())
        return _joined_strings(label_str, octave_str, "")


# ---------------------------------------------------------------------------
# SpecificPitch + SpecificPitchField
# ---------------------------------------------------------------------------


class SpecificPitch(TwelveTETPitchMixin, BaseModel):
    """Specific pitch scalar with full enharmonic identity.

    Storage struct (derived): ``{step: string, alter: int64, octave: int64,
    cents: float64?}``.  ``fifths``, ``midi_number``, ``pitch_class`` are
    derived (not stored).
    """

    model_config = ConfigDict(frozen=True)

    _REPR_ABBR: ClassVar[str] = "SP"

    step: _StepLiteral
    alter: int = 0
    octave: int
    cents: float | None = None

    @field_validator("step", mode="before")
    @classmethod
    def _normalise_step(cls, v: object) -> str:
        if not isinstance(v, str):
            raise TypeError(f"step must be a string, got {type(v).__name__}")
        upper = v.upper()
        if upper not in _SPECIFIC_PITCH_STEPS:
            raise ValueError(f"step must be one of {_SPECIFIC_PITCH_STEPS}, got {v!r}")
        return upper

    @computed_field  # type: ignore[prop-decorator]
    @property
    @data_shaped
    def fifths(self) -> int:
        return _step_alter_to_fifths(self.step, self.alter)

    @property
    @data_shaped
    def midi_number(self) -> int:
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (self.octave + 1) * 12 + base + self.alter

    @property
    @data_shaped
    def pitch_class(self) -> int:  # type: ignore[override]
        base = _STEP_TO_SEMITONE.get(self.step, 0)
        return (base + self.alter) % 12

    @property
    def semantic_type(self) -> str:
        return "SpecificPitch"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "pitch_type": "sp",
        }

    @data_shaped
    def to(
        self, target_type: type, *, format: str | None = None
    ) -> "TwelveTETPitchMixin":
        if target_type is SpecificPitch or target_type is type(self):
            return self
        if target_type is MidiPitch:
            return MidiPitch(midi_number=self.midi_number)
        if target_type is EnharmonicPitchClass:
            return EnharmonicPitchClass(pitch_class=self.pitch_class)
        if target_type is SpecificPitchClass:
            return SpecificPitchClass(step=self.step, alter=self.alter)
        raise TypeError(f"Cannot convert SpecificPitch to {target_type.__name__}")

    @data_shaped
    def get(self, *, format: str | None = None) -> str:
        if format == "midi":
            return str(self.midi_number)
        alter_str = ""
        if self.alter > 0:
            alter_str = "♯" * self.alter
        elif self.alter < 0:
            alter_str = "♭" * abs(self.alter)
        return f"{self.step}{alter_str}{self.octave}"

    def to_dict(self) -> dict[str, object]:
        """Return the declared and computed fields plus the derived labels."""
        return {
            **self.model_dump(),
            "midi_number": self.midi_number,
            "pitch_class": self.pitch_class,
            "label": self.get(),
        }

    @classmethod
    def from_string(cls, value: str) -> SpecificPitch:
        """Construct a spelled pitch from its written string.

        Args:
            value: Pitch string including its octave, such as ``"C♯4"``.

        Returns:
            The parsed specific pitch.

        Raises:
            ValueError: If the pitch string has no octave.
        """
        step, alter, octave = _parse_pitch_label(value)
        if octave is None:
            raise ValueError(
                f"Octave required for SpecificPitch, got {value!r}. "
                f"Use SpecificPitchClass.from_string() for octave-free pitches."
            )
        return cls(step=step, alter=alter, octave=octave)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "SpecificPitch | None":
        """Construct from a PyArrow struct row dict (trust-boundary regime).

        Accepts the canonical struct ``{step, alter, octave, cents}``.
        """
        if "step" in row and row.get("step") is not None:
            cents_raw = row.get("cents")
            cents = float(cents_raw) if cents_raw is not None else None
            return cls.model_validate(
                {
                    "step": str(row["step"]),
                    "alter": int(row.get("alter", 0) or 0),
                    "octave": int(row["octave"]),
                    "cents": cents,
                }
            )
        return None


class SpecificPitchField(SemanticField[SpecificPitch]):
    """Columnar wrapper for ``SpecificPitch`` (paired Field).

    Pa schema: ``{step: string, alter: int64, octave: int64, cents: float64}``.
    """

    def _step_array(self) -> pa.Array[Any]:
        return _member(self, "step")

    def _alter_array(self) -> pa.Array[Any]:
        return _member(self, "alter")

    def _octave_array(self) -> pa.Array[Any]:
        return _member(self, "octave")

    def _step_to_semitone(self) -> pa.Array[Any]:
        """Vectorized ``step`` → semitone offset from C (int64), null on a null row."""
        return _step_lookup(self._step_array(), _STEP_TO_SEMITONE)

    def _step_to_base_fifths(self) -> pa.Array[Any]:
        """Vectorized ``step`` → line-of-fifths position of the natural (int64)."""
        return _step_lookup(self._step_array(), _BASE_FIFTHS)

    @property
    def pitch_class(self) -> pa.Array[Any]:
        """Vectorized ``(step_semi + alter) mod 12``."""
        semi = self._step_to_semitone()
        alter = self._alter_array().cast(pa.int64())
        return _pc_mod(pc.add(semi, alter), 12)

    @property
    def midi_number(self) -> pa.Array[Any]:
        """Vectorized ``(octave + 1) * 12 + step_semi + alter``."""
        semi = self._step_to_semitone()
        alter = self._alter_array().cast(pa.int64())
        octave = self._octave_array().cast(pa.int64())
        return pc.add(pc.add(pc.multiply(pc.add(octave, 1), 12), semi), alter)

    @property
    def fifths(self) -> pa.Array[Any]:
        """Vectorized line-of-fifths position (``base_fifths + 7*alter``)."""
        base = self._step_to_base_fifths()
        alter = self._alter_array().cast(pa.int64())
        return pc.add(base, pc.multiply(alter, 7))

    def convert_to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Vectorized mirror of ``SpecificPitch.to(target_scalar_cls)``.

        Supports identity, ``MidiPitch``, ``EnharmonicPitchClass``,
        ``SpecificPitchClass`` — matching the scalar method's supported
        targets.  ``EnharmonicPitch`` is intentionally NOT supported
        here (the scalar ``to`` does not support it either; users go
        through ``MidiPitch`` as a thin alias).
        """
        if target_scalar_cls is SpecificPitch:
            return self
        if target_scalar_cls is MidiPitch:
            midi_arr = self.midi_number.cast(pa.int64())
            new_struct = _struct_like(
                self,
                [midi_arr],
                [pa.field("midi_number", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("midi_number", new_struct.type)
            return MidiPitchField.from_field((new_struct, pa_field))
        if target_scalar_cls is EnharmonicPitchClass:
            pc_arr = self.pitch_class.cast(pa.int64())
            new_struct = _struct_like(
                self,
                [pc_arr],
                [pa.field("pitch_class", pa.int64(), nullable=True)],
            )
            pa_field = pa.field("pitch_class", new_struct.type)
            return EnharmonicPitchClassField.from_field((new_struct, pa_field))
        if target_scalar_cls is SpecificPitchClass:
            step = self._step_array()
            alter = self._alter_array().cast(pa.int64())
            new_struct = _struct_like(
                self,
                [step, alter],
                [
                    pa.field("step", pa.string(), nullable=True),
                    pa.field("alter", pa.int64(), nullable=True),
                ],
            )
            pa_field = pa.field("spc", new_struct.type)
            return SpecificPitchClassField.from_field((new_struct, pa_field))
        raise TypeError(
            f"Cannot convert SpecificPitch field to {target_scalar_cls.__name__}"
        )

    def to(self, target_scalar_cls: type) -> SemanticField[Any]:
        """Parity alias for :meth:`convert_to`."""
        return self.convert_to(target_scalar_cls)

    def get(self, *, format: str | None = None) -> pa.Array[Any]:
        """Vectorized ``step + accidental + octave`` string concatenation."""
        if format == "midi":
            return self.midi_number.cast(pa.string())
        step = self._step_array()
        alter = self._alter_array().cast(pa.int64())
        octave_str = self._octave_array().cast(pa.string())
        out = pa.array([""] * len(step), type=pa.string())
        for n in (-2, -1, 1, 2):
            if n > 0:
                marker = "♯" * n
            else:
                marker = "♭" * abs(n)
            mask = pc.equal(alter, pa.scalar(n))
            out = pc.if_else(mask, pa.scalar(marker, type=pa.string()), out)
        return pc.binary_join_element_wise(
            pc.binary_join_element_wise(step, out, ""), octave_str, ""
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. HARMONY — scalars, paired Fields, plus DCML import schemas
# ═══════════════════════════════════════════════════════════════════════════


class Inversion(IntEnum):
    """Chord inversion as an enum.

    The DCML ``figbass`` string maps to these values.
    """

    ROOT = 0
    FIRST = 1
    SECOND = 2
    THIRD = 3


# DCML figbass -> Inversion enum mapping
FIGBASS_TO_INVERSION: dict[str, Inversion] = {
    "": Inversion.ROOT,
    "7": Inversion.ROOT,
    "6": Inversion.FIRST,
    "64": Inversion.SECOND,
    "2": Inversion.THIRD,
    "65": Inversion.FIRST,
    "43": Inversion.SECOND,
    "42": Inversion.THIRD,
}


def figbass_to_inversion(figbass: str) -> Inversion | None:
    """Convert a DCML ``figbass`` string to an ``Inversion`` enum member.

    Args:
        figbass: The DCML figured bass string.

    Returns:
        The corresponding ``Inversion`` enum member, or ``None`` if
        the string is not a recognised figured bass.
    """
    if not figbass:
        return Inversion.ROOT
    return FIGBASS_TO_INVERSION.get(figbass)


# ---------------------------------------------------------------------------
# DCML import schemas — distinct from the canonical paired XField.pa_schema.
# These are the *external* DCML TSV column layouts used at the import edge.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonyBaseSchema:
    """Minimal harmony schema: label and codec standard.

    Storage struct::

        {label: string, standard: string}
    """

    label: str = "label"
    standard: str = "standard"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class WesternTertianSchema:
    """Schema for Western tertian harmony import: root, bass, chord type, inversion."""

    label: str = "label"
    standard: str = "standard"
    root: str = "root"
    bass: str = "bass"
    chord_type: str = "chord_type"
    inversion: str = "inversion"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass", pa.int64(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("inversion", pa.int64(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class RomanNumeralSchema:
    """Schema for Roman numeral harmony import: adds numeral, localkey, globalkey."""

    label: str = "label"
    standard: str = "standard"
    root: str = "root"
    bass: str = "bass"
    chord_type: str = "chord_type"
    inversion: str = "inversion"
    numeral: str = "numeral"
    localkey: str = "localkey"
    globalkey: str = "globalkey"
    key_context: str = "key_context"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("standard", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass", pa.int64(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("inversion", pa.int64(), nullable=True),
            pa.field("numeral", pa.string(), nullable=True),
            pa.field("localkey", pa.string(), nullable=True),
            pa.field("globalkey", pa.string(), nullable=True),
            pa.field("key_context", pa.string(), nullable=True),
        ]
    )


@dataclass(frozen=True)
class DcmlStorageSchema:
    """Storage schema for DCML harmony labels (the raw DCML TSV columns).

    Import-edge schema, NOT the canonical ``DcmlHarmonyField.pa_schema``.
    Used by external DCML reads which then map to internal model via
    ``DcmlHarmony.from_row``.
    """

    label: str = "label"
    globalkey: str = "globalkey"
    localkey: str = "localkey"
    numeral: str = "numeral"
    form: str = "form"
    figbass: str = "figbass"
    chord_type: str = "chord_type"
    root: str = "root"
    bass_note: str = "bass_note"

    schema: ClassVar[pa.StructType] = pa.struct(
        [
            pa.field("label", pa.string(), nullable=True),
            pa.field("globalkey", pa.string(), nullable=True),
            pa.field("localkey", pa.string(), nullable=True),
            pa.field("numeral", pa.string(), nullable=True),
            pa.field("form", pa.string(), nullable=True),
            pa.field("figbass", pa.string(), nullable=True),
            pa.field("chord_type", pa.string(), nullable=True),
            pa.field("root", pa.int64(), nullable=True),
            pa.field("bass_note", pa.int64(), nullable=True),
        ]
    )


# ---------------------------------------------------------------------------
# Harmony scalars + paired Fields
# ---------------------------------------------------------------------------


class HarmonyLabel(ScalarVocabulary, BaseModel):
    """Root harmony scalar.

    Pydantic v2 ``BaseModel``, frozen.  Minimal: label + standard.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str

    @property
    def semantic_type(self) -> str:
        return "HarmonyLabel"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> HarmonyLabel | None:
        label = row.get("label")
        if label is None:
            return None
        return cls(label=str(label), standard=str(row.get("standard") or ""))

    def __repr__(self) -> str:
        return f"HarmonyLabel(label={self.label!r}, standard={self.standard!r})"

    def __str__(self) -> str:
        return self.label


class HarmonyLabelField(SemanticField[HarmonyLabel]):
    """Columnar wrapper for ``HarmonyLabel`` (paired Field)."""


class PitchBasedHarmony(ScalarVocabulary, BaseModel):
    """Harmony with root and bass (OHR model)."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None

    @property
    def semantic_type(self) -> str:
        return "PitchBasedHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PitchBasedHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
        )

    def __repr__(self) -> str:
        return f"PitchBasedHarmony(label={self.label!r}, root={self.root})"

    def __str__(self) -> str:
        return self.label


class PitchBasedHarmonyField(SemanticField[PitchBasedHarmony]):
    """Columnar wrapper for ``PitchBasedHarmony`` (paired Field)."""


class WesternTertianHarmony(ScalarVocabulary, BaseModel):
    """Western tertian chord."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None

    @property
    def semantic_type(self) -> str:
        return "WesternTertianHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> WesternTertianHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        inversion_raw = row.get("inversion")
        if inversion_raw is None and "figbass" in row:
            inv = figbass_to_inversion(str(row.get("figbass") or ""))
            inversion_raw = int(inv) if inv is not None else None
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
            chord_type=str(row.get("chord_type") or ""),
            inversion=int(inversion_raw) if inversion_raw is not None else None,
        )

    def __repr__(self) -> str:
        return (
            f"WesternTertianHarmony(label={self.label!r}, "
            f"chord_type={self.chord_type!r})"
        )

    def __str__(self) -> str:
        return self.label


class WesternTertianHarmonyField(SemanticField[WesternTertianHarmony]):
    """Columnar wrapper for ``WesternTertianHarmony`` (paired Field)."""


class RomanNumeralHarmony(ScalarVocabulary, BaseModel):
    """Roman-numeral analysis."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: str
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None
    numeral: str = ""
    localkey: str = ""
    globalkey: str = ""

    @property
    def semantic_type(self) -> str:
        return "RomanNumeralHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": self.standard,
        }

    def to_dict(self) -> dict[str, object]:
        return self.model_dump()

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RomanNumeralHarmony | None:
        label = row.get("label")
        if label is None:
            return None
        root_raw = row.get("root")
        bass_raw = row.get("bass", row.get("bass_note"))
        inversion_raw = row.get("inversion")
        if inversion_raw is None and "figbass" in row:
            inv = figbass_to_inversion(str(row.get("figbass") or ""))
            inversion_raw = int(inv) if inv is not None else None
        return cls(
            label=str(label),
            standard=str(row.get("standard") or ""),
            root=int(root_raw) if root_raw is not None else None,
            bass=int(bass_raw) if bass_raw is not None else None,
            chord_type=str(row.get("chord_type") or ""),
            inversion=int(inversion_raw) if inversion_raw is not None else None,
            numeral=str(row.get("numeral") or ""),
            localkey=str(row.get("localkey") or ""),
            globalkey=str(row.get("globalkey") or ""),
        )

    def __repr__(self) -> str:
        return (
            f"RomanNumeralHarmony(label={self.label!r}, "
            f"numeral={self.numeral!r}, key={self.globalkey}:{self.localkey})"
        )

    def __str__(self) -> str:
        return self.label


class RomanNumeralHarmonyField(SemanticField[RomanNumeralHarmony]):
    """Columnar wrapper for ``RomanNumeralHarmony`` (paired Field)."""


class DcmlHarmony(ScalarVocabulary, BaseModel):
    """DCML harmony annotation.  ``standard`` pinned to ``Literal["dcml"]``."""

    model_config = ConfigDict(frozen=True)

    label: str
    standard: Literal["dcml"] = "dcml"
    root: int | None = None
    bass: int | None = None
    chord_type: str = ""
    inversion: int | None = None
    numeral: str = ""
    localkey: str = ""
    globalkey: str = ""
    tonicized_key: str | None = None
    pedal: str | None = None

    @property
    def semantic_type(self) -> str:
        return "DcmlHarmony"

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "standard": "dcml",
        }

    def to_dict(self) -> dict[str, object]:
        """Return a summary dict of all harmony properties.

        Root and bass are shown both as pitch class integers and as
        ``EnharmonicPitchClass`` objects for readability.
        """
        root_gpc = (
            EnharmonicPitchClass(pitch_class=self.root)
            if self.root is not None
            else None
        )
        bass_gpc = (
            EnharmonicPitchClass(pitch_class=self.bass)
            if self.bass is not None
            else None
        )
        return {
            "label": self.label,
            "numeral": self.numeral,
            "chord_type": self.chord_type,
            "inversion": self.inversion,
            "root": root_gpc,
            "bass": bass_gpc,
            "globalkey": self.globalkey,
            "localkey": self.localkey,
            "tonicized_key": self.tonicized_key,
        }

    @classmethod
    def from_label(
        cls,
        label: str,
        *,
        globalkey: str = "C",
        localkey: str = "I",
    ) -> DcmlHarmony:
        """Construct a fully populated ``DcmlHarmony`` from a DCML label string.

        Raises:
            ValueError: If the label does not match the DCML syntax, or names
                no chord type (a seventh-chord form with a triad's figured
                bass, such as ``"VM6"``).
        """
        from ms3.expand_dcml import features2type
        from ms3.utils import fifths2pc, roman_numeral2fifths
        from ms3.utils.constants import DCML_REGEX

        m = DCML_REGEX.match(label)
        if m is None:
            raise ValueError(f"Cannot parse DCML label: {label!r}")

        parts = {k: v for k, v in m.groupdict().items() if v is not None}
        numeral = parts.get("numeral", "")
        form = parts.get("form")
        figbass = parts.get("figbass")
        relativeroot = parts.get("relativeroot")
        pedal = parts.get("pedal")

        chord_type = features2type(numeral, form, figbass) if numeral else ""
        if chord_type is None:
            raise ValueError(
                f"Cannot parse DCML label: {label!r} (form {form!r} names a seventh "
                f"chord, which cannot have figured bass {figbass!r})"
            )
        inv = figbass_to_inversion(figbass or "")
        inversion = int(inv) if inv is not None else None

        root: int | None = None
        bass: int | None = None
        if numeral:
            root_tpc = roman_numeral2fifths(numeral)
            root = fifths2pc(root_tpc)
            try:
                from ms3 import chord2tpcs

                chord_str = parts.get("chord", label)
                tpcs = chord2tpcs(chord_str)
                if tpcs:
                    bass = fifths2pc(tpcs[0])
            except Exception:
                bass = root

        return cls(
            label=label,
            globalkey=globalkey,
            localkey=localkey,
            numeral=numeral,
            chord_type=chord_type,
            inversion=inversion,
            root=root,
            bass=bass,
            tonicized_key=relativeroot,
            pedal=pedal,
        )

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> DcmlHarmony | None:
        """Construct from a DCML storage row dict.

        Maps DCML storage field names to internal model names:
        - ``bass_note`` -> ``bass``
        - ``figbass`` -> ``inversion``
        - ``relativeroot`` -> ``tonicized_key``
        """
        label = row.get("label")
        if label is None:
            return None

        root_raw = row.get("root")
        root = int(root_raw) if root_raw is not None else None
        bass_raw = row.get("bass_note", row.get("bass"))
        bass = int(bass_raw) if bass_raw is not None else None
        figbass_raw = row.get("figbass", "")
        inversion_raw = row.get("inversion")
        if inversion_raw is not None:
            inversion = int(inversion_raw)
        else:
            inv = figbass_to_inversion(str(figbass_raw or ""))
            inversion = int(inv) if inv is not None else None

        globalkey = str(row.get("globalkey") or "")
        localkey = str(row.get("localkey") or "")

        return cls(
            label=str(label),
            globalkey=globalkey,
            localkey=localkey,
            numeral=str(row.get("numeral") or ""),
            chord_type=str(row.get("chord_type") or ""),
            inversion=inversion,
            root=root,
            bass=bass,
            tonicized_key=row.get("relativeroot") or row.get("tonicized_key"),
            pedal=row.get("pedal"),
        )

    def __repr__(self) -> str:
        return (
            f"DcmlHarmony(label={self.label!r}, key={self.globalkey}:{self.localkey})"
        )

    def __str__(self) -> str:
        return self.label


class DcmlHarmonyField(SemanticField[DcmlHarmony]):
    """Columnar wrapper for ``DcmlHarmony`` (paired Field)."""


# ═══════════════════════════════════════════════════════════════════════════
# 3. EVENT SCALARS — Note, Measure, MeasureNumber, Id, Projection
# ═══════════════════════════════════════════════════════════════════════════


class Note(ScalarVocabulary, BaseModel):
    """A single note or rest event.

    A note holds what every note has, whether a score writes it or a
    performance plays it.  The staff and the voice a score writes a note
    on and in belong to :class:`ScoreNote`; a keyword this model does not
    declare is refused rather than dropped.

    ``Note.pitch`` is annotated ``EnharmonicPitch | SpecificPitch | None``
    for materialised scalars — both are legitimate pitch representations
    on a note.  This union MUST NOT translate to Arrow ``dense_union``;
    instead, the field is dropped from the pa.Schema and pitch is
    represented exactly once on the EventData.  ``NoteEventData`` (a
    spelled score source) stores ``specific_pitch`` as the sole default
    semantic pitch field and keeps the source MIDI pitch as a
    non-default raw ``midi`` int that affords an ``EnharmonicPitch`` view
    on request.  Number-only sources store the bare number instead (as a
    raw ``pitch`` int or an ``EnharmonicPitch`` ``{midi_number}`` struct).
    See Contributing → §2.4 Architectural decision log
    (union-of-BaseModels rejected; columnar separation).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    start: Coordinate
    end: Coordinate | None = None
    duration: Duration | None = None
    pitch: EnharmonicPitch | SpecificPitch | None = None
    velocity: int | None = None
    instrument: str | None = None

    @field_validator("duration", mode="before")
    @classmethod
    def _coerce_duration_from_coordinate(cls, v: object) -> object:
        if v is None or isinstance(v, Duration):
            return v
        if isinstance(v, Coordinate):
            return Duration(v.value, v.unit)
        return v

    @property
    @data_shaped
    def is_rest(self) -> bool:
        return self.pitch is None

    @property
    def semantic_type(self) -> str:
        return type(self).__name__

    def metadata_dict(self) -> dict[str, str]:
        return {
            **super().metadata_dict(),
            "has_pitch": str(self.pitch is not None).lower(),
        }

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dictionary.

        Pitch is intentionally absent — it is stored in a separate field
        on ``NoteEventData`` (columnar separation).
        """
        from .retrieval import coordinate_to_dict

        d: dict[str, object] = {
            "start": coordinate_to_dict(self.start),
            "end": coordinate_to_dict(self.end) if self.end is not None else None,
            "velocity": self.velocity,
            "instrument": self.instrument,
        }
        d["duration"] = (
            coordinate_to_dict(self.duration) if self.duration is not None else None
        )
        return d

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any] | None,
        *,
        unit: TimeUnit | str,
        number_type: NumberType | str | None,
    ) -> Self | None:
        """Construct from a row of an event table whose numbers lie on one axis.

        Each position (``start``, ``end``, ``duration``) is read from its
        number struct on the axis given, in *number_type*, exactly as a
        :class:`Box` reads its numbers; a position given as a coordinate
        dict (``to_dict``) or a scalar keeps the unit it states.  The pitch
        is read from the column representing it (``specific_pitch``, else
        ``midi``, else ``pitch``).  Every column this type does not declare
        is ignored, so a row of a score is read as a plain ``Note`` as well
        as a :class:`ScoreNote`.

        Args:
            row: The row, or ``None`` for a null cell.
            unit: The unit of every position.
            number_type: The number type every position is written in.

        Returns:
            The note, or ``None`` for a null cell or a row without a start.
        """
        if row is None:
            return None
        axis = TimeUnit(unit)
        values = cls._values_from_row(row, axis, axis.resolve_number_type(number_type))
        return None if values is None else cls(**values)

    @classmethod
    def _values_from_row(
        cls, row: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> dict[str, Any] | None:
        """Return the constructor keywords read from *row*, ``None`` without a start.

        Subtypes extend the mapping with the columns they add, so that every
        type of the hierarchy reads a row the same way.
        """
        from .retrieval import coordinate_from_dict

        def _position(
            raw: Any, kind: type[_PositionOrExtentT]
        ) -> _PositionOrExtentT | None:
            if raw is None or isinstance(raw, kind):
                return raw
            if isinstance(raw, TimeScalar):
                return kind(raw.value, raw.unit, number_type=raw.number_type)
            if not _is_dict(raw) or raw.get("value") is None:
                return None
            if "number_type" in raw:
                return _position(coordinate_from_dict(raw), kind)
            return _axis_value(kind, raw, unit, number_type)

        start = _position(row.get("start"), Coordinate)
        if start is None:
            return None
        end = _position(row.get("end"), Coordinate)
        duration = _position(row.get("duration"), Duration)

        # Represent-once resolution.  A spelled source stores the full
        # ``specific_pitch`` (the most-expressive faithful type); prefer
        # it.  Otherwise fall back to the bare MIDI pitch a number-only
        # source carries — either the non-default raw ``midi`` int on a
        # spelled NoteEventData, or a ``pitch`` value that arrives as an
        # ``EnharmonicPitch`` ``{midi_number}`` struct dict (or a plain
        # int) from a number-only timeline.  Pitch is never stored twice,
        # so at most one of these is populated.
        pitch: EnharmonicPitch | SpecificPitch | None = None
        sp_raw = row.get("specific_pitch")
        if _is_dict(sp_raw):
            pitch = SpecificPitch.from_row(sp_raw)
        if pitch is None:
            midi_raw = row.get("midi")
            pitch_raw = row.get("pitch")
            if isinstance(midi_raw, (int, float)):
                pitch = EnharmonicPitch(midi_number=int(midi_raw))
            elif _is_dict(pitch_raw):
                pitch = EnharmonicPitch.from_row(pitch_raw)
            elif isinstance(pitch_raw, (int, float)):
                pitch = EnharmonicPitch(midi_number=int(pitch_raw))

        return {
            "start": start,
            "end": end,
            "duration": duration,
            "pitch": pitch,
            "velocity": row.get("velocity"),
            "instrument": row.get("instrument"),
        }

    def __repr__(self) -> str:
        pitch_str = repr(self.pitch) if self.pitch is not None else "rest"
        return (
            f"{type(self).__name__}(start={self.start}, duration={self.duration}, "
            f"pitch={pitch_str})"
        )

    def __str__(self) -> str:
        pitch_str = str(self.pitch) if self.pitch is not None else "rest"
        return f"{pitch_str} @{self.start}+{self.duration}"


# Columnar-separation rule: drop ``Note.pitch`` from pa.Schema so that
# the represent-once pitch field (``specific_pitch`` for a spelled
# source, plus the non-default raw ``midi`` int) lives in its own
# column (see Contributing → §2.4 Architectural decision log).
register_value_projector(Note, "pitch", drop_from_schema)


class NoteField(SharedAxisField[Note]):
    """Paired Field for :class:`Note`: every position on one axis, declared once.

    ``EventData.get_field(Note)`` composes it from an event table: the
    Note's members from the columns of their names, the axis from the
    table's, and the pitch from the column representing it, carried under
    that column's own name (``specific_pitch``, ``midi`` or ``pitch``), so
    that each item is the note the table holds.
    """

    @property
    def is_rest(self) -> pa.Array[Any]:
        """Vectorized rest predicate.

        A row is a rest when the EventData carries no pitch for it — i.e.
        the default ``specific_pitch`` semantic field and the raw ``midi``
        number it affords an ``EnharmonicPitch`` view from are both null.
        At Field level, only the ``Note`` struct columns are available;
        the columnar-separation rule means ``Note.pitch``
        does NOT exist on the Field's struct.

        ``Note.is_rest`` is data-shaped, but the pitch data lives outside the
        ``Note`` struct in the represent-once TTA layout — so this method must
        be invoked on a higher-level container that knows about the separated
        pitch columns. Until that container exists, the field has no such
        attribute: the mirror raises ``AttributeError``, so ``hasattr`` and
        runtime protocol checks answer ``False``. The field's attribute
        delegation then reports the name missing on the raw struct field.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(
            "NoteField.is_rest requires access to the represent-once pitch "
            "columns on NoteEventData (the default specific_pitch field and "
            "the raw midi number), not available on the bare Note struct "
            "field."
        )


class ScoreNote(Note):
    """A note or rest of a score: a :class:`Note` written on a staff, in a voice.

    ``voice`` and ``staff`` are the source's numbers (an MEI ``<layer>`` and
    ``<staff>``, a MuseScore voice and staff, partitura's ``voice`` and
    ``staff``); a performance states neither, so they are not members of
    :class:`Note`.  Every other member, and the represent-once storage of
    the pitch, is the Note's.
    """

    voice: int | None = None
    staff: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the Note's JSON-safe dictionary with ``voice`` and ``staff``."""
        return {**super().to_dict(), "voice": self.voice, "staff": self.staff}

    @classmethod
    def _values_from_row(
        cls, row: dict[str, Any], unit: TimeUnit, number_type: NumberType
    ) -> dict[str, Any] | None:
        values = super()._values_from_row(row, unit, number_type)
        if values is None:
            return None
        return {**values, "voice": row.get("voice"), "staff": row.get("staff")}


class ScoreNoteField(NoteField):
    """Paired Field for :class:`ScoreNote`.

    A ``NoteField`` whose struct adds ``voice`` and ``staff``; the vectorized
    members are NoteField's own.
    """

    scalar_cls = ScoreNote


# ---------------------------------------------------------------------------
# Measure + MeasureField
# ---------------------------------------------------------------------------


class Measure(ScalarVocabulary, BaseModel):
    """One measure-like unit in a work's immutable measure map.

    ``count`` and ``qstamp`` may be supplied by a source, but a containing
    measure map derives both from printed order and exact actual lengths.
    The map warns when a supplied value disagrees with that derivation.

    ``volta`` is the volta bracket the measure lies under, as its source
    states it: the number of the pass it is played in, or a tuple of
    distinct numbers where the bracket stands for several passes (printed
    "1, 3"). Neither form is turned into the other, so ``1`` and ``(1,)``
    stay apart; :attr:`volta_numbers` gives the numbers of either form.
    Storage keeps the forms apart in two members, the integer under
    ``volta`` and the tuple under ``volta_list``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    id: str | None = Field(  # noqa: A003 - Measure Map vocabulary
        default=None,
        validation_alias=AliasChoices("id", "ID"),
        serialization_alias="ID",
    )
    count: int | None = None
    qstamp: Fraction | None = None
    number: int | None = None
    name: str | None = None
    time_signature: str | None = None
    nominal_length: Fraction | None = None
    actual_length: Fraction | None = None
    start_repeat: bool = False
    end_repeat: bool = False
    next: tuple[str, ...] | None = None
    volta: int | tuple[int, ...] | None = None

    @classmethod
    def from_values(cls, **values: Any) -> Measure:
        """Build the measure of the concrete class its values describe.

        This is the one classification of a measure, shared by every lane
        that builds measures from source facts or document rows:

        - no ``nominal_length``: a plain :class:`Measure`;
        - a ``cadenza`` time signature: :class:`CadenzaMeasure`;
        - a non-zero ``offset_within_measure``: :class:`MeasureConstituent`;
        - ``actual_length`` equal to ``nominal_length``: :class:`RegularMeasure`;
        - otherwise: :class:`IrregularMeasure`.

        Neither ``number`` nor a neighbouring measure takes part: a shared
        number also marks alternative endings, so it never identifies a
        split measure.

        Args:
            **values: Measure fields, optionally with ``offset_within_measure``.

        Returns:
            The measure, of the class its values describe.

        Raises:
            ValueError: If a non-zero offset is stated for a measure without
                nominal length or for a cadenza, which cannot lie inside a
                nominal measure.
        """
        offset = values.pop("offset_within_measure", None) or Fraction(0)
        signature = values.get("time_signature")
        target: type[Measure]
        if values.get("nominal_length") is None:
            target = Measure
        elif signature is not None and str(signature).lower() == "cadenza":
            target = CadenzaMeasure
        elif offset:
            return MeasureConstituent(**values, offset_within_measure=offset)
        elif values.get("actual_length") == values["nominal_length"]:
            target = RegularMeasure
        else:
            target = IrregularMeasure
        if offset:
            raise ValueError(
                f"A {target.__name__} cannot lie at offset {offset} inside a "
                "nominal measure"
            )
        return target(**values)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Measure:
        """Restore the concrete measure class named by a dictionary payload.

        Args:
            data: Measure dictionary carrying a concrete ``class`` tag.

        Returns:
            A measure of the tagged concrete class.

        Raises:
            ValueError: If the tag is missing, unknown, or incompatible with
                the receiving subclass.
        """

        def descendants(parent: type[Measure]) -> list[type[Measure]]:
            children: list[type[Measure]] = []
            for child in parent.__subclasses__():
                children.append(child)
                children.extend(descendants(child))
            return children

        known_classes = [Measure, *descendants(Measure)]
        known = {
            measure_class.__name__: measure_class for measure_class in known_classes
        }
        class_tag = data.get("class")
        target = known.get(class_tag) if isinstance(class_tag, str) else None
        if target is None:
            raise ValueError(
                f"Unknown or missing Measure class {class_tag!r}; "
                f"known classes: {', '.join(sorted(known))}"
            )
        if cls is not Measure and target is not cls:
            raise ValueError(
                f"Measure class {class_tag!r} does not match receiving "
                f"subclass {cls.__name__!r}"
            )

        values = {key: value for key, value in data.items() if key != "class"}
        unknown_keys = sorted(set(values) - set(target.model_fields))
        if unknown_keys:
            raise ValueError(
                f"Measure class {target.__name__!r} does not declare the keys: "
                f"{', '.join(unknown_keys)}"
            )
        for key in ("qstamp", "nominal_length", "actual_length"):
            if values.get(key) is not None:
                values[key] = number_from_json(values[key])
        if values.get("next") is not None:
            values["next"] = tuple(values["next"])
        if (
            issubclass(target, MeasureConstituent)
            and values.get("offset_within_measure") is not None
        ):
            values["offset_within_measure"] = number_from_json(
                values["offset_within_measure"]
            )
        return target(**values)

    @field_validator("qstamp", "nominal_length", "actual_length", mode="before")
    @classmethod
    def _as_exact_quarters(cls, value: Any) -> Any:
        if value is None or isinstance(value, Fraction):
            return value
        return Fraction(value)

    @field_validator("next", mode="before")
    @classmethod
    def _as_measure_ids(cls, value: Any) -> tuple[str, ...] | None:
        if value is None or value == "":
            return None
        if isinstance(value, str):
            text = value.strip().strip("()[]")
            return (
                tuple(part.strip() for part in text.split(",") if part.strip()) or None
            )
        return tuple(str(item) for item in value)

    @field_validator("volta", mode="after")
    @classmethod
    def _as_distinct_volta_numbers(
        cls, value: int | tuple[int, ...] | None
    ) -> int | tuple[int, ...] | None:
        if isinstance(value, tuple) and (not value or len(set(value)) != len(value)):
            raise ValueError(
                f"a volta naming several passes holds at least one number, each "
                f"once, not {list(value)!r}"
            )
        return value

    @model_validator(mode="after")
    def _default_name(self) -> Measure:
        if self.name is None and self.number is not None:
            object.__setattr__(self, "name", str(self.number))
        return self

    @property
    def semantic_type(self) -> str:
        """Return the scalar's semantic name."""
        return "Measure"

    @property
    def volta_numbers(self) -> tuple[int, ...]:
        """The numbers of the passes :attr:`volta` names, whichever form it has.

        Examples:
            >>> Measure(volta=2).volta_numbers, Measure(volta=(1, 3)).volta_numbers
            ((2,), (1, 3))
            >>> Measure().volta_numbers
            ()
        """
        if self.volta is None:
            return ()
        return self.volta if isinstance(self.volta, tuple) else (self.volta,)

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"count={self.count!r}", f"name={self.name!r}"]
        for member in ("number", "time_signature", "qstamp", "actual_length"):
            value = getattr(self, member)
            if value is not None:
                parts.append(f"{member}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.name or (self.id or "Measure")

    def to_dict(self) -> dict[str, Any]:
        """Return the concrete measure as a JSON-safe dictionary."""
        data: dict[str, Any] = {
            "class": type(self).__name__,
            "id": self.id,
            "count": self.count,
            "qstamp": (
                number_to_json(self.qstamp) if self.qstamp is not None else None
            ),
            "number": self.number,
            "name": self.name,
            "time_signature": self.time_signature,
            "nominal_length": (
                number_to_json(self.nominal_length)
                if self.nominal_length is not None
                else None
            ),
            "actual_length": (
                number_to_json(self.actual_length)
                if self.actual_length is not None
                else None
            ),
            "start_repeat": self.start_repeat,
            "end_repeat": self.end_repeat,
            "next": list(self.next) if self.next is not None else None,
            "volta": (
                list(self.volta) if isinstance(self.volta, tuple) else self.volta
            ),
        }
        if isinstance(self, MeasureConstituent):
            data["offset_within_measure"] = number_to_json(self.offset_within_measure)
        return data


class MeasureField(SemanticField[Measure]):
    """Columnar wrapper for ``Measure`` (paired Field)."""


class RegularMeasure(Measure):
    """A measure whose actual and nominal lengths agree."""


class RegularMeasureField(SemanticField[RegularMeasure]):
    """Columnar wrapper for :class:`RegularMeasure`."""


class IrregularMeasure(Measure):
    """A measure whose sounding length differs from its nominal length."""


class IrregularMeasureField(SemanticField[IrregularMeasure]):
    """Columnar wrapper for :class:`IrregularMeasure`."""


class SplitRegularMeasure(RegularMeasure):
    """A regular measure represented by more than one source constituent."""


class SplitRegularMeasureField(SemanticField[SplitRegularMeasure]):
    """Columnar wrapper for :class:`SplitRegularMeasure`."""


class SplitIrregularMeasure(IrregularMeasure):
    """An irregular split measure."""


class SplitIrregularMeasureField(SemanticField[SplitIrregularMeasure]):
    """Columnar wrapper for :class:`SplitIrregularMeasure`."""


class MeasureConstituent(IrregularMeasure):
    """A split-measure constituent with an offset in its nominal measure."""

    offset_within_measure: Fraction = Fraction(0)

    @field_validator("offset_within_measure", mode="before")
    @classmethod
    def _as_exact_offset(cls, value: Any) -> Fraction:
        return value if isinstance(value, Fraction) else Fraction(value)


class MeasureConstituentField(SemanticField[MeasureConstituent]):
    """Columnar wrapper for :class:`MeasureConstituent`."""


class CadenzaMeasure(IrregularMeasure):
    """An unmetered or freely measured span."""


class CadenzaMeasureField(SemanticField[CadenzaMeasure]):
    """Columnar wrapper for :class:`CadenzaMeasure`."""


@dataclass(frozen=True)
class Gap(_OpenObject):
    """A stretch of a played order that nothing of the timeline fills.

    A gap is an entry among the chunks of a FlowMap: the virtual axis goes
    on for *length* while no position of the timeline is played. The length
    is a bare number on the timeline's axis and strictly positive; it is
    kept as given and read in the timeline's number type when the FlowMap
    joins its timeline. In a manifest it is the object ``{gap: length}``,
    whose unknown keys are kept and written back.

    Attributes:
        length: The length of the gap.

    Raises:
        ValueError: If the length is no finite number, or is not positive.

    Examples:
        >>> Gap(4)
        Gap(4)
        >>> Gap(0)
        Traceback (most recent call last):
        ...
        ValueError: the length of a gap is positive, got 0
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("gap",)

    length: int | float | Fraction
    # A field, so that a copy made with ``dataclasses.replace`` keeps it.
    _unknown_keys: Mapping[str, Any] = field(
        default_factory=dict[str, Any], kw_only=True, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        length = self.length
        if isinstance(length, bool) or not isinstance(length, int | float | Fraction):
            raise ValueError(f"the length of a gap is a number, got {length!r}")
        if isinstance(length, float) and not math.isfinite(length):
            raise ValueError(f"the length of a gap is finite, got {length!r}")
        if length <= 0:
            raise ValueError(f"the length of a gap is positive, got {length}")

    def __repr__(self) -> str:
        return f"Gap({self.length})"


class BeatPolicy(BaseModel):
    """How a bar is counted in beats: a grouping cycle over one division.

    A time signature says two things at once — which note value is
    counted, and how those values group into beats.  ``division`` is the
    counted value in quarters (``Fraction(1, 2)`` for an eighth), and
    ``grouping`` says how many of them each successive beat spans.  A
    compound ``6/8`` bar is ``grouping=(3, 3)`` over eighths; a simple
    ``4/4`` bar is ``grouping=(1, 1, 1, 1)`` over quarters; an uneven
    ``(3+2+3)/8`` bar is ``grouping=(3, 2, 3)`` over eighths.

    The beat lengths (:attr:`rods`) are derived from that pair, never
    stored: the source encoding is what a policy holds, and every rod,
    offset and index question is answered from it.

    Attributes:
        grouping: How many divisions each beat spans, in order.
        division: The counted note value, in quarters. Given either
            directly or through *beat_size*, which completes it.
        beat_size: The counted value as a typed duration in whole notes
            or quarters. Completed from *division* when omitted.
        name: Optional human-readable label, typically the signature the
            source spells.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    grouping: tuple[int, ...] = (1,)
    division: Fraction | None = None
    beat_size: Duration | None = None
    name: str | None = None

    @classmethod
    def from_time_signature(cls, signature: str) -> BeatPolicy:
        """Derive the default counting of a bar from its time signature.

        A signature ``"n/d"`` is read as compound when ``n`` is a
        multiple of three greater than three and the denominator is an
        eighth or shorter — ``6/8``, ``9/8``, ``12/16`` count in dotted
        beats of three divisions.  Everything else counts one division
        per beat.  A composite signature ``"a/d+b/d+c/d"`` spells its own
        grouping.  ``"C"`` is ``4/4`` and ``"C|"`` (also spelled
        ``"cut"``) is ``2/2``.

        Args:
            signature: The time signature as the source spells it.

        Returns:
            The policy that counts a bar of that signature.

        Raises:
            ValueError: If *signature* cannot be read.  There is no
                silent fallback to ``4/4``: choosing a default for an
                unreadable source is a loader's decision, not a
                policy's.
        """
        text = signature.strip()
        if not text:
            raise ValueError("Time signature must not be empty")
        if text in ("C", "c"):
            return cls(grouping=(1,) * 4, division=Fraction(1), name=text)
        if text in ("C|", "c|", "cut", "CUT"):
            return cls(grouping=(1,) * 2, division=Fraction(2), name=text)

        numerators: list[int] = []
        denominator: int | None = None
        shorthand = re.fullmatch(r"(\d+(?:\s*\+\s*\d+)*)\s*/\s*(\d+)", text)
        if shorthand is not None:
            numerators = [int(numerator) for numerator in shorthand.group(1).split("+")]
            denominator = int(shorthand.group(2))
            if any(numerator < 1 for numerator in numerators) or denominator < 1:
                raise ValueError(f"Cannot read time signature {signature!r}")
        else:
            terms = [term.strip() for term in text.split("+")]
            for term in terms:
                match = re.fullmatch(r"(\d+)\s*/\s*(\d+)", term)
                if match is None:
                    raise ValueError(f"Cannot read time signature {signature!r}")
                numerator = int(match.group(1))
                term_denominator = int(match.group(2))
                if numerator < 1 or term_denominator < 1:
                    raise ValueError(f"Cannot read time signature {signature!r}")
                if denominator is None:
                    denominator = term_denominator
                elif denominator != term_denominator:
                    raise ValueError(
                        f"Composite time signature {signature!r} mixes denominators; "
                        "every term must share one denominator"
                    )
                numerators.append(numerator)
        assert denominator is not None
        division = Fraction(4, denominator)

        if len(numerators) > 1:
            return cls(grouping=tuple(numerators), division=division, name=text)

        count = numerators[0]
        if count % 3 == 0 and count > 3 and denominator >= 8:
            return cls(grouping=(3,) * (count // 3), division=division, name=text)
        return cls(grouping=(1,) * count, division=division, name=text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BeatPolicy:
        """Restore a beat policy from its JSON dictionary representation."""
        return cls(
            grouping=tuple(data["grouping"]),
            beat_size=_duration_from_dict(data["beat_size"]),
            name=data.get("name"),
        )

    @classmethod
    def uniform(
        cls, division: Fraction, count: int, *, name: str | None = None
    ) -> BeatPolicy:
        """Count *count* beats of one *division* each.

        Args:
            division: The counted note value, in quarters.
            count: How many such beats fill the bar.
            name: Optional label, typically the signature the source
                spells.

        Returns:
            A policy with an all-ones grouping.
        """
        return cls(grouping=(1,) * count, division=Fraction(division), name=name)

    @field_validator("grouping")
    @classmethod
    def _validate_grouping(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("BeatPolicy grouping must not be empty")
        if any(entry < 1 for entry in value):
            raise ValueError("BeatPolicy grouping entries must be >= 1")
        return value

    @field_validator("division")
    @classmethod
    def _validate_division(cls, value: Fraction | None) -> Fraction | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("BeatPolicy division must be positive")
        return value

    @model_validator(mode="after")
    def _complete_beat_size(self) -> BeatPolicy:
        """Keep the typed beat size and quarter-note division in agreement."""
        from .enums import TimeUnit

        division = self.division
        beat_size = self.beat_size
        if division is None and beat_size is None:
            raise ValueError("BeatPolicy requires beat_size or division")
        if division is None:
            assert beat_size is not None
            if beat_size.unit is TimeUnit.whole_notes:
                division = Fraction(beat_size.value) * 4
            elif beat_size.unit is TimeUnit.quarters:
                division = Fraction(beat_size.value)
            else:
                raise ValueError(
                    "BeatPolicy beat_size must use whole_notes or quarters"
                )
            object.__setattr__(self, "division", division)
        elif beat_size is not None:
            if beat_size.unit is TimeUnit.whole_notes:
                stated_division = Fraction(beat_size.value) * 4
            elif beat_size.unit is TimeUnit.quarters:
                stated_division = Fraction(beat_size.value)
            else:
                raise ValueError(
                    "BeatPolicy beat_size must use whole_notes or quarters"
                )
            if stated_division != division:
                raise ValueError(
                    "BeatPolicy division and beat_size disagree: "
                    f"{division} quarters != {stated_division} quarters"
                )
        if beat_size is None:
            object.__setattr__(
                self,
                "beat_size",
                Duration(Fraction(division, 4), TimeUnit.whole_notes),
            )
        return self

    @property
    def rods(self) -> tuple[Fraction, ...]:
        """Length of every beat in this bar, in quarters."""
        assert self.division is not None
        return tuple(entry * self.division for entry in self.grouping)

    @property
    def n_beats(self) -> int:
        """How many beats the bar is counted in."""
        return len(self.grouping)

    @property
    def span(self) -> Fraction:
        """Total length of one counted bar, in quarters."""
        return sum(self.rods, Fraction(0))

    def as_divisions(self) -> BeatPolicy:
        """Return the same bar counted once per source division.

        Returns:
            A policy whose grouping contains one entry for every counted
            division, with the original division and name preserved.
        """
        assert self.division is not None
        return type(self)(
            grouping=(1,) * sum(self.grouping),
            division=self.division,
            name=self.name,
        )

    def rod_for(self, index: int) -> Fraction:
        """Return the length of beat *index* (1-based), in quarters.

        Args:
            index: 1-based beat index; ``1`` is the downbeat.

        Returns:
            The beat's length in quarters.

        Raises:
            ValueError: If *index* is outside ``1..n_beats``.
        """
        self._check_index(index)
        return self.rods[index - 1]

    def offset_for(self, index: int) -> Fraction:
        """Return the distance from the downbeat to beat *index*.

        Args:
            index: 1-based beat index; ``1`` is the downbeat.

        Returns:
            Quarters from the bar's downbeat.

        Raises:
            ValueError: If *index* is outside ``1..n_beats``.
        """
        self._check_index(index)
        return sum(self.rods[: index - 1], Fraction(0))

    def index_at(self, offset: Fraction) -> int:
        """Return the 1-based index of the beat containing *offset*.

        Args:
            offset: Quarters from the bar's downbeat.

        Returns:
            The 1-based beat index.

        Raises:
            ValueError: If *offset* is negative or beyond the bar's span.
        """
        position = Fraction(offset)
        if position < 0 or position >= self.span:
            raise ValueError(
                f"Offset {position} lies outside a bar of {self.span} quarters"
            )
        running = Fraction(0)
        for index, rod in enumerate(self.rods, start=1):
            running += rod
            if position < running:
                return index
        raise ValueError(  # pragma: no cover - guarded by the span check above
            f"Offset {position} lies outside a bar of {self.span} quarters"
        )

    def _check_index(self, index: int) -> None:
        if not 1 <= index <= self.n_beats:
            raise ValueError(
                f"Beat index {index} is outside 1..{self.n_beats} for this policy"
            )

    def __repr__(self) -> str:
        grouping = "+".join(str(entry) for entry in self.grouping)
        name = f", {self.name!r}" if self.name is not None else ""
        return f"BeatPolicy({grouping} x {self.division}{name})"

    def __str__(self) -> str:
        return self.name or repr(self)

    def to_dict(self) -> dict[str, Any]:
        """Return this policy as a JSON-safe dictionary."""
        if self.beat_size is None:
            raise ValueError("BeatPolicy states no beat size")
        return {
            "grouping": list(self.grouping),
            "beat_size": _duration_to_dict(self.beat_size),
            "name": self.name,
        }


class BeatPolicyField(SemanticField[BeatPolicy]):
    """Paired Field for :class:`BeatPolicy`.

    Empty body — :class:`BeatPolicy` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(BeatPolicy)`` dispatch.
    """


def _exact_optional(value: object, *, field_name: str) -> Fraction | None:
    """Read an optional exact scalar without guessing a nearby ratio."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Fraction)):
        raise TypeError(f"Tempo {field_name} must be a number or decimal string")
    return Fraction(value)


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _duration_to_dict(value: Duration | None) -> dict[str, Any] | None:
    """Encode an optional duration with its exact value and unit."""
    if value is None:
        return None
    return {
        "value": number_to_json(value.value),
        "unit": value.unit.value,
        "number_type": value.number_type.name,
    }


def _duration_from_dict(value: dict[str, Any] | None) -> Duration | None:
    """Restore an optional duration from its typed dictionary."""
    if value is None:
        return None
    from .enums import NumberType, TimeUnit

    return Duration(
        number_from_json(value["value"]),
        TimeUnit(value["unit"]),
        number_type=NumberType[value["number_type"]],
    )


class Tempo(BaseModel):
    """A stated tempo rate, independent of any metrical reading.

    Attributes:
        bpm: Beats per minute at the start of the statement.
        beat: The duration unit for which the rate is stated.
        to_bpm: Optional target rate of a ramp.
        mean_at: Optional share of the ramp span at which its mean occurs.
        text: Optional verbal tempo indication supplied by the source.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    bpm: Fraction | None
    beat: Duration | None = None
    to_bpm: Fraction | None = None
    mean_at: Fraction | None = None
    text: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Tempo:
        """Restore a tempo from its JSON dictionary representation.

        Args:
            data: JSON-safe tempo dictionary.

        Returns:
            The restored exact tempo statement.
        """
        return cls(
            bpm=(
                None
                if data.get("bpm") is None
                else Fraction(number_from_json(data["bpm"]))
            ),
            beat=_duration_from_dict(data.get("beat")),
            to_bpm=(
                None
                if data.get("to_bpm") is None
                else Fraction(number_from_json(data["to_bpm"]))
            ),
            mean_at=(
                None
                if data.get("mean_at") is None
                else Fraction(number_from_json(data["mean_at"]))
            ),
            text=data.get("text"),
        )

    @field_validator("bpm", "to_bpm", "mean_at", mode="before")
    @classmethod
    def _coerce_exact_rate(cls, value: object, info: Any) -> Fraction | None:
        return _exact_optional(value, field_name=info.field_name)

    @field_validator("bpm", "to_bpm")
    @classmethod
    def _validate_positive_rate(cls, value: Fraction | None) -> Fraction | None:
        if value is not None and value <= 0:
            raise ValueError("Tempo rates must be positive")
        return value

    @field_validator("bpm", "to_bpm", "mean_at")
    @classmethod
    def _validate_exact_dict_range(
        cls, value: Fraction | None, info: Any
    ) -> Fraction | None:
        if value is None:
            return None
        if not (
            _INT64_MIN <= value.numerator <= _INT64_MAX
            and value.denominator <= _INT64_MAX
        ):
            raise ValueError(
                f"Tempo {info.field_name} value {value} exceeds the signed int64 "
                f"dictionary limit {_INT64_MAX}"
            )
        return value

    @model_validator(mode="after")
    def _validate_ramp_mean(self) -> Tempo:
        if self.mean_at is None:
            return self
        if not 0 < self.mean_at < 1:
            raise ValueError("Tempo mean_at must satisfy 0 < mean_at < 1")
        if self.to_bpm is None:
            raise ValueError("Tempo mean_at requires to_bpm")
        return self

    def __repr__(self) -> str:
        parts = [f"bpm={self.bpm!r}"]
        if self.beat is not None:
            parts.append(f"beat={self.beat!r}")
        if self.to_bpm is not None:
            parts.append(f"to_bpm={self.to_bpm!r}")
        if self.mean_at is not None:
            parts.append(f"mean_at={self.mean_at!r}")
        if self.text is not None:
            parts.append(f"text={self.text!r}")
        return f"Tempo({', '.join(parts)})"

    def to_dict(self) -> dict[str, Any]:
        """Return this tempo as a JSON-safe dictionary.

        Returns:
            A dictionary retaining every exact rational value.
        """
        return {
            "bpm": None if self.bpm is None else number_to_json(self.bpm),
            "beat": _duration_to_dict(self.beat),
            "to_bpm": (None if self.to_bpm is None else number_to_json(self.to_bpm)),
            "mean_at": (None if self.mean_at is None else number_to_json(self.mean_at)),
            "text": self.text,
        }


class TempoField(SemanticField[Tempo]):
    """Paired Field for :class:`Tempo`."""


class Address(BaseModel):
    """Abstract root of the typed positions that name measures and beats.

    An address identifies a place in a piece the way a musician does —
    "bar 12", "the second ending of bar 15", "beat 3" — rather than by a
    position on a coordinate axis.  The address itself is inert data:
    it carries no timeline and holds no coordinate.

    Subclasses are frozen and strict: a measure label is a string, a
    measure count is an integer, and neither is silently coerced into
    the other.
    """

    model_config = ConfigDict(frozen=True, strict=True)

    rendition: int | None = None

    @classmethod
    def parse(
        cls,
        text: str,
        *,
        offset_denomination: Fraction = Fraction(1, 1),
    ) -> Address:
        """Read an address out of its printed form.

        Two forms are recognised:

        * ``"N+p/q"`` — a measure label and a within-measure offset,
          the spelling used by performance-alignment sources.  The
          offset fraction is read in *offset_denomination* whole notes
          and converted exactly to quarters, so the default reads
          ``"12+3/8"`` as three eighths of a whole note, i.e. 3/2
          quarters.
        * anything else — a bare measure label, suffix included.

        Args:
            text: The printed address.
            offset_denomination: The note value the offset fraction
                counts, as a fraction of a whole note.

        Returns:
            A :class:`MeasureNumberAddress` for the first form, a
            :class:`MeasureNumber` for the second.

        Raises:
            ValueError: If the offset part of the first form is not a
                readable fraction.
        """
        from .enums import TimeUnit

        label, separator, offset_text = text.partition("+")
        if not separator:
            return MeasureNumber(mn=text)
        try:
            offset = Fraction(offset_text)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError(
                f"Cannot read the within-measure offset of address {text!r}"
            ) from error
        quarters = offset * Fraction(offset_denomination) * 4
        return MeasureNumberAddress(
            mn=label,
            at=Coordinate(quarters, TimeUnit.quarters),
        )

    @property
    def selects_span(self) -> bool:
        """Whether this address names a whole span rather than an instant."""
        raise NotImplementedError


class MeasureNumber(Address):
    """A measure addressed by its printed label.

    Storage struct (derived): ``{mc: int64, mn: string, volta: int64}``.
    The label is the number a musician reads in the margin, kept as a
    string because it may carry a suffix (``"237b"``) or be ``"0"`` for
    an anacrusis.  ``mc`` is a lossless carry-along of the source's own
    measure count where it has one; it is never a resolution input.

    Args:
        mn: The printed label.  An integer is accepted and stringified.
        mc: The source's measure count, when known.
        volta: The ending this label belongs to, when known.
    """

    mc: int | None = None
    mn: str
    volta: int | None = None
    section: str | None = None

    @classmethod
    def from_row(cls, row: object) -> MeasureNumber | None:
        """Rebuild a :class:`MeasureNumber` from a storage row."""
        if not _is_dict(row):
            return None
        mn = row.get("mn")
        if mn is None:
            return None
        return cls(
            mn=str(mn),
            mc=row.get("mc"),
            volta=row.get("volta"),
            section=row.get("section"),
            rendition=row.get("rendition"),
        )

    @field_validator("mn", mode="before")
    @classmethod
    def _coerce_label(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int) else value

    @property
    def selects_span(self) -> bool:
        """A bare measure label names the whole bar."""
        return True

    def __repr__(self) -> str:
        parts = [f"mn={self.mn!r}"]
        if self.mc is not None:
            parts.append(f"mc={self.mc}")
        if self.volta is not None:
            parts.append(f"volta={self.volta}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self) -> str:
        return self.mn


class MeasureNumberField(SemanticField[MeasureNumber]):
    """Paired Field for :class:`MeasureNumber`.

    Empty body — :class:`MeasureNumber` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(MeasureNumber)`` dispatch.
    """

    def from_array(
        self,
        source: pa.Array[Any] | pa.ChunkedArray[Any],
        *,
        name: str | None = None,
    ) -> SemanticField[MeasureNumber]:
        """Read a column of printed labels into the ``{mc, mn, volta}`` struct.

        A source column carries the label and nothing else — the
        measure count and volta are facts a measure map holds, not facts
        a label column states — so both are left null.

        Args:
            source: The label column, of any Arrow type.
            name: Optional output field name.

        Returns:
            A live :class:`MeasureNumberField`.
        """
        if isinstance(source, pa.ChunkedArray):
            source = source.combine_chunks()
        schema = type(self).struct_schema()
        members = _member_fields(schema)
        label_type = members[schema.get_field_index("mn")].type
        labels = source if source.type == label_type else source.cast(label_type)
        arrays: list[pa.Array[Any]] = [
            labels if field.name == "mn" else pa.nulls(len(labels), type=field.type)
            for field in members
        ]
        struct_arr = _struct_array_from(arrays, members)
        out_name = name if name is not None else self.name
        return type(self).from_field((struct_arr, pa.field(out_name, schema)))


class MeasureNumberAddress(MeasureNumber):
    """A measure label plus a position inside that measure.

    The within-measure position is one polymorphic field: a
    :class:`Beat` counts under a beat-size policy, a
    :class:`~timetoalign.core.Coordinate` measures a distance from the
    bar's notated downbeat.  Both resolve the same way — measure start
    plus within-measure position — which is why they share a field
    rather than splitting the class.

    Args:
        at: The position inside the measure.
    """

    at: Beat | Coordinate

    @property
    def selects_span(self) -> bool:
        """A measure-plus-offset address names an instant."""
        return False

    def __repr__(self) -> str:
        return f"MeasureNumberAddress(mn={self.mn!r}, at={self.at!r})"

    def __str__(self) -> str:
        return f"{self.mn}+{self.at}"


class MeasureId(Address):
    """A measure addressed by its identity.

    An integer value resolves positionally — the thirteenth measure-like
    unit in the piece — while a string value is looked up against the
    measures' own identifiers.  Sources that mint identifiers as
    ``str(count)`` make the two coincide; sources with opaque
    identifiers keep them apart.  Strictness is what keeps them apart:
    ``"13"`` is an identifier, never the count 13.

    Args:
        value: A positional count or an identifier.
    """

    value: int | str

    @classmethod
    def count(cls, value: object) -> MeasureId:
        """Address the *value*-th measure-like unit.

        Args:
            value: 1-based position.

        Returns:
            A positional address.

        Raises:
            ValueError: If *value* is not an integer.
        """
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"MeasureId.count requires an int position, got {value!r}")
        return cls(value=value)

    @classmethod
    def identifier(cls, value: object) -> MeasureId:
        """Address the measure whose identifier is *value*.

        Args:
            value: The measure's identifier.

        Returns:
            An identifier address.

        Raises:
            ValueError: If *value* is not a string.
        """
        if not isinstance(value, str):
            raise ValueError(f"MeasureId.identifier requires a str id, got {value!r}")
        return cls(value=value)

    def __init__(self, value: int | str | None = None, **data: Any) -> None:
        if value is not None:
            if "value" in data:
                raise TypeError("MeasureId value was supplied twice")
            data["value"] = value
        super().__init__(**data)

    @property
    def is_positional(self) -> bool:
        """Whether this address resolves by count rather than by identifier."""
        return isinstance(self.value, int)

    @property
    def selects_span(self) -> bool:
        """A bare measure identity names the whole bar."""
        return True

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.value!r})"

    def __str__(self) -> str:
        return str(self.value)


class MeasureIdAddress(MeasureId):
    """A measure identity plus a position inside that measure.

    Args:
        at: The position inside the measure.
    """

    at: Beat | Coordinate

    @property
    def selects_span(self) -> bool:
        """A measure-plus-offset address names an instant."""
        return False

    def __repr__(self) -> str:
        return f"MeasureIdAddress({self.value!r}, at={self.at!r})"

    def __str__(self) -> str:
        return f"{self.value}+{self.at}"


class Beat(Address):
    """A beat within an implied measure scope.

    Storage struct (derived): ``{index: int64, policy: struct<grouping,
    division, name>, level: int64}``.  The index is 1-based, so beat 1
    is the downbeat.  A beat carries a :class:`BeatPolicy` only when
    it means to override the bar's own counting: without one, the beat
    is counted the way the bar's time signature counts it, and only
    resolution — which knows the bar — can say how long it is.

    Args:
        index: 1-based beat index.
        policy: Counting override, or ``None`` for the bar's default.
        level: Metrical level; ``0`` is the pulse, negative levels are
            groove below the pulse, and positive levels above it carry no
            name of their own — a reading names the levels it states.
    """

    index: int
    policy: BeatPolicy | None = None
    level: int = 0

    @classmethod
    def from_row(cls, row: object) -> Beat | None:
        """Rebuild a :class:`Beat` from a storage row."""
        if not _is_dict(row):
            return None
        index = row.get("index")
        if index is None:
            return None
        policy_row = row.get("policy")
        policy: BeatPolicy | None = None
        if _is_dict(policy_row) and policy_row.get("grouping") is not None:
            policy = BeatPolicy(
                grouping=tuple(int(entry) for entry in policy_row["grouping"]),
                division=Fraction(struct_to_rational(policy_row["division"])),
                name=policy_row.get("name"),
            )
        return cls(
            index=int(index),
            policy=policy,
            level=int(row.get("level") or 0),
            rendition=row.get("rendition"),
        )

    @field_validator("index")
    @classmethod
    def _validate_index(cls, value: int) -> int:
        if value < 1:
            raise ValueError("Beat index is 1-based; beat 1 is the downbeat")
        return value

    @property
    def is_downbeat(self) -> bool:
        """Whether this beat is the bar's downbeat."""
        return self.index == 1 and self.level == 0

    @property
    def size(self) -> Fraction | None:
        """This beat's length in quarters, or ``None`` without a policy."""
        return None if self.policy is None else self.policy.rod_for(self.index)

    @property
    def selects_span(self) -> bool:
        """A beat names the span it occupies."""
        return True

    def offset(self, policy: BeatPolicy | None = None) -> Fraction:
        """Return the distance from the bar's downbeat to this beat.

        Args:
            policy: The bar's counting, used when the beat carries none.

        Returns:
            Quarters from the downbeat.

        Raises:
            ValueError: If neither the beat nor the caller supplies a
                policy, or the index is outside the policy's bar.
        """
        effective = self.policy if self.policy is not None else policy
        if effective is None:
            raise ValueError(
                f"Beat {self.index} needs a BeatPolicy to know where it sits"
            )
        return effective.offset_for(self.index)

    def __repr__(self) -> str:
        parts = [str(self.index)]
        if self.policy is not None:
            parts.append(f"policy={self.policy!r}")
        if self.level:
            parts.append(f"level={self.level}")
        return f"Beat({', '.join(parts)})"

    def __str__(self) -> str:
        return str(self.index)


class BeatField(SemanticField[Beat]):
    """Paired Field for :class:`Beat`.

    Empty body — :class:`Beat` carries no ``@data_shaped`` methods, so
    the parity check is trivially satisfied.  The class exists to give
    the paired-class shape (``Object`` + ``ObjectField``) a stable home
    and to integrate with ``EventData.get_field(Beat)`` dispatch.
    """


MeasureNumberAddress.model_rebuild()
MeasureIdAddress.model_rebuild()


class Id(BaseModel):
    """A string identity label for a tabular event.

    Storage struct (derived): ``{value: string}``.  Used to carry an
    event-level identity through loading and analysis — typically the
    note-id string emitted by a performance-analysis tool (``n1b8xktz``,
    ``nh9xux4``, …).  Equality / hashing are inherited from pydantic's
    frozen ``BaseModel``; two ``Id`` objects compare equal iff their
    ``value`` strings match.
    """

    model_config = ConfigDict(frozen=True)

    value: str

    @classmethod
    def from_row(cls, row: object) -> Id | None:
        if not _is_dict(row):
            return None
        v = row.get("value")
        if v is None:
            return None
        return cls(value=str(v))

    def __repr__(self) -> str:
        return f"Id({self.value!r})"

    def __str__(self) -> str:
        return self.value


class IdField(SemanticField[Id]):
    """Paired Field for :class:`Id`.

    Empty body — :class:`Id` carries no ``@data_shaped`` methods, so the
    parity check is trivially satisfied.  The class exists to give the
    paired-class shape (``Object`` + ``ObjectField``) a stable home and
    to integrate with ``EventData.get_field(Id)`` dispatch.
    """


# ---------------------------------------------------------------------------
# EventReference + Projection
# ---------------------------------------------------------------------------


class EventReference(ScalarVocabulary, BaseModel):
    """The address of one event: the timeline holding it and its ID there.

    An event ID is unique on its timeline, not beyond it, so an event is
    named by the pair, kept as two values and never joined into one string.

    Storage struct (derived): ``{timeline_id: string, event_id: string}``.

    Examples:
        >>> EventReference(timeline_id="clt_6", event_id="x1azmn68")
        EventReference(timeline_id='clt_6', event_id='x1azmn68')
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeline_id: str
    event_id: str

    @field_validator("timeline_id", "event_id")
    @classmethod
    def _valid_id(cls, value: str) -> str:
        return validate_id(value)

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> EventReference | None:
        """Construct from a storage-row dict; ``None`` for a null cell."""
        if row is None:
            return None
        return cls(timeline_id=row["timeline_id"], event_id=row["event_id"])

    def __repr__(self) -> str:
        return (
            f"EventReference(timeline_id={self.timeline_id!r}, "
            f"event_id={self.event_id!r})"
        )


class EventReferenceField(SemanticField[EventReference]):
    """Paired Field for :class:`EventReference`.

    The ``projection_of`` column of a timeline's events is one: a row whose
    dict matches the struct is stored under this field type.
    """


class Event(_OnOneAxis, ScalarVocabulary, BaseModel, frozen=True):
    """An event of a timeline: its ID and its position on the timeline's axis.

    The base of the event types that are scalars of their own
    (:class:`Projection`, :class:`Box`). ``start`` and ``duration`` are on
    one axis: the duration has the start's unit and number type. An instant
    has no duration. An event made without an ID has ``id`` ``None`` until
    the timeline it is added to mints one (:meth:`Timeline.add_events`).

    Storage struct (derived): ``{id, start, duration}``, the positions as
    number structs on the axis :class:`EventField` declares.

    Examples:
        >>> Event(start=Coordinate(3, "samples")).to_event_row()
        {'event_type': 'Event', 'start': Coordinate(3, samples), 'duration': None}
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    id: str | None = None  # noqa: A003 - the event's ID, a column of its table
    start: Coordinate
    duration: Duration | None = None

    @field_validator("id")
    @classmethod
    def _valid_id(cls, value: str | None) -> str | None:
        return None if value is None else validate_id(value)

    @model_validator(mode="after")
    def _duration_on_the_start_axis(self) -> Event:
        if self.duration is not None:
            self._refuse_off_the_start_axis("duration", self.duration)
        return self

    @property
    def _name(self) -> str:
        """The event as a message names it: ``Box 'm1'``, ``A Box``."""
        kind = type(self).__name__
        return f"A {kind}" if self.id is None else f"{kind} {quote_id(self.id)}"

    @property
    def semantic_type(self) -> str:
        """The event type the event is stored under: its class name."""
        return type(self).__name__

    @property
    def unit(self) -> TimeUnit:
        """The unit of the event's axis."""
        return self.start.unit

    @property
    def number_type(self) -> NumberType:
        """The number type of the event's axis."""
        return self.start.number_type

    @property
    def end(self) -> Coordinate | None:
        """``start + duration``; ``None`` for an instant."""
        if self.duration is None:
            return None
        return self.start + self.duration

    def _refuse_off_the_start_axis(self, what: str, held: Any) -> None:
        """Refuse *held* (it has a ``unit`` and a ``number_type``) when it is
        not on the start's axis.

        Raises:
            ValueError: Naming the event, *what* and both axes.
        """
        start = self.start
        if held.unit != start.unit:
            raise ValueError(
                f"{self._name}: its {what} is in {held.unit}, its start in "
                f"{start.unit}; both are on one axis."
            )
        if held.number_type is not start.number_type:
            raise ValueError(
                f"{self._name}: its {what} is written as {held.number_type}, its "
                f"start as {start.number_type}; both are on one axis."
            )

    def to_event_row(self) -> dict[str, Any]:
        """The event as an event row of a timeline.

        The row :meth:`Timeline.add_events` and
        :meth:`~timetoalign.storage.EventData.from_dicts` take: ``event_type``
        (:attr:`semantic_type`), ``id`` only where the event holds one, then
        every further property in field order, the positions and shapes as
        the scalars they are.

        Examples:
            >>> from timetoalign.core import Coordinate, TimeUnit
            >>> row = Projection(id="p", start=Coordinate(3, TimeUnit.samples)).to_event_row()
            >>> row["event_type"], row["id"], row["start"], row["duration"]
            ('Projection', 'p', Coordinate(3, samples), None)
        """
        row: dict[str, Any] = {"event_type": self.semantic_type}
        if self.id is not None:
            row["id"] = self.id
        for name in type(self).model_fields:
            if name != "id":
                row[name] = getattr(self, name)
        return row


class EventField(SharedAxisField[Event]):
    """Paired Field for :class:`Event`: ``{id, start, duration}`` on one axis."""


class Projection(Event, frozen=True):
    """An event standing on a timeline for an event not held there.

    A Projection is an event type of its own (``event_type`` ``"Projection"``
    in a timeline's events). It holds its own ID, its position on the
    timeline's axis and, when the event it stands for is known, the
    reference to it; it repeats nothing else of that event (its pitch, its
    name, its other properties stay where the event lives). A Projection
    without ``projection_of`` stands for a position a source states without
    an event of its own.

    ``start`` and ``duration`` are on one axis: the duration has the start's
    unit and number type. An instant has no duration. A Projection made
    without an ID has ``id`` ``None`` until the timeline it is added to
    mints one (:meth:`Timeline.add_events`).

    Storage struct (derived): ``{id, start, duration, projection_of}``, the
    two positions as coordinate cells
    (:data:`~timetoalign.core.time.COORDINATE_CELL_TYPE`), each naming its
    unit and number type, both sides of the number filled. A package's
    Projections table writes them otherwise: the timeline's unit and number
    type once per row, each number once, as its number type says
    (:func:`~timetoalign.timelines.manifest.projection_table`).

    Examples:
        >>> from timetoalign.core import Coordinate, TimeUnit
        >>> Projection(
        ...     start=Coordinate(850176, TimeUnit.samples),
        ...     projection_of=EventReference(timeline_id="clt_6", event_id="x1azmn68"),
        ... ).end is None
        True
    """

    projection_of: EventReference | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> Projection | None:
        """Construct from a storage-row dict; ``None`` for a null cell.

        Each position is read from its coordinate cell, on the axis the cell
        names, so an exact value stays exact. Keys beside the four of the
        struct (a Projections table's ``timeline_id``) are not read.
        """
        if row is None:
            return None
        # ``start`` is a required member of the struct: a missing one raises
        # ``KeyError`` here, a null one is refused by the constructor.
        start: dict[str, Any] = row["start"]
        return cls(
            id=row.get("id"),
            start=cell_coordinate(start),
            duration=cell_coordinate(row.get("duration"), Duration),
            projection_of=EventReference.from_row(row.get("projection_of")),
        )

    def __repr__(self) -> str:
        parts = [] if self.id is None else [f"id={self.id!r}"]
        parts.append(f"start={self.start}")
        if self.duration is not None:
            parts.append(f"duration={self.duration}")
        if self.projection_of is not None:
            parts.append(f"projection_of={self.projection_of!r}")
        return f"Projection({', '.join(parts)})"


register_value_projector(Projection, "start", coordinate_cell_projector)
register_value_projector(Projection, "duration", coordinate_cell_projector)


class ProjectionField(SemanticField[Projection]):
    """Paired Field for :class:`Projection`.

    Every position is a coordinate cell naming its own unit and number type,
    so one field holds Projections of any timelines. A package's Projections
    table holds them in another form: a ``timeline_id`` column, the
    timeline's unit and number type once per row, each number written once
    (:func:`~timetoalign.timelines.manifest.projection_table`,
    :func:`~timetoalign.timelines.manifest.read_projection_table`).
    """

    @classmethod
    def from_projections(
        cls,
        projections: Sequence[Projection | None],
        *,
        name: str = "projection",
    ) -> ProjectionField:
        """Build the field of *projections*, in row order.

        Args:
            projections: The Projections (``None`` for a null row).
            name: The field's name.

        Returns:
            The field, its struct ``{id, start, duration, projection_of}``.
        """
        items = list(projections)
        present = [projection is not None for projection in items]
        array = _struct_array_from(
            [
                pa.array([p and p.id for p in items], pa.string()),
                coordinate_cells([p and p.start for p in items]),
                coordinate_cells([p and p.duration for p in items]),
                build_struct_array(
                    EventReference, [p and p.projection_of for p in items]
                ),
            ],
            _member_fields(cls.struct_schema()),
            mask=pa.array([not flag for flag in present], pa.bool_()),
        )
        pa_field = pa.field(
            name, array.type, metadata=blob_metadata(field_type=cls.field_type())
        )
        return cls.from_field((array, pa_field))


class Box(Event, frozen=True):
    """An area of an image as an interval event of a timeline along its x.

    A Box holds exactly one shape, a :class:`BoundingBox` or a
    :class:`Quadrilateral`, in the image's space, and the interval of the
    shape's x extent on its timeline: ``start`` is the shape's ``x_min``
    less the x at which the timeline stands in the image, ``duration`` the
    shape's x extent (``x_max - x_min``). The shape is on the start's axis
    (one unit, one number type).

    Storage struct (derived): ``{id, start, duration, bounding_box,
    quadrilateral}``, the shape not held null; every number on the axis
    :class:`BoxField` declares. In a timeline's events, ``bounding_box`` and
    ``quadrilateral`` are two columns.

    Examples:
        >>> box = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
        >>> rect = Box.from_shape(box, origin=7, id="rect_h2")
        >>> rect.start, rect.duration
        (Coordinate(378, pixels), Duration(139, pixels))
    """

    duration: Duration = Field(...)
    bounding_box: BoundingBox | None = None
    quadrilateral: Quadrilateral | None = None

    @classmethod
    def from_shape(
        cls,
        shape: Shape,
        *,
        origin: Coordinate | int | float | Fraction = 0,
        id: str | None = None,  # noqa: A002 - the event's ID
    ) -> Box:
        """The Box of *shape* on a timeline standing at x *origin* of the image.

        Args:
            shape: A :class:`BoundingBox` or a :class:`Quadrilateral`.
            origin: The image's x at which the timeline starts (a raw number in
                the shape's unit, or a ``Coordinate``); 0 for a timeline
                spanning its image from the left edge.
            id: The Box's ID; minted by the timeline when ``None``.

        Returns:
            The Box: ``start`` = ``shape.x_min - origin``, ``duration`` =
            ``shape.x_max - shape.x_min``.

        Raises:
            ValueError: If *origin* is in another unit than the shape, or is
                a value the shape's axis cannot hold exactly (``7.5`` on
                ``pixels``: never rounded).
        """
        if isinstance(origin, TimeScalar) and origin.unit != shape.unit:
            raise ValueError(
                f"The origin is in {origin.unit}, the shape in {shape.unit}; both "
                "are on one axis."
            )
        shift = _on_axis(
            origin,
            Duration,
            what="origin",
            owner="A Box",
            unit=shape.unit,
            number_type=shape.number_type,
        )
        x_min = shape.x_min
        held = (
            {"bounding_box": shape}
            if isinstance(shape, BoundingBox)
            else {"quadrilateral": shape}
        )
        return cls._built(
            id=id, start=x_min - shift, duration=shape.x_max - x_min, **held
        )

    @model_validator(mode="after")
    def _one_shape_spanning_the_duration(self) -> Box:
        if (self.bounding_box is None) is (self.quadrilateral is None):
            raise ValueError(
                "A Box holds exactly one shape: give bounding_box or quadrilateral."
            )
        shape = self.shape
        self._refuse_off_the_start_axis("shape", shape)
        x_min, x_max = shape.x_min, shape.x_max
        if (x_max - x_min).value != self.duration.value:
            raise ValueError(
                f"{self._name}: its duration is {self.duration._format_value()} and "
                f"its shape spans {(x_max - x_min)._format_value()} in x "
                f"({x_min._format_value()} to {x_max._format_value()}); a Box's "
                "duration is its shape's x extent."
            )
        return self

    def __repr__(self) -> str:
        parts = [] if self.id is None else [f"id={self.id!r}"]
        parts.append(f"start={self.start}")
        parts.append(f"duration={self.duration}")
        kind = "bounding_box" if self.bounding_box is not None else "quadrilateral"
        parts.append(f"{kind}={self.shape!r}")
        return f"Box({', '.join(parts)})"

    @property
    def shape(self) -> Shape:
        """The shape the Box holds."""
        return (
            self.bounding_box if self.bounding_box is not None else self.quadrilateral
        )


class BoxField(SharedAxisField[Box]):
    """Paired Field for :class:`Box`: every number on one declared axis."""

    @classmethod
    def from_boxes(
        cls,
        boxes: Sequence[Box | None],
        *,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        name: str = "box",
    ) -> BoxField:
        """Build the field of *boxes*, in row order.

        Args:
            boxes: The Boxes (``None`` for a null row).
            unit: Their unit; required for an empty or all-null input.
            number_type: Their number type; required likewise.
            name: The field's name.

        Returns:
            The field, its struct ``{id, start, duration, bounding_box,
            quadrilateral}``.

        Raises:
            ValueError: If the Boxes lie on two axes, or an explicit axis
                contradicts them.
        """
        return cls._from_scalars(boxes, unit, number_type, name)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MIDI EVENT (BASE + SCORE SUBCLASS)
# ═══════════════════════════════════════════════════════════════════════════


class MidiEvent(BaseModel):
    """Performance MIDI event scalar (mido-shaped). 7 cross-loader fields.

    Storage struct (derived): nested — ``pitch`` is a
    ``struct<midi_number: int64>`` (from :class:`EnharmonicPitch`); the
    other six sub-fields are nullable ``int64``.  Velocity, channel,
    track, control, value, program are all optional 0–127 (channel
    0–15, track for multi-track files).  Pitch is present for note
    events, ``None`` for Control Change / Program Change.  The wider
    score-side variant is :class:`ScoreMidiEvent`, which extends this
    base with three partitura-only fields.

    This scalar is the cross-loader intersection of the columns mido
    and partitura can produce, and is the storage shape emitted by
    ``PerformanceMidiLoader``.  Score-only fields (``voice`` /
    ``staff`` / ``part_id``) live on the subclass rather than as
    always-null columns on every performance MIDI table, so the
    storage stays minimal.
    """

    model_config = ConfigDict(frozen=True)

    pitch: EnharmonicPitch | None = None
    velocity: int | None = None
    channel: int | None = None
    track: int | None = None
    control: int | None = None
    value: int | None = None
    program: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "MidiEvent | None":
        """Reconstruct a :class:`MidiEvent` from a storage-row dict.

        Accepts the dict shape produced by ``pa.StructArray.to_pylist()``
        on the column-builder output:

        * ``pitch``: either ``None`` or a nested ``{"midi_number": int}``
          dict (translated into an :class:`EnharmonicPitch` via its own
          ``from_row``).
        * The six other fields are pass-through ``int | None``.

        Returns ``None`` for a ``None`` row (the null-struct slot).
        """
        if row is None:
            return None
        pitch_raw = row.get("pitch")
        if _is_dict(pitch_raw):
            pitch = EnharmonicPitch.from_row(pitch_raw)
        elif pitch_raw is None:
            pitch = None
        else:
            pitch = EnharmonicPitch(midi_number=int(pitch_raw))
        return cls(
            pitch=pitch,
            velocity=row.get("velocity"),
            channel=row.get("channel"),
            track=row.get("track"),
            control=row.get("control"),
            value=row.get("value"),
            program=row.get("program"),
        )

    def _repr_parts(self) -> list[str]:
        """Return ``"field=val"`` strings for every non-None field.

        Subclasses extend this (never the rendered string) so the
        inherited :meth:`__repr__` stays the single rendering site.
        """
        parts: list[str] = []
        if self.pitch is not None:
            parts.append(f"pitch={self.pitch!r}")
        if self.velocity is not None:
            parts.append(f"velocity={self.velocity}")
        if self.channel is not None:
            parts.append(f"channel={self.channel}")
        if self.track is not None:
            parts.append(f"track={self.track}")
        if self.control is not None:
            parts.append(f"control={self.control}")
        if self.value is not None:
            parts.append(f"value={self.value}")
        if self.program is not None:
            parts.append(f"program={self.program}")
        return parts

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(self._repr_parts())})"


class MidiEventField(SemanticField[MidiEvent]):
    """Paired Field for :class:`MidiEvent`.

    Empty body — :class:`MidiEvent` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(MidiEvent)`` dispatch.
    """


class ScoreMidiEvent(MidiEvent):
    """Score MIDI event scalar (partitura-shaped).

    Extends :class:`MidiEvent` with three partitura-only fields:
    ``voice``, ``staff``, ``part_id``.  The derived ``pa.StructType``
    is a **separate (wider) struct** than :class:`MidiEvent`'s — the
    column-builder produces two distinct shapes; there is no shared
    layout and no struct subtyping.  This is the storage shape
    emitted by ``ScoreMidiLoader``.
    """

    voice: int | None = None
    staff: int | None = None
    part_id: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any] | None) -> "ScoreMidiEvent | None":
        """Reconstruct a :class:`ScoreMidiEvent` from a storage-row dict.

        Accepts the wider 10-field dict shape produced by
        :func:`build_struct_array` on this scalar; mirrors
        :meth:`MidiEvent.from_row` with the three score-only fields
        (``voice`` / ``staff`` / ``part_id``) appended.
        """
        if row is None:
            return None
        pitch_raw = row.get("pitch")
        if _is_dict(pitch_raw):
            pitch = EnharmonicPitch.from_row(pitch_raw)
        elif pitch_raw is None:
            pitch = None
        else:
            pitch = EnharmonicPitch(midi_number=int(pitch_raw))
        return cls(
            pitch=pitch,
            velocity=row.get("velocity"),
            channel=row.get("channel"),
            track=row.get("track"),
            control=row.get("control"),
            value=row.get("value"),
            program=row.get("program"),
            voice=row.get("voice"),
            staff=row.get("staff"),
            part_id=row.get("part_id"),
        )

    def _repr_parts(self) -> list[str]:
        parts = super()._repr_parts()
        if self.voice is not None:
            parts.append(f"voice={self.voice}")
        if self.staff is not None:
            parts.append(f"staff={self.staff}")
        if self.part_id is not None:
            parts.append(f"part_id={self.part_id!r}")
        return parts


class ScoreMidiEventField(SemanticField[ScoreMidiEvent]):
    """Paired Field for :class:`ScoreMidiEvent`.

    Empty body — :class:`ScoreMidiEvent` carries no ``@data_shaped``
    methods, so the parity check is trivially satisfied.  The class
    exists to give the paired-class shape (``Object`` + ``ObjectField``)
    a stable home and to integrate with
    ``EventData.get_field(ScoreMidiEvent)`` dispatch.
    """
