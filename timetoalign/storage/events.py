"""EventData: PyArrow-based bulk event storage for TimeToAlign!

This module provides the EventData class which stores events in a PyArrow table
with efficient columnar operations. Events are NOT wrapped in Python objects -
they are rows in the table.

NOTE: This class was renamed from EventStore to EventData in the 2026-01 API
refactoring. EventStore now refers to the container class (formerly EventBundle)
that holds one or more EventData tables.

Design principles:
- Bulk operations are the primary API (from_dicts, from_arrays, from_dataframe)
- Schema is fixed per class, with extension points for subclasses
- Coordinates stored with both original precision and float representation
- Unit metadata at the field level (all events share same unit)
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel
from typing_extensions import Self

from timetoalign.core import (
    BoundingBoxField,
    Box,
    Coordinate,
    CoordinateSpec,
    Duration,
    Event,
    EventReferenceField,
    IntervalPolicy,
    NumberType,
    Projection,
    QuadrilateralField,
    Shape,
    TimeUnit,
    resolve_coordinate_spec,
)
from timetoalign.core.fields import SemanticField, blob_metadata, field_metadata
from timetoalign.core.ids import id_column, mint_ids, quote_id
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    SharedAxisField,
    build_number_struct_array,
    combine_number_columns,
    combine_numbers,
    compare_number_column,
    coordinate_to_struct,
    express_as,
    in_lowest_terms,
    number_cell,
    number_from_json,
    struct_to_coordinate,
)

from .mixins import SemanticFieldAccessMixin
from .schema import (
    extend_schema,
    get_base_field_names,
    is_coordinate_type,
    make_base_schema,
    make_coordinate_field,
    make_table_metadata,
    parse_table_metadata,
)

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)

_timeline_factory: (
    Callable[["EventData", str | None, dict[str, Any] | None], "Timeline"] | None
) = None


#: The members of a storage cell given whole (``number_cell`` reads it).
_NUMBER_KEYS = frozenset({"value", "numerator", "denominator"})


def _register_timeline_factory(
    factory: Callable[["EventData", str | None, dict[str, Any] | None], "Timeline"],
) -> None:
    """Register the timeline-layer constructor used by ``create_timeline``."""
    global _timeline_factory
    _timeline_factory = factory


#: The event type a :class:`~timetoalign.core.Projection` is stored under.
PROJECTION_EVENT_TYPE = Projection.__name__

#: The event type a :class:`~timetoalign.core.Box` is stored under.
BOX_EVENT_TYPE = Box.__name__

#: Extra columns whose type is declared, not read off the rows: every batch
#: types them alike, whether its rows hold a value there or none, so batches
#: concatenate. A column of a field on one axis (a Box's shape) is written in
#: the table's unit and number type, as the coordinate columns are.
_DECLARED_EXTRA_FIELDS: dict[str, type[SemanticField[Any]]] = {
    "projection_of": EventReferenceField,
    "bounding_box": BoundingBoxField,
    "quadrilateral": QuadrilateralField,
}

#: The declared columns whose numbers lie on the table's axis, each with its
#: field: a Box's shape, in the order of :data:`_DECLARED_EXTRA_FIELDS`. Only
#: a Box row holds a value there.
SHAPE_FIELDS: dict[str, type[SharedAxisField[Any]]] = {
    name: declared
    for name, declared in _DECLARED_EXTRA_FIELDS.items()
    if issubclass(declared, SharedAxisField)
}


def mint_event_ids(
    event_types: pa.Array | pa.ChunkedArray | Iterable[object],
    taken: pa.Array | pa.ChunkedArray | Iterable[str] = (),
) -> pa.Array:
    """Mint one event ID per event type, each new to *taken* and to the others.

    The prefix of an event's ID is its event type in lower case (a missing
    or empty type counts as ``Event``): ``note_N``, ``projection_N``,
    ``event_N``. The rows of one prefix take the next numbers above the
    highest held (:func:`~timetoalign.core.ids.mint_ids`, the one minter), in
    row order.

    Args:
        event_types: One event type per ID to mint, in order.
        taken: The IDs already held.

    Returns:
        A string array of the minted IDs, in the order of *event_types*.

    Examples:
        >>> mint_event_ids(["Note", "Projection", "Note", None], {"note_1"}).to_pylist()
        ['note_2', 'projection_1', 'note_3', 'event_1']
    """
    if isinstance(event_types, pa.ChunkedArray):
        event_types = event_types.combine_chunks()
    if not isinstance(event_types, pa.Array):
        event_types = pa.array(
            [None if value is None else str(value) for value in event_types],
            pa.string(),
        )
    types = pc.cast(event_types, pa.string())
    types = pc.if_else(pc.equal(pc.fill_null(types, ""), ""), "Event", types)
    prefixes = pc.utf8_lower(types)
    held = id_column(taken)
    ids = pa.nulls(len(prefixes), pa.string())
    for prefix in pc.unique(prefixes).to_pylist():
        rows = pc.equal(prefixes, prefix)
        count = pc.sum(rows).as_py()
        ids = pc.replace_with_mask(ids, rows, mint_ids(prefix, held, count))
    return ids


def projection_ids(events: pa.Table) -> pa.Array:
    """The IDs of the Projection rows of an event table, in row order.

    Args:
        events: An event table (``EventData.table``).

    Returns:
        A string array, empty where the table holds no Projection.
    """
    if events.num_rows == 0 or "event_type" not in events.column_names:
        return pa.array([], pa.string())
    rows = pc.fill_null(
        pc.equal(events.column("event_type"), PROJECTION_EVENT_TYPE), False
    )
    return pc.cast(events.column("id").filter(rows).combine_chunks(), pa.string())


def refuse_shared_projection_ids(
    ids: pa.Array | pa.ChunkedArray,
    timeline_ids: pa.Array | pa.ChunkedArray,
    *,
    remedy: str | None = None,
) -> None:
    """Refuse a Projection ID that names two Projections.

    A Projection's ID is unique among all Projections of a package's
    Projections table. A new Projection is minted (or its given ID checked)
    against every Projection of the manifest it is added to; timelines that
    joined after creating their Projections may hold one ID each, which the
    package writer refuses.

    Args:
        ids: Projection IDs.
        timeline_ids: The ID of the timeline holding each, row for row.
        remedy: A sentence appended to the message saying how to resolve
            the collision, or None.

    Raises:
        ValueError: Naming the first ID given twice, in row order, and the
            timelines of its first two rows.
    """
    counts = pc.value_counts(ids)
    shared = counts.filter(pc.greater(counts.field("counts"), 1))
    if not len(shared):
        return
    shared_id = shared.field("values")[0].as_py()
    first, second = pc.filter(timeline_ids, pc.equal(ids, shared_id)).to_pylist()[:2]
    raise ValueError(
        f"Projection ID {quote_id(shared_id)} names a Projection on the timeline "
        f"{quote_id(first)} and one on {quote_id(second)}: a Projection's ID is "
        f"unique among all Projections of a manifest and its package"
        + ("" if remedy is None else f". {remedy}")
    )


def _event_rows(
    rows: Event | Sequence[dict[str, Any] | Event],
) -> list[dict[str, Any]]:
    """The event rows of *rows*: an event (a Projection, a Box) as its row, a
    dict as it is."""
    if isinstance(rows, Event):
        rows = [rows]
    return [row.to_event_row() if isinstance(row, Event) else row for row in rows]


def _refuse_shape_off_the_axis(
    value: Shape, key: str, unit: TimeUnit, number_type: NumberType
) -> None:
    """Refuse a shape in another unit or number type than the events.

    A shape is the source's statement in its own numbers: it is never
    converted into another unit nor rounded into another number type.

    Raises:
        ValueError: Naming the column and both axes.
    """
    if (value.unit, value.number_type) != (unit, number_type):
        raise ValueError(
            f"The event's {key} is in {value.unit} as {value.number_type}, the "
            f"events are in {unit} as {number_type}; a shape is never converted "
            "onto the events' axis."
        )


def refuse_shapes_off_the_axis(
    rows: Sequence[dict[str, Any]], unit: TimeUnit, number_type: NumberType
) -> None:
    """Refuse the rows if a shape among them is not on the events' axis.

    The one rule :meth:`EventData.from_dicts` applies to a shape, for a
    caller that must refuse before it converts anything else of the rows
    (:meth:`Timeline.add_events <timetoalign.timelines.Timeline.add_events>`,
    whose positions may be converted through a C-Map).

    Raises:
        ValueError: As :meth:`EventData.from_dicts` refuses a shape off the
            axis.
    """
    for row in rows:
        for key in SHAPE_FIELDS:
            value = row.get(key)
            if isinstance(value, Shape):
                _refuse_shape_off_the_axis(value, key, unit, number_type)


def _is_cell_of(value: Any, kind: pa.DataType) -> bool:
    """Whether *value* is a memory cell of *kind*: number structs nested as
    *kind* nests them, every number holding its value and its ratio."""
    if not isinstance(value, dict):
        return False
    if kind == RATIONAL_STRUCT_TYPE:
        if set(value) != {"value", "numerator", "denominator"}:
            return False
        ratio = (value["numerator"], value["denominator"])
        return isinstance(value["value"], (int, float)) and all(
            isinstance(member, int) and not isinstance(member, bool) for member in ratio
        )
    if not pa.types.is_struct(kind):
        return False
    return set(value) == {member.name for member in kind} and all(
        _is_cell_of(value[member.name], member.type) for member in kind
    )


def _axis_cells(
    rows: list[dict[str, Any]],
    key: str,
    declared: type[SharedAxisField[Any]],
    unit: TimeUnit,
    number_type: NumberType,
) -> list[Any]:
    """The cells of a column of a Box's shape, row by row.

    Only a Box row holds a value there. A shape (a
    :class:`~timetoalign.core.BoundingBox` under ``bounding_box``, a
    :class:`~timetoalign.core.Quadrilateral` under ``quadrilateral``)
    becomes its struct of number cells; a memory cell of the column (a
    table's own row, read back) is taken as it is.

    Raises:
        ValueError: If a row of another event type gives the column, a Box
            gives it a value that is neither the column's shape nor its
            memory cell, or a shape is in another unit or number type than
            the table: a shape is never converted or rounded onto an axis.
    """
    shape_class = declared.scalar_cls
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        event_type = row.get("event_type")
        if event_type != BOX_EVENT_TYPE:
            raise ValueError(
                f"An event of type {quote_id(str(event_type))} gives {key}: the "
                "columns bounding_box and quadrilateral hold the shape of a Box "
                "and belong to Box events only."
            )
        if isinstance(value, shape_class):
            _refuse_shape_off_the_axis(value, key, unit, number_type)
        elif not _is_cell_of(value, declared.pa_schema):
            hint = (
                f" Read a dict form with {shape_class.__name__}.from_dict."
                if isinstance(value, dict)
                else ""
            )
            raise ValueError(
                f"A Box's {key} is a {shape_class.__name__}, not {value!r}.{hint}"
            )
    values = [row.get(key) for row in rows]
    scalars = [value if isinstance(value, Shape) else None for value in values]
    built = declared.from_shapes(scalars, unit=unit, number_type=number_type)
    return [
        cells if isinstance(value, Shape) else value
        for value, cells in zip(values, built.to_pyarrow().to_pylist(), strict=True)
    ]


def _native_position(
    value: Any, key: str, unit: TimeUnit, number_type: NumberType
) -> Any:
    """A position given as a Coordinate or Duration, as the number it stores.

    A scalar in the table's unit is written in the table's number type;
    other values are returned as given.

    Raises:
        ValueError: If the scalar is in another unit: no table converts a
            unit, a timeline does (:meth:`Timeline.add_events`).
    """
    if not isinstance(value, (Coordinate, Duration)):
        return value
    if value.unit != unit:
        raise ValueError(
            f"The event's {key} is in {value.unit}, the events are in {unit}; "
            "add it to a timeline, which converts it through its C-Map"
        )
    return express_as(value.value, number_type)


def _cell_column(column: Any, n: int) -> pa.StructArray:
    """A coordinate column as number cells; all null where there is none."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if not isinstance(column, pa.StructArray):
        return pa.nulls(n, RATIONAL_STRUCT_TYPE)
    return column.cast(RATIONAL_STRUCT_TYPE)


def _completed_column(
    cells: pa.StructArray,
    rows: np.ndarray,
    left: pa.StructArray,
    right: pa.StructArray,
    operation: str,
    number_type: NumberType,
) -> pa.StructArray:
    """*cells*, holding ``left <operation> right`` in the selected *rows*.

    Interval completion (``duration = end - start``, ``end = start +
    duration``) runs in the number type the table declares, through the
    library's column arithmetic (:func:`combine_number_columns`): a float
    axis computes on its doubles and mirrors each result once, an exact
    axis computes on its ratios, exactly, and refuses (naming the row) only
    a result whose lowest terms do not fit int64. The rows not selected
    keep their cells.
    """
    if not rows.any():
        return cells
    selected = np.flatnonzero(rows)
    target = "duration (end - start)" if operation == "subtract" else "end"
    with _naming_the_computation(f"Completing the events' {target}", number_type):
        computed = combine_number_columns(
            left.take(selected),
            right.take(selected),
            operation,
            number_type,
            row_numbers=selected,
        )
    positions = np.arange(len(cells))
    positions[selected] = len(cells) + np.arange(len(selected))
    return pa.concat_arrays([cells, computed]).take(pa.array(positions))


@contextmanager
def _naming_the_computation(what: str, number_type: NumberType) -> Iterator[None]:
    """Prefix a refused computation's message with *what* and the axis."""
    try:
        yield
    except ValueError as error:
        article = "an" if number_type is NumberType.int else "a"
        raise ValueError(f"{what} on {article} {number_type} axis: {error}") from error


def _completed_cell(
    left: dict[str, Any],
    right: dict[str, Any],
    operation: str,
    number_type: NumberType,
) -> dict[str, Any]:
    """One cell of :func:`_completed_column`, by the same rule."""
    return number_cell(
        combine_numbers(
            struct_to_coordinate(left, number_type),
            struct_to_coordinate(right, number_type),
            operation,
            number_type,
        ),
        number_type,
    )


_FLOAT_TOLERANCE = 1e-9
"""How far ``end - start`` may lie from a stated ``duration`` on a float axis."""


def _inconsistent_rows(
    start: pa.StructArray,
    end: pa.StructArray,
    duration: pa.StructArray,
    rows: np.ndarray,
    number_type: NumberType,
) -> np.ndarray:
    """The selected *rows* whose stated duration is not ``end - start``.

    The comparison runs in the number type the table declares. On an exact
    axis (``int``, ``fraction``) it is exact: ``end - start`` is computed in
    that number type (:func:`combine_number_columns`) and compared with the
    duration as a ratio (:func:`compare_number_column`), so a duration off by
    ``2**-40`` of a quarter is inconsistent. On a ``float`` axis it is a
    float comparison on the doubles, within :data:`_FLOAT_TOLERANCE`, since
    the difference of two stated doubles is itself rounded.
    :func:`_inconsistency` decides a single row by the same rule.
    """
    inconsistent = np.zeros(len(rows), dtype=bool)
    if not rows.any():
        return inconsistent
    selected = np.flatnonzero(rows)
    if number_type is NumberType.float:

        def _doubles(cells: pa.StructArray) -> np.ndarray:
            return cells.field("value").to_numpy(zero_copy_only=False)[selected]

        expected = _doubles(end) - _doubles(start)
        inconsistent[selected] = np.abs(expected - _doubles(duration)) > (
            _FLOAT_TOLERANCE
        )
        return inconsistent
    indices = pa.array(selected)
    with _naming_the_computation(
        "Checking the events' durations against end - start", number_type
    ):
        expected = combine_number_columns(
            end.take(indices),
            start.take(indices),
            "subtract",
            number_type,
            row_numbers=selected,
        )
    equal = compare_number_column(
        duration.take(indices), expected, "equal", number_type
    )
    inconsistent[selected] = ~pc.fill_null(equal, True).to_numpy(zero_copy_only=False)
    return inconsistent


def _inconsistency(
    start: dict[str, Any],
    end: dict[str, Any],
    duration: dict[str, Any],
    number_type: NumberType,
) -> str | None:
    """The values of a row whose stated duration is not ``end - start``.

    The rule of :func:`_inconsistent_rows` for a single row: exact on an
    ``int`` or ``fraction`` axis, a float comparison within
    :data:`_FLOAT_TOLERANCE` on a ``float`` axis.

    Returns:
        The row's values in the table's number type, or None if the row is
        consistent.
    """
    start_value, end_value, duration_value = (
        struct_to_coordinate(cell, number_type) for cell in (start, end, duration)
    )
    expected = combine_numbers(end_value, start_value, "subtract", number_type)
    if number_type is NumberType.float:
        consistent = not abs(expected - duration_value) > _FLOAT_TOLERANCE
    else:
        consistent = expected == duration_value
    if consistent:
        return None
    return (
        f"start={start_value}, end={end_value}, duration={duration_value} "
        f"(expected duration={expected})"
    )


def _extreme_coordinate(
    cells: pa.ChunkedArray,
    values: pa.ChunkedArray,
    extreme: pa.Scalar,
    number_type: NumberType,
    pick: Callable[[Iterable[Any]], Any],
) -> int | float | Fraction:
    """The smallest or largest number of a column of number cells, decided exactly.

    ``extreme`` is the float extreme of the cells' ``value`` members. A cell's
    double is its number correctly rounded, and rounding never reverses an
    order, so the exact extreme is among the cells whose double equals it.
    On a float axis those cells hold one number (equal doubles differ at most
    in the sign of zero) and the first of them answers, as ``min`` and ``max``
    over the column in order do. On an exact axis the distinct candidate cells
    are read exactly and ``pick`` decides among them, so two ratios sharing a
    double are still ordered by their exact values.

    Args:
        cells: The number cells, nulls included.
        values: Their ``value`` members.
        extreme: The minimum or maximum of ``values``.
        number_type: The axis's declared number type.
        pick: ``min`` or ``max``.

    Returns:
        The extreme in the axis's number type.
    """
    candidates = pc.equal(values, extreme)
    if number_type is NumberType.float:
        return values[pc.index(candidates, True).as_py()].as_py()
    chosen = pa.Table.from_struct_array(pc.filter(cells, candidates).combine_chunks())
    distinct = chosen.group_by(chosen.column_names).aggregate([])
    return pick(
        struct_to_coordinate(cell, number_type) for cell in distinct.to_pylist()
    )


def _in_lowest_terms(table: pa.Table) -> pa.Table:
    """*table* with the ratio of every coordinate cell in lowest terms.

    Memory holds one spelling of each value (:func:`in_lowest_terms`): a
    cell stating ``2/4`` is held as ``1/2``. Every coordinate column is
    reduced, the base ones and any further one, and every number of a
    column on the table's axis (a Box's shape); a column's field and its
    metadata are kept. One vectorized pass per number column.
    """
    for index, field in enumerate(table.schema):
        if _is_number_column(field):
            table = table.set_column(index, field, in_lowest_terms(table.column(index)))
        elif field.name in SHAPE_FIELDS and pa.types.is_struct(field.type):
            table = table.set_column(
                index, field, _leaves_in_lowest_terms(table.column(index))
            )
    return table


def _leaves_in_lowest_terms(column: pa.Array | pa.ChunkedArray) -> pa.Array:
    """*column*, a struct nesting number structs, with each of them reduced."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    if column.type == RATIONAL_STRUCT_TYPE:
        return in_lowest_terms(column)
    if not pa.types.is_struct(column.type):
        return column
    return pa.StructArray.from_arrays(
        [_leaves_in_lowest_terms(member) for member in column.flatten()],
        fields=list(column.type),
        mask=column.is_null(),
    )


def _is_number_column(field: pa.Field) -> bool:
    """Whether *field* holds coordinate cells with an int64 ratio side."""
    return is_coordinate_type(field.type) and all(
        field.type.field(member).type == pa.int64()
        for member in ("numerator", "denominator")
    )


def _on_number_type(
    table: pa.Table, unit: TimeUnit, number_type: NumberType, target: NumberType
) -> pa.Table:
    """*table*, on *unit* in *number_type*, with its numbers written in *target*.

    The number-type rule of every axis: each value lands in the type the
    receiving table declares, ``float`` → ``fraction`` as the double's exact
    ratio, ``fraction`` → ``float`` as the nearest double, onto ``int``
    nearest with ties to even (as :func:`~timetoalign.core.time.express_as`
    writes one value). Every coordinate column is re-expressed and its
    field declares *target*. ``start`` and ``duration`` are the stored pair
    of an event and ``end`` is derived from them: on an ``int`` or
    ``fraction`` target a row holding ``start`` and ``duration`` then ends
    at ``start + duration``, exactly, so it keeps its duration and stays
    consistent in the exact arithmetic the table reads it in (a re-expressed
    end, rounded or the exact ratio of a double, need not lie at the
    re-expressed start plus the re-expressed duration). Vectorized, one pass
    per column.

    Raises:
        ValueError: If an exact ratio does not fit int64 (a double below
            about ``2**-10`` on a ``fraction`` target).
    """
    for index, field in enumerate(table.schema):
        if not _is_number_column(field):
            continue
        declared = field_metadata(field)
        source = (
            NumberType(declared["number_type"])
            if "number_type" in declared
            else number_type
        )
        if source is target:
            continue
        cells = _reexpressed(table.column(index), source, target)
        payload = {key: declared[key] for key in declared if key != "version"}
        payload.setdefault("unit", unit.value)
        payload["number_type"] = target.name
        retyped = pa.field(
            field.name, field.type, field.nullable, metadata=blob_metadata(**payload)
        )
        table = table.set_column(index, retyped, cells)
    names = table.schema.names
    if target is not NumberType.float and {"start", "end", "duration"} <= set(names):
        start, duration = table.column("start"), table.column("duration")
        ends = combine_number_columns(
            start.combine_chunks(), duration.combine_chunks(), "add", target
        )
        index = names.index("end")
        end = pc.if_else(
            pc.and_(start.is_valid(), duration.is_valid()),
            ends,
            table.column(index).combine_chunks(),
        )
        table = table.set_column(index, table.schema.field(index), end)
    return table


def _reexpressed(
    cells: pa.Array | pa.ChunkedArray, source: NumberType, target: NumberType
) -> pa.StructArray:
    """A column of number cells declared *source*, written in *target*."""
    if isinstance(cells, pa.ChunkedArray):
        cells = cells.combine_chunks()
    if source is NumberType.int and target is NumberType.fraction:
        return cells  # n/1 is a fraction cell as it stands
    if source is NumberType.fraction and target is NumberType.int:
        return _nearest_integers(cells)
    # The float side mirrors the canonical one; flatten() nulls every
    # member of a null cell, which the builder reads as missing.
    doubles = cells.flatten()[cells.type.get_field_index("value")]
    return build_number_struct_array(doubles, number_type=target)


def _nearest_integers(cells: pa.StructArray) -> pa.StructArray:
    """Integer cells of the ratios of *cells*, nearest, ties to even, exactly.

    Computed on the ratio side in int64: ``n = q·d + r`` with ``0 <= r <
    d``, rounded up where ``r > d - r`` or, at a tie, where ``q`` is odd.
    """
    members = cells.flatten()
    names = cells.type.names
    numerator = members[names.index("numerator")]
    denominator = members[names.index("denominator")]
    missing = pc.or_(numerator.is_null(), denominator.is_null())
    n = numerator.fill_null(0).to_numpy(zero_copy_only=False).astype(np.int64)
    d = denominator.fill_null(1).to_numpy(zero_copy_only=False).astype(np.int64)
    quotient, remainder = np.divmod(n, d)
    rest = d - remainder
    up = (remainder > rest) | ((remainder == rest) & (quotient % 2 == 1))
    rounded = quotient + up.astype(np.int64)
    mask = missing.to_numpy(zero_copy_only=False)
    return pa.StructArray.from_arrays(
        [
            pa.array(rounded.astype(np.float64), pa.float64(), mask=mask),
            pa.array(rounded, pa.int64(), mask=mask),
            pa.array(np.ones(len(rounded), np.int64), pa.int64(), mask=mask),
        ],
        fields=list(RATIONAL_STRUCT_TYPE),
        mask=pa.array(mask),
    )


def _has_value_subfield(pa_type: pa.DataType) -> bool:
    """Return True iff *pa_type* is a struct with a ``value`` sub-field.

    Matches the canonical ``{value, numerator, denominator}`` coordinate
    struct as well as reduced shapes that carry only ``value`` (e.g. a
    column built without the fraction components), so
    :meth:`EventData.column_values` decodes any rational-shaped column,
    not just the full three-field one.
    """
    if not pa.types.is_struct(pa_type):
        return False
    return any(pa_type.field(i).name == "value" for i in range(pa_type.num_fields))


def _first_non_null(rows: list[dict[str, Any]], name: str) -> Any:
    """Return the first non-null value carried under *name*, or ``None``."""
    for row in rows:
        val = row.get(name)
        if val is not None:
            return val
    return None


def _struct_fields_match_unordered(
    actual: pa.StructType, expected: pa.StructType
) -> bool:
    """Return True iff *actual* and *expected* have the same ``{name: type}`` set.

    Order-independent counterpart to ``core.fields._struct_types_match``:
    ``pa.array`` infers struct sub-fields in alphabetical order, which may
    differ from a paired class's canonical ``pa_schema`` order, so a carried
    struct dict must be matched by name-set, then reordered to canonical.
    """
    if not pa.types.is_struct(actual):
        return False
    if actual.num_fields != expected.num_fields:
        return False
    actual_map = {
        actual.field(i).name: actual.field(i).type for i in range(actual.num_fields)
    }
    for i in range(expected.num_fields):
        e = expected.field(i)
        a_type = actual_map.get(e.name)
        if a_type is None or not a_type.equals(e.type):
            return False
    return True


def _semantic_field_type_for_struct(
    struct_type: pa.StructType,
) -> tuple[str, pa.StructType] | None:
    """Return the paired ``SemanticField`` matching *struct_type* by name-set.

    Walks the scalar→field registry and returns the ``(field_type_str,
    canonical_pa_schema)`` of the first paired ``SemanticField`` whose
    ``pa_schema`` has the same sub-field ``{name: type}`` set as
    *struct_type* (order-independent).  Returns ``None`` when no paired
    class claims the shape — the column then stays a plain struct with no
    semantic identity.

    The generic coordinate struct ``{value, numerator, denominator}`` is
    deliberately excluded: ``start`` / ``end`` / ``duration`` already live
    in the base schema, so a stray coordinate-shaped extra column should
    stay a plain struct rather than masquerade as a CoordinateField.
    """
    from timetoalign.storage.mixins import _get_field_type_map

    for field_type_str, field_cls in _get_field_type_map().items():
        schema = field_cls.pa_schema
        if schema is None:
            continue
        names = [schema.field(i).name for i in range(schema.num_fields)]
        if names == ["value", "numerator", "denominator"]:
            continue
        if _struct_fields_match_unordered(struct_type, schema):
            return field_type_str, schema
    return None


def _build_extra_field_schema(
    schema: pa.Schema,
    extra_field_names: list[str],
    processed_rows: list[dict[str, Any]],
    unit: TimeUnit,
    number_type: NumberType,
) -> pa.Schema:
    """Append extra (non-base-schema) fields to *schema*, preserving shape.

    Each extra column is materialised under the most faithful PyArrow type
    its values support, so that the semantic-field affordance of a carried
    column survives the round-trip:

    * **struct-shaped** values (a ``dict`` sample) become a real
      ``pa.struct`` column whose type is inferred from the dicts.  When the
      inferred struct matches a paired ``SemanticField``'s ``pa_schema``,
      the field is stamped with the ``TIMETOALIGN_METADATA_KEY`` ``field_type``
      metadata blob so ``get_field()`` round-trips the semantic type
      directly (no reliance on shape discovery).
    * **list-shaped** values become a ``pa.list_`` column.
    * **scalar** values keep their native inferred type (int / float /
      bool / string).  Genuinely unrepresentable objects fall back to
      ``str`` — but a struct ``dict`` is NEVER serialised to a JSON string,
      because that would silently destroy the field's struct affordance.

    Mutates *processed_rows* in place to null-fill absent columns and to
    coerce non-native scalars to ``str``.

    Args:
        schema: The base + declared-extra schema to extend.
        extra_field_names: Sorted names of the carried columns not already
            in *schema*.
        processed_rows: The row dicts about to be turned into a table.
        unit: The table's unit, which a declared column on the table's axis
            is written in.
        number_type: The table's number type, likewise.

    Returns:
        A new ``pa.Schema`` with the extra fields appended.
    """
    from timetoalign.core.fields import (
        TIMETOALIGN_METADATA_KEY,
        metadata_blob_from_dict,
    )

    new_fields = list(schema)
    for name in extra_field_names:
        declared = _DECLARED_EXTRA_FIELDS.get(name)
        if declared is not None and issubclass(declared, SharedAxisField):
            cells = _axis_cells(
                processed_rows,
                name,
                declared,
                unit,
                number_type,
            )
            for row, cell in zip(processed_rows, cells):
                row[name] = cell
            new_fields.append(
                declared.from_field(
                    pa.field(name, declared.pa_schema),
                    unit=unit,
                    number_type=number_type,
                ).to_field()
            )
            continue
        if declared is not None:
            for row in processed_rows:
                value = row.get(name)
                row[name] = (
                    value.model_dump() if isinstance(value, BaseModel) else value
                )
            new_fields.append(
                pa.field(
                    name,
                    declared.pa_schema,
                    nullable=True,
                    metadata={
                        TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                            {"field_type": declared.field_type()}
                        )
                    },
                )
            )
            continue

        if name == "volta":
            for row in processed_rows:
                value = row.get(name)
                row[name] = int(value) if value is not None else None
            new_fields.append(pa.field(name, pa.int64(), nullable=True))
            continue

        sample = _first_non_null(processed_rows, name)

        if isinstance(sample, dict):
            # Infer the struct type from the carried dicts and keep it as a
            # real struct column.  Stamp paired-class metadata when the
            # shape is a known SemanticField so the affordance round-trips.
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                # Heterogeneous dict shapes that PyArrow cannot unify: keep
                # the column out of the typed path rather than crash.  Fall
                # back to a string rendering (last resort, logged).
                module_logger.warning(
                    "Carried struct column %r has heterogeneous shapes; "
                    "storing as string.",
                    name,
                )
                for row in processed_rows:
                    val = row.get(name)
                    row[name] = None if val is None else str(val)
                new_fields.append(pa.field(name, pa.string(), nullable=True))
                continue
            metadata = None
            struct_type = arr.type
            if pa.types.is_struct(struct_type):
                match = _semantic_field_type_for_struct(struct_type)
                if match is not None:
                    field_type_str, canonical_schema = match
                    # Use the paired class's canonical sub-field order so
                    # the stored struct round-trips through from_row and is
                    # claimed by shape discovery; from_pylist coerces the
                    # carried dicts to this order.
                    struct_type = canonical_schema
                    metadata = {
                        TIMETOALIGN_METADATA_KEY: metadata_blob_from_dict(
                            {"field_type": field_type_str}
                        )
                    }
            new_fields.append(
                pa.field(name, struct_type, nullable=True, metadata=metadata)
            )
            continue

        if isinstance(sample, list):
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
                new_fields.append(pa.field(name, arr.type, nullable=True))
                continue
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                pass  # fall through to string rendering below

        if sample is not None and not isinstance(sample, (str, bytes)):
            # Scalar of a native arrow type (int / float / bool): infer and
            # keep the native type so numeric carried columns stay numeric.
            values = [row.get(name) for row in processed_rows]
            try:
                arr = pa.array(values)
                if not pa.types.is_null(arr.type):
                    new_fields.append(pa.field(name, arr.type, nullable=True))
                    continue
            except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError):
                pass  # fall through to string rendering below
            # Unrepresentable scalar object: render to string.
            for row in processed_rows:
                val = row.get(name)
                if val is not None and not isinstance(val, str):
                    row[name] = str(val)
                elif name not in row:
                    row[name] = None
            new_fields.append(pa.field(name, pa.string(), nullable=True))
            continue

        # String (or all-null) column.
        for row in processed_rows:
            if name not in row:
                row[name] = None
        new_fields.append(pa.field(name, pa.string(), nullable=True))

    return pa.schema(new_fields, metadata=schema.metadata)


class EventData(SemanticFieldAccessMixin):
    """PyArrow-based storage for timeline events.

    EventData wraps a PyArrow table containing events. Events are rows in the
    table, not Python wrapper objects. The primary API is bulk operations:

    - from_dicts(): Create from list of row dictionaries
    - from_arrays(): Create from field-oriented arrays
    - from_dataframe(): Create from pandas DataFrame

    The schema is fixed at class definition time but can be extended by
    subclasses to add domain-specific fields (e.g., pitch, velocity for notes).

    NOTE: This class was renamed from EventStore to EventData in the 2026-01 API
    refactoring. EventStore now refers to the container class (formerly EventBundle)
    that holds one or more EventData tables.

    Attributes:
        table: The underlying PyArrow table.
        unit: The time unit for all coordinates.
        number_type: The number type used for coordinates.

    Examples:
        >>> data = EventData.from_dicts([
        ...     {"id": "e1", "temporal_type": "instant", "event_type": "Beat",
        ...      "instant": 0.0},
        ...     {"id": "e2", "temporal_type": "interval", "event_type": "Note",
        ...      "start": 0.0, "end": 1.0},
        ... ], unit=TimeUnit.seconds)
        >>> len(data)
        2
    """

    # Class-level schema configuration (subclasses can extend)
    _extra_fields: ClassVar[list[pa.Field]] = []

    def __init__(
        self,
        table: pa.Table,
        unit: TimeUnit,
        number_type: NumberType | None = None,
    ) -> None:
        """Initialize EventData with an existing table.

        Use class methods (from_dicts, from_arrays, etc.) to create instances.

        Every coordinate cell is reduced on entry: a ratio stated as ``2/4``
        is accepted and held as ``1/2``, so cells compare by value.

        Args:
            table: The PyArrow table containing events.
            unit: The time unit for coordinates.
            number_type: The number type for coordinate interpretation.
                Defaults to the one the unit itself uses.
        """
        # A table of no rows holds no cell to reduce.
        self._hold(
            _in_lowest_terms(table) if table.num_rows else table, unit, number_type
        )

    def _hold(
        self, table: pa.Table, unit: TimeUnit, number_type: NumberType | None
    ) -> None:
        """Hold *table*, as it stands, on *unit* in *number_type*."""
        self._table = table
        self._unit = unit
        self._number_type = unit.resolve_number_type(number_type)

    @classmethod
    def _of_reduced(
        cls, table: pa.Table, unit: TimeUnit, number_type: NumberType | None = None
    ) -> Self:
        """An instance over *table*, whose coordinate cells are in lowest terms.

        The constructor of tables derived from tables already held (rows
        selected, joined, retyped): their cells were reduced when they
        entered, so the reduction :meth:`__init__` runs on every table from
        outside is not run again. It sets the base's state only, so it serves
        classes whose ``__init__`` adds none; an instance of any class derives
        a table through :meth:`_derived`.
        """
        data = cls.__new__(cls)
        data._hold(table, unit, number_type)
        return data

    def _derived(self, table: pa.Table) -> Self:
        """This EventData's class and state over *table*, derived from its own.

        *table* holds cells of tables already held, so they are in lowest
        terms and not reduced again (:meth:`_of_reduced`). The state a
        subclass sets (``NoteEventData.has_rests``) is kept; the cached
        field views are not, since they wrap the former table.
        """
        derived = copy.copy(self)
        derived._table = table
        derived._reset_field_cache()
        return derived

    # region Schema

    @property
    def schema(self) -> pa.Schema:
        """The PyArrow schema of the underlying table.

        Returns:
            The schema of the stored PyArrow table, including all base and
            extra fields with their metadata.
        """
        return self._table.schema

    @classmethod
    def get_schema(
        cls, unit: TimeUnit, number_type: NumberType | None = None
    ) -> pa.Schema:
        """Get the canonical PyArrow schema for this EventData class.

        This is a class-level method that returns the schema for a given unit,
        independent of any specific instance. Useful for constructing empty
        tables or validating incoming data.

        Args:
            unit: The time unit for coordinate fields.
            number_type: The number type for coordinate fields.

        Returns:
            The complete schema including base and extra fields.
        """
        base = make_base_schema(unit, number_type=number_type)
        if cls._extra_fields:
            return extend_schema(base, cls._extra_fields)
        return base

    @classmethod
    def field_names(cls) -> list[str]:
        """Get the list of field names for this EventData class.

        Returns:
            List of all field names (base + extra).
        """
        base_names = get_base_field_names()
        extra_names = [f.name for f in cls._extra_fields]
        return base_names + extra_names

    # endregion

    # region Interval Normalisation

    @staticmethod
    def _normalize_intervals_vectorized(
        processed: dict[str, Any],
        policy: IntervalPolicy = IntervalPolicy.warn,
        *,
        number_type: NumberType,
    ) -> dict[str, Any]:
        """Normalise interval events in a field dict (FULLY VECTORIZED).

        Ensures that every interval event has both ``end`` and ``duration``
        populated, every instant event has both null, and that the
        ``temporal_type`` field is consistent. A missing ``end`` or
        ``duration`` is completed in *number_type* (see
        :func:`_completed_column`).

        The *policy* parameter controls behaviour when both ``end`` and
        ``duration`` are present but inconsistent
        (``end - start != duration``). Consistency is decided in
        *number_type* (:func:`_inconsistent_rows`): exactly on an ``int``
        or ``fraction`` axis, as a float comparison within ``1e-9`` on a
        ``float`` axis.

        Args:
            processed: Mutable field dict (modified **in-place**).
                Must already contain ``"start"`` as a ``pa.StructArray``.
                ``"end"`` and ``"duration"`` may be ``pa.StructArray``,
                ``pa.NullArray``, or absent.
            policy: How to resolve inconsistencies between ``end`` and
                ``duration``.
            number_type: The number type the table declares for its
                coordinates.

        Returns:
            The same *processed* dict (for chaining convenience).

        Raises:
            ValueError: If *policy* is ``strict`` and an inconsistency is
                detected.
        """

        start_arr = processed.get("start")
        if start_arr is None or not isinstance(start_arr, pa.StructArray):
            return processed

        n = len(start_arr)
        start = _cell_column(start_arr, n)
        end = _cell_column(processed.get("end"), n)
        duration = _cell_column(processed.get("duration"), n)

        # Boolean masks for each situation (vectorized)
        has_start = start.is_valid().to_numpy(zero_copy_only=False)
        temporal_type = processed.get("temporal_type")
        if temporal_type is None:
            instant_rows = np.zeros(n, dtype=bool)
        else:
            if isinstance(temporal_type, (pa.Array, pa.ChunkedArray)):
                temporal_values = temporal_type.to_pylist()
            else:
                temporal_values = list(temporal_type)
            instant_rows = np.array(
                [value == "instant" for value in temporal_values], dtype=bool
            )

        # Explicit instant events have no interval endpoints.  In particular,
        # do not let a populated start coordinate manufacture a zero-length
        # value-only interval during merged-table assembly.
        has_end = end.is_valid().to_numpy(zero_copy_only=False) & ~instant_rows
        has_dur = duration.is_valid().to_numpy(zero_copy_only=False) & ~instant_rows

        # ---- Policy-driven resolution ----

        # Rows where both end AND duration are present (potential conflict)
        both_present = has_start & has_end & has_dur
        inconsistent = _inconsistent_rows(
            start, end, duration, both_present, number_type
        )
        if inconsistent.any():
            n_bad = int(inconsistent.sum())
            first_idx = int(np.argmax(inconsistent))
            first = _inconsistency(
                *(cells[first_idx].as_py() for cells in (start, end, duration)),
                number_type,
            )
            if policy == IntervalPolicy.strict:
                raise ValueError(
                    f"Interval inconsistency in {n_bad} event(s): "
                    f"end - start != duration. First at index {first_idx}: "
                    f"{first}."
                )
            elif policy == IntervalPolicy.warn:
                module_logger.warning(
                    "Interval inconsistency in %d event(s): end - start != "
                    "duration. Recomputing duration from end. First at index "
                    "%d: %s.",
                    n_bad,
                    first_idx,
                    first,
                )

        # ---- Fill missing values ----
        #
        # After this block every interval row has both end and duration.
        # The policy controls which value is authoritative when both are
        # already present.

        # 1) Rows with end but no duration -> compute duration = end - start
        need_dur = has_start & has_end & ~has_dur
        if policy == IntervalPolicy.prefer_end:
            # Also recompute duration for rows that already have both
            need_dur = need_dur | both_present
        elif policy in (IntervalPolicy.warn, IntervalPolicy.strict):
            # Recompute duration for inconsistent rows (prefer end)
            need_dur = need_dur | (both_present & inconsistent)
        duration = _completed_column(
            duration, need_dur, end, start, "subtract", number_type
        )

        # 2) Rows with duration but no end -> compute end = start + duration
        need_end = has_start & has_dur & ~has_end
        if policy == IntervalPolicy.prefer_duration:
            # Also recompute end for rows that already have both
            need_end = need_end | both_present
        end = _completed_column(end, need_end, start, duration, "add", number_type)

        # Explicit instants carry no interval endpoints.
        if instant_rows.any():
            no_cells = pa.nulls(n, RATIONAL_STRUCT_TYPE)
            end = pc.if_else(pa.array(instant_rows), no_cells, end)
            duration = pc.if_else(pa.array(instant_rows), no_cells, duration)
        processed["end"] = end
        processed["duration"] = duration

        # ---- Ensure temporal_type is consistent ----
        # Recompute: interval if end is now non-null, instant otherwise
        new_has_end = end.is_valid().to_numpy(zero_copy_only=False)
        inferred_tt = np.where(new_has_end, "interval", "instant")

        # Only overwrite temporal_type if it was all-null or not provided;
        # otherwise just fix the rows that changed.
        tt = processed.get("temporal_type")
        if tt is None:
            processed["temporal_type"] = inferred_tt
        elif isinstance(tt, pa.Array):
            if tt.null_count == len(tt):
                processed["temporal_type"] = inferred_tt
            else:
                # Overwrite individual rows where our inference differs
                existing = tt.to_pylist()
                for i in range(n):
                    if new_has_end[i] and existing[i] != "interval":
                        existing[i] = "interval"
                    elif not new_has_end[i] and existing[i] != "instant":
                        existing[i] = "instant"
                processed["temporal_type"] = pa.array(existing, type=pa.string())
        elif isinstance(tt, np.ndarray):
            # Update mismatches
            tt[new_has_end & (tt != "interval")] = "interval"
            tt[~new_has_end & (tt != "instant")] = "instant"

        return processed

    @staticmethod
    def _normalize_intervals_row(
        processed: dict[str, Any],
        policy: IntervalPolicy = IntervalPolicy.warn,
        *,
        number_type: NumberType,
    ) -> dict[str, Any]:
        """Normalise interval fields in a single event row dict.

        Ensures that ``end`` and ``duration`` are both present (or both
        absent) and consistent with ``start``.  Also ensures that the
        ``temporal_type`` field is set correctly.  Coordinate fields are
        converted to struct-dict format via ``coordinate_to_struct``.

        This is the **row-based** counterpart to
        ``_normalize_intervals_vectorized`` and is called from
        ``from_dicts``; both decide consistency alike
        (:func:`_inconsistency`: exact on an ``int`` or ``fraction``
        axis, a float comparison within ``1e-9`` on a ``float`` axis) and
        complete a row alike (:func:`_completed_cell`).

        Args:
            processed: Mutable row dict (modified **in-place**).
            policy: How to resolve inconsistencies.
            number_type: The number type the table declares for its
                coordinates.

        Returns:
            The same *processed* dict (for chaining convenience).

        Raises:
            ValueError: If *policy* is ``strict`` and ``end - start !=
                duration``.
        """
        # ---- Ensure coordinate struct format ----
        for coord_col in ("start", "end", "duration"):
            val = processed.get(coord_col)
            if val is not None:
                if isinstance(val, dict):
                    if "value" not in val:
                        processed[coord_col] = coordinate_to_struct(val)
                    elif set(val) == _NUMBER_KEYS:
                        # A storage cell given whole, its ratio reduced: memory
                        # holds one spelling of a value (2/4 is held as 1/2).
                        processed[coord_col] = number_cell(val)
                elif isinstance(val, list):
                    processed[coord_col] = coordinate_to_struct(number_from_json(val))
                else:
                    processed[coord_col] = coordinate_to_struct(val)
            elif coord_col not in processed:
                processed[coord_col] = None

        if processed.get("temporal_type") == "instant":
            processed["end"] = None
            processed["duration"] = None
            return processed

        has_start = processed.get("start") is not None
        has_end = processed.get("end") is not None
        has_dur = processed.get("duration") is not None

        # ---- Consistency check when both present ----
        values = (
            _inconsistency(
                processed["start"], processed["end"], processed["duration"], number_type
            )
            if has_start and has_end and has_dur
            else None
        )
        inconsistent = values is not None
        if inconsistent:
            if policy == IntervalPolicy.strict:
                raise ValueError(f"Interval inconsistency: {values}.")
            elif policy == IntervalPolicy.warn:
                module_logger.warning(
                    "Interval inconsistency: %s. Recomputing duration from end.",
                    values,
                )
            # After warning: fall through to fill logic

        # ---- Fill missing values ----
        # The same rows as the vectorized path: duration from end where it
        # is missing, where prefer_end says so, or where warn/strict found
        # it inconsistent; end from duration where it is missing or where
        # prefer_duration says so.
        if has_start:
            if has_end and (
                not has_dur
                or policy == IntervalPolicy.prefer_end
                or (
                    inconsistent
                    and policy in (IntervalPolicy.warn, IntervalPolicy.strict)
                )
            ):
                processed["duration"] = _completed_cell(
                    processed["end"], processed["start"], "subtract", number_type
                )
            elif has_dur and (not has_end or policy == IntervalPolicy.prefer_duration):
                processed["end"] = _completed_cell(
                    processed["start"], processed["duration"], "add", number_type
                )

        # ---- Infer temporal_type ----
        if "temporal_type" not in processed or processed["temporal_type"] is None:
            now_has_end = processed.get("end") is not None
            now_has_dur = processed.get("duration") is not None
            if has_start and (now_has_end or now_has_dur):
                processed["temporal_type"] = "interval"
            else:
                processed["temporal_type"] = "instant"

        return processed

    # endregion

    # region Class Methods - Creation

    @classmethod
    def empty(cls, unit: TimeUnit, number_type: NumberType | None = None) -> Self:
        """Create an empty EventData.

        Args:
            unit: The time unit for coordinates.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.

        Returns:
            An empty EventData with the appropriate schema.
        """
        number_type = unit.resolve_number_type(number_type)
        return cls(cls._empty_table(unit, number_type), unit, number_type)

    @classmethod
    def _empty_table(cls, unit: TimeUnit, number_type: NumberType) -> pa.Table:
        """The table of no events of this class on *unit* in *number_type*."""
        schema = cls.get_schema(unit, number_type=number_type)
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)
        return pa.table({name: [] for name in cls.field_names()}, schema=schema)

    @classmethod
    def from_dicts(
        cls,
        rows: Event | Sequence[dict[str, Any] | Event],
        unit: TimeUnit,
        number_type: NumberType | None = None,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> Self:
        """Create EventData from a list of row dictionaries.

        Coordinate values (instant, start, end, duration) are automatically
        converted to the internal struct format. Convenience defaults are
        applied so that callers can omit boilerplate fields:

        - **id**: Minted if missing, above the IDs the rows give, its prefix
          the event type in lower case: ``note_1``, ``beat_1``,
          ``projection_1``.
        - **temporal_type**: Inferred from the keys present in the dict --
          ``"interval"`` when *both* ``start`` and ``end`` (or ``duration``)
          are given, ``"instant"`` otherwise.

        Missing ``end`` or ``duration`` values are computed automatically
        from the other (``end = start + duration`` or
        ``duration = end - start``).  Behaviour when both are present but
        inconsistent is controlled by *interval_policy*.

        Args:
            rows: Event dictionaries, or :class:`~timetoalign.core.Event`
                objects such as Projections and Boxes (one or a list), each
                standing for its event row
                (:meth:`~timetoalign.core.Event.to_event_row`). At
                minimum each dict needs a coordinate (``instant`` *or*
                ``start``/``end``) and an ``event_type``. A position may be
                a ``Coordinate`` or ``Duration`` in *unit*; a Box's shape
                (``bounding_box``, ``quadrilateral``) is in *unit* and
                *number_type*, given as the shape or as the cell a table's
                row holds; no other event type gives these two columns. All
                other fields have sensible defaults.
            unit: The time unit for coordinates.
            number_type: The number type for coordinates.
            interval_policy: How to handle end/duration inconsistencies.
                See `IntervalPolicy` for options.

        Returns:
            A new EventData containing the events.

        Raises:
            ValueError: If a position is in another unit, or a shape in
                another unit or number type, than the table; if a row of
                another event type than Box gives a shape column, or a Box
                gives one a value that is neither its shape nor its cell.

        Examples:
            >>> data = EventData.from_dicts([
            ...     {"event_type": "Beat", "instant": 0},
            ...     {"event_type": "Note", "start": 0, "end": 0.5},
            ... ], unit=TimeUnit.seconds)
        """
        number_type = unit.resolve_number_type(number_type)
        table = cls._rows_table(
            rows, unit, number_type, interval_policy=interval_policy
        )
        return cls(table, unit, number_type)

    @classmethod
    def _rows_table(
        cls,
        rows: Event | Sequence[dict[str, Any] | Event],
        unit: TimeUnit,
        number_type: NumberType,
        *,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> pa.Table:
        """The table :meth:`from_dicts` makes of *rows*, before it is held."""
        rows = _event_rows(rows)
        if not rows:
            return cls._empty_table(unit, number_type)

        # Mint the IDs the rows lack, above the IDs the rows give.
        minted = iter(
            mint_event_ids(
                [row.get("event_type") for row in rows if row.get("id") is None],
                [row["id"] for row in rows if row.get("id") is not None],
            ).to_pylist()
        )

        # Convert coordinate values to struct format
        processed_rows = []
        for row in rows:
            processed = dict(row)
            if processed.get("id") is None:
                processed["id"] = next(minted)
            for key in ("instant", "start", "end", "duration"):
                if key in processed:
                    processed[key] = _native_position(
                        processed[key], key, unit, number_type
                    )

            # Map 'instant' to 'start'
            if "instant" in processed and processed.get("start") is None:
                processed["start"] = processed.pop("instant")

            # Remove 'instant' key if it remains
            processed.pop("instant", None)

            # Unified interval normalisation: converts coordinate fields
            # to struct format, fills missing end/duration, infers
            # temporal_type, and checks consistency per policy.
            cls._normalize_intervals_row(
                processed, policy=interval_policy, number_type=number_type
            )

            # Ensure name field exists
            if "name" not in processed:
                processed["name"] = None
            processed_rows.append(processed)

        schema = cls.get_schema(unit, number_type=number_type)
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)

        # Collect extra fields not in the base schema and add them dynamically
        base_field_names = set(schema.names)
        extra_field_names: set[str] = set()
        for row in processed_rows:
            for key in row.keys():
                if key not in base_field_names:
                    extra_field_names.add(key)

        if extra_field_names:
            schema = _build_extra_field_schema(
                schema, sorted(extra_field_names), processed_rows, unit, number_type
            )

        return pa.Table.from_pylist(processed_rows, schema=schema)

    @classmethod
    def from_arrays(
        cls,
        fields: dict[str, np.ndarray | pa.Array | list[Any]],
        unit: TimeUnit,
        number_type: NumberType | None = None,
        *,
        validate: bool = True,
        extra_fields: list[pa.Field] | None = None,
        interval_policy: IntervalPolicy = IntervalPolicy.warn,
    ) -> Self:
        """Create EventData from field-oriented arrays (VECTORIZED).

        This is the PRIMARY construction method for loaders. All operations
        are vectorized - NO row iteration occurs.

        Missing ``end`` or ``duration`` values are computed automatically
        from the other (``end = start + duration`` or
        ``duration = end - start``).  Behaviour when both are present but
        inconsistent is controlled by *interval_policy*.

        Args:
            fields: Dict mapping field names to arrays. Supports:
                - np.ndarray: NumPy arrays
                - pa.Array: PyArrow arrays (including StructArray for coords)
                - list: Python lists (converted to numpy)

                For coordinate fields (start, end, duration):
                - If pa.StructArray: taken whole; any ratio is accepted and
                  reduced on entry (``2/4`` is held as ``1/2``)
                - If numeric/string array: parsed via build_number_struct_array

            unit: The time unit for coordinates.
            number_type: The number type for coordinates.
            validate: Whether to validate arrays before table construction.
            extra_fields: Optional list of PyArrow fields for extra data.
                These fields include metadata (e.g., unit for CoordinateFields).
                If not provided, fields are inferred from the data arrays.
            interval_policy: How to handle end/duration inconsistencies.
                See `IntervalPolicy` for options.

        Returns:
            A new EventData containing the events.

        Raises:
            ValueError: If validation fails (missing fields, length mismatch, etc.)

        Examples:
            >>> # Vectorized construction from arrays
            >>> data = EventData.from_arrays({
            ...     "id": np.array(["e1", "e2"]),
            ...     "temporal_type": np.array(["instant", "instant"]),
            ...     "event_type": np.array(["Beat", "Beat"]),
            ...     "start": build_number_struct_array([0, 480], number_type=NumberType.int),
            ... }, unit=TimeUnit.ticks)

            >>> # Direct from loader output (StructArrays already parsed)
            >>> data = EventData.from_arrays(loader_fields, unit=TimeUnit.quarters)
        """
        number_type = unit.resolve_number_type(number_type)
        if not fields:
            return cls.empty(unit, number_type)

        # Check if any field has data
        first_arr = next(iter(fields.values()), None)
        if first_arr is None or len(first_arr) == 0:
            return cls.empty(unit, number_type)

        n_rows = len(first_arr)

        # Helper to get field array (with instant->start mapping)
        def get_arr(name: str) -> Any:
            if name == "start" and "start" not in fields and "instant" in fields:
                return fields["instant"]
            return fields.get(name)

        # Build processed dict for PyArrow table
        processed: dict[str, Any] = {}
        schema = cls.get_schema(unit, number_type=number_type)

        for pa_field in schema:
            field_name = pa_field.name
            arr_data = get_arr(field_name)

            if field_name in ("start", "end", "duration"):
                # Coordinate fields - may be StructArray or need parsing
                if arr_data is None:
                    # Create null struct array (vectorized)
                    processed[field_name] = pa.nulls(n_rows, type=pa_field.type)
                elif isinstance(arr_data, (pa.Array, pa.ChunkedArray)):
                    arr = (
                        arr_data.combine_chunks()
                        if isinstance(arr_data, pa.ChunkedArray)
                        else arr_data
                    )
                    if isinstance(arr, pa.StructArray):
                        # Cells given whole keep their members, each ratio
                        # reduced: memory holds one spelling of a value.
                        processed[field_name] = (
                            in_lowest_terms(arr)
                            if {"numerator", "denominator"} <= set(arr.type.names)
                            else arr
                        )
                    elif pa.types.is_null(arr.type):
                        processed[field_name] = pa.nulls(n_rows, type=pa_field.type)
                    else:
                        null_mask = pc.is_null(arr, nan_is_null=True)
                        fill_value: str | int = (
                            "0"
                            if pa.types.is_string(arr.type)
                            or pa.types.is_large_string(arr.type)
                            else 0
                        )
                        filled = pc.if_else(
                            null_mask, pa.scalar(fill_value, type=arr.type), arr
                        )
                        parsed = build_number_struct_array(
                            filled.to_numpy(zero_copy_only=False),
                            number_type=number_type,
                        )
                        processed[field_name] = pa.StructArray.from_arrays(
                            [parsed.field(i) for i in range(parsed.type.num_fields)],
                            fields=list(parsed.type),
                            mask=null_mask,
                        )
                else:
                    # Need to build the storage cells from a raw column
                    arr = _as_numpy(arr_data)
                    # Handle None/NaN values (create mask)
                    if arr.dtype == object:
                        # Check for None values
                        mask = pd.Series(arr).isna().to_numpy()
                        if mask.any():
                            # Create valid array and null array, combine
                            valid_indices = ~mask
                            if valid_indices.any():
                                valid_arr = arr[valid_indices]
                                parsed = build_number_struct_array(
                                    valid_arr, number_type=number_type
                                )
                                # Build full array with nulls (VECTORIZED)
                                # Extract parsed struct fields
                                parsed_values = parsed.field("value").to_numpy(
                                    zero_copy_only=False
                                )
                                parsed_nums = parsed.field("numerator").to_numpy(
                                    zero_copy_only=False
                                )
                                parsed_dens = parsed.field("denominator").to_numpy(
                                    zero_copy_only=False
                                )

                                # Create full arrays with None placeholders (vectorized)
                                full_values = np.full(n_rows, np.nan, dtype=np.float64)
                                full_nums = np.full(n_rows, np.nan, dtype=np.float64)
                                full_dens = np.full(n_rows, np.nan, dtype=np.float64)

                                # Place valid values using boolean indexing (vectorized)
                                full_values[valid_indices] = parsed_values
                                full_nums[valid_indices] = parsed_nums
                                full_dens[valid_indices] = parsed_dens

                                # Convert to PyArrow with proper nulls
                                processed[field_name] = pa.StructArray.from_arrays(
                                    [
                                        pa.array(full_values),
                                        pa.array(
                                            np.where(
                                                np.isnan(full_nums), None, full_nums
                                            ).astype(object),
                                            type=pa.int64(),
                                        ),
                                        pa.array(
                                            np.where(
                                                np.isnan(full_dens), None, full_dens
                                            ).astype(object),
                                            type=pa.int64(),
                                        ),
                                    ],
                                    names=["value", "numerator", "denominator"],
                                    mask=pa.array(mask),
                                )
                            else:
                                processed[field_name] = pa.nulls(
                                    n_rows, type=pa_field.type
                                )
                        else:
                            processed[field_name] = build_number_struct_array(
                                arr, number_type=number_type
                            )
                    else:
                        processed[field_name] = build_number_struct_array(
                            arr, number_type=number_type
                        )
            elif field_name == "id":
                if arr_data is not None:
                    # Ensure string type (vectorized)
                    if isinstance(arr_data, np.ndarray):
                        processed[field_name] = pa.array(arr_data.astype(str))
                    elif isinstance(arr_data, pa.Array):
                        processed[field_name] = arr_data.cast(pa.string())
                    else:
                        processed[field_name] = pa.array(
                            [str(x) for x in arr_data], type=pa.string()
                        )
                else:
                    # Mint the IDs from the event types, if there are any.
                    event_types = get_arr("event_type")
                    if event_types is None:
                        event_types = pa.nulls(n_rows, pa.string())
                    elif isinstance(event_types, np.ndarray):
                        event_types = event_types.tolist()
                    processed[field_name] = mint_event_ids(event_types)
            elif arr_data is not None:
                # Regular field - convert to PyArrow array
                if isinstance(arr_data, (pa.Array, pa.ChunkedArray)):
                    processed[field_name] = arr_data
                elif isinstance(arr_data, np.ndarray):
                    processed[field_name] = pa.array(arr_data)
                else:
                    processed[field_name] = pa.array(arr_data)
            else:
                # Field not provided - fill with nulls
                processed[field_name] = pa.nulls(n_rows, type=pa_field.type)

        # Unified interval normalisation: compute missing end/duration,
        # check consistency, and infer temporal_type -- all vectorized.
        cls._normalize_intervals_vectorized(
            processed, policy=interval_policy, number_type=number_type
        )

        # Infer event_type if not provided or all null (vectorized)
        if "event_type" in processed:
            et_arr = processed["event_type"]
            if isinstance(et_arr, pa.Array) and et_arr.null_count == len(et_arr):
                # All null - use default
                processed["event_type"] = pa.array(["Event"] * n_rows)

        # Handle extra fields not in base schema
        # These are fields passed by loaders via extra_fields configuration
        base_field_names = set(schema.names)
        # In the order given, so the columns are the same in every process.
        extra_field_names = [
            name
            for name in fields
            if name not in base_field_names and name != "instant"
        ]

        # Build lookup for provided extra_fields (has proper metadata)
        provided_fields: dict[str, pa.Field] = {}
        if extra_fields:
            for f in extra_fields:
                provided_fields[f.name] = f

        inferred_fields = []
        for name in extra_field_names:
            data = fields[name]
            if data is None:
                continue

            # Infer PyArrow type from the data
            if isinstance(data, (pa.Array, pa.ChunkedArray)):
                arr = data
                if isinstance(arr, pa.ChunkedArray):
                    arr = arr.combine_chunks()
            elif isinstance(data, np.ndarray):
                arr = pa.array(data)
            else:
                arr = pa.array(data)

            processed[name] = arr

            # Use provided field if available (has metadata from CoordinateField etc.)
            if name in provided_fields:
                inferred_fields.append(provided_fields[name])
            elif is_coordinate_type(arr.type):
                # A further coordinate column is on this table's axis: its
                # unit and number type are written as the base columns'
                # are, so it reads back in the type it was given in.
                inferred_fields.append(
                    make_coordinate_field(
                        name, unit, nullable=True, number_type=number_type
                    ).with_type(arr.type)
                )
            else:
                inferred_fields.append(pa.field(name, arr.type, nullable=True))

        # Extend schema with extra fields
        if inferred_fields:
            schema = extend_schema(schema, inferred_fields)

        # Validate arrays if requested (vectorized validation)
        if validate:
            ArrayValidator.validate_field_dict(processed, schema)

        # Build table in single operation
        metadata = make_table_metadata(unit, number_type, loader_class=cls.__name__)
        schema = schema.with_metadata(metadata)

        table = pa.table(processed, schema=schema)
        return cls(table, unit, number_type)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        unit: TimeUnit,
        number_type: NumberType | None = None,
    ) -> Self:
        """Create EventData from a pandas DataFrame.

        Args:
            df: DataFrame with event data. Field names should match the schema.
            unit: The time unit for coordinates.
            number_type: The number type for coordinates.

        Returns:
            A new EventData containing the events.
        """
        if df.empty:
            return cls.empty(unit, number_type)

        return cls.from_dicts(df.to_dict("records"), unit, number_type)

    @classmethod
    def from_parquet(cls, path: Path | str) -> Self:
        """Load EventData from a Parquet file.

        Args:
            path: Path to the Parquet file.

        Returns:
            An EventData loaded from the file.

        Raises:
            ValueError: If the file lacks required TimeToAlign! metadata.
        """
        table = pq.read_table(path)
        metadata = parse_table_metadata(table.schema)

        if not metadata:
            raise ValueError(f"File {path} lacks TimeToAlign! metadata")

        unit = TimeUnit(metadata["unit"])
        number_type = NumberType(metadata["number_type"])

        return cls(table, unit, number_type)

    # endregion

    # region Properties

    @property
    def table(self) -> pa.Table:
        """The underlying PyArrow table."""
        return self._table

    @property
    def unit(self) -> TimeUnit:
        """The time unit for coordinates."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The number type for coordinate interpretation."""
        return self._number_type

    @property
    def count(self) -> int:
        """The number of events in the store."""
        return self._table.num_rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the number of events."""
        return self.count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over events as dictionaries."""
        for batch in self._table.to_batches():
            for row in batch.to_pylist():
                yield row

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"count={self.count}, unit={self.unit}, number_type={self.number_type})"
        )

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Lists the event count, unit, number type, and the semantic fields
        reachable via :meth:`get_field` — each as ``name : ScalarType``
        with the backing column name — followed by the typical next-step
        invocations.
        """
        from timetoalign.display.html import affordance_html, code
        from timetoalign.storage.mixins import (
            _get_field_type_map,
            _parse_field_type_metadata,
        )

        rows: list[tuple[str, str]] = [
            ("Events", str(self.count)),
            ("Unit", str(self.unit)),
            ("Number type", str(self.number_type)),
        ]

        # Enumerate reachable semantic fields best-effort from the table
        # schema's TIMETOALIGN_METADATA_KEY field_type metadata (the same source
        # get_field uses). Columns with no paired field are skipped.
        registry = _get_field_type_map()
        field_items: list[str] = []
        listed_columns: set[str] = set()
        schema = self._table.schema
        for i in range(len(schema)):
            pa_field = schema.field(i)
            ft_str = _parse_field_type_metadata(pa_field)
            if ft_str is None:
                continue
            field_cls = registry.get(ft_str)
            if field_cls is None:
                continue
            scalar_cls = field_cls.scalar_cls
            scalar_name = scalar_cls.__name__ if scalar_cls is not None else "—"
            field_items.append(f"<li>{code(pa_field.name)} : {scalar_name}</li>")
            listed_columns.add(pa_field.name)

        # Also surface fields *afforded* over a raw atomic column
        # (represent-once): the typed pitch view is reachable via
        # get_field() / get_pitch_field() even though the column itself
        # carries no semantic metadata, so an honest "reachable fields"
        # listing must include it.
        column_names = set(self._table.column_names)
        afforded: dict[str, Any] = getattr(type(self), "_afforded_fields", {})
        for column, afforded_cls in afforded.items():
            if column in listed_columns or column not in column_names:
                continue
            scalar_cls = afforded_cls.scalar_cls
            scalar_name = scalar_cls.__name__ if scalar_cls is not None else "—"
            field_items.append(f"<li>{code(column)} : {scalar_name}</li>")
            listed_columns.add(column)

        if field_items:
            fields_value = "<ul>" + "".join(field_items) + "</ul>"
        else:
            fields_value = "(none)"
        rows.append(("Fields", fields_value))

        affordances = ["get_field(<Scalar>)", "get_pitch_field()", "get_raw('<col>')"]
        return affordance_html(type(self).__name__, rows, affordances=affordances)

    # endregion

    # region Extend/Merge

    def _joining(self, other: "EventData") -> pa.Table:
        """*other*'s table as it joins this one: on this table's axis.

        Rows of another number type are re-expressed in this table's
        (:func:`_on_number_type`), as every value landing on an axis is;
        rows of another unit cannot be.

        Raises:
            ValueError: If *other* has another unit, or one of its values
                cannot be stored in this table's number type (an exact ratio
                past int64).
        """
        if other.unit != self.unit:
            raise ValueError(f"Unit mismatch: {self.unit} vs {other.unit}")
        if other.number_type is self.number_type:
            return other._table
        return _on_number_type(
            other._table, other.unit, other.number_type, self.number_type
        )

    def extend(self, other: "EventData") -> None:
        """Extend this data with events from another EventData (in-place).

        Rows of another number type join re-expressed in this table's: a
        float as its exact ratio on a ``fraction`` table, a ratio as the
        nearest float on a ``float`` table, nearest (ties to even) on an
        ``int`` table. On an ``int`` or ``fraction`` table an interval
        keeps its start and duration, so re-expressed, and ends at their
        exact sum.

        Args:
            other: Another EventData of the same unit (extra fields are
                allowed and will be merged using schema promotion).

        Raises:
            ValueError: If the units don't match, or a value of *other*
                cannot be stored in this table's number type.
        """
        joining = self._joining(other)

        # Use promote_options="default" to handle schema differences
        # (e.g., extra fields from JSON loaders). Missing fields are
        # filled with nulls.
        self._table = pa.concat_tables(
            [self._table, joining], promote_options="default"
        )
        # The table was replaced in place; any cached SemanticField (incl.
        # an afforded pitch view over a raw column) now wraps the stale
        # pre-extend array, so drop the cache and let the affordance
        # re-attach over the concatenated table on next access.
        self._invalidate_field_cache()

    def concat(self, *others: "EventData") -> "EventData":
        """Concatenate with other EventData, returning a new EventData.

        The result has this table's unit and number type; rows of another
        number type join re-expressed in it, as :meth:`extend` says.

        Args:
            *others: Other EventData of the same unit to concatenate (extra
                fields are allowed and will be merged using schema
                promotion).

        Returns:
            A new EventData containing all events.

        Raises:
            ValueError: If any units don't match, or a value cannot be
                stored in this table's number type.
        """
        tables = [self._table, *(self._joining(other) for other in others)]

        # Use promote_options="default" to handle schema differences
        new_table = pa.concat_tables(tables, promote_options="default")
        return self._derived(new_table)

    def prefix_ids(self, prefix: str) -> "EventData":
        """Return a new EventData with all event IDs prefixed.

        Prepends ``prefix:`` to every event ID. Used when events are placed
        onto a timeline so that IDs become globally unique and informative,
        e.g. ``clt_1:note:000001``.

        If the IDs already start with the prefix, they are left unchanged.

        Args:
            prefix: The prefix to prepend (without trailing colon).

        Returns:
            A new EventData with prefixed IDs.
        """
        id_arr = self._table.column("id")
        prefix_str = f"{prefix}:"
        new_ids = pa.array(
            [
                f"{prefix_str}{v}" if not v.startswith(prefix_str) else v
                for v in id_arr.to_pylist()
            ],
            type=pa.string(),
        )
        idx = self._table.schema.get_field_index("id")
        new_table = self._table.set_column(idx, self._table.schema.field(idx), new_ids)
        return self._derived(new_table)

    def _with_segment_renamed(self, old_id: str, new_id: str) -> "EventData":
        """Return a copy whose segment row for child *old_id* names *new_id*.

        A timeline records each child as one ``Segment`` row whose ``id`` and
        ``name`` are the child's id; this keeps that row in step when the
        child's id changes.
        """
        table = self._table
        names = table.schema.names
        if "event_type" not in names or "id" not in names:
            return self
        mask = pc.fill_null(
            pc.and_kleene(
                pc.equal(table.column("event_type"), "Segment"),
                pc.equal(table.column("id"), old_id),
            ),
            False,
        )
        for column in ("id", "name"):
            if column not in names:
                continue
            idx = names.index(column)
            values = table.column(idx)
            renamed = pc.if_else(mask, pa.scalar(new_id, values.type), values)
            table = table.set_column(idx, table.schema.field(idx), renamed)
        return self._derived(table)

    # endregion

    # region Query/Filter

    def filter(
        self,
        *,
        temporal_type: Literal["instant", "interval"] | None = None,
        event_type: str | None = None,
        min_coord: CoordinateSpec | None = None,
        max_coord: CoordinateSpec | None = None,
        **kwargs: Any,
    ) -> "EventData":
        """Filter events by criteria, returning a new EventData.

        All criteria are AND-ed together.

        Args:
            temporal_type: Filter by "instant" or "interval".
            event_type: Filter by event type name.
            min_coord: Minimum coordinate (inclusive), optionally with a unit.
            max_coord: Maximum coordinate (exclusive), optionally with a unit.
            **kwargs: Exact match filters for other fields (e.g. event_category="note").

        Coordinate bounds are compared with each event's ``start``
        (:func:`~timetoalign.core.time.compare_number_column`). An exact
        bound (an ``int``, a ``Fraction``) is compared exactly on an ``int``
        or ``fraction`` table (ratios cross-multiplied, so positions closer
        than a float can tell apart are told apart). A bound written as a
        float is compared as a key, with the float side of the cells: on a
        ``fraction`` table ``min_coord=0.1`` keeps a start at ``1/10``, and
        ``max_coord=math.inf`` keeps every row. A ``float`` table compares
        floats whatever the bound. The minimum is inclusive and the maximum
        exclusive. A NaN bound is refused (``ValueError``).

        Returns:
            A new EventData with filtered events.
        """
        mask = None

        if temporal_type is not None:
            expr = pc.equal(pc.field("temporal_type"), temporal_type)
            mask = expr if mask is None else (mask & expr)

        if event_type is not None:
            expr = pc.equal(pc.field("event_type"), event_type)
            mask = expr if mask is None else (mask & expr)

        bounds = []
        for bound, comparison in ((min_coord, "greater_equal"), (max_coord, "less")):
            if bound is None:
                continue
            resolved = resolve_coordinate_spec(bound)
            if resolved.unit is not None and resolved.unit != self.unit:
                raise ValueError(
                    f"Coordinate unit mismatch: {resolved.unit} vs {self.unit}"
                )
            bounds.append((resolved.value, comparison))

        # Generic kwargs filtering
        for name, val in kwargs.items():
            # Only if field exists in schema
            if name in self.field_names():
                expr = pc.equal(pc.field(name), val)
                mask = expr if mask is None else (mask & expr)

        if mask is None and not bounds:
            return self

        filtered = self._table if mask is None else self._table.filter(mask)
        for value, comparison in bounds:
            filtered = filtered.filter(
                compare_number_column(
                    filtered.column("start"), value, comparison, self.number_type
                )
            )
        return self._derived(filtered)

    def column_values(self, name: str, *, default: Any = None) -> list[Any]:
        """Return a column's values as plain, decoded Python objects.

        A rational-shaped column -- any struct column carrying a ``value``
        sub-field, such as the canonical ``{value, numerator, denominator}``
        coordinate struct used by ``start``, ``end``, ``duration``, and any
        extra field sharing that shape -- decodes to an exact `Fraction`
        via `struct_to_rational`. When a row's struct carries no exact
        ratio (missing or non-integral numerator/denominator), the float
        ``value`` member is used instead, wrapped in a `Fraction`. Every
        other column type is returned via ``to_pylist()`` unchanged,
        including ``None`` for a null cell -- *default* only substitutes
        for a null rational-shaped struct or a column absent from this
        table entirely.

        Args:
            name: Field name in the underlying table.
            default: Value substituted for every row when *name* is not a
                column of this table, and for individual null rational
                struct cells.

        Returns:
            A list of per-row values, one for each event.
        """
        if name not in self._table.column_names:
            return [default] * len(self)

        pa_field = self._table.schema.field(name)
        column = self._table.column(name)

        if not _has_value_subfield(pa_field.type):
            return column.to_pylist()

        values: list[Any] = []
        for cell in column.to_pylist():
            if cell is None:
                values.append(default)
                continue
            try:
                values.append(struct_to_coordinate(cell, NumberType.fraction))
            except (ValueError, TypeError, ZeroDivisionError):
                values.append(default)
        return values

    def select(self, fields: list[str]) -> pa.Table:
        """Select specific fields from the table.

        Args:
            fields: List of field names to select.

        Returns:
            A PyArrow table with only the selected fields.
        """
        return self._table.select(fields)

    def where(self, expression: pc.Expression) -> "EventData":
        """Filter with a custom PyArrow compute expression.

        Args:
            expression: A PyArrow compute expression.

        Returns:
            A new EventData with filtered events.
        """
        filtered = self._table.filter(expression)
        return self._derived(filtered)

    # endregion

    # region Stats/Overview

    def count_by(self, field: str) -> dict[str, int]:
        """Count events grouped by a field's values.

        Args:
            field: The field to group by.

        Returns:
            Dict mapping field values to counts.
        """
        result = self._table.group_by(field).aggregate([(field, "count")])
        return {row[field]: row[f"{field}_count"] for row in result.to_pylist()}

    def coordinate_range(self) -> tuple[float | Fraction, float | Fraction] | None:
        """Get the min and max coordinates across all events.

        Returns:
            Tuple of (min, max) coordinates, or None if store is empty.
            Returns Fraction values when number_type is fraction.
        """
        if self.count == 0:
            return None
        lows: list[int | float | Fraction] = []
        highs: list[int | float | Fraction] = []
        for name in ("start", "end"):
            cells = self._table.column(name)
            if pa.types.is_dictionary(cells.type):
                # The kernels read struct members, which a dictionary-encoded
                # column does not offer: decode it, chunk by chunk.
                cells = pa.chunked_array(
                    [chunk.dictionary_decode() for chunk in cells.chunks],
                    type=cells.type.value_type,
                )
            values = pc.struct_field(cells, "value")
            if values.null_count == len(values):
                continue
            extremes = pc.min_max(values)
            lows.append(
                _extreme_coordinate(
                    cells, values, extremes["min"], self._number_type, min
                )
            )
            highs.append(
                _extreme_coordinate(
                    cells, values, extremes["max"], self._number_type, max
                )
            )
        if not lows:
            return None
        return min(lows), max(highs)

    def event_types(self) -> list[str]:
        """Get the list of unique event types.

        Returns:
            List of event type names.
        """
        unique = pc.unique(self._table.column("event_type"))
        return [v.as_py() for v in unique if v.as_py() is not None]

    def summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the store.

        Returns:
            Dict with count, temporal type counts, event type counts,
            coordinate range, unit, and number type.
        """
        return {
            "count": self.count,
            "unit": str(self.unit),
            "number_type": str(self.number_type),
            "temporal_types": self.count_by("temporal_type"),
            "event_types": self.count_by("event_type"),
            "coordinate_range": self.coordinate_range(),
        }

    # endregion

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> "Timeline":
        """Create a Timeline from this EventData.

        This is a convenience method that creates a timeline with the data's
        events directly. The timeline class and number_type are inferred from
        the data's unit (e.g., ticks -> DiscreteLogicalTimeline with int).

        Args:
            uid: Unique ID for the timeline. Auto-generated if None.
            filters: Filter kwargs to apply before timeline creation.
                Example: {"event_type": "Note"} to exclude rests.

        Returns:
            A Timeline containing the (filtered) events.

        Examples:
            >>> timeline = data.create_timeline(uid="notes")
            >>> filtered = data.create_timeline(filters={"event_type": "Note"})
        """
        if _timeline_factory is None:
            # The timeline layer registers the factory when it is imported,
            # which ``import timetoalign`` alone does not do.
            import timetoalign.timelines  # noqa: F401
        if _timeline_factory is None:
            raise RuntimeError("Timeline creation requires timetoalign.timelines")
        return _timeline_factory(self, uid, filters)

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the EventData to a Parquet file.

        Args:
            path: Path to write the Parquet file.
        """
        pq.write_table(self._table, path)

    def to_dataframe(
        self,
        format: str = "pandas",
        *,
        raw: bool = False,
        coordinates: bool = False,
    ) -> pd.DataFrame:
        """Convert to a DataFrame in the specified format.

        Higher-level method that dispatches to format-specific implementations.
        Currently supports pandas; polars support can be added later.

        Args:
            format: DataFrame format ("pandas"). Default "pandas".
            raw: If True, return raw conversion with struct dicts for coordinates.
            coordinates: If True, wrap values in Coordinate objects with unit info.

        Returns:
            A DataFrame in the requested format.

        Raises:
            ValueError: If format is not supported.

        Examples:
            >>> df = events.to_dataframe()  # pandas DataFrame
            >>> df.iloc[0]['start']  # Fraction(1, 4) or 0.25
            >>> df = events.to_dataframe("pandas", raw=True)
            >>> df.iloc[0]['start']  # {'value': 0.25, 'numerator': 1, 'denominator': 4}
            >>> df = events.to_dataframe(coordinates=True)
            >>> df.iloc[0]['start']  # Coordinate(value=Fraction(1, 4), unit=quarters)
        """
        if format == "pandas":
            df = self._table.to_pandas()

            if raw:
                return df

            # Detect coordinate fields from schema (struct with value/num/den fields)
            # This handles both core fields (start, end, duration) and extra
            # CoordinateField fields.

            for pa_field in self._table.schema:
                field_name = pa_field.name
                if field_name not in df.columns:
                    continue

                if is_coordinate_type(pa_field.type):
                    # Get unit from field metadata, fall back to EventData unit
                    unit = self._unit
                    unit_str = field_metadata(pa_field).get("unit")
                    if unit_str:
                        try:
                            unit = TimeUnit(unit_str)
                        except ValueError:
                            pass  # Use default unit

                    if coordinates:
                        field_unit = unit
                        df[field_name] = df[field_name].apply(
                            lambda s, u=field_unit: (
                                Coordinate(
                                    struct_to_coordinate(s, self._number_type), u
                                )
                                if s is not None
                                else None
                            )
                        )
                    else:
                        df[field_name] = df[field_name].apply(
                            lambda s: (
                                struct_to_coordinate(s, self._number_type)
                                if s is not None
                                else None
                            )
                        )

            return df
        else:
            raise ValueError(
                f"Unsupported DataFrame format: {format!r}. "
                f"Supported formats: 'pandas'"
            )

    def head(self, n: int = 5) -> pd.DataFrame:
        """Return the first ``n`` events as a pandas DataFrame.

        A pandas-style preview of the stored rows. Coordinate columns render
        as their native numbers — the same conversion :meth:`to_dataframe`
        applies — so ``events.head()`` shows the leading events directly,
        without reaching for the raw PyArrow table.

        Args:
            n: Number of leading events to include. Values larger than the
                event count return every event; ``n <= 0`` returns an empty
                frame.

        Returns:
            A pandas DataFrame of the first ``n`` events.

        Examples:
            >>> events.head()      # first five events
            >>> events.head(3)     # first three events
        """
        preview = self._derived(self._table.slice(0, max(n, 0)))
        return preview.to_dataframe()

    # endregion


def _as_numpy(values: Any) -> np.ndarray:
    """Convert an array-like to a numpy array.

    Raises:
        TypeError: If conversion is not possible.
    """
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, pd.Series):
        return values.to_numpy()
    if isinstance(values, (pa.Array, pa.ChunkedArray)):
        return values.to_numpy(zero_copy_only=False)
    if isinstance(values, list):
        return np.array(values)
    raise TypeError(f"Cannot convert {type(values)} to numpy array")


class ArrayValidator:
    """Vectorized array validation before table construction.

    Validates:
    - Array lengths match
    - Required fields present
    - Temporal type consistency (instant vs interval)
    - ID uniqueness

    All checks use vectorized operations - NO row iteration.

    Examples:
        >>> fields = {
        ...     "id": np.array(["e1", "e2"]),
        ...     "temporal_type": np.array(["instant", "interval"]),
        ...     "event_type": np.array(["Beat", "Note"]),
        ...     "start": pa.array([...]),  # StructArray
        ...     "end": pa.array([None, ...]),  # StructArray with nulls
        ... }
        >>> ArrayValidator.validate_field_dict(fields, schema)
    """

    @staticmethod
    def validate_field_dict(
        fields: dict[str, np.ndarray | pd.Series | list | pa.Array],
        schema: pa.Schema,
    ) -> None:
        """Validate field dictionary before table creation (vectorized).

        Args:
            fields: Dict of {field_name: array_values}.
            schema: Expected PyArrow schema.

        Raises:
            ValueError: If validation fails.
        """
        # Check 1: All arrays have same length (vectorized check)
        lengths = {name: len(arr) for name, arr in fields.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) > 1:
            raise ValueError(f"Field length mismatch: {lengths}")

        # Check 2: Required fields present
        required = {"id", "temporal_type", "event_type", "start"}
        missing = required - set(fields.keys())

        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Check 3: ID uniqueness (vectorized)
        id_arr = _as_numpy(fields["id"])
        unique_ids = np.unique(id_arr)

        if len(unique_ids) != len(id_arr):
            duplicates = len(id_arr) - len(unique_ids)
            raise ValueError(
                f"Duplicate IDs found: {len(id_arr)} IDs, " f"{duplicates} duplicates"
            )

        # Check 4: Temporal type consistency (vectorized)
        ArrayValidator._validate_temporal_consistency(fields)

    @staticmethod
    def _validate_temporal_consistency(
        fields: dict[str, np.ndarray | pd.Series | list | pa.Array],
    ) -> None:
        """Validate instant vs interval consistency (vectorized).

        Strategy:
        - instant events: end must be None/null
        - interval events: end must be present
        - Use pandas boolean indexing for vectorized checks

        Args:
            fields: Field dictionary.

        Raises:
            ValueError: If temporal consistency is violated.
        """
        temporal_type = pd.Series(_as_numpy(fields["temporal_type"]))

        # Get masks for instant/interval events (vectorized)
        is_instant = temporal_type == "instant"
        is_interval = temporal_type == "interval"

        # Validate all values are either instant or interval
        valid_count = is_instant.sum() + is_interval.sum()
        if valid_count != len(temporal_type):
            invalid = temporal_type[~(is_instant | is_interval)]
            raise ValueError(
                f"Invalid temporal_type values: {invalid.unique().tolist()}"
            )

        # Validate instant events have no end coordinate (vectorized check)
        if "end" in fields:
            # Handle both PyArrow StructArray and numpy array
            end_arr = fields["end"]
            if isinstance(end_arr, pa.StructArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            elif isinstance(end_arr, pa.ChunkedArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            else:
                end_series = pd.Series(_as_numpy(end_arr))
                end_is_null = end_series.isna().to_numpy()

            instant_with_end = is_instant.to_numpy() & ~end_is_null

            if instant_with_end.any():
                raise ValueError(
                    f"Instant events cannot have 'end' coordinate: "
                    f"{instant_with_end.sum()} violations"
                )

        # Validate interval events have end coordinate (vectorized check)
        if "end" not in fields:
            if is_interval.any():
                raise ValueError(
                    f"Interval events require 'end' coordinate: "
                    f"{is_interval.sum()} events missing 'end'"
                )
        else:
            # Check that interval events have non-null end
            end_arr = fields["end"]
            if isinstance(end_arr, pa.StructArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            elif isinstance(end_arr, pa.ChunkedArray):
                end_is_null = end_arr.is_null().to_numpy(zero_copy_only=False)
            else:
                end_series = pd.Series(_as_numpy(end_arr))
                end_is_null = end_series.isna().to_numpy()

            interval_no_end = is_interval.to_numpy() & end_is_null

            if interval_no_end.any():
                raise ValueError(
                    f"Interval events missing 'end' coordinate: "
                    f"{interval_no_end.sum()} events"
                )
