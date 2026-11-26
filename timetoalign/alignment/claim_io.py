"""Readers and writers of the MatchClaim table.

Columns, MatchClaims, dictionaries, Arrow tables, package tables and
Parquet files are read into a :class:`~timetoalign.alignment.claims.MatchClaimField`
and written back from it, with the refusals of malformed input and the check
that a table written over an earlier one succeeds it.
"""

from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from timetoshare.statements import used_prefixes

from timetoalign.alignment.anchors import _axes, _Axis, _coordinate_to_dict
from timetoalign.core import Coordinate
from timetoalign.core.fields import (
    _compared,
    _filled,
    _inferred_array,
    _struct_array,
    _struct_array_of,
    _struct_child,
    _struct_members,
    _struct_rows,
)
from timetoalign.core.ids import CHUNK, NOMATCH, IdMinter, given_minter, quote_id
from timetoalign.core.provenance import Provenance, provenance_array
from timetoalign.core.retrieval import coordinate_from_dict
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    axis_coordinate_cells,
    cell_coordinate,
    coordinate_cells,
    in_lowest_terms,
)

if TYPE_CHECKING:
    from timetoalign.alignment.claims import MatchClaim, MatchClaimField


#: The keys of a coordinate dict (:func:`coordinate_to_dict`), which is how a
#: claim coordinate cell is written in :meth:`MatchClaimField.to_dict`.
_COORDINATE_DICT_KEYS: tuple[str, ...] = ("unit", "number_type", "value")


def _number_column(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
) -> pa.Array | np.ndarray:
    """One number per row as an Arrow array, or as objects where Arrow has no type.

    Exact numbers (Fractions) have no Arrow type and are kept as objects.
    """
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    try:
        return _inferred_array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return np.asarray(list(values), dtype=object)


def _given(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
) -> pa.BooleanArray:
    """Per row whether a number is given (not null), whatever number it is.

    A NaN is given, so that its cell (a null, since no cell holds it) is
    refused rather than read as no number.
    """
    column = _number_column(values)
    if isinstance(column, pa.Array):
        return column.is_valid()
    return pa.array([value is not None for value in column], type=pa.bool_())


def _timeline_cells(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
    timeline_ids: pa.Array,
    axes: dict[str, _Axis],
) -> pa.StructArray:
    """The coordinate cells of raw numbers, each a position on its row's timeline.

    A number takes the unit of the timeline its row names and is written in
    that timeline's declared number type. The rows of one timeline are built
    together (:func:`~timetoalign.core.time.axis_coordinate_cells`,
    vectorized for a numeric column), so the cost grows with the number of
    timelines, not of rows; the row order is kept.

    Args:
        values: One number per row (null for none).
        timeline_ids: The timeline of each row.
        axes: The axis of every timeline the rows name.

    Raises:
        ValueError: If a row names a timeline whose axis is not given, or
            the values and the timeline ids differ in length.
    """
    column = _number_column(values)
    if len(column) != len(timeline_ids):
        raise ValueError("Coordinate values and timeline ids must have equal length.")
    distinct = pc.unique(timeline_ids).to_pylist()
    unknown = [value for value in distinct if value not in axes]
    if unknown:
        raise ValueError(
            f"The rows name timeline {quote_id(str(unknown[0]))}, which is not "
            f"among the timelines given."
        )
    # Every timeline named is now a key of *axes*, so a string.
    named = [value for value in distinct if isinstance(value, str)]
    parts: list[pa.StructArray] = []
    rows: list[pa.Array] = []
    for timeline_id in named:
        axis = axes[timeline_id]
        where = pc.indices_nonzero(_compared("equal", timeline_ids, timeline_id))
        taken = (
            column.take(where)
            if isinstance(column, pa.Array)
            else column[where.to_numpy()].tolist()
        )
        parts.append(
            axis_coordinate_cells(taken, unit=axis.unit, number_type=axis.number_type)
        )
        rows.append(where)
    if not parts:
        return pa.nulls(0, COORDINATE_CELL_TYPE)
    cells = pa.concat_arrays(parts)
    return cells.take(pc.sort_indices(pa.concat_arrays(rows)))


def _coordinate_paths(
    struct_type: pa.StructType, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Yield the path of every claim coordinate cell in *struct_type*."""
    for sub_field in struct_type:
        path = (*prefix, sub_field.name)
        if sub_field.type == COORDINATE_CELL_TYPE:
            yield path
        elif pa.types.is_struct(sub_field.type):
            yield from _coordinate_paths(sub_field.type, path)


def _holder(row: dict[str, Any] | None, path: tuple[str, ...]) -> dict[str, Any] | None:
    """The mapping holding the last key of *path* in *row*, or ``None``.

    ``None`` where a mapping on the way is null (an absent end anchor), or
    where *row* itself is null, which :class:`MatchClaimField` refuses at
    construction, so its rows never reach here null.
    """
    holder: Any = row
    if holder is None:
        return None
    for key in path[:-1]:
        holder = holder[key]
        if holder is None:
            return None
    return holder


def _with_children(
    array: pa.StructArray,
    children: Mapping[tuple[str, ...], pa.Array],
    prefix: tuple[str, ...] = (),
) -> pa.StructArray:
    """Return *array* with the children at the given paths replaced.

    Every struct on a path is rebuilt with its own validity, so a null end
    anchor stays null.
    """
    arrays: list[pa.Array] = []
    for sub_field in array.type:
        path = (*prefix, sub_field.name)
        child = _struct_child(array, sub_field.name)
        if path in children:
            child = children[path]
        elif any(key[: len(path)] == path for key in children):
            # A path leading further passes through a struct child.
            child = _with_children(_struct_array(child), children, path)
        arrays.append(child)
    return pa.StructArray.from_arrays(
        arrays,
        fields=list(array.type),
        mask=array.is_null() if array.null_count else None,
    )


#: The MatchClaim table's anchor columns, each a struct of two coordinate cells.
_ANCHOR_COLUMNS = ("start_anchor", "end_anchor")


_ANCHOR_SIDES = ("coordinate_a", "coordinate_b")


def _anchors_converted(table: pa.Table, convert: Any) -> pa.Table:
    """*table* with each anchor's two coordinate cells passed through *convert*.

    *convert* takes a column of cells and the name of the column it is
    (``start_anchor.coordinate_a``) and returns the converted column; a null
    anchor stays null. A table without an anchor column, or whose anchor
    column is no struct of the two cells, is returned as it stands, for the
    reader's check of the columns to refuse.
    """
    for name in _ANCHOR_COLUMNS:
        if name not in table.column_names:
            continue
        anchors = table.column(name).combine_chunks()
        if not isinstance(anchors, pa.StructArray) or sorted(
            field.name for field in anchors.type
        ) != sorted(_ANCHOR_SIDES):
            continue
        sides = dict(
            zip(
                (field.name for field in anchors.type),
                _struct_members(anchors),
                strict=True,
            )
        )
        converted = [convert(sides[side], f"{name}.{side}") for side in _ANCHOR_SIDES]
        written = pa.StructArray.from_arrays(
            converted,
            fields=[
                pa.field(side, cells.type)
                for side, cells in zip(_ANCHOR_SIDES, converted, strict=True)
            ],
            mask=anchors.is_null(),
        )
        table = table.set_column(
            table.schema.get_field_index(name), pa.field(name, written.type), written
        )
    return table


def _leaf_columns(table: pa.Table) -> pa.Table:
    """*table* with every struct column flattened into its leaves."""
    while any(pa.types.is_struct(field.type) for field in table.schema):
        table = table.flatten()
    return table


def _from_columns(
    cls: type[MatchClaimField],
    timeline_a_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
    timeline_b_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
    event_a_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
    event_b_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
    coordinate_a: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
    coordinate_b: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
    *,
    timelines: Iterable[Any] | Mapping[str, Any],
    end_coordinate_a: (
        Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray | None
    ) = None,
    end_coordinate_b: (
        Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray | None
    ) = None,
    certainty: float = 1.0,
    provenance: Provenance | None = None,
    seed: int | None = None,
    minter: IdMinter | None = None,
) -> "MatchClaimField":
    """Implement :meth:`MatchClaimField.from_columns`."""
    given = given_minter(seed, minter)
    axes = _axes(timelines)
    tl_a = cls._string_array(timeline_a_ids)
    tl_b = cls._string_array(timeline_b_ids)
    ev_a = cls._string_array(event_a_ids)
    ev_b = cls._string_array(event_b_ids)
    if (end_coordinate_a is None) != (end_coordinate_b is None):
        raise ValueError(
            "from_columns takes the end columns of both sides, or neither."
        )
    columns = {
        "timeline_a_id": tl_a,
        "timeline_b_id": tl_b,
        "event_a_id": ev_a,
        "event_b_id": ev_b,
        "coordinate_a": coordinate_a,
        "coordinate_b": coordinate_b,
    }
    if end_coordinate_a is not None:
        columns["end_coordinate_a"] = end_coordinate_a
        columns["end_coordinate_b"] = end_coordinate_b
    lengths = {name: len(column) for name, column in columns.items()}
    if len(set(lengths.values())) != 1:
        described = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            f"from_columns requires equal-length columns; got lengths {described}."
        )
    n = len(tl_a)
    for side, events in (("a", ev_a), ("b", ev_b)):
        if events.null_count:
            row = pc.index(pc.is_null(events), True).as_py()
            raise ValueError(
                f"Row {row} names nothing on side {side}: a side of a "
                f"MatchClaim names an event, or {CHUNK} for a position on a "
                f"timeline holding no event for it."
            )
    anchors = {
        "start_anchor": cls._anchor_array(
            _timeline_cells(coordinate_a, tl_a, axes),
            _timeline_cells(coordinate_b, tl_b, axes),
        )
    }
    if end_coordinate_a is not None and end_coordinate_b is not None:
        # Only a row without an end on either side is an instant claim:
        # an end given on one side alone, or one no cell can hold (a
        # non-finite number), leaves a null coordinate in the end
        # anchor, which the reader refuses.
        anchors["end_anchor"] = cls._anchor_array(
            _timeline_cells(end_coordinate_a, tl_a, axes),
            _timeline_cells(end_coordinate_b, tl_b, axes),
            mask=pc.and_(
                pc.invert(_given(end_coordinate_a)),
                pc.invert(_given(end_coordinate_b)),
            ),
        )
    true_column = pa.array(np.ones(n, dtype=bool), type=pa.bool_())
    struct_array = cls._assemble(
        n,
        {
            "timeline_a_id": tl_a,
            "timeline_b_id": tl_b,
            "event_a_id": ev_a,
            "event_b_id": ev_b,
            "is_synchronous": true_column,
            "certainty": pa.array(np.full(n, float(certainty)), pa.float64()),
            **anchors,
        },
    )
    return cls._read(
        struct_array,
        provenance,
        source="column set",
        where="as provenance=",
        minter=given,
    ).check_on_timelines(axes)


def _from_claims(
    cls: type[MatchClaimField],
    claims: Sequence[MatchClaim],
    *,
    provenance: Provenance | None = None,
) -> "MatchClaimField":
    """Implement :meth:`MatchClaimField.from_claims`."""
    claims = list(claims)
    if provenance is None:
        provenance = cls._common_provenance(claims)
    schema = cls.struct_schema()
    columns: dict[str, pa.Array] = {
        name: pa.array(
            [getattr(claim, name) for claim in claims],
            type=schema.field(name).type,
        )
        for name in ("timeline_a_id", "timeline_b_id", *cls._ROW_VALUES)
    }
    for anchor_name in ("start_anchor", "end_anchor"):
        anchors = [getattr(claim, anchor_name) for claim in claims]
        columns[anchor_name] = cls._anchor_array(
            coordinate_cells([a and a.coordinate_a for a in anchors]),
            coordinate_cells([a and a.coordinate_b for a in anchors]),
            mask=pa.array([a is None for a in anchors], type=pa.bool_()),
        )
    if provenance is None:
        columns["provenance"] = provenance_array([claim.provenance for claim in claims])
    # Each claim holds the scalar's rules; a coordinate no cell can hold
    # (a non-finite number) shows only once the cells are written.
    struct_array = cls._assemble(len(claims), columns)
    cls._refuse_null_coordinates(struct_array)
    return cls._admitted(struct_array, provenance=provenance)


def _from_dict(
    cls: type[MatchClaimField],
    data: dict[str, Any],
    *,
    seed: int | None = None,
    minter: IdMinter | None = None,
) -> "MatchClaimField":
    """Implement :meth:`MatchClaimField.from_dict`."""
    given = given_minter(seed, minter)
    keys = ["provenance", "rows"]
    if not isinstance(data, dict) or sorted(data) != keys:
        raise ValueError(
            f"A MatchClaimField dictionary holds the keys {keys}, not "
            f"{sorted(data) if isinstance(data, dict) else data!r}."
        )
    shared = data["provenance"]
    provenance = None if shared is None else Provenance.from_dict(shared)
    rows = copy.deepcopy(data["rows"])
    for index, row in enumerate(rows):
        cls._refuse_other_keys(row, cls.struct_schema(), index)
    cells: dict[tuple[str, ...], pa.StructArray] = {}
    for path in _coordinate_paths(cls.struct_schema()):
        coordinates: list[Coordinate | None] = []
        for index, row in enumerate(rows):
            holder = _holder(row, path)
            if holder is None or holder[path[-1]] is None:
                coordinates.append(None)
                continue
            try:
                coordinate = coordinate_from_dict(holder[path[-1]])
            except ValueError as error:
                raise ValueError(
                    f"In a MatchClaimField dictionary, row {index} "
                    f"{'.'.join(path)}: {error}"
                ) from error
            coordinates.append(coordinate)
            holder[path[-1]] = None
        cells[path] = coordinate_cells(coordinates)
    return cls._read(
        _with_children(_struct_array_of(rows, cls.struct_schema()), cells),
        provenance,
        source="dictionary",
        where="under its key 'provenance'",
        minter=given,
    )


def _refuse_other_keys(
    cls: type[MatchClaimField],
    value: Any,
    data_type: pa.StructType | pa.ListType[Any],
    index: int,
    path: tuple[str, ...] = (),
) -> None:
    """Implement :meth:`MatchClaimField._refuse_other_keys`."""
    if value is None and path:
        return
    where = f"row {index}" + (f" {'.'.join(path)}" if path else "")
    if isinstance(data_type, pa.ListType):
        if not isinstance(value, list):
            raise ValueError(
                f"In a MatchClaimField dictionary, {where} must be a list, "
                f"not {value!r}."
            )
        for position, item in enumerate(value):
            cls._refuse_other_keys(
                item, data_type.value_type, index, (*path, str(position))
            )
        return
    coordinate = data_type == COORDINATE_CELL_TYPE
    keys = list(_COORDINATE_DICT_KEYS if coordinate else data_type.names)
    if not isinstance(value, dict):
        raise ValueError(
            f"In a MatchClaimField dictionary, {where} must be a mapping "
            f"holding the keys {keys}, not {value!r}."
        )
    missing = [key for key in keys if key not in value]
    unknown = [key for key in value if key not in keys]
    if missing or unknown:
        raise ValueError(
            f"In a MatchClaimField dictionary, {where} must hold the keys "
            f"{keys}; it lacks {missing} and has the unknown keys {unknown}."
        )
    if coordinate:
        return
    for sub_field in data_type:
        sub_type = sub_field.type
        if isinstance(sub_type, pa.StructType) or (
            isinstance(sub_type, pa.ListType)
            and pa.types.is_struct(sub_type.value_type)
        ):
            cls._refuse_other_keys(
                value[sub_field.name],
                sub_type,
                index,
                (*path, sub_field.name),
            )


def _from_table(
    cls: type[MatchClaimField],
    table: pa.Table,
    *,
    seed: int | None = None,
    minter: IdMinter | None = None,
) -> "MatchClaimField":
    """Implement :meth:`MatchClaimField.from_table`."""
    given = given_minter(seed, minter)
    columns = cls.struct_schema().names
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        raise ValueError(
            f"A MatchClaim table holds one column per MatchClaim property, "
            f"{columns}; this table lacks {missing}."
        )
    for sub_field in cls.struct_schema():
        stored = table.schema.field(sub_field.name).type
        if stored != sub_field.type:
            raise ValueError(
                f"Column {sub_field.name!r} has type {stored}, which is not "
                f"the type of the MatchClaim property, {sub_field.type}."
            )
    further_names = [name for name in table.column_names if name not in columns]
    further = (
        pa.table({name: table.column(name).combine_chunks() for name in further_names})
        if further_names
        else None
    )
    # Memory holds one spelling of a value (2/4 is held as 1/2), so the
    # cells compare by value wherever they are compared.
    table = _anchors_converted(
        table.select(columns), lambda cells, _name: in_lowest_terms(cells)
    )
    struct_array = pa.StructArray.from_arrays(
        [table.column(name).combine_chunks() for name in columns],
        fields=list(cls.struct_schema()),
    )
    stored = (table.schema.metadata or {}).get(cls._PROVENANCE_KEY)
    provenance = None if stored is None else Provenance.from_dict(json.loads(stored))
    return cls._read(
        struct_array,
        provenance,
        source="table",
        where=f"in its schema metadata ({cls._PROVENANCE_KEY.decode()!r})",
        further=further,
        minter=given,
    )


def _refuse_malformed(data: pa.StructArray) -> None:
    """Implement :meth:`MatchClaimField._refuse_malformed`."""
    from timetoalign.alignment.claims import MatchClaimField

    if len(data) == 0:
        return
    synchronous = data.field("is_synchronous")
    start = data.field("start_anchor")
    end = data.field("end_anchor")
    ev_a = _struct_child(data, "event_a_id")
    ev_b = _struct_child(data, "event_b_id")

    def on_a_side(sentinel: str) -> pa.Array:
        """Whether a row names *sentinel* on either side."""
        return pc.or_(
            _compared("equal", ev_a, sentinel), _compared("equal", ev_b, sentinel)
        )

    def named_nomatch(events: pa.Array, names: pa.Array) -> pa.Array:
        """Whether a side is NOMATCH holding a name."""
        return pc.and_(_compared("equal", events, NOMATCH), pc.is_valid(names))

    tl_a = data.field("timeline_a_id")
    tl_b = data.field("timeline_b_id")
    certainty = data.field("certainty")
    checks: tuple[tuple[pa.BooleanArray, str], ...] = (
        (
            pc.or_(pc.is_null(tl_a), pc.is_null(tl_b)),
            "names no timeline_a_id or no timeline_b_id",
        ),
        (pc.equal(tl_a, tl_b), "connects a timeline with itself"),
        (
            pc.or_(pc.is_null(ev_a), pc.is_null(ev_b)),
            "names no event_a_id or no event_b_id (a side without an event "
            f"holds {NOMATCH} or {CHUNK})",
        ),
        (pc.is_null(synchronous), "states no is_synchronous"),
        (
            pc.and_(pc.invert(synchronous), on_a_side(CHUNK)),
            f"is not synchronous and names {CHUNK}, which stands only on a "
            "synchronous MatchClaim",
        ),
        (
            pc.and_(synchronous, on_a_side(NOMATCH)),
            f"is synchronous and names {NOMATCH}, which stands only on a "
            "MatchClaim that is not synchronous",
        ),
        (
            pc.and_(
                _compared("equal", ev_a, NOMATCH), _compared("equal", ev_b, NOMATCH)
            ),
            f"names {NOMATCH} on both sides; a NOMATCH MatchClaim names an "
            "event on its other side",
        ),
        (
            pc.or_(
                named_nomatch(ev_a, data.field("event_a_name")),
                named_nomatch(ev_b, data.field("event_b_name")),
            ),
            "has a name on a NOMATCH side; a NOMATCH side has no name",
        ),
        (
            pc.and_(synchronous, start.is_null()),
            "is synchronous and holds no start anchor",
        ),
        (
            pc.and_(pc.invert(synchronous), pc.or_(start.is_valid(), end.is_valid())),
            "is not synchronous and holds an anchor",
        ),
        (
            pc.or_(
                pc.less(certainty, pa.scalar(0.0)),
                pc.greater(certainty, pa.scalar(1.0)),
            ),
            "has a certainty outside [0, 1]",
        ),
    )
    for check, reason in checks:
        mask = _filled(check, pa.scalar(False))
        if pc.any(mask).as_py():
            row = pc.index(mask, True).as_py()
            raise ValueError(f"MatchClaim table row {row} {reason}.")
    MatchClaimField._refuse_null_coordinates(data)
    MatchClaimField._refuse_two_axes(data)


def _refuse_null_coordinates(data: pa.StructArray) -> None:
    """Implement :meth:`MatchClaimField._refuse_null_coordinates`."""
    for anchor_name in ("start_anchor", "end_anchor"):
        anchor = _struct_array(data.field(anchor_name))
        for side in ("a", "b"):
            missing = pc.and_(
                anchor.is_valid(),
                anchor.field(f"coordinate_{side}").is_null(),
            )
            if pc.any(missing).as_py():
                row = pc.index(missing, True).as_py()
                raise ValueError(
                    f"MatchClaim table row {row} {anchor_name}.coordinate_{side}: "
                    f"the coordinate is null."
                )


def _refuse_two_axes(data: pa.StructArray) -> None:
    """Implement :meth:`MatchClaimField._refuse_two_axes`."""
    timelines: list[pa.Array] = []
    cells: list[pa.Array] = []
    for anchor_name in ("start_anchor", "end_anchor"):
        # ``flatten`` carries a null anchor over to its coordinates.
        anchor = _struct_members(_struct_array(data.field(anchor_name)))
        names = data.type.field(anchor_name).type.names
        for side in ("a", "b"):
            timelines.append(data.field(f"timeline_{side}_id"))
            cells.append(anchor[names.index(f"coordinate_{side}")])
    cell = _struct_array(pa.concat_arrays(cells))
    axes = pa.table(
        {
            "timeline": pa.concat_arrays(timelines),
            "unit": cell.field("unit"),
            "number_type": cell.field("number_type"),
        }
    ).filter(cell.is_valid())
    distinct = axes.group_by(["timeline", "unit", "number_type"]).aggregate([])
    counts = distinct.group_by("timeline").aggregate([("unit", "count")])
    repeated = counts.filter(pc.greater(counts.column("unit_count"), pa.scalar(1)))
    if repeated.num_rows:
        timeline_id = repeated.column("timeline")[0].as_py()
        stated = distinct.filter(
            pc.equal(distinct.column("timeline"), timeline_id)
        ).to_pylist()
        described = " and ".join(
            f"{axis['unit']} ({axis['number_type']})" for axis in stated[:2]
        )
        raise ValueError(
            f"The MatchClaim table holds coordinates of timeline "
            f"{quote_id(timeline_id)} in {described}: the coordinates of "
            f"one timeline are on its one axis."
        )


def _anchor_array(
    cls: type[MatchClaimField],
    coordinate_a: pa.StructArray,
    coordinate_b: pa.StructArray,
    *,
    mask: pa.Array | None = None,
) -> pa.StructArray:
    """Implement :meth:`MatchClaimField._anchor_array`."""
    anchor_type = cls.struct_schema().field("start_anchor").type
    sources: dict[str, pa.Array] = {
        "coordinate_a": coordinate_a,
        "coordinate_b": coordinate_b,
    }
    return pa.StructArray.from_arrays(
        [sources[field.name] for field in anchor_type],
        fields=list(anchor_type),
        mask=mask,
    )


def _used_prefixes(self: MatchClaimField) -> list[str]:
    """Implement :meth:`MatchClaimField._used_prefixes`."""
    entries: list[dict[str, Any]] = []
    if self._provenance is not None:
        entries.append(self._provenance.to_dict())
    data = self._struct_array()
    if len(data):
        # ``flatten`` carries a null parent over to its children.
        provenance = _struct_array(
            _struct_members(data)[self.struct_schema().get_field_index("provenance")]
        )
        members = {
            member.name: column
            for member, column in zip(provenance.type, _struct_members(provenance))
        }
        beliefs = pc.list_flatten(members["beliefs"])
        references = [
            pc.list_flatten(members["stated_by"]),
            pc.list_flatten(members["basis"]),
            members["service"],
            beliefs.flatten()[beliefs.type.get_field_index("actor")],
        ]
        entries.append(
            {
                "stated_by": sorted(
                    {
                        text
                        for column in references
                        for text in pc.unique(column).to_pylist()
                        if text is not None
                    }
                )
            }
        )
        entries.extend(
            {"method": method}
            for method in pc.unique(members["method"]).to_pylist()
            if method is not None
        )
    return used_prefixes({"provenance": entries})


def _to_dict(self: MatchClaimField) -> dict[str, Any]:
    """Implement :meth:`MatchClaimField.to_dict`."""
    rows = _struct_rows(self._struct_array())
    paths = list(_coordinate_paths(self.struct_schema()))
    for row in rows:
        for path in paths:
            holder = _holder(row, path)
            if holder is None:
                continue
            cell: dict[str, Any] | None = holder[path[-1]]
            if cell is not None:
                holder[path[-1]] = _coordinate_to_dict(cell_coordinate(cell))
    return {
        "provenance": (
            self._provenance.to_dict() if self._provenance is not None else None
        ),
        "rows": rows,
    }


def _to_parquet(self: MatchClaimField, path: str | os.PathLike[str]) -> Path:
    """Implement :meth:`MatchClaimField.to_parquet`."""
    import pyarrow.parquet as pq

    missing = pc.is_null(self._ids)
    if pc.any(missing).as_py():
        row = pc.index(missing, True).as_py()
        raise ValueError(
            f"MatchClaim table row {row} holds no id, and every MatchClaim "
            f"of the file holds one."
        )
    table = self.package_table
    target = self._file_path(path)
    if target.exists():
        self._refuse_unless_succeeding(type(self).from_parquet(target))
    temporary = target.with_name(f".{target.name}.writing")
    try:
        pq.write_table(table, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _refuse_unless_succeeding(
    self: MatchClaimField, existing: "MatchClaimField"
) -> None:
    """Implement :meth:`MatchClaimField._refuse_unless_succeeding`."""
    old_ids = existing._ids
    if len(old_ids) == 0:
        return
    positions = pc.index_in(old_ids, value_set=self._ids)
    if positions.null_count:
        row = pc.index(pc.is_null(positions), True).as_py()
        raise ValueError(
            f"The new MatchClaim table lacks the MatchClaim "
            f"{quote_id(old_ids[row].as_py())} of the existing one: a "
            f"MatchClaim is never removed; record the belief 'false' in it "
            f"instead (AlignmentBundle.add_match_claim_belief)."
        )

    def compared(field: "MatchClaimField") -> pa.Table:
        """Return *field*'s MatchClaim rows as a table without their provenance."""
        rows = pa.Table.from_struct_array(field._struct_array())
        return rows.drop_columns(["provenance"])

    old = _leaf_columns(compared(existing))
    new = _leaf_columns(compared(self).take(positions))
    for name in old.column_names:
        before = old.column(name).combine_chunks()
        after = new.column(name).combine_chunks()
        same = pc.or_(
            _filled(_compared("equal", before, after), pa.scalar(False)),
            pc.and_(pc.is_null(before), pc.is_null(after)),
        )
        if not pc.all(same).as_py():
            row = pc.index(same, False).as_py()
            raise ValueError(
                f"The new MatchClaim table changes the MatchClaim "
                f"{quote_id(old_ids[row].as_py())} of the existing one in "
                f"{name!r}: only a MatchClaim's provenance may change; a "
                f"correction is a new MatchClaim under a new id."
            )
