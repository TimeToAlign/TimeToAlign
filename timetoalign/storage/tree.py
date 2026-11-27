"""The events of a timeline tree, held by its root.

A timeline tree (a root timeline and every descendant of it) keeps the
events of all its timelines in its root's :class:`TreeStore`. Each timeline's
rows lie in that timeline's own coordinates, unit and number type: a child's
rows are child-local, never shifted by its offset. A timeline reads its own
rows as an ordinary :class:`EventData`, its axis declared in the field
metadata; the tree's table is the zero-copy concatenation of those rows,
each row named by its timeline in the dictionary-encoded column
:data:`~timetoalign.storage.events.TIMELINE_ID`, the timelines in the order
they joined the tree (the root first), each timeline's rows in the order they
were added.

The store holds once, in memory, the map of every timeline to its unit and
number type: a row's number type comes from its timeline, never from a
column. Event IDs are unique across the tree, checked column-wise whenever
rows join: a collision is refused naming both holders, nothing is renamed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core.fields import _null_column
from timetoalign.core.ids import quote_id

from .events import (
    TIMELINE_ID,
    Axis,
    EventData,
    declaring_axis,
    keyed_by_timeline,
    row_type_codes,
)

__all__ = ["TreeStore"]


# region Columns


def _ids_in(table: pa.Table) -> pa.ChunkedArray:
    """The event IDs *table* holds, as strings, nulls left out."""
    if "id" not in table.column_names:
        return pa.chunked_array([], pa.string())
    column = table.column("id")
    if column.null_count == len(column):
        return pa.chunked_array([], pa.string())
    if column.type != pa.string():
        column = pc.cast(column, pa.string())
    return pc.drop_null(column)


def _compacted(table: pa.Table) -> pa.Table:
    """*table* with its trailing small chunks joined: O(log n) chunks in all.

    Rows added one call at a time each add a chunk; each chunk is joined
    with the one before it as long as it is at least half as long, so a
    timeline's rows stay in few chunks and each row is copied O(log n)
    times over all additions.
    """
    batches = table.to_batches()
    if len(batches) < 2 or batches[-1].num_rows * 2 < batches[-2].num_rows:
        return table
    while len(batches) > 1 and batches[-1].num_rows * 2 >= batches[-2].num_rows:
        joined = pa.Table.from_batches(batches[-2:], schema=table.schema)
        batches[-2:] = joined.combine_chunks().to_batches()
    return pa.Table.from_batches(batches, schema=table.schema)


def _first_listed(candidates: pa.ChunkedArray, ids: pa.Array) -> str:
    """The first of *ids*, in their order, that *candidates* lists."""
    at = pc.index(pc.is_in(ids, value_set=candidates), True).as_py()
    return str(ids[at].as_py())


def _unified(held: pa.DataType, joining: pa.DataType) -> pa.DataType | None:
    """The type in which columns of *held* and of *joining* read every cell.

    Equal types are that type; a dictionary column reads as its values; a
    struct takes the members of both, each member unified likewise (a member
    one side lacks is null there, which changes no cell). Any other pair has
    none: None.
    """
    if held == joining:
        return held
    if pa.types.is_null(held):
        return joining
    if pa.types.is_null(joining):
        return held
    if pa.types.is_dictionary(held):
        return _unified(held.value_type, joining)
    if pa.types.is_dictionary(joining):
        return _unified(held, joining.value_type)
    if pa.types.is_struct(held) and pa.types.is_struct(joining):
        names = [member.name for member in held]
        members: list[pa.Field] = []
        for member in held:
            index = joining.get_field_index(member.name)
            if index < 0:
                members.append(member.with_nullable(True))
                continue
            unified = _unified(member.type, joining.field(index).type)
            if unified is None:
                return None
            members.append(member.with_type(unified))
        members.extend(
            member.with_nullable(True) for member in joining if member.name not in names
        )
        return pa.struct(members)
    return None


def _array_as(array: pa.Array, type_: pa.DataType) -> pa.Array:
    """*array* read in *type_*, a type :func:`_unified` gave for it."""
    if array.type == type_:
        return array
    if array.null_count == len(array):
        return _null_column(len(array), type_)
    if isinstance(array, pa.DictionaryArray):
        return _array_as(array.dictionary_decode(), type_)
    if pa.types.is_struct(type_):
        children = [
            (
                _array_as(pc.struct_field(array, [member.name]), member.type)
                if array.type.get_field_index(member.name) >= 0
                else _null_column(len(array), member.type)
            )
            for member in type_
        ]
        return pa.StructArray.from_arrays(
            children,
            fields=list(type_),
            mask=array.is_null() if array.null_count else None,
        )
    return pc.cast(array, type_)


class _Columns:
    """The columns of tables joining one table: names, fields, unified types.

    A column of nulls joins any other (it takes the other's type, which
    changes no cell); columns holding values join where :func:`_unified`
    gives them a type.
    """

    def __init__(self) -> None:
        self.fields: dict[str, pa.Field] = {}
        self.valued: set[str] = set()
        self.metadata: dict[bytes | str, bytes | str] | None = None

    def conflict(self, table: pa.Table) -> str | None:
        """The first column of *table* whose values no unified type reads."""
        for field in table.schema:
            held = self.fields.get(field.name)
            if held is None or held.name not in self.valued:
                continue
            column = table.column(field.name)
            if column.null_count == len(column):
                continue
            if _unified(held.type, field.type) is None:
                return field.name
        return None

    def take(self, table: pa.Table) -> None:
        """Hold the columns of *table*, which joins without a conflict."""
        held = table.schema.metadata
        if self.metadata is None and held is not None:
            self.metadata = {key: value for key, value in held.items()}
        for field in table.schema:
            column = table.column(field.name)
            valued = column.null_count < len(column)
            held = self.fields.get(field.name)
            if held is None:
                self.fields[field.name] = field
            elif valued:
                if held.name in self.valued:
                    unified = _unified(held.type, field.type)
                    assert unified is not None
                else:
                    unified = field.type
                self.fields[field.name] = held.with_type(unified)
            if valued:
                self.valued.add(field.name)

    def joined(self, tables: Sequence[pa.Table]) -> pa.Table:
        """*tables*, all taken, as one table: a column one lacks null there."""
        schema = pa.schema(list(self.fields.values()), metadata=self.metadata)
        conformed = []
        for table in tables:
            columns = []
            for field in schema:
                if field.name not in table.column_names:
                    columns.append(_null_column(table.num_rows, field.type))
                    continue
                column = table.column(field.name)
                if column.type != field.type:
                    column = pa.chunked_array(
                        [_array_as(chunk, field.type) for chunk in column.chunks],
                        type=field.type,
                    )
                columns.append(column)
            conformed.append(pa.Table.from_arrays(columns, schema=schema))
        return pa.concat_tables(conformed)


def _joined(tables: Sequence[pa.Table], timeline_id: str) -> pa.Table:
    """The rows of one timeline, *tables*, as one table.

    Raises:
        ValueError: If a column holds values of two types no type reads both
            in: a timeline's column has one type.
    """
    columns = _Columns()
    for table in tables:
        conflict = columns.conflict(table)
        if conflict is not None:
            raise ValueError(
                f"the column {conflict!r} of timeline {quote_id(timeline_id)} would "
                f"hold values of type {columns.fields[conflict].type} and of type "
                f"{table.schema.field(conflict).type}"
            )
        columns.take(table)
    return columns.joined(tables)


def _own(table: pa.Table) -> pa.Table:
    """*table* without the column naming timelines: one timeline's columns."""
    if TIMELINE_ID in table.column_names:
        return table.drop_columns([TIMELINE_ID])
    return table


# endregion


class TreeStore:
    """The events of a timeline tree, held by its root.

    Every timeline of the tree refers to its root's store and reads its own
    rows through :meth:`events_of`; rows reach the store only through its
    methods: :meth:`add` and :meth:`replace` (one timeline's rows),
    :meth:`attach` (a child's whole subtree joining the tree), :meth:`rename`
    (a timeline's ID).

    Each timeline's rows are held as they were added, so a timeline's read
    and every change of its rows touch its rows only; the tree's table
    (:attr:`table`) is their concatenation, which copies no row.

    Args:
        timeline_id: The ID of the root timeline.
        data: The root's rows, on its axis; their class is the class of the
            root's events.

    Raises:
        ValueError: If *data* holds one event ID twice.
    """

    def __init__(self, timeline_id: str, data: EventData) -> None:
        # Per timeline, in the order the timelines joined (the root first):
        # its rows, in the class and state of its events, its axis declared.
        self._parts: dict[str, EventData] = {}
        self._axes: dict[str, Axis] = {}
        # Per timeline the event IDs its rows hold, and the chunks of every
        # ID the tree holds (None: rebuilt from them when next read).
        self._ids: dict[str, pa.ChunkedArray] = {}
        self._held: list[pa.Array] | None = []
        # The tree's tables, built on demand from the parts.
        self._tables: list[pa.Table] | None = None
        # Per timeline: the events that copy another event of the tree (a
        # child cut from a region), by ID, to the ID of the event copied.
        self._copies: dict[str, dict[str, str]] = {}
        data = _spanning_cast(data, (data.unit, data.number_type))
        ids = _ids_in(data.table)
        self.refuse_held_ids(timeline_id, ids)
        self._hold(timeline_id, data, (data.unit, data.number_type), ids)
        self._held = list(ids.chunks)

    # region Reading

    @property
    def timeline_ids(self) -> tuple[str, ...]:
        """The IDs of the timelines the store holds rows of, root first."""
        return tuple(self._parts)

    @property
    def axes(self) -> Mapping[str, Axis]:
        """The unit and number type of every timeline, by timeline ID."""
        return dict(self._axes)

    @property
    def tables(self) -> tuple[pa.Table, ...]:
        """The tree's rows in tables, each row keyed by its timeline.

        The timelines of a tree share their columns: a column one lacks is
        null there, and a struct column takes the members of every timeline
        that holds it. A timeline that holds a column in a type no other
        type of it reads (an ``int`` column beside a ``string`` one) is put
        in the first table it joins without a conflict, else in a table of
        its own. Nothing is converted: a cell is never rewritten.
        """
        if self._tables is None:
            groups: list[tuple[_Columns, list[pa.Table]]] = []
            for timeline_id, part in self._parts.items():
                table = keyed_by_timeline(part.table, timeline_id)
                for columns, members in groups:
                    if columns.conflict(table) is None:
                        columns.take(table)
                        members.append(table)
                        break
                else:
                    columns = _Columns()
                    columns.take(table)
                    groups.append((columns, [table]))
            self._tables = [columns.joined(members) for columns, members in groups]
        return tuple(self._tables)

    @property
    def table(self) -> pa.Table:
        """The tree's table: the root's rows, then each timeline's, in order.

        Where a timeline's columns conflict with the root's
        (:attr:`tables`), its rows are in a table of their own and not in
        this one.
        """
        return self.tables[0]

    @property
    def data(self) -> EventData:
        """The tree's table (:attr:`table`) as an EventData spanning its timelines.

        Each row is read in its timeline's unit and number type
        (:meth:`EventData.to_number_type` casts it,
        ``filter(include_number_types=)`` selects by it); its ``unit`` and
        ``number_type`` are the root's.
        """
        table = self.table
        unit, number_type = self._axes[self.timeline_ids[0]]
        spanning = EventData._of_reduced(table, unit, number_type)
        named = set(_timelines_named(table))
        spanning._axes = {
            timeline_id: axis
            for timeline_id, axis in self._axes.items()
            if timeline_id in named
        }
        return spanning

    def row_types(self) -> pa.Int8Array:
        """The number-type code of every row of :attr:`table`, from its timeline.

        The row-type mask (:func:`~timetoalign.storage.events.row_type_codes`).
        """
        return row_type_codes(self.table, self._axes)

    def events_of(self, timeline_id: str) -> EventData:
        """The rows of *timeline_id*, as the tree's table holds them.

        Its columns are the timeline's own, their field metadata declaring
        the timeline's unit and number type, in an EventData of the
        timeline's class, so the semantic fields read it as any other. The
        same object is returned until the timeline's rows change.

        Raises:
            KeyError: If the store holds no timeline *timeline_id*.
        """
        held = self._parts.get(timeline_id)
        if held is None:
            raise KeyError(
                f"No timeline {quote_id(timeline_id)} in this tree. "
                f"Available timelines: {list(self._parts)}"
            )
        return held

    def holder_of(self, event_id: str) -> str | None:
        """The ID of the timeline holding the event *event_id*, or None."""
        return self._holder(event_id, None)

    def origin_of(self, event_id: str) -> str | None:
        """The ID of the event *event_id* copies, if it is a copy, else None.

        A child cut from a region of a timeline holds copies of its events,
        each with an ID of its own (:meth:`record_copies`).
        """
        for copies in self._copies.values():
            origin = copies.get(event_id)
            if origin is not None:
                return origin
        return None

    # endregion

    # region Checking

    def refuse_held_ids(
        self,
        timeline_id: str,
        ids: pa.Table | pa.Array | pa.ChunkedArray,
        *,
        own: bool = False,
    ) -> None:
        """Refuse event IDs joining *timeline_id* that the tree holds, or repeats.

        Column-wise: the IDs joining are counted, and the tree's held IDs
        looked up in them, never read as Python objects.

        Args:
            timeline_id: The timeline the IDs join.
            ids: The IDs (nulls are no IDs), or a table of rows whose ``id``
                column holds them.
            own: The rows replace the timeline's own, whose IDs are not held
                against them.

        Raises:
            ValueError: Naming the ID, the timeline holding it and the one
                it would join, or the ID given to two rows joining.
        """
        if isinstance(ids, pa.Table):
            column = _ids_in(ids)
        else:
            column = pa.chunked_array([ids]) if isinstance(ids, pa.Array) else ids
            if column.type != pa.string():
                column = pc.cast(column, pa.string())
        joining = pc.drop_null(column).combine_chunks()
        if not len(joining):
            return
        counts = pc.value_counts(joining)
        repeated = counts.filter(pc.greater(counts.field("counts"), pa.scalar(1)))
        if len(repeated):
            raise ValueError(
                f"Event id {quote_id(str(repeated.field('values')[0].as_py()))} is "
                f"given to {repeated.field('counts')[0].as_py()} events joining "
                f"timeline {quote_id(timeline_id)}; event ids are unique across a "
                "timeline tree"
            )
        skip = timeline_id if own else None
        held = self._held_ids(skip)
        if held is None:
            return
        hits = held.filter(pc.is_in(held, value_set=joining))
        if len(hits):
            event_id = _first_listed(hits, joining)
            holder = self._holder(event_id, skip)
            assert holder is not None
            raise ValueError(_collision(event_id, holder, timeline_id))

    def refuse_shared_events(self, other: TreeStore) -> None:
        """Refuse *other*'s subtree joining this tree where they share an ID.

        Raises:
            ValueError: If *other* holds a timeline this store holds, or an
                event ID this tree holds, naming the ID and both holders.
        """
        shared = [tid for tid in other._parts if tid in self._parts]
        if shared:
            raise ValueError(
                f"timeline {quote_id(shared[0])} is held in this tree already"
            )
        held = self._held_ids(None)
        joining = other._held_ids(None)
        if held is None or joining is None:
            return
        hits = held.filter(pc.is_in(held, value_set=joining))
        if len(hits):
            event_id = _first_listed(hits, joining.combine_chunks())
            holder, joiner = self.holder_of(event_id), other.holder_of(event_id)
            assert holder is not None and joiner is not None
            raise ValueError(_collision(event_id, holder, joiner))

    # endregion

    # region Changing

    def add(self, timeline_id: str, data: EventData, *, checked: bool = False) -> None:
        """Add *data*'s rows to the rows of *timeline_id*, after them.

        Rows of another number type are written in the timeline's, as every
        value put onto one timeline is (:meth:`EventData.to_number_type`).
        The timeline's columns grow by those *data* brings.

        Args:
            timeline_id: The timeline the rows join.
            data: The rows.
            checked: Their IDs were refused already where held
                (:meth:`refuse_held_ids`), before the caller changed
                anything.

        Raises:
            ValueError: If *data* has another unit, or an event ID of it is
                held in the tree or twice in *data*, or a column of it holds
                values of a type the timeline's column does not read; nothing
                changes then.
        """
        if not len(data):
            return
        data = data._onto(*self._axes[timeline_id])
        if not checked:
            self.refuse_held_ids(timeline_id, data.table)
        part = self._parts[timeline_id]
        rows = _compacted(_joined([part.table, _own(data.table)], timeline_id))
        added = _ids_in(data.table)
        self._set_part(
            timeline_id,
            part._derived(rows),
            pa.chunked_array(
                [*self._ids[timeline_id].chunks, *added.chunks], pa.string()
            ),
        )
        if self._held is not None:
            self._held.extend(added.chunks)

    def replace(self, timeline_id: str, data: EventData) -> None:
        """Make *data* the rows of *timeline_id*, its class the timeline's.

        Raises:
            ValueError: If *data* lies on another axis than the timeline's
                (a table spanning timelines is written in the timeline's
                number type), or an event ID of *data* is held by another
                timeline of the tree or twice in *data*; nothing changes then.
        """
        axis = self._axes[timeline_id]
        data = _spanning_cast(data, axis)
        if (data.unit, data.number_type) != axis:
            unit, number_type = axis
            raise ValueError(
                f"the rows of timeline {quote_id(timeline_id)} lie on its axis, "
                f"{unit} in {number_type.name}; these lie on {data.unit} in "
                f"{data.number_type.name}"
            )
        given = _ids_in(data.table)
        # Rows rewritten under the IDs they held (a Segment row given a new
        # end) need no check, and the tree's IDs stay as they are.
        unchanged = self._ids[timeline_id].equals(given)
        if not unchanged:
            self.refuse_held_ids(timeline_id, given, own=True)
            self._held = None
        self._hold(timeline_id, data, axis, given)
        copies = self._copies.get(timeline_id)
        if copies and not unchanged:
            kept = set(
                pc.filter(
                    given, pc.is_in(given, value_set=pa.array(list(copies)))
                ).to_pylist()
            )
            self._copies[timeline_id] = {
                copy: origin for copy, origin in copies.items() if copy in kept
            }

    def record_copies(
        self, timeline_id: str, copies: Sequence[str], origins: Sequence[str]
    ) -> None:
        """Record that the events *copies* of *timeline_id* copy *origins*.

        Each copy is a new event with an ID of its own; :meth:`origin_of`
        answers the event it copies, and a cut taking the events of a
        region takes an event and its copies once.
        """
        self._copies.setdefault(timeline_id, {}).update(
            zip(copies, origins, strict=True)
        )

    def attach(self, other: TreeStore) -> None:
        """Take every row of *other*, a subtree joining this tree, and its map.

        The rows keep their timelines, coordinates and number types and are
        not copied; *other* is left empty and must not be used again. The
        caller refuses a shared ID first (:meth:`refuse_shared_events`),
        before it changes anything.
        """
        self._parts.update(other._parts)
        self._axes.update(other._axes)
        self._ids.update(other._ids)
        if self._held is not None and other._held is not None:
            self._held.extend(other._held)
        else:
            self._held = None
        for timeline_id, copies in other._copies.items():
            self._copies.setdefault(timeline_id, {}).update(copies)
        self._tables = None
        other._release()

    def rename(self, old_id: str, new_id: str) -> None:
        """Give the timeline *old_id* the ID *new_id*; no row is rewritten.

        Raises:
            KeyError: If the store holds no timeline *old_id*.
            ValueError: If it holds a timeline *new_id*.
        """
        if old_id == new_id:
            return
        if old_id not in self._parts:
            raise KeyError(
                f"No timeline {quote_id(old_id)} in this tree. "
                f"Available timelines: {list(self._parts)}"
            )
        if new_id in self._parts:
            raise ValueError(
                f"timeline {quote_id(new_id)} is held in this tree already"
            )

        def renamed(mapping: dict[str, Any]) -> dict[str, Any]:
            """*mapping* with the key *old_id* replaced, in its place."""
            return {
                (new_id if key == old_id else key): value
                for key, value in mapping.items()
            }

        self._parts = renamed(self._parts)
        self._axes = renamed(self._axes)
        self._ids = renamed(self._ids)
        self._copies = renamed(self._copies)
        self._set_part(
            new_id,
            self._parts[new_id]._derived(self._parts[new_id].table),
            self._ids[new_id],
        )

    # endregion

    # region Internals

    def _hold(
        self, timeline_id: str, data: EventData, axis: Axis, ids: pa.ChunkedArray
    ) -> None:
        """Hold the rows of *data*, a table on *axis* holding *ids*, as *timeline_id*'s."""
        unit, number_type = axis
        part = data._derived(declaring_axis(_own(data.table), unit, number_type))
        self._axes[timeline_id] = axis
        self._set_part(timeline_id, part, ids)

    def _set_part(
        self, timeline_id: str, part: EventData, ids: pa.ChunkedArray
    ) -> None:
        """Make *part*, holding *ids*, the rows of *timeline_id*.

        The tree's tables are rebuilt when next read.
        """
        part._axes = None
        part._timeline_id = timeline_id
        self._parts[timeline_id] = part
        self._ids[timeline_id] = ids
        self._tables = None

    def _held_ids(self, skip: str | None) -> pa.ChunkedArray | None:
        """Every event ID the tree holds, but those of the timeline *skip*."""
        if skip is not None:
            chunks = [
                chunk
                for timeline_id, ids in self._ids.items()
                if timeline_id != skip
                for chunk in ids.chunks
            ]
        else:
            if self._held is None:
                self._held = [
                    chunk for ids in self._ids.values() for chunk in ids.chunks
                ]
            if len(self._held) > 32:
                self._held = [pa.concat_arrays(self._held)]
            chunks = self._held
        chunks = [chunk for chunk in chunks if len(chunk)]
        return pa.chunked_array(chunks, pa.string()) if chunks else None

    def _holder(self, event_id: str, skip: str | None) -> str | None:
        """The timeline holding *event_id*, the timeline *skip* left out."""
        wanted = pa.array([event_id], pa.string())
        for timeline_id, ids in self._ids.items():
            if (
                timeline_id != skip
                and len(ids)
                and pc.any(pc.is_in(ids, value_set=wanted)).as_py()
            ):
                return timeline_id
        return None

    def _release(self) -> None:
        """Drop every row and timeline: the rows moved to another store."""
        self._parts = {}
        self._axes = {}
        self._ids = {}
        self._held = []
        self._tables = None
        self._copies = {}

    # endregion


def _timelines_named(table: pa.Table) -> list[str]:
    """The timelines the rows of *table* name, in the order of its chunks."""
    names: list[str] = []
    for chunk in table.column(TIMELINE_ID).chunks:
        keys = chunk.dictionary if isinstance(chunk, pa.DictionaryArray) else chunk
        for name in pc.unique(keys).to_pylist():
            if name is not None and name not in names:
                names.append(str(name))
    return names


def _spanning_cast(data: EventData, axis: Axis) -> EventData:
    """*data*, if it spans timelines, written on *axis*: one timeline's rows."""
    if data._axes is None:
        return data
    return data._onto(*axis)


def _collision(event_id: str, holder: str, joining: str) -> str:
    """The refusal of the event ID *event_id*, held by *holder*, joining *joining*."""
    held = (
        f"Event id {quote_id(event_id)} is already held by an event of timeline "
        f"{quote_id(holder)}"
    )
    if holder == joining:
        return held
    return (
        f"{held}, so it cannot join timeline {quote_id(joining)}: event ids are "
        "unique across a timeline tree"
    )
