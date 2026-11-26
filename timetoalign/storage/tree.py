"""The events of a timeline tree, held by its root in one table.

A timeline tree (a root timeline and every descendant of it) keeps the
events of all its timelines in one ``pa.Table``, held by the root's
:class:`TreeStore`. Each row names its timeline in the dictionary-encoded
column :data:`~timetoalign.storage.events.TIMELINE_ID` and lies in that
timeline's own coordinates, unit and number type: a child's rows are
child-local, never shifted by its offset. The table is sorted by
``(timeline_id, start)``, again only when it is read after rows were added.
A timeline reads its own rows as a zero-copy slice of the table, its axis
stamped into the field metadata, so it is an ordinary :class:`EventData`.

The store holds once, in memory, the map of every timeline to its unit and
number type: a row's number type comes from its timeline, never from a
column. Event IDs are unique across the tree, checked against an index of
the IDs it holds whenever rows join: a collision is refused naming both
holders, nothing is renamed.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core.ids import quote_id

from .events import TIMELINE_ID, Axis, EventData, declaring_axis, row_type_codes

__all__ = ["TreeStore"]


def _ids_of(table: pa.Table) -> list[str]:
    """The event IDs *table* holds, nulls left out, in row order."""
    if "id" not in table.column_names:
        return []
    return [str(event_id) for event_id in pc.drop_null(table.column("id")).to_pylist()]


def _decoded(table: pa.Table) -> pa.Table:
    """*table* with every dictionary-encoded column decoded to its values.

    The tables of a tree's timelines share columns, which a column encoded
    in one table and plain in another could not.
    """
    for index, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            column = pa.chunked_array(
                [
                    chunk.dictionary_decode()
                    for chunk in table.column(index).chunks
                    if isinstance(chunk, pa.DictionaryArray)
                ],
                type=field.type.value_type,
            )
            table = table.set_column(
                index,
                pa.field(
                    field.name,
                    field.type.value_type,
                    field.nullable,
                    metadata=field.metadata,
                ),
                column,
            )
    return table


def _own_schema(schema: pa.Schema) -> pa.Schema:
    """*schema* without the column naming timelines: a timeline's own columns.

    A dictionary-encoded column is typed by its values (:func:`_decoded`).
    """
    index = schema.get_field_index(TIMELINE_ID)
    if index >= 0:
        schema = schema.remove(index)
    for position, field in enumerate(schema):
        if pa.types.is_dictionary(field.type):
            schema = schema.set(position, field.with_type(field.type.value_type))
    return schema


def _refuse_repeated(ids: Iterable[str], timeline_id: str) -> None:
    """Refuse an event ID held twice by the rows of one timeline joining.

    Raises:
        ValueError: Naming the ID and the timeline.
    """
    seen: set[str] = set()
    for event_id in ids:
        if event_id in seen:
            raise ValueError(
                f"Event id {quote_id(event_id)} is held by two events joining "
                f"timeline {quote_id(timeline_id)}; event ids are unique across "
                "a timeline tree"
            )
        seen.add(event_id)


def _nulls_typed_as(table: pa.Table, name: str, type_: pa.DataType) -> pa.Table:
    """*table* with its column *name*, every cell null, typed *type_*."""
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(
        index, field.with_type(type_), pc.cast(pa.nulls(table.num_rows), type_)
    )


def _concatenated(tables: list[pa.Table]) -> pa.Table:
    """*tables* joined into one, a column one lacks null there.

    The timelines of one tree may type a column differently where one holds
    no value in it (a row reader types a column of nulls by default): such a
    column takes the other's type, which changes no cell. Columns that hold
    values of two types cannot share the tree's table.

    Raises:
        ValueError: If a column holds values of two types.
    """
    joined = tables[0]
    for table in tables[1:]:
        for name in set(joined.column_names) & set(table.column_names):
            held = joined.schema.field(name).type
            joining = table.schema.field(name).type
            if held == joining:
                continue
            if table.column(name).null_count == table.num_rows:
                table = _nulls_typed_as(table, name, held)
            elif joined.column(name).null_count == joined.num_rows:
                joined = _nulls_typed_as(joined, name, joining)
            else:
                raise ValueError(
                    f"the column {name!r} holds values of type {held} and of type "
                    f"{joining} in one timeline tree; the timelines of a tree "
                    "share their columns"
                )
        joined = pa.concat_tables([joined, table], promote_options="default")
    return joined


class TreeStore:
    """The one table of a timeline tree's events, held by its root.

    Every timeline of the tree refers to its root's store and reads its own
    rows through :meth:`events_of`; rows reach the store only through its
    methods: :meth:`add` and :meth:`replace` (one timeline's rows),
    :meth:`attach` (a child's whole subtree joining the tree), :meth:`split`
    (a subtree leaving it, with a store of its own), :meth:`rename` (a
    timeline's ID).

    Args:
        timeline_id: The ID of the root timeline.
        data: The root's rows, on its axis; their class is the class of the
            root's events.

    Raises:
        ValueError: If *data* holds one event ID twice.
    """

    def __init__(self, timeline_id: str, data: EventData) -> None:
        # The dictionary of the TIMELINE_ID column: entries only append (a
        # rename replaces one), so a chunk's indices stay valid against
        # every later dictionary.
        self._timeline_ids: list[str] = []
        # Per timeline: the empty table of its class and state on its axis,
        # and the schema of its own columns (their field metadata included).
        self._prototypes: dict[str, EventData] = {}
        self._schemas: dict[str, pa.Schema] = {}
        self._axes: dict[str, Axis] = {}
        self._table: pa.Table | None = None
        self._sorted = True
        self._ranges: dict[str, tuple[int, int]] = {}
        self._slices: dict[str, EventData] = {}
        self._index: dict[str, str] | None = None
        if data._axes is not None:
            data = data.to_number_type(data.number_type)
        _refuse_repeated(_ids_of(data.table), timeline_id)
        self._register(timeline_id, data)
        self._table = self._keyed(data.table, timeline_id)
        self._sorted = False

    # region Reading

    @property
    def timeline_ids(self) -> tuple[str, ...]:
        """The IDs of the timelines the store holds rows of, root first."""
        return tuple(self._prototypes)

    @property
    def axes(self) -> Mapping[str, Axis]:
        """The unit and number type of every timeline, by timeline ID."""
        return dict(self._axes)

    @property
    def table(self) -> pa.Table:
        """The tree's table, sorted by ``(timeline_id, start)``."""
        self._sort()
        assert self._table is not None
        return self._table

    @property
    def data(self) -> EventData:
        """The tree's table as an EventData spanning its timelines.

        Each row is read in its timeline's unit and number type
        (:meth:`EventData.to_number_type` casts it,
        ``filter(include_number_types=)`` selects by it); its ``unit`` and
        ``number_type`` are the root's.
        """
        unit, number_type = self._axes[self.timeline_ids[0]]
        spanning = EventData._of_reduced(self.table, unit, number_type)
        spanning._axes = dict(self._axes)
        return spanning

    def row_types(self) -> pa.Int8Array:
        """The number-type code of every row of :attr:`table`, from its timeline.

        The row-type mask: one ``take`` of a code per timeline over the
        dictionary indices of the ``timeline_id`` column
        (:func:`~timetoalign.storage.events.row_type_codes`).
        """
        return row_type_codes(self.table, self._axes)

    def events_of(self, timeline_id: str) -> EventData:
        """The rows of *timeline_id*: a zero-copy slice of the tree's table.

        Its columns are the timeline's own, their field metadata declaring
        the timeline's unit and number type, in an EventData of the
        timeline's class, so the semantic fields read it as any other.
        The same object is returned until the timeline's rows change.

        Raises:
            KeyError: If the store holds no timeline *timeline_id*.
        """
        held = self._slices.get(timeline_id)
        if held is not None:
            return held
        if timeline_id not in self._prototypes:
            raise KeyError(
                f"No timeline {quote_id(timeline_id)} in this tree. "
                f"Available timelines: {list(self._prototypes)}"
            )
        table = self.table
        offset, length = self._ranges.get(timeline_id, (0, 0))
        rows = table.slice(offset, length)
        schema = self._schemas[timeline_id]
        own = pa.Table.from_arrays(
            [rows.column(name) for name in schema.names],
            schema=pa.schema(
                [
                    pa.field(
                        field.name,
                        rows.schema.field(field.name).type,
                        field.nullable,
                        metadata=field.metadata,
                    )
                    for field in schema
                ]
            ),
        ).replace_schema_metadata(schema.metadata)
        unit, number_type = self._axes[timeline_id]
        events = self._prototypes[timeline_id]._derived(
            declaring_axis(own, unit, number_type)
        )
        events._timeline_id = timeline_id
        self._slices[timeline_id] = events
        return events

    def holder_of(self, event_id: str) -> str | None:
        """The ID of the timeline holding the event *event_id*, or None."""
        return self._ids().get(event_id)

    # endregion

    # region Changing

    def add(self, timeline_id: str, data: EventData) -> None:
        """Add *data*'s rows to the rows of *timeline_id*.

        Rows of another number type are written in the timeline's, as every
        value put onto one timeline is (:meth:`EventData.to_number_type`).
        The timeline's columns grow by those *data* brings.

        Raises:
            ValueError: If *data* has another unit, or an event ID of it is
                held in the tree or twice in *data*; nothing changes then.
        """
        if not len(data):
            return
        data = self._on_axis(timeline_id, data)
        added = _ids_of(data.table)
        self._refuse_held(added, timeline_id)
        assert self._table is not None
        self._table = _concatenated([self._table, self._keyed(data.table, timeline_id)])
        held = self._schemas[timeline_id]
        for field in _own_schema(data.table.schema):
            if field.name not in held.names:
                held = held.append(field)
        self._schemas[timeline_id] = held
        self._sorted = False
        if self._index is not None:
            self._index.update(dict.fromkeys(added, timeline_id))
        self._slices.pop(timeline_id, None)

    def replace(self, timeline_id: str, data: EventData) -> None:
        """Make *data* the rows of *timeline_id*, its class and axis the timeline's.

        Raises:
            ValueError: If an event ID of *data* is held by another timeline
                of the tree or twice in *data*; nothing changes then.
        """
        if data._axes is not None:
            data = data.to_number_type(data.number_type)
        self._refuse_held(_ids_of(data.table), timeline_id, own=True)
        assert self._table is not None
        code = self._timeline_ids.index(timeline_id)
        others = self._table.filter(
            pc.not_equal(self._codes(self._table), pa.scalar(code, pa.int32()))
        )
        self._register(timeline_id, data)
        self._table = _concatenated([others, self._keyed(data.table, timeline_id)])
        self._sorted = False
        self._index = None
        self._slices.pop(timeline_id, None)

    def refuse_shared_events(self, other: TreeStore) -> None:
        """Refuse *other*'s subtree joining this tree where they share an ID.

        Raises:
            ValueError: If *other* holds a timeline this store holds, or an
                event ID this tree holds, naming the ID and both holders.
        """
        shared = [tid for tid in other._prototypes if tid in self._prototypes]
        if shared:
            raise ValueError(
                f"timeline {quote_id(shared[0])} is held in this tree already"
            )
        held = self._ids()
        for event_id, holder in other._ids().items():
            if event_id in held:
                raise ValueError(
                    f"Event id {quote_id(event_id)} of timeline {quote_id(holder)} "
                    f"is already held by an event of timeline "
                    f"{quote_id(held[event_id])}; event ids are unique across a "
                    "timeline tree"
                )

    def attach(self, other: TreeStore) -> None:
        """Take every row of *other*, a subtree joining this tree, and its map.

        The rows keep their timelines, coordinates and number types;
        *other* is left empty and must not be used again.

        Raises:
            ValueError: As :meth:`refuse_shared_events`; nothing changes then.
        """
        self.refuse_shared_events(other)
        tables = [self.table]
        for timeline_id in other._prototypes:
            self._register(timeline_id, other._prototypes[timeline_id])
            self._schemas[timeline_id] = other._schemas[timeline_id]
            tables.append(self._keyed(other.events_of(timeline_id).table, timeline_id))
        self._table = _concatenated(tables)
        self._sorted = False
        if self._index is not None:
            self._index.update(other._ids())
        other._release()

    def split(self, timeline_ids: Collection[str]) -> TreeStore:
        """Move the rows of *timeline_ids*, a subtree leaving the tree, to a store.

        Args:
            timeline_ids: The subtree's timelines, its root first.

        Returns:
            The subtree's own store, its root the first of *timeline_ids*.

        Raises:
            KeyError: If the store holds no timeline named.
            ValueError: If the root of this tree is named.
        """
        leaving = list(timeline_ids)
        missing = [tid for tid in leaving if tid not in self._prototypes]
        if missing:
            raise KeyError(
                f"No timeline {quote_id(missing[0])} in this tree. "
                f"Available timelines: {list(self._prototypes)}"
            )
        if self.timeline_ids[0] in leaving:
            raise ValueError(
                f"the root {quote_id(self.timeline_ids[0])} holds the tree; only "
                "a subtree leaves it"
            )
        root, *rest = leaving
        split = TreeStore(root, self.events_of(root))
        split._schemas[root] = self._schemas[root]
        tables = [split.table]
        for timeline_id in rest:
            split._register(timeline_id, self._prototypes[timeline_id])
            split._schemas[timeline_id] = self._schemas[timeline_id]
            tables.append(split._keyed(self.events_of(timeline_id).table, timeline_id))
        split._table = _concatenated(tables)
        split._sorted = False
        codes = pa.array([self._timeline_ids.index(tid) for tid in leaving], pa.int32())
        assert self._table is not None
        self._table = self._table.filter(
            pc.invert(pc.is_in(self._codes(self._table), value_set=codes))
        )
        for timeline_id in leaving:
            del self._prototypes[timeline_id]
            del self._schemas[timeline_id]
            del self._axes[timeline_id]
            self._slices.pop(timeline_id, None)
        self._sorted = False
        self._index = None
        return split

    def rename(self, old_id: str, new_id: str) -> None:
        """Give the timeline *old_id* the ID *new_id*; no row is rewritten.

        Raises:
            KeyError: If the store holds no timeline *old_id*.
            ValueError: If it holds a timeline *new_id*.
        """
        if old_id == new_id:
            return
        if old_id not in self._prototypes:
            raise KeyError(
                f"No timeline {quote_id(old_id)} in this tree. "
                f"Available timelines: {list(self._prototypes)}"
            )
        if new_id in self._prototypes:
            raise ValueError(
                f"timeline {quote_id(new_id)} is held in this tree already"
            )
        self._timeline_ids[self._timeline_ids.index(old_id)] = new_id
        assert self._table is not None
        index = self._table.schema.get_field_index(TIMELINE_ID)
        dictionary = pa.array(self._timeline_ids, pa.string())
        column = pa.chunked_array(
            [
                pa.DictionaryArray.from_arrays(chunk.indices, dictionary)
                for chunk in self._table.column(index).chunks
                if isinstance(chunk, pa.DictionaryArray)
            ]
            or [pa.DictionaryArray.from_arrays(pa.array([], pa.int32()), dictionary)]
        )
        self._table = self._table.set_column(
            index, self._table.schema.field(index), column
        )

        def renamed(mapping: dict[str, Any]) -> dict[str, Any]:
            """*mapping* with the key *old_id* replaced, in its place."""
            return {
                (new_id if key == old_id else key): value
                for key, value in mapping.items()
            }

        self._prototypes = renamed(self._prototypes)
        self._schemas = renamed(self._schemas)
        self._axes = renamed(self._axes)
        self._ranges = renamed(self._ranges)
        self._slices = renamed(self._slices)
        for events in (*self._slices.values(), *self._prototypes.values()):
            if events._timeline_id == old_id:
                events._timeline_id = new_id
        if self._index is not None:
            self._index = {
                event_id: (new_id if holder == old_id else holder)
                for event_id, holder in self._index.items()
            }

    # endregion

    # region Internals

    def _register(self, timeline_id: str, data: EventData) -> None:
        """Hold the class, axis and columns of *data* as *timeline_id*'s."""
        if timeline_id not in self._timeline_ids:
            self._timeline_ids.append(timeline_id)
        prototype = data._derived(data.table.slice(0, 0))
        prototype._axes = None
        prototype._timeline_id = timeline_id
        self._prototypes[timeline_id] = prototype
        self._schemas[timeline_id] = _own_schema(data.table.schema)
        self._axes[timeline_id] = (data.unit, data.number_type)

    def _on_axis(self, timeline_id: str, data: EventData) -> EventData:
        """*data* written on the axis of *timeline_id*.

        Raises:
            ValueError: If *data* has another unit.
        """
        unit, number_type = self._axes[timeline_id]
        if data.unit != unit:
            raise ValueError(f"Unit mismatch: {unit} vs {data.unit}")
        if data._axes is not None or data.number_type is not number_type:
            data = data.to_number_type(number_type)
        return data

    def _keyed(self, table: pa.Table, timeline_id: str) -> pa.Table:
        """*table*, its own column naming timelines dropped, keyed by *timeline_id*."""
        if TIMELINE_ID in table.column_names:
            table = table.drop_columns([TIMELINE_ID])
        table = _decoded(table)
        code = self._timeline_ids.index(timeline_id)
        column = pa.DictionaryArray.from_arrays(
            pa.array(np.full(table.num_rows, code, dtype=np.int32), pa.int32()),
            pa.array(self._timeline_ids, pa.string()),
        )
        return table.append_column(TIMELINE_ID, column)

    def _codes(self, table: pa.Table) -> pa.Array:
        """The dictionary index of every row's timeline in *table*."""
        chunks = [
            chunk.indices
            for chunk in table.column(TIMELINE_ID).chunks
            if isinstance(chunk, pa.DictionaryArray)
        ]
        if not chunks:
            return pa.array([], pa.int32())
        return pa.concat_arrays(chunks)

    def _sort(self) -> None:
        """Sort the table by ``(timeline_id, start)`` if rows were added since.

        A stable sort on the timeline's dictionary index and the float side
        of ``start`` (rows without a start last); a table already in that
        order is kept as it is, without a copy. The row range of each
        timeline is recorded for its slice.
        """
        if self._sorted:
            return
        assert self._table is not None
        table = self._table
        codes = self._codes(table).to_numpy(zero_copy_only=False)
        starts = (
            pc.struct_field(table.column("start"), "value").to_numpy()
            if "start" in table.column_names and table.num_rows
            else np.zeros(table.num_rows)
        )
        order = np.lexsort((np.where(np.isnan(starts), np.inf, starts), codes))
        if not np.array_equal(order, np.arange(len(order))):
            table = table.take(pa.array(order, pa.int64()))
            codes = codes[order]
        index = table.schema.get_field_index(TIMELINE_ID)
        column = pa.DictionaryArray.from_arrays(
            pa.array(codes.astype(np.int32), pa.int32()),
            pa.array(self._timeline_ids, pa.string()),
        )
        self._table = table.set_column(index, table.schema.field(index), column)
        bounds = np.searchsorted(codes, np.arange(len(self._timeline_ids) + 1))
        self._ranges = {
            timeline_id: (int(bounds[code]), int(bounds[code + 1] - bounds[code]))
            for code, timeline_id in enumerate(self._timeline_ids)
        }
        self._sorted = True

    def _ids(self) -> dict[str, str]:
        """The index: every event ID the tree holds, to its timeline's ID."""
        if self._index is None:
            table = self.table
            index: dict[str, str] = {}
            if "id" in table.column_names and table.num_rows:
                ids = table.column("id").to_pylist()
                holders = pc.cast(table.column(TIMELINE_ID), pa.string()).to_pylist()
                for event_id, holder in zip(ids, holders, strict=True):
                    if event_id is not None and holder is not None:
                        index[event_id] = holder
            self._index = index
        return self._index

    def _refuse_held(
        self, ids: list[str], timeline_id: str, *, own: bool = False
    ) -> None:
        """Refuse event IDs joining *timeline_id* that the tree holds.

        Args:
            ids: The IDs of the rows joining.
            timeline_id: The timeline they join.
            own: The rows replace the timeline's own, whose IDs are not held
                against them.

        Raises:
            ValueError: Naming the ID, the timeline holding it and the one
                it would join, or the ID held twice by the rows joining.
        """
        _refuse_repeated(ids, timeline_id)
        held = self._ids()
        for event_id in ids:
            holder = held.get(event_id)
            if holder is None or (own and holder == timeline_id):
                continue
            if holder == timeline_id:
                raise ValueError(
                    f"Event id {quote_id(event_id)} is already held by an event of "
                    f"timeline {quote_id(timeline_id)}"
                )
            raise ValueError(
                f"Event id {quote_id(event_id)} is already held by an event of "
                f"timeline {quote_id(holder)}, so it cannot join timeline "
                f"{quote_id(timeline_id)}: event ids are unique across a timeline "
                "tree"
            )

    def _release(self) -> None:
        """Drop every row and timeline: the rows moved to another store."""
        self._timeline_ids = []
        self._prototypes = {}
        self._schemas = {}
        self._axes = {}
        self._table = None
        self._ranges = {}
        self._slices = {}
        self._index = None

    # endregion
