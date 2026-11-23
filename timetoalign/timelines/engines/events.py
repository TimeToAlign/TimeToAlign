"""Provide event storage operations for timeline instances."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import ValidationError

from timetoalign.core import (
    Box,
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    Duration,
    Event,
    EventReference,
    NumberType,
    Projection,
    Shape,
    struct_to_coordinate,
)
from timetoalign.core.ids import quote_id
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    combine_number_columns,
    combine_numbers,
    compare_number_column,
    coordinate_numeric_value,
    exact_coordinate_value,
    express_as,
    shift_coordinate,
    to_canonical,
)
from timetoalign.core.timestamp import _format_coordinate_value
from timetoalign.storage import EventData
from timetoalign.storage.events import (
    BOX_EVENT_TYPE,
    PROJECTION_EVENT_TYPE,
    SHAPE_FIELDS,
    _event_rows,
    box_rows_on_the_axis,
    mint_event_ids,
    projection_ids,
    refuse_shared_projection_ids,
)

from .children import placed_on

if TYPE_CHECKING:
    from ..base import Timeline

SEGMENT_EVENT_TYPE = "Segment"

#: The keys a Projection row may hold a value under: its ID, its type, its
#: position and the event it stands for.
_PROJECTION_KEYS = frozenset(
    {
        "id",
        "event_type",
        "temporal_type",
        "start",
        "instant",
        "end",
        "duration",
        "projection_of",
    }
)


def _check_projection_row(row: dict[str, Any], timeline_id: str) -> None:
    """Refuse a Projection row that repeats more than a Projection holds.

    Args:
        row: The event row.
        timeline_id: The ID of the timeline the row is added to.

    Raises:
        ValueError: If the row holds a value under another key, its
            ``projection_of`` is not an event reference, or it names an
            event of the timeline the Projection is added to.
    """
    extra = sorted(
        key
        for key, value in row.items()
        if value is not None and key not in _PROJECTION_KEYS
    )
    if extra:
        raise ValueError(
            f"A Projection holds its ID, its position and projection_of, nothing "
            f"else of the event it stands for; the row also gives {extra}"
        )
    reference = row.get("projection_of")
    if reference is None:
        return
    if not isinstance(reference, EventReference):
        try:
            reference = EventReference.model_validate(reference)
        except ValueError as error:
            raise ValueError(
                f"A Projection's projection_of is {{'timeline_id': ..., "
                f"'event_id': ...}} or None, not {reference!r}"
            ) from error
    if reference.timeline_id == timeline_id:
        raise ValueError(
            f"A Projection added to timeline {quote_id(timeline_id)} stands for an "
            f"event of another timeline; its projection_of names "
            f"{quote_id(timeline_id)} itself"
        )


def _image_origin(timeline: Timeline) -> Coordinate:
    """The x at which *timeline* stands in the image whose space it measures.

    The ``x_min`` of the shape the timeline's ``meta["box"]`` states, read
    in the timeline's unit and number type
    (:meth:`~timetoalign.core.Shape.from_dict`); 0 where it states none.

    Raises:
        ValueError: If ``meta["box"]`` is no shape on the timeline's axis.
    """
    box = timeline.meta.get("box")
    if box is None:
        return Coordinate(0, timeline.unit, number_type=timeline.number_type)
    return Shape.from_dict(
        box, unit=timeline.unit, number_type=timeline.number_type
    ).x_min


def _named(event_id: str | None) -> str:
    """A Box as a message names it (:class:`~timetoalign.core.Event`'s own
    naming): ``Box 'm1'``, or ``A Box`` without ID."""
    return Box.model_construct(id=event_id)._name


def _refusal_of(error: ValidationError) -> ValueError:
    """The refusal a scalar's validator raised, as it raised it; *error*
    itself where pydantic refused the input before the validator ran."""
    for detail in error.errors():
        cause = detail.get("ctx", {}).get("error")
        if isinstance(cause, ValueError):
            return ValueError(*cause.args)
    return error


#: The cells and the reference that make two Projections of one timeline the
#: same one: the event it stands for (or none) and its exact position.
_PROJECTION_CELLS = ("start", "end", "duration")


def _projection_keys(table: pa.Table) -> pa.Table:
    """The columns that tell a timeline's Projections apart, without nulls.

    Per row the ``projection_of`` reference and the exact ``start``, ``end``
    and ``duration`` cells (each side of the number, so a float compares as
    its value and an exact number as its ratio), every nullable column
    beside a flag of its presence, so that a join matches a missing value
    with a missing one. A column the table lacks is missing in every row.
    """
    n = table.num_rows
    columns: dict[str, pa.Array] = {}

    def struct(name: str, fields: dict[str, pa.DataType]) -> list[pa.Array]:
        if name in table.column_names:
            data = table.column(name).combine_chunks()
            if isinstance(data, pa.StructArray) and len(data.type) > 0:
                flat = dict(
                    zip((f.name for f in data.type), data.flatten(), strict=True)
                )
                return [
                    pc.is_valid(data),
                    *(flat.get(key, pa.nulls(n, kind)) for key, kind in fields.items()),
                ]
        return [
            pa.array([False] * n, pa.bool_()),
            *(pa.nulls(n, kind) for kind in fields.values()),
        ]

    number = {"value": pa.float64(), "numerator": pa.int64(), "denominator": pa.int64()}
    fills = {pa.float64(): 0.0, pa.int64(): 0, pa.string(): ""}
    for name in _PROJECTION_CELLS:
        present, *parts = struct(name, number)
        columns[f"{name}_present"] = present
        for key, part in zip(number, parts, strict=True):
            part = pc.cast(part, number[key])
            columns[f"{name}_{key}"] = pc.fill_null(part, fills[number[key]])
    reference = {"timeline_id": pa.string(), "event_id": pa.string()}
    present, *parts = struct("projection_of", reference)
    columns["projection_of_present"] = present
    for key, part in zip(reference, parts, strict=True):
        columns[f"projection_of_{key}"] = pc.fill_null(pc.cast(part, pa.string()), "")
    return pa.table(columns)


def _held_by_key(
    held: pa.Table, rows: pa.Table, *, with_id: bool = False
) -> list[str | None]:
    """Per row of *rows* the ID of a held Projection with its keys, or ``None``.

    Both tables hold the columns of :func:`_projection_keys`, *held* also
    ``held_id``. Without ``with_id`` the first Projection held with a row's
    keys answers; with it, *rows* holds ``held_id`` as well (a given ID) and
    a row is answered only by the Projection held under that ID with equal
    keys. One hash join, whatever the number of Projections held.
    """
    keys = [name for name in held.column_names if name != "held_id"]
    if with_id:
        keys.append("held_id")
        right = held.group_by(keys).aggregate([])
        right = right.append_column(
            "match", pa.array(np.ones(right.num_rows, dtype=bool))
        )
    else:
        right = held.group_by(keys, use_threads=False).aggregate([("held_id", "first")])
        right = right.rename_columns([*keys, "match"])
    left = rows.append_column("row", pa.array(np.arange(rows.num_rows), pa.int64()))
    joined = left.join(right, keys=keys, join_type="left outer").sort_by("row")
    match = joined.column("match").to_pylist()
    if not with_id:
        return match
    given = joined.column("held_id").to_pylist()
    return [
        event_id if found else None
        for event_id, found in zip(given, match, strict=True)
    ]


def _null_columns_typed_as(data: EventData, sources: list[pa.Schema]) -> EventData:
    """*data* with each column holding only nulls typed as in *sources*.

    Rows rebuilt into a table (:meth:`EventData.from_dicts`) carry their
    further columns' types in their values alone, so a column holding no
    value (``mn_onset`` from a loader that states none) would come out as
    strings. Such a column takes the field (type and metadata) it has in
    the first of *sources* holding it, the tables the rows were read from.
    """
    table = data.table
    changed = False
    for schema in sources:
        for field in schema:
            index = table.schema.get_field_index(field.name)
            if (
                index < 0
                or table.schema.field(index).type == field.type
                or table.column(index).null_count != len(table)
            ):
                continue
            table = table.set_column(index, field, pa.nulls(len(table), field.type))
            changed = True
    return data._derived(table) if changed else data


def _held_by(row: dict[str, Any], child_id: str) -> dict[str, Any]:
    """*row*, an event a child reported, tagged with the timeline holding it.

    ``source_timeline`` names the timeline that holds the event: the child
    for its own events, which carry no tag (or a null one, beside a
    descendant's tagged rows), a deeper descendant's name kept. A row of the
    timeline's own events is never passed here, so in a parent's table a
    null ``source_timeline`` means the parent's own event at every depth.
    """
    if row.get("source_timeline") is None:
        row["source_timeline"] = child_id
    return row


def tree_projection_ids(
    roots: Sequence[Any], *, skip: Any = None
) -> tuple[pa.Array, pa.Array]:
    """The Projection IDs every timeline of the trees of *roots* holds.

    Read column-wise, never as Python objects: the trees of a manifest hold
    one Projections table between them.

    Args:
        roots: Root timelines, each tree once.
        skip: A timeline whose Projections are left out, or None.

    Returns:
        The IDs and, row for row, the ID of the timeline holding each, the
        trees in the order given, each root before its descendants.
    """
    ids: list[pa.Array] = []
    holders: list[pa.Array] = []
    for root in roots:
        for timeline in (root, *(child for _, child in root._iter_descendants())):
            if timeline is skip:
                continue
            held = projection_ids(timeline._events.table)
            if len(held):
                ids.append(held)
                holders.append(pa.array([timeline.id] * len(held), pa.string()))
    if not ids:
        return pa.array([], pa.string()), pa.array([], pa.string())
    return pa.concat_arrays(ids), pa.concat_arrays(holders)


def _decoded_rows(data: EventData) -> list[dict[str, Any]]:
    """Return rows with every rational-shaped column decoded."""
    rows = list(data)
    for name in data.table.column_names:
        values = data.column_values(name)
        for row, value in zip(rows, values, strict=True):
            row[name] = value
    return rows


class EventsMixin:
    """Provide event storage operations for timeline instances."""

    @property
    def n_events(self) -> int:
        """Number of events (excluding segment events)."""
        # Filter out segment events from count
        non_segment = self._events.filter(event_type=SEGMENT_EVENT_TYPE)
        return len(self._events) - len(non_segment)

    @property
    def events(self) -> EventData:
        """The underlying EventData (read-only access)."""
        return self._events

    def _ensure_capacity(
        self,
        required_end: CoordinateValue,
        allow_expansion: bool = False,
    ) -> None:
        """Ensure timeline is long enough, expanding if permitted.

        A timeline of unknown length has no length to enforce: positions on it
        are not bounded above, and adding content leaves the length unknown,
        except that a segment of unknown length ends at its successor's
        offset (the last one at the line's length). A timeline that expands
        must still fit its parent, which never grows with it.

        Args:
            required_end: The coordinate that must be within bounds.
            allow_expansion: If True, override lock to expand.

        Raises:
            ValueError: If expansion is needed but not permitted, the content
                would pass where its parent ends a segment of unknown length,
                or the parent cannot contain the new end.
        """
        required = (
            required_end.value if isinstance(required_end, Coordinate) else required_end
        )

        if self._length is None:
            bound = self._extent
            if bound is not None and required > bound:
                raise ValueError(
                    f"Adding content ending at {_format_coordinate_value(required)} "
                    f"to timeline '{self._id}' passes "
                    f"{_format_coordinate_value(bound)}, where its parent "
                    f"'{self._parent.id}' ends it while its length is unknown"
                )
            return

        if required <= self._length.value:
            return

        if self._locked and not allow_expansion:
            raise ValueError(
                f"Adding content ending at {_format_coordinate_value(required)} "
                f"exceeds timeline length "
                f"{_format_coordinate_value(self._length.value)}. "
                f"Timeline is locked. "
                "Use allow_expansion=True to override."
            )

        self._set_length(self._make_coordinate(required))

    def add_events(
        self,
        rows: Event | Sequence[dict[str, Any] | Event],
        allow_expansion: bool = False,
    ) -> list[str]:
        """Add events to the timeline.

        Only ``event_type`` and a coordinate are strictly required per dict.
        Missing fields are filled in automatically:

        - **id**: minted above every ID held on this timeline, its prefix the
          event type in lower case: ``note_1``, ``beat_1``, ``projection_1``.
          An ID once given on the timeline is never given again.
        - **temporal_type**: inferred from keys -- ``"interval"`` when both
          ``start`` and ``end`` (or ``duration``) are present, ``"instant"``
          otherwise.

        Positions are stored on this timeline's axis. A ``start``, ``end`` or
        ``instant`` given as a :class:`~timetoalign.core.Coordinate` in another
        unit is converted through the timeline's C-Map; a ``duration`` given as
        a :class:`~timetoalign.core.Duration` (or ``Coordinate``) in another
        unit is the span from the start it is added to, converted as the two
        ends of that span are, so its start must be given in the same unit.
        Each converted value is written in the timeline's number type. Bare
        numbers and struct dicts are taken as on this timeline's axis.

        A Projection (``event_type`` ``"Projection"``) holds ``id``, its
        position and ``projection_of``, the event of another timeline it
        stands for, as a dict ``{"timeline_id": ..., "event_id": ...}`` or an
        :class:`~timetoalign.core.EventReference` (``None`` where none is
        known); it repeats nothing else of that event.

        A Box (``event_type`` ``"Box"``) is an area of the image the
        timeline runs along: its shape, in the image's space, and the
        interval of its x extent on the timeline. Its shape must be on the
        timeline's axis (its unit and number type; a shape is never
        converted), and its start must be the shape's ``x_min`` less the x
        at which the timeline stands in its image: the ``x_min`` of the
        shape ``meta["box"]`` states, 0 where it states none. A shape given
        as the memory cell of its column is read through the shape's
        constructor, as every shape's numbers are (never rounded). A Box row
        may state ``unit``, ``number_type`` and ``timeline_id``, as a
        package's Box table does: they must name this timeline and its axis,
        and are not stored.

        An :class:`~timetoalign.core.Event` object (a
        :class:`~timetoalign.core.Projection`, a :class:`~timetoalign.core.Box`)
        may be given in place of its row, alone or in the list.

        Args:
            rows: Event dictionaries or events (one or a list).
                Required keys of a dictionary:
                - event_type: class name (e.g. ``"Beat"``, ``"Note"``,
                  ``"Projection"``)
                - instant: coordinate (for instant events), **or**
                - start, end: coordinates (for interval events)
            allow_expansion: If True, expand timeline if events exceed length.

        Returns:
            The IDs of the added events, in the order of *rows*.

        Raises:
            ValueError: If an ID is already held by an event of this timeline
                or given twice, a Projection row holds a key a Projection does
                not have or names an event of this timeline, a Box's shape is
                off the timeline's axis (or its row's ``unit``,
                ``number_type`` or ``timeline_id`` names another), a number of
                its shape is one the axis cannot hold exactly, or its start
                contradicts its shape, a
                position cannot be converted, or the events exceed the length
                and expansion is not allowed. Nothing is added then.

        Examples:
            >>> from timetoalign.timelines import ContinuousPhysicalTimeline
            >>> tl = ContinuousPhysicalTimeline(length=10.0)
            >>> tl.add_events([
            ...     {"event_type": "Beat", "instant": 0.0},
            ...     {"event_type": "Note", "start": 0.0, "end": 0.5},
            ... ])
            ['beat_1', 'note_1']
        """
        # A shape is never converted: refused before any position of the rows
        # is converted, so the refusal names the shape, not a missing C-Map.
        rows = box_rows_on_the_axis(
            _event_rows(rows), self._unit, self._number_type, timeline_id=self._id
        )
        if not rows:
            return []

        # Resolve unit-qualified coordinates into this timeline's native unit
        # before anything measures or stores them. A ``Coordinate`` whose unit
        # differs from the native one (e.g. a measure boundary given in seconds
        # on a samples timeline) is converted through the timeline's C-Maps;
        # bare numbers, struct dicts and native-unit coordinates pass through
        # unchanged. Without this, a derived-unit value would be stored verbatim
        # under the native unit label -- a silent mis-unit.
        rows = [self._resolve_row_coordinates(row) for row in rows]
        for row in rows:
            if row.get("event_type") == PROJECTION_EVENT_TYPE:
                _check_projection_row(row, self._id)
        given_ids = [row.get("id") for row in rows]
        rows = self._rows_with_ids(rows)
        added = self._event_data_of(rows)
        self._refuse_boxes_off_their_shapes(added, given_ids)

        # The farthest end, in this timeline's number type: exact on an
        # exact axis, so it is compared and grown to as it will be stored.
        max_coord = max(self._get_event_end_coordinate(row) for row in rows)
        self._ensure_capacity(max_coord, allow_expansion)
        self._events.extend(added)
        return [row["id"] for row in rows]

    def _refuse_boxes_off_their_shapes(
        self, added: EventData, given_ids: Sequence[str | None]
    ) -> None:
        """Refuse the rows *added* would add if a Box contradicts its shape.

        The one check of a Box row, whatever brought it (a
        :class:`~timetoalign.core.Box`, a dict, a manifest's event rows, a
        package's Box table): in this order, every Box row holds exactly one
        shape, a duration, a ``start`` equal to ``x_min - o`` (*o* the x at
        which this timeline stands in its image, :func:`_image_origin`) and
        a duration equal to the shape's x extent (``x_max - x_min``), each
        compared exactly on an exact axis (:func:`compare_number_column`),
        all rows at once. A Box's own rules (one shape, its duration) are
        refused with the Box's own message: the first row breaking one is
        read as a :class:`~timetoalign.core.Box`, which refuses it.

        Args:
            added: The rows as they will be stored.
            given_ids: Row for row of the rows given, the ID the caller gave,
                or None.

        Raises:
            ValueError: Naming the first Box breaking a rule; for the start,
                the timeline, both x and the start it must have.
        """
        table = added.table
        if "event_type" not in table.column_names:
            return
        is_box = pc.fill_null(
            pc.equal(table.column("event_type"), BOX_EVENT_TYPE), False
        )
        if not pc.any(is_box).as_py():
            return
        boxes = table.filter(is_box)
        positions = pc.indices_nonzero(is_box).to_pylist()
        number_type = self._number_type

        def refuse_as_the_box(wrong: pa.Array) -> None:
            if not pc.any(wrong).as_py():
                return
            at = pc.index(wrong, True).as_py()
            row = {**boxes.slice(at, 1).to_pylist()[0], "id": given_ids[positions[at]]}
            if row["duration"] is None:
                raise ValueError(
                    f"{_named(row['id'])}: it has no duration; a Box's duration "
                    "is its shape's x extent."
                )
            try:
                box = Box.from_row(row, unit=self._unit, number_type=number_type)
            except ValidationError as error:
                raise _refusal_of(error) from None
            raise ValueError(  # pragma: no cover - the Box refuses first
                f"{box._name} contradicts its shape."
            )

        n_shapes = pa.array(np.zeros(len(boxes), dtype=np.int64))
        x_min = x_max = pa.nulls(len(boxes), RATIONAL_STRUCT_TYPE)
        for name, field_class in SHAPE_FIELDS.items():
            if name not in boxes.column_names:
                continue
            data = boxes.column(name).combine_chunks()
            n_shapes = pc.add(n_shapes, pc.cast(data.is_valid(), pa.int64()))
            field = field_class.from_field((data, boxes.schema.field(name)))
            x_min = pc.if_else(data.is_valid(), field.x_min(), x_min)
            x_max = pc.if_else(data.is_valid(), field.x_max(), x_max)
        refuse_as_the_box(pc.not_equal(n_shapes, 1))
        duration = boxes.column("duration").combine_chunks()
        refuse_as_the_box(duration.is_null())

        origin = _image_origin(self)
        expected = combine_number_columns(x_min, origin.value, "subtract", number_type)
        start = boxes.column("start").combine_chunks()
        wrong = pc.invert(compare_number_column(start, expected, "equal", number_type))
        if pc.any(wrong).as_py():
            at = pc.index(wrong, True).as_py()

            def written(cells: pa.StructArray) -> str:
                return _format_coordinate_value(
                    struct_to_coordinate(cells[at].as_py(), number_type)
                )

            raise ValueError(
                f"{_named(given_ids[positions[at]])} starts at {written(start)} on "
                f"timeline {quote_id(self._id)}, but its shape starts at x "
                f"{written(x_min)} and the timeline stands at x "
                f"{_format_coordinate_value(origin.value)} of its image "
                f'(meta["box"]): its start is {written(expected)}.'
            )
        extent = combine_number_columns(x_max, x_min, "subtract", number_type)
        refuse_as_the_box(
            pc.invert(compare_number_column(duration, extent, "equal", number_type))
        )

    def _add_stated_events(self, rows: list[dict[str, Any]]) -> None:
        """Add the event rows a source states, every Box checked.

        For rows read as a source writes them (a manifest's event rows,
        :meth:`from_events`): no ID is minted or compared and the length is
        not grown, but a Box row is refused as :meth:`add_events` refuses it
        (:meth:`_refuse_boxes_off_their_shapes`), so the library reads no
        Box it would not add.

        Raises:
            ValueError: As :meth:`add_events` refuses a Box; nothing is added.
        """
        if not rows:
            return
        rows = box_rows_on_the_axis(
            rows, self._unit, self._number_type, timeline_id=self._id
        )
        added = self._event_data_of(rows)
        self._refuse_boxes_off_their_shapes(added, [row.get("id") for row in rows])
        self._events.extend(added)

    def add_projections(
        self,
        projections: Projection | Sequence[dict[str, Any] | Projection],
        allow_expansion: bool = False,
    ) -> list[str]:
        """Add Projections to the timeline, reusing those it already holds.

        Get-or-add: each Projection is placed on the timeline's axis as
        :meth:`add_events` places it, and the timeline keeps one Projection
        per thing it stands for. A Projection without an ``id`` reuses the
        Projection the timeline holds with the same ``projection_of`` (the
        same event of another timeline, or none) at the same exact ``start``
        and ``duration``; a Projection with an ``id`` reuses the one held
        under that id when its content is equal. Rows of one call are taken
        alike, so two rows standing for one thing give one Projection.
        Everything else is added, its id kept or minted
        (``projection_N``).

        Args:
            projections: :class:`~timetoalign.core.Projection` objects or
                their event rows (one or a list).
            allow_expansion: If True, expand the timeline if a Projection
                added exceeds its length.

        Returns:
            The ID of the Projection standing for each row, in row order.

        Raises:
            ValueError: If a row is not a Projection, a given ID is held by
                another event of the timeline or given twice for different
                positions, or a row is refused by :meth:`add_events`.
                Nothing is added then.

        Examples:
            >>> from timetoalign.core import Coordinate, EventReference, Projection
            >>> from timetoalign.timelines import ContinuousPhysicalTimeline
            >>> tl = ContinuousPhysicalTimeline(length=10.0)
            >>> n1 = EventReference(timeline_id="score", event_id="n1")
            >>> at = Coordinate(2.5, "seconds")
            >>> tl.add_projections([Projection(start=at, projection_of=n1)] * 2)
            ['projection_1', 'projection_1']
            >>> tl.add_projections(Projection(start=at, projection_of=n1))
            ['projection_1']
        """
        rows = [self._resolve_row_coordinates(row) for row in _event_rows(projections)]
        if not rows:
            return []
        for row in rows:
            if row.get("event_type") != PROJECTION_EVENT_TYPE:
                raise ValueError(
                    f"add_projections adds Projections, not a row of type "
                    f"{row.get('event_type')!r}"
                )
            _check_projection_row(row, self._id)
        # The cells each row is stored with, read as add_events would store them.
        placed = (
            type(self._events)
            .from_dicts(
                [{**row, "id": str(index)} for index, row in enumerate(rows)],
                self._unit,
                self._number_type,
            )
            .table
        )
        keys = list(zip(*(column.to_pylist() for column in _projection_keys(placed))))
        table = self._events.table
        held = table.filter(pc.equal(table.column("event_type"), PROJECTION_EVENT_TYPE))
        held_keys = _projection_keys(held).append_column(
            "held_id", held.column("id").combine_chunks()
        )
        by_key = _held_by_key(held_keys, _projection_keys(placed))
        given = pa.array([row.get("id") for row in rows], type=pa.string())
        same_held = _held_by_key(
            held_keys,
            _projection_keys(placed).append_column("held_id", given),
            with_id=True,
        )
        held_elsewhere = pc.is_in(given, value_set=table.column("id")).to_pylist()

        ids: list[str | int] = []
        first: dict[tuple[Any, ...], str | int] = {}
        given_keys: dict[str, tuple[Any, ...]] = {}
        added: list[dict[str, Any]] = []
        for index, (row, key) in enumerate(zip(rows, keys, strict=True)):
            event_id = row.get("id")
            if event_id is not None:
                seen = event_id in given_keys
                if given_keys.setdefault(event_id, key) != key:
                    raise ValueError(
                        f"Projection id {quote_id(event_id)} is given twice for "
                        f"different positions in one call on timeline "
                        f"{quote_id(self._id)}"
                    )
                if not seen and same_held[index] is None:
                    if held_elsewhere[index]:
                        raise ValueError(
                            f"Event id {quote_id(event_id)} is already held by "
                            f"another event of timeline {quote_id(self._id)}"
                        )
                    added.append(row)
                ids.append(event_id)
                first.setdefault(key, event_id)
                continue
            if by_key[index] is not None:
                ids.append(by_key[index])
            elif key in first:
                ids.append(first[key])
            else:
                first[key] = len(added)
                ids.append(len(added))
                added.append(row)
        minted = self.add_events(added, allow_expansion) if added else []
        return [minted[entry] if isinstance(entry, int) else entry for entry in ids]

    def _manifest_projection_ids(self) -> tuple[pa.Array, pa.Array]:
        """The Projection IDs every other timeline of this one's manifest holds.

        The manifest is this timeline's tree and every tree the groups and
        bundles holding it hold, as for every other ID of the manifest
        (``_manifest_ids``): the package written from a bundle holds one
        Projections table for all of them.

        Returns:
            The IDs and, row for row, the ID of the timeline holding each.
        """
        return tree_projection_ids(self._manifest_roots(), skip=self)

    def _rows_with_ids(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return *rows*, each with an ID new to this timeline.

        A given ID is kept; one already held by an event of this timeline, or
        given twice, is refused. A missing one is minted above both. A
        Projection's ID is unique in the timeline's manifest
        (:meth:`_manifest_projection_ids`): where the rows hold a Projection,
        a given Projection ID another timeline of the manifest holds as a
        Projection is refused, naming the first timeline holding it and this
        one, and every ID is minted above the Projection IDs held there too.
        Timelines that joined after creating their Projections may hold one
        ID each; that refuses nothing here (the package writer refuses it).
        The held IDs are read column-wise, never as Python objects.

        Raises:
            ValueError: If a given ID is held already or given twice, or a
                given Projection ID names a Projection of another timeline of
                the manifest.
        """
        given = [row["id"] for row in rows if row.get("id") is not None]
        held = self._events.table.column("id")
        given_ids = pa.array(given, type=pa.string())
        around = pa.array([], pa.string())
        if any(row.get("event_type") == PROJECTION_EVENT_TYPE for row in rows):
            around, holders = self._manifest_projection_ids()
            given_projections = pa.array(
                [
                    row["id"]
                    for row in rows
                    if row.get("id") is not None
                    and row.get("event_type") == PROJECTION_EVENT_TYPE
                ],
                pa.string(),
            )
            # Only a given ID is checked: timelines that joined after creating
            # their Projections may hold one ID each, refused by the writer.
            at = pc.index_in(given_projections, value_set=around)
            hits = at.filter(at.is_valid())
            refuse_shared_projection_ids(
                pa.concat_arrays(
                    [around.take(hits), given_projections.filter(at.is_valid())]
                ),
                pa.concat_arrays(
                    [holders.take(hits), pa.array([self._id] * len(hits), pa.string())]
                ),
            )
        if given:
            repeated = sorted(
                event_id for event_id, count in Counter(given).items() if count > 1
            )
            if repeated:
                raise ValueError(
                    f"Event id {quote_id(repeated[0])} is given twice in one "
                    f"call to add_events on timeline {quote_id(self._id)}"
                )
            clash = pc.filter(given_ids, pc.is_in(given_ids, value_set=held))
            if len(clash):
                raise ValueError(
                    f"Event id {quote_id(clash[0].as_py())} is already held by an "
                    f"event of timeline {quote_id(self._id)}"
                )
        missing = [row.get("event_type") for row in rows if row.get("id") is None]
        if not missing:
            return rows
        minted = iter(
            mint_event_ids(
                missing,
                pa.chunked_array(
                    [*pc.cast(held, pa.string()).chunks, given_ids, around],
                    pa.string(),
                ),
            ).to_pylist()
        )
        return [
            row if row.get("id") is not None else {**row, "id": next(minted)}
            for row in rows
        ]

    def _resolve_row_coordinates(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return a row whose coordinate fields are in the native unit.

        Each of ``start`` / ``end`` / ``instant`` given as a unit-qualified
        :class:`~timetoalign.core.Coordinate` is resolved through
        :meth:`get_coordinate_at`, so a value expressed in a derived unit is
        converted into the timeline's native unit via its C-Maps. A
        unit-qualified ``duration`` is the span from its start: the span's end
        is converted as the start is, and the duration is the difference of
        the two, on this timeline's axis. Raw numbers and coordinate struct
        dicts are left untouched, and the input dict is never mutated in place.

        Args:
            row: An event dictionary as passed to :meth:`add_events`.

        Returns:
            The row, copied only when a coordinate needed resolving.

        Raises:
            ValueError: If a duration in another unit is added to a start not
                given in that unit, or a unit has no conversion.
        """
        resolved: dict[str, Any] | None = None
        for key in ("start", "end", "instant"):
            value = row.get(key)
            if isinstance(value, Coordinate):
                if resolved is None:
                    resolved = dict(row)
                resolved[key] = self.get_coordinate_at(value, format="coordinate").value
        duration = row.get("duration")
        if isinstance(duration, (Coordinate, Duration)):
            if resolved is None:
                resolved = dict(row)
            resolved["duration"] = self._native_duration(row, duration)
        return resolved if resolved is not None else row

    def _native_duration(
        self, row: dict[str, Any], duration: Coordinate | Duration
    ) -> int | float | Fraction:
        """Return a unit-qualified *duration* of *row* on this timeline's axis.

        In this timeline's unit it is the value itself, written in the
        timeline's number type. In another unit it is the span from the row's
        start: start and ``start + duration`` are converted through the C-Map
        and the duration is their difference, exact on an exact axis.

        Raises:
            ValueError: If the duration is in another unit and the row's start
                is not given as a ``Coordinate`` in that unit.
        """
        if duration.unit == self._unit:
            return express_as(duration.value, self._number_type)
        start = row.get("start", row.get("instant"))
        if not isinstance(start, Coordinate) or start.unit != duration.unit:
            raise ValueError(
                f"A duration in {duration.unit} added to timeline "
                f"{quote_id(self._id)} (in {self._unit}) is a span from its start, "
                f"so the start must be given as a Coordinate in {duration.unit} too"
            )
        span = Duration(duration.value, duration.unit, number_type=duration.number_type)
        start_native = self.get_coordinate_at(start, format="coordinate").value
        end_native = self.get_coordinate_at(start + span, format="coordinate").value
        return express_as(end_native - start_native, self._number_type)

    def _add_events_unchecked(self, rows: list[dict[str, Any]]) -> None:
        """Add events without validation (internal use).

        Uses the concrete type of ``self._events`` to create new EventData,
        ensuring schema compatibility when the events store is a subclass
        (e.g., MeasureData) that was assigned directly via
        ``EventData.create_timeline()``.

        Args:
            rows: Event dictionaries to add.
        """
        if not rows:
            return
        self._events.extend(self._event_data_of(rows))

    def _event_data_of(self, rows: list[dict[str, Any]]) -> EventData:
        """The event table of *rows* on this timeline's axis, of the concrete
        type of ``self._events`` (a subclass such as MeasureData included)."""
        return type(self._events).from_dicts(rows, self._unit, self._number_type)

    def _get_event_end_coordinate(self, row: dict[str, Any]) -> int | float | Fraction:
        """Extract the end coordinate from an event dict.

        The end is written in this timeline's number type, as the row will
        be stored: on an ``int`` or ``fraction`` axis it is exact (``start +
        duration`` added on the ratios), on a ``float`` axis a float.

        Args:
            row: Event dictionary. Coordinate fields may be raw numbers
                or struct dicts (``{"value": ..., ...}``).

        Returns:
            The end coordinate in this timeline's number type.
        """
        number_type = self._number_type
        start = row.get("start")
        if row.get("instant") is not None:
            end = self._coord_to_value(row["instant"])
        elif row.get("end") is not None:
            end = self._coord_to_value(row["end"])
        elif start is not None and row.get("duration") is not None:
            return combine_numbers(
                self._coord_to_value(start),
                self._coord_to_value(row["duration"]),
                "add",
                number_type,
            )
        else:
            end = 0 if start is None else self._coord_to_value(start)
        return to_canonical(end, number_type)

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Look up a single event by its ``id`` field.

        Searches this timeline's events first, then recursively searches
        all children. When found in a child, coordinates are adjusted to
        the root timeline's coordinate system.

        Args:
            event_id: The event identifier to search for.

        Returns:
            A dictionary representing the event row, or ``None`` if not
            found anywhere in the hierarchy.

        Examples:
            >>> from timetoalign import Timeline, TimeUnit
            >>> timeline = Timeline(length=4, unit=TimeUnit.quarters)
            >>> timeline.add_events([{"event_type": "Note", "start": 0, "end": 1}])
            ['note_1']
            >>> timeline.get_event("note_1")["id"]
            'note_1'
            >>> timeline.get_event("nonexistent") is None
            True
        """
        # Search own events first
        table = self._events.table
        mask = pc.equal(table.column("id"), event_id)
        filtered = table.filter(mask)
        if filtered.num_rows > 0:
            return filtered.to_pylist()[0]

        # Search children
        for child_id, child in self._children.items():
            result = child.get_event(event_id)
            if result is not None:
                # Adjust coordinates to parent space
                child_offset = self._child_offsets[child_id].value
                for coord_key in ("start", "end"):
                    val = result.get(coord_key)
                    if val is not None:
                        result[coord_key] = shift_coordinate(
                            val, child_offset, subtract=False
                        )
                return _held_by(result, child_id)

        return None

    def get_events(
        self,
        temporal_type: Literal["instant", "interval"] | None = None,
        event_type: str | None = None,
        include_children: bool = True,
        min_coord: CoordinateSpec | None = None,
        max_coord: CoordinateSpec | None = None,
        **field_filters: Any,
    ) -> EventData:
        """Filter and retrieve events.

        When ``include_children=True`` (the default), events from all
        children are included with their coordinates adjusted to the
        root timeline's coordinate system. Each of them names the
        timeline holding it in the column ``source_timeline`` (a child, or
        a deeper descendant), at every depth; the timeline's own events
        leave it null. Segment events (internal bookkeeping) are always
        excluded.

        A bound written as a float is compared as a key, with the float
        side of each event's ``start``: on a ``fraction`` axis
        ``min_coord=0.1`` keeps an event starting at ``1/10``, on an
        ``int`` axis ``min_coord=10.5`` leaves out an event starting at 10
        and ``max_coord=10.5`` keeps it, and ``max_coord=math.inf`` keeps
        every event. An exact bound (an ``int``, a ``Fraction``) is compared
        exactly on an ``int`` or ``fraction`` axis, never rounded.

        Args:
            temporal_type: Filter by "instant" or "interval".
            event_type: Filter by event type name.
            include_children: If True (default), include events from all
                children with root-relative coordinates.
            min_coord: Minimum coordinate (inclusive). Can be a number in
                native units or a `Coordinate` with a different unit
                (converted via inverse C-Map).
            max_coord: Maximum coordinate (exclusive). Same conversion
                rules as min_coord.
            **field_filters: Additional field equality filters. Each
                kwarg name is a field name, and the value is the
                required value (or a list of values for OR logic).

        Returns:
            A filtered EventData with all matching events, a table of its
            own: changing it changes nothing on the timeline.

        Raises:
            ValueError: If a bound is NaN, or cannot be resolved onto this
                timeline.

        Examples:
            >>> # Filter by coordinate range
            >>> events = tl.get_events(min_coord=10.0, max_coord=20.0)

            >>> # Filter with a Coordinate in a different unit
            >>> from timetoalign import Coordinate, TimeUnit
            >>> coord = Coordinate(5.0, TimeUnit.seconds)
            >>> events = tl.get_events(min_coord=coord)

            >>> # Arbitrary field filters
            >>> events = tl.get_events(pitch=60)  # Only middle C
            >>> events = tl.get_events(pitch=[60, 62, 64])  # C, D, E
        """
        # Bounds on this timeline's axis, as written: a float compares as a
        # key, an exact bound exactly (``compare_number_column``).
        min_value = None if min_coord is None else self._query_position(min_coord)
        max_value = None if max_coord is None else self._query_position(max_coord)
        # Start with own events, always excluding segment events
        result = self._events
        sources = [result.schema]
        segment_data = result.filter(event_type=SEGMENT_EVENT_TYPE)
        if len(segment_data) > 0:
            result = _null_columns_typed_as(
                self._rebuilt(
                    row for row in result if row.get("event_type") != SEGMENT_EVENT_TYPE
                ),
                sources,
            )

        # Include children's events with offset-adjusted coordinates
        if include_children and self._children:
            all_rows: list[dict[str, Any]] = list(result)
            for child_id, child in self._children.items():
                child_offset = self._child_offsets[child_id].value
                child_events = child.get_events(
                    include_children=True,  # recurse
                )
                sources.append(child_events.schema)
                for event in child_events:
                    row = dict(event)
                    # Adjust coordinates to root space
                    for coord_key in ("start", "end"):
                        val = row.get(coord_key)
                        if val is not None:
                            row[coord_key] = shift_coordinate(
                                val, child_offset, subtract=False
                            )
                    all_rows.append(_held_by(row, child_id))
            result = _null_columns_typed_as(self._rebuilt(all_rows), sources)

        if temporal_type is not None:
            result = result.filter(temporal_type=temporal_type)

        if event_type is not None:
            result = result.filter(event_type=event_type)

        if min_value is not None or max_value is not None:
            result = result.filter(min_coord=min_value, max_coord=max_value)

        # Apply arbitrary field filters
        if field_filters:
            result = result.filter(**field_filters)

        if result is self._events:
            # Never the timeline's own table: rows added to the result would
            # land on the timeline past every check of ``add_events``.
            result = result._derived(result.table)
        return result

    def _rebuilt(self, rows: Iterable[dict[str, Any]]) -> EventData:
        """An EventData of *rows*, read from this timeline's tables or its children's.

        The rows' cells are in lowest terms already (held cells, shifted by
        the exact engine), so the table is held without being reduced again.
        """
        table = EventData._rows_table(list(rows), self._unit, self._number_type)
        return EventData._of_reduced(table, self._unit, self._number_type)

    def _get_exact_coordinate_value(
        self, timeline_id: str, axis: float
    ) -> Fraction | None:
        """Find an exact stored coordinate corresponding to a timestamp axis.

        Timestamp tables use float scalars for efficient lookup. This method
        reconnects an exact event pair to that float at the materialization
        boundary without deriving a fraction from the float.
        """
        if timeline_id == self._id:
            return self._find_exact_event_coordinate(axis)

        child = self._children.get(timeline_id)
        if child is None:
            return None

        child_offset = self._child_offsets[timeline_id].value
        offset_exact = exact_coordinate_value(child_offset)
        if offset_exact is None:
            return None
        for field in ("start", "end", "duration"):
            for value in child._events.column_values(field):
                exact = exact_coordinate_value(value)
                if exact is None or float(exact + offset_exact) != axis:
                    continue
                return exact
        return None

    def _find_exact_event_coordinate(self, axis: float) -> Fraction | None:
        """Find an exact coordinate pair in this timeline's own events."""
        for field in ("start", "end", "duration"):
            for value in self._events.column_values(field):
                exact = exact_coordinate_value(value)
                if exact is not None and float(exact) == axis:
                    return exact
        return None

    def _extract_coord_value(
        self,
        event: dict[str, Any],
        *keys: str,
    ) -> int | float | Fraction | None:
        """Extract coordinate value from event, trying multiple keys.

        Args:
            event: Event dictionary.
            *keys: Keys to try in order (e.g., "start", "instant").

        Returns:
            The coordinate value in this timeline's number type (exact on an
            ``int`` or ``fraction`` axis), or None if not found.
        """
        for key in keys:
            val = event.get(key)
            if val is not None:
                return express_as(coordinate_numeric_value(val), self._number_type)
        return None

    def get_events_at(
        self,
        coord: CoordinateSpec,
        tolerance: float = 0.0,
        include_children: bool = True,
    ) -> dict[str, list[dict[str, Any]]]:
        """Get all events active at a specific coordinate.

        Returns events from this timeline and all children that are
        active (containing or at) the specified coordinate.

        For instant events, an event is "at" the coordinate if its instant
        is within tolerance of the query coordinate.

        For interval events, an event is "active" if the coordinate falls
        within [start, end).

        A *coord* written as a float is compared as a key, with the float
        side of the stored positions: on a ``fraction`` axis
        ``get_events_at(0.1)`` finds an event at ``1/10``, on an ``int``
        axis ``get_events_at(11.5)`` finds an event spanning ``[11, 12)``.
        An exact *coord* (an ``int``, a ``Fraction``) is compared exactly on
        an ``int`` or ``fraction`` axis, never rounded. A float axis
        compares floats whatever the query. Children are decided on this
        axis too: their positions and boundaries are placed here by exact
        offsets and compared with the query, so a position lies on the same
        side of a child's boundary as in ``get_events(min_coord=,
        max_coord=)``, and a boundary two adjoining children share belongs
        to the one starting there.

        Args:
            coord: Coordinate to query (in this timeline's unit).
            tolerance: Tolerance for instant event matching (default 0).
            include_children: If True, include events from children.

        Returns:
            Dict mapping timeline_id to list of events active at that coordinate.
            Child events have coordinates in their local coordinate system.

        Raises:
            ValueError: If *coord* is NaN, or cannot be resolved onto this
                timeline.

        Examples:
            >>> events = score.get_events_at(50.0)
            >>> events["score:1"]  # Events in root at coord 50
            [{"id": "n1", "event_type": "Note", ...}]
            >>> events["measure_5"]  # Events in measure 5
            [...]
        """
        # On this timeline's axis, as written: a float compares as a key.
        position = self._query_position(coord)
        if self._number_type is NumberType.float:
            position = float(position)
        result: dict[str, list[dict[str, Any]]] = {}
        self._collect_events_at(position, (), tolerance, include_children, result)
        return result

    def _collect_events_at(
        self,
        key: int | float | Fraction,
        offsets: tuple[CoordinateValue, ...],
        tolerance: float,
        include_children: bool,
        result: dict[str, list[dict[str, Any]]],
    ) -> None:
        """Add the events active at *key* here, and in each child holding it.

        *key* stays on the axis it was queried on; *offsets* lead from there
        down to this timeline, outermost first. A child is entered where it
        holds the key (``_holds_position``, its exact boundaries placed on
        the queried axis), never by subtracting a float from them.
        """
        local_events = self._get_events_at_local(key, offsets, tolerance)
        if local_events:
            result[self._id] = local_events
        if not include_children:
            return
        for child_id, child in self._children.items():
            path = (*offsets, self._child_offsets[child_id].value)
            if child._holds_position(key, offsets=path):
                child._collect_events_at(key, path, tolerance, True, result)

    def _get_events_at_local(
        self,
        key: int | float | Fraction,
        offsets: tuple[CoordinateValue, ...],
        tolerance: float,
    ) -> list[dict[str, Any]]:
        """This timeline's own events active at *key*, rows as held (local).

        Each position is placed on the queried axis by the exact offsets of
        the path (*offsets*, outermost first), as ``get_events`` shifts a
        child's rows, and compared there: exactly where *key* is exact and
        the axis ``int`` or ``fraction``, as doubles where *key* is a float
        (a float compares as a key) or the axis is ``float``.
        """
        as_float = isinstance(key, float) or self._number_type is NumberType.float
        if as_float:
            key = float(key)

        def placed(value: int | float | Fraction) -> int | float | Fraction:
            return placed_on(value, offsets, as_float=as_float)

        result = []
        for event in _decoded_rows(self.get_events(include_children=False)):
            temporal_type = event.get("temporal_type")

            if temporal_type == "instant":
                instant_val = self._extract_coord_value(event, "instant", "start")
                if instant_val is not None:
                    if abs(placed(instant_val) - key) <= tolerance:
                        result.append(dict(event))

            elif temporal_type == "interval":
                start_val = self._extract_coord_value(event, "start")
                end_val = self._extract_coord_value(event, "end")

                if start_val is not None and end_val is not None:
                    # Left-inclusive, right-exclusive: [start, end)
                    if placed(start_val) <= key < placed(end_val):
                        result.append(dict(event))

        return result

    def _sorted_event_dicts(self) -> list[dict[str, Any]]:
        """Return events as dicts sorted by start coordinate."""
        events_list = _decoded_rows(self.get_events(include_children=False))
        events_list.sort(
            key=lambda e: float(self._extract_coord_value(e, "start", "instant") or 0)
        )
        return events_list
