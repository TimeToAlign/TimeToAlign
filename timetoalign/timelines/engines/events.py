"""Provide event storage operations for timeline instances."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Literal

import pyarrow.compute as pc

from timetoalign.core import Coordinate, CoordinateSpec, CoordinateValue
from timetoalign.core.time import (
    coordinate_numeric_value,
    exact_coordinate_value,
    shift_coordinate,
)
from timetoalign.core.timestamp import _format_coordinate_value
from timetoalign.storage import EventData

SEGMENT_EVENT_TYPE = "Segment"


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
                f"Adding content ending at {_format_coordinate_value(float(required))} "
                f"exceeds timeline length "
                f"{_format_coordinate_value(float(self._length.value))}. "
                f"Timeline is locked. "
                "Use allow_expansion=True to override."
            )

        self._set_length(self._make_coordinate(required))

    def add_events(
        self,
        rows: list[dict[str, Any]],
        allow_expansion: bool = False,
    ) -> None:
        """Add events to the timeline.

        Only ``event_type`` and a coordinate are strictly required per dict.
        Missing fields are filled in automatically:

        - **id**: auto-generated (``e000001``, ``e000002``, ...).
        - **temporal_type**: inferred from keys -- ``"interval"`` when both
          ``start`` and ``end`` (or ``duration``) are present, ``"instant"``
          otherwise.

        Args:
            rows: List of event dictionaries. Required keys:
                - event_type: class name (e.g. ``"Beat"``, ``"Note"``)
                - instant: coordinate (for instant events), **or**
                - start, end: coordinates (for interval events)
            allow_expansion: If True, expand timeline if events exceed length.

        Raises:
            ValueError: If events exceed length and expansion not allowed.
            RuntimeError: If timeline is locked and expansion not allowed.

        Examples:
            >>> tl.add_events([
            ...     {"event_type": "Beat", "instant": 0.0},
            ...     {"event_type": "Note", "start": 0.0, "end": 0.5},
            ... ])
        """
        if not rows:
            return

        # Resolve unit-qualified coordinates into this timeline's native unit
        # before anything measures or stores them. A ``Coordinate`` whose unit
        # differs from the native one (e.g. a measure boundary given in seconds
        # on a samples timeline) is converted through the timeline's C-Maps;
        # bare numbers, struct dicts and native-unit coordinates pass through
        # unchanged. Without this, a derived-unit value would be stored verbatim
        # under the native unit label — a silent mis-unit.
        rows = [self._resolve_row_coordinates(row) for row in rows]

        # Validate and find max coordinate
        max_coord = 0.0
        for row in rows:
            coord = self._get_event_end_coordinate(row)
            max_coord = max(max_coord, coord)

        # Ensure capacity
        self._ensure_capacity(max_coord, allow_expansion)

        # Add events
        self._add_events_unchecked(rows)

    def _resolve_row_coordinates(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return a row whose coordinate fields are in the native unit.

        Each of ``start`` / ``end`` / ``instant`` given as a unit-qualified
        :class:`~timetoalign.core.Coordinate` is resolved through
        :meth:`get_coordinate`, so a value expressed in a derived unit is
        converted into the timeline's native unit via its C-Maps. Raw numbers,
        coordinate struct dicts and durations are left untouched, and the input
        dict is never mutated in place.

        Args:
            row: An event dictionary as passed to :meth:`add_events`.

        Returns:
            The row, copied only when a coordinate needed resolving.
        """
        resolved: dict[str, Any] | None = None
        for key in ("start", "end", "instant"):
            value = row.get(key)
            if isinstance(value, Coordinate):
                if resolved is None:
                    resolved = dict(row)
                resolved[key] = self.get_coordinate_at(value, format="coordinate").value
        return resolved if resolved is not None else row

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

        data_class = type(self._events)
        new_data = data_class.from_dicts(rows, self._unit, self._number_type)
        self._events.extend(new_data)

    def _get_event_end_coordinate(self, row: dict[str, Any]) -> float:
        """Extract the end coordinate from an event dict.

        Args:
            row: Event dictionary. Coordinate fields may be raw floats
                or struct dicts (``{"value": ..., ...}``).

        Returns:
            The end coordinate as float.
        """
        if row.get("instant") is not None:
            return self._coord_to_float(row["instant"])
        if row.get("end") is not None:
            return self._coord_to_float(row["end"])
        if row.get("start") is not None and row.get("duration") is not None:
            return self._coord_to_float(row["start"]) + self._coord_to_float(
                row["duration"]
            )
        if row.get("start") is not None:
            return self._coord_to_float(row["start"])
        return 0.0

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
            >>> event = timeline.get_event("notes:note:000001")
            >>> event["id"]
            'notes:note:000001'
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
                result.setdefault("source_timeline", child_id)
                return result

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
        root timeline's coordinate system. Segment events (internal
        bookkeeping) are always excluded.

        Args:
            temporal_type: Filter by "instant" or "interval".
            event_type: Filter by event type name.
            include_children: If True (default), include events from all
                children with root-relative coordinates.
            min_coord: Minimum coordinate (inclusive). Can be a float in
                native units or a `Coordinate` with a different unit
                (converted via inverse C-Map).
            max_coord: Maximum coordinate (exclusive). Same conversion
                rules as min_coord.
            **field_filters: Additional field equality filters. Each
                kwarg name is a field name, and the value is the
                required value (or a list of values for OR logic).

        Returns:
            A filtered EventData with all matching events.

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
        min_coord_float: float | None = None
        max_coord_float: float | None = None

        if min_coord is not None:
            min_coord_float = float(self._resolve_axis_value(min_coord))

        if max_coord is not None:
            max_coord_float = float(self._resolve_axis_value(max_coord))
        # Start with own events, always excluding segment events
        result = self._events
        segment_data = result.filter(event_type=SEGMENT_EVENT_TYPE)
        if len(segment_data) > 0:
            result = EventData.from_dicts(
                [row for row in result if row.get("event_type") != SEGMENT_EVENT_TYPE],
                self._unit,
                self._number_type,
            )

        # Include children's events with offset-adjusted coordinates
        if include_children and self._children:
            all_rows: list[dict[str, Any]] = list(result)
            for child_id, child in self._children.items():
                child_offset = self._child_offsets[child_id].value
                child_events = child.get_events(
                    include_children=True,  # recurse
                )
                for event in child_events:
                    row = dict(event)
                    # Adjust coordinates to root space
                    for coord_key in ("start", "end"):
                        val = row.get(coord_key)
                        if val is not None:
                            row[coord_key] = shift_coordinate(
                                val, child_offset, subtract=False
                            )
                    # Tag with source timeline for provenance
                    row.setdefault("source_timeline", child_id)
                    all_rows.append(row)
            result = EventData.from_dicts(all_rows, self._unit, self._number_type)

        if temporal_type is not None:
            result = result.filter(temporal_type=temporal_type)

        if event_type is not None:
            result = result.filter(event_type=event_type)

        if min_coord_float is not None or max_coord_float is not None:
            result = result.filter(min_coord=min_coord_float, max_coord=max_coord_float)

        # Apply arbitrary field filters
        if field_filters:
            result = result.filter(**field_filters)

        return result

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
    ) -> float | None:
        """Extract coordinate value from event, trying multiple keys.

        Args:
            event: Event dictionary.
            *keys: Keys to try in order (e.g., "start", "instant").

        Returns:
            The coordinate value as float, or None if not found.
        """
        for key in keys:
            val = event.get(key)
            if val is not None:
                return float(coordinate_numeric_value(val))
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

        Args:
            coord: Coordinate to query (in this timeline's unit).
            tolerance: Tolerance for instant event matching (default 0).
            include_children: If True, include events from children.

        Returns:
            Dict mapping timeline_id to list of events active at that coordinate.
            Child events have coordinates in their local coordinate system.

        Examples:
            >>> events = score.get_events_at(50.0)
            >>> events["score:1"]  # Events in root at coord 50
            [{"id": "n1", "event_type": "Note", ...}]
            >>> events["measure_5"]  # Events in measure 5
            [...]
        """
        coord_val = float(self._resolve_axis_value(coord))
        result: dict[str, list[dict[str, Any]]] = {}

        # Get events from this timeline
        local_events = self._get_events_at_local(coord_val, tolerance)
        if local_events:
            result[self._id] = local_events

        # Check children
        if include_children:
            ts = self.get_timestamp_at(coord_val)

            for child_id in self._children.keys():
                try:
                    child_coord = ts.get_coordinate_for(child_id, format="coordinate")
                except KeyError:
                    continue
                if child_coord.value >= 0:
                    child = self._children[child_id]
                    extent = child._extent
                    if extent is None or child_coord.value <= extent:
                        # Recursively get events in child
                        child_result = child.get_events_at(
                            child_coord.value,
                            tolerance=tolerance,
                            include_children=True,
                        )
                        result.update(child_result)

        return result

    def _get_events_at_local(
        self,
        coord: float,
        tolerance: float,
    ) -> list[dict[str, Any]]:
        """Get events at a coordinate in this timeline (local, no children)."""
        result = []

        for event in _decoded_rows(self.get_events(include_children=False)):
            temporal_type = event.get("temporal_type")

            if temporal_type == "instant":
                instant_val = self._extract_coord_value(event, "instant", "start")
                if instant_val is not None:
                    if abs(instant_val - coord) <= tolerance:
                        result.append(dict(event))

            elif temporal_type == "interval":
                start_val = self._extract_coord_value(event, "start")
                end_val = self._extract_coord_value(event, "end")

                if start_val is not None and end_val is not None:
                    # Left-inclusive, right-exclusive: [start, end)
                    if start_val <= coord < end_val:
                        result.append(dict(event))

        return result

    def _sorted_event_dicts(self) -> list[dict[str, Any]]:
        """Return events as dicts sorted by start coordinate."""
        events_list = _decoded_rows(self.get_events(include_children=False))
        events_list.sort(
            key=lambda e: self._extract_coord_value(e, "start", "instant") or 0.0
        )
        return events_list
