"""Provide timestamp and tabular export operations for timeline instances."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import (
    RATIONAL_STRUCT_TYPE,
    Coordinate,
    CoordinateSpec,
    IdCoordinate,
    Interval,
    NumberType,
    TimeUnit,
    build_number_struct_array,
    combine_number_columns,
    resolve_coordinate_spec,
)
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateInput,
    KeyCollection,
    TableFormat,
    dispatch_retrieval,
    is_key_input,
    reject_dataframe_options,
    resolve_coordinate_collection,
    resolve_key_collection,
    validate_table_format,
)
from timetoalign.core.timestamp import (
    ConversionMapsSpec,
    TimeIntervalStamp,
    TimeStamp,
    TimestampColumn,
    build_timestamp_table,
    timestamp_table_to_dataframe,
)
from timetoalign.maps import ConversionMap

if TYPE_CHECKING:
    from timetoalign.core.enums import ColumnNaming


def _empty_coordinates() -> pa.StructArray:
    """Return an empty coordinate column of the canonical storage shape."""
    return pa.array([], type=RATIONAL_STRUCT_TYPE)


def _coordinate_column(values: Any, number_type: NumberType) -> pa.StructArray:
    """Encode positions as coordinate structs under one declared type."""
    return build_number_struct_array(
        values, number_type=number_type, rounding="round", on_error="raise"
    )


def _unique_sorted_coordinates(arrays: Sequence[pa.Array]) -> pa.StructArray:
    """Concatenate coordinate columns, drop repeats, and sort ascending.

    Ordering and deduplication read the float member, so they stay a single
    vectorized sort with no Python loop over the axis; the cells that survive
    are carried whole, which is what keeps an authored ratio out of the
    round trip through its own double.

    Args:
        arrays: Coordinate-struct columns to merge.

    Returns:
        One sorted column holding each distinct position once.
    """
    non_empty = [array for array in arrays if len(array) > 0]
    if not non_empty:
        return _empty_coordinates()
    combined = pa.concat_arrays(non_empty)
    values = combined.field("value")
    order = pc.sort_indices(values)
    if len(order) < 2:
        return pc.take(combined, order)
    ordered = pc.take(values, order)
    distinct = pc.not_equal(ordered.slice(1), ordered.slice(0, len(ordered) - 1))
    keep = pa.concat_arrays([pa.array([True]), distinct])
    return pc.take(combined, pc.filter(order, keep))


def _positions_within(within: pa.Array) -> pa.Array:
    """Index each kept row into the filtered column, null where it was dropped.

    The companion of ``filter``: taking with these indices scatters a
    computed-on-the-kept-rows column back to full length, nulls and all.
    """
    kept = within.to_numpy(zero_copy_only=False).astype(bool)
    return pa.array(np.cumsum(kept) - 1, type=pa.int64(), mask=~kept)


def _null_where(column: pa.StructArray, mask: pa.Array) -> pa.StructArray:
    """Null out whole coordinate cells wherever *mask* is true."""
    nulled = pc.fill_null(mask, True)
    return pa.StructArray.from_arrays(
        [
            pc.if_else(nulled, pa.scalar(None, type=member.type), column.field(name))
            for name, member in (
                (member.name, member) for member in RATIONAL_STRUCT_TYPE
            )
        ],
        fields=list(RATIONAL_STRUCT_TYPE),
        mask=nulled,
    )


class TabularExportMixin:
    """Provide timestamp and tabular export operations for timeline instances."""

    def get_timestamp_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        unit: TimeUnit | str | None = None,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp:
        """Get a TimeStamp at one coordinate on this timeline.

        This is the primary coordinate resolution API. The TimeStamp provides
        access to all equivalent coordinates across children and C-Map units.
        The axis is written in this timeline's declared ``number_type``, so a
        query of ``9.5`` on a fraction-canonical timeline gives an axis of
        ``Fraction(19, 2)``. Descendants are reached by exact offset
        arithmetic.

        Args:
            at: Coordinate value. Can be:
                - int/float/Fraction: Value in timeline's native unit
                - Coordinate: Must match unit or specify via `unit` param
                - IdCoordinate: May name this timeline or a descendant
            timeline_id: Result-axis validator. ``None`` or this timeline's
                own ID; any other ID raises.
            unit: If provided, interpret ``at`` as being in this unit.
                The coordinate is first converted via inverse C-Map.
            conversion_maps: C-Maps available through the returned stamp.

        Returns:
            TimeStamp object for the resolved coordinate.

        Raises:
            KeyError: If ``timeline_id`` names another timeline.
            ValueError: If unit specified but no inverse C-Map available.

        Examples:
            >>> from timetoalign.maps import TableMap
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=960, unit=TimeUnit.ticks, uid="timeline:1")
            >>> child_a = Timeline(length=200, unit=TimeUnit.ticks, uid="child_a")
            >>> parent.add_child(child_a, offset=100)
            >>> tempo_map = TableMap(
            ...     x_values=[0, 960], y_values=[0.0, 2.0],
            ...     source_unit=TimeUnit.ticks, target_unit=TimeUnit.seconds,
            ... )
            >>> parent.add_conversion_map(tempo_map)
            >>> ts = parent.get_timestamp_at(250)
            >>> ts.get_coordinate_for("child_a", format="float")
            150.0

            >>> # Query with unit conversion
            >>> ts = parent.get_timestamp_at(1.5, unit=TimeUnit.seconds)
            >>> ts.axis  # Converted from seconds to timeline's native unit (ticks)
            IdCoordinate(720, ticks, 'timeline:1')
        """
        if timeline_id is not None and timeline_id != self._id:
            raise KeyError(
                f"Unknown result timeline ID {timeline_id!r} on timeline {self._id!r}"
            )
        coord = at
        if unit is None:
            native_coord = self.get_coordinate_at(coord, format="coordinate")
        else:
            target_unit = TimeUnit(unit) if isinstance(unit, str) else unit
            decomposed = resolve_coordinate_spec(coord)
            if decomposed.timeline_id is None:
                qualified_coord: CoordinateSpec = Coordinate(
                    decomposed.value, target_unit
                )
            else:
                qualified_coord = IdCoordinate(
                    decomposed.value,
                    target_unit,
                    decomposed.timeline_id,
                )
            native_coord = self.get_coordinate_at(qualified_coord, format="coordinate")

        if not isinstance(native_coord, Coordinate):
            raise TypeError("Timeline coordinate resolution did not return Coordinate")
        coordinates: dict[str, Coordinate] = {self._id: native_coord}
        descendants = getattr(self, "_get_descendant_timeline_ids", lambda: [])()
        child_getter = getattr(self, "_get_child_coordinate", None)
        for timeline_id in descendants:
            if timeline_id == self._id or child_getter is None:
                continue
            value = child_getter(timeline_id, native_coord.value)
            if value is None:
                continue
            unit_for = self._get_unit_for_timeline(timeline_id)
            number_type_for = self._get_number_type_for_timeline(timeline_id)
            if unit_for is None:
                continue
            declared = number_type_for or unit_for.default_number_type
            coordinates[timeline_id] = Coordinate(value, unit_for, number_type=declared)

        return TimeStamp(
            coordinates=coordinates,
            source=self,
            source_id=self._id,
            conversion_maps=conversion_maps,
        )

    def get_timestamps_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        unit: TimeUnit | str | None = None,
        conversion_maps: ConversionMapsSpec = True,
    ) -> list[TimeStamp]:
        """Get TimeStamps at a collection of coordinates, in input order.

        Args:
            at: Coordinate positions to resolve.
            timeline_id: Result-axis validator.
            unit: Optional unit in which every position is expressed.
            conversion_maps: C-Maps available through the returned stamps.

        Returns:
            One TimeStamp per input coordinate; an empty collection gives
            an empty list.

        Raises:
            TypeError: If ``at`` is not an accepted coordinate collection.
        """
        stamps, _ = resolve_coordinate_collection(
            at,
            lambda value: self.get_timestamp_at(
                value,
                timeline_id,
                unit=unit,
                conversion_maps=conversion_maps,
            ),
        )
        return stamps

    def get_interval_stamp(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        unit: TimeUnit | str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeIntervalStamp:
        """Get a TimeIntervalStamp for a coordinate range.

        Args:
            start: Start coordinate.
            end: End coordinate.
            unit: If provided, interpret both coords as being in this unit.
            conversion_maps: C-Maps available through the returned stamps.

        Returns:
            TimeIntervalStamp with start and end TimeStamps.

        Examples:
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=20, unit=TimeUnit.seconds, uid="timeline:1")
            >>> child = Timeline(length=15, unit=TimeUnit.seconds, uid="cpt_2")
            >>> parent.add_child(child, offset=0)
            >>> interval = parent.get_interval_stamp(0.0, 10.0)
            >>> interval.duration
            Duration(10.0, seconds)
            >>> interval.get_interval("cpt_2")
            Interval(start=Coordinate(0.0, seconds), end=Coordinate(10.0, seconds))
        """
        start_stamp = self.get_timestamp_at(
            start, unit=unit, conversion_maps=conversion_maps
        )
        end_stamp = self.get_timestamp_at(
            end, unit=unit, conversion_maps=conversion_maps
        )
        common = [
            timeline_id
            for timeline_id in start_stamp.coordinates
            if timeline_id in end_stamp.coordinates
        ]
        intervals = {
            timeline_id: Interval(
                start=start_stamp.coordinates[timeline_id],
                end=end_stamp.coordinates[timeline_id],
            )
            for timeline_id in common
        }
        return TimeIntervalStamp(
            intervals=intervals,
            source_id=self._id,
            source=self,
            is_interpolated=(start_stamp.is_interpolated or end_stamp.is_interpolated),
        )

    def get_timestamp_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp | TimeIntervalStamp:
        """Get the timestamp for a specific event by its ID.

        Returns a TimeStamp for instant events, or a TimeIntervalStamp for
        interval events. Searches recursively through children if the event
        is not found on this timeline directly.

        Args:
            key: The event identifier to look up.
            timeline_id: Result-axis validator.
            conversion_maps: C-Maps available through the returned stamp or stamps.

        Returns:
            TimeStamp for instant events, TimeIntervalStamp for interval events.

        Raises:
            KeyError: If no event with the given ID exists, or if
                ``timeline_id`` names another timeline.

        Examples:
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=20, unit=TimeUnit.seconds, uid="timeline:1")
            >>> child = Timeline(length=10, unit=TimeUnit.seconds, uid="clt_1")
            >>> parent.add_child(child, offset=5)
            >>> parent.add_events([{"event_type": "Note", "instant": 5.0}])
            >>> child.add_events(
            ...     [{"id": "clt_1:note:000001", "event_type": "Note", "start": 0.0, "end": 2.5}]
            ... )
            >>> ts = parent.get_timestamp_for("note:000001")
            >>> ts.axis  # For instant events
            IdCoordinate(5.0, seconds, 'timeline:1')

            >>> ts = parent.get_timestamp_for("clt_1:note:000001")
            >>> ts.start.axis  # For interval events
            IdCoordinate(5.0, seconds, 'timeline:1')
            >>> ts.end.axis
            IdCoordinate(7.5, seconds, 'timeline:1')

        See Also:
            get_timestamp_at: Get timestamp by coordinate.
            get_event: Get the raw event dict by ID.
        """
        if not isinstance(key, str):
            raise TypeError("get_timestamp_for requires an event-ID string")
        if timeline_id is not None and timeline_id != self._id:
            raise KeyError(
                f"Unknown result timeline ID {timeline_id!r} on timeline {self._id!r}"
            )
        events = self.get_events(include_children=True, id=key)
        if len(events) == 0:
            raise KeyError(
                f"Event {key!r} not found on timeline {self._id!r} "
                f"({self.n_events} events)"
            )

        # The stored cell already carries the event's position in the
        # representation its field declares; taking float() of it here threw
        # away an exact ratio the store had kept, so an event authored at
        # 5/3 quarters came back as the dyadic of 1.6666666666666667.
        start_coord = events.column_values("start")[0]
        if start_coord is None:
            raise ValueError(f"Event {key!r} has no 'start' coordinate")

        # Check if this is an interval event
        end_coord = events.column_values("end")[0]
        if end_coord is not None:
            return self.get_interval_stamp(
                start_coord,
                end_coord,
                conversion_maps=conversion_maps,
            )

        return self.get_timestamp_at(start_coord, conversion_maps=conversion_maps)

    def get_timestamps_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> list[TimeStamp | TimeIntervalStamp]:
        """Get timestamps for a collection of event IDs, in input order.

        Args:
            keys: Event identifiers to look up.
            timeline_id: Result-axis validator.
            conversion_maps: C-Maps available through the returned stamps.

        Returns:
            One stamp per key — a TimeIntervalStamp wherever the event is an
            interval — and an empty list for an empty collection.

        Raises:
            KeyError: If any event ID is not found.

        Examples:
            >>> from timetoalign.core import TimeUnit
            >>> from timetoalign.timelines import Timeline
            >>> timeline = Timeline(length=60, unit=TimeUnit.seconds, uid="cpt_1")
            >>> timeline.add_events(
            ...     [{"id": "beat:1", "instant": 0.0}, {"id": "beat:2", "instant": 55.0}]
            ... )
            >>> stamps = timeline.get_timestamps_for(["beat:1", "beat:2"])
            >>> [stamp.get_coordinate_for("cpt_1", format="float") for stamp in stamps]
            [0.0, 55.0]

        See Also:
            get_timestamp_for: Get a single event's timestamp.
            get_timestamp_table: Tabulate many events at once.
        """
        stamps, _ = resolve_key_collection(
            keys,
            lambda key: self.get_timestamp_for(
                key, timeline_id, conversion_maps=conversion_maps
            ),
        )
        return stamps

    def get_timestamp(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        unit: TimeUnit | str | None = None,
        conversion_maps: ConversionMapsSpec = True,
    ) -> (
        TimeStamp
        | TimeIntervalStamp
        | list[TimeStamp]
        | list[TimeStamp | TimeIntervalStamp]
    ):
        """Dispatch a positional or event-key timestamp query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Result-axis validator.
            unit: Optional unit for a positional query.
            conversion_maps: C-Maps available through the returned stamps.

        Returns:
            The selected precise-getter result.

        Raises:
            TypeError: If ``at`` mixes keys and coordinates or is an
                unsupported runtime form.
        """
        return dispatch_retrieval(
            self,
            "get_timestamp",
            "get_timestamps",
            at,
            timeline_id,
            position_options={"unit": unit},
            conversion_maps=conversion_maps,
        )

    def _extract_event_coordinates(
        self,
        event_filter: dict[str, Any] | pc.Expression | None = None,
    ) -> pa.StructArray:
        """Extract all unique event coordinates as a sorted coordinate column.

        Uses PyArrow compute to efficiently extract coordinates from the
        EventData table without Python iteration.

        The stored cells are carried whole rather than reduced to their float
        member. A cell keeps its number twice, and the exact side is the only
        record that a position was authored as ``5/3`` rather than as the
        double nearest to it; re-deriving a ratio from that double afterwards
        answers with its exact dyadic, which is a different number.

        Args:
            event_filter: Optional filter to apply before extracting
                coordinates. Can be a dict (passed to ``EventData.filter()``)
                or a ``pc.Expression`` (passed to ``EventData.where()``).

        Returns:
            Sorted, deduplicated coordinate structs. Empty when there are no
            events.

        Notes:
            - Takes ``start`` from all events
            - Takes ``end`` from interval events (drops nulls)
            - Deduplicates and sorts the result
        """
        # Apply filter if provided
        if event_filter is not None:
            if isinstance(event_filter, pc.Expression):
                filtered_store = self._events.where(event_filter)
            else:
                filtered_store = self._events.filter(**event_filter)
            table = filtered_store.table
        else:
            table = self._events.table

        if table.num_rows == 0:
            return _empty_coordinates()

        starts = table.column("start").combine_chunks()
        ends = table.column("end").combine_chunks()
        return _unique_sorted_coordinates([starts, ends.filter(ends.is_valid())])

    def _collect_all_coordinates(
        self,
        recursion_limit: int | None = None,
        offset: int | float | Fraction = 0,
        event_filter: dict[str, Any] | pc.Expression | None = None,
        number_type: NumberType | None = None,
    ) -> pa.StructArray:
        """Collect coordinates from this timeline and all children.

        Recursively collects event coordinates, applying cumulative offset
        to convert to root-relative coordinates.

        Args:
            recursion_limit: Maximum depth for child traversal. None = unlimited.
            offset: Cumulative offset from root timeline (internal use).
            event_filter: Optional filter applied to each timeline's events.
                Can be a dict (passed to EventData.filter()) or a pc.Expression
                (passed to EventData.where()). The same filter is applied to
                all timelines in the hierarchy.
            number_type: Representation of the axis being built (internal
                use); defaults to this timeline's own.

        Returns:
            Unique, sorted, root-relative coordinate structs.
        """
        declared = self._number_type if number_type is None else number_type

        # Get this timeline's coordinates (with optional filter)
        local_coords = self._extract_event_coordinates(event_filter)

        # Shift onto the root axis. The offset is added in the axis's declared
        # representation, so an exact child position lands on an exact parent.
        if offset != 0 and len(local_coords) > 0:
            local_coords = combine_number_columns(local_coords, offset, "add", declared)

        arrays = [local_coords]

        # Recurse into children
        if recursion_limit is None or recursion_limit > 0:
            next_limit = None if recursion_limit is None else recursion_limit - 1
            for child_id, child in self._children.items():
                child_offset = self._child_offsets[child_id].value
                child_coords = child._collect_all_coordinates(
                    recursion_limit=next_limit,
                    offset=offset + child_offset,
                    event_filter=event_filter,
                    number_type=declared,
                )
                if len(child_coords) > 0:
                    arrays.append(child_coords)

        if len(arrays) == 1:
            return arrays[0]
        return _unique_sorted_coordinates(arrays)

    def _collect_boundary_coordinates(
        self,
        recursion_limit: int | None = None,
        offset: int | float | Fraction = 0,
        number_type: NumberType | None = None,
    ) -> pa.StructArray:
        """Collect timeline boundary coordinates (start=0, end=length).

        Recursively collects boundary coordinates from this timeline
        and all children, applying cumulative offsets.

        Args:
            recursion_limit: Maximum depth for child traversal. None = unlimited.
            offset: Cumulative offset from root timeline (internal use).
            number_type: Representation of the axis being built (internal
                use); defaults to this timeline's own.

        Returns:
            Unique, sorted, root-relative boundary coordinate structs.
        """
        declared = self._number_type if number_type is None else number_type

        # This timeline's boundaries, computed in Python so an exact length on
        # an exact offset stays exact.
        arrays = [_coordinate_column([offset, offset + self._length.value], declared)]

        # Recurse into children
        if recursion_limit is None or recursion_limit > 0:
            next_limit = None if recursion_limit is None else recursion_limit - 1
            for child_id, child in self._children.items():
                child_offset = self._child_offsets[child_id].value
                child_bounds = child._collect_boundary_coordinates(
                    recursion_limit=next_limit,
                    offset=offset + child_offset,
                    number_type=declared,
                )
                if len(child_bounds) > 0:
                    arrays.append(child_bounds)

        if len(arrays) == 1:
            return arrays[0]
        return _unique_sorted_coordinates(arrays)

    def _compute_local_coordinates(
        self,
        root_coords: pa.StructArray,
        offset: int | float | Fraction = 0,
    ) -> pa.StructArray:
        """Compute local coordinates from root coordinates.

        Offset subtraction with bounds checking. Coordinates outside
        ``[0, length]`` are replaced with null.

        The subtraction runs in this timeline's declared representation, so a
        child of an exact parent reports exact local positions; only the
        bounds comparison drops to the float member, where a comparison is all
        that is asked of it.

        Args:
            root_coords: Root-relative coordinate structs.
            offset: This timeline's offset from root.

        Returns:
            Local coordinate structs, null for out-of-bounds.
        """
        if len(root_coords) == 0:
            return _empty_coordinates()

        # Bound first, on the float member. The length scalar must reach the
        # kernel as a float: on a quarters/beats axis its value is a Fraction,
        # which the PyArrow compute kernel rejects.
        shifted = pc.subtract(root_coords.field("value"), float(offset))
        within = pc.and_(
            pc.greater_equal(shifted, 0.0),
            pc.less_equal(shifted, float(self._length.value)),
        )

        # At offset zero the local coordinate IS the root coordinate, so the
        # cells are reused rather than recomputed.
        if offset == 0:
            return _null_where(root_coords, pc.invert(within))

        # Only the rows this timeline actually covers are shifted. The exact
        # lane of the arithmetic reads rows in Python, so a child spanning a
        # tenth of the axis must not pay for the whole of it.
        kept = pc.filter(root_coords, within)
        local = combine_number_columns(kept, offset, "subtract", self._number_type)
        return pc.take(local, _positions_within(within))

    def _resolve_conversion_maps(
        self, spec: ConversionMapsSpec
    ) -> list[ConversionMap[Any]]:
        """Resolve a flexible conversion_maps specification to a list of C-Maps.

        Supports multiple input formats for convenience:
        - True: Return all attached conversion maps
        - False/None: Return empty list
        - str: Look up by map ID, or find map by target unit name
        - TimeUnit: Find map by target unit
        - ConversionMap: Return as single-element list
        - Iterable: Resolve each element recursively

        Args:
            spec: Flexible specification for which C-Maps to include.

        Returns:
            List of resolved ConversionMap objects.

        Raises:
            KeyError: If a string ID doesn't match any attached map.
            ValueError: If a TimeUnit doesn't match any attached map's target.

        Examples:
            >>> tl._resolve_conversion_maps(True)  # All maps
            >>> tl._resolve_conversion_maps("inches")  # Single map by ID/unit
            >>> tl._resolve_conversion_maps(["inches", "cm"])  # Multiple
            >>> tl._resolve_conversion_maps(TimeUnit.seconds)  # By unit enum
        """
        if spec is None or spec is False:
            return []

        if spec is True:
            # Return all attached conversion maps
            return list(self._conversion_maps.values())

        # Single string: could be map ID or target unit name
        if isinstance(spec, str):
            # First try exact ID match
            if spec in self._conversion_maps:
                return [self._conversion_maps[spec]]
            # Try as target unit name
            try:
                unit = TimeUnit(spec)
                cmap = self.get_conversion_map(unit)
                if cmap is not None:
                    return [cmap]
            except ValueError:
                pass
            raise KeyError(
                f"No conversion map with ID '{spec}' or target unit '{spec}'. "
                f"Available: {list(self._conversion_maps.keys())}"
            )

        # TimeUnit: find by target unit
        if isinstance(spec, TimeUnit):
            cmap = self.get_conversion_map(spec)
            if cmap is not None:
                return [cmap]
            raise ValueError(
                f"No conversion map with target unit '{spec}'. "
                f"Available: {list(self._conversion_maps.keys())}"
            )

        # Single ConversionMap object
        if isinstance(spec, ConversionMap):
            return [spec]

        # Iterable: resolve each element
        resolved: list[ConversionMap[Any]] = []
        for item in spec:
            resolved.extend(self._resolve_conversion_maps(item))
        return resolved

    def _build_timestamp_table(
        self,
        axis: pa.StructArray,
        conversion_maps: list[ConversionMap[Any]] | None = None,
        recursion_limit: int | None = None,
    ) -> pa.Table:
        """Build a timestamp table from axis coordinates.

        Constructs a PyArrow table with:
        - One field per timeline (this one and its children) with local
          coordinates
        - One field per C-Map with converted values

        Every column names the timeline or C-Map it belongs to and nothing
        else. This timeline's own column carries the axis positions, so there
        is no separate ``axis`` field duplicating them.

        Every column is a coordinate struct carrying the number twice, so a
        column's exact ratio survives the table and a parquet round trip; each
        field's metadata names its unit, its declared number type, and either
        the timeline it belongs to or the C-Map that produced it.

        Args:
            axis: Root-relative coordinate structs (the positions to tabulate).
            conversion_maps: Optional list of C-Maps to include as fields.
            recursion_limit: Maximum depth for child traversal. None = unlimited.

        Returns:
            PyArrow table with timestamp data and field-level unit metadata.
        """
        columns: list[TimestampColumn] = [
            TimestampColumn(
                name=self._id,
                values=self._compute_local_coordinates(axis, offset=0),
                unit=self._unit,
                number_type=self._number_type,
                timeline_id=self._id,
            ),
        ]

        for child_offset, child in self.iter_children(
            recursion_limit=recursion_limit,
            include_self=False,
        ):
            columns.append(
                TimestampColumn(
                    name=child.id,
                    values=child._compute_local_coordinates(
                        axis, offset=child_offset.value
                    ),
                    unit=child.unit,
                    number_type=child.number_type,
                    timeline_id=child.id,
                )
            )

        # C-Maps work on NumPy arrays; we convert PyArrow <-> NumPy at the boundary.
        if conversion_maps:
            # Allow copy for arrays with nulls (zero_copy_only=False)
            axis_np = axis.field("value").to_numpy(zero_copy_only=False)
            for cmap in conversion_maps:
                target_unit = getattr(cmap, "target_unit", None)
                if target_unit is None:
                    # Numeric unit conversions only. A structured map such as
                    # MetricalPositionMap answers {"mc": ..., "beat": ...},
                    # which has no column type here.
                    continue
                columns.append(
                    TimestampColumn(
                        name=cmap.name,
                        values=pa.array(cmap.convert_array(axis_np)),
                        unit=target_unit,
                        number_type=cmap.output_number_type,
                        metadata={"cmap_id": cmap.id},
                    )
                )

        return build_timestamp_table(columns)

    def _resolve_coordinate_spec(
        self,
        coordinates: CoordinateSpec | Sequence[CoordinateSpec],
    ) -> pa.StructArray:
        """Resolve CoordinateSpec to axis coordinates.

        Handles IdCoordinate objects by automatically applying child offsets.
        This enables the dead-simple pattern:

            child_coords = [IdCoordinate(v, unit, "child_id") for v in values]
            table = parent.get_timestamp_table(child_coords)

        When an IdCoordinate's timeline_id matches a child of this timeline,
        the coordinate is automatically converted to parent coordinates by
        adding the child's offset.

        Args:
            coordinates: Single CoordinateSpec or sequence of CoordinateSpec.
                - int/float/Fraction: Used directly as axis coordinate
                - Coordinate: Value extracted
                - IdCoordinate: Child offset applied if timeline_id matches a child
                - Sequence of the above: Each element processed individually

        Returns:
            The positions as coordinate structs on this timeline's axis, in
            the representation it declares.

        Examples:
            >>> from timetoalign.core import IdCoordinate, TimeUnit
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=5000, unit=TimeUnit.pixels, uid="dgt:1")
            >>> holes = Timeline(length=2000, unit=TimeUnit.pixels, uid="dgt_holes")
            >>> parent.add_child(holes, offset=500)
            >>> child_coord = IdCoordinate(1000, TimeUnit.pixels, "dgt_holes")
            >>> axis = parent._resolve_coordinate_spec([child_coord])
            >>> axis.field("numerator").to_pylist()  # 1000 + the child's offset
            [1500]
        """

        # A column already in storage shape is the axis; anything else is
        # encoded into it under this timeline's declared representation.
        if isinstance(coordinates, (pa.Array, pa.ChunkedArray)):
            if isinstance(coordinates, pa.ChunkedArray):
                coordinates = coordinates.combine_chunks()
            if coordinates.type == RATIONAL_STRUCT_TYPE:
                return coordinates
            return _coordinate_column(coordinates, self._number_type)
        if isinstance(coordinates, np.ndarray):
            return _coordinate_column(coordinates, self._number_type)

        # Single coordinate specification
        if isinstance(coordinates, (int, float, Fraction, Coordinate, IdCoordinate)):
            coordinates = [coordinates]

        # Process list of coordinates
        resolved: list[int | float | Fraction] = []
        for coord in coordinates:
            resolve_coordinate_spec(coord)
            resolved.append(self._resolve_axis_value(coord))

        return _coordinate_column(resolved, self._number_type)

    def get_timestamp_table(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
        recursion_limit: int | None = None,
        include_events: bool = True,
        include_boundaries: bool = False,
        format: TableFormat = "table",
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
        units: bool | None = None,
        include_ids: bool | None = None,
    ) -> pa.Table | pd.DataFrame:
        """Generate a timestamp table for this timeline hierarchy.

        A Timestamp is a cross-section through the timeline hierarchy showing
        synchronous coordinates. This method computes local coordinates for
        each timeline in the hierarchy at each axis coordinate.

        Supports IdCoordinate for automatic child offset resolution: naming a
        child in the query places the position on the parent axis for you.

        Args:
            at: Explicit positions to use as the axis, or event IDs (one row
                per event, in the order given). If None, coordinates are
                extracted from events (and optionally boundaries). Accepts
                IdCoordinate objects - if timeline_id matches a child, the
                offset is automatically applied.
            conversion_maps: C-Maps to include as fields. Flexible input:
                - True: Include all attached conversion maps
                - str: Map ID or target unit name (e.g., "inches", "seconds")
                - TimeUnit: Find map by target unit enum
                - ConversionMap: Include the specific map
                - list: Mix of the above
                - None/False: No conversion maps
            recursion_limit: Maximum depth for child traversal. None = unlimited.
            include_events: If True and ``at`` is None, extract from events.
            include_boundaries: If True, include timeline boundary coordinates.
            format: ``"table"`` (default) for a PyArrow table, ``"dataframe"``
                for a pandas DataFrame.
            fields: How to name the DataFrame fields (``format="dataframe"``):
                - None or ColumnNaming.name (default): Use timeline/cmap name
                - ColumnNaming.id: Use timeline/cmap id
                - Callable: Function taking (name, metadata_dict) -> new_name
                - list[str]: Explicit field names
            units: If True (the DataFrame default), append units to field
                names like "name (unit)".
            include_ids: If True (the DataFrame default), add event IDs as the
                DataFrame index when coordinates are collected from events.

        Returns:
            With ``format="table"``, a PyArrow Table whose fields are
            coordinate structs, one per timeline plus one per C-Map:
                - {timeline_id}: local coordinate per timeline, this one
                  first (carrying the queried positions themselves), then
                  each child; nullable where a timeline does not reach a row
                - {cmap_name}: converted value per C-Map

            There is no separate ``axis`` field: this timeline's own column
            already holds those positions, and a duplicate of it would be a
            second name for one thing. (Unrelated to ``Stamp.axis``, which is
            a stamp's typed source coordinate and keeps its name.)

            Each field includes metadata:
                - unit: TimeUnit.value string (e.g., "seconds", "pixels")
                - number_type: the representation the column declares
                - timeline_id: Timeline ID (for timeline fields)
                - cmap_id: C-Map ID (for C-Map fields)

            Metadata lives in the versioned blob, so read it through
            ``field_metadata(table.schema.field(col_name))`` rather than by
            indexing ``.metadata`` with a bare key.

            With ``format="dataframe"``, the same data as a pandas DataFrame
            with one scalar per cell, written in the representation each
            column declares.

        Raises:
            ValueError: On an unknown ``format``, or when a DataFrame-shaping
                option is supplied for an Arrow result.

        Examples:
            >>> from timetoalign.core import IdCoordinate, TimeUnit
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=60, unit=TimeUnit.seconds, uid="cpt_1")
            >>> child = Timeline(length=10, unit=TimeUnit.seconds, uid="cpt_2")
            >>> parent.add_child(child, offset=50)
            >>> parent.add_events([{"id": "beat:1", "instant": 0.0}])
            >>> parent.get_timestamp_table().column_names
            ['cpt_1', 'cpt_2']

            Every column names the timeline it belongs to; metadata lives in
            the versioned blob:

            >>> from timetoalign.core.fields import field_metadata
            >>> field_metadata(parent.get_timestamp_table().schema.field("cpt_1"))["unit"]
            'seconds'

            A child position is placed on the parent axis automatically:

            >>> coords = [IdCoordinate(v, TimeUnit.seconds, "cpt_2") for v in (0.0, 5.0)]
            >>> parent.get_timestamp_table(coords, format="dataframe", units=False)
               cpt_1  cpt_2
            0   50.0    0.0
            1   55.0    5.0
        """
        validate_table_format(format)
        reject_dataframe_options(
            format, fields=fields, units=units, include_ids=include_ids
        )

        coordinates: CoordinateSpec | Sequence[CoordinateSpec] | None = at
        event_keys: list[str] | None = None
        # An Arrow array is the ready-made numeric lane and never names keys;
        # it is also not one of the accepted public collection forms.
        if (
            at is not None
            and not isinstance(at, (pa.Array, pa.ChunkedArray))
            and is_key_input(at)
        ):
            event_keys = [at] if isinstance(at, str) else list(at)
            coordinates = [
                self.get_coordinate_for(key, format="coordinate") for key in event_keys
            ]

        # Resolve coordinates (handles IdCoordinate with auto child offset)
        if coordinates is not None:
            axis = self._resolve_coordinate_spec(coordinates)
        elif include_events:
            # Extract from events
            event_coords = self._collect_all_coordinates(
                recursion_limit=recursion_limit
            )
            if include_boundaries:
                boundary_coords = self._collect_boundary_coordinates(
                    recursion_limit=recursion_limit
                )
                axis = _unique_sorted_coordinates([event_coords, boundary_coords])
            else:
                axis = event_coords
        else:
            # Boundaries only
            axis = self._collect_boundary_coordinates(recursion_limit=recursion_limit)

        # Resolve C-Map references using flexible helper
        resolved_maps = self._resolve_conversion_maps(conversion_maps)

        table = self._build_timestamp_table(
            axis=axis,
            conversion_maps=resolved_maps if resolved_maps else None,
            recursion_limit=recursion_limit,
        )
        if format == "table":
            return table

        df = timestamp_table_to_dataframe(
            table=table,
            fields=fields,
            units=True if units is None else units,
        )
        if event_keys is not None:
            df.index = pd.Index(event_keys, name="id")
            return df
        if (include_ids is not False) and include_events and at is None:
            all_events = self.get_events(include_children=True)
            coord_to_ids: dict[float, str] = {}
            for event_id, start in zip(
                all_events.column_values("id"),
                all_events.column_values("start"),
                strict=True,
            ):
                if start is None:
                    continue
                coord_val = float(start)
                if coord_val not in coord_to_ids:
                    coord_to_ids[coord_val] = event_id or ""

            # The positions to match against are this timeline's own column,
            # found by name rather than by ordinal. If it is missing the table
            # is malformed and that is an error -- never a reason to fall back
            # to another column, which would silently index the frame by some
            # other timeline's coordinates.
            if self._id not in table.column_names:
                raise KeyError(
                    f"Timestamp table has no column for timeline {self._id!r}. "
                    f"Available columns: {table.column_names}"
                )
            own_column = df.columns[table.column_names.index(self._id)]
            ids = []
            for value in df[own_column]:
                float_value = float(value) if pd.notna(value) else None
                ids.append(coord_to_ids.get(float_value, ""))
            df.index = pd.Index(ids, name="id")
        return df

    def get_boundary_table(
        self,
        conversion_maps: ConversionMapsSpec = True,
        recursion_limit: int | None = None,
    ) -> pa.Table:
        """Get timestamps for timeline boundaries only.

        Returns a timestamp table containing only start (0) and end (length)
        coordinates for this timeline and all children.

        Args:
            conversion_maps: C-Maps to include as fields (see get_timestamp_table).
            recursion_limit: Maximum depth for child traversal.

        Returns:
            PyArrow Table with boundary timestamps.

        Examples:
            >>> from timetoalign.core import TimeUnit
            >>> from timetoalign.timelines import Timeline
            >>> parent = Timeline(length=60, unit=TimeUnit.seconds, uid="cpt_1")
            >>> child = Timeline(length=10, unit=TimeUnit.seconds, uid="cpt_2")
            >>> parent.add_child(child, offset=50)
            >>> parent.get_boundary_table().column_names
            ['cpt_1', 'cpt_2']

            Cells are coordinate structs, so read them through the frame lane
            rather than ``to_pandas()`` — the latter hands back one dict per
            cell:

            >>> parent.get_timestamp_table(
            ...     parent._collect_boundary_coordinates(),
            ...     include_events=False,
            ...     format="dataframe",
            ...     units=False,
            ... )
               cpt_1  cpt_2
            0    0.0    NaN
            1   50.0    0.0
            2   60.0   10.0
        """
        table = self.get_timestamp_table(
            self._collect_boundary_coordinates(recursion_limit=recursion_limit),
            conversion_maps=conversion_maps,
            recursion_limit=recursion_limit,
            include_events=False,
            include_boundaries=False,  # Already included in the positions
        )
        assert isinstance(table, pa.Table)
        return table

    def export_to_csv(
        self,
        filepath: str,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection | None = None,
        conversion_maps: ConversionMapsSpec = True,
        recursion_limit: int | None = None,
        include_events: bool = True,
        include_boundaries: bool = False,
        *,
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
        units: bool = True,
        sep: str = ",",
        header: bool = True,
        index: bool = False,
    ) -> int:
        """Export timeline data to a CSV file.

        This is a convenience method that generates a timestamp DataFrame and
        writes it to a CSV file. For more control over the output, use
        ``get_timestamp_table(format="dataframe")`` and save manually.

        Args:
            filepath: Output CSV file path.
            at: Explicit positions or event IDs to use as the axis.
            conversion_maps: C-Maps to include as additional fields. Defaults to True (all).
            recursion_limit: Maximum depth for child traversal.
            include_events: If True and ``at`` is None, extract from events.
            include_boundaries: If True, include timeline boundary coordinates.
            fields: How to name the DataFrame fields (see get_timestamp_table).
            units: If True (default), append units to field names.
            sep: Field separator. Default "," (comma).
            header: If True (default), write field headers.
            index: If True, write row indices. Default False.

        Returns:
            Number of rows written.

        Examples:
            >>> import tempfile
            >>> from pathlib import Path
            >>> from timetoalign.core import TimeUnit
            >>> from timetoalign.timelines import Timeline
            >>> timeline = Timeline(length=60, unit=TimeUnit.seconds, uid="cpt_1")
            >>> timeline.add_events(
            ...     [{"id": "beat:1", "instant": 0.0}, {"id": "beat:2", "instant": 55.0}]
            ... )
            >>> out = Path(tempfile.mkdtemp())
            >>> timeline.export_to_csv(str(out / "timestamps.csv"))
            2

            >>> # Tab-separated, no header
            >>> timeline.export_to_csv(str(out / "data.tsv"), sep="\\t", header=False)
            2
        """
        df = self.get_timestamp_table(
            at,
            conversion_maps=conversion_maps,
            recursion_limit=recursion_limit,
            include_events=include_events,
            include_boundaries=include_boundaries,
            format="dataframe",
            fields=fields,
            units=units,
        )
        assert isinstance(df, pd.DataFrame)
        df.to_csv(filepath, sep=sep, header=header, index=index)
        return len(df)
