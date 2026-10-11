"""Provide segment line and flow map operations for timeline instances."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from timetoalign.core import CoordinateSpec
from timetoalign.core.time import coordinate_numeric_value, shift_coordinate
from timetoalign.core.timestamp import _format_coordinate_value

if TYPE_CHECKING:
    from ..base import Timeline
    from ..flow import FlowMap
    from ..types import SegmentLine


class SegmentsMixin:
    """Provide segment line and flow map operations for timeline instances."""

    _flow_maps: dict[str, "FlowMap"]

    def create_segment_line(
        self,
        boundaries: Sequence[CoordinateSpec],
        *,
        copy_events: bool = True,
    ) -> "SegmentLine":
        """Create a SegmentLine by segmenting at boundary coordinates.

        Given k+1 sorted coordinates, produces a new SegmentLine with k
        contiguous segments. Each segment's class matches self's class.

        Does NOT modify self. Returns a new independent SegmentLine.

        Args:
            boundaries: k+1 monotonically increasing coordinates.
            copy_events: Copy events into their respective segments.

        Returns:
            A new SegmentLine with k segments.

        Raises:
            ValueError: If fewer than 2 boundaries or not monotonically
                increasing.

        Examples:
            >>> measures = audio_tl.create_segment_line(
            ...     [0.0] + measure_times.tolist() + [float(audio_tl.length)]
            ... )
        """
        from ..base import Timeline
        from ..types import SegmentLine

        if len(boundaries) < 2:
            raise ValueError(
                f"Need at least 2 boundary coordinates, got {len(boundaries)}"
            )

        coords = [self._resolve_axis_value(boundary) for boundary in boundaries]
        for i in range(1, len(coords)):
            if coords[i] <= coords[i - 1]:
                raise ValueError(
                    f"Boundaries must be monotonically increasing: "
                    f"boundaries[{i - 1}]="
                    f"{_format_coordinate_value(float(coords[i - 1]))} >= "
                    f"boundaries[{i}]={_format_coordinate_value(float(coords[i]))}"
                )

        line_class = (
            SegmentLine if self.__class__ is Timeline else SegmentLine[self.__class__]
        )
        sl = line_class(
            length=0,
            unit=self._unit,
            number_type=self._number_type,
        )

        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            length = end - start

            segment = self.__class__(
                length=length,
                unit=self._unit,
                number_type=self._number_type,
                name=f"segment_{i}",
            )

            if copy_events:
                events_in_range = self.get_events(
                    min_coord=start,
                    max_coord=end,
                )
                adjusted = []
                for event in events_in_range:
                    adj = dict(event)
                    for col in ("instant", "start", "end"):
                        val = adj.get(col)
                        if val is not None:
                            adj[col] = shift_coordinate(val, start, subtract=True)
                    adjusted.append(adj)
                if adjusted:
                    segment.add_events(adjusted)

            sl.append_segment(segment)

        return sl

    def create_segment_line_from_regions(
        self,
        region_ids: Sequence[str] | None = None,
        *,
        copy_events: bool = True,
    ) -> "SegmentLine":
        """Create a SegmentLine from contiguous regions.

        Validates that regions are contiguous and non-overlapping
        (each region's end == next region's start).

        Does NOT modify self.

        Args:
            region_ids: Ordered region ids. None = all regions sorted
                by start coordinate. Each segment is named after its region.
            copy_events: Copy events into segments.

        Returns:
            A new SegmentLine.

        Raises:
            ValueError: If regions are not contiguous or empty.

        Examples:
            >>> tl.create_regions_by_grouping("timesig")
            >>> seg_line = tl.create_segment_line_from_regions()
        """
        if region_ids is None:
            # Sort regions by start coordinate
            sorted_regions = sorted(
                self._regions.values(), key=lambda r: float(r.start.value)
            )
            region_ids = [r.id for r in sorted_regions]

        if not region_ids:
            raise ValueError("No regions to create segment line from")

        regions = [self.get_region(region_id) for region_id in region_ids]

        # Validate contiguity
        for i in range(1, len(regions)):
            prev_end = float(regions[i - 1].end.value)
            curr_start = float(regions[i].start.value)
            if abs(prev_end - curr_start) > 1e-10:
                raise ValueError(
                    f"Regions are not contiguous: '{regions[i - 1].id}' "
                    f"ends at {_format_coordinate_value(float(prev_end))} but "
                    f"'{regions[i].id}' starts at "
                    f"{_format_coordinate_value(float(curr_start))}"
                )

        # Build boundaries from regions
        boundaries = [regions[0].start.value]
        for r in regions:
            boundaries.append(r.end.value)

        # Create the segment line
        sl = self.create_segment_line(boundaries, copy_events=copy_events)

        # Rename segments to match region names
        for i, seg_id in enumerate(sl._segment_order):
            sl._children[seg_id]._name = regions[i].name

        return sl

    def create_segment_line_by_grouping(
        self,
        field: str,
        *,
        copy_events: bool = True,
        name_format: str = "{value}",
    ) -> "SegmentLine":
        """Create a SegmentLine by grouping adjacent events on a field value.

        Groups must form contiguous, non-overlapping spans. This is validated
        and raises if not satisfied.

        Does NOT modify self. Does NOT add intermediate regions to self.

        Args:
            field: Event field to group by.
            copy_events: Copy events into segments.
            name_format: Format string for segment names.

        Returns:
            A new SegmentLine.

        Raises:
            ValueError: If groups are not contiguous.

        Examples:
            >>> systems = page.create_segment_line_by_grouping("spacing_run_id")
        """
        # Build runs (same logic as create_regions_by_grouping but temporary)
        events_sorted = self._sorted_event_dicts()
        if not events_sorted:
            raise ValueError("No events to group")

        first_event = events_sorted[0]
        if field not in first_event:
            raise ValueError(
                f"Field '{field}' not found in events. "
                f"Available fields: {list(first_event.keys())}"
            )

        runs: list[tuple[Any, Any, Any]] = []
        value_counts: dict[Any, int] = {}

        current_value = None
        run_start = 0.0
        run_end = 0.0

        for event in events_sorted:
            val = event.get(field)

            ev_start_value = event.get("start", event.get("instant"))
            ev_end_value = event.get("end")
            if ev_end_value is None:
                ev_end_value = ev_start_value
            ev_start = (
                coordinate_numeric_value(ev_start_value)
                if ev_start_value is not None
                else None
            )
            ev_end = (
                coordinate_numeric_value(ev_end_value)
                if ev_end_value is not None
                else ev_start
            )

            if val != current_value:
                if current_value is not None:
                    runs.append((current_value, run_start, run_end))
                current_value = val
                run_start = ev_start if ev_start is not None else 0.0
                run_end = ev_end if ev_end is not None else run_start
            else:
                if ev_end is not None:
                    run_end = max(run_end, ev_end)

        if current_value is not None:
            runs.append((current_value, run_start, run_end))

        # Validate contiguity
        for i in range(1, len(runs)):
            prev_end = runs[i - 1][2]
            curr_start = runs[i][1]
            if abs(prev_end - curr_start) > 1e-10:
                raise ValueError(
                    f"Groups are not contiguous: group "
                    f"'{runs[i - 1][0]}' ends at "
                    f"{_format_coordinate_value(float(prev_end))} but group "
                    f"'{runs[i][0]}' starts at "
                    f"{_format_coordinate_value(float(curr_start))}"
                )

        # Build boundaries
        if not runs:
            raise ValueError("No groups found")

        boundaries = [runs[0][1]]
        for _, _, end in runs:
            boundaries.append(end)

        sl = self.create_segment_line(boundaries, copy_events=copy_events)

        # Rename segments
        for i, seg_id in enumerate(sl._segment_order):
            value_counts.setdefault(runs[i][0], 0)
            value_counts[runs[i][0]] += 1
            seg_name = name_format.format(
                value=runs[i][0],
                i=i,
                n=i + 1,
                run=value_counts[runs[i][0]],
            )
            sl._children[seg_id]._name = seg_name

        return sl

    def create_segment_line_by_splitting(
        self,
        predicate: str | dict[str, Any] | Callable[[dict], bool],
        *,
        copy_events: bool = True,
        names: Sequence[str] | None = None,
        name_format: str = "{prefix}_{n}",
        prefix: str = "section",
        include_before_first: bool = True,
        include_after_last: bool = True,
    ) -> "SegmentLine":
        """Create a SegmentLine by splitting at events matching a predicate.

        Shortcut for finding split points and creating a SegmentLine directly.
        Does NOT modify self (no intermediate regions are created).

        The predicate follows the same semantics as
        :meth:`create_regions_by_splitting`.

        Args:
            predicate: Column name, filter dict, or callable identifying
                split-point events.
            copy_events: Copy events into segments.
            names: Explicit segment names.
            name_format: Format string for segment names.
            prefix: Prefix for auto-generated names.
            include_before_first: Include segment before first split point.
            include_after_last: Include segment after last split point.

        Returns:
            A new SegmentLine.

        Examples:
            >>> sl = tl.create_segment_line_by_splitting(
            ...     {"breaks": "section"}, prefix="movement"
            ... )
        """
        match_fn = self._resolve_predicate(predicate)

        split_coords: list[float] = []
        events_sorted = self._sorted_event_dicts()

        for event in events_sorted:
            if match_fn(event):
                coord = self._extract_coord_value(event, "end")
                if coord is None:
                    coord = self._extract_coord_value(event, "start", "instant")
                if coord is not None:
                    split_coords.append(coord)

        split_coords = sorted(set(split_coords))

        boundaries: list[float] = []
        if include_before_first:
            boundaries.append(float(self.origin.value))
        boundaries.extend(split_coords)
        if include_after_last:
            boundaries.append(float(self.length.value))

        boundaries = sorted(set(boundaries))

        if len(boundaries) < 2:
            raise ValueError("Not enough split points to create segments")

        sl = self.create_segment_line(boundaries, copy_events=copy_events)

        # Rename segments
        n_segments = sl.n_segments
        if names is not None:
            if len(names) != n_segments:
                raise ValueError(f"Expected {n_segments} names, got {len(names)}")
            seg_names = list(names)
        else:
            seg_names = [
                name_format.format(prefix=prefix, i=i, n=i + 1)
                for i in range(n_segments)
            ]

        for i, seg_id in enumerate(sl._segment_order):
            sl._children[seg_id]._name = seg_names[i]

        return sl

    def create_flow_map(
        self,
        intervals: Any,
        *,
        id: str = "default",
        at: "Sequence[Any] | None" = None,
        target_length: CoordinateSpec | None = None,
    ) -> "FlowMap":
        """Construct a FlowMap from interval-like descriptors, attach it, return it.

        Mirrors the ``create_*`` verb×noun convention (as in
        :meth:`create_region` and :meth:`create_regions_from_boundaries`):
        it constructs the FlowMap, attaches it to this timeline under *id*,
        and returns it.

        By default the played spans described by *intervals* concatenate
        contiguously in the unfolded (target) axis; coordinates falling in a
        gap between spans map to nothing (an empty ``unfold_coordinate``
        result).

        To lay the spans out with holes between them instead — which is what
        restoring a cut needs — state the placement in any of three ways: mix
        :class:`~timetoalign.timelines.flow.Gap` entries into *intervals*,
        give each span its target coordinate in *at*, or pass *intervals* as a
        ``{target coordinate -> span}`` mapping.

        Args:
            intervals: One interval-like descriptor, a collection of them, or
                a ``{target coordinate -> span}`` mapping. Accepted descriptor
                forms are region ids (``str``, resolved via
                :meth:`get_region`), ``Region`` objects, ``(start, end)``
                coordinate pairs, ``Timeline`` objects, and interval events.
                ``Gap`` entries may be mixed into a collection to space the
                spans apart.
            id: Identifier for the FlowMap. Defaults to ``"default"``.
            at: Target coordinate for each played span, in the order given.
                One entry per span, or ``None`` for a span that should follow
                its predecessor. Cannot be combined with ``Gap`` entries.
            target_length: Total extent of the unfolded axis. Needed only when
                the flow ends in a gap, which no section would imply.

        Returns:
            The constructed FlowMap (also attached to this timeline).

        Examples:
            >>> child.create_flow_map(["A8_1", "A8_2"], id="A8")
            FlowMap(A8: 2 sections)
            >>> child.create_flow_map([(0, 123), (129, child.length)], id="A8")
            FlowMap(A8: 2 sections)
            >>> # Restore the two skipped measures as a 6-quarter hole:
            >>> child.create_flow_map(["A8_1", Gap(6), "A8_2"], id="restored")
            FlowMap(restored: 2 sections, 1 gap)
            >>> # The same placement, stated as coordinates:
            >>> child.create_flow_map(["A8_1", "A8_2"], at=[0, 129], id="restored")
            FlowMap(restored: 2 sections, 1 gap)
            >>> # ... or as a mapping pairing each coordinate with its span:
            >>> child.create_flow_map({0: "A8_1", 129: "A8_2"}, id="restored")
            FlowMap(restored: 2 sections, 1 gap)
        """
        from ..flow import FlowMap

        fm = FlowMap(
            intervals,
            id=id,
            resolve=self.get_region,
            at=at,
            source_length=self.length.value,
            target_length=(
                None
                if target_length is None
                else self.get_coordinate_at(target_length, format="coordinate").value
            ),
        )
        self.add_flow_map(fm, id=id)
        return fm

    def add_flow_map(self, flow_map: "FlowMap", id: str | None = None) -> None:
        """Add a FlowMap to this timeline.

        FlowMaps enable coordinate transformation for timelines with flow
        control (repeats, jumps, D.S., D.C., etc.). They are created by
        `timetoalign.ScoreFlowController` and added to the timeline for
        later use.

        Design Decision: Timelines store FlowMaps, NOT FlowControllers.
        FlowControllers are factories that produce FlowMaps.

        Args:
            flow_map: The FlowMap to add.
            id: Identifier for this FlowMap. If None, uses flow_map.id.
                Common values: "default", "atomic", "single".

        Examples:
            >>> controller = ScoreFlowController(measure_data)
            >>> flow_map = controller.create_flow_map()
            >>> timeline.add_flow_map(flow_map)
            >>> timeline.get_flow_map("default")  # Retrieve later
            FlowMap(default: 5 sections)
        """
        if id is None:
            id = flow_map.id
        self._flow_maps[id] = flow_map
        self._logger.debug(f"Added FlowMap '{id}'")

    def get_flow_map(self, id: str = "default") -> "FlowMap | None":
        """Get an attached FlowMap by id.

        Args:
            id: Identifier of the FlowMap. Default is "default".

        Returns:
            The FlowMap if found, None otherwise.
        """
        return self._flow_maps.get(id)

    def has_flow_map(self, id: str = "default") -> bool:
        """Check if a FlowMap with the given id is attached.

        Args:
            id: Identifier to check.

        Returns:
            True if a FlowMap with that id exists.
        """
        return id in self._flow_maps

    def list_flow_maps(self) -> list[str]:
        """List all attached FlowMap ids.

        Returns:
            List of id strings.
        """
        return list(self._flow_maps.keys())

    @property
    def n_flow_maps(self) -> int:
        """Number of attached FlowMaps."""
        return len(self._flow_maps)

    def apply_flow(
        self,
        id: str = "default",
        *,
        include_children: bool = True,
        uid: str | None = None,
        name: str | None = None,
        fill_gaps: bool = False,
    ) -> "Timeline":
        """Yield the unfolded timeline for an attached FlowMap.

        Slices this timeline at each of the attached FlowMap's sections and
        places the slices, in target (unfolded) order, as children of a new
        timeline of this timeline's **same concrete type**. Each section also
        becomes a matching named Region on the result, in unfolded
        coordinates.

        Each slice sits at its section's target coordinate. Ordinary
        concatenating FlowMaps stack the slices end to end; a FlowMap that
        places its spans apart — a restored cut, or any FlowMap's
        :meth:`~timetoalign.timelines.flow.FlowMap.inverse` — leaves the
        result empty between them.

        Each appended child takes the source section's label as its id and
        name — the id of the region a played span was built from — and the
        Region marking it in the result takes that label as its name (its id
        is minted). A span visited more than once (a repeat) is suffixed
        ``-rend2``, ``-rend3`` … so every child has a unique id. Section events live in the appended children;
        the flattened coordinates remain reachable via
        ``get_events(include_children=True)``.

        The returned timeline carries a reverse FlowMap (id ``"source"``) for
        tracing coordinates back to the folded source, plus the forward
        FlowMap (id ``f"forward_{flow_map.id}"``).

        Args:
            id: Which attached FlowMap to unfold along. Passed positionally,
                so ``timeline.apply_flow("A8")`` selects the FlowMap stored
                under ``"A8"``.
            include_children: If True (default), child timelines are
                recursively sliced and included in each section.
            uid: Optional identifier for the returned timeline.
            name: Optional name for the returned timeline. Defaults to
                ``f"{self.name}_unfolded"``.
            fill_gaps: If True, each hole between the placed spans becomes an
                empty child (plus a matching Region), so the result tiles its
                axis contiguously. Required when this timeline is a
                ``SegmentLine``, which admits no gaps between its segments.

        Returns:
            The unfolded timeline (same concrete type as ``self``), with one
            child and matching Region per played section, each placed at its
            target coordinate.

        Raises:
            ValueError: If no FlowMap with the given id is attached.

        Examples:
            >>> child.create_flow_map(["A8_1", "A8_2"], id="A8")
            FlowMap(A8: 2 sections)
            >>> unfolded = child.apply_flow("A8")
            >>> unfolded.n_children
            2
            >>> unfolded.list_children()
            ['A8_1', 'A8_2']
            >>> # Applying the inverse puts the spans back where they came
            >>> # from, restoring the hole the cut left:
            >>> restored = unfolded.apply_flow("source")
            >>> float(restored.get_child_offset("A8_2").value)
            129.0
        """
        flow_map = self._flow_maps.get(id)
        if flow_map is None:
            raise ValueError(f"No FlowMap attached with id '{id}'")

        from ..flow.unfolding import unfold_via_flowmap

        return unfold_via_flowmap(
            self,
            flow_map,
            uid=uid,
            include_children=include_children,
            name=name,
            fill_gaps=fill_gaps,
        )

    def unfold_coordinate(
        self, coord: CoordinateSpec, id: str = "default"
    ) -> list[float]:
        """Convert a folded coordinate to unfolded coordinates.

        Convenience method that delegates to the attached FlowMap.
        Since repeats can cause a folded coordinate to appear multiple times
        in the unfolded timeline, this returns a list.

        Args:
            coord: Coordinate in the folded timeline.
            id: Which FlowMap to use.

        Returns:
            List of coordinates in the unfolded timeline.

        Raises:
            ValueError: If no FlowMap with the given id is attached.
        """
        flow_map = self._flow_maps.get(id)
        if flow_map is None:
            raise ValueError(f"No FlowMap attached with id '{id}'")
        return [
            float(c)
            for c in flow_map.unfold_coordinate(self._resolve_axis_value(coord))
        ]

    def fold(self, coord: CoordinateSpec, id: str = "default") -> float:
        """Convert an unfolded coordinate to a folded coordinate.

        Convenience method that delegates to the attached FlowMap.

        Args:
            coord: Coordinate in the unfolded timeline.
            id: Which FlowMap to use.

        Returns:
            Coordinate in the folded timeline.

        Raises:
            ValueError: If no FlowMap with the given id is attached,
                        or if the coordinate is outside the flow range.
        """
        flow_map = self._flow_maps.get(id)
        if flow_map is None:
            raise ValueError(f"No FlowMap attached with id '{id}'")
        return float(flow_map.fold(self._resolve_axis_value(coord)))
