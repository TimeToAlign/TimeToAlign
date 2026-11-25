"""Provide segment line and flow map operations for timeline instances."""

from __future__ import annotations

import datetime
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import timetoshare
from timetoshare import DEFAULT_BELIEFS, BeliefState

from timetoalign.core import CoordinateSpec, Gap
from timetoalign.core.enums import FlowConvention
from timetoalign.core.ids import quote_id
from timetoalign.core.provenance import Provenance
from timetoalign.core.time import coordinate_numeric_value
from timetoalign.core.timestamp import _format_coordinate_value

from ..regions import Region

if TYPE_CHECKING:
    from ..flow.flowmap import (
        ComputedFlows,
        FlowMap,
        PlayedFlow,
        StatedFlow,
        VirtualTimeline,
    )
    from ..types import SegmentLine
    from .host import TimelineHost as _Host
else:
    _Host = object


class SegmentsMixin(_Host):
    """Provide segment line, FlowMap and virtual timeline operations for
    timeline instances."""

    _flow_maps: dict[str, "FlowMap"]
    _virtual_timelines: dict[str, "VirtualTimeline"]

    def create_segment_line(
        self,
        boundaries: Sequence[CoordinateSpec],
        *,
        copy_events: bool = True,
    ) -> "SegmentLine":
        """Create a SegmentLine by segmenting at boundary coordinates.

        Given k+1 sorted coordinates, produces a new SegmentLine with k
        contiguous segments. Each segment's class matches self's class.
        Each segment takes the events starting in its span, whole (it grows
        to hold one running past its end), and, where this timeline holds
        Boxes, the part of its area of an image (``meta["box"]``,
        ``meta["image"]``) over its span, as
        :meth:`~timetoalign.timelines.Timeline.get_slice` states it, so a Box
        keeps its shape and its start moves with the segment.

        Does NOT modify self. Returns a new independent SegmentLine.

        Args:
            boundaries: k+1 monotonically increasing coordinates.
            copy_events: Copy events into their respective segments.

        Returns:
            A new SegmentLine with k segments.

        Raises:
            ValueError: If fewer than 2 boundaries or not monotonically
                increasing; if this timeline holds Boxes and its
                ``meta["box"]`` is no shape; or if it states no
                ``meta["box"]`` and a Box would move that a descendant
                holds, or one of its own into a segment not starting at 0.

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

            segment = type(self._timeline)(
                length=length,
                unit=self._unit,
                number_type=self._number_type,
                name=f"segment_{i}",
            )

            self._copy_events_to_cut(segment, start, end, copy_events=copy_events)
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

        Raises:
            UnknownLengthError: If ``include_after_last`` is set and the
                length is unknown, so that the last segment has no end.

        Examples:
            >>> sl = tl.create_segment_line_by_splitting(
            ...     {"breaks": "section"}, prefix="movement"
            ... )
        """
        end = (
            self._require_length("split up to the end").value
            if include_after_last
            else None
        )
        match_fn = self._resolve_predicate(predicate)

        # Boundaries in this timeline's number type: exact on an exact axis.
        split_coords: list[int | float | Fraction] = []
        events_sorted = self._sorted_event_dicts()

        for event in events_sorted:
            if match_fn(event):
                coord = self._extract_coord_value(event, "end")
                if coord is None:
                    coord = self._extract_coord_value(event, "start", "instant")
                if coord is not None:
                    split_coords.append(coord)

        split_coords = sorted(set(split_coords))

        boundaries: list[int | float | Fraction] = []
        if include_before_first:
            boundaries.append(self.origin.value)
        boundaries.extend(split_coords)
        if end is not None:
            boundaries.append(end)

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

    # region FlowMaps and virtual timelines

    def add_flow_map(self, flow_map: "FlowMap") -> "FlowMap":
        """Attach a FlowMap to this root timeline.

        The FlowMap is read against the timeline first
        (:meth:`FlowMap.resolve <timetoalign.timelines.flow.FlowMap.resolve>`),
        so one that names nothing of the tree never joins. Its gaps are kept
        in the timeline's number type. Its ID must be unique in the
        timeline's manifest: a minted ID already held there is minted again,
        a given one raises.

        Args:
            flow_map: The FlowMap to attach, naming this timeline.

        Returns:
            The stored FlowMap: *flow_map* itself, or a copy with its gaps in
            the timeline's number type or under the newly minted ID.

        Raises:
            ValueError: If this timeline is a child, the FlowMap names
                another timeline or cannot be read against this one, the
                timeline already holds a canonical FlowMap and this is a
                second one, or its given ID is already held in the manifest.
        """
        stored = flow_map._on_axis(self._timeline)
        stored.resolve(self._timeline)
        first = next(
            (held for held in self._flow_maps.values() if held.canonical), None
        )
        if stored.canonical and first is not None:
            raise ValueError(
                f"FlowMap {quote_id(stored.id)} is a second canonical FlowMap of "
                f"timeline {quote_id(self._id)}, after {quote_id(first.id)}"
            )
        flow_map_id = self._admit_element_id(
            stored.id,
            minted=stored._minted,
            prefix=stored._id_prefix,
            element=f"FlowMap {quote_id(stored.id)}",
        )
        if flow_map_id != stored.id:
            stored = replace(stored, id=flow_map_id, _minted=True)
        self._flow_maps[stored.id] = stored
        self._logger.debug(f"Added FlowMap '{stored.id}'")
        return stored

    def create_flow_map(
        self,
        chunks: Iterable[Any],
        *,
        id: str | None = None,
        convention: str | None = None,
        canonical: bool = False,
        name: str | None = None,
        provenance: Sequence[Any] = (),
    ) -> "FlowMap":
        """Create a FlowMap of this root timeline and attach it.

        Args:
            chunks: What is played, in order: each a Region of this timeline
                or of a descendant, a child timeline at any depth or a
                virtual timeline of this tree, given as the object or by its
                ID, or a :class:`~timetoalign.core.Gap`.
            id: The FlowMap's ID, kept verbatim. When None, ``flow_map_{N}``
                is minted, unique in this timeline's manifest.
            convention: Which flow of the flow control the FlowMap states;
                None for one written by hand.
            canonical: Whether this is the timeline's canonical FlowMap.
            name: Optional display name.
            provenance: How the FlowMap was made: provenance entries.

        Returns:
            The created FlowMap.

        Raises:
            ValueError: As :meth:`add_flow_map`.

        Examples:
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> score = ContinuousLogicalTimeline(length=24, uid="score")
            >>> a = score.create_region(0, 16, id="a")
            >>> b = score.create_region(16, 24, id="b")
            >>> score.create_flow_map([a, a, "b"], name="with the repeat")
            FlowMap('flow_map_1', timeline='score', 3 chunks, name='with the repeat')
            >>> score.get_flow_map("flow_map_1").resolve(score).length
            Coordinate(Fraction(40, 1), quarters)
        """
        from ..flow.flowmap import FlowMap

        return self.add_flow_map(
            FlowMap(
                self._id,
                tuple(
                    chunk if isinstance(chunk, str | Gap) else chunk.id
                    for chunk in chunks
                ),
                id=id,
                convention=convention,
                canonical=canonical,
                name=name,
                provenance=tuple(provenance),
            )
        )

    def get_flow_map(self, flow_map_id: str) -> "FlowMap":
        """Return the FlowMap of this timeline with the given ID.

        Raises:
            KeyError: If the timeline holds no such FlowMap.
        """
        if flow_map_id not in self._flow_maps:
            raise KeyError(
                f"Timeline {quote_id(self._id)} holds no FlowMap "
                f"{quote_id(flow_map_id)}. Available FlowMaps: {list(self._flow_maps)}"
            )
        return self._flow_maps[flow_map_id]

    def list_flow_maps(self) -> list[str]:
        """List the IDs of the FlowMaps this timeline holds."""
        return list(self._flow_maps)

    def get_flow(
        self,
        convention: FlowConvention | str = FlowConvention.default,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
    ) -> "PlayedFlow":
        """Play this timeline's flow control and return the flow, writing nothing.

        What is played is the flow control of the timeline and of its
        descendants of its unit, on its axis (:attr:`tree_flow_control
        <timetoalign.timelines.Timeline.tree_flow_control>`): a root plays
        its whole tree, a child its own subtree.

        Args:
            convention: ``default`` (every Jump and Break acts), ``single``
                (once through) or ``printed`` (the axis as it stands).
            believer: The actor whose beliefs decide which flow-control
                elements are read; None reads the elements no actor rejects.
            beliefs: The belief states read.

        Returns:
            The runs of the flow (:func:`~timetoalign.timelines.flow.engine.play`)
            placed one after the other: positions of this timeline's axis
            and of the flow convert through it. Its ``findings`` are what
            playing found on the way (a flow that never ends, a flow ending
            at a Break, a stretch never played, with its cause where
            ``single`` leaves out what ``default`` plays); the ``printed``
            flow has none. A timeline whose tree holds no flow control is played from its
            start to its end.

        Raises:
            UnknownLengthError: If the length is unknown.
            ValueError: If the convention is unknown, or the flow control
                cannot be played (two Jumps of the tree at one position of
                this axis among them).

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> from timetoalign.timelines.flowcontrol import Jump, JumpDestination
            >>> score = ContinuousLogicalTimeline(length=12, uid="score")
            >>> _ = score.flow_control.add(
            ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)]))
            >>> score.get_flow().length, score.get_flow("single").length
            (Coordinate(Fraction(20, 1), quarters), Coordinate(Fraction(12, 1), quarters))
            >>> score.get_flow().unfold(2)
            [Coordinate(Fraction(2, 1), quarters), Coordinate(Fraction(10, 1), quarters)]
            >>> score.get_flow().findings
            ()
        """
        from ..flow.engine import play
        from ..flow.flowmap import PlayedFlow

        played = play(
            self.tree_flow_control,
            self._require_length("play its flow control"),
            convention=convention,
            believer=believer,
            beliefs=beliefs,
        )
        return PlayedFlow.from_stretches(
            self._id,
            played.runs,
            unit=self._unit,
            number_type=self._number_type,
            findings=played.findings,
        )

    def create_flow_maps_from_flow_control(
        self,
        *,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        actor: str | None = None,
        date: str | datetime.date | None = None,
    ) -> "ComputedFlows | None":
        """Write this timeline's atomic sections and its computed FlowMaps.

        The flow control of the timeline and of its descendants of its unit
        (:attr:`tree_flow_control
        <timetoalign.timelines.Timeline.tree_flow_control>`) is played on
        the timeline's axis with the three conventions
        (:func:`~timetoalign.timelines.flow.engine.play`): a root plays its
        whole tree, a child its own subtree. The atomic
        sections (:func:`~timetoalign.timelines.flow.engine.atomic_sections`)
        become Regions of the timeline, each named by its letter and stating
        the type ``tts:AtomicSection`` with the provenance method
        ``inferred``. The ``default`` flow becomes a FlowMap listing those
        Regions in the order played; ``single`` and ``printed`` become
        FlowMaps only where their chunks differ from ``default``'s. Each
        FlowMap names its ``convention`` and the provenance method
        ``inferred``.

        A FlowMap names a root timeline. On a root, the Regions are the
        root's, also where its flow control lies only in descendants. On a
        child, the Regions are the child's and the FlowMaps are stored on
        the root of its tree, listing the child's Regions; a FlowMap written there anew is named after the
        child and its convention (``"Allegro, default"``). The computed
        FlowMaps of a timeline are those of its root that list its Regions
        only, so the flows of several children of one root stand side by
        side, and a virtual timeline applying one to the root takes its new
        length when it is written again.

        Called again after the flow control changed, the call takes nothing
        away and never lets a Region's ID name another stretch:

        - an atomic section whose stretch is still one keeps its Region; it
          is renamed where the new sectioning gives it another letter;
        - a stretch that is new gets a new Region, also one that was a
          section before and whose Region has been rejected since;
        - a Region whose stretch is an atomic section no longer stays, with
          its ID, its interval and its name, and *actor*'s belief ``"false"``
          is recorded in its statement ``type: tts:AtomicSection``
          (:meth:`Region.with_belief <timetoalign.timelines.Region.with_belief>`),
          so that it is no atomic section any more
          (:attr:`Region.is_atomic_section
          <timetoalign.timelines.Region.is_atomic_section>`). What names it
          (a FlowMap written by hand, a statement, a Jump going on at it)
          keeps naming the same stretch;
        - the computed FlowMaps keep their IDs, names and canonical flags
          and list the new sections; one written before for ``single`` or
          ``printed`` stays, also where its chunks have come to equal
          ``default``'s; a virtual timeline applying one takes its new
          length.

        Args:
            believer: The actor whose beliefs decide which flow-control
                elements are read; None reads the elements no actor rejects.
            beliefs: The belief states read.
            actor: Who recomputes, an IRI: the actor whose belief rejects
                the atomic sections that are none any more. Needed only
                where there are such sections.
            date: The day of that belief, ``YYYY-MM-DD`` or a
                :class:`datetime.date`; today when None.

        Returns:
            The atomic sections as they stand now (Regions kept and
            written, in the order of the axis), the computed FlowMaps, and
            the findings of the ``default`` flow (those of ``single`` are
            :meth:`get_flow`'s); or None for a timeline whose tree holds no
            flow control, which writes nothing.

        Raises:
            ValueError: If the flow control cannot be played; an atomic
                section the timeline holds is none any more and no *actor* is
                given to reject it; or the date is invalid. Nothing has
                changed then.
            UnknownLengthError: If the length is unknown.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> from timetoalign.timelines.flowcontrol import (
            ...     Jump, JumpDestination, JumpTarget)
            >>> score = ContinuousLogicalTimeline(length=12, uid="score")
            >>> _ = score.flow_control.add(
            ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)]))
            >>> written = score.create_flow_maps_from_flow_control()
            >>> [(section.id, section.name, str(section.as_interval))
            ...  for section in written.sections]
            [('region_1', 'A', '[0, 8) quarters'), ('region_2', 'B', '[8, 12) quarters')]
            >>> for flow_map in written.flow_maps:
            ...     print(flow_map.convention, flow_map.chunks)
            default ('region_1', 'region_1', 'region_2')
            single ('region_1', 'region_2')
            printed ('region_1', 'region_2')
            >>> written.findings
            ()

            A sign added at 4 divides the first section. Its Region stays
            and is rejected; the last section keeps its Region under a new
            letter:

            >>> _ = score.flow_control.add(JumpTarget(Coordinate(4, "quarters")))
            >>> again = score.create_flow_maps_from_flow_control(
            ...     actor="https://orcid.org/0000-0002-1825-0097", date="2026-11-25")
            >>> [(section.id, section.name, str(section.as_interval))
            ...  for section in again.sections]
            [('region_3', 'A', '[0, 4) quarters'), ('region_4', 'B', '[4, 8) quarters'), \
('region_2', 'C', '[8, 12) quarters')]
            >>> first = score.get_region("region_1")
            >>> str(first.as_interval), first.is_atomic_section
            ('[0, 8) quarters', False)
            >>> again.flow_maps[0].id, again.flow_maps[0].chunks
            ('flow_map_1', ('region_3', 'region_4', 'region_3', 'region_4', 'region_2'))
        """
        from ..flow.engine import FlowFinding, atomic_sections, play
        from ..flow.flowmap import ComputedFlows, FlowMap

        owner = self._timeline
        while owner._parent is not None:
            owner = owner._parent
        flow_control = self.tree_flow_control
        if not flow_control.elements:
            return None
        length = self._require_length("play its flow control")
        filtered = {"believer": believer, "beliefs": beliefs}
        sections = atomic_sections(flow_control, length, **filtered)
        orders: dict[FlowConvention, tuple[int, ...]] = {}
        findings: tuple[FlowFinding, ...] = ()
        for convention in FlowConvention:
            played = play(flow_control, length, convention=convention, **filtered)
            orders[convention] = tuple(
                index
                for run in played.runs
                for index, section in enumerate(sections)
                if run.start <= section.start and section.end <= run.end
            )
            if convention is not FlowConvention.default:
                continue
            findings = played.findings
            for finding in findings:
                self._logger.warning(
                    f"Timeline '{self._id}', {convention} flow: {finding.message}"
                )

        # The sections held, by stretch; a second Region of one stretch is
        # none of the new sectioning.
        standing: dict[tuple[Any, Any], Region] = {}
        held = [region for region in self._regions.values() if region.is_atomic_section]
        for region in held:
            standing.setdefault((region.start.value, region.end.value), region)
        stretches = {(section.start.value, section.end.value) for section in sections}
        kept = {
            region.id for stretch, region in standing.items() if stretch in stretches
        }
        left = [region for region in held if region.id not in kept]
        if left and actor is None:
            raise ValueError(
                f"timeline {quote_id(self._id)} holds the atomic section(s) "
                f"{[region.id for region in left]}, whose stretches its flow "
                "control does not give any more: they stay and are rejected as "
                "atomic sections, which needs the actor who recomputes (actor=)"
            )
        rejected = [
            region.with_belief(
                "type",
                region._atomic_section_type(),
                actor=actor,
                belief="false",
                date=date,
            )
            for region in left
            if actor is not None
        ]

        for region in rejected:
            self._regions[region.id] = region
        inferred = Provenance(method="inferred")
        regions: list[Region] = []
        for section in sections:
            region = standing.get((section.start.value, section.end.value))
            if region is None:
                region = self.add_region(
                    Region(section.start, section.end, name=section.name)
                    .with_prefix(_FORMAT_PREFIX, timetoshare.NAMESPACE)
                    .with_statement("type", _ATOMIC_SECTION_TYPE, provenance=inferred)
                )
            elif region.name != section.name:
                region = replace(region, name=section.name)
                self._regions[region.id] = region
            regions.append(region)

        # The computed FlowMaps of this timeline's flow control are those of
        # its root listing Regions of this timeline.
        computed = {
            flow_map.convention: flow_map
            for flow_map in owner._flow_maps.values()
            if flow_map.is_computed
            and all(
                isinstance(chunk, str) and chunk in self._regions
                for chunk in flow_map.chunks
            )
        }
        listed = list(owner._flow_maps)
        written: list[FlowMap] = []
        for convention, order in orders.items():
            before = computed.get(convention)
            if (
                before is None
                and convention is not FlowConvention.default
                and order == orders[FlowConvention.default]
            ):
                continue
            if before is not None:
                del owner._flow_maps[before.id]
            if before is not None:
                name = before.name
            elif owner is not self:
                name = f"{self.name}, {convention.value}"
            else:
                name = None
            written.append(
                owner.add_flow_map(
                    FlowMap(
                        owner._id,
                        tuple(regions[index].id for index in order),
                        convention=convention,
                        provenance=(inferred,),
                        id=None if before is None else before.id,
                        _minted=before is not None and before._minted,
                        canonical=before is not None and before.canonical,
                        name=name,
                    )
                )
            )
        rewritten = {flow_map.id: flow_map for flow_map in written}
        # A FlowMap written again keeps its place among the timeline's.
        owner._flow_maps = {
            flow_map_id: owner._flow_maps[flow_map_id]
            for flow_map_id in dict.fromkeys([*listed, *owner._flow_maps])
        }
        for virtual_id, virtual in list(owner._virtual_timelines.items()):
            if (
                virtual.map in rewritten
                and virtual.timeline == owner._id
                and virtual.through is None
            ):
                owner._virtual_timelines[virtual_id] = replace(
                    virtual,
                    length=rewritten[virtual.map].resolve(owner).length,
                )
        return ComputedFlows(tuple(regions), tuple(written), findings)

    def create_flow_map_from_measure_order(
        self,
        order: Iterable[int],
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> "StatedFlow":
        """Write an order of measures a source states as a FlowMap of this root.

        A source may state the order in which its measures are played
        itself: the successors of a measure table
        (:func:`~timetoalign.timelines.flow.measure_order_from_successors`),
        or an expansion a renderer wrote out. Each stretch of measures
        consecutive in the timeline's measure map becomes a Region (one
        Region for a stretch played more than once), and the FlowMap lists
        them in the order stated. The FlowMap says by its provenance that
        the source states it (method ``imported``) and follows no
        convention: it is no flow computed from flow control.

        The stated order is compared with the default flow of the
        timeline's flow control (:meth:`get_flow`). Where they differ a
        warning is logged and the result says from where.

        Each call writes its Regions and its FlowMap anew.

        Args:
            order: The measure counts in the order played.
            id: The FlowMap's ID, kept verbatim. When None, ``flow_map_{N}``
                is minted.
            name: Optional display name of the FlowMap.

        Returns:
            The Regions and the FlowMap written, and where the default flow
            differs.

        Raises:
            ValueError: If this timeline is a child, holds no measure map or
                is not in quarters; the order is empty or names a count that
                is no measure of the measure map or whose length is unknown;
                the flow control cannot be played; or the given ID is
                already held. Nothing has changed then.
            TypeError: If the flow :meth:`get_flow` returns holds a gap,
                which a flow played from flow control never does. Nothing
                has changed then.
            UnknownLengthError: If the length is unknown.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap
            >>> from timetoalign.timelines.flowcontrol import Jump, JumpDestination
            >>> score = ContinuousLogicalTimeline(length=12, uid="score")
            >>> _ = score.add_measure_map(MeasureMap.from_rows(
            ...     [{"count": 1, "time_signature": "4/4"}, {"count": 2}, {"count": 3}]))
            >>> _ = score.flow_control.add(
            ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)]))
            >>> stated = score.create_flow_map_from_measure_order([1, 2, 1, 2, 3])
            >>> [str(region.as_interval) for region in stated.regions]
            ['[0, 8) quarters', '[0, 12) quarters']
            >>> stated.flow_map.chunks, stated.differs_at
            (('region_1', 'region_2'), None)
            >>> score.create_flow_map_from_measure_order([1, 2, 3]).differs_at
            Coordinate(Fraction(8, 1), quarters)
        """
        from timetoalign.core import Coordinate
        from timetoalign.core.enums import TimeUnit

        from ..flow.flowmap import FlowMap, PlacedChunk, StatedFlow

        what = f"timeline {quote_id(self._id)}"
        if self._parent is not None:
            raise ValueError(
                f"{what} is a child of {quote_id(self._parent.id)}: a FlowMap "
                "names a root timeline"
            )
        measure_map = self.measure_map
        if measure_map is None:
            raise ValueError(
                f"{what} holds no measure map: an order of measures names none "
                "of its stretches"
            )
        if self._unit is not TimeUnit.quarters:
            raise ValueError(
                f"{what} is in {self._unit}: measures lie on an axis in quarters"
            )
        # A measure map counts its measures in printed order from 1.
        spans = {
            count: (measure.qstamp, measure.actual_length)
            for count, measure in enumerate(measure_map, start=1)
        }
        stretches: list[tuple[Fraction, Fraction]] = []
        before: int | None = None
        for count in order:
            if count not in spans:
                raise ValueError(
                    f"the order names measure count {count!r}, which is no "
                    f"measure of the measure map of {what} (counts "
                    f"{min(spans)} to {max(spans)})"
                )
            start, length = spans[count]
            if start is None or length is None:
                raise ValueError(
                    f"measure count {count} of {what} states no position or no "
                    "length: it names no stretch"
                )
            if before == count - 1 and stretches and stretches[-1][1] == start:
                stretches[-1] = (stretches[-1][0], start + length)
            else:
                stretches.append((start, start + length))
            before = count
        if not stretches:
            raise ValueError(f"the order of measures given for {what} is empty")

        def joined(runs: Iterable[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
            """Merge consecutive ``(start, end)`` runs that meet into one run."""
            together: list[tuple[Any, Any]] = []
            for start, end in runs:
                if together and together[-1][1] == start:
                    together[-1] = (together[-1][0], end)
                else:
                    together.append((start, end))
            return together

        played: list[tuple[Any, Any]] = []
        for chunk in self.get_flow().chunks:
            if not isinstance(chunk, PlacedChunk):
                raise TypeError(
                    f"the flow get_flow returns for {what} holds a gap at "
                    f"{chunk.start}: an order of measures is compared with "
                    "stretches of the timeline only"
                )
            played.append((chunk.source.start.value, chunk.source.end.value))
        default = joined(played)
        agreed: Any = 0
        differs = False
        stated_runs = joined(stretches)
        for index in range(max(len(stated_runs), len(default))):
            if index >= len(stated_runs) or index >= len(default):
                differs = True
                break
            (start, end), (other_start, other_end) = stated_runs[index], default[index]
            if start != other_start:
                differs = True
                break
            agreed += min(end, other_end) - start
            if end != other_end:
                differs = True
                break
        differs_at = (
            Coordinate(agreed, self._unit, number_type=self._number_type)
            if differs
            else None
        )

        bounds = list(dict.fromkeys(stretches))
        regions = [self.create_region(start, end) for start, end in bounds]
        chunk_of = dict(zip(bounds, regions, strict=True))
        try:
            flow_map = self.add_flow_map(
                FlowMap(
                    self._id,
                    tuple(chunk_of[stretch].id for stretch in stretches),
                    id=id,
                    name=name,
                    provenance=(Provenance(method="imported"),),
                )
            )
        except ValueError:
            for region in regions:
                del self._regions[region.id]
            raise
        if differs_at is not None:
            self._logger.warning(
                f"Timeline '{self._id}': the order of measures its source states "
                "differs from the default flow of its flow control. They agree "
                f"for the first {differs_at} played."
            )
        return StatedFlow(tuple(regions), flow_map, differs_at)

    def add_virtual_timeline(self, virtual: "VirtualTimeline") -> "VirtualTimeline":
        """Attach a virtual timeline to this root timeline.

        The statement must be true of this timeline: it applies a FlowMap
        the timeline holds to the timeline itself, and states the length,
        unit and number type the FlowMap gives. Its ID must be unique in the
        timeline's manifest: a minted ID already held there is minted again,
        a given one raises.

        Args:
            virtual: The virtual timeline to attach.

        Returns:
            The stored virtual timeline: *virtual* itself, or a copy under
            the newly minted ID.

        Raises:
            ValueError: If this timeline is a child, the statement cannot be
                read against it (:meth:`VirtualTimeline.resolve
                <timetoalign.timelines.flow.VirtualTimeline.resolve>`) or
                states another length than its FlowMap gives, or its given
                ID is already held in the manifest.
        """
        length = virtual.resolve(self._timeline).length
        if virtual.length != length or virtual.number_type is not length.number_type:
            raise ValueError(
                f"virtual timeline {quote_id(virtual.id)} states the length "
                f"{virtual.length!r}; FlowMap {quote_id(virtual.map)} gives {length!r}"
            )
        virtual_id = self._admit_element_id(
            virtual.id,
            minted=virtual._minted,
            prefix=virtual._id_prefix,
            element=f"virtual timeline {quote_id(virtual.id)}",
        )
        if virtual_id != virtual.id:
            virtual = replace(virtual, id=virtual_id, _minted=True)
        self._virtual_timelines[virtual.id] = virtual
        self._logger.debug(f"Added virtual timeline '{virtual.id}'")
        return virtual

    def create_virtual_timeline(
        self,
        flow_map: "FlowMap | str",
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> "VirtualTimeline":
        """State that a FlowMap of this root timeline was applied to it.

        Nothing but the statement is created: the virtual timeline has an
        ID, this timeline's unit and number type and the FlowMap's virtual
        length, and holds no events.

        Args:
            flow_map: A FlowMap this timeline holds, or its ID.
            id: The virtual timeline's ID, kept verbatim. When None, an ID
                of this timeline's kind is minted (``clt_{N}`` on quarters),
                unique in the manifest.
            name: Optional display name.

        Returns:
            The created virtual timeline.

        Raises:
            KeyError: If the timeline holds no such FlowMap.
            ValueError: As :meth:`add_virtual_timeline`.
        """
        from ..flow.flowmap import VirtualTimeline

        held = self.get_flow_map(flow_map if isinstance(flow_map, str) else flow_map.id)
        return self.add_virtual_timeline(
            VirtualTimeline.from_flow_map(held, self._timeline, id=id, name=name)
        )

    def get_virtual_timeline(self, virtual_id: str) -> "VirtualTimeline":
        """Return the virtual timeline of this timeline with the given ID.

        Raises:
            KeyError: If the timeline holds no such virtual timeline.
        """
        if virtual_id not in self._virtual_timelines:
            raise KeyError(
                f"Timeline {quote_id(self._id)} holds no virtual timeline "
                f"{quote_id(virtual_id)}. Available virtual timelines: "
                f"{list(self._virtual_timelines)}"
            )
        return self._virtual_timelines[virtual_id]

    def list_virtual_timelines(self) -> list[str]:
        """List the IDs of the virtual timelines this timeline holds."""
        return list(self._virtual_timelines)

    def _reassign_flow_id(self, old_id: str, new_id: str) -> bool:
        """Store the FlowMap or virtual timeline *old_id* under the minted *new_id*.

        Returns:
            Whether this timeline holds such an element.
        """
        if old_id in self._flow_maps:
            flow_map = replace(self._flow_maps.pop(old_id), id=new_id, _minted=True)
            self._flow_maps[new_id] = flow_map
            return True
        if old_id in self._virtual_timelines:
            virtual = replace(
                self._virtual_timelines.pop(old_id), id=new_id, _minted=True
            )
            self._virtual_timelines[new_id] = virtual
            return True
        return False

    def _rename_flow_references(self, renamed: Mapping[str, str | None]) -> None:
        """Make this timeline's FlowMaps and virtual timelines name the
        elements of *renamed* by their new IDs; one that left is still named."""
        names = {old: new for old, new in renamed.items() if new is not None}
        if not names or not (self._flow_maps or self._virtual_timelines):
            return
        self._flow_maps = {
            flow_map_id: flow_map._with_references_renamed(names)
            for flow_map_id, flow_map in self._flow_maps.items()
        }
        self._virtual_timelines = {
            virtual_id: virtual._with_references_renamed(names)
            for virtual_id, virtual in self._virtual_timelines.items()
        }

    # endregion


_FORMAT_PREFIX = "tts"
_ATOMIC_SECTION_TYPE = (
    f"{_FORMAT_PREFIX}:{timetoshare.ATOMIC_SECTION.removeprefix(timetoshare.NAMESPACE)}"
)
"""The type of an atomic section as its Region states it: ``tts:AtomicSection``,
under the prefix the Region declares for the format's namespace."""
