"""Map coordinates between folded and unfolded flows."""

from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from .sections import Flow, Gap, _coerce_flow_entries, _coerce_intervals


@dataclass
class FlowMapSection:
    """A section mapping in a FlowMap.

    Represents one contiguous section that maps from a source coordinate range
    to a target coordinate range. Used internally by FlowMap for coordinate
    transformation.

    In the context of flow control:
    - Source coordinates: The "folded" timeline (with repeats/jumps compressed)
    - Target coordinates: The "unfolded" timeline (linear playthrough order)

    Both ranges are carried explicitly, which is what lets a FlowMap place its
    sections rather than only concatenate them. A section is one of two kinds:

    - **Played** — both ranges have the same non-zero extent. Source material
      is carried to the target axis, at ``target_start`` rather than
      necessarily at the end of the preceding section.
    - **Gap** — one range is empty. An *insertion* (empty source range) is a
      stretch of target time no material fills; an *elision* (empty target
      range) is source material the flow drops. Inverting a FlowMap swaps the
      two ranges, so an insertion inverts to an elision and back.

    Attributes:
        source_start: Start coordinate in source timeline (inclusive).
        source_end: End coordinate in source timeline (exclusive).
        target_start: Start coordinate in target timeline.
        label: Identity of the section — the source region/section name it was
            built from, or ``None`` when the section names nothing (e.g. built
            from a bare coordinate pair). Unfolding uses this to name the
            child timeline and Region it produces for the section.
        target_end: End coordinate in target timeline. Defaults to ``None``,
            which derives it from ``target_start`` plus the source extent —
            the concatenating behaviour of an ordinary played section.
    """

    source_start: Fraction
    source_end: Fraction
    target_start: Fraction
    label: str | None = None
    target_end: Fraction | None = None

    def __post_init__(self) -> None:
        if self.target_end is None:
            self.target_end = self.target_start + self.duration

    @property
    def duration(self) -> Fraction:
        """Extent of this section on the source axis."""
        return self.source_end - self.source_start

    @property
    def target_duration(self) -> Fraction:
        """Extent of this section on the target axis."""
        return self.target_end - self.target_start

    @property
    def is_gap(self) -> bool:
        """Whether this section carries no material across the two axes."""
        return self.duration == 0 or self.target_duration == 0

    @property
    def is_insertion(self) -> bool:
        """Whether this is target time that no source material fills."""
        return self.duration == 0 and self.target_duration > 0

    @property
    def is_elision(self) -> bool:
        """Whether this is source material the flow drops."""
        return self.target_duration == 0 and self.duration > 0


class FlowMap:
    """Coordinate transformation map for flow control.

    A FlowMap enables bidirectional coordinate conversion between a source
    timeline (with flow control) and a target (linearized) timeline:

    - Source -> Target conversion (1:N, since repeats duplicate coordinates)
    - Target -> Source lookup (N:1, always unique)

    The single positional argument, *source*, is polymorphic:

    - ``None`` — an empty map (no sections). Used by internal construction
      such as :meth:`inverse`, which fills the section tables afterwards.
    - a :class:`Flow` — sections are derived from the flow's measure-count
      ranges (integer MC space); *id* defaults to the flow's mode value.
    - a single interval-like descriptor **or** an iterable of them — sections
      are built directly from the resulting ``(start, end)`` ranges with a
      cumulative target position, so the played spans concatenate in the
      target axis and any gaps between them map to nothing. *id* defaults to
      ``"default"``. Accepted descriptors are region ids (resolved through
      *resolve*), ``Region`` objects, ``(start, end)`` coordinate pairs,
      ``Timeline`` objects, and interval events — see
      :func:`~timetoalign.timelines.flow.sections._coerce_intervals`.

    Concatenation is only the default. Three interchangeable mechanisms
    **place** the played spans on the target axis instead, so the assembled
    timeline can hold holes — which is what an inverted cut needs:

    - :class:`~timetoalign.timelines.flow.sections.Gap` entries mixed into
      *source*, each pushing everything after it later by its own duration.
    - The *at* argument, giving the target coordinate of each played span
      outright.
    - A ``{target coordinate -> span}`` mapping as *source*, which pairs the
      two directly; see :meth:`from_dict`.

    All three agree on the result, and no two may be combined in one call. See
    :meth:`_build_from_entries` for the placement rules.

    FlowMap stores sections for efficient lookup:

    - ``_sections``: List of FlowMapSection objects
    - ``_target_boundaries``: Sorted list of target section starts for
      binary search

    Attributes:
        flow: The computed Flow, or ``None`` for interval-built maps.
        id: Identifier for this FlowMap.
        source_length: Total extent of the source axis, when known. Recorded so
            that :meth:`inverse` can restore a target axis whose final stretch
            is a gap — material the flow drops off the end leaves no section to
            infer the length from.
        target_length: Total extent of the target axis, when known.
    """

    def __init__(
        self,
        source: Flow | object | None = None,
        *,
        id: str = "",
        resolve: Callable[[str], object] | None = None,
        at: Sequence[Fraction | float | int] | None = None,
        source_length: Fraction | float | int | None = None,
        target_length: Fraction | float | int | None = None,
    ) -> None:
        """Build a FlowMap from a Flow, interval-like descriptors, or nothing.

        Args:
            source: ``None`` for an empty map, a :class:`Flow` for MC-space
                sections, or a single/collection of interval-like descriptors
                for directly-built sections. May mix in
                :class:`~timetoalign.timelines.flow.sections.Gap` entries to
                place the spans rather than concatenate them.
            id: Identifier for this FlowMap. Defaults to the flow's mode value
                for a Flow source, or ``"default"`` for interval descriptors.
            resolve: Callable mapping a region id to an interval-like object
                (e.g. a timeline's ``get_region``), used only when *source*
                carries region-name strings.
            at: Target coordinate for each played span, in the order the spans
                are given. One entry per span; use ``None`` for a span that
                should follow its predecessor. Cannot be combined with ``Gap``
                entries in *source*.
            source_length: Total extent of the source axis. Defaults to the
                end of the last section.
            target_length: Total extent of the target axis. Defaults to the
                end of the last section.

        Raises:
            ValueError: If *at* is combined with ``Gap`` entries, if it does
                not hold one entry per played span, or if the placements
                overlap or run backwards.
        """
        self.flow: Flow | None = None
        self.id: str = id
        self._sections: list[FlowMapSection] = []
        self._target_boundaries: list[Fraction] = []
        self.source_length: Fraction | None = (
            None if source_length is None else Fraction(source_length)
        )
        self.target_length: Fraction | None = (
            None if target_length is None else Fraction(target_length)
        )

        if source is None:
            return

        if isinstance(source, Flow):
            self.flow = source
            if not self.id:
                self.id = source.mode.value
            self._build_section_tables()
            return

        if not self.id:
            self.id = "default"

        if isinstance(source, Mapping):
            # A {target coordinate -> span} mapping states the placement as
            # data; see `from_dict`, which this shares its rules with.
            if at is not None:
                raise ValueError(
                    "Cannot combine `at` with a mapping source: a mapping "
                    "already gives each span its target coordinate"
                )
            entries, at = self._entries_from_mapping(source, resolve=resolve)
            self._build_from_entries(entries, at=at)
            return

        # Interval-like singleton or collection: build sections directly in
        # the source coordinate space of the descriptors (e.g. quarterbeats).
        entries = _coerce_flow_entries(source, resolve=resolve)
        self._build_from_entries(entries, at=at)

    @classmethod
    def from_dict(
        cls,
        placements: "Mapping[Fraction | float | int, object]",
        *,
        id: str = "default",
        resolve: Callable[[str], object] | None = None,
        source_length: Fraction | float | int | None = None,
        target_length: Fraction | float | int | None = None,
    ) -> "FlowMap":
        """Build a FlowMap from a ``{target coordinate -> span}`` mapping.

        The third way to state a placement, and the most direct: each key is
        the target coordinate at which its span begins. Where ``Gap`` entries
        describe the holes and ``at`` parallels the span list, a mapping puts
        coordinate and span side by side, which reads well when the placement
        comes from a table, a file, or an analysis::

            FlowMap.from_dict({0: "A8_1", 129: "A8_2"}, resolve=tl.get_region)

        Keys are sorted, so the mapping need not be written in target order.
        A value may also be a *collection* of spans, which are laid end to end
        from that coordinate on.

        Args:
            placements: Mapping from target coordinate to one interval-like
                descriptor — or a collection of them — in the same shapes the
                constructor accepts (region ids, ``Region`` objects,
                ``(start, end)`` pairs, ``Timeline`` objects, interval events).
            id: Identifier for this FlowMap.
            resolve: Callable mapping a region id to an interval-like object,
                needed only when the values carry region-name strings.
            source_length: Total extent of the source axis, when known.
            target_length: Total extent of the target axis, when known.

        Returns:
            The constructed FlowMap, its spans placed at the given coordinates
            and the space between them recorded as gaps.

        Raises:
            ValueError: If two spans would overlap on the target axis.

        Examples:
            >>> FlowMap.from_dict({0: (0, 123), 129: (123, 228)}, id="restored")
            FlowMap(restored: 2 sections, 1 gap)
        """
        fm = cls(id=id, source_length=source_length, target_length=target_length)
        entries, at = cls._entries_from_mapping(placements, resolve=resolve)
        fm._build_from_entries(entries, at=at)
        return fm

    @staticmethod
    def _entries_from_mapping(
        placements: "Mapping[Fraction | float | int, object]",
        *,
        resolve: Callable[[str], object] | None = None,
    ) -> tuple[list[tuple[Fraction, Fraction, str | None]], list[Fraction | None]]:
        """Flatten a ``{coordinate -> span(s)}`` mapping into entries and placements.

        Args:
            placements: The mapping to flatten.
            resolve: Callable mapping a region id to an interval-like object.

        Returns:
            The played spans in target order, and the matching ``at`` list:
            the coordinate for each group's first span and ``None`` for the
            spans that follow it there.
        """
        entries: list[tuple[Fraction, Fraction, str | None]] = []
        at: list[Fraction | None] = []

        for coordinate in sorted(placements, key=Fraction):
            spans = _coerce_intervals(placements[coordinate], resolve=resolve)
            for offset, span in enumerate(spans):
                entries.append(span)
                at.append(Fraction(coordinate) if offset == 0 else None)

        return entries, at

    def _build_from_intervals(
        self, intervals: list[tuple[Fraction, Fraction, str | None]]
    ) -> None:
        """Build section tables from ``(start, end, label)`` source ranges.

        Each interval becomes one FlowMapSection whose target position is the
        cumulative sum of preceding interval durations, so the played spans
        concatenate contiguously in the target (unfolded) axis. The interval's
        label is carried onto the section so unfolding can name the child
        timeline and Region it derives from the span.

        Args:
            intervals: The ``(start, end, label)`` source ranges, in target
                order.
        """
        self._build_from_entries(list(intervals))

    def _build_from_entries(
        self,
        entries: Sequence[tuple[Fraction, Fraction, str | None] | Gap],
        *,
        at: Sequence[Fraction | float | int | None] | None = None,
    ) -> None:
        """Build section tables from played spans and gaps, in target order.

        Walks the entries with a target cursor. A played span normally starts
        where the cursor stands, so consecutive spans concatenate. Two things
        move the cursor on instead:

        - A :class:`~timetoalign.timelines.flow.sections.Gap` advances it by
          the gap's duration, recorded as an *insertion* section (empty source
          range, non-empty target range). An auto-sized gap takes the distance
          between the source end of the span before it and the source start of
          the span after it.
        - An *at* entry sets it outright for the span it belongs to, which is
          the same placement stated as absolute coordinates rather than as
          relative holes. The implied hole is recorded as an insertion too, so
          both routes produce the same sections.

        Args:
            entries: Played ``(start, end, label)`` spans and ``Gap`` objects,
                in target order.
            at: Target coordinate per played span, or ``None`` per span to
                leave it following its predecessor.

        Raises:
            ValueError: If *at* is combined with gaps, if its length does not
                match the number of played spans, if a placement runs backwards
                or overlaps its predecessor, or if an auto-sized gap has no
                played span on both sides.
        """
        gaps = [entry for entry in entries if isinstance(entry, Gap)]
        played = [entry for entry in entries if not isinstance(entry, Gap)]

        if at is not None:
            if gaps:
                raise ValueError(
                    "Cannot combine `at` placements with Gap entries: state the "
                    "placement either as absolute coordinates or as gaps, not both"
                )
            if len(at) != len(played):
                raise ValueError(
                    f"`at` must hold one target coordinate per played span: got "
                    f"{len(at)} for {len(played)} span(s)"
                )

        target_position = Fraction(0)
        placements = list(at) if at is not None else []
        span_index = 0

        for i, entry in enumerate(entries):
            if isinstance(entry, Gap):
                duration = self._gap_duration(entry, entries, i)
                anchor = self._gap_source_anchor(entries, i)
                self._append_section(
                    FlowMapSection(
                        source_start=anchor,
                        source_end=anchor,
                        target_start=target_position,
                        label=entry.label,
                        target_end=target_position + duration,
                    )
                )
                target_position += duration
                continue

            source_start, source_end, label = entry
            placement = placements[span_index] if placements else None
            span_index += 1

            if placement is not None:
                placement = Fraction(placement)
                if placement < target_position:
                    raise ValueError(
                        f"Placement {placement} for span {label or i} overlaps the "
                        f"preceding span, which ends at {target_position}"
                    )
                if placement > target_position:
                    # The hole the placement leaves behind is itself a section,
                    # so `at` and Gap specs produce identical section tables.
                    anchor = self._gap_source_anchor(entries, i)
                    self._append_section(
                        FlowMapSection(
                            source_start=anchor,
                            source_end=anchor,
                            target_start=target_position,
                            label=None,
                            target_end=placement,
                        )
                    )
                target_position = placement

            self._append_section(
                FlowMapSection(
                    source_start=source_start,
                    source_end=source_end,
                    target_start=target_position,
                    label=label,
                )
            )
            target_position += source_end - source_start

    def _append_section(self, section: FlowMapSection) -> None:
        """Append a section and its target boundary to the lookup tables."""
        self._sections.append(section)
        self._target_boundaries.append(section.target_start)

    @staticmethod
    def _gap_duration(
        gap: Gap,
        entries: Sequence[tuple[Fraction, Fraction, str | None] | Gap],
        index: int,
    ) -> Fraction:
        """Resolve a gap's target duration, auto-sizing it when unstated.

        An auto-sized gap measures the hole its neighbours leave on the
        **source** axis: the distance from the source end of the nearest
        preceding played span to the source start of the nearest following one.

        Args:
            gap: The gap to size.
            entries: The full entry sequence the gap belongs to.
            index: Position of *gap* within *entries*.

        Returns:
            The gap's extent on the target axis.

        Raises:
            ValueError: If an auto-sized gap lacks a played span on either
                side, or if its neighbours run backwards on the source axis.
        """
        if gap.duration is not None:
            return Fraction(gap.duration)

        tail = index + 1
        before = next(
            (e for e in reversed(entries[:index]) if not isinstance(e, Gap)), None
        )
        after = next((e for e in entries[tail:] if not isinstance(e, Gap)), None)
        if before is None or after is None:
            side = "start" if before is None else "end"
            raise ValueError(
                f"An auto-sized Gap needs a played span on both sides to measure "
                f"the hole between them; this one is at the {side} of the flow. "
                f"Give it an explicit duration, e.g. Gap(6)."
            )

        duration = after[0] - before[1]
        if duration < 0:
            raise ValueError(
                f"An auto-sized Gap cannot measure a negative hole: the span "
                f"before it ends at source {before[1]} but the span after it "
                f"starts earlier, at source {after[0]}"
            )
        return duration

    @staticmethod
    def _gap_source_anchor(
        entries: Sequence[tuple[Fraction, Fraction, str | None] | Gap],
        index: int,
    ) -> Fraction:
        """Locate the empty source range of the gap at *index*.

        A gap carries no material, so its source range is a single point: the
        seam between the spans it separates. That is the source end of the
        nearest preceding played span, or — with nothing before it — the source
        start of the nearest following one.

        Args:
            entries: The full entry sequence the gap belongs to.
            index: Position of the gap within *entries*.

        Returns:
            The source coordinate at which the gap sits.
        """
        before = next(
            (e for e in reversed(entries[:index]) if not isinstance(e, Gap)), None
        )
        if before is not None:
            return before[1]
        tail = index + 1
        after = next((e for e in entries[tail:] if not isinstance(e, Gap)), None)
        return after[0] if after is not None else Fraction(0)

    def _build_section_tables(self) -> None:
        """Build section tables from the Flow for coordinate lookup.

        Each PlaythroughSection defines a source coordinate range (start, end)
        and its position in the target sequence is determined cumulatively.
        """
        if self.flow is None or not self.flow.sections:
            return

        target_position = Fraction(0)

        for sec in self.flow.sections:
            source_start = Fraction(sec.mc_start)
            source_end = Fraction(sec.mc_end)
            section_duration = source_end - source_start
            label = "+".join(sec.atomic_section_ids) or None

            self._sections.append(
                FlowMapSection(
                    source_start=source_start,
                    source_end=source_end,
                    target_start=target_position,
                    label=label,
                )
            )
            self._target_boundaries.append(target_position)

            target_position += section_duration

    def unfold_coordinate(self, coord: Fraction | float | int) -> list[Fraction]:
        """Map source coordinate to target coordinates.

        Since a source coordinate may be visited multiple times (due to repeats),
        this returns a list of all corresponding target coordinates.

        Args:
            coord: Coordinate in source timeline.

        Returns:
            List of coordinates in target timeline. Empty list if coord
            is not within any section.

        Examples:
            >>> # Source coord 3 appears twice due to repeat
            >>> flow_map.unfold_coordinate(3)
            [Fraction(2), Fraction(6)]
        """
        coord = Fraction(coord)
        results: list[Fraction] = []

        for sec in self._sections:
            # An elision covers source material the flow drops: it has no
            # target extent, so coordinates inside it are played nowhere.
            if sec.is_gap:
                continue
            # Check if coord falls within this section [start, end)
            if sec.source_start <= coord < sec.source_end:
                offset = coord - sec.source_start
                results.append(sec.target_start + offset)

        return results

    def fold(self, coord: Fraction | float | int) -> Fraction:
        """Map target coordinate back to source coordinate.

        The target timeline has unique coordinates, so this always returns
        a single value.

        Args:
            coord: Coordinate in target timeline.

        Returns:
            Coordinate in source timeline.

        Raises:
            ValueError: If coordinate is outside the flow range.

        Examples:
            >>> # Target coord 6 maps back to source coord 3
            >>> flow_map.fold(6)
            Fraction(3)
        """
        coord = Fraction(coord)

        if not self._sections:
            raise ValueError(f"FlowMap has no sections, cannot fold coordinate {coord}")

        # Binary search to find the section containing this target coordinate
        idx = bisect.bisect_right(self._target_boundaries, coord) - 1

        if idx < 0:
            raise ValueError(
                f"Coordinate {coord} is before the start of the flow "
                f"(starts at {self._target_boundaries[0]})"
            )

        if idx >= len(self._sections):
            raise ValueError(f"Coordinate {coord} is beyond the end of the flow")

        sec = self._sections[idx]

        # Beyond the section the search landed on, the coordinate is either in
        # a hole before the next section or past the whole flow. An insertion
        # is a hole recorded outright; the space between two placed sections is
        # the same hole left implicit.
        if sec.is_insertion or coord >= sec.target_end:
            hole_start = sec.target_start if sec.is_insertion else sec.target_end
            rest = idx + 1
            hole_end = next(
                (s.target_start for s in self._sections[rest:] if not s.is_elision),
                self.total_target_length,
            )
            if coord < hole_end:
                raise ValueError(
                    f"Coordinate {coord} falls in a gap of the flow "
                    f"([{hole_start}, {hole_end})), which no source material fills"
                )
            raise ValueError(
                f"Coordinate {coord} is beyond the end of the flow "
                f"(ends at {self.total_target_length})"
            )

        offset = coord - sec.target_start
        return sec.source_start + offset

    def inverse(self) -> "FlowMap":
        """Create the inverse FlowMap (target -> source becomes source -> target).

        The inverse FlowMap swaps the source and target coordinate systems.
        This is useful for attaching to a target timeline to enable
        tracing back to the original source.

        Because each section carries both ranges outright, inverting is a plain
        swap of the two. That is what makes an inverse **placing** rather than
        concatenating: a flow that drops material leaves its spans at their
        original source coordinates, so the inverse puts them back where they
        came from and the dropped stretch reappears as a gap. Applying the
        inverse with :meth:`~timetoalign.timelines.Timeline.apply_flow`
        therefore rebuilds a timeline laid out like the original.

        Note:
            The inverse FlowMap's unfold_coordinate() returns coordinates in
            the original source space, which may yield multiple results if the
            source coord is visited multiple times.

        Returns:
            A new FlowMap with inverted sections.
        """
        inverse_sections = [
            FlowMapSection(
                source_start=sec.target_start,
                source_end=sec.target_end,
                target_start=sec.source_start,
                label=sec.label,
                target_end=sec.source_end,
            )
            for sec in self._sections
        ]

        inverse = FlowMap(id=f"{self.id}_inverse")
        inverse.flow = self.flow
        inverse._sections = inverse_sections
        inverse._target_boundaries = [sec.target_start for sec in inverse_sections]
        inverse.source_length = self.target_length
        inverse.target_length = self.source_length

        return inverse

    @property
    def total_target_length(self) -> Fraction:
        """Total length of the target timeline.

        The recorded ``target_length`` when there is one — a flow whose final
        stretch is a gap ends past its last section — otherwise the end of the
        last section.
        """
        if self.target_length is not None:
            return self.target_length
        if not self._sections:
            return Fraction(0)
        return max(sec.target_end for sec in self._sections)

    @property
    def n_sections(self) -> int:
        """Number of played sections in this FlowMap.

        Gaps carry no material, so they are not counted here; see
        :meth:`iter_gaps` for the holes between the played sections.
        """
        return sum(1 for sec in self._sections if not sec.is_gap)

    @property
    def n_gaps(self) -> int:
        """Number of holes on the target axis."""
        return len(self.iter_gaps())

    def iter_gaps(self) -> list[tuple[Fraction, Fraction, str | None]]:
        """Report every stretch of target time that no source material fills.

        A hole reaches the map two ways, and both are reported here:

        - **Recorded** — an insertion section, built from a
          :class:`~timetoalign.timelines.flow.sections.Gap` entry or from an
          ``at`` placement, which may carry a label.
        - **Implied** — the space left between two placed sections, or between
          the last section and a longer recorded ``target_length``. Inverting a
          FlowMap produces holes this way, since the inverse simply puts each
          span back at its source coordinates.

        Returns:
            The ``(start, end, label)`` of each hole, in target order.
            Adjacent holes are merged, and *label* is the recorded gap's label
            or ``None``.
        """
        raw: list[tuple[Fraction, Fraction, str | None]] = []
        cursor = Fraction(0)

        for sec in self._sections:
            if sec.is_elision:
                # Dropped source material occupies no target time at all.
                continue
            if sec.is_insertion:
                raw.append((sec.target_start, sec.target_end, sec.label))
            elif sec.target_start > cursor:
                raw.append((cursor, sec.target_start, None))
            cursor = max(cursor, sec.target_end)

        end = self.total_target_length
        if end > cursor:
            raw.append((cursor, end, None))

        # Merge holes that run into each other, keeping the first label given.
        merged: list[tuple[Fraction, Fraction, str | None]] = []
        for start, stop, label in raw:
            if merged and merged[-1][1] == start:
                prev_start, _, prev_label = merged[-1]
                merged[-1] = (prev_start, stop, prev_label or label)
            else:
                merged.append((start, stop, label))

        return merged

    @classmethod
    def from_qb_sections(
        cls,
        flow: Flow,
        qb_sections: list[tuple[Fraction, Fraction]],
        *,
        id: str = "",
    ) -> "FlowMap":
        """Create a FlowMap with QB-space source coordinates.

        Unlike the default constructor which derives source coordinates from
        MC numbers (integers), this factory accepts pre-computed quarterbeat
        boundaries.  This is the correct approach for scores with non-uniform
        measure durations, where MC-number space != QB-coordinate space.

        Args:
            flow: The computed Flow (retained for metadata and inverse ops).
            qb_sections: List of ``(qb_start, qb_end)`` tuples giving the
                quarterbeat boundaries of each section in the **folded**
                source timeline.  Must have the same length as ``flow.sections``.
            id: Optional identifier.  Defaults to ``flow.mode.value``.

        Returns:
            FlowMap with sections in QB-space.

        Raises:
            ValueError: If ``len(qb_sections) != len(flow.sections)``.

        See Also:
            `compute_qb_sections`: Computes QB boundaries from a Flow and
                ScoreFlowController.
        """
        if len(qb_sections) != len(flow.sections):
            raise ValueError(
                f"qb_sections has {len(qb_sections)} entries but flow has "
                f"{len(flow.sections)} sections"
            )

        fm = cls.__new__(cls)
        fm.flow = flow
        fm.id = id or flow.mode.value
        fm._sections = []
        fm._target_boundaries = []
        fm.source_length = None
        fm.target_length = None

        target_position = Fraction(0)

        for (qb_start, qb_end), sec in zip(qb_sections, flow.sections):
            section_duration = qb_end - qb_start
            label = "+".join(sec.atomic_section_ids) or None
            fm._sections.append(
                FlowMapSection(
                    source_start=qb_start,
                    source_end=qb_end,
                    target_start=target_position,
                    label=label,
                )
            )
            fm._target_boundaries.append(target_position)
            target_position += section_duration

        return fm

    def __repr__(self) -> str:
        n_gaps = self.n_gaps
        gaps = f", {n_gaps} gap{'s' if n_gaps != 1 else ''}" if n_gaps else ""
        return f"FlowMap({self.id}: {self.n_sections} sections{gaps})"
