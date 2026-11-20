"""Create unfolded timelines from computed flows."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import TYPE_CHECKING

from .flowmap import FlowMap, FlowMapSection

if TYPE_CHECKING:
    from timetoalign.core import TimeUnit

    from ..base import Timeline
    from .controller import FlowControllerBase, ScoreFlowController
    from .sections import Flow


def compute_qb_sections(
    flow: Flow,
    controller: FlowControllerBase | ScoreFlowController,
) -> list[tuple[Fraction, Fraction]]:
    """Compute quarterbeat boundaries for each PlaythroughSection in a Flow.

    Uses the controller's MeasureUnit data to convert MC-based section
    ranges to QB-based coordinate ranges. This is the critical function that
    translates from MC-number space (integers 1, 2, 3, ...) to quarterbeat
    space (actual coordinate positions on the score timeline).

    Args:
        flow: A computed Flow with PlaythroughSections.
        controller: The ScoreFlowController that computed the flow (must have
            MeasureUnit data with ``duration_qb`` for each MC).

    Returns:
        List of ``(qb_start, qb_end)`` tuples, one per PlaythroughSection.
        Each tuple gives the start and end quarterbeat coordinates in the
        **folded** source timeline. The intervals are right-open: ``[qb_start, qb_end)``.

    Raises:
        ValueError: If controller lacks MeasureUnit data for a required MC.
        ValueError: If flow has no sections.

    Examples:
        >>> controller = ScoreFlowController(measure_data)
        >>> flow = controller.compute_flow(FlowMode.default)
        >>> qb_sections = compute_qb_sections(flow, controller)
        >>> # Each section gives QB boundaries in the folded score
        >>> for qb_start, qb_end in qb_sections:
        ...     print(f"[{qb_start}, {qb_end})")
    """
    if not flow.sections:
        raise ValueError("Flow has no sections")

    # Build MC -> QB start lookup from MeasureUnit data.
    # mc_to_qb_start[mc] = cumulative QB position where measure mc begins.
    mc_to_qb_start: dict[int, Fraction] = {}
    mc_to_duration: dict[int, Fraction] = {}
    max_mc = 0

    for unit in controller.iter_units():
        mc_to_duration[unit.mc] = unit.duration_qb
        max_mc = max(max_mc, unit.mc)

    # Compute cumulative QB positions by summing durations in MC order.
    # MCs are sorted by MC number (1-indexed, monotonically increasing in the folded score).
    sorted_mcs = sorted(mc_to_duration.keys())
    cumulative = Fraction(0)
    for mc in sorted_mcs:
        mc_to_qb_start[mc] = cumulative
        cumulative += mc_to_duration[mc]

    # The QB position just past the last MC (used when mc_end is beyond max_mc)
    total_folded_qb = cumulative

    qb_sections: list[tuple[Fraction, Fraction]] = []

    for sec in flow.sections:
        # QB start: the cumulative QB position where mc_start begins
        if sec.mc_start not in mc_to_qb_start:
            raise ValueError(
                f"No MeasureUnit data for MC {sec.mc_start}. "
                f"Available MCs: {sorted(mc_to_qb_start.keys())}"
            )
        qb_start = mc_to_qb_start[sec.mc_start]

        # QB end: mc_end is right-open (exclusive), so its QB start IS the section's QB end.
        # If mc_end is beyond the last MC, use the total folded QB length.
        if sec.mc_end in mc_to_qb_start:
            qb_end = mc_to_qb_start[sec.mc_end]
        elif sec.mc_end > max_mc:
            # mc_end is one past the last MC -> section extends to the very end
            qb_end = total_folded_qb
        else:
            raise ValueError(
                f"No MeasureUnit data for MC {sec.mc_end}. "
                f"Available MCs: {sorted(mc_to_qb_start.keys())}"
            )

        qb_sections.append((qb_start, qb_end))

    return qb_sections


# endregion

# region Create Unfolded Timeline


def create_unfolded_timeline(
    source_timeline: "Timeline",
    flow: Flow,
    flow_controller: FlowControllerBase | None = None,
    *,
    uid: str | None = None,
    target_unit: "TimeUnit | str | None" = None,
    include_children: bool = True,
) -> "Timeline":
    """Create an unfolded timeline from a folded source via structural slicing.

    Computes QB-space boundaries for each `PlaythroughSection` in the Flow,
    extracts a slice from the source timeline at each boundary, and appends
    the slices, in target (unfolded) order, as children of a new timeline of
    the source's concrete type. Unfolding assembles a new timeline by selecting
    and concatenating contiguous portions of the folded source. See the
    Conceptual Model documentation (https://timetoalign.github.io/concepts.html).

    The returned timeline has:

    - The same concrete class as *source_timeline* (unless *target_unit*
      selects another), with correct QB-space coordinates.
    - One appended child per playthrough section, each named after the
      section (repeats add a ``-rend2``, ``-rend3`` … suffix).
    - A matching Region per child, in unfolded coordinates.
    - A reverse FlowMap attached (id ``"source"``) for folded ↔ unfolded
      conversion, and a forward FlowMap (id ``f"forward_{flow.id}"``).
    - Events structurally copied via `Timeline.get_slice()` (including
      truncation of interval events at section boundaries); the flattened
      coordinates remain reachable via
      ``get_events(include_children=True)``.

    Unfolding uses structural slicing because QB-space section boundaries
    preserve correct coordinates for scores with non-uniform measure
    durations.

    Args:
        source_timeline: The folded source timeline.
        flow: The computed Flow (sequence of sections).
        flow_controller: The controller that computed the flow. Required for
            QB-space boundary computation. If ``None``, the function raises
            ``ValueError`` because no QB-space boundaries can be computed.
        uid: Optional identifier for the returned timeline.
        target_unit: Optional unit for the unfolded timeline. When given,
            ``Timeline.resolve_subclass(target_unit, number_type)`` selects
            the timeline type; otherwise ``type(source_timeline)`` is used.
        include_children: If True (default), child timelines are
            recursively sliced and included in each section.

    Returns:
        New Timeline of the source's concrete type, with one appended child
        (plus matching Region) per playthrough section, a reverse FlowMap
        (id ``"source"``), and a forward FlowMap (id ``f"forward_{flow.id}"``).

    Raises:
        ValueError: If *flow_controller* is None and QB-space boundaries
            cannot be computed.

    Examples:
        >>> controller = ScoreFlowController(measure_data)
        >>> flow = controller.compute_flow(FlowMode.default)
        >>> unfolded = create_unfolded_timeline(source_tl, flow, controller)
        >>> type(unfolded).__name__  # preserves source type
        'ContinuousLogicalTimeline'
        >>> unfolded.get_flow_map("source")  # Reverse map to trace back
        FlowMap(default_inverse: 5 sections)

    See Also:
        `compute_qb_sections`: Computes QB boundaries from Flow + controller.
        `Timeline.get_slice`: Extracts a portion of a timeline.
    """
    # --- 1. Compute QB-space section boundaries ---
    if flow_controller is not None:
        qb_sections = compute_qb_sections(flow, flow_controller)
    else:
        raise ValueError(
            "flow_controller is required for QB-space unfolding. "
            "The legacy MC-number-space path has been removed."
        )

    # --- 2. Build the QB-space FlowMap and unfold along it ---
    forward_map = FlowMap.from_qb_sections(flow, qb_sections, id=flow.id)
    return unfold_via_flowmap(
        source_timeline,
        forward_map,
        uid=uid,
        target_unit=target_unit,
        include_children=include_children,
    )


def unfold_via_flowmap(
    source_timeline: "Timeline",
    forward_map: FlowMap,
    *,
    uid: str | None = None,
    target_unit: "TimeUnit | str | None" = None,
    include_children: bool = True,
    name: str | None = None,
    fill_gaps: bool = False,
) -> "Timeline":
    """Unfold a timeline by slicing it at a FlowMap's sections.

    Slices *source_timeline* once per section of *forward_map* — using each
    section's ``source_start``/``source_end`` range — and appends the slices,
    in target (unfolded) order, as children of a new timeline of the source's
    concrete type. Each section also becomes a named Region on the result, in
    unfolded coordinates. This is the structural model of unfolding: assembling
    a new timeline by selecting and concatenating contiguous portions of the
    folded source.

    Unlike `create_unfolded_timeline`, this works from an already-built
    `FlowMap`, so it applies to any FlowMap regardless of how it was
    constructed (a computed flow, quarterbeat boundaries, or interval-like
    played spans).

    Each slice is placed at its section's ``target_start``, which is what lets
    a FlowMap lay its spans out rather than only stack them end to end. Spans
    separated by a gap leave the assembled timeline empty in between — the
    layout an inverted cut needs to restore the source it came from. For the
    ordinary concatenating FlowMap, successive ``target_start`` values already
    stack the slices, so this is the same result as appending them.

    Each section is named after its FlowMap ``label`` (falling back to
    ``span_{i}`` when the section names nothing). A label visited more than
    once — a repeat — appends a ``-rend2``, ``-rend3`` … suffix so every child
    and Region has a unique name. That one name identifies the child timeline
    (both its id and name) and the Region for the section.

    The returned timeline carries:

    - A reverse FlowMap (id ``"source"``) for folded ↔ unfolded conversion.
    - The forward FlowMap (id ``f"forward_{forward_map.id}"``).
    - Events structurally copied via `Timeline.get_slice`, living in the
      appended children; the flattened coordinates remain reachable via
      ``get_events(include_children=True)``.
    - Children recursively sliced (when *include_children* is True).

    Args:
        source_timeline: The folded source timeline.
        forward_map: The FlowMap whose sections define the played spans, in
            the source timeline's coordinate space.
        uid: Optional identifier for the returned timeline.
        target_unit: Optional unit for the unfolded timeline. When given,
            ``Timeline.resolve_subclass(target_unit, number_type)`` selects
            the timeline type; otherwise ``type(source_timeline)`` is used.
        include_children: If True (default), child timelines are recursively
            sliced and included in each section.
        name: Optional name for the returned timeline. Defaults to
            ``f"{source_timeline.name}_unfolded"``.
        fill_gaps: If True, each hole between the placed spans becomes an
            empty child (plus a matching Region) of the result's own type, so
            the result tiles its axis contiguously. Required when the result
            is a ``SegmentLine``, which admits no gaps between its segments.
            Defaults to False, leaving the holes as plain empty space.

    Returns:
        New Timeline of the source's concrete type, with one child (plus
        matching Region) per played section, each placed at its target
        coordinate.

    See Also:
        `create_unfolded_timeline`: Computes QB-space boundaries from a Flow
            and controller, then delegates here.
        `Timeline.get_slice`: Extracts a portion of a timeline.
    """
    from ..base import Timeline
    from ..types import SegmentLine

    number_type = source_timeline.number_type
    unit = source_timeline.unit if target_unit is None else target_unit
    if target_unit is not None:
        timeline_cls = Timeline.resolve_subclass(unit, number_type)
    else:
        timeline_cls = type(source_timeline)

    _name = name if name is not None else f"{source_timeline.name}_unfolded"

    # Open the result at its full target extent so that a flow ending in a gap
    # keeps the trailing empty stretch, which no section would imply. A
    # SegmentLine is the exception: it measures contiguity against its own
    # length, so it has to grow with the segments rather than be pre-sized.
    is_segment_line = issubclass(timeline_cls, SegmentLine)
    result = timeline_cls(
        length=0 if is_segment_line else forward_map.total_target_length,
        unit=unit,
        number_type=number_type,
        uid=uid,
        name=_name,
    )

    # Holes reach the map both recorded (a Gap entry) and implied (the space
    # between two placed spans, as an inverse produces). Filling them makes the
    # result tile its axis contiguously, which is what a SegmentLine demands of
    # its children — so a placing flow can only be unfolded into one when the
    # holes are filled.
    gaps = forward_map.iter_gaps()
    if gaps and not fill_gaps and is_segment_line:
        raise ValueError(
            f"{type(result).__name__} requires contiguous segments, but this "
            f"FlowMap leaves {len(gaps)} hole(s) between its spans. Pass "
            f"fill_gaps=True to close them with empty segments."
        )

    # Walk spans and gaps together in target order: a SegmentLine accepts a
    # child only at its current end, so each hole has to be filled before the
    # span that follows it.
    placements: list[tuple[Fraction, Fraction, object]] = [
        (sec.target_start, sec.target_end, sec)
        for sec in forward_map._sections
        if not sec.is_gap
    ]
    if fill_gaps:
        placements += [(start, end, label) for start, end, label in gaps]
    placements.sort(key=lambda placement: placement[0])

    # Slice the source at each played section and place the slice — as a
    # same-type child plus a matching Region — at its target coordinate.
    # Repeated section labels are suffixed so every name is unique.
    base_counts: Counter[str] = Counter()
    n_spans = 0
    n_gaps = 0
    for start, end, payload in placements:
        if not isinstance(payload, FlowMapSection):
            # A gap: an empty segment of the result's own type, holding the
            # space that no source material fills. For a parameterized
            # SegmentLine, timeline_cls already carries the enforced segment
            # type, so the filler is a valid sibling of the placed slices
            # without any extra declaration.
            n_gaps += 1
            gap_name = payload or f"gap_{n_gaps}"
            filler = timeline_cls(
                length=end - start,
                unit=unit,
                number_type=number_type,
                uid=gap_name,
                name=gap_name,
            )
            result.add_child(filler, start, allow_expansion=True)
            result.create_region(start, end, name=gap_name)
            continue

        n_spans += 1
        base = payload.label or f"span_{n_spans}"
        base_counts[base] += 1
        occurrence = base_counts[base]
        section_name = base if occurrence == 1 else f"{base}-rend{occurrence}"

        slice_tl = source_timeline.get_slice(
            payload.source_start,
            payload.source_end,
            truncate_events=True,
            include_children=include_children,
        )
        slice_tl._set_id(section_name)
        slice_tl._name = section_name
        result.add_child(slice_tl, start, allow_expansion=True)
        result.create_region(start, end, name=section_name)

    # --- Add FlowMaps ---
    result.add_flow_map(forward_map.inverse(), id="source")
    result.add_flow_map(forward_map, id=f"forward_{forward_map.id}")

    return result


# endregion
