"""Unfold a timeline along a FlowMap."""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
from typing import TYPE_CHECKING

from .flowmap import FlowMap, FlowMapSection

if TYPE_CHECKING:
    from timetoalign.core import TimeUnit

    from ..base import Timeline


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

    It works from a built `FlowMap`, however that was constructed: from
    the runs of a played flow, from Regions, or from any other interval-like
    played spans.

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
