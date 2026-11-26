"""MatchStamp assembly of an AlignmentBundle: closure, warping and transfer.

The MatchStamp at a position joins what the MatchClaims in force state there
with what each timeline group's WarpMaps and C-Maps carry across; the
coordinate getters and the transfer of a coordinate between timelines read
the same assembly. The MatchGraphs, MatchLines and WarpMaps it builds are
cached on the bundle per belief filter.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, TypeGuard

import pandas as pd
from timetoshare import DEFAULT_BELIEFS, BeliefState, check_belief_filter

from timetoalign.alignment.claim_store import _ALL_BELIEFS, _belief_key, _BeliefKey
from timetoalign.alignment.claims import MatchClaim
from timetoalign.alignment.filters import ClaimFilter, belief_context, believed
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.matchline import MatchLine
from timetoalign.alignment.stamps import MatchIntervalStamp, MatchStamp
from timetoalign.alignment.warpmap import AmbiguousWarpMapError, WarpMap
from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    IdCoordinate,
    Interval,
    SupportPolicy,
    TimeUnit,
    express_as,
    resolve_coordinate_spec,
)
from timetoalign.core.enums import Domain
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    collection_values,
    dispatch_retrieval,
    format_coordinates,
    is_coordinate_input,
    resolve_key_collection,
    validate_coordinate_collection,
    validate_key_collection,
)
from timetoalign.core.time import CoordinateValue, exact_coordinate_value, shift_number
from timetoalign.core.timestamp import ConversionMapsSpec
from timetoalign.timelines import TimelineGroup, UnknownLengthError
from timetoalign.timelines.groups import _tree

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines import Timeline


def _resolve_coordinate_and_timeline(
    coordinate: CoordinateSpec,
    timeline_id: str | None,
) -> tuple[int | float | Fraction, str | None, "TimeUnit | None"]:
    """Decompose a coordinate specification and resolve its timeline.

    Accepts the three canonical coordinate forms and returns the underlying
    value, unit, and timeline the coordinate lives on. When the coordinate is
    an ``IdCoordinate`` it carries its own ``timeline_id``, which fills an
    omitted ``timeline_id`` argument.

    Args:
        coordinate: The query coordinate. One of:

            - int/float/Fraction: Raw value, ``timeline_id`` required.
            - Coordinate: Value with unit, ``timeline_id`` required.
            - IdCoordinate: Value, unit AND timeline_id (``timeline_id`` param
              optional).
        timeline_id: Explicit timeline id, or None to take it from an
            ``IdCoordinate``.

    Returns:
        A ``(coord_value, timeline_id, unit)`` tuple. ``timeline_id`` may
        still be None if a non-Id coordinate was passed without one; callers
        decide whether that is an error.

    Raises:
        TypeError: If ``coordinate`` is not one of the accepted forms.
    """
    resolved = resolve_coordinate_spec(coordinate, timeline_id=timeline_id)
    return resolved.value, resolved.timeline_id, resolved.unit


def _all_keys(values: list[object]) -> TypeGuard[list[str]]:
    """Whether every value of a collection is a key (an event id)."""
    return all(isinstance(value, str) for value in values)


def _all_positions(values: list[object]) -> TypeGuard[list[CoordinateInput | Interval]]:
    """Whether every value of a collection is a coordinate input or an Interval."""
    return all(
        is_coordinate_input(value) or isinstance(value, Interval) for value in values
    )


def _warp_units(
    self: AlignmentBundle, belief_key: _BeliefKey = _ALL_BELIEFS
) -> list[tuple[TimelineGroup | None, tuple[str, ...]]]:
    """Implement :meth:`AlignmentBundle._warp_units`."""
    membership = self.timeline_to_groups
    claimed = self._claims_in_force(belief_key).timeline_ids
    units: list[tuple[TimelineGroup | None, tuple[str, ...]]] = [
        (group, tuple(group.timeline_ids)) for group in self.groups.values()
    ]
    units.extend(
        (None, (timeline_id,))
        for timeline_id in self._known_timelines()
        if timeline_id not in membership
        and (timeline_id in self.timelines or timeline_id in claimed)
    )
    return units


def _group_component(self: AlignmentBundle, timeline_id: str) -> set[str]:
    """Implement :meth:`AlignmentBundle._group_component`."""
    membership = self.timeline_to_groups
    reached = {timeline_id}
    pending = [timeline_id]
    while pending:
        for group_id in membership.get(pending.pop(), ()):
            for member_id in self.groups[group_id].timeline_ids:
                if member_id not in reached:
                    reached.add(member_id)
                    pending.append(member_id)
    return reached


def transfer(
    self: AlignmentBundle,
    coord: CoordinateSpec,
    from_timeline: str,
    to_timeline: str,
    *,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> float | None:
    """Transfer a coordinate from one timeline to another.

    Automatically determines the conversion path:

    1. If both timelines are in the same group: typed group retrieval.
    2. If in different groups with MatchClaims: builds a ``MatchLine``
       and a cached ``WarpMap`` and calls it to
       interpolate the coordinate.
    3. If no path exists: returns ``None``.

    Low-level coordinate transfer utility. For user-facing coordinate
    queries, use ``get_matchstamp_at()`` which returns a full cross-section
    as a MatchStamp.

    Args:
        coord: The coordinate value to transfer.
        from_timeline: ID of the source timeline.
        to_timeline: ID of the target timeline.
        believer: The actor whose beliefs filter the MatchClaims read
            (the belief filter); ``None`` reads every claim.
        beliefs: The states read for the believer (``"true"``,
            ``"undecided"``, ``"unlikely"``, ``"false"``); default all
            but ``"false"``.

    Returns:
        The transferred coordinate, or None if no path exists.

    Raises:
        KeyError: If either id names no timeline of the bundle's trees.
        ValueError: If *beliefs* is not a collection of belief states.
    """
    belief_key = _belief_key(believer, beliefs)
    source_timeline = self._require_known_timeline(
        from_timeline, role="Source timeline"
    )
    self._require_known_timeline(to_timeline, role="Target timeline")

    coord_value, _resolved_timeline_id, _unit = _resolve_coordinate_and_timeline(
        coord, from_timeline
    )
    if isinstance(coord, IdCoordinate):
        query_input = coord.to_coordinate()
    elif isinstance(coord, Coordinate):
        query_input = coord
    else:
        query_input = Coordinate(
            coord_value,
            source_timeline.unit,
            number_type=source_timeline.number_type,
        )
    query_coordinate = source_timeline.get_coordinate_at(
        query_input,
        format="coordinate",
    )
    assert isinstance(query_coordinate, Coordinate)
    coord = float(query_coordinate.value)

    # Same timeline: no conversion needed
    if from_timeline == to_timeline:
        return coord

    # Within a group: direct conversion. Every group the two share
    # converts alike, so the first by id serves.
    group = self._shared_group(from_timeline, to_timeline)
    if group is not None:
        converted = group.get_coordinate_at(
            IdCoordinate(
                coord,
                self.get_timeline(from_timeline).unit,
                from_timeline,
            ),
            timeline_id=to_timeline,
            format="float",
        )
        return float(converted)

    # Cross-group transfer via MatchLine/WarpMap pipeline
    warp = self._get_or_build_warp_map(from_timeline, to_timeline, belief_key)
    if warp is not None:
        try:
            return warp._interpolate_float(coord)
        except ValueError as e:
            self._logger.warning(
                "WarpMap forward failed for %s -> %s at %s: %s",
                from_timeline,
                to_timeline,
                coord,
                e,
            )
            return None

    # Try indirect via a SOURCE group: convert within it, then warp
    for source_group in self.get_groups_for_timeline(from_timeline):
        for src_other_tl_id in source_group.timeline_ids:
            if src_other_tl_id == from_timeline:
                continue
            warp = self._get_or_build_warp_map(src_other_tl_id, to_timeline, belief_key)
            if warp is None:
                continue
            try:
                intermediate = source_group.get_coordinate_at(
                    IdCoordinate(
                        coord,
                        self.get_timeline(from_timeline).unit,
                        from_timeline,
                    ),
                    timeline_id=src_other_tl_id,
                    format="coordinate",
                )
                assert isinstance(intermediate, Coordinate)
                return warp._interpolate_float(intermediate.value)
            except (KeyError, ValueError):
                continue

    # Try indirect via a TARGET group: warp to a reachable group member,
    # then convert within that group.
    for target_group in self.get_groups_for_timeline(to_timeline):
        for tgt_other_tl_id in target_group.timeline_ids:
            if tgt_other_tl_id == to_timeline:
                continue
            warp = self._get_or_build_warp_map(
                from_timeline, tgt_other_tl_id, belief_key
            )
            if warp is None:
                continue
            try:
                warped = warp._interpolate_float(coord)
                target_source = self.get_timeline(tgt_other_tl_id)
                result = target_group.get_coordinate_at(
                    IdCoordinate(warped, target_source.unit, tgt_other_tl_id),
                    timeline_id=to_timeline,
                    format="coordinate",
                )
                assert isinstance(result, Coordinate)
                return float(result.value)
            except (KeyError, ValueError):
                continue

    # A single exact claim cannot construct a WarpMap, but it is still a
    # direct reachability edge. Reuse the match-stamp closure for that
    # exact-anchor case, including chains through standalone timelines.
    stamp = self.get_matchstamp_at(
        query_coordinate, from_timeline, believer=believer, beliefs=beliefs
    )
    if isinstance(stamp, MatchStamp) and to_timeline in stamp.coordinates:
        return float(stamp.coordinates[to_timeline].value)

    self._logger.debug(
        "No transfer path between '%s' and '%s'",
        from_timeline,
        to_timeline,
    )
    return None


def _transfer_interval(
    self: AlignmentBundle,
    start: CoordinateSpec,
    end: CoordinateSpec,
    from_timeline: str,
    to_timeline: str,
    *,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> tuple[float, float] | None:
    """Implement :meth:`AlignmentBundle.transfer_interval`."""
    start_value, _start_timeline_id, start_unit = _resolve_coordinate_and_timeline(
        start, from_timeline
    )
    end_value, _end_timeline_id, end_unit = _resolve_coordinate_and_timeline(
        end, from_timeline
    )
    source_timeline = self.get_timeline(from_timeline)
    start = float(
        source_timeline.get_coordinate_at(
            Coordinate(start_value, start_unit), format="coordinate"
        ).value
        if start_unit is not None
        else start_value
    )
    end = float(
        source_timeline.get_coordinate_at(
            Coordinate(end_value, end_unit), format="coordinate"
        ).value
        if end_unit is not None
        else end_value
    )
    transferred_start = self.transfer(
        start, from_timeline, to_timeline, believer=believer, beliefs=beliefs
    )
    transferred_end = self.transfer(
        end, from_timeline, to_timeline, believer=believer, beliefs=beliefs
    )

    if transferred_start is None or transferred_end is None:
        return None

    return (transferred_start, transferred_end)


def _group_positions(
    self: AlignmentBundle, timeline_id: str, query: Coordinate
) -> dict[str, Coordinate]:
    """Implement :meth:`AlignmentBundle._group_positions`."""
    positions = {timeline_id: query}
    pending = [timeline_id]
    while pending:
        member_id = pending.pop(0)
        for group in self.get_groups_for_timeline(member_id):
            if len(group.timeline_ids) < 2:
                continue
            converted = self._get_group_timestamp(
                group,
                positions[member_id].value,
                member_id,
                conversion_maps=False,
            )
            for other_id, value in converted.items():
                if other_id in positions or value is None:
                    continue
                other = self.get_timeline(other_id)
                positions[other_id] = Coordinate(
                    value, other.unit, number_type=other.number_type
                )
                pending.append(other_id)
    return positions


def _interval_claim_reading(
    claim: MatchClaim, positions: Mapping[str, Coordinate]
) -> tuple[str, Coordinate]:
    """Implement :meth:`AlignmentBundle._interval_claim_reading`."""
    for member_id, position in positions.items():
        if member_id not in claim.timelines:
            continue
        span = claim.get_interval_for(member_id)
        if span.start.value <= position.value <= span.end.value:
            return member_id, position
    raise ValueError(
        f"Interval claim {claim.id!r} holds none of the positions " f"{list(positions)}"
    )


def _interval_readings(
    self: AlignmentBundle,
    timeline_id: str,
    query: Coordinate,
    belief_key: _BeliefKey = _ALL_BELIEFS,
) -> list[tuple[MatchClaim, str, Coordinate]]:
    """Implement :meth:`AlignmentBundle._interval_readings`."""

    def names_interval_claims(member_id: str) -> bool:
        """Return whether a claim naming *member_id* under the belief filter is an interval."""
        rows = self._claims_naming(member_id, belief_key)._struct_array()
        return rows.field("end_anchor").null_count < len(rows)

    positions: dict[str, Coordinate] = {timeline_id: query}
    if any(
        names_interval_claims(member_id)
        for member_id in self._group_component(timeline_id)
        if member_id != timeline_id
    ):
        positions = self._group_positions(timeline_id, query)
    readings: list[tuple[MatchClaim, str, Coordinate]] = []
    read: set[str | None] = set()
    for member_id, position in positions.items():
        for claim in self._relevant_claims_at(member_id, position, belief_key):
            if not claim.is_interval or claim.id in read:
                continue
            read.add(claim.id)
            readings.append((claim, member_id, position))
    return readings


def _interval_claim_candidate(
    self: AlignmentBundle,
    claim: MatchClaim,
    queried_timeline_id: str,
    query: Coordinate,
) -> tuple[str, Coordinate | None, bool]:
    """Implement :meth:`AlignmentBundle._interval_claim_candidate`."""
    if not claim.is_interval or queried_timeline_id not in claim.timelines:
        raise ValueError("Interval claim does not contain the queried timeline")
    queried_interval = claim.get_interval_for(queried_timeline_id)
    counterpart_id = (
        claim.timeline_b_id
        if queried_timeline_id == claim.timeline_a_id
        else claim.timeline_a_id
    )
    counterpart_interval = claim.get_interval_for(counterpart_id)
    query_value = query.value
    queried_start = queried_interval.start.value
    queried_end = queried_interval.end.value
    counterpart_start = counterpart_interval.start.value
    counterpart_end = counterpart_interval.end.value
    is_anchor = query_value == queried_start or query_value == queried_end

    if queried_start == queried_end:
        if counterpart_start != counterpart_end:
            return counterpart_id, None, is_anchor
        mapped = counterpart_start
    elif query_value == queried_start:
        mapped = counterpart_start
    elif query_value == queried_end:
        mapped = counterpart_end
    else:
        operands = (
            query_value,
            queried_start,
            queried_end,
            counterpart_start,
            counterpart_end,
        )
        exact_query, exact_start, exact_end, exact_from, exact_to = (
            exact_coordinate_value(operand) for operand in operands
        )
        if (
            exact_query is not None
            and exact_start is not None
            and exact_end is not None
            and exact_from is not None
            and exact_to is not None
        ):
            query_value, queried_start, queried_end = (
                exact_query,
                exact_start,
                exact_end,
            )
            counterpart_start, counterpart_end = exact_from, exact_to
        mapped = counterpart_start + (
            (query_value - queried_start) / (queried_end - queried_start)
        ) * (counterpart_end - counterpart_start)

    timeline = self.get_timeline(counterpart_id)
    canonical = self._axis_on(mapped, counterpart_id)
    return (
        counterpart_id,
        Coordinate(
            canonical,
            timeline.unit,
            number_type=timeline.number_type,
        ),
        is_anchor,
    )


def _get_or_build_matchgraph(
    self: AlignmentBundle,
    timeline_id: str,
    coordinate: CoordinateValue,
    belief_key: _BeliefKey = _ALL_BELIEFS,
) -> MatchGraph:
    """Implement :meth:`AlignmentBundle._get_or_build_matchgraph`."""
    key = (belief_key, timeline_id, coordinate)
    if key in self._matchgraph_cache:
        return self._matchgraph_cache[key]

    timeline = self.get_timeline(timeline_id)
    query = Coordinate(coordinate, timeline.unit, number_type=timeline.number_type)
    relevant_claims = [
        claim
        for claim in self._relevant_claims_at(timeline_id, query, belief_key)
        if not claim.is_interval
    ]

    if not relevant_claims:
        raise ValueError(
            f"No synchronous claims touch ({timeline_id!r}, {coordinate}) "
            f"in this bundle."
        )

    # Build graph — transitive closure connects all timelines at
    # this coordinate
    mg = MatchGraph(
        claims=relevant_claims,
        units=self._get_unit_map(),
        axis=self._axis_on(coordinate, timeline_id),
        source=self,
        source_id=timeline_id,
    )

    # Cache keyed by ALL nodes in the graph
    for node in mg._graph.nodes():
        node_tl, node_coord = node
        self._matchgraph_cache[(belief_key, node_tl, node_coord)] = mg

    return mg


def _get_coordinate_at(
    self: AlignmentBundle,
    at: CoordinateInput,
    timeline_id: str | None = None,
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> CoordinateResult | pd.Series:
    """Implement :meth:`AlignmentBundle.get_coordinate_at`."""
    check_belief_filter(beliefs)
    if not (
        not isinstance(at, bool) and isinstance(at, (int, float, Fraction, Coordinate))
    ):
        raise TypeError("get_coordinate_at requires one scalar coordinate input")
    if isinstance(at, IdCoordinate):
        source_id = at.timeline_id
        result_id = timeline_id or source_id
    else:
        if timeline_id is None:
            raise ValueError(
                "timeline_id is required for raw or plain bundle coordinate queries"
            )
        source_id = timeline_id
        result_id = timeline_id
    if self._known_timeline(source_id) is None:
        raise KeyError(f"Unknown source timeline ID {source_id!r} in {self.id!r}")
    if self._known_timeline(result_id) is None:
        raise KeyError(f"Unknown result timeline ID {result_id!r} in {self.id!r}")
    if source_id == result_id:
        plain = at.to_coordinate() if isinstance(at, IdCoordinate) else at
        result = self.get_timeline(result_id).get_coordinate_at(
            plain,
            timeline_id=self.get_timeline(result_id).id,
            format="coordinate",
            rounding=rounding,
        )
        assert isinstance(result, Coordinate)
        identified = IdCoordinate.from_coordinate(result, result_id)
        return format_coordinates(
            [identified],
            format=format,
            rounding=rounding,
            scalar=True,
            series_name=result_id,
        )
    stamp = self.get_matchstamp_at(at, source_id, believer=believer, beliefs=beliefs)
    return stamp.get_coordinate_for(result_id, format=format, rounding=rounding)


def _get_coordinates_at(
    self: AlignmentBundle,
    at: CoordinateCollection,
    timeline_id: str | None = None,
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[CoordinateResult] | pd.Series:
    """Implement :meth:`AlignmentBundle.get_coordinates_at`."""
    check_belief_filter(beliefs)
    values, index = validate_coordinate_collection(at)
    if not values and timeline_id is None:
        raise ValueError("timeline_id is required for an empty bundle query")
    if timeline_id is not None and self._known_timeline(timeline_id) is None:
        raise KeyError(f"Unknown result timeline ID {timeline_id!r} in {self.id!r}")
    results: list[IdCoordinate] = []
    for value in values:
        result = self.get_coordinate_at(
            value,
            timeline_id=timeline_id,
            format="id_coordinate",
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )
        assert isinstance(result, IdCoordinate)
        results.append(result)
    return format_coordinates(
        results,
        format=format,
        rounding=rounding,
        scalar=False,
        index=index,
        series_name=timeline_id
        or (
            results[0].timeline_id
            if results
            and all(result.timeline_id == results[0].timeline_id for result in results)
            else "coordinate"
        ),
        empty_number_type=(
            self.get_timeline(timeline_id).number_type
            if timeline_id is not None
            else None
        ),
    )


def _get_coordinate_for(
    self: AlignmentBundle,
    key: str,
    timeline_id: str | None = None,
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> CoordinateResult | pd.Series:
    """Implement :meth:`AlignmentBundle.get_coordinate_for`."""
    check_belief_filter(beliefs)
    if not isinstance(key, str):
        raise TypeError("get_coordinate_for requires an event-ID string")
    owners = self._event_owners(key)
    if not owners:
        raise KeyError(f"Event {key!r} not found in bundle {self.id!r}")
    if len(owners) != 1:
        raise ValueError(f"Event {key!r} has competing owners {owners}")
    owner = owners[0]
    source = self.get_timeline(owner).get_coordinate_for(
        key, format="coordinate", rounding=rounding
    )
    assert isinstance(source, Coordinate)
    result = self.get_coordinate_at(
        IdCoordinate.from_coordinate(source, owner),
        timeline_id=timeline_id or owner,
        format="id_coordinate",
        rounding=rounding,
        believer=believer,
        beliefs=beliefs,
    )
    assert isinstance(result, IdCoordinate)
    return format_coordinates(
        [result],
        format=format,
        rounding=rounding,
        scalar=True,
        index=pd.Index([key]) if format == "series" else None,
        series_name="coordinate",
    )


def _get_coordinates_for(
    self: AlignmentBundle,
    keys: KeyCollection,
    timeline_id: str | None = None,
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[CoordinateResult] | pd.Series:
    """Implement :meth:`AlignmentBundle.get_coordinates_for`."""
    check_belief_filter(beliefs)
    values, index = validate_key_collection(keys)
    results: list[IdCoordinate] = []
    for key in values:
        result = self.get_coordinate_for(
            key,
            timeline_id=timeline_id,
            format="id_coordinate",
            rounding=rounding,
            believer=believer,
            beliefs=beliefs,
        )
        assert isinstance(result, IdCoordinate)
        results.append(result)
    if format == "series" and index is None:
        index = pd.Index(values)
    return format_coordinates(
        results,
        format=format,
        rounding=rounding,
        scalar=False,
        index=index,
        series_name="coordinate",
        empty_number_type=(
            self.get_timeline(timeline_id).number_type
            if timeline_id is not None
            else None
        ),
    )


def _get_coordinate(
    self: AlignmentBundle,
    at: CoordinateInput | CoordinateCollection | str | KeyCollection,
    timeline_id: str | None = None,
    *,
    format: CoordinateFormat = "id_coordinate",
    rounding: Rounding = "round",
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> CoordinateResult | list[CoordinateResult] | pd.Series:
    """Implement :meth:`AlignmentBundle.get_coordinate`."""
    return dispatch_retrieval(
        self,
        "get_coordinate",
        "get_coordinates",
        at,
        timeline_id,
        format=format,
        rounding=rounding,
        believer=believer,
        beliefs=beliefs,
    )


def _get_matchstamp_at(
    self: AlignmentBundle,
    at: CoordinateSpec | Interval,
    timeline_id: str | None = None,
    *,
    support_policy: "SupportPolicy | str | None" = None,
    conversion_maps: "ConversionMapsSpec" = False,
    within: set[str] | None = None,
    id_pattern: str | None = None,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> MatchStamp | MatchIntervalStamp:
    """Implement :meth:`AlignmentBundle.get_matchstamp_at`."""
    if isinstance(at, Interval):
        if timeline_id is None:
            raise ValueError(
                "timeline_id is required unless coordinate is an IdCoordinate"
            )
        start_stamp = self.get_matchstamp_at(
            at.start,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )
        end_stamp = self.get_matchstamp_at(
            at.end,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )
        assert type(start_stamp) is MatchStamp
        assert type(end_stamp) is MatchStamp
        queried_interval = Interval(
            start_stamp.coordinates[timeline_id],
            end_stamp.coordinates[timeline_id],
        )
        return MatchIntervalStamp(
            source_id=timeline_id,
            interval=queried_interval,
            start_stamp=start_stamp,
            end_stamp=end_stamp,
        )

    coordinate_value, timeline_id, _unit = _resolve_coordinate_and_timeline(
        at, timeline_id
    )
    if timeline_id is None:
        raise ValueError("timeline_id is required unless coordinate is an IdCoordinate")

    source_timeline = self._require_known_timeline(timeline_id)
    known = self._known_timelines()
    if isinstance(at, IdCoordinate):
        query_input = at.to_coordinate()
    elif isinstance(at, Coordinate):
        query_input = at
    else:
        query_input = Coordinate(
            coordinate_value,
            source_timeline.unit,
            number_type=source_timeline.number_type,
        )
    query_coordinate = source_timeline.get_coordinate_at(
        query_input,
        format="coordinate",
    )
    assert isinstance(query_coordinate, Coordinate)
    queried = query_coordinate.value

    belief_key = _belief_key(believer, beliefs)
    readings = self._interval_readings(timeline_id, query_coordinate, belief_key)
    interval_claims = [claim for claim, _, _ in readings]
    candidate_groups: dict[str, list[tuple[str, Coordinate | None, bool]]] = {}
    for claim, read_id, read_at in readings:
        target_id, candidate, is_anchor = self._interval_claim_candidate(
            claim, read_id, read_at
        )
        candidate_groups.setdefault(target_id, []).append(
            (read_id, candidate, is_anchor)
        )

    interval_coordinates: dict[str, CoordinateValue] = {}
    interval_stated_pairs: list[tuple[str, str]] = []
    interval_inferred_pairs: list[tuple[str, str]] = []
    for target_id, candidates in candidate_groups.items():
        values = [candidate for _, candidate, _ in candidates]
        if any(candidate is None for candidate in values):
            continue
        unique: list[Coordinate] = []
        for candidate in values:
            assert candidate is not None
            if candidate not in unique:
                unique.append(candidate)
        if len(unique) != 1:
            continue
        interval_coordinates[target_id] = unique[0].value
        # One edge per timeline the claims were read on: the queried
        # timeline, or a member of its groups.
        for read_id in dict.fromkeys(read_id for read_id, _, _ in candidates):
            edge = (read_id, target_id)
            if any(
                is_anchor
                for candidate_read_id, _, is_anchor in candidates
                if candidate_read_id == read_id
            ):
                interval_stated_pairs.append(edge)
            else:
                interval_inferred_pairs.append(edge)

    policy = (
        self.support_policy
        if support_policy is None
        else self._coerce_support_policy(support_policy)
    )

    coordinates, stated_pairs, inferred_pairs, query_has_anchor = (
        self._assemble_matchstamp(
            queried,
            timeline_id,
            support_policy=policy,
            conversion_maps=conversion_maps,
            belief_key=belief_key,
            interval_coordinates=interval_coordinates,
            interval_stated_pairs=interval_stated_pairs,
            interval_inferred_pairs=interval_inferred_pairs,
            interval_withheld=set(candidate_groups).difference(interval_coordinates),
            query_has_interval_anchor=any(
                is_anchor
                for candidates in candidate_groups.values()
                for read_id, _, is_anchor in candidates
                if read_id == timeline_id
            ),
        )
    )

    values = {
        coordinate_id: value
        for coordinate_id, value in coordinates.items()
        if value is not None
    }
    declared = [held for held in known if held in values]
    ordered_ids = [timeline_id]
    ordered_ids.extend(held for held in declared if held != timeline_id)
    ordered_ids.extend(sorted(set(values).difference(ordered_ids)))
    typed_coordinates: dict[str, Coordinate] = {}
    for ordered_id in ordered_ids:
        value = queried if ordered_id == timeline_id else values[ordered_id]
        timeline = self.get_timeline(ordered_id)
        typed_coordinates[ordered_id] = Coordinate(
            value,
            timeline.unit,
            number_type=timeline.number_type,
        )
    stamp = MatchStamp(
        coordinates=typed_coordinates,
        source_id=timeline_id,
        stated_pairs=list(stated_pairs),
        inferred_pairs=list(inferred_pairs),
        source=self,
        is_interpolated=not query_has_anchor,
        conversion_maps=conversion_maps,
        interval_claims=list(interval_claims),
    )

    # Apply post-hoc filtering if requested
    has_filter = (
        within is not None
        or id_pattern is not None
        or include_domains is not None
        or include_units is not None
    )
    if has_filter:
        filt = ClaimFilter.from_kwargs(
            within=within,
            id_pattern=self._known_id_pattern(id_pattern),
            include_domains=include_domains,
            include_units=include_units,
        )
        stamp = stamp._narrowed(filt, timelines=known)

    return stamp


def _get_matchstamps_at(
    self: AlignmentBundle,
    at: CoordinateCollection | Sequence[CoordinateInput | Interval],
    timeline_id: str | None = None,
    *,
    support_policy: "SupportPolicy | str | None" = None,
    conversion_maps: "ConversionMapsSpec" = False,
    within: set[str] | None = None,
    id_pattern: str | None = None,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[MatchStamp | MatchIntervalStamp]:
    """Implement :meth:`AlignmentBundle.get_matchstamps_at`."""
    values, _ = collection_values(at)
    for position, value in enumerate(values):
        if not is_coordinate_input(value) and not isinstance(value, Interval):
            raise TypeError(
                f"Matchstamp collection element {position} has unsupported "
                f"type {type(value).__name__}"
            )
    return [
        self.get_matchstamp_at(
            value,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        )
        for value in values
    ]


def _get_matchstamp_for(
    self: AlignmentBundle,
    key: str,
    timeline_id: str | None = None,
    *,
    support_policy: "SupportPolicy | str | None" = None,
    conversion_maps: "ConversionMapsSpec" = False,
    within: set[str] | None = None,
    id_pattern: str | None = None,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> MatchStamp:
    """Implement :meth:`AlignmentBundle.get_matchstamp_for`."""
    if not isinstance(key, str):
        raise TypeError("get_matchstamp_for requires an event-ID string")
    owners = self._event_owners(key)
    if not owners:
        raise KeyError(f"Event {key!r} not found in bundle {self.id!r}")
    if len(owners) != 1:
        raise ValueError(f"Event {key!r} has competing owners {owners}")
    owner = owners[0]
    source = self.get_timeline(owner).get_coordinate_for(key, format="coordinate")
    assert isinstance(source, Coordinate)
    return self.get_matchstamp_at(
        IdCoordinate.from_coordinate(source, owner),
        timeline_id or owner,
        support_policy=support_policy,
        conversion_maps=conversion_maps,
        within=within,
        id_pattern=id_pattern,
        include_domains=include_domains,
        include_units=include_units,
        believer=believer,
        beliefs=beliefs,
    )


def _get_matchstamps_for(
    self: AlignmentBundle,
    keys: KeyCollection,
    timeline_id: str | None = None,
    *,
    support_policy: "SupportPolicy | str | None" = None,
    conversion_maps: "ConversionMapsSpec" = False,
    within: set[str] | None = None,
    id_pattern: str | None = None,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[MatchStamp]:
    """Implement :meth:`AlignmentBundle.get_matchstamps_for`."""
    stamps, _ = resolve_key_collection(
        keys,
        lambda key: self.get_matchstamp_for(
            key,
            timeline_id,
            support_policy=support_policy,
            conversion_maps=conversion_maps,
            within=within,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
            believer=believer,
            beliefs=beliefs,
        ),
    )
    return stamps


def _get_matchstamp(
    self: AlignmentBundle,
    at: (
        CoordinateInput
        | Interval
        | CoordinateCollection
        | Sequence[CoordinateInput | Interval]
        | str
        | KeyCollection
    ),
    timeline_id: str | None = None,
    *,
    support_policy: "SupportPolicy | str | None" = None,
    conversion_maps: "ConversionMapsSpec" = False,
    within: set[str] | None = None,
    id_pattern: str | None = None,
    include_domains: set["Domain"] | None = None,
    include_units: set["TimeUnit"] | None = None,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> (
    MatchStamp
    | MatchIntervalStamp
    | list[MatchStamp]
    | list[MatchStamp | MatchIntervalStamp]
):
    """Implement :meth:`AlignmentBundle.get_matchstamp`."""
    options = {
        "support_policy": support_policy,
        "conversion_maps": conversion_maps,
        "within": within,
        "id_pattern": id_pattern,
        "include_domains": include_domains,
        "include_units": include_units,
        "believer": believer,
        "beliefs": beliefs,
    }
    if isinstance(at, str):
        return self.get_matchstamp_for(at, timeline_id, **options)
    if is_coordinate_input(at) or isinstance(at, Interval):
        return self.get_matchstamp_at(at, timeline_id, **options)
    values, _ = collection_values(at)
    if _all_keys(values):
        return self.get_matchstamps_for(values, timeline_id, **options)
    if _all_positions(values):
        return self.get_matchstamps_at(values, timeline_id, **options)
    raise TypeError(
        "Matchstamp dispatcher collections must contain only string keys or "
        "coordinate and Interval positions"
    )


def _assemble_matchstamp(
    self: AlignmentBundle,
    coordinate: CoordinateValue,
    actual_timeline_id: str,
    *,
    support_policy: SupportPolicy,
    conversion_maps: "ConversionMapsSpec",
    belief_key: _BeliefKey = _ALL_BELIEFS,
    interval_coordinates: dict[str, CoordinateValue] | None = None,
    interval_stated_pairs: list[tuple[str, str]] | None = None,
    interval_inferred_pairs: list[tuple[str, str]] | None = None,
    interval_withheld: set[str] | None = None,
    query_has_interval_anchor: bool = False,
) -> tuple[
    dict[str, CoordinateValue | None],
    list[tuple[str, str]],
    list[tuple[str, str]],
    bool,
]:
    """Implement :meth:`AlignmentBundle._assemble_matchstamp`."""
    coordinates: dict[str, CoordinateValue | None] = {}
    stated_pairs: list[tuple[str, str]] = []
    inferred_pairs: list[tuple[str, str]] = []

    def add_inferred_pair(timeline_a: str, timeline_b: str) -> None:
        """Record an inferred relationship once, regardless of orientation."""
        if timeline_a == timeline_b:
            return
        if (timeline_a, timeline_b) in inferred_pairs:
            return
        if (timeline_b, timeline_a) in inferred_pairs:
            return
        if (timeline_a, timeline_b) in stated_pairs:
            return
        if (timeline_b, timeline_a) in stated_pairs:
            return
        inferred_pairs.append((timeline_a, timeline_b))

    # 1. Exact-anchor seed: the query's connected component of stated
    #    claims (and any graph-inferred edges).
    try:
        mg = self._get_or_build_matchgraph(actual_timeline_id, coordinate, belief_key)
    except ValueError as error:
        if "No synchronous claims touch" not in str(error):
            raise
        mg = None

    query_has_anchor = mg is not None or query_has_interval_anchor
    if query_has_anchor:
        if mg is not None:
            graph_stamp = mg.get_matchstamp()
            coordinates.update(
                {
                    timeline_id: stored.value
                    for timeline_id, stored in graph_stamp.coordinates.items()
                }
            )
            stated_pairs.extend(graph_stamp.stated_pairs)
            inferred_pairs.extend(graph_stamp.inferred_pairs)
    for timeline_id, value in (interval_coordinates or {}).items():
        coordinates.setdefault(timeline_id, value)
    for timeline_id in interval_withheld or ():
        coordinates.setdefault(timeline_id, None)
    for edge in interval_stated_pairs or []:
        if edge not in stated_pairs and edge[::-1] not in stated_pairs:
            stated_pairs.append(edge)
    for edge in interval_inferred_pairs or []:
        add_inferred_pair(*edge)
    # The query timeline's coordinate is authoritative and never altered.
    coordinates[actual_timeline_id] = coordinate

    # 2. Transitive closure across groups (BFS worklist).
    worklist: list[str] = [
        timeline_id for timeline_id, value in coordinates.items() if value is not None
    ]
    processed: set[str] = set()
    tree_reached: set[str] = set()
    units = self._warp_units(belief_key)

    while worklist:
        node_id = worklist.pop()
        if node_id in processed:
            continue
        processed.add(node_id)
        node_coordinate = coordinates[node_id]
        assert node_coordinate is not None
        node_timeline = self.get_timeline(node_id)

        # Direct instant claims are reachability edges in their own right.
        # This precedes group handling so standalone timelines participate
        # in the same transitive closure.
        query = Coordinate(
            node_coordinate,
            node_timeline.unit,
            number_type=node_timeline.number_type,
        )
        for claim in self._relevant_claims_at(node_id, query, belief_key):
            if claim.is_interval:
                continue
            anchor = claim.start_anchor
            assert anchor is not None
            if anchor.timeline_a_id == node_id:
                target_id = anchor.timeline_b_id
                target_coordinate = anchor.coordinate_b.value
            else:
                target_id = anchor.timeline_a_id
                target_coordinate = anchor.coordinate_a.value
            if (node_id, target_id) not in stated_pairs and (
                target_id,
                node_id,
            ) not in stated_pairs:
                stated_pairs.append((node_id, target_id))
            if target_id in coordinates:
                continue
            coordinates[target_id] = target_coordinate
            worklist.append(target_id)

        # Every timeline reaches the others of its manifest tree (its root
        # and the root's descendants) by exact offsets. A timeline given
        # its coordinate that way has no relatives the first one lacked,
        # so its tree is not walked again from it.
        if node_id not in tree_reached:
            for relative_id, relative_coordinate in self._manifest_relatives(
                node_timeline, node_coordinate
            ).items():
                if relative_id in coordinates:
                    continue
                coordinates[relative_id] = relative_coordinate
                tree_reached.add(relative_id)
                add_inferred_pair(node_id, relative_id)
                worklist.append(relative_id)

        # The node's units: each of its groups, or the node alone when it
        # is ungrouped, registered or not.
        for source_group, _ in (unit for unit in units if node_id in unit[1]):
            # 2a. Within-group conversion to the members of each group of
            # the node: exact, and always within each member, since
            # members share one start and one end.
            if source_group is not None and len(source_group.timeline_ids) > 1:
                grouped = self._get_group_timestamp(
                    source_group,
                    node_coordinate,
                    node_id,
                    conversion_maps=conversion_maps,
                )
                for member_id, member_coordinate in grouped.items():
                    if member_id in coordinates or member_coordinate is None:
                        continue
                    coordinates[member_id] = member_coordinate
                    add_inferred_pair(node_id, member_id)
                    worklist.append(member_id)

            # 2b. Cross-group warp into every not-yet-reached unit the
            # node is no member of.
            for _, target_ids in units:
                if node_id in target_ids:
                    continue
                if all(member_id in coordinates for member_id in target_ids):
                    continue
                transferred = self._transfer_to_group(
                    float(node_coordinate),
                    node_id,
                    source_group,
                    target_ids,
                    support_policy,
                    belief_key,
                )
                if transferred is None:
                    continue
                target_id, target_coordinate = transferred
                if target_id in coordinates:
                    continue
                coordinates[target_id] = target_coordinate
                add_inferred_pair(node_id, target_id)
                worklist.append(target_id)

    return coordinates, stated_pairs, inferred_pairs, query_has_anchor


def _get_matchstamps(
    self: AlignmentBundle,
    claims: list[MatchClaim] | None = None,
    *,
    from_graph: bool = True,
    conversion_maps: "ConversionMapsSpec" = False,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> list[MatchStamp]:
    """Implement :meth:`AlignmentBundle.get_matchstamps`."""
    if claims is None:
        claims = self.get_match_claims(believer=believer, beliefs=beliefs)
    else:
        claims = believed(claims, believer, beliefs, context=belief_context(self))

    stamps = []
    for claim in claims:
        if not claim.is_synchronous or claim.start_anchor is None:
            continue
        if not from_graph:
            stamp = claim.get_matchstamp(
                bundle=self, from_graph=False, conversion_maps=conversion_maps
            )
        else:
            stamp = self.get_matchstamp_at(
                claim.start_anchor.get_coordinate_for(claim.timeline_a_id),
                conversion_maps=conversion_maps,
                believer=believer,
                beliefs=beliefs,
            )
        assert isinstance(stamp, MatchStamp)
        stamps.append(stamp)

    return stamps


def _get_or_build_match_line(
    self: AlignmentBundle, source_tl_id: str, belief_key: _BeliefKey = _ALL_BELIEFS
) -> MatchLine | None:
    """Implement :meth:`AlignmentBundle._get_or_build_match_line`."""
    cache_key = (belief_key, source_tl_id)
    if cache_key in self._matchline_cache:
        return self._matchline_cache[cache_key]

    naming = self._claims_naming(source_tl_id, belief_key)
    if len(naming) == 0:
        self._matchline_cache[cache_key] = None
        return None

    # Building a line from claims checked on entry has no failure that
    # means "no line": an error here propagates and is never cached.
    match_line = MatchLine.from_claims(
        claims=self._materialized(naming),
        source_timeline_id=source_tl_id,
    )
    self._matchline_cache[cache_key] = match_line
    return match_line


def _get_or_build_warp_map(
    self: AlignmentBundle,
    source_tl_id: str,
    target_tl_id: str,
    belief_key: _BeliefKey = _ALL_BELIEFS,
) -> WarpMap | None:
    """Implement :meth:`AlignmentBundle._get_or_build_warp_map`."""
    if len(self._claims_in_force(belief_key)) == 0:
        return None

    cache_key = (belief_key, source_tl_id, target_tl_id)
    if cache_key in self._warp_map_cache:
        return self._warp_map_cache[cache_key]

    # Build or retrieve cached MatchLine for this source
    match_line = self._get_or_build_match_line(source_tl_id, belief_key)
    if match_line is None:
        self._warp_map_cache[cache_key] = None
        return None

    # Check if target timeline is reachable
    if target_tl_id not in match_line.target_timeline_ids():
        self._warp_map_cache[cache_key] = None
        return None

    # Build WarpMap
    try:
        warp = WarpMap.from_match_line(match_line, target_tl_id)
    except AmbiguousWarpMapError:
        raise
    except ValueError as e:
        self._logger.debug(
            "Failed to build WarpMap %s -> %s: %s",
            source_tl_id,
            target_tl_id,
            e,
        )
        self._warp_map_cache[cache_key] = None
        return None

    self._warp_map_cache[cache_key] = warp
    self._logger.debug(
        "Built WarpMap %s -> %s (%d anchors)",
        source_tl_id,
        target_tl_id,
        warp.n_anchors,
    )
    return warp


def _get_group_timestamp(
    self: AlignmentBundle,
    group: TimelineGroup,
    coordinate: CoordinateValue,
    timeline_id: str,
    *,
    conversion_maps: "ConversionMapsSpec" = True,
) -> dict[str, CoordinateValue | None]:
    """Get the group members' exact coordinates at a position on one member.

    Args:
        group: The TimelineGroup.
        coordinate: Coordinate in the source timeline's native unit,
            or in a unit convertible via C-Map (see
            ``_resolve_claim_coordinate``).
        timeline_id: ID of the member the coordinate is on.

    Returns:
        Dict mapping member IDs to coordinates, each in its member's
        declared number type; a position outside the source member gives
        the source alone.
    """
    result: dict[str, CoordinateValue | None] = {}

    try:
        ts = group.get_timestamp_at(
            coordinate,
            timeline_id,
            conversion_maps=conversion_maps,
        )
        for tl_id in group.timeline_ids:
            try:
                coord = ts.get_coordinate_for(tl_id, format="coordinate")
            except KeyError:
                result[tl_id] = None
                continue
            assert isinstance(coord, Coordinate)
            result[tl_id] = coord.value
    except UnknownLengthError:
        # Converting between members needs both lengths; that is not a
        # position outside the source member, so it is not swallowed.
        raise
    except (KeyError, ValueError) as e:
        self._logger.debug(
            f"Failed to get group timestamp at {coordinate} on {timeline_id}: {e}"
        )
        # Return what we can: source coordinate at least
        result[timeline_id] = coordinate

    return result


def _manifest_relatives(
    self: AlignmentBundle, timeline: "Timeline", coordinate: CoordinateValue
) -> dict[str, CoordinateValue]:
    """Implement :meth:`AlignmentBundle._manifest_relatives`."""
    if not timeline._holds_position(coordinate, end_held=True):
        return {}
    carried: dict[str, CoordinateValue] = {timeline.id: coordinate}
    reached_from = [timeline]
    node = timeline
    while node._parent is not None:
        parent = node._parent
        carried[parent.id] = shift_number(
            carried[node.id], parent._child_offsets[node.id].value, subtract=False
        )
        reached_from.append(parent)
        node = parent
    root = node
    pending = reached_from
    while pending:
        node = pending.pop()
        for child_id, child in node._children.items():
            if child.id in carried:
                continue
            value = shift_number(
                carried[node.id], node._child_offsets[child_id].value, subtract=True
            )
            if child._holds_position(value):
                carried[child.id] = value
                pending.append(child)
    return {
        relative.id: express_as(carried[relative.id], relative.number_type)
        for relative in _tree(root)
        if relative is not timeline and relative.id in carried
    }


def _transfer_coordinate(
    self: AlignmentBundle,
    warp: WarpMap,
    entering: float,
    target_tl_id: str,
    support_policy: SupportPolicy,
) -> CoordinateValue | None:
    """Implement :meth:`AlignmentBundle._transfer_coordinate`."""
    source_coords = warp._source_float_array
    hull_low = float(source_coords[0])
    hull_high = float(source_coords[-1])
    entering = float(entering)
    length = self._timeline_length(target_tl_id)

    def _produce(value: float) -> float:
        return warp._interpolate_float(value)

    produced = _produce(entering)
    within_hull = hull_low <= entering <= hull_high
    within_length = produced >= 0.0 and (length is None or produced <= length)
    if within_hull and within_length:
        return produced

    if support_policy is SupportPolicy.omit:
        return None
    if support_policy is SupportPolicy.clamp:
        clamped = min(max(entering, hull_low), hull_high)
        return self._clip_to_length(target_tl_id, _produce(clamped))
    # extrapolate: keep the linear extrapolation, clipped into range.
    return self._clip_to_length(target_tl_id, produced)


def _transfer_to_group(
    self: AlignmentBundle,
    coordinate: float,
    source_tl_id: str,
    source_group: TimelineGroup | None,
    target_ids: Sequence[str],
    support_policy: SupportPolicy,
    belief_key: _BeliefKey = _ALL_BELIEFS,
) -> tuple[str, CoordinateValue] | None:
    """Transfer a coordinate from one group to another via WarpMap.

    Searches for a WarpMap connecting any timeline in the source
    group to any timeline in the target group.  Tries direct maps
    first, then indirect (convert within source group, then warp).
    Each transfer is governed by ``support_policy``
    (see :meth:`_transfer_coordinate`); a transfer omitted under the
    ``omit`` policy is skipped so another timeline in the target group may
    still be reached. An ungrouped timeline is a group of its own on
    either side (see :meth:`_warp_units`).

    Args:
        coordinate: Source coordinate.
        source_tl_id: Source timeline ID.
        source_group: Source group, or None when the source is ungrouped.
        target_ids: The target group's members, or the one ungrouped
            target timeline.
        support_policy: How out-of-support transfers are handled.
        belief_key: The belief filter the WarpMaps' claims are read under.

    Returns:
        ``(target_timeline_id, transferred_coordinate)`` or ``None``.
    """
    # Try direct: source_tl_id -> any timeline in target group
    for target_tl_id in target_ids:
        warp = self._get_or_build_warp_map(source_tl_id, target_tl_id, belief_key)
        if warp is None:
            continue
        try:
            transferred = self._transfer_coordinate(
                warp,
                coordinate,
                target_tl_id,
                support_policy,
            )
        except ValueError:
            continue
        if transferred is not None:
            return (target_tl_id, transferred)

    if source_group is None:
        return None

    # Try indirect: convert within source group, then warp
    for src_other_tl_id in source_group.timeline_ids:
        if src_other_tl_id == source_tl_id:
            continue
        # Convert within source group
        try:
            intermediate = source_group.get_coordinate_at(
                IdCoordinate(
                    coordinate,
                    self.get_timeline(source_tl_id).unit,
                    source_tl_id,
                ),
                timeline_id=src_other_tl_id,
                format="coordinate",
            )
            assert isinstance(intermediate, Coordinate)
        except (KeyError, ValueError):
            continue

        # Try WarpMap from intermediate to target group
        for target_tl_id in target_ids:
            warp = self._get_or_build_warp_map(
                src_other_tl_id, target_tl_id, belief_key
            )
            if warp is None:
                continue
            try:
                transferred = self._transfer_coordinate(
                    warp,
                    float(intermediate.value),
                    target_tl_id,
                    support_policy,
                )
            except ValueError:
                continue
            if transferred is not None:
                return (target_tl_id, transferred)

    return None
