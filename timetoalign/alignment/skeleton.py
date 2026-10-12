"""A shared temporal structure for multiple timelines of one work."""

from __future__ import annotations

from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from timetoalign.core import (
    Address,
    Beat,
    BeatPolicy,
    Coordinate,
    FlowMode,
    Gap,
    IdCoordinate,
    IdGenerator,
    MeasureId,
    MeasureNumber,
    NumberType,
    TimeUnit,
)
from timetoalign.timelines.flow import Flow, PlaythroughSection
from timetoalign.timelines.groups import _root
from timetoalign.timelines.structure import MetricHierarchy, SectionHierarchy

from .bundle import AlignmentBundle
from .claims import AlignmentAnchor, MatchClaim

if TYPE_CHECKING:
    from timetoalign.timelines import BeatGrid

_skeleton_id_generator = IdGenerator(scope="skeleton")


class TimeSkeleton:
    """One authored temporal structure shared by participating timelines.

    Equality is structural: two skeletons compare equal when their section
    hierarchy, metric hierarchy, beat grid, and authored flows (flow ids
    and step content) agree. Identity (``id``), participants, claims, and
    materialized reference timelines are excluded, so equal skeletons are
    not interchangeable objects and instances are deliberately unhashable.
    """

    def __init__(
        self,
        section_hierarchy: SectionHierarchy,
        metric_hierarchy: MetricHierarchy | None = None,
        *,
        uid: str | None = None,
        flows: dict[str, list[str | Gap]] | None = None,
        beat_grid: BeatGrid | None = None,
    ) -> None:
        self._id = uid or _skeleton_id_generator.create(type_hint="skel")
        self._section_hierarchy = section_hierarchy
        self._metric_hierarchy = metric_hierarchy
        self._beat_grid = beat_grid
        self._bundle = AlignmentBundle()
        self._participants: dict[str, Any] = {}
        self._timeline_flows: dict[str, str] = {}
        # Flow-keyed cache of materialized references; the ``None`` key is
        # reserved for the abstract no-flow reference (flow ids are never
        # empty, so no authored flow can collide with it).
        self._reference_timelines: dict[str | None, Any] = {}
        self._flows: dict[str, Flow] = {}
        self._flow_steps: dict[str, tuple[str | Gap, ...]] = {}
        self._install_source_flow()
        for flow_id, steps in (flows or {}).items():
            self.add_flow(steps, id=flow_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TimeSkeleton):
            return NotImplemented
        return (
            self._section_hierarchy == other._section_hierarchy
            and self._metric_hierarchy == other._metric_hierarchy
            and self._beat_grid == other._beat_grid
            and self._flow_steps == other._flow_steps
        )

    def __repr__(self) -> str:
        hierarchy = self._section_hierarchy
        flows = ", ".join(self._flows)
        plural = "" if self.n_participants == 1 else "s"
        return (
            f"{type(self).__name__}({self._id!r}: {hierarchy.n_sections} sections, "
            f"{hierarchy.n_measures} measures, flows [{flows}], "
            f"{self.n_participants} participant{plural})"
        )

    @property
    def id(self) -> str:
        """Unique skeleton identifier."""
        return self._id

    @property
    def section_hierarchy(self) -> SectionHierarchy:
        """Section and measure structure."""
        return self._section_hierarchy

    @property
    def metric_hierarchy(self) -> MetricHierarchy | None:
        """Metrical structure, when authored."""
        return self._metric_hierarchy

    @property
    def flows(self) -> MappingProxyType[str, Flow]:
        """Authored traversals, always including printed-order ``source``."""
        return MappingProxyType(self._flows)

    @property
    def participants(self) -> tuple[Any, ...]:
        """Attached source and rendition timelines, excluding references."""
        return tuple(self._participants.values())

    @property
    def n_participants(self) -> int:
        """Number of attached participant timelines."""
        return len(self._participants)

    def attach(
        self,
        timeline: Any,
        *,
        flow: str | tuple[str, str] = "source",
        claims: Any = None,
    ) -> TimeSkeleton:
        """Enroll a timeline and return this skeleton for chaining.

        A measure-range ``flow`` tuple mints a one-step flow whose id is the
        range step itself (e.g. ``"1-5"``) — content-derived, never named
        after the participant, so equal ranges share one flow and structural
        equality stays participant-blind. An existing flow under that id is
        reused when it holds exactly the one range step and refused otherwise.
        """
        if isinstance(flow, tuple):
            if len(flow) != 2:
                raise ValueError("A measure range flow must contain exactly two IDs")
            step = f"{flow[0]}-{flow[1]}"
            flow_id = step
            existing_steps = self._flow_steps.get(flow_id)
            if existing_steps is None:
                self.add_flow([step], id=flow_id)
            elif existing_steps != (step,):
                raise ValueError(
                    f"Flow {flow_id!r} already exists with steps "
                    f"{existing_steps!r}; a range attach requires it to hold "
                    f"exactly ({step!r},)"
                )
        else:
            flow_id = flow
        if flow_id not in self._flows:
            raise KeyError(f"Unknown flow {flow_id!r}")
        if timeline.id not in self._participants:
            # The private bundle holds the manifest the participant belongs
            # to: its root, which a sibling participant may have brought.
            root = _root(timeline)
            if self._bundle.timelines.get(root.id) is not root:
                self._bundle.add_timeline(timeline)
            self._participants[timeline.id] = timeline
            timeline._add_skeleton_attachment(self)
        self._timeline_flows[timeline.id] = flow_id
        if claims is not None:
            self._bundle.add_match_claims(claims)
        return self

    def detach(self, timeline: Any) -> None:
        """Remove a participant and its claims from this structure."""
        timeline_id = timeline if isinstance(timeline, str) else timeline.id
        if timeline_id not in self._participants:
            raise KeyError(f"Timeline {timeline_id!r} is not attached")
        participant = self._participants.pop(timeline_id)
        root = _root(participant)
        if not any(_root(other) is root for other in self._participants.values()):
            self._bundle.timelines.pop(root.id, None)
        self._timeline_flows.pop(timeline_id, None)
        self._bundle.cross_group_claims = [
            claim
            for claim in self._bundle.cross_group_claims
            if timeline_id not in claim.timelines
        ]
        participant._remove_skeleton_attachment(self)

    def create_match_claim(
        self,
        timeline_id: str,
        *,
        at: Address | Coordinate | IdCoordinate | int | float | Fraction,
        coordinate: Coordinate | IdCoordinate | int | float | Fraction,
    ) -> MatchClaim:
        """Anchor a participant coordinate to its structural reference axis."""
        if timeline_id not in self._participants:
            raise KeyError(f"Timeline {timeline_id!r} is not attached")
        flow_id = self._timeline_flows[timeline_id]
        reference = self.materialize(flow=flow_id)
        participant = self._participants[timeline_id]
        participant_coordinate = self._coordinate_for_timeline(coordinate, participant)
        reference_coordinate = self._resolve_reference_position(at, flow_id)
        anchor = AlignmentAnchor(
            timeline_a_id=timeline_id,
            coordinate_a=participant_coordinate,
            timeline_b_id=reference.id,
            coordinate_b=reference_coordinate,
        )
        claim = MatchClaim(
            timeline_a_id=timeline_id,
            timeline_b_id=reference.id,
            start_anchor=anchor,
        )
        self._bundle.add_match_claims([claim])
        return claim

    def add_flow(self, steps: list[str | Gap], *, id: str) -> Flow:  # noqa: A002
        """Author a traversal from measure ranges, section IDs, and gaps."""
        if not id:
            raise ValueError("A flow ID must not be empty")
        if id in self._flows:
            raise ValueError(f"Flow {id!r} already exists")
        normalized = tuple(steps)
        sections = [
            PlaythroughSection(
                mc_start=start,
                mc_end=end + 1,
                atomic_section_ids=(label,),
            )
            for step in normalized
            if not isinstance(step, Gap)
            for start, end, label in [self._resolve_step(step)]
        ]
        flow = Flow(
            sections=sections,
            mode=FlowMode.custom,
            folded_length=self._section_hierarchy.n_measures,
            id=id,
        )
        self._flows[id] = flow
        self._flow_steps[id] = normalized
        return flow

    def create_beatgrid(self) -> BeatGrid:
        """Return the tempo grid this structure was read from.

        Returns:
            The :class:`~timetoalign.timelines.BeatGrid` whose segments
            generated this structure's measures.

        Raises:
            ValueError: If the structure carries no tempo segments —
                a beat grid needs a tempo, which measure lengths alone
                do not state.
        """
        if self._beat_grid is None:
            raise ValueError(
                f"Structure {self._id!r} carries no tempo segments, so it states "
                "no beat grid"
            )
        return self._beat_grid

    def materialize(self, *, flow: str | None = None) -> Any:
        """Materialize the supported abstract or unfolded reference timeline."""
        from timetoalign.timelines import ContinuousLogicalTimeline

        if flow is None:
            measures = self._section_hierarchy.measure_map.measures
            if measures and any(
                measure.actual_length is not None for measure in measures
            ):
                raise ValueError(
                    "Concrete structures require materialize(flow='<id>'); "
                    "only an abstract single-original structure supports materialize()"
                )
            existing = self._reference_timelines.get(None)
            if existing is not None:
                return existing
            reference = ContinuousLogicalTimeline(
                length=float(len(measures) + 1),
                unit=TimeUnit.floating_measures,
                number_type=NumberType.float,
            )
            self._reference_timelines[None] = reference
            reference._add_skeleton_attachment(self)
            return reference
        if flow not in self._flows:
            raise KeyError(f"Unknown flow {flow!r}")
        existing = self._reference_timelines.get(flow)
        if existing is not None:
            return existing
        length = self._flow_length(flow)
        reference = ContinuousLogicalTimeline(
            length=length,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        self._reference_timelines[flow] = reference
        self._bundle.add_timeline(reference)
        reference._add_skeleton_attachment(self)
        return reference

    def _install_source_flow(self) -> None:
        steps = [section.id for section in self._section_hierarchy.sections]
        sections = [
            PlaythroughSection(
                mc_start=section.mc_start,
                mc_end=section.mc_end,
                atomic_section_ids=(section.id,),
            )
            for section in self._section_hierarchy.sections
        ]
        self._flows["source"] = Flow(
            sections=sections,
            mode=FlowMode.printed,
            folded_length=self._section_hierarchy.n_measures,
            id="source",
        )
        self._flow_steps["source"] = tuple(steps)

    def _resolve_step(self, step: str) -> tuple[int, int, str]:
        if step.startswith("sec"):
            for section in self._section_hierarchy.sections:
                if section.id == step:
                    return section.mc_start, section.mc_end - 1, step
            raise KeyError(f"Unknown section {step!r}")
        first, separator, last = step.partition("-")
        if not separator:
            last = first
        start_measure = self._section_hierarchy.measure_map.by_id(first)
        end_measure = self._section_hierarchy.measure_map.by_id(last)
        assert start_measure.count is not None and end_measure.count is not None
        if end_measure.count < start_measure.count:
            raise ValueError(f"Flow range {step!r} runs backwards")
        return start_measure.count, end_measure.count, step

    def _flow_length(self, flow_id: str) -> Fraction:
        measures = self._section_hierarchy.measure_map.measures
        total = Fraction(0)
        for step in self._flow_steps[flow_id]:
            if isinstance(step, Gap):
                if step.duration is not None:
                    total += step.duration
                continue
            start, end, _ = self._resolve_step(step)
            first = start - 1
            for measure in measures[first:end]:
                if measure.actual_length is None:
                    raise ValueError(
                        f"Flow {flow_id!r} includes measure {measure.id!r} without actual_length"
                    )
                total += measure.actual_length
        return total

    def _resolve_reference_position(self, at: Any, flow_id: str) -> Coordinate:
        if isinstance(at, Address):
            if at.rendition is not None:
                raise NotImplementedError(
                    "Address rendition qualifiers are not supported: "
                    "occurrence-qualified resolution is not available yet"
                )
            within = getattr(at, "at", None)
            if isinstance(within, Coordinate):
                self._require_reference_unit(within)
            target_id = self._measure_id_for_address(at)
            running = Fraction(0)
            for step in self._flow_steps[flow_id]:
                if isinstance(step, Gap):
                    if step.duration is not None:
                        running += step.duration
                    continue
                start, end, _ = self._resolve_step(step)
                first = start - 1
                for measure in self._section_hierarchy.measure_map.measures[first:end]:
                    if measure.id == target_id:
                        offset = Fraction(0)
                        if isinstance(within, Coordinate):
                            offset = Fraction(within.value)
                        elif isinstance(within, Beat):
                            if not measure.time_signature:
                                raise ValueError(
                                    f"Measure {measure.id!r} has no time signature"
                                )
                            offset = within.offset(
                                BeatPolicy.from_time_signature(measure.time_signature)
                            )
                        return Coordinate(running + offset, TimeUnit.quarters)
                    if measure.actual_length is None:
                        raise ValueError(f"Measure {measure.id!r} has no actual_length")
                    running += measure.actual_length
            raise ValueError(f"Address {at!r} is outside flow {flow_id!r}")
        if isinstance(at, Coordinate):
            self._require_reference_unit(at)
        return Coordinate(
            at.value if isinstance(at, Coordinate) else at,
            TimeUnit.quarters,
        )

    @staticmethod
    def _require_reference_unit(coordinate: Coordinate) -> None:
        if coordinate.unit is not TimeUnit.quarters:
            raise ValueError(
                f"Coordinate unit {coordinate.unit.value!r} does not match "
                f"the expected {TimeUnit.quarters.value!r} reference axis"
            )

    def _measure_id_for_address(self, address: Address) -> str:
        measures = self._section_hierarchy.measure_map.measures
        if isinstance(address, MeasureId):
            if address.is_positional:
                count = int(address.value)
                if not 1 <= count <= len(measures):
                    raise ValueError(f"Measure count {count} is outside the structure")
                return measures[count - 1].id
            return str(address.value)
        if isinstance(address, MeasureNumber):
            matches = [
                measure
                for measure in measures
                if measure.name == address.mn
                and (address.volta is None or measure.volta == address.volta)
            ]
            if len(matches) != 1:
                raise ValueError(f"Measure label {address.mn!r} is not unique")
            return matches[0].id or ""
        raise TypeError(f"Unsupported address type {type(address).__name__}")

    @staticmethod
    def _coordinate_for_timeline(value: Any, timeline: Any) -> Coordinate:
        if isinstance(value, IdCoordinate):
            if value.timeline_id != timeline.id:
                raise ValueError(
                    f"Coordinate timeline {value.timeline_id!r} does not match "
                    f"participant timeline {timeline.id!r}"
                )
            return Coordinate(value.value, value.unit, number_type=value.number_type)
        if isinstance(value, Coordinate):
            return value
        return Coordinate(value, timeline.unit, number_type=timeline.number_type)
