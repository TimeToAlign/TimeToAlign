"""Atomic sections, computed FlowMaps and virtual timelines
(see README, *Atomic sections and computed FlowMaps*).

The expected values are those of the first movement of Beethoven's
Op. 106, each derived by hand in the README.
"""

from __future__ import annotations

import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.core import (
    Coordinate,
    FlowConvention,
    FlowFindingCause,
    FlowFindingKind,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.core.provenance import Belief, Provenance
from timetoalign.timelines import (
    ComputedFlows,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    PlayedFlow,
    Region,
    ResolvedFlowMap,
    Timeline,
    UnknownLengthError,
    VirtualTimeline,
)
from timetoalign.timelines.flowcontrol import (
    Barline,
    Break,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)

Q = TimeUnit.quarters
INFERRED = (Provenance(method="inferred"),)
EDITOR = "https://example.org/people/editor"
REVIEWER = "https://example.org/people/reviewer"
FIXTURES = Path(os.path.commonpath(timetoshare.fixture_paths()))


def q(value: Any) -> Coordinate:
    return Coordinate(Fraction(value), Q)


def _op106() -> ContinuousLogicalTimeline:
    """The four flow-control elements of the first movement on its axis."""
    score = ContinuousLogicalTimeline(length=1644, uid="op106")
    score.flow_control.add_all(
        [
            JumpTarget(q(500), manifestation="volta", id="jump_target_b"),
            Jump(
                q(Fraction(953, 2)),
                [
                    JumpDestination(by=0, passes=(1,), manifestation="volta"),
                    JumpDestination(
                        to="jump_target_b", passes=(2,), manifestation="volta"
                    ),
                ],
                id="jump_a",
            ),
            Jump(
                q(500),
                [JumpDestination(repeat_count=1, manifestation="repeat_end")],
                id="jump_c",
            ),
            Break(q(500), manifestation="volta", id="break_d"),
        ]
    )
    return score


def _repeat() -> ContinuousLogicalTimeline:
    """``|: 8 :| 12``: a repeat sign at 8 on an axis of 12."""
    score = ContinuousLogicalTimeline(length=12, uid="score")
    score.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)], id="repeat"))
    return score


def _repeat_written() -> ContinuousLogicalTimeline:
    """``|: 8 :| 12`` with its sections and its three flows written."""
    score = _repeat()
    score.create_flow_maps_from_flow_control()
    return score


def _state(timeline: Timeline) -> tuple[Any, ...]:
    """Everything the call may change."""
    return (
        dict(timeline._regions),
        dict(timeline._flow_maps),
        dict(timeline._virtual_timelines),
        timeline.to_dict(minimal=False),
    )


class TestOp106:
    """The sections and flows of the Op. 106/i shape."""

    def test_the_atomic_sections_are_regions_stating_their_type(self) -> None:
        score = _op106()
        written = score.create_flow_maps_from_flow_control()
        assert written is not None
        assert [
            (region.id, region.name, region.start, region.end)
            for region in written.sections
        ] == [
            ("region_1", "A", q(0), q(Fraction(953, 2))),
            ("region_2", "B", q(Fraction(953, 2)), q(500)),
            ("region_3", "C", q(500), q(1644)),
        ]
        assert score.list_regions() == ["region_1", "region_2", "region_3"]
        for region in written.sections:
            assert score.get_region(region.id) is region
            assert region.statements == {"type": ["tts:AtomicSection"]}
            assert region.provenance == [
                {
                    "property": "rdf:type",
                    "value": {"id": "tts:AtomicSection"},
                    "method": "inferred",
                }
            ]
            assert region.prefixes == {"tts": timetoshare.NAMESPACE}
            assert region.is_atomic_section
            assert timetoshare.NAMESPACE + "AtomicSection" == timetoshare.ATOMIC_SECTION

    def test_the_three_flows_are_flow_maps_over_the_sections(self) -> None:
        score = _op106()
        written = score.create_flow_maps_from_flow_control()
        assert written is not None
        assert [
            (flow_map.id, flow_map.convention, flow_map.chunks)
            for flow_map in written.flow_maps
        ] == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_1", "region_2", "region_1", "region_3"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_3")),
            (
                "flow_map_3",
                FlowConvention.printed,
                ("region_1", "region_2", "region_3"),
            ),
        ]
        assert score.list_flow_maps() == ["flow_map_1", "flow_map_2", "flow_map_3"]
        for flow_map in written.flow_maps:
            assert score.get_flow_map(flow_map.id) is flow_map
            assert flow_map.timeline == "op106"
            assert flow_map.provenance == INFERRED
            assert flow_map.is_computed
            assert (flow_map.canonical, flow_map.name) == (False, None)

    def test_lengths_and_positions(self) -> None:
        score = _op106()
        written = score.create_flow_maps_from_flow_control()
        assert written is not None
        default, single, printed = (
            flow_map.resolve(score) for flow_map in written.flow_maps
        )
        assert default.length == q(Fraction(4241, 2))
        assert single.length == q(Fraction(3241, 2))
        assert printed.length == q(1644)
        assert default.unfold(Fraction(393, 2)) == [
            q(Fraction(393, 2)),
            q(Fraction(1393, 2)),
        ]
        assert default.unfold(Fraction(961, 2)) == [q(Fraction(961, 2))]
        assert default.unfold(504) == [q(Fraction(1961, 2))]
        assert default.fold(Fraction(1393, 2)) == q(Fraction(393, 2))
        # Section B, the first bracket, is no part of the single flow.
        assert single.unfold(Fraction(961, 2)) == []
        assert single.unfold(504) == [q(Fraction(961, 2))]

    def test_the_manifest_written_is_valid_and_reads_back_equal(self) -> None:
        score = _op106()
        score.create_flow_maps_from_flow_control()
        manifest = score.to_dict()
        timetoshare.validate(manifest)
        assert [
            (entry["id"], entry["convention"], entry["chunks"], entry["provenance"])
            for entry in manifest["flow_maps"]
        ] == [
            (
                "#flow_map_1",
                "default",
                ["#region_1", "#region_2", "#region_1", "#region_3"],
                [{"method": "inferred"}],
            ),
            (
                "#flow_map_2",
                "single",
                ["#region_1", "#region_3"],
                [{"method": "inferred"}],
            ),
            (
                "#flow_map_3",
                "printed",
                ["#region_1", "#region_2", "#region_3"],
                [{"method": "inferred"}],
            ),
        ]
        assert manifest["@context"]["tts"] == timetoshare.NAMESPACE
        read = Timeline.from_dict(manifest)
        assert read._flow_maps == score._flow_maps
        assert read._regions == score._regions
        assert read.to_dict() == manifest


def test_a_region_is_an_atomic_section_however_it_spells_the_type() -> None:
    region = Region(q(0), q(4))
    assert not region.is_atomic_section
    assert region.with_statement("type", timetoshare.ATOMIC_SECTION).is_atomic_section
    renamed = region.with_prefix("format", timetoshare.NAMESPACE)
    assert renamed.with_statement("type", "format:AtomicSection").is_atomic_section
    other = region.with_statement("type", "schema:MusicComposition")
    assert not other.is_atomic_section


@pytest.mark.parametrize(
    "spelled", [timetoshare.ATOMIC_SECTION, "format:AtomicSection"]
)
def test_a_region_whose_type_is_rejected_is_no_atomic_section(spelled: str) -> None:
    section = (
        Region(q(0), q(4))
        .with_prefix("format", timetoshare.NAMESPACE)
        .with_statement("type", spelled)
        .with_statement("type", "schema:MusicComposition")
    )

    def believed(actor: str, belief: str, value: str = spelled) -> Region:
        return section.with_belief(
            "type", value, actor=actor, belief=belief, date="2026-11-25"
        )

    assert section.is_atomic_section
    assert believed(EDITOR, "unlikely").is_atomic_section
    assert believed(EDITOR, "true").is_atomic_section
    assert not believed(EDITOR, "false").is_atomic_section
    # Rejecting the other type leaves the section one.
    assert believed(EDITOR, "false", "schema:MusicComposition").is_atomic_section
    # One actor's rejection stands whatever another believes.
    both = believed(EDITOR, "false").with_belief(
        "type", spelled, actor=REVIEWER, belief="true", date="2026-11-25"
    )
    assert not both.is_atomic_section
    assert both.statements == section.statements


class TestVirtualTimeline:
    """The statement that the default FlowMap was applied."""

    def test_it_states_the_axis_the_flow_map_gives(self) -> None:
        score = _op106()
        score.create_flow_maps_from_flow_control()
        virtual = score.create_virtual_timeline("flow_map_1", name="as played")
        assert virtual == VirtualTimeline(
            "flow_map_1",
            "op106",
            q(Fraction(4241, 2)),
            name="as played",
            id="clt_1",
        )
        assert (virtual.unit, virtual.number_type) == (Q, NumberType.fraction)
        assert virtual.through is None
        assert score.get_virtual_timeline("clt_1") is virtual
        assert score.list_virtual_timelines() == ["clt_1"]
        assert virtual.to_dict() == {
            "id": "clt_1",
            "map": "flow_map_1",
            "timeline": "op106",
            "unit": "quarters",
            "number_type": "fraction",
            "length": [4241, 2],
            "name": "as played",
        }
        assert VirtualTimeline.from_dict(virtual.to_dict()) == virtual

    def test_positions_convert_as_the_flow_map_s_do(self) -> None:
        score = _op106()
        score.create_flow_maps_from_flow_control()
        virtual = score.create_virtual_timeline("flow_map_1")
        assert virtual.unfold(Fraction(393, 2), score) == [
            q(Fraction(393, 2)),
            q(Fraction(1393, 2)),
        ]
        assert virtual.unfold(IdCoordinate(504, Q, timeline_id="op106"), score) == [
            q(Fraction(1961, 2))
        ]
        assert virtual.fold(Fraction(1393, 2), score) == q(Fraction(393, 2))
        assert virtual.fold(
            IdCoordinate(Fraction(1393, 2), Q, timeline_id="clt_1"), score
        ) == q(Fraction(393, 2))
        with pytest.raises(
            ValueError, match="is no position of virtual timeline 'clt_1'"
        ):
            virtual.fold(IdCoordinate(4, Q, timeline_id="op106"), score)
        with pytest.raises(ValueError, match="lies outside the virtual axis"):
            virtual.fold(Fraction(4241, 2), score)

    def test_a_statement_that_is_not_true_of_the_timeline_does_not_join(self) -> None:
        score = _op106()
        score.create_flow_maps_from_flow_control()
        with pytest.raises(ValueError, match="states the length"):
            score.add_virtual_timeline(VirtualTimeline("flow_map_1", "op106", q(1644)))
        with pytest.raises(ValueError, match="which is no FlowMap of timeline 'op106'"):
            score.add_virtual_timeline(VirtualTimeline("flow_map_9", "op106", q(1644)))
        with pytest.raises(KeyError, match="Available FlowMaps"):
            score.create_virtual_timeline("flow_map_9")
        assert score.list_virtual_timelines() == []

    def test_a_negative_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="a coordinate from 0 on"):
            VirtualTimeline("f", "t", q(-1))


class TestGetFlow:
    """The flow computed on the fly, with nothing written."""

    def test_the_three_conventions(self) -> None:
        score = _op106()
        lengths = {
            convention: score.get_flow(convention).length
            for convention in ("default", "single", "printed")
        }
        assert lengths == {
            "default": q(Fraction(4241, 2)),
            "single": q(Fraction(3241, 2)),
            "printed": q(1644),
        }
        flow = score.get_flow()
        assert flow.timeline == "op106"
        assert flow.unfold(Fraction(393, 2)) == [
            q(Fraction(393, 2)),
            q(Fraction(1393, 2)),
        ]
        assert flow.fold(Fraction(1393, 2)) == q(Fraction(393, 2))
        assert (score.list_regions(), score.list_flow_maps()) == ([], [])

    def test_single_finds_the_first_bracket_it_leaves_out(self) -> None:
        score = _op106()
        assert {
            convention: [
                (f.kind, f.coordinate, f.end, f.cause)
                for f in score.get_flow(convention).findings
            ]
            for convention in ("default", "single", "printed")
        } == {
            "default": [],
            "single": [
                (
                    FlowFindingKind.never_played,
                    q(Fraction(953, 2)),
                    q(500),
                    FlowFindingCause.single_playback,
                )
            ],
            "printed": [],
        }
        written = score.create_flow_maps_from_flow_control()
        assert written is not None and written.findings == ()

    def test_a_timeline_without_flow_control_is_played_as_it_stands(self) -> None:
        plain = ContinuousLogicalTimeline(length=12, uid="plain")
        flow = plain.get_flow()
        assert flow.length == q(12)
        assert flow.unfold(5) == [q(5)]

    def test_an_unknown_convention_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _op106().get_flow("backwards")


def _endless() -> ContinuousLogicalTimeline:
    """``Jump 4 {}`` on an axis of 8: a jump back that is always taken."""
    score = ContinuousLogicalTimeline(length=8, uid="score")
    score.flow_control.add(Jump(q(4), [JumpDestination()], id="round"))
    return score


def _found(findings: Any) -> list[tuple[Any, ...]]:
    return [(f.kind, f.coordinate, f.end) for f in findings]


ENDLESS = [
    (FlowFindingKind.endless_loop, q(4), None),
    (FlowFindingKind.never_played, q(4), q(8)),
]


class TestFindings:
    """What the engine found comes back with what is computed."""

    def test_get_flow_returns_the_findings_with_the_flow(self) -> None:
        flow = _endless().get_flow()
        assert type(flow) is PlayedFlow and isinstance(flow, ResolvedFlowMap)
        assert [
            (chunk.source.start, chunk.source.end, chunk.start) for chunk in flow.chunks
        ] == [(q(0), q(4), q(0)), (q(0), q(4), q(4))]
        assert _found(flow.findings) == ENDLESS
        assert flow.findings[0].message == (
            "The jump 'round' at 4 quarters is taken in a state the flow was in "
            "before: the flow never ends."
        )
        assert (flow.length, flow.unfold(1), flow.fold(5)) == (q(8), [q(1), q(5)], q(1))
        assert repr(flow) == (
            "PlayedFlow(timeline='score', 2 chunks, length=8 quarters, 2 findings)"
        )

    @pytest.mark.parametrize("convention", ["single", "printed"])
    def test_going_through_once_or_as_printed_finds_no_loop(
        self, convention: str
    ) -> None:
        flow = _endless().get_flow(convention)
        assert [(chunk.source.start, chunk.source.end) for chunk in flow.chunks] == [
            (q(0), q(8))
        ]
        assert flow.findings == ()

    def test_a_flow_that_plays_through_has_none(self) -> None:
        assert _repeat().get_flow().findings == ()
        written = _repeat().create_flow_maps_from_flow_control()
        assert written is not None and written.findings == ()

    def test_the_written_flows_come_with_the_findings(self) -> None:
        score = _endless()
        written = score.create_flow_maps_from_flow_control()
        assert type(written) is ComputedFlows
        sections, flow_maps, findings = written
        assert [(r.id, r.name, r.start, r.end) for r in sections] == [
            ("region_1", "A", q(0), q(4)),
            ("region_2", "B", q(4), q(8)),
        ]
        assert [(f.id, f.convention, f.chunks) for f in flow_maps] == [
            ("flow_map_1", FlowConvention.default, ("region_1", "region_1")),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_2")),
            ("flow_map_3", FlowConvention.printed, ("region_1", "region_2")),
        ]
        assert _found(findings) == ENDLESS
        assert findings is written.findings


class TestWhatIsWritten:
    """When FlowMaps are written, and when nothing is."""

    def test_single_and_printed_are_written_only_where_they_differ(self) -> None:
        score = ContinuousLogicalTimeline(length=12, uid="score")
        score.flow_control.add(SectionBreak(q(8)))
        written = score.create_flow_maps_from_flow_control()
        assert written is not None
        assert [(r.name, r.start, r.end) for r in written.sections] == [
            ("A", q(0), q(8)),
            ("B", q(8), q(12)),
        ]
        assert [(f.convention, f.chunks) for f in written.flow_maps] == [
            (FlowConvention.default, ("region_1", "region_2"))
        ]
        assert score.list_flow_maps() == ["flow_map_1"]

    def test_a_timeline_without_flow_control_writes_nothing(self) -> None:
        plain = ContinuousLogicalTimeline(length=12, uid="plain")
        before = _state(plain)
        assert plain.create_flow_maps_from_flow_control() is None
        assert _state(plain) == before
        assert (plain.list_regions(), plain.list_flow_maps()) == ([], [])
        assert "flow_maps" not in plain.to_dict()

    def test_an_unknown_length_is_refused(self) -> None:
        score = ContinuousLogicalTimeline(length=None, uid="score")
        score.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)]))
        with pytest.raises(UnknownLengthError) as raised:
            score.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "Cannot play its flow control: the length of timeline 'score' is unknown"
        )
        assert (score.list_regions(), score.list_flow_maps()) == ([], [])


def _named_sections() -> ContinuousLogicalTimeline:
    """``|: 8 :| 12`` with its sections written and three things naming them."""
    score = _repeat()
    score.create_flow_maps_from_flow_control()
    theme = score.create_region(2, 6, id="theme")
    score.replace_region(theme.with_statement("dcterms:relation", {"id": "region_1"}))
    score.create_flow_map(["region_2", "region_1"], id="backwards")
    score.create_virtual_timeline("flow_map_1", id="played")
    return score


def _sections(written: ComputedFlows | None) -> list[tuple[Any, ...]]:
    assert written is not None
    return [(r.id, r.name, r.start, r.end) for r in written.sections]


def _flow_maps(written: ComputedFlows | None) -> list[tuple[Any, ...]]:
    assert written is not None
    return [(f.id, f.convention, f.chunks) for f in written.flow_maps]


class TestWrittenAgain:
    """Computed again, nothing is taken away and no ID names another stretch."""

    def test_the_first_writing(self) -> None:
        score = _repeat()
        first = score.create_flow_maps_from_flow_control()
        assert _sections(first) == [
            ("region_1", "A", q(0), q(8)),
            ("region_2", "B", q(8), q(12)),
        ]
        assert _flow_maps(first) == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_1", "region_1", "region_2"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_2")),
            ("flow_map_3", FlowConvention.printed, ("region_1", "region_2")),
        ]
        assert _named_sections().get_virtual_timeline("played").length == q(20)

    def test_unchanged_flow_control_writes_the_same_and_asks_for_no_actor(
        self,
    ) -> None:
        score = _op106()
        first = score.create_flow_maps_from_flow_control()
        virtual = score.create_virtual_timeline("flow_map_1")
        before = _state(score)
        second = score.create_flow_maps_from_flow_control()
        assert second == first
        assert _state(score) == before
        assert score.list_flow_maps() == ["flow_map_1", "flow_map_2", "flow_map_3"]
        assert score.get_virtual_timeline(virtual.id) == virtual

    def test_a_divided_section_stays_under_its_id_and_is_rejected(self) -> None:
        score = _named_sections()
        whole = score.get_region("region_1")
        score.flow_control.add(JumpTarget(q(4), id="sign"))

        second = score.create_flow_maps_from_flow_control(
            actor=EDITOR, date="2026-11-25"
        )

        assert _sections(second) == [
            ("region_3", "A", q(0), q(4)),
            ("region_4", "B", q(4), q(8)),
            ("region_2", "C", q(8), q(12)),
        ]
        assert all(region.is_atomic_section for region in second.sections)
        assert _flow_maps(second) == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_3", "region_4", "region_3", "region_4", "region_2"),
            ),
            (
                "flow_map_2",
                FlowConvention.single,
                ("region_3", "region_4", "region_2"),
            ),
            (
                "flow_map_3",
                FlowConvention.printed,
                ("region_3", "region_4", "region_2"),
            ),
        ]
        assert score.list_regions() == [
            "region_1",
            "region_2",
            "theme",
            "region_3",
            "region_4",
        ]
        # The section that is none any more: same ID, interval and name.
        kept = score.get_region("region_1")
        assert (kept.name, kept.start, kept.end) == ("A", q(0), q(8))
        assert (kept.name, kept.start, kept.end) == (whole.name, whole.start, whole.end)
        assert not kept.is_atomic_section
        assert kept.statements == {"type": ["tts:AtomicSection"]}
        assert kept.provenance == [
            {
                "property": "rdf:type",
                "value": {"id": "tts:AtomicSection"},
                "method": "inferred",
                "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}],
            }
        ]
        # What named it names the same stretch.
        assert score.get_region("theme").statements == {
            "dcterms:relation": [{"id": "region_1"}]
        }
        backwards = score.get_flow_map("backwards")
        assert backwards.chunks == ("region_2", "region_1")
        assert [
            (chunk.source.start, chunk.source.end)
            for chunk in backwards.resolve(score).chunks
        ] == [(q(8), q(12)), (q(0), q(8))]
        assert score.get_virtual_timeline("played").length == q(20)
        assert score.list_flow_maps() == [
            "flow_map_1",
            "flow_map_2",
            "flow_map_3",
            "backwards",
        ]
        timetoshare.validate(score.to_dict())

        # Nothing changed since: no actor is needed, nothing is rejected again.
        before = _state(score)
        assert score.create_flow_maps_from_flow_control() == second
        assert _state(score) == before

    def test_a_stretch_that_comes_back_gets_a_new_region(self) -> None:
        score = _named_sections()
        score.flow_control.add(JumpTarget(q(4), id="sign"))
        score.create_flow_maps_from_flow_control(actor=EDITOR, date="2026-11-25")
        rejected_before = score.get_region("region_1")
        score.flow_control.with_belief("sign", EDITOR, "false", date="2026-11-26")

        third = score.create_flow_maps_from_flow_control(
            actor=REVIEWER, date="2026-11-26"
        )

        assert _sections(third) == [
            ("region_5", "A", q(0), q(8)),
            ("region_2", "B", q(8), q(12)),
        ]
        assert _flow_maps(third) == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_5", "region_5", "region_2"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_5", "region_2")),
            ("flow_map_3", FlowConvention.printed, ("region_5", "region_2")),
        ]
        # The Region rejected before is as it was; the two halves follow it.
        assert score.get_region("region_1") is rejected_before
        assert [
            (region.id, region.is_atomic_section)
            for region in score._regions.values()
            if region.id != "theme"
        ] == [
            ("region_1", False),
            ("region_2", True),
            ("region_3", False),
            ("region_4", False),
            ("region_5", True),
        ]
        assert [
            belief["actor"]
            for region_id in ("region_3", "region_4")
            for entry in score.get_region(region_id).provenance
            for belief in entry["beliefs"]
        ] == [REVIEWER, REVIEWER]
        timetoshare.validate(score.to_dict())

    def test_a_shortened_section_stays_and_the_flow_maps_keep_their_ids(self) -> None:
        score = _named_sections()
        score.flow_control.add(Break(q(10), id="stop"))

        second = score.create_flow_maps_from_flow_control(
            actor=EDITOR, date="2026-11-25"
        )

        assert _sections(second) == [
            ("region_1", "A", q(0), q(8)),
            ("region_3", "B", q(8), q(10)),
            ("region_4", "C", q(10), q(12)),
        ]
        assert _flow_maps(second) == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_1", "region_1", "region_3"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_3")),
            (
                "flow_map_3",
                FlowConvention.printed,
                ("region_1", "region_3", "region_4"),
            ),
        ]
        assert [(f.kind, f.coordinate, f.end) for f in second.findings] == [
            (FlowFindingKind.flow_ends_at_break, q(10), None),
            (FlowFindingKind.never_played, q(10), q(12)),
        ]
        gone = score.get_region("region_2")
        assert (gone.name, gone.start, gone.end) == ("B", q(8), q(12))
        assert not gone.is_atomic_section
        assert score.get_region("region_1").provenance == [
            {
                "property": "rdf:type",
                "value": {"id": "tts:AtomicSection"},
                "method": "inferred",
            }
        ]
        assert score.get_flow_map("backwards").resolve(score).length == q(12)
        assert score.get_virtual_timeline("played") == VirtualTimeline(
            "flow_map_1", "score", q(18), id="played"
        )
        timetoshare.validate(score.to_dict())

    def test_without_an_actor_changed_sections_are_refused(self) -> None:
        score = _named_sections()
        score.flow_control.add(JumpTarget(q(4), id="sign"))
        before = _state(score)
        with pytest.raises(ValueError) as raised:
            score.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "timeline 'score' holds the atomic section(s) ['region_1'], whose "
            "stretches its flow control does not give any more: they stay and are "
            "rejected as atomic sections, which needs the actor who recomputes "
            "(actor=)"
        )
        assert _state(score) == before
        with pytest.raises(ValueError, match="'2026-13-01' is no date of the calendar"):
            score.create_flow_maps_from_flow_control(actor=EDITOR, date="2026-13-01")
        assert _state(score) == before

    def test_a_flow_map_of_single_stays_where_it_comes_to_equal_default(self) -> None:
        score = _named_sections()
        score.flow_control.with_belief("repeat", EDITOR, "false", date="2026-11-25")
        score.flow_control.add(SectionBreak(q(8), id="movement"))

        second = score.create_flow_maps_from_flow_control()

        assert _sections(second) == [
            ("region_1", "A", q(0), q(8)),
            ("region_2", "B", q(8), q(12)),
        ]
        assert _flow_maps(second) == [
            ("flow_map_1", FlowConvention.default, ("region_1", "region_2")),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_2")),
            ("flow_map_3", FlowConvention.printed, ("region_1", "region_2")),
        ]
        assert score.get_virtual_timeline("played").length == q(12)


class TestAnotherOrigin:
    """FlowMaps that were not computed stand untouched beside the computed ones."""

    def test_a_computed_flow_map_of_another_convention_name(self) -> None:
        score = _repeat()
        score.create_flow_maps_from_flow_control()
        own = score.create_flow_map(
            ["region_1"], id="exposition", convention="as recorded", provenance=INFERRED
        )
        written = score.create_flow_maps_from_flow_control()
        assert [f.id for f in written.flow_maps] == [
            "flow_map_1",
            "flow_map_2",
            "flow_map_3",
        ]
        assert score.get_flow_map("exposition") is own

    def test_flow_maps_naming_a_convention_without_provenance(self) -> None:
        manifest = _repeat_written().to_dict()
        for flow_map in manifest["flow_maps"]:
            del flow_map["provenance"]
        score = Timeline.from_dict(manifest)
        regions = dict(score._regions)
        stated = {
            flow_map_id: score.get_flow_map(flow_map_id)
            for flow_map_id in ("flow_map_1", "flow_map_2", "flow_map_3")
        }
        assert not any(flow_map.is_computed for flow_map in stated.values())

        written = score.create_flow_maps_from_flow_control()

        assert _sections(written) == [
            ("region_1", "A", q(0), q(8)),
            ("region_2", "B", q(8), q(12)),
        ]
        assert score._regions == regions
        assert _flow_maps(written) == [
            (
                "flow_map_4",
                FlowConvention.default,
                ("region_1", "region_1", "region_2"),
            ),
            ("flow_map_5", FlowConvention.single, ("region_1", "region_2")),
            ("flow_map_6", FlowConvention.printed, ("region_1", "region_2")),
        ]
        for flow_map_id, flow_map in stated.items():
            assert score.get_flow_map(flow_map_id) is flow_map
        timetoshare.validate(score.to_dict())

    def test_the_format_s_computed_exposition_is_written_again_unchanged(self) -> None:
        score = Timeline.from_file(FIXTURES / "flow" / "sonata_exposition.yaml")
        before = _state(score)
        written = score.create_flow_maps_from_flow_control()
        assert _sections(written) == [
            ("region_1", "A", q(0), q(12)),
            ("region_2", "B", q(12), q(16)),
            ("region_3", "C", q(16), q(24)),
        ]
        assert _flow_maps(written) == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_1", "region_2", "region_1", "region_3"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_3")),
            (
                "flow_map_3",
                FlowConvention.printed,
                ("region_1", "region_2", "region_3"),
            ),
        ]
        assert _state(score) == before

    def test_a_jump_going_on_at_a_section_keeps_its_place(self) -> None:
        score = _repeat()
        score.create_flow_maps_from_flow_control()
        back = score.flow_control.add(
            Jump(q(12), [JumpDestination(to="region_2", repeat_count=1)], id="back")
        )
        score.flow_control.add(JumpTarget(q(10), id="sign"))

        written = score.create_flow_maps_from_flow_control(
            actor=EDITOR, date="2026-11-25"
        )

        assert _sections(written) == [
            ("region_1", "A", q(0), q(8)),
            ("region_3", "B", q(8), q(10)),
            ("region_4", "C", q(10), q(12)),
        ]
        gone = score.get_region("region_2")
        assert (gone.start, gone.end, gone.is_atomic_section) == (q(8), q(12), False)
        assert score.flow_control.place_of(back.targets[0], back) == q(8)
        assert _flow_maps(written)[0] == (
            "flow_map_1",
            FlowConvention.default,
            (
                "region_1",
                "region_1",
                "region_3",
                "region_4",
                "region_3",
                "region_4",
            ),
        )

    def test_a_flow_map_by_hand_over_other_regions_does_not_stand_in_the_way(
        self,
    ) -> None:
        score = _repeat()
        score.create_region(2, 6, id="theme")
        by_hand = score.create_flow_map(["theme", "theme"], id="twice")
        score.create_flow_maps_from_flow_control()
        written = score.create_flow_maps_from_flow_control()
        assert written is not None
        assert [f.id for f in written.flow_maps] == [
            "flow_map_1",
            "flow_map_2",
            "flow_map_3",
        ]
        assert score.get_flow_map("twice") is by_hand


def _sonata() -> tuple[ContinuousLogicalTimeline, Timeline, Timeline]:
    """A root holding two children, each with a repeat sign of its own."""
    root = ContinuousLogicalTimeline(length=40, uid="sonata")
    movement = root.create_child(12, 4, uid="movement", name="Allegro")
    finale = root.create_child(12, 20, uid="finale", name="Finale")
    movement.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)]))
    finale.flow_control.add(Jump(q(6), [JumpDestination(repeat_count=1)]))
    return root, movement, finale


def _maps(root: Timeline) -> list[tuple[str, Any, Any, tuple[Any, ...]]]:
    return [
        (flow_map.id, flow_map.convention, flow_map.name, flow_map.chunks)
        for flow_map in (root.get_flow_map(i) for i in root.list_flow_maps())
    ]


MOVEMENT_MAPS = [
    (
        "flow_map_1",
        FlowConvention.default,
        "Allegro, default",
        ("region_1", "region_1", "region_2"),
    ),
    ("flow_map_2", FlowConvention.single, "Allegro, single", ("region_1", "region_2")),
    (
        "flow_map_3",
        FlowConvention.printed,
        "Allegro, printed",
        ("region_1", "region_2"),
    ),
]
FINALE_MAPS = [
    (
        "flow_map_4",
        FlowConvention.default,
        "Finale, default",
        ("region_3", "region_3", "region_4"),
    ),
    ("flow_map_5", FlowConvention.single, "Finale, single", ("region_3", "region_4")),
    ("flow_map_6", FlowConvention.printed, "Finale, printed", ("region_3", "region_4")),
]


class TestChild:
    """A child's sections on the child, its FlowMaps on the root."""

    def test_the_sections_are_the_childs_and_the_flow_maps_the_roots(self) -> None:
        root, movement, _ = _sonata()
        written = movement.create_flow_maps_from_flow_control()
        assert written is not None and written.findings == ()
        assert [(r.id, r.name, str(r.as_interval)) for r in written.sections] == [
            ("region_1", "A", "[0, 8) quarters"),
            ("region_2", "B", "[8, 12) quarters"),
        ]
        assert movement.list_regions() == ["region_1", "region_2"]
        assert root.list_regions() == []
        assert movement.list_flow_maps() == []
        assert _maps(root) == MOVEMENT_MAPS
        assert {flow_map.timeline for flow_map in written.flow_maps} == {"sonata"}
        assert all(flow_map.provenance == INFERRED for flow_map in written.flow_maps)

    def test_the_flow_map_resolves_at_the_childs_offset(self) -> None:
        root, movement, _ = _sonata()
        movement.create_flow_maps_from_flow_control()
        resolved = root.get_flow_map("flow_map_1").resolve(root)
        assert [
            (chunk.source.start, chunk.source.end) for chunk in resolved.chunks
        ] == [
            (q(4), q(12)),
            (q(4), q(12)),
            (q(12), q(16)),
        ]
        assert resolved.length == q(20)
        virtual = root.create_virtual_timeline("flow_map_1", id="allegro_played")
        assert (virtual.timeline, virtual.map, virtual.length) == (
            "sonata",
            "flow_map_1",
            q(20),
        )

    def test_two_children_stand_side_by_side_and_survive_a_manifest(self) -> None:
        root, movement, finale = _sonata()
        movement.create_flow_maps_from_flow_control()
        finale.create_flow_maps_from_flow_control()
        assert finale.list_regions() == ["region_3", "region_4"]
        assert [
            str(finale.get_region(i).as_interval) for i in finale.list_regions()
        ] == [
            "[0, 6) quarters",
            "[6, 12) quarters",
        ]
        assert _maps(root) == MOVEMENT_MAPS + FINALE_MAPS
        read = Timeline.from_dict(root.to_dict())
        assert _maps(read) == MOVEMENT_MAPS + FINALE_MAPS
        assert read.get_child("movement").list_regions() == ["region_1", "region_2"]
        assert read.get_child("finale").list_regions() == ["region_3", "region_4"]

    def test_written_again_the_virtual_timeline_takes_the_new_length(self) -> None:
        root, movement, _ = _sonata()
        movement.create_flow_maps_from_flow_control()
        root.create_virtual_timeline("flow_map_1", id="allegro_played")
        movement.flow_control.add(Jump(q(12), [JumpDestination(repeat_count=1)]))
        again = movement.create_flow_maps_from_flow_control()
        assert again is not None
        assert [r.id for r in again.sections] == ["region_1", "region_2"]
        default = root.get_flow_map("flow_map_1")
        assert (default.name, default.chunks) == (
            "Allegro, default",
            ("region_1", "region_1", "region_2", "region_1", "region_1", "region_2"),
        )
        assert root.get_virtual_timeline("allegro_played").length == q(40)

    def test_written_again_a_divided_section_is_rejected(self) -> None:
        root, movement, finale = _sonata()
        movement.create_flow_maps_from_flow_control()
        finale.create_flow_maps_from_flow_control()
        movement.flow_control.add(JumpTarget(q(4)))
        before = _state(root)
        with pytest.raises(ValueError):
            movement.create_flow_maps_from_flow_control()
        assert _state(root) == before
        again = movement.create_flow_maps_from_flow_control(
            actor=EDITOR, date="2026-11-25"
        )
        assert again is not None
        assert [(r.id, r.name, str(r.as_interval)) for r in again.sections] == [
            ("region_5", "A", "[0, 4) quarters"),
            ("region_6", "B", "[4, 8) quarters"),
            ("region_2", "C", "[8, 12) quarters"),
        ]
        assert movement.get_region("region_1").is_atomic_section is False
        assert _maps(root)[0] == (
            "flow_map_1",
            FlowConvention.default,
            "Allegro, default",
            ("region_5", "region_6", "region_5", "region_6", "region_2"),
        )
        assert _maps(root)[3:] == FINALE_MAPS


# region A timeline played with its tree


def _work(*, dc: bool = True) -> tuple[Timeline, Timeline, Timeline]:
    """The root ``work`` with ``allegro`` (holding ``codetta``) and ``finale``.

    ``allegro`` states the repeat (``start`` at 0, ``repeat`` at 8),
    ``codetta`` a barline at its end, the root the da capo ``dc`` at 32.
    """
    work = ContinuousLogicalTimeline(length=40, uid="work")
    allegro = work.create_child(12, 4, uid="allegro", name="Allegro")
    finale = work.create_child(12, 20, uid="finale", name="Finale")
    codetta = allegro.create_child(4, 8, uid="codetta")
    allegro.flow_control.add_all(
        [
            JumpTarget(q(0), manifestation="repeat_start", id="start"),
            Jump(
                q(8),
                [
                    JumpDestination(
                        to="start", repeat_count=1, manifestation="repeat_end"
                    )
                ],
                id="repeat",
            ),
        ]
    )
    codetta.flow_control.add(
        Barline(q(4), manifestation="final_barline", id="codetta_end")
    )
    if dc:
        work.flow_control.add(_dc())
    return work, allegro, finale


def _dc() -> Jump:
    return Jump(
        q(32),
        [
            JumpDestination(
                to="work", repeat_count=1, reset=False, manifestation="da_capo"
            )
        ],
        id="dc",
    )


def _runs(flow: PlayedFlow) -> list[tuple[Fraction, Fraction]]:
    return [(chunk.source.start.value, chunk.source.end.value) for chunk in flow.chunks]


def _placed(elements: Any) -> list[tuple[str, Fraction]]:
    return [(element.id, element.coordinate.value) for element in elements]


F = Fraction


class TestTree:
    """Flow control stated anywhere in a tree, played on one axis."""

    def test_the_view_gathers_the_subtree_on_the_axis(self) -> None:
        work, allegro, finale = _work()
        assert _placed(work.tree_flow_control.elements) == [
            ("start", F(4)),
            ("repeat", F(12)),
            ("codetta_end", F(16)),
            ("dc", F(32)),
        ]
        assert _placed(allegro.tree_flow_control.elements) == [
            ("start", F(0)),
            ("repeat", F(8)),
            ("codetta_end", F(12)),
        ]
        assert finale.tree_flow_control.elements == ()
        assert _placed(work.flow_control.elements) == [("dc", F(32))]
        assert _placed(allegro.flow_control.elements) == [
            ("start", F(0)),
            ("repeat", F(8)),
        ]

    def test_the_root_plays_its_tree(self) -> None:
        work, _, _ = _work()
        before = _state(work)
        flow = work.get_flow()
        assert _runs(flow) == [(F(0), F(12)), (F(4), F(32)), (F(0), F(40))]
        assert flow.length == q(80)
        assert flow.findings == ()
        for convention in ("single", "printed"):
            once = work.get_flow(convention)
            assert (_runs(once), once.length) == ([(F(0), F(40))], q(40))
        assert _state(work) == before

    def test_the_same_elements_stated_on_the_root_play_the_same(self) -> None:
        work, _, _ = _work()
        flat = ContinuousLogicalTimeline(length=40, uid="work")
        flat.flow_control.add_all(
            [
                JumpTarget(q(4), manifestation="repeat_start", id="start"),
                Jump(
                    q(12),
                    [
                        JumpDestination(
                            to="start", repeat_count=1, manifestation="repeat_end"
                        )
                    ],
                    id="repeat",
                ),
                Barline(q(16), manifestation="final_barline", id="codetta_end"),
                _dc(),
            ]
        )
        assert work.tree_flow_control.elements == flat.flow_control.elements
        for convention in FlowConvention:
            gathered, stated = work.get_flow(convention), flat.get_flow(convention)
            assert (_runs(gathered), gathered.length) == (
                _runs(stated),
                stated.length,
            )

    def test_a_child_plays_its_subtree(self) -> None:
        _, allegro, _ = _work()
        flow = allegro.get_flow()
        assert _runs(flow) == [(F(0), F(8)), (F(0), F(12))]
        assert (flow.timeline, flow.length, flow.findings) == ("allegro", q(20), ())
        alone = ContinuousLogicalTimeline(length=12, uid="alone")
        alone.flow_control.add_all(
            [
                JumpTarget(q(0), id="start"),
                Jump(q(8), [JumpDestination(to="start", repeat_count=1)], id="repeat"),
            ]
        )
        assert (_runs(alone.get_flow()), alone.get_flow().length) == (
            _runs(flow),
            flow.length,
        )

    def test_a_destination_without_place_goes_back_to_the_nearest_section_boundary(
        self,
    ) -> None:
        def tree(
            root_breaks: tuple[int, ...] = (), child_breaks: tuple[int, ...] = ()
        ) -> tuple[Timeline, Timeline]:
            work = ContinuousLogicalTimeline(length=40, uid="work")
            allegro = work.create_child(12, 4, uid="allegro")
            allegro.flow_control.add_all(
                [
                    Jump(q(8), [JumpDestination(repeat_count=1)], id="repeat"),
                    *(SectionBreak(q(at)) for at in child_breaks),
                ]
            )
            work.flow_control.add_all(SectionBreak(q(at)) for at in root_breaks)
            return work, allegro

        work, allegro = tree()
        flow = work.get_flow()
        assert (_runs(flow), flow.length) == ([(F(0), F(12)), (F(4), F(40))], q(48))
        assert _runs(allegro.get_flow()) == [(F(0), F(8)), (F(0), F(12))]
        back_to = {
            ((2,), ()): 4,
            ((6,), ()): 6,
            ((), (3,)): 7,
            ((6,), (3,)): 7,
            ((12,), ()): 4,
        }
        for (root_breaks, child_breaks), back in back_to.items():
            work, _ = tree(root_breaks, child_breaks)
            assert _runs(work.get_flow()) == [(F(0), F(12)), (F(back), F(40))], (
                root_breaks,
                child_breaks,
            )
        assert _runs(tree((), (3,))[1].get_flow()) == [(F(0), F(8)), (F(3), F(12))]
        assert _runs(tree((6,), ())[1].get_flow()) == [(F(0), F(8)), (F(0), F(12))]
        flat = ContinuousLogicalTimeline(length=40, uid="work")
        flat.flow_control.add(Jump(q(12), [JumpDestination(repeat_count=1)]))
        flow = flat.get_flow()
        assert (_runs(flow), flow.length) == ([(F(0), F(12)), (F(0), F(40))], q(52))

    def test_a_root_whose_flow_control_lies_in_descendants_gets_the_sections(
        self,
    ) -> None:
        work, allegro, _ = _work(dc=False)
        written = work.create_flow_maps_from_flow_control()
        assert written is not None and written.findings == ()
        assert [(r.id, r.name, str(r.as_interval)) for r in written.sections] == [
            ("region_1", "A", "[0, 4) quarters"),
            ("region_2", "B", "[4, 12) quarters"),
            ("region_3", "C", "[12, 40) quarters"),
        ]
        assert work.list_regions() == ["region_1", "region_2", "region_3"]
        assert allegro.list_regions() == []
        assert _maps(work) == [
            (
                "flow_map_1",
                FlowConvention.default,
                None,
                ("region_1", "region_2", "region_2", "region_3"),
            ),
            (
                "flow_map_2",
                FlowConvention.single,
                None,
                ("region_1", "region_2", "region_3"),
            ),
            (
                "flow_map_3",
                FlowConvention.printed,
                None,
                ("region_1", "region_2", "region_3"),
            ),
        ]
        timetoshare.validate(work.to_dict())

    def test_one_jump_lies_at_a_position_of_the_roots_axis(self) -> None:
        work, allegro, _ = _work(dc=False)
        codetta = allegro.get_child("codetta")
        before = _state(work)
        refusal = _second("skip", "12 quarters", "work", "repeat", "allegro")
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            work.flow_control.add(Jump(q(12), [JumpDestination(by=4)], id="skip"))
        assert _state(work) == before
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            work.flow_control.add_all(
                [
                    Barline(q(2), id="bar"),
                    Jump(q(12), [JumpDestination(by=4)], id="skip"),
                ]
            )
        assert _state(work) == before
        codetta.flow_control.add(Jump(q(2), [JumpDestination(by=2)], id="on"))
        timetoshare.validate(work.to_dict())
        refusal = _second("again", "0 quarters", "codetta", "repeat", "allegro")
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            codetta.flow_control.add(Jump(q(0), [JumpDestination(by=2)], id="again"))
        rejected = Jump(q(12), [JumpDestination(by=4)], id="skip").with_belief(
            Belief(actor=EDITOR, belief="false", date="2026-11-25")
        )
        assert work.flow_control.add(rejected).is_rejected
        timetoshare.validate(work.to_dict())
        assert _runs(work.get_flow()) == [(F(0), F(12)), (F(4), F(14)), (F(16), F(40))]

    def test_a_child_bringing_a_jump_to_a_taken_position_is_refused(self) -> None:
        root = ContinuousLogicalTimeline(length=40, uid="root")
        root.flow_control.add(Jump(q(12), [JumpDestination(by=4)], id="skip"))
        child = ContinuousLogicalTimeline(length=12, uid="movement")
        child.flow_control.add(
            Jump(q(8), [JumpDestination(repeat_count=1)], id="repeat")
        )
        before = (root.to_dict(), child.to_dict())
        refusal = _second("repeat", "8 quarters", "movement", "skip", "root")
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            root.add_child(child, 4)
        assert (root.n_children, (root.to_dict(), child.to_dict())) == (0, before)
        root.add_child(child, 5)
        assert _placed(root.tree_flow_control.jumps) == [
            ("skip", F(12)),
            ("repeat", F(13)),
        ]
        timetoshare.validate(root.to_dict())

    def test_positions_on_a_float_axis_are_summed_exactly(self) -> None:
        root = ContinuousPhysicalTimeline(length=2.0, uid="root")
        inner = root.create_child(1.0, 0.2, uid="ch").create_child(0.5, 0.5, uid="g")
        inner.flow_control.add(Jump(s(0.2), [JumpDestination(by=0.1)], id="inner"))
        assert (0.2 + 0.5) + 0.2 == 0.8999999999999999
        assert _placed(root.tree_flow_control.jumps) == [("inner", 0.9)]
        written = root.to_dict()
        outer = {"coordinate": 0.9, "id": "#outer", "role": "jump"}
        outer["targets"] = [{"by": 0.1}]
        written["timelines"][0]["flow_control"] = [outer]
        with pytest.raises(timetoshare.SchemaValidationError) as raised:
            timetoshare.validate(written)
        assert raised.value.code == "jump_repeated"
        refusal = _second("outer", "0.9 seconds", "root", "inner", "g")
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            root.flow_control.add(Jump(s(0.9), [JumpDestination(by=0.1)], id="outer"))

    def test_a_float_axis_that_cannot_separate_two_jumps_refuses_playing(
        self,
    ) -> None:
        root = ContinuousPhysicalTimeline(length=1.0, uid="root")
        child = root.create_child(0.5, 0.1, uid="ch")
        child.flow_control.add(Jump(s(0.2), [JumpDestination(by=0.1)], id="inner"))
        root.flow_control.add(
            Jump(s(0.30000000000000004), [JumpDestination(by=0.1)], id="outer")
        )
        assert Fraction(0.1) + Fraction(0.2) != Fraction(0.30000000000000004)
        assert float(Fraction(0.1) + Fraction(0.2)) == 0.30000000000000004
        timetoshare.validate(root.to_dict())
        refusal = (
            "jump 'outer' lies at 0.30000000000000004 seconds, where jump 'inner' "
            "lies already: one jump lies at a coordinate, and several signs there "
            "are its destinations"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(refusal)}$"):
            root.get_flow()


def _second(jump: str, at: str, holder: str, other: str, other_holder: str) -> str:
    """The refusal of a second Jump at a position of the root's axis."""
    return (
        f"jump '{jump}' is a second jump at {at} of timeline '{holder}', after jump "
        f"'{other}' of timeline '{other_holder}' at the same position on their "
        "root's axis: several signs at one place are destinations of one jump"
    )


def s(value: float) -> Coordinate:
    return Coordinate(value, TimeUnit.seconds)


def _op106_manifest() -> dict[str, Any]:
    """The timelines and flow control of Op. 106 as its manifest states them."""

    def barline(identifier: str, at: list[int], manifestation: str) -> dict[str, Any]:
        return {
            "coordinate": at,
            "id": identifier,
            "manifestation": manifestation,
            "role": "barline",
        }

    def movement(
        identifier: str,
        name: str,
        length: list[int],
        offset: list[int],
        flow_control: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "flow_control": flow_control,
            "id": identifier,
            "length": length,
            "locked": True,
            "name": name,
            "offset": offset,
            "unit": "quarters",
        }

    return {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "children": [
                    movement(
                        "#clt_2",
                        "Allegro",
                        [1644, 1],
                        [0, 1],
                        [
                            {
                                "coordinate": [953, 2],
                                "id": "#jump_1",
                                "role": "jump",
                                "targets": [
                                    {
                                        "by": [0, 1],
                                        "manifestation": "volta",
                                        "passes": [1],
                                    },
                                    {
                                        "manifestation": "volta",
                                        "passes": [2],
                                        "to": "#jump_target_1",
                                    },
                                ],
                            },
                            {
                                "coordinate": [500, 1],
                                "id": "#break_1",
                                "manifestation": "volta",
                                "role": "break",
                            },
                            {
                                "coordinate": [500, 1],
                                "id": "#jump_2",
                                "role": "jump",
                                "targets": [
                                    {"manifestation": "repeat_end", "repeat_count": 1}
                                ],
                            },
                            {
                                "coordinate": [500, 1],
                                "id": "#jump_target_1",
                                "manifestation": "volta",
                                "role": "jump_target",
                            },
                            barline("#barline_1", [824, 1], "double_barline"),
                            barline("#barline_2", [1644, 1], "final_barline"),
                        ],
                    ),
                    movement(
                        "#clt_3",
                        "Scherzo. Assai vivace",
                        [516, 1],
                        [1644, 1],
                        [
                            barline("#barline_3", [138, 1], "double_barline"),
                            barline("#barline_4", [516, 1], "final_barline"),
                        ],
                    ),
                    movement(
                        "#clt_4",
                        "Adagio sostenuto",
                        [561, 1],
                        [2160, 1],
                        [barline("#barline_5", [561, 1], "final_barline")],
                    ),
                    movement(
                        "#clt_5",
                        "Introduzione. Largo \u2013 Allegro risoluto",
                        [4949, 4],
                        [2721, 1],
                        [
                            barline("#barline_6", [75, 4], "double_barline"),
                            barline("#barline_7", [269, 4], "double_barline"),
                            barline("#barline_8", [4949, 4], "final_barline"),
                        ],
                    ),
                ],
                "id": "#clt_1",
                "length": [15833, 4],
                "name": "Op. 106",
                "unit": "quarters",
            }
        ],
    }


def test_op106_the_root_plays_the_first_movement_unfolded_then_the_others() -> None:
    manifest = _op106_manifest()
    timetoshare.validate(manifest)
    sonata = Timeline.from_dict(manifest)
    flow = sonata.get_flow()
    assert _runs(flow) == [(F(0), F(500)), (F(0), F(953, 2)), (F(500), F(15833, 4))]
    assert flow.length == q(F(15833, 4) - 1644 + F(4241, 2)) == q(F(17739, 4))
    assert flow.findings == ()
    first = sonata.get_child("clt_2").get_flow()
    assert (first.length, first.findings) == (q(F(4241, 2)), ())


# endregion
