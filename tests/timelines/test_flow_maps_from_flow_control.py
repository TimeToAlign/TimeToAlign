"""Atomic sections, computed FlowMaps and virtual timelines
(see README, *Atomic sections and computed FlowMaps*).

The expected values are the ones stated for the first movement of
Beethoven's Op. 106, each derived by hand in the README.
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.core import (
    Coordinate,
    FlowConvention,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.core.provenance import Provenance
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    Region,
    Timeline,
    UnknownLengthError,
    VirtualTimeline,
)
from timetoalign.timelines.flowcontrol import (
    Break,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)

Q = TimeUnit.quarters
INFERRED = (Provenance(method="inferred"),)
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

    def test_a_timeline_without_flow_control_is_played_as_it_stands(self) -> None:
        plain = ContinuousLogicalTimeline(length=12, uid="plain")
        flow = plain.get_flow()
        assert flow.length == q(12)
        assert flow.unfold(5) == [q(5)]

    def test_an_unknown_convention_is_refused(self) -> None:
        with pytest.raises(ValueError):
            _op106().get_flow("backwards")


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

    def test_a_child_timeline_is_refused(self) -> None:
        root = ContinuousLogicalTimeline(length=24, uid="root")
        movement = root.create_child(12, 0, uid="movement")
        movement.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)]))
        before = _state(root)
        with pytest.raises(ValueError) as raised:
            movement.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "timeline 'movement' is a child of 'root': a FlowMap names a root "
            "timeline, and the flows of a child's flow control are not written yet"
        )
        assert _state(root) == before
        # The flow itself is computed on the child's own axis.
        assert movement.get_flow().length == q(20)

    def test_an_unknown_length_is_refused(self) -> None:
        score = ContinuousLogicalTimeline(length=None, uid="score")
        score.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)]))
        with pytest.raises(UnknownLengthError) as raised:
            score.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "Cannot play its flow control: the length of timeline 'score' is unknown"
        )
        assert (score.list_regions(), score.list_flow_maps()) == ([], [])


class TestWrittenAgain:
    """The call replaces what it wrote before."""

    def test_unchanged_flow_control_writes_the_same(self) -> None:
        score = _op106()
        first = score.create_flow_maps_from_flow_control()
        virtual = score.create_virtual_timeline("flow_map_1")
        before = _state(score)
        second = score.create_flow_maps_from_flow_control()
        assert second == first
        assert _state(score) == before
        assert score.get_virtual_timeline(virtual.id) == virtual

    def test_changed_flow_control_replaces_sections_and_keeps_flow_map_ids(
        self,
    ) -> None:
        score = _repeat()
        first = score.create_flow_maps_from_flow_control()
        assert first is not None
        assert [(r.name, r.start, r.end) for r in first.sections] == [
            ("A", q(0), q(8)),
            ("B", q(8), q(12)),
        ]
        assert [(f.id, f.convention, f.chunks) for f in first.flow_maps] == [
            (
                "flow_map_1",
                FlowConvention.default,
                ("region_1", "region_1", "region_2"),
            ),
            ("flow_map_2", FlowConvention.single, ("region_1", "region_2")),
            ("flow_map_3", FlowConvention.printed, ("region_1", "region_2")),
        ]
        virtual = score.create_virtual_timeline("flow_map_1", id="played")
        assert virtual.length == q(20)
        user_region = score.create_region(2, 6, id="theme")
        assert not user_region.is_atomic_section

        score.flow_control.add(Break(q(10), id="stop"))
        second = score.create_flow_maps_from_flow_control()

        assert second is not None
        assert [(r.name, r.start, r.end) for r in second.sections] == [
            ("A", q(0), q(8)),
            ("B", q(8), q(10)),
            ("C", q(10), q(12)),
        ]
        a, b, c = (region.id for region in second.sections)
        assert [(f.id, f.convention, f.chunks) for f in second.flow_maps] == [
            ("flow_map_1", FlowConvention.default, (a, a, b)),
            ("flow_map_2", FlowConvention.single, (a, b)),
            ("flow_map_3", FlowConvention.printed, (a, b, c)),
        ]
        # Only the new sections and the Region that is no atomic section remain.
        assert sorted(score.list_regions()) == sorted([a, b, c, "theme"])
        assert score.get_region("theme") is user_region
        assert score.list_flow_maps() == ["flow_map_1", "flow_map_2", "flow_map_3"]
        assert score.get_virtual_timeline("played") == VirtualTimeline(
            "flow_map_1", "score", q(18), id="played"
        )
        timetoshare.validate(score.to_dict())


class TestAnotherOrigin:
    """Sections named by a FlowMap that was not computed are not replaced."""

    def test_a_flow_map_written_by_hand_over_a_computed_section(self) -> None:
        score = _repeat()
        score.create_flow_maps_from_flow_control()
        score.create_flow_map(["region_2", "region_1"], id="backwards")
        before = _state(score)
        with pytest.raises(ValueError) as raised:
            score.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "timeline 'score' holds atomic sections that the FlowMap(s) "
            "['backwards'] name, which were not computed from its flow control: "
            "the sections are not replaced"
        )
        assert _state(score) == before

    def test_flow_maps_naming_a_convention_without_provenance(self) -> None:
        score = Timeline.from_file(FIXTURES / "flow" / "sonata_exposition.yaml")
        before = _state(score)
        with pytest.raises(ValueError) as raised:
            score.create_flow_maps_from_flow_control()
        assert "['flow_map_1', 'flow_map_2', 'flow_map_3']" in str(raised.value)
        assert _state(score) == before

    def test_a_computed_flow_map_of_another_convention_name(self) -> None:
        score = _repeat()
        score.create_flow_maps_from_flow_control()
        score.create_flow_map(
            ["region_1"], id="exposition", convention="as recorded", provenance=INFERRED
        )
        before = _state(score)
        with pytest.raises(ValueError, match=r"FlowMap\(s\) \['exposition'\] name"):
            score.create_flow_maps_from_flow_control()
        assert _state(score) == before

    def test_a_section_a_jump_lands_on_is_not_replaced(self) -> None:
        score = _repeat()
        score.create_flow_maps_from_flow_control()
        score.flow_control.add(
            Jump(q(12), [JumpDestination(to="region_2", repeat_count=1)], id="back")
        )
        before = _state(score)
        with pytest.raises(ValueError) as raised:
            score.create_flow_maps_from_flow_control()
        assert str(raised.value) == (
            "timeline 'score' holds the atomic section(s) ['region_2'] that a jump "
            "of its flow control names as a place: the sections are not replaced"
        )
        assert _state(score) == before

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
