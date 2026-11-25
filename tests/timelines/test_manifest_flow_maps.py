"""FlowMaps and virtual timelines at the top level of a manifest
(see README, *FlowMaps and virtual timelines in a manifest*).
"""

from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, FlowConvention, Gap, TimeUnit
from timetoalign.core.provenance import Belief, Provenance
from timetoalign.maps import LinearMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    FlowMap,
    Timeline,
    TimelineGroup,
    VirtualTimeline,
)
from timetoalign.timelines.flowcontrol import Jump, JumpDestination

Q = TimeUnit.quarters
FIXTURES = Path(os.path.commonpath(timetoshare.fixture_paths())) / "flow"
EDITOR = "https://example.org/people/editor"


def q(value: Any) -> Coordinate:
    return Coordinate(Fraction(value), Q)


class TestFixtures:
    """The format's own manifests are read into objects and written back."""

    @pytest.mark.parametrize("reader", [Timeline, AlignmentBundle])
    @pytest.mark.parametrize(
        "name",
        [
            "sonata_exposition.yaml",
            "sonata_exposition.minimal.yaml",
            "flow_map_by_hand.yaml",
            "flow_map_by_hand.minimal.yaml",
        ],
    )
    def test_read_and_written_byte_for_byte(
        self, reader: Any, name: str, tmp_path: Path
    ) -> None:
        source = FIXTURES / name
        written = reader.from_file(source).to_file(tmp_path / name)
        assert written.read_bytes() == source.read_bytes()

    @pytest.mark.parametrize(
        "name", ["sonata_exposition.yaml", "sonata_exposition.minimal.yaml"]
    )
    def test_the_exposition_is_read_into_objects(self, name: str) -> None:
        score = Timeline.from_file(FIXTURES / name)
        assert [
            score.get_flow_map(flow_map_id) for flow_map_id in score.list_flow_maps()
        ] == [
            FlowMap(
                "score",
                ("region_1", "region_2", "region_1", "region_3"),
                convention=FlowConvention.default,
                id="flow_map_1",
            ),
            FlowMap(
                "score",
                ("region_1", "region_3"),
                convention=FlowConvention.single,
                id="flow_map_2",
            ),
            FlowMap(
                "score",
                ("region_1", "region_2", "region_3"),
                convention=FlowConvention.printed,
                id="flow_map_3",
            ),
        ]
        virtual = score.get_virtual_timeline("virtual_1")
        assert virtual == VirtualTimeline(
            "flow_map_1", "score", q(36), name="Score, as played", id="virtual_1"
        )
        assert score.list_virtual_timelines() == ["virtual_1"]
        # A 12 + B 4 + A 12 + C 8: the stated length is the FlowMap's.
        assert virtual.resolve(score).length == q(36)
        assert virtual.unfold(2, score) == [q(2), q(18)]
        assert virtual.fold(30, score) == q(18)

    @pytest.mark.parametrize(
        "name", ["flow_map_by_hand.yaml", "flow_map_by_hand.minimal.yaml"]
    )
    def test_the_flow_map_by_hand_is_read_into_an_object(self, name: str) -> None:
        concert = Timeline.from_file(FIXTURES / name)
        flow_map = concert.get_flow_map("flow_map_1")
        assert flow_map == FlowMap(
            "set",
            ("second", Gap(Fraction(4)), "first"),
            name="Encore order",
            provenance=(
                Provenance(
                    stated_by=(EDITOR,),
                    date="2026-11-25",
                    method="manual",
                    beliefs=(Belief(actor=EDITOR, belief="true", date="2026-11-25"),),
                ),
            ),
            id="flow_map_1",
        )
        assert type(flow_map.chunks[1].length) is Fraction  # type: ignore[union-attr]
        assert concert.list_virtual_timelines() == []
        # second [8, 20), a gap of 4, first [0, 8)
        resolved = flow_map.resolve(concert)
        assert resolved.length == q(24)
        assert resolved.unfold(2) == [q(18)]
        assert resolved.fold(13) is None

    def test_a_bundle_hands_the_flow_maps_to_their_roots(self) -> None:
        bundle = AlignmentBundle.from_file(FIXTURES / "sonata_exposition.yaml")
        score = bundle.timelines["score"]
        assert score.list_flow_maps() == ["flow_map_1", "flow_map_2", "flow_map_3"]
        assert score.list_virtual_timelines() == ["virtual_1"]


def _score() -> ContinuousLogicalTimeline:
    score = ContinuousLogicalTimeline(length=12, uid="score")
    score.flow_control.add(Jump(q(8), [JumpDestination(repeat_count=1)], id="repeat"))
    score.create_flow_maps_from_flow_control()
    score.create_virtual_timeline("flow_map_1", id="as_played")
    return score


def _concert() -> ContinuousLogicalTimeline:
    concert = ContinuousLogicalTimeline(length=20, uid="concert")
    concert.create_child(8, 0, uid="first")
    concert.create_child(12, 8, uid="second")
    concert.create_flow_map(
        ["second", Gap(4), "first"], id="encore", name="Encore order", canonical=True
    )
    return concert


class TestBuiltInMemory:
    """Objects built by calls are written, read and equal."""

    def test_a_bundle_of_two_roots_round_trips(self) -> None:
        bundle = AlignmentBundle(name="evening")
        score, concert = _score(), _concert()
        bundle.add_timeline(score)
        bundle.add_timeline(concert)
        manifest = bundle.to_dict()
        timetoshare.validate(manifest)

        # Sorted by ID across both roots, every ID as a file writes it.
        assert [entry["id"] for entry in manifest["flow_maps"]] == [
            "#encore",
            "#flow_map_1",
            "#flow_map_2",
            "#flow_map_3",
        ]
        assert manifest["flow_maps"][0] == {
            "id": "#encore",
            "timeline": "#concert",
            "chunks": ["#second", {"gap": [4, 1]}, "#first"],
            "canonical": True,
            "name": "Encore order",
        }
        assert manifest["flow_maps"][1] == {
            "id": "#flow_map_1",
            "timeline": "#score",
            "chunks": ["#region_1", "#region_1", "#region_2"],
            "convention": "default",
            "provenance": [{"method": "inferred"}],
        }
        assert manifest["virtual_timelines"] == [
            {
                "id": "#as_played",
                "map": "#flow_map_1",
                "timeline": "#score",
                "unit": "quarters",
                "number_type": "fraction",
                "length": [20, 1],
            }
        ]

        read = AlignmentBundle.from_dict(manifest)
        for original in (score, concert):
            copy = read.timelines[original.id]
            assert copy._flow_maps == original._flow_maps
            assert copy._virtual_timelines == original._virtual_timelines
        assert read.to_dict() == manifest
        assert read.to_dict(minimal=False) == bundle.to_dict(minimal=False)

    def test_a_timeline_and_a_group_write_them_too(self) -> None:
        score = _score()
        alone = score.to_dict()
        assert [entry["id"] for entry in alone["flow_maps"]] == [
            "#flow_map_1",
            "#flow_map_2",
            "#flow_map_3",
        ]
        assert Timeline.from_dict(alone).to_dict() == alone
        group = TimelineGroup(id="pair", timelines=[_score(), _concert()])
        grouped = group.to_dict()
        assert [entry["id"] for entry in grouped["flow_maps"]] == [
            "#encore",
            "#flow_map_1",
            "#flow_map_2",
            "#flow_map_3",
        ]
        assert [entry["id"] for entry in grouped["virtual_timelines"]] == ["#as_played"]
        assert TimelineGroup.from_dict(grouped).to_dict() == grouped

    def test_the_complete_form_writes_the_defaults_of_a_flow_map(self) -> None:
        complete = _concert().to_dict(minimal=False)
        assert complete["flow_maps"] == [
            {
                "id": "#encore",
                "timeline": "#concert",
                "chunks": ["#second", {"gap": [4, 1]}, "#first"],
                "canonical": True,
                "name": "Encore order",
            }
        ]
        score = _score().to_dict(minimal=False)
        assert score["flow_maps"][1] == {
            "id": "#flow_map_2",
            "timeline": "#score",
            "chunks": ["#region_1", "#region_2"],
            "convention": "single",
            "canonical": False,
            "name": None,
            "provenance": [{"method": "inferred"}],
        }
        assert score["virtual_timelines"][0]["name"] is None

    def test_no_key_is_written_where_there_is_nothing(self) -> None:
        plain = ContinuousLogicalTimeline(length=12, uid="plain")
        for manifest in (plain.to_dict(), plain.to_dict(minimal=False)):
            assert "flow_maps" not in manifest
            assert "virtual_timelines" not in manifest


class TestProvenancePrefixes:
    """A FlowMap's provenance is read under the manifest's context."""

    ENTRY = {"method": "manual", "stated_by": ["ex:editor"]}

    def _concert(self) -> ContinuousLogicalTimeline:
        concert = ContinuousLogicalTimeline(length=20, uid="concert")
        concert.create_child(8, 0, uid="first")
        concert.create_flow_map(["first"], id="encore", provenance=[self.ENTRY])
        return concert

    def test_a_prefix_the_manifest_declares_is_written_back(self) -> None:
        bundle = AlignmentBundle()
        bundle.add_prefix("ex", "https://example.org/people/")
        bundle.add_timeline(self._concert())
        manifest = bundle.to_dict()
        assert manifest["@context"]["ex"] == "https://example.org/people/"
        assert manifest["flow_maps"][0]["provenance"] == [self.ENTRY]
        timetoshare.validate(manifest)
        assert AlignmentBundle.from_dict(manifest).to_dict() == manifest
        # A timeline read from a manifest as a whole writes that manifest back.
        alone = self._concert()
        with pytest.raises(ValueError, match="the prefix 'ex' is declared neither"):
            alone.to_dict()
        assert Timeline.from_dict(manifest).to_dict() == manifest

    def test_a_prefix_of_the_catalogue_needs_no_declaration(self) -> None:
        concert = ContinuousLogicalTimeline(length=20, uid="concert")
        concert.create_child(8, 0, uid="first")
        concert.create_flow_map(
            ["first"], provenance=[{"method": "manual", "stated_by": ["wd:Q255"]}]
        )
        manifest = concert.to_dict()
        assert manifest["@context"]["wd"] == "http://www.wikidata.org/entity/"
        assert Timeline.from_dict(manifest).to_dict() == manifest


class TestKeptAsRead:
    """Statements the library does not evaluate are written back unchanged."""

    @staticmethod
    def _manifest(virtual: dict[str, Any]) -> dict[str, Any]:
        score = ContinuousLogicalTimeline(length=12, uid="score")
        score.add_conversion_map(
            LinearMap(
                scalar=0.5,
                source_unit=Q,
                target_unit=TimeUnit.seconds,
                uid="to_seconds",
            )
        )
        section = score.create_region(0, 12, id="all")
        score.create_flow_map([section], id="whole")
        manifest = score.to_dict()
        manifest["virtual_timelines"] = [virtual]
        return manifest

    def test_a_conversion_map_applied(self) -> None:
        manifest = self._manifest(
            {
                "id": "#in_seconds",
                "map": "#to_seconds",
                "timeline": "#score",
                "unit": "seconds",
                "number_type": "float",
                "length": 6.0,
            }
        )
        score = Timeline.from_dict(manifest)
        assert score.to_dict() == manifest
        virtual = score.get_virtual_timeline("in_seconds")
        assert virtual.length == Coordinate(6.0, TimeUnit.seconds)
        with pytest.raises(ValueError) as raised:
            virtual.fold(3.0, score)
        assert str(raised.value) == (
            "virtual timeline 'in_seconds' applies 'to_seconds', which is no "
            "FlowMap of timeline 'score': its positions are not converted. "
            "FlowMaps of the timeline: ['whole']"
        )

    def test_a_flow_map_carried_through_something(self) -> None:
        manifest = self._manifest(
            {
                "id": "#carried",
                "map": "#whole",
                "timeline": "#score",
                "through": "#to_seconds",
                "unit": "quarters",
                "number_type": "fraction",
                "length": [12, 1],
            }
        )
        score = Timeline.from_dict(manifest)
        assert score.to_dict() == manifest
        virtual = score.get_virtual_timeline("carried")
        assert virtual.through == "to_seconds"
        with pytest.raises(ValueError, match="through 'to_seconds'.*not supported yet"):
            virtual.unfold(3, score)

    def test_a_virtual_timeline_of_a_virtual_timeline_joins_the_same_root(self) -> None:
        manifest = self._manifest(
            {
                "id": "#first_view",
                "map": "#whole",
                "timeline": "#score",
                "unit": "quarters",
                "number_type": "fraction",
                "length": [12, 1],
            }
        )
        manifest["virtual_timelines"].append(
            {
                "id": "#second_view",
                "map": "#whole",
                "timeline": "#first_view",
                "unit": "quarters",
                "number_type": "fraction",
                "length": [12, 1],
            }
        )
        score = Timeline.from_dict(manifest)
        assert score.list_virtual_timelines() == ["first_view", "second_view"]
        assert score.to_dict() == manifest
        assert score.get_virtual_timeline("first_view").fold(3, score) == q(3)
        with pytest.raises(
            ValueError, match="not to the FlowMap's own timeline 'score'"
        ):
            score.get_virtual_timeline("second_view").fold(3, score)


def test_provenance_entries_the_format_admits_are_kept() -> None:
    entries = [
        {},
        {"method": "manual", "x-tool": "abc", "x-run": None},
        {
            "beliefs": [
                {
                    "actor": EDITOR,
                    "belief": "true",
                    "date": "2026-11-25",
                    "x-reason": "heard",
                }
            ]
        },
    ]
    manifest = {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "#score",
                "unit": "quarters",
                "length": [12, 1],
                "regions": [{"id": "#a", "start": [0, 1], "end": [8, 1]}],
            }
        ],
        "flow_maps": [
            {
                "id": "#twice",
                "timeline": "#score",
                "chunks": ["#a", "#a"],
                "provenance": entries,
            }
        ],
    }
    timetoshare.validate(manifest)
    score = Timeline.from_dict(manifest)
    nothing, tool, believed = score.get_flow_map("twice").provenance
    assert (nothing.method, tool.method) == (None, "manual")
    assert believed.get_belief(EDITOR).belief == "true"
    written = score.to_dict()
    timetoshare.validate(written)
    assert written["flow_maps"] == manifest["flow_maps"]
    assert Timeline.from_dict(written).to_dict() == written
