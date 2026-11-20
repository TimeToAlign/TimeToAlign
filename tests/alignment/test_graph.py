"""Tests for MatchGraph and MatchStamp classes.

This module tests the graph-building layer of the alignment infrastructure:
- MatchStamp: Cross-group timestamp representation
- MatchGraph: Graph of MatchClaims with networkx integration
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment import (
    Agent,
    AlignmentAnchor,
    MatchClaim,
    MatchLine,
    MatchMetadata,
)
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.graph import MatchStamp as MatchStampType
from timetoalign.core import Coordinate, IdCoordinate
from timetoalign.core.enums import AgentType, Domain, TimeUnit
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    TimelineGroup,
)

from .helpers import make_match_stamp as MatchStamp

# region Fixtures


@pytest.fixture
def simple_instant_claim() -> MatchClaim:
    """Simple instant match between two timelines."""
    return MatchClaim(
        timeline_a_id="tl_a",
        timeline_b_id="tl_b",
        start_anchor=AlignmentAnchor(
            timeline_a_id="tl_a",
            coordinate_a=Coordinate(100.0, TimeUnit.number),
            timeline_b_id="tl_b",
            coordinate_b=Coordinate(50.0, TimeUnit.number),
        ),
        metadata=MatchMetadata(
            agent=Agent(name="test", type=AgentType.software, identifier="manual")
        ),
    )


@pytest.fixture
def simple_interval_claim() -> MatchClaim:
    """Simple interval match between two timelines."""
    return MatchClaim(
        timeline_a_id="tl_a",
        timeline_b_id="tl_b",
        start_anchor=AlignmentAnchor(
            timeline_a_id="tl_a",
            coordinate_a=Coordinate(0.0, TimeUnit.number),
            timeline_b_id="tl_b",
            coordinate_b=Coordinate(0.0, TimeUnit.number),
        ),
        end_anchor=AlignmentAnchor(
            timeline_a_id="tl_a",
            coordinate_a=Coordinate(100.0, TimeUnit.number),
            timeline_b_id="tl_b",
            coordinate_b=Coordinate(50.0, TimeUnit.number),
        ),
        metadata=MatchMetadata(
            agent=Agent(name="test", type=AgentType.software, identifier="manual")
        ),
    )


@pytest.fixture
def three_timeline_claims() -> list[MatchClaim]:
    """Claims connecting three timelines: A <-> B <-> C."""
    return [
        MatchClaim(
            timeline_a_id="tl_a",
            timeline_b_id="tl_b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_a",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl_b",
                coordinate_b=Coordinate(50.0, TimeUnit.number),
            ),
        ),
        MatchClaim(
            timeline_a_id="tl_b",
            timeline_b_id="tl_c",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_b",
                coordinate_a=Coordinate(50.0, TimeUnit.number),
                timeline_b_id="tl_c",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        ),
    ]


@pytest.fixture
def dgt1_1000px_timeline() -> DiscreteGraphicalTimeline:
    """DGT1 timeline for group tests (1000 pixels)."""
    return DiscreteGraphicalTimeline(
        length=1000,
        unit="pixels",
        uid="dgt_1",
    )


@pytest.fixture
def dgt2_800px_timeline() -> DiscreteGraphicalTimeline:
    """DGT2 timeline for group tests (800 pixels)."""
    return DiscreteGraphicalTimeline(
        length=800,
        unit="pixels",
        uid="dgt_2",
    )


@pytest.fixture
def audio_100s_timeline() -> ContinuousPhysicalTimeline:
    """Audio timeline for group tests (100 seconds)."""
    return ContinuousPhysicalTimeline(
        length=100.0,
        unit="seconds",
        uid="audio",
    )


@pytest.fixture
def dgt1_group(
    dgt1_1000px_timeline: DiscreteGraphicalTimeline,
    audio_100s_timeline: ContinuousPhysicalTimeline,
) -> TimelineGroup:
    """Group with DGT1 and audio timelines.

    DGT1: 1000 pixels, Audio: 100 seconds
    Audio's full extent (0-100) maps to DGT1's full extent (0-1000).
    """
    group = TimelineGroup(id="dgt1_group", name="DGT1_Group")
    group.add_timeline(dgt1_1000px_timeline)
    group.add_timeline(audio_100s_timeline)
    return group


# endregion


# region MatchStamp Tests


class TestMatchStamp:
    """Tests for MatchStamp class."""

    def test_basic_creation(self) -> None:
        """Create a basic MatchStamp."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            anchor_edges=[("tl_a", "tl_b")],
            inferred_edges=[],
        )
        assert stamp.n_timelines == 2
        assert stamp.n_explicit_edges == 1
        assert stamp.n_inferred_edges == 0

    def test_empty_stamp_is_invalid(self) -> None:
        """Reject a stamp without the required source-axis coordinate."""
        with pytest.raises((TypeError, ValueError)):
            MatchStampType()

    def test_get_coordinate_for(self) -> None:
        """Project a stored coordinate and reject an absent timeline."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
        )
        assert stamp.get_coordinate_for("tl_a", format="float") == 100.0
        assert stamp.get_coordinate_for("tl_b", format="float") == 50.0
        with pytest.raises(KeyError):
            stamp.get_coordinate_for("tl_c")

    def test_get_coordinate(self) -> None:
        """Get an exact Coordinate carrying the timeline unit."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0},
            coordinate_units={"tl_a": "seconds"},
        )

        assert stamp.get_coordinate("tl_a") == stamp.get_coordinate_for("tl_a")

    def test_has_timeline(self) -> None:
        """Check if timeline is in stamp."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
        )
        assert stamp.has_timeline("tl_a") is True
        assert stamp.has_timeline("tl_c") is False

    def test_present_timelines(self) -> None:
        """Get list of timeline IDs."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
        )
        assert set(stamp.present_timelines) == {"tl_a", "tl_b", "tl_c"}

    def test_filter_by_timelines_include(self) -> None:
        """Filter stamp to include only specific timelines."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
            anchor_edges=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        filtered = stamp.filter_by_timelines(
            timeline_ids={"tl_a", "tl_b"},
            id_pattern=r"^tl_[ab]$",
        )

        assert filtered.n_timelines == 2
        assert "tl_c" not in filtered.coordinates
        assert len(filtered.anchor_edges) == 1
        assert ("tl_a", "tl_b") in filtered.anchor_edges

    def test_filter_by_timelines_id_pattern(self) -> None:
        """Filter stamp with a canonical timeline ID pattern."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
            anchor_edges=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        filtered = stamp.filter_by_timelines(id_pattern=r"^(?!tl_c$).*")

        assert filtered.n_timelines == 2
        assert "tl_c" not in filtered.coordinates
        assert len(filtered.anchor_edges) == 1

    def test_to_dict_roundtrip(self) -> None:
        """Serialize and deserialize MatchStamp."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            anchor_edges=[("tl_a", "tl_b")],
            inferred_edges=[("tl_b", "tl_c")],
        )
        data = stamp.to_dict(format="graph")
        restored = MatchStampType.from_dict(data)

        assert restored.coordinates == stamp.coordinates
        assert restored.anchor_edges == stamp.anchor_edges
        assert restored.inferred_edges == stamp.inferred_edges

    def test_graph_dict_isolation(self) -> None:
        """Graph dictionaries do not expose the stamp's mutable containers."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            anchor_edges=[("tl_a", "tl_b")],
            inferred_edges=[("tl_b", "tl_c")],
        )
        data = stamp.to_dict(format="graph")
        data["coordinates"]["tl_a"] = 999.0
        data["anchor_edges"].append(("tl_c", "tl_d"))
        data["inferred_edges"].clear()

        assert stamp.coordinates == {
            "tl_a": Coordinate(100.0, TimeUnit.number),
            "tl_b": Coordinate(50.0, TimeUnit.number),
        }
        assert stamp.anchor_edges == [("tl_a", "tl_b")]
        assert stamp.inferred_edges == [("tl_b", "tl_c")]

    def test_from_dict_isolation(self) -> None:
        """Mutating input dictionaries does not mutate the restored stamp."""
        data = {
            "coordinates": {
                "tl_a": {
                    "unit": "number",
                    "number_type": "float",
                    "value": 100.0,
                },
                "tl_b": {
                    "unit": "number",
                    "number_type": "float",
                    "value": 50.0,
                },
            },
            "anchor_edges": [("tl_a", "tl_b")],
            "inferred_edges": [("tl_b", "tl_c")],
        }
        stamp = MatchStampType.from_dict(data)
        data["coordinates"]["tl_a"] = 999.0
        data["anchor_edges"].append(("tl_c", "tl_d"))
        data["inferred_edges"].clear()

        assert stamp.coordinates == {
            "tl_a": Coordinate(100.0, TimeUnit.number),
            "tl_b": Coordinate(50.0, TimeUnit.number),
        }
        assert stamp.anchor_edges == [("tl_a", "tl_b")]
        assert stamp.inferred_edges == [("tl_b", "tl_c")]

    def test_source_axis_is_required(self) -> None:
        """A stamp cannot be constructed without a source-axis coordinate."""
        with pytest.raises((TypeError, ValueError)):
            MatchStamp()

    def test_repr(self) -> None:
        """Test string representation."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
        )
        assert repr(stamp) == "MatchStamp(tl_a=100 number, tl_b=50 number)"

    def test_exact_unit_bearing_renderings(self) -> None:
        """All MatchStamp displays retain exact rational values and units."""
        stamp = MatchStamp(
            coordinates={"audio": Fraction(25, 2), "score": Fraction(415, 24)},
            coordinate_units={"audio": "seconds", "score": "quarters"},
            anchor_edges=[("audio", "score")],
        )

        assert repr(stamp) == "MatchStamp(audio=25/2 seconds, score=415/24 quarters)"
        assert str(stamp) == (
            "MatchStamp (2 timelines, 1 edges)\n"
            "  audio     25/2 seconds  anchor\n"
            "  score  415/24 quarters  anchor"
        )
        assert stamp._repr_html_() == (
            "<div style='font-family: monospace;'><strong>MatchStamp</strong> "
            "<span style='background: #e3f2fd; padding: 0 4px; border-radius: 3px; "
            "font-size: 0.8em;'>2 timelines, 1 edges</span><table "
            "style='border-collapse: collapse; margin-top: 4px;'><thead><tr "
            "style='border-bottom: 1px solid #ccc;'><th style='text-align: left; "
            "padding: 2px 8px;'>ID</th><th style='text-align: right; padding: 2px 8px;'>"
            "Coordinate</th><th style='text-align: left; padding: 2px 8px;'>Type</th>"
            "</tr></thead><tbody><tr><td><strong>audio</strong></td><td "
            "style='text-align: right;'>25/2 seconds</td><td><em>anchor</em></td></tr>"
            "<tr><td><strong>score</strong></td><td style='text-align: right;'>"
            "415/24 quarters</td><td><em>anchor</em></td></tr></tbody></table><div "
            "style='margin-top: 4px; color: #666; font-size: 0.85em;'>Try: "
            "<code>stamp.get_coordinate_for(&lt;tl_id&gt;)</code>, "
            "<code>stamp.get_coordinates_for(&lt;tl_ids&gt;)</code>, "
            "<code>stamp.get_unit(&lt;unit&gt;)</code>, "
            "<code>stamp.get_conversion_for(&lt;key&gt;)</code></div></div>"
        )


# endregion


# region MatchGraph Basic Tests


class TestMatchGraphBasic:
    """Basic tests for MatchGraph class."""

    def test_empty_graph(self) -> None:
        """Create an empty MatchGraph."""
        graph = MatchGraph()
        assert graph.n_claims == 0
        assert graph.n_nodes == 0
        assert graph.n_edges == 0

    def test_single_instant_claim(self, simple_instant_claim: MatchClaim) -> None:
        """Graph with single instant claim."""
        graph = MatchGraph([simple_instant_claim])

        assert graph.n_claims == 1
        assert graph.n_nodes == 2
        assert graph.n_edges == 1
        assert graph.timeline_ids == {"tl_a", "tl_b"}

    def test_single_interval_claim(self, simple_interval_claim: MatchClaim) -> None:
        """Graph with single interval claim."""
        graph = MatchGraph([simple_interval_claim])

        assert graph.n_claims == 1
        assert graph.n_nodes == 4  # 2 start nodes + 2 end nodes
        assert graph.n_edges == 2  # start edge + end edge

    def test_multiple_claims(self, three_timeline_claims: list[MatchClaim]) -> None:
        """Graph with multiple claims connecting three timelines."""
        graph = MatchGraph(three_timeline_claims)

        assert graph.n_claims == 2
        assert graph.n_edges == 2
        assert graph.timeline_ids == {"tl_a", "tl_b", "tl_c"}

    def test_get_nodes_for_timeline(self, simple_interval_claim: MatchClaim) -> None:
        """Get nodes for specific timeline."""
        graph = MatchGraph([simple_interval_claim])

        nodes_a = graph.get_nodes_for_timeline("tl_a")
        assert len(nodes_a) == 2  # start and end
        assert nodes_a == [
            IdCoordinate(0.0, TimeUnit.number, "tl_a"),
            IdCoordinate(100.0, TimeUnit.number, "tl_a"),
        ]

    def test_get_coordinates_for(self, simple_interval_claim: MatchClaim) -> None:
        """Get coordinates for specific timeline."""
        graph = MatchGraph([simple_interval_claim])

        coords = graph.get_coordinates_for("tl_a", format="float")
        assert coords == [0.0, 100.0]

    def test_get_connected_nodes(self, simple_instant_claim: MatchClaim) -> None:
        """Get nodes connected to a given node."""
        graph = MatchGraph([simple_instant_claim])

        connected = graph.get_connected_nodes(graph.get_coordinate_at(100.0, "tl_a"))
        assert len(connected) == 1
        assert connected[0] == graph.get_coordinate_at(50.0, "tl_b")

    def test_get_connected_timelines(
        self, three_timeline_claims: list[MatchClaim]
    ) -> None:
        """Get timelines connected to a given timeline."""
        graph = MatchGraph(three_timeline_claims)

        # tl_a connects to tl_b
        connected_a = graph.get_connected_timelines("tl_a")
        assert connected_a == {"tl_b"}

        # tl_b connects to both tl_a and tl_c
        connected_b = graph.get_connected_timelines("tl_b")
        assert connected_b == {"tl_a", "tl_c"}


# endregion


# region MatchGraph MatchStamp Extraction Tests


class TestMatchGraphStamps:
    """Tests for MatchStamp extraction from MatchGraph."""

    def test_instant_claim_yields_one_stamp(
        self, simple_instant_claim: MatchClaim
    ) -> None:
        """Instant claim yields one MatchStamp."""
        graph = MatchGraph([simple_instant_claim])
        stamps = graph.get_stamps()

        assert len(stamps) == 1
        stamp = stamps[0]
        assert stamp.get_coordinate_for("tl_a", format="float") == 100.0
        assert stamp.get_coordinate_for("tl_b", format="float") == 50.0
        assert len(stamp.anchor_edges) == 1

    def test_interval_claim_yields_two_stamps(
        self, simple_interval_claim: MatchClaim
    ) -> None:
        """Interval claim yields two MatchStamps."""
        graph = MatchGraph([simple_interval_claim])
        stamps = sorted(
            graph.get_stamps(),
            key=lambda stamp: stamp.get_coordinate_for("tl_a", format="float"),
        )

        assert len(stamps) == 2
        start_stamp, end_stamp = stamps

        # Start stamp
        assert start_stamp.get_coordinate_for("tl_a", format="float") == 0.0
        assert start_stamp.get_coordinate_for("tl_b", format="float") == 0.0

        # End stamp
        assert end_stamp.get_coordinate_for("tl_a", format="float") == 100.0
        assert end_stamp.get_coordinate_for("tl_b", format="float") == 50.0

    def test_get_stamps_three_timeline_chain(
        self, three_timeline_claims: list[MatchClaim]
    ) -> None:
        """Get all stamps from graph with multiple connected components."""
        # These claims create a single connected component (A-B-C chain)
        graph = MatchGraph(three_timeline_claims)
        stamps = graph.get_stamps()

        # Should be 1 stamp since all nodes are connected
        assert len(stamps) == 1
        stamp = stamps[0]
        assert stamp.n_timelines == 3

    def test_disconnected_components_yield_multiple_stamps(self) -> None:
        """Disconnected components yield separate stamps."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_c",
                timeline_b_id="tl_d",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_c",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_d",
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            ),  # Disconnected
        ]
        graph = MatchGraph(claims)
        stamps = graph.get_stamps()

        assert len(stamps) == 2
        # Each stamp has 2 timelines
        assert all(s.n_timelines == 2 for s in stamps)


# endregion


# region MatchGraph Group Extension Tests


class TestMatchGraphGroupExtension:
    """Tests for extend_to_groups() method."""

    def test_extend_to_groups_adds_inferred_edges(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Extending to groups adds inferred edges."""
        # Claim between dgt1 and some external timeline
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        # Set up group mapping
        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)

        # Original graph: 2 nodes, 1 edge
        # Extended graph should have 3 nodes (dgt1, external, audio)
        # and 2 edges (dgt1-external explicit, dgt1-audio inferred)
        assert extended.n_nodes == 3
        assert extended.n_edges == 2

    def test_extend_creates_correct_coordinates(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Extended edges have correct converted coordinates."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.pixels),  # Midpoint
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)

        # Get stamp and check audio coordinate
        # 500px in dgt1 (0-1000) should be 50s in audio (0-100)
        stamps = extended.get_stamps()
        assert len(stamps) == 1
        stamp = stamps[0]
        assert stamp.get_coordinate_for("audio", format="float") == 50.0

    def test_extend_marks_inferred_edges(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Extended edges are marked as inferred, not explicit."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)
        stamps = extended.get_stamps()
        assert len(stamps) == 1
        stamp = stamps[0]

        # Should have 1 explicit edge (dgt1-external)
        # and 1 inferred edge (dgt1-audio or audio-external)
        assert stamp.n_explicit_edges == 1
        assert stamp.n_inferred_edges == 1

    def test_extend_with_include_inferred_false(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """include_inferred=False returns original graph."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups, include_inferred=False)

        # Should be same as original
        assert extended.n_nodes == 2
        assert extended.n_edges == 1


# endregion


# region MatchGraph Filter Tests


class TestMatchGraphFilter:
    """Tests for MatchGraph.filter() method."""

    def test_filter_by_timeline_ids(self) -> None:
        """Filter to include only specific timelines."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_b",
                timeline_b_id="tl_c",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_b",
                    coordinate_a=Coordinate(50.0, TimeUnit.number),
                    timeline_b_id="tl_c",
                    coordinate_b=Coordinate(25.0, TimeUnit.number),
                ),
            ),
        ]
        graph = MatchGraph(claims)

        filtered = graph.filter(timeline_ids={"tl_a", "tl_b"})

        assert filtered.timeline_ids == {"tl_a", "tl_b"}
        assert filtered.n_edges == 1

    def test_filter_by_id_pattern(self) -> None:
        """Filter to exclude a timeline through an ID pattern."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_b",
                timeline_b_id="tl_c",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_b",
                    coordinate_a=Coordinate(50.0, TimeUnit.number),
                    timeline_b_id="tl_c",
                    coordinate_b=Coordinate(25.0, TimeUnit.number),
                ),
            ),
        ]
        graph = MatchGraph(claims)

        filtered = graph.filter(id_pattern=r"^(?!tl_c$).*")

        assert "tl_c" not in filtered.timeline_ids
        assert filtered.n_edges == 1

    def test_filter_synchronous_only(self) -> None:
        """Filter to include only synchronous edges."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
                is_synchronous=True,
            ),
            MatchClaim(
                timeline_a_id="tl_b",
                timeline_b_id="tl_c",
                start_anchor=None,
                end_anchor=None,
                is_synchronous=False,
                metadata=MatchMetadata(
                    agent=Agent(
                        name="test", type=AgentType.software, identifier="structural"
                    )
                ),
            ),
        ]
        graph = MatchGraph(claims)

        filtered = graph.filter(synchronous_only=True)

        # Only the synchronous edge should remain
        assert filtered.n_edges == 1
        # tl_c should be removed (isolated after edge removal)
        assert "tl_c" not in filtered.timeline_ids

    def test_filter_explicit_only(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Filter to include only explicit edges (remove inferred)."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)
        filtered = extended.filter(explicit_only=True)

        # Should only have the explicit edge
        assert filtered.n_edges == 1
        # Audio should be removed (only connected via inferred edge)
        assert "audio" not in filtered.timeline_ids

    def test_filter_removes_isolated_nodes(self) -> None:
        """Filtering edges removes nodes that become isolated."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
                is_synchronous=True,
            ),
            MatchClaim(
                timeline_a_id="tl_c",
                timeline_b_id="tl_d",
                start_anchor=None,
                end_anchor=None,
                is_synchronous=False,
            ),
        ]
        graph = MatchGraph(claims)

        filtered = graph.filter(synchronous_only=True)

        # tl_c and tl_d should be removed
        assert filtered.timeline_ids == {"tl_a", "tl_b"}


# endregion


# region MatchGraph Serialization Tests


class TestMatchGraphSerialization:
    """Tests for MatchGraph serialization."""

    def test_to_dict_roundtrip(self, simple_interval_claim: MatchClaim) -> None:
        """Serialize and deserialize MatchGraph."""
        graph = MatchGraph([simple_interval_claim])
        data = graph.to_dict()
        restored = MatchGraph.from_dict(data)

        assert restored.n_claims == graph.n_claims
        assert restored.n_nodes == graph.n_nodes
        assert restored.n_edges == graph.n_edges

    def test_repr(self, simple_instant_claim: MatchClaim) -> None:
        """Test string representation."""
        graph = MatchGraph([simple_instant_claim])
        repr_str = repr(graph)

        assert "MatchGraph" in repr_str
        assert "claims=1" in repr_str
        assert "nodes=2" in repr_str
        assert "edges=1" in repr_str


# endregion


# region Integration with Thoresen PoC


class TestMatchGraphThoresenIntegration:
    """Integration tests using Thoresen PoC data."""

    # thoresen_segment_claims fixture is provided by conftest.py

    def test_thoresen_graph_structure(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """Thoresen segment claims create correct graph structure."""
        graph = MatchGraph(thoresen_segment_claims)

        assert graph.n_claims == 5
        # Each interval claim has 2 anchors, but adjacent segments share
        # boundary coordinates. So we have 6 unique boundary points
        # (0, 967, 1934, 2901, 3868, 4835) for DGT1 and
        # (0, 866, 1733, 2600, 3464, 4328) for DGT2.
        # This gives 12 nodes and 6 unique edges.
        assert graph.n_nodes == 12
        assert graph.n_edges == 6
        assert graph.timeline_ids == {"dgt_1", "dgt_2"}

    def test_thoresen_coordinates(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """Thoresen graph has correct boundary coordinates."""
        graph = MatchGraph(thoresen_segment_claims)

        dgt1_coords = graph.get_coordinates_for("dgt_1", format="float")
        dgt2_coords = graph.get_coordinates_for("dgt_2", format="float")

        # DGT1: 5 segments of 967 each = 4835 total
        assert dgt1_coords[0] == 0.0
        assert dgt1_coords[-1] == 4835.0

        # DGT2: segments of [866, 867, 867, 864, 864] = 4328 total
        assert dgt2_coords[0] == 0.0
        assert dgt2_coords[-1] == 4328.0

    def test_thoresen_stamps_at_segment_boundaries(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """MatchStamps align at segment boundaries."""
        graph = MatchGraph(thoresen_segment_claims)

        # Stamps are one per connected component; find segment-1's start (0,0)
        # and end (967,866) boundary stamps.
        stamps = graph.get_stamps()
        coord_pairs = {
            (
                stamp.get_coordinate_for("dgt_1", format="float"),
                stamp.get_coordinate_for("dgt_2", format="float"),
            )
            for stamp in stamps
        }
        assert (0.0, 0.0) in coord_pairs
        assert (967.0, 866.0) in coord_pairs


# endregion


# region MatchGraph Overhaul Tests


class TestMatchGraphNonSynchronousClaims:
    """Non-synchronous claims stored as metadata, no edges."""

    def test_non_synchronous_claims_no_edges(self) -> None:
        """Non-synchronous claims do not create graph edges."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
                is_synchronous=True,
            ),
            MatchClaim.nomatch(
                event={"start": 200.0},
                source_tl_id="tl_a",
                target_tl_id="tl_b",
                unit=TimeUnit.number,
            ),
        ]
        graph = MatchGraph(claims)

        # Only the synchronous claim creates edges
        assert graph.n_claims == 2
        assert graph.n_nodes == 2  # Only from the synchronous claim
        assert graph.n_edges == 1

    def test_non_synchronous_claims_accessible_via_claims(self) -> None:
        """Non-synchronous claims are accessible via claims property."""
        sync_claim = MatchClaim(
            timeline_a_id="tl_a",
            timeline_b_id="tl_b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_a",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl_b",
                coordinate_b=Coordinate(50.0, TimeUnit.number),
            ),
        )
        nomatch_claim = MatchClaim.nomatch(
            event={"start": 200.0},
            source_tl_id="tl_a",
            target_tl_id="tl_c",
            unit=TimeUnit.number,
        )
        graph = MatchGraph([sync_claim, nomatch_claim])

        assert len(graph.claims) == 2
        assert len(graph.synchronous_claims) == 1
        assert len(graph.non_synchronous_claims) == 1
        assert graph.non_synchronous_claims[0] is nomatch_claim

    def test_only_non_synchronous_claims_empty_graph(self) -> None:
        """Graph with only non-synchronous claims has no nodes or edges."""
        claims = [
            MatchClaim.nomatch(
                event={"start": 100.0},
                source_tl_id="tl_a",
                target_tl_id="tl_b",
                unit=TimeUnit.number,
            ),
            MatchClaim.nomatch(
                event={"start": 200.0},
                source_tl_id="tl_c",
                target_tl_id="tl_d",
                unit=TimeUnit.number,
            ),
        ]
        graph = MatchGraph(claims)

        assert graph.n_claims == 2
        assert graph.n_nodes == 0
        assert graph.n_edges == 0
        assert len(graph.timeline_ids) == 0


class TestMatchGraphGetStamps:
    """Tests for get_stamps() method."""

    def test_get_stamps_returns_list(self, simple_instant_claim: MatchClaim) -> None:
        """get_stamps() returns a list of MatchStamps."""
        graph = MatchGraph([simple_instant_claim])
        stamps = graph.get_stamps()

        assert isinstance(stamps, list)
        assert len(stamps) == 1
        assert stamps[0].get_coordinate_for("tl_a", format="float") == 100.0
        assert stamps[0].get_coordinate_for("tl_b", format="float") == 50.0

    def test_get_stamps_one_per_component(self) -> None:
        """get_stamps() returns one MatchStamp per connected component."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_c",
                timeline_b_id="tl_d",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_c",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_d",
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            ),  # Disconnected
        ]
        graph = MatchGraph(claims)
        stamps = graph.get_stamps()

        assert len(stamps) == 2
        assert all(s.n_timelines == 2 for s in stamps)

    def test_get_stamps_connected_chain(
        self, three_timeline_claims: list[MatchClaim]
    ) -> None:
        """get_stamps() merges connected chain into single stamp."""
        graph = MatchGraph(three_timeline_claims)
        stamps = graph.get_stamps()

        assert len(stamps) == 1
        assert stamps[0].n_timelines == 3

    def test_get_stamps_empty_graph(self) -> None:
        """get_stamps() on empty graph returns empty list."""
        graph = MatchGraph()
        stamps = graph.get_stamps()
        assert stamps == []


class TestMatchGraphExtendToGroupsImplicitClaims:
    """Tests for extend_to_groups() creating implicit MatchClaim objects."""

    def test_extension_creates_implicit_claims(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """extend_to_groups() adds implicit MatchClaims (case d)."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)

        # Original had 1 claim, extended should have 1 original + 1 implicit
        assert extended.n_claims == 2

        # Check that implicit claims exist
        implicit = [c for c in extended.claims if not c.is_explicit]
        assert len(implicit) == 1

        # Implicit claims should be synchronous
        for ic in implicit:
            assert ic.is_synchronous is True
            assert ic.is_explicit is False
            assert ic.start_anchor is not None

    def test_implicit_claims_have_source_claim_id(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Implicit claims have source_claim_id for traceability."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)

        implicit = [c for c in extended.claims if not c.is_explicit]
        assert len(implicit) == 1

        for ic in implicit:
            assert ic.source_claim_id == claim.id

    def test_two_groups_five_implicit_claims(self) -> None:
        """Two groups {TL1, TL4, TL5} and {TL2, TL6}: all implicit claims added.

        Scenario:
        - Group A: tl1 (1000px), tl4 (500px), tl5 (200px)
        - Group B: tl2 (800px), tl6 (400px)
        - Explicit claim: tl1@500 <-> tl2@400
        - Extension should add:
          - tl4 via group A (from tl1@500 -> tl4@250)
          - tl5 via group A (from tl1@500 -> tl5@100)
          - tl6 via group B (from tl2@400 -> tl6@200)
        """
        # Build group A: tl1, tl4, tl5
        tl1 = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="tl1")
        tl4 = DiscreteGraphicalTimeline(length=500, unit="pixels", uid="tl4")
        tl5 = DiscreteGraphicalTimeline(length=200, unit="pixels", uid="tl5")

        group_a = TimelineGroup(id="group_a")
        group_a.add_timeline(tl1)
        group_a.add_timeline(tl4)
        group_a.add_timeline(tl5)

        # Build group B: tl2, tl6
        tl2 = DiscreteGraphicalTimeline(length=800, unit="pixels", uid="tl2")
        tl6 = DiscreteGraphicalTimeline(length=400, unit="pixels", uid="tl6")

        group_b = TimelineGroup(id="group_b")
        group_b.add_timeline(tl2)
        group_b.add_timeline(tl6)

        # Explicit claim: tl1@500 <-> tl2@400
        claim = MatchClaim(
            timeline_a_id="tl1",
            timeline_b_id="tl2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="tl2",
                coordinate_b=Coordinate(400.0, TimeUnit.number),
            ),
        )
        graph = MatchGraph([claim])

        groups = {"group_a": group_a, "group_b": group_b}

        extended = graph.extend_to_groups(groups)

        # All 5 timelines should be present
        assert extended.timeline_ids == {"tl1", "tl2", "tl4", "tl5", "tl6"}

        # Check stamps: all 5 timelines should be in one connected component
        stamps = extended.get_stamps()
        assert len(stamps) == 1
        stamp = stamps[0]
        assert stamp.n_timelines == 5

        # Verify coordinates via linear interpolation
        # tl1: 1000px, tl4: 500px, tl5: 200px (all linear from 0)
        # tl1@500 -> tl4: 500 * (500/1000) = 250
        assert stamp.get_coordinate_for("tl4", format="float") == 250.0
        # tl1@500 -> tl5: 500 * (200/1000) = 100
        assert stamp.get_coordinate_for("tl5", format="float") == 100.0
        # tl2@400 -> tl6: 400 * (400/800) = 200
        assert stamp.get_coordinate_for("tl6", format="float") == 200.0

        # Count implicit claims
        implicit = [c for c in extended.claims if not c.is_explicit]
        # Exactly 3: tl1->tl4, tl1->tl5, tl2->tl6
        assert len(implicit) == 3


class TestExtendToSeveralGroups:
    """A timeline of several groups carries extension from one to the other."""

    def test_extension_follows_a_shared_member_into_its_other_group(self) -> None:
        """One claim on ``a`` reaches ``b`` only through ``t`` and ``u``.

        - Group g1: t (1200px), a (800px), u (600px)
        - Group g2: t, u, b (900px)
        - Explicit claim: a@400 <-> e@100, ``e`` in no group
        - g1 gives t@600 and u@300; their second group g2 gives b@450.
        """
        t = DiscreteGraphicalTimeline(length=1200, unit="pixels", uid="t")
        a = DiscreteGraphicalTimeline(length=800, unit="pixels", uid="a")
        u = DiscreteGraphicalTimeline(length=600, unit="pixels", uid="u")
        b = DiscreteGraphicalTimeline(length=900, unit="pixels", uid="b")
        groups = {
            "g1": TimelineGroup(id="g1", timelines=[t, a, u]),
            "g2": TimelineGroup(id="g2", timelines=[t, u, b]),
        }
        claim = MatchClaim(
            timeline_a_id="a",
            timeline_b_id="e",
            start_anchor=AlignmentAnchor(
                timeline_a_id="a",
                coordinate_a=Coordinate(400.0, TimeUnit.number),
                timeline_b_id="e",
                coordinate_b=Coordinate(100.0, TimeUnit.number),
            ),
        )

        extended = MatchGraph([claim]).extend_to_groups(groups)

        assert extended.timeline_ids == {"a", "e", "t", "u", "b"}
        (stamp,) = extended.get_stamps()
        assert stamp.get_coordinate_for("t", format="float") == 600.0
        assert stamp.get_coordinate_for("u", format="float") == 300.0
        assert stamp.get_coordinate_for("b", format="float") == 450.0
        # a->t and a->u through g1; t->u and t->b through g2 from t; u->b
        # through g2 from u (u->t exists already). A node is never extended
        # again through the group that reached it.
        implicit = [c for c in extended.claims if not c.is_explicit]
        assert len(implicit) == 5


class TestMatchGraphExtendToGroupsFilters:
    """Tests for extend_to_groups() filter parameters."""

    @pytest.fixture
    def multi_group_setup(
        self,
    ) -> tuple[
        MatchClaim,
        dict[str, TimelineGroup],
        dict[str, str],
        dict[str, DiscreteGraphicalTimeline | ContinuousPhysicalTimeline],
    ]:
        """Set up a multi-group scenario for filter tests.

        Group A: dgt_1 (1000 pixels), audio (100 seconds)
        External claim: dgt_1@500 <-> external@25
        """
        dgt1 = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="dgt_1")
        audio = ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="audio")

        group_a = TimelineGroup(id="group_a")
        group_a.add_timeline(dgt1)
        group_a.add_timeline(audio)

        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="external",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.number),
                timeline_b_id="external",
                coordinate_b=Coordinate(25.0, TimeUnit.number),
            ),
        )

        groups = {"group_a": group_a}
        timelines = {"dgt_1": dgt1, "audio": audio}

        return claim, groups, timelines

    def test_timeline_ids_filter(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """timeline_ids restricts which timelines get implicit claims."""
        claim, groups, timelines = multi_group_setup
        graph = MatchGraph([claim])

        # Only extend to dgt1 (should NOT add audio)
        extended = graph.extend_to_groups(
            groups,
            timeline_ids={"dgt_1", "external"},
        )

        # Audio should NOT be in the graph
        assert "audio" not in extended.timeline_ids
        assert extended.n_nodes == 2
        assert extended.n_edges == 1

    def test_id_pattern_filter(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """id_pattern prevents specific timelines from extension."""
        claim, groups, timelines = multi_group_setup
        graph = MatchGraph([claim])

        # Exclude audio
        extended = graph.extend_to_groups(
            groups,
            id_pattern=r"^(?!audio$).*",
        )

        assert "audio" not in extended.timeline_ids

    def test_canonical_filters_preserve_pinned_extension_count(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """Canonical ID filters reproduce the unextended graph count."""
        claim, groups, _timelines = multi_group_setup

        extended = MatchGraph([claim]).extend_to_groups(
            groups,
            timeline_ids={"dgt_1", "external"},
            id_pattern=r"^(dgt_1|external)$",
        )

        assert extended.n_nodes == 2
        assert extended.n_edges == 1

    def test_matchline_from_claims_passes_canonical_filters(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """MatchLine forwards canonical filters to group extension."""
        claim, groups, _timelines = multi_group_setup

        line = MatchLine.from_claims(
            [claim],
            source_timeline_id="dgt_1",
            groups=groups,
            timeline_ids={"dgt_1", "external"},
            id_pattern=r"^(dgt_1|external)$",
        )

        assert line.n_stamps == 1
        assert line.stamps[0].coordinates == {
            "dgt_1": Coordinate(500.0, TimeUnit.number),
            "external": Coordinate(25.0, TimeUnit.number),
        }

    def test_include_domains_filter(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """include_domains restricts extension by timeline domain."""
        claim, groups, timelines = multi_group_setup
        graph = MatchGraph([claim])

        # Only extend to graphical domain (should NOT add audio)
        extended = graph.extend_to_groups(
            groups,
            include_domains={Domain.graphical},
        )

        # Audio (physical) should NOT be extended into the graph
        assert "audio" not in extended.timeline_ids

    def test_include_units_filter(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """include_units restricts extension by timeline unit."""
        claim, groups, timelines = multi_group_setup
        graph = MatchGraph([claim])

        # Only extend to pixels (should NOT add audio/seconds)
        extended = graph.extend_to_groups(
            groups,
            include_units={TimeUnit.pixels},
        )

        # Audio (seconds) should NOT be extended
        assert "audio" not in extended.timeline_ids


class TestMatchGraphFilterPhase64:
    """Tests for filter() method with domain/unit filters."""

    def test_filter_by_include_domains(self) -> None:
        """filter() with include_domains removes timelines of wrong domain."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="audio",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.pixels),
                timeline_b_id="audio",
                coordinate_b=Coordinate(50.0, TimeUnit.seconds),
            ),
        )
        graph = MatchGraph([claim])

        # Filter to only graphical
        filtered = graph.filter(
            include_domains={Domain.graphical},
        )

        # Audio should be removed
        assert "audio" not in filtered.timeline_ids
        # Graph should have no edges (only dgt1 remains, isolated)
        assert filtered.n_edges == 0

    def test_canonical_id_filters_preserve_pinned_graph_count(self) -> None:
        """timeline_ids and id_pattern combine with AND semantics."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_b",
                timeline_b_id="tl_c",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_b",
                    coordinate_a=Coordinate(50.0, TimeUnit.number),
                    timeline_b_id="tl_c",
                    coordinate_b=Coordinate(25.0, TimeUnit.number),
                ),
            ),
        ]

        filtered = MatchGraph(claims).filter(
            timeline_ids={"tl_a", "tl_b"},
            id_pattern=r"^tl_[ab]$",
        )

        assert filtered.n_nodes == 2
        assert filtered.n_edges == 1

    def test_filter_by_include_units(self) -> None:
        """filter() with include_units removes timelines of wrong unit."""
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="audio",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(500.0, TimeUnit.pixels),
                timeline_b_id="audio",
                coordinate_b=Coordinate(50.0, TimeUnit.seconds),
            ),
        )
        graph = MatchGraph([claim])

        filtered = graph.filter(
            include_units={TimeUnit.seconds},
        )

        # dgt1 (pixels) should be removed, audio should remain but isolated
        assert "dgt_1" not in filtered.timeline_ids
        # No edges remain (isolated nodes removed)
        assert filtered.n_edges == 0

    def test_filter_keeps_non_synchronous_claims(self) -> None:
        """filter() preserves non-sync claims connecting remaining timelines."""
        sync_claim = MatchClaim(
            timeline_a_id="tl_a",
            timeline_b_id="tl_b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_a",
                coordinate_a=Coordinate(100.0, TimeUnit.number),
                timeline_b_id="tl_b",
                coordinate_b=Coordinate(50.0, TimeUnit.number),
            ),
        )
        nomatch_claim = MatchClaim.nomatch(
            event={"start": 200.0},
            source_tl_id="tl_a",
            target_tl_id="tl_b",
            unit=TimeUnit.number,
        )
        unrelated_nomatch = MatchClaim.nomatch(
            event={"start": 300.0},
            source_tl_id="tl_a",
            target_tl_id="tl_c",
            unit=TimeUnit.number,
        )
        graph = MatchGraph([sync_claim, nomatch_claim, unrelated_nomatch])

        filtered = graph.filter(timeline_ids={"tl_a", "tl_b"})

        # The sync claim and the nomatch between tl_a and tl_b should remain
        # The nomatch involving tl_c should be dropped
        assert len(filtered.claims) == 2
        claim_tl_sets = [{c.timeline_a_id, c.timeline_b_id} for c in filtered.claims]
        assert {"tl_a", "tl_b"} in claim_tl_sets
        assert {"tl_a", "tl_c"} not in claim_tl_sets


class TestMatchStampGroupRetrieval:
    """Tests for plural typed MatchStamp retrieval over group members."""

    def test_group_coordinates(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Plural retrieval uses timeline_ids in the group's declared order."""
        stamp = MatchStamp(
            coordinates={"dgt_1": 500.0, "audio": 50.0, "external": 25.0},
            anchor_edges=[("dgt_1", "external")],
            inferred_edges=[("dgt_1", "audio")],
        )

        values = stamp.get_coordinates_for(dgt1_group.timeline_ids, format="float")
        group_coords = dict(zip(dgt1_group.timeline_ids, values, strict=True))

        assert group_coords == {"dgt_1": 500.0, "audio": 50.0}
        assert "external" not in group_coords


# endregion


# region MatchStamp determinism


class TestMatchStampDeterminism:
    """A stamp renders the same in every process.

    Validation logic is documented in ``tests/alignment/README.md`` under
    "A stamp renders the same in every process".
    """

    @staticmethod
    def _claims() -> list[MatchClaim]:
        """One instant shared by five timelines, named out of lexical order."""
        names = [
            "score:clt_1",
            "Chopin_op10_no3_p20",
            "perf:dlt_1",
            "Chopin_op10_no3_p04",
            "audio:dpt_1",
        ]
        return [
            MatchClaim(
                timeline_a_id=names[0],
                timeline_b_id=other,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=names[0],
                    coordinate_a=Coordinate(Fraction(1, 2), TimeUnit.quarters),
                    timeline_b_id=other,
                    coordinate_b=Coordinate(261, TimeUnit.ticks),
                ),
            )
            for other in names[1:]
        ]

    def test_stamp_order_survives_claim_insertion_order(self) -> None:
        """Shuffled insertion gives one stamp, byte for byte."""
        claims = self._claims()
        orders = [
            claims,
            list(reversed(claims)),
            [claims[2], claims[0], claims[3], claims[1]],
            [claims[3], claims[2], claims[1], claims[0]],
        ]

        renderings = set()
        orderings = set()
        for order in orders:
            stamp = MatchGraph(claims=list(order)).get_stamps()[0]
            renderings.add(repr(stamp))
            orderings.add(tuple(stamp.present_timelines))

        assert len(renderings) == 1
        assert len(orderings) == 1

    def test_stamp_order_is_the_source_then_lexical_order(self) -> None:
        """Which deterministic order: the one the retrieval order names.

        Pinned alongside the agreement tests so an implementation that is
        stable but stable on the wrong thing fails too.
        """
        stamp = MatchGraph(claims=self._claims()).get_stamps()[0]

        assert stamp.present_timelines == [
            "Chopin_op10_no3_p04",
            "Chopin_op10_no3_p20",
            "audio:dpt_1",
            "perf:dlt_1",
            "score:clt_1",
        ]
        assert stamp.source_id == "Chopin_op10_no3_p04"

    @pytest.mark.slow
    def test_stamp_order_survives_a_changed_hash_seed(self) -> None:
        """The seed is the trigger, so the seed is what the test varies.

        Set iteration order is per-process, so one pytest run agrees with
        itself however wrong it is. Only separate interpreters under
        different seeds can show the property.
        """
        import subprocess
        import sys
        import textwrap

        program = textwrap.dedent("""
            from fractions import Fraction

            from timetoalign.alignment import AlignmentAnchor, MatchClaim
            from timetoalign.alignment.graph import MatchGraph
            from timetoalign.core import Coordinate, TimeUnit

            names = [
                "score:clt_1",
                "Chopin_op10_no3_p20",
                "perf:dlt_1",
                "Chopin_op10_no3_p04",
                "audio:dpt_1",
            ]
            claims = [
                MatchClaim(
                    timeline_a_id=names[0],
                    timeline_b_id=other,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id=names[0],
                        coordinate_a=Coordinate(Fraction(1, 2), TimeUnit.quarters),
                        timeline_b_id=other,
                        coordinate_b=Coordinate(261, TimeUnit.ticks),
                    ),
                )
                for other in names[1:]
            ]
            stamp = MatchGraph(claims=claims).get_stamps()[0]
            print(repr(stamp))
            print(stamp.present_timelines)
            """)

        outputs = set()
        for seed in ("0", "1", "12345", "99991"):
            completed = subprocess.run(
                [sys.executable, "-c", program],
                capture_output=True,
                text=True,
                check=True,
                env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            )
            outputs.add(completed.stdout)

        assert len(outputs) == 1


# endregion
