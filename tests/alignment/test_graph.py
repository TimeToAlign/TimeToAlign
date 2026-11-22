"""Tests for MatchGraph and MatchStamp classes.

This module tests the graph-building layer of the alignment infrastructure:
- MatchStamp: Cross-group timestamp representation
- MatchGraph: Graph of MatchClaims with networkx integration
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from timetoshare import DEFAULT_BELIEFS

from timetoalign.alignment import (
    AlignmentAnchor,
    AlignmentBundle,
    Belief,
    InferredEdge,
    MatchClaim,
    MatchLine,
    Provenance,
)
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.graph import MatchStamp as MatchStampType
from timetoalign.core import Coordinate, IdCoordinate
from timetoalign.core.enums import Domain, TimeUnit
from timetoalign.core.timestamp import TimeStamp
from timetoalign.maps import LinearMap, TableMap
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    Timeline,
    TimelineGroup,
)

from .helpers import event_id
from .helpers import make_match_stamp as MatchStamp
from .helpers import match_claim

# region Fixtures


def _claim(
    timeline_a: str,
    value_a: float | int,
    timeline_b: str,
    value_b: float | int,
    *,
    unit_a: TimeUnit = TimeUnit.number,
    unit_b: TimeUnit = TimeUnit.number,
    **kwargs,
) -> MatchClaim:
    """A synchronous instant claim between two fixture events.

    The graph reads claims without their timelines, so dimensionless
    positions serve unless a test needs a timeline's own unit.
    """
    return match_claim(
        timeline_a,
        Coordinate(value_a, unit_a),
        timeline_b,
        Coordinate(value_b, unit_b),
        **kwargs,
    )


def _nomatch(
    timeline_id: str,
    value: float | int,
    other_timeline_id: str,
    *,
    unit: TimeUnit = TimeUnit.number,
    **kwargs,
) -> MatchClaim:
    """A NOMATCH claim: the event at *value* has no counterpart on the other side."""
    return MatchClaim(
        timeline_a_id=timeline_id,
        timeline_b_id=other_timeline_id,
        is_synchronous=False,
        event_a_id=event_id(timeline_id, Coordinate(value, unit)),
        **kwargs,
    )


def _believed_by(actor: str, belief: str) -> Provenance:
    """Provenance holding one actor's belief."""
    return Provenance(
        method="manual",
        beliefs=(Belief(actor=actor, belief=belief, date="2026-11-21"),),
    )


def _conceptual(timeline_a: str, timeline_b: str, **kwargs) -> MatchClaim:
    """A conceptual (non-synchronous) claim between two fixture sections."""
    return MatchClaim(
        timeline_a_id=timeline_a,
        timeline_b_id=timeline_b,
        is_synchronous=False,
        event_a_id=f"{timeline_a}@section",
        event_b_id=f"{timeline_b}@section",
        **kwargs,
    )


@pytest.fixture
def simple_instant_claim() -> MatchClaim:
    """Simple instant match between two timelines."""
    return _claim(
        "tl_a",
        100.0,
        "tl_b",
        50.0,
        provenance=Provenance(method="inferred", note="test manual"),
    )


@pytest.fixture
def simple_interval_claim() -> MatchClaim:
    """Simple interval match between two timelines."""
    return _claim(
        "tl_a",
        0.0,
        "tl_b",
        0.0,
        end_a=Coordinate(100.0, TimeUnit.number),
        end_b=Coordinate(50.0, TimeUnit.number),
        provenance=Provenance(method="inferred", note="test manual"),
    )


@pytest.fixture
def three_timeline_claims() -> list[MatchClaim]:
    """Claims connecting three timelines: A <-> B <-> C."""
    return [_claim("tl_a", 100.0, "tl_b", 50.0), _claim("tl_b", 50.0, "tl_c", 25.0)]


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
            stated_pairs=[("tl_a", "tl_b")],
            inferred_pairs=[],
        )
        assert stamp.n_timelines == 2
        assert stamp.n_stated_pairs == 1
        assert stamp.n_inferred_pairs == 0

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
            stated_pairs=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        filtered = stamp.filter_by_timelines(
            within={"tl_a", "tl_b"},
            id_pattern=r"^tl_[ab]$",
        )

        assert filtered.n_timelines == 2
        assert "tl_c" not in filtered.coordinates
        assert len(filtered.stated_pairs) == 1
        assert ("tl_a", "tl_b") in filtered.stated_pairs

    def test_filter_by_timelines_id_pattern(self) -> None:
        """Filter stamp with a canonical timeline ID pattern."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
            stated_pairs=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        filtered = stamp.filter_by_timelines(id_pattern=r"^(?!tl_c$).*")

        assert filtered.n_timelines == 2
        assert "tl_c" not in filtered.coordinates
        assert len(filtered.stated_pairs) == 1

    def test_filter_by_timelines_keeps_the_source(self) -> None:
        """The source is always kept; the filter selects the other timelines."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
            stated_pairs=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        for narrowing in (
            {"within": {"zzz"}},
            {"within": set()},
            {"id_pattern": r"^zzz$"},
        ):
            filtered = stamp.filter_by_timelines(**narrowing)
            assert filtered.source_id == "tl_a"
            assert filtered.coordinates == {"tl_a": stamp.coordinates["tl_a"]}
            assert filtered.stated_pairs == []

        filtered = stamp.filter_by_timelines(within={"tl_b"})
        assert list(filtered.coordinates) == ["tl_a", "tl_b"]
        assert filtered.stated_pairs == [("tl_a", "tl_b")]

        from_c = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0, "tl_c": 25.0},
            source_id="tl_c",
            stated_pairs=[("tl_a", "tl_b"), ("tl_b", "tl_c")],
        )
        filtered = from_c.filter_by_timelines(within={"tl_b"})
        assert filtered.source_id == "tl_c"
        assert list(filtered.coordinates) == ["tl_b", "tl_c"]
        assert filtered.stated_pairs == [("tl_b", "tl_c")]

    def test_to_dict_roundtrip(self) -> None:
        """Serialize and deserialize MatchStamp."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            stated_pairs=[("tl_a", "tl_b")],
            inferred_pairs=[("tl_b", "tl_c")],
        )
        data = stamp.to_dict(format="graph")
        restored = MatchStampType.from_dict(data)

        assert restored.coordinates == stamp.coordinates
        assert restored.stated_pairs == stamp.stated_pairs
        assert restored.inferred_pairs == stamp.inferred_pairs

    def test_graph_dict_isolation(self) -> None:
        """Graph dictionaries do not expose the stamp's mutable containers."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            stated_pairs=[("tl_a", "tl_b")],
            inferred_pairs=[("tl_b", "tl_c")],
        )
        data = stamp.to_dict(format="graph")
        data["coordinates"]["tl_a"] = 999.0
        data["stated_pairs"].append(("tl_c", "tl_d"))
        data["inferred_pairs"].clear()

        assert stamp.coordinates == {
            "tl_a": Coordinate(100.0, TimeUnit.number),
            "tl_b": Coordinate(50.0, TimeUnit.number),
        }
        assert stamp.stated_pairs == [("tl_a", "tl_b")]
        assert stamp.inferred_pairs == [("tl_b", "tl_c")]

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
            "stated_pairs": [("tl_a", "tl_b")],
            "inferred_pairs": [("tl_b", "tl_c")],
        }
        stamp = MatchStampType.from_dict(data)
        data["coordinates"]["tl_a"] = 999.0
        data["stated_pairs"].append(("tl_c", "tl_d"))
        data["inferred_pairs"].clear()

        assert stamp.coordinates == {
            "tl_a": Coordinate(100.0, TimeUnit.number),
            "tl_b": Coordinate(50.0, TimeUnit.number),
        }
        assert stamp.stated_pairs == [("tl_a", "tl_b")]
        assert stamp.inferred_pairs == [("tl_b", "tl_c")]

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
            stated_pairs=[("audio", "score")],
        )

        assert repr(stamp) == "MatchStamp(audio=25/2 seconds, score=415/24 quarters)"
        assert str(stamp) == (
            "MatchStamp (2 timelines, 1 edges)\n"
            "  audio     25/2 seconds  stated\n"
            "  score  415/24 quarters  stated"
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
            "style='text-align: right;'>25/2 seconds</td><td><em>stated</em></td></tr>"
            "<tr><td><strong>score</strong></td><td style='text-align: right;'>"
            "415/24 quarters</td><td><em>stated</em></td></tr></tbody></table><div "
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
        assert len(stamp.stated_pairs) == 1

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
            _claim("tl_a", 100.0, "tl_b", 50.0),
            _claim("tl_c", 200.0, "tl_d", 75.0),  # Disconnected
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
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        graph = MatchGraph([claim])

        # Set up group mapping
        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)

        # Original graph: 2 nodes, 1 edge
        # Extended graph should have 3 nodes (dgt1, external, audio)
        # and 2 edges (dgt1-external stated, dgt1-audio inferred)
        assert extended.n_nodes == 3
        assert extended.n_edges == 2

    def test_extend_creates_correct_coordinates(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Extended edges have correct converted coordinates."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
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
        """Extended edges are marked as inferred, not stated."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups)
        stamps = extended.get_stamps()
        assert len(stamps) == 1
        stamp = stamps[0]

        # Should have 1 stated pair (dgt1-external)
        # and 1 inferred pair (dgt1-audio or audio-external)
        assert stamp.n_stated_pairs == 1
        assert stamp.n_inferred_pairs == 1

    def test_extend_with_include_inferred_false(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """include_inferred=False returns original graph."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        graph = MatchGraph([claim])

        groups = {"group1": dgt1_group}

        extended = graph.extend_to_groups(groups, include_inferred=False)

        # Should be same as original
        assert extended.n_nodes == 2
        assert extended.n_edges == 1


# endregion


# region MatchGraph Filter Tests


# The belief filter over TestMatchGraphBeliefFilter's five claims: the
# believer, the states kept, the ids kept in order.
_BELIEF_CASES = [
    # No believer: nothing is filtered.
    (None, DEFAULT_BELIEFS, ["claim_1", "claim_2", "claim_3", "claim_4", "claim_5"]),
    (None, ["true"], ["claim_1", "claim_2", "claim_3", "claim_4", "claim_5"]),
    # ana believes claim_2 false; ben's belief in claim_4 is not hers.
    ("ana", DEFAULT_BELIEFS, ["claim_1", "claim_3", "claim_4", "claim_5"]),
    # Only what ana believes true.
    ("ana", ["true"], ["claim_3"]),
    # What ana holds no belief on.
    ("ana", ["undecided"], ["claim_1", "claim_4"]),
    # What ana believes unlikely.
    ("ana", ["unlikely"], ["claim_5"]),
    # ben believes claim_4 false; ana's beliefs are not his.
    ("ben", DEFAULT_BELIEFS, ["claim_1", "claim_2", "claim_3", "claim_5"]),
    ("ben", ["true"], []),
    ("ben", ["false"], ["claim_4"]),
]


class TestMatchGraphBeliefFilter:
    """A graph holds exactly the claims the belief filter keeps."""

    @staticmethod
    def _claims() -> list[MatchClaim]:
        """Five claims at distinct positions, every belief state occurring."""

        return [
            _claim("tl_a", 1.0, "tl_b", 10.0, id="claim_1"),
            _claim(
                "tl_a",
                2.0,
                "tl_b",
                20.0,
                id="claim_2",
                provenance=_believed_by("ana", "false"),
            ),
            _claim(
                "tl_a",
                3.0,
                "tl_b",
                30.0,
                id="claim_3",
                provenance=_believed_by("ana", "true"),
            ),
            _claim(
                "tl_a",
                4.0,
                "tl_b",
                40.0,
                id="claim_4",
                provenance=_believed_by("ben", "false"),
            ),
            _claim(
                "tl_a",
                5.0,
                "tl_b",
                50.0,
                id="claim_5",
                provenance=_believed_by("ana", "unlikely"),
            ),
        ]

    @pytest.mark.parametrize(("believer", "beliefs", "kept"), _BELIEF_CASES)
    def test_graph_keeps_the_believed_claims(
        self, believer: str | None, beliefs: list[str], kept: list[str]
    ) -> None:
        """Claims, nodes and edges are those of the kept claims alone."""
        graph = MatchGraph(self._claims(), believer=believer, beliefs=beliefs)

        assert [claim.id for claim in graph.claims] == kept
        assert graph.n_edges == len(kept)
        assert graph.n_nodes == 2 * len(kept)

    def test_invalid_state_is_refused(self) -> None:
        """A word that is no belief state is refused."""
        with pytest.raises(ValueError, match="'not_false' is no belief state"):
            MatchGraph(self._claims(), believer="ana", beliefs=["not_false"])


def _edges(graph: MatchGraph) -> set[frozenset]:
    """A graph's edges, each as the set of the two nodes it joins."""
    return {frozenset(edge) for edge in graph._graph.edges()}


def _assert_rebuilt(filtered: MatchGraph, rebuilt: MatchGraph) -> None:
    """*filtered* holds exactly what *rebuilt* holds.

    The claims as objects in order, the edge sets, the inferred edges by
    anchor, group and source claim identity, and the rendered stamps.
    """
    assert len(filtered.claims) == len(rebuilt.claims)
    assert all(
        mine is theirs
        for mine, theirs in zip(filtered.claims, rebuilt.claims, strict=True)
    )
    assert _edges(filtered) == _edges(rebuilt)
    assert [
        (edge.anchor, edge.group_id, id(edge.source_claim))
        for edge in filtered.inferred_edges
    ] == [
        (edge.anchor, edge.group_id, id(edge.source_claim))
        for edge in rebuilt.inferred_edges
    ]
    assert [str(stamp) for stamp in filtered.get_stamps()] == [
        str(stamp) for stamp in rebuilt.get_stamps()
    ]


def _inferred(graph: MatchGraph) -> list[tuple[AlignmentAnchor, str, MatchClaim]]:
    """A graph's inferred edges as (anchor, group, source claim)."""
    return [
        (edge.anchor, edge.group_id, edge.source_claim) for edge in graph.inferred_edges
    ]


def _pixels_to_seconds(pixels: int, seconds: float) -> AlignmentAnchor:
    """The anchor the group {dgt_1 1000 px, audio 100 s} infers."""
    return AlignmentAnchor(
        timeline_a_id="dgt_1",
        coordinate_a=Coordinate(pixels, TimeUnit.pixels),
        timeline_b_id="audio",
        coordinate_b=Coordinate(seconds, TimeUnit.seconds),
    )


def _is(found: list[MatchClaim], expected: list[MatchClaim]) -> bool:
    """Whether *found* holds exactly the *expected* objects, in order."""
    return len(found) == len(expected) and all(
        mine is theirs for mine, theirs in zip(found, expected, strict=True)
    )


class TestMatchGraphFilterByClaims:
    """``filter`` holds what the graph rebuilt from the kept claims holds."""

    @pytest.mark.parametrize(("believer", "beliefs", "kept"), _BELIEF_CASES)
    def test_filter_keeps_the_believed_claims(
        self, believer: str | None, beliefs: list[str], kept: list[str]
    ) -> None:
        """The constructor's answer, asked of a whole graph."""
        claims = TestMatchGraphBeliefFilter._claims()
        graph = MatchGraph(claims)

        filtered = graph.filter(believer=believer, beliefs=beliefs)

        assert [claim.id for claim in filtered.claims] == kept
        assert filtered.n_edges == len(kept)
        assert filtered.n_nodes == 2 * len(kept)
        _assert_rebuilt(
            filtered, MatchGraph(claims, believer=believer, beliefs=beliefs)
        )

    def test_filter_drops_the_inferred_edges_of_a_disbelieved_claim(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """An inferred edge goes with the only claims it was inferred from."""
        claims = [
            _claim(
                "dgt_1",
                500,
                "external",
                25.0,
                unit_a=TimeUnit.pixels,
                id="claim_1",
                provenance=_believed_by("ana", "false"),
            ),
            _claim(
                "dgt_1", 200, "external", 10.0, unit_a=TimeUnit.pixels, id="claim_2"
            ),
        ]
        groups = {"group1": dgt1_group}
        extended = MatchGraph(claims).extend_to_groups(groups)
        assert (extended.n_edges, len(extended.inferred_edges)) == (4, 2)

        filtered = extended.filter(believer="ana")

        assert _is(filtered.claims, [claims[1]])
        assert (filtered.n_edges, filtered.n_nodes) == (2, 3)
        assert filtered.timeline_ids == {"dgt_1", "external", "audio"}
        assert _inferred(filtered) == [
            (_pixels_to_seconds(200, 20.0), "group1", claims[1])
        ]
        rebuilt = MatchGraph(claims, believer="ana").extend_to_groups(groups)
        _assert_rebuilt(filtered, rebuilt)

    def test_filter_keeps_an_inferred_edge_another_kept_claim_grounds(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """A node still stated by a kept claim keeps its inferred edges."""
        claims = [
            _claim(
                "dgt_1",
                500,
                "external",
                25.0,
                unit_a=TimeUnit.pixels,
                id="claim_1",
                provenance=_believed_by("ana", "false"),
            ),
            _claim("dgt_1", 500, "score", 7, unit_a=TimeUnit.pixels, id="claim_2"),
        ]
        groups = {"group1": dgt1_group}
        extended = MatchGraph(claims).extend_to_groups(groups)
        # The node dgt_1@500 traces back to the first claim naming it.
        assert extended.inferred_edges[0].source_claim is claims[0]

        filtered = extended.filter(believer="ana")

        assert _is(filtered.claims, [claims[1]])
        assert (filtered.n_edges, filtered.n_nodes) == (2, 3)
        assert filtered.timeline_ids == {"dgt_1", "score", "audio"}
        assert _inferred(filtered) == [
            (_pixels_to_seconds(500, 50.0), "group1", claims[1])
        ]
        rebuilt = MatchGraph(claims, believer="ana").extend_to_groups(groups)
        _assert_rebuilt(filtered, rebuilt)

    def test_filter_infers_again_an_anchor_a_dropped_claim_stated(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """A dropped claim stating the group's own anchor leaves it inferred."""
        claims = [
            _claim(
                "dgt_1",
                500,
                "audio",
                50.0,
                unit_a=TimeUnit.pixels,
                unit_b=TimeUnit.seconds,
                id="claim_1",
                provenance=_believed_by("ana", "false"),
            ),
            _claim(
                "dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels, id="claim_2"
            ),
        ]
        groups = {"group1": dgt1_group}
        extended = MatchGraph(claims).extend_to_groups(groups)
        assert (extended.n_edges, extended.inferred_edges) == (2, [])

        filtered = extended.filter(believer="ana")

        assert _is(filtered.claims, [claims[1]])
        assert (filtered.n_nodes, filtered.n_edges) == (3, 2)
        assert _inferred(filtered) == [
            (_pixels_to_seconds(500, 50.0), "group1", claims[1])
        ]
        assert [str(stamp) for stamp in filtered.get_stamps()] == [
            "MatchStamp (3 timelines, 2 edges)\n"
            "  audio     50 seconds  inferred\n"
            "  dgt_1     500 pixels  stated\n"
            "  external   25 number  stated"
        ]
        rebuilt = MatchGraph(claims, believer="ana").extend_to_groups(groups)
        _assert_rebuilt(filtered, rebuilt)

    def test_inferred_edges_name_only_kept_claims(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """A kept inferred edge traces back to a kept claim."""
        claims = [
            _claim(
                "dgt_1",
                200,
                "external",
                10.0,
                unit_a=TimeUnit.pixels,
                id="claim_3",
                provenance=_believed_by("ana", "false"),
            ),
            _claim("dgt_1", 200, "score", 3, unit_a=TimeUnit.pixels, id="claim_4"),
        ]
        groups = {"group1": dgt1_group}
        extended = MatchGraph(claims).extend_to_groups(groups)
        assert _inferred(extended) == [
            (_pixels_to_seconds(200, 20.0), "group1", claims[0])
        ]

        filtered = extended.filter(believer="ana")

        assert _is(filtered.claims, [claims[1]])
        assert _inferred(filtered) == [
            (_pixels_to_seconds(200, 20.0), "group1", claims[1])
        ]
        rebuilt = MatchGraph(claims, believer="ana").extend_to_groups(groups)
        _assert_rebuilt(filtered, rebuilt)

    def test_filter_keeps_an_anchor_a_kept_claim_also_states(self) -> None:
        """An edge two claims state stays, naming the kept claim."""
        kept = _claim("tl_a", 1.0, "tl_b", 10.0, id="claim_2")
        dropped = _claim(
            "tl_a",
            1.0,
            "tl_b",
            10.0,
            id="claim_1",
            provenance=_believed_by("ana", "false"),
        )
        graph = MatchGraph([kept, dropped])

        filtered = graph.filter(believer="ana")

        assert _is(filtered.claims, [kept])
        assert filtered.n_edges == 1
        [(_, _, data)] = filtered._graph.edges(data=True)
        assert data["claim"] is kept
        assert data["claim_id"] == "claim_2"
        # The graph filtered keeps its own edge as it was.
        [(_, _, original)] = graph._graph.edges(data=True)
        assert original["claim"] is dropped
        _assert_rebuilt(filtered, MatchGraph([kept]))

    def test_filter_by_timeline_id_ids_and_pattern(self) -> None:
        """A claim with a matching timeline on either side is kept."""
        claims = [
            _claim("tl_a", 100.0, "tl_b", 50.0, id="claim_1"),
            _claim("tl_b", 50.0, "tl_c", 25.0, id="claim_2"),
        ]
        graph = MatchGraph(claims)
        first, second = claims

        cases = [
            ({"timeline_ids": {"tl_a"}}, [first]),
            ({"timeline_ids": {"tl_a", "tl_b"}}, [first, second]),
            ({"timeline_id": "tl_c"}, [second]),
            ({"timeline_id": "tl_x"}, []),
            ({"id_pattern": r"^tl_c$"}, [second]),
            ({"id_pattern": r"^(?!tl_c$).*"}, [first, second]),
            ({"timeline_ids": {"tl_a", "tl_b"}, "id_pattern": r"^tl_c$"}, [second]),
        ]
        for criteria, kept in cases:
            filtered = graph.filter(**criteria)
            assert _is(filtered.claims, kept), criteria
            _assert_rebuilt(filtered, MatchGraph(kept))

        both = graph.filter(timeline_ids={"tl_a", "tl_b"})
        assert both.timeline_ids == {"tl_a", "tl_b", "tl_c"}
        assert (both.n_edges, [stamp.n_timelines for stamp in both.get_stamps()]) == (
            2,
            [3],
        )
        assert graph.filter(timeline_ids={"tl_a"}).timeline_ids == {"tl_a", "tl_b"}
        assert graph.filter(timeline_id="tl_x").n_nodes == 0

    def test_filter_within(self) -> None:
        """A claim is kept when both its timelines are in the set."""
        claims = [
            _claim("tl_a", 100.0, "tl_b", 50.0, id="claim_1"),
            _claim("tl_b", 50.0, "tl_c", 25.0, id="claim_2"),
        ]
        graph = MatchGraph(claims)

        within = graph.filter(within={"tl_a", "tl_b"})

        assert _is(within.claims, [claims[0]])
        assert (within.timeline_ids, within.n_edges) == ({"tl_a", "tl_b"}, 1)
        _assert_rebuilt(within, MatchGraph([claims[0]]))
        assert graph.filter(within={"tl_a"}).claims == []

    def test_filter_within_on_an_extended_graph(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """The extension reaches only the timelines ``within`` admits."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        groups = {"group1": dgt1_group}
        extended = MatchGraph([claim]).extend_to_groups(groups)

        nobody = extended.filter(within={"dgt_1", "audio"})
        assert (nobody.claims, nobody.n_nodes, nobody.n_edges) == ([], 0, 0)
        assert (nobody.inferred_edges, nobody.get_stamps()) == ([], [])

        stated = extended.filter(within={"dgt_1", "external"})
        assert _is(stated.claims, [claim])
        assert (stated.n_nodes, stated.n_edges, stated.inferred_edges) == (2, 1, [])
        _assert_rebuilt(
            stated,
            MatchGraph([claim]).extend_to_groups(groups, within={"dgt_1", "external"}),
        )

        either = extended.filter(timeline_ids={"dgt_1", "audio"})
        assert _is(either.claims, [claim])
        assert (either.n_nodes, either.n_edges) == (3, 2)
        assert _inferred(either) == [(_pixels_to_seconds(500, 50.0), "group1", claim)]
        _assert_rebuilt(either, extended)

    def test_filter_stated_only(self, dgt1_group: TimelineGroup) -> None:
        """``stated_only`` is the graph of the kept claims, not extended."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        extended = MatchGraph([claim]).extend_to_groups({"group1": dgt1_group})

        filtered = extended.filter(stated_only=True)

        assert _is(filtered.claims, [claim])
        assert (filtered.n_edges, filtered.inferred_edges) == (1, [])
        assert filtered.timeline_ids == {"dgt_1", "external"}
        _assert_rebuilt(filtered, MatchGraph([claim]))

    def test_filter_nomatch_only(self) -> None:
        """NOMATCH claims alone, under the other filters; sync claims alone."""
        sync = _claim("tl_a", 1.0, "tl_b", 10.0, id="sync")
        nomatch_1 = _nomatch(
            "tl_a", 2.0, "tl_b", id="nomatch_1", provenance=_believed_by("ana", "false")
        )
        nomatch_2 = _nomatch("tl_a", 3.0, "tl_c", id="nomatch_2")
        conceptual = _conceptual("tl_b", "tl_c", id="conceptual")
        graph = MatchGraph([sync, nomatch_1, nomatch_2, conceptual])

        def ids(filtered: MatchGraph) -> list[str | None]:
            return [claim.id for claim in filtered.claims]

        nomatches = graph.filter(nomatch_only=True)
        assert ids(nomatches) == ["nomatch_1", "nomatch_2"]
        assert (nomatches.n_nodes, nomatches.n_edges) == (0, 0)
        assert ids(graph.filter(nomatch_only=True, timeline_ids={"tl_a", "tl_b"})) == [
            "nomatch_1",
            "nomatch_2",
        ]
        assert ids(graph.filter(nomatch_only=True, within={"tl_a", "tl_b"})) == [
            "nomatch_1"
        ]
        assert ids(graph.filter(nomatch_only=True, believer="ana")) == ["nomatch_2"]
        synchronous = graph.filter(synchronous_only=True)
        assert ids(synchronous) == ["sync"]
        assert synchronous.n_edges == 1
        assert ids(graph.filter()) == ["sync", "nomatch_1", "nomatch_2", "conceptual"]
        with pytest.raises(
            ValueError,
            match="^synchronous_only and nomatch_only are mutually exclusive$",
        ):
            graph.filter(synchronous_only=True, nomatch_only=True)

    def test_filter_timeline_ids_keeps_either_side(self) -> None:
        """``timeline_ids`` keeps what the bundle's MatchClaim query keeps."""
        sync = _claim("tl_a", 1.0, "tl_b", 10.0, id="sync")
        nm = _nomatch("tl_c", 1.0, "tl_d", id="nm")
        nm_ab = _nomatch("tl_a", 2.0, "tl_b", id="nm_ab")

        filtered = MatchGraph([sync, nm, nm_ab]).filter(timeline_ids={"tl_a"})

        assert _is(filtered.claims, [sync, nm_ab])
        _assert_rebuilt(filtered, MatchGraph([sync, nm_ab]))

    def test_filter_nomatch_only_on_an_extended_graph(
        self, dgt1_group: TimelineGroup
    ) -> None:
        """A NOMATCH claim holds no anchor: no stated and no inferred edge."""
        sync = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        nomatch = _nomatch("dgt_1", 100, "external", unit=TimeUnit.pixels)
        extended = MatchGraph([sync, nomatch]).extend_to_groups({"group1": dgt1_group})

        filtered = extended.filter(nomatch_only=True)

        assert _is(filtered.claims, [nomatch])
        assert (filtered.n_nodes, filtered.n_edges) == (0, 0)
        assert filtered.inferred_edges == []


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
            _claim("tl_a", 100.0, "tl_b", 50.0),
            _nomatch("tl_a", 200.0, "tl_b"),
        ]
        graph = MatchGraph(claims)

        # Only the synchronous claim creates edges
        assert graph.n_claims == 2
        assert graph.n_nodes == 2  # Only from the synchronous claim
        assert graph.n_edges == 1

    def test_non_synchronous_claims_accessible_via_claims(self) -> None:
        """Non-synchronous claims are accessible via claims property."""
        sync_claim = _claim("tl_a", 100.0, "tl_b", 50.0)
        nomatch_claim = _nomatch("tl_a", 200.0, "tl_c")
        graph = MatchGraph([sync_claim, nomatch_claim])

        assert len(graph.claims) == 2
        assert len(graph.synchronous_claims) == 1
        assert len(graph.non_synchronous_claims) == 1
        assert graph.non_synchronous_claims[0] is nomatch_claim

    def test_only_non_synchronous_claims_empty_graph(self) -> None:
        """Graph with only non-synchronous claims has no nodes or edges."""
        claims = [
            _nomatch("tl_a", 100.0, "tl_b"),
            _nomatch("tl_c", 200.0, "tl_d"),
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
            _claim("tl_a", 100.0, "tl_b", 50.0),
            _claim("tl_c", 200.0, "tl_d", 75.0),  # Disconnected
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


class TestMatchGraphInferredEdges:
    """extend_to_groups() records what it infers as the graph's own edges."""

    def test_extension_infers_edges_not_claims(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """The extended graph holds the given claim and one inferred edge."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        graph = MatchGraph([claim])

        extended = graph.extend_to_groups({"group1": dgt1_group})

        assert extended.claims == [claim]
        assert extended.n_claims == 1
        assert graph.inferred_edges == []
        (edge,) = extended.inferred_edges
        assert isinstance(edge, InferredEdge)
        assert edge.group_id == "group1"
        assert edge.source_claim is claim
        # dgt_1 is 1000 px and audio 100 s: 500 px is 50 s, each coordinate on
        # its timeline's own axis.
        assert edge.anchor == AlignmentAnchor(
            timeline_a_id="dgt_1",
            coordinate_a=Coordinate(500, TimeUnit.pixels),
            timeline_b_id="audio",
            coordinate_b=Coordinate(50.0, TimeUnit.seconds),
        )
        assert (extended.n_nodes, extended.n_edges) == (3, 2)

    def test_inferred_edge_names_its_source_claim(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """An inferred edge traces back to the claim it started from."""
        claim = _claim(
            "dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels, id="claim_1"
        )

        extended = MatchGraph([claim]).extend_to_groups({"group1": dgt1_group})

        assert [edge.source_claim.id for edge in extended.inferred_edges] == ["claim_1"]
        assert repr(extended.inferred_edges[0]) == (
            "InferredEdge(AlignmentAnchor(dgt_1@500 pixels <-> audio@50 seconds) "
            "via group1, from claim_1)"
        )

    def test_two_groups_three_inferred_edges(self) -> None:
        """Two groups {TL1, TL4, TL5} and {TL2, TL6}: every edge inferred.

        Scenario:
        - Group A: tl1 (1000px), tl4 (500px), tl5 (200px)
        - Group B: tl2 (800px), tl6 (400px)
        - Claim: tl1@500 <-> tl2@400
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

        # The claim: tl1@500 <-> tl2@400
        claim = _claim(
            "tl1", 500, "tl2", 400, unit_a=TimeUnit.pixels, unit_b=TimeUnit.pixels
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

        # Exactly 3 inferred edges: tl1->tl4, tl1->tl5, tl2->tl6
        assert [
            (edge.anchor.timeline_a_id, edge.anchor.timeline_b_id)
            for edge in extended.inferred_edges
        ] == [("tl1", "tl4"), ("tl1", "tl5"), ("tl2", "tl6")]
        assert extended.claims == [claim]


class TestExtendToSeveralGroups:
    """A timeline of several groups carries extension from one to the other."""

    def test_extension_follows_a_shared_member_into_its_other_group(self) -> None:
        """One claim on ``a`` reaches ``b`` only through ``t`` and ``u``.

        - Group g1: t (1200px), a (800px), u (600px)
        - Group g2: t, u, b (900px)
        - Claim: a@400 <-> e@100, ``e`` in no group
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
        claim = _claim("a", 400, "e", 100.0, unit_a=TimeUnit.pixels)

        extended = MatchGraph([claim]).extend_to_groups(groups)

        assert extended.timeline_ids == {"a", "e", "t", "u", "b"}
        (stamp,) = extended.get_stamps()
        assert stamp.get_coordinate_for("t", format="float") == 600.0
        assert stamp.get_coordinate_for("u", format="float") == 300.0
        assert stamp.get_coordinate_for("b", format="float") == 450.0
        # a->t and a->u through g1; t->u and t->b through g2 from t; u->b
        # through g2 from u (u->t exists already). A node is never extended
        # again through the group that reached it.
        assert extended.claims == [claim]
        assert len(extended.inferred_edges) == 5
        assert [edge.group_id for edge in extended.inferred_edges] == [
            "g1",
            "g1",
            "g2",
            "g2",
            "g2",
        ]


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

        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)

        groups = {"group_a": group_a}
        timelines = {"dgt_1": dgt1, "audio": audio}

        return claim, groups, timelines

    def test_within_filter(
        self,
        multi_group_setup: tuple,
    ) -> None:
        """``within`` restricts which timelines get inferred edges."""
        claim, groups, timelines = multi_group_setup
        graph = MatchGraph([claim])

        # Only extend to dgt1 (should NOT add audio)
        extended = graph.extend_to_groups(
            groups,
            within={"dgt_1", "external"},
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
            within={"dgt_1", "external"},
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
            within={"dgt_1", "external"},
            id_pattern=r"^(dgt_1|external)$",
        )

        assert line.n_stamps == 1
        assert line.stamps[0].coordinates == {
            "dgt_1": Coordinate(500, TimeUnit.pixels),
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


class TestMatchGraphFilterByDomainAndUnit:
    """``filter()`` by domain and unit: both timelines of a claim pass."""

    def test_filter_by_include_domains(self) -> None:
        """A claim with one timeline in another domain is dropped."""
        claim = _claim(
            "dgt_1", 500, "audio", 50.0, unit_a=TimeUnit.pixels, unit_b=TimeUnit.seconds
        )
        graph = MatchGraph([claim])

        graphical = graph.filter(include_domains={Domain.graphical})
        assert (graphical.claims, graphical.n_nodes) == ([], 0)
        both = graph.filter(include_domains={Domain.graphical, Domain.physical})
        assert _is(both.claims, [claim])
        _assert_rebuilt(both, graph)

    def test_filter_by_include_units(self) -> None:
        """A claim with one timeline in another unit is dropped."""
        claim = _claim(
            "dgt_1", 500, "audio", 50.0, unit_a=TimeUnit.pixels, unit_b=TimeUnit.seconds
        )
        graph = MatchGraph([claim])

        seconds = graph.filter(include_units={TimeUnit.seconds})
        assert (seconds.claims, seconds.n_nodes) == ([], 0)
        both = graph.filter(include_units={TimeUnit.pixels, TimeUnit.seconds})
        assert _is(both.claims, [claim])
        _assert_rebuilt(both, graph)

    def test_domain_and_unit_filters_read_the_source_bundle(self) -> None:
        """A claim on timelines no anchor names passes by its bundle's units."""
        bundle = AlignmentBundle(id="units")
        for uid in ("tl_a", "tl_b", "tl_c"):
            bundle.add_timeline(
                ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid=uid)
            )
        bundle.add_timeline(
            DiscreteGraphicalTimeline(length=100, unit="pixels", uid="tl_d")
        )
        sync = _claim(
            "tl_a",
            1.0,
            "tl_b",
            10.0,
            unit_a=TimeUnit.seconds,
            unit_b=TimeUnit.seconds,
            id="sync",
        )
        nm_cd = _nomatch("tl_c", 1.0, "tl_d", unit=TimeUnit.seconds, id="nm_cd")
        graph = MatchGraph([sync, nm_cd], source=bundle)

        seconds = {TimeUnit.seconds}
        assert _is(graph.filter(include_units=seconds).claims, [sync])
        assert _is(
            graph.filter(include_units={TimeUnit.seconds, TimeUnit.pixels}).claims,
            [sync, nm_cd],
        )
        assert _is(
            graph.filter(
                nomatch_only=True,
                include_domains={Domain.physical, Domain.graphical},
            ).claims,
            [nm_cd],
        )
        # Without a source, no anchor names tl_c or tl_d: their units are unknown.
        assert _is(
            MatchGraph([sync, nm_cd])
            .filter(include_units={TimeUnit.seconds, TimeUnit.pixels})
            .claims,
            [sync],
        )

    def test_filters_compose_without_a_source(self, dgt1_group: TimelineGroup) -> None:
        """A filtered graph keeps the units of the graph it was filtered from."""
        claim = _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels)
        nm = _nomatch("audio", 30.0, "external", unit=TimeUnit.seconds, id="nm")
        extended = MatchGraph([claim, nm]).extend_to_groups({"group1": dgt1_group})
        units = {TimeUnit.seconds, TimeUnit.number}

        at_once = extended.filter(nomatch_only=True, include_units=units)
        nomatch_first = extended.filter(nomatch_only=True).filter(include_units=units)
        units_first = extended.filter(include_units=units).filter(nomatch_only=True)

        assert _is(at_once.claims, [nm])
        assert _is(nomatch_first.claims, [nm])
        assert _is(units_first.claims, [nm])

    def test_canonical_id_filters_combine(self) -> None:
        """``timeline_ids`` and ``id_pattern`` combine with AND semantics."""
        claims = [
            _claim("tl_a", 100.0, "tl_b", 50.0),
            _claim("tl_b", 50.0, "tl_c", 25.0),
        ]

        filtered = MatchGraph(claims).filter(
            timeline_ids={"tl_a", "tl_b"},
            id_pattern=r"^tl_[ab]$",
        )

        assert _is(filtered.claims, claims)
        assert (filtered.n_nodes, filtered.n_edges) == (3, 2)

    def test_filter_keeps_non_synchronous_claims(self) -> None:
        """``within`` keeps the non-synchronous claims between its timelines."""
        sync_claim = _claim("tl_a", 100.0, "tl_b", 50.0)
        nomatch_claim = _nomatch("tl_a", 200.0, "tl_b")
        unrelated_nomatch = _nomatch("tl_a", 300.0, "tl_c")
        graph = MatchGraph([sync_claim, nomatch_claim, unrelated_nomatch])

        filtered = graph.filter(within={"tl_a", "tl_b"})

        assert _is(filtered.claims, [sync_claim, nomatch_claim])
        _assert_rebuilt(filtered, MatchGraph([sync_claim, nomatch_claim]))


class TestMatchStampGroupRetrieval:
    """Tests for plural typed MatchStamp retrieval over group members."""

    def test_group_coordinates(
        self,
        dgt1_group: TimelineGroup,
    ) -> None:
        """Plural retrieval uses timeline_ids in the group's declared order."""
        stamp = MatchStamp(
            coordinates={"dgt_1": 500.0, "audio": 50.0, "external": 25.0},
            stated_pairs=[("dgt_1", "external")],
            inferred_pairs=[("dgt_1", "audio")],
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
            match_claim(
                names[0],
                Coordinate(Fraction(1, 2), TimeUnit.quarters),
                other,
                Coordinate(261, TimeUnit.ticks),
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
        import os
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
                    event_a_id=f"{names[0]}@1/2",
                    event_b_id=f"{other}@261",
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
                env={
                    "PYTHONHASHSEED": seed,
                    "PATH": "/usr/bin:/bin",
                    # The interpreters import the library under test.
                    **{
                        key: os.environ[key]
                        for key in ("PYTHONPATH",)
                        if key in os.environ
                    },
                },
            )
            outputs.add(completed.stdout)

        assert len(outputs) == 1


# endregion


# region Conversion map choice


class TestConversionMapChoice:
    """A ``conversion_maps`` spec chooses the same map on both stamps.

    Validation logic is documented in ``tests/alignment/README.md`` under
    "One spec, one map, for both stamps".
    """

    AMBIGUOUS = (
        "Timeline 'clock' has 2 conversion maps targeting 'milliseconds'; name"
        " the one you mean. Candidates: clock-ms (seconds_to_milliseconds),"
        " clock-ms-offset (seconds_to_milliseconds)"
    )

    @staticmethod
    def _clock() -> tuple[Timeline, LinearMap]:
        """``clock`` with two millisecond maps, 25 s -> 25000 and 25005 ms."""
        clock = Timeline(length=100, unit=TimeUnit.seconds, uid="clock")
        clock.add_conversion_map(
            TableMap(
                x_values=[0.0, 100.0],
                y_values=[0.0, 100000.0],
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.milliseconds,
                uid="clock-ms",
            )
        )
        offset = LinearMap(
            scalar=1000,
            offset=5,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="clock-ms-offset",
        )
        clock.add_conversion_map(offset)
        return clock, offset

    @classmethod
    def _stamp(cls, kind: str, spec: object) -> TimeStamp | MatchStampType:
        clock, offset = cls._clock()
        if spec == "the offset map":
            spec = [offset]
        if kind == "TimeStamp":
            return clock.get_timestamp(25.0, conversion_maps=spec)
        bundle = AlignmentBundle(id="maps")
        bundle.add_timeline(clock, as_group="clock-group")
        return bundle.get_matchstamp_at(25.0, "clock", conversion_maps=spec)

    @pytest.mark.parametrize("kind", ["TimeStamp", "MatchStamp"])
    @pytest.mark.parametrize(
        ("spec", "expected"),
        [
            (["clock-ms-offset"], 25005.0),
            ("the offset map", 25005.0),
            (["clock-ms"], 25000.0),
        ],
    )
    def test_a_spec_naming_one_of_two_maps_converts_through_it(
        self, kind: str, spec: object, expected: float
    ) -> None:
        stamp = self._stamp(kind, spec)

        assert type(stamp).__name__ == kind
        assert stamp.get_unit(TimeUnit.milliseconds, format="float") == expected
        # The text shows the one conversion row, with the value `get_unit` gave.
        rows = [line.split() for line in str(stamp).splitlines()[1:]]
        assert rows == [
            ["clock", "25", "seconds"],
            ["milliseconds", str(int(expected)), "milliseconds"],
        ]

    @pytest.mark.parametrize("kind", ["TimeStamp", "MatchStamp"])
    @pytest.mark.parametrize("spec", [True, ["clock-ms", "clock-ms-offset"]])
    def test_two_enabled_maps_are_ambiguous(self, kind: str, spec: object) -> None:
        stamp = self._stamp(kind, spec)

        with pytest.raises(ValueError) as refusal:
            stamp.get_unit(TimeUnit.milliseconds)
        assert str(refusal.value) == self.AMBIGUOUS

    @pytest.mark.parametrize(
        ("kind", "name"),
        [("TimeStamp", "stamp from 'clock'"), ("MatchStamp", "MatchStamp")],
    )
    def test_no_enabled_map_converts_nothing(self, kind: str, name: str) -> None:
        stamp = self._stamp(kind, False)

        with pytest.raises(KeyError) as refusal:
            stamp.get_unit(TimeUnit.milliseconds)
        assert refusal.value.args[0] == (
            f"No eligible conversion to 'milliseconds' on {name}"
        )


# endregion
