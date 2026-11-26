"""Tests for MatchLine class.

MatchLine: ordered sequence of MatchStamps for WarpMap generation.

Tests cover:
- Construction from stamps (direct)
- from_claims() with ordering verification
- from_graphs() merging multiple MatchGraphs (Hendrix M6-M9 pattern)
- get_alignment_anchors() extraction
- target_timeline_ids() returns only timelines with >= 2 stamps
- Serialization round-trip
- Edge cases (empty, single stamp, stamps missing source)
"""

from __future__ import annotations

import pytest

from timetoalign.alignment import MatchClaim
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import NOMATCH, Coordinate, TimeUnit
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
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
) -> MatchClaim:
    """A synchronous instant claim between two fixture events.

    A MatchLine reads claims without their timelines, so dimensionless
    positions serve; a claim on a pixel timeline states pixels.
    """
    return match_claim(
        timeline_a,
        Coordinate(value_a, unit_a),
        timeline_b,
        Coordinate(value_b, unit_b),
    )


def _nomatch(timeline_id: str, value: float, other_timeline_id: str) -> MatchClaim:
    """A NOMATCH claim: the event at *value* has no counterpart on the other side."""
    return MatchClaim(
        timeline_a_id=timeline_id,
        timeline_b_id=other_timeline_id,
        is_synchronous=False,
        event_a_id=event_id(timeline_id, Coordinate(value, TimeUnit.number)),
        event_b_id=NOMATCH,
    )


def _anchor_values(
    line: MatchLine, target_timeline_id: str
) -> list[tuple[int | float, int | float]]:
    """Return values after validating typed alignment-anchor identities."""
    anchors = line.get_alignment_anchors(target_timeline_id)
    assert all(anchor.timeline_a_id == line.source_timeline_id for anchor in anchors)
    assert all(anchor.timeline_b_id == target_timeline_id for anchor in anchors)
    return [
        (anchor.coordinate_a.value, anchor.coordinate_b.value) for anchor in anchors
    ]


@pytest.fixture
def three_stamp_line() -> MatchLine:
    """MatchLine with three stamps spanning score <-> audio <-> video.

    Stamps:
        score=0.0, audio=0.0, video=0.0
        score=100.0, audio=45.5, video=1365.0
        score=200.0, audio=91.0, video=2730.0
    """
    stamps = [
        MatchStamp(
            coordinates={"score": 0.0, "audio": 0.0, "video": 0.0},
            stated_pairs=[("score", "audio")],
            inferred_pairs=[("audio", "video")],
        ),
        MatchStamp(
            coordinates={"score": 100.0, "audio": 45.5, "video": 1365.0},
            stated_pairs=[("score", "audio")],
            inferred_pairs=[("audio", "video")],
        ),
        MatchStamp(
            coordinates={"score": 200.0, "audio": 91.0, "video": 2730.0},
            stated_pairs=[("score", "audio")],
            inferred_pairs=[("audio", "video")],
        ),
    ]
    return MatchLine(source_timeline_id="score", stamps=stamps)


# thoresen_segment_claims fixture is provided by conftest.py

# endregion


# region Direct Construction Tests


class TestMatchLineBasic:
    """Basic MatchLine construction and properties."""

    def test_basic_creation(self, three_stamp_line: MatchLine) -> None:
        """Create a MatchLine with three stamps."""
        assert three_stamp_line.n_stamps == 3
        assert three_stamp_line.source_timeline_id == "score"

    def test_stamps_sorted_by_source_coordinate(self) -> None:
        """Stamps are sorted by source coordinate even if provided out of order."""
        stamps = [
            MatchStamp(coordinates={"src": 200.0, "tgt": 20.0}),
            MatchStamp(coordinates={"src": 0.0, "tgt": 0.0}),
            MatchStamp(coordinates={"src": 100.0, "tgt": 10.0}),
        ]
        line = MatchLine(source_timeline_id="src", stamps=stamps)

        assert line.n_stamps == 3
        coords = line.source_coordinates
        assert [coordinate.value for coordinate in coords] == [0.0, 100.0, 200.0]
        assert all(coordinate.timeline_id == "src" for coordinate in coords)

    def test_empty_matchline(self) -> None:
        """Create an empty MatchLine."""
        line = MatchLine(source_timeline_id="src")
        assert line.n_stamps == 0
        assert line.source_coordinates == []
        assert line.target_timeline_ids() == set()

    def test_single_stamp(self) -> None:
        """MatchLine with a single stamp."""
        stamp = MatchStamp(
            coordinates={"src": 50.0, "tgt": 25.0},
        )
        line = MatchLine(source_timeline_id="src", stamps=[stamp])

        assert line.n_stamps == 1
        assert [coordinate.value for coordinate in line.source_coordinates] == [50.0]
        # Single stamp: no target has >= 2 stamps
        assert line.target_timeline_ids() == set()

    def test_stamps_without_source_are_dropped(self) -> None:
        """Stamps that don't contain the source timeline are dropped."""
        stamps = [
            MatchStamp(coordinates={"src": 0.0, "tgt": 0.0}),
            MatchStamp(coordinates={"other": 50.0, "tgt": 25.0}),  # No src
            MatchStamp(coordinates={"src": 100.0, "tgt": 10.0}),
        ]
        line = MatchLine(source_timeline_id="src", stamps=stamps)

        assert line.n_stamps == 2
        assert [coordinate.value for coordinate in line.source_coordinates] == [
            0.0,
            100.0,
        ]

    def test_source_coordinates_property(self, three_stamp_line: MatchLine) -> None:
        """source_coordinates returns sorted list of source coordinates."""
        assert [
            coordinate.value for coordinate in three_stamp_line.source_coordinates
        ] == [
            0.0,
            100.0,
            200.0,
        ]


# endregion


# region target_timeline_ids Tests


class TestTargetTimelineIds:
    """Tests for target_timeline_ids() method."""

    def test_returns_timelines_with_two_or_more_stamps(
        self, three_stamp_line: MatchLine
    ) -> None:
        """Returns timelines appearing in >= 2 stamps."""
        targets = three_stamp_line.target_timeline_ids()
        # audio and video appear in all 3 stamps
        assert targets == {"audio", "video"}

    def test_excludes_source_timeline(self, three_stamp_line: MatchLine) -> None:
        """Source timeline is never in target_timeline_ids."""
        targets = three_stamp_line.target_timeline_ids()
        assert "score" not in targets

    def test_excludes_timelines_with_single_stamp(self) -> None:
        """Timelines appearing in only 1 stamp are excluded."""
        stamps = [
            MatchStamp(coordinates={"src": 0.0, "tgt_a": 0.0, "tgt_b": 0.0}),
            MatchStamp(coordinates={"src": 100.0, "tgt_a": 10.0}),  # tgt_b absent
        ]
        line = MatchLine(source_timeline_id="src", stamps=stamps)

        targets = line.target_timeline_ids()
        assert "tgt_a" in targets
        assert "tgt_b" not in targets

    def test_empty_line_returns_empty_set(self) -> None:
        """Empty MatchLine returns empty set."""
        line = MatchLine(source_timeline_id="src")
        assert line.target_timeline_ids() == set()


# endregion


# region get_alignment_anchors Tests


class TestGetAlignmentAnchors:
    """Tests for get_alignment_anchors() method."""

    def test_basic_extraction(self, three_stamp_line: MatchLine) -> None:
        """Extract coordinate pairs for a target timeline."""
        pairs = _anchor_values(three_stamp_line, "audio")
        assert pairs == [(0.0, 0.0), (100.0, 45.5), (200.0, 91.0)]

    def test_video_pairs(self, three_stamp_line: MatchLine) -> None:
        """Extract coordinate pairs for video target."""
        pairs = _anchor_values(three_stamp_line, "video")
        assert pairs == [(0.0, 0.0), (100.0, 1365.0), (200.0, 2730.0)]

    def test_missing_target_returns_partial(self) -> None:
        """Stamps without the target timeline are skipped."""
        stamps = [
            MatchStamp(coordinates={"src": 0.0, "tgt": 0.0}),
            MatchStamp(coordinates={"src": 50.0}),  # No tgt
            MatchStamp(coordinates={"src": 100.0, "tgt": 10.0}),
        ]
        line = MatchLine(source_timeline_id="src", stamps=stamps)
        pairs = _anchor_values(line, "tgt")

        assert len(pairs) == 2
        assert pairs == [(0.0, 0.0), (100.0, 10.0)]

    def test_nonexistent_target_returns_empty(
        self, three_stamp_line: MatchLine
    ) -> None:
        """Non-existent target returns empty list."""
        with pytest.raises(KeyError):
            three_stamp_line.get_alignment_anchors("nonexistent")

    def test_same_as_source_raises(self, three_stamp_line: MatchLine) -> None:
        """target_timeline_id same as source raises ValueError."""
        with pytest.raises(ValueError, match="cannot be the same"):
            three_stamp_line.get_alignment_anchors("score")

    def test_pairs_ordered_by_source_coordinate(self) -> None:
        """Pairs are always ordered by source coordinate."""
        stamps = [
            MatchStamp(coordinates={"src": 300.0, "tgt": 30.0}),
            MatchStamp(coordinates={"src": 100.0, "tgt": 10.0}),
            MatchStamp(coordinates={"src": 200.0, "tgt": 20.0}),
        ]
        line = MatchLine(source_timeline_id="src", stamps=stamps)
        pairs = _anchor_values(line, "tgt")

        assert pairs == [(100.0, 10.0), (200.0, 20.0), (300.0, 30.0)]


# endregion


# region from_claims Tests


class TestFromClaims:
    """Tests for MatchLine.from_claims() class method."""

    def test_from_claims_ordering(self) -> None:
        """from_claims() produces stamps ordered by source coordinate."""
        claims = [
            _claim("tl_a", 200.0, "tl_b", 100.0),
            _claim("tl_a", 0.0, "tl_b", 0.0),
            _claim("tl_a", 100.0, "tl_b", 50.0),
        ]
        line = MatchLine.from_claims(claims, source_timeline_id="tl_a")

        # Each claim creates a separate connected component -> 3 stamps
        assert line.n_stamps == 3
        coords = line.source_coordinates
        assert [coordinate.value for coordinate in coords] == [0.0, 100.0, 200.0]

    def test_from_claims_coordinate_pairs(self) -> None:
        """from_claims() yields correct coordinate pairs."""
        claims = [
            _claim("tl_a", 0.0, "tl_b", 0.0),
            _claim("tl_a", 100.0, "tl_b", 50.0),
            _claim("tl_a", 200.0, "tl_b", 100.0),
        ]
        line = MatchLine.from_claims(claims, source_timeline_id="tl_a")
        pairs = _anchor_values(line, "tl_b")

        assert pairs == [(0.0, 0.0), (100.0, 50.0), (200.0, 100.0)]

    def test_from_claims_with_interval_claims(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """from_claims() works with interval claims (Thoresen PoC)."""
        line = MatchLine.from_claims(
            thoresen_segment_claims, source_timeline_id="dgt_1"
        )

        # 5 intervals produce 6 unique boundary coordinates on dgt1
        # (0, 967, 1934, 2901, 3868, 4835), yielding exactly 6 stamps.
        assert line.n_stamps == 6
        assert "dgt_2" in line.target_timeline_ids()

        # Verify ordering
        coords = line.source_coordinates
        assert [coordinate.value for coordinate in coords] == sorted(
            coordinate.value for coordinate in coords
        )

        # First coordinate should be 0.0
        assert coords[0].value == 0.0

    def test_from_claims_with_group_extension(self) -> None:
        """from_claims() with group extension adds group member coordinates."""
        dgt1 = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="dgt_1")
        audio = ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="audio")

        group = TimelineGroup(id="g1")
        group.add_timeline(dgt1)
        group.add_timeline(audio)

        claims = [
            _claim("dgt_1", 0, "external", 0.0, unit_a=TimeUnit.pixels),
            _claim("dgt_1", 500, "external", 25.0, unit_a=TimeUnit.pixels),
            _claim("dgt_1", 1000, "external", 50.0, unit_a=TimeUnit.pixels),
        ]

        line = MatchLine.from_claims(
            claims,
            source_timeline_id="dgt_1",
            groups={"g1": group},
        )

        # Should have 3 stamps, each containing dgt1, external, and audio
        assert line.n_stamps == 3
        assert "audio" in line.target_timeline_ids()
        assert "external" in line.target_timeline_ids()

        # Audio coordinates should be linearly mapped
        pairs = _anchor_values(line, "audio")
        assert len(pairs) == 3
        assert pairs[0] == (0.0, 0.0)
        assert pairs[1] == (500.0, 50.0)
        assert pairs[2] == (1000.0, 100.0)
        # Each value is on its timeline's axis: pixels are integers, seconds floats.
        assert all(type(source) is int for source, _ in pairs)
        assert all(type(target) is float for _, target in pairs)

    def test_from_claims_non_synchronous_excluded(self) -> None:
        """Non-synchronous claims do not produce stamps."""
        claims = [
            _claim("tl_a", 0.0, "tl_b", 0.0),
            _nomatch("tl_a", 100.0, "tl_b"),
            _claim("tl_a", 200.0, "tl_b", 100.0),
        ]
        line = MatchLine.from_claims(claims, source_timeline_id="tl_a")

        # Only 2 synchronous claims produce stamps
        assert line.n_stamps == 2
        assert [coordinate.value for coordinate in line.source_coordinates] == [
            0.0,
            200.0,
        ]


# endregion


# region from_graphs Tests


class TestFromGraphs:
    """Tests for MatchLine.from_graphs() class method."""

    def test_from_graphs_merges_stamps(self) -> None:
        """from_graphs() merges MatchStamps from multiple graphs."""
        # Graph 1: tl_a@0 <-> tl_b@0, tl_a@100 <-> tl_b@50
        graph1 = MatchGraph(
            [
                _claim("tl_a", 0.0, "tl_b", 0.0),
                _claim("tl_a", 100.0, "tl_b", 50.0),
            ]
        )

        # Graph 2: tl_a@200 <-> tl_b@100, tl_a@300 <-> tl_b@150
        graph2 = MatchGraph(
            [
                _claim("tl_a", 200.0, "tl_b", 100.0),
                _claim("tl_a", 300.0, "tl_b", 150.0),
            ]
        )

        line = MatchLine.from_graphs([graph1, graph2], source_timeline_id="tl_a")

        assert line.n_stamps == 4
        assert [coordinate.value for coordinate in line.source_coordinates] == [
            0.0,
            100.0,
            200.0,
            300.0,
        ]

    def test_from_graphs_deduplicates_by_source_coordinate(self) -> None:
        """from_graphs() deduplicates stamps at the same source coordinate."""
        # Both graphs have a stamp at tl_a@100
        graph1 = MatchGraph(
            [
                _claim("tl_a", 100.0, "tl_b", 50.0),
            ]
        )
        graph2 = MatchGraph(
            [
                _claim("tl_a", 100.0, "tl_c", 25.0),
            ]
        )

        line = MatchLine.from_graphs([graph1, graph2], source_timeline_id="tl_a")

        # Should be 1 stamp (deduplicated), keeping the one with more TLs
        # But both have 2 timelines, so either could be kept
        assert line.n_stamps == 1
        assert [coordinate.value for coordinate in line.source_coordinates] == [100.0]

    def test_from_graphs_keeps_richer_stamp(self) -> None:
        """When deduplicating, keeps the stamp with more timelines."""
        # Graph 1: stamp at tl_a@100 with 2 timelines
        graph1 = MatchGraph(
            [
                _claim("tl_a", 100.0, "tl_b", 50.0),
            ]
        )

        # Graph 2: stamp at tl_a@100 with 3 timelines (chain A->B->C)
        graph2 = MatchGraph(
            [
                _claim("tl_a", 100.0, "tl_b", 50.0),
                _claim("tl_b", 50.0, "tl_c", 25.0),
            ]
        )

        line = MatchLine.from_graphs([graph1, graph2], source_timeline_id="tl_a")

        assert line.n_stamps == 1
        # The graph2 stamp has 3 timelines (A, B, C connected)
        assert line.stamps[0].n_timelines == 3

    def test_from_graphs_hendrix_pattern(self) -> None:
        """Hendrix M6-M9 pattern: merge four contiguous M-box graphs."""
        # Simulate 4 contiguous M-boxes, each with one claim
        graphs = []
        for i in range(4):
            offset = i * 100.0
            graph = MatchGraph(
                [
                    _claim("score", offset, "audio", offset * 0.45),
                    _claim("score", offset + 100.0, "audio", (offset + 100.0) * 0.45),
                ]
            )
            graphs.append(graph)

        line = MatchLine.from_graphs(graphs, source_timeline_id="score")

        # 5 unique source coordinates: 0, 100, 200, 300, 400
        # (boundaries are shared/deduplicated)
        assert line.n_stamps == 5
        assert [coordinate.value for coordinate in line.source_coordinates] == [
            0.0,
            100.0,
            200.0,
            300.0,
            400.0,
        ]
        assert "audio" in line.target_timeline_ids()

    def test_from_graphs_empty_list(self) -> None:
        """from_graphs() with empty list returns empty MatchLine."""
        line = MatchLine.from_graphs([], source_timeline_id="src")
        assert line.n_stamps == 0

    def test_from_graphs_single_graph(self) -> None:
        """from_graphs() with single graph is equivalent to from_claims."""
        claims = [
            _claim("tl_a", 0.0, "tl_b", 0.0),
            _claim("tl_a", 100.0, "tl_b", 50.0),
        ]
        graph = MatchGraph(claims)

        line = MatchLine.from_graphs([graph], source_timeline_id="tl_a")

        assert line.n_stamps == 2
        assert [coordinate.value for coordinate in line.source_coordinates] == [
            0.0,
            100.0,
        ]


# endregion


# region Serialization Tests


class TestMatchLineSerialization:
    """Tests for MatchLine serialization."""

    def test_to_dict_roundtrip(self, three_stamp_line: MatchLine) -> None:
        """Serialize and deserialize MatchLine."""
        data = three_stamp_line.to_dict()
        restored = MatchLine.from_dict(data)

        assert restored.source_timeline_id == "score"
        assert restored.n_stamps == 3
        assert [coordinate.value for coordinate in restored.source_coordinates] == [
            0.0,
            100.0,
            200.0,
        ]

        # Coordinate pairs should be preserved
        original_pairs = three_stamp_line.get_alignment_anchors("audio")
        restored_pairs = restored.get_alignment_anchors("audio")
        assert original_pairs == restored_pairs

    def test_to_dict_structure(self, three_stamp_line: MatchLine) -> None:
        """to_dict() has expected structure."""
        data = three_stamp_line.to_dict()
        assert "source_timeline_id" in data
        assert "stamps" in data
        assert isinstance(data["stamps"], list)
        assert len(data["stamps"]) == 3

    def test_empty_roundtrip(self) -> None:
        """Empty MatchLine survives round-trip."""
        line = MatchLine(source_timeline_id="src")
        data = line.to_dict()
        restored = MatchLine.from_dict(data)

        assert restored.source_timeline_id == "src"
        assert restored.n_stamps == 0

    def test_repr(self, three_stamp_line: MatchLine) -> None:
        """Test string representation."""
        repr_str = repr(three_stamp_line)
        assert "MatchLine" in repr_str
        assert "score" in repr_str
        assert "stamps=3" in repr_str
        assert "audio" in repr_str
        assert "video" in repr_str


# endregion


# region Integration Tests


class TestMatchLineIntegration:
    """Integration tests with real-world-like scenarios."""

    def test_thoresen_matchline(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """Build MatchLine from Thoresen segment claims and extract pairs."""
        line = MatchLine.from_claims(
            thoresen_segment_claims,
            source_timeline_id="dgt_1",
        )

        # One stamp per segment boundary: 6 on each side.
        assert line.n_stamps == 6
        assert line.target_timeline_ids() == {"dgt_2"}

        # DGT1: 5 x 967 px; DGT2: [866, 867, 867, 864, 864] px.
        assert _anchor_values(line, "dgt_2") == [
            (0, 0),
            (967, 866),
            (1934, 1733),
            (2901, 2600),
            (3868, 3464),
            (4835, 4328),
        ]

    def test_matchline_with_group_extension_coordinate_pairs(self) -> None:
        """MatchLine from group-extended claims has correct coordinate pairs."""
        dgt1 = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="dgt_1")
        audio = ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="audio")

        group = TimelineGroup(id="g1")
        group.add_timeline(dgt1)
        group.add_timeline(audio)

        # Two claims: boundaries of a single segment
        claims = [
            _claim(
                "dgt_1", 0, "dgt_2", 0, unit_a=TimeUnit.pixels, unit_b=TimeUnit.pixels
            ),
            _claim(
                "dgt_1",
                1000,
                "dgt_2",
                800,
                unit_a=TimeUnit.pixels,
                unit_b=TimeUnit.pixels,
            ),
        ]

        line = MatchLine.from_claims(
            claims,
            source_timeline_id="dgt_1",
            groups={"g1": group},
        )

        # Audio should be available as a target
        assert "audio" in line.target_timeline_ids()

        audio_pairs = _anchor_values(line, "audio")
        assert len(audio_pairs) == 2
        assert audio_pairs[0] == (0.0, 0.0)
        assert audio_pairs[1] == (1000.0, 100.0)

        dgt2_pairs = _anchor_values(line, "dgt_2")
        assert len(dgt2_pairs) == 2
        assert dgt2_pairs == [(0.0, 0.0), (1000.0, 800.0)]


# endregion
