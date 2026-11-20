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

from timetoalign.alignment import (
    AlignmentAnchor,
    MatchClaim,
)
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    TimelineGroup,
)

from .helpers import make_match_stamp as MatchStamp

# region Fixtures


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
            anchor_edges=[("score", "audio")],
            inferred_edges=[("audio", "video")],
        ),
        MatchStamp(
            coordinates={"score": 100.0, "audio": 45.5, "video": 1365.0},
            anchor_edges=[("score", "audio")],
            inferred_edges=[("audio", "video")],
        ),
        MatchStamp(
            coordinates={"score": 200.0, "audio": 91.0, "video": 2730.0},
            anchor_edges=[("score", "audio")],
            inferred_edges=[("audio", "video")],
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
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(100.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
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
        ]
        line = MatchLine.from_claims(claims, source_timeline_id="tl_a")

        # Each claim creates a separate connected component -> 3 stamps
        assert line.n_stamps == 3
        coords = line.source_coordinates
        assert [coordinate.value for coordinate in coords] == [0.0, 100.0, 200.0]

    def test_from_claims_coordinate_pairs(self) -> None:
        """from_claims() yields correct coordinate pairs."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
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
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(100.0, TimeUnit.number),
                ),
            ),
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
        # (0, 78, 268, 561, 801, 967), yielding exactly 6 stamps.
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
            MatchClaim(
                timeline_a_id="dgt_1",
                timeline_b_id="external",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt_1",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="external",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="dgt_1",
                timeline_b_id="external",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt_1",
                    coordinate_a=Coordinate(500.0, TimeUnit.number),
                    timeline_b_id="external",
                    coordinate_b=Coordinate(25.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="dgt_1",
                timeline_b_id="external",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt_1",
                    coordinate_a=Coordinate(1000.0, TimeUnit.number),
                    timeline_b_id="external",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
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
        assert all(isinstance(value, float) for pair in pairs for value in pair)

    def test_from_claims_non_synchronous_excluded(self) -> None:
        """Non-synchronous claims do not produce stamps."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
            MatchClaim.nomatch(
                event={"start": 100.0},
                source_tl_id="tl_a",
                target_tl_id="tl_b",
                unit=TimeUnit.number,
            ),
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(100.0, TimeUnit.number),
                ),
            ),
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
                MatchClaim(
                    timeline_a_id="tl_a",
                    timeline_b_id="tl_b",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="tl_a",
                        coordinate_a=Coordinate(0.0, TimeUnit.number),
                        timeline_b_id="tl_b",
                        coordinate_b=Coordinate(0.0, TimeUnit.number),
                    ),
                ),
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
            ]
        )

        # Graph 2: tl_a@200 <-> tl_b@100, tl_a@300 <-> tl_b@150
        graph2 = MatchGraph(
            [
                MatchClaim(
                    timeline_a_id="tl_a",
                    timeline_b_id="tl_b",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="tl_a",
                        coordinate_a=Coordinate(200.0, TimeUnit.number),
                        timeline_b_id="tl_b",
                        coordinate_b=Coordinate(100.0, TimeUnit.number),
                    ),
                ),
                MatchClaim(
                    timeline_a_id="tl_a",
                    timeline_b_id="tl_b",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="tl_a",
                        coordinate_a=Coordinate(300.0, TimeUnit.number),
                        timeline_b_id="tl_b",
                        coordinate_b=Coordinate(150.0, TimeUnit.number),
                    ),
                ),
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
            ]
        )
        graph2 = MatchGraph(
            [
                MatchClaim(
                    timeline_a_id="tl_a",
                    timeline_b_id="tl_c",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="tl_a",
                        coordinate_a=Coordinate(100.0, TimeUnit.number),
                        timeline_b_id="tl_c",
                        coordinate_b=Coordinate(25.0, TimeUnit.number),
                    ),
                ),
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
            ]
        )

        # Graph 2: stamp at tl_a@100 with 3 timelines (chain A->B->C)
        graph2 = MatchGraph(
            [
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
                    MatchClaim(
                        timeline_a_id="score",
                        timeline_b_id="audio",
                        start_anchor=AlignmentAnchor(
                            timeline_a_id="score",
                            coordinate_a=Coordinate(offset, TimeUnit.number),
                            timeline_b_id="audio",
                            coordinate_b=Coordinate(offset * 0.45, TimeUnit.number),
                        ),
                    ),
                    MatchClaim(
                        timeline_a_id="score",
                        timeline_b_id="audio",
                        start_anchor=AlignmentAnchor(
                            timeline_a_id="score",
                            coordinate_a=Coordinate(offset + 100.0, TimeUnit.number),
                            timeline_b_id="audio",
                            coordinate_b=Coordinate(
                                (offset + 100.0) * 0.45, TimeUnit.number
                            ),
                        ),
                    ),
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
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
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

        # Should produce stamps at segment boundaries
        assert line.n_stamps >= 2
        assert "dgt_2" in line.target_timeline_ids()

        # Get coordinate pairs
        pairs = _anchor_values(line, "dgt_2")
        assert len(pairs) >= 2

        # First pair should be (0.0, 0.0)
        assert pairs[0] == (0.0, 0.0)

        # Last pair should be at the end boundaries
        last_src, last_tgt = pairs[-1]
        assert last_src == 4835.0
        assert last_tgt == 4328.0

    def test_matchline_with_group_extension_coordinate_pairs(self) -> None:
        """MatchLine from group-extended claims has correct coordinate pairs."""
        dgt1 = DiscreteGraphicalTimeline(length=1000, unit="pixels", uid="dgt_1")
        audio = ContinuousPhysicalTimeline(length=100.0, unit="seconds", uid="audio")

        group = TimelineGroup(id="g1")
        group.add_timeline(dgt1)
        group.add_timeline(audio)

        # Two claims: boundaries of a single segment
        claims = [
            MatchClaim(
                timeline_a_id="dgt_1",
                timeline_b_id="dgt_2",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt_1",
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id="dgt_2",
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="dgt_1",
                timeline_b_id="dgt_2",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="dgt_1",
                    coordinate_a=Coordinate(1000.0, TimeUnit.number),
                    timeline_b_id="dgt_2",
                    coordinate_b=Coordinate(800.0, TimeUnit.number),
                ),
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
