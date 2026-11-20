"""Tests for WarpMap.

WarpMap produces warped timeline copies from alignment data.

Test categories:
    1. Construction (from_match_line, from_coordinate_pairs)
    2. Forward / inverse round-trip precision
    3. Materialise: events, children, regions
    4. Type conversion (CLT -> CPT when warping logical -> physical)
    5. Rejection when < 2 coordinate pairs
    6. Serialization (to_dict / from_dict round-trip)
    7. Deduplication of chord coordinates
    8. Non-linear warping (tempo changes)
"""

from __future__ import annotations

import numpy as np
import pytest

from timetoalign.alignment.claims import AlignmentAnchor, MatchClaim
from timetoalign.alignment.matchline import MatchLine
from timetoalign.alignment.warpmap import WarpMap
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.timelines.base import Timeline
from timetoalign.timelines.regions import Region
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
)

from .helpers import make_match_stamp as MatchStamp

# region Fixtures


def _warp_from_values(
    source_timeline_id: str,
    target_timeline_id: str,
    source_values: list[int | float],
    target_values: list[int | float],
    *,
    source_unit: TimeUnit = TimeUnit.number,
    target_unit: TimeUnit = TimeUnit.number,
) -> WarpMap:
    """Build a WarpMap from typed ID-bearing coordinate fixtures."""
    return WarpMap.from_coordinate_pairs(
        source_timeline_id,
        target_timeline_id,
        [
            IdCoordinate(value, source_unit, source_timeline_id)
            for value in source_values
        ],
        [
            IdCoordinate(value, target_unit, target_timeline_id)
            for value in target_values
        ],
    )


def _float_at(warp: WarpMap, value: int | float) -> float:
    """Project one typed WarpMap query to a float."""
    return warp.get_coordinate_at(value, format="float")


@pytest.fixture
def linear_match_line() -> MatchLine:
    """A MatchLine with a simple linear relationship (2:1 tempo).

    Source (score) coords: [0, 100, 200]
    Target (audio) coords: [0, 50, 100]

    score_coord * 0.5 = audio_coord
    """
    stamps = [
        MatchStamp(
            coordinates={"score": 0.0, "audio": 0.0},
            coordinate_units={"score": "quarters", "audio": "seconds"},
            anchor_edges=[("score", "audio")],
        ),
        MatchStamp(
            coordinates={"score": 100.0, "audio": 50.0},
            coordinate_units={"score": "quarters", "audio": "seconds"},
            anchor_edges=[("score", "audio")],
        ),
        MatchStamp(
            coordinates={"score": 200.0, "audio": 100.0},
            coordinate_units={"score": "quarters", "audio": "seconds"},
            anchor_edges=[("score", "audio")],
        ),
    ]
    return MatchLine(source_timeline_id="score", stamps=stamps)


@pytest.fixture
def nonlinear_match_line() -> MatchLine:
    """A MatchLine with a non-linear relationship (accelerando).

    Source: [0, 100, 200, 300]
    Target: [0, 60, 100, 120]

    First half is slower (0.6 ratio), second half faster (0.2 ratio).
    """
    stamps = [
        MatchStamp(
            coordinates={"score": 0.0, "audio": 0.0},
            anchor_edges=[("score", "audio")],
        ),
        MatchStamp(
            coordinates={"score": 100.0, "audio": 60.0},
            anchor_edges=[("score", "audio")],
        ),
        MatchStamp(
            coordinates={"score": 200.0, "audio": 100.0},
            anchor_edges=[("score", "audio")],
        ),
        MatchStamp(
            coordinates={"score": 300.0, "audio": 120.0},
            anchor_edges=[("score", "audio")],
        ),
    ]
    return MatchLine(source_timeline_id="score", stamps=stamps)


@pytest.fixture
def multi_target_match_line() -> MatchLine:
    """A MatchLine with three target timelines.

    Source: score
    Targets: audio, pixels, ticks
    """
    stamps = [
        MatchStamp(
            coordinates={"score": 0.0, "audio": 0.0, "pixels": 0.0, "ticks": 0},
            anchor_edges=[("score", "audio"), ("score", "pixels")],
        ),
        MatchStamp(
            coordinates={"score": 100.0, "audio": 50.0, "pixels": 500.0, "ticks": 480},
            anchor_edges=[("score", "audio"), ("score", "pixels")],
        ),
        MatchStamp(
            coordinates={
                "score": 200.0,
                "audio": 100.0,
                "pixels": 1000.0,
                "ticks": 960,
            },
            anchor_edges=[("score", "audio"), ("score", "pixels")],
        ),
    ]
    return MatchLine(source_timeline_id="score", stamps=stamps)


@pytest.fixture
def source_timeline_with_events() -> Timeline:
    """A CLT score timeline with events, children, and regions."""
    score = ContinuousLogicalTimeline(
        length=200, unit=TimeUnit.quarters, uid="score", name="Score"
    )

    # Add instant events (beats)
    score.add_events(
        [
            {"event_type": "Beat", "instant": 0.0},
            {"event_type": "Beat", "instant": 50.0},
            {"event_type": "Beat", "instant": 100.0},
            {"event_type": "Beat", "instant": 150.0},
        ]
    )

    # Add interval events (notes)
    score.add_events(
        [
            {
                "event_type": "Note",
                "start": 0.0,
                "end": 25.0,
                "duration": 25.0,
                "pitch": "C4",
            },
            {
                "event_type": "Note",
                "start": 50.0,
                "end": 75.0,
                "duration": 25.0,
                "pitch": "E4",
            },
            {
                "event_type": "Note",
                "start": 100.0,
                "end": 150.0,
                "duration": 50.0,
                "pitch": "G4",
            },
        ]
    )

    # Add a region
    score.add_region(
        Region(
            name="Verse",
            start=Coordinate(0.0, TimeUnit.quarters),
            end=Coordinate(100.0, TimeUnit.quarters),
        )
    )
    score.add_region(
        Region(
            name="Chorus",
            start=Coordinate(100.0, TimeUnit.quarters),
            end=Coordinate(200.0, TimeUnit.quarters),
        )
    )

    return score


@pytest.fixture
def source_timeline_with_children() -> Timeline:
    """A CPT timeline with two child timelines."""
    parent = ContinuousPhysicalTimeline(
        length=100.0, unit=TimeUnit.seconds, uid="parent", name="Parent"
    )

    child1 = ContinuousPhysicalTimeline(
        length=40.0, unit=TimeUnit.seconds, uid="child1", name="Child1"
    )
    child1.add_events(
        [
            {"event_type": "Note", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"event_type": "Note", "start": 20.0, "end": 30.0, "duration": 10.0},
        ]
    )

    child2 = ContinuousPhysicalTimeline(
        length=30.0, unit=TimeUnit.seconds, uid="child2", name="Child2"
    )
    child2.add_events(
        [
            {"event_type": "Note", "start": 5.0, "end": 15.0, "duration": 10.0},
        ]
    )

    parent.add_child(child1, offset=10.0)
    parent.add_child(child2, offset=60.0)

    return parent


# endregion


# region Test Construction


class TestConstruction:
    """Test WarpMap construction from various sources."""

    def test_from_match_line_basic(self, linear_match_line: MatchLine) -> None:
        """WarpMap.from_match_line creates a valid WarpMap."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        assert warp.source_timeline_id == "score"
        assert warp.target_timeline_id == "audio"
        assert warp.n_anchors == 3

    def test_from_match_line_with_units(self, linear_match_line: MatchLine) -> None:
        """Units are preserved when specified."""
        warp = WarpMap.from_match_line(
            linear_match_line,
            "audio",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        assert warp.source_unit == TimeUnit.quarters
        assert warp.target_unit == TimeUnit.seconds

    def test_from_match_line_rejects_insufficient_pairs(self) -> None:
        """Fewer than 2 coordinate pairs raises ValueError."""
        stamps = [
            MatchStamp(
                coordinates={"score": 0.0, "audio": 0.0},
                anchor_edges=[("score", "audio")],
            ),
        ]
        line = MatchLine(source_timeline_id="score", stamps=stamps)
        with pytest.raises(ValueError, match="at least 2 coordinate pairs"):
            WarpMap.from_match_line(line, "audio")

    def test_from_match_line_rejects_missing_target(
        self, linear_match_line: MatchLine
    ) -> None:
        """Target timeline not in stamps raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            WarpMap.from_match_line(linear_match_line, "nonexistent")

    def test_from_coordinate_pairs(self) -> None:
        """Direct construction from coordinate arrays."""
        warp = _warp_from_values(
            "src",
            "tgt",
            [0.0, 100.0, 200.0],
            [0.0, 50.0, 100.0],
        )
        assert warp.source_timeline_id == "src"
        assert warp.target_timeline_id == "tgt"
        assert warp.n_anchors == 3

    def test_from_coordinate_pairs_rejects_too_few(self) -> None:
        """Fewer than 2 coordinate pairs raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 anchor points"):
            _warp_from_values("src", "tgt", [0.0], [0.0])

    def test_from_coordinate_pairs_rejects_non_monotonic(self) -> None:
        """Non-monotonic source coords raises ValueError."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            _warp_from_values("src", "tgt", [0.0, 100.0, 50.0], [0.0, 50.0, 25.0])

    def test_from_coordinate_pairs_rejects_ambiguous_duplicate_source(self) -> None:
        """A repeated source location cannot select different target times."""
        with pytest.raises(
            ValueError,
            match=r"source timeline 'edition'.*get_matchstamp_at\(\)",
        ):
            _warp_from_values(
                "edition",
                "spine",
                [0.0, 50.0, 50.0, 100.0],
                [0.0, 0.0, 22272.0, 44544.0],
            )

    def test_chord_deduplication(self) -> None:
        """Duplicate source coordinates are averaged."""
        stamps = [
            MatchStamp(
                coordinates={"score": 0.0, "audio": 0.0},
                anchor_edges=[("score", "audio")],
            ),
            # Two stamps at the same source coord (chord)
            MatchStamp(
                coordinates={"score": 100.0, "audio": 48.0},
                anchor_edges=[("score", "audio")],
            ),
            MatchStamp(
                coordinates={"score": 100.0, "audio": 52.0},
                anchor_edges=[("score", "audio")],
            ),
            MatchStamp(
                coordinates={"score": 200.0, "audio": 100.0},
                anchor_edges=[("score", "audio")],
            ),
        ]
        line = MatchLine(source_timeline_id="score", stamps=stamps)
        warp = WarpMap.from_match_line(line, "audio")

        # The duplicate at coord 100 should be averaged to 50.0
        assert warp.n_anchors == 3
        result = _float_at(warp, 100.0)
        assert result == 50.0

    def test_from_match_line_rejects_ambiguous_duplicate_source(self) -> None:
        """Repeated graphical positions cannot define an interpolation."""
        line = MatchLine(
            source_timeline_id="edition",
            stamps=[
                MatchStamp(
                    coordinates={"edition": 0.0, "spine": 0.0},
                    anchor_edges=[("edition", "spine")],
                ),
                MatchStamp(
                    coordinates={"edition": 50.0, "spine": 0.0},
                    anchor_edges=[("edition", "spine")],
                ),
                MatchStamp(
                    coordinates={"edition": 50.0, "spine": 22272.0},
                    anchor_edges=[("edition", "spine")],
                ),
                MatchStamp(
                    coordinates={"edition": 100.0, "spine": 44544.0},
                    anchor_edges=[("edition", "spine")],
                ),
            ],
        )

        with pytest.raises(
            ValueError,
            match=r"source timeline 'edition'.*get_matchstamp_at\(\)",
        ):
            WarpMap.from_match_line(line, "spine")

    def test_monotone_duplicate_source_still_averages_exactly(self) -> None:
        """Compatible duplicate anchors retain the established chord behavior."""
        warp = _warp_from_values(
            "score",
            "audio",
            [0.0, 100.0, 100.0, 200.0],
            [0.0, 48.0, 52.0, 100.0],
        )

        assert _float_at(warp, 100.0) == 50.0


# endregion


# region Test Forward / Inverse


class TestForwardInverse:
    """Test forward/inverse conversion precision."""

    def test_forward_linear(self, linear_match_line: MatchLine) -> None:
        """Linear warp: forward maps correctly."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        assert _float_at(warp, 0.0) == 0.0
        assert _float_at(warp, 100.0) == 50.0
        assert _float_at(warp, 200.0) == 100.0
        # Interpolated midpoint
        assert _float_at(warp, 50.0) == 25.0

    def test_inverse_linear(self, linear_match_line: MatchLine) -> None:
        """Linear warp: inverse maps correctly."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        inverse = warp.inverse()
        assert _float_at(inverse, 0.0) == 0.0
        assert _float_at(inverse, 50.0) == 100.0
        assert _float_at(inverse, 100.0) == 200.0
        assert warp.inverse() is inverse
        assert inverse.inverse() is warp

    def test_round_trip_precision(self, linear_match_line: MatchLine) -> None:
        """Applying a warp after its inverse recovers linear values."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        for val in [0.0, 25.0, 50.0, 75.0, 100.0]:
            # These fixed inputs recover bit-exactly through inverse-then-forward.
            inverse_value = warp.inverse().get_coordinate_at(val, format="float")
            assert warp.get_coordinate_at(inverse_value, format="float") == val

    def test_round_trip_nonlinear(self, nonlinear_match_line: MatchLine) -> None:
        """Applying a warp after its inverse recovers non-linear values."""
        warp = WarpMap.from_match_line(nonlinear_match_line, "audio")
        for val in [0.0, 30.0, 60.0, 80.0, 100.0, 120.0]:
            # These fixed inputs recover bit-exactly through inverse-then-forward.
            inverse_value = warp.inverse().get_coordinate_at(val, format="float")
            assert warp.get_coordinate_at(inverse_value, format="float") == val

    def test_forward_nonlinear(self, nonlinear_match_line: MatchLine) -> None:
        """Non-linear warp: anchor points map exactly."""
        warp = WarpMap.from_match_line(nonlinear_match_line, "audio")
        assert _float_at(warp, 0.0) == 0.0
        assert _float_at(warp, 100.0) == 60.0
        assert _float_at(warp, 200.0) == 100.0
        assert _float_at(warp, 300.0) == 120.0

    def test_forward_collection(self, linear_match_line: MatchLine) -> None:
        """The plural typed getter accepts NumPy arrays."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        coords = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
        result = warp.get_coordinates_at(coords, format="float")
        assert result == [0.0, 25.0, 50.0, 75.0, 100.0]

    def test_inverse_collection(self, linear_match_line: MatchLine) -> None:
        """Inverse accepts NumPy arrays through the plural typed getter."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        coords = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        result = warp.inverse().get_coordinates_at(coords, format="float")
        assert result == [0.0, 50.0, 100.0, 150.0, 200.0]

    def test_queries_beyond_anchor_support_raise(
        self, linear_match_line: MatchLine
    ) -> None:
        """Public retrieval rejects positions beyond the anchor hull."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        with pytest.raises(ValueError):
            _float_at(warp, 300.0)
        with pytest.raises(ValueError):
            _float_at(warp, -100.0)

    def test_is_invertible(self, linear_match_line: MatchLine) -> None:
        """Monotonic target coords -> invertible."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        assert warp.is_invertible is True


# endregion


# region Test Materialise


class TestMaterialise:
    """Tests for materialise() — producing warped timeline copies."""

    def test_materialise_warps_length(
        self,
        linear_match_line: MatchLine,
        source_timeline_with_events: Timeline,
    ) -> None:
        """Materialised timeline has correct warped length."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        warped = warp.materialise(source_timeline_with_events)
        # Source length 200, linear warp * 0.5 = 100
        assert float(warped.length.value) == 100.0

    def test_materialise_preserves_event_count(
        self,
        linear_match_line: MatchLine,
        source_timeline_with_events: Timeline,
    ) -> None:
        """Materialised timeline has same number of non-segment events."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        warped = warp.materialise(source_timeline_with_events)
        # 4 beats + 3 notes = 7 events
        assert warped.n_events == 7

    def test_materialise_warps_instant_events(
        self,
        linear_match_line: MatchLine,
        source_timeline_with_events: Timeline,
    ) -> None:
        """Instant event coordinates are warped correctly."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        warped = warp.materialise(source_timeline_with_events)

        beats = warped.get_events(event_type="Beat")
        # EventData stores instant events under "start" (as a struct dict),
        # not "instant". The "instant" key is converted on ingestion.
        instants = sorted(
            (
                float(e["start"]["value"])
                if isinstance(e["start"], dict)
                else float(e["start"])
            )
            for e in beats
        )
        # Original: [0, 50, 100, 150] -> warped by 0.5: [0, 25, 50, 75]
        assert instants == [0.0, 25.0, 50.0, 75.0]

    def test_materialise_warps_interval_events(
        self,
        linear_match_line: MatchLine,
        source_timeline_with_events: Timeline,
    ) -> None:
        """Interval event start/end/duration are warped correctly."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        warped = warp.materialise(source_timeline_with_events)

        notes = list(warped.get_events(event_type="Note"))
        # Sort by start coordinate
        notes.sort(
            key=lambda e: (
                float(e["start"])
                if not isinstance(e["start"], dict)
                else float(e["start"]["value"])
            )
        )

        def _val(v):
            return float(v["value"]) if isinstance(v, dict) else float(v)

        # Note 1: start=0, end=25, dur=25 -> start=0, end=12.5, dur=12.5
        assert _val(notes[0]["start"]) == 0.0
        assert _val(notes[0]["end"]) == 12.5
        assert _val(notes[0]["duration"]) == 12.5

        # Note 2: start=50, end=75, dur=25 -> start=25, end=37.5, dur=12.5
        assert _val(notes[1]["start"]) == 25.0
        assert _val(notes[1]["end"]) == 37.5
        assert _val(notes[1]["duration"]) == 12.5

        # Note 3: start=100, end=150, dur=50 -> start=50, end=75, dur=25
        assert _val(notes[2]["start"]) == 50.0
        assert _val(notes[2]["end"]) == 75.0
        assert _val(notes[2]["duration"]) == 25.0

    def test_materialise_warps_regions(
        self,
        linear_match_line: MatchLine,
        source_timeline_with_events: Timeline,
    ) -> None:
        """Regions are warped correctly."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        warped = warp.materialise(source_timeline_with_events)

        assert len(warped._regions) == 2

        # Each warped region keeps its source region's id and name.
        verse = warped.get_region("region_1")
        assert verse.name == "Verse"
        assert float(verse.start.value) == 0.0
        assert float(verse.end.value) == 50.0

        chorus = warped.get_region("region_2")
        assert chorus.name == "Chorus"
        assert float(chorus.start.value) == 50.0
        assert float(chorus.end.value) == 100.0

    def test_materialise_warps_children(
        self,
        source_timeline_with_children: Timeline,
    ) -> None:
        """Children are warped recursively with correct offsets."""
        # Create a WarpMap that doubles all coordinates
        warp = _warp_from_values(
            "parent",
            "target",
            [0.0, 100.0],
            [0.0, 200.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.seconds,
        )
        warped = warp.materialise(source_timeline_with_children)

        # Warped length: 100 -> 200
        assert float(warped.length.value) == 200.0

        # 2 children should be present
        assert warped.n_children == 2

        # Check that child events exist (warped)
        children = list(warped._children.values())
        # Children are keyed by new IDs, but we can check count
        assert len(children) == 2

    def test_materialise_nonlinear_durations(
        self,
        nonlinear_match_line: MatchLine,
    ) -> None:
        """Non-linear warp correctly stretches/compresses durations.

        In the nonlinear_match_line:
        - [0, 100] -> [0, 60] (ratio 0.6, slow)
        - [100, 200] -> [60, 100] (ratio 0.4, faster)

        A note at [50, 100] spans the slow region:
          warped start = warp(50) = 30.0
          warped end = warp(100) = 60.0
          warped duration = 30.0

        A note at [150, 200] spans the fast region:
          warped start = warp(150) = 80.0
          warped end = warp(200) = 100.0
          warped duration = 20.0
        """
        score = ContinuousLogicalTimeline(
            length=300, unit=TimeUnit.quarters, uid="score", name="Score"
        )
        score.add_events(
            [
                {"event_type": "Note", "start": 50.0, "end": 100.0, "duration": 50.0},
                {"event_type": "Note", "start": 150.0, "end": 200.0, "duration": 50.0},
            ]
        )

        warp = WarpMap.from_match_line(nonlinear_match_line, "audio")
        warped = warp.materialise(score)

        notes = sorted(
            list(warped.get_events(event_type="Note")),
            key=lambda e: (
                float(e["start"])
                if not isinstance(e["start"], dict)
                else float(e["start"]["value"])
            ),
        )

        def _val(v):
            return float(v["value"]) if isinstance(v, dict) else float(v)

        # Note 1: [50, 100] -> [30, 60], duration 30
        assert _val(notes[0]["start"]) == 30.0
        assert _val(notes[0]["end"]) == 60.0
        assert _val(notes[0]["duration"]) == 30.0

        # Note 2: [150, 200] -> [80, 100], duration 20
        assert _val(notes[1]["start"]) == 80.0
        assert _val(notes[1]["end"]) == 100.0
        assert _val(notes[1]["duration"]) == 20.0

    def test_materialise_rejects_wrong_source_id(
        self, linear_match_line: MatchLine
    ) -> None:
        """Materialise rejects timeline with wrong ID."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        wrong_tl = ContinuousLogicalTimeline(
            length=100, unit=TimeUnit.quarters, uid="wrong_id"
        )
        with pytest.raises(ValueError, match="does not match"):
            warp.materialise(wrong_tl)

    def test_materialise_empty_timeline(self, linear_match_line: MatchLine) -> None:
        """Materialise works on an empty timeline."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        empty = ContinuousLogicalTimeline(
            length=200, unit=TimeUnit.quarters, uid="score"
        )
        warped = warp.materialise(empty)
        assert float(warped.length.value) == 100.0
        assert warped.n_events == 0
        assert warped.n_children == 0


# endregion


# region Test Type Conversion


class TestTypeConversion:
    """Test that materialise() changes timeline type when units differ."""

    def test_logical_to_physical_type(self) -> None:
        """Warping CLT to CPT produces a ContinuousPhysicalTimeline."""
        score = ContinuousLogicalTimeline(
            length=200, unit=TimeUnit.quarters, uid="score"
        )
        score.add_events(
            [
                {"event_type": "Beat", "instant": 0.0},
                {"event_type": "Beat", "instant": 100.0},
            ]
        )

        warp = _warp_from_values(
            "score",
            "audio",
            [0.0, 200.0],
            [0.0, 100.0],
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        warped = warp.materialise(score)

        assert isinstance(warped, ContinuousPhysicalTimeline)
        assert warped.unit == TimeUnit.seconds
        assert float(warped.length.value) == 100.0

    def test_same_unit_preserves_type(self) -> None:
        """Warping within same unit preserves timeline type."""
        dgt = DiscreteGraphicalTimeline(length=1000, unit=TimeUnit.pixels, uid="dgt_1")
        dgt.add_events(
            [
                {"event_type": "Mark", "instant": 100},
                {"event_type": "Mark", "instant": 500},
            ]
        )

        warp = _warp_from_values(
            "dgt_1",
            "dgt_2",
            [0, 1000],
            [0, 2000],
            source_unit=TimeUnit.pixels,
            target_unit=TimeUnit.pixels,
        )
        warped = warp.materialise(dgt)

        # Same unit -> same type
        assert isinstance(warped, DiscreteGraphicalTimeline)
        assert warped.unit == TimeUnit.pixels


# endregion


# region Test Serialization


class TestSerialization:
    """Test to_dict / from_dict round-trip."""

    def test_round_trip(self, linear_match_line: MatchLine) -> None:
        """to_dict then from_dict preserves all state."""
        warp = WarpMap.from_match_line(
            linear_match_line,
            "audio",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        data = warp.to_dict()
        restored = WarpMap.from_dict(data)

        assert restored.source_timeline_id == warp.source_timeline_id
        assert restored.target_timeline_id == warp.target_timeline_id
        assert restored.source_unit == warp.source_unit
        assert restored.target_unit == warp.target_unit
        assert restored.n_anchors == warp.n_anchors
        assert restored.get_coordinates_at(
            [0.0, 100.0, 200.0], format="float"
        ) == warp.get_coordinates_at([0.0, 100.0, 200.0], format="float")

    def test_round_trip_default_units(self) -> None:
        """Serialization preserves inferred dimensionless units."""
        warp = _warp_from_values("a", "b", [0.0, 100.0], [0.0, 50.0])
        data = warp.to_dict()
        restored = WarpMap.from_dict(data)
        assert restored.source_unit == TimeUnit.number
        assert restored.target_unit == TimeUnit.number
        assert _float_at(restored, 50.0) == 25.0

    def test_to_dict_structure(self, linear_match_line: MatchLine) -> None:
        """to_dict contains expected keys and types."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        data = warp.to_dict()
        assert data["source_timeline_id"] == "score"
        assert data["target_timeline_id"] == "audio"
        assert data["source_unit"] == "quarters"
        assert data["target_unit"] == "seconds"
        assert WarpMap.from_dict(data).n_anchors == 3


# endregion


# region Test Display


class TestDisplay:
    """Test __repr__ and __str__."""

    def test_repr(self, linear_match_line: MatchLine) -> None:
        """repr includes source, target, and anchor count."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        r = repr(warp)
        assert "score" in r
        assert "audio" in r
        assert "3" in r

    def test_str(self, linear_match_line: MatchLine) -> None:
        """str shows source -> target."""
        warp = WarpMap.from_match_line(linear_match_line, "audio")
        s = str(warp)
        assert "score" in s
        assert "audio" in s
        assert "->" in s


# endregion


# region Test Multi-Target


class TestMultiTarget:
    """Test WarpMap with MatchLines that have multiple target timelines."""

    def test_different_targets_from_same_match_line(
        self, multi_target_match_line: MatchLine
    ) -> None:
        """Different WarpMaps for different targets from same MatchLine."""
        warp_audio = WarpMap.from_match_line(multi_target_match_line, "audio")
        warp_pixels = WarpMap.from_match_line(multi_target_match_line, "pixels")

        # Audio: 200 -> 100 (0.5x)
        assert _float_at(warp_audio, 200.0) == 100.0

        # Pixels: 200 -> 1000 (5x)
        assert _float_at(warp_pixels, 200.0) == 1000.0


# endregion


# region Test Integration with MatchClaims


class TestIntegrationWithClaims:
    """End-to-end: MatchClaim -> MatchGraph -> MatchLine -> WarpMap."""

    def test_full_pipeline(self) -> None:
        """Build a WarpMap from MatchClaims end-to-end."""
        claims = []
        for s, t in [(0.0, 0.0), (100.0, 50.0), (200.0, 100.0)]:
            anchor = AlignmentAnchor(
                timeline_a_id="score",
                coordinate_a=Coordinate(s, TimeUnit.number),
                timeline_b_id="audio",
                coordinate_b=Coordinate(t, TimeUnit.number),
            )
            claim = MatchClaim(
                timeline_a_id="score",
                timeline_b_id="audio",
                start_anchor=anchor,
                is_synchronous=True,
            )
            claims.append(claim)

        line = MatchLine.from_claims(claims, source_timeline_id="score")
        warp = WarpMap.from_match_line(line, "audio")

        assert _float_at(warp, 50.0) == 25.0
        assert _float_at(warp.inverse(), 75.0) == 150.0


# endregion


# region Test Thoresen Integration


class TestThoresenIntegration:
    """Integration tests for the Thoresen graphical timeline scenario.

    This tests the proportional transfer of events between two graphical
    timelines (DGT1, DGT2) that are described in the WarpMap specification.
    """

    def test_proportional_warp_between_dgt_timelines(self) -> None:
        """Segment boundary claims produce correct proportional warping.

        DGT1 has 5 segments of 967 pixels each (total 4835).
        DGT2 has 5 segments of [866, 867, 867, 864, 864] pixels (total 4328).

        Segment boundary alignment:
          (0, 0), (866, 967), (1733, 1934), (2600, 2901), (3464, 3868), (4328, 4835)
        """
        # Segment boundary coordinate pairs (DGT2 -> DGT1)
        dgt2_boundaries = [0, 866, 1733, 2600, 3464, 4328]
        dgt1_boundaries = [0, 967, 1934, 2901, 3868, 4835]

        warp = _warp_from_values(
            "dgt_2",
            "dgt_1",
            dgt2_boundaries,
            dgt1_boundaries,
            source_unit=TimeUnit.pixels,
            target_unit=TimeUnit.pixels,
        )

        # Verify boundary points map exactly
        for s, t in zip(dgt2_boundaries, dgt1_boundaries):
            assert warp.get_coordinate_at(s, format="int") == t

        # Event H is at position 378 within segment 2 of DGT2.
        # Segment 2 of DGT2 starts at 866, so H is at global 866 + 378 = 1244.
        event_h_global = 866 + 378
        warped_h = warp.get_coordinate_at(event_h_global, format="int")

        # Segment 2 of DGT2 is [866, 1733] -> DGT1 [967, 1934]
        # 378 / 867 * 967 + 967 = proportional position
        assert warped_h == 1389

        # Verify Event H' lands in segment 2 of DGT1: [967, 1934)
        assert 967.0 <= warped_h < 1934.0


# endregion
