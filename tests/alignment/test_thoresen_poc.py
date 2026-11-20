"""Thoresen Proof of Concept - Integration Test Scaffold.

This test validates the alignment infrastructure using the Thoresen example
discussed in the TISMIR article (https://doi.org/10.5334/tismir.296).

The example involves two graphical analyses of the same musical content:
- DGT1 (Thoresen 2009): 5 equal-width segments (single image, 5 systems)
- DGT2 (Thoresen 2010): 5 varying-width segments (5 separate images)

The goal is to transfer Event H from DGT2 to DGT1 via piecewise linear
interpolation based on segment correspondence.

Test data is now loaded via GraphicalLoader fixtures in conftest.py.
Coordinate values come from Applications.ipynb and ground truth TSV files.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment import MatchClaim
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    TimelineGroup,
)

# Import shared Thoresen constants from conftest (canonical source)
from .conftest import (
    AUDIO_DURATION_SECONDS,
    DGT1_SEGMENT_LENGTH,
    DGT1_TOTAL_WIDTH,
    DGT2_SEGMENT_BOUNDS,
    DGT2_SEGMENT_LENGTHS,
    DGT2_TOTAL_WIDTH,
)

# Derived constant used throughout the tests
DGT1_SEGMENT_LENGTHS = [DGT1_SEGMENT_LENGTH] * 5  # [967, 967, 967, 967, 967]

# region Event H Data (specific to this PoC test)

# === Event H (rect_h2) from ground truth TSV ===
# From events_df row 4: annot_cue_005, rect_h2
# Ground truth: start_time_sec=43.5, duration_sec=4.50
# Pixel coords in image: {'x': 385, 'y': 46, 'width': 139, 'height': 20}
# Image: page1_2.jpeg (segment index 1, 0-indexed)
#
# The x coordinate (385) is in image space. To get segment-local coordinate:
# Segment 2 has x0=7, so local_x = 385 - 7 = 378
EVENT_H_SEGMENT_INDEX = 1  # Second segment (page1_2.jpeg)
EVENT_H_IMAGE_X = 385  # x in image coordinates
EVENT_H_IMAGE_WIDTH = 139  # width in image coordinates
EVENT_H_X0_IN_IMAGE = DGT2_SEGMENT_BOUNDS[EVENT_H_SEGMENT_INDEX][0]  # x0=7
EVENT_H_START_IN_SEGMENT = EVENT_H_IMAGE_X - EVENT_H_X0_IN_IMAGE  # 385 - 7 = 378
EVENT_H_END_IN_SEGMENT = (
    EVENT_H_START_IN_SEGMENT + EVENT_H_IMAGE_WIDTH
)  # 378 + 139 = 517

# Ground truth timing for validation
EVENT_H_GROUND_TRUTH_START_SEC = 43.5
EVENT_H_GROUND_TRUTH_DURATION_SEC = 4.50
EVENT_H_GROUND_TRUTH_END_SEC = (
    EVENT_H_GROUND_TRUTH_START_SEC + EVENT_H_GROUND_TRUTH_DURATION_SEC
)  # 48.0


# endregion


# region Fixtures
# Timeline fixtures (dgt1_timeline, dgt2_timeline, audio_timeline) and
# thoresen_thoresen_segment_claims are provided by conftest.py.


@pytest.fixture
def dgt1_group(
    dgt1_timeline: DiscreteGraphicalTimeline,
    audio_timeline: ContinuousPhysicalTimeline,
) -> TimelineGroup:
    """Create TimelineGroup for DGT1 with audio alignment.

    Maps audio (0-150s) to DGT1 (0-4835 pixels) linearly.
    """
    group = TimelineGroup(id="dgt1_group", name="DGT1_Group")
    group.add_timeline(dgt1_timeline)
    group.add_timeline(audio_timeline)
    return group


@pytest.fixture
def dgt2_group(dgt2_timeline: DiscreteGraphicalTimeline) -> TimelineGroup:
    """Create TimelineGroup for DGT2."""
    return TimelineGroup(id="dgt2_group", name="DGT2_Group", timelines=[dgt2_timeline])


# endregion


# region Data Validation Tests


class TestThoresenDataIntegrity:
    """Validate the input data consistency."""

    def test_dgt1_segments_sum_to_total(self) -> None:
        """DGT1 segment lengths must sum to total width."""
        assert sum(DGT1_SEGMENT_LENGTHS) == DGT1_TOTAL_WIDTH, (
            f"DGT1 segments sum to {sum(DGT1_SEGMENT_LENGTHS)}, "
            f"expected {DGT1_TOTAL_WIDTH}"
        )

    def test_dgt2_segments_sum_to_total(self) -> None:
        """DGT2 segment lengths must sum to total width."""
        assert sum(DGT2_SEGMENT_LENGTHS) == DGT2_TOTAL_WIDTH, (
            f"DGT2 segments sum to {sum(DGT2_SEGMENT_LENGTHS)}, "
            f"expected {DGT2_TOTAL_WIDTH}"
        )

    def test_dgt1_exact_values(self) -> None:
        """DGT1 values match Applications.ipynb exactly."""
        assert DGT1_SEGMENT_LENGTH == 967
        assert DGT1_TOTAL_WIDTH == 4835
        assert DGT1_SEGMENT_LENGTHS == [967, 967, 967, 967, 967]

    def test_dgt2_exact_values(self) -> None:
        """DGT2 values match Applications.ipynb exactly."""
        assert DGT2_SEGMENT_LENGTHS == [866, 867, 867, 864, 864]
        assert DGT2_TOTAL_WIDTH == 4328

    def test_five_segments_each(self) -> None:
        """Both analyses have exactly 5 segments."""
        assert len(DGT1_SEGMENT_LENGTHS) == 5
        assert len(DGT2_SEGMENT_LENGTHS) == 5

    def test_event_h_within_segment(self) -> None:
        """Event H coordinates must be within its segment."""
        segment_length = DGT2_SEGMENT_LENGTHS[EVENT_H_SEGMENT_INDEX]
        assert (
            0 <= EVENT_H_START_IN_SEGMENT < segment_length
        ), f"Event H start {EVENT_H_START_IN_SEGMENT} outside segment [0, {segment_length})"
        assert (
            EVENT_H_START_IN_SEGMENT < EVENT_H_END_IN_SEGMENT <= segment_length
        ), f"Event H end {EVENT_H_END_IN_SEGMENT} invalid"

    def test_event_h_exact_values(self) -> None:
        """Event H pixel coordinates match ground truth TSV."""
        # From rect_coords_json: {'x': 385, 'y': 46, 'width': 139, 'height': 20}
        assert EVENT_H_IMAGE_X == 385
        assert EVENT_H_IMAGE_WIDTH == 139
        assert EVENT_H_START_IN_SEGMENT == 378  # 385 - 7 (x0 of segment 2)
        assert EVENT_H_END_IN_SEGMENT == 517  # 378 + 139


# endregion


# region Group Setup Tests


class TestThoresenGroupSetup:
    """Test that groups are correctly configured."""

    def test_dgt1_group_structure(self, dgt1_group: TimelineGroup) -> None:
        """DGT1 group has reference + audio timelines."""
        assert dgt1_group.n_timelines == 2
        assert "dgt_1" in dgt1_group
        assert "audio" in dgt1_group

    def test_dgt2_group_structure(self, dgt2_group: TimelineGroup) -> None:
        """DGT2 group has reference timeline."""
        assert dgt2_group.n_timelines == 1
        assert "dgt_2" in dgt2_group

    def test_pixel_to_second_conversion(self, dgt1_group: TimelineGroup) -> None:
        """DGT1 pixels convert correctly to seconds."""
        # The width is odd, so its midpoint falls between two pixels. A pixel
        # axis holds integers, so the query is the pixel it names -- 2418 --
        # and the answer is that pixel's position, not the half-pixel's.
        mid_pixels = DGT1_TOTAL_WIDTH / 2
        mid_seconds = dgt1_group.get_coordinate_at(
            IdCoordinate(mid_pixels, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        )
        assert mid_seconds == Coordinate(
            2418 / DGT1_TOTAL_WIDTH * AUDIO_DURATION_SECONDS, TimeUnit.seconds
        )
        assert mid_seconds == Coordinate(75.01551189245087, TimeUnit.seconds)

        # Endpoints
        assert dgt1_group.get_coordinate_at(
            IdCoordinate(0, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(Fraction(0), TimeUnit.seconds)
        assert dgt1_group.get_coordinate_at(
            IdCoordinate(DGT1_TOTAL_WIDTH, TimeUnit.pixels, "dgt_1"),
            timeline_id="audio",
            format="coordinate",
        ) == Coordinate(Fraction(AUDIO_DURATION_SECONDS), TimeUnit.seconds)


# endregion


# region Segment Claims Tests


class TestThoresenSegmentClaims:
    """Test segment correspondence claims."""

    def test_five_interval_claims(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """All 5 claims are interval (not instant) matches."""
        assert len(thoresen_segment_claims) == 5
        assert all(c.is_interval for c in thoresen_segment_claims)

    def test_claims_connect_correct_timelines(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """All claims connect DGT1 and DGT2."""
        for claim in thoresen_segment_claims:
            assert claim.connects_both("dgt_1", "dgt_2")

    def test_claims_are_contiguous(
        self, thoresen_segment_claims: list[MatchClaim]
    ) -> None:
        """Segments form contiguous coverage (no gaps)."""
        # Check DGT1 side
        prev_end = 0.0
        for claim in thoresen_segment_claims:
            interval = claim.get_interval_for("dgt_1")
            assert interval.start.value == prev_end
            prev_end = interval.end.value
        assert prev_end == DGT1_TOTAL_WIDTH

        # Check DGT2 side
        prev_end = 0.0
        for claim in thoresen_segment_claims:
            interval = claim.get_interval_for("dgt_2")
            assert interval.start.value == prev_end
            prev_end = interval.end.value
        assert prev_end == DGT2_TOTAL_WIDTH

    def test_claim_metadata(self, thoresen_segment_claims: list[MatchClaim]) -> None:
        """Claims have proper provenance metadata."""
        for claim in thoresen_segment_claims:
            assert claim.metadata is not None
            assert claim.metadata.agent.name == "thoresen_analysis"
            assert claim.metadata.agent.identifier == "segment_correspondence"


# endregion


# region Helper: Calculate Event H Global Coordinates


def get_event_h_global_coords() -> tuple[float, float]:
    """Calculate Event H absolute coordinates in DGT2.

    Returns:
        (start_pixel, end_pixel) in DGT2 global coordinates.
    """
    # Sum of segments before Event H's segment
    segment_offset = sum(DGT2_SEGMENT_LENGTHS[:EVENT_H_SEGMENT_INDEX])

    start = segment_offset + EVENT_H_START_IN_SEGMENT
    end = segment_offset + EVENT_H_END_IN_SEGMENT

    return (float(start), float(end))


def calculate_expected_h_prime() -> tuple[float, float]:
    """Calculate expected H' position in DGT1 using proportional transfer.

    Returns:
        (start_pixel, end_pixel) in DGT1 global coordinates.
    """
    dgt2_seg_length = DGT2_SEGMENT_LENGTHS[EVENT_H_SEGMENT_INDEX]
    dgt1_seg_length = DGT1_SEGMENT_LENGTHS[EVENT_H_SEGMENT_INDEX]

    # Proportional positions within segment
    start_ratio = EVENT_H_START_IN_SEGMENT / dgt2_seg_length
    end_ratio = EVENT_H_END_IN_SEGMENT / dgt2_seg_length

    # DGT1 segment offset
    dgt1_seg_offset = sum(DGT1_SEGMENT_LENGTHS[:EVENT_H_SEGMENT_INDEX])

    # H' positions
    h_prime_start = dgt1_seg_offset + start_ratio * dgt1_seg_length
    h_prime_end = dgt1_seg_offset + end_ratio * dgt1_seg_length

    return (h_prime_start, h_prime_end)


class TestEventHCalculations:
    """Test helper calculations for Event H."""

    def test_event_h_global_coords(self) -> None:
        """Verify Event H global coordinate calculation."""
        start, end = get_event_h_global_coords()

        # Segment 1 ends at 866, so segment 2 starts there
        expected_start = (
            sum(DGT2_SEGMENT_LENGTHS[:EVENT_H_SEGMENT_INDEX]) + EVENT_H_START_IN_SEGMENT
        )
        expected_end = (
            sum(DGT2_SEGMENT_LENGTHS[:EVENT_H_SEGMENT_INDEX]) + EVENT_H_END_IN_SEGMENT
        )

        assert start == expected_start
        assert end == expected_end

    def test_h_prime_calculation(self) -> None:
        """Verify H' calculation matches expected proportional transfer."""
        h_prime_start, h_prime_end = calculate_expected_h_prime()

        # Verify H' is in segment 2 of DGT1
        seg2_start = sum(DGT1_SEGMENT_LENGTHS[:EVENT_H_SEGMENT_INDEX])
        seg2_end = seg2_start + DGT1_SEGMENT_LENGTHS[EVENT_H_SEGMENT_INDEX]

        assert seg2_start < h_prime_start < seg2_end
        assert seg2_start < h_prime_end < seg2_end
        assert h_prime_start < h_prime_end


# endregion


# region Graphical Loader Integration Tests


class TestThoresenGraphicalBundles:
    """Test Thoresen data loaded via GraphicalLoader.

    These tests validate that the graphical loader produces bundles
    that match the expected dimensions from Applications.ipynb.
    """

    def test_dgt1_bundle_matches_expected_dimensions(self, dgt1_bundle) -> None:
        """DGT1 bundle dimensions match test_thoresen_poc.py constants."""
        assert dgt1_bundle.total_length == DGT1_TOTAL_WIDTH
        assert dgt1_bundle.n_segments == 5

        # Each segment should have DGT1_SEGMENT_LENGTH
        for seg in dgt1_bundle.segments:
            assert seg.length == DGT1_SEGMENT_LENGTH

    def test_dgt2_bundle_matches_expected_dimensions(self, dgt2_bundle) -> None:
        """DGT2 bundle dimensions match test_thoresen_poc.py constants."""
        assert dgt2_bundle.total_length == DGT2_TOTAL_WIDTH
        assert dgt2_bundle.n_segments == 5

        # Segment lengths should match DGT2_SEGMENT_LENGTHS
        for seg, expected_len in zip(dgt2_bundle.segments, DGT2_SEGMENT_LENGTHS):
            assert seg.length == expected_len

    def test_event_h_location_via_bundle(self, dgt2_bundle) -> None:
        """Event H coordinates can be looked up via bundle."""
        # Get Event H start location
        h_start_global, h_end_global = get_event_h_global_coords()

        # Verify it's in the correct segment
        seg_idx = dgt2_bundle.get_segment_index_for_coord(h_start_global)
        assert seg_idx == EVENT_H_SEGMENT_INDEX

        # Convert to image coordinates
        src_idx, (x, y) = dgt2_bundle.timeline_to_image(h_start_global)
        assert src_idx == EVENT_H_SEGMENT_INDEX

        # x should be x0 + local offset
        expected_x = (
            DGT2_SEGMENT_BOUNDS[EVENT_H_SEGMENT_INDEX][0] + EVENT_H_START_IN_SEGMENT
        )
        assert x == expected_x

    def test_bundles_create_correct_timelines(self, dgt1_bundle, dgt2_bundle) -> None:
        """Bundles create timelines matching the fixtures."""
        dgt1_tl = dgt1_bundle.create_timeline(uid="dgt_1", name="Thoresen 2009")
        dgt2_tl = dgt2_bundle.create_timeline(uid="dgt_2", name="Thoresen 2010")

        # timeline.length is a Coordinate object, compare the value
        assert dgt1_tl.length.value == DGT1_TOTAL_WIDTH
        assert dgt2_tl.length.value == DGT2_TOTAL_WIDTH

        # These should match the fixtures created at the top of this file
        assert dgt1_tl.id == "dgt_1"
        assert dgt2_tl.id == "dgt_2"


# endregion
