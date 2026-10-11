"""Tests for the graphical timeline loader.

This module tests the graphical loading infrastructure:
- TimeAxisPath variants (horizontal, vertical, diagonal, parametric)
- ImageSource loading and operations
- GraphicalSegment coordinate conversion
- GraphicalBundle operations
- GraphicalLoader factory

Tests are organized from low-level (paths) to high-level (loader).
"""

from __future__ import annotations

import math

import pytest

# Import path classes directly (no pymupdf dependency)
from timetoalign.loader.graphical.paths import (
    DiagonalLinePath,
    HorizontalLinePath,
    ParametricPath,
    VerticalLinePath,
)

# Import conftest data
from .conftest import (
    DGT1_IMAGE,
    DGT1_SEGMENT_LENGTH,
    DGT1_TOTAL_WIDTH,
    DGT1_X0,
    DGT1_X1,
    DGT1_Y_POSITIONS,
    DGT2_IMAGES,
    DGT2_SEGMENT_BOUNDS,
    DGT2_TOTAL_WIDTH,
    EVENT_H_END_GLOBAL,
    EVENT_H_SEGMENT_INDEX,
    EVENT_H_START_GLOBAL,
    EVENT_H_START_LOCAL,
    THORESEN_TEST_EVENTS,
    event_to_coords,
    validate_dgt1_bundle,
    validate_dgt2_bundle,
)

# region TimeAxisPath Tests


class TestHorizontalLinePath:
    """Tests for HorizontalLinePath coordinate conversion."""

    def test_basic_properties(self) -> None:
        """Path stores x0, x1, y and computes length."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        assert path.x0 == 10
        assert path.x1 == 500
        assert path.y == 100
        assert path.length == 490

    def test_to_2d_start(self) -> None:
        """Coordinate 0 maps to (x0, y)."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        x, y = path.to_2d(0)
        assert x == 10
        assert y == 100

    def test_to_2d_end(self) -> None:
        """Coordinate length maps to (x1, y)."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        x, y = path.to_2d(490)
        assert x == 500
        assert y == 100

    def test_to_2d_middle(self) -> None:
        """Midpoint coordinate maps correctly."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        x, y = path.to_2d(245)
        assert x == 255
        assert y == 100

    def test_from_2d_valid(self) -> None:
        """Valid point on path converts back."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        coord = path.from_2d(255, 100)
        assert coord == 245

    def test_from_2d_off_path_y(self) -> None:
        """Point too far from y returns None."""
        path = HorizontalLinePath(x0=10, x1=500, y=100, tolerance=5)
        assert path.from_2d(255, 110) is None  # 10 pixels away > tolerance

    def test_from_2d_off_path_x(self) -> None:
        """Point outside x range returns None."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        assert path.from_2d(5, 100) is None  # Before x0
        assert path.from_2d(505, 100) is None  # After x1

    def test_distance_in_range(self) -> None:
        """Distance is vertical offset when x is in range."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        assert path.distance_to_path(255, 115) == 15

    def test_distance_outside_range(self) -> None:
        """Distance uses endpoint when x is outside range."""
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        # Left of path: distance to (10, 100)
        dist = path.distance_to_path(0, 100)
        assert dist == 10

    def test_invalid_x_range_raises(self) -> None:
        """x1 <= x0 raises ValueError."""
        with pytest.raises(ValueError, match="must be greater than"):
            HorizontalLinePath(x0=500, x1=10, y=100)

    def test_dgt1_segment_dimensions(self) -> None:
        """DGT1 segment path has correct dimensions."""
        path = HorizontalLinePath(x0=DGT1_X0, x1=DGT1_X1, y=DGT1_Y_POSITIONS[0])
        assert path.length == DGT1_SEGMENT_LENGTH
        assert path.length == 967


class TestVerticalLinePath:
    """Tests for VerticalLinePath coordinate conversion."""

    def test_basic_properties(self) -> None:
        """Path stores x, y0, y1 and computes length."""
        path = VerticalLinePath(x=50, y0=10, y1=500)
        assert path.x == 50
        assert path.y0 == 10
        assert path.y1 == 500
        assert path.length == 490

    def test_to_2d(self) -> None:
        """Coordinates map to (x, y0+coord)."""
        path = VerticalLinePath(x=50, y0=10, y1=500)
        assert path.to_2d(0) == (50, 10)
        assert path.to_2d(245) == (50, 255)
        assert path.to_2d(490) == (50, 500)

    def test_from_2d_valid(self) -> None:
        """Valid point on path converts back."""
        path = VerticalLinePath(x=50, y0=10, y1=500)
        assert path.from_2d(50, 255) == 245

    def test_invalid_y_range_raises(self) -> None:
        """y1 <= y0 raises ValueError."""
        with pytest.raises(ValueError):
            VerticalLinePath(x=50, y0=500, y1=10)


class TestDiagonalLinePath:
    """Tests for DiagonalLinePath coordinate conversion."""

    def test_3_4_5_triangle(self) -> None:
        """Classic 3-4-5 right triangle: length = 5."""
        path = DiagonalLinePath(start=(0, 0), end=(3, 4))
        assert path.length == 5.0

    def test_to_2d_midpoint(self) -> None:
        """Midpoint of diagonal is at half coordinates."""
        path = DiagonalLinePath(start=(0, 0), end=(300, 400))
        x, y = path.to_2d(250)  # Half of 500
        assert x == 150
        assert y == 200

    def test_from_2d_on_path(self) -> None:
        """Point on diagonal converts correctly."""
        path = DiagonalLinePath(start=(0, 0), end=(300, 400))
        coord = path.from_2d(150, 200)
        assert coord == 250

    def test_from_2d_off_path(self) -> None:
        """Point far from diagonal returns None."""
        path = DiagonalLinePath(start=(0, 0), end=(300, 400), tolerance=5)
        assert path.from_2d(100, 100) is None  # Not on 3:4 ratio line

    def test_same_point_raises(self) -> None:
        """Start == end raises ValueError."""
        with pytest.raises(ValueError):
            DiagonalLinePath(start=(100, 100), end=(100, 100))


class TestParametricPath:
    """Tests for ParametricPath with arbitrary functions."""

    def test_circle_quarter(self) -> None:
        """Quarter circle has length pi/2 * radius."""
        radius = 100
        # x = r*cos(t), y = r*sin(t), t in [0, pi/2]
        path = ParametricPath(
            x_func=lambda t: radius * math.cos(t),
            y_func=lambda t: radius * math.sin(t),
            t_start=0,
            t_end=math.pi / 2,
            samples=1000,
        )
        expected_length = radius * math.pi / 2  # ~157.08
        assert path.length == pytest.approx(expected_length, rel=0.01)

    def test_straight_line_as_parametric(self) -> None:
        """Straight line via parametric matches diagonal."""
        # Line from (0,0) to (100,0)
        path = ParametricPath(
            x_func=lambda t: t * 100,
            y_func=lambda t: 0,
            t_start=0,
            t_end=1,
            samples=100,
        )
        # Every sampled segment lies on y=0, so the accumulated arc length is
        # exactly 100.0 with no interpolation error.
        assert path.length == 100.0

    def test_to_2d_endpoints(self) -> None:
        """Parametric path endpoints map correctly."""
        path = ParametricPath(
            x_func=lambda t: t * 100,
            y_func=lambda t: t * 50,
            t_start=0,
            t_end=1,
        )
        # Start. _arc_to_t clamps arc <= 0 to t_start (paths.py:450-455) and
        # returns the exact endpoint, so to_2d(0) is bit-exact (0, 0).
        x0, y0 = path.to_2d(0)
        assert x0 == 0
        assert y0 == 0

        # End. _arc_to_t clamps arc >= total_length to t_end and returns the
        # exact endpoint, so to_2d(length) is bit-exact (100, 50).
        x1, y1 = path.to_2d(path.length)
        assert x1 == 100
        assert y1 == 50


# endregion


# region GraphicalSegment Tests


class TestGraphicalSegment:
    """Tests for GraphicalSegment coordinate handling."""

    def test_segment_properties(self) -> None:
        """Segment exposes path length and timeline bounds."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=0, path=path, timeline_offset=100)

        assert seg.length == 490
        assert seg.timeline_start == 100
        assert seg.timeline_end == 590

    def test_contains_coord(self) -> None:
        """Half-open interval [start, end) semantics."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=0, path=path, timeline_offset=100)

        assert seg.contains_coord(100)  # Start included
        assert seg.contains_coord(300)  # Middle
        assert not seg.contains_coord(590)  # End excluded
        assert not seg.contains_coord(99)  # Before

    def test_to_image_valid(self) -> None:
        """Coordinate within segment converts to (source_index, (x, y))."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=2, path=path, timeline_offset=100)

        src_idx, (x, y) = seg.to_image(200)  # 100 into segment
        assert src_idx == 2
        assert x == 110  # 10 + 100
        assert y == 100

    def test_to_image_out_of_range_raises(self) -> None:
        """Coordinate outside segment raises ValueError."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=0, path=path, timeline_offset=100)

        with pytest.raises(ValueError, match="not in segment"):
            seg.to_image(50)  # Before segment

    def test_from_image_valid(self) -> None:
        """Image coordinates on path convert to timeline coordinate."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=0, path=path, timeline_offset=100)

        coord = seg.from_image(110, 100)  # 100 pixels into path
        assert coord == 200  # 100 offset + 100 local

    def test_from_image_wrong_source(self) -> None:
        """Mismatched source_index returns None."""
        from timetoalign.loader.graphical.segment import GraphicalSegment

        path = HorizontalLinePath(x0=10, x1=500, y=100)
        seg = GraphicalSegment(source_index=0, path=path, timeline_offset=100)

        assert seg.from_image(110, 100, source_index=1) is None


# endregion


# region GraphicalBundle Tests (with pymupdf)


@pytest.mark.skipif(
    not DGT1_IMAGE.exists(),
    reason="Test data not found",
)
class TestGraphicalBundleDGT1:
    """Tests for DGT1 bundle creation and operations."""

    def test_bundle_structure(self, dgt1_bundle) -> None:
        """DGT1 bundle has correct structure."""
        validate_dgt1_bundle(dgt1_bundle)

    def test_segment_names(self, dgt1_bundle) -> None:
        """Segments have expected names."""
        names = [s.name for s in dgt1_bundle.segments]
        expected = ["system_1", "system_2", "system_3", "system_4", "system_5"]
        assert names == expected

    def test_timeline_to_image_first_segment(self, dgt1_bundle) -> None:
        """Coordinate in first segment maps to correct image position."""
        src_idx, (x, y) = dgt1_bundle.timeline_to_image(100)
        assert src_idx == 0
        assert x == DGT1_X0 + 100  # 2 + 100 = 102
        assert y == DGT1_Y_POSITIONS[0]  # 18

    def test_timeline_to_image_third_segment(self, dgt1_bundle) -> None:
        """Coordinate in third segment maps correctly."""
        # Third segment starts at 2 * 967 = 1934
        coord = 1934 + 200  # 200 into third segment
        src_idx, (x, y) = dgt1_bundle.timeline_to_image(coord)
        assert src_idx == 0  # Same source for all segments
        assert x == DGT1_X0 + 200  # 2 + 200 = 202
        assert y == DGT1_Y_POSITIONS[2]  # 396

    def test_image_to_timeline_first_segment(self, dgt1_bundle) -> None:
        """Image coordinate converts back to timeline."""
        coord = dgt1_bundle.image_to_timeline(0, DGT1_X0 + 100, DGT1_Y_POSITIONS[0])
        assert coord == 100

    def test_total_length_exact(self, dgt1_bundle) -> None:
        """Total length matches expected DGT1 width."""
        assert dgt1_bundle.total_length == DGT1_TOTAL_WIDTH

    def test_create_timeline_creates_correct_timeline(self, dgt1_bundle) -> None:
        """Bundle creates DiscreteGraphicalTimeline with correct length."""
        timeline = dgt1_bundle.create_timeline(uid="dgt_1", name="Thoresen 2009")
        # timeline.length is a Coordinate object, compare the value
        assert timeline.length.value == DGT1_TOTAL_WIDTH
        assert timeline.id == "dgt_1"
        assert timeline.name == "Thoresen 2009"


@pytest.mark.skipif(
    not all(p.exists() for p in DGT2_IMAGES),
    reason="Test data not found",
)
class TestGraphicalBundleDGT2:
    """Tests for DGT2 bundle creation and operations."""

    def test_bundle_structure(self, dgt2_bundle) -> None:
        """DGT2 bundle has correct structure."""
        validate_dgt2_bundle(dgt2_bundle)

    def test_segments_use_different_sources(self, dgt2_bundle) -> None:
        """Each segment uses a different source image."""
        source_indices = [s.source_index for s in dgt2_bundle.segments]
        assert source_indices == [0, 1, 2, 3, 4]

    def test_timeline_to_image_event_h_start(self, dgt2_bundle) -> None:
        """Event H start coordinate maps to correct image position."""
        src_idx, (x, y) = dgt2_bundle.timeline_to_image(EVENT_H_START_GLOBAL)
        assert src_idx == EVENT_H_SEGMENT_INDEX  # Second image (index 1)
        # Local x should be EVENT_H_START_LOCAL + x0 of that segment
        x0, x1, expected_y = DGT2_SEGMENT_BOUNDS[EVENT_H_SEGMENT_INDEX]
        expected_x = x0 + EVENT_H_START_LOCAL
        assert x == expected_x
        assert y == expected_y

    def test_event_h_spans_within_segment(self, dgt2_bundle) -> None:
        """Event H start and end are in the same segment."""
        start_seg = dgt2_bundle.get_segment_index_for_coord(EVENT_H_START_GLOBAL)
        end_seg = dgt2_bundle.get_segment_index_for_coord(EVENT_H_END_GLOBAL - 0.1)
        assert start_seg == EVENT_H_SEGMENT_INDEX
        assert end_seg == EVENT_H_SEGMENT_INDEX

    def test_total_length_exact(self, dgt2_bundle) -> None:
        """Total length matches expected DGT2 width."""
        assert dgt2_bundle.total_length == DGT2_TOTAL_WIDTH


# endregion


# region GraphicalLoader Tests


class TestGraphicalLoader:
    """Tests for the GraphicalLoader factory."""

    def test_loader_initialization(self) -> None:
        """Loader starts empty."""
        from timetoalign.loader.graphical import GraphicalLoader

        loader = GraphicalLoader()
        assert loader.n_sources == 0
        assert loader.n_segments == 0
        assert loader.current_offset == 0.0

    def test_contiguous_offset_tracking(self) -> None:
        """Loader tracks offset for contiguous segments."""
        from timetoalign.loader.graphical import GraphicalLoader

        pytest.importorskip("pymupdf")

        if not DGT1_IMAGE.exists():
            pytest.skip("Test data not found")

        loader = GraphicalLoader()
        idx = loader.add_image(DGT1_IMAGE)

        # Add first segment
        loader.add_horizontal_segment(idx, x0=0, x1=100, y=50)
        assert loader.current_offset == 100

        # Add second segment (contiguous)
        loader.add_horizontal_segment(idx, x0=0, x1=150, y=100)
        assert loader.current_offset == 250

    def test_explicit_offset_override(self) -> None:
        """Explicit offset overrides contiguous tracking."""
        from timetoalign.loader.graphical import GraphicalLoader

        pytest.importorskip("pymupdf")

        if not DGT1_IMAGE.exists():
            pytest.skip("Test data not found")

        loader = GraphicalLoader()
        idx = loader.add_image(DGT1_IMAGE)

        loader.add_horizontal_segment(idx, x0=0, x1=100, y=50)
        assert loader.current_offset == 100

        # Explicit offset doesn't update current_offset
        loader.add_horizontal_segment(idx, x0=0, x1=100, y=100, offset=500)
        assert loader.current_offset == 100  # Unchanged

    def test_invalid_source_index_raises(self) -> None:
        """Adding segment with invalid source raises IndexError."""
        from timetoalign.loader.graphical import GraphicalLoader

        loader = GraphicalLoader()
        path = HorizontalLinePath(x0=0, x1=100, y=50)

        with pytest.raises(IndexError, match="out of range"):
            loader.add_segment(source_index=0, path=path)

    def test_clear_resets_state(self) -> None:
        """Clear removes all sources and segments."""
        from timetoalign.loader.graphical import GraphicalLoader

        pytest.importorskip("pymupdf")

        if not DGT1_IMAGE.exists():
            pytest.skip("Test data not found")

        loader = GraphicalLoader()
        idx = loader.add_image(DGT1_IMAGE)
        loader.add_horizontal_segment(idx, x0=0, x1=100, y=50)

        assert loader.n_sources == 1
        assert loader.n_segments == 1

        loader.clear()

        assert loader.n_sources == 0
        assert loader.n_segments == 0
        assert loader.current_offset == 0.0

    def test_bundle_is_independent(self) -> None:
        """Bundle is a copy, not a reference to loader state."""
        from timetoalign.loader.graphical import GraphicalLoader

        pytest.importorskip("pymupdf")

        if not DGT1_IMAGE.exists():
            pytest.skip("Test data not found")

        loader = GraphicalLoader()
        idx = loader.add_image(DGT1_IMAGE)
        loader.add_horizontal_segment(idx, x0=0, x1=100, y=50)

        bundle1 = loader.store
        loader.add_horizontal_segment(idx, x0=0, x1=100, y=100)
        bundle2 = loader.store

        assert bundle1.n_segments == 1
        assert bundle2.n_segments == 2


# endregion


# region All Thoresen Events Tests


@pytest.mark.skipif(
    not all(p.exists() for p in DGT2_IMAGES),
    reason="Test data not found",
)
class TestThoresenAllEvents:
    """Test all 11 events from thoresen_test.tsv.

    These tests validate coordinate conversion for every event,
    ensuring the graphical loader handles all test cases correctly.
    """

    @pytest.mark.parametrize("event_id", [evt[0] for evt in THORESEN_TEST_EVENTS])
    def test_event_to_image_coordinates(self, dgt2_bundle, event_id: str) -> None:
        """Event global coordinate converts to correct image position."""
        evt = event_to_coords(event_id)

        # Convert start position
        src_idx, (x, y) = dgt2_bundle.timeline_to_image(evt["start_global"])

        # Verify correct source image
        assert (
            src_idx == evt["segment_index"]
        ), f"{event_id}: Expected source {evt['segment_index']}, got {src_idx}"

        # Verify x coordinate (should match original x from TSV)
        assert x == evt["x"], f"{event_id}: Expected x={evt['x']}, got {x}"

        # Verify y coordinate
        x0, x1, expected_y = DGT2_SEGMENT_BOUNDS[evt["segment_index"]]
        assert y == expected_y, f"{event_id}: Expected y={expected_y}, got {y}"

    @pytest.mark.parametrize("event_id", [evt[0] for evt in THORESEN_TEST_EVENTS])
    def test_event_roundtrip_image_to_timeline(
        self, dgt2_bundle, event_id: str
    ) -> None:
        """Image coordinates round-trip correctly to timeline coordinates."""
        evt = event_to_coords(event_id)

        # Start: timeline -> image -> timeline
        src_idx, (x_start, y_start) = dgt2_bundle.timeline_to_image(evt["start_global"])
        coord_start_back = dgt2_bundle.image_to_timeline(src_idx, x_start, y_start)
        assert (
            coord_start_back == evt["start_global"]
        ), f"{event_id} start: {evt['start_global']} -> {coord_start_back}"

        # End: timeline -> image -> timeline
        src_idx, (x_end, y_end) = dgt2_bundle.timeline_to_image(evt["end_global"] - 0.5)
        coord_end_back = dgt2_bundle.image_to_timeline(src_idx, x_end, y_end)
        assert (
            coord_end_back == evt["end_global"] - 0.5
        ), f"{event_id} end: {evt['end_global']} -> {coord_end_back}"

    @pytest.mark.parametrize("event_id", [evt[0] for evt in THORESEN_TEST_EVENTS])
    def test_event_within_correct_segment(self, dgt2_bundle, event_id: str) -> None:
        """Event coordinates are within the correct segment."""
        evt = event_to_coords(event_id)

        # Check start is in correct segment
        seg_idx_start = dgt2_bundle.get_segment_index_for_coord(evt["start_global"])
        assert seg_idx_start == evt["segment_index"], (
            f"{event_id}: start at {evt['start_global']} "
            f"in segment {seg_idx_start}, expected {evt['segment_index']}"
        )

        # Check end is in correct segment (use end-1 to stay within interval)
        seg_idx_end = dgt2_bundle.get_segment_index_for_coord(evt["end_global"] - 1)
        assert seg_idx_end == evt["segment_index"], (
            f"{event_id}: end at {evt['end_global']} "
            f"in segment {seg_idx_end}, expected {evt['segment_index']}"
        )

    def test_all_events_non_overlapping_segments(self, dgt2_bundle) -> None:
        """All events are contained within their segments (no overlap)."""
        for evt_tuple in THORESEN_TEST_EVENTS:
            event_id = evt_tuple[0]
            evt = event_to_coords(event_id)

            seg = dgt2_bundle.segments[evt["segment_index"]]
            assert seg.timeline_start <= evt["start_global"] < seg.timeline_end, (
                f"{event_id} start {evt['start_global']} "
                f"not in segment [{seg.timeline_start}, {seg.timeline_end})"
            )
            assert seg.timeline_start < evt["end_global"] <= seg.timeline_end, (
                f"{event_id} end {evt['end_global']} "
                f"not in segment [{seg.timeline_start}, {seg.timeline_end})"
            )

    def test_events_ordered_by_time(self) -> None:
        """Events are ordered by start_time_sec in the TSV."""
        start_times = [evt[6] for evt in THORESEN_TEST_EVENTS]
        assert start_times == sorted(
            start_times
        ), "Events not sorted by start_time_sec in THORESEN_TEST_EVENTS"

    def test_event_count_exact(self) -> None:
        """Exactly 11 events from thoresen_test.tsv."""
        assert len(THORESEN_TEST_EVENTS) == 11


# endregion
