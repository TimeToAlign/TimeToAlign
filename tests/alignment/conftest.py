"""Shared fixtures for alignment tests.

This module provides pytest fixtures for alignment testing, including:
- ID generator resets (ensuring test isolation)
- Thoresen graphical timeline bundles (DGT1, DGT2)
- Thoresen test data constants

Test data is in tests/data/thoresen/ and comes from the published Thoresen
example.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.alignment import Agent, AlignmentAnchor, MatchClaim, MatchMetadata
from timetoalign.alignment.bundle import _reset_bundle_ids
from timetoalign.alignment.claims import _reset_anchor_ids, _reset_claim_ids
from timetoalign.core import AgentType, Coordinate, TimeUnit
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
)
from timetoalign.timelines.groups import _reset_group_ids

ensure_data("thoresen", "supra", "vienna_1x22", "score")

# region Test Data Paths

# Base path for test data (fetched on demand via pooch — see tests/_testdata.py)
TEST_DATA_DIR = Path(__file__).parents[1] / "data"
THORESEN_DATA_DIR = TEST_DATA_DIR / "thoresen"

# DGT1 (2009) - Single image with 5 horizontal systems
DGT1_IMAGE = THORESEN_DATA_DIR / "thoresen_2009_sound-objects_p312_page1_1.jpeg"

# DGT2 (2010) - 5 separate images
DGT2_IMAGES = [
    THORESEN_DATA_DIR / "thoresen_2010_form-building-patterns_p90-91_page1_1.jpeg",
    THORESEN_DATA_DIR / "thoresen_2010_form-building-patterns_p90-91_page1_2.jpeg",
    THORESEN_DATA_DIR / "thoresen_2010_form-building-patterns_p90-91_page1_3.jpeg",
    THORESEN_DATA_DIR / "thoresen_2010_form-building-patterns_p90-91_page1_4.jpeg",
    THORESEN_DATA_DIR / "thoresen_2010_form-building-patterns_p90-91_page2_1.jpeg",
]

# endregion


# region Thoresen Coordinate Data (from Applications.ipynb)

# DGT1 (2009): Single image, 5 horizontal systems
# x-boundaries: x0=2, x1=969 for all systems = 967 pixels per segment
# y-positions of each system
DGT1_X0 = 2
DGT1_X1 = 969
DGT1_SEGMENT_LENGTH = DGT1_X1 - DGT1_X0  # 967 pixels
DGT1_Y_POSITIONS = [18, 205, 396, 588, 785]
DGT1_TOTAL_WIDTH = DGT1_SEGMENT_LENGTH * 5  # 4835 pixels

# DGT2 (2010): 5 separate images with varying dimensions
# (x0, x1, y) for each segment
DGT2_SEGMENT_BOUNDS = [
    (8, 874, 15),  # page1_1: 866 px
    (7, 874, 18),  # page1_2: 867 px
    (7, 874, 19),  # page1_3: 867 px
    (8, 872, 15),  # page1_4: 864 px
    (9, 873, 20),  # page2_1: 864 px
]
DGT2_SEGMENT_LENGTHS = [x1 - x0 for x0, x1, _ in DGT2_SEGMENT_BOUNDS]
DGT2_TOTAL_WIDTH = sum(DGT2_SEGMENT_LENGTHS)  # 4328 pixels

# All 11 events from thoresen_test.tsv (rect_coords_json)
# Format: (event_id, segment_index, x, y, width, height, start_time_sec, duration_sec)
THORESEN_TEST_EVENTS = [
    ("rect_a", 0, 10, 90, 148, 55, 0.0, 5.0),
    ("rect_b", 0, 40, 37, 127, 21, 1.5, 4.0),
    ("rect_c", 0, 111, 60, 57, 23, 3.5, 2.0),
    ("rect_a2", 1, 145, 90, 160, 58, 34.6, 5.2),
    ("rect_h2", 1, 385, 46, 139, 20, 43.5, 4.5),
    ("rect_d3", 2, 310, 93, 154, 18, 71.0, 4.75),
    ("rect_b3", 2, 456, 69, 229, 18, 76.0, 7.5),
    ("rect_i4", 3, 14, 115, 127, 31, 90.5, 4.0),
    ("rect_a4", 3, 663, 82, 97, 23, 113.4, 3.0),
    ("rect_i5", 4, 19, 119, 251, 29, 121.0, 7.5),
    ("rect_f5", 4, 595, 45, 64, 21, 141.0, 1.5),
]


# Helper function to convert event to local/global coordinates
def event_to_coords(event_id: str) -> dict:
    """Get local and global coordinates for an event.

    Returns:
        dict with keys: event_id, segment_index, x, y, width, height,
                       start_local, end_local, start_global, end_global,
                       start_time_sec, duration_sec
    """
    for evt in THORESEN_TEST_EVENTS:
        if evt[0] == event_id:
            event_id, seg_idx, x, y, width, height, start_time, duration = evt
            x0, x1, _ = DGT2_SEGMENT_BOUNDS[seg_idx]

            # Local coordinates (within segment)
            start_local = x - x0
            end_local = start_local + width

            # Global coordinates (across all segments)
            segment_offset = sum(DGT2_SEGMENT_LENGTHS[:seg_idx])
            start_global = segment_offset + start_local
            end_global = segment_offset + end_local

            return {
                "event_id": event_id,
                "segment_index": seg_idx,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "start_local": start_local,
                "end_local": end_local,
                "start_global": start_global,
                "end_global": end_global,
                "start_time_sec": start_time,
                "duration_sec": duration,
            }
    raise ValueError(f"Event {event_id} not found")


# Event H (rect_h2) constants used by test_graphical_loader
EVENT_H_SEGMENT_INDEX = 1
EVENT_H_START_LOCAL = 378  # 385 - 7
EVENT_H_END_LOCAL = 517  # 378 + 139
EVENT_H_START_GLOBAL = DGT2_SEGMENT_LENGTHS[0] + EVENT_H_START_LOCAL
EVENT_H_END_GLOBAL = DGT2_SEGMENT_LENGTHS[0] + EVENT_H_END_LOCAL

# Audio timing for validation
AUDIO_DURATION_SECONDS = 150.0

# endregion


# region ID Reset Fixtures


@pytest.fixture(autouse=True)
def reset_ids() -> None:
    """Reset ID generators before each test.

    This ensures test isolation - each test starts with fresh IDs.
    Resets all four generators: group, anchor, claim, and bundle.
    """
    _reset_group_ids()
    _reset_anchor_ids()
    _reset_claim_ids()
    _reset_bundle_ids()


# endregion


# region Thoresen Timeline Fixtures


@pytest.fixture
def dgt1_timeline() -> DiscreteGraphicalTimeline:
    """DGT1 (2009) timeline: 4835 pixels total (5 x 967)."""
    return DiscreteGraphicalTimeline(
        length=DGT1_TOTAL_WIDTH,
        unit="pixels",
        uid="dgt_1",
        name="Thoresen 2009",
    )


@pytest.fixture
def dgt2_timeline() -> DiscreteGraphicalTimeline:
    """DGT2 (2010) timeline: 4328 pixels total."""
    return DiscreteGraphicalTimeline(
        length=DGT2_TOTAL_WIDTH,
        unit="pixels",
        uid="dgt_2",
        name="Thoresen 2010",
    )


@pytest.fixture
def audio_timeline() -> ContinuousPhysicalTimeline:
    """Audio timeline: 150 seconds (shared reference for Thoresen PoC)."""
    return ContinuousPhysicalTimeline(
        length=AUDIO_DURATION_SECONDS,
        unit="seconds",
        uid="audio",
        name="Audio",
    )


@pytest.fixture
def thoresen_segment_claims() -> list[MatchClaim]:
    """5 segment correspondence claims between DGT1 and DGT2.

    Each claim maps a DGT1 segment (967 px each) to the corresponding
    DGT2 segment ([866, 867, 867, 864, 864] px).
    """
    claims = []
    offset_dgt1 = 0
    offset_dgt2 = 0

    for i in range(5):
        claim = MatchClaim(
            timeline_a_id="dgt_1",
            timeline_b_id="dgt_2",
            start_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(offset_dgt1, TimeUnit.pixels),
                timeline_b_id="dgt_2",
                coordinate_b=Coordinate(offset_dgt2, TimeUnit.pixels),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="dgt_1",
                coordinate_a=Coordinate(
                    offset_dgt1 + DGT1_SEGMENT_LENGTH, TimeUnit.pixels
                ),
                timeline_b_id="dgt_2",
                coordinate_b=Coordinate(
                    offset_dgt2 + DGT2_SEGMENT_LENGTHS[i], TimeUnit.pixels
                ),
            ),
            metadata=MatchMetadata(
                agent=Agent(
                    name="thoresen_analysis",
                    type=AgentType.human,
                    identifier="segment_correspondence",
                ),
            ),
        )
        claims.append(claim)
        offset_dgt1 += DGT1_SEGMENT_LENGTH
        offset_dgt2 += DGT2_SEGMENT_LENGTHS[i]

    return claims


# endregion


# region Graphical Bundle Fixtures


@pytest.fixture
def dgt1_bundle():
    """Create DGT1 (2009) graphical bundle.

    DGT1 is a single image with 5 horizontal systems.
    All systems have the same x-range (2, 969) = 967 pixels.
    Y-positions: [18, 205, 396, 588, 785]

    Returns:
        GraphicalBundle with 1 source and 5 contiguous segments.
    """
    # Skip if pymupdf not installed
    pytest.importorskip("pymupdf")

    from timetoalign.loader.graphical import GraphicalLoader

    if not DGT1_IMAGE.exists():
        pytest.skip(f"Test data not found: {DGT1_IMAGE}")

    loader = GraphicalLoader(metadata={"source": "Thoresen 2009"})
    idx = loader.add_image(DGT1_IMAGE)

    for i, y in enumerate(DGT1_Y_POSITIONS):
        loader.add_horizontal_segment(
            source_index=idx,
            x0=DGT1_X0,
            x1=DGT1_X1,
            y=y,
            name=f"system_{i+1}",
        )

    return loader.store


@pytest.fixture
def dgt2_bundle():
    """Create DGT2 (2010) graphical bundle.

    DGT2 is 5 separate images, each containing one horizontal segment.
    Segment dimensions vary slightly between images.

    Returns:
        GraphicalBundle with 5 sources and 5 contiguous segments.
    """
    # Skip if pymupdf not installed
    pytest.importorskip("pymupdf")

    from timetoalign.loader.graphical import GraphicalLoader

    # Check all images exist
    for img_path in DGT2_IMAGES:
        if not img_path.exists():
            pytest.skip(f"Test data not found: {img_path}")

    loader = GraphicalLoader(metadata={"source": "Thoresen 2010"})

    for i, (img_path, (x0, x1, y)) in enumerate(zip(DGT2_IMAGES, DGT2_SEGMENT_BOUNDS)):
        idx = loader.add_image(img_path)
        loader.add_horizontal_segment(
            source_index=idx,
            x0=x0,
            x1=x1,
            y=y,
            name=f"page_{i+1}",
        )

    return loader.store


# endregion


# region Validation Helpers


def validate_dgt1_bundle(bundle) -> None:
    """Validate DGT1 bundle structure and dimensions.

    Raises AssertionError if validation fails.
    """
    assert bundle.n_sources == 1, f"Expected 1 source, got {bundle.n_sources}"
    assert bundle.n_segments == 5, f"Expected 5 segments, got {bundle.n_segments}"
    assert (
        bundle.total_length == DGT1_TOTAL_WIDTH
    ), f"Expected length {DGT1_TOTAL_WIDTH}, got {bundle.total_length}"

    # Verify segments are contiguous
    expected_offset = 0.0
    for i, seg in enumerate(bundle.segments):
        assert (
            seg.timeline_offset == expected_offset
        ), f"Segment {i} offset: expected {expected_offset}, got {seg.timeline_offset}"
        assert (
            seg.length == DGT1_SEGMENT_LENGTH
        ), f"Segment {i} length: expected {DGT1_SEGMENT_LENGTH}, got {seg.length}"
        expected_offset += DGT1_SEGMENT_LENGTH


def validate_dgt2_bundle(bundle) -> None:
    """Validate DGT2 bundle structure and dimensions.

    Raises AssertionError if validation fails.
    """
    assert bundle.n_sources == 5, f"Expected 5 sources, got {bundle.n_sources}"
    assert bundle.n_segments == 5, f"Expected 5 segments, got {bundle.n_segments}"
    assert (
        bundle.total_length == DGT2_TOTAL_WIDTH
    ), f"Expected length {DGT2_TOTAL_WIDTH}, got {bundle.total_length}"

    # Verify segments are contiguous with correct lengths
    expected_offset = 0.0
    for i, (seg, expected_len) in enumerate(zip(bundle.segments, DGT2_SEGMENT_LENGTHS)):
        assert (
            seg.timeline_offset == expected_offset
        ), f"Segment {i} offset: expected {expected_offset}, got {seg.timeline_offset}"
        assert (
            seg.length == expected_len
        ), f"Segment {i} length: expected {expected_len}, got {seg.length}"
        expected_offset += expected_len


# endregion
