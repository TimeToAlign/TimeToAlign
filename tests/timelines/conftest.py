"""Test fixtures for timeline tests.

This module provides pytest fixtures for:
- Timeline instances of all 6 types
- Sample events for testing
- Loader integration fixtures (MIDI, Score)
- Profiling utilities
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    Timeline,
)

# region Path Constants

MIDI_DATA_DIR, VIENNA_DIR, SCORE_DIR = ensure_data("midi", "vienna_1x22", "score")
MIDI_SCORE_DIR = MIDI_DATA_DIR / "score"

# endregion


# region Sample Event Data


@pytest.fixture
def instant_event_rows() -> list[dict[str, Any]]:
    """Sample instant events for testing."""
    return [
        {
            "id": "beat_1",
            "temporal_type": "instant",
            "event_type": "Beat",
            "instant": 0.0,
        },
        {
            "id": "beat_2",
            "temporal_type": "instant",
            "event_type": "Beat",
            "instant": 1.0,
        },
        {
            "id": "beat_3",
            "temporal_type": "instant",
            "event_type": "Beat",
            "instant": 2.0,
        },
        {
            "id": "beat_4",
            "temporal_type": "instant",
            "event_type": "Beat",
            "instant": 3.0,
        },
    ]


@pytest.fixture
def interval_event_rows() -> list[dict[str, Any]]:
    """Sample interval events for testing."""
    return [
        {
            "id": "note_1",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 0.0,
            "end": 0.5,
            "duration": 0.5,
        },
        {
            "id": "note_2",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 1.0,
            "end": 1.25,
            "duration": 0.25,
        },
        {
            "id": "note_3",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 2.0,
            "end": 3.0,
            "duration": 1.0,
        },
    ]


@pytest.fixture
def mixed_event_rows(
    instant_event_rows: list[dict[str, Any]],
    interval_event_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Combined instant and interval events."""
    return instant_event_rows + interval_event_rows


@pytest.fixture
def tick_event_rows() -> list[dict[str, Any]]:
    """Sample events in MIDI ticks (discrete)."""
    return [
        {
            "id": "note_1",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 0,
            "end": 480,
            "duration": 480,
        },
        {
            "id": "note_2",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 480,
            "end": 720,
            "duration": 240,
        },
        {
            "id": "note_3",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": 960,
            "end": 1920,
            "duration": 960,
        },
    ]


@pytest.fixture
def fraction_event_rows() -> list[dict[str, Any]]:
    """Sample events with fractional coordinates (quarters)."""
    return [
        {
            "id": "note_1",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": Fraction(0, 1),
            "end": Fraction(1, 2),
            "duration": Fraction(1, 2),
        },
        {
            "id": "note_2",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": Fraction(1, 1),
            "end": Fraction(5, 4),
            "duration": Fraction(1, 4),
        },
        {
            "id": "note_3",
            "temporal_type": "interval",
            "event_type": "Note",
            "start": Fraction(2, 1),
            "end": Fraction(3, 1),
            "duration": Fraction(1, 1),
        },
    ]


# endregion


# region Timeline Fixtures


@pytest.fixture
def empty_timeline() -> Timeline:
    """An empty base Timeline with default settings."""
    return Timeline.empty()


@pytest.fixture
def physical_timeline() -> ContinuousPhysicalTimeline:
    """A ContinuousPhysicalTimeline in seconds."""
    return ContinuousPhysicalTimeline(length=10.0, unit=TimeUnit.seconds)


@pytest.fixture
def logical_timeline() -> ContinuousLogicalTimeline:
    """A ContinuousLogicalTimeline in quarters."""
    return ContinuousLogicalTimeline(length=Fraction(8, 1), unit=TimeUnit.quarters)


@pytest.fixture
def discrete_logical_timeline() -> DiscreteLogicalTimeline:
    """A DiscreteLogicalTimeline in ticks."""
    return DiscreteLogicalTimeline(length=1920, unit=TimeUnit.ticks)


@pytest.fixture
def graphical_timeline() -> ContinuousGraphicalTimeline:
    """A ContinuousGraphicalTimeline in centimeters."""
    return ContinuousGraphicalTimeline(length=100.0, unit=TimeUnit.centimeters)


@pytest.fixture
def discrete_physical_timeline() -> DiscretePhysicalTimeline:
    """A DiscretePhysicalTimeline in samples (44100 Hz = 1 second)."""
    return DiscretePhysicalTimeline(length=44100, unit=TimeUnit.samples)


@pytest.fixture
def discrete_graphical_timeline() -> DiscreteGraphicalTimeline:
    """A DiscreteGraphicalTimeline in pixels."""
    return DiscreteGraphicalTimeline(length=1920, unit=TimeUnit.pixels)


@pytest.fixture
def timeline_with_events(
    physical_timeline: ContinuousPhysicalTimeline,
    mixed_event_rows: list[dict[str, Any]],
) -> ContinuousPhysicalTimeline:
    """A physical timeline populated with events."""
    physical_timeline.add_events(mixed_event_rows)
    return physical_timeline


@pytest.fixture
def nested_timeline_structure() -> ContinuousPhysicalTimeline:
    """A timeline with nested children for testing hierarchy traversal.

    Structure:
        parent (0-20s)
        ├── child_a (offset 0, length 5s)
        │   └── grandchild_a1 (offset 1, length 2s)
        └── child_b (offset 10, length 8s)
            ├── grandchild_b1 (offset 1, length 3s)
            └── grandchild_b2 (offset 5, length 2s)
    """
    # Create parent
    parent = ContinuousPhysicalTimeline(length=20.0, uid="parent")

    # Create child_a and its grandchild
    grandchild_a1 = ContinuousPhysicalTimeline(length=2.0, uid="grandchild_a1")
    child_a = ContinuousPhysicalTimeline(length=5.0, uid="child_a")
    child_a.add_child(grandchild_a1, offset=1.0)

    # Create child_b and its grandchildren
    grandchild_b1 = ContinuousPhysicalTimeline(length=3.0, uid="grandchild_b1")
    grandchild_b2 = ContinuousPhysicalTimeline(length=2.0, uid="grandchild_b2")
    child_b = ContinuousPhysicalTimeline(length=8.0, uid="child_b")
    child_b.add_child(grandchild_b1, offset=1.0)
    child_b.add_child(grandchild_b2, offset=5.0)

    # Add children to parent
    parent.add_child(child_a, offset=0.0)
    parent.add_child(child_b, offset=10.0)

    return parent


# endregion


# region All Timeline Types Fixture


@pytest.fixture(
    params=[
        (
            ContinuousLogicalTimeline,
            TimeUnit.quarters,
            NumberType.fraction,
            Fraction(4, 1),
        ),
        (DiscreteLogicalTimeline, TimeUnit.ticks, NumberType.int, 1920),
        (ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float, 10.0),
        (DiscretePhysicalTimeline, TimeUnit.samples, NumberType.int, 44100),
        (ContinuousGraphicalTimeline, TimeUnit.centimeters, NumberType.float, 100.0),
        (DiscreteGraphicalTimeline, TimeUnit.pixels, NumberType.int, 1920),
    ],
    ids=[
        "ContinuousLogical",
        "DiscreteLogical",
        "ContinuousPhysical",
        "DiscretePhysical",
        "ContinuousGraphical",
        "DiscreteGraphical",
    ],
)
def timeline_type_fixture(request):
    """Parametrized fixture providing all 6 timeline types.

    Returns:
        Tuple of (TimelineClass, default_unit, default_number_type, sample_length)
    """
    return request.param


# endregion


# region MIDI Loader Fixtures


@pytest.fixture
def midi_performance_path() -> Path:
    """Path to a sample MIDI performance file."""
    path = VIENNA_DIR / "Chopin_op10_no3_p01.mid"
    return path


@pytest.fixture
def midi_score_path() -> Path:
    """Path to a sample MIDI score file."""
    path = MIDI_SCORE_DIR / "beethoven_mtd.mid"
    return path


@pytest.fixture
def musicxml_score_path() -> Path:
    """Path to a sample MusicXML score file."""
    path = VIENNA_DIR / "Chopin_op10_no3.musicxml"
    return path


# endregion
