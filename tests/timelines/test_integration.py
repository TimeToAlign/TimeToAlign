"""Integration tests for Timeline with loaders.

This module tests:
- Creating Timelines from MIDI loaders
- Creating Timelines from Score loaders
- Nesting events from multiple sources in a single Timeline
- Real-world MIDI file specimens

Validity Rationale:
    The Timeline class must integrate seamlessly with the loader infrastructure.
    These tests verify:
    1. Events from MidiEventData can populate a DiscreteLogicalTimeline
    2. Events from ScoreEventData can populate a ContinuousLogicalTimeline
    3. Multiple sources can be combined into a hierarchical timeline structure
    4. Real-world test data produces valid timelines with expected event counts
"""

from __future__ import annotations

import time
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    Timeline,
)

from .conftest import MIDI_PERFORMANCE_DIR, MIDI_SCORE_DIR, VIENNA_DIR

# Try to import loaders (they may not all be available)
try:
    from timetoalign.loader.midi import (
        MidiEventData,
        PerformanceMidiLoader,
        ScoreMidiLoader,
    )

    HAS_MIDI_LOADERS = True
except ImportError:
    HAS_MIDI_LOADERS = False


# region EventData Integration Tests


class TestEventDataIntegration:
    """Test Timeline integration with EventData."""

    def test_timeline_from_event_data_empty(self):
        """Empty EventData creates empty timeline."""
        from timetoalign.storage import EventData

        data = EventData.empty(TimeUnit.seconds, NumberType.float)
        tl = Timeline.from_event_data(data)

        assert tl.length.value == 0
        assert tl.n_events == 0
        assert tl.unit == TimeUnit.seconds

    def test_timeline_from_event_data_with_events(self):
        """EventData with events creates populated timeline."""
        from timetoalign.storage import EventData

        events = [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "e2",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 5.0,
            },
            {
                "id": "e3",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 2.0,
                "end": 8.0,
            },
        ]
        data = EventData.from_dicts(events, TimeUnit.seconds, NumberType.float)
        tl = Timeline.from_event_data(data)

        assert tl.length.value == 8.0  # Max end coordinate
        assert tl.n_events == 3
        assert tl.unit == TimeUnit.seconds

    def test_timeline_preserves_event_data_unit(self):
        """Timeline inherits unit from EventData."""
        from timetoalign.storage import EventData

        events = [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Tick",
                "instant": 480,
            },
        ]
        data = EventData.from_dicts(events, TimeUnit.ticks, NumberType.int)
        tl = Timeline.from_event_data(data)

        assert tl.unit == TimeUnit.ticks
        assert tl.number_type == NumberType.int


# endregion


# region MIDI Loader Integration Tests


@pytest.mark.skipif(not HAS_MIDI_LOADERS, reason="MIDI loaders not available")
class TestMidiLoaderIntegration:
    """Test Timeline integration with MIDI loaders."""

    def test_discrete_timeline_from_midi_data(self):
        """DiscreteLogicalTimeline can hold MidiEventData events."""
        # Create sample MIDI-like events
        events = [
            {
                "id": "n1",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 0,
                "end": 480,
                "duration": 480,
            },
            {
                "id": "n2",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 480,
                "end": 960,
                "duration": 480,
            },
            {
                "id": "n3",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 960,
                "end": 1920,
                "duration": 960,
            },
        ]

        data = MidiEventData.from_dicts(events, TimeUnit.ticks, NumberType.int)
        tl = DiscreteLogicalTimeline.from_event_data(data)

        assert tl.unit == TimeUnit.ticks
        assert tl.number_type == NumberType.int
        assert tl.length.value == 1920
        assert tl.n_events == 3

    @pytest.fixture
    def performance_midi_path(self) -> Path:
        """Path to a MIDI performance file."""
        path = VIENNA_DIR / "Chopin_op10_no3_p01.mid"
        return path

    @pytest.fixture
    def score_midi_path(self) -> Path:
        """Path to a MIDI score file."""
        path = MIDI_SCORE_DIR / "beethoven_mtd.mid"
        return path

    def test_timeline_from_performance_midi(self, performance_midi_path: Path):
        """Load performance MIDI into Timeline (using loader's unit/number_type)."""
        loader = PerformanceMidiLoader()
        loader.load(performance_midi_path)

        # Use base Timeline to respect loader's number_type (which may be float)
        tl = Timeline.from_event_data(loader.events)

        assert tl.unit == TimeUnit.ticks
        assert tl.n_events == 3874  # Chopin_op10_no3_p01.mid
        assert tl.length.value == 83034.0

    def test_timeline_from_score_midi(self, score_midi_path: Path):
        """Load score MIDI into Timeline (using loader's unit/number_type)."""
        loader = ScoreMidiLoader()
        loader.load(score_midi_path)

        # Use base Timeline to respect loader's number_type
        tl = Timeline.from_event_data(loader.events)

        assert tl.unit == TimeUnit.ticks
        assert tl.n_events == 16  # beethoven_mtd.mid

    def test_nested_midi_timelines(
        self, performance_midi_path: Path, score_midi_path: Path
    ):
        """Create parent timeline with MIDI children."""
        perf_loader = PerformanceMidiLoader()
        perf_loader.load(performance_midi_path)

        score_loader = ScoreMidiLoader()
        score_loader.load(score_midi_path)

        # Use base Timeline to respect loader's number_type
        perf_tl = Timeline.from_event_data(perf_loader.events, uid="performance")
        score_tl = Timeline.from_event_data(score_loader.events, uid="score")

        # Determine max length for parent
        max_len = max(perf_tl.length.value, score_tl.length.value)
        parent = Timeline(length=max_len * 2, unit=TimeUnit.ticks, uid="parent")

        # Add as children at different offsets
        parent.add_child(perf_tl, offset=0)
        parent.add_child(score_tl, offset=max_len)

        assert parent.n_children == 2
        assert "performance" in parent
        assert "score" in parent


# endregion


# region Score Loader Integration Tests


class TestScoreLoaderIntegration:
    """Test Timeline integration with Score loaders."""

    def test_continuous_timeline_from_note_data(self):
        """ContinuousLogicalTimeline can hold NoteEventData events."""
        from timetoalign.storage import EventData

        # Create sample note events in quarter notes using base EventData
        events = [
            {
                "id": "n1",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": float(Fraction(0)),
                "end": float(Fraction(1)),
                "duration": float(Fraction(1)),
            },
            {
                "id": "n2",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": float(Fraction(1)),
                "end": float(Fraction(2)),
                "duration": float(Fraction(1)),
            },
            {
                "id": "n3",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": float(Fraction(2)),
                "end": float(Fraction(4)),
                "duration": float(Fraction(2)),
            },
        ]

        data = EventData.from_dicts(events, TimeUnit.quarters, NumberType.float)
        tl = ContinuousLogicalTimeline.from_event_data(data)

        assert tl.unit == TimeUnit.quarters
        assert tl.number_type == NumberType.float
        assert tl.length.value == 4.0  # Max end coordinate
        assert tl.n_events == 3

    def test_timeline_from_score_store_combined(self):
        """Create timeline containing all events from a ScoreStore."""
        from timetoalign.storage import EventData

        # Create a mock score data with events using base EventData
        notes = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 1.0,
                    "end": 2.0,
                },
            ],
            TimeUnit.quarters,
            NumberType.float,
        )

        # Create parent timeline and add notes as child
        max_coord = 2.0  # Based on note end coordinates
        parent = ContinuousLogicalTimeline(
            length=max_coord * 2, unit=TimeUnit.quarters, uid="score_parent"
        )

        # Create child timeline from note data
        notes_tl = ContinuousLogicalTimeline.from_event_data(notes, uid="notes")
        parent.add_child(notes_tl, offset=0.0)

        assert parent.n_children == 1
        assert notes_tl.n_events == 2


# endregion


# region Complex Hierarchy Tests


class TestComplexHierarchy:
    """Test complex timeline hierarchies with multiple sources."""

    def test_multi_level_nesting_with_events(self):
        """Multi-level hierarchy with events at each level."""
        # Root timeline
        root = ContinuousPhysicalTimeline(length=100.0, uid="root")

        # Add events to root
        root.add_events(
            [
                {
                    "id": "root_e1",
                    "temporal_type": "instant",
                    "event_type": "Marker",
                    "instant": 0.0,
                },
                {
                    "id": "root_e2",
                    "temporal_type": "instant",
                    "event_type": "Marker",
                    "instant": 50.0,
                },
            ]
        )

        # Section 1
        section1 = ContinuousPhysicalTimeline(length=30.0, uid="section1")
        section1.add_events(
            [
                {
                    "id": "s1_e1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0.0,
                    "end": 10.0,
                },
                {
                    "id": "s1_e2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 10.0,
                    "end": 20.0,
                },
            ]
        )

        # Phrase within section 1
        phrase1 = ContinuousPhysicalTimeline(length=10.0, uid="phrase1")
        phrase1.add_events(
            [
                {
                    "id": "p1_e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 0.0,
                },
                {
                    "id": "p1_e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 5.0,
                },
            ]
        )
        section1.add_child(phrase1, offset=5.0)

        # Section 2
        section2 = ContinuousPhysicalTimeline(length=40.0, uid="section2")
        section2.add_events(
            [
                {
                    "id": "s2_e1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0.0,
                    "end": 20.0,
                },
            ]
        )

        # Add sections to root
        root.add_child(section1, offset=10.0)
        root.add_child(section2, offset=50.0)

        # Verify structure
        assert root.n_events == 2
        assert root.n_children == 2
        assert section1.n_events == 2
        assert section1.n_children == 1
        assert phrase1.n_events == 2

        # Verify iteration
        all_children = list(root.iter_children(order="sorted"))
        assert len(all_children) == 3  # section1, phrase1, section2

        # Check offsets are absolute
        offsets = {c.id: o.value for o, c in all_children}
        assert offsets["section1"] == 10.0
        assert offsets["phrase1"] == 15.0  # 10 + 5
        assert offsets["section2"] == 50.0

    def test_serialization_roundtrip_complex(self):
        """Complex hierarchy survives serialization roundtrip."""
        # Build structure
        root = ContinuousPhysicalTimeline(length=50.0, uid="root")
        root.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 0.0,
                },
            ]
        )

        child = ContinuousPhysicalTimeline(length=20.0, uid="child")
        child.add_events(
            [
                {
                    "id": "e2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 5.0,
                    "end": 15.0,
                },
            ]
        )

        grandchild = ContinuousPhysicalTimeline(length=5.0, uid="grandchild")
        grandchild.add_events(
            [
                {
                    "id": "e3",
                    "temporal_type": "instant",
                    "event_type": "Tick",
                    "instant": 2.0,
                },
            ]
        )

        child.add_child(grandchild, offset=10.0)
        root.add_child(child, offset=10.0)

        # Serialize
        data = root.to_dict(events=True)

        # Deserialize
        restored = ContinuousPhysicalTimeline.from_dict(data)

        # Verify
        assert restored.id == "root"
        assert restored.n_events == 1
        assert restored.n_children == 1

        restored_child = restored.get_child("child")
        assert restored_child.n_events == 1
        assert restored_child.n_children == 1

        restored_grandchild = restored_child.get_child("grandchild")
        assert restored_grandchild.n_events == 1


# endregion


# region Performance Integration Tests


class TestPerformanceIntegration:
    """Performance tests for loader integration."""

    def test_large_event_data_to_timeline(self, profiler):
        """Benchmark creating timeline from large EventData."""
        from timetoalign.storage import EventData

        n_events = 50000
        events = [
            {
                "id": f"e_{i}",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": float(i),
                "end": float(i + 1),
                "duration": 1.0,
            }
            for i in range(n_events)
        ]

        data = EventData.from_dicts(events, TimeUnit.seconds, NumberType.float)

        start = time.perf_counter()
        tl = Timeline.from_event_data(data)
        elapsed = time.perf_counter() - start

        profiler.record("timeline_from_50k_events", elapsed)

        assert tl.n_events == n_events
        # Should complete in reasonable time
        assert (
            elapsed < 5.0
        ), f"Creating timeline from {n_events} events took {elapsed:.2f}s"

    @pytest.mark.skipif(not HAS_MIDI_LOADERS, reason="MIDI loaders not available")
    def test_real_midi_file_performance(self, profiler):
        """Benchmark loading real MIDI file into timeline."""
        # Try to find any available MIDI file
        midi_files = list(MIDI_PERFORMANCE_DIR.glob("*.mid"))
        if not midi_files:
            midi_files = list(MIDI_SCORE_DIR.glob("*.mid"))
        if not midi_files:
            pytest.skip("No MIDI test files available")

        midi_path = midi_files[0]

        start = time.perf_counter()
        loader = PerformanceMidiLoader()
        loader.load(midi_path)
        load_elapsed = time.perf_counter() - start

        start = time.perf_counter()
        # Use base Timeline to respect loader's number_type
        tl = Timeline.from_event_data(loader.events)
        timeline_elapsed = time.perf_counter() - start

        profiler.record(f"load_midi_{midi_path.name}", load_elapsed)
        profiler.record(f"timeline_from_midi_{midi_path.name}", timeline_elapsed)

        assert tl.n_events > 0


# endregion


# region Edge Cases


class TestEdgeCases:
    """Test edge cases in loader integration."""

    def test_timeline_from_empty_event_list(self):
        """Timeline.from_events handles empty list gracefully."""
        tl = Timeline.from_events([])
        assert tl.length.value == 0
        assert tl.n_events == 0

    def test_timeline_from_events_instant_only(self):
        """Timeline length calculated correctly for instant-only events."""
        events = [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "e2",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 10.0,
            },
        ]
        tl = Timeline.from_events(events)
        assert tl.length.value == 10.0

    def test_timeline_from_events_interval_only(self):
        """Timeline length calculated from interval end coordinates."""
        events = [
            {
                "id": "e1",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 0.0,
                "end": 5.0,
            },
            {
                "id": "e2",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 2.0,
                "end": 8.0,
            },
        ]
        tl = Timeline.from_events(events)
        assert tl.length.value == 8.0  # Max end coordinate

    def test_timeline_with_zero_duration_interval(self):
        """Zero-duration intervals are handled correctly."""
        events = [
            {
                "id": "e1",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 5.0,
                "end": 5.0,
                "duration": 0.0,
            },
        ]
        tl = Timeline.from_events(events)
        assert tl.length.value == 5.0
        assert tl.n_events == 1


# endregion
