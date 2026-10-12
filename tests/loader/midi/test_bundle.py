"""Tests for MidiStore.

This module tests the MidiStore class which splits a MidiEventData
into separate notes and controls data.
"""

from __future__ import annotations

import pytest

from timetoalign.core import TimeUnit
from timetoalign.loader.midi import MidiEventData, MidiStore
from timetoalign.timelines import Timeline


def _child_names(timeline: Timeline) -> list[str]:
    """The names of a timeline's children, in child order."""
    return [timeline.get_child(child_id).name for child_id in timeline.list_children()]


def _child_named(timeline: Timeline, name: str) -> Timeline:
    """The one child of *timeline* named *name*; child ids are minted."""
    (child,) = [
        timeline.get_child(child_id)
        for child_id in timeline.list_children()
        if timeline.get_child(child_id).name == name
    ]
    return child


class TestMidiStoreFromData:
    """Tests for MidiStore.from_data() splitting behavior."""

    @pytest.fixture
    def mixed_midi_data(self) -> MidiEventData:
        """MidiEventData with notes and control events."""
        return MidiEventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 60,
                    "velocity": 64,
                    "channel": 0,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                    "pitch": 62,
                    "velocity": 80,
                    "channel": 0,
                },
                {
                    "id": "cc1",
                    "temporal_type": "instant",
                    "event_type": "ControlChange",
                    "instant": 0,
                    "control": 64,
                    "value": 127,
                    "channel": 0,
                },
                {
                    "id": "pc1",
                    "temporal_type": "instant",
                    "event_type": "ProgramChange",
                    "instant": 0,
                    "program": 1,
                    "channel": 0,
                },
            ],
            unit=TimeUnit.ticks,
        )

    def test_from_store_splits_notes(self, mixed_midi_data: MidiEventData):
        """from_store separates Note events into notes data."""
        store = MidiStore.from_data(mixed_midi_data)

        # Exact count: 2 notes
        assert len(store.notes) == 2

    def test_from_store_splits_controls(self, mixed_midi_data: MidiEventData):
        """from_store separates control events into controls data."""
        store = MidiStore.from_data(mixed_midi_data)

        # Exact count: 1 ControlChange + 1 ProgramChange = 2
        assert len(store.controls) == 2

    def test_notes_have_correct_event_type(self, mixed_midi_data: MidiEventData):
        """All events in notes data have event_type='Note'."""
        store = MidiStore.from_data(mixed_midi_data)

        for event in store.notes:
            assert event["event_type"] == "Note"

    def test_controls_have_control_event_types(self, mixed_midi_data: MidiEventData):
        """All events in controls data have control event types."""
        store = MidiStore.from_data(mixed_midi_data)

        valid_types = {"ControlChange", "ProgramChange", "PitchBend"}
        for event in store.controls:
            assert event["event_type"] in valid_types

    def test_metadata_preserved(self, mixed_midi_data: MidiEventData):
        """Metadata is attached to store."""
        metadata = {"ticks_per_beat": 480, "format": 1}
        store = MidiStore.from_data(mixed_midi_data, metadata=metadata)

        assert store.metadata["ticks_per_beat"] == 480
        assert store.metadata["format"] == 1


class TestMidiStoreEmpty:
    """Tests for MidiStore.empty()."""

    def test_empty_store_has_empty_data(self):
        """Empty store has zero events in both data objects."""
        store = MidiStore.empty()

        assert len(store.notes) == 0
        assert len(store.controls) == 0

    def test_empty_store_metadata(self):
        """Empty store has empty metadata."""
        store = MidiStore.empty()

        assert store.metadata == {}


class TestMidiStoreExtend:
    """Tests for MidiStore.extend()."""

    @pytest.fixture
    def store_a(self) -> MidiStore:
        """First MidiStore with 2 notes."""
        data = MidiEventData.from_dicts(
            [
                {
                    "id": "a_n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 60,
                },
                {
                    "id": "a_n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                    "pitch": 62,
                },
            ],
            unit=TimeUnit.ticks,
        )
        return MidiStore.from_data(data, metadata={"source": "a"})

    @pytest.fixture
    def store_b(self) -> MidiStore:
        """Second MidiStore with 1 note and 1 control."""
        data = MidiEventData.from_dicts(
            [
                {
                    "id": "b_n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 64,
                },
                {
                    "id": "b_cc1",
                    "temporal_type": "instant",
                    "event_type": "ControlChange",
                    "instant": 0,
                    "control": 64,
                },
            ],
            unit=TimeUnit.ticks,
        )
        return MidiStore.from_data(data, metadata={"source": "b"})

    def test_extend_merges_notes(self, store_a: MidiStore, store_b: MidiStore):
        """extend() merges notes from both stores."""
        store_a.extend(store_b)

        # Exact count: 2 from a + 1 from b = 3
        assert len(store_a.notes) == 3

    def test_extend_merges_controls(self, store_a: MidiStore, store_b: MidiStore):
        """extend() merges controls from both stores."""
        store_a.extend(store_b)

        # Exact count: 0 from a + 1 from b = 1
        assert len(store_a.controls) == 1

    def test_extend_updates_metadata(self, store_a: MidiStore, store_b: MidiStore):
        """extend() updates metadata (last wins)."""
        store_a.extend(store_b)

        assert store_a.metadata["source"] == "b"


class TestMidiStoreProtocol:
    """Tests for MidiStore EventStore protocol compliance."""

    @pytest.fixture
    def sample_store(self) -> MidiStore:
        """MidiStore for protocol testing."""
        data = MidiEventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 60,
                },
                {
                    "id": "cc1",
                    "temporal_type": "instant",
                    "event_type": "ControlChange",
                    "instant": 0,
                },
            ],
            unit=TimeUnit.ticks,
        )
        return MidiStore.from_data(data)

    def test_keys_canonical_order(self, sample_store: MidiStore):
        """keys() returns canonical order."""
        assert sample_store.keys() == ("notes", "controls")

    def test_iteration_order(self, sample_store: MidiStore):
        """Iteration yields data in canonical order."""
        data_items = list(sample_store)

        assert data_items[0] is sample_store.notes
        assert data_items[1] is sample_store.controls

    def test_items_order(self, sample_store: MidiStore):
        """items() yields in canonical order."""
        items = list(sample_store.items())

        assert items[0] == ("notes", sample_store.notes)
        assert items[1] == ("controls", sample_store.controls)

    def test_getitem_notes(self, sample_store: MidiStore):
        """Can access notes by name."""
        assert sample_store["notes"] is sample_store.notes

    def test_getitem_controls(self, sample_store: MidiStore):
        """Can access controls by name."""
        assert sample_store["controls"] is sample_store.controls

    def test_getitem_invalid_raises(self, sample_store: MidiStore):
        """Invalid name raises KeyError."""
        with pytest.raises(KeyError, match="invalid"):
            _ = sample_store["invalid"]

    def test_len(self, sample_store: MidiStore):
        """Length is always 2."""
        assert len(sample_store) == 2

    def test_contains(self, sample_store: MidiStore):
        """Membership check works."""
        assert "notes" in sample_store
        assert "controls" in sample_store
        assert "invalid" not in sample_store


class TestMidiStoreSummary:
    """Tests for MidiStore.summary()."""

    def test_summary_counts(self):
        """summary() includes correct counts."""
        data = MidiEventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                },
                {
                    "id": "cc1",
                    "temporal_type": "instant",
                    "event_type": "ControlChange",
                    "instant": 0,
                },
            ],
            unit=TimeUnit.ticks,
        )
        store = MidiStore.from_data(data)

        summary = store.summary()

        # Exact counts
        assert summary["tables"]["notes"]["count"] == 2
        assert summary["tables"]["controls"]["count"] == 1


class TestMidiStoreToTimeline:
    """Tests for MidiStore timeline creation."""

    @pytest.fixture
    def sample_store(self) -> MidiStore:
        """MidiStore with notes and controls."""
        data = MidiEventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                    "pitch": 60,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                    "pitch": 62,
                },
                {
                    "id": "cc1",
                    "temporal_type": "instant",
                    "event_type": "ControlChange",
                    "instant": 0,
                    "control": 64,
                },
            ],
            unit=TimeUnit.ticks,
        )
        return MidiStore.from_data(data)

    def test_create_timeline_creates_children(self, sample_store: MidiStore):
        """create_timeline creates parent with children."""
        timeline = sample_store.create_timeline(uid="test")

        assert timeline.id == "test"
        # 2 children: notes and controls
        assert timeline.n_children == 2

    def test_children_at_offset_zero(self, sample_store: MidiStore):
        """Children are embedded at offset 0."""
        timeline = sample_store.create_timeline()

        assert timeline.get_child_offset(_child_named(timeline, "notes").id).value == 0
        assert (
            timeline.get_child_offset(_child_named(timeline, "controls").id).value == 0
        )

    def test_notes_child_has_correct_count(self, sample_store: MidiStore):
        """Notes child has exact event count."""
        timeline = sample_store.create_timeline()

        notes_tl = _child_named(timeline, "notes")
        # Exact count: 2 notes
        assert notes_tl.n_events == 2

    def test_controls_child_has_correct_count(self, sample_store: MidiStore):
        """Controls child has exact event count."""
        timeline = sample_store.create_timeline()

        controls_tl = _child_named(timeline, "controls")
        # Exact count: 1 control change
        assert controls_tl.n_events == 1
