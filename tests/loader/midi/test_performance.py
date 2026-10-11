"""Tests for PerformanceMidiLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.loader.midi import (
    MidiEventData,
    MidiEventType,
    PerformanceMidiLoader,
    ScoreMidiEventData,
)
from timetoalign.timelines import TimelineGroup


class TestPerformanceMidiLoader:
    """Tests for loading performance MIDI files."""

    @pytest.mark.slow
    def test_supra_raw_preserves_control_events(self, supra_raw_path: Path) -> None:
        """Pin unique real-file event-type golds for the raw performance MIDI."""
        if not supra_raw_path.exists():
            pytest.skip("Test data not found")

        loader = PerformanceMidiLoader()
        loader.load(supra_raw_path)

        event_counts = loader.count_events_by_type()
        assert len(loader) == 30096
        assert event_counts[MidiEventType.NOTE] == 30092
        assert event_counts[MidiEventType.CONTROL_CHANGE] == 4

    @pytest.mark.slow
    def test_supra_raw_extent_creates_timeline_group(
        self, supra_raw_path: Path
    ) -> None:
        """Pin unique real-file extent golds and timeline-group construction."""
        if not supra_raw_path.exists():
            pytest.skip("Test data not found")

        loader = PerformanceMidiLoader()
        loader.load(supra_raw_path)

        assert loader.events.summary()["coordinate_range"] == (0.0, 277776.0)
        timeline = loader.create_timeline(uid="supra:dlt_1")
        assert timeline.length.value == 277776
        group = TimelineGroup(id="supra", timelines=[timeline])
        assert group.timeline_ids == ["supra:dlt_1"]

    def test_note_pairing(self, tmp_path: Path) -> None:
        """Loader correctly pairs note_on and note_off."""
        # Create a simple MIDI file
        import mido

        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Note C4, vel 100, duration 480 ticks
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))

        midi_path = tmp_path / "test.mid"
        mid.save(midi_path)

        loader = PerformanceMidiLoader()
        loader.load(midi_path)

        assert len(loader) == 1
        event = loader.events.to_dataframe().iloc[0]

        assert event["event_type"] == MidiEventType.NOTE
        assert event["pitch"] == 60
        assert event["duration"] == 480

    def test_performance_emits_narrow_schema(self, tmp_path: Path) -> None:
        """Performance MIDI emits the narrower 7-extra-column schema.

        ``ScoreMidiEventData``-only columns (``voice``, ``staff``,
        ``part_id``) must NOT appear on the performance-MIDI store —
        partitura is the only loader that can populate them.
        """
        import mido

        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        track.append(mido.Message("control_change", control=64, value=127, time=0))
        midi_path = tmp_path / "narrow.mid"
        mid.save(midi_path)

        loader = PerformanceMidiLoader(include_controls=True)
        loader.load(midi_path)

        # The concrete EventData class is the narrower base, not the
        # wider score-side subclass.
        assert type(loader.events) is MidiEventData
        assert not isinstance(loader.events, ScoreMidiEventData)

        columns = set(loader.events.table.column_names)
        for required in (
            "pitch",
            "velocity",
            "channel",
            "track",
            "control",
            "value",
            "program",
        ):
            assert required in columns, f"missing {required} column"
        for forbidden in ("voice", "staff", "part_id"):
            assert forbidden not in columns, f"unexpected {forbidden} column"

    def test_nullable_coordinates_preserve_exact_extent(self, tmp_path: Path) -> None:
        """Mixed instant/interval coordinates stay nullable Arrow values."""
        import mido

        mid = mido.MidiFile(type=0)
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.Message("program_change", program=4, time=0))
        track.append(mido.Message("note_on", note=60, velocity=100, time=240))
        track.append(mido.Message("control_change", control=64, value=127, time=240))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        midi_path = tmp_path / "nullable.mid"
        mid.save(midi_path)

        loader = PerformanceMidiLoader.from_file(midi_path)

        starts = loader.events.table.column("start").combine_chunks().field("value")
        ends = loader.events.table.column("end").combine_chunks().field("value")
        assert sorted(starts.to_pylist()) == [0.0, 240.0, 480.0]
        assert ends.to_pylist() == [None, None, 960.0]
        assert loader.events.summary()["coordinate_range"] == (0.0, 960.0)

        timeline = loader.create_timeline(uid="perf:dlt_1")
        assert timeline.length.value == 960
        group = TimelineGroup(id="perf", timelines=[timeline])
        assert group.timeline_ids == ["perf:dlt_1"]
