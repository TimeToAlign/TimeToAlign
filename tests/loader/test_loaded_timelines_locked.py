"""Loaders lock a timeline whose source states its length.

Validation logic: ``tests/loader/README.md``, section "Loaded timelines are
locked where the source states their length".
"""

from __future__ import annotations

import pytest

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.loader.midi import PerformanceMidiLoader, ScoreMidiLoader
from timetoalign.loader.physical import AudioLoader
from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.tabular import TsvLoader
from timetoalign.testdata import ensure_data

WOO71 = ensure_data("score") / "beethoven_woo71"
NOTES = WOO71 / "WoO71.notes.tsv"
MIDI = ensure_data("midi")
BASS_KICK = ensure_data("audio") / "hard_techno" / "Bass Kick.mp3"


class _NotesTsv(TsvLoader):
    """A notes TSV read as plain table rows, with no score structure."""

    id_column = None
    name_column = "name"
    start_column = "quarterbeats"
    end_column = None
    duration_column = "duration"
    event_type_column = None
    default_event_type = "Note"
    _default_unit = TimeUnit.quarters
    coordinate_type = NumberType.fraction


class TestAudio:
    def test_both_timelines_are_locked_at_the_stated_length(self) -> None:
        loader = AudioLoader.from_file(BASS_KICK)
        samples = loader.create_timeline()
        seconds = loader.create_continuous_timeline()
        assert (samples.is_locked, samples.length.value) == (
            True,
            loader.audio_info.n_samples,
        )
        assert (seconds.is_locked, seconds.length.value) == (
            True,
            loader.audio_info.duration_seconds,
        )

    def test_the_lock_refuses_growth_unless_stated(self) -> None:
        loader = AudioLoader.from_file(BASS_KICK)
        timeline = loader.create_timeline()
        end = loader.audio_info.n_samples
        row = {"id": "late", "event_type": "Onset", "instant": end + 1}
        with pytest.raises(ValueError):
            timeline.add_events([row])
        assert timeline.length == Coordinate(end, "samples")
        assert timeline.add_events([row], allow_expansion=True) == ["late"]
        assert timeline.length == Coordinate(end + 1, "samples")


class TestInferredLengthsStayUnlocked:
    def test_performance_midi(self) -> None:
        loader = PerformanceMidiLoader.from_file(
            MIDI / "performance" / "rachmaninoff_perf.mid"
        )
        assert loader.create_timeline().is_locked is False

    def test_score_midi(self) -> None:
        loader = ScoreMidiLoader.from_file(MIDI / "score" / "beethoven_op18.mid")
        assert loader.create_timeline().is_locked is False

    def test_score_without_measures(self) -> None:
        timeline = Ms3Loader.from_file(NOTES, auto_discover=False).create_timeline()
        assert timeline.measure_map is None
        assert timeline.is_locked is False

    def test_tabular(self) -> None:
        assert _NotesTsv.from_file(NOTES).create_timeline().is_locked is False
