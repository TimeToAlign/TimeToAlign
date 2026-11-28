"""Loaders lock a timeline whose source states its length.

Validation logic: ``tests/loader/README.md``, section "Loaded timelines are
locked where the source states their length".
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.loader.midi import PerformanceMidiLoader, ScoreMidiLoader
from timetoalign.loader.physical import AudioLoader
from timetoalign.loader.score import (
    MeasureMapLoader,
    Ms3Loader,
    Music21Loader,
    PartituraLoader,
    VerovioLoader,
)
from timetoalign.loader.tabular import TsvLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import Timeline

WOO71 = ensure_data("score") / "beethoven_woo71"
NOTES = WOO71 / "WoO71.notes.tsv"
MEASURES = WOO71 / "WoO71.measures.tsv"
CHOPIN_XML = ensure_data("vienna_1x22") / "Chopin_op10_no3.musicxml"
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

    def test_a_region_gives_a_child(self) -> None:
        loader = AudioLoader.from_file(BASS_KICK)
        timeline = loader.create_timeline()
        end = loader.audio_info.n_samples
        timeline.create_region(0, end // 2, id="first_half")

        child = timeline.create_child_from_region("first_half")

        assert timeline.get_child_offset(child.id) == Coordinate(0, "samples")
        assert child.length == Coordinate(end // 2, "samples")
        assert (timeline.length, timeline.is_locked) == (
            Coordinate(end, "samples"),
            True,
        )


class TestMidiStatesItsLength:
    """The end of track, read off the bytes with mido (README)."""

    def test_performance_midi(self) -> None:
        loader = PerformanceMidiLoader.from_file(
            MIDI / "performance" / "rachmaninoff_perf.mid"
        )
        timeline = loader.create_timeline()
        assert (timeline.is_locked, timeline.length) == (
            True,
            Coordinate(11537, "ticks"),
        )

    @pytest.mark.parametrize("parser", ["partitura", "mido"])
    def test_score_midi(self, parser: str) -> None:
        loader = ScoreMidiLoader(parser=parser).load(
            MIDI / "score" / "beethoven_op18.mid"
        )
        timeline = loader.create_timeline()
        assert (timeline.is_locked, timeline.length) == (
            True,
            Coordinate(534240, "ticks"),
        )


class TestInferredLengthsStayUnlocked:
    def test_score_without_measures(self) -> None:
        timeline = Ms3Loader.from_file(NOTES, auto_discover=False).create_timeline()
        assert timeline.measure_map is None
        assert timeline.is_locked is False

    def test_tabular(self) -> None:
        assert _NotesTsv.from_file(NOTES).create_timeline().is_locked is False


_ONE_MOVEMENT_MEI = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
<meiHead><fileDesc><titleStmt><title>Fixture</title></titleStmt><pubStmt/></fileDesc>
</meiHead><music><body>
<mdiv xml:id="mov1"><score>
  <scoreDef xml:id="sd1" meter.count="4" meter.unit="4"><staffGrp xml:id="sg1">
    <staffDef xml:id="sdef1" n="1" lines="5" clef.shape="G" clef.line="2"/>
  </staffGrp></scoreDef>
  <section xml:id="s1">
    <measure xml:id="m1" n="1"><staff xml:id="m1s" n="1"><layer xml:id="m1l" n="1">
      <note xml:id="n1" pname="c" oct="4" dur="1"/>
    </layer></staff></measure>
    <measure xml:id="m2" n="2"><staff xml:id="m2s" n="1"><layer xml:id="m2l" n="1">
      <note xml:id="n2" pname="d" oct="4" dur="1"/>
    </layer></staff></measure>
  </section></score></mdiv>
</body></music></mei>"""
"""One movement, two 4/4 bars of one whole note each: no repeat to expand."""


def _length_and_lock(timeline: Timeline) -> tuple[Coordinate | None, bool]:
    return timeline.length, timeline.is_locked


class TestScoresWithMeasuresAreLocked:
    def test_ms3_notes_and_measures(self) -> None:
        timeline = Ms3Loader.from_file(NOTES, MEASURES).create_timeline()
        assert _length_and_lock(timeline) == (Coordinate(882, "quarters"), True)

    def test_ms3_measures_alone(self) -> None:
        timeline = Ms3Loader.from_file(MEASURES).create_timeline()
        assert _length_and_lock(timeline) == (Coordinate(882, "quarters"), True)

    def test_partitura(self) -> None:
        timeline = PartituraLoader.from_file(WOO71 / "WoO71.musicxml").create_timeline()
        assert _length_and_lock(timeline) == (Coordinate(882, "quarters"), True)

    def test_partitura_and_music21_on_chopin(self) -> None:
        for loader in (PartituraLoader, Music21Loader):
            timeline = loader.from_file(CHOPIN_XML).create_timeline()
            assert _length_and_lock(timeline) == (
                Coordinate(Fraction(83, 2), "quarters"),
                True,
            )

    def test_measure_map(self) -> None:
        timeline = MeasureMapLoader.from_file(
            WOO71 / "WoO71.measures.mm.json"
        ).create_timeline()
        assert _length_and_lock(timeline) == (Coordinate(882, "quarters"), True)

    @pytest.mark.filterwarnings("ignore")
    def test_verovio_expanded(self) -> None:
        timeline = VerovioLoader.from_file(WOO71 / "WoO71.mei").create_timeline()
        assert timeline.measure_map.total_actual_length == 1056
        assert _length_and_lock(timeline) == (Coordinate(1056, "quarters"), True)

    def test_verovio_without_repeats(self, tmp_path: Path) -> None:
        path = tmp_path / "one_movement.mei"
        path.write_text(_ONE_MOVEMENT_MEI, encoding="utf-8")
        timeline = VerovioLoader.from_file(path).create_timeline()
        assert _length_and_lock(timeline) == (Coordinate(8, "quarters"), True)

    def test_the_lock_refuses_growth_unless_stated(self) -> None:
        timeline = Ms3Loader.from_file(NOTES, MEASURES).create_timeline()
        row = {"id": "late", "event_type": "Note", "start": 882, "duration": 1}
        with pytest.raises(ValueError):
            timeline.add_events([row])
        assert timeline.length == Coordinate(882, "quarters")
        assert timeline.add_events([row], allow_expansion=True) == ["late"]
        assert timeline.length == Coordinate(883, "quarters")

    def test_a_region_gives_a_child(self) -> None:
        timeline = Ms3Loader.from_file(NOTES, MEASURES).create_timeline()
        timeline.create_region(0, 12, id="opening")

        child = timeline.create_child_from_region("opening")

        assert timeline.get_child_offset(child.id) == Coordinate(0, "quarters")
        assert child.length == Coordinate(12, "quarters")
        assert _length_and_lock(timeline) == (Coordinate(882, "quarters"), True)
