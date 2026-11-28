"""Tests for ScoreMidiLoader."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import mido
import pytest

import timetoalign.loader.midi.score as score_module
from tests.loader.minting import shape
from timetoalign.loader.midi import (
    MidiEventData,
    MidiEventType,
    PerformanceMidiLoader,
    ScoreMidiEventData,
    ScoreMidiLoader,
)

SUPRA_RAW_NOTE_COUNT = 30092
SUPRA_RAW_FIRST_NOTE_START = 741.0
SUPRA_RAW_LAST_NOTE_START = 277754.0
SUPRA_RAW_PITCH_SUM = 1938603
BEETHOVEN_NOTE_COUNT = 3751
BEETHOVEN_MIDO_FIRST_NOTE_COORDINATES = (0.0, 252.0, 252.0)
BEETHOVEN_PARTITURA_FIRST_NOTE_COORDINATES = (0.0, 252.0, 252.0)
MIDO_BEETHOVEN_SORTED_PITCH_MULTISETS = (
    (36, 22),
    (38, 10),
    (39, 8),
    (40, 1),
    (41, 21),
    (42, 1),
    (43, 109),
    (44, 46),
    (45, 8),
    (46, 33),
    (47, 18),
    (48, 191),
    (49, 17),
    (50, 46),
    (51, 50),
    (52, 20),
    (53, 39),
    (54, 17),
    (55, 283),
    (56, 37),
    (57, 31),
    (58, 71),
    (59, 80),
    (60, 306),
    (61, 29),
    (62, 172),
    (63, 211),
    (64, 83),
    (65, 145),
    (66, 32),
    (67, 200),
    (68, 71),
    (69, 41),
    (70, 49),
    (71, 126),
    (72, 201),
    (73, 18),
    (74, 193),
    (75, 139),
    (76, 69),
    (77, 135),
    (78, 31),
    (79, 151),
    (80, 51),
    (81, 20),
    (82, 26),
    (83, 5),
    (84, 26),
    (85, 4),
    (86, 14),
    (87, 7),
    (88, 9),
    (89, 7),
    (90, 1),
    (91, 14),
    (92, 1),
    (96, 5),
)
PARTITURA_BEETHOVEN_SORTED_PITCH_MULTISETS = (
    (36, 22),
    (38, 10),
    (39, 8),
    (40, 1),
    (41, 21),
    (42, 1),
    (43, 109),
    (44, 46),
    (45, 8),
    (46, 33),
    (47, 18),
    (48, 191),
    (49, 17),
    (50, 46),
    (51, 50),
    (52, 20),
    (53, 39),
    (54, 17),
    (55, 283),
    (56, 37),
    (57, 31),
    (58, 71),
    (59, 80),
    (60, 306),
    (61, 29),
    (62, 172),
    (63, 211),
    (64, 83),
    (65, 145),
    (66, 32),
    (67, 200),
    (68, 71),
    (69, 41),
    (70, 49),
    (71, 126),
    (72, 201),
    (73, 18),
    (74, 193),
    (75, 139),
    (76, 69),
    (77, 135),
    (78, 31),
    (79, 151),
    (80, 51),
    (81, 20),
    (82, 26),
    (83, 5),
    (84, 26),
    (85, 4),
    (86, 14),
    (87, 7),
    (88, 9),
    (89, 7),
    (90, 1),
    (91, 14),
    (92, 1),
    (96, 5),
)


def _notes_in_the_file(path: Path, quantum: int | None = None) -> Counter:
    """Every note of a MIDI file as the bytes state it, read with mido alone.

    A note-on of velocity above 0 starts a note of its track, channel and
    pitch; the next note-off or zero-velocity note-on of the same key ends it.

    Args:
        path: The MIDI file.
        quantum: Move every message time to the nearest multiple of it (ties
            to the even multiple) before pairing, as ``quantization_unit``
            asks of Partitura.

    Returns:
        The multiset of ``(start tick, pitch, duration in ticks, velocity)``.
    """
    notes: Counter = Counter()
    for track in mido.MidiFile(path).tracks:
        time = 0
        sounding: dict[tuple[int, int], tuple[int, int]] = {}
        for message in track:
            time += message.time
            if message.type not in ("note_on", "note_off"):
                continue
            tick = quantum * round(time / quantum) if quantum else time
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                sounding[key] = (tick, message.velocity)
            elif key in sounding:
                start, velocity = sounding.pop(key)
                notes[(start, message.note, tick - start, velocity)] += 1
    return notes


def _notes_loaded(loader: ScoreMidiLoader) -> Counter:
    """The loader's notes as ``(start, pitch, duration, velocity)``, ticks exact."""
    rows = loader.events.table.to_pylist()
    assert {row["event_type"] for row in rows} == {MidiEventType.NOTE}
    for row in rows:
        assert row["start"]["denominator"] == 1
        assert row["duration"]["denominator"] == 1
    return Counter(
        (
            row["start"]["numerator"],
            row["pitch"],
            row["duration"]["numerator"],
            row["velocity"],
        )
        for row in rows
    )


class TestScoreMidiVelocities:
    """Both parsers write the velocity each note-on states (see README)."""

    @pytest.mark.parametrize(
        ("file_name", "velocities"),
        [
            ("beethoven_op18.mid", {100: 3751}),
            ("beethoven_mtd.mid", {80: 16}),
            ("rachmaninoff_piano.mid", {112: 98, 127: 13}),
        ],
    )
    def test_the_default_parser_reads_the_stated_velocities(
        self, score_midi_dir: Path, file_name: str, velocities: dict[int, int]
    ) -> None:
        path = score_midi_dir / file_name
        loader = ScoreMidiLoader().load(path)

        assert Counter(loader.events.table.column("velocity").to_pylist()) == velocities
        stated = _notes_in_the_file(path)
        stated_velocities: Counter = Counter()
        for (_, _, _, velocity), count in stated.items():
            stated_velocities[velocity] += count
        assert stated_velocities == velocities
        assert _notes_loaded(loader) == stated

    def test_the_mido_parser_reads_the_stated_velocities(
        self, beethoven_score_path: Path
    ) -> None:
        loader = ScoreMidiLoader(parser="mido").load(beethoven_score_path)

        assert Counter(loader.events.table.column("velocity").to_pylist()) == {
            100: 3751
        }
        assert _notes_loaded(loader) == _notes_in_the_file(beethoven_score_path)

    def test_quantized_notes_keep_their_velocities(self, score_midi_dir: Path) -> None:
        path = score_midi_dir / "rachmaninoff_piano.mid"
        loader = ScoreMidiLoader(quantization_unit=10).load(path)

        assert Counter(loader.events.table.column("velocity").to_pylist()) == {
            112: 98,
            127: 13,
        }
        assert _notes_loaded(loader) == _notes_in_the_file(path, quantum=10)

    def test_a_fractional_unit_compares_partituras_times_unaltered(
        self, chopin_perf_path: Path
    ) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader = ScoreMidiLoader(quantization_unit=2.5).load(chopin_perf_path)

        velocities = loader.events.table.column("velocity").to_pylist()
        assert (velocities.count(None), len(velocities)) == (288, 451)
        messages = [str(w.message) for w in caught if w.category is UserWarning]
        assert len(messages) == 1
        assert messages[0].startswith(
            f"288 of 451 notes of '{chopin_perf_path}' are left without a velocity: "
        )

    def test_a_note_that_cannot_be_joined_keeps_an_empty_velocity(
        self, score_midi_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = score_midi_dir / "rachmaninoff_piano.mid"
        read = score_module._partitura_note_velocities

        def renumbered(mid: mido.MidiFile, quantization_unit: int | None) -> list:
            notes = read(mid, quantization_unit)
            return notes[1:] + notes[:1]

        monkeypatch.setattr(score_module, "_partitura_note_velocities", renumbered)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            loader = ScoreMidiLoader().load(path)

        rows = loader.events.table.to_pylist()
        assert [row["velocity"] for row in rows] == [None] * 111
        assert Counter(
            (start, pitch, duration)
            for start, pitch, duration, _ in _notes_loaded(loader).elements()
        ) == Counter(
            (start, pitch, duration)
            for start, pitch, duration, _ in _notes_in_the_file(path).elements()
        )
        # Partitura's note n0 is the first row: rows keep Partitura's order,
        # and their IDs are minted, never Partitura's numbering.
        first = rows[0]
        assert all(shape(row["id"]) == "note" for row in rows)
        assert (
            first["start"]["numerator"],
            first["pitch"],
            first["duration"]["numerator"],
        ) == (0, 24, 106)
        assert [(w.category, str(w.message)) for w in caught] == [
            (
                UserWarning,
                f"111 of 111 notes of '{path}' are left without a velocity: "
                "Partitura's note 'n0' (onset 0, pitch 24, duration 106) is the first "
                "that is not the file's note of that number. parser='mido' reads "
                "every velocity.",
            )
        ]


class TestScoreMidiLoader:
    """Tests for loading score MIDI files."""

    @pytest.mark.slow
    def test_load_beethoven_quartet(self, beethoven_score_path: Path) -> None:
        """Can load multi-part score MIDI."""
        loader = ScoreMidiLoader(part_voice_assign_mode=0)
        loader.load(beethoven_score_path)

        assert len(loader) == 3751  # beethoven_op18.mid has 3751 events
        assert loader.ticks_per_beat == 480  # the file header's division

        # Check metadata
        meta = loader.metadata["sources"][0]
        assert meta["parser"] == "partitura"
        assert meta["parts"] == 4  # String quartet: 4 parts

        # Check events have score info
        df = loader.events.to_dataframe()
        assert not df["pitch"].isnull().all()
        # The file states velocity 100 on every note.
        assert loader.events.table.column("velocity").to_pylist() == [100] * 3751

    def test_load_empty_raises(self, tmp_path: Path) -> None:
        """Loading invalid/empty file raises error."""
        # ScoreMidiLoader uses partitura which raises its own errors or returns empty
        empty_file = tmp_path / "empty.mid"
        empty_file.touch()

        loader = ScoreMidiLoader()
        # ScoreMidiLoader uses partitura which raises EOFError on empty file
        # (no exception wrapping in score.py, raw partitura error escapes)
        with pytest.raises(EOFError):
            loader.load(empty_file)

    @pytest.mark.slow
    def test_score_emits_wide_schema(self, beethoven_score_path: Path) -> None:
        """Score MIDI emits the wider 10-extra-column schema.

        Partitura supplies ``voice``, ``staff`` and ``part_id`` per
        note, so the storage class is :class:`ScoreMidiEventData`
        (not the narrower performance-side :class:`MidiEventData`).
        """
        loader = ScoreMidiLoader(part_voice_assign_mode=0)
        loader.load(beethoven_score_path)

        # Concrete class is the wider subclass.
        assert type(loader.events) is ScoreMidiEventData
        # Subclass relationship is preserved.
        assert isinstance(loader.events, MidiEventData)

        columns = set(loader.events.table.column_names)
        for required in (
            "pitch",
            "velocity",
            "channel",
            "track",
            "control",
            "value",
            "program",
            "voice",
            "staff",
            "part_id",
        ):
            assert required in columns, f"missing {required} column"

    @pytest.mark.slow
    def test_mido_fast_path_matches_performance_on_supra_raw(
        self, supra_raw_path: Path
    ) -> None:
        """The score mido path preserves raw note pairing exactly."""
        score_loader = ScoreMidiLoader(parser="mido")
        performance_loader = PerformanceMidiLoader(
            include_controls=False,
            include_program_changes=False,
        )
        score_loader.load(supra_raw_path)
        performance_loader.load(supra_raw_path)

        score_notes = [
            event
            for event in score_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        performance_notes = [
            event
            for event in performance_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]

        score_metrics = (
            len(score_notes),
            score_notes[0]["start"]["value"],
            score_notes[-1]["start"]["value"],
            sum(event["pitch"] for event in score_notes),
        )
        performance_metrics = (
            len(performance_notes),
            performance_notes[0]["start"]["value"],
            performance_notes[-1]["start"]["value"],
            sum(event["pitch"] for event in performance_notes),
        )

        assert score_metrics == (
            SUPRA_RAW_NOTE_COUNT,
            SUPRA_RAW_FIRST_NOTE_START,
            SUPRA_RAW_LAST_NOTE_START,
            SUPRA_RAW_PITCH_SUM,
        )
        assert performance_metrics == (
            SUPRA_RAW_NOTE_COUNT,
            SUPRA_RAW_FIRST_NOTE_START,
            SUPRA_RAW_LAST_NOTE_START,
            SUPRA_RAW_PITCH_SUM,
        )
        assert score_metrics == performance_metrics

    def test_mido_fast_path_matches_partitura_pitch_content(
        self, beethoven_score_path: Path
    ) -> None:
        """The fast path preserves the quantized score's note pitch multiset."""
        mido_loader = ScoreMidiLoader(parser="mido")
        partitura_loader = ScoreMidiLoader(parser="partitura")
        mido_loader.load(beethoven_score_path)
        partitura_loader.load(beethoven_score_path)

        mido_notes = [
            event
            for event in mido_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        partitura_notes = [
            event
            for event in partitura_loader.events
            if event["event_type"] == MidiEventType.NOTE
        ]
        mido_pitch_multiset = tuple(
            sorted(Counter(event["pitch"] for event in mido_notes).items())
        )
        partitura_pitch_multiset = tuple(
            sorted(Counter(event["pitch"] for event in partitura_notes).items())
        )
        mido_coordinates = tuple(
            mido_notes[0][field]["value"] for field in ("start", "end", "duration")
        )
        partitura_coordinates = tuple(
            partitura_notes[0][field]["value"] for field in ("start", "end", "duration")
        )

        assert len(mido_notes) == BEETHOVEN_NOTE_COUNT
        assert len(partitura_notes) == BEETHOVEN_NOTE_COUNT
        assert mido_pitch_multiset == MIDO_BEETHOVEN_SORTED_PITCH_MULTISETS
        assert partitura_pitch_multiset == PARTITURA_BEETHOVEN_SORTED_PITCH_MULTISETS
        assert mido_pitch_multiset == partitura_pitch_multiset
        assert mido_coordinates == BEETHOVEN_MIDO_FIRST_NOTE_COORDINATES
        assert partitura_coordinates == BEETHOVEN_PARTITURA_FIRST_NOTE_COORDINATES

    def test_mido_fast_path_emits_narrow_schema(
        self, beethoven_score_path: Path
    ) -> None:
        """The fast path does not invent score-structure columns."""
        mido_loader = ScoreMidiLoader(parser="mido")
        partitura_loader = ScoreMidiLoader(parser="partitura")
        mido_loader.load(beethoven_score_path)
        partitura_loader.load(beethoven_score_path)

        assert isinstance(mido_loader.events, MidiEventData)
        assert not isinstance(mido_loader.events, ScoreMidiEventData)
        assert isinstance(partitura_loader.events, ScoreMidiEventData)
        assert mido_loader.events.to_dataframe().columns.tolist() == [
            "id",
            "name",
            "temporal_type",
            "event_type",
            "start",
            "end",
            "duration",
            "pitch",
            "velocity",
            "channel",
            "track",
            "control",
            "value",
            "program",
            "locator",
            "extraction_id",
        ]
        assert "voice" not in mido_loader.events.to_dataframe().columns
        assert "staff" not in mido_loader.events.to_dataframe().columns
        assert "part_id" not in mido_loader.events.to_dataframe().columns

    def test_mido_fast_path_pins_ticks_per_beat(
        self, beethoven_score_path: Path
    ) -> None:
        """The mido path exposes the source file's exact PPQ."""
        loader = ScoreMidiLoader(parser="mido")
        loader.load(beethoven_score_path)

        assert loader.ticks_per_beat == 480

    @pytest.mark.parametrize(
        "options",
        [
            {"part_voice_assign_mode": 1},
            {"quantization_unit": 16},
            {"estimate_voice_info": True},
            {"estimate_key": True},
        ],
    )
    def test_mido_fast_path_rejects_structural_options(self, options: dict) -> None:
        """Structural score options require partitura's parser."""
        with pytest.raises(ValueError, match="need partitura"):
            ScoreMidiLoader(parser="mido", **options)

    def test_unknown_parser_raises(self) -> None:
        """Only the two supported parser names are accepted."""
        with pytest.raises(ValueError, match="parser must be"):
            ScoreMidiLoader(parser="nonsense")


def _write_two_tracks(path: Path, meta_end: int) -> Path:
    """A type-1 file: a meta track ending at *meta_end*, a note track at 1,441.

    The note track holds one note from tick 0 to 1,440, its ``end_of_track``
    one tick later.
    """
    meta = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500_000, time=0),
            mido.MetaMessage("end_of_track", time=meta_end),
        ]
    )
    notes = mido.MidiTrack(
        [
            mido.Message("note_on", note=60, velocity=64, time=0),
            mido.Message("note_off", note=60, velocity=0, time=1440),
            mido.MetaMessage("end_of_track", time=1),
        ]
    )
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks += [meta, notes]
    midi.save(path)
    return path


class TestEndOfTrack:
    """The file's extent: the tick at which its latest track ends."""

    @pytest.mark.parametrize(
        "make",
        [
            lambda: ScoreMidiLoader(),
            lambda: ScoreMidiLoader(parser="mido"),
            lambda: PerformanceMidiLoader(),
        ],
        ids=["partitura", "mido", "performance"],
    )
    def test_one_tick_after_the_last_note(
        self,
        make: Callable[[], ScoreMidiLoader | PerformanceMidiLoader],
        tmp_path: Path,
    ) -> None:
        loader = make()
        assert loader.end_of_track is None
        loader.load(_write_two_tracks(tmp_path / "theme.mid", meta_end=1))
        assert loader.end_of_track == 1441
        assert loader.metadata["sources"][0]["end_of_track"] == 1441

    @pytest.mark.parametrize(
        "make",
        [
            lambda: ScoreMidiLoader(),
            lambda: ScoreMidiLoader(parser="mido"),
            lambda: PerformanceMidiLoader(),
        ],
        ids=["partitura", "mido", "performance"],
    )
    def test_clear_forgets_the_file(
        self,
        make: Callable[[], ScoreMidiLoader | PerformanceMidiLoader],
        tmp_path: Path,
    ) -> None:
        loader = make()
        loader.load(_write_two_tracks(tmp_path / "theme.mid", meta_end=1))
        assert (loader.end_of_track, loader.ticks_per_beat) == (1441, 480)
        loader.clear()
        assert (loader.end_of_track, loader.ticks_per_beat) == (None, None)
        loader.load(_write_two_tracks(tmp_path / "other.mid", meta_end=2000))
        assert (loader.end_of_track, loader.ticks_per_beat) == (2000, 480)

    def test_the_latest_track_decides(self, tmp_path: Path) -> None:
        loader = ScoreMidiLoader(parser="mido").load(
            _write_two_tracks(tmp_path / "theme.mid", meta_end=2000)
        )
        assert loader.end_of_track == 2000


if __name__ == "__main__":
    pytest.main([__file__])
