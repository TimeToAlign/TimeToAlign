"""Extractions and locators: what a loader read, where each event lives in it.

The validation logic is written down in ``tests/loader/README.md``
("Extractions and locators"); every expected value is read off the file
with mido or ``hashlib`` by the test, never by the loader.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import mido
import pyarrow.compute as pc
import pytest

from timetoalign import __version__
from timetoalign.loader.alignment.matchfile import MatchfileLoader
from timetoalign.loader.midi import PerformanceMidiLoader
from timetoalign.loader.tabular import CsvLoader
from timetoalign.timelines import ContinuousLogicalTimeline
from timetoalign.timelines.factory import create_timeline_from_event_data

from .conftest import VIENNA_DATA_DIR

#: The notes the test file holds: (onset, end, pitch, velocity) in ticks.
NOTES = [(0, 240, 60, 64), (480, 720, 64, 70), (960, 1200, 67, 80)]


def write_midi(path: Path, notes: list[tuple[int, int, int, int]]) -> Path:
    """A type 1 file: a meta track, then a program change and *notes* in track 1."""
    meta = mido.MidiTrack([mido.MetaMessage("end_of_track", time=0)])
    track = mido.MidiTrack([mido.Message("program_change", program=0, time=0)])
    now = 0
    for start, end, pitch, velocity in notes:
        track.append(
            mido.Message("note_on", note=pitch, velocity=velocity, time=start - now)
        )
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=end - start))
        now = end
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks += [meta, track]
    midi.save(path)
    return path


def message_locators(path: Path) -> dict[str, str]:
    """Per note-on and program change, its locator, by kind and pitch: read with mido."""
    found: dict[str, str] = {}
    for i, track in enumerate(mido.MidiFile(path).tracks):
        for j, message in enumerate(track):
            if message.type == "note_on" and message.velocity > 0:
                found[f"Note {message.note}"] = f"track:{i} message:{j}"
            elif message.type == "program_change":
                found["ProgramChange"] = f"track:{i} message:{j}"
    return found


def digest(path: Path) -> str:
    """The SHA-256 digest of the file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows_by_locator(timeline: object) -> dict[str, dict[str, object]]:
    """The timeline's registry rows that name a locator, by locator."""
    table = timeline.registry.table  # type: ignore[attr-defined]
    table = table.filter(pc.is_valid(table.column("locator")))
    return {row["locator"]: row for row in table.to_pylist()}


PARAMETERS = {
    "unit": "ticks",
    "number_type": "int",
    "interval_policy": "warn",
    "parse_durations": True,
    "on0_means_off": True,
    "include_controls": True,
    "include_program_changes": True,
}


class TestOneExtractionPerFile:
    """One extraction per file read, its fields the run's."""

    def test_the_fields(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        (extraction,) = PerformanceMidiLoader().load(path).extractions
        assert extraction.source == path.as_posix()
        assert extraction.source_integrity == {
            "algorithm": "SHA-256",
            "digest": digest(path),
        }
        assert extraction.parser == "PerformanceMidiLoader"
        assert extraction.parser_version == (
            f"timetoalign {__version__}; mido {importlib.metadata.version('mido')}"
        )
        assert extraction.parameters == PARAMETERS
        assert (extraction.minter, extraction.id_length, extraction.seed) == (
            "splitmix64-base36-1",
            13,
            None,
        )

    def test_a_given_seed(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        (extraction,) = PerformanceMidiLoader(seed=7).load(path).extractions
        assert extraction.seed == "7"

    def test_two_files_two_extractions(self, tmp_path: Path) -> None:
        first = write_midi(tmp_path / "a.mid", NOTES)
        second = write_midi(tmp_path / "b.mid", NOTES[:1])
        loader = PerformanceMidiLoader().load(first, second)
        assert [e.source for e in loader.extractions] == [
            first.as_posix(),
            second.as_posix(),
        ]
        named = loader.events.table.column("extraction_id").to_pylist()
        assert named == [loader.extractions[0].id] * 4 + [loader.extractions[1].id] * 2

    def test_one_file_read_twice(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        loader = PerformanceMidiLoader().load(path).load(path)
        assert len(loader.extractions) == 1


class TestLocators:
    """The locator of an event is its place in the file."""

    def test_midi_messages(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        table = PerformanceMidiLoader().load(path).events.table
        kinds = table.column("event_type").to_pylist()
        pitches = table.column("pitch").to_pylist()
        keys = [
            kind if kind == "ProgramChange" else f"{kind} {pitch}"
            for kind, pitch in zip(kinds, pitches, strict=True)
        ]
        expected = message_locators(path)
        assert dict(zip(keys, table.column("locator").to_pylist(), strict=True)) == (
            expected
        )
        assert len(set(expected.values())) == 4

    def test_the_rows_reach_the_timeline(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        loader = PerformanceMidiLoader().load(path)
        timeline = loader.create_timeline()
        (extraction,) = loader.extractions
        rows = rows_by_locator(timeline)
        assert set(rows) == set(message_locators(path).values())
        assert {row["extraction_id"] for row in rows.values()} == {extraction.id}
        assert extraction in timeline.registry.extractions
        for uid in [timeline.id, *timeline.list_children()]:
            events = (
                timeline if uid == timeline.id else timeline.get_child(uid)
            ).events.table
            assert "locator" not in events.column_names
            assert "extraction_id" not in events.column_names

    def test_the_table_carries_them_to_any_timeline(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        loader = PerformanceMidiLoader().load(path)
        (extraction,) = loader.extractions
        assert loader.events.extractions == (extraction,)
        timeline = create_timeline_from_event_data(loader.events, uid="dpt_store")
        assert timeline.registry.extractions == [extraction]
        assert set(timeline.registry.table.column("extraction_id").to_pylist()) == {
            extraction.id
        }

    def test_saved_events_keep_them(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        loader = PerformanceMidiLoader().load(path)
        (extraction,) = loader.extractions
        saved = tmp_path / "events.parquet"
        loader.to_parquet(saved)
        back = PerformanceMidiLoader.from_parquet(saved)
        assert back.extractions == [extraction]
        assert back.create_timeline().registry.extractions == [extraction]

    def test_rows_added_as_dictionaries_bring_them(self) -> None:
        loader = MatchfileLoader().load(VIENNA_DATA_DIR / "Chopin_op10_no3_p01.match")
        (extraction,) = loader.extractions
        timelines = loader.create_timelines()
        assert timelines
        for timeline in timelines:
            assert timeline.registry.extractions == [extraction]
            named = timeline.registry.table.column("extraction_id").to_pylist()
            assert set(named) == {extraction.id}

    def test_csv_data_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "events.csv"
        path.write_text("start,end\n0.5,1.0\n1.0,1.5\n2.0,2.5\n", encoding="utf-8")
        loader = CsvLoader().load(path)
        assert loader.events.table.column("locator").to_pylist() == [
            "row:0",
            "row:1",
            "row:2",
        ]
        (extraction,) = loader.extractions
        assert extraction.parser == "CsvLoader"
        assert extraction.parser_version == (
            f"timetoalign {__version__}; pandas {importlib.metadata.version('pandas')}"
        )


class TestUpdate:
    """``update`` keeps the rows it finds by locator and marks what is gone."""

    def test_found_changed_and_vanished(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        timeline = PerformanceMidiLoader().load(path).create_timeline()
        before = rows_by_locator(timeline)
        old_extraction = next(iter(before.values()))["extraction_id"]
        changed = [NOTES[0], (480, 720, 64, 99)]
        write_midi(path, changed)
        loader = PerformanceMidiLoader.from_registry(timeline.registry)
        loader.update(timeline.registry)
        (new_extraction,) = loader.extractions
        assert new_extraction.source_integrity == {
            "algorithm": "SHA-256",
            "digest": digest(path),
        }
        after = rows_by_locator(timeline)
        assert set(after) == set(before)
        assert {locator: row["id"] for locator, row in after.items()} == {
            locator: row["id"] for locator, row in before.items()
        }
        gone = message_locators(tmp_path / "a.mid")
        vanished = {
            locator for locator, row in after.items() if row["vanished_in"] is not None
        }
        assert vanished == set(before) - set(gone.values())
        assert {after[locator]["vanished_in"] for locator in vanished} == {
            new_extraction.id
        }
        assert {after[locator]["extraction_id"] for locator in gone.values()} == {
            new_extraction.id
        }
        assert {after[locator]["extraction_id"] for locator in vanished} == {
            old_extraction
        }
        notes = timeline.get_events(event_type="Note").table
        by_pitch = dict(
            zip(
                notes.column("pitch").to_pylist(),
                notes.column("velocity").to_pylist(),
                strict=True,
            )
        )
        assert by_pitch[64] == 99
        assert {e.id for e in timeline.registry.extractions} >= {
            old_extraction,
            new_extraction.id,
        }

    def test_an_added_note(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        timeline = PerformanceMidiLoader().load(path).create_timeline()
        before = rows_by_locator(timeline)
        write_midi(path, [*NOTES, (1440, 1680, 72, 90)])
        PerformanceMidiLoader.from_registry(timeline.registry).update(timeline.registry)
        after = rows_by_locator(timeline)
        (added,) = set(after) - set(before)
        assert added == message_locators(path)["Note 72"]
        assert after[added]["id"] not in {row["id"] for row in before.values()}
        assert all(after[locator]["id"] == row["id"] for locator, row in before.items())

    def test_an_unchanged_file(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        loader = PerformanceMidiLoader().load(path)
        timeline = loader.create_timeline()
        before = timeline.registry.table.to_pylist()
        loader.update(timeline.registry)
        assert timeline.registry.table.to_pylist() == before


class TestFromRegistry:
    """A loader set up from a registry runs as its extractions record."""

    def test_the_recorded_parameters(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        registry = (
            PerformanceMidiLoader(include_controls=False)
            .load(path)
            .create_timeline()
            .registry
        )
        loader = PerformanceMidiLoader.from_registry(registry)
        assert loader._seed_config()["include_controls"] is False

    def test_other_parameters_are_refused(self, tmp_path: Path) -> None:
        path = write_midi(tmp_path / "a.mid", NOTES)
        registry = (
            PerformanceMidiLoader(include_controls=False)
            .load(path)
            .create_timeline()
            .registry
        )
        with pytest.raises(ValueError, match="include_controls"):
            PerformanceMidiLoader.from_registry(registry, include_controls=True)

    def test_no_source_of_the_class(self, tmp_path: Path) -> None:
        path = tmp_path / "events.csv"
        path.write_text("start,end\n0.5,1.0\n", encoding="utf-8")
        registry = CsvLoader().load(path).create_timeline().registry
        with pytest.raises(ValueError, match="no source file PerformanceMidiLoader"):
            PerformanceMidiLoader.from_registry(registry)

    def test_nothing_to_read_again(self) -> None:
        timeline = ContinuousLogicalTimeline(length=4, unit="quarters", uid="clt_a")
        with pytest.raises(RuntimeError, match="no load\\(\\) call"):
            PerformanceMidiLoader().update(timeline.registry)
