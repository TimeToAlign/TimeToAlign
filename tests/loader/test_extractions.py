"""Extractions and locators: what a loader read, where each event lives in it.

The validation logic is written down in ``tests/loader/README.md``
("Extractions and locators"); every expected value is read off the file
with mido or ``hashlib`` by the test, never by the loader.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path

import mido
import pyarrow.compute as pc
import pytest

from timetoalign import AlignmentBundle, __version__
from timetoalign.loader.alignment.matchfile import MatchfileLoader
from timetoalign.loader.alignment.tilia import TiliaJsonLoader
from timetoalign.loader.base import locate, locators
from timetoalign.loader.midi import PerformanceMidiLoader
from timetoalign.loader.score import MeasureMapLoader, Music21Loader, PartituraLoader
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


class TestLocatorText:
    """One text for every loader: the pointer, then the key the source states."""

    def test_locate(self) -> None:
        assert locate("row:3") == "row:3"
        assert locate("row:3", "n12") == "row:3 key:n12"
        assert locate("p", "a key:b") == "p key:a key:b"
        assert locate("p", "") == "p"
        assert locate("p", None) == "p"
        assert locate("p", float("nan")) == "p"
        assert locate("p", 12) == "p key:12"

    def test_locators_row_for_row(self) -> None:
        assert locators(["a", "b", "c"], ["x", None, ""]).to_pylist() == [
            "a key:x",
            "b",
            "c",
        ]
        assert locators(["a", "b"], [12, None]).to_pylist() == ["a key:12", "b"]

    def test_a_csv_s_id_column(self, tmp_path: Path) -> None:
        path = tmp_path / "events.csv"
        path.write_text(
            "id,start,end\nn1,0.5,1.0\nn7,1.0,1.5\nn2,2.0,2.5\n", encoding="utf-8"
        )
        table = CsvLoader().load(path).events.table
        assert table.column("locator").to_pylist() == [
            "row:0 key:n1",
            "row:1 key:n7",
            "row:2 key:n2",
        ]


SCORE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>1</divisions>
        <time><beats>2</beats><beat-type>4</beat-type></time></attributes>
      <note id="n1"><pitch><step>C</step><octave>4</octave></pitch>
        <duration>2</duration><type>half</type></note>
    </measure>
    <measure number="2a">
      <note><pitch><step>D</step><octave>4</octave></pitch>
        <duration>2</duration><type>half</type></note>
    </measure>
  </part>
</score-partwise>
"""


class TestLocatorsOfTheLoaders:
    """Each loader's locators, written out from the file the test writes."""

    @pytest.fixture
    def score(self, tmp_path: Path) -> Path:
        path = tmp_path / "score.musicxml"
        path.write_text(SCORE_XML, encoding="utf-8")
        return path

    def test_partitura(self, score: Path) -> None:
        import partitura

        part = partitura.load_score(str(score)).parts[0]
        at = {id(obj): k for k, obj in enumerate(part.iter_all())}
        measures = [f"part:0 element:{at[id(m)]}" for m in part.measures]
        notes = [f"part:0 element:{at[id(n)]}" for n in part.notes]
        store = PartituraLoader().load(score).store
        assert store.measures.table.column("locator").to_pylist() == [
            f"{measures[0]} key:1",
            f"{measures[1]} key:2a",
        ]
        assert store.notes.table.column("locator").to_pylist() == [
            f"{notes[0]} key:n1",
            notes[1],
        ]

    def test_music21(self, score: Path) -> None:
        import music21

        flat = music21.converter.parse(str(score)).parts[0].flatten()
        notes = [
            f"part:0 element:{k}"
            for k, obj in enumerate(flat)
            if isinstance(obj, music21.note.Note)
        ]
        store = Music21Loader().load(score).store
        assert store.measures.table.column("locator").to_pylist() == [
            "part:0 measure:0 key:1",
            "part:0 measure:1 key:2",
        ]
        assert store.notes.table.column("locator").to_pylist() == notes

    def test_measure_map(self, tmp_path: Path) -> None:
        rows = [
            {
                "ID": "m1",
                "count": 1,
                "qstamp": 0,
                "number": 1,
                "time_signature": "2/4",
                "nominal_length": 2,
                "actual_length": 2,
                "next": [2],
            },
            {"count": 2},
            {"count": 3},
        ]
        path = tmp_path / "piece.mm.json"
        path.write_text(json.dumps(rows), encoding="utf-8")
        loader = MeasureMapLoader().load(path)
        table = loader.store.measures.table
        (extraction,) = loader.extractions
        assert table.column("locator").to_pylist() == [
            "measure:1 key:m1",
            "measure:2",
            "measure:3",
        ]
        assert table.column("extraction_id").to_pylist() == [extraction.id] * 3

    @pytest.mark.parametrize(
        ("document", "expected"),
        [
            (
                {
                    "timelines": {
                        "t7": {
                            "kind": "MARKER_TIMELINE",
                            "components": {"c1": {"time": 1.0}, "c2": {"time": 2.0}},
                        }
                    },
                    "media_metadata": {"media length": 10},
                },
                [
                    "/timelines/t7/components/c1 key:c1",
                    "/timelines/t7/components/c2 key:c2",
                ],
            ),
            (
                {
                    "timelines": [
                        {
                            "kind": "MARKER_TIMELINE",
                            "components": [{"id": "x", "time": 1.0}, {"time": 2.0}],
                        }
                    ],
                    "media_metadata": {"media length": 10},
                },
                ["/timelines/0/components/0 key:x", "/timelines/0/components/1"],
            ),
        ],
        ids=["v2", "v1"],
    )
    def test_tilia(
        self, tmp_path: Path, document: dict[str, object], expected: list[str]
    ) -> None:
        path = tmp_path / "markers.tla"
        path.write_text(json.dumps(document), encoding="utf-8")
        (timeline,) = TiliaJsonLoader().load(path).create_timelines()
        assert timeline.registry.table.column("locator").to_pylist() == expected


MATCH_FILE = VIENNA_DATA_DIR / "Chopin_op10_no3_p01.match"


def statements(path: Path, kind: str) -> list[str]:
    """The IDs of a match file's ``snote(`` or ``note(`` statements, in file order."""
    pattern = re.compile(rf"(?:^|-){kind}\(([^,)]+)")
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match is not None:
            found.append(match.group(1))
    return found


def id_locator_pairs(bundle: AlignmentBundle) -> dict[str, list[tuple[str, str]]]:
    """Per timeline of *bundle*, its registry's (id, locator) pairs of its events."""
    pairs = {}
    for timeline in bundle.get_timelines(bundle.timeline_ids):
        table = timeline.registry.table
        table = table.filter(pc.is_valid(table.column("locator")))
        pairs[timeline.id] = sorted(
            zip(
                table.column("id").to_pylist(),
                table.column("locator").to_pylist(),
                strict=True,
            )
        )
    return pairs


class TestMatchfileLocatorsAndIds:
    """Matchfile notes are located by their statement; a repeated ID is minted."""

    @pytest.fixture(scope="class")
    def bundle(self) -> AlignmentBundle:
        return MatchfileLoader().load(MATCH_FILE).create_bundle()

    def test_the_locators(self, bundle: AlignmentBundle) -> None:
        snotes, notes = statements(MATCH_FILE, "snote"), statements(MATCH_FILE, "note")
        score, performance = bundle.get_timelines(bundle.timeline_ids)
        by_id = dict(
            zip(
                score.registry.table.column("id").to_pylist(),
                score.registry.table.column("locator").to_pylist(),
                strict=True,
            )
        )
        assert by_id == {sid: f"snote:{k} key:{sid}" for k, sid in enumerate(snotes)}
        assert sorted(
            performance.registry.table.column("locator").to_pylist()
        ) == sorted(f"note:{k} key:{nid}" for k, nid in enumerate(notes))
        assert (len(snotes), len(notes)) == (454, 451)

    def test_a_performed_id_held_by_a_score_note_is_minted(
        self, bundle: AlignmentBundle
    ) -> None:
        score, performance = bundle.get_timelines(bundle.timeline_ids)
        by_locator = dict(
            zip(
                performance.registry.table.column("locator").to_pylist(),
                performance.registry.table.column("id").to_pylist(),
                strict=True,
            )
        )
        score_ids = set(score.registry.table.column("id").to_pylist())
        assert by_locator["note:0 key:n0"] == "n0"
        assert by_locator["note:1 key:n2"].startswith("event_")
        assert not set(by_locator.values()) & score_ids
        claim = next(
            claim for claim in bundle.get_match_claims() if claim.event_a_id == "n2"
        )
        assert claim.event_b_id == by_locator["note:1 key:n2"]

    def test_a_second_performance_mints_what_the_first_holds(self) -> None:
        second = VIENNA_DATA_DIR / "Chopin_op10_no3_p02.match"
        loader = MatchfileLoader().load(MATCH_FILE, second)
        score, first, again = loader.create_timelines()
        ids = [set(t.registry.table.column("id").to_pylist()) for t in (score, first)]
        held = again.registry.table
        stated = [
            locator.split(" key:", 1)[1]
            for locator in held.column("locator").to_pylist()
        ]
        for stated_id, event_id in zip(
            stated, held.column("id").to_pylist(), strict=True
        ):
            if stated_id in ids[0] | ids[1]:
                assert event_id.startswith("event_"), stated_id
            else:
                assert event_id == stated_id
        assert "n0" in ids[1]
        assert not (ids[0] | ids[1]) & set(held.column("id").to_pylist())

    def test_the_file_s_own_performed_id(self, tmp_path: Path) -> None:
        text = MATCH_FILE.read_text(encoding="utf-8")
        assert text.count("-note(n0,") == 1
        path = tmp_path / "renamed.match"
        path.write_text(text.replace("-note(n0,", "-note(p0,"), encoding="utf-8")
        bundle = MatchfileLoader().load(path).create_bundle()
        _, performance = bundle.get_timelines(bundle.timeline_ids)
        rows = performance.registry.table
        first = dict(
            zip(rows.column("locator").to_pylist(), rows.column("id").to_pylist())
        )
        assert first["note:0 key:p0"] == "p0"
        (claim,) = [c for c in bundle.get_match_claims() if c.event_a_id == "n1"]
        assert claim.event_b_id == "p0"

    def test_the_bundle_writes_and_reads_back(
        self, bundle: AlignmentBundle, tmp_path: Path
    ) -> None:
        path = bundle.to_file(tmp_path / "match" / "manifest.yaml", events=True)
        read = AlignmentBundle.from_file(path)
        assert id_locator_pairs(read) == id_locator_pairs(bundle)


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
