"""Tests for MtdLoader: themes of the Musical Theme Dataset as an AlignmentBundle.

The corpus ``mtd`` holds ``MTD_sample.zip``: four themes in the layout of
the MTD archive, all sixteen files per theme, MTD's licensed files copied
and documented stand-ins for the others (the corpus README). Most tests read
it as it is; a test that needs a file MTD does not have changes a copy of
it. Every expected value is read off the corpus files by hand or chosen by
the test that changes a file; the validation logic is written down in
``tests/loader/alignment/README.md`` ("``MtdLoader`` validation logic").
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import mido
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import timetoshare

from tests.loader.minting import minted
from timetoalign import AlignmentBundle, MatchClaim
from timetoalign.core import CHUNK, NumberType, Provenance, TimeUnit
from timetoalign.core.ids import seed_from
from timetoalign.loader._sources import member_uri
from timetoalign.loader.alignment import MtdLoader
from timetoalign.loader.alignment.mtd import THEME_FILES
from timetoalign.loader.midi import ScoreMidiLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    TimelineGroup,
    UnknownLengthError,
)

#: The corpus: four themes, all sixteen files each.
SAMPLE = ensure_data("mtd") / "MTD_sample.zip"

PIANO = "Beethoven_Op002No1-01"
RELEASE_BRENDEL = "814c53fd-0732-4819-9753-4578f5ab992c"
RELEASE_ABBADO = "25f51187-906f-48fb-a16f-9e61520f2730"
TRACK_BRENDEL = (RELEASE_BRENDEL, "049", "Track01")
DOI = "https://doi.org/10.5334/tismir.68"
ZIP_URL = "https://www.audiolabs-erlangen.de/resources/MIR/MTD/MTD_1.0.0.zip"
STEM_0807 = "MTD0807_Beethoven_Op002No1-01"
STEM_1044 = "MTD1044_Beethoven_Op055-02"
ALIGNMENT_0807 = f"data_ALIGNMENT/{STEM_0807}.csv"

#: MTD0807's onsets in quarters and MIDI pitches, as its MIDI files hold them.
ONSETS_0807 = [
    Fraction(0),
    Fraction(1),
    Fraction(2),
    Fraction(3),
    Fraction(4),
    Fraction(5),
    Fraction(13, 2),
    Fraction(20, 3),
    Fraction(41, 6),
    Fraction(7),
]
PITCHES_0807 = [60, 65, 68, 72, 77, 80, 79, 77, 76, 77]

#: The same onsets in ticks, the MIDI files' own axis (480 per quarter).
TICKS_0807 = [0, 480, 960, 1440, 1920, 2400, 3120, 3200, 3280, 3360]

#: The labels of a theme's file timelines, in the order of create_timelines.
LABELS = [
    "EDM-orig MIDI",
    "EDM-orig CSV",
    "EDM-orig PDF",
    "EDM-corr MIDI",
    "EDM-corr CSV",
    "EDM-corr PDF",
    "SCORE Sibelius",
    "SCORE MusicXML",
    "SCORE MIDI",
    "SCORE CSV",
    "SCORE PDF",
    "EDM-alig MIDI",
    "EDM-alig CSV",
]

#: The files of MTD0807's timelines, by label, relative to the archive.
MEMBERS_0807 = {
    "EDM-orig MIDI": f"MTD/data_EDM-orig_MID/{STEM_0807}.mid",
    "EDM-orig CSV": f"MTD/data_EDM-orig_CSV/{STEM_0807}.csv",
    "EDM-orig PDF": f"MTD/data_EDM-orig_IMG/{STEM_0807}.pdf",
    "EDM-corr MIDI": f"MTD/data_EDM-corr_MID/{STEM_0807}.mid",
    "EDM-corr CSV": f"MTD/data_EDM-corr_CSV/{STEM_0807}.csv",
    "EDM-corr PDF": f"MTD/data_EDM-corr_IMG/{STEM_0807}.pdf",
    "SCORE Sibelius": f"MTD/data_SCORE_SIB/{STEM_0807}.sib",
    "SCORE MusicXML": f"MTD/data_SCORE_XML/{STEM_0807}.xml",
    "SCORE MIDI": f"MTD/data_SCORE_MID/{STEM_0807}.mid",
    "SCORE CSV": f"MTD/data_SCORE_CSV/{STEM_0807}.csv",
    "SCORE PDF": f"MTD/data_SCORE_IMG/{STEM_0807}.pdf",
    "EDM-alig MIDI": f"MTD/data_EDM-alig_MID/{STEM_0807}.mid",
    "EDM-alig CSV": f"MTD/data_EDM-alig_CSV/{STEM_0807}.csv",
}


# region The corpus and copies of it


def notes_of(path: Path) -> list[tuple[int, int, int]]:
    """The notes of a MIDI file as (onset, end, pitch) ticks, read with mido."""
    notes: list[tuple[int, int, int]] = []
    for track in mido.MidiFile(path).tracks:
        now = 0
        sounding: dict[int, int] = {}
        for message in track:
            now += message.time
            if message.type == "note_on" and message.velocity > 0:
                sounding[message.note] = now
            elif message.type in ("note_on", "note_off") and message.note in sounding:
                notes.append((sounding.pop(message.note), now, message.note))
    return sorted(notes)


def write_edm_shaped(path: Path, notes: list[tuple[int, int, int]]) -> None:
    """Write *notes* as MTD writes EDM-alig files, the track ending a tick after the last.

    Type 1, 480 ticks per quarter; a meta track (tempo 1,000,000 µs per
    quarter, MTD0807's 2/2 and A♭ major, ending at tick 1) and a note track
    (a program change, the notes at velocity 60 each ended by a note-on of
    velocity 0).
    """
    meta = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=1_000_000, time=0),
            mido.MetaMessage("time_signature", numerator=2, denominator=2, time=0),
            mido.MetaMessage("key_signature", key="Ab", time=0),
            mido.MetaMessage("end_of_track", time=1),
        ]
    )
    track = mido.MidiTrack([mido.Message("program_change", program=0, time=0)])
    now = 0
    for start, end, pitch in notes:
        track.append(mido.Message("note_on", note=pitch, velocity=60, time=start - now))
        track.append(mido.Message("note_on", note=pitch, velocity=0, time=end - start))
        now = end
    track.append(mido.MetaMessage("end_of_track", time=1))
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks += [meta, track]
    midi.save(path)


@pytest.fixture(scope="module")
def archive() -> Path:
    """The corpus archive, read in place."""
    return SAMPLE


@pytest.fixture(scope="module")
def complete_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The corpus archive extracted: the folder holding ``MTD/``. Read only."""
    root = tmp_path_factory.mktemp("mtd") / "extracted"
    with zipfile.ZipFile(SAMPLE) as zipped:
        zipped.extractall(root)
    return root


@pytest.fixture
def folder(complete_folder: Path, tmp_path: Path) -> Path:
    """A copy of the extracted corpus, to change one file in."""
    copy = tmp_path / "extracted"
    shutil.copytree(complete_folder, copy)
    return copy


# endregion


# region Helpers


def exact(column: pa.ChunkedArray | pa.Array) -> list[Fraction | int]:
    """The exact values of a coordinate column: a struct's, an int column's own."""
    if pa.types.is_integer(column.type):
        return column.to_pylist()
    numerators = pc.struct_field(column, "numerator").to_pylist()
    denominators = pc.struct_field(column, "denominator").to_pylist()
    return [Fraction(n, d) for n, d in zip(numerators, denominators, strict=True)]


def by_label(timelines: list[Any], theme: str = "B689 (MTD 0807)") -> dict[str, Any]:
    """A theme's file timelines by label, and its excerpt under ``excerpt``."""
    found = {
        timeline.name.removeprefix(f"{theme}, "): timeline
        for timeline in timelines
        if timeline.name.startswith(f"{theme}, ")
    }
    for track in timelines:
        for uid in track.list_children():
            child = track.get_child(uid)
            if child.name == f"Theme {theme}":
                found["excerpt"] = child
    return found


def length(timeline: Any) -> Fraction | float | None:
    """A timeline's length value, None when unknown."""
    return None if timeline.length is None else timeline.length.value


def children(timeline: Any) -> list[Any]:
    """A timeline's children."""
    return [timeline.get_child(uid) for uid in timeline.list_children()]


def claims_of(loader: MtdLoader) -> list[dict[str, Any]]:
    """The loader's claims as rows of the MatchClaim table."""
    return loader.get_field(MatchClaim).table.to_pylist()


def ungrouped_bundle(loader: MtdLoader) -> AlignmentBundle:
    """The loader's timelines and claims in a bundle without the groups."""
    bundle = AlignmentBundle(name="ungrouped")
    for timeline in loader.create_timelines():
        bundle.add_timeline(timeline)
    bundle.add_match_claim_field(loader.get_field(MatchClaim))
    return bundle


def edit(folder: Path, member: str, old: str, new: str) -> None:
    """Replace *old* by *new* in one file of the folder copy."""
    path = folder / "MTD" / member
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# endregion


# region One timeline per file


class TestFileTimelines:
    """Fourteen timelines per theme, one per file, each as long as its file states."""

    def test_fourteen_timelines_of_mtd0807(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        track, *files = loader.create_timelines()
        assert [timeline.name for timeline in files] == [
            f"B689 (MTD 0807), {label}" for label in LABELS
        ]
        assert [child.name for child in children(track)] == ["Theme B689 (MTD 0807)"]
        axes = [
            (type(timeline), timeline.unit, timeline.number_type) for timeline in files
        ]
        midi = (DiscreteLogicalTimeline, TimeUnit.ticks, NumberType.int)
        score = (ContinuousLogicalTimeline, TimeUnit.quarters, NumberType.fraction)
        csv = (ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float)
        pdf = (ContinuousGraphicalTimeline, TimeUnit.points, NumberType.float)
        assert axes[:6] == [midi, csv, pdf, midi, csv, pdf]
        assert axes[6:] == [score, score, midi, csv, pdf, midi, csv]
        assert [timeline.id.split("_")[0] for timeline in files] == [
            "dlt", "cpt", "cgt", "dlt", "cpt", "cgt",
            "clt", "clt", "dlt", "cpt", "cgt", "dlt", "cpt",
        ]  # fmt: skip

    def test_lengths_of_mtd0807(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        lengths = {label: length(t) for label, t in by_label(timelines).items()}
        assert lengths == {
            "EDM-orig MIDI": 3839,
            "EDM-orig CSV": None,
            "EDM-orig PDF": None,
            "EDM-corr MIDI": 3840,
            "EDM-corr CSV": None,
            "EDM-corr PDF": None,
            "SCORE Sibelius": None,
            "SCORE MusicXML": Fraction(9),
            "SCORE MIDI": 3840,
            "SCORE CSV": None,
            "SCORE PDF": None,
            "EDM-alig MIDI": 1369,
            "EDM-alig CSV": None,
            "excerpt": 2.8,
        }
        assert all(
            type(value) is int
            for label, value in lengths.items()
            if label.endswith("MIDI")
        )
        assert isinstance(lengths["SCORE MusicXML"], Fraction)
        midis = [label for label in lengths if label.endswith("MIDI")]
        assert [by_label(timelines)[label].is_locked for label in midis] == [True] * 4

    def test_lengths_of_every_theme(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive).create_timelines()
        assert len(timelines) == 3 + 4 * 13
        expected = {
            "B689 (MTD 0807)": (3839, 3840, 3840, 1369, Fraction(9), 2.8),
            "B690 (MTD 0808)": (7679, 7680, 7680, 2185, Fraction(17), 4.5),
            "B926 (MTD 1044)": (2399, 2400, 2400, 3817, Fraction(6), 7.9),
            "B962 (MTD 1080)": (7679, 7680, 7680, 2689, Fraction(17), 5.55),
        }
        for theme, (orig, corr, score, alig, xml, excerpt) in expected.items():
            found = by_label(timelines, theme)
            assert [
                length(found[label])
                for label in (
                    "EDM-orig MIDI",
                    "EDM-corr MIDI",
                    "SCORE MIDI",
                    "EDM-alig MIDI",
                    "SCORE MusicXML",
                    "excerpt",
                )
            ] == [orig, corr, score, alig, xml, excerpt], theme

    def test_every_midi_timeline_holds_its_notes(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        found = by_label(timelines)
        for label in ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI", "EDM-alig MIDI"):
            table = found[label].get_events().table
            assert table.column("pitch").to_pylist() == PITCHES_0807, label
        for label in ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI"):
            table = found[label].get_events().table
            assert exact(table.column("start")) == TICKS_0807, label
        alig = found["EDM-alig MIDI"].get_events().table
        assert exact(alig.column("start")) == [
            62,
            218,
            361,
            512,
            667,
            822,
            1027,
            1058,
            1098,
            1130,
        ]
        assert exact(alig.column("end"))[-1] == 1368
        for label in ("EDM-orig CSV", "SCORE MusicXML", "SCORE Sibelius", "SCORE PDF"):
            assert found[label].get_events().table.num_rows == 0

    def test_file_timelines_are_roots(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        track, *files = loader.create_timelines()
        (excerpt,) = children(track)
        assert [child.name for child in children(excerpt)] == []
        assert loader.create_bundle().timeline_ids[:14] == [
            track.id,
            *(timeline.id for timeline in files),
        ]


# endregion


# region Children through which a longer member takes part


class TestParticipation:
    """The longer of two members of known length takes part through a child."""

    def test_the_children_of_every_theme(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive).create_timelines()
        found = {
            theme: by_label(timelines, theme)
            for theme in (
                "B689 (MTD 0807)",
                "B690 (MTD 0808)",
                "B926 (MTD 1044)",
                "B962 (MTD 1080)",
            )
        }
        spans = {
            (theme, label): [
                (child.name, length(child), child.number_type)
                for child in children(timeline)
            ]
            for theme, labels in found.items()
            for label, timeline in labels.items()
            if label != "excerpt" and children(timeline)
        }
        assert spans == {
            ("B689 (MTD 0807)", "EDM-alig MIDI"): [
                (
                    "B689 (MTD 0807), EDM-alig MIDI, span of the excerpt",
                    1344,
                    NumberType.int,
                )
            ],
            ("B689 (MTD 0807)", "SCORE MusicXML"): [
                (
                    "B689 (MTD 0807), SCORE MusicXML, span of the SCORE MIDI",
                    Fraction(8),
                    NumberType.fraction,
                )
            ],
            ("B690 (MTD 0808)", "EDM-alig MIDI"): [
                (
                    "B690 (MTD 0808), EDM-alig MIDI, span of the excerpt",
                    2160,
                    NumberType.int,
                )
            ],
            ("B690 (MTD 0808)", "SCORE MusicXML"): [
                (
                    "B690 (MTD 0808), SCORE MusicXML, span of the SCORE MIDI",
                    Fraction(16),
                    NumberType.fraction,
                )
            ],
            ("B926 (MTD 1044)", "EDM-alig MIDI"): [
                (
                    "B926 (MTD 1044), EDM-alig MIDI, span of the excerpt",
                    3792,
                    NumberType.int,
                )
            ],
            ("B926 (MTD 1044)", "SCORE MusicXML"): [
                (
                    "B926 (MTD 1044), SCORE MusicXML, span of the SCORE MIDI",
                    Fraction(5),
                    NumberType.fraction,
                )
            ],
            ("B962 (MTD 1080)", "EDM-alig MIDI"): [
                (
                    "B962 (MTD 1080), EDM-alig MIDI, span of the excerpt",
                    2664,
                    NumberType.int,
                )
            ],
            ("B962 (MTD 1080)", "SCORE MusicXML"): [
                (
                    "B962 (MTD 1080), SCORE MusicXML, span of the SCORE MIDI",
                    Fraction(16),
                    NumberType.fraction,
                )
            ],
        }
        for labels in found.values():
            assert children(labels["excerpt"]) == []
            for timeline in labels.values():
                for child in children(timeline):
                    assert timeline.get_child_offset(child.id).value == 0

    def test_an_excerpt_longer_than_its_midi(self, folder: Path) -> None:
        edit(
            folder,
            f"data_META/{STEM_0807}.json",
            '"EndTime": "00:03"',
            '"EndTime": "00:04"',
        )
        timelines = MtdLoader.from_file(folder, themes=["MTD0807"]).create_timelines()
        found = by_label(timelines)
        assert length(found["excerpt"]) == 3.8
        (child,) = children(found["excerpt"])
        assert child.name == "Theme B689 (MTD 0807), span of the EDM-alig MIDI"
        assert length(child) == 2.8520833333333333
        assert child.number_type == NumberType.float
        assert found["excerpt"].get_child_offset(child.id).value == 0.0
        assert children(found["EDM-alig MIDI"]) == []

    def test_a_midi_longer_than_its_musicxml(self, folder: Path) -> None:
        path = folder / f"MTD/data_SCORE_XML/{STEM_1044}.xml"
        text = path.read_text(encoding="utf-8")
        start = text.index('<measure number="3">')
        end = text.index("</measure>", start) + len("</measure>")
        path.write_text(text[:start] + text[end:], encoding="utf-8")
        loader = MtdLoader.from_file(folder, themes=["MTD1044"])
        timelines = loader.create_timelines()
        found = by_label(timelines, "B926 (MTD 1044)")
        assert length(found["SCORE MusicXML"]) == Fraction(4)
        assert length(found["SCORE MIDI"]) == 2400
        assert children(found["SCORE MusicXML"]) == []
        (child,) = children(found["SCORE MIDI"])
        assert child.name == "B926 (MTD 1044), SCORE MIDI, span of the SCORE MusicXML"
        assert length(child) == 1920
        assert child.number_type == NumberType.int
        assert found["SCORE MIDI"].get_child_offset(child.id).value == 0
        score = loader.create_groups(timelines)[2]
        assert score.timeline_ids == [
            found["SCORE Sibelius"].id,
            found["SCORE MusicXML"].id,
            child.id,
            found["SCORE CSV"].id,
            found["SCORE PDF"].id,
        ]

    @pytest.mark.parametrize(
        ("end_time", "excerpt", "ticks"),
        [("00:02.9033", 2.7033, 1298), ("00:02.896875", 2.696875, 1294)],
        ids=["nearer-tick", "tie-to-even"],
    )
    def test_a_span_between_two_ticks(
        self, folder: Path, end_time: str, excerpt: float, ticks: int
    ) -> None:
        edit(
            folder,
            f"data_META/{STEM_0807}.json",
            '"EndTime": "00:03"',
            f'"EndTime": "{end_time}"',
        )
        found = by_label(
            MtdLoader.from_file(folder, themes=["MTD0807"]).create_timelines()
        )
        assert length(found["excerpt"]) == excerpt
        (child,) = children(found["EDM-alig MIDI"])
        assert child.name == "B689 (MTD 0807), EDM-alig MIDI, span of the excerpt"
        assert length(child) == ticks
        assert type(length(child)) is int

    def test_equal_lengths_take_part_whole(self, folder: Path) -> None:
        notes = notes_of(folder / f"MTD/data_EDM-alig_MID/{STEM_0807}.mid")
        last = notes[-1]
        notes[-1] = (last[0], 1343, last[2])
        write_edm_shaped(folder / f"MTD/data_EDM-alig_MID/{STEM_0807}.mid", notes)
        loader = MtdLoader.from_file(folder, themes=["MTD0807"])
        timelines = loader.create_timelines()
        found = by_label(timelines)
        assert length(found["EDM-alig MIDI"]) == 1344
        assert children(found["EDM-alig MIDI"]) == []
        assert children(found["excerpt"]) == []
        alig = loader.create_groups(timelines)[3]
        assert alig.timeline_ids == [
            found["excerpt"].id,
            found["EDM-alig MIDI"].id,
            found["EDM-alig CSV"].id,
        ]


# endregion


# region Groups


class TestGroups:
    """Four groups per theme, one per encoding."""

    def test_the_groups_of_mtd0807(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        timelines = loader.create_timelines()
        found = by_label(timelines)
        groups = loader.create_groups(timelines)
        assert [group.name for group in groups] == [
            "B689 (MTD 0807), EDM-orig",
            "B689 (MTD 0807), EDM-corr",
            "B689 (MTD 0807), SCORE",
            "B689 (MTD 0807), EDM-alig",
        ]
        (xml_span,) = children(found["SCORE MusicXML"])
        (alig_span,) = children(found["EDM-alig MIDI"])
        expected = [
            [found["EDM-orig MIDI"], found["EDM-orig CSV"], found["EDM-orig PDF"]],
            [found["EDM-corr MIDI"], found["EDM-corr CSV"], found["EDM-corr PDF"]],
            [
                found["SCORE Sibelius"],
                xml_span,
                found["SCORE MIDI"],
                found["SCORE CSV"],
                found["SCORE PDF"],
            ],
            [found["excerpt"], alig_span, found["EDM-alig CSV"]],
        ]
        for group, members in zip(groups, expected, strict=True):
            assert group.timeline_ids == [member.id for member in members]
            assert all(group.get_timeline(member.id) is member for member in members)

    def test_the_groups_of_every_theme(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive)
        groups = loader.create_groups()
        assert len(groups) == 16
        assert [group.name for group in groups[4:8]] == [
            "B690 (MTD 0808), EDM-orig",
            "B690 (MTD 0808), EDM-corr",
            "B690 (MTD 0808), SCORE",
            "B690 (MTD 0808), EDM-alig",
        ]
        assert [group.n_timelines for group in groups] == [3, 3, 5, 3] * 4

    def test_the_bundle_holds_the_groups(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        bundle = loader.create_bundle()
        groups = loader.create_groups()
        assert list(bundle.groups) == [group.id for group in groups]
        for group in groups:
            assert bundle.groups[group.id].timeline_ids == group.timeline_ids
            assert bundle.groups[group.id].name == group.name

    def test_positions_convert_without_a_stretch(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        found = by_label(loader.create_timelines())
        excerpt = found["excerpt"]
        (span,) = children(found["EDM-alig MIDI"])
        assert span.length.value == 1344
        assert excerpt.length.value == 2.8
        pair = TimelineGroup(id="pair", timelines=[excerpt, span])

        def ticks(seconds: float) -> int:
            return pair.get_timestamp_at(seconds, excerpt.id).get_coordinate_for(
                span.id, format="int"
            )

        def seconds(tick: int) -> float:
            return pair.get_timestamp_at(tick, span.id).get_coordinate_for(
                excerpt.id, format="float"
            )

        # Start and end meet exactly; a position between them carries the
        # rounding of the float axis, never a stretch of either member.
        assert ticks(0.0) == 0
        assert ticks(1.4) == 672
        assert ticks(2.8) == 1344
        assert ticks(1.0) == 480
        assert seconds(1344) == 2.8
        assert seconds(480) == 0.9999999999999999
        assert seconds(1120) == 2.333333333333333

    def test_a_member_of_unknown_length_stops_conversion(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        timelines = loader.create_timelines()
        found = by_label(timelines)
        alig = loader.create_groups(timelines)[3]
        with pytest.raises(UnknownLengthError, match=found["EDM-alig CSV"].id):
            alig.get_timestamp_at(1.0, found["excerpt"].id)
        bundle = loader.create_bundle()
        corr_id = claims_of(loader)[0]["timeline_a_id"]
        with pytest.raises(UnknownLengthError, match="is unknown"):
            bundle.get_matchstamp_at(3200, corr_id)


# endregion


# region Statements


def item(member: str) -> dict[str, Any]:
    """The item a timeline names: *member* of the MTD archive."""
    return {
        "type": ["melod:Item"],
        "label": member,
        "schema:isPartOf": [{"id": ZIP_URL}],
    }


class TestStatements:
    """Each timeline names its file; derivations, theme and excerpt statements."""

    def test_every_timeline_names_its_file(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        found = by_label(timelines)
        members = {**MEMBERS_0807, "excerpt": f"MTD/data_AUDIO/{STEM_0807}.wav"}
        assert {
            label: timeline.statements["lrmoo:R7i_is_exemplified_by"]
            for label, timeline in found.items()
        } == {label: [item(member)] for label, member in members.items()}
        for label, timeline in found.items():
            entries = [
                entry
                for entry in timeline.provenance
                if entry["property"] == "lrmoo:R7i_is_exemplified_by"
            ]
            assert [(e["method"], e["basis"], e["note"]) for e in entries] == [
                ("imported", [DOI], f"MTD 1.0.0 {members[label]}")
            ], label
        for timeline in found.values():
            for child in children(timeline):
                assert child.statements == {}

    def test_derivations(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        found = by_label(timelines)
        derived = {
            label: timeline.statements.get("dcterms:source")
            for label, timeline in found.items()
        }
        ids = {label: timeline.id for label, timeline in found.items()}

        def ref(label: str) -> list[dict[str, str]]:
            return [{"id": f"#{ids[label]}"}]

        assert derived == {
            "EDM-orig MIDI": None,
            "EDM-orig CSV": ref("EDM-orig MIDI"),
            "EDM-orig PDF": ref("EDM-orig MIDI"),
            "EDM-corr MIDI": ref("EDM-orig MIDI"),
            "EDM-corr CSV": ref("EDM-corr MIDI"),
            "EDM-corr PDF": ref("EDM-corr MIDI"),
            "SCORE Sibelius": None,
            "SCORE MusicXML": ref("SCORE Sibelius"),
            "SCORE MIDI": ref("SCORE Sibelius"),
            "SCORE CSV": ref("SCORE Sibelius"),
            "SCORE PDF": ref("SCORE Sibelius"),
            "EDM-alig MIDI": ref("EDM-corr MIDI"),
            "EDM-alig CSV": ref("EDM-corr MIDI"),
            "excerpt": None,
        }
        notes = {
            label: [
                (entry["method"], entry["basis"], entry["note"])
                for entry in timeline.provenance
                if entry["property"] == "dcterms:source"
            ]
            for label, timeline in found.items()
            if derived[label] is not None
        }
        export = "exported from the Sibelius file, as MTD's paper states"
        rendering = "made from the encoding's MIDI file, as MTD's paper states"
        aligned = "aligned to the excerpt by data_ALIGNMENT, as MTD's paper states"
        assert notes == {
            "EDM-orig CSV": [("imported", [DOI], rendering)],
            "EDM-orig PDF": [("imported", [DOI], rendering)],
            "EDM-corr MIDI": [
                (
                    "imported",
                    [DOI],
                    "MTD's correction of the EDM encoding, as MTD's paper states",
                )
            ],
            "EDM-corr CSV": [("imported", [DOI], rendering)],
            "EDM-corr PDF": [("imported", [DOI], rendering)],
            "SCORE MusicXML": [("imported", [DOI], export)],
            "SCORE MIDI": [("imported", [DOI], export)],
            "SCORE CSV": [("imported", [DOI], export)],
            "SCORE PDF": [("imported", [DOI], export)],
            "EDM-alig MIDI": [("imported", [DOI], aligned)],
            "EDM-alig CSV": [("imported", [DOI], aligned)],
        }

    def test_the_theme_is_stated_on_the_derivation_roots(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        found = by_label(timelines)
        theme = {
            "melod:hasIdentifier": [
                {
                    "type": ["melod:Identifier"],
                    "rdfs:label": ["MTD"],
                    "owl:hasValue": ["0807"],
                },
                {
                    "type": ["melod:Identifier"],
                    "rdfs:label": ["Barlow and Morgenstern"],
                    "owl:hasValue": ["B689"],
                },
            ],
            "schema:isPartOf": [{"label": "Beethoven Op002No1-01"}],
        }
        assert found["EDM-orig MIDI"].statements == {
            "lrmoo:R7i_is_exemplified_by": [item(MEMBERS_0807["EDM-orig MIDI"])],
            **theme,
        }
        assert found["SCORE Sibelius"].statements == {
            "lrmoo:R7i_is_exemplified_by": [item(MEMBERS_0807["SCORE Sibelius"])],
            **theme,
        }
        for label in ("EDM-orig MIDI", "SCORE Sibelius"):
            assert {
                entry["note"]
                for entry in found[label].provenance
                if entry["property"] in theme
            } == {f"MTD 1.0.0 data_META/{STEM_0807}.json"}
        stating = {
            label
            for label, timeline in found.items()
            if "schema:isPartOf" in timeline.statements
            or "melod:hasIdentifier" in timeline.statements
        }
        assert stating == {"EDM-orig MIDI", "SCORE Sibelius"}

    def test_the_excerpt_statements(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        excerpt = by_label(timelines)["excerpt"]
        assert excerpt.statements == {
            "lrmoo:R7i_is_exemplified_by": [item(f"MTD/data_AUDIO/{STEM_0807}.wav")],
            "rdfs:comment": [
                "MTD's MidiTransposition: 0 semitones, the recording against the "
                "symbolic encodings"
            ],
            "dcterms:rights": [{"id": "http://rightsstatements.org/vocab/InC/1.0/"}],
            "dcterms:accessRights": [
                {"id": "http://purl.org/coar/access_right/c_14cb"}
            ],
        }
        assert {
            (entry["method"], tuple(entry["basis"]), entry["note"])
            for entry in excerpt.provenance
            if entry["property"] != "lrmoo:R7i_is_exemplified_by"
        } == {("imported", (DOI,), f"MTD 1.0.0 data_META/{STEM_0807}.json")}

    def test_a_transposition_of_five_semitones(self, archive: Path) -> None:
        timelines = MtdLoader.from_file(archive, themes=["MTD1080"]).create_timelines()
        excerpt = by_label(timelines, "B962 (MTD 1080)")["excerpt"]
        assert excerpt.statements["rdfs:comment"] == [
            "MTD's MidiTransposition: 5 semitones, the recording against the "
            "symbolic encodings"
        ]

    def test_a_transposition_of_one_semitone(self, folder: Path) -> None:
        edit(
            folder,
            f"data_META/{STEM_0807}.json",
            '"MidiTransposition": "0"',
            '"MidiTransposition": "-1"',
        )
        timelines = MtdLoader.from_file(folder, themes=["MTD0807"]).create_timelines()
        assert by_label(timelines)["excerpt"].statements["rdfs:comment"] == [
            "MTD's MidiTransposition: -1 semitone, the recording against the "
            "symbolic encodings"
        ]


# endregion


# region The notes


class TestNotes:
    """The notes of a MIDI file, on its ticks."""

    def test_notes_of_mtd0807(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        score = by_label(loader.create_timelines())["EDM-corr MIDI"]
        assert length(score) == 3840
        table = score.get_events().table
        assert exact(table.column("start")) == TICKS_0807
        assert exact(table.column("end")) == [
            479,
            959,
            1439,
            1919,
            2399,
            3119,
            3199,
            3279,
            3359,
            3839,
        ]
        assert table.column("pitch").to_pylist() == PITCHES_0807
        assert table.column("velocity").to_pylist() == [60] * 10
        assert table.column("event_type").to_pylist() == ["Note"] * 10
        rows = claims_of(loader)
        assert [row["event_a_id"] for row in rows] == table.column("id").to_pylist()
        assert exact(
            pa.array([row["start_anchor"]["coordinate_a"] for row in rows])
        ) == exact(table.column("start"))

    def test_sixths_of_mtd1044(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD1044"])
        score = by_label(loader.create_timelines(), "B926 (MTD 1044)")["EDM-corr MIDI"]
        assert exact(score.get_events().table.column("start")) == [
            0,
            480,
            960,
            1520,
            1600,
            1680,
            1760,
            1840,
            1920,
            2040,
        ]


# endregion


class TestMidiFilesOnTicks:
    """A MIDI file's timeline is the one the MIDI loader builds, quarters a C-Map away."""

    def test_the_midi_loaders_timeline(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        found = by_label(loader.create_timelines())
        member = zipfile.Path(archive, at=MEMBERS_0807["EDM-corr MIDI"])
        alone = ScoreMidiLoader(parser="mido").load(member)
        corr = found["EDM-corr MIDI"]
        assert corr.get_events().table.column("id").to_pylist() == (
            alone.events.table.column("id").to_pylist()
        )
        rows = corr.registry.table
        assert rows.column("locator").to_pylist() == [
            f"track:1 message:{j}" for j in range(1, 20, 2)
        ]
        (parse,) = corr.registry.parses
        assert (parse.parser, parse.source) == (
            "ScoreMidiLoader",
            member_uri(member),
        )
        assert set(rows.column("parse_id").to_pylist()) == {parse.id}
        assert [e.parser for e in loader.parses] == ["ScoreMidiLoader"] * 4
        assert [e.source for e in loader.parses] == [
            member_uri(zipfile.Path(archive, at=MEMBERS_0807[label]))
            for label in (
                "EDM-orig MIDI",
                "EDM-corr MIDI",
                "SCORE MIDI",
                "EDM-alig MIDI",
            )
        ]

    def test_quarters_through_the_c_map(self, archive: Path) -> None:
        found = by_label(
            MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        )
        cmap = found["EDM-corr MIDI"].get_conversion_map("quarters")
        assert cmap.ppq == 480
        assert cmap(3840) == Fraction(8)
        assert [cmap(tick) for tick in TICKS_0807] == ONSETS_0807

    def test_the_maps_the_midi_loader_states(self, archive: Path) -> None:
        found = by_label(
            MtdLoader.from_file(archive, themes=["MTD0807"]).create_timelines()
        )
        corr, orig = found["EDM-corr MIDI"], found["EDM-orig MIDI"]
        assert corr.n_conversion_maps == 2
        assert [
            (cmap.source_unit, cmap.target_unit)
            for cmap in (
                corr.get_conversion_map("quarters"),
                corr.get_conversion_map("seconds"),
            )
        ] == [(TimeUnit.ticks, TimeUnit.quarters), (TimeUnit.ticks, TimeUnit.seconds)]
        assert corr.get_conversion_map("seconds")(3840) == 8.0
        assert orig.get_conversion_map("seconds")(3839) == 3.9989583333333334
        assert 3.9989583333333334 == float(Fraction(3839, 960))
        for label in ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI", "EDM-alig MIDI"):
            assert found[label].measure_map is not None, label


# endregion


# region The CD track and its excerpts


class TestTrackAndExcerpts:
    """The CD track's timeline and the excerpts as its children."""

    def test_two_themes_of_one_movement_are_two_children_of_one_track(
        self, archive: Path
    ) -> None:
        loader = MtdLoader.from_file(archive, movements=[PIANO])
        assert loader.tracks == [TRACK_BRENDEL]
        track = loader.create_timelines()[0]
        assert isinstance(track, ContinuousPhysicalTimeline)
        assert track.name == "Beethoven Op002No1-01, Brendel (CD 49, track 1)"
        assert track.unit == TimeUnit.seconds
        assert track.number_type == NumberType.float
        assert track.length is None
        excerpts = children(track)
        assert [child.name for child in excerpts] == [
            "Theme B689 (MTD 0807)",
            "Theme B690 (MTD 0808)",
        ]
        assert [track.get_child_offset(child.id).value for child in excerpts] == [
            0.2,
            84.7,
        ]
        assert [child.length.value for child in excerpts] == [2.8, 4.5]
        assert all(child.number_type == NumberType.float for child in excerpts)

    def test_excerpt_times_are_read_as_written(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD1080"])
        track = loader.create_timelines()[0]
        (excerpt,) = children(track)
        assert track.get_child_offset(excerpt.id).value == 60.75
        # "5.55" as written: subtracting the floats would give 5.549999999999997.
        assert excerpt.length.value == 5.55

    def test_a_track_of_two_movements_is_named_after_both(self, folder: Path) -> None:
        edit(
            folder,
            "data_META/MTD0808_Beethoven_Op002No1-01.json",
            '"WorkID": "Op002No1-01"',
            '"WorkID": "Op002No1-02"',
        )
        loader = MtdLoader.from_file(folder, themes=["MTD0807", "MTD0808"])
        assert loader.tracks == [TRACK_BRENDEL]
        assert loader.movements == [PIANO, "Beethoven_Op002No1-02"]
        track = loader.create_timelines()[0]
        assert track.name == (
            "Beethoven Op002No1-01 and Beethoven Op002No1-02, Brendel (CD 49, track 1)"
        )
        assert track.statements["schema:recordingOf"] == [
            {"label": "Beethoven Op002No1-01"},
            {"label": "Beethoven Op002No1-02"},
        ]
        assert loader.create_bundle().name == "MTD, 2 movements"

    def test_track_lengths_are_the_callers(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, movements=[PIANO])
        track = loader.create_timelines(track_lengths={TRACK_BRENDEL: 400.5})[0]
        assert track.length.value == 400.5
        with pytest.raises(KeyError, match="not read"):
            loader.create_bundle(
                track_lengths={(RELEASE_ABBADO, "007", "Track06"): 1.0}
            )

    def test_track_statements(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, movements=[PIANO])
        track = loader.create_timelines()[0]
        assert track.statements == {
            "dcterms:accessRights": [
                {"id": "http://purl.org/coar/access_right/c_14cb"}
            ],
            "dcterms:rights": [{"id": "http://rightsstatements.org/vocab/InC/1.0/"}],
            "melod:hasIdentifier": [
                {
                    "type": ["melod:Identifier"],
                    "rdfs:label": ["MTD CollectionID"],
                    "owl:hasValue": ["Composer/Beethoven_CompleteEdition_BC"],
                },
                {
                    "type": ["melod:Identifier"],
                    "rdfs:label": ["MTD NameCD"],
                    "owl:hasValue": ["049"],
                },
                {
                    "type": ["melod:Identifier"],
                    "rdfs:label": ["MTD NameTrack"],
                    "owl:hasValue": ["Track01"],
                },
            ],
            "rel:prf": [{"label": "Brendel"}],
            "schema:inAlbum": [
                {"id": f"http://musicbrainz.org/release/{RELEASE_BRENDEL}"}
            ],
            "schema:recordingOf": [{"label": "Beethoven Op002No1-01"}],
            "type": ["schema:MusicRecording"],
        }
        entries = track.provenance
        assert len(entries) == 9
        assert {entry["method"] for entry in entries} == {"imported"}
        assert {tuple(entry["basis"]) for entry in entries} == {(DOI,)}
        assert {entry["note"] for entry in entries} == {
            "MTD 1.0.0 data_META of MTD0807, MTD0808"
        }


# endregion


# region The claims


class TestClaims:
    """One claim per alignment row but the last, at the exact onset."""

    def test_claim_counts_are_rows_minus_one(self, archive: Path) -> None:
        counts = {
            mtd_id: len(MtdLoader.from_file(archive, themes=[mtd_id]))
            for mtd_id in ("MTD0807", "MTD0808", "MTD1044", "MTD1080")
        }
        assert counts == {"MTD0807": 10, "MTD0808": 16, "MTD1044": 10, "MTD1080": 28}
        assert len(MtdLoader.from_file(archive)) == 64

    def test_a_triplet_row_is_the_exact_onset(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        found = by_label(loader.create_timelines())
        score, excerpt = found["EDM-corr MIDI"], found["excerpt"]
        rows = claims_of(loader)
        assert [row["timeline_a_id"] for row in rows] == [score.id] * 10
        assert [row["timeline_b_id"] for row in rows] == [excerpt.id] * 10
        assert [row["event_b_id"] for row in rows] == [CHUNK] * 10
        assert [row["is_synchronous"] for row in rows] == [True] * 10
        ticks = [
            Fraction(
                row["start_anchor"]["coordinate_a"]["numerator"],
                row["start_anchor"]["coordinate_a"]["denominator"],
            )
            for row in rows
        ]
        seconds = [row["start_anchor"]["coordinate_b"]["value"] for row in rows]
        assert ticks == TICKS_0807
        assert {
            (
                row["start_anchor"]["coordinate_a"]["unit"],
                row["start_anchor"]["coordinate_a"]["number_type"],
            )
            for row in rows
        } == {("ticks", "int")}
        assert seconds == [
            0.12855,
            0.45519,
            0.75268,
            1.06608,
            1.38863,
            1.713,
            2.13951,
            2.20401,
            2.2885,
            2.35321,
        ]
        notes = score.get_events().table
        assert [row["event_a_id"] for row in rows] == notes.column("id").to_pylist()
        triplet = rows[7]
        note = notes.filter(pc.equal(notes.column("id"), triplet["event_a_id"]))
        assert note.column("pitch").to_pylist() == [77]

    def test_the_score_midi_when_chosen(self, archive: Path) -> None:
        loader = MtdLoader.from_file(
            archive, themes=["MTD0807"], theme_midi="data_SCORE_MID"
        )
        assert loader.theme_midi == "data_SCORE_MID"
        score = by_label(loader.create_timelines())["SCORE MIDI"]
        rows = claims_of(loader)
        assert [row["timeline_a_id"] for row in rows] == [score.id] * 10
        notes = score.get_events().table
        assert [row["event_a_id"] for row in rows] == notes.column("id").to_pylist()
        assert exact(notes.column("start")) == TICKS_0807
        assert notes.column("velocity").to_pylist() == [80] * 10

    def test_provenance(self, archive: Path) -> None:
        field = MtdLoader.from_file(archive, themes=["MTD1044"]).get_field(MatchClaim)
        claim = field.to_claims()[3]
        assert claim.provenance == Provenance(
            method="imported", basis=(DOI,), note="MTD 1.0.0 data_ALIGNMENT"
        )
        assert claim.start_anchor.coordinate_a.value == 1520

    def test_get_field_refuses_other_selectors(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        selector: Any = Provenance
        with pytest.raises(TypeError, match="MatchClaim"):
            loader.get_field(selector)


# endregion


# region Known issues of the data


class TestSecondsOutsideTheExcerpt:
    """A claim's second outside its excerpt is stored as given."""

    def test_a_negative_second(self, folder: Path) -> None:
        edit(folder, ALIGNMENT_0807, "0.12855", "-0.10000")
        loader = MtdLoader.from_file(folder, themes=["MTD0807"])
        bundle = ungrouped_bundle(loader)
        found = by_label(loader.create_timelines())
        track_id = loader.create_timelines()[0].id
        score_id, excerpt_id = found["EDM-corr MIDI"].id, found["excerpt"].id
        first = claims_of(loader)[0]
        assert first["start_anchor"]["coordinate_b"]["value"] == -0.1
        assert exact(pa.array([first["start_anchor"]["coordinate_a"]])) == [0]
        stamp = bundle.get_matchstamp_at(0, score_id)
        assert sorted(stamp.present_timelines) == sorted([score_id, excerpt_id])
        assert stamp.get_coordinate_for(excerpt_id, format="float") == -0.1
        unchanged = bundle.get_matchstamp_at(480, score_id)
        assert sorted(unchanged.present_timelines) == sorted(
            [score_id, excerpt_id, track_id]
        )

    def test_a_second_past_the_end(self, folder: Path) -> None:
        edit(folder, ALIGNMENT_0807, "2.35321", "2.90000")
        loader = MtdLoader.from_file(folder, themes=["MTD0807"])
        bundle = ungrouped_bundle(loader)
        found = by_label(loader.create_timelines())
        score_id, excerpt_id = found["EDM-corr MIDI"].id, found["excerpt"].id
        last = claims_of(loader)[-1]
        assert last["start_anchor"]["coordinate_b"]["value"] == 2.9
        stamp = bundle.get_matchstamp_at(3360, score_id)
        assert sorted(stamp.present_timelines) == sorted([score_id, excerpt_id])
        assert stamp.get_coordinate_for(excerpt_id, format="float") == 2.9


# endregion


# region Selection, sources and IDs


class TestSelection:
    """What is read is what is named."""

    def test_by_movement(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, movements=[PIANO])
        assert loader.themes == ["MTD0807", "MTD0808"]
        assert loader.movements == [PIANO]
        assert len(loader) == 26

    def test_by_theme_and_both(self, archive: Path) -> None:
        assert MtdLoader.from_file(archive, themes=["MTD1044"]).themes == ["MTD1044"]
        both = MtdLoader.from_file(archive, movements=[PIANO], themes=["MTD1080"])
        assert both.themes == ["MTD0807", "MTD0808", "MTD1080"]

    def test_everything(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive)
        assert loader.themes == ["MTD0807", "MTD0808", "MTD1044", "MTD1080"]
        assert loader.movements == [PIANO, "Beethoven_Op055-02", "Beethoven_Op067-04"]
        assert loader.tracks == [
            TRACK_BRENDEL,
            (RELEASE_ABBADO, "007", "Track06"),
            (RELEASE_ABBADO, "009", "Track04"),
        ]
        bundle = loader.create_bundle()
        assert bundle.name == "MTD, 3 movements"
        assert len(bundle.match_claim_table) == 64
        assert len(bundle.groups) == 16

    def test_unknown_names_raise(self, archive: Path) -> None:
        with pytest.raises(KeyError, match="Beethoven_Op999"):
            MtdLoader.from_file(archive, movements=["Beethoven_Op999"])
        with pytest.raises(KeyError, match="MTD9999"):
            MtdLoader.from_file(archive, themes=["MTD9999"])


class TestSources:
    """The archive and an extracted folder are one source."""

    def test_archive_and_folders_agree(
        self, archive: Path, complete_folder: Path
    ) -> None:
        def read(source: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
            loader = MtdLoader.from_file(source, movements=[PIANO])
            return loader.create_bundle().to_dict(), claims_of(loader)

        expected = read(archive)
        assert read(str(archive)) == expected
        assert read(zipfile.Path(archive)) == expected
        assert read(complete_folder) == expected
        assert read(complete_folder / "MTD") == expected

    def test_metadata_names_the_source(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD1044"])
        entry = loader.metadata["sources"][0]
        assert entry["path"] == str(archive)
        assert entry["themes"] == ["MTD1044"]
        assert entry["theme_midi"] == "data_EDM-corr_MID"


class TestIds:
    """Every ID is the minter's, seeded from the files read."""

    def test_seeded_ids_in_build_order(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"], seed=0)
        files = ["dlt", "cpt", "cgt", "dlt", "cpt", "cgt"]
        files += ["clt", "clt", "dlt", "cpt", "cgt", "dlt", "cpt"]
        maps = ["cmap", "cmap", "measure_map", "cmap", "cmap", "cmap"]
        expected = minted(
            0,
            "cpt",
            *files,
            "dlt",
            "clt",
            *["group"] * 4,
            "cpt",
            *["claim"] * 10,
            *maps * 4,
        )
        excerpt_id, file_ids = expected[0], expected[1:14]
        spans, group_ids, track_id = expected[14:16], expected[16:20], expected[20]
        claim_ids, map_ids = expected[21:31], expected[31:]
        timelines = loader.create_timelines()
        found = by_label(timelines)
        for label in ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI", "EDM-alig MIDI"):
            below_mtd = MEMBERS_0807[label].removeprefix("MTD/")
            notes = minted(seed_from(0, below_mtd), *["note"] * 10)
            timeline = found[label]
            assert timeline.get_events().table.column("id").to_pylist() == notes, label
        midi_labels = ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI", "EDM-alig MIDI")
        firsts = zip(map_ids[0::6], map_ids[1::6], map_ids[2::6], strict=True)
        for label, (quarters, seconds, measures) in zip(
            midi_labels, firsts, strict=True
        ):
            timeline = found[label]
            assert timeline.n_conversion_maps == 2, label
            assert timeline.get_conversion_map("quarters").id == quarters, label
            assert timeline.get_conversion_map("seconds").id == seconds, label
            assert timeline.measure_map.id == measures, label
        assert found["excerpt"].id == excerpt_id
        assert [found[label].id for label in LABELS] == file_ids
        assert [
            found["EDM-alig MIDI"].list_children()[0],
            found["SCORE MusicXML"].list_children()[0],
        ] == spans
        assert [group.id for group in loader.create_groups(timelines)] == group_ids
        assert timelines[0].id == track_id
        assert [row["id"] for row in claims_of(loader)] == claim_ids

    def test_seeded_map_ids_in_the_bundle(self, archive: Path) -> None:
        bundle = MtdLoader.from_file(
            archive, themes=["MTD0807"], seed=0
        ).create_bundle()
        maps = ["cmap", "cmap", "measure_map", "cmap", "cmap", "cmap"]
        expected = minted(0, *["x"] * 31, "bundle", *maps * 4)
        assert bundle.id == expected[31]
        written = json.dumps(bundle.to_dict())
        held = re.findall(r'"#((?:cmap|measure_map)_[0-9a-z]{13})"', written)
        assert sorted(set(held)) == sorted(expected[32:])

    def test_equal_midi_files_keep_their_maps_apart(self, folder: Path) -> None:
        mtd = folder / "MTD"
        shutil.copyfile(
            mtd / f"data_EDM-orig_MID/{STEM_0807}.mid",
            mtd / "data_EDM-orig_MID/MTD0808_Beethoven_Op002No1-01.mid",
        )
        bundle = MtdLoader.from_file(folder, movements=[PIANO]).create_bundle()
        roots = bundle.get_timelines(bundle.timeline_ids)
        first, second = (
            by_label(roots, theme)["EDM-orig MIDI"]
            for theme in ("B689 (MTD 0807)", "B690 (MTD 0808)")
        )
        assert [length(first), length(second)] == [3839, 3839]
        ids = [
            {
                timeline.get_conversion_map("quarters").id,
                timeline.get_conversion_map("seconds").id,
                timeline.measure_map.id,
            }
            for timeline in (first, second)
        ]
        assert [len(ids[0]), len(ids[1]), len(ids[0] & ids[1])] == [3, 3, 0]

    def test_seeded_ids_do_not_follow_the_folder(
        self, archive: Path, complete_folder: Path
    ) -> None:
        def note_ids(source: Path) -> dict[str, list[str]]:
            loader = MtdLoader.from_file(source, themes=["MTD0807"], seed=0)
            found = by_label(loader.create_timelines())
            return {
                label: found[label].get_events().table.column("id").to_pylist()
                for label in ("EDM-orig MIDI", "EDM-corr MIDI", "SCORE MIDI")
            }

        assert note_ids(complete_folder) == note_ids(archive)

    def test_fresh_loaders_mint_equal_ids(self, archive: Path) -> None:
        first = MtdLoader.from_file(archive, movements=[PIANO])
        second = MtdLoader.from_file(archive, movements=[PIANO])
        assert [t.id for t in first.create_timelines()] == [
            t.id for t in second.create_timelines()
        ]
        assert claims_of(first) == claims_of(second)
        assert first.create_bundle().id == second.create_bundle().id


# endregion


# region One timeline by its ID


class TestCreateTimeline:
    """``create_timeline`` gives one timeline of ``create_timelines`` by its ID."""

    def test_the_track_and_a_file(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        track, *files = loader.create_timelines()
        assert loader.create_timeline(track.id).to_dict() == track.to_dict()
        for timeline in files:
            again = loader.create_timeline(timeline.id)
            assert (again.id, again.name, again.length, again.statements) == (
                timeline.id,
                timeline.name,
                timeline.length,
                timeline.statements,
            )
        sized = loader.create_timeline(track.id, track_lengths={TRACK_BRENDEL: 400.5})
        assert sized.length.value == 400.5

    def test_an_unknown_id(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        with pytest.raises(KeyError, match="cpt_unknown"):
            loader.create_timeline("cpt_unknown")

    def test_a_derived_timeline_alone_is_not_written(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        found = by_label(loader.create_timelines())
        midi, csv = found["EDM-corr MIDI"], found["EDM-corr CSV"]
        for alone in (csv, loader.create_timeline(csv.id)):
            with pytest.raises(timetoshare.SchemaValidationError, match=f"#{midi.id}"):
                alone.to_dict()
        bundle = loader.create_bundle()
        written = bundle.get_timeline(csv.id).to_dict()
        (entry,) = written["timelines"]
        assert entry["statements"] == {
            "lrmoo:R7i_is_exemplified_by": [item(MEMBERS_0807["EDM-corr CSV"])]
        }

    def test_a_group_with_a_derived_member_alone_is_not_written(
        self, archive: Path
    ) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        found = by_label(loader.create_timelines())
        orig, corr, score, alig = loader.create_groups()
        orig.to_dict()
        score.to_dict()
        with pytest.raises(
            timetoshare.SchemaValidationError, match=f"#{found['EDM-orig MIDI'].id}"
        ):
            corr.to_dict()
        with pytest.raises(
            timetoshare.SchemaValidationError, match=f"#{found['EDM-corr MIDI'].id}"
        ):
            alig.to_dict()


# endregion


# region The bundle and its package


class TestBundle:
    """The bundle, written as a package and read back."""

    def test_bundle_of_one_movement(self, archive: Path) -> None:
        bundle = MtdLoader.from_file(archive, movements=[PIANO]).create_bundle()
        assert isinstance(bundle, AlignmentBundle)
        assert bundle.name == "Beethoven Op002No1-01"
        assert len(bundle.groups) == 8
        assert len(bundle.match_claim_table) == 26

    def test_package_round_trip(self, archive: Path, tmp_path: Path) -> None:
        loader = MtdLoader.from_file(archive)
        bundle = loader.create_bundle()
        manifest = tmp_path / "op002no1_01.tts.json"
        bundle.to_file(manifest, serialization="json", events=False)
        timetoshare.validate(json.loads(manifest.read_text(encoding="utf-8")))
        assert (tmp_path / "match_claims.parquet").is_file()
        read = AlignmentBundle.from_file(manifest)
        assert len(read.match_claim_table) == 64
        assert read.match_claim_table.table.to_pylist() == claims_of(loader)
        timelines = loader.create_timelines()
        for timeline in timelines:
            for original in (timeline, *children(timeline)):
                again = read.get_timeline(original.id)
                assert again.statements == original.statements, original.name
                assert again.length == original.length, original.name
                assert again.name == original.name
        for group in loader.create_groups(timelines):
            assert read.groups[group.id].timeline_ids == group.timeline_ids
            assert read.groups[group.id].name == group.name


# endregion


# region Refusals


class TestRefusals:
    """Each refusal comes from one change to an otherwise valid source."""

    def test_a_row_off_its_onset(self, folder: Path) -> None:
        edit(folder, ALIGNMENT_0807, "6.66667", "6.70000")
        with pytest.raises(ValueError, match="6.70000;2.20401 .* meets 0 onsets"):
            MtdLoader.from_file(folder, themes=["MTD0807"])

    def test_a_row_half_way_between_two_onsets(self, folder: Path) -> None:
        midi = folder / f"MTD/data_EDM-corr_MID/{STEM_0807}.mid"
        (_, end, pitch), *rest = notes_of(midi)
        # The first note shortened to one tick, two more at ticks 1 and 2.
        write_edm_shaped(midi, [(0, 1, pitch), (1, 2, 61), (2, end, 62), *rest])
        edit(
            folder,
            ALIGNMENT_0807,
            "0.00000;0.12855\n",
            "0.00000;0.12855\n0.003125;0.13000\n",
        )
        with pytest.raises(
            ValueError,
            match=r"0\.003125;0\.13000 .* meets 2 onsets .*\['1/480', '1/240'\]",
        ):
            MtdLoader.from_file(folder, themes=["MTD0807"])

    def test_a_last_row_off_the_end(self, folder: Path) -> None:
        edit(folder, ALIGNMENT_0807, "7.99792", "8.00000")
        with pytest.raises(ValueError, match="last alignment row 8.00000"):
            MtdLoader.from_file(folder, themes=["MTD0807"])

    def test_a_time_not_written_mm_ss(self, folder: Path) -> None:
        edit(folder, "data_META/MTD1044_Beethoven_Op055-02.json", '"04:07.1"', '"4.07"')
        with pytest.raises(ValueError, match="StartTime '4.07'"):
            MtdLoader.from_file(folder, themes=["MTD1044"])

    def test_an_end_before_the_start(self, folder: Path) -> None:
        edit(folder, "data_META/MTD1044_Beethoven_Op055-02.json", '"04:15"', '"04:00"')
        with pytest.raises(ValueError, match="ends"):
            MtdLoader.from_file(folder, themes=["MTD1044"])

    def test_a_missing_midi_file(self, folder: Path) -> None:
        (folder / "MTD/data_EDM-corr_MID/MTD1080_Beethoven_Op067-04.mid").unlink()
        with pytest.raises(FileNotFoundError, match="MTD1080"):
            MtdLoader.from_file(folder, themes=["MTD1080"])
        assert MtdLoader.from_file(folder, themes=["MTD1044"]).themes == ["MTD1044"]

    def test_files_named_but_not_read_are_required(self, folder: Path) -> None:
        (folder / f"MTD/data_SCORE_IMG/{STEM_0807}.pdf").unlink()
        (folder / f"MTD/data_AUDIO/{STEM_0807}.wav").unlink()
        with pytest.raises(FileNotFoundError) as missing:
            MtdLoader.from_file(folder, themes=["MTD0807"])
        assert f"data_SCORE_IMG/{STEM_0807}.pdf" in str(missing.value)
        assert f"data_AUDIO/{STEM_0807}.wav" in str(missing.value)
        assert MtdLoader.from_file(folder, themes=["MTD1044"]).themes == ["MTD1044"]

    def test_no_encoding_folder(self, archive: Path) -> None:
        with pytest.raises(ValueError, match="data_EDM-orig_MID"):
            MtdLoader.from_file(archive, theme_midi="data_EDM-orig_MID")

    def test_no_mtd_folder(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data_META"):
            MtdLoader.from_file(tmp_path)

    def test_a_second_source(self, archive: Path, folder: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        with pytest.raises(ValueError, match="reads one source"):
            loader.load(folder, themes=["MTD1044"])
        assert loader.themes == ["MTD0807"]
        loader.clear()
        assert loader.load(folder, themes=["MTD1044"]).themes == ["MTD1044"]

    def test_one_source(self, archive: Path) -> None:
        with pytest.raises(ValueError, match="got none"):
            MtdLoader().load()
        with pytest.raises(ValueError, match="got 2"):
            MtdLoader().load(archive, archive)

    def test_the_refusal_names_the_encoding_read(self, folder: Path) -> None:
        edit(folder, ALIGNMENT_0807, "6.66667", "6.70000")
        reading = (
            "The alignment counts the quarters of data_EDM-corr_MID; this theme "
            "encoding was read from {}."
        )
        with pytest.raises(ValueError) as default:
            MtdLoader.from_file(folder, themes=["MTD0807"])
        assert str(default.value).endswith(reading.format("data_EDM-corr_MID"))
        with pytest.raises(ValueError) as score:
            MtdLoader.from_file(folder, themes=["MTD0807"], theme_midi="data_SCORE_MID")
        assert f"data_SCORE_MID/{STEM_0807}.mid" in str(score.value)
        assert str(score.value).endswith(reading.format("data_SCORE_MID"))

    def test_an_unknown_keyword_of_create_bundle(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        with pytest.raises(TypeError, match="lengths"):
            loader.create_bundle(lengths={})

    def test_an_unknown_keyword_of_create_timeline(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        with pytest.raises(TypeError, match="lenghts"):
            loader.create_timeline(loader.create_timelines()[0].id, lenghts={})

    def test_groups_of_timelines_lacking_a_member(self, archive: Path) -> None:
        loader = MtdLoader.from_file(archive, themes=["MTD0807"])
        track, *files = loader.create_timelines()
        with pytest.raises(KeyError, match=track.id):
            loader.create_groups(files)

    def test_nothing_loaded(self) -> None:
        with pytest.raises(RuntimeError, match="load"):
            MtdLoader().create_bundle()
        with pytest.raises(RuntimeError, match="load"):
            MtdLoader().create_groups()


# endregion


# region Representation


def test_html_card_counts_claims(archive: Path) -> None:
    html = MtdLoader.from_file(archive, movements=[PIANO])._repr_html_()
    assert "<tr><td><b>Claims</b></td><td>26</td></tr>" in html
    assert "<tr><td><b>Themes</b></td><td>2</td></tr>" in html
    assert "Events" not in html
    assert repr(MtdLoader.from_file(archive, movements=[PIANO])) == (
        "MtdLoader(movements=1, themes=2, claims=26)"
    )


def test_theme_files_are_the_labels() -> None:
    assert [file.label for file in THEME_FILES] == LABELS
