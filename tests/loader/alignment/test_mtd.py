"""Tests for MtdLoader: themes of the Musical Theme Dataset as an AlignmentBundle.

The corpus ``mtd`` holds ``MTD_sample.zip``: four themes in the layout of
the MTD archive. Every expected value is read off its files by hand; the
validation logic is written down in ``tests/loader/alignment/README.md``
("``MtdLoader`` validation logic").
"""

from __future__ import annotations

import json
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pytest
import timetoshare

from tests.loader.minting import minted
from timetoalign import AlignmentBundle, MatchClaim
from timetoalign.core import CHUNK, NumberType, Provenance, TimeUnit
from timetoalign.loader.alignment import MtdLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import ContinuousLogicalTimeline, ContinuousPhysicalTimeline

#: The sample archive.
ARCHIVE = ensure_data("mtd") / "MTD_sample.zip"

PIANO = "Beethoven_Op002No1-01"
RELEASE_BRENDEL = "814c53fd-0732-4819-9753-4578f5ab992c"
RELEASE_ABBADO = "25f51187-906f-48fb-a16f-9e61520f2730"
TRACK_BRENDEL = (RELEASE_BRENDEL, "049", "Track01")
DOI = "https://doi.org/10.5334/tismir.68"


# region Helpers


def exact(column: pa.ChunkedArray | pa.Array) -> list[Fraction]:
    """The exact values of a coordinate struct column."""
    numerators = pc.struct_field(column, "numerator").to_pylist()
    denominators = pc.struct_field(column, "denominator").to_pylist()
    return [Fraction(n, d) for n, d in zip(numerators, denominators, strict=True)]


def by_name(timelines: list[Any], name: str) -> Any:
    """The timeline named *name*."""
    return next(timeline for timeline in timelines if timeline.name == name)


def claims_of(loader: MtdLoader) -> list[dict[str, Any]]:
    """The loader's claims as rows of the MatchClaim table."""
    return loader.get_field(MatchClaim).table.to_pylist()


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """The sample extracted to ``tmp_path``: the folder holding ``MTD/``."""
    with zipfile.ZipFile(ARCHIVE) as archive:
        archive.extractall(tmp_path / "extracted")
    return tmp_path / "extracted"


def edit(folder: Path, member: str, old: str, new: str) -> None:
    """Replace *old* by *new* in one file of the extracted sample."""
    path = folder / "MTD" / member
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# endregion


# region The theme encoding


class TestThemeEncoding:
    """The notes of the theme encoding, in exact quarters."""

    def test_notes_of_mtd0807(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        score = by_name(loader.create_timelines(), "B689 (MTD 0807)")
        assert isinstance(score, ContinuousLogicalTimeline)
        assert score.unit == TimeUnit.quarters
        assert score.number_type == NumberType.fraction
        table = score.get_events().table
        assert exact(table.column("start")) == [
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
        ends = exact(table.column("end"))
        assert ends[0] == Fraction(479, 480)
        assert ends[-1] == Fraction(3839, 480)
        assert pc.struct_field(table.column("pitch"), "midi_number").to_pylist() == [
            60,
            65,
            68,
            72,
            77,
            80,
            79,
            77,
            76,
            77,
        ]
        assert table.column("velocity").to_pylist() == [80] * 10
        assert table.column("event_type").to_pylist() == ["Note"] * 10
        assert Fraction(score.length.value) == Fraction(3839, 480)

    def test_sixths_of_mtd1044(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD1044"])
        score = by_name(loader.create_timelines(), "B926 (MTD 1044)")
        assert exact(score.get_events().table.column("start")) == [
            Fraction(0),
            Fraction(1),
            Fraction(2),
            Fraction(19, 6),
            Fraction(10, 3),
            Fraction(7, 2),
            Fraction(11, 3),
            Fraction(23, 6),
            Fraction(4),
            Fraction(17, 4),
        ]

    def test_the_theme_is_a_root_not_a_child_of_the_excerpt(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        track, score = loader.create_timelines()
        excerpt = track.get_child(track.list_children()[0])
        assert excerpt.list_children() == []
        assert loader.create_bundle().timeline_ids == [track.id, score.id]


# endregion


# region The CD track and its excerpts


class TestTrackAndExcerpts:
    """The CD track's timeline and the excerpts as its children."""

    def test_two_themes_of_one_movement_are_two_children_of_one_track(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
        assert loader.tracks == [TRACK_BRENDEL]
        track = loader.create_timelines()[0]
        assert isinstance(track, ContinuousPhysicalTimeline)
        assert track.name == "Beethoven Op002No1-01, Brendel (CD 49, track 1)"
        assert track.unit == TimeUnit.seconds
        assert track.number_type == NumberType.float
        assert track.length is None
        children = [track.get_child(uid) for uid in track.list_children()]
        assert [child.name for child in children] == [
            "Theme B689 (MTD 0807)",
            "Theme B690 (MTD 0808)",
        ]
        assert [track.get_child_offset(child.id).value for child in children] == [
            0.2,
            84.7,
        ]
        assert [child.length.value for child in children] == [2.8, 4.5]
        assert all(child.number_type == NumberType.float for child in children)

    def test_excerpt_times_are_read_as_written(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD1080"])
        track = loader.create_timelines()[0]
        excerpt = track.get_child(track.list_children()[0])
        assert track.get_child_offset(excerpt.id).value == 60.75
        assert excerpt.length.value == 5.55
        assert 66.3 - 60.75 != 5.55

    def test_track_lengths_are_the_callers(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
        track = loader.create_timelines(track_lengths={TRACK_BRENDEL: 400.5})[0]
        assert track.length.value == 400.5
        with pytest.raises(KeyError, match="not read"):
            loader.create_bundle(
                track_lengths={(RELEASE_ABBADO, "007", "Track06"): 1.0}
            )

    def test_track_statements(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
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

    def test_theme_statements(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        score = loader.create_timelines()[1]
        assert score.statements == {
            "dcterms:source": [
                {
                    "label": "MTD's corrected EDM encoding of the theme, MTD 1.0.0 "
                    "data_EDM-corr_MID/MTD0807_Beethoven_Op002No1-01.mid"
                }
            ],
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

    def test_the_score_encoding_is_stated_when_chosen(self) -> None:
        loader = MtdLoader.from_file(
            ARCHIVE, themes=["MTD0807"], theme_midi="data_SCORE_MID"
        )
        assert loader.theme_midi == "data_SCORE_MID"
        score = loader.create_timelines()[1]
        assert score.statements["dcterms:source"] == [
            {
                "label": "MTD's encoding of the engraved theme, MTD 1.0.0 "
                "data_SCORE_MID/MTD0807_Beethoven_Op002No1-01.mid"
            }
        ]


# endregion


# region The claims


class TestClaims:
    """One claim per alignment row but the last, at the exact onset."""

    def test_claim_counts_are_rows_minus_one(self) -> None:
        counts = {
            mtd_id: len(MtdLoader.from_file(ARCHIVE, themes=[mtd_id]))
            for mtd_id in ("MTD0807", "MTD0808", "MTD1044", "MTD1080")
        }
        assert counts == {"MTD0807": 10, "MTD0808": 16, "MTD1044": 10, "MTD1080": 28}
        assert len(MtdLoader.from_file(ARCHIVE)) == 64

    def test_a_triplet_row_is_the_exact_onset(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        track, score = loader.create_timelines()
        excerpt_id = track.list_children()[0]
        rows = claims_of(loader)
        assert [row["timeline_a_id"] for row in rows] == [score.id] * 10
        assert [row["timeline_b_id"] for row in rows] == [excerpt_id] * 10
        assert [row["event_b_id"] for row in rows] == [CHUNK] * 10
        assert [row["is_synchronous"] for row in rows] == [True] * 10
        quarters = [
            Fraction(
                row["start_anchor"]["coordinate_a"]["numerator"],
                row["start_anchor"]["coordinate_a"]["denominator"],
            )
            for row in rows
        ]
        seconds = [row["start_anchor"]["coordinate_b"]["value"] for row in rows]
        assert quarters == [
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
        assert pc.struct_field(note.column("pitch"), "midi_number").to_pylist() == [77]

    def test_provenance(self) -> None:
        field = MtdLoader.from_file(ARCHIVE, themes=["MTD1044"]).get_field(MatchClaim)
        claim = field.to_claims()[3]
        assert claim.provenance == Provenance(
            method="imported", basis=(DOI,), note="MTD 1.0.0 data_ALIGNMENT"
        )
        assert Fraction(claim.start_anchor.coordinate_a.value) == Fraction(19, 6)

    def test_get_field_refuses_other_selectors(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        selector: Any = Provenance
        with pytest.raises(TypeError, match="MatchClaim"):
            loader.get_field(selector)


# endregion


# region Selection, sources and IDs


class TestSelection:
    """What is read is what is named."""

    def test_by_movement(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
        assert loader.themes == ["MTD0807", "MTD0808"]
        assert loader.movements == [PIANO]
        assert len(loader) == 26

    def test_by_theme_and_both(self) -> None:
        assert MtdLoader.from_file(ARCHIVE, themes=["MTD1044"]).themes == ["MTD1044"]
        both = MtdLoader.from_file(ARCHIVE, movements=[PIANO], themes=["MTD1080"])
        assert both.themes == ["MTD0807", "MTD0808", "MTD1080"]

    def test_everything(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE)
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

    def test_unknown_names_raise(self) -> None:
        with pytest.raises(KeyError, match="Beethoven_Op999"):
            MtdLoader.from_file(ARCHIVE, movements=["Beethoven_Op999"])
        with pytest.raises(KeyError, match="MTD9999"):
            MtdLoader.from_file(ARCHIVE, themes=["MTD9999"])


class TestSources:
    """The archive and an extracted folder are one source."""

    def test_archive_and_folders_agree(self, folder: Path) -> None:
        def read(source: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            loader = MtdLoader.from_file(source, movements=[PIANO])
            return (
                [timeline.to_dict() for timeline in loader.create_timelines()],
                claims_of(loader),
            )

        expected = read(ARCHIVE)
        assert read(str(ARCHIVE)) == expected
        assert read(zipfile.Path(ARCHIVE)) == expected
        assert read(folder) == expected
        assert read(folder / "MTD") == expected

    def test_metadata_names_the_source(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD1044"])
        entry = loader.metadata["sources"][0]
        assert entry["path"] == str(ARCHIVE)
        assert entry["themes"] == ["MTD1044"]
        assert entry["theme_midi"] == "data_EDM-corr_MID"


class TestIds:
    """Every ID is the minter's, seeded from the files read."""

    def test_seeded_ids_in_build_order(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"], seed=0)
        expected = minted(0, *["note"] * 10, "cpt", "clt", "cpt", *["claim"] * 10)
        track, score = loader.create_timelines()
        notes = score.get_events().table.column("id").to_pylist()
        assert notes == expected[:10]
        assert track.list_children() == [expected[10]]
        assert score.id == expected[11]
        assert track.id == expected[12]
        assert [row["id"] for row in claims_of(loader)] == expected[13:]

    def test_fresh_loaders_mint_equal_ids(self) -> None:
        first = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
        second = MtdLoader.from_file(ARCHIVE, movements=[PIANO])
        assert [t.id for t in first.create_timelines()] == [
            t.id for t in second.create_timelines()
        ]
        assert claims_of(first) == claims_of(second)
        assert first.create_bundle().id == second.create_bundle().id


# endregion


# region The bundle and its package


class TestBundle:
    """The bundle, written as a package and read back."""

    def test_bundle_of_one_movement(self) -> None:
        bundle = MtdLoader.from_file(ARCHIVE, movements=[PIANO]).create_bundle()
        assert isinstance(bundle, AlignmentBundle)
        assert bundle.name == "Beethoven Op002No1-01"
        assert len(bundle.timeline_ids) == 3
        assert len(bundle.match_claim_table) == 26

    def test_package_round_trip(self, tmp_path: Path) -> None:
        loader = MtdLoader.from_file(ARCHIVE)
        bundle = loader.create_bundle()
        manifest = tmp_path / "manifest.json"
        bundle.to_file(manifest, serialization="json")
        timetoshare.validate(json.loads(manifest.read_text(encoding="utf-8")))
        assert (tmp_path / "match_claims.parquet").is_file()
        read = AlignmentBundle.from_file(manifest)
        assert len(read.match_claim_table) == 64
        assert read.match_claim_table.table.to_pylist() == claims_of(loader)
        score = by_name(loader.create_timelines(), "B689 (MTD 0807)")
        track = by_name(
            loader.create_timelines(), "Beethoven Op002No1-01, Brendel (CD 49, track 1)"
        )
        for original in (score, track):
            assert read.get_timeline(original.id).statements == original.statements
        excerpt_id = track.list_children()[0]
        stamp = read.get_matchstamp_at(Fraction(20, 3), score.id)
        assert stamp.get_coordinate_for(excerpt_id, format="float") == 2.20401


# endregion


# region Refusals


class TestRefusals:
    """Each refusal comes from one change to an otherwise valid source."""

    def test_a_row_off_its_onset(self, folder: Path) -> None:
        edit(
            folder,
            "data_ALIGNMENT/MTD0807_Beethoven_Op002No1-01.csv",
            "6.66667",
            "6.70000",
        )
        with pytest.raises(ValueError, match="6.70000;2.20401 .* meets 0 onsets"):
            MtdLoader.from_file(folder, themes=["MTD0807"])

    def test_a_last_row_off_the_end(self, folder: Path) -> None:
        edit(
            folder,
            "data_ALIGNMENT/MTD0807_Beethoven_Op002No1-01.csv",
            "7.99792",
            "8.00000",
        )
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

    def test_no_encoding_folder(self) -> None:
        with pytest.raises(ValueError, match="data_EDM-orig_MID"):
            MtdLoader.from_file(ARCHIVE, theme_midi="data_EDM-orig_MID")

    def test_no_mtd_folder(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="data_META"):
            MtdLoader.from_file(tmp_path)

    def test_a_second_source(self, folder: Path) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        with pytest.raises(ValueError, match="reads one source"):
            loader.load(folder, themes=["MTD1044"])
        assert loader.themes == ["MTD0807"]
        loader.clear()
        assert loader.load(folder, themes=["MTD1044"]).themes == ["MTD1044"]

    def test_one_source(self) -> None:
        with pytest.raises(ValueError, match="got none"):
            MtdLoader().load()
        with pytest.raises(ValueError, match="got 2"):
            MtdLoader().load(ARCHIVE, ARCHIVE)

    def test_an_unknown_keyword_of_create_bundle(self) -> None:
        loader = MtdLoader.from_file(ARCHIVE, themes=["MTD0807"])
        with pytest.raises(TypeError, match="lengths"):
            loader.create_bundle(lengths={})

    def test_nothing_loaded(self) -> None:
        with pytest.raises(RuntimeError, match="load"):
            MtdLoader().create_bundle()


# endregion


# region Representation


def test_html_card_counts_claims() -> None:
    html = MtdLoader.from_file(ARCHIVE, movements=[PIANO])._repr_html_()
    assert "<tr><td><b>Claims</b></td><td>26</td></tr>" in html
    assert "<tr><td><b>Themes</b></td><td>2</td></tr>" in html
    assert "Events" not in html
    assert repr(MtdLoader.from_file(ARCHIVE, movements=[PIANO])) == (
        "MtdLoader(movements=1, themes=2, claims=26)"
    )
