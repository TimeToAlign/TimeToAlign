"""Loaders read sources inside ZIP archives as they read them from the filesystem.

The validation logic is in ``README.md`` beside this file: an archive source
is right if and only if loading the member gives exactly what loading the
same bytes from the filesystem gives.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import pytest

from timetoalign.loader import AudioLoader, RekordboxLoader, materialize
from timetoalign.loader._sources import (
    as_directory,
    as_source,
)
from timetoalign.loader._sources import cache_dir as archive_cache_dir
from timetoalign.loader._sources import (
    describe_source,
    find_sources,
    join_source,
    sort_sources,
    source_parent,
)
from timetoalign.loader.alignment import MatchfileLoader, PerformancePrecisionLoader
from timetoalign.loader.format import XmlLoader
from timetoalign.loader.graphical import IIIFManifestLoader
from timetoalign.loader.midi import PerformanceMidiLoader, ScoreMidiLoader
from timetoalign.loader.score import MeasureMapLoader, Ms3Loader
from timetoalign.loader.tabular import CsvLoader
from timetoalign.testdata import ensure_data

from .helpers import copies, file_members, lay_out, member, pack
from .helpers import same_timeline as same_result
from .helpers import strings_in

SCORE = ensure_data("score")
TABULAR = ensure_data("tabular")
MIDI = ensure_data("midi")
AUDIO = ensure_data("audio")
SUPRA = ensure_data("supra")
VIENNA = ensure_data("vienna_1x22")
PRECISION = ensure_data("performance_precision")

WOO71 = SCORE / "beethoven_woo71"
EVENTS_CSV = TABULAR / "test_events.csv"
PERFORMANCE_MIDI = MIDI / "performance" / "supra_raw.mid"
SCORE_MIDI = MIDI / "score" / "beethoven_mtd.mid"
REKORDBOX_XML = AUDIO / "moriero_dj_set" / "rekordbox.xml"
MEASURE_MAP = (
    SUPRA
    / "dcml"
    / "WWV096-Meistersinger_01_Vorspiel-Prelude_SchottKleinmichel.measures.mm.json"
)
IIIF_MANIFEST = SUPRA / "image" / "ifff_manifest.json"
MATCH_FILE = VIENNA / "Chopin_op10_no3_p20.match"
WAV = (
    SCORE
    / "beethoven_op18-4iv_multimodal"
    / "StringQuartetEEP_I_Normal"
    / "StringQuartetEEP_I_Normal_mono.wav.tonal.ChordsStrength.wav"
)


# region Same result as from the filesystem


class TestSameResultAsFilesystem:
    """One loader per family; the filesystem reading is the oracle."""

    def test_csv_loader(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        archived = CsvLoader.from_file(member(archive, "tables/test_events.csv"))
        on_disk = CsvLoader.from_file(EVENTS_CSV)
        assert archived.events.table.equals(on_disk.events.table)
        same_result(archived, on_disk)

    def test_ms3_measures(self, tmp_path: Path) -> None:
        source = WOO71 / "WoO71.measures.tsv"
        archive = pack(tmp_path / "dcml.zip", {"repo/WoO71.measures.tsv": source})
        archived = Ms3Loader.from_file(member(archive, "repo/WoO71.measures.tsv"))
        on_disk = Ms3Loader.from_file(source)
        assert archived.store.measures.table.equals(on_disk.store.measures.table)
        same_result(archived, on_disk)

    def test_ms3_auto_discover_flat(self, tmp_path: Path) -> None:
        facets = ("notes", "measures", "chords")
        archive = pack(
            tmp_path / "dcml.zip",
            {
                f"repo/WoO71.{facet}.tsv": WOO71 / f"WoO71.{facet}.tsv"
                for facet in facets
            },
        )
        archived = Ms3Loader.from_file(
            member(archive, "repo/WoO71.notes.tsv"), auto_discover=True
        )
        on_disk = Ms3Loader.from_file(WOO71 / "WoO71.notes.tsv", auto_discover=True)
        assert [source.name for source in on_disk.sources] == [
            "WoO71.notes.tsv",
            "WoO71.measures.tsv",
            "WoO71.chords.tsv",
        ]
        assert [
            source.at for source in archived.sources if isinstance(source, zipfile.Path)
        ] == [
            "repo/WoO71.notes.tsv",
            "repo/WoO71.measures.tsv",
            "repo/WoO71.chords.tsv",
        ]
        for facet in ("notes", "measures", "controls"):
            assert getattr(archived.store, facet).table.equals(
                getattr(on_disk.store, facet).table
            )
        same_result(archived, on_disk)

    def test_ms3_auto_discover_facet_folders(self, tmp_path: Path) -> None:
        layout = {
            "notes/WoO71.notes.tsv": WOO71 / "WoO71.notes.tsv",
            "measures/WoO71.measures.tsv": WOO71 / "WoO71.measures.tsv",
        }
        archive = pack(tmp_path / "dcml.zip", layout)
        on_disk_root = tmp_path / "corpus"
        for name, source in layout.items():
            (on_disk_root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, on_disk_root / name)
        archived = Ms3Loader.from_file(
            member(archive, "notes/WoO71.notes.tsv"), auto_discover=True
        )
        on_disk = Ms3Loader.from_file(
            on_disk_root / "notes" / "WoO71.notes.tsv", auto_discover=True
        )
        assert [
            source.at for source in archived.sources if isinstance(source, zipfile.Path)
        ] == ["notes/WoO71.notes.tsv", "measures/WoO71.measures.tsv"]
        assert archived.store.measures.table.equals(on_disk.store.measures.table)
        same_result(archived, on_disk)

    def test_performance_midi(self, tmp_path: Path) -> None:
        archive = pack(
            tmp_path / "midi.zip", {"performance/supra_raw.mid": PERFORMANCE_MIDI}
        )
        archived = PerformanceMidiLoader.from_file(
            member(archive, "performance/supra_raw.mid")
        )
        on_disk = PerformanceMidiLoader.from_file(PERFORMANCE_MIDI)
        assert archived.ticks_per_beat == on_disk.ticks_per_beat
        assert archived.events.table.equals(on_disk.events.table)
        same_result(archived, on_disk)

    def test_score_midi_partitura(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "midi.zip", {"score/beethoven_mtd.mid": SCORE_MIDI})
        archived = ScoreMidiLoader.from_file(member(archive, "score/beethoven_mtd.mid"))
        on_disk = ScoreMidiLoader.from_file(SCORE_MIDI)
        assert archived.events.table.equals(on_disk.events.table)
        same_result(archived, on_disk)

    def test_xml_loader(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "dj.zip", {"set/rekordbox.xml": REKORDBOX_XML})
        archived = XmlLoader.from_file(member(archive, "set/rekordbox.xml"))
        on_disk = XmlLoader.from_file(REKORDBOX_XML)
        assert (
            archived.keys()
            == on_disk.keys()
            == ["POSITION_MARK", "TEMPO", "TRACK", "NODE"]
        )
        for tag in on_disk.keys():
            assert archived.get_table(tag).equals(on_disk.get_table(tag))

    def test_rekordbox_loader(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "dj.zip", {"set/rekordbox.xml": REKORDBOX_XML})
        archived = RekordboxLoader.from_file(member(archive, "set/rekordbox.xml"))
        on_disk = RekordboxLoader.from_file(REKORDBOX_XML)
        assert archived.tracks == on_disk.tracks
        assert [timeline.to_dict() for timeline in archived.create_timelines()] == [
            timeline.to_dict() for timeline in on_disk.create_timelines()
        ]

    def test_measure_map_json(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "dcml.zip", {f"dcml/{MEASURE_MAP.name}": MEASURE_MAP})
        archived = MeasureMapLoader.from_file(
            member(archive, f"dcml/{MEASURE_MAP.name}")
        )
        on_disk = MeasureMapLoader.from_file(MEASURE_MAP)
        assert archived.store.measures.table.equals(on_disk.store.measures.table)
        same_result(archived, on_disk)

    def test_iiif_manifest_json(self, tmp_path: Path) -> None:
        archive = pack(
            tmp_path / "roll.zip", {"image/ifff_manifest.json": IIIF_MANIFEST}
        )
        archived = IIIFManifestLoader.from_file(
            member(archive, "image/ifff_manifest.json")
        )
        on_disk = IIIFManifestLoader.from_file(IIIF_MANIFEST)
        assert archived.manifest_info == on_disk.manifest_info

    def test_audio_loader_reads_in_place(self, tmp_path: Path, cache_dir: Path) -> None:
        archive = pack(tmp_path / "audio.zip", {f"audio/{WAV.name}": WAV})
        source = member(archive, f"audio/{WAV.name}")
        archived = AudioLoader.from_file(source)
        on_disk = AudioLoader.from_file(WAV)
        archived_info = archived.audio_info
        assert archived_info.source_path is source
        assert archived.source_path is source
        on_disk_info = on_disk.audio_info
        for name in (
            "n_samples",
            "sample_rate",
            "channels",
            "duration_seconds",
            "format",
            "subtype",
            "bits_per_sample",
            "extra",
            "backend",
        ):
            assert getattr(archived_info, name) == getattr(on_disk_info, name), name
        same_result(archived, on_disk)
        # Every audio backend reads a file object: nothing is copied.
        assert copies(cache_dir) == []

    def test_matchfile_loader_reads_a_copy(self, tmp_path: Path) -> None:
        archive = pack(
            tmp_path / "vienna.zip", {f"vienna/{MATCH_FILE.name}": MATCH_FILE}
        )
        archived = MatchfileLoader.from_file(
            member(archive, f"vienna/{MATCH_FILE.name}")
        )
        on_disk = MatchfileLoader.from_file(MATCH_FILE)
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()

    def test_performance_precision_directory(self, tmp_path: Path) -> None:
        archive = pack(
            tmp_path / "specimen.zip", file_members(PRECISION, PRECISION.name)
        )
        # Named without its closing "/": the directory member is meant.
        archived = PerformancePrecisionLoader().load(member(archive, PRECISION.name))
        on_disk = PerformancePrecisionLoader().load(PRECISION)
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()
        assert [
            source.at for source in archived.sources if isinstance(source, zipfile.Path)
        ] == [
            f"{PRECISION.name}/Chopin Nocturne Op. 9 No. 2.solo",
            f"{PRECISION.name}/Chopin Nocturne Op. 9 No. 2.json",
            f"{PRECISION.name}/Alignments/",
        ]


# endregion

# region Provenance


class TestProvenance:
    """A member's metadata names the archive and the member."""

    def test_member_records_archive_and_member(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        source = member(archive, "tables/test_events.csv")
        loader = CsvLoader.from_file(source)
        (entry,) = loader.metadata["sources"]
        assert entry["path"] == str(archive.absolute())
        assert entry["member"] == "tables/test_events.csv"
        assert loader.sources == [source]

    def test_filesystem_source_records_path_only(self) -> None:
        loader = CsvLoader.from_file(EVENTS_CSV)
        (entry,) = loader.metadata["sources"]
        assert entry["path"] == str(EVENTS_CSV)
        assert "member" not in entry
        assert loader.sources == [EVENTS_CSV]

    def test_score_loader_records_member(self, tmp_path: Path) -> None:
        source_file = WOO71 / "WoO71.measures.tsv"
        archive = pack(tmp_path / "dcml.zip", {"repo/WoO71.measures.tsv": source_file})
        loader = Ms3Loader.from_file(member(archive, "repo/WoO71.measures.tsv"))
        (entry,) = loader.metadata["sources"]
        assert (entry["path"], entry["member"]) == (
            str(archive.absolute()),
            "repo/WoO71.measures.tsv",
        )


# endregion

# region materialize


class TestMaterialize:
    """The one way to a file on disk for a parser that needs one."""

    def test_filesystem_path_unchanged(self, cache_dir: Path) -> None:
        assert materialize(EVENTS_CSV) == EVENTS_CSV
        assert materialize(str(EVENTS_CSV)) == EVENTS_CSV
        assert not cache_dir.exists()

    def test_cache_dir_honours_environment(self, cache_dir: Path) -> None:
        assert archive_cache_dir() == cache_dir

    def test_cache_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TTA_CACHE_DIR")
        assert archive_cache_dir() == Path.home() / ".cache" / "timetoalign"

    def test_copies_once(self, tmp_path: Path, cache_dir: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"a/b/test_events.csv": EVENTS_CSV})
        copy = materialize(member(archive, "a/b/test_events.csv"))
        assert copy.parent.parent.parent.parent == cache_dir / "archives"
        assert copy.relative_to(copy.parents[2]).as_posix() == "a/b/test_events.csv"
        assert copy.read_bytes() == EVENTS_CSV.read_bytes()
        first = copy.stat()
        again = materialize(member(archive, "a/b/test_events.csv"))
        assert again == copy
        second = again.stat()
        assert (second.st_ino, second.st_mtime_ns) == (first.st_ino, first.st_mtime_ns)
        assert not list(cache_dir.rglob("*.part"))

    def test_member_changed_in_place_gets_a_fresh_copy(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"inner.txt": b"first"})
        status = archive.stat()
        first = materialize(member(archive, "inner.txt"))
        # Same member size, so the same archive size; the old mtime restored.
        pack(archive, {"inner.txt": b"other"})
        os.utime(archive, ns=(status.st_atime_ns, status.st_mtime_ns))
        rewritten = archive.stat()
        assert (rewritten.st_size, rewritten.st_mtime_ns) == (
            status.st_size,
            status.st_mtime_ns,
        )
        second = materialize(member(archive, "inner.txt"))
        assert second != first
        assert first.read_bytes() == b"first"
        assert second.read_bytes() == b"other"

    def test_changed_archive_gets_a_fresh_copy(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"inner.txt": b"first"})
        first = materialize(member(archive, "inner.txt"))
        pack(archive, {"inner.txt": b"second, longer"})
        status = archive.stat()
        os.utime(archive, ns=(status.st_atime_ns, status.st_mtime_ns + 1_000_000_000))
        second = materialize(member(archive, "inner.txt"))
        assert second != first
        assert first.read_bytes() == b"first"
        assert second.read_bytes() == b"second, longer"

    def test_directory_member_copies_its_subtree(self, tmp_path: Path) -> None:
        archive = pack(
            tmp_path / "data.zip",
            {
                "corpus/one.txt": b"1",
                "corpus/deep/two.txt": b"2",
                "corpus_sibling.txt": b"x",
                "other/three.txt": b"3",
            },
        )
        directory = materialize(member(archive, "corpus/"))
        assert directory.name == "corpus"
        copied = sorted(
            path.relative_to(directory.parent).as_posix()
            for path in directory.parent.rglob("*")
            if path.is_file()
        )
        assert copied == ["corpus/deep/two.txt", "corpus/one.txt"]
        assert (directory / "deep" / "two.txt").read_bytes() == b"2"

    def test_member_outside_archive_is_refused(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"../escape.txt": b"x"})
        with pytest.raises(
            ValueError, match=r"'\.\./escape\.txt' names a path outside"
        ):
            materialize(member(archive, "../escape.txt"))


# endregion

# region Navigation inside an archive


class TestNavigation:
    """Searching and joining inside an archive mirror the filesystem."""

    def test_find_sources_matches_glob(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "specimen.zip", file_members(PRECISION, "specimen"))
        alignments = member(archive, "specimen/Alignments/")
        found = find_sources(alignments, "*_bar.csv")
        expected = sorted(
            path.name for path in (PRECISION / "Alignments").glob("*_bar.csv")
        )
        assert [source.name for source in found] == expected
        assert len(expected) == 7
        recursive = find_sources(member(archive, "specimen/"), "*.csv", recursive=True)
        assert [source.name for source in recursive] == [
            path.name for path in find_sources(PRECISION, "*.csv", recursive=True)
        ]

    def test_find_sources_sees_implied_directories(self, tmp_path: Path) -> None:
        # pack() writes file entries only: "corpus/audio/" has no entry.
        archive = pack(
            tmp_path / "data.zip",
            {"corpus/audio/take.wav": b"w", "corpus/score.mei": b"m"},
        )
        found = find_sources(member(archive, "corpus/"), "aud*")
        assert [source.at for source in found] == ["corpus/audio/"]
        assert found[0].is_dir()
        assert [
            source.at
            for source in find_sources(member(archive, "corpus/"), "*", recursive=True)
        ] == ["corpus/audio/", "corpus/audio/take.wav", "corpus/score.mei"]

    def test_one_order_for_paths_and_members(self, tmp_path: Path) -> None:
        names = ["b.wav", "C.wav", "a_1.wav", "a.wav", "a-2.wav"]
        expected = ["C.wav", "a-2.wav", "a.wav", "a_1.wav", "b.wav"]
        content = {f"take/{name}": name.encode() for name in names}
        archive = pack(tmp_path / "data.zip", content)
        disk = lay_out(tmp_path / "disk", content) / "take"
        archived = find_sources(member(archive, "take/"), "*.wav")
        on_disk = find_sources(disk, "*.wav")
        assert [source.name for source in archived] == expected
        assert [source.name for source in on_disk] == expected
        mixed = sort_sources([*archived[::-1], *on_disk[::-1]])
        assert mixed == [*on_disk, *archived]

    def test_join_resolves_dot_segments(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"project/audio/take.png": b"png"})
        joined = join_source(member(archive, "project/"), "./audio/take.png")
        assert isinstance(joined, zipfile.Path)
        assert joined.at == "project/audio/take.png"
        assert joined.read_bytes() == b"png"
        assert not join_source(
            member(archive, "project/"), "../../outside.png"
        ).exists()

    def test_the_archive_root_is_a_source(self, tmp_path: Path) -> None:
        # zipfile.Path(archive).exists() is False although the root holds everything.
        archive = pack(tmp_path / "data.zip", {"corpus/score.mei": b"m"})
        root = zipfile.Path(archive)
        assert as_source(root) is root
        assert [source.at for source in find_sources(as_source(root), "*")] == [
            "corpus/"
        ]
        assert describe_source(root) == str(archive.absolute())
        assert join_source(root, "corpus/score.mei").read_bytes() == b"m"

    def test_a_zip_file_on_disk_is_read_as_its_root(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"corpus/score.mei": b"m"})
        directory = as_directory(as_source(archive))
        assert isinstance(directory, zipfile.Path)
        assert directory.at == ""
        assert directory.root.filename == str(archive)
        other = lay_out(tmp_path / "disk", {"corpus/score.mei": b"m"})
        assert as_directory(other) == other
        assert (
            as_directory(other / "corpus" / "score.mei")
            == other / "corpus" / "score.mei"
        )
        inner = member(archive, "corpus/")
        assert as_directory(inner) is inner

    def test_parent_stays_inside_archive(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"top.txt": b"t"})
        parent = source_parent(member(archive, "top.txt"))
        assert isinstance(parent, zipfile.Path)
        assert parent.at == ""
        assert isinstance(source_parent(parent), zipfile.Path)


# endregion

# region Errors


class TestErrors:
    """Missing members, string paths and directories."""

    def test_missing_member_names_archive_and_member(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        with pytest.raises(FileNotFoundError) as caught:
            CsvLoader.from_file(member(archive, "tables/missing.csv"))
        message = str(caught.value)
        assert str(archive.absolute()) in message
        assert "'tables/missing.csv'" in message

    def test_missing_member_for_materialize(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"inner.txt": b"x"})
        with pytest.raises(FileNotFoundError, match="holds no member 'other.txt'"):
            materialize(member(archive, "other.txt"))

    def test_string_is_a_filesystem_path(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "x.zip", {"inner.csv": EVENTS_CSV})
        string = f"{archive}/inner.csv"
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            CsvLoader.from_file(string)
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            CsvLoader.from_file(Path(string))

    def test_directory_member_fails_as_a_directory(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        with pytest.raises(IsADirectoryError):
            CsvLoader.from_file(TABULAR)
        with pytest.raises(IsADirectoryError):
            CsvLoader.from_file(member(archive, "tables"))


# endregion

# region Archive members never become path strings


FLOW_ONLY = SCORE / "flow_control" / "out_of_the_flow_experience-flow_only"
REPOVIZZ = SCORE / "beethoven_op18-4iv_multimodal" / "StringQuartetEEP_I_Normal"


def _score_loaders() -> list[tuple[str, Path, dict[str, object]]]:
    """(loader class name, file, constructor keywords) per store-writing loader."""
    return [
        ("Ms3Loader", WOO71 / "WoO71.measures.tsv", {}),
        ("Ms3Loader", WOO71 / "WoO71.notes.tsv", {}),
        ("MeasureMapLoader", MEASURE_MAP, {}),
        (
            "VerovioLoader",
            FLOW_ONLY.with_suffix(".mei"),
            {"options": {"expandNever": True}},
        ),
        ("PartituraLoader", FLOW_ONLY.with_suffix(".musicxml"), {}),
        ("Music21Loader", FLOW_ONLY.with_suffix(".musicxml"), {}),
    ]


class TestNoArchivePathInMetadata:
    """A member is recorded as its archive's path plus ``member``, nowhere joined."""

    @pytest.mark.parametrize(
        ("name", "source", "keywords"),
        _score_loaders(),
        ids=[f"{name}-{source.suffix}" for name, source, _ in _score_loaders()],
    )
    def test_score_store_metadata(
        self, tmp_path: Path, name: str, source: Path, keywords: dict[str, object]
    ) -> None:
        from timetoalign.loader import score

        loader_class = getattr(score, name)
        archive = pack(tmp_path / "scores.zip", {f"inner/{source.name}": source})
        loader = loader_class.from_file(
            member(archive, f"inner/{source.name}"), **keywords
        )
        recorded = [*strings_in(loader.metadata), *strings_in(loader.store.metadata)]
        assert not [text for text in recorded if ".zip/" in text]
        assert loader.store.metadata["source"] == str(archive.absolute())
        assert loader.store.metadata["member"] == f"inner/{source.name}"
        on_disk = loader_class.from_file(source, **keywords)
        assert on_disk.store.metadata["source"] == str(source)
        assert "member" not in on_disk.store.metadata

    def test_repovizz_manifest_metadata(self, tmp_path: Path) -> None:
        from timetoalign.loader.physical import RepoVizzLoader

        manifest = REPOVIZZ / "StringQuartetEEP_I_Normal.xml"
        archive = pack(tmp_path / "repovizz.zip", {f"take/{manifest.name}": manifest})
        loader = RepoVizzLoader().load(member(archive, f"take/{manifest.name}"))
        assert not [
            text for text in strings_in(loader._xml_metadata) if ".zip/" in text
        ]
        assert loader._xml_metadata["source_path"] == str(archive.absolute())
        assert loader._xml_metadata["member"] == f"take/{manifest.name}"
        on_disk = RepoVizzLoader().load(manifest)
        assert on_disk._xml_metadata["source_path"] == str(manifest)
        assert "member" not in on_disk._xml_metadata


# endregion

# region MeLODy: relative IRIs inside an archive

RELATIVE_TURTLE = """\
@prefix here: <#> .
<other.ttl> <https://example.org/p> <../persons/3.ttl> .
here:self <https://example.org/p> "v" .
"""


class TestMelodyMembers:
    """A Turtle member is parsed in place; its base IRI is the member's ``jar:`` URI."""

    def test_relative_iris_resolve_inside_the_archive(self, tmp_path: Path) -> None:
        from timetoalign.loader import MelodyLoader

        archive = pack(tmp_path / "melo dy.zip", {"repo/works/1.ttl": RELATIVE_TURTLE})
        loader = MelodyLoader.from_file(member(archive, "repo/works/1.ttl"))
        ((relative, graph),) = loader.records[0].files
        base = f"jar:{archive.absolute().as_uri()}!/repo/"
        assert base.startswith("jar:file:///") and "melo%20dy.zip!/" in base
        assert relative == "1.ttl"
        assert {(str(s), str(p), str(o)) for s, p, o in graph} == {
            (f"{base}works/other.ttl", "https://example.org/p", f"{base}persons/3.ttl"),
            (f"{base}works/1.ttl#self", "https://example.org/p", "v"),
        }

    def test_filesystem_file_resolves_as_before(self, tmp_path: Path) -> None:
        from timetoalign.loader import MelodyLoader

        path = lay_out(tmp_path / "disk", {"repo/works/1.ttl": RELATIVE_TURTLE})
        loader = MelodyLoader.from_file(path / "repo" / "works" / "1.ttl")
        ((_, graph),) = loader.records[0].files
        base = f"{(path / 'repo').absolute().as_uri()}/"
        assert {(str(s), str(p), str(o)) for s, p, o in graph} == {
            (f"{base}works/other.ttl", "https://example.org/p", f"{base}persons/3.ttl"),
            (f"{base}works/1.ttl#self", "https://example.org/p", "v"),
        }

    def test_directory_member_reads_each_file(
        self, tmp_path: Path, cache_dir: Path
    ) -> None:
        from timetoalign.loader import MelodyLoader

        files = {
            "repo/works/1.ttl": RELATIVE_TURTLE,
            "repo/persons/3.ttl": '<#p> <https://example.org/name> "P" .\n',
            "repo/indexes/all.ttl": "<x> <y> <z> .\n",
        }
        archive = pack(tmp_path / "repo.zip", files)
        disk = lay_out(tmp_path / "disk", files) / "repo"
        archived = MelodyLoader.from_file(member(archive, "repo")).records[0]
        on_disk = MelodyLoader.from_file(disk).records[0]
        assert (
            [relative for relative, _ in archived.files]
            == [relative for relative, _ in on_disk.files]
            == ["persons/3.ttl", "works/1.ttl"]
        )
        archive_base = f"jar:{archive.absolute().as_uri()}!/repo/"
        disk_base = f"{disk.absolute().as_uri()}/"
        for (_, archived_graph), (_, disk_graph) in zip(archived.files, on_disk.files):
            assert {
                tuple(str(node).replace(archive_base, disk_base) for node in triple)
                for triple in archived_graph
            } == {tuple(str(node) for node in triple) for triple in disk_graph}
        assert copies(cache_dir) == []


# endregion

# region Display


class TestDisplay:
    """Messages name a member as ``<archive> member <name>``."""

    def test_describe_source(self, tmp_path: Path) -> None:
        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        assert describe_source(member(archive, "tables/test_events.csv")) == (
            f"{archive.absolute()} member tables/test_events.csv"
        )
        assert describe_source(EVENTS_CSV) == str(EVENTS_CSV)

    def test_event_summary_and_refusal_name_the_member(self, tmp_path: Path) -> None:
        from timetoalign.loader.score import VerovioLoader

        archive = pack(tmp_path / "data.zip", {"tables/test_events.csv": EVENTS_CSV})
        loader = CsvLoader.from_file(member(archive, "tables/test_events.csv"))
        assert loader.event_summary()["sources"] == [
            f"{archive.absolute()} member tables/test_events.csv"
        ]
        mei = FLOW_ONLY.with_suffix(".mei")
        scores = pack(tmp_path / "scores.zip", {f"s/{mei.name}": mei})
        verovio = VerovioLoader.from_file(
            member(scores, f"s/{mei.name}"), options={"expandNever": True}
        )
        with pytest.raises(ValueError) as caught:
            verovio.load(member(scores, f"s/{mei.name}"))
        named = f"{scores.absolute()} member s/{mei.name}"
        assert str(caught.value) == (
            f"VerovioLoader reads one source and already holds {named}; use a new "
            f"loader for {named}, or call clear() first."
        )

    def test_parser_error_on_a_copy_names_the_member(self, tmp_path: Path) -> None:
        from timetoalign.loader.score import PartituraLoader

        archive = pack(tmp_path / "data.zip", {"broken/score.musicxml": b"not xml"})
        with pytest.raises(Exception) as caught:
            PartituraLoader.from_file(member(archive, "broken/score.musicxml"))
        (note,) = caught.value.__notes__
        assert note.endswith(
            f"is the copy of {archive.absolute()} member broken/score.musicxml."
        )


# endregion
