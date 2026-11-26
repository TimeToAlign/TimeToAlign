"""Each loader's read path gives from a ZIP member exactly what it gives from disk.

The validation logic is in ``README.md`` beside this file: an archive source
is right if and only if loading the member gives exactly what loading the
same bytes from the filesystem gives. One test per read path; the families
of ``test_archive_sources.py`` are not repeated here.
"""

from __future__ import annotations

import dataclasses
import json
import wave
from pathlib import Path
from typing import Any

import pytest

from timetoalign.loader import AudioLoader
from timetoalign.testdata import ensure_data

from .helpers import copies, file_members, lay_out, member, pack, same_timeline

SCORE = ensure_data("score")
AUDIO = ensure_data("audio")
SUPRA = ensure_data("supra")
FIXTURES = ensure_data("fixtures")
VIENNA = ensure_data("vienna_1x22")
PRECISION = ensure_data("performance_precision")
MPM_PROJECT = ensure_data("mpm_toolbox") / "MPRproject_1971Curzon_VariationXIV"
PARANGONADA = ensure_data("parangonar") / "Beethoven_Eroica_op35-cpjku"
IEEE_ANIMALS = ensure_data("ieee1599") / "Animals and their Sounds"

FLOW_ONLY = SCORE / "flow_control" / "out_of_the_flow_experience-flow_only"
REPOVIZZ = SCORE / "beethoven_op18-4iv_multimodal" / "StringQuartetEEP_I_Normal"
SOLO = PRECISION / "Chopin Nocturne Op. 9 No. 2.solo"


def _both(tmp_path: Path, files: dict[str, Any], name: str) -> tuple[Any, Path]:
    """*files* packed into an archive and laid out on disk; the member and path *name*."""
    archive = pack(tmp_path / "specimen.zip", files)
    disk = lay_out(tmp_path / "disk", files)
    return member(archive, name), disk / name


# region Score loaders reading a copy


class TestScoreLoaders:
    """Verovio, Partitura and music21 need a file on disk: they read a copy."""

    @pytest.mark.parametrize(
        ("loader_name", "suffix", "keywords"),
        [
            ("VerovioLoader", ".mei", {"options": {"expandNever": True}}),
            ("PartituraLoader", ".musicxml", {}),
            ("Music21Loader", ".musicxml", {}),
        ],
    )
    def test_same_events_and_timeline(
        self,
        tmp_path: Path,
        cache_dir: Path,
        loader_name: str,
        suffix: str,
        keywords: dict[str, Any],
    ) -> None:
        from timetoalign.loader import score

        loader_class = getattr(score, loader_name)
        source = FLOW_ONLY.with_suffix(suffix)
        archive = pack(tmp_path / "scores.zip", {f"flow/{source.name}": source})
        archived = loader_class.from_file(
            member(archive, f"flow/{source.name}"), **keywords
        )
        on_disk = loader_class.from_file(source, **keywords)
        assert archived.events.table.equals(on_disk.events.table)
        same_timeline(archived, on_disk)
        assert [path.name for path in copies(cache_dir)] == [source.name]


# endregion

# region Alignment and project loaders


class TestProjectLoaders:
    """Loaders that navigate a project: siblings, searches, headers of recordings."""

    def test_mpm_project(self, tmp_path: Path, cache_dir: Path) -> None:
        from timetoalign.loader.alignment import MpmLoader

        prefix = MPM_PROJECT.name
        archive = pack(tmp_path / "mpm.zip", file_members(MPM_PROJECT, prefix))
        mpr = "Beethoven_op35_1971Curzon_Var14only.mpr"
        archived = MpmLoader().load(member(archive, f"{prefix}/{mpr}"))
        on_disk = MpmLoader().load(MPM_PROJECT / mpr)
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()
        assert [source.at for source in archived.sources] == [
            f"{prefix}/{mpr}",
            f"{prefix}/Beethoven_op35_1971Curzon_Var14only.msm",
            f"{prefix}/Beethoven_op35_1971Curzon_Var14only.mpm",
        ]
        # The spectrogram timeline's length is its PNG's width, read in place.
        assert [t.to_dict() for t in archived.create_timelines()] == [
            t.to_dict() for t in on_disk.create_timelines()
        ]
        # The recursive *.wav search picks the same recording in both.
        from timetoalign.loader._sources import find_sources, source_parent

        archived_wavs = find_sources(
            source_parent(archived.sources[0]), "*.wav", recursive=True
        )
        disk_wavs = find_sources(MPM_PROJECT, "*.wav", recursive=True)
        assert (
            [w.name for w in archived_wavs]
            == [w.name for w in disk_wavs]
            == [
                "DECCASXL6523_Curzon_Var14.wav",
                "DECCASXL6523_Curzon_Var14_16bit.wav",
            ]
        )
        assert copies(cache_dir) == []

    def test_listen_here(self, tmp_path: Path, cache_dir: Path) -> None:
        from timetoalign.loader.alignment import ListenHereLoader

        spec = {
            "header": {"ref": "audio/ref.wav", "createdBy": "Listen Here! v0.20.0"},
            "body": {
                "audio": {
                    key: {"times": times, "peaks": [0.1], "duration": times[-1]}
                    for key, times in {
                        "audio/a.wav": [0.0, 0.02, 0.04],
                        "audio/b.wav": [0.01, 0.03, 0.05],
                        "audio/ref.wav": [0.0, 0.025, 0.045],
                    }.items()
                }
            },
        }
        files: dict[str, Any] = {"project/alignment.json": json.dumps(spec)}
        source, path = _both(tmp_path, files, "project/alignment.json")
        archived = ListenHereLoader.from_file(source)
        on_disk = ListenHereLoader.from_file(path)
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()
        assert [timeline.length.value for timeline in archived.create_timelines()] == [
            0.04,
            0.05,
            0.045,
        ]
        assert copies(cache_dir) == []

    def test_parangonada(self, tmp_path: Path, cache_dir: Path) -> None:
        from timetoalign.loader.alignment import ParangonadaLoader

        prefix = PARANGONADA.name
        archive = pack(
            tmp_path / "parangonada.zip",
            file_members(
                PARANGONADA,
                prefix,
                only=("match/match_transkun/", "features/", "audio/"),
            ),
        )
        archived = ParangonadaLoader().load(member(archive, prefix))
        on_disk = ParangonadaLoader().load(PARANGONADA)
        assert archived.performer_keys == on_disk.performer_keys
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()
        assert copies(cache_dir) == []

    def test_repovizz_manifest_and_notes(self, tmp_path: Path) -> None:
        from timetoalign.loader.physical import RepoVizzLoader

        manifest = "StringQuartetEEP_I_Normal.xml"
        files: dict[str, Any] = {
            f"take/{path.name}": path for path in sorted(REPOVIZZ.glob("*.notes"))
        }
        files[f"take/{manifest}"] = REPOVIZZ / manifest
        files["take/vln1_bow_vel.csv"] = REPOVIZZ / "vln1_bow_vel.csv"
        source, path = _both(tmp_path, files, f"take/{manifest}")
        archived = RepoVizzLoader().load(source)
        on_disk = RepoVizzLoader().load(path)
        assert archived.catalogue == on_disk.catalogue
        assert archived.store.notes is not None and on_disk.store.notes is not None
        assert len(on_disk.store.notes.table) > 0
        assert archived.store.notes.table.equals(on_disk.store.notes.table)
        entry = next(
            key
            for key, value in on_disk.catalogue.items()
            if value.filename == "vln1_bow_vel.csv"
        )
        assert (
            archived.create_timeline(entry).to_dict()
            == on_disk.create_timeline(entry).to_dict()
        )

    def test_repovizz_csv(self, tmp_path: Path) -> None:
        from timetoalign.loader.physical import RepoVizzLoader

        csv = REPOVIZZ / "vln1_bow_vel.csv"
        archive = pack(tmp_path / "repovizz.zip", {f"take/{csv.name}": csv})
        archived = RepoVizzLoader().load(member(archive, f"take/{csv.name}"))
        on_disk = RepoVizzLoader().load(csv)
        same_timeline(archived, on_disk)

    def test_ieee1599(self, tmp_path: Path) -> None:
        from timetoalign.loader.alignment import Ieee1599Loader

        prefix = IEEE_ANIMALS.name
        xml = "animals_and_their_sounds.xml"
        archive = pack(tmp_path / "ieee.zip", file_members(IEEE_ANIMALS, prefix))
        source = member(archive, f"{prefix}/{xml}")
        archived = Ieee1599Loader.from_file(source)
        on_disk = Ieee1599Loader.from_file(IEEE_ANIMALS / xml)
        assert archived.create_bundle().to_dict() == on_disk.create_bundle().to_dict()
        (entry,) = archived.metadata["sources"]
        assert (entry["path"], entry["member"]) == (
            str(archive.absolute()),
            f"{prefix}/{xml}",
        )
        named = f"{archive.absolute()} member {prefix}/{xml}"
        with pytest.raises(ValueError) as caught:
            archived.load(source)
        assert str(caught.value) == (
            f"Ieee1599Loader reads one source and already holds {named}; use a new "
            f"loader for {named}, or call clear() first."
        )


# endregion

# region Graphical and metadata loaders

MEI_FACSIMILE = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Chorale</title>
  </titleStmt><pubStmt/></fileDesc></meiHead>
  <music><facsimile>
    <surface xml:id="surface_1" n="15" ulx="0" uly="0" lrx="3999" lry="2999">
      <graphic xml:id="graphic_1" target="facsimile/page.jpg" width="4000" height="3000"/>
      <zone xml:id="zone_1" type="measure" ulx="490" uly="344" lrx="973" lry="877"/>
      <zone xml:id="zone_2" type="measure" ulx="944" uly="368" lrx="1523" lry="864"/>
    </surface>
  </facsimile></music>
</mei>
"""

MEI_HEAD = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.1">
 <meiHead>
  <workList>
   <work xml:id="work_1">
    <titleStmt><title xml:id="title_1">Andante</title></titleStmt>
    <composer xml:id="composer_1">Anonymous</composer>
   </work>
  </workList>
 </meiHead>
</mei>
"""

SVG_AXIS = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="792" height="175">'
    '<g class="timeAxis">'
    '<line x1="76.5" x2="932.25" y1="155" y2="155" class="horizontalAxis"/>'
    '<text x="1" y="151">Time (s)</text>'
    + "".join(
        f'<line x1="{x}" x2="{x}" y1="155" y2="159"/>'
        f'<text x="{x}" y="168">{label}</text>'
        for x, label in ((76.5, "0"), (290.5, "10"), (504.5, "20"))
    )
    + "</g></svg>"
)


def _dmos_document() -> dict[str, Any]:
    """A DMOS file: page 1, one system with two measures."""
    url = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688"
    system_region = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
    measures = [
        [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
        [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]],
    ]
    system = {
        "id": 1,
        "region": system_region,
        "headers": [
            {"id_part": "P1", "no_staff": 1, "name": "Voix", "region": system_region}
        ],
        "measures": [
            {
                "region": region,
                "headers": [{"no_staff": 1, "region": region}],
                "voices": [{"id": "1", "id_part": "P1", "elements": [{"no_group": 1}]}],
            }
            for region in measures
        ],
    }
    return {
        "id": "C055_0",
        "date": "2025-06-10-11h",
        "score_image_url": url,
        "pages": [
            {
                "page_url": f"{url}/f2",
                "no_page": 1,
                "header_systems": {},
                "systems": [system],
            }
        ],
    }


class TestGraphicalAndMetadataLoaders:
    """Parsers taking a file object read the member in place."""

    def test_mei_facsimile(self, tmp_path: Path) -> None:
        from timetoalign.loader.graphical import MeiFacsimileLoader

        source, path = _both(
            tmp_path, {"ed/facsimile.mei": MEI_FACSIMILE}, "ed/facsimile.mei"
        )
        archived = MeiFacsimileLoader.from_file(source)
        on_disk = MeiFacsimileLoader.from_file(path)
        assert archived.create_paths() == on_disk.create_paths()
        assert [t.to_dict() for t in archived.create_timelines()] == [
            t.to_dict() for t in on_disk.create_timelines()
        ]

    def test_mei_head(self, tmp_path: Path) -> None:
        from timetoalign.loader import MeiHeadLoader

        source, path = _both(tmp_path, {"cat/cnw0001.xml": MEI_HEAD}, "cat/cnw0001.xml")
        archived = MeiHeadLoader.from_file(source).create_bundle(date="2026-11-19")
        on_disk = MeiHeadLoader.from_file(path).create_bundle(date="2026-11-19")
        assert archived.to_dict() == on_disk.to_dict()

    def test_scorewarp(self, tmp_path: Path) -> None:
        from timetoalign.loader.graphical import ScoreWarpLoader

        source, path = _both(
            tmp_path, {"plots/warped.svg": SVG_AXIS}, "plots/warped.svg"
        )
        same_timeline(
            ScoreWarpLoader.from_file(source), ScoreWarpLoader.from_file(path)
        )

    def test_dmos(self, tmp_path: Path) -> None:
        from timetoalign.loader.graphical import DmosLoader

        text = json.dumps(_dmos_document())
        source, path = _both(tmp_path, {"omr/iiif.json": text}, "omr/iiif.json")
        archived = DmosLoader.from_file(source)
        on_disk = DmosLoader.from_file(path)
        assert archived.create_paths() == on_disk.create_paths()
        assert [t.to_dict() for t in archived.create_timelines()] == [
            t.to_dict() for t in on_disk.create_timelines()
        ]

    def test_aton(self, tmp_path: Path) -> None:
        from timetoalign.loader.graphical.aton import ATONLoader

        aton = SUPRA / "image" / "fd660zf8362_analysis.txt"
        archive = pack(tmp_path / "roll.zip", {f"image/{aton.name}": aton})
        archived = ATONLoader.from_file(member(archive, f"image/{aton.name}"))
        on_disk = ATONLoader.from_file(aton)
        assert archived.holes == on_disk.holes
        same_timeline(archived, on_disk)

    def test_image_source(self, tmp_path: Path) -> None:
        from timetoalign.loader.graphical import ImageSource

        png = VIENNA / "Chopin_op10_no3.png"
        archive = pack(tmp_path / "scans.zip", {f"scans/{png.name}": png})
        source = member(archive, f"scans/{png.name}")
        archived = ImageSource.from_image_file(source)
        on_disk = ImageSource.from_image_file(png)
        assert archived.metadata.source_path is source
        assert dataclasses.replace(archived.metadata, source_path=None) == (
            dataclasses.replace(on_disk.metadata, source_path=None)
        )
        assert archived.pixmap.samples == on_disk.pixmap.samples


# endregion

# region Tabular, JSON and MAPS loaders

MAPS_MEI = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
<meiHead><fileDesc><titleStmt><title>Fixture</title></titleStmt><pubStmt/></fileDesc>
</meiHead><music><body><mdiv xml:id="mdiv1"><score>
<scoreDef xml:id="sd" meter.count="4" meter.unit="4"><staffGrp xml:id="sg">
  <staffDef xml:id="sdef1" n="1" lines="5" clef.shape="G" clef.line="2"/>
</staffGrp></scoreDef>
<section xml:id="s1"><measure xml:id="m1" n="1">
  <staff xml:id="m1s1" n="1"><layer xml:id="m1s1l1" n="1">
    <note xml:id="n1" pname="c" oct="4" dur="4"/>
    <note xml:id="n2" pname="d" oct="4" dur="4"/>
    <note xml:id="n3" pname="e" oct="4" dur="2"/>
  </layer></staff>
</measure></section></score></mdiv></body></music></mei>
"""

MAPS_ROWS = [
    {"obs_mean_onset": 0.5, "xml_id": ["n1"], "obs_num": 1},
    {"obs_mean_onset": 1.0, "xml_id": ["n2"], "obs_num": 2},
    {"obs_mean_onset": 1.75, "xml_id": ["n3"], "obs_num": 3},
]


class TestTabularJsonAndMaps:
    """pandas, json and csv read the member's open file object."""

    def test_lab_loader(self, tmp_path: Path) -> None:
        from timetoalign.loader.tabular import LabLoader

        lab = FIXTURES / "lab" / "beats.lab"
        archive = pack(tmp_path / "labels.zip", {f"lab/{lab.name}": lab})
        archived = LabLoader().load(member(archive, f"lab/{lab.name}"))
        on_disk = LabLoader().load(lab)
        assert archived.events.table.equals(on_disk.events.table)
        same_timeline(archived, on_disk)

    def test_solo_loader(self, tmp_path: Path) -> None:
        from timetoalign.loader.tabular import SoloLoader

        archive = pack(tmp_path / "solo.zip", {f"pp/{SOLO.name}": SOLO})
        archived = SoloLoader.from_file(member(archive, f"pp/{SOLO.name}"))
        on_disk = SoloLoader.from_file(SOLO)
        assert archived.events.table.equals(on_disk.events.table)
        same_timeline(archived, on_disk)

    def test_json_loader(self, tmp_path: Path) -> None:
        from timetoalign.loader.format import JsonLoader

        data = AUDIO / "hard_techno" / "dj_studio_data.json"
        archive = pack(tmp_path / "dj.zip", {f"set/{data.name}": data})
        archived = JsonLoader.from_file(member(archive, f"set/{data.name}"))
        on_disk = JsonLoader.from_file(data)
        assert archived.keys() == on_disk.keys()
        assert len(on_disk.keys()) > 0
        for key in on_disk.keys():
            assert archived.get_table(key).equals(on_disk.get_table(key))

    def test_maps_loader(self, tmp_path: Path) -> None:
        from timetoalign.loader import MapsLoader
        from timetoalign.loader.score import VerovioLoader
        from timetoalign.timelines import DiscretePhysicalTimeline

        source, path = _both(
            tmp_path,
            {"maps/alignment.json": json.dumps(MAPS_ROWS)},
            "maps/alignment.json",
        )
        mei = tmp_path / "score.mei"
        mei.write_text(MAPS_MEI, encoding="utf-8")

        def bundle(loader: MapsLoader) -> dict[str, Any]:
            from timetoalign.core import TimeUnit
            from timetoalign.maps import SamplesToSeconds

            score = VerovioLoader.from_file(mei).create_timeline()
            performance = DiscretePhysicalTimeline(
                length=4 * 44100, unit=TimeUnit.samples, uid="performance"
            )
            performance.add_conversion_map(
                SamplesToSeconds(sample_rate=44100, uid="samples_to_seconds")
            )
            return loader.create_bundle(score, performance).to_dict()

        archived = MapsLoader.from_file(source)
        on_disk = MapsLoader.from_file(path)
        assert archived.rows == on_disk.rows == MAPS_ROWS
        assert bundle(archived) == bundle(on_disk)


# endregion

# region Silent WAV for headers


def test_wav_written_in_place_reads_as_on_disk(tmp_path: Path, cache_dir: Path) -> None:
    """The wave and RIFF readers read a member's header from its file object."""
    path = tmp_path / "silent.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(22050)
        handle.writeframes(b"\x00\x00\x00\x00" * 441)
    archive = pack(tmp_path / "wav.zip", {"a/silent.wav": path})
    loader = AudioLoader()
    archived = member(archive, "a/silent.wav")
    for read in (loader._load_with_wave, loader._load_with_riff_parser):
        from_archive, from_disk = read(archived), read(path)
        assert from_archive.source_path is archived
        assert dataclasses.replace(from_archive, source_path=None) == (
            dataclasses.replace(from_disk, source_path=None)
        )
        assert (from_disk.n_samples, from_disk.sample_rate) == (441, 22050)
    assert copies(cache_dir) == []


# endregion
