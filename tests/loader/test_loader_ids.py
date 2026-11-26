"""The IDs a loader mints: one seeded minter per load (validation logic: README.md)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from tests.loader.minting import minted, shape
from timetoalign.core import TimeUnit
from timetoalign.core.ids import IdMinter, reset_content_minters, seed_from
from timetoalign.loader._sources import source_digest
from timetoalign.loader.alignment import Ieee1599Loader, MpmLoader, TiliaJsonLoader
from timetoalign.loader.graphical.loader import GraphicalLoader
from timetoalign.loader.midi import ScoreMidiLoader
from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.score.stores import (
    AnnotationEventData,
    ControlEventData,
    MeasureData,
    NoteEventData,
)
from timetoalign.loader.tabular import CsvLoader
from timetoalign.testdata import ensure_data

WOO71_NOTES = ensure_data("score") / "beethoven_woo71" / "WoO71.notes.tsv"
MUSETTE = ensure_data("score") / "couperin_concerts"
BEETHOVEN_MIDI = ensure_data("midi") / "score" / "beethoven_op18.mid"
CURZON = ensure_data("mpm_toolbox") / "MPRproject_1971Curzon_VariationXIV"
CURZON_STEM = "Beethoven_op35_1971Curzon_Var14only"

CSV = "start,end,event_type\n0.25,1.0,Event\n1.0,2.0,Note\n2.0,3.0,Event\n"


def _write(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def _ids(loader: CsvLoader | Ms3Loader) -> list[str]:
    """The ids of the loader's events, in row order."""
    return loader.events.table.column("id").to_pylist()


@pytest.fixture
def csv(tmp_path: Path) -> Path:
    return _write(tmp_path, "events.csv", CSV)


def test_a_seed_gives_the_minters_ids_in_build_order(csv: Path) -> None:
    loader = CsvLoader.from_file(csv, seed=7)
    # One batch per event type, in order of first row: both Events, then the Note.
    event_1, event_2, note, timeline = minted(7, "event", "event", "note", "cpt")
    assert _ids(loader) == [event_1, note, event_2]
    assert loader.create_timeline().id == timeline


def test_a_fresh_loader_mints_the_same_ids(csv: Path) -> None:
    once, again = CsvLoader.from_file(csv), CsvLoader.from_file(csv)
    assert _ids(once) == _ids(again)
    assert once.create_timeline().id == again.create_timeline().id
    cleared = CsvLoader.from_file(csv)
    cleared.clear()
    cleared.load(csv)
    assert _ids(cleared) == _ids(once)


def test_another_source_mints_other_ids(csv: Path, tmp_path: Path) -> None:
    other = _write(tmp_path, "other.csv", CSV.replace("0.25", "0.5"))
    once, changed = _ids(CsvLoader.from_file(csv)), _ids(CsvLoader.from_file(other))
    assert len(set(once) | set(changed)) == 6


def test_stated_ids_are_kept(tmp_path: Path) -> None:
    stated = _write(
        tmp_path, "stated.csv", "id,start,end\na,0.25,1.0\nb,1.0,2.0\nc,2.0,3.0\n"
    )
    loader = CsvLoader.from_file(stated, seed=7)
    assert _ids(loader) == ["a", "b", "c"]
    assert loader.create_timeline().id == minted(7, "cpt")[0]


def test_every_load_call_seeds_a_new_minter(tmp_path: Path, csv: Path) -> None:
    other = _write(tmp_path, "other.csv", CSV.replace("0.25", "0.5"))
    third = _write(tmp_path, "third.csv", CSV.replace("0.25", "0.75"))
    together = CsvLoader(seed=7).load(csv, other)
    one_by_one = CsvLoader(seed=7).load(csv).load(other)
    # Each source is one batch per event type: its Events, then its Note.
    ids = minted(7, "event", "event", "note", "event", "event", "note")
    assert _ids(together) == [ids[0], ids[2], ids[1], ids[3], ids[5], ids[4]]
    # The second call's minter is seeded from the first's seed and its source.
    second = minted(seed_from(7, source_digest(other)), "event", "event", "note")
    assert _ids(one_by_one) == [ids[0], ids[2], ids[1], second[0], second[2], second[1]]
    # A later source counts: the same first call, another second one.
    with_third = _ids(CsvLoader(seed=7).load(csv).load(third))
    assert with_third[:3] == _ids(one_by_one)[:3]
    assert set(with_third[3:]).isdisjoint(_ids(one_by_one)[3:])


def test_ms3_notes_are_the_minters_first_ids() -> None:
    loader = Ms3Loader.from_file(WOO71_NOTES, seed=0)
    ids = _ids(loader)
    assert ids == minted(0, *["note"] * len(ids))


def test_the_seed_is_the_bytes_not_the_path(csv: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy" / "renamed.csv"
    copy.parent.mkdir()
    shutil.copy(csv, copy)
    assert _ids(CsvLoader.from_file(copy)) == _ids(CsvLoader.from_file(csv))


def test_the_configuration_counts() -> None:
    by_mido = ScoreMidiLoader.from_file(BEETHOVEN_MIDI, parser="mido")
    by_partitura = ScoreMidiLoader.from_file(BEETHOVEN_MIDI)
    assert set(_ids(by_mido)).isdisjoint(_ids(by_partitura))


def test_the_unit_counts(csv: Path) -> None:
    seconds = CsvLoader.from_file(csv)
    quarters = CsvLoader.from_file(csv, unit=TimeUnit.quarters)
    assert set(_ids(seconds)).isdisjoint(_ids(quarters))


KHOMUS = next((ensure_data("ieee1599") / "Khorus Music").glob("*.xml"))
BRUCKNER = (
    ensure_data("score")
    / "bruckner5_scherzo"
    / "harnoncourt"
    / "Bruckner5_Scherzo.json"
)


def test_an_xml_string_and_element_seed_from_their_text() -> None:
    text = KHOMUS.read_text(encoding="utf-8-sig")
    once = Ieee1599Loader().load_string(text).timeline_uids
    assert Ieee1599Loader().load_string(text).timeline_uids == once
    element = Ieee1599Loader().load_element(ElementTree.fromstring(text)).timeline_uids
    again = Ieee1599Loader().load_element(ElementTree.fromstring(text)).timeline_uids
    assert element == again
    changed = Ieee1599Loader().load_string(
        text.replace("</ieee1599>", "<!-- --></ieee1599>")
    )
    assert set(changed.timeline_uids).isdisjoint(once)


def _bruckner_ids(data: dict) -> list[str]:
    """The timeline IDs TiLiA's JSON *data* gives through ``load_dict``."""
    return [
        timeline.id for timeline in TiliaJsonLoader().load_dict(data).create_timelines()
    ]


def test_a_dict_seeds_from_its_json() -> None:
    data = json.loads(BRUCKNER.read_text(encoding="utf-8"))
    once = _bruckner_ids(data)
    assert _bruckner_ids(json.loads(BRUCKNER.read_text(encoding="utf-8"))) == once
    changed = {**data, "title": f"{data.get('title', '')} (copy)"}
    assert set(_bruckner_ids(changed)).isdisjoint(once)


def _curzon(folder: Path, *, changed: bool = False) -> Path:
    """The Curzon MPR triple copied into *folder*, its MPM one byte longer if *changed*."""
    for suffix in (".mpr", ".msm", ".mpm"):
        shutil.copy(
            CURZON / f"{CURZON_STEM}{suffix}", folder / f"{CURZON_STEM}{suffix}"
        )
    (folder / "audio").symlink_to(CURZON / "audio", target_is_directory=True)
    if changed:
        with (folder / f"{CURZON_STEM}.mpm").open("a", encoding="utf-8") as handle:
            handle.write("\n")
    return folder / f"{CURZON_STEM}.mpr"


def test_a_folder_seeds_from_the_files_read(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    same, changed = _curzon(tmp_path / "a"), _curzon(tmp_path / "b", changed=True)
    once = MpmLoader.from_file(CURZON / f"{CURZON_STEM}.mpr").create_bundle()
    again = MpmLoader.from_file(same).create_bundle()
    other = MpmLoader.from_file(changed).create_bundle()
    assert again.timeline_ids == once.timeline_ids
    assert set(other.timeline_ids).isdisjoint(once.timeline_ids)


def test_from_parquet_seeds_from_the_file(csv: Path, tmp_path: Path) -> None:
    path = tmp_path / "events.parquet"
    CsvLoader.from_file(csv).to_parquet(path)
    once = CsvLoader.from_parquet(path).create_timeline().id
    assert CsvLoader.from_parquet(path).create_timeline().id == once
    assert shape(once) == "cpt"
    seeded = CsvLoader.from_parquet(path, seed=7).create_timeline().id
    assert seeded == minted(7, "cpt")[0]


def test_create_cmap_draws_from_the_loaders_minter(csv: Path) -> None:
    loader = CsvLoader.from_file(csv, seed=7)
    *_, cmap_id = minted(7, "event", "event", "note", "cmap")
    assert loader.create_cmap("start", "end").id == cmap_id


def _graphical(tmp_path: Path, y: int) -> GraphicalLoader:
    """A GraphicalLoader of one blank image and one horizontal segment at *y*."""
    from PIL import Image

    image = tmp_path / "page.png"
    if not image.exists():
        Image.new("RGB", (40, 30), "white").save(image)
    loader = GraphicalLoader()
    loader.add_horizontal_segment(loader.add_image(image), x0=0, x1=30, y=y)
    return loader


def test_a_graphical_loader_seeds_from_what_is_added(tmp_path: Path) -> None:
    once = _graphical(tmp_path, 10).create_timeline().id
    assert shape(once) == "dgt"
    assert _graphical(tmp_path, 10).create_timeline().id == once
    assert _graphical(tmp_path, 20).create_timeline().id != once


@pytest.mark.parametrize(
    "store, rows",
    [
        (NoteEventData, [{"event_type": "Note", "quarterbeats": 0, "midi": 60}]),
        (MeasureData, [{"event_type": "Measure", "quarterbeats": 0, "mc": 1}]),
        (
            ControlEventData,
            [{"event_type": "Control", "quarterbeats": 0, "subtype": "Tempo"}],
        ),
        (
            AnnotationEventData,
            [{"event_type": "Annotation", "quarterbeats": 0, "text": "A"}],
        ),
    ],
)
def test_score_stores_mint_after_the_table_is_built(store, rows) -> None:
    stated = store.from_dicts([*rows, {**rows[0], "id": "given"}], minter=IdMinter(7))
    prefix = rows[0]["event_type"].lower()
    assert stated.table.column("id").to_pylist() == [minted(7, prefix)[0], "given"]
    # Without a minter the seed is the content: another start, other IDs.
    reset_content_minters()
    first = store.from_dicts(rows).table.column("id").to_pylist()
    reset_content_minters()
    moved = [{**rows[0], "quarterbeats": 1}]
    assert store.from_dicts(moved).table.column("id").to_pylist() != first
    reset_content_minters()
    assert store.from_dicts(rows).table.column("id").to_pylist() == first


def _mpm_built() -> dict[str, Any]:
    """Every ID of the bundle a fresh MPM loader builds from the Curzon project."""
    bundle = MpmLoader.from_file(CURZON / f"{CURZON_STEM}.mpr").create_bundle()
    return {
        "manifest": bundle.to_dict(),
        "events": {
            timeline.id: timeline.events.table.column("id").to_pylist()
            for timeline in bundle.timelines.values()
        },
        "claims": [claim.id for claim in bundle.get_match_claims()],
    }


def _ms3_built() -> dict[str, Any]:
    """Every ID of the timeline a fresh ms3 loader builds from the Musette's measures."""
    loader = Ms3Loader.from_file(
        MUSETTE / "c05n05_musete.measures.tsv", MUSETTE / "c05n05_musete.notes.tsv"
    )
    timeline = loader.create_timeline()
    loader.create_stated_flow_map(timeline)
    return {
        "manifest": timeline.to_dict(),
        "events": timeline.events.table.column("id").to_pylist(),
    }


def _id_shapes(value: Any) -> set[str]:
    """The shapes of every minted ``id`` a manifest states, at any depth."""
    if isinstance(value, dict):
        stated = value.get("id")
        found = (
            {shape(stated.removeprefix("#"))} if stated not in (None, "@id") else set()
        )
        return found.union(*(_id_shapes(member) for member in value.values()))
    if isinstance(value, list):
        return set().union(*(_id_shapes(member) for member in value))
    return set()


def test_a_fresh_loader_builds_everything_with_the_same_ids() -> None:
    mpm = _mpm_built()
    # The first build advanced every minter the process keeps for objects built
    # in code; a loader draws from none of them.
    assert _mpm_built() == mpm
    assert _id_shapes(mpm["manifest"]) == {
        "group",
        "cmap",
        "score:clt",
        "score:dlt",
        "perf:cpt",
        "perf:dpt",
        "perf:dgt",
    }
    assert {shape(key): len(ids) for key, ids in mpm["events"].items()} == {
        "score:clt": 251,
        "score:dlt": 499,
        "perf:cpt": 251,
        "perf:dpt": 251,
        "perf:dgt": 0,
    }
    assert len(set(mpm["claims"])) == 251

    ms3 = _ms3_built()
    assert _ms3_built() == ms3
    assert _id_shapes(ms3["manifest"]) == {
        "clt",
        "cmap",
        "measure_map",
        "jump",
        "jump_target",
        "break",
        "region",
        "flow",
    }
    assert len(set(ms3["events"])) == 530
    assert len(ms3["manifest"]["measure_maps"]) == 1
