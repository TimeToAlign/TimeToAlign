"""The IDs a loader mints: one seeded minter per load (validation logic: README.md)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.loader.minting import minted
from timetoalign.loader.score import Ms3Loader
from timetoalign.loader.tabular import CsvLoader
from timetoalign.testdata import ensure_data

WOO71_NOTES = ensure_data("score") / "beethoven_woo71" / "WoO71.notes.tsv"

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


def test_one_minter_per_load(tmp_path: Path, csv: Path) -> None:
    other = _write(tmp_path, "other.csv", CSV.replace("0.25", "0.5"))
    together = CsvLoader(seed=7).load(csv, other)
    one_by_one = CsvLoader(seed=7).load(csv).load(other)
    # Each source is one batch per event type: its Events, then its Note.
    ids = minted(7, "event", "event", "note", "event", "event", "note")
    expected = [ids[0], ids[2], ids[1], ids[3], ids[5], ids[4]]
    assert _ids(together) == expected
    assert _ids(one_by_one) == expected


def test_ms3_notes_are_the_minters_first_ids() -> None:
    loader = Ms3Loader.from_file(WOO71_NOTES, seed=0)
    ids = _ids(loader)
    assert ids == minted(0, *["note"] * len(ids))


def test_the_seed_is_the_bytes_not_the_path(csv: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy" / "renamed.csv"
    copy.parent.mkdir()
    shutil.copy(csv, copy)
    assert _ids(CsvLoader.from_file(copy)) == _ids(CsvLoader.from_file(csv))
