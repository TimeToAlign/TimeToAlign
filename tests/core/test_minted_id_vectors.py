"""Tests of the minter against the minted-ID vectors TimeToShare ships."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from timetoshare.validate import data_root

from timetoalign.core.ids import IdMinter, _mix64

VECTORS_PATH: Path = data_root() / "fixtures" / "ids" / "minted_ids.json"


def _vectors() -> dict[str, Any]:
    """Read the vectors, failing with the path where the file is absent."""
    if not VECTORS_PATH.is_file():
        pytest.fail(f"the minted-ID vectors are missing: {VECTORS_PATH}")
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


VECTORS = _vectors()


def test_the_file_states_seeds_as_strings_and_lengths_as_integers() -> None:
    for entry in VECTORS["minters"]:
        assert isinstance(entry["seed"], str)
        assert isinstance(entry["length"], int)
    for entry in VECTORS["later_ids"]:
        assert isinstance(entry["seed"], str)
        assert isinstance(entry["counter"], int)
    for pair in VECTORS["mix64"]:
        assert isinstance(pair["input"], str)
        assert isinstance(pair["output"], str)


@pytest.mark.parametrize(
    "pair", VECTORS["mix64"], ids=[pair["input"] for pair in VECTORS["mix64"]]
)
def test_mix64(pair: dict[str, str]) -> None:
    assert _mix64(int(pair["input"])) == int(pair["output"])


@pytest.mark.parametrize(
    "entry",
    VECTORS["minters"],
    ids=[f"{entry['seed']}-L{entry['length']}" for entry in VECTORS["minters"]],
)
def test_minter(entry: dict[str, Any]) -> None:
    minter = IdMinter(int(entry["seed"]), length=entry["length"])
    assert [minter.mint(prefix) for prefix in entry["prefixes"]] == entry["ids"]


@pytest.mark.parametrize(
    "entry",
    VECTORS["minters"],
    ids=[f"{entry['seed']}-L{entry['length']}" for entry in VECTORS["minters"]],
)
def test_batch_equals_one_by_one(entry: dict[str, Any]) -> None:
    prefix = entry["prefixes"][0]
    count = len(entry["ids"])
    batch = IdMinter(int(entry["seed"]), length=entry["length"]).mint_many(
        prefix, count
    )
    twin = IdMinter(int(entry["seed"]), length=entry["length"])
    assert batch.to_pylist() == [twin.mint(prefix) for _ in range(count)]
    assert batch[0].as_py() == entry["ids"][0]


@pytest.mark.parametrize(
    "entry",
    VECTORS["later_ids"],
    ids=[f"{entry['seed']}-L{entry['length']}" for entry in VECTORS["later_ids"]],
)
def test_later_id(entry: dict[str, Any]) -> None:
    minter = IdMinter(int(entry["seed"]), length=entry["length"])
    minter.mint_many(entry["prefix"], entry["counter"] - 1)
    assert minter.mint(entry["prefix"]) == entry["id"]
