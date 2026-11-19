"""Ms3Loader companion discovery in the ms3 and DCML corpus layouts.

Validation logic: ``tests/loader/score/README.md``, "MS3 facet discovery".
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.loader.score.ms3 import Ms3Loader

MEASURES = (
    "mc\tmn\tquarterbeats\tact_dur\ttimesig\n"
    "1\t1\t0\t1\t4/4\n"
    "2\t2\t4\t1\t4/4\n"
    "3\t3\t8\t1\t4/4\n"
)
NOTES = (
    "mc\tmn\tquarterbeats\tduration_qb\tduration\tmidi\tname\toctave\n"
    "3\t3\t9\t1.0\t1/4\t60\tC4\t4\n"
)
# The last label's duration_qb is empty: its span can only come from the
# measures facet.
HARMONIES = (
    "mc\tmn\tquarterbeats\tduration_qb\tlabel\n" "1\t1\t0\t6.0\tI\n" "2\t2\t6\t\tV\n"
)
FACETS = {"notes": NOTES, "measures": MEASURES, "harmonies": HARMONIES}


def _write_piece(root: Path, layout: str, piece: str = "p") -> dict[str, Path]:
    """Write the piece's three facets in *layout* and return their paths."""
    paths = {}
    for facet, text in FACETS.items():
        directory = root / facet if layout == "dcml" else root
        directory.mkdir(parents=True, exist_ok=True)
        paths[facet] = directory / f"{piece}.{facet}.tsv"
        paths[facet].write_text(text, encoding="utf-8")
    return paths


def _load(path: Path, **kwargs) -> Ms3Loader:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return Ms3Loader.from_file(path, **kwargs)


def _spans(loader: Ms3Loader) -> list[Fraction]:
    durations = loader.store.annotations.table.column("duration").to_pylist()
    return [Fraction(d["numerator"], d["denominator"]) for d in durations]


@pytest.mark.parametrize("layout", ["flat", "dcml"])
def test_notes_file_discovers_every_present_facet(tmp_path, layout):
    paths = _write_piece(tmp_path, layout)

    loader = _load(paths["notes"], auto_discover=True)
    timeline = loader.create_timeline()

    assert loader.sources == [
        paths["notes"].resolve(),
        paths["measures"].resolve(),
        paths["harmonies"].resolve(),
    ]
    assert isinstance(timeline.length.value, Fraction)
    assert timeline.length.value == Fraction(12)
    assert len(timeline.measure_map) == 3
    assert len(loader.store.notes) == 1
    assert _spans(loader) == [Fraction(6), Fraction(6)]


def test_measures_file_discovers_notes_and_harmonies_in_dcml_layout(tmp_path):
    paths = _write_piece(tmp_path, "dcml")

    loader = _load(paths["measures"], auto_discover=True)

    assert loader.sources == [
        paths["measures"].resolve(),
        paths["notes"].resolve(),
        paths["harmonies"].resolve(),
    ]
    assert len(loader.store.notes) == 1
    assert len(loader.store.annotations) == 2


@pytest.mark.parametrize("layout", ["flat", "dcml"])
def test_folded_and_unfolded_files_are_never_mixed(tmp_path, layout):
    folded = _write_piece(tmp_path, layout)
    unfolded = _write_piece(tmp_path, layout, piece="p_unfolded")

    assert Ms3Loader._discover_companions(folded["notes"]) == [
        folded["measures"].resolve(),
        folded["harmonies"].resolve(),
    ]
    assert Ms3Loader._discover_companions(unfolded["notes"]) == [
        unfolded["measures"].resolve(),
        unfolded["harmonies"].resolve(),
    ]


def test_a_facet_beside_the_source_is_taken_before_its_facet_directory(tmp_path):
    paths = _write_piece(tmp_path, "dcml")
    beside = paths["notes"].parent / "p.measures.tsv"
    beside.write_text(MEASURES, encoding="utf-8")

    assert Ms3Loader._discover_companions(paths["notes"]) == [
        beside.resolve(),
        paths["harmonies"].resolve(),
    ]


def test_no_companion_warns_naming_every_path_looked_at(tmp_path):
    notes = tmp_path / "notes" / "p.notes.tsv"
    notes.parent.mkdir()
    notes.write_text(NOTES, encoding="utf-8")

    with pytest.warns(UserWarning) as record:
        loader = Ms3Loader.from_file(notes, auto_discover=True)

    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith(
        f"auto_discover found no companion facet file for '{notes.resolve()}': "
        "none of these files exists: "
    )
    for facet in ("measures", "chords", "harmonies"):
        assert str(notes.parent / f"p.{facet}.tsv") in message
        assert str(tmp_path / facet / f"p.{facet}.tsv") in message
    assert message.endswith("It is loaded alone.")
    assert loader.sources == [notes.resolve()]
    assert len(loader.store.notes) == 1


def test_a_file_naming_no_facet_warns_that_it_states_none(tmp_path):
    notes = tmp_path / "p.tsv"
    notes.write_text(NOTES, encoding="utf-8")

    with pytest.warns(UserWarning, match="its file name states no facet") as record:
        Ms3Loader.from_file(notes, auto_discover=True)

    assert len(record) == 1


def test_without_auto_discover_nothing_is_looked_for(tmp_path):
    paths = _write_piece(tmp_path, "dcml")

    loader = _load(paths["notes"])

    assert loader.sources == [paths["notes"].resolve()]
