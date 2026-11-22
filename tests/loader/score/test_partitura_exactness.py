"""Exact Partitura coordinates and flattened timeline semantics."""

from __future__ import annotations

from fractions import Fraction

import partitura as pt
import partitura.score as pts
import pytest

from timetoalign.loader.score import PartituraLoader
from timetoalign.testdata import ensure_data

CHOPIN_XML = ensure_data("vienna_1x22") / "Chopin_op10_no3.musicxml"
BEETHOVEN_XML = ensure_data("score") / "beethoven_woo71" / "WoO71.musicxml"
# Divisions per quarter, as each file's <divisions> elements state them.
DIVISIONS = {CHOPIN_XML: 16, BEETHOVEN_XML: 72}


def _pair(value: dict[str, object]) -> Fraction:
    """Return the exact Fraction carried by a coordinate struct."""
    numerator = value["numerator"]
    denominator = value["denominator"]
    assert isinstance(numerator, int)
    assert isinstance(denominator, int)
    return Fraction(numerator, denominator)


def _assert_coordinate_structs_are_exact(rows: list[dict[str, object]]) -> None:
    """Require a pair on every populated coordinate struct."""
    for row in rows:
        for name in ("start", "end", "duration"):
            value = row[name]
            if value is None:
                continue
            assert isinstance(value, dict)
            exact = _pair(value)
            assert value["value"] == float(exact)


def test_flattened_partitura_timeline_preserves_instant_nulls() -> None:
    """Merged score events retain exact coordinates and null instant bounds."""
    loader = PartituraLoader.from_file(CHOPIN_XML)
    timeline = loader.create_timeline(uid="score", flatten=True)
    rows = timeline._events.table.to_pylist()

    assert len(rows) == 547
    _assert_coordinate_structs_are_exact(rows)
    for row in rows:
        if row["temporal_type"] == "instant":
            assert row["end"] is None
            assert row["duration"] is None

    assert sum(row["temporal_type"] == "instant" for row in rows) == 31
    assert sum(row["end"] is not None for row in rows) == 516
    assert sum(row["duration"] is not None for row in rows) == 516


def test_chopin_division_positions_are_exact() -> None:
    """Dotted Partitura figures use exact integer-division quarter values."""
    loader = PartituraLoader.from_file(CHOPIN_XML)
    notes = loader.store.notes.table.to_pylist()
    part = pt.load_score(str(CHOPIN_XML), force_note_ids=True).parts[0]
    partitura_notes = [
        obj
        for obj in part.iter_all(include_subclasses=True)
        if isinstance(obj, (pts.Note, pts.Rest, pts.GraceNote))
    ]

    dotted_index = next(
        index
        for index, obj in enumerate(partitura_notes)
        if obj.start.t == 88 and obj.duration == 12
    )
    dotted = notes[dotted_index]
    assert _pair(dotted["start"]) == Fraction(11, 2)
    assert _pair(dotted["duration"]) == Fraction(3, 4)

    triplet_index = next(
        index
        for index, obj in enumerate(partitura_notes)
        if obj.start.t == 12 and obj.duration == 4
    )
    triplet = notes[triplet_index]
    assert _pair(triplet["start"]) == Fraction(3, 4)
    assert _pair(triplet["duration"]) == Fraction(1, 4)

    # The file states 16 divisions per quarter and its first note at
    # division 0, so a note at division t starts at exactly t/16 quarters.
    for index in (dotted_index, triplet_index):
        exact = _pair(notes[index]["start"])
        assert exact == Fraction(partitura_notes[index].start.t, DIVISIONS[CHOPIN_XML])


@pytest.mark.slow
def test_partitura_exact_coordinates_are_the_file_divisions() -> None:
    """Every start is its division count over the file's divisions per quarter."""
    for path, divisions in DIVISIONS.items():
        loader = PartituraLoader.from_file(path)
        part = pt.load_score(str(path), force_note_ids=True).parts[0]
        partitura_events = [
            obj
            for obj in part.iter_all(include_subclasses=True)
            if isinstance(obj, (pts.Note, pts.Rest, pts.GraceNote))
        ]
        for row, obj in zip(
            loader.store.notes.table.to_pylist(), partitura_events, strict=True
        ):
            assert _pair(row["start"]) == Fraction(obj.start.t, divisions)
