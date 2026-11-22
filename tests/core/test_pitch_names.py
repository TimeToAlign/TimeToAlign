"""Pitch names in one spelling: accidentals written ``♯`` and ``♭``.

Validation logic: ``tests/core/README.md``, "Pitch names in one spelling".
"""

from __future__ import annotations

import pandas as pd
import pytest

from timetoalign.core.events import (
    SpecificPitch,
    canonical_pitch_name,
    canonical_pitch_names,
)

CASES = [
    # (input, output, (step, alter, octave) read back, or None)
    ("Bb3", "B♭3", ("B", -1, 3)),
    ("F#4", "F♯4", ("F", 1, 4)),
    ("F##4", "F♯♯4", ("F", 2, 4)),
    ("Bbb2", "B♭♭2", ("B", -2, 2)),
    ("F#♯4", "F♯♯4", ("F", 2, 4)),
    ("C#", "C♯", None),
    ("C-1", "C-1", ("C", 0, -1)),
    ("Cb-1", "C♭-1", ("C", -1, -1)),
    ("eb5", "e♭5", ("E", -1, 5)),
    ("bb3", "b♭3", ("B", -1, 3)),
    ("E♭5", "E♭5", ("E", -1, 5)),
    ("C4", "C4", ("C", 0, 4)),
    ("", "", None),
    ("rest", "rest", None),
]


@pytest.mark.parametrize(
    ("name", "expected", "pitch"), CASES, ids=[c[0] or "empty" for c in CASES]
)
def test_a_name_is_respelled_with_sharp_and_flat_signs(name, expected, pitch):
    assert canonical_pitch_name(name) == expected
    if pitch is not None:
        read = SpecificPitch.from_string(expected)
        assert (read.step, read.alter, read.octave) == pitch


def test_the_column_form_respells_each_cell_as_the_function_does():
    names = pd.Series(["Bb3", None, "rest", "Bb3", "C-1", "F#4"], dtype=object)
    before = names.copy()

    respelled = canonical_pitch_names(names)

    assert respelled.tolist() == ["B♭3", None, "rest", "B♭3", "C-1", "F♯4"]
    for name, cell in zip(names, respelled):
        if name is not None:
            assert cell == canonical_pitch_name(name)
    pd.testing.assert_series_equal(names, before)


def test_the_column_form_respells_every_name_a_cell_lists():
    names = pd.Series(["G3,D4,Bb4,F#5", "rest", "Eb5"])

    assert canonical_pitch_names(names, sep=",").tolist() == [
        "G3,D4,B♭4,F♯5",
        "rest",
        "E♭5",
    ]
