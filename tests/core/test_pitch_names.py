"""Pitch names in one spelling: accidentals written ``♯`` and ``♭``.

Validation logic: ``tests/core/README.md``, "Pitch names in one spelling".
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa
import pytest

from timetoalign.core.events import (
    SpecificPitch,
    canonical_pitch_name,
    canonical_pitch_names,
)

CASES = [
    # (input, output, (fifths, octave) read back, or None)
    ("Bb3", "B♭3", (-2, 3)),
    ("F#4", "F♯4", (6, 4)),
    ("F##4", "F♯♯4", (13, 4)),
    ("Bbb2", "B♭♭2", (-9, 2)),
    ("F#♯4", "F♯♯4", (13, 4)),
    ("C#", "C♯", None),
    ("C-1", "C-1", (0, -1)),
    ("Cb-1", "C♭-1", (-7, -1)),
    ("eb5", "e♭5", (-3, 5)),
    ("bb3", "b♭3", (-2, 3)),
    ("E♭5", "E♭5", (-3, 5)),
    ("C4", "C4", (0, 4)),
    ("", "", None),
    ("rest", "rest", None),
    ("Bb3\n", "B♭3\n", (-2, 3)),
    (" Bb3", " B♭3", (-2, 3)),
    ("Bb3 ", "B♭3 ", (-2, 3)),
    ("G3, Bb4", "G3, Bb4", None),
]


@pytest.mark.parametrize(
    ("name", "expected", "pitch"), CASES, ids=[repr(c[0]) for c in CASES]
)
def test_a_name_is_respelled_with_sharp_and_flat_signs(name, expected, pitch):
    assert canonical_pitch_name(name) == expected
    if pitch is not None:
        read = SpecificPitch.from_string(expected)
        assert (read.fifths, read.octave) == pitch


def test_the_column_form_respells_each_cell_as_the_function_does():
    names = pd.Series(["Bb3", None, "rest", "Bb3", "C-1", "F#4"], dtype=object)
    before = names.copy()

    respelled = canonical_pitch_names(names)

    assert respelled.tolist() == ["B♭3", None, "rest", "B♭3", "C-1", "F♯4"]
    for name, cell in zip(names, respelled):
        if name is not None:
            assert cell == canonical_pitch_name(name)
    pd.testing.assert_series_equal(names, before)


def test_a_value_that_is_not_a_string_is_refused():
    with pytest.raises(TypeError) as error:
        canonical_pitch_name(60)
    assert str(error.value) == "A pitch name is a string, not int: 60"


def test_the_column_form_respells_every_name_a_cell_lists():
    names = pd.Series(["G3,D4,Bb4,F#5", "rest", "Eb5", "G3, Bb4"])

    assert canonical_pitch_names(names, sep=",").tolist() == [
        "G3,D4,B♭4,F♯5",
        "rest",
        "E♭5",
        "G3, B♭4",
    ]


def _cells(column: pd.Series) -> list:
    """The column's values, every null as ``None``."""
    return [None if pd.isna(value) else value for value in column.tolist()]


DICTIONARY = pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))
COLUMN_TYPES = [
    (object, object),
    ("str", "str"),
    ("string[python]", "string[python]"),
    ("string[pyarrow]", "string[pyarrow]"),
    (pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.string())),
    ("category", pd.CategoricalDtype(["B♭3", "C4", "F♯4"])),
    (DICTIONARY, DICTIONARY),
]


@pytest.mark.parametrize(
    ("dtype", "returned"), COLUMN_TYPES, ids=[str(c[0]) for c in COLUMN_TYPES]
)
def test_the_column_form_keeps_the_column_type(dtype, returned):
    names = pd.Series(
        ["Bb3", None, "F#4", "C4", "Bb3"], dtype=dtype, index=range(10, 15), name="name"
    )
    before = names.copy()

    respelled = canonical_pitch_names(names)

    assert _cells(respelled) == ["B♭3", None, "F♯4", "C4", "B♭3"]
    assert respelled.dtype == pd.Series([], dtype=returned).dtype
    if dtype == "category":
        assert respelled.cat.categories.tolist() == ["B♭3", "C4", "F♯4"]
    assert respelled.index.tolist() == [10, 11, 12, 13, 14]
    assert respelled.name == "name"
    pd.testing.assert_series_equal(names, before)


def test_categories_that_respell_to_one_name_merge():
    order = pd.CategoricalDtype(["F#4", "Bb3", "B♭3"], ordered=True)
    names = pd.Series(["Bb3", "B♭3", None, "F#4"], dtype=order)

    respelled = canonical_pitch_names(names)

    assert _cells(respelled) == ["B♭3", "B♭3", None, "F♯4"]
    assert respelled.cat.categories.tolist() == ["F♯4", "B♭3"]
    assert respelled.cat.ordered is True
    assert respelled.cat.codes.tolist() == [1, 1, -1, 0]


def test_dictionary_values_that_respell_to_one_name_merge():
    dtype = pd.ArrowDtype(pa.dictionary(pa.int8(), pa.string()))
    names = pd.Series(["Bb3", "B♭3", None, "F#4"], dtype=dtype)

    respelled = canonical_pitch_names(names)

    assert respelled.dtype == dtype
    column = pa.array(respelled.array)
    assert column.dictionary.to_pylist() == ["B♭3", "F♯4"]
    assert column.indices.to_pylist() == [0, 0, None, 1]


def test_each_chunk_of_a_dictionary_column_is_respelled():
    chunks = pa.chunked_array(
        [
            pa.array(["Bb3", None]).dictionary_encode(),
            pa.array(["F#4", "Bb3"]).dictionary_encode(),
        ]
    )
    names = pd.Series(pd.arrays.ArrowExtensionArray(chunks))

    respelled = canonical_pitch_names(names)

    assert _cells(respelled) == ["B♭3", None, "F♯4", "B♭3"]
    assert respelled.dtype == names.dtype
