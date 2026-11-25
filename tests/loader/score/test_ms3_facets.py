"""Ms3Loader: companion discovery, empty note cells, pitch names, measures it cannot read.

Validation logic: ``tests/loader/score/README.md``, "MS3 facet discovery",
"MS3 empty note cells", "MS3 pitch names", "MS3 measures without a count or a
length" and "MS3 files that hold no table".
"""

from __future__ import annotations

import warnings
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.core import SpecificPitch
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


# Four notes with the four tie cells (empty, 0, 1, -1), then a spelling
# without an octave.
TIED_NOTES = (
    "mc\tmn\tquarterbeats\tduration_qb\tduration\ttied\tmidi\tname\toctave\n"
    "1\t1\t0\t1.0\t1/4\t\t62\tD4\t4\n"
    "1\t1\t1\t1.0\t1/4\t0\t64\tE4\t4\n"
    "1\t1\t2\t1.0\t1/4\t1\t65\tF4\t4\n"
    "1\t1\t3\t1.0\t1/4\t-1\t65\tF4\t4\n"
    "2\t2\t4\t1.0\t1/4\t\t60\tC\t\n"
)


def _tied_notes(tmp_path: Path):
    path = tmp_path / "p.notes.tsv"
    path.write_text(TIED_NOTES, encoding="utf-8")
    return _load(path).store.notes.table


def test_an_empty_tie_cell_is_no_tie_and_every_value_is_kept(tmp_path):
    table = _tied_notes(tmp_path)

    assert table.column("tied").to_pylist() == [None, 0, 1, -1, None]


def test_an_empty_octave_states_no_specific_pitch(tmp_path):
    table = _tied_notes(tmp_path)

    assert table.column("octave").to_pylist() == [4, 4, 4, 4, None]
    assert table.column("midi").to_pylist() == [62, 64, 65, 65, 60]
    assert table.column("specific_pitch").to_pylist() == [
        {"step": "D", "alter": 0, "octave": 4, "cents": 0.0},
        {"step": "E", "alter": 0, "octave": 4, "cents": 0.0},
        {"step": "F", "alter": 0, "octave": 4, "cents": 0.0},
        {"step": "F", "alter": 0, "octave": 4, "cents": 0.0},
        None,
    ]


# ASCII accidentals, a negative octave, a double sharp, an empty name cell.
NAMED_NOTES = (
    "mc\tmn\tquarterbeats\tduration_qb\tduration\tmidi\tname\toctave\n"
    "1\t1\t0\t1.0\t1/4\t58\tBb3\t3\n"
    "1\t1\t1\t1.0\t1/4\t66\tF#4\t4\n"
    "1\t1\t2\t1.0\t1/4\t0\tC-1\t-1\n"
    "1\t1\t3\t1.0\t1/4\t67\tF##4\t4\n"
    "2\t2\t4\t1.0\t1/4\t60\t\t4\n"
)


def test_names_are_spelled_with_sharp_and_flat_signs_and_an_empty_one_is_null(
    tmp_path,
):
    path = tmp_path / "p.notes.tsv"
    path.write_text(NAMED_NOTES, encoding="utf-8")
    table = _load(path).store.notes.table

    assert table.column("name").to_pylist() == ["B♭3", "F♯4", "C-1", "F♯♯4", None]
    assert table.column("midi").to_pylist() == [58, 66, 0, 67, 60]
    pitches = [
        None if row is None else SpecificPitch.from_row(row)
        for row in table.column("specific_pitch").to_pylist()
    ]
    # Line of fifths: B♭ = 5 - 7, F♯ = -1 + 7, C = 0, F♯♯ = -1 + 14.
    assert [None if p is None else (p.fifths, p.octave) for p in pitches] == [
        (-2, 3),
        (6, 4),
        (0, -1),
        (13, 4),
        None,
    ]


# Three 4/4 measures; each refusal case removes one cell (see "MS3 measures
# without a count or a length").
def _measures(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "p.measures.tsv"
    path.write_text(text, encoding="utf-8")
    return path


def _measure_map(path: Path) -> list[tuple[int, Fraction]]:
    timeline = _load(path).create_timeline()
    return [(m.count, m.actual_length) for m in timeline.measure_map]


def test_a_complete_measures_table_loads_its_counts_and_lengths(tmp_path):
    measures = _measure_map(_measures(tmp_path, MEASURES))

    assert measures == [(1, Fraction(4)), (2, Fraction(4)), (3, Fraction(4))]
    assert all(type(length) is Fraction for _, length in measures)


def test_duration_qb_alone_states_the_length(tmp_path):
    text = (
        "mc\tmn\tquarterbeats\tact_dur\tduration_qb\ttimesig\n"
        "1\t1\t0\t1\t4.0\t4/4\n"
        "2\t2\t4\t\t3.0\t4/4\n"
        "3\t3\t7\t1\t4.0\t4/4\n"
    )

    measures = _measure_map(_measures(tmp_path, text))

    assert measures == [(1, Fraction(4)), (2, Fraction(3)), (3, Fraction(4))]
    assert type(measures[1][1]) is Fraction


def test_a_row_without_mc_is_refused(tmp_path):
    text = MEASURES.replace("2\t2\t4\t1\t4/4\n", "\t2\t4\t1\t4/4\n")

    with pytest.raises(ValueError) as raised:
        _load(_measures(tmp_path, text))

    assert (
        str(raised.value) == "The measures table 'p.measures.tsv' states no mc in row 2"
    )


NO_LENGTH = (
    "The measures table 'p.measures.tsv' states no length of mc 2 (row 2): "
    "neither act_dur nor duration_qb holds a value"
)


@pytest.mark.parametrize(
    "text",
    [
        MEASURES.replace("2\t2\t4\t1\t4/4\n", "2\t2\t4\t\t4/4\n"),
        (
            "mc\tmn\tquarterbeats\tact_dur\tduration_qb\ttimesig\n"
            "1\t1\t0\t1\t4.0\t4/4\n"
            "2\t2\t4\t\t\t4/4\n"
            "3\t3\t8\t1\t4.0\t4/4\n"
        ),
    ],
    ids=["no_duration_qb_column", "both_cells_empty"],
)
def test_a_row_without_a_length_is_refused(tmp_path, text):
    with pytest.raises(ValueError) as raised:
        _load(_measures(tmp_path, text))

    assert str(raised.value) == NO_LENGTH


def test_a_length_that_cannot_be_read_is_refused(tmp_path):
    text = (
        "mc\tmn\tquarterbeats\tact_dur\tduration_qb\ttimesig\n"
        "1\t1\t0\t1\t4.0\t4/4\n"
        "2\t2\t4\t\tfour\t4/4\n"
        "3\t3\t8\t1\t4.0\t4/4\n"
    )

    with pytest.raises(ValueError) as raised:
        _load(_measures(tmp_path, text))

    assert str(raised.value) == (
        "The measures table 'p.measures.tsv' states no readable length of mc 2 "
        "(row 2)"
    )


def _holds_no_table(path: Path) -> str:
    return (
        f"Ms3Loader cannot read '{path}': the file holds no table (an ms3 TSV file "
        "states its columns in its first line)."
    )


@pytest.mark.parametrize(
    "name",
    ["p.notes.tsv", "p.chords.tsv", "p.harmonies.tsv", "p.tsv", "p.measures.tsv"],
)
def test_an_empty_file_is_refused_naming_it(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"")
    with pytest.raises(ValueError) as excinfo:
        Ms3Loader.from_file(path)
    assert str(excinfo.value) == _holds_no_table(path)


@pytest.mark.parametrize("auto_discover", [True, False])
def test_an_empty_measures_companion_is_refused_naming_it(tmp_path, auto_discover):
    harmonies = tmp_path / "p.harmonies.tsv"
    harmonies.write_text(HARMONIES, encoding="utf-8")
    Ms3Loader.from_file(harmonies, auto_discover=auto_discover)
    measures = tmp_path / "p.measures.tsv"
    measures.write_bytes(b"")
    with pytest.raises(ValueError) as excinfo:
        Ms3Loader.from_file(harmonies, auto_discover=auto_discover)
    assert str(excinfo.value) == _holds_no_table(measures)
