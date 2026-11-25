"""Regression tests for structural pitch protocol discovery."""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import timetoalign.core as core
from timetoalign.core.enums import TimeUnit
from timetoalign.core.events import (
    EnharmonicPitch,
    EnharmonicPitchClass,
    EnharmonicPitchField,
    MidiPitchField,
    SpecificPitch,
    SpecificPitchClassField,
    SpecificPitchField,
)
from timetoalign.core.fields import SemanticField
from timetoalign.core.protocols import (
    EnharmonicPitchLike,
    GenericPitchLike,
    SpecificPitchClassLike,
    SpecificPitchLike,
)
from timetoalign.core.time import Coordinate, CoordinateField, DurationField
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.loader.score.stores.notes import NoteEventData
from timetoalign.testdata import ensure_data


@pytest.fixture(scope="module")
def vienna_chopin_notes() -> NoteEventData:
    """Load the Vienna Chopin notes table through the public score loader."""
    path = ensure_data("vienna_1x22") / "ms3" / "chopin_op10_no3.notes.tsv"
    return Ms3Loader.from_file(path).get_events()


def test_generic_pitch_discovery_returns_only_pitch_fields(
    vienna_chopin_notes: NoteEventData,
) -> None:
    """Pitch discovery excludes temporal fields on a real notes table."""
    fields = vienna_chopin_notes.get_fields_satisfying(GenericPitchLike)

    assert {field.name for field in fields} == {"midi", "specific_pitch"}
    assert not any(isinstance(field, CoordinateField) for field in fields)
    assert not any(isinstance(field, DurationField) for field in fields)


@pytest.mark.parametrize(
    "protocol",
    [
        GenericPitchLike,
        SpecificPitchClassLike,
        EnharmonicPitchLike,
        SpecificPitchLike,
    ],
)
def test_coordinate_is_not_pitch_like(protocol: type) -> None:
    """A coordinate scalar satisfies none of the pitch protocols."""
    coordinate = Coordinate(Fraction(1, 2), TimeUnit.quarters)

    assert not isinstance(coordinate, protocol)


def test_pitch_specificity_ladder() -> None:
    """Spelled and enharmonic pitches retain their protocol relationships."""
    specific = SpecificPitch(step="C", alter=1, octave=4)
    enharmonic = EnharmonicPitch(midi_number=61)

    assert isinstance(specific, SpecificPitchLike)
    assert isinstance(specific, EnharmonicPitchLike)
    assert isinstance(specific, GenericPitchLike)
    assert isinstance(enharmonic, EnharmonicPitchLike)
    assert isinstance(enharmonic, GenericPitchLike)
    assert not isinstance(enharmonic, SpecificPitchLike)


def test_pitch_like_is_not_exported_from_core() -> None:
    """The removed empty pitch protocol is not importable from core."""
    assert "PitchLike" not in vars(core)


# ---------------------------------------------------------------------------
# A pitch field without data has no pitch columns
# ---------------------------------------------------------------------------

_COLUMNS: dict[type[SemanticField[Any]], tuple[str, ...]] = {
    SpecificPitchClassField: ("pitch_class", "fifths"),
    EnharmonicPitchField: ("pitch_class", "octave"),
    MidiPitchField: ("pitch_class", "octave"),
    SpecificPitchField: ("pitch_class", "midi_number", "fifths"),
}

_ONE_ROW: dict[type[SemanticField[Any]], dict[str, Any]] = {
    SpecificPitchClassField: {"step": "B", "alter": -1},
    EnharmonicPitchField: {"midi_number": 58},
    MidiPitchField: {"midi_number": 58},
    SpecificPitchField: {"step": "B", "alter": -1, "octave": 3, "cents": 0.0},
}


def _without_data(field_cls: type[SemanticField[Any]]) -> list[SemanticField[Any]]:
    """The blueprint and the schema-only field of *field_cls*."""
    return [
        field_cls(source_fields="p"),
        field_cls.from_field(pa.field("p", field_cls.pa_schema)),
    ]


@pytest.mark.parametrize("field_cls", list(_COLUMNS))
def test_a_pitch_field_without_data_has_no_pitch_columns(
    field_cls: type[SemanticField[Any]],
) -> None:
    """Each column is a missing attribute; ``hasattr`` answers ``False``."""
    for field in _without_data(field_cls):
        for name in _COLUMNS[field_cls]:
            assert hasattr(field, name) is False
            with pytest.raises(AttributeError) as caught:
                getattr(field, name)
            assert str(caught.value) == (
                f"'StructField' object has no attribute '{name}'"
            )


@pytest.mark.parametrize("field_cls", list(_COLUMNS))
def test_the_methods_of_a_pitch_field_without_data_still_refuse_it(
    field_cls: type[SemanticField[Any]],
) -> None:
    """``get()`` and ``convert_to()`` raise the ``TypeError`` of ``to_pyarrow()``."""
    for field in _without_data(field_cls):
        for call in (field.get, lambda: field.convert_to(EnharmonicPitchClass)):
            with pytest.raises(TypeError) as caught:
                call()
            assert str(caught.value) == (
                "Cannot convert schema-only field 'p' to PyArrow array"
            )


@pytest.mark.parametrize("field_cls", list(_COLUMNS))
def test_a_pitch_field_with_rows_keeps_its_pitch_columns(
    field_cls: type[SemanticField[Any]],
) -> None:
    """A field over one row answers every column."""
    schema = field_cls.pa_schema
    field = field_cls.from_field(
        (pa.array([_ONE_ROW[field_cls]], type=schema), pa.field("p", schema))
    )

    assert [hasattr(field, name) for name in _COLUMNS[field_cls]] == [True] * len(
        _COLUMNS[field_cls]
    )


_PROTOCOL_SCRIPT = """
import inspect, warnings
import pyarrow as pa
warnings.simplefilter("ignore")
import timetoalign.core.protocols as pr
from timetoalign.core.events import (
    EnharmonicPitchField, MidiPitchField, SpecificPitchClassField, SpecificPitchField,
)

protocols = sorted(
    (obj for _, obj in inspect.getmembers(pr, inspect.isclass)
     if getattr(obj, "_is_runtime_protocol", False) and obj.__module__ == pr.__name__),
    key=lambda p: p.__name__,
)
for cls in (SpecificPitchClassField, EnharmonicPitchField, MidiPitchField, SpecificPitchField):
    for label, field in (
        ("blueprint", cls(source_fields="p")),
        ("schema-only", cls.from_field(pa.field("p", cls.pa_schema))),
    ):
        print(cls.__name__, label, sorted(p.__name__ for p in protocols if isinstance(field, p)))
"""


def test_protocol_checks_of_a_pitch_field_without_data_answer_under_every_hash_seed(
    tmp_path: Path,
) -> None:
    """Runtime protocol checks answer, identically, whatever the hash seed."""
    script = tmp_path / "protocols.py"
    script.write_text(_PROTOCOL_SCRIPT)
    outputs = []
    for seed in ("0", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        done = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(done.stdout)
    assert outputs[0] == outputs[1]
    assert outputs[0].splitlines() == [
        f"{name} {label} []"
        for name in (
            "SpecificPitchClassField",
            "EnharmonicPitchField",
            "MidiPitchField",
            "SpecificPitchField",
        )
        for label in ("blueprint", "schema-only")
    ]
