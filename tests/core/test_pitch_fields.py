"""A null pitch row stays null in every vectorized answer of a pitch field.

Validation logic is documented in ``tests/core/README.md`` under
"``test_pitch_fields.py`` - A null pitch row stays null".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from timetoalign.core.events import (
    EnharmonicPitch,
    EnharmonicPitchClass,
    EnharmonicPitchClassField,
    EnharmonicPitchField,
    GenericPitchClass,
    GenericPitchClassField,
    GenericPitchField,
    MidiPitch,
    MidiPitchField,
    SpecificPitch,
    SpecificPitchClass,
    SpecificPitchClassField,
    SpecificPitchField,
)
from timetoalign.core.fields import SemanticField


def _column(
    field_cls: type[SemanticField[Any]],
    members: dict[str, list[Any]],
    valid: list[bool],
) -> SemanticField[Any]:
    """A field over a struct column whose null rows keep the given children."""
    schema = field_cls.pa_schema
    arrays = [
        (
            pa.array(members.get(schema.field(i).name), type=schema.field(i).type)
            if schema.field(i).name in members
            else pa.nulls(len(valid), type=schema.field(i).type)
        )
        for i in range(schema.num_fields)
    ]
    column = pa.StructArray.from_arrays(
        arrays,
        fields=list(schema),
        mask=pa.array([not v for v in valid]),
    )
    return field_cls.from_field((column, pa.field("pitch", column.type)))


def _specific() -> SpecificPitchField:
    """B♭3, a null row holding the ms3 loader's placeholders, F♯♯4, C-1."""
    return _column(
        SpecificPitchField,
        {
            "step": ["B", "", "F", "C"],
            "alter": [-1, 0, 2, 0],
            "octave": [3, 0, 4, -1],
        },
        [True, False, True, True],
    )


class TestSpecificPitchField:
    def test_fifths(self) -> None:
        assert _specific().fifths.to_pylist() == [-2, None, 13, 0]

    def test_pitch_class(self) -> None:
        assert _specific().pitch_class.to_pylist() == [10, None, 7, 0]

    def test_midi_number(self) -> None:
        assert _specific().midi_number.to_pylist() == [58, None, 67, 0]

    def test_get(self) -> None:
        field = _specific()
        assert field.get().to_pylist() == ["B♭3", None, "F♯♯4", "C-1"]
        assert field.get(format="midi").to_pylist() == ["58", None, "67", "0"]

    @pytest.mark.parametrize(
        ("target", "expected"),
        [
            (
                MidiPitch,
                [{"midi_number": 58}, None, {"midi_number": 67}, {"midi_number": 0}],
            ),
            (
                EnharmonicPitchClass,
                [{"pitch_class": 10}, None, {"pitch_class": 7}, {"pitch_class": 0}],
            ),
            (
                SpecificPitchClass,
                [
                    {"step": "B", "alter": -1},
                    None,
                    {"step": "F", "alter": 2},
                    {"step": "C", "alter": 0},
                ],
            ),
        ],
    )
    def test_convert_to(self, target: type, expected: list[Any]) -> None:
        assert _specific().convert_to(target).to_pyarrow().to_pylist() == expected

    def test_a_slice_starting_on_the_null_row(self) -> None:
        column = _specific().to_pyarrow()[1:]
        field = SpecificPitchField.from_field((column, pa.field("pitch", column.type)))

        assert field.fifths.to_pylist() == [None, 13, 0]
        assert field.midi_number.to_pylist() == [None, 67, 0]
        assert field.get().to_pylist() == [None, "F♯♯4", "C-1"]


class TestSpecificPitchClassField:
    @staticmethod
    def _field() -> SpecificPitchClassField:
        return _column(
            SpecificPitchClassField,
            {"step": ["B", "", "F"], "alter": [-1, 0, 2]},
            [True, False, True],
        )

    def test_fifths_pitch_class_and_names(self) -> None:
        field = self._field()

        assert field.fifths.to_pylist() == [-2, None, 13]
        assert field.pitch_class.to_pylist() == [10, None, 7]
        assert field.get().to_pylist() == ["B♭", None, "F♯♯"]

    def test_convert_to_enharmonic_pitch_class(self) -> None:
        converted = self._field().convert_to(EnharmonicPitchClass)

        assert converted.to_pyarrow().to_pylist() == [
            {"pitch_class": 10},
            None,
            {"pitch_class": 7},
        ]


@pytest.mark.parametrize("field_cls", [EnharmonicPitchField, MidiPitchField])
class TestMidiNumberFields:
    @staticmethod
    def _field(field_cls: type[SemanticField[Any]]) -> SemanticField[Any]:
        return _column(field_cls, {"midi_number": [58, 0, 67]}, [True, False, True])

    def test_pitch_class_and_octave(self, field_cls: type) -> None:
        field = self._field(field_cls)

        assert field.pitch_class.to_pylist() == [10, None, 7]
        assert field.octave.to_pylist() == [3, None, 4]

    def test_convert_to_enharmonic_pitch_class(self, field_cls: type) -> None:
        converted = self._field(field_cls).convert_to(EnharmonicPitchClass)

        assert converted.to_pyarrow().to_pylist() == [
            {"pitch_class": 10},
            None,
            {"pitch_class": 7},
        ]

    def test_retyping_keeps_the_null_row(self, field_cls: type) -> None:
        target = MidiPitch if field_cls is EnharmonicPitchField else EnharmonicPitch
        converted = self._field(field_cls).convert_to(target)

        assert converted.to_pyarrow().to_pylist() == [
            {"midi_number": 58},
            None,
            {"midi_number": 67},
        ]


def test_enharmonic_pitch_class_names() -> None:
    field = _column(EnharmonicPitchClassField, {"pitch_class": [10, 0]}, [True, False])

    assert field.get().to_pylist() == ["10", None]


def test_generic_pitch_names_and_class() -> None:
    field = _column(
        GenericPitchField, {"step": [6, 0], "octave": [3, 0]}, [True, False]
    )

    assert field.get().to_pylist() == ["B3", None]
    assert field.convert_to(GenericPitchClass).to_pyarrow().to_pylist() == [
        {"step": 6},
        None,
    ]


def test_generic_pitch_class_names() -> None:
    field = _column(GenericPitchClassField, {"step": [6, 0]}, [True, False])

    assert field.get().to_pylist() == ["B", None]


def test_the_scalars_agree_with_the_fields() -> None:
    """The spelled rows read back as the scalars the expectations name."""
    field = _specific()

    assert [field[i] for i in range(4)] == [
        SpecificPitch(step="B", alter=-1, octave=3),
        None,
        SpecificPitch(step="F", alter=2, octave=4),
        SpecificPitch(step="C", alter=0, octave=-1),
    ]
    assert [p.fifths for p in (field[0], field[2], field[3])] == [-2, 13, 0]
    assert [p.midi_number for p in (field[0], field[2], field[3])] == [58, 67, 0]


_SPELLED = [
    {"step": "B", "alter": -1, "octave": 3},
    None,
    {"step": "F", "alter": 2, "octave": 4},
]
_MIDI = [{"midi_number": 58}, None, {"midi_number": 67}]
_PITCH_CLASSES = [{"pitch_class": 10}, None, {"pitch_class": 7}]


@pytest.mark.parametrize(
    ("field_cls", "rows", "target", "expected"),
    [
        (
            GenericPitchField,
            [{"step": 6, "octave": 3}, None, {"step": 0, "octave": 4}],
            GenericPitchClass,
            [{"step": 6}, None, {"step": 0}],
        ),
        (
            SpecificPitchClassField,
            [{"step": "B", "alter": -1}, None, {"step": "F", "alter": 2}],
            EnharmonicPitchClass,
            _PITCH_CLASSES,
        ),
        (EnharmonicPitchField, _MIDI, EnharmonicPitchClass, _PITCH_CLASSES),
        (MidiPitchField, _MIDI, EnharmonicPitchClass, _PITCH_CLASSES),
        (SpecificPitchField, _SPELLED, EnharmonicPitchClass, _PITCH_CLASSES),
        (
            SpecificPitchField,
            _SPELLED,
            SpecificPitchClass,
            [{"step": "B", "alter": -1}, None, {"step": "F", "alter": 2}],
        ),
        (SpecificPitchField, _SPELLED, MidiPitch, _MIDI),
    ],
)
@pytest.mark.parametrize("container", [list, tuple, "ndarray"])
@pytest.mark.parametrize("method", ["convert_to", "to"])
def test_a_field_over_plain_data_converts(
    field_cls: type[SemanticField[Any]],
    rows: list[Any],
    target: type,
    expected: list[Any],
    container: Any,
    method: str,
) -> None:
    data = np.array(rows, dtype=object) if container == "ndarray" else container(rows)
    field = field_cls.from_field((data, pa.field("pitch", field_cls.pa_schema)))

    converted = getattr(field, method)(target)

    assert converted.to_pyarrow().to_pylist() == expected
