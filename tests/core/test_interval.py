"""Exact tests for the paired Interval scalar and semantic field."""

from __future__ import annotations

import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pytest

from timetoalign.core import (
    Coordinate,
    CoordinateField,
    Duration,
    DurationField,
    Interval,
    IntervalField,
    NumberType,
    TimeUnit,
)
from timetoalign.core.events import NoteField
from timetoalign.core.fields import StructField
from timetoalign.core.time import RATIONAL_STRUCT_TYPE


def _quarter(
    value: Fraction, number_type: NumberType = NumberType.fraction
) -> Coordinate:
    return Coordinate(value, TimeUnit.quarters, number_type=number_type)


def test_interval_preserves_exact_duration_metadata() -> None:
    """Duration uses the endpoints' exact value, unit, and number type."""
    interval = Interval(start=_quarter(Fraction(1, 2)), end=_quarter(Fraction(5, 2)))

    assert interval.unit is TimeUnit.quarters
    assert interval.number_type is NumberType.fraction
    assert interval.duration == Duration(
        Fraction(2), TimeUnit.quarters, number_type=NumberType.fraction
    )


def test_interval_accepts_zero_length() -> None:
    """Equal endpoints form a valid zero-length half-open interval."""
    endpoint = _quarter(Fraction(3, 2))

    assert Interval(start=endpoint, end=endpoint).duration.value == Fraction(0)


def test_interval_rejects_reversed_endpoints() -> None:
    """A reversed interval cannot manufacture a negative duration."""
    with pytest.raises(ValueError, match="start must not exceed"):
        Interval(start=_quarter(Fraction(2)), end=_quarter(Fraction(1)))


def test_interval_rejects_mixed_units() -> None:
    """Both endpoints must describe one coordinate axis."""
    with pytest.raises(ValueError, match="same unit"):
        Interval(
            start=_quarter(Fraction(1)),
            end=Coordinate(2.0, TimeUnit.seconds),
        )


def test_interval_rejects_mixed_number_types() -> None:
    """Both endpoints must share one canonical representation."""
    with pytest.raises(ValueError, match="same number_type"):
        Interval(
            start=_quarter(Fraction(1)),
            end=_quarter(Fraction(2), NumberType.float),
        )


def test_interval_field_materializes_scalar_and_vector_accessors() -> None:
    """Field indexing and vector accessors retain exact endpoint values."""
    interval = Interval(start=_quarter(Fraction(1, 2)), end=_quarter(Fraction(5, 2)))
    field = IntervalField.from_intervals([interval, None])

    assert field[0] == interval
    assert field[1] is None
    assert isinstance(field.start, CoordinateField)
    assert isinstance(field.end, CoordinateField)
    assert isinstance(field.duration, DurationField)
    assert field.start[0] == interval.start
    assert field.end[0] == interval.end
    assert field.duration[0] == interval.duration
    assert field.start[1] is None
    assert field.end[1] is None
    assert field.duration[1] is None


def test_empty_interval_field_requires_explicit_axis_metadata() -> None:
    """Empty storage cannot infer its shared axis declaration."""
    with pytest.raises(ValueError, match="requires unit and number_type"):
        IntervalField.from_intervals([])

    field = IntervalField.from_intervals(
        [], unit=TimeUnit.seconds, number_type=NumberType.float
    )
    assert field.unit is TimeUnit.seconds
    assert field.number_type is NumberType.float
    assert len(field) == 0


# ---------------------------------------------------------------------------
# An interval field answers its endpoints whenever it is an interval
# ---------------------------------------------------------------------------

_R = RATIONAL_STRUCT_TYPE
_INTERVAL_STRUCT = pa.struct([pa.field("start", _R), pa.field("end", _R)])
_ONE = {"value": 1.0, "numerator": 1, "denominator": 1}
_TWO = {"value": 2.0, "numerator": 2, "denominator": 1}


def _interval_field(
    data: pa.Array | None, struct: pa.DataType, **axis: object
) -> IntervalField:
    axis = axis or {"unit": "quarters", "number_type": "fraction"}
    return IntervalField(StructField(data, pa.field("iv", struct)), **axis)


@pytest.mark.parametrize(
    ("unit", "number_type"),
    [
        (TimeUnit.quarters, NumberType.fraction),
        (TimeUnit.ticks, NumberType.int),
    ],
)
def test_interval_field_without_data_answers_fields_without_data(
    unit: TimeUnit, number_type: NumberType
) -> None:
    """A bound field carrying no data answers endpoint fields carrying none."""
    field = IntervalField.from_field(
        pa.field("interval", _INTERVAL_STRUCT), unit=unit, number_type=number_type
    )
    assert field.is_empty
    expected = [
        (field.start, CoordinateField, "start"),
        (field.end, CoordinateField, "end"),
        (field.duration, DurationField, "value"),
    ]
    for answered, cls, name in expected:
        assert type(answered) is cls
        assert answered is not None
        assert answered.name == name
        assert answered.unit is unit
        assert answered.number_type is number_type
        assert answered.is_empty is True


def test_interval_field_with_zero_rows_answers_zero_length_fields() -> None:
    """Zero rows answer zero-length endpoint fields on the axis."""
    field = IntervalField.from_intervals(
        [], unit=TimeUnit.quarters, number_type=NumberType.fraction
    )
    for answered, cls, name in [
        (field.start, CoordinateField, "start"),
        (field.end, CoordinateField, "end"),
        (field.duration, DurationField, "value"),
    ]:
        assert type(answered) is cls
        assert answered is not None
        assert answered.name == name
        assert answered.unit is TimeUnit.quarters
        assert answered.number_type is NumberType.fraction
        assert answered.is_empty is False
        assert answered.to_pyarrow().to_pylist() == []


def test_interval_field_null_entries_answer_nulls() -> None:
    """A null row or a null endpoint answers a null cell, never a raise."""
    rows = [None, {"start": None, "end": _TWO}, {"start": None, "end": None}]
    field = _interval_field(pa.array(rows, type=_INTERVAL_STRUCT), _INTERVAL_STRUCT)
    start, end, duration = field.start, field.end, field.duration
    assert start is not None and end is not None and duration is not None
    assert start.to_pyarrow().to_pylist() == [None, None, None]
    assert end.to_pyarrow().to_pylist() == [None, _TWO, None]
    assert duration.to_pyarrow().to_pylist() == [None, None, None]
    assert [end[i] for i in range(3)] == [None, _quarter(Fraction(2)), None]


def test_unbound_interval_field_answers_none() -> None:
    """A field declared without data has no axis, and no endpoints yet."""
    field = IntervalField(source_fields="span")
    assert field.unit is None
    assert field.start is None
    assert field.end is None
    assert field.duration is None


_SCHEMA_ONLY = "Cannot convert schema-only field 'iv' to PyArrow array"
_MALFORMED_STRUCTS = {
    "without end": pa.struct([pa.field("start", _R)]),
    "without start": pa.struct([pa.field("end", _R)]),
    "start int64": pa.struct([pa.field("start", pa.int64()), pa.field("end", _R)]),
    "end string": pa.struct([pa.field("start", _R), pa.field("end", pa.string())]),
    "start of other members": pa.struct(
        [
            pa.field("start", pa.struct([pa.field("numerator", pa.int64())])),
            pa.field("end", _R),
        ]
    ),
}


@pytest.mark.parametrize(
    "struct", _MALFORMED_STRUCTS.values(), ids=_MALFORMED_STRUCTS.keys()
)
@pytest.mark.parametrize("accessor", ["start", "end", "duration"])
def test_malformed_interval_field_without_data_raises(
    struct: pa.DataType, accessor: str
) -> None:
    """A declared struct that is not an interval's never answers empty."""
    field = _interval_field(None, struct)
    with pytest.raises(TypeError) as caught:
        getattr(field, accessor)
    assert str(caught.value) == _SCHEMA_ONLY


_NUMBER_STRUCT_REFUSED = (
    "DenominateNumberField requires the number struct schema "
    "struct<value: double, numerator: int64, denominator: int64>; "
    "got struct<numerator: int64>"
)


@pytest.mark.parametrize(
    ("struct_key", "row", "accessor", "exception", "message"),
    [
        ("without end", {"start": _ONE}, "end", KeyError, "'end'"),
        ("without end", {"start": _ONE}, "duration", KeyError, "'end'"),
        ("without start", {"end": _ONE}, "start", KeyError, "'start'"),
        ("without start", {"end": _ONE}, "duration", KeyError, "'start'"),
        (
            "start int64",
            {"start": 1, "end": _ONE},
            "start",
            TypeError,
            "StructField requires a struct type, got int64",
        ),
        (
            "start int64",
            {"start": 1, "end": _ONE},
            "duration",
            TypeError,
            "StructField requires a struct type, got int64",
        ),
        (
            "end string",
            {"start": _ONE, "end": "x"},
            "end",
            TypeError,
            "StructField requires a struct type, got string",
        ),
        (
            "end string",
            {"start": _ONE, "end": "x"},
            "duration",
            TypeError,
            "StructField requires a struct type, got string",
        ),
        (
            "start of other members",
            {"start": {"numerator": 1}, "end": _ONE},
            "start",
            TypeError,
            _NUMBER_STRUCT_REFUSED,
        ),
        (
            "start of other members",
            {"start": {"numerator": 1}, "end": _ONE},
            "duration",
            TypeError,
            _NUMBER_STRUCT_REFUSED,
        ),
    ],
)
def test_malformed_interval_field_with_rows_raises(
    struct_key: str,
    row: dict[str, object],
    accessor: str,
    exception: type[Exception],
    message: str,
) -> None:
    """With rows, the endpoint a malformed struct cannot give raises as it did."""
    struct = _MALFORMED_STRUCTS[struct_key]
    field = _interval_field(pa.array([row], type=struct), struct)
    with pytest.raises(exception) as caught:
        getattr(field, accessor)
    assert str(caught.value) == message


def test_non_struct_column_is_no_interval_field() -> None:
    """A column that is not a struct cannot be read as an interval field at all."""
    for data in (pa.array([1]), None):
        with pytest.raises(TypeError) as caught:
            _interval_field(data, pa.int64())
        assert str(caught.value) == "StructField requires a struct type, got int64"


# ---------------------------------------------------------------------------
# NoteField.is_rest is a missing attribute; protocol checks always answer
# ---------------------------------------------------------------------------


def _note_field() -> NoteField:
    row = {
        "start": _ONE,
        "end": _TWO,
        "duration": _ONE,
        "velocity": 64,
        "instrument": "p",
    }
    return NoteField.from_field(
        pa.array([row], type=NoteField.struct_schema()),
        unit="quarters",
        number_type="fraction",
    )


def test_note_field_is_rest_is_a_missing_attribute() -> None:
    """A note field cannot tell rests: ``is_rest`` behaves as absent."""
    field = _note_field()
    with pytest.raises(AttributeError) as caught:
        field.is_rest  # noqa: B018
    assert str(caught.value) == "'StructField' object has no attribute 'is_rest'"
    assert hasattr(field, "is_rest") is False


_PROTOCOL_SCRIPT = """
import inspect, warnings
import pyarrow as pa
warnings.simplefilter("ignore")
import timetoalign.core.protocols as pr
from timetoalign.core.events import NoteField
from timetoalign.core.time import IntervalField

one = {"value": 1.0, "numerator": 1, "denominator": 1}
row = {"start": one, "end": one, "duration": one, "velocity": 64, "instrument": "p"}
fields = {
    "interval schema-only": IntervalField.from_field(
        pa.field("interval", IntervalField.struct_schema()), unit="quarters", number_type="fraction"
    ),
    "interval unbound": IntervalField(source_fields="span"),
    "note": NoteField.from_field(
        pa.array([row], type=NoteField.struct_schema()), unit="quarters", number_type="fraction"
    ),
}
protocols = sorted(
    (obj for _, obj in inspect.getmembers(pr, inspect.isclass)
     if getattr(obj, "_is_runtime_protocol", False) and obj.__module__ == pr.__name__),
    key=lambda p: p.__name__,
)
for label, field in fields.items():
    print(label, sorted(p.__name__ for p in protocols if isinstance(field, p)))
"""


def test_protocol_checks_answer_the_same_under_every_hash_seed(tmp_path: Path) -> None:
    """Runtime protocol checks answer, identically, whatever the hash seed."""
    script = tmp_path / "protocols.py"
    script.write_text(_PROTOCOL_SCRIPT)
    outputs = []
    for seed in ("0", "42"):
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
    interval = (
        "['CoordinateLike', 'InstantEventLike', 'IntervalEventLike', "
        "'SemanticTypeLike', 'TimeScalarLike', 'TimedObjectLike']"
    )
    assert outputs[0].splitlines() == [
        f"interval schema-only {interval}",
        f"interval unbound {interval}",
        "note ['CoordinateLike', 'SemanticTypeLike', 'TimeScalarLike']",
    ]
