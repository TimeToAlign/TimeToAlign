"""The Projection event, its reference and their paired fields.

Strategy: ``tests/core/README.md``, section ``test_projection.py``.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from timetoalign.core import (
    Coordinate,
    Duration,
    EventReference,
    EventReferenceField,
    NumberType,
    Projection,
    ProjectionField,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.core.time import RATIONAL_STRUCT_TYPE

REFERENCE = EventReference(timeline_id="clt_6", event_id="x1azmn68")


def _round_trip(field: ProjectionField, path: Path) -> ProjectionField:
    """Write *field* as the one column of a Parquet file and read it back."""
    table = pa.Table.from_arrays([field.data], schema=pa.schema([field.to_field()]))
    pq.write_table(table, path)
    read = pq.read_table(path)
    return ProjectionField.from_field(
        (read.column(0).combine_chunks(), read.schema.field(0))
    )


class TestStoredShape:
    def test_projection_struct_is_the_scalar_properties(self) -> None:
        assert ProjectionField.pa_schema == pa.struct(
            [
                pa.field("id", pa.string()),
                pa.field("start", RATIONAL_STRUCT_TYPE),
                pa.field("duration", RATIONAL_STRUCT_TYPE),
                pa.field(
                    "projection_of",
                    pa.struct(
                        [
                            pa.field("timeline_id", pa.string()),
                            pa.field("event_id", pa.string()),
                        ]
                    ),
                ),
            ]
        )

    def test_reference_struct_is_two_values(self) -> None:
        assert EventReferenceField.pa_schema == pa.struct(
            [pa.field("timeline_id", pa.string()), pa.field("event_id", pa.string())]
        )


class TestConstruction:
    def test_instant_has_no_duration_and_no_end(self) -> None:
        projection = Projection(
            id="projection_1",
            start=Coordinate(850176, TimeUnit.samples),
            projection_of=REFERENCE,
        )
        assert projection.duration is None
        assert projection.end is None
        assert projection.projection_of == REFERENCE
        assert projection.semantic_type == "Projection"

    def test_interval_end_is_exact(self) -> None:
        projection = Projection(
            id="projection_1",
            start=Coordinate(Fraction(27, 2), TimeUnit.quarters),
            duration=Duration(Fraction(1, 3), TimeUnit.quarters),
        )
        end = projection.end
        assert end == Coordinate(Fraction(83, 6), TimeUnit.quarters)
        assert type(end.value) is Fraction
        assert end.value == Fraction(83, 6)

    def test_reference_refuses_an_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            EventReference.model_validate(
                {"timeline_id": "clt_6", "event_id": "x1azmn68", "flow": "flow_1"}
            )

    def test_reference_refuses_an_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            EventReference(timeline_id="clt_6", event_id="")

    def test_projection_refuses_an_empty_id(self) -> None:
        with pytest.raises(ValidationError):
            Projection(id="", start=Coordinate(1, TimeUnit.samples))

    def test_duration_in_another_unit_is_refused(self) -> None:
        with pytest.raises(ValidationError) as refused:
            Projection(
                id="projection_1",
                start=Coordinate(44100, TimeUnit.samples),
                duration=Duration(0.5, TimeUnit.seconds),
            )
        assert (
            "Projection 'projection_1': its duration is in seconds, its start in "
            "samples; both are on one axis"
        ) in str(refused.value)

    def test_duration_in_another_number_type_is_refused(self) -> None:
        with pytest.raises(ValidationError) as refused:
            Projection(
                id="projection_1",
                start=Coordinate(Fraction(1, 2), TimeUnit.seconds),
                duration=Duration(0.5, TimeUnit.seconds),
            )
        assert (
            "Projection 'projection_1': its duration is written as float, its "
            "start as fraction; both are on one axis"
        ) in str(refused.value)


class TestArrowRoundTrip:
    def test_fraction_axis(self, tmp_path: Path) -> None:
        written = [
            Projection(
                id="projection_1",
                start=Coordinate(Fraction(27, 2), TimeUnit.quarters),
                duration=Duration(Fraction(1, 3), TimeUnit.quarters),
                projection_of=REFERENCE,
            ),
            Projection(id="x1", start=Coordinate(Fraction(1, 2), TimeUnit.quarters)),
        ]
        field = _round_trip(
            ProjectionField.from_projections(written), tmp_path / "q.parquet"
        )
        assert [field[0], field[1]] == written
        assert field.unit is TimeUnit.quarters
        assert field.number_type is NumberType.fraction
        assert field_metadata(field.field) == {
            "field_type": "ProjectionField",
            "unit": "quarters",
            "number_type": "fraction",
            "version": 1,
        }
        first = field[0]
        assert type(first.start.value) is Fraction
        assert first.start.value == Fraction(27, 2)
        assert first.duration.value == Fraction(1, 3)
        assert first.projection_of == REFERENCE
        assert field[1].projection_of is None
        assert field[1].duration is None
        cells = field.data.to_pylist()
        assert cells[0]["start"] == {
            "value": 13.5,
            "numerator": 27,
            "denominator": 2,
        }
        assert cells[0]["duration"]["numerator"] == 1
        assert cells[0]["duration"]["denominator"] == 3
        assert cells[0]["projection_of"] == {
            "timeline_id": "clt_6",
            "event_id": "x1azmn68",
        }
        assert cells[1]["projection_of"] is None

    def test_int_axis(self, tmp_path: Path) -> None:
        written = [
            Projection(
                id="projection_27",
                start=Coordinate(850176, TimeUnit.samples),
                duration=Duration(22050, TimeUnit.samples),
                projection_of=REFERENCE,
            )
        ]
        field = _round_trip(
            ProjectionField.from_projections(written), tmp_path / "s.parquet"
        )
        assert field[0] == written[0]
        assert type(field[0].start.value) is int
        assert field[0].start.value == 850176
        assert field[0].duration.value == 22050
        assert field.data.to_pylist()[0]["start"] == {
            "value": 850176.0,
            "numerator": 850176,
            "denominator": 1,
        }

    def test_float_axis(self, tmp_path: Path) -> None:
        written = [Projection(id="p", start=Coordinate(0.1, TimeUnit.seconds))]
        field = _round_trip(
            ProjectionField.from_projections(written), tmp_path / "f.parquet"
        )
        assert field[0] == written[0]
        assert type(field[0].start.value) is float
        assert field[0].start.value == 0.1
        assert field.number_type is NumberType.float


class TestOneAxisPerField:
    def test_two_units_are_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            ProjectionField.from_projections(
                [
                    Projection(id="a", start=Coordinate(1, TimeUnit.samples)),
                    Projection(id="b", start=Coordinate(1.0, TimeUnit.seconds)),
                ]
            )
        assert str(refused.value) == (
            "Projection 'b' is at seconds written as float; the field's axis is "
            "samples written as int"
        )

    def test_two_number_types_are_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            ProjectionField.from_projections(
                [
                    Projection(
                        id="a", start=Coordinate(Fraction(1, 2), TimeUnit.quarters)
                    ),
                    Projection(
                        id="b",
                        start=Coordinate(
                            0.5, TimeUnit.quarters, number_type=NumberType.float
                        ),
                    ),
                ]
            )
        assert str(refused.value) == (
            "Projection 'b' is at quarters written as float; the field's axis is "
            "quarters written as fraction"
        )

    def test_an_empty_field_needs_a_unit(self) -> None:
        with pytest.raises(ValueError) as refused:
            ProjectionField.from_projections([])
        assert str(refused.value) == (
            "ProjectionField.from_projections: an empty field needs a unit"
        )
        empty = ProjectionField.from_projections([], unit=TimeUnit.samples)
        assert len(empty) == 0
        assert empty.unit is TimeUnit.samples
        assert empty.number_type is NumberType.int
