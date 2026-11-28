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
from timetoshare.tables import PROJECTION_OF_TYPE, fixture_path

from timetoalign.alignment import AlignmentBundle
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
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    PACKAGE_NUMBER_TYPE,
    package_cells,
)
from timetoalign.timelines import Timeline

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
                pa.field("start", COORDINATE_CELL_TYPE),
                pa.field("duration", COORDINATE_CELL_TYPE),
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

    def test_an_id_is_optional_until_a_timeline_mints_one(self) -> None:
        projection = Projection(start=Coordinate(3, TimeUnit.samples))
        assert projection.id is None
        assert repr(projection) == "Projection(start=3 samples)"

    def test_projection_refuses_an_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            Projection.model_validate(
                {"start": Coordinate(3, TimeUnit.samples), "pitch": 60}
            )

    def test_the_event_row(self) -> None:
        start = Coordinate(Fraction(27, 2), TimeUnit.quarters)
        duration = Duration(Fraction(1, 3), TimeUnit.quarters)
        assert Projection(
            id="p", start=start, duration=duration, projection_of=REFERENCE
        ).to_event_row() == {
            "event_type": "Projection",
            "id": "p",
            "start": start,
            "duration": duration,
            "projection_of": REFERENCE,
        }
        assert Projection(start=start).to_event_row() == {
            "event_type": "Projection",
            "start": start,
            "duration": None,
            "projection_of": None,
        }

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

    def test_a_projection_without_id_is_named_in_the_message(self) -> None:
        with pytest.raises(ValidationError) as refused:
            Projection(
                start=Coordinate(44100, TimeUnit.samples),
                duration=Duration(0.5, TimeUnit.seconds),
            )
        assert (
            "A Projection: its duration is in seconds, its start in samples; both "
            "are on one axis"
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
        assert field_metadata(field.field) == {
            "field_type": "ProjectionField",
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
            "unit": "quarters",
            "number_type": "fraction",
        }
        assert cells[0]["duration"] == {
            "value": 0.3333333333333333,
            "numerator": 1,
            "denominator": 3,
            "unit": "quarters",
            "number_type": "fraction",
        }
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
            "unit": "samples",
            "number_type": "int",
        }

    def test_float_axis(self, tmp_path: Path) -> None:
        written = [Projection(id="p", start=Coordinate(0.1, TimeUnit.seconds))]
        field = _round_trip(
            ProjectionField.from_projections(written), tmp_path / "f.parquet"
        )
        assert field[0] == written[0]
        assert type(field[0].start.value) is float
        assert field[0].start.value == 0.1
        assert field.data.to_pylist()[0]["start"] == {
            "value": 0.1,
            "numerator": 3602879701896397,
            "denominator": 36028797018963968,
            "unit": "seconds",
            "number_type": "float",
        }


class TestAxesPerCell:
    def test_one_field_holds_projections_of_several_axes(self, tmp_path: Path) -> None:
        written = [
            Projection(id="a", start=Coordinate(1, TimeUnit.samples)),
            None,
            Projection(id="b", start=Coordinate(1.0, TimeUnit.seconds)),
            Projection(
                id="c",
                start=Coordinate(0.5, TimeUnit.quarters, number_type=NumberType.float),
            ),
            Projection(start=Coordinate(Fraction(1, 2), TimeUnit.quarters)),
        ]
        field = _round_trip(
            ProjectionField.from_projections(written), tmp_path / "m.parquet"
        )
        assert [field[i] for i in range(5)] == written
        assert [
            (
                None
                if cell is None
                else (cell["start"]["unit"], cell["start"]["number_type"])
            )
            for cell in field.data.to_pylist()
        ] == [
            ("samples", "int"),
            None,
            ("seconds", "float"),
            ("quarters", "float"),
            ("quarters", "fraction"),
        ]
        assert type(field[3].start.value) is float
        assert type(field[4].start.value) is Fraction
        assert field[4].id is None

    def test_an_empty_field(self) -> None:
        empty = ProjectionField.from_projections([])
        assert len(empty) == 0
        assert empty.data.type == ProjectionField.pa_schema


def _fixture_events() -> list[dict]:
    """The event rows of TimeToShare's fixture registry, which the installed
    package holds, in the registry's order."""
    registry = pq.read_table(fixture_path("registry.parquet")).to_pylist()
    return [row for row in registry if row["start"] is not None]


#: The Projections of TimeToShare's fixture package, in its registry's order
#: (by timeline, each timeline's by exact start: clt_2, cpt_1, dpt_1), each
#: on its own timeline's axis.
FIXTURE_PROJECTIONS = [
    Projection(
        id="projection_10",
        start=Coordinate(Fraction(7, 2), TimeUnit.quarters),
        duration=Duration(Fraction(1, 2), TimeUnit.quarters),
        projection_of=EventReference(timeline_id="dpt_1", event_id="onset_1"),
    ),
    Projection(
        id="projection_12",
        start=Coordinate(11.25, TimeUnit.seconds),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_14"),
    ),
    Projection(
        id="projection_11",
        start=Coordinate(12.5, TimeUnit.seconds),
        duration=Duration(0.75, TimeUnit.seconds),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_16"),
    ),
    Projection(
        id="projection_1",
        start=Coordinate(0, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_1"),
    ),
    Projection(
        id="projection_2",
        start=Coordinate(44100, TimeUnit.samples),
        duration=Duration(44100, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_2"),
    ),
    Projection(
        id="projection_3",
        start=Coordinate(132300, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_3"),
    ),
    Projection(
        id="projection_4",
        start=Coordinate(132300, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_4"),
    ),
    Projection(
        id="projection_5",
        start=Coordinate(176400, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_5"),
    ),
    Projection(
        id="projection_9",
        start=Coordinate(396900, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_2", event_id="note_13"),
    ),
    Projection(
        id="projection_8",
        start=Coordinate(441000, TimeUnit.samples),
        projection_of=EventReference(timeline_id="clt_1", event_id="note_11"),
    ),
    Projection(id="inserted_1", start=Coordinate(573300, TimeUnit.samples)),
]


def _package_numbers(cells: pa.StructArray) -> pa.StructArray:
    """The numbers of coordinate cells, each written once, as a package writes them."""
    written = package_cells(cells)
    return pa.StructArray.from_arrays(
        [written.field(name) for name in PACKAGE_NUMBER_TYPE.names],
        fields=list(PACKAGE_NUMBER_TYPE),
        mask=written.is_null(),
    )


def _projections(timeline: Timeline) -> list[Projection]:
    """The Projections *timeline* holds, each on its own axis, in its order."""
    axis = {"unit": timeline.unit.value, "number_type": timeline.number_type.name}
    return [
        Projection.from_row(
            {
                **row,
                "start": {**row["start"], **axis},
                "duration": (
                    None if row["duration"] is None else {**row["duration"], **axis}
                ),
            }
        )
        for row in timeline.events.table.to_pylist()
        if row["event_type"] == "Projection"
    ]


class TestTimeToShareFixture:
    def test_the_fixture_rows_read_as_projections(self) -> None:
        bundle = AlignmentBundle.from_file(fixture_path("manifest.yaml"))
        read = {
            timeline_id: _projections(timeline)
            for timeline_id, timeline in sorted(bundle._known_timelines().items())
            if timeline.n_events
        }
        assert read == {
            "clt_2": [FIXTURE_PROJECTIONS[0]],
            "cpt_1": FIXTURE_PROJECTIONS[1:3],
            "dpt_1": FIXTURE_PROJECTIONS[3:],
        }
        registry = pq.read_table(fixture_path("registry.parquet")).to_pylist()
        events = [row for row in registry if row["start"] is not None]
        assert [row["timeline_id"] for row in events] == (
            ["clt_2"] + ["cpt_1"] * 2 + ["dpt_1"] * 8
        )

        def types(projections: list[Projection]) -> list[tuple[type, type | None]]:
            return [
                (
                    type(p.start.value),
                    None if p.duration is None else type(p.duration.value),
                )
                for p in projections
            ]

        assert types(read["dpt_1"]) == [(int, None), (int, int)] + [(int, None)] * 6
        assert types(read["cpt_1"]) == [(float, None), (float, float)]
        assert types(read["clt_2"]) == [(Fraction, Fraction)]

    def test_the_field_in_the_package_form_is_the_fixture_registry(self) -> None:
        field = ProjectionField.from_projections(FIXTURE_PROJECTIONS)
        events = _fixture_events()
        # The event each Projection stands for is its registry row's.
        assert [row["kind"] for row in events] == ["Projection"] * 11
        assert all(row["location"] is None for row in events)
        assert [(row["id"], row["projection_of"]) for row in events] == list(
            zip(
                field.data.field("id").to_pylist(),
                field.data.field("projection_of").cast(PROJECTION_OF_TYPE).to_pylist(),
                strict=True,
            )
        )
        start = field.data.field("start")
        # The registry holds each start on its root's axis: clt_2 lies at 8
        # on clt_1, the others are roots.
        offsets = {"clt_2": 8}
        assert [(row["id"], row["unit"], row["number_type"]) for row in events] == list(
            zip(
                field.data.field("id").to_pylist(),
                start.field("unit").to_pylist(),
                start.field("number_type").to_pylist(),
                strict=True,
            )
        )
        packaged = _package_numbers(start).to_pylist()
        assert [row["start"] for row in events[1:]] == packaged[1:]
        assert events[0]["start"] == {"numerator": 23, "denominator": 2, "value": None}
        assert Fraction(7, 2) + offsets["clt_2"] == Fraction(23, 2)
        assert [row["duration"] for row in events] == _package_numbers(
            field.data.field("duration")
        ).to_pylist()
