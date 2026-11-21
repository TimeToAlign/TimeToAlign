"""A coordinate cell names its own axis.

Validation logic is documented in ``tests/core/README.md`` under
"test_coordinate_cell.py"; every expectation here is an exact value.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest
from timetoshare.tables import COORDINATE_TYPE

from timetoalign.core import time as time_module
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    RATIONAL_STRUCT_TYPE,
    Coordinate,
    Duration,
    axis_coordinate_cells,
    cell_coordinate,
    coordinate_cells,
)

SPELLED_OUT = pa.struct(
    [
        pa.field("value", pa.float64()),
        pa.field("numerator", pa.int64()),
        pa.field("denominator", pa.int64()),
        pa.field("unit", pa.string()),
        pa.field("number_type", pa.string()),
    ]
)

COORDINATES = [
    Coordinate(Fraction(1, 3), "quarters"),
    None,
    Coordinate(0.1, "seconds"),
    Coordinate(139776, "samples"),
    Coordinate(Fraction(2, 3), "quarters"),
]

EXPECTED_CELLS = [
    {
        "value": 0.3333333333333333,
        "numerator": 1,
        "denominator": 3,
        "unit": "quarters",
        "number_type": "fraction",
    },
    None,
    {
        "value": 0.1,
        "numerator": 3602879701896397,
        "denominator": 36028797018963968,
        "unit": "seconds",
        "number_type": "float",
    },
    {
        "value": 139776.0,
        "numerator": 139776,
        "denominator": 1,
        "unit": "samples",
        "number_type": "int",
    },
    {
        "value": 0.6666666666666666,
        "numerator": 2,
        "denominator": 3,
        "unit": "quarters",
        "number_type": "fraction",
    },
]


class TestCellType:
    def test_the_cell_type_is_spelled_out(self) -> None:
        assert COORDINATE_CELL_TYPE == SPELLED_OUT

    def test_the_cell_type_is_timetoshares(self) -> None:
        assert COORDINATE_CELL_TYPE == COORDINATE_TYPE


class TestCoordinateCells:
    def test_each_cell_holds_both_sides_on_its_own_axis(self) -> None:
        cells = coordinate_cells(COORDINATES)
        assert cells.type == COORDINATE_CELL_TYPE
        assert cells.to_pylist() == EXPECTED_CELLS

    def test_a_duration_makes_the_cell_of_its_value(self) -> None:
        cells = coordinate_cells([Duration(Fraction(1, 3), "quarters")])
        assert cells.to_pylist() == [EXPECTED_CELLS[0]]

    def test_empty_input_gives_an_empty_column(self) -> None:
        cells = coordinate_cells([])
        assert cells.type == COORDINATE_CELL_TYPE
        assert len(cells) == 0


class TestAxisCoordinateCells:
    def test_integers_on_a_samples_axis(self) -> None:
        cells = axis_coordinate_cells([0, None, 44100], unit="samples")
        assert cells.type == COORDINATE_CELL_TYPE
        assert cells.to_pylist() == [
            {
                "value": 0.0,
                "numerator": 0,
                "denominator": 1,
                "unit": "samples",
                "number_type": "int",
            },
            None,
            {
                "value": 44100.0,
                "numerator": 44100,
                "denominator": 1,
                "unit": "samples",
                "number_type": "int",
            },
        ]

    def test_a_float_axis_fills_the_ratio(self) -> None:
        cells = axis_coordinate_cells([0.25], unit="seconds", number_type="float")
        assert cells.to_pylist() == [
            {
                "value": 0.25,
                "numerator": 1,
                "denominator": 4,
                "unit": "seconds",
                "number_type": "float",
            }
        ]


def _quarters(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "unit": "quarters",
        "number_type": "fraction",
    }


class TestNumberStructColumns:
    NUMBERS = [
        {"value": 0.75, "numerator": 3, "denominator": 4},
        None,
        {"value": 0.5, "numerator": 2, "denominator": 4},
    ]
    EXPECTED = [_quarters(3, 4), None, _quarters(2, 4)]

    def _column(self) -> pa.StructArray:
        return pa.array(self.NUMBERS, type=RATIONAL_STRUCT_TYPE)

    def test_a_struct_column_keeps_its_cells(self) -> None:
        cells = axis_coordinate_cells(self._column(), unit="quarters")
        assert cells.type == COORDINATE_CELL_TYPE
        assert cells.to_pylist() == self.EXPECTED

    def test_the_per_row_reader_gives_the_same_cells(self) -> None:
        cells = axis_coordinate_cells(self.NUMBERS, unit="quarters")
        assert cells.to_pylist() == self.EXPECTED

    def test_a_slice_gives_the_cells_of_its_own_rows(self) -> None:
        # Row 0 is dropped by the slice; row 2 is null with numbers underneath.
        column = pa.StructArray.from_arrays(
            [
                pa.array([9.0, 0.75, 7.0, 0.5]),
                pa.array([9, 3, 7, 2]),
                pa.array([1, 4, 1, 4]),
            ],
            fields=list(RATIONAL_STRUCT_TYPE),
            mask=pa.array([False, False, True, False]),
        ).slice(1)
        cells = axis_coordinate_cells(column, unit="quarters")
        assert cells.to_pylist() == self.EXPECTED

    def test_a_chunked_column_gives_the_cells_of_all_chunks(self) -> None:
        column = pa.chunked_array(
            [self._column().slice(0, 1), self._column().slice(1)],
            type=RATIONAL_STRUCT_TYPE,
        )
        cells = axis_coordinate_cells(column, unit="quarters")
        assert cells.to_pylist() == self.EXPECTED

    def test_no_cell_is_encoded_one_by_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a struct column was read cell by cell")

        monkeypatch.setattr(time_module, "number_cell", refuse)
        cells = axis_coordinate_cells(self._column(), unit="quarters")
        assert cells.to_pylist() == self.EXPECTED

    def test_a_cell_missing_its_ratio_mirrors_its_float(self) -> None:
        column = pa.array(
            [{"value": 0.5, "numerator": None, "denominator": None}],
            type=RATIONAL_STRUCT_TYPE,
        )
        cells = axis_coordinate_cells(column, unit="seconds")
        assert cells.to_pylist() == [
            {
                "value": 0.5,
                "numerator": 1,
                "denominator": 2,
                "unit": "seconds",
                "number_type": "float",
            }
        ]


class TestCellCoordinate:
    def test_reading_back_keeps_value_and_number_type(self) -> None:
        cells = coordinate_cells(COORDINATES).to_pylist()
        read = [cell_coordinate(cell) for cell in cells]
        assert read == COORDINATES
        assert [type(c.value) if c is not None else None for c in read] == [
            Fraction,
            None,
            float,
            int,
            Fraction,
        ]

    def test_a_duration_is_read_as_one(self) -> None:
        duration = cell_coordinate(EXPECTED_CELLS[2], Duration)
        assert isinstance(duration, Duration)
        assert duration == Duration(0.1, "seconds")

    def test_a_null_cell_reads_none(self) -> None:
        assert cell_coordinate(None) is None

    def test_a_cell_without_axis_is_refused(self) -> None:
        cell = {**EXPECTED_CELLS[0], "unit": None}
        with pytest.raises(ValueError, match="names no unit or number type"):
            cell_coordinate(cell)
