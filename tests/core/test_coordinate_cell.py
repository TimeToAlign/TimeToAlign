"""A coordinate cell names its own axis.

Validation logic is documented in ``tests/core/README.md`` under
"test_coordinate_cell.py"; every expectation here is an exact value.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest
from timetoshare.tables import COORDINATE_TYPE, NUMBER_TYPE

from timetoalign.core import time as time_module
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    PACKAGE_COORDINATE_CELL_TYPE,
    PACKAGE_NUMBER_TYPE,
    RATIONAL_STRUCT_TYPE,
    Coordinate,
    Duration,
    axis_coordinate_cells,
    cell_coordinate,
    cells_from_package,
    coordinate_cells,
    package_cells,
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


PACKAGE_NUMBER_SPELLED_OUT = pa.struct(
    [
        pa.field("numerator", pa.int64()),
        pa.field("denominator", pa.int64()),
        pa.field("value", pa.float64()),
    ]
)

PACKAGE_CELLS = [
    {
        "numerator": 1,
        "denominator": 3,
        "value": None,
        "unit": "quarters",
        "number_type": "fraction",
    },
    None,
    {
        "numerator": None,
        "denominator": None,
        "value": 0.1,
        "unit": "seconds",
        "number_type": "float",
    },
    {
        "numerator": 139776,
        "denominator": None,
        "value": None,
        "unit": "samples",
        "number_type": "int",
    },
    {
        "numerator": 2,
        "denominator": 3,
        "value": None,
        "unit": "quarters",
        "number_type": "fraction",
    },
]


def _package_cell(
    numerator: int | None,
    denominator: int | None,
    value: float | None,
    number_type: str | None = "int",
    unit: str | None = "samples",
) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": value,
        "unit": unit,
        "number_type": number_type,
    }


class TestPackageForm:
    def test_the_types_are_spelled_out_and_timetoshares(self) -> None:
        assert PACKAGE_NUMBER_TYPE == PACKAGE_NUMBER_SPELLED_OUT
        assert PACKAGE_COORDINATE_CELL_TYPE == pa.struct(
            [
                *PACKAGE_NUMBER_SPELLED_OUT,
                pa.field("unit", pa.string()),
                pa.field("number_type", pa.string()),
            ]
        )
        assert PACKAGE_NUMBER_TYPE == NUMBER_TYPE
        assert PACKAGE_COORDINATE_CELL_TYPE == COORDINATE_TYPE

    def test_each_number_is_written_once(self) -> None:
        written = package_cells(coordinate_cells(COORDINATES))
        assert written.type == PACKAGE_COORDINATE_CELL_TYPE
        assert written.to_pylist() == PACKAGE_CELLS

    def test_a_fraction_keeps_its_denominator_one(self) -> None:
        cells = coordinate_cells([Coordinate(Fraction(4), "quarters")])
        assert package_cells(cells).to_pylist() == [
            _package_cell(4, 1, None, "fraction", "quarters")
        ]

    def test_reading_back_fills_both_sides(self) -> None:
        cells = coordinate_cells(COORDINATES)
        read = cells_from_package(package_cells(cells))
        assert read.type == COORDINATE_CELL_TYPE
        assert read.equals(cells)
        assert read.to_pylist() == EXPECTED_CELLS

    def test_a_slice_reads_its_own_rows(self) -> None:
        written = pa.array(PACKAGE_CELLS, PACKAGE_COORDINATE_CELL_TYPE)
        assert cells_from_package(written.slice(2)).to_pylist() == EXPECTED_CELLS[2:]

    def test_members_are_read_by_name(self) -> None:
        reordered = pa.struct(
            [PACKAGE_COORDINATE_CELL_TYPE.field(i) for i in (4, 2, 3, 0, 1)]
        )
        written = pa.array(PACKAGE_CELLS, reordered)
        assert cells_from_package(written).to_pylist() == EXPECTED_CELLS

    def test_the_first_wrong_cell_is_named_exactly(self) -> None:
        written = pa.array(
            [None, _package_cell(3, 1, None)], PACKAGE_COORDINATE_CELL_TYPE
        )
        with pytest.raises(ValueError) as raised:
            cells_from_package(written, where="Test")
        assert str(raised.value) == (
            "Test row 1: the int cell (numerator, denominator, value) holds "
            "(3, 1, null), where an int is (n, null, null), a fraction "
            "(n, d >= 1, null) and a float (null, null, x) with x finite"
        )

    @pytest.mark.parametrize(
        ("cell", "problem"),
        [
            (
                _package_cell(3, None, 3.0),
                "the int cell (numerator, denominator, value) holds (3, null, 3.0)",
            ),
            (
                _package_cell(1, None, None, "fraction"),
                "the fraction cell (numerator, denominator, value) holds (1, null, null)",
            ),
            (
                _package_cell(1, 0, None, "fraction"),
                "the fraction cell (numerator, denominator, value) holds (1, 0, null)",
            ),
            (
                _package_cell(1, 2, 0.5, "fraction"),
                "the fraction cell (numerator, denominator, value) holds (1, 2, 0.5)",
            ),
            (
                _package_cell(1, 2, 0.5, "float"),
                "the float cell (numerator, denominator, value) holds (1, 2, 0.5)",
            ),
            (
                _package_cell(None, None, float("inf"), "float"),
                "the float cell (numerator, denominator, value) holds (null, null, inf)",
            ),
            (
                _package_cell(None, None, None, "float"),
                "the float cell (numerator, denominator, value) holds (null, null, null)",
            ),
            (
                _package_cell(3, None, None, "decimal"),
                "its number type is 'decimal', not 'int', 'fraction' or 'float'",
            ),
            (
                _package_cell(3, None, None, None),
                "its number type is None, not 'int', 'fraction' or 'float'",
            ),
            (_package_cell(3, None, None, "int", None), "it names no unit"),
        ],
        ids=[
            "int_with_value",
            "fraction_without_denominator",
            "fraction_denominator_zero",
            "fraction_with_value",
            "float_with_ratio",
            "float_infinite",
            "float_without_value",
            "unknown_type",
            "no_type",
            "no_unit",
        ],
    )
    def test_a_cell_not_as_its_number_type_says_is_refused(
        self, cell: dict, problem: str
    ) -> None:
        written = pa.array([cell], PACKAGE_COORDINATE_CELL_TYPE)
        with pytest.raises(ValueError) as raised:
            cells_from_package(written, where="Test")
        message = str(raised.value)
        if problem.startswith("the "):
            problem += (
                ", where an int is (n, null, null), a fraction (n, d >= 1, null) "
                "and a float (null, null, x) with x finite"
            )
        assert message == f"Test row 0: {problem}"

    def test_no_cell_is_converted_one_by_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("a cell was converted one by one")

        cells = coordinate_cells(COORDINATES * 200)
        monkeypatch.setattr(time_module, "number_cell", refuse)
        assert cells_from_package(package_cells(cells)).equals(cells)
