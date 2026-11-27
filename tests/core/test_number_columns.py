"""Column kernels that read each row in its own number type, and the cell validator.

Validation logic is documented in ``tests/core/README.md`` under
"test_number_columns.py"; every expectation here is an exact value.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import time as time_module
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    NUMBER_TYPE_CODES,
    NUMBER_TYPES_BY_CODE,
    RATIONAL_STRUCT_TYPE,
    DenominateNumber,
    RedundantNumberField,
    build_number_struct_array,
    cast_number_column,
    compare_number_column,
    in_lowest_terms,
    number_cell,
    number_type_codes,
)
from timetoalign.storage.events import EventData

INT, FRACTION, FLOAT = NumberType.int, NumberType.fraction, NumberType.float

# The cell whose two sides hold two numbers: 1/2 and 0.7.
TWO_NUMBERS = {"value": 0.7, "numerator": 1, "denominator": 2}
AS_FRACTION = {"value": 0.5, "numerator": 1, "denominator": 2}
AS_FLOAT = {
    "value": 0.7,
    "numerator": 3152519739159347,
    "denominator": 4503599627370496,
}
ZERO_DENOMINATOR = {"value": 1.0, "numerator": 1, "denominator": 0}
# The double 0.3 and its mirror; the double 2.7 and its mirror.
POINT_THREE = {
    "value": 0.3,
    "numerator": 5404319552844595,
    "denominator": 18014398509481984,
}
TWO_POINT_SEVEN = {
    "value": 2.7,
    "numerator": 3039929748475085,
    "denominator": 1125899906842624,
}
VALUE_ONLY = {"value": 0.3, "numerator": None, "denominator": None}


def _cells(*rows: tuple[Any, NumberType] | None) -> pa.StructArray:
    """A column of the cells of *rows*, each built in its own number type."""
    return pa.array(
        [None if row is None else number_cell(*row) for row in rows],
        RATIONAL_STRUCT_TYPE,
    )


def _cell(numerator: int, denominator: int, value: float) -> dict[str, Any]:
    return {"value": value, "numerator": numerator, "denominator": denominator}


class TestRowTypeCodes:
    def test_the_codes_and_their_inverse(self) -> None:
        assert NUMBER_TYPE_CODES == {INT: 0, FRACTION: 1, FLOAT: 2}
        assert NUMBER_TYPES_BY_CODE == (INT, FRACTION, FLOAT)
        for number_type, code in NUMBER_TYPE_CODES.items():
            assert NUMBER_TYPES_BY_CODE[code] is number_type

    def test_codes_of_members_and_names(self) -> None:
        codes = number_type_codes([FRACTION, "float", INT])
        assert codes.type == pa.int8()
        assert codes.to_pylist() == [1, 2, 0]

    @pytest.mark.parametrize(
        "row_types",
        [
            pa.array([1, 1], pa.int8()),
            pa.array([1, None, 1], pa.int8()),
            pa.array([1, 3, 1], pa.int8()),
            pa.array([1, -1, 1], pa.int8()),
        ],
        ids=["another length", "a null", "past the codes", "below the codes"],
    )
    def test_row_types_that_do_not_fit_are_refused(
        self, row_types: pa.Int8Array
    ) -> None:
        column = _cells((1, INT), (2, INT), (3, INT))
        with pytest.raises(ValueError):
            cast_number_column(column, row_types, FLOAT)
        with pytest.raises(ValueError):
            compare_number_column(column, 1, "equal", row_types)

    def test_another_integer_type_is_read_by_its_values(self) -> None:
        column = _cells((1, INT), (2, INT), (3, INT))
        assert cast_number_column(
            column, pa.array([1, 1, 1], pa.int16()), FLOAT
        ).equals(cast_number_column(column, pa.array([1, 1, 1], pa.int8()), FLOAT))

    @pytest.mark.parametrize(
        "row_types",
        [pa.array([1, 258, 1], pa.int16()), pa.array([1.0, 1.0, 1.0], pa.float64())],
        ids=["past int8, never wrapped", "not an integer type"],
    )
    def test_codes_that_narrowing_would_change_are_refused(
        self, row_types: pa.Int8Array
    ) -> None:
        column = _cells((1, INT), (2, INT), (3, INT))
        with pytest.raises(ValueError):
            cast_number_column(column, row_types, FLOAT)
        with pytest.raises(ValueError):
            compare_number_column(column, 1, "equal", row_types)


class TestCast:
    def test_to_fraction(self) -> None:
        column = _cells(
            (1, INT), (Fraction(1, 2), FRACTION), (0.25, FLOAT), (0.1, FLOAT), None
        )
        cast = cast_number_column(
            column, number_type_codes([INT, FRACTION, FLOAT, FLOAT, INT]), FRACTION
        )
        assert cast.type == RATIONAL_STRUCT_TYPE
        assert cast.to_pylist() == [
            _cell(1, 1, 1.0),
            _cell(1, 2, 0.5),
            _cell(1, 4, 0.25),
            _cell(3602879701896397, 36028797018963968, 0.1),
            None,
        ]

    def test_fractions_to_int_nearest_ties_to_even(self) -> None:
        column = _cells(
            *[
                (Fraction(n, d), FRACTION)
                for n, d in [(1, 2), (3, 2), (5, 2), (3, 4), (-1, 2)]
            ]
        )
        assert cast_number_column(column, FRACTION, INT).to_pylist() == [
            _cell(0, 1, 0.0),
            _cell(2, 1, 2.0),
            _cell(2, 1, 2.0),
            _cell(1, 1, 1.0),
            _cell(0, 1, 0.0),
        ]

    def test_floats_to_int_rint(self) -> None:
        column = _cells((2.5, FLOAT), (3.5, FLOAT), (-0.5, FLOAT))
        cast = cast_number_column(column, FLOAT, "int").to_pylist()
        assert cast == [_cell(2, 1, 2.0), _cell(4, 1, 4.0), _cell(0, 1, 0.0)]
        # -0.5 rounds to the integer 0, written as the float 0.0, never -0.0.
        assert str(cast[2]["value"]) == "0.0"

    def test_to_float_rebuilds_the_ratio_side(self) -> None:
        column = _cells((Fraction(1, 3), FRACTION), (7, INT))
        assert cast_number_column(
            column, number_type_codes([FRACTION, INT]), FLOAT
        ).to_pylist() == [
            _cell(6004799503160661, 18014398509481984, 0.3333333333333333),
            _cell(7, 1, 7.0),
        ]

    def test_a_double_without_an_int64_ratio_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not fit in int64"):
            cast_number_column(_cells((1e-300, FLOAT)), FLOAT, FRACTION)

    def test_int_to_fraction_returns_the_cells_as_they_are(self) -> None:
        column = _cells((3, INT), (-7, INT), None)
        assert cast_number_column(column, INT, FRACTION).equals(column)

    def test_an_int_row_cast_to_float_keeps_its_measured_float(self) -> None:
        measured = build_number_struct_array(
            [2.7], number_type=INT, preserve_source_floats=True
        )
        assert measured.to_pylist() == [_cell(3, 1, 2.7)]
        assert cast_number_column(measured, INT, FLOAT).to_pylist() == [TWO_POINT_SEVEN]

    def test_a_row_lacking_its_side_is_read_from_the_other(self) -> None:
        value_only = pa.array([VALUE_ONLY], RATIONAL_STRUCT_TYPE)
        assert cast_number_column(value_only, FRACTION, FLOAT).to_pylist() == [
            POINT_THREE
        ]
        assert cast_number_column(value_only, FRACTION, INT).to_pylist() == [
            _cell(0, 1, 0.0)
        ]
        ratio_only = pa.array(
            [{"value": None, "numerator": 3, "denominator": 1}], RATIONAL_STRUCT_TYPE
        )
        assert cast_number_column(ratio_only, INT, FLOAT).to_pylist() == [
            _cell(3, 1, 3.0)
        ]

    def test_a_cell_holding_neither_side_is_refused(self) -> None:
        empty = pa.array(
            [AS_FRACTION, {"value": None, "numerator": None, "denominator": None}],
            RATIONAL_STRUCT_TYPE,
        )
        with pytest.raises(ValueError, match="row 1"):
            cast_number_column(empty, FRACTION, FLOAT)

    @pytest.mark.parametrize(
        ("numerator", "denominator", "expected"),
        [
            (2**62, 2**62 + 1, 1),
            (2**62 + 3, 2**62 + 1, 1),
            (-(2**63), 3, -3074457345618258603),
        ],
        ids=["remainder 2**62", "just above 1", "int64 minimum"],
    )
    def test_ratios_near_the_int64_limits_round_exactly(
        self, numerator: int, denominator: int, expected: int
    ) -> None:
        column = pa.array([_cell(numerator, denominator, 1.0)], RATIONAL_STRUCT_TYPE)
        assert round(Fraction(numerator, denominator)) == expected
        assert cast_number_column(column, FRACTION, INT).to_pylist() == [
            _cell(expected, 1, float(expected))
        ]

    @pytest.mark.parametrize("target", [INT, FLOAT])
    def test_a_zero_denominator_is_refused(self, target: NumberType) -> None:
        cell = _cell(1, 0, 0.5)
        column = pa.array([cell], RATIONAL_STRUCT_TYPE)
        with pytest.raises(ValueError) as refusal:
            cast_number_column(column, FRACTION, target)
        assert "row 0" in str(refusal.value)
        assert repr(cell) in str(refusal.value)
        assert "denominator 0" in str(refusal.value)

    def test_rows_of_the_target_stay_untouched(self) -> None:
        # A float cell whose ratio side is 1/10 agrees with its float 0.1;
        # a cast to float keeps it bit for bit, ratio side included.
        kept = _cell(1, 10, 0.1)
        column = pa.array([kept, number_cell(3, INT)], RATIONAL_STRUCT_TYPE)
        assert cast_number_column(
            column, number_type_codes([FLOAT, INT]), FLOAT
        ).to_pylist() == [kept, _cell(3, 1, 3.0)]

    @pytest.mark.parametrize("target", [INT, FRACTION, FLOAT])
    def test_one_type_equals_its_code_array(self, target: NumberType) -> None:
        column = _cells((Fraction(5, 2), FRACTION), (Fraction(-1, 3), FRACTION), None)
        assert cast_number_column(column, FRACTION, target).equals(
            cast_number_column(column, pa.array([1, 1, 1], pa.int8()), target)
        )

    def test_a_chunked_column_is_cast(self) -> None:
        chunked = pa.chunked_array([_cells((0.5, FLOAT)), _cells((0.25, FLOAT))])
        assert cast_number_column(chunked, FLOAT, FRACTION).to_pylist() == [
            _cell(1, 2, 0.5),
            _cell(1, 4, 0.25),
        ]

    @pytest.mark.parametrize("target", [INT, FRACTION, FLOAT])
    def test_vectorised(
        self, monkeypatch: pytest.MonkeyPatch, target: NumberType
    ) -> None:
        values: list[tuple[Any, NumberType]] = []
        for k in range(1000):
            kind = NUMBER_TYPES_BY_CODE[k % 3]
            values.append(
                (k - 500, INT)
                if kind is INT
                else (
                    (Fraction(k - 500, 7), FRACTION)
                    if kind is FRACTION
                    else ((k - 500) / 8, FLOAT)
                )
            )
        column = _cells(*values)
        codes = number_type_codes([kind for _, kind in values])
        expected = [
            number_cell(
                time_module.express_as(time_module.parse_number(v), target), target
            )
            for v, _ in values
        ]

        def fail(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("a row was computed one at a time")

        monkeypatch.setattr(time_module, "number_cell", fail)
        monkeypatch.setattr(time_module, "combine_numbers", fail)
        cast = cast_number_column(column, codes, target).to_pylist()
        monkeypatch.undo()
        # Rows of the target are kept as they are; the others as express_as
        # writes them, both sides of each cell carrying one number.
        assert cast == [
            original if kind is target else want
            for original, want, (_, kind) in zip(column.to_pylist(), expected, values)
        ]


class TestJoinNeverRefusesForTheNumberType:
    def test_a_value_only_cell_on_an_exact_table(self) -> None:
        fractions = EventData.from_arrays(
            {
                "start": pa.array([AS_FRACTION, VALUE_ONLY], RATIONAL_STRUCT_TYPE),
                "duration": pa.array(
                    [_cell(1, 1, 1.0), _cell(1, 1, 1.0)], RATIONAL_STRUCT_TYPE
                ),
            },
            TimeUnit.quarters,
            FRACTION,
            seed=1,
        )
        floats = EventData.from_arrays(
            {"start": [0.25], "duration": [1.0]}, TimeUnit.quarters, FLOAT, seed=2
        )
        joined = floats.concat(fractions)
        # The join keeps every row as stored (T136); the cast on request
        # reads the value-only cell from its float side, never refusing it.
        assert joined.table.column("start").to_pylist() == [
            _cell(1, 4, 0.25),
            AS_FRACTION,
            VALUE_ONLY,
        ]
        assert joined.to_number_type(FLOAT).table.column("start").to_pylist()[2] == (
            POINT_THREE
        )

    def test_an_int_table_keeps_its_measured_float(self) -> None:
        ints = EventData.from_arrays(
            {
                "start": build_number_struct_array(
                    [2.7], number_type=INT, preserve_source_floats=True
                ),
                "duration": [1],
            },
            TimeUnit.number,
            INT,
            seed=3,
        )
        floats = EventData.from_arrays(
            {"start": [0.25], "duration": [1.0]}, TimeUnit.number, FLOAT, seed=2
        )
        joined = floats.concat(ints)
        assert joined.table.column("start").to_pylist()[0] == _cell(1, 4, 0.25)
        # Cast to float, the int row is read from its float side: the measured
        # 2.7 survives.
        assert joined.to_number_type(FLOAT).table.column("start").to_pylist() == [
            _cell(1, 4, 0.25),
            TWO_POINT_SEVEN,
        ]


class TestCompare:
    def test_int_query_on_mixed_rows(self) -> None:
        rows = _cells((Fraction(3, 4), FRACTION), (1, FRACTION), (1.0, FLOAT), (1, INT))
        types = number_type_codes([FRACTION, FRACTION, FLOAT, INT])
        assert compare_number_column(rows, 1, "equal", types).to_pylist() == [
            False,
            True,
            True,
            True,
        ]

    def test_float_query_is_a_key(self) -> None:
        rows = _cells((Fraction(1, 10), FRACTION), (0.1, FLOAT), (0, INT))
        types = number_type_codes([FRACTION, FLOAT, INT])
        assert compare_number_column(rows, 0.1, "equal", types).to_pylist() == [
            True,
            True,
            False,
        ]

    def test_less_with_an_exact_query(self) -> None:
        rows = _cells((Fraction(1, 3), FRACTION), (0.5, FLOAT), (0, INT))
        types = number_type_codes([FRACTION, FLOAT, INT])
        assert compare_number_column(
            rows, Fraction(1, 2), "less", types
        ).to_pylist() == [
            True,
            False,
            True,
        ]

    def test_a_float_row_compares_as_a_key_under_an_exact_query(self) -> None:
        rows = _cells(
            (Fraction(1, 3), FRACTION),
            (0.3333333333333333, FLOAT),
            (Fraction(333, 1000), FRACTION),
        )
        types = number_type_codes([FRACTION, FLOAT, FRACTION])
        assert compare_number_column(
            rows, Fraction(1, 3), "equal", types
        ).to_pylist() == [
            True,
            True,
            False,
        ]
        # The same float row read as an exact one cross-multiplies its dyadic
        # ratio, which is not 1/3.
        assert compare_number_column(
            rows, Fraction(1, 3), "equal", FRACTION
        ).to_pylist() == [True, False, False]

    def test_a_second_column_compares_row_for_row(self) -> None:
        rows = _cells((Fraction(1, 10), FRACTION), (0.1, FLOAT), None)
        others = _cells(
            (Fraction(1, 10), FRACTION), (Fraction(1, 10), FRACTION), (1, INT)
        )
        types = number_type_codes([FRACTION, FLOAT, INT])
        assert compare_number_column(rows, others, "equal", types).to_pylist() == [
            True,
            True,
            None,
        ]

    def test_nan_is_refused_and_nulls_stay_null(self) -> None:
        rows = _cells((1, INT), None)
        types = number_type_codes([INT, FLOAT])
        with pytest.raises(ValueError, match="NaN names no position"):
            compare_number_column(rows, float("nan"), "equal", types)
        assert compare_number_column(rows, 1, "equal", types).to_pylist() == [
            True,
            None,
        ]


class TestZeroDenominator:
    def test_a_cell_is_refused(self) -> None:
        with pytest.raises(ValueError) as refusal:
            number_cell(ZERO_DENOMINATOR)
        assert repr(ZERO_DENOMINATOR) in str(refusal.value)
        assert "denominator 0" in str(refusal.value)

    def test_a_column_is_refused_naming_the_row(self) -> None:
        column = pa.array([AS_FRACTION, ZERO_DENOMINATOR], RATIONAL_STRUCT_TYPE)
        for read in (
            lambda: build_number_struct_array(column, number_type=FRACTION),
            lambda: in_lowest_terms(column),
        ):
            with pytest.raises(ValueError) as refusal:
                read()
            assert "row 1" in str(refusal.value)
            assert repr(ZERO_DENOMINATOR) in str(refusal.value)
            assert "denominator 0" in str(refusal.value)

    def test_an_event_table_is_refused(self) -> None:
        table = pa.table(
            {
                "id": ["e1", "e2"],
                "event_type": ["Note", "Note"],
                "start": pa.array(
                    [AS_FRACTION, ZERO_DENOMINATOR], RATIONAL_STRUCT_TYPE
                ),
            }
        )
        with pytest.raises(ValueError) as refusal:
            EventData(table, TimeUnit.quarters, FRACTION)
        assert "row 1" in str(refusal.value)
        assert "denominator 0" in str(refusal.value)


class TestTwoSidesResolvedByTheNumberType:
    @pytest.mark.parametrize(
        ("number_type", "expected"), [(FRACTION, AS_FRACTION), (FLOAT, AS_FLOAT)]
    )
    def test_every_entry(
        self, number_type: NumberType, expected: dict[str, Any]
    ) -> None:
        assert number_cell(TWO_NUMBERS, number_type) == expected
        column = pa.array([TWO_NUMBERS], RATIONAL_STRUCT_TYPE)
        assert build_number_struct_array(
            column, number_type=number_type
        ).to_pylist() == [expected]
        assert (
            DenominateNumber.from_cell(
                TWO_NUMBERS, TimeUnit.quarters, number_type
            ).to_cell()
            == expected
        )
        coordinate_cell = {
            **TWO_NUMBERS,
            "unit": "quarters",
            "number_type": number_type.name,
        }
        assert in_lowest_terms(
            pa.array([coordinate_cell], COORDINATE_CELL_TYPE)
        ).to_pylist() == [
            {**expected, "unit": "quarters", "number_type": number_type.name}
        ]

    @pytest.mark.parametrize(
        ("number_type", "expected"), [(FRACTION, AS_FRACTION), (FLOAT, AS_FLOAT)]
    )
    def test_an_event_table(
        self, number_type: NumberType, expected: dict[str, Any]
    ) -> None:
        unit = TimeUnit.quarters if number_type is FRACTION else TimeUnit.seconds
        schema = EventData.get_schema(unit, number_type=number_type)
        table = pa.Table.from_arrays(
            [
                pa.array(["e1"]),
                pa.array(["Note"]),
                pa.array([TWO_NUMBERS], RATIONAL_STRUCT_TYPE),
            ],
            schema=pa.schema(
                [schema.field(name) for name in ("id", "event_type", "start")]
            ),
        )
        held = EventData(table, unit, number_type)
        assert held.table.column("start").to_pylist() == [expected]

    def test_without_a_number_type_both_sides_are_kept(self) -> None:
        assert number_cell(TWO_NUMBERS) == TWO_NUMBERS
        column = pa.array([TWO_NUMBERS], RATIONAL_STRUCT_TYPE)
        assert in_lowest_terms(column).to_pylist() == [TWO_NUMBERS]

    def test_a_guessed_number_type_resolves_nothing(self) -> None:
        column = pa.array([TWO_NUMBERS, _cell(2, 4, 0.5)], RATIONAL_STRUCT_TYPE)
        settled = [TWO_NUMBERS, AS_FRACTION]
        assert (
            build_number_struct_array(column, number_type=None).to_pylist() == settled
        )
        blueprint = RedundantNumberField(name="x")
        assert blueprint.from_array(column).data.to_pylist() == settled
        zero = pa.array([ZERO_DENOMINATOR], RATIONAL_STRUCT_TYPE)
        for read in (
            lambda: build_number_struct_array(zero, number_type=None),
            lambda: blueprint.from_array(zero),
        ):
            with pytest.raises(ValueError, match="row 0"):
                read()

    def test_a_plain_column_without_a_number_type_keeps_its_kind(self) -> None:
        assert build_number_struct_array([0.5], number_type=None).to_pylist() == [
            _cell(1, 2, 0.5)
        ]
        assert build_number_struct_array(
            pa.array([3], pa.int64()), number_type=None
        ).to_pylist() == [_cell(3, 1, 3.0)]

    @pytest.mark.parametrize("number_type", [INT, FRACTION, FLOAT])
    def test_agreeing_cells_are_kept(self, number_type: NumberType) -> None:
        agreeing = [
            _cell(1, 10, 0.1),  # the ratio's nearest double
            _cell(3602879701896397, 36028797018963968, 0.1),  # the float's mirror
            _cell(0, 1, 1e-300),  # the mirror over 2**62 of a double below 2**-10
        ]
        for cell in agreeing:
            assert number_cell(cell, number_type) == cell
        column = pa.array(agreeing, RATIONAL_STRUCT_TYPE)
        assert (
            build_number_struct_array(column, number_type=number_type).to_pylist()
            == agreeing
        )

    def test_an_int_cell_keeps_the_float_it_was_rounded_from(self) -> None:
        measured = _cell(1, 1, 1.4)
        assert number_cell(measured, INT) == measured
        assert number_cell(_cell(1, 1, 3.0), INT) == _cell(1, 1, 1.0)
        column = pa.array([measured, _cell(1, 1, 3.0)], RATIONAL_STRUCT_TYPE)
        assert build_number_struct_array(column, number_type=INT).to_pylist() == [
            measured,
            _cell(1, 1, 1.0),
        ]

    def test_reduction_stays(self) -> None:
        assert number_cell(_cell(2, 4, 0.5), FRACTION) == _cell(1, 2, 0.5)
        assert number_cell(_cell(6, -4, -1.5), FLOAT) == _cell(-3, 2, -1.5)

    @pytest.mark.parametrize("number_type", [INT, FRACTION, FLOAT])
    def test_one_rule_two_routes(self, number_type: NumberType) -> None:
        rng = random.Random(20261126)
        cells = []
        for _ in range(400):
            numerator = rng.randint(-(10**6), 10**6)
            denominator = rng.choice([1, 2, 3, 4, 7, 10, 1024, 999_983])
            exact = Fraction(numerator, denominator)
            kind = rng.randrange(4)
            if kind == 0:  # agreeing: the ratio's nearest double
                value = float(exact)
            elif kind == 1:  # agreeing: a float and its mirror
                value = rng.uniform(-1e4, 1e4)
                mirror = number_cell(value, FLOAT)
                numerator, denominator = mirror["numerator"], mirror["denominator"]
            elif kind == 2:  # disagreeing by far
                value = float(exact) + rng.choice([-3.5, 0.75, 12.0])
            else:  # an int-shaped ratio with a float within 1 of it
                denominator = 1
                value = numerator + rng.uniform(-0.99, 0.99)
            cells.append(_cell(numerator, denominator, value))
        column = pa.array(cells, RATIONAL_STRUCT_TYPE)
        settled = build_number_struct_array(column, number_type=number_type).to_pylist()
        assert settled == [number_cell(cell, number_type) for cell in cells]
