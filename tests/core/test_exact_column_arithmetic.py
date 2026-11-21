"""Exact column arithmetic: no intermediate overflows, only a result can be refused.

The expected ratios are Python ``Fraction`` arithmetic, written out; see the
README section of this file for why they prove the engine exact.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import NumberType
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    build_number_struct_array,
    combine_number_columns,
    combine_numbers,
    number_cell,
    struct_to_coordinate,
)

_FRACTION = NumberType.fraction


def _column(values: list[Any], number_type: NumberType = _FRACTION) -> pa.StructArray:
    return build_number_struct_array(values, number_type=number_type)


def _cell(numerator: int, denominator: int, value: float) -> dict[str, Any]:
    return {"value": value, "numerator": numerator, "denominator": denominator}


def test_float_mirrored_sums_are_exact_past_int64_cross_products() -> None:
    starts = _column([0.1, Fraction(3, 2), 0.7, Fraction(1, 3), -0.3])
    durations = _column([0.2, 0.1, 0.2, 0.25, 0.1])

    ends = combine_number_columns(starts, durations, "add", _FRACTION)

    assert ends.to_pylist() == [
        _cell(10808639105689191, 2**55, 0.30000000000000004),
        _cell(57646075230342349, 2**55, 1.6),
        _cell(16212958658533785, 2**54, 0.8999999999999999),
        _cell(7, 12, 0.5833333333333334),
        _cell(-7205759403792793, 2**55, -0.19999999999999998),
    ]


def test_subtraction_is_exact_past_int64_cross_products() -> None:
    ends = _column([0.3, Fraction(7, 2)])
    starts = _column([0.1, 0.7])

    durations = combine_number_columns(ends, starts, "subtract", _FRACTION)

    assert durations.to_pylist() == [
        _cell(7205759403792793, 2**55, 0.19999999999999998),
        _cell(12610078956637389, 2**52, 2.8),
    ]


_REFUSAL = (
    "row {row}: 1129/12 + 3602879701896397/36028797018963968 = "
    "10179936597708269159/108086391056891904, whose numerator "
    "10179936597708269159 in lowest terms does not fit int64, the width of an "
    "exact cell; the result cannot be stored exactly. Give the operands exact "
    "values (a ratio written as a ratio, not as a decimal), or declare the axis "
    "number_type float"
)


def test_a_result_whose_lowest_terms_do_not_fit_is_refused() -> None:
    starts = _column([Fraction(1), Fraction(1129, 12)])
    durations = _column([Fraction(1), 0.1])

    with pytest.raises(ValueError) as refused:
        combine_number_columns(starts, durations, "add", _FRACTION)
    assert str(refused.value) == _REFUSAL.format(row=1)

    with pytest.raises(ValueError) as refused:
        combine_number_columns(
            starts, durations, "add", _FRACTION, row_numbers=[17, 42]
        )
    assert str(refused.value) == _REFUSAL.format(row=42)


def test_an_int_result_past_int64_is_refused() -> None:
    column = _column([2**62], NumberType.int)

    with pytest.raises(ValueError) as refused:
        combine_number_columns(column, column, "add", NumberType.int)
    assert str(refused.value) == (
        "row 0: 4611686018427387904 + 4611686018427387904 = 9223372036854775808, "
        "whose numerator 9223372036854775808 in lowest terms does not fit int64, "
        "the width of an exact cell; the result cannot be stored exactly. Give "
        "the operands exact values (a ratio written as a ratio, not as a "
        "decimal), or declare the axis number_type float"
    )


def test_the_int_lane_computes_on_integers_not_doubles() -> None:
    ends = _column([2**53 + 4, 2**53 + 1], NumberType.int)
    starts = _column([2**53 + 1, 0], NumberType.int)

    difference = combine_number_columns(ends, starts, "subtract", NumberType.int)

    assert difference.to_pylist() == [
        _cell(3, 1, 3.0),
        _cell(2**53 + 1, 1, 9007199254740992.0),
    ]


# -- the column path against the scalar engine, cell for cell ---------------


def _random_number(rng: random.Random, number_type: NumberType) -> Any:
    kind = rng.randrange(5)
    if number_type is NumberType.int:
        if kind == 0:
            return rng.randrange(-(2**40), 2**40)
        return rng.randrange(-500, 500)
    if kind == 0:
        return Fraction(rng.randrange(-4000, 4000), rng.choice([1, 2, 3, 4, 6, 12, 48]))
    if kind == 1:
        return rng.uniform(-100.0, 100.0)  # a float-mirrored dyadic
    if kind == 2:
        return rng.choice([0.1, 0.2, 0.7, -0.3, 1 / 12, 2 / 3])
    if kind == 3:
        return rng.randrange(-50, 50)
    return Fraction(rng.randrange(1, 10**6), rng.randrange(1, 10**6))


def _random_cells(
    rng: random.Random, number_type: NumberType, size: int
) -> pa.StructArray:
    """Cells as number_cell writes them, with nulls and value-only cells."""
    cells: list[dict[str, Any] | None] = []
    for _ in range(size):
        draw = rng.random()
        if draw < 0.03:
            cells.append(None)
        elif draw < 0.06:
            # Value-only cell: the number lives on its float side only.
            whole = float(rng.randrange(-20, 20))
            value = whole if number_type is NumberType.int else whole / 8
            cells.append({"value": value, "numerator": None, "denominator": None})
        else:
            cells.append(number_cell(_random_number(rng, number_type), number_type))
    return pa.array(cells, type=RATIONAL_STRUCT_TYPE)


def _comparable(
    left: pa.StructArray,
    right: pa.StructArray | Any,
    operation: str,
    number_type: NumberType,
    rounding: str,
) -> tuple[pa.StructArray, pa.StructArray | Any, list[dict[str, Any] | None]]:
    """The rows the scalar engine can store, and its cell for each.

    A row whose result the scalar engine refuses (lowest terms past int64) or
    that divides by zero is dropped from both operands: the column engine
    raises on such a row for the whole column, which the refusal tests
    cover. The rows kept still include many whose cross-products exceed
    int64.
    """
    kept: list[int] = []
    cells: list[dict[str, Any] | None] = []
    for position, left_cell in enumerate(left.to_pylist()):
        right_cell = right[position].as_py() if isinstance(right, pa.Array) else right
        if left_cell is None or right_cell is None:
            kept.append(position)
            cells.append(None)
            continue
        left_value = struct_to_coordinate(left_cell, number_type)
        right_value = (
            struct_to_coordinate(right_cell, number_type)
            if isinstance(right_cell, dict)
            else right_cell
        )
        try:
            cell = number_cell(
                combine_numbers(
                    left_value, right_value, operation, number_type, rounding=rounding
                ),
                number_type,
            )
        except (ValueError, ZeroDivisionError):
            continue
        kept.append(position)
        cells.append(cell)
    indices = pa.array(kept, pa.int64())
    right_kept = right.take(indices) if isinstance(right, pa.Array) else right
    return left.take(indices), right_kept, cells


def _crosses_int64(left: pa.StructArray, right: pa.StructArray | Any) -> bool:
    """Whether some row's cross-product ``|n1|·d2`` exceeds int64."""
    for position, cell in enumerate(left.to_pylist()):
        other = right[position].as_py() if isinstance(right, pa.Array) else right
        if cell is None or other is None or cell["numerator"] is None:
            continue
        if isinstance(other, dict):
            if other["denominator"] is None:
                continue
            denominator = other["denominator"]
        else:
            denominator = Fraction(other).denominator
        if abs(cell["numerator"]) * denominator > 2**63 - 1:
            return True
    return False


_OPERATIONS = ["add", "subtract", "multiply", "divide", "floor_divide"]


@pytest.mark.parametrize("number_type", [NumberType.int, NumberType.fraction])
@pytest.mark.parametrize("operation", _OPERATIONS)
def test_column_and_scalar_engine_agree_cell_for_cell(
    number_type: NumberType, operation: str
) -> None:
    rng = random.Random(f"{number_type}-{operation}")
    left, right, expected = _comparable(
        _random_cells(rng, number_type, 400),
        _random_cells(rng, number_type, 400),
        operation,
        number_type,
        "round",
    )
    if number_type is NumberType.fraction and operation in ("add", "subtract"):
        assert _crosses_int64(left, right)

    computed = combine_number_columns(left, right, operation, number_type)

    assert computed.to_pylist() == expected


@pytest.mark.parametrize("rounding", ["round", "floor", "ceil", "truncate"])
@pytest.mark.parametrize("operation", ["multiply", "divide"])
@pytest.mark.parametrize("factor", [Fraction(1, 2), 0.1, Fraction(-7, 3), 3])
def test_int_scaling_rounds_as_the_scalar_engine_does(
    rounding: str, operation: str, factor: Any
) -> None:
    rng = random.Random(f"{rounding}-{operation}-{factor}")
    values = [rng.randrange(-1000, 1000) for _ in range(300)]
    values += [-5, -3, -1, 1, 3, 5, 2**52 + 1]
    left, _, expected = _comparable(
        _column(values, NumberType.int), factor, operation, NumberType.int, rounding
    )

    computed = combine_number_columns(
        left, factor, operation, NumberType.int, rounding=rounding
    )

    assert computed.to_pylist() == expected


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("scalar", [Fraction(1, 3), 0.1, -2, Fraction(5, 1)])
def test_fraction_column_with_a_scalar_agrees_with_the_scalar_engine(
    operation: str, scalar: Any
) -> None:
    rng = random.Random(f"{operation}-{scalar}")
    left, _, expected = _comparable(
        _random_cells(rng, _FRACTION, 300), scalar, operation, _FRACTION, "round"
    )

    computed = combine_number_columns(left, scalar, operation, _FRACTION)

    assert computed.to_pylist() == expected
