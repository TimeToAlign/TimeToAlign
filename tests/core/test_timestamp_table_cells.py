"""The timestamp table decoder refuses a cell without a number, as the cell decoder does."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import (
    RATIONAL_STRUCT_TYPE,
    NumberType,
    struct_to_coordinate,
    timestamp_table_to_dataframe,
)
from timetoalign.core.fields import blob_metadata

NUMERATOR_ONLY = {"value": None, "numerator": 5, "denominator": None}


def _table(cells: list[dict[str, Any] | None], **metadata: str) -> pa.Table:
    column = pa.array(cells, type=RATIONAL_STRUCT_TYPE)
    field = pa.field("x", RATIONAL_STRUCT_TYPE, metadata=blob_metadata(**metadata))
    return pa.Table.from_arrays([column], schema=pa.schema([field]))


@pytest.mark.parametrize(
    "metadata",
    [{"number_type": "fraction"}, {"unit": "quarters"}],
    ids=["number_type", "unit"],
)
def test_a_cell_without_a_number_is_refused(metadata: dict[str, str]) -> None:
    half = {"value": 0.5, "numerator": 1, "denominator": 2}
    table = _table([half, NUMERATOR_ONLY], **metadata)
    with pytest.raises(ValueError) as refused:
        timestamp_table_to_dataframe(table)
    assert str(refused.value) == (
        "cell 1 of column carries no usable value: no ratio and no float side"
    )


def test_the_cell_decoder_refuses_the_same_cell() -> None:
    with pytest.raises(ValueError):
        struct_to_coordinate(NUMERATOR_ONLY, NumberType.fraction)


def test_an_incomplete_ratio_with_a_float_side_still_decodes() -> None:
    half = {"value": 0.5, "numerator": 1, "denominator": 2}
    with_float = {"value": 2.5, "numerator": 5, "denominator": None}
    nothing = {"value": None, "numerator": None, "denominator": None}
    table = _table([half, with_float, nothing], number_type="fraction")
    column = timestamp_table_to_dataframe(table).iloc[:, 0]
    assert column.tolist() == [Fraction(1, 2), Fraction(5, 2), None]
    assert [type(cell) for cell in column.tolist()] == [Fraction, Fraction, type(None)]
