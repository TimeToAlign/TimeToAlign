"""``EventData.coordinate_range()`` agrees with the per-cell reading it replaced.

See ``README.md`` (``test_coordinate_range.py``) for the strategy.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import (
    Coordinate,
    EventReference,
    NumberType,
    Projection,
    TimeUnit,
)
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    build_number_struct_array,
    struct_to_coordinate,
)
from timetoalign.storage.events import EventData

_UNITS = {
    NumberType.int: TimeUnit.ticks,
    NumberType.float: TimeUnit.seconds,
    NumberType.fraction: TimeUnit.quarters,
}


def _reference_range(data: EventData) -> tuple[Any, Any] | None:
    """The per-cell reading: every non-null cell decoded, then ``min`` / ``max``."""
    if data.count == 0:
        return None
    values = []
    for name in ("start", "end"):
        for cell in data.table.column(name).to_pylist():
            if cell is None or cell.get("value") is None:
                continue
            values.append(struct_to_coordinate(cell, data.number_type))
    if not values:
        return None
    return min(values), max(values)


def _assert_agrees(data: EventData) -> tuple[Any, Any] | None:
    result = data.coordinate_range()
    expected = _reference_range(data)
    assert result == expected
    assert repr(result) == repr(expected)
    if expected is not None:
        assert [type(end) for end in result] == [type(end) for end in expected]
    return result


def _cells(values: list[Any], number_type: NumberType) -> pa.StructArray:
    return build_number_struct_array(values, number_type=number_type)


def _table(
    starts: list[Any], ends: list[Any], number_type: NumberType, *, chunks: int = 1
) -> EventData:
    """Events whose ``start`` and ``end`` cells are exactly the given numbers."""

    def column(values: list[Any]) -> pa.ChunkedArray:
        size = -(-len(values) // chunks) if values else 0
        bounds = range(0, len(values), size) if size else range(0)
        return pa.chunked_array(
            [_cells(values[i:][:size], number_type) for i in bounds],
            type=RATIONAL_STRUCT_TYPE,
        )

    table = pa.table({"start": column(starts), "end": column(ends)})
    return EventData(table, _UNITS[number_type], number_type)


def _split(values: list[Any], rng: random.Random) -> tuple[list[Any], list[Any]]:
    """The values shuffled over two columns, with nulls between."""
    values = values + [None] * (len(values) // 4)
    rng.shuffle(values)
    half = len(values) // 2
    starts, ends = values[:half], values[half:][:half]
    return starts, ends


class TestTheStoredTablesOfTheOtherTests:
    """The event tables built by the storage and core tests."""

    def test_decimal_and_exact_durations_on_quarters(self) -> None:
        data = EventData.from_dicts(
            [
                {"event_type": "Note", "start": 0, "duration": Fraction(1, 3)},
                {"event_type": "Note", "start": 0, "duration": 0.1},
                {"event_type": "Beat", "instant": 0},
            ],
            unit=TimeUnit.quarters,
        )
        assert _assert_agrees(data) == (Fraction(0), Fraction(1, 3))

    @pytest.mark.parametrize("path", ["from_dicts", "from_arrays"])
    def test_completion_past_int64_cross_products(self, path: str) -> None:
        start = Fraction(1, 5 * 2**40)
        end = Fraction(3 * 2**40 + 1, 3 * 2**40)
        if path == "from_dicts":
            data = EventData.from_dicts(
                [{"start": start, "end": end}], unit=TimeUnit.quarters
            )
        else:
            data = EventData.from_arrays(
                {"start": [start], "end": [end]}, unit=TimeUnit.quarters
            )
        assert _assert_agrees(data) == (start, end)

    @pytest.mark.parametrize("path", ["from_dicts", "from_arrays"])
    def test_ticks_past_two_to_the_53(self, path: str) -> None:
        rows = {
            "start": [2**53 + 1, 1],
            "end": [2**53 + 4, None],
            "duration": [None, 2**53 - 1],
        }
        if path == "from_dicts":
            data = EventData.from_dicts(
                [dict(zip(rows, cells)) for cells in zip(*rows.values())],
                unit=TimeUnit.ticks,
            )
        else:
            data = EventData.from_arrays(rows, unit=TimeUnit.ticks)
        assert _assert_agrees(data) == (1, 2**53 + 4)

    def test_three_decimal_seconds(self) -> None:
        data = EventData.from_arrays(
            {
                "start": [1.234, 2.718, 0.003, 123.457],
                "end": [2.718, 3.1, 1.5, 123.458],
            },
            unit=TimeUnit.seconds,
        )
        assert _assert_agrees(data) == (0.003, 123.458)

    def test_projections(self) -> None:
        data = EventData.from_dicts(
            [
                Projection(
                    start=Coordinate(Fraction(3, 2), TimeUnit.quarters),
                    projection_of=EventReference(timeline_id="clt_6", event_id="x1"),
                ),
                {
                    "event_type": "Note",
                    "start": Fraction(1, 2),
                    "duration": Fraction(1, 2),
                },
            ],
            unit=TimeUnit.quarters,
        )
        assert _assert_agrees(data) == (Fraction(1, 2), Fraction(3, 2))


class TestSeededColumns:
    """Random columns of each number type, compared with the per-cell reading."""

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize(
        ("centre", "denominator"), [(1, 3 * 2**60), (-1, 3 * 2**60), (2**9, 3 * 2**50)]
    )
    def test_fractions_sharing_one_double(
        self, seed: int, centre: int, denominator: int
    ) -> None:
        rng = random.Random(seed)
        values = [
            Fraction(centre * denominator + rng.randrange(-200, 200), denominator)
            for _ in range(300)
        ]
        assert len({float(v) for v in values}) < len(set(values))
        starts, ends = _split(values, rng)
        result = _assert_agrees(_table(starts, ends, NumberType.fraction, chunks=3))
        present = [v for v in starts + ends if v is not None]
        assert result == (min(present), max(present))

    @pytest.mark.parametrize("seed", range(5))
    def test_fractions_huge_and_negative(self, seed: int) -> None:
        rng = random.Random(seed)
        values = [
            Fraction(rng.randrange(-(2**62), 2**62), rng.randrange(1, 2**40))
            for _ in range(200)
        ]
        starts, ends = _split(values, rng)
        result = _assert_agrees(_table(starts, ends, NumberType.fraction, chunks=2))
        present = [v for v in starts + ends if v is not None]
        assert result == (min(present), max(present))

    @pytest.mark.parametrize("seed", range(5))
    def test_integers_past_two_to_the_53(self, seed: int) -> None:
        rng = random.Random(seed)
        values = [2**62 - rng.randrange(2000) for _ in range(150)]
        values += [-(2**62) + rng.randrange(2000) for _ in range(150)]
        assert len({float(v) for v in values}) < len(set(values))
        starts, ends = _split(values, rng)
        result = _assert_agrees(_table(starts, ends, NumberType.int, chunks=3))
        present = [v for v in starts + ends if v is not None]
        assert result == (min(present), max(present))

    @pytest.mark.parametrize("seed", range(5))
    def test_floats(self, seed: int) -> None:
        rng = random.Random(seed)
        values = [rng.uniform(-1e18, 1e18) for _ in range(100)]
        values += [rng.uniform(-1e-6, 1e-6) for _ in range(100)]
        starts, ends = _split(values, rng)
        result = _assert_agrees(_table(starts, ends, NumberType.float, chunks=2))
        present = [v for v in starts + ends if v is not None]
        assert result == (min(present), max(present))


class TestEdges:
    """Signs of zero, one-sided extremes, nulls, sizes."""

    @pytest.mark.parametrize(
        ("starts", "ends", "expected"),
        [
            ([0.0, -0.0], [None, None], "(0.0, 0.0)"),
            ([-0.0, 0.0], [None, None], "(-0.0, -0.0)"),
            ([0.0], [-0.0], "(0.0, 0.0)"),
            ([None, -0.0], [0.0, None], "(-0.0, -0.0)"),
        ],
    )
    def test_the_first_of_equal_floats_answers(
        self, starts: list[Any], ends: list[Any], expected: str
    ) -> None:
        assert repr(_assert_agrees(_table(starts, ends, NumberType.float))) == expected

    def test_the_extremes_in_end_only(self) -> None:
        data = _table(
            [Fraction(1, 3), Fraction(2, 3)],
            [Fraction(-1, 7), Fraction(9, 7)],
            NumberType.fraction,
        )
        assert _assert_agrees(data) == (Fraction(-1, 7), Fraction(9, 7))

    def test_end_entirely_null(self) -> None:
        data = _table([5, -3, 2], [None, None, None], NumberType.int)
        assert _assert_agrees(data) == (-3, 5)

    @pytest.mark.parametrize("number_type", list(_UNITS))
    def test_a_single_row(self, number_type: NumberType) -> None:
        value = {
            NumberType.int: 7,
            NumberType.float: 0.1,
            NumberType.fraction: Fraction(1, 3),
        }
        data = _table([value[number_type]], [None], number_type)
        assert _assert_agrees(data) == (value[number_type], value[number_type])

    @pytest.mark.parametrize("number_type", list(_UNITS))
    def test_every_cell_null(self, number_type: NumberType) -> None:
        assert _assert_agrees(_table([None, None], [None, None], number_type)) is None

    @pytest.mark.parametrize("number_type", list(_UNITS))
    def test_an_empty_table(self, number_type: NumberType) -> None:
        assert _assert_agrees(_table([], [], number_type)) is None

    def test_cells_holding_only_a_value(self) -> None:
        cells = pa.array(
            [
                {"value": 0.1, "numerator": None, "denominator": None},
                {"value": 0.5, "numerator": 1, "denominator": 2},
            ],
            type=RATIONAL_STRUCT_TYPE,
        )
        nulls = pa.nulls(2, type=RATIONAL_STRUCT_TYPE)
        data = EventData(
            pa.table({"start": cells, "end": nulls}),
            TimeUnit.quarters,
            NumberType.fraction,
        )
        assert _assert_agrees(data) == (Fraction(0.1), Fraction(1, 2))
