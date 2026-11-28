"""Exact-value tests for ``EventData.column_values()``.

See ``README.md`` for the documented contract.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timetoalign import storage
from timetoalign.core import (
    BoundingBox,
    BoundingBoxField,
    Box,
    BoxField,
    Coordinate,
    Duration,
    EnharmonicPitch,
    Event,
    EventReference,
    IntervalPolicy,
    Note,
    NoteField,
    NumberType,
    Projection,
    ProjectionField,
    Quadrilateral,
    QuadrilateralField,
    ScoreNote,
    ScoreNoteField,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.core.ids import IdMinter, reset_content_minters
from timetoalign.core.time import (
    RATIONAL_STRUCT_TYPE,
    CoordinateField,
    build_number_struct_array,
)
from timetoalign.storage.events import EventData, mint_event_ids
from timetoalign.timelines import ContinuousLogicalTimeline


class TestColumnValues:
    """Decoded, plain-Python column reads via ``column_values()``."""

    def _data(self) -> EventData:
        return EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0,
                    "duration": Fraction(1, 3),
                    "flag": True,
                },
                {
                    "event_type": "Note",
                    "start": 0,
                    "duration": 0.1,
                    "flag": None,
                },
                {
                    "event_type": "Beat",
                    "instant": 0,
                    "flag": False,
                },
            ],
            unit=TimeUnit.quarters,
        )

    def test_rational_column_decodes_exact_fraction(self) -> None:
        data = self._data()
        values = data.column_values("duration", default=Fraction(0))
        assert values[0] == Fraction(1, 3)

    def test_rational_column_falls_back_to_float_value(self) -> None:
        data = self._data()
        values = data.column_values("duration", default=Fraction(0))
        # duration=0.1 carries no exact numerator/denominator, so the
        # float `value` is used, wrapped in a Fraction.
        assert values[1] == Fraction(0.1)

    def test_null_coordinate_cell_yields_default(self) -> None:
        data = self._data()
        values = data.column_values("duration", default=Fraction(4))
        # The instant event has no duration at all.
        assert values[2] == Fraction(4)

    def test_null_plain_cell_passes_through_as_none(self) -> None:
        # `default` only substitutes for a missing *column* or a null
        # coordinate struct; a null plain-column cell is not a coordinate
        # struct, so it passes through as `None` from `to_pylist()`.
        data = self._data()
        values = data.column_values("flag", default="missing")
        assert values == [True, None, False]

    def test_missing_column_yields_default_list(self) -> None:
        data = self._data()
        values = data.column_values("no_such_column", default="fallback")
        assert values == ["fallback"] * len(data)

    def test_non_coordinate_column_passes_through(self) -> None:
        data = self._data()
        assert data.column_values("event_type") == ["Note", "Note", "Beat"]


class TestLegacyFloatRatioMembers:
    """Compatibility with rational cells persisted by earlier releases."""

    @staticmethod
    def _legacy_type() -> pa.StructType:
        return pa.struct(
            [
                pa.field("value", pa.float64()),
                pa.field("numerator", pa.float64()),
                pa.field("denominator", pa.float64()),
            ]
        )

    def test_to_dataframe_decodes_integer_valued_float_members(self) -> None:
        table = pa.table(
            {
                "start": pa.array(
                    [
                        {
                            "value": 8379000.0,
                            "numerator": 8379000.0,
                            "denominator": 1.0,
                        }
                    ],
                    type=self._legacy_type(),
                )
            }
        )
        data = EventData(table, TimeUnit.quarters, NumberType.fraction)

        assert data.to_dataframe()["start"].tolist() == [Fraction(8379000, 1)]

    def test_from_dicts_does_not_infer_carried_coordinate_struct_members(self) -> None:
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": {
                        "value": 8379000.0,
                        "numerator": 8379000.0,
                        "denominator": 1.0,
                    },
                    "legacy_coordinate": {
                        "value": 2.5,
                        "numerator": 5.0,
                        "denominator": 2.0,
                    },
                }
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        start = data.table.column("start").combine_chunks()
        column = data.table.column("legacy_coordinate").combine_chunks()

        assert start.to_pylist() == [
            {"value": 8379000.0, "numerator": 8379000, "denominator": 1}
        ]
        assert column.type.field("numerator").type == pa.float64()
        assert column.type.field("denominator").type == pa.float64()
        assert column.to_pylist() == [
            {"value": 2.5, "numerator": 5.0, "denominator": 2.0}
        ]
        assert data.to_dataframe()["start"].tolist() == [Fraction(8379000, 1)]
        assert data.to_dataframe()["legacy_coordinate"].tolist() == [Fraction(5, 2)]


class TestCreateTimeline:
    """EventData timelines retain inferred type and selected event rows."""

    def test_create_timeline_places_selected_events_directly(self) -> None:
        """Quarter-note EventData creates a continuous logical event timeline."""
        data = TestColumnValues()._data()
        timeline = data.create_timeline()

        assert type(timeline) is ContinuousLogicalTimeline
        assert timeline.n_events == 3

    def test_create_timeline_filters_before_assigning_events(self) -> None:
        """The public filter parameter restricts the direct event assignment."""
        data = TestColumnValues()._data()
        timeline = data.create_timeline(filters={"event_type": "Note"})

        assert type(timeline) is ContinuousLogicalTimeline
        assert timeline.n_events == 2


class TestIntervalFractionFidelity:
    """Validate exact pairs produced while completing interval coordinates."""

    @staticmethod
    def _pairs(data: EventData, name: str) -> list[tuple[int | None, int | None]]:
        """Return numerator/denominator pairs from a coordinate column."""
        column = data._table.column(name).combine_chunks()
        return list(
            zip(
                column.field("numerator").to_pylist(),
                column.field("denominator").to_pylist(),
                strict=True,
            )
        )

    @staticmethod
    def _coord_type() -> pa.StructType:
        """Return the canonical coordinate struct type used by the test arrays."""
        return pa.struct(
            [
                pa.field("value", pa.float64()),
                pa.field("numerator", pa.int64()),
                pa.field("denominator", pa.int64()),
            ]
        )

    def test_from_dicts_computes_exact_end(self) -> None:
        """Compute an exact end from exact start and duration structs."""
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": {
                        "value": 3.5,
                        "numerator": 7,
                        "denominator": 2,
                    },
                    "duration": {
                        "value": 0.75,
                        "numerator": 3,
                        "denominator": 4,
                    },
                }
            ],
            unit=TimeUnit.quarters,
        )

        assert data.column_values("end") == [Fraction(17, 4)]
        assert self._pairs(data, "end") == [(17, 4)]

    def test_from_arrays_computes_exact_end(self) -> None:
        """Compute an exact end in the vectorized construction path."""
        coord_type = self._coord_type()
        data = EventData.from_arrays(
            {
                "id": ["exact"],
                "event_type": ["Note"],
                "start": pa.array(
                    [{"value": 3.5, "numerator": 7, "denominator": 2}],
                    type=coord_type,
                ),
                "duration": pa.array(
                    [{"value": 0.75, "numerator": 3, "denominator": 4}],
                    type=coord_type,
                ),
            },
            unit=TimeUnit.quarters,
        )

        assert data.column_values("end") == [Fraction(17, 4)]
        assert self._pairs(data, "end") == [(17, 4)]

    def test_float_only_computed_fields_still_carry_both_sides(self) -> None:
        """Bare float inputs still compute a coordinate with both sides.

        Neither operand arrives with a ratio, but a stored cell always has
        one, so the computed ``end`` carries the exact pair alongside its
        double rather than a populated value and a hole.
        """
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 3.5, "duration": 0.75}],
            unit=TimeUnit.quarters,
        )

        assert data.column_values("end") == [Fraction(4.25)]
        assert self._pairs(data, "end") == [(17, 4)]

    def test_mixed_rows_preserve_exactness_per_row(self) -> None:
        """Preserve exact output only for rows with complete exact operands."""
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": {"value": 3.5, "numerator": 7, "denominator": 2},
                    "duration": {
                        "value": 0.75,
                        "numerator": 3,
                        "denominator": 4,
                    },
                },
                {"event_type": "Note", "start": 3.5, "duration": 0.75},
            ],
            unit=TimeUnit.quarters,
        )

        assert data.column_values("end") == [Fraction(17, 4), Fraction(4.25)]
        assert self._pairs(data, "end") == [(17, 4), (17, 4)]


# The seconds of a real annotation file: three decimals, no short ratio. The
# last two rows span magnitudes: the exact difference of their dyadic mirrors
# needs more than 53 bits (and, for the last, a numerator past int64).
_SECONDS_STARTS = [0.0, 1.234, 2.718, 0.003, 0.001]
_SECONDS_ENDS = [1.234, 2.718, 3.1, 1.5, 123.457]
# The IEEE differences of the stored doubles, and the sums back.
_SECONDS_DURATIONS = [1.234, 1.484, 0.3820000000000001, 1.497, 123.45599999999999]
_SECONDS_SUMS = [1.234, 2.718, 3.1, 1.5, 123.457]

# Two quarter positions whose denominators multiply past 2**63; their
# difference does not need that much.
_WIDE_END = Fraction(3 * 2**40 + 1, 3 * 2**40)
_WIDE_START = Fraction(1, 5 * 2**40)
_WIDE_DURATION = Fraction(15 * 2**39 + 1, 15 * 2**39)


def _rows(starts: list, **others: list) -> list[dict]:
    """Event rows of one note per start, with the other coordinate given."""
    ((key, values),) = others.items()
    return [
        {"event_type": "Note", "start": start, key: value}
        for start, value in zip(starts, values, strict=True)
    ]


def _arrays(starts: list, **others: list) -> dict:
    """The same events as columns, for ``EventData.from_arrays``."""
    ((key, values),) = others.items()
    return {"event_type": ["Note"] * len(starts), "start": starts, key: values}


def _cells(data: EventData, name: str) -> list[dict]:
    return data.table.column(name).combine_chunks().to_pylist()


def _assert_mirrors_value(cells: list[dict]) -> None:
    """Float axis: every ratio is the exact mirror of the stored double."""
    for cell in cells:
        assert cell["denominator"] > 0
        assert Fraction(cell["numerator"], cell["denominator"]) == Fraction(
            cell["value"]
        )


def _assert_value_mirrors_ratio(cells: list[dict]) -> None:
    """Exact axis: every double is the mirror of the stored ratio."""
    for cell in cells:
        assert cell["denominator"] > 0
        assert cell["value"] == float(Fraction(cell["numerator"], cell["denominator"]))


def _build(path: str, starts: list, unit: TimeUnit, **others: list) -> EventData:
    if path == "row":
        return EventData.from_dicts(_rows(starts, **others), unit=unit)
    return EventData.from_arrays(_arrays(starts, **others), unit=unit)


# A duration a float tolerance would take for 1/3.
_NEAR_THIRD = Fraction(1, 3) + Fraction(1, 2**40)
_NEAR_MISS = (
    "start=0, end=1/3, duration=1099511627779/3298534883328 " "(expected duration=1/3)"
)
_NEAR_MISS_WARNINGS = {
    "row": f"Interval inconsistency: {_NEAR_MISS}. Recomputing duration from end.",
    "column": (
        "Interval inconsistency in 1 event(s): end - start != duration. "
        f"Recomputing duration from end. First at index 0: {_NEAR_MISS}."
    ),
}
_NEAR_MISS_REFUSALS = {
    "row": f"Interval inconsistency: {_NEAR_MISS}.",
    "column": (
        "Interval inconsistency in 1 event(s): end - start != duration. "
        f"First at index 0: {_NEAR_MISS}."
    ),
}


def _stated(
    path: str,
    unit: TimeUnit,
    start: Any,
    end: Any,
    duration: Any,
    policy: IntervalPolicy,
) -> EventData:
    """One note stating start, end and duration, built on *path*."""
    if path == "row":
        return EventData.from_dicts(
            [{"event_type": "Note", "start": start, "end": end, "duration": duration}],
            unit=unit,
            interval_policy=policy,
        )
    return EventData.from_arrays(
        {
            "event_type": ["Note"],
            "start": [start],
            "end": [end],
            "duration": [duration],
        },
        unit=unit,
        interval_policy=policy,
    )


def _ratios(data: EventData, name: str) -> list[Fraction]:
    return [
        Fraction(cell["numerator"], cell["denominator"]) for cell in _cells(data, name)
    ]


class TestIntervalCompletion:
    """Completion of ``end`` / ``duration`` in the declared number type."""

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_seconds_duration_is_the_float_difference(self, path: str) -> None:
        data = _build(path, _SECONDS_STARTS, TimeUnit.seconds, end=_SECONDS_ENDS)
        durations = _cells(data, "duration")
        assert [cell["value"] for cell in durations] == _SECONDS_DURATIONS
        for name in ("start", "end", "duration"):
            _assert_mirrors_value(_cells(data, name))

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_seconds_end_is_the_float_sum(self, path: str) -> None:
        data = _build(
            path, _SECONDS_STARTS, TimeUnit.seconds, duration=_SECONDS_DURATIONS
        )
        ends = _cells(data, "end")
        assert [cell["value"] for cell in ends] == _SECONDS_SUMS
        for name in ("start", "end", "duration"):
            _assert_mirrors_value(_cells(data, name))

    def test_ten_thousand_seconds_rows(self, tmp_path) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        starts = np.round(rng.uniform(0, 300, 10_000), 3)
        ends = starts + np.round(rng.uniform(0.05, 2, 10_000), 3)
        data = EventData.from_arrays(
            {"event_type": np.array(["Note"] * 10_000), "start": starts, "end": ends},
            unit=TimeUnit.seconds,
        )
        durations = _cells(data, "duration")
        assert sum(1 for cell in durations if cell["denominator"] <= 0) == 0
        _assert_mirrors_value(durations)
        assert [cell["value"] for cell in durations] == (ends - starts).tolist()

        path = tmp_path / "seconds.parquet"
        data.to_parquet(path)
        read = EventData.from_parquet(path)
        for name in ("start", "end", "duration"):
            assert _cells(read, name) == _cells(data, name)

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_quarters_beyond_int64_cross_products(self, path: str) -> None:
        by_end = _build(path, [_WIDE_START], TimeUnit.quarters, end=[_WIDE_END])
        (duration,) = _cells(by_end, "duration")
        assert Fraction(duration["numerator"], duration["denominator"]) == (
            _WIDE_DURATION
        )
        _assert_value_mirrors_ratio(_cells(by_end, "duration"))

        by_duration = _build(
            path, [_WIDE_START], TimeUnit.quarters, duration=[_WIDE_DURATION]
        )
        (end,) = _cells(by_duration, "end")
        assert Fraction(end["numerator"], end["denominator"]) == _WIDE_END
        _assert_value_mirrors_ratio(_cells(by_duration, "end"))

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_ticks_complete_as_integers(self, path: str) -> None:
        by_end = _build(path, [0, 480, 960], TimeUnit.ticks, end=[480, 1200, 961])
        assert _cells(by_end, "duration") == [
            {"value": 480.0, "numerator": 480, "denominator": 1},
            {"value": 720.0, "numerator": 720, "denominator": 1},
            {"value": 1.0, "numerator": 1, "denominator": 1},
        ]
        by_duration = _build(
            path, [0, 480, 960], TimeUnit.ticks, duration=[480, 720, 1]
        )
        assert _cells(by_duration, "end") == [
            {"value": 480.0, "numerator": 480, "denominator": 1},
            {"value": 1200.0, "numerator": 1200, "denominator": 1},
            {"value": 961.0, "numerator": 961, "denominator": 1},
        ]

    @pytest.mark.parametrize("path", ["row", "column"])
    @pytest.mark.parametrize(
        ("policy", "end", "duration", "warned"),
        [
            (IntervalPolicy.warn, Fraction(1, 3), Fraction(1, 3), True),
            (IntervalPolicy.prefer_end, Fraction(1, 3), Fraction(1, 3), False),
            (IntervalPolicy.prefer_duration, _NEAR_THIRD, _NEAR_THIRD, False),
        ],
    )
    def test_quarters_near_miss_is_inconsistent(
        self,
        path: str,
        policy: IntervalPolicy,
        end: Fraction,
        duration: Fraction,
        warned: bool,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="timetoalign.storage.events"):
            data = _stated(
                path, TimeUnit.quarters, 0, Fraction(1, 3), _NEAR_THIRD, policy
            )
        assert _ratios(data, "end") == [end]
        assert _ratios(data, "duration") == [duration]
        assert [record.getMessage() for record in caplog.records] == (
            [_NEAR_MISS_WARNINGS[path]] if warned else []
        )

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_quarters_near_miss_is_refused_under_strict(self, path: str) -> None:
        with pytest.raises(ValueError) as refused:
            _stated(
                path,
                TimeUnit.quarters,
                0,
                Fraction(1, 3),
                _NEAR_THIRD,
                IntervalPolicy.strict,
            )
        assert str(refused.value) == _NEAR_MISS_REFUSALS[path]

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_ticks_inconsistency_is_exact(self, path: str) -> None:
        with pytest.raises(ValueError, match="^Interval inconsistency"):
            _stated(path, TimeUnit.ticks, 0, 4, 3, IntervalPolicy.strict)
        data = _stated(path, TimeUnit.ticks, 0, 4, 3, IntervalPolicy.warn)
        assert _cells(data, "duration") == [
            {"value": 4.0, "numerator": 4, "denominator": 1}
        ]

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_seconds_consistency_is_a_float_comparison(self, path: str) -> None:
        data = _stated(path, TimeUnit.seconds, 0.1, 0.3, 0.2, IntervalPolicy.strict)
        assert [cell["value"] for cell in _cells(data, "duration")] == [0.2]
        assert [cell["value"] for cell in _cells(data, "end")] == [0.3]

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_int_past_two_to_the_53_is_exact(self, path: str) -> None:
        data = _build(path, [2**53 + 1], TimeUnit.ticks, end=[2**53 + 4])
        assert _cells(data, "duration") == [
            {"value": 3.0, "numerator": 3, "denominator": 1}
        ]
        data = _build(path, [1], TimeUnit.ticks, duration=[2**53 - 1])
        assert _cells(data, "end") == [
            {"value": 9007199254740992.0, "numerator": 2**53, "denominator": 1}
        ]

    @pytest.mark.parametrize("path", ["row", "column"])
    def test_fraction_duration_is_exact_past_int64_cross_products(
        self, path: str
    ) -> None:
        starts, ends = [0.1, 0.7], [0.3, 0.9]
        if path == "row":
            rows = [
                {"event_type": "Note", "start": start, "end": end}
                for start, end in zip(starts, ends, strict=True)
            ]
            data = EventData.from_dicts(
                rows, unit=TimeUnit.quarters, number_type=NumberType.fraction
            )
        else:
            data = EventData.from_arrays(
                {"event_type": ["Note"] * 2, "start": starts, "end": ends},
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
        assert _cells(data, "duration") == [
            {
                "value": 0.19999999999999998,
                "numerator": 7205759403792793,
                "denominator": 2**55,
            },
            {
                "value": 0.20000000000000007,
                "numerator": 1801439850948199,
                "denominator": 2**53,
            },
        ]

    def test_fraction_end_past_int64_in_lowest_terms_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_arrays(
                {
                    "event_type": ["Note"] * 2,
                    "start": [Fraction(0), Fraction(1129, 12)],
                    "duration": [Fraction(1), 0.1],
                },
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
        assert str(refused.value) == (
            "Completing the events' end on a fraction axis: row 1: 1129/12 + "
            "3602879701896397/36028797018963968 = "
            "10179936597708269159/108086391056891904, whose numerator "
            "10179936597708269159 in lowest terms does not fit int64, the width "
            "of an exact cell; the result cannot be stored exactly. Give the "
            "operands exact values (a ratio written as a ratio, not as a "
            "decimal), or declare the axis number_type float"
        )

    @pytest.mark.parametrize(
        ("unit", "starts", "others"),
        [
            (TimeUnit.seconds, _SECONDS_STARTS, {"end": _SECONDS_ENDS}),
            (TimeUnit.seconds, _SECONDS_STARTS, {"duration": _SECONDS_DURATIONS}),
            (TimeUnit.ticks, [0, 480, 960], {"end": [480, 1200, 961]}),
            (TimeUnit.ticks, [0, 480, 960], {"duration": [480, 720, 1]}),
            (TimeUnit.quarters, [Fraction(1, 3), _WIDE_START], {"end": [2, _WIDE_END]}),
            (
                TimeUnit.quarters,
                [Fraction(1, 3), _WIDE_START],
                {"duration": [Fraction(3, 4), _WIDE_DURATION]},
            ),
        ],
    )
    def test_row_and_column_paths_give_the_same_cells(
        self, unit: TimeUnit, starts: list, others: dict
    ) -> None:
        row = _build("row", starts, unit, **others)
        column = _build("column", starts, unit, **others)
        for name in ("start", "end", "duration"):
            assert _cells(row, name) == _cells(column, name)


class TestHead:
    """`EventData.head()` — pandas-style leading-row preview."""

    @staticmethod
    def _data(count: int = 6) -> EventData:
        return EventData.from_dicts(
            [{"event_type": "Beat", "instant": Fraction(i, 4)} for i in range(count)],
            unit=TimeUnit.quarters,
        )

    def test_default_returns_first_five_rows(self) -> None:
        df = self._data(6).head()
        assert len(df) == 5
        assert list(df["event_type"]) == ["Beat"] * 5

    def test_explicit_n_returns_that_many_rows(self) -> None:
        df = self._data(6).head(3)
        assert len(df) == 3
        assert list(df["start"]) == [Fraction(0, 4), Fraction(1, 4), Fraction(2, 4)]

    def test_n_larger_than_count_returns_all_rows(self) -> None:
        data = self._data(3)
        assert len(data.head(10)) == 3

    def test_non_positive_n_returns_empty_frame(self) -> None:
        assert len(self._data(6).head(0)) == 0
        assert len(self._data(6).head(-2)) == 0

    def test_equivalent_to_to_dataframe_head(self) -> None:
        # head(n) is the redundancy target for `.to_dataframe().head(n)`
        # (and for the raw `.table.slice(0, n).to_pandas()` idiom): the
        # preview must match the leading rows of the full conversion.
        data = self._data(6)
        from pandas.testing import assert_frame_equal

        assert_frame_equal(
            data.head(4).reset_index(drop=True),
            data.to_dataframe().head(4).reset_index(drop=True),
        )

    def test_coordinate_columns_render_as_numbers(self) -> None:
        # Not raw struct dicts — the same conversion `to_dataframe` applies.
        df = self._data(2).head()
        assert list(df["start"]) == [Fraction(0, 4), Fraction(1, 4)]


class TestMintedIds:
    """One minting function; a declared ``projection_of`` column."""

    TYPES = ["Note", "Projection", "Note", None, ""]

    @staticmethod
    def _rows() -> list[dict[str, Any]]:
        return [
            {"event_type": "Note", "start": 0},
            {"event_type": "Note", "id": "given", "start": 1},
            {"event_type": "Beat", "start": 2},
            {"event_type": "Note", "start": 3},
        ]

    def test_mint_event_ids(self) -> None:
        assert storage.mint_event_ids is mint_event_ids
        expected = IdMinter(5)
        notes = [expected.mint("note"), expected.mint("note")]
        projection = expected.mint("projection")
        events = [expected.mint("event"), expected.mint("event")]
        assert mint_event_ids(self.TYPES, IdMinter(5)).to_pylist() == [
            notes[0],
            projection,
            notes[1],
            *events,
        ]

    def test_mint_event_ids_without_a_minter_is_content_seeded(self) -> None:
        first = mint_event_ids(self.TYPES).to_pylist()
        second = mint_event_ids(self.TYPES).to_pylist()
        assert len(set(first) | set(second)) == 10
        reset_content_minters()
        assert mint_event_ids(self.TYPES).to_pylist() == first

    def test_from_dicts_keeps_given_ids_and_mints_from_the_seed(self) -> None:
        data = EventData.from_dicts(self._rows(), TimeUnit.samples, seed=3)
        minted = mint_event_ids(["Note", "Beat", "Note"], IdMinter(3)).to_pylist()
        assert data.column_values("id") == [minted[0], "given", minted[1], minted[2]]
        again = EventData.from_dicts(self._rows(), TimeUnit.samples, seed=3)
        assert again.table.equals(data.table)

    def test_from_dicts_continues_a_given_minter(self) -> None:
        minter = IdMinter(3)
        first = EventData.from_dicts(self._rows(), TimeUnit.samples, minter=minter)
        second = EventData.from_dicts(self._rows(), TimeUnit.samples, minter=minter)
        expected = IdMinter(3)
        drawn = [
            mint_event_ids(["Note", "Beat", "Note"], expected).to_pylist()
            for _ in range(2)
        ]
        assert first.column_values("id") == [drawn[0][0], "given", *drawn[0][1:]]
        assert second.column_values("id") == [drawn[1][0], "given", *drawn[1][1:]]

    def test_from_dicts_without_a_seed_is_content_seeded(self) -> None:
        first = EventData.from_dicts(self._rows(), TimeUnit.samples)
        second = EventData.from_dicts(self._rows(), TimeUnit.samples)
        minted = {*first.column_values("id"), *second.column_values("id")} - {"given"}
        assert len(minted) == 6
        assert first.table.column("id").null_count == 0
        reset_content_minters()
        again = EventData.from_dicts(self._rows(), TimeUnit.samples)
        assert again.column_values("id") == first.column_values("id")

    def test_from_arrays_mints_an_absent_column_and_a_null(self) -> None:
        types = ["Projection", "Projection", "Note"]
        absent = EventData.from_arrays(
            {"event_type": types, "start": [0, 1, 2]}, TimeUnit.samples, seed=8
        )
        assert (
            absent.column_values("id") == mint_event_ids(types, IdMinter(8)).to_pylist()
        )
        with_null = EventData.from_arrays(
            {
                "id": pa.array(["a", None, "c"], pa.string()),
                "event_type": types,
                "start": [0, 1, 2],
            },
            TimeUnit.samples,
            seed=8,
        )
        assert with_null.column_values("id") == [
            "a",
            IdMinter(8).mint("projection"),
            "c",
        ]
        unseeded = EventData.from_arrays(
            {"event_type": types, "start": [0, 1, 2]}, TimeUnit.samples
        )
        assert unseeded.table.column("id").null_count == 0
        assert set(unseeded.column_values("id")).isdisjoint(absent.column_values("id"))

    def test_from_dataframe_passes_the_seed_on(self) -> None:
        frame = pd.DataFrame({"event_type": ["Note", "Beat"], "start": [0, 1]})
        data = EventData.from_dataframe(frame, TimeUnit.samples, seed=2)
        assert (
            data.column_values("id")
            == mint_event_ids(["Note", "Beat"], IdMinter(2)).to_pylist()
        )

    @pytest.mark.parametrize("constructor", ["from_dicts", "from_arrays"])
    def test_seed_and_minter_together_are_refused(self, constructor: str) -> None:
        build = getattr(EventData, constructor)
        source = (
            self._rows()
            if constructor == "from_dicts"
            else {"event_type": ["Note"], "start": [0]}
        )
        with pytest.raises(TypeError, match="Give seed or minter, not both"):
            build(source, TimeUnit.samples, seed=1, minter=IdMinter(1))

    def test_projection_of_is_typed_by_declaration(self) -> None:
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Projection",
                    "id": "a",
                    "start": 0,
                    "projection_of": None,
                }
            ],
            TimeUnit.samples,
        )
        column = data.schema.field("projection_of")
        assert column.type == pa.struct(
            [pa.field("timeline_id", pa.string()), pa.field("event_id", pa.string())]
        )
        assert field_metadata(column)["field_type"] == "EventReferenceField"
        data = data.concat(
            EventData.from_dicts(
                [
                    {
                        "event_type": "Projection",
                        "id": "b",
                        "start": 1,
                        "projection_of": {
                            "timeline_id": "clt_6",
                            "event_id": "x1azmn68",
                        },
                    }
                ],
                TimeUnit.samples,
            )
        )
        assert data.column_values("projection_of") == [
            None,
            {"timeline_id": "clt_6", "event_id": "x1azmn68"},
        ]


class TestProjectionRows:
    """A Projection object stands for its event row."""

    def test_from_dicts_takes_projections_beside_dicts(self) -> None:
        data = EventData.from_dicts(
            [
                Projection(
                    start=Coordinate(2, TimeUnit.samples),
                    duration=Duration(3, TimeUnit.samples),
                    projection_of=EventReference(timeline_id="clt_6", event_id="x1"),
                    id="p1",
                ),
                {"event_type": "Note", "start": 7, "id": "n1"},
            ],
            TimeUnit.samples,
        )
        rows = data.table.select(
            ["id", "event_type", "start", "duration", "projection_of"]
        ).to_pylist()
        assert rows == [
            {
                "id": "p1",
                "event_type": "Projection",
                "start": {"value": 2.0, "numerator": 2, "denominator": 1},
                "duration": {"value": 3.0, "numerator": 3, "denominator": 1},
                "projection_of": {"timeline_id": "clt_6", "event_id": "x1"},
            },
            {
                "id": "n1",
                "event_type": "Note",
                "start": {"value": 7.0, "numerator": 7, "denominator": 1},
                "duration": None,
                "projection_of": None,
            },
        ]

    def test_one_projection_with_its_id(self) -> None:
        data = EventData.from_dicts(
            Projection(id="given", start=Coordinate(2, TimeUnit.samples)),
            TimeUnit.samples,
        )
        assert data.column_values("id") == ["given"]

    def test_a_coordinate_in_the_unit_is_written_in_the_number_type(self) -> None:
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": Coordinate(1.5, TimeUnit.quarters)}],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        assert data.table.column("start").to_pylist() == [
            {"value": 1.5, "numerator": 3, "denominator": 2}
        ]

    def test_a_coordinate_in_another_unit_is_refused(self) -> None:
        with pytest.raises(
            ValueError,
            match=(
                "^The event's start is in seconds, the events are in samples; add it "
                "to a timeline, which converts it through its C-Map$"
            ),
        ):
            EventData.from_dicts(
                Projection(start=Coordinate(2.0, TimeUnit.seconds)), TimeUnit.samples
            )


class TestBoxRows:
    """A Box stands for its event row; its shape columns are declared."""

    @staticmethod
    def _measure(unit: str = "pixels", number_type: str | None = None) -> Quadrilateral:
        """Measure 1 of *Aimons-nous*, page 1 (DMOS), in the canonical order."""
        return Quadrilateral.from_points(
            [(1221, 1589), (2151, 1589), (2161, 2266), (1221, 2266)],
            unit=unit,
            number_type=number_type,
        )

    def test_from_dicts_takes_a_box(self) -> None:
        m1 = self._measure()
        data = EventData.from_dicts(
            [Box.from_shape(m1, origin=1221, id="m1").to_event_row()],
            TimeUnit.pixels,
            NumberType.int,
        )
        schema = data.table.schema
        for name, field_cls in (
            ("bounding_box", BoundingBoxField),
            ("quadrilateral", QuadrilateralField),
        ):
            assert schema.field(name).type == field_cls.pa_schema
            assert field_metadata(schema.field(name)) == {
                "field_type": field_cls.field_type(),
                "unit": "pixels",
                "domain": "graphical",
                "number_type": "int",
                "version": 1,
            }
        assert data.table.column("bounding_box").to_pylist() == [None]
        assert data.get_field(Quadrilateral)[0] == m1
        row = data.table.select(["id", "event_type", "start", "duration"]).to_pylist()
        assert row == [
            {
                "id": "m1",
                "event_type": "Box",
                "start": {"value": 0.0, "numerator": 0, "denominator": 1},
                "duration": {"value": 940.0, "numerator": 940, "denominator": 1},
            }
        ]

    def test_a_shape_in_another_unit_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [{"event_type": "Box", "start": 0, "quadrilateral": self._measure()}],
                TimeUnit.css_pixels,
                NumberType.float,
            )
        assert str(refused.value) == (
            "The event's quadrilateral is in pixels as int, the events are in "
            "css_pixels as float; a shape is never converted onto the events' axis."
        )

    def test_a_shape_in_another_number_type_is_refused(self) -> None:
        third = BoundingBox.from_xywh(
            Fraction(1, 3),
            0,
            Fraction(2, 3),
            1,
            unit="user_units",
            number_type="fraction",
        )
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                Box.from_shape(third, id="f"), TimeUnit.user_units, NumberType.float
            )
        assert str(refused.value) == (
            "The event's bounding_box is in user_units as fraction, the events are "
            "in user_units as float; a shape is never converted onto the events' "
            "axis."
        )

    @pytest.mark.parametrize(
        "value",
        [
            {"x": 1, "y": 2, "width": 3, "height": 4},
            "left",
            BoundingBox.from_xywh(1, 2, 3, 4, unit="pixels"),
        ],
    )
    def test_another_event_type_giving_a_shape_column_is_refused(
        self, value: Any
    ) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [{"event_type": "Note", "start": 0, "bounding_box": value}],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert str(refused.value) == (
            "An event of type 'Note' gives bounding_box: the columns bounding_box "
            "and quadrilateral hold the shape of a Box and belong to Box events "
            "only."
        )

    def test_a_box_giving_a_dict_form_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [
                    {
                        "event_type": "Box",
                        "start": 1,
                        "duration": 3,
                        "bounding_box": {"x": 1, "y": 2, "width": 3, "height": 4},
                    }
                ],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert str(refused.value) == (
            "A Box's bounding_box is a BoundingBox, not {'x': 1, 'y': 2, 'width': 3, "
            "'height': 4}. Read a dict form with BoundingBox.from_dict."
        )

    def test_a_tables_own_rows_read_back(self) -> None:
        data = EventData.from_dicts(
            [
                {"event_type": "Note", "start": 0, "duration": 5},
                Box.from_shape(self._measure(), origin=1221, id="m1"),
            ],
            TimeUnit.pixels,
            NumberType.int,
        )

        again = EventData.from_dicts(list(data), TimeUnit.pixels, NumberType.int)

        assert again.table.to_pylist() == data.table.to_pylist()
        assert again.table.schema == data.table.schema

    def test_a_row_without_an_event_type_giving_a_shape_is_named_so(self) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [
                    {
                        "start": 0,
                        "duration": 1,
                        "bounding_box": BoundingBox.from_xywh(
                            1, 2, 3, 4, unit="pixels"
                        ),
                    }
                ],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert str(refused.value) == (
            "An event without an event type gives bounding_box: the columns "
            "bounding_box and quadrilateral hold the shape of a Box and belong to "
            "Box events only."
        )

    @staticmethod
    def _cell(value: Any) -> dict[str, Any]:
        """The memory cell of an exact value: its float, numerator, denominator."""
        value = Fraction(value)
        return {
            "value": float(value),
            "numerator": value.numerator,
            "denominator": value.denominator,
        }

    def _rect_h2_row(self, **cells: Any) -> dict[str, Any]:
        """``rect_h2`` as a dict row, its bounding box given as memory cells."""
        box = {
            name: self._cell(value)
            for name, value in {"x": 385, "y": 46, "width": 139, "height": 20}.items()
        }
        return {
            "event_type": "Box",
            "id": "a",
            "start": 378,
            "duration": 139,
            "bounding_box": {**box, **cells},
        }

    def test_a_cell_the_axis_cannot_hold_is_refused_as_its_shape_is(self) -> None:
        with pytest.raises(ValueError) as constructed:
            BoundingBox.from_xywh(385, Fraction(1, 3), 139, 20, unit="pixels")
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [self._rect_h2_row(y=self._cell(Fraction(1, 3)))],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert (
            str(refused.value)
            == str(constructed.value)
            == (
                "A shape's y is 1/3, which pixels as int cannot hold exactly: geometry "
                "is never rounded."
            )
        )

    def test_a_cell_holding_two_numbers_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [
                    self._rect_h2_row(
                        y={"value": 46.0, "numerator": 47, "denominator": 1}
                    )
                ],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert str(refused.value) == (
            "Box 'a': its bounding_box.y is the cell {'value': 46.0, 'numerator': 47, "
            "'denominator': 1}, whose sides hold two numbers (46.0 and 47/1); a cell "
            "holds one number on both sides."
        )

    def test_a_corner_cell_holding_two_numbers_is_named_by_its_path(self) -> None:
        row = Box.from_shape(self._measure(), origin=1221, id="m1").to_event_row()
        cells = QuadrilateralField.from_shapes(
            [row["quadrilateral"]], unit="pixels", number_type="int"
        )
        quadrilateral = cells.to_pyarrow().to_pylist()[0]
        quadrilateral["top_left"]["x"] = {
            "value": 1221.0,
            "numerator": 1222,
            "denominator": 1,
        }
        with pytest.raises(ValueError) as refused:
            EventData.from_dicts(
                [{**row, "quadrilateral": quadrilateral}],
                TimeUnit.pixels,
                NumberType.int,
            )
        assert str(refused.value) == (
            "Box 'm1': its quadrilateral.top_left.x is the cell {'value': 1221.0, "
            "'numerator': 1222, 'denominator': 1}, whose sides hold two numbers "
            "(1221.0 and 1222/1); a cell holds one number on both sides."
        )

    def test_a_cell_in_another_spelling_is_its_shape(self) -> None:
        thirds = BoundingBox.from_xywh(
            Fraction(1, 3),
            0,
            Fraction(2, 3),
            1,
            unit="user_units",
            number_type="fraction",
        )
        scalar = EventData.from_dicts(
            Box.from_shape(thirds, id="f"), TimeUnit.user_units, NumberType.fraction
        )
        row = scalar.table.to_pylist()[0]
        row["bounding_box"]["x"] = {
            "value": 1 / 3,
            "numerator": 2,
            "denominator": 6,
        }

        given = EventData.from_dicts([row], TimeUnit.user_units, NumberType.fraction)

        assert given.table.to_pylist() == scalar.table.to_pylist()
        assert given.table.to_pylist()[0]["bounding_box"]["x"] == {
            "value": 1 / 3,
            "numerator": 1,
            "denominator": 3,
        }

    def test_shape_leaves_are_reduced_at_the_door(self) -> None:
        measure = self._measure("user_units", "fraction")
        data = EventData.from_dicts(
            Box.from_shape(measure, origin=1221, id="m1"),
            TimeUnit.user_units,
            NumberType.fraction,
        )
        table = data.table
        index = table.schema.get_field_index("quadrilateral")
        rows = table.column(index).to_pylist()
        rows[0]["top_left"]["x"] = {
            "value": 1221.0,
            "numerator": 2442,
            "denominator": 2,
        }
        stated = table.set_column(
            index,
            table.schema.field(index),
            pa.array(rows, type=table.schema.field(index).type),
        )

        held = EventData(stated, TimeUnit.user_units, NumberType.fraction)

        corner = held.table.column("quadrilateral").to_pylist()[0]["top_left"]["x"]
        assert corner == {"value": 1221.0, "numerator": 1221, "denominator": 1}
        assert held.table.schema.field("quadrilateral").metadata == (
            table.schema.field("quadrilateral").metadata
        )


class TestEventTypeFields:
    """``get_field(<event type>)``: the events of that type, one per table row."""

    THIRDS = BoundingBox.from_xywh(
        Fraction(1, 3),
        0,
        Fraction(2, 3),
        1,
        unit="user_units",
        number_type="fraction",
    )

    def _mixed(self) -> tuple[EventData, Box, Projection]:
        box = Box.from_shape(self.THIRDS, id="thirds")
        projection = Projection(
            id="p",
            start=Coordinate(Fraction(1, 2), "user_units", number_type="fraction"),
        )
        data = EventData.from_dicts(
            [box, {"event_type": "Note", "start": 0, "duration": 1}, projection],
            TimeUnit.user_units,
            NumberType.fraction,
        )
        return data, box, projection

    def test_a_field_per_event_type_row_aligned(self) -> None:
        data, box, projection = self._mixed()
        boxes = data.get_field(Box)
        assert type(boxes) is BoxField
        assert boxes.name == "box"
        assert (boxes.unit, boxes.number_type) == (
            TimeUnit.user_units,
            NumberType.fraction,
        )
        assert list(boxes) == [box, None, None]
        assert boxes[0].start.value == Fraction(1, 3)
        assert type(boxes[0].start.value) is Fraction
        assert boxes[0].duration.value == Fraction(2, 3)
        projections = data.get_field(Projection)
        assert type(projections) is ProjectionField
        assert list(projections) == [None, None, projection]
        assert projections[2].start.value == Fraction(1, 2)
        assert projections[2].start.unit is TimeUnit.user_units

    def test_a_float_table_reads_floats(self) -> None:
        shape = BoundingBox.from_xywh(1.5, 0, 2.5, 1, unit="css_pixels")
        box = Box.from_shape(shape, id="b")
        data = EventData.from_dicts([box], TimeUnit.css_pixels, NumberType.float)
        (read,) = data.get_field(Box)
        assert read == box
        assert read.start.value == 1.5
        assert type(read.start.value) is float
        assert type(read.bounding_box.width.value) is float

    def test_every_spelling_of_the_call_gives_one_field(self) -> None:
        data, _, _ = self._mixed()
        boxes = data.get_field(Box)
        assert data.get_field(BoxField) is boxes
        assert data.get_field(Box, name="box") is boxes
        assert data.has_field(BoxField)

    def test_no_row_of_the_type_no_field(self) -> None:
        data = EventData.from_dicts(
            [{"event_type": "Note", "start": 0, "duration": 1}],
            TimeUnit.user_units,
            NumberType.fraction,
        )
        assert not data.has_field(BoxField)
        with pytest.raises(KeyError) as refused:
            data.get_field(Box)
        assert refused.value.args[0] == (
            "No field matching 'BoxField' found in table. Available columns: "
            "['id', 'name', 'temporal_type', 'event_type', 'start', 'end', "
            "'duration']"
        )

    @staticmethod
    def _one_note(**columns: Any) -> EventData:
        row = {"event_type": "Note", "start": Fraction(0), "duration": Fraction(1)}
        return EventData.from_dicts(
            [{**row, **columns}], TimeUnit.quarters, NumberType.fraction
        )

    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param(None, id="no pitch column"),
            pytest.param({"pitch": "E♭5"}, id="pitch name string"),
            pytest.param({"midi": 63}, id="bare midi on a base EventData"),
            pytest.param(
                {"step": "E", "octave": 5, "actual_accidental": "flat"},
                id="raw spelling columns",
            ),
        ],
    )
    def test_no_column_the_pitch_is_read_from_no_field(
        self, columns: dict[str, Any] | None
    ) -> None:
        data = self._mixed()[0] if columns is None else self._one_note(**columns)
        assert type(data) is EventData
        assert not data.has_field(NoteField)
        with pytest.raises(KeyError) as refused:
            data.get_field(Note)
        assert refused.value.args[0] == (
            "No field matching 'NoteField' found in table. "
            f"Available columns: {data.table.column_names}"
        )

    def test_a_pitch_number_of_its_own_name_is_read(self) -> None:
        (note,) = self._one_note(pitch=63).get_field(Note)
        assert note == Note(
            start=Coordinate(Fraction(0), "quarters"),
            end=Coordinate(Fraction(1), "quarters"),
            duration=Duration(Fraction(1), "quarters"),
            pitch=EnharmonicPitch(midi_number=63),
        )


#: The text of the refusal of a field the table cannot give.
NO_SCORE_NOTE = "No field matching 'ScoreNoteField' found in table."


class TestANoteField:
    """A Note's pitch from its own column; a ScoreNote on the rows of a Note."""

    @staticmethod
    def _seconds(value: float) -> Coordinate:
        return Coordinate(value, "seconds", number_type="float")

    def test_the_pitch_from_its_own_column(self) -> None:
        data = EventData.from_dicts(
            [
                {
                    "event_type": "Note",
                    "start": 0.5,
                    "duration": 0.25,
                    "pitch": {"midi_number": 60},
                },
                {"event_type": "Rest", "start": 0.75, "duration": 0.25},
                {
                    "event_type": "Note",
                    "start": 1.0,
                    "duration": 0.5,
                    "pitch": {"midi_number": 62},
                },
            ],
            TimeUnit.seconds,
            NumberType.float,
        )
        columns = data.table.schema
        notes = data.get_field(Note)
        assert type(notes) is NoteField
        assert (notes.unit, notes.number_type) == (TimeUnit.seconds, NumberType.float)

        def note(start: float, duration: float, midi: int) -> Note:
            return Note(
                start=self._seconds(start),
                end=self._seconds(start + duration),
                duration=Duration(duration, "seconds", number_type="float"),
                pitch=EnharmonicPitch(midi_number=midi),
            )

        assert list(notes) == [note(0.5, 0.25, 60), None, note(1.0, 0.5, 62)]
        assert notes[0].start.unit is TimeUnit.seconds
        assert type(notes[0].start.value) is float
        assert data.table.schema == columns
        assert notes.get_raw().field.type.names == [*NoteField.pa_schema.names, "pitch"]

    @staticmethod
    def _score(**columns: Any) -> EventData:
        row = {
            "event_type": "Note",
            "start": Fraction(0),
            "duration": Fraction(1, 2),
            "pitch": {"midi_number": 60},
        }
        return EventData.from_dicts(
            [{**row, **columns}], TimeUnit.quarters, NumberType.fraction
        )

    def test_a_score_note_on_the_rows_of_a_note(self) -> None:
        data = self._score(staff=1, voice=2)
        members = {
            "start": Coordinate(Fraction(0), "quarters"),
            "end": Coordinate(Fraction(1, 2), "quarters"),
            "duration": Duration(Fraction(1, 2), "quarters"),
            "pitch": EnharmonicPitch(midi_number=60),
        }
        score_notes = data.get_field(ScoreNote)
        assert type(score_notes) is ScoreNoteField
        assert list(score_notes) == [ScoreNote(**members, voice=2, staff=1)]
        (plain,) = data.get_field(Note)
        assert type(plain) is Note
        assert plain == Note(**members)

    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param({"staff": 1}, id="staff alone"),
            pytest.param({"staff": "P1_staff1", "voice": "P1_voice1"}, id="text ids"),
            pytest.param({"staff": None, "voice": None}, id="null"),
        ],
    )
    def test_what_is_no_score_note(self, columns: dict[str, Any]) -> None:
        data = self._score(**columns)
        assert not data.has_field(ScoreNoteField)
        with pytest.raises(KeyError) as refused:
            data.get_field(ScoreNote)
        assert refused.value.args[0].startswith(NO_SCORE_NOTE)
        (note,) = data.get_field(Note)
        assert note.pitch == EnharmonicPitch(midi_number=60)

    def test_a_base_row_is_no_box(self) -> None:
        event = Event(
            id="e", start=Coordinate(0, "pixels"), duration=Duration(1, "pixels")
        )
        data = EventData.from_dicts([event], TimeUnit.pixels, NumberType.int)
        assert data.table.column("event_type").to_pylist() == ["Event"]
        with pytest.raises(KeyError) as refused:
            data.get_field(Box)
        assert refused.value.args[0].startswith("No field matching 'BoxField'")


class TestOneAxisPerTable:
    """One unit, one number type and one spelling of each value per table."""

    @staticmethod
    def _quarters(*rows: dict[str, Any]) -> EventData:
        return EventData.from_dicts(list(rows), TimeUnit.quarters)

    def test_a_further_coordinate_column_keeps_its_number_type(self, tmp_path):
        data = EventData.from_arrays(
            {
                "id": ["e1"],
                "event_type": ["Beat"],
                "start": build_number_struct_array(
                    [Fraction(1, 3)], number_type=NumberType.fraction
                ),
                "onset": build_number_struct_array(
                    [Fraction(1, 3)], number_type=NumberType.fraction
                ),
            },
            TimeUnit.quarters,
        )
        path = tmp_path / "events.parquet"
        data.to_parquet(path)
        back = EventData.from_parquet(path)

        assert field_metadata(back.schema.field("onset")) == {
            "unit": "quarters",
            "number_type": "fraction",
            "version": 1,
        }
        assert field_metadata(back.schema.field("onset")) == field_metadata(
            back.schema.field("start")
        )
        (onset,) = [f for f in back.get_fields(CoordinateField) if f.name == "onset"]
        assert onset.number_type is NumberType.fraction
        assert onset[0].value == Fraction(1, 3)
        assert type(onset[0].value) is Fraction

    def test_a_file_written_before_reads_as_before(self, tmp_path):
        data = self._quarters({"id": "e1", "event_type": "Beat", "instant": 0})
        table = data.table.append_column(
            pa.field("onset", RATIONAL_STRUCT_TYPE, metadata={"unit": "quarters"}),
            build_number_struct_array(
                [Fraction(1, 3)], number_type=NumberType.fraction
            ),
        )
        path = tmp_path / "events.parquet"
        pq.write_table(table, path)
        back = EventData.from_parquet(path)

        (onset,) = [f for f in back.get_fields(CoordinateField) if f.name == "onset"]
        assert onset.number_type is NumberType.float
        assert onset[0].value == 0.3333333333333333

    def test_rows_of_the_same_axis_join(self):
        first = self._quarters({"id": "e1", "event_type": "Beat", "instant": 1})
        second = self._quarters({"id": "e2", "event_type": "Beat", "instant": 2})

        joined = first.concat(second)

        assert len(joined) == 2
        assert len(first) == 1
        assert "timeline_id" not in joined.table.column_names

    @staticmethod
    def _spelled(data: EventData, pairs: list[tuple[int, int]]) -> pa.Table:
        """*data*'s table with its ``start`` cells spelled as *pairs*."""
        table = data.table
        index = table.schema.get_field_index("start")
        cells = table.column(index).combine_chunks()
        spelled = pa.StructArray.from_arrays(
            [
                cells.field("value"),
                pa.array([n for n, _ in pairs], pa.int64()),
                pa.array([d for _, d in pairs], pa.int64()),
            ],
            fields=list(cells.type),
        )
        return table.set_column(index, table.schema.field(index), spelled)

    @staticmethod
    def _start_pairs(table: pa.Table) -> list[tuple[int, int]]:
        return [
            (cell["numerator"], cell["denominator"])
            for cell in table.column("start").to_pylist()
        ]

    def test_cells_are_reduced_on_entry(self, tmp_path):
        half = Fraction(1, 2)
        data = self._quarters(
            *({"id": f"e{i}", "event_type": "Beat", "instant": half} for i in range(3))
        )
        table = self._spelled(data, [(2, 4), (4, 8), (1, 2)])

        entered = EventData(table, TimeUnit.quarters)
        assert self._start_pairs(entered.table) == [(1, 2)] * 3
        assert entered.column_values("start") == [half] * 3

        path = tmp_path / "spelled.parquet"
        pq.write_table(table, path)
        read = EventData.from_parquet(path)
        assert self._start_pairs(read.table) == [(1, 2)] * 3

        written = tmp_path / "written.parquet"
        read.to_parquet(written)
        assert self._start_pairs(pq.read_table(written)) == [(1, 2)] * 3

        joined = self._quarters({"id": "x", "event_type": "Beat", "instant": 1})
        assert self._start_pairs(joined.concat(entered).table)[1:] == [(1, 2)] * 3

    def test_a_projection_read_unreduced_is_the_same_projection(self, tmp_path):
        half = Coordinate(Fraction(1, 2), TimeUnit.quarters)
        timeline = ContinuousLogicalTimeline(length=2)
        (held,) = timeline.add_projections([Projection(start=half)])
        path = tmp_path / "events.parquet"
        pq.write_table(self._spelled(timeline.events, [(2, 4)]), path)

        read = EventData.from_parquet(path).create_timeline()

        assert read.add_projections([Projection(start=half)]) == [held]
        assert read.n_events == 1


def _row_cells(data: EventData, *names: str) -> list[tuple[Any, ...]]:
    """Each row's id and its cells under *names*, as ``(value, n, d)`` or None."""
    return [
        (
            row["id"],
            *(
                (
                    None
                    if row[name] is None
                    else (
                        row[name]["value"],
                        row[name]["numerator"],
                        row[name]["denominator"],
                    )
                )
                for name in names
            ),
        )
        for row in data.table.to_pylist()
    ]


def _declared(data: EventData) -> dict[str, str]:
    """The number type each coordinate field of *data* declares."""
    return {
        name: field_metadata(data.schema.field(name))["number_type"]
        for name in ("start", "end", "duration")
    }


def _ids_of_type(data: EventData, number_type: NumberType) -> list[str]:
    """The ids of the rows of *data* read in *number_type*."""
    return data.filter(include_number_types={number_type}).column_values("id")


class TestJoiningKeepsNumberTypes:
    """Rows of another number type join as they are; the cast is on request."""

    @staticmethod
    def _thirds() -> EventData:
        return EventData.from_dicts(
            [
                {
                    "id": "a",
                    "event_type": "Note",
                    "start": Fraction(1, 3),
                    "end": Fraction(2, 3),
                }
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )

    @staticmethod
    def _decimals() -> EventData:
        return EventData.from_dicts(
            [
                {"id": "b", "event_type": "Note", "start": 0.1, "duration": 0.2},
                {"id": "c", "event_type": "Beat", "instant": 0.5},
            ],
            TimeUnit.quarters,
            NumberType.float,
        )

    @staticmethod
    def _numbers(number_type: NumberType, *rows: tuple[str, Any, Any]) -> EventData:
        return EventData.from_dicts(
            [
                {"id": name, "event_type": "Note", "start": start, "end": end}
                for name, start, end in rows
            ],
            TimeUnit.number,
            number_type,
        )

    def test_floats_joining_a_fraction_table_stay_floats(self):
        thirds, decimals = self._thirds(), self._decimals()

        joined = thirds.concat(decimals)

        assert joined.number_type is NumberType.fraction
        assert _row_cells(joined, "start", "end", "duration") == [
            *_row_cells(thirds, "start", "end", "duration"),
            *_row_cells(decimals, "start", "end", "duration"),
        ]
        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters fraction>",
            "<quarters float>",
            "<quarters float>",
        ]
        assert _ids_of_type(joined, NumberType.fraction) == ["a"]
        assert _ids_of_type(joined, NumberType.float) == ["b", "c"]
        assert len(thirds) == 1 and len(decimals) == 2

    def test_the_cast_to_fraction_takes_floats_as_their_exact_ratios(self):
        cast = self._thirds().concat(self._decimals()).to_number_type("fraction")

        assert cast.number_type is NumberType.fraction
        assert _declared(cast) == dict.fromkeys(
            ("start", "end", "duration"), "fraction"
        )
        assert _row_cells(cast, "start", "end", "duration") == [
            (
                "a",
                (0.3333333333333333, 1, 3),
                (0.6666666666666666, 2, 3),
                (0.3333333333333333, 1, 3),
            ),
            (
                "b",
                (0.1, 3602879701896397, 36028797018963968),
                (0.30000000000000004, 10808639105689191, 36028797018963968),
                (0.2, 3602879701896397, 18014398509481984),
            ),
            ("c", (0.5, 1, 2), None, None),
        ]
        assert _ids_of_type(cast, NumberType.fraction) == ["a", "b", "c"]
        assert cast.column_values("end")[1] == Fraction(0.1) + Fraction(0.2)

    def test_the_cast_to_float_takes_ratios_as_the_nearest_floats(self):
        joined = self._decimals().concat(self._thirds())
        assert _ids_of_type(joined, NumberType.fraction) == ["a"]

        cast = joined.to_number_type(NumberType.float)

        assert cast.number_type is NumberType.float
        assert _declared(cast) == dict.fromkeys(("start", "end", "duration"), "float")
        assert _row_cells(cast, "start", "end", "duration")[2] == (
            "a",
            (0.3333333333333333, 6004799503160661, 18014398509481984),
            (0.6666666666666666, 6004799503160661, 9007199254740992),
            (0.3333333333333333, 6004799503160661, 18014398509481984),
        )
        assert _row_cells(cast, "start")[:2] == [
            ("b", (0.1, 3602879701896397, 36028797018963968)),
            ("c", (0.5, 1, 2)),
        ]

    def test_the_cast_to_int_takes_the_nearest_integers_ties_to_even(self):
        receiving = self._numbers(NumberType.int, ("x", 10, 11))
        ratios = self._numbers(
            NumberType.fraction,
            ("y", Fraction(5, 2), Fraction(7, 2)),
            ("z", Fraction(3, 2), Fraction(9, 2)),
        )
        decimals = self._numbers(NumberType.float, ("w", 2.5, 3.5))

        joined = receiving.concat(ratios).concat(decimals)

        assert _row_cells(joined, "start") == [
            ("x", (10.0, 10, 1)),
            ("y", (2.5, 5, 2)),
            ("z", (1.5, 3, 2)),
            ("w", (2.5, 5, 2)),
        ]
        assert [
            _ids_of_type(joined, number_type)
            for number_type in (NumberType.int, NumberType.fraction, NumberType.float)
        ] == [["x"], ["y", "z"], ["w"]]
        cast = joined.to_number_type("int")
        assert cast.number_type is NumberType.int
        assert _declared(cast) == dict.fromkeys(("start", "end", "duration"), "int")
        assert _row_cells(cast, "start", "end", "duration") == [
            ("x", (10.0, 10, 1), (11.0, 11, 1), (1.0, 1, 1)),
            ("y", (2.0, 2, 1), (3.0, 3, 1), (1.0, 1, 1)),
            ("z", (2.0, 2, 1), (5.0, 5, 1), (3.0, 3, 1)),
            ("w", (2.0, 2, 1), (3.0, 3, 1), (1.0, 1, 1)),
        ]

    def test_the_cast_to_fraction_takes_integers_as_they_stand(self):
        receiving = self._numbers(
            NumberType.fraction, ("y", Fraction(5, 2), Fraction(7, 2))
        )
        joined = receiving.concat(self._numbers(NumberType.int, ("x", 10, 11)))

        assert _ids_of_type(joined, NumberType.int) == ["x"]
        cast = joined.to_number_type("fraction")
        assert _row_cells(cast, "start", "end", "duration") == [
            ("y", (2.5, 5, 2), (3.5, 7, 2), (1.0, 1, 1)),
            ("x", (10.0, 10, 1), (11.0, 11, 1), (1.0, 1, 1)),
        ]
        assert _declared(cast) == dict.fromkeys(
            ("start", "end", "duration"), "fraction"
        )

    def test_another_unit_joins_on_its_own_axis(self):
        receiving = self._thirds()
        seconds = EventData.from_dicts(
            [{"id": "s", "event_type": "Note", "start": 1.0, "end": 2.0}],
            TimeUnit.seconds,
        )

        joined = receiving.concat(seconds)

        assert joined.unit is TimeUnit.quarters
        assert joined.table.column("timeline_id").to_pylist() == [
            "<quarters fraction>",
            "<seconds float>",
        ]
        assert _row_cells(joined, "start")[1] == ("s", (1.0, 1, 1))
        assert len(receiving) == 1

    def test_a_partial_cast_of_a_table_spanning_timelines_is_refused(self):
        joined = self._thirds().concat(self._decimals())

        with pytest.raises(ValueError) as refusal:
            joined.to_number_type("fraction", columns=["start"])

        assert str(refusal.value) == (
            "a table spanning timelines reads every coordinate column in its rows' "
            "number types, so it casts all of them; ['end', 'duration'] would be "
            "left out"
        )

    def test_a_partial_cast_of_a_table_standing_alone_casts_the_columns_named(self):
        cast = self._decimals().to_number_type("fraction", columns=["start"])

        assert cast.number_type is NumberType.float
        assert _declared(cast) == {
            "start": "fraction",
            "end": "float",
            "duration": "float",
        }
        assert cast.column_values("start")[0] == Fraction(0.1)

    def test_a_number_type_filter_on_a_table_standing_alone_is_all_or_nothing(self):
        decimals = self._decimals()

        assert _ids_of_type(decimals, NumberType.float) == ["b", "c"]
        assert _ids_of_type(decimals, NumberType.fraction) == []


class TestFilterFields:
    """A further field is a column or a declared field; any other is refused."""

    @staticmethod
    def _notes() -> EventData:
        return EventData.from_dicts(
            [
                {"id": "n1", "event_type": "Note", "start": 0, "pitch": 60},
                {"id": "n2", "event_type": "Note", "start": 1, "pitch": 62},
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )

    def test_a_column_is_a_field(self):
        assert self._notes().filter(pitch=62).column_values("id") == ["n2"]

    def test_a_declared_field_the_table_lacks_keeps_no_row(self):
        from timetoalign.loader.score.stores.annotations import AnnotationEventData

        held = AnnotationEventData.from_dicts(
            [{"id": "h1", "event_type": "Harmony", "start": 0, "text": "I"}],
            TimeUnit.quarters,
            NumberType.fraction,
        )
        data = AnnotationEventData(
            held.table.drop_columns(["subtype"]), TimeUnit.quarters, NumberType.fraction
        )

        assert len(data.filter(subtype="Harmony")) == 0

    def test_an_unknown_field_is_refused(self):
        with pytest.raises(KeyError) as refusal:
            self._notes().filter(colour="red")

        assert str(refusal.value) == (
            "\"No field 'colour' to filter events by. Available fields: ['id', "
            "'name', 'temporal_type', 'event_type', 'start', 'end', 'duration', "
            "'pitch']\""
        )


class TestFilterBounds:
    """A float bound compares as a key, an exact one exactly; NaN is refused."""

    @staticmethod
    def _tenths() -> EventData:
        return EventData.from_dicts(
            [
                {"id": "a", "event_type": "Beat", "instant": Fraction(1, 10)},
                {"id": "b", "event_type": "Beat", "instant": Fraction(1, 3)},
            ],
            TimeUnit.quarters,
            NumberType.fraction,
        )

    @staticmethod
    def _ids(data: EventData) -> list[str]:
        return data.column_values("id")

    def test_a_float_bound_on_a_fraction_table_is_a_key(self):
        data = self._tenths()

        assert self._ids(data.filter(min_coord=0.1)) == ["a", "b"]
        assert self._ids(data.filter(max_coord=0.1)) == []
        assert self._ids(data.filter(min_coord=1 / 3)) == ["b"]

    def test_an_exact_bound_on_a_fraction_table_is_exact(self):
        data = self._tenths()

        assert self._ids(data.filter(min_coord=Fraction(0.1))) == ["b"]
        assert self._ids(data.filter(max_coord=Fraction(1, 3))) == ["a"]
        assert self._ids(data.filter(min_coord=Fraction(1, 10))) == ["a", "b"]

    def test_a_float_bound_on_an_int_table_is_not_rounded(self):
        data = EventData.from_dicts(
            [
                {"id": "x", "event_type": "Beat", "instant": 10},
                {"id": "y", "event_type": "Beat", "instant": 11},
            ],
            TimeUnit.number,
            NumberType.int,
        )

        assert self._ids(data.filter(min_coord=10.5)) == ["y"]
        assert self._ids(data.filter(max_coord=10.5)) == ["x"]
        assert self._ids(data.filter(min_coord=Fraction(21, 2))) == ["y"]

    @pytest.mark.parametrize(
        "number_type", [NumberType.float, NumberType.fraction, NumberType.int]
    )
    def test_infinite_bounds_are_ordinary_doubles(self, number_type):
        data = EventData.from_dicts(
            [
                {"id": "x", "event_type": "Beat", "instant": 1},
                {"id": "y", "event_type": "Beat", "instant": 2},
            ],
            TimeUnit.number,
            number_type,
        )

        assert self._ids(data.filter(max_coord=math.inf)) == ["x", "y"]
        assert self._ids(data.filter(min_coord=-math.inf)) == ["x", "y"]
        assert self._ids(data.filter(min_coord=math.inf)) == []
        assert self._ids(data.filter(max_coord=-math.inf)) == []

    @pytest.mark.parametrize("bound", ["min_coord", "max_coord"])
    def test_a_nan_bound_is_refused(self, bound):
        with pytest.raises(ValueError) as refusal:
            self._tenths().filter(**{bound: math.nan})

        assert str(refusal.value) == "NaN names no position: it compares with no cell"


class TestReducedAtTheDoorsOnly:
    """Tables derived from held tables are not reduced a second time."""

    @staticmethod
    def _counted(monkeypatch: pytest.MonkeyPatch) -> list[pa.Table]:
        """Record every table the reducing function is given."""
        reduced: list[pa.Table] = []
        reduce = storage.events._in_lowest_terms

        def counting(table: pa.Table) -> pa.Table:
            reduced.append(table)
            return reduce(table)

        monkeypatch.setattr(storage.events, "_in_lowest_terms", counting)
        return reduced

    @staticmethod
    def _notes(*starts: Fraction) -> list[dict[str, Any]]:
        return [
            {
                "id": f"n{k}",
                "event_type": "Note",
                "start": start,
                "duration": Fraction(1, 3),
            }
            for k, start in enumerate(starts)
        ]

    @staticmethod
    def _starts(data: EventData) -> list[tuple[str, tuple[int, int]]]:
        return [
            (row["id"], (row["start"]["numerator"], row["start"]["denominator"]))
            for row in data
        ]

    def test_the_doors_reduce(self, monkeypatch, tmp_path):
        data = EventData.from_dicts(self._notes(Fraction(1, 2)), TimeUnit.quarters)
        path = tmp_path / "notes.parquet"
        data.to_parquet(path)
        reduced = self._counted(monkeypatch)

        EventData(data.table, TimeUnit.quarters)
        EventData.from_parquet(path)

        assert len(reduced) == 2

    def test_derived_tables_are_not_reduced_again(self, monkeypatch):
        data = EventData.from_dicts(
            self._notes(Fraction(1, 2), Fraction(2, 3)), TimeUnit.quarters
        )
        other = EventData.from_dicts(
            [{"id": "b", "event_type": "Beat", "instant": Fraction(5, 4)}],
            TimeUnit.quarters,
        )
        reduced = self._counted(monkeypatch)

        filtered = data.filter(min_coord=Fraction(2, 3))
        joined = data.concat(other)
        timeline = data.create_timeline()

        assert reduced == []
        assert self._starts(filtered) == [("n1", (2, 3))]
        assert self._starts(joined) == [("n0", (1, 2)), ("n1", (2, 3)), ("b", (5, 4))]
        assert self._starts(timeline.events) == [("n0", (1, 2)), ("n1", (2, 3))]

    def test_a_filtered_get_events_reduces_nothing(self, monkeypatch):
        flat = ContinuousLogicalTimeline(length=4, uid="flat")
        flat.add_events(self._notes(Fraction(1, 2), Fraction(2, 3), Fraction(3, 2)))
        parent = ContinuousLogicalTimeline(length=4, uid="parent")
        parent.add_events(self._notes(Fraction(1, 2)))
        child = ContinuousLogicalTimeline(length=2, uid="child")
        child.add_events(
            [{"id": "c", "event_type": "Note", "start": Fraction(1, 6), "end": 1}]
        )
        parent.add_child(child, offset=Fraction(1, 3))
        reduced = self._counted(monkeypatch)
        query = {
            "temporal_type": "interval",
            "event_type": "Note",
            "min_coord": Fraction(1, 2),
            "max_coord": Fraction(3, 2),
        }

        flat_rows = flat.get_events(**query)
        parent_rows = parent.get_events(**query)

        assert reduced == []
        assert self._starts(flat_rows) == [("n0", (1, 2)), ("n1", (2, 3))]
        assert self._starts(parent_rows) == [("n0", (1, 2)), ("c", (1, 2))]


# region A measure's volta, carried as stated


def _measure_rows(*voltas: Any) -> list[dict[str, Any]]:
    return [
        {"event_type": "Measure", "start": index, "end": index + 1, "volta": volta}
        for index, volta in enumerate(voltas)
    ]


def test_a_volta_number_and_a_volta_list_are_carried_in_two_columns() -> None:
    data = EventData.from_dicts(
        _measure_rows(2, [1, 3], [1], None), TimeUnit.quarters, NumberType.fraction
    )
    assert data.table.schema.field("volta").type == pa.int64()
    assert data.table.schema.field("volta_list").type == pa.list_(pa.int64())
    assert data.table.column("volta").to_pylist() == [2, None, None, None]
    assert data.table.column("volta_list").to_pylist() == [None, [1, 3], [1], None]


def test_the_list_column_has_its_type_whatever_its_rows_hold() -> None:
    without = EventData.from_dicts(
        [{**_measure_rows(None)[0], "volta_list": None}],
        TimeUnit.quarters,
        NumberType.fraction,
    )
    assert without.table.schema.field("volta_list").type == pa.list_(pa.int64())
    joined = without.concat(
        EventData.from_dicts(
            _measure_rows([1, 3]), TimeUnit.quarters, NumberType.fraction
        )
    )
    assert joined.table.column("volta").to_pylist() == [None, None]
    assert joined.table.column("volta_list").to_pylist() == [None, [1, 3]]


def test_rows_stating_numbers_only_carry_no_list_column() -> None:
    data = EventData.from_dicts(
        _measure_rows(None, 2), TimeUnit.quarters, NumberType.fraction
    )
    assert "volta_list" not in data.table.column_names
    assert data.table.schema.field("volta").type == pa.int64()
    assert data.table.column("volta").to_pylist() == [None, 2]


# endregion


class TestFromDataFrameLabels:
    """``from_dataframe`` names the column labels that are not strings."""

    def test_non_string_labels_are_named(self) -> None:
        frame = pd.DataFrame({"start": [0.0], 1: [2], 2.5: [3]})
        with pytest.raises(TypeError) as caught:
            EventData.from_dataframe(frame, TimeUnit.seconds)
        assert str(caught.value) == "Column labels must be strings: [1, 2.5]"

    def test_string_labels_build(self) -> None:
        data = EventData.from_dataframe(
            pd.DataFrame({"start": [0.0], "x": [2]}), TimeUnit.seconds
        )
        assert data.count == 1
        assert data.table.column("x").to_pylist() == [2]

    def test_empty_frame_with_an_integer_label(self) -> None:
        data = EventData.from_dataframe(pd.DataFrame({1: []}), TimeUnit.seconds)
        assert data.count == 0
