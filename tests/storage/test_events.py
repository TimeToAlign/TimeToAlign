"""Exact-value tests for ``EventData.column_values()``.

See ``README.md`` for the documented contract.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign import storage
from timetoalign.core import (
    Coordinate,
    Duration,
    EventReference,
    IntervalPolicy,
    NumberType,
    Projection,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
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

    def test_mint_event_ids(self) -> None:
        assert storage.mint_event_ids is mint_event_ids
        assert mint_event_ids(
            ["Note", "Projection", "Note", None],
            {"note_1", "projection_2", "note:000005"},
        ).to_pylist() == ["note_2", "projection_3", "note_3", "event_1"]

    def test_from_dicts_mints_above_the_given_ids(self) -> None:
        data = EventData.from_dicts(
            [
                {"event_type": "Note", "id": "note_4", "start": 0},
                {"event_type": "Note", "start": 1},
            ],
            TimeUnit.samples,
        )
        assert data.column_values("id") == ["note_4", "note_5"]

    def test_from_arrays_mints_from_the_event_types(self) -> None:
        data = EventData.from_arrays(
            {
                "event_type": ["Projection", "Projection", "Note"],
                "start": [0, 1, 2],
            },
            TimeUnit.samples,
        )
        assert data.column_values("id") == [
            "projection_1",
            "projection_2",
            "note_1",
        ]

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
        data.extend(
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
                ),
                {"event_type": "Note", "start": 7},
            ],
            TimeUnit.samples,
        )
        rows = data.table.select(
            ["id", "event_type", "start", "duration", "projection_of"]
        ).to_pylist()
        assert rows == [
            {
                "id": "projection_1",
                "event_type": "Projection",
                "start": {"value": 2.0, "numerator": 2, "denominator": 1},
                "duration": {"value": 3.0, "numerator": 3, "denominator": 1},
                "projection_of": {"timeline_id": "clt_6", "event_id": "x1"},
            },
            {
                "id": "note_1",
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
