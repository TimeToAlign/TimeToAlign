"""Exact-value tests for ``EventData.column_values()``.

See ``README.md`` for the documented contract.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa

from timetoalign.core import NumberType, TimeUnit
from timetoalign.storage.events import EventData
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
