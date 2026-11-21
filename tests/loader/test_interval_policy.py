"""Tests for IntervalPolicy-driven interval normalization.

Tests both the vectorized path (``EventData.from_arrays``) and the
row-based path (``EventData.from_dicts``), exercising all four
``IntervalPolicy`` modes: ``warn``, ``prefer_end``, ``prefer_duration``,
and ``strict``.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any

import pyarrow as pa
import pytest

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.core.enums import IntervalPolicy
from timetoalign.core.time import (
    coordinate_to_struct,
)
from timetoalign.storage import EventData
from timetoalign.testdata import ensure_data

ensure_data("tabular")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    *,
    start: float | Fraction,
    end: float | Fraction | None = None,
    duration: float | Fraction | None = None,
    event_id: str = "e1",
    event_type: str = "Note",
) -> dict[str, Any]:
    """Build a single event dict for ``from_dicts``."""
    row: dict[str, Any] = {
        "id": event_id,
        "event_type": event_type,
        "start": start,
    }
    if end is not None:
        row["end"] = end
    if duration is not None:
        row["duration"] = duration
    return row


def _value_of(struct: dict | None) -> float | None:
    """Extract the float ``value`` from a coordinate struct or None."""
    if struct is None:
        return None
    if isinstance(struct, dict):
        return struct.get("value")
    return None


# ---------------------------------------------------------------------------
# Row-based path: EventData._normalize_intervals_row
# ---------------------------------------------------------------------------


class TestNormalizeIntervalsRow:
    """Unit tests for ``EventData._normalize_intervals_row``."""

    def test_end_only_computes_duration(self):
        """When only end is given, duration = end - start."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
        }
        EventData._normalize_intervals_row(processed, number_type=NumberType.float)
        dur = _value_of(processed["duration"])
        assert dur == 2.0
        assert processed["temporal_type"] == "interval"

    def test_duration_only_computes_end(self):
        """When only duration is given, end = start + duration."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "duration": coordinate_to_struct(2.0),
        }
        EventData._normalize_intervals_row(processed, number_type=NumberType.float)
        end = _value_of(processed["end"])
        assert end == 3.0
        assert processed["temporal_type"] == "interval"

    def test_no_end_no_duration_is_instant(self):
        """When neither end nor duration is given, event is instant."""
        processed = {"start": coordinate_to_struct(1.0)}
        EventData._normalize_intervals_row(processed, number_type=NumberType.float)
        assert processed["temporal_type"] == "instant"

    def test_consistent_both_present(self):
        """When end and duration agree, both are kept."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
            "duration": coordinate_to_struct(2.0),
        }
        EventData._normalize_intervals_row(
            processed, policy=IntervalPolicy.warn, number_type=NumberType.float
        )
        assert _value_of(processed["end"]) == 3.0
        assert _value_of(processed["duration"]) == 2.0

    # ---- Inconsistent rows ----

    def test_strict_raises_on_inconsistency(self):
        """strict policy raises ValueError when end-start != duration."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
            "duration": coordinate_to_struct(5.0),  # wrong!
        }
        with pytest.raises(ValueError, match="Interval inconsistency"):
            EventData._normalize_intervals_row(
                processed, policy=IntervalPolicy.strict, number_type=NumberType.float
            )

    def test_warn_logs_and_recomputes_duration(self, caplog):
        """warn policy logs a warning and recomputes duration from end."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
            "duration": coordinate_to_struct(5.0),
        }
        with caplog.at_level(logging.WARNING, logger="timetoalign.storage.events"):
            EventData._normalize_intervals_row(
                processed, policy=IntervalPolicy.warn, number_type=NumberType.float
            )
        assert _value_of(processed["duration"]) == 2.0
        assert _value_of(processed["end"]) == 3.0
        assert any("inconsistency" in r.message.lower() for r in caplog.records)

    def test_prefer_end_recomputes_duration(self):
        """prefer_end always recomputes duration from end."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
            "duration": coordinate_to_struct(5.0),
        }
        EventData._normalize_intervals_row(
            processed, policy=IntervalPolicy.prefer_end, number_type=NumberType.float
        )
        assert _value_of(processed["duration"]) == 2.0
        assert _value_of(processed["end"]) == 3.0

    def test_prefer_duration_recomputes_end(self):
        """prefer_duration always recomputes end from duration."""
        processed = {
            "start": coordinate_to_struct(1.0),
            "end": coordinate_to_struct(3.0),
            "duration": coordinate_to_struct(5.0),
        }
        EventData._normalize_intervals_row(
            processed,
            policy=IntervalPolicy.prefer_duration,
            number_type=NumberType.float,
        )
        assert _value_of(processed["end"]) == 6.0
        assert _value_of(processed["duration"]) == 5.0

    # ---- Fraction inputs ----

    def test_fraction_end_only(self):
        """Fraction coordinates: duration computed correctly."""
        processed = {
            "start": coordinate_to_struct(Fraction(1, 4)),
            "end": coordinate_to_struct(Fraction(3, 4)),
        }
        EventData._normalize_intervals_row(processed, number_type=NumberType.fraction)
        dur = _value_of(processed["duration"])
        assert dur == 0.5

    def test_fraction_duration_only(self):
        """Fraction coordinates: end computed correctly."""
        processed = {
            "start": coordinate_to_struct(Fraction(1, 4)),
            "duration": coordinate_to_struct(Fraction(1, 2)),
        }
        EventData._normalize_intervals_row(processed, number_type=NumberType.fraction)
        end = _value_of(processed["end"])
        assert end == 0.75

    # ---- Exact recomputation when both end and duration are present ----

    def test_both_present_exact_fractions_recomputes_duration_exactly(self):
        """warn: recomputed duration keeps the exact ratio when start/end do."""
        processed = {
            "start": coordinate_to_struct(Fraction(1, 3)),
            "end": coordinate_to_struct(Fraction(3, 1)),
            "duration": coordinate_to_struct(Fraction(1, 2)),  # inconsistent
        }
        EventData._normalize_intervals_row(
            processed, policy=IntervalPolicy.warn, number_type=NumberType.fraction
        )
        dur = processed["duration"]
        assert dur["numerator"] == 8
        assert dur["denominator"] == 3

    def test_both_present_mixed_exactness_still_completes_the_cell(self):
        """warn: recomputation stays exact and completes both sides."""
        processed = {
            "start": coordinate_to_struct(Fraction(1, 3)),
            "end": coordinate_to_struct(3.0),  # no exact ratio
            "duration": coordinate_to_struct(Fraction(1, 2)),  # inconsistent
        }
        EventData._normalize_intervals_row(
            processed, policy=IntervalPolicy.warn, number_type=NumberType.fraction
        )
        dur = processed["duration"]
        assert (dur["numerator"], dur["denominator"]) == (8, 3)
        assert dur["value"] == float(Fraction(8, 3))


# ---------------------------------------------------------------------------
# Vectorized path: EventData.from_arrays
# ---------------------------------------------------------------------------


def _coord_struct_type() -> pa.DataType:
    """Return the coordinate struct type used by EventData."""
    return pa.struct(
        [
            pa.field("value", pa.float64(), nullable=True),
            pa.field("numerator", pa.int64(), nullable=True),
            pa.field("denominator", pa.int64(), nullable=True),
        ]
    )


def _make_coord_array(
    values: list[float],
    numerators: list[int | None] | None = None,
    denominators: list[int | None] | None = None,
    null_mask: list[bool] | None = None,
) -> pa.StructArray:
    """Build a coordinate StructArray from explicit values."""
    n = len(values)
    if numerators is None:
        numerators = [None] * n
    if denominators is None:
        denominators = [None] * n

    val_arr = pa.array(values, type=pa.float64())
    # For numerator/denominator: treat None as null
    num_arr = pa.array(
        [0 if v is None else v for v in numerators],
        type=pa.int64(),
        mask=pa.array([v is None for v in numerators]),
    )
    den_arr = pa.array(
        [1 if v is None else v for v in denominators],
        type=pa.int64(),
        mask=pa.array([v is None for v in denominators]),
    )

    struct = pa.StructArray.from_arrays(
        [val_arr, num_arr, den_arr],
        names=["value", "numerator", "denominator"],
    )
    if null_mask:
        struct = pa.StructArray.from_arrays(
            [val_arr, num_arr, den_arr],
            names=["value", "numerator", "denominator"],
            mask=pa.array(null_mask),
        )
    return struct


class TestNormalizeIntervalsVectorized:
    """Unit tests for ``EventData._normalize_intervals_vectorized``."""

    def test_end_only_computes_duration(self):
        """Vectorized: duration computed from end - start."""
        processed = {
            "start": _make_coord_array([0.0, 1.0, 2.0]),
            "end": _make_coord_array([1.0, 2.0, 3.5]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed, number_type=NumberType.float
        )
        dur = result["duration"]
        dur_vals = dur.field("value").to_pylist()
        assert dur_vals == [1.0, 1.0, 1.5]

    def test_duration_only_computes_end(self):
        """Vectorized: end computed from start + duration."""
        processed = {
            "start": _make_coord_array([0.0, 1.0, 2.0]),
            "duration": _make_coord_array([1.0, 1.0, 1.5]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed, number_type=NumberType.float
        )
        end = result["end"]
        end_vals = end.field("value").to_pylist()
        assert end_vals == [1.0, 2.0, 3.5]

    def test_strict_raises_on_inconsistency(self):
        """Vectorized: strict policy raises ValueError."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
            "end": _make_coord_array([1.0, 3.0]),
            "duration": _make_coord_array([1.0, 5.0]),  # second row inconsistent
        }
        with pytest.raises(ValueError, match="Interval inconsistency"):
            EventData._normalize_intervals_vectorized(
                processed, policy=IntervalPolicy.strict, number_type=NumberType.float
            )

    def test_warn_recomputes_duration(self, caplog):
        """Vectorized: warn policy recomputes duration from end."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
            "end": _make_coord_array([1.0, 3.0]),
            "duration": _make_coord_array([1.0, 5.0]),
        }
        with caplog.at_level(logging.WARNING, logger="timetoalign.storage.events"):
            result = EventData._normalize_intervals_vectorized(
                processed, policy=IntervalPolicy.warn, number_type=NumberType.float
            )
        dur_vals = result["duration"].field("value").to_pylist()
        assert dur_vals == [1.0, 2.0]
        # Warning should have been logged
        assert any("inconsistency" in r.message.lower() for r in caplog.records)

    def test_prefer_end_always_recomputes_duration(self):
        """Vectorized: prefer_end recomputes duration even when consistent."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
            "end": _make_coord_array([1.0, 3.0]),
            "duration": _make_coord_array([1.0, 5.0]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed, policy=IntervalPolicy.prefer_end, number_type=NumberType.float
        )
        dur_vals = result["duration"].field("value").to_pylist()
        assert dur_vals == [1.0, 2.0]
        end_vals = result["end"].field("value").to_pylist()
        assert end_vals == [1.0, 3.0]

    def test_prefer_duration_recomputes_end(self):
        """Vectorized: prefer_duration recomputes end from duration."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
            "end": _make_coord_array([1.0, 3.0]),
            "duration": _make_coord_array([1.0, 5.0]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed,
            policy=IntervalPolicy.prefer_duration,
            number_type=NumberType.float,
        )
        end_vals = result["end"].field("value").to_pylist()
        assert end_vals == [1.0, 6.0]
        dur_vals = result["duration"].field("value").to_pylist()
        assert dur_vals == [1.0, 5.0]

    def test_temporal_type_inferred(self):
        """Vectorized: temporal_type set to 'interval' when end present."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
            "end": _make_coord_array([1.0, 2.0]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed, number_type=NumberType.float
        )
        tt = result["temporal_type"]
        if isinstance(tt, pa.Array):
            tt = tt.to_pylist()
        assert list(tt) == ["interval", "interval"]

    def test_no_end_no_duration_stays_instant(self):
        """Vectorized: rows with no end/duration remain instant."""
        processed = {
            "start": _make_coord_array([0.0, 1.0]),
        }
        result = EventData._normalize_intervals_vectorized(
            processed, number_type=NumberType.float
        )
        tt = result["temporal_type"]
        if isinstance(tt, pa.Array):
            tt = tt.to_pylist()
        assert list(tt) == ["instant", "instant"]


# ---------------------------------------------------------------------------
# Integration: from_dicts path
# ---------------------------------------------------------------------------


class TestFromDictsIntegration:
    """Integration tests: ``EventData.from_dicts`` with ``interval_policy``."""

    def test_end_only_fills_duration(self):
        """from_dicts with end-only rows produces correct duration."""
        rows = [
            _make_row(start=0.0, end=1.0, event_id="n1"),
            _make_row(start=1.0, end=2.5, event_id="n2"),
        ]
        store = EventData.from_dicts(rows, TimeUnit.seconds)
        df = store.to_dataframe()
        # to_dataframe() flattens coordinate structs to float64
        for idx, expected_dur in enumerate([1.0, 1.5]):
            dur_val = df.iloc[idx]["duration"]
            assert dur_val == expected_dur

    def test_duration_only_fills_end(self):
        """from_dicts with duration-only rows produces correct end."""
        rows = [
            _make_row(start=0.0, duration=1.0, event_id="n1"),
            _make_row(start=1.0, duration=1.5, event_id="n2"),
        ]
        store = EventData.from_dicts(rows, TimeUnit.seconds)
        df = store.to_dataframe()
        for idx, expected_end in enumerate([1.0, 2.5]):
            end_val = df.iloc[idx]["end"]
            assert end_val == expected_end

    def test_strict_policy_raises(self):
        """from_dicts with strict policy raises on inconsistency."""
        rows = [
            _make_row(start=0.0, end=1.0, duration=5.0, event_id="n1"),
        ]
        with pytest.raises(ValueError, match="Interval inconsistency"):
            EventData.from_dicts(
                rows, TimeUnit.seconds, interval_policy=IntervalPolicy.strict
            )

    def test_filter_rejects_wrong_coordinate_unit(self):
        """Coordinate bounds with another unit fail explicitly."""
        store = EventData.from_dicts(
            [_make_row(start=0.0), _make_row(start=1.0, event_id="n2")],
            TimeUnit.seconds,
        )
        with pytest.raises(ValueError, match="ticks.*seconds|seconds.*ticks"):
            store.filter(min_coord=Coordinate(0, TimeUnit.ticks))

    def test_filter_accepts_matching_coordinate_unit(self):
        """Coordinate bounds with the EventData unit preserve numeric filtering."""
        store = EventData.from_dicts(
            [_make_row(start=0.0), _make_row(start=1.0, event_id="n2")],
            TimeUnit.seconds,
        )
        filtered = store.filter(min_coord=Coordinate(1, TimeUnit.seconds))
        assert filtered.table.column("id").to_pylist() == ["n2"]

    def test_filter_fraction_bounds_and_half_open_range(self) -> None:
        """Fraction bounds select exact rows with inclusive/exclusive limits."""
        store = EventData.from_dicts(
            [
                _make_row(start=0.0, event_id="before"),
                _make_row(start=0.5, event_id="at_min"),
                _make_row(start=1.0, event_id="at_max"),
            ],
            TimeUnit.seconds,
        )

        raw = store.filter(min_coord=Fraction(1, 2), max_coord=Fraction(1))
        coordinate = store.filter(
            min_coord=Coordinate(Fraction(1, 2), TimeUnit.seconds),
            max_coord=Coordinate(Fraction(1), TimeUnit.seconds),
        )
        integer_and_float = store.filter(min_coord=0, max_coord=1.0)

        assert raw.table.column("id").to_pylist() == ["at_min"]
        assert coordinate.table.column("id").to_pylist() == ["at_min"]
        assert integer_and_float.table.column("id").to_pylist() == [
            "before",
            "at_min",
        ]


# ---------------------------------------------------------------------------
# Integration: CsvLoader end-to-end
# ---------------------------------------------------------------------------


class TestCsvLoaderIntegration:
    """Verify CsvLoader computes duration from end-start (the original bug)."""

    def test_csv_loader_fills_duration(self):
        """CsvLoader on test_events.csv should produce duration for all rows."""
        from timetoalign.loader.tabular.csv import CsvLoader
        from timetoalign.testdata import ensure_data

        csv_path = ensure_data("tabular") / "test_events.csv"

        loader = CsvLoader()
        loader.load(csv_path)
        df = loader.events.to_dataframe()

        # The CSV has 5 rows with start and end, no duration column.
        # After normalization, every row must have duration = end - start.
        assert len(df) == 5
        for idx, row in df.iterrows():
            start_val = float(row["start"])
            end_val = float(row["end"])
            dur_val = float(row["duration"])
            expected = end_val - start_val
            assert (
                dur_val == expected
            ), f"Row {idx}: duration={dur_val}, expected={expected}"


# ---------------------------------------------------------------------------
# IntervalPolicy enum basics
# ---------------------------------------------------------------------------


class TestIntervalPolicyEnum:
    """Basic enum behaviour and string coercion."""

    def test_values_exist(self):
        assert IntervalPolicy.warn is not None
        assert IntervalPolicy.prefer_end is not None
        assert IntervalPolicy.prefer_duration is not None
        assert IntervalPolicy.strict is not None

    def test_string_coercion(self):
        """FancyStrEnum should support string-based construction."""
        assert IntervalPolicy("warn") == IntervalPolicy.warn
        assert IntervalPolicy("prefer_end") == IntervalPolicy.prefer_end
        assert IntervalPolicy("prefer_duration") == IntervalPolicy.prefer_duration
        assert IntervalPolicy("strict") == IntervalPolicy.strict

    def test_default_is_warn(self):
        """Default policy in the normalization methods is 'warn'."""
        import inspect

        sig = inspect.signature(EventData._normalize_intervals_row)
        default = sig.parameters["policy"].default
        assert default == IntervalPolicy.warn
