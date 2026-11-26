"""Tests for loader/store.py (EventData)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.storage import EventData


class TestEventDataCreation:
    """Tests for EventData creation methods."""

    def test_empty_store(self) -> None:
        """Can create an empty EventData."""
        store = EventData.empty(TimeUnit.ticks)
        assert len(store) == 0
        assert store.unit == TimeUnit.ticks
        assert store.number_type == NumberType.int

    def test_empty_with_number_type(self) -> None:
        """Can specify number_type for empty store."""
        store = EventData.empty(TimeUnit.seconds, NumberType.fraction)
        assert store.number_type == NumberType.fraction

    def test_from_dicts_empty_list(self) -> None:
        """from_dicts with empty list returns empty store."""
        store = EventData.from_dicts([], TimeUnit.ticks)
        assert len(store) == 0

    def test_from_dicts_instant_events(
        self, sample_instant_events: list[dict[str, Any]]
    ) -> None:
        """from_dicts creates store from instant events."""
        store = EventData.from_dicts(sample_instant_events, TimeUnit.ticks)
        assert len(store) == 3
        assert store.unit == TimeUnit.ticks

    def test_from_dicts_interval_events(
        self, sample_interval_events: list[dict[str, Any]]
    ) -> None:
        """from_dicts creates store from interval events."""
        store = EventData.from_dicts(sample_interval_events, TimeUnit.ticks)
        assert len(store) == 2

    def test_from_dicts_mixed_events(
        self, sample_mixed_events: list[dict[str, Any]]
    ) -> None:
        """from_dicts creates store from mixed events."""
        store = EventData.from_dicts(sample_mixed_events, TimeUnit.ticks)
        assert len(store) == 5

    def test_from_dicts_with_fractions(
        self, sample_fraction_events: list[dict[str, Any]]
    ) -> None:
        """from_dicts handles Fraction coordinates."""
        store = EventData.from_dicts(
            sample_fraction_events, TimeUnit.quarters, NumberType.fraction
        )
        assert len(store) == 3

    def test_from_arrays_empty(self) -> None:
        """from_arrays with empty columns returns empty store."""
        store = EventData.from_arrays({}, TimeUnit.ticks)
        assert len(store) == 0

    def test_from_arrays_instant_events(self) -> None:
        """from_arrays creates store from column arrays."""
        store = EventData.from_arrays(
            {
                "id": ["b1", "b2"],
                "temporal_type": ["instant", "instant"],
                "event_type": ["Beat", "Beat"],
                "instant": [0, 480],
            },
            TimeUnit.ticks,
        )
        assert len(store) == 2

    def test_from_arrays_missing_columns(self) -> None:
        """from_arrays infers missing temporal_type/event_type from data.

        When temporal_type and event_type are not provided:
        - temporal_type is inferred from presence of end coordinate:
          - "instant" if no end coordinate
          - "interval" if end coordinate present
        - event_type defaults to "Event"
        - name remains None
        """
        # Missing temporal_type, event_type, name
        store = EventData.from_arrays(
            {
                "id": ["b1"],
                "instant": [0],
            },
            TimeUnit.ticks,
        )
        assert len(store) == 1
        row = list(store)[0]
        # temporal_type inferred as "instant" because no end coordinate
        assert row["temporal_type"] == "instant"
        # event_type defaults to "Event"
        assert row["event_type"] == "Event"
        # name remains None
        assert row["name"] is None

    def test_from_dataframe_empty(self) -> None:
        """from_dataframe with empty DataFrame returns empty store."""
        df = pd.DataFrame()
        store = EventData.from_dataframe(df, TimeUnit.ticks)
        assert len(store) == 0

    def test_from_dataframe(self) -> None:
        """from_dataframe creates store from DataFrame."""
        df = pd.DataFrame(
            {
                "id": ["b1", "b2"],
                "temporal_type": ["instant", "instant"],
                "event_type": ["Beat", "Beat"],
                "instant": [0, 480],
            }
        )
        store = EventData.from_dataframe(df, TimeUnit.ticks)
        assert len(store) == 2

    def test_from_dicts_with_explicit_none_coords(self) -> None:
        """from_dicts handles explicit None in coordinate columns.

        Note: The schema no longer has an 'instant' column. InstantEvents
        use 'start' with 'end' being null.
        """
        events = [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "NullEvent",
                "start": None,
                "end": None,
            }
        ]
        store = EventData.from_dicts(events, TimeUnit.ticks)
        assert len(store) == 1
        row = list(store)[0]
        # Check raw row values (structs should be null)
        assert row["start"] is None
        assert row["end"] is None


class TestEventDataSchema:
    """Tests for EventData schema methods."""

    def test_get_schema_returns_pyarrow_schema(self) -> None:
        """get_schema() returns a PyArrow schema."""
        schema = EventData.get_schema(TimeUnit.ticks)
        assert isinstance(schema, pa.Schema)

    def test_get_schema_has_base_columns(self) -> None:
        """get_schema() includes all base columns.

        Note: The schema no longer has a separate 'instant' column.
        """
        schema = EventData.get_schema(TimeUnit.ticks)
        assert "id" in schema.names
        assert "start" in schema.names
        assert "end" in schema.names

    def test_instance_schema_property(self) -> None:
        """Instance .schema returns the table's PyArrow schema."""
        data = EventData.from_dicts(
            [{"event_type": "Beat", "instant": 0}],
            unit=TimeUnit.ticks,
        )
        assert isinstance(data.schema, pa.Schema)
        assert "id" in data.schema.names
        assert "start" in data.schema.names

    def test_field_names(self) -> None:
        """field_names() returns the list of base EventData field names.

        Note: Base fields are: id, name, temporal_type, event_type, start, end, duration.
        """
        names = EventData.field_names()
        assert isinstance(names, list)
        assert "id" in names
        assert len(names) == 7  # Base fields only


class TestEventDataProperties:
    """Tests for EventData properties."""

    def test_table_property(self, store_with_instants: EventData) -> None:
        """table property returns PyArrow table."""
        assert isinstance(store_with_instants.table, pa.Table)

    def test_unit_property(self, store_with_instants: EventData) -> None:
        """unit property returns TimeUnit."""
        assert store_with_instants.unit == TimeUnit.ticks

    def test_number_type_property(self, store_with_instants: EventData) -> None:
        """number_type property returns NumberType."""
        assert store_with_instants.number_type == NumberType.int

    def test_count_property(self, store_with_instants: EventData) -> None:
        """count property returns number of events."""
        assert store_with_instants.count == 3

    def test_len(self, store_with_instants: EventData) -> None:
        """__len__ returns count."""
        assert len(store_with_instants) == 3


class TestEventDataIteration:
    """Tests for EventData iteration."""

    def test_iter_yields_dicts(self, store_with_instants: EventData) -> None:
        """__iter__ yields row dictionaries."""
        rows = list(store_with_instants)
        assert len(rows) == 3
        assert all(isinstance(r, dict) for r in rows)

    def test_iter_includes_id(self, store_with_instants: EventData) -> None:
        """Iterated rows include id field."""
        rows = list(store_with_instants)
        ids = [r["id"] for r in rows]
        assert "beat_1" in ids


class TestEventDataRepr:
    """Tests for EventData __repr__."""

    def test_repr(self, store_with_instants: EventData) -> None:
        """__repr__ returns informative string."""
        r = repr(store_with_instants)
        assert "EventData" in r
        assert "3" in r  # count
        assert "ticks" in r


class TestEventDataConcat:
    """Tests for EventData concat, the one join, which changes no table."""

    def test_concat(
        self,
        store_with_instants: EventData,
        store_with_intervals: EventData,
    ) -> None:
        """concat() returns new store with all events."""
        combined = store_with_instants.concat(store_with_intervals)
        assert len(combined) == 5
        # Original unchanged
        assert len(store_with_instants) == 3

    def test_concat_of_another_unit_keeps_each_rows_axis(
        self, store_with_instants: EventData
    ) -> None:
        """concat() of another unit keys each row by its axis."""
        other = EventData.from_dicts(
            [{"id": "s", "event_type": "Beat", "instant": 1.0}], TimeUnit.seconds
        )
        combined = store_with_instants.concat(other)
        assert combined.table.column("timeline_id").to_pylist()[-2:] == [
            "<ticks int>",
            "<seconds float>",
        ]
        assert len(store_with_instants) == 3


class TestEventDataFilter:
    """Tests for EventData filter method."""

    def test_filter_by_temporal_type(self, store_with_mixed: EventData) -> None:
        """filter() by temporal_type works."""
        instants = store_with_mixed.filter(temporal_type="instant")
        assert len(instants) == 3

        intervals = store_with_mixed.filter(temporal_type="interval")
        assert len(intervals) == 2

    def test_filter_by_event_type(self, store_with_mixed: EventData) -> None:
        """filter() by event_type works."""
        beats = store_with_mixed.filter(event_type="Beat")
        assert len(beats) == 3

        notes = store_with_mixed.filter(event_type="Note")
        assert len(notes) == 2

    def test_filter_by_min_coord(self, store_with_instants: EventData) -> None:
        """filter() by min_coord works."""
        filtered = store_with_instants.filter(min_coord=480)
        assert len(filtered) == 2  # beat_2 at 480, beat_3 at 960

    def test_filter_by_max_coord(self, store_with_instants: EventData) -> None:
        """filter() by max_coord works."""
        filtered = store_with_instants.filter(max_coord=480)
        assert len(filtered) == 1  # beat_1 at 0

    def test_filter_combined(self, store_with_mixed: EventData) -> None:
        """filter() with multiple criteria."""
        filtered = store_with_mixed.filter(
            temporal_type="interval", event_type="Note", min_coord=100
        )
        assert len(filtered) == 1  # note_2 starts at 240

    def test_filter_no_criteria_returns_self(
        self, store_with_instants: EventData
    ) -> None:
        """filter() with no criteria returns same store."""
        filtered = store_with_instants.filter()
        assert filtered is store_with_instants

    def test_filter_returns_new_eventstore(
        self, store_with_instants: EventData
    ) -> None:
        """filter() returns new EventData instance."""
        filtered = store_with_instants.filter(min_coord=0)
        assert isinstance(filtered, EventData)


class TestEventDataSelect:
    """Tests for EventData select method."""

    def test_select_columns(self, store_with_instants: EventData) -> None:
        """select() returns table with specified columns."""
        result = store_with_instants.select(["id", "event_type"])
        assert isinstance(result, pa.Table)
        assert result.num_columns == 2
        assert "id" in result.schema.names


class TestEventDataWhere:
    """Tests for EventData where method."""

    def test_where_with_expression(self, store_with_instants: EventData) -> None:
        """where() filters with custom expression."""
        expr = pc.equal(pc.field("event_type"), "Beat")
        filtered = store_with_instants.where(expr)
        assert len(filtered) == 3


class TestEventDataStats:
    """Tests for EventData stats methods."""

    def test_count_by(self, store_with_mixed: EventData) -> None:
        """count_by() groups and counts."""
        counts = store_with_mixed.count_by("temporal_type")
        assert counts["instant"] == 3
        assert counts["interval"] == 2

    def test_coordinate_range_empty(self, empty_store: EventData) -> None:
        """coordinate_range() returns None for empty store."""
        assert empty_store.coordinate_range() is None

    def test_coordinate_range(self, store_with_mixed: EventData) -> None:
        """coordinate_range() returns min/max."""
        range_ = store_with_mixed.coordinate_range()
        assert range_ is not None
        assert range_[0] == 0.0
        assert range_[1] == 960.0  # beat_3 at 960

    def test_coordinate_range_all_nulls(self) -> None:
        """coordinate_range() returns None if all coords are null."""
        events = [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "NullEvent",
                "instant": None,
            }
        ]
        store = EventData.from_dicts(events, TimeUnit.ticks)
        assert store.count == 1
        assert store.coordinate_range() is None

    def test_event_types(self, store_with_mixed: EventData) -> None:
        """event_types() returns unique types."""
        types = store_with_mixed.event_types()
        assert set(types) == {"Beat", "Note"}

    def test_summary(self, store_with_mixed: EventData) -> None:
        """summary() returns comprehensive stats."""
        summary = store_with_mixed.summary()
        assert summary["count"] == 5
        assert "temporal_types" in summary
        assert "event_types" in summary
        assert "coordinate_range" in summary


class TestEventDataSerialization:
    """Tests for EventData Parquet serialization."""

    def test_to_parquet(
        self, store_with_mixed: EventData, temp_parquet_path: Path
    ) -> None:
        """to_parquet() writes Parquet file."""
        store_with_mixed.to_parquet(temp_parquet_path)
        assert temp_parquet_path.exists()

    def test_from_parquet(
        self, store_with_mixed: EventData, temp_parquet_path: Path
    ) -> None:
        """from_parquet() loads EventData."""
        store_with_mixed.to_parquet(temp_parquet_path)
        loaded = EventData.from_parquet(temp_parquet_path)

        assert len(loaded) == len(store_with_mixed)
        assert loaded.unit == store_with_mixed.unit

    def test_from_parquet_no_metadata_raises(self, temp_parquet_path: Path) -> None:
        """from_parquet() raises if no TimeToAlign metadata."""
        # Create a parquet file without our metadata
        table = pa.table({"id": ["a", "b"]})
        import pyarrow.parquet as pq

        pq.write_table(table, temp_parquet_path)

        with pytest.raises(ValueError, match="lacks TimeToAlign"):
            EventData.from_parquet(temp_parquet_path)

    def test_to_dataframe(self, store_with_mixed: EventData) -> None:
        """to_dataframe() returns pandas DataFrame."""
        df = store_with_mixed.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5


class TestEventDataSubclass:
    """Tests for EventData subclassing."""

    def test_subclass_with_extra_fields(self) -> None:
        """Subclass can add extra fields."""

        class NoteEventData(EventData):
            _extra_fields = [
                pa.field("pitch", pa.int8()),
                pa.field("velocity", pa.int8()),
            ]

        schema = NoteEventData.get_schema(TimeUnit.ticks)
        assert "pitch" in schema.names
        assert "velocity" in schema.names

        names = NoteEventData.field_names()
        assert "pitch" in names
