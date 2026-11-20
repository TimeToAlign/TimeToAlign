"""Tests for Timeline timestamp generation functionality.

Tests for cross-section timestamp tables that show
synchronous coordinates across timeline hierarchies.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from tests.helpers import coordinate_column, coordinate_values, table_column
from timetoalign.core import NumberType, TimeUnit
from timetoalign.maps import ScalarMap
from timetoalign.timelines import Timeline

# region Fixtures


@pytest.fixture
def simple_timeline() -> Timeline:
    """A simple timeline with instant and interval events."""
    tl = Timeline(length=10, unit=TimeUnit.seconds, uid="simple")
    tl.add_events(
        [
            {
                "id": "e1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "e2",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 2.5,
            },
            {
                "id": "e3",
                "temporal_type": "interval",
                "event_type": "Note",
                "start": 1.0,
                "end": 3.0,
            },
            {
                "id": "e4",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 5.0,
            },
        ]
    )
    return tl


@pytest.fixture
def empty_timeline() -> Timeline:
    """An empty timeline with no events."""
    return Timeline(length=10, unit=TimeUnit.seconds, uid="empty")


@pytest.fixture
def nested_timeline() -> tuple[Timeline, Timeline, Timeline]:
    """A parent timeline with two nested children."""
    parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent")
    parent.add_events(
        [
            {
                "id": "p1",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 0.0,
            },
            {
                "id": "p2",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": 50.0,
            },
        ]
    )

    child1 = Timeline(length=20, unit=TimeUnit.seconds, uid="child1")
    child1.add_events(
        [
            {
                "id": "c1a",
                "temporal_type": "instant",
                "event_type": "Note",
                "instant": 0.0,
            },
            {
                "id": "c1b",
                "temporal_type": "instant",
                "event_type": "Note",
                "instant": 10.0,
            },
        ]
    )

    child2 = Timeline(length=15, unit=TimeUnit.seconds, uid="child2")
    child2.add_events(
        [
            {
                "id": "c2a",
                "temporal_type": "instant",
                "event_type": "Note",
                "instant": 5.0,
            },
        ]
    )

    # Add children at different offsets
    parent.add_child(child1, offset=10)  # child1 spans [10, 30]
    parent.add_child(child2, offset=60)  # child2 spans [60, 75]

    return parent, child1, child2


@pytest.fixture
def deeply_nested_timeline() -> Timeline:
    """A timeline with 3 levels of nesting."""
    root = Timeline(length=100, unit=TimeUnit.seconds, uid="root")

    level1 = Timeline(length=50, unit=TimeUnit.seconds, uid="level1")
    level1.add_events(
        [
            {
                "id": "l1",
                "temporal_type": "instant",
                "event_type": "Mark",
                "instant": 25.0,
            },
        ]
    )

    level2 = Timeline(length=20, unit=TimeUnit.seconds, uid="level2")
    level2.add_events(
        [
            {
                "id": "l2",
                "temporal_type": "instant",
                "event_type": "Mark",
                "instant": 10.0,
            },
        ]
    )

    # Nest: root -> level1 (at 10) -> level2 (at 15)
    level1.add_child(level2, offset=15)  # level2 in level1 coords: [15, 35]
    root.add_child(level1, offset=10)  # level1 in root coords: [10, 60]
    # level2 in root coords: [10+15, 10+35] = [25, 45]

    return root


# endregion


# region Tests: _extract_event_coordinates


class TestExtractEventCoordinates:
    """Tests for Timeline._extract_event_coordinates()."""

    def test_extracts_instant_coordinates(self, simple_timeline: Timeline) -> None:
        """Instant event coordinates are extracted."""
        coords = coordinate_values(simple_timeline._extract_event_coordinates())
        assert 0.0 in coords
        assert 2.5 in coords
        assert 5.0 in coords

    def test_extracts_interval_start_and_end(self, simple_timeline: Timeline) -> None:
        """Both start and end of intervals are extracted."""
        coords = coordinate_values(simple_timeline._extract_event_coordinates())
        assert 1.0 in coords  # interval start
        assert 3.0 in coords  # interval end

    def test_deduplicates_coordinates(self) -> None:
        """Duplicate coordinates are deduplicated."""
        tl = Timeline(length=10, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 1.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 1.0,
                },
                {
                    "id": "e3",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 1.0,
                    "end": 2.0,
                },
            ]
        )
        coords = coordinate_values(tl._extract_event_coordinates())
        assert coords.count(1.0) == 1

    def test_returns_sorted_coordinates(self, simple_timeline: Timeline) -> None:
        """Coordinates are returned in ascending order."""
        coords = coordinate_values(simple_timeline._extract_event_coordinates())
        assert coords == sorted(coords)

    def test_empty_timeline_returns_empty_array(self, empty_timeline: Timeline) -> None:
        """Empty timeline returns empty array."""
        coords = empty_timeline._extract_event_coordinates()
        assert len(coords) == 0

    def test_returns_pyarrow_type(self, simple_timeline: Timeline) -> None:
        """Returns PyArrow ChunkedArray."""
        coords = simple_timeline._extract_event_coordinates()
        assert isinstance(coords, (pa.Array, pa.ChunkedArray))


# endregion


# region Tests: _collect_all_coordinates


class TestCollectAllCoordinates:
    """Tests for Timeline._collect_all_coordinates()."""

    def test_includes_own_coordinates(self, simple_timeline: Timeline) -> None:
        """Own event coordinates are included."""
        coords = coordinate_values(simple_timeline._collect_all_coordinates())
        assert 0.0 in coords
        assert 2.5 in coords

    def test_includes_child_coordinates_with_offset(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """Child coordinates are offset-adjusted to parent coordinates."""
        parent, child1, child2 = nested_timeline

        coord_list = coordinate_values(parent._collect_all_coordinates())

        # child1 at offset 10: events at 0, 10 become 10, 20
        assert 10.0 in coord_list  # child1 event at local 0
        assert 20.0 in coord_list  # child1 event at local 10

        # child2 at offset 60: event at 5 becomes 65
        assert 65.0 in coord_list

    def test_recursion_limit_zero_excludes_children(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """recursion_limit=0 excludes all children."""
        parent, _, _ = nested_timeline

        coord_list = coordinate_values(
            parent._collect_all_coordinates(recursion_limit=0)
        )

        # Should only have parent's own events (plus segment events)
        # Parent events: 0.0, 50.0, segment events at 10, 30, 60, 75
        assert 0.0 in coord_list
        assert 50.0 in coord_list

        # Child-specific coordinates should NOT be present
        assert 20.0 not in coord_list  # child1 local 10 + offset 10
        assert 65.0 not in coord_list  # child2 local 5 + offset 60

    def test_recursion_limit_one_includes_direct_children(
        self, deeply_nested_timeline: Timeline
    ) -> None:
        """recursion_limit=1 includes direct children only."""
        coord_list = coordinate_values(
            deeply_nested_timeline._collect_all_coordinates(recursion_limit=1)
        )

        # level1's event at local 25 becomes root 35
        assert 35.0 in coord_list

        # level2's event (nested in level1) should NOT be included
        # level2 at level1-local 15, event at local 10 -> level1-local 25 -> root 35
        # But with limit=1, we don't recurse into level1's children

    def test_deduplicates_across_hierarchy(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """Coordinates are deduplicated across parent and children."""
        parent, child1, _ = nested_timeline

        # Add an event to parent at same coord as offset child event
        parent.add_events(
            [
                {
                    "id": "dup",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 10.0,
                }
            ],
            allow_expansion=True,
        )

        coords = coordinate_values(parent._collect_all_coordinates())
        # child1's event at local 0 with offset 10 = root 10.0
        # parent's new event at 10.0
        # Should only appear once
        assert coords.count(10.0) == 1


# endregion


# region Tests: _compute_local_coordinates


class TestComputeLocalCoordinates:
    """Tests for Timeline._compute_local_coordinates()."""

    def test_zero_offset_returns_same_coordinates(
        self, simple_timeline: Timeline
    ) -> None:
        """With offset=0, local coords equal root coords."""
        root_coords = coordinate_column([0.0, 2.5, 5.0, 10.0])
        local = simple_timeline._compute_local_coordinates(root_coords, offset=0.0)
        assert coordinate_values(local) == [0.0, 2.5, 5.0, 10.0]

    def test_positive_offset_subtracts(self, simple_timeline: Timeline) -> None:
        """Positive offset is subtracted from root coordinates."""
        root_coords = coordinate_column([10.0, 15.0, 20.0])
        local = simple_timeline._compute_local_coordinates(root_coords, offset=10.0)
        assert coordinate_values(local) == [0.0, 5.0, 10.0]

    def test_out_of_bounds_low_returns_null(self, simple_timeline: Timeline) -> None:
        """Coordinates below 0 (local) return null."""
        root_coords = coordinate_column([0.0, 5.0, 10.0])
        local = simple_timeline._compute_local_coordinates(root_coords, offset=5.0)
        # local = [0-5=-5, 5-5=0, 10-5=5]
        # -5 is out of bounds (< 0)
        result = coordinate_values(local)
        assert result[0] is None  # -5 is invalid
        assert result[1] == 0.0
        assert result[2] == 5.0

    def test_out_of_bounds_high_returns_null(self, simple_timeline: Timeline) -> None:
        """Coordinates above timeline length return null."""
        root_coords = coordinate_column([0.0, 5.0, 15.0])  # timeline length is 10
        local = simple_timeline._compute_local_coordinates(root_coords, offset=0.0)
        result = coordinate_values(local)
        assert result[0] == 0.0
        assert result[1] == 5.0
        assert result[2] is None  # 15 > 10

    def test_empty_input_returns_empty(self, simple_timeline: Timeline) -> None:
        """Empty input returns empty array."""
        root_coords = coordinate_column([])
        local = simple_timeline._compute_local_coordinates(root_coords, offset=0.0)
        assert len(local) == 0


# endregion


# region Tests: get_timestamp_table


class TestGetTimestampTable:
    """Tests for Timeline.get_timestamp_table()."""

    def test_returns_pyarrow_table(self, simple_timeline: Timeline) -> None:
        """Returns a PyArrow Table."""
        table = simple_timeline.get_timestamp_table()
        assert isinstance(table, pa.Table)

    def test_every_column_names_a_timeline(self, simple_timeline: Timeline) -> None:
        """No `axis` field duplicating the receiver's own column."""
        table = simple_timeline.get_timestamp_table()
        assert table.column_names == ["simple"]
        assert "axis" not in table.column_names

    def test_has_timeline_id_column(self, simple_timeline: Timeline) -> None:
        """Table has field for the timeline's ID."""
        table = simple_timeline.get_timestamp_table()
        assert "simple" in table.column_names

    def test_includes_child_columns(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """Table includes fields for all children."""
        parent, child1, child2 = nested_timeline
        table = parent.get_timestamp_table()

        assert "parent" in table.column_names
        assert "child1" in table.column_names
        assert "child2" in table.column_names

    def test_explicit_coordinates(self, simple_timeline: Timeline) -> None:
        """Explicit coordinates are used when provided."""
        coords = [0.0, 2.5, 5.0, 7.5, 10.0]
        table = simple_timeline.get_timestamp_table(coords)

        assert table_column(table, "simple") == coords

    def test_explicit_coordinates_as_numpy(self, simple_timeline: Timeline) -> None:
        """Accepts numpy array for coordinates."""
        coords = np.array([0.0, 2.5, 5.0])
        table = simple_timeline.get_timestamp_table(coords)
        assert len(table) == 3

    def test_explicit_coordinates_as_pyarrow(self, simple_timeline: Timeline) -> None:
        """Accepts PyArrow array for coordinates."""
        coords = pa.array([0.0, 2.5, 5.0])
        table = simple_timeline.get_timestamp_table(coords)
        assert len(table) == 3

    def test_include_boundaries_adds_endpoints(self, simple_timeline: Timeline) -> None:
        """include_boundaries=True adds 0 and length."""
        table = simple_timeline.get_timestamp_table(include_boundaries=True)
        axis = table_column(table, "simple")
        assert 0.0 in axis
        assert 10.0 in axis  # timeline length

    def test_recursion_limit_controls_depth(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """recursion_limit controls how many levels of children appear."""
        parent, _, _ = nested_timeline

        # With limit=0, should only have parent field
        table = parent.get_timestamp_table(recursion_limit=0)
        assert "parent" in table.column_names
        # Children should still appear in fields (they're iterated separately)
        # but their coordinates won't be in the axis

    def test_empty_timeline_returns_empty_table(self, empty_timeline: Timeline) -> None:
        """Empty timeline returns table with 0 rows."""
        table = empty_timeline.get_timestamp_table()
        assert len(table) == 0
        assert table.column_names == [empty_timeline.id]

    def test_local_coordinates_are_correct(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """Local coordinates are correctly computed for children."""
        parent, child1, child2 = nested_timeline

        # Use explicit coordinate that's in child1's range
        table = parent.get_timestamp_table([15.0])

        # At root=15, child1 (offset=10) should show local=5
        assert table_column(table, "child1") == [5.0]

        # child2 (offset=60) is out of range, so its cell is absent
        assert pd.isna(table_column(table, "child2")[0])


# endregion


# region Tests: DataFrame rendering


class TestDataFrameFormat:
    """Tests for ``Timeline.get_timestamp_table(format="dataframe")``."""

    def test_returns_dataframe(self, simple_timeline: Timeline) -> None:
        """Returns a pandas DataFrame."""
        df = simple_timeline.get_timestamp_table(format="dataframe")
        assert isinstance(df, pd.DataFrame)

    def test_matches_the_decoded_arrow_table(self, simple_timeline: Timeline) -> None:
        """The frame carries the same values the Arrow table carries."""
        df = simple_timeline.get_timestamp_table(
            format="dataframe", units=False, include_ids=False
        )
        table = simple_timeline.get_timestamp_table()
        assert df.columns.tolist() == table.column_names
        for name in table.column_names:
            assert df[name].tolist() == table_column(table, name)

    def test_passes_through_parameters(self, simple_timeline: Timeline) -> None:
        """All parameters reach the table builder."""
        df = simple_timeline.get_timestamp_table(
            [0.0, 5.0, 10.0],
            include_boundaries=True,
            format="dataframe",
        )
        assert len(df) == 3

    def test_include_ids_uses_event_ids_as_index(
        self, simple_timeline: Timeline
    ) -> None:
        """include_ids retains the absorbed event-index behavior."""
        df = simple_timeline.get_timestamp_table(
            format="dataframe", units=False, include_ids=True
        )
        assert df.index.name == "id"
        assert df.index.tolist() == ["e1", "e3", "e2", "", "e4"]


# endregion


# region Tests: get_boundary_table


class TestGetBoundaryTable:
    """Tests for Timeline.get_boundary_table()."""

    def test_includes_own_boundaries(self, simple_timeline: Timeline) -> None:
        """Includes this timeline's start (0) and end (length)."""
        table = simple_timeline.get_boundary_table()
        axis = table_column(table, "simple")
        assert 0.0 in axis
        assert 10.0 in axis

    def test_includes_child_boundaries(
        self, nested_timeline: tuple[Timeline, Timeline, Timeline]
    ) -> None:
        """Includes boundaries of all children."""
        parent, child1, child2 = nested_timeline
        table = parent.get_boundary_table()
        axis = table_column(table, "parent")

        # child1 at offset 10, length 20: boundaries at 10, 30
        assert 10.0 in axis
        assert 30.0 in axis

        # child2 at offset 60, length 15: boundaries at 60, 75
        assert 60.0 in axis
        assert 75.0 in axis

    def test_does_not_include_events(self, simple_timeline: Timeline) -> None:
        """Only boundaries are included, not event coordinates."""
        table = simple_timeline.get_boundary_table()
        axis = table_column(table, "simple")

        # Event at 2.5 should not be in boundaries
        assert 2.5 not in axis

    def test_recursion_limit_controls_depth(
        self, deeply_nested_timeline: Timeline
    ) -> None:
        """recursion_limit controls how deep to collect boundaries."""
        root = deeply_nested_timeline

        # With limit=1, should have root and level1 boundaries
        table = root.get_boundary_table(recursion_limit=1)
        axis = table_column(table, root.id)

        # root: 0, 100
        assert 0.0 in axis
        assert 100.0 in axis

        # level1 at offset 10, length 50: 10, 60
        assert 10.0 in axis
        assert 60.0 in axis


# endregion


# region Tests: Timestamp Access


class TestTimestampAccess:
    """Tests for Timeline timestamp accessors."""

    def test_get_timestamp_at_without_conversion_maps(self) -> None:
        """Disabled conversion maps leave timeline access available."""
        timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="timeline")
        timeline.add_conversion_map(
            ScalarMap(
                scalar=1000.0,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.milliseconds,
                uid="seconds-to-milliseconds",
            )
        )

        ts = timeline.get_timestamp_at(3.0, conversion_maps=False)

        with pytest.raises(KeyError):
            ts.get_unit(TimeUnit.milliseconds)
        with pytest.raises(KeyError):
            ts.get_conversion_for("milliseconds")
        assert ts.get_coordinate_for("timeline", format="float") == 3.0

    def test_get_timestamp_at_with_restricted_conversion_maps(self) -> None:
        """Restricted conversion maps expose only their target unit."""
        timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="timeline")
        timeline.add_conversion_map(
            ScalarMap(
                scalar=1000.0,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.milliseconds,
                uid="seconds-to-milliseconds",
            )
        )
        timeline.add_conversion_map(
            ScalarMap(
                scalar=2.0,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.samples,
                uid="seconds-to-samples",
            )
        )

        ts = timeline.get_timestamp_at(
            3.0,
            conversion_maps=[TimeUnit.milliseconds],
        )

        assert ts.get_unit(TimeUnit.milliseconds, format="float") == 3000.0
        assert ts.get_conversion_for("milliseconds") == 3000.0
        with pytest.raises(KeyError):
            ts.get_unit(TimeUnit.samples)
        with pytest.raises(KeyError):
            ts.get_conversion_for("samples")


# endregion


# region Tests: Edge Cases and Performance


class TestTimestampEdgeCases:
    """Edge cases and error handling for timestamp generation."""

    def test_single_event_timeline(self) -> None:
        """Timeline with single event works correctly."""
        tl = Timeline(length=10, unit=TimeUnit.seconds, uid="single")
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 5.0,
                }
            ]
        )
        table = tl.get_timestamp_table()
        assert len(table) == 1
        assert table_column(table, "single") == [5.0]

    def test_overlapping_children(self) -> None:
        """Children with overlapping ranges work correctly."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent")

        child1 = Timeline(length=30, unit=TimeUnit.seconds, uid="overlap1")
        child2 = Timeline(length=30, unit=TimeUnit.seconds, uid="overlap2")

        # Overlapping: child1 [10, 40], child2 [20, 50]
        parent.add_child(child1, offset=10)
        parent.add_child(child2, offset=20)

        table = parent.get_timestamp_table([25.0])

        # At root=25:
        # child1 (offset=10): local=15, within [0,30] -> valid
        # child2 (offset=20): local=5, within [0,30] -> valid
        assert table_column(table, "overlap1") == [15.0]
        assert table_column(table, "overlap2") == [5.0]

    def test_coordinate_at_exact_boundary(self) -> None:
        """Coordinate exactly at timeline boundary is included."""
        tl = Timeline(length=10, unit=TimeUnit.seconds, uid="boundary")
        local = tl._compute_local_coordinates(
            coordinate_column([0.0, 10.0]), offset=0.0
        )
        # Exactly at 0 and length should be valid
        assert coordinate_values(local) == [0.0, 10.0]

    def test_large_hierarchy_performance(self) -> None:
        """Reasonable performance with many children."""
        parent = Timeline(length=1000, unit=TimeUnit.seconds, uid="big")

        # Add 100 children
        for i in range(100):
            child = Timeline(length=5, unit=TimeUnit.seconds, uid=f"child_{i}")
            child.add_events(
                [
                    {
                        "id": f"e_{i}",
                        "temporal_type": "instant",
                        "event_type": "Beat",
                        "instant": 2.5,
                    }
                ]
            )
            parent.add_child(child, offset=i * 10)

        # Should complete in reasonable time
        table = parent.get_timestamp_table()
        assert len(table.column_names) == 101  # big + 100 children


class TestFractionLengthTimestamps:
    """Timestamp generation on a quarters/beats timeline whose length is a Fraction.

    A logical (quarters/beats) timeline stores its length as a ``Coordinate``
    whose ``value`` is a ``Fraction``. The bounds check inside
    ``_compute_local_coordinates`` compares the local coordinate array against
    that length exactly, on the ratio side of the cells. These tests pin that
    both public timestamp entry points run on a Fraction-length timeline and
    keep the in-bounds coordinates exact.
    """

    def test_compute_local_coordinates_with_fraction_length(self) -> None:
        """The bounds check accepts a Fraction-valued length."""
        from timetoalign.timelines.types import ContinuousLogicalTimeline

        tl = ContinuousLogicalTimeline(length=Fraction(9, 2))
        assert isinstance(tl._length.value, Fraction)

        local = tl._compute_local_coordinates(
            coordinate_column([0.0, 2.0, 4.5, 5.0], NumberType.fraction), offset=0
        )
        # In bounds [0, 9/2]: 0.0, 2.0, 4.5 kept; 5.0 > 4.5 -> null.
        assert coordinate_values(local, NumberType.fraction) == [
            Fraction(0),
            Fraction(2),
            Fraction(9, 2),
            None,
        ]

    def test_get_timestamp_table_with_fraction_length(self) -> None:
        """get_timestamp_table() runs on a Fraction-length quarters timeline."""
        from timetoalign.timelines.types import ContinuousLogicalTimeline

        tl = ContinuousLogicalTimeline(length=Fraction(9, 2))
        tl.add_events(
            [
                {"start": Fraction(0)},
                {"start": Fraction(3, 2)},
                {"start": Fraction(3)},
            ]
        )

        table = tl.get_timestamp_table()
        assert table.num_rows == 3
        assert table_column(table, tl.id) == [Fraction(0), Fraction(3, 2), Fraction(3)]

    def test_dataframe_with_fraction_length(self) -> None:
        """The frame runs and returns exact Fraction coordinates."""
        from timetoalign.timelines.types import ContinuousLogicalTimeline

        tl = ContinuousLogicalTimeline(length=Fraction(9, 2))
        tl.add_events(
            [
                {"start": Fraction(0)},
                {"start": Fraction(3, 2)},
                {"start": Fraction(3)},
            ]
        )

        df = tl.get_timestamp_table(format="dataframe")
        assert df.shape == (3, 1)
        # Fraction number-type timeline -> coordinates auto-rendered as Fractions.
        assert list(df[df.columns[0]]) == [Fraction(0), Fraction(3, 2), Fraction(3)]


# endregion
