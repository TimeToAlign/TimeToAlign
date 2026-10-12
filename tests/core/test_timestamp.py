"""Tests for unified TimeStamp and TimeIntervalStamp.

These tests verify the unified timestamp architecture where Timeline and
TimelineGroup use the same coordinate resolution mechanism via InterpolationMaps.
"""

from __future__ import annotations

import pytest

from timetoalign.core import (
    Coordinate,
    IdCoordinate,
    Interval,
    TimeIntervalStamp,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.maps import IntervalToConstantMap, TableMap
from timetoalign.timelines import Timeline, TimelineGroup


class TestTimeStampBasics:
    """Basic TimeStamp functionality tests."""

    def test_get_timestamp_simple(self):
        """Get a timestamp at a specific coordinate."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        ts = tl.get_timestamp(50.0)

        assert ts.axis == IdCoordinate(50.0, TimeUnit.seconds, tl.id)
        assert ts.source_id == tl.id
        assert ts.is_interpolated is False

    def test_get_timestamp_from_coordinate(self):
        """Get timestamp from a Coordinate object."""
        from timetoalign.core import Coordinate

        tl = Timeline(length=100, unit=TimeUnit.seconds)
        coord = Coordinate(25.0, TimeUnit.seconds)
        ts = tl.get_timestamp(coord)

        assert ts.axis == IdCoordinate(25.0, TimeUnit.seconds, tl.id)

    def test_timestamp_source(self):
        """TimeStamp references its source timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="my_timeline")
        ts = tl.get_timestamp(50.0)

        assert ts.source is tl
        assert ts.source_id == "my_timeline"


class TestTimeStampWithChildren:
    """TimeStamp coordinate resolution with nested timelines."""

    def test_get_child_coordinate(self):
        """Get coordinate on a child timeline."""
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        # Child embedded at offset 20 in parent
        parent.add_child(child, offset=20)

        # Get timestamp at parent coordinate 30
        ts = parent.get_timestamp(30.0)

        # Parent 30 = Child 10 (30 - 20)
        assert ts.get_coordinate_for("child:1", format="float") == 10.0

    def test_get_child_coordinate_at_boundary(self):
        """Get child coordinate at exact boundaries.

        Child spans [offset, offset + length) on the parent, i.e. [25, 75).
        The right endpoint is *exclusive* per the TTA interval model.
        """
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=50, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=25)

        # At child start (parent 25) -- left-inclusive
        ts_start = parent.get_timestamp(25.0)
        assert ts_start.get_coordinate_for("child:1", format="float") == 0.0

        # At child end (parent 75) -- right-exclusive, so None
        ts_end = parent.get_timestamp(75.0)
        with pytest.raises(KeyError):
            ts_end.get_coordinate_for("child:1")

        # Just before child end (parent 74.999...) -- still inside
        ts_just_before = parent.get_timestamp(74.999)
        assert (
            ts_just_before.get_coordinate_for("child:1", format="float")
            == 49.998999999999995
        )

    def test_child_out_of_range_returns_none(self):
        """Coordinates outside child range return None.

        Child spans [30, 70) on the parent.  Parent coordinate 10 is
        outside that span, so typed retrieval raises ``KeyError`` rather than
        extrapolating a meaningless negative local coordinate.
        """
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=30)

        # Before child (parent 10) -- outside [30, 70)
        ts_before = parent.get_timestamp(10.0)
        with pytest.raises(KeyError):
            ts_before.get_coordinate_for("child:1")

    def test_multiple_children(self):
        """Timestamps work with multiple children.

        child1 spans [10, 40) and child2 spans [60, 90) on the parent.
        At parent coordinate 40, child1 is *excluded* (right-exclusive)
        and child2 has not started yet, so both return None.
        At parent coordinate 35, child1 returns 25.0 and child2 returns
        None.
        """
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child1 = Timeline(length=30, unit=TimeUnit.seconds, uid="child:1")
        child2 = Timeline(length=30, unit=TimeUnit.seconds, uid="child:2")

        parent.add_child(child1, offset=10)
        parent.add_child(child2, offset=60)

        # Parent 40 is at the right-exclusive boundary of child1 [10, 40)
        ts_boundary = parent.get_timestamp(40.0)
        with pytest.raises(KeyError):
            ts_boundary.get_coordinate_for("child:1")
        with pytest.raises(KeyError):
            ts_boundary.get_coordinate_for("child:2")

        # Parent 35 is inside child1 [10, 40): local = 35 - 10 = 25
        ts_inside = parent.get_timestamp(35.0)
        assert ts_inside.get_coordinate_for("child:1", format="float") == 25.0
        with pytest.raises(KeyError):
            ts_inside.get_coordinate_for("child:2")

        # Parent 70 is inside child2 [60, 90): local = 70 - 60 = 10
        ts_child2 = parent.get_timestamp(70.0)
        with pytest.raises(KeyError):
            ts_child2.get_coordinate_for("child:1")
        assert ts_child2.get_coordinate_for("child:2", format="float") == 10.0

    def test_to_dict(self):
        """TimeStamp can be materialized to dict."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        ts = parent.get_timestamp(30.0)
        result = ts.to_dict()

        assert result["parent:1"] == {
            "unit": "seconds",
            "number_type": "float",
            "value": 30.0,
        }
        assert result["child:1"] == {
            "unit": "seconds",
            "number_type": "float",
            "value": 10.0,
        }

    def test_present_timelines(self):
        """Get list of timelines with coordinates."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        ts = parent.get_timestamp(30.0)
        present = ts.present_timelines

        assert "parent:1" in present
        assert "child:1" in present


class TestTimeStampWithCMaps:
    """TimeStamp with C-Map unit conversion."""

    def test_unit_conversion(self):
        """Get coordinate converted to another unit."""
        tl = Timeline(length=960, unit=TimeUnit.ticks)

        # Add tempo map: ticks -> seconds
        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],  # 960 ticks = 2 seconds
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        ts = tl.get_timestamp(480)

        # 480 ticks = 1.0 seconds (linear TableMap interp is bit-exact here)
        assert ts.get_unit(TimeUnit.seconds, format="float") == 1.0

    def test_timestamp_from_unit(self):
        """Create timestamp specifying coordinate in different unit."""
        tl = Timeline(length=960, unit=TimeUnit.ticks)

        # Add tempo map: ticks -> seconds
        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        # Query at 1.5 seconds
        ts = tl.get_timestamp(1.5, unit=TimeUnit.seconds)

        # 1.5 seconds = 720 ticks (linear inverse is bit-exact here)
        assert ts.axis.value == 720

    def test_typed_unit_access(self):
        """Unit conversion uses the explicit typed getter."""
        tl = Timeline(length=960, unit=TimeUnit.ticks)

        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        ts = tl.get_timestamp(480)

        result = ts.get_unit(TimeUnit.seconds, format="float")
        assert result == 1.0

    def test_no_cmap_raises_keyerror(self):
        """get_unit raises KeyError when no C-Map is available."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        ts = tl.get_timestamp(50.0)

        with pytest.raises(KeyError):
            ts.get_unit(TimeUnit.pixels)

    def test_timestamp_from_unit_raises_without_cmap(self):
        """get_timestamp raises when unit specified but no C-Map."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)

        with pytest.raises(ValueError, match="No C-Map available"):
            tl.get_timestamp(50.0, unit=TimeUnit.pixels)

    def test_to_dict_and_unit_conversion_are_separate(self):
        """Serialization and unit conversion use separate typed surfaces."""
        tl = Timeline(length=960, unit=TimeUnit.ticks, uid="dlt_1")

        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        ts = tl.get_timestamp(480)
        result = ts.to_dict()

        assert result["dlt_1"] == {
            "unit": "ticks",
            "number_type": "int",
            "value": 480,
        }
        assert ts.get_unit(TimeUnit.seconds, format="float") == 1.0


class TestTimeIntervalStamp:
    """TimeIntervalStamp tests."""

    def test_create_interval_stamp(self):
        """Create an interval stamp."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        interval = tl.get_interval_stamp(10.0, 50.0)

        assert interval.start.axis.value == 10.0
        assert interval.end.axis.value == 50.0
        assert interval.duration.value == 40.0
        assert interval.present_timelines == [tl.id]

    def test_get_interval_for_child(self):
        """Get interval on child timeline."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=60, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        # Parent interval [30, 70] should be child interval [10, 50]
        interval = parent.get_interval_stamp(30.0, 70.0)

        child_interval = interval.get_interval("child:1")
        assert child_interval == Interval(
            start=Coordinate(10.0, TimeUnit.seconds),
            end=Coordinate(50.0, TimeUnit.seconds),
        )

    def test_get_duration_for_child(self):
        """Get duration on child timeline."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=80, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=10)

        interval = parent.get_interval_stamp(20.0, 60.0)

        # Child duration = 40 (same as parent, linear mapping)
        duration = interval.get_duration_for("child:1")
        assert duration.value == 40.0
        assert duration.timeline_id == "child:1"

    def test_get_intervals(self):
        """Retrieve all typed intervals across timelines."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child1 = Timeline(length=80, unit=TimeUnit.seconds, uid="child:1")
        child2 = Timeline(length=60, unit=TimeUnit.seconds, uid="child:2")

        parent.add_child(child1, offset=10)
        parent.add_child(child2, offset=30)

        interval = parent.get_interval_stamp(40.0, 80.0)
        intervals = interval.get_intervals()

        assert (intervals["parent:1"].start.value, intervals["parent:1"].end.value) == (
            40.0,
            80.0,
        )
        assert (intervals["child:1"].start.value, intervals["child:1"].end.value) == (
            30.0,
            70.0,
        )
        assert (intervals["child:2"].start.value, intervals["child:2"].end.value) == (
            10.0,
            50.0,
        )

    def test_interval_access(self):
        """Interval access returns a typed Interval."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=80, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=10)

        interval = parent.get_interval_stamp(20.0, 60.0)

        child_interval = interval.get_interval("child:1")
        assert (child_interval.start.value, child_interval.end.value) == (10.0, 50.0)

    def test_str_basic(self):
        """__str__ shows header and aligned start/end columns."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="cpt_1")
        child = Timeline(length=60, unit=TimeUnit.seconds, uid="child:1")
        parent.add_child(child, offset=20)

        # Both endpoints inside child [20, 80)
        interval = parent.get_interval_stamp(30.0, 70.0)
        text = str(interval)

        assert "TimeIntervalStamp [30, 70) seconds" in text
        assert "cpt_1" in text
        assert "child:1" in text

    def test_str_straddling_children(self):
        """__str__ shows '-' when an endpoint falls outside a child's span.

        child1 spans [0, 40), child2 spans [50, 90) on the parent.
        An interval [35, 55) has its start in child1 and its end in child2.
        """
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child1 = Timeline(length=40, unit=TimeUnit.seconds, uid="sec:A")
        child2 = Timeline(length=40, unit=TimeUnit.seconds, uid="sec:B")
        parent.add_child(child1, offset=0)
        parent.add_child(child2, offset=50)

        interval = parent.get_interval_stamp(35.0, 55.0)
        text = str(interval)

        # Parent row: both endpoints present
        assert "parent:1" in text
        # An Interval requires both endpoints on the same axis, so axes with
        # only one present endpoint are not stored.
        assert "sec:A" not in text
        assert "sec:B" not in text

    def test_str_omits_fully_out_of_range_children(self):
        """Children where both endpoints are out of range are omitted."""
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="cpt_1")
        child = Timeline(length=10, unit=TimeUnit.seconds, uid="far_child")
        parent.add_child(child, offset=80)

        # Interval [5, 15) - far_child is at [80, 90), both endpoints outside
        interval = parent.get_interval_stamp(5.0, 15.0)
        text = str(interval)

        assert "far_child" not in text


class TestTimeStampValidation:
    """Validation and error handling tests."""

    def test_interval_same_source(self):
        """Start and end must be from same source."""
        tl1 = Timeline(length=100, unit=TimeUnit.seconds, uid="tl1")

        ts1 = tl1.get_timestamp(10.0)

        with pytest.raises(ValueError, match="absent"):
            TimeIntervalStamp(
                intervals={
                    "tl1": Interval(
                        start=ts1.coordinates["tl1"], end=ts1.coordinates["tl1"]
                    )
                },
                source_id="tl2",
            )


class TestTimelineTimeStampSourceProtocol:
    """Verify Timeline implements TimeStampSource protocol."""

    def test_timeline_has_required_methods(self):
        """Timeline has all TimeStampSource protocol methods."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)

        # These methods are required by the protocol
        assert hasattr(tl, "_get_unit_map")
        assert hasattr(tl, "_get_related_timeline_ids")
        assert hasattr(tl, "_get_available_units")

    def test_get_related_timeline_ids(self):
        """_get_related_timeline_ids returns child IDs."""
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child1 = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")
        child2 = Timeline(length=40, unit=TimeUnit.seconds, uid="child:2")

        parent.add_child(child1, offset=0)
        parent.add_child(child2, offset=50)

        ids = parent._get_related_timeline_ids()
        assert "child:1" in ids
        assert "child:2" in ids

    def test_get_available_units(self):
        """_get_available_units returns C-Map target units."""
        tl = Timeline(length=960, unit=TimeUnit.ticks)

        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        units = tl._get_available_units()
        assert TimeUnit.seconds in units


class TestTimeStampWithUnits:
    """Tests for unit metadata and Coordinate access in TimeStamps."""

    def test_typed_axis_property(self):
        """axis returns the proper IdCoordinate object."""

        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")
        ts = tl.get_timestamp(50.0)

        coord = ts.axis
        assert isinstance(coord, IdCoordinate)
        assert coord.value == 50.0
        assert coord.unit == TimeUnit.seconds
        assert coord.timeline_id == "test:1"

    def test_get_coordinate_for_self(self):
        """get_coordinate returns Coordinate for source timeline."""
        from timetoalign.core import Coordinate

        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")
        ts = tl.get_timestamp(50.0)

        coord = ts.get_coordinate_for("test:1", format="coordinate")
        assert isinstance(coord, Coordinate)
        assert coord.value == 50.0
        assert coord.unit == TimeUnit.seconds

    def test_get_coordinate_for_child(self):
        """get_coordinate returns proper Coordinate for child."""
        from timetoalign.core import Coordinate

        # Children must share parent's unit (TTA model constraint)
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        ts = parent.get_timestamp(30.0)

        # Child coordinate at parent 30 is 10 (30 - 20)
        child_coord = ts.get_coordinate_for("child:1", format="coordinate")
        assert isinstance(child_coord, Coordinate)
        assert child_coord.value == 10.0
        assert child_coord.unit == TimeUnit.seconds  # Same unit as parent

    def test_get_coordinate_unknown_timeline_raises(self):
        """Typed coordinate retrieval raises for an unknown timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")
        ts = tl.get_timestamp(50.0)

        with pytest.raises(KeyError):
            ts.get_coordinate_for("unknown:1")

    def test_get_unit_for_timeline_self(self):
        """_get_unit_for_timeline returns own unit for source."""
        tl = Timeline(length=100, unit=TimeUnit.pixels, uid="test:1")

        unit = tl._get_unit_for_timeline("test:1")
        assert unit == TimeUnit.pixels

    def test_get_unit_for_timeline_child(self):
        """_get_unit_for_timeline returns child's unit (same as parent per TTA model)."""
        # Children must share parent's unit (TTA model constraint)
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        unit = parent._get_unit_for_timeline("child:1")
        assert unit == TimeUnit.seconds  # Same as parent

    def test_get_unit_for_timeline_unknown(self):
        """_get_unit_for_timeline returns None for unknown."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")

        unit = tl._get_unit_for_timeline("unknown:1")
        assert unit is None


class TestTimeIntervalStampWithUnits:
    """Tests for unit metadata in TimeIntervalStamp."""

    def test_get_interval(self):
        """get_interval returns an Interval."""

        # Children must share parent's unit (TTA model constraint)
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=80, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=10)

        interval = parent.get_interval_stamp(20.0, 60.0)

        # Child interval should be [10, 50] in seconds (same as parent)
        child_interval = interval.get_interval("child:1")
        start, end = child_interval.start, child_interval.end
        assert isinstance(start, Coordinate)
        assert isinstance(end, Coordinate)
        assert start.value == 10.0
        assert end.value == 50.0
        assert start.unit == TimeUnit.seconds
        assert end.unit == TimeUnit.seconds

    def test_get_interval_unknown(self):
        """get_interval raises KeyError for an unknown timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")
        interval = tl.get_interval_stamp(10.0, 50.0)

        with pytest.raises(KeyError):
            interval.get_interval("unknown:1")


class TestTimestampTableMetadata:
    """Tests for unit metadata in timestamp tables."""

    def test_timestamp_table_has_unit_metadata(self):
        """get_timestamp_table includes unit metadata on fields."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="test:1")
        table = tl.get_timestamp_table([0.0, 50.0, 100.0])

        # Every column names the timeline it belongs to; there is no
        # separate axis field duplicating this timeline's own column.
        assert table.column_names == ["test:1"]

        # Check timeline field metadata
        tl_field = table.schema.field("test:1")
        assert tl_field.metadata is not None
        assert field_metadata(tl_field)["unit"] == "seconds"

    def test_timestamp_table_child_metadata(self):
        """Child fields have unit metadata (same as parent per TTA model)."""
        # Children must share parent's unit (TTA model constraint)
        parent = Timeline(length=100, unit=TimeUnit.seconds, uid="parent:1")
        child = Timeline(length=40, unit=TimeUnit.seconds, uid="child:1")

        parent.add_child(child, offset=20)

        table = parent.get_timestamp_table([30.0, 50.0])

        # Check child field has its unit metadata
        child_field = table.schema.field("child:1")
        assert child_field.metadata is not None
        assert field_metadata(child_field)["unit"] == "seconds"  # Same as parent
        assert field_metadata(child_field)["timeline_id"] == "child:1"

    def test_timestamp_table_cmap_metadata(self):
        """C-Map fields have target unit metadata."""
        tl = Timeline(length=960, unit=TimeUnit.ticks, uid="test:1")

        tempo_map = TableMap(
            x_values=[0, 960],
            y_values=[0.0, 2.0],
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tempo_map)

        table = tl.get_timestamp_table([0.0, 480.0, 960.0], conversion_maps=[tempo_map])

        # Find the C-Map field by name (field names use cmap.name, not cmap.id)
        cmap_field = table.schema.field(tempo_map.name)
        assert cmap_field.metadata is not None
        assert field_metadata(cmap_field)["unit"] == "seconds"  # Target unit
        assert field_metadata(cmap_field)["cmap_id"] == tempo_map.id


class TestTimeStampReprHtml:
    """TimeStamp ``_repr_html_`` affordance footer."""

    def test_repr_html_renders_coordinate_table(self):
        """The coordinate cross-section table still renders."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="cpt_1")
        ts = tl.get_timestamp(50.0)
        html = ts._repr_html_()
        assert "<table" in html
        assert "TimeStamp" in html
        assert "cpt_1" in html

    def test_repr_html_try_footer(self):
        """A Try footer surfaces the real TimeStamp accessors after the table."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, uid="cpt_1")
        ts = tl.get_timestamp(50.0)
        html = ts._repr_html_()
        assert (
            "Try: <code>ts.get_coordinate_for(&lt;tl_id&gt;)</code>, "
            "<code>ts.get_unit(&lt;unit&gt;)</code>" in html
        )
        assert html.index("</table>") < html.index("Try:")


class TestTimeStampCrossSectionConversions:
    """Timestamps surface every C-Map across the whole subtree (Contract §4).

    A timestamp is a cross-section, so it must expose ALL attached conversion
    maps -- including maps with no ``target_unit`` (labels, structured values)
    and maps registered on any descendant -- not merely the ``TimeUnit``-keyed
    subset. Exact expected values only (Contract §12).
    """

    @staticmethod
    def _parent_with_label_child() -> tuple[Timeline, Timeline]:
        """Parent [0,40) with a child at offset 10 carrying a label map.

        Child coordinate ``x`` maps: ``[0,5) -> {"page": 1}``,
        ``[5,12) -> {"page": 2}``, ``[12,20) -> {"page": 3}``.
        """
        parent = Timeline(length=40, unit=TimeUnit.quarters, uid="p")
        child = Timeline(length=20, unit=TimeUnit.quarters, uid="c")
        parent.add_child(child, offset=10)
        child.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0, 5, 12],
                values=[{"page": 1}, {"page": 2}, {"page": 3}],
                source_unit=TimeUnit.quarters,
                name="pages",
            )
        )
        return parent, child

    def test_non_unit_map_on_source_surfaces(self):
        """A ``target_unit``-less map on the source surfaces everywhere."""
        tl = Timeline(length=40, unit=TimeUnit.quarters, uid="tl")
        tl.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0, 20],
                values=["intro", "main"],
                source_unit=TimeUnit.quarters,
                name="section",
            )
        )
        ts = tl.get_timestamp(25)  # [20, .) -> "main"

        assert ts.get_conversion_for("section") == "main"
        rendered = str(ts)
        assert "section" in rendered
        assert "main" in rendered

    def test_direct_child_map_surfaces_on_parent(self):
        """A label map on a direct child surfaces on the parent's timestamp."""
        parent, _child = self._parent_with_label_child()
        ts = parent.get_timestamp(17)  # 17 -> child 7 -> [5,12) -> {"page": 2}

        assert ts.get_conversion_for("pages") == {"page": 2}

    def test_grandchild_map_surfaces(self):
        """A label map on a grandchild surfaces via composed offset arithmetic."""
        parent, child = self._parent_with_label_child()
        grand = Timeline(length=8, unit=TimeUnit.quarters, uid="g")
        child.add_child(grand, offset=3)
        grand.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0, 4],
                values=["A", "B"],
                source_unit=TimeUnit.quarters,
                name="accolade",
            )
        )
        ts = parent.get_timestamp(17)  # 17 -> child 7 -> grand 4 -> [4, .) -> "B"

        assert ts.get_coordinate_for(grand.id, format="float") == 4.0
        assert ts.get_conversion_for("accolade") == "B"

    def test_descendant_unit_map_surfaces_via_get_unit(self):
        """A ``TimeUnit`` map on a child surfaces through the parent's get_unit."""
        parent = Timeline(length=40, unit=TimeUnit.quarters, uid="p2")
        child = Timeline(length=20, unit=TimeUnit.quarters, uid="c2")
        parent.add_child(child, offset=10)
        child.add_conversion_map(
            TableMap(
                x_values=[0, 20],
                y_values=[0.0, 10.0],  # child 10 -> 5.0 seconds
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
            )
        )
        ts = parent.get_timestamp(20)  # child coordinate 10

        assert TimeUnit.seconds in parent._get_available_units()
        assert ts.get_unit(TimeUnit.seconds, format="float") == 5.0

    def test_non_numeric_value_not_coerced(self):
        """Structured/label outputs render as themselves, never through float."""
        parent, _child = self._parent_with_label_child()
        ts = parent.get_timestamp(11)  # child 1 -> [0,5) -> {"page": 1}

        assert ts.get_conversion_for("pages") == {"page": 1}
        assert "{'page': 1}" in str(ts)

    def test_collision_qualified_by_owner(self):
        """Two children present at one axis, same label, are qualified by owner."""
        parent = Timeline(length=40, unit=TimeUnit.quarters, uid="root")
        left = Timeline(length=20, unit=TimeUnit.quarters, uid="left")
        right = Timeline(length=20, unit=TimeUnit.quarters, uid="right")
        # Overlapping spans: left [0,20), right [5,25); axis 10 is inside both.
        parent.add_child(left, offset=0)
        parent.add_child(right, offset=5)
        for child in (left, right):
            child.add_conversion_map(
                IntervalToConstantMap(
                    boundaries=[0],
                    values=[child.id],
                    source_unit=TimeUnit.quarters,
                    name="tag",
                )
            )
        ts = parent.get_timestamp(10)  # both children present -> label collision
        rendered = str(ts)
        assert "left:tag" in rendered
        assert "right:tag" in rendered
        assert ts.get_conversion_for("tag") == "left"  # first owner in subtree order

    def test_conversion_maps_false_suppresses(self):
        """conversion_maps=False surfaces no conversions at all."""
        tl = Timeline(length=40, unit=TimeUnit.quarters, uid="tl")
        tl.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0],
                values=["x"],
                source_unit=TimeUnit.quarters,
                name="lbl",
            )
        )
        ts = tl.get_timestamp(10, conversion_maps=False)

        with pytest.raises(KeyError):
            ts.get_conversion_for("lbl")
        assert set(ts.to_dict()) == {"tl"}

    def test_selector_list_surfaces_only_matching_maps(self):
        """conversion_maps=[selector] surfaces only the maps it names."""
        tl = Timeline(length=40, unit=TimeUnit.quarters, uid="tl")
        tl.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0],
                values=["shown"],
                source_unit=TimeUnit.quarters,
                name="wanted",
            )
        )
        tl.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0],
                values=["hidden"],
                source_unit=TimeUnit.quarters,
                name="other",
            )
        )
        ts = tl.get_timestamp(10, conversion_maps=["wanted"])

        assert ts.get_conversion_for("wanted") == "shown"
        with pytest.raises(KeyError):
            ts.get_conversion_for("other")

    def test_unknown_selector_raises_keyerror(self):
        """An unknown conversion selector raises KeyError."""
        tl = Timeline(length=40, unit=TimeUnit.quarters, uid="tl")
        ts = tl.get_timestamp(10)

        with pytest.raises(KeyError):
            ts.get_conversion_for("does_not_exist")

    def test_group_member_map_surfaces(self):
        """A label map on a group member surfaces on the group's timestamp."""
        audio = Timeline(length=100, unit=TimeUnit.seconds, uid="audio")
        score = Timeline(length=40, unit=TimeUnit.quarters, uid="score")
        score.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0, 20],
                values=["intro", "main"],
                source_unit=TimeUnit.quarters,
                name="section",
            )
        )
        group = TimelineGroup(name="g", timelines=[audio, score])
        ts = group.get_timestamp_at(25, "score")  # score 25 -> "main"

        assert ts.get_conversion_for("section") == "main"

    def test_group_member_descendant_map_surfaces(self):
        """A map on a member's child surfaces on the group's timestamp."""
        audio = Timeline(length=100, unit=TimeUnit.seconds, uid="audio2")
        score = Timeline(length=40, unit=TimeUnit.quarters, uid="score2")
        phrase = Timeline(length=10, unit=TimeUnit.quarters, uid="phrase2")
        score.add_child(phrase, offset=15)
        phrase.add_conversion_map(
            IntervalToConstantMap(
                boundaries=[0, 5],
                values=["x", "y"],
                source_unit=TimeUnit.quarters,
                name="phraselabel",
            )
        )
        group = TimelineGroup(name="g", timelines=[audio, score])
        ts = group.get_timestamp_at(
            22, "score2"
        )  # score 22 -> phrase 7 -> [5,.) -> "y"

        assert ts.get_conversion_for("phraselabel") == "y"
