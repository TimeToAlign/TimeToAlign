"""Tests for Timeline Relationship Concepts: Region, SegmentLine, derive().

This module tests the relationship concepts that distinguish between:
- **Region**: A named TimeInterval (NOT a timeline)
- **Child**: A timeline nested in a parent (same unit)
- **Segment**: A Child that is contiguous with siblings
- **SegmentLine**: A parent where ALL children are Segments
- **Derivative**: A new timeline created via C-map conversion (different unit)

A Region is a named part of a timeline that is defined by a TimeInterval.
Regions are useful for referring to parts of a timeline by name.

When all Children of the same parent timeline (siblings) are contiguous
with each other, they are Segments and the parent is a SegmentLine.

A ConversionMap implies the presence of a derived timeline in the target unit.

Validity Rationale:
    These tests verify:
    1. Region is an immutable dataclass (NOT a timeline)
    2. Regions can be used to create Children via create_child_from_region()
    3. SegmentLine enforces contiguity of children
    4. derive() creates timelines in different units via C-maps
    5. Roundtrip conversions work correctly (inverse C-maps)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest

from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.core.time import Duration, Interval
from timetoalign.maps import LinearMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    Region,
    SegmentLine,
    Timeline,
    get_timeline_class,
)

# region Region Tests


class TestRegionDataclass:
    """Test the Region dataclass itself (not attached to a timeline)."""

    def test_region_creation(self):
        """Region can be created with name, start, end."""
        start = Coordinate(10.0, TimeUnit.seconds)
        end = Coordinate(30.0, TimeUnit.seconds)
        region = Region(name="Chorus", start=start, end=end)

        assert region.name == "Chorus"
        assert region.start.value == 10.0
        assert region.end.value == 30.0
        assert region.unit == TimeUnit.seconds

    def test_region_duration(self):
        """Region.duration is a typed Duration of end - start."""
        start = Coordinate(10.0, TimeUnit.seconds)
        end = Coordinate(30.0, TimeUnit.seconds)
        region = Region(name="Test", start=start, end=end)

        assert region.duration == Duration(20.0, TimeUnit.seconds)

    def test_region_duration_keeps_an_exact_extent_exact(self):
        """A fraction-canonical region reports its exact length, not a decimal."""
        region = Region(
            name="Test",
            start=Coordinate(Fraction(1, 3), TimeUnit.quarters),
            end=Coordinate(Fraction(2, 3), TimeUnit.quarters),
        )

        assert region.duration == Duration(Fraction(1, 3), TimeUnit.quarters)
        assert region.duration.value == Fraction(1, 3)

    def test_region_as_interval(self):
        """Region.as_interval returns a typed Interval over the endpoints."""
        start = Coordinate(5.0, TimeUnit.seconds)
        end = Coordinate(15.0, TimeUnit.seconds)
        region = Region(name="Test", start=start, end=end)

        assert region.as_interval == Interval(start=start, end=end)
        assert region.as_interval.start == start
        assert region.as_interval.end == end

    def test_region_contains_left_inclusive(self):
        """Region.contains follows [start, end) convention."""
        start = Coordinate(10.0, TimeUnit.seconds)
        end = Coordinate(20.0, TimeUnit.seconds)
        region = Region(name="Test", start=start, end=end)

        # Start is included
        assert region.contains(10.0) is True
        # Middle is included
        assert region.contains(15.0) is True
        # End is NOT included (right-exclusive)
        assert region.contains(20.0) is False
        # Before start is not included
        assert region.contains(9.99) is False

    def test_region_contains_rejects_a_coordinate_in_another_unit(self):
        """A coordinate object must name the region's unit; raw numbers do not."""
        region = Region(
            name="Test",
            start=Coordinate(10.0, TimeUnit.seconds),
            end=Coordinate(20.0, TimeUnit.seconds),
        )

        with pytest.raises(
            ValueError,
            match="Coordinate unit quarters does not match region unit seconds",
        ):
            region.contains(Coordinate(15, TimeUnit.quarters))
        assert region.contains(Coordinate(15.0, TimeUnit.seconds)) is True
        assert region.contains(15.0) is True

    def test_region_overlaps(self):
        """Region.overlaps detects overlapping regions."""
        r1 = Region(
            name="A",
            start=Coordinate(0.0, TimeUnit.seconds),
            end=Coordinate(10.0, TimeUnit.seconds),
        )
        r2 = Region(
            name="B",
            start=Coordinate(5.0, TimeUnit.seconds),
            end=Coordinate(15.0, TimeUnit.seconds),
        )
        r3 = Region(
            name="C",
            start=Coordinate(10.0, TimeUnit.seconds),
            end=Coordinate(20.0, TimeUnit.seconds),
        )
        r4 = Region(
            name="D",
            start=Coordinate(15.0, TimeUnit.seconds),
            end=Coordinate(25.0, TimeUnit.seconds),
        )

        # Overlapping
        assert r1.overlaps(r2) is True
        assert r2.overlaps(r1) is True

        # Adjacent (not overlapping - [0,10) and [10,20) don't share coords)
        assert r1.overlaps(r3) is False
        assert r3.overlaps(r1) is False

        # Non-overlapping
        assert r1.overlaps(r4) is False
        assert r4.overlaps(r1) is False

    def test_region_rejects_end_before_start(self):
        """Region rejects end < start in __post_init__."""
        start = Coordinate(20.0, TimeUnit.seconds)
        end = Coordinate(10.0, TimeUnit.seconds)

        with pytest.raises(ValueError, match="end.*cannot be before"):
            Region(name="Invalid", start=start, end=end)

    def test_region_rejects_mismatched_units(self):
        """Region rejects start/end with different units."""
        start = Coordinate(10.0, TimeUnit.seconds)
        end = Coordinate(20.0, TimeUnit.quarters)  # Different unit!

        with pytest.raises(ValueError, match="unit.*must match"):
            Region(name="Invalid", start=start, end=end)

    def test_region_with_metadata(self):
        """Region can store metadata."""
        start = Coordinate(0.0, TimeUnit.seconds)
        end = Coordinate(10.0, TimeUnit.seconds)
        region = Region(
            name="Verse",
            start=start,
            end=end,
            meta={"repeat": 2, "label": "verse_1"},
        )

        assert region.meta["repeat"] == 2
        assert region.meta["label"] == "verse_1"

    def test_region_is_frozen(self):
        """Region is immutable (frozen dataclass)."""
        start = Coordinate(0.0, TimeUnit.seconds)
        end = Coordinate(10.0, TimeUnit.seconds)
        region = Region(name="Test", start=start, end=end)

        with pytest.raises(FrozenInstanceError):
            region.name = "Changed"


# endregion


# region Timeline Region Management Tests


class TestTimelineRegionManagement:
    """Test add_region, get_region, iter_regions on Timeline."""

    def test_add_region_returns_region_object(self):
        """add_region returns the created Region object."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        region = tl.create_region(30, 60, id="Chorus", name="Chorus")

        assert isinstance(region, Region)
        assert region.name == "Chorus"
        assert region.duration == Duration(30.0, TimeUnit.seconds)

    def test_add_region_stores_region(self):
        """Added regions are stored and retrievable."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 30, id="Verse", name="Verse")
        tl.create_region(30, 60, id="Chorus", name="Chorus")

        assert tl.n_regions == 2
        assert tl.has_region("Verse") is True
        assert tl.has_region("Chorus") is True
        assert tl.has_region("Bridge") is False

    def test_get_region_returns_region_object(self):
        """get_region returns the Region instance directly."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(10, 20, id="Test", name="Test")

        result = tl.get_region("Test")
        assert isinstance(result, Region)
        assert result.name == "Test"
        assert result.start.value == 10.0
        assert result.end.value == 20.0

    def test_get_region_nonexistent_raises_keyerror(self):
        """get_region raises KeyError for nonexistent region."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        with pytest.raises(KeyError, match="No region with id"):
            tl.get_region("NonExistent")

    def test_iter_regions(self):
        """iter_regions yields all Region objects."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 25, id="A", name="A")
        tl.create_region(25, 50, id="B", name="B")
        tl.create_region(50, 75, id="C", name="C")

        regions = list(tl.iter_regions())
        assert len(regions) == 3
        names = {r.name for r in regions}
        assert names == {"A", "B", "C"}

    def test_list_regions(self):
        """list_regions returns region ids."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 50, id="X", name="X")
        tl.create_region(50, 100, id="Y", name="Y")

        names = tl.list_regions()
        assert set(names) == {"X", "Y"}

    def test_add_region_rejects_duplicate_given_id(self):
        """A given region id already held raises; a repeated name does not."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(30, 60, id="Chorus", name="Chorus")

        with pytest.raises(ValueError, match="already held by region 'Chorus'"):
            tl.create_region(70, 90, id="Chorus", name="Chorus")
        assert tl.create_region(70, 90, name="Chorus").id == "region_1"

    def test_add_region_on_locked_timeline_succeeds(self):
        """add_region works on a locked timeline; regions are pure annotations.

        Length-locking guards a timeline's own coordinate contract with its
        parent (``set length``), not the addition of named annotations within
        the existing coordinate space. Creating a region changes nothing
        structural, so it must succeed on a locked timeline.
        """
        tl = Timeline(length=100, unit=TimeUnit.seconds, locked=True)

        region = tl.create_region(0, 10, id="Test", name="Test")

        assert region.name == "Test"
        assert float(region.start.value) == 0.0
        assert float(region.end.value) == 10.0
        assert tl.get_region("Test") is region

    def test_create_region_on_locked_timeline_succeeds(self):
        """create_region works on a locked timeline with exact bounds."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, locked=True)

        region = tl.create_region(0, 40, id="A", name="A")

        assert region.name == "A"
        assert float(region.start.value) == 0.0
        assert float(region.end.value) == 40.0
        assert tl.get_region("A") is region

    def test_create_regions_from_boundaries_on_locked_timeline_succeeds(self):
        """create_regions_from_boundaries works on a locked timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, locked=True)

        regions = tl.create_regions_from_boundaries([0, 40, 100], prefix="sec")

        assert [r.name for r in regions] == ["sec_1", "sec_2"]
        assert [(float(r.start.value), float(r.end.value)) for r in regions] == [
            (0.0, 40.0),
            (40.0, 100.0),
        ]

    def test_create_regions_by_splitting_on_locked_timeline_succeeds(self):
        """create_regions_by_splitting works on a locked timeline.

        With no split-point events, the whole timeline becomes one region.
        """
        tl = Timeline(length=100, unit=TimeUnit.seconds, locked=True)

        regions = tl.create_regions_by_splitting("breaks", prefix="mov")

        assert [r.name for r in regions] == ["mov_1"]
        assert (float(regions[0].start.value), float(regions[0].end.value)) == (
            0.0,
            100.0,
        )


# endregion


# region create_child_from_region Tests


class TestCreateChildFromRegion:
    """Test Timeline.create_child_from_region() - creating Children from Regions."""

    def test_create_child_from_region_creates_child(self):
        """create_child_from_region creates a child at the region's offset."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(30, 60, id="Chorus", name="Chorus")

        child = tl.create_child_from_region("Chorus")

        assert tl.n_children == 1
        assert child.length.value == 30.0  # 60 - 30
        assert child.name == "Chorus"

    def test_create_child_from_region_child_offset(self):
        """Created child has correct offset in parent."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(20, 40, id="Verse", name="Verse")

        child = tl.create_child_from_region("Verse")

        offset = tl.get_child_offset(child.id)
        assert offset.value == 20.0

    def test_create_child_from_region_copies_events_by_default(self):
        """create_child_from_region copies events within the region."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 10.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 35.0,
                },
                {
                    "id": "e3",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 55.0,
                },
                {
                    "id": "e4",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 80.0,
                },
            ]
        )
        tl.create_region(30, 60, id="Middle", name="Middle")

        child = tl.create_child_from_region("Middle")

        # Child should have 2 events (e2 at 35 -> 5, e3 at 55 -> 25)
        assert child.n_events == 2

    def test_create_child_from_region_adjusts_event_coordinates(self):
        """Copied events have coordinates relative to child's origin."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 45.0,
                },
            ]
        )
        tl.create_region(40, 60, id="Section", name="Section")

        child = tl.create_child_from_region("Section")

        # Event at 45 in parent should be at 5 in child (45 - 40)
        events = list(child.events)
        assert len(events) == 1
        event = events[0]
        start_coord = event.get("start")
        if isinstance(start_coord, dict) and "value" in start_coord:
            assert start_coord["value"] == 5.0
        elif start_coord is not None:
            assert float(start_coord) == 5.0
        else:
            event_instant = event.get("instant")
            if isinstance(event_instant, dict) and "value" in event_instant:
                assert event_instant["value"] == 5.0
            else:
                assert float(event_instant) == 5.0

    def test_create_child_from_region_copy_events_false(self):
        """create_child_from_region with copy_events=False creates empty child."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 35.0,
                },
            ]
        )
        tl.create_region(30, 60, id="Section", name="Section")

        child = tl.create_child_from_region("Section", copy_events=False)

        assert child.n_events == 0

    def test_create_child_from_region_nonexistent_raises(self):
        """create_child_from_region raises KeyError for nonexistent region."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)

        with pytest.raises(KeyError, match="not found"):
            tl.create_child_from_region("NonExistent")

    def test_create_child_from_region_locked_timeline_raises(self):
        """create_child_from_region raises RuntimeError on locked timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 10, id="Test", name="Test")
        tl._locked = True

        with pytest.raises(RuntimeError, match="locked"):
            tl.create_child_from_region("Test")


# endregion


# region SegmentLine Tests


class TestSegmentLineBasics:
    """Test SegmentLine basic functionality."""

    def test_segmentline_creation(self):
        """SegmentLine can be created empty."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        assert sl.n_segments == 0
        assert sl.length.value == 0

    def test_append_segment_adds_contiguous_children(self):
        """append_segment adds children contiguously."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        s1 = ContinuousLogicalTimeline(length=4)
        s2 = ContinuousLogicalTimeline(length=4)
        s3 = ContinuousLogicalTimeline(length=4)

        sl.append_segment(s1, name="m1")
        sl.append_segment(s2, name="m2")
        sl.append_segment(s3, name="m3")

        assert sl.n_segments == 3
        assert sl.length.value == 12  # 4 + 4 + 4

    def test_segment_offsets_are_contiguous(self):
        """Segment offsets form a contiguous sequence."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        s1 = ContinuousLogicalTimeline(length=4)
        s2 = ContinuousLogicalTimeline(length=3)
        s3 = ContinuousLogicalTimeline(length=5)

        sl.append_segment(s1)
        sl.append_segment(s2)
        sl.append_segment(s3)

        offsets = [offset.value for _, offset, _ in sl.iter_segments()]
        assert offsets == [0, 4, 7]  # 0, 0+4, 4+3

    def test_add_child_with_wrong_offset_raises(self):
        """add_child with non-contiguous offset raises ValueError."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        s1 = ContinuousLogicalTimeline(length=4)
        sl.append_segment(s1)

        s2 = ContinuousLogicalTimeline(length=4)

        # Offset 5 is wrong - should be 4 (immediately after s1)
        with pytest.raises(ValueError, match="contiguous"):
            sl.add_child(s2, offset=5)

    def test_first_segment_must_start_at_zero(self):
        """First segment must have offset 0."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        s1 = ContinuousLogicalTimeline(length=4)

        # First segment at offset 5 is invalid
        with pytest.raises(ValueError, match="contiguous"):
            sl.add_child(s1, offset=5)

    def test_get_segment_by_index(self):
        """get_segment_by_index retrieves segments by 0-based index."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        s1 = ContinuousLogicalTimeline(length=4, name="first")
        s2 = ContinuousLogicalTimeline(length=4, name="second")
        s3 = ContinuousLogicalTimeline(length=4, name="third")

        sl.append_segment(s1)
        sl.append_segment(s2)
        sl.append_segment(s3)

        offset, segment = sl.get_segment_by_index(1)
        assert segment.name == "second"
        assert offset.value == 4

    def test_get_segment_by_index_out_of_range(self):
        """get_segment_by_index raises IndexError for invalid index."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        sl.append_segment(ContinuousLogicalTimeline(length=4))

        with pytest.raises(IndexError):
            sl.get_segment_by_index(5)

    def test_get_segment_at_coordinate(self):
        """get_segment_at finds segment containing a coordinate."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        s1 = ContinuousLogicalTimeline(length=4, name="m1")
        s2 = ContinuousLogicalTimeline(length=4, name="m2")
        s3 = ContinuousLogicalTimeline(length=4, name="m3")

        sl.append_segment(s1)
        sl.append_segment(s2)
        sl.append_segment(s3)

        # Coordinate 6 is in segment 2 (offset 4-8)
        idx, segment, _ = sl.get_segment_at(6.0)
        assert idx == 1
        assert segment.name == "m2"


class TestSegmentLineFromSegmentation:
    """Test SegmentLine.from_segmentation() factory method."""

    def test_segmentation_creates_correct_segments(self):
        """from_segmentation splits timeline at given coordinates."""
        source = ContinuousLogicalTimeline(length=100)
        sl = SegmentLine.from_segmentation(source, [0, 25, 50, 75, 100])

        assert sl.n_segments == 4
        assert sl.length.value == 100

    def test_segmentation_segment_lengths(self):
        """Segmented segments have correct lengths."""
        source = ContinuousLogicalTimeline(length=100)
        sl = SegmentLine.from_segmentation(source, [0, 30, 70, 100])

        lengths = [seg.length.value for _, _, seg in sl.iter_segments()]
        assert lengths == [30, 40, 30]  # 30-0, 70-30, 100-70

    def test_segmentation_copies_events(self):
        """from_segmentation copies events to respective segments."""
        source = ContinuousLogicalTimeline(length=100)
        source.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 10.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 35.0,
                },
                {
                    "id": "e3",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 60.0,
                },
                {
                    "id": "e4",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 90.0,
                },
            ]
        )

        sl = SegmentLine.from_segmentation(source, [0, 25, 50, 75, 100])

        # Each segment should have 1 event
        for _, _, seg in sl.iter_segments():
            assert seg.n_events == 1

    def test_segmentation_requires_at_least_two_coords(self):
        """from_segmentation requires at least 2 split coordinates."""
        source = ContinuousLogicalTimeline(length=100)

        with pytest.raises(ValueError, match="at least 2"):
            SegmentLine.from_segmentation(source, [50])

    def test_segmentation_fails_if_source_has_children(self):
        """from_segmentation fails if source already has children."""
        source = ContinuousLogicalTimeline(length=100)
        child = ContinuousLogicalTimeline(length=10)
        source.add_child(child, offset=0)

        with pytest.raises(ValueError, match="already has children"):
            SegmentLine.from_segmentation(source, [0, 50, 100])


class TestSegmentLineParameterized:
    """Test parameterized SegmentLine (segment_type tracking and enforcement)."""

    def test_explicit_segment_type_at_construction(self):
        """SegmentLine parameter records its enforced segment class."""
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
        )
        assert sl.segment_type is ContinuousLogicalTimeline

    def test_segment_type_inferred_from_first_append(self):
        """segment_type is inferred from the first appended segment."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        assert sl.segment_type is None

        sl.append_segment(ContinuousLogicalTimeline(length=4))
        assert sl.segment_type is ContinuousLogicalTimeline

    def test_segment_type_enforced_on_add(self):
        """Adding a segment of wrong type raises TypeError."""
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
        )
        wrong_type = Timeline(length=4, unit=TimeUnit.quarters)

        with pytest.raises(TypeError, match="expects segments of type"):
            sl.append_segment(wrong_type)

    def test_segment_type_enforced_after_inference(self):
        """After inferring type from first segment, subsequent must match."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        # First segment sets the type
        sl.append_segment(ContinuousLogicalTimeline(length=4))
        assert sl.segment_type is ContinuousLogicalTimeline

        # Base Timeline is NOT ContinuousLogicalTimeline -- should reject
        with pytest.raises(TypeError, match="expects segments of type"):
            sl.append_segment(Timeline(length=4, unit=TimeUnit.quarters))

    def test_segment_type_accepts_subclass(self):
        """segment_type enforcement allows subclasses (isinstance check)."""

        class AnnotatedLogicalTimeline(ContinuousLogicalTimeline):
            """A subclass of the declared segment type."""

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
        )
        sl.append_segment(AnnotatedLogicalTimeline(length=4))
        assert sl.n_segments == 1

    def test_from_segmentation_sets_segment_type(self):
        """from_segmentation sets segment_type to source's class."""
        source = ContinuousLogicalTimeline(length=100)
        sl = SegmentLine.from_segmentation(source, [0, 50, 100])
        assert sl.segment_type is ContinuousLogicalTimeline

    def test_from_segmentation_segments_are_correct_type(self):
        """from_segmentation creates segments of the source's class."""
        source = ContinuousPhysicalTimeline(length=60.0)
        sl = SegmentLine.from_segmentation(source, [0.0, 30.0, 60.0])
        assert sl.segment_type is ContinuousPhysicalTimeline
        for _, _, seg in sl.iter_segments():
            assert type(seg) is ContinuousPhysicalTimeline

    def test_create_segment_line_sets_segment_type(self):
        """create_segment_line on typed timeline sets segment_type."""
        tl = ContinuousPhysicalTimeline(length=60.0)
        sl = tl.create_segment_line([0.0, 30.0, 60.0])
        assert sl.segment_type is ContinuousPhysicalTimeline

    def test_segment_type_none_for_no_segments(self):
        """Unparameterized SegmentLine with no segments has None segment_type."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        assert sl.segment_type is None

    # -- Recursive SegmentLine parameterization --

    def test_explicit_nested_parameter(self):
        """Nested parameterization records the complete inner dynamic class."""
        inner_class = SegmentLine[DiscreteGraphicalTimeline]
        sl = SegmentLine[inner_class](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        assert sl.segment_type is inner_class
        assert isinstance(sl, inner_class)

    def test_bare_line_infers_dynamic_child_class(self):
        """A bare line infers the exact dynamic class of its first child."""
        parent = SegmentLine(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        page = SegmentLine[DiscreteGraphicalTimeline](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        page.append_segment(DiscreteGraphicalTimeline(length=100))
        parent.append_segment(page, name="page_0")

        assert parent.segment_type is SegmentLine[DiscreteGraphicalTimeline]

    def test_nested_parameter_rejects_mismatched_child(self):
        """Differently-parameterized SegmentLine children are rejected."""

        class InferredAlternateGraphicalTimeline(DiscreteGraphicalTimeline):
            pass

        inner_class = SegmentLine[DiscreteGraphicalTimeline]
        parent = SegmentLine[inner_class](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        good_page = inner_class(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        good_page.append_segment(DiscreteGraphicalTimeline(length=100))
        parent.append_segment(good_page, name="page_0")

        bad_page = SegmentLine[InferredAlternateGraphicalTimeline](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        with pytest.raises(TypeError, match="expects segments of type"):
            parent.append_segment(bad_page, name="page_1")

    def test_bare_inference_rejects_mismatched_dynamic_child(self):
        """After bare inference, a different dynamic child class is rejected."""

        class AlternateGraphicalTimeline(DiscreteGraphicalTimeline):
            pass

        parent = SegmentLine(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        good_page = SegmentLine[DiscreteGraphicalTimeline](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        good_page.append_segment(DiscreteGraphicalTimeline(length=100))
        parent.append_segment(good_page, name="page_0")

        bad_page = SegmentLine[AlternateGraphicalTimeline](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        with pytest.raises(TypeError, match="expects segments of type"):
            parent.append_segment(bad_page, name="page_1")

    def test_unparameterized_child_segmentline_rejected_when_inner_type_set(self):
        """A bare SegmentLine is not an instance of a nested parameter class."""
        parent = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        unparameterized = SegmentLine(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        with pytest.raises(TypeError, match="expects segments of type"):
            parent.append_segment(unparameterized, name="page_0")

    def test_class_name_recursive_display(self):
        """class_name shows SegmentLine[SegmentLine[DGT]] for nested types."""
        sl = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        assert sl.class_name == "SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]"

    def test_class_name_recursive_after_inference(self):
        """class_name updates to recursive form after inferring inner type."""
        parent = SegmentLine(
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        assert parent.class_name == "SegmentLine"

        page = SegmentLine[DiscreteGraphicalTimeline](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        page.append_segment(DiscreteGraphicalTimeline(length=100))
        parent.append_segment(page, name="page_0")

        assert (
            parent.class_name == "SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]"
        )

    # -- Defaults inherited from the segment type --

    def test_defaults_from_continuous_logical_segment_type(self):
        """segment_type supplies unit and number_type when both are omitted."""
        sl = SegmentLine[ContinuousLogicalTimeline]()

        assert sl.unit == TimeUnit.quarters
        assert sl.number_type == NumberType.fraction

    def test_defaults_from_discrete_logical_segment_type(self):
        """A discrete logical segment type yields its own default unit."""
        sl = SegmentLine[DiscreteLogicalTimeline]()

        assert sl.unit == TimeUnit.ticks
        assert sl.number_type == NumberType.int

    def test_defaults_from_nested_parameter(self):
        """A nested SegmentLine inherits defaults through its parameter class."""
        sl = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]()

        assert sl.unit == TimeUnit.pixels
        assert sl.number_type == NumberType.int

    def test_explicit_unit_overrides_segment_type_default(self):
        """An explicitly passed unit is never overridden by the segment type."""
        sl = SegmentLine[ContinuousLogicalTimeline](
            unit=TimeUnit.whole_note,
        )

        assert sl.unit == TimeUnit.whole_note

    def test_unparameterized_segment_line_keeps_base_defaults(self):
        """Without segment_type, the base Timeline defaults still apply."""
        sl = SegmentLine()

        assert sl.segment_type is None
        assert sl.unit == TimeUnit.seconds
        assert sl.number_type == NumberType.float

    def test_class_name_non_segmentline_type_unchanged(self):
        """class_name for non-SegmentLine segment_type is unaffected."""
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
        )
        assert sl.class_name == "SegmentLine[ContinuousLogicalTimeline]"

    def test_repr_includes_recursive_type(self):
        """__repr__ includes the full recursive parameterization."""
        sl = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
            length=0,
            unit=TimeUnit.pixels,
            number_type=NumberType.int,
        )
        r = repr(sl)
        assert r.startswith("SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]")


class TestSegmentLineCasting:
    """Test structural predicates and casts between timelines and segment lines."""

    def test_is_segment_line_exact_empty_and_zero_length_semantics(self):
        """Coverage uses exact arithmetic; empty differs from one zero child."""
        empty = Timeline(length=0.0, unit=TimeUnit.seconds)
        assert empty.is_segment_line() is False

        zero_child = Timeline(length=0.0, unit=TimeUnit.seconds, uid="zero")
        empty.add_child(zero_child, offset=0.0)
        assert empty.is_segment_line() is True

        exact = Timeline(
            length=Fraction(3, 10),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        exact.add_child(
            Timeline(
                length=Fraction(1, 10),
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                uid="first",
            ),
            offset=Fraction(0),
        )
        exact.add_child(
            Timeline(
                length=Fraction(1, 5),
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                uid="second",
            ),
            offset=Fraction(1, 10),
        )
        assert exact.is_segment_line() is True

    def test_as_segment_line_preserves_content_and_copies_the_children(self):
        """A valid homogeneous hierarchy casts to its dynamic line class."""
        parent = Timeline(
            length=4.0,
            unit=TimeUnit.seconds,
            uid="parent",
            name="Parent",
            meta={"source": "fixture"},
        )
        parent.add_events([{"id": "marker", "event_type": "Marker", "instant": 1.0}])
        parent.create_region(0.0, 4.0, id="whole", name="whole")
        cmap = LinearMap(
            scalar=1000.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="milliseconds",
        )
        parent.add_conversion_map(cmap)
        flow_marker = object()
        parent._flow_maps["flow"] = flow_marker
        first = ContinuousPhysicalTimeline(length=1.5, uid="first")
        second = ContinuousPhysicalTimeline(length=2.5, uid="second")
        parent.add_child(first, offset=0.0)
        parent.add_child(second, offset=1.5)

        cast = parent.as_segment_line()

        assert type(cast) is SegmentLine[ContinuousPhysicalTimeline]
        assert cast.id == "parent"
        assert cast.name == "Parent"
        assert cast.meta == {"source": "fixture"}
        segments = [cast.get_segment_by_index(i)[1] for i in range(2)]
        assert [(s.id, s.length.value) for s in segments] == [
            ("first", 1.5),
            ("second", 2.5),
        ]
        assert segments[0] is not first and segments[1] is not second
        assert first._parent is parent and second._parent is parent
        assert len(cast.get_events(include_children=False)) == 1
        assert cast.get_conversion_map(TimeUnit.milliseconds) is cmap
        assert cast.get_region("whole").end.value == 4.0
        assert cast._flow_maps["flow"] is flow_marker
        assert cast.as_segment_line() is cast

    def test_boundary_children_cast_with_exact_names_offsets_and_type(self) -> None:
        """Boundary children cast to the parent's concrete SegmentLine class."""
        parent = ContinuousPhysicalTimeline(length=9.0, uid="movements")
        children = parent.create_children_from_boundaries(
            [0.0, 2.5, 6.0, 9.0],
            names=["opening", "development", "closing"],
        )

        assert parent.is_segment_line() is True

        segment_line = parent.as_segment_line()

        assert type(segment_line) is SegmentLine[ContinuousPhysicalTimeline]
        assert segment_line.n_segments == len(children) == 3
        assert segment_line.list_segments() == [
            "opening",
            "development",
            "closing",
        ]
        assert [
            segment_line.get_segment_by_index(index)[0].value
            for index in range(segment_line.n_segments)
        ] == [0.0, 2.5, 6.0]
        assert [
            segment_line.get_segment_by_index(index)[1].name
            for index in range(segment_line.n_segments)
        ] == ["opening", "development", "closing"]

    def test_nested_segment_line_diagram_shows_its_segments(self) -> None:
        """A nested SegmentLine renders its segments like any other child."""
        parent = ContinuousPhysicalTimeline(length=8.0, uid="score")
        sections = SegmentLine[ContinuousPhysicalTimeline](
            length=0.0,
            uid="sections",
            name="sections",
        )
        sections.append_segment(
            ContinuousPhysicalTimeline(length=3.0),
            name="opening",
        )
        sections.append_segment(
            ContinuousPhysicalTimeline(length=5.0),
            name="closing",
        )
        parent.add_child(sections, offset=0.0)

        rendered = parent.diagram()
        collapsed = parent.diagram(depth=1)

        assert "  └─ sections" in rendered
        assert rendered.count("sections") == 1
        assert "     ├─ opening" in rendered
        assert "     └─ closing" in rendered
        assert "opening" not in collapsed
        assert "closing" not in collapsed

    def test_as_segment_line_gap_names_first_offending_pair(self):
        """A gap reports both adjacent child IDs and exact offsets."""
        timeline = Timeline(length=3.0, unit=TimeUnit.seconds)
        timeline.add_child(
            ContinuousPhysicalTimeline(length=1.0, uid="left"),
            offset=0.0,
        )
        timeline.add_child(
            ContinuousPhysicalTimeline(length=1.0, uid="right"),
            offset=2.0,
        )

        with pytest.raises(ValueError) as error:
            timeline.as_segment_line()

        assert str(error.value) == (
            "Children 'left' and 'right' are not contiguous: "
            "expected offset 1, got 2"
        )

    def test_as_segment_line_rejects_heterogeneous_child_classes(self):
        """A structurally valid hierarchy still requires one exact child class."""
        timeline = Timeline(length=2.0, unit=TimeUnit.seconds)
        timeline.add_child(
            ContinuousPhysicalTimeline(length=1.0, uid="typed"),
            offset=0.0,
        )
        timeline.add_child(
            Timeline(length=1.0, unit=TimeUnit.seconds, uid="base"),
            offset=1.0,
        )

        with pytest.raises(TypeError, match="heterogeneous child classes"):
            timeline.as_segment_line()

    def test_to_timeline_drops_outer_segment_enforcement(self):
        """The plain cast keeps content but returns the parameter class."""
        line = SegmentLine[ContinuousPhysicalTimeline](
            length=0,
            uid="line",
            meta={"kind": "sections"},
        )
        child = ContinuousPhysicalTimeline(length=2.0, uid="section")
        line.append_segment(child)

        timeline = line.to_timeline()

        assert type(timeline) is ContinuousPhysicalTimeline
        assert not isinstance(timeline, SegmentLine)
        assert timeline.id == "line"
        assert timeline.meta == {"kind": "sections"}
        copied = timeline.get_child("section")
        assert (copied.id, copied.length.value) == ("section", 2.0)
        assert copied is not child
        assert child._parent is line

    def test_parameterized_from_segmentation_rejects_source_type_mismatch(self):
        """A bound factory accepts only its exact parameter class."""
        source = ContinuousLogicalTimeline(length=4)

        with pytest.raises(TypeError, match="requires source type"):
            SegmentLine[ContinuousPhysicalTimeline].from_segmentation(
                source,
                [0, 4],
            )


# endregion


# region derive() Tests


class TestTimelineDerive:
    """Test Timeline.derive() - creating derivative timelines via C-maps."""

    def test_derive_creates_timeline_in_target_unit(self):
        """derive() creates a new timeline in the C-map's target unit."""
        audio = ContinuousPhysicalTimeline(length=60.0, unit=TimeUnit.seconds)

        # 2 quarters per second = 120 BPM
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="tempo",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters)

        assert score.unit == TimeUnit.quarters
        assert score.length.value == 120.0  # 60 * 2

    def test_derive_creates_correct_timeline_type(self):
        """derive() creates appropriate Timeline subclass for target domain."""
        audio = ContinuousPhysicalTimeline(length=60.0)
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters)

        # Should be ContinuousLogicalTimeline (quarters is in logical domain)
        assert isinstance(score, ContinuousLogicalTimeline)

    def test_derive_attaches_inverse_cmap(self):
        """derive() attaches inverse C-map for roundtrip conversion."""
        audio = ContinuousPhysicalTimeline(length=60.0)
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters)

        # Derived timeline should have C-map back to seconds
        inverse = score.get_conversion_map(TimeUnit.seconds)
        assert inverse is not None

        # Roundtrip test: 60 quarters -> seconds -> quarters
        seconds = inverse(60.0)  # 60 / 2 = 30
        assert seconds == 30.0

    def test_derive_roundtrip_accuracy(self):
        """derive() enables accurate roundtrip conversions."""
        audio = ContinuousPhysicalTimeline(length=100.0)
        tempo_map = LinearMap(
            scalar=1.5,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters)
        inverse = score.get_conversion_map(TimeUnit.seconds)

        # Test multiple values
        test_values = [0.0, 25.0, 50.0, 75.0, 100.0]
        for original in test_values:
            quarters = tempo_map(original)
            back = inverse(quarters)
            # These fixed inputs recover bit-exactly through *1.5 then *(2/3).
            assert back == original

    def test_derive_without_cmap_raises(self):
        """derive() raises ValueError if no C-map for target unit."""
        audio = ContinuousPhysicalTimeline(length=60.0)

        with pytest.raises(ValueError, match="No C-Map"):
            audio.derive(TimeUnit.quarters)

    def test_derive_with_custom_name(self):
        """derive() uses custom name if provided."""
        audio = ContinuousPhysicalTimeline(length=60.0)
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters, name="my_score")

        assert score.name == "my_score"

    def test_derive_copies_events_when_requested(self):
        """derive() copies and converts events when copy_events=True."""
        audio = ContinuousPhysicalTimeline(length=60.0)
        audio.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 10.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 30.0,
                },
            ]
        )
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters, copy_events=True)

        # Events should be copied and coordinates converted
        assert score.n_events == 2

    def test_derive_does_not_copy_events_by_default(self):
        """derive() does not copy events by default."""
        audio = ContinuousPhysicalTimeline(length=60.0)
        audio.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 10.0,
                },
            ]
        )
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        score = audio.derive(TimeUnit.quarters)

        assert score.n_events == 0


# endregion


# region get_timeline_class Tests


class TestGetTimelineClass:
    """Test the get_timeline_class() factory function."""

    def test_logical_continuous(self):
        """get_timeline_class('logical', False) returns ContinuousLogicalTimeline."""
        cls = get_timeline_class("logical", discrete=False)
        assert cls == ContinuousLogicalTimeline

    def test_logical_discrete(self):
        """get_timeline_class('logical', True) returns DiscreteLogicalTimeline."""
        from timetoalign.timelines import DiscreteLogicalTimeline

        cls = get_timeline_class("logical", discrete=True)
        assert cls == DiscreteLogicalTimeline

    def test_physical_continuous(self):
        """get_timeline_class('physical', False) returns ContinuousPhysicalTimeline."""
        cls = get_timeline_class("physical", discrete=False)
        assert cls == ContinuousPhysicalTimeline

    def test_physical_discrete(self):
        """get_timeline_class('physical', True) returns DiscretePhysicalTimeline."""
        from timetoalign.timelines import DiscretePhysicalTimeline

        cls = get_timeline_class("physical", discrete=True)
        assert cls == DiscretePhysicalTimeline

    def test_graphical_continuous(self):
        """get_timeline_class('graphical', False) returns ContinuousGraphicalTimeline."""
        from timetoalign.timelines import ContinuousGraphicalTimeline

        cls = get_timeline_class("graphical", discrete=False)
        assert cls == ContinuousGraphicalTimeline

    def test_graphical_discrete(self):
        """get_timeline_class('graphical', True) returns DiscreteGraphicalTimeline."""
        from timetoalign.timelines import DiscreteGraphicalTimeline

        cls = get_timeline_class("graphical", discrete=True)
        assert cls == DiscreteGraphicalTimeline

    def test_unknown_domain_raises(self):
        """get_timeline_class raises ValueError for unknown domain."""
        with pytest.raises(ValueError, match="Unknown domain"):
            get_timeline_class("imaginary")


# endregion


class TestGetEventsAt:
    """Test get_events_at() method for point-in-time queries."""

    def test_get_events_at_instant(self):
        """get_events_at returns instant events at exact coordinate."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 50.0,
                },
            ]
        )

        result = tl.get_events_at(50.0)

        assert tl.id in result
        assert len(result[tl.id]) == 1
        assert result[tl.id][0]["id"] == "e1"

    def test_get_events_at_with_tolerance(self):
        """get_events_at uses tolerance for instant matching."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 50.0,
                },
            ]
        )

        # Without tolerance - no match
        result_strict = tl.get_events_at(50.5)
        assert tl.id not in result_strict or len(result_strict.get(tl.id, [])) == 0

        # With tolerance - matches
        result_tolerant = tl.get_events_at(50.5, tolerance=1.0)
        assert tl.id in result_tolerant
        assert len(result_tolerant[tl.id]) == 1

    def test_get_events_at_interval_left_inclusive(self):
        """get_events_at includes intervals where coord is at start (left-inclusive)."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 30.0,
                    "end": 40.0,
                },
            ]
        )

        # At start coordinate (30.0) - included
        result = tl.get_events_at(30.0)
        assert tl.id in result
        assert len(result[tl.id]) == 1

    def test_get_events_at_interval_right_exclusive(self):
        """get_events_at excludes intervals where coord is at end (right-exclusive)."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 30.0,
                    "end": 40.0,
                },
            ]
        )

        # At end coordinate (40.0) - NOT included (right-exclusive)
        result = tl.get_events_at(40.0)
        assert tl.id not in result or len(result.get(tl.id, [])) == 0

    def test_get_events_at_interval_middle(self):
        """get_events_at includes intervals where coord is in the middle."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 30.0,
                    "end": 40.0,
                },
            ]
        )

        result = tl.get_events_at(35.0)
        assert tl.id in result
        assert len(result[tl.id]) == 1

    def test_get_events_at_includes_children(self):
        """get_events_at includes events from children."""
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=20, unit=TimeUnit.seconds, name="child")

        child.add_events(
            [
                {
                    "id": "c1",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 5.0,
                },
            ]
        )
        parent.add_child(child, offset=30)

        # Query at coord 35 in parent = coord 5 in child
        result = parent.get_events_at(35.0)

        # Should find the child event
        assert any("c1" in str(events) for events in result.values())

    def test_get_events_at_exclude_children(self):
        """get_events_at with include_children=False."""
        parent = Timeline(length=100, unit=TimeUnit.seconds)
        parent.add_events(
            [
                {
                    "id": "p1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 35.0,
                },
            ]
        )

        child = Timeline(length=20, unit=TimeUnit.seconds)
        child.add_events(
            [
                {
                    "id": "c1",
                    "temporal_type": "instant",
                    "event_type": "Note",
                    "instant": 5.0,
                },
            ]
        )
        parent.add_child(child, offset=30)

        result = parent.get_events_at(35.0, include_children=False)

        # Only parent event, not child
        assert len(result) == 1
        assert parent.id in result

    def test_get_events_at_returns_dict_by_timeline_id(self):
        """get_events_at returns dict keyed by timeline ID."""
        tl = Timeline(length=100, unit=TimeUnit.seconds, name="my_timeline")
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 50.0,
                },
            ]
        )

        result = tl.get_events_at(50.0)

        assert isinstance(result, dict)
        assert tl.id in result

    def test_get_events_at_empty_when_no_match(self):
        """get_events_at returns empty dict when no events match."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 50.0,
                },
            ]
        )

        result = tl.get_events_at(99.0)  # No events here

        assert result == {} or tl.id not in result

    def test_get_events_at_accepts_coordinate_object(self):
        """get_events_at accepts Coordinate object as input."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 50.0,
                },
            ]
        )

        coord = Coordinate(50.0, TimeUnit.seconds)
        result = tl.get_events_at(coord)

        assert tl.id in result
        assert len(result[tl.id]) == 1


# endregion


# region Integration Tests


class TestTimelineRelationshipsIntegration:
    """Integration tests combining multiple relationship concepts."""

    def test_region_to_segmentline(self):
        """Regions can be used to create a SegmentLine structure."""
        # Create a timeline representing a song
        song = ContinuousPhysicalTimeline(length=180.0, unit=TimeUnit.seconds)

        # Define song structure via regions
        song.create_region(0, 30, id="Intro", name="Intro")
        song.create_region(30, 60, id="Verse1", name="Verse1")
        song.create_region(60, 90, id="Chorus1", name="Chorus1")
        song.create_region(90, 120, id="Verse2", name="Verse2")
        song.create_region(120, 150, id="Chorus2", name="Chorus2")
        song.create_region(150, 180, id="Outro", name="Outro")

        # create_child_from_region creates children
        assert song.n_regions == 6

        intro = song.create_child_from_region("Intro", copy_events=False)
        assert intro.length.value == 30.0

    def test_derive_then_add_children(self):
        """Derived timeline can have children added (same unit)."""
        # Physical timeline with tempo
        audio = ContinuousPhysicalTimeline(length=60.0)
        tempo_map = LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.quarters,
            uid="t",
        )
        audio.add_conversion_map(tempo_map)

        # Derive logical timeline
        score = audio.derive(TimeUnit.quarters)

        # Add children (measures) to derived timeline
        for i in range(30):
            measure = ContinuousLogicalTimeline(length=4, name=f"m{i+1}")
            score.add_child(measure, offset=i * 4)

        assert score.n_children == 30

    def test_segmentline_with_cmaps(self):
        """SegmentLine segments can have individual C-maps."""
        # Create a SegmentLine with segments having different tempos
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)

        # Segment 1: 8 quarters at 120 BPM (4 seconds)
        s1 = ContinuousLogicalTimeline(length=8, name="fast")
        s1.add_conversion_map(
            LinearMap(
                scalar=0.5,  # 1 quarter = 0.5 seconds at 120 BPM
                offset=0.0,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
                uid="s1_tempo",
            )
        )
        sl.append_segment(s1)

        # Segment 2: 8 quarters at 60 BPM (8 seconds)
        s2 = ContinuousLogicalTimeline(length=8, name="slow")
        s2.add_conversion_map(
            LinearMap(
                scalar=1.0,  # 1 quarter = 1 second at 60 BPM
                offset=0.0,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
                uid="s2_tempo",
            )
        )
        sl.append_segment(s2)

        assert sl.length.value == 16  # 8 + 8 quarters
        # Fast section: 8 * 0.5 = 4 seconds
        # Slow section: 8 * 1.0 = 8 seconds
        # Total physical duration would be 12 seconds


# endregion


# region Unified verb×noun API Tests


class TestCreateRegion:
    """Test Timeline.create_region() — the new explicit name for region creation."""

    def test_create_region_returns_region(self):
        """create_region returns a Region object."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        region = tl.create_region(30, 60, id="Chorus", name="Chorus")

        assert isinstance(region, Region)
        assert region.name == "Chorus"
        assert region.start.value == 30.0
        assert region.end.value == 60.0

    def test_create_region_attaches_to_timeline(self):
        """create_region adds the region to the timeline."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(30, 60, id="Chorus", name="Chorus")

        assert tl.n_regions == 1
        assert tl.has_region("Chorus")

    def test_create_region_with_meta(self):
        """create_region passes metadata to the Region."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        region = tl.create_region(
            30, 60, id="Chorus", name="Chorus", meta={"repeat": 2}
        )

        assert region.meta == {"repeat": 2}


class TestAddRegionOverloaded:
    """Test overloaded add_region(Region | str, ...)."""

    def test_add_region_with_region_object(self):
        """add_region(Region) attaches an existing Region object."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        r = Region(
            Coordinate(30, TimeUnit.seconds),
            Coordinate(60, TimeUnit.seconds),
            id="Chorus",
            name="Chorus",
        )
        result = tl.add_region(r)

        assert result is r
        assert tl.has_region("Chorus")
        assert tl.get_region("Chorus") is r

    def test_add_region_with_string_delegates_to_create(self):
        """add_region(name, start, end) delegates to create_region."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        result = tl.create_region(0, 30, id="Verse", name="Verse")

        assert isinstance(result, Region)
        assert result.name == "Verse"
        assert tl.has_region("Verse")

    def test_add_region_object_validates_unit(self):
        """add_region(Region) rejects unit mismatch."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        r = Region(
            Coordinate(0, TimeUnit.quarters),
            Coordinate(10, TimeUnit.quarters),
            id="X",
            name="X",
        )
        with pytest.raises(ValueError, match="unit"):
            tl.add_region(r)


class TestCreateRegionsFromBoundaries:
    """Test Timeline.create_regions_from_boundaries()."""

    def test_basic_boundaries(self):
        """Create regions from boundary coordinates."""
        tl = Timeline(length=90, unit=TimeUnit.seconds)
        regions = tl.create_regions_from_boundaries([0, 30, 60, 90], prefix="movement")

        assert len(regions) == 3
        assert regions[0].name == "movement_1"
        assert regions[0].start.value == 0.0
        assert regions[0].end.value == 30.0
        assert regions[1].name == "movement_2"
        assert regions[2].name == "movement_3"

    def test_custom_names(self):
        """Explicit names override format string."""
        tl = Timeline(length=90, unit=TimeUnit.seconds)
        regions = tl.create_regions_from_boundaries(
            [0, 30, 60, 90], names=["Intro", "Verse", "Chorus"]
        )

        assert [r.name for r in regions] == ["Intro", "Verse", "Chorus"]

    def test_wrong_name_count_raises(self):
        """Providing wrong number of names raises ValueError."""
        tl = Timeline(length=90, unit=TimeUnit.seconds)
        with pytest.raises(ValueError, match="Expected 3"):
            tl.create_regions_from_boundaries([0, 30, 60, 90], names=["A", "B"])

    def test_fewer_than_two_boundaries_raises(self):
        """Fewer than 2 boundaries raises ValueError."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        with pytest.raises(ValueError, match="at least 2"):
            tl.create_regions_from_boundaries([50])

    def test_non_monotonic_raises(self):
        """Non-monotonically-increasing boundaries raises ValueError."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        with pytest.raises(ValueError, match="monotonically"):
            tl.create_regions_from_boundaries([0, 60, 30, 100])


class TestCreateRegionsByGrouping:
    """Test Timeline.create_regions_by_grouping() with adjacent grouping.

    Note: Base EventData only preserves standard schema fields (id, name,
    temporal_type, event_type, start, end, duration). Extra columns like
    'timesig' are dropped. We use 'event_type' for synthetic grouping tests.
    Real data tests with Ms3Loader (MeasureData) preserve all columns.
    """

    def test_basic_grouping(self):
        """Adjacent grouping creates runs of same-value events."""
        tl = Timeline(length=120, unit=TimeUnit.quarters)
        tl.add_events(
            [
                {"event_type": "Fast", "start": 0, "end": 16},
                {"event_type": "Fast", "start": 16, "end": 32},
                {"event_type": "Slow", "start": 32, "end": 44},
                {"event_type": "Slow", "start": 44, "end": 56},
                {"event_type": "Fast", "start": 56, "end": 72},
                {"event_type": "Fast", "start": 72, "end": 120},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_grouping("event_type")

        # 3 runs: Fast, Slow, Fast — NOT 2 groups as standard group-by
        assert len(regions) == 3
        assert regions[0].name == "Fast"
        assert regions[0].start.value == 0.0
        assert regions[0].end.value == 32.0
        assert regions[1].name == "Slow"
        assert regions[1].start.value == 32.0
        assert regions[1].end.value == 56.0
        # A recurring value repeats the name; the ids tell the runs apart
        assert regions[2].name == "Fast"
        assert [region.id for region in regions] == ["region_1", "region_2", "region_3"]

    def test_custom_name_format(self):
        """name_format with {run} distinguishes recurrences."""
        tl = Timeline(length=60, unit=TimeUnit.quarters)
        tl.add_events(
            [
                {"event_type": "Alpha", "start": 0, "end": 20},
                {"event_type": "Beta", "start": 20, "end": 40},
                {"event_type": "Alpha", "start": 40, "end": 60},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_grouping(
            "event_type", name_format="{value}_run{run}"
        )

        assert len(regions) == 3
        assert regions[0].name == "Alpha_run1"
        assert regions[1].name == "Beta_run1"
        assert regions[2].name == "Alpha_run2"

    def test_nonexistent_column_raises(self):
        """Grouping on missing column raises ValueError."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events([{"event_type": "X", "instant": 10}])
        with pytest.raises(ValueError, match="not found"):
            tl.create_regions_by_grouping("nonexistent_column")


class TestCreateRegionsBySplitting:
    """Test Timeline.create_regions_by_splitting().

    Note: Splitting predicates operate on event dicts. Since base EventData
    only preserves standard fields, synthetic tests use 'event_type' or
    callable predicates. Real data tests with MeasureData use 'breaks'.
    """

    def test_split_by_event_type(self):
        """Split at events with a specific event_type."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {"event_type": "Measure", "start": 0, "end": 25},
                {"event_type": "Break", "start": 25, "end": 50},
                {"event_type": "Measure", "start": 50, "end": 75},
                {"event_type": "Break", "start": 75, "end": 100},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_splitting({"event_type": "Break"}, prefix="part")

        # Split points at end coordinates 50 and 100 of the "Break" events
        # Regions: [0, 50), [50, 100)
        assert len(regions) == 2
        assert regions[0].start.value == 0.0
        assert regions[0].end.value == 50.0
        assert regions[1].start.value == 50.0
        assert regions[1].end.value == 100.0

    def test_split_by_dict_predicate(self):
        """Split at events matching a dict filter on event_type."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {"event_type": "Line", "start": 0, "end": 20},
                {"event_type": "Note", "start": 20, "end": 40},
                {"event_type": "Section", "start": 40, "end": 60},
                {"event_type": "Line", "start": 60, "end": 80},
                {"event_type": "Note", "start": 80, "end": 100},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_splitting(
            {"event_type": "Section"}, prefix="movement"
        )

        # Only "Section" event at end=60
        # Regions: [0, 60), [60, 100)
        assert len(regions) == 2
        assert regions[0].name == "movement_1"
        assert regions[0].end.value == 60.0

    def test_split_by_dict_list_predicate(self):
        """Dict predicate with list of acceptable values."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {"event_type": "PageBreak", "start": 0, "end": 30},
                {"event_type": "Line", "start": 30, "end": 60},
                {"event_type": "SectionBreak", "start": 60, "end": 80},
                {"event_type": "Note", "start": 80, "end": 100},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_splitting(
            {"event_type": ["SectionBreak", "PageBreak"]}, prefix="part"
        )

        # Split points at end=30 (PageBreak) and end=80 (SectionBreak)
        # Regions: [0, 30), [30, 80), [80, 100)
        assert len(regions) == 3

    def test_split_by_callable(self):
        """Split at events matching a callable predicate."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {"event_type": "M", "start": 0, "end": 30, "name": "m1"},
                {"event_type": "M", "start": 30, "end": 60, "name": "split_here"},
                {"event_type": "M", "start": 60, "end": 100, "name": "m3"},
            ],
            allow_expansion=True,
        )

        regions = tl.create_regions_by_splitting(
            lambda e: e.get("name") == "split_here", prefix="section"
        )

        # Split point at end=60 of "split_here" event
        # Regions: [0, 60), [60, 100)
        assert len(regions) == 2


class TestCreateChildrenFromRegions:
    """Test Timeline.create_children_from_regions()."""

    def test_batch_create_all_regions(self):
        """create_children_from_regions() with no args uses all regions."""
        tl = Timeline(length=90, unit=TimeUnit.seconds)
        tl.create_region(0, 30, id="A", name="A")
        tl.create_region(30, 60, id="B", name="B")
        tl.create_region(60, 90, id="C", name="C")

        children = tl.create_children_from_regions()

        assert len(children) == 3
        assert tl.n_children == 3
        assert children[0].name == "A"
        assert children[1].name == "B"
        assert children[2].name == "C"

    def test_batch_create_subset(self):
        """create_children_from_regions() with names creates only those."""
        tl = Timeline(length=90, unit=TimeUnit.seconds)
        tl.create_region(0, 30, id="A", name="A")
        tl.create_region(30, 60, id="B", name="B")
        tl.create_region(60, 90, id="C", name="C")

        children = tl.create_children_from_regions(["A", "C"])

        assert len(children) == 2
        assert tl.n_children == 2
        assert children[0].name == "A"
        assert children[1].name == "C"


class TestGetRegionsAt:
    """Test Timeline.get_regions_at()."""

    def test_regions_at_point(self):
        """get_regions_at returns regions containing the coordinate."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 50, id="A", name="A")
        tl.create_region(30, 70, id="B", name="B")
        tl.create_region(60, 100, id="C", name="C")

        result = tl.get_regions_at(40)
        names = [r.name for r in result]
        assert names == ["A", "B"]

    def test_regions_at_boundary(self):
        """Left-inclusive, right-exclusive boundary behavior."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(0, 50, id="A", name="A")
        tl.create_region(50, 100, id="B", name="B")

        # At boundary 50: A excludes (right-exclusive), B includes (left-inclusive)
        result = tl.get_regions_at(50)
        assert len(result) == 1
        assert result[0].name == "B"


class TestGetChildrenAt:
    """Test Timeline.get_children_at()."""

    def test_children_at_coordinate(self):
        """get_children_at returns children covering the coordinate."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        c1 = Timeline(length=30, unit=TimeUnit.seconds, uid="c1")
        c2 = Timeline(length=40, unit=TimeUnit.seconds, uid="c2")
        tl.add_child(c1, offset=10)
        tl.add_child(c2, offset=20)

        # At 25: c1 covers [10, 40), c2 covers [20, 60) — both contain 25
        result = tl.get_children_at(25)
        ids = [c.id for c in result]
        assert "c1" in ids
        assert "c2" in ids

    def test_children_at_no_match(self):
        """get_children_at returns empty list when no children contain coord."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        c1 = Timeline(length=10, unit=TimeUnit.seconds, uid="c1")
        tl.add_child(c1, offset=20)

        result = tl.get_children_at(5)
        assert result == []


class TestListChildrenAndHasChild:
    """Test list_children() and has_child()."""

    def test_list_children(self):
        """list_children returns child IDs in insertion order."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_child(Timeline(length=10, uid="b"), offset=0)
        tl.add_child(Timeline(length=10, uid="a"), offset=20)

        assert tl.list_children() == ["b", "a"]

    def test_has_child(self):
        """has_child checks by ID."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.add_child(Timeline(length=10, uid="x"), offset=0)

        assert tl.has_child("x") is True
        assert tl.has_child("y") is False


class TestCreateSegmentLine:
    """Test Timeline.create_segment_line() instance method."""

    def test_create_segment_line_from_boundaries(self):
        """create_segment_line creates a new SegmentLine."""
        tl = ContinuousLogicalTimeline(length=100)
        sl = tl.create_segment_line([0, 25, 50, 75, 100])

        assert isinstance(sl, SegmentLine)
        assert sl.n_segments == 4
        assert sl.length.value == 100

    def test_create_segment_line_preserves_unit(self):
        """SegmentLine inherits parent's unit."""
        tl = ContinuousLogicalTimeline(length=100)
        sl = tl.create_segment_line([0, 50, 100])

        assert sl.unit == tl.unit

    def test_create_segment_line_copies_events(self):
        """create_segment_line copies events into segments."""
        tl = ContinuousLogicalTimeline(length=100)
        tl.add_events(
            [
                {"event_type": "Note", "instant": 10},
                {"event_type": "Note", "instant": 60},
            ]
        )

        sl = tl.create_segment_line([0, 50, 100])

        # Each segment should have 1 event
        _, seg0 = sl.get_segment_by_index(0)
        _, seg1 = sl.get_segment_by_index(1)
        assert seg0.n_events == 1
        assert seg1.n_events == 1

    def test_create_segment_line_does_not_modify_self(self):
        """create_segment_line does NOT modify the source timeline."""
        tl = ContinuousLogicalTimeline(length=100)
        tl.create_segment_line([0, 50, 100])

        assert tl.n_children == 0
        assert tl.n_regions == 0


class TestCreateSegmentLineFromRegions:
    """Test Timeline.create_segment_line_from_regions()."""

    def test_segment_line_from_contiguous_regions(self):
        """Creates SegmentLine from contiguous regions."""
        tl = ContinuousLogicalTimeline(length=90)
        tl.create_region(0, 30, id="A", name="A")
        tl.create_region(30, 60, id="B", name="B")
        tl.create_region(60, 90, id="C", name="C")

        sl = tl.create_segment_line_from_regions()

        assert sl.n_segments == 3
        assert sl.length.value == 90

    def test_non_contiguous_regions_raise(self):
        """Non-contiguous regions raise ValueError."""
        tl = ContinuousLogicalTimeline(length=100)
        tl.create_region(0, 30, id="A", name="A")
        tl.create_region(40, 70, id="B", name="B")  # Gap between 30 and 40

        with pytest.raises(ValueError, match="not contiguous"):
            tl.create_segment_line_from_regions()


class TestCreateSegmentLineByGrouping:
    """Test Timeline.create_segment_line_by_grouping()."""

    def test_basic_grouping_segment_line(self):
        """Creates SegmentLine by adjacent grouping on event_type."""
        tl = ContinuousLogicalTimeline(length=60)
        tl.add_events(
            [
                {"event_type": "Alpha", "start": 0, "end": 20},
                {"event_type": "Beta", "start": 20, "end": 40},
                {"event_type": "Alpha", "start": 40, "end": 60},
            ],
            allow_expansion=True,
        )

        sl = tl.create_segment_line_by_grouping("event_type")

        assert isinstance(sl, SegmentLine)
        assert sl.n_segments == 3
        assert sl.length.value == 60


class TestCreateSegmentLineBySplitting:
    """Test Timeline.create_segment_line_by_splitting()."""

    def test_basic_splitting_segment_line(self):
        """Creates SegmentLine by splitting at event_type matches."""
        tl = ContinuousLogicalTimeline(length=100)
        tl.add_events(
            [
                {"event_type": "Measure", "start": 0, "end": 30},
                {"event_type": "SectionBreak", "start": 30, "end": 60},
                {"event_type": "Measure", "start": 60, "end": 100},
            ],
            allow_expansion=True,
        )

        sl = tl.create_segment_line_by_splitting(
            {"event_type": "SectionBreak"}, prefix="part"
        )

        assert isinstance(sl, SegmentLine)
        assert sl.n_segments == 2
        assert sl.length.value == 100


class TestSegmentLineListAndHas:
    """Test SegmentLine.list_segments() and has_segment()."""

    def test_list_segments(self):
        """list_segments returns segment IDs in order."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        s1 = ContinuousLogicalTimeline(length=4, uid="seg_a")
        s2 = ContinuousLogicalTimeline(length=4, uid="seg_b")
        sl.append_segment(s1)
        sl.append_segment(s2)

        result = sl.list_segments()
        assert result == ["seg_a", "seg_b"]

    def test_has_segment(self):
        """has_segment checks segment ID."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        sl.append_segment(ContinuousLogicalTimeline(length=4, uid="seg_x"))

        assert sl.has_segment("seg_x") is True
        assert sl.has_segment("seg_y") is False

    def test_segmentline_contains_segment(self):
        """__contains__ on SegmentLine checks segments."""
        sl = SegmentLine(length=0, unit=TimeUnit.quarters)
        sl.append_segment(ContinuousLogicalTimeline(length=4, uid="seg_1"))

        assert "seg_1" in sl


class TestContainsRegionsAndChildren:
    """Test updated __contains__ behavior on Timeline."""

    def test_contains_region_name(self):
        """String 'in tl' checks regions."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(30, 60, id="Chorus", name="Chorus")

        assert "Chorus" in tl

    def test_contains_child_id(self):
        """String 'in tl' checks children."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=10, unit=TimeUnit.seconds, uid="child_1")
        tl.add_child(child, offset=0)

        assert "child_1" in tl

    def test_contains_region_object(self):
        """Region object 'in tl' checks by name."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        tl.create_region(30, 60, id="Chorus", name="Chorus")
        region = tl.get_region("Chorus")

        assert region in tl

    def test_contains_timeline_object(self):
        """Timeline object 'in tl' checks by identity."""
        tl = Timeline(length=100, unit=TimeUnit.seconds)
        child = Timeline(length=10, unit=TimeUnit.seconds)
        tl.add_child(child, offset=0)

        assert child in tl


# endregion


# region Real Data Tests (Ms3Loader)


class TestUnifiedAPIWithRealData:
    """Test the unified verb×noun API using real score data from Ms3Loader.

    Uses the Wagner Walküre Act III measures.tsv which has:
    - 1733 measures
    - 8 unique timesig values forming 22 adjacent runs
    - 405 break events (347 'line', 58 'page')
    - Timeline length 6699.5 quarter beats

    These tests verify that the new API methods work correctly with the
    data shapes produced by real musicological sources.

    Validation Strategy:
        Gold standard counts are derived from the TSV file itself via
        independent counting (Counter, set operations) and verified
        against the API output. No approximations or ranges.
    """

    WAGNER_PATH = (
        "tests/data/score/wagner_walkure/01_RawData/score_musescore/"
        "Wagner_WWV086B-3.measures.tsv"
    )

    @pytest.fixture
    def wagner_timeline(self):
        """Load Wagner Walküre measures into a ContinuousLogicalTimeline.

        Uses ``MeasureData.create_timeline()`` which directly assigns the
        MeasureData as the timeline's event store, preserving all extra
        columns (timesig, breaks, keysig, etc.) that base EventData would
        discard.

        Returns:
            A ContinuousLogicalTimeline with 1733 Measure interval events
            including timesig, keysig, breaks columns.
        """
        from pathlib import Path

        from timetoalign.loader.score import Ms3Loader

        path = Path(self.WAGNER_PATH)
        if not path.exists():
            pytest.skip(f"Test data not found: {path}")

        loader = Ms3Loader()
        loader.load(path)
        measures = loader.store.measures

        if len(measures) == 0:
            pytest.skip("No measure events loaded")

        return measures.create_timeline()

    def test_event_count(self, wagner_timeline):
        """Timeline has exactly 1733 measure events."""
        assert wagner_timeline.n_events == 1733

    def test_create_regions_by_grouping_timesig(self, wagner_timeline):
        """Grouping by timesig produces exactly 22 adjacent runs."""
        regions = wagner_timeline.create_regions_by_grouping("timesig")

        assert len(regions) == 22

        # Verify first region is 9/8
        assert regions[0].name == "9/8"

        # All regions should be contiguous
        for i in range(1, len(regions)):
            assert abs(regions[i].start.value - regions[i - 1].end.value) < 1e-10, (
                f"Gap between region {i - 1} ({regions[i - 1].name}) end="
                f"{regions[i - 1].end.value} and region {i} ({regions[i].name}) "
                f"start={regions[i].start.value}"
            )

    def test_create_regions_by_grouping_name_format(self, wagner_timeline):
        """Grouping with run-indexed names distinguishes recurrences."""
        regions = wagner_timeline.create_regions_by_grouping(
            "timesig", name_format="ts_{value}_run{run}"
        )

        # First 9/8 run gets run=1, second 9/8 run gets run=2
        r98_names = [r.name for r in regions if "9/8" in r.name]
        assert "ts_9/8_run1" in r98_names
        assert "ts_9/8_run2" in r98_names

    def test_create_regions_by_splitting_breaks(self, wagner_timeline):
        """Splitting by 'breaks' column creates regions at all break events.

        The Wagner Act III TSV has 405 break events (347 line + 58 page),
        each at a unique end coordinate. With 405 unique split points,
        splitting produces 406 contiguous regions.
        """
        regions = wagner_timeline.create_regions_by_splitting("breaks", prefix="system")

        assert len(regions) == 406
        assert regions[0].start.value == 0.0
        assert regions[-1].end.value == wagner_timeline.length.value

    def test_create_regions_by_splitting_page_breaks_only(self, wagner_timeline):
        """Splitting by {'breaks': 'page'} uses only page breaks.

        The Wagner Act III TSV has 58 page breaks, each at a unique
        coordinate. Splitting by page breaks produces 59 contiguous regions.
        """
        regions = wagner_timeline.create_regions_by_splitting(
            {"breaks": "page"}, prefix="page"
        )

        assert len(regions) == 59
        assert regions[0].name == "page_1"

        # All regions contiguous
        for i in range(1, len(regions)):
            assert abs(regions[i].start.value - regions[i - 1].end.value) < 1e-10

    def test_create_regions_by_splitting_section_breaks(self, wagner_timeline):
        """Splitting by {'breaks': ['section']} finds section breaks.

        The Wagner Act III TSV has 0 section breaks, so splitting produces
        exactly 1 region spanning the entire timeline.
        """
        regions = wagner_timeline.create_regions_by_splitting(
            {"breaks": ["section"]}, prefix="section"
        )

        assert len(regions) == 1

    def test_create_segment_line_by_grouping_timesig(self, wagner_timeline):
        """SegmentLine by timesig grouping creates 22 segments."""
        sl = wagner_timeline.create_segment_line_by_grouping("timesig")

        assert isinstance(sl, SegmentLine)
        assert sl.n_segments == 22
        assert sl.length.value == wagner_timeline.length.value

    def test_create_segment_line_by_splitting_page(self, wagner_timeline):
        """SegmentLine by page breaks creates correct segments."""
        sl = wagner_timeline.create_segment_line_by_splitting(
            {"breaks": "page"}, prefix="page"
        )

        assert isinstance(sl, SegmentLine)
        assert sl.n_segments == 59
        assert sl.length.value == wagner_timeline.length.value

    def test_create_child_from_region_with_real_data(self, wagner_timeline):
        """create_child_from_region works with real timesig regions.

        The first timesig region is '9/8' spanning 0.0-968.0 quarter beats,
        containing exactly 216 measures.
        """
        regions = wagner_timeline.create_regions_by_grouping("timesig")
        first_region = regions[0]

        child = wagner_timeline.create_child_from_region(first_region.id)

        assert child.length.value == 968.0
        assert child.name == "9/8"
        assert child.n_events == 216
        assert wagner_timeline.has_child(child.id)

    @pytest.mark.slow
    def test_create_children_from_regions_all(self, wagner_timeline):
        """create_children_from_regions creates children for all timesig regions."""
        wagner_timeline.create_regions_by_grouping("timesig")
        children = wagner_timeline.create_children_from_regions()

        assert len(children) == 22
        assert wagner_timeline.n_children == 22

    def test_get_regions_at_with_real_data(self, wagner_timeline):
        """get_regions_at finds the correct timesig region for a coordinate."""
        regions = wagner_timeline.create_regions_by_grouping("timesig")

        # Query in the middle of the first region
        mid = (regions[0].start.value + regions[0].end.value) / 2
        found = wagner_timeline.get_regions_at(mid)

        assert len(found) == 1
        assert found[0].name == regions[0].name

    def test_segment_line_does_not_modify_source(self, wagner_timeline):
        """create_segment_line_by_grouping does NOT modify the source."""
        n_children_before = wagner_timeline.n_children
        n_regions_before = wagner_timeline.n_regions

        wagner_timeline.create_segment_line_by_grouping("timesig")

        assert wagner_timeline.n_children == n_children_before
        assert wagner_timeline.n_regions == n_regions_before

    def test_roundtrip_grouping_regions_to_segment_line(self, wagner_timeline):
        """Regions from grouping can be used to create a SegmentLine."""
        regions = wagner_timeline.create_regions_by_grouping("timesig")
        sl = wagner_timeline.create_segment_line_from_regions()

        assert sl.n_segments == len(regions)
        assert sl.length.value == wagner_timeline.length.value


# endregion
