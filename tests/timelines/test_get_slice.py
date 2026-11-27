"""Tests for slicing a timeline.

1. ``Timeline.get_slice()`` (unit tests on synthetic timelines)
2. Assembly of slices into a SegmentLine (integration tests)
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
)
from timetoalign.timelines.types import SegmentLine

# region Fixtures


@pytest.fixture
def simple_logical_timeline() -> ContinuousLogicalTimeline:
    """A ContinuousLogicalTimeline with known events for slice testing.

    Timeline span: [0, 100) in quarters (Fraction).

    Events:
        Instant events:
            - Beat at 0
            - Beat at 10
            - Beat at 25
            - Beat at 50
            - Beat at 75
            - Beat at 99

        Interval events:
            - Note [5, 15)   -- straddles a typical slice boundary at 10
            - Note [20, 30)  -- fully inside [10, 50)
            - Note [45, 55)  -- straddles a typical slice boundary at 50
            - Note [80, 95)  -- fully inside [50, 100)
    """
    tl = ContinuousLogicalTimeline(
        length=Fraction(100),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    tl.add_events(
        [
            # Instant events
            {"event_type": "Beat", "instant": Fraction(0)},
            {"event_type": "Beat", "instant": Fraction(10)},
            {"event_type": "Beat", "instant": Fraction(25)},
            {"event_type": "Beat", "instant": Fraction(50)},
            {"event_type": "Beat", "instant": Fraction(75)},
            {"event_type": "Beat", "instant": Fraction(99)},
            # Interval events
            {"event_type": "Note", "start": Fraction(5), "end": Fraction(15)},
            {"event_type": "Note", "start": Fraction(20), "end": Fraction(30)},
            {"event_type": "Note", "start": Fraction(45), "end": Fraction(55)},
            {"event_type": "Note", "start": Fraction(80), "end": Fraction(95)},
        ]
    )
    return tl


@pytest.fixture
def timeline_with_child() -> ContinuousLogicalTimeline:
    """A ContinuousLogicalTimeline with a child for recursive slice testing.

    Parent span: [0, 100) in quarters.
    Parent events:
        - Beat at 10
        - Note [30, 60)

    Child span: [0, 40) embedded at offset 20, covering parent [20, 60).
    Child events:
        - Beat at 5  (= parent 25)
        - Note [10, 30) (= parent [30, 50))
    """
    parent = ContinuousLogicalTimeline(
        length=Fraction(100),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    parent.add_events(
        [
            {"event_type": "Beat", "instant": Fraction(10)},
            {"event_type": "Note", "start": Fraction(30), "end": Fraction(60)},
        ]
    )

    child = ContinuousLogicalTimeline(
        length=Fraction(40),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )
    child.add_events(
        [
            {"event_type": "Beat", "instant": Fraction(5)},
            {"event_type": "Note", "start": Fraction(10), "end": Fraction(30)},
        ]
    )
    parent.add_child(child, offset=Fraction(20))

    return parent


# endregion


# region TestGetSlice — Unit tests for Timeline.get_slice()


class TestGetSlice:
    """Test Timeline.get_slice() primitive.

    Uses synthetic timelines with known events to validate slicing logic
    in isolation. No external data files required.
    """

    def test_basic_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice extracts events in range with shifted coordinates."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert sliced.length.value == Fraction(40)

        # Events in [10, 50):
        #   Instant: Beat@10 -> 0, Beat@25 -> 15
        #   Interval: Note[20,30) -> [10,20), Note[5,15) truncated to [10,15) -> [0,5)
        #   Note[45,55) truncated to [45,50) -> [35,40)
        events = list(sliced.get_events())
        instant_events = [e for e in events if e.get("temporal_type") == "instant"]
        # interval_events = [e for e in events if e.get("temporal_type") == "interval"]

        # Beat@10 -> 0, Beat@25 -> 15 (Beat@0 is outside, Beat@50 is right-exclusive)
        assert len(instant_events) == 2
        # Note: In EventData, all events (including instants) store coords in "start"
        instant_coords = sorted(Fraction(e["start"]["value"]) for e in instant_events)
        assert instant_coords == [Fraction(0), Fraction(15)]

    def test_instant_at_boundary_included(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Instant at start boundary IS included (left-inclusive)."""
        sliced = simple_logical_timeline.get_slice(Fraction(25), Fraction(75))

        events = list(sliced.get_events(temporal_type="instant"))
        coords = sorted(Fraction(e["start"]["value"]) for e in events)

        # Beat@25 -> 0 (included), Beat@50 -> 25 (included)
        # Beat@75 is at end boundary -> excluded (right-exclusive)
        assert Fraction(0) in coords, "Instant at start boundary must be included"

    def test_instant_at_end_excluded(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Instant at end boundary is NOT included (right-exclusive)."""
        sliced = simple_logical_timeline.get_slice(Fraction(25), Fraction(50))

        events = list(sliced.get_events(temporal_type="instant"))
        coords = [Fraction(e["start"]["value"]) for e in events]

        # Beat@50 is exactly at end -> excluded
        assert Fraction(25) not in coords, "Instant at end boundary must be excluded"

    def test_interval_truncation(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Interval straddling boundary is clipped (truncate_events=True)."""
        # Slice [10, 50). Note [5, 15) straddles start: clipped to [10, 15) -> [0, 5)
        # Note [45, 55) straddles end: clipped to [45, 50) -> [35, 40)
        sliced = simple_logical_timeline.get_slice(
            Fraction(10), Fraction(50), truncate_events=True
        )

        intervals = list(sliced.get_events(temporal_type="interval"))

        # Should have 3 intervals: truncated[0,5), full[10,20), truncated[35,40)
        assert len(intervals) == 3

        # Check the truncated interval from Note[5,15)
        starts = sorted(Fraction(e["start"]["value"]) for e in intervals)
        ends = sorted(Fraction(e["end"]["value"]) for e in intervals)

        assert Fraction(0) in starts, "Truncated interval should start at 0"
        assert Fraction(5) in ends, "Truncated interval [0,5) should end at 5"
        assert Fraction(40) in ends, "Truncated interval at end should stop at 40"

    def test_interval_no_truncation(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """With truncate_events=False, straddling intervals are excluded."""
        sliced = simple_logical_timeline.get_slice(
            Fraction(10), Fraction(50), truncate_events=False
        )

        intervals = list(sliced.get_events(temporal_type="interval"))

        # Only Note[20,30) is fully contained in [10, 50)
        assert len(intervals) == 1
        start = Fraction(intervals[0]["start"]["value"])
        end = Fraction(intervals[0]["end"]["value"])
        assert start == Fraction(10)  # 20 - 10
        assert end == Fraction(20)  # 30 - 10

    def test_coordinate_shifting(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """All coordinates shifted by -start."""
        sliced = simple_logical_timeline.get_slice(Fraction(75), Fraction(100))

        events = list(sliced.get_events(temporal_type="instant"))
        # Beat@75 -> 0, Beat@99 -> 24
        coords = sorted(Fraction(e["start"]["value"]) for e in events)
        assert coords == [Fraction(0), Fraction(24)]

        intervals = list(sliced.get_events(temporal_type="interval"))
        # Note[80,95) -> [5, 20)
        assert len(intervals) == 1
        assert Fraction(intervals[0]["start"]["value"]) == Fraction(5)
        assert Fraction(intervals[0]["end"]["value"]) == Fraction(20)

    def test_empty_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice of range with no events returns empty timeline."""
        # Range [96, 99) has no events (Beat@99 is at 99, not in [96, 99))
        sliced = simple_logical_timeline.get_slice(Fraction(96), Fraction(99))

        assert sliced.length.value == Fraction(3)
        events = list(sliced.get_events())
        assert len(events) == 0

    def test_full_slice(self, simple_logical_timeline: ContinuousLogicalTimeline):
        """Slice of entire timeline returns copy with all events."""
        sliced = simple_logical_timeline.get_slice(Fraction(0), Fraction(100))

        assert sliced.length.value == Fraction(100)
        original_events = list(simple_logical_timeline.get_events())
        sliced_events = list(sliced.get_events())
        assert len(sliced_events) == len(original_events)

    def test_children_sliced(self, timeline_with_child: ContinuousLogicalTimeline):
        """Child timelines are recursively sliced when include_children=True."""
        # Parent [0, 100), child at offset 20 with length 40, covering [20, 60)
        # Slice [25, 55) should include child's overlap [25, 55) -> child coords [5, 35)
        sliced = timeline_with_child.get_slice(
            Fraction(25), Fraction(55), include_children=True
        )

        assert sliced.length.value == Fraction(30)  # 55 - 25
        assert sliced.n_children == 1, "Sliced child should be present"

    def test_children_excluded(self, timeline_with_child: ContinuousLogicalTimeline):
        """Child timelines are excluded when include_children=False."""
        sliced = timeline_with_child.get_slice(
            Fraction(25), Fraction(55), include_children=False
        )

        assert sliced.n_children == 0

    def test_number_type_preserved(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Fraction coordinates remain Fraction in slice."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert sliced.number_type == NumberType.fraction

    def test_concrete_class_preserved(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """ContinuousLogicalTimeline.get_slice() returns ContinuousLogicalTimeline."""
        sliced = simple_logical_timeline.get_slice(Fraction(10), Fraction(50))

        assert type(sliced) is ContinuousLogicalTimeline

    def test_slice_of_discrete_timeline(self):
        """get_slice works on DiscreteLogicalTimeline with int coordinates."""
        tl = DiscreteLogicalTimeline(
            length=100,
            unit=TimeUnit.ticks,
        )
        tl.add_events(
            [
                {"event_type": "Tick", "instant": 10},
                {"event_type": "Tick", "instant": 50},
                {"event_type": "Tick", "instant": 90},
            ]
        )

        sliced = tl.get_slice(20, 80)
        assert sliced.length.value == 60
        assert type(sliced) is DiscreteLogicalTimeline

        events = list(sliced.get_events())
        assert len(events) == 1  # Only Tick@50 -> 30
        assert events[0]["start"]["value"] == 30

    def test_slice_of_physical_timeline(self):
        """get_slice works on ContinuousPhysicalTimeline with float coordinates."""
        tl = ContinuousPhysicalTimeline(
            length=10.0,
            unit=TimeUnit.seconds,
        )
        tl.add_events(
            [
                {"event_type": "Onset", "instant": 1.0},
                {"event_type": "Onset", "instant": 5.0},
                {"event_type": "Onset", "instant": 9.0},
            ]
        )

        sliced = tl.get_slice(2.0, 8.0)
        assert sliced.length.value == 6.0
        assert type(sliced) is ContinuousPhysicalTimeline

        events = list(sliced.get_events())
        assert len(events) == 1  # Only Onset@5.0 -> 3.0

    def test_invalid_range_start_ge_end(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Raises ValueError if start >= end."""
        with pytest.raises(ValueError, match="start.*end"):
            simple_logical_timeline.get_slice(Fraction(50), Fraction(50))

        with pytest.raises(ValueError, match="start.*end"):
            simple_logical_timeline.get_slice(Fraction(60), Fraction(40))

    def test_invalid_range_outside_bounds(
        self, simple_logical_timeline: ContinuousLogicalTimeline
    ):
        """Raises ValueError if range is outside timeline bounds."""
        with pytest.raises(ValueError):
            simple_logical_timeline.get_slice(Fraction(-10), Fraction(50))

        with pytest.raises(ValueError):
            simple_logical_timeline.get_slice(Fraction(50), Fraction(110))


# endregion


# region TestSegmentLineAssembly — Integration tests for slice assembly


class TestSegmentLineAssembly:
    """Test assembling slices into a SegmentLine.

    Uses synthetic timelines to validate the structural properties of
    SegmentLine construction from get_slice() outputs.
    """

    def test_contiguity(self):
        """Segments are contiguous (each starts where previous ended)."""
        # Create source timeline
        source = ContinuousLogicalTimeline(
            length=Fraction(120),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(0)},
                {"event_type": "Beat", "instant": Fraction(40)},
                {"event_type": "Beat", "instant": Fraction(80)},
            ]
        )

        # Slice at [0, 40), [40, 80), [80, 120)
        boundaries = [
            (Fraction(0), Fraction(40)),
            (Fraction(40), Fraction(80)),
            (Fraction(80), Fraction(120)),
        ]

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        for qb_start, qb_end in boundaries:
            seg = source.get_slice(qb_start, qb_end)
            sl.append_segment(seg)

        assert sl.n_segments == 3

        # Verify contiguity: each segment starts where previous ended
        offsets = []
        for seg_id in sl._segment_order:
            offset = sl._child_offsets[seg_id].value
            child = sl._children[seg_id]
            offsets.append((offset, offset + child.length.value))

        for i in range(1, len(offsets)):
            assert offsets[i][0] == offsets[i - 1][1], (
                f"Segment {i} start {offsets[i][0]} != "
                f"segment {i - 1} end {offsets[i - 1][1]}"
            )

    def test_total_length(self):
        """Total SegmentLine length equals sum of slice lengths."""
        source = ContinuousLogicalTimeline(
            length=Fraction(100),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        # Simulate unfolding: two passes through [0, 100)
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        slice1 = source.get_slice(Fraction(0), Fraction(100))
        slice2 = source.get_slice(Fraction(0), Fraction(100))
        sl.append_segment(slice1)
        sl.append_segment(slice2)

        assert sl.length.value == Fraction(200)
        assert sl.n_segments == 2

    def test_events_preserved(self):
        """All events from individual slices appear in the assembled SegmentLine."""
        source = ContinuousLogicalTimeline(
            length=Fraction(60),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(10)},
                {"event_type": "Beat", "instant": Fraction(30)},
                {"event_type": "Beat", "instant": Fraction(50)},
            ]
        )

        # Slice at [0, 30), [30, 60)
        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        seg1 = source.get_slice(Fraction(0), Fraction(30))
        seg2 = source.get_slice(Fraction(30), Fraction(60))
        sl.append_segment(seg1)
        sl.append_segment(seg2)

        # Each segment should have its events
        seg1_events = list(sl._children[sl._segment_order[0]].get_events())
        seg2_events = list(sl._children[sl._segment_order[1]].get_events())

        assert len(seg1_events) == 1  # Beat@10 -> 10
        assert len(seg2_events) == 2  # Beat@30 -> 0, Beat@50 -> 20

    def test_segment_type_matches_source(self):
        """SegmentLine's segment_type matches the source timeline class."""
        source = ContinuousLogicalTimeline(
            length=Fraction(40),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )

        seg = source.get_slice(Fraction(0), Fraction(40))
        sl.append_segment(seg)

        assert sl.segment_type is ContinuousLogicalTimeline
        assert type(sl._children[sl._segment_order[0]]) is ContinuousLogicalTimeline

    def test_a_repeated_section_is_refused_at_the_join(self):
        """Two slices of one range hold the same events: the second is refused.

        This is a simple repeat, the same [0, 40) section played twice. Event
        IDs are unique among a root timeline and its descendants and a
        collision is refused at a join, nothing renamed: the second slice,
        holding the first's event IDs, does not join the line, which keeps its
        one segment.
        """
        source = ContinuousLogicalTimeline(
            length=Fraction(40),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        first_beat, _ = source.add_events(
            [
                {"event_type": "Beat", "instant": Fraction(0)},
                {"event_type": "Beat", "instant": Fraction(20)},
            ]
        )

        sl = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        first = source.get_slice(Fraction(0), Fraction(40))
        sl.append_segment(first)
        second = source.get_slice(Fraction(0), Fraction(40))

        with pytest.raises(ValueError) as refusal:
            sl.append_segment(second)

        assert str(refusal.value) == (
            f"Event id '{first_beat}' is already held by an event of timeline "
            f"'{first.id}', so it cannot join timeline '{second.id}': event ids "
            "are unique among a root timeline and its descendants"
        )
        assert sl.length.value == Fraction(40)
        assert sl.n_segments == 1
        assert second._parent is None
        assert len(list(first.get_events())) == 2


# endregion
