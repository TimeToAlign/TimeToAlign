"""Tests for create_timeline factory function.

This module tests the timeline creation API including:
- create_timeline() universal factory
- EventStore.create_timeline()
- EventData.create_timeline()

Behavior tested:
- **Single data source**: Events are placed directly on the timeline (no children).
- **Multiple data sources** (e.g., ScoreStore): Each data becomes a child timeline.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.storage import EventData, SingleStore
from timetoalign.storage.store import EventStore
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    Timeline,
    create_timeline,
)
from timetoalign.timelines.factory import _infer_timeline_class

if TYPE_CHECKING:
    pass


def _child_names(timeline: Timeline) -> list[str]:
    """The names of a timeline's children, in child order."""
    return [timeline.get_child(child_id).name for child_id in timeline.list_children()]


def _child_named(timeline: Timeline, name: str) -> Timeline:
    """The one child of *timeline* named *name*; child ids are minted."""
    (child,) = [
        timeline.get_child(child_id)
        for child_id in timeline.list_children()
        if timeline.get_child(child_id).name == name
    ]
    return child


# region Test Helper: DictStore


@dataclass
class DictStore(EventStore):
    """Simple multi-data store for testing.

    Implements EventStore protocol with a dict of name -> EventData.
    """

    _data: dict[str, EventData]

    def __init__(self, data: dict[str, EventData]) -> None:
        """Initialize DictStore."""
        self._data = data

    def __iter__(self) -> Iterator[EventData]:
        """Iterate over data."""
        yield from self._data.values()

    def items(self) -> Iterator[tuple[str, EventData]]:
        """Iterate over (name, data) pairs."""
        yield from self._data.items()

    def keys(self) -> tuple[str, ...]:
        """Return data names."""
        return tuple(self._data.keys())

    def __getitem__(self, name: str) -> EventData:
        """Get data by name."""
        return self._data[name]

    def __len__(self) -> int:
        """Return number of data."""
        return len(self._data)

    def __contains__(self, name: object) -> bool:
        """Check if name is in data."""
        return name in self._data


# endregion


class TestInferTimelineClass:
    """Tests for _infer_timeline_class helper."""

    def test_logical_discrete(self):
        """Ticks with int -> DiscreteLogicalTimeline."""
        cls = _infer_timeline_class(TimeUnit.ticks, NumberType.int)
        assert cls is DiscreteLogicalTimeline

    def test_logical_continuous_float(self):
        """Quarters with float -> ContinuousLogicalTimeline."""
        cls = _infer_timeline_class(TimeUnit.quarters, NumberType.float)
        assert cls is ContinuousLogicalTimeline

    def test_logical_continuous_fraction(self):
        """Quarters with fraction -> ContinuousLogicalTimeline."""
        cls = _infer_timeline_class(TimeUnit.quarters, NumberType.fraction)
        assert cls is ContinuousLogicalTimeline

    def test_physical_continuous(self):
        """Seconds with float -> ContinuousPhysicalTimeline."""
        cls = _infer_timeline_class(TimeUnit.seconds, NumberType.float)
        assert cls is ContinuousPhysicalTimeline


class TestCreateTimelineFromEventData:
    """Tests for create_timeline with EventData source (single data).

    Since EventData is a single data source, events should be placed
    directly on the timeline (no children).
    """

    @pytest.fixture
    def sample_data(self) -> EventData:
        """EventData with sample events."""
        return EventData.from_dicts(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 480,
                },
            ],
            unit=TimeUnit.ticks,
        )

    def test_creates_timeline_with_events_directly(self, sample_data: EventData):
        """Single data source: events are placed directly on timeline (no children)."""
        timeline = create_timeline(sample_data, uid="test")

        assert timeline.id == "test"
        # No children for single data source
        assert timeline.n_children == 0
        # Events are directly on the timeline
        # Exact count: 2 events
        assert timeline.n_events == 2

    def test_timeline_has_correct_length(self, sample_data: EventData):
        """Timeline length matches max event coordinate."""
        timeline = create_timeline(sample_data)

        # Events go from 0 to 480
        assert timeline.length.value == 480

    def test_flatten_mode_same_as_default(self, sample_data: EventData):
        """flatten=True behaves same as default for single data source."""
        timeline_default = create_timeline(sample_data)
        timeline_flatten = create_timeline(sample_data, flatten=True)

        assert timeline_default.n_children == timeline_flatten.n_children == 0
        assert timeline_default.n_events == timeline_flatten.n_events == 2


class TestCreateTimelineFromStore:
    """Tests for create_timeline with EventStore source.

    SingleStore has 1 data source -> events directly on timeline (no children).
    DictStore with 2+ data sources -> children created for each.
    """

    @pytest.fixture
    def single_store(self) -> SingleStore:
        """SingleStore with one data source (notes)."""
        data = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                },
                {
                    "id": "r1",
                    "temporal_type": "interval",
                    "event_type": "Rest",
                    "start": 960,
                    "end": 1440,
                },
            ],
            unit=TimeUnit.ticks,
        )
        return SingleStore(data, name="notes")

    @pytest.fixture
    def multi_store(self) -> DictStore:
        """DictStore with multiple data sources."""
        notes = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                },
            ],
            unit=TimeUnit.ticks,
        )
        measures = EventData.from_dicts(
            [
                {
                    "id": "m1",
                    "temporal_type": "interval",
                    "event_type": "Measure",
                    "start": 0,
                    "end": 1920,
                },
            ],
            unit=TimeUnit.ticks,
        )

        return DictStore({"notes": notes, "measures": measures})

    def test_single_store_no_children(self, single_store: SingleStore):
        """SingleStore (1 data source): events directly on timeline, no children."""
        timeline = create_timeline(single_store, uid="test")

        assert timeline.id == "test"
        # SingleStore has 1 data source -> no children
        assert timeline.n_children == 0
        # Events directly on timeline: 3 events
        assert timeline.n_events == 3

    def test_multi_store_creates_children(self, multi_store: DictStore):
        """DictStore (2 data sources): children created at offset 0."""
        timeline = create_timeline(multi_store, uid="test")

        assert timeline.id == "test"
        # DictStore has 2 data sources -> 2 children
        assert timeline.n_children == 2
        assert "notes" in _child_names(timeline)
        assert "measures" in _child_names(timeline)

    def test_store_filters_applied(self, single_store: SingleStore):
        """store_filters excludes filtered events (single store case)."""
        timeline = create_timeline(
            single_store,
            store_filters={"notes": {"event_type": "Note"}},
        )

        # Single store: events directly on timeline, no children
        assert timeline.n_children == 0
        # Exact count: 2 notes, rest excluded
        assert timeline.n_events == 2

    def test_store_filters_on_multi_store(self, multi_store: DictStore):
        """store_filters work on multi-store (children case)."""
        timeline = create_timeline(
            multi_store,
            store_filters={"notes": {"event_type": "Note"}},
        )

        # Multi-store: children created
        child = _child_named(timeline, "notes")
        # Exact count: 2 notes (all notes pass filter)
        assert child.n_events == 2

    def test_store_surface_places_single_store_events_directly(
        self, single_store: SingleStore
    ) -> None:
        """A one-data store returns its discrete logical events without children."""
        timeline = single_store.create_timeline()

        assert type(timeline) is DiscreteLogicalTimeline
        assert timeline.n_children == 0
        assert timeline.n_events == 3

    def test_store_surface_creates_named_children_for_multiple_data(
        self, multi_store: DictStore
    ) -> None:
        """A multi-data store retains each named event table as a child."""
        timeline = multi_store.create_timeline()

        assert type(timeline) is DiscreteLogicalTimeline
        assert list(timeline._children) == ["dlt_2", "dlt_3"]
        assert _child_names(timeline) == ["notes", "measures"]
        assert _child_named(timeline, "notes").n_events == 2
        assert _child_named(timeline, "measures").n_events == 1

    def test_store_surface_include_filter_selects_one_data_table(
        self, multi_store: DictStore
    ) -> None:
        """Selecting one source converts its events into the direct timeline payload."""
        timeline = multi_store.create_timeline(include_stores=["notes"])

        assert type(timeline) is DiscreteLogicalTimeline
        assert timeline.n_children == 0
        assert timeline.n_events == 2


class TestCreateTimelineIncludeExclude:
    """Tests for include_stores and exclude_stores parameters."""

    @pytest.fixture
    def multi_store(self) -> DictStore:
        """DictStore with multiple data sources for filtering tests."""
        notes = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                },
            ],
            unit=TimeUnit.ticks,
        )
        measures = EventData.from_dicts(
            [
                {
                    "id": "m1",
                    "temporal_type": "interval",
                    "event_type": "Measure",
                    "start": 0,
                    "end": 1920,
                },
            ],
            unit=TimeUnit.ticks,
        )
        controls = EventData.from_dicts(
            [
                {
                    "id": "c1",
                    "temporal_type": "instant",
                    "event_type": "Tempo",
                    "instant": 0,
                },
            ],
            unit=TimeUnit.ticks,
        )

        return DictStore({"notes": notes, "measures": measures, "controls": controls})

    def test_include_stores_limits_children(self, multi_store: DictStore):
        """include_stores only includes specified data."""
        timeline = create_timeline(
            multi_store,
            include_stores=["notes", "measures"],
        )

        # Only notes and measures included
        assert timeline.n_children == 2
        assert "notes" in _child_names(timeline)
        assert "measures" in _child_names(timeline)
        assert "controls" not in _child_names(timeline)

    def test_exclude_stores_removes_children(self, multi_store: DictStore):
        """exclude_stores removes specified data."""
        timeline = create_timeline(
            multi_store,
            exclude_stores=["controls"],
        )

        # Controls excluded
        assert "notes" in _child_names(timeline)
        assert "measures" in _child_names(timeline)
        assert "controls" not in _child_names(timeline)

    def test_empty_after_filtering_raises(self, multi_store: DictStore):
        """ValueError if all data filtered out."""
        with pytest.raises(ValueError, match="No data"):
            create_timeline(
                multi_store,
                include_stores=["nonexistent"],
            )


class TestCreateTimelineErrors:
    """Tests for error handling in create_timeline."""

    def test_invalid_source_type_raises(self):
        """TypeError for unsupported source type."""
        with pytest.raises(TypeError, match="must be EventStore"):
            create_timeline("invalid")  # type: ignore[arg-type]


class TestEventDataCreateTimeline:
    """Tests for EventData.create_timeline() method."""

    @pytest.fixture
    def sample_data(self) -> EventData:
        """EventData for testing."""
        return EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 480,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 480,
                    "end": 960,
                },
                {
                    "id": "r1",
                    "temporal_type": "interval",
                    "event_type": "Rest",
                    "start": 960,
                    "end": 1440,
                },
            ],
            unit=TimeUnit.ticks,
        )

    def test_creates_timeline_directly(self, sample_data: EventData):
        """create_timeline creates timeline with data's events."""
        timeline = sample_data.create_timeline(uid="direct")

        assert timeline.id == "direct"
        # Exact count: 3 events (2 notes + 1 rest)
        assert timeline.n_events == 3

    def test_with_filters(self, sample_data: EventData):
        """create_timeline with filters excludes filtered events."""
        timeline = sample_data.create_timeline(
            uid="filtered",
            filters={"event_type": "Note"},
        )

        # Exact count: 2 notes only
        assert timeline.n_events == 2

    def test_infers_correct_timeline_class(self, sample_data: EventData):
        """create_timeline creates appropriate timeline subclass."""
        timeline = sample_data.create_timeline()

        # ticks -> DiscreteLogicalTimeline
        assert isinstance(timeline, DiscreteLogicalTimeline)

    def test_seconds_data_creates_physical_timeline(self):
        """Seconds data creates ContinuousPhysicalTimeline."""
        data = EventData.from_dicts(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 0.0,
                },
            ],
            unit=TimeUnit.seconds,
        )

        timeline = data.create_timeline()

        assert isinstance(timeline, ContinuousPhysicalTimeline)


class TestTimelineChildrenStructure:
    """Tests for timeline structure with children.

    Uses DictStore (multi-data) to test parent-child relationships.
    SingleStore results in no children (events directly on timeline).
    """

    @pytest.fixture
    def multi_store(self) -> DictStore:
        """DictStore with notes and measures for children testing."""
        notes = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 100,
                },
                {
                    "id": "n2",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 100,
                    "end": 200,
                },
            ],
            unit=TimeUnit.ticks,
        )
        measures = EventData.from_dicts(
            [
                {
                    "id": "m1",
                    "temporal_type": "interval",
                    "event_type": "Measure",
                    "start": 0,
                    "end": 400,
                },
            ],
            unit=TimeUnit.ticks,
        )

        return DictStore({"notes": notes, "measures": measures})

    def test_parent_length_equals_max_child(self, multi_store: DictStore):
        """Parent timeline length is max of children lengths."""
        timeline = multi_store.create_timeline()

        # Measures end at 400, so parent length should be exactly 400
        assert timeline.length.value == 400

    def test_children_locked_after_addition(self, multi_store: DictStore):
        """Children are locked after being added to parent."""
        timeline = multi_store.create_timeline()

        child = _child_named(timeline, "notes")
        assert child.is_locked

    def test_children_maintain_own_coordinates(self, multi_store: DictStore):
        """Children maintain their own 0-based coordinate system."""
        timeline = multi_store.create_timeline()

        child = _child_named(timeline, "notes")

        # Child's origin is always 0
        assert child.origin.value == 0

        # Child's events start at their original coordinates
        events = list(child.events)
        first_event = events[0]
        assert first_event["start"]["value"] == 0

    def test_single_store_no_children(self):
        """SingleStore (1 data) puts events directly on timeline, no children."""
        data = EventData.from_dicts(
            [
                {
                    "id": "n1",
                    "temporal_type": "interval",
                    "event_type": "Note",
                    "start": 0,
                    "end": 100,
                },
            ],
            unit=TimeUnit.ticks,
        )
        store = SingleStore(data, name="notes")
        timeline = store.create_timeline()

        # SingleStore: no children, events directly on timeline
        assert timeline.n_children == 0
        assert timeline.n_events == 1
        assert timeline.length.value == 100
