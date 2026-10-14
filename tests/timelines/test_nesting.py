"""Tests for Timeline nesting (child timelines / segments).

This module tests:
- Adding children to timelines
- Child validation (unit matching, bounds checking)
- Iterating children (sorted, depth-first, breadth-first)
- Child locking behavior
- Segment events in EventStore

Validity Rationale:
    The TTA model specifies that timelines can contain nested "children"
    (segments) that share the same coordinate type. These tests verify:
    1. Children must have matching units (type safety)
    2. Children are locked upon embedding (immutability)
    3. Children appear as interval events in the parent's EventStore
    4. Traversal orders work correctly for hierarchical access
    5. Offset coordinates are correctly managed
"""

from __future__ import annotations

import time
from fractions import Fraction
from inspect import Parameter, signature

import pytest

from timetoalign.core import Coordinate, NumberType, TimeUnit, number_to_json
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    SegmentLine,
    Timeline,
)
from timetoalign.timelines.base import SEGMENT_EVENT_TYPE

# region Child Validation Tests


class TestChildValidation:
    """Test validation when adding children."""

    def test_validate_child_accepts_matching_units(self):
        """Children with matching units are accepted."""
        parent = ContinuousPhysicalTimeline(length=20.0, unit=TimeUnit.seconds)
        child = ContinuousPhysicalTimeline(length=5.0, unit=TimeUnit.seconds)

        # Should not raise
        parent.validate_child(child, offset=0.0)

    def test_validate_child_rejects_mismatched_units(self):
        """Children with different units are rejected."""
        parent = ContinuousPhysicalTimeline(length=20.0, unit=TimeUnit.seconds)
        child = ContinuousPhysicalTimeline(length=5.0, unit=TimeUnit.milliseconds)

        with pytest.raises(ValueError, match="does not match"):
            parent.validate_child(child, offset=0.0)

    def test_validate_child_rejects_non_timeline(self):
        """Non-Timeline objects are rejected."""
        parent = Timeline(length=20.0)

        with pytest.raises(TypeError, match="must be a Timeline"):
            parent.validate_child("not a timeline", offset=0.0)  # type: ignore

    def test_validate_child_rejects_duplicate(self):
        """Same child cannot be added twice."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0, uid="child")
        parent.add_child(child, offset=0.0)

        with pytest.raises(ValueError, match="already a child"):
            parent.validate_child(child, offset=10.0)

    def test_validate_child_rejects_negative_offset(self):
        """Negative offsets are rejected."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0)

        with pytest.raises(ValueError, match="negative"):
            parent.validate_child(child, offset=-1.0)


# endregion


# region Adding Children Tests


class TestAddChild:
    """Test add_child functionality."""

    def test_add_child_basic(self):
        """Basic child addition works."""
        parent = Timeline(length=20.0, uid="parent")
        child = Timeline(length=5.0, uid="child")

        parent.add_child(child, offset=0.0)

        assert parent.n_children == 1
        assert "child" in parent

    def test_add_child_stores_offset(self):
        """Child offset is stored correctly."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0, uid="child")
        parent.add_child(child, offset=10.0)

        offset = parent.get_child_offset("child")
        assert offset.value == 10.0
        assert offset.unit == parent.unit

    def test_add_child_locks_child(self):
        """Child is locked after being added."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0)

        assert not child.is_locked
        parent.add_child(child, offset=0.0)
        assert child.is_locked

    def test_add_child_creates_segment_event(self):
        """Adding child creates segment event in EventStore."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0, uid="child")
        parent.add_child(child, offset=10.0)

        # Segment events are internal bookkeeping; verify via private access
        segment_events = parent._events.filter(event_type=SEGMENT_EVENT_TYPE)
        assert len(segment_events) == 1

        # Check segment event properties
        segment_events = parent._events.filter(event_type=SEGMENT_EVENT_TYPE)
        assert len(segment_events) == 1

    def test_add_child_auto_expands_parent(self):
        """Parent auto-expands if child exceeds bounds."""
        parent = Timeline(length=10.0)
        child = Timeline(length=5.0)

        # Child at offset 8 would end at 13
        parent.add_child(child, offset=8.0)

        assert parent.length.value == 13.0

    def test_add_child_locked_parent_raises(self):
        """Locked parent rejects children that exceed bounds."""
        parent = Timeline(length=10.0, locked=True)
        child = Timeline(length=5.0)

        with pytest.raises(ValueError, match="locked"):
            parent.add_child(child, offset=8.0)

    def test_add_child_allow_expansion_overrides_lock(self):
        """allow_expansion=True overrides parent lock."""
        parent = Timeline(length=10.0, locked=True)
        child = Timeline(length=5.0)

        parent.add_child(child, offset=8.0, allow_expansion=True)
        assert parent.length.value == 13.0

    def test_add_multiple_children(self):
        """Multiple children can be added."""
        parent = Timeline(length=20.0)
        child1 = Timeline(length=5.0, uid="c1")
        child2 = Timeline(length=5.0, uid="c2")
        child3 = Timeline(length=5.0, uid="c3")

        parent.add_child(child1, offset=0.0)
        parent.add_child(child2, offset=5.0)
        parent.add_child(child3, offset=10.0)

        assert parent.n_children == 3

    def test_add_child_with_coordinate_offset(self):
        """Offset can be a Coordinate object."""
        parent = Timeline(length=20.0, unit=TimeUnit.seconds)
        child = Timeline(length=5.0, unit=TimeUnit.seconds)
        offset = Coordinate(10.0, TimeUnit.seconds)

        parent.add_child(child, offset=offset)

        stored_offset = parent.get_child_offset(child.id)
        assert stored_offset.value == 10.0


class TestAppendChild:
    """Test append_child placement, identity, and locking."""

    def test_locked_parent_raises_standard_error_before_mutating_child(self) -> None:
        """A locked parent rejects append without changing child identity."""
        parent = Timeline(length=5.0, uid="locked_parent", locked=True)
        child = Timeline(length=2.0, uid="original_id", name="Original")

        with pytest.raises(RuntimeError) as error:
            parent.append_child(child, uid="replacement_id", name="Replacement")

        assert str(error.value) == (
            "Cannot append child on locked timeline 'locked_parent'. "
            "A timeline is locked when embedded as a child, built with "
            "locked=True, or locked with lock()."
        )
        assert child.id == "original_id"
        assert child.name == "Original"

    def test_unlocked_parent_appends_at_previous_end_and_expands(self) -> None:
        """Append places a named child at the old end and expands the parent."""
        parent = Timeline(length=5.0)
        child = Timeline(length=2.5)

        parent.append_child(child, uid="ending", name="Ending")

        assert parent.get_child("ending") is child
        assert parent.get_child_offset("ending").value == 5.0
        assert parent.length.value == 7.5
        assert child.id == "ending"
        assert child.name == "Ending"

    def test_name_and_uid_are_keyword_only(self) -> None:
        """Identity overrides remain keyword-only parameters."""
        parameters = signature(Timeline.append_child).parameters

        assert parameters["name"].kind is Parameter.KEYWORD_ONLY
        assert parameters["uid"].kind is Parameter.KEYWORD_ONLY

    def test_embedded_segment_line_rejects_append_segment(self) -> None:
        """An embedded SegmentLine applies the append lock gate."""
        segment_line = SegmentLine[ContinuousLogicalTimeline](
            length=0,
            uid="embedded_line",
        )
        parent = ContinuousLogicalTimeline(length=4)
        parent.add_child(segment_line, offset=0)

        with pytest.raises(RuntimeError) as error:
            segment_line.append_segment(ContinuousLogicalTimeline(length=4))

        assert str(error.value) == (
            "Cannot append child on locked timeline 'embedded_line'. "
            "A timeline is locked when embedded as a child, built with "
            "locked=True, or locked with lock()."
        )


class TestAddChildDurationExactness:
    """Segment event duration stays an exact ratio for exact child lengths."""

    def test_integral_fraction_length_duration_is_exact(self):
        """A Fraction(2, 1) child length round-trips as an exact ratio."""
        parent = Timeline(
            length=Fraction(10, 1),
            unit=TimeUnit.seconds,
            number_type=NumberType.fraction,
        )
        child = Timeline(
            length=Fraction(2, 1),
            unit=TimeUnit.seconds,
            number_type=NumberType.fraction,
            uid="child",
        )
        parent.add_child(child, offset=Fraction(0, 1))

        duration = parent.to_dict(events=True)["events"][0]["duration"]
        assert duration == number_to_json(Fraction(2, 1))

    def test_non_integral_fraction_length_duration_is_exact(self):
        """A Fraction(7, 3) child length round-trips as an exact ratio."""
        parent = Timeline(
            length=Fraction(10, 1),
            unit=TimeUnit.seconds,
            number_type=NumberType.fraction,
        )
        child = Timeline(
            length=Fraction(7, 3),
            unit=TimeUnit.seconds,
            number_type=NumberType.fraction,
            uid="child",
        )
        parent.add_child(child, offset=Fraction(0, 1))

        duration = parent.to_dict(events=True)["events"][0]["duration"]
        assert duration == number_to_json(Fraction(7, 3))


class TestCreateChild:
    """Test child construction and concrete class selection."""

    @pytest.mark.parametrize(
        ("parent_class", "length"),
        [
            (ContinuousLogicalTimeline, Fraction(100, 1)),
            (ContinuousPhysicalTimeline, 100.0),
            (ContinuousGraphicalTimeline, 100.0),
            (DiscreteLogicalTimeline, 100),
            (DiscretePhysicalTimeline, 100),
            (DiscreteGraphicalTimeline, 100),
        ],
    )
    def test_create_child_inherits_exact_concrete_class(
        self,
        parent_class: type[Timeline],
        length: int | float | Fraction,
    ) -> None:
        """Each concrete timeline type creates a child of its exact class."""
        parent = parent_class(length=length)

        child = parent.create_child(length=10, uid="child")

        assert type(child) is parent_class
        assert parent.get_child_offset("child").value == 0

    def test_create_child_accepts_coordinate_length_and_offset(self) -> None:
        """Coordinate objects remain accepted for child length and offset."""
        parent = ContinuousPhysicalTimeline(length=10.0, unit=TimeUnit.seconds)

        child = parent.create_child(
            length=Coordinate(2.0, TimeUnit.seconds),
            offset=Coordinate(3.0, TimeUnit.seconds),
            uid="coordinate_child",
        )

        assert child.length.value == 2.0
        assert parent.get_child_offset("coordinate_child").value == 3.0


class TestCreateChildrenFromBoundaries:
    """Test contiguous child construction from boundary coordinates."""

    def test_generated_children_have_exact_names_offsets_and_lengths(self) -> None:
        """Generated children use exact interval names, offsets, and lengths."""
        parent = ContinuousPhysicalTimeline(length=90.0, unit=TimeUnit.seconds)
        parent.add_events(
            [
                {
                    "id": "source",
                    "temporal_type": "instant",
                    "event_type": "Marker",
                    "instant": 15.0,
                }
            ]
        )

        children = parent.create_children_from_boundaries(
            [0.0, 30.0, 60.0, 90.0],
            prefix="movement",
        )

        assert [child.id for child in children] == [
            "movement_1",
            "movement_2",
            "movement_3",
        ]
        assert [child.name for child in children] == [
            "movement_1",
            "movement_2",
            "movement_3",
        ]
        assert [parent.get_child_offset(child.id).value for child in children] == [
            0.0,
            30.0,
            60.0,
        ]
        assert [child.length.value for child in children] == [30.0, 30.0, 30.0]
        assert [child.n_events for child in children] == [0, 0, 0]

    def test_explicit_names_are_used_for_ids_and_names(self) -> None:
        """Explicit boundary names become both child IDs and names."""
        parent = ContinuousPhysicalTimeline(length=90.0)

        children = parent.create_children_from_boundaries(
            [0.0, 30.0, 60.0, 90.0],
            names=["opening", "middle", "closing"],
        )

        assert [child.id for child in children] == ["opening", "middle", "closing"]
        assert [child.name for child in children] == [
            "opening",
            "middle",
            "closing",
        ]

    @pytest.mark.parametrize(
        ("boundaries", "expected_message"),
        [
            ([50.0], "Need at least 2 boundary coordinates, got 1"),
            (
                [0.0, 60.0, 30.0, 100.0],
                "Boundaries must be monotonically increasing: "
                "boundaries[1]=60 >= boundaries[2]=30",
            ),
        ],
    )
    def test_boundary_errors_match_region_variant(
        self,
        boundaries: list[float],
        expected_message: str,
    ) -> None:
        """Child and region boundary creation report identical validation errors."""
        child_parent = ContinuousPhysicalTimeline(length=100.0)
        region_parent = ContinuousPhysicalTimeline(length=100.0)

        with pytest.raises(ValueError) as child_error:
            child_parent.create_children_from_boundaries(boundaries)
        with pytest.raises(ValueError) as region_error:
            region_parent.create_regions_from_boundaries(boundaries)

        assert str(child_error.value) == str(region_error.value)
        assert str(child_error.value) == expected_message

    def test_children_tile_parent_exactly(self) -> None:
        """Boundary-created children cover the parent without gaps or overlaps."""
        parent = ContinuousPhysicalTimeline(length=90.0)

        children = parent.create_children_from_boundaries(
            [0.0, 30.0, 60.0, 90.0],
            prefix="movement",
        )
        extents = [
            (
                parent.get_child_offset(child.id).value,
                parent.get_child_offset(child.id).value + child.length.value,
            )
            for child in children
        ]

        assert extents == [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]
        assert extents[0][0] == 0.0
        assert extents[-1][1] == parent.length.value
        assert parent.is_segment_line() is True


# endregion


# region Get Child Tests


class TestGetChild:
    """Test child retrieval."""

    def test_get_child_by_id(self):
        """get_child retrieves child by ID."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0, uid="my_child")
        parent.add_child(child, offset=0.0)

        retrieved = parent.get_child("my_child")
        assert retrieved is child

    def test_get_child_nonexistent_raises(self):
        """get_child raises KeyError for unknown ID."""
        parent = Timeline(length=20.0)

        with pytest.raises(KeyError, match="nonexistent"):
            parent.get_child("nonexistent")

    def test_get_child_offset_by_id(self):
        """get_child_offset retrieves offset by ID."""
        parent = Timeline(length=20.0)
        child = Timeline(length=5.0, uid="my_child")
        parent.add_child(child, offset=7.5)

        offset = parent.get_child_offset("my_child")
        assert offset.value == 7.5

    def test_get_child_offset_nonexistent_raises(self):
        """get_child_offset raises KeyError for unknown ID."""
        parent = Timeline(length=20.0)

        with pytest.raises(KeyError, match="nonexistent"):
            parent.get_child_offset("nonexistent")


# endregion


# region Iteration Tests


class TestIterChildren:
    """Test child iteration with different traversal orders."""

    def test_iter_children_empty(self):
        """Iterating children of childless timeline yields nothing."""
        parent = Timeline(length=20.0)
        children = list(parent.iter_children())
        assert children == []

    def test_iter_children_sorted_by_offset(self):
        """Sorted order yields children by offset."""
        parent = Timeline(length=30.0)
        c1 = Timeline(length=5.0, uid="c1")
        c2 = Timeline(length=5.0, uid="c2")
        c3 = Timeline(length=5.0, uid="c3")

        # Add in non-sorted order
        parent.add_child(c2, offset=10.0)
        parent.add_child(c3, offset=20.0)
        parent.add_child(c1, offset=0.0)

        children = list(parent.iter_children(order="sorted"))
        child_ids = [c.id for _, c in children]
        assert child_ids == ["c1", "c2", "c3"]

    def test_iter_children_include_self(self):
        """include_self=True yields parent first."""
        parent = Timeline(length=20.0, uid="parent")
        child = Timeline(length=5.0, uid="child")
        parent.add_child(child, offset=0.0)

        results = list(parent.iter_children(include_self=True))
        assert len(results) == 2
        assert results[0][1].id == "parent"
        assert results[0][0].value == 0  # Parent offset is 0

    def test_iter_children_nested_sorted(self, nested_timeline_structure):
        """Sorted iteration on nested structure yields correct order."""
        parent = nested_timeline_structure

        # Sorted order: by offset, including recursion
        results = list(parent.iter_children(order="sorted"))
        offsets = [offset.value for offset, _ in results]

        # Expected: child_a(0), grandchild_a1(0+1=1), child_b(10),
        #           grandchild_b1(10+1=11), grandchild_b2(10+5=15)
        assert offsets == [0.0, 1.0, 10.0, 11.0, 15.0]

    def test_iter_children_nested_breadth_first(self, nested_timeline_structure):
        """Breadth-first iteration yields levels in order."""
        parent = nested_timeline_structure

        results = list(parent.iter_children(order="breadth_first"))
        child_ids = [c.id for _, c in results]

        # Level 1: child_a, child_b
        # Level 2: grandchild_a1, grandchild_b1, grandchild_b2
        assert child_ids[:2] == ["child_a", "child_b"]
        assert set(child_ids[2:]) == {"grandchild_a1", "grandchild_b1", "grandchild_b2"}

    def test_iter_children_nested_depth_first(self, nested_timeline_structure):
        """Depth-first iteration follows branches."""
        parent = nested_timeline_structure

        results = list(parent.iter_children(order="depth_first"))
        child_ids = [c.id for _, c in results]

        # child_a first, then its descendants, then child_b, then its descendants
        # Order within siblings may vary by dict order
        assert child_ids[0] in ["child_a", "child_b"]
        if child_ids[0] == "child_a":
            assert child_ids[1] == "grandchild_a1"

    def test_iter_children_recursion_limit(self, nested_timeline_structure):
        """recursion_limit controls depth of iteration."""
        parent = nested_timeline_structure

        # Limit 1: only direct children
        results_1 = list(parent.iter_children(recursion_limit=1))
        assert len(results_1) == 2  # child_a, child_b

        # Limit 2: children and grandchildren
        results_2 = list(parent.iter_children(recursion_limit=2))
        assert len(results_2) == 5  # 2 children + 3 grandchildren

    def test_iter_children_recursion_limit_zero(self, nested_timeline_structure):
        """recursion_limit=0 yields nothing (unless include_self)."""
        parent = nested_timeline_structure

        results = list(parent.iter_children(recursion_limit=0))
        assert results == []

        results_with_self = list(
            parent.iter_children(recursion_limit=0, include_self=True)
        )
        assert len(results_with_self) == 1

    def test_iter_children_offsets_are_absolute(self, nested_timeline_structure):
        """Offsets in iteration are relative to the root parent."""
        parent = nested_timeline_structure

        results = dict(
            (c.id, offset.value) for offset, c in parent.iter_children(order="sorted")
        )

        # grandchild_a1 is at offset 1 within child_a, which is at offset 0
        # So absolute offset is 0 + 1 = 1
        assert results["grandchild_a1"] == 1.0

        # grandchild_b1 is at offset 1 within child_b, which is at offset 10
        # So absolute offset is 10 + 1 = 11
        assert results["grandchild_b1"] == 11.0


# endregion


# region Unit Validation Tests


class TestUnitValidation:
    """Test that unit mismatches are caught."""

    def test_logical_cannot_contain_physical(self):
        """Logical timeline cannot contain physical timeline."""
        logical = ContinuousLogicalTimeline(length=8.0, unit=TimeUnit.quarters)
        physical = ContinuousPhysicalTimeline(length=5.0, unit=TimeUnit.seconds)

        with pytest.raises(ValueError, match="does not match"):
            logical.add_child(physical, offset=0.0)

    def test_physical_cannot_contain_logical(self):
        """Physical timeline cannot contain logical timeline."""
        physical = ContinuousPhysicalTimeline(length=20.0, unit=TimeUnit.seconds)
        logical = ContinuousLogicalTimeline(length=4.0, unit=TimeUnit.quarters)

        with pytest.raises(ValueError, match="does not match"):
            physical.add_child(logical, offset=0.0)

    def test_same_domain_different_unit_rejected(self):
        """Same domain but different units are rejected."""
        seconds_tl = ContinuousPhysicalTimeline(length=20.0, unit=TimeUnit.seconds)
        ms_tl = ContinuousPhysicalTimeline(length=5000.0, unit=TimeUnit.milliseconds)

        with pytest.raises(ValueError, match="does not match"):
            seconds_tl.add_child(ms_tl, offset=0.0)

    def test_discrete_and_continuous_same_domain_rejected(self):
        """Discrete and continuous timelines with different units rejected."""
        ticks_tl = DiscreteLogicalTimeline(length=1920, unit=TimeUnit.ticks)
        quarters_tl = ContinuousLogicalTimeline(length=4.0, unit=TimeUnit.quarters)

        with pytest.raises(ValueError, match="does not match"):
            ticks_tl.add_child(quarters_tl, offset=0)


# endregion


# region Performance Tests


class TestNestingPerformance:
    """Performance tests for nesting operations."""

    def test_add_many_children_performance(self, profiler):
        """Benchmark adding many children."""
        n_children = 1000
        parent = Timeline(length=float(n_children * 2))
        children = [Timeline(length=1.0, uid=f"c_{i}") for i in range(n_children)]

        start = time.perf_counter()
        for i, child in enumerate(children):
            parent.add_child(child, offset=float(i * 2))
        elapsed = time.perf_counter() - start

        profiler.record("add_1000_children", elapsed)

        assert parent.n_children == n_children
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Adding {n_children} children took {elapsed:.2f}s"

    def test_iterate_deep_hierarchy_performance(self, profiler):
        """Benchmark iterating a deep hierarchy."""
        # Create a chain of 100 nested timelines
        depth = 100
        root = Timeline(length=float(depth), uid="root")
        current = root

        for i in range(depth - 1):
            child = Timeline(length=float(depth - i - 1), uid=f"level_{i + 1}")
            current.add_child(child, offset=1.0)
            current = child

        start = time.perf_counter()
        results = list(root.iter_children(order="depth_first"))
        elapsed = time.perf_counter() - start

        profiler.record("iterate_depth_100", elapsed)

        assert len(results) == depth - 1
        # Should be fast even for deep hierarchies
        assert elapsed < 1.0, f"Iterating depth {depth} took {elapsed:.2f}s"


# endregion


# region Use Conversion Map Tests


class TestAddChildWithConversionMap:
    """Test add_child with use_conversion_map parameter.

    When the child uses a different unit than the parent, the
    ``use_conversion_map`` parameter finds a C-Map on the parent,
    inverts it, and derives a converted copy of the child in the
    parent's unit.
    """

    def test_auto_select_conversion_map(self):
        """use_conversion_map=True auto-selects parent's C-Map."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=10.0, unit=TimeUnit.seconds, uid="notes"
        )
        child.add_events(
            [
                {"event_type": "Note", "start": 0.0, "end": 0.5},
                {"event_type": "Note", "start": 1.0, "end": 1.5},
            ]
        )

        parent.add_child(child, offset=0, use_conversion_map=True)

        assert parent.n_children == 1
        converted = parent.get_child("dpt_1")
        assert converted.unit == TimeUnit.samples
        assert converted.length.value == 441000.0

    def test_converted_child_has_correct_id_and_name(self):
        """A converted child's id is minted; its name links to the original."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="my_notes"
        )

        parent.add_child(child, offset=0, use_conversion_map=True)

        converted = parent.get_child("dpt_1")
        assert converted.id == "dpt_1"
        assert converted.name == "my_notes [samples]"

    def test_converted_child_events_have_correct_coordinates(self):
        """Event coordinates are converted from child unit to parent unit."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=10.0, unit=TimeUnit.seconds, uid="notes"
        )
        child.add_events(
            [
                {"event_type": "Note", "start": 1.0, "end": 2.0},
            ]
        )

        parent.add_child(child, offset=0, use_conversion_map=True)

        converted = parent.get_child("dpt_1")
        events = list(converted._events)
        assert len(events) == 1
        event = events[0]
        # 1.0 seconds * 44100 = 44100 samples
        start_val = event["start"]
        if isinstance(start_val, dict):
            start_val = start_val["value"]
        assert start_val == 44100.0
        # 2.0 seconds * 44100 = 88200 samples
        end_val = event["end"]
        if isinstance(end_val, dict):
            end_val = end_val["value"]
        assert end_val == 88200.0

    def test_original_child_is_not_modified(self):
        """The original child timeline is not locked or mutated."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        parent.add_child(child, offset=0, use_conversion_map=True)

        # Original child should not be locked (the converted copy is)
        assert not child.is_locked
        assert child.unit == TimeUnit.seconds
        assert child.length.value == 5.0

    def test_use_conversion_map_by_string_target_unit(self):
        """use_conversion_map='seconds' finds the map by target unit name."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        parent.add_child(child, offset=0, use_conversion_map="seconds")

        assert parent.n_children == 1
        assert "dpt_1" in parent
        assert parent.get_child("dpt_1").name == "notes [samples]"

    def test_use_conversion_map_by_cmap_object(self):
        """use_conversion_map accepts a ConversionMap object directly."""
        from timetoalign.maps.convenience import SamplesToSeconds

        cmap = SamplesToSeconds(sample_rate=44100)
        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        # Attach the map (so parent knows about it for timestamp system)
        parent.add_conversion_map(cmap)

        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        parent.add_child(child, offset=0, use_conversion_map=cmap)

        assert parent.n_children == 1

    def test_no_conversion_map_raises_on_unit_mismatch(self):
        """Without use_conversion_map, mismatched units still raise."""
        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        with pytest.raises(ValueError, match="does not match"):
            parent.add_child(child, offset=0)

    def test_auto_select_raises_when_no_cmap_matches(self):
        """use_conversion_map=True raises if no C-Map targets child's unit."""
        parent = Timeline(length=100, unit=TimeUnit.samples, uid="audio")
        # No conversion map attached!
        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        with pytest.raises(ValueError, match="no C-Map"):
            parent.add_child(child, offset=0, use_conversion_map=True)

    def test_same_unit_child_ignores_conversion_map(self):
        """If units already match, use_conversion_map is a no-op."""
        parent = Timeline(length=20.0, unit=TimeUnit.seconds, uid="parent")
        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="child"
        )

        # Even with use_conversion_map=True, same-unit child is added as-is
        parent.add_child(child, offset=0, use_conversion_map=True)

        assert parent.n_children == 1
        # The child is added with its original ID (no [unit] suffix)
        assert "child" in parent

    def test_allow_expansion_works_with_conversion(self):
        """allow_expansion works when combined with use_conversion_map."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=44100, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        # Child in seconds is 5s = 220500 samples, exceeds parent length
        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        parent.add_child(child, offset=0, use_conversion_map=True, allow_expansion=True)

        # Parent should have expanded
        assert parent.length.value == 220500.0
        assert parent.n_children == 1

    def test_converted_child_is_locked(self):
        """The converted copy (not the original) is locked after embedding."""
        from timetoalign.maps.convenience import SamplesToSeconds

        parent = Timeline(length=441000, unit=TimeUnit.samples, uid="audio")
        parent.add_conversion_map(SamplesToSeconds(sample_rate=44100))

        child = ContinuousPhysicalTimeline(
            length=5.0, unit=TimeUnit.seconds, uid="notes"
        )

        parent.add_child(child, offset=0, use_conversion_map=True)

        converted = parent.get_child("dpt_1")
        assert converted.is_locked
        assert not child.is_locked


# endregion
