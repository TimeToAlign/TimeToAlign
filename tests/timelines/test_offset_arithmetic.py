"""Tests for parent-child offset arithmetic.

Verifies that parent-child coordinate conversion uses exact offset
arithmetic (addition/subtraction) rather than InterpolationMap interpolation.
This eliminates floating-point drift for hierarchical coordinate resolution.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign import Coordinate, IdCoordinate, Timeline, TimeUnit
from timetoalign.timelines import ContinuousPhysicalTimeline

# region Fixtures


@pytest.fixture
def parent() -> ContinuousPhysicalTimeline:
    """Parent timeline: 100 seconds."""
    return ContinuousPhysicalTimeline(
        length=100.0,
        unit="seconds",
        uid="parent",
    )


@pytest.fixture
def child() -> ContinuousPhysicalTimeline:
    """Child timeline: 30 seconds."""
    return ContinuousPhysicalTimeline(
        length=30.0,
        unit="seconds",
        uid="child",
    )


@pytest.fixture
def grandchild() -> ContinuousPhysicalTimeline:
    """Grandchild timeline: 10 seconds."""
    return ContinuousPhysicalTimeline(
        length=10.0,
        unit="seconds",
        uid="grandchild",
    )


# endregion


# region Exact Offset Arithmetic Tests


class TestParentToChildTransfer:
    """Test parent -> child coordinate conversion via exact offset arithmetic."""

    def test_exact_parent_to_child(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify parent->child is exact subtraction, no float drift."""
        parent.add_child(child, offset=10.0)

        # parent coord 10.0 -> child coord 0.0
        result = parent._get_child_coordinate("child", 10.0)
        assert result == 0.0

        # parent coord 25.0 -> child coord 15.0 (exact)
        result = parent._get_child_coordinate("child", 25.0)
        assert result == 15.0

        # Fractional coordinates: exact arithmetic, no drift
        result = parent._get_child_coordinate("child", 25.123456789012345)
        assert result == 25.123456789012345 - 10.0

    def test_exact_child_to_parent(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify child->parent is exact addition, no float drift."""
        parent.add_child(child, offset=10.0)

        # child coord 0.0 -> parent coord 10.0
        result = parent._get_parent_coordinate_from_child("child", 0.0)
        assert result == 10.0

        # child coord 15.0 -> parent coord 25.0 (exact)
        result = parent._get_parent_coordinate_from_child("child", 15.0)
        assert result == 25.0

    def test_out_of_bounds_returns_none(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify out-of-bounds parent coordinates return None."""
        parent.add_child(child, offset=10.0)

        # Before child span
        assert parent._get_child_coordinate("child", 5.0) is None

        # After child span [10, 40)
        assert parent._get_child_coordinate("child", 40.0) is None

        # Negative
        assert parent._get_child_coordinate("child", -1.0) is None

    def test_unknown_child_returns_none(
        self,
        parent: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify unknown child ID returns None."""
        assert parent._get_child_coordinate("nonexistent", 0.0) is None

    def test_unknown_child_raises_keyerror(
        self,
        parent: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify unknown child ID raises KeyError for parent coord."""
        with pytest.raises(KeyError):
            parent._get_parent_coordinate_from_child("nonexistent", 0.0)


# endregion


# region Recursive Offset Tests


class TestRecursiveOffset:
    """Test grandchild (nested) coordinate resolution via offset arithmetic."""

    def test_recursive_offset_exact(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
        grandchild: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify grandchild offset is exact through two levels."""
        parent.add_child(child, offset=10.0)
        child.add_child(grandchild, offset=5.0)

        assert parent.get_coordinate(
            IdCoordinate(0.0, TimeUnit.seconds, "grandchild"), format="coordinate"
        ) == Coordinate(15.0, TimeUnit.seconds)
        assert parent.get_coordinate(
            IdCoordinate(5.0, TimeUnit.seconds, "grandchild"), format="coordinate"
        ) == Coordinate(20.0, TimeUnit.seconds)
        assert parent._get_parent_coordinate_from_child("grandchild", 5.0) == 20.0

    def test_recursive_fraction_offset_exact(self) -> None:
        """Verify rational offsets compose without float coercion."""
        parent = Timeline(length=10, unit=TimeUnit.quarters, uid="parent")
        child = Timeline(length=4, unit=TimeUnit.quarters, uid="child")
        grandchild = Timeline(length=2, unit=TimeUnit.quarters, uid="grandchild")
        child.add_child(grandchild, offset=Fraction(3, 2))
        parent.add_child(child, offset=Fraction(9, 2))

        assert parent.get_coordinate(
            IdCoordinate(Fraction(0), TimeUnit.quarters, "grandchild"),
            format="coordinate",
        ) == Coordinate(Fraction(6), TimeUnit.quarters)
        assert parent._get_parent_coordinate_from_child(
            "grandchild", Fraction(1, 2)
        ) == Fraction(13, 2)


# endregion


# region TimeStamp Integration Tests


class TestTimeStampOffsetArithmetic:
    """Test that TimeStamp cross-section uses offset arithmetic."""

    def test_timestamp_uses_offset_arithmetic(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify typed timestamp retrieval uses child offset arithmetic."""
        parent.add_child(child, offset=10.0)

        ts = parent.get_timestamp(25.0)
        assert ts.axis == IdCoordinate(25.0, TimeUnit.seconds, "parent")
        assert ts.get_coordinate_for("child", format="float") == 15.0

    def test_timestamp_out_of_bounds_returns_none(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify TimeStamp raises for coordinates outside a child span."""
        parent.add_child(child, offset=10.0)

        ts = parent.get_timestamp(5.0)
        with pytest.raises(KeyError):
            ts.get_coordinate_for("child")

    def test_timestamp_boundary_values(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify boundary values: start inclusive, end exclusive."""
        parent.add_child(child, offset=10.0)

        # Start: inclusive (child coord = 0.0)
        ts_start = parent.get_timestamp(10.0)
        assert ts_start.get_coordinate_for("child", format="float") == 0.0

        # Just before end: in range
        ts_near_end = parent.get_timestamp(39.9)
        assert ts_near_end.get_coordinate_for("child", format="float") == 29.9

        # End: exclusive (child coord = 30.0 == length, out of range)
        ts_end = parent.get_timestamp(40.0)
        with pytest.raises(KeyError):
            ts_end.get_coordinate_for("child")

    def test_no_interpolation_maps_attribute(
        self,
        parent: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify that Timeline has no _interpolation_maps attribute."""
        assert not hasattr(parent, "_interpolation_maps")


# endregion
