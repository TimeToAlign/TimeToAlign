"""Tests for parent-child offset arithmetic.

Verifies that parent-child coordinate conversion uses exact offset
arithmetic (addition/subtraction) rather than InterpolationMap interpolation.
This eliminates floating-point drift for hierarchical coordinate resolution.
"""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import pytest

from timetoalign import AlignmentBundle, Coordinate, IdCoordinate, Timeline, TimeUnit
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
)

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

        # After child span [10, 40]
        assert parent._get_child_coordinate("child", 40.5) is None

        # Negative
        assert parent._get_child_coordinate("child", -1.0) is None

        # The end of a child no sibling follows is on the child
        assert parent._get_child_coordinate("child", 40.0) == 30.0

    def test_shared_boundary_belongs_to_the_later_sibling(
        self,
        parent: ContinuousPhysicalTimeline,
        child: ContinuousPhysicalTimeline,
        grandchild: ContinuousPhysicalTimeline,
    ) -> None:
        """Where one child ends and the next starts, only the next holds it.

        ``child`` spans [10, 40] and ``grandchild``, here a second child,
        starts at 40: parent 40 is ``grandchild`` 0 and not ``child`` 30, in
        the stamp and in the table alike. ``grandchild`` ends at 50 with no
        sibling after it, so parent 50 is its 10.
        """
        parent.add_child(child, offset=10.0)
        parent.add_child(grandchild, offset=40.0)

        assert parent._get_child_coordinate("child", 40.0) is None
        assert parent._get_child_coordinate("grandchild", 40.0) == 0.0
        assert parent._get_child_coordinate("grandchild", 50.0) == 10.0
        stamps = [
            {
                timeline_id: coordinate.value
                for timeline_id, coordinate in parent.get_timestamp_at(
                    position
                ).coordinates.items()
            }
            for position in (40.0, 50.0)
        ]
        assert stamps == [
            {"parent": 40.0, "grandchild": 0.0},
            {"parent": 50.0, "grandchild": 10.0},
        ]
        table = parent.get_timestamp_table([40.0, 50.0], format="dataframe")
        assert list(table.columns) == [
            "parent (seconds)",
            "child (seconds)",
            "grandchild (seconds)",
        ]
        assert table["parent (seconds)"].tolist() == [40.0, 50.0]
        assert table["child (seconds)"].isna().tolist() == [True, True]
        assert table["grandchild (seconds)"].tolist() == [0.0, 10.0]

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
        """Verify boundary values: start inclusive, a lone child's end inclusive."""
        parent.add_child(child, offset=10.0)

        # Start: inclusive (child coord = 0.0)
        ts_start = parent.get_timestamp(10.0)
        assert ts_start.get_coordinate_for("child", format="float") == 0.0

        # Just before end: in range
        ts_near_end = parent.get_timestamp(39.9)
        assert ts_near_end.get_coordinate_for("child", format="float") == 29.9

        # End: inclusive, since no sibling starts there (child coord 30.0)
        ts_end = parent.get_timestamp(40.0)
        assert ts_end.get_coordinate_for("child", format="float") == 30.0

        # Past the end: out of range
        ts_after = parent.get_timestamp(40.5)
        with pytest.raises(KeyError):
            ts_after.get_coordinate_for("child")

    def test_no_interpolation_maps_attribute(
        self,
        parent: ContinuousPhysicalTimeline,
    ) -> None:
        """Verify that Timeline has no _interpolation_maps attribute."""
        assert not hasattr(parent, "_interpolation_maps")


# endregion


# region Containment Along the Path


def _present(stamp) -> dict[str, object]:
    """The values a stamp names, by timeline id."""
    return {
        timeline_id: coordinate.value
        for timeline_id, coordinate in stamp.coordinates.items()
    }


def _row(timeline: Timeline, position: object) -> dict[str, object]:
    """The non-null values of a timeline's table row at one position."""
    table = timeline.get_timestamp_table([position], format="dataframe")
    return {
        column.rsplit(" (", 1)[0]: table[column].iloc[0]
        for column in table.columns
        if not pd.isna(table[column].iloc[0])
    }


def _sections() -> dict[str, ContinuousLogicalTimeline]:
    """``P`` (8) with ``A`` at 0 and ``B`` at 4, each filled by one child."""
    timelines = {
        uid: ContinuousLogicalTimeline(length=length, uid=uid)
        for uid, length in (("P", 8), ("A", 4), ("B", 4), ("A1", 4), ("B1", 4))
    }
    timelines["A"].add_child(timelines["A1"], 0)
    timelines["B"].add_child(timelines["B1"], 0)
    timelines["P"].add_child(timelines["A"], 0)
    timelines["P"].add_child(timelines["B"], 4)
    return timelines


class TestContainmentAlongThePath:
    """A position reaches a descendant only through every timeline on the way."""

    def test_grandchild_at_shared_boundary(self) -> None:
        """``A1`` holds its own end inside ``A``, but ``A`` leaves 4 to ``B``."""
        tl = _sections()
        expected = {"P": 4, "B": 0, "B1": 0}
        assert _present(tl["P"].get_timestamp_at(4)) == expected
        assert _row(tl["P"], 4) == expected
        table = tl["P"].get_timestamp_table([4], format="dataframe")
        assert table["A (quarters)"].isna().tolist() == [True]
        assert table["A1 (quarters)"].isna().tolist() == [True]
        assert tl["P"]._get_child_coordinate("A1", 4) is None
        assert [child.id for child in tl["P"].get_children_at(4)] == ["B"]

    def test_bundle_follows_the_path_from_every_holder(self) -> None:
        """The bundle's stamp is one set from ``P`` 4, ``B`` 0 and ``B1`` 0."""
        tl = _sections()
        bundle = AlignmentBundle()
        bundle.add_timeline(tl["P"])
        for timeline_id, position in (("P", 4), ("B", 0), ("B1", 0)):
            stamp = bundle.get_matchstamp_at(position, timeline_id)
            assert _present(stamp) == {"P": 4, "B": 0, "B1": 0}

    def test_own_end_is_held_by_the_queried_timeline(self) -> None:
        """Queried on ``A``, its end is its own, and ``A1``'s with it."""
        tl = _sections()
        assert _present(tl["A"].get_timestamp_at(4)) == {"A": 4, "A1": 4}
        assert _row(tl["A"], 4) == {"A": 4, "A1": 4}
        bundle = AlignmentBundle()
        bundle.add_timeline(tl["P"])
        assert _present(bundle.get_matchstamp_at(4, "A")) == {
            "A": 4,
            "P": 4,
            "A1": 4,
            "B": 0,
            "B1": 0,
        }

    def test_lone_child_end_listed_by_get_children_at(self) -> None:
        """A child ending inside its parent, nothing after it, holds its end."""
        parent = ContinuousLogicalTimeline(length=8, uid="P")
        section = ContinuousLogicalTimeline(length=4, uid="S")
        parent.add_child(section, 0)
        assert _present(parent.get_timestamp_at(4)) == {"P": 4, "S": 4}
        assert [child.id for child in parent.get_children_at(4)] == ["S"]


class TestExactTableContainment:
    """On an exact axis the table decides containment as the stamp does."""

    def test_end_reached_by_exact_sum(self) -> None:
        """``7/10 + 1/10`` is ``4/5`` exactly; ``0.7 + 0.1`` is not ``0.8``."""
        parent = ContinuousLogicalTimeline(length=1, uid="P")
        child = ContinuousLogicalTimeline(length=Fraction(1, 10), uid="C")
        parent.add_child(child, Fraction(7, 10))
        expected = {"P": Fraction(4, 5), "C": Fraction(1, 10)}
        assert _present(parent.get_timestamp_at(Fraction(4, 5))) == expected
        assert _row(parent, Fraction(4, 5)) == expected

    def test_shared_boundary_at_an_inexact_float(self) -> None:
        """At ``3/10`` ``B`` starts and ``A`` (``1/10`` + ``1/5``) ends."""
        parent = ContinuousLogicalTimeline(length=1, uid="P")
        first = ContinuousLogicalTimeline(length=Fraction(1, 5), uid="A")
        second = ContinuousLogicalTimeline(length=Fraction(1, 5), uid="B")
        parent.add_child(first, Fraction(1, 10))
        parent.add_child(second, Fraction(3, 10))
        expected = {"P": Fraction(3, 10), "B": 0}
        assert _present(parent.get_timestamp_at(Fraction(3, 10))) == expected
        assert _row(parent, Fraction(3, 10)) == expected


def _three_levels(cls: type[Timeline], number: object) -> Timeline:
    """A tree of three levels, siblings at shared and unshared boundaries.

    On the root (length 1): ``A`` [1/10, 4/10] holding ``A1`` [0, 1/10] (in
    turn holding ``A11`` filling it) and ``A2`` [1/10, 3/10]; ``B``
    [4/10, 7/10] filled by ``B1``; ``C`` [7/10, 9/10]. *number* writes each
    value on the axis.
    """

    def make(uid: str, length: Fraction) -> Timeline:
        return cls(length=number(length), uid=uid)

    root = make("P", Fraction(1))
    a, b, c = (
        make(u, n)
        for u, n in (
            ("A", Fraction(3, 10)),
            ("B", Fraction(3, 10)),
            ("C", Fraction(1, 5)),
        )
    )
    a1, a2, a11, b1 = (
        make(u, n)
        for u, n in (
            ("A1", Fraction(1, 10)),
            ("A2", Fraction(1, 5)),
            ("A11", Fraction(1, 10)),
            ("B1", Fraction(3, 10)),
        )
    )
    a1.add_child(a11, number(0))
    a.add_child(a1, number(0))
    a.add_child(a2, number(Fraction(1, 10)))
    b.add_child(b1, number(0))
    root.add_child(a, number(Fraction(1, 10)))
    root.add_child(b, number(Fraction(2, 5)))
    root.add_child(c, number(Fraction(7, 10)))
    return root


@pytest.mark.parametrize(
    ("cls", "number"),
    [
        (ContinuousLogicalTimeline, Fraction),
        (ContinuousPhysicalTimeline, float),
        (DiscretePhysicalTimeline, lambda value: int(value * 1000)),
    ],
    ids=["quarters", "seconds", "samples"],
)
def test_lanes_agree_at_every_position(cls: type[Timeline], number: object) -> None:
    """Stamp, table row and bundle stamp name the same values everywhere."""
    root = _three_levels(cls, number)
    bundle = AlignmentBundle()
    bundle.add_timeline(root)
    positions = [number(Fraction(k, 100)) for k in range(101)]
    table = root.get_timestamp_table(positions, format="dataframe")
    for index, position in enumerate(positions):
        stamp = _present(root.get_timestamp_at(position))
        row = {
            column.rsplit(" (", 1)[0]: table[column].iloc[index]
            for column in table.columns
            if not pd.isna(table[column].iloc[index])
        }
        assert row == stamp, position
        assert _present(bundle.get_matchstamp_at(position, "P")) == stamp, position
    # The shared boundaries: A2 ends where A does, B starts there.
    assert _present(root.get_timestamp_at(number(Fraction(2, 5)))) == {
        "P": number(Fraction(2, 5)),
        "B": number(0),
        "B1": number(0),
    }
    # A1 and A11 end where A2 starts: A2 alone holds A's 1/10.
    assert _present(root.get_timestamp_at(number(Fraction(1, 5)))) == {
        "P": number(Fraction(1, 5)),
        "A": number(Fraction(1, 10)),
        "A2": number(0),
    }


class TestZeroLengthChild:
    """A zero-length child is held at its offset and takes no end away."""

    @staticmethod
    def _parent(*, with_follower: bool) -> ContinuousLogicalTimeline:
        parent = ContinuousLogicalTimeline(length=8, uid="P")
        parent.add_child(ContinuousLogicalTimeline(length=4, uid="A"), 0)
        parent.add_child(ContinuousLogicalTimeline(length=0, uid="Z"), 4)
        if with_follower:
            parent.add_child(ContinuousLogicalTimeline(length=4, uid="C"), 4)
        return parent

    def test_held_beside_a_sibling_starting_there(self) -> None:
        parent = self._parent(with_follower=True)
        expected = {"P": 4, "C": 0, "Z": 0}
        assert _present(parent.get_timestamp_at(4)) == expected
        assert _row(parent, 4) == expected
        assert [child.id for child in parent.get_children_at(4)] == ["Z", "C"]

    def test_takes_no_end_away(self) -> None:
        parent = self._parent(with_follower=False)
        expected = {"P": 4, "A": 4, "Z": 0}
        assert _present(parent.get_timestamp_at(4)) == expected
        assert _row(parent, 4) == expected


# endregion
