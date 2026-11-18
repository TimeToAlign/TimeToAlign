"""Tests for children calibrated by position/value pairs."""

from __future__ import annotations

import json
import re
from fractions import Fraction

import pytest

from timetoalign.core import Coordinate, IdCoordinate, NumberType
from timetoalign.maps import LinearMap
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    Timeline,
)

# A seconds axis drawn in points with five labelled ticks, positions as a
# program computed and printed them.
TICKS = [
    76.70371459980326,
    290.5859487023929,
    504.4681828049824,
    718.350416907572,
    932.2326510101616,
]
LABELS = [0, 10, 20, 30, 40]
AXIS_START = 76.70371459980326
AXIS_LENGTH = 855.5289364103584


def seconds(value: float) -> Coordinate:
    return Coordinate(value, "seconds")


def drawing() -> ContinuousGraphicalTimeline:
    return ContinuousGraphicalTimeline(length=932.2326510101616, unit="points")


def calibrate(parent: ContinuousGraphicalTimeline, ticks: list[float]) -> Timeline:
    return parent.create_calibrated_child(
        length=AXIS_LENGTH,
        offset=AXIS_START,
        calibration=[(tick, seconds(label)) for tick, label in zip(ticks, LABELS)],
    )


def small_parent() -> ContinuousGraphicalTimeline:
    return ContinuousGraphicalTimeline(length=100.0, unit="points")


def calibrate_small(calibration: list) -> Timeline:
    return small_parent().create_calibrated_child(80.0, 10.0, calibration)


class TestDrawnAxis:
    def test_child_spans_the_axis(self):
        parent = drawing()
        child = calibrate(parent, TICKS)
        assert parent.get_child_offset(child.id).value == 76.70371459980326
        assert child.length.value == 855.5289364103584
        assert type(child) is ContinuousGraphicalTimeline
        assert child.unit.value == "points"

    def test_map_is_the_line_through_the_end_ticks(self):
        cmap = calibrate(drawing(), TICKS).get_conversion_map("seconds")
        assert type(cmap) is LinearMap
        assert cmap.source_unit.value == "points"
        assert cmap.target_unit.value == "seconds"
        # 40.0 / (932.2326510101616 - 76.70371459980326)
        assert cmap.scalar == 0.04675470144567245
        assert type(cmap.scalar) is float
        assert cmap.offset == 0.0

    def test_map_reads_the_ticks(self):
        cmap = calibrate(drawing(), TICKS).get_conversion_map("seconds")
        # One ulp above the label: the float the line gives, not the label.
        assert cmap(213.88223410258962) == 10.000000000000002
        assert cmap(855.5289364103584) == 40.0

    def test_unsorted_calibration_gives_the_same_map(self):
        shuffled = [TICKS[i] for i in (3, 0, 4, 1, 2)]
        calibration = [(tick, seconds(LABELS[TICKS.index(tick)])) for tick in shuffled]
        unsorted = drawing().create_calibrated_child(
            AXIS_LENGTH, AXIS_START, calibration
        )
        assert unsorted.get_conversion_map("seconds") == calibrate(
            drawing(), TICKS
        ).get_conversion_map("seconds")

    def test_scalar_survives_serialization_bit_for_bit(self):
        child = calibrate(drawing(), TICKS)
        text = json.dumps(child.to_dict())
        restored = Timeline.from_dict(json.loads(text))
        cmap = restored.get_conversion_map("seconds")
        assert type(cmap.scalar) is float
        assert cmap.scalar.hex() == (0.04675470144567245).hex()
        assert cmap.offset == 0.0
        # The declaration is not written; seconds write floats by default.
        assert cmap.output_number_type is NumberType.float

    def test_a_label_off_the_line_changes_nothing(self):
        parent = drawing()
        ids_before = parent._manifest_ids()
        events_before = len(parent.events)
        moved = [TICKS[0], 300.0, *TICKS[2:]]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (300.0, 10.0) lies off the line through the "
                "first and last pair, which reaches 10.0 at 290.58594870239284."
            ),
        ):
            calibrate(parent, moved)
        assert parent.n_children == 0
        assert len(parent.events) == events_before
        assert parent._manifest_ids() == ids_before
        assert parent.create_child(length=1.0).id == "cgt_2"


class TestBound:
    def test_inner_ticks_are_checked_on_the_parent_positions(self):
        # Exact labels, positions as printed: the inner ticks lie 0.9375,
        # 0.375 and 0.15625 of their ulps off the exact line through the end
        # ticks, on the parent's positions.
        child = drawing().create_calibrated_child(
            AXIS_LENGTH, AXIS_START, list(zip(TICKS, LABELS)), value_unit="seconds"
        )
        cmap = child.get_conversion_map("seconds")
        assert cmap.scalar == 0.04675470144567245
        assert cmap.offset == 0.0

    def test_a_tick_two_floats_off_is_refused(self):
        # Two floats above the fourth tick: 2.16 of its ulps off the line.
        parent = drawing()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 3 (718.3504169075723, 30) lies off the line "
                "through the first and last pair, which reaches 30 at "
                "718.350416907572."
            ),
        ):
            parent.create_calibrated_child(
                AXIS_LENGTH,
                AXIS_START,
                list(zip([*TICKS[:3], 718.3504169075723, TICKS[4]], LABELS)),
                value_unit="seconds",
            )
        assert parent.n_children == 0


class TestExactAxis:
    def test_fraction_pairs_give_a_fraction_map(self):
        parent = ContinuousGraphicalTimeline(
            length=Fraction(10), unit="points", number_type="fraction"
        )
        child = parent.create_calibrated_child(
            6,
            Fraction(1, 2),
            [
                (Fraction(1, 2), Coordinate(Fraction(1), "quarters")),
                (Fraction(7, 2), Coordinate(Fraction(2), "quarters")),
                (Fraction(13, 2), Coordinate(Fraction(3), "quarters")),
            ],
        )
        cmap = child.get_conversion_map("quarters")
        assert cmap.scalar == Fraction(1, 3)
        assert type(cmap.scalar) is Fraction
        assert cmap.offset == Fraction(1)
        assert type(cmap.offset) is Fraction
        result = cmap(Fraction(3, 2))
        assert result == Fraction(3, 2)
        assert type(result) is Fraction


class TestInputs:
    def test_coordinate_positions_and_identity(self):
        parent = small_parent()
        child = parent.create_calibrated_child(
            Coordinate(80.0, "points"),
            Coordinate(10.0, "points"),
            [
                (Coordinate(10.0, "points"), seconds(0)),
                (IdCoordinate(90.0, "points", timeline_id=parent.id), seconds(4)),
            ],
            uid="axis",
            name="seconds axis",
        )
        assert child.id == "axis"
        assert child.name == "seconds axis"
        assert child.number_type == parent.number_type
        assert child.get_conversion_map("seconds").scalar == 0.05

    def test_numbers_in_value_unit(self):
        by_number = small_parent().create_calibrated_child(
            80.0, 10.0, [(10.0, 0.0), (50.0, 2.0), (90.0, 4.0)], value_unit="seconds"
        )
        by_coordinate = calibrate_small(
            [(10.0, seconds(0)), (50.0, seconds(2)), (90.0, seconds(4))]
        )
        cmap = by_number.get_conversion_map("seconds")
        assert cmap.target_unit.value == "seconds"
        assert cmap.scalar == 0.05
        assert cmap == by_coordinate.get_conversion_map("seconds")

    def test_value_unit_disagreeing_with_a_coordinate(self):
        parent = small_parent()
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 has a value in seconds, not in quarters."
            ),
        ):
            parent.create_calibrated_child(
                80.0, 10.0, [(10.0, 0), (90.0, seconds(4))], value_unit="quarters"
            )
        assert parent.n_children == 0

    def test_explicit_law(self):
        child = small_parent().create_calibrated_child(
            80.0, 10.0, [(10.0, seconds(0)), (90.0, seconds(4))], law=LinearMap
        )
        assert type(child.get_conversion_map("seconds")) is LinearMap


# Each leaf class with its default unit, the number its axis takes, the unit
# the values read in, and the scalar 4/80 in the arithmetic of the pairs.
LEAVES = [
    (ContinuousPhysicalTimeline, float, "quarters", 0.05),
    (DiscretePhysicalTimeline, int, "quarters", Fraction(1, 20)),
    (ContinuousLogicalTimeline, Fraction, "seconds", 0.05),
    (DiscreteLogicalTimeline, int, "seconds", 0.05),
    (ContinuousGraphicalTimeline, float, "quarters", 0.05),
    (DiscreteGraphicalTimeline, int, "quarters", Fraction(1, 20)),
]


@pytest.mark.parametrize(
    "cls, number, value_unit, scalar",
    LEAVES,
    ids=[leaf[0].__name__ for leaf in LEAVES],
)
def test_every_leaf_class(cls, number, value_unit, scalar):
    parent = cls(length=number(100))
    child = parent.create_calibrated_child(
        number(80),
        number(10),
        [
            (number(position), Coordinate(value, value_unit))
            for position, value in [(10, 0), (50, 2), (90, 4)]
        ],
    )
    assert type(child) is cls
    assert child.unit == parent.unit
    assert child.number_type == parent.number_type
    assert parent.get_child_offset(child.id).value == number(10)
    cmap = child.get_conversion_map(value_unit)
    assert cmap.source_unit == parent.unit
    assert cmap.target_unit.value == value_unit
    assert cmap.scalar == scalar
    assert type(cmap.scalar) is type(scalar)
    assert cmap(number(40)) == 2


class TestRefusals:
    def test_fewer_than_two_pairs(self):
        with pytest.raises(
            ValueError,
            match=re.escape("A calibration needs at least two pairs, got 1."),
        ):
            calibrate_small([(10.0, seconds(0))])

    def test_repeated_position(self):
        with pytest.raises(
            ValueError,
            match=re.escape("Calibration pairs 0 and 1 share the position 30.0."),
        ):
            calibrate_small(
                [(30.0, seconds(0)), (30.0, seconds(1)), (90.0, seconds(4))]
            )

    def test_values_in_two_units(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 has a value in quarters, not in seconds."
            ),
        ):
            calibrate_small([(10.0, seconds(0)), (90.0, Coordinate(4, "quarters"))])

    def test_inner_pair_off_the_line(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (50.0, 3.0) lies off the line through the first "
                "and last pair, which reaches 3.0 at 70.0."
            ),
        ):
            calibrate_small(
                [(10.0, seconds(0)), (50.0, seconds(3)), (90.0, seconds(4))]
            )

    def test_position_before_the_offset(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 0 lies at 5.0, outside the child from 10.0 to 90.0."
            ),
        ):
            calibrate_small([(5.0, seconds(0)), (90.0, seconds(4))])

    def test_position_after_the_end(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 lies at 90.5, outside the child from 10.0 to 90.0."
            ),
        ):
            calibrate_small([(10.0, seconds(0)), (90.5, seconds(4))])

    def test_number_without_value_unit(self):
        parent = small_parent()
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Calibration pair 0 has a value of type int, not a Coordinate "
                "carrying its unit; give value_unit to read numbers."
            ),
        ):
            parent.create_calibrated_child(80.0, 10.0, [(10.0, 0), (90.0, seconds(4))])
        assert parent.n_children == 0
