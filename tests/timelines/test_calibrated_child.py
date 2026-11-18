"""Tests for children calibrated by position/value pairs."""

from __future__ import annotations

import json
import re
from fractions import Fraction

import pytest

from timetoalign.core import Coordinate, IdCoordinate
from timetoalign.maps import LinearMap
from timetoalign.timelines import ContinuousGraphicalTimeline, Timeline

# A seconds axis drawn in points: the five labelled ticks of a ScoreWarp drawing.
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
SUFFIX = " Positions are read from the child's offset 10.0."


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
        scalar = restored.get_conversion_map("seconds").scalar
        assert type(scalar) is float
        assert scalar.hex() == (0.04675470144567245).hex()

    def test_a_label_off_the_line_changes_nothing(self):
        parent = drawing()
        ids_before = parent._manifest_ids()
        events_before = len(parent.events)
        moved = [TICKS[0], 300.0, *TICKS[2:]]
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (223.29628540019672, 10.0) lies off the line "
                "through the first and last pair, which reads 10.440151157813867 at "
                "223.29628540019672. Positions are read from the child's offset "
                "76.70371459980326."
            ),
        ):
            calibrate(parent, moved)
        assert parent.n_children == 0
        assert len(parent.events) == events_before
        assert parent._manifest_ids() == ids_before
        assert parent.create_child(length=1.0).id == "cgt_2"


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

    def test_explicit_law(self):
        child = small_parent().create_calibrated_child(
            80.0, 10.0, [(10.0, seconds(0)), (90.0, seconds(4))], law=LinearMap
        )
        assert type(child.get_conversion_map("seconds")) is LinearMap


class TestRefusals:
    def test_fewer_than_two_pairs(self):
        with pytest.raises(
            ValueError,
            match=re.escape("A calibration needs at least two pairs, got 1." + SUFFIX),
        ):
            calibrate_small([(10.0, seconds(0))])

    def test_repeated_position(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pairs 0 and 1 share the position 20.0." + SUFFIX
            ),
        ):
            calibrate_small(
                [(30.0, seconds(0)), (30.0, seconds(1)), (90.0, seconds(4))]
            )

    def test_values_in_two_units(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 has a value in quarters, not in seconds." + SUFFIX
            ),
        ):
            calibrate_small([(10.0, seconds(0)), (90.0, Coordinate(4, "quarters"))])

    def test_inner_pair_off_the_line(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (40.0, 3.0) lies off the line through the first "
                "and last pair, which reads 2.0 at 40.0." + SUFFIX
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

    def test_value_without_unit(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Calibration pair 0 has a value of type int, not a Coordinate "
                "carrying its unit."
            ),
        ):
            calibrate_small([(10.0, 0), (90.0, seconds(4))])
