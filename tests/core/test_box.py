"""A Box: an area of an image as an interval event of its timeline.

Strategy: ``tests/core/README.md``, section ``test_box.py``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from pydantic import ValidationError

from timetoalign.core import (
    BoundingBox,
    Box,
    BoxField,
    Coordinate,
    Duration,
    Event,
    NumberType,
    Projection,
    Quadrilateral,
    TimeUnit,
)

MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


def _message(refused: pytest.ExceptionInfo[ValidationError]) -> str:
    """The message a validator raised, as the library composed it."""
    return str(refused.value.errors()[0]["ctx"]["error"])


@pytest.fixture
def thoresen() -> BoundingBox:
    return BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")


@pytest.fixture
def measure_1() -> Quadrilateral:
    return Quadrilateral.from_points(
        [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
        order=MEASURE_ORDER,
        unit="pixels",
    )


@pytest.fixture
def measure_2() -> Quadrilateral:
    return Quadrilateral.from_points(
        [(2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266)], unit="pixels"
    )


class TestFromShape:
    def test_thoresen_rectangle_on_its_crop(self, thoresen: BoundingBox) -> None:
        box = Box.from_shape(thoresen, origin=7, id="rect_h2")
        assert box.start == Coordinate(378, TimeUnit.pixels)
        assert box.duration == Duration(139, TimeUnit.pixels)
        assert type(box.start.value) is int
        assert box.shape is thoresen
        assert box.quadrilateral is None
        assert box.end == Coordinate(517, TimeUnit.pixels)

    def test_measures_on_their_system(
        self, measure_1: Quadrilateral, measure_2: Quadrilateral
    ) -> None:
        first = Box.from_shape(measure_1, origin=1221, id="m1")
        second = Box.from_shape(measure_2, origin=Coordinate(1221, TimeUnit.pixels))
        assert (first.start.value, first.duration.value) == (0, 940)
        assert (second.start.value, second.duration.value) == (930, 624)
        assert first.shape == measure_1
        assert first.bounding_box is None
        assert second.id is None

    def test_event_row(self, measure_1: Quadrilateral) -> None:
        box = Box.from_shape(measure_1, origin=1221, id="m1")
        assert box.to_event_row() == {
            "event_type": "Box",
            "id": "m1",
            "start": Coordinate(0, TimeUnit.pixels),
            "duration": Duration(940, TimeUnit.pixels),
            "bounding_box": None,
            "quadrilateral": measure_1,
        }
        assert "id" not in Box.from_shape(measure_1).to_event_row()

    def test_fraction_box(self) -> None:
        third = BoundingBox.from_xywh(
            Fraction(1, 3),
            0,
            Fraction(2, 3),
            1,
            unit="user_units",
            number_type="fraction",
        )
        box = Box.from_shape(third)
        assert box.start.value == Fraction(1, 3)
        assert box.duration.value == Fraction(2, 3)
        assert type(box.start.value) is Fraction
        assert type(box.duration.value) is Fraction
        assert box.number_type is NumberType.fraction

    def test_a_whole_float_origin_is_its_integer(
        self, measure_1: Quadrilateral
    ) -> None:
        box = Box.from_shape(measure_1, origin=7.0)
        assert box.start == Coordinate(1214, TimeUnit.pixels)
        assert type(box.start.value) is int

    @pytest.mark.parametrize(
        ("origin", "shown"), [(7.5, "7.5"), (Fraction(15, 2), "15/2")]
    )
    def test_the_origin_is_never_rounded(
        self, origin: float | Fraction, shown: str, measure_1: Quadrilateral
    ) -> None:
        with pytest.raises(ValueError) as refused:
            Box.from_shape(measure_1, origin=origin)
        assert type(refused.value) is ValueError
        assert str(refused.value) == (
            f"A Box's origin is {shown}, which pixels as int cannot hold exactly: "
            "geometry is never rounded."
        )

    @pytest.mark.parametrize(
        ("origin", "shown"), [(float("nan"), "nan"), (float("-inf"), "-inf")]
    )
    def test_a_non_finite_origin_is_refused(
        self, origin: float, shown: str, measure_1: Quadrilateral
    ) -> None:
        with pytest.raises(ValueError) as refused:
            Box.from_shape(measure_1, origin=origin)
        assert type(refused.value) is ValueError
        assert str(refused.value) == (
            f"A Box's origin is {shown}, which is no finite number: geometry holds "
            "finite numbers only."
        )

    def test_a_ratio_origin_on_a_float_axis_is_refused(self) -> None:
        box = BoundingBox.from_xywh(1.5, 0, 2.5, 1, unit="css_pixels")
        with pytest.raises(ValueError) as refused:
            Box.from_shape(box, origin=Fraction(1, 2))
        assert type(refused.value) is ValueError
        assert str(refused.value) == (
            "A Box's origin is 1/2, a ratio, which css_pixels as float does not "
            "take: the declared number type governs and a ratio is read as a float "
            "only when it is a whole number; give a float."
        )


class TestRefusals:
    def test_a_duration_other_than_the_extent(self, measure_1: Quadrilateral) -> None:
        with pytest.raises(ValidationError) as refused:
            Box(
                id="m1",
                start=Coordinate(0, TimeUnit.pixels),
                duration=Duration(10, TimeUnit.pixels),
                quadrilateral=measure_1,
            )
        assert _message(refused) == (
            "Box 'm1': its duration is 10 and its shape spans 940 in x (1221 to "
            "2161); a Box's duration is its shape's x extent."
        )

    @pytest.mark.parametrize("both", [True, False])
    def test_not_exactly_one_shape(
        self, both: bool, thoresen: BoundingBox, measure_1: Quadrilateral
    ) -> None:
        shapes = {"bounding_box": thoresen, "quadrilateral": measure_1} if both else {}
        with pytest.raises(ValidationError) as refused:
            Box(
                id="m1",
                start=Coordinate(0, TimeUnit.pixels),
                duration=Duration(940, TimeUnit.pixels),
                **shapes,
            )
        assert _message(refused) == (
            "A Box holds exactly one shape: give bounding_box or quadrilateral."
        )

    def test_a_shape_in_another_unit(self, measure_1: Quadrilateral) -> None:
        skewed = Quadrilateral.from_points(
            [(corner.x.value, corner.y.value) for corner in measure_1.corners],
            unit="css_pixels",
            number_type="float",
        )
        with pytest.raises(ValidationError) as refused:
            Box(
                id="m1",
                start=Coordinate(0, TimeUnit.pixels),
                duration=Duration(940, TimeUnit.pixels),
                quadrilateral=skewed,
            )
        assert _message(refused) == (
            "Box 'm1': its shape is in css_pixels, its start in pixels; both are on "
            "one axis."
        )

    def test_a_shape_in_another_number_type(self) -> None:
        box = BoundingBox.from_xywh(
            0.5, 0, 1.5, 1, unit="user_units", number_type="float"
        )
        start = Coordinate(Fraction(1, 2), TimeUnit.user_units, number_type="fraction")
        with pytest.raises(ValidationError) as refused:
            Box(
                id="m1",
                start=start,
                duration=Duration(
                    Fraction(3, 2), TimeUnit.user_units, number_type="fraction"
                ),
                bounding_box=box,
            )
        assert _message(refused) == (
            "Box 'm1': its shape is written as float, its start as fraction; both "
            "are on one axis."
        )


class TestField:
    def test_round_trip(self, measure_1: Quadrilateral) -> None:
        box = Box.from_shape(measure_1, origin=1221, id="m1")
        field = BoxField.from_boxes([box, None])
        assert field.name == "box"
        assert field.unit is TimeUnit.pixels
        assert field.number_type is NumberType.int
        assert field.pa_type == BoxField.pa_schema
        assert field[0] == box
        assert field[1] is None
        row = field.to_pyarrow().to_pylist()[0]
        assert row["start"] == {"value": 0.0, "numerator": 0, "denominator": 1}
        assert row["bounding_box"] is None
        assert Box.from_row(row, unit="pixels", number_type="int") == box


class TestEventBase:
    def test_projection_and_box_are_events(self) -> None:
        assert issubclass(Projection, Event)
        assert issubclass(Box, Event)
        assert Projection(start=Coordinate(3, TimeUnit.samples)).semantic_type == (
            "Projection"
        )
