"""Points, bounding boxes and quadrilaterals in an image's pixel space.

Strategy: ``tests/core/README.md``, section ``test_shapes.py``.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest
from pydantic import ValidationError

from timetoalign.core import (
    BoundingBox,
    BoundingBoxField,
    Coordinate,
    Duration,
    NumberType,
    Point,
    Quadrilateral,
    QuadrilateralField,
    Shape,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.core.time import RATIONAL_STRUCT_TYPE

#: Thoresen, ``thoresen_test.tsv``, row ``rect_h2``.
THORESEN = (385, 46, 139, 20)

#: CollabScore, *Aimons-nous*, page 1: DMOS's region of measure 1, written
#: top-left, top-right, bottom-left, bottom-right.
MEASURE_1 = [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]]
MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")

#: System 1 of that page, written top-left, bottom-left, top-right,
#: bottom-right.
SYSTEM_1 = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")

#: Neuma's selector of measure 1: DMOS's order copied into SVG points.
NEUMA_M1 = "1221,1589 2151,1589 1221,2266 2161,2266"


def _message(refused: pytest.ExceptionInfo[ValidationError]) -> str:
    """The message a validator raised, as the library composed it."""
    return str(refused.value.errors()[0]["ctx"]["error"])


def _px(value: int) -> Coordinate:
    return Coordinate(value, TimeUnit.pixels)


def _corners(shape: Shape) -> list[tuple[int, int]]:
    return [(corner.x.value, corner.y.value) for corner in shape.corners]


def _xywh(box: BoundingBox) -> tuple[object, object, object, object]:
    return (box.x.value, box.y.value, box.width.value, box.height.value)


@pytest.fixture
def measure_1() -> Quadrilateral:
    return Quadrilateral.from_points(MEASURE_1, order=MEASURE_ORDER, unit="pixels")


@pytest.fixture
def measure_2() -> Quadrilateral:
    return Quadrilateral.from_points(
        [(2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266)], unit="pixels"
    )


class TestThoresenRectangle:
    @pytest.fixture
    def box(self) -> BoundingBox:
        return BoundingBox.from_xywh(*THORESEN, unit="pixels")

    def test_extremes(self, box: BoundingBox) -> None:
        assert box.x_max == _px(524)
        assert type(box.x_max.value) is int
        assert box.y_max == _px(66)
        assert box.x_min == _px(385)
        assert box.y_min == _px(46)
        assert box.unit is TimeUnit.pixels
        assert box.number_type is NumberType.int
        assert box.bounding_box is box

    def test_formats(self, box: BoundingBox) -> None:
        assert box.as_media_fragment() == "xywh=385,46,139,20"
        assert box.as_svg_points() == "385,46 524,46 524,66 385,66"
        assert box.as_mei_zone() == {
            "ulx": "385",
            "uly": "46",
            "lrx": "524",
            "lry": "66",
        }

    def test_media_fragment_read_back(self, box: BoundingBox) -> None:
        assert (
            BoundingBox.from_media_fragment("xywh=385,46,139,20", unit="pixels") == box
        )
        assert (
            BoundingBox.from_media_fragment("xywh=pixel:385,46,139,20", unit="pixels")
            == box
        )

    def test_a_percent_fragment_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            BoundingBox.from_media_fragment("xywh=percent:10,10,50,50", unit="pixels")
        assert str(refused.value) == (
            "'xywh=percent:10,10,50,50' is a percent media fragment, which needs the "
            "image's size: give the box in pixels (xywh=x,y,w,h)."
        )

    def test_dict_form(self, box: BoundingBox) -> None:
        assert box.to_dict() == {"x": 385, "y": 46, "width": 139, "height": 20}
        read = Shape.from_dict(box.to_dict(), unit="pixels", number_type="int")
        assert type(read) is BoundingBox
        assert read == box


class TestDmosQuadrilaterals:
    def test_measure_corners(self, measure_1: Quadrilateral) -> None:
        assert _corners(measure_1) == [
            (1221, 1589),
            (2151, 1589),
            (2161, 2266),
            (1221, 2266),
        ]
        assert measure_1.top_left == Point(x=_px(1221), y=_px(1589))
        assert measure_1.bottom_right == Point(x=_px(2161), y=_px(2266))

    def test_measure_bounding_box(self, measure_1: Quadrilateral) -> None:
        box = measure_1.bounding_box
        assert type(box) is BoundingBox
        assert _xywh(box) == (1221, 1589, 940, 677)
        assert box == BoundingBox.from_xywh(1221, 1589, 940, 677, unit="pixels")
        assert measure_1.x_max == _px(2161)

    def test_measure_svg_points(self, measure_1: Quadrilateral) -> None:
        points = measure_1.as_svg_points()
        assert points == "1221,1589 2151,1589 2161,2266 1221,2266"
        assert Quadrilateral.from_svg_points(points, unit="pixels") == measure_1

    def test_measure_dict_form(self, measure_1: Quadrilateral) -> None:
        assert measure_1.to_dict() == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 2151, "y": 1589},
            "bottom_right": {"x": 2161, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        read = Shape.from_dict(measure_1.to_dict(), unit="pixels", number_type="int")
        assert type(read) is Quadrilateral
        assert read == measure_1

    def test_system_corners(self) -> None:
        system = Quadrilateral.from_points(SYSTEM_1, order=SYSTEM_ORDER, unit="pixels")
        assert _corners(system) == [
            (1221, 1589),
            (4097, 1589),
            (4101, 2266),
            (1221, 2266),
        ]

    def test_the_crossed_outline_is_refused(self) -> None:
        with pytest.raises(ValidationError) as refused:
            Quadrilateral.from_svg_points(NEUMA_M1, unit="pixels")
        assert _message(refused) == (
            "The corners top_left (1221, 1589), top_right (2151, 1589), bottom_right "
            "(1221, 2266), bottom_left (2161, 2266) do not run clockwise around a "
            "convex outline: the turn at bottom_right is counter-clockwise."
        )


class TestFields:
    def test_quadrilateral_extremes(
        self, measure_1: Quadrilateral, measure_2: Quadrilateral
    ) -> None:
        field = QuadrilateralField.from_shapes([measure_1, measure_2])
        assert [cell["numerator"] for cell in field.x_min().to_pylist()] == [1221, 2151]
        assert [cell["numerator"] for cell in field.x_max().to_pylist()] == [2161, 2775]

    def test_quadrilateral_storage(
        self, measure_1: Quadrilateral, measure_2: Quadrilateral
    ) -> None:
        field = QuadrilateralField.from_shapes([measure_1, measure_2])
        assert field.name == "quadrilateral"
        assert field.pa_type == QuadrilateralField.pa_schema
        rows = field.to_pyarrow().to_pylist()
        assert rows[0]["top_left"]["x"] == {
            "value": 1221.0,
            "numerator": 1221,
            "denominator": 1,
        }
        for row in rows:
            for corner in row.values():
                for leaf in corner.values():
                    assert set(leaf) == {"value", "numerator", "denominator"}
                    assert None not in leaf.values()
                    assert leaf["value"] == leaf["numerator"]
                    assert leaf["denominator"] == 1
        metadata = field_metadata(field.to_field())
        assert metadata["unit"] == "pixels"
        assert metadata["number_type"] == "int"
        assert field.unit is TimeUnit.pixels
        assert field.number_type is NumberType.int
        assert field[0] == measure_1
        assert field[1] == measure_2

    def test_quadrilateral_storage_type(self) -> None:
        corner = pa.struct([("x", RATIONAL_STRUCT_TYPE), ("y", RATIONAL_STRUCT_TYPE)])
        assert QuadrilateralField.pa_schema == pa.struct(
            [
                ("top_left", corner),
                ("top_right", corner),
                ("bottom_right", corner),
                ("bottom_left", corner),
            ]
        )
        assert BoundingBoxField.pa_schema == pa.struct(
            [
                ("x", RATIONAL_STRUCT_TYPE),
                ("y", RATIONAL_STRUCT_TYPE),
                ("width", RATIONAL_STRUCT_TYPE),
                ("height", RATIONAL_STRUCT_TYPE),
            ]
        )

    def test_bounding_box_extremes_with_a_null_row(self) -> None:
        field = BoundingBoxField.from_shapes(
            [BoundingBox.from_xywh(*THORESEN, unit="pixels"), None]
        )
        assert field.name == "bounding_box"
        assert field.x_max().to_pylist() == [
            {"value": 524.0, "numerator": 524, "denominator": 1},
            None,
        ]
        assert field.x_min().to_pylist()[1] is None
        assert field[1] is None


class TestMeiZone:
    ZONE = {"ulx": "520", "uly": "390", "lrx": "874", "lry": "1019"}

    def test_corners_are_points(self) -> None:
        box = BoundingBox.from_corners(520, 390, 874, 1019, unit="pixels")
        assert _xywh(box) == (520, 390, 354, 629)

    def test_zone_read_and_written(self) -> None:
        zone = Shape.from_mei_zone(self.ZONE, unit="pixels")
        assert zone == BoundingBox.from_corners(520, 390, 874, 1019, unit="pixels")
        assert zone.as_mei_zone() == self.ZONE

    def test_zone_with_points(self, measure_1: Quadrilateral) -> None:
        attributes = {
            "ulx": "1221",
            "uly": "1589",
            "lrx": "2161",
            "lry": "2266",
            "points": "1221,1589 2151,1589 2161,2266 1221,2266",
        }
        zone = Shape.from_mei_zone(attributes, unit="pixels")
        assert zone == measure_1
        assert zone.as_mei_zone() == attributes


class TestExactness:
    @pytest.fixture
    def third(self) -> BoundingBox:
        return BoundingBox.from_xywh(
            Fraction(1, 3),
            0,
            Fraction(2, 3),
            1,
            unit="user_units",
            number_type="fraction",
        )

    def test_extent_is_exact(self, third: BoundingBox) -> None:
        assert third.x_max.value == Fraction(1)
        assert type(third.x_max.value) is Fraction
        assert third.number_type is NumberType.fraction

    def test_dict_form(self, third: BoundingBox) -> None:
        assert third.to_dict() == {
            "x": [1, 3],
            "y": [0, 1],
            "width": [2, 3],
            "height": [1, 1],
        }
        assert (
            Shape.from_dict(third.to_dict(), unit="user_units", number_type="fraction")
            == third
        )

    def test_media_fragment_refuses_a_ratio(self, third: BoundingBox) -> None:
        with pytest.raises(ValueError) as refused:
            third.as_media_fragment()
        assert str(refused.value) == (
            "A media fragment writes whole numbers; this shape's x is 1/3."
        )

    def test_mei_zone_refuses_a_ratio(self, third: BoundingBox) -> None:
        with pytest.raises(ValueError) as refused:
            third.as_mei_zone()
        assert str(refused.value) == (
            "MEI zone coordinates are whole numbers; this shape's ulx is 1/3."
        )

    def test_svg_refuses_an_endless_decimal(self, third: BoundingBox) -> None:
        with pytest.raises(ValueError) as refused:
            third.as_svg_points()
        assert str(refused.value) == (
            "SVG points are decimal numbers; 1/3 has no finite decimal expansion."
        )

    def test_svg_writes_a_finite_decimal(self) -> None:
        quarter = BoundingBox.from_xywh(
            Fraction(1, 4), 0, 1, 1, unit="user_units", number_type="fraction"
        )
        assert quarter.as_svg_points() == "0.25,0 1.25,0 1.25,1 0.25,1"

    def test_a_ratio_on_pixels_is_refused_by_the_construction_protocol(self) -> None:
        with pytest.raises(ValueError) as refused:
            BoundingBox.from_xywh(Fraction(3, 2), 0, 1, 1, unit="pixels")
        assert str(refused.value) == (
            "exact value 3/2 cannot be stored in an integer-valued field: rounding "
            "modes apply to inexact measurements, not to a ratio that was written "
            "down deliberately"
        )


class TestRefusals:
    def test_members_on_two_axes(self) -> None:
        with pytest.raises(ValidationError) as refused:
            BoundingBox(
                x=_px(1),
                y=_px(2),
                width=Duration(3.0, TimeUnit.css_pixels),
                height=Duration(4, TimeUnit.pixels),
            )
        assert _message(refused) == (
            "A BoundingBox lies on one axis: its width is in css_pixels as float, "
            "its x in pixels as int."
        )

    def test_a_negative_width(self) -> None:
        with pytest.raises(ValidationError) as refused:
            BoundingBox.from_corners(10, 0, 9, 5, unit="pixels")
        assert _message(refused) == (
            "A BoundingBox's width is -1; its width and height are never negative."
        )

    def test_an_order_naming_a_corner_twice(self) -> None:
        with pytest.raises(ValueError) as refused:
            Quadrilateral.from_points(
                MEASURE_1,
                order=("top_left", "top_left", "bottom_left", "bottom_right"),
                unit="pixels",
            )
        assert str(refused.value) == (
            "The order ['top_left', 'top_left', 'bottom_left', 'bottom_right'] names "
            "each corner once: top_left, top_right, bottom_right and bottom_left."
        )

    def test_raw_numbers_need_a_unit(self) -> None:
        with pytest.raises(TypeError) as refused:
            BoundingBox.from_xywh(*THORESEN)
        assert str(refused.value) == (
            "A shape's x is the bare number 385, which names no unit: pass unit=."
        )
