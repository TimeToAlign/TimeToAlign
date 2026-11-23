"""Tests for time paths: the accolade rule, piecewise paths, paths from boxes.

The validation logic is documented in ``tests/loader/graphical/README.md``
("Time paths and pages"). All numbers are literals taken by hand from two
real sources (a scan's systems, a chorale page's measure zones); every
expected value is exact.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core import BoundingBox, Box, Quadrilateral
from timetoalign.loader.graphical import (
    DiagonalLinePath,
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
    VerticalLinePath,
    group_into_accolades,
)

#: The nine measure zones of the chorale page: ulx, uly, lrx, lry.
CHORALE_ZONES = [
    (490, 344, 973, 877),
    (944, 368, 1523, 864),
    (1481, 388, 2082, 912),
    (2034, 402, 2694, 884),
    (2669, 389, 3296, 916),
    (498, 858, 1496, 1337),
    (1465, 885, 2146, 1353),
    (2110, 886, 2748, 1357),
    (2704, 886, 3302, 1387),
]

#: The chorale page's two lines, computed by hand from the zones.
CHORALE_LINES = (
    HorizontalLinePath(x0=490, x1=3296, y=630, tolerance=286),
    HorizontalLinePath(
        x0=498, x1=3302, y=Fraction(2245, 2), tolerance=Fraction(529, 2)
    ),
)

#: System 1 and 2 of the scan's first page, corners in canonical order.
SYSTEM_1 = [(1221, 1589), (4097, 1589), (4101, 2266), (1221, 2266)]
SYSTEM_2 = [(971, 2493), (4104, 2493), (4112, 3192), (971, 3192)]


def _zone(corners: tuple[int, int, int, int]) -> BoundingBox:
    return BoundingBox.from_corners(*corners, unit="pixels")


def _same(result: object, expected: object) -> None:
    """Equal in value and in Python type, member by member for a tuple."""
    assert result == expected
    if isinstance(expected, tuple):
        assert [type(member) for member in result] == [
            type(member) for member in expected
        ]
    else:
        assert type(result) is type(expected)


class TestGroupIntoAccolades:
    def test_chorale_page(self) -> None:
        left_edges = [zone[0] for zone in CHORALE_ZONES]
        assert group_into_accolades(left_edges) == [0, 0, 0, 0, 0, 1, 1, 1, 1]

    @pytest.mark.parametrize(
        ("left_edges", "expected"),
        [
            ([0, 10, 5], [0, 0, 0]),
            ([0, 10, 4], [0, 0, 1]),
            ([Fraction(0), Fraction(10), Fraction(5)], [0, 0, 0]),
            ([Fraction(0), Fraction(10), Fraction(49, 10)], [0, 0, 1]),
            ([0.0, 10.0, 5.0], [0, 0, 0]),
            ([0.0, 10.0, 4.5], [0, 0, 1]),
            ([5, 9, 1, 8, 0, 9], [0, 0, 1, 1, 2, 2]),
            ([7], [0]),
            ([3, 3, 3], [0, 0, 0]),
            ([], []),
        ],
    )
    def test_the_inequality(self, left_edges: list, expected: list[int]) -> None:
        assert group_into_accolades(left_edges) == expected


class TestNumbersKept:
    def test_horizontal_line_of_integers(self) -> None:
        path = HorizontalLinePath(x0=1221, x1=4101, y=1927, tolerance=338)
        _same(path.length, 2880)
        _same(path.to_2d(930), (2151, 1927))
        _same(path.from_2d(2151, 1700), 930)
        _same(path.distance_to_path(2151, 1700), 227)

    def test_horizontal_line_of_fractions(self) -> None:
        path = HorizontalLinePath(
            x0=Fraction(1, 3),
            x1=Fraction(7, 3),
            y=Fraction(3855, 2),
            tolerance=Fraction(677, 2),
        )
        _same(path.length, Fraction(2))
        _same(path.to_2d(Fraction(1, 2)), (Fraction(5, 6), Fraction(3855, 2)))
        _same(path.from_2d(Fraction(4, 3), 1589), Fraction(1))

    def test_horizontal_line_of_floats(self) -> None:
        path = HorizontalLinePath(x0=10.0, x1=500.5, y=100.0)
        _same(path.length, 490.5)
        _same(path.to_2d(0.5), (10.5, 100.0))
        _same(path.from_2d(10.5, 100.0), 0.5)

    def test_vertical_line_of_integers(self) -> None:
        path = VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
        _same(path.length, 299400)
        _same(path.to_2d(1500), (2048, 1500))
        _same(path.from_2d(0, 1500), 1500)
        _same(path.distance_to_path(4000, 1500), 1952)

    def test_vertical_line_of_fractions(self) -> None:
        path = VerticalLinePath(x=Fraction(1, 2), y0=Fraction(1, 4), y1=Fraction(9, 4))
        _same(path.length, Fraction(2))
        _same(path.to_2d(Fraction(1, 4)), (Fraction(1, 2), Fraction(1, 2)))
        _same(path.from_2d(Fraction(1, 2), Fraction(5, 4)), Fraction(1))

    def test_the_strip_of_a_horizontal_line_is_y_plus_minus_tolerance(self) -> None:
        path = HorizontalLinePath(
            x0=1221, x1=4101, y=Fraction(3855, 2), tolerance=Fraction(677, 2)
        )
        assert path.from_2d(1221, 1589) == 0
        assert path.from_2d(4101, 2266) == 2880
        assert path.from_2d(1221, 1588) is None
        assert path.from_2d(1221, 2267) is None
        assert path.from_2d(1220, 1900) is None
        assert path.from_2d(4102, 1900) is None

    def test_the_strip_of_a_vertical_line_is_x_plus_minus_tolerance(self) -> None:
        path = VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
        assert path.from_2d(0, 7) == 7
        assert path.from_2d(4096, 7) == 7
        assert path.from_2d(-1, 7) is None
        assert path.from_2d(4097, 7) is None


class TestPiecewisePath:
    path = PiecewisePath(CHORALE_LINES)

    def test_offsets_and_length(self) -> None:
        _same(self.path.offsets, (0, 2806))
        _same(self.path.length, 5610)

    def test_pieces_are_a_tuple(self) -> None:
        assert PiecewisePath(list(CHORALE_LINES)).pieces == CHORALE_LINES

    def test_to_2d_reads_piece_after_piece(self) -> None:
        _same(self.path.to_2d(0), (490, 630))
        _same(self.path.to_2d(2805), (3295, 630))
        _same(self.path.to_2d(2806), (498, Fraction(2245, 2)))
        _same(self.path.to_2d(3260), (952, Fraction(2245, 2)))
        _same(self.path.to_2d(5610), (3302, Fraction(2245, 2)))

    def test_from_2d_counts_from_the_start_of_the_path(self) -> None:
        _same(self.path.from_2d(944, 400), 454)
        _same(self.path.from_2d(498, 1300), 2806)
        _same(self.path.from_2d(3302, 1387), 5610)

    def test_from_2d_takes_the_nearer_line_where_strips_overlap(self) -> None:
        # y 858 to 916 is in both strips. 870: 240 from 630, 252.5 from 1122.5.
        _same(self.path.from_2d(1000, 870), 510)
        # 880: 250 from 630, 242.5 from 1122.5.
        _same(self.path.from_2d(1000, 880), 2806 + 502)

    def test_from_2d_outside_every_strip(self) -> None:
        assert self.path.from_2d(1000, 300) is None
        assert self.path.from_2d(1000, 1400) is None
        assert self.path.from_2d(400, 630) is None

    def test_distance_is_the_least_over_the_pieces(self) -> None:
        _same(self.path.distance_to_path(1000, 870), 240)
        _same(self.path.distance_to_path(1000, 880), Fraction(485, 2))

    def test_mixed_pieces(self) -> None:
        path = PiecewisePath(
            (
                HorizontalLinePath(x0=0, x1=100, y=10, tolerance=5),
                VerticalLinePath(x=100, y0=10, y1=60, tolerance=5),
            )
        )
        _same(path.offsets, (0, 100))
        _same(path.length, 150)
        _same(path.to_2d(120), (100, 30))
        _same(path.from_2d(103, 40), 130)

    def test_piece_of_places_every_zone(self) -> None:
        pieces = [self.path.piece_of(_zone(zone)) for zone in CHORALE_ZONES]
        assert pieces == [0, 0, 0, 0, 0, 1, 1, 1, 1]

    def test_piece_of_takes_the_nearer_line_where_strips_overlap(self) -> None:
        assert self.path.piece_of(_zone((600, 860, 1000, 880))) == 0
        assert self.path.piece_of(_zone((600, 870, 1000, 890))) == 1

    def test_piece_of_reads_a_quadrilateral(self) -> None:
        path = TimeAxisPath.from_boxes(
            [
                Quadrilateral.from_points(SYSTEM_1, unit="pixels"),
                Quadrilateral.from_points(SYSTEM_2, unit="pixels"),
            ],
            accolades=range(2),
        )
        measure = Quadrilateral.from_points(
            [(2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266)], unit="pixels"
        )
        assert path.piece_of(measure) == 0

    def test_piece_of_ignores_pieces_that_are_no_horizontal_lines(self) -> None:
        path = PiecewisePath(
            (
                VerticalLinePath(x=50, y0=0, y1=100, tolerance=50),
                HorizontalLinePath(x0=0, x1=100, y=50, tolerance=50),
            )
        )
        assert path.piece_of(_zone((10, 40, 20, 60))) == 1

    def test_piece_of_refuses_a_shape_past_every_x_extent(self) -> None:
        with pytest.raises(ValueError) as error:
            self.path.piece_of(_zone((3000, 400, 3300, 800)))
        assert str(error.value) == (
            "The shape spanning x 3000 to 3300 with its centre at y 600 lies on "
            "no piece of the path."
        )

    def test_piece_of_refuses_a_centre_in_no_strip(self) -> None:
        with pytest.raises(ValueError) as error:
            self.path.piece_of(_zone((600, 100, 1000, 201)))
        assert str(error.value) == (
            "The shape spanning x 600 to 1000 with its centre at y 301/2 lies on "
            "no piece of the path."
        )

    def test_piece_of_refuses_a_centre_equally_near_two_lines(self) -> None:
        path = PiecewisePath(
            (
                HorizontalLinePath(x0=0, x1=500, y=100, tolerance=60),
                HorizontalLinePath(x0=0, x1=500, y=200, tolerance=60),
            )
        )
        with pytest.raises(ValueError) as error:
            path.piece_of(_zone((10, 140, 90, 160)))
        assert str(error.value) == (
            "The shape spanning x 10 to 90 with its centre at y 150 is equally "
            "near pieces 0 and 1 of the path."
        )

    def test_no_piece_is_refused(self) -> None:
        with pytest.raises(ValueError) as error:
            PiecewisePath(())
        assert str(error.value) == "A PiecewisePath needs at least one piece."

    def test_a_nested_piecewise_path_is_refused(self) -> None:
        with pytest.raises(ValueError) as error:
            PiecewisePath((CHORALE_LINES[0], self.path))
        assert str(error.value) == (
            "Piece 1 of a PiecewisePath is itself a PiecewisePath: the pieces "
            "are listed flat."
        )


class TestFromBoxes:
    def test_chorale_page_is_grouped_into_two_lines(self) -> None:
        path = TimeAxisPath.from_boxes([_zone(zone) for zone in CHORALE_ZONES])
        assert type(path) is PiecewisePath
        assert path.pieces == CHORALE_LINES
        _same(path.pieces[0].y, 630)
        _same(path.pieces[0].tolerance, 286)
        _same(path.pieces[1].y, Fraction(2245, 2))
        _same(path.pieces[1].tolerance, Fraction(529, 2))
        _same(path.offsets, (0, 2806))
        _same(path.length, 5610)

    def test_is_inherited_by_every_path_class(self) -> None:
        zones = [_zone(zone) for zone in CHORALE_ZONES]
        assert HorizontalLinePath.from_boxes(zones) == TimeAxisPath.from_boxes(zones)

    def test_stated_accolades_are_not_grouped(self) -> None:
        systems = [
            Quadrilateral.from_points(SYSTEM_1, unit="pixels"),
            Quadrilateral.from_points(SYSTEM_2, unit="pixels"),
        ]
        path = TimeAxisPath.from_boxes(systems, accolades=range(2))
        assert path.pieces == (
            HorizontalLinePath(
                x0=1221, x1=4101, y=Fraction(3855, 2), tolerance=Fraction(677, 2)
            ),
            HorizontalLinePath(
                x0=971, x1=4112, y=Fraction(5685, 2), tolerance=Fraction(699, 2)
            ),
        )
        _same(path.offsets, (0, 2880))
        _same(path.length, 6021)

    def test_one_box_is_a_path_of_one_piece(self) -> None:
        path = TimeAxisPath.from_boxes([_zone(CHORALE_ZONES[0])])
        assert path == PiecewisePath(
            (
                HorizontalLinePath(
                    x0=490, x1=973, y=Fraction(1221, 2), tolerance=Fraction(533, 2)
                ),
            )
        )

    def test_boxes_are_read_through_their_shapes(self) -> None:
        zones = [_zone(zone) for zone in CHORALE_ZONES]
        boxes = [Box.from_shape(zone) for zone in zones]
        assert TimeAxisPath.from_boxes(boxes) == TimeAxisPath.from_boxes(zones)

    def test_labels_order_the_accolades_by_their_first_box(self) -> None:
        zones = [_zone(zone) for zone in CHORALE_ZONES]
        labels = [7, 7, 7, 7, 7, 2, 2, 2, 2]
        assert TimeAxisPath.from_boxes(zones, accolades=labels).pieces == CHORALE_LINES

    def test_no_box_is_refused(self) -> None:
        with pytest.raises(ValueError) as error:
            TimeAxisPath.from_boxes([])
        assert str(error.value) == "A time path needs at least one box."

    def test_another_number_of_labels_is_refused(self) -> None:
        zones = [_zone(zone) for zone in CHORALE_ZONES]
        with pytest.raises(ValueError) as error:
            TimeAxisPath.from_boxes(zones, accolades=range(8))
        assert str(error.value) == "accolades assigns 8 boxes; 9 were given."

    def test_two_axes_are_refused(self) -> None:
        boxes = [
            _zone(CHORALE_ZONES[0]),
            _zone(CHORALE_ZONES[1]),
            BoundingBox.from_corners(1.0, 2.0, 3.0, 4.0, unit="seconds"),
        ]
        with pytest.raises(ValueError) as error:
            TimeAxisPath.from_boxes(boxes)
        assert str(error.value) == (
            "The boxes of a time path are in one unit and number type: box 0 is "
            "in pixels as int, box 2 in seconds as float."
        )

    def test_an_accolade_without_x_extent_is_refused(self) -> None:
        boxes = [
            _zone(CHORALE_ZONES[0]),
            BoundingBox.from_xywh(500, 900, 0, 40, unit="pixels"),
        ]
        with pytest.raises(ValueError) as error:
            TimeAxisPath.from_boxes(boxes, accolades=[0, 1])
        assert str(error.value) == (
            "accolade 2: x1 (500) must be greater than x0 (500)"
        )


class TestDrawPath:
    def test_a_piecewise_path_is_drawn_piece_by_piece(self, tmp_path) -> None:
        pytest.importorskip("pymupdf")
        image_module = pytest.importorskip("PIL.Image")
        from timetoalign.loader.graphical import ImageSource

        file = tmp_path / "page.png"
        image_module.new("RGB", (100, 100), (255, 255, 255)).save(file)
        path = PiecewisePath(
            (
                HorizontalLinePath(x0=10, x1=90, y=20, tolerance=5),
                HorizontalLinePath(x0=10, x1=90, y=80, tolerance=5),
            )
        )
        drawn = (
            ImageSource.from_image_file(file)
            .draw_path(path, color=(0, 255, 0), line_width=1)
            .to_pil()
            .convert("RGB")
        )
        green, white = (0, 255, 0), (255, 255, 255)
        assert drawn.getpixel((10, 20)) == green
        assert drawn.getpixel((89, 20)) == green
        assert drawn.getpixel((10, 80)) == green
        assert drawn.getpixel((89, 80)) == green
        # Nothing joins the end of the first line to the start of the second.
        assert drawn.getpixel((50, 50)) == white
        assert {drawn.getpixel((x, 50)) for x in range(100)} == {white}

    def test_a_diagonal_stays_one_line(self, tmp_path) -> None:
        pytest.importorskip("pymupdf")
        image_module = pytest.importorskip("PIL.Image")
        from timetoalign.loader.graphical import ImageSource

        file = tmp_path / "page.png"
        image_module.new("RGB", (100, 100), (255, 255, 255)).save(file)
        drawn = (
            ImageSource.from_image_file(file)
            .draw_path(DiagonalLinePath(start=(10, 10), end=(90, 90)), line_width=1)
            .to_pil()
            .convert("RGB")
        )
        assert drawn.getpixel((50, 50)) == (0, 255, 0)
