"""Tests for the builder of page timelines from time paths.

The validation logic is documented in ``tests/loader/graphical/README.md``
("Time paths and pages"). All numbers are literals taken by hand from two
real sources (a scan's systems and measures, a chorale page's measure
zones); every expected value is exact.
"""

from __future__ import annotations

from typing import Any

import pytest

from timetoalign.core import (
    BoundingBox,
    Coordinate,
    Duration,
    Quadrilateral,
    Shape,
    TimeUnit,
)
from timetoalign.loader.graphical import (
    DiagonalLinePath,
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
    VerticalLinePath,
)
from timetoalign.loader.graphical._pages import (
    Page,
    PageAccolade,
    PageBox,
    build_pages,
    build_volume,
)
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline
from timetoalign.timelines.factory import _mint_timeline_id

#: Per page of the scan: its image, and per system (x0, x1, y0, y1).
SCAN = {
    "1": (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2",
        [
            (1221, 4101, 1589, 2266),
            (971, 4112, 2493, 3192),
            (983, 4125, 3414, 4135),
            (994, 4142, 4367, 5091),
        ],
    ),
    "2": (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f3",
        [
            (519, 3661, 896, 1647),
            (522, 3659, 1832, 2524),
            (522, 3659, 2713, 3369),
            (523, 3658, 3563, 4260),
            (522, 3661, 4444, 5129),
        ],
    ),
    "3": (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f4",
        [
            (913, 4060, 920, 1680),
            (918, 4054, 1966, 2813),
            (913, 4051, 3090, 3861),
            (907, 4048, 4144, 4925),
        ],
    ),
    "4": (
        "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f5",
        [
            (508, 3650, 963, 1702),
            (504, 3643, 1989, 2767),
            (494, 3634, 3030, 3854),
            (489, 3628, 4144, 4939),
        ],
    ),
}

#: System 1 of page 1 and its first two measures, corners in canonical order.
SYSTEM_1 = [(1221, 1589), (4097, 1589), (4101, 2266), (1221, 2266)]
MEASURE_1 = [(1221, 1589), (2151, 1589), (2161, 2266), (1221, 2266)]
MEASURE_2 = [(2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266)]

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
CHORALE_IMAGE = "facsimile/20251230_180023.jpg"


class _Loader:
    """Stands for the loader the timeline ids are minted from."""


def _quad(corners: list[tuple[int, int]]) -> Quadrilateral:
    return Quadrilateral.from_points(corners, unit="pixels")


def _rect(x0: int, y0: int, x1: int, y1: int) -> BoundingBox:
    return BoundingBox.from_corners(x0, y0, x1, y1, unit="pixels")


def _scan_pages() -> list[Page]:
    """The scan's four pages; page 1's first system holds two measures."""
    pages = []
    for name, (image, systems) in SCAN.items():
        areas: list[Shape] = [_rect(x0, y0, x1, y1) for x0, x1, y0, y1 in systems]
        accolades = [
            PageAccolade(name=str(k), area=area) for k, area in enumerate(areas, 1)
        ]
        if name == "1":
            areas[0] = _quad(SYSTEM_1)
            accolades[0] = PageAccolade(
                name="1",
                area=areas[0],
                boxes=[PageBox(_quad(MEASURE_1)), PageBox(_quad(MEASURE_2))],
            )
        pages.append(
            Page(
                name=name,
                image=image,
                path=TimeAxisPath.from_boxes(areas, accolades=range(len(areas))),
                accolades=accolades,
            )
        )
    return pages


def _chorale_page(**changes: Any) -> Page:
    zones = [
        PageBox(_rect(ulx, uly, lrx, lry), source_id=f"zone_{k}")
        for k, (ulx, uly, lrx, lry) in enumerate(CHORALE_ZONES, start=1)
    ]
    arguments: dict[str, Any] = {
        "name": "15",
        "image": CHORALE_IMAGE,
        "path": TimeAxisPath.from_boxes([zone.shape for zone in zones]),
        "source_id": "surface_15",
        "boxes": zones,
    }
    arguments.update(changes)
    return Page(**arguments)


def _segments(line: Timeline) -> list[tuple[int, Timeline]]:
    """Each segment's offset and timeline, in order."""
    return [(offset.value, segment) for _, offset, segment in line.iter_segments()]


def _boxes(timeline: Timeline) -> list[tuple[str, int, int]]:
    """Each Box of *timeline*: id, start, duration."""
    return [
        (row["id"], row["start"]["numerator"], row["duration"]["numerator"])
        for row in timeline.get_events().table.to_pylist()
    ]


def _assert_invariant(accolade: Timeline) -> int:
    """Check every Box of *accolade* against its shape; return their number."""
    origin = Shape.from_dict(
        accolade.meta["box"], unit=accolade.unit, number_type=accolade.number_type
    ).x_min
    rows = accolade.get_events().table.to_pylist()
    for row in rows:
        cell = row["bounding_box"] or row["quadrilateral"]
        xs = (
            [
                cell["x"]["numerator"],
                cell["x"]["numerator"] + cell["width"]["numerator"],
            ]
            if row["bounding_box"]
            else [corner["x"]["numerator"] for corner in cell.values()]
        )
        assert row["start"]["numerator"] == min(xs) - origin.value
        assert row["duration"]["numerator"] == max(xs) - min(xs)
    return len(rows)


def _assert_nothing_minted(loader: _Loader) -> None:
    """The next timeline id drawn from *loader* is the first."""
    assert _mint_timeline_id(loader, TimeUnit.pixels) == "dgt_1"


class TestScanVolume:
    @pytest.fixture(scope="class")
    def volume(self) -> SegmentLine:
        return build_volume(_Loader(), _scan_pages(), uid="scan", box_ids="m{n}")

    def test_the_volume(self, volume: SegmentLine) -> None:
        assert type(volume) is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        assert volume.id == "scan"
        assert volume.unit is TimeUnit.pixels
        assert volume.length == Coordinate(53123, TimeUnit.pixels)
        assert volume.is_locked
        assert volume.meta == {}

    def test_the_pages(self, volume: SegmentLine) -> None:
        pages = _segments(volume)
        assert [offset for offset, _ in pages] == [0, 12311, 28001, 40563]
        assert [page.length.value for _, page in pages] == [12311, 15690, 12562, 12560]
        assert [page.name for _, page in pages] == ["1", "2", "3", "4"]
        assert [page.id for _, page in pages] == ["dgt_2", "dgt_7", "dgt_13", "dgt_18"]
        for (_, page), (image, _) in zip(pages, SCAN.values()):
            assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
            assert page.is_locked
            assert page.meta == {"image": image}

    def test_the_accolades_of_page_1(self, volume: SegmentLine) -> None:
        page = _segments(volume)[0][1]
        accolades = _segments(page)
        assert [offset for offset, _ in accolades] == [0, 2880, 6021, 9163]
        assert [a.length.value for _, a in accolades] == [2880, 3141, 3142, 3148]
        assert [a.name for _, a in accolades] == ["1", "2", "3", "4"]
        assert [a.id for _, a in accolades] == ["dgt_3", "dgt_4", "dgt_5", "dgt_6"]
        image = SCAN["1"][0]
        for _, accolade in accolades:
            assert type(accolade) is DiscreteGraphicalTimeline
            assert not accolade.is_locked
            assert accolade.meta["image"] == image
        assert accolades[0][1].meta["box"] == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 4097, "y": 1589},
            "bottom_right": {"x": 4101, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        assert accolades[1][1].meta["box"] == {
            "x": 971,
            "y": 2493,
            "width": 3141,
            "height": 699,
        }

    def test_every_page_is_the_sum_of_its_accolades(self, volume: SegmentLine) -> None:
        lengths = {
            page.name: [a.length.value for _, a in _segments(page)]
            for _, page in _segments(volume)
        }
        assert lengths == {
            "1": [2880, 3141, 3142, 3148],
            "2": [3142, 3137, 3137, 3135, 3139],
            "3": [3147, 3136, 3138, 3141],
            "4": [3142, 3139, 3140, 3139],
        }
        assert [a.id for _, a in _segments(_segments(volume)[3][1])] == [
            "dgt_19",
            "dgt_20",
            "dgt_21",
            "dgt_22",
        ]

    def test_the_measures_are_boxes_of_their_accolade(
        self, volume: SegmentLine
    ) -> None:
        page = _segments(volume)[0][1]
        first, second = _segments(page)[0][1], _segments(page)[1][1]
        assert _boxes(first) == [("m1", 0, 940), ("m2", 930, 624)]
        assert _assert_invariant(first) == 2
        assert len(second.get_events()) == 0
        row = first.get_events().table.to_pylist()[1]
        assert row["event_type"] == "Box"
        stored = Quadrilateral.from_points(
            [
                (corner["x"]["numerator"], corner["y"]["numerator"])
                for corner in row["quadrilateral"].values()
            ],
            unit="pixels",
        )
        assert stored == _quad(MEASURE_2)

    def test_a_coordinate_of_the_volume_reaches_its_accolade(
        self, volume: SegmentLine
    ) -> None:
        # Page 2 starts at 12311; its second accolade at 3142 on the page.
        stamp = volume.get_timestamp_at(12311 + 3142 + 5)
        coordinate = stamp.get_coordinate_for("dgt_9")
        assert (coordinate.value, coordinate.unit) == (5, TimeUnit.pixels)


class TestChoralePage:
    @pytest.fixture(scope="class")
    def page(self) -> Timeline:
        (page,) = build_pages(_Loader(), [_chorale_page()])
        return page

    def test_the_page(self, page: Timeline) -> None:
        assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
        assert (page.id, page.name) == ("surface_15", "15")
        assert page.length == Coordinate(5610, TimeUnit.pixels)
        assert page.is_locked
        assert page.meta == {"image": CHORALE_IMAGE}

    def test_the_accolades(self, page: Timeline) -> None:
        (first_offset, first), (second_offset, second) = _segments(page)
        assert (first_offset, second_offset) == (0, 2806)
        assert (first.length.value, second.length.value) == (2806, 2804)
        assert (first.name, second.name) == ("accolade_1", "accolade_2")
        assert (first.id, second.id) == ("dgt_2", "dgt_3")
        assert not first.is_locked and not second.is_locked
        assert first.meta == {
            "box": {"x": 490, "y": 344, "width": 2806, "height": 572},
            "image": CHORALE_IMAGE,
        }
        assert second.meta == {
            "box": {"x": 498, "y": 858, "width": 2804, "height": 529},
            "image": CHORALE_IMAGE,
        }

    def test_the_zones_are_boxes_of_their_accolade(self, page: Timeline) -> None:
        (_, first), (_, second) = _segments(page)
        assert _boxes(first) == [
            ("zone_1", 0, 483),
            ("zone_2", 454, 579),
            ("zone_3", 991, 601),
            ("zone_4", 1544, 660),
            ("zone_5", 2179, 627),
        ]
        assert _boxes(second) == [
            ("zone_6", 0, 998),
            ("zone_7", 967, 681),
            ("zone_8", 1612, 638),
            ("zone_9", 2206, 598),
        ]
        assert _assert_invariant(first) == 5
        assert _assert_invariant(second) == 4
        assert first.get_events().table.to_pylist()[1]["bounding_box"] == {
            "x": {"value": 944.0, "numerator": 944, "denominator": 1},
            "y": {"value": 368.0, "numerator": 368, "denominator": 1},
            "width": {"value": 579.0, "numerator": 579, "denominator": 1},
            "height": {"value": 496.0, "numerator": 496, "denominator": 1},
        }

    def test_boxes_without_source_ids_are_minted_per_accolade(self) -> None:
        zones = [PageBox(_rect(*zone)) for zone in CHORALE_ZONES]
        (page,) = build_pages(_Loader(), [_chorale_page(boxes=zones, source_id=None)])
        assert page.id == "dgt_1"
        (_, first), (_, second) = _segments(page)
        assert (first.id, second.id) == ("dgt_2", "dgt_3")
        assert [box[0] for box in _boxes(first)] == [f"box_{n}" for n in range(1, 6)]
        assert [box[0] for box in _boxes(second)] == [f"box_{n}" for n in range(1, 5)]

    def test_a_page_without_name_or_image(self) -> None:
        (page,) = build_pages(_Loader(), [_chorale_page(name=None, image=None)])
        assert page.name == "page_1"
        assert page.meta == {}
        assert _segments(page)[0][1].meta == {
            "box": {"x": 490, "y": 344, "width": 2806, "height": 572}
        }

    def test_statements(self) -> None:
        canvas = {"id": "https://example.org/canvas/15"}
        volume = build_volume(
            _Loader(),
            [_chorale_page(statements=[("dcterms:source", canvas)])],
            name="Chorale book",
            image="facsimile/",
            statements=[("dcterms:source", {"id": "https://example.org/manifest"})],
        )
        assert volume.name == "Chorale book"
        assert volume.meta == {"image": "facsimile/"}
        assert volume.statements["dcterms:source"] == [
            {"id": "https://example.org/manifest"}
        ]
        ((_, page),) = _segments(volume)
        assert page.statements["dcterms:source"] == [canvas]


class TestIdentity:
    def test_sequences_name_in_document_order(self) -> None:
        (page,) = build_pages(
            _Loader(),
            [_chorale_page()],
            page_ids=["p15"],
            accolade_ids=["upper", "lower"],
            box_ids=[f"m{n}" for n in range(1, 10)],
        )
        (_, first), (_, second) = _segments(page)
        assert (page.id, first.id, second.id) == ("p15", "upper", "lower")
        assert [box[0] for box in _boxes(first)] == ["m1", "m2", "m3", "m4", "m5"]
        assert [box[0] for box in _boxes(second)] == ["m6", "m7", "m8", "m9"]

    def test_templates_are_filled_from_positions(self) -> None:
        volume = build_volume(
            _Loader(),
            _scan_pages(),
            page_ids="page{page}",
            accolade_ids="p{page}_s{accolade}_{n}",
            box_ids="p{page}s{accolade}b{box}",
        )
        assert volume.id == "dgt_1"
        pages = [page for _, page in _segments(volume)]
        assert [page.id for page in pages] == ["page1", "page2", "page3", "page4"]
        assert [a.id for _, a in _segments(pages[0])] == [
            "p1_s1_1",
            "p1_s2_2",
            "p1_s3_3",
            "p1_s4_4",
        ]
        assert [a.id for _, a in _segments(pages[3])] == [
            "p4_s1_14",
            "p4_s2_15",
            "p4_s3_16",
            "p4_s4_17",
        ]
        assert [box[0] for box in _boxes(_segments(pages[0])[0][1])] == [
            "p1s1b1",
            "p1s1b2",
        ]

    def test_boxes_count_through_the_pages(self) -> None:
        first, second = _chorale_page(source_id="a"), _chorale_page(source_id="b")
        pages = build_pages(_Loader(), [first, second], box_ids="m{n}")
        ids = [
            [box[0] for box in _boxes(accolade)]
            for page in pages
            for _, accolade in _segments(page)
        ]
        assert ids == [
            ["m1", "m2", "m3", "m4", "m5"],
            ["m6", "m7", "m8", "m9"],
            ["m10", "m11", "m12", "m13", "m14"],
            ["m15", "m16", "m17", "m18"],
        ]

    def test_a_sequence_names_boxes_in_the_order_given(self) -> None:
        # Zones given alternating between the two lines: 1, 6, 2, 7, ...
        order = [0, 5, 1, 6, 2, 7, 3, 8, 4]
        zones = [PageBox(_rect(*CHORALE_ZONES[k])) for k in order]
        page = _chorale_page(
            boxes=zones,
            path=TimeAxisPath.from_boxes([_rect(*zone) for zone in CHORALE_ZONES]),
        )
        names = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
        (built,) = build_pages(_Loader(), [page], box_ids=names)
        (_, first), (_, second) = _segments(built)
        assert _boxes(first) == [
            ("a", 0, 483),
            ("c", 454, 579),
            ("e", 991, 601),
            ("g", 1544, 660),
            ("i", 2179, 627),
        ]
        assert _boxes(second) == [
            ("b", 0, 998),
            ("d", 967, 681),
            ("f", 1612, 638),
            ("h", 2206, 598),
        ]
        (templated,) = build_pages(_Loader(), [page], box_ids="n{n}_b{box}")
        assert [box[0] for box in _boxes(_segments(templated)[1][1])] == [
            "n2_b1",
            "n4_b2",
            "n6_b3",
            "n8_b4",
        ]

    def test_a_named_timeline_takes_its_number_all_the_same(self) -> None:
        (page,) = build_pages(_Loader(), [_chorale_page()], accolade_ids=["upper", "x"])
        assert page.id == "surface_15"
        loader = _Loader()
        build_pages(loader, [_chorale_page()], accolade_ids=["upper", "x"])
        assert _mint_timeline_id(loader, TimeUnit.pixels) == "dgt_4"

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (
                {"box_ids": ["m1", "m2"]},
                "box_ids names 2 Boxes; this call creates 9.",
            ),
            (
                {"accolade_ids": ["a", "b", "c"]},
                "accolade_ids names 3 accolades; this call creates 2.",
            ),
            ({"page_ids": []}, "page_ids names 0 pages; this call creates 1."),
            (
                {"box_ids": "m{measure}"},
                "box_ids holds the template field {measure}; a template is filled "
                "from page, accolade, box, n, of which a Box has page, accolade, "
                "box, n.",
            ),
            (
                {"page_ids": "p{accolade}"},
                "page_ids holds the template field {accolade}; a template is "
                "filled from page, accolade, box, n, of which a page has page, n.",
            ),
            (
                {"accolade_ids": "s{}"},
                "accolade_ids holds the template field {}; a template is filled "
                "from page, accolade, box, n, of which an accolade has page, "
                "accolade, n.",
            ),
            (
                {"accolade_ids": ["a", ""]},
                "Invalid id '': an id must be non-empty and contain no control "
                "character",
            ),
            (
                {"accolade_ids": "system"},
                "The id 'system' is given to accolade 'accolade_1' of page '15' and "
                "to accolade 'accolade_2' of page '15': every timeline of a call "
                "has its own id.",
            ),
            (
                {"accolade_ids": ["surface_15", "b"]},
                "The id 'surface_15' is given to page '15' and to accolade "
                "'accolade_1' of page '15': every timeline of a call has its own id.",
            ),
        ],
    )
    def test_refused_before_anything_is_built(
        self, arguments: dict[str, Any], message: str
    ) -> None:
        loader = _Loader()
        with pytest.raises(ValueError) as error:
            build_pages(loader, [_chorale_page()], **arguments)
        assert str(error.value) == message
        _assert_nothing_minted(loader)

    def test_the_volume_and_a_page_cannot_share_an_id(self) -> None:
        loader = _Loader()
        with pytest.raises(ValueError) as error:
            build_volume(loader, [_chorale_page()], uid="surface_15")
        assert str(error.value) == (
            "The id 'surface_15' is given to the volume and to page '15': every "
            "timeline of a call has its own id."
        )
        _assert_nothing_minted(loader)


class TestOtherPaths:
    def test_a_roll_along_a_vertical_line(self) -> None:
        path = VerticalLinePath(x=2048, y0=0, y1=299400, tolerance=2048)
        (roll,) = build_pages(_Loader(), [Page(name="roll", image="canvas", path=path)])
        assert type(roll) is DiscreteGraphicalTimeline
        assert (roll.id, roll.name) == ("dgt_1", "roll")
        assert roll.length == Coordinate(299400, TimeUnit.pixels)
        assert roll.is_locked
        assert roll.meta == {"image": "canvas"}

    def test_a_horizontal_line_states_its_strip_and_holds_boxes(self) -> None:
        path = HorizontalLinePath(x0=10, x1=500, y=100)
        zone = PageBox(_rect(20, 95, 50, 105), source_id="z1")
        (page,) = build_pages(
            _Loader(), [Page(name="p", image="img", path=path, boxes=[zone])]
        )
        assert type(page) is DiscreteGraphicalTimeline
        assert page.length == Coordinate(490, TimeUnit.pixels)
        assert page.is_locked
        assert page.meta == {
            "box": {"x": 10, "y": 90, "width": 490, "height": 20},
            "image": "img",
        }
        assert _boxes(page) == [("z1", 10, 30)]
        assert page.get_events().table.to_pylist()[0]["duration"]["numerator"] == (
            Duration(30, TimeUnit.pixels).value
        )

    def test_a_diagonal_of_whole_length(self) -> None:
        path = DiagonalLinePath(start=(0, 0), end=(300, 400))
        (page,) = build_pages(_Loader(), [Page(name="d", image=None, path=path)])
        assert page.length == Coordinate(500, TimeUnit.pixels)
        assert page.meta == {}

    def test_a_length_that_is_no_whole_pixel_is_refused(self) -> None:
        loader = _Loader()
        path = DiagonalLinePath(start=(0, 0), end=(100, 100))
        with pytest.raises(ValueError) as error:
            build_pages(loader, [Page(name="d", image=None, path=path)])
        assert str(error.value) == (
            "Page 'd': the length of its time path is 141.4213562373095, which "
            "pixels as int cannot hold exactly: geometry is never rounded."
        )
        _assert_nothing_minted(loader)

    def test_a_volume_of_plain_pages(self) -> None:
        pages = [
            Page(name="a", image="a.jpg", path=VerticalLinePath(x=5, y0=0, y1=700)),
            Page(name="b", image="b.jpg", path=VerticalLinePath(x=5, y0=10, y1=910)),
        ]
        volume = build_volume(_Loader(), pages)
        assert type(volume) is SegmentLine[DiscreteGraphicalTimeline]
        assert volume.length == Coordinate(1600, TimeUnit.pixels)
        assert volume.is_locked
        segments = _segments(volume)
        assert [offset for offset, _ in segments] == [0, 700]
        assert [page.id for _, page in segments] == ["dgt_2", "dgt_3"]
        assert [page.is_locked for _, page in segments] == [False, False]

    def test_a_piece_that_is_no_horizontal_line_states_no_area(self) -> None:
        path = PiecewisePath(
            (
                HorizontalLinePath(x0=0, x1=100, y=10, tolerance=5),
                VerticalLinePath(x=100, y0=10, y1=60, tolerance=5),
            )
        )
        (page,) = build_pages(_Loader(), [Page(name="p", image="img", path=path)])
        (_, first), (offset, second) = _segments(page)
        assert page.length == Coordinate(150, TimeUnit.pixels)
        assert first.meta == {
            "box": {"x": 0, "y": 5, "width": 100, "height": 10},
            "image": "img",
        }
        assert (offset, second.length.value) == (100, 50)
        assert second.meta == {"image": "img"}


class TestRefusals:
    def _refused(self, message: str, *pages: Page, volume: bool = False) -> None:
        loader = _Loader()
        with pytest.raises(ValueError) as error:
            (build_volume if volume else build_pages)(loader, list(pages))
        assert str(error.value) == message
        _assert_nothing_minted(loader)

    def test_a_box_on_no_piece(self) -> None:
        stray = PageBox(_rect(600, 100, 1000, 201))
        self._refused(
            "Page '15', Box 10: The shape spanning x 600 to 1000 with its centre "
            "at y 301/2 lies on no piece of the path.",
            _chorale_page(boxes=[*_chorale_page().boxes, stray]),
        )

    def test_a_box_leaving_its_accolade(self) -> None:
        system = _quad(SYSTEM_1)
        measure = _rect(1200, 1589, 2161, 2266)
        self._refused(
            "Page '1', accolade 1, Box 2 spans x 1200 to 2161, which leaves its "
            "accolade, spanning x 1221 to 4101.",
            Page(
                name="1",
                image=None,
                path=TimeAxisPath.from_boxes([system]),
                accolades=[
                    PageAccolade(
                        area=system, boxes=[PageBox(_quad(MEASURE_1)), PageBox(measure)]
                    )
                ],
            ),
        )

    def test_a_stated_area_that_is_not_its_piece(self) -> None:
        path = PiecewisePath(
            (HorizontalLinePath(x0=1221, x1=4100, y=1927, tolerance=338),)
        )
        self._refused(
            "Page '1', accolade 1 states an area spanning x 1221 to 4101, while "
            "its piece of the time path spans x 1221 to 4100: the two must agree.",
            Page(
                name="1",
                image=None,
                path=path,
                accolades=[PageAccolade(area=_quad(SYSTEM_1))],
            ),
        )

    def test_another_number_of_stated_accolades(self) -> None:
        self._refused(
            "Page '15' states 1 accolades; its time path has 2 pieces.",
            _chorale_page(accolades=[PageAccolade(name="only")]),
        )

    def test_accolades_on_a_path_without_pieces(self) -> None:
        self._refused(
            "Page 'p' states accolades, which are the pieces of a PiecewisePath; "
            "its time path is a HorizontalLinePath.",
            Page(
                name="p",
                image=None,
                path=HorizontalLinePath(x0=0, x1=10, y=5),
                accolades=[PageAccolade()],
            ),
        )

    def test_a_strip_that_is_no_whole_pixels(self) -> None:
        path = PiecewisePath((HorizontalLinePath(x0=0, x1=10, y=5, tolerance=2.5),))
        self._refused(
            "Page 'p', accolade 1: A shape's y is 2.5, which pixels as int cannot "
            "hold exactly: geometry is never rounded.",
            Page(name="p", image=None, path=path),
        )

    def test_boxes_on_a_vertical_line(self) -> None:
        self._refused(
            "Page 'roll', Box 1: Boxes are placed along horizontal lines only; the "
            "page's time path is a VerticalLinePath.",
            Page(
                name="roll",
                image=None,
                path=VerticalLinePath(x=50, y0=0, y1=100, tolerance=50),
                boxes=[PageBox(_rect(10, 10, 20, 20))],
            ),
        )

    def test_a_box_in_another_unit(self) -> None:
        box = BoundingBox.from_corners(1.0, 2.0, 3.0, 4.0, unit="seconds")
        self._refused(
            "Page 'p', Box 1 is in seconds as float; a page image is measured in "
            "pixels as int.",
            Page(
                name="p",
                image=None,
                path=HorizontalLinePath(x0=0, x1=10, y=5),
                boxes=[PageBox(box)],
            ),
        )

    def test_pages_of_two_kinds_in_one_volume(self) -> None:
        self._refused(
            "The pages of a volume are all built of accolades or none is: page "
            "'15' has a PiecewisePath, page 'roll' has none.",
            _chorale_page(),
            Page(name="roll", image=None, path=VerticalLinePath(x=5, y0=0, y1=70)),
            volume=True,
        )

    def test_a_volume_without_pages(self) -> None:
        self._refused("A volume needs at least one page.", volume=True)

    def test_no_pages_give_no_roots(self) -> None:
        assert build_pages(_Loader(), []) == []
