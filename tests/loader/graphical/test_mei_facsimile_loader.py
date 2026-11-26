"""Tests for MeiFacsimileLoader.

The fixtures are excerpts of an MEI file of the Anhaltisches Choralbuch digital
edition (``001_Ach bleib bei uns, Herr Jesu Christ_facsimile.mei``, CC BY 4.0):
its surface with all nine zones, and variations written by the test. The
validation logic is documented in ``tests/loader/graphical/README.md``
("``MeiFacsimileLoader``"). All expected values are exact and derived from the
source file by hand.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from tests.loader.minting import shape, shapes
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    BoundingBox,
    Box,
    Coordinate,
    Duration,
    NumberType,
    Shape,
    TimeUnit,
)
from timetoalign.loader.graphical import MeiFacsimileLoader
from timetoalign.loader.graphical.paths import (
    HorizontalLinePath,
    PiecewisePath,
    VerticalLinePath,
)
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

SURFACE = "surface_01M7DF2D8Q8FYGETPJXXMG03QP"
IMAGE = "facsimile/20251230_180023.jpg"
# The nine zones of the surface: xml:id, ulx, uly, lrx, lry.
ZONES = [
    ("zone_01M7DF2D8RNCKHKF6SMCWB94Z5", 490, 344, 973, 877),
    ("zone_01M7DF2D8R00HP8JYGXRTE13G3", 944, 368, 1523, 864),
    ("zone_01M7DF2D8R4E9KQCXCPC9RV59F", 1481, 388, 2082, 912),
    ("zone_01M7DF2D8S9S18YJ6CYJVF4QCJ", 2034, 402, 2694, 884),
    ("zone_01M7DF2D8SY6CN7SFAZFPE58FT", 2669, 389, 3296, 916),
    ("zone_01M7DF2D8SF9X38YEGDA0JCD04", 498, 858, 1496, 1337),
    ("zone_01M7DF2D8S1F9S83VHS68E6N7W", 1465, 885, 2146, 1353),
    ("zone_01M7DF2D8TV0F9DCHY6QNWYTDX", 2110, 886, 2748, 1357),
    ("zone_01M7DF2D8TSXCHCSA07C235VTV", 2704, 886, 3302, 1387),
]
ZONE_IDS = [zone[0] for zone in ZONES]
ZONE_1, ZONE_2 = ZONE_IDS[:2]


def _zone(zone_id: str, ulx: int, uly: int, lrx: int, lry: int, type_: str) -> str:
    kind = f' type="{type_}"' if type_ else ""
    return (
        f'\n        <zone xml:id="{zone_id}"{kind} ulx="{ulx}" uly="{uly}" '
        f'lrx="{lrx}" lry="{lry}"/>'
    )


GRAPHIC = (
    f'<graphic xml:id="graphic_01M7DF2D8RMG9SGXTZRR3S1BA3" target="{IMAGE}" '
    'width="4000" height="3000"/>'
)
EXCERPT_SURFACE = f"""
      <surface xml:id="{SURFACE}" n="15" ulx="0" uly="0" lrx="3999" lry="2999">
        {GRAPHIC}{"".join(_zone(*zone, "measure") for zone in ZONES)}
      </surface>"""

EXCERPT_BODY = f"""
    <body>
      <mdiv>
        <score>
          <section>
            <measure xml:id="m-43" metcon="false" n="0" facs="#{ZONE_1}"/>
            <measure xml:id="m-57" n="1" facs="#{ZONE_2}"/>
          </section>
        </score>
      </mdiv>
    </body>"""

ACCOLADE_1_BOX = {"x": 490, "y": 344, "width": 2806, "height": 572}
ACCOLADE_2_BOX = {"x": 498, "y": 858, "width": 2804, "height": 529}
# Per accolade: (zone number, start, duration, bounding box x, y, width, height).
ACCOLADE_1_ZONES = [
    (1, 0, 483, (490, 344, 483, 533)),
    (2, 454, 579, (944, 368, 579, 496)),
    (3, 991, 601, (1481, 388, 601, 524)),
    (4, 1544, 660, (2034, 402, 660, 482)),
    (5, 2179, 627, (2669, 389, 627, 527)),
]
ACCOLADE_2_ZONES = [
    (6, 0, 998, (498, 858, 998, 479)),
    (7, 967, 681, (1465, 885, 681, 468)),
    (8, 1612, 638, (2110, 886, 638, 471)),
    (9, 2206, 598, (2704, 886, 598, 501)),
]


def _mei(surfaces: str, body: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Ach bleib bei uns, Herr Jesu Christ</title></titleStmt>
  <pubStmt/></fileDesc></meiHead>
  <music>
    <facsimile>{surfaces}
    </facsimile>{body}
  </music>
</mei>
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "chorale_facsimile.mei"
    path.write_text(text, encoding="utf-8")
    return path


def _load(tmp_path: Path, surfaces: str, body: str = "") -> MeiFacsimileLoader:
    return MeiFacsimileLoader.from_file(_write(tmp_path, _mei(surfaces, body)))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    return _write(tmp_path, _mei(EXCERPT_SURFACE, EXCERPT_BODY))


def _px(value: int) -> Coordinate:
    return Coordinate(value, TimeUnit.pixels)


def _extent(value: int) -> Duration:
    return Duration(value, TimeUnit.pixels)


def _boxes(timeline: Timeline) -> list[Box]:
    """The Box events of a timeline, in table order."""
    table = timeline.get_events(include_children=False).table
    return [
        Box.from_row(row, unit=timeline.unit, number_type=timeline.number_type)
        for row in table.to_pylist()
        if row["event_type"] == "Box"
    ]


def _placed(timeline: Timeline) -> list[tuple[str, int, int]]:
    """Id, start and duration of the Boxes of a timeline."""
    return [(box.id, box.start.value, box.duration.value) for box in _boxes(timeline)]


def _segments(line: Timeline) -> list[Timeline]:
    return [line.get_child(child_id) for child_id in line.list_children()]


def _offsets(line: Timeline) -> list[int]:
    return [line.get_child_offset(child_id).value for child_id in line.list_children()]


def _expected(
    zones: list[tuple[int, int, int, tuple[int, int, int, int]]],
) -> list[Box]:
    return [
        Box(
            id=ZONE_IDS[number - 1],
            start=_px(start),
            duration=_extent(duration),
            bounding_box=BoundingBox.from_xywh(*area, unit="pixels"),
        )
        for number, start, duration, area in zones
    ]


def _assert_image_space_invariant(timeline: Timeline) -> None:
    """start = x_min - o(T) and duration = x_max - x_min for every Box of T."""
    stated = timeline.meta.get("box")
    origin = (
        0
        if stated is None
        else Shape.from_dict(
            stated, unit=timeline.unit, number_type=timeline.number_type
        ).x_min.value
    )
    for box in _boxes(timeline):
        assert box.start.value == box.shape.x_min.value - origin
        assert box.duration.value == box.shape.x_max.value - box.shape.x_min.value


def _surface(zones: str, *, graphic: str = "", attributes: str = "") -> str:
    return f"""
      <surface{attributes}>{graphic}{zones}
      </surface>"""


CANVAS = '<graphic target="page.jpg" width="4512" height="6174"/>'
SMALL_ZONE = '<zone xml:id="z1" ulx="100" uly="50" lrx="400" lry="90"/>'


class TestPaths:
    def test_the_zones_group_into_two_accolades(self, source: Path) -> None:
        assert MeiFacsimileLoader.from_file(source).create_paths() == {
            SURFACE: PiecewisePath(
                (
                    HorizontalLinePath(x0=490, x1=3296, y=630, tolerance=286),
                    HorizontalLinePath(
                        x0=498, x1=3302, y=Fraction(2245, 2), tolerance=Fraction(529, 2)
                    ),
                )
            )
        }

    def test_a_surface_without_an_id_is_keyed_by_its_position(
        self, tmp_path: Path
    ) -> None:
        loader = _load(
            tmp_path,
            _surface("", graphic=CANVAS) + _surface(SMALL_ZONE, graphic=CANVAS),
        )
        assert loader.create_paths() == {
            1: PiecewisePath((HorizontalLinePath(x0=100, x1=400, y=70, tolerance=20),))
        }


class TestPage:
    def test_the_page_is_a_segment_line_of_its_accolades(self, source: Path) -> None:
        (page,) = MeiFacsimileLoader.from_file(source).create_timelines()
        assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
        assert page.id == SURFACE
        assert page.name == "15"
        assert page.unit is TimeUnit.pixels
        assert page.number_type is NumberType.int
        assert page.length.value == 5610
        assert page.is_locked
        assert page.meta == {"image": IMAGE}
        assert _boxes(page) == []

        first, second = _segments(page)
        assert _offsets(page) == [0, 2806]
        assert [type(first), type(second)] == [DiscreteGraphicalTimeline] * 2
        assert (shape(first.id), first.name, first.length.value) == (
            "dgt",
            "accolade_1",
            2806,
        )
        assert (shape(second.id), second.name, second.length.value) == (
            "dgt",
            "accolade_2",
            2804,
        )
        assert first.meta == {"box": ACCOLADE_1_BOX, "image": IMAGE}
        assert second.meta == {"box": ACCOLADE_2_BOX, "image": IMAGE}

    def test_a_surface_without_id_and_n(self, tmp_path: Path) -> None:
        (page,) = _load(tmp_path, _surface(SMALL_ZONE)).create_timelines()
        assert (shape(page.id), page.name, page.length.value) == ("dgt", "page_1", 300)
        assert page.meta == {}
        (accolade,) = _segments(page)
        assert shape(accolade.id) == "dgt"
        assert accolade.meta == {"box": {"x": 100, "y": 50, "width": 300, "height": 40}}

    def test_a_surface_without_zones_is_left_out(self, tmp_path: Path) -> None:
        loader = _load(
            tmp_path,
            _surface("", graphic=CANVAS, attributes=' n="14"') + EXCERPT_SURFACE,
        )
        assert [page.name for page in loader.create_timelines()] == ["15"]
        assert list(loader.create_paths()) == [SURFACE]

    def test_pages_in_document_order(self, tmp_path: Path) -> None:
        loader = _load(
            tmp_path,
            EXCERPT_SURFACE
            + _surface(SMALL_ZONE, graphic=CANVAS, attributes=' n="16"'),
        )
        pages = loader.create_timelines()
        assert [page.id for page in pages][0] == SURFACE
        assert shape(pages[1].id) == "dgt"
        assert [(page.name, page.length.value) for page in pages] == [
            ("15", 5610),
            ("16", 300),
        ]
        assert shapes([accolade.id for accolade in _segments(pages[1])]) == ["dgt"] * 1


class TestVolume:
    def test_the_volume_is_a_segment_line_of_pages(self, source: Path) -> None:
        volume = MeiFacsimileLoader.from_file(source).create_timeline()
        assert type(volume) is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        assert shape(volume.id) == "dgt"
        assert volume.length.value == 5610
        assert volume.is_locked
        assert volume.meta == {}
        (page,) = _segments(volume)
        assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
        assert (page.id, page.name, page.length.value) == (SURFACE, "15", 5610)
        assert page.meta == {"image": IMAGE}
        assert shapes([accolade.id for accolade in _segments(page)]) == ["dgt"] * 2
        assert _expected(ACCOLADE_2_ZONES) == _boxes(_segments(page)[1])

    def test_uid_and_name(self, source: Path) -> None:
        volume = MeiFacsimileLoader.from_file(source).create_timeline(
            uid="choralbuch", name="Anhaltisches Choralbuch"
        )
        assert (volume.id, volume.name) == ("choralbuch", "Anhaltisches Choralbuch")

    def test_two_pages(self, tmp_path: Path) -> None:
        loader = _load(
            tmp_path,
            EXCERPT_SURFACE
            + _surface(SMALL_ZONE, graphic=CANVAS, attributes=' n="16"'),
        )
        volume = loader.create_timeline()
        assert volume.length.value == 5910
        assert _offsets(volume) == [0, 5610]
        assert [page.name for page in _segments(volume)] == ["15", "16"]

    def test_no_surface_with_zones(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, _surface("", graphic=CANVAS))
        assert loader.create_timelines() == []
        with pytest.raises(ValueError) as raised:
            loader.create_timeline()
        assert str(raised.value) == "A volume needs at least one page."


class TestZones:
    def test_zones_are_boxes_of_their_accolade(self, source: Path) -> None:
        (page,) = MeiFacsimileLoader.from_file(source).create_timelines()
        first, second = _segments(page)
        assert _boxes(first) == _expected(ACCOLADE_1_ZONES)
        assert _boxes(second) == _expected(ACCOLADE_2_ZONES)
        _assert_image_space_invariant(first)
        _assert_image_space_invariant(second)

    def test_a_zone_stays_in_the_accolade_it_was_grouped_into(
        self, tmp_path: Path
    ) -> None:
        # Zone b's centre (y 350) is nearer the second accolade's line (y 400)
        # than its own (y 200); the grouping, not the nearest line, places it.
        zones = (
            '<zone xml:id="a" ulx="100" uly="0" lrx="400" lry="100"/>'
            '<zone xml:id="b" ulx="400" uly="300" lrx="800" lry="400"/>'
            '<zone xml:id="c" ulx="100" uly="200" lrx="800" lry="600"/>'
        )
        (page,) = _load(tmp_path, _surface(zones)).create_timelines()
        first, second = _segments(page)
        assert first.meta == {"box": {"x": 100, "y": 0, "width": 700, "height": 400}}
        assert second.meta == {"box": {"x": 100, "y": 200, "width": 700, "height": 400}}
        assert _placed(first) == [("a", 0, 300), ("b", 300, 400)]
        assert _placed(second) == [("c", 0, 700)]

    def test_facs_links(self, source: Path) -> None:
        assert MeiFacsimileLoader.from_file(source).facs == {
            "m-43": (ZONE_1,),
            "m-57": (ZONE_2,),
        }

    def test_an_element_pointing_to_two_zones(self, tmp_path: Path) -> None:
        body = (
            '<body><mdiv><score><section><measure xml:id="m-1" '
            f'facs="#{ZONE_2} #{ZONE_1}"/></section></score></mdiv></body>'
        )
        assert _load(tmp_path, EXCERPT_SURFACE, body).facs == {"m-1": (ZONE_2, ZONE_1)}

    @pytest.mark.parametrize(
        "points",
        [
            # clockwise, from each corner
            "1221,1589 2151,1589 2161,2266 1221,2266",
            "2151,1589 2161,2266 1221,2266 1221,1589",
            "2161,2266 1221,2266 1221,1589 2151,1589",
            "1221,2266 1221,1589 2151,1589 2161,2266",
            # counter-clockwise, from each corner
            "1221,1589 1221,2266 2161,2266 2151,1589",
            "1221,2266 2161,2266 2151,1589 1221,1589",
            "2161,2266 2151,1589 1221,1589 1221,2266",
            "2151,1589 1221,1589 1221,2266 2161,2266",
        ],
    )
    def test_points_read_as_a_quadrilateral_from_any_corner(
        self, tmp_path: Path, points: str
    ) -> None:
        zone = f'<zone xml:id="m1" points="{points}"/>'
        (page,) = _load(tmp_path, _surface(zone, graphic=CANVAS)).create_timelines()
        (accolade,) = _segments(page)
        (box,) = _boxes(accolade)
        assert box.id == "m1"
        assert box.bounding_box is None
        assert box.quadrilateral.to_dict() == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 2151, "y": 1589},
            "bottom_right": {"x": 2161, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        assert (box.start, box.duration) == (_px(0), _extent(940))
        assert accolade.meta == {
            "box": {"x": 1221, "y": 1589, "width": 940, "height": 677},
            "image": "page.jpg",
        }
        _assert_image_space_invariant(accolade)

    def test_zones_without_coordinates_are_listed(self, tmp_path: Path) -> None:
        zones = (
            '<zone xml:id="empty_1" type="measure"/>'
            f'<zone xml:id="{ZONE_1}" ulx="520" uly="390" lrx="874" lry="1019"/>'
            '<zone xml:id="empty_2"/>'
        )
        loader = _load(tmp_path, _surface(zones, graphic=CANVAS))
        assert loader.zones_without_coordinates == ("empty_1", "empty_2")
        (page,) = loader.create_timelines()
        (accolade,) = _segments(page)
        assert _placed(accolade) == [(ZONE_1, 0, 354)]

    def test_a_zone_without_an_id_is_minted(self, tmp_path: Path) -> None:
        zone = '<zone ulx="520" uly="390" lrx="874" lry="1019"/>'
        (page,) = _load(tmp_path, _surface(zone, graphic=CANVAS)).create_timelines()
        (accolade,) = _segments(page)
        assert shapes([box.id for box in _boxes(accolade)]) == ["box"] * 1


TYPED_ZONES = (
    _zone("staff_1", 300, 300, 3400, 950, "staff")
    + _zone(*ZONES[0], "measure")
    + _zone(*ZONES[1], "measure")
    + _zone("untyped", 1000, 400, 1100, 500, "")
)


class TestZoneTypes:
    def test_one_type(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, _surface(TYPED_ZONES, graphic=CANVAS))
        (page,) = loader.create_timelines(zone_types="measure")
        (accolade,) = _segments(page)
        assert page.length.value == 1033
        assert accolade.meta["box"] == {
            "x": 490,
            "y": 344,
            "width": 1033,
            "height": 533,
        }
        assert _placed(accolade) == [(ZONE_1, 0, 483), (ZONE_2, 454, 579)]
        assert loader.create_paths(zone_types="measure") == {
            0: PiecewisePath(
                (
                    HorizontalLinePath(
                        x0=490, x1=1523, y=Fraction(1221, 2), tolerance=Fraction(533, 2)
                    ),
                )
            )
        }

    def test_every_zone_by_default(self, tmp_path: Path) -> None:
        (page,) = _load(
            tmp_path, _surface(TYPED_ZONES, graphic=CANVAS)
        ).create_timelines()
        (accolade,) = _segments(page)
        assert page.length.value == 3100
        assert _placed(accolade) == [
            ("staff_1", 0, 3100),
            (ZONE_1, 190, 483),
            (ZONE_2, 644, 579),
            ("untyped", 700, 100),
        ]

    def test_several_types(self, tmp_path: Path) -> None:
        volume = _load(tmp_path, _surface(TYPED_ZONES, graphic=CANVAS)).create_timeline(
            zone_types=["measure", "staff"]
        )
        (page,) = _segments(volume)
        (accolade,) = _segments(page)
        assert [box.id for box in _boxes(accolade)] == ["staff_1", ZONE_1, ZONE_2]

    def test_a_surface_none_of_whose_zones_takes_part_is_left_out(
        self, tmp_path: Path
    ) -> None:
        loader = _load(tmp_path, _surface(TYPED_ZONES, graphic=CANVAS))
        assert loader.create_timelines(zone_types="system") == []
        assert loader.create_paths(zone_types="system") == {}


WRITTEN = PiecewisePath(
    (
        HorizontalLinePath(x0=400, x1=3400, y=630, tolerance=300),
        HorizontalLinePath(x0=400, x1=3400, y=1120, tolerance=300),
    )
)


class TestGivenPaths:
    @pytest.mark.parametrize(
        "paths",
        [WRITTEN, {SURFACE: WRITTEN}, {0: WRITTEN}],
        ids=["one", "id", "position"],
    )
    def test_a_written_path_replaces_the_grouping(
        self, source: Path, paths: object
    ) -> None:
        (page,) = MeiFacsimileLoader.from_file(source).create_timelines(paths=paths)
        first, second = _segments(page)
        assert page.length.value == 6000
        assert _offsets(page) == [0, 3000]
        assert first.meta["box"] == {"x": 400, "y": 330, "width": 3000, "height": 600}
        assert second.meta["box"] == {"x": 400, "y": 820, "width": 3000, "height": 600}
        assert _placed(first) == [
            (ZONE_IDS[0], 90, 483),
            (ZONE_IDS[1], 544, 579),
            (ZONE_IDS[2], 1081, 601),
            (ZONE_IDS[3], 1634, 660),
            (ZONE_IDS[4], 2269, 627),
        ]
        assert _placed(second) == [
            (ZONE_IDS[5], 98, 998),
            (ZONE_IDS[6], 1065, 681),
            (ZONE_IDS[7], 1710, 638),
            (ZONE_IDS[8], 2304, 598),
        ]
        _assert_image_space_invariant(first)
        _assert_image_space_invariant(second)

    def test_one_horizontal_line_is_one_plain_timeline(self, tmp_path: Path) -> None:
        loader = _load(
            tmp_path, _surface(SMALL_ZONE, graphic=CANVAS, attributes=' n="3"')
        )
        line = HorizontalLinePath(x0=0, x1=1000, y=70, tolerance=30)
        (page,) = loader.create_timelines(paths=line)
        assert type(page) is DiscreteGraphicalTimeline
        assert (shape(page.id), page.name, page.length.value) == ("dgt", "3", 1000)
        assert page.is_locked
        assert page.meta == {
            "box": {"x": 0, "y": 40, "width": 1000, "height": 60},
            "image": "page.jpg",
        }
        assert _placed(page) == [("z1", 100, 300)]
        _assert_image_space_invariant(page)

    def test_a_vertical_line_where_zones_take_part(self, source: Path) -> None:
        roll = VerticalLinePath(x=2000, y0=0, y1=3000, tolerance=2000)
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader.from_file(source).create_timelines(paths=roll)
        assert str(raised.value) == (
            f"The time path given for MEI surface '{SURFACE}' is a "
            "VerticalLinePath; its 9 zones are placed along a HorizontalLinePath "
            "or the pieces of a PiecewisePath."
        )

    def test_a_path_for_a_surface_without_zones(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, _surface("", graphic=CANVAS, attributes=' n="roll"'))
        roll = VerticalLinePath(x=2000, y0=0, y1=2000, tolerance=2000)
        (page,) = loader.create_timelines(paths={0: roll})
        assert type(page) is DiscreteGraphicalTimeline
        assert (page.name, page.length.value) == ("roll", 2000)
        assert page.meta == {"image": "page.jpg"}

    def test_one_path_for_two_surfaces(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, EXCERPT_SURFACE + _surface(SMALL_ZONE))
        with pytest.raises(ValueError) as raised:
            loader.create_timelines(paths=WRITTEN)
        assert str(raised.value) == (
            "One time path names one surface; this file holds 2: give a mapping "
            "from each surface's xml:id or position to its path."
        )

    def test_a_surface_the_file_does_not_hold(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, EXCERPT_SURFACE + _surface(SMALL_ZONE))
        for key in ("surface_9", 2):
            with pytest.raises(KeyError) as raised:
                loader.create_timeline(paths={key: WRITTEN})
            assert raised.value.args[0] == (
                f"paths names the surface {key!r}, which this file does not hold. "
                f"Available surfaces: ['{SURFACE}', None] (or their positions "
                "from 0)."
            )

    def test_one_surface_named_twice(self, source: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader.from_file(source).create_timelines(
                paths={SURFACE: WRITTEN, 0: WRITTEN}
            )
        assert str(raised.value) == (
            f"paths gives MEI surface '{SURFACE}' two paths, one by its xml:id and "
            "one by its position."
        )

    def test_a_zone_on_no_piece(self, source: Path) -> None:
        short = PiecewisePath(
            (HorizontalLinePath(x0=400, x1=3400, y=630, tolerance=300),)
        )
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader.from_file(source).create_timelines(paths=short)
        assert str(raised.value) == (
            "Page '15', Box 6: The shape spanning x 498 to 1496 with its centre "
            "at y 2195/2 lies on no piece of the path."
        )

    def test_the_second_surface_alone_is_given_a_path(self, tmp_path: Path) -> None:
        loader = _load(tmp_path, EXCERPT_SURFACE + _surface(SMALL_ZONE))
        line = PiecewisePath((HorizontalLinePath(x0=0, x1=1000, y=70, tolerance=30),))
        first, second = loader.create_timelines(paths={1: line})
        assert (first.length.value, second.length.value) == (5610, 1000)
        assert _placed(_segments(second)[0]) == [("z1", 100, 300)]


class TestIdentity:
    def test_box_ids_number_the_measures_through_the_page(self, source: Path) -> None:
        loader = MeiFacsimileLoader.from_file(source)
        (page,) = loader.create_timelines(
            box_ids="m{n}", accolade_ids="p{page}_s{accolade}", page_ids=["folio"]
        )
        first, second = _segments(page)
        assert page.id == "folio"
        assert (first.id, second.id) == ("p1_s1", "p1_s2")
        assert [box.id for box in _boxes(first)] == ["m1", "m2", "m3", "m4", "m5"]
        assert [box.id for box in _boxes(second)] == ["m6", "m7", "m8", "m9"]
        assert loader.facs == {"m-43": (ZONE_1,), "m-57": (ZONE_2,)}

    def test_a_sequence_of_another_length(self, source: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader.from_file(source).create_timeline(
                box_ids=[f"m{n}" for n in range(8)]
            )
        assert str(raised.value) == "box_ids names 8 Boxes; this call creates 9."


class TestTheImageSpace:
    def test_a_zone_past_the_image_width(self, tmp_path: Path) -> None:
        zone = '<zone xml:id="z9" ulx="4000" uly="0" lrx="4600" lry="10"/>'
        path = _write(tmp_path, _mei(_surface(zone, graphic=CANVAS)))
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            "MEI zone 'z9' ends at x 4600, past the right edge of its image at x 4512."
        )

    def test_a_size_in_centimetres_states_no_pixel_width(self, tmp_path: Path) -> None:
        graphic = '<graphic target="page.jpg" width="21cm" height="29.7cm"/>'
        zone = '<zone xml:id="z9" ulx="4000" uly="0" lrx="4600" lry="10"/>'
        (page,) = _load(tmp_path, _surface(zone, graphic=graphic)).create_timelines()
        assert page.length.value == 600
        assert page.meta == {"image": "page.jpg"}

    def test_the_surface_corners_are_not_read(self, tmp_path: Path) -> None:
        # Known issue: the surface's corners state another area; unused.
        graphic = '<graphic target="page.jpg" width="4000px" height="3000px"/>'
        surface_corners = ' ulx="100" uly="0" lrx="4100" lry="3000"'
        zone = '<zone xml:id="z1" ulx="150" uly="10" lrx="200" lry="20"/>'
        (page,) = _load(
            tmp_path, _surface(zone, graphic=graphic, attributes=surface_corners)
        ).create_timelines()
        (accolade,) = _segments(page)
        assert page.meta == {"image": "page.jpg"}
        assert accolade.meta["box"] == {"x": 150, "y": 10, "width": 50, "height": 10}
        assert _placed(accolade) == [("z1", 0, 50)]

        past = '<zone xml:id="z2" ulx="3900" uly="10" lrx="4050" lry="20"/>'
        path = _write(
            tmp_path, _mei(_surface(past, graphic=graphic, attributes=surface_corners))
        )
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            "MEI zone 'z2' ends at x 4050, past the right edge of its image at x 4000."
        )


class TestPointsThatAreNoQuadrilateral:
    """Known issue: ``@points`` other than a convex quadrilateral refuse the file."""

    @staticmethod
    def _refusal(tmp_path: Path, points: str) -> str:
        zone = f'<zone xml:id="m1" points="{points}"/>'
        path = _write(tmp_path, _mei(_surface(zone, graphic=CANVAS)))
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        return str(raised.value)

    def test_five_points(self, tmp_path: Path) -> None:
        points = "1221,1589 2151,1589 2161,2266 1221,2266 1221,1900"
        assert self._refusal(tmp_path, points) == (
            f"MEI zone 'm1': The SVG points '{points}' hold 5 points; a "
            "Quadrilateral's are four."
        )

    def test_a_concave_outline(self, tmp_path: Path) -> None:
        assert self._refusal(tmp_path, "0,0 10,0 4,4 0,10") == (
            "MEI zone 'm1': The SVG points '0,0 10,0 4,4 0,10' bound no convex "
            "quadrilateral: the outline turns clockwise at points 1, 2 and 4 and "
            "counter-clockwise at point 3."
        )

    def test_a_crossed_outline(self, tmp_path: Path) -> None:
        points = "1221,1589 2151,1589 1221,2266 2161,2266"
        assert self._refusal(tmp_path, points) == (
            f"MEI zone 'm1': The SVG points '{points}' bound no convex "
            "quadrilateral: the outline turns clockwise at points 1 and 2 and "
            "counter-clockwise at points 3 and 4."
        )


class TestPackageRoundTrip:
    def test_the_volume_s_boxes_survive_a_package(
        self, source: Path, tmp_path: Path
    ) -> None:
        volume = MeiFacsimileLoader.from_file(source).create_timeline(uid="chorale")
        bundle = AlignmentBundle(id="choralbuch")
        bundle.add_timeline(volume)
        path = bundle.to_file(tmp_path / "package" / "package.yaml")
        assert sorted(entry.name for entry in path.parent.iterdir()) == [
            "boxes.parquet",
            "package.yaml",
        ]

        read = AlignmentBundle.from_file(path).get_timeline("chorale")

        assert isinstance(read, SegmentLine)
        assert read.is_locked
        assert read.length == _px(5610)
        assert read.meta == {}
        (page_id,) = read.list_children()
        page = read.get_child(page_id)
        assert (page.id, page.name, page.length) == (SURFACE, "15", _px(5610))
        assert type(page).__name__ == "SegmentLine[DiscreteGraphicalTimeline]"
        assert page.is_locked
        assert page.meta == {"image": IMAGE}
        assert len(page.get_events(include_children=False)) == 0
        accolades = [page.get_child(uid) for uid in page.list_children()]
        assert [
            (shape(a.id), a.name, page.get_child_offset(a.id).value, a.length.value)
            for a in accolades
        ] == [("dgt", "accolade_1", 0, 2806), ("dgt", "accolade_2", 2806, 2804)]
        assert [a.meta for a in accolades] == [
            {"box": ACCOLADE_1_BOX, "image": IMAGE},
            {"box": ACCOLADE_2_BOX, "image": IMAGE},
        ]
        for accolade, zones in zip(accolades, (ACCOLADE_1_ZONES, ACCOLADE_2_ZONES)):
            assert _boxes(accolade) == _expected(zones)
            _assert_image_space_invariant(accolade)


class TestLifecycle:
    def test_no_mei_root(self, tmp_path: Path) -> None:
        path = _write(tmp_path, '<svg xmlns="http://www.w3.org/2000/svg"/>')
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            f"{path} is no MEI document: its root "
            "<{http://www.w3.org/2000/svg}svg> is not in the MEI namespace "
            "http://www.music-encoding.org/ns/mei."
        )

    def test_nothing_loaded(self) -> None:
        loader = MeiFacsimileLoader()
        for create in (
            loader.create_timeline,
            loader.create_timelines,
            loader.create_paths,
        ):
            with pytest.raises(RuntimeError) as raised:
                create()
            assert str(raised.value) == "No MEI file loaded. Call load() first."

    def test_one_source(self, source: Path, tmp_path: Path) -> None:
        loader = MeiFacsimileLoader.from_file(source)
        other = tmp_path / "other.mei"
        with pytest.raises(ValueError) as raised:
            loader.load(other)
        assert str(raised.value) == (
            f"MeiFacsimileLoader reads one source and already holds {source}; use a "
            f"new loader for {other}, or call clear() first."
        )

    def test_repr(self, source: Path) -> None:
        assert repr(MeiFacsimileLoader()) == "MeiFacsimileLoader(not loaded)"
        assert repr(MeiFacsimileLoader.from_file(source)) == (
            "MeiFacsimileLoader(surfaces=1, zones=9)"
        )
