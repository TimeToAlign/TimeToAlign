"""Tests for MeiFacsimileLoader.

The fixtures are excerpts of an MEI file of the Anhaltisches Choralbuch digital
edition (``061_Gelobet seist du Jesu Christ_facsimile.mei``, CC BY 4.0) and
variations of its surface, written by the test. The validation logic is
documented in ``tests/loader/graphical/README.md`` ("``MeiFacsimileLoader``").
All expected values are exact and read from the source file by hand.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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
from timetoalign.timelines import DiscreteGraphicalTimeline, Timeline

SURFACE = "surface_01M7DF2DF3J8BVY6RSGDQBJ6W6"
ZONE_1 = "zone_01M7DF2DF4MQT5E96R26XFVEYG"
ZONE_2 = "zone_01M7DF2DF4R34CYF6DVH8GZD9P"
IMAGE = "../facsimile/20251230_181354.jpg"

EXCERPT_SURFACE = f"""
      <surface xml:id="{SURFACE}" n="54" ulx="0" uly="0" lrx="3999" lry="2999">
        <graphic xml:id="graphic_01M7DF2DF43DZBVPP3758RC994" target="{IMAGE}" width="4000" height="3000"/>
        <zone xml:id="{ZONE_1}" type="measure" ulx="520" uly="390" lrx="874" lry="1019"/>
        <zone xml:id="{ZONE_2}" type="measure" ulx="814" uly="408" lrx="1129" lry="1033"/>
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


def _mei(surfaces: str, body: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Gelobet seist du Jesu Christ</title></titleStmt>
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


class TestSurface:
    def test_the_surface_is_a_root_of_the_image_width(self, source: Path) -> None:
        (surface,) = MeiFacsimileLoader.from_file(source).create_timelines()
        assert type(surface) is DiscreteGraphicalTimeline
        assert surface.id == SURFACE
        assert surface.name == "54"
        assert surface.unit is TimeUnit.pixels
        assert surface.number_type is NumberType.int
        assert surface.length.value == 4000
        assert surface.is_locked
        assert surface.meta == {
            "image": IMAGE,
            "box": {"x": 0, "y": 0, "width": 4000, "height": 3000},
        }

    def test_without_a_graphic_the_surface_corners_state_the_area(
        self, tmp_path: Path
    ) -> None:
        loader = _load(
            tmp_path,
            _surface(
                '<zone xml:id="z1" ulx="150" uly="10" lrx="200" lry="20"/>',
                attributes=' ulx="100" uly="0" lrx="600" lry="400"',
            ),
        )
        (surface,) = loader.create_timelines()
        assert surface.id == "dgt_1"
        assert surface.name == "dgt_1"
        assert surface.length.value == 500
        assert surface.meta == {"box": {"x": 100, "y": 0, "width": 500, "height": 400}}
        assert _boxes(surface) == [
            Box(
                id="z1",
                start=_px(50),
                duration=_extent(50),
                bounding_box=BoundingBox.from_xywh(150, 10, 50, 10, unit="pixels"),
            )
        ]
        _assert_image_space_invariant(surface)

    def test_a_size_in_centimetres_states_no_pixel_size(self, tmp_path: Path) -> None:
        graphic = '<graphic target="page.jpg" width="21cm" height="29.7cm"/>'
        (surface,) = _load(tmp_path, _surface("", graphic=graphic)).create_timelines()
        assert surface.length is None
        assert not surface.is_locked
        assert surface.meta == {"image": "page.jpg"}

    def test_a_size_in_px(self, tmp_path: Path) -> None:
        graphic = '<graphic target="page.jpg" width="4000px" height="3000px"/>'
        (surface,) = _load(tmp_path, _surface("", graphic=graphic)).create_timelines()
        assert surface.length.value == 4000
        assert surface.meta == {
            "image": "page.jpg",
            "box": {"x": 0, "y": 0, "width": 4000, "height": 3000},
        }

    def test_the_graphic_size_decides_the_space_not_the_surface_corners(
        self, tmp_path: Path
    ) -> None:
        # Known issue: the surface's corners state another area; unused.
        graphic = '<graphic target="page.jpg" width="4000" height="3000"/>'
        surface_corners = ' ulx="100" uly="0" lrx="4100" lry="3000"'
        zone = '<zone xml:id="z1" ulx="150" uly="10" lrx="200" lry="20"/>'
        (surface,) = _load(
            tmp_path, _surface(zone, graphic=graphic, attributes=surface_corners)
        ).create_timelines()
        assert surface.length.value == 4000
        assert surface.meta == {
            "image": "page.jpg",
            "box": {"x": 0, "y": 0, "width": 4000, "height": 3000},
        }
        assert [(box.id, box.start, box.duration) for box in _boxes(surface)] == [
            ("z1", _px(150), _extent(50))
        ]

        past = '<zone xml:id="z2" ulx="3900" uly="10" lrx="4050" lry="20"/>'
        path = _write(
            tmp_path, _mei(_surface(past, graphic=graphic, attributes=surface_corners))
        )
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            "MEI zone 'z2' ends at x 4050, past the right edge of its image at x 4000."
        )

    def test_surfaces_in_document_order(self, tmp_path: Path) -> None:
        loader = _load(
            tmp_path,
            EXCERPT_SURFACE + _surface("", graphic=CANVAS, attributes=' n="55"'),
        )
        surfaces = loader.create_timelines()
        assert [(surface.id, surface.name) for surface in surfaces] == [
            (SURFACE, "54"),
            ("dgt_2", "55"),
        ]


class TestZones:
    def test_zones_are_boxes_with_their_ids(self, source: Path) -> None:
        (surface,) = MeiFacsimileLoader.from_file(source).create_timelines()
        assert _boxes(surface) == [
            Box(
                id=ZONE_1,
                start=_px(520),
                duration=_extent(354),
                bounding_box=BoundingBox.from_xywh(520, 390, 354, 629, unit="pixels"),
            ),
            Box(
                id=ZONE_2,
                start=_px(814),
                duration=_extent(315),
                bounding_box=BoundingBox.from_xywh(814, 408, 315, 625, unit="pixels"),
            ),
        ]
        _assert_image_space_invariant(surface)

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
        (surface,) = _load(tmp_path, _surface(zone, graphic=CANVAS)).create_timelines()
        (box,) = _boxes(surface)
        assert box.id == "m1"
        assert box.bounding_box is None
        assert box.quadrilateral.to_dict() == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 2151, "y": 1589},
            "bottom_right": {"x": 2161, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        assert (box.start, box.duration) == (_px(1221), _extent(940))
        _assert_image_space_invariant(surface)


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

    def test_a_zone_past_the_image_width(self, tmp_path: Path) -> None:
        zone = '<zone xml:id="z9" ulx="4000" uly="0" lrx="4600" lry="10"/>'
        path = _write(tmp_path, _mei(_surface(zone, graphic=CANVAS)))
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            "MEI zone 'z9' ends at x 4600, past the right edge of its image at x 4512."
        )

    def test_zones_without_coordinates_are_listed(self, tmp_path: Path) -> None:
        zones = (
            '<zone xml:id="empty_1" type="measure"/>'
            f'<zone xml:id="{ZONE_1}" ulx="520" uly="390" lrx="874" lry="1019"/>'
            '<zone xml:id="empty_2"/>'
        )
        loader = _load(tmp_path, _surface(zones, graphic=CANVAS))
        assert loader.zones_without_coordinates == ("empty_1", "empty_2")
        (surface,) = loader.create_timelines()
        assert [box.id for box in _boxes(surface)] == [ZONE_1]

    def test_a_zone_without_an_id_is_minted(self, tmp_path: Path) -> None:
        zone = '<zone ulx="520" uly="390" lrx="874" lry="1019"/>'
        (surface,) = _load(tmp_path, _surface(zone, graphic=CANVAS)).create_timelines()
        assert [box.id for box in _boxes(surface)] == ["box_1"]


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
        with pytest.raises(RuntimeError) as raised:
            MeiFacsimileLoader().create_timelines()
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
            "MeiFacsimileLoader(surfaces=1, zones=2)"
        )
