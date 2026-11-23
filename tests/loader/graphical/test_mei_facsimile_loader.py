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
    Quadrilateral,
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

    def test_points_read_as_a_quadrilateral(self, tmp_path: Path) -> None:
        zone = '<zone xml:id="m1" points="1221,1589 2151,1589 2161,2266 1221,2266"/>'
        (surface,) = _load(tmp_path, _surface(zone, graphic=CANVAS)).create_timelines()
        quadrilateral = Quadrilateral.from_points(
            [(1221, 1589), (2151, 1589), (2161, 2266), (1221, 2266)], unit="pixels"
        )
        assert _boxes(surface) == [
            Box(
                id="m1",
                start=_px(1221),
                duration=_extent(940),
                quadrilateral=quadrilateral,
            )
        ]
        _assert_image_space_invariant(surface)

    def test_crossed_points_are_refused(self, tmp_path: Path) -> None:
        zone = '<zone xml:id="m1" points="1221,1589 2151,1589 1221,2266 2161,2266"/>'
        path = _write(tmp_path, _mei(_surface(zone, graphic=CANVAS)))
        with pytest.raises(ValueError) as raised:
            MeiFacsimileLoader().load(path)
        assert str(raised.value) == (
            "MEI zone 'm1': The corners top_left (1221, 1589), top_right (2151, 1589), "
            "bottom_right (1221, 2266), bottom_left (2161, 2266) do not run clockwise "
            "around a convex outline: the turn at bottom_right is counter-clockwise."
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
