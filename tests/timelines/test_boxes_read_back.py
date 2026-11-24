"""A Box is read back by the calls that read any event type.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes_read_back.py`` - A Box is read back by the calls that
read any event").
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    BoundingBox,
    Box,
    BoxField,
    Coordinate,
    Event,
    Note,
    NumberType,
    Projection,
    ProjectionField,
    Quadrilateral,
)
from timetoalign.loader.graphical import DmosLoader, MeiFacsimileLoader
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    DiscreteGraphicalTimeline,
    Timeline,
)

#: Thoresen's ``rect_h2`` on its 875 x 270 image, and its second crop (x0 = 7).
RECT_H2 = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
CROP_2 = {"x": 7, "y": 0, "width": 867, "height": 270}
#: Where the crop stands on the page built of the crops: after the first (866).
CROP_2_OFFSET = 866

#: Two measure zones of the Anhalt chorale facsimile: xml:id, ulx, uly, lrx, lry.
MEI_ZONES = [
    ("zone_01M7DF2D8RNCKHKF6SMCWB94Z5", 490, 344, 973, 877),
    ("zone_01M7DF2D8R00HP8JYGXRTE13G3", 944, 368, 1523, 864),
]
#: *Aimons-nous* (DMOS), page 1, system 1 and its first two measures, as DMOS
#: writes them (systems TL, BL, TR, BR; measures TL, TR, BL, BR).
SCORE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688"
DMOS_SYSTEM = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
DMOS_MEASURES = [
    [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
    [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]],
]
#: The thirds box of a user-units timeline.
THIRDS = BoundingBox.from_xywh(
    Fraction(1, 3),
    0,
    Fraction(2, 3),
    1,
    unit="user_units",
    number_type="fraction",
)


def _projection() -> Projection:
    return Projection(id="p1", start=Coordinate(378, "pixels"))


def _box() -> Box:
    return Box.from_shape(RECT_H2, origin=7, id="rect_h2")


#: The event types whose field holds every property of their scalar.
EVENT_TYPES = [
    pytest.param(Box, BoxField, _box, id="Box"),
    pytest.param(Projection, ProjectionField, _projection, id="Projection"),
]


def _page_with(event: Event) -> tuple[Timeline, Timeline]:
    """A page holding a Note, and *event* on its child ``seg_2`` at 866."""
    page = DiscreteGraphicalTimeline(length=2000, uid="page")
    seg_2 = DiscreteGraphicalTimeline(length=867, uid="seg_2", meta={"box": CROP_2})
    page.add_child(seg_2, CROP_2_OFFSET)
    page.add_events([{"event_type": "Note", "id": "n1", "start": 0, "duration": 5}])
    seg_2.add_events(event)
    return page, seg_2


class TestTheSameCalls:
    """Every assertion is written once and run for each event type."""

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_the_events_of_a_type(
        self, event_type: type[Event], field_class: type, make: Any
    ) -> None:
        added = make()
        _, seg_2 = _page_with(added)
        field = seg_2.get_events(event_type=event_type.__name__).get_field(event_type)
        assert type(field) is field_class
        assert list(field) == [added]
        assert field[0].start.value == 378
        assert list(seg_2.events.get_field(event_type)) == [added]

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_a_child_s_event_on_its_parent(
        self, event_type: type[Event], field_class: type, make: Any
    ) -> None:
        added = make()
        page, _ = _page_with(added)
        events = page.get_events()
        assert events.column_values("id") == ["n1", added.id]
        field = events.get_field(event_type)
        assert field[0] is None
        read = field[1]
        assert type(read) is event_type
        assert read.start == Coordinate(CROP_2_OFFSET + 378, "pixels")
        assert read == added.model_copy(update={"start": read.start})

    @pytest.mark.parametrize(("event_type", "field_class", "make"), EVENT_TYPES)
    def test_a_field_filter_and_the_field_class(
        self, event_type: type[Event], field_class: type, make: Any
    ) -> None:
        added = make()
        page, _ = _page_with(added)
        events = page.get_events(id=added.id)
        field = events.get_field(event_type)
        assert len(field) == 1
        assert field[0].id == added.id
        assert events.get_field(field_class) is field

    def test_known_issue_a_note_is_not_read_back_as_a_field(self) -> None:
        page, _ = _page_with(_box())
        with pytest.raises(KeyError) as refused:
            page.get_events(event_type="Note").get_field(Note)
        assert refused.value.args[0].startswith(
            "No field matching 'NoteField' found in table."
        )


def _mei(tmp_path: Path) -> Path:
    zones = "".join(
        f'<zone xml:id="{zone}" type="measure" ulx="{ulx}" uly="{uly}" '
        f'lrx="{lrx}" lry="{lry}"/>'
        for zone, ulx, uly, lrx, lry in MEI_ZONES
    )
    text = f"""<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
  <meiHead><fileDesc><titleStmt><title>Ach bleib bei uns, Herr Jesu Christ</title>
  </titleStmt><pubStmt/></fileDesc></meiHead>
  <music><facsimile>
    <surface xml:id="surface_1" n="15" ulx="0" uly="0" lrx="3999" lry="2999">
      <graphic xml:id="graphic_1" target="facsimile/20251230_180023.jpg"
        width="4000" height="3000"/>{zones}
    </surface>
  </facsimile></music>
</mei>
"""
    path = tmp_path / "chorale_facsimile.mei"
    path.write_text(text, encoding="utf-8")
    return path


def _dmos(tmp_path: Path) -> Path:
    def measure(region: list[list[int]]) -> dict[str, Any]:
        return {
            "region": region,
            "headers": [{"no_staff": 1, "region": region}],
            "voices": [{"id": "1", "id_part": "P1", "elements": [{"no_group": 1}]}],
        }

    system = {
        "id": 1,
        "region": DMOS_SYSTEM,
        "headers": [
            {"id_part": "P1", "no_staff": 1, "name": "Voix", "region": DMOS_SYSTEM}
        ],
        "measures": [measure(region) for region in DMOS_MEASURES],
    }
    document = {
        "id": "C055_0",
        "date": "2025-06-10-11h",
        "score_image_url": SCORE_URL,
        "pages": [
            {
                "page_url": f"{SCORE_URL}/f2",
                "no_page": 1,
                "header_systems": {},
                "systems": [system],
            }
        ],
    }
    path = tmp_path / "iiif.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _mei_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """The MEI page, and its Boxes built from the zones' own numbers."""
    (page,) = MeiFacsimileLoader.from_file(_mei(tmp_path)).create_timelines()
    x0 = MEI_ZONES[0][1]
    expected = [
        Box.from_shape(
            BoundingBox.from_corners(ulx, uly, lrx, lry, unit="pixels"),
            origin=x0,
            id=zone,
        )
        for zone, ulx, uly, lrx, lry in MEI_ZONES
    ]
    assert [(box.start.value, box.duration.value) for box in expected] == [
        (0, 483),
        (454, 579),
    ]
    return page, expected


def _dmos_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """The DMOS page, and its Boxes built from the regions' own numbers."""
    (page,) = DmosLoader.from_file(_dmos(tmp_path)).create_timelines()
    order = ("top_left", "top_right", "bottom_left", "bottom_right")
    expected = [
        Box.from_shape(
            Quadrilateral.from_points(region, order=order, unit="pixels"),
            origin=DMOS_SYSTEM[0][0],
            id=f"box_{n}",
        )
        for n, region in enumerate(DMOS_MEASURES, start=1)
    ]
    assert [(box.start.value, box.duration.value) for box in expected] == [
        (0, 940),
        (930, 624),
    ]
    return page, expected


def _thirds_page(tmp_path: Path) -> tuple[Timeline, list[Box]]:
    """A user-units timeline written as fractions, holding the thirds Box."""
    page = ContinuousGraphicalTimeline(
        length=2, unit="user_units", number_type="fraction", uid="uu"
    )
    accolade = ContinuousGraphicalTimeline(
        length=1, unit="user_units", number_type="fraction", uid="uu_1"
    )
    page.add_child(accolade, 0)
    box = Box.from_shape(THIRDS, id="thirds")
    accolade.add_events(box)
    return page, [box]


PAGES = [
    pytest.param(_mei_page, int, id="MEI zones"),
    pytest.param(_dmos_page, int, id="DMOS measures"),
    pytest.param(_thirds_page, Fraction, id="thirds"),
]


def _boxes(page: Timeline) -> list[Box]:
    """The Boxes of the page's one accolade, read by the ordinary calls."""
    (accolade_id,) = page.list_children()
    accolade = page.get_child(accolade_id)
    return list(accolade.get_events(event_type="Box").get_field(Box))


def _numbers(box: Box) -> list[Any]:
    """Every number of a Box: its interval, then every number of its shape."""
    shape = box.shape
    if isinstance(shape, BoundingBox):
        members = [shape.x, shape.y, shape.width, shape.height]
    else:
        members = [c for corner in shape.corners for c in (corner.x, corner.y)]
    return [box.start.value, box.duration.value, *(m.value for m in members)]


class TestRoundTrips:
    """A Box added is the Box read back, in memory and through every written form."""

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_in_memory(self, tmp_path: Path, make: Any, number_type: type) -> None:
        page, expected = make(tmp_path)
        read = _boxes(page)
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_through_the_dict_form(
        self, tmp_path: Path, make: Any, number_type: type
    ) -> None:
        page, expected = make(tmp_path)
        read = _boxes(Timeline.from_dict(page.to_dict(events=True)))
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}

    @pytest.mark.parametrize(("make", "number_type"), PAGES)
    def test_through_the_package(
        self, tmp_path: Path, make: Any, number_type: type
    ) -> None:
        page, expected = make(tmp_path)
        bundle = AlignmentBundle(id="boxes")
        bundle.add_timeline(page)
        directory = tmp_path / "package"
        directory.mkdir()
        bundle.to_file(directory / "package.yaml")
        assert sorted(path.name for path in directory.iterdir()) == [
            "boxes.parquet",
            "package.yaml",
        ]
        read_page = AlignmentBundle.from_file(directory / "package.yaml").get_timeline(
            page.id
        )
        read = _boxes(read_page)
        assert read == expected
        assert {type(value) for box in read for value in _numbers(box)} == {number_type}
        assert read_page.number_type is page.number_type is NumberType(number_type)
