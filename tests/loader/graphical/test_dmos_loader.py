"""Tests for DmosLoader.

The fixture is an excerpt of a DMOS JSON file (page 1 of the CollabScore
specimen *Aimons-nous*, C055), written by the test. The validation logic is
documented in ``tests/loader/graphical/README.md`` ("``DmosLoader``"). All
expected values are exact and read from the source file by hand.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from timetoalign.core import (
    Box,
    Coordinate,
    Duration,
    NumberType,
    Quadrilateral,
    Shape,
    TimeUnit,
)
from timetoalign.loader.graphical import DmosLoader
from timetoalign.timelines import DiscreteGraphicalTimeline, Timeline

PAGE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2"
SYSTEM_1 = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
SYSTEM_2 = [[971, 2493], [971, 3192], [4104, 2493], [4112, 3192]]
MEASURE_1 = [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]]
MEASURE_2 = [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]]


def _canonical(*corners: tuple[int, int]) -> Quadrilateral:
    """A quadrilateral of corners given top left, top right, bottom right, bottom left."""
    return Quadrilateral.from_points(list(corners), unit="pixels")


def _px(value: int) -> Coordinate:
    return Coordinate(value, TimeUnit.pixels)


def _extent(value: int) -> Duration:
    return Duration(value, TimeUnit.pixels)


def _measure(region: list[list[int]]) -> dict[str, Any]:
    """A DMOS measure: its region, and headers and voices the loader leaves alone."""
    return {
        "region": region,
        "headers": [{"no_staff": 1, "region": region}],
        "voices": [{"id": "1", "id_part": "P1", "elements": [{"no_group": 1}]}],
    }


def _system(system_id: int, region: list[list[int]], measures: list) -> dict[str, Any]:
    return {
        "id": system_id,
        "region": region,
        "headers": [{"id_part": "P1", "no_staff": 1, "name": "Voix", "region": region}],
        "measures": measures,
    }


def _document(systems: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "C055_0",
        "date": "2025-06-10-11h",
        "score_image_url": "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688",
        "pages": [
            {
                "page_url": PAGE_URL,
                "no_page": 1,
                "header_systems": {},
                "systems": systems,
            }
        ],
    }


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    path = tmp_path / "iiif.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """Page 1: system 1 with measures 1 and 2, system 2 with none."""
    return _write(
        tmp_path,
        _document(
            [
                _system(1, SYSTEM_1, [_measure(MEASURE_1), _measure(MEASURE_2)]),
                _system(2, SYSTEM_2, []),
            ]
        ),
    )


def _boxes(timeline: Timeline) -> list[Box]:
    """The Box events a timeline holds itself, in table order."""
    table = timeline.get_events(include_children=False).table
    return [
        Box.from_row(row, unit=timeline.unit, number_type=timeline.number_type)
        for row in table.to_pylist()
        if row["event_type"] == "Box"
    ]


def _assert_image_space_invariant(timeline: Timeline) -> None:
    """start = x_min - o(T) and duration = x_max - x_min, on T and its subtree."""
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
    for child_id in timeline.list_children():
        _assert_image_space_invariant(timeline.get_child(child_id))


class TestPages:
    def test_a_page_of_known_size(self, source: Path) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines(
            page_sizes={1: (4512, 6174)}
        )
        assert type(page) is DiscreteGraphicalTimeline
        assert page.id == "dgt_1"
        assert page.name == "1"
        assert page.unit is TimeUnit.pixels
        assert page.number_type is NumberType.int
        assert page.length.value == 4512
        assert page.is_locked
        assert page.meta == {
            "image": PAGE_URL,
            "box": {"x": 0, "y": 0, "width": 4512, "height": 6174},
        }

    def test_a_page_of_unknown_size(self, source: Path) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines()
        assert page.length is None
        assert not page.is_locked
        assert page.meta == {"image": PAGE_URL}

    def test_a_size_for_a_page_the_file_does_not_hold(self, source: Path) -> None:
        loader = DmosLoader.from_file(source)
        with pytest.raises(KeyError) as raised:
            loader.create_timelines(page_sizes={2: (1, 1)})
        assert raised.value.args[0] == (
            "page_sizes names page 2, which the DMOS file does not hold; its pages "
            "are 1."
        )

    def test_a_system_past_the_image_width(self, source: Path) -> None:
        loader = DmosLoader.from_file(source)
        with pytest.raises(ValueError) as raised:
            loader.create_timelines(page_sizes={1: (4101, 6174)})
        assert str(raised.value) == (
            "DMOS page 1, system 2 ends at x 4112, past the right edge of its image, "
            "4101 pixels wide."
        )


class TestSystems:
    @pytest.mark.parametrize("page_sizes", [None, {1: (4512, 6174)}])
    def test_systems_are_children_at_their_least_x(
        self, source: Path, page_sizes: dict | None
    ) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines(page_sizes=page_sizes)
        assert page.list_children() == ["dgt_2", "dgt_3"]
        first, second = (page.get_child(uid) for uid in ("dgt_2", "dgt_3"))
        assert [first.name, second.name] == ["1", "2"]
        assert [page.get_child_offset(uid).value for uid in ("dgt_2", "dgt_3")] == [
            1221,
            971,
        ]
        assert [first.length.value, second.length.value] == [2880, 3141]
        assert first.meta == {
            "box": _canonical(
                (1221, 1589), (4097, 1589), (4101, 2266), (1221, 2266)
            ).to_dict()
        }
        assert second.meta == {
            "box": _canonical(
                (971, 2493), (4104, 2493), (4112, 3192), (971, 3192)
            ).to_dict()
        }
        assert first.meta["box"]["bottom_right"] == {"x": 4101, "y": 2266}

    def test_repeated_system_ids_name_two_children(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _document(
                [
                    _system(1, SYSTEM_1, [_measure(MEASURE_1)]),
                    _system(1, SYSTEM_2, []),
                ]
            ),
        )
        (page,) = DmosLoader.from_file(source).create_timelines()
        children = [page.get_child(uid) for uid in page.list_children()]
        assert [(child.id, child.name) for child in children] == [
            ("dgt_2", "1"),
            ("dgt_3", "1"),
        ]
        assert [len(_boxes(child)) for child in children] == [1, 0]


class TestMeasures:
    def test_measures_are_boxes_of_their_system(self, source: Path) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines()
        system = page.get_child("dgt_2")
        m1 = _canonical((1221, 1589), (2151, 1589), (2161, 2266), (1221, 2266))
        m2 = _canonical((2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266))
        assert _boxes(system) == [
            Box(id="box_1", start=_px(0), duration=_extent(940), quadrilateral=m1),
            Box(id="box_2", start=_px(930), duration=_extent(624), quadrilateral=m2),
        ]
        assert m1.bounding_box.to_dict() == {
            "x": 1221,
            "y": 1589,
            "width": 940,
            "height": 677,
        }

    def test_headers_and_voices_leave_no_trace(self, source: Path) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines()
        assert len(page.get_events(include_children=False)) == 0
        assert len(page.get_child("dgt_3").get_events()) == 0
        system = page.get_child("dgt_2").get_events().table
        assert system.column("event_type").to_pylist() == ["Box", "Box"]
        assert system.column_names == [
            "id",
            "name",
            "temporal_type",
            "event_type",
            "start",
            "end",
            "duration",
            "bounding_box",
            "quadrilateral",
        ]

    @pytest.mark.parametrize("page_sizes", [None, {1: (4512, 6174)}])
    def test_every_box_keeps_the_image_space_invariant(
        self, source: Path, page_sizes: dict | None
    ) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines(page_sizes=page_sizes)
        _assert_image_space_invariant(page)

    def test_a_crossed_measure_is_refused_where_it_stands(self, tmp_path: Path) -> None:
        # Measure 1 written in DMOS's system order: read in the measure order,
        # its outline is crossed.
        crossed = [[1221, 1589], [1221, 2266], [2151, 1589], [2161, 2266]]
        source = _write(
            tmp_path, _document([_system(1, SYSTEM_1, [_measure(crossed)])])
        )
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == (
            "DMOS page 1, system 1, measure 1: The corners top_left (1221, 1589), "
            "top_right (1221, 2266), bottom_right (2161, 2266), bottom_left "
            "(2151, 1589) do not run clockwise around a convex outline: the turn at "
            "top_left is counter-clockwise."
        )

    def test_a_measure_without_a_region(self, tmp_path: Path) -> None:
        source = _write(tmp_path, _document([_system(1, SYSTEM_1, [{"voices": []}])]))
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == "DMOS page 1, system 1, measure 1 has no 'region'."


class TestLifecycle:
    def test_nothing_loaded(self) -> None:
        with pytest.raises(RuntimeError) as raised:
            DmosLoader().create_timelines()
        assert str(raised.value) == "No DMOS file loaded. Call load() first."

    def test_one_source(self, source: Path, tmp_path: Path) -> None:
        loader = DmosLoader.from_file(source)
        other = tmp_path / "other.json"
        with pytest.raises(ValueError) as raised:
            loader.load(other)
        assert str(raised.value) == (
            f"DmosLoader reads one source and already holds {source}; use a new "
            f"loader for {other}, or call clear() first."
        )

    def test_repr(self, source: Path) -> None:
        assert repr(DmosLoader()) == "DmosLoader(not loaded)"
        assert repr(DmosLoader.from_file(source)) == (
            "DmosLoader(pages=1, systems=2, measures=2)"
        )
