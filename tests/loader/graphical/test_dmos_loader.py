"""Tests for DmosLoader.

The fixtures are excerpts of a DMOS JSON file (the DMOS output for the
Gallica score ``ark:/12148/bpt6k11620688``), written by the tests. The
validation logic is documented in ``tests/loader/graphical/README.md``
("``DmosLoader``"). All expected values are exact and derived from the source
file by hand.
"""

from __future__ import annotations

import json
from fractions import Fraction
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
from timetoalign.loader.graphical import DmosLoader, HorizontalLinePath, PiecewisePath
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

SCORE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688"
PAGE_URL = f"{SCORE_URL}/f2"
SYSTEM_1 = [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]]
SYSTEM_2 = [[971, 2493], [971, 3192], [4104, 2493], [4112, 3192]]
MEASURE_1 = [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]]
MEASURE_2 = [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]]
#: Measure 1 of page 1's system 2, and of page 2's system 1.
MEASURE_3 = [[971, 2493], [1935, 2493], [971, 3192], [1945, 3192]]
MEASURE_4 = [[519, 896], [1520, 896], [519, 1647], [1522, 1647]]
#: A system region that is a vertical line, as DMOS output holds some.
NO_WIDTH = [[22, 929], [22, 1185], [22, 929], [22, 1185]]

#: The regions of all systems of the scan, per page, as the file writes them.
SCAN_SYSTEMS = [
    [
        SYSTEM_1,
        SYSTEM_2,
        [[983, 3414], [983, 4135], [4115, 3414], [4125, 4135]],
        [[994, 4367], [994, 5091], [4129, 4367], [4142, 5091]],
    ],
    [
        [[519, 896], [519, 1647], [3661, 896], [3655, 1647]],
        [[522, 1832], [522, 2524], [3659, 1832], [3652, 2524]],
        [[522, 2713], [522, 3369], [3656, 2713], [3659, 3369]],
        [[523, 3563], [523, 4260], [3656, 3563], [3658, 4260]],
        [[522, 4444], [522, 5129], [3657, 4444], [3661, 5129]],
    ],
    [
        [[913, 920], [913, 1680], [4053, 920], [4060, 1680]],
        [[918, 1966], [918, 2813], [4054, 1966], [4053, 2813]],
        [[913, 3090], [913, 3861], [4051, 3090], [4051, 3861]],
        [[907, 4144], [907, 4925], [4048, 4144], [4047, 4925]],
    ],
    [
        [[508, 963], [508, 1702], [3650, 963], [3649, 1702]],
        [[504, 1989], [504, 2767], [3643, 1989], [3638, 2767]],
        [[494, 3030], [494, 3854], [3633, 3030], [3634, 3854]],
        [[489, 4144], [489, 4939], [3609, 4144], [3628, 4939]],
    ],
]
#: The measures the scan excerpt keeps, by (page, system) position from 0.
SCAN_MEASURES = {
    (0, 0): [MEASURE_1, MEASURE_2],
    (0, 1): [MEASURE_3],
    (1, 0): [MEASURE_4],
}


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


def _page(number: int, systems: list[dict[str, Any]]) -> dict[str, Any]:
    """A DMOS page; page ``n`` of the scan is image ``f{n + 1}``."""
    return {
        "page_url": f"{SCORE_URL}/f{number + 1}",
        "no_page": number,
        "header_systems": {},
        "systems": systems,
    }


def _pages_document(pages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "C055_0",
        "date": "2025-06-10-11h",
        "score_image_url": SCORE_URL,
        "pages": pages,
    }


def _document(systems: list[dict[str, Any]]) -> dict[str, Any]:
    """A DMOS file of page 1 alone."""
    return _pages_document([_page(1, systems)])


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


@pytest.fixture
def scan(tmp_path: Path) -> Path:
    """All four pages with their 17 system regions, and four measures."""
    return _write(
        tmp_path,
        _pages_document(
            [
                _page(
                    p + 1,
                    [
                        _system(
                            k + 1,
                            region,
                            [_measure(m) for m in SCAN_MEASURES.get((p, k), [])],
                        )
                        for k, region in enumerate(systems)
                    ],
                )
                for p, systems in enumerate(SCAN_SYSTEMS)
            ]
        ),
    )


def _segments(line: Timeline) -> list[Timeline]:
    return [line.get_child(uid) for uid in line.list_children()]


def _offsets(line: Timeline) -> list[int]:
    return [line.get_child_offset(uid).value for uid in line.list_children()]


def _box_ids(timeline: Timeline) -> list[str]:
    return timeline.get_events(include_children=False).table.column("id").to_pylist()


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


class TestPaths:
    def test_a_page_s_path_is_its_systems(self, scan: Path) -> None:
        paths = DmosLoader.from_file(scan).create_paths()
        assert list(paths) == [1, 2, 3, 4]
        assert paths[1] == PiecewisePath(
            (
                HorizontalLinePath(
                    x0=1221, x1=4101, y=Fraction(3855, 2), tolerance=Fraction(677, 2)
                ),
                HorizontalLinePath(
                    x0=971, x1=4112, y=Fraction(5685, 2), tolerance=Fraction(699, 2)
                ),
                HorizontalLinePath(
                    x0=983, x1=4125, y=Fraction(7549, 2), tolerance=Fraction(721, 2)
                ),
                HorizontalLinePath(x0=994, x1=4142, y=4729, tolerance=362),
            )
        )
        last = paths[1].pieces[3]
        assert type(last.y) is int and type(last.tolerance) is int
        assert [len(path.pieces) for path in paths.values()] == [4, 5, 4, 4]
        assert [path.length for path in paths.values()] == [12311, 15690, 12562, 12560]
        assert all(type(path.length) is int for path in paths.values())
        assert paths[1].offsets == (0, 2880, 6021, 9163)

    def test_a_page_without_systems_has_no_path(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _pages_document(
                [
                    _page(1, [_system(1, SYSTEM_1, [])]),
                    _page(2, []),
                    _page(3, [_system(1, SYSTEM_2, [])]),
                ]
            ),
        )
        paths = DmosLoader.from_file(source).create_paths()
        assert list(paths) == [1, 3]
        assert [path.length for path in paths.values()] == [2880, 3141]


class TestVolume:
    def test_the_volume_is_the_pages_in_a_row(self, scan: Path) -> None:
        volume = DmosLoader.from_file(scan).create_timeline()
        assert type(volume) is SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]
        assert volume.id == "dgt_1"
        assert volume.unit is TimeUnit.pixels
        assert volume.number_type is NumberType.int
        assert volume.length.value == 53123
        assert volume.is_locked
        assert volume.meta == {"image": SCORE_URL}
        assert len(volume.get_events(include_children=False)) == 0

        pages = _segments(volume)
        assert [page.id for page in pages] == ["dgt_2", "dgt_7", "dgt_13", "dgt_18"]
        assert [page.name for page in pages] == ["1", "2", "3", "4"]
        assert _offsets(volume) == [0, 12311, 28001, 40563]
        assert [page.length.value for page in pages] == [12311, 15690, 12562, 12560]
        for n, page in enumerate(pages, start=2):
            assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
            assert page.is_locked
            assert page.meta == {"image": f"{SCORE_URL}/f{n}"}

        first, second = pages[0], pages[1]
        assert first.list_children() == ["dgt_3", "dgt_4", "dgt_5", "dgt_6"]
        assert _offsets(first) == [0, 2880, 6021, 9163]
        assert [a.length.value for a in _segments(first)] == [2880, 3141, 3142, 3148]
        assert second.list_children() == [
            "dgt_8",
            "dgt_9",
            "dgt_10",
            "dgt_11",
            "dgt_12",
        ]
        assert _offsets(second) == [0, 3142, 6279, 9416, 12551]
        assert [a.length.value for a in _segments(second)] == [
            3142,
            3137,
            3137,
            3135,
            3139,
        ]
        assert [a.length.value for a in _segments(pages[2])] == [3147, 3136, 3138, 3141]
        assert [a.length.value for a in _segments(pages[3])] == [3142, 3139, 3140, 3139]
        for page in pages:
            for accolade in _segments(page):
                assert type(accolade) is DiscreteGraphicalTimeline
                assert not accolade.is_locked
        _assert_image_space_invariant(volume)

    def test_volume_identity(self, scan: Path) -> None:
        volume = DmosLoader.from_file(scan).create_timeline(
            uid="scan", name="Aimons-nous"
        )
        assert (volume.id, volume.name) == ("scan", "Aimons-nous")
        assert volume.list_children() == ["dgt_2", "dgt_7", "dgt_13", "dgt_18"]

    def test_a_file_without_a_score_image(self, source: Path) -> None:
        document = json.loads(source.read_text(encoding="utf-8"))
        del document["score_image_url"]
        source.write_text(json.dumps(document), encoding="utf-8")
        assert DmosLoader.from_file(source).create_timeline().meta == {}

    def test_pages_without_systems_are_left_out(self, tmp_path: Path) -> None:
        source = _write(
            tmp_path,
            _pages_document(
                [
                    _page(1, [_system(1, SYSTEM_1, [])]),
                    _page(2, []),
                    _page(3, [_system(1, SYSTEM_2, [])]),
                ]
            ),
        )
        loader = DmosLoader.from_file(source)
        volume = loader.create_timeline()
        assert [page.name for page in _segments(volume)] == ["1", "3"]
        assert _offsets(volume) == [0, 2880]
        assert volume.length.value == 6021
        assert [page.name for page in loader.create_timelines()] == ["1", "3"]

    def test_a_file_without_any_system(self, tmp_path: Path) -> None:
        loader = DmosLoader.from_file(_write(tmp_path, _document([])))
        assert loader.create_paths() == {}
        assert loader.create_timelines() == []
        with pytest.raises(ValueError) as raised:
            loader.create_timeline()
        assert str(raised.value) == "A volume needs at least one page."


class TestPages:
    def test_one_root_per_page(self, scan: Path) -> None:
        pages = DmosLoader.from_file(scan).create_timelines()
        assert [page.id for page in pages] == ["dgt_1", "dgt_6", "dgt_12", "dgt_17"]
        assert [page.name for page in pages] == ["1", "2", "3", "4"]
        assert [page.length.value for page in pages] == [12311, 15690, 12562, 12560]
        for n, page in enumerate(pages, start=2):
            assert type(page) is SegmentLine[DiscreteGraphicalTimeline]
            assert page.unit is TimeUnit.pixels
            assert page.number_type is NumberType.int
            assert page.is_locked
            assert page.meta == {"image": f"{SCORE_URL}/f{n}"}
            assert len(page.get_events(include_children=False)) == 0
        assert pages[0].list_children() == ["dgt_2", "dgt_3", "dgt_4", "dgt_5"]
        assert _offsets(pages[0]) == [0, 2880, 6021, 9163]

    def test_no_argument_takes_an_image_size(self, source: Path) -> None:
        loader = DmosLoader.from_file(source)
        with pytest.raises(TypeError):
            loader.create_timelines(page_sizes={1: (4512, 6174)})  # type: ignore[call-arg]


class TestSystems:
    def test_systems_are_accolades_one_after_the_other(self, source: Path) -> None:
        (page,) = DmosLoader.from_file(source).create_timelines()
        assert page.length.value == 6021
        assert page.list_children() == ["dgt_2", "dgt_3"]
        first, second = _segments(page)
        assert [first.name, second.name] == ["1", "2"]
        assert _offsets(page) == [0, 2880]
        assert [first.length.value, second.length.value] == [2880, 3141]
        assert first.meta == {
            "box": _canonical(
                (1221, 1589), (4097, 1589), (4101, 2266), (1221, 2266)
            ).to_dict(),
            "image": PAGE_URL,
        }
        assert second.meta == {
            "box": _canonical(
                (971, 2493), (4104, 2493), (4112, 3192), (971, 3192)
            ).to_dict(),
            "image": PAGE_URL,
        }
        assert first.meta["box"]["bottom_right"] == {"x": 4101, "y": 2266}

    def test_repeated_system_ids_name_two_accolades(self, tmp_path: Path) -> None:
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
        accolades = _segments(page)
        assert [(accolade.id, accolade.name) for accolade in accolades] == [
            ("dgt_2", "1"),
            ("dgt_3", "1"),
        ]
        assert [len(_boxes(accolade)) for accolade in accolades] == [1, 0]


class TestSystemWithoutWidth:
    """Known issue: a system whose region is a vertical line is no piece of a path."""

    MESSAGE = "DMOS page 1: accolade 2: x1 (22) must be greater than x0 (22)"

    @pytest.fixture
    def loader(self, tmp_path: Path) -> DmosLoader:
        source = _write(
            tmp_path,
            _document(
                [
                    _system(1, SYSTEM_1, [_measure(MEASURE_1)]),
                    _system(1, NO_WIDTH, []),
                ]
            ),
        )
        return DmosLoader.from_file(source)

    def test_the_file_loads(self, loader: DmosLoader) -> None:
        assert repr(loader) == "DmosLoader(pages=1, systems=2, measures=1)"

    @pytest.mark.parametrize(
        "method", ["create_paths", "create_timelines", "create_timeline"]
    )
    def test_no_path_and_no_timeline_is_made(
        self, loader: DmosLoader, method: str
    ) -> None:
        with pytest.raises(ValueError) as raised:
            getattr(loader, method)()
        assert str(raised.value) == self.MESSAGE


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

    def test_boxes_start_anew_on_every_accolade(self, scan: Path) -> None:
        pages = DmosLoader.from_file(scan).create_timelines()
        second = pages[0].get_child("dgt_3")
        assert _boxes(second) == [
            Box(
                id="box_1",
                start=_px(0),
                duration=_extent(974),
                quadrilateral=_canonical(
                    (971, 2493), (1935, 2493), (1945, 3192), (971, 3192)
                ),
            )
        ]
        assert _boxes(pages[1].get_child("dgt_7")) == [
            Box(
                id="box_1",
                start=_px(0),
                duration=_extent(1003),
                quadrilateral=_canonical(
                    (519, 896), (1520, 896), (1522, 1647), (519, 1647)
                ),
            )
        ]
        for page in pages:
            _assert_image_space_invariant(page)

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

    def test_every_box_keeps_the_image_space_invariant(self, source: Path) -> None:
        loader = DmosLoader.from_file(source)
        (page,) = loader.create_timelines()
        _assert_image_space_invariant(page)
        _assert_image_space_invariant(loader.create_timeline())

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

    def test_a_corner_that_is_no_whole_pixel(self, tmp_path: Path) -> None:
        fractional = [[1221.5, 1589], [2151, 1589], [1221.5, 2266], [2161, 2266]]
        source = _write(
            tmp_path, _document([_system(1, SYSTEM_1, [_measure(fractional)])])
        )
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == (
            "DMOS page 1, system 1, measure 1: A shape's x is 1221.5, which pixels "
            "as int cannot hold exactly: geometry is never rounded."
        )

    def test_a_measure_without_a_region(self, tmp_path: Path) -> None:
        source = _write(tmp_path, _document([_system(1, SYSTEM_1, [{"voices": []}])]))
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == "DMOS page 1, system 1, measure 1 has no 'region'."


class TestIdentity:
    def test_a_template_numbers_the_measures_through_the_score(
        self, scan: Path
    ) -> None:
        volume = DmosLoader.from_file(scan).create_timeline(uid="scan", box_ids="m{n}")
        first, second = _segments(volume)[:2]
        assert first.id == "dgt_2"
        assert _box_ids(first.get_child("dgt_3")) == ["m1", "m2"]
        assert _box_ids(first.get_child("dgt_4")) == ["m3"]
        assert _box_ids(second.get_child("dgt_8")) == ["m4"]
        m2 = _boxes(first.get_child("dgt_3"))[1]
        assert (m2.id, m2.start.value, m2.duration.value) == ("m2", 930, 624)

    def test_a_template_of_positions(self, scan: Path) -> None:
        pages = DmosLoader.from_file(scan).create_timelines(
            page_ids="page{page}",
            accolade_ids="p{page}_s{accolade}",
            box_ids="p{page}_s{accolade}_b{box}",
        )
        assert [page.id for page in pages] == ["page1", "page2", "page3", "page4"]
        assert [page.list_children() for page in pages] == [
            ["p1_s1", "p1_s2", "p1_s3", "p1_s4"],
            ["p2_s1", "p2_s2", "p2_s3", "p2_s4", "p2_s5"],
            ["p3_s1", "p3_s2", "p3_s3", "p3_s4"],
            ["p4_s1", "p4_s2", "p4_s3", "p4_s4"],
        ]
        assert _box_ids(pages[0].get_child("p1_s1")) == ["p1_s1_b1", "p1_s1_b2"]
        assert _box_ids(pages[0].get_child("p1_s2")) == ["p1_s2_b1"]
        assert _box_ids(pages[1].get_child("p2_s1")) == ["p2_s1_b1"]

    def test_sequences_name_in_document_order(self, scan: Path) -> None:
        volume = DmosLoader.from_file(scan).create_timeline(
            page_ids=["f2", "f3", "f4", "f5"],
            accolade_ids=[f"system_{n}" for n in range(1, 18)],
            box_ids=["a", "b", "c", "d"],
        )
        assert volume.id == "dgt_1"
        assert volume.list_children() == ["f2", "f3", "f4", "f5"]
        first, second = _segments(volume)[:2]
        assert first.list_children() == [
            "system_1",
            "system_2",
            "system_3",
            "system_4",
        ]
        assert second.list_children()[0] == "system_5"
        assert _segments(volume)[3].list_children()[-1] == "system_17"
        assert _box_ids(first.get_child("system_1")) == ["a", "b"]
        assert _box_ids(first.get_child("system_2")) == ["c"]
        assert _box_ids(second.get_child("system_5")) == ["d"]

    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (
                {"box_ids": ["a", "b", "c"]},
                "box_ids names 3 Boxes; this call creates 4.",
            ),
            (
                {"box_ids": "m{system}"},
                "box_ids holds the template field {system}; a template is filled "
                "from page, accolade, box, n, of which a Box has page, accolade, "
                "box, n.",
            ),
            (
                {"accolade_ids": "s\x01{n}"},
                "Invalid id 's\\x011': an id must be non-empty and contain no "
                "control character",
            ),
            (
                {"uid": "scan", "page_ids": ["scan", "f3", "f4", "f5"]},
                "The id 'scan' is given to the volume and to page '1': every "
                "timeline of a call has its own id.",
            ),
        ],
    )
    def test_a_refused_argument_builds_nothing(
        self, scan: Path, arguments: dict[str, Any], message: str
    ) -> None:
        loader = DmosLoader.from_file(scan)
        with pytest.raises(ValueError) as raised:
            loader.create_timeline(**arguments)
        assert str(raised.value) == message
        assert loader.create_timeline().id == "dgt_1"


class TestMeasuresLeavingTheirSystem:
    """Known issue: a measure outside its system's x extent refuses the file."""

    def test_a_measure_left_of_its_system(self, tmp_path: Path) -> None:
        left = [[1200, 1589], [2151, 1589], [1200, 2266], [2161, 2266]]
        source = _write(tmp_path, _document([_system(1, SYSTEM_1, [_measure(left)])]))
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == (
            "DMOS page 1, system 1, measure 1 spans x 1200 to 2161, which leaves "
            "its system's region, spanning x 1221 to 4101."
        )

    def test_a_measure_right_of_its_system(self, tmp_path: Path) -> None:
        right = [[3500, 1589], [4110, 1589], [3500, 2266], [4120, 2266]]
        source = _write(
            tmp_path,
            _document([_system(1, SYSTEM_1, [_measure(MEASURE_1), _measure(right)])]),
        )
        with pytest.raises(ValueError) as raised:
            DmosLoader().load(source)
        assert str(raised.value) == (
            "DMOS page 1, system 1, measure 2 spans x 3500 to 4120, which leaves "
            "its system's region, spanning x 1221 to 4101."
        )

    def test_a_measure_spanning_its_system_exactly(self, tmp_path: Path) -> None:
        whole = [[1221, 1589], [4097, 1589], [1221, 2266], [4101, 2266]]
        source = _write(tmp_path, _document([_system(1, SYSTEM_1, [_measure(whole)])]))
        (page,) = DmosLoader.from_file(source).create_timelines()
        assert page.length.value == 2880
        system = page.get_child("dgt_2")
        assert _boxes(system) == [
            Box(
                id="box_1",
                start=_px(0),
                duration=_extent(2880),
                quadrilateral=_canonical(
                    (1221, 1589), (4097, 1589), (4101, 2266), (1221, 2266)
                ),
            )
        ]
        _assert_image_space_invariant(page)


class TestLifecycle:
    def test_nothing_loaded(self) -> None:
        for method in ("create_paths", "create_timelines", "create_timeline"):
            with pytest.raises(RuntimeError) as raised:
                getattr(DmosLoader(), method)()
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
