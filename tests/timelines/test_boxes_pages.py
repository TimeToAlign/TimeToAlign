"""Cuts of pages and volumes: segment lines of accolades holding Boxes.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes_pages.py`` - Cuts of pages and volumes").
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from timetoalign.core import BoundingBox, Box
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

A1_AREA = {"x": 100, "y": 200, "width": 1000, "height": 300}
A2_AREA = {"x": 120, "y": 700, "width": 900, "height": 300}
#: Each accolade's Boxes: id suffix and shape (x, y, width, height).
A1_BOXES = (("m1", (150, 200, 300, 300)), ("m2", (600, 200, 300, 300)))
A2_BOXES = (("m3", (220, 700, 300, 300)), ("m4", (700, 700, 300, 300)))
OWN = (200, 50, 400, 100)
IMAGE = "page_1.jpg"

NO_AREA = (
    "Box '{box}' cannot move onto a timeline cut at {start} from timeline "
    "'{timeline}': the timeline states no area (meta[\"box\"]), so the cut's "
    "area cannot be stated."
)
CLIPPED = (
    "Box 'm1' runs from 50 to 350 on timeline 'a1' and crosses the slice's "
    "boundary at 200: a slice clips the intervals that cross its boundaries, "
    "and a clipped Box would contradict its shape. Slice with "
    "truncate_events=False to leave it out."
)
HAS_CHILDREN = (
    "Cannot segment a timeline that already has children. This would create "
    "conflicting nesting hierarchies."
)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _no_area(box: str, start: int, timeline: str) -> str:
    return _exact(NO_AREA.format(box=box, start=start, timeline=timeline))


def _shape(numbers: tuple[int, int, int, int]) -> BoundingBox:
    return BoundingBox.from_xywh(*numbers, unit="pixels")


def _cells(numbers: tuple[int, int, int, int]) -> dict[str, Any]:
    """The ``bounding_box`` cell of the shape (x, y, width, height)."""
    return {
        key: {"value": float(value), "numerator": value, "denominator": 1}
        for key, value in zip(("x", "y", "width", "height"), numbers)
    }


def _accolade(
    uid: str,
    area: dict[str, int],
    boxes: tuple[tuple[str, tuple[int, int, int, int]], ...],
    *,
    prefix: str = "",
    image: str = IMAGE,
) -> DiscreteGraphicalTimeline:
    accolade = DiscreteGraphicalTimeline(
        length=area["width"],
        uid=f"{prefix}{uid}",
        meta={"box": dict(area), "image": image},
    )
    accolade.add_events(
        [
            Box.from_shape(_shape(numbers), origin=area["x"], id=f"{prefix}{box_id}")
            for box_id, numbers in boxes
        ]
    )
    return accolade


def _page(uid: str = "page", *, prefix: str = "", image: str = IMAGE) -> SegmentLine:
    """A page: the accolades ``a1`` and ``a2`` end to end, stating no area."""
    page = SegmentLine[DiscreteGraphicalTimeline](
        unit="pixels", uid=uid, meta={"image": image}
    )
    page.append_segment(_accolade("a1", A1_AREA, A1_BOXES, prefix=prefix, image=image))
    page.append_segment(_accolade("a2", A2_AREA, A2_BOXES, prefix=prefix, image=image))
    return page


def _volume() -> SegmentLine:
    """A volume: the pages ``p1`` and ``p2`` end to end, stating no area."""
    volume = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
        unit="pixels", uid="vol", meta={"image": "volume"}
    )
    volume.append_segment(_page("p1", prefix="p1_", image="page_1.jpg"))
    volume.append_segment(_page("p2", prefix="p2_", image="page_2.jpg"))
    return volume


def _sheet(*, own: bool = False) -> DiscreteGraphicalTimeline:
    """A plain timeline stating no area, holding ``a1`` at offset 1500."""
    sheet = DiscreteGraphicalTimeline(length=3000, uid="sheet", meta={"image": IMAGE})
    sheet.add_child(_accolade("a1", A1_AREA, A1_BOXES), offset=1500)
    if own:
        sheet.add_events([Box.from_shape(_shape(OWN), id="own")])
    return sheet


def _boxless_sheet() -> DiscreteGraphicalTimeline:
    """A tree holding no Box, whose child states an area all the same."""
    strip = DiscreteGraphicalTimeline(
        length=1000, uid="strip", meta={"box": dict(A1_AREA), "image": IMAGE}
    )
    strip.add_events([{"event_type": "Note", "id": "e2", "start": 50, "end": 350}])
    sheet = DiscreteGraphicalTimeline(length=3000, uid="sheet", meta={"image": IMAGE})
    sheet.add_events([{"event_type": "Note", "id": "e1", "start": 200, "end": 600}])
    sheet.add_child(strip, offset=1500)
    return sheet


def _segments(line: Timeline) -> list[tuple[int, Timeline]]:
    return [(offset.value, segment) for _, offset, segment in line.iter_segments()]


def _children(parent: Timeline) -> list[tuple[int, Timeline]]:
    return [
        (parent.get_child_offset(child_id).value, parent.get_child(child_id))
        for child_id in parent.list_children()
    ]


def _boxes(timeline: Timeline) -> dict[str, tuple[int, int, Any]]:
    """Each Box among *timeline*'s own rows: start, duration, shape cell."""
    return {
        row["id"]: (
            row["start"]["numerator"],
            row["duration"]["numerator"],
            row["bounding_box"],
        )
        for row in timeline.get_events(include_children=False, event_type="Box")
    }


def _starts(timeline: Timeline) -> dict[str, tuple[int, int]]:
    """Each of *timeline*'s own rows: start and end."""
    return {
        row["id"]: (row["start"]["numerator"], row["end"]["numerator"])
        for row in timeline.get_events(include_children=False)
    }


def _own_rows(timeline: Timeline) -> list[dict[str, Any]]:
    return list(timeline.get_events(include_children=False))


def _area(x: int, y: int, width: int, image: str = IMAGE) -> dict[str, Any]:
    return {"box": {"x": x, "y": y, "width": width, "height": 300}, "image": image}


def _round_trips(timeline: Timeline) -> None:
    written = timeline.to_dict(events=True)
    assert type(timeline).from_dict(written).to_dict(events=True) == written


class TestRowsOfDescendants:
    """What tells a descendant's row from a timeline's own."""

    @staticmethod
    def _read(timeline: Timeline, start: int, end: int) -> list[tuple[Any, ...]]:
        return [
            (row["id"], row["start"]["numerator"], row["source_timeline"])
            for row in timeline.get_events(min_coord=start, max_coord=end)
        ]

    def test_a_page_reads_its_accolades_boxes(self) -> None:
        page = _page()
        assert page.length.value == 1900
        assert self._read(page, 0, 1900) == [
            ("m1", 50, "a1"),
            ("m2", 500, "a1"),
            ("m3", 1100, "a2"),
            ("m4", 1580, "a2"),
        ]
        assert _own_rows(page) == []

    def test_a_volume_names_the_holding_accolade(self) -> None:
        volume = _volume()
        assert volume.length.value == 3800
        assert self._read(volume, 1000, 2000) == [
            ("p1_m3", 1100, "p1_a2"),
            ("p1_m4", 1580, "p1_a2"),
            ("p2_m1", 1950, "p2_a1"),
        ]

    def test_an_own_box_has_no_source_timeline(self) -> None:
        assert self._read(_sheet(own=True), 0, 1000) == [("own", 200, None)]


class TestASliceCutsEachAccolade:
    def test_a_slice_of_the_page(self) -> None:
        sliced = _page().get_slice(500, 1500)
        assert sliced.length.value == 1000
        assert sliced.meta == {"image": IMAGE}
        assert _own_rows(sliced) == []
        (first_offset, first), (second_offset, second) = _segments(sliced)
        assert (first_offset, second_offset) == (0, 500)
        assert (first.length.value, second.length.value) == (500, 500)
        assert first.meta == _area(600, 200, 500)
        assert second.meta == _area(120, 700, 500)
        assert _boxes(first) == {"m2": (0, 300, _cells((600, 200, 300, 300)))}
        assert _boxes(second) == {"m3": (100, 300, _cells((220, 700, 300, 300)))}
        _round_trips(sliced)

    def test_the_whole_page(self) -> None:
        page = _page()
        sliced = page.get_slice(0, 1900)
        assert sliced.length.value == 1900
        assert sliced.meta == {"image": IMAGE}
        (first_offset, first), (second_offset, second) = _segments(sliced)
        assert (first_offset, second_offset) == (0, 1000)
        assert first.meta == {"box": A1_AREA, "image": IMAGE}
        assert second.meta == {"box": A2_AREA, "image": IMAGE}
        assert (
            _boxes(first)
            == _boxes(page.get_child("a1"))
            == {
                "m1": (50, 300, _cells((150, 200, 300, 300))),
                "m2": (500, 300, _cells((600, 200, 300, 300))),
            }
        )
        assert (
            _boxes(second)
            == _boxes(page.get_child("a2"))
            == {
                "m3": (100, 300, _cells((220, 700, 300, 300))),
                "m4": (580, 300, _cells((700, 700, 300, 300))),
            }
        )
        _round_trips(sliced)

    def test_a_box_crossing_the_boundary_is_refused(self) -> None:
        with pytest.raises(ValueError, match=_exact(CLIPPED)):
            _page().get_slice(200, 1900)

    def test_without_truncating_the_crossing_box_is_left_out(self) -> None:
        sliced = _page().get_slice(200, 1900, truncate_events=False)
        assert sliced.length.value == 1700
        assert sliced.meta == {"image": IMAGE}
        (first_offset, first), (second_offset, second) = _segments(sliced)
        assert (first_offset, second_offset) == (0, 800)
        assert (first.length.value, second.length.value) == (800, 900)
        assert first.meta == _area(300, 200, 800)
        assert second.meta == {"box": A2_AREA, "image": IMAGE}
        assert _boxes(first) == {"m2": (300, 300, _cells((600, 200, 300, 300)))}
        assert _boxes(second) == {
            "m3": (100, 300, _cells((220, 700, 300, 300))),
            "m4": (580, 300, _cells((700, 700, 300, 300))),
        }
        _round_trips(sliced)

    def test_a_slice_of_the_volume(self) -> None:
        sliced = _volume().get_slice(500, 2400)
        assert sliced.length.value == 1900
        assert sliced.meta == {"image": "volume"}
        assert _own_rows(sliced) == []
        (first_offset, first), (second_offset, second) = _segments(sliced)
        assert (first_offset, second_offset) == (0, 1400)
        assert (first.length.value, second.length.value) == (1400, 500)
        assert first.meta == {"image": "page_1.jpg"}
        assert second.meta == {"image": "page_2.jpg"}
        assert _own_rows(first) == _own_rows(second) == []

        (cut_offset, cut), (whole_offset, whole) = _segments(first)
        assert (cut_offset, whole_offset) == (0, 500)
        assert (cut.length.value, whole.length.value) == (500, 900)
        assert cut.meta == _area(600, 200, 500, "page_1.jpg")
        assert whole.meta == {"box": A2_AREA, "image": "page_1.jpg"}
        assert _boxes(cut) == {"p1_m2": (0, 300, _cells((600, 200, 300, 300)))}
        assert _boxes(whole) == {
            "p1_m3": (100, 300, _cells((220, 700, 300, 300))),
            "p1_m4": (580, 300, _cells((700, 700, 300, 300))),
        }

        [(offset, head)] = _segments(second)
        assert offset == 0
        assert head.length.value == 500
        assert head.meta == _area(100, 200, 500, "page_2.jpg")
        assert _boxes(head) == {"p2_m1": (50, 300, _cells((150, 200, 300, 300)))}
        _round_trips(sliced)


class TestAChildFromARegionIsRefused:
    @pytest.mark.parametrize(
        ("region", "box"), [((0, 1000), "m1"), ((500, 1500), "m2")]
    )
    def test_of_a_page(self, region: tuple[int, int], box: str) -> None:
        page = _page()
        page.create_region(*region, id="r")
        with pytest.raises(ValueError, match=_no_area(box, region[0], "page")):
            page.create_child_from_region("r")
        assert page.list_children() == ["a1", "a2"]
        assert page.length.value == 1900

    @pytest.mark.parametrize(
        ("region", "box"), [((0, 1900), "p1_m1"), ((1900, 3800), "p2_m1")]
    )
    def test_of_a_volume(self, region: tuple[int, int], box: str) -> None:
        volume = _volume()
        volume.create_region(*region, id="r")
        with pytest.raises(ValueError, match=_no_area(box, region[0], "vol")):
            volume.create_child_from_region("r")
        assert volume.list_children() == ["p1", "p2"]
        assert volume.length.value == 3800


class TestSegmentsTakingBoxesAlongAreRefused:
    def test_of_a_page(self) -> None:
        with pytest.raises(ValueError, match=_no_area("m1", 0, "page")):
            _page().create_segment_line([0, 1000, 1900])

    def test_of_a_volume(self) -> None:
        with pytest.raises(ValueError, match=_no_area("p1_m1", 0, "vol")):
            _volume().create_segment_line([0, 1900, 3800])

    def test_without_events_nothing_moves(self) -> None:
        line = _page().create_segment_line([0, 1000, 1900], copy_events=False)
        segments = _segments(line)
        assert [offset for offset, _ in segments] == [0, 1000]
        assert [segment.length.value for _, segment in segments] == [1000, 900]
        assert [segment.meta for _, segment in segments] == [
            {"image": IMAGE},
            {"image": IMAGE},
        ]
        assert [list(segment.get_events()) for _, segment in segments] == [[], []]

    def test_a_segmentation_refuses_a_timeline_with_children(self) -> None:
        with pytest.raises(ValueError, match=_exact(HAS_CHILDREN)):
            SegmentLine.from_segmentation(_page(), [0, 1000, 1900])


class TestAParentThatIsNoSegmentLine:
    def test_a_child_taking_no_box_is_made(self) -> None:
        sheet = _sheet()
        sheet.create_region(0, 1000, id="r")
        child = sheet.create_child_from_region("r")
        assert child.length.value == 1000
        assert child.meta == {"image": IMAGE}
        assert list(child.get_events()) == []
        assert sheet.list_children() == ["a1", child.id]
        assert sheet.get_child_offset(child.id).value == 0

    def test_a_child_taking_the_accolades_box_is_refused(self) -> None:
        sheet = _sheet()
        sheet.create_region(0, 2000, id="r")
        with pytest.raises(ValueError, match=_no_area("m1", 0, "sheet")):
            sheet.create_child_from_region("r")
        assert sheet.list_children() == ["a1"]


class TestOwnBoxesBesideAnAccolade:
    """A timeline's own Boxes stand at x 0 and move onto a cut at 0 only."""

    def test_a_child_at_0_takes_the_own_box(self) -> None:
        sheet = _sheet(own=True)
        sheet.create_region(0, 1000, id="r")
        child = sheet.create_child_from_region("r")
        assert child.length.value == 1000
        assert child.meta == {"image": IMAGE}
        assert _boxes(child) == _boxes(sheet) == {"own": (200, 400, _cells(OWN))}
        _round_trips(child)

    def test_a_child_elsewhere_is_refused(self) -> None:
        sheet = _sheet(own=True)
        sheet.create_region(100, 1000, id="r")
        with pytest.raises(ValueError, match=_no_area("own", 100, "sheet")):
            sheet.create_child_from_region("r")
        assert sheet.list_children() == ["a1"]

    def test_the_accolades_box_is_refused_at_0(self) -> None:
        sheet = _sheet(own=True)
        sheet.create_region(0, 2000, id="r")
        with pytest.raises(ValueError, match=_no_area("m1", 0, "sheet")):
            sheet.create_child_from_region("r")
        assert sheet.list_children() == ["a1"]

    def test_a_slice_at_0(self) -> None:
        sliced = _sheet(own=True).get_slice(0, 3000)
        assert sliced.meta == {"image": IMAGE}
        assert _boxes(sliced) == {"own": (200, 400, _cells(OWN))}
        [(offset, accolade)] = _children(sliced)
        assert offset == 1500
        assert accolade.length.value == 1000
        assert accolade.meta == {"box": A1_AREA, "image": IMAGE}
        assert _boxes(accolade) == {
            "m1": (50, 300, _cells((150, 200, 300, 300))),
            "m2": (500, 300, _cells((600, 200, 300, 300))),
        }
        _round_trips(sliced)


class TestNoBoxAnywhere:
    """A tree holding no Box is cut as before, descendants' rows included."""

    def test_a_child_takes_the_descendants_row(self) -> None:
        sheet = _boxless_sheet()
        sheet.create_region(100, 2000, id="r")
        child = sheet.create_child_from_region("r")
        assert child.length.value == 1900
        assert child.meta == {}
        assert _starts(child) == {"e1": (100, 500), "e2": (1450, 1750)}
        assert sheet.get_child_offset(child.id).value == 100

    def test_a_slice(self) -> None:
        sliced = _boxless_sheet().get_slice(100, 3000)
        assert sliced.length.value == 2900
        assert sliced.meta == {}
        assert _starts(sliced) == {"e1": (100, 500)}
        [(offset, strip)] = _children(sliced)
        assert offset == 1400
        assert strip.length.value == 1000
        assert strip.meta == {}
        assert _starts(strip) == {"e2": (50, 350)}

    def test_segments(self) -> None:
        line = _boxless_sheet().create_segment_line([0, 1000, 3000])
        segments = _segments(line)
        assert [offset for offset, _ in segments] == [0, 1000]
        assert [segment.length.value for _, segment in segments] == [1000, 2000]
        assert [segment.meta for _, segment in segments] == [{}, {}]
        assert [_starts(segment) for _, segment in segments] == [
            {"e1": (200, 600)},
            {"e2": (550, 850)},
        ]
