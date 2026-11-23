"""Boxes moved onto a timeline cut from theirs: slices, children, segments.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes_rebased.py`` - Boxes moved onto a timeline cut from
theirs").
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from timetoalign.core import BoundingBox, Box, Quadrilateral
from timetoalign.timelines import DiscreteGraphicalTimeline, SegmentLine, Timeline

B1 = BoundingBox.from_xywh(1221, 1589, 940, 677, unit="pixels")
B2 = BoundingBox.from_xywh(2161, 1589, 800, 677, unit="pixels")
PAGE_BOX = {"x": 0, "y": 0, "width": 4564, "height": 6174}
IMAGE = "page_1.jpg"

#: DMOS's corner orders: systems TL, BL, TR, BR; measures TL, TR, BL, BR.
SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")
MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
SYSTEM_1 = Quadrilateral.from_points(
    [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]],
    order=SYSTEM_ORDER,
    unit="pixels",
)
MEASURE_1 = Quadrilateral.from_points(
    [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
    order=MEASURE_ORDER,
    unit="pixels",
)
MEASURE_2 = Quadrilateral.from_points(
    [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]],
    order=MEASURE_ORDER,
    unit="pixels",
)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _page(*, meta: bool = True) -> DiscreteGraphicalTimeline:
    page = DiscreteGraphicalTimeline(
        length=4564,
        uid="page",
        meta={"box": PAGE_BOX, "image": IMAGE} if meta else None,
    )
    page.add_events([Box.from_shape(B1, id="b1"), Box.from_shape(B2, id="b2")])
    return page


def _area(x: int, width: int) -> dict[str, Any]:
    return {"box": {"x": x, "y": 0, "width": width, "height": 6174}, "image": IMAGE}


def _boxes(timeline: Timeline) -> dict[str, tuple[int, int, Any]]:
    """Each Box of *timeline*'s own rows: start, duration, shape cell."""
    return {
        row["id"]: (
            row["start"]["numerator"],
            row["duration"]["numerator"],
            row["bounding_box"] or row["quadrilateral"],
        )
        for row in timeline.get_events(include_children=False, event_type="Box")
    }


def _shape_cell(source: Timeline, box_id: str) -> dict[str, Any]:
    """The shape cell of the Box *box_id* as *source* holds it."""
    return _boxes(source)[box_id][2]


def _cell(value: int) -> dict[str, Any]:
    return {"value": float(value), "numerator": value, "denominator": 1}


def _round_trips(timeline: Timeline) -> None:
    written = timeline.to_dict(events=True)
    assert type(timeline).from_dict(written).to_dict(events=True) == written


class TestTheAreaMoves:
    def test_slice(self) -> None:
        page = _page()
        sliced = page.get_slice(1000, 3000)
        assert sliced.length.value == 2000
        assert sliced.meta == _area(1000, 2000)
        assert _boxes(sliced) == {
            "b1": (221, 940, _shape_cell(page, "b1")),
            "b2": (1161, 800, _shape_cell(page, "b2")),
        }
        assert _shape_cell(sliced, "b1") == {
            "x": _cell(1221),
            "y": _cell(1589),
            "width": _cell(940),
            "height": _cell(677),
        }
        _round_trips(sliced)

    def test_child_from_region(self) -> None:
        page = _page()
        page.create_region(1000, 3000, id="r")
        child = page.create_child_from_region("r")
        assert page.get_child_offset(child.id).value == 1000
        assert child.length.value == 2000
        assert child.meta == _area(1000, 2000)
        assert _boxes(child) == {
            "b1": (221, 940, _shape_cell(page, "b1")),
            "b2": (1161, 800, _shape_cell(page, "b2")),
        }
        _round_trips(page)

    @pytest.mark.parametrize("make", ["create_segment_line", "from_segmentation"])
    def test_segment_line(self, make: str) -> None:
        page = _page()
        cuts = [0, 1000, 3000, 4564]
        line = (
            page.create_segment_line(cuts)
            if make == "create_segment_line"
            else SegmentLine.from_segmentation(page, cuts)
        )
        segments = [segment for _, _, segment in line.iter_segments()]
        assert [offset.value for _, offset, _ in line.iter_segments()] == [
            0,
            1000,
            3000,
        ]
        assert [segment.length.value for segment in segments] == [1000, 2000, 1564]
        assert [segment.meta for segment in segments] == [
            _area(0, 1000),
            _area(1000, 2000),
            _area(3000, 1564),
        ]
        assert [_boxes(segment) for segment in segments] == [
            {},
            {
                "b1": (221, 940, _shape_cell(page, "b1")),
                "b2": (1161, 800, _shape_cell(page, "b2")),
            },
            {},
        ]
        _round_trips(line)

    def test_a_quadrilateral_source_through_its_child(self) -> None:
        page = DiscreteGraphicalTimeline(
            length=4564, uid="page", meta={"box": PAGE_BOX, "image": IMAGE}
        )
        system = DiscreteGraphicalTimeline(
            length=2880, uid="sys", meta={"box": SYSTEM_1.to_dict()}
        )
        system.add_events(
            [
                Box.from_shape(MEASURE_1, origin=1221, id="m1"),
                Box.from_shape(MEASURE_2, origin=1221, id="m2"),
            ]
        )
        page.add_child(system, offset=1221)
        sliced = page.get_slice(1000, 3000)
        assert sliced.meta == _area(1000, 2000)
        [child_id] = sliced.list_children()
        cut = sliced.get_child(child_id)
        assert sliced.get_child_offset(child_id).value == 221
        assert cut.length.value == 1779
        assert cut.meta == {"box": {"x": 1221, "y": 1589, "width": 1779, "height": 677}}
        assert _boxes(cut) == {
            "m1": (0, 940, _boxes(system)["m1"][2]),
            "m2": (930, 624, _boxes(system)["m2"][2]),
        }
        _round_trips(sliced)


class TestCrossing:
    """The known issue: a slice never clips a Box; the cuts keep it whole."""

    def test_a_slice_refuses_a_box_across_its_start(self) -> None:
        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'b1' runs from 1221 to 2161 on timeline 'page' and crosses "
                "the slice's boundary at 1500: a slice clips the intervals that "
                "cross its boundaries, and a clipped Box would contradict its "
                "shape. Slice with truncate_events=False to leave it out."
            ),
        ):
            _page().get_slice(1500, 3000)

    def test_a_slice_refuses_a_box_across_its_end(self) -> None:
        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'b2' runs from 2161 to 2961 on timeline 'page' and crosses "
                "the slice's boundary at 2500: a slice clips the intervals that "
                "cross its boundaries, and a clipped Box would contradict its "
                "shape. Slice with truncate_events=False to leave it out."
            ),
        ):
            _page().get_slice(1000, 2500)

    def test_a_slice_without_truncating_leaves_it_out(self) -> None:
        page = _page()
        sliced = page.get_slice(1500, 3000, truncate_events=False)
        assert sliced.meta == _area(1500, 1500)
        assert _boxes(sliced) == {"b2": (661, 800, _shape_cell(page, "b2"))}
        _round_trips(sliced)

    def test_a_child_takes_a_box_whole_and_grows(self) -> None:
        page = _page()
        page.create_region(1500, 2500, id="r")
        child = page.create_child_from_region("r")
        assert child.length.value == 1461
        assert child.meta == _area(1500, 1461)
        assert _boxes(child) == {"b2": (661, 800, _shape_cell(page, "b2"))}
        _round_trips(page)

    def test_segments_take_boxes_whole_and_grow(self) -> None:
        page = _page()
        line = page.create_segment_line([0, 1500, 2500, 4564])
        segments = [segment for _, _, segment in line.iter_segments()]
        assert [offset.value for _, offset, _ in line.iter_segments()] == [
            0,
            2161,
            3622,
        ]
        assert [segment.length.value for segment in segments] == [2161, 1461, 2064]
        assert [segment.meta for segment in segments] == [
            _area(0, 2161),
            _area(1500, 1461),
            _area(2500, 2064),
        ]
        assert [_boxes(segment) for segment in segments] == [
            {"b1": (1221, 940, _shape_cell(page, "b1"))},
            {"b2": (661, 800, _shape_cell(page, "b2"))},
            {},
        ]
        _round_trips(line)


class TestASourceWithoutMetaBox:
    MESSAGE = _exact(
        "Box 'b1' cannot move onto a timeline cut at 1000 from timeline 'page': "
        "'page' states no meta[\"box\"], so the cut cannot state where it stands "
        "in the image."
    )

    def test_a_cut_at_0_keeps_its_boxes(self) -> None:
        page = _page(meta=False)
        sliced = page.get_slice(0, 3000)
        assert sliced.meta == {}
        assert _boxes(sliced) == {
            "b1": (1221, 940, _shape_cell(page, "b1")),
            "b2": (2161, 800, _shape_cell(page, "b2")),
        }
        _round_trips(sliced)

    def test_a_slice_elsewhere_is_refused(self) -> None:
        with pytest.raises(ValueError, match=self.MESSAGE):
            _page(meta=False).get_slice(1000, 3000)

    def test_a_child_elsewhere_is_refused(self) -> None:
        page = _page(meta=False)
        page.create_region(1000, 3000, id="r")
        with pytest.raises(ValueError, match=self.MESSAGE):
            page.create_child_from_region("r")
        assert page.list_children() == []

    def test_a_segment_elsewhere_is_refused(self) -> None:
        with pytest.raises(ValueError, match=self.MESSAGE):
            _page(meta=False).create_segment_line([0, 1000, 3000, 4564])
