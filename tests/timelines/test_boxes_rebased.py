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
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    DiscreteGraphicalTimeline,
    SegmentLine,
    Timeline,
)

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


def _system() -> DiscreteGraphicalTimeline:
    """DMOS's system 1, holding the measures ``m1`` and ``m2``."""
    system = DiscreteGraphicalTimeline(
        length=2880, uid="sys", meta={"box": SYSTEM_1.to_dict()}
    )
    system.add_events(
        [
            Box.from_shape(MEASURE_1, origin=1221, id="m1"),
            Box.from_shape(MEASURE_2, origin=1221, id="m2"),
        ]
    )
    return system


def _dmos_page(meta: dict[str, Any] | None = None) -> DiscreteGraphicalTimeline:
    """A page holding no Box of its own; its child ``sys`` holds the measures."""
    page = DiscreteGraphicalTimeline(
        length=4564,
        uid="page",
        meta={"box": PAGE_BOX, "image": IMAGE} if meta is None else meta,
    )
    page.add_child(_system(), offset=1221)
    return page


def _volume() -> SegmentLine:
    """Two pages in a segment line stating no area of its own."""
    volume = SegmentLine(unit="pixels", uid="vol")
    for page_id, box_id, x, width in (("p1", "a1", 200, 300), ("p2", "b2", 400, 400)):
        page = DiscreteGraphicalTimeline(
            length=1100,
            uid=page_id,
            meta={
                "box": {"x": 0, "y": 0, "width": 1100, "height": 1500},
                "image": f"{page_id}.jpg",
            },
        )
        page.add_events(
            [
                Box.from_shape(
                    BoundingBox.from_xywh(x, 100, width, 200, unit="pixels"),
                    id=box_id,
                )
            ]
        )
        volume.append_segment(page)
    return volume


def _only_child(parent: Timeline) -> Timeline:
    [child_id] = parent.list_children()
    return parent.get_child(child_id)


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


def _copies(child: Timeline, source: Timeline) -> dict[str, tuple[int, int, Any]]:
    """*child*'s Boxes, each keyed by the ID of the Box of *source* it copies.

    A child cut from a region holds copies, new events of the tree: each is
    minted an ID of its own (event IDs are unique among a root and its descendants) and is
    matched to the Box it copies by its shape, which a copy keeps.
    """
    held = _boxes(source)
    by_shape = {repr(cells[2]): box_id for box_id, cells in held.items()}
    copies = _boxes(child)
    assert not set(copies) & set(held)
    return {by_shape[repr(cells[2])]: cells for cells in copies.values()}


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
        assert _copies(child, page) == {
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
        page = _dmos_page()
        system = page.get_child("sys")
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

    def test_a_slice_refuses_a_childs_box_across_its_boundary(self) -> None:
        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'm1' runs from 0 to 940 on timeline 'sys' and crosses "
                "the slice's boundary at 279: a slice clips the intervals that "
                "cross its boundaries, and a clipped Box would contradict its "
                "shape. Slice with truncate_events=False to leave it out."
            ),
        ):
            _dmos_page().get_slice(1500, 3500)

    def test_without_truncating_the_childs_slice_leaves_it_out(self) -> None:
        page = _dmos_page()
        sliced = page.get_slice(1500, 3500, truncate_events=False)
        assert sliced.length.value == 2000
        assert sliced.meta == _area(1500, 2000)
        cut = _only_child(sliced)
        assert sliced.get_child_offset(cut.id).value == 0
        assert cut.length.value == 2000
        assert cut.meta == {"box": {"x": 1500, "y": 1589, "width": 2000, "height": 677}}
        system = page.get_child("sys")
        assert _boxes(cut) == {"m2": (651, 624, _boxes(system)["m2"][2])}
        _round_trips(sliced)

    def test_a_slice_refuses_a_segments_box_across_its_boundary(self) -> None:
        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'b2' runs from 400 to 800 on timeline 'p2' and crosses "
                "the slice's boundary at 600: a slice clips the intervals that "
                "cross its boundaries, and a clipped Box would contradict its "
                "shape. Slice with truncate_events=False to leave it out."
            ),
        ):
            _volume().get_slice(500, 1700)

    def test_without_truncating_the_segments_slice_leaves_it_out(self) -> None:
        sliced = _volume().get_slice(500, 1700, truncate_events=False)
        assert sliced.length.value == 1200
        cuts = [
            (offset.value, segment) for _, offset, segment in sliced.iter_segments()
        ]
        assert [offset for offset, _ in cuts] == [0, 600]
        assert [segment.length.value for _, segment in cuts] == [600, 600]
        assert [segment.meta for _, segment in cuts] == [
            {
                "box": {"x": 500, "y": 0, "width": 600, "height": 1500},
                "image": "p1.jpg",
            },
            {"box": {"x": 0, "y": 0, "width": 600, "height": 1500}, "image": "p2.jpg"},
        ]
        assert [_boxes(segment) for _, segment in cuts] == [{}, {}]

    def test_a_child_takes_a_box_whole_and_grows(self) -> None:
        page = _page()
        page.create_region(1500, 2500, id="r")
        child = page.create_child_from_region("r")
        assert child.length.value == 1461
        assert child.meta == _area(1500, 1461)
        assert _copies(child, page) == {"b2": (661, 800, _shape_cell(page, "b2"))}
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


class TestKnownIssueASourceWithoutArea:
    """A timeline holding Boxes without ``meta["box"]`` states no y extent."""

    MESSAGE = _exact(
        "Box 'b1' cannot move onto a timeline cut at 1000 from timeline 'page': "
        'the timeline states no area (meta["box"]), so the cut\'s area cannot '
        "be stated."
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

    def test_a_child_of_a_volume_is_refused(self) -> None:
        volume = _volume()
        volume.create_region(150, 1000, id="r")
        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'a1' cannot move onto a timeline cut at 150 from timeline "
                "'vol': the timeline states no area (meta[\"box\"]), so the "
                "cut's area cannot be stated."
            ),
        ):
            volume.create_child_from_region("r")
        assert volume.list_children() == ["p1", "p2"]


class TestTheWholeExtentKeepsTheArea:
    def test_a_slice_of_the_page(self) -> None:
        page = _dmos_page()
        sliced = page.get_slice(0, 4564)
        assert sliced.meta == page.meta == {"box": PAGE_BOX, "image": IMAGE}
        cut = _only_child(sliced)
        assert sliced.get_child_offset(cut.id).value == 1221
        assert cut.length.value == 2880
        assert cut.meta == {"box": SYSTEM_1.to_dict()}
        assert cut.meta["box"] == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 4097, "y": 1589},
            "bottom_right": {"x": 4101, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        _round_trips(sliced)

    def test_a_slice_and_a_child_of_the_system(self) -> None:
        system = _system()
        assert system.get_slice(0, 2880).meta == {"box": SYSTEM_1.to_dict()}
        system.create_region(0, 2880, id="all")
        child = system.create_child_from_region("all")
        assert child.meta == {"box": SYSTEM_1.to_dict()}
        assert _copies(child, system) == _boxes(system)


class TestKnownIssueAProperCutOfAQuadrilateral:
    """The cut of a quadrilateral area is the cut of its bounding box."""

    def test_a_slice(self) -> None:
        sliced = _system().get_slice(0, 1600)
        assert sliced.meta == {
            "box": {"x": 1221, "y": 1589, "width": 1600, "height": 677}
        }
        _round_trips(sliced)

    def test_a_child_from_a_region(self) -> None:
        system = _system()
        system.create_region(930, 2880, id="r")
        child = system.create_child_from_region("r")
        assert child.meta == {
            "box": {"x": 2151, "y": 1589, "width": 1950, "height": 677}
        }
        assert _copies(child, system) == {"m2": (0, 624, _boxes(system)["m2"][2])}


#: ``meta["box"]`` values a timeline without Boxes may hold: free-form data.
FREE_FORM_BOXES = {
    "string": "cardboard",
    "other_dict": {"left": 1, "top": 2},
    "valid_shape": {"x": [0, 1], "y": [0, 1], "width": [20, 1], "height": [1, 1]},
}


def _notes(timeline: Timeline) -> list[tuple[str, int, int]]:
    """Each event of *timeline*'s own rows: id, start, end (whole quarters)."""
    return [
        (row["id"], row["start"]["numerator"], row["end"]["numerator"])
        for row in timeline.get_events(include_children=False)
    ]


@pytest.mark.parametrize("box", FREE_FORM_BOXES.values(), ids=FREE_FORM_BOXES.keys())
class TestFreeFormMetaWithoutBoxes:
    """A timeline holding no Box is cut as before Boxes existed."""

    @staticmethod
    def _timeline(box: Any) -> ContinuousLogicalTimeline:
        timeline = ContinuousLogicalTimeline(
            length=20, uid="t", meta={"box": box, "image": "scan.png"}
        )
        timeline.add_events(
            [
                {"event_type": "Note", "id": "n1", "start": 0, "end": 5},
                {"event_type": "Note", "id": "n2", "start": 12, "end": 15},
                {"event_type": "Note", "id": "n3", "start": 8, "end": 14},
            ]
        )
        return timeline

    def test_slice(self, box: Any) -> None:
        timeline = self._timeline(box)
        sliced = timeline.get_slice(10, 20)
        assert sliced.length.value == 10
        assert sliced.meta == {}
        assert sorted(_notes(sliced)) == [("n2", 2, 5), ("n3", 0, 4)]
        kept = timeline.get_slice(10, 20, truncate_events=False)
        assert kept.meta == {}
        assert _notes(kept) == [("n2", 2, 5)]
        assert timeline.meta == {"box": box, "image": "scan.png"}

    def test_child_from_region(self, box: Any) -> None:
        timeline = self._timeline(box)
        timeline.create_region(10, 20, id="r")
        child = timeline.create_child_from_region("r")
        assert child.length.value == 10
        assert child.meta == {}
        # The copy of n2, a new event of the tree with an ID of its own.
        [(copy_id, *span)] = _notes(child)
        assert span == [2, 5]
        assert copy_id.startswith("note_")
        assert timeline.meta == {"box": box, "image": "scan.png"}

    @pytest.mark.parametrize("make", ["create_segment_line", "from_segmentation"])
    def test_segments(self, box: Any, make: str) -> None:
        timeline = self._timeline(box)
        line = (
            timeline.create_segment_line([0, 10, 20])
            if make == "create_segment_line"
            else SegmentLine.from_segmentation(timeline, [0, 10, 20])
        )
        segments = [segment for _, _, segment in line.iter_segments()]
        assert [segment.length.value for segment in segments] == [14, 10]
        assert [segment.meta for segment in segments] == [{}, {}]
        assert [sorted(_notes(segment)) for segment in segments] == [
            [("n1", 0, 5), ("n3", 8, 14)],
            [("n2", 2, 5)],
        ]
        assert timeline.meta == {"box": box, "image": "scan.png"}


class TestAMetaBoxThatIsNoShape:
    """A timeline holding Boxes reads ``meta["box"]`` as its area."""

    PREFIX = (
        "Timeline 'page' holds Boxes, so its meta[\"box\"] must be its area of "
        "their image, a shape in pixels as int: "
    )
    REFUSALS = {
        "string": (
            "cardboard",
            "it holds 'cardboard', not the dict form of a BoundingBox or a "
            "Quadrilateral.",
        ),
        "other_dict": (
            {"left": 1, "top": 2},
            "A shape's dict holds ['left', 'top']: a BoundingBox's keys are x, y, "
            "width and height, a Quadrilateral's top_left, top_right, "
            "bottom_right and bottom_left.",
        ),
    }

    @pytest.mark.parametrize("kind", REFUSALS)
    @pytest.mark.parametrize("operation", ["slice", "child", "segments"])
    def test_refused(self, kind: str, operation: str) -> None:
        box, reason = self.REFUSALS[kind]
        page = _dmos_page({"box": box})
        with pytest.raises(ValueError, match=_exact(self.PREFIX + reason)):
            if operation == "slice":
                page.get_slice(0, 3000)
            elif operation == "child":
                page.create_region(0, 3000, id="r")
                page.create_child_from_region("r")
            else:
                page.create_segment_line([0, 3000, 4564])
