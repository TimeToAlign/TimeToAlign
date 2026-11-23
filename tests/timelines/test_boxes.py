"""Boxes added to a timeline: the shape in the image, the interval on the timeline.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes.py`` - Boxes added to a timeline").
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import BoundingBox, Box, Quadrilateral
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    DiscreteGraphicalTimeline,
    SegmentLine,
)

#: Thoresen's ``rect_h2`` on its 875 x 270 image.
RECT_H2 = BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels")
#: The second crop of the 2010 example: x0 = 7 to x1 = 874.
CROP_2 = {"x": 7, "y": 0, "width": 867, "height": 270}
#: DMOS's corner orders: systems TL, BL, TR, BR; measures TL, TR, BL, BR.
SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")
MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")
#: *Aimons-nous*, page 1, as DMOS writes its regions.
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
#: The fraction box of a user-units timeline.
THIRDS = BoundingBox.from_xywh(
    Fraction(1, 3),
    0,
    Fraction(2, 3),
    1,
    unit="user_units",
    number_type="fraction",
)


def _exact(message: str) -> str:
    return f"^{re.escape(message)}$"


def _cell(value: Any) -> dict[str, Any]:
    """The memory cell of an exact value: its float, numerator, denominator."""
    value = Fraction(value)
    return {
        "value": float(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _seg_2(*, box: bool = True) -> DiscreteGraphicalTimeline:
    return DiscreteGraphicalTimeline(
        length=867, uid="seg_2", meta={"box": CROP_2} if box else None
    )


def _page_and_system() -> tuple[DiscreteGraphicalTimeline, DiscreteGraphicalTimeline]:
    page = DiscreteGraphicalTimeline(
        length=4512,
        uid="page",
        meta={"box": {"x": 0, "y": 0, "width": 4512, "height": 6174}},
    )
    system = DiscreteGraphicalTimeline(
        length=2880, uid="sys", meta={"box": SYSTEM_1.to_dict()}
    )
    page.add_child(system, 1221)
    return page, system


def _uu(meta: dict[str, Any] | None = None) -> ContinuousGraphicalTimeline:
    return ContinuousGraphicalTimeline(
        length=2, unit="user_units", number_type="fraction", uid="uu", meta=meta
    )


class TestAccepted:
    def test_rect_h2_on_its_crop(self) -> None:
        seg_2 = _seg_2()

        ids = seg_2.add_events(Box.from_shape(RECT_H2, origin=7, id="rect_h2"))

        assert ids == ["rect_h2"]
        (row,) = seg_2.events.table.to_pylist()
        assert row["event_type"] == "Box"
        assert row["start"] == _cell(378)
        assert row["duration"] == _cell(139)
        assert row["end"] == _cell(517)
        assert row["bounding_box"]["x"] == _cell(385)
        assert row["quadrilateral"] is None

    def test_rect_h2_on_the_segment_line(self) -> None:
        dgt_2 = SegmentLine[DiscreteGraphicalTimeline](
            length=0, unit="pixels", uid="dgt_2"
        )
        dgt_2.append_segment(DiscreteGraphicalTimeline(length=866, uid="seg_1"))
        seg_2 = _seg_2()
        dgt_2.append_segment(seg_2)
        seg_2.add_events([Box.from_shape(RECT_H2, origin=7, id="rect_h2")])

        stamp = dgt_2.get_timestamp_for("rect_h2")

        assert (stamp.start.axis.value, stamp.end.axis.value) == (1244, 1383)

    def test_quadrilaterals_on_a_child(self) -> None:
        _, system = _page_and_system()

        ids = system.add_events(
            [
                Box.from_shape(MEASURE_1, origin=1221, id="m1"),
                Box.from_shape(MEASURE_2, origin=1221, id="m2"),
            ]
        )

        assert ids == ["m1", "m2"]
        rows = system.events.table.to_pylist()
        assert [(r["start"], r["duration"]) for r in rows] == [
            (_cell(0), _cell(940)),
            (_cell(930), _cell(624)),
        ]
        assert [r["quadrilateral"]["bottom_right"] for r in rows] == [
            {"x": _cell(2161), "y": _cell(2266)},
            {"x": _cell(2775), "y": _cell(2266)},
        ]
        assert [r["bounding_box"] for r in rows] == [None, None]

    def test_exact_on_a_fraction_axis(self) -> None:
        uu = _uu()

        uu.add_events(Box.from_shape(THIRDS, id="f"))

        (row,) = uu.events.table.to_pylist()
        assert row["start"] == _cell(Fraction(1, 3))
        assert row["duration"] == _cell(Fraction(2, 3))
        assert row["bounding_box"]["width"] == _cell(Fraction(2, 3))

    def test_a_fraction_origin(self) -> None:
        uu = _uu({"box": {"x": [1, 3], "y": [0, 1], "width": [5, 3], "height": [1, 1]}})

        assert uu.add_events(Box.from_shape(THIRDS, origin=Fraction(1, 3), id="f")) == [
            "f"
        ]
        assert uu.events.table.to_pylist()[0]["start"] == _cell(0)

    def test_other_events_pass_unchanged(self) -> None:
        seg_2 = _seg_2()

        ids = seg_2.add_events(
            [
                {"event_type": "Note", "start": 0, "duration": 10},
                Box.from_shape(RECT_H2, origin=7, id="rect_h2"),
            ]
        )

        assert ids == ["note_1", "rect_h2"]

    def test_no_box_reads_no_meta_box(self) -> None:
        timeline = DiscreteGraphicalTimeline(
            length=867, uid="seg_2", meta={"box": {"x": "left"}}
        )

        assert timeline.add_events([{"event_type": "Note", "start": 0}]) == ["note_1"]


class TestRefused:
    def test_a_start_off_its_shape(self) -> None:
        seg_2 = _seg_2(box=False)

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'rect_h2' starts at 378 on timeline 'seg_2', but its shape "
                "starts at x 385 and the timeline stands at x 0 of its image "
                '(meta["box"]): its start is 385.'
            ),
        ):
            seg_2.add_events([Box.from_shape(RECT_H2, origin=7, id="rect_h2")])
        assert seg_2.n_events == 0
        assert seg_2.length.value == 867

    def test_a_box_off_the_axis(self) -> None:
        seg_2 = _seg_2()
        css = BoundingBox.from_xywh(385.0, 46.0, 139.0, 20.0, unit="css_pixels")

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'rect_h2' is in css_pixels as float, timeline 'seg_2' in "
                "pixels as int: a Box is added on its timeline's axis."
            ),
        ):
            seg_2.add_events([Box.from_shape(css, origin=7.0, id="rect_h2")])
        assert seg_2.n_events == 0

    def test_a_box_without_id_is_named_a_box(self) -> None:
        css = BoundingBox.from_xywh(385.0, 46.0, 139.0, 20.0, unit="css_pixels")
        with pytest.raises(
            ValueError,
            match=_exact(
                "A Box is in css_pixels as float, timeline 'seg_2' in pixels as "
                "int: a Box is added on its timeline's axis."
            ),
        ):
            _seg_2().add_events(Box.from_shape(css, origin=7.0))
        with pytest.raises(
            ValueError,
            match=_exact(
                "A Box starts at 378 on timeline 'seg_2', but its shape starts at "
                "x 385 and the timeline stands at x 0 of its image "
                '(meta["box"]): its start is 385.'
            ),
        ):
            _seg_2(box=False).add_events(Box.from_shape(RECT_H2, origin=7))

    def test_atomic_in_a_batch(self) -> None:
        _, system = _page_and_system()

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'm2' starts at 2151 on timeline 'sys', but its shape starts "
                "at x 2151 and the timeline stands at x 1221 of its image "
                '(meta["box"]): its start is 930.'
            ),
        ):
            system.add_events(
                [
                    Box.from_shape(MEASURE_1, origin=1221, id="m1"),
                    Box.from_shape(MEASURE_2, id="m2"),
                ]
            )
        assert system.n_events == 0

    def test_a_fraction_start_off_its_shape(self) -> None:
        uu = _uu({"box": {"x": [1, 3], "y": [0, 1], "width": [5, 3], "height": [1, 1]}})

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'f' starts at 1/3 on timeline 'uu', but its shape starts at "
                "x 1/3 and the timeline stands at x 1/3 of its image "
                '(meta["box"]): its start is 0.'
            ),
        ):
            uu.add_events(Box.from_shape(THIRDS, id="f"))
        assert uu.n_events == 0
