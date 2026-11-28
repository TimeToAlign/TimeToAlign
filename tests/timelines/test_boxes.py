"""Boxes added to a timeline: the shape in the image, the interval on the timeline.

Validation strategy and exact values: ``tests/timelines/README.md`` (the
section "``test_boxes.py`` - Boxes added to a timeline").
"""

from __future__ import annotations

import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import BoundingBox, Box, Coordinate, Duration, Quadrilateral
from timetoalign.core.ids import IdMinter
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    DiscreteGraphicalTimeline,
    SegmentLine,
    Timeline,
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
            ],
            seed=7,
        )

        assert ids == [IdMinter(7).mint("note"), "rect_h2"]

    def test_no_box_reads_no_meta_box(self) -> None:
        timeline = DiscreteGraphicalTimeline(
            length=867, uid="seg_2", meta={"box": {"x": "left"}}
        )

        assert timeline.add_events([{"event_type": "Note", "start": 0}], seed=7) == [
            IdMinter(7).mint("note")
        ]


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
                "The event's bounding_box is in css_pixels as float, the events are "
                "in pixels as int; a shape is never converted onto the events' axis."
            ),
        ):
            seg_2.add_events([Box.from_shape(css, origin=7.0, id="rect_h2")])
        assert seg_2.n_events == 0

    def test_a_box_without_id_is_named_a_box(self) -> None:
        with pytest.raises(
            ValueError,
            match=_exact(
                "A Box starts at 378 on timeline 'seg_2', but its shape starts at "
                "x 385 and the timeline stands at x 0 of its image "
                '(meta["box"]): its start is 385.'
            ),
        ):
            _seg_2(box=False).add_events(Box.from_shape(RECT_H2, origin=7))
        with pytest.raises(
            ValueError,
            match=_exact(
                "A Box: its duration is 10 and its shape spans 139 in x (385 to 524); "
                "a Box's duration is its shape's x extent."
            ),
        ):
            _seg_2().add_events(
                [
                    {
                        "event_type": "Box",
                        "start": 378,
                        "duration": 10,
                        "bounding_box": RECT_H2,
                    }
                ]
            )

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


#: ``rect_h2`` as a quadrilateral: the same corners, in the canonical order.
RECT_H2_QUADRILATERAL = Quadrilateral.from_points(
    [(385, 46), (524, 46), (524, 66), (385, 66)], unit="pixels"
)
#: The Box's own refusal of ``rect_h2`` with a duration of 10, on any path.
DURATION_10 = (
    "Box 'a': its duration is 10 and its shape spans 139 in x (385 to 524); a "
    "Box's duration is its shape's x extent."
)
ONE_SHAPE = "A Box holds exactly one shape: give bounding_box or quadrilateral."


def _refused_on_seg_2(rows: list[dict[str, Any]], message: str) -> None:
    """*rows* given to ``add_events`` on ``seg_2`` are refused, nothing added."""
    seg_2 = _seg_2()
    with pytest.raises(ValueError, match=_exact(message)):
        seg_2.add_events(rows)
    assert seg_2.n_events == 0
    assert seg_2.length.value == 867


def _box_row(**given: Any) -> dict[str, Any]:
    return {"event_type": "Box", "id": "a", "start": 378, **given}


def _written_seg_2() -> dict[str, Any]:
    seg_2 = _seg_2()
    seg_2.add_events(Box.from_shape(RECT_H2, origin=7, id="rect_h2"))
    return seg_2.to_dict(events=True)


def _box_written(data: dict[str, Any]) -> dict[str, Any]:
    """The event row of the one Box of the one root *data* writes."""
    (row,) = [r for r in data["timelines"][0]["events"] if r["event_type"] == "Box"]
    return row


class TestOneCheckEveryPath:
    def test_the_scalar_refuses_its_duration(self) -> None:
        with pytest.raises(ValueError, match=re.escape(DURATION_10)):
            Box(
                id="a",
                start=Coordinate(378, "pixels"),
                duration=Duration(10, "pixels"),
                bounding_box=RECT_H2,
            )

    def test_a_dict_row_with_another_duration(self) -> None:
        _refused_on_seg_2([_box_row(duration=10, bounding_box=RECT_H2)], DURATION_10)

    def test_a_dict_row_with_an_end_in_place_of_a_duration(self) -> None:
        _refused_on_seg_2(
            [_box_row(end=400, bounding_box=RECT_H2)],
            DURATION_10.replace("is 10", "is 22"),
        )

    def test_a_dict_row_without_a_duration(self) -> None:
        _refused_on_seg_2(
            [_box_row(bounding_box=RECT_H2)],
            "Box 'a': it has no duration; a Box's duration is its shape's x extent.",
        )

    def test_a_dict_row_without_a_shape(self) -> None:
        _refused_on_seg_2([_box_row(duration=139)], ONE_SHAPE)

    def test_a_dict_row_with_two_shapes(self) -> None:
        _refused_on_seg_2(
            [
                _box_row(
                    duration=139,
                    bounding_box=RECT_H2,
                    quadrilateral=RECT_H2_QUADRILATERAL,
                )
            ],
            ONE_SHAPE,
        )

    def test_atomic_beside_another_event(self) -> None:
        _refused_on_seg_2(
            [
                {"event_type": "Note", "start": 0, "duration": 10},
                _box_row(duration=10, bounding_box=RECT_H2),
            ],
            DURATION_10,
        )

    def test_from_events(self) -> None:
        with pytest.raises(ValueError, match=_exact(DURATION_10)):
            DiscreteGraphicalTimeline.from_events(
                [_box_row(duration=10, bounding_box=RECT_H2)],
                unit="pixels",
                meta={"box": CROP_2},
            )

    def test_a_manifest_with_another_shape(self) -> None:
        data = _written_seg_2()
        _box_written(data)["bounding_box"]["width"] = 10

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'rect_h2': its duration is 139 and its shape spans 10 in x "
                "(385 to 395); a Box's duration is its shape's x extent."
            ),
        ):
            Timeline.from_dict(data)

    def test_a_manifest_with_another_start(self) -> None:
        data = _written_seg_2()
        _box_written(data).update(start=1, end=140)

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'rect_h2' starts at 1 on timeline 'seg_2', but its shape starts "
                "at x 385 and the timeline stands at x 7 of its image "
                '(meta["box"]): its start is 378.'
            ),
        ):
            Timeline.from_dict(data)

    def test_a_manifest_without_a_shape(self) -> None:
        data = _written_seg_2()
        _box_written(data)["bounding_box"] = None

        with pytest.raises(ValueError, match=_exact(ONE_SHAPE)):
            Timeline.from_dict(data)

    @staticmethod
    def _package(tmp_path: Path) -> tuple[Path, Path]:
        seg_2 = _seg_2()
        seg_2.add_events(Box.from_shape(RECT_H2, origin=7, id="rect_h2"))
        bundle = AlignmentBundle()
        bundle.add_timeline(seg_2)
        manifest = bundle.to_file(tmp_path / "manifest.yaml")
        return manifest, tmp_path / "registry.parquet"

    @staticmethod
    def _with_number(path: Path, column: str, numerator: int) -> None:
        """The registry at *path* with *column* of the Box's row the int *numerator*.

        The registry holds non-event rows too: only the row of ``rect_h2``
        changes.
        """
        table = pq.read_table(path)
        index = table.schema.get_field_index(column)
        cell = pa.array(
            [{"numerator": numerator, "denominator": None, "value": None}]
            * table.num_rows,
            type=table.schema.field(index).type,
        )
        new = pc.if_else(
            pc.equal(table["id"], "rect_h2"), cell, table[column].combine_chunks()
        )
        pq.write_table(table.set_column(index, table.schema.field(index), new), path)

    def test_a_package_with_another_start(self, tmp_path: Path) -> None:
        manifest, boxes = self._package(tmp_path)
        self._with_number(boxes, "start", 1)

        with pytest.raises(
            ValueError,
            match=_exact(
                "Box 'rect_h2' starts at 1 on timeline 'seg_2', but its shape starts "
                "at x 385 and the timeline stands at x 7 of its image "
                '(meta["box"]): its start is 378.'
            ),
        ):
            AlignmentBundle.from_file(manifest)

    def test_a_package_with_another_duration(self, tmp_path: Path) -> None:
        manifest, boxes = self._package(tmp_path)
        self._with_number(boxes, "duration", 10)

        with pytest.raises(
            ValueError, match=re.escape(DURATION_10.replace("Box 'a'", "Box 'rect_h2'"))
        ):
            AlignmentBundle.from_file(manifest)


class TestDictForm:
    @staticmethod
    def _round_trip(timeline: Timeline) -> dict[str, Any]:
        """*timeline*'s dict, after checking it reads back to itself."""
        data = timeline.to_dict(events=True)
        back = Timeline.from_dict(data)
        assert back.to_dict(events=True) == data
        assert back.events.table.to_pylist() == timeline.events.table.to_pylist()
        return data

    def test_int(self) -> None:
        seg_2 = _seg_2()
        seg_2.add_events([_box_row(id="rect_h2", duration=139, bounding_box=RECT_H2)])

        written = _box_written(self._round_trip(seg_2))

        assert json.dumps(written) == (
            '{"id": "rect_h2", "name": null, "temporal_type": "interval", '
            '"event_type": "Box", "start": 378, "end": 517, "duration": 139, '
            '"bounding_box": {"x": 385, "y": 46, "width": 139, "height": 20}}'
        )

    def test_fraction(self) -> None:
        uu = _uu()
        uu.add_events(Box.from_shape(THIRDS, id="f"))

        written = _box_written(self._round_trip(uu))

        assert json.dumps(
            {key: written[key] for key in ("start", "end", "duration", "bounding_box")}
        ) == (
            '{"start": [1, 3], "end": [1, 1], "duration": [2, 3], "bounding_box": '
            '{"x": [1, 3], "y": [0, 1], "width": [2, 3], "height": [1, 1]}}'
        )

    def test_float(self) -> None:
        fl = ContinuousGraphicalTimeline(length=900.0, unit="css_pixels", uid="fl")
        q = Quadrilateral.from_points(
            [(10.5, 1.0), (20.25, 1.0), (20.25, 5.5), (10.5, 5.5)], unit="css_pixels"
        )
        fl.add_events(Box.from_shape(q, id="q"))

        written = _box_written(self._round_trip(fl))

        assert json.dumps(
            {key: written[key] for key in ("start", "end", "duration", "quadrilateral")}
        ) == (
            '{"start": 10.5, "end": 20.25, "duration": 9.75, "quadrilateral": '
            '{"top_left": {"x": 10.5, "y": 1.0}, "top_right": {"x": 20.25, "y": 1.0}, '
            '"bottom_right": {"x": 20.25, "y": 5.5}, '
            '"bottom_left": {"x": 10.5, "y": 5.5}}}'
        )

    def test_a_child_and_its_package(self, tmp_path: Path) -> None:
        page, system = _page_and_system()
        system.add_events(Box.from_shape(MEASURE_1, origin=1221, id="m1"))

        data = self._round_trip(page)
        (m1,) = data["timelines"][0]["children"][0]["events"]
        bundle = AlignmentBundle()
        bundle.add_timeline(page)
        read = AlignmentBundle.from_file(bundle.to_file(tmp_path / "manifest.yaml"))

        assert m1["quadrilateral"] == {
            "top_left": {"x": 1221, "y": 1589},
            "top_right": {"x": 2151, "y": 1589},
            "bottom_right": {"x": 2161, "y": 2266},
            "bottom_left": {"x": 1221, "y": 2266},
        }
        assert (
            read._known_timeline("sys").events.table.to_pylist()
            == system.events.table.to_pylist()
        )

    def test_a_memory_cell_in_a_manifest_is_refused(self) -> None:
        data = _written_seg_2()
        written = _box_written(data)
        written["bounding_box"] = {
            key: {"value": float(value), "numerator": value, "denominator": 1}
            for key, value in written["bounding_box"].items()
        }

        with pytest.raises(ValueError) as refused:
            Timeline.from_dict(data)

        assert str(refused.value).startswith(
            "Box 'rect_h2' of timeline 'seg_2': its bounding_box is not the dict "
            "form of a BoundingBox in pixels as int ("
        )
