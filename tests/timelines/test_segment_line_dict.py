"""The dictionary form of a SegmentLine: the unit plus ``is_segment_line``.

The strategy is documented in ``tests/timelines/README.md`` ("The dictionary
form of a SegmentLine").
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from timetoalign.timelines import SegmentLine, Timeline
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    DiscreteGraphicalTimeline,
)


def _segment(uid: str, length: int) -> DiscreteGraphicalTimeline:
    return DiscreteGraphicalTimeline(length=length, uid=uid)


def _flat() -> SegmentLine:
    """A line of two pixel segments, 300 and 200 px."""
    line = SegmentLine[DiscreteGraphicalTimeline](length=0, uid="line")
    line.append_segment(_segment("first", 300))
    line.append_segment(_segment("second", 200))
    return line


def _nested() -> SegmentLine:
    """An IEEE-shaped edition: a page line of two accolade lines."""
    page_line = SegmentLine[DiscreteGraphicalTimeline]
    edition = SegmentLine[page_line](length=0, uid="edition")
    page_1 = page_line(length=0, uid="page_1")
    page_1.append_segment(_segment("accolade_1", 100))
    page_1.append_segment(_segment("accolade_2", 150))
    page_2 = page_line(length=0, uid="page_2")
    page_2.append_segment(_segment("accolade_3", 120))
    edition.append_segment(page_1)
    edition.append_segment(page_2)
    return edition


def _object(
    timeline_id: str,
    length: int,
    flag: bool,
    children: list[dict[str, Any]],
    *,
    locked: bool,
    offset: int | None = None,
) -> dict[str, Any]:
    """The expected timeline object of one pixel timeline."""
    data: dict[str, Any] = {
        "id": timeline_id,
        "name": None,
        "unit": "pixels",
        "number_type": "int",
    }
    if offset is not None:
        data["offset"] = offset
    return {
        **data,
        "length": length,
        "locked": locked,
        "is_segment_line": flag,
        "meta": {},
        "children": children,
        "conversion_maps": [],
        "regions": [],
        "flow_control": [],
    }


def _manifest(timeline: dict[str, Any]) -> dict[str, Any]:
    """The complete manifest of one root timeline object."""
    return {
        "timeskeleton": "0.7.0",
        "name": None,
        "meta": {},
        "timelines": [timeline],
        "groups": [],
        "measure_maps": [],
    }


def _root(data: dict[str, Any]) -> dict[str, Any]:
    """The root timeline object of a one-root manifest."""
    return data["timelines"][0]


# region Exact dictionaries


def test_plain_timeline_writes_false_and_no_class() -> None:
    """A plain timeline writes the flag false; no timeline object has a class."""
    data = _root(ContinuousLogicalTimeline(length=4, uid="score").to_dict())

    assert "class" not in data
    assert data["is_segment_line"] is False


def test_flat_segment_line_dict_is_exact() -> None:
    """The flag, and the segments as children, which are never locked."""
    assert _flat().to_dict() == _manifest(
        _object(
            "line",
            500,
            True,
            [
                _object("first", 300, False, [], locked=False, offset=0),
                _object("second", 200, False, [], locked=False, offset=300),
            ],
            locked=False,
        )
    )


def test_nested_segment_line_dict_is_exact() -> None:
    """Every line of the nesting carries the flag; only the pages are locked."""
    accolade = lambda uid, length, offset: _object(  # noqa: E731
        uid, length, False, [], locked=False, offset=offset
    )
    page_1 = _object(
        "page_1",
        250,
        True,
        [accolade("accolade_1", 100, 0), accolade("accolade_2", 150, 100)],
        locked=True,
        offset=0,
    )
    page_2 = _object(
        "page_2", 120, True, [accolade("accolade_3", 120, 0)], locked=True, offset=250
    )

    assert _nested().to_dict() == _manifest(
        _object("edition", 370, True, [page_1, page_2], locked=False)
    )


# endregion


# region Round trips


@pytest.mark.parametrize(
    ("build", "expected_type"),
    [
        (_flat, SegmentLine[DiscreteGraphicalTimeline]),
        (_nested, SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]),
    ],
    ids=["flat", "nested"],
)
def test_round_trip_rebuilds_the_line_type(build: Any, expected_type: type) -> None:
    """The flag and the children's class rebuild the parameterized type."""
    data = build().to_dict()

    restored = Timeline.from_dict(data)

    assert type(restored) is expected_type
    assert restored.to_dict() == data


def test_children_out_of_order_rebuild_in_offset_order() -> None:
    """A dictionary listing segments out of order reads as the same line."""
    data = _flat().to_dict()
    _root(data)["children"].reverse()

    restored = Timeline.from_dict(data)

    assert restored.list_segments() == ["first", "second"]
    assert restored.to_dict() == _flat().to_dict()


def test_segments_rebuild_in_offset_order_not_key_order() -> None:
    """Minted ids that sort differently from their offsets rebuild by offset.

    Eleven minted segments are ``dgt_1`` … ``dgt_11``; a writer that sorts
    object keys lists ``dgt_10`` before ``dgt_2``, and the line must still
    append its segments in offset order.
    """
    line = SegmentLine[DiscreteGraphicalTimeline](length=0, uid="line")
    for _ in range(11):
        line.append_segment(DiscreteGraphicalTimeline(length=10))
    data = line.to_dict()
    _root(data)["children"].sort(key=lambda child: child["id"])
    assert [child["id"] for child in _root(data)["children"]][:4] == [
        "dgt_1",
        "dgt_10",
        "dgt_11",
        "dgt_2",
    ]

    restored = Timeline.from_dict(data)

    assert restored.list_segments() == [f"dgt_{n}" for n in range(1, 12)]
    assert restored.to_dict() == line.to_dict()


def test_empty_parameterized_line_round_trips() -> None:
    """A line with no segments rebuilds as a line of its unit's class."""
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="empty")
    data = line.to_dict()

    assert _root(data)["is_segment_line"] is True
    assert type(Timeline.from_dict(data)) is SegmentLine[ContinuousLogicalTimeline]


# endregion


# region Refusals


def _refused(change: Any) -> dict[str, Any]:
    data = copy.deepcopy(_flat().to_dict())
    change(_root(data))
    return data


def _second(data: dict[str, Any]) -> dict[str, Any]:
    return data["children"][1]


def _mixed_classes(data: dict[str, Any]) -> None:
    _second(data).update(unit="points", number_type="float", length=200.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_mixed_classes, r"Segment line timeline 'line' has segments of mixed classes"),
        (
            lambda d: _second(d).update(is_segment_line=True, length=0),
            r"Segment line timeline 'line' has segments of mixed classes",
        ),
        (
            lambda d: _second(d).update(offset=250),
            r"SegmentLine requires contiguous segments\. Expected offset 300, "
            r"got 250\.",
        ),
        (
            lambda d: _second(d).update(offset=350),
            r"^child 'second' ends at 550, beyond the length 500 of its parent "
            r"'line'$",
        ),
        (
            lambda d: d.update(length=600),
            r"Segment line 'line' has segments ending at 500, but its length is 600",
        ),
        (
            lambda d: d.update(is_segment_line="yes"),
            r"timeline 'line' has a non-boolean 'is_segment_line'",
        ),
    ],
    ids=[
        "mixed_classes",
        "mixed_flags",
        "gap",
        "overlap",
        "short_of_length",
        "non_boolean_flag",
    ],
)
def test_from_dict_refuses(change: Any, message: str) -> None:
    """Each malformed segment line is refused with a ValueError naming it."""
    with pytest.raises(ValueError, match=message):
        Timeline.from_dict(_refused(change))


def test_a_stale_class_key_is_not_read() -> None:
    """A ``class`` key names nothing: the unit and the flag decide the type."""
    data = _flat().to_dict()
    _root(data)["class"] = "ContinuousLogicalTimeline"

    restored = Timeline.from_dict(data)

    assert type(restored) is SegmentLine[DiscreteGraphicalTimeline]
    assert restored.to_dict() == data


# endregion


# region Every line can be written


def test_bare_untyped_segment_line_reads_as_a_line_of_its_unit() -> None:
    """A never-typed SegmentLine() writes its unit and reads as a typed line."""
    data = SegmentLine(length=0, unit="pixels", uid="bare").to_dict(minimal=True)

    assert data == {
        "timeskeleton": "0.7.0",
        "timelines": [
            {"id": "bare", "unit": "pixels", "length": 0, "is_segment_line": True}
        ],
    }
    assert type(Timeline.from_dict(data)) is SegmentLine[DiscreteGraphicalTimeline]


def test_empty_nested_segment_line_reads_one_level_shallower() -> None:
    """No key carries the depth of an empty nested line, so it is not restored."""
    line = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
        length=0, uid="empty_edition"
    )

    restored = Timeline.from_dict(line.to_dict())

    assert type(restored) is SegmentLine[DiscreteGraphicalTimeline]
    assert restored.to_dict() == line.to_dict()


# endregion
