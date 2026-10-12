"""The dictionary form of a SegmentLine: plain class plus ``is_segment_line``.

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


def _px(value: int) -> dict[str, Any]:
    """A pixel coordinate dictionary."""
    return {"unit": "pixels", "number_type": "int", "value": value}


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


def _manifest(
    timeline_id: str,
    length: int,
    flag: bool,
    children: dict[str, Any],
    *,
    locked: bool,
) -> dict[str, Any]:
    """The expected dictionary of one pixel timeline."""
    return {
        "id": timeline_id,
        "name": None,
        "class": "DiscreteGraphicalTimeline",
        "is_segment_line": flag,
        "unit": "pixels",
        "number_type": "int",
        "length": _px(length),
        "locked": locked,
        "meta": {},
        "children": children,
        "conversion_maps": [],
        "regions": [],
        "flow_control": {"breaks": [], "jumps": [], "markers": []},
        "measure_map": None,
        "metric_hierarchy": None,
    }


def _child(offset: int, manifest: dict[str, Any]) -> dict[str, Any]:
    return {"offset": _px(offset), "timeline": manifest}


# region Exact dictionaries


def test_plain_timeline_writes_its_class_and_false() -> None:
    """A plain timeline names its own class; the flag is false."""
    data = ContinuousLogicalTimeline(length=4, uid="score").to_dict()

    assert data["class"] == "ContinuousLogicalTimeline"
    assert data["is_segment_line"] is False


def test_flat_segment_line_dict_is_exact() -> None:
    """The segments' class, the flag, and the segments as children."""
    assert _flat().to_dict() == _manifest(
        "line",
        500,
        True,
        {
            "first": _child(0, _manifest("first", 300, False, {}, locked=True)),
            "second": _child(300, _manifest("second", 200, False, {}, locked=True)),
        },
        locked=False,
    )


def test_nested_segment_line_dict_is_exact() -> None:
    """Every line of the nesting names the bottom class and carries the flag."""
    accolade = lambda uid, length: _manifest(  # noqa: E731
        uid, length, False, {}, locked=True
    )
    page_1 = _manifest(
        "page_1",
        250,
        True,
        {
            "accolade_1": _child(0, accolade("accolade_1", 100)),
            "accolade_2": _child(100, accolade("accolade_2", 150)),
        },
        locked=True,
    )
    page_2 = _manifest(
        "page_2",
        120,
        True,
        {"accolade_3": _child(0, accolade("accolade_3", 120))},
        locked=True,
    )

    assert _nested().to_dict() == _manifest(
        "edition",
        370,
        True,
        {"page_1": _child(0, page_1), "page_2": _child(250, page_2)},
        locked=False,
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
    data["children"] = dict(reversed(list(data["children"].items())))

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
    data["children"] = dict(sorted(data["children"].items()))
    assert list(data["children"])[:4] == ["dgt_1", "dgt_10", "dgt_11", "dgt_2"]

    restored = Timeline.from_dict(data)

    assert restored.list_segments() == [f"dgt_{n}" for n in range(1, 12)]
    assert restored.to_dict() == line.to_dict()


def test_empty_parameterized_line_round_trips() -> None:
    """A line with no segments but a plain segment type names that type."""
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="empty")
    data = line.to_dict()

    assert data["class"] == "ContinuousLogicalTimeline"
    assert data["is_segment_line"] is True
    assert type(Timeline.from_dict(data)) is SegmentLine[ContinuousLogicalTimeline]


# endregion


# region Refusals


def _refused(change: Any) -> dict[str, Any]:
    data = copy.deepcopy(_flat().to_dict())
    change(data)
    return data


def _mixed_classes(data: dict[str, Any]) -> None:
    data["children"]["second"]["timeline"]["class"] = "ContinuousGraphicalTimeline"
    data["children"]["second"]["timeline"]["unit"] = "points"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_mixed_classes, r"Segment line 'line' has segments of mixed classes"),
        (
            lambda d: d["children"]["second"]["timeline"].update(is_segment_line=True),
            r"Segment line 'line' has segments of mixed classes",
        ),
        (
            lambda d: d["children"]["second"].update(offset=_px(250)),
            r"SegmentLine requires contiguous segments\. Expected offset 300, "
            r"got 250\.",
        ),
        (
            lambda d: d["children"]["second"].update(offset=_px(350)),
            r"SegmentLine requires contiguous segments\. Expected offset 300, "
            r"got 350\.",
        ),
        (
            lambda d: d.update(length=_px(600)),
            r"Segment line 'line' has segments ending at 500, but its length is 600",
        ),
        (
            lambda d: d.update({"class": "ContinuousLogicalTimeline"}),
            r"Segment line 'line' names class 'ContinuousLogicalTimeline' but its "
            r"segments are 'DiscreteGraphicalTimeline'",
        ),
        (
            lambda d: d.pop("is_segment_line"),
            r"Serialized timeline 'line' is missing a boolean 'is_segment_line'",
        ),
        (
            lambda d: d.update(is_segment_line="yes"),
            r"Serialized timeline 'line' is missing a boolean 'is_segment_line'",
        ),
        (
            lambda d: d.update({"class": "SegmentLine[DiscreteGraphicalTimeline]"}),
            r"Unknown serialized timeline class "
            r"'SegmentLine\[DiscreteGraphicalTimeline\]'",
        ),
    ],
    ids=[
        "mixed_classes",
        "mixed_flags",
        "gap",
        "overlap",
        "short_of_length",
        "class_mismatch",
        "missing_flag",
        "non_boolean_flag",
        "dynamic_class_name",
    ],
)
def test_from_dict_refuses(change: Any, message: str) -> None:
    """Each malformed segment line is refused with a ValueError naming it."""
    with pytest.raises(ValueError, match=message):
        Timeline.from_dict(_refused(change))


# endregion


# region What cannot be written


def test_bare_untyped_segment_line_cannot_be_written() -> None:
    """A never-typed SegmentLine() cannot say what it is."""
    with pytest.raises(ValueError) as error:
        SegmentLine(length=0, unit="pixels", uid="bare").to_dict()

    assert str(error.value) == (
        "SegmentLine 'bare' has no segments and no plain segment type, so its "
        "dictionary form cannot say what it is"
    )


def test_empty_nested_segment_line_cannot_be_written() -> None:
    """An empty nested line would read back one level shallower, so it refuses."""
    line = SegmentLine[SegmentLine[DiscreteGraphicalTimeline]](
        length=0, uid="empty_edition"
    )

    with pytest.raises(ValueError) as error:
        line.to_dict()

    assert str(error.value) == (
        "SegmentLine 'empty_edition' has no segments and no plain segment type, "
        "so its dictionary form cannot say what it is"
    )


# endregion
