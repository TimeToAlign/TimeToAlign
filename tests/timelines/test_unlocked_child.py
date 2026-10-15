"""Unlocked children: embedding means boundary containment only.

The validation logic is recorded in ``tests/timelines/README.md`` under
"test_unlocked_child.py". A change of a timeline's length must stay within
its parent's known length, or it raises ``ValueError`` before anything
changes; a parent never grows because a child grew.
"""

from __future__ import annotations

from typing import Any

import pytest

from timetoalign.timelines import ContinuousPhysicalTimeline, SegmentLine, Timeline


def _s(length: float | None, uid: str, **kwargs: Any) -> ContinuousPhysicalTimeline:
    return ContinuousPhysicalTimeline(length=length, uid=uid, **kwargs)


def _segment_row(parent: Timeline, child_id: str) -> tuple[Any, Any, Any]:
    """The (start, end, duration) of the parent's Segment row for a child."""
    rows = parent.events.filter(event_type="Segment")
    ids = rows.column_values("id")
    index = ids.index(child_id)
    assert ids.count(child_id) == 1
    return tuple(
        rows.column_values(column)[index] for column in ("start", "end", "duration")
    )


def _lengths(*timelines: Timeline) -> list[Any]:
    return [None if tl.length is None else tl.length.value for tl in timelines]


def _manifest(*roots: dict[str, Any]) -> dict[str, Any]:
    return {"timeskeleton": "0.7.0", "timelines": list(roots)}


def _child(
    uid: str, offset: float, length: float | None, **keys: Any
) -> dict[str, Any]:
    return {"id": uid, "offset": offset, "unit": "seconds", "length": length, **keys}


def _three_levels(
    *, parent_locked: bool = False, root_locked: bool = False
) -> tuple[Timeline, Timeline, Timeline]:
    """G (15) holds P (10) at 5, which holds the unlocked C (4) at 3."""
    root = _s(15.0, "G", locked=root_locked)
    parent = _s(10.0, "P")
    child = _s(4.0, "C")
    root.add_child(parent, 5.0, locked=parent_locked)
    parent.add_child(child, 3.0, locked=False)
    return root, parent, child


# region Reading and writing `locked`


def test_a_childs_locked_is_read_and_written_as_it_stands() -> None:
    data = _manifest(
        {
            "id": "rec",
            "unit": "seconds",
            "length": 30.0,
            "children": [
                _child("a", 0.0, 4.0),
                _child("b", 4.0, 4.0, locked=False),
                _child("c", 8.0, 4.0, locked=True),
            ],
        }
    )

    rec = Timeline.from_dict(data)
    complete = rec.to_dict()
    minimal = rec.to_dict(minimal=True)

    assert [rec.get_child(uid).is_locked for uid in "abc"] == [False, False, True]
    assert [child["locked"] for child in complete["timelines"][0]["children"]] == [
        False,
        False,
        True,
    ]
    assert [
        child.get("locked", "absent") for child in minimal["timelines"][0]["children"]
    ] == ["absent", "absent", True]
    assert Timeline.from_dict(complete).to_dict() == complete
    assert Timeline.from_dict(minimal).to_dict(minimal=True) == minimal


def test_add_child_locks_by_default_and_keeps_the_childs_state_when_told() -> None:
    parent = _s(20.0, "P")
    default, unlocked, locked = _s(2.0, "d"), _s(2.0, "u"), _s(2.0, "l", locked=True)

    parent.add_child(default, 0.0)
    parent.add_child(unlocked, 2.0, locked=False)
    parent.add_child(locked, 4.0, locked=False)
    unlocked.length = 3.0

    assert (default.is_locked, unlocked.is_locked, locked.is_locked) == (
        True,
        False,
        True,
    )
    assert _lengths(unlocked, parent) == [3.0, 20.0]
    assert _segment_row(parent, "u") == (2.0, 5.0, 3.0)


def test_a_converted_copy_survives_to_typed_unlocked() -> None:
    parent = Timeline(length=20.0, uid="P")
    parent.add_child(Timeline(length=4.0, uid="C"), 2.0, locked=False)

    typed = parent.to_typed()

    assert typed.get_child("C").is_locked is False


# endregion

# region Growth inside known parents


def _contained(child: str, end: int, parent: str, bound: int) -> str:
    return f"Timeline '{child}' cannot end at {end} on its parent '{parent}', which ends at {bound}"


def test_an_unlocked_child_grows_within_its_parent_only() -> None:
    root, parent, child = _three_levels()

    child.add_events([{"event_type": "Note", "start": 0.0, "end": 6.0}])

    assert _lengths(child, parent, root) == [6.0, 10.0, 15.0]
    assert _segment_row(parent, "C") == (3.0, 9.0, 6.0)
    assert _segment_row(root, "P") == (5.0, 15.0, 10.0)

    child.length = 7.0

    assert _lengths(child, parent, root) == [7.0, 10.0, 15.0]
    assert _segment_row(parent, "C") == (3.0, 10.0, 7.0)


@pytest.mark.parametrize("parent_locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    ("grow", "message"),
    [
        (
            lambda child: child.add_events(
                [{"event_type": "Note", "start": 0.0, "end": 9.0}]
            ),
            _contained("C", 12, "P", 10),
        ),
        (lambda child: setattr(child, "length", 8.0), _contained("C", 11, "P", 10)),
    ],
    ids=["content", "setter"],
)
def test_a_parent_refuses_growth_beyond_it_atomically(
    grow: Any, message: str, parent_locked: bool
) -> None:
    root, parent, child = _three_levels(parent_locked=parent_locked)

    with pytest.raises(ValueError) as error:
        grow(child)

    assert str(error.value) == message
    assert _lengths(child, parent, root) == [4.0, 10.0, 15.0]
    assert child.n_events == 0
    assert _segment_row(parent, "C") == (3.0, 7.0, 4.0)
    assert _segment_row(root, "P") == (5.0, 15.0, 10.0)


def test_content_added_into_an_unlocked_timeline_grows_it_only() -> None:
    root = _s(20.0, "G")
    parent = _s(10.0, "P")
    root.add_child(parent, 5.0, locked=False)

    parent.add_events([{"event_type": "Note", "start": 0.0, "end": 12.0}])

    assert _lengths(parent, root) == [12.0, 20.0]
    assert _segment_row(root, "P") == (5.0, 17.0, 12.0)


@pytest.mark.parametrize(
    "add",
    [
        lambda parent: parent.add_events(
            [{"event_type": "Note", "start": 0.0, "end": 12.0}]
        ),
        lambda parent: parent.add_child(_s(4.0, "K"), 8.0),
    ],
    ids=["content", "child"],
)
def test_growth_beyond_the_own_parent_refuses_the_addition(add: Any) -> None:
    root, parent, child = _three_levels()

    with pytest.raises(ValueError) as error:
        add(parent)

    assert str(error.value) == _contained("P", 17, "G", 15)
    assert _lengths(child, parent, root) == [4.0, 10.0, 15.0]
    assert (parent.n_events, parent.list_children()) == (0, ["C"])
    assert _segment_row(parent, "C") == (3.0, 7.0, 4.0)
    assert _segment_row(root, "P") == (5.0, 15.0, 10.0)


# endregion

# region Unknown length


def test_setting_an_unlocked_childs_length_to_none_and_back() -> None:
    parent, child = _s(20.0, "P"), _s(4.0, "C")
    parent.add_child(child, 2.0, locked=False)

    child.length = None
    assert _lengths(child, parent) == [None, 20.0]
    assert _segment_row(parent, "C") == (2.0, None, None)

    child.length = 6.0
    assert _lengths(child, parent) == [6.0, 20.0]
    assert _segment_row(parent, "C") == (2.0, 8.0, 6.0)

    with pytest.raises(ValueError) as error:
        child.length = 25.0
    assert str(error.value) == _contained("C", 27, "P", 20)
    assert _lengths(child, parent) == [6.0, 20.0]
    assert _segment_row(parent, "C") == (2.0, 8.0, 6.0)


def test_a_parent_of_unknown_length_checks_nothing() -> None:
    parent, child = _s(None, "P"), _s(4.0, "C")
    parent.add_child(child, 2.0, locked=False)

    child.length = 100.0

    assert _lengths(child, parent) == [100.0, None]
    assert _segment_row(parent, "C") == (2.0, 102.0, 100.0)


# endregion

# region Segments


def _line(*, locked: bool = False) -> SegmentLine:
    """A segment line of length 6 with the unlocked segments A [0, 4) and B [4, 6)."""
    line = Timeline.from_dict(
        _manifest(
            {
                "id": "line",
                "unit": "seconds",
                "length": 6.0,
                "locked": locked,
                "is_segment_line": True,
                "children": [_child("A", 0.0, 4.0), _child("B", 4.0, 2.0)],
            }
        )
    )
    assert isinstance(line, SegmentLine)
    return line


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
def test_a_segment_with_a_successor_stays_contiguous(locked: bool) -> None:
    line = _line(locked=locked)
    first = line.get_child("A")

    with pytest.raises(ValueError) as error:
        first.length = 5.0

    assert str(error.value) == (
        "SegmentLine requires contiguous segments. Segment 'A' cannot end at 5: "
        "its successor 'B' starts at 4."
    )
    assert _lengths(first, line.get_child("B"), line) == [4.0, 2.0, 6.0]
    assert _segment_row(line, "A") == (0.0, 4.0, 4.0)

    first.length = None
    assert _lengths(first, line) == [None, 6.0]
    assert _segment_row(line, "A") == (0.0, None, None)

    first.length = 4.0
    assert _lengths(first, line) == [4.0, 6.0]
    assert _segment_row(line, "A") == (0.0, 4.0, 4.0)


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    ("length", "end"), [(3.0, 7), (1.0, 5)], ids=["grow", "shrink"]
)
def test_the_last_segment_keeps_ending_at_the_lines_length(
    locked: bool, length: float, end: int
) -> None:
    line = _line(locked=locked)
    last = line.get_child("B")

    with pytest.raises(ValueError) as error:
        last.length = length

    assert str(error.value) == (
        f"SegmentLine requires contiguous segments. Segment 'B' cannot end at {end}: "
        "the line 'line' ends at 6."
    )
    assert _lengths(last, line) == [2.0, 6.0]
    assert _segment_row(line, "B") == (4.0, 6.0, 2.0)

    last.length = None
    assert _lengths(last, line) == [None, 6.0]
    assert _segment_row(line, "B") == (4.0, None, None)

    last.length = 2.0
    assert _lengths(last, line) == [2.0, 6.0]
    assert _segment_row(line, "B") == (4.0, 6.0, 2.0)


# endregion

# region Segments are never locked

LINE = SegmentLine[ContinuousPhysicalTimeline]
REFUSED_LOCK = "A segment is never locked: lock its segment line '{}' instead."


def _appended(append: str) -> SegmentLine:
    line = LINE(length=0, uid="line")
    getattr(line, append)(_s(4.0, "A"))
    getattr(line, append)(_s(2.0, "B"))
    return line


SEGMENT_LINE_BUILDS = {
    "append_segment": lambda: _appended("append_segment"),
    "append_child": lambda: _appended("append_child"),
    "create_segment_line": lambda: _s(6.0, "src").create_segment_line([0.0, 4.0, 6.0]),
    "from_segmentation": lambda: SegmentLine.from_segmentation(
        _s(6.0, "src"), [0.0, 4.0, 6.0]
    ),
}


def _segments(line: Timeline) -> list[Timeline]:
    return [line.get_child(segment_id) for segment_id in line.list_segments()]


def _written_children(line: Timeline, *, minimal: bool) -> list[dict[str, Any]]:
    return line.to_dict(minimal=minimal)["timelines"][0]["children"]


@pytest.mark.parametrize(
    "build", list(SEGMENT_LINE_BUILDS), ids=list(SEGMENT_LINE_BUILDS)
)
def test_every_construction_path_gives_unlocked_segments(build: str) -> None:
    line = SEGMENT_LINE_BUILDS[build]()

    assert _lengths(*_segments(line), line) == [4.0, 2.0, 6.0]
    assert [segment.is_locked for segment in _segments(line)] == [False, False]
    assert [child["locked"] for child in _written_children(line, minimal=False)] == [
        False,
        False,
    ]
    assert ["locked" in child for child in _written_children(line, minimal=True)] == [
        False,
        False,
    ]


def test_a_segment_built_locked_is_unlocked_once_appended() -> None:
    line = LINE(length=0, uid="line")
    segment = _s(4.0, "A", locked=True)

    line.append_segment(segment)
    line.lock()

    assert (segment.is_locked, line.is_locked) == (False, True)


def _edition() -> SegmentLine:
    """An edition line of two page lines, each a line of plain segments."""
    edition = SegmentLine[LINE](length=0, uid="edition")
    for page_id, lengths in (("page_1", (1.0, 2.0)), ("page_2", (3.0,))):
        page = LINE(length=0, uid=page_id)
        for index, length in enumerate(lengths):
            page.append_segment(_s(length, f"{page_id}_s{index + 1}"))
        edition.append_segment(page)
    return edition


def test_a_segment_that_is_a_segment_line_is_locked() -> None:
    edition = _edition()
    pages = _segments(edition)

    assert [page.is_locked for page in pages] == [True, True]
    assert [[s.is_locked for s in _segments(page)] for page in pages] == [
        [False, False],
        [False],
    ]
    written = _written_children(edition, minimal=True)
    assert [page.get("locked") for page in written] == [True, True]
    assert ["locked" in s for page in written for s in page["children"]] == [
        False,
        False,
        False,
    ]


def test_asking_to_lock_a_plain_segment_is_refused() -> None:
    line = LINE(length=0, uid="line")
    segment = _s(4.0, "A")

    with pytest.raises(ValueError) as error:
        line.add_child(segment, 0.0, allow_expansion=True, locked=True)

    assert str(error.value) == REFUSED_LOCK.format("line")
    assert (line.list_segments(), segment.is_locked) == ([], False)

    line.add_child(segment, 0.0, allow_expansion=True, locked=False)
    with pytest.raises(ValueError) as error:
        segment.lock()

    assert str(error.value) == REFUSED_LOCK.format("line")
    assert segment.is_locked is False

    page = _segments(_edition())[0]
    page.lock()
    assert page.is_locked is True


REFUSED_UNLOCK = (
    "Unlocking a segment line that is a child, a segment or a group member is "
    "not supported yet."
)


def _parents() -> dict[str, Timeline]:
    """A plain parent and an empty edition line, each to receive ``line``."""
    return {
        "child": _s(20.0, "P"),
        "segment": SegmentLine[LINE](length=0, uid="edition"),
    }


@pytest.mark.parametrize("place", ["child", "segment"])
def test_unlocking_a_segment_line_is_refused(place: str) -> None:
    parent = _parents()[place]
    line = _appended("append_segment")

    with pytest.raises(ValueError) as error:
        parent.add_child(line, 0.0, allow_expansion=True, locked=False)

    assert str(error.value) == REFUSED_UNLOCK
    assert (parent.list_children(), line.is_locked, line._parent) == ([], False, None)
    assert _lengths(parent) == [20.0 if place == "child" else 0]

    line.lock()
    parent.add_child(line, 0.0, allow_expansion=True, locked=False)

    assert (parent.list_children(), line.is_locked) == (["line"], True)


def _read_lines(*, page_locked: bool) -> tuple[Timeline, Timeline]:
    """A line whose plain segments state ``locked: true``, and an edition of pages."""

    def page(uid: str, offset: float) -> dict[str, Any]:
        return _child(
            uid,
            offset,
            2.0,
            locked=page_locked,
            is_segment_line=True,
            children=[_child(f"{uid}_s1", 0.0, 2.0, locked=True)],
        )

    plain = {
        "id": "line",
        "unit": "seconds",
        "length": 6.0,
        "is_segment_line": True,
        "children": [
            _child("A", 0.0, 4.0, locked=True),
            _child("B", 4.0, 2.0, locked=True),
        ],
    }
    edition = {
        "id": "edition",
        "unit": "seconds",
        "length": 4.0,
        "is_segment_line": True,
        "children": [page("page_1", 0.0), page("page_2", 2.0)],
    }
    return Timeline.from_dict(_manifest(plain)), Timeline.from_dict(_manifest(edition))


@pytest.mark.parametrize(
    "page_locked", [False, True], ids=["page_unlocked", "page_locked"]
)
def test_reading_unlocks_a_plain_segment_and_keeps_a_lines_lock(
    page_locked: bool,
) -> None:
    line, pages = _read_lines(page_locked=page_locked)
    plain, edition = (timeline.to_dict()["timelines"][0] for timeline in (line, pages))
    plain_minimal, edition_minimal = (
        timeline.to_dict(minimal=True)["timelines"][0] for timeline in (line, pages)
    )

    assert [segment.is_locked for segment in _segments(line)] == [False, False]
    assert [page.is_locked for page in _segments(pages)] == [page_locked] * 2
    assert [child["locked"] for child in plain["children"]] == [False, False]
    assert ["locked" in child for child in plain_minimal["children"]] == [False, False]
    assert [page["locked"] for page in edition["children"]] == [page_locked] * 2
    assert [page.get("locked", False) for page in edition_minimal["children"]] == [
        page_locked
    ] * 2
    assert [
        segment["locked"]
        for page in edition["children"]
        for segment in page["children"]
    ] == [False, False]


# endregion


# region An unlocked segment keeps its line contiguous


def _contiguity(segment: str, end: int, where: str, bound: int) -> str:
    return (
        f"SegmentLine requires contiguous segments. Segment '{segment}' cannot end "
        f"at {end}: {where} {bound}."
    )


def _event(end: float) -> dict[str, Any]:
    return {"event_type": "Note", "start": 0.0, "end": end}


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    ("segment", "add", "message"),
    [
        (
            "A",
            lambda seg: seg.add_events([_event(5.0)]),
            _contiguity("A", 5, "its successor 'B' starts at", 4),
        ),
        (
            "B",
            lambda seg: seg.add_events([_event(3.0)]),
            _contiguity("B", 7, "the line 'line' ends at", 6),
        ),
        (
            "A",
            lambda seg: seg.append_child(_s(1.0, "x")),
            _contiguity("A", 5, "its successor 'B' starts at", 4),
        ),
    ],
    ids=["event_past_successor", "event_past_line", "child_past_successor"],
)
def test_growing_a_segment_past_its_boundary_is_refused(
    locked: bool, segment: str, add: Any, message: str
) -> None:
    line = _appended("append_segment")
    if locked:
        line.lock()
    target = line.get_child(segment)

    with pytest.raises(ValueError) as error:
        add(target)

    assert str(error.value) == message
    assert _lengths(*_segments(line), line) == [4.0, 2.0, 6.0]
    assert (target.n_events, target.n_children) == (0, 0)


def test_content_inside_a_segment_changes_no_length() -> None:
    line = _appended("append_segment")

    line.get_child("A").add_events([_event(4.0)])
    line.get_child("B").add_events([_event(2.0)])

    assert _lengths(*_segments(line), line) == [4.0, 2.0, 6.0]


def test_the_last_segment_of_a_line_of_unknown_length_grows() -> None:
    line = _appended("append_segment")
    line.length = None
    first, last = _segments(line)

    last.add_events([_event(7.0)])

    assert _lengths(first, last, line) == [4.0, 7.0, None]
    assert _segment_row(line, "B") == (4.0, 11.0, 7.0)
    with pytest.raises(ValueError) as error:
        first.add_events([_event(5.0)])
    assert str(error.value) == _contiguity("A", 5, "its successor 'B' starts at", 4)


# endregion


# region Copies, casts and slices


def _two_children(cls: type[Timeline] = ContinuousPhysicalTimeline) -> Timeline:
    """P (4 s) holding the unlocked children A [0, 2) and B [2, 4)."""
    parent = cls(length=4.0, unit="seconds", uid="P")
    parent.add_child(cls(length=2.0, unit="seconds", uid="A"), 0.0, locked=False)
    parent.add_child(cls(length=2.0, unit="seconds", uid="B"), 2.0, locked=False)
    return parent


def _typed_copy() -> tuple[Timeline, Timeline]:
    source = _two_children(Timeline)
    return source, source.to_typed()


def _line_copy() -> tuple[Timeline, Timeline]:
    source = _two_children()
    return source, source.as_segment_line()


def _plain_copy() -> tuple[Timeline, Timeline]:
    source = _two_children().as_segment_line()
    return source, source.to_timeline()


@pytest.mark.parametrize(
    "build",
    [_typed_copy, _line_copy, _plain_copy],
    ids=["to_typed", "as_segment_line", "to_timeline"],
)
def test_a_copy_shares_no_timeline_with_its_source(build: Any) -> None:
    source, result = build()
    original, copied = source.get_child("A"), result.get_child("A")

    copied.length = None

    assert copied is not original
    assert original._parent is source
    assert copied._parent is result
    assert _lengths(original, copied) == [2.0, None]
    assert _segment_row(source, "A") == (0.0, 2.0, 2.0)
    assert _segment_row(result, "A") == (0.0, None, None)


def test_casts_keep_the_root_offset_and_the_keys_read() -> None:
    data = {
        **_manifest(
            {
                "id": "P",
                "unit": "seconds",
                "length": 4.0,
                "offset": -2.5,
                "x-root": 1,
                "children": [_child("A", 0.0, 2.0), _child("B", 2.0, 2.0)],
            }
        ),
        "meta": {"k": 1},
    }
    read = Timeline.from_dict(data)

    for cast in (read.as_segment_line(), read.as_segment_line().to_timeline()):
        written = cast.to_dict(minimal=True)
        root = written["timelines"][0]
        assert cast.root_offset == read.root_offset
        assert (root["offset"], root["x-root"], written["meta"]) == (-2.5, 1, {"k": 1})


def test_a_slice_keeps_each_childs_lock() -> None:
    parent = _s(10.0, "P")
    parent.add_child(_s(4.0, "U"), 0.0, locked=False)
    parent.add_child(_s(4.0, "L"), 5.0)

    sliced = parent.get_slice(0.0, 10.0)

    assert [
        (sliced.get_child_offset(cid).value, sliced.get_child(cid).is_locked)
        for cid in sliced.list_children()
    ] == [(0.0, False), (5.0, True)]


# endregion
