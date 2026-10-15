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
