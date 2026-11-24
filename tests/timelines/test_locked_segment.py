"""A locked timeline placed in another keeps a fixed length.

The validation logic is recorded in ``tests/timelines/README.md`` under
"test_locked_segment.py". As a child, a locked timeline stays locked on
every route; as a plain segment it reads unlocked, and its line's
boundaries fix its length instead.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate
from timetoalign.timelines import ContinuousPhysicalTimeline, SegmentLine, Timeline

LOCKED_GROWTH = (
    "Adding content ending at 5 exceeds timeline length 4. Timeline is locked. "
    "Use allow_expansion=True to override."
)
LOCKED_CLAIM = (
    "Pair 0: the side on timeline 'rec' reaches 5.0 seconds, past the end of "
    "timeline 'rec' at 4 seconds. The timeline is locked: its length is fixed, as "
    "a loader fixes a length its source states, and a MatchClaim never lengthens "
    "it. Correct the span; or, if the source is known to be longer than the "
    "timeline states, make the timeline with the source's length."
)


def _recording(uid: str, length: float) -> ContinuousPhysicalTimeline:
    """A recording whose source states its length, locked as a loader locks it."""
    return ContinuousPhysicalTimeline(length=length, uid=uid, locked=True)


def _note(end: float) -> dict[str, Any]:
    return {"event_type": "Note", "start": 0.0, "end": end}


def _bundle(root: Timeline) -> AlignmentBundle:
    bundle = AlignmentBundle()
    bundle.add_timeline(root, as_group=root.id)
    reference = ContinuousPhysicalTimeline(length=100.0, uid="ref")
    bundle.add_timeline(reference, as_group="ref")
    return bundle


def _claim_refused(bundle: AlignmentBundle, timeline_id: str, at: float) -> str:
    with pytest.raises(ValueError) as error:
        bundle.create_match_claims(
            [({"start": 1.0}, "ref", {"start": at}, timeline_id)]
        )
    assert len(bundle.match_claim_table) == 0
    return str(error.value)


# region As a child, the lock is kept


def _added(parent: Timeline, child: Timeline) -> None:
    parent.add_child(child, 0.0)


def _added_keeping(parent: Timeline, child: Timeline) -> None:
    parent.add_child(child, 0.0, locked=False)


def _appended(parent: Timeline, child: Timeline) -> None:
    parent.append_child(child)


CHILD_ROUTES: dict[str, tuple[float, Callable[[Timeline, Timeline], None]]] = {
    "add_child": (20.0, _added),
    "add_child_locked_false": (20.0, _added_keeping),
    "append_child": (0.0, _appended),
}


@pytest.mark.parametrize("route", list(CHILD_ROUTES), ids=list(CHILD_ROUTES))
def test_a_locked_timeline_stays_locked_as_a_child(route: str) -> None:
    parent_length, embed = CHILD_ROUTES[route]
    parent = ContinuousPhysicalTimeline(length=parent_length, uid="P")
    recording = _recording("rec", 4.0)

    embed(parent, recording)

    assert (recording.is_locked, recording.length) == (True, Coordinate(4.0, "seconds"))

    with pytest.raises(ValueError) as error:
        recording.add_events([_note(5.0)])
    assert str(error.value) == LOCKED_GROWTH
    assert (recording.length, recording.n_events) == (Coordinate(4.0, "seconds"), 0)

    assert _claim_refused(_bundle(parent), "rec", 5.0) == LOCKED_CLAIM
    assert recording.length == Coordinate(4.0, "seconds")

    read = Timeline.from_dict(parent.to_dict()).get_child("rec")
    assert (read.is_locked, read.length) == (True, Coordinate(4.0, "seconds"))


# endregion


# region As a plain segment, the line's boundaries fix the length


def _contiguity(segment: str, end: int, where: str, bound: int) -> str:
    return (
        f"SegmentLine requires contiguous segments. Segment '{segment}' cannot end "
        f"at {end}: {where} {bound}."
    )


def _disc() -> SegmentLine:
    """An empty line given the locked recordings A (4 s) and B (2 s) as segments."""
    line = SegmentLine[ContinuousPhysicalTimeline](length=0, uid="line")
    line.append_segment(_recording("A", 4.0))
    line.append_segment(_recording("B", 2.0))
    return line


def _lengths(line: SegmentLine) -> list[Any]:
    return [line.get_child(uid).length.value for uid in ("A", "B")] + [
        line.length.value
    ]


SEGMENT_CASES = {
    "with_successor": ("A", 5.0, _contiguity("A", 5, "its successor 'B' starts at", 4)),
    "last": ("B", 3.0, _contiguity("B", 7, "the line 'line' ends at", 6)),
}


@pytest.mark.parametrize("case", list(SEGMENT_CASES), ids=list(SEGMENT_CASES))
@pytest.mark.parametrize("allow_expansion", [False, True], ids=["plain", "expanding"])
def test_a_locked_timeline_appended_as_a_segment_keeps_its_length(
    case: str, allow_expansion: bool
) -> None:
    segment_id, past_end, refusal = SEGMENT_CASES[case]
    line = _disc()
    segment = line.get_child(segment_id)

    assert [line.get_child(uid).is_locked for uid in ("A", "B")] == [False, False]

    with pytest.raises(ValueError) as error:
        segment.add_events([_note(past_end)], allow_expansion=allow_expansion)
    assert str(error.value) == refusal
    assert (_lengths(line), segment.n_events) == ([4.0, 2.0, 6.0], 0)


@pytest.mark.parametrize("case", list(SEGMENT_CASES), ids=list(SEGMENT_CASES))
def test_a_claim_past_the_end_of_such_a_segment_is_refused(case: str) -> None:
    segment_id, past_end, refusal = SEGMENT_CASES[case]
    line = _disc()
    length = line.get_child(segment_id).length.value

    message = _claim_refused(_bundle(line), segment_id, past_end)

    assert message == (
        f"Pair 0: the side on timeline '{segment_id}' reaches {past_end} seconds, "
        f"past the end of timeline '{segment_id}' at {length:g} seconds, and the "
        f"timeline cannot grow to hold it: {refusal.rstrip('.')}. A MatchClaim "
        "lengthens an unlocked timeline only where its parent contains the new end, "
        "and a parent never grows with its child. Correct the span; or, if the "
        "position lies beyond the timeline, state it on its parent 'line'."
    )
    assert _lengths(line) == [4.0, 2.0, 6.0]


# endregion
