"""Children from Regions of a locked timeline.

The validation logic is recorded in ``tests/timelines/README.md`` under
"test_locked_region_child.py". A lock fixes a timeline's length and nothing
more: a child made from a Region inside that length changes nothing the lock
protects.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import Coordinate
from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap, Timeline

_FOUR_FOUR = {
    "time_signature": "4/4",
    "nominal_length": [4, 1],
    "actual_length": [4, 1],
    "start_repeat": False,
    "end_repeat": False,
}
FOUR_BARS = [
    {
        "count": 1,
        "qstamp": [0, 1],
        "number": 1,
        "name": "1",
        **_FOUR_FOUR,
        "next": ["2"],
    },
    {"count": 4, "qstamp": [12, 1], "number": 4, "name": "4", **_FOUR_FOUR},
]
"""Four bars of 4/4 (16 quarters), IDs ``str(count)``, as two compressed rows."""


def _score(*, locked: bool = True) -> ContinuousLogicalTimeline:
    """A quarters timeline of 16 holding a note at 3 (1 long)."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score", locked=locked)
    score.add_events([{"id": "n1", "event_type": "Note", "start": 3, "duration": 1}])
    return score


def _segment_ids(timeline: Timeline) -> list[Any]:
    return timeline.events.filter(event_type="Segment").column_values("id")


class TestALockedTimelineGivesAChildFromARegion:
    def test_the_child_is_made_at_the_regions_offset_and_length(self) -> None:
        score = _score()
        score.create_region(Fraction(5, 2), 10, id="part")

        child = score.create_child_from_region("part", uid="part_child")

        assert score.get_child_offset("part_child") == Coordinate(
            Fraction(5, 2), "quarters"
        )
        assert child.length == Coordinate(Fraction(15, 2), "quarters")
        assert child.events.column_values("start") == [Fraction(1, 2)]
        assert (score.length, score.is_locked) == (Coordinate(16, "quarters"), True)
        assert child.is_locked is True

    def test_with_measures_the_child_holds_the_bars_of_the_region(self) -> None:
        score = _score()
        score.add_measure_map(MeasureMap.from_rows(FOUR_BARS))
        score.create_region(4, 12, id="middle")

        child = score.create_child_from_region("middle", measures=True)

        assert child.length == Coordinate(8, "quarters")
        assert [m.actual_length for m in child.measure_map] == [
            Fraction(4),
            Fraction(4),
        ]
        assert (score.length, score.is_locked) == (Coordinate(16, "quarters"), True)


class TestARegionPastTheLockedEnd:
    def test_the_region_exists_but_gives_no_child(self) -> None:
        score = _score()
        region = score.create_region(12, 20, id="past")
        assert (region.start.value, region.end.value) == (12, 20)
        segments = _segment_ids(score)

        with pytest.raises(
            ValueError, match=r"exceeds timeline length 16\. Timeline is locked\."
        ):
            score.create_child_from_region("past")

        assert score.list_children() == []
        assert _segment_ids(score) == segments
        assert (score.length, score.is_locked) == (Coordinate(16, "quarters"), True)

    def test_an_unlocked_timeline_grows_to_hold_it(self) -> None:
        score = _score(locked=False)
        score.create_region(12, 20, id="past")

        score.create_child_from_region("past")

        assert score.length == Coordinate(20, "quarters")


class TestTheOtherGuardsStay:
    def test_appending_a_child_is_refused(self) -> None:
        score = _score()
        with pytest.raises(
            RuntimeError, match="Cannot append child on locked timeline"
        ):
            score.append_child(ContinuousLogicalTimeline(length=4, unit="quarters"))
        assert score.length == Coordinate(16, "quarters")

    def test_setting_the_length_is_refused(self) -> None:
        score = _score()
        with pytest.raises(RuntimeError, match="Cannot set length on locked timeline"):
            score.length = 20
        assert score.length == Coordinate(16, "quarters")
