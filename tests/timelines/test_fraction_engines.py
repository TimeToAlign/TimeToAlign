"""Regression tests for exact coordinate propagation through timeline engines."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import NumberType, TimeUnit, struct_to_rational
from timetoalign.core.time import struct_to_coordinate
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    SegmentLine,
    Timeline,
)


def _fraction_timeline(uid: str, length: Fraction) -> Timeline:
    """Create a fraction-valued timeline in quarter-note units."""
    return Timeline(
        length=length,
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
        uid=uid,
    )


def _assert_coordinate_struct_is_consistent(value: object) -> None:
    """Require every exact coordinate struct to agree with its float value."""
    if not isinstance(value, dict) or value.get("numerator") is None:
        return
    rational = struct_to_rational(value)
    assert value["value"] == float(rational)


def test_child_event_offsets_preserve_exact_coordinate_pairs() -> None:
    """Child event coordinates keep exact pairs after parent-space shifting."""
    parent = _fraction_timeline("parent", Fraction(4))
    child = _fraction_timeline("child", Fraction(2))
    child.add_events(
        [
            {
                "id": "child:instant",
                "event_type": "Beat",
                "instant": Fraction(1, 3),
            },
            {
                "id": "child:note",
                "event_type": "Note",
                "start": Fraction(1, 3),
                "end": Fraction(5, 6),
            },
        ]
    )
    parent.add_child(child, offset=Fraction(1, 6))

    single = parent.get_event("child:note")
    bulk = {event["id"]: event for event in parent.get_events()}

    assert single is not None
    assert single["start"]["value"] == float(Fraction(1, 2))
    assert single["start"]["numerator"] == 1
    assert single["start"]["denominator"] == 2
    assert struct_to_rational(single["start"]) == Fraction(1, 2)
    assert single["end"]["value"] == 1.0
    assert single["end"]["numerator"] == 1
    assert single["end"]["denominator"] == 1
    assert struct_to_rational(single["end"]) == Fraction(1)
    shifted_instant = parent.get_event("child:instant")
    assert shifted_instant is not None
    assert struct_to_rational(shifted_instant["start"]) == Fraction(1, 2)
    assert bulk["child:note"]["start"] == single["start"]
    assert bulk["child:note"]["end"] == single["end"]

    for event in [single, *bulk.values()]:
        for field in ("start", "end", "duration"):
            _assert_coordinate_struct_is_consistent(event.get(field))


def test_segment_extraction_preserves_exact_pairs_after_shift() -> None:
    """Copied segment events retain exact values relative to the segment start."""
    source = _fraction_timeline("source", Fraction(3))
    source.add_events(
        [
            {
                "id": "source:note",
                "event_type": "Note",
                "start": Fraction(1, 2),
                "end": Fraction(3, 2),
            }
        ]
    )

    segments = source.create_segment_line(
        [Fraction(1, 3), Fraction(2)],
        copy_events=True,
    )
    copied = list(
        segments._children[segments._segment_order[0]].get_events(
            include_children=False
        )
    )[0]

    assert copied["start"]["numerator"] == 1
    assert copied["start"]["denominator"] == 6
    assert struct_to_rational(copied["start"]) == Fraction(1, 6)
    assert copied["end"]["numerator"] == 7
    assert copied["end"]["denominator"] == 6
    assert struct_to_rational(copied["end"]) == Fraction(7, 6)
    assert copied["duration"]["numerator"] == 1
    assert copied["duration"]["denominator"] == 1
    assert struct_to_rational(copied["duration"]) == Fraction(1)
    for field in ("start", "end", "duration"):
        _assert_coordinate_struct_is_consistent(copied[field])


def test_region_and_grouped_segment_boundaries_preserve_exact_values() -> None:
    """Region-derived boundaries retain exact event and input coordinates."""
    timeline = _fraction_timeline("regions", Fraction(2))
    timeline.add_events(
        [
            {
                "id": "regions:break",
                "event_type": "Break",
                "instant": Fraction(1, 3),
                "breaks": True,
            },
            {
                "id": "regions:one",
                "event_type": "Marker",
                "start": Fraction(1, 3),
                "end": Fraction(2, 3),
                "group": "one",
            },
            {
                "id": "regions:two",
                "event_type": "Marker",
                "start": Fraction(2, 3),
                "end": Fraction(1),
                "group": "two",
            },
        ]
    )

    regions = timeline.create_regions_from_boundaries([Fraction(1, 3), Fraction(1)])
    assert regions[0].start.value == Fraction(1, 3)
    assert regions[0].end.value == Fraction(1)

    split_regions = timeline.create_regions_by_splitting("breaks", prefix="split")
    assert split_regions[1].start.value == Fraction(1, 3)
    assert split_regions[1].end.value == Fraction(2)

    grouped_regions = timeline.create_regions_by_grouping("group")
    assert grouped_regions[0].start.value == Fraction(1, 3)
    assert grouped_regions[0].end.value == Fraction(2, 3)

    grouped_segments = timeline.create_segment_line_by_grouping("group")
    grouped_child = grouped_segments._children[grouped_segments._segment_order[0]]
    assert grouped_child.length.value == Fraction(1, 3)


def test_fraction_timestamp_coordinate_uses_stored_pair() -> None:
    """Exact event coordinates materialize from the stored rational pair."""
    timeline = _fraction_timeline("fraction", Fraction(2))
    timeline.add_events(
        [{"id": "fraction:beat", "event_type": "Beat", "instant": Fraction(2, 3)}]
    )
    stored = timeline.events.table.column("start").combine_chunks()[0].as_py()

    stamp = timeline.get_timestamp(Fraction(stored["numerator"], stored["denominator"]))
    coordinate = stamp.get_coordinate(timeline.id, format="coordinate")

    assert stamp.get_coordinate_for(timeline.id, format="float") == stored["value"]
    assert coordinate.value == Fraction(2, 3)
    assert isinstance(coordinate.value, Fraction)

    direct_query = timeline.get_timestamp(0.5)
    assert direct_query.is_interpolated is False


# -- Boundaries reach the event paths in the axis's number type -------------

_AXES = {
    "fraction": (ContinuousLogicalTimeline, Fraction(1, 10)),
    "int": (DiscreteLogicalTimeline, 10),
    "float": (ContinuousPhysicalTimeline, 0.5),
}
_PARTS = [["n0"], ["n1", "n2"], ["n3", "n4", "n5", "n6"], ["n7", "n8", "n9"]]


def _ten_notes(axis: str) -> tuple[Timeline, Any]:
    """Notes ``n0``..``n9`` on ``[k*s, (k+1)*s)``; ``n0``, ``n2``, ``n6`` cut."""
    timeline_class, step = _AXES[axis]
    timeline = timeline_class(length=10 * step)
    timeline.add_events(
        [
            {
                "id": f"n{k}",
                "event_type": "Note",
                "start": k * step,
                "end": (k + 1) * step,
                **({"cut": True} if k in (0, 2, 6) else {}),
            }
            for k in range(10)
        ]
    )
    return timeline, step


def _typed(value: Any) -> tuple[type, Any]:
    return type(value), value


def _local_starts(timeline: Timeline) -> list[tuple[str, tuple[type, Any]]]:
    """Each own event's ID and child-local start, typed as the axis reads it."""
    return [
        (row["id"], _typed(struct_to_coordinate(row["start"], timeline.number_type)))
        for row in timeline.get_events(include_children=False)
    ]


def _expected_part(ids: list[str], step: Any) -> list[tuple[str, tuple[type, Any]]]:
    return [(event_id, _typed(j * step)) for j, event_id in enumerate(ids)]


def _parts(line: SegmentLine) -> list[tuple[Any, Any, Any]]:
    """Each segment's typed offset, typed length and typed local starts."""
    parts = []
    for index in range(line.n_segments):
        offset, segment = line.get_segment_by_index(index)
        parts.append(
            (_typed(offset.value), _typed(segment.length.value), _local_starts(segment))
        )
    return parts


def _expected_parts(step: Any) -> list[tuple[Any, Any, Any]]:
    offsets, lengths = [0, 1, 3, 7], [1, 2, 4, 3]
    return [
        (_typed(offset * step), _typed(length * step), _expected_part(ids, step))
        for offset, length, ids in zip(offsets, lengths, _PARTS, strict=True)
    ]


@pytest.mark.parametrize("axis", list(_AXES))
class TestBoundariesInTheAxisType:
    """Library callers hand ``get_events`` boundaries in the axis's type."""

    def test_child_from_region(self, axis: str) -> None:
        timeline, step = _ten_notes(axis)
        timeline.create_region(1 * step, 3 * step, id="r")

        child = timeline.create_child_from_region("r")

        assert _typed(timeline.get_child_offset(child.id).value) == _typed(1 * step)
        assert _typed(child.length.value) == _typed(2 * step)
        # The copies are new events of the tree: IDs of their own (event IDs
        # are unique among a root and its descendants), the starts of n1 and n2.
        copies = _local_starts(child)
        assert [start for _, start in copies] == [
            start for _, start in _expected_part(["n1", "n2"], step)
        ]
        assert [event_id[:5] for event_id, _ in copies] == ["note_"] * 2
        assert not {event_id for event_id, _ in copies} & set(
            timeline.events.column_values("id")
        )

    def test_segment_line_by_splitting(self, axis: str) -> None:
        timeline, step = _ten_notes(axis)

        line = timeline.create_segment_line_by_splitting({"cut": True})

        assert _parts(line) == _expected_parts(step)

    def test_segment_line_from_segmentation(self, axis: str) -> None:
        timeline, step = _ten_notes(axis)

        line = SegmentLine.from_segmentation(
            timeline, [0, 1 * step, 3 * step, 7 * step, 10 * step]
        )

        assert _parts(line) == _expected_parts(step)


def _ids(rows: Any) -> list[str]:
    return [row["id"] for row in rows]


def test_a_slice_ends_before_its_exact_end() -> None:
    """``get_slice`` compares instants with its exact bounds."""
    timeline = _fraction_timeline("beats", Fraction(1))
    timeline.add_events(
        [
            {"id": f"b{k}", "event_type": "Beat", "instant": Fraction(k, 10)}
            for k in range(10)
        ]
    )

    sliced = timeline.get_slice(0, Fraction(1, 10))

    assert _ids(sliced.get_events()) == ["b0"]


def test_a_slice_holds_the_intervals_fully_inside() -> None:
    """Without truncation, an interval ending at the slice's end is inside."""
    timeline, _ = _ten_notes("fraction")

    sliced = timeline.get_slice(Fraction(3, 10), Fraction(7, 10), truncate_events=False)

    assert _ids(sliced.get_events()) == ["n3", "n4", "n5", "n6"]


def test_the_region_at_an_exact_boundary_is_the_one_starting_there() -> None:
    timeline, _ = _ten_notes("fraction")
    timeline.create_region(Fraction(1, 10), Fraction(3, 10), id="before")
    timeline.create_region(Fraction(3, 10), Fraction(7, 10), id="after")

    assert [region.id for region in timeline.get_regions_at(Fraction(3, 10))] == [
        "after"
    ]


def _tenth_and_third() -> Timeline:
    timeline = _fraction_timeline("score", Fraction(1))
    timeline.add_events(
        [
            {"id": "tenth", "event_type": "Beat", "instant": Fraction(1, 10)},
            {"id": "third", "event_type": "Beat", "instant": Fraction(1, 3)},
        ]
    )
    return timeline


def _found(found: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
    """``get_events_at``'s answer as event IDs per timeline."""
    return {key: _ids(rows) for key, rows in found.items()}


def _two_notes(number_type: str) -> Timeline:
    """Notes at 1 and 2 on a timeline of *number_type* (seconds, quarters, ticks)."""
    timeline = {
        "float": ContinuousPhysicalTimeline(length=4.0, uid="axis"),
        "fraction": ContinuousLogicalTimeline(length=Fraction(4), uid="axis"),
        "int": DiscreteLogicalTimeline(length=4, uid="axis"),
    }[number_type]
    timeline.add_events(
        [
            {"id": "one", "event_type": "Note", "start": 1, "end": 2},
            {"id": "two", "event_type": "Note", "start": 2, "end": 3},
        ]
    )
    return timeline


class TestQueriesOnExactAxes:
    """A float query compares as a key, an exact one exactly, never rounded."""

    @pytest.mark.parametrize(
        ("query", "ids"),
        [
            ({"min_coord": 0.1}, ["tenth", "third"]),
            ({"max_coord": 0.1}, []),
            ({"min_coord": Fraction(1, 10)}, ["tenth", "third"]),
            ({"min_coord": Fraction(0.1)}, ["third"]),
        ],
        ids=["min-float", "max-float", "min-fraction", "min-double-ratio"],
    )
    def test_bounds_on_a_fraction_axis(
        self, query: dict[str, Any], ids: list[str]
    ) -> None:
        assert _ids(_tenth_and_third().get_events(**query)) == ids

    @pytest.mark.parametrize(
        ("at", "ids"),
        [
            (0.1, ["tenth"]),
            (1 / 3, ["third"]),
            (Fraction(1, 10), ["tenth"]),
            (Fraction(0.1), []),
        ],
        ids=["tenth-float", "third-float", "tenth-fraction", "tenth-double-ratio"],
    )
    def test_positions_on_a_fraction_axis(self, at: Any, ids: list[str]) -> None:
        found = _tenth_and_third().get_events_at(at)

        assert _found(found) == ({"score": ids} if ids else {})

    @pytest.mark.parametrize("at", [2 / 3, Fraction(2, 3)], ids=["float", "fraction"])
    def test_an_interval_ending_at_the_query_is_not_active(self, at: Any) -> None:
        timeline = _fraction_timeline("score", Fraction(1))
        timeline.add_events(
            [
                {
                    "id": "b",
                    "event_type": "Note",
                    "start": Fraction(1, 3),
                    "end": Fraction(2, 3),
                },
                {
                    "id": "c",
                    "event_type": "Note",
                    "start": Fraction(1, 2),
                    "end": Fraction(1),
                },
            ]
        )

        assert _found(timeline.get_events_at(at)) == {"score": ["c"]}

    @staticmethod
    def _ticks() -> Timeline:
        timeline = DiscreteLogicalTimeline(length=20, uid="ticks")
        timeline.add_events(
            [
                {"id": "x", "event_type": "Note", "start": 10, "end": 11},
                {"id": "y", "event_type": "Note", "start": 11, "end": 12},
            ]
        )
        return timeline

    @pytest.mark.parametrize(
        ("query", "ids"),
        [
            ({"min_coord": 10.5}, ["y"]),
            ({"max_coord": 10.5}, ["x"]),
            ({"min_coord": Fraction(21, 2)}, ["y"]),
        ],
        ids=["min-float", "max-float", "min-fraction"],
    )
    def test_bounds_on_an_int_axis_are_not_rounded(
        self, query: dict[str, Any], ids: list[str]
    ) -> None:
        assert _ids(self._ticks().get_events(**query)) == ids

    @pytest.mark.parametrize("at", [11.5, Fraction(23, 2)], ids=["float", "fraction"])
    def test_positions_on_an_int_axis_are_not_rounded(self, at: Any) -> None:
        assert _found(self._ticks().get_events_at(at)) == {"ticks": ["y"]}


class TestFloatKeysAcrossChildren:
    """A float key meets each child's exact boundaries on the queried axis."""

    @staticmethod
    def _adjoining() -> Timeline:
        """``A`` [1/10, 3/10) beside ``B`` [3/10, 1), ``B1`` [2/5, 1/2) in ``B``."""
        parent = _fraction_timeline("P", Fraction(1))
        a = _fraction_timeline("A", Fraction(1, 5))
        a.add_events(
            [
                {
                    "id": "a_last",
                    "event_type": "Note",
                    "start": Fraction(1, 10),
                    "end": Fraction(1, 5),
                }
            ]
        )
        b = _fraction_timeline("B", Fraction(7, 10))
        b.add_events(
            [
                {
                    "id": "b_first",
                    "event_type": "Note",
                    "start": 0,
                    "end": Fraction(1, 10),
                }
            ]
        )
        b1 = _fraction_timeline("B1", Fraction(1, 10))
        b1.add_events([{"id": "b1_start", "event_type": "Beat", "instant": 0}])
        b.add_child(b1, offset=Fraction(1, 10))
        parent.add_child(a, offset=Fraction(1, 10))
        parent.add_child(b, offset=Fraction(3, 10))
        return parent

    @staticmethod
    def _last_child() -> Timeline:
        """``C`` [0, 1/10), the only child, with a Beat at its end."""
        parent = _fraction_timeline("P", Fraction(1))
        c = _fraction_timeline("C", Fraction(1, 10))
        c.add_events(
            [
                {
                    "id": "c_note",
                    "event_type": "Note",
                    "start": 0,
                    "end": Fraction(1, 10),
                },
                {"id": "c_end", "event_type": "Beat", "instant": Fraction(1, 10)},
            ]
        )
        parent.add_child(c, offset=0)
        return parent

    @pytest.mark.parametrize(
        ("at", "found"),
        [
            (0.3, {"B": ["b_first"]}),
            (Fraction(3, 10), {"B": ["b_first"]}),
            (0.4, {"B1": ["b1_start"]}),
            (Fraction(2, 5), {"B1": ["b1_start"]}),
        ],
        ids=[
            "boundary-float",
            "boundary-fraction",
            "grandchild-float",
            "grandchild-fraction",
        ],
    )
    def test_a_boundary_of_two_siblings_is_counted_once(
        self, at: Any, found: dict[str, list[str]]
    ) -> None:
        assert _found(self._adjoining().get_events_at(at)) == found

    @pytest.mark.parametrize(
        ("query", "ids"),
        [
            ({"min_coord": 0.3}, ["b_first", "b1_start"]),
            ({"max_coord": 0.3}, ["a_last"]),
        ],
        ids=["min", "max"],
    )
    def test_bounds_place_the_boundary_as_the_position_does(
        self, query: dict[str, Any], ids: list[str]
    ) -> None:
        assert _ids(self._adjoining().get_events(**query)) == ids

    @pytest.mark.parametrize("at", [0.1, Fraction(1, 10)], ids=["float", "fraction"])
    def test_the_end_of_a_last_child_is_held(self, at: Any) -> None:
        assert _found(self._last_child().get_events_at(at)) == {"C": ["c_end"]}

    @pytest.mark.parametrize(
        ("query", "ids"),
        [({"min_coord": 0.1}, ["c_end"]), ({"max_coord": 0.1}, ["c_note"])],
        ids=["min", "max"],
    )
    def test_bounds_at_the_end_of_a_last_child(
        self, query: dict[str, Any], ids: list[str]
    ) -> None:
        assert _ids(self._last_child().get_events(**query)) == ids

    def test_a_child_is_reached_with_the_position_as_written(self) -> None:
        parent = _fraction_timeline("parent", Fraction(1))
        child = _fraction_timeline("child", Fraction(1, 2))
        child.add_events([{"id": "downbeat", "event_type": "Beat", "instant": 0}])
        parent.add_child(child, offset=Fraction(1, 10))

        assert _found(parent.get_events_at(0.1)) == {"child": ["downbeat"]}


@pytest.mark.parametrize("number_type", ["float", "fraction", "int"])
class TestNonFinitePositions:
    """An infinite float is an ordinary double; NaN names no position."""

    @pytest.mark.parametrize(
        ("query", "ids"),
        [
            ({"max_coord": math.inf}, ["one", "two"]),
            ({"min_coord": -math.inf}, ["one", "two"]),
            ({"min_coord": math.inf}, []),
            ({"max_coord": -math.inf}, []),
        ],
        ids=["max-inf", "min-minus-inf", "min-inf", "max-minus-inf"],
    )
    def test_infinite_bounds(
        self, number_type: str, query: dict[str, Any], ids: list[str]
    ) -> None:
        assert _ids(_two_notes(number_type).get_events(**query)) == ids

    def test_an_infinite_position_finds_nothing(self, number_type: str) -> None:
        assert _two_notes(number_type).get_events_at(math.inf) == {}

    @pytest.mark.parametrize(
        "query",
        [{"min_coord": math.nan}, {"max_coord": math.nan}],
        ids=["min", "max"],
    )
    def test_a_nan_bound_is_refused(
        self, number_type: str, query: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError) as refusal:
            _two_notes(number_type).get_events(**query)

        assert str(refusal.value) == "NaN names no position: it compares with no cell"

    def test_a_nan_position_is_refused(self, number_type: str) -> None:
        with pytest.raises(ValueError) as refusal:
            _two_notes(number_type).get_events_at(math.nan)

        assert str(refusal.value) == "NaN names no position: it compares with no cell"


class TestGetEventsIsACopy:
    """``get_events`` never hands out the timeline's own table."""

    def test_joining_onto_the_result_leaves_the_timeline(self) -> None:
        timeline = _fraction_timeline("short", Fraction(4))
        timeline.add_events([{"id": "n", "event_type": "Note", "start": 0, "end": 1}])
        late = type(timeline.events).from_dicts(
            [{"id": "late", "event_type": "Note", "start": 10, "end": 11}],
            TimeUnit.quarters,
            NumberType.fraction,
        )

        result = timeline.get_events()
        assert result is not timeline.events
        joined = result.concat(late)

        assert _ids(joined) == ["n", "late"]
        assert _ids(result) == ["n"]
        assert _ids(timeline.get_events()) == ["n"]
        assert timeline.length.value == Fraction(4)
