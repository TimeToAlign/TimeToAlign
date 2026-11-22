"""Regression tests for exact coordinate propagation through timeline engines."""

from __future__ import annotations

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
        assert _local_starts(child) == _expected_part(["n1", "n2"], step)

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


@pytest.mark.parametrize(
    ("query", "ids"),
    [
        ({"min_coord": 0.1}, ["third"]),
        ({"max_coord": 0.1}, ["tenth"]),
        ({"min_coord": Fraction(1, 10)}, ["tenth", "third"]),
    ],
    ids=["min-float", "max-float", "min-fraction"],
)
def test_a_float_bound_on_a_fraction_axis_is_its_double(
    query: dict[str, Any], ids: list[str]
) -> None:
    """A float bound is the exact value of its double, not a nearby ratio."""
    assert _ids(_tenth_and_third().get_events(**query)) == ids


@pytest.mark.parametrize(
    ("at", "ids"),
    [(0.1, []), (1 / 3, []), (Fraction(1, 10), ["tenth"])],
    ids=["tenth-float", "third-float", "tenth-fraction"],
)
def test_a_float_query_on_a_fraction_axis_is_its_double(
    at: Any, ids: list[str]
) -> None:
    """``get_events_at`` finds an event at the double's exact value only."""
    found = _tenth_and_third().get_events_at(at)

    assert {key: _ids(rows) for key, rows in found.items()} == (
        {"score": ids} if ids else {}
    )
