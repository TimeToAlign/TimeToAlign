"""Regression tests for coordinate resolution through timeline entry points."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

import pytest

from timetoalign import Coordinate, Timeline, TimeUnit
from timetoalign.maps import LinearMap
from timetoalign.timelines import SegmentLine
from timetoalign.timelines.types import ContinuousLogicalTimeline

CoordinateOperation = Callable[[Timeline, Coordinate], object]


def _get_events(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.get_events(min_coord=coord)


def _get_events_at(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.get_events_at(coord)


def _get_regions_at(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.get_regions_at(coord)


def _get_children_at(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.get_children_at(coord)


def _set_length(timeline: Timeline, coord: Coordinate) -> object:
    timeline.length = coord
    return timeline.length


def _create_region(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.create_region(coord, 10, id="section", name="section")


def _get_segment_line_slice(timeline: Timeline, coord: Coordinate) -> object:
    return timeline.get_slice(coord, 10)


@pytest.mark.parametrize(
    "operation",
    [
        _get_events,
        _get_events_at,
        _get_regions_at,
        _get_children_at,
        _set_length,
        _create_region,
        _get_segment_line_slice,
    ],
)
def test_foreign_coordinate_without_map_raises_from_public_entry_point(
    operation: CoordinateOperation,
) -> None:
    """Public timeline entry points propagate a missing C-Map error."""
    timeline = SegmentLine(length=20, unit=TimeUnit.seconds, uid="audio")
    coordinate = Coordinate(2000, TimeUnit.milliseconds)

    with pytest.raises(ValueError) as exc_info:
        operation(timeline, coordinate)

    assert str(exc_info.value) == (
        "No C-Map available to convert coordinate from unit 'milliseconds' to "
        "'seconds' on timeline 'audio'"
    )


def _exact_score() -> ContinuousLogicalTimeline:
    """A fraction axis whose C-Maps are both declared to write fractions."""
    score = ContinuousLogicalTimeline(length=Fraction(8), uid="score")
    for uid, unit, scalar in (("q2s", "seconds", Fraction(1, 2)), ("q2n", "number", 2)):
        score.add_conversion_map(
            LinearMap(
                source_unit="quarters",
                target_unit=unit,
                scalar=scalar,
                output_number_type="fraction",
                uid=uid,
            )
        )
    return score


@pytest.mark.parametrize("getter", ["get_coordinate_at", "get_timestamp_at"])
@pytest.mark.parametrize(
    ("coordinate", "names"),
    [
        (Coordinate(1.5, TimeUnit.seconds), "C-Map 'q2s' ('quarters' to 'seconds')"),
        (Coordinate(3.0, TimeUnit.number), "C-Map 'q2n' ('quarters' to 'number')"),
    ],
    ids=["seconds", "number"],
)
def test_a_float_through_an_exact_map_is_a_query_error(
    getter: str, coordinate: Coordinate, names: str
) -> None:
    with pytest.raises(ValueError) as refused:
        getattr(_exact_score(), getter)(coordinate)
    assert str(refused.value) == (
        f"{names} is declared to write fractions and computes exactly, so it "
        f"cannot resolve the float {coordinate.value!r} {coordinate.unit} on "
        "timeline 'score': pass an int or a Fraction."
    )


@pytest.mark.parametrize(
    ("coordinate", "quarters", "seconds", "number"),
    [
        (
            Coordinate(Fraction(3, 2), TimeUnit.seconds),
            Fraction(3),
            Fraction(3, 2),
            Fraction(6),
        ),
        (
            Coordinate(3, TimeUnit.number, number_type="int"),
            Fraction(3, 2),
            Fraction(3, 4),
            Fraction(3),
        ),
    ],
    ids=["fraction", "int"],
)
def test_an_exact_value_through_an_exact_map_answers_exactly(
    coordinate: Coordinate, quarters: Fraction, seconds: Fraction, number: Fraction
) -> None:
    score = _exact_score()
    resolved = score.get_coordinate_at(coordinate, format="coordinate")
    assert type(resolved.value) is Fraction
    assert resolved == Coordinate(quarters, TimeUnit.quarters)
    stamp = score.get_timestamp_at(coordinate)
    assert stamp.axis.value == quarters
    for unit, expected in ((TimeUnit.seconds, seconds), (TimeUnit.number, number)):
        value = stamp.get_unit(unit, format="coordinate").value
        assert type(value) is Fraction
        assert value == expected
