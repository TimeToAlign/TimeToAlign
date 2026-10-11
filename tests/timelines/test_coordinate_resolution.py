"""Regression tests for coordinate resolution through timeline entry points."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from timetoalign import Coordinate, Timeline, TimeUnit
from timetoalign.timelines import SegmentLine

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
