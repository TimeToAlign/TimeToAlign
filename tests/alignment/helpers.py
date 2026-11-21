"""Typed construction helpers shared by alignment tests."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Any

from timetoalign.alignment import AlignmentAnchor, MatchClaim
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, Provenance, TimeUnit


def make_match_stamp(
    *,
    coordinates: Mapping[str, Coordinate | int | float | Fraction],
    source_id: str | None = None,
    coordinate_units: Mapping[str, TimeUnit | str] | None = None,
    **kwargs: Any,
) -> MatchStamp:
    """Construct a MatchStamp with canonical typed coordinate storage.

    Args:
        coordinates: Timeline coordinates, already typed or numeric fixture values.
        source_id: Source timeline ID, defaulting to the first coordinate key.
        coordinate_units: Units for numeric fixture values; unspecified values use
            the dimensionless number unit.
        **kwargs: Remaining MatchStamp constructor fields.

    Returns:
        A MatchStamp whose coordinate entries are plain Coordinate scalars.
    """
    units = coordinate_units or {}
    typed = {
        timeline_id: (
            value
            if type(value) is Coordinate
            else Coordinate(value, TimeUnit(units.get(timeline_id, TimeUnit.number)))
        )
        for timeline_id, value in coordinates.items()
    }
    resolved_source = source_id or next(iter(typed), None)
    if resolved_source is None:
        raise ValueError("A typed test stamp requires at least one coordinate")
    return MatchStamp(coordinates=typed, source_id=resolved_source, **kwargs)


def event_id(timeline_id: str, coordinate: Coordinate) -> str:
    """The id a fixture gives the event at *coordinate* on *timeline_id*.

    A MatchClaim names the events it matches; a fixture claim built from
    positions names its events after them (``score@4``, ``audio@2.0``).
    The ids are names only: no timeline holds such an event unless the
    test adds it.
    """
    return f"{timeline_id}@{coordinate.value}"


def match_claim(
    timeline_a: str,
    coordinate_a: Coordinate,
    timeline_b: str,
    coordinate_b: Coordinate,
    *,
    end_a: Coordinate | None = None,
    end_b: Coordinate | None = None,
    id: str | None = None,  # noqa: A002 - the claim's id
    provenance: Provenance | None = None,
    certainty: float | None = 1.0,
) -> MatchClaim:
    """A synchronous, explicit MatchClaim between two fixture events.

    The claim names an event on each side (:func:`event_id`), its start
    anchor joins *coordinate_a* on *timeline_a* and *coordinate_b* on
    *timeline_b* in that order; with *end_a* and *end_b* it is an interval
    claim whose end anchor joins them. The coordinates are taken as given:
    a bundle refuses them unless each is on its timeline's own axis (unit
    and declared number type).
    """
    end_anchor = None
    if end_a is not None or end_b is not None:
        if end_a is None or end_b is None:
            raise ValueError("An interval fixture claim takes end_a and end_b")
        end_anchor = AlignmentAnchor(
            timeline_a_id=timeline_a,
            coordinate_a=end_a,
            timeline_b_id=timeline_b,
            coordinate_b=end_b,
        )
    return MatchClaim(
        timeline_a_id=timeline_a,
        timeline_b_id=timeline_b,
        event_a_id=event_id(timeline_a, coordinate_a),
        event_b_id=event_id(timeline_b, coordinate_b),
        start_anchor=AlignmentAnchor(
            timeline_a_id=timeline_a,
            coordinate_a=coordinate_a,
            timeline_b_id=timeline_b,
            coordinate_b=coordinate_b,
        ),
        end_anchor=end_anchor,
        id=id,
        provenance=provenance,
        certainty=certainty,
    )
