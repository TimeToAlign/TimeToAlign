"""Regression tests for exact logical coordinate generation."""

from __future__ import annotations

from fractions import Fraction

from timetoalign.core import NumberType, TimeUnit
from timetoalign.timelines import ContinuousLogicalTimeline, Timeline


def _rational_pair(event: dict, field: str) -> Fraction:
    """Return the exact rational represented by a stored coordinate struct."""
    coordinate = event[field]
    return Fraction(coordinate["numerator"], coordinate["denominator"])


def test_generated_metrical_events_store_exact_coordinates() -> None:
    """Beat and measure rows retain their exact quarter-note coordinates.

    Six eighth-note beats fill one 6/8 bar of three quarters. Every beat
    lands on a multiple of ``1/2``, a ratio no float represents exactly,
    so a lane that rounded on the way into storage would be visible in
    the stored numerator/denominator pair.
    """
    grid = ContinuousLogicalTimeline(
        length=Fraction(3, 1),
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
        uid="six_eight",
    )
    beat_length = Fraction(1, 2)
    grid.add_events(
        [
            {
                "id": f"beat_{index}",
                "temporal_type": "interval",
                "event_type": "Beat",
                "start": index * beat_length,
                "end": (index + 1) * beat_length,
            }
            for index in range(6)
        ]
        + [
            {
                "id": "measure_1",
                "temporal_type": "interval",
                "event_type": "Measure",
                "start": Fraction(0),
                "end": Fraction(3),
            }
        ]
    )

    beat_rows = list(grid.get_events(event_type="Beat"))
    measure_rows = list(grid.get_events(event_type="Measure"))

    assert [_rational_pair(event, "start") for event in beat_rows] == [
        Fraction(0),
        Fraction(1, 2),
        Fraction(1),
        Fraction(3, 2),
        Fraction(2),
        Fraction(5, 2),
    ]
    assert _rational_pair(beat_rows[3], "end") == Fraction(2)
    assert [
        (_rational_pair(event, "start"), _rational_pair(event, "end"))
        for event in measure_rows
    ] == [(Fraction(0), Fraction(3))]


def test_from_events_preserves_exact_fraction_length() -> None:
    """Timeline.from_events retains an exact maximum event coordinate."""
    timeline = Timeline.from_events(
        [
            {
                "id": "beat",
                "temporal_type": "instant",
                "event_type": "Beat",
                "instant": Fraction(3, 2),
            }
        ],
        unit=TimeUnit.quarters,
        number_type=NumberType.fraction,
    )

    assert timeline.length.value == Fraction(3, 2)
