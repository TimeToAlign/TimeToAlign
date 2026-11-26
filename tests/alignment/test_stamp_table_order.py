"""MatchStamp table order follows registration (see README, "MatchStamp table order")."""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.maps import LinearMap, SecondsToSamples
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)


@pytest.fixture
def bundle() -> AlignmentBundle:
    """Three timelines registered in the reverse of their id order."""
    score = ContinuousLogicalTimeline(length=16, uid="z_score")
    score.add_conversion_map(
        LinearMap(scalar=2, source_unit="quarters", target_unit="ticks", uid="to ticks")
    )
    perf = ContinuousPhysicalTimeline(length=8.0, uid="m_perf")
    perf.add_conversion_map(SecondsToSamples(sample_rate=10, uid="to samples"))
    third = ContinuousLogicalTimeline(length=16, uid="a_third")
    third.add_conversion_map(
        LinearMap(
            scalar=Fraction(1, 4),
            source_unit="quarters",
            target_unit="whole_notes",
            uid="to whole notes",
        )
    )
    bundle = AlignmentBundle()
    for timeline in (score, perf, third):
        bundle.add_timeline(timeline, as_group=timeline.id)
    bundle.create_match_claims(
        [
            ({"start": 8}, "z_score", {"start": 4.0}, "m_perf"),
            ({"start": 0}, "z_score", {"start": 0.0}, "m_perf"),
            ({"start": 2.0}, "m_perf", {"start": 10}, "a_third"),
        ]
    )
    return bundle


def test_columns_follow_registration_each_followed_by_its_derived_columns(
    bundle: AlignmentBundle,
) -> None:
    table = bundle.get_matchstamp_table(from_graph=True, conversion_maps=True)
    assert table.column_names == [
        "z_score",
        "ticks",
        "m_perf",
        "samples",
        "a_third",
        "whole_notes",
    ]


def test_collapsed_rows_by_first_registered_timeline_then_value(
    bundle: AlignmentBundle,
) -> None:
    frame = bundle.get_matchstamp_table(
        from_graph=True, conversion_maps=True, format="dataframe", units=False
    )
    assert frame.to_dict("list") == {
        "z_score": [Fraction(0), Fraction(8), None],
        "ticks": [0, 16, None],
        "m_perf": [0.0, 4.0, 2.0],
        "samples": [0, 40, 20],
        "a_third": [None, None, Fraction(10)],
        "whole_notes": [None, None, Fraction(5, 2)],
    }
