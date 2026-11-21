"""Note matching: sounding onsets of an ms3 notes table, and claims from matches.

Validation logic: ``tests/alignment/README.md``, "Note Onsets for Matching".
"""

from __future__ import annotations

from fractions import Fraction

import pandas as pd
import pytest

from timetoalign.alignment.matching import (
    match_notes_by_attributes,
    prepare_abc_notes_for_matching,
)
from timetoalign.core import Coordinate, Provenance, TimeUnit
from timetoalign.core.time import struct_to_rational
from timetoalign.maps import SamplesToSeconds
from timetoalign.timelines import ContinuousLogicalTimeline, DiscretePhysicalTimeline

IDS = ["a", "b", "c", "d", "e", "f", "g"]
# untied, a chain 1 0 0 -1, a two-note tie 1 -1
TIES = [None, 1, 0, 0, -1, 1, -1]
COORDS = ["0", "1", "2", "3", "4", "707/2", "355"]


def _frame(tie_dtype: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": IDS,
            "name": ["C4", "D4", "D4", "D4", "D4", "E4", "E4"],
            "staff": [1] * 7,
            "tied": pd.array(TIES, dtype=tie_dtype),
            "quarterbeats_playthrough": COORDS,
        }
    )


@pytest.mark.parametrize("tie_dtype", ["float64", "Int64"])
def test_untied_notes_and_tie_starts_are_the_onsets(tie_dtype):
    prepared = prepare_abc_notes_for_matching(_frame(tie_dtype))

    assert prepared["id"].tolist() == ["a", "b", "f"]
    assert prepared.index.tolist() == [0, 1, 2]
    assert prepared["quarterbeats_playthrough"].tolist() == [0.0, 1.0, 353.5]
    assert prepared["pitch"].tolist() == ["C4", "D4", "E4"]
    assert "name" not in prepared.columns


@pytest.mark.parametrize("blank", ["", " "], ids=["empty", "whitespace"])
def test_blank_text_tie_cells_are_no_tie(blank):
    frame = _frame("float64")
    frame["tied"] = pd.array([blank, "1", "0", "0", "-1", "1", "-1"], dtype="str")

    prepared = prepare_abc_notes_for_matching(frame)

    assert prepared["id"].tolist() == ["a", "b", "f"]
    assert prepared.index.tolist() == [0, 1, 2]


def test_a_tie_value_outside_ms3s_is_refused():
    frame = _frame("float64").iloc[[1, 2]].copy()
    frame["tied"] = [1.0, 2.0]

    with pytest.raises(ValueError) as raised:
        prepare_abc_notes_for_matching(frame)

    assert str(raised.value) == (
        "Column 'tied' holds tie values other than 1, 0, -1: [2.0]"
    )


# region Claims from matched notes

SAMPLE_RATE = 44100


def _performance() -> DiscretePhysicalTimeline:
    timeline = DiscretePhysicalTimeline(
        length=3 * SAMPLE_RATE, unit=TimeUnit.samples, uid="dpt_1"
    )
    timeline.add_conversion_map(SamplesToSeconds(sample_rate=SAMPLE_RATE))
    timeline.add_events(
        [
            {"id": f"p{n}", "event_type": "Note", "instant": n * SAMPLE_RATE // 2}
            for n in (1, 2, 3)
        ]
    )
    return timeline


def _score() -> ContinuousLogicalTimeline:
    timeline = ContinuousLogicalTimeline(length=4, unit=TimeUnit.quarters, uid="clt_1")
    timeline.add_events(
        [
            {"id": f"n{n}", "event_type": "Note", "instant": Fraction(n, 2)}
            for n in (1, 2, 3)
        ]
    )
    return timeline


PERFORMANCE_ROWS = pd.DataFrame(
    {"id": ["p1", "p2", "p3"], "pitch": ["C4", "D4", "E4"], "start": [0.5, 1.0, 1.5]}
)
SCORE_ROWS = pd.DataFrame(
    {
        "id": ["n1", "n2", "n3"],
        "pitch": ["C4", "E4", "D4"],
        "start": ["1/2", "1", "3/2"],
    }
)


def _match(performance, score, **kwargs):
    return match_notes_by_attributes(
        PERFORMANCE_ROWS,
        SCORE_ROWS,
        ["pitch"],
        source_timeline=performance,
        target_timeline=score,
        source_unit=TimeUnit.seconds,
        **kwargs,
    )


def _named(claims) -> list[tuple]:
    return [
        (c.timeline_a_id, c.event_a_id, c.timeline_b_id, c.event_b_id) for c in claims
    ]


def _anchors(claims) -> list[tuple]:
    return [c.start_anchor.coordinates for c in claims]


def _projections(timeline) -> list[tuple]:
    rows = timeline.events.filter(event_type="Projection").table.to_pylist()
    return [
        (row["id"], struct_to_rational(row["start"]), row["projection_of"])
        for row in rows
    ]


SAMPLES = [22050, 44100, 66150]
QUARTERS = [Fraction(1, 2), Fraction(3, 2), Fraction(1)]


def test_claims_name_both_events():
    performance, score = _timelines()
    claims = _match(
        performance, score, source_id_column="id", target_id_column="id"
    ).match_claims

    assert _named(claims) == [
        ("dpt_1", "p1", "clt_1", "n1"),
        ("dpt_1", "p2", "clt_1", "n3"),
        ("dpt_1", "p3", "clt_1", "n2"),
    ]
    assert _anchors(claims) == [
        (Coordinate(sample, TimeUnit.samples), Coordinate(quarter, TimeUnit.quarters))
        for sample, quarter in zip(SAMPLES, QUARTERS, strict=True)
    ]
    assert type(claims[1].start_anchor.coordinate_b.value) is Fraction
    assert (len(performance.events), len(score.events)) == (3, 3)
    assert {claim.provenance for claim in claims} == {
        Provenance(method="inferred", note="greedy sequential matching by pitch")
    }


def test_a_side_without_ids_gets_projections_of_the_other():
    performance, score = _timelines()
    claims = _match(performance, score, source_id_column="id").match_claims

    assert _named(claims) == [
        ("dpt_1", f"p{n}", "clt_1", f"projection_{k}")
        for k, n in enumerate((1, 2, 3), start=1)
    ]
    assert _projections(score) == [
        (f"projection_{k}", quarter, {"timeline_id": "dpt_1", "event_id": f"p{k}"})
        for k, quarter in enumerate(QUARTERS, start=1)
    ]
    assert len(performance.events) == 3


def test_positions_only_give_projections_of_no_event():
    performance, score = _timelines()
    claims = _match(performance, score).match_claims

    assert _named(claims) == [
        ("dpt_1", f"projection_{k}", "clt_1", f"projection_{k}") for k in (1, 2, 3)
    ]
    assert _projections(performance) == [
        (f"projection_{k}", sample, None) for k, sample in enumerate(SAMPLES, start=1)
    ]
    assert _projections(score) == [
        (f"projection_{k}", quarter, None)
        for k, quarter in enumerate(QUARTERS, start=1)
    ]
    assert _anchors(claims) == [
        (Coordinate(sample, TimeUnit.samples), Coordinate(quarter, TimeUnit.quarters))
        for sample, quarter in zip(SAMPLES, QUARTERS, strict=True)
    ]


def test_an_id_the_timeline_does_not_hold_is_refused():
    performance, score = _timelines()
    rows = PERFORMANCE_ROWS.assign(id=["p1", "p9", "p3"])

    with pytest.raises(KeyError) as raised:
        match_notes_by_attributes(
            rows,
            SCORE_ROWS,
            ["pitch"],
            source_timeline=performance,
            target_timeline=score,
            source_id_column="id",
            source_unit=TimeUnit.seconds,
        )

    assert raised.value.args[0] == (
        "Pair 1: no event with id 'p9' on timeline 'dpt_1' or its children "
        "(3 events of its own)"
    )
    assert (len(performance.events), len(score.events)) == (3, 3)


def test_matching_again_adds_nothing():
    performance, score = _timelines()
    first = _match(performance, score, source_id_column="id").match_claims
    again = _match(performance, score, source_id_column="id").match_claims
    assert _named(again) == _named(first)
    assert (len(performance.events), len(score.events)) == (3, 6)

    performance, score = _timelines()
    first = _match(performance, score).match_claims
    again = _match(performance, score).match_claims
    assert (
        _named(again)
        == _named(first)
        == [("dpt_1", f"projection_{k}", "clt_1", f"projection_{k}") for k in (1, 2, 3)]
    )
    assert (len(performance.events), len(score.events)) == (6, 6)


def test_claims_need_both_timelines():
    with pytest.raises(ValueError, match="pass source_timeline and target_timeline"):
        match_notes_by_attributes(PERFORMANCE_ROWS, SCORE_ROWS, ["pitch"])

    result = match_notes_by_attributes(
        PERFORMANCE_ROWS, SCORE_ROWS, ["pitch"], create_claims=False
    )
    assert result.match_claims == []
    assert result.matched["source_id"].tolist() == ["p1", "p2", "p3"]
    assert result.matched["target_id"].tolist() == ["n1", "n3", "n2"]


def _timelines() -> tuple[DiscretePhysicalTimeline, ContinuousLogicalTimeline]:
    """A fresh performance and score timeline (each test changes its own)."""
    return _performance(), _score()


# endregion
