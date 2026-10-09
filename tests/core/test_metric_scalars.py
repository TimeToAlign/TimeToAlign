"""Exact scalar validation below the measure-anchored hierarchy.

The validation rationale and specimens are documented in ``README.md`` under
"Metrical and realization scalars". ``BeatPolicy`` and ``Tempo`` remain core
event scalars; reading nodes, exact spans, and attribution remain timeline
structure and must not acquire paired semantic fields.
"""

from __future__ import annotations

import json
from fractions import Fraction

import pytest
from pydantic import ValidationError

import timetoalign.core as core
from timetoalign.core import Address, Beat, BeatPolicy, Duration, Tempo, fields


@pytest.mark.parametrize(
    ("signature", "grouping", "division", "rods", "span"),
    [
        (
            "6/8",
            (3, 3),
            Fraction(1, 2),
            (Fraction(3, 2), Fraction(3, 2)),
            Fraction(3),
        ),
        ("4/4", (1, 1, 1, 1), Fraction(1), (Fraction(1),) * 4, Fraction(4)),
        ("C", (1, 1, 1, 1), Fraction(1), (Fraction(1),) * 4, Fraction(4)),
        ("C|", (1, 1), Fraction(2), (Fraction(2),) * 2, Fraction(4)),
        (
            "3+2+3/8",
            (3, 2, 3),
            Fraction(1, 2),
            (Fraction(3, 2), Fraction(1), Fraction(3, 2)),
            Fraction(4),
        ),
    ],
)
def test_time_signature_policy_is_exact_and_round_trips(
    signature: str,
    grouping: tuple[int, ...],
    division: Fraction,
    rods: tuple[Fraction, ...],
    span: Fraction,
) -> None:
    policy = BeatPolicy.from_time_signature(signature)

    assert policy.grouping == grouping
    assert policy.division == division
    assert policy.rods == rods
    assert policy.n_beats == len(grouping)
    assert policy.span == span
    assert policy.name == signature

    divided = policy.as_divisions()
    assert divided.grouping == (1,) * sum(grouping)
    assert divided.division == division
    assert divided.name == signature
    assert divided.as_divisions() == divided

    payload = json.loads(json.dumps(policy.to_dict()))
    restored = BeatPolicy.from_dict(payload)
    assert restored == policy
    assert restored.to_dict() == payload


def test_policy_rejects_contradictory_size_fields() -> None:
    with pytest.raises(ValidationError, match="disagree"):
        BeatPolicy(
            division=Fraction(1),
            beat_size=Duration(Fraction(1, 8), "w"),
        )

    policy = BeatPolicy(
        division=Fraction(1, 2),
        beat_size=Duration(Fraction(1, 8), "w"),
    )
    assert policy.division == Fraction(1, 2)

    payload = json.loads(json.dumps(policy.to_dict()))
    restored = BeatPolicy.from_dict(payload)
    assert restored == policy
    assert restored.to_dict() == payload
    assert restored.division == Fraction(1, 2)
    assert restored.beat_size == Duration(Fraction(1, 8), "w")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (138, Fraction(138)),
        (159.96, Fraction(159.96)),
        ("159.96", Fraction("159.96")),
        (Fraction(3999, 25), Fraction(3999, 25)),
    ],
)
def test_tempo_rate_inputs_remain_exact(
    source: int | float | str | Fraction, expected: Fraction
) -> None:
    assert Tempo(bpm=source).bpm == expected


@pytest.mark.parametrize(
    ("field", "value"), [("bpm", 0), ("bpm", -1), ("to_bpm", 0), ("to_bpm", -1)]
)
def test_tempo_rates_must_be_positive(field: str, value: int) -> None:
    kwargs = {"bpm": 60, field: value}
    with pytest.raises(ValidationError, match="positive"):
        Tempo(**kwargs)


def test_tempo_ramp_wire_is_an_exact_fixpoint() -> None:
    tempo = Tempo(
        bpm=Fraction(60),
        beat=Duration(Fraction(3, 8), "w"),
        to_bpm=Fraction(127, 2),
        mean_at=Fraction(1, 3),
        text="Andantino",
    )

    payload = json.loads(json.dumps(tempo.to_dict()))
    restored = Tempo.from_dict(payload)

    assert restored == tempo
    assert restored.to_dict() == payload
    assert payload["mean_at"]["numerator"] == 1
    assert payload["mean_at"]["denominator"] == 3


@pytest.mark.parametrize("mean_at", [Fraction(0), Fraction(1), Fraction(5)])
def test_tempo_mean_must_lie_inside_ramp(mean_at: Fraction) -> None:
    with pytest.raises(ValidationError, match="0 < mean_at < 1"):
        Tempo(bpm=60, to_bpm=90, mean_at=mean_at)


def test_tempo_mean_requires_target_rate() -> None:
    with pytest.raises(ValidationError, match="requires to_bpm"):
        Tempo(bpm=60, mean_at=Fraction(1, 2))


def test_tempo_rejects_exact_rate_outside_wire_limit_at_construction() -> None:
    limit = 2**63 - 1
    accepted = Tempo(bpm=Fraction(1, limit))
    payload = json.loads(json.dumps(accepted.to_dict()))
    assert Tempo.from_dict(payload) == accepted

    with pytest.raises(ValidationError, match=rf"{limit + 1}.*int64.*{limit}"):
        Tempo(bpm=Fraction(1, limit + 1))


def test_forest_values_are_not_core_event_scalars() -> None:
    moved_names = {"Expansion", "MetricNode", "Reading", "TempoEntry", "TempoMap"}
    moved_fields = {f"{name}Field" for name in moved_names}
    assert moved_names.isdisjoint(core.__all__)
    assert moved_fields.isdisjoint(core.__all__)
    assert all(not hasattr(core, name) for name in moved_names | moved_fields)

    registry = fields._PAIRED_SEMANTIC_FIELDS
    assert registry, "the paired-field registry is empty, so absence proves nothing"
    assert moved_names.isdisjoint({scalar.__name__ for scalar in registry})
    assert moved_fields.isdisjoint({paired.__name__ for paired in registry.values()})


def test_negative_beat_level_is_valid_but_index_remains_one_based() -> None:
    assert Beat(index=1, level=-1).level == -1
    with pytest.raises(ValidationError, match="1-based"):
        Beat(index=0)


def test_address_has_no_skeleton_identifier() -> None:
    assert "skeleton_id" not in Address.model_fields
