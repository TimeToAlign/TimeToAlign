"""A claim's bundle is never copied with it.

Validation strategy, fixture and exact values: ``tests/alignment/README.md``
(the "A Claim's Bundle Is Never Copied With It" section).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.timelines import Timeline

from .helpers import match_claim

FM = TimeUnit.floating_measures
N_CHAIN = 300
OVERLAPPING = [
    ((10, 12), (20, 24)),
    ((10.25, 10.75), (20.5, 21.5)),
    ((9, 11), (18, 22)),
]


def _coordinate(value: float) -> Coordinate:
    """One float-canonical floating-measure coordinate."""
    return Coordinate(value, FM, number_type=NumberType.float)


def _claim(
    claim_id: str, score: tuple[float, float], perf: tuple[float, float]
) -> MatchClaim:
    """One interval claim *claim_id*, ``score [a, b]`` -> ``perf [c, d]``."""
    return match_claim(
        "score",
        _coordinate(score[0]),
        "perf",
        _coordinate(perf[0]),
        end_a=_coordinate(score[1]),
        end_b=_coordinate(perf[1]),
        id=claim_id,
    )


@pytest.fixture
def bundle() -> AlignmentBundle:
    """Two standalone timelines and 303 interval claims."""
    held = AlignmentBundle(id="back-reference")
    for timeline_id in ("score", "perf"):
        held.add_timeline(
            Timeline(
                length=1200, uid=timeline_id, unit=FM, number_type=NumberType.float
            )
        )
    chain = [
        _claim(f"chain_{i}", (i, i + 1), (2 * i, 2 * i + 2)) for i in range(N_CHAIN)
    ]
    overlapping = [
        _claim(f"overlap_{k}", score, perf)
        for k, (score, perf) in enumerate(OVERLAPPING, start=1)
    ]
    held.add_match_claims(chain + overlapping)
    return held


@pytest.fixture
def copies(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[MatchClaim]]:
    """Record every claim deep-copied while the test runs."""
    copied: list[MatchClaim] = []
    original = MatchClaim.__deepcopy__

    def counting(self: MatchClaim, memo: dict[int, Any] | None = None) -> MatchClaim:
        copied.append(self)
        return original(self, memo)

    monkeypatch.setattr(MatchClaim, "__deepcopy__", counting)
    yield copied


class TestReadingAClaim:
    """A claim read from a bundle records that bundle."""

    def test_claims_read_record_the_bundle(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        claims = bundle.get_match_claims()
        assert copies == []
        assert len(claims) == N_CHAIN + len(OVERLAPPING)
        assert [claim.id for claim in claims[:2]] == ["chain_0", "chain_1"]
        assert claims[-1].id == "overlap_3"
        assert all(claim.bundle is bundle for claim in claims)


class TestCopyingAClaim:
    """A deep copy of a claim read from a bundle copies that claim only."""

    def test_deepcopy_copies_one_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        held = bundle.get_match_claims()[0]
        copied = copy.deepcopy(held)
        assert copies == [held]
        assert copied.bundle is None
        assert held.bundle is bundle
        assert copied == held
        assert copied is not held

    def test_model_copy_deep_copies_one_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        held = bundle.get_match_claims()[0]
        copied = held.model_copy(deep=True)
        assert copies == [held]
        assert copied.bundle is None
        assert held.bundle is bundle
        assert copied == held


class TestCopyingABundle:
    """A copied bundle copies its table; its claims record the copy."""

    def test_claims_follow_the_copied_bundle(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        copied = copy.deepcopy(bundle)
        assert copies == []
        assert copied.match_claim_table is not bundle.match_claim_table
        assert copied.match_claim_table.table.equals(bundle.match_claim_table.table)
        assert len(copied.match_claim_table) == N_CHAIN + len(OVERLAPPING)
        claims = copied.get_match_claims()
        assert all(claim.bundle is copied for claim in claims)
        assert all(claim.bundle is bundle for claim in bundle.get_match_claims())
        assert claims == bundle.get_match_claims()


class TestRetrievalCopiesNoClaim:
    """A stamp holds the claims read from the bundle, recording it."""

    def test_matchstamp_at_copies_no_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        stamp = bundle.get_matchstamp_at(10.5, "score")
        assert copies == []
        held = bundle.get_match_claims()
        assert [claim.id for claim in stamp.interval_claims] == [
            "chain_10",
            "overlap_1",
            "overlap_2",
            "overlap_3",
        ]
        assert stamp.interval_claims == [held[10], held[300], held[301], held[302]]
        assert all(claim.bundle is bundle for claim in stamp.interval_claims)
        assert stamp.get_coordinate_for("perf", format="float") == 21.0

    def test_stamp_copies_its_list(self, bundle: AlignmentBundle) -> None:
        held = bundle.get_match_claims()
        claims = [held[10]]
        stamp = MatchStamp(
            coordinates={"score": _coordinate(10.5)},
            source_id="score",
            interval_claims=claims,
        )
        claims.append(held[300])
        assert len(stamp.interval_claims) == 1
        assert stamp.interval_claims[0] is held[10]
