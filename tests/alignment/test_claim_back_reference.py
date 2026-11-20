"""A claim's bundle is never copied with it.

Validation strategy, fixture and exact values: ``tests/alignment/README.md``
(the "A Claim's Bundle Is Never Copied With It" section).
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import Any

import pytest

from timetoalign.alignment import AlignmentAnchor, AlignmentBundle, MatchClaim
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, NumberType, TimeUnit
from timetoalign.timelines import Timeline

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


def _claim(score: tuple[float, float], perf: tuple[float, float]) -> MatchClaim:
    """One interval claim ``score [a, b]`` -> ``perf [c, d]``."""

    def anchor(index: int) -> AlignmentAnchor:
        return AlignmentAnchor(
            timeline_a_id="score",
            coordinate_a=_coordinate(score[index]),
            timeline_b_id="perf",
            coordinate_b=_coordinate(perf[index]),
        )

    return MatchClaim(
        timeline_a_id="score",
        timeline_b_id="perf",
        start_anchor=anchor(0),
        end_anchor=anchor(1),
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
    chain = [_claim((i, i + 1), (2 * i, 2 * i + 2)) for i in range(N_CHAIN)]
    held.add_match_claims(chain + [_claim(score, perf) for score, perf in OVERLAPPING])
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


class TestCopyingAClaim:
    """A deep copy of a held claim copies that claim only."""

    def test_deepcopy_copies_one_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        held = bundle.cross_group_claims[0]
        copied = copy.deepcopy(held)
        assert copies == [held]
        assert copied.bundle is None
        assert held.bundle is bundle
        assert copied == held
        assert copied is not held

    def test_model_copy_deep_copies_one_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        held = bundle.cross_group_claims[0]
        copied = held.model_copy(deep=True)
        assert copies == [held]
        assert copied.bundle is None
        assert copied == held


class TestCopyingABundle:
    """A copied bundle's claims name the copied bundle."""

    def test_claims_follow_the_copied_bundle(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        copied = copy.deepcopy(bundle)
        assert len(copies) == N_CHAIN + len(OVERLAPPING)
        assert len(copied.cross_group_claims) == N_CHAIN + len(OVERLAPPING)
        assert all(claim.bundle is copied for claim in copied.cross_group_claims)
        assert copied.cross_group_claims == bundle.cross_group_claims


class TestRetrievalCopiesNoClaim:
    """A stamp holds the bundle's own claims."""

    def test_matchstamp_at_copies_no_claim(
        self, bundle: AlignmentBundle, copies: list[MatchClaim]
    ) -> None:
        stamp = bundle.get_matchstamp_at(10.5, "score")
        assert copies == []
        held = bundle.cross_group_claims
        expected = [held[10], held[300], held[301], held[302]]
        assert len(stamp.interval_claims) == len(expected)
        assert all(
            got is want
            for got, want in zip(stamp.interval_claims, expected, strict=True)
        )
        assert all(claim.bundle is bundle for claim in stamp.interval_claims)
        assert stamp.get_coordinate_for("perf", format="float") == 21.0

    def test_stamp_copies_its_list(self, bundle: AlignmentBundle) -> None:
        claims = [bundle.cross_group_claims[10]]
        stamp = MatchStamp(
            coordinates={"score": _coordinate(10.5)},
            source_id="score",
            interval_claims=claims,
        )
        claims.append(bundle.cross_group_claims[300])
        assert len(stamp.interval_claims) == 1
        assert stamp.interval_claims[0] is bundle.cross_group_claims[10]
