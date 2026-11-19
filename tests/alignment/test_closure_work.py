"""A matchstamp reads only the claims naming each timeline it visits.

Validation strategy, fixture, derivation and exact values:
``tests/alignment/README.md`` (the "A Matchstamp Reads Only the Claims Naming
Each Timeline" section).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from timetoalign.alignment import AlignmentAnchor, AlignmentBundle, MatchClaim
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.timelines import Timeline, TimelineGroup

Q = TimeUnit.quarters
S = TimeUnit.seconds
N_PARTS = 20
N_TRACKS = 10
PERFORMANCES = (1, 2, 3)


def _timeline(uid: str, length: int, unit: TimeUnit) -> Timeline:
    """One timeline of the fixture."""
    return Timeline(length=length, unit=unit, uid=uid)


def _claim(a_id: str, a: Coordinate, b_id: str, b: Coordinate) -> MatchClaim:
    """One instant claim between two timelines."""
    return MatchClaim(
        timeline_a_id=a_id,
        timeline_b_id=b_id,
        start_anchor=AlignmentAnchor(
            timeline_a_id=a_id, coordinate_a=a, timeline_b_id=b_id, coordinate_b=b
        ),
    )


def _bundle() -> AlignmentBundle:
    """Build the score group, three performances and the recording."""
    score = _timeline("score", 40, Q)
    for k in range(1, N_PARTS + 1):
        score.add_child(_timeline(f"part_{k}", 40, Q), offset=0)
    bundle = AlignmentBundle(id="closure-work")
    bundle.add_group(
        TimelineGroup(
            id="score_group",
            timelines=[
                score,
                _timeline("facsimile", 40, Q),
                _timeline("edition", 40, Q),
            ],
        )
    )
    for n in PERFORMANCES:
        perf = _timeline(f"perf_{n}", 120, S)
        for k in range(1, N_TRACKS + 1):
            perf.add_child(_timeline(f"perf_{n}_track_{k}", 120, S), offset=0)
        bundle.add_group(TimelineGroup(id=f"perf_{n}_group", timelines=[perf]))
    bundle.add_group(
        TimelineGroup(id="recording_group", timelines=[_timeline("recording", 50, S)])
    )
    claims = [
        _claim("score", Coordinate(q, Q), f"perf_{n}", Coordinate(float(n * q), S))
        for n in PERFORMANCES
        for q in range(0, 41, 2)
    ]
    claims += [
        _claim("edition", Coordinate(q, Q), "recording", Coordinate(q + 5.0, S))
        for q in (0, 40)
    ]
    bundle.add_match_claims(claims)
    return bundle


@pytest.fixture
def work(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Count claims expressed, MatchLines built and warps attempted."""
    counts: Counter[str] = Counter()
    sources: list[str] = []
    native = AlignmentBundle._claim_in_native_units
    transfer = AlignmentBundle._transfer_to_group
    from_claims = MatchLine.from_claims.__func__  # type: ignore[attr-defined]

    def counting_native(self: AlignmentBundle, claim: MatchClaim) -> MatchClaim:
        counts["expressed"] += 1
        return native(self, claim)

    def counting_transfer(self: AlignmentBundle, *args: Any, **kwargs: Any) -> Any:
        counts["transfers"] += 1
        return transfer(self, *args, **kwargs)

    def counting_from_claims(cls: type[MatchLine], *args: Any, **kwargs: Any) -> Any:
        sources.append(kwargs["source_timeline_id"])
        return from_claims(cls, *args, **kwargs)

    monkeypatch.setattr(AlignmentBundle, "_claim_in_native_units", counting_native)
    monkeypatch.setattr(AlignmentBundle, "_transfer_to_group", counting_transfer)
    monkeypatch.setattr(MatchLine, "from_claims", classmethod(counting_from_claims))
    return {"counts": counts, "sources": sources}


def test_closure_expresses_only_the_claims_naming_each_timeline(
    work: dict[str, Any],
) -> None:
    """386 claims expressed, two MatchLines, four warps (README derivation)."""
    bundle = _bundle()
    stamp = bundle.get_matchstamp_at(11, "score")

    assert work["counts"]["expressed"] == 386
    assert work["sources"] == ["score", "edition"]
    assert work["counts"]["transfers"] == 4

    expected = (
        {"score", "facsimile", "edition", "recording"}
        | {f"part_{k}" for k in range(1, N_PARTS + 1)}
        | {f"perf_{n}" for n in PERFORMANCES}
        | {f"perf_{n}_track_{k}" for n in PERFORMANCES for k in range(1, N_TRACKS + 1)}
    )
    assert len(expected) == 57
    assert set(stamp.present_timelines) == expected

    def value(timeline_id: str) -> Any:
        return stamp.get_coordinate_for(timeline_id, format="coordinate").value

    assert [value(f"perf_{n}") for n in PERFORMANCES] == [11.0, 22.0, 33.0]
    assert value("edition") == 11
    assert value("part_7") == 11
    assert value("recording") == 16.0
