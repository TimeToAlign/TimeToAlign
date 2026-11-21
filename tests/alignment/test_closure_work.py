"""A matchstamp reads only the claims naming each timeline it visits.

Validation strategy, fixture, derivation and exact values:
``tests/alignment/README.md`` (the "A Matchstamp Reads Only the Claims Naming
Each Timeline" section).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim
from timetoalign.alignment.claims import MatchClaimField
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.timelines import Timeline, TimelineGroup

from .helpers import match_claim

Q = TimeUnit.quarters
S = TimeUnit.seconds
N_PARTS = 20
N_TRACKS = 10
PERFORMANCES = (1, 2, 3)


def _timeline(uid: str, length: int, unit: TimeUnit) -> Timeline:
    """One timeline of the fixture."""
    return Timeline(length=length, unit=unit, uid=uid)


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
        match_claim("score", Coordinate(q, Q), f"perf_{n}", Coordinate(float(n * q), S))
        for n in PERFORMANCES
        for q in range(0, 41, 2)
    ]
    claims += [
        match_claim("edition", Coordinate(q, Q), "recording", Coordinate(q + 5.0, S))
        for q in (0, 40)
    ]
    bundle.add_match_claims(claims)
    return bundle


@pytest.fixture
def work(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Count table cuts, claims materialised, MatchLines built, warps attempted."""
    cuts: Counter[str] = Counter()
    rows_cut: Counter[str] = Counter()
    materialized: list[int] = []
    sources: list[str] = []
    transfers: list[str] = []
    connecting = MatchClaimField.connecting
    as_claims = AlignmentBundle._materialized
    transfer = AlignmentBundle._transfer_to_group
    from_claims = MatchLine.from_claims.__func__  # type: ignore[attr-defined]

    def counting_connecting(self: MatchClaimField, timeline_id: str) -> MatchClaimField:
        cut = connecting(self, timeline_id)
        cuts[timeline_id] += 1
        rows_cut[timeline_id] += len(cut)
        return cut

    def counting_materialized(
        self: AlignmentBundle, rows: MatchClaimField
    ) -> list[MatchClaim]:
        claims = as_claims(self, rows)
        materialized.append(len(claims))
        return claims

    def counting_transfer(self: AlignmentBundle, *args: Any, **kwargs: Any) -> Any:
        transfers.append(args[1])
        return transfer(self, *args, **kwargs)

    def counting_from_claims(cls: type[MatchLine], *args: Any, **kwargs: Any) -> Any:
        sources.append(kwargs["source_timeline_id"])
        return from_claims(cls, *args, **kwargs)

    monkeypatch.setattr(MatchClaimField, "connecting", counting_connecting)
    monkeypatch.setattr(AlignmentBundle, "_materialized", counting_materialized)
    monkeypatch.setattr(AlignmentBundle, "_transfer_to_group", counting_transfer)
    monkeypatch.setattr(MatchLine, "from_claims", classmethod(counting_from_claims))
    return {
        "cuts": cuts,
        "rows_cut": rows_cut,
        "materialized": materialized,
        "sources": sources,
        "transfers": transfers,
    }


def test_closure_reads_only_the_claims_naming_each_timeline(
    work: dict[str, Any],
) -> None:
    """57 cuts of 130 rows, 65 claims materialised, two lines, four warps."""
    bundle = _bundle()
    stamp = bundle.get_matchstamp_at(11, "score")

    expected = (
        {"score", "facsimile", "edition", "recording"}
        | {f"part_{k}" for k in range(1, N_PARTS + 1)}
        | {f"perf_{n}" for n in PERFORMANCES}
        | {f"perf_{n}_track_{k}" for n in PERFORMANCES for k in range(1, N_TRACKS + 1)}
    )
    assert len(expected) == 57
    assert work["cuts"] == Counter({timeline_id: 1 for timeline_id in expected})
    assert +work["rows_cut"] == Counter(
        {
            "score": 63,
            "edition": 2,
            "recording": 2,
            "perf_1": 21,
            "perf_2": 21,
            "perf_3": 21,
        }
    )
    assert sum(work["rows_cut"].values()) == 130

    assert work["sources"] == ["score", "edition"]
    assert work["materialized"] == [0, 0, 0, 63, 2] + [0] * 56
    assert work["transfers"] == ["score", "score", "score", "score"]

    assert set(stamp.present_timelines) == expected

    def value(timeline_id: str) -> Any:
        return stamp.get_coordinate_for(timeline_id, format="coordinate").value

    assert [value(f"perf_{n}") for n in PERFORMANCES] == [11.0, 22.0, 33.0]
    assert value("edition") == 11
    assert value("part_7") == 11
    assert value("recording") == 16.0
