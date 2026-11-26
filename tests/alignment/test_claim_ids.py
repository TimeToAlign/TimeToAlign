"""Claim ids: each addresses one claim of its table or bundle, the same in every process.

Validation strategy, fixture and exact values: ``tests/alignment/README.md``
(the "Claim Ids" section).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim, MatchClaimField
from timetoalign.alignment.graph import MatchGraph
from timetoalign.core.ids import IdMinter, reset_content_minters
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

STORED = [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
NEW = [(2, 1.0), (6, 3.0)]

SCORE = ContinuousLogicalTimeline(length=16, uid="score")
AUDIO = ContinuousPhysicalTimeline(length=8.0, uid="audio")


def _claims(
    pairs: list[tuple[int, float]], ids: list[str] | None = None
) -> list[MatchClaim]:
    """Fresh claims between a ``score`` note and an ``audio`` onset at each pair.

    Without *ids* every claim mints its id from its content.
    """
    return [
        MatchClaim.from_events(
            {"id": f"note_{quarters}", "start": quarters},
            SCORE,
            {"id": f"onset_{seconds}", "start": seconds},
            AUDIO,
            id=None if ids is None else ids[index],
        )
        for index, (quarters, seconds) in enumerate(pairs)
    ]


def _bundle() -> AlignmentBundle:
    """``score`` and ``score_copy`` in one group, ``audio`` in another."""
    bundle = AlignmentBundle()
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=16, uid="score"), as_group="score"
    )
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=16, uid="score_copy"), grouped_with="score"
    )
    bundle.add_timeline(
        ContinuousPhysicalTimeline(length=8.0, uid="audio"), as_group="audio"
    )
    return bundle


def _stored(path: Path) -> MatchClaimField:
    """Store a bundle's four ``STORED`` claims as a Parquet file, read them back."""
    _holding(*_claims(STORED)).match_claim_table.to_parquet(path)
    return MatchClaimField.from_parquet(path)


def _ids(field: MatchClaimField) -> list[str]:
    return field.table.column("id").to_pylist()


def _claim_ids(claims: list[MatchClaim]) -> list[str | None]:
    return [claim.id for claim in claims]


def _columns(
    pairs: list[tuple[int, float]] = STORED, **minting: Any
) -> MatchClaimField:
    """*pairs* as event matches built by ``from_columns``."""
    return MatchClaimField.from_columns(
        timeline_a_ids=["score"] * len(pairs),
        timeline_b_ids=["audio"] * len(pairs),
        event_a_ids=[f"note_{q}" for q, _ in pairs],
        event_b_ids=[f"onset_{s}" for _, s in pairs],
        coordinate_a=[q for q, _ in pairs],
        coordinate_b=[s for _, s in pairs],
        timelines=[SCORE, AUDIO],
        **minting,
    )


def _seeded(seed: int, count: int) -> list[str]:
    """The first *count* claim ids of ``IdMinter(seed)``."""
    return IdMinter(seed).mint_many("claim", count).to_pylist()


# region Field part


def test_claim_mints_its_id_at_construction() -> None:
    (expected,) = _seeded(3, 1)
    seeded = MatchClaim.from_events(
        {"id": "note_0", "start": 0},
        SCORE,
        {"id": "onset_0", "start": 0.0},
        AUDIO,
        seed=3,
    )
    nomatch = MatchClaim.nomatch({"id": "note_0", "start": 0}, SCORE, AUDIO, seed=3)
    assert (seeded.id, nomatch.id) == (expected, expected)

    first, second = _claims(NEW[:1]) + _claims(NEW[:1])
    assert first == second.model_copy(update={"id": first.id})
    assert first.id != second.id
    assert all(re.fullmatch(r"claim_[0-9a-z]{13}", c.id or "") for c in (first, second))
    reset_content_minters()
    assert _claims(NEW[:1])[0].id == first.id

    named = MatchClaim.from_dict(first.to_dict() | {"id": "onset 7"})
    assert named.id == "onset 7"
    with pytest.raises(ValueError, match=re.escape("Invalid id 'a\\tb'")):
        MatchClaim.from_dict(first.to_dict() | {"id": "a\tb"})


def test_field_holds_each_id_once() -> None:
    field = MatchClaimField.from_claims(_claims(STORED[:3], ["s0", "s1", "s2"]))
    assert _ids(field) == ["s0", "s1", "s2"]
    with pytest.raises(ValueError, match=re.escape("claim id 'a' is held by 2 claims")):
        MatchClaimField.from_claims(_claims(NEW, ["a", "a"]))


def test_from_columns_mints_its_ids() -> None:
    assert _ids(_columns(STORED[:3], seed=4)) == _seeded(4, 3)

    minter = IdMinter(4)
    first, second = _columns(STORED[:3], minter=minter), _columns(
        STORED[:3], minter=minter
    )
    expected = _seeded(4, 6)
    assert (_ids(first), _ids(second)) == (expected[:3], expected[3:])

    unseeded = _ids(_columns()) + _ids(_columns())
    assert len(set(unseeded)) == 8
    with pytest.raises(TypeError, match="Give seed or minter, not both"):
        _columns(seed=1, minter=IdMinter(1))


def test_every_reading_constructor_mints_the_missing_ids(tmp_path: Path) -> None:
    table = _columns().table
    table = table.set_column(
        table.column_names.index("id"),
        "id",
        pa.array(["claim_2", None, "x", None], type=pa.string()),
    )
    minted = _seeded(6, 2)
    expected = ["claim_2", minted[0], "x", minted[1]]
    assert _ids(MatchClaimField.from_table(table, seed=6)) == expected

    package = MatchClaimField.from_table(table).package_table
    package = package.set_column(
        package.column_names.index("id"), "id", table.column("id")
    )
    assert _ids(MatchClaimField.from_package_table(package, seed=6)) == expected
    path = tmp_path / "claims.parquet"
    pq.write_table(package, path)
    assert _ids(MatchClaimField.from_parquet(path, seed=6)) == expected

    data = _columns().to_dict()
    for row, claim_id in zip(data["rows"], table.column("id").to_pylist(), strict=True):
        row["id"] = claim_id
    assert _ids(MatchClaimField.from_dict(data, seed=6)) == expected


def test_joining_mints_nothing() -> None:
    stored, later = _columns(), _columns(NEW)
    bundle = _holding().add_match_claim_field(stored).add_match_claim_field(later)
    assert _claim_ids(bundle.get_match_claims()) == _ids(stored) + _ids(later)
    assert None not in _ids(stored) + _ids(later)


def test_two_runs_store_the_same_ids(tmp_path: Path) -> None:
    program = textwrap.dedent("""
        import sys

        from timetoalign.alignment import AlignmentBundle, MatchClaimField
        from timetoalign.timelines.types import (
            ContinuousLogicalTimeline,
            ContinuousPhysicalTimeline,
        )

        pairs = [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
        score = ContinuousLogicalTimeline(length=16, uid="score")
        audio = ContinuousPhysicalTimeline(length=8.0, uid="audio")
        bundle = AlignmentBundle()
        bundle.add_timeline(score)
        bundle.add_timeline(audio)
        bundle.add_match_claim_field(
            MatchClaimField.from_columns(
                timeline_a_ids=["score"] * 4,
                timeline_b_ids=["audio"] * 4,
                event_a_ids=[f"note_{q}" for q, _ in pairs],
                event_b_ids=[f"onset_{s}" for _, s in pairs],
                coordinate_a=[q for q, _ in pairs],
                coordinate_b=[s for _, s in pairs],
                timelines=[score, audio],
            )
        ).match_claim_table.to_parquet(sys.argv[1])
        """)
    first_path = tmp_path / "first"
    first_path.mkdir()
    subprocess.run(
        [sys.executable, "-c", program, str(first_path)],
        check=True,
        env=dict(os.environ),
    )
    first = MatchClaimField.from_parquet(first_path)
    second = MatchClaimField.from_parquet(
        _holding()
        .add_match_claim_field(_columns())
        .match_claim_table.to_parquet(tmp_path / "second.parquet")
    )
    assert _ids(first) == _ids(second)
    assert len(set(_ids(first))) == 4
    assert first.table.equals(second.table, check_metadata=True)


# endregion


# region Bundle part


def _copy_claim(quarters: int) -> MatchClaim:
    """A free claim between a ``score`` note and its ``score_copy`` twin."""
    return MatchClaim.from_events(
        {"id": f"note_{quarters}", "start": quarters},
        SCORE,
        {"id": f"copy_{quarters}", "start": quarters},
        ContinuousLogicalTimeline(length=16, uid="score_copy"),
    )


def _holding(*claims: MatchClaim) -> AlignmentBundle:
    """``score`` and ``audio``, ungrouped, holding *claims*."""
    bundle = AlignmentBundle()
    bundle.add_timeline(SCORE)
    bundle.add_timeline(AUDIO)
    return bundle.add_match_claims(list(claims))


def test_two_claims_hold_distinct_ids() -> None:
    first, second = _claims(STORED[:1])[0], _copy_claim(4)
    ids = [first.id, second.id]
    assert None not in ids and first.id != second.id

    graph = MatchGraph([first, second])
    kept = graph.filter(within={"score", "audio"}).claims
    assert len(kept) == 1 and kept[0] is first
    kept = graph.filter(within={"score", "score_copy"}).claims
    assert len(kept) == 1 and kept[0] is second
    edge_ids = [data["claim_id"] for _, _, data in graph._graph.edges(data=True)]
    assert edge_ids == ids

    assert _ids(MatchClaimField.from_claims([first, second])) == ids

    bundle = _bundle()
    bundle.add_match_claims([first, second])
    assert [first.id, second.id] == ids
    kept = MatchGraph(bundle.get_match_claims()).filter(within={"score", "score_copy"})
    assert _claim_ids(kept.claims) == [second.id]


def test_stored_and_new_claims_are_distinct(tmp_path: Path) -> None:
    stored = _stored(tmp_path / "claims.parquet")
    assert len(set(_ids(stored))) == 4

    rebuilt = _claims(STORED) + _claims(NEW)
    rebuilt_ids = _claim_ids(rebuilt)
    assert len(set(rebuilt_ids)) == 6
    assert set(rebuilt_ids).isdisjoint(_ids(stored))
    bundle = _bundle()
    bundle.add_match_claim_field(stored)
    bundle.add_match_claims(rebuilt)

    assert _claim_ids(rebuilt) == rebuilt_ids
    assert all(claim.bundle is bundle for claim in rebuilt)
    held = bundle.get_match_claims()
    assert held[4:] == rebuilt
    assert all(claim.bundle is bundle for claim in held)
    assert _claim_ids(held) == _ids(stored) + rebuilt_ids


def test_graph_filter_unaffected(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.add_match_claim_field(_stored(tmp_path / "claims.parquet"))
    bundle.add_match_claims(_claims(NEW))
    claims = bundle.get_match_claims()
    held = sorted(_claim_ids(claims))
    assert len(set(held)) == 6

    graph = MatchGraph(claims).extend_to_groups(bundle.groups)
    # The graph's claims are the six it was given; what it infers is its own.
    assert graph.claims == claims
    inferred = graph.inferred_edges
    assert len(inferred) == 6
    assert all(edge.source_claim in claims for edge in inferred)
    for edge in inferred:
        assert edge.anchor.timelines == ("score", "score_copy")
        assert edge.anchor.coordinate_a == edge.source_claim.start_anchor.coordinate_a
    edges = [data for _, _, data in graph._graph.edges(data=True)]
    stated_edges = [data for data in edges if "claim" in data]
    assert [data["claim_id"] for data in stated_edges] == [
        data["claim"].id for data in stated_edges
    ]
    assert len({data["claim_id"] for data in stated_edges}) == len(stated_edges) == 6
    assert [data["inferred"] for data in edges if "claim" not in data] == inferred

    kept = graph.filter(stated_only=True)
    assert sorted(_claim_ids(kept.claims)) == held
    assert kept.inferred_edges == []

    kept = graph.filter(within={"score", "audio"})
    assert sorted(_claim_ids(kept.claims)) == held
    assert kept.inferred_edges == []


def test_held_ids_are_refused_never_reminted() -> None:
    source = _claims(STORED[:1], ["anchor"])[0]
    other = _copy_claim(0)
    field = MatchClaimField.from_claims([source, other])
    assert _ids(field) == ["anchor", other.id]

    new = _claims(NEW)
    bundle = _bundle().add_match_claims(new)
    bundle.add_match_claim_field(field)
    held = [*_claim_ids(new), "anchor", other.id]
    assert _claim_ids(bundle.get_match_claims()) == held
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"claim id '{new[0].id}' is already held by another claim of this bundle"
        ),
    ):
        bundle.add_match_claim_field(_holding(*new).match_claim_table)
    assert _claim_ids(bundle.get_match_claims()) == held


def test_two_runs_merged(tmp_path: Path) -> None:
    program = textwrap.dedent("""
        import sys

        from timetoalign.alignment import AlignmentBundle, MatchClaim
        from timetoalign.timelines.types import (
            ContinuousLogicalTimeline,
            ContinuousPhysicalTimeline,
        )

        score = ContinuousLogicalTimeline(length=16, uid="score")
        audio = ContinuousPhysicalTimeline(length=8.0, uid="audio")
        claims = [
            MatchClaim.from_events(
                {"id": f"note_{quarters}", "start": quarters},
                score,
                {"id": f"onset_{seconds}", "start": seconds},
                audio,
            )
            for quarters, seconds in [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
        ]
        bundle = AlignmentBundle()
        bundle.add_timeline(score)
        bundle.add_timeline(audio)
        bundle.add_match_claims(claims).match_claim_table.to_parquet(sys.argv[1])
        """)
    first_path = tmp_path / "first.parquet"
    subprocess.run(
        [sys.executable, "-c", program, str(first_path)],
        check=True,
        env=dict(os.environ),
    )
    first = MatchClaimField.from_parquet(first_path)
    second = _stored(tmp_path / "second.parquet")
    assert _ids(first) == _ids(second)
    assert len(set(_ids(first))) == 4
    assert first.table.equals(second.table, check_metadata=True)
    shared = _ids(first)[0]

    bundle = _holding().add_match_claim_field(first)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"claim id '{shared}' is already held by another claim of this bundle"
        ),
    ):
        bundle.add_match_claim_field(second)
    assert len(bundle.match_claim_table) == 4

    runs = [_holding().add_match_claim_field(run) for run in (first, second)]
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles(runs)
    assert str(raised.value) == (
        f"MatchClaim id '{shared}' is held by bundle '{runs[0].id}' and by bundle "
        f"'{runs[1].id}' (bundles 0 and 1 of the merge): a MatchClaim is "
        f"addressed by its id alone, so the two merge once one of them states "
        f"its MatchClaims under other ids."
    )
    for run, stored in zip(runs, (first, second), strict=True):
        assert run.match_claim_table.table.equals(stored.table, check_metadata=True)


def test_given_id_refused_atomically() -> None:
    held = _claims(STORED[:1])[0].model_copy(update={"id": "a"})
    bundle = _bundle()
    bundle.add_match_claims([held])
    message = re.escape("claim id 'a' is already held by another claim of this bundle")

    newcomer, given = _claims(NEW)
    newcomer_id = newcomer.id
    given = given.model_copy(update={"id": "a"})
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claims([newcomer, given])
    assert bundle.get_match_claims() == [held]
    assert (newcomer.id, newcomer.bundle) == (newcomer_id, None)
    assert (given.id, given.bundle) == ("a", None)

    field = MatchClaimField.from_claims([given])
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claim_field(field)
    assert bundle.get_match_claims() == [held]
    assert _ids(field) == ["a"]


def test_merge_keeps_ids_and_refuses_a_shared_one() -> None:
    sources = [_holding(*_claims(NEW, ["c1", "c2"])) for _ in range(2)]
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles(sources)
    assert str(raised.value) == (
        f"MatchClaim id 'c1' is held by bundle '{sources[0].id}' and by "
        f"bundle '{sources[1].id}' (bundles 0 and 1 of the merge): a MatchClaim "
        f"is addressed by its id alone, so the two merge once one of them "
        f"states its MatchClaims under other ids."
    )
    for source in sources:
        assert _claim_ids(source.get_match_claims()) == ["c1", "c2"]
        assert all(claim.bundle is source for claim in source.get_match_claims())

    first, second = _claims(NEW[:1]), _claims(NEW[1:], ["c2"])
    sources = [_holding(*first), _holding(*second)]
    merged = AlignmentBundle.from_bundles(sources)
    held = merged.get_match_claims()
    assert _claim_ids(held) == [first[0].id, "c2"]
    assert held == [*first, *second]
    assert all(claim.bundle is merged for claim in held)
    assert merged.match_claim_table.table.column_names == list(
        MatchClaimField.pa_schema.names
    )

    added = _claims(STORED[:1])
    added_id = added[0].id
    merged.add_match_claims(added)
    assert added[0].id == added_id
    assert _claim_ids(merged.get_match_claims())[-1] == added_id

    taken = sources[0].get_match_claims()[0]
    other = _holding(taken)
    assert taken.bundle is sources[0]
    assert other.get_match_claims()[0].bundle is other


def test_merge_refuses_an_id_two_distant_sources_hold() -> None:
    """The one check, before the merge begins, compares every pair of sources."""
    sources = [
        _holding(*_claims(NEW[:1], ["claim_1"])),
        _holding(*_claims(NEW[1:], ["claim_2"])),
        _holding(*_claims(STORED[:1], ["claim_1"])),
    ]
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles(sources)
    assert str(raised.value) == (
        f"MatchClaim id 'claim_1' is held by bundle '{sources[0].id}' and by "
        f"bundle '{sources[2].id}' (bundles 0 and 2 of the merge): a MatchClaim "
        f"is addressed by its id alone, so the two merge once one of them "
        f"states its MatchClaims under other ids."
    )
    assert [
        [claim.id for claim in source.get_match_claims()] for source in sources
    ] == [
        ["claim_1"],
        ["claim_2"],
        ["claim_1"],
    ]
    assert all(
        claim.bundle is source
        for source in sources
        for claim in source.get_match_claims()
    )


# endregion
