"""Claim ids: unique in the scope holding the claims, the same in every process.

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

import pyarrow.parquet as pq
import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim, MatchClaimField
from timetoalign.alignment.graph import MatchGraph
from timetoalign.core import TimeUnit
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

STORED = [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
NEW = [(2, 1.0), (6, 3.0)]


def _claims(pairs: list[tuple[int, float]]) -> list[MatchClaim]:
    """Fresh projections ``score`` (quarters) -> ``audio`` (seconds)."""
    return [
        MatchClaim.from_projection(
            event={"start": quarters},
            source_tl_id="score",
            target_tl_id="audio",
            target_coord=seconds,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        for quarters, seconds in pairs
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
    """Store the four ``STORED`` claims as a Parquet file and read them back."""
    pq.write_table(MatchClaimField.from_claims(_claims(STORED)).table, path)
    return MatchClaimField.from_table(pq.read_table(path))


def _ids(field: MatchClaimField) -> list[str]:
    return field.table.column("match_claim").combine_chunks().field("id").to_pylist()


def test_free_claim_mints_claim_1() -> None:
    first, second = _claims(NEW)
    assert [(c.id, c.id_minted) for c in (first, second)] == [
        ("claim_1", True),
        ("claim_1", True),
    ]
    named = MatchClaim.from_dict(
        first.to_dict() | {"id": "onset 7", "id_minted": False}
    )
    assert (named.id, named.id_minted) == ("onset 7", False)
    with pytest.raises(ValueError, match=re.escape("Invalid id 'a\\tb'")):
        MatchClaim.from_dict(first.to_dict() | {"id": "a\tb", "id_minted": False})


def test_two_free_claims_become_distinct() -> None:
    def pair() -> tuple[MatchClaim, MatchClaim]:
        to_audio = _claims(STORED[:1])[0]
        to_copy = MatchClaim.from_projection(
            event={"start": 4},
            source_tl_id="score",
            target_tl_id="score_copy",
            target_coord=4,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.quarters,
        )
        return to_audio, to_copy

    first, second = pair()
    assert (first.id, second.id) == ("claim_1", "claim_1")

    graph = MatchGraph([first, second])
    kept = graph.filter(timeline_ids={"score", "audio"}).claims
    assert len(kept) == 1 and kept[0] is first
    kept = graph.filter(timeline_ids={"score", "score_copy"}).claims
    assert len(kept) == 1 and kept[0] is second
    edge_ids = sorted(data["claim_id"] for _, _, data in graph._graph.edges(data=True))
    assert edge_ids == ["claim_1", "claim_1"]

    assert _ids(MatchClaimField.from_claims([first, second])) == ["claim_1", "claim_2"]
    assert (first.id, second.id) == ("claim_1", "claim_1")

    bundle = AlignmentBundle()
    bundle.add_match_claims([first, second])
    assert (first.id, second.id) == ("claim_1", "claim_2")
    kept = MatchGraph(bundle.cross_group_claims).filter(
        timeline_ids={"score", "score_copy"}
    )
    assert [claim.id for claim in kept.claims] == ["claim_2"]


def test_field_is_a_scope() -> None:
    field = MatchClaimField.from_claims(_claims(STORED[:3]))
    assert _ids(field) == ["claim_1", "claim_2", "claim_3"]
    assert [claim.id_minted for claim in field] == [True, True, True]

    minted, given = _claims(NEW)
    given = given.model_copy(update={"id": "claim_1", "id_minted": False})
    assert _ids(MatchClaimField.from_claims([minted, given])) == ["claim_2", "claim_1"]

    twice = [
        claim.model_copy(update={"id": "a", "id_minted": False})
        for claim in _claims(NEW)
    ]
    with pytest.raises(ValueError, match=re.escape("claim id 'a' is given twice")):
        MatchClaimField.from_claims(twice)


def test_from_columns_mints_ids() -> None:
    field = MatchClaimField.from_columns(
        timeline_a_ids=["A", "A", "B"],
        timeline_b_ids=["B", "C", "C"],
        coordinate_a=[0.0, 0.0, 1.0],
        coordinate_b=[10.0, 20.0, 21.0],
        unit_a=TimeUnit.number,
        unit_b=TimeUnit.number,
    )
    assert _ids(field) == ["claim_1", "claim_2", "claim_3"]
    assert [claim.id_minted for claim in field] == [True, True, True]


def test_stored_and_new_claims_are_distinct(tmp_path: Path) -> None:
    stored = _stored(tmp_path / "claims.parquet")
    assert _ids(stored) == ["claim_1", "claim_2", "claim_3", "claim_4"]

    rebuilt = _claims(STORED) + _claims(NEW)
    assert {claim.id for claim in rebuilt} == {"claim_1"}
    bundle = _bundle()
    bundle.add_match_claim_field(stored)
    bundle.add_match_claims(rebuilt)

    assert [claim.id for claim in rebuilt] == [f"claim_{n}" for n in range(5, 11)]
    assert all(claim.bundle is bundle for claim in rebuilt)
    assert [
        claim is kept for claim, kept in zip(rebuilt, bundle.cross_group_claims)
    ] == [True] * 6
    ids = [claim.id for claim in bundle.get_match_claims()]
    assert len(ids) == 10
    assert sorted(ids, key=lambda claim_id: int(claim_id.split("_")[1])) == [
        f"claim_{n}" for n in range(1, 11)
    ]


def test_graph_filter_unaffected(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.add_match_claim_field(_stored(tmp_path / "claims.parquet"))
    bundle.add_match_claims(_claims(NEW))
    explicit = bundle.get_match_claims()
    assert sorted(claim.id for claim in explicit) == [f"claim_{n}" for n in range(1, 7)]

    graph = MatchGraph(explicit).extend_to_groups(bundle.groups)
    implicit = [claim for claim in graph.claims if not claim.is_explicit]
    assert len(implicit) == 6
    assert {claim.id for claim in implicit} == {"claim_1"}

    kept = graph.filter(explicit_only=True).claims
    assert sorted(claim.id for claim in kept) == [f"claim_{n}" for n in range(1, 7)]
    assert all(claim.is_explicit for claim in kept)

    kept = graph.filter(timeline_ids={"score", "audio"}).claims
    assert sorted(claim.id for claim in kept) == [f"claim_{n}" for n in range(1, 7)]
    assert all(claim.is_explicit for claim in kept)


def test_two_runs_merged(tmp_path: Path) -> None:
    program = textwrap.dedent("""
        import sys

        import pyarrow.parquet as pq

        from timetoalign.alignment import MatchClaim, MatchClaimField
        from timetoalign.core import TimeUnit

        claims = [
            MatchClaim.from_projection(
                event={"start": quarters},
                source_tl_id="score",
                target_tl_id="audio",
                target_coord=seconds,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
            )
            for quarters, seconds in [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
        ]
        pq.write_table(MatchClaimField.from_claims(claims).table, sys.argv[1])
        """)
    first_path = tmp_path / "first.parquet"
    subprocess.run(
        [sys.executable, "-c", program, str(first_path)],
        check=True,
        env=dict(os.environ),
    )
    first = MatchClaimField.from_table(pq.read_table(first_path))
    second = _stored(tmp_path / "second.parquet")
    assert _ids(first) == _ids(second) == ["claim_1", "claim_2", "claim_3", "claim_4"]
    assert first.table.equals(second.table, check_metadata=True)

    bundle = _bundle()
    bundle.add_match_claim_field(first)
    bundle.add_match_claim_field(second)
    merged_first, merged_second = bundle.cross_group_claim_fields
    assert merged_first is first
    assert _ids(merged_second) == ["claim_5", "claim_6", "claim_7", "claim_8"]
    assert _ids(second) == ["claim_1", "claim_2", "claim_3", "claim_4"]
    assert [claim.id_minted for claim in merged_second] == [True] * 4
    column = merged_second.table.column("match_claim").combine_chunks()
    stored = second.table.column("match_claim").combine_chunks()
    for sub_field in MatchClaimField.pa_schema:
        if sub_field.name != "id":
            assert column.field(sub_field.name).equals(stored.field(sub_field.name))


def test_given_id_refused_atomically() -> None:
    held = _claims(STORED[:1])[0].model_copy(update={"id": "a", "id_minted": False})
    bundle = _bundle()
    bundle.add_match_claims([held])
    message = re.escape("claim id 'a' is already held by another claim of this bundle")

    newcomer, given = _claims(NEW)
    given = given.model_copy(update={"id": "a", "id_minted": False})
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claims([newcomer, given])
    assert bundle.cross_group_claims == [held]
    assert (newcomer.id, newcomer.bundle) == ("claim_1", None)
    assert (given.id, given.bundle) == ("a", None)

    field = MatchClaimField.from_claims([given])
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claim_field(field)
    assert bundle.cross_group_claim_fields == []
    assert _ids(field) == ["a"]


def test_held_claim_joins_as_copy() -> None:
    sources = [AlignmentBundle(), AlignmentBundle()]
    for source in sources:
        source.add_match_claims(_claims(NEW))
        assert [claim.id for claim in source.cross_group_claims] == [
            "claim_1",
            "claim_2",
        ]

    merged = AlignmentBundle.from_bundles(sources)
    assert [claim.id for claim in merged.cross_group_claims] == [
        "claim_1",
        "claim_2",
        "claim_3",
        "claim_4",
    ]
    assert all(claim.bundle is merged for claim in merged.cross_group_claims)
    for source in sources:
        assert [claim.id for claim in source.cross_group_claims] == [
            "claim_1",
            "claim_2",
        ]
        assert all(claim.bundle is source for claim in source.cross_group_claims)
