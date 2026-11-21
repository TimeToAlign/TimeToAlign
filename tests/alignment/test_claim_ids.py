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

import pyarrow as pa
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

SCORE = ContinuousLogicalTimeline(length=16, uid="score")
AUDIO = ContinuousPhysicalTimeline(length=8.0, uid="audio")


def _claims(
    pairs: list[tuple[int, float]], ids: list[str] | None = None
) -> list[MatchClaim]:
    """Fresh claims between a ``score`` note and an ``audio`` onset at each pair.

    Without *ids* every claim mints its own ``claim_1``.
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
    """Store the four ``STORED`` claims as a Parquet file and read them back."""
    pq.write_table(MatchClaimField.from_claims(_claims(STORED)).table, path)
    return MatchClaimField.from_table(pq.read_table(path))


def _ids(field: MatchClaimField) -> list[str]:
    return field.table.column("id").to_pylist()


def _columns() -> MatchClaimField:
    """The ``STORED`` pairs as event matches built by ``from_columns``."""
    return MatchClaimField.from_columns(
        timeline_a_ids=["score"] * 4,
        timeline_b_ids=["audio"] * 4,
        event_a_ids=[f"note_{q}" for q, _ in STORED],
        event_b_ids=[f"onset_{s}" for _, s in STORED],
        coordinate_a=[q for q, _ in STORED],
        coordinate_b=[s for _, s in STORED],
        timelines=[SCORE, AUDIO],
    )


# region Field part


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


def test_field_is_a_scope() -> None:
    field = MatchClaimField.from_claims(_claims(STORED[:3], ["s0", "s1", "s2"]))
    assert _ids(field) == ["s0", "s1", "s2"]
    assert [claim.id_minted for claim in field] == [False, False, False]

    with pytest.raises(
        ValueError, match=re.escape("claim id 'claim_1' is held by 2 claims")
    ):
        MatchClaimField.from_claims(_claims(NEW))
    with pytest.raises(ValueError, match=re.escape("claim id 'a' is held by 2 claims")):
        MatchClaimField.from_claims(_claims(NEW, ["a", "a"]))


def test_from_columns_mints_ids() -> None:
    field = MatchClaimField.from_columns(
        timeline_a_ids=["A", "A", "B"],
        timeline_b_ids=["B", "C", "C"],
        event_a_ids=["a0", "a0", "b1"],
        event_b_ids=["b0", "c0", "c1"],
        coordinate_a=[0, 0, 1],
        coordinate_b=[10.0, 20.0, 21.0],
        timelines={
            "A": ("quarters", "fraction"),
            "B": ("seconds", "float"),
            "C": ("seconds", "float"),
        },
    )
    assert _ids(field) == ["claim_1", "claim_2", "claim_3"]
    assert [claim.id_minted for claim in field] == [True, True, True]


def test_null_ids_are_minted_around_held_ones() -> None:
    table = _columns().table
    table = table.set_column(
        table.column_names.index("id"),
        "id",
        pa.array(["claim_2", None, "x", None], type=pa.string()),
    )
    table = table.set_column(
        table.column_names.index("id_minted"),
        "id_minted",
        pa.array([False, None, False, None], type=pa.bool_()),
    )
    field = MatchClaimField.from_table(table)
    assert _ids(field) == ["claim_2", "claim_1", "x", "claim_3"]
    assert [claim.id_minted for claim in field] == [False, True, False, True]


def test_two_runs_store_the_same_ids(tmp_path: Path) -> None:
    program = textwrap.dedent("""
        import sys

        from timetoalign.alignment import MatchClaimField
        from timetoalign.timelines.types import (
            ContinuousLogicalTimeline,
            ContinuousPhysicalTimeline,
        )

        pairs = [(0, 0.0), (4, 2.0), (8, 4.0), (12, 6.0)]
        MatchClaimField.from_columns(
            timeline_a_ids=["score"] * 4,
            timeline_b_ids=["audio"] * 4,
            event_a_ids=[f"note_{q}" for q, _ in pairs],
            event_b_ids=[f"onset_{s}" for _, s in pairs],
            coordinate_a=[q for q, _ in pairs],
            coordinate_b=[s for _, s in pairs],
            timelines=[
                ContinuousLogicalTimeline(length=16, uid="score"),
                ContinuousPhysicalTimeline(length=8.0, uid="audio"),
            ],
        ).to_parquet(sys.argv[1])
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
        _columns().to_parquet(tmp_path / "second.parquet")
    )
    assert _ids(first) == _ids(second) == ["claim_1", "claim_2", "claim_3", "claim_4"]
    assert first.table.equals(second.table, check_metadata=True)


# endregion


# region Bundle part


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
    assert [claim.id for claim in implicit] == [f"claim_{n}" for n in range(7, 13)]
    assert all(claim.id_minted for claim in implicit)
    edges = [data for _, _, data in graph._graph.edges(data=True)]
    assert [data["claim_id"] for data in edges] == [data["claim"].id for data in edges]
    assert len({data["claim_id"] for data in edges}) == len(edges) == 12
    by_id = {claim.id: claim for claim in explicit}
    for claim in implicit:
        source = by_id[claim.source_claim_id]
        assert (
            source.start_anchor.coordinate_a.value
            == claim.start_anchor.coordinate_a.value
        )

    kept = graph.filter(explicit_only=True).claims
    assert sorted(claim.id for claim in kept) == [f"claim_{n}" for n in range(1, 7)]
    assert all(claim.is_explicit for claim in kept)

    kept = graph.filter(timeline_ids={"score", "audio"}).claims
    assert sorted(claim.id for claim in kept) == [f"claim_{n}" for n in range(1, 7)]
    assert all(claim.is_explicit for claim in kept)


def test_references_follow_reminted_ids() -> None:
    def pair() -> tuple[MatchClaim, MatchClaim]:
        source = _claims(STORED[:1])[0]
        implicit = MatchClaim.implicit(
            "score",
            0,
            "score_copy",
            0,
            unit_a=TimeUnit.quarters,
            unit_b=TimeUnit.quarters,
            source_claim=source,
        )
        return source, implicit

    def holding_two() -> AlignmentBundle:
        return AlignmentBundle().add_match_claims(_claims(NEW))

    field = MatchClaimField.from_claims(list(pair()))
    assert _ids(field) == ["claim_1", "claim_2"]
    assert [claim.source_claim_id for claim in field] == [None, "claim_1"]

    bundle = holding_two().add_match_claim_field(field)
    joined = bundle.cross_group_claim_fields[0]
    assert _ids(joined) == ["claim_3", "claim_4"]
    assert [claim.source_claim_id for claim in joined] == [None, "claim_3"]

    source, implicit = pair()
    holding_two().add_match_claims([source, implicit])
    assert (source.id, implicit.id) == ("claim_3", "claim_4")
    assert implicit.source_claim_id == "claim_3"

    anchor = MatchClaim.from_dict(
        _claims(STORED[:1])[0].to_dict() | {"id": "anchor", "id_minted": False}
    )
    bundle = AlignmentBundle().add_match_claims([anchor, _claims(NEW[:1])[0]])
    implicit = MatchClaim.implicit(
        "score",
        0,
        "score_copy",
        0,
        unit_a=TimeUnit.quarters,
        unit_b=TimeUnit.quarters,
        source_claim=bundle.cross_group_claims[0],
    )
    bundle.add_match_claims([implicit])
    assert (implicit.id, implicit.source_claim_id) == ("claim_2", "anchor")


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
    merged_table = merged_second.table
    stored = second.table
    for name in MatchClaimField.pa_schema.names:
        if name != "id":
            assert merged_table.column(name).equals(stored.column(name))


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


# endregion
