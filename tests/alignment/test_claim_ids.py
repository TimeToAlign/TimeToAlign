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
import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim, MatchClaimField
from timetoalign.alignment.graph import MatchGraph
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

    Without *ids* every claim holds none until a scope mints it one.
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


def test_free_claim_holds_no_id() -> None:
    first, second = _claims(NEW)
    assert [c.id for c in (first, second)] == [None, None]
    named = MatchClaim.from_dict(first.to_dict() | {"id": "onset 7"})
    assert named.id == "onset 7"
    with pytest.raises(ValueError, match=re.escape("Invalid id 'a\\tb'")):
        MatchClaim.from_dict(first.to_dict() | {"id": "a\tb"})


def test_field_is_a_scope() -> None:
    field = MatchClaimField.from_claims(_claims(STORED[:3], ["s0", "s1", "s2"]))
    assert _ids(field) == ["s0", "s1", "s2"]

    free = _claims(NEW)
    field = MatchClaimField.from_claims(free)
    assert _ids(field) == [None, None]
    assert [claim.id for claim in free] == [None, None]
    with pytest.raises(ValueError, match=re.escape("claim id 'a' is held by 2 claims")):
        MatchClaimField.from_claims(_claims(NEW, ["a", "a"]))


def test_from_columns_mints_no_id() -> None:
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
    assert _ids(field) == [None, None, None]

    bundle = AlignmentBundle()
    for uid, timeline_class in (
        ("A", ContinuousLogicalTimeline),
        ("B", ContinuousPhysicalTimeline),
        ("C", ContinuousPhysicalTimeline),
    ):
        bundle.add_timeline(timeline_class(length=32, uid=uid))
    bundle.add_match_claim_field(field)
    held = bundle.get_match_claims()
    assert [claim.id for claim in held] == ["claim_1", "claim_2", "claim_3"]
    assert _ids(field) == [None, None, None]


def test_two_fields_join_one_bundle() -> None:
    later = MatchClaimField.from_columns(
        timeline_a_ids=["score"] * 2,
        timeline_b_ids=["audio"] * 2,
        event_a_ids=[f"note_{q}" for q, _ in NEW],
        event_b_ids=[f"onset_{s}" for _, s in NEW],
        coordinate_a=[q for q, _ in NEW],
        coordinate_b=[s for _, s in NEW],
        timelines=[SCORE, AUDIO],
    )
    bundle = _holding().add_match_claim_field(_columns()).add_match_claim_field(later)
    assert [claim.id for claim in bundle.get_match_claims()] == [
        f"claim_{n}" for n in range(1, 7)
    ]


def test_null_ids_are_minted_around_held_ones() -> None:
    table = _columns().table
    table = table.set_column(
        table.column_names.index("id"),
        "id",
        pa.array(["claim_2", None, "x", None], type=pa.string()),
    )
    field = MatchClaimField.from_table(table)
    assert _ids(field) == ["claim_2", None, "x", None]
    held = _holding().add_match_claim_field(field).get_match_claims()
    assert [claim.id for claim in held] == ["claim_2", "claim_3", "x", "claim_4"]


def test_a_field_without_ids_is_not_written(tmp_path: Path) -> None:
    path = tmp_path / "claims.parquet"
    with pytest.raises(ValueError) as raised:
        _columns().to_parquet(path)
    assert str(raised.value) == (
        "MatchClaim table row 0 holds no id, and every MatchClaim of the file "
        "holds one: add the field to an AlignmentBundle, which mints the ids, "
        "and write its match_claim_table."
    )
    assert list(tmp_path.iterdir()) == []


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
    assert _ids(first) == _ids(second) == ["claim_1", "claim_2", "claim_3", "claim_4"]
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


def test_two_free_claims_become_distinct() -> None:
    first, second = _claims(STORED[:1])[0], _copy_claim(4)
    assert (first.id, second.id) == (None, None)

    graph = MatchGraph([first, second])
    kept = graph.filter(timeline_ids={"score", "audio"}).claims
    assert len(kept) == 1 and kept[0] is first
    kept = graph.filter(timeline_ids={"score", "score_copy"}).claims
    assert len(kept) == 1 and kept[0] is second
    edge_ids = [data["claim_id"] for _, _, data in graph._graph.edges(data=True)]
    assert edge_ids == [None, None]

    assert _ids(MatchClaimField.from_claims([first, second])) == [None, None]
    assert (first.id, second.id) == (None, None)

    bundle = _bundle()
    bundle.add_match_claims([first, second])
    assert (first.id, second.id) == ("claim_1", "claim_2")
    kept = MatchGraph(bundle.get_match_claims()).filter(
        timeline_ids={"score", "score_copy"}
    )
    assert [claim.id for claim in kept.claims] == ["claim_2"]


def test_stored_and_new_claims_are_distinct(tmp_path: Path) -> None:
    stored = _stored(tmp_path / "claims.parquet")
    assert _ids(stored) == ["claim_1", "claim_2", "claim_3", "claim_4"]

    rebuilt = _claims(STORED) + _claims(NEW)
    assert {claim.id for claim in rebuilt} == {None}
    bundle = _bundle()
    bundle.add_match_claim_field(stored)
    bundle.add_match_claims(rebuilt)

    assert [claim.id for claim in rebuilt] == [f"claim_{n}" for n in range(5, 11)]
    assert all(claim.bundle is bundle for claim in rebuilt)
    held = bundle.get_match_claims()
    assert held[4:] == rebuilt
    assert all(claim.bundle is bundle for claim in held)
    assert [claim.id for claim in held] == [f"claim_{n}" for n in range(1, 11)]


def test_graph_filter_unaffected(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.add_match_claim_field(_stored(tmp_path / "claims.parquet"))
    bundle.add_match_claims(_claims(NEW))
    claims = bundle.get_match_claims()
    assert sorted(claim.id for claim in claims) == [f"claim_{n}" for n in range(1, 7)]

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
    assert sorted(claim.id for claim in kept.claims) == [
        f"claim_{n}" for n in range(1, 7)
    ]
    assert kept.inferred_edges == []

    kept = graph.filter(timeline_ids={"score", "audio"})
    assert sorted(claim.id for claim in kept.claims) == [
        f"claim_{n}" for n in range(1, 7)
    ]
    assert kept.inferred_edges == []


def test_held_ids_are_refused_never_reminted() -> None:
    source = _claims(STORED[:1], ["anchor"])[0]
    field = MatchClaimField.from_claims([source, _copy_claim(0)])
    assert _ids(field) == ["anchor", None]

    bundle = _bundle().add_match_claims(_claims(NEW))
    bundle.add_match_claim_field(field)
    held = ["claim_1", "claim_2", "anchor", "claim_3"]
    assert [claim.id for claim in bundle.get_match_claims()] == held
    with pytest.raises(
        ValueError,
        match=re.escape(
            "claim id 'claim_1' is already held by another claim of this bundle"
        ),
    ):
        bundle.add_match_claim_field(_holding(*_claims(NEW)).match_claim_table)
    assert [claim.id for claim in bundle.get_match_claims()] == held

    bundle = _bundle().add_match_claims([source, _claims(NEW[:1])[0]])
    assert [claim.id for claim in bundle.get_match_claims()] == ["anchor", "claim_1"]
    free = _copy_claim(0)
    bundle.add_match_claims([free])
    assert free.id == "claim_2"


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
    assert _ids(first) == _ids(second) == ["claim_1", "claim_2", "claim_3", "claim_4"]
    assert first.table.equals(second.table, check_metadata=True)

    bundle = _holding().add_match_claim_field(first)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "claim id 'claim_1' is already held by another claim of this bundle"
        ),
    ):
        bundle.add_match_claim_field(second)
    assert len(bundle.match_claim_table) == 4

    runs = [_holding().add_match_claim_field(run) for run in (first, second)]
    merged = AlignmentBundle.from_bundles(runs)
    assert merged.claim_scopes == (merged.id, runs[0].id, runs[1].id)
    table = merged.claim_table
    assert table.column("id").to_pylist() == _ids(first) + _ids(second)
    assert table.column("scope_key").to_pylist() == [1] * 4 + [2] * 4
    stored = pa.concat_tables([first.table, second.table])
    for name in MatchClaimField.pa_schema.names:
        assert table.column(name).equals(stored.column(name))


def test_given_id_refused_atomically() -> None:
    held = _claims(STORED[:1])[0].model_copy(update={"id": "a"})
    bundle = _bundle()
    bundle.add_match_claims([held])
    message = re.escape("claim id 'a' is already held by another claim of this bundle")

    newcomer, given = _claims(NEW)
    given = given.model_copy(update={"id": "a"})
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claims([newcomer, given])
    assert bundle.get_match_claims() == [held]
    assert (newcomer.id, newcomer.bundle) == (None, None)
    assert (given.id, given.bundle) == ("a", None)

    field = MatchClaimField.from_claims([given])
    with pytest.raises(ValueError, match=message):
        bundle.add_match_claim_field(field)
    assert bundle.get_match_claims() == [held]
    assert _ids(field) == ["a"]


def test_merge_keeps_ids_in_their_scopes() -> None:
    sources = [_holding(*_claims(NEW)) for _ in range(2)]
    for source in sources:
        assert [claim.id for claim in source.get_match_claims()] == [
            "claim_1",
            "claim_2",
        ]

    merged = AlignmentBundle.from_bundles(sources)
    held = merged.get_match_claims()
    assert [claim.id for claim in held] == ["claim_1", "claim_2"] * 2
    assert merged.claim_table.column("scope_key").to_pylist() == [1, 1, 2, 2]
    assert merged.claim_scopes == (merged.id, sources[0].id, sources[1].id)
    assert all(claim.bundle is merged for claim in held)
    for source in sources:
        assert [claim.id for claim in source.get_match_claims()] == [
            "claim_1",
            "claim_2",
        ]
        assert "scope_key" not in source.claim_table.column_names

    added = _claims(STORED[:1])
    merged.add_match_claims(added)
    assert added[0].id == "claim_1"
    assert merged.claim_table.column("scope_key").to_pylist() == [1, 1, 2, 2, 0]

    taken = sources[0].get_match_claims()[0]
    other = _holding(taken)
    assert taken.bundle is sources[0]
    assert other.get_match_claims()[0].bundle is other


# endregion
