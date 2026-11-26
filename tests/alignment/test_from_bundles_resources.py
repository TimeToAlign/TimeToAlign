"""A merge keeps its sources' resources.

Validation strategy and exact values: ``tests/alignment/README.md`` (the
section "A Merge Keeps Its Sources' Resources").
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.alignment import AlignmentBundle, MatchClaim
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.manifest import Resource
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    DiscretePhysicalTimeline,
)


def _take(prefix: str, uid: str | None = None) -> Resource:
    """The recording a source names: a resource no table reader opens."""
    return Resource(
        uid or f"{prefix}_take",
        "Audio",
        (f"{prefix}_pcm",),
        f"{prefix}_take.wav",
        "audio/wav",
    )


def _source(
    prefix: str,
    *,
    resources: list[Resource] | None = None,
    group_id: str | None = None,
) -> AlignmentBundle:
    """``source_<prefix>``: a score and a samples timeline in one group, one
    MatchClaim ``n1`` ↔ ``o1`` (``claim_<prefix>``), the given resources
    (its recording alone by default)."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid=f"{prefix}_score")
    pcm = DiscretePhysicalTimeline(length=705600, uid=f"{prefix}_pcm")
    score.add_events([{"id": "n1", "start": Fraction(0), "event_type": "Note"}])
    pcm.add_events([{"id": "o1", "start": 0, "event_type": "Note"}])
    source = AlignmentBundle(
        id=f"source_{prefix}",
        resources=[_take(prefix)] if resources is None else resources,
    )
    source.add_group(
        TimelineGroup(id=group_id or f"{prefix}_pair", timelines=[score, pcm])
    )
    source.add_match_claims(
        [
            MatchClaim.from_events(
                {"id": "n1", "start": Fraction(0)},
                score,
                {"id": "o1", "start": 0},
                pcm,
                id=f"claim_{prefix}",
            )
        ]
    )
    return source


def _claim_ids(bundle: AlignmentBundle) -> list[str]:
    return [claim.id for claim in bundle.get_match_claims()]


def _table(prefix: str, table_type: str, uid: str | None = None) -> Resource:
    return Resource(
        uid or f"{prefix}_table",
        table_type,
        (f"{prefix}_pcm", f"{prefix}_score"),
        f"{prefix}_table.parquet",
    )


def test_resources_are_kept_with_their_ids_and_fields(tmp_path: Path) -> None:
    sources = []
    for prefix in ("a", "b"):
        path = _source(prefix).to_file(tmp_path / prefix / "package.yaml")
        sources.append(AlignmentBundle.from_file(path))
    source_a, source_b = sources
    (b_entry,) = [r for r in source_b.resources if r.type == "MatchClaim"]
    assert [r.type for r in source_a.resources] == ["Audio", "MatchClaim"]

    merged = AlignmentBundle.from_bundles(sources, id="merged")

    assert merged.resources == [*source_a.resources, _take("b")]
    assert b_entry.id not in [r.id for r in merged.resources]
    assert _claim_ids(merged) == ["claim_a", "claim_b"]


@pytest.mark.parametrize("table_type", ["MatchClaim", "Projection", "Box"])
def test_one_entry_per_table_type(table_type: str) -> None:
    sources = [
        _source(prefix, resources=[_table(prefix, table_type), _take(prefix)])
        for prefix in ("a", "b")
    ]
    merged = AlignmentBundle.from_bundles(sources, id="merged")
    assert merged.resources == [_table("a", table_type), _take("a"), _take("b")]

    sources = [
        _source(prefix, resources=[_table(prefix, table_type, "table"), _take(prefix)])
        for prefix in ("a", "b")
    ]
    merged = AlignmentBundle.from_bundles(sources, id="merged")
    assert merged.resources == [
        _table("a", table_type, "table"),
        _take("a"),
        _take("b"),
    ]


def test_a_shared_resource_id_is_refused_before_anything_is_merged() -> None:
    source_a = _source("a", group_id="pair")
    source_b = _source("b", resources=[_take("b", "a_take")], group_id="pair")
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles([source_a, source_b], id="merged")
    assert str(raised.value) == (
        "resource 'a_take' of bundle 'source_b' cannot join bundle 'merged': its "
        "id 'a_take' is already held by resource 'a_take' of bundle 'source_a'"
    )
    for source, prefix in ((source_a, "a"), (source_b, "b")):
        assert list(source.timelines) == [f"{prefix}_score", f"{prefix}_pcm"]
        assert list(source.groups) == ["pair"]
        (group,) = source.groups.values()
        for timeline in source.timelines.values():
            scopes = timeline._scope_containers()
            assert len(scopes) == 2
            assert {id(scope) for scope in scopes} == {id(source), id(group)}
    assert source_a.resources == [_take("a")]
    assert source_b.resources == [_take("b", "a_take")]

    # Without the shared resource, the merge reaches the shared group id.
    source_b.resources = [_take("b")]
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles([source_a, source_b], id="merged")
    assert str(raised.value) == (
        "group 'pair' cannot join bundle 'merged': its id 'pair' is already held "
        "by group 'pair'"
    )


def test_a_resource_id_held_by_a_timeline_or_the_merge_is_refused() -> None:
    source_a = _source("a")
    source_b = _source("b", resources=[_take("b", "a_score")])
    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles([source_a, source_b], id="merged")
    assert str(raised.value) == (
        "resource 'a_score' of bundle 'source_b' cannot join bundle 'merged': its "
        "id 'a_score' is already held by timeline 'a_score'"
    )

    with pytest.raises(ValueError) as raised:
        AlignmentBundle.from_bundles([_source("a"), _source("b")], id="a_take")
    assert str(raised.value) == (
        "resource 'a_take' of bundle 'source_a' cannot join bundle 'a_take': its "
        "id 'a_take' is already held by bundle 'a_take'"
    )


def test_the_merged_package_writes_and_reads_back(tmp_path: Path) -> None:
    a_claims = Resource(
        "a_claims", "MatchClaim", ("a_pcm", "a_score"), "tables/a_claims.parquet"
    )
    b_claims = Resource(
        "b_claims", "MatchClaim", ("b_pcm", "b_score"), "b_claims.parquet"
    )
    sources = [
        _source("a", resources=[a_claims, _take("a")]),
        _source("b", resources=[b_claims, _take("b")]),
    ]
    merged = AlignmentBundle.from_bundles(sources, id="merged")

    path = merged.to_file(tmp_path / "package.yaml")

    assert sorted(
        str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*") if p.is_file()
    ) == ["package.yaml", "tables/a_claims.parquet"]
    written = Resource.of_file(
        tmp_path / "tables" / "a_claims.parquet",
        id="a_claims",
        type="MatchClaim",
        timelines=["a_pcm", "a_score", "b_pcm", "b_score"],
        url="tables/a_claims.parquet",
    )
    assert written.media_type == "application/vnd.apache.parquet"
    assert merged.resources == [written, _take("a"), _take("b")]

    read = AlignmentBundle.from_file(path)
    assert sorted(read.resources, key=lambda r: r.id) == [
        written,
        _take("a"),
        _take("b"),
    ]
    assert _claim_ids(read) == ["claim_a", "claim_b"]
