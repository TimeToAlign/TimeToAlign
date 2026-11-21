"""A package: the manifest's resources, the bundle's tables beside it.

Validation strategy and exact values: ``tests/alignment/README.md`` (the
section "A Package: the Manifest and Its Tables").
"""

from __future__ import annotations

import hashlib
import re
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import timetoshare
from timetoshare.tables import (
    TableError,
    check_match_claims,
    match_claim_problems,
    projection_problems,
)

from timetoalign.alignment import AlignmentBundle, MatchClaimField, MatchGraph
from timetoalign.core import Provenance
from timetoalign.core.time import number_to_json
from timetoalign.timelines import Timeline
from timetoalign.timelines.manifest import (
    PARQUET_MEDIA_TYPE,
    Integrity,
    Manifest,
    Resource,
    projection_table,
)
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscretePhysicalTimeline,
)

#: TimeToShare's checkout beside this repository, where its fixtures live.
_TIMETOSHARE = Path(__file__).resolve().parents[2].parent / "TimeToShare"
_VERSION = timetoshare.SCHEMA_VERSION
_TABLES = _TIMETOSHARE / "fixtures" / "tables" / _VERSION
_RESOURCES = _TIMETOSHARE / "fixtures" / _VERSION / "bundles" / "bundle_resources.yaml"

needs_tables = pytest.mark.skipif(
    not (_TABLES / "manifest.yaml").is_file(),
    reason=(
        f"TimeToShare's table fixture {_TABLES} is absent: no TimeToShare "
        f"checkout beside this repository"
    ),
)
needs_resources = pytest.mark.skipif(
    not _RESOURCES.is_file(),
    reason=(
        f"TimeToShare's fixture {_RESOURCES} is absent: no TimeToShare "
        f"checkout beside this repository"
    ),
)

FILES = ["match_claims.parquet", "package.yaml", "projections.parquet"]


def _manifest(**extra: Any) -> dict[str, Any]:
    """A manifest of two roots, ``score`` and ``pcm``, IDs written ``#``."""
    return {
        "timetoshare": _VERSION,
        "timelines": [
            {"id": "#score", "unit": "quarters", "length": [16, 1]},
            {"id": "#pcm", "unit": "samples", "length": 705600},
        ],
        **extra,
    }


def _resource(**members: Any) -> dict[str, Any]:
    entry = {
        "id": "#resource_1",
        "type": "MatchClaim",
        "timelines": ["#score", "#pcm"],
        "url": "match_claims.parquet",
    }
    entry.update(members)
    return entry


def _bundle(*, nomatch: bool = True) -> AlignmentBundle:
    """The synthetic package of the README, before it is written."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    part = ContinuousLogicalTimeline(length=Fraction(4), uid="part")
    score.add_child(part, Fraction(8))
    score.add_events(
        [
            {
                "id": "n1",
                "start": Fraction(0),
                "duration": Fraction(1),
                "event_type": "Note",
            },
            {"id": "n2", "start": Fraction(3, 2), "event_type": "Note"},
        ]
    )
    part.add_events([{"id": "p1", "start": Fraction(1), "event_type": "Note"}])
    bundle = AlignmentBundle(id="package")
    bundle.add_timeline(score)
    bundle.add_timeline(DiscretePhysicalTimeline(length=705600, uid="pcm"))
    bundle.add_timeline(ContinuousPhysicalTimeline(length=16.0, uid="audio"))
    provenance = Provenance(method="imported", service="https://example.org/aligner")
    bundle.create_match_claims(
        [
            ("n1", "score", {"start": 0, "end": 44100}, "pcm"),
            ("n2", "score", {"start": 66150}, "pcm"),
            ({"start": 0.5, "id": "onset_a"}, "audio", {"start": 22050}, "pcm"),
            ("p1", "part", {"start": 9.0}, "audio"),
        ],
        provenance=provenance,
    )
    if nomatch:
        bundle.create_match_claims(
            [("n2", "score", None, "audio")], provenance=provenance
        )
    return bundle


def _cell(
    numerator: int, denominator: int, unit: str, number_type: str
) -> dict[str, Any]:
    """A coordinate cell: the number as a float and as its exact ratio."""
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "unit": unit,
        "number_type": number_type,
    }


def _merged(*ids: str | None) -> AlignmentBundle:
    """``from_bundles`` of two bundles, ``a`` and ``b``, each holding ``claim_1``.

    Each source holds a score and a samples timeline (``a_score``,
    ``a_pcm``; ``b_score``, ``b_pcm``) and one instant claim between them;
    a source given ``None`` has no ID of its own (it is minted).
    """
    sources = []
    for prefix, identifier in zip(("a", "b"), ids, strict=True):
        source = (
            AlignmentBundle() if identifier is None else AlignmentBundle(id=identifier)
        )
        source.add_timeline(
            ContinuousLogicalTimeline(length=Fraction(16), uid=f"{prefix}_score")
        )
        source.add_timeline(
            DiscretePhysicalTimeline(length=705600, uid=f"{prefix}_pcm")
        )
        source.create_match_claims(
            [({"start": 0}, f"{prefix}_score", {"start": 0}, f"{prefix}_pcm")]
        )
        sources.append(source)
    return AlignmentBundle.from_bundles(sources, id="merged")


def _bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


class TestResources:
    @needs_resources
    def test_the_fixture_s_resources_are_read_as_descriptors(self) -> None:
        bundle = AlignmentBundle.from_file(_RESOURCES)

        assert bundle.resources == [
            Resource(
                "resource_1",
                "MatchClaim",
                ("score", "take_1"),
                "match_claims.parquet",
                PARQUET_MEDIA_TYPE,
            ),
            Resource("resource_2", "Projection", ("take_1",), "projections.parquet"),
        ]
        assert len(bundle.match_claim_table) == 0

    def test_unknown_keys_of_a_resource_and_its_integrity_are_kept(self) -> None:
        entry = _resource(
            integrity={"algorithm": "SHA-256", "digest": "ab", "note": "kept"},
            origin="kept too",
        )
        manifest = _manifest(resources=[entry])

        bundle = AlignmentBundle.from_dict(manifest)

        assert bundle.resources[0].integrity == Integrity("SHA-256", "ab")
        assert bundle.to_dict()["resources"] == [entry]

    def test_a_timeline_read_whole_writes_its_resources_back(self) -> None:
        entry = _resource(timelines=["#score"])
        manifest = {
            "timetoshare": _VERSION,
            "timelines": [{"id": "#score", "unit": "quarters", "length": [16, 1]}],
            "resources": [entry],
        }

        assert Timeline.from_dict(manifest).to_dict()["resources"] == [entry]

    @pytest.mark.parametrize(
        ("resources", "message"),
        [
            (
                [_resource(timelines=["#score", "#ghost"])],
                "resource 'resource_1' names timelines the manifest lacks: ['ghost']",
            ),
            (
                [_resource(), _resource(id="#resource_2")],
                "at most one resource has the type 'MatchClaim'; the manifest "
                "names 2: ['resource_1', 'resource_2']",
            ),
            (
                [_resource(integrity={"algorithm": "CRC32", "digest": "ab"})],
                "'CRC32' is no hash function a resource's integrity names: write "
                "one of ['MD5', 'SHA-1', 'SHA-256', 'SHA-384', 'SHA-512']",
            ),
            (
                [_resource(id="#table_1")],
                "resource 'table_1': a resource's ID is 'resource_' followed by a "
                "positive integer",
            ),
            (
                [_resource(id="#resource_1"), _resource(id="#resource_1", type="Note")],
                "the id 'resource_1' is held twice in the manifest: by resource "
                "'resource_1' and by resource 'resource_1'",
            ),
        ],
        ids=["timeline_unknown", "match_claim_twice", "algorithm", "id", "id_twice"],
    )
    def test_malformed_resources_are_refused(
        self, resources: list[dict[str, Any]], message: str
    ) -> None:
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            Manifest.read(_manifest(resources=resources))


class TestWriting:
    def test_the_package_is_written_and_named(self, tmp_path: Path) -> None:
        bundle = _bundle()

        path = bundle.to_file(tmp_path / "package.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == FILES
        projections = pq.read_table(tmp_path / "projections.parquet")
        assert projections.to_pylist() == [
            {
                "id": "onset_a",
                "timeline_id": "audio",
                "start": _cell(1, 2, "seconds", "float"),
                "duration": None,
                "projection_of": None,
            },
            {
                "id": "projection_1",
                "timeline_id": "audio",
                "start": _cell(9, 1, "seconds", "float"),
                "duration": None,
                "projection_of": {"timeline_id": "part", "event_id": "p1"},
            },
            {
                "id": "projection_1",
                "timeline_id": "pcm",
                "start": _cell(0, 1, "samples", "int"),
                "duration": _cell(44100, 1, "samples", "int"),
                "projection_of": {"timeline_id": "score", "event_id": "n1"},
            },
            {
                "id": "projection_2",
                "timeline_id": "pcm",
                "start": _cell(66150, 1, "samples", "int"),
                "duration": None,
                "projection_of": {"timeline_id": "score", "event_id": "n2"},
            },
            {
                "id": "projection_3",
                "timeline_id": "pcm",
                "start": _cell(22050, 1, "samples", "int"),
                "duration": None,
                "projection_of": None,
            },
        ]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        check_match_claims(pq.read_table(tmp_path / "match_claims.parquet"), manifest)
        # An event ID is unique on its timeline: ``projection_1`` on ``audio``
        # and on ``pcm`` are two events, as TimeToShare's check agrees.
        assert projection_problems(projections, manifest) == []

        expected = []
        for number, (kind, timelines, name) in enumerate(
            [
                (
                    "MatchClaim",
                    ("audio", "part", "pcm", "score"),
                    "match_claims.parquet",
                ),
                ("Projection", ("audio", "pcm"), "projections.parquet"),
            ],
            start=1,
        ):
            data = (tmp_path / name).read_bytes()
            expected.append(
                Resource(
                    f"resource_{number}",
                    kind,
                    timelines,
                    name,
                    PARQUET_MEDIA_TYPE,
                    len(data),
                    Integrity("SHA-256", hashlib.sha256(data).hexdigest()),
                )
            )
        assert bundle.resources == expected
        assert (
            manifest["resources"]
            == timetoshare.with_file_ids(
                {"resources": [resource.to_dict() for resource in expected]}
            )["resources"]
        )

    def test_the_package_reads_back_exactly(self, tmp_path: Path) -> None:
        bundle = _bundle()
        (tmp_path / "first").mkdir()
        bundle.to_file(tmp_path / "first" / "package.yaml")

        read = AlignmentBundle.from_file(tmp_path / "first" / "package.yaml")

        assert read.match_claim_table == bundle.match_claim_table
        written = pq.read_table(tmp_path / "first" / "projections.parquet")
        known = read._known_timelines()
        assert projection_table(known[tid] for tid in sorted(known)).equals(written)
        assert read.to_dict() == bundle.to_dict()
        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "package.yaml")
        assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")

    def test_writing_over_the_package_obeys_the_succession_rule(
        self, tmp_path: Path
    ) -> None:
        _bundle().to_file(tmp_path / "package.yaml")
        before = _bytes(tmp_path)

        _bundle().to_file(tmp_path / "package.yaml")
        assert _bytes(tmp_path) == before

        message = (
            "The new MatchClaim table lacks the MatchClaim 'claim_5' of the "
            "existing one: a MatchClaim is never removed (state the belief "
            "'false' instead)."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _bundle(nomatch=False).to_file(tmp_path / "package.yaml")
        assert _bytes(tmp_path) == before

    def test_a_bundle_without_claims_writes_the_manifest_alone(
        self, tmp_path: Path
    ) -> None:
        bundle = AlignmentBundle(id="plain")
        bundle.add_timeline(DiscretePhysicalTimeline(length=705600, uid="pcm"))

        path = bundle.to_file(tmp_path / "plain.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == ["plain.yaml"]
        assert bundle.resources == []
        assert "resources" not in timetoshare.loads(path.read_text(encoding="utf-8"))

    @needs_resources
    def test_a_manifest_whose_tables_are_absent_loads(self, tmp_path: Path) -> None:
        copy = shutil.copy(_RESOURCES, tmp_path / _RESOURCES.name)

        bundle = AlignmentBundle.from_file(copy)
        bundle.to_file(tmp_path / "out.yaml")

        assert len(bundle.match_claim_table) == 0
        assert [resource.id for resource in bundle.resources] == [
            "resource_1",
            "resource_2",
        ]
        assert (tmp_path / "out.yaml").read_bytes() == _RESOURCES.read_bytes()
        assert sorted(p.name for p in tmp_path.iterdir()) == [
            _RESOURCES.name,
            "out.yaml",
        ]

    def test_event_rows_and_the_projections_table_give_each_projection_once(
        self, tmp_path: Path
    ) -> None:
        _bundle().to_file(tmp_path / "package.yaml", events=True)

        read = AlignmentBundle.from_file(tmp_path / "package.yaml")

        assert read.timelines["pcm"].n_events == 3
        assert read.timelines["audio"].n_events == 2

    def test_an_entry_read_keeps_its_id_and_location(self, tmp_path: Path) -> None:
        bundle = _bundle()
        bundle.resources = [
            Resource("resource_7", "MatchClaim", ("score",), "tables/claims.parquet")
        ]

        bundle.to_file(tmp_path / "package.yaml")

        assert sorted(
            path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
        ) == ["package.yaml", "projections.parquet", "tables", "tables/claims.parquet"]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            (
                "resource_7",
                "MatchClaim",
                ("audio", "part", "pcm", "score"),
                "tables/claims.parquet",
            ),
            ("resource_8", "Projection", ("audio", "pcm"), "projections.parquet"),
        ]

    def test_a_url_with_a_scheme_is_refused(self, tmp_path: Path) -> None:
        bundle = _bundle()
        url = "https://example.org/claims.parquet"
        bundle.resources = [Resource("resource_1", "MatchClaim", ("score",), url)]

        message = (
            f"a resource lies at {url!r}; the library reads and writes only a "
            f"file named relative to the manifest"
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bundle.to_file(tmp_path / "package.yaml")
        assert list(tmp_path.iterdir()) == []

    def test_claims_merged_from_other_bundles_keep_their_scopes(
        self, tmp_path: Path
    ) -> None:
        merged = _merged("source_a", "source_b")
        merged.create_match_claims(
            [({"start": 1}, "a_score", {"start": 44100}, "b_pcm")]
        )
        (tmp_path / "first").mkdir()

        path = merged.to_file(tmp_path / "first" / "merged.yaml")

        table = pq.read_table(tmp_path / "first" / "match_claims.parquet")
        assert table.column_names[-1] == "scope"
        assert table.select(["id", "scope"]).to_pylist() == [
            {"id": "claim_1", "scope": "source_a"},
            {"id": "claim_1", "scope": "source_b"},
            {"id": "claim_1", "scope": None},
        ]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        assert match_claim_problems(table, manifest) == []
        message = (
            "This MatchClaim table mixes scopes (its column 'scope'), which a "
            "MatchClaimField does not hold: read the package with "
            "AlignmentBundle.from_file."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            MatchClaimField.from_parquet(tmp_path / "first" / "match_claims.parquet")

        read = AlignmentBundle.from_file(path)

        assert read.claim_scopes == ("merged", "source_a", "source_b")
        assert read.claim_table.column("scope").to_pylist() == [1, 2, 0]
        assert read.claim_table.equals(merged.claim_table)
        assert read.match_claim_table == merged.match_claim_table
        assert read.to_dict() == merged.to_dict()
        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "merged.yaml")
        assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")

    def test_a_merged_package_obeys_the_succession_rule(self, tmp_path: Path) -> None:
        _merged("source_a", "source_b").to_file(tmp_path / "merged.yaml")
        before = _bytes(tmp_path)

        _merged("source_a", "source_b").to_file(tmp_path / "merged.yaml")
        assert _bytes(tmp_path) == before

        message = (
            "The new MatchClaim table lacks the MatchClaim 'claim_1' of the scope "
            "'source_b' of the existing one: a MatchClaim is never removed (state "
            "the belief 'false' instead)."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _merged("source_a", "source_c").to_file(tmp_path / "merged.yaml")
        assert _bytes(tmp_path) == before

    def test_two_scopes_of_one_name_are_refused(self, tmp_path: Path) -> None:
        merged = _merged(None, None)

        message = (
            "Bundle 'merged' holds MatchClaims of 2 scopes named 'bundle_1' "
            "(keys [1, 2] of claim_scopes), which the MatchClaim table's column "
            "'scope' cannot tell apart: give the bundles merged distinct ids."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            merged.to_file(tmp_path / "merged.yaml")
        assert list(tmp_path.iterdir()) == []

    def test_a_projections_table_breaking_the_format_is_refused(
        self, tmp_path: Path
    ) -> None:
        _bundle().to_file(tmp_path / "package.yaml")
        table = pq.read_table(tmp_path / "projections.parquet")
        ids = table.column("timeline_id").to_pylist()
        ids[0] = "ghost"
        pq.write_table(
            table.set_column(1, "timeline_id", pa.array(ids, pa.string())),
            tmp_path / "projections.parquet",
        )

        with pytest.raises(TableError):
            AlignmentBundle.from_file(tmp_path / "package.yaml")

    def test_an_id_one_timeline_holds_twice_is_refused(self, tmp_path: Path) -> None:
        _bundle().to_file(tmp_path / "package.yaml")
        table = pq.read_table(tmp_path / "projections.parquet")
        ids = table.column("id").to_pylist()
        ids[3] = "projection_1"
        pq.write_table(
            table.set_column(0, "id", pa.array(ids, pa.string())),
            tmp_path / "projections.parquet",
        )

        message = "the ID 'projection_1' is given twice on the timeline 'pcm'"
        with pytest.raises(TableError, match=f"^{re.escape(message)}$"):
            AlignmentBundle.from_file(tmp_path / "package.yaml")


def _coordinate(timeline_id: str, value: Any) -> dict[str, Any]:
    return {"timeline_id": timeline_id, "value": number_to_json(value)}


def _sort_key(coordinate: dict[str, Any]) -> tuple[str, Fraction]:
    value = coordinate["value"]
    return (
        coordinate["timeline_id"],
        Fraction(*value) if isinstance(value, list) else Fraction(value),
    )


def _sorted_sets(items: list[dict[str, Any]]) -> str:
    """The anchors or MatchStamps, coordinates sorted, as TimeToShare's text."""
    normal = [
        {
            "coordinates": sorted(item["coordinates"], key=_sort_key),
            "match_claims": sorted(item["match_claims"]),
        }
        for item in items
    ]
    normal.sort(key=lambda item: [_sort_key(c) for c in item["coordinates"]])
    return timetoshare.dumps({"items": normal})


def _derived(bundle: AlignmentBundle, case: dict[str, Any]) -> tuple[str, str, int]:
    """The unique anchors, the potential MatchStamps and the edge count."""
    graph = MatchGraph(
        bundle.get_match_claims(),
        believer=case["believer"],
        belief=case["belief"],
        source=bundle,
    )
    stated: dict[tuple[tuple[str, Any], ...], set[str]] = {}
    for claim in graph.synchronous_claims:
        for anchor in claim.anchors:
            pair = tuple(
                sorted(
                    [
                        (anchor.timeline_a_id, anchor.coordinate_a.value),
                        (anchor.timeline_b_id, anchor.coordinate_b.value),
                    ],
                    key=lambda node: (node[0], Fraction(node[1])),
                )
            )
            stated.setdefault(pair, set()).add(claim.id)
    anchors = [
        {"coordinates": [_coordinate(*node) for node in pair], "match_claims": ids}
        for pair, ids in stated.items()
    ]
    stamps = []
    for component in graph.split_components():
        nodes = {
            (node.timeline_id, node.value)
            for timeline_id in component.timeline_ids
            for node in component.get_nodes_for_timeline(timeline_id)
        }
        claims = {i for pair, ids in stated.items() if pair[0] in nodes for i in ids}
        stamps.append(
            {
                "coordinates": [_coordinate(*node) for node in nodes],
                "match_claims": claims,
            }
        )
    return _sorted_sets(anchors), _sorted_sets(stamps), graph.n_edges


@needs_tables
class TestTimeToShareFixturePackage:
    @pytest.fixture(scope="class")
    def bundle(self) -> AlignmentBundle:
        return AlignmentBundle.from_file(_TABLES / "manifest.yaml")

    def test_the_package_is_read(self, bundle: AlignmentBundle) -> None:
        fixture = pq.read_table(_TABLES / "projections.parquet")
        known = bundle._known_timelines()

        assert len(bundle.match_claim_table) == 15
        assert {tid: known[tid].n_events for tid in sorted(known)} == {
            "clt_1": 0,
            "clt_2": 0,
            "cpt_1": 1,
            "dpt_1": 8,
        }
        assert projection_table(known[tid] for tid in sorted(known)).equals(
            fixture.sort_by("timeline_id")
        )

    def test_the_package_is_written_back(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle.from_file(_TABLES / "manifest.yaml")
        fixture = timetoshare.loads(
            (_TABLES / "manifest.yaml").read_text(encoding="utf-8")
        )

        bundle.to_file(tmp_path / "manifest.yaml")

        written = timetoshare.loads(
            (tmp_path / "manifest.yaml").read_text(encoding="utf-8")
        )
        assert (tmp_path / "match_claims.parquet").read_bytes() == (
            _TABLES / "match_claims.parquet"
        ).read_bytes()
        claims, projections = written["resources"]
        assert claims == fixture["resources"][0]
        data = (tmp_path / "projections.parquet").read_bytes()
        assert projections == {
            **fixture["resources"][1],
            "size": len(data),
            "integrity": {
                "algorithm": "SHA-256",
                "digest": hashlib.sha256(data).hexdigest(),
            },
        }

    @pytest.mark.parametrize(
        "index",
        [0, 1, 2, 3],
        ids=["no_filter", "actor_not_false", "actor_true", "other_actor_true"],
    )
    def test_anchors_and_potential_matchstamps_are_the_expected(
        self, bundle: AlignmentBundle, index: int
    ) -> None:
        case = timetoshare.loads(
            (_TABLES / "expected.yaml").read_text(encoding="utf-8")
        )["cases"][index]

        anchors, stamps, edges = _derived(bundle, case)

        assert anchors == _sorted_sets(case["alignment_anchors"])
        assert stamps == _sorted_sets(case["potential_matchstamps"])
        assert edges == len(case["alignment_anchors"])
        kept = bundle.get_match_claims(believer=case["believer"], belief=case["belief"])
        graph = MatchGraph(
            bundle.get_match_claims(),
            believer=case["believer"],
            belief=case["belief"],
            source=bundle,
        )
        assert [claim.id for claim in kept] == [claim.id for claim in graph.claims]
