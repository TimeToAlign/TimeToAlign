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
from timetoshare.fixtures import fixture_paths
from timetoshare.tables import (
    NUMBER_TYPE,
    PROJECTION_SCHEMA,
    TableError,
    check_match_claims,
    fixture_path,
    match_claim_problems,
    projection_problems,
)

from tests.helpers import refusal
from timetoalign.alignment import (
    AlignmentBundle,
    MatchClaim,
    MatchClaimField,
    MatchGraph,
)
from timetoalign.core import (
    BoundingBox,
    Box,
    Coordinate,
    Duration,
    EventReference,
    Note,
    Projection,
    Provenance,
    Quadrilateral,
    ScoreNote,
    SpecificPitch,
    TimeUnit,
)
from timetoalign.core.time import number_to_json, struct_to_rational
from timetoalign.loader.midi import ScoreMidiLoader
from timetoalign.loader.score.ms3 import Ms3Loader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import Timeline, TimelineGroup
from timetoalign.timelines.manifest import (
    PARQUET_MEDIA_TYPE,
    Integrity,
    Manifest,
    Resource,
    event_table,
    event_table_schema,
    projection_table,
)
from timetoalign.timelines.types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscretePhysicalTimeline,
)

_VERSION = timetoshare.SCHEMA_VERSION
#: TimeToShare's table fixture package, as its package carries it.
_TABLES = fixture_path("manifest.yaml").parent
(_RESOURCES,) = [
    path for path in fixture_paths() if path.name == "bundle_resources.yaml"
]

FILES = [
    "match_claims.parquet",
    "notes.parquet",
    "package.yaml",
    "projections.parquet",
]


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


def _number(
    numerator: int | None = None,
    denominator: int | None = None,
    value: float | None = None,
) -> dict[str, Any]:
    """A number as a package writes it: once, the other members null."""
    return {"numerator": numerator, "denominator": denominator, "value": value}


def _merged(b_claim: str = "claim_2") -> AlignmentBundle:
    """``from_bundles`` of ``source_a`` holding ``claim_1`` and ``source_b``.

    Each source holds a score and a samples timeline (``a_score``,
    ``a_pcm``; ``b_score``, ``b_pcm``), the Note ``n1`` and the onset ``o1``
    at their origins, and one instant claim between them: ``source_a``'s
    minted ``claim_1``, ``source_b``'s given *b_claim*. A MatchClaim is
    addressed by its id alone, so the two sources hold different ids.
    ``a_score`` also holds ``n2`` at 1 quarter and ``b_pcm`` ``o2`` at
    44100 samples, for a claim the merged bundle adds.
    """
    sources = []
    for prefix in ("a", "b"):
        source = AlignmentBundle(id=f"source_{prefix}")
        score = ContinuousLogicalTimeline(length=Fraction(16), uid=f"{prefix}_score")
        pcm = DiscretePhysicalTimeline(length=705600, uid=f"{prefix}_pcm")
        notes = [("n1", Fraction(0))] + ([("n2", Fraction(1))] if prefix == "a" else [])
        onsets = [("o1", 0)] + ([("o2", 44100)] if prefix == "b" else [])
        score.add_events(
            [{"id": i, "start": at, "event_type": "Note"} for i, at in notes]
        )
        pcm.add_events(
            [{"id": i, "start": at, "event_type": "Note"} for i, at in onsets]
        )
        source.add_timeline(score)
        source.add_timeline(pcm)
        claim = MatchClaim.from_events(
            {"id": "n1", "start": Fraction(0)},
            score,
            {"id": "o1", "start": 0},
            pcm,
            id=None if prefix == "a" else b_claim,
        )
        source.add_match_claims([claim])
        sources.append(source)
    return AlignmentBundle.from_bundles(sources, id="merged")


def _bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


class TestResources:
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

    def test_a_resource_id_of_any_shape_is_kept(self) -> None:
        entry = _resource(id="#table_1")

        bundle = AlignmentBundle.from_dict(_manifest(resources=[entry]))

        assert bundle.resources[0].id == "table_1"
        assert bundle.to_dict()["resources"] == [entry]

    @pytest.mark.parametrize(
        ("resources", "code", "path", "message"),
        [
            (
                [_resource(timelines=["#score", "#ghost"])],
                "resource_timeline",
                "#/resources/0/timelines/1",
                "'ghost' names no timeline of this manifest",
            ),
            (
                [_resource(), _resource(id="#resource_2")],
                "resource_type",
                "#/resources/1/type",
                "a manifest names at most one resource of the type MatchClaim; "
                "'resource_1' is one",
            ),
            (
                [_resource(integrity={"algorithm": "CRC32", "digest": "ab"})],
                "schema",
                "#/resources/0/integrity/algorithm",
                "'CRC32' is not one of ['MD5', 'SHA-1', 'SHA-256', 'SHA-384', 'SHA-512']",
            ),
            (
                [_resource(id="#resource_1"), _resource(id="#resource_1", type="Note")],
                "duplicate_id",
                "#/resources/1/id",
                "'resource_1' is not unique in the manifest, first at #/resources/0/id",
            ),
        ],
        ids=["timeline_unknown", "match_claim_twice", "algorithm", "id_twice"],
    )
    def test_malformed_resources_are_refused(
        self, resources: list[dict[str, Any]], code: str, path: str, message: str
    ) -> None:
        with pytest.raises(ValueError) as refused:
            Manifest.read(_manifest(resources=resources))
        assert refusal(refused.value) == (code, path, message)


class TestWriting:
    def test_a_manifest_the_format_refuses_writes_no_file(self, tmp_path: Path) -> None:
        bundle = _bundle()
        bundle.styling = {
            "timestamps": [
                {
                    "id": "timestamp_1",
                    "name": "T1",
                    "timeline": "score",
                    "coordinate": [20, 1],
                }
            ]
        }

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "package.yaml")

        assert refusal(refused.value) == (
            "timestamp_coordinate",
            "#/graphical/timestamps/0/coordinate",
            "coordinate [20, 1] of 'timestamp_1' is beyond the length [16, 1] of "
            "its timeline 'score'",
        )
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("serialization", "message"),
        [
            (
                "jcs",
                "the integer 9007199254740993 cannot be written in RFC 8785 JSON, "
                "whose numbers are doubles",
            ),
            ("rfc8785", "unknown serialization 'rfc8785'"),
        ],
        ids=["value", "name"],
    )
    def test_a_serialization_refusal_writes_no_file(
        self, tmp_path: Path, serialization: Any, message: str
    ) -> None:
        bundle = _bundle()
        bundle.add_timeline(DiscretePhysicalTimeline(length=2**53 + 1, uid="long"))

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "package.yaml", serialization=serialization)

        assert type(refused.value) is ValueError
        assert str(refused.value) == message
        assert list(tmp_path.iterdir()) == []
        assert bundle.resources == []

    def test_a_serialization_refusal_keeps_a_written_package(
        self, tmp_path: Path
    ) -> None:
        bundle = _bundle()
        bundle.add_timeline(DiscretePhysicalTimeline(length=2**53 + 1, uid="long"))
        bundle.to_file(tmp_path / "package.yaml")
        written = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
        resources = list(bundle.resources)
        assert sorted(written) == FILES
        # A belief changes the MatchClaim table a writing would write, so a
        # writer that wrote the tables before refusing changes the bytes.
        bundle.add_match_claim_belief(
            "claim_1", actor="wd:Q1", belief="false", date="2026-11-22"
        )

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "package.yaml", serialization="jcs")

        assert str(refused.value) == (
            "the integer 9007199254740993 cannot be written in RFC 8785 JSON, "
            "whose numbers are doubles"
        )
        assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == written
        assert bundle.resources == resources

    def test_the_package_is_written_and_named(self, tmp_path: Path) -> None:
        bundle = _bundle()

        path = bundle.to_file(tmp_path / "package.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == FILES
        projections = pq.read_table(tmp_path / "projections.parquet")
        assert projections.to_pylist() == [
            {
                "id": "onset_a",
                "timeline_id": "audio",
                "unit": "seconds",
                "number_type": "float",
                "start": _number(value=0.5),
                "duration": None,
                "projection_of": None,
            },
            {
                "id": "projection_4",
                "timeline_id": "audio",
                "unit": "seconds",
                "number_type": "float",
                "start": _number(value=9.0),
                "duration": None,
                "projection_of": {"timeline_id": "part", "event_id": "p1"},
            },
            {
                "id": "projection_1",
                "timeline_id": "pcm",
                "unit": "samples",
                "number_type": "int",
                "start": _number(0),
                "duration": _number(44100),
                "projection_of": {"timeline_id": "score", "event_id": "n1"},
            },
            {
                "id": "projection_2",
                "timeline_id": "pcm",
                "unit": "samples",
                "number_type": "int",
                "start": _number(66150),
                "duration": None,
                "projection_of": {"timeline_id": "score", "event_id": "n2"},
            },
            {
                "id": "projection_3",
                "timeline_id": "pcm",
                "unit": "samples",
                "number_type": "int",
                "start": _number(22050),
                "duration": None,
                "projection_of": None,
            },
        ]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        claims = pq.read_table(tmp_path / "match_claims.parquet")
        check_match_claims(claims, manifest)
        # A Projection's ID is unique in the package: ``audio``'s Projection,
        # created after ``pcm``'s three, is ``projection_4``, and the claim
        # naming it follows.
        assert claims.select(["id", "event_a_id", "event_b_id"]).to_pylist()[3] == {
            "id": "claim_4",
            "event_a_id": "p1",
            "event_b_id": "projection_4",
        }
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
                ("Note", ("part", "score"), "notes.parquet"),
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

    def test_the_actor_context_survives_the_package(self, tmp_path: Path) -> None:
        bundle = _bundle().add_prefix("orcid", "https://orcid.org/")
        bundle.add_match_claim_belief(
            "claim_2", actor="orcid:0000-0001", belief="true", date="2026-11-21"
        )
        believer = "https://orcid.org/0000-0001"
        (tmp_path / "first").mkdir()
        path = bundle.to_file(tmp_path / "first" / "package.yaml")

        read = AlignmentBundle.from_file(path)

        context = timetoshare.loads(path.read_text(encoding="utf-8"))["@context"]
        assert context == {**timetoshare.FORMAT_CONTEXT, "orcid": "https://orcid.org/"}
        assert read.prefixes == {"orcid": "https://orcid.org/"}
        assert [
            belief.actor
            for claim in read.get_match_claims()
            if claim.provenance is not None
            for belief in claim.provenance.beliefs
        ] == ["orcid:0000-0001"]
        for holder in (bundle, read):
            assert [
                claim.id
                for claim in holder.get_match_claims(
                    believer=believer, beliefs=["true"]
                )
            ] == ["claim_2"]
        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "package.yaml")
        assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")

    def test_a_prefix_nothing_uses_is_not_written(self, tmp_path: Path) -> None:
        bundle = (
            _bundle()
            .add_prefix("ex", "http://example.org/ns/")
            .add_prefix("orcid", "https://orcid.org/")
        )
        bundle.add_match_claim_belief(
            "claim_2", actor="orcid:0000-0001", belief="true", date="2026-11-21"
        )
        bundle.add_match_claim_belief(
            "claim_3", actor="wd:Q1", belief="true", date="2026-11-21"
        )

        path = bundle.to_file(tmp_path / "package.yaml")

        context = timetoshare.loads(path.read_text(encoding="utf-8"))["@context"]
        assert context == {
            **timetoshare.FORMAT_CONTEXT,
            "orcid": "https://orcid.org/",
            "wd": "http://www.wikidata.org/entity/",
        }
        assert AlignmentBundle.from_file(path).prefixes == {
            "orcid": "https://orcid.org/"
        }

    def test_writing_over_the_package_obeys_the_succession_rule(
        self, tmp_path: Path
    ) -> None:
        _bundle().to_file(tmp_path / "package.yaml")
        before = _bytes(tmp_path)

        _bundle().to_file(tmp_path / "package.yaml")
        assert _bytes(tmp_path) == before

        message = (
            "The new MatchClaim table lacks the MatchClaim 'claim_5' of the "
            "existing one: a MatchClaim is never removed; record the belief "
            "'false' in it instead (AlignmentBundle.add_match_claim_belief)."
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

    def test_projections_without_claims_are_written(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="plain")
        pcm = DiscretePhysicalTimeline(length=705600, uid="pcm")
        pcm.add_projections([Projection(start=Coordinate(22050, TimeUnit.samples))])
        bundle.add_timeline(pcm)

        bundle.to_file(tmp_path / "plain.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "plain.yaml",
            "projections.parquet",
        ]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            ("resource_1", "Projection", ("pcm",), "projections.parquet")
        ]
        read = AlignmentBundle.from_file(tmp_path / "plain.yaml")
        known = read._known_timelines()
        assert projection_table(known[tid] for tid in sorted(known)).equals(
            pq.read_table(tmp_path / "projections.parquet")
        )
        assert len(read.match_claim_table) == 0

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

    def test_an_entry_whose_file_is_neither_written_nor_there_goes(
        self, tmp_path: Path
    ) -> None:
        bundle = AlignmentBundle.from_file(_RESOURCES)
        bundle.timelines["take_1"].add_projections(
            [Projection(start=Coordinate(22050, TimeUnit.samples))]
        )

        bundle.to_file(tmp_path / "out.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "out.yaml",
            "projections.parquet",
        ]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            ("resource_2", "Projection", ("take_1",), "projections.parquet")
        ]

    def test_no_event_row_is_written_beside_the_tables(self, tmp_path: Path) -> None:
        message = (
            "A package's manifest names no event ID: the bundle writes its "
            "MatchClaim table and its event tables beside the manifest, so its "
            "event rows are not written into it; write it with events=False."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _bundle().to_file(tmp_path / "package.yaml", events=True)
        bundle = AlignmentBundle(id="plain")
        pcm = DiscretePhysicalTimeline(length=705600, uid="pcm")
        pcm.add_projections([Projection(start=Coordinate(22050, TimeUnit.samples))])
        bundle.add_timeline(pcm)
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bundle.to_file(tmp_path / "plain.yaml", events=True)
        assert list(tmp_path.iterdir()) == []

    def test_event_rows_of_a_manifest_and_its_table_give_each_projection_once(
        self, tmp_path: Path
    ) -> None:
        bundle = _bundle()
        bundle.to_file(tmp_path / "package.yaml")
        # A manifest written by hand that names the Projections in its event rows.
        (tmp_path / "package.yaml").write_text(
            timetoshare.dumps(bundle.to_dict(events=True)), encoding="utf-8"
        )

        read = AlignmentBundle.from_file(tmp_path / "package.yaml")

        assert read.timelines["pcm"].n_events == 3
        assert read.timelines["audio"].n_events == 2
        # The Note table's rows of IDs the manifest's rows gave are not added.
        notes = read.timelines["score"].get_events(
            event_type="Note", include_children=False
        )
        assert notes.column_values("id") == ["n1", "n2"]

    def test_an_entry_read_keeps_its_id_and_location(self, tmp_path: Path) -> None:
        bundle = _bundle()
        bundle.resources = [
            Resource("resource_7", "MatchClaim", ("score",), "tables/claims.parquet")
        ]

        bundle.to_file(tmp_path / "package.yaml")

        assert sorted(
            path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")
        ) == [
            "notes.parquet",
            "package.yaml",
            "projections.parquet",
            "tables",
            "tables/claims.parquet",
        ]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            (
                "resource_7",
                "MatchClaim",
                ("audio", "part", "pcm", "score"),
                "tables/claims.parquet",
            ),
            ("resource_8", "Projection", ("audio", "pcm"), "projections.parquet"),
            ("resource_9", "Note", ("part", "score"), "notes.parquet"),
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

    def test_claims_merged_from_other_bundles_keep_their_ids(
        self, tmp_path: Path
    ) -> None:
        merged = _merged()
        merged.create_match_claims([("n2", "a_score", "o2", "b_pcm")])
        (tmp_path / "first").mkdir()

        path = merged.to_file(tmp_path / "first" / "merged.yaml")

        table = pq.read_table(tmp_path / "first" / "match_claims.parquet")
        assert table.column_names == list(MatchClaimField.pa_schema.names)
        assert table.column("id").to_pylist() == ["claim_1", "claim_2", "claim_3"]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        assert match_claim_problems(table, manifest) == []

        read = AlignmentBundle.from_file(path)

        assert read.match_claim_table == merged.match_claim_table
        assert read.to_dict() == merged.to_dict()
        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "merged.yaml")
        assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")

    def test_a_merged_package_obeys_the_succession_rule(self, tmp_path: Path) -> None:
        _merged().to_file(tmp_path / "merged.yaml")
        before = _bytes(tmp_path)

        _merged().to_file(tmp_path / "merged.yaml")
        assert _bytes(tmp_path) == before

        message = (
            "The new MatchClaim table lacks the MatchClaim 'claim_2' of the "
            "existing one: a MatchClaim is never removed; record the belief "
            "'false' in it instead (AlignmentBundle.add_match_claim_belief)."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _merged("claim_7").to_file(tmp_path / "merged.yaml")
        assert _bytes(tmp_path) == before

    def test_a_scope_column_is_kept_as_a_further_column(self, tmp_path: Path) -> None:
        (tmp_path / "first").mkdir()
        path = _merged().to_file(tmp_path / "first" / "merged.yaml")
        file = tmp_path / "first" / "match_claims.parquet"
        written = pq.read_table(file)
        scope = pa.array(["source_a", None], pa.string())
        with_scope = written.append_column("scope", scope)
        pq.write_table(with_scope, file)

        field = MatchClaimField.from_parquet(file)
        assert field.table.column_names[-1] == "scope"
        assert field.table.column("scope").to_pylist() == ["source_a", None]
        assert field.package_table.equals(with_scope)
        read = AlignmentBundle.from_file(path)
        assert read.match_claim_table == field
        assert read.match_claim_table.filter(timeline_id="b_score").table.column(
            "scope"
        ).to_pylist() == [None]

        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "merged.yaml")
        assert pq.read_table(tmp_path / "second" / "match_claims.parquet").equals(
            with_scope, check_metadata=True
        )

        # The manifest holds no event rows: the claim names the events it read.
        read.add_match_claims(
            [
                MatchClaim.from_events(
                    {"id": "n2", "start": Fraction(1)},
                    read.get_timeline("a_score"),
                    {"id": "o2", "start": 44100},
                    read.get_timeline("b_pcm"),
                )
            ]
        )
        read.add_match_claim_belief(
            "claim_1",
            actor="https://orcid.org/0000-0001",
            belief="false",
            date="2026-11-21",
        )
        read.to_file(tmp_path / "second" / "merged.yaml")
        rewritten = pq.read_table(tmp_path / "second" / "match_claims.parquet")
        assert rewritten.select(["id", "scope"]).to_pylist() == [
            {"id": "claim_1", "scope": "source_a"},
            {"id": "claim_2", "scope": None},
            {"id": "claim_3", "scope": None},
        ]

    def test_unreduced_fractions_in_the_tables_read_as_their_value(
        self, tmp_path: Path
    ) -> None:
        score = ContinuousLogicalTimeline(length=Fraction(8), uid="score")
        score.add_events([{"id": "n1", "start": Fraction(1, 2), "event_type": "Note"}])
        audio = ContinuousPhysicalTimeline(length=60.0, uid="audio")
        audio.add_events([{"id": "o1", "start": 1.0, "event_type": "Note"}])
        bundle = AlignmentBundle(id="halves")
        bundle.add_timeline(score)
        bundle.add_timeline(audio)
        pair = ({"start": Fraction(3, 2)}, "score", "o1", "audio")
        bundle.create_match_claims([("n1", "score", "o1", "audio"), pair])
        bundle.to_file(tmp_path / "package.yaml")
        first = _bytes(tmp_path)

        def doubled(number: dict[str, Any] | None) -> None:
            if number is not None and number["denominator"] is not None:
                number["numerator"] *= 2
                number["denominator"] *= 2

        claims = pq.read_table(tmp_path / "match_claims.parquet")
        index = claims.schema.get_field_index("start_anchor")
        anchors = claims.column(index).to_pylist()
        for anchor in anchors:
            doubled(anchor["coordinate_a"])
        assert [a["coordinate_a"] for a in anchors] == [
            {
                "numerator": n,
                "denominator": 4,
                "value": None,
                "unit": "quarters",
                "number_type": "fraction",
            }
            for n in (2, 6)
        ]
        pq.write_table(
            claims.set_column(
                index,
                claims.schema.field(index),
                pa.array(anchors, claims.schema.field(index).type),
            ),
            tmp_path / "match_claims.parquet",
        )
        projections = pq.read_table(tmp_path / "projections.parquet")
        index = projections.schema.get_field_index("start")
        starts = projections.column(index).to_pylist()
        for start in starts:
            doubled(start)
        assert starts == [_number(6, 4)]
        pq.write_table(
            projections.set_column(
                index,
                projections.schema.field(index),
                pa.array(starts, projections.schema.field(index).type),
            ),
            tmp_path / "projections.parquet",
        )

        read = AlignmentBundle.from_file(tmp_path / "package.yaml")
        assert read.match_claim_table == bundle.match_claim_table
        held = read.get_timeline("score").get_events(event_type="Projection")
        assert [(row["id"], row["start"]) for row in held] == [
            ("projection_1", {"value": 1.5, "numerator": 3, "denominator": 2})
        ]
        # The score read holds the Projection (the Projections table is read
        # first) and the Note n1 (from the Note table), and get-or-add names
        # the Projection again.
        again = AlignmentBundle.from_file(tmp_path / "package.yaml").get_timeline(
            "score"
        )
        projection = Projection(
            start=Coordinate(Fraction(3, 2), TimeUnit.quarters),
            projection_of=EventReference(timeline_id="audio", event_id="o1"),
        )
        assert again.add_projections([projection]) == ["projection_1"]
        assert again.events.column_values("id") == ["projection_1", "n1"]

        read.to_file(tmp_path / "package.yaml")
        assert _bytes(tmp_path) == first

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

    @pytest.mark.parametrize("row", [3, 1], ids=["one_timeline", "two_timelines"])
    def test_a_projection_id_given_twice_in_the_package_is_refused(
        self, tmp_path: Path, row: int
    ) -> None:
        _bundle().to_file(tmp_path / "package.yaml")
        table = pq.read_table(tmp_path / "projections.parquet")
        ids = table.column("id").to_pylist()
        ids[row] = "projection_1"
        pq.write_table(
            table.set_column(0, "id", pa.array(ids, pa.string())),
            tmp_path / "projections.parquet",
        )

        message = "the ID 'projection_1' is given twice"
        with pytest.raises(TableError, match=f"^{re.escape(message)}$"):
            AlignmentBundle.from_file(tmp_path / "package.yaml")

    def test_one_projection_id_on_two_timelines_is_not_written(self) -> None:
        pcm = DiscretePhysicalTimeline(length=705600, uid="pcm")
        audio = ContinuousPhysicalTimeline(length=16.0, uid="audio")
        bundle = AlignmentBundle(id="package")
        bundle.add_timeline(pcm)
        bundle.add_timeline(audio)
        assert audio.add_projections(
            Projection(start=Coordinate(0.5, TimeUnit.seconds))
        ) == ["projection_1"]
        # Free, so it mints on its own.
        part = DiscretePhysicalTimeline(length=1000, uid="part")
        assert part.add_projections(
            Projection(start=Coordinate(5, TimeUnit.samples))
        ) == ["projection_1"]

        # Joining the bundle's tree as a child keeps the ID ...
        pcm.add_child(part, 0)
        assert pcm.n_children == 1
        assert part._parent is pcm
        assert part.events.table.column("id").to_pylist() == ["projection_1"]
        # ... and the table of both timelines is not written.
        with pytest.raises(ValueError) as raised:
            projection_table([audio, part])
        assert str(raised.value) == _shared_message("audio", "part")


def _shared_message(first: str, second: str) -> str:
    """The writer's refusal of ``projection_1`` on *first* and *second*."""
    return (
        f"Projection ID 'projection_1' names a Projection on the timeline "
        f"'{first}' and one on '{second}': a Projection's ID is unique among all "
        "Projections of a manifest and its package. Create the MatchClaims once "
        "the timelines share the bundle (for instance with "
        "AlignmentBundle.create_match_claims), so that their Projection IDs are "
        "minted unique"
    )


def _apart() -> tuple[DiscretePhysicalTimeline, ContinuousPhysicalTimeline]:
    """``pcm`` and ``audio``, each given ``projection_1`` while free."""
    pcm = DiscretePhysicalTimeline(length=705600, uid="pcm")
    audio = ContinuousPhysicalTimeline(length=16.0, uid="audio")
    assert pcm.add_projections(
        Projection(start=Coordinate(22050, TimeUnit.samples))
    ) == ["projection_1"]
    assert audio.add_projections(
        Projection(start=Coordinate(0.5, TimeUnit.seconds))
    ) == ["projection_1"]
    return pcm, audio


def _package_projections(bundle: AlignmentBundle, path: Path) -> list[tuple[str, str]]:
    """The ``(timeline_id, id)`` rows of the Projections table *bundle* writes."""
    bundle.to_file(path / "package.yaml")
    table = pq.read_table(path / "projections.parquet")
    return list(
        zip(table.column("timeline_id").to_pylist(), table.column("id").to_pylist())
    )


def _held_projections(bundle: AlignmentBundle) -> dict[str, list[tuple[str, Any]]]:
    """``(id, start)`` of the Projections of each timeline of *bundle*."""
    return {
        timeline_id: [
            (row["id"], struct_to_rational(row["start"]))
            for row in bundle.timelines[timeline_id]
            .events.filter(event_type="Projection")
            .table.to_pylist()
        ]
        for timeline_id in bundle.timelines
    }


class TestProjectionIdsAtAJoin:
    """A join keeps a Projection ID the manifest holds; the writer refuses it."""

    HELD = {"pcm": [("projection_1", 22050)], "audio": [("projection_1", 0.5)]}

    @staticmethod
    def _not_written(bundle: AlignmentBundle, path: Path) -> None:
        with pytest.raises(ValueError) as raised:
            bundle.to_file(path / "package.yaml")
        assert str(raised.value) == _shared_message("audio", "pcm")
        assert list(path.iterdir()) == []

    @pytest.mark.parametrize("grouped", [False, True], ids=["standalone", "grouped"])
    def test_add_timeline_joins(self, tmp_path: Path, grouped: bool) -> None:
        pcm, audio = _apart()
        bundle = AlignmentBundle(id="package")
        bundle.add_timeline(pcm, as_group="recording" if grouped else None)
        bundle.add_timeline(audio, grouped_with="pcm" if grouped else None)
        assert list(bundle.timelines) == ["pcm", "audio"]
        assert [group.timeline_ids for group in bundle.groups.values()] == (
            [["pcm", "audio"]] if grouped else []
        )
        assert _held_projections(bundle) == self.HELD
        self._not_written(bundle, tmp_path)
        # A new Projection is minted above every Projection of the manifest.
        assert audio.add_projections(
            Projection(start=Coordinate(1.0, TimeUnit.seconds))
        ) == ["projection_2"]

    def test_add_group_joins(self, tmp_path: Path) -> None:
        pcm, audio = _apart()
        bundle = AlignmentBundle(id="package")
        bundle.add_timeline(pcm)
        bundle.add_group(TimelineGroup(timelines=[audio]))
        assert list(bundle.timelines) == ["pcm", "audio"]
        assert [group.timeline_ids for group in bundle.groups.values()] == [["audio"]]
        assert _held_projections(bundle) == self.HELD
        self._not_written(bundle, tmp_path)

    def test_from_bundles_joins(self, tmp_path: Path) -> None:
        pcm, audio = _apart()
        first = AlignmentBundle(id="first").add_timeline(pcm)
        second = AlignmentBundle(id="second").add_timeline(audio)
        merged = AlignmentBundle.from_bundles([first, second])
        assert (list(merged.timelines), merged.groups) == (["pcm", "audio"], {})
        assert _held_projections(merged) == self.HELD
        self._not_written(merged, tmp_path)

    def test_another_id_is_written(self, tmp_path: Path) -> None:
        pcm, _ = _apart()
        audio = ContinuousPhysicalTimeline(length=16.0, uid="audio")
        audio.add_projections(
            Projection(id="onset_a", start=Coordinate(0.5, TimeUnit.seconds))
        )
        bundle = AlignmentBundle(id="package").add_timeline(pcm).add_timeline(audio)
        assert _package_projections(bundle, tmp_path) == [
            ("audio", "onset_a"),
            ("pcm", "projection_1"),
        ]


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
        beliefs=case["beliefs"],
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
            "clt_2": 1,
            "cpt_1": 2,
            "dpt_1": 8,
        }
        assert projection_table(known[tid] for tid in sorted(known)).equals(fixture)

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
        assert (tmp_path / "projections.parquet").read_bytes() == (
            _TABLES / "projections.parquet"
        ).read_bytes()
        assert written["resources"] == fixture["resources"]

    @pytest.mark.parametrize(
        "index",
        [0, 1, 2, 3, 4, 5, 6],
        ids=[
            "no_filter",
            "actor_default",
            "actor_true",
            "other_actor_true",
            "actor_undecided",
            "actor_unlikely_false",
            "compact_actor_true",
        ],
    )
    def test_anchors_and_potential_matchstamps_are_the_expected(
        self, bundle: AlignmentBundle, index: int
    ) -> None:
        cases = timetoshare.loads(
            (_TABLES / "expected.yaml").read_text(encoding="utf-8")
        )["cases"]
        assert len(cases) == 7
        case = cases[index]

        anchors, stamps, edges = _derived(bundle, case)

        assert anchors == _sorted_sets(case["alignment_anchors"])
        assert stamps == _sorted_sets(case["potential_matchstamps"])
        assert edges == len(case["alignment_anchors"])
        kept = bundle.get_match_claims(
            believer=case["believer"], beliefs=case["beliefs"]
        )
        graph = MatchGraph(
            bundle.get_match_claims(),
            believer=case["believer"],
            beliefs=case["beliefs"],
            source=bundle,
        )
        assert [claim.id for claim in kept] == [claim.id for claim in graph.claims]
        # Every case is read under the fixture manifest's own @context (the
        # seventh names its believer by a compact IRI only it expands).
        context = timetoshare.loads(
            (_TABLES / "manifest.yaml").read_text(encoding="utf-8")
        )["@context"]
        table = bundle.match_claim_table.filter(
            believer=case["believer"], beliefs=case["beliefs"], context=context
        )
        assert table.table.column("id").to_pylist() == [
            claim.id for claim in graph.claims
        ]


# The package of Boxes: values in tests/alignment/README.md ("Boxes in the
# package") and tests/timelines/README.md (``test_boxes.py``).
_PAGE_URL = "https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2"
_SYSTEM_ORDER = ("top_left", "bottom_left", "top_right", "bottom_right")
_MEASURE_ORDER = ("top_left", "top_right", "bottom_left", "bottom_right")


def _boxes() -> dict[str, list[Box]]:
    """The Boxes of the package, by the ID of the timeline holding them."""
    measure_1 = Quadrilateral.from_points(
        [[1221, 1589], [2151, 1589], [1221, 2266], [2161, 2266]],
        order=_MEASURE_ORDER,
        unit="pixels",
    )
    measure_2 = Quadrilateral.from_points(
        [[2151, 1589], [2766, 1589], [2161, 2266], [2775, 2266]],
        order=_MEASURE_ORDER,
        unit="pixels",
    )
    thirds = BoundingBox.from_xywh(
        Fraction(1, 3), 0, Fraction(2, 3), 1, unit="user_units", number_type="fraction"
    )
    return {
        "seg_2": [
            Box.from_shape(
                BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"),
                origin=7,
                id="rect_h2",
            )
        ],
        "sys": [
            Box.from_shape(measure_1, origin=1221, id="m1"),
            Box.from_shape(measure_2, origin=1221, id="m2"),
        ],
        "uu": [Box.from_shape(thirds, id="f")],
    }


def _box_bundle() -> AlignmentBundle:
    """A page with its system, a Thoresen crop and a user-units timeline."""
    system_1 = Quadrilateral.from_points(
        [[1221, 1589], [1221, 2266], [4097, 1589], [4101, 2266]],
        order=_SYSTEM_ORDER,
        unit="pixels",
    )
    page = DiscreteGraphicalTimeline(
        length=4512,
        uid="page",
        meta={
            "box": {"x": 0, "y": 0, "width": 4512, "height": 6174},
            "image": _PAGE_URL,
        },
    )
    system = DiscreteGraphicalTimeline(
        length=2880, uid="sys", meta={"box": system_1.to_dict()}
    )
    page.add_child(system, 1221)
    seg_2 = DiscreteGraphicalTimeline(
        length=867,
        uid="seg_2",
        meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
    )
    uu = ContinuousGraphicalTimeline(
        length=2, unit="user_units", number_type="fraction", uid="uu"
    )
    holders = {"seg_2": seg_2, "sys": system, "uu": uu}
    for timeline_id, boxes in _boxes().items():
        holders[timeline_id].add_events(boxes)
    bundle = AlignmentBundle(id="boxes")
    for root in (page, seg_2, uu):
        bundle.add_timeline(root)
    return bundle


def _held_boxes(bundle: AlignmentBundle) -> dict[str, list[Box]]:
    """Each timeline's Box rows read as Boxes on its axis, for those holding one."""
    held = {}
    for timeline_id, timeline in sorted(bundle._known_timelines().items()):
        rows = [
            row
            for row in timeline.events.table.to_pylist()
            if row["event_type"] == "Box"
        ]
        if rows:
            held[timeline_id] = [
                Box.from_row(row, unit=timeline.unit, number_type=timeline.number_type)
                for row in rows
            ]
    return held


class TestBoxes:
    def test_the_projection_table_is_the_event_table_of_projection(self) -> None:
        assert event_table_schema(Projection) == PROJECTION_SCHEMA

    def test_the_box_table_schema(self) -> None:
        def numbers(*names: str) -> pa.StructType:
            return pa.struct([pa.field(name, NUMBER_TYPE) for name in names])

        point = numbers("x", "y")
        assert event_table_schema(Box) == pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("timeline_id", pa.string()),
                pa.field("unit", pa.string()),
                pa.field("number_type", pa.string()),
                pa.field("start", NUMBER_TYPE),
                pa.field("duration", NUMBER_TYPE),
                pa.field("bounding_box", numbers("x", "y", "width", "height")),
                pa.field(
                    "quadrilateral",
                    pa.struct(
                        [
                            pa.field(corner, point)
                            for corner in Quadrilateral.CORNER_ORDER
                        ]
                    ),
                ),
            ]
        )

    def test_the_box_table_is_written_and_named(self, tmp_path: Path) -> None:
        bundle = _box_bundle()

        path = bundle.to_file(tmp_path / "package.yaml")

        assert sorted(p.name for p in tmp_path.iterdir()) == [
            "boxes.parquet",
            "package.yaml",
        ]
        table = pq.read_table(tmp_path / "boxes.parquet")
        assert table.schema == event_table_schema(Box)
        rows = {row["id"]: row for row in table.to_pylist()}
        assert [
            (row["id"], row["timeline_id"], row["unit"], row["number_type"])
            for row in table.to_pylist()
        ] == [
            ("rect_h2", "seg_2", "pixels", "int"),
            ("m1", "sys", "pixels", "int"),
            ("m2", "sys", "pixels", "int"),
            ("f", "uu", "user_units", "fraction"),
        ]
        assert rows["rect_h2"]["start"] == _number(378)
        assert rows["rect_h2"]["duration"] == _number(139)
        assert rows["rect_h2"]["bounding_box"] == {
            "x": _number(385),
            "y": _number(46),
            "width": _number(139),
            "height": _number(20),
        }
        assert rows["rect_h2"]["quadrilateral"] is None
        assert (rows["m1"]["start"], rows["m2"]["start"]) == (_number(0), _number(930))
        assert rows["m1"]["quadrilateral"]["bottom_right"] == {
            "x": _number(2161),
            "y": _number(2266),
        }
        assert rows["m1"]["bounding_box"] is None
        assert rows["f"]["start"] == _number(1, 3)
        assert rows["f"]["duration"] == _number(2, 3)
        assert rows["f"]["bounding_box"]["y"] == _number(0, 1)
        assert rows["f"]["bounding_box"]["height"] == _number(1, 1)

        data = (tmp_path / "boxes.parquet").read_bytes()
        expected = Resource(
            "resource_1",
            "Box",
            ("seg_2", "sys", "uu"),
            "boxes.parquet",
            PARQUET_MEDIA_TYPE,
            len(data),
            Integrity("SHA-256", hashlib.sha256(data).hexdigest()),
        )
        assert bundle.resources == [expected]
        manifest = timetoshare.loads(path.read_text(encoding="utf-8"))
        assert (
            manifest["resources"]
            == timetoshare.with_file_ids({"resources": [expected.to_dict()]})[
                "resources"
            ]
        )

    def test_the_boxes_read_back_exactly(self, tmp_path: Path) -> None:
        bundle = _box_bundle()
        (tmp_path / "first").mkdir()
        bundle.to_file(tmp_path / "first" / "package.yaml")

        read = AlignmentBundle.from_file(tmp_path / "first" / "package.yaml")

        assert _held_boxes(read) == _boxes()
        known, written = read._known_timelines(), bundle._known_timelines()
        assert known["page"].meta == written["page"].meta
        assert known["sys"].meta == written["sys"].meta
        assert read.to_dict() == bundle.to_dict()
        (tmp_path / "second").mkdir()
        read.to_file(tmp_path / "second" / "package.yaml")
        assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")

    def test_a_box_table_contradicting_its_shapes_is_refused(
        self, tmp_path: Path
    ) -> None:
        _box_bundle().to_file(tmp_path / "package.yaml")
        table = pq.read_table(tmp_path / "boxes.parquet")
        starts = table.column("start").to_pylist()
        starts[1] = _number(1)
        pq.write_table(
            table.set_column(
                table.schema.get_field_index("start"),
                "start",
                pa.array(starts, NUMBER_TYPE),
            ),
            tmp_path / "boxes.parquet",
        )

        message = (
            "Box 'm1' starts at 1 on timeline 'sys', but its shape starts at x "
            "1221 and the timeline stands at x 1221 of its image "
            '(meta["box"]): its start is 0.'
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            AlignmentBundle.from_file(tmp_path / "package.yaml")

    def test_a_box_refused_by_its_own_rule_reads_as_add_events_refuses_it(
        self, tmp_path: Path
    ) -> None:
        _box_bundle().to_file(tmp_path / "package.yaml")
        table = pq.read_table(tmp_path / "boxes.parquet")
        durations = table.column("duration").to_pylist()
        durations[1] = _number(10)
        pq.write_table(
            table.set_column(
                table.schema.get_field_index("duration"),
                "duration",
                pa.array(durations, NUMBER_TYPE),
            ),
            tmp_path / "boxes.parquet",
        )
        message = (
            "Box 'm1': its duration is 10 and its shape spans 940 in x (1221 to "
            "2161); a Box's duration is its shape's x extent."
        )
        with pytest.raises(ValueError) as read:
            AlignmentBundle.from_file(tmp_path / "package.yaml")
        assert type(read.value) is ValueError
        assert str(read.value) == message

        # The same row as a dict, on an empty timeline standing where sys does.
        written = _box_bundle()._known_timelines()["sys"]
        empty = DiscreteGraphicalTimeline(length=2880, uid="sys", meta=written.meta)
        m1 = _boxes()["sys"][0]
        with pytest.raises(ValueError) as added:
            empty.add_events([{**m1.to_event_row(), "duration": 10}])
        assert str(added.value) == str(read.value)

    def test_no_event_row_is_written_beside_the_box_table(self, tmp_path: Path) -> None:
        message = (
            "A package's manifest names no event ID: the bundle writes its "
            "MatchClaim table and its event tables beside the manifest, so its "
            "event rows are not written into it; write it with events=False."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            _box_bundle().to_file(tmp_path / "package.yaml", events=True)
        assert list(tmp_path.iterdir()) == []


def _seg_2() -> DiscreteGraphicalTimeline:
    """Thoresen's second crop: pixels, int, standing at x 7 of its image."""
    return DiscreteGraphicalTimeline(
        length=867,
        uid="seg_2",
        meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
    )


def _cell(value: Any) -> dict[str, Any]:
    """The memory cell of an exact value: its float, numerator, denominator."""
    value = Fraction(value)
    return {
        "value": float(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _rect_h2_row(**given: Any) -> dict[str, Any]:
    """``rect_h2`` as a dict row, its bounding box given as memory cells."""
    box = {"x": 385, "y": 46, "width": 139, "height": 20}
    return {
        "event_type": "Box",
        "id": "a",
        "start": 378,
        "duration": 139,
        "bounding_box": {name: _cell(value) for name, value in box.items()},
        **given,
    }


def _read_back(timeline: Timeline, directory: Path) -> list[dict[str, Any]]:
    """*timeline*'s events written in a bundle's package and read back, in the
    columns *timeline* holds; every further column read back is null."""
    bundle = AlignmentBundle()
    bundle.add_timeline(timeline)
    read = AlignmentBundle.from_file(bundle.to_file(directory / "package.yaml"))
    table = read._known_timeline(timeline.id).events.table
    names = timeline.events.table.column_names
    for name in set(table.column_names) - set(names):
        assert table.column(name).null_count == table.num_rows
    return table.select(names).to_pylist()


class TestBoxRowsAsTheBoxTableStatesThem:
    """Values in tests/alignment/README.md ("Nothing is accepted in memory that
    the package would write otherwise", "Box rows as the Box table states
    them", "The package never drops data silently")."""

    @pytest.mark.parametrize(
        ("y", "message"),
        [
            (
                _cell(Fraction(1, 3)),
                "A shape's y is 1/3, which pixels as int cannot hold exactly: "
                "geometry is never rounded.",
            ),
            (
                {"value": 46.0, "numerator": 47, "denominator": 1},
                "Box 'a': its bounding_box.y is the cell {'value': 46.0, "
                "'numerator': 47, 'denominator': 1}, whose sides hold two numbers "
                "(46.0 and 47/1); a cell holds one number on both sides.",
            ),
        ],
    )
    def test_a_cell_the_package_would_write_otherwise_is_refused(
        self, y: dict[str, Any], message: str
    ) -> None:
        seg_2 = _seg_2()
        row = _rect_h2_row()
        row["bounding_box"]["y"] = y

        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            seg_2.add_events([row])
        assert seg_2.n_events == 0

    def test_an_accepted_cell_reads_back_as_memory_holds_it(
        self, tmp_path: Path
    ) -> None:
        seg_2 = _seg_2()
        seg_2.add_events([_rect_h2_row()])

        assert _read_back(seg_2, tmp_path) == seg_2.events.table.to_pylist()

    def test_a_row_off_the_axis_is_refused_as_the_scalar_is(self) -> None:
        in_seconds = BoundingBox.from_xywh(
            385.0, 46.0, 139.0, 20.0, unit="seconds", number_type="float"
        )
        with pytest.raises(ValueError) as scalar:
            _seg_2().add_events(Box.from_shape(in_seconds, origin=7.0, id="a"))
        seg_2 = _seg_2()
        with pytest.raises(ValueError) as row:
            seg_2.add_events([_rect_h2_row(unit="seconds", number_type="float")])

        assert (
            str(row.value)
            == str(scalar.value)
            == (
                "The event's bounding_box is in seconds as float, the events are in "
                "pixels as int; a shape is never converted onto the events' axis."
            )
        )
        assert seg_2.n_events == 0

    def test_a_row_naming_another_timeline_is_refused(self) -> None:
        seg_2 = _seg_2()
        with pytest.raises(ValueError) as refused:
            seg_2.add_events([_rect_h2_row(timeline_id="other")])

        assert str(refused.value) == (
            "Box 'a' names the timeline 'other' (timeline_id), but is added to the "
            "timeline 'seg_2'."
        )
        assert seg_2.n_events == 0

    def test_a_row_stating_its_timeline_and_axis_is_stored_without_them(
        self, tmp_path: Path
    ) -> None:
        plain = _seg_2()
        plain.add_events([_rect_h2_row()])
        seg_2 = _seg_2()

        seg_2.add_events(
            [_rect_h2_row(timeline_id="seg_2", unit="pixels", number_type="int")]
        )

        assert seg_2.events.table.schema == plain.events.table.schema
        assert seg_2.events.table.to_pylist() == plain.events.table.to_pylist()
        (written,) = seg_2.to_dict(events=True)["timelines"][0]["events"]
        assert {"timeline_id", "unit", "number_type"}.isdisjoint(written)
        assert _read_back(seg_2, tmp_path) == seg_2.events.table.to_pylist()

    @pytest.mark.parametrize(
        ("given", "named"),
        [({"name": "crop"}, "name 'crop'"), ({"staff": 1}, "staff 1")],
    )
    def test_a_box_value_the_box_table_has_no_column_for_is_refused(
        self, tmp_path: Path, given: dict[str, Any], named: str
    ) -> None:
        seg_2 = _seg_2()
        seg_2.add_events([_rect_h2_row(id="rect_h2", **given)])
        bundle = AlignmentBundle()
        bundle.add_timeline(seg_2)

        message = (
            f"Box 'rect_h2' of timeline 'seg_2' gives {named}, which the Box table "
            "has no column for: a Box holds id, start, duration, bounding_box, "
            "quadrilateral."
        )
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            bundle.to_file(tmp_path / "package.yaml")
        assert list(tmp_path.iterdir()) == []


# Event tables of timelines holding other events: values in
# tests/alignment/README.md ("Event tables of timelines holding other events").


def _score_with_notes(*, n2: bool = True) -> ContinuousLogicalTimeline:
    """``score`` holding ``n1`` (and ``n2``), Notes with a ``midi`` column."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    notes = [
        {
            "id": "n1",
            "start": Fraction(0),
            "duration": Fraction(1),
            "event_type": "Note",
            "midi": 60,
        },
        {"id": "n2", "start": Fraction(3, 2), "event_type": "Note", "midi": 62},
    ]
    score.add_events(notes if n2 else notes[:1])
    return score


def _projection_rows(timeline: Timeline) -> list[tuple[Any, ...]]:
    """Each Projection the timeline holds: id, start, duration, reference."""
    return [
        (
            row["id"],
            struct_to_rational(row["start"]),
            struct_to_rational(row["duration"]),
            row["projection_of"],
        )
        for row in timeline.events.table.to_pylist()
        if row["event_type"] == "Projection"
    ]


def _written_twice(
    bundle: AlignmentBundle, tmp_path: Path, name: str
) -> AlignmentBundle:
    """*bundle* written, read back and written again: the bundle read back.

    Both directories hold the same files byte for byte.
    """
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    bundle.to_file(tmp_path / "first" / name)
    read = AlignmentBundle.from_file(tmp_path / "first" / name)
    read.to_file(tmp_path / "second" / name)
    assert _bytes(tmp_path / "second") == _bytes(tmp_path / "first")
    return read


#: The ms3 notes of Chopin's Op. 10 No. 3 (498 Notes).
CHOPIN_NOTES = ensure_data("vienna_1x22") / "ms3" / "chopin_op10_no3.notes.tsv"
#: The columns the Chopin score's Notes give a value in, beyond the Note's.
CHOPIN_FURTHER = [
    "chord_id",
    "gracenote",
    "mc",
    "mc_onset",
    "midi",
    "mn",
    "mn_onset",
    "name",
    "octave",
    "part_id",
    "specific_pitch",
    "staff",
    "tied",
    "tpc",
    "voice",
]
_NOTE_REFUSAL = (
    "A package's manifest names no event ID: the bundle writes its "
    "MatchClaim table and its event tables beside the manifest, so its "
    "event rows are not written into it; write it with events=False."
)


def _chopin() -> Timeline:
    return Ms3Loader.from_file(CHOPIN_NOTES).create_timeline()


def _one_note(uid: str, unit: Any, start: Any, **further: Any) -> Timeline:
    """A timeline of *unit* holding the Note ``n1`` at *start*, with *further*."""
    if unit == "quarters":
        timeline: Timeline = ContinuousLogicalTimeline(length=Fraction(8), uid=uid)
    else:
        timeline = ContinuousPhysicalTimeline(length=60.0, uid=uid)
    timeline.add_events([{"id": "n1", "start": start, "event_type": "Note", **further}])
    return timeline


class TestNotes:
    def test_the_table_s_columns(self) -> None:
        number = pa.struct(NUMBER_TYPE)
        assert event_table_schema(Note) == pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("timeline_id", pa.string()),
                pa.field("unit", pa.string()),
                pa.field("number_type", pa.string()),
                pa.field("start", number),
                pa.field("duration", number),
                pa.field("velocity", pa.int64()),
                pa.field("instrument", pa.string()),
            ]
        )
        table = event_table([_chopin()], Note)
        assert table.num_rows == 498
        assert table.column_names == [*event_table_schema(Note).names, *CHOPIN_FURTHER]
        first = table.slice(0, 1).to_pylist()[0]
        assert {
            name: first[name]
            for name in [
                "id",
                "timeline_id",
                "unit",
                "number_type",
                "start",
                "duration",
            ]
        } == {
            "id": "clt_1:note:000001",
            "timeline_id": "clt_1",
            "unit": "quarters",
            "number_type": "fraction",
            "start": _number(0, 1),
            "duration": _number(1, 2),
        }
        assert (first["velocity"], first["instrument"]) == (None, None)
        assert {
            name: first[name]
            for name in ("name", "specific_pitch", "midi", "staff", "voice")
        } == {
            "name": "B3",
            "specific_pitch": {"step": "B", "alter": 0, "octave": 3, "cents": 0.0},
            "midi": 59,
            "staff": 1,
            "voice": 1,
        }

    def test_written_and_read_back(self, tmp_path: Path) -> None:
        score = _chopin()
        bundle = AlignmentBundle(id="chopin")
        bundle.add_timeline(score)

        read = _written_twice(bundle, tmp_path, "chopin.yaml")

        assert sorted(p.name for p in (tmp_path / "first").iterdir()) == [
            "chopin.yaml",
            "notes.parquet",
        ]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            ("resource_1", "Note", ("clt_1",), "notes.parquet")
        ]
        read_score = read.get_timeline("clt_1")
        notes = list(read_score.get_events(event_type="Note").get_field(ScoreNote))
        assert len(notes) == 498
        assert notes == list(score.get_events(event_type="Note").get_field(ScoreNote))
        assert {type(note.start.value) for note in notes} == {Fraction}
        assert notes[0].pitch == SpecificPitch(step="B", alter=0, octave=3, cents=0.0)
        compared = ["id", "name", "mc", "mn", "mc_onset", "specific_pitch", "midi"]
        compared += ["tpc", "chord_id", "staff", "voice"]
        assert (
            read_score.events.table.select(compared).to_pylist()
            == score.events.table.select(compared).to_pylist()
        )

    def test_integers_of_two_widths_join(self, tmp_path: Path) -> None:
        directory = ensure_data("vienna_1x22")
        midi = ScoreMidiLoader.from_file(
            directory / "Chopin_op10_no3_p01.mid"
        ).create_timeline()
        assert midi.events.table.schema.field("staff").type == pa.int8()
        table = event_table([_chopin(), midi], Note)
        assert table.schema.field("staff").type == pa.int64()
        assert table.num_rows == 498 + 451

    def test_a_further_column_in_two_types_is_refused(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="two")
        bundle.add_timeline(_one_note("score", "quarters", Fraction(0), staff=1))
        bundle.add_timeline(_one_note("los", "seconds", 0.0, staff="P1_staff1"))

        with pytest.raises(ValueError) as refused:
            bundle.to_file(tmp_path / "two.yaml")

        assert str(refused.value) == (
            "The Note table holds the column 'staff' once: timeline 'los' holds it "
            "as string, timeline 'score' as int64, which no one column holds."
        )
        assert list(tmp_path.iterdir()) == []

    def test_a_row_on_another_axis_is_refused_on_reading(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="one")
        bundle.add_timeline(_one_note("score", "quarters", Fraction(1, 2)))
        bundle.to_file(tmp_path / "one.yaml")
        table = pq.read_table(tmp_path / "notes.parquet")
        table = table.set_column(
            table.schema.get_field_index("unit"), "unit", pa.array(["seconds"])
        ).set_column(
            table.schema.get_field_index("number_type"),
            "number_type",
            pa.array(["float"]),
        )
        table = table.set_column(
            table.schema.get_field_index("start"),
            "start",
            pa.array([_number(value=0.5)], table.schema.field("start").type),
        )
        pq.write_table(table, tmp_path / "notes.parquet")

        with pytest.raises(ValueError) as refused:
            AlignmentBundle.from_file(tmp_path / "one.yaml")

        assert str(refused.value) == (
            "Note 'n1' of timeline 'score' is in seconds as float in the Note "
            "table, but the timeline is in quarters as fraction."
        )

    def test_notes_travel_in_the_event_rows_on_request(self, tmp_path: Path) -> None:
        bundle = AlignmentBundle(id="one")
        bundle.add_timeline(_one_note("score", "quarters", Fraction(1, 2)))

        bundle.to_file(tmp_path / "one.yaml", events=True)

        assert [p.name for p in tmp_path.iterdir()] == ["one.yaml"]
        assert bundle.resources == []
        read = AlignmentBundle.from_file(tmp_path / "one.yaml").get_timeline("score")
        assert read.events.column_values("id") == ["n1"]
        bundle.add_timeline(_one_note("score_2", "quarters", Fraction(1, 2)))
        bundle.get_timeline("score_2").add_projections(
            [Projection(start=Coordinate(Fraction(1), TimeUnit.quarters))]
        )
        with pytest.raises(ValueError, match=f"^{re.escape(_NOTE_REFUSAL)}$"):
            bundle.to_file(tmp_path / "two.yaml", events=True)
        assert [p.name for p in tmp_path.iterdir()] == ["one.yaml"]


class TestEventTablesBesideOtherEvents:
    def test_one_projection_beside_notes(self, tmp_path: Path) -> None:
        score = _score_with_notes()
        assert score.add_projections(
            [
                Projection(
                    start=Coordinate(Fraction(1, 3), TimeUnit.quarters),
                    duration=Duration(Fraction(2, 3), TimeUnit.quarters),
                )
            ]
        ) == ["projection_1"]
        bundle = AlignmentBundle(id="one")
        bundle.add_timeline(score)

        read = _written_twice(bundle, tmp_path, "one.yaml")

        first = tmp_path / "first"
        assert sorted(p.name for p in first.iterdir()) == [
            "notes.parquet",
            "one.yaml",
            "projections.parquet",
        ]
        assert pq.read_table(first / "projections.parquet").to_pylist() == [
            {
                "id": "projection_1",
                "timeline_id": "score",
                "unit": "quarters",
                "number_type": "fraction",
                "start": _number(1, 3),
                "duration": _number(2, 3),
                "projection_of": None,
            }
        ]
        assert [(r.id, r.type, r.timelines, r.url) for r in bundle.resources] == [
            ("resource_1", "Projection", ("score",), "projections.parquet"),
            ("resource_2", "Note", ("score",), "notes.parquet"),
        ]
        assert _projection_rows(read._known_timelines()["score"]) == [
            ("projection_1", Fraction(1, 3), Fraction(2, 3), None)
        ]

    def test_two_boxes_beside_a_note(self, tmp_path: Path) -> None:
        seg_2 = DiscreteGraphicalTimeline(
            length=867,
            uid="seg_2",
            meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}},
        )
        boxes = [
            Box.from_shape(
                BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"),
                origin=7,
                id="rect_h2",
            ),
            Box.from_shape(
                BoundingBox.from_xywh(400, 100, 50, 30, unit="pixels"),
                origin=7,
                id="rect_b",
            ),
        ]
        assert seg_2.add_events(
            [{"event_type": "Note", "start": 0, "duration": 10, "midi": 60}, *boxes]
        ) == ["note_1", "rect_h2", "rect_b"]
        bundle = AlignmentBundle(id="boxes")
        bundle.add_timeline(seg_2)

        read = _written_twice(bundle, tmp_path, "boxes.yaml")

        first = tmp_path / "first"
        assert sorted(p.name for p in first.iterdir()) == [
            "boxes.parquet",
            "boxes.yaml",
            "notes.parquet",
        ]

        def box(x: int, y: int, width: int, height: int) -> dict[str, Any]:
            return {
                "x": _number(x),
                "y": _number(y),
                "width": _number(width),
                "height": _number(height),
            }

        assert pq.read_table(first / "boxes.parquet").to_pylist() == [
            {
                "id": "rect_h2",
                "timeline_id": "seg_2",
                "unit": "pixels",
                "number_type": "int",
                "start": _number(378),
                "duration": _number(139),
                "bounding_box": box(385, 46, 139, 20),
                "quadrilateral": None,
            },
            {
                "id": "rect_b",
                "timeline_id": "seg_2",
                "unit": "pixels",
                "number_type": "int",
                "start": _number(393),
                "duration": _number(50),
                "bounding_box": box(400, 100, 50, 30),
                "quadrilateral": None,
            },
        ]
        assert _held_boxes(read) == {"seg_2": boxes}

    def test_two_timelines_one_holding_a_projection(self, tmp_path: Path) -> None:
        score = _score_with_notes(n2=False)
        pcm = DiscretePhysicalTimeline(length=705600, uid="pcm")
        pcm.add_events([{"id": "o1", "start": 22050, "event_type": "Note"}])
        bundle = AlignmentBundle(id="two")
        bundle.add_timeline(score)
        bundle.add_timeline(pcm)
        bundle.create_match_claims([({"start": 0, "end": 4}, "score", "o1", "pcm")])

        read = _written_twice(bundle, tmp_path, "two.yaml")

        first = tmp_path / "first"
        reference = {"timeline_id": "pcm", "event_id": "o1"}
        assert pq.read_table(first / "projections.parquet").to_pylist() == [
            {
                "id": "projection_1",
                "timeline_id": "score",
                "unit": "quarters",
                "number_type": "fraction",
                "start": _number(0, 1),
                "duration": _number(4, 1),
                "projection_of": reference,
            }
        ]
        assert [(r.type, r.timelines) for r in bundle.resources] == [
            ("MatchClaim", ("pcm", "score")),
            ("Projection", ("score",)),
            ("Note", ("pcm", "score")),
        ]
        assert _projection_rows(read._known_timelines()["score"]) == [
            ("projection_1", Fraction(0), Fraction(4), reference)
        ]
        assert [
            (claim.id, claim.event_a_id, claim.timeline_a_id, claim.event_b_id)
            for claim in read.get_match_claims()
        ] == [("claim_1", "projection_1", "score", "o1")]
