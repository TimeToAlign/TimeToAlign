"""The MeLODy import: MeLODy's Turtle as statements (see README.md)."""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from tests.loader.minting import minted
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import TimeUnit
from timetoalign.timelines import ContinuousLogicalTimeline
from timetoalign.timelines.statements import LIBRARY_AGENT

rdflib = pytest.importorskip("rdflib")

from timetoalign.loader.metadata.melody import MelodyLoader  # noqa: E402

#: The timeline IDs an import by a loader of seed 0 mints first.
NEW = minted(0, "cgt", "cgt")

PREFIXES = """\
@prefix melod: <https://lod.academy/melod/vocab/ontology#> .
@prefix melodTerms: <https://lod.academy/melod/vocab/terms#> .
@prefix lrmoo: <http://iflastandards.info/ns/lrm/lrmoo/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
@prefix dct: <http://purl.org/dc/terms/> .
"""

WORK = """\
<urn:uuid:works/1> a melod:Work ;
  melod:hasTitle [ a melod:Title ; rdfs:label "Clair de lune"@fr ; melod:hasTitleType melodTerms:MainTitle ] ;
  melod:hasIdentifier [ a melod:Identifier ; rdfs:label "Test catalogue" ; owl:hasValue "1" ] ;
  schema:creativeWorkStatus "authentic" ;
  owl:sameAs <http://www.wikidata.org/entity/Q29117946> ;
  lrmoo:R3_is_realised_in <urn:uuid:expressions/2> ;
  dct:conformsTo melod:Work .
"""

EXPRESSION = """\
<urn:uuid:expressions/2> a melod:Expression ;
  rdfs:label "Clair de lune, piano" ;
  melod:hasContribution [ a melod:Contribution ; melod:hasAgent <urn:uuid:persons/3> ;
    melod:hasRole melodTerms:Composer ] ;
  schema:dateCreated [ a melod:Date ; melod:notBefore "1890" ; melod:notAfter "1905" ] ;
  melod:hasMeter [ a melod:Meter ; melod:hasMeterCount 9 ; melod:inMeterUnit 8 ] ;
  melod:hasDuration 297.5 .
"""

PERSON = """\
<urn:uuid:persons/3> a melod:Person ; schema:givenName "Claude" ; schema:familyName "Debussy" ;
  owl:sameAs <http://www.wikidata.org/entity/Q4700> .
"""

MANIFESTATION = """\
<urn:uuid:manifestations/4> a melod:Manifestation ;
  melod:hasTitle [ a melod:Title ; rdfs:label "Suite bergamasque, first edition" ] ;
  lrmoo:R4_embodies <urn:uuid:expressions/2> .
"""

ITEM = """\
<urn:uuid:items/5> a melod:Item ; rdfs:label "Copy 1" ; melod:hasShelfmark "TEST 1" ;
  lrmoo:R7_exemplifies <urn:uuid:manifestations/4> .
"""

INDEX = """\
<urn:uuid:works/1> a melod:WorkEntity .
"""

PART = """\
<urn:uuid:works/9> a melod:Work ;
  melod:isWorkPartOf <urn:uuid:works/1> .
"""

REPOSITORY = {
    "works/1.ttl": WORK,
    "expressions/2.ttl": EXPRESSION,
    "persons/3.ttl": PERSON,
    "manifestations/4.ttl": MANIFESTATION,
    "items/5.ttl": ITEM,
    "indexes/works.ttl": INDEX,
}

# Every literal and object kind of the mapping the main repository lacks.
KINDS = """\
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
<urn:uuid:works/1> a melod:Work ;
  lrmoo:R3_is_realised_in <urn:uuid:expressions/2> ;
  schema:keywords "nocturne"^^xsd:string ;
  melod:isAdLib true ;
  melod:hasExtent ( "a" 2 ) ;
  schema:dateModified "2020-01-01"^^xsd:date ;
  schema:version "1"^^<http://example.org/types/version> ;
  melod:hasClassification <http://example.org/terms/nocturne> ;
  schema:locationCreated <urn:uuid:places/6> ;
  schema:contributor <urn:uuid:persons/7> , <urn:uuid:persons/8> , <urn:uuid:persons/9> , <urn:uuid:persons/11> ;
  schema:citation <urn:uuid:bibliography/10> .
<urn:uuid:places/6> a melod:Place ; skos:prefLabel "Paris" ; schema:name "Paris, France" .
<urn:uuid:persons/7> a melod:Person ; <http://schema.org/name> "Claude Debussy" .
<urn:uuid:persons/8> a melod:Person ; schema:name "Paul Verlaine" .
<urn:uuid:persons/9> a melod:Person ; rdfs:label "Ernest Guiraud" ; skos:prefLabel "Guiraud" .
<urn:uuid:persons/11> a melod:Person ; schema:givenName "Achille" .
<urn:uuid:bibliography/10> owl:sameAs <https://d-nb.info/gnd/119018934> .
"""

KINDS_EXPRESSION = """\
<urn:uuid:expressions/2> a melod:Expression ; rdfs:label "Clair de lune, piano" .
"""

CYCLE = """\
<urn:uuid:works/1> a melod:Work ; melod:hasTitle _:title .
_:title a melod:Title ; rdfs:seeAlso _:title .
"""

DATE = "2026-11-19"
EDITOR = "https://example.org/people/editor"

WORK_NODE = {
    "id": "urn:uuid:works/1",
    "type": ["melod:Work"],
    "dcterms:conformsTo": [{"id": "melod:Work"}],
    "lrmoo:R3_is_realised_in": [
        {"id": "urn:uuid:expressions/2", "label": "Clair de lune, piano"}
    ],
    "melod:hasIdentifier": [
        {
            "type": ["melod:Identifier"],
            "owl:hasValue": ["1"],
            "rdfs:label": ["Test catalogue"],
        }
    ],
    "melod:hasTitle": [
        {
            "type": ["melod:Title"],
            "melod:hasTitleType": [{"id": "melodTerms:MainTitle"}],
            "rdfs:label": [{"@value": "Clair de lune", "@language": "fr"}],
        }
    ],
    "owl:sameAs": [{"id": "wd:Q29117946"}],
    "schema:creativeWorkStatus": ["authentic"],
}

DEBUSSY = {
    "id": "urn:uuid:persons/3",
    "label": "Claude Debussy",
    "owl:sameAs": [{"id": "wd:Q4700"}],
}


def expression_node(agent: dict[str, Any]) -> dict[str, Any]:
    """The node of ``expressions/2``, its agent written as *agent*."""
    return {
        "id": "urn:uuid:expressions/2",
        "type": ["melod:Expression"],
        "melod:hasContribution": [
            {
                "type": ["melod:Contribution"],
                "melod:hasAgent": [agent],
                "melod:hasRole": [{"id": "melodTerms:Composer"}],
            }
        ],
        "melod:hasDuration": [{"@value": "297.5", "@type": "xsd:decimal"}],
        "melod:hasMeter": [
            {
                "type": ["melod:Meter"],
                "melod:hasMeterCount": [9],
                "melod:inMeterUnit": [8],
            }
        ],
        "rdfs:label": ["Clair de lune, piano"],
        "schema:dateCreated": [
            {
                "type": ["melod:Date"],
                "melod:notAfter": ["1905"],
                "melod:notBefore": ["1890"],
            }
        ],
    }


TITLE = {"type": ["melod:Title"], "rdfs:label": ["Suite bergamasque, first edition"]}


def item_node(manifestation: str) -> dict[str, Any]:
    """The node of ``items/5``, exemplifying the element *manifestation*."""
    return {
        "id": "urn:uuid:items/5",
        "type": ["melod:Item"],
        "lrmoo:R7_exemplifies": [{"id": f"#{manifestation}"}],
        "melod:hasShelfmark": ["TEST 1"],
        "rdfs:label": ["Copy 1"],
    }


def manifestation_statements(item: dict[str, Any]) -> dict[str, list[Any]]:
    """The statements of the element standing for ``manifestations/4``."""
    return {
        "type": ["melod:Manifestation"],
        "owl:sameAs": [{"id": "urn:uuid:manifestations/4"}],
        "lrmoo:R4_embodies": [expression_node(DEBUSSY)],
        "melod:hasTitle": [TITLE],
        "lrmoo:R7i_is_exemplified_by": [item],
    }


def C(property: str, value: Any, *basis: str) -> dict[str, Any]:
    """The entry of a pair the import states: how it was made, and no belief."""
    return {
        "property": property,
        "value": value,
        "stated_by": [LIBRARY_AGENT],
        "date": DATE,
        "method": "imported",
        "basis": list(basis),
    }


def manifestation_provenance(item: dict[str, Any]) -> list[dict[str, Any]]:
    m4 = "manifestations/4.ttl"
    return [
        C("rdf:type", {"id": "melod:Manifestation"}, m4),
        C("owl:sameAs", {"id": "urn:uuid:manifestations/4"}, m4),
        C(
            "lrmoo:R4_embodies",
            {"id": "urn:uuid:expressions/2"},
            m4,
            "expressions/2.ttl",
        ),
        C("melod:hasTitle", TITLE, m4),
        C("lrmoo:R7i_is_exemplified_by", item, "items/5.ttl"),
    ]


def write(folder: Path, files: dict[str, str]) -> Path:
    """Write each file under *folder*, MeLODy's prefixes first."""
    for relative, text in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(PREFIXES + text, encoding="utf-8")
    return folder


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return write(tmp_path / "repo", REPOSITORY)


@pytest.fixture
def two_works(tmp_path: Path) -> Path:
    return write(tmp_path / "repo", {**REPOSITORY, "works/9.ttl": PART})


@pytest.fixture
def layerit() -> AlignmentBundle:
    """A bundle holding ``clt_6`` (quarters, 27) with ``region_1`` (0 to 12)."""
    bundle = AlignmentBundle(id="layerit")
    score = ContinuousLogicalTimeline(length=Fraction(27), uid="clt_6")
    score.create_region(0, 12, id="region_1")
    bundle.add_timeline(score)
    return bundle


def prefixes_of(manifest: dict[str, Any]) -> list[str]:
    format_terms = timetoshare.FORMAT_CONTEXT
    return sorted(key for key in manifest["@context"] if key not in format_terms)


def assert_valid_round_trip(bundle: AlignmentBundle) -> dict[str, Any]:
    written = bundle.to_dict()
    timetoshare.validate(written)
    assert timetoshare.manifest_warnings(written) == []
    assert AlignmentBundle.from_dict(written).to_dict() == written
    return written


class TestRepository:
    def test_the_entity_files_are_read_and_the_others_are_not(self, repo: Path) -> None:
        loader = MelodyLoader.from_file(repo, seed=0)
        assert [relative for relative, _ in loader.records[0].files] == [
            "expressions/2.ttl",
            "items/5.ttl",
            "manifestations/4.ttl",
            "persons/3.ttl",
            "works/1.ttl",
        ]

    def test_the_bundle_is_about_the_work(self, repo: Path) -> None:
        bundle = MelodyLoader.from_file(repo, seed=0).create_bundle(
            id="clair_de_lune", date=DATE
        )
        assert bundle.id == "clair_de_lune"
        assert bundle.name is None
        assert bundle.groups == {}
        assert bundle.statements == {"schema:about": [WORK_NODE]}
        assert bundle.provenance == [
            C("schema:about", {"id": "urn:uuid:works/1"}, "works/1.ttl")
        ]

    def test_the_manifestation_is_a_timeline_and_the_rest_values(
        self, repo: Path
    ) -> None:
        bundle = MelodyLoader.from_file(repo, seed=0).create_bundle(
            id="clair_de_lune", date=DATE
        )
        assert list(bundle.timelines) == [NEW[0]]
        edition = bundle.timelines[NEW[0]]
        assert edition.unit == TimeUnit.millimeters
        assert edition.length is None
        assert edition.name == "Suite bergamasque, first edition"
        assert edition.statements == manifestation_statements(item_node(NEW[0]))
        assert edition.provenance == manifestation_provenance(
            {"id": "urn:uuid:items/5"}
        )

    def test_the_manifest_validates_and_round_trips(self, repo: Path) -> None:
        bundle = MelodyLoader.from_file(repo, seed=0).create_bundle(
            id="clair_de_lune", date=DATE
        )
        written = assert_valid_round_trip(bundle)
        assert prefixes_of(written) == [
            "dcterms",
            "lrmoo",
            "melod",
            "melodTerms",
            "owl",
            "rdf",
            "rdfs",
            "schema",
            "wd",
            "xsd",
        ]

    def test_without_id_the_bundle_id_is_minted_and_not_written(
        self, repo: Path
    ) -> None:
        bundle = MelodyLoader.from_file(repo, seed=0).create_bundle(date=DATE)
        assert bundle.id == minted(0, "bundle")[0]
        assert "id" not in bundle.to_dict()

    def test_importing_again_adds_nothing(self, repo: Path) -> None:
        loader = MelodyLoader.from_file(repo, seed=0)
        bundle = loader.create_bundle(id="clair_de_lune", date=DATE)
        written = bundle.to_dict()
        again = loader.create_bundle(
            into=bundle,
            subjects={"urn:uuid:manifestations/4": NEW[0]},
            date="2026-11-21",
        )
        assert again is bundle
        assert bundle.to_dict() == written

    def test_a_pair_believed_false_keeps_its_belief(self, repo: Path) -> None:
        loader = MelodyLoader.from_file(repo, seed=0)
        bundle = loader.create_bundle(id="clair_de_lune", date=DATE)
        edition = bundle.timelines[NEW[0]]
        edition.add_belief(
            "melod:hasTitle", TITLE, actor=EDITOR, belief="false", date="2026-11-20"
        )
        written = bundle.to_dict()
        loader.create_bundle(
            into=bundle,
            subjects={"urn:uuid:manifestations/4": NEW[0]},
            date="2026-11-21",
        )
        assert bundle.to_dict() == written
        assert edition.statements == manifestation_statements(item_node(NEW[0]))
        provenance = manifestation_provenance({"id": "urn:uuid:items/5"})
        believed = {"actor": EDITOR, "belief": "false", "date": "2026-11-20"}
        assert edition.provenance == [
            *provenance[:3],
            {**provenance[3], "beliefs": [believed]},
            provenance[4],
        ]
        assert edition.get_provenance(believer=EDITOR) == [
            *provenance[:3],
            provenance[4],
        ]


class TestMappedElements:
    def test_a_manifestation_and_an_item_land_on_existing_elements(
        self, two_works: Path, layerit: AlignmentBundle
    ) -> None:
        name = layerit.timelines["clt_6"].name
        result = MelodyLoader.from_file(two_works, seed=0).create_bundle(
            work="urn:uuid:works/1",
            into=layerit,
            subjects={
                "urn:uuid:manifestations/4": "clt_6",
                "urn:uuid:items/5": "region_1",
            },
            date=DATE,
        )
        assert result is layerit
        assert list(layerit.timelines) == ["clt_6"]
        assert layerit.statements == {"schema:about": [WORK_NODE]}
        score = layerit.timelines["clt_6"]
        assert score.name == name
        assert score.statements == manifestation_statements({"id": "#region_1"})
        assert score.provenance == manifestation_provenance({"id": "#region_1"})
        region = score.get_region("region_1")
        assert region.statements == {
            "type": ["melod:Item"],
            "owl:sameAs": [{"id": "urn:uuid:items/5"}],
            "lrmoo:R7_exemplifies": [{"id": "#clt_6"}],
            "melod:hasShelfmark": ["TEST 1"],
            "rdfs:label": ["Copy 1"],
        }
        i5 = "items/5.ttl"
        assert region.provenance == [
            C("rdf:type", {"id": "melod:Item"}, i5),
            C("owl:sameAs", {"id": "urn:uuid:items/5"}, i5),
            C("lrmoo:R7_exemplifies", {"id": "#clt_6"}, i5),
            C("melod:hasShelfmark", "TEST 1", i5),
            C("rdfs:label", "Copy 1", i5),
        ]
        assert_valid_round_trip(layerit)


class TestErrors:
    @pytest.mark.parametrize(
        ("arguments", "message"),
        [
            (
                {},
                "the graph holds 2 works: give work, one of 'urn:uuid:works/1', "
                "'urn:uuid:works/9'",
            ),
            (
                {"work": "urn:uuid:persons/3"},
                "'urn:uuid:persons/3' is no melod:Work of the graph",
            ),
            (
                {"id": "other"},
                "id names a new bundle; the bundle given as into keeps its own",
            ),
            (
                {"subjects": {"urn:uuid:nothing": "clt_6"}},
                "subjects names 'urn:uuid:nothing', which is no entity of the graph",
            ),
            (
                {"subjects": {"urn:uuid:works/9": "region_1"}},
                "subjects maps 'urn:uuid:works/9', a work; works and expressions are "
                "values, never elements",
            ),
            (
                {"subjects": {"urn:uuid:expressions/2": "clt_6"}},
                "subjects maps 'urn:uuid:expressions/2', an expression; works and "
                "expressions are values, never elements",
            ),
            (
                {"subjects": {"urn:uuid:persons/3": "clt_6"}},
                "subjects maps 'urn:uuid:persons/3', which is no manifestation or item "
                "of the work 'urn:uuid:works/1'",
            ),
            (
                {"subjects": {"urn:uuid:items/5": "nowhere"}},
                "subjects maps 'urn:uuid:items/5' to 'nowhere', which is no timeline or "
                "region of bundle 'layerit'",
            ),
        ],
    )
    def test_into_is_left_unchanged(
        self,
        two_works: Path,
        layerit: AlignmentBundle,
        arguments: dict[str, Any],
        message: str,
    ) -> None:
        if arguments and "work" not in arguments:
            arguments = {"work": "urn:uuid:works/1", **arguments}
        before = layerit.to_dict()
        loader = MelodyLoader.from_file(two_works, seed=0)
        with pytest.raises(ValueError) as raised:
            loader.create_bundle(into=layerit, date=DATE, **arguments)
        assert str(raised.value) == message
        assert layerit.to_dict() == before

    def test_subjects_needs_into(self, repo: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MelodyLoader.from_file(repo, seed=0).create_bundle(
                subjects={"urn:uuid:items/5": "region_1"}
            )
        assert str(raised.value) == (
            "subjects needs into: a new bundle holds no element to map to"
        )

    def test_a_graph_without_a_work(self, repo: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MelodyLoader.from_file(repo / "persons" / "3.ttl", seed=0).create_bundle()
        assert str(raised.value) == "the graph holds no melod:Work"

    def test_a_blank_node_containing_itself(
        self, tmp_path: Path, layerit: AlignmentBundle
    ) -> None:
        folder = write(tmp_path / "repo", {"works/1.ttl": CYCLE})
        before = layerit.to_dict()
        with pytest.raises(ValueError) as raised:
            MelodyLoader.from_file(folder, seed=0).create_bundle(
                into=layerit, date=DATE
            )
        assert str(raised.value) == "a blank node of works/1.ttl contains itself"
        assert layerit.to_dict() == before

    def test_a_refused_pair_changes_nothing(
        self, tmp_path: Path, layerit: AlignmentBundle
    ) -> None:
        undeclared = MANIFESTATION.replace(
            " .\n", " ;\n  owl:sameAs <doi:10.1000/x> .\n"
        )
        folder = write(
            tmp_path / "repo", {**REPOSITORY, "manifestations/4.ttl": undeclared}
        )
        before = layerit.to_dict()
        with pytest.raises(ValueError) as raised:
            MelodyLoader.from_file(folder, seed=0).create_bundle(
                into=layerit, date=DATE
            )
        assert str(raised.value) == (
            "the prefix 'doi' is declared neither by the catalogue nor on the new "
            "timeline of manifestation 'urn:uuid:manifestations/4'"
        )
        assert list(layerit.timelines) == ["clt_6"]
        assert layerit.to_dict() == before

    def test_a_file_that_is_no_turtle(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.ttl"
        path.write_text(PREFIXES + "<urn:uuid:works/1> a .\n", encoding="utf-8")
        with pytest.raises(Exception) as oracle:
            rdflib.Graph().parse(path, format="turtle")
        with pytest.raises(ValueError) as raised:
            MelodyLoader.from_file(path, seed=0)
        assert str(raised.value) == f"{path}: {oracle.value}"

    def test_rdflib_missing(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "rdflib", None)
        with pytest.raises(ImportError) as raised:
            MelodyLoader.from_file(repo, seed=0)
        assert str(raised.value) == (
            'MelodyLoader needs rdflib: pip install "timetoalign[rdf]"'
        )


class TestKinds:
    def test_files_given_directly(self, repo: Path) -> None:
        bundle = MelodyLoader.from_file(
            repo / "works" / "1.ttl", repo / "expressions" / "2.ttl", seed=0
        ).create_bundle(date=DATE)
        unembodied = {
            **WORK_NODE,
            "lrmoo:R3_is_realised_in": [expression_node({"id": "urn:uuid:persons/3"})],
        }
        assert list(bundle.timelines) == []
        assert bundle.statements == {"schema:about": [unembodied]}
        assert bundle.provenance == [
            C("schema:about", {"id": "urn:uuid:works/1"}, "1.ttl", "2.ttl")
        ]

    def test_two_files_of_one_name_given_directly(self, tmp_path: Path) -> None:
        first = write(tmp_path / "a", {"x.ttl": WORK}) / "x.ttl"
        second = write(tmp_path / "b", {"x.ttl": EXPRESSION + MANIFESTATION + PERSON})
        loader = MelodyLoader.from_file(first, second / "x.ttl", seed=0)
        assert [[name for name, _ in record.files] for record in loader.records] == [
            ["x.ttl"],
            ["x.ttl"],
        ]
        bundle = loader.create_bundle(id="clair_de_lune", date=DATE)
        assert list(bundle.timelines) == [NEW[0]]
        assert bundle.statements == {"schema:about": [WORK_NODE]}
        assert bundle.provenance == [
            C("schema:about", {"id": "urn:uuid:works/1"}, "x.ttl")
        ]
        edition = bundle.timelines[NEW[0]]
        assert edition.statements == {
            "type": ["melod:Manifestation"],
            "owl:sameAs": [{"id": "urn:uuid:manifestations/4"}],
            "lrmoo:R4_embodies": [expression_node(DEBUSSY)],
            "melod:hasTitle": [TITLE],
        }
        assert edition.provenance == [
            C("rdf:type", {"id": "melod:Manifestation"}, "x.ttl"),
            C("owl:sameAs", {"id": "urn:uuid:manifestations/4"}, "x.ttl"),
            C("lrmoo:R4_embodies", {"id": "urn:uuid:expressions/2"}, "x.ttl"),
            C("melod:hasTitle", TITLE, "x.ttl"),
        ]

    @pytest.mark.parametrize(
        ("base", "basis"),
        [
            ("", ["works/nocturne%201.ttl", "expressions/2.ttl"]),
            (
                "https://example.org/repo/",
                [
                    "https://example.org/repo/works/nocturne%201.ttl",
                    "https://example.org/repo/expressions/2.ttl",
                ],
            ),
        ],
    )
    def test_every_kind_of_value(
        self, tmp_path: Path, base: str, basis: list[str]
    ) -> None:
        folder = write(
            tmp_path / "repo",
            {"works/nocturne 1.ttl": KINDS, "expressions/2.ttl": KINDS_EXPRESSION},
        )
        bundle = MelodyLoader.from_file(folder, seed=0).create_bundle(
            id="nocturne", base=base, date=DATE
        )
        assert bundle.statements == {
            "schema:about": [
                {
                    "id": "urn:uuid:works/1",
                    "type": ["melod:Work"],
                    "lrmoo:R3_is_realised_in": [
                        {
                            "id": "urn:uuid:expressions/2",
                            "type": ["melod:Expression"],
                            "rdfs:label": ["Clair de lune, piano"],
                        }
                    ],
                    "melod:hasClassification": [
                        {"id": "http://example.org/terms/nocturne"}
                    ],
                    "melod:hasExtent": [{"list": ["a", 2]}],
                    "melod:isAdLib": [True],
                    "schema:citation": [
                        {
                            "id": "urn:uuid:bibliography/10",
                            "owl:sameAs": [{"id": "gnd:119018934"}],
                        }
                    ],
                    "schema:contributor": [
                        {"id": "urn:uuid:persons/11"},
                        {"id": "urn:uuid:persons/7", "label": "Claude Debussy"},
                        {"id": "urn:uuid:persons/8", "label": "Paul Verlaine"},
                        {"id": "urn:uuid:persons/9", "label": "Ernest Guiraud"},
                    ],
                    "schema:dateModified": [
                        {"@value": "2020-01-01", "@type": "xsd:date"}
                    ],
                    "schema:keywords": ["nocturne"],
                    "schema:locationCreated": [
                        {"id": "urn:uuid:places/6", "label": "Paris"}
                    ],
                    "schema:version": [
                        {"@value": "1", "@type": "http://example.org/types/version"}
                    ],
                }
            ]
        }
        assert bundle.provenance == [
            {**C("schema:about", {"id": "urn:uuid:works/1"}), "basis": basis}
        ]
        assert_valid_round_trip(bundle)
