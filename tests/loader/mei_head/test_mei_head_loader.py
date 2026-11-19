"""The MEI header import: what each header element becomes, and where it lands.

Every expectation is written from the mapping (see README.md), never from a
run; whole statements and provenance lists are compared for equality.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.loader import MeiHeadLoader
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    Region,
)

DATE = "2026-11-19"
AGENT = "https://timetoalign.github.io/"

CNW0133 = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="5.1">
 <meiHead>
  <fileDesc><titleStmt><title>Hr. Olufs Sang</title></titleStmt><pubStmt/></fileDesc>
  <workList>
   <work xml:id="work_1">
    <identifier xml:id="identifier_1" label="CNW">133</identifier>
    <title xml:id="title_1" type="main" xml:lang="da">Hr. Olufs Sang</title>
    <title xml:id="title_2" type="alternative" xml:lang="en">Sir Oluf's Song</title>
    <composer><persName xml:id="persName_1" role="composer">Carl Nielsen</persName></composer>
    <lyricist><persName xml:id="persName_2"
      auth.uri="https://d-nb.info/gnd/" codedval="119018934">Holger Drachmann</persName></lyricist>
    <creation><date xml:id="date_1" isodate="1906">1906</date></creation>
    <expressionList>
     <expression xml:id="expression_1">
      <title xml:id="title_3">Version for voice and piano</title>
      <key xml:id="key_1" pname="f" accid="s" mode="minor"/>
      <meter xml:id="meter_1" sym="common"/>
      <perfMedium><perfResList>
       <perfRes xml:id="perfRes_1" codedval="vn">voice</perfRes>
       <perfRes xml:id="perfRes_2" codedval="ka">pf.</perfRes>
      </perfResList></perfMedium>
     </expression>
    </expressionList>
    <relationList><relation xml:id="relation_1" rel="isArrangementOf" target="cnw0007.xml"/></relationList>
   </work>
  </workList>
  <manifestationList>
   <manifestation xml:id="source_1">
    <identifier xml:id="identifier_2" label="CNU Source">A</identifier>
    <titleStmt><title xml:id="title_4">Score, print</title></titleStmt>
    <relationList><relation xml:id="relation_2" rel="isEmbodimentOf" target="#expression_1"/></relationList>
   </manifestation>
   <manifestation xml:id="source_2" sameas="https://example.org/sources/autograph">
    <titleStmt><title xml:id="title_5" type="main">Score, autograph</title></titleStmt>
    <itemList>
     <item xml:id="item_1">
      <physLoc>
       <repository><corpName xml:id="corpName_1">DK-Kk</corpName></repository>
       <identifier xml:id="identifier_3">CNS 342e</identifier>
      </physLoc>
     </item>
    </itemList>
   </manifestation>
  </manifestationList>
 </meiHead>
 <music/>
</mei>
"""

ANDANTE = """<?xml version="1.0" encoding="UTF-8"?>
<mei xmlns="http://www.music-encoding.org/ns/mei" meiversion="4.0.1">
 <meiHead>
  <workList>
   <work xml:id="work_1">
    <titleStmt>
     <title xml:id="title_1">Andante</title>
     <respStmt><persName xml:id="persName_1" role="dedicatee">Anna</persName></respStmt>
    </titleStmt>
    <composer xml:id="composer_1">Anonymous</composer>
    <creation><date xml:id="date_1" notbefore="1890" notafter="1895">c. 1890</date></creation>
    <expressionList>
     <expression xml:id="expression_1">
      <title xml:id="title_2" xml:lang="de">Fassung für Orgel</title>
      <contributor><persName xml:id="persName_2" role="organ builder"
        sameas="http://viaf.org/viaf/1">Orgelbau X</persName></contributor>
     </expression>
    </expressionList>
   </work>
  </workList>
  <manifestationList>
   <manifestation xml:id="source_1" singleton="true">
    <titleStmt><title xml:id="title_3">Autograph</title></titleStmt>
    <physLoc>
     <repository><corpName xml:id="corpName_1"
       auth.uri="http://viaf.org/viaf/" codedval="123">Bibliothek</corpName></repository>
     <identifier xml:id="identifier_1">Mus. ms. 1</identifier>
    </physLoc>
    <relationList><relation xml:id="relation_1" rel="hasPart" target="https://example.org/parts/1"/></relationList>
   </manifestation>
  </manifestationList>
 </meiHead>
</mei>
"""

TWO_WORKS = """<mei xmlns="http://www.music-encoding.org/ns/mei"><meiHead><workList>
 <work xml:id="work_1"/><work xml:id="work_2"/>
</workList></meiHead></mei>
"""


# region Expectations, written from the mapping


def entry(
    status: str, property: str, value: Any, basis: str, note: str | None = None
) -> dict[str, Any]:
    """A provenance entry of the import."""
    result = {
        "property": property,
        "value": value,
        "status": status,
        "stated_by": [AGENT],
        "date": DATE,
        "method": "imported",
        "basis": [basis],
    }
    if note is not None:
        result["note"] = note
    return result


def confirmed(
    property: str, value: Any, xml_id: str | None, base: str = "cnw0133.xml", **kw: Any
) -> dict[str, Any]:
    """C(p, v, x): a confirmed entry whose basis is the element *xml_id* of *base*."""
    return entry(
        "confirmed",
        property,
        value,
        base if xml_id is None else f"{base}#{xml_id}",
        **kw,
    )


def proposed(
    property: str,
    value: Any,
    xml_id: str,
    base: str = "cnw0133.xml",
    note: str = "unreconciled",
) -> dict[str, Any]:
    """P(p, v, x): a proposal of an unreconciled name."""
    return entry("proposed", property, value, f"{base}#{xml_id}", note=note)


def title(text: str, language: str | None, title_type: str | None) -> dict[str, Any]:
    """T(text, lang, type): MeLODy's title node."""
    label: Any = {"@value": text, "@language": language} if language else text
    node: dict[str, Any] = {"type": ["melod:Title"], "rdfs:label": [label]}
    if title_type is not None:
        node["melod:hasTitleType"] = [{"id": f"melodTerms:{title_type}"}]
    return node


WORK = "cnw0133.xml#work_1"
EXPRESSION = "cnw0133.xml#expression_1"
CNW_IDENTIFIER = {
    "type": ["melod:Identifier"],
    "rdfs:label": ["CNW"],
    "owl:hasValue": ["133"],
}
CREATED = {
    "type": ["melod:ExactDate"],
    "melod:isodate": [{"@value": "1906", "@type": "xsd:gYear"}],
}
KEY = {
    "type": ["melod:Key"],
    "melod:hasPitch": [{"id": "melodTerms:F"}],
    "melod:hasAccidental": [{"id": "melodTerms:Sharp"}],
    "melod:isInMode": [{"id": "melodTerms:Minor"}],
}
METER = {"type": ["melod:Meter"], "melod:usesMeterSymbol": ["common"]}
VOICE = {"rdfs:label": ["voice"], "skos:notation": ["vn"]}
PIANO = {"rdfs:label": ["pf."], "skos:notation": ["ka"]}
MEDIUM = "mus:U2_foresees_use_of_medium_of_performance"
SOURCE_IDENTIFIER = {
    "type": ["melod:Identifier"],
    "rdfs:label": ["CNU Source"],
    "owl:hasValue": ["A"],
}
ITEM = {"type": ["melod:Item"], "melod:hasShelfmark": ["CNS 342e"]}
KEEPER = "cidoc:P50_has_current_keeper"


def work_node(
    base: str = "cnw0133.xml", related: str = "cnw0007.xml"
) -> dict[str, Any]:
    """The work under the bundle's schema:about: its pairs and their provenance."""
    work, expression = f"{base}#work_1", f"{base}#expression_1"
    return {
        "id": work,
        "type": ["melod:Work"],
        "melod:hasTitle": [
            title("Hr. Olufs Sang", "da", "MainTitle"),
            title("Sir Oluf's Song", "en", "AlternativeTitle"),
        ],
        "melod:hasIdentifier": [CNW_IDENTIFIER],
        "rel:lyr": [{"id": "gnd:119018934", "label": "Holger Drachmann"}],
        "schema:dateCreated": [CREATED],
        "lrmoo:R3_is_realised_in": [{"id": expression}],
        "melod:isArrangementOf": [{"id": related}],
        "dcterms:source": [{"id": related}],
        "provenance": [
            confirmed("rdf:type", {"id": "melod:Work"}, "work_1", base),
            confirmed(
                "melod:hasTitle",
                title("Hr. Olufs Sang", "da", "MainTitle"),
                "title_1",
                base,
            ),
            confirmed(
                "melod:hasTitle",
                title("Sir Oluf's Song", "en", "AlternativeTitle"),
                "title_2",
                base,
            ),
            confirmed("melod:hasIdentifier", CNW_IDENTIFIER, "identifier_1", base),
            proposed("rel:cmp", {"label": "Carl Nielsen"}, "persName_1", base),
            confirmed("rel:lyr", {"id": "gnd:119018934"}, "persName_2", base),
            confirmed("schema:dateCreated", CREATED, "date_1", base),
            confirmed(
                "lrmoo:R3_is_realised_in", {"id": expression}, "expression_1", base
            ),
            confirmed("melod:isArrangementOf", {"id": related}, "relation_1", base),
            confirmed("dcterms:source", {"id": related}, "relation_1", base),
        ],
    }


def expression_node(base: str = "cnw0133.xml") -> dict[str, Any]:
    """The expression, in full under the manifestation embodying it."""
    return {
        "id": f"{base}#expression_1",
        "type": ["melod:Expression"],
        "rdfs:label": ["Version for voice and piano"],
        "mus:U11_has_key": [KEY],
        "melod:hasMeter": [METER],
        MEDIUM: [VOICE, PIANO],
        "provenance": [
            confirmed("rdf:type", {"id": "melod:Expression"}, "expression_1", base),
            confirmed("rdfs:label", "Version for voice and piano", "title_3", base),
            confirmed("mus:U11_has_key", KEY, "key_1", base),
            confirmed("melod:hasMeter", METER, "meter_1", base),
            confirmed(MEDIUM, VOICE, "perfRes_1", base),
            confirmed(MEDIUM, PIANO, "perfRes_2", base),
        ],
    }


def print_statements(base: str = "cnw0133.xml") -> dict[str, Any]:
    """source_1: a manifestation embodying the expression."""
    return {
        "type": ["melod:Manifestation"],
        "melod:hasTitle": [title("Score, print", None, None)],
        "melod:hasIdentifier": [SOURCE_IDENTIFIER],
        "lrmoo:R4_embodies": [expression_node(base)],
    }


def print_provenance(base: str = "cnw0133.xml") -> list[dict[str, Any]]:
    return [
        confirmed("rdf:type", {"id": "melod:Manifestation"}, "source_1", base),
        confirmed("melod:hasTitle", title("Score, print", None, None), "title_4", base),
        confirmed("melod:hasIdentifier", SOURCE_IDENTIFIER, "identifier_2", base),
        confirmed(
            "lrmoo:R4_embodies", {"id": f"{base}#expression_1"}, "relation_2", base
        ),
    ]


AUTOGRAPH_STATEMENTS = {
    "type": ["melod:Manifestation"],
    "melod:hasTitle": [title("Score, autograph", None, "MainTitle")],
    "owl:sameAs": [{"id": "https://example.org/sources/autograph"}],
    "lrmoo:R7i_is_exemplified_by": [
        {**ITEM, "provenance": [proposed(KEEPER, {"label": "DK-Kk"}, "corpName_1")]}
    ],
}
AUTOGRAPH_PROVENANCE = [
    confirmed("rdf:type", {"id": "melod:Manifestation"}, "source_2"),
    confirmed(
        "melod:hasTitle", title("Score, autograph", None, "MainTitle"), "title_5"
    ),
    confirmed(
        "owl:sameAs", {"id": "https://example.org/sources/autograph"}, "source_2"
    ),
    confirmed("lrmoo:R7i_is_exemplified_by", ITEM, "item_1"),
]

PREFIXES = "cidoc dcterms gnd lrmoo melod melodTerms mus owl rdf rdfs rel schema skos xsd".split()


# endregion


@pytest.fixture
def cnw0133(tmp_path: Path) -> Path:
    path = tmp_path / "cnw0133.xml"
    path.write_text(CNW0133, encoding="utf-8")
    return path


@pytest.fixture
def andante(tmp_path: Path) -> Path:
    path = tmp_path / "andante.xml"
    path.write_text(ANDANTE, encoding="utf-8")
    return path


def layerit() -> AlignmentBundle:
    """A bundle with a score and a scan (holding a region), to import into."""
    bundle = AlignmentBundle(id="layerit")
    bundle.add_timeline(ContinuousLogicalTimeline(length=Fraction(27), uid="clt_6"))
    scan = ContinuousGraphicalTimeline(
        length=None, unit=TimeUnit.millimeters, uid="cgt_6"
    )
    scan.add_region(
        Region(
            Coordinate(0.0, TimeUnit.millimeters),
            Coordinate(10.0, TimeUnit.millimeters),
            id="region_1",
        )
    )
    bundle.add_timeline(scan)
    return bundle


def context_prefixes(manifest: dict[str, Any]) -> list[str]:
    return sorted(set(manifest["@context"]) - set(timetoshare.FORMAT_CONTEXT))


def assert_written(bundle: AlignmentBundle) -> dict[str, Any]:
    """The manifest written is valid, warning-free, and reads back equal."""
    manifest = bundle.to_dict()
    timetoshare.validate(manifest)
    assert timetoshare.manifest_warnings(manifest) == []
    assert AlignmentBundle.from_dict(manifest).to_dict() == manifest
    return manifest


class TestNewBundle:
    """A header imported into a new bundle (rows a–l, MEI 5.1)."""

    def test_bundle(self, cnw0133: Path) -> None:
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(date=DATE)
        assert bundle.id == "cnw0133"
        assert bundle.name is None
        assert bundle.timeline_ids == ["cgt_1", "cgt_2"]
        assert bundle.groups == {}
        assert bundle.statements == {
            "rdfs:seeAlso": [{"id": "cnw0133.xml"}],
            "schema:about": [work_node()],
        }
        assert bundle.provenance == [
            confirmed("schema:about", {"id": WORK}, "work_1"),
            confirmed("rdfs:seeAlso", {"id": "cnw0133.xml"}, None),
        ]

    def test_manifestations(self, cnw0133: Path) -> None:
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(date=DATE)
        printed, autograph = bundle.get_timeline("cgt_1"), bundle.get_timeline("cgt_2")
        for timeline, name in (
            (printed, "Score, print"),
            (autograph, "Score, autograph"),
        ):
            assert type(timeline) is ContinuousGraphicalTimeline
            assert timeline.unit is TimeUnit.millimeters
            assert timeline.length is None
            assert timeline.name == name
        assert printed.statements == print_statements()
        assert printed.provenance == print_provenance()
        assert autograph.statements == AUTOGRAPH_STATEMENTS
        assert autograph.provenance == AUTOGRAPH_PROVENANCE

    def test_manifest_written(self, cnw0133: Path) -> None:
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(date=DATE)
        manifest = assert_written(bundle)
        assert context_prefixes(manifest) == sorted(PREFIXES)

    def test_stated_by(self, cnw0133: Path) -> None:
        agents = ["https://example.org/people/editor", "https://example.org/agents/x"]
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(
            stated_by=agents, date=DATE
        )
        assert [e["stated_by"] for e in bundle.provenance] == [agents, agents]
        work = bundle.statements["schema:about"][0]
        assert {tuple(e["stated_by"]) for e in work["provenance"]} == {tuple(agents)}


class TestSecondHeader:
    """MEI 4.0.1, titles in titleStmt, names in other places, a bounded date,
    an expression no manifestation embodies, a singleton manifestation."""

    def test_work(self, andante: Path) -> None:
        bundle = MeiHeadLoader.from_file(andante).create_bundle(date=DATE)
        base = "andante.xml"
        dated = {
            "type": ["melod:Date"],
            "melod:notBefore": [{"@value": "1890", "@type": "xsd:gYear"}],
            "melod:notAfter": [{"@value": "1895", "@type": "xsd:gYear"}],
            "schema:description": ["c. 1890"],
        }
        label = {"@value": "Fassung für Orgel", "@language": "de"}
        builder = {"id": "viaf:1", "label": "Orgelbau X"}
        expression = {
            "id": "andante.xml#expression_1",
            "type": ["melod:Expression"],
            "rdfs:label": [label],
            "rel:ctb": [builder],
            "provenance": [
                confirmed("rdf:type", {"id": "melod:Expression"}, "expression_1", base),
                confirmed("rdfs:label", label, "title_2", base),
                confirmed(
                    "rel:ctb",
                    {"id": "viaf:1"},
                    "persName_2",
                    base,
                    note="role 'organ builder'",
                ),
            ],
        }
        assert bundle.id == "andante"
        assert bundle.statements == {
            "rdfs:seeAlso": [{"id": "andante.xml"}],
            "schema:about": [
                {
                    "id": "andante.xml#work_1",
                    "type": ["melod:Work"],
                    "melod:hasTitle": [title("Andante", None, None)],
                    "schema:dateCreated": [dated],
                    "lrmoo:R3_is_realised_in": [expression],
                    "provenance": [
                        confirmed("rdf:type", {"id": "melod:Work"}, "work_1", base),
                        confirmed(
                            "melod:hasTitle",
                            title("Andante", None, None),
                            "title_1",
                            base,
                        ),
                        proposed("rel:dte", {"label": "Anna"}, "persName_1", base),
                        proposed("rel:cmp", {"label": "Anonymous"}, "composer_1", base),
                        confirmed("schema:dateCreated", dated, "date_1", base),
                        confirmed(
                            "lrmoo:R3_is_realised_in",
                            {"id": "andante.xml#expression_1"},
                            "expression_1",
                            base,
                        ),
                    ],
                }
            ],
        }
        assert bundle.provenance == [
            confirmed("schema:about", {"id": "andante.xml#work_1"}, "work_1", base),
            confirmed("rdfs:seeAlso", {"id": "andante.xml"}, None, base),
        ]

    def test_singleton_manifestation(self, andante: Path) -> None:
        bundle = MeiHeadLoader.from_file(andante).create_bundle(date=DATE)
        base = "andante.xml"
        item = {
            "type": ["melod:Item"],
            "melod:hasShelfmark": ["Mus. ms. 1"],
            KEEPER: [{"id": "viaf:123", "label": "Bibliothek"}],
        }
        part = {"id": "https://example.org/parts/1"}
        assert bundle.timeline_ids == ["cgt_1"]
        autograph = bundle.get_timeline("cgt_1")
        assert autograph.name == "Autograph"
        assert autograph.statements == {
            "type": ["melod:Manifestation"],
            "melod:hasTitle": [title("Autograph", None, None)],
            "lrmoo:R7i_is_exemplified_by": [item],
            "melod:hasManifestationPart": [part],
            "dcterms:hasPart": [part],
        }
        assert autograph.provenance == [
            confirmed("rdf:type", {"id": "melod:Manifestation"}, "source_1", base),
            confirmed(
                "melod:hasTitle", title("Autograph", None, None), "title_3", base
            ),
            confirmed("lrmoo:R7i_is_exemplified_by", item, "source_1", base),
            confirmed("melod:hasManifestationPart", part, "relation_1", base),
            confirmed("dcterms:hasPart", part, "relation_1", base),
        ]

    def test_manifest_written(self, andante: Path) -> None:
        manifest = assert_written(
            MeiHeadLoader.from_file(andante).create_bundle(date=DATE)
        )
        assert context_prefixes(manifest) == sorted(
            "cidoc dcterms lrmoo melod rdf rdfs rel schema viaf xsd".split()
        )


class TestInto:
    """An import into a bundle given, with subjects."""

    def test_manifestation_mapped(self, cnw0133: Path) -> None:
        bundle = layerit()
        loader = MeiHeadLoader.from_file(cnw0133)
        result = loader.create_bundle(
            into=bundle, subjects={"source_1": "cgt_6"}, date=DATE
        )
        assert result is bundle
        assert bundle.timeline_ids == ["clt_6", "cgt_6", "cgt_1"]
        assert bundle.statements == {
            "rdfs:seeAlso": [{"id": "cnw0133.xml"}],
            "schema:about": [work_node()],
        }
        assert bundle.get_timeline("clt_6").statements == {}
        scan = bundle.get_timeline("cgt_6")
        assert scan.statements == print_statements()
        assert scan.provenance == print_provenance()
        assert scan.get_region("region_1").statements == {}
        autograph = bundle.get_timeline("cgt_1")
        assert autograph.name == "Score, autograph"
        assert autograph.statements == AUTOGRAPH_STATEMENTS
        written = assert_written(bundle)

        again = {"source_1": "cgt_6", "source_2": "cgt_1"}
        loader.create_bundle(into=bundle, subjects=again, date="2026-11-20")
        assert bundle.to_dict() == written

    def test_item_mapped_to_region(self, cnw0133: Path) -> None:
        bundle = layerit()
        MeiHeadLoader.from_file(cnw0133).create_bundle(
            into=bundle, subjects={"source_1": "cgt_6", "item_1": "region_1"}, date=DATE
        )
        region = bundle.get_timeline("cgt_6").get_region("region_1")
        assert region.statements == ITEM
        assert region.provenance == [
            confirmed("rdf:type", {"id": "melod:Item"}, "item_1"),
            confirmed("melod:hasShelfmark", "CNS 342e", "identifier_3"),
            proposed(KEEPER, {"label": "DK-Kk"}, "corpName_1"),
        ]
        autograph = bundle.get_timeline("cgt_1")
        assert autograph.statements["lrmoo:R7i_is_exemplified_by"] == [
            {"id": "#region_1"}
        ]
        assert autograph.provenance[-1] == confirmed(
            "lrmoo:R7i_is_exemplified_by", {"id": "#region_1"}, "item_1"
        )
        assert_written(bundle)

    def test_rejection_is_kept(self, cnw0133: Path) -> None:
        loader = MeiHeadLoader.from_file(cnw0133)
        bundle = loader.create_bundle(date=DATE)
        bundle.reject_statement(
            "rel:cmp",
            {"label": "Carl Nielsen"},
            stated_by="https://example.org/people/editor",
            date="2026-11-20",
            note="reconciled by hand elsewhere",
            node=("schema:about", 0),
        )
        rejected = bundle.to_dict()
        loader.create_bundle(
            into=bundle,
            subjects={"source_1": "cgt_1", "source_2": "cgt_2"},
            date="2026-11-21",
        )
        assert bundle.to_dict() == rejected
        statuses = [
            e["status"]
            for e in bundle.get_provenance(node=("schema:about", 0))
            if e["property"] == "rel:cmp"
        ]
        assert statuses == ["rejected"]


class TestBase:
    def test_absolute_base(self, cnw0133: Path) -> None:
        base = "https://www.kb.dk/dcm/cnw/document.xq?doc=cnw0133.xml"
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(base=base, date=DATE)
        related = "https://www.kb.dk/dcm/cnw/cnw0007.xml"
        assert bundle.statements == {
            "rdfs:seeAlso": [{"id": base}],
            "schema:about": [work_node(base, related)],
        }
        assert bundle.provenance == [
            confirmed("schema:about", {"id": f"{base}#work_1"}, "work_1", base),
            confirmed("rdfs:seeAlso", {"id": base}, None, base),
        ]
        assert bundle.get_timeline("cgt_1").statements == print_statements(base)
        assert_written(bundle)

    def test_file_name_with_space(self, tmp_path: Path) -> None:
        path = tmp_path / "clair-de-lune M6-basic.mei"
        path.write_text(CNW0133, encoding="utf-8")
        bundle = MeiHeadLoader.from_file(path).create_bundle(date=DATE)
        assert bundle.id == "clair-de-lune_M6-basic"
        assert bundle.statements["rdfs:seeAlso"] == [
            {"id": "clair-de-lune%20M6-basic.mei"}
        ]
        assert (
            bundle.statements["schema:about"][0]["id"]
            == "clair-de-lune%20M6-basic.mei#work_1"
        )

    def test_id_given(self, cnw0133: Path) -> None:
        bundle = MeiHeadLoader.from_file(cnw0133).create_bundle(id="nielsen", date=DATE)
        assert bundle.id == "nielsen"


class TestRefusals:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {"id": "other"},
                "id names a new bundle; the bundle given as into keeps its own",
            ),
            (
                {"subjects": {"nothing_1": "clt_6"}},
                "subjects names 'nothing_1', which is no work, expression, manifestation "
                "or item of the meiHead",
            ),
            (
                {"subjects": {"work_1": "clt_6"}},
                "subjects maps 'work_1', a work; works and expressions are values, never "
                "elements",
            ),
            (
                {"subjects": {"expression_1": "clt_6"}},
                "subjects maps 'expression_1', an expression; works and expressions are "
                "values, never elements",
            ),
            (
                {"subjects": {"source_1": "clt_9"}},
                "subjects maps 'source_1' to 'clt_9', which is no timeline or region of "
                "bundle 'layerit'",
            ),
        ],
    )
    def test_into_unchanged(
        self, cnw0133: Path, kwargs: dict[str, Any], message: str
    ) -> None:
        bundle = layerit()
        before = bundle.to_dict()
        loader = MeiHeadLoader.from_file(cnw0133)
        with pytest.raises(ValueError) as raised:
            loader.create_bundle(into=bundle, date=DATE, **kwargs)
        assert str(raised.value) == message
        assert bundle.to_dict() == before

    def test_subjects_without_into(self, cnw0133: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MeiHeadLoader.from_file(cnw0133).create_bundle(
                subjects={"source_1": "cgt_1"}
            )
        assert str(raised.value) == (
            "subjects needs into: a new bundle holds no element to map to"
        )

    def test_two_works(self, tmp_path: Path) -> None:
        path = tmp_path / "two.xml"
        path.write_text(TWO_WORKS, encoding="utf-8")
        bundle = layerit()
        before = bundle.to_dict()
        with pytest.raises(ValueError) as raised:
            MeiHeadLoader.from_file(path).create_bundle(into=bundle)
        assert (
            str(raised.value) == "the meiHead holds 2 works; a bundle is about one work"
        )
        assert bundle.to_dict() == before

    def test_no_mei_head(self, tmp_path: Path) -> None:
        path = tmp_path / "score.mei"
        path.write_text(
            '<mei xmlns="http://www.music-encoding.org/ns/mei"><music/></mei>',
            encoding="utf-8",
        )
        with pytest.raises(ValueError) as raised:
            MeiHeadLoader.from_file(path)
        assert str(raised.value) == f"{path} holds no meiHead"

    def test_reads_one_source(self, cnw0133: Path, andante: Path) -> None:
        with pytest.raises(ValueError) as raised:
            MeiHeadLoader().load(cnw0133, andante)
        assert str(raised.value) == (
            f"MeiHeadLoader reads one source; got 2: {cnw0133}, {andante}."
        )
