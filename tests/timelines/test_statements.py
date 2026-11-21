"""Statements and provenance on a bundle, a timeline and a region.

The pairs themselves are TimeToShare's; these tests prove what the library
adds: storage, the three faces agreeing, the prefix rule and the manifest
boundary. The validation logic is documented in ``tests/timelines/README.md``
under "Statements".
"""

from __future__ import annotations

import copy
import datetime
import re
from collections.abc import Callable
from typing import Any

import pytest
import timetoshare
from timetoshare import FORMAT_CONTEXT, load_catalogue

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Timeline,
)
from timetoalign.core.provenance import Provenance
from timetoalign.timelines import statements as statements_module

EDITOR = "https://example.org/people/editor"
REVIEWER = "https://example.org/people/reviewer"
TILIE = "https://example.org/agents/tilie"
FOAF = "http://xmlns.com/foaf/0.1/"
PIRES = {"id": "wd:Q235650", "label": "Maria João Pires"}
MUSICBRAINZ_WORK = "http://musicbrainz.org/work/8d331505-4d88-39ae-81c7-bec5da77af96"


def _raises(message: str, call: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        call()


def _context(*prefixes: str, **declared: str) -> dict[str, Any]:
    """The format's terms with catalogue prefixes and *declared* ones."""
    catalogue = load_catalogue().prefixes
    return {
        **FORMAT_CONTEXT,
        **{prefix: catalogue[prefix] for prefix in prefixes},
        **declared,
    }


# region The three faces


_COPY_ON_WRITE = {
    "add_statement": "with_statement",
    "remove_statement": "without_statement",
    "add_candidate": "with_candidate",
    "add_belief": "with_belief",
    "add_verb": "with_verb",
    "add_prefix": "with_prefix",
}


class _RegionFace:
    """A region driven through the timeline face's names.

    Each change calls the region's copy-on-write method, stores the copy
    with ``replace_region`` and reads it back with ``get_region``.
    """

    def __init__(self) -> None:
        self.timeline = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
        self.timeline.create_region(0.0, 10.0, id="region_1")

    @property
    def region(self) -> Any:
        return self.timeline.get_region("region_1")

    def __getattr__(self, name: str) -> Any:
        if name in _COPY_ON_WRITE:

            def change(*args: Any, **kwargs: Any) -> _RegionFace:
                copied = getattr(self.region, _COPY_ON_WRITE[name])(*args, **kwargs)
                self.timeline.replace_region(copied)
                return self

            return change
        return getattr(self.region, name)


HOLDERS: dict[str, tuple[Callable[[], Any], str]] = {
    "timeline": (
        lambda: ContinuousPhysicalTimeline(length=None, uid="cpt_1"),
        "timeline 'cpt_1'",
    ),
    "bundle": (lambda: AlignmentBundle(id="rec"), "bundle 'rec'"),
    "region": (_RegionFace, "region 'region_1'"),
}


@pytest.fixture(params=sorted(HOLDERS))
def face(request: pytest.FixtureRequest) -> tuple[Any, str]:
    make, name = HOLDERS[request.param]
    return make(), name


def _described(holder: Any) -> Any:
    holder.add_statement("type", "schema:VideoObject")
    holder.add_verb("performed_by", "wikidata", "wd:Q235650", label=PIRES["label"])
    return holder


DESCRIBED = {"type": ["schema:VideoObject"], "wdt:P175": [PIRES]}


def test_pairs_are_stored_as_written(face: tuple[Any, str]) -> None:
    holder, _ = face
    _described(holder)
    assert holder.statements == DESCRIBED
    assert list(holder.statements) == ["type", "wdt:P175"]
    assert holder.provenance == []
    assert holder.prefixes == {}
    assert holder.get_pairs() == [("type", "schema:VideoObject"), ("wdt:P175", PIRES)]


def test_a_pair_stated_already_is_refused_in_the_package_s_words(
    face: tuple[Any, str],
) -> None:
    holder, _ = face
    _described(holder)
    message = "'wdt:P175' already has this value"
    _raises(
        message,
        lambda: holder.add_verb(
            "performed_by", "wikidata", "wd:Q235650", label=PIRES["label"]
        ),
    )
    _raises(
        message,
        lambda: holder.add_statement(
            "wdt:P175", {"id": "http://www.wikidata.org/entity/Q235650"}
        ),
    )
    assert holder.statements == DESCRIBED


def test_an_undeclared_prefix_is_refused_and_nothing_changes(
    face: tuple[Any, str],
) -> None:
    holder, name = face
    _described(holder)
    _raises(
        f"the prefix 'foaf' is declared neither by the catalogue nor on {name}",
        lambda: holder.add_statement("foaf:maker", {"id": "wd:Q4700"}),
    )
    assert holder.statements == DESCRIBED
    assert holder.provenance == []


def test_a_declared_prefix_admits_the_pair(face: tuple[Any, str]) -> None:
    holder, name = face
    holder.add_prefix("foaf", FOAF)
    holder.add_statement("foaf:maker", {"id": "wd:Q4700"})
    assert holder.prefixes == {"foaf": FOAF}
    assert holder.statements == {"foaf:maker": [{"id": "wd:Q4700"}]}
    holder.add_prefix("foaf", FOAF)
    assert holder.prefixes == {"foaf": FOAF}
    _raises(
        f"the prefix 'foaf' already stands for '{FOAF}' on {name}",
        lambda: holder.add_prefix("foaf", "http://example.org/foaf/"),
    )
    assert holder.prefixes == {"foaf": FOAF}


def test_a_prefix_the_format_refuses_is_refused_with_the_package_s_problem(
    face: tuple[Any, str],
) -> None:
    holder, _ = face
    namespace = "http://example.org/unit/"
    problem, *_ = timetoshare.statement_problems(
        {"@context": {**FORMAT_CONTEXT, "unit": namespace}}
    )
    _raises(
        f"cannot declare the prefix 'unit' as '{namespace}': {problem}",
        lambda: holder.add_prefix("unit", namespace),
    )
    assert holder.prefixes == {}


def test_a_candidate_and_a_belief_agree_on_every_face(face: tuple[Any, str]) -> None:
    holder, _ = face
    holder.add_candidate(
        "wdt:P175", PIRES, stated_by=TILIE, method="inferred", date="2026-11-18"
    )
    holder.add_belief(
        "wdt:P175", PIRES, actor=EDITOR, belief="false", date="2026-11-19"
    )
    entry = {
        "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-19"}],
        "date": "2026-11-18",
        "method": "inferred",
        "property": "wdt:P175",
        "stated_by": [TILIE],
        "value": {"id": "wd:Q235650"},
    }
    assert holder.statements == {}
    assert holder.provenance == [entry]
    assert holder.get_candidates() == [entry]
    assert holder.get_candidates(believer=EDITOR) == []


def test_removing_every_pair_leaves_nothing(face: tuple[Any, str]) -> None:
    holder, _ = face
    _described(holder)
    holder.remove_statement("type", "schema:VideoObject")
    holder.remove_statement("wdt:P175", {"id": "wd:Q235650"})
    assert holder.statements == {}
    assert holder.provenance == []


# endregion

# region Provenance


def test_a_stated_pair_s_provenance_is_written_whole() -> None:
    video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    stated = {"stated_by": [EDITOR], "date": "2026-11-18", "method": "manual"}
    video.add_statement("wdt:P175", PIRES, provenance=stated)
    video.add_statement(
        "type",
        "schema:VideoObject",
        provenance=Provenance(stated_by=(EDITOR,), date="2026-11-18", method="manual"),
    )
    assert video.statements == {"type": ["schema:VideoObject"], "wdt:P175": [PIRES]}
    assert video.provenance == [
        {**stated, "property": "wdt:P175", "value": {"id": "wd:Q235650"}},
        {**stated, "property": "rdf:type", "value": {"id": "schema:VideoObject"}},
    ]
    video.remove_statement("wdt:P175", PIRES)
    video.remove_statement("type", "schema:VideoObject")
    assert (video.statements, video.provenance) == ({}, [])


def test_a_candidate_is_offered_believed_and_stated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    work = {"id": "wd:Q29117946"}
    candidate = {
        "basis": ["#cpt_2"],
        "date": "2026-11-18",
        "method": "inferred",
        "property": "wdt:P2550",
        "stated_by": [TILIE],
        "value": work,
    }
    offer = {"stated_by": TILIE, "method": "inferred", "basis": ["#cpt_2"]}
    video.add_candidate("wdt:P2550", work, date="2026-11-18", **offer)
    assert (video.statements, video.provenance) == ({}, [candidate])
    assert video.get_candidates() == [candidate]

    monkeypatch.setattr(
        statements_module, "_today", lambda: datetime.date(2026, 11, 21)
    )
    video.add_candidate("wdt:P2550", work, **offer)
    candidate["date"] = "2026-11-21"
    assert video.provenance == [candidate]

    video.add_belief("wdt:P2550", work, actor=EDITOR, belief="false", date="2026-11-19")
    disbelieved = {
        **candidate,
        "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-19"}],
    }
    assert video.get_candidates() == [disbelieved]
    assert video.get_candidates(believer=EDITOR) == []
    assert video.get_candidates(believer=REVIEWER) == [disbelieved]

    labelled = {**work, "label": "Clair de lune"}
    video.add_statement(
        "wdt:P2550", labelled, actor=EDITOR, date=datetime.date(2026, 11, 20)
    )
    stated = {
        **candidate,
        "beliefs": [{"actor": EDITOR, "belief": "true", "date": "2026-11-20"}],
    }
    assert (video.statements, video.provenance) == ({"wdt:P2550": [labelled]}, [stated])
    assert video.get_candidates() == []
    assert video.get_provenance(believer=EDITOR, beliefs=["true"]) == [stated]

    video.add_belief(
        "wdt:P2550",
        work,
        actor=REVIEWER,
        belief="false",
        date="2026-11-21",
        note="the video holds an excerpt",
    )
    doubted = {
        **stated,
        "beliefs": [
            *stated["beliefs"],
            {
                "actor": REVIEWER,
                "belief": "false",
                "date": "2026-11-21",
                "note": "the video holds an excerpt",
            },
        ],
    }
    assert (video.statements, video.provenance) == (
        {"wdt:P2550": [labelled]},
        [doubted],
    )
    assert video.get_provenance(believer=REVIEWER) == []
    assert video.get_provenance(believer=REVIEWER, beliefs=["false"]) == [doubted]
    assert video.get_provenance(believer=EDITOR, beliefs=["true"]) == [doubted]

    video.add_belief("wdt:P2550", work, actor=REVIEWER, belief="unlikely")
    unlikely = {
        **stated,
        "beliefs": [
            *stated["beliefs"],
            {"actor": REVIEWER, "belief": "unlikely", "date": "2026-11-21"},
        ],
    }
    assert video.provenance == [unlikely]
    assert video.get_provenance(believer=REVIEWER) == [unlikely]
    assert video.get_provenance(believer=REVIEWER, beliefs=["true"]) == []
    assert video.get_provenance(believer=REVIEWER, beliefs=["unlikely"]) == [unlikely]
    assert video.get_provenance(believer=TILIE, beliefs=["undecided"]) == [unlikely]


@pytest.mark.parametrize(
    ("actor", "believer"),
    [
        ("wd:Q4700", "http://www.wikidata.org/entity/Q4700"),
        ("http://www.wikidata.org/entity/Q4700", "wd:Q4700"),
    ],
)
def test_actors_compare_as_their_iris(actor: str, believer: str) -> None:
    video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    video.add_candidate(
        "wdt:P175", PIRES, stated_by=TILIE, method="inferred", date="2026-11-18"
    )
    video.add_belief("wdt:P175", PIRES, actor=actor, belief="false", date="2026-11-19")
    assert video.get_candidates(believer=believer) == []
    assert video.get_provenance(believer=believer) == []
    assert video.get_provenance(believer=believer, beliefs=["true"]) == []
    assert len(video.get_candidates(believer=EDITOR)) == 1


def test_refusals_change_nothing() -> None:
    video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    work = {"id": "wd:Q29117946"}
    video.add_statement("wdt:P2550", work)
    before = (video.statements, video.provenance)

    with pytest.raises(ValueError) as refused:
        video.add_belief("wdt:P2550", work, actor=EDITOR, belief="maybe")  # type: ignore[arg-type]
    assert [(error["type"], error["loc"]) for error in refused.value.errors()] == [
        ("literal_error", ("belief",))
    ]
    for filtered in (video.get_provenance, video.get_candidates):
        for believer in (None, EDITOR):
            _raises(
                "The belief filter keeps the states ['true', 'undecided', "
                "'unlikely', 'false'], not ['not_false']",
                lambda: filtered(believer=believer, beliefs=["not_false"]),  # type: ignore[arg-type]
            )
    _raises(
        "a date dates the belief of the actor stating the pair; give actor=",
        lambda: video.add_statement("rdfs:comment", "x", date="2026-11-19"),
    )
    _raises(
        "'wdt:P2550' already has this value; a stated pair is no candidate",
        lambda: video.add_candidate(
            "wdt:P2550", work, stated_by=TILIE, method="inferred"
        ),
    )
    assert (video.statements, video.provenance) == before


def test_a_node_path_reaches_the_node_and_not_the_holder() -> None:
    video = ContinuousPhysicalTimeline(length=None, uid="cpt_1")
    video.add_statement(
        "wdt:P629",
        {
            "id": "wd:Q29117946",
            "label": "Clair de lune",
            "schema:sameAs": [{"id": "gnd:300042558"}],
        },
    )
    node = ("wdt:P629", 0)
    song = {"id": "gnd:1156628156"}
    video.add_candidate(
        "schema:sameAs",
        song,
        node=node,
        stated_by=TILIE,
        method="reconciled",
        service="https://lobid.org/gnd/search",
        date="2026-11-18",
    )
    video.add_belief(
        "schema:sameAs",
        song,
        node=node,
        actor=EDITOR,
        belief="false",
        date="2026-11-19",
        note="Debussy's song L 32",
    )
    expected = [
        {
            "beliefs": [
                {
                    "actor": EDITOR,
                    "belief": "false",
                    "date": "2026-11-19",
                    "note": "Debussy's song L 32",
                }
            ],
            "date": "2026-11-18",
            "method": "reconciled",
            "property": "schema:sameAs",
            "service": "https://lobid.org/gnd/search",
            "stated_by": [TILIE],
            "value": song,
        }
    ]
    assert video.get_provenance(node=node) == expected
    assert video.get_candidates(node=node) == expected
    assert video.get_candidates(node=node, believer=EDITOR) == []
    assert video.statements["wdt:P629"][0]["provenance"] == expected
    assert video.get_pairs(node=node) == [("schema:sameAs", {"id": "gnd:300042558"})]
    assert video.provenance == []


# endregion

# region Regions copy on write


def test_a_region_changed_is_a_copy() -> None:
    score = ContinuousLogicalTimeline(length=27, uid="clt_6")
    intro = score.create_region(0, 12, id="region_1")
    described = intro.with_statement("rdfs:comment", "the opening")
    assert (intro.statements, described.statements) == (
        {},
        {"rdfs:comment": ["the opening"]},
    )
    assert score.get_region("region_1").statements == {}
    assert intro != described
    score.replace_region(described)
    assert score.get_region("region_1") == described
    assert score.get_region("region_1").statements == {"rdfs:comment": ["the opening"]}


# endregion

# region Rectangles


def _m2() -> dict[str, Any]:
    """The package's second example: a score encoding and its video."""
    return {
        "@context": _context(
            "dcterms", "gnd", "melod", "rel", "schema", "viaf", "wd", "wdt", "xsd"
        ),
        "timetoshare": "0.2.0",
        "id": "#layerit",
        "timelines": [
            {
                "id": "#clt_6",
                "unit": "quarters",
                "length": [27, 1],
                "statements": {
                    "dcterms:created": [{"@value": "2025-12-05", "@type": "xsd:date"}],
                    "schema:exampleOfWork": [
                        {"id": MUSICBRAINZ_WORK, "label": "Clair de lune"}
                    ],
                    "type": ["melod:Manifestation", "schema:SheetMusic"],
                    "wdt:P629": [
                        {
                            "id": "wd:Q29117946",
                            "label": "Clair de lune",
                            "schema:sameAs": [
                                {"id": MUSICBRAINZ_WORK},
                                {"id": "gnd:300042558"},
                            ],
                        }
                    ],
                },
            },
            {
                "id": "#cpt_1",
                "unit": "seconds",
                "length": None,
                "statements": {
                    "rel:prf": [{"id": "viaf:113931950"}],
                    "schema:sameAs": [
                        {"id": "https://www.youtube.com/watch?v=Ch2mrPm1JnM"}
                    ],
                    "type": ["schema:VideoObject", "melod:Manifestation"],
                    "wdt:P175": [
                        {
                            **PIRES,
                            "schema:sameAs": [
                                {"id": "viaf:113931950"},
                                {"id": "gnd:124927521"},
                            ],
                        }
                    ],
                    "wdt:P31": [{"id": "wd:Q98069877"}],
                },
            },
        ],
    }


def test_rectangles_gather_the_pairs_by_verb() -> None:
    video = AlignmentBundle.from_dict(_m2()).get_timeline("cpt_1")
    assert [(r.verb, r.inverse, r.things, r.label) for r in video.get_rectangles()] == [
        (
            "is_a",
            False,
            (
                "http://schema.org/VideoObject",
                "https://lod.academy/melod/vocab/ontology#Manifestation",
                "http://www.wikidata.org/entity/Q98069877",
            ),
            None,
        ),
        (
            "performed_by",
            False,
            (
                "http://viaf.org/viaf/113931950",
                "http://www.wikidata.org/entity/Q235650",
                "https://d-nb.info/gnd/124927521",
            ),
            "Maria João Pires",
        ),
        ("same_as", False, ("https://www.youtube.com/watch?v=Ch2mrPm1JnM",), None),
    ]


# endregion

# region The manifest boundary


def test_a_manifest_s_statements_land_on_their_holders_and_write_back() -> None:
    data = _m2()
    bundle = AlignmentBundle.from_dict(copy.deepcopy(data))
    for index, timeline_id in enumerate(("clt_6", "cpt_1")):
        expected = data["timelines"][index]["statements"]
        assert bundle.get_timeline(timeline_id).statements == expected
        assert bundle.get_timeline(timeline_id).prefixes == {}
    assert (bundle.statements, bundle.provenance, bundle.prefixes) == ({}, [], {})
    written = bundle.to_dict()
    assert written["@context"] == data["@context"]
    assert written == data
    assert bundle.to_dict(minimal=False) == timetoshare.complete(data)


def test_a_plain_timeline_writes_the_format_s_terms() -> None:
    assert ContinuousLogicalTimeline(length=8, uid="piece").to_dict() == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "#piece", "unit": "quarters", "length": [8, 1]}],
    }


def test_a_manifest_without_context_reads_and_writes_the_terms() -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "#piece", "unit": "quarters", "length": [8, 1]}],
    }
    assert Timeline.from_dict(copy.deepcopy(data)).to_dict() == {
        "@context": dict(FORMAT_CONTEXT),
        **data,
    }


def _one_root(context: dict[str, Any], *statements: dict[str, Any]) -> dict[str, Any]:
    """A manifest of one root ``score`` with regions ``first``, ``second``.

    The first of *statements* is the root's, the others the regions'.
    """
    root: dict[str, Any] = {"id": "#score", "unit": "quarters", "length": [8, 1]}
    regions = [
        {"id": "#first", "start": [0, 1], "end": [4, 1]},
        {"id": "#second", "start": [4, 1], "end": [8, 1]},
    ]
    for region, held in zip(regions, statements[1:], strict=False):
        if held:
            region["statements"] = held
    if statements and statements[0]:
        root["statements"] = statements[0]
    root["regions"] = regions
    return {"@context": context, "timetoshare": "0.2.0", "timelines": [root]}


def test_a_prefix_declared_but_unused_is_not_written() -> None:
    comment = {"rdfs:comment": ["the score"]}
    read = _one_root(_context("rdfs", foaf=FOAF), comment)
    score = Timeline.from_dict(read)
    assert score.prefixes == {}
    assert score.to_dict() == _one_root(_context("rdfs"), comment)


def test_a_prefix_beyond_the_catalogue_stays_on_the_holder_using_it() -> None:
    mbw = "http://musicbrainz.org/work/"
    work = {
        "schema:exampleOfWork": [{"id": "mbw:8d331505-4d88-39ae-81c7-bec5da77af96"}]
    }
    data = _one_root(
        _context("rdfs", "schema", mbw=mbw),
        {"rdfs:comment": ["the score"]},
        work,
        {"rdfs:comment": ["the end"]},
    )
    score = Timeline.from_dict(copy.deepcopy(data))
    assert score.get_region("first").prefixes == {"mbw": mbw}
    assert score.prefixes == {}
    assert score.get_region("second").prefixes == {}
    assert score.to_dict() == data


def test_a_prefix_the_catalogue_writes_otherwise_stays_on_the_holder() -> None:
    https = "https://schema.org/"
    data = _one_root(_context(schema=https), {"type": ["schema:MusicComposition"]})
    score = Timeline.from_dict(copy.deepcopy(data))
    assert score.prefixes == {"schema": https}
    assert score.to_dict() == data


def test_one_prefix_for_two_namespaces_is_refused_on_writing() -> None:
    def read(timeline_id: str, namespace: str) -> Timeline:
        return Timeline.from_dict(
            {
                "@context": _context(ex=namespace),
                "timetoshare": "0.2.0",
                "timelines": [
                    {
                        "id": f"#{timeline_id}",
                        "unit": "quarters",
                        "length": [8, 1],
                        "statements": {"ex:p": ["v"]},
                    }
                ],
            }
        )

    bundle = AlignmentBundle()
    bundle.add_timeline(read("clt_2", "http://b.example/"))
    bundle.add_timeline(read("clt_1", "http://a.example/"))
    _raises(
        "the prefix 'ex' stands for 'http://a.example/' on timeline 'clt_1' and for "
        "'http://b.example/' on timeline 'clt_2'",
        bundle.to_dict,
    )


def test_statements_without_context_are_refused_on_reading() -> None:
    data = _one_root(_context(), {"rdfs:comment": ["the score"]})
    del data["@context"]
    _raises(
        "#/timelines/0/statements: statements and provenance need the manifest's @context",
        lambda: Timeline.from_dict(data),
    )


def test_a_value_naming_no_element_is_refused_on_writing() -> None:
    score = ContinuousLogicalTimeline(length=8, uid="score")
    score.add_statement("dcterms:source", {"id": "#ghost"})
    _raises(
        "#/timelines/0/statements/dcterms:source/0/id: '#ghost' names no element "
        "of this manifest",
        score.to_dict,
    )


def test_a_timeline_read_whole_writes_the_manifest_s_statements_back() -> None:
    data = _one_root(_context("rdfs", "wd", "wdt"), {"rdfs:comment": ["the score"]})
    data.update(
        id="#rec",
        statements={"wdt:P175": [PIRES]},
        provenance=[
            {
                "date": "2026-11-18",
                "method": "manual",
                "property": "wdt:P175",
                "stated_by": [EDITOR],
                "beliefs": [{"actor": EDITOR, "belief": "true", "date": "2026-11-18"}],
                "value": {"id": "wd:Q235650"},
            }
        ],
    )
    score = Timeline.from_dict(copy.deepcopy(data))
    assert score.statements == {"rdfs:comment": ["the score"]}
    assert score.to_dict() == data
    assert AlignmentBundle.from_dict(copy.deepcopy(data)).to_dict() == data


# endregion
