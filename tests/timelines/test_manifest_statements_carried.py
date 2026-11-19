"""``@context``, ``statements`` and ``provenance`` are carried unread.

The strategy is documented in ``tests/timelines/README.md`` ("Statements
carried"). Every expected value is a literal written by hand in the file
form, or ``timetoshare.complete`` / ``timetoshare.dumps`` of one, never the
library's own output.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from timetoshare import complete, dumps, minimal

from timetoalign.alignment import AlignmentBundle
from timetoalign.timelines import Timeline, TimelineGroup

NS = "https://timetoalign.github.io/timetoshare/ns#"
EDITOR = "https://example.org/people/editor"
TILIE = "https://example.org/agents/tilie"
WORK = "http://musicbrainz.org/work/8d331505-4d88-39ae-81c7-bec5da77af96"

CONTEXT: dict[str, Any] = {
    "@version": 1.1,
    "id": "@id",
    "type": "@type",
    "statements": "@nest",
    "list": "@list",
    "timelines": "@included",
    "children": "@included",
    "regions": "@included",
    "groups": "@included",
    "label": "http://www.w3.org/2000/01/rdf-schema#label",
    "provenance": {"@reverse": "http://www.w3.org/1999/02/22-rdf-syntax-ns#subject"},
    "property": {
        "@id": "http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate",
        "@type": "@vocab",
    },
    "value": "http://www.w3.org/1999/02/22-rdf-syntax-ns#object",
    "stated_by": {"@id": "http://www.w3.org/ns/prov#wasAttributedTo", "@type": "@id"},
    "date": {
        "@id": "http://purl.org/dc/terms/date",
        "@type": "http://www.w3.org/2001/XMLSchema#date",
    },
    "method": {"@id": f"{NS}method", "@type": "@vocab"},
    "service": {"@id": f"{NS}service", "@type": "@id"},
    "basis": {"@id": "http://www.w3.org/ns/prov#hadPrimarySource", "@type": "@id"},
    "match": {"@id": f"{NS}match", "@type": "@vocab"},
    "note": "http://www.w3.org/2000/01/rdf-schema#comment",
    "status": {"@id": f"{NS}status", "@type": "@vocab"},
    "manual": f"{NS}Manual",
    "reconciled": f"{NS}Reconciled",
    "imported": f"{NS}Imported",
    "inferred": f"{NS}Inferred",
    "confirmed": f"{NS}Confirmed",
    "proposed": f"{NS}Proposed",
    "rejected": f"{NS}Rejected",
    "exact": "http://www.w3.org/2004/02/skos/core#exactMatch",
    "close": "http://www.w3.org/2004/02/skos/core#closeMatch",
    "broad": "http://www.w3.org/2004/02/skos/core#broadMatch",
    "narrow": "http://www.w3.org/2004/02/skos/core#narrowMatch",
    "related": "http://www.w3.org/2004/02/skos/core#relatedMatch",
    "dcterms": "http://purl.org/dc/terms/",
    "gnd": "https://d-nb.info/gnd/",
    "melod": "https://w3id.org/melod/melod/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rel": "http://id.loc.gov/vocabulary/relators/",
    "schema": "http://schema.org/",
    "viaf": "http://viaf.org/viaf/",
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}
"""The format's terms and the prefixes the manifests below use."""

BUNDLE_DESCRIPTION: dict[str, Any] = {
    "statements": {"schema:sameAs": [{"id": "wd:Q29117946"}, {"id": WORK}]},
    "provenance": [
        {
            "basis": ["#cpt_2"],
            "date": "2026-11-18",
            "method": "inferred",
            "property": "schema:sameAs",
            "stated_by": [TILIE],
            "status": "proposed",
            "value": {"id": "gnd:300042558"},
        }
    ],
}
"""What the manifest states: the work, and a proposed identifier of it."""

RECORDING_DESCRIPTION: dict[str, Any] = {
    "statements": {
        "rel:prf": [{"id": "viaf:113931950"}],
        "schema:sameAs": [{"id": "https://www.youtube.com/watch?v=Ch2mrPm1JnM"}],
        "type": ["schema:VideoObject", "melod:Manifestation"],
        "wdt:P175": [
            {
                "id": "wd:Q235650",
                "label": "Maria João Pires",
                "schema:sameAs": [{"id": "viaf:113931950"}, {"id": "gnd:124927521"}],
            }
        ],
        "wdt:P31": [{"id": "wd:Q98069877"}],
    },
    "provenance": [
        {
            "date": "2026-11-18",
            "method": "manual",
            "property": "wdt:P175",
            "stated_by": [EDITOR],
            "status": "confirmed",
            "value": {"id": "wd:Q235650"},
        },
        {
            "basis": ["#cpt_2"],
            "date": "2026-11-20",
            "method": "inferred",
            "note": "the video holds an excerpt",
            "property": "wdt:P2550",
            "stated_by": [EDITOR],
            "status": "rejected",
            "value": {"id": "wd:Q29117946"},
        },
    ],
}
"""What the root states: a video of a performance, and a rejected pair."""

EXCERPT_DESCRIPTION: dict[str, Any] = {
    "statements": {
        "schema:isPartOf": [{"id": "#cpt_1"}],
        "type": ["schema:VideoObject"],
    },
    "provenance": [
        {
            "date": "2026-11-18",
            "method": "manual",
            "property": "rdf:type",
            "stated_by": [EDITOR],
            "status": "confirmed",
            "value": {"id": "schema:VideoObject"},
        }
    ],
}
"""What the child states: a part of the root, named by its ``#`` reference."""

PIECE_DESCRIPTION: dict[str, Any] = {
    "statements": {
        "wdt:P2550": [{"id": "wd:Q29117946", "label": "Clair de lune"}],
    },
    "provenance": [
        {
            "basis": ["#cpt_2"],
            "date": "2026-11-19",
            "method": "inferred",
            "property": "wdt:P2550",
            "stated_by": [EDITOR],
            "status": "confirmed",
            "value": {"id": "wd:Q29117946"},
        }
    ],
}
"""What the region states: the piece it holds, confirmed after a proposal."""

EDITION_DESCRIPTION: dict[str, Any] = {
    "statements": {
        "dcterms:created": [{"@value": "2025-12-05", "@type": "xsd:date"}],
        "schema:exampleOfWork": [{"id": WORK, "label": "Clair de lune"}],
        "type": ["melod:Manifestation", "schema:SheetMusic"],
        "wdt:P629": [
            {
                "id": "wd:Q29117946",
                "label": "Clair de lune",
                "schema:sameAs": [{"id": WORK}, {"id": "gnd:300042558"}],
            }
        ],
    },
    "provenance": [
        {
            "date": "2026-11-18",
            "method": "imported",
            "property": "dcterms:created",
            "stated_by": [EDITOR],
            "status": "confirmed",
            "value": {"@value": "2025-12-05", "@type": "xsd:date"},
        }
    ],
}
"""What the second root of the group manifest states: a printed edition."""


def _recording() -> dict[str, Any]:
    """The root ``cpt_1`` with its region and its child, each described."""
    return {
        "id": "#cpt_1",
        "unit": "seconds",
        "length": None,
        "children": [
            {
                "id": "#cpt_2",
                "unit": "seconds",
                "offset": 0.0,
                "length": 5.0,
                **copy.deepcopy(EXCERPT_DESCRIPTION),
            }
        ],
        "regions": [
            {
                "id": "#region_1",
                "start": 0.0,
                "end": 10.0,
                **copy.deepcopy(PIECE_DESCRIPTION),
            }
        ],
        **copy.deepcopy(RECORDING_DESCRIPTION),
    }


ONE_ROOT: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "@context": CONTEXT,
    "id": "#rec",
    "timelines": [_recording()],
    **BUNDLE_DESCRIPTION,
}
"""One root and no group, the minimal form: what ``Timeline`` reads whole."""

ONE_GROUP: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "@context": CONTEXT,
    "timelines": [
        {
            "id": "#clt_6",
            "unit": "quarters",
            "length": [16, 1],
            **copy.deepcopy(EDITION_DESCRIPTION),
        },
        _recording(),
    ],
    "groups": [{"id": "#group_1", "members": ["#clt_6", "#cpt_1"]}],
    "graphical": {"order": {"#group_1": ["#clt_6", "#cpt_1"]}},
    **BUNDLE_DESCRIPTION,
}
"""An anonymous bundle with one group, the minimal form: what ``TimelineGroup`` reads whole."""

CASES = [
    pytest.param(Timeline, ONE_ROOT, id="timeline"),
    pytest.param(AlignmentBundle, ONE_ROOT, id="bundle-one-root"),
    pytest.param(TimelineGroup, ONE_GROUP, id="group"),
    pytest.param(AlignmentBundle, ONE_GROUP, id="bundle-one-group"),
]
FORMS = [pytest.param(True, id="minimal"), pytest.param(False, id="complete")]
SERIALIZATIONS = ("yaml", "json", "jcs")


def _form(manifest: dict[str, Any], is_minimal: bool) -> dict[str, Any]:
    """Return *manifest* in the form asked for, as the package writes it."""
    return minimal(manifest) if is_minimal else complete(manifest)


def test_the_literals_are_minimal() -> None:
    """The literals are already the minimal form, so they are what is written."""
    for manifest in (ONE_ROOT, ONE_GROUP):
        assert minimal(manifest) == manifest


@pytest.mark.parametrize(("reader", "manifest"), CASES)
@pytest.mark.parametrize("read_minimal", FORMS)
def test_from_dict_to_dict_carries_them(
    reader: type, manifest: dict[str, Any], read_minimal: bool
) -> None:
    """Read from either form, the object writes both forms exactly."""
    read = reader.from_dict(_form(manifest, read_minimal))
    assert read.to_dict() == manifest
    assert read.to_dict(minimal=False) == complete(manifest)


@pytest.mark.parametrize(("reader", "manifest"), CASES)
@pytest.mark.parametrize("is_minimal", FORMS)
@pytest.mark.parametrize("serialization", SERIALIZATIONS)
def test_files_carry_them(
    reader: type,
    manifest: dict[str, Any],
    is_minimal: bool,
    serialization: str,
    tmp_path: Path,
) -> None:
    """A file read and written back is the same text, byte for byte."""
    text = dumps(_form(manifest, is_minimal), serialization)
    source = tmp_path / "source.txt"
    source.write_bytes(text.encode("utf-8"))
    written = reader.from_file(source).to_file(tmp_path / "written.txt")
    assert written.read_bytes().decode("utf-8") == text


def _kept(holder: Any) -> dict[str, Any]:
    """The keys *holder* keeps unread, ``statements`` and ``provenance`` only."""
    return {
        key: value
        for key, value in holder._unknown_keys.items()
        if key in ("statements", "provenance")
    }


def test_values_inside_them_keep_their_hash_in_memory() -> None:
    """IRIs inside the keys are held as the file writes them, ``#`` included.

    An element's ID is held bare in memory; an IRI naming an element inside
    ``statements`` or ``provenance`` is not an ID of the file form, so the
    object holds it unchanged.
    """
    root = Timeline.from_dict(ONE_ROOT)
    assert root.id == "cpt_1"
    assert _kept(root) == RECORDING_DESCRIPTION
    assert _kept(root.get_child("cpt_2")) == EXCERPT_DESCRIPTION
    assert _kept(root.get_region("region_1")) == PIECE_DESCRIPTION
    assert root._manifest is not None
    assert root._manifest._unknown_keys == {"@context": CONTEXT, **BUNDLE_DESCRIPTION}

    bundle = AlignmentBundle.from_dict(ONE_GROUP)
    assert _kept(bundle.timelines["clt_6"]) == EDITION_DESCRIPTION
    assert bundle._unknown_keys == {"@context": CONTEXT, **BUNDLE_DESCRIPTION}
