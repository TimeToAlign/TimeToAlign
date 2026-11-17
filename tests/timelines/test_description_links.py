"""Links and schemes: the description layer a bundle, a timeline and a region carry.

The validation logic is documented in ``tests/timelines/README.md`` under
"Links and schemes".
"""

from __future__ import annotations

import contextvars
import copy
import datetime
import json
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import timetoshare
from timetoshare import complete, dumps, minimal, validate

import timetoalign
from timetoalign import (
    AlignmentBundle,
    Assertion,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Coordinate,
    Link,
    PartSlot,
    Scheme,
    Timeline,
    TimelineGroup,
    TimeUnit,
    get_default_assertion,
    set_default_assertion,
)
from timetoalign.timelines import Region, description

MB = "http://musicbrainz.org/work/"
IMSLP = "http://www.wikidata.org/entity/P839"
OP_106 = "Piano_Sonata_No.29,_Op.106_(Beethoven,_Ludwig_van)"
NAME = "Beethoven, Piano Sonata No. 29, Op. 106"
LABEL = "Piano Sonata No. 29 in B♭ major, Op. 106"
SERIES = "https://musicbrainz.org/series/004f17aa-0f46-4fec-b939-c5dfc849e5c6"
SCHEME_NAME = "Works of Ludwig van Beethoven by opus number"
EDITOR = "https://example.org/people/editor"
OTHER = "https://example.org/people/other"
MOVEMENTS = [
    ("80cbaf7b-a1da-3fc5-b650-6eb72a803980", "I. Allegro"),
    ("7813c9cf-6054-3ef3-b9ed-8ae4690581c9", "II. Scherzo. Assai vivace"),
    ("7d110747-f011-41d4-9ce5-a1fad9abed2f", "III. Adagio sostenuto"),
    (
        "cfa82cd6-bcb5-39dd-9beb-c98e275c7c84",
        "IV. Introduzione. Largo – Allegro risoluto",
    ),
]
WORK_URL = "https://musicbrainz.org/work/4a622b0d-b0c0-405e-b49e-05b70c108284"
URLS = (
    WORK_URL,
    "https://www.wikidata.org/wiki/Q146730",
    "https://viaf.org/viaf/186299955",
    f"https://imslp.org/wiki/{OP_106}",
)
RELEASE_URL = "https://musicbrainz.org/release/814c53fd-0732-4819-9753-4578f5ab992c"
MOVEMENT_I = f"https://musicbrainz.org/work/{MOVEMENTS[0][0]}"
MOVEMENT_II = f"https://musicbrainz.org/work/{MOVEMENTS[1][0]}"
TODAY = datetime.date(2026, 11, 17)
EDITOR_ASSERTION = Assertion(
    asserted_by=(EDITOR,), asserted_on=TODAY, method="manual", status="confirmed"
)

# region The work bundle of Op. 106, as the format's rules write it

OP_106_MINIMAL: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "name": NAME,
    "timelines": [],
    "schemes": [{"id": "mb_beethoven_opus", "name": SCHEME_NAME, "uri": [SERIES]}],
    "links": [
        {
            "relation": "pertains_to",
            "level": "work",
            "label": LABEL,
            "uri": [
                f"{MB}4a622b0d-b0c0-405e-b49e-05b70c108284",
                "http://www.wikidata.org/entity/Q146730",
                "http://viaf.org/viaf/186299955",
            ],
            "identifiers": [
                {"scheme": IMSLP, "value": OP_106},
                {"scheme": "mb_beethoven_opus", "value": "op. 106"},
            ],
            "parts_relation": "has_part",
            "parts": [
                {"uri": [f"{MB}{mbid}"], "label": label, "absent": "not_represented"}
                for mbid, label in MOVEMENTS
            ],
            "assertion": {
                "asserted_by": [EDITOR],
                "asserted_on": "2026-11-17",
                "method": "manual",
                "status": "confirmed",
            },
        }
    ],
}

OP_106_COMPLETE: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "name": NAME,
    "meta": {},
    "groups": [],
    "measure_maps": [],
    "timelines": [],
    "schemes": [
        {
            "id": "mb_beethoven_opus",
            "name": SCHEME_NAME,
            "uri": [SERIES],
            "criteria": None,
        }
    ],
    "links": [
        {
            "relation": "pertains_to",
            "level": "work",
            "uri": OP_106_MINIMAL["links"][0]["uri"],
            "identifiers": OP_106_MINIMAL["links"][0]["identifiers"],
            "records": [],
            "label": LABEL,
            "assertion": {
                "asserted_by": [EDITOR],
                "asserted_on": "2026-11-17",
                "method": "manual",
                "service": None,
                "based_on": [],
                "match": None,
                "status": "confirmed",
            },
            "parts_relation": "has_part",
            "parts": [
                {
                    "uri": [f"{MB}{mbid}"],
                    "identifiers": [],
                    "label": label,
                    "represented_by": [],
                    "absent": "not_represented",
                    "assertion": None,
                }
                for mbid, label in MOVEMENTS
            ],
            "parts_ordered": True,
            "part_of": [],
        }
    ],
}


def _op_106() -> AlignmentBundle:
    """Build the work bundle of Op. 106 through the library."""
    bundle = AlignmentBundle(name=NAME)
    bundle.add_scheme("mb_beethoven_opus", name=SCHEME_NAME, uri=SERIES)
    bundle.add_link(
        Link.from_urls(
            *URLS,
            identifiers={"mb_beethoven_opus": "op. 106"},
            label=LABEL,
            assertion=EDITOR_ASSERTION,
        ).with_parts(
            "has_part",
            [
                PartSlot.from_url(
                    f"https://musicbrainz.org/work/{mbid}",
                    label=label,
                    absent="not_represented",
                )
                for mbid, label in MOVEMENTS
            ],
        )
    )
    return bundle


# endregion


@pytest.fixture(autouse=True)
def _no_default_assertion() -> Iterator[None]:
    """Hold every test in a context without a default assertion."""
    token = description._DEFAULT_ASSERTION.set(None)
    yield
    description._DEFAULT_ASSERTION.reset(token)


@pytest.fixture
def today(monkeypatch: pytest.MonkeyPatch) -> datetime.date:
    """Pin the day a link takes the default assertion on."""
    monkeypatch.setattr(description, "_today", lambda: TODAY)
    return TODAY


def _quarters(timeline_id: str | None = None, length: int = 16) -> Timeline:
    return ContinuousLogicalTimeline(length=length, uid=timeline_id)


def _raises(error: type[Exception], message: str, call: Callable[[], Any]) -> None:
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        call()


# region The model is the package's


@pytest.mark.parametrize(
    "name",
    ["Link", "PartSlot", "PartOf", "Identifier", "Assertion", "Scheme", "Descriptor"],
)
def test_the_classes_are_the_package_s(name: str) -> None:
    assert getattr(timetoalign, name) is getattr(timetoshare, name)
    assert getattr(timetoalign.timelines, name) is getattr(timetoshare, name)


# endregion

# region The work bundle of Op. 106


def test_the_work_bundle_of_op_106_in_both_forms() -> None:
    built = _op_106()
    written, written_minimal = built.to_dict(), built.to_dict(minimal=True)
    assert written_minimal == OP_106_MINIMAL
    assert written == OP_106_COMPLETE
    assert dumps(written_minimal) == dumps(OP_106_MINIMAL)
    assert dumps(written) == dumps(OP_106_COMPLETE)
    for form in (written, written_minimal):
        validate(form)
    assert dumps(complete(written_minimal)) == dumps(written)
    assert dumps(minimal(written)) == dumps(written_minimal)
    for literal in (OP_106_COMPLETE, OP_106_MINIMAL):
        read = AlignmentBundle.from_dict(copy.deepcopy(literal))
        assert read.to_dict() == OP_106_COMPLETE
        assert read.to_dict(minimal=True) == OP_106_MINIMAL
        assert read == built


# endregion

# region Adding links and querying them


def test_a_url_is_a_link() -> None:
    score = _quarters()
    returned = score.add_link(WORK_URL, label="Op. 106").add_link(RELEASE_URL)
    assert returned is score
    assert score.links == (
        Link.from_url(WORK_URL, label="Op. 106"),
        Link.from_url(RELEASE_URL),
    )
    _raises(
        TypeError,
        "keyword arguments ['label'] build a link from a URL; a Link is added as it is",
        lambda: score.add_link(Link.from_url(WORK_URL), label="Op. 106"),
    )
    assert len(score.links) == 2


def _four_links() -> tuple[Timeline, list[Link]]:
    """A timeline holding four links each filter splits differently."""
    links = [
        Link.from_url(
            WORK_URL,
            identifiers={"mb_beethoven_opus": "op. 106"},
            assertion=EDITOR_ASSERTION,
        ),
        Link.from_url(URLS[3], assertion=Assertion(asserted_by=(OTHER,))),
        Link.from_url(RELEASE_URL, assertion=EDITOR_ASSERTION),
        Link.from_url(
            "https://example.org/people/composer", level="agent", relation="marcrel:cmp"
        ),
    ]
    score = _quarters()
    for link in links:
        score.add_link(link)
    return score, links


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, [0, 1, 2, 3]),
        ({"level": "work"}, [0, 1]),
        ({"relation": "marcrel:cmp"}, [3]),
        ({"asserted_by": EDITOR}, [0, 2]),
        ({"scheme": "mb_beethoven_opus"}, [0]),
        ({"scheme": IMSLP}, [1]),
        ({"scheme": "imslp"}, [1]),
        ({"scheme": "musicbrainz_work"}, [0]),
        ({"scheme": "http://www.wikidata.org/entity/P5813"}, [2]),
        ({"level": "work", "asserted_by": EDITOR}, [0]),
    ],
    ids=[
        "none",
        "level",
        "relation",
        "asserted_by",
        "manifest_scheme",
        "scheme_uri",
        "registry_id_of_identifiers",
        "registry_id_of_uris",
        "registry_uri_of_uris",
        "combined",
    ],
)
def test_every_filter_of_get_links(
    filters: dict[str, str], expected: list[int]
) -> None:
    score, links = _four_links()
    assert score.get_links(**filters) == [links[index] for index in expected]


def test_a_region_s_links() -> None:
    quarters = TimeUnit.quarters
    region = Region(
        Coordinate(0, quarters), Coordinate(16, quarters), name="Exposition"
    )
    first = Link.from_url(MOVEMENT_I, label="I. Allegro")
    linked = region.with_link(first)
    again = linked.with_link(RELEASE_URL)
    assert region.links == ()
    assert linked.links == (first,)
    assert again.links == (first, Link.from_url(RELEASE_URL))
    assert (again.id, again.start, again.end, again.name, again._minted) == (
        "region_1",
        region.start,
        region.end,
        "Exposition",
        True,
    )

    score = _quarters("score")
    score.create_region(0, 4, id="intro", links=[first])
    score.create_region(4, 8, id="verse")
    assert score.get_region("intro").links == (first,)
    regions = score.to_dict()["timelines"][0]["regions"]
    assert regions == [
        {
            "id": "intro",
            "name": None,
            "start": [0, 1],
            "end": [4, 1],
            "meta": {},
            "links": [first.to_dict()],
        },
        {
            "id": "verse",
            "name": None,
            "start": [4, 1],
            "end": [8, 1],
            "meta": {},
            "links": [],
        },
    ]
    assert score.to_dict(minimal=True)["timelines"][0]["regions"] == [
        {
            "id": "intro",
            "start": [0, 1],
            "end": [4, 1],
            "links": [{**first.to_dict(minimal=True)}],
        },
        {"id": "verse", "start": [4, 1], "end": [8, 1]},
    ]


# endregion

# region Schemes


def _scheme_bundle() -> AlignmentBundle:
    """A bundle holding a timeline with a region, a given group and a scheme."""
    score = _quarters()
    score.create_region(0, 4, id="intro")
    bundle = AlignmentBundle().add_timeline(score)
    bundle.add_group(TimelineGroup(id="g", timelines=[_quarters("member", 4)]))
    return bundle.add_scheme("opus")


@pytest.mark.parametrize(
    ("scheme_id", "holder"),
    [
        ("clt_1", "timeline 'clt_1'"),
        ("intro", "region 'intro' of timeline 'clt_1'"),
        ("g", "group 'g'"),
        ("opus", "scheme 'opus'"),
    ],
)
def test_a_scheme_id_held_in_the_bundle_is_refused(scheme_id: str, holder: str) -> None:
    bundle = _scheme_bundle()
    _raises(
        ValueError,
        f"scheme '{scheme_id}' cannot join bundle 'bundle_1': its id '{scheme_id}' "
        f"is already held by {holder}",
        lambda: bundle.add_scheme(scheme_id, name="Refused"),
    )
    assert [scheme.to_dict() for scheme in bundle.schemes] == [
        {"id": "opus", "name": None, "uri": [], "criteria": None}
    ]


def test_a_scheme_id_is_never_the_bundle_s_given_id() -> None:
    bundle = AlignmentBundle(id="study")
    _raises(
        ValueError,
        "scheme 'study' cannot join bundle 'study': its id 'study' is already held by "
        "bundle 'study'",
        lambda: bundle.add_scheme("study"),
    )
    assert bundle.schemes == ()


def test_a_minted_group_id_gives_way_to_a_scheme() -> None:
    bundle = AlignmentBundle().add_timeline(_quarters(), as_group="Score")
    assert list(bundle.groups) == ["group_1"]
    bundle.add_scheme("group_1")
    assert list(bundle.groups) == ["group_2"]
    assert bundle.groups["group_2"].id == "group_2"
    assert bundle.schemes == (Scheme("group_1"),)


def test_a_scheme_id_refuses_a_newcomer() -> None:
    bundle = _scheme_bundle()
    newcomer = _quarters("newcomer", 4)
    newcomer.create_region(0, 1, id="opus")
    _raises(
        ValueError,
        "region 'opus' of timeline 'newcomer' cannot join bundle 'bundle_1': its id "
        "'opus' is already held by scheme 'opus'",
        lambda: bundle.add_timeline(newcomer),
    )
    assert "newcomer" not in bundle.timelines


def test_schemes_are_built_added_and_sorted() -> None:
    given = Scheme("catalogue", name="Catalogue", uri=("https://example.org/c",))
    bundle = AlignmentBundle().add_scheme(
        "opus", name="Opus", uri=SERIES, criteria=None
    )
    bundle.add_scheme(given)
    bundle.add_scheme("bwv", uri=["https://example.org/bwv", "https://example.org/b"])
    assert bundle.schemes == (
        Scheme("bwv", uri=("https://example.org/bwv", "https://example.org/b")),
        given,
        Scheme("opus", name="Opus", uri=(SERIES,)),
    )
    assert [scheme["id"] for scheme in bundle.to_dict()["schemes"]] == [
        "bwv",
        "catalogue",
        "opus",
    ]
    _raises(
        TypeError,
        "name, uri and criteria build a scheme from its id; a Scheme is added as it is",
        lambda: bundle.add_scheme(Scheme("x"), name="X"),
    )


def _one_root(**keys: Any) -> dict[str, Any]:
    return {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        **keys,
    }


def test_reading_refuses_a_scheme_id_a_group_holds() -> None:
    data = _one_root(groups=[{"id": "g", "members": ["score"]}], schemes=[{"id": "g"}])
    _raises(
        ValueError,
        "the id 'g' is held twice in the manifest: by group 'g' and by scheme 'g'",
        lambda: AlignmentBundle.from_dict(data),
    )


def test_a_root_cannot_take_a_scheme_s_id() -> None:
    _raises(
        ValueError,
        "root timeline 'score' cannot take the id 'opus': it is already held by "
        "scheme 'opus'",
        lambda: Timeline.from_dict(
            _one_root(schemes=[{"id": "opus"}]), timeline_id="opus"
        ),
    )


# endregion

# region The default assertion


def test_without_a_default_a_link_is_added_as_it_is() -> None:
    assert get_default_assertion() is None
    score = _quarters().add_link(WORK_URL)
    assert score.links[0].assertion is None


def test_the_default_assertion_is_dated_the_day_a_link_is_added(
    today: datetime.date,
) -> None:
    own = Link.from_url(RELEASE_URL, assertion=Assertion(asserted_by=(OTHER,)))
    region = Region(Coordinate(0, TimeUnit.quarters), Coordinate(4, TimeUnit.quarters))
    score = _quarters()
    with set_default_assertion(asserted_by=EDITOR, status="confirmed") as default:
        assert default == Assertion(
            asserted_by=(EDITOR,), method="manual", status="confirmed"
        )
        assert get_default_assertion() == default
        score.add_link(WORK_URL).add_link(own)
        region = region.with_link(MOVEMENT_I)
    expected = Assertion(
        asserted_by=(EDITOR,), asserted_on=today, method="manual", status="confirmed"
    )
    assert [link.assertion for link in score.links] == [expected, own.assertion]
    assert region.links[0].assertion == expected
    assert get_default_assertion() is None


def test_every_member_of_the_default_assertion() -> None:
    with set_default_assertion(
        asserted_by=[EDITOR, OTHER],
        method="reconciled",
        service="https://example.org/reconcile",
        based_on="https://example.org/editions/london-1819",
        match="close",
        status="unconfirmed",
    ) as default:
        assert default == Assertion(
            asserted_by=(EDITOR, OTHER),
            method="reconciled",
            service="https://example.org/reconcile",
            based_on=("https://example.org/editions/london-1819",),
            match="close",
            status="unconfirmed",
        )


def test_the_context_manager_restores_the_previous_default() -> None:
    with set_default_assertion(asserted_by=EDITOR):
        with set_default_assertion(asserted_by=OTHER):
            assert get_default_assertion().asserted_by == (OTHER,)
        assert get_default_assertion().asserted_by == (EDITOR,)
    assert get_default_assertion() is None


def test_a_default_set_in_another_context_is_not_seen() -> None:
    def set_inside() -> Assertion | None:
        set_default_assertion(asserted_by=EDITOR)
        return get_default_assertion()

    inside = contextvars.copy_context().run(set_inside)
    assert inside == Assertion(asserted_by=(EDITOR,), method="manual")
    assert get_default_assertion() is None


# endregion

# region Round trips


def _described() -> AlignmentBundle:
    """A scheme and a link on the manifest, a link on a root and on its region."""
    take = ContinuousPhysicalTimeline(length=690.0, uid="take")
    take.add_link(RELEASE_URL, assertion=Assertion(method="manual", match="broad"))
    take.create_region(
        1.5,
        688.0,
        id="first",
        name="I. Allegro",
        links=[Link.from_url(MOVEMENT_I, label="I. Allegro")],
    )
    bundle = AlignmentBundle(name="Described")
    bundle.add_scheme("mb_beethoven_opus", name=SCHEME_NAME, uri=SERIES)
    bundle.add_link(
        Link.from_url(
            WORK_URL,
            identifiers={"mb_beethoven_opus": "op. 106"},
            assertion=EDITOR_ASSERTION,
        ).with_parts(
            "has_part",
            [
                PartSlot.from_url(MOVEMENT_I, represented_by=["first"]),
                PartSlot.from_url(MOVEMENT_II, absent="not_represented"),
            ],
        )
    )
    return bundle.add_timeline(take)


@pytest.mark.parametrize("form", [False, True], ids=["complete", "minimal"])
def test_both_forms_round_trip(form: bool) -> None:
    bundle = _described()
    written = bundle.to_dict(minimal=form)
    validate(written)
    read = AlignmentBundle.from_dict(json.loads(json.dumps(written)))
    assert read.to_dict() == bundle.to_dict()
    assert read.to_dict(minimal=True) == bundle.to_dict(minimal=True)
    assert read.links == bundle.links
    assert read.schemes == bundle.schemes
    take = read.get_timeline("take")
    assert take.links == bundle.get_timeline("take").links
    assert take.get_region("first").links == (
        Link.from_url(MOVEMENT_I, label="I. Allegro"),
    )


def _noted(data: dict[str, Any]) -> dict[str, Any]:
    """The complete form of the described bundle with unknown keys inside links."""
    noted = copy.deepcopy(data)
    link = noted["links"][0]
    link["x-note"] = "manifest link"
    link["assertion"]["x-note"] = "assertion"
    link["parts"][0]["x-note"] = "slot"
    noted["schemes"][0]["x-note"] = "scheme"
    root = noted["timelines"][0]
    root["links"][0]["x-note"] = "timeline link"
    root["regions"][0]["links"][0]["x-note"] = "region link"
    return noted


def test_unknown_keys_inside_links_are_written_where_found() -> None:
    noted = _noted(_described().to_dict())
    noted_minimal = minimal(noted)
    for literal in (noted, noted_minimal):
        read = AlignmentBundle.from_dict(copy.deepcopy(literal))
        assert dumps(read.to_dict()) == dumps(noted)
        assert dumps(read.to_dict(minimal=True)) == dumps(noted_minimal)
    written = AlignmentBundle.from_dict(noted).to_dict(minimal=True)
    assert written["links"][0]["x-note"] == "manifest link"
    assert written["links"][0]["assertion"]["x-note"] == "assertion"
    assert written["links"][0]["parts"][0]["x-note"] == "slot"
    assert written["schemes"][0]["x-note"] == "scheme"
    root = written["timelines"][0]
    assert root["links"][0]["x-note"] == "timeline link"
    assert root["regions"][0]["links"][0]["x-note"] == "region link"


ONE_ROOT_DESCRIBED: dict[str, Any] = _one_root(
    schemes=[{"id": "opus"}],
    links=[
        {
            "relation": "pertains_to",
            "level": "work",
            "identifiers": [{"scheme": "opus", "value": "106"}],
        }
    ],
)


def test_a_timeline_read_whole_keeps_the_manifest_s_links() -> None:
    score = Timeline.from_dict(copy.deepcopy(ONE_ROOT_DESCRIBED))
    assert score.links == ()
    assert score.to_dict(minimal=True) == ONE_ROOT_DESCRIBED


def test_a_group_read_whole_keeps_the_manifest_s_links() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(ONE_ROOT_DESCRIBED), group_id="g")
    grouped = {
        **ONE_ROOT_DESCRIBED,
        "groups": [{"id": "g", "members": ["score"]}],
        "graphical": {"order": {"g": ["score"]}},
    }
    assert group.to_dict(minimal=True) == grouped
    assert (
        TimelineGroup.from_dict(copy.deepcopy(grouped)).to_dict(minimal=True) == grouped
    )


# endregion

# region What the validator refuses


def test_a_slot_naming_nothing_is_refused_in_the_validator_s_words() -> None:
    data = _one_root(
        links=[
            {
                "relation": "pertains_to",
                "level": "work",
                "parts_relation": "has_part",
                "parts": [{"represented_by": ["ghost"]}],
            }
        ]
    )
    for reader in (Timeline.from_dict, AlignmentBundle.from_dict):
        _raises(
            ValueError,
            "#/links/0/parts/0/represented_by/0: 'ghost' names no timeline or region "
            "of this manifest",
            lambda reader=reader: reader(data),
        )


def test_a_region_s_link_is_refused_at_its_path() -> None:
    data = _one_root()
    data["timelines"][0]["regions"] = [
        {
            "id": "intro",
            "start": [0, 1],
            "end": [4, 1],
            "links": [{"relation": "pertains_to", "level": "score"}],
        }
    ]
    _raises(
        ValueError,
        "#/timelines/0/regions/0/links/0/level: 'score' is not one of ['work', "
        "'expression', 'manifestation', 'item', 'agent']",
        lambda: Timeline.from_dict(data),
    )


# endregion
