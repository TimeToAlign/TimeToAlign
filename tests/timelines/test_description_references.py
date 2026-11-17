"""What links name: the schemes and elements they name stay true wherever written.

A partial write declares the schemes its links name and leaves out of a part
slot what it leaves out of the manifest; an id minted again is named by its
new id; a removed timeline is named no more. The validation logic is
documented in ``tests/timelines/README.md`` under "Links and schemes".
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from timetoshare import validate

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Identifier,
    Link,
    PartOf,
    PartSlot,
    Scheme,
    Timeline,
    TimelineGroup,
)
from timetoalign.timelines import description

WORK = "http://musicbrainz.org/work/4a622b0d-b0c0-405e-b49e-05b70c108284"
MOVEMENT = "http://musicbrainz.org/work/80cbaf7b-a1da-3fc5-b650-6eb72a803980"
LEFT_OUT = "not_represented"


@pytest.fixture(autouse=True)
def _no_default_assertion() -> Iterator[None]:
    """Hold every test in a context without a default assertion."""
    token = description._DEFAULT_ASSERTION.set(None)
    yield
    description._DEFAULT_ASSERTION.reset(token)


def _raises(error: type[Exception], message: str, call: Callable[[], Any]) -> None:
    with pytest.raises(error, match=f"^{re.escape(message)}$"):
        call()


def _slot(*elements: str, scheme: str | None = None) -> PartSlot:
    """A slot of the first movement, represented by *elements* or absent."""
    identifiers = () if scheme is None else (Identifier(scheme, "106/1"),)
    if elements:
        return PartSlot(
            uri=(MOVEMENT,), identifiers=identifiers, represented_by=elements
        )
    return PartSlot(uri=(MOVEMENT,), identifiers=identifiers, absent=LEFT_OUT)


def _naming(
    *slots: PartSlot | list[str],
    identifiers: dict[str, str] | None = None,
    part_of: tuple[PartOf, ...] = (),
) -> Link:
    """A work link whose slots name the elements of each list given."""
    parts = tuple(
        slot if isinstance(slot, PartSlot) else _slot(*slot) for slot in slots
    )
    return Link(
        "pertains_to",
        "work",
        uri=(WORK,),
        identifiers=tuple(
            Identifier(scheme, value) for scheme, value in (identifiers or {}).items()
        ),
        parts_relation="has_part" if parts else None,
        parts=parts,
        part_of=part_of,
    )


def _named(link: Link) -> list[list[str] | str]:
    """What each slot of *link* names: its elements, or ``not_represented``."""
    return [list(slot.represented_by) or slot.absent for slot in link.parts]


def _written_named(link: dict[str, Any]) -> list[list[str] | str]:
    """What each slot of a written link names, as :func:`_named` reads it."""
    return [
        slot.get("represented_by") or slot["absent"] for slot in link.get("parts", [])
    ]


def _quarters(timeline_id: str | None = None, length: int = 8) -> Timeline:
    return ContinuousLogicalTimeline(length=length, uid=timeline_id)


def _minted_tree() -> Timeline:
    """``clt_1`` holding the child ``clt_2`` and the region ``region_1``."""
    root = _quarters()
    root.add_child(_quarters(length=4), offset=0)
    root.create_region(0, 4)
    return root


# region A partial write keeps the manifest valid


def _bundle_of_score() -> AlignmentBundle:
    """The roots ``score`` (region ``intro``) and ``perf``; schemes ``opus``, ``bwv``."""
    score = _quarters("score")
    score.create_region(0, 4, id="intro")
    score.add_link(_naming(["perf"], ["perf", "intro"], identifiers={"opus": "106"}))
    bundle = AlignmentBundle().add_timeline(score).add_timeline(_quarters("perf"))
    return bundle.add_scheme("opus").add_scheme("bwv")


SCORE_ALONE: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "timelines": [
        {
            "id": "score",
            "unit": "quarters",
            "length": [8, 1],
            "regions": [{"id": "intro", "start": [0, 1], "end": [4, 1]}],
            "links": [
                {
                    "relation": "pertains_to",
                    "level": "work",
                    "uri": [WORK],
                    "identifiers": [{"scheme": "opus", "value": "106"}],
                    "parts_relation": "has_part",
                    "parts": [
                        {"uri": [MOVEMENT], "absent": LEFT_OUT},
                        {"uri": [MOVEMENT], "represented_by": ["intro"]},
                    ],
                }
            ],
        }
    ],
    "schemes": [{"id": "opus"}],
}


def test_a_timeline_of_a_bundle_writes_what_its_links_name() -> None:
    bundle = _bundle_of_score()
    score = bundle.get_timeline("score")
    written, complete = score.to_dict(minimal=True), score.to_dict(minimal=False)
    assert written == SCORE_ALONE
    assert complete["timelines"][0]["links"][0]["parts"][0] == {
        "uri": [MOVEMENT],
        "identifiers": [],
        "label": None,
        "represented_by": [],
        "absent": LEFT_OUT,
        "assertion": None,
    }
    validate(written)
    validate(complete)
    assert _named(score.links[0]) == [["perf"], ["perf", "intro"]]
    whole = bundle.to_dict(minimal=True)
    validate(whole)
    assert [scheme["id"] for scheme in whole["schemes"]] == ["bwv", "opus"]
    assert [timeline["id"] for timeline in whole["timelines"]] == ["perf", "score"]
    assert _written_named(whole["timelines"][1]["links"][0]) == [
        ["perf"],
        ["perf", "intro"],
    ]


TWO_ROOTS: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "schemes": [{"id": "bwv"}, {"id": "opus"}],
    "timelines": [
        {"id": "perf", "unit": "seconds", "length": 60.0},
        {
            "id": "score",
            "unit": "quarters",
            "length": [8, 1],
            "links": [
                _naming(["perf"], identifiers={"opus": "106"}).to_dict(minimal=True)
            ],
        },
    ],
}


def test_a_timeline_selected_from_a_manifest_writes_what_its_links_name() -> None:
    score = Timeline.from_dict(copy.deepcopy(TWO_ROOTS), timeline_id="score")
    assert _named(score.links[0]) == [LEFT_OUT]
    written = score.to_dict(minimal=True)
    assert written == {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [8, 1],
                "links": [
                    {
                        "relation": "pertains_to",
                        "level": "work",
                        "uri": [WORK],
                        "identifiers": [{"scheme": "opus", "value": "106"}],
                        "parts_relation": "has_part",
                        "parts": [{"uri": [MOVEMENT], "absent": LEFT_OUT}],
                    }
                ],
            }
        ],
        "schemes": [{"id": "opus"}],
    }
    validate(written)
    validate(score.to_dict())


TWO_GROUPS: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "schemes": [{"id": "bwv"}, {"id": "opus"}],
    "timelines": [
        {
            "id": "perf",
            "unit": "seconds",
            "length": 60.0,
            "links": [_naming(identifiers={"bwv": "1"}).to_dict(minimal=True)],
        },
        {
            "id": "score",
            "unit": "quarters",
            "length": [8, 1],
            "regions": [
                {
                    "id": "intro",
                    "start": [0, 1],
                    "end": [4, 1],
                    "links": [
                        _naming(_slot("perf", scheme="opus")).to_dict(minimal=True)
                    ],
                }
            ],
        },
    ],
    "groups": [{"id": "g1", "members": ["score"]}, {"id": "g2", "members": ["perf"]}],
}


def test_a_group_selected_from_a_manifest_writes_what_its_links_name() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(TWO_GROUPS), group_id="g1")
    assert _named(group.get_timeline("score").get_region("intro").links[0]) == [
        LEFT_OUT
    ]
    written = group.to_dict(minimal=True)
    assert written == {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [
                    {
                        "id": "intro",
                        "start": [0, 1],
                        "end": [4, 1],
                        "links": [
                            {
                                "relation": "pertains_to",
                                "level": "work",
                                "uri": [WORK],
                                "parts_relation": "has_part",
                                "parts": [
                                    {
                                        "uri": [MOVEMENT],
                                        "identifiers": [
                                            {"scheme": "opus", "value": "106/1"}
                                        ],
                                        "absent": LEFT_OUT,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        "groups": [{"id": "g1", "members": ["score"]}],
        "schemes": [{"id": "opus"}],
        "graphical": {"order": {"g1": ["score"]}},
    }
    validate(written)
    validate(group.to_dict())


def _source(scheme: Scheme, root_id: str, *, on_region: bool) -> AlignmentBundle:
    """A bundle declaring *scheme*, named by its root's link or its region's."""
    if on_region:
        root = ContinuousPhysicalTimeline(length=60.0, uid=root_id)
        root.create_region(
            0.0,
            30.0,
            id=f"{root_id}_first",
            links=[_naming(identifiers={scheme.id: "1"})],
        )
    else:
        root = _quarters(root_id).add_link(
            _naming(
                part_of=(
                    PartOf("has_part", identifiers=(Identifier(scheme.id, "106"),)),
                )
            )
        )
    return AlignmentBundle().add_timeline(root).add_scheme(scheme)


def test_from_bundles_carries_the_schemes_its_links_name() -> None:
    first = _source(Scheme("opus", name="Opus"), "score", on_region=False)
    second = _source(Scheme("bwv"), "take", on_region=True).add_scheme("unused")
    merged = AlignmentBundle.from_bundles([first, second])
    assert merged.schemes == (Scheme("bwv"), Scheme("opus", name="Opus"))
    written = merged.to_dict()
    validate(written)
    assert [scheme["id"] for scheme in written["schemes"]] == ["bwv", "opus"]
    assert AlignmentBundle.from_dict(copy.deepcopy(written)).to_dict() == written


def test_two_different_schemes_with_one_id_refuse() -> None:
    message = "two different schemes carry the id 'opus'"
    first = _source(Scheme("opus", name="A"), "score", on_region=False)
    second = _source(Scheme("opus", name="B"), "take", on_region=True)
    _raises(ValueError, message, lambda: AlignmentBundle.from_bundles([first, second]))
    equal = _source(Scheme("opus", name="A"), "take", on_region=True)
    merged = AlignmentBundle.from_bundles([first, equal])
    assert merged.schemes == (Scheme("opus", name="A"),)
    validate(merged.to_dict())

    read = Timeline.from_dict(
        {
            "timetoshare": "0.2.0",
            "schemes": [{"id": "opus", "name": "read"}],
            "timelines": [
                {
                    "id": "score",
                    "unit": "quarters",
                    "length": [8, 1],
                    "links": [_naming(identifiers={"opus": "106"}).to_dict()],
                }
            ],
        }
    )
    bundle = AlignmentBundle().add_scheme(Scheme("opus", name="declared"))
    bundle.add_timeline(read)
    _raises(ValueError, message, bundle.to_dict)
    _raises(ValueError, message, read.to_dict)


# endregion

# region Re-minting keeps references


@pytest.mark.parametrize("scope", ["group", "bundle"])
def test_joining_a_scope_renames_what_slots_name(scope: str) -> None:
    member = _minted_tree().add_link(_naming(["region_1"], ["clt_2"]))
    newcomer = _minted_tree().add_link(_naming(["clt_1"], ["clt_2"], ["region_1"]))
    if scope == "group":
        holder: Any = TimelineGroup(timelines=[member])
        holder.add_timeline(newcomer)
    else:
        holder = AlignmentBundle().add_timeline(member)
        holder.add_link(_naming(["clt_1"], ["region_1"]))
        holder.add_timeline(newcomer)
        assert _named(holder.links[0]) == [["clt_1"], ["region_1"]]
    assert (newcomer.id, list(newcomer._children), list(newcomer._regions)) == (
        "clt_3",
        ["clt_4"],
        ["region_2"],
    )
    assert _named(newcomer.links[0]) == [["clt_3"], ["clt_4"], ["region_2"]]
    assert _named(member.links[0]) == [["region_1"], ["clt_2"]]
    validate(holder.to_dict())


def test_add_child_renames_what_the_child_s_slots_name() -> None:
    parent = _quarters()
    parent.create_region(0, 4)
    parent.add_link(_naming(["region_1"], ["clt_1"]))
    child = _quarters(length=4)
    child.create_region(0, 2, links=[_naming(["region_1"], ["clt_1"])])
    parent.add_child(child, offset=0)
    assert (child.id, list(child._regions)) == ("clt_2", ["region_2"])
    assert _named(child.get_region("region_2").links[0]) == [["region_2"], ["clt_2"]]
    assert _named(parent.links[0]) == [["region_1"], ["clt_1"]]
    validate(parent.to_dict())


def test_a_root_given_an_id_on_reading_is_named_by_it() -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [{"id": "intro", "start": [0, 1], "end": [4, 1]}],
                "links": [_naming(["score"]).to_dict(minimal=True)],
            }
        ],
        "links": [_naming(["score"], ["intro"]).to_dict(minimal=True)],
    }
    piece = Timeline.from_dict(data, timeline_id="piece")
    assert _named(piece.links[0]) == [["piece"]]
    written = piece.to_dict(minimal=True)
    validate(written)
    assert _written_named(written["links"][0]) == [["piece"], ["intro"]]
    assert _written_named(written["timelines"][0]["links"][0]) == [["piece"]]


def test_from_bundles_renames_what_slots_name() -> None:
    first, second = _minted_tree(), _minted_tree()
    for root in (first, second):
        root.add_link(_naming(["clt_2"], ["region_1"]))
    merged = AlignmentBundle.from_bundles(
        [AlignmentBundle().add_timeline(first), AlignmentBundle().add_timeline(second)]
    )
    assert sorted(merged.timelines) == ["clt_1", "clt_3"]
    assert (second.id, list(second._children), list(second._regions)) == (
        "clt_3",
        ["clt_4"],
        ["region_2"],
    )
    assert _named(first.links[0]) == [["clt_2"], ["region_1"]]
    assert _named(second.links[0]) == [["clt_4"], ["region_2"]]
    validate(merged.to_dict())


# endregion

# region Removal leaves the removed out of what slots name


def _root(timeline_id: str, region_id: str, link: Link) -> dict[str, Any]:
    return {
        "id": timeline_id,
        "unit": "quarters",
        "length": [8, 1],
        "regions": [{"id": region_id, "start": [0, 1], "end": [4, 1]}],
        "links": [link.to_dict(minimal=True)],
    }


GROUPED: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "timelines": [
        _root("a", "a1", _naming(["b1"], ["a1", "b"])),
        _root("b", "b1", _naming(["a"])),
    ],
    "groups": [{"id": "g", "members": ["a", "b"]}],
    "links": [_naming(["b"], ["a", "b1"]).to_dict(minimal=True)],
}


def test_removing_a_member_leaves_it_out_of_what_slots_name() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(GROUPED))
    removed = group.remove_timeline("b")
    a = group.get_timeline("a")
    assert _named(a.links[0]) == [LEFT_OUT, ["a1"]]
    assert _named(removed.links[0]) == [LEFT_OUT]
    written = group.to_dict(minimal=True)
    validate(written)
    assert _written_named(written["links"][0]) == [LEFT_OUT, ["a"]]
    assert _written_named(written["timelines"][0]["links"][0]) == [LEFT_OUT, ["a1"]]
    validate(removed.to_dict())


def test_a_bundle_holding_the_group_keeps_the_removed_member_named() -> None:
    bundle = AlignmentBundle.from_dict(copy.deepcopy(GROUPED))
    before = bundle.to_dict(minimal=False)
    removed = bundle.groups["g"].remove_timeline("b")
    after = bundle.to_dict(minimal=False)
    validate(after)
    assert after["links"] == before["links"]
    assert after["timelines"] == before["timelines"]
    assert after["groups"] == [{"id": "g", "name": None, "meta": {}, "members": ["a"]}]
    assert _named(removed.links[0]) == [["a"]]


def test_a_member_whose_root_stays_is_still_named() -> None:
    root = _quarters("r")
    first, second = _quarters("c1", 4), _quarters("c2", 4)
    root.add_child(first, offset=0)
    root.add_child(second, offset=4)
    first.add_link(_naming(["c2"]))
    group = TimelineGroup(id="g", timelines=[first, second])
    group.remove_timeline("c2")
    assert _named(first.links[0]) == [["c2"]]
    validate(group.to_dict())


# endregion
