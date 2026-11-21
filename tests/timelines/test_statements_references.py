"""What statements name stays true wherever they are written.

An element given a new id is named by it; an element leaving the manifest —
removed, left behind by a selection or by a partial write — is named no
more, its pair, provenance entry and basis item dropped together. The
validation logic is documented in ``tests/timelines/README.md`` under
"Statements".
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import pytest
from timetoshare import FORMAT_CONTEXT, load_catalogue, validate

from timetoalign import (
    AlignmentBundle,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    Timeline,
    TimelineGroup,
)
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.maps import LinearMap
from timetoalign.timelines.flowcontrol import Break

EDITOR = "https://example.org/people/editor"
SOURCE = "dcterms:source"


def _raises(message: str, call: Callable[[], Any]) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        call()


def _context(*prefixes: str, **declared: str) -> dict[str, Any]:
    catalogue = load_catalogue().prefixes
    return {
        **FORMAT_CONTEXT,
        **{prefix: catalogue[prefix] for prefix in prefixes},
        **declared,
    }


def _entry(element: str) -> dict[str, Any]:
    """The confirmed entry of the pair naming *element*, resting on it."""
    return {
        "basis": [f"#{element}"],
        "date": "2026-11-18",
        "method": "manual",
        "property": SOURCE,
        "stated_by": [EDITOR],
        "value": {"id": f"#{element}"},
    }


def _naming(holder: Any, *elements: str) -> Any:
    """Make *holder* name each of *elements* in a value, an entry and a basis."""
    for element in elements:
        entry = _entry(element)
        holder.add_statement(
            SOURCE,
            {"id": f"#{element}"},
            provenance={
                key: entry[key] for key in ("basis", "date", "method", "stated_by")
            },
        )
    return holder


def _state(*elements: str) -> dict[str, Any]:
    """The statements and provenance of a holder naming *elements*, as written."""
    if not elements:
        return {}
    return {
        "statements": {SOURCE: [{"id": f"#{element}"} for element in elements]},
        "provenance": [_entry(element) for element in elements],
    }


def _named(holder: Any) -> list[str]:
    """What *holder* names, the same in its values, entries and basis items."""
    if isinstance(holder, dict):
        statements = holder.get("statements", {})
        provenance = holder.get("provenance", [])
    else:
        statements, provenance = holder.statements, holder.provenance
    values = [value["id"] for value in statements.get(SOURCE, [])]
    entries = [entry["value"]["id"] for entry in provenance]
    basis = [item for entry in provenance for item in entry.get("basis", [])]
    assert values == entries == basis
    return [value.removeprefix("#") for value in values]


def _quarters(timeline_id: str | None = None, length: int = 8) -> Timeline:
    return ContinuousLogicalTimeline(length=length, uid=timeline_id)


def _minted_tree() -> Timeline:
    """``clt_1`` holding the child ``clt_2`` and the region ``region_1``."""
    root = _quarters()
    root.add_child(_quarters(length=4), offset=0)
    root.create_region(0, 4)
    return root


# region A partial write keeps the manifest valid


def test_a_timeline_of_a_bundle_writes_what_its_statements_name() -> None:
    score = _quarters("score")
    score.create_region(0, 4, id="intro")
    _naming(score, "perf", "intro")
    bundle = AlignmentBundle().add_timeline(score).add_timeline(_quarters("perf"))
    written = score.to_dict()
    assert written == {
        "@context": _context("dcterms"),
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "#score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [{"id": "#intro", "start": [0, 1], "end": [4, 1]}],
                **_state("intro"),
            }
        ],
    }
    validate(written)
    validate(score.to_dict(minimal=False))
    assert _named(score) == ["perf", "intro"]
    whole = bundle.to_dict()
    validate(whole)
    assert [timeline["id"] for timeline in whole["timelines"]] == ["#perf", "#score"]
    assert _named(whole["timelines"][1]) == ["perf", "intro"]


def test_a_timeline_selected_from_a_manifest_names_nothing_left_behind() -> None:
    data = {
        "@context": _context("dcterms"),
        "timetoshare": "0.2.0",
        "timelines": [
            {"id": "#perf", "unit": "seconds", "length": 60.0},
            {"id": "#score", "unit": "quarters", "length": [8, 1], **_state("perf")},
        ],
    }
    score = Timeline.from_dict(data, timeline_id="score")
    assert _named(score) == []
    written = score.to_dict()
    assert written == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "#score", "unit": "quarters", "length": [8, 1]}],
    }
    validate(written)


def test_a_group_selected_from_a_manifest_names_nothing_left_behind() -> None:
    data = {
        "@context": _context("dcterms"),
        "timetoshare": "0.2.0",
        "timelines": [
            {"id": "#perf", "unit": "seconds", "length": 60.0, **_state("score")},
            {
                "id": "#score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [
                    {
                        "id": "#intro",
                        "start": [0, 1],
                        "end": [4, 1],
                        **_state("perf", "score"),
                    }
                ],
            },
        ],
        "groups": [
            {"id": "#g1", "members": ["#score"]},
            {"id": "#g2", "members": ["#perf"]},
        ],
    }
    group = TimelineGroup.from_dict(data, group_id="g1")
    assert _named(group.get_timeline("score").get_region("intro")) == ["score"]
    written = group.to_dict()
    assert written == {
        "@context": _context("dcterms"),
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "#score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [
                    {"id": "#intro", "start": [0, 1], "end": [4, 1], **_state("score")}
                ],
            }
        ],
        "groups": [{"id": "#g1", "members": ["#score"]}],
        "graphical": {"order": {"#g1": ["#score"]}},
    }
    validate(written)


# endregion

# region Prefixes travel with their holders


def _source(timeline_id: str, prefix: str, namespace: str, *, on_region: bool) -> Any:
    """A bundle whose root, or its region, uses *prefix* for *namespace*."""
    root = ContinuousPhysicalTimeline(length=60.0, uid=timeline_id)
    pair = (f"{prefix}:work", {"id": f"{prefix}:{timeline_id}"})
    if on_region:
        region = root.create_region(0.0, 30.0, id=f"{timeline_id}_first")
        root.replace_region(region.with_prefix(prefix, namespace).with_statement(*pair))
    else:
        root.add_prefix(prefix, namespace).add_statement(*pair)
    bundle = AlignmentBundle().add_timeline(root)
    return bundle.add_statement("rdfs:comment", f"the bundle of {timeline_id}")


def test_from_bundles_carries_the_prefixes_its_holders_use() -> None:
    first = _source("score", "mbw", "http://musicbrainz.org/work/", on_region=False)
    second = _source("take", "ex", "http://a.example/", on_region=True)
    merged = AlignmentBundle.from_bundles([first, second])
    assert (merged.statements, merged.prefixes) == ({}, {})
    assert merged.get_timeline("score").prefixes == {
        "mbw": "http://musicbrainz.org/work/"
    }
    assert merged.get_timeline("take").get_region("take_first").prefixes == {
        "ex": "http://a.example/"
    }
    written = merged.to_dict()
    validate(written)
    assert written["@context"] == _context(
        ex="http://a.example/", mbw="http://musicbrainz.org/work/"
    )
    assert AlignmentBundle.from_dict(copy.deepcopy(written)).to_dict() == written


def test_two_namespaces_for_one_prefix_refuse_to_be_written() -> None:
    first = _source("score", "ex", "http://a.example/", on_region=False)
    second = _source("take", "ex", "http://b.example/", on_region=True)
    merged = AlignmentBundle.from_bundles([first, second])
    _raises(
        "the prefix 'ex' stands for 'http://a.example/' on timeline 'score' and for "
        "'http://b.example/' on region 'take_first'",
        merged.to_dict,
    )
    equal = _source("take", "ex", "http://a.example/", on_region=True)
    validate(AlignmentBundle.from_bundles([first, equal]).to_dict())


# endregion

# region Every new id renames


@pytest.mark.parametrize("scope", ["group", "bundle"])
def test_joining_a_scope_renames_what_statements_name(scope: str) -> None:
    member = _naming(_minted_tree(), "region_1", "clt_2")
    newcomer = _naming(_minted_tree(), "clt_1", "clt_2", "region_1")
    if scope == "group":
        holder: Any = TimelineGroup(timelines=[member])
        holder.add_timeline(newcomer)
    else:
        holder = _naming(AlignmentBundle().add_timeline(member), "clt_1", "region_1")
        holder.add_timeline(newcomer)
        assert _named(holder) == ["clt_1", "region_1"]
    assert (newcomer.id, list(newcomer._children), list(newcomer._regions)) == (
        "clt_3",
        ["clt_4"],
        ["region_2"],
    )
    assert _named(newcomer) == ["clt_3", "clt_4", "region_2"]
    assert _named(member) == ["region_1", "clt_2"]
    validate(holder.to_dict())


def test_add_child_renames_what_the_child_s_statements_name() -> None:
    parent = _quarters()
    parent.create_region(0, 4)
    _naming(parent, "region_1", "clt_1")
    child = _quarters(length=4)
    region = child.create_region(0, 2)
    child.replace_region(_naming(_RegionHolder(region), "region_1", "clt_1").region)
    parent.add_child(child, offset=0)
    assert (child.id, list(child._regions)) == ("clt_2", ["region_2"])
    assert _named(child.get_region("region_2")) == ["region_2", "clt_2"]
    assert _named(parent) == ["region_1", "clt_1"]
    validate(parent.to_dict())


class _RegionHolder:
    """A region taking ``add_statement`` as the timelines do, by copying."""

    def __init__(self, region: Any) -> None:
        self.region = region

    def add_statement(self, *args: Any, **kwargs: Any) -> _RegionHolder:
        self.region = self.region.with_statement(*args, **kwargs)
        return self


def test_a_root_given_an_id_on_reading_is_named_by_it() -> None:
    data = {
        "@context": _context("dcterms"),
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "#score",
                "unit": "quarters",
                "length": [8, 1],
                "regions": [{"id": "#intro", "start": [0, 1], "end": [4, 1]}],
                **_state("score"),
            }
        ],
        **_state("score", "intro"),
    }
    piece = Timeline.from_dict(data, timeline_id="piece")
    assert _named(piece) == ["piece"]
    written = piece.to_dict()
    validate(written)
    assert _named(written) == ["piece", "intro"]
    assert _named(written["timelines"][0]) == ["piece"]


def test_from_bundles_renames_what_statements_name() -> None:
    first, second = _minted_tree(), _minted_tree()
    for root in (first, second):
        _naming(root, "clt_2", "region_1")
    merged = AlignmentBundle.from_bundles(
        [AlignmentBundle().add_timeline(first), AlignmentBundle().add_timeline(second)]
    )
    assert sorted(merged.timelines) == ["clt_1", "clt_3"]
    assert (second.id, list(second._children), list(second._regions)) == (
        "clt_3",
        ["clt_4"],
        ["region_2"],
    )
    assert _named(first) == ["clt_2", "region_1"]
    assert _named(second) == ["clt_4", "region_2"]
    validate(merged.to_dict())


def test_from_bundles_renames_a_group_minted_again() -> None:
    sources = []
    for _ in range(2):
        score = _quarters()
        bundle = AlignmentBundle().add_timeline(score, as_group="score")
        _naming(score, "group_1")
        sources.append((bundle, score))
    merged = AlignmentBundle.from_bundles([bundle for bundle, _ in sources])
    assert sorted(merged.groups) == ["group_1", "group_2"]
    assert [_named(score) for _, score in sources] == [["group_1"], ["group_2"]]
    validate(merged.to_dict())


def test_add_group_renames_the_group_in_the_trees_joining_with_it() -> None:
    p = _naming(_quarters("p"), "group_1")
    s = _naming(_quarters("s"), "group_1")
    bundle = AlignmentBundle().add_timeline(p, as_group="first")
    group = TimelineGroup(timelines=[p, s])
    assert group.id == "group_1"
    bundle.add_group(group)
    assert (sorted(bundle.groups), group.id) == (["group_1", "group_2"], "group_2")
    assert (_named(p), _named(s)) == (["group_1"], ["group_2"])
    validate(bundle.to_dict())


def test_from_bundles_renames_every_tree_of_the_source_at_once() -> None:
    f = _naming(_quarters("f"), "group_1")
    first = AlignmentBundle().add_timeline(f, as_group="f")
    m, n = _naming(_quarters("m"), "group_2"), _quarters("n")
    u = _naming(_quarters("u"), "group_1", "group_2")
    source = (
        AlignmentBundle().add_timeline(m, as_group="a").add_timeline(n, as_group="b")
    )
    source.add_timeline(u)
    assert sorted(source.groups) == ["group_1", "group_2"]
    merged = AlignmentBundle.from_bundles([first, source])
    assert {key: group.timeline_ids for key, group in merged.groups.items()} == {
        "group_1": ["f"],
        "group_2": ["m"],
        "group_3": ["n"],
    }
    assert (_named(f), _named(m), _named(u)) == (
        ["group_1"],
        ["group_3"],
        ["group_2", "group_3"],
    )
    validate(merged.to_dict())
    # The source shares its trees with the merged bundle: it no longer writes.
    _raises(
        "#/timelines/0/statements/dcterms:source/0/id: '#group_3' names no element "
        "of this manifest",
        source.to_dict,
    )


def _grouped_naming_group_1() -> tuple[Timeline, TimelineGroup, AlignmentBundle]:
    """``m`` in the minted ``group_1`` of a bundle, both naming ``#group_1``."""
    m = _naming(_quarters("m"), "group_1")
    group = TimelineGroup(timelines=[m])
    bundle = _naming(AlignmentBundle().add_group(group), "group_1")
    return m, group, bundle


@pytest.mark.parametrize("joins", ["bundle", "group"])
def test_a_group_giving_way_to_a_given_id_is_renamed_but_in_the_newcomer(
    joins: str,
) -> None:
    m, group, bundle = _grouped_naming_group_1()
    newcomer = _naming(_quarters("group_1"), "group_1")
    (bundle if joins == "bundle" else group).add_timeline(newcomer)
    assert (group.id, sorted(bundle.groups)) == ("group_2", ["group_2"])
    assert (_named(m), _named(bundle), _named(newcomer)) == (
        ["group_2"],
        ["group_2"],
        ["group_1"],
    )
    validate(bundle.to_dict())


def test_a_region_given_the_id_of_a_minted_group_renames_the_group() -> None:
    m, group, bundle = _grouped_naming_group_1()
    m.create_region(0, 4, id="group_1")
    assert group.id == "group_2"
    assert (_named(m), _named(bundle)) == (["group_2"], ["group_2"])
    validate(bundle.to_dict())


def _q(value: int) -> Coordinate:
    return Coordinate(Fraction(value), TimeUnit.quarters)


def _with_map_and_break() -> Timeline:
    """A root holding the minted map ``lmap_1`` and break ``flow_marker_1``."""
    root = _quarters(length=16)
    root.add_conversion_map(
        LinearMap(scalar=2, source_unit="quarters", target_unit="seconds")
    )
    root.flow_control.add_break(Break(_q(8)))
    return root


def test_a_map_and_a_break_minted_again_with_their_tree_are_renamed() -> None:
    held = _with_map_and_break()
    newcomer = _naming(_with_map_and_break(), "lmap_1", "flow_marker_1")
    bundle = AlignmentBundle().add_timeline(held).add_timeline(newcomer)
    (cmap,) = newcomer._conversion_maps
    (element,) = newcomer.flow_control._elements()
    assert (cmap, element.id) == ("lmap_2", "flow_marker_2")
    assert _named(newcomer) == ["lmap_2", "flow_marker_2"]
    validate(bundle.to_dict())


def test_a_map_added_beside_one_named_leaves_the_name_with_it() -> None:
    root = _naming(_with_map_and_break(), "lmap_1")
    root.add_conversion_map(
        LinearMap(scalar=3, source_unit="quarters", target_unit="seconds")
    )
    assert sorted(root._conversion_maps) == ["lmap_1", "lmap_2"]
    assert _named(root) == ["lmap_1"]


# endregion

# region Removal leaves the removed out of what statements name


def _root(timeline_id: str, region_id: str, *named: str) -> dict[str, Any]:
    return {
        "id": f"#{timeline_id}",
        "unit": "quarters",
        "length": [8, 1],
        "regions": [{"id": f"#{region_id}", "start": [0, 1], "end": [4, 1]}],
        **_state(*named),
    }


GROUPED: dict[str, Any] = {
    "@context": _context("dcterms"),
    "timetoshare": "0.2.0",
    "timelines": [_root("a", "a1", "b1", "a1", "b"), _root("b", "b1", "a")],
    "groups": [{"id": "#g", "members": ["#a", "#b"]}],
    **_state("b", "a", "b1"),
}


def test_removing_a_member_leaves_it_out_of_what_statements_name() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(GROUPED))
    removed = group.remove_timeline("b")
    a = group.get_timeline("a")
    assert _named(a) == ["a1"]
    assert _named(removed) == []
    written = group.to_dict()
    validate(written)
    assert _named(written) == ["a"]
    assert _named(written["timelines"][0]) == ["a1"]
    validate(removed.to_dict())


def test_a_bundle_holding_the_group_keeps_the_removed_member_named() -> None:
    bundle = AlignmentBundle.from_dict(copy.deepcopy(GROUPED))
    before = bundle.to_dict(minimal=False)
    removed = bundle.groups["g"].remove_timeline("b")
    after = bundle.to_dict(minimal=False)
    validate(after)
    assert after["statements"] == before["statements"]
    assert after["provenance"] == before["provenance"]
    assert after["timelines"] == before["timelines"]
    assert after["groups"] == [
        {"id": "#g", "name": None, "meta": {}, "members": ["#a"]}
    ]
    assert _named(removed) == ["a"]


def test_a_member_whose_root_stays_is_still_named() -> None:
    root = _quarters("r")
    first, second = _quarters("c1", 4), _quarters("c2", 4)
    root.add_child(first, offset=0)
    root.add_child(second, offset=4)
    _naming(first, "c2")
    group = TimelineGroup(id="g", timelines=[first, second])
    group.remove_timeline("c2")
    assert _named(first) == ["c2"]
    validate(group.to_dict())


# endregion

# region What starts empty, what copies


def test_children_from_regions_and_slices_start_without_statements() -> None:
    score = _quarters("score", 16)
    score.add_statement("rdfs:comment", "the score")
    region = score.create_region(0, 8, id="intro")
    score.replace_region(region.with_statement("rdfs:comment", "the opening"))
    child = score.create_child_from_region("intro")
    sliced = score.get_slice(0, 8)
    assert (child.statements, child.provenance) == ({}, [])
    assert (sliced.statements, sliced.provenance) == ({}, [])
    assert score.get_region("intro").statements == {"rdfs:comment": ["the opening"]}


def test_to_typed_copies_statements_and_prefixes_independently() -> None:
    source = Timeline(length=8, unit="quarters", uid="score")
    source.add_prefix("ex", "http://a.example/").add_statement("ex:p", "v")
    typed = source.to_typed()
    assert typed is not source
    assert (typed.statements, typed.prefixes) == (
        {"ex:p": ["v"]},
        {"ex": "http://a.example/"},
    )
    typed.add_statement("ex:p", "w").add_prefix("mbw", "http://musicbrainz.org/work/")
    assert (source.statements, source.prefixes) == (
        {"ex:p": ["v"]},
        {"ex": "http://a.example/"},
    )


# endregion
