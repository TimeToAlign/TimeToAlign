"""What statements name stays true wherever they are written.

An element keeps its id wherever it joins, and what names it names it still;
a timeline given a new id is named by it; an element leaving the manifest —
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


def _tree(seed: int | None = None) -> Timeline:
    """A root holding a child and a region, every ID minted (from *seed*)."""
    root = ContinuousLogicalTimeline(length=8, seed=seed)
    child_seed = None if seed is None else seed + 1
    root.add_child(ContinuousLogicalTimeline(length=4, seed=child_seed), offset=0)
    root.create_region(0, 4, seed=seed)
    return root


def _tree_ids(root: Timeline) -> list[str]:
    """The root's ID, its child's and its region's."""
    return [root.id, *root._children, *root._regions]


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

# region Joining keeps every ID; a collision is refused, nothing renamed


def _ids(root: Timeline) -> tuple[str, list[str], list[str]]:
    return (root.id, list(root._children), list(root._regions))


def _written(holder: Any) -> list[list[str]]:
    """What every timeline of *holder*'s manifest names, in the order written."""
    return [_named(timeline) for timeline in holder.to_dict()["timelines"]]


@pytest.mark.parametrize("scope", ["group", "bundle"])
def test_joining_a_scope_keeps_what_statements_name(scope: str) -> None:
    member, newcomer = _tree(), _tree()
    root, (child,), (region,) = _ids(newcomer)
    _naming(member, *_ids(member)[2], *_ids(member)[1])
    _naming(newcomer, root, child, region)
    # Built alike, from their content: no ID of the one is held by the other.
    assert set(_tree_ids(member)).isdisjoint(_tree_ids(newcomer))
    if scope == "group":
        holder: Any = TimelineGroup(timelines=[member])
        holder.add_timeline(newcomer)
    else:
        holder = _naming(AlignmentBundle().add_timeline(member), member.id, region)
        holder.add_timeline(newcomer)
        assert _named(holder) == [member.id, region]
    assert _ids(newcomer) == (root, [child], [region])
    assert _named(newcomer) == [root, child, region]
    assert _named(member) == [*_ids(member)[2], *_ids(member)[1]]
    validate(holder.to_dict())


@pytest.mark.parametrize("scope", ["group", "bundle"])
def test_a_tree_holding_an_id_of_the_scope_is_refused(scope: str) -> None:
    member, newcomer = _tree(7), _tree(7)
    _naming(newcomer, *_tree_ids(newcomer))
    holder: Any = (
        TimelineGroup(timelines=[member], id="scope")
        if scope == "group"
        else AlignmentBundle(id="scope").add_timeline(member)
    )
    _raises(
        f"timeline '{member.id}' cannot join {scope} 'scope': its id "
        f"'{member.id}' is already held by timeline '{member.id}'",
        lambda: holder.add_timeline(newcomer),
    )
    assert holder.timeline_ids == [member.id]
    assert _named(newcomer) == _tree_ids(newcomer) == _tree_ids(member)


def test_add_child_keeps_what_the_child_s_statements_name() -> None:
    parent = _quarters("parent")
    held = parent.create_region(0, 4)
    _naming(parent, held.id, "parent")
    child = _quarters("child", length=4)
    region = child.create_region(0, 2)
    child.replace_region(_naming(_RegionHolder(region), region.id, "child").region)
    parent.add_child(child, offset=0)
    assert (child.id, list(child._regions)) == ("child", [region.id])
    assert _named(child.get_region(region.id)) == [region.id, "child"]
    assert _named(parent) == [held.id, "parent"]
    validate(parent.to_dict())


def test_add_child_refuses_a_child_holding_an_id_of_the_tree() -> None:
    parent = _quarters("parent")
    held = parent.create_region(0, 4, seed=7)
    child = _quarters("child", length=4)
    region = child.create_region(0, 2, seed=7)
    assert region.id == held.id
    _raises(
        f"region '{held.id}' of timeline 'child' cannot join: its id '{held.id}' is "
        f"already held by region '{held.id}' of timeline 'parent'",
        lambda: parent.add_child(child, offset=0),
    )
    assert (parent.n_children, child._parent) == (0, None)
    assert list(child._regions) == [held.id]


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


def test_from_bundles_keeps_what_statements_name() -> None:
    first, second = _tree(), _tree()
    for root in (first, second):
        _naming(root, *_ids(root)[1], *_ids(root)[2])
    before = [_ids(first), _ids(second)]
    merged = AlignmentBundle.from_bundles(
        [AlignmentBundle().add_timeline(first), AlignmentBundle().add_timeline(second)]
    )
    assert merged.timeline_ids == [first.id, second.id]
    assert [_ids(first), _ids(second)] == before
    assert _named(first) == [*_ids(first)[1], *_ids(first)[2]]
    assert _named(second) == [*_ids(second)[1], *_ids(second)[2]]
    validate(merged.to_dict())


def test_from_bundles_refuses_sources_sharing_an_id() -> None:
    first, second = _tree(7), _tree(7)
    sources = [AlignmentBundle().add_timeline(t) for t in (first, second)]
    held = re.escape(first.id)
    with pytest.raises(
        ValueError,
        match=f"^timeline '{held}' cannot join bundle '[^']+': its id '{held}' is "
        f"already held by timeline '{held}'$",
    ):
        AlignmentBundle.from_bundles(sources)
    assert [source.timeline_ids for source in sources] == [[first.id], [second.id]]
    assert _ids(second) == _ids(first)


def test_groups_minted_alike_keep_their_ids_through_from_bundles() -> None:
    sources = []
    for _ in range(2):
        score = _quarters()
        bundle = AlignmentBundle().add_timeline(score, as_group="score")
        (group_id,) = bundle.groups
        _naming(score, group_id)
        sources.append((bundle, score, group_id))
    merged = AlignmentBundle.from_bundles([bundle for bundle, _, _ in sources])
    assert list(merged.groups) == [group_id for _, _, group_id in sources]
    assert len(set(merged.groups)) == 2
    assert [_named(score) for _, score, _ in sources] == [
        [group_id] for _, _, group_id in sources
    ]
    validate(merged.to_dict())


def test_a_group_whose_id_the_bundle_holds_is_refused() -> None:
    p, s = _quarters("p"), _quarters("s")
    bundle = AlignmentBundle().add_group(TimelineGroup(timelines=[p], seed=7))
    group = TimelineGroup(timelines=[s], seed=7)
    _raises(
        f"Group '{group.id}' already exists in bundle", lambda: bundle.add_group(group)
    )
    assert list(bundle.groups) == [group.id]
    assert bundle.groups[group.id].timeline_ids == ["p"]
    sources = [
        AlignmentBundle().add_group(TimelineGroup(timelines=[_quarters(t)], seed=7))
        for t in ("a", "b")
    ]
    _raises(
        f"Group '{group.id}' already exists in bundle",
        lambda: AlignmentBundle.from_bundles(sources),
    )


@pytest.mark.parametrize("joins", ["bundle", "group", "region"])
def test_a_given_id_a_minted_group_holds_is_refused(joins: str) -> None:
    m = _naming(_quarters("m"), "m")
    group = TimelineGroup(timelines=[m])
    bundle = _naming(AlignmentBundle(id="rec").add_group(group), group.id)
    held = f"its id '{group.id}' is already held by group '{group.id}'"
    if joins == "region":
        _raises(
            f"region '{group.id}' of timeline 'm' cannot join: {held}",
            lambda: m.create_region(0, 4, id=group.id),
        )
        assert m.list_regions() == []
    else:
        newcomer = _quarters(group.id)
        scope = "bundle 'rec'" if joins == "bundle" else f"group '{group.id}'"
        _raises(
            f"timeline '{group.id}' cannot join {scope}: {held}",
            lambda: (bundle if joins == "bundle" else group).add_timeline(newcomer),
        )
    assert (list(bundle.groups), group.timeline_ids) == ([group.id], ["m"])
    assert (_named(m), _named(bundle)) == (["m"], [group.id])
    validate(bundle.to_dict())


def _q(value: int) -> Coordinate:
    return Coordinate(Fraction(value), TimeUnit.quarters)


def _with_map_and_break(
    timeline_id: str | None = None, seed: int | None = None
) -> Timeline:
    """A root holding one conversion map and one break, minted."""
    root = _quarters(timeline_id, length=16)
    root.add_conversion_map(
        LinearMap(scalar=2, source_unit="quarters", target_unit="seconds", seed=seed)
    )
    root.flow_control.add(Break(_q(8), seed=seed))
    return root


def _map_and_break(root: Timeline) -> tuple[str, str]:
    (cmap,) = root._conversion_maps
    (element,) = root.flow_control.elements
    return cmap, element.id


def test_a_map_and_a_break_keep_their_ids_with_their_tree() -> None:
    held = _with_map_and_break()
    newcomer = _with_map_and_break()
    named = _map_and_break(newcomer)
    assert set(named).isdisjoint(_map_and_break(held))
    _naming(newcomer, *named)
    bundle = AlignmentBundle().add_timeline(held).add_timeline(newcomer)
    assert _map_and_break(newcomer) == named
    assert _named(newcomer) == list(named)
    validate(bundle.to_dict())


@pytest.mark.parametrize("held_seed", [None, 7])
def test_a_map_or_a_break_whose_id_the_bundle_holds_is_refused(
    held_seed: int | None,
) -> None:
    held = _with_map_and_break("held", 7)
    newcomer = _quarters("newcomer", length=16)
    if held_seed is None:
        newcomer.flow_control.add(Break(_q(8), seed=7))
        kind, (_, refused) = "break", _map_and_break(held)
    else:
        newcomer.add_conversion_map(
            LinearMap(scalar=2, source_unit="quarters", target_unit="seconds", seed=7)
        )
        kind, (refused, _) = "conversion map", _map_and_break(held)
    bundle = AlignmentBundle(id="rec").add_timeline(held)
    _raises(
        f"{kind} '{refused}' of timeline 'newcomer' cannot join bundle 'rec': its id "
        f"'{refused}' is already held by {kind} '{refused}' of timeline 'held'",
        lambda: bundle.add_timeline(newcomer),
    )
    assert bundle.timeline_ids == ["held"]


def test_a_map_added_beside_one_named_leaves_the_name_with_it() -> None:
    root = _with_map_and_break()
    (first,) = root._conversion_maps
    _naming(root, first)
    root.add_conversion_map(
        LinearMap(scalar=3, source_unit="quarters", target_unit="seconds")
    )
    second = list(root._conversion_maps)[1]
    assert list(root._conversion_maps) == [first, second] and first != second
    assert _named(root) == [first]


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
