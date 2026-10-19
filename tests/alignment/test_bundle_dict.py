"""AlignmentBundle dictionary form and identity.

The strategy is documented in ``tests/alignment/README.md`` ("AlignmentBundle
Dictionary Form and Identity"). MatchClaims are not part of the dictionary form
and no assertion here consults them.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import SupportPolicy
from timetoalign.loader import RekordboxLoader
from timetoalign.loader.alignment import (
    Ieee1599Loader,
    ListenHereLoader,
    MatchfileLoader,
    MpmLoader,
    ParangonadaLoader,
    PerformancePrecisionLoader,
)
from timetoalign.loader.alignment.tilia import TiliaJsonLoader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import TimelineGroup
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
)

from ..loader.test_listen_here_loader import _CANONICAL_SPEC, _write_spec
from ..timelines.test_group_definition import STYLING, _answers, _group_manifest

# region Fixtures


def _direct_bundle() -> AlignmentBundle:
    """Two groups, an ungrouped root and a member that is a child.

    - ``score`` (quarters) groups with ``audio`` (seconds) as ``edition``;
    - ``image`` (pixels) holds the child ``holes``, which groups with
      ``midi`` (quarters) as ``roll``; ``image`` itself is ungrouped;
    - ``notes`` (seconds) is ungrouped.
    """
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    audio = ContinuousPhysicalTimeline(length=40.0, uid="audio")
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    midi = ContinuousLogicalTimeline(length=Fraction(8), uid="midi")
    notes = ContinuousPhysicalTimeline(length=12.5, uid="notes")

    bundle = AlignmentBundle(
        name="study", meta={"corpus": "test"}, support_policy=SupportPolicy.clamp
    )
    bundle.add_timeline(score, as_group="edition")
    bundle.add_timeline(audio, grouped_with="score")
    bundle.add_timeline(holes, as_group="roll")
    bundle.add_timeline(midi, grouped_with="holes")
    bundle.add_timeline(notes)
    return bundle


def _bundle_answers(bundle: AlignmentBundle) -> dict[str, Any]:
    """Every group's cross-sections, keyed by group id."""
    return {
        group_id: _answers(group) for group_id, group in sorted(bundle.groups.items())
    }


def _assert_round_trip(bundle: AlignmentBundle, **flags: bool) -> AlignmentBundle:
    """Rebuild a bundle from its dictionary and require identical answers."""
    data = bundle.to_dict(**flags)
    rebuilt = AlignmentBundle.from_dict(data)
    assert rebuilt.to_dict(**flags) == data
    assert _bundle_answers(rebuilt) == _bundle_answers(bundle)
    return rebuilt


def _parangonada() -> AlignmentBundle:
    return ParangonadaLoader.from_file(
        ensure_data("parangonar") / "Beethoven_Eroica_op35-cpjku"
    ).create_bundle()


def _matchfile() -> AlignmentBundle:
    files = sorted(ensure_data("vienna_1x22").glob("*.match"))
    return MatchfileLoader().load(*files).create_bundle()


def _mpm(project: str) -> Callable[[], AlignmentBundle]:
    def build() -> AlignmentBundle:
        return MpmLoader().load(ensure_data("mpm_toolbox") / project).create_bundle()

    return build


def _ieee1599() -> AlignmentBundle:
    path = ensure_data("ieee1599") / "SatiePetriNets/ieee1599/gymnopedie_01.xml"
    return Ieee1599Loader.from_file(path).create_bundle()


def _tilia() -> AlignmentBundle:
    path = ensure_data("score") / "bruckner5_scherzo/harnoncourt/Bruckner5_Scherzo.json"
    return TiliaJsonLoader().load(path).create_bundle()


def _performance_precision() -> AlignmentBundle:
    return PerformancePrecisionLoader.from_file(
        ensure_data("performance_precision")
    ).create_bundle()


def _rekordbox(specimen: Path) -> AlignmentBundle:
    return RekordboxLoader.from_file(specimen).create_bundle()


LOADER_BUNDLES: dict[str, Callable[..., AlignmentBundle]] = {
    "parangonada": _parangonada,
    "matchfile": _matchfile,
    "mpm_beethoven": _mpm(
        "MPRproject_1971Curzon_VariationXIV/Beethoven_op35_1971Curzon_Var14only.mpr"
    ),
    "mpm_reger": _mpm(
        "Max Reger - Moment Musical (MPM Toolbox Tutorial)/"
        "Reger - Moment Musical op 13 no 4.mpr"
    ),
    "ieee1599": _ieee1599,
    "tilia": _tilia,
    "performance_precision": _performance_precision,
    "rekordbox": _rekordbox,
}


# endregion


# region Dictionary form


def test_bundle_dict_is_exact() -> None:
    """The whole dictionary of a direct construction, pinned."""
    bundle = _direct_bundle()
    timelines = {timeline.id: timeline for timeline in bundle.timelines.values()}

    assert bundle.id == "bundle_1"
    assert bundle.to_dict() == {
        "timeskeleton": "0.7.0",
        "name": "study",
        "meta": {"corpus": "test"},
        "timelines": [
            timelines[timeline_id].to_dict()["timelines"][0]
            for timeline_id in ["audio", "image", "midi", "notes", "score"]
        ],
        "groups": [
            {
                "id": "group_1",
                "name": "edition",
                "meta": {},
                "members": ["audio", "score"],
            },
            {
                "id": "group_2",
                "name": "roll",
                "meta": {},
                "members": ["holes", "midi"],
            },
        ],
        "measure_maps": [],
        "graphical": {
            "order": {"group_1": ["score", "audio"], "group_2": ["holes", "midi"]},
            "styles": {},
            "statements": [],
        },
    }
    assert bundle.to_dict(minimal=True)["graphical"] == {
        "order": {"group_1": ["score", "audio"], "group_2": ["holes", "midi"]}
    }


def test_given_bundle_id_is_written() -> None:
    """A given id is the manifest's; a minted one is never written."""
    bundle = AlignmentBundle(id="op35")

    assert bundle.to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "id": "op35",
        "timelines": [],
    }
    rebuilt = AlignmentBundle.from_dict(_direct_bundle().to_dict())
    assert rebuilt.id == "bundle_1"
    assert "id" not in rebuilt.to_dict()


def test_direct_bundle_round_trip() -> None:
    """Rebuilt from its dictionary, a bundle writes and answers the same."""
    rebuilt = _assert_round_trip(_direct_bundle())
    _assert_round_trip(_direct_bundle(), events=True, external_references=True)

    # The support policy is a setting of the bundle, not part of the manifest.
    assert rebuilt.support_policy is SupportPolicy.omit
    assert rebuilt.timeline_to_group == {
        "score": "group_1",
        "audio": "group_1",
        "holes": "group_2",
        "midi": "group_2",
    }
    assert sorted(rebuilt.timelines) == [
        "audio",
        "holes",
        "image",
        "midi",
        "notes",
        "score",
    ]


def test_child_in_one_group_and_root_in_another() -> None:
    """A child's root may belong to another group; both are written once."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()
    bundle.add_timeline(holes, as_group="part")
    bundle.add_timeline(image, as_group="page")
    bundle.add_timeline(
        ContinuousPhysicalTimeline(length=30.0, uid="audio"), grouped_with="image"
    )

    data = bundle.to_dict()
    _assert_round_trip(bundle)

    assert [timeline["id"] for timeline in data["timelines"]] == ["audio", "image"]
    assert [group["members"] for group in data["groups"]] == [
        ["holes"],
        ["audio", "image"],
    ]


def test_no_claims_are_written() -> None:
    """The dictionary has no claim key, whatever claims the bundle holds."""
    data = _direct_bundle().to_dict()

    assert set(data) == {
        "timeskeleton",
        "name",
        "meta",
        "timelines",
        "groups",
        "measure_maps",
        "graphical",
    }


def test_missing_graphical_reads_members_order() -> None:
    """A group without an order entry joins in ``members`` order."""
    data = _direct_bundle().to_dict()
    del data["graphical"]["order"]["group_1"]

    rebuilt = AlignmentBundle.from_dict(data)

    assert rebuilt.groups["group_1"].timeline_ids == ["audio", "score"]
    assert rebuilt.groups["group_2"].timeline_ids == ["holes", "midi"]


def test_styling_round_trips_unchanged() -> None:
    """The bundle's ``styles`` and ``statements`` survive beside every order."""
    data = _direct_bundle().to_dict()
    data["graphical"].update(copy.deepcopy(STYLING))

    rebuilt = AlignmentBundle.from_dict(data)

    assert rebuilt.styling == STYLING
    assert rebuilt.to_dict() == data
    assert rebuilt.to_dict()["graphical"] == {
        "order": {"group_1": ["score", "audio"], "group_2": ["holes", "midi"]},
        **STYLING,
    }
    assert all(group.styling == {} for group in rebuilt.groups.values())
    assert rebuilt.styling["styles"] is not data["graphical"]["styles"]
    assert rebuilt.styling["statements"] is not data["graphical"]["statements"]
    data["graphical"]["statements"].clear()
    assert rebuilt.styling == STYLING


def test_styling_without_order_reads_members_order() -> None:
    """A ``graphical`` holding only styling joins every group in ``members`` order."""
    data = _direct_bundle().to_dict()
    data["graphical"] = {"statements": STYLING["statements"]}

    rebuilt = AlignmentBundle.from_dict(data)

    assert rebuilt.groups["group_1"].timeline_ids == ["audio", "score"]
    assert rebuilt.to_dict(minimal=True)["graphical"] == {
        "order": {"group_1": ["audio", "score"], "group_2": ["holes", "midi"]},
        "statements": STYLING["statements"],
    }


def test_unknown_graphical_key_is_kept() -> None:
    """A ``graphical`` key other than order, styles and statements is kept."""
    data = _direct_bundle().to_dict()
    data["graphical"].update(STYLING, frame={"inset": 4})

    rebuilt = AlignmentBundle.from_dict(data)

    assert rebuilt.styling == {**STYLING, "frame": {"inset": 4}}
    assert rebuilt.to_dict() == data


@pytest.mark.slow
@pytest.mark.parametrize("loader", list(LOADER_BUNDLES), ids=list(LOADER_BUNDLES))
def test_loader_bundle_round_trip(loader: str, request: pytest.FixtureRequest) -> None:
    """Every alignment loader's bundle survives its dictionary form."""
    # The Rekordbox specimen is a shared fixture, which skips where it is absent.
    build = LOADER_BUNDLES[loader]
    bundle = (
        build(request.getfixturevalue("rekordbox_specimen"))
        if loader == "rekordbox"
        else build()
    )

    rebuilt = _assert_round_trip(bundle)

    assert rebuilt.group_ids == sorted(bundle.group_ids)
    assert sorted(rebuilt.timelines) == sorted(bundle.timelines)
    assert rebuilt.timeline_to_group == bundle.timeline_to_group


def test_listen_here_bundle_round_trip(tmp_path: Path) -> None:
    """The ListenHere bundle (synthetic specimen) survives its dictionary form."""
    bundle = ListenHereLoader.from_file(
        _write_spec(tmp_path, _CANONICAL_SPEC)
    ).create_bundle()

    _assert_round_trip(bundle)


def _malformed(change: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """A valid bundle dictionary with one change applied."""
    data = _direct_bundle().to_dict()
    change(data)
    return data


def _second_group(data: dict[str, Any], members: list[str]) -> None:
    """Append a group named ``group_9`` holding *members*."""
    data["groups"].append(
        {"id": "group_9", "name": None, "meta": {}, "members": members}
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("timeskeleton"), "not a TimeSkeleton manifest"),
        (lambda d: d.pop("timelines"), r"lacks the key\(s\) \['timelines'\]"),
        (lambda d: d.update(id=""), "Invalid id ''"),
        (
            lambda d: d["timelines"].append(d["timelines"][0]),
            "timeline id 'audio' occurs twice",
        ),
        (
            lambda d: d["groups"].append(d["groups"][0]),
            "group id 'group_1' occurs twice",
        ),
        (
            lambda d: (
                d["groups"][0].update(id="notes"),
                d["graphical"]["order"].update(
                    notes=d["graphical"]["order"].pop("group_1")
                ),
            ),
            "group id 'notes' is also a timeline id",
        ),
        (
            lambda d: d["groups"][0]["members"].append("ghost"),
            "names member 'ghost'",
        ),
        (
            lambda d: _second_group(d, ["score"]),
            "Timeline 'score' is already a member of group 'group_1'",
        ),
        (
            lambda d: (
                d["groups"][1]["members"].append("image"),
                d["graphical"]["order"]["group_2"].append("image"),
            ),
            "Timeline 'image' cannot join group 'group_2': it is nested with "
            "member 'holes'",
        ),
        (
            lambda d: d["graphical"]["order"].update(group_7=["score"]),
            r"'graphical.order' names no group or timeline: \['group_7'\]",
        ),
        (
            lambda d: d["graphical"]["order"].update(group_1=["score"]),
            "the order of group 'group_1' is not a permutation of its members",
        ),
    ],
    ids=[
        "missing_marker",
        "missing_timelines",
        "empty_bundle_id",
        "duplicate_timeline",
        "duplicate_group",
        "group_id_is_timeline_id",
        "unknown_member",
        "member_of_two_groups",
        "parent_beside_its_child",
        "order_names_no_group",
        "order_not_a_permutation",
    ],
)
def test_from_dict_refuses_malformed_input(
    change: Callable[[dict[str, Any]], Any], message: str
) -> None:
    """Each malformed dictionary is refused with a message naming the problem."""
    data = _malformed(change)

    with pytest.raises(ValueError, match=message):
        AlignmentBundle.from_dict(data)


# endregion


# region Identity


def test_default_ids_are_minted() -> None:
    """A fresh bundle is ``bundle_1``; ``as_group`` names a minted group."""
    bundle = AlignmentBundle()
    bundle.add_timeline(ContinuousLogicalTimeline(length=Fraction(4)), as_group="a")
    bundle.add_timeline(ContinuousPhysicalTimeline(length=4.0), as_group="b")

    assert bundle.id == "bundle_1"
    assert AlignmentBundle().id == "bundle_1"
    assert bundle.group_ids == ["group_1", "group_2"]
    assert [group.name for group in bundle.groups.values()] == ["a", "b"]


def test_child_brings_its_root_ungrouped() -> None:
    """A child joining a bundle brings its root, which stays ungrouped."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()

    bundle.add_timeline(holes, as_group="roll")

    assert bundle.timeline_ids == ["holes", "image"]
    assert bundle.timeline_to_group == {"holes": "group_1"}
    assert bundle.get_timeline("image") is image


def test_ungrouped_child_registers_its_root() -> None:
    """An ungrouped participant is a root: a child joins as its root does.

    The round trip restores exactly the participants the bundle holds.
    """
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()

    bundle.add_timeline(holes)

    assert bundle.timeline_ids == ["image"]
    assert bundle.timeline_to_group == {}
    rebuilt = AlignmentBundle.from_dict(bundle.to_dict())
    assert sorted(rebuilt.timelines) == sorted(bundle.timelines) == ["image"]
    assert rebuilt.get_timeline("image").get_child("holes").length.value == 2000
    with pytest.raises(ValueError, match="already exists in bundle"):
        bundle.add_timeline(holes)


def test_loader_minted_ids_are_minted_again() -> None:
    """Three loads of one file mint ``cpt_1`` each; one bundle tells them apart."""
    path = ensure_data("score") / "bruckner5_scherzo/harnoncourt/Bruckner5_Scherzo.json"
    timelines = [
        TiliaJsonLoader().load(path).create_timeline("cpt_1") for _ in range(3)
    ]
    assert [timeline.id for timeline in timelines] == ["cpt_1", "cpt_1", "cpt_1"]
    bundle = AlignmentBundle()

    for timeline in timelines:
        bundle.add_timeline(timeline)

    assert [timeline.id for timeline in timelines] == ["cpt_1", "cpt_2", "cpt_3"]
    assert bundle.timeline_ids == ["cpt_1", "cpt_2", "cpt_3"]


def test_root_brought_along_can_be_grouped_later() -> None:
    """A root that joined with its child can still join a group of its own."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()
    bundle.add_timeline(holes, as_group="part")

    bundle.add_timeline(image, as_group="page")

    assert bundle.timeline_to_group == {"holes": "group_1", "image": "group_2"}
    with pytest.raises(ValueError, match="already exists in bundle"):
        bundle.add_timeline(image, as_group="again")


def test_member_of_two_groups_is_refused() -> None:
    """A grouped timeline cannot join another group."""
    score = ContinuousLogicalTimeline(length=Fraction(4), uid="score")
    bundle = AlignmentBundle()
    bundle.add_timeline(score, as_group="one")

    with pytest.raises(ValueError, match="already a member of group 'group_1'"):
        bundle.add_group(TimelineGroup(timelines=[score]))
    assert bundle.group_ids == ["group_1"]


def test_given_id_collision_raises_and_leaves_the_bundle() -> None:
    """A given id held by another timeline of the bundle is refused."""
    bundle = _direct_bundle()
    before = bundle.to_dict()

    with pytest.raises(ValueError, match="Timeline 'holes' cannot join bundle"):
        bundle.add_timeline(DiscreteGraphicalTimeline(length=10, uid="holes"))
    assert bundle.to_dict() == before


def test_descendant_id_collision_raises() -> None:
    """A newcomer's descendant may not share an id with the bundle's trees."""
    bundle = _direct_bundle()
    newcomer = ContinuousPhysicalTimeline(length=10.0, uid="fresh")
    newcomer.add_child(ContinuousPhysicalTimeline(length=1.0, uid="midi"), offset=0.0)

    with pytest.raises(ValueError, match="Timeline 'midi' cannot join bundle"):
        bundle.add_timeline(newcomer)
    assert "fresh" not in bundle.timelines


def test_minted_root_id_is_minted_again() -> None:
    """A minted root id held in the bundle is minted again for the newcomer."""
    bundle = AlignmentBundle()
    first = ContinuousLogicalTimeline(length=Fraction(4))
    second = ContinuousLogicalTimeline(length=Fraction(2))
    bundle.add_timeline(first)

    bundle.add_timeline(second)

    assert (first.id, second.id) == ("clt_1", "clt_2")
    assert bundle.timeline_ids == ["clt_1", "clt_2"]


def test_given_group_id_collision_raises() -> None:
    """A given group id held in the bundle is refused; a minted one re-mints."""
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])
    )
    with pytest.raises(ValueError, match="Group 'g' already exists in bundle"):
        bundle.add_group(
            TimelineGroup(id="g", timelines=[ContinuousPhysicalTimeline(length=4.0)])
        )

    minted = TimelineGroup(timelines=[ContinuousPhysicalTimeline(length=4.0)])
    other = TimelineGroup(timelines=[ContinuousPhysicalTimeline(length=2.0)])
    bundle.add_group(minted)
    bundle.add_group(other)

    assert bundle.group_ids == ["g", "group_1", "group_2"]


def test_timeline_id_equal_to_a_given_group_id_is_refused() -> None:
    """A timeline whose id a group of the bundle holds as given cannot join."""
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])
    )
    before = bundle.to_dict()

    with pytest.raises(ValueError, match="already held by a group there"):
        bundle.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid="g"))
    assert bundle.to_dict() == before


def test_minted_group_id_taken_by_a_timeline_is_minted_again() -> None:
    """A given timeline id that a minted group holds re-mints the group."""
    bundle = _direct_bundle()

    bundle.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid="group_1"))

    assert bundle.group_ids == ["group_2", "group_3"]
    assert bundle.groups["group_3"].timeline_ids == ["score", "audio"]
    assert bundle.timeline_to_group == {
        "score": "group_3",
        "audio": "group_3",
        "holes": "group_2",
        "midi": "group_2",
    }
    assert "group_1" in bundle.timelines
    _assert_round_trip(bundle)


def test_refused_newcomer_keeps_its_ids() -> None:
    """Every collision is checked before a minted id of the tree is re-minted."""
    bundle = AlignmentBundle()
    bundle.add_timeline(ContinuousPhysicalTimeline(length=4.0))
    bundle.add_timeline(ContinuousPhysicalTimeline(length=4.0, uid="B"))
    newcomer = ContinuousPhysicalTimeline(length=2.0, uid="B")
    child = ContinuousPhysicalTimeline(length=1.0)
    newcomer.add_child(child, offset=0.0)

    with pytest.raises(ValueError, match="Timeline 'B' cannot join bundle"):
        bundle.add_timeline(newcomer)
    assert (newcomer.id, child.id) == ("B", "cpt_1")
    assert bundle.timeline_ids == ["cpt_1", "B"]


def test_refused_group_keeps_its_member_ids() -> None:
    """A group refused for its given id leaves its members' minted ids alone."""
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])
    )
    member = ContinuousLogicalTimeline(length=2)
    refused = TimelineGroup(id="g", timelines=[member])

    with pytest.raises(ValueError, match="Group 'g' already exists in bundle"):
        bundle.add_group(refused)
    assert (refused.id, member.id) == ("g", "clt_1")
    assert refused.timeline_ids == ["clt_1"]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "ungrouped"])
def test_parent_and_child_cannot_share_a_group(grouped: bool) -> None:
    """``grouped_with`` a member's ancestor is refused before anything changes."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()
    bundle.add_timeline(image, as_group="page" if grouped else None)
    before = bundle.to_dict()
    message = (
        "Timeline 'holes' cannot join group 'group_1': it is nested with member "
        "'image'"
        if grouped
        else "Timeline 'holes' cannot be grouped with 'image': one is nested"
    )

    with pytest.raises(ValueError, match=message):
        bundle.add_timeline(holes, grouped_with="image")
    assert bundle.to_dict() == before


def test_from_bundles_re_mints_minted_group_ids() -> None:
    """Merging two bundles keeps both groups under distinct minted ids."""
    first = AlignmentBundle()
    first.add_timeline(ContinuousLogicalTimeline(length=4, uid="a"), as_group="x")
    second = AlignmentBundle()
    second.add_timeline(ContinuousPhysicalTimeline(length=4.0, uid="b"), as_group="y")

    merged = AlignmentBundle.from_bundles([first, second])

    assert merged.group_ids == ["group_1", "group_2"]
    assert [group.name for group in merged.groups.values()] == ["x", "y"]
    assert merged.timeline_to_group == {"a": "group_1", "b": "group_2"}


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
def test_from_bundles_keeps_each_members_lock(locked: bool) -> None:
    """Merging locks no member and leaves its source bundles as they were."""
    source = AlignmentBundle.from_dict(_group_manifest(locked))
    other = AlignmentBundle()
    other.add_timeline(ContinuousLogicalTimeline(length=4, uid="c"), as_group="y")
    before = source.to_dict()

    merged = AlignmentBundle.from_bundles([source, other])

    assert merged.timeline_to_group == {"line": "g", "plain": "g", "c": "group_1"}
    assert [merged.get_timeline(uid).is_locked for uid in ("line", "plain", "c")] == [
        locked,
        False,
        False,
    ]
    assert {root["id"]: root["locked"] for root in merged.to_dict()["timelines"]} == {
        "line": locked,
        "plain": False,
        "c": False,
    }
    assert source.get_timeline("line").is_locked is locked
    assert source.to_dict() == before


# endregion
