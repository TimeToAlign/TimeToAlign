"""AlignmentBundle dictionary form and identity.

The strategy is documented in ``tests/alignment/README.md`` ("AlignmentBundle
Dictionary Form and Identity"). MatchClaims are not part of the dictionary form
and no assertion here consults them.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from timetoshare import FORMAT_CONTEXT

from tests.helpers import bare, filed, refusal
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import SupportPolicy
from timetoalign.core.ids import IdMinter
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

    - ``score`` (quarters) groups with ``audio`` (seconds) in ``g1``, named
      ``edition``;
    - ``image`` (pixels) holds the child ``holes``, which groups with
      ``midi`` (quarters) in ``g2``, named ``roll``; ``image`` itself is
      ungrouped;
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
    bundle.add_group(TimelineGroup(id="g1", name="edition", timelines=[score, audio]))
    bundle.add_group(TimelineGroup(id="g2", name="roll", timelines=[holes, midi]))
    bundle.add_timeline(notes)
    return bundle


def _group_ids_by_name(bundle: AlignmentBundle) -> dict[str, str]:
    """Every group's id, keyed by its name."""
    return {group.name: group_id for group_id, group in bundle.groups.items()}


def _bundle_answers(bundle: AlignmentBundle) -> dict[str, Any]:
    """Every group's cross-sections, keyed by group id."""
    return {
        group_id: _answers(group) for group_id, group in sorted(bundle.groups.items())
    }


def _assert_round_trip(bundle: AlignmentBundle, **flags: bool) -> AlignmentBundle:
    """Rebuild a bundle from its dictionary and require identical answers."""
    data = bare(bundle.to_dict(**flags))
    rebuilt = AlignmentBundle.from_dict(filed(data))
    assert bare(rebuilt.to_dict(**flags)) == data
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

    assert re.fullmatch(r"bundle_[0-9a-z]{13}", bundle.id)
    assert bare(bundle.to_dict(minimal=False)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "name": "study",
        "meta": {"corpus": "test"},
        "timelines": [
            bare(timelines[timeline_id].to_dict(minimal=False))["timelines"][0]
            for timeline_id in ["audio", "image", "midi", "notes", "score"]
        ],
        "groups": [
            {
                "id": "g1",
                "name": "edition",
                "meta": {},
                "members": ["audio", "score"],
            },
            {
                "id": "g2",
                "name": "roll",
                "meta": {},
                "members": ["holes", "midi"],
            },
        ],
        "measure_maps": [],
        "resources": [],
        "graphical": {
            "order": {"g1": ["score", "audio"], "g2": ["holes", "midi"]},
            "styles": {},
            "rules": [],
            "timestamps": [],
            "matchstamps": [],
            "elisions": [],
        },
    }
    assert bare(bundle.to_dict(minimal=True))["graphical"] == {
        "order": {"g1": ["score", "audio"], "g2": ["holes", "midi"]}
    }


def test_given_bundle_id_is_written() -> None:
    """A given id is the manifest's; a minted one is never written."""
    bundle = AlignmentBundle(id="op35")

    assert bare(bundle.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "id": "op35",
        "timelines": [],
    }
    rebuilt = AlignmentBundle.from_dict(_direct_bundle().to_dict())
    assert re.fullmatch(r"bundle_[0-9a-z]{13}", rebuilt.id)
    assert "id" not in bare(rebuilt.to_dict())


def test_direct_bundle_round_trip() -> None:
    """Rebuilt from its dictionary, a bundle writes and answers the same."""
    rebuilt = _assert_round_trip(_direct_bundle())
    _assert_round_trip(_direct_bundle(), events=True, external_references=True)

    # The support policy is a setting of the bundle, not part of the manifest.
    assert rebuilt.support_policy is SupportPolicy.omit
    assert rebuilt.timeline_to_groups == {
        "score": ("g1",),
        "audio": ("g1",),
        "holes": ("g2",),
        "midi": ("g2",),
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

    data = bare(bundle.to_dict())
    _assert_round_trip(bundle)

    assert [timeline["id"] for timeline in data["timelines"]] == ["audio", "image"]
    groups = _group_ids_by_name(bundle)
    assert [group["id"] for group in data["groups"]] == sorted(groups.values())
    assert {group["id"]: group["members"] for group in data["groups"]} == {
        groups["part"]: ["holes"],
        groups["page"]: ["audio", "image"],
    }


def test_no_claims_are_written() -> None:
    """The dictionary has no claim key, whatever claims the bundle holds."""
    data = bare(_direct_bundle().to_dict(minimal=False))

    assert set(data) == {
        "@context",
        "timetoshare",
        "name",
        "meta",
        "timelines",
        "groups",
        "measure_maps",
        "resources",
        "graphical",
    }


def test_missing_graphical_reads_members_order() -> None:
    """A group without an order entry joins in ``members`` order."""
    data = bare(_direct_bundle().to_dict())
    del data["graphical"]["order"]["g1"]

    rebuilt = AlignmentBundle.from_dict(filed(data))

    assert rebuilt.groups["g1"].timeline_ids == ["audio", "score"]
    assert rebuilt.groups["g2"].timeline_ids == ["holes", "midi"]


def test_styling_round_trips_unchanged() -> None:
    """The bundle's ``styles`` and ``rules`` survive beside every order."""
    data = bare(_direct_bundle().to_dict(minimal=False))
    data["graphical"].update(copy.deepcopy(STYLING))

    rebuilt = AlignmentBundle.from_dict(filed(data))

    assert rebuilt.styling == STYLING
    assert bare(rebuilt.to_dict(minimal=False)) == data
    assert bare(rebuilt.to_dict(minimal=False))["graphical"] == {
        "order": {"g1": ["score", "audio"], "g2": ["holes", "midi"]},
        **STYLING,
        "timestamps": [],
        "matchstamps": [],
        "elisions": [],
    }
    assert all(group.styling == {} for group in rebuilt.groups.values())
    assert rebuilt.styling["styles"] is not data["graphical"]["styles"]
    assert rebuilt.styling["rules"] is not data["graphical"]["rules"]
    data["graphical"]["rules"].clear()
    assert rebuilt.styling == STYLING


def test_styling_without_order_reads_members_order() -> None:
    """A ``graphical`` holding only styling joins every group in ``members`` order."""
    data = bare(_direct_bundle().to_dict())
    data["graphical"] = {"rules": STYLING["rules"]}

    rebuilt = AlignmentBundle.from_dict(filed(data))

    assert rebuilt.groups["g1"].timeline_ids == ["audio", "score"]
    assert bare(rebuilt.to_dict(minimal=True))["graphical"] == {
        "order": {"g1": ["audio", "score"], "g2": ["holes", "midi"]},
        "rules": STYLING["rules"],
    }


def test_unknown_graphical_key_is_kept() -> None:
    """A ``graphical`` key other than order, styles and rules is kept."""
    data = bare(_direct_bundle().to_dict())
    data["graphical"].update(STYLING, frame={"inset": 4})

    rebuilt = AlignmentBundle.from_dict(filed(data))

    assert rebuilt.styling == {**STYLING, "frame": {"inset": 4}}
    assert bare(rebuilt.to_dict()) == data


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
    assert rebuilt.timeline_to_groups == bundle.timeline_to_groups


def test_listen_here_bundle_round_trip(tmp_path: Path) -> None:
    """The ListenHere bundle (synthetic specimen) survives its dictionary form."""
    bundle = ListenHereLoader.from_file(
        _write_spec(tmp_path, _CANONICAL_SPEC)
    ).create_bundle()

    _assert_round_trip(bundle)


def _malformed(change: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """A valid bundle dictionary with one change applied."""
    data = bare(_direct_bundle().to_dict())
    change(data)
    return data


@pytest.mark.parametrize(
    ("change", "code", "path", "message"),
    [
        (
            lambda d: d.pop("timetoshare"),
            "not_a_manifest",
            "#",
            "not a TimeToShare manifest",
        ),
        (
            lambda d: d.pop("timelines"),
            "schema",
            "#",
            "'timelines' is a required property",
        ),
        (
            lambda d: d.update(id=""),
            "schema",
            "#/id",
            r"'#' does not match '^#[\\s\\S]'",
        ),
        (
            lambda d: d["timelines"].append(d["timelines"][0]),
            "duplicate_id",
            "#/timelines/5/id",
            "'audio' is not unique in the manifest, first at #/timelines/0/id",
        ),
        (
            lambda d: d["groups"].append(d["groups"][0]),
            "duplicate_id",
            "#/groups/2/id",
            "'g1' is not unique in the manifest, first at #/groups/0/id",
        ),
        (
            lambda d: (
                d["groups"][0].update(id="notes"),
                d["graphical"]["order"].update(notes=d["graphical"]["order"].pop("g1")),
            ),
            "duplicate_id",
            "#/groups/0/id",
            "'notes' is not unique in the manifest, first at #/timelines/3/id",
        ),
        (
            lambda d: d["groups"][0]["members"].append("ghost"),
            "group_member",
            "#/groups/0/members/2",
            "'ghost' names no timeline or virtual timeline of this manifest",
        ),
        (
            lambda d: d["groups"][0]["members"].append("score"),
            "schema",
            "#/groups/0/members",
            "['#audio', '#score', '#score'] has non-unique elements",
        ),
        (
            lambda d: (
                d["groups"][1]["members"].append("image"),
                d["graphical"]["order"]["g2"].append("image"),
            ),
            "group_member",
            "#/groups/1/members/2",
            "member 'image' contains member 'holes' of the same group",
        ),
        (
            lambda d: d["graphical"]["order"].update(g7=["score"]),
            "order_key",
            "#/graphical/order/g7",
            "'g7' names no group, timeline or virtual timeline of this manifest",
        ),
        (
            lambda d: d["graphical"]["order"].update(g1=["score"]),
            "display_order",
            "#/graphical/order/g1",
            "member 'audio' of group 'g1' is missing",
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
        "member_named_twice",
        "parent_beside_its_child",
        "order_names_no_group",
        "order_not_a_permutation",
    ],
)
def test_from_dict_refuses_malformed_input(
    change: Callable[[dict[str, Any]], Any], code: str, path: str, message: str
) -> None:
    """Each malformed dictionary is refused by the format's validator, its words."""
    data = _malformed(change)

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))
    assert refusal(refused.value) == (code, path, message)


# endregion


# region Identity


def test_default_ids_are_minted() -> None:
    """A bundle and a group ``as_group`` names mint their ids at construction."""
    bundle = AlignmentBundle(seed=5)
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=Fraction(4)), as_group="a", seed=6
    )
    bundle.add_timeline(ContinuousPhysicalTimeline(length=4.0), as_group="b")

    assert bundle.id == IdMinter(5).mint("bundle")
    assert AlignmentBundle(seed=5).id == bundle.id
    first, second = bundle.group_ids
    assert first == IdMinter(6).mint("group")
    assert re.fullmatch(r"group_[0-9a-z]{13}", second)
    assert second != first
    assert [group.name for group in bundle.groups.values()] == ["a", "b"]
    assert AlignmentBundle().id != AlignmentBundle().id


def test_child_brings_its_root_ungrouped() -> None:
    """A child joining a bundle brings its root, which stays ungrouped."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()

    bundle.add_timeline(holes, as_group="roll")

    assert bundle.timeline_ids == ["holes", "image"]
    (roll,) = bundle.group_ids
    assert bundle.timeline_to_groups == {"holes": (roll,)}
    assert bundle.groups[roll].name == "roll"
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
    assert bundle.timeline_to_groups == {}
    rebuilt = AlignmentBundle.from_dict(bundle.to_dict())
    assert sorted(rebuilt.timelines) == sorted(bundle.timelines) == ["image"]
    assert rebuilt.get_timeline("image").get_child("holes").length.value == 2000
    with pytest.raises(ValueError, match="already exists in bundle"):
        bundle.add_timeline(holes)


def test_loader_ids_held_twice_are_refused() -> None:
    """Three loads of one file mint one id; a bundle refuses it a second time."""
    path = ensure_data("score") / "bruckner5_scherzo/harnoncourt/Bruckner5_Scherzo.json"
    timelines = [
        TiliaJsonLoader().load(path).create_timeline("HIERARCHY_TIMELINE_0")
        for _ in range(3)
    ]
    minted = timelines[0].id
    assert [timeline.id for timeline in timelines] == [minted] * 3
    bundle = AlignmentBundle()
    bundle.add_timeline(timelines[0])
    before = bare(bundle.to_dict())

    for timeline in timelines[1:]:
        with pytest.raises(ValueError) as refused:
            bundle.add_timeline(timeline)
        assert str(refused.value) == (
            f"timeline '{minted}' cannot join bundle '{bundle.id}': its id "
            f"'{minted}' is already held by timeline '{minted}'"
        )

    assert [timeline.id for timeline in timelines] == [minted] * 3
    assert bundle.timeline_ids == [minted]
    assert bare(bundle.to_dict()) == before


def test_root_brought_along_can_be_grouped_later() -> None:
    """A root that joined with its child can still join a group of its own."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()
    bundle.add_timeline(holes, as_group="part")

    bundle.add_timeline(image, as_group="page")

    groups = _group_ids_by_name(bundle)
    assert bundle.timeline_to_groups == {
        "holes": (groups["part"],),
        "image": (groups["page"],),
    }
    with pytest.raises(ValueError, match="already exists in bundle"):
        bundle.add_timeline(image)

    # A grouped timeline joins a further group.
    bundle.add_timeline(image, as_group="again")

    groups = _group_ids_by_name(bundle)
    assert bundle.timeline_to_groups == {
        "holes": (groups["part"],),
        "image": (groups["page"], groups["again"]),
    }


def test_member_of_another_group_is_admitted() -> None:
    """A pre-built group may hold a timeline already grouped in the bundle."""
    score = ContinuousLogicalTimeline(length=Fraction(4), uid="score")
    bundle = AlignmentBundle()
    bundle.add_timeline(score, as_group="one")
    (one,) = bundle.group_ids
    group = TimelineGroup(timelines=[score])

    bundle.add_group(group)

    assert bundle.group_ids == [one, group.id]
    assert bundle.timeline_to_groups == {"score": (one, group.id)}
    assert bundle.timeline_ids == ["score"]


def _shared_member_bundle() -> AlignmentBundle:
    """Groups ``g1`` (``t``, ``a``, ``u``) and ``g2`` (``t``, ``u``, ``b``)."""
    t = ContinuousLogicalTimeline(length=Fraction(12), uid="t")
    u = ContinuousLogicalTimeline(length=Fraction(6), uid="u")
    a = ContinuousLogicalTimeline(length=Fraction(8), uid="a")
    b = ContinuousLogicalTimeline(length=Fraction(9), uid="b")
    bundle = AlignmentBundle()
    bundle.add_group(TimelineGroup(id="g1", timelines=[t, a, u]))
    bundle.add_group(TimelineGroup(id="g2", timelines=[t, u, b]))
    return bundle


@pytest.mark.parametrize("minimal", [False, True], ids=["complete", "minimal"])
def test_groups_sharing_a_member_round_trip(minimal: bool) -> None:
    """A member of two groups is written in both and read back as one object."""
    bundle = _shared_member_bundle()
    data = bare(bundle.to_dict(minimal=minimal))

    assert [timeline["id"] for timeline in data["timelines"]] == ["a", "b", "t", "u"]
    assert [(group["id"], group["members"]) for group in data["groups"]] == [
        ("g1", ["a", "t", "u"]),
        ("g2", ["b", "t", "u"]),
    ]
    assert data["graphical"]["order"] == {"g1": ["t", "a", "u"], "g2": ["t", "u", "b"]}
    rebuilt = _assert_round_trip(bundle, minimal=minimal)
    assert rebuilt.timeline_to_groups == {
        "t": ("g1", "g2"),
        "a": ("g1",),
        "u": ("g1", "g2"),
        "b": ("g2",),
    }
    assert rebuilt.groups["g1"].get_timeline("t") is rebuilt.get_timeline("t")
    assert rebuilt.groups["g2"].get_timeline("t") is rebuilt.get_timeline("t")


def test_given_id_collision_raises_and_leaves_the_bundle() -> None:
    """A given id held by another timeline of the bundle is refused."""
    bundle = _direct_bundle()
    before = bare(bundle.to_dict())

    with pytest.raises(ValueError) as refused:
        bundle.add_timeline(DiscreteGraphicalTimeline(length=10, uid="holes"))
    assert str(refused.value) == (
        f"timeline 'holes' cannot join bundle '{bundle.id}': its id 'holes' is "
        "already held by timeline 'holes'"
    )
    assert bare(bundle.to_dict()) == before


def test_descendant_id_collision_raises() -> None:
    """A newcomer's descendant may not share an id with the bundle's trees."""
    bundle = _direct_bundle()
    newcomer = ContinuousPhysicalTimeline(length=10.0, uid="fresh")
    newcomer.add_child(ContinuousPhysicalTimeline(length=1.0, uid="midi"), offset=0.0)

    with pytest.raises(ValueError, match="timeline 'midi' cannot join bundle"):
        bundle.add_timeline(newcomer)
    assert "fresh" not in bundle.timelines


def test_minted_root_id_held_is_refused() -> None:
    """A minted id joins as it is; held already, it is refused, never re-minted."""
    bundle = AlignmentBundle()
    first = ContinuousLogicalTimeline(length=Fraction(4))
    second = ContinuousLogicalTimeline(length=Fraction(2))
    ids = (first.id, second.id)
    bundle.add_timeline(first)
    bundle.add_timeline(second)

    assert (first.id, second.id) == ids
    assert first.id != second.id
    assert bundle.timeline_ids == list(ids)

    seeded = ContinuousLogicalTimeline(length=Fraction(4), seed=1)
    again = ContinuousLogicalTimeline(length=Fraction(4), seed=1)
    assert seeded.id == again.id == IdMinter(1).mint("clt")
    bundle.add_timeline(seeded)
    before = bare(bundle.to_dict())
    with pytest.raises(ValueError) as refused:
        bundle.add_timeline(again)
    assert str(refused.value) == (
        f"timeline '{seeded.id}' cannot join bundle '{bundle.id}': its id "
        f"'{seeded.id}' is already held by timeline '{seeded.id}'"
    )
    assert again.id == seeded.id
    assert bundle.timeline_ids == [*ids, seeded.id]
    assert bare(bundle.to_dict()) == before


def test_given_group_id_collision_raises() -> None:
    """A given group id held in the bundle is refused; minted ones join as they are."""
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

    assert bundle.group_ids == ["g", minted.id, other.id]


def test_timeline_id_equal_to_a_given_group_id_is_refused() -> None:
    """A timeline whose id a group of the bundle holds as given cannot join."""
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])
    )
    before = bare(bundle.to_dict())

    with pytest.raises(ValueError) as refused:
        bundle.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid="g"))
    assert str(refused.value) == (
        f"timeline 'g' cannot join bundle '{bundle.id}': its id 'g' is already "
        "held by group 'g'"
    )
    assert bare(bundle.to_dict()) == before


def test_timeline_id_equal_to_a_minted_group_id_is_refused() -> None:
    """A minted group id refuses a timeline that carries it, as a given one does."""
    bundle = AlignmentBundle()
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=Fraction(16), uid="score"), as_group="edition"
    )
    (edition,) = bundle.group_ids
    before = bare(bundle.to_dict())

    with pytest.raises(ValueError) as refused:
        bundle.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid=edition))
    assert str(refused.value) == (
        f"timeline '{edition}' cannot join bundle '{bundle.id}': its id "
        f"'{edition}' is already held by group '{edition}'"
    )
    assert bundle.group_ids == [edition]
    assert bundle.timeline_to_groups == {"score": (edition,)}
    assert bare(bundle.to_dict()) == before


def test_refused_newcomer_keeps_its_ids() -> None:
    """A refused newcomer and its minted descendants keep their ids."""
    bundle = AlignmentBundle()
    held = ContinuousPhysicalTimeline(length=4.0)
    bundle.add_timeline(held)
    bundle.add_timeline(ContinuousPhysicalTimeline(length=4.0, uid="B"))
    newcomer = ContinuousPhysicalTimeline(length=2.0, uid="B")
    child = ContinuousPhysicalTimeline(length=1.0)
    newcomer.add_child(child, offset=0.0)
    child_id = child.id

    with pytest.raises(ValueError, match="timeline 'B' cannot join bundle"):
        bundle.add_timeline(newcomer)
    assert (newcomer.id, child.id) == ("B", child_id)
    assert bundle.timeline_ids == [held.id, "B"]


def test_refused_group_keeps_its_member_ids() -> None:
    """A group refused for its given id leaves its members' minted ids alone."""
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(id="g", timelines=[ContinuousLogicalTimeline(length=4)])
    )
    member = ContinuousLogicalTimeline(length=2)
    member_id = member.id
    refused = TimelineGroup(id="g", timelines=[member])

    with pytest.raises(ValueError, match="Group 'g' already exists in bundle"):
        bundle.add_group(refused)
    assert (refused.id, member.id) == ("g", member_id)
    assert refused.timeline_ids == [member_id]


@pytest.mark.parametrize("grouped", [True, False], ids=["grouped", "ungrouped"])
def test_parent_and_child_cannot_share_a_group(grouped: bool) -> None:
    """``grouped_with`` a member's ancestor is refused before anything changes."""
    image = DiscreteGraphicalTimeline(length=5000, uid="image")
    holes = DiscreteGraphicalTimeline(length=2000, uid="holes")
    image.add_child(holes, offset=1000)
    bundle = AlignmentBundle()
    bundle.add_timeline(image, as_group="page" if grouped else None)
    before = bare(bundle.to_dict())
    message = (
        f"Timeline 'holes' cannot join group '{bundle.group_ids[0]}': it is nested "
        "with member 'image'"
        if grouped
        else "Timeline 'holes' cannot be grouped with 'image': one is nested"
    )

    with pytest.raises(ValueError, match=message):
        bundle.add_timeline(holes, grouped_with="image")
    assert bare(bundle.to_dict()) == before


def test_from_bundles_keeps_minted_group_ids() -> None:
    """Merging keeps every group under its id; a group id held twice is refused."""
    first = AlignmentBundle()
    first.add_timeline(ContinuousLogicalTimeline(length=4, uid="a"), as_group="x")
    second = AlignmentBundle()
    second.add_timeline(ContinuousPhysicalTimeline(length=4.0, uid="b"), as_group="y")
    (x,), (y,) = first.group_ids, second.group_ids

    merged = AlignmentBundle.from_bundles([first, second])

    assert merged.group_ids == [x, y]
    assert [group.name for group in merged.groups.values()] == ["x", "y"]
    assert merged.timeline_to_groups == {"a": (x,), "b": (y,)}

    seeded = AlignmentBundle()
    seeded.add_timeline(
        ContinuousLogicalTimeline(length=4, uid="c"), as_group="x", seed=1
    )
    again = AlignmentBundle()
    again.add_timeline(
        ContinuousPhysicalTimeline(length=4.0, uid="d"), as_group="y", seed=1
    )
    shared = IdMinter(1).mint("group")
    assert seeded.group_ids == again.group_ids == [shared]
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_bundles([seeded, again])
    assert str(refused.value) == f"Group '{shared}' already exists in bundle"
    assert (seeded.group_ids, again.group_ids) == ([shared], [shared])


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
def test_from_bundles_keeps_each_members_lock(locked: bool) -> None:
    """Merging locks no member and leaves its source bundles as they were."""
    source = AlignmentBundle.from_dict(filed(_group_manifest(locked)))
    other = AlignmentBundle()
    other.add_timeline(ContinuousLogicalTimeline(length=4, uid="c"), as_group="y")
    (y,) = other.group_ids
    before = bare(source.to_dict(minimal=False))

    merged = AlignmentBundle.from_bundles([source, other])

    assert merged.timeline_to_groups == {
        "line": ("g",),
        "plain": ("g",),
        "c": (y,),
    }
    assert [merged.get_timeline(uid).is_locked for uid in ("line", "plain", "c")] == [
        locked,
        False,
        False,
    ]
    assert {
        root["id"]: root["locked"]
        for root in bare(merged.to_dict(minimal=False))["timelines"]
    } == {
        "line": locked,
        "plain": False,
        "c": False,
    }
    assert source.get_timeline("line").is_locked is locked
    assert bare(source.to_dict(minimal=False)) == before


# endregion
