"""TimelineGroup reduced to its definition: one start, one end, linear between.

The strategy is documented in ``tests/timelines/README.md`` ("Timeline Groups").
"""

from __future__ import annotations

import copy
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.time import express_as
from timetoalign.timelines import SegmentLine, TimelineGroup
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
)

# region Fixtures


def _three_types() -> TimelineGroup:
    """A fraction, a float and an int member over one span."""
    return TimelineGroup(
        timelines=[
            ContinuousLogicalTimeline(length=Fraction(16), uid="A"),
            ContinuousPhysicalTimeline(length=40.0, uid="B"),
            DiscreteGraphicalTimeline(length=4800, uid="C"),
        ]
    )


def _child_member() -> tuple[TimelineGroup, DiscreteGraphicalTimeline]:
    """A group of a logical timeline and the child of an image."""
    root = DiscreteGraphicalTimeline(length=5000, uid="R")
    root.add_child(DiscreteGraphicalTimeline(length=2000, uid="K"), offset=1000)
    group = TimelineGroup(
        timelines=[ContinuousLogicalTimeline(length=Fraction(8), uid="A2")]
    )
    group.add_timeline(root.get_child("K"))
    return group, root


def _converted(group: TimelineGroup, value: Any, source: str, target: str) -> Any:
    """The target member's canonical value at a position on the source."""
    return (
        group.get_timestamp_at(value, source)
        .get_coordinate_for(target, format="coordinate")
        .value
    )


def _answers(group: TimelineGroup) -> dict[tuple[str, Any], dict[str, Any]]:
    """Every member's cross-section at 0, a third, a half and its length."""
    answers: dict[tuple[str, Any], dict[str, Any]] = {}
    for member in group:
        length = Fraction(member.length.value)
        for share in (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(1)):
            position = express_as(length * share, member.number_type)
            stamp = group.get_timestamp_at(position, member.id)
            answers[(member.id, position)] = {
                timeline_id: (coordinate.value, type(coordinate.value), coordinate.unit)
                for timeline_id, coordinate in stamp.coordinates.items()
            }
    return answers


def _assert_round_trip(group: TimelineGroup, **flags: bool) -> TimelineGroup:
    """Rebuild a group from its dictionary and require identical answers."""
    data = group.to_dict(**flags)
    rebuilt = TimelineGroup.from_dict(data)
    assert rebuilt.to_dict(**flags) == data
    assert rebuilt.timeline_ids == group.timeline_ids
    assert _answers(rebuilt) == _answers(group)
    return rebuilt


# endregion


# region Exact linear conversion


@pytest.mark.parametrize(
    ("source", "value", "target", "expected"),
    [
        ("A", 8, "B", 20.0),
        ("A", 8, "C", 2400),
        ("A", Fraction(1, 3), "B", 0.8333333333333334),
        ("A", Fraction(1, 3), "C", 100),
        ("C", 1, "A", Fraction(1, 300)),
        ("C", 1, "B", float(Fraction(1, 120))),
        ("A", 16, "B", 40.0),
        ("A", 16, "C", 4800),
        ("A", Fraction(1, 7), "C", 43),
        ("A", Fraction(1, 600), "C", 0),
        ("A", Fraction(1, 200), "C", 2),
        ("B", 20.0, "A", Fraction(8)),
    ],
)
def test_linear_conversion_is_exact(
    source: str, value: Any, target: str, expected: Any
) -> None:
    """``p · len_B / len_A`` on exact ratios, written in the target's type."""
    result = _converted(_three_types(), value, source, target)

    assert result == expected
    assert type(result) is type(expected)


def test_float_one_third_is_rounded_once() -> None:
    """The float answer is the exact 5/6 rounded once, not a float pipeline."""
    assert _converted(_three_types(), Fraction(1, 3), "A", "B") == float(Fraction(5, 6))


def test_float_query_enters_as_its_dyadic() -> None:
    """A float on a float member converts from its exact binary value."""
    group = _three_types()

    assert _converted(group, 0.1, "B", "A") == Fraction(0.1) * Fraction(16, 40)


def test_stamp_provenance_marks_the_stored_rows() -> None:
    """The common start and end are stored rows; positions between are not."""
    group = _three_types()

    start = group.get_timestamp_at(0, "A")
    end = group.get_timestamp_at(16, "A")
    inside = group.get_timestamp_at(8, "A")

    assert start.is_interpolated is False
    assert end.is_interpolated is False
    assert inside.is_interpolated is True


def test_position_outside_the_member_raises() -> None:
    """Every member spans the group; only a position off the member fails."""
    group = _three_types()

    with pytest.raises(ValueError, match="outside range for 'A'"):
        group.get_timestamp_at(Fraction(33, 2), "A")
    with pytest.raises(ValueError, match="outside range for 'C'"):
        group.get_timestamp_at(-1, "C")


def test_stored_rows_are_start_and_end_exact() -> None:
    """Each member at 0 and at its length, in its declared number type."""
    group = TimelineGroup(
        timelines=[
            ContinuousLogicalTimeline(length=Fraction(5, 3), uid="A"),
            ContinuousPhysicalTimeline(length=40.0, uid="B"),
            DiscreteGraphicalTimeline(length=4800, uid="C"),
        ]
    )
    frame = group.get_timestamp_table(format="dataframe")

    assert frame["A (quarters)"].tolist() == [Fraction(0), Fraction(5, 3)]
    assert frame["B (seconds)"].tolist() == [0.0, 40.0]
    assert frame["C (pixels)"].tolist() == [0, 4800]


def test_interval_of_a_member_is_its_whole_axis() -> None:
    """``get_interval_for`` is ``[0, length]`` of the member."""
    interval = _three_types().get_interval_for("A")

    assert interval.start == Coordinate(Fraction(0), TimeUnit.quarters)
    assert interval.end == Coordinate(Fraction(16), TimeUnit.quarters)
    assert type(interval.start.value) is Fraction


# endregion


# region Child member


def test_child_member_converts_from_its_own_zero() -> None:
    """A child spans the group from its own 0; its offset places it in the root."""
    group, root = _child_member()

    assert _converted(group, 2, "A2", "K") == 500
    assert group.get_timestamp_at(2, "A2").present_timelines == ["A2", "K"]
    assert root.get_timestamp_at(1500).get_coordinate_for("K", format="int") == 500


def test_child_member_writes_its_root() -> None:
    """The root of a child member is written; the member is the child's id."""
    group, _ = _child_member()
    data = group.to_dict()

    assert [timeline["id"] for timeline in data["timelines"]] == ["A2", "R"]
    assert data["groups"][0]["members"] == ["A2", "K"]
    assert data["graphical"] == {
        "order": {"group_1": ["A2", "K"]},
        "styles": {},
        "statements": [],
    }


# endregion


# region Dictionary form


def test_group_dict_is_exact() -> None:
    """The whole dictionary of a group, pinned."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    image = DiscreteGraphicalTimeline(length=4800, uid="image")
    group = TimelineGroup(
        name="edition", meta={"source": "test"}, timelines=[score, image]
    )

    assert group.to_dict() == {
        "timetoshare": "0.1.0",
        "name": None,
        "meta": {},
        "timelines": [image.to_dict()["timelines"][0], score.to_dict()["timelines"][0]],
        "groups": [
            {
                "id": "group_1",
                "name": "edition",
                "meta": {"source": "test"},
                "members": ["image", "score"],
            }
        ],
        "measure_maps": [],
        "graphical": {
            "order": {"group_1": ["score", "image"]},
            "styles": {},
            "statements": [],
        },
    }
    assert group.to_dict(minimal=True) == {
        "timetoshare": "0.1.0",
        "timelines": [
            {"id": "image", "unit": "pixels", "length": 4800},
            {"id": "score", "unit": "quarters", "length": [16, 1]},
        ],
        "groups": [
            {
                "id": "group_1",
                "name": "edition",
                "meta": {"source": "test"},
                "members": ["image", "score"],
            }
        ],
        "graphical": {"order": {"group_1": ["score", "image"]}},
    }


def test_flags_pass_through_to_every_root() -> None:
    """``events`` and ``external_references`` reach every root's dictionary."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    score.add_events([{"start": Fraction(1, 3), "event_type": "Note"}])
    group = TimelineGroup(timelines=[score])

    data = group.to_dict(events=True, external_references=True)

    assert (
        data["timelines"]
        == score.to_dict(events=True, external_references=True)["timelines"]
    )


@pytest.mark.parametrize(
    "build",
    [_three_types, lambda: _child_member()[0]],
    ids=["three_types", "child_member"],
)
def test_group_round_trip(build: Any) -> None:
    """Rebuilt from its dictionary, a group writes and answers the same."""
    _assert_round_trip(build())
    _assert_round_trip(build(), events=True, external_references=True)


def test_round_trip_of_two_children_of_one_root() -> None:
    """Two members inside one root write the root once."""
    root = ContinuousPhysicalTimeline(length=90.0, uid="audio")
    root.add_child(ContinuousPhysicalTimeline(length=30.0, uid="first"), offset=0.0)
    root.add_child(ContinuousPhysicalTimeline(length=30.0, uid="last"), offset=60.0)
    group = TimelineGroup(timelines=[root.get_child("last"), root.get_child("first")])

    rebuilt = _assert_round_trip(group)

    assert [timeline["id"] for timeline in group.to_dict()["timelines"]] == ["audio"]
    assert rebuilt.timeline_ids == ["last", "first"]


def test_order_defaults_to_members_without_graphical() -> None:
    """Without a ``graphical`` part the members join in ``members`` order."""
    data = _three_types().to_dict()
    del data["graphical"]

    assert TimelineGroup.from_dict(data).timeline_ids == ["A", "B", "C"]


STYLING: dict[str, Any] = {
    "styles": {"frame-highlight": {"stroke": "#7F00FF", "stroke-width": 2}},
    "statements": [
        {"select": {"element": "frame"}, "set": "frame-highlight"},
        {
            "select": {
                "element": "name",
                "addresses": [{"id": "group_1", "part": ["name"]}],
            },
            "set": {"fill": "#7F00FF"},
        },
    ],
}
"""A styling block the library cannot produce itself, three levels deep."""


@pytest.mark.parametrize(
    "keys",
    [("styles", "statements"), ("styles",), ("statements",)],
    ids=["both", "styles", "statements"],
)
def test_styling_round_trips_unchanged(keys: tuple[str, ...]) -> None:
    """``styles`` and ``statements`` survive the round trip beside ``order``."""
    data = _three_types().to_dict()
    data["graphical"].update(copy.deepcopy({key: STYLING[key] for key in keys}))
    expected = {key: STYLING[key] for key in keys}

    rebuilt = TimelineGroup.from_dict(data)

    assert rebuilt.styling == expected
    assert rebuilt.to_dict() == data
    assert rebuilt.to_dict(minimal=True)["graphical"] == {
        "order": {"group_1": ["A", "B", "C"]},
        **expected,
    }
    for key in keys:
        assert rebuilt.styling[key] is not data["graphical"][key]
    data["graphical"][keys[0]].clear()
    assert rebuilt.styling == expected


def test_absent_styling_stays_absent() -> None:
    """A group without styling writes ``graphical`` as its order alone."""
    rebuilt = TimelineGroup.from_dict(_three_types().to_dict())

    assert rebuilt.styling == {}
    assert rebuilt.to_dict(minimal=True)["graphical"] == {
        "order": {"group_1": ["A", "B", "C"]}
    }


def test_styling_without_order_reads_members_order() -> None:
    """A ``graphical`` holding only styling joins the members in ``members`` order."""
    data = _three_types().to_dict()
    data["groups"][0]["members"] = ["C", "A", "B"]
    data["graphical"] = {"styles": STYLING["styles"]}

    rebuilt = TimelineGroup.from_dict(data)

    assert rebuilt.timeline_ids == ["C", "A", "B"]
    assert rebuilt.to_dict(minimal=True)["graphical"] == {
        "order": {"group_1": ["C", "A", "B"]},
        "styles": STYLING["styles"],
    }


def test_unknown_graphical_key_is_kept() -> None:
    """A ``graphical`` key other than order, styles and statements is kept."""
    data = _three_types().to_dict()
    data["graphical"].update(STYLING, frame={"inset": 4})

    rebuilt = TimelineGroup.from_dict(data)

    assert rebuilt.styling == {**STYLING, "frame": {"inset": 4}}
    assert rebuilt.to_dict() == data


def test_manifest_meta_round_trips() -> None:
    """The manifest's own ``meta`` belongs to the manifest, not to the group."""
    data = _three_types().to_dict()
    data["meta"] = {"source": "an edition"}

    rebuilt = TimelineGroup.from_dict(data)

    assert rebuilt.meta == {}
    assert rebuilt.to_dict() == data


def _malformed(change: Any) -> dict[str, Any]:
    """A valid group dictionary with one change applied."""
    data = _three_types().to_dict()
    change(data)
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("timetoshare"), "not a TimeToShare manifest"),
        (lambda d: d["groups"][0].pop("members"), r"lacks the key\(s\) \['members'\]"),
        (lambda d: d["groups"][0]["members"].append("Z"), "names member 'Z'"),
        (lambda d: d["groups"][0]["members"].append("A"), "names a member twice"),
        (
            lambda d: d["timelines"].append(d["timelines"][0]),
            "the id 'A' is held twice in the manifest: by timeline 'A' and by "
            "timeline 'A'",
        ),
        (
            lambda d: d["graphical"]["order"].update(group_1=["A", "B"]),
            "not a permutation of its members",
        ),
        (
            lambda d: d["graphical"].update(order=["A", "B", "C"]),
            "'graphical.order' must map group or timeline ids to orders",
        ),
        (
            lambda d: (d["groups"][0].update(id="A"), d["graphical"].pop("order")),
            "the id 'A' is held twice in the manifest: by timeline 'A' and by "
            "group 'A'",
        ),
        (
            lambda d: (d["groups"][0].update(id="a\tb"), d["graphical"].pop("order")),
            r"Invalid id 'a\\tb'",
        ),
    ],
    ids=[
        "missing_marker",
        "missing_members",
        "unknown_member",
        "duplicate_member",
        "duplicate_timeline",
        "short_order",
        "order_as_a_list",
        "group_id_is_timeline_id",
        "invalid_group_id",
    ],
)
def test_from_dict_refuses_malformed_input(change: Any, message: str) -> None:
    """Each malformed dictionary is refused with a message naming the problem."""
    data = _malformed(change)

    with pytest.raises(ValueError, match=message):
        TimelineGroup.from_dict(data)


def test_from_dict_refuses_a_parent_beside_its_child() -> None:
    """A group object naming a member's root is refused, naming both."""
    group, _ = _child_member()
    data = group.to_dict()
    data["groups"][0]["members"].append("R")
    data["graphical"]["order"]["group_1"].append("R")

    with pytest.raises(
        ValueError,
        match="Timeline 'R' cannot join group 'group_1': it is nested with member 'K'",
    ):
        TimelineGroup.from_dict(data)


def test_groups_sharing_a_member_read_each_group() -> None:
    """Either group of a manifest whose two groups share a member reads alone."""
    shared = ContinuousLogicalTimeline(length=Fraction(12), uid="t")
    bundle = AlignmentBundle()
    bundle.add_group(
        TimelineGroup(
            id="g1",
            timelines=[shared, ContinuousLogicalTimeline(length=Fraction(8), uid="a")],
        )
    )
    bundle.add_group(
        TimelineGroup(
            id="g2",
            timelines=[shared, ContinuousLogicalTimeline(length=Fraction(9), uid="b")],
        )
    )
    data = bundle.to_dict()

    first = TimelineGroup.from_dict(data, group_id="g1")
    second = TimelineGroup.from_dict(data, group_id="g2")

    assert first.timeline_ids == ["t", "a"]
    assert second.timeline_ids == ["t", "b"]
    assert _converted(first, Fraction(3), "t", "a") == Fraction(2)
    assert _converted(second, Fraction(3), "t", "b") == Fraction(9, 4)


# endregion


# region Identity


def test_default_id_is_minted_statelessly() -> None:
    """Two fresh groups in one process are both ``group_1``."""
    assert TimelineGroup().id == "group_1"
    assert TimelineGroup().id == "group_1"


def test_given_id_collision_raises_and_leaves_the_group() -> None:
    """A given id held by another timeline of the group's trees is refused."""
    group = _three_types()
    before = group.to_dict()

    with pytest.raises(ValueError) as refused:
        group.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid="B"))
    assert str(refused.value) == (
        "timeline 'B' cannot join group 'group_1': its id 'B' is already held by "
        "timeline 'B'"
    )
    assert group.to_dict() == before


def test_descendant_id_collision_raises() -> None:
    """A newcomer's descendant may not share an id with the group's trees."""
    group, _ = _child_member()
    newcomer = ContinuousLogicalTimeline(length=Fraction(4), uid="other")
    newcomer.add_child(ContinuousLogicalTimeline(length=Fraction(1), uid="R"), offset=0)

    with pytest.raises(ValueError, match="timeline 'R' cannot join group"):
        group.add_timeline(newcomer)
    assert newcomer.id == "other"


def test_minted_root_id_is_minted_again() -> None:
    """A minted root id held in the group is minted again for the newcomer."""
    held = ContinuousLogicalTimeline(length=Fraction(4))
    group = TimelineGroup(timelines=[held])
    newcomer = ContinuousLogicalTimeline(length=Fraction(2))

    group.add_timeline(newcomer)

    assert (held.id, newcomer.id) == ("clt_1", "clt_2")
    assert group.timeline_ids == ["clt_1", "clt_2"]


def test_minted_descendant_id_is_minted_again() -> None:
    """A held minted descendant id is minted again; its parent keeps it."""
    held = ContinuousLogicalTimeline(length=Fraction(4))
    group = TimelineGroup(timelines=[held])
    newcomer = ContinuousLogicalTimeline(length=Fraction(2), uid="part_parent")
    child = ContinuousLogicalTimeline(length=Fraction(1))
    newcomer.add_child(child, offset=0)

    group.add_timeline(newcomer)

    assert (held.id, child.id) == ("clt_1", "clt_2")
    assert newcomer.get_child("clt_2") is child
    assert group.get_timestamp_at(1, "clt_1").present_timelines == [
        "clt_1",
        "part_parent",
        "clt_2",
    ]


def test_refused_newcomer_keeps_its_ids() -> None:
    """Every collision is checked before a minted id of the tree is re-minted."""
    group = TimelineGroup(
        timelines=[
            ContinuousPhysicalTimeline(length=4.0),
            ContinuousPhysicalTimeline(length=4.0, uid="B"),
        ]
    )
    newcomer = ContinuousPhysicalTimeline(length=2.0, uid="B")
    child = ContinuousPhysicalTimeline(length=1.0)
    newcomer.add_child(child, offset=0.0)

    with pytest.raises(ValueError, match="timeline 'B' cannot join group"):
        group.add_timeline(newcomer)
    assert (newcomer.id, child.id) == ("B", "cpt_1")
    assert group.timeline_ids == ["cpt_1", "B"]


def test_group_id_may_not_be_a_timeline_id() -> None:
    """A timeline whose id is the group's own given id cannot join."""
    group = TimelineGroup(id="shared")

    with pytest.raises(ValueError) as refused:
        group.add_timeline(ContinuousLogicalTimeline(length=Fraction(1), uid="shared"))
    assert str(refused.value) == (
        "timeline 'shared' cannot join group 'shared': its id 'shared' is already "
        "held by group 'shared'"
    )
    assert group.timeline_ids == []


def test_minted_group_id_is_minted_again() -> None:
    """A minted group id that a newcomer's given id takes is minted again."""
    group = TimelineGroup()

    group.add_timeline(ContinuousLogicalTimeline(length=Fraction(1), uid="group_1"))

    assert group.id == "group_2"
    assert group.timeline_ids == ["group_1"]


@pytest.mark.parametrize(
    ("first", "second"), [("R", "K"), ("K", "R")], ids=["parent_first", "child_first"]
)
def test_parent_and_child_cannot_share_a_group(first: str, second: str) -> None:
    """A member's ancestor or descendant is refused, naming both timelines."""
    root = DiscreteGraphicalTimeline(length=5000, uid="R")
    root.add_child(DiscreteGraphicalTimeline(length=2000, uid="K"), offset=1000)
    timelines = {"R": root, "K": root.get_child("K")}
    group = TimelineGroup(timelines=[timelines[first]])

    with pytest.raises(
        ValueError,
        match=f"Timeline '{second}' cannot join group 'group_1': it is nested "
        f"with member '{first}'",
    ):
        group.add_timeline(timelines[second])
    assert group.timeline_ids == [first]
    with pytest.raises(ValueError, match="it is nested with member"):
        TimelineGroup(timelines=[timelines[first], timelines[second]])


def test_same_timeline_twice_raises() -> None:
    """A member cannot be added again."""
    group = _three_types()

    with pytest.raises(ValueError, match="already in group"):
        group.add_timeline(group.get_timeline("A"))


def test_get_coordinate_at_matches_the_stamp() -> None:
    """The coordinate lane and the stamp lane answer one value."""
    group = _three_types()

    assert group.get_coordinate_at(
        IdCoordinate(Fraction(1, 3), TimeUnit.quarters, "A"), "C"
    ) == IdCoordinate(100, TimeUnit.pixels, "C")


# endregion


# region Segment-line members


def _line(uid: str = "line") -> SegmentLine:
    """An unlocked segment line of two segments, 4 and 2 quarters."""
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid=uid)
    line.append_segment(ContinuousLogicalTimeline(length=Fraction(4), uid=f"{uid}_a"))
    line.append_segment(ContinuousLogicalTimeline(length=Fraction(2), uid=f"{uid}_b"))
    return line


def _join_by_construction(line: SegmentLine, plain: ContinuousPhysicalTimeline) -> Any:
    return TimelineGroup(timelines=[line, plain])


def _join_by_add_timeline(line: SegmentLine, plain: ContinuousPhysicalTimeline) -> Any:
    group = TimelineGroup(timelines=[plain])
    group.add_timeline(line)
    return group


def _join_by_bundle(line: SegmentLine, plain: ContinuousPhysicalTimeline) -> Any:
    bundle = AlignmentBundle()
    bundle.add_timeline(plain, as_group="g")
    bundle.add_timeline(line, grouped_with=plain.id)
    return bundle


JOINS = {
    "constructor": _join_by_construction,
    "add_timeline": _join_by_add_timeline,
    "bundle": _join_by_bundle,
}


def _written_locks(holder: Any, *, minimal: bool) -> dict[str, Any]:
    """Each written root's ``locked`` (None where the key is left out)."""
    return {
        root["id"]: root.get("locked")
        for root in holder.to_dict(minimal=minimal)["timelines"]
    }


@pytest.mark.parametrize("join", list(JOINS), ids=list(JOINS))
def test_a_segment_line_member_is_locked_by_construction(join: str) -> None:
    """Joining locks a segment line, never a plain member or the segments."""
    line = _line()
    plain = ContinuousPhysicalTimeline(length=6.0, uid="plain")
    assert line.is_locked is False

    holder = JOINS[join](line, plain)

    assert (line.is_locked, plain.is_locked) == (True, False)
    assert [line.get_child(uid).is_locked for uid in line.list_segments()] == [
        False,
        False,
    ]
    assert _written_locks(holder, minimal=False) == {"line": True, "plain": False}
    assert _written_locks(holder, minimal=True) == {"line": True, "plain": None}


def _group_manifest(locked: bool) -> dict[str, Any]:
    """A manifest whose group holds a segment line stating *locked*."""
    data = _line().to_dict(minimal=True)
    data["timelines"][0]["locked"] = locked
    data["timelines"].append({"id": "plain", "unit": "seconds", "length": 6.0})
    data["groups"] = [{"id": "g", "members": ["line", "plain"]}]
    return data


@pytest.mark.parametrize(
    "read",
    [TimelineGroup.from_dict, AlignmentBundle.from_dict],
    ids=["group", "bundle"],
)
@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
def test_reading_keeps_the_lock_a_member_line_states(read: Any, locked: bool) -> None:
    """Reading is not construction: the file decides a member line's lock."""
    holder = read(_group_manifest(locked))

    assert holder.get_timeline("line").is_locked is locked
    assert _written_locks(holder, minimal=False) == {"line": locked, "plain": False}


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
def test_a_group_made_from_one_line_keeps_the_lock_it_states(locked: bool) -> None:
    """A group of the one root of a manifest without groups reads its lock too."""
    data = _line().to_dict(minimal=True)
    data["timelines"][0]["locked"] = locked

    group = TimelineGroup.from_dict(data, group_id="g")

    assert (group.timeline_ids, group.get_timeline("line").is_locked) == (
        ["line"],
        locked,
    )
    assert _written_locks(group, minimal=False) == {"line": locked}
    assert _written_locks(group, minimal=True) == {"line": True if locked else None}


# endregion
