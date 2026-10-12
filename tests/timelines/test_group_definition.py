"""TimelineGroup reduced to its definition: one start, one end, linear between.

The strategy is documented in ``tests/timelines/README.md`` ("Timeline Groups").
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.time import express_as
from timetoalign.timelines import TimelineGroup
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
    assert data["group"]["members"] == ["A2", "K"]
    assert data["graphical"] == {"order": ["A2", "K"]}


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
        "timelines": [image.to_dict(), score.to_dict()],
        "group": {
            "id": "group_1",
            "name": "edition",
            "meta": {"source": "test"},
            "members": ["image", "score"],
        },
        "graphical": {"order": ["score", "image"]},
    }


def test_flags_pass_through_to_every_root() -> None:
    """``events`` and ``external_references`` reach every root's dictionary."""
    score = ContinuousLogicalTimeline(length=Fraction(16), uid="score")
    score.add_events([{"start": Fraction(1, 3), "event_type": "Note"}])
    group = TimelineGroup(timelines=[score])

    data = group.to_dict(events=True, external_references=True)

    assert data["timelines"] == [score.to_dict(events=True, external_references=True)]


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


def test_file_meta_is_accepted_and_ignored() -> None:
    """A dictionary carrying the file's ``meta`` reads as the same group."""
    data = _three_types().to_dict()

    rebuilt = TimelineGroup.from_dict({**data, "meta": {"schema_version": "0.7.0"}})

    assert rebuilt.to_dict() == data


def _malformed(change: Any) -> dict[str, Any]:
    """A valid group dictionary with one change applied."""
    data = _three_types().to_dict()
    change(data)
    return data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("group"), r"lacks the key\(s\) \['group'\]"),
        (lambda d: d.update(extra=1), r"unknown key\(s\) \['extra'\]"),
        (lambda d: d["group"].pop("meta"), r"lacks the key\(s\) \['meta'\]"),
        (lambda d: d["group"]["members"].append("Z"), "names member 'Z'"),
        (lambda d: d["group"]["members"].append("A"), "names a member twice"),
        (
            lambda d: d["timelines"].append(d["timelines"][0]),
            "timeline id 'A' occurs twice",
        ),
        (
            lambda d: d["graphical"].update(order=["A", "B"]),
            "not a permutation of its members",
        ),
        (
            lambda d: (
                d["group"]["members"].remove("C"),
                d["graphical"]["order"].remove("C"),
            ),
            "root timeline 'C' contains no member",
        ),
        (lambda d: d["group"].update(id="A"), "group id 'A' is also a timeline id"),
        (lambda d: d["group"].update(id="a/b"), "Invalid id 'a/b'"),
    ],
    ids=[
        "missing_group",
        "unknown_key",
        "missing_meta",
        "unknown_member",
        "duplicate_member",
        "duplicate_timeline",
        "short_order",
        "memberless_root",
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
    data["group"]["members"].append("R")
    data["graphical"]["order"].append("R")

    with pytest.raises(
        ValueError,
        match="Timeline 'R' cannot join group 'group_1': it is nested with member 'K'",
    ):
        TimelineGroup.from_dict(data)


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

    with pytest.raises(ValueError, match="Timeline 'B' cannot join group"):
        group.add_timeline(ContinuousPhysicalTimeline(length=1.0, uid="B"))
    assert group.to_dict() == before


def test_descendant_id_collision_raises() -> None:
    """A newcomer's descendant may not share an id with the group's trees."""
    group, _ = _child_member()
    newcomer = ContinuousLogicalTimeline(length=Fraction(4), uid="other")
    newcomer.add_child(ContinuousLogicalTimeline(length=Fraction(1), uid="R"), offset=0)

    with pytest.raises(ValueError, match="Timeline 'R' cannot join group"):
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

    with pytest.raises(ValueError, match="Timeline 'B' cannot join group"):
        group.add_timeline(newcomer)
    assert (newcomer.id, child.id) == ("B", "cpt_1")
    assert group.timeline_ids == ["cpt_1", "B"]


def test_group_id_may_not_be_a_timeline_id() -> None:
    """A timeline whose id is the group's own given id cannot join."""
    group = TimelineGroup(id="shared")

    with pytest.raises(ValueError, match="already held by a group there"):
        group.add_timeline(ContinuousLogicalTimeline(length=Fraction(1), uid="shared"))
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
