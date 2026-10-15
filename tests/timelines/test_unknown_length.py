"""Timelines of unknown length: ``length=None`` through the core.

The validation logic is recorded in ``tests/timelines/README.md`` under
"test_unknown_length.py". An operation that needs the end raises
``UnknownLengthError`` before it changes anything; every other operation
works, and positions are not bounded above.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

import timetoalign
from timetoalign.alignment import AlignmentAnchor, AlignmentBundle, MatchClaim, WarpMap
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.enums import FlowMode
from timetoalign.storage import EventData
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    MeasureMap,
    SegmentLine,
    Timeline,
    TimelineGroup,
    UnknownLengthError,
)
from timetoalign.timelines.flow import ScoreFlowController
from timetoalign.timelines.flowcontrol import Break, Jump, Marker

Q = TimeUnit.quarters


def _q(length: int | Fraction | None = None, uid: str | None = None, **kwargs):
    return ContinuousLogicalTimeline(length=length, uid=uid, **kwargs)


class _MeasureTable(EventData):
    """A synthetic measure table, the input a flow controller reads."""

    def __init__(self, table: pa.Table) -> None:
        super().__init__(table, unit=Q)


def _message(uid: str, operation: str) -> str:
    return f"Cannot {operation}: the length of timeline '{uid}' is unknown"


def _nested_example() -> ContinuousLogicalTimeline:
    """The hand-computed reach example of the README (reach 22)."""
    root = _q(uid="R")
    root.add_child(_q(4, uid="A"), 2)
    inner = _q(uid="U")
    inner.create_region(1, 3)
    innermost = _q(uid="V")
    innermost.flow_control.add_marker(Marker(coordinate=Coordinate(7, Q)))
    inner.add_child(innermost, 5)
    root.add_child(inner, 10)
    root.create_region(0, 15)
    root.flow_control.add_break(Break(coordinate=Coordinate(18, Q)))
    root.flow_control.add_jump(
        Jump(from_coordinate=Coordinate(20, Q), to_coordinate=Coordinate(4, Q))
    )
    return root


# region The error class


def test_error_is_one_value_error_exported_at_both_levels() -> None:
    assert issubclass(UnknownLengthError, ValueError)
    assert timetoalign.UnknownLengthError is UnknownLengthError
    error = UnknownLengthError("perf", "append a child")
    assert str(error) == _message("perf", "append a child")
    assert (error.timeline_id, error.operation) == ("perf", "append a child")


# endregion

# region Construction and setting


@pytest.mark.parametrize(
    "cls",
    [
        Timeline,
        ContinuousLogicalTimeline,
        DiscreteLogicalTimeline,
        ContinuousPhysicalTimeline,
        DiscretePhysicalTimeline,
        ContinuousGraphicalTimeline,
        DiscreteGraphicalTimeline,
        SegmentLine[ContinuousLogicalTimeline],
    ],
)
def test_every_constructor_accepts_an_unknown_length(cls: type[Timeline]) -> None:
    timeline = cls(length=None)
    assert timeline.length is None
    assert timeline.end is None
    assert timeline.is_length_known is False
    assert cls(length=0).is_length_known is True


def test_a_locked_timeline_and_a_child_may_be_of_unknown_length() -> None:
    locked = _q(uid="locked", locked=True)
    assert (locked.is_locked, locked.length) == (True, None)

    parent = _q(8, uid="parent")
    child = parent.create_child(length=None, offset=6, uid="child")
    assert (child.is_locked, child.length) == (True, None)
    assert parent.length == Coordinate(8, Q)


def test_setting_to_and_from_unknown_is_an_ordinary_edit() -> None:
    timeline = _q(8, uid="t")
    timeline.add_events([{"event_type": "Note", "start": 1, "end": 6}])
    timeline.length = None
    assert timeline.length is None
    timeline.length = 10
    assert timeline.length == Coordinate(10, Q)
    timeline.length = None
    with pytest.raises(ValueError, match="content extends to 6"):
        timeline.length = 5
    assert timeline.length is None


def test_a_locked_length_cannot_be_set_either_way() -> None:
    known = _q(8, uid="known", locked=True)
    unknown = _q(uid="unknown", locked=True)
    with pytest.raises(RuntimeError, match="Cannot set length on locked timeline"):
        known.length = None
    with pytest.raises(RuntimeError, match="Cannot set length on locked timeline"):
        unknown.length = 8
    assert (known.length, unknown.length) == (Coordinate(8, Q), None)


def test_repr_and_summary_show_unknown() -> None:
    timeline = _q(uid="t")
    assert repr(timeline) == (
        "ContinuousLogicalTimeline(id='t', length=unknown, unit=quarters, "
        "events=0, children=0)"
    )
    assert timeline.summary()["length"] is None


# endregion

# region Growth and containment


def test_adding_content_leaves_the_length_unknown() -> None:
    timeline = _q(uid="t")
    timeline.add_events([{"event_type": "Note", "start": 40, "end": 50}])
    timeline.add_child(_q(4, uid="c"), 100)
    timeline.create_region(0, 200)
    assert timeline.length is None
    assert timeline.n_events == 1
    assert timeline.get_child_offset("c") == Coordinate(100, Q)


def test_a_locked_timeline_of_unknown_length_has_no_end_to_exceed() -> None:
    timeline = _q(uid="t", locked=True)
    timeline.add_events([{"event_type": "Note", "start": 1000, "end": 1001}])
    assert (timeline.length, timeline.n_events) == (None, 1)


def test_a_child_of_unknown_length_needs_only_its_offset_to_fit() -> None:
    parent = _q(8, uid="parent", locked=True)
    parent.add_child(_q(uid="inside"), 8)
    with pytest.raises(ValueError, match="exceeds timeline length 8"):
        parent.add_child(_q(uid="outside"), 9)
    assert parent.list_children() == ["inside"]
    with pytest.raises(ValueError, match="Offset cannot be negative"):
        _q(uid="any").add_child(_q(uid="neg"), -1)


def test_an_unlocked_parent_grows_to_an_unknown_childs_offset() -> None:
    parent = _q(8, uid="parent")
    parent.add_child(_q(uid="late"), 12)
    assert parent.length == Coordinate(12, Q)


def test_known_child_in_known_parent_is_still_checked() -> None:
    parent = _q(8, uid="parent", locked=True)
    with pytest.raises(ValueError, match="exceeds timeline length 8"):
        parent.add_child(_q(4, uid="c"), 6)


def test_the_segment_row_of_an_unknown_child_has_no_end() -> None:
    parent = _q(8, uid="parent")
    parent.add_child(_q(uid="kid"), 6)
    (row,) = [e for e in parent.events if e["event_type"] == "Segment"]
    assert row["start"] == {"value": 6.0, "numerator": 6, "denominator": 1}
    assert (row["end"], row["duration"]) == (None, None)


# endregion

# region End-independent operations


def test_positions_are_not_bounded_above() -> None:
    parent = _q(8, uid="parent")
    parent.add_child(_q(uid="kid"), 6)
    stamp = parent.get_timestamp_at(100)
    assert stamp.coordinates == {
        "parent": Coordinate(100, Q),
        "kid": Coordinate(94, Q),
    }
    assert [child.id for child in parent.get_children_at(100)] == ["kid"]
    assert _q(uid="root").get_timestamp_at(1000).coordinates == {
        "root": Coordinate(1000, Q)
    }


def test_events_of_an_unknown_child_are_found_past_its_parent() -> None:
    parent = _q(8, uid="parent")
    kid = _q(uid="kid")
    kid.add_events([{"id": "late", "event_type": "Beat", "instant": 30}])
    parent.add_child(kid, 6)
    found = parent.get_events_at(36)
    assert [event["id"] for event in found["kid"]] == ["late"]


def test_slicing_to_an_explicit_end_works_past_all_content() -> None:
    timeline = _q(uid="t")
    timeline.add_child(_q(uid="kid"), 4)
    sliced = timeline.get_slice(2, 100)
    assert sliced.length == Coordinate(98, Q)
    assert sliced.get_child_offset(sliced.list_children()[0]) == Coordinate(2, Q)
    assert sliced.get_child(sliced.list_children()[0]).length == Coordinate(96, Q)
    with pytest.raises(ValueError, match=r"outside timeline bounds \[0, unknown\)"):
        timeline.get_slice(-1, 4)


def test_the_tables_bound_an_unknown_member_below_only() -> None:
    root = _nested_example()
    frame = root.get_timestamp_table(at=[2, 50], format="dataframe")
    assert frame.to_dict("list") == {
        "R (quarters)": [Fraction(2), Fraction(50)],
        "A (quarters)": [Fraction(0), None],
        "U (quarters)": [None, Fraction(40)],
        "V (quarters)": [None, Fraction(35)],
    }
    boundaries = root.get_boundary_table()

    def values(name: str) -> list[float | None]:
        return boundaries.column(name).combine_chunks().field("value").to_pylist()

    assert values("R") == [0, 2, 6, 10, 15]
    assert values("A") == [
        None,
        0,
        4,
        None,
        None,
    ]
    assert values("U") == [
        None,
        None,
        None,
        0,
        5,
    ]


def test_derive_keeps_the_length_unknown() -> None:
    from timetoalign.maps import LinearMap

    audio = ContinuousPhysicalTimeline(length=None, uid="audio")
    audio.add_conversion_map(
        LinearMap(scalar=2.0, source_unit=TimeUnit.seconds, target_unit=Q)
    )
    assert audio.derive(Q).length is None


def test_explicit_flow_sections_unfold_without_the_end() -> None:
    timeline = _q(uid="t")
    timeline.create_flow_map([(0, 4), (0, 4)], id="twice")
    unfolded = timeline.apply_flow("twice")
    assert unfolded.length == Coordinate(8, Q)
    assert timeline.get_flow_map("twice").source_length is None


def test_to_typed_keeps_the_length_unknown() -> None:
    generic = Timeline(length=None, unit="seconds", uid="g")
    typed = generic.to_typed()
    assert type(typed) is ContinuousPhysicalTimeline
    assert typed.length is None


def test_a_bundle_transfers_into_a_timeline_of_unknown_length() -> None:
    bundle = AlignmentBundle()
    bundle.add_timeline(_q(8, uid="x"))
    bundle.add_timeline(_q(uid="z"))
    bundle.add_match_claims([_claim("z", 40, "x", 2)])
    assert bundle.get_matchstamp_at(Coordinate(2, Q), "x").coordinates == {
        "x": Coordinate(2, Q),
        "z": Coordinate(40, Q),
    }


def _claim(a: str, at_a: int, b: str, at_b: int) -> MatchClaim:
    return MatchClaim(
        timeline_a_id=a,
        timeline_b_id=b,
        start_anchor=AlignmentAnchor(
            timeline_a_id=a,
            coordinate_a=Coordinate(at_a, Q),
            timeline_b_id=b,
            coordinate_b=Coordinate(at_b, Q),
        ),
    )


# endregion

# region Operations that need the end


@pytest.fixture
def group() -> TimelineGroup:
    """A known quarters member and a seconds member of unknown length."""
    return TimelineGroup(
        timelines=[
            _q(Fraction(16), uid="a"),
            ContinuousPhysicalTimeline(length=None, uid="b"),
        ]
    )


def test_a_group_member_may_be_of_unknown_length(group: TimelineGroup) -> None:
    assert group.timeline_ids == ["a", "b"]
    assert group.get_timeline("b").length is None


def test_group_conversion_between_members_raises(group: TimelineGroup) -> None:
    with pytest.raises(UnknownLengthError) as raised:
        group.get_timestamp_at(8, "a")
    assert str(raised.value) == _message("b", "convert a position from 'a' to 'b'")
    assert group.get_timeline("b").length is None
    assert group.get_timeline("a").length == Coordinate(16, Q)


def test_a_group_position_needs_no_end_until_it_is_converted(
    group: TimelineGroup,
) -> None:
    alone = TimelineGroup(timelines=[ContinuousPhysicalTimeline(length=None, uid="b")])
    stamp = alone.get_timestamp_at(1000.0, "b")
    assert stamp.coordinates == {"b": Coordinate(1000.0, TimeUnit.seconds)}
    assert stamp.is_interpolated is True
    with pytest.raises(ValueError) as refused:
        alone.get_timestamp_at(-1.0, "b")
    assert str(refused.value) == "Coordinate -1 outside range for 'b'"
    with pytest.raises(UnknownLengthError) as raised:
        group.get_timestamp_at(8.0, "b")
    assert str(raised.value) == _message("b", "convert a position from 'b' to 'a'")


def test_unfolding_a_group_with_an_unknown_member_raises_before_building(
    group: TimelineGroup,
) -> None:
    controller = ScoreFlowController(
        _MeasureTable(
            pa.table(
                {
                    "mc": [1, 2, 3, 4],
                    "mn": ["1", "2", "3", "4"],
                    "duration": [{"value": 4.0}] * 4,
                    "start": [{"value": float(4 * i)} for i in range(4)],
                    "timesig": ["4/4"] * 4,
                }
            )
        )
    )
    flow = controller.compute_flow(FlowMode.default)
    with pytest.raises(UnknownLengthError) as raised:
        group.apply_flow(flow, controller, "a")
    assert str(raised.value) == _message("b", "convert a position from 'a' to 'b'")
    for member in group:
        assert (member.list_children(), member.list_flow_maps()) == ([], [])
    assert group.get_timeline("b").length is None


def test_the_group_tables_and_intervals_need_the_end(group: TimelineGroup) -> None:
    with pytest.raises(UnknownLengthError) as raised:
        group.get_timestamp_table()
    assert str(raised.value) == _message("b", "tabulate the group's common end")
    with pytest.raises(UnknownLengthError) as raised:
        group.get_interval_for("b")
    assert str(raised.value) == _message("b", "give the interval of a group member")
    assert str(group.get_interval_for("a")) == "[0, 16) quarters"


def test_a_matchstamp_through_such_a_group_raises() -> None:
    bundle = AlignmentBundle()
    bundle.add_timeline(_q(8, uid="x"), as_group="g")
    bundle.add_timeline(
        ContinuousPhysicalTimeline(length=None, uid="y"), grouped_with="x"
    )
    bundle.add_timeline(_q(8, uid="z"))
    bundle.add_match_claims([_claim("z", 4, "x", 2)])
    with pytest.raises(UnknownLengthError) as raised:
        bundle.get_matchstamp_at(Coordinate(4, Q), "z")
    assert str(raised.value) == _message("y", "convert a position from 'x' to 'y'")


def test_appending_to_a_parent_of_unknown_length_raises() -> None:
    parent = _q(uid="p")
    with pytest.raises(UnknownLengthError) as raised:
        parent.append_child(_q(4, uid="c"))
    assert str(raised.value) == _message("p", "append a child")
    assert parent.list_children() == []


def test_a_line_of_unknown_length_appends_where_the_next_offset_is_known() -> None:
    line = SegmentLine[ContinuousLogicalTimeline](length=None, uid="line")
    line.append_segment(_q(4, uid="a"))
    line.append_segment(_q(2, uid="b"))
    line.append_segment(_q(uid="c"))
    assert [line.get_child_offset(uid) for uid in "abc"] == [
        Coordinate(0, Q),
        Coordinate(4, Q),
        Coordinate(6, Q),
    ]
    assert line.length is None
    with pytest.raises(UnknownLengthError) as raised:
        line.append_segment(_q(2, uid="d"))
    assert str(raised.value) == _message("c", "append a segment to 'line'")
    assert line.list_segments() == ["a", "b", "c"]


def test_appending_a_segment_after_one_of_unknown_length_raises() -> None:
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
    line.append_segment(_q(4, uid="a"))
    line.append_segment(_q(uid="b"))
    assert line.length is None
    line.length = 9
    with pytest.raises(UnknownLengthError) as raised:
        line.append_segment(_q(2, uid="c"))
    assert str(raised.value) == _message("b", "append a segment to 'line'")
    assert (line.list_segments(), line.length) == (["a", "b"], Coordinate(9, Q))


@pytest.mark.parametrize(
    "method", ["create_regions_by_splitting", "create_segment_line_by_splitting"]
)
def test_splitting_up_to_the_end_raises(method: str) -> None:
    timeline = _q(uid="t")
    timeline.add_events([{"event_type": "Beat", "instant": 4, "split": True}])
    with pytest.raises(UnknownLengthError) as raised:
        getattr(timeline, method)("split", include_after_last=True)
    assert str(raised.value) == _message("t", "split up to the end")
    assert timeline.n_regions == 0
    assert timeline.length is None


def test_splitting_short_of_the_end_works() -> None:
    timeline = _q(uid="t")
    timeline.add_events([{"event_type": "Beat", "instant": 4, "split": True}])
    regions = timeline.create_regions_by_splitting("split", include_after_last=False)
    assert [(r.start.value, r.end.value) for r in regions] == [(0, 4)]


def test_a_timeline_of_unknown_length_is_no_flow_interval() -> None:
    parent = _q(8, uid="parent")
    kid = parent.create_child(length=None, offset=0, uid="kid")
    with pytest.raises(UnknownLengthError) as raised:
        parent.create_flow_map([kid], id="by_child")
    assert str(raised.value) == _message("kid", "read the timeline as an interval")
    assert parent.list_flow_maps() == []


def test_materialising_needs_every_length_in_the_tree() -> None:
    warp = WarpMap.from_coordinate_pairs(
        "src",
        "tgt",
        [IdCoordinate(value, Q, "src") for value in (0, 10)],
        [IdCoordinate(value, TimeUnit.seconds, "tgt") for value in (0.0, 5.0)],
    )
    with pytest.raises(UnknownLengthError) as raised:
        warp.materialise(_q(uid="src"))
    assert str(raised.value) == _message("src", "materialise a warped copy")

    source = _q(10, uid="src")
    source.add_child(_q(uid="deep"), 2)
    with pytest.raises(UnknownLengthError) as raised:
        warp.materialise(source)
    assert str(raised.value) == _message("deep", "materialise a warped copy")


def test_the_warp_support_ends_at_the_last_anchor() -> None:
    warp = WarpMap.from_coordinate_pairs(
        "src",
        "tgt",
        [IdCoordinate(value, Q, "src") for value in (0, 10)],
        [IdCoordinate(value, TimeUnit.seconds, "tgt") for value in (0.0, 5.0)],
    )
    assert warp.get_coordinate_at(10) == IdCoordinate(5.0, TimeUnit.seconds, "tgt")
    with pytest.raises(ValueError, match="outside WarpMap source support"):
        warp.get_coordinate_at(11)


# endregion

# region Segments


def _line_with_unknown_middle() -> SegmentLine:
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
    line.add_child(_q(4, uid="a"), 0)
    line.add_child(_q(uid="b"), 4)
    line.add_child(_q(2, uid="c"), 10)
    return line


def test_a_segment_of_unknown_length_ends_at_its_successors_offset() -> None:
    line = _line_with_unknown_middle()
    assert line.length is None
    assert line.is_segment_line() is True
    assert line.get_timestamp_at(7).coordinates == {
        "line": Coordinate(7, Q),
        "b": Coordinate(3, Q),
    }
    assert line.get_timestamp_at(11).coordinates == {
        "line": Coordinate(11, Q),
        "c": Coordinate(1, Q),
    }
    assert [child.id for child in line.get_children_at(11)] == ["c"]
    line.length = 12
    assert line.is_segment_line() is True


def _passes(uid: str, end: int, bound: int) -> str:
    return (
        f"Adding content ending at {end} to timeline '{uid}' passes {bound}, where "
        f"its parent 'line' ends it while its length is unknown"
    )


def test_content_in_an_unknown_segment_stops_at_its_successor() -> None:
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
    line.add_child(_q(uid="x"), 0)
    line.add_child(_q(4, uid="y"), 5)
    x = line.get_child("x")

    x.add_events([{"event_type": "Note", "start": 1, "end": 5}])
    with pytest.raises(ValueError) as refused:
        x.add_events([{"event_type": "Note", "start": 8, "end": 9}])

    assert str(refused.value) == _passes("x", 9, 5)
    assert (x.length, x.n_events) == (None, 1)


def test_content_in_an_unknown_last_segment_stops_at_the_lines_length() -> None:
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
    line.append_segment(_q(4, uid="a"))
    line.append_segment(_q(uid="b"))
    line.length = 9
    b = line.get_child("b")

    b.add_events([{"event_type": "Note", "start": 0, "end": 5}])
    with pytest.raises(ValueError) as refused:
        b.add_events([{"event_type": "Note", "start": 0, "end": 6}])

    assert str(refused.value) == _passes("b", 6, 5)
    assert (b.length, b.n_events, line.length) == (None, 1, Coordinate(9, Q))


def test_a_successor_may_not_start_before_an_unknown_segment() -> None:
    line = SegmentLine[ContinuousLogicalTimeline](length=0, uid="line")
    line.add_child(_q(4, uid="a"), 0)
    line.add_child(_q(uid="b"), 4)
    with pytest.raises(ValueError, match="Segment 'b' of unknown length starts at 4"):
        line.add_child(_q(2, uid="c"), 3)
    assert line.list_segments() == ["a", "b"]


def test_a_plain_hierarchy_with_an_unknown_segment_casts_to_a_line() -> None:
    plain = _q(12, uid="plain")
    plain.add_child(_q(4, uid="a"), 0)
    plain.add_child(_q(uid="b"), 4)
    plain.add_child(_q(2, uid="c"), 10)
    assert plain.is_segment_line() is True
    line = plain.as_segment_line()
    assert line.list_segments() == ["a", "b", "c"]
    assert line.length == Coordinate(12, Q)


def test_contiguity_between_known_values_is_still_checked() -> None:
    plain = _q(12, uid="plain")
    plain.add_child(_q(4, uid="a"), 0)
    plain.add_child(_q(uid="b"), 5)
    assert plain.is_segment_line() is False


# endregion

# region Reach and display


def test_the_reach_of_the_nested_example() -> None:
    root = _nested_example()
    inner = root.get_child("U")
    assert inner.get_child("V").reach == Coordinate(7, Q)
    assert inner.reach == Coordinate(12, Q)
    assert root.reach == Coordinate(22, Q)


def test_each_content_kind_reaches_on_its_own() -> None:
    assert _q(uid="empty").reach == Coordinate(0, Q)

    measures = _q(uid="measures")
    measures.add_measure_map(
        MeasureMap.from_rows([{"count": 1, "time_signature": "3/4"}, {"count": 5}])
    )
    assert measures.reach == Coordinate(15, Q)

    jump = _q(uid="jump")
    jump.flow_control.add_jump(
        Jump(from_coordinate=Coordinate(9, Q), to_coordinate=Coordinate(30, Q))
    )
    assert jump.reach == Coordinate(30, Q)


def test_events_are_not_part_of_the_reach() -> None:
    timeline = _q(uid="t")
    timeline.add_events([{"event_type": "Note", "start": 40, "end": 50}])
    assert timeline.reach == Coordinate(0, Q)


def test_the_diagram_scales_to_the_reach_and_labels_the_end_unknown() -> None:
    assert str(_nested_example().diagram()) == "\n".join(
        [
            "ContinuousLogicalTimeline[R] (2 children, 1 regions)",
            "                      0 ___________________________________ ? quarters",
            "  ├─ A                2    ______                           6",
            "  └─ U               10                ____________________ ?",
            "     └─ V            15                        ____________ ?",
        ]
    )


# endregion

# region The retired zero sentinel


def test_a_zero_length_child_is_not_unbounded() -> None:
    parent = _q(10, uid="p")
    parent.add_child(_q(0, uid="z"), 3)
    assert parent.get_timestamp_at(5).coordinates == {"p": Coordinate(5, Q)}
    assert [child.id for child in parent.get_children_at(5)] == []


def test_a_zero_extent_draws_at_the_origin() -> None:
    root = _q(0, uid="r")
    root.add_child(_q(0, uid="k"), 0)
    assert str(root.diagram()).splitlines()[-1] == (
        "  └─ k                0 _                                   0"
    )


# endregion
