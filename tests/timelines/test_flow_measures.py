"""Measures and flow control: the translation both ways, and flows in measure counts.

Every measure map here is synthetic and of 4/4 bars unless a test says
otherwise, so that measure count ``n`` starts at ``4 * (n - 1)`` quarters.
Every expected value is a literal; see ``README.md``, "Measures and flow
control".
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from timetoalign.core import Coordinate, Duration, Interval
from timetoalign.core.enums import MeasureFindingKind
from timetoalign.core.provenance import Provenance
from timetoalign.testdata import ensure_data
from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap
from timetoalign.timelines.flow import (
    MeasureFinding,
    MeasureRange,
    flow_control_from_measures,
    measure_keys_from_flow_control,
    measure_order_from_successors,
    measure_ranges,
    read_flow_csv,
    repeat_end_findings,
)
from timetoalign.timelines.flow.engine import atomic_sections, play
from timetoalign.timelines.flow.flowmap import PlayedFlow
from timetoalign.timelines.flowcontrol import (
    FlowControlRegistry,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)

TARGET_FLOWS = ensure_data("target_flows")


def bars(*rows: dict[str, Any], signature: str = "4/4") -> MeasureMap:
    """A measure map of one row per argument, counted from 1."""
    return MeasureMap.from_rows(
        [
            {
                "count": count,
                **({"time_signature": signature} if count == 1 else {}),
                **row,
            }
            for count, row in enumerate(rows, start=1)
        ]
    )


def written(measures: Any) -> list[dict[str, Any]]:
    return [element.to_dict() for element in flow_control_from_measures(measures)]


def target(id: str, at: int, manifestation: str) -> dict[str, Any]:
    return {
        "id": id,
        "role": "jump_target",
        "coordinate": [at, 1],
        "manifestation": manifestation,
    }


def jump(id: str, at: int, *targets: dict[str, Any]) -> dict[str, Any]:
    return {"id": id, "role": "jump", "coordinate": [at, 1], "targets": list(targets)}


def volta_break(id: str, at: int) -> dict[str, Any]:
    return {"id": id, "role": "break", "coordinate": [at, 1], "manifestation": "volta"}


def repeat_end(to: str | None = None, **keys: Any) -> dict[str, Any]:
    place = {} if to is None else {"to": to}
    return {**place, "repeat_count": 1, **keys, "manifestation": "repeat_end"}


def bracket(passes: list[int], to: str | None = None) -> dict[str, Any]:
    place: dict[str, Any] = {"by": [0, 1]} if to is None else {"to": to}
    return {**place, "passes": passes, "manifestation": "volta"}


START, END = {"start_repeat": True}, {"end_repeat": True}

# region Rows into elements


def test_measures_without_signs_state_no_flow_control() -> None:
    assert written(bars({}, {}, {})) == []


def test_a_repeat_start_is_a_jump_target_at_the_rows_start() -> None:
    assert written(bars({}, START, {})) == [target("jump_target_1", 4, "repeat_start")]


def test_a_repeat_end_goes_back_to_its_repeat_start() -> None:
    assert written(bars({}, START, END, {})) == [
        target("jump_target_1", 4, "repeat_start"),
        jump("jump_1", 12, repeat_end("jump_target_1")),
    ]


def test_a_repeat_end_without_a_printed_start_has_no_place() -> None:
    assert written(bars({}, END, {})) == [jump("jump_1", 8, repeat_end())]


def test_nested_repeats_close_the_nearest_open_start() -> None:
    assert written(bars(START, START, END, END)) == [
        target("jump_target_1", 0, "repeat_start"),
        target("jump_target_2", 4, "repeat_start"),
        jump("jump_1", 12, repeat_end("jump_target_2")),
        jump("jump_2", 16, repeat_end("jump_target_1")),
    ]


def test_a_start_once_closed_serves_no_later_repeat_end() -> None:
    assert written(bars(START, END, {}, END)) == [
        target("jump_target_1", 0, "repeat_start"),
        jump("jump_1", 8, repeat_end("jump_target_1")),
        jump("jump_2", 16, repeat_end()),
    ]


def test_a_start_and_an_end_on_one_measure_repeat_that_measure() -> None:
    assert written(bars({}, {**START, **END}, {})) == [
        target("jump_target_1", 4, "repeat_start"),
        jump("jump_1", 8, repeat_end("jump_target_1")),
    ]


def test_two_brackets_are_one_jump_one_landing_and_one_break() -> None:
    assert written(bars(START, {}, {"volta": 1, **END}, {"volta": 2}, {})) == [
        target("jump_target_1", 0, "repeat_start"),
        jump("jump_1", 8, bracket([1]), bracket([2], "jump_target_2")),
        volta_break("break_1", 12),
        jump("jump_2", 12, repeat_end("jump_target_1")),
        target("jump_target_2", 12, "volta"),
    ]


def test_a_later_brackets_repeat_end_goes_to_the_same_start_without_reset() -> None:
    assert written(
        bars(START, {}, {"volta": 1, **END}, {"volta": 2, **END}, {"volta": 3}, {})
    ) == [
        target("jump_target_1", 0, "repeat_start"),
        jump(
            "jump_1",
            8,
            bracket([1]),
            bracket([2], "jump_target_2"),
            bracket([3], "jump_target_3"),
        ),
        volta_break("break_1", 12),
        jump("jump_2", 12, repeat_end("jump_target_1")),
        target("jump_target_2", 12, "volta"),
        volta_break("break_2", 16),
        jump("jump_3", 16, repeat_end("jump_target_1", reset=False)),
        target("jump_target_3", 16, "volta"),
    ]


def test_a_bracket_of_two_measures_is_one_bracket() -> None:
    assert written(bars({}, {"volta": 1}, {"volta": 1, **END}, {"volta": 2})) == [
        jump("jump_1", 4, bracket([1]), bracket([2], "jump_target_1")),
        volta_break("break_1", 12),
        jump("jump_2", 12, repeat_end()),
        target("jump_target_1", 12, "volta"),
    ]


BOTH_INPUTS = pytest.mark.parametrize(
    "read", [MeasureMap.from_rows, MeasureMap.expand_rows], ids=["map", "rows"]
)
"""The two inputs the translation reads: a measure map, and full rows."""


@BOTH_INPUTS
def test_a_volta_stated_as_a_list_gives_those_passes(read: Any) -> None:
    rows = read(
        [
            {"count": 1, "time_signature": "4/4"},
            {"count": 2, "volta": [1, 3], "end_repeat": True},
            {"count": 3, "volta": [2], "end_repeat": True},
            {"count": 4},
        ]
    )
    assert written(rows) == [
        jump("jump_1", 4, bracket([1, 3]), bracket([2], "jump_target_1")),
        volta_break("break_1", 8),
        jump("jump_2", 8, repeat_end()),
        target("jump_target_1", 8, "volta"),
        jump("jump_3", 12, repeat_end(reset=False)),
    ]


def test_a_bracket_naming_a_pass_again_begins_the_next_run() -> None:
    assert written(
        bars({}, {"volta": 1, **END}, {"volta": 2}, {"volta": 1, **END}, {"volta": 2})
    ) == [
        jump("jump_1", 4, bracket([1]), bracket([2], "jump_target_1")),
        volta_break("break_1", 8),
        jump("jump_2", 8, repeat_end()),
        target("jump_target_1", 8, "volta"),
        jump("jump_3", 12, bracket([1]), bracket([2], "jump_target_2")),
        volta_break("break_2", 16),
        jump("jump_4", 16, repeat_end()),
        target("jump_target_2", 16, "volta"),
    ]


def test_signs_on_one_coordinate_are_destinations_of_one_jump() -> None:
    assert written(bars({}, END, {"volta": 1, **END}, {"volta": 2})) == [
        jump("jump_1", 8, repeat_end(), bracket([1]), bracket([2], "jump_target_1")),
        volta_break("break_1", 12),
        jump("jump_2", 12, repeat_end()),
        target("jump_target_1", 12, "volta"),
    ]


def test_coordinates_are_the_measures_exact_positions() -> None:
    elements = flow_control_from_measures(bars({}, START, END, signature="3/8"))
    assert [element.to_dict()["coordinate"] for element in elements] == [[3, 2], [9, 2]]
    assert [element.coordinate for element in elements] == [
        Coordinate(Fraction(3, 2), "quarters"),
        Coordinate(Fraction(9, 2), "quarters"),
    ]


def test_the_elements_join_a_registry_and_play() -> None:
    measures = bars(START, {}, {"volta": 1, **END}, {"volta": 2}, {})
    registry = FlowControlRegistry(unit="quarters")
    registry.add_all(flow_control_from_measures(measures))
    played = play(registry, 20)
    assert [(run.start.value, run.end.value) for run in played.runs] == [
        (0, 12),
        (0, 8),
        (12, 20),
    ]
    assert played.findings == ()


def test_an_abstract_measure_map_is_refused() -> None:
    abstract = MeasureMap.from_rows([{"count": 1}, {"count": 2, "end_repeat": True}])
    with pytest.raises(ValueError) as error:
        flow_control_from_measures(abstract)
    assert str(error.value) == (
        "measure count 1 states no actual_length: flow control lies at positions, "
        "which an abstract measure map has none of"
    )


def test_a_volta_that_is_no_pass_number_is_refused() -> None:
    rows = MeasureMap.expand_rows(
        [{"count": 1, "time_signature": "4/4"}, {"count": 2, "volta": "1."}]
    )
    with pytest.raises(ValueError) as error:
        flow_control_from_measures(rows)
    assert str(error.value) == (
        "the volta '1.' of measure count 2 is neither a pass number nor a list of "
        "pass numbers"
    )


# endregion

# region Elements into row keys


def keys(*rows: tuple[bool, bool, Any]) -> list[dict[str, Any]]:
    return [
        {"count": count, "start_repeat": start, "end_repeat": end, "volta": volta}
        for count, (start, end, volta) in enumerate(rows, start=1)
    ]


def test_the_row_keys_come_back_from_the_elements() -> None:
    measures = bars(
        START, {}, {"volta": 1, **END}, {"volta": 2, **END}, {"volta": 3}, START, END
    )
    assert measure_keys_from_flow_control(
        flow_control_from_measures(measures), measures
    ) == keys(
        (True, False, None),
        (False, False, None),
        (False, True, 1),
        (False, True, 2),
        (False, False, 3),
        (True, False, None),
        (False, True, None),
    )


def test_where_the_last_bracket_ends_stays_with_the_rows() -> None:
    measures = bars({}, {"volta": 1, **END}, {"volta": 2}, {"volta": 2}, {})
    elements = flow_control_from_measures(measures)
    # The rows state that the second bracket covers two measures.
    assert measure_keys_from_flow_control(elements, measures) == keys(
        (False, False, None),
        (False, True, 1),
        (False, False, 2),
        (False, False, 2),
        (False, False, None),
    )
    # The same positions without that statement: flow control knows only
    # where the last bracket begins.
    bare = bars({}, {}, {}, {}, {})
    assert measure_keys_from_flow_control(elements, bare) == keys(
        (False, False, None),
        (False, True, 1),
        (False, False, 2),
        (False, False, None),
        (False, False, None),
    )


@BOTH_INPUTS
def test_several_passes_come_back_as_a_list(read: Any) -> None:
    rows = read(
        [
            {"count": 1, "time_signature": "4/4"},
            {"count": 2, "volta": [1, 3], "end_repeat": True},
            {"count": 3, "volta": [2]},
            {"count": 4, "volta": [2]},
        ]
    )
    # One pass comes back as a number, though these rows state it as a list:
    # flow control states passes, not how a source wrote them. The last
    # bracket covers the two measures whose rows state its passes.
    assert measure_keys_from_flow_control(
        flow_control_from_measures(rows), rows
    ) == keys(
        (False, False, None),
        (False, True, [1, 3]),
        (False, False, 2),
        (False, False, 2),
    )


def test_row_keys_are_read_from_what_an_element_looks_like() -> None:
    measures = bars({}, {}, {})

    def at(value: int) -> Coordinate:
        return Coordinate(value, "quarters")

    elements = [
        # A place without a manifestation is no repeat start of the notation.
        JumpTarget(at(4), id="place"),
        # A repeat start inside a measure is no key of a row.
        JumpTarget(at(6), id="inside", manifestation="repeat_start"),
        JumpTarget(at(8), id="start", manifestation="repeat_start"),
        Jump(at(8), [JumpDestination(to="place", repeat_count=1)], id="leap"),
        Jump(
            at(12),
            [JumpDestination(to="start", repeat_count=1, manifestation="repeat_end")],
            id="repeat",
        ),
    ]
    assert measure_keys_from_flow_control(elements, measures) == keys(
        (False, False, None), (False, False, None), (True, True, None)
    )


# endregion

# region Measure counts


def run(start: Any, end: Any) -> Interval:
    return Interval(
        start=Coordinate(start, "quarters"), end=Coordinate(end, "quarters")
    )


def test_a_flow_is_stated_in_measure_counts() -> None:
    measures = bars(START, {}, {"volta": 1, **END}, {"volta": 2}, {})
    registry = FlowControlRegistry(unit="quarters")
    registry.add_all(flow_control_from_measures(measures))
    sections = atomic_sections(registry, 20)
    assert measure_ranges(sections, measures, sections=sections) == (
        MeasureRange(1, 3, ("A",)),
        MeasureRange(3, 4, ("B",)),
        MeasureRange(4, 6, ("C",)),
    )
    assert measure_ranges(play(registry, 20).runs, measures, sections=sections) == (
        MeasureRange(1, 4, ("A", "B")),
        MeasureRange(1, 3, ("A",)),
        MeasureRange(4, 6, ("C",)),
    )
    assert MeasureRange(1, 4, ("A", "B")).n_measures == 3


def test_a_run_is_divided_where_asked_and_nowhere_else() -> None:
    measures = bars({}, {}, {}, {})
    assert measure_ranges([run(0, 16)], measures) == (MeasureRange(1, 5),)
    section_break = SectionBreak(Coordinate(8, "quarters"))
    assert measure_ranges(
        [run(0, 16), run(8, 16), run(0, 8)],
        measures,
        divide_at=[section_break.coordinate],
    ) == (
        MeasureRange(1, 3),
        MeasureRange(3, 5),
        MeasureRange(3, 5),
        MeasureRange(1, 3),
    )


def test_a_stretch_inside_a_measure_has_no_measure_counts() -> None:
    measures = bars({}, {}, signature="3/4")
    with pytest.raises(ValueError) as error:
        measure_ranges([run(0, 4)], measures)
    assert str(error.value) == (
        "4 lies inside measure count 2: a stretch beginning or ending there has no "
        "measure counts"
    )
    with pytest.raises(ValueError) as error:
        measure_ranges([run(3, 9)], measures)
    assert str(error.value) == (
        "9 lies outside the measures: a stretch beginning or ending there has no "
        "measure counts"
    )


# endregion

# region The flow file


def test_a_flow_file_is_read_by_mode(tmp_path: Path) -> None:
    path = tmp_path / "piece.flow.csv"
    path.write_text(
        "flow_mode,source_file,software_version,mc_start,mc_end,atomic_segments,comment\n"
        "atomic,piece.tsv,a 1,1,3,A,\n"
        "atomic,piece.tsv,a 1,3,5,B,\n"
        "default,piece.tsv,a 1,1,5,A;B,\n"
        "default,piece.tsv,a 1,3,5,B,again\n"
        "other_tool,piece.xml,b 2,1,5,A;B,\n"
        "failing_tool,piece.xml,c 3,ERROR,no repeats,,\n"
    )
    assert read_flow_csv(path) == {
        "atomic": (MeasureRange(1, 3, ("A",)), MeasureRange(3, 5, ("B",))),
        "default": (MeasureRange(1, 5, ("A", "B")), MeasureRange(3, 5, ("B",))),
        "other_tool": (MeasureRange(1, 5, ("A", "B")),),
    }


def test_a_file_without_the_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    path.write_text("flow_mode,start,end\ndefault,1,5\n")
    with pytest.raises(ValueError) as error:
        read_flow_csv(path)
    assert str(error.value) == (
        f"{path} is no .flow.csv file: it lacks the columns ['mc_start', 'mc_end', "
        "'atomic_segments']. Available columns: ['flow_mode', 'start', 'end']"
    )


def test_the_rows_of_other_tools_stay_readable() -> None:
    flows = read_flow_csv(
        TARGET_FLOWS / "out_of_the_flow_experience-flow_only.flow.csv"
    )
    assert {mode: len(rows) for mode, rows in flows.items()} == {
        "atomic": 13,
        "default": 15,
        "single": 4,
        "ms3": 16,
        "music21": 1,
    }
    assert flows["music21"] == (MeasureRange(1, 16, tuple("ABCDEFGHIJKLM")),)
    assert flows["ms3"][4] == MeasureRange(8, 10, ("F", "G"))


# endregion

# region The order a source states


@pytest.mark.parametrize(
    ("successors", "first", "order"),
    [
        pytest.param({1: [2], 2: [1, 3], 3: [-1]}, 1, [1, 2, 1, 2, 3], id="repeat"),
        pytest.param({1: [2, 3], 2: [1], 3: []}, 1, [1, 2, 1, 3], id="brackets"),
        pytest.param({1: [2, -1], 2: [1]}, 1, [1, 2, 1], id="da capo al fine"),
        pytest.param(
            {1: [2], 2: [1, 3], 3: [2, -1]},
            1,
            [1, 2, 1, 2, 3, 2, 3],
            id="the last successor serves again",
        ),
        pytest.param({1: [2], 2: [3], 3: [-1]}, 2, [2, 3], id="begun at 2"),
        pytest.param({1: [2], 2: [3], 3: []}, 1, [1, 2, 3], id="printed order"),
        pytest.param({1: [2], 2: [7]}, 1, [1, 2], id="a successor that is no measure"),
    ],
)
def test_successors_are_walked_by_arrival(
    successors: dict[int, list[int]], first: int, order: list[int]
) -> None:
    assert measure_order_from_successors(successors, first) == tuple(order)


def test_successors_that_never_end_are_refused() -> None:
    with pytest.raises(ValueError) as refused:
        measure_order_from_successors({1: [2], 2: [1]}, 1)
    assert str(refused.value) == (
        "the successors of measure count 1 lead back to it for ever: the order "
        "they state never ends"
    )
    with pytest.raises(ValueError, match="measure count 4 is no measure"):
        measure_order_from_successors({1: [2], 2: []}, 4)


def _repeated() -> ContinuousLogicalTimeline:
    """Three 4/4 bars with a repeat end at 8: the default flow is [0, 8) [0, 12)."""
    score = ContinuousLogicalTimeline(length=12, uid="score")
    score.add_measure_map(bars({}, END, {}))
    score.flow_control.add_all(flow_control_from_measures(score.measure_map))
    return score


@pytest.mark.parametrize(
    ("order", "regions", "chunks", "differs_at"),
    [
        ([1, 2, 1, 2, 3], [(0, 8), (0, 12)], [0, 1], None),
        ([1, 2, 1, 2], [(0, 8)], [0, 0], 16),
        ([1, 2, 3], [(0, 12)], [0], 8),
        ([1, 3], [(0, 4), (8, 12)], [0, 1], 4),
        ([3, 1], [(8, 12), (0, 4)], [0, 1], 0),
    ],
)
def test_a_stated_order_is_written_as_a_flow_map(
    order: list[int],
    regions: list[tuple[int, int]],
    chunks: list[int],
    differs_at: int | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    score = _repeated()
    assert [str(c.source) for c in score.get_flow().chunks] == [
        "[0, 8) quarters",
        "[0, 12) quarters",
    ]
    with caplog.at_level("WARNING"):
        stated = score.create_flow_map_from_measure_order(order)

    assert [
        (region.start.value, region.end.value, region.name) for region in stated.regions
    ] == [(start, end, None) for start, end in regions]
    assert [region.id for region in stated.regions] == score.list_regions()
    flow_map = stated.flow_map
    assert flow_map.chunks == tuple(stated.regions[index].id for index in chunks)
    assert (flow_map.timeline, flow_map.convention, flow_map.canonical) == (
        "score",
        None,
        False,
    )
    assert flow_map.provenance == (Provenance(method="imported"),)
    assert not flow_map.is_computed
    assert score.list_flow_maps() == [flow_map.id] == ["flow_map_1"]
    assert stated.differs_at == (
        None if differs_at is None else Coordinate(Fraction(differs_at), "quarters")
    )
    assert [record.getMessage() for record in caplog.records] == (
        []
        if differs_at is None
        else [
            "Timeline 'score': the order of measures its source states differs "
            "from the default flow of its flow control. They agree for the first "
            f"{differs_at} quarters played."
        ]
    )


def test_computing_the_flows_leaves_a_stated_flow_map_standing() -> None:
    score = _repeated()
    stated = score.create_flow_map_from_measure_order([1, 2, 1, 2, 3], id="stated")
    computed = score.create_flow_maps_from_flow_control()
    assert score.get_flow_map("stated") == stated.flow_map
    assert [region.id for region in stated.regions] == ["region_1", "region_2"]
    assert score.list_regions() == [
        "region_1",
        "region_2",
        *[section.id for section in computed.sections],
    ]
    # Two stretches there, three atomic sections here: the same flow.
    by_source, by_engine = (
        flow_map.resolve(score) for flow_map in (stated.flow_map, computed.flow_maps[0])
    )
    assert (len(by_source.chunks), len(by_engine.chunks)) == (2, 3)
    assert by_source.length == by_engine.length == Coordinate(Fraction(20), "quarters")
    assert [by_source.fold(position) for position in (0, 7, 8, 15, 16, 19)] == [
        by_engine.fold(position) for position in (0, 7, 8, 15, 16, 19)
    ]


def test_a_stated_order_that_cannot_be_written_changes_nothing() -> None:
    score = _repeated()
    score.create_region(0, 4, id="held")
    score.create_flow_map(["held"], id="taken")

    def held() -> tuple[list[str], list[str]]:
        return score.list_regions(), score.list_flow_maps()

    before = held()
    with pytest.raises(ValueError) as refused:
        score.create_flow_map_from_measure_order([1, 2, 9])
    assert str(refused.value) == (
        "the order names measure count 9, which is no measure of the measure "
        "map of timeline 'score' (counts 1 to 3)"
    )
    with pytest.raises(ValueError, match="the order of measures given .* is empty"):
        score.create_flow_map_from_measure_order([])
    with pytest.raises(ValueError, match="'taken'"):
        score.create_flow_map_from_measure_order([1, 2, 1, 2, 3], id="taken")
    assert held() == before == (["held"], ["taken"])

    without_measures = ContinuousLogicalTimeline(length=12, uid="plain")
    with pytest.raises(ValueError, match="timeline 'plain' holds no measure map"):
        without_measures.create_flow_map_from_measure_order([1])

    parent = ContinuousLogicalTimeline(length=24, uid="parent")
    child = _repeated()
    parent.add_child(child, 0)
    with pytest.raises(ValueError, match="a FlowMap names a root timeline"):
        child.create_flow_map_from_measure_order([1, 2, 3])
    assert (child.list_regions(), child.list_flow_maps()) == ([], [])


class _FlowWithGap(ContinuousLogicalTimeline):
    """A timeline whose get_flow answers a flow holding a gap."""

    def get_flow(self, *args: Any, **kwargs: Any) -> PlayedFlow:
        return PlayedFlow.from_stretches(
            self.id,
            [
                Interval(Coordinate(0, "quarters"), Coordinate(4, "quarters")),
                Duration(2, "quarters"),
            ],
            unit=self.unit,
            number_type=self.number_type,
        )


def test_a_played_flow_holding_a_gap_is_refused() -> None:
    score = _FlowWithGap(length=12, uid="score")
    score.add_measure_map(bars({}, {}, {}))
    with pytest.raises(TypeError) as refused:
        score.create_flow_map_from_measure_order([1, 2, 3])
    assert str(refused.value) == (
        "the flow get_flow returns for timeline 'score' holds a gap at 4 "
        "quarters: an order of measures is compared with stretches of the "
        "timeline only"
    )
    assert (score.list_regions(), score.list_flow_maps()) == ([], [])


# endregion

# region Findings


def test_a_repeat_end_behind_a_last_bracket_is_found() -> None:
    measures = bars(START, {"volta": 1, **END}, {"volta": 2}, END, {})
    assert repeat_end_findings(measures) == (
        MeasureFinding(MeasureFindingKind.repeat_end_behind_brackets, 4),
    )
    # Nothing is added because of it: the repeat end keeps no place.
    assert written(measures)[-1] == jump("jump_3", 16, repeat_end())


@pytest.mark.parametrize(
    ("rows", "found"),
    [
        pytest.param(
            (START, {"volta": 1, **END}, {"volta": 2}, START, END), [], id="start"
        ),
        pytest.param(({}, END, {}, END), [], id="behind a repeat end"),
        pytest.param(({}, END), [], id="nothing before"),
        pytest.param(({}, {"volta": 1, **END}, {"volta": 2}), [], id="its own bracket"),
        pytest.param(
            ({"volta": 1, **END}, {"volta": 2}, END, {}, END),
            [3],
            id="only the first behind the brackets",
        ),
    ],
)
def test_which_repeat_ends_are_found(
    rows: tuple[dict[str, Any], ...], found: list[int]
) -> None:
    assert [finding.count for finding in repeat_end_findings(bars(*rows))] == found


# endregion
