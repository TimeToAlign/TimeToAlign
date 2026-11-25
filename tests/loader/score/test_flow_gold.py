"""The gold flows, and the signs a score loader writes as flow control.

A piece's source is loaded, its timeline's flow control is played by the
engine, and the flow is stated in measure counts and compared with the rows
of the piece's ``.flow.csv`` file. See ``README.md``, "Gold flows", "Flows
written from a loaded score", "The order a source states" and "Signs
written as flow control".
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
import timetoshare

from timetoalign.core import Coordinate
from timetoalign.core.enums import (
    FlowFindingKind,
    FlowManifestation,
    MeasureFindingKind,
)
from timetoalign.loader.score import MeasureMapLoader, Ms3Loader, PartituraLoader
from timetoalign.timelines import StatedFlow, Timeline, TimelineGroup
from timetoalign.timelines.flow import (
    MeasureFinding,
    MeasureRange,
    measure_ranges,
    repeat_end_findings,
)
from timetoalign.timelines.flow.engine import play
from timetoalign.timelines.flowcontrol import Jump, JumpDestination

from .conftest import (
    EDITOR,
    SPECIMENS,
    atomic_ranges,
    get_specimen_path,
    gold_flows,
    played_ranges,
    repaired_stress_score,
)

# region Helpers


def _timeline(specimen: str) -> Timeline:
    """The timeline of a specimen's ms3 measures table."""
    path = get_specimen_path(SPECIMENS[specimen], "tsv")
    return Ms3Loader().load(path).create_timeline()


# endregion

# region Gold flows


@pytest.mark.parametrize(
    "specimen", ["c05n05_musete", "c11n08_Rondeau", "op18_no4_mov4_flow", "WoO71"]
)
def test_the_source_plays_the_gold_flows(specimen: str) -> None:
    timeline, gold = _timeline(specimen), gold_flows(specimen)
    assert atomic_ranges(timeline) == gold["atomic"]
    default, result = played_ranges(timeline, "default")
    assert default == gold["default"]
    assert result.findings == ()
    single, result = played_ranges(timeline, "single")
    assert single == gold["single"]
    assert result.findings == ()
    assert repeat_end_findings(timeline.measure_map) == ()


@pytest.mark.parametrize("specimen", ["rachmaninoff", "polyrhythm_only"])
def test_a_piece_without_flow_control_is_one_section_played_once(specimen: str) -> None:
    timeline, gold = _timeline(specimen), gold_flows(specimen)
    registry = timeline.flow_control
    assert registry.jumps == registry.breaks == registry.jump_targets == ()
    last = SPECIMENS[specimen].folded_measures + 1
    assert gold["atomic"] == gold["default"] == (MeasureRange(1, last, ("A",)),)
    assert "single" not in gold
    assert atomic_ranges(timeline) == gold["atomic"]
    for convention in ("default", "single"):
        ranges, result = played_ranges(timeline, convention)
        assert ranges == gold["default"]
        assert result.findings == ()


def test_dividing_at_section_breaks_is_the_form_of_the_file() -> None:
    """WoO 71's rows end at its twelve section breaks; the engine's runs do not."""
    timeline, gold = _timeline("WoO71"), gold_flows("WoO71")
    result = play(timeline.flow_control, timeline.length)
    undivided = measure_ranges(result.runs, timeline.measure_map)
    assert [(row.start, row.end) for row in undivided[:3]] == [
        (1, 22),
        (12, 43),
        (33, 94),
    ]
    assert [(row.start, row.end) for row in gold["default"][:5]] == [
        (1, 22),
        (12, 22),
        (22, 43),
        (33, 43),
        (43, 73),
    ]
    assert len(undivided) == 13
    assert len(gold["default"]) == 25


STRESS = "flow_only"


def test_the_stress_score_as_its_source_stands_ends_at_a_bracket() -> None:
    """The second of three brackets has no sign at its end: the flow ends there."""
    timeline, gold = _timeline(STRESS), gold_flows(STRESS)
    registry = timeline.flow_control
    assert atomic_ranges(timeline) == gold["atomic"]
    default, result = played_ranges(timeline, "default")
    assert default == gold["default"][:4]
    assert default == (
        MeasureRange(1, 4, ("A",)),
        MeasureRange(1, 6, ("A", "B", "C")),
        MeasureRange(4, 5, ("B",)),
        MeasureRange(6, 7, ("D",)),
    )
    assert [
        (finding.kind, finding.coordinate.value, finding.end and finding.end.value)
        for finding in result.findings
    ] == [
        (FlowFindingKind.flow_ends_at_break, 10, None),
        (FlowFindingKind.never_played, 10, 37),
    ]
    # The break it ends at lies between the second and the third bracket.
    ending = [element for element in registry.breaks if element.coordinate.value == 10]
    assert [(element.manifestation, element.ends) for element in ending] == [
        (FlowManifestation.volta, None)
    ]
    assert repeat_end_findings(timeline.measure_map) == (
        MeasureFinding(MeasureFindingKind.repeat_end_behind_brackets, 8),
    )


@pytest.mark.parametrize("convention", ["default", "single"])
def test_the_repaired_stress_score_plays_its_gold_flow(convention: str) -> None:
    timeline, gold = _timeline(STRESS), gold_flows(STRESS)
    loaded = timeline.flow_control.to_dict()
    registry = repaired_stress_score(timeline)

    # The repair worked on a copy: the loaded registry holds what it held.
    assert timeline.flow_control.to_dict() == loaded
    # Nothing was deleted or changed in place: five elements were added, and
    # two of the loader's are rejected, by the editor.
    assert len(registry.elements) == len(loaded) + 5
    rejected = [element for element in registry.elements if element.is_rejected]
    assert [
        (type(element), element.coordinate.value, len(element.targets))
        for element in rejected
    ] == [
        (Jump, 16, 1),
        (Jump, 34, 1),
    ]  # MC 8 is [12, 16), MC 14 is [32, 34)
    for element in rejected:
        (entry,) = element.provenance
        assert entry.to_dict() == {
            "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}]
        }
        assert registry.jump_at(element.coordinate).id != element.id
    assert len(registry.active_elements()) == len(loaded) + 3
    assert [
        (element.id, [entry.to_dict() for entry in element.provenance])
        for element in registry.active_elements()
        if element.provenance is not None
    ] == [
        (added, [{"stated_by": [EDITOR], "method": "manual"}])
        for added in (
            "on_to_8",
            "start_8",
            "repeat_to_start_8",
            "start_13",
            "repeat_to_start_13",
        )
    ]

    ranges, result = played_ranges(timeline, convention, registry=registry)
    assert ranges == gold[convention]
    assert result.findings == ()
    assert atomic_ranges(timeline, registry=registry) == gold["atomic"]


# endregion

# region Flows written from a loaded score


@pytest.mark.parametrize(
    ("specimen", "chunks"),
    [("op18_no4_mov4_flow", (20, 10)), ("WoO71", (37, 25))],
)
def test_the_flows_written_for_a_loaded_score_are_the_gold_file_s(
    specimen: str, chunks: tuple[int, int]
) -> None:
    timeline, gold = _timeline(specimen), gold_flows(specimen)
    written = timeline.create_flow_maps_from_flow_control()

    sections = written.sections
    assert all(region.is_atomic_section for region in sections)
    in_counts = measure_ranges(
        [region.as_interval for region in sections], timeline.measure_map
    )
    assert [
        MeasureRange(row.start, row.end, (region.name,))
        for row, region in zip(in_counts, sections, strict=True)
    ] == list(gold["atomic"])

    name = {region.id: region.name for region in sections}
    flow_maps = {str(flow_map.convention): flow_map for flow_map in written.flow_maps}
    assert sorted(flow_maps) == ["default", "printed", "single"]
    for convention, expected in zip(("default", "single"), chunks, strict=True):
        letters = [name[chunk] for chunk in flow_maps[convention].chunks]
        assert letters == [
            letter for row in gold[convention] for letter in row.sections
        ]
        assert len(letters) == expected

    manifest = timeline.to_dict()
    timetoshare.validate(manifest)
    assert [flow_map["convention"] for flow_map in manifest["flow_maps"]] == [
        "default",
        "single",
        "printed",
    ]
    again = Timeline.from_dict(manifest)
    assert again.to_dict() == manifest
    assert [again.get_region(id) for id in again.list_regions()] == list(sections)
    assert [again.get_flow_map(id) for id in again.list_flow_maps()] == list(
        written.flow_maps
    )
    assert again.flow_control.to_dict() == timeline.flow_control.to_dict()


# endregion

# region The order a source states


def _stated_rows(timeline: Timeline, stated: StatedFlow) -> list[tuple[int, int]]:
    """A stated FlowMap in measure counts, divided as a gold file's rows are."""
    stretches = [chunk.source for chunk in stated.flow_map.resolve(timeline).chunks]
    rows = measure_ranges(
        stretches,
        timeline.measure_map,
        divide_at=[e.coordinate for e in timeline.flow_control.section_breaks],
    )
    return [(row.start, row.end) for row in rows]


@pytest.mark.parametrize(
    "specimen", ["c05n05_musete", "c11n08_Rondeau", "op18_no4_mov4_flow", "WoO71"]
)
def test_the_successors_of_an_ms3_table_state_the_gold_flow(
    specimen: str, caplog: pytest.LogCaptureFixture
) -> None:
    loader = Ms3Loader().load(get_specimen_path(SPECIMENS[specimen], "tsv"))
    timeline = loader.create_timeline()
    with caplog.at_level(logging.WARNING):
        stated = loader.create_stated_flow_map(timeline)
    assert _stated_rows(timeline, stated) == [
        (row.start, row.end) for row in gold_flows(specimen)["default"]
    ]
    assert stated.differs_at is None
    assert caplog.records == []
    assert stated.flow_map.to_dict() == {
        "id": "flow_map_1",
        "timeline": timeline.id,
        "chunks": list(stated.flow_map.chunks),
        "provenance": [{"method": "imported"}],
    }
    assert timeline.list_flow_maps() == ["flow_map_1"]


def test_the_stress_score_s_successors_state_another_flow_than_its_signs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    loader = Ms3Loader().load(get_specimen_path(SPECIMENS[STRESS], "tsv"))
    timeline = loader.create_timeline()
    with caplog.at_level(logging.WARNING):
        stated = loader.create_stated_flow_map(timeline)
    ms3_rows = gold_flows(STRESS)["ms3"]
    assert len(ms3_rows) == 16
    assert _stated_rows(timeline, stated) == [(row.start, row.end) for row in ms3_rows]
    # The default flow ends at the Break at 10: [0,6) [0,9) [6,7) [9,10).
    assert stated.differs_at == Coordinate(Fraction(17), "quarters")
    assert [record.getMessage() for record in caplog.records] == [
        f"Timeline '{timeline.id}': the order of measures its source states "
        "differs from the default flow of its flow control. They agree for the "
        "first 17 quarters played."
    ]


def test_a_measure_map_file_s_successors_against_its_repeats_without_brackets() -> None:
    specimen = SPECIMENS["op18_no4_mov4_flow"]
    loader = MeasureMapLoader().load(get_specimen_path(specimen, "mm_json"))
    timeline = loader.create_timeline()
    assert {measure.volta for measure in timeline.measure_map} == {None}
    stated = loader.create_stated_flow_map(timeline)

    from_table = Ms3Loader().load(get_specimen_path(specimen, "tsv"))
    table_timeline = from_table.create_timeline()
    assert [str(region.as_interval) for region in stated.regions] == [
        str(region.as_interval)
        for region in from_table.create_stated_flow_map(table_timeline).regions
    ]
    assert [str(region.as_interval) for region in stated.regions[:5]] == [
        "[0, 32) quarters",
        "[0, 64) quarters",
        "[32, 96) quarters",
        "[64, 160) quarters",
        "[96, 157) quarters",
    ]
    # 32 + 64 + 64 + 96 + 61: then the file's repeats play the first ending again.
    assert stated.differs_at == Coordinate(Fraction(317), "quarters")


def test_a_source_without_successors_states_no_order() -> None:
    specimen = SPECIMENS["op18_no4_mov4_flow"]
    loader = PartituraLoader().load(get_specimen_path(specimen, "musicxml"))
    timeline = loader.create_timeline()
    assert loader.create_stated_flow_map(timeline) is None
    assert (timeline.list_regions(), timeline.list_flow_maps()) == ([], [])


# endregion

# region Signs written as flow control


def test_the_musette_states_two_dal_segno_and_one_fine() -> None:
    """Every element of the Musette, and the column of the table it is read from."""
    timeline = _timeline("c05n05_musete")
    assert timeline.flow_control.to_dict() == [
        # markers "segno" (MC 6), named by jump_bwd
        {
            "id": "jump_target_3",
            "role": "jump_target",
            "coordinate": [12, 1],
            "manifestation": "segno",
            "name": "segno",
        },
        # markers "fine" (MC 16) for the play_until of MC 58, which has no jump_fwd
        {
            "id": "break_1",
            "role": "break",
            "coordinate": [42, 1],
            "manifestation": "fine",
            "ends": {"after": {"jump": "jump_3", "to": "jump_target_3"}},
            "name": "fine",
        },
        # repeats "end" (MC 16); markers "fine" for the play_until and jump_fwd of MC 31
        {
            "id": "jump_1",
            "role": "jump",
            "coordinate": [42, 1],
            "targets": [
                {"repeat_count": 1, "manifestation": "repeat_end"},
                {
                    "to": "jump_target_5",
                    "after": {"jump": "jump_2", "to": "jump_target_3"},
                    "label": "fine",
                },
            ],
        },
        # repeats "start" (MC 17)
        {
            "id": "jump_target_1",
            "role": "jump_target",
            "coordinate": [42, 1],
            "manifestation": "repeat_start",
        },
        # markers "1e" (MC 17), named by no row
        {
            "id": "jump_target_4",
            "role": "jump_target",
            "coordinate": [42, 1],
            "name": "1e",
        },
        # repeats "end" and jump_bwd "segno" (MC 31)
        {
            "id": "jump_2",
            "role": "jump",
            "coordinate": [84, 1],
            "targets": [
                {
                    "to": "jump_target_1",
                    "repeat_count": 1,
                    "manifestation": "repeat_end",
                },
                {
                    "to": "jump_target_3",
                    "repeat_count": 1,
                    "reset": False,
                    "manifestation": "dal_segno_al_fine",
                },
            ],
        },
        # repeats "start" (MC 32)
        {
            "id": "jump_target_2",
            "role": "jump_target",
            "coordinate": [84, 1],
            "manifestation": "repeat_start",
        },
        # markers "2e" (MC 32), named by jump_fwd
        {
            "id": "jump_target_5",
            "role": "jump_target",
            "coordinate": [84, 1],
            "name": "2e",
        },
        # repeats "end" and jump_bwd "segno" (MC 58)
        {
            "id": "jump_3",
            "role": "jump",
            "coordinate": [162, 1],
            "targets": [
                {
                    "to": "jump_target_2",
                    "repeat_count": 1,
                    "manifestation": "repeat_end",
                },
                {
                    "to": "jump_target_3",
                    "repeat_count": 1,
                    "reset": False,
                    "manifestation": "dal_segno_al_fine",
                },
            ],
        },
    ]


def test_a_marker_only_played_until_is_no_jump_target() -> None:
    """The Rondeau's segno ends the refrain: both D.C. play until it and go on."""
    timeline = _timeline("c11n08_Rondeau")
    registry = timeline.flow_control
    assert [target.name for target in registry.jump_targets if target.name] == [
        "1e",
        "2e",
        "3e",
    ]
    refrain_end = registry.jump_at(12)
    assert refrain_end is not None
    assert [destination.to_dict() for destination in refrain_end.targets] == [
        {"repeat_count": 1, "manifestation": "repeat_end"},
        {"to": "jump_target_5", "after": {"jump": "jump_2"}, "label": "segno"},
        {"to": "jump_target_6", "after": {"jump": "jump_3"}, "label": "segno"},
    ]
    assert [
        destination.to_dict()
        for jump in registry.jumps
        for destination in jump.targets
        if destination.manifestation is FlowManifestation.da_capo
    ] == [{"repeat_count": 1, "reset": False, "manifestation": "da_capo"}] * 2


def test_section_breaks_and_barlines_lie_at_the_ends_of_their_measures() -> None:
    timeline = _timeline("WoO71")
    registry = timeline.flow_control
    ends = {
        measure.qstamp + measure.actual_length: measure.count
        for measure in timeline.measure_map
    }
    assert [ends[element.coordinate.value] for element in registry.section_breaks] == [
        21,
        42,
        72,
        93,
        114,
        135,
        156,
        177,
        198,
        219,
        240,
        261,
    ]
    assert [
        (ends[element.coordinate.value], element.manifestation)
        for element in registry.barlines
    ] == [
        (53, FlowManifestation.double_barline),
        (72, FlowManifestation.final_barline),
        (261, FlowManifestation.double_barline),
        (329, FlowManifestation.double_barline),
        (371, FlowManifestation.double_barline),
    ]


def test_the_successors_a_source_states_stay_with_the_measures() -> None:
    """``next`` is not flow control; the table and the measure map keep it."""
    loader = Ms3Loader().load(get_specimen_path(SPECIMENS["c05n05_musete"], "tsv"))
    assert loader.store.measures.column_values("next")[15] == "1, 17, 32, -1"
    measure_map = loader.create_timeline().measure_map
    assert measure_map[15].next == ("1", "17", "32")
    assert measure_map[30].next == ("17", "6")


COLUMNS = (
    "mc",
    "mn",
    "quarterbeats",
    "duration_qb",
    "timesig",
    "act_dur",
    "repeats",
    "volta",
    "breaks",
    "barline",
    "markers",
    "jump_bwd",
    "jump_fwd",
    "play_until",
)


def _table(directory: Path, *rows: dict[str, Any]) -> Timeline:
    """The timeline of a measures table of 4/4 bars, one per row."""
    lines = ["\t".join(COLUMNS)]
    for count, row in enumerate(rows, start=1):
        cells = {
            "mc": count,
            "mn": count,
            "quarterbeats": 4 * (count - 1),
            "duration_qb": 4.0,
            "timesig": "4/4",
            "act_dur": "1",
            **row,
        }
        lines.append("\t".join(str(cells.get(column, "")) for column in COLUMNS))
    path = directory / "piece.measures.tsv"
    path.write_text("\n".join(lines) + "\n")
    return Ms3Loader().load(path).create_timeline()


def _runs(timeline: Timeline) -> list[tuple[Any, Any]]:
    result = play(timeline.flow_control, timeline.length)
    return [(run.start.value, run.end.value) for run in result.runs]


def test_a_dal_segno_al_coda_names_its_leap(tmp_path: Path) -> None:
    timeline = _table(
        tmp_path,
        {},
        {"markers": "segno", "repeats": "start"},
        {},
        {"repeats": "end", "markers": "coda"},
        {
            "jump_bwd": "segno",
            "jump_fwd": "codab",
            "play_until": "coda",
            "barline": "double",
        },
        {"markers": "codab", "breaks": "section"},
    )
    assert timeline.flow_control.to_dict() == [
        {
            "id": "jump_target_1",
            "role": "jump_target",
            "coordinate": [4, 1],
            "manifestation": "repeat_start",
        },
        {
            "id": "jump_target_2",
            "role": "jump_target",
            "coordinate": [4, 1],
            "manifestation": "segno",
            "name": "segno",
        },
        {
            "id": "jump_1",
            "role": "jump",
            "coordinate": [16, 1],
            "targets": [
                {
                    "to": "jump_target_1",
                    "repeat_count": 1,
                    "manifestation": "repeat_end",
                },
                {
                    "to": "jump_target_3",
                    "after": {"jump": "jump_2", "to": "jump_target_2"},
                    "manifestation": "to_coda",
                    "label": "coda",
                },
            ],
        },
        {
            "id": "barline_1",
            "role": "barline",
            "coordinate": [20, 1],
            "manifestation": "double_barline",
        },
        {
            "id": "jump_2",
            "role": "jump",
            "coordinate": [20, 1],
            "targets": [
                {
                    "to": "jump_target_2",
                    "repeat_count": 1,
                    "reset": False,
                    "manifestation": "dal_segno_al_coda",
                }
            ],
        },
        {
            "id": "jump_target_3",
            "role": "jump_target",
            "coordinate": [20, 1],
            "manifestation": "coda",
            "name": "codab",
        },
        {"id": "section_break_1", "role": "section_break", "coordinate": [24, 1]},
    ]
    assert _runs(timeline) == [(0, 16), (4, 20), (4, 16), (20, 24)]


def test_a_da_capo_al_fine_is_a_break_that_ends_after_it(tmp_path: Path) -> None:
    timeline = _table(
        tmp_path,
        {},
        {"markers": "fine"},
        {"jump_bwd": "start", "play_until": "fine", "barline": "end"},
    )
    assert timeline.flow_control.to_dict() == [
        {
            "id": "break_1",
            "role": "break",
            "coordinate": [8, 1],
            "manifestation": "fine",
            "ends": {"after": {"jump": "jump_1"}},
            "name": "fine",
        },
        {
            "id": "barline_1",
            "role": "barline",
            "coordinate": [12, 1],
            "manifestation": "final_barline",
        },
        {
            "id": "jump_1",
            "role": "jump",
            "coordinate": [12, 1],
            "targets": [
                {"repeat_count": 1, "reset": False, "manifestation": "da_capo_al_fine"}
            ],
        },
    ]
    assert _runs(timeline) == [(0, 12), (0, 8)]


def test_two_leaps_playing_until_one_fine_give_one_break_each(tmp_path: Path) -> None:
    timeline = _table(
        tmp_path,
        {"markers": "segno"},
        {"markers": "fine"},
        {"jump_bwd": "segno", "play_until": "fine"},
        {},
        {"jump_bwd": "start", "play_until": "fine"},
    )
    assert [element.to_dict() for element in timeline.flow_control.breaks] == [
        {
            "id": "break_1",
            "role": "break",
            "coordinate": [8, 1],
            "manifestation": "fine",
            "ends": {"after": {"jump": "jump_1", "to": "jump_target_1"}},
            "name": "fine",
        },
        {
            "id": "break_2",
            "role": "break",
            "coordinate": [8, 1],
            "manifestation": "fine",
            "ends": {"after": {"jump": "jump_2"}},
            "name": "fine",
        },
    ]
    # The D.S. is taken first, and the Fine ends the flow after it.
    assert _runs(timeline) == [(0, 12), (0, 8)]


def test_both_repeat_signs_on_one_measure_are_read(tmp_path: Path) -> None:
    timeline = _table(tmp_path, {}, {"repeats": "startend"}, {})
    assert timeline.flow_control.to_dict() == [
        {
            "id": "jump_target_1",
            "role": "jump_target",
            "coordinate": [4, 1],
            "manifestation": "repeat_start",
        },
        {
            "id": "jump_1",
            "role": "jump",
            "coordinate": [8, 1],
            "targets": [
                {
                    "to": "jump_target_1",
                    "repeat_count": 1,
                    "manifestation": "repeat_end",
                }
            ],
        },
    ]
    assert _runs(timeline) == [(0, 8), (4, 12)]


def test_only_a_section_break_ends_a_section(tmp_path: Path) -> None:
    timeline = _table(
        tmp_path,
        {"breaks": "line"},
        {"breaks": "section & page"},
        {"breaks": "section"},
    )
    assert timeline.flow_control.to_dict() == [
        {"id": "section_break_1", "role": "section_break", "coordinate": [8, 1]},
        {"id": "section_break_2", "role": "section_break", "coordinate": [12, 1]},
    ]


def test_a_sign_no_row_names_is_a_jump_target(tmp_path: Path) -> None:
    timeline = _table(tmp_path, {}, {"markers": "segno & coda2"}, {})
    assert timeline.flow_control.to_dict() == [
        {
            "id": "jump_target_1",
            "role": "jump_target",
            "coordinate": [4, 1],
            "manifestation": "segno",
            "name": "segno",
        },
        {
            "id": "jump_target_2",
            "role": "jump_target",
            "coordinate": [4, 1],
            "manifestation": "coda",
            "name": "coda2",
        },
    ]


def test_a_fine_no_leap_names_ends_the_flow_from_its_second_arrival(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="timetoalign.loader.score.base"):
        timeline = _table(tmp_path, {}, {"markers": "fine"}, {})
    assert caplog.records == []
    assert timeline.flow_control.to_dict() == [
        {
            "id": "break_1",
            "role": "break",
            "coordinate": [8, 1],
            "manifestation": "fine",
            "ends": {"passes": {"from": 2}},
            "name": "fine",
        }
    ]
    # Arrived at once, the Fine does not act.
    played = timeline.get_flow()
    assert _runs(timeline) == [(0, 12)]
    assert played.findings == ()
    # With the D.C. the source lost, it ends the flow where it stands.
    timeline.flow_control.add(
        Jump(
            Coordinate(12, "quarters"),
            [JumpDestination(repeat_count=1, reset=False, manifestation="da_capo")],
        )
    )
    assert _runs(timeline) == [(0, 12), (0, 8)]
    assert timeline.get_flow().findings == ()
    timetoshare.validate(timeline.to_dict())


def test_a_da_capo_beside_a_repeat_end_without_start_names_the_timeline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="timetoalign.loader.score.base"):
        timeline = _table(
            tmp_path,
            {"breaks": "section"},
            {},
            {"repeats": "end", "jump_bwd": "start"},
            {},
        )
    assert caplog.records == []
    assert timeline.id == "clt_1"
    assert timeline.flow_control.to_dict() == [
        {"id": "section_break_1", "role": "section_break", "coordinate": [4, 1]},
        {
            "id": "jump_1",
            "role": "jump",
            "coordinate": [12, 1],
            "targets": [
                {"repeat_count": 1, "manifestation": "repeat_end"},
                {
                    "to": "clt_1",
                    "repeat_count": 1,
                    "reset": False,
                    "manifestation": "da_capo",
                },
            ],
        },
    ]
    (jump,) = timeline.flow_control.jumps
    assert [
        timeline.flow_control.place_of(destination, jump).value
        for destination in jump.targets
    ] == [4, 0]
    # The repeat goes back to its section's start, the D.C. to the timeline's.
    assert _runs(timeline) == [(0, 12), (4, 12), (0, 16)]
    assert timeline.get_flow().findings == ()
    timetoshare.validate(timeline.to_dict())


def test_a_da_capo_naming_the_timeline_follows_its_id(tmp_path: Path) -> None:
    rows = ({}, {"repeats": "end", "jump_bwd": "start"})
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
    first = _table(tmp_path / "a", *rows)
    second = _table(tmp_path / "b", *rows)
    assert (first.id, second.id) == ("clt_1", "clt_1")
    TimelineGroup(timelines=[first, second])
    assert (first.id, second.id) == ("clt_1", "clt_2")
    assert [
        [destination.to for destination in timeline.flow_control.jumps[0].targets]
        for timeline in (first, second)
    ] == [[None, "clt_1"], [None, "clt_2"]]
    assert _runs(second) == [(0, 8), (0, 8), (0, 8)]


LEFT_OUT = [
    pytest.param(
        ({}, {"jump_bwd": "segno"}),
        [],
        "The D.S. of MC 2 goes back to 'segno', which no measure carries; it is not "
        "written.",
        id="a D.S. without its segno",
    ),
    pytest.param(
        ({}, {"jump_fwd": "coda"}, {"markers": "coda"}),
        [{"role": "jump_target", "coordinate": [8, 1], "name": "coda"}],
        "MC 2 goes on at 'coda' without a D.C. or D.S. it follows; the jump is not "
        "written.",
        id="a jump forward without a leap",
    ),
    pytest.param(
        (
            {"markers": "segno"},
            {"markers": "coda"},
            {"jump_bwd": "segno", "play_until": "coda", "jump_fwd": "codab"},
            {"jump_bwd": "start", "play_until": "coda", "jump_fwd": "codab"},
            {"markers": "codab"},
        ),
        [
            {"role": "jump_target", "coordinate": [0, 1], "name": "segno"},
            {"role": "jump", "coordinate": [8, 1], "manifestations": ["to_coda"]},
            {
                "role": "jump",
                "coordinate": [12, 1],
                "manifestations": ["dal_segno_al_coda"],
            },
            {
                "role": "jump",
                "coordinate": [16, 1],
                "manifestations": ["da_capo_al_coda"],
            },
            {"role": "jump_target", "coordinate": [16, 1], "name": "codab"},
        ],
        "The jump from 'coda' to 'codab' that MC 4 states cannot be told from another "
        "sign of its barline; it is not written.",
        id="a To Coda shared by two leaps",
    ),
    pytest.param(
        ({}, {"jump_bwd": "start", "play_until": "fine"}),
        [{"role": "jump", "coordinate": [8, 1], "manifestations": ["da_capo_al_fine"]}],
        "MC 2 plays until 'fine', which no measure carries; it is not written.",
        id="a D.C. al Fine without a Fine",
    ),
    pytest.param(
        (
            {"markers": "coda"},
            {"jump_bwd": "start", "play_until": "coda", "jump_fwd": "codab"},
        ),
        [{"role": "jump", "coordinate": [8, 1], "manifestations": ["da_capo_al_coda"]}],
        "MC 2 goes on at 'codab', which no measure carries; the jump is not written.",
        id="an al Coda without its coda",
    ),
    pytest.param(
        ({"markers": "x"}, {"jump_bwd": "start", "jump_fwd": "x"}),
        [
            {"role": "jump_target", "coordinate": [0, 1], "name": "x"},
            {"role": "jump", "coordinate": [8, 1], "manifestations": ["da_capo"]},
        ],
        "MC 2 goes on at 'x' without saying from where; the jump is not written.",
        id="a jump forward without play_until",
    ),
    pytest.param(
        (
            {},
            {
                "markers": "coda",
                "jump_bwd": "start",
                "play_until": "coda",
                "jump_fwd": "codab",
            },
            {"markers": "codab"},
        ),
        [
            {
                "role": "jump",
                "coordinate": [8, 1],
                "manifestations": ["da_capo_al_coda"],
            },
            {"role": "jump_target", "coordinate": [8, 1], "name": "codab"},
        ],
        "The jump from 'coda' to 'codab' that MC 2 states cannot be told from another "
        "sign of its barline; it is not written.",
        id="a To Coda in the measure of its own leap",
    ),
]


@pytest.mark.parametrize(("rows", "elements", "warning"), LEFT_OUT)
def test_what_cannot_be_written_is_left_out_and_logged(
    rows: tuple[dict[str, Any], ...],
    elements: list[dict[str, Any]],
    warning: str,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="timetoalign.loader.score.base"):
        timeline = _table(tmp_path, *rows)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "timetoalign.loader.score.base"
    ] == [warning]

    def summary(element: Any) -> dict[str, Any]:
        stated = element.to_dict()
        brief = {"role": stated["role"], "coordinate": stated["coordinate"]}
        if isinstance(element, Jump):
            brief["manifestations"] = [str(m.value) for m in element.manifestations]
        else:
            brief["name"] = element.name
        return brief

    assert [summary(element) for element in timeline.flow_control.elements] == elements


# endregion
