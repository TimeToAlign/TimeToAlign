"""Exact timeline-native structure tests for score loaders."""

from __future__ import annotations

import json
import warnings
from fractions import Fraction

import pytest

from timetoalign.core import Coordinate, FlowControlElement, TimeUnit
from timetoalign.loader.score import Ms3Loader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap, Timeline

COUPERIN = ensure_data("score") / "couperin_concerts"
RONDEAU = COUPERIN / "c11n08_Rondeau.measures.tsv"
RONDEAU_UNFOLDED = COUPERIN / "c11n08_Rondeau_unfolded.measures.tsv"
REPEAT_TARGETS = {9: 1, 18: 10, 27: 19, 60: 28}
DA_CAPO_SOURCES = (18, 27)


def _rondeau_timeline(*, flatten: bool = False) -> Timeline:
    return Ms3Loader.from_file(RONDEAU).create_timeline(flatten=flatten)


def test_rondeau_stores_one_measure_map_shared_with_the_skeleton() -> None:
    timeline = _rondeau_timeline()
    assert isinstance(timeline.measure_map, MeasureMap)
    assert timeline.skeleton.section_hierarchy.measure_map is timeline.measure_map


def test_repeat_end_jumps_follow_exact_measure_boundaries() -> None:
    timeline = _rondeau_timeline()
    assert timeline.measure_map is not None
    measures = {measure.count: measure for measure in timeline.measure_map}
    all_repeat_jumps = [
        jump
        for jump in timeline.flow_control.jumps
        if jump.control_type is FlowControlElement.repeat_end
    ]
    assert len(all_repeat_jumps) == 4
    repeat_jumps = {
        source_mc: next(
            jump
            for jump in timeline.flow_control.jumps
            if jump.control_type is FlowControlElement.repeat_end
            and jump.from_coordinate
            == Coordinate(
                measures[source_mc].qstamp + measures[source_mc].actual_length,
                TimeUnit.quarters,
            )
        )
        for source_mc in REPEAT_TARGETS
    }
    assert set(repeat_jumps) == {9, 18, 27, 60}
    for source_mc, target_mc in REPEAT_TARGETS.items():
        source = measures[source_mc]
        target = measures[target_mc]
        jump = repeat_jumps[source_mc]
        assert jump.from_coordinate == Coordinate(
            source.qstamp + source.actual_length, TimeUnit.quarters
        )
        assert jump.to_coordinate == Coordinate(target.qstamp, TimeUnit.quarters)


def test_da_capo_jumps_leave_measure_ends_for_the_piece_start() -> None:
    timeline = _rondeau_timeline()
    assert timeline.measure_map is not None
    measures = {measure.count: measure for measure in timeline.measure_map}
    assert len(timeline.flow_control.jumps) == 6
    da_capo_jumps = [
        jump
        for jump in timeline.flow_control.jumps
        if jump.control_type is FlowControlElement.da_capo
    ]
    assert len(da_capo_jumps) == 2
    start = Coordinate(measures[1].qstamp, TimeUnit.quarters)
    for jump, source_mc in zip(da_capo_jumps, DA_CAPO_SOURCES):
        source = measures[source_mc]
        assert jump.from_coordinate == Coordinate(
            source.qstamp + source.actual_length, TimeUnit.quarters
        )
        assert jump.to_coordinate == start
    assert [(jump.from_position, jump.to_position) for jump in da_capo_jumps] == [
        (Fraction(24), Fraction(0)),
        (Fraction(36), Fraction(0)),
    ]


def test_break_and_marker_facts_are_derived_from_the_tsv_columns() -> None:
    timeline = _rondeau_timeline()
    assert timeline.flow_control.breaks == []
    assert [
        (marker.id, marker.name, marker.coordinate)
        for marker in timeline.flow_control.markers
    ] == [("marker_1", "segno", Coordinate(Fraction(11), TimeUnit.quarters))]


def test_flatten_keeps_source_structure_without_attaching_a_skeleton() -> None:
    timeline = _rondeau_timeline(flatten=True)
    structured = _rondeau_timeline()
    assert isinstance(timeline.measure_map, MeasureMap)
    assert timeline.measure_map == structured.measure_map
    assert timeline.skeletons == ()
    assert timeline.flow_control.to_dict() == structured.flow_control.to_dict()


def test_rondeau_timeline_structure_round_trips_through_json() -> None:
    timeline = _rondeau_timeline()
    restored = Timeline.from_dict(json.loads(json.dumps(timeline.to_dict())))
    assert restored.to_dict() == timeline.to_dict()
    assert len(restored.flow_control.jumps) == 6
    assert restored.flow_control.to_dict() == timeline.flow_control.to_dict()
    assert restored.measure_map is not None
    assert timeline.measure_map is not None
    assert restored.measure_map.to_dict() == timeline.measure_map.to_dict()


def test_unfolded_measures_table_cannot_form_a_timeline_measure_map() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError, match="Measure IDs must be unique within a MeasureMap"
        ):
            Ms3Loader.from_file(RONDEAU_UNFOLDED).create_timeline()
    assert recorded == []
