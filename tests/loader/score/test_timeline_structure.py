"""Exact timeline-native structure tests for score loaders."""

from __future__ import annotations

import json
import warnings
from fractions import Fraction

import pytest

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.core.enums import FlowManifestation
from timetoalign.loader.score import Ms3Loader
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap, Timeline
from timetoalign.timelines.flowcontrol import FlowControlRegistry

COUPERIN = ensure_data("score") / "couperin_concerts"
RONDEAU = COUPERIN / "c11n08_Rondeau.measures.tsv"
RONDEAU_UNFOLDED = COUPERIN / "c11n08_Rondeau_unfolded.measures.tsv"
REPEAT_TARGETS = {9: 1, 18: 10, 27: 19, 60: 28}
DA_CAPO_SOURCES = (18, 27)


def _rondeau_timeline(*, flatten: bool = False) -> Timeline:
    return Ms3Loader.from_file(RONDEAU).create_timeline(flatten=flatten)


def _places(timeline: Timeline, manifestation: FlowManifestation) -> dict:
    """Where each destination that looks like *manifestation* goes on, by its Jump."""
    registry = timeline.flow_control
    return {
        jump.coordinate: registry.place_of(destination, jump)
        for jump in registry.jumps
        for destination in jump.targets
        if destination.manifestation is manifestation
    }


def test_rondeau_stores_one_measure_map() -> None:
    timeline = _rondeau_timeline()
    assert isinstance(timeline.measure_map, MeasureMap)
    assert timeline.measure_map is timeline.measure_map
    assert len(timeline.measure_map) == 60


def test_repeat_ends_follow_exact_measure_boundaries() -> None:
    timeline = _rondeau_timeline()
    assert timeline.measure_map is not None
    measures = {measure.count: measure for measure in timeline.measure_map}
    assert _places(timeline, FlowManifestation.repeat_end) == {
        Coordinate(
            measures[source].qstamp + measures[source].actual_length, TimeUnit.quarters
        ): Coordinate(measures[target].qstamp, TimeUnit.quarters)
        for source, target in REPEAT_TARGETS.items()
    }
    # The first repeat end has no printed start: it states no place, and the
    # place it goes on at is the start of its section.
    first = timeline.flow_control.jump_at(12)
    assert first is not None
    assert first.targets[0].to_dict() == {
        "repeat_count": 1,
        "manifestation": "repeat_end",
    }


def test_da_capo_leaves_measure_ends_for_the_piece_start() -> None:
    timeline = _rondeau_timeline()
    assert len(timeline.flow_control.jumps) == 4
    start = Coordinate(Fraction(0), TimeUnit.quarters)
    assert _places(timeline, FlowManifestation.da_capo) == {
        Coordinate(Fraction(24), TimeUnit.quarters): start,
        Coordinate(Fraction(36), TimeUnit.quarters): start,
    }
    assert timeline.measure_map is not None
    measures = {measure.count: measure for measure in timeline.measure_map}
    assert [
        measures[source].qstamp + measures[source].actual_length
        for source in DA_CAPO_SOURCES
    ] == [Fraction(24), Fraction(36)]


def test_break_and_marker_facts_are_read_from_the_tsv_columns() -> None:
    timeline = _rondeau_timeline()
    assert timeline.flow_control.breaks == []
    assert timeline.flow_control.section_breaks == []
    # The markers 1e, 2e and 3e are places; the segno of MC 9 is where both
    # D.C. play until, and no place.
    assert [
        (target.name, target.manifestation, target.coordinate)
        for target in timeline.flow_control.jump_targets
        if target.name is not None
    ] == [
        ("1e", None, Coordinate(Fraction(12), TimeUnit.quarters)),
        ("2e", None, Coordinate(Fraction(24), TimeUnit.quarters)),
        ("3e", None, Coordinate(Fraction(36), TimeUnit.quarters)),
    ]


def test_flatten_keeps_source_structure() -> None:
    timeline = _rondeau_timeline(flatten=True)
    structured = _rondeau_timeline()
    assert isinstance(timeline.measure_map, MeasureMap)
    assert timeline.measure_map == structured.measure_map
    assert timeline.flow_control.to_dict() == structured.flow_control.to_dict()


def test_rondeau_timeline_structure_round_trips_through_json() -> None:
    timeline = _rondeau_timeline()
    restored = Timeline.from_dict(json.loads(json.dumps(timeline.to_dict())))
    assert restored.to_dict() == timeline.to_dict()
    assert len(restored.flow_control.jumps) == 4
    assert restored.flow_control.to_dict() == timeline.flow_control.to_dict()
    assert restored.measure_map is not None
    assert timeline.measure_map is not None
    assert restored.measure_map.to_dict() == timeline.measure_map.to_dict()


def test_rondeau_flow_control_round_trips_through_json() -> None:
    timeline = _rondeau_timeline()
    written = timeline.flow_control.to_dict()
    restored = FlowControlRegistry.from_dict(
        json.loads(json.dumps(written)), unit=TimeUnit.quarters
    )
    assert restored.to_dict() == written
    assert [element.role for element in restored.elements].count("jump") == 4


def test_unfolded_measures_table_cannot_form_a_timeline_measure_map() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises(
            ValueError, match="Measure IDs must be unique within a MeasureMap"
        ):
            Ms3Loader.from_file(RONDEAU_UNFOLDED).create_timeline()
    assert recorded == []
