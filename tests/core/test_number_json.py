"""Tests for the canonical JSON number and coordinate forms.

Three contracts are exercised here:

1. the codec itself (``number_to_json`` / ``number_from_json`` /
   ``is_fraction_json``);
2. the fixpoint guarantee — ``from_dict(json.loads(json.dumps(x.to_dict())))``
   reproduces ``x.to_dict()`` — for timelines and maps, including exact
   ``Fraction`` recovery and conversion-map name round-trips;
3. JSON-safety of every ``to_dict`` in the library that a caller is
   expected to persist.
"""

from __future__ import annotations

import json
from fractions import Fraction
from typing import Any, Callable

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.alignment.claims import (
    Agent,
    AlignmentAnchor,
    MatchClaim,
    MatchMetadata,
)
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.matchline import MatchLine
from timetoalign.alignment.warpmap import WarpMap
from timetoalign.core import (
    ActivationCondition,
    AgentType,
    Coordinate,
    FlowControlElement,
    IdCoordinate,
    Measure,
    Tempo,
    TimeUnit,
)
from timetoalign.core.retrieval import coordinate_from_dict, coordinate_to_dict
from timetoalign.core.time import (
    is_fraction_json,
    number_from_json,
    number_to_json,
    struct_to_rational,
)
from timetoalign.maps import (
    ChainMap,
    CombinationMap,
    ConstantMap,
    ConversionMap,
    FloorMap,
    InterpolationMap,
    LinearMap,
    PiecewiseMap,
    QuartersToTicks,
    RotationMap,
    SamplesToSeconds,
    ScalarMap,
    SecondsToSamples,
    ShiftMap,
    TableMap,
    TicksToQuarters,
)
from timetoalign.maps.interval import (
    IntervalToConstantMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
)
from timetoalign.maps.meter import BeatInMeasureMap, MetricalPositionMap, MetricMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    MeasureMap,
    MetricHierarchy,
    TempoEntry,
    TempoMap,
    Timeline,
)
from timetoalign.timelines.flow.measures import MeasureUnit
from timetoalign.timelines.flow.sections import AtomicSection, PlaythroughSection
from timetoalign.timelines.flowcontrol import Break, Jump

# region Fixtures


def _fraction_timeline() -> Timeline:
    """A quarters timeline whose length, offset, and events are exact."""
    timeline = Timeline(
        length=Fraction(10, 3),
        unit=TimeUnit.quarters,
        number_type="fraction",
        uid="parent",
    )
    timeline.add_events([{"instant": Fraction(1, 3), "event_type": "Note"}])
    child = Timeline(
        length=Fraction(1, 1),
        unit=TimeUnit.quarters,
        number_type="fraction",
        uid="child",
    )
    timeline.add_child(child, offset=Fraction(5, 3))
    return timeline


def _float_timeline() -> Timeline:
    """A seconds timeline with inexact coordinates throughout."""
    timeline = Timeline(length=10.0, unit=TimeUnit.seconds, uid="clock")
    timeline.add_events([{"start": 0.1, "end": 0.3, "event_type": "Note"}])
    timeline.add_child(
        Timeline(length=2.0, unit=TimeUnit.seconds, uid="tick"), offset=4.5
    )
    return timeline


def _structured_timeline() -> Timeline:
    """A timeline carrying every timeline-native structural object."""
    timeline = Timeline(
        length=Fraction(4),
        unit=TimeUnit.quarters,
        number_type="fraction",
        uid="structure",
    )
    measure_map = MeasureMap([Measure(actual_length=Fraction(4, 3)) for _ in range(3)])
    timeline.add_measure_map(measure_map)
    timeline.add_metric_hierarchy(
        MetricHierarchy(
            measure_map=measure_map,
            tempo_maps=(
                TempoMap(
                    kind="tempo_indication",
                    entries=(
                        TempoEntry(
                            at=Coordinate(Fraction(0), "quarters"),
                            tempo=Tempo(bpm=Fraction(120), text="triple"),
                        ),
                    ),
                    provenance={},
                ),
            ),
        )
    )
    timeline.create_region("middle", Fraction(1, 3), Fraction(2, 3))
    timeline.flow_control.add_break(
        Break(
            Coordinate(Fraction(2, 3), TimeUnit.quarters),
            control_type=FlowControlElement.section_break,
            condition=ActivationCondition.always,
        )
    )
    timeline.flow_control.add_jump(
        Jump(
            Coordinate(Fraction(4), TimeUnit.quarters),
            Coordinate(Fraction(1, 3), TimeUnit.quarters),
            control_type=FlowControlElement.repeat_end,
            condition=ActivationCondition.first_n,
        )
    )
    timeline.flow_control.add_marker(
        "segno", Coordinate(Fraction(1, 3), TimeUnit.quarters)
    )
    return timeline


def _meter_map() -> MetricMap:
    return MetricMap.from_uniform(4, Fraction(4, 1))


def _conversion_maps() -> list[ConversionMap[Any]]:
    """One live instance of every registered ConversionMap subclass."""
    meter = _meter_map()
    return [
        LinearMap(scalar=Fraction(1, 3), offset=Fraction(5, 3)),
        ScalarMap(scalar=Fraction(2, 3)),
        ShiftMap(offset=Fraction(1, 7)),
        ConstantMap(value=Fraction(3, 4)),
        TableMap(x_values=[Fraction(0), Fraction(1, 3)], y_values=[0.0, 1.0]),
        IntervalToConstantMap(
            boundaries=[Fraction(0), Fraction(1, 3)], values=["a", "b"]
        ),
        QuartersToMeasureNumber(boundaries=[Fraction(0), Fraction(4)], mns=["1", "2"]),
        QuartersToFloatingMeasures(
            x_values=[Fraction(0), Fraction(4)], y_values=[1.0, 2.0]
        ),
        meter,
        BeatInMeasureMap(meter),
        MetricalPositionMap(meter),
        ChainMap([ScalarMap(scalar=2.0), ShiftMap(offset=1.0)]),
        PiecewiseMap(
            breaks=[Fraction(0), Fraction(1, 2), Fraction(1)],
            maps=[ScalarMap(scalar=1.0), ScalarMap(scalar=2.0)],
        ),
        CombinationMap([("a", ScalarMap(scalar=1.0))]),
        RotationMap(period=4.0),
        FloorMap(divisor=4.0),
        TicksToQuarters(),
        QuartersToTicks(),
        SamplesToSeconds(),
        SecondsToSamples(),
        InterpolationMap(
            source_coords=[0.0, 1.0],
            target_coords=[0.0, 2.0],
            source_id="a",
            target_id="b",
        ),
    ]


def _claim() -> MatchClaim:
    """A synchronous claim whose A-side coordinate is exact."""
    return MatchClaim.from_events(
        event_a={"id": "e1", "name": "Note C4", "start": Fraction(1, 3)},
        tl_a_id="score",
        event_b={"id": "e2", "name": "Note C4", "start": 45.5},
        tl_b_id="audio",
        unit_a=TimeUnit.quarters,
        unit_b=TimeUnit.seconds,
    )


def _measure_unit() -> MeasureUnit:
    return MeasureUnit(
        mc=1,
        mn="1",
        duration_qb=Fraction(4),
        next=(2,),
        timesig="4/4",
        timesig_duration_qb=Fraction(4),
    )


# endregion

# region Codec


class TestNumberJson:
    """The schema's number forms are encoded and decoded exactly."""

    def test_fraction_encodes_as_a_pair(self) -> None:
        assert number_to_json(Fraction(10, 3)) == [10, 3]

    def test_float_encodes_as_a_json_number(self) -> None:
        assert number_to_json(2.5) == 2.5

    def test_integer_encodes_as_a_json_integer(self) -> None:
        result = number_to_json(7)
        assert result == 7
        assert type(result) is int

    def test_fraction_decodes_to_a_fraction(self) -> None:
        decoded = number_from_json([1, 3])
        assert decoded == Fraction(1, 3)
        assert isinstance(decoded, Fraction)

    def test_float_decodes_to_a_float(self) -> None:
        decoded = number_from_json(2.5)
        assert decoded == 2.5
        assert isinstance(decoded, float)

    @pytest.mark.parametrize("value", [Fraction(-7, 9), Fraction(0, 1), Fraction(5, 1)])
    def test_round_trip_is_exact(self, value: Fraction) -> None:
        assert number_from_json(number_to_json(value)) == value

    def test_dictionary_is_json_native(self) -> None:
        data = number_to_json(Fraction(10, 3))
        assert json.loads(json.dumps(data)) == data

    def test_boolean_fraction_components_are_refused(self) -> None:
        with pytest.raises(ValueError):
            number_from_json([True, 1])

    def test_is_fraction_json_recognises_the_shape(self) -> None:
        assert is_fraction_json([1, 2])
        assert not is_fraction_json([1, 0])
        assert not is_fraction_json([1, 2, 3])
        assert not is_fraction_json([True, 1])

    def test_non_numeric_input_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            number_to_json("3/4")  # type: ignore[arg-type]

    def test_string_encoding_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            number_from_json("3/4")

    def test_zero_denominator_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            number_from_json([1, 0])

    def test_negative_denominator_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            number_from_json([1, -3])

    def test_wrong_length_fraction_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            number_from_json([1, 2, 3])

    def test_coordinate_restores_its_declared_number_type(self) -> None:
        data = {"unit": "quarters", "number_type": "fraction", "value": [1, 3]}
        restored = coordinate_from_dict(data)
        assert coordinate_to_dict(restored) == data
        assert type(restored.value) is Fraction

    def test_timeline_length_axis_mismatch_is_refused(self) -> None:
        payload = ContinuousLogicalTimeline(length=12, unit="quarters").to_dict()
        payload["length"]["unit"] = "seconds"
        with pytest.raises(ValueError, match="length axis"):
            ContinuousLogicalTimeline.from_dict(payload)

    def test_timeline_offset_axis_mismatch_is_refused(self) -> None:
        parent = ContinuousLogicalTimeline(length=12, unit="quarters")
        child = ContinuousLogicalTimeline(length=4, unit="quarters", uid="child")
        parent.add_child(child, offset=Fraction(1, 3))
        payload = parent.to_dict()
        payload["children"]["child"]["offset"]["number_type"] = "int"
        with pytest.raises(ValueError, match="timeline .*child offset"):
            ContinuousLogicalTimeline.from_dict(payload)


# endregion

# region Timelines


class TestTimelineDictForm:
    """``length``, child ``offset``s, and event coordinates share one shape."""

    def test_fraction_length_is_a_coordinate(self) -> None:
        data = _fraction_timeline().to_dict()
        assert data["length"] == {
            "unit": "quarters",
            "number_type": "fraction",
            "value": [10, 3],
        }

    def test_fraction_offset_is_a_coordinate(self) -> None:
        data = _fraction_timeline().to_dict()
        offset = data["children"]["child"]["offset"]
        assert offset == {
            "unit": "quarters",
            "number_type": "fraction",
            "value": [5, 3],
        }

    def test_float_length_is_a_coordinate(self) -> None:
        data = _float_timeline().to_dict()
        assert data["length"] == {
            "unit": "seconds",
            "number_type": "float",
            "value": 10.0,
        }

    def test_event_coordinate_survives_json_exactly(self) -> None:
        restored = Timeline.from_dict(
            json.loads(json.dumps(_fraction_timeline().to_dict(events=True)))
        )
        coordinate = struct_to_rational(list(restored.events)[0]["start"])
        assert coordinate == Fraction(1, 3)
        assert isinstance(coordinate, Fraction)

    def test_length_and_offset_survive_json_exactly(self) -> None:
        restored = Timeline.from_dict(
            json.loads(json.dumps(_fraction_timeline().to_dict()))
        )
        assert restored.length.value == Fraction(10, 3)
        assert restored.get_child_offset("child").value == Fraction(5, 3)

    @pytest.mark.parametrize(
        "factory",
        [_fraction_timeline, _float_timeline, _structured_timeline],
        ids=["fraction", "float", "structure"],
    )
    def test_to_dict_is_a_json_fixpoint(self, factory: Callable[[], Timeline]) -> None:
        data = factory().to_dict(events=True, external_references=True)
        restored = Timeline.from_dict(json.loads(json.dumps(data)))
        assert restored.to_dict(events=True, external_references=True) == data


# endregion

# region Maps


class TestMapDictForm:
    """Every map serializes rationals and its name the same way."""

    @pytest.mark.parametrize("cmap", _conversion_maps(), ids=lambda m: type(m).__name__)
    def test_custom_name_round_trips(self, cmap: ConversionMap[Any]) -> None:
        data = dict(cmap.to_dict(), name="my-map")
        assert ConversionMap.from_dict(json.loads(json.dumps(data))).name == "my-map"

    @pytest.mark.parametrize("cmap", _conversion_maps(), ids=lambda m: type(m).__name__)
    def test_name_is_always_emitted(self, cmap: ConversionMap[Any]) -> None:
        assert cmap.to_dict()["name"] == cmap.name

    def test_linear_map_keeps_exact_fractions(self) -> None:
        data = LinearMap(scalar=Fraction(1, 3), offset=Fraction(5, 3)).to_dict()
        assert data["scalar"] == [1, 3]
        restored = LinearMap.from_dict(json.loads(json.dumps(data)))
        assert restored.scalar == Fraction(1, 3)
        assert restored.offset == Fraction(5, 3)

    def test_constant_map_keeps_an_exact_fraction(self) -> None:
        cmap = ConstantMap(value=Fraction(3, 4), name="tuplet")
        data = json.loads(json.dumps(cmap.to_dict()))
        restored = ConstantMap.from_dict(data)
        assert restored.value == Fraction(3, 4)
        assert isinstance(restored.value, Fraction)
        assert restored.name == "tuplet"

    def test_constant_map_passes_a_label_through(self) -> None:
        cmap = ConstantMap(value="page1.jpeg", name="filename")
        restored = ConstantMap.from_dict(json.loads(json.dumps(cmap.to_dict())))
        assert restored.value == "page1.jpeg"

    def test_table_map_keeps_exact_fractions(self) -> None:
        table = TableMap(x_values=[Fraction(0), Fraction(1, 3)], y_values=[0.0, 1.0])
        restored = TableMap.from_dict(json.loads(json.dumps(table.to_dict())))
        assert restored.to_dict()["x_values"][1] == [1, 3]

    def test_metric_map_keeps_exact_fractions(self) -> None:
        data = json.loads(json.dumps(_meter_map().to_dict()))
        restored = MetricMap.from_dict(data)
        assert restored.to_dict() == data


# endregion

# region JSON safety


def _stamp_cases() -> list[tuple[str, dict[str, Any]]]:
    """``to_dict`` payloads of the stamp family, keyed by a readable id."""
    timeline = _float_timeline()
    bundle = AlignmentBundle(id="bundle")
    bundle.add_timeline(timeline, uid="clock", as_group="clock-group")
    matchstamp = bundle.get_matchstamp_at(4.5, "clock")
    return [
        ("TimeStamp", timeline.get_timestamp(4.5).to_dict()),
        ("MatchStamp.flat", matchstamp.to_dict(format="flat")),
        ("MatchStamp.prefix", matchstamp.to_dict(format="prefix")),
        ("MatchStamp.nested", matchstamp.to_dict(format="nested")),
        ("MatchStamp.graph", matchstamp.to_dict(format="graph")),
    ]


def _json_safety_cases() -> list[tuple[str, dict[str, Any]]]:
    """Every ``to_dict`` payload a caller is expected to persist."""
    claim = _claim()
    anchor = claim.start_anchor
    assert anchor is not None
    agent = Agent(name="dtw", type=AgentType.software, identifier="v2")
    graph = MatchGraph([claim])
    cases: list[tuple[str, dict[str, Any]]] = [
        (
            "Timeline.fraction",
            _fraction_timeline().to_dict(events=True, external_references=True),
        ),
        (
            "Timeline.float",
            _float_timeline().to_dict(events=True, external_references=True),
        ),
        ("Agent", agent.to_dict()),
        ("MatchMetadata", MatchMetadata(agent=agent, certainty=0.85).to_dict()),
        ("AlignmentAnchor", anchor.to_dict()),
        ("MatchClaim", claim.to_dict()),
        ("MatchGraph", graph.to_dict()),
        (
            "MatchLine",
            MatchLine(source_timeline_id="score", stamps=graph.get_stamps()).to_dict(),
        ),
        (
            "WarpMap",
            WarpMap.from_coordinate_pairs(
                source_timeline_id="score",
                target_timeline_id="audio",
                source_coords=[
                    IdCoordinate(0.0, TimeUnit.quarters, "score"),
                    IdCoordinate(1.0, TimeUnit.quarters, "score"),
                ],
                target_coords=[
                    IdCoordinate(0.0, TimeUnit.seconds, "audio"),
                    IdCoordinate(2.0, TimeUnit.seconds, "audio"),
                ],
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
            ).to_dict(),
        ),
        ("MeasureUnit", _measure_unit().to_dict()),
        (
            "AtomicSection",
            AtomicSection(id="A", mc_start=1, mc_end=5, to=("A", "B")).to_dict(),
        ),
        (
            "PlaythroughSection",
            PlaythroughSection(
                mc_start=1, mc_end=9, atomic_section_ids=("A", "B")
            ).to_dict(),
        ),
    ]
    cases.extend((type(cmap).__name__, cmap.to_dict()) for cmap in _conversion_maps())
    cases.extend(_stamp_cases())
    return cases


_JSON_CASES = _json_safety_cases()


@pytest.mark.parametrize(
    "payload",
    [case for _, case in _JSON_CASES],
    ids=[name for name, _ in _JSON_CASES],
)
def test_to_dict_is_json_serializable(payload: dict[str, Any]) -> None:
    """No ``to_dict`` may leak a Fraction, a Coordinate, or any other
    non-JSON object into its output."""
    assert json.loads(json.dumps(payload)) is not None


class TestClaimDictForm:
    """The claim family round-trips through its pydantic-backed codec."""

    def test_claim_coordinate_survives_json_exactly(self) -> None:
        claim = _claim()
        restored = MatchClaim.from_dict(json.loads(json.dumps(claim.to_dict())))
        assert restored.start_anchor is not None
        assert restored.start_anchor.coordinate_a.value == Fraction(1, 3)

    def test_claim_to_dict_is_a_json_fixpoint(self) -> None:
        data = _claim().to_dict()
        assert MatchClaim.from_dict(json.loads(json.dumps(data))).to_dict() == data

    def test_anchor_to_dict_is_a_json_fixpoint(self) -> None:
        anchor = AlignmentAnchor(
            timeline_a_id="score",
            coordinate_a=Coordinate(Fraction(1, 3), TimeUnit.quarters),
            timeline_b_id="audio",
            coordinate_b=Coordinate(45.5, TimeUnit.seconds),
        )
        data = anchor.to_dict()
        assert AlignmentAnchor.from_dict(json.loads(json.dumps(data))).to_dict() == data


# endregion
