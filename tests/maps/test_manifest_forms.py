"""The TimeToShare forms of conversion maps (see README, *Manifest forms*).

Every library map is written as one of the format's ten abstract types. A
written map is proved faithful functionally: the map read back converts every
probe exactly as the written map does, value and Python type alike.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from timetoalign import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    Timeline,
)
from timetoalign.maps import (
    BeatInMeasureMap,
    ChainMap,
    CombinationMap,
    ConstantMap,
    ConversionMap,
    FloatingMeasureMap,
    FloorMap,
    InterpolationMap,
    IntervalToConstantMap,
    LinearMap,
    MeasureBeatMap,
    MeasureNumberMap,
    MeasureRecordMap,
    MetricalPositionMap,
    MetricMap,
    PiecewiseMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
    QuartersToTicks,
    RotationMap,
    SamplesToSeconds,
    ScalarMap,
    SecondsToSamples,
    ShiftMap,
    TableMap,
    TicksToQuarters,
)
from timetoalign.timelines import MeasureMap

F = Fraction

# region Helpers


def _bars() -> MeasureMap:
    """An anacrusis (offset 3 in 4/4), a 4/4 bar, a 3/4 bar, an irregular bar."""
    return MeasureMap.from_rows(
        [
            {
                "count": 1,
                "number": 0,
                "time_signature": "4/4",
                "actual_length": 1,
                "offset_within_measure": 3,
            },
            {"count": 2, "number": 1},
            {"count": 3, "number": 2, "time_signature": "3/4"},
            {"count": 4, "number": 3, "actual_length": 2},
        ],
        keep_unknown_keys=True,
    )


BAR_PROBES: list[Any] = [
    -1,
    0,
    F(1, 2),
    1,
    3,
    F(9, 2),
    5,
    6,
    F(13, 2),
    7,
    8,
    9,
    9.5,
    10,
    11,
    12,
]
"""Before, on and inside every bar of :func:`_bars`, and after its end."""


def _outcome(cmap: ConversionMap[Any], probe: Any) -> Any:
    """What *cmap* answers at *probe*, with its Python type, or the error class."""
    try:
        result = cmap(probe)
    except ValueError:
        return ValueError
    if isinstance(result, dict):
        return {key: (value, type(value)) for key, value in result.items()}
    if isinstance(result, float) and math.isnan(result):
        return "nan"
    return (result, type(result))


def _assert_converts_alike(
    written: ConversionMap[Any], read: ConversionMap[Any], probes: list[Any]
) -> None:
    """The read map answers every probe as the written one, value and type."""
    assert [_outcome(read, p) for p in probes] == [_outcome(written, p) for p in probes]


_TIMELINE_FOR_UNIT = {
    None: lambda: ContinuousLogicalTimeline(length=F(12), uid="host"),
    "quarters": lambda: ContinuousLogicalTimeline(length=F(12), uid="host"),
    "ticks": lambda: DiscreteLogicalTimeline(length=960, uid="host"),
    "samples": lambda: DiscretePhysicalTimeline(length=96000, uid="host"),
    "seconds": lambda: ContinuousPhysicalTimeline(length=12.0, uid="host"),
}


def _through_manifest(cmap: ConversionMap[Any]) -> ConversionMap[Any]:
    """Attach *cmap* to a fresh timeline, write it, read it back, return the map."""
    unit = cmap.source_unit.value if cmap.source_unit is not None else None
    timeline: Timeline = _TIMELINE_FOR_UNIT[unit]()
    timeline.add_conversion_map(cmap)
    data = json.loads(json.dumps(timeline.to_dict()))
    read = Timeline.from_dict(data).get_conversion_map(cmap.id)
    assert read is not None
    return read


# endregion

# region Each abstract type is itself


def _abstract_maps() -> list[tuple[ConversionMap[Any], list[Any]]]:
    return [
        (
            LinearMap(scalar=F(3, 2), offset=F(1, 3), uid="lin"),
            [-4, 0, 1, F(1, 3), 2.5],
        ),
        (
            TableMap(
                x_values=[0, F(4), 8],
                y_values=[0.0, 2.0, 5.0],
                extrapolate="extrapolate",
                uid="tab",
            ),
            [-2, 0, 2, 4, 6, 8, 10],
        ),
        (
            TableMap(
                x_values=[0, 4, 8],
                y_values=[F(1, 3), 2, 0.5],
                kind="previous",
                extrapolate="constant",
                uid="steps",
            ),
            [-1, 0, 3, 4, 7.5, 8, 9],
        ),
        (ConstantMap(value="page.png", uid="page"), [-1, 0, 7.25]),
        (ConstantMap(value=F(3, 4), uid="three_quarters"), [0, 3]),
        (
            PiecewiseMap(
                breaks=[0, 4, 8],
                maps=[
                    LinearMap(scalar=2, uid="double"),
                    LinearMap(offset=1, uid="plus"),
                ],
                uid="pieces",
            ),
            [-1, 0, 2, 4, 6, 8],
        ),
        (
            ChainMap(
                [
                    LinearMap(scalar=F(1, 2), uid="half"),
                    LinearMap(offset=3, uid="three"),
                ],
                uid="chain",
            ),
            [0, 1, F(5, 3), 2.5],
        ),
        (
            CombinationMap(
                {
                    "x": LinearMap(scalar=2, uid="x"),
                    "label": ConstantMap(value="a", uid="a"),
                },
                uid="combo",
            ),
            [0, F(1, 3), 4.5],
        ),
    ]


@pytest.mark.parametrize(
    ("cmap", "probes"), _abstract_maps(), ids=lambda value: getattr(value, "id", "")
)
def test_each_general_type_reads_back_as_itself(
    cmap: ConversionMap[Any], probes: list[Any]
) -> None:
    for minimal in (False, True):
        data = json.loads(json.dumps(cmap.to_dict(minimal=minimal)))
        read = ConversionMap.from_dict(data)
        assert type(read) is type(cmap)
        assert read.to_dict() == cmap.to_dict()
        assert read.to_dict(minimal=True) == cmap.to_dict(minimal=True)
        assert read == cmap
        _assert_converts_alike(cmap, read, probes)


@pytest.mark.parametrize(
    "view", [MeasureRecordMap, MeasureNumberMap, FloatingMeasureMap, MeasureBeatMap]
)
def test_each_view_reads_back_over_the_measure_map_it_names(view: type) -> None:
    bars = _bars()
    cmap = view(bars, uid="view")
    data = json.loads(json.dumps(cmap.to_dict()))
    assert data["measure_map"] == bars.id == "measuremap_1"
    read = ConversionMap.from_dict(data, measure_maps={bars.id: bars})
    assert type(read) is view
    assert read.measure_map is bars
    assert read.to_dict() == cmap.to_dict()
    assert read == cmap
    _assert_converts_alike(cmap, read, BAR_PROBES)


# endregion

# region Lowering


@pytest.mark.parametrize(
    ("cmap", "complete", "minimal"),
    [
        (
            ScalarMap(
                F(2, 3), source_unit="quarters", target_unit="quarters", uid="affine"
            ),
            {"scalar": [2, 3], "offset": 0},
            {"scalar": [2, 3]},
        ),
        (
            ShiftMap(
                offset=F(1, 2),
                source_unit="quarters",
                target_unit="quarters",
                uid="affine",
            ),
            {"scalar": 1, "offset": [1, 2]},
            {"offset": [1, 2]},
        ),
        (
            LinearMap(
                scalar=1.0, source_unit="quarters", target_unit="quarters", uid="affine"
            ),
            {"scalar": 1.0, "offset": 0},
            {"scalar": 1.0},
        ),
    ],
    ids=["ScalarMap", "ShiftMap", "float_scalar_is_no_default"],
)
def test_affine_maps_write_a_linear_map(
    cmap: ConversionMap[Any], complete: dict[str, Any], minimal: dict[str, Any]
) -> None:
    base = {
        "type": "LinearMap",
        "id": "affine",
        "source_unit": "quarters",
        "target_unit": "quarters",
    }
    assert cmap.to_dict() == {**base, "name": None, **complete}
    assert cmap.to_dict(minimal=True) == {**base, **minimal}
    read = _through_manifest(cmap)
    assert type(read) is LinearMap
    _assert_converts_alike(cmap, read, [0, 1, F(1, 3), 2.5, -7])


@pytest.mark.parametrize(
    ("cmap", "units", "scalar", "probes"),
    [
        (
            TicksToQuarters(ppq=480, uid="conversion"),
            ("ticks", "quarters"),
            [1, 480],
            [0, 240, 480, 961],
        ),
        (
            QuartersToTicks(ppq=480, uid="conversion"),
            ("quarters", "ticks"),
            480,
            [0, F(1, 3), 2, 2.5],
        ),
        (
            SamplesToSeconds(sample_rate=48000, uid="conversion"),
            ("samples", "seconds"),
            [1, 48000],
            [0, 24000, 48001],
        ),
        (
            SecondsToSamples(sample_rate=48000, uid="conversion"),
            ("seconds", "samples"),
            48000,
            [0.0, 0.5, 1.25],
        ),
    ],
    ids=["TicksToQuarters", "QuartersToTicks", "SamplesToSeconds", "SecondsToSamples"],
)
def test_unit_conversions_write_a_linear_map_without_their_parameters(
    cmap: ConversionMap[Any], units: tuple[str, str], scalar: Any, probes: list[Any]
) -> None:
    assert cmap.to_dict(minimal=True) == {
        "type": "LinearMap",
        "id": "conversion",
        "source_unit": units[0],
        "target_unit": units[1],
        "scalar": scalar,
    }
    read = _through_manifest(cmap)
    assert type(read) is LinearMap
    assert read != cmap
    _assert_converts_alike(cmap, read, probes)


def test_an_interpolation_map_writes_a_linear_table_map() -> None:
    cmap = InterpolationMap(
        source_coords=np.array([0.0, 8.0, 16.0]),
        target_coords=np.array([0.0, 4.25, 9.0]),
        source_id="score",
        target_id="audio",
        source_unit="quarters",
        target_unit="seconds",
        uid="warp",
    )
    assert cmap.to_dict() == {
        "type": "TableMap",
        "id": "warp",
        "name": None,
        "source_unit": "quarters",
        "target_unit": "seconds",
        "x_values": [0.0, 8.0, 16.0],
        "y_values": [0.0, 4.25, 9.0],
        "kind": "linear",
        "extrapolate": "linear",
    }
    read = _through_manifest(cmap)
    assert type(read) is TableMap
    _assert_converts_alike(cmap, read, [-4.0, 0.0, 4.0, 8.0, 12.0, 16.0, 20.0])


@pytest.mark.parametrize("policy", ["error", "nan"])
def test_error_and_nan_extrapolation_both_write_none(policy: str) -> None:
    cmap = TableMap(x_values=[0, 4], y_values=[0.0, 2.0], extrapolate=policy, uid="t")
    assert cmap.to_dict()["extrapolate"] == "none"
    assert "extrapolate" not in cmap.to_dict(minimal=True)
    read = ConversionMap.from_dict(cmap.to_dict())
    _assert_converts_alike(cmap, read, [0, 1, 4])
    assert _outcome(cmap, 5) == (ValueError if policy == "error" else "nan")
    assert _outcome(read, 5) is ValueError


def test_constant_extrapolation_writes_constant() -> None:
    cmap = TableMap(
        x_values=[0, 4], y_values=[0.0, 2.0], extrapolate="constant", uid="t"
    )
    assert cmap.to_dict()["extrapolate"] == "constant"
    _assert_converts_alike(
        cmap, ConversionMap.from_dict(cmap.to_dict()), [-1, 0, 2, 4, 9]
    )


def test_a_step_table_answers_its_anchor_values_as_stored() -> None:
    cmap = TableMap(x_values=[0, 4], y_values=[F(1, 3), 2], kind="previous")
    assert _outcome(cmap, 1) == (F(1, 3), Fraction)
    assert _outcome(cmap, 4) == (2, int)


def test_a_step_function_of_numbers_writes_a_clamping_step_table() -> None:
    cmap = IntervalToConstantMap(
        boundaries=[0, 4, 8], values=[1, F(5, 2), 7], uid="steps"
    )
    assert cmap.to_dict(minimal=True) == {
        "type": "TableMap",
        "id": "steps",
        "source_unit": None,
        "target_unit": None,
        "x_values": [0, 4, 8],
        "y_values": [1, [5, 2], 7],
        "kind": "previous",
        "extrapolate": "constant",
    }
    read = _through_manifest(cmap)
    assert type(read) is TableMap
    _assert_converts_alike(cmap, read, [-1, 0, 2, 4, 6, 8, 10])


@pytest.mark.parametrize("value", [5, "only"])
def test_a_step_function_of_one_boundary_writes_a_constant(value: Any) -> None:
    cmap = IntervalToConstantMap(boundaries=[0], values=[value], uid="one")
    assert cmap.to_dict(minimal=True) == {
        "type": "ConstantMap",
        "id": "one",
        "source_unit": None,
        "target_unit": None,
        "value": value,
    }
    read = _through_manifest(cmap)
    assert type(read) is ConstantMap
    _assert_converts_alike(cmap, read, [-1, 0, 3, 100])


def test_a_step_function_with_an_end_writes_constants_piecewise() -> None:
    cmap = IntervalToConstantMap(
        boundaries=[0, 4],
        values=["a.png", "b.png"],
        end=8,
        source_unit="quarters",
        uid="pages",
    )
    assert cmap.to_dict(minimal=True) == {
        "type": "PiecewiseMap",
        "id": "pages",
        "source_unit": "quarters",
        "target_unit": None,
        "breaks": [0, 4, 8],
        "maps": [
            {
                "type": "ConstantMap",
                "id": "cmap_1",
                "source_unit": "quarters",
                "target_unit": None,
                "value": "a.png",
            },
            {
                "type": "ConstantMap",
                "id": "cmap_2",
                "source_unit": "quarters",
                "target_unit": None,
                "value": "b.png",
            },
        ],
    }
    read = _through_manifest(cmap)
    assert type(read) is PiecewiseMap
    _assert_converts_alike(cmap, read, [-1, 0, 2, 4, 6, 8, 9])
    assert _outcome(cmap, -1) is ValueError
    assert _outcome(cmap, 8) is ValueError


def test_a_combination_writes_its_maps_by_name_without_names() -> None:
    cmap = CombinationMap(
        [("y", LinearMap(scalar=3, uid="y")), ("x", LinearMap(offset=1, uid="x"))],
        uid="xy",
    )
    data = cmap.to_dict()
    assert "names" not in data
    assert list(data["maps"]) == ["y", "x"]
    read = ConversionMap.from_dict(json.loads(json.dumps(data)))
    assert read.names == ["y", "x"]
    reordered = data | {"maps": {"x": data["maps"]["x"], "y": data["maps"]["y"]}}
    assert ConversionMap.from_dict(reordered).names == ["x", "y"]
    _assert_converts_alike(cmap, read, [0, F(1, 2), 2.5])


# endregion

# region Measure-derived maps


def _view(data: dict[str, Any], view: str) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != "measure_map"} | {
        "type": view
    }


def test_a_floating_measure_map_names_the_measure_map_of_its_knots() -> None:
    cmap = QuartersToFloatingMeasures(
        x_values=[0, 4, 7], y_values=[1.0, 2.0, 3.0], uid="floating"
    )
    assert cmap.to_dict() == {
        "type": "FloatingMeasureMap",
        "id": "floating",
        "name": None,
        "source_unit": "quarters",
        "target_unit": "floating_measures",
        "measure_map": "measuremap_1",
    }
    read = _through_manifest(cmap)
    assert type(read) is FloatingMeasureMap
    assert [(m.qstamp, m.actual_length) for m in read.measure_map] == [(0, 4), (4, 3)]
    _assert_converts_alike(cmap, read, [-2, 0, 2, 4, F(11, 2), 7, 9])


def test_a_floating_measure_map_starting_at_ordinal_zero_writes_an_anacrusis() -> None:
    cmap = QuartersToFloatingMeasures(
        x_values=[-1, 3, 7], y_values=[0.0, 1.0, 2.0], uid="floating"
    )
    read = _through_manifest(cmap)
    first = read.measure_map.measures[0]
    assert (first.qstamp, first.actual_length, first.nominal_length) == (0, 3, 4)
    assert first.offset_within_measure == 1
    _assert_converts_alike(cmap, read, [-2, 0, 1, 3, 5, 7, 9])


def test_from_measure_map_returns_a_view_remembering_its_measure_map() -> None:
    bars = _bars()
    cmap = QuartersToFloatingMeasures.from_measure_map(bars)
    assert type(cmap) is FloatingMeasureMap
    assert cmap.measure_map is bars


def test_measure_number_labels_read_back_as_numbers() -> None:
    cmap = QuartersToMeasureNumber(boundaries=[0, 3, 7], mns=["0", "1", "2"], uid="mn")
    assert cmap.to_dict(minimal=True) == {
        "type": "MeasureNumberMap",
        "id": "mn",
        "source_unit": "quarters",
        "target_unit": None,
        "measure_map": "measuremap_1",
    }
    read = _through_manifest(cmap)
    assert type(read) is MeasureNumberMap
    probes = [-1, 0, 2, 3, 6, 7, 20]
    assert (
        [read(p) for p in probes]
        == [int(cmap(p)) for p in probes]
        == [0, 0, 0, 1, 1, 2, 2]
    )


def _meter() -> MetricMap:
    return MetricMap(
        starts=[F(0), F(1), F(5)],
        mcs=[1, 2, 3],
        mns=["0", "1", "2"],
        lengths=[F(1), F(4), F(4)],
        uid="meter",
    )


METER_PROBES: list[Any] = [-1, 0, 0.5, 1, F(9, 2), 5, 8.75, 9, 12]


def test_a_metric_map_writes_the_numbers_of_its_measure_counts() -> None:
    cmap = _meter()
    assert cmap.to_dict(minimal=True) == {
        "type": "MeasureNumberMap",
        "id": "meter",
        "source_unit": "quarters",
        "target_unit": None,
        "measure_map": "measuremap_1",
    }
    read = _through_manifest(cmap)
    assert type(read) is MeasureNumberMap
    assert [
        (m.qstamp, m.number, m.name, m.actual_length) for m in read.measure_map
    ] == [
        (0, 1, "0", 1),
        (1, 2, "1", 4),
        (5, 3, "2", 4),
    ]
    _assert_converts_alike(cmap, read, METER_PROBES)


def test_a_beat_in_measure_map_is_the_beat_of_a_measure_beat_map() -> None:
    cmap = BeatInMeasureMap(_meter(), uid="beats")
    assert cmap.to_dict()["type"] == "MeasureBeatMap"
    read = _through_manifest(cmap)
    assert type(read) is MeasureBeatMap
    assert [(read(p)["beat"], type(read(p)["beat"])) for p in METER_PROBES] == [
        (cmap(p), type(cmap(p))) for p in METER_PROBES
    ]


def test_a_metrical_position_map_reads_back_with_the_count_as_number() -> None:
    cmap = MetricalPositionMap(_meter(), uid="positions")
    assert cmap.to_dict()["type"] == "MeasureBeatMap"
    read = _through_manifest(cmap)
    assert type(read) is MeasureBeatMap
    assert [read(p) for p in METER_PROBES] == [
        {"number": cmap(p)["mc"], "beat": cmap(p)["beat"]} for p in METER_PROBES
    ]


# endregion

# region The measure views


def test_the_views_answer_as_the_former_measure_maps() -> None:
    bars = _bars()
    qstamps = [m.qstamp for m in bars]
    numbers = [m.number for m in bars]
    lengths = [m.actual_length for m in bars]
    records, number_map = MeasureRecordMap(bars), MeasureNumberMap(bars)
    floating, beats = FloatingMeasureMap(bars), MeasureBeatMap(bars)
    meter = MetricMap(
        starts=qstamps, mcs=numbers, mns=[str(n) for n in numbers], lengths=lengths
    )
    labels = QuartersToMeasureNumber(boundaries=qstamps, mns=[str(n) for n in numbers])

    assert [records(p).id for p in BAR_PROBES] == [
        "1", "1", "1", "2", "2", "2", "3", "3", "3", "3", "4", "4", "4", "4", "4", "4",
    ]  # fmt: skip
    assert [number_map(p) for p in BAR_PROBES] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
    ]  # fmt: skip
    assert [number_map(p) for p in BAR_PROBES] == [int(labels(p)) for p in BAR_PROBES]
    assert [floating(p) for p in BAR_PROBES] == [
        0.5, 0.75, 0.875, 1.0, 1.5, 1.875, 2.0, 2.333, 2.5, 2.666,
        3.0, 3.5, 3.75, 4.0, 4.5, 5.0,
    ]  # fmt: skip
    assert [beats(p) for p in BAR_PROBES] == [
        {"number": number_map(p), "beat": meter.beat_in_measure(p)} for p in BAR_PROBES
    ]
    assert [beats(p)["beat"] for p in BAR_PROBES] == [
        0, 1, F(3, 2), 1, 3, F(9, 2), 1, 2, F(5, 2), 3, 1, 2, F(5, 2), 3, 4, 5,
    ]  # fmt: skip


# endregion

# region Refusals


@pytest.mark.parametrize(
    ("cmap", "reason"),
    [
        (
            TableMap(x_values=[0, 4], y_values=[0, 1], kind="nearest", uid="m"),
            "the format defines no TableMap of kind 'nearest'",
        ),
        (
            TableMap(x_values=[0, 4], y_values=[0, 1], kind="next", uid="m"),
            "the format defines no TableMap of kind 'next'",
        ),
        (FloorMap(divisor=4, uid="m"), "its type has no TimeToShare form"),
        (RotationMap(period=4, uid="m"), "its type has no TimeToShare form"),
        (
            IntervalToConstantMap(boundaries=[0, 4], values=["a", "b"], uid="m"),
            "a step function of labels needs an end for its last interval (give end=)",
        ),
        (
            QuartersToMeasureNumber(boundaries=[0, 4], mns=["19", "19a"], uid="m"),
            "its label '19a' is not a measure number",
        ),
        (
            QuartersToMeasureNumber(boundaries=[0, 4], mns=["01", "2"], uid="m"),
            "its label '01' is not a measure number",
        ),
        (
            QuartersToMeasureNumber(boundaries=[2, 4], mns=["1", "2"], uid="m"),
            "its first measure starts at 2, not at 0",
        ),
        (
            MetricMap(
                starts=[F(1), F(5)],
                mcs=[1, 2],
                mns=["1", "2"],
                lengths=[F(4)] * 2,
                uid="m",
            ),
            "its measure 1 starts at 1, not at 0",
        ),
        (
            MetricMap(
                starts=[F(0), F(5)],
                mcs=[1, 2],
                mns=["1", "2"],
                lengths=[F(4)] * 2,
                uid="m",
            ),
            "its measure 2 starts at 5, not at 4",
        ),
        (
            QuartersToFloatingMeasures(x_values=[0, 4], y_values=[1.0, 3.0], uid="m"),
            "its ordinals do not count measures up by one from 0 or 1",
        ),
        (
            QuartersToFloatingMeasures(x_values=[2, 6], y_values=[1.0, 2.0], uid="m"),
            "its first downbeat lies at 2, not at 0",
        ),
        (
            QuartersToFloatingMeasures(x_values=[1, 5], y_values=[0.0, 1.0], uid="m"),
            "its anacrusis bar [1, 5) does not contain 0",
        ),
    ],
    ids=[
        "nearest",
        "next",
        "FloorMap",
        "RotationMap",
        "labels_without_end",
        "volta_label",
        "padded_label",
        "numbers_not_from_zero",
        "meter_not_from_zero",
        "meter_gap",
        "ordinal_gap",
        "late_downbeat",
        "anacrusis_after_zero",
    ],
)
def test_what_has_no_form_is_refused(cmap: ConversionMap[Any], reason: str) -> None:
    message = f"{type(cmap).__name__} 'm' cannot be written to a TimeToShare manifest: {reason}"
    with pytest.raises(ValueError) as refused:
        cmap.to_dict()
    assert str(refused.value) == message
    timeline = ContinuousLogicalTimeline(length=F(12), uid="host")
    timeline.add_conversion_map(cmap)
    with pytest.raises(ValueError) as refused:
        timeline.to_dict()
    assert str(refused.value) == message


def test_reading_refuses_a_former_type_name() -> None:
    data = {
        "type": "ScalarMap",
        "id": "affine",
        "source_unit": None,
        "target_unit": None,
    }
    with pytest.raises(ValueError) as refused:
        ConversionMap.from_dict(data)
    assert str(refused.value) == (
        "Unknown map type: 'ScalarMap'. Known types: ChainMap, CombinationMap, "
        "ConstantMap, FloatingMeasureMap, LinearMap, MeasureBeatMap, MeasureNumberMap, "
        "MeasureRecordMap, PiecewiseMap, TableMap"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"kind": "nearest"}, "a TableMap has an unknown kind 'nearest'"),
        ({"extrapolate": "nan"}, "a TableMap has an unknown extrapolate 'nan'"),
    ],
)
def test_reading_refuses_a_table_the_format_does_not_define(
    change: dict[str, Any], message: str
) -> None:
    data = TableMap(x_values=[0, 4], y_values=[0, 1], uid="t").to_dict() | change
    with pytest.raises(ValueError) as refused:
        ConversionMap.from_dict(data)
    assert str(refused.value) == message


def test_reading_refuses_a_view_naming_no_measure_map() -> None:
    data = MeasureNumberMap(_bars(), uid="n").to_dict() | {"measure_map": "bars"}
    with pytest.raises(ValueError) as refused:
        ConversionMap.from_dict(data)
    assert str(refused.value) == (
        "a MeasureNumberMap names the measure map 'bars', which the manifest's "
        "measure_maps do not hold. Available measure maps: []"
    )


def test_a_key_without_default_is_required() -> None:
    data = TableMap(x_values=[0, 4], y_values=[0, 1], uid="t").to_dict()
    del data["x_values"]
    with pytest.raises(ValueError) as refused:
        ConversionMap.from_dict(data)
    assert str(refused.value) == "a TableMap lacks the key(s) ['x_values']"
    minimal = {"type": "LinearMap", "id": "l", "source_unit": None, "target_unit": None}
    read = ConversionMap.from_dict(minimal)
    assert (read.scalar, read.offset) == (1, 0)
    assert (type(read.scalar), type(read.offset)) == (int, int)


# endregion

# region Equality


def test_a_map_and_the_type_it_writes_are_not_equal() -> None:
    scalar = ScalarMap(2, uid="affine")
    linear = LinearMap(scalar=2, offset=0, uid="affine")
    assert scalar.to_dict() == linear.to_dict()
    assert scalar != linear
    assert ConversionMap.from_dict(linear.to_dict()) == linear


# endregion
