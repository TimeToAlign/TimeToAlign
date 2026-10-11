"""Prove eventless manifests match the installed TimeSkeleton contract."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from timeskeleton import (
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    SchemaValidationError,
    canonical_order,
    fixture_paths,
    validate,
)

from timetoalign import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    Ms3Loader,
    Timeline,
)
from timetoalign.core import ActivationCondition, FlowControlElement
from timetoalign.core.time import number_from_json, number_to_json
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.loader.graphical.iiif import IIIFManifestLoader
from timetoalign.loader.midi.score import ScoreMidiLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.maps import (
    BeatInMeasureMap,
    ChainMap,
    CombinationMap,
    ConstantMap,
    FloorMap,
    InterpolationMap,
    IntervalToConstantMap,
    LinearMap,
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
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap
from timetoalign.timelines.flowcontrol import Break, Jump

SCORE = ensure_data("score")
MIDI = ensure_data("midi")
AUDIO = ensure_data("audio")
PARANGONAR = ensure_data("parangonar")
SUPRA = ensure_data("supra")


def _manifest(timeline: Timeline) -> dict[str, Any]:
    """Wrap one timeline in the current manifest shape."""
    return {
        "meta": {"schema_version": SCHEMA_VERSION, "generator": "timetoalign tests"},
        "timeline": timeline.to_dict(),
    }


def _with_structure(cls: type[Timeline], unit: str, length: Any) -> Timeline:
    """Build one direct timeline with every timeline-native structural member."""
    timeline = cls(length=length, unit=unit)
    timeline.add_child(cls(length=length / 3, unit=unit), length / 6)
    timeline.add_child(cls(length=length / 4, unit=unit), length / 2)
    timeline.create_region("theme", length / 6, length / 2)
    timeline.flow_control.add_break(
        Break(
            coordinate=timeline.make_coordinate(length / 2),
            control_type=FlowControlElement.fine,
            condition=ActivationCondition.always,
        )
    )
    timeline.flow_control.add_jump(
        Jump(
            from_coordinate=timeline.make_coordinate(length / 2),
            to_coordinate=timeline.make_coordinate(length / 6),
            control_type=FlowControlElement.repeat_end,
            condition=ActivationCondition.first_n,
            repeat_count=2,
        )
    )
    timeline.flow_control.add_marker("start", length / 6)
    if unit != "quarters":
        return timeline
    timeline.add_measure_map(
        MeasureMap.from_rows(
            [
                {
                    "count": 1,
                    "qstamp": 0,
                    "number": 1,
                    "name": "1",
                    "time_signature": "4/4",
                    "actual_length": 4,
                },
                {
                    "count": 2,
                    "qstamp": 4,
                    "number": 2,
                    "name": "2",
                    "time_signature": "4/4",
                    "actual_length": 4,
                },
            ]
        )
    )
    return timeline


def _conversion_map_timelines() -> list[Timeline]:
    """Attach every registered conversion-map class to direct timelines."""
    meter_for_position = MetricMap(
        starts=[Fraction(0), Fraction(4)],
        mcs=[1, 2],
        mns=["1", "2"],
        lengths=[Fraction(4), Fraction(4)],
    )
    meter_for_beat = MetricMap(
        starts=[Fraction(0), Fraction(3)],
        mcs=[1, 2],
        mns=["1", "2"],
        lengths=[Fraction(3), Fraction(3)],
    )
    timeline = _with_structure(ContinuousLogicalTimeline, "quarters", Fraction(16))
    maps = [
        MetricalPositionMap(meter_for_position),
        BeatInMeasureMap(meter_for_beat),
        ChainMap([ScalarMap(Fraction(1, 2)), ScalarMap(60)]),
        CombinationMap({"label": ConstantMap(value="verse"), "beat": RotationMap(4)}),
        ConstantMap(value="section-a"),
        FloorMap(4, base=1),
        InterpolationMap(
            source_coords=np.array([0.0, 8.0, 16.0]),
            target_coords=np.array([0.0, 4.25, 9.0]),
            source_id=timeline.id,
            target_id="seconds",
        ),
        IntervalToConstantMap(boundaries=[Fraction(0), Fraction(8)], values=["a", "b"]),
        LinearMap(scalar=Fraction(3, 2), offset=Fraction(1, 3)),
        MetricMap(starts=[Fraction(0)], mcs=[1], mns=["1"], lengths=[Fraction(4)]),
        PiecewiseMap(breaks=[Fraction(0), Fraction(8)], maps=[ShiftMap(offset=1)]),
        QuartersToFloatingMeasures(
            x_values=[Fraction(0), Fraction(4)], y_values=[1.0, 2.0]
        ),
        QuartersToMeasureNumber(boundaries=[Fraction(0)], mns=["1"]),
        QuartersToTicks(ppq=480),
        RotationMap(4, scale=0.5, base=1, offset=0.25),
        ScalarMap(Fraction(2, 3)),
        ShiftMap(offset=Fraction(1, 2)),
        TableMap(x_values=[Fraction(0), Fraction(4)], y_values=[0.0, 2.5]),
    ]
    for conversion_map in maps:
        timeline.add_conversion_map(conversion_map)
    ticks = DiscreteLogicalTimeline(length=960, unit="ticks")
    ticks.add_conversion_map(TicksToQuarters(ppq=480))
    samples = DiscretePhysicalTimeline(length=96000, unit="samples")
    samples.add_conversion_map(SamplesToSeconds(sample_rate=48000))
    seconds = ContinuousPhysicalTimeline(length=2.0, unit="seconds")
    seconds.add_conversion_map(SecondsToSamples(sample_rate=48000))
    return [timeline, ticks, samples, seconds]


def _ms3() -> Timeline:
    return (
        Ms3Loader(auto_discover=True)
        .load(
            SCORE
            / "flow_control/flow_only/out_of_the_flow_experience-flow_only.notes.tsv"
        )
        .create_timeline()
    )


def _midi() -> Timeline:
    return ScoreMidiLoader().load(MIDI / "score/beethoven_mtd.mid").create_timeline()


def _audio() -> Timeline:
    return AudioLoader().load(AUDIO / "hard_techno/Bass Kick.mp3").create_timeline()


def _parangonada() -> Timeline:
    return (
        ParangonadaLoader()
        .load(PARANGONAR / "Beethoven_Eroica_op35-cpjku")
        .create_timelines()[0]
    )


def _graphical() -> Timeline:
    return (
        IIIFManifestLoader().load(SUPRA / "image/ifff_manifest.json").create_timeline()
    )


CATALOGUE: list[tuple[str, Callable[[], list[Timeline]]]] = [
    (
        "continuous_logical",
        lambda: [_with_structure(ContinuousLogicalTimeline, "quarters", Fraction(12))],
    ),
    (
        "discrete_logical",
        lambda: [_with_structure(DiscreteLogicalTimeline, "ticks", Fraction(480))],
    ),
    (
        "continuous_physical",
        lambda: [_with_structure(ContinuousPhysicalTimeline, "seconds", 12.0)],
    ),
    (
        "discrete_physical",
        lambda: [_with_structure(DiscretePhysicalTimeline, "samples", Fraction(44100))],
    ),
    (
        "continuous_graphical",
        lambda: [_with_structure(ContinuousGraphicalTimeline, "points", 600.0)],
    ),
    (
        "discrete_graphical",
        lambda: [_with_structure(DiscreteGraphicalTimeline, "pixels", Fraction(1920))],
    ),
    ("conversion_maps", _conversion_map_timelines),
    ("ms3", lambda: [_ms3()]),
    ("midi", lambda: [_midi()]),
    ("audio", lambda: [_audio()]),
    ("parangonada", lambda: [_parangonada()]),
    ("graphical", lambda: [_graphical()]),
]


@pytest.mark.parametrize(
    ("name", "builder"), CATALOGUE, ids=[name for name, _ in CATALOGUE]
)
def test_emitted_manifests_validate(
    name: str, builder: Callable[[], list[Timeline]]
) -> None:
    """Every direct and loader-produced eventless manifest satisfies the schema.

    The library writes the canonical order the schema package defines: putting
    the manifest in that order changes nothing.
    """
    del name
    for timeline in builder():
        manifest = _manifest(timeline)
        validate(manifest)
        assert json.dumps(canonical_order(manifest)) == json.dumps(manifest)


@pytest.mark.parametrize("version", SUPPORTED_VERSIONS)
def test_bundled_fixtures_round_trip(version: str) -> None:
    """Bundled fixtures reconstruct without changing any timeline manifest.

    The comparison is by JSON text, so the canonical order of children,
    conversion maps, regions and flow control must match as well.
    """
    for path in fixture_paths(version):
        value = json.loads(path.read_text(encoding="utf-8"))
        manifests = (
            [value["timeline"]]
            if version == "0.4.0"
            else value["manifest"]["timelines"].values()
        )
        for manifest in manifests:
            expected = canonical_order({"timeline": manifest})["timeline"]
            assert json.dumps(Timeline.from_dict(manifest).to_dict()) == json.dumps(
                expected
            )


def _quarters_manifest(measure_map: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a valid manifest carrying the given row list."""
    result = _manifest(ContinuousLogicalTimeline(length=Fraction(8), unit="quarters"))
    result["timeline"]["measure_map"] = measure_map
    return result


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [{"count": 1, "time_signature": "4/4"}, {"count": 3}, {"count": 2}],
            "count 2 does not increase",
        ),
        (
            [{"count": 1, "time_signature": "4/4", "next": [3]}, {"count": 2}],
            "next 3 at count 1 lies beyond the last row",
        ),
    ],
)
def test_row_rules_beyond_the_schema_are_the_library_s(
    rows: list[dict[str, Any]], message: str
) -> None:
    """The schema cannot express these rules; the library reader refuses them."""
    value = _quarters_manifest(rows)
    validate(value)
    manifest = value["timeline"]
    with pytest.raises(ValueError, match=message):
        Timeline.from_dict(manifest)


def test_both_readers_agree_on_integral_json_numbers() -> None:
    """JSON writes 480.0 as 480; both readers take it as the integer it is."""
    ticks = _manifest(DiscreteLogicalTimeline(length=960, unit="ticks"))
    manifest = ticks["timeline"]
    manifest["length"]["value"] = 480.0
    validate(ticks)
    restored = Timeline.from_dict(manifest).to_dict()["length"]
    assert restored == {"unit": "ticks", "number_type": "int", "value": 480}
    assert type(restored["value"]) is int
    manifest["length"]["value"] = 480.5
    with pytest.raises(SchemaValidationError):
        validate(ticks)
    with pytest.raises(ValueError, match="does not match declared number_type 'int'"):
        Timeline.from_dict(manifest)
    manifest["length"]["value"] = "480"
    with pytest.raises(ValueError, match="invalid coordinate value for key 'value'"):
        Timeline.from_dict(manifest)
    seconds = _manifest(ContinuousPhysicalTimeline(length=4.0, unit="seconds"))
    manifest = seconds["timeline"]
    manifest["length"]["value"] = 2
    validate(seconds)
    restored = Timeline.from_dict(manifest).to_dict()["length"]
    assert restored == {"unit": "seconds", "number_type": "float", "value": 2.0}
    assert type(restored["value"]) is float


def test_int64_boundary() -> None:
    """Integer coordinates preserve exact values through the JSON boundary."""
    assert number_to_json(2**63 - 1) == 2**63 - 1
    with pytest.raises(ValueError):
        number_to_json(2**63)
    assert number_from_json(9007199254740993) == 9007199254740993
    timeline = DiscreteLogicalTimeline(length=9007199254740993, unit="ticks")
    assert timeline.to_dict()["length"] == {
        "unit": "ticks",
        "number_type": "int",
        "value": 9007199254740993,
    }
    validate(_manifest(timeline))


def test_library_never_imports_timeskeleton() -> None:
    """The runtime package remains independent of its test-only schema validator."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, timetoalign; print('timeskeleton' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"
