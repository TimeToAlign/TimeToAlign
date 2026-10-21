"""Measure maps in a manifest (see README, *Measure maps in a manifest*).

A manifest holds its measure maps in the top-level ``measure_maps`` array; a
timeline reaches its own through a ``MeasureRecordMap`` among its conversion
maps. The library keeps one measure map per timeline.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from typing import Any

import pytest

from timetoalign import Ms3Loader
from timetoalign.alignment import AlignmentBundle
from timetoalign.maps import (
    ChainMap,
    CombinationMap,
    FloatingMeasureMap,
    LinearMap,
    MeasureNumberMap,
    MeasureRecordMap,
    MetricMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import MeasureMap, Timeline
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
)

F = Fraction
SCORE = ensure_data("score")

ROWS: list[dict[str, Any]] = [
    {
        "count": 1,
        "qstamp": [0, 1],
        "number": 1,
        "name": "1",
        "time_signature": "4/4",
        "nominal_length": [4, 1],
        "actual_length": [4, 1],
        "start_repeat": False,
        "end_repeat": False,
        "next": ["2"],
    },
    {
        "count": 2,
        "qstamp": [4, 1],
        "number": 2,
        "name": "2",
        "time_signature": "4/4",
        "nominal_length": [4, 1],
        "actual_length": [4, 1],
        "start_repeat": False,
        "end_repeat": False,
    },
]
"""Two 4/4 bars, as the library writes them."""


def _bars() -> MeasureMap:
    return MeasureMap.from_rows(copy.deepcopy(ROWS))


def _record(
    measure_map: str = "measuremap_1", map_id: str = "cmap_1"
) -> dict[str, Any]:
    return {
        "type": "MeasureRecordMap",
        "id": map_id,
        "source_unit": "quarters",
        "target_unit": None,
        "measure_map": measure_map,
    }


# region Writing


def test_writing_lifts_a_quarters_timeline_s_measure_map() -> None:
    score = ContinuousLogicalTimeline(length=F(8), uid="score")
    score.add_measure_map(_bars())

    assert score.to_dict(minimal=True) == {
        "timetoshare": "0.1.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [8, 1],
                "conversion_maps": [_record()],
            }
        ],
        "measure_maps": [{"id": "measuremap_1", "rows": ROWS}],
    }
    written = score.to_dict()["timelines"][0]
    assert "measure_map" not in written and "metric_hierarchy" not in written


def test_a_whole_notes_timeline_reads_its_measure_map_through_times_four() -> None:
    piece = ContinuousLogicalTimeline(length=F(2), unit="whole_notes", uid="piece")
    piece.add_measure_map(_bars())

    assert piece.to_dict(minimal=True)["timelines"][0]["conversion_maps"] == [
        {
            "type": "ChainMap",
            "id": "cmap_1",
            "source_unit": "whole_notes",
            "target_unit": None,
            "maps": [
                {
                    "type": "LinearMap",
                    "id": "lmap_1",
                    "source_unit": "whole_notes",
                    "target_unit": "quarters",
                    "scalar": 4,
                },
                _record(map_id="cmap_2"),
            ],
        }
    ]
    assert piece.reach.value == 2
    assert type(piece.reach.value) is Fraction


def test_a_ticks_timeline_reads_its_measure_map_through_the_map_given() -> None:
    midi = DiscreteLogicalTimeline(length=3840, uid="midi")
    to_quarters = LinearMap(
        scalar=F(1, 480), source_unit="ticks", target_unit="quarters", uid="ppq"
    )
    midi.add_measure_map(_bars(), to_quarters=to_quarters)

    (chain,) = midi.to_dict(minimal=True)["timelines"][0]["conversion_maps"]
    assert chain["maps"][0] == {
        "type": "LinearMap",
        "id": "ppq",
        "source_unit": "ticks",
        "target_unit": "quarters",
        "scalar": [1, 480],
    }
    assert midi.reach.value == 3840
    restored = Timeline.from_dict(json.loads(json.dumps(midi.to_dict())))
    assert restored.measure_map == _bars()
    assert restored.to_dict() == midi.to_dict()


@pytest.mark.parametrize(
    ("timeline", "to_quarters", "message"),
    [
        (
            lambda: DiscreteLogicalTimeline(length=960, uid="midi"),
            None,
            "Timeline 'midi' is in ticks: give the LinearMap from ticks to quarters "
            "(to_quarters=) its measure map is read through",
        ),
        (
            lambda: ContinuousLogicalTimeline(length=F(8), uid="score"),
            LinearMap(scalar=1, source_unit="quarters", target_unit="quarters"),
            "Timeline 'score' is on the quarter-note axis: its measure map needs no "
            "LinearMap to quarters",
        ),
        (
            lambda: ContinuousPhysicalTimeline(length=8.0, uid="take"),
            None,
            "Timeline unit 'seconds' does not reach the quarter notes a measure map is "
            "anchored on: give it a quarters, whole_notes or ticks timeline",
        ),
    ],
    ids=["ticks_without_map", "quarters_with_map", "seconds"],
)
def test_add_measure_map_refuses_what_reaches_no_quarters(
    timeline: Any, to_quarters: LinearMap | None, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        timeline().add_measure_map(_bars(), to_quarters=to_quarters)
    assert str(refused.value) == message


# endregion

# region Reading

READ_MINIMAL: dict[str, Any] = {
    "timetoshare": "0.1.0",
    "timelines": [
        {
            "id": "midi",
            "unit": "ticks",
            "length": 3840,
            "conversion_maps": [
                {
                    "type": "ChainMap",
                    "id": "to_bars",
                    "source_unit": "ticks",
                    "target_unit": None,
                    "maps": [
                        {
                            "type": "LinearMap",
                            "id": "ppq",
                            "source_unit": "ticks",
                            "target_unit": "quarters",
                            "scalar": [1, 480],
                        },
                        _record("score_bars", "midi_bars"),
                    ],
                }
            ],
        },
        {
            "id": "score",
            "unit": "quarters",
            "length": [8, 1],
            "conversion_maps": [_record("score_bars", "bars")],
        },
    ],
    "measure_maps": [
        {"id": "score_bars", "name": "Bars", "meta": {"source": "hand"}, "rows": ROWS}
    ],
}


def _complete_map(data: dict[str, Any]) -> dict[str, Any]:
    """A map's complete form: every map writes its ``name``."""
    result = {"name": None, **data}
    if "maps" in result:
        result["maps"] = [_complete_map(member) for member in result["maps"]]
    if result["type"] == "LinearMap":
        result.setdefault("offset", 0)
    return result


def _complete_timeline(
    timeline_id: str, unit: str, number_type: str, **keys: Any
) -> dict:
    return {
        "id": timeline_id,
        "name": None,
        "unit": unit,
        "number_type": number_type,
        "locked": False,
        "is_segment_line": False,
        "meta": {},
        "children": [],
        "regions": [],
        "flow_control": [],
        **keys,
    }


READ_COMPLETE: dict[str, Any] = {
    "timetoshare": "0.1.0",
    "name": None,
    "meta": {},
    "timelines": [
        _complete_timeline(
            "midi",
            "ticks",
            "int",
            length=3840,
            conversion_maps=[
                _complete_map(READ_MINIMAL["timelines"][0]["conversion_maps"][0])
            ],
        ),
        _complete_timeline(
            "score",
            "quarters",
            "fraction",
            length=[8, 1],
            conversion_maps=[_complete_map(_record("score_bars", "bars"))],
        ),
    ],
    "groups": [],
    "measure_maps": READ_MINIMAL["measure_maps"],
}


@pytest.mark.parametrize("form", ["minimal", "complete"])
def test_reading_puts_the_measure_map_back_and_keeps_every_id(form: str) -> None:
    literal = READ_MINIMAL if form == "minimal" else READ_COMPLETE
    bundle = AlignmentBundle.from_dict(copy.deepcopy(literal))
    score, midi = bundle.timelines["score"], bundle.timelines["midi"]

    assert score.measure_map == _bars()
    assert midi.measure_map is score.measure_map
    assert (score.measure_map.id, score.measure_map.name) == ("score_bars", "Bars")
    assert score.measure_map.meta == {"source": "hand"}
    assert score._conversion_maps == {}
    assert bundle.to_dict(minimal=True) == READ_MINIMAL
    assert bundle.to_dict() == READ_COMPLETE


def _one_root(conversion_maps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "timetoshare": "0.1.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [8, 1],
                "conversion_maps": conversion_maps,
            }
        ],
        "measure_maps": [{"id": "bars_map", "rows": ROWS}],
    }


@pytest.mark.parametrize(
    "conversion_maps",
    [
        [_record("bars_map", "bars"), _record("bars_map", "bars2")],
        [
            _record("bars_map", "bars"),
            {
                "type": "ChainMap",
                "id": "bars2",
                "source_unit": "quarters",
                "target_unit": None,
                "maps": [
                    {
                        "type": "LinearMap",
                        "id": "shift",
                        "source_unit": "quarters",
                        "target_unit": "quarters",
                        "offset": 1,
                    },
                    _record("bars_map", "inner"),
                ],
            },
        ],
    ],
    ids=["two_direct", "direct_and_chained"],
)
def test_two_maps_reading_a_measure_map_are_refused(
    conversion_maps: list[dict[str, Any]],
) -> None:
    for reader in (Timeline.from_dict, AlignmentBundle.from_dict):
        with pytest.raises(ValueError) as refused:
            reader(_one_root(copy.deepcopy(conversion_maps)))
        assert str(refused.value) == (
            "timeline 'score' holds 2 MeasureRecordMaps ['bars', 'bars2']: the library "
            "keeps one measure map per timeline"
        )


def test_a_record_map_directly_on_a_ticks_timeline_is_refused() -> None:
    manifest = _one_root([_record("bars_map", "bars")])
    manifest["timelines"][0].update(unit="ticks", length=3840)
    for reader in (Timeline.from_dict, AlignmentBundle.from_dict):
        with pytest.raises(ValueError) as refused:
            reader(copy.deepcopy(manifest))
        assert str(refused.value) == (
            "Map source unit 'quarters' does not match timeline unit 'ticks'"
        )


def test_a_record_map_nested_elsewhere_is_a_view() -> None:
    nested = {
        "type": "CombinationMap",
        "id": "both",
        "source_unit": "quarters",
        "target_unit": None,
        "maps": {"record": _record("bars_map", "inner")},
    }
    manifest = _one_root([_record("bars_map", "bars"), nested])
    score = Timeline.from_dict(copy.deepcopy(manifest))

    kept = score.get_conversion_map("both")
    assert isinstance(kept, CombinationMap)
    assert kept.get_map("record").measure_map is score.measure_map
    assert score.to_dict(minimal=True) == manifest


# endregion

# region Sharing and ids


def _entries(manifest: dict[str, Any]) -> list[str]:
    return [entry["id"] for entry in manifest["measure_maps"]]


def _named(manifest: dict[str, Any]) -> dict[str, str]:
    """Each timeline's record map's measure map id."""
    return {
        timeline["id"]: timeline["conversion_maps"][0]["measure_map"]
        for timeline in manifest["timelines"]
    }


@pytest.mark.parametrize(
    ("second", "entries", "named"),
    [
        ("same", ["measuremap_1"], {"a": "measuremap_1", "b": "measuremap_1"}),
        ("equal", ["measuremap_1"], {"a": "measuremap_1", "b": "measuremap_1"}),
        (
            "different",
            ["measuremap_1", "measuremap_2"],
            {"a": "measuremap_1", "b": "measuremap_2"},
        ),
    ],
)
def test_timelines_share_an_entry_when_their_measure_maps_are_equal(
    second: str, entries: list[str], named: dict[str, str]
) -> None:
    first_map = _bars()
    second_map = {
        "same": first_map,
        "equal": _bars(),
        "different": MeasureMap.from_rows([{"count": 1, "time_signature": "3/4"}]),
    }[second]
    bundle = AlignmentBundle()
    for timeline_id, measure_map in (("a", first_map), ("b", second_map)):
        timeline = ContinuousLogicalTimeline(length=F(8), uid=timeline_id)
        timeline.add_measure_map(measure_map)
        bundle.add_timeline(timeline)

    written = bundle.to_dict(minimal=True)
    assert _entries(written) == entries
    assert _named(written) == named
    read = AlignmentBundle.from_dict(written)
    shared = read.timelines["a"].measure_map is read.timelines["b"].measure_map
    assert shared is (second != "different")


def test_an_entry_no_timeline_names_is_kept() -> None:
    manifest = {
        "timetoshare": "0.1.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "measure_maps": [{"id": "spare", "rows": ROWS}],
    }
    for reader in (Timeline.from_dict, AlignmentBundle.from_dict):
        assert reader(copy.deepcopy(manifest)).to_dict(minimal=True) == manifest


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"measure_maps": [{"id": "score", "rows": ROWS}]},
            "the id 'score' is held twice in the manifest: by measure map 'score' "
            "and by timeline 'score'",
        ),
        (
            {
                "measure_maps": [{"id": "g", "rows": ROWS}],
                "groups": [{"id": "g", "members": ["score"]}],
            },
            "the id 'g' is held twice in the manifest: by measure map 'g' and by "
            "group 'g'",
        ),
        (
            {"id": "op35", "measure_maps": [{"id": "op35", "rows": ROWS}]},
            "the id 'op35' is held twice in the manifest: by the manifest and by "
            "measure map 'op35'",
        ),
        (
            {
                "measure_maps": [
                    {"id": "bars", "rows": ROWS},
                    {"id": "bars", "rows": ROWS},
                ]
            },
            "the id 'bars' is held twice in the manifest: by measure map 'bars' "
            "and by measure map 'bars'",
        ),
        (
            {
                "measure_maps": [{"id": "bars", "rows": ROWS}],
                "timelines": [
                    {
                        "id": "score",
                        "unit": "quarters",
                        "length": [8, 1],
                        "regions": [{"id": "bars", "start": [0, 1], "end": [4, 1]}],
                    }
                ],
            },
            "the id 'bars' is held twice in the manifest: by measure map 'bars' "
            "and by region 'bars' of timeline 'score'",
        ),
    ],
    ids=["timeline", "group", "manifest", "twice", "region"],
)
def test_a_measure_map_id_is_unique_with_the_manifest_s_ids(
    change: dict[str, Any], message: str
) -> None:
    manifest = {
        "timetoshare": "0.1.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        **change,
    }
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(manifest)
    assert str(refused.value) == message


def test_a_minted_measure_map_id_is_minted_around_the_manifest_s_ids() -> None:
    score = ContinuousLogicalTimeline(length=F(8), uid="measuremap_1")
    score.add_measure_map(_bars())
    assert _entries(score.to_dict()) == ["measuremap_2"]


def test_a_given_measure_map_id_held_by_a_timeline_is_refused_on_writing() -> None:
    score = ContinuousLogicalTimeline(length=F(8), uid="score")
    score.add_measure_map(MeasureMap(_bars().measures, id="score"))
    with pytest.raises(ValueError) as refused:
        score.to_dict()
    assert str(refused.value) == (
        "measure map 'score' cannot be written: its id is already held by "
        "timeline 'score'"
    )


def test_two_different_measure_maps_of_one_given_id_are_refused_on_writing() -> None:
    bundle = AlignmentBundle()
    for timeline_id, rows in (
        ("a", ROWS),
        ("b", [{"count": 1, "time_signature": "3/4"}]),
    ):
        timeline = ContinuousLogicalTimeline(length=F(8), uid=timeline_id)
        measure_map = MeasureMap.from_rows(copy.deepcopy(rows))
        timeline.add_measure_map(MeasureMap(measure_map.measures, id="bars"))
        bundle.add_timeline(timeline)
    with pytest.raises(ValueError) as refused:
        bundle.to_dict()
    assert str(refused.value) == "two different measure maps carry the id 'bars'"


# endregion

# region Measure-derived maps


def test_a_score_s_floating_measure_map_names_the_timeline_s_entry() -> None:
    score = (
        Ms3Loader()
        .load(SCORE / "flow_control/full/out_of_the_flow_experience-full.measures.tsv")
        .create_timeline()
    )
    floating = [
        cmap
        for cmap in score._conversion_maps.values()
        if isinstance(cmap, FloatingMeasureMap)
    ]
    assert len(floating) == 1
    assert floating[0].measure_map is not score.measure_map

    written = score.to_dict(minimal=True)
    assert _entries(written) == ["measuremap_1"]
    views = [
        cmap["measure_map"]
        for cmap in written["timelines"][0]["conversion_maps"]
        if "measure_map" in cmap
    ]
    assert views == ["measuremap_1", "measuremap_1"]
    read = Timeline.from_dict(json.loads(json.dumps(written)))
    read_floating = read.get_conversion_map(floating[0].id)
    assert read_floating.measure_map is read.measure_map
    assert read.to_dict(minimal=True) == written


def _timeline_with(*maps: Any) -> ContinuousLogicalTimeline:
    score = ContinuousLogicalTimeline(length=F(8), uid="score")
    score.add_measure_map(_bars())
    for cmap in maps:
        score.add_conversion_map(cmap)
    return score


def test_legacy_maps_built_from_the_timeline_s_measures_name_its_entry() -> None:
    bars = _bars()
    maps = [
        QuartersToFloatingMeasures(
            x_values=[0, 4, 8], y_values=[1.0, 2.0, 3.0], uid="f"
        ),
        QuartersToMeasureNumber(boundaries=[0, 4], mns=["1", "2"], uid="n"),
        MetricMap(
            starts=[m.qstamp for m in bars],
            mcs=[m.number for m in bars],
            mns=[m.name for m in bars],
            lengths=[m.actual_length for m in bars],
            uid="m",
        ),
    ]
    written = _timeline_with(*maps).to_dict(minimal=True)

    assert _entries(written) == ["measuremap_1"]
    assert {
        cmap["id"]: cmap["measure_map"]
        for cmap in written["timelines"][0]["conversion_maps"]
    } == {
        "cmap_1": "measuremap_1",
        "f": "measuremap_1",
        "m": "measuremap_1",
        "n": "measuremap_1",
    }
    read = Timeline.from_dict(written)
    for map_id in ("f", "m", "n"):
        assert read.get_conversion_map(map_id).measure_map is read.measure_map


PROBES: list[Any] = [-1, 0, F(1, 2), 3, 4, 5.5, 7, 8, 11]


def test_legacy_maps_built_from_other_data_write_entries_of_their_own() -> None:
    floating = QuartersToFloatingMeasures(
        x_values=[0, 3, 8], y_values=[1.0, 2.0, 3.0], uid="f"
    )
    meter = MetricMap(
        starts=[F(0), F(2)], mcs=[1, 2], mns=["0", "1"], lengths=[F(2), F(6)], uid="m"
    )
    written = _timeline_with(floating, meter).to_dict(minimal=True)

    assert _entries(written) == ["measuremap_1", "measuremap_2", "measuremap_3"]
    assert {
        cmap["id"]: cmap["measure_map"]
        for cmap in written["timelines"][0]["conversion_maps"]
    } == {"cmap_1": "measuremap_1", "f": "measuremap_2", "m": "measuremap_3"}
    read = Timeline.from_dict(json.loads(json.dumps(written)))
    read_floating, read_meter = read.get_conversion_map("f"), read.get_conversion_map(
        "m"
    )
    assert isinstance(read_floating, FloatingMeasureMap)
    assert isinstance(read_meter, MeasureNumberMap)
    assert [read_floating(p) for p in PROBES] == [floating(p) for p in PROBES]
    assert [read_meter(p) for p in PROBES] == [meter(p) for p in PROBES]
    assert [
        (m.qstamp, m.number, m.name, m.actual_length) for m in read_meter.measure_map
    ] == [
        (0, 1, "0", 2),
        (2, 2, "1", 6),
    ]


def test_a_view_over_the_timeline_s_measure_map_reads_back_over_it() -> None:
    score = _timeline_with()
    score.add_conversion_map(MeasureNumberMap(score.measure_map, uid="numbers"))
    read = Timeline.from_dict(score.to_dict())
    assert read.get_conversion_map("numbers").measure_map is read.measure_map
    assert isinstance(read._measure_record_map, MeasureRecordMap)
    assert not isinstance(read._measure_record_map, ChainMap)


# endregion
