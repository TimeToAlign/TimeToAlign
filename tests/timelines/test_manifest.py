"""TimeSkeleton manifests: the one shape every reader and writer shares.

The strategy is documented in ``tests/timelines/README.md`` ("Manifests").
Manifests are written by hand from the format's rules, never taken from the
library's own writer, and what the library writes back is compared with the
same literal text.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, NumberType, SupportPolicy, TimeUnit
from timetoalign.timelines import Timeline, TimelineGroup
from timetoalign.timelines.manifest import TIMESKELETON_VERSION
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

READERS = {
    "timeline": Timeline.from_dict,
    "group": lambda data: TimelineGroup.from_dict(data, group_id="g"),
    "bundle": AlignmentBundle.from_dict,
}


def _text(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


# region The version marker


@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
def test_a_manifest_without_the_marker_is_refused(reader: str) -> None:
    with pytest.raises(ValueError) as refused:
        READERS[reader]({"timelines": []})
    assert str(refused.value) == "not a TimeSkeleton manifest"


@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
def test_another_version_is_refused_naming_the_current_one(reader: str) -> None:
    with pytest.raises(ValueError) as refused:
        READERS[reader]({"timeskeleton": "0.6.0", "timelines": []})
    assert str(refused.value) == (
        "unsupported TimeSkeleton version '0.6.0': this library reads "
        "TimeSkeleton 0.7.0 only"
    )


def test_the_marker_is_written_from_one_constant() -> None:
    assert TIMESKELETON_VERSION == "0.7.0"
    assert AlignmentBundle().to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "timelines": [],
    }


# endregion


# region Defaults


MINIMAL: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "timelines": [
        {
            "id": "score",
            "unit": "quarters",
            "length": [32, 1],
            "children": [
                {
                    "id": "coda",
                    "unit": "quarters",
                    "offset": [24, 1],
                    "length": [8, 1],
                    "locked": True,
                }
            ],
            "conversion_maps": [
                {
                    "type": "LinearMap",
                    "id": "tempo",
                    "source_unit": "quarters",
                    "target_unit": "seconds",
                    "scalar": [1, 2],
                    "offset": 0,
                }
            ],
            "regions": [{"id": "theme", "start": [0, 1], "end": [8, 1]}],
            "flow_control": {
                "breaks": [{"id": "fine", "coordinate": [32, 1]}],
                "jumps": [
                    {"id": "back", "from_coordinate": [16, 1], "to_coordinate": [0, 1]}
                ],
                "markers": [{"id": "segno", "coordinate": [4, 1]}],
            },
        }
    ],
}

COMPLETE: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "name": None,
    "meta": {},
    "timelines": [
        {
            "id": "score",
            "name": None,
            "unit": "quarters",
            "number_type": "fraction",
            "length": [32, 1],
            "locked": False,
            "is_segment_line": False,
            "meta": {},
            "children": [
                {
                    "id": "coda",
                    "name": None,
                    "unit": "quarters",
                    "number_type": "fraction",
                    "offset": [24, 1],
                    "length": [8, 1],
                    "locked": True,
                    "is_segment_line": False,
                    "meta": {},
                    "children": [],
                    "conversion_maps": [],
                    "regions": [],
                    "flow_control": {"breaks": [], "jumps": [], "markers": []},
                    "measure_map": None,
                    "metric_hierarchy": None,
                }
            ],
            "conversion_maps": [
                {
                    "type": "LinearMap",
                    "id": "tempo",
                    "name": None,
                    "source_unit": "quarters",
                    "target_unit": "seconds",
                    "scalar": [1, 2],
                    "offset": 0,
                }
            ],
            "regions": [
                {
                    "id": "theme",
                    "name": None,
                    "start": [0, 1],
                    "end": [8, 1],
                    "meta": {},
                }
            ],
            "flow_control": {
                "breaks": [
                    {
                        "id": "fine",
                        "coordinate": [32, 1],
                        "control_type": "section_break",
                        "condition": "always",
                        "repeat_count": 1,
                        "label": None,
                        "name": None,
                        "meta": {},
                    }
                ],
                "jumps": [
                    {
                        "id": "back",
                        "from_coordinate": [16, 1],
                        "to_coordinate": [0, 1],
                        "control_type": "repeat_end",
                        "condition": "first_n",
                        "repeat_count": 1,
                        "target": None,
                        "label": None,
                        "meta": {},
                    }
                ],
                "markers": [{"id": "segno", "name": None, "coordinate": [4, 1]}],
            },
            "measure_map": None,
            "metric_hierarchy": None,
        }
    ],
    "groups": [],
}


def test_minimal_and_complete_read_to_equal_objects() -> None:
    from_minimal = Timeline.from_dict(copy.deepcopy(MINIMAL))
    from_complete = Timeline.from_dict(copy.deepcopy(COMPLETE))

    assert from_minimal.to_dict() == from_complete.to_dict() == COMPLETE
    assert (
        from_minimal.to_dict(minimal=True)
        == from_complete.to_dict(minimal=True)
        == MINIMAL
    )
    assert _text(from_complete.to_dict()) == _text(COMPLETE)
    assert from_minimal.get_region("theme") == from_complete.get_region("theme")
    assert from_minimal.flow_control.breaks == from_complete.flow_control.breaks
    assert from_minimal.flow_control.jumps == from_complete.flow_control.jumps
    assert from_minimal.flow_control.markers == from_complete.flow_control.markers


def test_a_reader_supplies_every_default() -> None:
    timeline = Timeline.from_dict(copy.deepcopy(MINIMAL))
    (brk,) = timeline.flow_control.breaks
    (jump,) = timeline.flow_control.jumps

    assert (timeline.name, timeline.meta, timeline.is_locked) == ("score", {}, False)
    assert timeline.number_type is NumberType.fraction
    assert timeline.measure_map is None and timeline.metric_hierarchy is None
    assert (brk.control_type.value, brk.condition.value, brk.repeat_count) == (
        "section_break",
        "always",
        1,
    )
    assert (brk.label, brk.name, brk.meta) == (None, None, {})
    assert (jump.control_type.value, jump.condition.value, jump.repeat_count) == (
        "repeat_end",
        "first_n",
        1,
    )
    assert (jump.target, jump.label, jump.meta) == (None, None, {})
    assert timeline.get_conversion_map("tempo").to_dict()["name"] is None


UNIT_DEFAULTS = {
    "ticks": ("int", 4),
    "samples": ("int", 4),
    "frames": ("int", 4),
    "pixels": ("int", 4),
    "quarters": ("fraction", [4, 1]),
    "whole_note": ("fraction", [4, 1]),
    "number": ("float", 4.5),
    "floating_measures": ("float", 4.5),
    "milliseconds": ("float", 4.5),
    "seconds": ("float", 4.5),
    "minutes": ("float", 4.5),
    "meters": ("float", 4.5),
    "centimeters": ("float", 4.5),
    "millimeters": ("float", 4.5),
    "inches": ("float", 4.5),
    "points": ("float", 4.5),
}


def test_the_unit_defaults_cover_every_unit() -> None:
    assert sorted(UNIT_DEFAULTS) == sorted(unit.value for unit in TimeUnit)


@pytest.mark.parametrize("unit", list(UNIT_DEFAULTS), ids=list(UNIT_DEFAULTS))
def test_a_number_type_left_out_is_the_units_own(unit: str) -> None:
    number_type, length = UNIT_DEFAULTS[unit]
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }

    timeline = Timeline.from_dict(data)

    assert timeline.number_type is NumberType[number_type]
    assert timeline.to_dict(minimal=True) == data
    assert timeline.to_dict()["timelines"][0]["number_type"] == number_type


def test_the_class_comes_from_the_unit_and_a_class_key_is_kept() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [
            {
                "id": "perf",
                "class": "ContinuousGraphicalTimeline",
                "unit": "seconds",
                "length": 12.5,
            }
        ],
    }

    timeline = Timeline.from_dict(copy.deepcopy(data))

    assert type(timeline) is ContinuousPhysicalTimeline
    assert timeline.to_dict(minimal=True) == data


# endregion


# region Representation


def _with_value(path: tuple[Any, ...], value: Any) -> dict[str, Any]:
    """The minimal manifest with the value at *path* (below its root) replaced."""
    data = copy.deepcopy(MINIMAL)
    target = data["timelines"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("length",),
            32,
            "length 32 of 'score' is an int on a fraction axis: write [32, 1]",
        ),
        (
            ("offset",),
            -4,
            "offset -4 of 'score' is an int on a fraction axis: write [-4, 1]",
        ),
        (
            ("children", 0, "offset"),
            24.5,
            "offset 24.5 of 'coda' is a float on a fraction axis: write [49, 2]",
        ),
        (
            ("regions", 0, "start"),
            0,
            "start 0 of 'theme' is an int on a fraction axis: write [0, 1]",
        ),
        (
            ("regions", 0, "end"),
            8.0,
            "end 8.0 of 'theme' is a float on a fraction axis: write [8, 1]",
        ),
        (
            ("flow_control", "breaks", 0, "coordinate"),
            32,
            "coordinate 32 of 'fine' is an int on a fraction axis: write [32, 1]",
        ),
        (
            ("flow_control", "markers", 0, "coordinate"),
            4,
            "coordinate 4 of 'segno' is an int on a fraction axis: write [4, 1]",
        ),
        (
            ("flow_control", "jumps", 0, "from_coordinate"),
            16,
            "from_coordinate 16 of 'back' is an int on a fraction axis: write [16, 1]",
        ),
        (
            ("flow_control", "jumps", 0, "to_coordinate"),
            0.25,
            "to_coordinate 0.25 of 'back' is a float on a fraction axis: write [1, 4]",
        ),
    ],
    ids=[
        "length",
        "root_offset",
        "child_offset",
        "region_start",
        "region_end",
        "break",
        "marker",
        "jump_from",
        "jump_to",
    ],
)
def test_each_bare_key_checks_its_representation(
    path: tuple[Any, ...], value: Any, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(_with_value(path, value))
    assert str(refused.value) == message


@pytest.mark.parametrize(
    ("unit", "length", "message"),
    [
        (
            "seconds",
            [5, 2],
            "length [5, 2] of 't' is a fraction on a float axis: write 2.5",
        ),
        (
            "seconds",
            [1, 3],
            "length [1, 3] of 't' is a fraction on a float axis: write a float",
        ),
        (
            "quarters",
            0.1,
            "length 0.1 of 't' is a float on a fraction axis: "
            "write [3602879701896397, 36028797018963968]",
        ),
        (
            "ticks",
            [480, 1],
            "length [480, 1] of 't' is a fraction on an int axis: write 480",
        ),
        (
            "ticks",
            [1, 2],
            "length [1, 2] of 't' is a fraction on an int axis: write an int",
        ),
        (
            "ticks",
            480.5,
            "length 480.5 of 't' is a float on an int axis: write an int",
        ),
        ("seconds", "8", "length \"8\" of 't' is not a number"),
    ],
    ids=[
        "fraction_on_float",
        "inexact_fraction_on_float",
        "float_on_fraction",
        "integral_pair_on_int",
        "pair_on_int",
        "float_on_int",
        "string",
    ],
)
def test_every_axis_kind_checks_its_representation(
    unit: str, length: Any, message: str
) -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(data)
    assert str(refused.value) == message


@pytest.mark.parametrize(
    ("unit", "length", "expected"),
    [("ticks", 480.0, 480), ("seconds", 8, 8.0)],
    ids=["integral_number_on_int", "integer_on_float"],
)
def test_numbers_javascript_cannot_tell_apart_are_accepted(
    unit: str, length: Any, expected: Any
) -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }

    value = Timeline.from_dict(data).length.value

    assert (value, type(value)) == (expected, type(expected))


# endregion


# region Unknown keys


UNKNOWN: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "$schema": "https://timetoalign.github.io/timeskeleton/0.7.0/timeskeleton.schema.json",
    "x-note": "manifest",
    "timelines": [
        {"id": "perf", "unit": "seconds", "length": 12.5},
        {
            "id": "score",
            "unit": "quarters",
            "length": [32, 1],
            "x-note": "root",
            "children": [
                {
                    "id": "coda",
                    "unit": "quarters",
                    "offset": [24, 1],
                    "length": [8, 1],
                    "locked": True,
                    "x-note": "child",
                }
            ],
            "conversion_maps": [
                {
                    "type": "ChainMap",
                    "id": "chain",
                    "source_unit": None,
                    "target_unit": None,
                    "x-note": "map",
                    "maps": [
                        {
                            "type": "ShiftMap",
                            "id": "shift",
                            "source_unit": None,
                            "target_unit": None,
                            "offset": 1,
                            "x-note": "nested map",
                        }
                    ],
                }
            ],
            "regions": [
                {"id": "theme", "start": [0, 1], "end": [8, 1], "x-note": "region"}
            ],
            "flow_control": {
                "x-note": "flow control",
                "breaks": [{"id": "fine", "coordinate": [32, 1], "x-note": "break"}],
                "jumps": [
                    {
                        "id": "back",
                        "from_coordinate": [16, 1],
                        "to_coordinate": [0, 1],
                        "x-note": "jump",
                    }
                ],
                "markers": [{"id": "segno", "coordinate": [4, 1], "x-note": "marker"}],
            },
        },
    ],
    "groups": [{"id": "g", "members": ["coda", "perf"], "x-note": "group"}],
    "graphical": {"order": {"g": ["perf", "coda"]}, "x-note": "graphical"},
}


def test_unknown_keys_are_written_back_where_they_were_found() -> None:
    bundle = AlignmentBundle.from_dict(copy.deepcopy(UNKNOWN))

    assert bundle.to_dict(minimal=True) == UNKNOWN
    assert AlignmentBundle.from_dict(bundle.to_dict()).to_dict(minimal=True) == UNKNOWN


def test_the_group_reader_keeps_the_unknown_keys_of_what_it_reads() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(UNKNOWN))

    assert group.to_dict(minimal=True) == UNKNOWN


def test_the_timeline_reader_keeps_the_unknown_keys_of_what_it_reads() -> None:
    score = Timeline.from_dict(copy.deepcopy(UNKNOWN), timeline_id="score")

    assert score.to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "timelines": [UNKNOWN["timelines"][1]],
    }


def test_a_one_root_manifest_keeps_its_envelope() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "id": "op35",
        "name": "Eroica Variations",
        "meta": {"opus": 35},
        "x-note": "manifest",
        "timelines": [{"id": "score", "unit": "quarters", "length": [32, 1]}],
        "graphical": {"styles": {"thin": {"stroke-width": 1}}},
    }

    assert Timeline.from_dict(copy.deepcopy(data)).to_dict(minimal=True) == data


# endregion


# region A root's offset


def _offset_manifest(offset: float | None) -> dict[str, Any]:
    root: dict[str, Any] = {
        "id": "recording",
        "unit": "seconds",
        "length": 10.0,
        "children": [
            {
                "id": "take",
                "unit": "seconds",
                "offset": 2.0,
                "length": 3.0,
                "locked": True,
            }
        ],
    }
    if offset is not None:
        root["offset"] = offset
    return {"timeskeleton": "0.7.0", "timelines": [root]}


def test_a_negative_root_offset_round_trips() -> None:
    data = _offset_manifest(-5.5)

    timeline = Timeline.from_dict(copy.deepcopy(data))

    assert timeline.root_offset == Coordinate(-5.5, TimeUnit.seconds)
    assert timeline.to_dict(minimal=True) == data
    assert timeline.to_dict()["timelines"][0]["offset"] == -5.5


def test_a_root_offset_changes_nothing_computed() -> None:
    placed = Timeline.from_dict(_offset_manifest(-5.5))
    plain = Timeline.from_dict(_offset_manifest(None))

    assert plain.root_offset is None
    assert placed.get_child_offset("take") == plain.get_child_offset("take")
    for position in (0.0, 2.0, 3.5, 10.0):
        assert placed.get_timestamp_at(position).coordinates == (
            plain.get_timestamp_at(position).coordinates
        )


# endregion


# region Selection


SELECTION: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "timelines": [
        {
            "id": "score",
            "name": "Score",
            "unit": "quarters",
            "length": [32, 1],
            "children": [
                {
                    "id": "part",
                    "name": "Score",
                    "unit": "quarters",
                    "offset": [0, 1],
                    "length": [16, 1],
                    "locked": True,
                }
            ],
        },
        {"id": "perf", "name": "Performance", "unit": "seconds", "length": 60.0},
    ],
    "groups": [
        {"id": "g1", "name": "pair", "members": ["score"]},
        {"id": "g2", "name": "pair", "members": ["perf"]},
    ],
}

TIMELINES = "'score' (name 'Score'), 'part' (name 'Score'), 'perf' (name 'Performance')"
GROUPS = "'g1' (name 'pair'), 'g2' (name 'pair')"


def test_a_timeline_is_selected_by_id_or_name() -> None:
    part = Timeline.from_dict(copy.deepcopy(SELECTION), timeline_id="part")
    perf = Timeline.from_dict(copy.deepcopy(SELECTION), name="Performance")

    assert (part.id, part.length) == (
        "part",
        Coordinate(Fraction(16), TimeUnit.quarters),
    )
    assert part.to_dict(minimal=True)["timelines"] == [
        {
            "id": "part",
            "name": "Score",
            "unit": "quarters",
            "length": [16, 1],
            "locked": True,
        }
    ]
    assert perf.id == "perf"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"name": "Score"},
            f"the manifest holds several timeline with name 'Score'; its timelines: {TIMELINES}",
        ),
        (
            {"timeline_id": "nope"},
            f"the manifest holds no timeline with id 'nope'; its timelines: {TIMELINES}",
        ),
        (
            {"timeline_id": "perf", "name": "Score"},
            f"the manifest holds no timeline with id 'perf', name 'Score'; its timelines: {TIMELINES}",
        ),
        (
            {},
            "the manifest holds several timelines or groups: give timeline_id or name to "
            f"select one; its timelines: {TIMELINES}",
        ),
    ],
    ids=["two_matches", "no_match", "id_and_name_disagree", "no_argument"],
)
def test_a_timeline_selection_must_match_once(
    arguments: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(copy.deepcopy(SELECTION), **arguments)
    assert str(refused.value) == message


def test_a_one_root_manifest_is_read_whole_with_overrides() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    timeline = Timeline.from_dict(data, timeline_id="edition", name="Edition")

    assert (timeline.id, timeline.name) == ("edition", "Edition")


FAMILY: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "timelines": [
        {
            "id": "ex",
            "name": "Example",
            "unit": "quarters",
            "length": [16, 1],
            "children": [
                {
                    "id": "ex_child",
                    "name": "Child",
                    "unit": "quarters",
                    "offset": [0, 1],
                    "length": [8, 1],
                    "locked": True,
                    "regions": [{"id": "intro", "start": [0, 1], "end": [2, 1]}],
                },
                {
                    "id": "ex_coda",
                    "name": "Example",
                    "unit": "quarters",
                    "offset": [8, 1],
                    "length": [8, 1],
                    "locked": True,
                },
            ],
        }
    ],
}

FAMILY_TIMELINES = (
    "'ex' (name 'Example'), 'ex_child' (name 'Child'), 'ex_coda' (name 'Example')"
)


def _tree_ids(timeline: Timeline) -> list[str]:
    root = timeline
    while root._parent is not None:
        root = root._parent
    return [root.id, *(descendant.id for _, descendant in root._iter_descendants())]


@pytest.mark.parametrize(
    "arguments",
    [
        {"timeline_id": "ex_child"},
        {"name": "Child"},
        {"timeline_id": "ex_child", "name": "Child"},
    ],
    ids=["by_id", "by_name", "by_id_and_name"],
)
def test_a_descendant_of_one_root_is_selected_not_taken_by_the_root(
    arguments: dict[str, str],
) -> None:
    child = Timeline.from_dict(copy.deepcopy(FAMILY), **arguments)

    assert (child.id, child.name, child.length, child._parent.id) == (
        "ex_child",
        "Child",
        Coordinate(Fraction(8), TimeUnit.quarters),
        "ex",
    )
    assert _tree_ids(child) == ["ex", "ex_child", "ex_coda"]


@pytest.mark.parametrize(
    ("arguments", "root"),
    [
        ({"timeline_id": "edition"}, {"id": "edition"}),
        ({"name": "Edition"}, {"name": "Edition"}),
        (
            {"timeline_id": "edition", "name": "Edition"},
            {"id": "edition", "name": "Edition"},
        ),
    ],
    ids=["id", "name", "id_and_name"],
)
def test_arguments_matching_no_timeline_override_the_one_root(
    arguments: dict[str, str], root: dict[str, str]
) -> None:
    timeline = Timeline.from_dict(copy.deepcopy(FAMILY), **arguments)

    assert timeline._parent is None
    assert timeline.to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "timelines": [{**FAMILY["timelines"][0], **root}],
    }


@pytest.mark.parametrize(
    ("data", "timeline_id", "message"),
    [
        (
            FAMILY,
            "intro",
            "root timeline 'ex' cannot take the id 'intro': it is already held by "
            "region 'intro' of timeline 'ex_child'",
        ),
        (
            {**FAMILY, "id": "op35"},
            "op35",
            "root timeline 'ex' cannot take the id 'op35': it is already held by the manifest",
        ),
    ],
    ids=["descendant_region", "manifest"],
)
def test_an_override_never_takes_an_id_held_in_the_manifest(
    data: dict[str, Any], timeline_id: str, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(copy.deepcopy(data), timeline_id=timeline_id)
    assert str(refused.value) == message


TWO_ROOTS: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "timelines": [
        {"id": "score", "unit": "quarters", "length": [8, 1]},
        {"id": "perf", "unit": "seconds", "length": 60.0},
    ],
}

ONE_ROOT_ONE_GROUP: dict[str, Any] = {
    "timeskeleton": "0.7.0",
    "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    "groups": [{"id": "g", "members": ["score"]}],
}


@pytest.mark.parametrize(
    ("data", "arguments", "message"),
    [
        (
            FAMILY,
            {"name": "Example"},
            "the manifest holds several timeline with name 'Example'; its timelines: "
            f"{FAMILY_TIMELINES}",
        ),
        (
            FAMILY,
            {"timeline_id": "ex_child", "name": "Edition"},
            "the manifest holds no timeline with id 'ex_child', name 'Edition'; its "
            f"timelines: {FAMILY_TIMELINES}",
        ),
        (
            TWO_ROOTS,
            {"timeline_id": "edition"},
            "the manifest holds no timeline with id 'edition'; its timelines: "
            "'score' (name None), 'perf' (name None)",
        ),
        (
            ONE_ROOT_ONE_GROUP,
            {"timeline_id": "edition"},
            "the manifest holds no timeline with id 'edition'; its timelines: "
            "'score' (name None)",
        ),
        (
            TWO_ROOTS,
            {},
            "the manifest holds several timelines or groups: give timeline_id or name to "
            "select one; its timelines: 'score' (name None), 'perf' (name None)",
        ),
    ],
    ids=[
        "several_matches",
        "id_matches_name_does_not",
        "two_roots_unmatched",
        "one_group_unmatched",
        "two_roots_no_argument",
    ],
)
def test_a_selection_that_cannot_override_raises(
    data: dict[str, Any], arguments: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(copy.deepcopy(data), **arguments)
    assert str(refused.value) == message


def test_a_group_is_selected_by_id() -> None:
    group = TimelineGroup.from_dict(copy.deepcopy(SELECTION), group_id="g1")

    assert (group.id, group.name, group.timeline_ids) == ("g1", "pair", ["score"])
    assert group.to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "timelines": [SELECTION["timelines"][0]],
        "groups": [SELECTION["groups"][0]],
        "graphical": {"order": {"g1": ["score"]}},
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"name": "pair"},
            f"the manifest holds several group with name 'pair'; its groups: {GROUPS}",
        ),
        (
            {"group_id": "nope"},
            f"the manifest holds no group with id 'nope'; its groups: {GROUPS}",
        ),
        (
            {},
            "the manifest holds several groups or is named: give group_id or name to select a group",
        ),
    ],
    ids=["two_matches", "no_match", "no_argument"],
)
def test_a_group_selection_must_match_once(
    arguments: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(copy.deepcopy(SELECTION), **arguments)
    assert str(refused.value) == message


ONE_TIMELINE = {
    "timeskeleton": "0.7.0",
    "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
}


def test_a_manifest_of_one_timeline_makes_a_group_when_asked() -> None:
    group = TimelineGroup.from_dict(
        copy.deepcopy(ONE_TIMELINE), group_id="edition", name="Edition"
    )

    assert (group.id, group.name, group.timeline_ids) == (
        "edition",
        "Edition",
        ["score"],
    )
    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(copy.deepcopy(ONE_TIMELINE))
    assert str(refused.value) == (
        "the manifest holds one timeline and no group: give group_id or name to make a group of it"
    )


def test_a_one_group_manifest_is_read_whole_with_overrides() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "meta": {"source": "an edition"},
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "name": "old", "members": ["score"]}],
    }

    group = TimelineGroup.from_dict(
        copy.deepcopy(data), group_id="edition", name="Edition"
    )

    assert (group.id, group.name) == ("edition", "Edition")
    assert group.to_dict(minimal=True) == {
        **data,
        "groups": [{"id": "edition", "name": "Edition", "members": ["score"]}],
        "graphical": {"order": {"edition": ["score"]}},
    }
    assert TimelineGroup.from_dict(copy.deepcopy(data)).to_dict(minimal=True) == {
        **data,
        "graphical": {"order": {"g": ["score"]}},
    }


@pytest.mark.parametrize(
    "data",
    [
        {**FAMILY},
        {**FAMILY, "groups": [{"id": "g", "members": ["ex"]}]},
    ],
    ids=["one_timeline_no_group", "one_anonymous_group"],
)
def test_a_group_never_takes_the_id_of_a_descendant_timeline(
    data: dict[str, Any],
) -> None:
    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(copy.deepcopy(data), group_id="ex_child")
    assert str(refused.value) == "group id 'ex_child' is also a timeline id"


def test_a_group_id_matching_the_one_group_reads_it_unchanged() -> None:
    data = {**FAMILY, "groups": [{"id": "g", "name": "pair", "members": ["ex"]}]}

    group = TimelineGroup.from_dict(copy.deepcopy(data), group_id="g")

    assert group.to_dict(minimal=True) == {
        **data,
        "graphical": {"order": {"g": ["ex"]}},
    }


def test_a_named_one_group_manifest_needs_a_selection() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "name": "Study",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "members": ["score"]}],
    }

    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(copy.deepcopy(data))
    assert str(refused.value) == (
        "the manifest holds several groups or is named: give group_id or name to select a group"
    )
    assert TimelineGroup.from_dict(copy.deepcopy(data), group_id="g").id == "g"


# endregion


# region The manifest's identity


def test_an_anonymous_bundle_writes_no_id() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(copy.deepcopy(data))

    assert bundle.id == "bundle_1"
    assert bundle.to_dict(minimal=True) == data


def test_a_manifest_with_a_name_and_no_id() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "name": "Eroica Variations",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(copy.deepcopy(data))

    assert (bundle.name, bundle.to_dict(minimal=True)) == ("Eroica Variations", data)
    assert Timeline.from_dict(copy.deepcopy(data)).to_dict(minimal=True) == data


def test_a_support_policy_key_is_not_read() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "support_policy": "clamp",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(copy.deepcopy(data))

    assert bundle.support_policy is SupportPolicy.omit
    assert bundle.to_dict(minimal=True) == data


def test_the_manifest_id_is_unique_with_the_timeline_ids() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "id": "score",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(data)
    assert str(refused.value) == "manifest id 'score' is also a timeline or group id"


# endregion


# region graphical.order


def test_the_order_is_keyed_by_group_id() -> None:
    data = copy.deepcopy(SELECTION)
    data["graphical"] = {"order": [["score"], ["perf"]]}

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(data)
    assert str(refused.value) == "'graphical.order' must map group ids to orders"


def test_an_order_naming_no_group_is_refused() -> None:
    data = copy.deepcopy(SELECTION)
    data["graphical"] = {"order": {"g3": ["perf"]}}

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(data)
    assert str(refused.value) == "'graphical.order' names no group: ['g3']"


# endregion


def test_a_written_manifest_is_a_json_fixpoint() -> None:
    score = ContinuousLogicalTimeline(length=Fraction(10, 3), uid="score")
    data = score.to_dict()

    assert Timeline.from_dict(json.loads(json.dumps(data))).to_dict() == data


def test_a_length_of_null_reads_as_unknown_and_is_always_written() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "stream", "unit": "seconds", "length": None}],
    }

    timeline = Timeline.from_dict(copy.deepcopy(data))

    assert timeline.length is None
    assert timeline.to_dict(minimal=True) == data
    assert timeline.to_dict()["timelines"][0]["length"] is None


# region Measure-map rows and nested maps


def _quarters_root(**keys: Any) -> dict[str, Any]:
    return {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [12, 1], **keys}],
    }


def _full_row(count: int, **keys: Any) -> dict[str, Any]:
    """A 4/4 row as the library writes every row it keeps."""
    return {
        "count": count,
        "qstamp": [4 * (count - 1), 1],
        "number": count,
        "name": str(count),
        "time_signature": "4/4",
        "nominal_length": [4, 1],
        "actual_length": [4, 1],
        "start_repeat": False,
        "end_repeat": False,
        **keys,
    }


def test_unknown_row_keys_are_kept_and_force_their_row() -> None:
    rows = [
        {"count": 1, "time_signature": "4/4"},
        {"count": 2, "x-row": "kept"},
        {"count": 3, "tempo": 96},
    ]
    score = Timeline.from_dict(_quarters_root(measure_map=copy.deepcopy(rows)))

    written = score.to_dict(minimal=True)

    assert written["timelines"][0]["measure_map"] == [
        _full_row(1, next=["2"]),
        _full_row(2, next=["3"], **{"x-row": "kept"}),
        _full_row(3, tempo=96),
    ]
    assert Timeline.from_dict(copy.deepcopy(written)).to_dict(minimal=True) == written
    plain = Timeline.from_dict(
        _quarters_root(
            measure_map=[{"count": 1, "time_signature": "4/4"}, {"count": 3}]
        )
    )
    assert score.measure_map == plain.measure_map
    assert plain.to_dict(minimal=True)["timelines"][0]["measure_map"] == [
        _full_row(1, next=["2"]),
        _full_row(3),
    ]


def test_a_nested_meter_map_keeps_its_unknown_keys() -> None:
    from timetoalign.maps import MetricalPositionMap, MetricMap

    score = ContinuousLogicalTimeline(length=Fraction(8), uid="score")
    score.add_conversion_map(
        MetricalPositionMap(
            MetricMap(
                starts=[Fraction(0), Fraction(4)],
                mcs=[1, 2],
                mns=["1", "2"],
                lengths=[Fraction(4), Fraction(4)],
            )
        )
    )
    data = score.to_dict(minimal=True)
    data["timelines"][0]["conversion_maps"][0]["meter_map"]["x-deep"] = 7

    written = Timeline.from_dict(copy.deepcopy(data)).to_dict(minimal=True)

    assert written["timelines"][0]["conversion_maps"][0]["meter_map"]["x-deep"] == 7
    assert written == data


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"measure_map": []},
            "MeasureMap document form must be a non-empty list of Measure Map rows",
        ),
        (
            {"measure_map": [{"time_signature": "4/4"}]},
            "Measure Map row 1 carries neither count nor ID",
        ),
        (
            {"measure_map": [{"count": 1, "time_signature": "4/4", "qstamp": -1}]},
            "Measure Map row 1 states a negative qstamp -1",
        ),
        (
            {
                "measure_map": [
                    {"count": 1, "time_signature": "4/4", "actual_length": -1}
                ]
            },
            "Measure Map row 1 states actual_length -1, which is not positive",
        ),
        (
            {
                "conversion_maps": [
                    {
                        "type": "LinearMap",
                        "id": "lmap_1",
                        "source_unit": "quarters",
                        "target_unit": "seconds",
                        "offset": 0,
                    }
                ]
            },
            "a LinearMap lacks the key(s) ['scalar']",
        ),
    ],
    ids=["empty", "no_count_or_id", "negative_qstamp", "negative_length", "map_key"],
)
def test_rows_and_maps_a_manifest_refuses(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(_quarters_root(**change))
    assert str(refused.value) == message


# endregion


# region Lengths and ids


def _parent_with(child: dict[str, Any], *, locked: bool) -> dict[str, Any]:
    return _quarters_root(length=[10, 1], locked=locked, children=[child])


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    ("child", "message"),
    [
        (
            {"id": "coda", "unit": "quarters", "offset": [8, 1], "length": [4, 1]},
            "child 'coda' ends at [12, 1], beyond the length [10, 1] of its parent 'score'",
        ),
        (
            {"id": "late", "unit": "quarters", "offset": [12, 1], "length": None},
            "offset [12, 1] of 'late' is beyond the length [10, 1] of its parent 'score'",
        ),
    ],
    ids=["end", "start"],
)
def test_reading_never_changes_a_length(
    child: dict[str, Any], message: str, locked: bool
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(_parent_with(child, locked=locked))
    assert str(refused.value) == message

    fits = {"id": "coda", "unit": "quarters", "offset": [6, 1], "length": [4, 1]}
    score = Timeline.from_dict(_parent_with(fits, locked=locked))
    for form in (score.to_dict(), score.to_dict(minimal=True)):
        assert form["timelines"][0]["length"] == [10, 1]


@pytest.mark.parametrize(
    ("children", "message"),
    [
        (
            [{"id": "take", "unit": "seconds", "offset": 50.0, "length": 12.5}],
            "child 'take' ends at 62.5, beyond the length 60.0 of its parent 'rec'",
        ),
        (
            [{"id": "take", "unit": "seconds", "offset": 75.0, "length": None}],
            "offset 75.0 of 'take' is beyond the length 60.0 of its parent 'rec'",
        ),
        (
            [
                {"id": "a", "unit": "seconds", "offset": 0.0, "length": None},
                {"id": "b", "unit": "seconds", "offset": 61.0, "length": 1.0},
            ],
            "child 'a' ends at 61.0, beyond the length 60.0 of its parent 'rec'",
        ),
    ],
    ids=["end", "offset", "unknown_segment"],
)
def test_a_child_beyond_its_parent_is_refused_in_the_validators_words(
    children: list[dict[str, Any]], message: str
) -> None:
    root = {"id": "rec", "unit": "seconds", "length": 60.0, "children": children}
    if len(children) == 2:
        root["is_segment_line"] = True
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict({"timeskeleton": "0.7.0", "timelines": [root]})
    assert str(refused.value) == message


def _reserved(word: str) -> str:
    return f"Invalid id '{word}': it is reserved for the parts of an address"


@pytest.mark.parametrize(
    ("data", "word"),
    [
        (
            _quarters_root()
            | {"timelines": [{"id": "start", "unit": "quarters", "length": [4, 1]}]},
            "start",
        ),
        (
            _quarters_root(regions=[{"id": "end", "start": [0, 1], "end": [4, 1]}]),
            "end",
        ),
        (_quarters_root() | {"id": "card"}, "card"),
        (_quarters_root() | {"groups": [{"id": "name", "members": ["score"]}]}, "name"),
        (
            _quarters_root(
                conversion_maps=[
                    {
                        "type": "ShiftMap",
                        "id": "label",
                        "source_unit": None,
                        "target_unit": None,
                        "offset": 1,
                    }
                ]
            ),
            "label",
        ),
    ],
    ids=["root", "region", "manifest", "group", "map"],
)
def test_a_part_word_is_no_id(data: dict[str, Any], word: str) -> None:
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(copy.deepcopy(data))
    assert str(refused.value) == _reserved(word)


def test_a_part_word_is_no_override() -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(_quarters_root(), timeline_id="start")
    assert str(refused.value) == _reserved("start")
    for data in (
        _quarters_root(),
        _quarters_root() | {"groups": [{"id": "g", "members": ["score"]}]},
    ):
        with pytest.raises(ValueError) as refused:
            TimelineGroup.from_dict(copy.deepcopy(data), group_id="end")
        assert str(refused.value) == _reserved("end")


@pytest.mark.parametrize(
    ("timeline", "message"),
    [
        (
            {"id": "x", "unit": "quarters", "length": [-12, 1]},
            "length -12 of 'x' is negative",
        ),
        (
            {"id": "x", "unit": "seconds", "length": -8.0},
            "length -8 of 'x' is negative",
        ),
    ],
    ids=["fraction", "float"],
)
def test_a_negative_length_is_refused(timeline: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict({"timeskeleton": "0.7.0", "timelines": [timeline]})
    assert str(refused.value) == message
    with pytest.raises(ValueError) as refused:
        ContinuousLogicalTimeline(length=Fraction(-12), uid="x")
    assert str(refused.value) == "length -12 of 'x' is negative"


@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
@pytest.mark.parametrize(
    ("unit", "length", "offset", "spelled"),
    [
        ("quarters", [10, 1], [-5, 2], "[-5, 2]"),
        ("samples", 10, -1, "-1"),
        ("seconds", 10.0, -2.5, "-2.5"),
    ],
    ids=["fraction", "int", "float"],
)
def test_a_negative_child_offset_is_not_supported_yet(
    reader: str, unit: str, length: Any, offset: Any, spelled: str
) -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [
            {
                "id": "root",
                "unit": unit,
                "length": length,
                "children": [
                    {"id": "neg", "unit": unit, "length": length, "offset": offset}
                ],
            }
        ],
    }
    with pytest.raises(ValueError) as refused:
        READERS[reader](data)
    assert str(refused.value) == (
        f"A negative child offset is not supported yet: child 'neg' has offset {spelled}."
    )


# endregion


# region The envelope of a group or a timeline


def test_a_group_made_from_one_timeline_keeps_the_manifests_keys() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "id": "op",
        "name": "Op",
        "meta": {"k": 1},
        "x-top": 1,
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"styles": {"a": {"fill": "red"}}},
    }

    group = TimelineGroup.from_dict(copy.deepcopy(data), group_id="g")

    assert group.to_dict(minimal=True) == {
        "timeskeleton": "0.7.0",
        "meta": {"k": 1},
        "x-top": 1,
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "groups": [{"id": "g", "members": ["r"]}],
        "graphical": {"order": {"g": ["r"]}, "styles": {"a": {"fill": "red"}}},
    }


@pytest.mark.parametrize(
    "reader",
    [Timeline.from_dict, AlignmentBundle.from_dict],
    ids=["timeline", "bundle"],
)
def test_a_graphical_object_read_is_written_back(reader: Any) -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
    }

    read = reader(copy.deepcopy({**data, "graphical": {}}))

    assert read.to_dict(minimal=True) == {**data, "graphical": {}}
    assert read.to_dict()["graphical"] == {"order": {}, "statements": [], "styles": {}}
    assert "graphical" not in reader(copy.deepcopy(data)).to_dict()


def test_mixed_group_arguments_raise() -> None:
    data = {
        "timeskeleton": "0.7.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "name": "G", "members": ["score"]}],
    }

    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(copy.deepcopy(data), group_id="g", name="Other")

    assert str(refused.value) == (
        "the manifest holds no group with id 'g', name 'Other'; its groups: 'g' (name 'G')"
    )
    group = TimelineGroup.from_dict(copy.deepcopy(data), group_id="g", name="G")
    assert (group.id, group.name) == ("g", "G")


# endregion
