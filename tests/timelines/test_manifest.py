"""TimeToShare manifests: the one shape every reader and writer shares.

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
import timetoshare
from timetoshare import FORMAT_CONTEXT, complete, minimal

from tests.helpers import bare, filed, reading, refusal
from timetoalign.alignment import AlignmentBundle
from timetoalign.core import Coordinate, NumberType, SupportPolicy, TimeUnit
from timetoalign.maps import LinearMap
from timetoalign.timelines import Timeline, TimelineGroup
from timetoalign.timelines.structure import MeasureMap
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

READERS = {
    "timeline": reading(Timeline.from_dict),
    "group": lambda data: TimelineGroup.from_dict(filed(data), group_id="g"),
    "bundle": reading(AlignmentBundle.from_dict),
}


def _text(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


# region The version marker


@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
def test_a_manifest_without_the_marker_is_refused(reader: str) -> None:
    with pytest.raises(ValueError) as refused:
        READERS[reader]({"timelines": []})
    assert refusal(refused.value) == (
        "not_a_manifest",
        "#",
        "not a TimeToShare manifest",
    )


@pytest.mark.parametrize("reader", list(READERS), ids=list(READERS))
def test_another_version_is_refused_naming_the_current_one(reader: str) -> None:
    with pytest.raises(timetoshare.UnsupportedVersionError) as refused:
        READERS[reader]({"timetoshare": "0.1.0", "timelines": []})
    assert str(refused.value) == (
        'unsupported TimeToShare version "0.1.0"; this package reads 0.2.0 only'
    )


def test_the_marker_is_the_format_s_version() -> None:
    assert timetoshare.SCHEMA_VERSION == "0.2.0"
    assert bare(AlignmentBundle().to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [],
    }


# endregion


# region Defaults


MINIMAL: dict[str, Any] = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
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
                }
            ],
            "regions": [{"id": "theme", "start": [0, 1], "end": [8, 1]}],
            "flow_control": [
                {
                    "id": "segno",
                    "type": "jump_to",
                    "manifestation": "segno",
                    "coordinate": [4, 1],
                },
                {
                    "id": "back",
                    "type": "jump_from",
                    "manifestation": "repeat_end",
                    "coordinate": [16, 1],
                },
                {
                    "id": "fine",
                    "type": "flow_marker",
                    "manifestation": "section_break",
                    "coordinate": [32, 1],
                },
            ],
        }
    ],
}

COMPLETE: dict[str, Any] = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
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
                    "flow_control": [],
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
            "flow_control": [
                {
                    "id": "segno",
                    "type": "jump_to",
                    "manifestation": "segno",
                    "coordinate": [4, 1],
                    "name": None,
                    "label": None,
                    "meta": {},
                },
                {
                    "id": "back",
                    "type": "jump_from",
                    "manifestation": "repeat_end",
                    "coordinate": [16, 1],
                    "repeat_count": 1,
                    "name": None,
                    "label": None,
                    "meta": {},
                },
                {
                    "id": "fine",
                    "type": "flow_marker",
                    "manifestation": "section_break",
                    "coordinate": [32, 1],
                    "name": None,
                    "label": None,
                    "meta": {},
                },
            ],
        }
    ],
    "groups": [],
    "measure_maps": [],
    "resources": [],
}


def test_minimal_and_complete_read_to_equal_objects() -> None:
    from_minimal = Timeline.from_dict(filed(copy.deepcopy(MINIMAL)))
    from_complete = Timeline.from_dict(filed(copy.deepcopy(COMPLETE)))

    assert (
        bare(from_minimal.to_dict(minimal=False))
        == bare(from_complete.to_dict(minimal=False))
        == COMPLETE
    )
    assert (
        bare(from_minimal.to_dict(minimal=True))
        == bare(from_complete.to_dict(minimal=True))
        == MINIMAL
    )
    assert _text(bare(from_complete.to_dict(minimal=False))) == _text(COMPLETE)
    assert from_minimal.get_region("theme") == from_complete.get_region("theme")
    assert from_minimal.flow_control.breaks == from_complete.flow_control.breaks
    assert from_minimal.flow_control.jumps == from_complete.flow_control.jumps
    assert from_minimal.flow_control.markers == from_complete.flow_control.markers


@pytest.mark.parametrize("reader", sorted(READERS))
def test_to_dict_writes_the_minimal_form_by_default(reader: str) -> None:
    read = READERS[reader](copy.deepcopy(COMPLETE))
    assert bare(read.to_dict()) == bare(read.to_dict(minimal=True))
    assert bare(read.to_dict()) != bare(read.to_dict(minimal=False))
    if reader == "timeline":
        assert bare(read.to_dict()) == MINIMAL


ELEMENTS: dict[str, tuple[Any, str, int | None]] = {
    # element kind: (the element of the timeline read, its key, its index)
    "region": (lambda t: t.get_region("theme"), "regions", 0),
    "conversion map": (lambda t: t.get_conversion_map("tempo"), "conversion_maps", 0),
    "marker": (lambda t: t.flow_control.markers[0], "flow_control", 0),
    "jump": (lambda t: t.flow_control.jumps[0], "flow_control", 1),
    "break": (lambda t: t.flow_control.breaks[0], "flow_control", 2),
    "flow control registry": (lambda t: t.flow_control, "flow_control", None),
}


@pytest.mark.parametrize("kind", list(ELEMENTS))
def test_every_element_writer_writes_the_minimal_form_by_default(kind: str) -> None:
    element_of, key, index = ELEMENTS[kind]
    element = element_of(Timeline.from_dict(filed(copy.deepcopy(COMPLETE))))
    minimal = MINIMAL["timelines"][0][key]
    complete = COMPLETE["timelines"][0][key]
    if index is not None:
        minimal, complete = minimal[index], complete[index]

    assert element.to_dict() == minimal
    assert element.to_dict(minimal=False) == complete


def test_the_minimal_form_compares_a_default_by_value() -> None:
    timeline = ContinuousPhysicalTimeline(length=10.0, uid="take")
    timeline.add_conversion_map(
        LinearMap(
            scalar=2.0,
            offset=0.0,
            source_unit="seconds",
            target_unit="seconds",
            uid="double",
        )
    )
    written = bare(timeline.to_dict())
    complete = bare(timeline.to_dict(minimal=False))
    complete_map = {
        "type": "LinearMap",
        "id": "double",
        "name": None,
        "source_unit": "seconds",
        "target_unit": "seconds",
        "scalar": 2.0,
        "offset": 0.0,
    }
    minimal_map = {
        "type": "LinearMap",
        "id": "double",
        "source_unit": "seconds",
        "target_unit": "seconds",
        "scalar": 2.0,
    }

    assert written == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": timetoshare.SCHEMA_VERSION,
        "timelines": [
            {
                "id": "take",
                "unit": "seconds",
                "length": 10.0,
                "conversion_maps": [minimal_map],
            }
        ],
    }
    assert json.dumps(written) == json.dumps(minimal(complete))
    assert json.dumps(complete["timelines"][0]["conversion_maps"]) == json.dumps(
        [complete_map]
    )
    assert timeline.get_conversion_map("double").to_dict() == minimal_map
    reread = bare(Timeline.from_dict(filed(written)).to_dict(minimal=False))
    assert json.dumps(reread["timelines"][0]["conversion_maps"]) == json.dumps(
        [{**complete_map, "offset": 0}]
    )


GROUP_OBJECTS = {
    "plain": (
        {},
        {"id": "g", "members": ["perf", "score"]},
        {"id": "g", "name": None, "meta": {}, "members": ["perf", "score"]},
    ),
    "named": (
        {"name": "Study", "meta": {"source": "an edition"}},
        {
            "id": "g",
            "name": "Study",
            "meta": {"source": "an edition"},
            "members": ["perf", "score"],
        },
        {
            "id": "g",
            "name": "Study",
            "meta": {"source": "an edition"},
            "members": ["perf", "score"],
        },
    ),
}
"""Each group's keyword arguments, then its object in the minimal and complete form."""


def _grouped(writer: str, members: dict[str, Any]) -> Any:
    """A group ``g`` of a score and a performance, alone or in a bundle."""
    group = TimelineGroup(
        id="g",
        timelines=[
            ContinuousLogicalTimeline(length=Fraction(8), uid="score"),
            ContinuousPhysicalTimeline(length=4.0, uid="perf"),
        ],
        **members,
    )
    if writer == "group":
        return group
    return AlignmentBundle().add_group(group)


@pytest.mark.parametrize("writer", ["group", "bundle"])
@pytest.mark.parametrize("case", list(GROUP_OBJECTS), ids=list(GROUP_OBJECTS))
def test_a_group_takes_its_form_with_the_whole_manifest(writer: str, case: str) -> None:
    members, minimal_group, complete_group = GROUP_OBJECTS[case]
    written = _grouped(writer, members)

    minimal_form = bare(written.to_dict(minimal=True))
    complete_form = bare(written.to_dict(minimal=False))

    assert minimal_form["groups"] == [minimal_group]
    assert complete_form["groups"] == [complete_group]
    assert _text(minimal_form) == _text(minimal(complete_form))
    assert _text(complete_form) == _text(complete(minimal_form))


def test_a_reader_supplies_every_default() -> None:
    timeline = Timeline.from_dict(filed(copy.deepcopy(MINIMAL)))
    (brk,) = timeline.flow_control.breaks
    (jump,) = timeline.flow_control.jumps

    assert (timeline.name, timeline.meta, timeline.is_locked) == ("score", {}, False)
    assert timeline.number_type is NumberType.fraction
    assert timeline.measure_map is None and timeline.metric_hierarchy is None
    assert (brk.control_type.value, brk.condition.value) == ("section_break", "always")
    assert (brk.label, brk.name, brk.meta) == (None, None, {})
    assert (jump.control_type.value, jump.condition.value, jump.repeat_count) == (
        "repeat_end",
        "first_n",
        1,
    )
    assert jump.to_coordinate == Coordinate(0, TimeUnit.quarters)
    assert (jump.target, jump.name, jump.label, jump.meta) == (None, None, None, {})
    assert timeline.get_conversion_map("tempo").to_dict(minimal=False)["name"] is None


UNIT_DEFAULTS = {
    "ticks": ("int", 4),
    "samples": ("int", 4),
    "frames": ("int", 4),
    "pixels": ("int", 4),
    "quarters": ("fraction", [4, 1]),
    "whole_notes": ("fraction", [4, 1]),
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
    "css_pixels": ("float", 4.5),
    "user_units": ("float", 4.5),
}


def test_the_unit_defaults_cover_every_unit() -> None:
    assert sorted(UNIT_DEFAULTS) == sorted(unit.value for unit in TimeUnit)


@pytest.mark.parametrize("unit", list(UNIT_DEFAULTS), ids=list(UNIT_DEFAULTS))
def test_a_number_type_left_out_is_the_units_own(unit: str) -> None:
    number_type, length = UNIT_DEFAULTS[unit]
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }

    timeline = Timeline.from_dict(filed(data))

    assert timeline.number_type is NumberType[number_type]
    assert bare(timeline.to_dict(minimal=True)) == data
    assert (
        bare(timeline.to_dict(minimal=False))["timelines"][0]["number_type"]
        == number_type
    )


def test_the_class_comes_from_the_unit_and_a_class_key_is_kept() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "perf",
                "class": "ContinuousGraphicalTimeline",
                "unit": "seconds",
                "length": 12.5,
            }
        ],
    }

    timeline = Timeline.from_dict(filed(copy.deepcopy(data)))

    assert type(timeline) is ContinuousPhysicalTimeline
    assert bare(timeline.to_dict(minimal=True)) == data


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
            ("flow_control", 2, "coordinate"),
            32,
            "coordinate 32 of 'fine' is an int on a fraction axis: write [32, 1]",
        ),
        (
            ("flow_control", 0, "coordinate"),
            4,
            "coordinate 4 of 'segno' is an int on a fraction axis: write [4, 1]",
        ),
        (
            ("flow_control", 1, "coordinate"),
            0.25,
            "coordinate 0.25 of 'back' is a float on a fraction axis: write [1, 4]",
        ),
    ],
    ids=[
        "length",
        "root_offset",
        "child_offset",
        "region_start",
        "region_end",
        "flow_marker",
        "jump_to",
        "jump_from",
    ],
)
def test_each_bare_key_checks_its_representation(
    path: tuple[Any, ...], value: Any, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(_with_value(path, value)))
    assert refusal(refused.value) == (
        "representation",
        timetoshare.json_pointer(("timelines", 0, *path)),
        message,
    )


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
        ("seconds", "8", "'8' is not of type 'number'"),
        ("quarters", [1, 8.5], "8.5 is not of type 'integer'"),
        (
            "ticks",
            [480, 1.0],
            "length [480, 1.0] of 't' is a fraction on an int axis: write 480",
        ),
    ],
    ids=[
        "fraction_on_float",
        "inexact_fraction_on_float",
        "float_on_fraction",
        "integral_pair_on_int",
        "pair_on_int",
        "float_on_int",
        "string",
        "non_integral_member",
        "respelled_pair_on_int",
    ],
)
def test_every_axis_kind_checks_its_representation(
    unit: str, length: Any, message: str
) -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed(data))
    code = (
        "schema" if isinstance(length, str) or length == [1, 8.5] else "representation"
    )
    # A member of a fraction is refused where it lies, every other value as the length.
    path = "#/timelines/0/length/1" if length == [1, 8.5] else "#/timelines/0/length"
    assert refusal(refused.value) == (code, path, message)


@pytest.mark.parametrize(
    ("unit", "length", "expected"),
    [
        ("ticks", 480.0, 480),
        ("seconds", 8, 8.0),
        ("quarters", [1, 8.0], Fraction(1, 8)),
    ],
    ids=["integral_number_on_int", "integer_on_float", "integral_float_member"],
)
def test_numbers_javascript_cannot_tell_apart_are_accepted(
    unit: str, length: Any, expected: Any
) -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "t", "unit": unit, "length": length}],
    }

    value = Timeline.from_dict(filed(data)).length.value

    assert (value, type(value)) == (expected, type(expected))


# endregion


# region Unknown keys


UNKNOWN: dict[str, Any] = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
    "$schema": "https://timetoalign.github.io/timetoshare/0.2.0/timetoshare.schema.json",
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
                    "type": "MeasureRecordMap",
                    "id": "bars",
                    "source_unit": "quarters",
                    "target_unit": None,
                    "measure_map": "score_bars",
                    "x-note": "record map",
                },
                {
                    "type": "ChainMap",
                    "id": "chain",
                    "source_unit": None,
                    "target_unit": None,
                    "x-note": "map",
                    "maps": [
                        {
                            "type": "LinearMap",
                            "id": "shift",
                            "source_unit": None,
                            "target_unit": None,
                            "offset": 1,
                            "x-note": "nested map",
                        }
                    ],
                },
            ],
            "regions": [
                {"id": "theme", "start": [0, 1], "end": [8, 1], "x-note": "region"}
            ],
            "flow_control": [
                {
                    "id": "segno",
                    "type": "jump_to",
                    "manifestation": "segno",
                    "coordinate": [4, 1],
                    "x-note": "jump_to",
                },
                {
                    "id": "back",
                    "type": "jump_from",
                    "manifestation": "repeat_end",
                    "coordinate": [16, 1],
                    "x-note": "jump_from",
                },
                {
                    "id": "fine",
                    "type": "flow_marker",
                    "manifestation": "section_break",
                    "coordinate": [32, 1],
                    "x-note": "flow_marker",
                },
            ],
        },
    ],
    "groups": [{"id": "g", "members": ["coda", "perf"], "x-note": "group"}],
    "measure_maps": [
        {
            "id": "score_bars",
            "x-note": "measure map",
            "rows": [
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
                }
            ],
        }
    ],
    "graphical": {"order": {"g": ["perf", "coda"]}, "x-note": "graphical"},
}


def test_unknown_keys_are_written_back_where_they_were_found() -> None:
    bundle = AlignmentBundle.from_dict(filed(copy.deepcopy(UNKNOWN)))

    assert bare(bundle.to_dict(minimal=True)) == UNKNOWN
    assert (
        bare(AlignmentBundle.from_dict(bundle.to_dict()).to_dict(minimal=True))
        == UNKNOWN
    )


def test_the_group_reader_keeps_the_unknown_keys_of_what_it_reads() -> None:
    group = TimelineGroup.from_dict(filed(copy.deepcopy(UNKNOWN)))

    assert bare(group.to_dict(minimal=True)) == UNKNOWN


def test_the_timeline_reader_keeps_the_unknown_keys_of_what_it_reads() -> None:
    score = Timeline.from_dict(filed(copy.deepcopy(UNKNOWN)), timeline_id="score")

    assert bare(score.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [UNKNOWN["timelines"][1]],
        "measure_maps": UNKNOWN["measure_maps"],
    }


def test_a_one_root_manifest_keeps_its_envelope() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "id": "op35",
        "name": "Eroica Variations",
        "meta": {"opus": 35},
        "x-note": "manifest",
        "timelines": [{"id": "score", "unit": "quarters", "length": [32, 1]}],
        "graphical": {"styles": {"thin": {"stroke-width": 1}}},
    }

    assert (
        bare(Timeline.from_dict(filed(copy.deepcopy(data))).to_dict(minimal=True))
        == data
    )


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
    return {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [root],
    }


def test_a_negative_root_offset_round_trips() -> None:
    data = _offset_manifest(-5.5)

    timeline = Timeline.from_dict(filed(copy.deepcopy(data)))

    assert timeline.root_offset == Coordinate(-5.5, TimeUnit.seconds)
    assert bare(timeline.to_dict(minimal=True)) == data
    assert bare(timeline.to_dict())["timelines"][0]["offset"] == -5.5


def test_a_root_offset_changes_nothing_computed() -> None:
    placed = Timeline.from_dict(filed(_offset_manifest(-5.5)))
    plain = Timeline.from_dict(filed(_offset_manifest(None)))

    assert plain.root_offset is None
    assert placed.get_child_offset("take") == plain.get_child_offset("take")
    for position in (0.0, 2.0, 3.5, 10.0):
        assert placed.get_timestamp_at(position).coordinates == (
            plain.get_timestamp_at(position).coordinates
        )


# endregion


# region Selection


SELECTION: dict[str, Any] = {
    "timetoshare": "0.2.0",
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
    part = Timeline.from_dict(filed(copy.deepcopy(SELECTION)), timeline_id="part")
    perf = Timeline.from_dict(filed(copy.deepcopy(SELECTION)), name="Performance")

    assert (part.id, part.length) == (
        "part",
        Coordinate(Fraction(16), TimeUnit.quarters),
    )
    assert bare(part.to_dict(minimal=True))["timelines"] == [
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
        Timeline.from_dict(filed(copy.deepcopy(SELECTION)), **arguments)
    assert str(refused.value) == message


def test_a_one_root_manifest_is_read_whole_with_overrides() -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    timeline = Timeline.from_dict(filed(data), timeline_id="edition", name="Edition")

    assert (timeline.id, timeline.name) == ("edition", "Edition")


FAMILY: dict[str, Any] = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
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
    child = Timeline.from_dict(filed(copy.deepcopy(FAMILY)), **arguments)

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
    timeline = Timeline.from_dict(filed(copy.deepcopy(FAMILY)), **arguments)

    assert timeline._parent is None
    assert bare(timeline.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
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
        Timeline.from_dict(filed(copy.deepcopy(data)), timeline_id=timeline_id)
    assert str(refused.value) == message


TWO_ROOTS: dict[str, Any] = {
    "timetoshare": "0.2.0",
    "timelines": [
        {"id": "score", "unit": "quarters", "length": [8, 1]},
        {"id": "perf", "unit": "seconds", "length": 60.0},
    ],
}

ONE_ROOT_ONE_GROUP: dict[str, Any] = {
    "timetoshare": "0.2.0",
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
        Timeline.from_dict(filed(copy.deepcopy(data)), **arguments)
    assert str(refused.value) == message


def test_a_group_is_selected_by_id() -> None:
    group = TimelineGroup.from_dict(filed(copy.deepcopy(SELECTION)), group_id="g1")

    assert (group.id, group.name, group.timeline_ids) == ("g1", "pair", ["score"])
    assert bare(group.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
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
        TimelineGroup.from_dict(filed(copy.deepcopy(SELECTION)), **arguments)
    assert str(refused.value) == message


ONE_TIMELINE = {
    "timetoshare": "0.2.0",
    "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
}


def test_a_manifest_of_one_timeline_makes_a_group_when_asked() -> None:
    group = TimelineGroup.from_dict(
        filed(copy.deepcopy(ONE_TIMELINE)), group_id="edition", name="Edition"
    )

    assert (group.id, group.name, group.timeline_ids) == (
        "edition",
        "Edition",
        ["score"],
    )
    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(filed(copy.deepcopy(ONE_TIMELINE)))
    assert str(refused.value) == (
        "the manifest holds one timeline and no group: give group_id or name to make a group of it"
    )


def test_a_one_group_manifest_is_read_whole_with_overrides() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "meta": {"source": "an edition"},
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "name": "old", "members": ["score"]}],
    }

    group = TimelineGroup.from_dict(
        filed(copy.deepcopy(data)), group_id="edition", name="Edition"
    )

    assert (group.id, group.name) == ("edition", "Edition")
    assert bare(group.to_dict(minimal=True)) == {
        **data,
        "groups": [{"id": "edition", "name": "Edition", "members": ["score"]}],
        "graphical": {"order": {"edition": ["score"]}},
    }
    assert bare(
        TimelineGroup.from_dict(filed(copy.deepcopy(data))).to_dict(minimal=True)
    ) == {
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
        TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="ex_child")
    assert str(refused.value) == (
        "group id 'ex_child' is already held by timeline 'ex_child'"
    )


def test_a_group_id_matching_the_one_group_reads_it_unchanged() -> None:
    data = {**FAMILY, "groups": [{"id": "g", "name": "pair", "members": ["ex"]}]}

    group = TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="g")

    assert bare(group.to_dict(minimal=True)) == {
        **data,
        "graphical": {"order": {"g": ["ex"]}},
    }


def test_a_named_one_group_manifest_needs_a_selection() -> None:
    data = {
        "timetoshare": "0.2.0",
        "name": "Study",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "members": ["score"]}],
    }

    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(filed(copy.deepcopy(data)))
    assert str(refused.value) == (
        "the manifest holds several groups or is named: give group_id or name to select a group"
    )
    assert TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="g").id == "g"


# endregion


# region The manifest's identity


def test_an_anonymous_bundle_writes_no_id() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(filed(copy.deepcopy(data)))

    assert bundle.id == "bundle_1"
    assert bare(bundle.to_dict(minimal=True)) == data


def test_a_manifest_with_a_name_and_no_id() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "name": "Eroica Variations",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(filed(copy.deepcopy(data)))

    assert (bundle.name, bare(bundle.to_dict(minimal=True))) == (
        "Eroica Variations",
        data,
    )
    assert (
        bare(Timeline.from_dict(filed(copy.deepcopy(data))).to_dict(minimal=True))
        == data
    )


def test_a_support_policy_key_is_not_read() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "support_policy": "clamp",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    bundle = AlignmentBundle.from_dict(filed(copy.deepcopy(data)))

    assert bundle.support_policy is SupportPolicy.omit
    assert bare(bundle.to_dict(minimal=True)) == data


def test_the_manifest_id_is_unique_with_the_timeline_ids() -> None:
    data = {
        "timetoshare": "0.2.0",
        "id": "score",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
    }

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))
    assert refusal(refused.value) == (
        "manifest_id",
        "#/id",
        "'score' is not unique in the manifest, first at #/timelines/0/id",
    )


# endregion


# region graphical.order


def test_the_order_is_an_object() -> None:
    data = copy.deepcopy(SELECTION)
    data["graphical"] = {"order": [["score"], ["perf"]]}

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))
    assert refusal(refused.value) == (
        "schema",
        "#/graphical/order",
        "[['score'], ['perf']] is not of type 'object'",
    )


def test_an_order_naming_no_group_or_timeline_is_refused() -> None:
    data = copy.deepcopy(SELECTION)
    data["graphical"] = {"order": {"g3": ["perf"]}}

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))
    assert refusal(refused.value) == (
        "order_key",
        "#/graphical/order/g3",
        "'g3' names no group or timeline of this manifest",
    )


SCORE_BANDS = [
    "children",
    "markers",
    "line",
    "conversion_maps",
    "regions",
    "flow_control",
]
"""Markers above the line, children beyond them, the rest below."""

PART_BANDS = [
    "flow_control",
    "regions",
    "markers",
    "conversion_maps",
    "children",
    "line",
]
"""Every band above the line, in reverse default order."""

PERF_BANDS = [
    "line",
    "flow_control",
    "regions",
    "markers",
    "conversion_maps",
    "children",
]
"""Every band below the line, in reverse default order."""

SCORE_MINIMAL: dict[str, Any] = {
    "id": "score",
    "unit": "quarters",
    "length": [8, 1],
    "children": [
        {"id": "part", "unit": "quarters", "offset": [2, 1], "length": [4, 1]}
    ],
}
PERF_MINIMAL: dict[str, Any] = {"id": "perf", "unit": "seconds", "length": 20.0}


def _empty_complete(
    timeline_id: str, unit: str, number_type: str, **keys: Any
) -> dict[str, Any]:
    """A timeline object holding nothing, every default of the format written."""
    return {
        "id": timeline_id,
        "name": None,
        "unit": unit,
        "number_type": number_type,
        "locked": False,
        "is_segment_line": False,
        "meta": {},
        "children": [],
        "conversion_maps": [],
        "regions": [],
        "flow_control": [],
        **keys,
    }


SCORE_COMPLETE = _empty_complete(
    "score",
    "quarters",
    "fraction",
    length=[8, 1],
    children=[
        _empty_complete("part", "quarters", "fraction", offset=[2, 1], length=[4, 1])
    ],
)
PERF_COMPLETE = _empty_complete("perf", "seconds", "float", length=20.0)

ONE_ROOT_BANDS = {"part": PART_BANDS, "score": SCORE_BANDS}
GROUPED_BANDS = {
    "g": ["perf", "part"],
    "part": PART_BANDS,
    "perf": PERF_BANDS,
    "score": SCORE_BANDS,
}

ONE_ROOT = {
    "minimal": {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [SCORE_MINIMAL],
        "graphical": {"order": ONE_ROOT_BANDS},
    },
    "complete": {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "name": None,
        "meta": {},
        "timelines": [SCORE_COMPLETE],
        "groups": [],
        "measure_maps": [],
        "resources": [],
        "graphical": {
            "order": ONE_ROOT_BANDS,
            "elisions": [],
            "styles": {},
            "rules": [],
            "timestamps": [],
            "matchstamps": [],
        },
    },
}
GROUPED = {
    "minimal": {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [PERF_MINIMAL, SCORE_MINIMAL],
        "groups": [{"id": "g", "members": ["part", "perf"]}],
        "graphical": {"order": GROUPED_BANDS},
    },
    "complete": {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "name": None,
        "meta": {},
        "timelines": [PERF_COMPLETE, SCORE_COMPLETE],
        "groups": [{"id": "g", "name": None, "meta": {}, "members": ["part", "perf"]}],
        "measure_maps": [],
        "resources": [],
        "graphical": {
            "order": GROUPED_BANDS,
            "elisions": [],
            "styles": {},
            "rules": [],
            "timestamps": [],
            "matchstamps": [],
        },
    },
}
WHOLE_READS = {
    "timeline": (reading(Timeline.from_dict), ONE_ROOT),
    "group": (reading(TimelineGroup.from_dict), GROUPED),
    "bundle_one_root": (reading(AlignmentBundle.from_dict), ONE_ROOT),
    "bundle_grouped": (reading(AlignmentBundle.from_dict), GROUPED),
}


@pytest.mark.parametrize("form", ["complete", "minimal"])
@pytest.mark.parametrize("case", list(WHOLE_READS), ids=list(WHOLE_READS))
def test_band_orders_survive_a_round_trip(case: str, form: str) -> None:
    reader, literals = WHOLE_READS[case]
    other = "minimal" if form == "complete" else "complete"

    read = reader(copy.deepcopy(literals[form]))

    assert _text(bare(read.to_dict(minimal=form == "minimal"))) == _text(literals[form])
    assert _text(bare(read.to_dict(minimal=other == "minimal"))) == _text(
        literals[other]
    )


@pytest.mark.parametrize("case", list(WHOLE_READS), ids=list(WHOLE_READS))
def test_band_orders_are_copied_on_reading(case: str) -> None:
    reader, literals = WHOLE_READS[case]
    data = copy.deepcopy(literals["minimal"])

    read = reader(data)
    for entry in data["graphical"]["order"].values():
        entry.reverse()

    assert _text(bare(read.to_dict(minimal=True))) == _text(literals["minimal"])


def test_a_band_order_is_never_checked() -> None:
    data = copy.deepcopy(ONE_ROOT["minimal"])
    data["graphical"]["order"]["part"] = ["line", "markers"]

    assert (
        bare(Timeline.from_dict(filed(copy.deepcopy(data))).to_dict(minimal=True))
        == data
    )


def test_a_removed_timeline_takes_its_band_order_along() -> None:
    group = TimelineGroup.from_dict(filed(copy.deepcopy(GROUPED["minimal"])))

    group.remove_timeline("perf")

    assert bare(group.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [SCORE_MINIMAL],
        "groups": [{"id": "g", "members": ["part"]}],
        "graphical": {
            "order": {"g": ["part"], "part": PART_BANDS, "score": SCORE_BANDS}
        },
    }


def test_a_new_id_on_load_leaves_the_old_band_order_out() -> None:
    timeline = Timeline.from_dict(
        filed(copy.deepcopy(ONE_ROOT["minimal"])), timeline_id="edition"
    )

    assert bare(timeline.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [SCORE_MINIMAL | {"id": "edition"}],
        "graphical": {"order": {"part": PART_BANDS}},
    }


def test_a_selected_timeline_keeps_no_graphical() -> None:
    perf = Timeline.from_dict(
        filed(copy.deepcopy(GROUPED["minimal"])), timeline_id="perf"
    )

    assert bare(perf.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [PERF_MINIMAL],
    }


# endregion


def test_a_written_manifest_is_a_json_fixpoint() -> None:
    score = ContinuousLogicalTimeline(length=Fraction(10, 3), uid="score")
    data = bare(score.to_dict())

    assert (
        bare(Timeline.from_dict(filed(json.loads(json.dumps(data)))).to_dict()) == data
    )


def test_a_length_of_null_reads_as_unknown_and_is_always_written() -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "stream", "unit": "seconds", "length": None}],
    }

    timeline = Timeline.from_dict(filed(copy.deepcopy(data)))

    assert timeline.length is None
    assert bare(timeline.to_dict(minimal=True)) == data
    assert bare(timeline.to_dict())["timelines"][0]["length"] is None


# region Measure-map rows


RECORD_MAP = {
    "type": "MeasureRecordMap",
    "id": "record",
    "source_unit": "quarters",
    "target_unit": None,
    "measure_map": "bars",
}


def _quarters_root(
    rows: list[dict[str, Any]] | None = None, **keys: Any
) -> dict[str, Any]:
    """A quarters root of length 12, reading the measure map of *rows* if given."""
    manifest: dict[str, Any] = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [12, 1], **keys}],
    }
    if rows is not None:
        manifest["timelines"][0]["conversion_maps"] = [RECORD_MAP]
        manifest["measure_maps"] = [{"id": "bars", "rows": rows}]
    return manifest


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
    score = Timeline.from_dict(filed(_quarters_root(copy.deepcopy(rows))))

    written = bare(score.to_dict(minimal=True))

    assert written["measure_maps"][0]["rows"] == [
        _full_row(1, next=["2"]),
        _full_row(2, next=["3"], **{"x-row": "kept"}),
        _full_row(3, tempo=96),
    ]
    assert (
        bare(Timeline.from_dict(filed(copy.deepcopy(written))).to_dict(minimal=True))
        == written
    )
    plain = Timeline.from_dict(
        filed(_quarters_root([{"count": 1, "time_signature": "4/4"}, {"count": 3}]))
    )
    assert score.measure_map == plain.measure_map
    assert bare(plain.to_dict(minimal=True))["measure_maps"][0]["rows"] == [
        _full_row(1, next=["2"]),
        _full_row(3),
    ]


@pytest.mark.parametrize(
    ("manifest", "path", "message"),
    [
        (
            _quarters_root([]),
            "#/measure_maps/0/rows",
            "[] should be non-empty",
        ),
        (
            _quarters_root([{"time_signature": "4/4"}]),
            "#/measure_maps/0/rows/0",
            "'ID' is a required property",
        ),
        (
            _quarters_root([{"count": 1, "time_signature": "4/4", "qstamp": -1}]),
            "#/measure_maps/0/rows/0/qstamp",
            "-1 is less than the minimum of 0",
        ),
        (
            _quarters_root(
                [{"count": 1, "time_signature": "4/4", "actual_length": -1}]
            ),
            "#/measure_maps/0/rows/0/actual_length",
            "-1 is less than or equal to the minimum of 0",
        ),
        (
            _quarters_root(
                conversion_maps=[
                    {
                        "type": "TableMap",
                        "id": "tmap_1",
                        "source_unit": "quarters",
                        "target_unit": "seconds",
                        "y_values": [0.0, 1.0],
                    }
                ]
            ),
            "#/timelines/0/conversion_maps/0",
            "'x_values' is a required property",
        ),
    ],
    ids=["empty", "no_count_or_id", "negative_qstamp", "negative_length", "map_key"],
)
def test_rows_and_maps_a_manifest_refuses(
    manifest: dict[str, Any], path: str, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed(manifest))
    assert refusal(refused.value) == ("schema", path, message)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [],
            "the rows of measure map 'bars' must be a non-empty list of Measure Map rows",
        ),
        ([{"time_signature": "4/4"}], "Measure Map row 1 carries neither count nor ID"),
        (
            [{"count": 1, "time_signature": "4/4", "qstamp": -1}],
            "Measure Map row 1 states a negative qstamp -1",
        ),
        (
            [{"count": 1, "time_signature": "4/4", "actual_length": -1}],
            "Measure Map row 1 states actual_length -1, which is not positive",
        ),
    ],
    ids=["empty", "no_count_or_id", "negative_qstamp", "negative_length"],
)
def test_the_measure_map_reader_refuses_rows_the_format_refuses(
    rows: list[dict[str, Any]], message: str
) -> None:
    # The reader of one measure-map entry, outside a manifest, keeps these
    # rules of the format itself: nothing validates its input before.
    with pytest.raises(ValueError) as refused:
        MeasureMap.from_dict({"id": "bars", "rows": rows})
    assert str(refused.value) == message


# endregion


# region Lengths and ids


def _parent_with(child: dict[str, Any], *, locked: bool) -> dict[str, Any]:
    return _quarters_root(length=[10, 1], locked=locked, children=[child])


@pytest.mark.parametrize("locked", [False, True], ids=["unlocked", "locked"])
@pytest.mark.parametrize(
    ("child", "path", "message"),
    [
        (
            {"id": "coda", "unit": "quarters", "offset": [8, 1], "length": [4, 1]},
            "#/timelines/0/children/0/length",
            "child 'coda' ends at [12, 1], beyond the length [10, 1] of its parent 'score'",
        ),
        (
            {"id": "late", "unit": "quarters", "offset": [12, 1], "length": None},
            "#/timelines/0/children/0/offset",
            "offset [12, 1] of 'late' is beyond the length [10, 1] of its parent 'score'",
        ),
    ],
    ids=["end", "start"],
)
def test_reading_never_changes_a_length(
    child: dict[str, Any], path: str, message: str, locked: bool
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed(_parent_with(child, locked=locked)))
    code = "child_end" if child["length"] is not None else "child_offset"
    assert refusal(refused.value) == (code, path, message)

    fits = {"id": "coda", "unit": "quarters", "offset": [6, 1], "length": [4, 1]}
    score = Timeline.from_dict(filed(_parent_with(fits, locked=locked)))
    for form in (bare(score.to_dict()), bare(score.to_dict(minimal=True))):
        assert form["timelines"][0]["length"] == [10, 1]


@pytest.mark.parametrize(
    ("children", "path", "message"),
    [
        (
            [{"id": "take", "unit": "seconds", "offset": 50.0, "length": 12.5}],
            "#/timelines/0/children/0/length",
            "child 'take' ends at 62.5, beyond the length 60.0 of its parent 'rec'",
        ),
        (
            [{"id": "take", "unit": "seconds", "offset": 75.0, "length": None}],
            "#/timelines/0/children/0/offset",
            "offset 75.0 of 'take' is beyond the length 60.0 of its parent 'rec'",
        ),
        (
            [
                {"id": "a", "unit": "seconds", "offset": 0.0, "length": None},
                {"id": "b", "unit": "seconds", "offset": 61.0, "length": 1.0},
            ],
            "#/timelines/0/children/0/length",
            "child 'a' ends at 61.0, beyond the length 60.0 of its parent 'rec'",
        ),
    ],
    ids=["end", "offset", "unknown_segment"],
)
def test_a_child_beyond_its_parent_is_refused_in_the_validators_words(
    children: list[dict[str, Any]], path: str, message: str
) -> None:
    root = {"id": "rec", "unit": "seconds", "length": 60.0, "children": children}
    if len(children) == 2:
        root["is_segment_line"] = True
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed({"timetoshare": "0.2.0", "timelines": [root]}))
    code = "child_offset" if message.startswith("offset") else "child_end"
    assert refusal(refused.value) == (code, path, message)


PAGE, AT, SYSTEM = 932.2326510101616, 76.70371459980326, 855.5289364103584
"""A page length, and a child's offset and length that end there in doubles only."""


def _pair(value: float) -> list[int]:
    exact = Fraction(value)
    return [exact.numerator, exact.denominator]


def _root(**timeline: Any) -> dict[str, Any]:
    return {"timetoshare": "0.2.0", "timelines": [{"id": "page", **timeline}]}


@pytest.mark.parametrize(
    "manifest",
    [
        _root(
            unit="points",
            length=PAGE,
            children=[
                {"id": "system", "unit": "points", "offset": AT, "length": SYSTEM}
            ],
        ),
        _root(
            unit="points",
            length=PAGE,
            is_segment_line=True,
            children=[
                {"id": "a", "unit": "points", "offset": 0.0, "length": AT},
                {"id": "b", "unit": "points", "offset": AT, "length": SYSTEM},
            ],
        ),
    ],
    ids=["child", "segment"],
)
def test_a_float_child_ends_at_the_double_sum(manifest: dict[str, Any]) -> None:
    assert AT + SYSTEM == PAGE
    page = Timeline.from_dict(filed(manifest))
    assert bare(page.to_dict())["timelines"][0]["children"][-1]["offset"] == AT


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (
            _root(
                unit="points",
                number_type="fraction",
                length=_pair(PAGE),
                children=[
                    {
                        "id": "system",
                        "unit": "points",
                        "number_type": "fraction",
                        "offset": _pair(AT),
                        "length": _pair(SYSTEM),
                    }
                ],
            ),
            "child 'system' ends at [65600040932999587, 70368744177664], beyond the "
            "length [2050001279156237, 2199023255552] of its parent 'page'",
        ),
        (
            _root(
                unit="points",
                length=PAGE,
                children=[
                    {
                        "id": "system",
                        "unit": "points",
                        "number_type": "fraction",
                        "offset": AT,
                        "length": _pair(SYSTEM),
                    }
                ],
            ),
            "child 'system' ends at [65600040932999587, 70368744177664], beyond the "
            "length 932.2326510101616 of its parent 'page'",
        ),
        (
            _root(
                unit="ticks",
                length=2**53,
                children=[
                    {"id": "system", "unit": "ticks", "offset": 1, "length": 2**53}
                ],
            ),
            "child 'system' ends at 9007199254740993, beyond the length "
            "9007199254740992 of its parent 'page'",
        ),
    ],
    ids=["fraction", "mixed", "int"],
)
def test_any_other_child_ends_at_the_exact_sum(
    manifest: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed(manifest))
    assert refusal(refused.value) == (
        "child_end",
        "#/timelines/0/children/0/length",
        message,
    )


WORDS: dict[str, Any] = {
    "@context": dict(FORMAT_CONTEXT),
    "timetoshare": "0.2.0",
    "id": "card",
    "timelines": [
        {
            "id": "start",
            "unit": "quarters",
            "length": [12, 1],
            "children": [
                {
                    "id": "markers_10",
                    "unit": "quarters",
                    "offset": [0, 1],
                    "length": [4, 1],
                }
            ],
            "conversion_maps": [
                {
                    "type": "LinearMap",
                    "id": "label",
                    "source_unit": "quarters",
                    "target_unit": "seconds",
                    "scalar": 2,
                }
            ],
            "regions": [{"id": "end", "start": [0, 1], "end": [4, 1]}],
        }
    ],
    "groups": [{"id": "regions_1", "members": ["start"]}],
    "graphical": {"order": {"regions_1": ["start"]}},
}
"""Words naming the parts and facets of a drawn timeline, each an id."""


def test_words_naming_parts_and_facets_are_ids() -> None:
    bundle = AlignmentBundle.from_dict(filed(copy.deepcopy(WORDS)))

    assert bundle.id == "card"
    assert list(bundle.groups) == ["regions_1"]
    root = bundle.timelines["start"]
    assert root.list_children() == ["markers_10"]
    assert root.list_regions() == ["end"]
    assert root.get_conversion_map("label").id == "label"
    assert bare(bundle.to_dict(minimal=True)) == WORDS


def test_words_naming_parts_and_facets_are_overrides() -> None:
    timeline = Timeline.from_dict(filed(_quarters_root()), timeline_id="start")
    group = TimelineGroup.from_dict(filed(_quarters_root()), group_id="regions_1")

    assert timeline.id == "start"
    assert group.id == "regions_1"
    assert group.timeline_ids == ["score"]


@pytest.mark.parametrize(
    ("timeline", "path", "message"),
    [
        (
            {"id": "x", "unit": "quarters", "length": [-12, 1]},
            "#/timelines/0/length/0",
            "-12 is less than the minimum of 0",
        ),
        (
            {"id": "x", "unit": "seconds", "length": -8.0},
            "#/timelines/0/length",
            "-8.0 is less than the minimum of 0",
        ),
    ],
    ids=["fraction", "float"],
)
def test_a_negative_length_is_refused(
    timeline: dict[str, Any], path: str, message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(filed({"timetoshare": "0.2.0", "timelines": [timeline]}))
    assert refusal(refused.value) == ("schema", path, message)
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
        "timetoshare": "0.2.0",
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
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "id": "op",
        "name": "Op",
        "meta": {"k": 1},
        "x-top": 1,
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"styles": {"a": {"fill": "#FF0000"}}},
    }

    group = TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="g")

    assert bare(group.to_dict(minimal=True)) == {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "meta": {"k": 1},
        "x-top": 1,
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "groups": [{"id": "g", "members": ["r"]}],
        "graphical": {"order": {"g": ["r"]}, "styles": {"a": {"fill": "#FF0000"}}},
    }


@pytest.mark.parametrize(
    "reader",
    [reading(Timeline.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["timeline", "bundle"],
)
def test_a_graphical_object_read_is_written_back(reader: Any) -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
    }

    read = reader(copy.deepcopy({**data, "graphical": {}}))

    assert bare(read.to_dict(minimal=True)) == {**data, "graphical": {}}
    assert bare(read.to_dict(minimal=False))["graphical"] == {
        "elisions": [],
        "order": {},
        "rules": [],
        "styles": {},
        "timestamps": [],
        "matchstamps": [],
    }
    assert "graphical" not in bare(reader(copy.deepcopy(data)).to_dict())


ELISIONS = [
    {
        "id": "elision_2",
        "timeline": "r",
        "start": [3, 2],
        "end": [3, 1],
        "width": {"ratio": 0.5},
    },
    {"id": "elision_1", "timeline": "r", "start": [0, 1], "end": [1, 2]},
]
ELISION_DISPLAY = ["strip", "name", "coordinate"]


@pytest.mark.parametrize(
    "reader",
    [reading(Timeline.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["timeline", "bundle"],
)
def test_a_stored_elision_is_written_back_unchanged(reader: Any) -> None:
    data = {
        "@context": dict(FORMAT_CONTEXT),
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"elisions": copy.deepcopy(ELISIONS)},
    }

    read = reader(copy.deepcopy(data))

    assert bare(read.to_dict(minimal=True)) == data
    assert bare(read.to_dict(minimal=False))["graphical"] == {
        "elisions": [{**elision, "display": ELISION_DISPLAY} for elision in ELISIONS],
        "order": {},
        "rules": [],
        "styles": {},
        "timestamps": [],
        "matchstamps": [],
    }


def _stamp(number: int, **members: Any) -> dict[str, Any]:
    """A named timestamp on ``r`` at 1 with the given further members."""
    return {
        "id": f"timestamp_{number}",
        "timeline": "r",
        "coordinate": [1, 1],
        "name": f"T{number}",
        **members,
    }


@pytest.mark.parametrize(
    "reader",
    [reading(Timeline.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["timeline", "bundle"],
)
def test_a_timestamp_display_is_written_in_canonical_order_with_its_default(
    reader: Any,
) -> None:
    given = [
        _stamp(1),
        _stamp(2, display=["cross_section", "wedge", "name"]),
        _stamp(3, display=["name", "wedge"]),
        _stamp(4, display=["coordinate", "id", "wedge"], note={"x": 1}),
    ]
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"timestamps": given},
    }
    kept = copy.deepcopy(data)

    read = reader(data)

    default = ["wedge", "name", "cross_section"]
    flags = {"matchstamp": False, "groupstamp": False}
    assert bare(read.to_dict(minimal=False))["graphical"]["timestamps"] == [
        _stamp(1, display=default, **flags),
        _stamp(2, display=default, **flags),
        _stamp(3, display=["wedge", "name"], **flags),
        _stamp(4, display=["wedge", "id", "coordinate"], note={"x": 1}, **flags),
    ]
    assert bare(read.to_dict(minimal=True))["graphical"]["timestamps"] == [
        _stamp(1),
        _stamp(2),
        _stamp(3, display=["wedge", "name"]),
        _stamp(4, display=["wedge", "id", "coordinate"], note={"x": 1}),
    ]
    assert data == kept


def test_a_timestamp_display_word_the_format_lacks_is_refused() -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"timestamps": [_stamp(1, display=["flag", "wedge"])]},
    }

    with pytest.raises(ValueError) as refused:
        AlignmentBundle.from_dict(filed(data))

    assert refusal(refused.value) == (
        "schema",
        "#/graphical/timestamps/0/display/0",
        "'flag' is not one of ['wedge', 'name', 'id', 'coordinate', 'cross_section']",
    )


def _stamped(timeline: str, coordinate: list[int]) -> dict[str, Any]:
    """A root ``clt_1`` of length 12 with one named timestamp on *timeline*."""
    return {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "clt_1", "unit": "quarters", "length": [12, 1]}],
        "graphical": {
            "timestamps": [
                {
                    "id": "timestamp_1",
                    "name": "T1",
                    "timeline": timeline,
                    "coordinate": coordinate,
                }
            ]
        },
    }


@pytest.mark.parametrize(
    "reader",
    [reading(Timeline.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["timeline", "bundle"],
)
@pytest.mark.parametrize(
    ("timeline", "coordinate", "code", "path", "message"),
    [
        (
            "clt_9",
            [3, 1],
            "timestamp_timeline",
            "#/graphical/timestamps/0/timeline",
            "'clt_9' names no timeline of this manifest",
        ),
        (
            "clt_1",
            [13, 1],
            "timestamp_coordinate",
            "#/graphical/timestamps/0/coordinate",
            "coordinate [13, 1] of 'timestamp_1' is beyond the length [12, 1] of "
            "its timeline 'clt_1'",
        ),
    ],
    ids=["no_such_timeline", "beyond_its_length"],
)
def test_a_manifest_the_format_refuses_is_not_read(
    reader: Any,
    timeline: str,
    coordinate: list[int],
    code: str,
    path: str,
    message: str,
) -> None:
    with pytest.raises(ValueError) as refused:
        reader(_stamped(timeline, coordinate))
    assert refusal(refused.value) == (code, path, message)


def test_a_manifest_the_format_refuses_is_not_written(tmp_path: Any) -> None:
    score = Timeline.from_dict(filed(_stamped("clt_1", [11, 1])))
    score.length = 10
    message = (
        "coordinate [11, 1] of 'timestamp_1' is beyond the length [10, 1] of its "
        "timeline 'clt_1'"
    )

    for write in (score.to_dict, lambda: score.to_file(tmp_path / "score.yaml")):
        with pytest.raises(ValueError) as refused:
            write()
        assert refusal(refused.value) == (
            "timestamp_coordinate",
            "#/graphical/timestamps/0/coordinate",
            message,
        )
    assert list(tmp_path.iterdir()) == []


GROUPSTAMP_REFUSAL = (
    "#/graphical/timestamps/0/groupstamp: 'timestamp_1' is a groupstamp, but its "
    "timeline '{timeline}' is a member of no group and lies in none"
)


@pytest.mark.parametrize(
    "reader",
    [reading(Timeline.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["timeline", "bundle"],
)
def test_a_groupstamp_lying_in_no_group_is_refused_on_reading(reader: Any) -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "graphical": {"timestamps": [_stamp(1, groupstamp=True)]},
    }

    with pytest.raises(ValueError) as refusal:
        reader(data)

    assert str(refusal.value) == GROUPSTAMP_REFUSAL.format(timeline="r")


def test_a_groupstamp_on_a_member_s_child_is_read_and_written_back() -> None:
    stamp = {**_stamp(1, groupstamp=True), "timeline": "c"}
    data = {
        "timetoshare": "0.2.0",
        "timelines": [
            {
                "id": "r",
                "unit": "quarters",
                "length": [4, 1],
                "children": [
                    {"id": "c", "unit": "quarters", "length": [2, 1], "offset": [1, 1]}
                ],
            }
        ],
        "groups": [{"id": "g", "members": ["r"]}],
        "graphical": {"timestamps": [stamp]},
    }

    read = reading(AlignmentBundle.from_dict)(data)

    assert bare(read.to_dict())["graphical"]["timestamps"] == [stamp]


def test_a_groupstamp_whose_group_lost_its_timeline_is_refused_on_writing() -> None:
    stamp = {**_stamp(1, groupstamp=True), "timeline": "s"}
    data = {
        "timetoshare": "0.2.0",
        "timelines": [
            {"id": "r", "unit": "quarters", "length": [4, 1]},
            {"id": "s", "unit": "quarters", "length": [4, 1]},
        ],
        "groups": [{"id": "g", "members": ["r", "s"]}],
        "graphical": {"timestamps": [stamp]},
    }
    bundle = reading(AlignmentBundle.from_dict)(data)
    bundle.groups["g"].remove_timeline("s")

    with pytest.raises(ValueError) as refusal:
        bundle.to_dict()

    assert str(refusal.value) == GROUPSTAMP_REFUSAL.format(timeline="s")


def _match_groupstamp_bundle(*, grouped: bool) -> dict[str, Any]:
    """Roots ``r`` and ``s``, grouped in ``g`` where *grouped*, joined by one
    MatchStamp whose match timestamp on ``r`` is a groupstamp."""
    data: dict[str, Any] = {
        "timetoshare": "0.2.0",
        "timelines": [
            {"id": "r", "unit": "quarters", "length": [4, 1]},
            {"id": "s", "unit": "quarters", "length": [4, 1]},
        ],
        "graphical": {
            "timestamps": [
                {
                    "id": "timestamp_1",
                    "timeline": "r",
                    "coordinate": [1, 1],
                    "display": ["cross_section"],
                    "matchstamp": True,
                    "groupstamp": True,
                },
                {
                    "id": "timestamp_2",
                    "timeline": "s",
                    "coordinate": [2, 1],
                    "display": ["cross_section"],
                    "matchstamp": True,
                    "groupstamp": False,
                },
            ],
            "matchstamps": [
                {
                    "id": "matchstamp_1",
                    "name": "Bar 1",
                    "alignment_anchors": [
                        {
                            "id": "anchor_1",
                            "timestamps": ["timestamp_1", "timestamp_2"],
                            "display": ["line"],
                        }
                    ],
                    "match_claims": ["claim_1"],
                }
            ],
        },
    }
    if grouped:
        data["groups"] = [{"id": "g", "members": ["r", "s"]}]
    return data


def test_a_match_timestamp_that_is_a_groupstamp_is_read_and_written_back() -> None:
    data = _match_groupstamp_bundle(grouped=True)
    graphical = copy.deepcopy(data["graphical"])

    read = reading(AlignmentBundle.from_dict)(data)

    written = bare(read.to_dict(minimal=False))["graphical"]
    assert written["timestamps"] == graphical["timestamps"]
    assert written["matchstamps"] == graphical["matchstamps"]


def test_a_match_timestamp_that_is_a_groupstamp_lying_in_no_group_is_refused() -> None:
    data = _match_groupstamp_bundle(grouped=False)

    with pytest.raises(ValueError) as refusal:
        reading(AlignmentBundle.from_dict)(data)

    assert str(refusal.value) == GROUPSTAMP_REFUSAL.format(timeline="r")


STYLING_CASES = {
    "styles_default": ({"styles": {}}, {}),
    "rules_default": ({"rules": []}, {}),
    "timestamps_default": ({"timestamps": []}, {}),
    "matchstamps_default": ({"matchstamps": []}, {}),
    "matchstamp_flag_default": (
        {"timestamps": [_stamp(1, matchstamp=False)]},
        {"timestamps": [_stamp(1)]},
    ),
    "groupstamp_flag_default": (
        {"timestamps": [_stamp(1, groupstamp=False)]},
        {"timestamps": [_stamp(1)]},
    ),
    "display_default": (
        {"timestamps": [_stamp(1, display=["cross_section", "name", "wedge"])]},
        {"timestamps": [_stamp(1)]},
    ),
    "display_reordered": (
        {"timestamps": [_stamp(1, display=["name", "wedge"])]},
        {"timestamps": [_stamp(1, display=["wedge", "name"])]},
    ),
    "zero_and_false": (
        {
            "styles": {"a": {"stroke-width": 0, "visible": False}},
            "rules": [],
            "frame": False,
            "inset": 0,
        },
        {
            "styles": {"a": {"stroke-width": 0, "visible": False}},
            "frame": False,
            "inset": 0,
        },
    ),
}
"""Each styling read with the styling the format's defaults make of it."""


@pytest.mark.parametrize(
    "reader",
    [reading(TimelineGroup.from_dict), reading(AlignmentBundle.from_dict)],
    ids=["group", "bundle"],
)
@pytest.mark.parametrize("case", list(STYLING_CASES), ids=list(STYLING_CASES))
def test_a_styling_member_at_its_default_is_dropped_as_timetoshare_drops_it(
    reader: Any, case: str
) -> None:
    given, expected = STYLING_CASES[case]
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "r", "unit": "quarters", "length": [4, 1]}],
        "groups": [{"id": "g", "members": ["r"]}],
        "graphical": copy.deepcopy(given),
    }

    styling = reader(data).styling

    assert styling == expected
    assert styling == minimal({"graphical": given})["graphical"]
    assert _text(styling) == _text(expected)


def test_mixed_group_arguments_raise() -> None:
    data = {
        "timetoshare": "0.2.0",
        "timelines": [{"id": "score", "unit": "quarters", "length": [8, 1]}],
        "groups": [{"id": "g", "name": "G", "members": ["score"]}],
    }

    with pytest.raises(ValueError) as refused:
        TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="g", name="Other")

    assert str(refused.value) == (
        "the manifest holds no group with id 'g', name 'Other'; its groups: 'g' (name 'G')"
    )
    group = TimelineGroup.from_dict(filed(copy.deepcopy(data)), group_id="g", name="G")
    assert (group.id, group.name) == ("g", "G")


# endregion
