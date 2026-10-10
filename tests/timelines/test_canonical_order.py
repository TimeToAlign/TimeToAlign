"""The canonical order of a serialized timeline, independent of insertion order."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Any

import pytest

from timetoalign.core import FlowControlElement
from timetoalign.core.retrieval import canonical_json
from timetoalign.maps import ConstantMap, ScalarMap, ShiftMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    Timeline,
)
from timetoalign.timelines.flowcontrol import Break, Jump


@dataclass(frozen=True)
class Axis:
    """One timeline axis with three positions whose JSON text order is not numeric."""

    cls: type[Timeline]
    unit: str
    length: Any
    child_length: Any
    lo: Any
    mid: Any
    hi: Any


AXES = {
    # "10" < "20" < "9" as text
    "ticks_int": Axis(DiscreteLogicalTimeline, "ticks", 100, 1, 9, 10, 20),
    # [2, 3] < [5, 8] < [7, 2] as text
    "quarters_fraction": Axis(
        ContinuousLogicalTimeline,
        "quarters",
        Fraction(16),
        Fraction(1, 4),
        Fraction(5, 8),
        Fraction(2, 3),
        Fraction(7, 2),
    ),
    # "10.5" < "20.75" < "9.25" as text
    "seconds_float": Axis(
        ContinuousPhysicalTimeline, "seconds", 100.0, 0.5, 9.25, 10.5, 20.75
    ),
}

# Reference insertion order of every collection; none is canonical.
CHILDREN = [("z", "lo"), ("m", "lo"), ("b", "mid"), ("a", "hi")]
MAPS = ["shift", "scale", "label"]
REGIONS = [
    ("verse", "lo", "hi"),
    ("intro", "lo", "mid"),
    ("alpha", "lo", "hi"),
    ("coda", "mid", "hi"),
]
BREAKS = [
    ("section", FlowControlElement.section_break, "mid"),
    ("fine", FlowControlElement.fine, "mid"),
    ("early", FlowControlElement.section_break, "lo"),
]
JUMPS = [
    ("repeat", FlowControlElement.repeat_end, "hi", "lo"),
    ("inner", FlowControlElement.repeat_end, "hi", "mid"),
    ("segno", FlowControlElement.dal_segno, "hi", "lo"),
    ("first", FlowControlElement.repeat_end, "mid", "lo"),
]
MARKERS = [("segno", "lo"), ("coda", "mid"), ("Fine", "hi")]

COLLECTIONS = {
    "children": CHILDREN,
    "maps": MAPS,
    "regions": REGIONS,
    "breaks": BREAKS,
    "jumps": JUMPS,
    "markers": MARKERS,
}

# The canonical order, spelled out.
CANONICAL_CHILDREN = ["m", "z", "b", "a"]  # lo (m before z), mid, hi
CANONICAL_MAPS = ["label", "scale", "shift"]
CANONICAL_REGIONS = ["intro", "alpha", "verse", "coda"]  # (lo,mid), (lo,hi)x2, (mid,hi)
# Both breaks at mid share condition "always"; "fine" < "section_break" in their JSON text.
CANONICAL_BREAKS = ["early", "fine", "section"]
# hi->lo twice: "dal_segno" < "repeat_end" in their JSON text; hi->mid follows.
CANONICAL_JUMPS = ["first", "segno", "repeat", "inner"]
CANONICAL_MARKERS = ["Fine", "coda", "segno"]  # code point: "F" < "c" < "s"


def _map(map_id: str) -> Any:
    if map_id == "shift":
        return ShiftMap(offset=1, uid=map_id)
    if map_id == "scale":
        return ScalarMap(2, uid=map_id)
    return ConstantMap(value="verse", uid=map_id)


def _build(axis: Axis, orders: dict[str, list[Any]]) -> Timeline:
    """Build the fixed content set, each collection inserted in the given order."""
    position = {"lo": axis.lo, "mid": axis.mid, "hi": axis.hi}
    timeline = axis.cls(length=axis.length, unit=axis.unit, uid="parent")
    for child_id, offset in orders["children"]:
        child = axis.cls(length=axis.child_length, unit=axis.unit, uid=child_id)
        timeline.add_child(child, position[offset])
    for map_id in orders["maps"]:
        timeline.add_conversion_map(_map(map_id))
    for name, start, end in orders["regions"]:
        timeline.create_region(name, position[start], position[end])
    registry = timeline.flow_control
    for label, control_type, at in orders["breaks"]:
        registry.add_break(
            Break(
                coordinate=timeline.make_coordinate(position[at]),
                control_type=control_type,
                label=label,
            )
        )
    for label, control_type, source, target in orders["jumps"]:
        registry.add_jump(
            Jump(
                from_coordinate=timeline.make_coordinate(position[source]),
                to_coordinate=timeline.make_coordinate(position[target]),
                control_type=control_type,
                label=label,
            )
        )
    for name, at in orders["markers"]:
        registry.add_marker(name, position[at])
    return timeline


def _text(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _insertion_orders() -> list[tuple[str, dict[str, list[Any]]]]:
    """Every permutation of each collection, the others in reference order."""
    cases = []
    for name, entries in COLLECTIONS.items():
        for index, permuted in enumerate(permutations(entries)):
            cases.append((f"{name}-{index}", {**COLLECTIONS, name: list(permuted)}))
    return cases


INSERTION_ORDERS = _insertion_orders()


@pytest.mark.parametrize("axis_name", AXES)
def test_to_dict_is_invariant_under_every_insertion_order(axis_name: str) -> None:
    axis = AXES[axis_name]
    reference = _build(axis, COLLECTIONS).to_dict()
    assert len(INSERTION_ORDERS) == 24 + 6 + 24 + 6 + 24 + 6
    for case, orders in INSERTION_ORDERS:
        payload = _build(axis, orders).to_dict()
        assert payload == reference, case
        assert _text(payload) == _text(reference), case


@pytest.mark.parametrize("axis_name", AXES)
def test_to_dict_writes_the_canonical_order(axis_name: str) -> None:
    payload = _build(AXES[axis_name], COLLECTIONS).to_dict()
    assert list(payload["children"]) == CANONICAL_CHILDREN
    assert [entry["id"] for entry in payload["conversion_maps"]] == CANONICAL_MAPS
    assert [entry["name"] for entry in payload["regions"]] == CANONICAL_REGIONS
    flow = payload["flow_control"]
    assert [entry["label"] for entry in flow["breaks"]] == CANONICAL_BREAKS
    assert [entry["label"] for entry in flow["jumps"]] == CANONICAL_JUMPS
    assert list(flow["markers"]) == CANONICAL_MARKERS


def test_offsets_compare_by_value_not_by_their_json_spelling() -> None:
    offsets = {
        name: [
            entry["offset"]["value"]
            for entry in _build(axis, COLLECTIONS).to_dict()["children"].values()
        ]
        for name, axis in AXES.items()
    }
    assert offsets == {
        "ticks_int": [9, 9, 10, 20],
        "quarters_fraction": [[5, 8], [5, 8], [2, 3], [7, 2]],
        "seconds_float": [9.25, 9.25, 10.5, 20.75],
    }


def _reordered(payload: dict[str, Any], name: str, order: tuple[int, ...]) -> dict:
    """Return a copy of *payload* with one collection rearranged by *order*."""
    result = copy.deepcopy(payload)
    flow = result["flow_control"]
    if name == "children":
        items = list(result["children"].items())
        result["children"] = dict(items[index] for index in order)
    elif name == "maps":
        maps = result["conversion_maps"]
        result["conversion_maps"] = [maps[index] for index in order]
    elif name == "regions":
        result["regions"] = [result["regions"][index] for index in order]
    elif name == "markers":
        items = list(flow["markers"].items())
        flow["markers"] = dict(items[index] for index in order)
    else:
        flow[name] = [flow[name][index] for index in order]
    return result


@pytest.mark.parametrize("axis_name", AXES)
def test_from_dict_accepts_any_order(axis_name: str) -> None:
    canonical = _build(AXES[axis_name], COLLECTIONS).to_dict()
    checked = 0
    for name, entries in COLLECTIONS.items():
        for order in permutations(range(len(entries))):
            payload = _reordered(canonical, name, order)
            restored = Timeline.from_dict(payload).to_dict()
            assert restored == canonical, (name, order)
            assert _text(restored) == _text(canonical), (name, order)
            checked += 1
    assert checked == 90


def test_storage_keeps_its_own_order() -> None:
    timeline = _build(AXES["quarters_fraction"], COLLECTIONS)
    timeline.to_dict()
    assert timeline.list_children() == ["z", "m", "b", "a"]
    # Sorted by position, the two breaks at one position in insertion order.
    assert [brk.label for brk in timeline.flow_control.breaks] == [
        "early",
        "section",
        "fine",
    ]
    assert list(timeline.flow_control.markers) == ["segno", "coda", "Fine"]
    copied = timeline.flow_control.copy()
    assert [brk.label for brk in copied.breaks] == ["early", "section", "fine"]
    assert copied.to_dict() == timeline.flow_control.to_dict()


def test_canonical_json_text() -> None:
    value = {"\U0001f600": {"b": True, "a": [1, 2.5, None]}, "\u00e9": 1.0, "a": []}
    assert (
        canonical_json(value)
        == '{"a":[],"\u00e9":1.0,"\U0001f600":{"a":[1,2.5,null],"b":true}}'
    )


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf], ids=repr)
def test_canonical_json_refuses_what_json_cannot_hold(number: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"meta": {"x": number}})
