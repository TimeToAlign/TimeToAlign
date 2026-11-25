"""The canonical order of a serialized timeline, independent of insertion order."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Any

import pytest

from tests.helpers import bare
from timetoalign.core import FlowManifestation
from timetoalign.maps import ConstantMap, ScalarMap, ShiftMap
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
    Timeline,
)
from timetoalign.timelines.flowcontrol import (
    Break,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)


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
MAPS = ["shift", "scale", "constant"]
REGIONS = [
    ("verse", "lo", "hi"),
    ("intro", "lo", "mid"),
    ("alpha", "lo", "hi"),
    ("coda", "mid", "hi"),
]
BREAKS = [
    ("section", SectionBreak, None, "mid"),
    ("fine", Break, FlowManifestation.fine, "mid"),
    ("early", SectionBreak, None, "lo"),
]
# One jump lies at a coordinate; the signs sharing one are its destinations.
JUMPS = [("back", "hi"), ("first", "mid"), ("skip", "lo")]
# The destinations of the jump ``back``, each a label and a place: none,
# ``by`` back to a position, or ``to`` a marker.
DESTINATIONS = [
    ("segno", "to", "segno"),
    ("inner", "by", "mid"),
    ("capo", None, None),
    ("Coda", "to", "Coda"),
    ("repeat", "by", "lo"),
]
MARKERS = [
    ("segno", FlowManifestation.segno, "lo"),
    ("coda", FlowManifestation.coda, "mid"),
    ("Repeat", FlowManifestation.repeat_start, "hi"),
    ("Coda", FlowManifestation.coda, "mid"),
]

COLLECTIONS = {
    "children": CHILDREN,
    "maps": MAPS,
    "regions": REGIONS,
    "breaks": BREAKS,
    "jumps": JUMPS,
    "destinations": DESTINATIONS,
    "markers": MARKERS,
}

# The canonical order, spelled out.
CANONICAL_CHILDREN = ["m", "z", "b", "a"]  # lo (m before z), mid, hi
CANONICAL_MAPS = ["constant", "scale", "shift"]
CANONICAL_REGIONS = ["intro", "alpha", "verse", "coda"]  # (lo,mid), (lo,hi)x2, (mid,hi)
# Flow-control elements by coordinate, ties by id; each element's id is its
# kind and label. Code point: "marker-Coda" < "marker-coda" ("C" < "c").
CANONICAL_FLOW = [
    "break-early",
    "jump-skip",
    "marker-segno",
    "break-fine",
    "break-section",
    "jump-first",
    "marker-Coda",
    "marker-coda",
    "jump-back",
    "marker-Repeat",
]
# The destinations of a jump by place: the one without, then by distance
# (back to lo lies further than back to mid), then by the id named.
CANONICAL_DESTINATIONS = ["capo", "repeat", "inner", "Coda", "segno"]
FLOW_ROLES = {
    "breaks": {"break", "section_break"},
    "jumps": {"jump"},
    "markers": {"jump_target"},
}


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
        timeline.create_region(position[start], position[end], id=name, name=name)
    registry = timeline.flow_control
    for name, manifestation, at in orders["markers"]:
        registry.add(
            JumpTarget(
                timeline.make_coordinate(position[at]),
                name=name,
                manifestation=manifestation,
                id=f"marker-{name}",
            )
        )
    for label, element_class, manifestation, at in orders["breaks"]:
        registry.add(
            element_class(
                timeline.make_coordinate(position[at]),
                manifestation=manifestation,
                label=label,
                id=f"break-{label}",
            )
        )
    destinations = {
        "back": [
            JumpDestination(
                label=label,
                repeat_count=1,
                to=f"marker-{place}" if kind == "to" else None,
                by=position[place] - position["hi"] if kind == "by" else None,
            )
            for label, kind, place in orders["destinations"]
        ],
        "first": [JumpDestination(to="marker-segno", repeat_count=1)],
        "skip": [JumpDestination(to="marker-coda")],
    }
    for label, at in orders["jumps"]:
        registry.add(
            Jump(
                timeline.make_coordinate(position[at]),
                destinations[label],
                label=label,
                id=f"jump-{label}",
            )
        )
    return timeline


def _text(payload: dict[str, Any]) -> str:
    return json.dumps(payload)


def _root(payload: dict[str, Any]) -> dict[str, Any]:
    """The timeline object of a one-root manifest."""
    return payload["timelines"][0]


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
    assert len(INSERTION_ORDERS) == 24 + 6 + 24 + 6 + 6 + 120 + 24
    for case, orders in INSERTION_ORDERS:
        payload = _build(axis, orders).to_dict()
        assert payload == reference, case
        assert _text(payload) == _text(reference), case


@pytest.mark.parametrize("axis_name", AXES)
def test_to_dict_writes_the_canonical_order(axis_name: str) -> None:
    payload = _root(bare(_build(AXES[axis_name], COLLECTIONS).to_dict()))
    assert [entry["id"] for entry in payload["children"]] == CANONICAL_CHILDREN
    assert [entry["id"] for entry in payload["conversion_maps"]] == CANONICAL_MAPS
    assert [entry["id"] for entry in payload["regions"]] == CANONICAL_REGIONS
    assert [entry["id"] for entry in payload["flow_control"]] == CANONICAL_FLOW
    assert {
        entry["id"]: [
            destination.get("label", destination.get("to"))
            for destination in entry["targets"]
        ]
        for entry in payload["flow_control"]
        if entry["role"] == "jump"
    } == {
        "jump-skip": ["marker-coda"],
        "jump-first": ["marker-segno"],
        "jump-back": CANONICAL_DESTINATIONS,
    }


def test_offsets_compare_by_value_not_by_their_json_spelling() -> None:
    offsets = {
        name: [
            entry["offset"]
            for entry in _root(bare(_build(axis, COLLECTIONS).to_dict()))["children"]
        ]
        for name, axis in AXES.items()
    }
    assert offsets == {
        "ticks_int": [9, 9, 10, 20],
        "quarters_fraction": [[5, 8], [5, 8], [2, 3], [7, 2]],
        "seconds_float": [9.25, 9.25, 10.5, 20.75],
    }


def test_distances_compare_by_value_not_by_their_json_spelling() -> None:
    distances = {}
    for name, axis in AXES.items():
        flow = _root(bare(_build(axis, COLLECTIONS).to_dict()))["flow_control"]
        (back,) = (entry for entry in flow if entry["id"] == "jump-back")
        distances[name] = [
            destination["by"] for destination in back["targets"] if "by" in destination
        ]
    assert distances == {
        # "-10" < "-11" as text
        "ticks_int": [-11, -10],
        # [-17, 6] < [-23, 8] as text
        "quarters_fraction": [[-23, 8], [-17, 6]],
        # "-10.25" < "-11.5" as text
        "seconds_float": [-11.5, -10.25],
    }


def _reordered(payload: dict[str, Any], name: str, order: tuple[int, ...]) -> dict:
    """Return a copy of *payload* with one collection rearranged by *order*."""
    manifest = copy.deepcopy(payload)
    result = _root(manifest)
    if name == "children":
        result["children"] = [result["children"][index] for index in order]
    elif name == "maps":
        maps = result["conversion_maps"]
        result["conversion_maps"] = [maps[index] for index in order]
    elif name == "regions":
        result["regions"] = [result["regions"][index] for index in order]
    elif name == "destinations":
        (back,) = (
            entry for entry in result["flow_control"] if entry["id"] == "#jump-back"
        )
        back["targets"] = [back["targets"][index] for index in order]
    else:
        flow = result["flow_control"]
        kind = [entry for entry in flow if entry["role"] in FLOW_ROLES[name]]
        rest = [entry for entry in flow if entry["role"] not in FLOW_ROLES[name]]
        result["flow_control"] = [kind[index] for index in order] + rest
    return manifest


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
    assert checked == 24 + 6 + 24 + 6 + 6 + 120 + 24


def test_children_keep_their_stored_order_and_flow_control_is_held_canonically() -> (
    None
):
    timeline = _build(AXES["quarters_fraction"], COLLECTIONS)
    timeline.to_dict()
    assert timeline.list_children() == ["z", "m", "b", "a"]
    registry = timeline.flow_control
    assert [element.id for element in registry.elements] == CANONICAL_FLOW
    (back,) = (jump for jump in registry.jumps if jump.id == "jump-back")
    assert [destination.label for destination in back.targets] == (
        CANONICAL_DESTINATIONS
    )
    copied = registry.copy()
    assert [element.id for element in copied.elements] == CANONICAL_FLOW
    assert copied.to_dict() == registry.to_dict()
