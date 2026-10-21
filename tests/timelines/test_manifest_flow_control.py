"""Flow control in a manifest (see README, *Flow control in a manifest*).

``flow_control`` is one array of ``flow_marker``, ``jump_from`` and
``jump_to`` elements; the library's breaks, jumps and markers write and read
it.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from typing import Any

import pytest

from timetoalign import Ms3Loader
from timetoalign.core import (
    ActivationCondition,
    Coordinate,
    FlowControlElement,
    TimeUnit,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import Timeline
from timetoalign.timelines.flowcontrol import Break, Jump, Marker
from timetoalign.timelines.types import ContinuousLogicalTimeline

F = Fraction
E = FlowControlElement
SCORE = ensure_data("score")


def _q(value: Any) -> Coordinate:
    return Coordinate(F(value), TimeUnit.quarters)


def _manifest(flow_control: list[dict[str, Any]], **timeline: Any) -> dict[str, Any]:
    return {
        "timetoshare": "0.1.0",
        "timelines": [
            {
                "id": "score",
                "unit": "quarters",
                "length": [32, 1],
                "flow_control": flow_control,
                **timeline,
            }
        ],
    }


def _element(
    element_id: str, kind: str, manifestation: str, at: int, **keys: Any
) -> dict:
    return {
        "id": element_id,
        "type": kind,
        "manifestation": manifestation,
        "coordinate": [at, 1],
        **keys,
    }


# region The literal

MINIMAL_FLOW: list[dict[str, Any]] = [
    _element("top", "jump_to", "repeat_start", 0, name="||:"),
    _element("segno", "jump_to", "segno", 4, meta={"glyph": "U+1D10B"}),
    _element("section", "flow_marker", "section_break", 8, label="end of A"),
    _element("back", "jump_from", "repeat_end", 8, target="top", repeat_count=2),
    _element("coda", "jump_to", "coda", 24, **{"x-note": "kept"}),
    _element("to_coda", "jump_from", "to_coda", 20, target="coda"),
    _element("ds", "jump_from", "dal_segno_al_coda", 22, target="segno", label="D.S."),
    _element("fine", "flow_marker", "fine", 32, name="Fine"),
]
MINIMAL_FLOW.sort(key=lambda element: (element["coordinate"][0], element["id"]))


def _complete(element: dict[str, Any]) -> dict[str, Any]:
    result = {"name": None, "label": None, "meta": {}, **element}
    if element["manifestation"] == "repeat_end":
        result.setdefault("repeat_count", 1)
    return result


@pytest.mark.parametrize("form", ["minimal", "complete"])
def test_the_literal_round_trips(form: str) -> None:
    flow = MINIMAL_FLOW if form == "minimal" else [_complete(e) for e in MINIMAL_FLOW]
    timeline = Timeline.from_dict(_manifest(copy.deepcopy(flow)))
    written = timeline.to_dict(minimal=form == "minimal")["timelines"][0][
        "flow_control"
    ]
    assert written == flow
    other = timeline.to_dict(minimal=form != "minimal")["timelines"][0]["flow_control"]
    assert other == (
        [_complete(e) for e in MINIMAL_FLOW] if form == "minimal" else MINIMAL_FLOW
    )


def test_reading_builds_the_library_s_elements() -> None:
    registry = Timeline.from_dict(_manifest(copy.deepcopy(MINIMAL_FLOW))).flow_control

    assert [
        (b.id, b.control_type, b.position, b.name, b.label) for b in registry.breaks
    ] == [
        ("section", E.section_break, 8, None, "end of A"),
        ("fine", E.fine, 32, "Fine", None),
    ]
    assert [(m.id, m.control_type, m.position) for m in registry.markers] == [
        ("top", E.repeat_start, 0),
        ("segno", E.segno, 4),
        ("coda", E.coda, 24),
    ]
    assert [
        (j.id, j.control_type, j.from_position, j.to_position, j.target, j.repeat_count)
        for j in registry.jumps
    ] == [
        ("back", E.repeat_end, 8, 0, "top", 2),
        ("to_coda", E.to_coda, 20, 24, "coda", 1),
        ("ds", E.dal_segno_al_coda, 22, 4, "segno", 1),
    ]
    assert [j.condition for j in registry.jumps] == [
        ActivationCondition.first_n,
        ActivationCondition.after_dc_ds,
        ActivationCondition.always,
    ]
    assert [b.condition for b in registry.breaks] == [ActivationCondition.always] * 2


# endregion

# region Targets


def test_a_jump_without_target_lands_at_the_start() -> None:
    flow = [
        _element("back", "jump_from", "repeat_end", 8),
        _element("dc", "jump_from", "da_capo", 16),
    ]
    registry = Timeline.from_dict(_manifest(flow)).flow_control
    assert [(j.to_coordinate, j.target) for j in registry.jumps] == [
        (_q(0), None),
        (_q(0), None),
    ]


def test_a_jump_landing_elsewhere_than_its_target_is_refused() -> None:
    timeline = ContinuousLogicalTimeline(length=F(32), uid="score")
    segno = timeline.flow_control.add_marker(
        Marker(_q(4), control_type=E.segno, id="segno")
    )
    timeline.flow_control.add_jump(
        Jump(_q(16), _q(6), control_type=E.dal_segno, target=segno.id, id="ds")
    )
    with pytest.raises(ValueError) as refused:
        timeline.to_dict()
    assert (
        str(refused.value) == "jump 'ds' lands at 6, but its target 'segno' lies at 4"
    )


def test_a_destination_without_a_marker_gets_one() -> None:
    timeline = ContinuousLogicalTimeline(length=F(32), uid="score")
    timeline.flow_control.add_break(Break(_q(2), id="jump_to_1"))
    for jump_id, control_type, at, to in [
        ("rep", E.repeat_end, 16, 8),
        ("rep2", E.repeat_end, 18, 8),
        ("ds", E.dal_segno, 20, 4),
        ("tc", E.to_coda, 6, 12),
        ("dcc", E.da_capo_al_coda, 24, 10),
        ("rep0", E.repeat_end, 26, 0),
        ("dc0", E.da_capo, 28, 0),
    ]:
        timeline.flow_control.add_jump(
            Jump(_q(at), _q(to), control_type=control_type, id=jump_id)
        )

    flow = timeline.to_dict(minimal=True)["timelines"][0]["flow_control"]
    landings = [
        (element["id"], element["manifestation"], element["coordinate"])
        for element in flow
        if element["type"] == "jump_to"
    ]
    # Minted in the order the jumps are written (by from coordinate), around
    # the break holding jump_to_1.
    assert landings == [
        ("jump_to_4", "segno", [4, 1]),
        ("jump_to_3", "repeat_start", [8, 1]),
        ("jump_to_5", "coda", [10, 1]),
        ("jump_to_2", "coda", [12, 1]),
    ]
    assert {
        element["id"]: element.get("target")
        for element in flow
        if element["type"] == "jump_from"
    } == {
        "rep": "jump_to_3",
        "rep2": "jump_to_3",
        "ds": "jump_to_4",
        "tc": "jump_to_2",
        "dcc": "jump_to_5",
        "rep0": None,
        "dc0": None,
    }
    read = Timeline.from_dict(timeline.to_dict())
    assert [(j.id, j.to_position) for j in read.flow_control.jumps] == [
        (j.id, j.to_position) for j in timeline.flow_control.jumps
    ]
    assert read.to_dict() == Timeline.from_dict(read.to_dict()).to_dict()


def test_a_da_capo_landing_elsewhere_than_the_start_is_refused() -> None:
    timeline = ContinuousLogicalTimeline(length=F(32), uid="score")
    timeline.flow_control.add_jump(Jump(_q(16), _q(3), control_type=E.da_capo, id="dc"))
    with pytest.raises(ValueError) as refused:
        timeline.to_dict()
    assert (
        str(refused.value)
        == "jump 'dc' (da_capo) lands at 3, not at the timeline's start"
    )


# endregion

# region What each element is


def test_each_element_class_holds_its_own_manifestations() -> None:
    with pytest.raises(ValueError) as refused:
        Break(_q(0), control_type=E.segno)
    assert str(refused.value) == (
        "A Break is one of ['double_barline', 'final_barline', 'fine', "
        "'section_break'], not 'segno'"
    )
    with pytest.raises(ValueError) as refused:
        Marker(_q(0), control_type=E.fine)
    assert str(refused.value) == (
        "A Marker is one of ['coda', 'repeat_start', 'segno'], not 'fine'"
    )
    with pytest.raises(ValueError) as refused:
        Jump(_q(4), _q(0), control_type=E.segno)
    assert str(refused.value).startswith("A Jump is one of ['da_capo', ")
    with pytest.raises(TypeError):
        Marker(_q(0))  # type: ignore[call-arg]


def test_only_a_repeat_end_has_a_repeat_count() -> None:
    with pytest.raises(ValueError) as refused:
        Jump(_q(4), _q(0), control_type=E.da_capo, repeat_count=2)
    assert str(refused.value) == (
        "A da_capo jump has no repeat_count: only a repeat_end is taken several times"
    )
    repeat = Jump(_q(4), _q(0), control_type=E.repeat_end, repeat_count=2)
    assert [repeat.is_active(n) for n in (1, 2, 3)] == [True, True, False]
    to_coda = Jump(_q(4), _q(8), control_type=E.to_coda)
    assert [to_coda.is_active(1), to_coda.is_active(1, after_dc_ds=True)] == [
        False,
        True,
    ]
    assert Jump(_q(4), _q(0), control_type=E.da_capo).is_active(5) is True


@pytest.mark.parametrize(
    ("flow", "message"),
    [
        (
            [_element("dc", "jump_from", "da_capo", 8, repeat_count=2)],
            "jump_from 'dc' holds ['repeat_count'], which only a repeat end or a "
            "jump has",
        ),
        (
            [_element("s", "jump_to", "segno", 4, target="s")],
            "jump_to 's' holds ['target'], which only a repeat end or a jump has",
        ),
        (
            [_element("f", "flow_marker", "fine", 4, repeat_count=1)],
            "flow_marker 'f' holds ['repeat_count'], which only a repeat end or a "
            "jump has",
        ),
        (
            [_element("ds", "jump_from", "dal_segno", 8, target="nowhere")],
            "jump_from 'ds' targets 'nowhere', which names no jump_to of this "
            "timeline. Available jump_to elements: []",
        ),
        (
            [_element("ds", "jump_from", "dal_segno", 8)],
            "jump_from 'ds' (dal_segno) names no target: only a repeat end or a D.C. "
            "lands at the timeline's start",
        ),
        (
            [_element("x", "jump_over", "segno", 8)],
            "a flow-control element has the unknown type 'jump_over'; an element is "
            "one of ['flow_marker', 'jump_from', 'jump_to']",
        ),
        (
            [_element("s", "jump_to", "fine", 8)],
            "jump_to 's' has the manifestation 'fine'; a jump_to is one of "
            "['coda', 'repeat_start', 'segno']",
        ),
        (
            [_element("back", "jump_from", "repeat_end", 8, repeat_count=0)],
            "jump_from 'back' has a repeat_count 0, which is no integer of at least 1",
        ),
    ],
    ids=[
        "repeat_count_on_da_capo",
        "target_on_jump_to",
        "repeat_count_on_flow_marker",
        "dangling_target",
        "missing_target",
        "unknown_type",
        "foreign_manifestation",
        "repeat_count_zero",
    ],
)
def test_reading_refuses_what_the_format_does_not_define(
    flow: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError) as refused:
        Timeline.from_dict(_manifest(flow))
    assert str(refused.value) == message


# endregion

# region Score loaders


def test_a_score_s_registry_writes_and_reads_back() -> None:
    timeline = (
        Ms3Loader()
        .load(SCORE / "couperin_concerts/c05n05_musete.measures.tsv")
        .create_timeline()
    )
    written = json.loads(json.dumps(timeline.to_dict()))
    read = Timeline.from_dict(written)
    source, restored = timeline.flow_control, read.flow_control

    def jumps(registry: Any) -> list[tuple[Any, ...]]:
        return [
            (
                j.id,
                j.control_type,
                j.from_position,
                j.to_position,
                j.repeat_count,
                j.condition,
            )
            for j in registry.jumps
        ]

    assert restored.breaks == source.breaks
    assert (
        jumps(restored)
        == jumps(source)
        == [
            ("jump_from_1", E.repeat_end, 42, 0, 1, ActivationCondition.first_n),
            ("jump_from_2", E.repeat_end, 84, 42, 1, ActivationCondition.first_n),
            ("jump_from_3", E.dal_segno_al_fine, 84, 12, 1, ActivationCondition.always),
            ("jump_from_4", E.repeat_end, 162, 84, 1, ActivationCondition.first_n),
            (
                "jump_from_5",
                E.dal_segno_al_fine,
                162,
                12,
                1,
                ActivationCondition.always,
            ),
        ]
    )
    assert [(m.id, m.control_type, m.position) for m in restored.markers] == [
        ("jump_to_1", E.segno, 12),
        ("jump_to_2", E.repeat_start, 42),
        ("jump_to_3", E.repeat_start, 84),
    ]
    assert read.to_dict() == written


# endregion
