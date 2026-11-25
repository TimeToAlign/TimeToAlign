"""Flow control as a manifest writes it (see README, *Flow control*).

``flow_control`` is one array of elements, each stating its ``role``; the
library's five classes read and write it. The tests here read and write the
array itself; whole manifests are read through the format's validator.
"""

from __future__ import annotations

import copy
import json
import re
from fractions import Fraction
from typing import Any

import pytest

from timetoalign import Ms3Loader
from timetoalign.core import Coordinate, Duration, FlowManifestation, TimeUnit
from timetoalign.core.provenance import Belief, Provenance
from timetoalign.testdata import ensure_data
from timetoalign.timelines.flowcontrol import (
    Barline,
    Break,
    DestinationReference,
    FlowCondition,
    FlowControlElement,
    FlowControlRegistry,
    Jump,
    JumpDestination,
    JumpTarget,
    PassesFrom,
    SectionBreak,
)

F = Fraction
M = FlowManifestation
SCORE = ensure_data("score")
EDITOR = "https://example.org/people/editor"


def _q(value: Any) -> Coordinate:
    return Coordinate(F(value), TimeUnit.quarters)


def _read(data: list[dict[str, Any]], **axis: Any) -> FlowControlRegistry:
    return FlowControlRegistry.from_dict(
        copy.deepcopy(data), **({"unit": "quarters"} | axis)
    )


# region The literal

LITERAL: list[dict[str, Any]] = [
    {
        "id": "start",
        "role": "jump_target",
        "coordinate": [0, 1],
        "manifestation": "repeat_start",
        "name": "||:",
    },
    {
        "id": "segno",
        "role": "jump_target",
        "coordinate": [4, 1],
        "manifestation": "segno",
        "meta": {"glyph": "U+1D10B"},
    },
    {
        "id": "movement",
        "role": "section_break",
        "coordinate": [8, 1],
        "label": "end of A",
    },
    {
        "id": "back",
        "role": "jump",
        "coordinate": [8, 1],
        "targets": [
            {
                "to": "start",
                "repeat_count": 2,
                "manifestation": "repeat_end",
                "label": "twice",
                "x-note": "kept",
            },
            {
                "by": [-4, 1],
                "passes": {"from": 2},
                "after": {"jump": "ds", "to": "segno", "x-why": "kept"},
                "determined": False,
                "reset": False,
                "manifestation": "ossia",
            },
        ],
        "label": "end of the first part",
    },
    {
        "id": "brackets",
        "role": "jump",
        "coordinate": [12, 1],
        "targets": [
            {"by": [0, 1], "passes": [1, 3], "manifestation": "volta"},
            {"to": "second", "passes": [2], "manifestation": "volta"},
        ],
    },
    {
        "id": "second",
        "role": "jump_target",
        "coordinate": [16, 1],
        "manifestation": "volta",
    },
    {
        "id": "between",
        "role": "break",
        "coordinate": [16, 1],
        "manifestation": "volta",
    },
    {
        "id": "fine",
        "role": "break",
        "coordinate": [20, 1],
        "manifestation": "fine",
        "ends": {"passes": [2], "after": {"jump": "ds"}, "x-ends": 1},
        "name": "Fine",
    },
    {
        "id": "to_coda",
        "role": "jump",
        "coordinate": [24, 1],
        "targets": [
            {
                "to": "coda",
                "after": {"jump": "back", "by": [-4, 1]},
                "manifestation": "to_coda",
            }
        ],
    },
    {
        "id": "ds",
        "role": "jump",
        "coordinate": [28, 1],
        "targets": [
            {
                "to": "segno",
                "repeat_count": 1,
                "reset": False,
                "manifestation": "dal_segno_al_coda",
            },
            {"repeat_count": 1, "reset": False, "manifestation": "da_capo"},
        ],
        "provenance": {
            "stated_by": [EDITOR],
            "method": "manual",
            "beliefs": [{"actor": EDITOR, "belief": "true", "date": "2026-11-25"}],
        },
    },
    {
        "id": "coda",
        "role": "jump_target",
        "coordinate": [30, 1],
        "manifestation": "coda",
    },
    {
        "id": "double",
        "role": "barline",
        "coordinate": [32, 1],
        "manifestation": "double_barline",
        "x-note": "kept",
    },
]
LITERAL.sort(key=lambda element: (element["coordinate"][0], element["id"]))


def _complete(element: dict[str, Any]) -> dict[str, Any]:
    return {"name": None, "label": None, "meta": {}, **element}


def test_the_literal_round_trips() -> None:
    registry = _read(LITERAL)
    assert registry.to_dict() == LITERAL
    assert registry.to_dict(minimal=False) == [_complete(e) for e in LITERAL]
    assert _read([_complete(e) for e in LITERAL]).to_dict() == LITERAL
    assert json.loads(json.dumps(registry.to_dict())) == LITERAL


def test_the_defaults_of_a_destination_are_read_and_never_written() -> None:
    element = {
        "id": "j",
        "role": "jump",
        "coordinate": [8, 1],
        "targets": [{"repeat_count": 1, "determined": True, "reset": True}],
    }
    registry = _read([element])
    (destination,) = registry.jumps[0].targets
    assert (destination.determined, destination.reset) == (True, True)
    for minimal in (True, False):
        (written,) = registry.to_dict(minimal=minimal)
        assert written["targets"] == [{"repeat_count": 1}]


def test_reading_builds_the_library_s_elements() -> None:
    registry = _read(LITERAL)

    assert [(type(e), e.id, e.coordinate) for e in registry.elements] == [
        (JumpTarget, "start", _q(0)),
        (JumpTarget, "segno", _q(4)),
        (Jump, "back", _q(8)),
        (SectionBreak, "movement", _q(8)),
        (Jump, "brackets", _q(12)),
        (Break, "between", _q(16)),
        (JumpTarget, "second", _q(16)),
        (Break, "fine", _q(20)),
        (Jump, "to_coda", _q(24)),
        (Jump, "ds", _q(28)),
        (JumpTarget, "coda", _q(30)),
        (Barline, "double", _q(32)),
    ]
    assert [e.id for e in registry.jumps] == ["back", "brackets", "to_coda", "ds"]
    assert [e.id for e in registry.jump_targets] == ["start", "segno", "second", "coda"]
    assert [e.id for e in registry.breaks] == ["between", "fine"]
    assert [e.id for e in registry.section_breaks] == ["movement"]
    assert [e.id for e in registry.barlines] == ["double"]

    assert [
        (e.id, e.manifestation, e.name, e.label, e.meta)
        for e in registry.elements
        if not isinstance(e, Jump)
    ] == [
        ("start", M.repeat_start, "||:", None, {}),
        ("segno", M.segno, None, None, {"glyph": "U+1D10B"}),
        ("movement", None, None, "end of A", {}),
        ("between", M.volta, None, None, {}),
        ("second", M.volta, None, None, {}),
        ("fine", M.fine, "Fine", None, {}),
        ("coda", M.coda, None, None, {}),
        ("double", M.double_barline, None, None, {}),
    ]

    back, brackets, to_coda, ds = registry.jumps
    assert back.label == "end of the first part"
    assert back.targets == (
        JumpDestination(
            to="start", repeat_count=2, manifestation=M.repeat_end, label="twice"
        ),
        JumpDestination(
            by=Duration(F(-4), "quarters"),
            passes=PassesFrom(2),
            after=DestinationReference(jump="ds", to="segno"),
            determined=False,
            reset=False,
            manifestation=M.ossia,
        ),
    )
    assert back.manifestations == (M.repeat_end, M.ossia)
    assert brackets.targets == (
        JumpDestination(
            by=Duration(F(0), "quarters"), passes=(1, 3), manifestation="volta"
        ),
        JumpDestination(to="second", passes=(2,), manifestation="volta"),
    )
    assert [d.goes_on for d in brackets.targets] == [True, False]
    assert to_coda.targets[0].after == DestinationReference(
        jump="back", by=Duration(F(-4), "quarters")
    )
    assert ds.provenance == Provenance(
        stated_by=(EDITOR,),
        method="manual",
        beliefs=(Belief(actor=EDITOR, belief="true", date="2026-11-25"),),
    )
    assert registry.breaks[1].ends == FlowCondition(
        passes=(2,), after=DestinationReference(jump="ds")
    )
    assert registry.destination_of(registry.breaks[1].ends.after) == (ds, 1)
    assert registry.destination_of(back.targets[1].after) == (ds, 0)
    assert registry.destination_of(to_coda.targets[0].after) == (back, 1)


@pytest.mark.parametrize(
    ("number_type", "coordinate", "by", "value", "distance"),
    [
        ("int", 9, -2, 9, -2),
        ("fraction", [9, 2], [-1, 2], F(9, 2), F(-1, 2)),
        ("float", 4.5, -0.5, 4.5, -0.5),
    ],
)
def test_coordinate_and_by_are_bare_values_of_the_axis(
    number_type: str, coordinate: Any, by: Any, value: Any, distance: Any
) -> None:
    unit = "ticks" if number_type == "int" else "quarters"
    element = {
        "id": "j",
        "role": "jump",
        "coordinate": coordinate,
        "targets": [{"by": by, "repeat_count": 1}],
    }
    registry = _read([element], unit=unit, number_type=number_type)
    (jump,) = registry.jumps
    assert jump.coordinate.value == value
    assert type(jump.coordinate.value) is type(value)
    assert jump.targets[0].by == Duration(distance, unit, number_type=number_type)
    assert type(jump.targets[0].by.value) is type(distance)
    assert registry.to_dict() == [element]


def test_a_value_of_another_number_type_is_refused() -> None:
    element = {"id": "t", "role": "jump_target", "coordinate": 8}
    with pytest.raises(ValueError) as error:
        _read([element])
    assert str(error.value) == (
        "coordinate 8 of 't' is an int on a fraction axis: write [8, 1]"
    )
    jump = {"id": "j", "role": "jump", "coordinate": [8, 1], "targets": [{"by": -4}]}
    with pytest.raises(ValueError) as error:
        _read([jump])
    assert str(error.value) == (
        "by -4 of 'destination 1 of jump 'j'' is an int on a fraction axis: "
        "write [-4, 1]"
    )


# endregion

# region Manifestations


def test_a_manifestation_outside_the_vocabulary_is_kept() -> None:
    word = "x-wavy-line"
    elements = [
        {"id": "a", "role": "jump_target", "coordinate": [0, 1], "manifestation": word},
        {"id": "b", "role": "break", "coordinate": [16, 1], "manifestation": word},
        {
            "id": "c",
            "role": "section_break",
            "coordinate": [16, 1],
            "manifestation": word,
        },
        {"id": "d", "role": "barline", "coordinate": [16, 1], "manifestation": word},
        {
            "id": "e",
            "role": "jump",
            "coordinate": [16, 1],
            "targets": [{"to": "a", "repeat_count": 1, "manifestation": word}],
        },
    ]
    registry = _read(elements)
    assert [e.manifestation for e in registry.elements if not isinstance(e, Jump)] == [
        word
    ] * 4
    assert registry.jumps[0].manifestations == (word,)
    assert all(type(m) is str for m in registry.jumps[0].manifestations)
    assert registry.to_dict() == elements


@pytest.mark.parametrize(
    ("element", "message"),
    [
        (
            {
                "id": "b",
                "role": "break",
                "coordinate": [8, 1],
                "manifestation": "segno",
            },
            "break 'b' is one of ['fine', 'volta'] or a word outside the "
            "vocabulary, not 'segno'",
        ),
        (
            {
                "id": "s",
                "role": "section_break",
                "coordinate": [8, 1],
                "manifestation": "fine",
            },
            "section_break 's' is one of [] or a word outside the vocabulary, "
            "not 'fine'",
        ),
        (
            {
                "id": "l",
                "role": "barline",
                "coordinate": [8, 1],
                "manifestation": "repeat_start",
            },
            "barline 'l' is one of ['double_barline', 'final_barline'] or a word "
            "outside the vocabulary, not 'repeat_start'",
        ),
        (
            {
                "id": "t",
                "role": "jump_target",
                "coordinate": [8, 1],
                "manifestation": "fine",
            },
            "jump_target 't' is one of ['coda', 'ossia', 'repeat_start', 'segno', "
            "'volta'] or a word outside the vocabulary, not 'fine'",
        ),
        (
            {
                "id": "j",
                "role": "jump",
                "coordinate": [8, 1],
                "targets": [{"manifestation": "coda"}],
            },
            "destination 1 of jump 'j': a destination is one of ['da_capo', "
            "'da_capo_al_coda', 'da_capo_al_fine', 'dal_segno', 'dal_segno_al_coda', "
            "'dal_segno_al_fine', 'ossia', 'repeat_end', 'to_coda', 'volta'] or a "
            "word outside the vocabulary, not 'coda'",
        ),
    ],
)
def test_each_class_holds_its_own_manifestations(
    element: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        _read([element])
    assert str(error.value) == message


# endregion

# region The form with type and target


@pytest.mark.parametrize(
    ("element", "message"),
    [
        (
            {
                "id": "back",
                "type": "jump_from",
                "manifestation": "repeat_end",
                "coordinate": [8, 1],
            },
            "flow-control element 'back' is a 'jump_from': an element states its "
            "`role`, one of ['barline', 'break', 'jump', 'jump_target', "
            "'section_break']: a jump_from is written as a `jump` with `targets`",
        ),
        (
            {
                "id": "top",
                "type": "jump_to",
                "manifestation": "segno",
                "coordinate": [8, 1],
            },
            "flow-control element 'top' is a 'jump_to': an element states its "
            "`role`, one of ['barline', 'break', 'jump', 'jump_target', "
            "'section_break']: a jump_to is written as a `jump_target`",
        ),
        (
            {"id": "fine", "role": "flow_marker", "coordinate": [8, 1]},
            "flow-control element 'fine' is a 'flow_marker': an element states its "
            "`role`, one of ['barline', 'break', 'jump', 'jump_target', "
            "'section_break']: a flow_marker is written as a `break`, a "
            "`section_break` or a `barline`",
        ),
        (
            {"id": "back", "role": "jump", "coordinate": [8, 1], "target": "top"},
            "flow-control element 'back' holds `target`: a jump names where it "
            "goes on in `targets`, a list of destinations such as {to: <ID>}",
        ),
        (
            {
                "id": "back",
                "role": "jump",
                "coordinate": [8, 1],
                "targets": [{}],
                "repeat_count": 2,
            },
            "flow-control element 'back' holds `repeat_count`: it is a key of a "
            "destination in the `targets` of a jump",
        ),
        (
            {"id": "x", "role": "marker", "coordinate": [8, 1]},
            "flow-control element 'x' has the unknown role 'marker'; an element is "
            "one of ['barline', 'break', 'jump', 'jump_target', 'section_break']",
        ),
    ],
)
def test_the_form_with_type_and_target_is_refused(
    element: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        _read([element])
    assert str(error.value) == message


def test_a_class_reads_its_own_role_only() -> None:
    element = {"id": "t", "role": "jump_target", "coordinate": [8, 1]}
    axis = {
        "unit": TimeUnit.quarters,
        "number_type": TimeUnit.quarters.default_number_type,
    }
    assert type(FlowControlElement.from_dict(element, **axis)) is JumpTarget
    assert JumpTarget.from_dict(element, **axis).coordinate == _q(8)
    with pytest.raises(ValueError) as error:
        Break.from_dict(element, **axis)
    assert str(error.value) == "flow-control element 't' is a jump_target, not a break"


# endregion

# region Refusals on reading

_J = {"id": "j", "role": "jump", "coordinate": [8, 1]}


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        (
            [{**_J, "targets": []}],
            "jump 'j' has no destination: `targets` lists at least one",
        ),
        (
            [{"id": "j", "role": "jump", "coordinate": [8, 1]}],
            "jump 'j' lacks `targets`, the list of its destinations",
        ),
        (
            [{**_J, "targets": [{"to": "t", "by": [0, 1]}]}],
            "destination 1 of jump 'j': a destination states both `to` ('t') and "
            "`by`: a place is one of them",
        ),
        (
            [{**_J, "targets": [{"repeat_count": 0}]}],
            "destination 1 of jump 'j': repeat_count 0 is no integer of at least 1",
        ),
        (
            [{**_J, "targets": [{"repeat_count": True}]}],
            "destination 1 of jump 'j': repeat_count True is no integer of at least 1",
        ),
        (
            [{**_J, "targets": [{"repeat_count": 1.5}]}],
            "destination 1 of jump 'j': repeat_count 1.5 is no integer of at least 1",
        ),
        (
            [{**_J, "targets": [{"passes": []}]}],
            "destination 1 of jump 'j': passes lists at least one pass",
        ),
        (
            [{**_J, "targets": [{"passes": [1, 0]}]}],
            "destination 1 of jump 'j': passes [1, 0]: passes are counted from 1",
        ),
        (
            [{**_J, "targets": [{"passes": [2, 2]}]}],
            "destination 1 of jump 'j': passes [2, 2] names a pass twice",
        ),
        (
            [{**_J, "targets": [{"passes": {"from": 0}}]}],
            "destination 1 of jump 'j': passes from 0: passes are counted from 1",
        ),
        (
            [{**_J, "targets": [{"passes": 2}]}],
            "destination 1 of jump 'j': passes 2 is neither a list of pass numbers "
            "nor {from: n}",
        ),
        (
            [{**_J, "targets": [{"repeat_count": 1}, {"determined": "no"}]}],
            "destination 2 of jump 'j': determined 'no' is neither true nor false",
        ),
        (
            [{"id": "b", "role": "break", "coordinate": [8, 1], "ends": {}}],
            "the `ends` of break 'b' states no condition: leave `ends` out for a "
            "break that always ends the flow",
        ),
        (
            [{**_J, "targets": [{"to": "nowhere"}]}],
            "jump 'j' goes on at 'nowhere', which names no jump target of this "
            "registry and no place of its timeline. Available jump targets: []. "
            "Available places: []",
        ),
        (
            [{"id": "t", "role": "jump_target"}],
            "a flow-control element lacks the key(s) ['coordinate']",
        ),
        (
            [{**_J, "targets": [{}], "provenance": {"method": "guessed"}}],
            "the provenance of jump 'j': 1 validation error for Provenance",
        ),
    ],
)
def test_reading_refuses_what_cannot_be_played(
    elements: list[dict[str, Any]], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        _read(elements)
    assert str(error.value).startswith(message)
    if "validation error" not in message:
        assert str(error.value) == message


def test_the_array_must_be_an_array() -> None:
    with pytest.raises(
        ValueError, match=re.escape("the flow_control must be an array")
    ):
        FlowControlRegistry.from_dict({}, unit="quarters")  # type: ignore[arg-type]


# endregion

# region Score loaders


def test_a_score_s_flow_control_round_trips_as_its_array() -> None:
    timeline = (
        Ms3Loader()
        .load(SCORE / "couperin_concerts/c05n05_musete.measures.tsv")
        .create_timeline()
    )
    registry = timeline.flow_control

    # The measures table: a segno at MC 6 (12 quarters); MC 16 ends at 42
    # with a repeat end and a Fine; MC 17 and MC 32 start repeats at 42 and
    # 84 and carry the markers 1e and 2e; MC 31 and MC 58 end at 84 and 162,
    # each with a repeat end and a D.S. al Fine.
    assert [(type(e), e.id, e.coordinate, e.name) for e in registry.elements] == [
        (JumpTarget, "jump_target_3", _q(12), "segno"),
        (Break, "break_1", _q(42), "fine"),
        (Jump, "jump_1", _q(42), None),
        (JumpTarget, "jump_target_1", _q(42), None),
        (JumpTarget, "jump_target_4", _q(42), "1e"),
        (Jump, "jump_2", _q(84), None),
        (JumpTarget, "jump_target_2", _q(84), None),
        (JumpTarget, "jump_target_5", _q(84), "2e"),
        (Jump, "jump_3", _q(162), None),
    ]
    assert _read(registry.to_dict()).to_dict() == registry.to_dict()


# endregion
