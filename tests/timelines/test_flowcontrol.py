"""Flow-control elements and their registry (see README, *Flow control*)."""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import (
    Coordinate,
    Duration,
    FlowManifestation,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.timelines import Timeline
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
from timetoalign.timelines.types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    SegmentLine,
)

F = Fraction
EDITOR = "https://example.org/people/editor"
REVIEWER = "https://example.org/people/reviewer"


def _q(value: Any) -> Coordinate:
    return Coordinate(value, TimeUnit.quarters)


def _t(value: Any) -> Coordinate:
    return Coordinate(value, TimeUnit.ticks)


def D(**keys: Any) -> JumpDestination:
    return JumpDestination(**keys)


def after(jump: str, **place: Any) -> DestinationReference:
    return DestinationReference(jump=jump, **place)


def _refused(build: Any) -> str:
    with pytest.raises(ValueError) as error:
        build()
    return str(error.value)


# region The classes


def test_each_class_names_its_role_and_mints_its_id_from_it() -> None:
    made = [
        Jump(_q(8), [D(repeat_count=1)]),
        JumpTarget(_q(8)),
        Break(_q(8)),
        SectionBreak(_q(8)),
        Barline(_q(8)),
    ]
    assert [(type(e).role, e.id) for e in made] == [
        ("jump", "jump_1"),
        ("jump_target", "jump_target_1"),
        ("break", "break_1"),
        ("section_break", "section_break_1"),
        ("barline", "barline_1"),
    ]
    assert all(isinstance(e, FlowControlElement) for e in made)
    assert [e.to_dict()["role"] for e in made] == [type(e).role for e in made]


def test_a_destination_states_its_place_and_its_conditions() -> None:
    plain = D()
    assert (plain.to, plain.by, plain.passes, plain.repeat_count, plain.after) == (
        None,
        None,
        None,
        None,
        None,
    )
    assert (plain.determined, plain.reset, plain.manifestation, plain.label) == (
        True,
        True,
        None,
        None,
    )
    assert plain.place is None and not plain.goes_on
    assert D(to="segno").place == ("to", "segno")
    assert D(passes=[1, 3]).passes == (1, 3)
    assert D(passes=PassesFrom(2)).passes == PassesFrom(2)
    assert D(manifestation="volta").manifestation is FlowManifestation.volta
    assert (
        D(manifestation=FlowManifestation.dc).manifestation is FlowManifestation.da_capo
    )


def test_a_jump_puts_its_distances_on_its_axis() -> None:
    jump = Jump(
        _q(F(9, 2)),
        [D(by=0, passes=[1]), D(by=F(-1, 2), repeat_count=1, after=after("x", by=-1))],
    )
    assert [d.by for d in jump.targets] == [
        Duration(F(0), "quarters"),
        Duration(F(-1, 2), "quarters"),
    ]
    assert [d.goes_on for d in jump.targets] == [True, False]
    assert jump.targets[1].after == after("x", by=Duration(F(-1), "quarters"))
    assert _refused(lambda: Jump(_q(8), [D(by=Duration(1, "seconds"))])) == (
        "a `by` of jump 'jump_1' is a distance in seconds on an axis in quarters"
    )


def test_a_single_destination_may_be_given_bare() -> None:
    assert Jump(_q(8), D(repeat_count=1)).targets == (D(repeat_count=1),)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Jump(_q(8), []),
            "jump 'jump_1' has no destination: `targets` lists at least one",
        ),
        (
            lambda: Jump(_q(8), [D(to="a"), D(to="a", repeat_count=1)]),
            "jump 'jump_1' has 2 destinations with `to: a`: the destinations of one "
            "jump are distinct places",
        ),
        (
            lambda: Jump(_q(8), [D(), D(repeat_count=1, reset=False)]),
            "jump 'jump_1' has 2 destinations with no place: the destinations of "
            "one jump are distinct places",
        ),
        (
            lambda: Jump(
                Coordinate(8.0, "seconds"), [D(by=1), D(by=Duration(1.0, "seconds"))]
            ),
            "jump 'jump_1' has 2 destinations with `by: 1.0`: the destinations of "
            "one jump are distinct places",
        ),
        (
            lambda: D(to="a", by=0),
            "a destination states both `to` ('a') and `by`: a place is one of them",
        ),
        (lambda: D(repeat_count=0), "repeat_count 0 is no integer of at least 1"),
        (lambda: D(passes=[]), "passes lists at least one pass"),
        (lambda: D(passes=[0]), "passes [0]: passes are counted from 1"),
        (lambda: D(passes=[-1, 2]), "passes [-1, 2]: passes are counted from 1"),
        (lambda: D(passes=[1, 1]), "passes [1, 1] names a pass twice"),
        (lambda: D(passes=[1.0]), "passes [1.0]: a pass is an integer"),
        (lambda: PassesFrom(0), "passes from 0: passes are counted from 1"),
        (
            lambda: Break(_q(8), ends=FlowCondition()),
            "the `ends` of break 'break_1' states no condition: leave `ends` out "
            "for a break that always ends the flow",
        ),
        (
            lambda: Break(_q(8), ends=D(repeat_count=1)),
            "the `ends` of break 'break_1' is a FlowCondition, not JumpDestination",
        ),
        (
            lambda: after("j", to="a", by=1),
            "`after` naming jump 'j' states both `to` ('a') and `by`: a place is "
            "one of them",
        ),
        (
            lambda: Break(_q(8), manifestation=FlowManifestation.segno),
            "break 'break_1' is one of ['fine', 'volta'] or a word outside the "
            "vocabulary, not 'segno'",
        ),
    ],
)
def test_an_object_that_states_nothing_playable_is_refused(
    build: Any, message: str
) -> None:
    assert _refused(build) == message


def test_an_element_lies_at_a_coordinate() -> None:
    with pytest.raises(TypeError) as error:
        JumpTarget(8)  # type: ignore[arg-type]
    assert str(error.value) == "a jump_target lies at a Coordinate, not at int 8"
    target = JumpTarget(IdCoordinate(8, "quarters", "clt_1"))
    assert type(target.coordinate) is Coordinate and target.coordinate == _q(8)


def test_the_names_of_the_form_before_are_gone() -> None:
    import timetoalign.core as core
    import timetoalign.timelines.flowcontrol as flowcontrol

    for name in ("Marker", "ActivationCondition"):
        assert not hasattr(flowcontrol, name) and not hasattr(core, name)
    for name in ("control_type", "condition", "is_active", "remaining_activations"):
        assert not hasattr(Jump, name) and not hasattr(Break, name)
    for name in ("from_coordinate", "to_coordinate", "is_backward", "distance"):
        assert not hasattr(Jump, name)
    assert not hasattr(FlowControlRegistry, "add_break")


# endregion

# region The registry refuses


def _registry(
    *elements: FlowControlElement, unit: str = "ticks"
) -> FlowControlRegistry:
    registry = FlowControlRegistry(unit=unit)
    registry.add_all(elements)
    return registry


def test_a_second_jump_at_a_coordinate_is_refused() -> None:
    registry = _registry(Jump(_t(8), D(repeat_count=1), id="repeat"))
    before = list(registry.elements)
    assert _refused(
        lambda: registry.add(Jump(_t(8), D(repeat_count=1, reset=False), id="ds"))
    ) == (
        "jump 'ds' lies at 8 ticks, where jump 'repeat' lies already: one jump lies "
        "at a coordinate, and several signs there are its destinations"
    )
    assert registry.elements == before
    assert registry.jump_at(8).id == "repeat"
    assert registry.jump_at(_t(8)).id == "repeat"
    assert registry.jump_at(4) is None


def test_a_to_naming_nothing_is_refused() -> None:
    registry = _registry(JumpTarget(_t(4), id="segno"))
    assert _refused(lambda: registry.add(Jump(_t(8), D(to="coda"), id="j"))) == (
        "jump 'j' goes on at 'coda', which names no jump target of this registry "
        "and no place of its timeline. Available jump targets: ['segno']. "
        "Available places: []"
    )
    assert [e.id for e in registry.elements] == ["segno"]


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (
            after("nowhere"),
            "the `after` of jump 'j' names 'nowhere', which is no jump of this "
            "registry. Available jumps: ['two', 'j']",
        ),
        (
            after("j"),
            "the `after` of jump 'j' names a destination of the same jump: it names "
            "one of another jump",
        ),
        (
            after("two", to="coda"),
            "the `after` of jump 'j' names the destination of jump 'two' with "
            "`to: coda`, which names 0 of its 2 destinations instead of one",
        ),
        (
            after("two"),
            "the `after` of jump 'j' names the destination of jump 'two' without "
            "place, which names 0 of its 2 destinations instead of one",
        ),
    ],
)
def test_an_after_names_exactly_one_destination_of_another_jump(
    reference: DestinationReference, message: str
) -> None:
    registry = _registry(
        JumpTarget(_t(2), id="segno"),
        Jump(
            _t(4), [D(to="segno", repeat_count=1), D(by=-1, repeat_count=1)], id="two"
        ),
    )
    before = list(registry.elements)
    assert _refused(lambda: registry.add(Jump(_t(8), D(after=reference), id="j"))) == (
        message
    )
    assert registry.elements == before


def test_an_after_finds_its_destination_by_its_place() -> None:
    registry = _registry(
        JumpTarget(_t(2), id="segno"),
        Jump(
            _t(4), [D(to="segno", repeat_count=1), D(by=-1, repeat_count=1)], id="two"
        ),
        Jump(_t(6), D(repeat_count=1), id="one"),
        Break(_t(8), ends=FlowCondition(after=after("two", by=-1)), id="fine"),
    )
    two, one = registry.jumps
    assert registry.destination_of(after("two", to="segno")) == (two, 0)
    assert registry.destination_of(after("two", by=Duration(-1, "ticks"))) == (two, 1)
    assert registry.destination_of(after("one")) == (one, 0)
    assert registry.breaks[0].ends.after == after("two", by=Duration(-1, "ticks"))


def test_another_unit_is_refused_naming_both() -> None:
    registry = FlowControlRegistry(unit="quarters")
    for element in (
        JumpTarget(Coordinate(1, "seconds")),
        Jump(Coordinate(1, "seconds"), D(repeat_count=1)),
    ):
        assert _refused(lambda element=element: registry.add(element)) == (
            "Flow-control coordinate unit seconds does not match registry unit "
            "quarters"
        )
    assert _refused(lambda: registry.jump_at(Coordinate(1, "seconds"))) == (
        "Flow-control coordinate unit seconds does not match registry unit quarters"
    )
    with pytest.raises(TypeError):
        registry.add("jump")  # type: ignore[arg-type]


# endregion

# region Adding together


def test_elements_naming_one_another_are_added_together() -> None:
    fine = Break(_t(4), ends=FlowCondition(after=after("dc")), id="fine")
    to_coda = Jump(_t(6), D(to="coda", after=after("dc")), id="to_coda")
    dc = Jump(_t(8), D(repeat_count=1, reset=False), id="dc")
    coda = JumpTarget(_t(8), id="coda")

    registry = FlowControlRegistry(unit="ticks")
    assert _refused(lambda: registry.add(fine)) == (
        "the `after` of break 'fine' names 'dc', which is no jump of this "
        "registry. Available jumps: []"
    )
    assert registry.elements == []
    stored = registry.add_all([fine, to_coda, dc, coda])
    assert [e.id for e in stored] == ["fine", "to_coda", "dc", "coda"]
    assert [e.id for e in registry.elements] == ["fine", "to_coda", "coda", "dc"]

    one_by_one = FlowControlRegistry(unit="ticks")
    for element in (coda, dc, to_coda, fine):
        one_by_one.add(element)
    assert one_by_one.elements == registry.elements
    assert FlowControlRegistry([fine, to_coda, dc, coda], unit="ticks") == registry


def test_a_minted_id_that_is_taken_is_minted_again() -> None:
    registry = FlowControlRegistry(unit="ticks")
    first = registry.add(JumpTarget(_t(4)))
    second = registry.add(JumpTarget(_t(2)))
    assert (first.id, second.id) == ("jump_target_1", "jump_target_2")
    assert [e.id for e in registry.elements] == ["jump_target_2", "jump_target_1"]
    assert _refused(lambda: registry.add(JumpTarget(_t(6), id="jump_target_1"))) == (
        "jump_target 'jump_target_1' cannot join: its id 'jump_target_1' is already "
        "held by jump_target 'jump_target_1'"
    )


def test_a_destination_follows_a_target_minted_again() -> None:
    source = FlowControlRegistry(unit="quarters")
    target = source.add(JumpTarget(_q(4), manifestation="segno"))
    source.add(Jump(_q(8), D(to=target.id, repeat_count=1, reset=False)))
    source.add(Break(_q(6), ends=FlowCondition(after=after("jump_1", to=target.id))))

    timeline = ContinuousLogicalTimeline(length=16)
    held = timeline.flow_control.add(JumpTarget(_q(2)))
    timeline.flow_control.add(Jump(_q(12), D(to=held.id, repeat_count=1)))
    timeline.flow_control._absorb(source)

    assert [(e.id, e.coordinate.value) for e in timeline.flow_control.elements] == [
        ("jump_target_1", 2),
        ("jump_target_2", 4),
        ("break_1", 6),
        ("jump_2", 8),
        ("jump_1", 12),
    ]
    absorbed = timeline.flow_control.jump_at(8)
    assert absorbed.targets == (D(to="jump_target_2", repeat_count=1, reset=False),)
    assert timeline.flow_control.breaks[0].ends.after == after(
        "jump_2", to="jump_target_2"
    )
    assert timeline.flow_control.jump_at(12).targets[0].to == "jump_target_1"
    # The source registry is what it was.
    assert [e.id for e in source.elements] == ["jump_target_1", "break_1", "jump_1"]


# endregion

# region Canonical order


@pytest.mark.parametrize(
    ("unit", "number_type", "values"),
    [
        ("ticks", "int", [2, 10, 100]),
        ("quarters", "fraction", [F(1, 3), F(1, 2), F(10, 3)]),
        ("seconds", "float", [0.5, 2.0, 10.25]),
    ],
)
def test_elements_are_held_and_written_by_coordinate_then_id(
    unit: str, number_type: str, values: list[Any]
) -> None:
    low, middle, high = values
    registry = FlowControlRegistry(unit=unit, number_type=number_type)
    for at, element_id in ((high, "a"), (middle, "c"), (middle, "b"), (low, "d")):
        registry.add(JumpTarget(Coordinate(at, unit), id=element_id))
    registry.add(Break(Coordinate(middle, unit), id="B"))
    expected = [(low, "d"), (middle, "B"), (middle, "b"), (middle, "c"), (high, "a")]
    assert [(e.coordinate.value, e.id) for e in registry.elements] == expected
    assert [e["id"] for e in registry.to_dict()] == [id_ for _, id_ in expected]


# endregion

# region Places


def test_place_of_a_target_a_distance_and_no_place() -> None:
    registry = _registry(
        JumpTarget(_t(4), id="segno"),
        Jump(
            _t(20),
            [
                D(to="segno"),
                D(by=0),
                D(by=-3),
                D(by=5, repeat_count=1),
                D(repeat_count=1),
            ],
            id="j",
        ),
    )
    (jump,) = registry.jumps
    assert [registry.place_of(d, jump) for d in jump.targets] == [
        _t(4),
        _t(20),
        _t(17),
        _t(25),
        _t(0),
    ]
    assert all(type(registry.place_of(d, jump).value) is int for d in jump.targets)


def test_no_place_is_the_nearest_section_break_before_the_jump() -> None:
    registry = _registry(
        SectionBreak(_t(8)),
        SectionBreak(_t(16)),
        SectionBreak(_t(24)),
        Jump(_t(8), D(repeat_count=1), id="first"),
        Jump(_t(24), D(repeat_count=1), id="third"),
        Jump(_t(30), D(repeat_count=1), id="last"),
    )
    assert [registry.place_of(j.targets[0], j) for j in registry.jumps] == [
        _t(0),
        _t(16),
        _t(24),
    ]
    only_first = [e for e in registry.elements if e.coordinate.value <= 8]
    last = registry.jump_at(30)
    assert registry.place_of(last.targets[0], last, elements=only_first) == _t(8)


def test_a_distance_is_added_exactly_on_its_axis() -> None:
    fraction = FlowControlRegistry(unit="quarters")
    jump = fraction.add(Jump(_q(F(9, 2)), D(by=F(-1, 2), repeat_count=1)))
    assert fraction.place_of(jump.targets[0], jump) == _q(F(4))
    assert type(fraction.place_of(jump.targets[0], jump).value) is Fraction
    seconds = FlowControlRegistry(unit="seconds")
    jump = seconds.add(Jump(Coordinate(4.5, "seconds"), D(by=-0.25, repeat_count=1)))
    assert seconds.place_of(jump.targets[0], jump) == Coordinate(4.25, "seconds")
    assert type(seconds.place_of(jump.targets[0], jump).value) is float


def test_a_destination_names_a_region_a_child_and_the_timeline() -> None:
    timeline = ContinuousLogicalTimeline(length=64, uid="score")
    timeline.create_region(16, 32, id="chorus")
    timeline.create_child(8, 40, uid="trio")
    jump = timeline.flow_control.add(
        Jump(
            _q(48),
            [
                D(to="chorus", repeat_count=1),
                D(to="trio", repeat_count=1),
                D(to="score", repeat_count=1, reset=False),
            ],
        )
    )
    registry = timeline.flow_control
    assert [registry.place_of(d, jump) for d in jump.targets] == [
        _q(F(16)),
        _q(F(40)),
        _q(F(0)),
    ]
    assert _refused(lambda: registry.add(Jump(_q(60), D(to="verse")))) == (
        "jump 'jump_2' goes on at 'verse', which names no jump target of this "
        "registry and no place of its timeline. Available jump targets: []. "
        "Available places: ['score', 'chorus', 'trio']"
    )
    with pytest.raises(KeyError):
        FlowControlRegistry(unit="quarters").place_of(jump.targets[0], jump)


def test_a_destination_follows_a_place_renamed() -> None:
    timeline = ContinuousLogicalTimeline(length=64, uid="score")
    timeline.create_region(16, 32, id="chorus")
    timeline.flow_control.add(Jump(_q(48), D(to="chorus", repeat_count=1)))
    timeline._rename_region("chorus", "refrain")
    timeline._rename_references("chorus", "refrain")
    (jump,) = timeline.flow_control.jumps
    assert jump.targets[0].to == "refrain"
    assert timeline.flow_control.place_of(jump.targets[0], jump) == _q(F(16))


# endregion

# region The axis


def test_a_bound_registry_stores_on_its_number_type() -> None:
    fractions = FlowControlRegistry(unit="quarters", number_type="fraction")
    target = fractions.add(JumpTarget(Coordinate(1.5, "quarters", number_type="float")))
    jump = fractions.add(
        Jump(
            Coordinate(4.5, "quarters", number_type="float"),
            D(by=Duration(-1.5, "quarters", number_type="float"), repeat_count=1),
        )
    )
    assert target.coordinate.value == F(3, 2) and type(target.coordinate.value) is F
    assert jump.coordinate.value == F(9, 2)
    assert jump.targets[0].by.value == F(-3, 2)
    assert type(jump.targets[0].by.value) is F

    floats = FlowControlRegistry(unit="seconds", number_type="float")
    target = floats.add(
        JumpTarget(Coordinate(F(1, 3), "seconds", number_type="fraction"))
    )
    assert target.coordinate.value == 0.3333333333333333
    assert type(target.coordinate.value) is float
    assert floats.number_type is NumberType.float


def test_an_unbound_registry_stores_the_object_given() -> None:
    registry = FlowControlRegistry()
    target = JumpTarget(_q(4), id="segno")
    jump = Jump(_q(8), D(to="segno", repeat_count=1))
    assert registry.add(target) is target
    assert registry.add(jump) is jump
    assert _refused(lambda: registry.jump_at(8.5) or registry._coordinate(8)) == (
        "A bare coordinate requires a bound registry unit"
    )


def test_constructor_content_is_added_and_the_caller_s_list_untouched() -> None:
    given = [JumpTarget(_q(8), id="b"), JumpTarget(_q(4), id="a")]
    registry = FlowControlRegistry(given, unit="quarters")
    assert [e.id for e in given] == ["b", "a"]
    assert [e.id for e in registry.elements] == ["a", "b"]
    assert registry.elements is not given


def test_a_copy_is_equal_and_independent() -> None:
    timeline = ContinuousLogicalTimeline(length=16)
    timeline.flow_control.add(JumpTarget(_q(4), id="segno"))
    duplicate = timeline.flow_control.copy()
    assert duplicate == timeline.flow_control
    assert duplicate.elements is not timeline.flow_control.elements
    duplicate.add(JumpTarget(_q(8), id="segno2"))
    assert [e.id for e in timeline.flow_control.elements] == ["segno"]
    assert timeline.flow_control.has_flow_control()
    duplicate.clear()
    assert not duplicate.has_flow_control()
    assert timeline.flow_control.has_flow_control()


@pytest.mark.parametrize(
    "timeline",
    [
        Timeline(length=1),
        ContinuousLogicalTimeline(length=1),
        DiscreteLogicalTimeline(length=1),
        ContinuousPhysicalTimeline(length=1),
        DiscretePhysicalTimeline(length=1),
        ContinuousGraphicalTimeline(length=1),
        DiscreteGraphicalTimeline(length=1),
        SegmentLine[ContinuousPhysicalTimeline](),
    ],
    ids=(
        "timeline",
        "continuous-logical",
        "discrete-logical",
        "continuous-physical",
        "discrete-physical",
        "continuous-graphical",
        "discrete-graphical",
        "segment-line",
    ),
)
def test_every_timeline_has_fresh_axis_bound_storage(timeline: Any) -> None:
    assert timeline.flow_control.unit is timeline.unit
    assert timeline.flow_control.number_type is timeline.number_type
    assert timeline.flow_control.elements == []
    assert timeline.measure_map is None
    assert timeline.metric_hierarchy is None


def test_reach_counts_every_element_and_every_distance() -> None:
    timeline = ContinuousLogicalTimeline(length=None)
    timeline.flow_control.add(JumpTarget(_q(6)))
    assert timeline.reach == _q(F(6))
    timeline.flow_control.add(Jump(_q(8), D(by=4)))
    assert timeline.reach == _q(F(12))


# endregion

# region Rejection


def _believed(actor: str, belief: str) -> dict[str, Any]:
    return {
        "method": "imported",
        "beliefs": [{"actor": actor, "belief": belief, "date": "2026-11-25"}],
    }


def test_a_rejected_element_is_left_out_for_its_believer() -> None:
    registry = _registry(
        JumpTarget(_t(2), id="kept"),
        JumpTarget(_t(4), id="doubted", provenance=_believed(EDITOR, "unlikely")),
        JumpTarget(_t(6), id="rejected", provenance=_believed(EDITOR, "false")),
    )
    assert [e.is_rejected for e in registry.elements] == [False, False, True]
    assert [e.id for e in registry.active_elements()] == ["kept", "doubted", "rejected"]
    assert [e.id for e in registry.active_elements(believer=EDITOR)] == [
        "kept",
        "doubted",
    ]
    assert [e.id for e in registry.active_elements(believer=REVIEWER)] == [
        "kept",
        "doubted",
        "rejected",
    ]
    assert [
        e.id for e in registry.active_elements(believer=EDITOR, beliefs=["false"])
    ] == ["rejected"]


def test_a_rejected_jump_and_its_replacement_lie_at_one_coordinate() -> None:
    registry = _registry(
        Jump(
            _t(8),
            D(repeat_count=1),
            id="printed",
            provenance=_believed(EDITOR, "false"),
        ),
        Jump(_t(8), D(repeat_count=2), id="repair", provenance={"method": "manual"}),
    )
    assert registry.jump_at(8).id == "repair"
    assert [e.id for e in registry.active_elements(believer=EDITOR)] == ["repair"]
    assert _refused(registry.active_elements) == (
        "jumps 'printed' and 'repair' both lie at 8 ticks: name the believer who "
        "rejected one of them"
    )
    assert _refused(
        lambda: registry.add(Jump(_t(8), D(repeat_count=3), id="third"))
    ) == (
        "jump 'third' lies at 8 ticks, where jump 'repair' lies already: one jump "
        "lies at a coordinate, and several signs there are its destinations"
    )


# endregion
