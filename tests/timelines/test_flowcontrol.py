"""Flow-control elements and their registry (see README, *Flow control*)."""

from __future__ import annotations

import datetime
from fractions import Fraction
from typing import Any

import pytest
from timetoshare.validation.flow import rejected

from timetoalign.core import (
    Coordinate,
    Duration,
    FlowManifestation,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.core.ids import IdMinter
from timetoalign.core.provenance import Provenance
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


def test_flow_control_is_exported_where_a_user_looks_for_it() -> None:
    import timetoalign
    from timetoalign import timelines
    from timetoalign.timelines import flow, flowcontrol
    from timetoalign.timelines.flow import engine

    homes = {
        flowcontrol: (
            "FlowControlRegistry",
            "FlowControlElement",
            "Jump",
            "JumpTarget",
            "Break",
            "SectionBreak",
            "Barline",
            "JumpDestination",
            "FlowCondition",
            "DestinationReference",
            "PassesFrom",
        ),
        engine: (
            "play",
            "atomic_sections",
            "FlowResult",
            "FlowFinding",
            "AtomicSection",
        ),
        flow.flowmap: ("PlayedFlow",),
    }
    for home, names in homes.items():
        for name in names:
            for package in (flow, timelines, timetoalign):
                assert name in package.__all__, (package.__name__, name)
                assert getattr(package, name) is getattr(home, name)
    assert flow.FlowControlDraft is flow.measures.FlowControlDraft
    assert "FlowControlDraft" in flow.__all__


def test_each_class_names_its_role_and_mints_its_id_from_it() -> None:
    made = [
        Jump(_q(8), [D(repeat_count=1)], seed=7),
        JumpTarget(_q(8), seed=7),
        Break(_q(8), seed=7),
        SectionBreak(_q(8), seed=7),
        Barline(_q(8), seed=7),
    ]
    # One seed, a fresh minter per element: the first ID of that minter,
    # with the role as prefix.
    assert [(type(e).role, e.id) for e in made] == [
        (role, IdMinter(7).mint(role))
        for role in ("jump", "jump_target", "break", "section_break", "barline")
    ]
    assert [e.id for e in made] == [
        "jump_1hvif5xc2r1fv",
        "jump_target_1hvif5xc2r1fv",
        "break_1hvif5xc2r1fv",
        "section_break_1hvif5xc2r1fv",
        "barline_1hvif5xc2r1fv",
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
        Duration(F(-1, 2), "quarters"),
        Duration(F(0), "quarters"),
    ]
    assert [d.goes_on for d in jump.targets] == [False, True]
    assert jump.targets[0].after == after("x", by=Duration(F(-1), "quarters"))
    assert _refused(lambda: Jump(_q(8), [D(by=Duration(1, "seconds"))])) == (
        "a `by` of jump (unnamed) is a distance in seconds on an axis in quarters"
    )


def test_a_single_destination_may_be_given_bare() -> None:
    assert Jump(_q(8), D(repeat_count=1)).targets == (D(repeat_count=1),)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Jump(_q(8), []),
            "jump (unnamed) has no destination: `targets` lists at least one",
        ),
        (
            lambda: Jump(_q(8), [D(to="a"), D(to="a", repeat_count=1)]),
            "jump (unnamed) has 2 destinations with `to: a`: the destinations of one "
            "jump are distinct places",
        ),
        (
            lambda: Jump(_q(8), [D(), D(repeat_count=1, reset=False)]),
            "jump (unnamed) has 2 destinations with no place: the destinations of "
            "one jump are distinct places",
        ),
        (
            lambda: Jump(
                Coordinate(8.0, "seconds"), [D(by=1), D(by=Duration(1.0, "seconds"))]
            ),
            "jump (unnamed) has 2 destinations with `by: 1.0`: the destinations of "
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
            "the `ends` of break (unnamed) states no condition: leave `ends` out "
            "for a break that always ends the flow",
        ),
        (
            lambda: Break(_q(8), ends=D(repeat_count=1)),
            "the `ends` of break (unnamed) is a FlowCondition, not JumpDestination",
        ),
        (
            lambda: after("j", to="a", by=1),
            "`after` naming jump 'j' states both `to` ('a') and `by`: a place is "
            "one of them",
        ),
        (
            lambda: Break(_q(8), manifestation=FlowManifestation.segno),
            "break (unnamed) is one of ['fine', 'volta'] or a word outside the "
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
    before = registry.elements
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
    before = registry.elements
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
    assert [d.place for d in two.targets] == [("by", -1), ("to", "segno")]
    assert registry.destination_of(after("two", to="segno")) == (two, 1)
    assert registry.destination_of(after("two", by=Duration(-1, "ticks"))) == (two, 0)
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
    assert registry.elements == ()
    stored = registry.add_all([fine, to_coda, dc, coda])
    assert [e.id for e in stored] == ["fine", "to_coda", "dc", "coda"]
    assert [e.id for e in registry.elements] == ["fine", "to_coda", "coda", "dc"]

    one_by_one = FlowControlRegistry(unit="ticks")
    for element in (coda, dc, to_coda, fine):
        one_by_one.add(element)
    assert one_by_one.elements == registry.elements
    assert FlowControlRegistry([fine, to_coda, dc, coda], unit="ticks") == registry


def test_an_id_held_is_refused_given_or_minted() -> None:
    registry = FlowControlRegistry(unit="ticks")
    first = registry.add(JumpTarget(_t(4)))
    second = registry.add(JumpTarget(_t(2)))
    assert first.id != second.id
    assert [e.id for e in registry.elements] == [second.id, first.id]
    assert _refused(lambda: registry.add(JumpTarget(_t(6), id=first.id))) == (
        f"jump_target '{first.id}' cannot join: its id '{first.id}' is already "
        f"held by jump_target '{first.id}'"
    )
    assert [e.id for e in registry.elements] == [second.id, first.id]


def test_an_absorbed_registry_keeps_its_ids_and_a_shared_one_is_refused() -> None:
    source = FlowControlRegistry(unit="quarters")
    target = source.add(JumpTarget(_q(4), manifestation="segno"))
    dal_segno = source.add(Jump(_q(8), D(to=target.id, repeat_count=1, reset=False)))
    fine = source.add(
        Break(_q(6), ends=FlowCondition(after=after(dal_segno.id, to=target.id)))
    )

    timeline = ContinuousLogicalTimeline(length=16, uid="score")
    held = timeline.flow_control.add(JumpTarget(_q(2)))
    back = timeline.flow_control.add(Jump(_q(12), D(to=held.id, repeat_count=1)))
    timeline.flow_control._absorb(source)

    assert [(e.id, e.coordinate.value) for e in timeline.flow_control.elements] == [
        (held.id, 2),
        (target.id, 4),
        (fine.id, 6),
        (dal_segno.id, 8),
        (back.id, 12),
    ]
    assert len({e.id for e in timeline.flow_control.elements}) == 5
    absorbed = timeline.flow_control.jump_at(8)
    assert absorbed.targets == (D(to=target.id, repeat_count=1, reset=False),)
    assert timeline.flow_control.breaks[0].ends.after == after(
        dal_segno.id, to=target.id
    )
    assert timeline.flow_control.jump_at(12).targets[0].to == held.id
    # The source registry is what it was.
    assert [e.id for e in source.elements] == [target.id, fine.id, dal_segno.id]

    # A registry holding an ID the timeline holds is refused whole.
    clash = FlowControlRegistry(unit="quarters")
    clash.add(JumpTarget(_q(14), id="coda"))
    clash.add(JumpTarget(_q(15), id=held.id))
    before = timeline.flow_control.elements
    assert _refused(lambda: timeline.flow_control._absorb(clash)) == (
        f"jump_target '{held.id}' of timeline 'score' cannot join: its id "
        f"'{held.id}' is already held by jump_target '{held.id}' of timeline "
        "'score'"
    )
    assert timeline.flow_control.elements == before


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


def test_the_destinations_of_a_jump_are_held_by_place() -> None:
    given = [
        D(to="b", repeat_count=1),
        D(by=F(7, 2), repeat_count=1),
        D(to="B", repeat_count=1),
        D(repeat_count=1),
        D(by=F(-2, 3), repeat_count=1),
        D(by=0),
        D(to="a", repeat_count=1),
    ]
    places = [
        None,
        ("by", F(-2, 3)),
        ("by", F(0)),
        ("by", F(7, 2)),
        ("to", "B"),
        ("to", "a"),
        ("to", "b"),
    ]
    reference = Jump(_q(8), given, id="j")
    assert [d.place for d in reference.targets] == places
    assert [
        "to" in d and d["to"] or d.get("by") for d in reference.to_dict()["targets"]
    ] == [None, [-2, 3], [0, 1], [7, 2], "B", "a", "b"]
    for shift in range(1, len(given)):
        rotated = Jump(_q(8), given[shift:] + given[:shift], id="j")
        assert rotated == reference
        assert rotated.to_dict() == reference.to_dict()
    assert Jump(_q(8), list(reversed(given)), id="j") == reference
    seconds = Jump(
        Coordinate(20.75, "seconds"), [D(by=-9.25), D(by=-10.5), D(by=-20.75)], id="j"
    )
    assert [d.by.value for d in seconds.targets] == [-20.75, -10.5, -9.25]


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
    assert [(d.place, registry.place_of(d, jump)) for d in jump.targets] == [
        (None, _t(0)),
        (("by", -3), _t(17)),
        (("by", 0), _t(20)),
        (("by", 5), _t(25)),
        (("to", "segno"), _t(4)),
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
    assert [(d.to, registry.place_of(d, jump)) for d in jump.targets] == [
        ("chorus", _q(F(16))),
        ("score", _q(F(0))),
        ("trio", _q(F(40))),
    ]
    verse = Jump(_q(60), D(to="verse"))
    assert _refused(lambda: registry.add(verse)) == (
        f"jump '{verse.id}' goes on at 'verse', which names no jump target of this "
        "registry and no place of its timeline. Available jump targets: []. "
        "Available places: ['score', 'chorus', 'trio']"
    )
    with pytest.raises(KeyError):
        FlowControlRegistry(unit="quarters").place_of(jump.targets[0], jump)


def test_a_destination_follows_its_timeline_given_a_new_id() -> None:
    verse = ContinuousLogicalTimeline(length=8, uid="draft")
    verse.flow_control.add(Jump(_q(8), D(to="draft", repeat_count=1)))
    song = ContinuousLogicalTimeline(length=0, uid="song")
    song.append_child(verse, uid="verse")
    (jump,) = verse.flow_control.jumps
    assert jump.targets[0].to == "verse"
    assert verse.flow_control.place_of(jump.targets[0], jump) == _q(F(0))


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
    assert [e.id for e in duplicate.elements] == ["segno", "segno2"]
    assert timeline.flow_control.has_flow_control()
    assert not FlowControlRegistry(unit="quarters").has_flow_control()
    # The copy belongs to no timeline: the length 16 does not bound it.
    duplicate.add(Break(_q(20), id="late"))
    assert _refused(lambda: timeline.flow_control.add(Break(_q(20), id="late"))) == (
        "break 'late' lies at 20 quarters, beyond the length 16 quarters: a "
        "flow-control element lies from 0 to the length of its timeline"
    )


def test_the_collection_is_read_only() -> None:
    registry = _registry(
        Jump(_t(8), D(repeat_count=1), id="j"),
        JumpTarget(_t(4), id="t"),
        Break(_t(8), id="b"),
        SectionBreak(_t(12), id="s"),
        Barline(_t(12), id="l"),
    )
    assert type(registry.elements) is tuple
    assert [e.id for e in registry.elements] == ["t", "b", "j", "l", "s"]
    views = {
        "jumps": ["j"],
        "jump_targets": ["t"],
        "breaks": ["b"],
        "section_breaks": ["s"],
        "barlines": ["l"],
    }
    for view, ids in views.items():
        held = getattr(registry, view)
        assert type(held) is tuple
        assert [e.id for e in held] == ids
        with pytest.raises(AttributeError):
            setattr(registry, view, ())
    with pytest.raises(AttributeError):
        registry.elements = ()  # type: ignore[misc]
    assert not hasattr(registry, "clear")
    assert not hasattr(registry.elements, "append")
    assert [e.id for e in registry.elements] == ["t", "b", "j", "l", "s"]


def test_jump_at_takes_a_bare_value_a_coordinate_and_an_id_coordinate() -> None:
    registry = FlowControlRegistry(unit="quarters")
    jump = registry.add(Jump(_q(F(9, 2)), D(repeat_count=1), id="j"))
    for position in (
        4.5,
        F(9, 2),
        _q(4.5),
        _q(F(9, 2)),
        IdCoordinate(F(9, 2), "quarters", "score"),
    ):
        assert registry.jump_at(position) is jump
    assert registry.jump_at(4) is None
    assert registry.jump_at(_q(4)) is None
    assert _refused(lambda: registry.jump_at(Coordinate(4.5, "seconds"))) == (
        "Flow-control coordinate unit seconds does not match registry unit quarters"
    )
    with pytest.raises(TypeError) as bare:
        Jump(4, D(repeat_count=1))  # type: ignore[arg-type]
    assert str(bare.value) == "a jump lies at a Coordinate, not at int 4"


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
    assert timeline.flow_control.elements == ()
    assert timeline.measure_map is None
    assert timeline.metric_hierarchy is None


def test_reach_counts_every_element_and_every_distance() -> None:
    timeline = ContinuousLogicalTimeline(length=None)
    timeline.flow_control.add(JumpTarget(_q(6)))
    assert timeline.reach == _q(F(6))
    timeline.flow_control.add(Jump(_q(8), D(by=4)))
    assert timeline.reach == _q(F(12))


# endregion

# region The axis

OFF_THE_AXIS = [
    (
        lambda: Jump(_q(4), D(by=-8, repeat_count=1), id="back"),
        "jump 'back' at 4 quarters goes on by -8 quarters, at -4 quarters, before "
        "0: a place lies from 0 to the length of the timeline",
    ),
    (
        lambda: Jump(_q(4), D(by=20, repeat_count=1), id="far"),
        "jump 'far' at 4 quarters goes on by 20 quarters, at 24 quarters, beyond "
        "the length 12 quarters: a place lies from 0 to the length of the timeline",
    ),
    (
        lambda: Break(_q(20), id="late"),
        "break 'late' lies at 20 quarters, beyond the length 12 quarters: a "
        "flow-control element lies from 0 to the length of its timeline",
    ),
    (
        lambda: JumpTarget(_q(-1), id="early"),
        "jump_target 'early' lies at -1 quarters, before 0: a flow-control "
        "element lies from 0 to the length of its timeline",
    ),
    (
        lambda: Jump(_q(4), D(to="beyond"), id="out"),
        "jump 'out' at 4 quarters goes on at 'beyond', which lies at 14 quarters, "
        "beyond the length 12 quarters: a place lies from 0 to the length of the "
        "timeline",
    ),
]


@pytest.mark.parametrize(("build", "message"), OFF_THE_AXIS)
def test_an_element_or_a_place_outside_the_length_is_refused(
    build: Any, message: str
) -> None:
    timeline = ContinuousLogicalTimeline(length=12, uid="score")
    timeline.create_region(14, 16, id="beyond")
    before = timeline.flow_control.elements
    assert _refused(lambda: timeline.flow_control.add(build())) == message
    assert _refused(lambda: timeline.flow_control.add_all([build()])) == message
    assert timeline.flow_control.elements == before == ()
    assert timeline.reach == _q(F(16))


def test_the_bounds_of_the_axis_are_places_of_it() -> None:
    timeline = ContinuousLogicalTimeline(length=12, uid="score")
    registry = timeline.flow_control
    registry.add_all(
        [
            JumpTarget(_q(0), id="start"),
            Jump(_q(4), D(by=8), id="to_the_end"),
            Jump(_q(12), D(by=-12, repeat_count=1), id="from_the_end"),
            Break(_q(12), id="end"),
        ]
    )
    assert [registry.place_of(j.targets[0], j) for j in registry.jumps] == [
        _q(F(12)),
        _q(F(0)),
    ]
    # What a rejected element states is not read, so it is not bounded either.
    late = Break(
        _q(20),
        id="late",
        provenance={
            "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}]
        },
    )
    assert registry.add(late).is_rejected
    assert _refused(
        lambda: registry.with_belief("late", EDITOR, "true", date="2026-11-26")
    ) == (
        "break 'late' lies at 20 quarters, beyond the length 12 quarters: a "
        "flow-control element lies from 0 to the length of its timeline"
    )
    assert registry.elements[-1].is_rejected


def test_a_length_that_becomes_known_must_hold_the_flow_control() -> None:
    timeline = ContinuousLogicalTimeline(length=None, uid="score")
    registry = timeline.flow_control
    registry.add(Break(_q(20), id="late"))
    registry.add(Jump(_q(4), D(by=20, repeat_count=1), id="far"))
    assert timeline.reach == _q(F(24))
    assert _refused(
        lambda: registry.add(Jump(_q(6), D(by=-8, repeat_count=1), id="back"))
    ) == (
        "jump 'back' at 6 quarters goes on by -8 quarters, at -2 quarters, before "
        "0: a place lies from 0 to the length of the timeline"
    )

    def set_length(value: int) -> None:
        timeline.length = value

    assert _refused(lambda: set_length(12)) == (
        "jump 'far' at 4 quarters goes on by 20 quarters, at 24 quarters, beyond "
        "the length 12 quarters: a place lies from 0 to the length of the timeline"
    )
    assert timeline.length is None
    set_length(24)
    assert timeline.length == _q(F(24))
    assert _refused(lambda: set_length(22)) == (
        "jump 'far' at 4 quarters goes on by 20 quarters, at 24 quarters, beyond "
        "the length 22 quarters: a place lies from 0 to the length of the timeline"
    )
    assert timeline.length == _q(F(24))
    set_length(30)
    assert timeline.length == _q(F(30))
    assert [e.id for e in registry.elements] == ["far", "late"]


def test_a_registry_of_no_timeline_is_bounded_at_0_only() -> None:
    registry = FlowControlRegistry(unit="ticks")
    assert registry.add(Break(_t(20), id="late")).coordinate == _t(20)
    assert registry.add(Jump(_t(4), D(by=20), id="far")).id == "far"
    assert _refused(lambda: registry.add(Jump(_t(6), D(by=-8), id="back"))) == (
        "jump 'back' at 6 ticks goes on by -8 ticks, at -2 ticks, before 0: a place "
        "lies from 0 to the length of the timeline"
    )
    assert _refused(lambda: registry.add(Barline(_t(-4), id="early"))) == (
        "barline 'early' lies at -4 ticks, before 0: a flow-control element lies "
        "from 0 to the length of its timeline"
    )
    assert [e.id for e in registry.elements] == ["far", "late"]


# endregion

# region Rejection


def _believed(*beliefs: tuple[str, str], method: str = "imported") -> dict[str, Any]:
    """One provenance entry holding a belief for each (actor, belief) given."""
    entry: dict[str, Any] = {"method": method}
    if beliefs:
        entry["beliefs"] = [
            {"actor": actor, "belief": belief, "date": "2026-11-25"}
            for actor, belief in beliefs
        ]
    return entry


def test_an_element_any_actor_believes_false_is_rejected() -> None:
    registry = _registry(
        JumpTarget(_t(2), id="kept"),
        JumpTarget(_t(4), id="doubted", provenance=_believed((EDITOR, "unlikely"))),
        JumpTarget(_t(6), id="by_editor", provenance=_believed((EDITOR, "false"))),
        JumpTarget(
            _t(8),
            id="by_reviewer",
            provenance=[
                _believed((EDITOR, "true")),
                _believed((REVIEWER, "false"), method="manual"),
            ],
        ),
    )
    assert [e.is_rejected for e in registry.elements] == [False, False, True, True]
    assert [rejected(written) for written in registry.to_dict()] == [
        False,
        False,
        True,
        True,
    ]

    def read(**filter: Any) -> list[str]:
        return [element.id for element in registry.active_elements(**filter)]

    assert read() == ["kept", "doubted"]
    assert read(believer=EDITOR) == ["kept", "doubted", "by_reviewer"]
    assert read(believer=REVIEWER) == ["kept", "doubted", "by_editor"]
    assert read(believer=EDITOR, beliefs=["false"]) == ["by_editor"]
    assert read(believer=EDITOR, beliefs=["true"]) == ["by_reviewer"]
    assert read(believer=EDITOR, beliefs=["undecided"]) == ["kept"]
    for believer in (EDITOR, None):
        assert _refused(lambda: read(believer=believer, beliefs=["maybe"])) == (
            "'maybe' is no belief state: write 'true', 'undecided', 'unlikely' or "
            "'false'"
        )


def test_two_beliefs_of_one_actor_read_as_the_more_critical() -> None:
    registry = _registry(
        Barline(
            _t(4),
            id="both",
            provenance=[_believed((EDITOR, "true")), _believed((EDITOR, "false"))],
        )
    )
    assert registry.elements[0].is_rejected
    assert registry.active_elements(believer=EDITOR) == []
    assert [
        e.id for e in registry.active_elements(believer=EDITOR, beliefs=["false"])
    ] == ["both"]
    assert registry.active_elements(believer=EDITOR, beliefs=["true"]) == []


@pytest.mark.parametrize("rejected_first", [True, False])
def test_a_rejected_jump_and_its_replacement_lie_at_one_coordinate(
    rejected_first: bool,
) -> None:
    printed = Jump(
        _t(8),
        D(repeat_count=1),
        id="printed",
        provenance=_believed((EDITOR, "false")),
    )
    repair = Jump(
        _t(8), D(repeat_count=2), id="repair", provenance={"method": "manual"}
    )
    registry = FlowControlRegistry(unit="ticks")
    for jump in (printed, repair) if rejected_first else (repair, printed):
        registry.add(jump)
    assert [e.id for e in registry.elements] == ["printed", "repair"]
    assert registry.jump_at(8).id == "repair"
    assert [e.id for e in registry.active_elements()] == ["repair"]
    assert [e.id for e in registry.active_elements(believer=EDITOR)] == ["repair"]
    assert [e.id for e in registry.active_elements(believer=REVIEWER)] == [
        "printed",
        "repair",
    ]
    assert _refused(
        lambda: registry.add(Jump(_t(8), D(repeat_count=3), id="third"))
    ) == (
        "jump 'third' lies at 8 ticks, where jump 'repair' lies already: one jump "
        "lies at a coordinate, and several signs there are its destinations"
    )
    assert [e.id for e in registry.elements] == ["printed", "repair"]


def test_a_rejected_element_is_out_of_every_rule() -> None:
    registry = _registry(
        JumpTarget(_t(4), id="gone", provenance=_believed((EDITOR, "false"))),
        Jump(
            _t(8),
            D(to="nowhere"),
            id="unread",
            provenance=_believed((EDITOR, "false")),
        ),
    )
    before = registry.elements
    assert _refused(lambda: registry.add(Jump(_t(12), D(to="gone"), id="j"))) == (
        "jump 'j' goes on at 'gone', which names no jump target of this registry "
        "and no place of its timeline: jump_target 'gone' is rejected. Available "
        "jump targets: []. Available places: []"
    )
    assert _refused(
        lambda: registry.add(
            Break(_t(6), id="fine", ends=FlowCondition(after=after("unread")))
        )
    ) == (
        "the `after` of break 'fine' names 'unread', which is no jump of this "
        "registry: jump 'unread' is rejected. Available jumps: []"
    )
    assert registry.elements == before


def test_a_single_entry_is_held_as_the_list_of_it() -> None:
    entry = Provenance(method="imported")
    for given in (entry, {"method": "imported"}, [entry], ({"method": "imported"},)):
        assert Barline(_t(4), provenance=given).provenance == (entry,)
    assert Barline(_t(4)).provenance is None
    assert _refused(lambda: Barline(_t(4), id="b", provenance=[])) == (
        "the provenance of barline 'b' lists no entry: leave `provenance` out for "
        "an element that states none"
    )
    assert _refused(lambda: Barline(_t(4), id="b", provenance=["imported"])) == (
        "the provenance of barline 'b' holds 'imported', which is no provenance entry"
    )
    assert _refused(lambda: Barline(_t(4), id="b", provenance="imported")) == (
        "the provenance of barline 'b' is a list of provenance entries, not "
        "'imported'"
    )


# endregion

# region Recording a belief


def test_a_belief_is_recorded_on_the_element_s_copy() -> None:
    registry = _registry(
        JumpTarget(_t(4), id="bare", manifestation="segno", name="S"),
        Barline(_t(6), id="imported", provenance={"method": "imported"}),
    )
    bare, imported = registry.elements

    believed = registry.with_belief("bare", EDITOR, "false", date="2026-11-25")

    assert believed is registry.elements[0]
    assert bare.provenance is None and not bare.is_rejected
    assert believed.is_rejected
    assert believed.to_dict() == {
        "id": "bare",
        "role": "jump_target",
        "coordinate": 4,
        "manifestation": "segno",
        "name": "S",
        "provenance": [
            {"beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}]}
        ],
    }

    registry.with_belief(
        "imported", EDITOR, "unlikely", date=datetime.date(2026, 11, 24), note="?"
    )
    registry.with_belief("imported", REVIEWER, "true", date="2026-11-25")
    registry.with_belief("imported", EDITOR, "false", date="2026-11-25")
    assert imported.provenance == (Provenance(method="imported"),)
    assert registry.to_dict()[1]["provenance"] == [
        {
            "method": "imported",
            "beliefs": [
                {"actor": EDITOR, "belief": "false", "date": "2026-11-25"},
                {"actor": REVIEWER, "belief": "true", "date": "2026-11-25"},
            ],
        }
    ]
    assert [e.id for e in registry.elements] == ["bare", "imported"]


def test_a_belief_joins_the_entry_holding_the_actor_s_earlier_one() -> None:
    registry = _registry(
        Barline(
            _t(4),
            id="b",
            provenance=[
                _believed(),
                _believed((REVIEWER, "true"), (EDITOR, "true"), method="manual"),
            ],
        )
    )
    registry.with_belief("b", EDITOR, "false", date="2026-11-26")
    registry.with_belief(
        "b", "https://example.org/people/third", "true", date="2026-11-26"
    )
    assert registry.to_dict()[0]["provenance"] == [
        {
            "method": "imported",
            "beliefs": [
                {
                    "actor": "https://example.org/people/third",
                    "belief": "true",
                    "date": "2026-11-26",
                }
            ],
        },
        {
            "method": "manual",
            "beliefs": [
                {"actor": REVIEWER, "belief": "true", "date": "2026-11-25"},
                {"actor": EDITOR, "belief": "false", "date": "2026-11-26"},
            ],
        },
    ]


def test_recording_a_belief_refuses_what_it_cannot_record() -> None:
    registry = _registry(Barline(_t(4), id="b"))
    before = registry.elements
    with pytest.raises(KeyError) as unknown:
        registry.with_belief("x", EDITOR, "false")
    assert unknown.value.args[0] == (
        "'x' names no flow-control element. Available elements: ['b']"
    )
    with pytest.raises(
        ValueError, match="Input should be 'true', 'false' or 'unlikely'"
    ):
        registry.with_belief("b", EDITOR, "maybe")
    with pytest.raises(ValueError, match="'2026-13-01' is no date of the calendar"):
        registry.with_belief("b", EDITOR, "false", date="2026-13-01")
    assert registry.elements == before
    assert registry.elements[0].provenance is None


def _repeat_with_a_fine() -> FlowControlRegistry:
    return _registry(
        JumpTarget(_t(4), id="start"),
        Break(_t(6), id="fine", ends=FlowCondition(after=after("printed"))),
        Jump(_t(8), D(to="start", repeat_count=1), id="printed"),
    )


NOT_ALONE = (
    " it. Reject the elements naming it in the same step (with_beliefs), then add "
    "their replacements"
)


def test_an_element_another_one_names_is_not_rejected_alone() -> None:
    registry = _repeat_with_a_fine()
    before = registry.elements

    def reject(*ids: str) -> list[FlowControlElement]:
        return registry.with_beliefs(ids, EDITOR, "false", date="2026-11-25")

    assert _refused(
        lambda: registry.with_belief("start", EDITOR, "false", date="2026-11-25")
    ) == ("jump_target 'start' is not rejected alone: jump 'printed' names" + NOT_ALONE)
    assert _refused(lambda: reject("printed")) == (
        "jump 'printed' is not rejected alone: break 'fine' names" + NOT_ALONE
    )
    assert _refused(lambda: reject("start", "printed")) == (
        "jump 'printed' is not rejected alone: break 'fine' names" + NOT_ALONE
    )
    assert registry.elements == before
    assert [e.id for e in registry.active_elements()] == ["start", "fine", "printed"]

    rejected_now = reject("printed", "start", "fine")

    assert [e.id for e in rejected_now] == ["printed", "start", "fine"]
    assert all(e.is_rejected for e in rejected_now)
    assert [e.id for e in registry.elements] == ["start", "fine", "printed"]
    assert all(held is e for held, e in zip(registry.jumps, rejected_now[:1]))
    assert registry.active_elements() == []
    assert registry.to_dict()[0]["provenance"] == [
        {"beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}]}
    ]


def test_every_element_naming_the_rejected_one_is_named() -> None:
    registry = _registry(
        JumpTarget(_t(4), id="start"),
        Jump(_t(8), D(to="start", repeat_count=1), id="a"),
        Jump(_t(12), D(to="start", repeat_count=1), id="b"),
    )
    assert _refused(lambda: registry.with_belief("start", EDITOR, "false")) == (
        "jump_target 'start' is not rejected alone: jump 'a', jump 'b' name" + NOT_ALONE
    )
    # What only a rejected element names is rejected alone.
    registry.with_beliefs(["a", "b"], EDITOR, "false", date="2026-11-25")
    assert registry.with_belief("start", EDITOR, "false").is_rejected


def test_a_belief_recorded_in_several_elements_that_rejects_nothing() -> None:
    registry = _repeat_with_a_fine()
    doubted = registry.with_beliefs(
        ["printed", "start"], REVIEWER, "unlikely", date="2026-11-25"
    )
    assert [(e.id, e.is_rejected) for e in doubted] == [
        ("printed", False),
        ("start", False),
    ]
    assert [e.provenance is None for e in registry.elements] == [False, True, False]
    assert [e.id for e in registry.active_elements()] == ["start", "fine", "printed"]


def test_a_repair_rejects_in_one_step_and_a_rejection_taken_back_is_checked() -> None:
    registry = _repeat_with_a_fine()
    registry.with_beliefs(["printed", "fine"], EDITOR, "false", date="2026-11-25")
    registry.add_all(
        [
            Jump(_t(8), D(repeat_count=1), id="repair"),
            Break(_t(6), id="fine2", ends=FlowCondition(after=after("repair"))),
        ]
    )
    before = registry.elements
    assert [e.id for e in registry.active_elements()] == ["start", "fine2", "repair"]
    assert _refused(
        lambda: registry.with_belief("printed", EDITOR, "true", date="2026-11-26")
    ) == (
        "jump 'printed' lies at 8 ticks, where jump 'repair' lies already: one jump "
        "lies at a coordinate, and several signs there are its destinations"
    )
    assert _refused(
        lambda: registry.with_belief("fine", EDITOR, "true", date="2026-11-26")
    ) == (
        "the `after` of break 'fine' names 'printed', which is no jump of this "
        "registry: jump 'printed' is rejected. Available jumps: ['repair']"
    )
    assert registry.elements == before


def test_several_beliefs_are_recorded_together_or_not_at_all() -> None:
    registry = _repeat_with_a_fine()
    before = registry.elements
    with pytest.raises(KeyError) as unknown:
        registry.with_beliefs(["fine", "x"], EDITOR, "false")
    assert unknown.value.args[0] == (
        "'x' names no flow-control element. Available elements: "
        "['start', 'fine', 'printed']"
    )
    with pytest.raises(TypeError) as one:
        registry.with_beliefs("fine", EDITOR, "false")  # type: ignore[arg-type]
    assert str(one.value) == (
        "with_beliefs takes the IDs of several elements, not the string 'fine': "
        "with_belief takes one"
    )
    assert registry.elements == before
    assert registry.with_beliefs([], EDITOR, "false") == []


# endregion
