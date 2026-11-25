"""The flow engine (see README, *The flow engine*).

Every expected flow is derived by hand from the rules and written as a
literal; the gold flows of the stress score are read from their file.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from typing import Any

import pytest

from timetoalign.core import Coordinate, FlowConvention, FlowFindingKind, TimeUnit
from timetoalign.testdata import ensure_data
from timetoalign.timelines.flow.engine import FlowResult, atomic_sections, play
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
    SectionBreak,
)

K = FlowFindingKind
TARGET_FLOWS = ensure_data("target_flows")


def _at(value: Any, unit: str = "ticks") -> Coordinate:
    return Coordinate(value, unit)


def D(**keys: Any) -> JumpDestination:
    return JumpDestination(**keys)


def after(jump: str, **place: Any) -> DestinationReference:
    return DestinationReference(jump=jump, **place)


def J(at: Any, *targets: JumpDestination, id: str | None = None) -> Jump:
    return Jump(_at(at), targets, id=id or f"J{at}")


def T(at: Any, id: str) -> JumpTarget:
    return JumpTarget(_at(at), id=id)


def B(at: Any, ends: FlowCondition | None = None) -> Break:
    return Break(_at(at), ends=ends, id=f"B{at}")


def _registry(
    *elements: FlowControlElement, unit: str = "ticks"
) -> FlowControlRegistry:
    registry = FlowControlRegistry(unit=unit)
    registry.add_all(elements)
    return registry


def _runs(result: FlowResult) -> list[tuple[Any, Any]]:
    return [(run.start.value, run.end.value) for run in result.runs]


def _findings(result: FlowResult) -> list[tuple[Any, ...]]:
    return [
        (f.kind, f.coordinate.value, None if f.end is None else f.end.value)
        for f in result.findings
    ]


# region One rule per case

BRACKETS_1_2 = (
    J(4, D(by=0, passes=[1]), D(to="v2", passes=[2])),
    T(8, "v2"),
    B(8),
    J(8, D(repeat_count=1)),
)
THREE_BRACKETS = (
    J(4, D(by=0, passes=[1]), D(to="v2", passes=[2]), D(to="v3", passes=[3])),
    T(8, "v2"),
    T(12, "v3"),
    B(8),
    B(12),
    J(8, D(repeat_count=1)),
    J(12, D(repeat_count=1)),
)
BRACKETS_13_2 = (
    J(4, D(by=0, passes=[1, 3]), D(to="v2", passes=[2])),
    T(8, "v2"),
    B(8),
    J(8, D(repeat_count=2)),
    J(12, D(repeat_count=1)),
)

CASES: dict[str, tuple[tuple[FlowControlElement, ...], int, list[Any], list[Any]]] = {
    "repeat without start": (
        (J(8, D(repeat_count=1)),),
        12,
        [(0, 8), (0, 12)],
        [],
    ),
    "repeat with start": (
        (T(4, "start"), J(8, D(to="start", repeat_count=1))),
        12,
        [(0, 8), (4, 12)],
        [],
    ),
    "repeat_count 3": (
        (J(4, D(repeat_count=3)),),
        8,
        [(0, 4), (0, 4), (0, 4), (0, 8)],
        [],
    ),
    "nested repeats": (
        (T(4, "inner"), J(8, D(to="inner", repeat_count=1)), J(12, D(repeat_count=1))),
        16,
        [(0, 8), (4, 12), (0, 8), (4, 16)],
        [],
    ),
    "brackets 1, 2": (BRACKETS_1_2, 12, [(0, 8), (0, 4), (8, 12)], []),
    "three brackets": (
        THREE_BRACKETS,
        16,
        [(0, 8), (0, 4), (8, 12), (0, 4), (12, 16)],
        [],
    ),
    "brackets (1, 3), (2)": (
        BRACKETS_13_2,
        16,
        [(0, 8), (0, 4), (8, 12), (0, 8), (0, 4), (8, 16)],
        [],
    ),
    "first bracket without sign": (
        BRACKETS_1_2[:3],
        12,
        [(0, 8)],
        [(K.flow_ends_at_break, 8, None), (K.never_played, 8, 12)],
    ),
    "D.C. al Fine": (
        (
            B(4, FlowCondition(after=after("J8"))),
            J(8, D(repeat_count=1, reset=False)),
        ),
        8,
        [(0, 8), (0, 4)],
        [],
    ),
    "D.S. al Coda": (
        (
            T(4, "segno"),
            J(8, D(to="coda", after=after("J16"))),
            J(16, D(to="segno", repeat_count=1, reset=False)),
            T(16, "coda"),
        ),
        20,
        [(0, 16), (4, 8), (16, 20)],
        [],
    ),
    "repeat end and D.S. on one barline": (
        (
            T(4, "segno"),
            J(8, D(repeat_count=1), D(to="segno", repeat_count=1, reset=False)),
        ),
        8,
        [(0, 8), (0, 8), (4, 8)],
        [],
    ),
    "A Fine B repeat and D.C. al Fine on one barline": (
        (
            T(0, "start"),
            J(8, D(to="start", repeat_count=1), D(repeat_count=1, reset=False)),
            B(4, FlowCondition(after=after("J8"))),
        ),
        8,
        [(0, 8), (0, 8), (0, 4)],
        [],
    ),
    "D.C. with reset at its default": (
        (J(4, D(repeat_count=1)), J(8, D(repeat_count=1))),
        8,
        [(0, 4), (0, 8), (0, 4), (0, 8)],
        [],
    ),
    "ossia": (
        (
            J(4, D(by=0), D(to="ossia", determined=False, manifestation="ossia")),
            T(8, "ossia"),
            J(8, D(to="on")),
            J(12, D(to="on")),
            T(12, "on"),
        ),
        16,
        [(0, 8), (12, 16)],
        [(K.undetermined_target, 4, None), (K.never_played, 8, 12)],
    ),
    "endless loop": (
        (J(4, D()),),
        8,
        [(0, 4), (0, 4)],
        [(K.endless_loop, 4, None), (K.never_played, 4, 8)],
    ),
    "two sections, D.C. in the second": (
        (SectionBreak(_at(8)), J(16, D(repeat_count=1, reset=False))),
        16,
        [(0, 16), (8, 16)],
        [],
    ),
    "a Fine ending the first section": (
        (
            B(4, FlowCondition(after=after("J8"))),
            J(8, D(repeat_count=1, reset=False)),
            SectionBreak(_at(8)),
        ),
        12,
        [(0, 8), (0, 4), (8, 12)],
        [],
    ),
    "barline": ((Barline(_at(4), manifestation="double_barline"),), 8, [(0, 8)], []),
    "two destinations of one kind": (
        (J(4, D(to="near"), D(to="far")), T(6, "near"), T(8, "far")),
        10,
        [(0, 4), (6, 10)],
        [(K.ambiguous_targets, 4, None), (K.never_played, 4, 6)],
    ),
}


@pytest.mark.parametrize("case", CASES)
def test_one_rule_per_case(case: str) -> None:
    elements, length, runs, findings = CASES[case]
    result = play(_registry(*elements), length)
    assert _runs(result) == runs
    assert _findings(result) == findings
    assert all(type(bound) is int for run in _runs(result) for bound in run)


def test_a_finding_names_its_place_and_says_what_was_found() -> None:
    elements, length, _, _ = CASES["first bracket without sign"]
    ends, never = play(_registry(*elements), length).findings
    assert ends.coordinate == _at(8) and ends.end is None
    assert ends.message == (
        "The flow ends at the break 'B8' at 8 ticks: no jump there has a "
        "destination that holds."
    )
    assert (never.coordinate, never.end) == (_at(8), _at(12))
    assert never.message == "The stretch [8, 12) ticks is never played."


def test_a_loop_on_a_seconds_axis_is_taken_127_times() -> None:
    registry = FlowControlRegistry(unit="seconds")
    registry.add(Jump(Coordinate(2.5, "seconds"), [D(repeat_count=127)]))
    result = play(registry, 4.0)
    assert _runs(result) == [(0.0, 2.5)] * 127 + [(0.0, 4.0)]
    assert all(type(bound) is float for run in _runs(result) for bound in run)
    assert result.runs[0].start == Coordinate(0.0, TimeUnit.seconds)
    assert result.findings == ()


def test_a_fraction_axis_is_played_exactly() -> None:
    registry = FlowControlRegistry(unit="quarters")
    third = Fraction(1, 3)
    registry.add(
        Jump(Coordinate(Fraction(953, 2), "quarters"), [D(by=-third, repeat_count=1)])
    )
    result = play(registry, Coordinate(Fraction(500), "quarters"))
    assert _runs(result) == [
        (Fraction(0), Fraction(953, 2)),
        (Fraction(953, 2) - third, Fraction(500)),
    ]


def test_a_barline_divides_no_atomic_section() -> None:
    elements, length, _, _ = CASES["barline"]
    assert [
        (s.name, s.start, s.end) for s in atomic_sections(_registry(*elements), length)
    ] == [("A", _at(0), _at(8))]


def test_atomic_sections_are_lettered_beyond_z() -> None:
    registry = _registry(*(T(at, f"t{at}") for at in range(1, 28)))
    names = [section.name for section in atomic_sections(registry, 28)]
    assert names[:3] == ["A", "B", "C"]
    assert names[24:] == ["Y", "Z", "AA", "AB"]


def test_by_zero_divides_no_atomic_section_and_a_place_does() -> None:
    elements, length, _, _ = CASES["three brackets"]
    assert [
        (s.name, s.start.value, s.end.value)
        for s in atomic_sections(_registry(*elements), length)
    ] == [("A", 0, 4), ("B", 4, 8), ("C", 8, 12), ("D", 12, 16)]
    by = _registry(J(6, D(by=-4, repeat_count=1)))
    assert [(s.start.value, s.end.value) for s in atomic_sections(by, 8)] == [
        (0, 2),
        (2, 6),
        (6, 8),
    ]


# endregion

# region Conventions

SINGLE = {
    "brackets 1, 2": ([(0, 4), (8, 12)], []),
    "three brackets": ([(0, 4), (12, 16)], []),
    "brackets (1, 3), (2)": ([(0, 8)], [(K.flow_ends_at_break, 8, None)]),
}


@pytest.mark.parametrize("case", SINGLE)
def test_single_goes_through_once(case: str) -> None:
    elements, length, _, _ = CASES[case]
    runs, findings = SINGLE[case]
    result = play(_registry(*elements), length, convention=FlowConvention.single)
    assert _runs(result) == runs
    assert _findings(result) == findings


@pytest.mark.parametrize("case", SINGLE)
def test_printed_is_the_axis(case: str) -> None:
    elements, length, _, _ = CASES[case]
    result = play(_registry(*elements), length, convention="printed")
    assert _runs(result) == [(0, length)]
    assert result.findings == ()


def test_single_takes_nothing_back_and_nothing_after() -> None:
    elements, length, _, _ = CASES["D.S. al Coda"]
    assert _runs(play(_registry(*elements), length, convention="single")) == [(0, 20)]


def test_an_unknown_convention_is_refused() -> None:
    with pytest.raises(ValueError, match="ms3"):
        play(_registry(), 8, convention="ms3")


# endregion

# region Rejected elements

EDITOR = "https://example.org/people/editor"


def test_a_rejected_element_is_not_played_for_its_believer() -> None:
    rejected = {
        "method": "imported",
        "beliefs": [{"actor": EDITOR, "belief": "false", "date": "2026-11-25"}],
    }
    registry = _registry(
        Jump(_at(8), [D(repeat_count=1)], id="printed", provenance=rejected),
        Jump(_at(8), [D(repeat_count=2)], id="repair", provenance={"method": "manual"}),
    )
    assert _runs(play(registry, 12, believer=EDITOR)) == [(0, 8), (0, 8), (0, 12)]
    with pytest.raises(ValueError) as error:
        play(registry, 12)
    assert str(error.value) == (
        "jumps 'printed' and 'repair' both lie at 8 ticks: name the believer who "
        "rejected one of them"
    )


# endregion

# region Three pieces


def test_musette() -> None:
    registry = _registry(
        T(5, "segno"),
        J(16, D(repeat_count=1), D(to="e2", after=after("J31", to="segno"))),
        B(16, FlowCondition(after=after("J58", to="segno"))),
        T(16, "rs17"),
        T(16, "e1"),
        J(31, D(to="rs17", repeat_count=1), D(to="segno", repeat_count=1, reset=False)),
        T(31, "rs32"),
        T(31, "e2"),
        J(58, D(to="rs32", repeat_count=1), D(to="segno", repeat_count=1, reset=False)),
    )
    default = play(registry, 58)
    assert _runs(default) == [
        (0, 16),
        (0, 31),
        (16, 31),
        (5, 16),
        (31, 58),
        (31, 58),
        (5, 16),
    ]
    assert default.findings == ()
    single = play(registry, 58, convention="single")
    assert _runs(single) == [(0, 58)]
    assert single.findings == ()
    assert [
        (s.name, s.start.value, s.end.value) for s in atomic_sections(registry, 58)
    ] == [("A", 0, 5), ("B", 5, 16), ("C", 16, 31), ("D", 31, 58)]


def test_rondeau() -> None:
    registry = _registry(
        J(
            9,
            D(repeat_count=1),
            D(to="e2", after=after("J18")),
            D(to="e3", after=after("J27")),
        ),
        T(9, "rs10"),
        T(9, "e1"),
        J(18, D(to="rs10", repeat_count=1), D(repeat_count=1, reset=False)),
        T(18, "rs19"),
        T(18, "e2"),
        J(27, D(to="rs19", repeat_count=1), D(repeat_count=1, reset=False)),
        T(27, "rs28"),
        T(27, "e3"),
        J(60, D(to="rs28", repeat_count=1)),
    )
    default = play(registry, 60)
    assert _runs(default) == [
        (0, 9),
        (0, 18),
        (9, 18),
        (0, 9),
        (18, 27),
        (18, 27),
        (0, 9),
        (27, 60),
        (27, 60),
    ]
    assert default.findings == ()
    assert _runs(play(registry, 60, convention="single")) == [(0, 60)]


# The stress score: where each of its fifteen measures starts, and its end.
MEASURE_STARTS = [0, 1, 5, 6, 7, 9, 10, 12, 16, 20, 24, 28, 30, 32, 34, 37]
STRESS_LENGTH = MEASURE_STARTS[-1]
REPAIR = {"method": "manual"}


def _q(value: int) -> Coordinate:
    return Coordinate(value, "quarters")


def _stress_score(*, repaired: bool) -> FlowControlRegistry:
    """The stress score as its source prints it, or with three elements added."""

    def jump(at: int, *targets: JumpDestination, **keys: Any) -> Jump:
        return Jump(_q(at), targets, id=f"J{at}", **keys)

    def target(at: int, id: str, **keys: Any) -> JumpTarget:
        return JumpTarget(_q(at), id=id, **keys)

    elements: list[FlowControlElement] = [
        jump(6, D(repeat_count=1)),
        target(6, "start_4"),
        jump(
            7,
            D(by=0, passes=[1]),
            D(to="bracket_2", passes=[2]),
            D(to="bracket_3", passes=[3]),
        ),
        target(9, "bracket_2"),
        target(10, "bracket_3"),
        Break(_q(9), manifestation="volta"),
        Break(_q(10), manifestation="volta"),
        jump(9, D(to="start_4", repeat_count=1)),
        Break(_q(12), manifestation="fine", ends=FlowCondition(after=after("J37"))),
        jump(
            16,
            D(to="start_8", repeat_count=1) if repaired else D(repeat_count=1),
        ),
        target(16, "segno"),
        target(16, "start_9"),
        jump(20, D(to="coda", after=after("J30"))),
        target(20, "start_10"),
        jump(24, D(to="start_10", repeat_count=1)),
        jump(28, D(to="start_9", repeat_count=1)),
        jump(30, D(to="segno", repeat_count=1, reset=False)),
        target(30, "coda"),
        jump(32, D(by=0, passes=[1]), D(to="ending_2", passes=[2])),
        target(34, "ending_2"),
        Break(_q(34), manifestation="volta"),
        jump(
            34,
            D(to="start_13", repeat_count=1) if repaired else D(repeat_count=1),
        ),
        jump(37, D(repeat_count=1, reset=False)),
    ]
    if repaired:
        elements += [
            target(12, "start_8", provenance=REPAIR),
            target(30, "start_13", provenance=REPAIR),
            jump(10, D(to="start_8"), provenance=REPAIR),
        ]
    registry = FlowControlRegistry(unit="quarters")
    registry.add_all(elements)
    return registry


def _gold(mode: str) -> list[tuple[int, int]]:
    """The rows of *mode* in the gold file, measure counts turned into quarters."""
    path = TARGET_FLOWS / "out_of_the_flow_experience-flow_only.flow.csv"
    with path.open(newline="") as handle:
        return [
            (
                MEASURE_STARTS[int(row["mc_start"]) - 1],
                MEASURE_STARTS[int(row["mc_end"]) - 1],
            )
            for row in csv.DictReader(handle)
            if row["flow_mode"] == mode
        ]


def test_the_stress_score_as_printed_ends_at_the_second_bracket() -> None:
    result = play(_stress_score(repaired=False), STRESS_LENGTH)
    assert _runs(result) == [(0, 6), (0, 9), (6, 7), (9, 10)]
    assert _findings(result) == [
        (K.flow_ends_at_break, 10, None),
        (K.never_played, 10, 37),
    ]


def test_the_gold_file_states_the_flows_the_readme_lists() -> None:
    def measures(rows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(MEASURE_STARTS.index(a) + 1, MEASURE_STARTS.index(b)) for a, b in rows]

    assert measures(_gold("default")) == [
        (1, 3),
        (1, 5),
        (4, 4),
        (6, 6),
        (8, 8),
        (8, 10),
        (10, 11),
        (9, 10),
        (10, 12),
        (9, 9),
        (13, 14),
        (13, 13),
        (15, 15),
        (1, 4),
        (7, 7),
    ]
    assert measures(_gold("single")) == [(1, 4), (6, 6), (8, 13), (15, 15)]


@pytest.mark.parametrize("convention", ["default", "single"])
def test_the_repaired_stress_score_plays_the_gold_flow(convention: str) -> None:
    result = play(_stress_score(repaired=True), STRESS_LENGTH, convention=convention)
    assert _runs(result) == _gold(convention)
    assert result.findings == ()
    assert all(type(bound) is Fraction for run in _runs(result) for bound in run)


def test_the_repaired_stress_score_has_the_gold_atomic_sections() -> None:
    sections = atomic_sections(_stress_score(repaired=True), STRESS_LENGTH)
    assert [(s.start.value, s.end.value) for s in sections] == _gold("atomic")
    assert [s.name for s in sections] == list("ABCDEFGHIJKLM")


# endregion
