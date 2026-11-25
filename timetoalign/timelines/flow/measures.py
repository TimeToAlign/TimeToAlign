"""Measures and flow control: one translation both ways, and flows in measure counts.

A measure map states three things about how a score is played through: a
row's ``start_repeat``, its ``end_repeat`` and its ``volta``, the pass
numbers of the bracket of alternative endings it lies under.
:func:`flow_control_from_measures` turns them into the elements of
:mod:`timetoalign.timelines.flowcontrol`, on the axis of the measures'
``qstamp`` values (quarters, exact); :func:`measure_keys_from_flow_control`
reads the same three keys back from elements.

- ``start_repeat``: a ``JumpTarget`` (``repeat_start``) at the row's start.
- ``end_repeat``: a destination (``repeat_end``, ``repeat_count: 1``) of the
  Jump at the row's end.
- ``volta``, for each run of adjacent brackets: one Jump at the start of the
  first bracket with a destination per bracket, and a ``JumpTarget``
  (``volta``) and a ``Break`` (``volta``) at the start of every further
  bracket.

Signs falling on one coordinate are destinations of one Jump. Every other
sign of a score (a D.C., a segno, a Fine, a section break) is no key of a
measure map row and comes from the score's own source.

A flow is played on the axis (:func:`timetoalign.timelines.flow.engine.play`);
:func:`measure_ranges` states its runs, or the atomic sections, as right-open
ranges of measure counts, the form of a ``.flow.csv`` file, which
:func:`read_flow_csv` reads.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from timetoalign.core import Coordinate, Duration, TimeUnit
from timetoalign.core.enums import FlowManifestation, MeasureFindingKind

from ..flowcontrol import (
    Barline,
    Break,
    FlowControlElement,
    Jump,
    JumpDestination,
    JumpTarget,
    SectionBreak,
)

if TYPE_CHECKING:
    from ..structure import MeasureMap

_Placed = TypeVar("_Placed", JumpTarget, Break, SectionBreak, Barline)

# region Rows


@dataclass(frozen=True)
class _Row:
    """What the translation reads of one measure."""

    count: int
    start: Fraction
    end: Fraction
    start_repeat: bool
    end_repeat: bool
    passes: tuple[int, ...] | None


def _passes(volta: Any, count: int) -> tuple[int, ...] | None:
    """Return a row's ``volta`` as pass numbers: an integer or a list of them.

    Raises:
        ValueError: If the value is neither.
    """
    if volta is None:
        return None
    numbers = [volta] if isinstance(volta, int) else volta
    if (
        isinstance(numbers, (str, bytes))
        or not isinstance(numbers, Sequence)
        or not numbers
        or any(isinstance(n, bool) or not isinstance(n, int) for n in numbers)
    ):
        raise ValueError(
            f"the volta {volta!r} of measure count {count} is neither a pass "
            "number nor a list of pass numbers"
        )
    return tuple(numbers)


def _rows(measures: MeasureMap | Iterable[Any]) -> list[_Row]:
    """Read the measures of a map, or full rows, in printed order.

    Raises:
        ValueError: If a measure states no ``qstamp`` or no ``actual_length``.
    """
    rows: list[_Row] = []
    for position, measure in enumerate(measures, start=1):
        if isinstance(measure, Mapping):
            read = measure.get
        else:

            def read(key: str, measure: Any = measure) -> Any:
                return getattr(measure, key, None)

        count = read("count") or position
        qstamp, length = read("qstamp"), read("actual_length")
        if qstamp is None or length is None:
            raise ValueError(
                f"measure count {count} states no "
                f"{'qstamp' if qstamp is None else 'actual_length'}: flow control "
                "lies at positions, which an abstract measure map has none of"
            )
        start = Fraction(qstamp)
        rows.append(
            _Row(
                count=int(count),
                start=start,
                end=start + Fraction(length),
                start_repeat=bool(read("start_repeat")),
                end_repeat=bool(read("end_repeat")),
                passes=_passes(read("volta"), count),
            )
        )
    return rows


@dataclass
class _Bracket:
    """One bracket of alternative endings: its passes and its rows."""

    passes: tuple[int, ...]
    first: int
    last: int


def _bracket_runs(rows: Sequence[_Row]) -> list[list[_Bracket]]:
    """Group the rows that state ``volta`` into runs of adjacent brackets.

    Adjacent rows stating the same passes are one bracket. Adjacent brackets
    are one run for as long as each names passes no earlier bracket of the
    run names: a bracket naming a pass again begins the next run.
    """
    runs: list[list[_Bracket]] = []
    adjacent = False
    for index, row in enumerate(rows):
        if row.passes is None:
            adjacent = False
            continue
        if adjacent and runs[-1][-1].passes == row.passes:
            runs[-1][-1].last = index
            continue
        bracket = _Bracket(row.passes, index, index)
        named = {number for held in runs[-1] for number in held.passes} if runs else ()
        if adjacent and not set(row.passes) & set(named):
            runs[-1].append(bracket)
        else:
            runs.append([bracket])
        adjacent = True
    return runs


# endregion

# region Writing elements


def _quarters(value: Fraction) -> Coordinate:
    return Coordinate(value, TimeUnit.quarters)


class _FlowControlDraft:
    """Flow control being written from a score, before it joins a registry.

    The signs at one coordinate are collected as the destinations of the
    one Jump there, and every element gets an ID of its own, minted in the
    order written, so that the elements can name one another before a
    registry admits them together.
    """

    def __init__(self) -> None:
        self._placed: list[FlowControlElement] = []
        self._destinations: dict[Fraction, list[JumpDestination]] = {}
        self._jump_ids: dict[Fraction, str] = {}
        self._numbers: dict[str, int] = {}

    def _mint(self, prefix: str) -> str:
        self._numbers[prefix] = self._numbers.get(prefix, 0) + 1
        return f"{prefix}_{self._numbers[prefix]}"

    def add(self, element_class: type[_Placed], at: Fraction, **keys: Any) -> _Placed:
        """Write a jump target, a break, a section break or a barline at *at*."""
        element = element_class(
            _quarters(at), id=self._mint(element_class.role), _minted=True, **keys
        )
        self._placed.append(element)
        return element

    def jump_at(self, at: Fraction) -> str:
        """Return the ID of the Jump at *at*, minted when first asked for."""
        if at not in self._jump_ids:
            self._jump_ids[at] = self._mint(Jump.role)
        return self._jump_ids[at]

    def destinations_at(self, at: Fraction) -> tuple[JumpDestination, ...]:
        """Return the destinations written so far for the Jump at *at*."""
        return tuple(self._destinations.get(at, ()))

    def add_destination(self, at: Fraction, destination: JumpDestination) -> str:
        """Write *destination* on the Jump at *at*; return that Jump's ID."""
        self._destinations.setdefault(at, []).append(destination)
        return self.jump_at(at)

    def add_measures(self, measures: MeasureMap | Iterable[Any]) -> None:
        """Write what the rows of *measures* state: repeats and brackets."""
        rows = _rows(measures)
        place: dict[int, tuple[int, int, _Bracket]] = {}
        run_starts: dict[int, Fraction] = {}
        for run, brackets in enumerate(_bracket_runs(rows)):
            run_starts[run] = rows[brackets[0].first].start
            for number, bracket in enumerate(brackets):
                place[bracket.first] = place[bracket.last] = (run, number, bracket)

        open_starts: list[str] = []
        repeated_from: dict[int, str | None] = {}
        for index, row in enumerate(rows):
            if row.start_repeat:
                start = self.add(
                    JumpTarget, row.start, manifestation=FlowManifestation.repeat_start
                )
                open_starts.append(start.id)
            run, number, bracket = place.get(index, (-1, -1, None))
            if bracket is not None and bracket.first == index:
                if number == 0:
                    self.add_destination(
                        row.start,
                        JumpDestination(
                            by=0,
                            passes=bracket.passes,
                            manifestation=FlowManifestation.volta,
                        ),
                    )
                else:
                    landing = self.add(
                        JumpTarget, row.start, manifestation=FlowManifestation.volta
                    )
                    self.add(Break, row.start, manifestation=FlowManifestation.volta)
                    self.add_destination(
                        run_starts[run],
                        JumpDestination(
                            to=landing.id,
                            passes=bracket.passes,
                            manifestation=FlowManifestation.volta,
                        ),
                    )
            if not row.end_repeat:
                continue
            closes_bracket = bracket is not None and bracket.last == index
            if closes_bracket and run in repeated_from:
                to = repeated_from[run]
            else:
                to = open_starts.pop() if open_starts else None
                if closes_bracket:
                    repeated_from[run] = to
            self.add_destination(
                row.end,
                JumpDestination(
                    to=to,
                    repeat_count=1,
                    reset=not (closes_bracket and number > 0),
                    manifestation=FlowManifestation.repeat_end,
                ),
            )

    def elements(self) -> list[FlowControlElement]:
        """Return everything written, by coordinate and then by ID."""
        jumps = [
            Jump(_quarters(at), destinations, id=self._jump_ids[at], _minted=True)
            for at, destinations in self._destinations.items()
        ]
        return sorted(
            [*self._placed, *jumps],
            key=lambda element: (element.coordinate.value, element.id),
        )


def flow_control_from_measures(
    measures: MeasureMap | Iterable[Any],
) -> list[FlowControlElement]:
    """Return the flow control a measure map states.

    Coordinates are the measures' ``qstamp`` values: quarters, exact.

    - A row with ``start_repeat`` gives a :class:`JumpTarget`
      (``repeat_start``) at its start.
    - A row with ``end_repeat`` gives a destination (``repeat_end``,
      ``repeat_count: 1``) of the Jump at its end, ``to`` the nearest repeat
      start before it that no earlier repeat end has closed, and without
      place where there is none: nothing is added that the rows do not
      state, a repeat start least of all.
    - Rows with ``volta`` are brackets of alternative endings; adjacent rows
      stating the same passes are one bracket, and adjacent brackets naming
      different passes are one run. A run gives ONE Jump at the start of its
      first bracket with a destination per bracket (``volta``, ``passes``
      the row's value): the first ``by: 0``, every other ``to`` a
      :class:`JumpTarget` (``volta``) at that bracket's first row, where a
      :class:`Break` (``volta``) lies as well.
    - The repeat ends closing the brackets of one run go back to one place,
      the one the first of them closes. One closing another bracket than
      the first is written ``reset: false``: it must not start the Jump
      before the brackets again, whose passes it counts.

    Signs falling on one coordinate are destinations of one Jump. No row
    key says where a bracket ends, so no element does.

    Args:
        measures: A :class:`~timetoalign.timelines.MeasureMap`, or full rows
            as :meth:`MeasureMap.expand_rows
            <timetoalign.timelines.MeasureMap.expand_rows>` returns them, in
            which ``volta`` may be a list of pass numbers.

    Returns:
        The elements by coordinate and then by ID, each with an ID of its
        own (``jump_1``, ``jump_target_1``, ...) that a registry mints again
        where it is taken.

    Raises:
        ValueError: If a measure states no ``qstamp`` or no
            ``actual_length``, or a ``volta`` that is neither a pass number
            nor a list of them.

    Examples:
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows(
        ...     [
        ...         {"count": 1, "time_signature": "4/4"},
        ...         {"count": 2, "start_repeat": True},
        ...         {"count": 3, "end_repeat": True},
        ...         {"count": 4},
        ...     ]
        ... )
        >>> start, repeat = flow_control_from_measures(bars)
        >>> start
        JumpTarget('jump_target_1', repeat_start @ 4 quarters)
        >>> repeat
        Jump('jump_1' @ 12 quarters, 1 destination)
        >>> repeat.targets[0].to_dict()
        {'to': 'jump_target_1', 'repeat_count': 1, 'manifestation': 'repeat_end'}
    """
    draft = _FlowControlDraft()
    draft.add_measures(measures)
    return draft.elements()


# endregion

# region Reading row keys


def measure_keys_from_flow_control(
    elements: Iterable[FlowControlElement], measures: MeasureMap | Iterable[Any]
) -> list[dict[str, Any]]:
    """Return the row keys that flow control states, one object per measure.

    The reverse of :func:`flow_control_from_measures` for the three keys it
    covers, read from what the elements look like (their manifestations):

    - ``start_repeat``: a :class:`JumpTarget` (``repeat_start``) lies at the
      measure's start;
    - ``end_repeat``: the Jump at the measure's end has a ``repeat_end``
      destination;
    - ``volta``: the measure lies under a bracket, whose passes are the
      value (one pass as an integer, several as a list). A bracket begins
      where a ``volta`` destination goes on (``by: 0`` at its Jump, ``to``
      at its jump target) and ends where the next bracket of that Jump
      begins.

    **Where the last bracket of a run ends is no fact of flow control.** It
    stays with the rows: the last bracket is given its first measure, and
    the measures following it for as long as *measures* state these very
    passes as their ``volta``.

    An element lying inside a measure, a bracket stating its passes as
    ``{from: n}`` and a ``to`` naming no jump target among *elements* state
    no row key.

    Args:
        elements: The flow control elements read.
        measures: The measures whose positions (and ``volta`` values, for
            the last bracket of a run) are read, as for
            :func:`flow_control_from_measures`.

    Returns:
        ``{"count", "start_repeat", "end_repeat", "volta"}`` per measure, in
        printed order; ``volta`` is None outside a bracket.

    Raises:
        ValueError: As :func:`flow_control_from_measures`.
    """
    rows = _rows(measures)
    held = list(elements)
    starts = {row.start: index for index, row in enumerate(rows)}
    ends = {row.end: index for index, row in enumerate(rows)}
    keys: list[dict[str, Any]] = [
        {"count": row.count, "start_repeat": False, "end_repeat": False, "volta": None}
        for row in rows
    ]
    landings = {
        element.id: element.coordinate.value
        for element in held
        if isinstance(element, JumpTarget)
    }
    for element in held:
        at = element.coordinate.value
        if isinstance(element, JumpTarget):
            if element.manifestation is FlowManifestation.repeat_start and at in starts:
                keys[starts[at]]["start_repeat"] = True
            continue
        if not isinstance(element, Jump):
            continue
        brackets: list[tuple[Any, tuple[int, ...]]] = []
        for destination in element.targets:
            if destination.manifestation is FlowManifestation.repeat_end:
                if at in ends:
                    keys[ends[at]]["end_repeat"] = True
            elif destination.manifestation is FlowManifestation.volta:
                if not isinstance(destination.passes, tuple):
                    continue
                if destination.to in landings:
                    begins = landings[destination.to]
                elif isinstance(destination.by, Duration):
                    begins = at + destination.by.value
                else:
                    continue
                brackets.append((begins, destination.passes))
        brackets.sort(key=lambda bracket: bracket[0])
        for number, (begins, passes) in enumerate(brackets):
            if begins not in starts:
                continue
            first = starts[begins]
            if number + 1 < len(brackets):
                stop = starts.get(brackets[number + 1][0], first + 1)
            else:
                stop = first + 1
                while stop < len(rows) and rows[stop].passes == passes:
                    stop += 1
            value: int | list[int] = passes[0] if len(passes) == 1 else list(passes)
            for index in range(first, stop):
                keys[index]["volta"] = value
    return keys


# endregion

# region Measure counts


@dataclass(frozen=True)
class MeasureRange:
    """A right-open range of measure counts, as a ``.flow.csv`` row states one.

    Attributes:
        start: The count of the first measure.
        end: The count of the measure behind the last one.
        sections: The names of the atomic sections the range covers, in the
            order of the axis.

    Examples:
        >>> MeasureRange(1, 5, ("A",)).n_measures
        4
    """

    start: int
    end: int
    sections: tuple[str, ...] = ()

    @property
    def n_measures(self) -> int:
        """How many measures the range holds."""
        return self.end - self.start


def _value(position: Any) -> Any:
    return position.value if isinstance(position, Coordinate) else position


def measure_ranges(
    intervals: Iterable[Any],
    measures: MeasureMap | Iterable[Any],
    *,
    sections: Iterable[Any] = (),
    divide_at: Iterable[Any] = (),
) -> tuple[MeasureRange, ...]:
    """State stretches of the axis as right-open ranges of measure counts.

    The form of a ``.flow.csv`` file: a flow is the runs
    :func:`~timetoalign.timelines.flow.engine.play` returns, one range per
    run, and the atomic sections are one range each. The rows of such a
    file never span a section break, where the flow goes on without a jump:
    *divide_at* names the coordinates at which a run is divided.

    Args:
        intervals: The stretches in order, each with a ``start`` and an
            ``end`` on the measures' axis (an :class:`~timetoalign.core.Interval`,
            an atomic section).
        measures: The measures counted, as for
            :func:`flow_control_from_measures`.
        sections: The atomic sections, each with ``name``, ``start`` and
            ``end``; a range lists those lying inside it.
        divide_at: Coordinates at which a stretch crossing them is divided,
            the section breaks' for the form of a ``.flow.csv`` file.

    Returns:
        One range per stretch, or per part of a divided one.

    Raises:
        ValueError: If a stretch begins or ends inside a measure or outside
            the measures: it has no measure counts then.

    Examples:
        >>> from timetoalign.core import Interval
        >>> from timetoalign.timelines import MeasureMap
        >>> bars = MeasureMap.from_rows(
        ...     [{"count": 1, "time_signature": "3/4"}, {"count": 8}]
        ... )
        >>> run = Interval(start=Coordinate(3, "quarters"), end=Coordinate(12, "quarters"))
        >>> measure_ranges([run], bars)
        (MeasureRange(start=2, end=5, sections=()),)
    """
    rows = _rows(measures)
    counts = {row.start: row.count for row in rows}
    if rows:
        counts[rows[-1].end] = rows[-1].count + 1

    def count_at(position: Any) -> int:
        if position in counts:
            return counts[position]
        inside = next(
            (row.count for row in rows if row.start < position < row.end), None
        )
        where = (
            "outside the measures"
            if inside is None
            else f"inside measure count {inside}"
        )
        raise ValueError(
            f"{position} lies {where}: a stretch beginning or ending there has no "
            "measure counts"
        )

    named = [
        (_value(section.start), _value(section.end), section.name)
        for section in sections
    ]
    cuts = sorted({_value(position) for position in divide_at})
    ranges: list[MeasureRange] = []
    for interval in intervals:
        start, end = _value(interval.start), _value(interval.end)
        bounds = [start, *(cut for cut in cuts if start < cut < end), end]
        for first, last in zip(bounds, bounds[1:], strict=False):
            ranges.append(
                MeasureRange(
                    count_at(first),
                    count_at(last),
                    tuple(
                        name
                        for section_start, section_end, name in named
                        if first <= section_start and section_end <= last
                    ),
                )
            )
    return tuple(ranges)


def read_flow_csv(path: Path | str) -> dict[str, tuple[MeasureRange, ...]]:
    """Read a ``.flow.csv`` file: the measure-count ranges of each flow.

    The file has the columns ``flow_mode``, ``source_file``,
    ``software_version``, ``mc_start``, ``mc_end`` and ``atomic_segments``
    (section names joined by ``;``), and any further ones. ``atomic`` rows
    are the atomic sections; ``default`` and ``single`` rows are the flows
    of those conventions; rows of another mode (``ms3``, ``music21``, ...)
    state how another tool unfolds the score and are read like them. A row
    whose ``mc_start`` is ``ERROR`` records that a tool failed and is left
    out.

    Args:
        path: The file.

    Returns:
        The rows of each ``flow_mode`` in file order, by mode.

    Raises:
        ValueError: If the file lacks one of the columns.
    """
    required = ("flow_mode", "mc_start", "mc_end", "atomic_segments")
    flows: dict[str, list[MeasureRange]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in required if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"{path} is no .flow.csv file: it lacks the columns {missing}. "
                f"Available columns: {list(reader.fieldnames or ())}"
            )
        for row in reader:
            if row["mc_start"] == "ERROR":
                continue
            flows.setdefault(row["flow_mode"], []).append(
                MeasureRange(
                    int(row["mc_start"]),
                    int(row["mc_end"]),
                    tuple(
                        name.strip()
                        for name in (row["atomic_segments"] or "").split(";")
                        if name.strip()
                    ),
                )
            )
    return {mode: tuple(ranges) for mode, ranges in flows.items()}


# endregion

# region Findings


@dataclass(frozen=True)
class MeasureFinding:
    """Something a look at the measures found about their flow control.

    A finding is a value handed to whoever corrects the score; nothing is
    changed because of it.

    Attributes:
        kind: What was found.
        count: The count of the measure it was found at.
    """

    kind: MeasureFindingKind
    count: int


def repeat_end_findings(
    measures: MeasureMap | Iterable[Any],
) -> tuple[MeasureFinding, ...]:
    """Find the repeat ends whose repeat would begin right behind a last bracket.

    A repeat end without a printed start is played from the start of its
    section. Where the last sign before it that closes anything (a repeat
    end, or the last bracket of alternative endings) is a last bracket, the
    passage it repeats begins, as a reader takes it, right behind that
    bracket, and playing it from the start of the section goes back over the
    brackets. Each such repeat end is returned with the count of its
    measure; nothing is added to the measures or their flow control.

    Args:
        measures: The measures, as for :func:`flow_control_from_measures`.

    Returns:
        One finding of kind ``repeat_end_behind_brackets`` per such repeat
        end, in printed order.

    Raises:
        ValueError: As :func:`flow_control_from_measures`.
    """
    rows = _rows(measures)
    run_of: dict[int, int] = {}
    closing: dict[int, int] = {}
    last_of_run: set[int] = set()
    for run, brackets in enumerate(_bracket_runs(rows)):
        last_of_run.add(brackets[-1].last)
        for bracket in brackets:
            closing[bracket.last] = run
            for index in range(bracket.first, bracket.last + 1):
                run_of[index] = run

    findings: list[MeasureFinding] = []
    open_starts = 0
    served: set[int] = set()
    for index, row in enumerate(rows):
        if row.start_repeat:
            open_starts += 1
        if not row.end_repeat:
            continue
        own = closing.get(index)
        if own is not None and own in served:
            # A later bracket's repeat end goes where the run's first went.
            continue
        if own is not None:
            served.add(own)
        if open_starts:
            open_starts -= 1
            continue
        # The last sign before it that closes anything, its own run apart.
        before = next(
            (
                earlier
                for earlier in range(index - 1, -1, -1)
                if (earlier not in run_of or run_of[earlier] != run_of.get(index))
                and (rows[earlier].end_repeat or earlier in last_of_run)
            ),
            None,
        )
        if before is not None and before in last_of_run:
            findings.append(
                MeasureFinding(MeasureFindingKind.repeat_end_behind_brackets, row.count)
            )
    return tuple(findings)


# endregion
