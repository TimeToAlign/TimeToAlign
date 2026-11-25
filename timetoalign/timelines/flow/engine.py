"""Play a timeline's flow control into a flow.

:func:`play` walks an axis from 0 to its length and follows the Jumps and
Breaks of a :class:`~timetoalign.timelines.flowcontrol.FlowControlView`.
It needs the elements and the length and nothing else: no measures, no
events, no timeline, and it never reads a manifestation. The result is the
flow as runs of the axis in the order played, and what was found on the way.

The walk::

    position <- 0
    loop
        next <- the nearest coordinate behind position with a Jump or a Break,
                else the length
        play the run [position, next)
        the Jump at next counts an arrival; of its destinations that hold,
            one is taken: the flow goes on at its place
        otherwise a Break at next that holds ends the flow, which begins
            again at the next section break if there is one
        otherwise the flow goes on at next, and ends at the length

An element acts when the flow plays up to it, never when a jump lands on
its coordinate.

**When a destination holds.** All its conditions must: ``passes`` names the
arrival (beyond the highest number the Jump names in a list, the destination
naming it holds), it was taken fewer than ``repeat_count`` times, the
destination its ``after`` names has been taken since it last served, and it
is ``determined``. ``by: 0`` is going on and is never taken.

**When a Break holds.** Without ``ends``, always. With it, when its
``passes`` name the arrival, and exactly it (the passes a Break lists are
those arrivals and no later one; ``{from: n}`` is every arrival from the
n-th on), and the destination its ``after`` names has been taken since the
Break was last restored.

**Which one is taken.** A destination lying before the Jump that restores
(a repeat), the nearest first; then one lying before it that does not (a
leap); then one lying ahead, the nearest first. Two of one kind holding
together are reported, and the nearest is taken.

**Restoring.** A destination lying before its Jump with ``reset`` (the
default) restores every Jump and Break lying wholly inside the stretch gone
back over (its coordinate and all its places in ``[place, Jump)``): its
arrivals, takes and ``after`` flags start again.

**Where the flow ends.** At the length, or at a Break that holds where no
Jump is taken. Ending at a Break before the length is found worth reporting
(``flow_ends_at_break``) unless a section break lies there or one of the
Breaks ending the flow states a condition (``ends``): a Fine ends a piece
regularly, also where it lies between two brackets beside their Break.

**Which elements are read.** Those no actor rejects
(:meth:`FlowControlView.active_elements
<timetoalign.timelines.flowcontrol.FlowControlView.active_elements>`),
or, for a named believer, those that actor does not reject. A ``to`` or an
``after`` naming an element that is not read names nothing, and two Jumps
read at one coordinate cannot both be played: either is refused with a
``ValueError`` in the words the registry refuses such an element with.

**The axis.** Every element read lies from 0 to the length, and so does the
place of every destination: an element or a place outside is refused with a
``ValueError`` naming the element, so that no run ever lies outside the
axis.

**Conventions.** ``default`` is the walk. ``single`` goes through once: no
destination lying before its Jump is taken, no ``after`` holds, and each
Jump and Break is arrived at on pass 1 + the sum of the ``repeat_count`` of
the restoring destinations of other Jumps whose stretch contains its
coordinate. ``printed`` is the axis from 0 to its length.

**Findings of every convention.** ``default`` and ``single`` report what
their walk finds and the stretches they never play; ``printed`` plays the
whole axis and finds nothing. A stretch ``single`` never plays that the
``default`` flow plays has the cause
:attr:`~timetoalign.core.enums.FlowFindingCause.single_playback`: going
through once took the last bracket, or did not take a repeat. Every other
finding has no cause.
"""

from __future__ import annotations

from collections.abc import Collection, Hashable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from timetoshare import DEFAULT_BELIEFS, BeliefState

from timetoalign.core import Coordinate, Interval
from timetoalign.core.enums import FlowConvention, FlowFindingCause, FlowFindingKind

from ..flowcontrol import (
    Break,
    FlowCondition,
    FlowControlElement,
    FlowControlView,
    Jump,
    JumpDestination,
    JumpTarget,
    PassesFrom,
    SectionBreak,
)

_Number = int | float | Fraction
_Key = tuple[str, int]
"""A destination: its Jump's ID and its index in ``targets``."""


@dataclass(frozen=True)
class FlowFinding:
    """Something playing flow control found.

    Attributes:
        kind: What was found.
        coordinate: Where: the element's coordinate, or the start of a
            stretch never played.
        end: The end of a stretch never played; None otherwise.
        message: The finding as a sentence.
        cause: Why the convention played found it: ``single_playback`` for
            a stretch the ``single`` flow leaves out and the ``default``
            flow plays; None for every other finding.
    """

    kind: FlowFindingKind
    coordinate: Coordinate
    end: Coordinate | None
    message: str
    cause: FlowFindingCause | None = None


@dataclass(frozen=True)
class FlowResult:
    """A flow: what is played, in order.

    Attributes:
        runs: The stretches of the axis in the order played, each a
            half-open :class:`~timetoalign.core.Interval`; stretches played
            one after the other without a jump between them are one run.
        findings: What was found on the way, in the order found, the
            stretches never played last, in the order of the axis. The
            ``printed`` flow has none.
    """

    runs: tuple[Interval, ...]
    findings: tuple[FlowFinding, ...] = ()


@dataclass(frozen=True)
class AtomicSection:
    """A stretch of the axis that flow control never divides.

    Attributes:
        name: Its letter: ``A``, ``B``, ... ``Z``, ``AA``, ... in the order
            of the axis.
        start: Where it begins.
        end: Where it ends, exclusive.
    """

    name: str
    start: Coordinate
    end: Coordinate


def _letters(index: int) -> str:
    """The name of the section at *index*: A, ..., Z, AA, AB, ..."""
    name = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        name = chr(ord("A") + digit) + name
    return name


def _top(conditions: Sequence[FlowCondition | None]) -> int:
    """The highest pass number *conditions* name in a list, 0 without one."""
    return max(
        (
            max(condition.passes)
            for condition in conditions
            if condition is not None and isinstance(condition.passes, tuple)
        ),
        default=0,
    )


def _passes_hold(condition: FlowCondition, arrival: int, top: int | None) -> bool:
    """Whether the ``passes`` of *condition* name the *arrival*.

    *top* is the highest pass number the destinations of a Jump name in a
    list: beyond it, the destination naming it holds. None for a Break,
    whose listed passes are exactly those arrivals.
    """
    passes = condition.passes
    if passes is None:
        return True
    if isinstance(passes, PassesFrom):
        return arrival >= passes.first
    return arrival in passes or (top is not None and arrival > top and top in passes)


@dataclass
class _Walk:
    """The elements read once, and the counts of one walk through them."""

    registry: FlowControlView
    length: Coordinate
    elements: list[FlowControlElement]
    jumps: dict[Any, Jump] = field(default_factory=dict[Any, Jump])
    breaks: dict[Any, list[Break]] = field(default_factory=dict[Any, list[Break]])
    section_breaks: list[Any] = field(default_factory=list[Any])
    places: dict[_Key, Any] = field(default_factory=dict[_Key, Any])
    destinations: dict[_Key, JumpDestination] = field(
        default_factory=dict[_Key, JumpDestination]
    )
    # The conditions waiting for each destination to be taken.
    watchers: dict[_Key, list[Hashable]] = field(
        default_factory=dict[_Key, list[Hashable]]
    )
    arrivals: dict[str, int] = field(default_factory=dict[str, int])
    takes: dict[_Key, int] = field(default_factory=dict[_Key, int])
    flags: set[Hashable] = field(default_factory=set[Hashable])
    findings: list[FlowFinding] = field(default_factory=list[FlowFinding])
    reported: set[tuple[FlowFindingKind, str]] = field(
        default_factory=set[tuple[FlowFindingKind, str]]
    )

    def __post_init__(self) -> None:
        end = self.length.value
        by_id = {
            element.id: element
            for element in self.elements
            if isinstance(element, Jump)
        }
        for element in self.elements:
            at = element.coordinate.value
            if isinstance(element, Jump):
                self.jumps[at] = element
                for index, destination in enumerate(element.targets):
                    key = (element.id, index)
                    self.destinations[key] = destination
                    self.places[key] = self.registry.place_of(
                        destination, element, elements=self.elements
                    ).value
                    self._watch(destination, key, by_id)
            elif isinstance(element, Break):
                self.breaks.setdefault(at, []).append(element)
                if element.ends is not None:
                    self._watch(element.ends, element.id, by_id)
            elif isinstance(element, SectionBreak):
                self.section_breaks.append(at)
        self.stops = sorted(
            {at for at in (*self.jumps, *self.breaks) if 0 < at <= end} | {end}
        )

    def _watch(
        self, condition: FlowCondition, holder: Hashable, jumps: dict[str, Jump]
    ) -> None:
        """Note that *holder*'s condition waits for the destination it names."""
        if condition.after is None:
            return
        jump, index = self.registry._destination_among(
            jumps, condition.after, "a condition"
        )
        self.watchers.setdefault((jump.id, index), []).append(holder)

    def coordinate(self, value: Any) -> Coordinate:
        """Return *value* as a coordinate in the unit and number type of the walked length."""
        return Coordinate(value, self.length.unit, number_type=self.length.number_type)

    def find(
        self,
        kind: FlowFindingKind,
        element: FlowControlElement,
        message: str,
    ) -> None:
        """Report *kind* at *element*, once."""
        if (kind, element.id) in self.reported:
            return
        self.reported.add((kind, element.id))
        self.findings.append(FlowFinding(kind, element.coordinate, None, message))

    # -- conventions --------------------------------------------------------

    def single_pass(self, at: Any, own: Jump | None) -> int:
        """The pass an element at *at* is arrived at when going through once."""
        count = 1
        for jump in self.jumps.values():
            if jump is own:
                continue
            for index, destination in enumerate(jump.targets):
                place = self.places[(jump.id, index)]
                if (
                    destination.reset
                    and destination.determined
                    and not destination.goes_on
                    and place <= at < jump.coordinate.value
                ):
                    count += destination.repeat_count or 0
        return count

    # -- one arrival --------------------------------------------------------

    def holding(self, jump: Jump, single: bool) -> list[_Key]:
        """The destinations of *jump* that hold at this arrival, best first."""
        at = jump.coordinate.value
        if single:
            arrival = self.single_pass(at, jump)
        else:
            arrival = self.arrivals[jump.id] = self.arrivals.get(jump.id, 0) + 1
        top = _top(jump.targets)
        kinds: tuple[list[_Key], list[_Key], list[_Key]] = ([], [], [])
        for index, destination in enumerate(jump.targets):
            key = (jump.id, index)
            if destination.goes_on:
                continue
            if not destination.determined:
                self.find(
                    FlowFindingKind.undetermined_target,
                    jump,
                    f"A destination of the jump {jump.id!r} at {jump.coordinate} is "
                    "not determined and is never taken.",
                )
                continue
            before = self.places[key] < at
            if single and (before or destination.after is not None):
                continue
            if not self._holds(destination, key, arrival, top):
                continue
            if (
                destination.repeat_count is not None
                and self.takes.get(key, 0) >= destination.repeat_count
            ):
                continue
            kinds[0 if before and destination.reset else 1 if before else 2].append(key)
        if any(len(kind) > 1 for kind in kinds):
            self.find(
                FlowFindingKind.ambiguous_targets,
                jump,
                f"Two destinations of the jump {jump.id!r} at {jump.coordinate} of "
                "one kind hold together; the nearest is taken.",
            )
        for kind in kinds[:2]:
            kind.sort(key=lambda key: self.places[key], reverse=True)
        kinds[2].sort(key=lambda key: self.places[key])
        return [key for kind in kinds for key in kind]

    def _holds(
        self, condition: FlowCondition, holder: Hashable, arrival: int, top: int | None
    ) -> bool:
        if not _passes_hold(condition, arrival, top):
            return False
        return condition.after is None or holder in self.flags

    def take(self, jump: Jump, key: _Key) -> None:
        """Take the destination *key* of *jump*: count, restore, raise flags."""
        destination = self.destinations[key]
        self.takes[key] = self.takes.get(key, 0) + 1
        self.flags.discard(key)
        place, at = self.places[key], jump.coordinate.value
        if place < at and destination.reset:
            self._restore(place, at)
        self.flags.update(self.watchers.get(key, ()))

    def _restore(self, start: Any, end: Any) -> None:
        """Start again every Jump and Break lying wholly inside ``[start, end)``."""
        for at, jump in self.jumps.items():
            keys = [(jump.id, index) for index in range(len(jump.targets))]
            if start <= at < end and all(
                start <= self.places[key] < end for key in keys
            ):
                self.arrivals.pop(jump.id, None)
                for key in keys:
                    self.takes.pop(key, None)
                    self.flags.discard(key)
        for at, breaks in self.breaks.items():
            if start <= at < end:
                for brk in breaks:
                    self.arrivals.pop(brk.id, None)
                    self.flags.discard(brk.id)

    def ends_at(self, at: Any, single: bool) -> list[Break]:
        """The Breaks at *at* that end the flow at this arrival."""
        ending: list[Break] = []
        for brk in self.breaks.get(at, ()):
            if single:
                arrival = self.single_pass(at, None)
            else:
                arrival = self.arrivals[brk.id] = self.arrivals.get(brk.id, 0) + 1
            if brk.ends is None:
                ending.append(brk)
            elif single and brk.ends.after is not None:
                continue
            elif self._holds(brk.ends, brk.id, arrival, None):
                ending.append(brk)
        return ending

    def state(self, place: Any) -> Hashable:
        """What decides how the flow goes on from *place*.

        An arrival count is read no further than the conditions tell passes
        apart, so that a flow going round for ever comes back to a state.
        """
        arrivals: list[tuple[str, int]] = []
        for jump in self.jumps.values():
            conditions: list[FlowCondition | None] = list(jump.targets)
            arrivals.append((jump.id, self._told_apart(jump.id, conditions)))
        for breaks in self.breaks.values():
            for brk in breaks:
                arrivals.append((brk.id, self._told_apart(brk.id, [brk.ends])))
        takes = sorted(
            (key, count)
            for key, count in self.takes.items()
            if count and self.destinations[key].repeat_count is not None
        )
        return (place, tuple(arrivals), tuple(takes), frozenset(self.flags))

    def _told_apart(
        self, element_id: str, conditions: Sequence[FlowCondition | None]
    ) -> int:
        highest = _top(conditions)
        for condition in conditions:
            if condition is not None and isinstance(condition.passes, PassesFrom):
                highest = max(highest, condition.passes.first)
        return min(self.arrivals.get(element_id, 0), highest + 1)


def _walk(
    registry: FlowControlView,
    length: _Number | Coordinate,
    believer: str | None,
    beliefs: Collection[BeliefState],
) -> _Walk:
    end = registry._coordinate(length)
    if end.value < 0:
        raise ValueError(f"a flow is played over a length, not over {end}")
    elements = registry.active_elements(believer=believer, beliefs=beliefs)
    registry._check(elements, elements)
    registry._check_on_axis(elements, elements, end)
    return _Walk(registry, end, elements)


def _merged(walk: _Walk, runs: Sequence[tuple[Any, Any]]) -> tuple[Interval, ...]:
    """Runs as intervals, each joined to the one before that it goes on from."""
    merged: list[tuple[Any, Any]] = []
    for start, end in runs:
        if start == end:
            continue
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(
        Interval(start=walk.coordinate(start), end=walk.coordinate(end))
        for start, end in merged
    )


def _gaps(start: Any, end: Any, runs: Sequence[Interval]) -> list[tuple[Any, Any]]:
    """The stretches of ``[start, end)`` that lie in no run, in the order of the axis."""
    gaps: list[tuple[Any, Any]] = []
    reached = start
    for first, last in sorted((run.start.value, run.end.value) for run in runs):
        if first > reached:
            gaps.append((reached, min(first, end)))
        reached = max(reached, last)
        if reached >= end:
            break
    if reached < end:
        gaps.append((reached, end))
    return gaps


def _never_played(
    walk: _Walk, runs: Sequence[Interval], default: Sequence[Interval] | None
) -> list[FlowFinding]:
    """A finding for each stretch of the axis that lies in no run.

    With the runs of the *default* flow, a stretch is divided where the
    ``default`` flow begins or ends playing it: what it plays has the cause
    ``single_playback``.
    """
    findings: list[FlowFinding] = []
    for start, end in _gaps(walk.coordinate(0).value, walk.length.value, runs):
        if default is None:
            findings.append(_gap(walk, start, end, None))
            continue
        unplayed = _gaps(start, end, default)
        bounds = sorted({start, end, *(at for gap in unplayed for at in gap)})
        pieces: list[tuple[Any, Any, FlowFindingCause | None]] = []
        for first, last in zip(bounds, bounds[1:], strict=False):
            cause = (
                None
                if any(a <= first and last <= b for a, b in unplayed)
                else FlowFindingCause.single_playback
            )
            if pieces and pieces[-1][2] is cause:
                pieces[-1] = (pieces[-1][0], last, cause)
            else:
                pieces.append((first, last, cause))
        findings.extend(_gap(walk, *piece) for piece in pieces)
    return findings


def _gap(
    walk: _Walk, start: Any, end: Any, cause: FlowFindingCause | None
) -> FlowFinding:
    first, last = walk.coordinate(start), walk.coordinate(end)
    stretch = Interval(start=first, end=last)
    message = (
        f"The stretch {stretch} is never played."
        if cause is None
        else f"The stretch {stretch} is never played: going through once leaves "
        "it out, the default flow plays it."
    )
    return FlowFinding(FlowFindingKind.never_played, first, last, message, cause)


def _runs(walk: _Walk, single: bool) -> list[tuple[Any, Any]]:
    """Walk the axis: the runs played, as pairs of values, in the order played."""
    start, end = walk.coordinate(0).value, walk.length.value
    runs: list[tuple[Any, Any]] = []
    seen: set[Hashable] = set()
    position = start
    while position < end:
        nxt = next(stop for stop in walk.stops if stop > position)
        runs.append((position, nxt))
        jump = walk.jumps.get(nxt)
        if jump is not None:
            holding = walk.holding(jump, single)
            if holding:
                walk.take(jump, holding[0])
                place = walk.places[holding[0]]
                state = walk.state(place)
                if state in seen:
                    walk.find(
                        FlowFindingKind.endless_loop,
                        jump,
                        f"The jump {jump.id!r} at {jump.coordinate} is taken in a "
                        "state the flow was in before: the flow never ends.",
                    )
                    break
                seen.add(state)
                position = place
                continue
        ending = walk.ends_at(nxt, single)
        if ending:
            if (
                all(brk.ends is None for brk in ending)
                and nxt != end
                and nxt not in walk.section_breaks
            ):
                walk.find(
                    FlowFindingKind.flow_ends_at_break,
                    ending[0],
                    f"The flow ends at the break {ending[0].id!r} at "
                    f"{ending[0].coordinate}: no jump there has a "
                    "destination that holds.",
                )
            restart = min(
                (at for at in walk.section_breaks if nxt <= at < end), default=None
            )
            if restart is None:
                break
            position = restart
            continue
        position = nxt
    return runs


def play(
    registry: FlowControlView,
    length: _Number | Coordinate,
    *,
    convention: FlowConvention | str = FlowConvention.default,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> FlowResult:
    """Play flow control over an axis of *length*.

    Args:
        registry: The flow control.
        length: The length of the axis: a coordinate on the registry's axis,
            or a bare number on a bound registry.
        convention: ``default`` (every Jump and Break acts), ``single``
            (once through) or ``printed`` (the axis as it stands).
        believer: The actor whose beliefs alone decide which elements are
            read (:meth:`FlowControlView.active_elements
            <timetoalign.timelines.flowcontrol.FlowControlView.active_elements>`);
            None reads the elements no actor rejects.
        beliefs: With a believer, the belief states read; every state but
            ``"false"`` by default.

    Returns:
        The runs in the order played and the findings. ``default`` and
        ``single`` find a flow that never ends, a flow ending at a Break,
        destinations not determined or holding together, and the
        stretches never played; ``printed`` finds nothing. A stretch
        ``single`` never plays that the ``default`` flow plays has the
        cause ``single_playback``; every other finding has none.

    Raises:
        ValueError: If the length is negative or not on the registry's
            axis, or the elements read cannot be played, in the words the
            registry refuses an element with: a ``to`` or an ``after``
            names an element that is not read (a rejected one names
            nothing), two of the Jumps read lie at one coordinate (a
            rejected Jump and its replacement, read for a believer who
            rejects neither), or an element, or the place a destination
            goes on at, lies before 0 or beyond the length.

    Examples:
        >>> from timetoalign.core import Coordinate
        >>> from timetoalign.timelines.flowcontrol import (
        ...     FlowControlRegistry, Jump, JumpDestination)
        >>> registry = FlowControlRegistry(unit="quarters")
        >>> _ = registry.add(
        ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)]))
        >>> [str(run) for run in play(registry, 12).runs]
        ['[0, 8) quarters', '[0, 12) quarters']
        >>> [str(run) for run in play(registry, 12, convention="single").runs]
        ['[0, 12) quarters']

        Going through once takes the second of two brackets at once:

        >>> from timetoalign.timelines.flowcontrol import JumpTarget
        >>> brackets = FlowControlRegistry(unit="quarters")
        >>> _ = brackets.add_all([
        ...     Jump(Coordinate(4, "quarters"), [
        ...         JumpDestination(by=0, passes=[1]),
        ...         JumpDestination(to="second", passes=[2])]),
        ...     JumpTarget(Coordinate(8, "quarters"), id="second"),
        ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)])])
        >>> [(str(f.kind), str(f.coordinate), str(f.end), str(f.cause))
        ...  for f in play(brackets, 12, convention="single").findings]
        [('never_played', '4 quarters', '8 quarters', 'single_playback')]
    """
    convention = FlowConvention(convention)
    walk = _walk(registry, length, believer, beliefs)
    start, end = walk.coordinate(0).value, walk.length.value
    if convention is FlowConvention.printed:
        return FlowResult(_merged(walk, [(start, end)]))
    single = convention is FlowConvention.single
    played = _merged(walk, _runs(walk, single))
    default = (
        _merged(walk, _runs(_Walk(registry, walk.length, walk.elements), False))
        if single
        else None
    )
    return FlowResult(played, (*walk.findings, *_never_played(walk, played, default)))


def atomic_sections(
    registry: FlowControlView,
    length: _Number | Coordinate,
    *,
    believer: str | None = None,
    beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
) -> tuple[AtomicSection, ...]:
    """Return the stretches of the axis that flow control never divides.

    The intervals between consecutive boundaries: 0, the length, the
    coordinates of Jumps, JumpTargets, Breaks and SectionBreaks, and the
    places of every destination other than ``by: 0``. They are lettered in
    the order of the axis; a run of a flow is the atomic sections it covers.

    Args:
        registry: The flow control.
        length: The length of the axis, as for :func:`play`.
        believer: The actor whose beliefs alone decide which elements are
            read; None reads the elements no actor rejects.
        beliefs: With a believer, the belief states read.

    Raises:
        ValueError: As :func:`play`.

    Examples:
        >>> from timetoalign.core import Coordinate
        >>> from timetoalign.timelines.flowcontrol import (
        ...     FlowControlRegistry, Jump, JumpDestination)
        >>> registry = FlowControlRegistry(unit="quarters")
        >>> _ = registry.add(
        ...     Jump(Coordinate(8, "quarters"), [JumpDestination(repeat_count=1)]))
        >>> [(s.name, s.start.value, s.end.value) for s in atomic_sections(registry, 12)]
        [('A', Fraction(0, 1), Fraction(8, 1)), ('B', Fraction(8, 1), Fraction(12, 1))]
    """
    walk = _walk(registry, length, believer, beliefs)
    start, end = walk.coordinate(0).value, walk.length.value
    boundaries = {start, end}
    for element in walk.elements:
        if isinstance(element, Jump | JumpTarget | Break | SectionBreak):
            boundaries.add(element.coordinate.value)
        if isinstance(element, Jump):
            boundaries.update(
                walk.places[(element.id, index)]
                for index, destination in enumerate(element.targets)
                if not destination.goes_on
            )
    ordered = sorted(at for at in boundaries if start <= at <= end)
    return tuple(
        AtomicSection(_letters(index), walk.coordinate(first), walk.coordinate(last))
        for index, (first, last) in enumerate(zip(ordered, ordered[1:], strict=False))
    )
