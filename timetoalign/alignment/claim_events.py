"""MatchClaims from events: event lookup, Projections and claim creation.

The sides of a claimed pair (an event id, an event dict, or a sentinel) are
resolved to the events their timelines hold; a position no event holds is
stood for by a Projection. :func:`_claims_between_events` is the one
projection operation of the library, behind
:meth:`~timetoalign.alignment.bundle.AlignmentBundle.create_match_claims`
and the loaders that state claims from positions.
"""

from __future__ import annotations

import datetime
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.alignment.anchors import _axis_of, _on_timeline, _refuse_off_axis
from timetoalign.alignment.filters import belief_context
from timetoalign.core import (
    Coordinate,
    Duration,
    EventReference,
    IdCoordinate,
    NumberType,
    Projection,
    TimeUnit,
)
from timetoalign.core.fields import _filled
from timetoalign.core.ids import CHUNK, NOMATCH, SENTINELS, quote_id
from timetoalign.core.provenance import Belief, Provenance
from timetoalign.core.retrieval import coordinate_from_dict
from timetoalign.core.time import number_cell, struct_to_coordinate, struct_to_rational
from timetoalign.timelines.statements import _day

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.claims import MatchClaim, MatchClaimField
    from timetoalign.storage import Extraction

#: The keys of a side stating a position that name where a source file
#: states it: the registry columns of the Projection standing there.
_SOURCE_KEYS = ("locator", "extraction_id")


def _event_coordinate(raw: Any) -> Any:
    """Read an event's coordinate entry, keeping a unit where it states one.

    A :class:`Coordinate` and a coordinate dict (``{unit, number_type,
    value}``) come back as a coordinate; a storage cell (``{value,
    numerator, denominator}``) as its exact value; a number as itself.
    """
    if isinstance(raw, Coordinate):
        return raw
    if isinstance(raw, dict):
        if "unit" in raw:
            return coordinate_from_dict(raw)
        return struct_to_rational(raw)
    return raw


def _refuse_sentinel_id(event_id: object, where: str) -> None:
    """Refuse a sentinel given as the id of the event a side names.

    A sentinel is no event: a CHUNK side states its position without an
    ``id``, a NOMATCH side is :data:`NOMATCH` itself.

    Raises:
        ValueError: If *event_id* is :data:`NOMATCH` or :data:`CHUNK`.
    """
    if event_id in SENTINELS:
        raise ValueError(
            f"{where} names {quote_id(str(event_id))} as an event, a string "
            f"reserved for a sentinel: a CHUNK side is an event dict stating its "
            f"position without 'id', a NOMATCH side the constant NOMATCH."
        )


def _event_id(event: dict[str, Any], where: str) -> str:
    """The id of the event a NOMATCH MatchClaim names, which it must state."""
    event_id = event.get("id")
    _refuse_sentinel_id(event_id, where)
    if event_id is None:
        raise ValueError(
            f"{where} names no event id: a NOMATCH MatchClaim names the event "
            f"that has no counterpart. Give a position without an event of its "
            f"own a Projection event first."
        )
    return str(event_id)


def _side_id(event: dict[str, Any], where: str, *, synchronous: bool) -> str:
    """The id a MatchClaim names on a side: the event's, or :data:`CHUNK`.

    A synchronous claim's side stating its position without an ``id`` is
    CHUNK: the anchor holds its coordinate and no event stands there. A
    claim that is not synchronous holds no anchor, so it names two events.

    Raises:
        ValueError: If the ``id`` is a sentinel, or the side names no event
            on a claim that is not synchronous.
    """
    event_id = event.get("id")
    _refuse_sentinel_id(event_id, where)
    if event_id is not None:
        return str(event_id)
    if synchronous:
        return CHUNK
    raise ValueError(
        f"{where} names no event id: a conceptual MatchClaim names two events. "
        f"Give a position without an event of its own a Projection event first."
    )


def _side_text(event_id: str) -> str:
    """A side's event ID as text renderings show it: a sentinel by its name."""
    return {NOMATCH: "NOMATCH", CHUNK: "CHUNK"}.get(event_id, event_id)


def _event_name(event: dict[str, Any]) -> str | None:
    """The name a side states (an event's, or a CHUNK position's), if any."""
    name = event.get("name")
    return None if name is None else str(name)


#: A side of a claimed pair: the id of an event its timeline holds, an event
#: dict (naming a held event by its ``id``, or stating a position), or
#: :data:`NOMATCH`, the NOMATCH sentinel.
ClaimSide = str | dict[str, Any]


def _held_events(timeline: Any, ids: list[str]) -> list[dict[str, Any] | None]:
    """The events of *timeline* (its descendants included) named by *ids*.

    One lookup per call: the timeline's events, its descendants' at their
    positions on its axis, are read once and the ids found in the id column
    (:func:`pyarrow.compute.index_in`). An event of a descendant counts as an
    event of the timeline, at the timeline's coordinate; an event ID is
    unique among a root timeline and its descendants, so one row answers.

    Returns:
        Per id its event row, ``None`` where the timeline holds none.
    """
    if not ids:
        return []
    table = timeline.get_events(include_children=True).table
    if table.num_rows == 0 or "id" not in table.column_names:
        return [None] * len(ids)
    held = pc.cast(table.column("id").combine_chunks(), pa.string())
    positions = pc.index_in(pa.array(ids, type=pa.string()), value_set=held)
    found = table.take(positions.fill_null(0)).to_pylist()
    return [
        row if valid else None
        for row, valid in zip(found, pc.is_valid(positions).to_pylist(), strict=True)
    ]


def _event_position(raw: Any, timeline: Any) -> Any:
    """A position a held event row states, as a value on *timeline*'s axis.

    A storage cell is read in the timeline's declared number type, so a
    float axis gives back the float it stored; ``None`` stays ``None``.
    """
    if isinstance(raw, dict) and "unit" not in raw:
        return struct_to_coordinate(raw, NumberType(timeline.number_type))
    return raw


def _placed(value: Any, timeline: Any, where: str) -> Coordinate:
    """A stated position placed on *timeline*'s own axis, or refused.

    The one placement (:func:`_on_timeline`): a number is written in the
    timeline's unit and number type, a coordinate in another unit goes
    through the timeline's C-Map. A coordinate left in another number type
    is refused here, before anything is added, and so is a bare float on a
    samples axis: a sample position is a count, and a float there is either
    a count given in the wrong number type or a position in seconds.

    Raises:
        ValueError: If *value* is a float and the timeline counts samples,
            or the position cannot be placed on the timeline's axis.
    """
    if isinstance(value, (float, np.floating)) and TimeUnit(timeline.unit) is (
        TimeUnit.samples
    ):
        raise ValueError(
            f"{where} is the float {value!r}, but timeline "
            f"{quote_id(timeline.id)} counts samples, which are whole counts. "
            f"Either {value!r} is a sample count given in the wrong number type "
            f"(give it as an int), or it is a position in seconds that needs "
            f"converting (give Coordinate({value!r}, 'seconds'), which the "
            f"timeline's C-Map converts). Give positions as Coordinates, which "
            f"state their unit."
        )
    coordinate = _on_timeline(_event_coordinate(value), timeline, where)
    _refuse_off_axis(
        coordinate.unit,
        coordinate.number_type,
        _axis_of(timeline),
        timeline_id=timeline.id,
        where=where,
    )
    return coordinate


_STORED_FLOATS = (-(2.0**63), 2.0**63)
"""The range whose floats a number cell's integer ratio holds: ``[-2**63, 2**63)``."""


def _refuse_unstorable(values: Iterable[Any]) -> None:
    """Refuse a float no number cell stores, before anything is added.

    A timeline stores a float as a number cell (:func:`number_cell`), and so
    does the MatchClaim table an anchor's coordinate; the cell refuses a
    non-finite value (``nan``, ``inf``, ``-inf``) and one too large for its
    integer ratio. The refusal is the number cell's own, so it names the
    same value as the timeline or the table would, and nothing has been
    added.

    Raises:
        ValueError: If a value is a float the number cell refuses.
    """
    low, high = _STORED_FLOATS
    for value in values:
        # A comparison with nan is false, so nan is checked too.
        if isinstance(value, float) and not low <= value < high:
            number_cell(value, NumberType.float)


def _projection_values(projections: Sequence[Projection]) -> list[Any]:
    """The values a timeline stores for its Projections, in its order.

    The timeline refuses its Projections with the number cell's refusal when
    they are added, one timeline after the other, so
    :func:`_refuse_unstorable` runs on these for every timeline first: in
    the order they are added and, within one, row by row and value by value
    in the timeline's order (start, duration, then the end the timeline
    completes as their sum).
    """
    values: list[Any] = []
    for projection in projections:
        start = projection.start.value
        duration = None if projection.duration is None else projection.duration.value
        values.append(start)
        if duration is not None:
            values.append(duration)
            if isinstance(start, float) and isinstance(duration, float):
                values.append(start + duration)
    return values


class _Side(NamedTuple):
    """One side of a claimed pair, resolved before anything is added.

    ``event`` is the held event's row; ``projection`` the Projection to add
    where the side states a position a Projection stands at; ``chunk`` the
    placed ``start`` and ``end`` of a CHUNK side (a synchronous claim between
    two positions, where no Projection stands); all ``None`` for the
    NOMATCH sentinel. ``source`` holds the ``locator`` and ``extraction_id``
    the side states for its Projection (``None`` where it states neither).
    """

    timeline: Any
    event: dict[str, Any] | None
    projection: Projection | None
    name: str | None
    chunk: tuple[Coordinate, Coordinate | None] | None = None
    source: dict[str, Any] | None = None

    @property
    def projection_row(self) -> Projection | dict[str, Any]:
        """The Projection to add, as its row where the side names its source."""
        assert self.projection is not None
        if not self.source:
            return self.projection
        return {**self.projection.to_event_row(), **self.source}


def _claims_between_events(
    pairs: Sequence[tuple[ClaimSide, Any, ClaimSide, Any]],
    *,
    synchronous: bool = True,
    certainty: float = 1.0,
    provenance: Provenance | None = None,
    earlier: MatchClaimField | None = None,
    extractions: Iterable[Extraction] = (),
) -> list[MatchClaim]:
    """MatchClaims between events, a Projection or CHUNK standing for a side without one.

    The one projection operation of the library: the bundle
    (:meth:`~timetoalign.alignment.bundle.AlignmentBundle.create_match_claims`)
    and every loader stating claims from positions use it. Each pair is
    ``(side_a, timeline_a, side_b, timeline_b)``, each side one of:

    - the id of an event its timeline (or a descendant of it) holds;
    - an event dict whose ``id`` its timeline holds: that event;
    - an event dict stating a position: ``start``, and ``end`` for a span,
      each a number on the timeline's axis or a :class:`Coordinate` in any
      unit one of the timeline's C-Maps converts, and optionally ``name``
      (the claim's name of the event), ``locator`` and ``extraction_id``
      (where a source file states the position, and the extraction that
      read it: the Projection's registry row, as
      :meth:`Timeline.add_events` takes them). A **Projection** stands for the
      side there (:meth:`Timeline.add_projections`, get-or-add): under the
      dict's ``id`` (an event the timeline does not hold, ``projection_of``
      null), or, without one, under the id of a Projection the timeline
      already holds for the same event at the same position, else a
      minted ``projection_…``; it names the other side's event in
      ``projection_of`` where the other side is a held event (null where
      that side is a position or :data:`NOMATCH`). Except where both sides
      of a synchronous pair are such dicts without ``id``: no Projection
      stands there, and both sides are :data:`CHUNK`, the claim's anchor
      holding their positions and a ``name`` stated there naming that
      side's position (``event_X_name``);
    - :data:`NOMATCH`, the NOMATCH sentinel, on one side at most.

    A synchronous pair gives :meth:`MatchClaim.from_events` between the two
    sides, an interval claim where both have an end; a pair with
    :data:`NOMATCH` gives :meth:`MatchClaim.nomatch` for the other side's
    event. A second call with the same pairs adds no Projection: it reuses
    every one.

    A side's ``name`` is the claim's name of the Projection: the name the
    first claim on a Projection recorded (in *earlier*, else in this call)
    is the one every later claim on it shows. A CHUNK side is no element,
    so its name is the one its pair states. A side's ``locator`` and
    ``extraction_id`` go into the Projection it adds; a Projection reused
    keeps its own (:meth:`Timeline.add_projections`), and a side naming a
    held event or a CHUNK side adds none, so they name nothing there. An
    extraction a side names that is neither in *extractions* nor held by
    its timeline's family is refused before anything is added; a locator
    naming two events of one extraction is refused as
    :meth:`Timeline.add_projections` refuses it, timeline by timeline.

    Every pair is resolved and every position placed and checked before any
    Projection is added: a float on a samples axis, a span running
    backwards, a position past where its timeline can reach (the end of a
    locked timeline, or an end its parent cannot contain) and a held event
    named with a span not its own are refused, and so is a float start,
    end or duration a timeline or an anchor cannot store, so a refusal
    leaves every timeline unchanged. The Projections of one timeline are
    then added in one call, and a CHUNK side's timeline grows to its
    position as a Projection's would.

    Args:
        pairs: ``(side_a, timeline_a, side_b, timeline_b)`` per claim, the
            timelines as objects.
        synchronous: Whether the matches are temporally synchronous; a
            non-synchronous (conceptual) claim holds no anchors, so a
            Projection stands for each side stating a position. A NOMATCH
            pair is never synchronous.
        certainty: How certain every claim is, in [0, 1].
        provenance: How the claims were made and who believes them.
        earlier: MatchClaims already stated, whose names of Projections
            the new claims keep.
        extractions: The extractions the sides' ``extraction_id`` names
            (or held by the timelines' families already); one no added
            Projection names is not held.

    Returns:
        The claims, in pair order, each holding the id minted at its
        construction (:class:`MatchClaim`).

    Raises:
        KeyError: If an id names no event of its timeline.
        ValueError: If both sides of a pair are :data:`NOMATCH`, a side is
            :data:`CHUNK` or a dict whose ``id`` is :data:`NOMATCH` or
            :data:`CHUNK` (a CHUNK side is a position: a dict without
            ``id``; a NOMATCH side is :data:`NOMATCH`), a pair with CHUNK
            on both sides states an end on one side only, a dict naming no held
            event states no ``start``, a position is a float on a
            samples axis, a span runs backwards, a position lies past where
            its timeline can reach (the end of a locked timeline, or an end
            its parent cannot contain), a held event is named with a span
            not its own, a position cannot be placed on its timeline's
            axis, a float start, end or duration is one its timeline
            cannot store, or a side names an extraction neither given nor
            held by its timeline's family.
        TypeError: If a side is neither a string nor a dict.
    """
    from timetoalign.alignment.claims import MatchClaim

    pairs = list(pairs)
    # One lookup per timeline: every id a side names, read in one pass.
    named: dict[int, tuple[Any, list[str]]] = {}
    for index, (side_a, timeline_a, side_b, timeline_b) in enumerate(pairs):
        if side_a == NOMATCH and side_b == NOMATCH:
            raise ValueError(
                f"Pair {index}: event_a and event_b cannot both be {NOMATCH}"
            )
        for side, timeline in ((side_a, timeline_a), (side_b, timeline_b)):
            if not isinstance(side, (str, dict)):
                raise TypeError(
                    f"Pair {index}: a side of a claimed pair is an event id, an "
                    f"event dict or {NOMATCH}, not {type(side).__name__}"
                )
            if side == NOMATCH:
                continue
            event_id = side if isinstance(side, str) else side.get("id")
            _refuse_sentinel_id(event_id, f"Pair {index}")
            if event_id is not None:
                named.setdefault(id(timeline), (timeline, []))[1].append(str(event_id))
    held: dict[tuple[int, str], dict[str, Any] | None] = {}
    for key, (timeline, ids) in named.items():
        unique = list(dict.fromkeys(ids))
        for event_id, row in zip(unique, _held_events(timeline, unique), strict=True):
            held[(key, event_id)] = row

    def lookup(side: ClaimSide, timeline: Any, index: int) -> dict[str, Any] | None:
        """Return the held event row *side* names, or ``None``.

        Raises:
            KeyError: If *side* is an event id that is held on neither the
                timeline nor its children.
        """
        if side == NOMATCH:
            return None
        event_id = side if isinstance(side, str) else side.get("id")
        row = None if event_id is None else held[(id(timeline), str(event_id))]
        if row is None and isinstance(side, str):
            raise KeyError(
                f"Pair {index}: no event with id {quote_id(side)} on timeline "
                f"{quote_id(timeline.id)} or its children "
                f"({timeline.n_events} events of its own)"
            )
        return row

    resolved: list[tuple[_Side, _Side]] = []
    for index, (side_a, timeline_a, side_b, timeline_b) in enumerate(pairs):
        event_a = lookup(side_a, timeline_a, index)
        event_b = lookup(side_b, timeline_b, index)
        # A synchronous pair of two positions naming no event: CHUNK on both
        # sides, the anchor holding the positions, and no Projection.
        chunk = synchronous and all(
            isinstance(side, dict) and side.get("id") is None
            for side in (side_a, side_b)
        )
        sides = (
            _resolved_side(
                side_a, timeline_a, event_a, timeline_b, event_b, index, chunk=chunk
            ),
            _resolved_side(
                side_b, timeline_b, event_b, timeline_a, event_a, index, chunk=chunk
            ),
        )
        ends = [side.chunk[1] is not None for side in sides if side.chunk is not None]
        if chunk and ends[0] != ends[1]:
            raise ValueError(
                f"Pair {index} states two positions where neither timeline holds "
                f"an event ({CHUNK} on both sides) and an end on side "
                f"{'A' if ends[0] else 'B'} only: the claim's anchors hold its "
                f"positions, an interval an end on both sides. State the end on "
                f"both sides, or on neither."
            )
        resolved.append(sides)

    # Every side is checked; now each timeline's Projections are added at once.
    projected: dict[int, tuple[Any, list[_Side]]] = {}
    for sides in resolved:
        for side in sides:
            if side.projection is not None:
                projected.setdefault(id(side.timeline), (side.timeline, []))[1].append(
                    side
                )
    for _, projecting in projected.values():
        _refuse_unstorable(
            _projection_values(
                [side.projection for side in projecting if side.projection is not None]
            )
        )
    extractions = tuple(extractions)
    _refuse_unknown_extractions(projected.values(), extractions)
    projection_ids = {
        key: iter(
            timeline.add_projections(
                [side.projection_row for side in projecting], extractions=extractions
            )
        )
        for key, (timeline, projecting) in projected.items()
    }
    # A CHUNK side's timeline is made to reach its position, as a Projection
    # there would make it (checked above: the timeline can be made to hold it).
    for sides in resolved:
        for side in sides:
            if side.chunk is not None:
                start, end = side.chunk
                side.timeline._ensure_capacity(start if end is None else end)

    claimed = [
        (_claimed_event(side_a, projection_ids), _claimed_event(side_b, projection_ids))
        for side_a, side_b in resolved
    ]
    _kept_names(resolved, claimed, earlier)

    claims: list[MatchClaim] = []
    for (side_a, side_b), (event_a, event_b) in zip(resolved, claimed, strict=True):
        if event_a is None or event_b is None:
            event, side, other = (
                (event_a, side_a, side_b)
                if event_b is None
                else (event_b, side_b, side_a)
            )
            assert event is not None
            claims.append(
                MatchClaim.nomatch(
                    event,
                    side.timeline,
                    other.timeline,
                    certainty=certainty,
                    provenance=provenance,
                )
            )
            continue
        interval = event_a.get("end") is not None and event_b.get("end") is not None
        claims.append(
            MatchClaim.from_events(
                event_a,
                side_a.timeline,
                event_b,
                side_b.timeline,
                end_coord_key="end" if interval and synchronous else None,
                is_synchronous=synchronous,
                certainty=certainty,
                provenance=provenance,
            )
        )
    return claims


def _refuse_unknown_extractions(
    projected: Iterable[tuple[Any, list[_Side]]], extractions: Sequence[Extraction]
) -> None:
    """Refuse a side naming an extraction neither given nor held, before anything is added.

    A Projection's ``extraction_id`` names an extraction of *extractions* or
    one its timeline's family holds (:meth:`Timeline.add_events` refuses any
    other); checked for every timeline first, so that a refusal leaves every
    timeline unchanged, not only those whose Projections come later.

    Raises:
        ValueError: Naming the timeline and the first such extraction ID.
    """
    given = {extraction.id for extraction in extractions}
    for timeline, sides in projected:
        named = {
            str(side.source["extraction_id"])
            for side in sides
            if side.source is not None and "extraction_id" in side.source
        } - given
        if not named:
            continue
        held = {
            extraction.id for extraction in timeline._tree_root().registry.extractions
        }
        unknown = sorted(named - held)
        if unknown:
            raise ValueError(
                f"A side on timeline {quote_id(timeline.id)} names the extraction "
                f"{quote_id(unknown[0])} for its Projection, which is neither given "
                f"(extractions=) nor held by the timeline's family; nothing is added."
            )


def _resolved_side(
    side: ClaimSide,
    timeline: Any,
    event: dict[str, Any] | None,
    other_timeline: Any,
    other_event: dict[str, Any] | None,
    index: int,
    *,
    chunk: bool = False,
) -> _Side:
    """Resolve one side: its held event, the Projection standing for it, or CHUNK.

    Every stated position is placed on the timeline's axis (:func:`_placed`)
    and checked before anything is added: a span runs forward, the
    timeline can be made to hold it (:func:`_refuse_past_the_end`), and a
    held event named by its ID is given its own span or none. With
    *chunk*, the side is a position of a synchronous pair of two positions:
    it is :data:`CHUNK`, placed and checked alike, and no Projection
    stands there.

    Raises:
        ValueError: If a dict naming no held event states no ``start``, a
            span runs backwards, a position lies past where its timeline
            can reach, a held event is named with a span not its own, or a
            position cannot be placed.
    """
    if side == NOMATCH:
        return _Side(timeline, None, None, None)
    where = f"Pair {index}: the side on timeline {quote_id(timeline.id)}"
    if event is not None:
        if isinstance(side, dict):
            _refuse_other_span(side, timeline, event, where)
        return _Side(timeline, event, None, None)
    assert isinstance(side, dict)
    if side.get("start") is None:
        raise ValueError(
            f"{where} names no event of it and states no 'start': a side of a "
            f"MatchClaim is an event, or a position, where a Projection stands "
            f"for one or, between two positions, {CHUNK}."
        )
    start = _placed(side["start"], timeline, f"{where}, its start")
    end = duration = None
    if side.get("end") is not None:
        end = _placed(side["end"], timeline, f"{where}, its end")
        if end.value < start.value:
            raise ValueError(
                f"{where} runs backwards: its end "
                f"{_as_given(side['end'], end)} lies before its start "
                f"{_as_given(side['start'], start)}. A span ends at or after "
                f"its start: swap the two if they were given in the wrong "
                f"order, or correct the one that is wrong."
            )
        duration = Duration(
            end.value - start.value, start.unit, number_type=start.number_type
        )
    _refuse_past_the_end(
        side["start" if end is None else "end"],
        start if end is None else end,
        timeline,
        where,
    )
    if chunk:
        _refuse_unstorable([start.value] if end is None else [start.value, end.value])
        return _Side(timeline, None, None, _event_name(side), (start, end))
    given = side.get("id")
    reference = (
        EventReference(timeline_id=other_timeline.id, event_id=str(other_event["id"]))
        if given is None and other_event is not None
        else None
    )
    projection = Projection(
        id=None if given is None else str(given),
        start=start,
        duration=duration,
        projection_of=reference,
    )
    source = {key: side[key] for key in _SOURCE_KEYS if side.get(key) is not None}
    return _Side(timeline, None, projection, _event_name(side), source=source or None)


def _as_given(value: Any, placed: Coordinate) -> str:
    """A stated position spelled as given, with its place on the axis.

    A coordinate in another unit is followed by the coordinate it was
    converted to; a number is spelled as written, in the timeline's unit.
    """
    given = _event_coordinate(value)
    if isinstance(given, IdCoordinate):
        given = given.to_coordinate()
    if isinstance(given, Coordinate):
        if given.unit is placed.unit:
            return str(given)
        return f"{given} ({placed} on the timeline)"
    return f"{given} {placed.unit.value}"


def _refuse_past_the_end(
    given: Any, last: Coordinate, timeline: Any, where: str
) -> None:
    """Refuse a stated position the timeline cannot be made to hold.

    Adding the Projection standing there is decided as the timeline decides
    any content it is given, so the refusal comes here, before anything is
    added, instead of halfway through the call. A position within the
    timeline's length is held. Past it, a locked timeline refuses, its
    length being fixed; an unlocked one grows to the position, provided its
    parent contains the new end (asked of the parent itself, which never
    grows with its child, so no ancestor further up is affected). A timeline
    of unknown length is unbounded, except where its parent ends it (a
    segment ends at its successor's offset); that end never moves.

    Args:
        given: The farthest position as the side states it.
        last: That position placed on the timeline's axis.
        timeline: The timeline.
        where: Which side it is, for the message.

    Raises:
        ValueError: If *last* lies past the end of a locked timeline, past
            where the parent ends a timeline of unknown length, or past an
            unlocked timeline's end where its parent cannot contain the
            timeline grown to it.
    """
    length = timeline.length
    parent = timeline._parent
    if length is None:
        bound = timeline._extent
        if bound is None or not last.value > bound:
            return
        raise ValueError(
            f"{where} reaches {_as_given(given, last)}, past the end of timeline "
            f"{quote_id(timeline.id)} at "
            f"{Coordinate(bound, timeline.unit, number_type=timeline.number_type)}, "
            f"where its parent {quote_id(parent.id)} ends it while its length is "
            f"unknown; a MatchClaim never moves that end. Correct the span; or, if "
            f"the position lies beyond the timeline, state it on its parent "
            f"{quote_id(parent.id)}."
        )
    if not last.value > length.value:
        return
    if timeline.is_locked:
        raise ValueError(
            f"{where} reaches {_as_given(given, last)}, past the end of timeline "
            f"{quote_id(timeline.id)} at {length}. The timeline is locked: its "
            f"length is fixed, as a loader fixes a length its source states, and a "
            f"MatchClaim never lengthens it. Correct the "
            f"span; or, if the source is known to be longer than the timeline "
            f"states, make the timeline with the source's length."
        )
    if parent is None:
        return
    try:
        parent._check_child_length(timeline.id, last)
    except ValueError as refusal:
        raise ValueError(
            f"{where} reaches {_as_given(given, last)}, past the end of timeline "
            f"{quote_id(timeline.id)} at {length}, and the timeline cannot grow to "
            f"hold it: {str(refusal).rstrip('.')}. A MatchClaim lengthens an unlocked timeline only "
            f"where its parent contains the new end, and a parent never grows with "
            f"its child. Correct the span; or, if the position lies beyond the "
            f"timeline, state it on its parent {quote_id(parent.id)}."
        ) from refusal


def _refuse_other_span(
    side: dict[str, Any], timeline: Any, event: dict[str, Any], where: str
) -> None:
    """Refuse a held event named by its ID with a span that is not its own.

    A side naming an event the timeline holds is that event: stating its own
    span with the ID, or no span, names it; stating another span would claim
    a position the event does not have.

    Raises:
        ValueError: If the side states a ``start`` or an ``end`` other than
            the event's (an instant stated for an event with an end counts as
            another span, and the reverse).
    """
    if side.get("start") is None and side.get("end") is None:
        return
    own_start = _event_position(event.get("start"), timeline)
    own_end = _event_position(event.get("end"), timeline)
    start = (
        None
        if side.get("start") is None
        else _placed(side["start"], timeline, f"{where}, its start")
    )
    end = (
        None
        if side.get("end") is None
        else _placed(side["end"], timeline, f"{where}, its end")
    )
    if end is None:
        same_end = start is None or own_end is None
    else:
        same_end = own_end is not None and end.value == own_end
    if same_end and (start is None or start.value == own_start):
        return

    def spelled(first: Any, second: Any) -> str:
        """Spell a span for the message: ``from … to …``, ``at …`` or ``the end …``."""
        if first is None:
            return f"the end {second}"
        return f"from {first} to {second}" if second is not None else f"at {first}"

    def on_axis(value: Any) -> Coordinate | None:
        """Return *value* as a coordinate on the timeline's axis, ``None`` for ``None``."""
        if value is None:
            return None
        return Coordinate(value, timeline.unit, number_type=timeline.number_type)

    raise ValueError(
        f"{where} names the event {quote_id(str(event['id']))} it holds, which "
        f"lies {spelled(on_axis(own_start), on_axis(own_end))}, and states "
        f"{spelled(start, end)}: an event named by its ID is given its own span "
        f"or none. Give its own span or leave out 'start' and 'end'; for another "
        f"span, leave out the ID (or give a new one), so that a Projection "
        f"stands there."
    )


def _is_projection(side: _Side) -> bool:
    """Whether a Projection stands for *side*: one to add, or one held."""
    if side.projection is not None:
        return True
    return side.event is not None and side.event.get("event_type") == "Projection"


def _recorded_names(
    earlier: MatchClaimField, keys: Collection[tuple[str, str]]
) -> dict[tuple[str, str], str]:
    """The names earlier MatchClaims recorded for the events *keys* name.

    Each key is ``(timeline_id, event_id)``; its name is the first non-null
    name a claim of *earlier* gives that event, in row order (side A before
    side B within a row). Read column-wise: only the rows naming one of the
    wanted event IDs with a name are materialised.
    """
    if not keys or len(earlier) == 0:
        return {}
    wanted = set(keys)
    event_ids = pa.array(sorted({event_id for _, event_id in wanted}), pa.string())
    table = earlier.table
    found: list[tuple[int, str, tuple[str, str], str]] = []
    for side in ("a", "b"):
        names = table.column(f"event_{side}_name").combine_chunks()
        ids = table.column(f"event_{side}_id").combine_chunks()
        mask = _filled(
            pc.and_(pc.is_valid(names), pc.is_in(ids, value_set=event_ids)),
            pa.scalar(False),
        )
        rows = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        timeline_ids = table.column(f"timeline_{side}_id").combine_chunks().take(rows)
        for row, timeline_id, event_id, name in zip(
            rows.tolist(),
            timeline_ids.to_pylist(),
            ids.take(rows).to_pylist(),
            names.take(rows).to_pylist(),
            strict=True,
        ):
            # The mask keeps the rows naming a wanted event with a name; a row
            # without a timeline ID names no wanted key.
            if timeline_id is None or event_id is None or name is None:
                continue
            found.append((row, side, (timeline_id, event_id), name))
    recorded: dict[tuple[str, str], str] = {}
    for _, _, key, name in sorted(found):
        if key in wanted:
            recorded.setdefault(key, name)
    return recorded


def _kept_names(
    resolved: Sequence[tuple[_Side, _Side]],
    claimed: Sequence[tuple[dict[str, Any] | None, dict[str, Any] | None]],
    earlier: MatchClaimField | None,
) -> None:
    """Give every claim on a Projection the name it was first claimed under.

    A Projection holds no name: the name a side states is the claim's name of
    the event. The first MatchClaim recording a name for a Projection (in
    *earlier*, else the first pair of this call stating one) decides it, and
    every later claim on that Projection shows it, whether it states the
    name again, states none, or states another. The event dicts in
    *claimed* are changed in place.
    """
    projected = [
        (side, event)
        for sides, events in zip(resolved, claimed, strict=True)
        for side, event in zip(sides, events, strict=True)
        if event is not None and _is_projection(side)
    ]
    keys = [(side.timeline.id, str(event["id"])) for side, event in projected]
    recorded = {} if earlier is None else _recorded_names(earlier, keys)
    for key, (_, event) in zip(keys, projected, strict=True):
        if key in recorded:
            event["name"] = recorded[key]
        elif event["name"] is not None:
            recorded[key] = event["name"]


def _claimed_event(
    side: _Side, projection_ids: dict[int, Iterator[str]]
) -> dict[str, Any] | None:
    """The event dict a claim names for *side* (``None`` for NOMATCH).

    A held event's positions are read on its timeline's axis; a Projection
    takes the id its timeline gave it and keeps the name the side stated; a
    CHUNK side states its positions and its name without ``id``.
    """
    if side.chunk is not None:
        start, end = side.chunk
        return {"name": side.name, "start": start, "end": end}
    if side.event is not None:
        return {
            "id": side.event["id"],
            "name": side.event.get("name"),
            "start": _event_position(side.event.get("start"), side.timeline),
            "end": _event_position(side.event.get("end"), side.timeline),
        }
    if side.projection is None:
        return None
    projection = side.projection
    return {
        "id": next(projection_ids[id(side.timeline)]),
        "name": side.name,
        "start": projection.start,
        "end": projection.end,
    }


def _create_match_claims(
    self: AlignmentBundle,
    event_pairs: Sequence[tuple[dict | str, str, dict | str, str]],
    *,
    synchronous: bool = True,
    certainty: float = 1.0,
    provenance: Provenance | None = None,
    actor: str | None = None,
    date: str | datetime.date | None = None,
) -> list[MatchClaim]:
    """Implement :meth:`AlignmentBundle.create_match_claims`."""
    from timetoalign.alignment.claims import _believed_entry

    if actor is None:
        if date is not None:
            raise ValueError(
                "a date dates the belief of the actor stating the MatchClaims; "
                "give actor="
            )
    else:
        provenance = _believed_entry(
            provenance,
            Belief(actor=actor, belief="true", date=_day(date)),
            belief_context(self),
        )
    timelines = {
        timeline_id: self._require_known_timeline(timeline_id)
        for _, tl_a, _, tl_b in event_pairs
        for timeline_id in (tl_a, tl_b)
    }
    claims = _claims_between_events(
        [
            (event_a, timelines[tl_a], event_b, timelines[tl_b])
            for event_a, tl_a, event_b, tl_b in event_pairs
        ],
        synchronous=synchronous,
        certainty=certainty,
        provenance=provenance,
        earlier=self.match_claim_table,
    )
    if claims:
        self.add_match_claims(claims)
    return claims
