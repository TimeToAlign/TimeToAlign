"""AlignmentAnchor and MatchClaim scalars and the paired MatchClaimField.

This module implements the atomic and low-level alignment claims from the
multi-level alignment hierarchy:

- AlignmentAnchor: A coordinate pair, one coordinate on each of two timelines.
- MatchClaim: A claim that two events on different timelines correspond, or
  that one event has no counterpart on the other (NOMATCH), with its
  provenance (:class:`~timetoalign.core.provenance.Provenance`) and certainty.
  A synchronous MatchClaim may state a position on a side whose timeline
  holds no event for it: that side is the sentinel CHUNK.
- MatchClaimField: The MatchClaim table, a ``SemanticField[MatchClaim]``
  holding every kind of MatchClaim as columns, read from and written to
  ``match_claims.parquet``.

The hierarchy is:
    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine

Design:
    AlignmentAnchor and MatchClaim are frozen pydantic v2 ``BaseModel``
    scalars. A MatchClaim always knows its two timelines via top-level
    fields and names an event or a sentinel on each side: two events, one
    event and :data:`NOMATCH` (the event has no counterpart), or, on a
    synchronous claim, :data:`CHUNK` on a side whose position only the
    anchor states. Only synchronous MatchClaims produce AlignmentAnchors, and every coordinate a claim holds is on its
    timeline's own axis, in the timeline's unit and declared number type
    (:meth:`MatchClaim.check_on_timelines`); a position given in another
    unit is converted through the timeline's C-Map before the claim exists.

    ``MatchClaim`` is paired with ``MatchClaimField``, the MatchClaim table:
    one row per claim, one column per ``MatchClaim`` property. A provenance
    shared by every row is held once, at field level, and injected on read.
    A claim's id never changes once given, and a MatchClaim is addressed by
    its id alone: a table or a bundle holding an id refuses a second claim
    under it.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, overload

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from timetoshare import DEFAULT_BELIEFS, BeliefState, check_belief_filter
from timetoshare.statements import used_prefixes
from timetoshare.tables import PROVENANCE_METADATA_KEY

from timetoalign.core import (
    ClaimType,
    Coordinate,
    CoordinateSpec,
    Duration,
    EventReference,
    IdCoordinate,
    Interval,
    NumberType,
    Projection,
    TimeUnit,
)
from timetoalign.core.fields import (
    SemanticField,
    StructField,
    _compared,
    _filled,
    _if_else,
    _inferred_array,
    _null_column,
    _struct_array,
    _struct_array_of,
    _struct_child,
    _struct_members,
    _struct_rows,
    _valid_struct_cell,
    drop_from_schema,
    register_value_projector,
)
from timetoalign.core.ids import (
    CHUNK,
    CLAIM_PREFIX,
    NOMATCH,
    SENTINELS,
    IdMinter,
    content_minter,
    given_minter,
    quote_id,
    table_seed_part,
    validate_id,
)
from timetoalign.core.provenance import (
    Belief,
    Provenance,
    belief_mask,
    provenance_array,
)
from timetoalign.core.retrieval import (
    CoordinateFormat,
    CoordinateResult,
    KeyCollection,
    Rounding,
    classify_dispatch_input,
    coordinate_from_dict,
    coordinate_to_dict,
    format_coordinates,
    validate_key_collection,
)
from timetoalign.core.time import (
    COORDINATE_CELL_TYPE,
    axis_coordinate_cells,
    cell_coordinate,
    cells_from_package,
    compare_number_column,
    coordinate_cell_projector,
    coordinate_cells,
    in_lowest_terms,
    number_cell,
    package_cells,
    struct_to_coordinate,
    struct_to_rational,
)
from timetoalign.core.timestamp import _format_coordinate

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.graph import MatchStamp
    from timetoalign.core.timestamp import ConversionMapsSpec

module_logger = logging.getLogger(__name__)

# region Axes


class _Axis(NamedTuple):
    """A timeline's axis: the unit it counts in and the number type it declares."""

    unit: TimeUnit
    number_type: NumberType


def _axis_of(timeline: Any) -> _Axis:
    """The axis of a timeline, or of a ``(unit, number_type)`` pair.

    Args:
        timeline: A timeline (anything with ``unit`` and ``number_type``) or
            a ``(unit, number_type)`` pair.
    """
    if isinstance(timeline, tuple):
        unit, number_type = timeline
        unit = TimeUnit(unit)
        return _Axis(unit, unit.resolve_number_type(number_type))
    return _Axis(TimeUnit(timeline.unit), NumberType(timeline.number_type))


def _axes(timelines: Iterable[Any] | Mapping[str, Any]) -> dict[str, _Axis]:
    """The axes of *timelines*, keyed by timeline id.

    Args:
        timelines: Timelines (each with ``id``, ``unit`` and
            ``number_type``), or a mapping from a timeline id to a timeline
            or to its ``(unit, number_type)`` pair.
    """
    if isinstance(timelines, Mapping):
        return {str(key): _axis_of(value) for key, value in timelines.items()}
    return {timeline.id: _axis_of(timeline) for timeline in timelines}


def _refuse_off_axis(
    unit: TimeUnit,
    number_type: NumberType,
    axis: _Axis,
    *,
    timeline_id: str,
    where: str,
) -> None:
    """Refuse a coordinate that is not on its timeline's own axis.

    The one check of the module: a coordinate of a MatchClaim is in its
    timeline's unit and declared number type, or it is refused, so one
    position can never appear as two coordinates.

    Args:
        unit: The coordinate's unit.
        number_type: The coordinate's number type.
        axis: The timeline's axis.
        timeline_id: The timeline's id, for the message.
        where: Which coordinate it is, for the message.

    Raises:
        ValueError: If the unit or the number type is not the timeline's.
    """
    if unit is axis.unit and number_type is axis.number_type:
        return
    raise ValueError(
        f"{where} is in {unit.value} ({number_type.name}), but timeline "
        f"{quote_id(timeline_id)} counts in {axis.unit.value} "
        f"({axis.number_type.name}): a MatchClaim's coordinates are on its "
        f"timelines' own axes. Convert a position through the timeline's C-Map "
        f"before the MatchClaim is made."
    )


def _on_timeline(value: Any, timeline: Any, where: str) -> Coordinate:
    """Place a factory's coordinate input on *timeline*'s axis.

    A raw number is a position on the timeline: it takes the timeline's unit
    and is written in its declared number type. A coordinate in another unit
    is converted through the timeline's C-Map. A coordinate in the
    timeline's unit is kept as given, so that the check refuses one in
    another number type rather than rewriting it.

    Args:
        value: A raw number, a :class:`Coordinate` or an
            :class:`IdCoordinate` naming the timeline.
        timeline: The timeline.
        where: Which coordinate it is, for the messages.

    Raises:
        ValueError: If an ``IdCoordinate`` names another timeline, or no
            C-Map converts its unit.
    """
    if isinstance(value, IdCoordinate):
        if value.timeline_id != timeline.id:
            raise ValueError(
                f"{where} names timeline {quote_id(value.timeline_id)}, not "
                f"{quote_id(timeline.id)}"
            )
        value = value.to_coordinate()
    if isinstance(value, Coordinate):
        if value.unit is TimeUnit(timeline.unit):
            return value
        converted = timeline.get_coordinate_at(value, format="coordinate")
        if not isinstance(converted, Coordinate):
            raise TypeError(
                f"{where}: timeline {quote_id(timeline.id)} converted "
                f"{value!r} to {converted!r}, not a coordinate"
            )
        return Coordinate(
            converted.value, converted.unit, number_type=converted.number_type
        )
    return Coordinate(value, timeline.unit, number_type=timeline.number_type)


# endregion


# region Claim ids


def _refuse_held_claim_ids(ids: pa.Array, taken: pa.Array, *, holder: str) -> None:
    """Refuse claim ids already held, or that newcomers hold twice.

    A claim's id never changes once it holds one, and a MatchClaim is
    addressed by its id alone, so an id the holder already holds, or one
    held by two newcomers, is refused, whether it was given or minted.
    Vectorized: the cost is that of hashing the ids, whatever their number.

    Args:
        ids: The newcomers' ids (a string array).
        taken: The ids the holder already holds.
        holder: What holds the ids, for the message (``"bundle"``,
            ``"field"``).

    Raises:
        ValueError: If an id is already held or held twice.
    """
    ids = pc.cast(ids, pa.string())
    present = ids.filter(pc.is_valid(ids))
    held = present.filter(pc.is_in(present, value_set=pc.cast(taken, pa.string())))
    if len(held):
        raise ValueError(
            f"claim id {quote_id(held[0].as_py())} is already held by another "
            f"claim of this {holder}"
        )
    counts = pc.value_counts(present)
    repeated = counts.filter(_compared("greater", _struct_child(counts, "counts"), 1))
    if len(repeated):
        raise ValueError(
            f"claim id {quote_id(repeated.field('values')[0].as_py())} is held "
            f"by {repeated.field('counts')[0].as_py()} claims"
        )


def _minted_claim_ids(
    struct_array: pa.StructArray, minter: IdMinter | None
) -> pa.StructArray:
    """Return the claim rows *struct_array* with an id minted for each row without one.

    The ids are drawn from *minter* in row order (``claim_…``); without a
    minter, from the content minter of the rows
    (:func:`~timetoalign.core.ids.content_minter` of ``"claim"`` and the
    Arrow IPC bytes of every column of the rows but ``id``). Given ids are
    kept.
    """
    ids = pc.cast(struct_array.field("id"), pa.string())
    count = ids.null_count
    if count == 0:
        return struct_array
    if minter is None:
        rows = pa.Table.from_batches(
            [pa.RecordBatch.from_struct_array(struct_array)]
        ).drop_columns(["id"])
        minter = content_minter(CLAIM_PREFIX, table_seed_part(rows))
    filled = pc.replace_with_mask(
        ids, pc.is_null(ids), minter.mint_many(CLAIM_PREFIX, count)
    )
    fields = list(struct_array.type)
    return pa.StructArray.from_arrays(
        [
            filled if field.name == "id" else struct_array.field(field.name)
            for field in fields
        ],
        fields=fields,
    )


def _given_or_seeded_claim_id(claim_id: str | None, seed: int | None) -> str | None:
    """*claim_id*, else with a *seed* the first id of ``IdMinter(seed)``.

    None leaves the claim to mint its id from its content.
    """
    if claim_id is not None or seed is None:
        return claim_id
    return IdMinter(seed).mint(CLAIM_PREFIX)


def _join_claims(
    claims: Sequence["MatchClaim"], taken: pa.Array, *, holder: str
) -> list["MatchClaim"]:
    """Return *claims* as they join a holder of the ids *taken*.

    Every claim holds its id from construction, and joining mints none: a
    claim whose id the holder holds, or two claims under one id, are
    refused before anything changes (:func:`_refuse_held_claim_ids`). A
    claim that belongs to no bundle joins as itself; a claim a bundle
    already holds joins as a copy, so the bundle it came from keeps its
    back-reference.

    Args:
        claims: The newcomers.
        taken: The ids the holder already holds.
        holder: What holds the ids, for the message.

    Returns:
        The claims to store, in order.
    """
    _refuse_held_claim_ids(
        pa.array([claim.id for claim in claims], type=pa.string()),
        pc.cast(taken, pa.string()),
        holder=holder,
    )
    return [claim if claim._bundle is None else claim.model_copy() for claim in claims]


# endregion


# region AlignmentAnchor


def _coordinate_to_dict(coordinate: Coordinate) -> dict[str, Any]:
    """Serialize a coordinate through the shared representation."""
    return dict(coordinate_to_dict(coordinate))


def _coordinate_from_dict(data: dict[str, Any]) -> Coordinate:
    """Restore a coordinate through the shared representation."""
    return coordinate_from_dict(data)


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
    """The name of an event, if it states one."""
    name = event.get("name")
    return None if name is None else str(name)


def _believed_entry(
    provenance: Provenance | None,
    belief: Belief,
    context: Mapping[str, Any] | None,
) -> Provenance:
    """*provenance* holding *belief*, which replaces its actor's earlier one.

    A MatchClaim stated without provenance gets an entry holding the belief
    alone; actors compare as IRIs under *context*
    (:meth:`~timetoalign.core.provenance.Provenance.with_belief`).
    """
    if provenance is None:
        return Provenance(beliefs=(belief,))
    return provenance.with_belief(belief, context=context)


def _provenance_lines(provenance: Provenance | None) -> list[tuple[str, str]]:
    """The labelled display lines of a claim's provenance (none without one).

    The paradata keys set, in order, on one line; the beliefs on another.
    """
    if provenance is None:
        return []
    described = provenance.to_dict()
    beliefs = described.pop("beliefs", [])
    lines: list[tuple[str, str]] = []
    if described:
        lines.append(
            (
                "Provenance",
                ", ".join(
                    f"{key}={' '.join(value) if isinstance(value, list) else value}"
                    for key, value in described.items()
                ),
            )
        )
    if beliefs:
        lines.append(
            (
                "Beliefs",
                "; ".join(
                    f"{belief['actor']} {belief['belief']} ({belief['date']})"
                    for belief in beliefs
                ),
            )
        )
    return lines


class AlignmentAnchor(BaseModel):
    """A coordinate pair associating one coordinate on timeline A with one
    coordinate on timeline B.

    A neutral record with no claim semantics. AlignmentAnchors are value
    objects: two anchors with the same coordinates and timeline IDs are
    equivalent regardless of how they were created.

    Each coordinate is kept as given. An anchor knows its timelines only by
    id; the claim holding it checks both coordinates against the two
    timelines (:meth:`MatchClaim.check_on_timelines`), since only the
    timeline knows its declared number type.

    Only synchronous `MatchClaim` instances produce AlignmentAnchors.
    Non-synchronous claims (conceptual matches, NOMATCH) have no anchors.

    Attributes:
        timeline_a_id: First timeline's unique identifier.
        coordinate_a: Coordinate in first timeline.
        timeline_b_id: Second timeline's unique identifier.
        coordinate_b: Coordinate in second timeline.

    Examples:
        >>> anchor = AlignmentAnchor(
        ...     timeline_a_id="score:1",
        ...     coordinate_a=Coordinate(100, TimeUnit.quarters),
        ...     timeline_b_id="recording:1",
        ...     coordinate_b=Coordinate(45.5, TimeUnit.seconds),
        ... )
        >>> anchor.get_coordinate_for("score:1")
        IdCoordinate(Fraction(100, 1), quarters, 'score:1')
    """

    model_config = ConfigDict(frozen=True)

    timeline_a_id: str
    coordinate_a: Coordinate
    timeline_b_id: str
    coordinate_b: Coordinate

    @property
    def timelines(self) -> tuple[str, str]:
        """Return tuple of timeline IDs."""
        return (self.timeline_a_id, self.timeline_b_id)

    @property
    def coordinates(self) -> tuple[Coordinate, Coordinate]:
        """Return tuple of coordinates."""
        return (self.coordinate_a, self.coordinate_b)

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["id_coordinate"] = ...,
        rounding: Rounding = ...,
    ) -> IdCoordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["coordinate"],
        rounding: Rounding = ...,
    ) -> Coordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat,
        rounding: Rounding = ...,
    ) -> CoordinateResult | pd.Series: ...

    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Get the coordinate for a specific timeline.

        Args:
            timeline_id: The timeline to get coordinate for.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The requested coordinate projection.

        Raises:
            KeyError: If the timeline is not part of this anchor.
        """
        if timeline_id == self.timeline_a_id:
            coordinate = self.coordinate_a
        elif timeline_id == self.timeline_b_id:
            coordinate = self.coordinate_b
        else:
            raise KeyError(f"Timeline {timeline_id!r} is not part of this anchor")
        return format_coordinates(
            [IdCoordinate.from_coordinate(coordinate, timeline_id)],
            format=format,
            rounding=rounding,
            scalar=True,
            index=pd.Index([timeline_id]),
            series_name="coordinate",
        )

    def connects(self, timeline_id: str) -> bool:
        """Check if this anchor connects to a specific timeline.

        Args:
            timeline_id: The timeline to check.

        Returns:
            True if the anchor involves this timeline.
        """
        return timeline_id in (self.timeline_a_id, self.timeline_b_id)

    def connects_both(self, timeline_a_id: str, timeline_b_id: str) -> bool:
        """Check if this anchor connects two specific timelines.

        Args:
            timeline_a_id: First timeline.
            timeline_b_id: Second timeline.

        Returns:
            True if the anchor connects exactly these two timelines.
        """
        return {self.timeline_a_id, self.timeline_b_id} == {
            timeline_a_id,
            timeline_b_id,
        }

    @field_serializer("coordinate_a", "coordinate_b")
    def _serialize_coordinate(self, coordinate: Coordinate) -> dict[str, Any]:
        """Emit a coordinate through the shared JSON representation."""
        return _coordinate_to_dict(coordinate)

    @field_validator("coordinate_a", "coordinate_b", mode="before")
    @classmethod
    def _parse_coordinate(cls, value: Any) -> Any:
        """Accept the serialized coordinate dict as well as a Coordinate."""
        if isinstance(value, dict) and "unit" in value:
            return _coordinate_from_dict(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentAnchor":
        """Deserialize from dictionary."""
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return (
            f"AlignmentAnchor("
            f"{self.timeline_a_id}@"
            f"{_format_coordinate(self.coordinate_a)}"
            f" <-> {self.timeline_b_id}@"
            f"{_format_coordinate(self.coordinate_b)})"
        )

    def __str__(self) -> str:
        """Return the anchor with exact, unit-bearing coordinates."""
        return repr(self)

    def _repr_html_(self) -> str:
        """Return an HTML representation with exact, unit-bearing coordinates."""
        import html as html_mod

        coordinate_a = _format_coordinate(self.coordinate_a)
        coordinate_b = _format_coordinate(self.coordinate_b)
        return (
            "<div style='font-family: monospace;'>"
            "<strong>AlignmentAnchor</strong>("
            f"{html_mod.escape(self.timeline_a_id)}@{html_mod.escape(coordinate_a)} "
            "&lt;-&gt; "
            f"{html_mod.escape(self.timeline_b_id)}@{html_mod.escape(coordinate_b)})"
            "</div>"
        )


# endregion


# region MatchClaim


_ANCHOR_KEYS: tuple[str, ...] = ("coordinate_a", "coordinate_b")
"""The keys of an anchor in a MatchClaim's dictionary and in a row of the
MatchClaim table: its two coordinates. Its timelines are the claim's own
(``timeline_a_id``, ``timeline_b_id``), stated once."""


class MatchClaim(BaseModel):
    """A claim that two events on different timelines correspond.

    A MatchClaim always knows which two timelines it connects via top-level
    ``timeline_a_id`` and ``timeline_b_id`` fields, and it names on each
    side an event or a sentinel, a reserved string no element takes as its
    ID:

    - a synchronous claim names an event or :data:`CHUNK` on each side. A
      CHUNK side is a position the anchor states on a timeline holding no
      event for it: not an event and not a Projection, its coordinate the
      anchor's on that side;
    - a claim that is not synchronous names two events (conceptual), or one
      event and :data:`NOMATCH` on the other side (the event has no
      counterpart there).

    A sentinel side has no name. Anchors are only present for synchronous
    claims, and each of their coordinates is on its timeline's own axis
    (:meth:`check_on_timelines`).

    Two factories:
        (a) ``from_events``: Two sides on different timelines correspond
            (synchronous, with anchors, each side an event or CHUNK; or
            conceptual, between two events, without).
        (b) ``nomatch``: An event has no equivalent on the other timeline.

    A MatchClaim is always stated: the correspondences a
    :class:`~timetoalign.alignment.graph.MatchGraph` infers through groups
    are the graph's own inferred edges, never MatchClaims.

    Attributes:
        timeline_a_id: First timeline's unique identifier.
        timeline_b_id: Second timeline's unique identifier.
        start_anchor: AlignmentAnchor for event starts (None for non-synchronous).
        end_anchor: AlignmentAnchor for event ends (None for instants or
            non-synchronous claims).
        is_synchronous: True if temporally synchronous matches.
        provenance: How the claim was made and who believes it
            (:class:`~timetoalign.core.provenance.Provenance`).
        certainty: How certain the claim is, in [0, 1] (1.0, certain;
            ``None``, not stated).
        event_a_id: ID of the event on timeline A, or a sentinel
            (:data:`NOMATCH`, :data:`CHUNK`).
        event_a_name: Name/label of the event on timeline A (if known;
            ``None`` on a sentinel side).
        event_b_id: ID of the event on timeline B, or a sentinel.
        event_b_name: Name/label of the event on timeline B (if known;
            ``None`` on a sentinel side).
        id: The claim's identifier, which alone addresses it: unique in
            the :class:`MatchClaimField` or ``AlignmentBundle`` holding the
            claim. It never changes: a table or bundle already holding it
            refuses the claim. A claim made without one mints one at
            construction (``claim_…``): from ``IdMinter(seed)`` when
            :meth:`from_events` or :meth:`nomatch` is given ``seed=``, else
            from the content minter of its other properties (:meth:`to_dict`
            without ``id``).

    Examples:
        >>> from timetoalign.timelines.types import (
        ...     ContinuousLogicalTimeline,
        ...     ContinuousPhysicalTimeline,
        ... )
        >>> score = ContinuousLogicalTimeline(length=16, uid="score")
        >>> recording = ContinuousPhysicalTimeline(length=60.0, uid="recording")
        >>> claim = MatchClaim.from_events(
        ...     {"id": "e001", "name": "Note C4", "start": 8},
        ...     score,
        ...     {"id": "e042", "name": "Note C4", "start": 45.5},
        ...     recording,
        ... )
        >>> claim.event_a_id, claim.start_anchor.coordinate_a
        ('e001', Coordinate(Fraction(8, 1), quarters))

        >>> claim = MatchClaim.nomatch({"id": "orphan", "start": 12}, score, recording)
        >>> claim.start_anchor is None, claim.claim_type.name, claim.event_b_id
        (True, 'nomatch', 'NOMATCH_SENTINEL')

        Two positions where neither timeline holds an event, CHUNK on both
        sides:

        >>> claim = MatchClaim.from_events({"start": 4}, score, {"start": 30.0}, recording)
        >>> claim.chunk_sides, claim.claim_type.name
        (('score', 'recording'), 'event_match')
        >>> claim
        MatchClaim(instant: score@4 quarters [CHUNK] <-> recording@30 seconds [CHUNK])
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeline_a_id: str
    timeline_b_id: str
    start_anchor: AlignmentAnchor | None = None
    end_anchor: AlignmentAnchor | None = None
    is_synchronous: bool = True
    provenance: Provenance | None = None
    certainty: float | None = 1.0
    event_a_id: str
    event_a_name: str | None = None
    event_b_id: str
    event_b_name: str | None = None
    id: str | None = None  # noqa: A003 — minted at construction where not given

    # Runtime association with the owning bundle. A private attr takes no
    # part in the derived Arrow schema or the repr, and ``__eq__`` excludes it.
    _bundle: Any = PrivateAttr(default=None)

    @field_validator("certainty")
    @classmethod
    def _validate_certainty(cls, value: float | None) -> float | None:
        """Validate certainty is in the closed range [0, 1] where stated."""
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"Certainty must be in [0, 1], got {value}")
        return value

    @model_validator(mode="after")
    def _validate_claim(self) -> "MatchClaim":
        """Validate the ID, and the timelines, anchors and events against the kind.

        A claim without an ID is minted one from the content minter of its
        other properties.
        """
        if self.id is not None:
            validate_id(self.id)
        if self.timeline_a_id == self.timeline_b_id:
            raise ValueError(
                f"A MatchClaim connects two timelines; both sides are "
                f"{quote_id(self.timeline_a_id)}."
            )

        # Validate anchor presence based on synchrony
        if self.is_synchronous:
            if self.start_anchor is None:
                raise ValueError(
                    "Synchronous MatchClaims require a start_anchor. "
                    "Use is_synchronous=False for conceptual and NOMATCH claims."
                )
            # Anchors connect the claim's timelines, in the claim's order
            for anchor in (self.start_anchor, self.end_anchor):
                if anchor is not None and anchor.timelines != self.timelines:
                    raise ValueError(
                        f"An anchor of a MatchClaim joins its timeline A and "
                        f"its timeline B in that order: the claim joins "
                        f"{quote_id(self.timeline_a_id)} and "
                        f"{quote_id(self.timeline_b_id)}, the anchor "
                        f"{quote_id(anchor.timeline_a_id)} and "
                        f"{quote_id(anchor.timeline_b_id)}."
                    )
        else:
            if self.start_anchor is not None or self.end_anchor is not None:
                raise ValueError(
                    "Non-synchronous MatchClaims must not have anchors. "
                    "Set start_anchor=None and end_anchor=None, or use "
                    "is_synchronous=True."
                )

        # Each side names an event or the sentinel its kind admits.
        sides = (self.event_a_id, self.event_b_id)
        if self.is_synchronous and NOMATCH in sides:
            raise ValueError(
                f"{NOMATCH} stands only on a MatchClaim that is not synchronous: "
                f"a synchronous MatchClaim names an event or {CHUNK} on each side."
            )
        if not self.is_synchronous and CHUNK in sides:
            raise ValueError(
                f"{CHUNK} stands only on a synchronous MatchClaim, whose anchor "
                f"holds its coordinate."
            )
        if sides == (NOMATCH, NOMATCH):
            raise ValueError("A NOMATCH MatchClaim names an event on its other side.")
        for side, (event_id, name) in (
            ("a", (self.event_a_id, self.event_a_name)),
            ("b", (self.event_b_id, self.event_b_name)),
        ):
            if event_id in SENTINELS and name is not None:
                raise ValueError(
                    f"event_{side}_name is {name!r}, but side {side.upper()} is "
                    f"{event_id}: a sentinel side has no name."
                )
        if self.id is None:
            content = {
                key: value for key, value in self.to_dict().items() if key != "id"
            }
            # Frozen model: its id is set once, here, through object.__setattr__.
            object.__setattr__(
                self,
                "id",
                content_minter(CLAIM_PREFIX, content).mint(CLAIM_PREFIX),
            )
        return self

    def __eq__(self, other: object) -> bool:
        """Compare every field; the bundle back-reference takes no part.

        A claim records the bundle holding it (:attr:`bundle`) for
        :meth:`get_matchstamp`; that is where the claim is, not what it
        claims, so a claim equals its copy read back from a
        :class:`MatchClaimField`, in a bundle or not.
        """
        if not isinstance(other, MatchClaim):
            return NotImplemented
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> "MatchClaim":
        """Copy every field deeply; never copy the bundle holding the claim.

        The bundle back-reference (:attr:`bundle`) records where the claim
        is. A copy is not among that bundle's claims, so it is held by no
        bundle and :attr:`bundle` is ``None`` -- unless the bundle itself is
        being copied in the same operation, whose copy then holds the
        claim's copy. Copying the bundle along with one claim would copy
        every other claim it holds, so a deep copy would cost as much as the
        whole alignment.

        Args:
            memo: The memo dictionary of :func:`copy.deepcopy`.

        Returns:
            The copy.
        """
        memo = {} if memo is None else memo
        copied = copy.copy(self)
        object.__setattr__(copied, "__dict__", copy.deepcopy(self.__dict__, memo))
        held_by = self._bundle
        copied._bundle = None if held_by is None else memo.get(id(held_by))
        return copied

    @property
    def is_interval(self) -> bool:
        """Whether this is an interval match (has end anchor)."""
        return self.end_anchor is not None

    @property
    def claim_type(self) -> ClaimType:
        """The semantic kind of this claim, derived from its structure.

        The classification reads only ``is_synchronous`` and whether
        ``event_a_id`` / ``event_b_id`` is :data:`NOMATCH` — no stored field
        records the kind, so it always reflects the claim's current shape.

        Returns:
            The :class:`~timetoalign.core.enums.ClaimType` for this claim.

        The cases, in evaluation order:

        | Condition                                   | Result          |
        |---------------------------------------------|-----------------|
        | synchronous (each side an event or CHUNK)   | ``event_match`` |
        | non-synchronous, one side NOMATCH           | ``nomatch``     |
        | non-synchronous, an event on each side      | ``conceptual``  |
        """
        if self.is_synchronous:
            return ClaimType.event_match
        if NOMATCH in (self.event_a_id, self.event_b_id):
            return ClaimType.nomatch
        return ClaimType.conceptual

    @property
    def chunk_sides(self) -> tuple[str, ...]:
        """The timeline IDs of the sides that are :data:`CHUNK`, in the claim's order.

        Empty for a claim naming an event on each side, a NOMATCH or a
        conceptual claim.
        """
        return tuple(
            timeline_id
            for timeline_id, event_id in (
                (self.timeline_a_id, self.event_a_id),
                (self.timeline_b_id, self.event_b_id),
            )
            if event_id == CHUNK
        )

    @property
    def is_nomatch(self) -> bool:
        """Whether a named event is claimed to have no counterpart."""
        return self.claim_type is ClaimType.nomatch

    @property
    def is_conceptual(self) -> bool:
        """Whether this is a structural correspondence with no anchors."""
        return self.claim_type is ClaimType.conceptual

    @property
    def bundle(self) -> Any:
        """The AlignmentBundle this claim belongs to, if set."""
        return self._bundle

    def set_bundle(self, bundle: Any) -> None:
        """Associate this claim with an AlignmentBundle.

        This is called automatically when claims are added to a bundle.
        Once set, ``get_matchstamp(from_graph=True)`` can be called
        without passing the bundle explicitly.

        Args:
            bundle: The AlignmentBundle containing this claim.
        """
        # Private attrs are mutable on frozen pydantic models.
        self._bundle = bundle

    @property
    def timelines(self) -> tuple[str, str]:
        """Return tuple of timeline IDs."""
        return (self.timeline_a_id, self.timeline_b_id)

    def check_on_timelines(self, timeline_a: Any, timeline_b: Any) -> "MatchClaim":
        """Refuse this claim unless every coordinate is on its timeline's axis.

        Every coordinate the claim holds (both coordinates of each anchor)
        must be in its timeline's unit and declared number type. Every
        factory that receives the timelines runs this check.

        Args:
            timeline_a: Timeline A, or its ``(unit, number_type)`` pair. A
                timeline must be the one ``timeline_a_id`` names.
            timeline_b: Timeline B, likewise. Either may be ``None`` where
                the claim holds no coordinate (a claim without anchors).

        Returns:
            This claim.

        Raises:
            ValueError: If a timeline is not the one the claim names, or a
                coordinate is not on its timeline's axis.
        """
        axes: dict[str, _Axis] = {}
        for claim_id, timeline, side in (
            (self.timeline_a_id, timeline_a, "a"),
            (self.timeline_b_id, timeline_b, "b"),
        ):
            if timeline is None:
                continue
            if not isinstance(timeline, tuple) and timeline.id != claim_id:
                raise ValueError(
                    f"timeline_{side} is {quote_id(timeline.id)}, but the "
                    f"MatchClaim names {quote_id(claim_id)}"
                )
            axes[claim_id] = _axis_of(timeline)

        def refuse(coordinate: Coordinate, timeline_id: str, where: str) -> None:
            """Refuse *coordinate* unless its timeline's axis is given and admits it.

            Raises:
                ValueError: If the axis of *timeline_id* is not given, or the
                    coordinate's unit or number type is off that axis.
            """
            if timeline_id not in axes:
                raise ValueError(
                    f"{where} is on timeline {quote_id(timeline_id)}, whose "
                    f"axis is not given"
                )
            _refuse_off_axis(
                coordinate.unit,
                coordinate.number_type,
                axes[timeline_id],
                timeline_id=timeline_id,
                where=where,
            )

        for anchor_name in ("start_anchor", "end_anchor"):
            anchor = getattr(self, anchor_name)
            if anchor is None:
                continue
            for side in ("a", "b"):
                refuse(
                    getattr(anchor, f"coordinate_{side}"),
                    getattr(anchor, f"timeline_{side}_id"),
                    f"{anchor_name}.coordinate_{side}",
                )
        return self

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["id_coordinate"] = ...,
        rounding: Rounding = ...,
    ) -> IdCoordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["coordinate"],
        rounding: Rounding = ...,
    ) -> Coordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat,
        rounding: Rounding = ...,
    ) -> CoordinateResult | pd.Series: ...

    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return the start anchor coordinate for one member timeline.

        Args:
            timeline_id: The timeline to get coordinates for.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The requested start-coordinate projection.

        Raises:
            ValueError: If the claim is non-synchronous.
            KeyError: If the timeline is not in this claim.
        """
        if self.start_anchor is None:
            raise ValueError(
                f"Claim has no anchors (non-synchronous). "
                f"Cannot get a coordinate for '{timeline_id}'."
            )
        return self.start_anchor.get_coordinate_for(
            timeline_id, format=format, rounding=rounding
        )

    def get_coordinates_for(
        self,
        timeline_ids: KeyCollection,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return start-anchor coordinates for timeline identities.

        Args:
            timeline_ids: Member timeline IDs in retrieval order.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        keys, index = validate_key_collection(timeline_ids)
        results: list[IdCoordinate] = []
        for key in keys:
            result = self.get_coordinate_for(
                key, format="id_coordinate", rounding=rounding
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        if format == "series" and index is None:
            index = pd.Index(keys)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name="coordinate",
        )

    def get_interval_for(self, timeline_id: str) -> Interval:
        """Return the asserted interval for one member timeline.

        Args:
            timeline_id: Member timeline ID.

        Returns:
            The typed asserted interval.

        Raises:
            ValueError: If the claim is non-synchronous or instantaneous.
            KeyError: If the timeline is not in the claim.
        """
        if self.start_anchor is None:
            raise ValueError("Non-synchronous claims contain no coordinate assertion")
        if self.end_anchor is None:
            raise ValueError("Instant claims contain no interval assertion")
        start = self.start_anchor.get_coordinate_for(timeline_id, format="coordinate")
        end = self.end_anchor.get_coordinate_for(timeline_id, format="coordinate")
        assert isinstance(start, Coordinate) and isinstance(end, Coordinate)
        return Interval(start=start, end=end)

    def get_coordinate(
        self,
        at: str | KeyCollection,
        timeline_id: None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a singular or plural timeline-key claim query.

        Args:
            at: One member timeline ID or a collection of member IDs.
            timeline_id: Must be ``None`` because ``at`` carries the keys.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The selected precise-getter result.
        """
        if timeline_id is not None:
            raise TypeError("MatchClaim.get_coordinate does not accept timeline_id")
        # A string is the one input classified as a key.
        if isinstance(at, str):
            return self.get_coordinate_for(at, format=format, rounding=rounding)
        branch = classify_dispatch_input(at, empty_is_keys=True)
        if branch == "keys":
            return self.get_coordinates_for(at, format=format, rounding=rounding)
        raise TypeError("MatchClaim.get_coordinate accepts timeline keys only")

    def connects(self, timeline_id: str) -> bool:
        """Check if this claim connects to a specific timeline.

        Args:
            timeline_id: The timeline to check.

        Returns:
            True if the claim involves this timeline.
        """
        return timeline_id in (self.timeline_a_id, self.timeline_b_id)

    def connects_both(self, timeline_a_id: str, timeline_b_id: str) -> bool:
        """Check if this claim connects two specific timelines.

        Args:
            timeline_a_id: First timeline.
            timeline_b_id: Second timeline.

        Returns:
            True if the claim connects exactly these two timelines.
        """
        return {self.timeline_a_id, self.timeline_b_id} == {
            timeline_a_id,
            timeline_b_id,
        }

    @property
    def anchors(self) -> list[AlignmentAnchor]:
        """Get all anchors in this claim (0, 1, or 2).

        Non-synchronous claims return an empty list.
        """
        if self.start_anchor is None:
            return []
        if self.end_anchor is not None:
            return [self.start_anchor, self.end_anchor]
        return [self.start_anchor]

    def get_matchstamp(
        self,
        *,
        bundle: "AlignmentBundle | None" = None,
        from_graph: bool = True,
        conversion_maps: "ConversionMapsSpec" = False,
    ) -> "MatchStamp | None":
        """Return a MatchStamp for this claim's start anchor.

        With ``from_graph=True`` (the default), the method builds or retrieves
        the full MatchGraph for this claim's coordinate from the bundle's
        cache and returns the FULL MatchStamp combining timestamps from ALL
        groups connected at this coordinate.

        With ``from_graph=False``, constructs a **reduced** MatchStamp from
        only the two timelines in this claim (no graph construction needed).

        Args:
            bundle: The ``AlignmentBundle`` containing this claim. Required
                for ``from_graph=True``. Provides group info and the
                MatchGraph cache.
            from_graph: If True (default), build/retrieve the full
                MatchGraph and return the FULL MatchStamp across ALL
                connected groups. If False, return a reduced 2-timeline
                MatchStamp.
            conversion_maps: C-map conversions available through unit lookup.
                Opt-in: defaults to ``False``.

        Returns:
            MatchStamp with coordinates, or None if this claim is
            non-synchronous (NOMATCH).

        Raises:
            ValueError: If ``from_graph=True`` but no bundle is available
                (neither passed nor set via ``set_bundle()``).

        Examples:
            A score and two performances, each in its own group, each
            claimed at the score's quarter 8:

            >>> from timetoalign.alignment import AlignmentBundle
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ... )
            >>> bundle = AlignmentBundle()
            >>> score = ContinuousLogicalTimeline(length=16, uid="score")
            >>> _ = bundle.add_timeline(score, as_group="score")
            >>> for n, seconds in ((1, 40.0), (2, 32.0)):
            ...     performance = ContinuousPhysicalTimeline(
            ...         length=seconds, uid=f"perf_{n}"
            ...     )
            ...     _ = bundle.add_timeline(performance, as_group=f"perf_{n}")
            ...     claim = MatchClaim.from_events(
            ...         {"id": "note_8", "start": 8},
            ...         score,
            ...         {"id": f"onset_{n}", "start": seconds / 2},
            ...         performance,
            ...         id=f"claim_{n}",
            ...     )
            ...     _ = bundle.add_match_claims([claim])
            >>> # claim: score <-> perf_2

            The full stamp (from the graph) reaches every connected timeline:

            >>> stamp = claim.get_matchstamp()  # uses the claim's bundle
            >>> stamp.n_timelines
            3
            >>> stamp.get_coordinate_for("perf_1", format="float")
            20.0

            The reduced stamp holds the claim's two timelines only:

            >>> claim.get_matchstamp(from_graph=False).n_timelines
            2
        """
        if not self.is_synchronous or self.start_anchor is None:
            return None

        if from_graph:
            # Use provided bundle, or fall back to the claim's bundle
            effective_bundle = bundle if bundle is not None else self._bundle
            if effective_bundle is None:
                raise ValueError(
                    "bundle is required for from_graph=True. Either pass the "
                    "AlignmentBundle or ensure this claim was added to a bundle."
                )
            # Delegate to the bundle's cached MatchGraph mechanism
            # The query is the claim's coordinate on its timeline's axis.
            return effective_bundle.get_matchstamp_at(
                self.start_anchor.get_coordinate_for(self.timeline_a_id),
                conversion_maps=conversion_maps,
            )
        else:
            # Reduced stamp: just the two coordinates from this claim
            from timetoalign.alignment.graph import MatchStamp

            coords = {
                timeline_id: self.start_anchor.get_coordinate_for(
                    timeline_id, format="coordinate"
                )
                for timeline_id in (self.timeline_a_id, self.timeline_b_id)
            }
            effective_bundle = bundle if bundle is not None else self._bundle
            return MatchStamp(
                coordinates=coords,
                source_id=self.timeline_a_id,
                stated_pairs=[(self.timeline_a_id, self.timeline_b_id)],
                inferred_pairs=[],
                source=effective_bundle,
                is_interpolated=False,
                conversion_maps=conversion_maps,
            )

    def with_belief(
        self, belief: Belief, *, context: Mapping[str, Any] | None = None
    ) -> "MatchClaim":
        """A copy of the claim whose provenance holds *belief*.

        Only the provenance changes: the belief replaces its actor's earlier
        one, every other actor's belief stands, and a claim stated without
        provenance gets an entry holding the belief alone. The copy is held
        by no bundle (:attr:`bundle` is ``None``); a bundle records a belief
        in a MatchClaim it holds with :meth:`AlignmentBundle.add_match_claim_belief
        <timetoalign.alignment.bundle.AlignmentBundle.add_match_claim_belief>`.

        Args:
            belief: The belief.
            context: The JSON-LD context under which actors compare as IRIs;
                ``None`` declares no prefix.

        Returns:
            The copy.
        """
        copied = self.model_copy(
            update={"provenance": _believed_entry(self.provenance, belief, context)}
        )
        copied._bundle = None
        return copied

    # region Factory Methods

    @classmethod
    def from_events(
        cls,
        event_a: dict[str, Any],
        timeline_a: Any,
        event_b: dict[str, Any],
        timeline_b: Any,
        *,
        coord_key: str = "start",
        end_coord_key: str | None = None,
        is_synchronous: bool = True,
        certainty: float = 1.0,
        provenance: Provenance | None = None,
        id: str | None = None,  # noqa: A002 — the claim's id
        seed: int | None = None,
    ) -> "MatchClaim":
        """Case (a): Two sides on different timelines correspond.

        On a synchronous claim each side is an event or :data:`CHUNK`: an
        event dict without ``id`` states a position where its timeline holds
        no event, and the claim names CHUNK there, the anchor holding the
        position. A claim that is not synchronous (conceptual) names two
        events.

        Each side's coordinate is placed on its timeline's axis: a number is
        a position on the timeline, in its unit and declared number type; a
        :class:`Coordinate` in another unit is converted through the
        timeline's C-Map; a coordinate in the timeline's unit but another
        number type is refused (:meth:`check_on_timelines`, which runs on
        the claim made).

        Args:
            event_a: Event dict from timeline A; holds ``coord_key`` (and
                ``end_coord_key`` for an interval match) and the event's
                ``id`` (without one, side A is CHUNK, which only a
                synchronous claim names), may hold ``name`` (an event's
                only: a CHUNK side has no name).
            timeline_a: Timeline A.
            event_b: Event dict from timeline B, likewise.
            timeline_b: Timeline B.
            coord_key: Key for the start coordinate in event dicts.
            end_coord_key: Key for the end coordinate (creates an interval
                match).
            is_synchronous: Whether the events are temporally synchronous; a
                non-synchronous (conceptual) claim holds no anchors.
            certainty: How certain the claim is, in [0, 1].
            provenance: How the claim was made and who believes it.
            id: The claim's id; without one, it is minted (:class:`MatchClaim`).
            seed: The seed of the minter of a missing id.

        Returns:
            A MatchClaim naming both sides, with 1 or 2 anchors when
            synchronous.

        Raises:
            ValueError: If an event dict's ``id`` is :data:`NOMATCH` or
                :data:`CHUNK` (a sentinel is no event), an event dict names
                no id on a claim that is not synchronous, a CHUNK side states
                a name, or a coordinate cannot be placed on its timeline's
                axis.
        """
        event_a_id = _side_id(event_a, "event_a", synchronous=is_synchronous)
        event_b_id = _side_id(event_b, "event_b", synchronous=is_synchronous)
        anchors: dict[str, AlignmentAnchor | None] = {
            "start_anchor": None,
            "end_anchor": None,
        }
        if is_synchronous:
            for anchor_name, key in (
                ("start_anchor", coord_key),
                ("end_anchor", end_coord_key),
            ):
                if key is None:
                    continue
                anchors[anchor_name] = AlignmentAnchor(
                    timeline_a_id=timeline_a.id,
                    coordinate_a=_on_timeline(
                        _event_coordinate(event_a[key]),
                        timeline_a,
                        f"{anchor_name}.coordinate_a",
                    ),
                    timeline_b_id=timeline_b.id,
                    coordinate_b=_on_timeline(
                        _event_coordinate(event_b[key]),
                        timeline_b,
                        f"{anchor_name}.coordinate_b",
                    ),
                )

        claim = cls(
            timeline_a_id=timeline_a.id,
            timeline_b_id=timeline_b.id,
            **anchors,
            is_synchronous=is_synchronous,
            certainty=certainty,
            provenance=provenance,
            event_a_id=event_a_id,
            event_a_name=_event_name(event_a),
            event_b_id=event_b_id,
            event_b_name=_event_name(event_b),
            id=_given_or_seeded_claim_id(id, seed),
        )
        return claim.check_on_timelines(timeline_a, timeline_b)

    @classmethod
    def nomatch(
        cls,
        event: dict[str, Any],
        timeline: Any,
        other_timeline: Any,
        *,
        certainty: float = 1.0,
        provenance: Provenance | None = None,
        id: str | None = None,  # noqa: A002 — the claim's id
        seed: int | None = None,
    ) -> "MatchClaim":
        """Case (b): An event has no equivalent on the other timeline.

        Creates a non-synchronous claim with no anchors, naming the event on
        side A and the NOMATCH sentinel (:data:`NOMATCH`) on side B.

        Args:
            event: Event dict from *timeline*; must hold ``id``, may hold
                ``name``.
            timeline: The event's timeline.
            other_timeline: The timeline on which the event has no
                counterpart (a timeline, or its id).
            certainty: How certain the claim is, in [0, 1].
            provenance: How the claim was made and who believes it.
            id: The claim's id; without one, it is minted (:class:`MatchClaim`).
            seed: The seed of the minter of a missing id.

        Returns:
            A non-synchronous MatchClaim with no anchors.

        Raises:
            ValueError: If the event names no id, or names :data:`NOMATCH`
                or :data:`CHUNK` (a sentinel is no event).
        """
        other_id = (
            other_timeline if isinstance(other_timeline, str) else other_timeline.id
        )
        claim = cls(
            timeline_a_id=timeline.id,
            timeline_b_id=other_id,
            is_synchronous=False,
            certainty=certainty,
            provenance=provenance,
            event_a_id=_event_id(event, "event"),
            event_a_name=_event_name(event),
            event_b_id=NOMATCH,
            id=_given_or_seeded_claim_id(id, seed),
        )
        return claim.check_on_timelines(
            timeline, None if isinstance(other_timeline, str) else other_timeline
        )

    # endregion

    @field_serializer("provenance")
    def _serialize_provenance(
        self, provenance: Provenance | None
    ) -> dict[str, Any] | None:
        """Emit the provenance as its dictionary, absent keys left out."""
        return None if provenance is None else provenance.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        The keys are the columns of the MatchClaim table, in its order
        (:attr:`MatchClaimField.table`), and the values those of a row of
        its dictionary (:meth:`MatchClaimField.to_dict`): an anchor holds
        its two coordinates (coordinate dicts), its timelines being the
        claim's ``timeline_a_id`` and ``timeline_b_id``.
        """
        dumped = self.model_dump(mode="json")
        for slot in ("start_anchor", "end_anchor"):
            if dumped[slot] is not None:
                dumped[slot] = {key: dumped[slot][key] for key in _ANCHOR_KEYS}
        return dumped

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchClaim":
        """Deserialize from dictionary (inverse of :meth:`to_dict`).

        An anchor is a mapping holding ``coordinate_a`` and ``coordinate_b``
        (coordinate dicts or coordinates), rebuilt as an
        :class:`AlignmentAnchor` on the claim's ``timeline_a_id`` and
        ``timeline_b_id``; an :class:`AlignmentAnchor` is taken as it is. A
        missing or null ``id`` is minted (:class:`MatchClaim`).

        Raises:
            ValueError: If the timeline IDs are missing, or an anchor mapping
                holds other keys than its two coordinates.
        """
        payload = dict(data)
        for slot in ("start_anchor", "end_anchor", "provenance"):
            if not payload.get(slot):
                payload[slot] = None
        if payload.get("timeline_a_id") is None or payload.get("timeline_b_id") is None:
            raise ValueError("Cannot deserialize MatchClaim: missing timeline IDs.")
        for slot in ("start_anchor", "end_anchor"):
            anchor = payload[slot]
            if not isinstance(anchor, dict):
                continue
            if sorted(anchor) != sorted(_ANCHOR_KEYS):
                raise ValueError(
                    f"A MatchClaim's {slot} holds the keys {list(_ANCHOR_KEYS)}, "
                    f"not {sorted(anchor)}: its timelines are the claim's "
                    f"timeline_a_id and timeline_b_id."
                )
            payload[slot] = AlignmentAnchor(
                timeline_a_id=payload["timeline_a_id"],
                timeline_b_id=payload["timeline_b_id"],
                **anchor,
            )
        return cls.model_validate(payload)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MatchClaim":
        """Reconstruct a claim from a struct-shaped Arrow row dict.

        This is the entry point ``SemanticField.__getitem__`` uses to
        materialise a scalar from the backing struct array. Every coordinate
        cell (both coordinates of each anchor) is read as the coordinate it
        stores, in its unit and number type; the rest of the row is the
        shape :meth:`from_dict` accepts, each anchor rebuilt on the row's
        ``timeline_a_id`` and ``timeline_b_id``.

        Args:
            row: The ``.as_py()`` dict of one struct-column element.

        Returns:
            A reconstructed :class:`MatchClaim`.
        """
        payload = copy.deepcopy(row)
        for path in _coordinate_paths(MatchClaimField.struct_schema()):
            holder = _holder(payload, path)
            if holder is not None and holder[path[-1]] is not None:
                holder[path[-1]] = cell_coordinate(holder[path[-1]])
        return cls.from_dict(payload)

    def __repr__(self) -> str:
        match_type = "interval" if self.is_interval else "instant"
        # The badge is the claim kind, shown for every kind except a plain
        # event_match (which keeps the clean, badge-free form).
        claim_type = self.claim_type
        badge = (
            ""
            if claim_type is ClaimType.event_match
            else f" [{claim_type.name.upper()}]"
        )

        if not self.is_synchronous:
            return f"MatchClaim({self.timeline_a_id} <-> {self.timeline_b_id}{badge})"

        # A CHUNK side is marked after its coordinate.
        chunk_a = " [CHUNK]" if self.event_a_id == CHUNK else ""
        chunk_b = " [CHUNK]" if self.event_b_id == CHUNK else ""
        if self.is_interval:
            interval_a = self.get_interval_for(self.timeline_a_id)
            interval_b = self.get_interval_for(self.timeline_b_id)
            start_a, end_a = interval_a.start, interval_a.end
            start_b, end_b = interval_b.start, interval_b.end
            return (
                f"MatchClaim({match_type}: "
                f"{self.timeline_a_id}["
                f"{_format_coordinate(start_a)}-{_format_coordinate(end_a)}]"
                f"{chunk_a} <-> "
                f"{self.timeline_b_id}["
                f"{_format_coordinate(start_b)}-{_format_coordinate(end_b)}]"
                f"{chunk_b}{badge})"
            )
        else:
            # The validator gives every synchronous claim a start anchor.
            anchor = self.start_anchor
            assert anchor is not None
            return (
                f"MatchClaim({match_type}: "
                f"{self.timeline_a_id}@"
                f"{_format_coordinate(anchor.coordinate_a)}{chunk_a}"
                f" <-> "
                f"{self.timeline_b_id}@"
                f"{_format_coordinate(anchor.coordinate_b)}{chunk_b}"
                f"{badge})"
            )

    def __str__(self) -> str:
        """Readable multi-line display showing claim details.

        Shows claim type, timelines with coordinates, events, certainty
        (where below 1), provenance and beliefs.

        Examples:
            >>> from timetoalign.core.provenance import Belief
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     DiscreteLogicalTimeline,
            ... )
            >>> claim = MatchClaim.from_events(
            ...     {"id": "e001", "name": "Intro", "start": 0, "end": Fraction(1, 2)},
            ...     ContinuousLogicalTimeline(length=4, uid="score:clt_1"),
            ...     {"id": "e042", "name": "Intro", "start": 0, "end": 261},
            ...     DiscreteLogicalTimeline(length=1000, uid="perf:dlt_1"),
            ...     end_coord_key="end",
            ...     certainty=0.9,
            ...     provenance=Provenance(
            ...         method="imported",
            ...         service="https://example.org/partitura",
            ...         beliefs=(Belief(actor="ana", belief="true", date="2026-11-21"),),
            ...     ),
            ... )
            >>> print(claim)
            MatchClaim (synchronous, interval)
              Timeline A:  score:clt_1  [0 quarters -- 1/2 quarters]
              Event A:     e001 "Intro"
              Timeline B:  perf:dlt_1  [0 ticks -- 261 ticks]
              Event B:     e042 "Intro"
              Certainty:   0.9
              Provenance:  method=imported, service=https://example.org/partitura
              Beliefs:     ana true (2026-11-21)
        """

        def _event_str(ev_id: str, ev_name: str | None) -> str:
            """Format event info as 'id "name"', a sentinel by its name."""
            if ev_name:
                return f'{_side_text(ev_id)} "{ev_name}"'
            return _side_text(ev_id)

        # Header: synchronous claims report their instant/interval shape;
        # non-synchronous ones report their derived kind (NOMATCH vs CONCEPTUAL).
        if not self.is_synchronous:
            header = f"MatchClaim ({self.claim_type.name.upper()})"
        elif self.is_interval:
            header = "MatchClaim (synchronous, interval)"
        else:
            header = "MatchClaim (synchronous, instant)"

        lines = [header]

        # Timeline A and Event A
        if self.is_synchronous and self.start_anchor is not None:
            if self.is_interval and self.end_anchor is not None:
                lines.append(
                    f"  Timeline A:  {self.timeline_a_id}  "
                    f"[{_format_coordinate(self.start_anchor.coordinate_a)} -- "
                    f"{_format_coordinate(self.end_anchor.coordinate_a)}]"
                )
            else:
                lines.append(
                    f"  Timeline A:  {self.timeline_a_id}  "
                    f"@{_format_coordinate(self.start_anchor.coordinate_a)}"
                )
        else:
            lines.append(f"  Timeline A:  {self.timeline_a_id}")

        lines.append(f"  Event A:     {_event_str(self.event_a_id, self.event_a_name)}")

        # Timeline B and Event B
        if self.is_synchronous and self.start_anchor is not None:
            if self.is_interval and self.end_anchor is not None:
                lines.append(
                    f"  Timeline B:  {self.timeline_b_id}  "
                    f"[{_format_coordinate(self.start_anchor.coordinate_b)} -- "
                    f"{_format_coordinate(self.end_anchor.coordinate_b)}]"
                )
            else:
                lines.append(
                    f"  Timeline B:  {self.timeline_b_id}  "
                    f"@{_format_coordinate(self.start_anchor.coordinate_b)}"
                )
        else:
            lines.append(f"  Timeline B:  {self.timeline_b_id}")

        lines.append(f"  Event B:     {_event_str(self.event_b_id, self.event_b_name)}")

        if self.certainty is not None and self.certainty < 1.0:
            lines.append(f"  Certainty:   {self.certainty}")
        for label, text in _provenance_lines(self.provenance):
            lines.append(f"  {label + ':':<12} {text}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the MatchClaim as a compact HTML block with claim details.
        """
        import html as html_mod

        from timetoalign.display.html import affordance_line

        # Header badge derives from the claim kind. A plain event_match keeps
        # the instant/interval colour scheme; every other kind shows its name.
        claim_type = self.claim_type
        if claim_type is ClaimType.event_match:
            if self.is_interval:
                badge_bg = "#e3f2fd"
                badge_text = "synchronous, interval"
            else:
                badge_bg = "#e8f5e9"
                badge_text = "synchronous, instant"
        else:  # nomatch, conceptual
            badge_bg = "#ffcdd2"
            badge_text = claim_type.name.upper()

        badge = (
            f"<span style='background: {badge_bg}; padding: 0 4px; "
            f"border-radius: 3px; font-size: 0.8em;'>{badge_text}</span>"
        )

        def _event_html(ev_id: str, ev_name: str | None, label: str) -> str:
            """Format event info as an HTML row, a sentinel by its name."""
            shown = html_mod.escape(_side_text(ev_id))
            if ev_name:
                return (
                    f"<tr style='color: #555; font-size: 0.9em;'><td>{label}</td>"
                    f"<td>{shown}</td>"
                    f'<td>"{html_mod.escape(ev_name)}"</td></tr>'
                )
            return (
                f"<tr style='color: #555; font-size: 0.9em;'><td>{label}</td>"
                f"<td colspan='2'>{shown}</td></tr>"
            )

        rows = []

        # Timeline A row
        if self.is_synchronous and self.start_anchor is not None:
            if self.is_interval and self.end_anchor is not None:
                coord_a = (
                    f"[{_format_coordinate(self.start_anchor.coordinate_a)} &ndash; "
                    f"{_format_coordinate(self.end_anchor.coordinate_a)}]"
                )
            else:
                coord_a = f"@{_format_coordinate(self.start_anchor.coordinate_a)}"

            rows.append(
                f"<tr><td>Timeline A</td>"
                f"<td><strong>{html_mod.escape(self.timeline_a_id)}</strong></td>"
                f"<td>{coord_a}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>Timeline A</td>"
                f"<td>{html_mod.escape(self.timeline_a_id)}</td>"
                f"<td></td></tr>"
            )

        rows.append(_event_html(self.event_a_id, self.event_a_name, "Event A"))

        # Timeline B row
        if self.is_synchronous and self.start_anchor is not None:
            if self.is_interval and self.end_anchor is not None:
                coord_b = (
                    f"[{_format_coordinate(self.start_anchor.coordinate_b)} &ndash; "
                    f"{_format_coordinate(self.end_anchor.coordinate_b)}]"
                )
            else:
                coord_b = f"@{_format_coordinate(self.start_anchor.coordinate_b)}"

            rows.append(
                f"<tr><td>Timeline B</td>"
                f"<td><strong>{html_mod.escape(self.timeline_b_id)}</strong></td>"
                f"<td>{coord_b}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>Timeline B</td>"
                f"<td>{html_mod.escape(self.timeline_b_id)}</td>"
                f"<td></td></tr>"
            )

        rows.append(_event_html(self.event_b_id, self.event_b_name, "Event B"))

        # Certainty and provenance rows
        described = _provenance_lines(self.provenance)
        if self.certainty is not None and self.certainty < 1.0:
            described.insert(0, ("Certainty", str(self.certainty)))
        for label, text in described:
            rows.append(
                f"<tr style='color: #666;'><td>{label}</td>"
                f"<td colspan='2'>{html_mod.escape(text)}</td></tr>"
            )

        return (
            f"<div style='font-family: monospace;'>"
            f"<strong>MatchClaim</strong> {badge}"
            f"<table style='border-collapse: collapse; margin-top: 4px;'>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"{affordance_line(['claim.get_matchstamp()'])}"
            f"</div>"
        )


# endregion


# region MatchClaimField


# An anchor's timelines are its row's: the table cell holds its coordinates.
register_value_projector(AlignmentAnchor, "timeline_a_id", drop_from_schema)
register_value_projector(AlignmentAnchor, "timeline_b_id", drop_from_schema)
register_value_projector(AlignmentAnchor, "coordinate_a", coordinate_cell_projector)
register_value_projector(AlignmentAnchor, "coordinate_b", coordinate_cell_projector)

#: The keys of a coordinate dict (:func:`coordinate_to_dict`), which is how a
#: claim coordinate cell is written in :meth:`MatchClaimField.to_dict`.
_COORDINATE_DICT_KEYS: tuple[str, ...] = ("unit", "number_type", "value")


def _number_column(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
) -> pa.Array | np.ndarray:
    """One number per row as an Arrow array, or as objects where Arrow has no type.

    Exact numbers (Fractions) have no Arrow type and are kept as objects.
    """
    if isinstance(values, pa.ChunkedArray):
        return values.combine_chunks()
    if isinstance(values, pa.Array):
        return values
    try:
        return _inferred_array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return np.asarray(list(values), dtype=object)


def _given(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
) -> pa.BooleanArray:
    """Per row whether a number is given (not null), whatever number it is.

    A NaN is given, so that its cell (a null, since no cell holds it) is
    refused rather than read as no number.
    """
    column = _number_column(values)
    if isinstance(column, pa.Array):
        return column.is_valid()
    return pa.array([value is not None for value in column], type=pa.bool_())


def _timeline_cells(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
    timeline_ids: pa.Array,
    axes: dict[str, _Axis],
) -> pa.StructArray:
    """The coordinate cells of raw numbers, each a position on its row's timeline.

    A number takes the unit of the timeline its row names and is written in
    that timeline's declared number type. The rows of one timeline are built
    together (:func:`~timetoalign.core.time.axis_coordinate_cells`,
    vectorized for a numeric column), so the cost grows with the number of
    timelines, not of rows; the row order is kept.

    Args:
        values: One number per row (null for none).
        timeline_ids: The timeline of each row.
        axes: The axis of every timeline the rows name.

    Raises:
        ValueError: If a row names a timeline whose axis is not given, or
            the values and the timeline ids differ in length.
    """
    column = _number_column(values)
    if len(column) != len(timeline_ids):
        raise ValueError("Coordinate values and timeline ids must have equal length.")
    distinct = pc.unique(timeline_ids).to_pylist()
    unknown = [value for value in distinct if value not in axes]
    if unknown:
        raise ValueError(
            f"The rows name timeline {quote_id(str(unknown[0]))}, which is not "
            f"among the timelines given."
        )
    # Every timeline named is now a key of *axes*, so a string.
    named = [value for value in distinct if isinstance(value, str)]
    parts: list[pa.StructArray] = []
    rows: list[pa.Array] = []
    for timeline_id in named:
        axis = axes[timeline_id]
        where = pc.indices_nonzero(_compared("equal", timeline_ids, timeline_id))
        taken = (
            column.take(where)
            if isinstance(column, pa.Array)
            else column[where.to_numpy()].tolist()
        )
        parts.append(
            axis_coordinate_cells(taken, unit=axis.unit, number_type=axis.number_type)
        )
        rows.append(where)
    if not parts:
        return pa.nulls(0, COORDINATE_CELL_TYPE)
    cells = pa.concat_arrays(parts)
    return cells.take(pc.sort_indices(pa.concat_arrays(rows)))


def _coordinate_paths(
    struct_type: pa.StructType, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Yield the path of every claim coordinate cell in *struct_type*."""
    for sub_field in struct_type:
        path = (*prefix, sub_field.name)
        if sub_field.type == COORDINATE_CELL_TYPE:
            yield path
        elif pa.types.is_struct(sub_field.type):
            yield from _coordinate_paths(sub_field.type, path)


def _holder(row: dict[str, Any] | None, path: tuple[str, ...]) -> dict[str, Any] | None:
    """The mapping holding the last key of *path* in *row*, or ``None``.

    ``None`` where a mapping on the way is null (an absent end anchor), or
    where *row* itself is null, which :class:`MatchClaimField` refuses at
    construction, so its rows never reach here null.
    """
    holder: Any = row
    if holder is None:
        return None
    for key in path[:-1]:
        holder = holder[key]
        if holder is None:
            return None
    return holder


def _with_children(
    array: pa.StructArray,
    children: Mapping[tuple[str, ...], pa.Array],
    prefix: tuple[str, ...] = (),
) -> pa.StructArray:
    """Return *array* with the children at the given paths replaced.

    Every struct on a path is rebuilt with its own validity, so a null end
    anchor stays null.
    """
    arrays: list[pa.Array] = []
    for sub_field in array.type:
        path = (*prefix, sub_field.name)
        child = _struct_child(array, sub_field.name)
        if path in children:
            child = children[path]
        elif any(key[: len(path)] == path for key in children):
            # A path leading further passes through a struct child.
            child = _with_children(_struct_array(child), children, path)
        arrays.append(child)
    return pa.StructArray.from_arrays(
        arrays,
        fields=list(array.type),
        mask=array.is_null() if array.null_count else None,
    )


#: The MatchClaim table's anchor columns, each a struct of two coordinate cells.
_ANCHOR_COLUMNS = ("start_anchor", "end_anchor")
_ANCHOR_SIDES = ("coordinate_a", "coordinate_b")


def _anchors_converted(table: pa.Table, convert: Any) -> pa.Table:
    """*table* with each anchor's two coordinate cells passed through *convert*.

    *convert* takes a column of cells and the name of the column it is
    (``start_anchor.coordinate_a``) and returns the converted column; a null
    anchor stays null. A table without an anchor column, or whose anchor
    column is no struct of the two cells, is returned as it stands, for the
    reader's check of the columns to refuse.
    """
    for name in _ANCHOR_COLUMNS:
        if name not in table.column_names:
            continue
        anchors = table.column(name).combine_chunks()
        if not isinstance(anchors, pa.StructArray) or sorted(
            field.name for field in anchors.type
        ) != sorted(_ANCHOR_SIDES):
            continue
        sides = dict(
            zip(
                (field.name for field in anchors.type),
                _struct_members(anchors),
                strict=True,
            )
        )
        converted = [convert(sides[side], f"{name}.{side}") for side in _ANCHOR_SIDES]
        written = pa.StructArray.from_arrays(
            converted,
            fields=[
                pa.field(side, cells.type)
                for side, cells in zip(_ANCHOR_SIDES, converted, strict=True)
            ],
            mask=anchors.is_null(),
        )
        table = table.set_column(
            table.schema.get_field_index(name), pa.field(name, written.type), written
        )
    return table


def _leaf_columns(table: pa.Table) -> pa.Table:
    """*table* with every struct column flattened into its leaves."""
    while any(pa.types.is_struct(field.type) for field in table.schema):
        table = table.flatten()
    return table


class MatchClaimField(SemanticField[MatchClaim]):
    """The MatchClaim table: a ``SemanticField[MatchClaim]`` columnar store.

    ``MatchClaimField`` is the Field paired with the :class:`MatchClaim`
    scalar. It holds a large set of MatchClaims (millions of rows) as Arrow
    columns instead of one frozen :class:`MatchClaim` per claim; individual
    claims are materialised lazily, only when a row is indexed or iterated.
    Every kind of MatchClaim is a row like any other: synchronous claims
    with their anchors, NOMATCH and conceptual claims without. The event ID
    columns are never null: a side without an event holds its sentinel,
    :data:`NOMATCH` or :data:`CHUNK`, and its name is null.

    Like every paired Field, its struct schema is derived from the scalar by
    :func:`~timetoalign.core.fields.derive_arrow_struct` — so a row's type is
    exactly ``MatchClaimField.pa_schema``. Its :attr:`table` (and the
    Parquet file :meth:`to_parquet` writes, ``match_claims.parquet``) has
    one column per top-level property of :class:`MatchClaim`, each nested as
    the property is. A provenance shared by every row is held once at field
    level (or ``None``) and **injected on read**: the ``provenance`` column
    is left null, and :meth:`__getitem__` gives each materialised claim the
    field's provenance. Claims whose provenance differs, given to
    :meth:`from_claims`, carry it per row.

    A MatchClaim is addressed by its id alone: an id held twice is refused,
    and an id never changes. A field never holds a row without an id: every
    constructor mints the missing ones (``claim_…``), from ``IdMinter(seed)``
    when given ``seed=``, from the minter given as ``minter=`` (a loader's),
    else from the content minter of the rows (every column but ``id``).
    Joining a :class:`~timetoalign.alignment.bundle.AlignmentBundle` mints
    nothing.

    Attributes:
        table: The claims as a :class:`pyarrow.Table` (read-only property),
            one column per ``MatchClaim`` property.
        provenance: Shared :class:`~timetoalign.core.provenance.Provenance`
            (or ``None``) applied to every materialised claim.

    Examples:
        >>> field = MatchClaimField.from_columns(
        ...     timeline_a_ids=["A", "A", "B"],
        ...     timeline_b_ids=["B", "C", "C"],
        ...     event_a_ids=["a1", "a1", "b1"],
        ...     event_b_ids=["b0", "c0", "c1"],
        ...     coordinate_a=[0, 0, 1],
        ...     coordinate_b=[10.0, 20.0, 21.0],
        ...     timelines={
        ...         "A": ("quarters", "fraction"),
        ...         "B": ("seconds", "float"),
        ...         "C": ("seconds", "float"),
        ...     },
        ...     provenance=Provenance(service="https://example.org/aligner"),
        ... )
        >>> len(field)
        3
        >>> field[0].timeline_a_id, field[0].timeline_b_id
        ('A', 'B')
    """

    #: The Arrow schema-metadata key carrying the field-level provenance,
    #: TimeToShare's (``timetoshare:provenance``).
    _PROVENANCE_KEY: ClassVar[bytes] = PROVENANCE_METADATA_KEY.encode()
    #: The file name of the MatchClaim table beside a manifest.
    file_name: ClassVar[str] = "match_claims.parquet"
    #: The per-row values beside the anchors and the provenance that
    #: :meth:`from_claims` reads from each claim under its own name; the
    #: types are the derived struct's.
    _ROW_VALUES: ClassVar[tuple[str, ...]] = (
        "id",
        "is_synchronous",
        "certainty",
        "event_a_id",
        "event_a_name",
        "event_b_id",
        "event_b_name",
    )

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def from_columns(
        cls,
        timeline_a_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        timeline_b_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        event_a_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        event_b_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        coordinate_a: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
        coordinate_b: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
        *,
        timelines: Iterable[Any] | Mapping[str, Any],
        end_coordinate_a: (
            Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray | None
        ) = None,
        end_coordinate_b: (
            Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray | None
        ) = None,
        certainty: float = 1.0,
        provenance: Provenance | None = None,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Build a field of event matches from parallel columns (the vectorized path).

        This is the constructor a bulk producer (e.g. an alignment loader)
        uses. It assembles the derived ``MatchClaim`` struct array straight
        from the parallel inputs and **never** materialises a single
        :class:`MatchClaim` Python object, so it scales to millions of rows.

        Each row is a synchronous claim between two sides: the timeline
        and event ids are filled from the inputs, a side an event ID or
        :data:`CHUNK` (a position on a timeline holding no event for it),
        and every coordinate is a position on its row's timeline, written in that timeline's unit and
        declared number type. With the two end columns, every row with an
        end on both sides is an interval claim. The ids are minted, in row
        order (:class:`MatchClaimField`). Names are left null,
        and so is the ``provenance`` column (the shared provenance is
        injected on read). The field is checked against the timelines
        (:meth:`check_on_timelines`).

        Args:
            timeline_a_ids: Timeline-A ids, one per claim.
            timeline_b_ids: Timeline-B ids, one per claim.
            event_a_ids: The event on timeline A, or :data:`CHUNK`, one per
                claim.
            event_b_ids: The event on timeline B, or :data:`CHUNK`, one per
                claim.
            coordinate_a: The position of side A on timeline A.
            coordinate_b: The position of side B on timeline B.
            timelines: Every timeline the rows name: timelines, or a mapping
                from a timeline id to a timeline or its ``(unit,
                number_type)`` pair.
            end_coordinate_a: The end of each event on timeline A (null on
                both sides for an instant claim).
            end_coordinate_b: The end of each event on timeline B.
            certainty: How certain every claim is, in [0, 1].
            provenance: Shared provenance for every claim in the field.
            seed: The seed of the minter of the ids.
            minter: The minter of the ids (a loader's).

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            TypeError: If both *seed* and *minter* are given.
            ValueError: If the inputs do not all have the same length, a row
                names no event ID or sentinel on a side (a null), names
                :data:`NOMATCH` (a synchronous claim names an event or
                :data:`CHUNK`), a row names a timeline not among
                *timelines*, the end columns are given one without the
                other, a row gives an end on one side only, or a coordinate
                is a number no cell holds (a NaN).
        """
        given = given_minter(seed, minter)
        axes = _axes(timelines)
        tl_a = cls._string_array(timeline_a_ids)
        tl_b = cls._string_array(timeline_b_ids)
        ev_a = cls._string_array(event_a_ids)
        ev_b = cls._string_array(event_b_ids)
        if (end_coordinate_a is None) != (end_coordinate_b is None):
            raise ValueError(
                "from_columns takes the end columns of both sides, or neither."
            )
        columns = {
            "timeline_a_id": tl_a,
            "timeline_b_id": tl_b,
            "event_a_id": ev_a,
            "event_b_id": ev_b,
            "coordinate_a": coordinate_a,
            "coordinate_b": coordinate_b,
        }
        if end_coordinate_a is not None:
            columns["end_coordinate_a"] = end_coordinate_a
            columns["end_coordinate_b"] = end_coordinate_b
        lengths = {name: len(column) for name, column in columns.items()}
        if len(set(lengths.values())) != 1:
            described = ", ".join(
                f"{name}={length}" for name, length in lengths.items()
            )
            raise ValueError(
                f"from_columns requires equal-length columns; got lengths {described}."
            )
        n = len(tl_a)
        for side, events in (("a", ev_a), ("b", ev_b)):
            if events.null_count:
                row = pc.index(pc.is_null(events), True).as_py()
                raise ValueError(
                    f"Row {row} names nothing on side {side}: a side of a "
                    f"MatchClaim names an event, or {CHUNK} for a position on a "
                    f"timeline holding no event for it."
                )
        anchors = {
            "start_anchor": cls._anchor_array(
                _timeline_cells(coordinate_a, tl_a, axes),
                _timeline_cells(coordinate_b, tl_b, axes),
            )
        }
        if end_coordinate_a is not None and end_coordinate_b is not None:
            # Only a row without an end on either side is an instant claim:
            # an end given on one side alone, or one no cell can hold (a
            # non-finite number), leaves a null coordinate in the end
            # anchor, which the reader refuses.
            anchors["end_anchor"] = cls._anchor_array(
                _timeline_cells(end_coordinate_a, tl_a, axes),
                _timeline_cells(end_coordinate_b, tl_b, axes),
                mask=pc.and_(
                    pc.invert(_given(end_coordinate_a)),
                    pc.invert(_given(end_coordinate_b)),
                ),
            )
        true_column = pa.array(np.ones(n, dtype=bool), type=pa.bool_())
        struct_array = cls._assemble(
            n,
            {
                "timeline_a_id": tl_a,
                "timeline_b_id": tl_b,
                "event_a_id": ev_a,
                "event_b_id": ev_b,
                "is_synchronous": true_column,
                "certainty": pa.array(np.full(n, float(certainty)), pa.float64()),
                **anchors,
            },
        )
        return cls._read(
            struct_array,
            provenance,
            source="column set",
            where="as provenance=",
            minter=given,
        ).check_on_timelines(axes)

    @classmethod
    def from_claims(
        cls,
        claims: Sequence[MatchClaim],
        *,
        provenance: Provenance | None = None,
    ) -> "MatchClaimField":
        """Build a field from existing :class:`MatchClaim` objects.

        Every kind of claim is a row: synchronous claims with their anchors
        (interval claims with both), NOMATCH and conceptual claims without.
        Each row keeps every property of its claim, so every claim
        materialises equal to the one stored. A MatchClaim is addressed by
        its id alone: two claims under one id are refused (an id is never
        renamed); every claim holds its id from construction.

        If ``provenance`` is ``None`` and all claims share one equal
        :class:`~timetoalign.core.provenance.Provenance`, it is adopted as
        the field-level provenance and the rows carry none. If the claims'
        provenance differs (``None`` included), the field's stays ``None``
        and each row carries its own claim's.

        Args:
            claims: The claims to store.
            provenance: Explicit field-level provenance. When provided it
                replaces every claim's own.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If two claims hold the same id, or an anchor holds a
                coordinate no cell can hold (a non-finite number), which
                every reader of the MatchClaim table refuses.
        """
        claims = list(claims)
        if provenance is None:
            provenance = cls._common_provenance(claims)
        schema = cls.struct_schema()
        columns: dict[str, pa.Array] = {
            name: pa.array(
                [getattr(claim, name) for claim in claims],
                type=schema.field(name).type,
            )
            for name in ("timeline_a_id", "timeline_b_id", *cls._ROW_VALUES)
        }
        for anchor_name in ("start_anchor", "end_anchor"):
            anchors = [getattr(claim, anchor_name) for claim in claims]
            columns[anchor_name] = cls._anchor_array(
                coordinate_cells([a and a.coordinate_a for a in anchors]),
                coordinate_cells([a and a.coordinate_b for a in anchors]),
                mask=pa.array([a is None for a in anchors], type=pa.bool_()),
            )
        if provenance is None:
            columns["provenance"] = provenance_array(
                [claim.provenance for claim in claims]
            )
        # Each claim holds the scalar's rules; a coordinate no cell can hold
        # (a non-finite number) shows only once the cells are written.
        struct_array = cls._assemble(len(claims), columns)
        cls._refuse_null_coordinates(struct_array)
        return cls._admitted(struct_array, provenance=provenance)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Deserialize a field from a plain dictionary (inverse of :meth:`to_dict`).

        The mapping holds ``provenance`` (the field's
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`, or ``None``)
        and ``rows``: one mapping per claim whose keys are the
        :attr:`table`'s columns. Each coordinate is a coordinate dict
        (``{unit, number_type, value}``), read by
        :func:`~timetoalign.core.retrieval.coordinate_from_dict` and written
        into its storage cell as :meth:`from_claims` writes a claim's; every
        other cell is read as stored. A field built from claims therefore
        comes back with an equal table and equal claims. A row without an id
        is minted one (:class:`MatchClaimField`).

        Args:
            data: A mapping as :meth:`to_dict` writes it.
            seed: The seed of the minter of the missing ids.
            minter: The minter of the missing ids (a loader's).

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            TypeError: If both *seed* and *minter* are given.
            ValueError: If a mapping, at any depth, lacks a key it must hold
                or holds an unknown one (the mapping itself, row, anchor,
                provenance, belief, coordinate), or a coordinate dict is
                invalid; if a row is not a MatchClaim (:meth:`_read`); if the
                mapping carries provenance both under its key ``provenance``
                and in its rows; or if two rows hold the same id.
        """
        given = given_minter(seed, minter)
        keys = ["provenance", "rows"]
        if not isinstance(data, dict) or sorted(data) != keys:
            raise ValueError(
                f"A MatchClaimField dictionary holds the keys {keys}, not "
                f"{sorted(data) if isinstance(data, dict) else data!r}."
            )
        shared = data["provenance"]
        provenance = None if shared is None else Provenance.from_dict(shared)
        rows = copy.deepcopy(data["rows"])
        for index, row in enumerate(rows):
            cls._refuse_other_keys(row, cls.struct_schema(), index)
        cells: dict[tuple[str, ...], pa.StructArray] = {}
        for path in _coordinate_paths(cls.struct_schema()):
            coordinates: list[Coordinate | None] = []
            for index, row in enumerate(rows):
                holder = _holder(row, path)
                if holder is None or holder[path[-1]] is None:
                    coordinates.append(None)
                    continue
                try:
                    coordinate = coordinate_from_dict(holder[path[-1]])
                except ValueError as error:
                    raise ValueError(
                        f"In a MatchClaimField dictionary, row {index} "
                        f"{'.'.join(path)}: {error}"
                    ) from error
                coordinates.append(coordinate)
                holder[path[-1]] = None
            cells[path] = coordinate_cells(coordinates)
        return cls._read(
            _with_children(_struct_array_of(rows, cls.struct_schema()), cells),
            provenance,
            source="dictionary",
            where="under its key 'provenance'",
            minter=given,
        )

    @classmethod
    def _refuse_other_keys(
        cls,
        value: Any,
        data_type: pa.StructType | pa.ListType[Any],
        index: int,
        path: tuple[str, ...] = (),
    ) -> None:
        """Refuse a dictionary mapping whose keys are not those it must hold.

        A coordinate cell must hold the keys of a coordinate dict, any other
        struct its fields; the check descends into every nested struct and
        every item of a list of structs (the beliefs), since Arrow would fill
        a missing key with null and drop an unknown one.

        Args:
            value: The value read (``None`` for a null nested value).
            data_type: The type it is read into.
            index: The row's index, for the message.
            path: The keys leading from the row to the value (none for the
                row itself, which may not be null).

        Raises:
            ValueError: Naming the path, the keys lacking and the unknown keys.
        """
        if value is None and path:
            return
        where = f"row {index}" + (f" {'.'.join(path)}" if path else "")
        if isinstance(data_type, pa.ListType):
            if not isinstance(value, list):
                raise ValueError(
                    f"In a MatchClaimField dictionary, {where} must be a list, "
                    f"not {value!r}."
                )
            for position, item in enumerate(value):
                cls._refuse_other_keys(
                    item, data_type.value_type, index, (*path, str(position))
                )
            return
        coordinate = data_type == COORDINATE_CELL_TYPE
        keys = list(_COORDINATE_DICT_KEYS if coordinate else data_type.names)
        if not isinstance(value, dict):
            raise ValueError(
                f"In a MatchClaimField dictionary, {where} must be a mapping "
                f"holding the keys {keys}, not {value!r}."
            )
        missing = [key for key in keys if key not in value]
        unknown = [key for key in value if key not in keys]
        if missing or unknown:
            raise ValueError(
                f"In a MatchClaimField dictionary, {where} must hold the keys "
                f"{keys}; it lacks {missing} and has the unknown keys {unknown}."
            )
        if coordinate:
            return
        for sub_field in data_type:
            sub_type = sub_field.type
            if isinstance(sub_type, pa.StructType) or (
                isinstance(sub_type, pa.ListType)
                and pa.types.is_struct(sub_type.value_type)
            ):
                cls._refuse_other_keys(
                    value[sub_field.name],
                    sub_type,
                    index,
                    (*path, sub_field.name),
                )

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        *,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Read a field back from its Arrow table (inverse of :attr:`table`).

        The table must hold one column per top-level property of
        :class:`MatchClaim`, named and typed as the fields of
        :attr:`pa_schema`; the columns are read by name, in any order, and
        may be chunked, as they are when the table was read from Parquet.
        Further columns (any other name, such as a ``scope`` column a
        package written by an earlier tool holds) are kept as they stand,
        after the properties, and never read: :attr:`table` and the file
        :meth:`to_parquet` writes hold them again, and the field's own
        queries carry them row for row. The field-level provenance travels
        in the table's schema metadata under TimeToShare's key
        ``timetoshare:provenance`` (the JSON of
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`); a table
        without that key gives a field whose provenance is ``None``. Every
        row is kept as stored; a row without an id is minted one
        (:class:`MatchClaimField`).

        Args:
            table: A table as written by :attr:`table`.
            seed: The seed of the minter of the missing ids.
            minter: The minter of the missing ids (a loader's).

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            TypeError: If both *seed* and *minter* are given.
            ValueError: If the table lacks a column of a ``MatchClaim``
                property, or a column's type is not the property's; if a
                row is not a MatchClaim (:meth:`_read`); if the table
                carries provenance both in its schema metadata and in its
                rows; or if two rows hold the same id (a MatchClaim is
                addressed by its id alone).
        """
        given = given_minter(seed, minter)
        columns = cls.struct_schema().names
        missing = [name for name in columns if name not in table.column_names]
        if missing:
            raise ValueError(
                f"A MatchClaim table holds one column per MatchClaim property, "
                f"{columns}; this table lacks {missing}."
            )
        for sub_field in cls.struct_schema():
            stored = table.schema.field(sub_field.name).type
            if stored != sub_field.type:
                raise ValueError(
                    f"Column {sub_field.name!r} has type {stored}, which is not "
                    f"the type of the MatchClaim property, {sub_field.type}."
                )
        further_names = [name for name in table.column_names if name not in columns]
        further = (
            pa.table(
                {name: table.column(name).combine_chunks() for name in further_names}
            )
            if further_names
            else None
        )
        # Memory holds one spelling of a value (2/4 is held as 1/2), so the
        # cells compare by value wherever they are compared.
        table = _anchors_converted(
            table.select(columns), lambda cells, _name: in_lowest_terms(cells)
        )
        struct_array = pa.StructArray.from_arrays(
            [table.column(name).combine_chunks() for name in columns],
            fields=list(cls.struct_schema()),
        )
        stored = (table.schema.metadata or {}).get(cls._PROVENANCE_KEY)
        provenance = (
            None if stored is None else Provenance.from_dict(json.loads(stored))
        )
        return cls._read(
            struct_array,
            provenance,
            source="table",
            where=f"in its schema metadata ({cls._PROVENANCE_KEY.decode()!r})",
            further=further,
            minter=given,
        )

    @classmethod
    def from_package_table(
        cls,
        table: pa.Table,
        *,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Read a field from a MatchClaim table in the package form.

        The inverse of :attr:`package_table`: every anchor coordinate cell is
        read back from the package form, then the table is read as
        :meth:`from_table` reads one. A cell whose numbers do not stand as
        its ``number_type`` says (an int ``(n, null, null)``, a fraction
        ``(n, d >= 1, null)``, a float ``(null, null, x)``) is refused.

        Args:
            table: A table as :attr:`package_table` gives it, or as read from
                a package's ``match_claims.parquet``.
            seed: The seed of the minter of the missing ids.
            minter: The minter of the missing ids (a loader's).

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            TypeError: If both *seed* and *minter* are given.
            ValueError: As :meth:`from_table`, or naming the column and row
                of a cell not in the package form.
        """
        return cls.from_table(cls._memory_table(table), seed=seed, minter=minter)

    @staticmethod
    def _memory_table(table: pa.Table) -> pa.Table:
        """A package-form MatchClaim table with its anchor cells as memory holds them."""
        return _anchors_converted(
            table,
            lambda cells, name: cells_from_package(
                cells, where=f"MatchClaim table {name}"
            ),
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | os.PathLike[str],
        *,
        timelines: Iterable[Any] | Mapping[str, Any] | None = None,
        seed: int | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Read the MatchClaim table from its Parquet file.

        Args:
            path: The file, or a directory holding :attr:`file_name`
                (``match_claims.parquet``).
            timelines: Where given, the timelines the rows name (as
                :meth:`check_on_timelines` takes them); the field is checked
                against them.
            seed: The seed of the minter of the missing ids.
            minter: The minter of the missing ids (a loader's).

        Returns:
            The field :meth:`to_parquet` wrote, value for value.

        Raises:
            TypeError: If both *seed* and *minter* are given.
            ValueError: As :meth:`from_table`, or as
                :meth:`check_on_timelines`.
        """
        import pyarrow.parquet as pq

        field = cls.from_package_table(
            pq.read_table(cls._file_path(path)), seed=seed, minter=minter
        )
        return field if timelines is None else field.check_on_timelines(timelines)

    @classmethod
    def _file_path(cls, path: str | os.PathLike[str]) -> Path:
        """The file *path* names: itself, or :attr:`file_name` in a directory."""
        path = Path(path)
        return path / cls.file_name if path.is_dir() else path

    @classmethod
    def _read(
        cls,
        struct_array: pa.StructArray,
        provenance: Provenance | None,
        *,
        source: str,
        where: str,
        further: pa.Table | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Admit rows read from a table, a dictionary or columns.

        Refuses a row no :class:`MatchClaim` could be (:meth:`_refuse_malformed`)
        and provenance stated twice (:meth:`_refuse_provenance_twice`), then
        mints the missing ids and admits them (:meth:`_admitted`).

        Args:
            struct_array: The rows, of type :attr:`pa_schema`.
            provenance: The field-level provenance the source states, if any.
            source: What is read, for the messages (``"table"``).
            where: Where the source states the field-level provenance.
            further: The table's further columns, row for row, if any.
            minter: The minter of the missing ids; None for the content
                minter of the rows.
        """
        cls._refuse_malformed(struct_array)
        cls._refuse_provenance_twice(
            provenance, struct_array.field("provenance"), source=source, where=where
        )
        return cls._admitted(
            struct_array, provenance=provenance, further=further, minter=minter
        )

    @staticmethod
    def _refuse_malformed(data: pa.StructArray) -> None:
        """Refuse a row no :class:`MatchClaim` could be, naming the first.

        The rules of the scalar, vectorized, which are the rules of
        TimeToShare's MatchClaim table: both timelines named, and not the
        same one; both event IDs written (an event's, or a sentinel);
        ``is_synchronous`` stated; :data:`CHUNK` only on a synchronous row,
        :data:`NOMATCH` only on one that is not, and never on both sides; a
        sentinel side has no name; a synchronous row holds a start anchor, a
        non-synchronous one no anchor; a stated certainty lies in [0, 1]; an anchor holds both its
        coordinates (:meth:`_refuse_null_coordinates`). Across rows, every
        coordinate of one timeline is in one unit and number type
        (:meth:`_refuse_two_axes`).
        An anchor's coordinates are on the row's timelines, ``coordinate_a``
        on timeline A and ``coordinate_b`` on timeline B.

        Raises:
            ValueError: Naming the first offending row and what it lacks.
        """
        if len(data) == 0:
            return
        synchronous = data.field("is_synchronous")
        start = data.field("start_anchor")
        end = data.field("end_anchor")
        ev_a = _struct_child(data, "event_a_id")
        ev_b = _struct_child(data, "event_b_id")

        def on_a_side(sentinel: str) -> pa.Array:
            """Whether a row names *sentinel* on either side."""
            return pc.or_(
                _compared("equal", ev_a, sentinel), _compared("equal", ev_b, sentinel)
            )

        def named_sentinel(events: pa.Array, names: pa.Array) -> pa.Array:
            """Whether a side is a sentinel holding a name."""
            return pc.and_(
                pc.is_in(events, value_set=pa.array(SENTINELS, pa.string())),
                pc.is_valid(names),
            )

        tl_a = data.field("timeline_a_id")
        tl_b = data.field("timeline_b_id")
        certainty = data.field("certainty")
        checks: tuple[tuple[pa.BooleanArray, str], ...] = (
            (
                pc.or_(pc.is_null(tl_a), pc.is_null(tl_b)),
                "names no timeline_a_id or no timeline_b_id",
            ),
            (pc.equal(tl_a, tl_b), "connects a timeline with itself"),
            (
                pc.or_(pc.is_null(ev_a), pc.is_null(ev_b)),
                "names no event_a_id or no event_b_id (a side without an event "
                f"holds {NOMATCH} or {CHUNK})",
            ),
            (pc.is_null(synchronous), "states no is_synchronous"),
            (
                pc.and_(pc.invert(synchronous), on_a_side(CHUNK)),
                f"is not synchronous and names {CHUNK}, which stands only on a "
                "synchronous MatchClaim",
            ),
            (
                pc.and_(synchronous, on_a_side(NOMATCH)),
                f"is synchronous and names {NOMATCH}, which stands only on a "
                "MatchClaim that is not synchronous",
            ),
            (
                pc.and_(
                    _compared("equal", ev_a, NOMATCH), _compared("equal", ev_b, NOMATCH)
                ),
                f"names {NOMATCH} on both sides; a NOMATCH MatchClaim names an "
                "event on its other side",
            ),
            (
                pc.or_(
                    named_sentinel(ev_a, data.field("event_a_name")),
                    named_sentinel(ev_b, data.field("event_b_name")),
                ),
                "has a name on a sentinel side; a sentinel side has no name",
            ),
            (
                pc.and_(synchronous, start.is_null()),
                "is synchronous and holds no start anchor",
            ),
            (
                pc.and_(
                    pc.invert(synchronous), pc.or_(start.is_valid(), end.is_valid())
                ),
                "is not synchronous and holds an anchor",
            ),
            (
                pc.or_(
                    pc.less(certainty, pa.scalar(0.0)),
                    pc.greater(certainty, pa.scalar(1.0)),
                ),
                "has a certainty outside [0, 1]",
            ),
        )
        for check, reason in checks:
            mask = _filled(check, pa.scalar(False))
            if pc.any(mask).as_py():
                row = pc.index(mask, True).as_py()
                raise ValueError(f"MatchClaim table row {row} {reason}.")
        MatchClaimField._refuse_null_coordinates(data)
        MatchClaimField._refuse_two_axes(data)

    @staticmethod
    def _refuse_null_coordinates(data: pa.StructArray) -> None:
        """Refuse an anchor holding a null coordinate, naming the first.

        An anchor joins two coordinates; a null one (a non-finite number
        written into a cell, say) is TimeToShare's ``invalid_coordinate``,
        and a claim could not be made of the row.

        Raises:
            ValueError: ``MatchClaim table row {i} start_anchor.coordinate_a:
                the coordinate is null.``
        """
        for anchor_name in ("start_anchor", "end_anchor"):
            anchor = _struct_array(data.field(anchor_name))
            for side in ("a", "b"):
                missing = pc.and_(
                    anchor.is_valid(),
                    anchor.field(f"coordinate_{side}").is_null(),
                )
                if pc.any(missing).as_py():
                    row = pc.index(missing, True).as_py()
                    raise ValueError(
                        f"MatchClaim table row {row} {anchor_name}.coordinate_{side}: "
                        f"the coordinate is null."
                    )

    @staticmethod
    def _refuse_two_axes(data: pa.StructArray) -> None:
        """Refuse a table holding one timeline's coordinates on two axes.

        Every coordinate cell names its axis (unit and number type); the
        cells of one timeline, wherever they are (either side of either
        anchor, each on its row's timeline of that side), name one axis, or
        one position could be written two ways. Vectorized: the cells are
        gathered once and grouped by timeline.

        Raises:
            ValueError: Naming the timeline and two of its axes.
        """
        timelines: list[pa.Array] = []
        cells: list[pa.Array] = []
        for anchor_name in ("start_anchor", "end_anchor"):
            # ``flatten`` carries a null anchor over to its coordinates.
            anchor = _struct_members(_struct_array(data.field(anchor_name)))
            names = data.type.field(anchor_name).type.names
            for side in ("a", "b"):
                timelines.append(data.field(f"timeline_{side}_id"))
                cells.append(anchor[names.index(f"coordinate_{side}")])
        cell = _struct_array(pa.concat_arrays(cells))
        axes = pa.table(
            {
                "timeline": pa.concat_arrays(timelines),
                "unit": cell.field("unit"),
                "number_type": cell.field("number_type"),
            }
        ).filter(cell.is_valid())
        distinct = axes.group_by(["timeline", "unit", "number_type"]).aggregate([])
        counts = distinct.group_by("timeline").aggregate([("unit", "count")])
        repeated = counts.filter(pc.greater(counts.column("unit_count"), pa.scalar(1)))
        if repeated.num_rows:
            timeline_id = repeated.column("timeline")[0].as_py()
            stated = distinct.filter(
                pc.equal(distinct.column("timeline"), timeline_id)
            ).to_pylist()
            described = " and ".join(
                f"{axis['unit']} ({axis['number_type']})" for axis in stated[:2]
            )
            raise ValueError(
                f"The MatchClaim table holds coordinates of timeline "
                f"{quote_id(timeline_id)} in {described}: the coordinates of "
                f"one timeline are on its one axis."
            )

    @staticmethod
    def _refuse_provenance_twice(
        provenance: Provenance | None,
        row_provenance: pa.Array,
        *,
        source: str,
        where: str,
    ) -> None:
        """Refuse provenance stated for the whole field and in rows as well.

        A field states a provenance once: for every row, or per row. A
        source stating both would lose one of them on reading.

        Args:
            provenance: The field-level provenance the source states, if any.
            row_provenance: The rows' provenance (null where a row states none).
            source: What is read, for the message (``"table"``).
            where: Where the source states the field-level provenance.

        Raises:
            ValueError: If both are stated.
        """
        if provenance is None:
            return
        with_provenance = len(row_provenance) - row_provenance.null_count
        if with_provenance:
            raise ValueError(
                f"A MatchClaimField {source} states its provenance once: this "
                f"{source} carries it {where} and in {with_provenance} row(s)."
            )

    @classmethod
    def _admitted(
        cls,
        struct_array: pa.StructArray,
        *,
        provenance: Provenance | None,
        taken: pa.Array | None = None,
        holder: str = "field",
        further: pa.Table | None = None,
        minter: IdMinter | None = None,
    ) -> "MatchClaimField":
        """Wrap *struct_array* with its ids minted and admitted.

        A row without an id is minted one (:func:`_minted_claim_ids`); then
        an id already held or held twice is refused
        (:func:`_refuse_held_claim_ids`).

        Args:
            struct_array: The claim rows.
            provenance: The field-level provenance.
            taken: The ids already held (none: the field holds only its
                own).
            holder: What holds the ids, for the message.
            further: The further columns of the rows, if any.
            minter: The minter of the missing ids; None for the content
                minter of the rows.
        """
        taken = pa.array([], type=pa.string()) if taken is None else taken
        struct_array = _minted_claim_ids(struct_array, minter)
        _refuse_held_claim_ids(struct_array.field("id"), taken, holder=holder)
        return cls._from_struct_array(
            struct_array, provenance=provenance, further=further
        )

    @classmethod
    def _from_struct_array(
        cls,
        struct_array: pa.StructArray,
        *,
        provenance: Provenance | None = None,
        further: pa.Table | None = None,
    ) -> "MatchClaimField":
        """Wrap a derived ``MatchClaim`` struct array as a live field.

        The in-memory struct is named as :meth:`SemanticField.from_array`
        names a field by default; :attr:`table` spells each of its fields as
        a column of its own. *further* holds the further columns a table
        read held, row for row (:meth:`from_table`).
        """
        pa_field = pa.field(
            cls.__name__.removesuffix("Field").lower(), cls.struct_schema()
        )
        raw = StructField(struct_array, pa_field)
        field = cls(raw, provenance=provenance)
        field._further = further
        return field

    @classmethod
    def _assemble(cls, n: int, columns: dict[str, pa.Array]) -> pa.StructArray:
        """Assemble the derived ``MatchClaim`` struct array from named columns.

        The children are produced in the exact field order of
        :attr:`pa_schema` (so the resulting type equals ``pa_schema``); a
        column not given is all null.
        """
        arrays = [
            (
                columns[sub_field.name]
                if sub_field.name in columns
                else pa.nulls(n, type=sub_field.type)
            )
            for sub_field in cls.struct_schema()
        ]
        return pa.StructArray.from_arrays(arrays, fields=list(cls.struct_schema()))

    @classmethod
    def _anchor_array(
        cls,
        coordinate_a: pa.StructArray,
        coordinate_b: pa.StructArray,
        *,
        mask: pa.Array | None = None,
    ) -> pa.StructArray:
        """Build one anchor struct from its two coordinate columns (null where *mask*).

        The anchor's timelines are its row's, so the struct holds the two
        coordinates alone.
        """
        anchor_type = cls.struct_schema().field("start_anchor").type
        sources: dict[str, pa.Array] = {
            "coordinate_a": coordinate_a,
            "coordinate_b": coordinate_b,
        }
        return pa.StructArray.from_arrays(
            [sources[field.name] for field in anchor_type],
            fields=list(anchor_type),
            mask=mask,
        )

    @staticmethod
    def _string_array(
        values: Sequence[str | None] | pa.Array | pa.ChunkedArray,
    ) -> pa.Array:
        """Coerce a sequence/array of ids into a plain ``string`` array."""
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if isinstance(values, pa.Array):
            return pc.cast(values, pa.string())
        return pa.array(list(values), type=pa.string())

    @staticmethod
    def _common_provenance(claims: list[MatchClaim]) -> Provenance | None:
        """Return the shared provenance if all claims agree, else ``None``."""
        if not claims:
            return None
        first = claims[0].provenance
        if first is None:
            return None
        for claim in claims[1:]:
            if claim.provenance != first:
                return None
        return first

    def __init__(
        self,
        raw: "StructField | None" = None,
        *,
        source_fields: "str | dict[str, Any] | None" = None,
        name: str | None = None,
        provenance: Provenance | None = None,
    ) -> None:
        """Wrap a derived ``MatchClaim`` struct field, carrying shared provenance.

        Prefer the :meth:`from_columns` / :meth:`from_claims` /
        :meth:`from_dict` / :meth:`from_parquet` classmethods; this
        constructor mirrors the base ``SemanticField`` so blueprint
        construction keeps working, plus a field-level ``provenance``.

        Raises:
            ValueError: If a row of *raw* is null: a MatchClaim table holds
                no null rows.
        """
        super().__init__(raw, source_fields=source_fields, name=name)
        data = self._raw.data
        if data is not None and data.null_count > 0:
            row = pc.index(pc.is_null(data), True).as_py()
            raise ValueError(
                f"MatchClaimField row {row} is null: a MatchClaim table holds no "
                "null rows."
            )
        self._provenance: Provenance | None = provenance
        # The further columns of a table read (:meth:`from_table`), row for
        # row; ``None`` where there are none.
        self._further: pa.Table | None = None

    # ── access ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of claims (rows) in the field."""
        data = self._raw.data
        return 0 if data is None else len(data)

    def __getitem__(self, i: int) -> MatchClaim:
        """Materialise the claim at row ``i`` as a :class:`MatchClaim`.

        Supports negative indexing. The field-level shared provenance, when
        set, is injected into the materialised claim; otherwise the claim
        carries the row's own provenance (``None`` where the row has none).

        Args:
            i: Zero-based row index (negative counts from the end).

        Returns:
            The :class:`MatchClaim` of the row.

        Raises:
            IndexError: If ``i`` is out of range.
        """
        n = len(self)
        if i < 0:
            i += n
        if i < 0 or i >= n:
            raise IndexError(f"MatchClaimField index {i} out of range (len={n})")
        row = self._raw[i]
        claim = MatchClaim.from_row(row)
        if self._provenance is not None:
            claim = claim.model_copy(update={"provenance": self._provenance})
        return claim

    def __iter__(self) -> Iterator[MatchClaim]:
        """Yield each claim, materialised lazily one row at a time."""
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        return (
            f"MatchClaimField(claims={len(self)}, "
            f"timelines={len(self.timeline_ids)})"
        )

    def __eq__(self, other: object) -> bool:
        """Whether two MatchClaim tables hold equal rows, in the same order.

        Every column is compared, provenance and further columns included,
        as each row is read: a provenance the field holds once counts as
        every row's, so a field sharing one provenance equals a field whose
        rows each carry it. The file-level form is not compared.
        """
        if not isinstance(other, MatchClaimField):
            return NotImplemented
        mine, theirs = self._further, other._further
        if (mine is None) != (theirs is None) or (
            mine is not None and theirs is not None and not mine.equals(theirs)
        ):
            return False
        return self._rows_with_provenance().equals(other._rows_with_provenance())

    @property
    def table(self) -> pa.Table:
        """The claims as a :class:`pyarrow.Table`, one column per property.

        The columns are the top-level properties of :class:`MatchClaim`, the
        fields of the derived struct ``MatchClaimField.pa_schema`` in its
        order, each nested as the property is (``start_anchor`` a struct of
        two coordinate structs, ``provenance`` a struct holding its list of
        beliefs). The field-level provenance, when set, travels in the
        schema metadata under ``timetoshare:provenance`` as the JSON of
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`, so the table
        (and the Parquet file written from it) reads back whole through
        :meth:`from_table`. The further columns of a table read follow the
        properties, as read.
        """
        schema_metadata = (
            None
            if self._provenance is None
            else {self._PROVENANCE_KEY: json.dumps(self._provenance.to_dict())}
        )
        table = pa.Table.from_struct_array(self._struct_array())
        if self._further is not None:
            for name in self._further.column_names:
                table = table.append_column(
                    self._further.schema.field(name), self._further.column(name)
                )
        return table.replace_schema_metadata(schema_metadata)

    @property
    def package_table(self) -> pa.Table:
        """The claims as a package writes them: :attr:`table` in the package form.

        The columns and the schema metadata of :attr:`table`, every anchor
        coordinate cell written in the package form
        (:data:`~timetoalign.core.time.PACKAGE_COORDINATE_CELL_TYPE`): its
        number once, in the member its ``number_type`` names, the others
        null, beside its ``unit`` and ``number_type``. It is TimeToShare's
        MatchClaim table, the table :meth:`to_parquet` writes and
        :meth:`from_package_table` reads; in memory the cells keep both sides
        of every number.
        """
        return _anchors_converted(self.table, lambda cells, _name: package_cells(cells))

    @property
    def provenance(self) -> Provenance | None:
        """Shared provenance applied to every materialised claim."""
        return self._provenance

    def _used_prefixes(self) -> list[str]:
        """The sorted prefixes the table's provenance uses, its actors among them.

        Every prefix of a compact IRI at an IRI position of the shared
        provenance and of the rows' (``stated_by``, ``basis``, ``service``,
        a ``method`` that is no method word, each belief's ``actor``), as
        :func:`timetoshare.used_prefixes` finds them. Vectorized: each
        distinct string is read once, no row is materialised.
        """
        entries: list[dict[str, Any]] = []
        if self._provenance is not None:
            entries.append(self._provenance.to_dict())
        data = self._struct_array()
        if len(data):
            # ``flatten`` carries a null parent over to its children.
            provenance = _struct_array(
                _struct_members(data)[
                    self.struct_schema().get_field_index("provenance")
                ]
            )
            members = {
                member.name: column
                for member, column in zip(provenance.type, _struct_members(provenance))
            }
            beliefs = pc.list_flatten(members["beliefs"])
            references = [
                pc.list_flatten(members["stated_by"]),
                pc.list_flatten(members["basis"]),
                members["service"],
                beliefs.flatten()[beliefs.type.get_field_index("actor")],
            ]
            entries.append(
                {
                    "stated_by": sorted(
                        {
                            text
                            for column in references
                            for text in pc.unique(column).to_pylist()
                            if text is not None
                        }
                    )
                }
            )
            entries.extend(
                {"method": method}
                for method in pc.unique(members["method"]).to_pylist()
                if method is not None
            )
        return used_prefixes({"provenance": entries})

    @property
    def timeline_ids(self) -> set[str]:
        """The distinct timeline ids appearing in either id column.

        Reads the struct's two id sub-fields vectorized (no claim is
        materialised), once per table: the ids are kept beside the struct
        array they were read from and read again only when the field holds
        another array. Every call returns a new set.
        """
        data = self._raw.data
        if data is None or len(data) == 0:
            return set()
        cached = self.__dict__.get("_timeline_ids_read")
        if cached is None or cached[0] is not data:
            cached = (data, self._distinct_timeline_ids(_struct_array(data)))
            self._timeline_ids_read = cached
        return set(cached[1])

    @staticmethod
    def _distinct_timeline_ids(data: pa.StructArray) -> tuple[str, ...]:
        """The distinct timeline ids of *data*'s two id columns, A's first.

        Each id appears where it first occurs in the A column followed by
        the B column, so a set built from them iterates as one built row by
        row from both columns.
        """
        both = pa.concat_arrays(
            [data.field("timeline_a_id"), data.field("timeline_b_id")]
        )
        # Without its nulls, the column holds strings only.
        return tuple(
            value
            for value in pc.unique(both.drop_null()).to_pylist()
            if isinstance(value, str)
        )

    @property
    def _ids(self) -> pa.Array:
        """The claim ids, one per row."""
        return self._struct_array().field("id")

    def check_on_timelines(
        self, timelines: Iterable[Any] | Mapping[str, Any]
    ) -> "MatchClaimField":
        """Refuse this field unless every coordinate is on its timeline's axis.

        The vectorized form of :meth:`MatchClaim.check_on_timelines`: every
        coordinate cell (both coordinates of each anchor, each on its row's
        timeline of that side) is compared with the unit and declared number type of the timeline it
        is on; the first offending cell is refused with the scalar check's
        message.

        Args:
            timelines: Every timeline the rows name: timelines, or a mapping
                from a timeline id to a timeline or its ``(unit,
                number_type)`` pair.

        Returns:
            This field.

        Raises:
            ValueError: If a row names a timeline not given, or a coordinate
                is not on its timeline's axis.
        """
        data = self._struct_array()
        if len(data) == 0:
            return self
        axes = _axes(timelines)
        keys = pa.array(list(axes), type=pa.string())
        units = pa.array([axis.unit.value for axis in axes.values()], pa.string())
        number_types = pa.array(
            [axis.number_type.name for axis in axes.values()], pa.string()
        )
        cells_by_place: list[tuple[str, pa.Array, pa.StructArray, pa.Array]] = []
        for anchor_name in ("start_anchor", "end_anchor"):
            anchor = _struct_array(data.field(anchor_name))
            for side in ("a", "b"):
                cells = _struct_array(anchor.field(f"coordinate_{side}"))
                cells_by_place.append(
                    (
                        f"{anchor_name}.coordinate_{side}",
                        data.field(f"timeline_{side}_id"),
                        cells,
                        pc.and_(anchor.is_valid(), cells.is_valid()),
                    )
                )
        for where, timeline_ids, cells, valid in cells_by_place:
            position = pc.index_in(timeline_ids, value_set=keys)
            unknown = pc.and_(valid, pc.is_null(position))
            if pc.any(unknown).as_py():
                row = pc.index(unknown, True).as_py()
                raise ValueError(
                    f"Row {row} {where} is on timeline "
                    f"{quote_id(timeline_ids[row].as_py())}, which is not among "
                    f"the timelines given."
                )
            off = pc.and_(
                valid,
                pc.or_(
                    pc.not_equal(cells.field("unit"), pc.take(units, position)),
                    pc.not_equal(
                        cells.field("number_type"), pc.take(number_types, position)
                    ),
                ),
            )
            off = _filled(off, pa.scalar(False))
            if pc.any(off).as_py():
                row = pc.index(off, True).as_py()
                timeline_id = timeline_ids[row].as_py()
                # An off-axis cell is a valid one.
                cell = _valid_struct_cell(cells[row])
                _refuse_off_axis(
                    TimeUnit(cell["unit"]),
                    NumberType(cell["number_type"]),
                    axes[timeline_id],
                    timeline_id=timeline_id,
                    where=f"Row {row} {where}",
                )
        return self

    def connecting(self, timeline_id: str) -> "MatchClaimField":
        """Return the claims that involve ``timeline_id`` on either side.

        The filter is a vectorized boolean mask over the struct's id
        sub-fields; no claim is materialised. The shared provenance carries
        over.

        Args:
            timeline_id: The exact timeline id to match against either column.

        Returns:
            A new :class:`MatchClaimField` holding only the matching rows.
        """
        mask = self._involves_mask(timeline_id)
        return self._filtered(mask)

    def at(self, timeline_id: str, coordinate: CoordinateSpec) -> "MatchClaimField":
        """Return claims relevant at ``(timeline_id, coordinate)``.

        The selection is a vectorized PyArrow boolean mask over the struct
        column. Instant rows compare their start coordinate exactly. Interval
        rows use closed containment between their start and end coordinates.
        Rows without anchors (NOMATCH, conceptual) are never relevant. No
        claim is materialised and the rest of the field is never touched;
        the shared provenance carries over.

        Coordinate matching is exact in the query coordinate's declared
        number representation. There is no tolerance or nearest-value
        fallback.

        Args:
            timeline_id: The timeline the queried coordinate lives on.
            coordinate: A raw or typed coordinate on that timeline.

        Returns:
            A new :class:`MatchClaimField` holding only the matching rows
            (possibly empty).
        """
        data = self._struct_array()
        if len(data) == 0:
            return self._empty()
        if isinstance(coordinate, IdCoordinate):
            if coordinate.timeline_id != timeline_id:
                raise ValueError(
                    f"coordinate timeline_id {coordinate.timeline_id!r} does not "
                    f"match {timeline_id!r}"
                )
            query = coordinate.to_coordinate()
        elif isinstance(coordinate, Coordinate):
            query = coordinate
        elif not isinstance(coordinate, bool) and isinstance(
            coordinate, (int, float, Fraction)
        ):
            number_type = (
                NumberType.fraction
                if isinstance(coordinate, Fraction)
                else (
                    NumberType.float
                    if isinstance(coordinate, float)
                    else NumberType.int
                )
            )
            query = Coordinate(coordinate, TimeUnit.number, number_type=number_type)
        else:
            raise TypeError(
                "MatchClaimField.at coordinate must be int, float, Fraction, "
                "Coordinate, or IdCoordinate"
            )
        masks: list[pa.Array] = []
        for side in ("a", "b"):
            side_id = _compared(
                "equal", _struct_child(data, f"timeline_{side}_id"), timeline_id
            )
            start_cells = _struct_array(data.field("start_anchor")).field(
                f"coordinate_{side}"
            )
            end_cells = _struct_array(data.field("end_anchor")).field(
                f"coordinate_{side}"
            )
            start = compare_number_column(
                start_cells, query.value, "equal", query.number_type
            )
            contained = pc.and_(
                compare_number_column(
                    start_cells, query.value, "less_equal", query.number_type
                ),
                compare_number_column(
                    end_cells, query.value, "greater_equal", query.number_type
                ),
            )
            relevant = _if_else(pc.is_valid(data.field("end_anchor")), contained, start)
            # A side whose cells cannot answer the query (null) is not relevant.
            masks.append(
                _filled(
                    pc.and_(
                        pc.and_(side_id, data.field("start_anchor").is_valid()),
                        relevant,
                    ),
                    pa.scalar(False),
                )
            )
        mask = pc.or_(*masks)
        # Empty-mask short-circuit: skip building a filtered sub-field.
        if not pc.any(mask).as_py():
            return self._empty()
        return self._filtered(mask)

    def filter(
        self,
        *,
        timeline_id: str | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        between: tuple[str, str] | None = None,
        within: set[str] | None = None,
        synchronous_only: bool = False,
        nomatch_only: bool = False,
        believer: str | None = None,
        beliefs: Collection[BeliefState] = DEFAULT_BELIEFS,
        context: Mapping[str, Any] | None = None,
    ) -> "MatchClaimField":
        """Return a filtered view following the unified filter API.

        Every criterion is a vectorized boolean mask over the claim columns;
        no claim is materialised and the shared provenance carries over. All
        non-default criteria are combined with logical AND.

        ``include_domains`` / ``include_units`` are deliberately absent: they
        need per-timeline metadata (domain, unit) that this store does not
        hold. A caller that owns that metadata — an
        :class:`~timetoalign.alignment.bundle.AlignmentBundle`, say — resolves
        them into the set of timeline ids that pass and hands that set to
        ``within``.

        Args:
            timeline_id: Keep rows involving this exact id on either side
                (equivalent to :meth:`connecting`).
            timeline_ids: Keep rows involving **any** id in this set on either
                side.
            id_pattern: Keep rows involving any id matching this regex
                (``re.search`` against the field's :attr:`timeline_ids`).
            between: Keep rows connecting exactly these two ids
                (order-independent).
            within: Keep rows whose **both** ids are in this set. This is the
                primitive behind domain- and unit-restricted queries.
            synchronous_only: Keep the synchronous rows (those with anchors).
            nomatch_only: Keep the NOMATCH rows (not synchronous,
                :data:`NOMATCH` on one side).
            believer: The actor whose beliefs filter the rows (the belief
                filter, :func:`~timetoalign.core.provenance.belief_mask`);
                ``None`` filters nothing.
            beliefs: The states kept: a row passes when the believer's
                belief in it (``"true"``, ``"unlikely"``, ``"false"``), or
                ``"undecided"`` where the believer holds none, is one of
                them. Default: every state but ``"false"``.
            context: The JSON-LD context under which the believer and the
                rows' actors compare as IRIs (a bundle passes its manifest's
                ``@context``); ``None`` declares no prefix.

        Returns:
            A new :class:`MatchClaimField`. With no criteria set, a copy of
            the whole field is returned.

        Raises:
            ValueError: If ``synchronous_only`` and ``nomatch_only`` are both
                set, or ``beliefs`` is not a collection of belief states.
        """
        if synchronous_only and nomatch_only:
            raise ValueError("synchronous_only and nomatch_only are mutually exclusive")

        data = self._struct_array()
        masks: list[pa.Array] = []
        if timeline_id is not None:
            masks.append(self._involves_mask(timeline_id))
        if timeline_ids is not None:
            masks.append(self._involves_any_mask(timeline_ids))
        if id_pattern is not None:
            pattern = re.compile(id_pattern)
            masks.append(
                self._involves_any_mask(
                    {tid for tid in self.timeline_ids if pattern.search(tid)}
                )
            )
        if between is not None:
            masks.append(self._between_mask(*between))
        if within is not None:
            masks.append(self._within_mask(within))
        if synchronous_only:
            masks.append(data.field("is_synchronous").fill_null(False))
        if nomatch_only:
            masks.append(self._nomatch_mask())
        if believer is not None:
            masks.append(self._belief_mask(believer, beliefs, context))
        else:
            check_belief_filter(beliefs)

        if not masks:
            return self._from_struct_array(
                data, provenance=self._provenance, further=self._further
            )
        mask = masks[0]
        for extra in masks[1:]:
            mask = pc.and_(mask, extra)
        return self._filtered(mask)

    def connects_groups(self, group_a_ids: set[str], group_b_ids: set[str]) -> bool:
        """Whether any synchronous claim spans the two given sets of timeline ids.

        True when at least one row with anchors has one side in
        ``group_a_ids`` and the other side in ``group_b_ids``; a NOMATCH or
        conceptual row states no temporal equivalence and connects nothing.
        The test is a vectorized mask reduced with :func:`pyarrow.compute.any`;
        no claim is materialised, so it answers commensurability over a
        million-row field in one pass.

        Args:
            group_a_ids: Timeline ids on one side.
            group_b_ids: Timeline ids on the other side.

        Returns:
            True if some claim connects the two sets.
        """
        data = self._struct_array()
        if len(data) == 0:
            return False
        column_a = data.field("timeline_a_id")
        column_b = data.field("timeline_b_id")
        set_a = self._id_value_set(group_a_ids)
        set_b = self._id_value_set(group_b_ids)
        mask = pc.or_(
            pc.and_(
                pc.is_in(column_a, value_set=set_a),
                pc.is_in(column_b, value_set=set_b),
            ),
            pc.and_(
                pc.is_in(column_a, value_set=set_b),
                pc.is_in(column_b, value_set=set_a),
            ),
        )
        mask = pc.and_(mask, data.field("start_anchor").is_valid())
        return pc.any(mask).as_py() is True

    def to_claims(self) -> list[MatchClaim]:
        """Materialise every row into a list of :class:`MatchClaim` objects.

        This is ``O(n)`` and defeats the columnar purpose for large sets; it
        is provided for convenience and round-tripping, not for hot paths.

        Returns:
            A list of claims, one per row.
        """
        return [self[i] for i in range(len(self))]

    def to_dict(self) -> dict[str, Any]:
        """Render the :attr:`table` as a plain dictionary (inverse of :meth:`from_dict`).

        ``{"provenance": ..., "rows": [...]}``: the field's provenance
        (:meth:`~timetoalign.core.provenance.Provenance.to_dict`, or
        ``None``) and one mapping per claim whose keys are the table's
        columns. A coordinate cell is written as a coordinate dict
        (:func:`~timetoalign.core.retrieval.coordinate_to_dict`: ``unit``,
        ``number_type``, ``value``, a Fraction as ``[numerator,
        denominator]``), the value the materialised claim holds, as
        :meth:`MatchClaim.to_dict` writes it; every other cell as stored, a
        nested property a nested mapping. Every value is a string, number,
        boolean, ``None``, mapping or list, so the dictionary can be written
        as JSON or YAML.

        Returns:
            A round-trippable mapping.

        Examples:
            >>> field = MatchClaimField.from_columns(
            ...     ["score"], ["audio"], ["n1"], ["o1"], [Fraction(1, 3)], [0.5],
            ...     timelines={
            ...         "score": ("quarters", "fraction"),
            ...         "audio": ("seconds", "float"),
            ...     },
            ...     seed=0,
            ... )
            >>> row = field.to_dict()["rows"][0]
            >>> row["start_anchor"]["coordinate_a"]
            {'unit': 'quarters', 'number_type': 'fraction', 'value': [1, 3]}
            >>> row["start_anchor"]["coordinate_b"]
            {'unit': 'seconds', 'number_type': 'float', 'value': 0.5}
            >>> row["id"], row["end_anchor"]
            ('claim_1be6k8ij6dzad', None)
        """
        rows = _struct_rows(self._struct_array())
        paths = list(_coordinate_paths(self.struct_schema()))
        for row in rows:
            for path in paths:
                holder = _holder(row, path)
                if holder is None:
                    continue
                cell: dict[str, Any] | None = holder[path[-1]]
                if cell is not None:
                    holder[path[-1]] = _coordinate_to_dict(cell_coordinate(cell))
        return {
            "provenance": (
                self._provenance.to_dict() if self._provenance is not None else None
            ),
            "rows": rows,
        }

    def to_parquet(self, path: str | os.PathLike[str]) -> Path:
        """Write the MatchClaim table to its Parquet file.

        The file holds :attr:`package_table` (each anchor coordinate's
        number written once, by its number type), the field-level
        provenance in its schema metadata, so :meth:`from_parquet` reads
        back every value exactly; further columns of a table read follow
        the properties (:meth:`from_table`). Where the file exists, the new
        table must succeed it: every MatchClaim of the existing table, by
        its id, is in this one, with the same timelines, events, anchors and
        every other value; only its provenance may differ (a MatchClaim is
        never removed and never changed, only believed or disbelieved). The
        file is replaced whole, never in part: a write that fails leaves the
        file as it was and no temporary file.

        Every MatchClaim written holds an id, which a named MatchStamp may
        record: a field holds an id in every row from its construction.

        Args:
            path: The file, or a directory to hold :attr:`file_name`
                (``match_claims.parquet``).

        Returns:
            The path of the file written.

        Raises:
            ValueError: If a row holds no id, or the existing table holds a
                MatchClaim this one lacks or changes.
        """
        import pyarrow.parquet as pq

        missing = pc.is_null(self._ids)
        if pc.any(missing).as_py():
            row = pc.index(missing, True).as_py()
            raise ValueError(
                f"MatchClaim table row {row} holds no id, and every MatchClaim "
                f"of the file holds one."
            )
        table = self.package_table
        target = self._file_path(path)
        if target.exists():
            self._refuse_unless_succeeding(type(self).from_parquet(target))
        temporary = target.with_name(f".{target.name}.writing")
        try:
            pq.write_table(table, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def _refuse_unless_succeeding(self, existing: "MatchClaimField") -> None:
        """Refuse this table as a successor of *existing* unless it keeps it.

        Every MatchClaim of *existing*, by its id, must be here, its row
        equal in every column of the MatchClaim table but ``provenance``;
        further columns are not compared. Vectorized: the rows are matched
        by id and compared leaf column by leaf column.

        Args:
            existing: The table written before.

        Raises:
            ValueError: Naming the first MatchClaim missing or changed, and
                for a change the column that differs.
        """
        old_ids = existing._ids
        if len(old_ids) == 0:
            return
        positions = pc.index_in(old_ids, value_set=self._ids)
        if positions.null_count:
            row = pc.index(pc.is_null(positions), True).as_py()
            raise ValueError(
                f"The new MatchClaim table lacks the MatchClaim "
                f"{quote_id(old_ids[row].as_py())} of the existing one: a "
                f"MatchClaim is never removed; record the belief 'false' in it "
                f"instead (AlignmentBundle.add_match_claim_belief)."
            )

        def compared(field: "MatchClaimField") -> pa.Table:
            """Return *field*'s MatchClaim rows as a table without their provenance."""
            rows = pa.Table.from_struct_array(field._struct_array())
            return rows.drop_columns(["provenance"])

        old = _leaf_columns(compared(existing))
        new = _leaf_columns(compared(self).take(positions))
        for name in old.column_names:
            before = old.column(name).combine_chunks()
            after = new.column(name).combine_chunks()
            same = pc.or_(
                _filled(_compared("equal", before, after), pa.scalar(False)),
                pc.and_(pc.is_null(before), pc.is_null(after)),
            )
            if not pc.all(same).as_py():
                row = pc.index(same, False).as_py()
                raise ValueError(
                    f"The new MatchClaim table changes the MatchClaim "
                    f"{quote_id(old_ids[row].as_py())} of the existing one in "
                    f"{name!r}: only a MatchClaim's provenance may change; a "
                    f"correction is a new MatchClaim under a new id."
                )

    def with_belief(
        self,
        claim_id: str,
        belief: Belief,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> "MatchClaimField":
        """A copy of the table whose MatchClaim *claim_id* holds *belief*.

        Only that row's provenance changes, as :meth:`MatchClaim.with_belief`
        changes a claim's; every other row, and every other column, is kept.
        A provenance the field holds once for every row stays shared where
        the field has one row; otherwise every row then carries its own,
        since one row's differs.

        Args:
            claim_id: The MatchClaim's id, held by one row.
            belief: The belief.
            context: The JSON-LD context under which actors compare as IRIs;
                ``None`` declares no prefix.

        Returns:
            The new field; this one is left as it is.

        Raises:
            KeyError: If no row holds *claim_id*.
        """
        held = pc.indices_nonzero(_compared("equal", self._ids, claim_id))
        # An index is never null.
        rows = [row for row in held.to_pylist() if row is not None]
        if not rows:
            raise KeyError(
                f"The MatchClaim table holds no MatchClaim {quote_id(claim_id)} "
                f"({len(self)} MatchClaims)."
            )
        return self._with_belief_at(rows[0], belief, context)

    def _with_belief_at(
        self, row: int, belief: Belief, context: Mapping[str, Any] | None
    ) -> "MatchClaimField":
        """A copy of the table whose row *row* holds *belief* (:meth:`with_belief`)."""
        if self._provenance is not None and len(self) == 1:
            return self._from_struct_array(
                self._struct_array(),
                provenance=_believed_entry(self._provenance, belief, context),
                further=self._further,
            )
        data = self._rows_with_provenance()
        index = self.struct_schema().get_field_index("provenance")
        children = data.flatten()
        column = children[index]
        earlier = column[row].as_py()
        entry = _believed_entry(
            None if earlier is None else Provenance.from_row(earlier), belief, context
        )
        children[index] = pa.concat_arrays(
            [
                column.slice(0, row),
                provenance_array([entry]).cast(column.type),
                column.slice(row + 1),
            ]
        )
        rows = pa.StructArray.from_arrays(children, fields=list(self.struct_schema()))
        return self._from_struct_array(rows, provenance=None, further=self._further)

    def _rows_with_provenance(self) -> pa.StructArray:
        """The rows' struct, each row carrying the provenance it is read with.

        A field's shared provenance is written into every row's
        ``provenance`` cell, so the rows can join rows of another provenance
        and compare with them.
        """
        data = self._struct_array()
        if self._provenance is None:
            return data
        schema = self.struct_schema()
        children = data.flatten()
        children[schema.get_field_index("provenance")] = provenance_array(
            [self._provenance]
        ).take(np.zeros(len(data), dtype=np.int64))
        return pa.StructArray.from_arrays(children, fields=list(schema))

    def _concatenated(self, other: "MatchClaimField") -> "MatchClaimField":
        """The rows of this table, then those of *other*, in one table.

        A provenance both share stays shared; otherwise every row carries
        its own. Further columns are joined by name, null in the rows of a
        table without them. The ids are neither checked nor changed here.

        Raises:
            ValueError: If the two tables hold a further column of one name
                in two types.
        """
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        if self._provenance == other._provenance:
            provenance = self._provenance
            parts = [self._struct_array(), other._struct_array()]
        else:
            provenance = None
            parts = [self._rows_with_provenance(), other._rows_with_provenance()]
        return self._from_struct_array(
            pa.concat_arrays(parts),
            provenance=provenance,
            further=self._further_concatenated(other),
        )

    def _further_concatenated(self, other: "MatchClaimField") -> pa.Table | None:
        """The further columns of this table's rows, then of *other*'s."""
        if self._further is None and other._further is None:
            return None
        names: dict[str, pa.DataType] = {}
        for further in (self._further, other._further):
            if further is None:
                continue
            for column in further.schema:
                held = names.setdefault(column.name, column.type)
                if held != column.type:
                    raise ValueError(
                        f"The further column {column.name!r} of the MatchClaim "
                        f"tables joined has the types {held} and {column.type}."
                    )

        def full(further: pa.Table | None, n: int) -> list[pa.Array]:
            """Return one array per joined further column, *n* nulls where *further* lacks it."""
            return [
                (
                    further.column(name).combine_chunks()
                    if further is not None and name in further.column_names
                    else _null_column(n, kind)
                )
                for name, kind in names.items()
            ]

        columns = zip(
            full(self._further, len(self)),
            full(other._further, len(other)),
            strict=True,
        )
        return pa.table(
            {
                name: pa.concat_arrays(list(pair))
                for name, pair in zip(names, columns, strict=True)
            }
        )

    def _joined(self, taken: pa.Array, *, holder: str) -> "MatchClaimField":
        """Return this field as it joins a bundle holding the ids *taken*.

        Joining mints nothing: an id the bundle holds, or one held twice, is
        refused (an id never changes). The field itself is left as it is.
        """
        return self._admitted(
            self._struct_array(),
            provenance=self._provenance,
            taken=taken,
            holder=holder,
            further=self._further,
        )

    def _struct_array(self) -> pa.StructArray:
        """The claim rows, an empty array of the struct type when none."""
        data = self._raw.data
        return (
            _struct_array_of([], self.struct_schema())
            if data is None
            else _struct_array(data)
        )

    def _empty(self) -> "MatchClaimField":
        """Return a zero-row field of this schema, keeping the provenance."""
        return self._from_struct_array(
            self._struct_array().slice(0, 0),
            provenance=self._provenance,
            further=None if self._further is None else self._further.slice(0, 0),
        )

    def _filtered(self, mask: pa.Array) -> "MatchClaimField":
        """Return a new field with the struct array filtered by ``mask``."""
        filtered = self._struct_array().filter(mask)
        return self._from_struct_array(
            filtered,
            provenance=self._provenance,
            further=None if self._further is None else self._further.filter(mask),
        )

    def _nomatch_mask(self) -> pa.Array:
        """Boolean mask: the NOMATCH rows (not synchronous, :data:`NOMATCH` on a side)."""
        data = self._struct_array()
        return _filled(
            pc.and_(
                pc.invert(data.field("is_synchronous")),
                pc.or_(
                    _compared("equal", _struct_child(data, "event_a_id"), NOMATCH),
                    _compared("equal", _struct_child(data, "event_b_id"), NOMATCH),
                ),
            ),
            pa.scalar(False),
        )

    def _belief_mask(
        self,
        believer: str | None,
        beliefs: Collection[BeliefState],
        context: Mapping[str, Any] | None = None,
    ) -> pa.Array:
        """Boolean mask: the rows the belief filter keeps for *believer*.

        The shared provenance, where the field holds one, decides for every
        row; otherwise each row's own (:func:`belief_mask` either way, actors
        compared as IRIs under *context*).
        """
        data = self._struct_array()
        if self._provenance is not None:
            keep = self._provenance.passes_belief_filter(
                believer, beliefs, context=context
            )
            return pa.array(np.full(len(data), keep), type=pa.bool_())
        return belief_mask(data.field("provenance"), believer, beliefs, context=context)

    @staticmethod
    def _id_value_set(timeline_ids: set[str]) -> pa.Array:
        """A deterministic ``string`` value set for :func:`pyarrow.compute.is_in`."""
        return pa.array(sorted(timeline_ids), type=pa.string())

    def _involves_mask(self, timeline_id: str) -> pa.Array:
        """Boolean mask: rows where either id sub-field equals ``timeline_id``."""
        data = self._struct_array()
        return pc.or_(
            _compared("equal", _struct_child(data, "timeline_a_id"), timeline_id),
            _compared("equal", _struct_child(data, "timeline_b_id"), timeline_id),
        )

    def _involves_any_mask(self, timeline_ids: set[str]) -> pa.Array:
        """Boolean mask: rows involving any id in ``timeline_ids``."""
        data = self._struct_array()
        wanted = self._id_value_set(timeline_ids)
        return pc.or_(
            pc.is_in(data.field("timeline_a_id"), value_set=wanted),
            pc.is_in(data.field("timeline_b_id"), value_set=wanted),
        )

    def _within_mask(self, timeline_ids: set[str]) -> pa.Array:
        """Boolean mask: rows whose BOTH id sub-fields are in ``timeline_ids``."""
        data = self._struct_array()
        wanted = self._id_value_set(timeline_ids)
        return pc.and_(
            pc.is_in(data.field("timeline_a_id"), value_set=wanted),
            pc.is_in(data.field("timeline_b_id"), value_set=wanted),
        )

    def _between_mask(self, timeline_a_id: str, timeline_b_id: str) -> pa.Array:
        """Boolean mask: rows connecting exactly the two ids, either way round."""
        data = self._struct_array()
        column_a = data.field("timeline_a_id")
        column_b = data.field("timeline_b_id")
        return pc.or_(
            pc.and_(
                _compared("equal", column_a, timeline_a_id),
                _compared("equal", column_b, timeline_b_id),
            ),
            pc.and_(
                _compared("equal", column_a, timeline_b_id),
                _compared("equal", column_b, timeline_a_id),
            ),
        )

    def _repr_html_(self) -> str:
        """Compact Jupyter summary: claim count, timeline count, head sample."""
        import html as html_mod

        n = len(self)
        n_timelines = len(self.timeline_ids)
        head = min(n, 5)
        rows = []
        for i in range(head):
            claim = self[i]
            anchor = claim.start_anchor
            coordinate_a = (
                "—" if anchor is None else _format_coordinate(anchor.coordinate_a)
            )
            coordinate_b = (
                "—" if anchor is None else _format_coordinate(anchor.coordinate_b)
            )
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{html_mod.escape(claim.timeline_a_id)}</td>"
                f"<td>{html_mod.escape(coordinate_a)}</td>"
                f"<td>{html_mod.escape(claim.timeline_b_id)}</td>"
                f"<td>{html_mod.escape(coordinate_b)}</td></tr>"
            )
        if n > head:
            rows.append("<tr><td colspan='5'>&hellip;</td></tr>")
        header = (
            "<tr><th>#</th><th>timeline A</th><th>coord A</th>"
            "<th>timeline B</th><th>coord B</th></tr>"
        )
        return (
            f"<div style='font-family: monospace;'>"
            f"<strong>MatchClaimField</strong> "
            f"<span style='color: #555;'>claims={n}, timelines={n_timelines}</span>"
            f"<table style='border-collapse: collapse; margin-top: 4px;'>"
            f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody>"
            f"</table></div>"
        )


# endregion


# region Claims between events

#: A side of a claimed pair: the id of an event its timeline holds, an event
#: dict (naming a held event by its ``id``, or stating a position), or
#: :data:`NOMATCH`, the NOMATCH sentinel.
ClaimSide = str | dict[str, Any]


def _held_events(timeline: Any, ids: list[str]) -> list[dict[str, Any] | None]:
    """The events of *timeline* (its descendants included) named by *ids*.

    One lookup per call: the timeline's events, its descendants' at their
    positions on its axis, are read once and the ids found in the id column
    (:func:`pyarrow.compute.index_in`). An event of a descendant counts as an
    event of the timeline, at the timeline's coordinate. An event ID is
    unique on its own timeline only: the timeline's own event answers for
    its ID, whatever its descendants hold; an ID the timeline does not hold
    and two of its descendants do names no one event and is refused, never
    resolved to the first that holds it.

    Returns:
        Per id its event row, ``None`` where the timeline holds none.

    Raises:
        ValueError: If the timeline does not hold one of *ids* itself and
            two events of its descendants do, naming the timelines holding
            it.
    """
    if not ids:
        return []
    table = timeline.get_events(include_children=True).table
    if table.num_rows == 0 or "id" not in table.column_names:
        return [None] * len(ids)
    wanted = pa.array(ids, type=pa.string())
    held = pc.cast(table.column("id").combine_chunks(), pa.string())
    # A descendant's row names the timeline holding it; the timeline's own
    # rows name none.
    holders = (
        table.column("source_timeline").combine_chunks()
        if "source_timeline" in table.column_names
        else pa.nulls(table.num_rows, pa.string())
    )
    own = holders.is_null()
    # The rows that may answer: the timeline's own, and a descendant's whose
    # ID the timeline does not hold itself.
    answering = pc.and_(
        pc.is_in(held, value_set=wanted),
        pc.or_(own, pc.invert(pc.is_in(held, value_set=held.filter(own)))),
    )
    counts = pc.value_counts(held.filter(answering))
    repeated = counts.filter(_compared("greater", _struct_child(counts, "counts"), 1))
    if len(repeated):
        event_id = repeated.field("values")[0].as_py()
        spelled = ", ".join(
            quote_id(str(holder))
            for holder in holders.filter(pc.equal(held, event_id)).to_pylist()
        )
        raise ValueError(
            f"The event ID {quote_id(event_id)} is held by "
            f"{repeated.field('counts')[0].as_py()} events of descendants of "
            f"timeline {quote_id(timeline.id)} (on {spelled}) and not by the "
            f"timeline itself, so it names no one event: name the timeline "
            f"holding the event instead."
        )
    table = table.filter(answering)
    if table.num_rows == 0:
        return [None] * len(ids)
    positions = pc.index_in(wanted, value_set=held.filter(answering))
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
    NOMATCH sentinel.
    """

    timeline: Any
    event: dict[str, Any] | None
    projection: Projection | None
    name: str | None
    chunk: tuple[Coordinate, Coordinate | None] | None = None


def _claims_between_events(
    pairs: Sequence[tuple[ClaimSide, Any, ClaimSide, Any]],
    *,
    synchronous: bool = True,
    certainty: float = 1.0,
    provenance: Provenance | None = None,
    earlier: MatchClaimField | None = None,
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
      (the claim's name of the event). A **Projection** stands for the
      side there (:meth:`Timeline.add_projections`, get-or-add): under the
      dict's ``id`` (an event the timeline does not hold, ``projection_of``
      null), or, without one, under the id of a Projection the timeline
      already holds for the same event at the same position, else a
      minted ``projection_…``; it names the other side's event in
      ``projection_of`` where the other side is a held event (null where
      that side is a position or :data:`NOMATCH`). Except where both sides
      of a synchronous pair are such dicts without ``id``: no Projection
      stands there, and both sides are :data:`CHUNK`, the claim's anchor
      holding their positions (a CHUNK side states no ``name``);
    - :data:`NOMATCH`, the NOMATCH sentinel, on one side at most.

    A synchronous pair gives :meth:`MatchClaim.from_events` between the two
    sides, an interval claim where both have an end; a pair with
    :data:`NOMATCH` gives :meth:`MatchClaim.nomatch` for the other side's
    event. A second call with the same pairs adds no Projection: it reuses
    every one.

    A side's ``name`` is the claim's name of the Projection: the name the
    first claim on a Projection recorded (in *earlier*, else in this call)
    is the one every later claim on it shows.

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

    Returns:
        The claims, in pair order, each holding the id minted at its
        construction (:class:`MatchClaim`).

    Raises:
        KeyError: If an id names no event of its timeline.
        ValueError: If both sides of a pair are :data:`NOMATCH`, a side is
            :data:`CHUNK` or a dict whose ``id`` is :data:`NOMATCH` or
            :data:`CHUNK` (a CHUNK side is a position: a dict without
            ``id``; a NOMATCH side is :data:`NOMATCH`), a CHUNK side states
            a name, a pair with CHUNK on both
            sides states an end on one side only, a dict naming no held
            event states no ``start``, a position is a float on a
            samples axis, a span runs backwards, a position lies past where
            its timeline can reach (the end of a locked timeline, or an end
            its parent cannot contain), a held event is named with a span
            not its own, a position cannot be placed on its timeline's
            axis, or a float start, end or duration is one its timeline
            cannot store.
        TypeError: If a side is neither a string nor a dict.
    """
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
    projected: dict[int, tuple[Any, list[Projection]]] = {}
    for sides in resolved:
        for side in sides:
            if side.projection is not None:
                projected.setdefault(id(side.timeline), (side.timeline, []))[1].append(
                    side.projection
                )
    for _, projections in projected.values():
        _refuse_unstorable(_projection_values(projections))
    projection_ids = {
        key: iter(timeline.add_projections(projections))
        for key, (timeline, projections) in projected.items()
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
            can reach, a held event is named with a span not its own, a
            position cannot be placed, or a CHUNK side states a name.
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
        if side.get("name") is not None:
            raise ValueError(
                f"{where} states the name {side['name']!r}, but it is {CHUNK}: "
                f"the pair states two positions where neither timeline holds an "
                f"event, and a sentinel side has no name."
            )
        return _Side(timeline, None, None, None, (start, end))
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
    name = side.get("name")
    return _Side(timeline, None, projection, None if name is None else str(name))


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
    CHUNK side states its positions without ``id`` or name.
    """
    if side.chunk is not None:
        start, end = side.chunk
        return {"start": start, "end": end}
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


# endregion
