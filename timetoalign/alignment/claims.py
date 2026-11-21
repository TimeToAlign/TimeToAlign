"""AlignmentAnchor and MatchClaim scalars and the paired MatchClaimField.

This module implements the atomic and low-level alignment claims from the
multi-level alignment hierarchy:

- AlignmentAnchor: A coordinate pair, one coordinate on each of two timelines.
- MatchClaim: A claim that two events on different timelines correspond, or
  that one event has no counterpart on the other (NOMATCH), with its
  provenance (:class:`~timetoalign.core.provenance.Provenance`) and certainty.
- MatchClaimField: The MatchClaim table, a ``SemanticField[MatchClaim]``
  holding every kind of MatchClaim as columns, read from and written to
  ``match_claims.parquet``.

The hierarchy is:
    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine

Design:
    AlignmentAnchor and MatchClaim are frozen pydantic v2 ``BaseModel``
    scalars. A MatchClaim always knows its two timelines via top-level
    fields and names two events (or one event and the NOMATCH sentinel,
    ``None``), never a bare coordinate: a side without an event of its own
    is given a Projection event first. Only synchronous MatchClaims produce
    AlignmentAnchors, and every coordinate a claim holds is on its
    timeline's own axis, in the timeline's unit and declared number type
    (:meth:`MatchClaim.check_on_timelines`); a position given in another
    unit is converted through the timeline's C-Map before the claim exists.

    ``MatchClaim`` is paired with ``MatchClaimField``, the MatchClaim table:
    one row per claim, one column per ``MatchClaim`` property. A provenance
    shared by every row is held once, at field level, and injected on read.
    A claim's id never changes once given; a scope holding it refuses a
    second claim under the same id.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

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

from timetoalign.core import (
    ClaimType,
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    IdCoordinate,
    Interval,
    NumberType,
    TimeUnit,
)
from timetoalign.core.fields import (
    SemanticField,
    StructField,
    register_value_projector,
)
from timetoalign.core.ids import mint_id, mint_ids, quote_id, validate_id
from timetoalign.core.provenance import (
    BeliefFilter,
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
from timetoalign.core.time import compare_number_column, struct_to_rational
from timetoalign.core.timestamp import _format_coordinate

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.graph import MatchStamp
    from timetoalign.core.timestamp import ConversionMapsSpec

module_logger = logging.getLogger(__name__)

#: The prefix of a minted claim id (``claim_1``).
_CLAIM_ID_PREFIX = "claim"


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
        assert isinstance(converted, Coordinate)
        return Coordinate(
            converted.value, converted.unit, number_type=converted.number_type
        )
    return Coordinate(value, timeline.unit, number_type=timeline.number_type)


# endregion


# region Claim ids


def _admitted_claim_ids(
    ids: pa.Array, minted: pa.Array, taken: pa.Array, *, scope: str
) -> tuple[pa.Array, pa.Array]:
    """Return the ids and minted flags rows keep when they join a claim scope.

    A claim's id never changes once it holds one: an id held in the scope,
    or held by two newcomers, is refused, whether it was given or minted. A
    row without an id is minted one (``claim_N``, the smallest N free), which
    is the only id a scope ever chooses. Every check runs before any id is
    chosen, so a refusal changes nothing. Vectorized: the cost is that of
    hashing the ids, whatever their number.

    Args:
        ids: The newcomers' ids (a string array; null for a row without one).
        minted: Whether each newcomer's id was minted.
        taken: The ids the scope already holds.
        scope: What the scope is, for the message (``"bundle"``, ``"field"``).

    Returns:
        The ids the newcomers keep, in row order, and their minted flags.

    Raises:
        ValueError: If an id is already held in the scope or held twice.
    """
    ids = pc.cast(ids, pa.string())
    taken = pc.cast(taken, pa.string())
    present = ids.filter(pc.is_valid(ids))
    held = present.filter(pc.is_in(present, value_set=taken))
    if len(held):
        raise ValueError(
            f"claim id {quote_id(held[0].as_py())} is already held by another "
            f"claim of this {scope}"
        )
    counts = pc.value_counts(present)
    repeated = counts.filter(pc.greater(counts.field("counts"), 1))
    if len(repeated):
        raise ValueError(
            f"claim id {quote_id(repeated.field('values')[0].as_py())} is held "
            f"by {repeated.field('counts')[0].as_py()} claims"
        )
    missing = pc.is_null(ids)
    minted = pc.or_(pc.fill_null(pc.cast(minted, pa.bool_()), False), missing)
    count = pc.sum(missing).as_py() or 0
    if count == 0:
        return ids, minted
    fresh = mint_ids(_CLAIM_ID_PREFIX, pa.concat_arrays([taken, present]), count)
    return pc.replace_with_mask(ids, missing, fresh), minted


def _join_claims(
    claims: Sequence["MatchClaim"], taken: pa.Array, *, scope: str
) -> list["MatchClaim"]:
    """Return *claims* as they join a scope holding the ids *taken*.

    The ids are admitted by :func:`_admitted_claim_ids`, which never renames
    one: a claim whose id the scope holds, or two claims under one id, are
    refused before anything changes. A claim that belongs to no bundle joins
    as itself; a claim a bundle already holds joins as a copy, so the bundle
    it came from keeps its back-reference.

    Args:
        claims: The newcomers.
        taken: The ids the scope already holds.
        scope: What the scope is, for the message.

    Returns:
        The claims to store, in order.
    """
    _admitted_claim_ids(
        pa.array([claim.id for claim in claims], type=pa.string()),
        pa.array([claim.id_minted for claim in claims], type=pa.bool_()),
        taken,
        scope=scope,
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


def _coordinate_from_input(
    value: int | float | Fraction | Coordinate,
    unit: TimeUnit | str | None,
    timeline_id: str,
    parameter_name: str,
) -> Coordinate:
    """Resolve a raw or unit-bearing coordinate factory argument."""
    if isinstance(value, Coordinate):
        if isinstance(value, IdCoordinate) and value.timeline_id != timeline_id:
            raise ValueError(
                f"{parameter_name} timeline_id {value.timeline_id!r} does not "
                f"match {timeline_id!r}"
            )
        if unit is not None and TimeUnit(unit) != value.unit:
            raise ValueError(
                f"{parameter_name} unit {value.unit.value!r} does not match "
                f"explicit unit {TimeUnit(unit).value!r}"
            )
        return value

    if unit is None:
        raise ValueError(
            f"{parameter_name} requires an explicit unit for a raw coordinate"
        )
    return Coordinate(value, unit)


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


def _event_id(event: dict[str, Any], where: str) -> str:
    """The id of an event a MatchClaim names, which it must state."""
    event_id = event.get("id")
    if event_id is None:
        raise ValueError(
            f"{where} names no event id: a MatchClaim names the events it "
            f"matches. Give a side without an event of its own a Projection "
            f"event first, never a bare coordinate."
        )
    return str(event_id)


def _event_name(event: dict[str, Any]) -> str | None:
    """The name of an event, if it states one."""
    name = event.get("name")
    return None if name is None else str(name)


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


_CLAIM_KEY_ORDER: tuple[str, ...] = (
    "id",
    "id_minted",
    "timeline_a_id",
    "timeline_b_id",
    "start_anchor",
    "end_anchor",
    "is_explicit",
    "is_synchronous",
    "certainty",
    "provenance",
    "event_a_id",
    "event_a_name",
    "event_b_id",
    "event_b_name",
    "source_coordinate",
    "source_claim_id",
)
"""Key order of the stored :meth:`MatchClaim.to_dict` shape.

``model_dump`` emits fields in declaration order; the stored dictionary
predates that order and is pinned here so the two can diverge without
either becoming the accident.
"""


class MatchClaim(BaseModel):
    """A claim that two events on different timelines correspond.

    A MatchClaim always knows which two timelines it connects via top-level
    ``timeline_a_id`` and ``timeline_b_id`` fields, and it names the events
    it is about: an explicit claim names an event on each side, or one event
    and the NOMATCH sentinel (``None``) on the other. It never states a bare
    coordinate: a side without an event of its own is given a Projection
    event first. Anchors are only present for synchronous claims, and each of
    their coordinates is on its timeline's own axis (:meth:`check_on_timelines`).

    Four cases:
        (a) ``from_events``: Two events on different timelines correspond
            (synchronous, with anchors; or conceptual, without).
        (b) ``nomatch``: An event has no equivalent on the other timeline.
        (c) ``implicit``: Implicit claim generated by MatchGraph group
            extension (derived, never stored as a statement).

    Attributes:
        timeline_a_id: First timeline's unique identifier.
        timeline_b_id: Second timeline's unique identifier.
        start_anchor: AlignmentAnchor for event starts (None for non-synchronous).
        end_anchor: AlignmentAnchor for event ends (None for instants or
            non-synchronous claims).
        is_synchronous: True if temporally synchronous matches.
        is_explicit: True if directly claimed, False if inferred.
        certainty: How certain the claim is, in [0, 1] (1.0, certain).
        provenance: How the claim was made and who believes it
            (:class:`~timetoalign.core.provenance.Provenance`).
        source_claim_id: For implicit claims, the ID of the claim that
            generated this one.
        event_a_id: ID of the event on timeline A (``None``: the NOMATCH
            sentinel).
        event_a_name: Name/label of the event on timeline A (if known).
        event_b_id: ID of the event on timeline B (``None``: the NOMATCH
            sentinel).
        event_b_name: Name/label of the event on timeline B (if known).
        source_coordinate: The position of a NOMATCH claim's event, on the
            timeline of its event. None for synchronous claims, whose
            coordinates are held by the anchors instead.
        id: The claim's identifier, unique in the scope holding the claim
            (a :class:`MatchClaimField` or an ``AlignmentBundle``). It never
            changes once given: a scope already holding it refuses the
            claim. Without one, the claim mints ``claim_1`` (its own scope
            is empty).
        id_minted: Whether ``id`` was minted rather than given.

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
        >>> claim.start_anchor is None, claim.claim_type.name
        (True, 'nomatch')
    """

    model_config = ConfigDict(frozen=True)

    timeline_a_id: str
    timeline_b_id: str
    start_anchor: AlignmentAnchor | None = None
    end_anchor: AlignmentAnchor | None = None
    is_synchronous: bool = True
    is_explicit: bool = True
    certainty: float = 1.0
    provenance: Provenance | None = None
    source_claim_id: str | None = None
    event_a_id: str | None = None
    event_a_name: str | None = None
    event_b_id: str | None = None
    event_b_name: str | None = None
    source_coordinate: Coordinate | None = None
    id: str = ""  # noqa: A003 — minted when not given
    id_minted: bool = False

    # Runtime association with the owning bundle. A private attr takes no
    # part in the derived Arrow schema or the repr, and ``__eq__`` excludes it.
    _bundle: Any = PrivateAttr(default=None)

    @field_validator("certainty")
    @classmethod
    def _validate_certainty(cls, value: float) -> float:
        """Validate certainty is in the closed range [0, 1]."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Certainty must be in [0, 1], got {value}")
        return value

    @model_validator(mode="after")
    def _validate_claim(self) -> "MatchClaim":
        """Mint or validate the ID; validate anchors and events against the kind."""
        # Frozen model: set through object.__setattr__.
        if not self.id:
            object.__setattr__(self, "id", mint_id(_CLAIM_ID_PREFIX, ()))
            object.__setattr__(self, "id_minted", True)
        else:
            validate_id(self.id)

        # Validate anchor presence based on synchrony
        if self.is_synchronous:
            if self.start_anchor is None:
                raise ValueError(
                    "Synchronous MatchClaims require a start_anchor. "
                    "Use is_synchronous=False for conceptual/NOMATCH claims."
                )
            # If both anchors exist, they must connect the same pair
            if self.end_anchor is not None:
                start_tls = {
                    self.start_anchor.timeline_a_id,
                    self.start_anchor.timeline_b_id,
                }
                end_tls = {
                    self.end_anchor.timeline_a_id,
                    self.end_anchor.timeline_b_id,
                }
                if start_tls != end_tls:
                    raise ValueError(
                        f"Start and end anchors must connect same timelines. "
                        f"Start connects {start_tls}, end connects {end_tls}"
                    )
            # Anchors must connect the claim's timelines
            anchor_tls = {
                self.start_anchor.timeline_a_id,
                self.start_anchor.timeline_b_id,
            }
            claim_tls = {self.timeline_a_id, self.timeline_b_id}
            if anchor_tls != claim_tls:
                raise ValueError(
                    f"Anchor timelines {anchor_tls} must match claim "
                    f"timelines {claim_tls}"
                )
        else:
            if self.start_anchor is not None or self.end_anchor is not None:
                raise ValueError(
                    "Non-synchronous MatchClaims must not have anchors. "
                    "Set start_anchor=None and end_anchor=None, or use "
                    "is_synchronous=True."
                )

        # An explicit claim names its events, never a bare coordinate.
        if self.is_explicit:
            n_events = int(self.event_a_id is not None) + int(
                self.event_b_id is not None
            )
            if self.is_synchronous and n_events != 2:
                raise ValueError(
                    "A synchronous MatchClaim names an event on each side "
                    "(event_a_id and event_b_id). Give a side without an event "
                    "of its own a Projection event first, never a bare "
                    "coordinate."
                )
            if not self.is_synchronous and n_events == 0:
                raise ValueError(
                    "A MatchClaim names its events: a NOMATCH claim one event "
                    "(the other side None, the NOMATCH sentinel), a conceptual "
                    "claim two."
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

        The classification reads only ``is_explicit``, ``is_synchronous`` and
        which of ``event_a_id`` / ``event_b_id`` are set — no stored field
        records the kind, so it always reflects the claim's current shape.

        Returns:
            The :class:`~timetoalign.core.enums.ClaimType` for this claim.

        The cases, in evaluation order:

        | Condition                            | Result          |
        |--------------------------------------|-----------------|
        | ``is_explicit`` is False             | ``implicit``    |
        | synchronous (an event on each side)  | ``event_match`` |
        | non-synchronous, one side names one  | ``nomatch``     |
        | non-synchronous, both sides name one | ``conceptual``  |
        """
        if not self.is_explicit:
            return ClaimType.implicit
        if self.is_synchronous:
            return ClaimType.event_match
        n_events = int(self.event_a_id is not None) + int(self.event_b_id is not None)
        if n_events == 1:
            return ClaimType.nomatch
        return ClaimType.conceptual

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

        Every coordinate the claim holds (both coordinates of each anchor,
        and a NOMATCH claim's ``source_coordinate``, which is on the side of
        its event) must be in its timeline's unit and declared number type.
        Every factory that receives the timelines runs this check.

        Args:
            timeline_a: Timeline A, or its ``(unit, number_type)`` pair. A
                timeline must be the one ``timeline_a_id`` names.
            timeline_b: Timeline B, likewise. Either may be ``None`` where
                the claim holds no coordinate on that side (a NOMATCH
                claim's other side).

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
        if self.source_coordinate is not None:
            refuse(
                self.source_coordinate,
                (
                    self.timeline_a_id
                    if self.event_a_id is not None
                    else self.timeline_b_id
                ),
                "source_coordinate",
            )
        return self

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
        branch = classify_dispatch_input(at, empty_is_keys=True)
        if branch == "key":
            return self.get_coordinate_for(at, format=format, rounding=rounding)
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
                anchor_edges=[(self.timeline_a_id, self.timeline_b_id)],
                inferred_edges=[],
                source=effective_bundle,
                is_interpolated=False,
                conversion_maps=conversion_maps,
            )

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
    ) -> "MatchClaim":
        """Case (a): Two events on different timelines correspond.

        Each event's coordinate is placed on its timeline's axis: a number is
        a position on the timeline, in its unit and declared number type; a
        :class:`Coordinate` in another unit is converted through the
        timeline's C-Map; a coordinate in the timeline's unit but another
        number type is refused (:meth:`check_on_timelines`, which runs on
        the claim made).

        Args:
            event_a: Event dict from timeline A; must hold ``id`` and
                ``coord_key`` (and ``end_coord_key`` for an interval match),
                may hold ``name``.
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
            id: The claim's id; minted when not given.

        Returns:
            A MatchClaim naming both events, with 1 or 2 anchors when
            synchronous.

        Raises:
            ValueError: If an event names no id, or a coordinate cannot be
                placed on its timeline's axis.
        """
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
            event_a_id=_event_id(event_a, "event_a"),
            event_a_name=_event_name(event_a),
            event_b_id=_event_id(event_b, "event_b"),
            event_b_name=_event_name(event_b),
            id=id or "",
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
    ) -> "MatchClaim":
        """Case (b): An event has no equivalent on the other timeline.

        Creates a non-synchronous claim with no anchors, naming the event on
        side A and the NOMATCH sentinel (``None``) on side B.

        Args:
            event: Event dict from *timeline*; must hold ``id``, may hold
                ``name`` and ``start``.
            timeline: The event's timeline.
            other_timeline: The timeline on which the event has no
                counterpart (a timeline, or its id).
            certainty: How certain the claim is, in [0, 1].
            provenance: How the claim was made and who believes it.
            id: The claim's id; minted when not given.

        Returns:
            A non-synchronous MatchClaim with no anchors. The event's
            ``start``, placed on its timeline's axis, is kept as
            ``source_coordinate``.

        Raises:
            ValueError: If the event names no id, or its start cannot be
                placed on its timeline's axis.
        """
        other_id = (
            other_timeline if isinstance(other_timeline, str) else other_timeline.id
        )
        source_coordinate = (
            _on_timeline(
                _event_coordinate(event["start"]), timeline, "source_coordinate"
            )
            if event.get("start") is not None
            else None
        )
        claim = cls(
            timeline_a_id=timeline.id,
            timeline_b_id=other_id,
            is_synchronous=False,
            is_explicit=True,
            certainty=certainty,
            provenance=provenance,
            event_a_id=_event_id(event, "event"),
            event_a_name=_event_name(event),
            source_coordinate=source_coordinate,
            id=id or "",
        )
        return claim.check_on_timelines(
            timeline, None if isinstance(other_timeline, str) else other_timeline
        )

    @classmethod
    def implicit(
        cls,
        tl_a_id: str,
        coord_a: CoordinateSpec,
        tl_b_id: str,
        coord_b: CoordinateSpec,
        *,
        unit_a: TimeUnit | str | None = None,
        unit_b: TimeUnit | str | None = None,
        source_claim: "MatchClaim | None" = None,
        provenance: Provenance | None = None,
    ) -> "MatchClaim":
        """Case (c): Implicit claim generated by MatchGraph group extension.

        An implicit claim is derived, never stated: it names no events.

        Args:
            tl_a_id: First timeline's ID.
            coord_a: Coordinate specification on timeline A.
            tl_b_id: Second timeline's ID.
            coord_b: Coordinate specification on timeline B.
            unit_a: Unit of timeline A's coordinate.
            unit_b: Unit of timeline B's coordinate.
            source_claim: The claim that generated this one.
            provenance: How the claim was made.

        Returns:
            A synchronous, non-explicit MatchClaim with one anchor.
        """
        coordinate_a = _coordinate_from_input(coord_a, unit_a, tl_a_id, "coord_a")
        coordinate_b = _coordinate_from_input(coord_b, unit_b, tl_b_id, "coord_b")

        return cls(
            timeline_a_id=tl_a_id,
            timeline_b_id=tl_b_id,
            start_anchor=AlignmentAnchor(
                timeline_a_id=tl_a_id,
                coordinate_a=coordinate_a,
                timeline_b_id=tl_b_id,
                coordinate_b=coordinate_b,
            ),
            is_synchronous=True,
            is_explicit=False,
            source_claim_id=source_claim.id if source_claim else None,
            provenance=provenance,
        )

    # endregion

    @field_serializer("source_coordinate")
    def _serialize_source_coordinate(
        self, coordinate: Coordinate | None
    ) -> dict[str, Any] | None:
        """Emit the NOMATCH coordinate through the shared JSON representation."""
        return None if coordinate is None else _coordinate_to_dict(coordinate)

    @field_serializer("provenance")
    def _serialize_provenance(
        self, provenance: Provenance | None
    ) -> dict[str, Any] | None:
        """Emit the provenance as its dictionary, absent keys left out."""
        return None if provenance is None else provenance.to_dict()

    @field_validator("source_coordinate", mode="before")
    @classmethod
    def _parse_source_coordinate(cls, value: Any) -> Any:
        """Accept the serialized coordinate dict as well as a Coordinate."""
        if isinstance(value, dict) and "unit" in value:
            return _coordinate_from_dict(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        The values come straight from ``model_dump``; only the key order
        and the omission of an absent ``source_claim_id`` are applied on
        top, since both are part of the stored shape.
        """
        dumped = self.model_dump(mode="json")
        if dumped["source_claim_id"] is None:
            del dumped["source_claim_id"]
        return {key: dumped[key] for key in _CLAIM_KEY_ORDER if key in dumped}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchClaim":
        """Deserialize from dictionary.

        Two normalisations precede validation: an absent timeline ID is
        taken from the start anchor, and a missing or null ``id`` becomes
        the empty string that triggers ID generation.
        """
        payload = dict(data)
        for slot in ("start_anchor", "end_anchor", "provenance"):
            if not payload.get(slot):
                payload[slot] = None
        anchor = payload.get("start_anchor")
        for own_key, anchor_key in (
            ("timeline_a_id", "timeline_a_id"),
            ("timeline_b_id", "timeline_b_id"),
        ):
            if payload.get(own_key) is not None:
                continue
            if isinstance(anchor, AlignmentAnchor):
                payload[own_key] = getattr(anchor, anchor_key)
            elif isinstance(anchor, dict):
                payload[own_key] = anchor.get(anchor_key)
        if payload.get("timeline_a_id") is None or payload.get("timeline_b_id") is None:
            raise ValueError(
                "Cannot deserialize MatchClaim: missing timeline IDs and no anchor."
            )
        payload["id"] = payload.get("id") or ""
        return cls.model_validate(payload)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "MatchClaim":
        """Reconstruct a claim from a struct-shaped Arrow row dict.

        This is the entry point ``SemanticField.__getitem__`` uses to
        materialise a scalar from the backing struct array. Every coordinate
        cell (both anchors' coordinates and ``source_coordinate``) is read
        as the coordinate it stores, in its unit and number type; the rest of
        the row is the shape :meth:`from_dict` accepts.

        Args:
            row: The ``.as_py()`` dict of one struct-column element.

        Returns:
            A reconstructed :class:`MatchClaim`.
        """
        payload = copy.deepcopy(row)
        for path in _coordinate_paths(MatchClaimField.pa_schema):
            holder = _holder(payload, path)
            if holder is not None and holder[path[-1]] is not None:
                holder[path[-1]] = _cell_coordinate(holder[path[-1]])
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
            if self.source_coordinate is not None:
                timeline_a = (
                    f"{self.timeline_a_id}@"
                    f"{_format_coordinate(self.source_coordinate)}"
                )
            else:
                timeline_a = self.timeline_a_id
            return f"MatchClaim({timeline_a} <-> {self.timeline_b_id}{badge})"

        if self.is_interval:
            interval_a = self.get_interval_for(self.timeline_a_id)
            interval_b = self.get_interval_for(self.timeline_b_id)
            start_a, end_a = interval_a.start, interval_a.end
            start_b, end_b = interval_b.start, interval_b.end
            return (
                f"MatchClaim({match_type}: "
                f"{self.timeline_a_id}["
                f"{_format_coordinate(start_a)}-{_format_coordinate(end_a)}] <-> "
                f"{self.timeline_b_id}["
                f"{_format_coordinate(start_b)}-{_format_coordinate(end_b)}]{badge})"
            )
        else:
            return (
                f"MatchClaim({match_type}: "
                f"{self.timeline_a_id}@"
                f"{_format_coordinate(self.start_anchor.coordinate_a)}"
                f" <-> "
                f"{self.timeline_b_id}@"
                f"{_format_coordinate(self.start_anchor.coordinate_b)}"
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

        def _event_str(ev_id: str | None, ev_name: str | None) -> str | None:
            """Format event info as 'id "name"' or just id/name if one is missing."""
            if ev_id and ev_name:
                return f'{ev_id} "{ev_name}"'
            elif ev_id:
                return ev_id
            elif ev_name:
                return f'"{ev_name}"'
            return None

        # Header: synchronous claims report their instant/interval shape;
        # non-synchronous ones report their derived kind (NOMATCH vs CONCEPTUAL).
        if not self.is_synchronous:
            header = f"MatchClaim ({self.claim_type.name.upper()})"
        elif self.is_interval:
            header = "MatchClaim (synchronous, interval)"
        else:
            header = "MatchClaim (synchronous, instant)"

        if not self.is_explicit:
            header += " [inferred]"

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
        elif self.source_coordinate is not None:
            lines.append(
                f"  Timeline A:  {self.timeline_a_id}  "
                f"@{_format_coordinate(self.source_coordinate)}"
            )
        else:
            lines.append(f"  Timeline A:  {self.timeline_a_id}")

        # Event A (always show if present)
        event_a_str = _event_str(self.event_a_id, self.event_a_name)
        if event_a_str:
            lines.append(f"  Event A:     {event_a_str}")

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

        # Event B (always show if present)
        event_b_str = _event_str(self.event_b_id, self.event_b_name)
        if event_b_str:
            lines.append(f"  Event B:     {event_b_str}")

        if self.certainty < 1.0:
            lines.append(f"  Certainty:   {self.certainty}")
        for label, text in _provenance_lines(self.provenance):
            lines.append(f"  {label + ':':<12} {text}")

        if self.source_claim_id is not None:
            lines.append(f"  Source:      {self.source_claim_id}")

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
        elif claim_type in (ClaimType.nomatch, ClaimType.conceptual):
            badge_bg = "#ffcdd2"
            badge_text = claim_type.name.upper()
        else:  # implicit — synchronous with anchors
            badge_bg = "#fff3e0"
            badge_text = claim_type.name.upper()

        badge = (
            f"<span style='background: {badge_bg}; padding: 0 4px; "
            f"border-radius: 3px; font-size: 0.8em;'>{badge_text}</span>"
        )

        def _event_html(
            ev_id: str | None, ev_name: str | None, label: str
        ) -> str | None:
            """Format event info as an HTML row, or None if no event info."""
            if ev_id and ev_name:
                return (
                    f"<tr style='color: #555; font-size: 0.9em;'><td>{label}</td>"
                    f"<td>{html_mod.escape(ev_id)}</td>"
                    f'<td>"{html_mod.escape(ev_name)}"</td></tr>'
                )
            elif ev_id:
                return (
                    f"<tr style='color: #555; font-size: 0.9em;'><td>{label}</td>"
                    f"<td colspan='2'>{html_mod.escape(ev_id)}</td></tr>"
                )
            elif ev_name:
                return (
                    f"<tr style='color: #555; font-size: 0.9em;'><td>{label}</td>"
                    f"<td colspan='2'>\"{html_mod.escape(ev_name)}\"</td></tr>"
                )
            return None

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
        elif self.source_coordinate is not None:
            source_coordinate = _format_coordinate(self.source_coordinate)
            rows.append(
                f"<tr><td>Timeline A</td>"
                f"<td>{html_mod.escape(self.timeline_a_id)}</td>"
                f"<td>@{html_mod.escape(source_coordinate)}</td></tr>"
            )
        else:
            rows.append(
                f"<tr><td>Timeline A</td>"
                f"<td>{html_mod.escape(self.timeline_a_id)}</td>"
                f"<td></td></tr>"
            )

        # Event A row (if present)
        event_a_row = _event_html(self.event_a_id, self.event_a_name, "Event A")
        if event_a_row:
            rows.append(event_a_row)

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

        # Event B row (if present)
        event_b_row = _event_html(self.event_b_id, self.event_b_name, "Event B")
        if event_b_row:
            rows.append(event_b_row)

        # Certainty and provenance rows
        described = _provenance_lines(self.provenance)
        if self.certainty < 1.0:
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


_CLAIM_COORDINATE_TYPE = pa.struct(
    [
        pa.field("value", pa.float64(), nullable=True),
        pa.field("numerator", pa.int64(), nullable=True),
        pa.field("denominator", pa.int64(), nullable=True),
        pa.field("unit", pa.string(), nullable=True),
        pa.field("number_type", pa.string(), nullable=True),
    ]
)
"""A claim coordinate cell: the number, its unit and its number type.

A ``float`` cell holds its number in ``value`` (``numerator`` and
``denominator`` null); an ``int`` or ``fraction`` cell holds it exactly in
``numerator``/``denominator``, ``value`` mirroring it as a float.
"""


def _claim_coordinate_projector(
    _model_cls: type[BaseModel], name: str, _info: object
) -> list[pa.Field]:
    """Store a claim coordinate's exact value, unit and number type in each row."""
    return [pa.field(name, _CLAIM_COORDINATE_TYPE, nullable=True)]


register_value_projector(AlignmentAnchor, "coordinate_a", _claim_coordinate_projector)
register_value_projector(AlignmentAnchor, "coordinate_b", _claim_coordinate_projector)
register_value_projector(MatchClaim, "source_coordinate", _claim_coordinate_projector)

#: The keys of a coordinate dict (:func:`coordinate_to_dict`), which is how a
#: claim coordinate cell is written in :meth:`MatchClaimField.to_dict`.
_COORDINATE_DICT_KEYS: tuple[str, ...] = ("unit", "number_type", "value")


def _cell_coordinate(cell: dict[str, Any]) -> Coordinate:
    """The coordinate a claim reads from a stored coordinate cell.

    The cell's number type says which side holds the number: ``value`` for
    a float, ``numerator``/``denominator`` for an int or a fraction.
    """
    number_type = NumberType(cell["number_type"])
    if number_type is NumberType.float:
        value: int | float | Fraction = cell["value"]
    elif number_type is NumberType.int:
        value = cell["numerator"]
    else:
        value = Fraction(cell["numerator"], cell["denominator"])
    return Coordinate(value, cell["unit"], number_type=number_type)


def _coordinate_cells(coordinates: Sequence[Coordinate | None]) -> pa.StructArray:
    """The cells of *coordinates*, each in its own unit and number type."""
    values: list[float | None] = []
    numerators: list[int | None] = []
    denominators: list[int | None] = []
    units: list[str | None] = []
    number_types: list[str | None] = []
    nulls: list[bool] = []
    for coordinate in coordinates:
        nulls.append(coordinate is None)
        if coordinate is None:
            for column in (values, numerators, denominators, units, number_types):
                column.append(None)
            continue
        values.append(float(coordinate.value))
        if coordinate.number_type is NumberType.float:
            numerators.append(None)
            denominators.append(None)
        else:
            exact = Fraction(coordinate.value)
            numerators.append(exact.numerator)
            denominators.append(exact.denominator)
        units.append(coordinate.unit.value)
        number_types.append(coordinate.number_type.name)
    return pa.StructArray.from_arrays(
        [
            pa.array(values, type=pa.float64()),
            pa.array(numerators, type=pa.int64()),
            pa.array(denominators, type=pa.int64()),
            pa.array(units, type=pa.string()),
            pa.array(number_types, type=pa.string()),
        ],
        fields=list(_CLAIM_COORDINATE_TYPE),
        mask=pa.array(nulls, type=pa.bool_()),
    )


def _axis_cells(
    values: Sequence[int | float | Fraction | None] | pa.Array | pa.ChunkedArray,
    axes: _Axis | Sequence[_Axis],
) -> pa.StructArray:
    """The cells of raw numbers, each a position on its row's axis.

    A number takes its axis's unit and is written in the axis's declared
    number type, as :class:`Coordinate` writes it. One axis for every row
    and a column of integers, or of floats on a float axis, is built
    vectorized; anything else goes number by number.

    Args:
        values: One number per row (null for none).
        axes: The axis of every row, or one per row.
    """
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    if isinstance(axes, _Axis):
        array: pa.Array | None = values if isinstance(values, pa.Array) else None
        if array is None:
            try:
                array = pa.array(list(values))
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                array = None
        if array is not None and (
            pa.types.is_integer(array.type)
            or (
                pa.types.is_floating(array.type)
                and axes.number_type is NumberType.float
            )
            or pa.types.is_null(array.type)
        ):
            n = len(array)
            exact = pa.types.is_integer(array.type) and axes.number_type is not (
                NumberType.float
            )
            return pa.StructArray.from_arrays(
                [
                    pc.cast(array, pa.float64()),
                    pc.cast(array, pa.int64()) if exact else pa.nulls(n, pa.int64()),
                    (
                        pa.array(np.ones(n, dtype=np.int64))
                        if exact
                        else pa.nulls(n, pa.int64())
                    ),
                    pa.array([axes.unit.value] * n, pa.string()),
                    pa.array([axes.number_type.name] * n, pa.string()),
                ],
                fields=list(_CLAIM_COORDINATE_TYPE),
                mask=pc.is_null(array),
            )
        raw = values.to_pylist() if isinstance(values, pa.Array) else list(values)
        row_axes: Sequence[_Axis] = [axes] * len(raw)
    else:
        raw = values.to_pylist() if isinstance(values, pa.Array) else list(values)
        row_axes = axes
    if len(row_axes) != len(raw):
        raise ValueError("Coordinate values and axes must have equal length.")
    return _coordinate_cells(
        [
            (
                None
                if value is None
                else Coordinate(
                    value.item() if hasattr(value, "item") else value,
                    axis.unit,
                    number_type=axis.number_type,
                )
            )
            for value, axis in zip(raw, row_axes, strict=True)
        ]
    )


def _coordinate_paths(
    struct_type: pa.StructType, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Yield the path of every claim coordinate cell in *struct_type*."""
    for sub_field in struct_type:
        path = (*prefix, sub_field.name)
        if sub_field.type == _CLAIM_COORDINATE_TYPE:
            yield path
        elif pa.types.is_struct(sub_field.type):
            yield from _coordinate_paths(sub_field.type, path)


def _holder(row: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any] | None:
    """The mapping holding the last key of *path* in *row*, or ``None``.

    ``None`` where a mapping on the way is null (an absent end anchor).
    """
    holder: Any = row
    for key in path[:-1]:
        holder = holder[key]
        if holder is None:
            return None
    return holder


def _with_children(
    array: pa.StructArray,
    children: dict[tuple[str, ...], pa.Array],
    prefix: tuple[str, ...] = (),
) -> pa.StructArray:
    """Return *array* with the children at the given paths replaced.

    Every struct on a path is rebuilt with its own validity, so a null end
    anchor stays null.
    """
    arrays = []
    for index, sub_field in enumerate(array.type):
        path = (*prefix, sub_field.name)
        child = array.field(index)
        if path in children:
            child = children[path]
        elif any(key[: len(path)] == path for key in children):
            child = _with_children(child, children, path)
        arrays.append(child)
    return pa.StructArray.from_arrays(
        arrays,
        fields=list(array.type),
        mask=array.is_null() if array.null_count else None,
    )


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
    with their anchors, NOMATCH and conceptual claims without.

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

    The field is the scope of its claims' ids: a row without an id is minted
    one, and an id held twice is refused. An id never changes.

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

    #: The Arrow schema-metadata key carrying the field-level provenance.
    _PROVENANCE_KEY: ClassVar[bytes] = b"timetoalign:provenance"
    #: The file name of the MatchClaim table beside a manifest.
    file_name: ClassVar[str] = "match_claims.parquet"
    #: The per-row values beside the anchors, the source coordinate and the
    #: provenance that :meth:`from_claims` reads from each claim under its
    #: own name; the types are the derived struct's.
    _ROW_VALUES: ClassVar[tuple[str, ...]] = (
        "id",
        "id_minted",
        "is_synchronous",
        "is_explicit",
        "certainty",
        "source_claim_id",
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
    ) -> "MatchClaimField":
        """Build a field of event matches from parallel columns (the vectorized path).

        This is the constructor a bulk producer (e.g. an alignment loader)
        uses. It assembles the derived ``MatchClaim`` struct array straight
        from the parallel inputs and **never** materialises a single
        :class:`MatchClaim` Python object, so it scales to millions of rows.

        Each row is a synchronous, explicit claim between two events: the
        timeline and event ids are filled from the inputs, and every
        coordinate is a position on its row's timeline, written in that
        timeline's unit and declared number type. With the two end columns,
        every row with an end on both sides is an interval claim. The ids
        ``claim_1`` ... ``claim_n`` are minted (the field is the scope of its
        claims' ids); names, ``source_coordinate`` and ``source_claim_id``
        are left null, and so is the ``provenance`` column (the shared
        provenance is injected on read). The field is checked against the
        timelines (:meth:`check_on_timelines`).

        Args:
            timeline_a_ids: Timeline-A ids, one per claim.
            timeline_b_ids: Timeline-B ids, one per claim.
            event_a_ids: The event on timeline A, one per claim.
            event_b_ids: The event on timeline B, one per claim.
            coordinate_a: The position of the event on timeline A.
            coordinate_b: The position of the event on timeline B.
            timelines: Every timeline the rows name: timelines, or a mapping
                from a timeline id to a timeline or its ``(unit,
                number_type)`` pair.
            end_coordinate_a: The end of each event on timeline A (null for
                an instant).
            end_coordinate_b: The end of each event on timeline B.
            certainty: How certain every claim is, in [0, 1].
            provenance: Shared provenance for every claim in the field.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If the inputs do not all have the same length, a row
                names no event, a row names a timeline not among
                *timelines*, or the end columns are given one without the
                other.
        """
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
                    f"Row {row} names no event on side {side}: a MatchClaim names "
                    f"the events it matches. Give a side without an event of its "
                    f"own a Projection event first, never a bare coordinate."
                )
        axes_a = cls._row_axes(tl_a, axes)
        axes_b = cls._row_axes(tl_b, axes)
        anchors = {
            "start_anchor": cls._anchor_array(
                tl_a,
                _axis_cells(coordinate_a, axes_a),
                tl_b,
                _axis_cells(coordinate_b, axes_b),
            )
        }
        if end_coordinate_a is not None:
            end_a = _axis_cells(end_coordinate_a, axes_a)
            end_b = _axis_cells(end_coordinate_b, axes_b)
            anchors["end_anchor"] = cls._anchor_array(
                tl_a,
                end_a,
                tl_b,
                end_b,
                mask=pc.or_(pc.is_null(end_a), pc.is_null(end_b)),
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
                "is_explicit": true_column,
                "certainty": pa.array(np.full(n, float(certainty)), pa.float64()),
                **anchors,
            },
        )
        return cls._read(
            struct_array, provenance, source="column set", where="as provenance="
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
        materialises equal to the one stored. The field is the scope of its
        claims' ids: two claims under one id are refused (an id is never
        renamed).

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
            ValueError: If two claims hold the same id.
        """
        claims = list(claims)
        if provenance is None:
            provenance = cls._common_provenance(claims)
        schema = cls.pa_schema
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
                cls._string_array([a and a.timeline_a_id for a in anchors]),
                _coordinate_cells([a and a.coordinate_a for a in anchors]),
                cls._string_array([a and a.timeline_b_id for a in anchors]),
                _coordinate_cells([a and a.coordinate_b for a in anchors]),
                mask=pa.array([a is None for a in anchors], type=pa.bool_()),
            )
        columns["source_coordinate"] = _coordinate_cells(
            [claim.source_coordinate for claim in claims]
        )
        if provenance is None:
            columns["provenance"] = provenance_array(
                [claim.provenance for claim in claims]
            )
        return cls._scoped(cls._assemble(len(claims), columns), provenance=provenance)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchClaimField":
        """Deserialize a field from a plain dictionary (inverse of :meth:`to_dict`).

        The mapping holds ``provenance`` (the field's
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`, or ``None``)
        and ``rows``: one mapping per claim whose keys are the
        :attr:`table`'s columns. Each coordinate is a coordinate dict
        (``{unit, number_type, value}``), read by
        :func:`~timetoalign.core.retrieval.coordinate_from_dict` and written
        into its storage cell as :meth:`from_claims` writes a claim's; every
        other cell is read as stored. A field built from claims therefore
        comes back with an equal table and equal claims.

        Args:
            data: A mapping as :meth:`to_dict` writes it.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If a mapping, at any depth, lacks a key it must hold
                or holds an unknown one (the mapping itself, row, anchor,
                provenance, belief, coordinate), or a coordinate dict is
                invalid; if a row is not a MatchClaim (:meth:`_read`); if the
                mapping carries provenance both under its key ``provenance``
                and in its rows; or if two rows hold the same id.
        """
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
            cls._refuse_other_keys(row, cls.pa_schema, index)
        cells: dict[tuple[str, ...], pa.StructArray] = {}
        for path in _coordinate_paths(cls.pa_schema):
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
            cells[path] = _coordinate_cells(coordinates)
        return cls._read(
            _with_children(pa.array(rows, type=cls.pa_schema), cells),
            provenance,
            source="dictionary",
            where="under its key 'provenance'",
        )

    @classmethod
    def _refuse_other_keys(
        cls,
        value: Any,
        data_type: pa.DataType,
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
        if pa.types.is_list(data_type):
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
        coordinate = data_type == _CLAIM_COORDINATE_TYPE
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
            if pa.types.is_struct(sub_field.type) or (
                pa.types.is_list(sub_field.type)
                and pa.types.is_struct(sub_field.type.value_type)
            ):
                cls._refuse_other_keys(
                    value[sub_field.name],
                    sub_field.type,
                    index,
                    (*path, sub_field.name),
                )

    @classmethod
    def from_table(cls, table: pa.Table) -> "MatchClaimField":
        """Read a field back from its Arrow table (inverse of :attr:`table`).

        The table must hold one column per top-level property of
        :class:`MatchClaim`, named and typed as the fields of
        :attr:`pa_schema`, in that order; the columns may be chunked, as they
        are when the table was read from Parquet. The field-level
        provenance travels in the table's schema metadata under the key
        ``timetoalign:provenance`` (the JSON of
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`); a table
        without that key gives a field whose provenance is ``None``. Every
        row is kept as stored; a row without an id is minted one.

        Args:
            table: A table as written by :attr:`table`.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If the table's columns are not exactly the
                ``MatchClaim`` properties, or a column's type is not the
                property's; if a row is not a MatchClaim (:meth:`_read`); if
                the table carries provenance both in its schema metadata and
                in its rows; or if two rows hold the same id.
        """
        columns = cls.pa_schema.names
        if table.column_names != columns:
            raise ValueError(
                f"A MatchClaimField table holds one column per MatchClaim "
                f"property, {columns}; this table has the columns "
                f"{table.column_names}."
            )
        for sub_field in cls.pa_schema:
            stored = table.schema.field(sub_field.name).type
            if stored != sub_field.type:
                raise ValueError(
                    f"Column {sub_field.name!r} has type {stored}, which is not "
                    f"the type of the MatchClaim property, {sub_field.type}."
                )
        struct_array = pa.StructArray.from_arrays(
            [table.column(name).combine_chunks() for name in columns],
            fields=list(cls.pa_schema),
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
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | os.PathLike[str],
        *,
        timelines: Iterable[Any] | Mapping[str, Any] | None = None,
    ) -> "MatchClaimField":
        """Read the MatchClaim table from its Parquet file.

        Args:
            path: The file, or a directory holding :attr:`file_name`
                (``match_claims.parquet``).
            timelines: Where given, the timelines the rows name (as
                :meth:`check_on_timelines` takes them); the field is checked
                against them.

        Returns:
            The field :meth:`to_parquet` wrote, value for value.

        Raises:
            ValueError: As :meth:`from_table`, or as
                :meth:`check_on_timelines`.
        """
        import pyarrow.parquet as pq

        field = cls.from_table(pq.read_table(cls._file_path(path)))
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
    ) -> "MatchClaimField":
        """Admit rows read from a table, a dictionary or columns.

        Refuses a row no :class:`MatchClaim` could be (:meth:`_refuse_malformed`)
        and provenance stated twice (:meth:`_refuse_provenance_twice`), then
        admits the ids (:meth:`_scoped`).

        Args:
            struct_array: The rows, of type :attr:`pa_schema`.
            provenance: The field-level provenance the source states, if any.
            source: What is read, for the messages (``"table"``).
            where: Where the source states the field-level provenance.
        """
        cls._refuse_malformed(struct_array)
        cls._refuse_provenance_twice(
            provenance, struct_array.field("provenance"), source=source, where=where
        )
        return cls._scoped(struct_array, provenance=provenance)

    @staticmethod
    def _refuse_malformed(data: pa.StructArray) -> None:
        """Refuse a row no :class:`MatchClaim` could be, naming the first.

        The rules of the scalar, vectorized: ``is_synchronous`` and
        ``is_explicit`` stated; a synchronous row holds a start anchor, a
        non-synchronous one no anchor; an explicit synchronous row names an
        event on each side, an explicit non-synchronous one at least one
        event; each anchor connects the row's two timelines; the certainty
        lies in [0, 1].

        Raises:
            ValueError: Naming the first offending row and what it lacks.
        """
        if len(data) == 0:
            return
        synchronous = data.field("is_synchronous")
        explicit = data.field("is_explicit")
        start = data.field("start_anchor")
        end = data.field("end_anchor")
        n_events = pc.add(
            pc.cast(pc.is_valid(data.field("event_a_id")), pa.int8()),
            pc.cast(pc.is_valid(data.field("event_b_id")), pa.int8()),
        )
        tl_a = data.field("timeline_a_id")
        tl_b = data.field("timeline_b_id")
        certainty = data.field("certainty")

        def off_timelines(anchor: pa.StructArray) -> pa.Array:
            a = anchor.field("timeline_a_id")
            b = anchor.field("timeline_b_id")
            same = pc.or_(
                pc.and_(pc.equal(a, tl_a), pc.equal(b, tl_b)),
                pc.and_(pc.equal(a, tl_b), pc.equal(b, tl_a)),
            )
            return pc.and_(anchor.is_valid(), pc.invert(pc.fill_null(same, False)))

        checks = (
            (
                pc.or_(pc.is_null(synchronous), pc.is_null(explicit)),
                "states no is_synchronous or no is_explicit",
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
                pc.and_(pc.and_(explicit, synchronous), pc.not_equal(n_events, 2)),
                "is synchronous and does not name an event on each side; give a "
                "side without an event of its own a Projection event first",
            ),
            (
                pc.and_(
                    pc.and_(explicit, pc.invert(synchronous)), pc.equal(n_events, 0)
                ),
                "names no event",
            ),
            (off_timelines(start), "has a start anchor on other timelines"),
            (off_timelines(end), "has an end anchor on other timelines"),
            (
                pc.or_(
                    pc.is_null(certainty),
                    pc.or_(pc.less(certainty, 0.0), pc.greater(certainty, 1.0)),
                ),
                "has a certainty outside [0, 1]",
            ),
        )
        for mask, reason in checks:
            mask = pc.fill_null(mask, False)
            if pc.any(mask).as_py():
                row = pc.index(mask, True).as_py()
                raise ValueError(f"MatchClaim table row {row} {reason}.")

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
    def _scoped(
        cls,
        struct_array: pa.StructArray,
        *,
        provenance: Provenance | None,
        taken: pa.Array | None = None,
        scope: str = "field",
    ) -> "MatchClaimField":
        """Wrap *struct_array* with its ids admitted into a scope.

        A row without an id is minted one; an id held in the scope or held
        twice is refused (:func:`_admitted_claim_ids`).

        Args:
            struct_array: The claim rows.
            provenance: The field-level provenance.
            taken: The ids the scope already holds (none: the field is its
                own scope).
            scope: What the scope is, for the message.
        """
        ids, minted = _admitted_claim_ids(
            struct_array.field("id"),
            struct_array.field("id_minted"),
            pa.array([], type=pa.string()) if taken is None else taken,
            scope=scope,
        )
        children = {"id": ids, "id_minted": minted}
        arrays = [
            children.get(sub_field.name, struct_array.field(sub_field.name))
            for sub_field in cls.pa_schema
        ]
        rebuilt = pa.StructArray.from_arrays(arrays, fields=list(cls.pa_schema))
        return cls._from_struct_array(rebuilt, provenance=provenance)

    @classmethod
    def _from_struct_array(
        cls,
        struct_array: pa.StructArray,
        *,
        provenance: Provenance | None = None,
    ) -> "MatchClaimField":
        """Wrap a derived ``MatchClaim`` struct array as a live field.

        The in-memory struct is named as :meth:`SemanticField.from_array`
        names a field by default; :attr:`table` spells each of its fields as
        a column of its own.
        """
        pa_field = pa.field(cls.__name__.removesuffix("Field").lower(), cls.pa_schema)
        raw = StructField(struct_array, pa_field)
        return cls(raw, provenance=provenance)

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
            for sub_field in cls.pa_schema
        ]
        return pa.StructArray.from_arrays(arrays, fields=list(cls.pa_schema))

    @classmethod
    def _anchor_array(
        cls,
        timeline_a_id: pa.Array,
        coordinate_a: pa.StructArray,
        timeline_b_id: pa.Array,
        coordinate_b: pa.StructArray,
        *,
        mask: pa.Array | None = None,
    ) -> pa.StructArray:
        """Build one anchor struct from parallel typed columns (null where *mask*)."""
        anchor_type = cls.pa_schema.field("start_anchor").type
        sources: dict[str, pa.Array] = {
            "timeline_a_id": timeline_a_id,
            "coordinate_a": coordinate_a,
            "timeline_b_id": timeline_b_id,
            "coordinate_b": coordinate_b,
        }
        return pa.StructArray.from_arrays(
            [sources[field.name] for field in anchor_type],
            fields=list(anchor_type),
            mask=mask,
        )

    @staticmethod
    def _row_axes(
        timeline_ids: pa.Array, axes: dict[str, _Axis]
    ) -> _Axis | list[_Axis]:
        """The axis of each row's timeline: one for all rows, or one per row.

        Raises:
            ValueError: If a row names a timeline whose axis is not given.
        """
        named = [value for value in pc.unique(timeline_ids).to_pylist()]
        unknown = [value for value in named if value not in axes]
        if unknown:
            raise ValueError(
                f"The rows name timeline {quote_id(str(unknown[0]))}, which is not "
                f"among the timelines given."
            )
        if len(named) == 1:
            return axes[named[0]]
        return [axes[value] for value in timeline_ids.to_pylist()]

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
        """
        super().__init__(raw, source_fields=source_fields, name=name)
        self._provenance: Provenance | None = provenance

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

    @property
    def table(self) -> pa.Table:
        """The claims as a :class:`pyarrow.Table`, one column per property.

        The columns are the top-level properties of :class:`MatchClaim`, the
        fields of the derived struct ``MatchClaimField.pa_schema`` in its
        order, each nested as the property is (``start_anchor`` a struct of
        two coordinate structs, ``provenance`` a struct holding its list of
        beliefs). The field-level provenance, when set, travels in the
        schema metadata under ``timetoalign:provenance`` as the JSON of
        :meth:`~timetoalign.core.provenance.Provenance.to_dict`, so the table
        (and the Parquet file written from it) reads back whole through
        :meth:`from_table`.
        """
        schema_metadata = (
            None
            if self._provenance is None
            else {self._PROVENANCE_KEY: json.dumps(self._provenance.to_dict())}
        )
        table = pa.Table.from_struct_array(self._struct_array())
        return table.replace_schema_metadata(schema_metadata)

    @property
    def provenance(self) -> Provenance | None:
        """Shared provenance applied to every materialised claim."""
        return self._provenance

    @property
    def timeline_ids(self) -> set[str]:
        """The distinct timeline ids appearing in either id column.

        Reads the struct's two id sub-fields vectorized (no claim is
        materialised).
        """
        data = self._raw.data
        if data is None or len(data) == 0:
            return set()
        ids: set[str] = set()
        for name in ("timeline_a_id", "timeline_b_id"):
            column = data.field(name)
            ids.update(v for v in column.to_pylist() if v is not None)
        return ids

    @property
    def _ids(self) -> pa.Array:
        """The claim ids, one per row."""
        return self._struct_array().field("id")

    def check_on_timelines(
        self, timelines: Iterable[Any] | Mapping[str, Any]
    ) -> "MatchClaimField":
        """Refuse this field unless every coordinate is on its timeline's axis.

        The vectorized form of :meth:`MatchClaim.check_on_timelines`: every
        coordinate cell (both coordinates of each anchor, and the
        ``source_coordinate`` of a NOMATCH row, on the side of its event) is
        compared with the unit and declared number type of the timeline it
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
            anchor = data.field(anchor_name)
            for side in ("a", "b"):
                cells = anchor.field(f"coordinate_{side}")
                cells_by_place.append(
                    (
                        f"{anchor_name}.coordinate_{side}",
                        anchor.field(f"timeline_{side}_id"),
                        cells,
                        pc.and_(anchor.is_valid(), cells.is_valid()),
                    )
                )
        source = data.field("source_coordinate")
        cells_by_place.append(
            (
                "source_coordinate",
                pc.if_else(
                    pc.is_valid(data.field("event_a_id")),
                    data.field("timeline_a_id"),
                    data.field("timeline_b_id"),
                ),
                source,
                source.is_valid(),
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
            off = pc.fill_null(off, False)
            if pc.any(off).as_py():
                row = pc.index(off, True).as_py()
                timeline_id = timeline_ids[row].as_py()
                cell = cells[row].as_py()
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
        data = self._raw.data
        if data is None or len(data) == 0:
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
            side_id = pc.equal(data.field(f"timeline_{side}_id"), timeline_id)
            start_cells = data.field("start_anchor").field(f"coordinate_{side}")
            end_cells = data.field("end_anchor").field(f"coordinate_{side}")
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
            relevant = pc.if_else(
                pc.is_valid(data.field("end_anchor")), contained, start
            )
            # A side whose cells cannot answer the query (null) is not relevant.
            masks.append(
                pc.fill_null(
                    pc.and_(
                        pc.and_(side_id, data.field("start_anchor").is_valid()),
                        relevant,
                    ),
                    False,
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
        belief: BeliefFilter = "not_false",
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
            nomatch_only: Keep the NOMATCH rows (explicit, not synchronous,
                one event named).
            believer: The actor whose beliefs filter the rows (the belief
                filter, :func:`~timetoalign.core.provenance.belief_mask`);
                ``None`` filters nothing.
            belief: ``"not_false"`` (default) keeps every row the believer
                does not believe ``false``; ``"true"`` only the rows the
                believer believes ``true``.

        Returns:
            A new :class:`MatchClaimField`. With no criteria set, a copy of
            the whole field is returned.

        Raises:
            ValueError: If ``synchronous_only`` and ``nomatch_only`` are both
                set, or ``belief`` is neither ``"not_false"`` nor ``"true"``.
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
            masks.append(pc.fill_null(data.field("is_synchronous"), False))
        if nomatch_only:
            masks.append(self._nomatch_mask())
        if believer is not None or belief != "not_false":
            masks.append(self._belief_mask(believer, belief))

        if not masks:
            return self._from_struct_array(data, provenance=self._provenance)
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
        data = self._raw.data
        if data is None or len(data) == 0:
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

    def max_coordinate(self, timeline_id: str) -> float | None:
        """The largest start-anchor coordinate this field carries for ``timeline_id``.

        Both anchor sides are scanned vectorized with :func:`pyarrow.compute.max`;
        no claim is materialised.

        Args:
            timeline_id: The timeline to look up.

        Returns:
            The maximum coordinate value carried for that timeline, or
            ``None`` if the timeline does not appear in an anchor.
        """
        data = self._raw.data
        if data is None or len(data) == 0:
            return None
        largest: float | None = None
        for side in ("a", "b"):
            mask = pc.equal(data.field(f"timeline_{side}_id"), timeline_id)
            values = pc.filter(self._anchor_values(side), mask)
            if len(values) == 0:
                continue
            side_max = pc.max(values).as_py()
            if side_max is not None and (largest is None or side_max > largest):
                largest = side_max
        return None if largest is None else float(largest)

    def coordinate_pairs(
        self,
    ) -> tuple[list[str], list[str], list[CoordinateValue], list[CoordinateValue]]:
        """Read the four defining columns in bulk, materialising no claim.

        This is the read a tabular consumer wants: the two timeline ids and
        the two start-anchor coordinates of every row, as four parallel
        Python lists pulled straight off the Arrow columns. Materialising
        the rows as :class:`MatchClaim` objects instead costs orders of
        magnitude more for the same information.

        Each coordinate is read in the number type its cell declares, so an
        anchor authored at ``5/3`` quarters comes back as ``Fraction(5, 3)``
        rather than as the double nearest to it. A row without anchors gives
        ``None`` on both sides.

        Returns:
            ``(timeline_a_ids, timeline_b_ids, coordinates_a, coordinates_b)``,
            all of length ``len(self)``.
        """
        data = self._raw.data
        if data is None or len(data) == 0:
            return [], [], [], []
        return (
            data.field("timeline_a_id").to_pylist(),
            data.field("timeline_b_id").to_pylist(),
            self._anchor_coordinates("a"),
            self._anchor_coordinates("b"),
        )

    def _anchor_coordinates(self, side: str) -> list[CoordinateValue | None]:
        """Read one start-anchor column, each cell in its declared number type."""
        anchor = self._raw.data.field("start_anchor")
        cells = anchor.field(f"coordinate_{side}")
        valid = pc.and_(anchor.is_valid(), cells.is_valid()).to_pylist()
        values = cells.field("value").to_pylist()
        numerators = cells.field("numerator").to_pylist()
        denominators = cells.field("denominator").to_pylist()
        number_types = cells.field("number_type").to_pylist()
        read: list[CoordinateValue | None] = []
        for is_valid, value, numerator, denominator, number_type in zip(
            valid, values, numerators, denominators, number_types, strict=True
        ):
            if not is_valid:
                read.append(None)
            elif number_type == NumberType.float.name:
                read.append(value)
            elif number_type == NumberType.int.name:
                read.append(numerator)
            else:
                read.append(Fraction(numerator, denominator))
        return read

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
            ... )
            >>> row = field.to_dict()["rows"][0]
            >>> row["start_anchor"]["coordinate_a"]
            {'unit': 'quarters', 'number_type': 'fraction', 'value': [1, 3]}
            >>> row["start_anchor"]["coordinate_b"]
            {'unit': 'seconds', 'number_type': 'float', 'value': 0.5}
            >>> row["id"], row["id_minted"], row["end_anchor"]
            ('claim_1', True, None)
        """
        rows = self._struct_array().to_pylist()
        paths = list(_coordinate_paths(self.pa_schema))
        for row in rows:
            for path in paths:
                holder = _holder(row, path)
                if holder is not None and holder[path[-1]] is not None:
                    holder[path[-1]] = _coordinate_to_dict(
                        _cell_coordinate(holder[path[-1]])
                    )
        return {
            "provenance": (
                self._provenance.to_dict() if self._provenance is not None else None
            ),
            "rows": rows,
        }

    def to_parquet(self, path: str | os.PathLike[str]) -> Path:
        """Write the MatchClaim table to its Parquet file.

        The file holds :attr:`table`, the field-level provenance in its
        schema metadata, so :meth:`from_parquet` reads back every value
        exactly. Where the file exists, the new table must succeed it: every
        id of the existing table is in this one, with the same timelines,
        events, anchors and every other value; only its provenance may
        differ (a MatchClaim is never removed and never changed, only
        believed or disbelieved). The file is replaced whole, never in part.

        Args:
            path: The file, or a directory to hold :attr:`file_name`
                (``match_claims.parquet``).

        Returns:
            The path of the file written.

        Raises:
            ValueError: If the existing table holds a MatchClaim this one
                lacks or changes.
        """
        import pyarrow.parquet as pq

        target = self._file_path(path)
        if target.exists():
            self._refuse_unless_succeeding(type(self).from_parquet(target))
        temporary = target.with_name(f".{target.name}.writing")
        pq.write_table(self.table, temporary)
        os.replace(temporary, target)
        return target

    def _refuse_unless_succeeding(self, existing: "MatchClaimField") -> None:
        """Refuse this table as a successor of *existing* unless it keeps it.

        Every id of *existing* must be here, its row equal in every column
        but ``provenance``. Vectorized: the rows are matched by id and
        compared leaf column by leaf column.

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
                f"MatchClaim is never removed (state the belief 'false' instead)."
            )
        old = _leaf_columns(existing.table.drop_columns(["provenance"]))
        new = _leaf_columns(self.table.drop_columns(["provenance"]).take(positions))
        for name in old.column_names:
            before = old.column(name).combine_chunks()
            after = new.column(name).combine_chunks()
            same = pc.or_(
                pc.fill_null(pc.equal(before, after), False),
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

    def _joined(self, taken: pa.Array, *, scope: str) -> "MatchClaimField":
        """Return this field as it joins a scope holding the ids *taken*.

        No id changes: an id the scope holds is refused (see
        :func:`_admitted_claim_ids`), and the field itself is returned.
        """
        _admitted_claim_ids(
            self._ids, self._struct_array().field("id_minted"), taken, scope=scope
        )
        return self

    def _struct_array(self) -> pa.StructArray:
        """The claim rows, an empty array of the struct type when none."""
        data = self._raw.data
        return pa.array([], type=self.pa_schema) if data is None else data

    def _empty(self) -> "MatchClaimField":
        """Return a zero-row field of this schema, keeping the provenance."""
        return self._from_struct_array(
            self._struct_array().slice(0, 0), provenance=self._provenance
        )

    def _filtered(self, mask: pa.Array) -> "MatchClaimField":
        """Return a new field with the struct array filtered by ``mask``."""
        filtered = self._struct_array().filter(mask)
        return self._from_struct_array(filtered, provenance=self._provenance)

    def _anchor_values(self, side: str) -> pa.Array:
        """The stored start-anchor coordinate values on ``side`` (``"a"`` / ``"b"``)."""
        anchor = self._raw.data.field("start_anchor")
        return anchor.field(f"coordinate_{side}").field("value")

    def _nomatch_mask(self) -> pa.Array:
        """Boolean mask: the NOMATCH rows (explicit, not synchronous, one event)."""
        data = self._struct_array()
        n_events = pc.add(
            pc.cast(pc.is_valid(data.field("event_a_id")), pa.int8()),
            pc.cast(pc.is_valid(data.field("event_b_id")), pa.int8()),
        )
        return pc.fill_null(
            pc.and_(
                pc.and_(
                    data.field("is_explicit"), pc.invert(data.field("is_synchronous"))
                ),
                pc.equal(n_events, 1),
            ),
            False,
        )

    def _belief_mask(self, believer: str | None, belief: BeliefFilter) -> pa.Array:
        """Boolean mask: the rows the belief filter keeps for *believer*.

        The shared provenance, where the field holds one, decides for every
        row; otherwise each row's own (:func:`belief_mask` either way).
        """
        data = self._struct_array()
        if self._provenance is not None:
            keep = self._provenance.passes_belief_filter(believer, belief)
            return pa.array(np.full(len(data), keep), type=pa.bool_())
        return belief_mask(data.field("provenance"), believer, belief)

    @staticmethod
    def _id_value_set(timeline_ids: set[str]) -> pa.Array:
        """A deterministic ``string`` value set for :func:`pyarrow.compute.is_in`."""
        return pa.array(sorted(timeline_ids), type=pa.string())

    def _involves_mask(self, timeline_id: str) -> pa.Array:
        """Boolean mask: rows where either id sub-field equals ``timeline_id``."""
        data = self._struct_array()
        return pc.or_(
            pc.equal(data.field("timeline_a_id"), timeline_id),
            pc.equal(data.field("timeline_b_id"), timeline_id),
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
                pc.equal(column_a, timeline_a_id),
                pc.equal(column_b, timeline_b_id),
            ),
            pc.and_(
                pc.equal(column_a, timeline_b_id),
                pc.equal(column_b, timeline_a_id),
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
