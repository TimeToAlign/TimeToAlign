"""AlignmentAnchor and MatchClaim scalars and the paired MatchClaimField.

This module implements the atomic and low-level alignment claims from the
multi-level alignment hierarchy:

- Agent: The human or software author of a match claim.
- MatchMetadata: Provenance information for a match (agent + certainty).
- AlignmentAnchor: Neutral coordinate pair (pure value object).
- MatchClaim: Alignment claim between two timelines (with provenance).
- MatchClaimField: A ``SemanticField[MatchClaim]`` columnar store.

The hierarchy is:
    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine

Design:
    Agent, MatchMetadata, AlignmentAnchor, and MatchClaim are all frozen
    pydantic v2 ``BaseModel`` scalars.  AlignmentAnchor is a pure coordinate
    pair with no claim semantics; its coordinates are unit-bearing values.
    Only synchronous MatchClaims produce AlignmentAnchors.  A MatchClaim
    always knows its two timelines via top-level fields.

    ``MatchClaim`` is paired with ``MatchClaimField`` — a genuine
    ``SemanticField[MatchClaim]`` whose single derived struct column holds a
    large set of synchronous pairwise claims columnar, with shared
    provenance injected on read.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

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
    AgentType,
    ClaimType,
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    IdCoordinate,
    IdGenerator,
    Interval,
    NumberType,
    TimeUnit,
)
from timetoalign.core.fields import (
    SemanticField,
    StructField,
    register_value_projector,
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
from timetoalign.core.time import express_scalar_as, struct_to_rational
from timetoalign.core.timestamp import _format_coordinate

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.alignment.graph import MatchStamp
    from timetoalign.core.timestamp import ConversionMapsSpec

module_logger = logging.getLogger(__name__)

# Module-level ID generators
_anchor_id_generator = IdGenerator(scope="anchor")
_claim_id_generator = IdGenerator(scope="claim")


def _column_declares_exact(units: pa.Array) -> bool:
    """Whether any unit in a column declares an exact representation.

    ``TimeUnit`` is the single source of number policy, so the units a
    column stores are what say whether its ratio members are the value or a
    mirror of it. Read once per column rather than per row.
    """
    for name in pc.unique(units).to_pylist():
        if name is None:
            continue
        try:
            unit = TimeUnit(name)
        except ValueError:
            continue
        if unit.default_number_type is NumberType.fraction:
            return True
    return False


def _reset_anchor_ids() -> None:
    """Reset the module-level ID generator. For testing only."""
    global _anchor_id_generator
    _anchor_id_generator = IdGenerator(scope="anchor")


def _reset_claim_ids() -> None:
    """Reset the module-level ID generator. For testing only."""
    global _claim_id_generator
    _claim_id_generator = IdGenerator(scope="claim")


# region Agent


class Agent(BaseModel):
    """The author of a match claim — a human annotator or a software aligner.

    Attributes:
        name: A human-readable label for the agent (a person's name or a
            tool's name).
        type: Whether the agent is a human or a piece of software.
        identifier: A stable handle for the agent — a URI for a human (e.g.
            an ORCID), a version string for software (e.g. ``"v0.20.0"``).

    Examples:
        >>> Agent(name="J. Doe", type=AgentType.human, identifier="0000-0001")
        Agent(name='J. Doe', type="human", identifier='0000-0001')
        >>> Agent(name="dtw", type=AgentType.software, identifier="v2")
        Agent(name='dtw', type="software", identifier='v2')
    """

    model_config = ConfigDict(frozen=True)

    name: str
    type: AgentType  # noqa: A003 — distinguishes human / software author
    identifier: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (``type`` as its string value)."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """Deserialize from a dictionary (``type`` coerced to ``AgentType``)."""
        return cls.model_validate(data)


# endregion


# region MatchMetadata


class MatchMetadata(BaseModel):
    """Provenance information for a match claim.

    A match claim records who authored it and how confident that author is.

    Attributes:
        agent: The human or software :class:`Agent` that created this match.
        certainty: Confidence level in [0, 1]. 1.0 = certain.

    Examples:
        >>> meta = MatchMetadata(
        ...     agent=Agent(
        ...         name="analyst_jh",
        ...         type=AgentType.human,
        ...         identifier="https://orcid.org/0000-0001",
        ...     ),
        ...     certainty=1.0,
        ... )

        >>> meta = MatchMetadata(
        ...     agent=Agent(name="dtw", type=AgentType.software, identifier="v2"),
        ...     certainty=0.85,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    agent: Agent
    certainty: float = 1.0

    @field_validator("certainty")
    @classmethod
    def _validate_certainty(cls, value: float) -> float:
        """Validate certainty is in the closed range [0, 1]."""
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Certainty must be in [0, 1], got {value}")
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchMetadata":
        """Deserialize from dictionary (``agent`` rebuilt via ``Agent``)."""
        return cls.model_validate(data)


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


def _event_coordinate_value(raw: Any) -> int | float | Fraction:
    """Extract an event coordinate while preserving its native numeric value."""
    if isinstance(raw, Coordinate):
        return raw.value
    if isinstance(raw, dict):
        if "number_type" in raw:
            return coordinate_from_dict(raw).value
        return struct_to_rational(raw)
    return raw


class AlignmentAnchor(BaseModel):
    """A coordinate pair associating one coordinate on timeline A with one
    coordinate on timeline B.

    A neutral record with no claim semantics. AlignmentAnchors are value
    objects: two anchors with the same coordinates and timeline IDs are
    equivalent regardless of how they were created.

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
        ...     coordinate_a=Coordinate(100.0, TimeUnit.quarters),
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

    @field_validator("coordinate_a", "coordinate_b", mode="after")
    @classmethod
    def _express_on_declared_axis(cls, coordinate: Coordinate) -> Coordinate:
        """Write each anchored coordinate the way its axis writes numbers.

        An anchor coordinate is not a free-standing number: it is a position
        on a named timeline, so the axis decides its representation rather
        than whatever the caller happened to hand over. An event value read
        off storage arrives as the cell's exact side, and keeping that on a
        float-canonical axis is how the claim lane came to disagree with the
        matchstamp lane about one and the same position.

        An anchor knows its timelines only by id, so the declaration it can
        reach is the unit's own default. A bundle, which holds the timeline
        objects, refines this to each timeline's declared representation when
        the claim is read in its context.
        """
        return express_scalar_as(coordinate, coordinate.unit.default_number_type)

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
    "timeline_a_id",
    "timeline_b_id",
    "start_anchor",
    "end_anchor",
    "is_explicit",
    "is_synchronous",
    "metadata",
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
    """A claim that two events or coordinates on different timelines correspond.

    A MatchClaim always knows which two timelines it connects via top-level
    ``timeline_a_id`` and ``timeline_b_id`` fields. Anchors are only present
    for synchronous claims.

    Four cases:
        (a) ``from_events``: Two timed things on different timelines correspond.
        (b) ``from_projection``: An event is projected onto a timeline with no
            matching event.
        (c) ``nomatch``: An event has no equivalent on the other timeline.
        (d) ``implicit``: Implicit claim generated by MatchGraph group extension.

    Attributes:
        timeline_a_id: First timeline's unique identifier.
        timeline_b_id: Second timeline's unique identifier.
        start_anchor: AlignmentAnchor for event starts (None for non-synchronous).
        end_anchor: AlignmentAnchor for event ends (None for instants or
            non-synchronous claims).
        is_synchronous: True if temporally synchronous matches.
        is_explicit: True if directly claimed, False if inferred.
        metadata: Provenance information (agent, criteria, certainty).
        source_claim_id: For implicit claims, the ID of the claim that
            generated this one.
        event_a_id: ID of the event on timeline A (if known).
        event_a_name: Name/label of the event on timeline A (if known).
        event_b_id: ID of the event on timeline B (if known).
        event_b_name: Name/label of the event on timeline B (if known).
        source_coordinate: Unmatched source-side coordinate for a NOMATCH
            claim. None for synchronous claims, whose coordinate is held by
            the anchor instead.
        id: Unique identifier for this claim.

    Examples:
        >>> # Synchronous instant match via factory
        >>> claim = MatchClaim.from_events(
        ...     event_a={"id": "e001", "name": "Note C4", "start": 100.0},
        ...     tl_a_id="score:1",
        ...     event_b={"id": "e042", "name": "Note C4", "start": 45.5},
        ...     tl_b_id="recording:1",
        ...     unit_a=TimeUnit.quarters,
        ...     unit_b=TimeUnit.seconds,
        ... )
        >>> claim.event_a_id
        'e001'
        >>> claim.event_a_name
        'Note C4'

        >>> # Non-synchronous claim (no anchors)
        >>> claim = MatchClaim.nomatch(
        ...     event={"id": "orphan", "start": 100.0},
        ...     source_tl_id="score:1",
        ...     target_tl_id="recording:1",
        ...     unit=TimeUnit.quarters,
        ... )
        >>> claim.start_anchor is None
        True
    """

    model_config = ConfigDict(frozen=True)

    timeline_a_id: str
    timeline_b_id: str
    start_anchor: AlignmentAnchor | None = None
    end_anchor: AlignmentAnchor | None = None
    is_synchronous: bool = True
    is_explicit: bool = True
    metadata: MatchMetadata | None = None
    source_claim_id: str | None = None
    event_a_id: str | None = None
    event_a_name: str | None = None
    event_b_id: str | None = None
    event_b_name: str | None = None
    source_coordinate: Coordinate | None = None
    id: str = ""  # noqa: A003 — auto-generated claim id

    # Runtime association with the owning bundle. A private attr is excluded
    # from equality, hash, repr, and the derived Arrow schema.
    _bundle: Any = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_claim(self) -> "MatchClaim":
        """Auto-generate the ID and validate anchor presence vs synchrony."""
        # Auto-generate ID (frozen model: bypass via object.__setattr__).
        if not self.id:
            object.__setattr__(
                self,
                "id",
                _claim_id_generator.create(type_hint="MatchClaim"),
            )

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
        return self

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

        | Condition                          | Result        |
        |------------------------------------|---------------|
        | ``is_explicit`` is False           | ``implicit``  |
        | synchronous, both sides name event | ``event_match`` |
        | synchronous, one side names event  | ``projection``  |
        | synchronous, neither names event   | ``anchor``    |
        | non-synchronous, one side names it | ``nomatch``   |
        | non-synchronous, neither / both    | ``conceptual`` |
        """
        if not self.is_explicit:
            return ClaimType.implicit
        n_events = int(self.event_a_id is not None) + int(self.event_b_id is not None)
        if self.is_synchronous:
            if n_events == 2:
                return ClaimType.event_match
            if n_events == 1:
                return ClaimType.projection
            return ClaimType.anchor
        # non-synchronous (no anchors)
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
    def is_anonymous(self) -> bool:
        """Whether this anchors two coordinates with no event identity."""
        return self.claim_type is ClaimType.anchor

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
            Full stamp (default -- from graph, all groups)::

                >>> stamp = claim.get_matchstamp()  # uses claim's bundle
                >>> stamp.n_timelines
                23  # score + 22 performers at this coordinate

            Reduced stamp (two timelines only)::

                >>> stamp = claim.get_matchstamp(from_graph=False)
                >>> stamp.n_timelines
                2   # just the two timelines in this claim
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
            return effective_bundle.get_matchstamp_at(
                self.start_anchor.coordinate_a.value,
                self.timeline_a_id,
                conversion_maps=conversion_maps,
            )
        else:
            # Reduced stamp: just the two coordinates from this claim
            from timetoalign.alignment.graph import MatchStamp

            coords = {
                self.timeline_a_id: self.start_anchor.coordinate_a,
                self.timeline_b_id: self.start_anchor.coordinate_b,
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
        tl_a_id: str,
        event_b: dict[str, Any],
        tl_b_id: str,
        *,
        unit_a: TimeUnit | str,
        unit_b: TimeUnit | str,
        coord_key: str = "start",
        end_coord_key: str | None = None,
        is_synchronous: bool = True,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaim":
        """Case (a): Two timed things on different timelines correspond.

        Args:
            event_a: Event dict from timeline A (must contain ``coord_key``).
                May also contain ``id`` and ``name`` fields.
            tl_a_id: Timeline A's ID.
            event_b: Event dict from timeline B (must contain ``coord_key``).
                May also contain ``id`` and ``name`` fields.
            tl_b_id: Timeline B's ID.
            unit_a: Unit of timeline A's coordinates.
            unit_b: Unit of timeline B's coordinates.
            coord_key: Key for start coordinate in event dicts.
            end_coord_key: Key for end coordinate (creates interval match).
            is_synchronous: Whether events are temporally synchronous.
            metadata: Provenance information.

        Returns:
            A synchronous MatchClaim with 1 or 2 anchors and event info.
        """

        def _extract_coord(event: dict, key: str) -> int | float | Fraction:
            """Extract coordinate value from event dict.

            Handles both raw float values and structured coordinate dicts
            with a 'value' key.
            """
            return _event_coordinate_value(event[key])

        def _extract_str(event: dict, key: str) -> str | None:
            """Extract string value from event dict, returning None if absent."""
            val = event.get(key)
            return str(val) if val is not None else None

        start_anchor = AlignmentAnchor(
            timeline_a_id=tl_a_id,
            coordinate_a=Coordinate(_extract_coord(event_a, coord_key), unit_a),
            timeline_b_id=tl_b_id,
            coordinate_b=Coordinate(_extract_coord(event_b, coord_key), unit_b),
        )

        end_anchor = None
        if end_coord_key is not None:
            end_anchor = AlignmentAnchor(
                timeline_a_id=tl_a_id,
                coordinate_a=Coordinate(_extract_coord(event_a, end_coord_key), unit_a),
                timeline_b_id=tl_b_id,
                coordinate_b=Coordinate(_extract_coord(event_b, end_coord_key), unit_b),
            )

        return cls(
            timeline_a_id=tl_a_id,
            timeline_b_id=tl_b_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            is_synchronous=is_synchronous,
            metadata=metadata,
            event_a_id=_extract_str(event_a, "id"),
            event_a_name=_extract_str(event_a, "name"),
            event_b_id=_extract_str(event_b, "id"),
            event_b_name=_extract_str(event_b, "name"),
        )

    @classmethod
    def from_projection(
        cls,
        event: dict[str, Any],
        source_tl_id: str,
        target_tl_id: str,
        target_coord: CoordinateSpec,
        *,
        source_unit: TimeUnit | str,
        target_unit: TimeUnit | str | None = None,
        coord_key: str = "start",
        target_end_coord: CoordinateSpec | None = None,
        end_coord_key: str | None = None,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaim":
        """Case (b): An event is projected onto a timeline with no matching event.

        The source event has a coordinate; the target coordinate is specified
        explicitly (e.g., computed by interpolation or DTW).

        Args:
            event: Event dict from the source timeline. May contain ``id``
                and ``name`` fields.
            source_tl_id: Source timeline's ID.
            target_tl_id: Target timeline's ID.
            target_coord: Projected coordinate specification on the target timeline.
            source_unit: Unit of the source timeline's coordinates.
            target_unit: Unit of the target timeline's coordinates.
            coord_key: Key for start coordinate in event dict.
            target_end_coord: Projected end coordinate specification (creates
                interval match).
            end_coord_key: Key for end coordinate in event dict.
            metadata: Provenance information.

        Returns:
            A synchronous MatchClaim with 1 or 2 anchors and source event info.
        """
        target_coordinate = _coordinate_from_input(
            target_coord, target_unit, target_tl_id, "target_coord"
        )

        start_anchor = AlignmentAnchor(
            timeline_a_id=source_tl_id,
            coordinate_a=Coordinate(
                _event_coordinate_value(event[coord_key]), source_unit
            ),
            timeline_b_id=target_tl_id,
            coordinate_b=target_coordinate,
        )

        end_anchor = None
        if target_end_coord is not None and end_coord_key is not None:
            target_end_coordinate = _coordinate_from_input(
                target_end_coord, target_unit, target_tl_id, "target_end_coord"
            )
            end_anchor = AlignmentAnchor(
                timeline_a_id=source_tl_id,
                coordinate_a=Coordinate(
                    _event_coordinate_value(event[end_coord_key]), source_unit
                ),
                timeline_b_id=target_tl_id,
                coordinate_b=target_end_coordinate,
            )

        # Extract event info from source event (target has no event)
        event_a_id = str(event["id"]) if event.get("id") is not None else None
        event_a_name = str(event["name"]) if event.get("name") is not None else None

        return cls(
            timeline_a_id=source_tl_id,
            timeline_b_id=target_tl_id,
            start_anchor=start_anchor,
            end_anchor=end_anchor,
            is_synchronous=True,
            metadata=metadata,
            event_a_id=event_a_id,
            event_a_name=event_a_name,
            event_b_id=None,
            event_b_name=None,
        )

    @classmethod
    def nomatch(
        cls,
        event: dict[str, Any],
        source_tl_id: str,
        target_tl_id: str,
        *,
        unit: TimeUnit | str,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaim":
        """Case (c): An event has no equivalent on the other timeline.

        Creates a non-synchronous claim with no anchors (NOMATCH sentinel).

        Args:
            event: Event dict from the source timeline. May contain ``id``
                and ``name`` fields.
            source_tl_id: Source timeline's ID.
            target_tl_id: Target timeline's ID.
            unit: Unit of the source timeline's coordinate.
            metadata: Provenance information.

        Returns:
            A non-synchronous MatchClaim with no anchors but source event
            info. The source-side coordinate (``event["start"]``) is
            preserved as ``source_coordinate`` for display.
        """
        # Extract event info from source event
        event_a_id = str(event["id"]) if event.get("id") is not None else None
        event_a_name = str(event["name"]) if event.get("name") is not None else None
        source_coordinate = (
            Coordinate(_event_coordinate_value(event["start"]), unit)
            if event.get("start") is not None
            else None
        )

        return cls(
            timeline_a_id=source_tl_id,
            timeline_b_id=target_tl_id,
            start_anchor=None,
            end_anchor=None,
            is_synchronous=False,
            is_explicit=True,
            metadata=metadata,
            event_a_id=event_a_id,
            event_a_name=event_a_name,
            event_b_id=None,
            event_b_name=None,
            source_coordinate=source_coordinate,
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
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaim":
        """Case (d): Implicit claim generated by MatchGraph group extension.

        Args:
            tl_a_id: First timeline's ID.
            coord_a: Coordinate specification on timeline A.
            tl_b_id: Second timeline's ID.
            coord_b: Coordinate specification on timeline B.
            unit_a: Unit of timeline A's coordinate.
            unit_b: Unit of timeline B's coordinate.
            source_claim: The claim that generated this one.
            metadata: Provenance information.

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
            metadata=metadata,
        )

    # endregion

    @field_serializer("source_coordinate")
    def _serialize_source_coordinate(
        self, coordinate: Coordinate | None
    ) -> dict[str, Any] | None:
        """Emit the NOMATCH coordinate through the shared JSON representation."""
        return None if coordinate is None else _coordinate_to_dict(coordinate)

    @field_validator("source_coordinate", mode="before")
    @classmethod
    def _parse_source_coordinate(cls, value: Any) -> Any:
        """Accept the serialized coordinate dict as well as a Coordinate."""
        if isinstance(value, dict) and "unit" in value:
            return _coordinate_from_dict(value)
        return value

    @field_validator("source_coordinate", mode="after")
    @classmethod
    def _express_source_on_declared_axis(
        cls, coordinate: Coordinate | None
    ) -> Coordinate | None:
        """Write the unmatched coordinate the way its axis writes numbers.

        A NOMATCH claim carries its position outside an anchor, so it needs
        the axis rule stated separately -- and it was missed. An event value
        read off storage arrives as the cell's exact side, which on a
        float-canonical axis turns a plain ``164.3`` seconds into a
        sixteen-digit ratio in the claim's own repr. Same rule, same reason
        as :class:`AlignmentAnchor`: with only timeline ids in hand the
        reachable declaration is the unit's default.
        """
        if coordinate is None:
            return None
        return express_scalar_as(coordinate, coordinate.unit.default_number_type)

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
        for slot in ("start_anchor", "end_anchor", "metadata"):
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
        materialise a scalar from the backing struct column.  The row dict
        carries the nested ``start_anchor`` / ``end_anchor`` / ``metadata``
        sub-dicts (or ``None`` for absent slots), so the shape is exactly the
        one :meth:`from_dict` already accepts.

        Args:
            row: The ``.as_py()`` dict of one struct-column element.

        Returns:
            A reconstructed :class:`MatchClaim`.
        """
        payload = dict(row)
        for anchor_name in ("start_anchor", "end_anchor"):
            anchor = payload.get(anchor_name)
            if not isinstance(anchor, dict):
                continue
            restored = dict(anchor)
            for coordinate_name in ("coordinate_a", "coordinate_b"):
                coordinate = restored.get(coordinate_name)
                if not isinstance(coordinate, dict):
                    continue
                numerator = coordinate.get("numerator")
                denominator = coordinate.get("denominator")
                value = (
                    Fraction(numerator, denominator)
                    if numerator is not None and denominator is not None
                    else coordinate["value"]
                )
                restored[coordinate_name] = Coordinate(value, coordinate["unit"])
            payload[anchor_name] = restored
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

        Shows claim type, timelines with coordinates, events, and metadata.

        Examples:
            >>> print(claim)
            MatchClaim (synchronous, interval)
              Timeline A:  score:clt1  [0.0 -- 0.5]
              Event A:     e001 "Intro"
              Timeline B:  perf:dlt1   [0 -- 261]
              Event B:     e042 "Intro"
              Metadata:    agent=partitura, certainty=1.0
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

        # Metadata
        if self.metadata is not None:
            meta_parts = [f"agent={self.metadata.agent.name}"]
            if self.metadata.certainty < 1.0:
                meta_parts.append(f"certainty={self.metadata.certainty}")
            lines.append(f"  Metadata:    {', '.join(meta_parts)}")

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
        else:  # projection, anchor, implicit — synchronous with anchors
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

        # Metadata row
        if self.metadata is not None:
            meta_parts = [
                html_mod.escape(f"agent={self.metadata.agent.name}"),
            ]
            if self.metadata.certainty < 1.0:
                meta_parts.append(f"certainty={self.metadata.certainty}")
            rows.append(
                f"<tr style='color: #666;'><td>Metadata</td>"
                f"<td colspan='2'>{', '.join(meta_parts)}</td></tr>"
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
    ]
)


def _claim_coordinate_projector(
    _model_cls: type[BaseModel], name: str, _info: object
) -> list[pa.Field]:
    """Store a claim coordinate's exact value and unit in each row."""
    return [pa.field(name, _CLAIM_COORDINATE_TYPE, nullable=True)]


register_value_projector(AlignmentAnchor, "coordinate_a", _claim_coordinate_projector)
register_value_projector(AlignmentAnchor, "coordinate_b", _claim_coordinate_projector)
register_value_projector(MatchClaim, "source_coordinate", _claim_coordinate_projector)


class MatchClaimField(SemanticField[MatchClaim]):
    """A ``SemanticField[MatchClaim]`` columnar store for pairwise claims.

    ``MatchClaimField`` is the Field paired with the :class:`MatchClaim`
    scalar.  It holds a large set of synchronous pairwise alignment
    claims (millions of rows) in a single derived struct column instead of one
    frozen :class:`MatchClaim` per claim.  Individual claims are materialised
    lazily, only when a row is indexed or iterated.

    Like every paired Field, its struct schema is derived from the scalar by
    :func:`~timetoalign.core.fields.derive_arrow_struct` — so the inner column
    type is exactly ``MatchClaimField.pa_schema``.  Shared provenance is held
    once at field level as a single :class:`MatchMetadata` (or ``None``) and
    **injected on read**, mirroring how
    :class:`~timetoalign.core.time.CoordinateField` carries its ``unit``
    outside the data: the struct column's ``metadata`` sub-field is left null
    in bulk, and :meth:`__getitem__` injects the field-level metadata into each
    materialised :class:`MatchClaim`.  This keeps the store compact (one struct
    column, no per-row metadata) while remaining a genuine SemanticField.

    Every row represents a synchronous claim with a start anchor and an
    optional end anchor. NOMATCH claims remain outside this store because
    they carry no coordinate pair to query.

    Attributes:
        table: The backing single-column :class:`pyarrow.Table` (read-only
            property) whose one column is the derived ``MatchClaim`` struct.
        metadata: Shared :class:`MatchMetadata` (or ``None``) applied to every
            materialised claim.

    Examples:
        >>> agent = Agent(name="dtw", type=AgentType.software, identifier="v1")
        >>> meta = MatchMetadata(agent=agent)
        >>> field = MatchClaimField.from_columns(
        ...     timeline_a_ids=["A", "A", "B"],
        ...     timeline_b_ids=["B", "C", "C"],
        ...     coordinate_a=[0.0, 0.0, 1.0],
        ...     coordinate_b=[10.0, 20.0, 21.0],
        ...     unit_a=TimeUnit.quarters,
        ...     unit_b=TimeUnit.seconds,
        ...     metadata=meta,
        ... )
        >>> len(field)
        3
        >>> field[0].timeline_a_id, field[0].timeline_b_id
        ('A', 'B')
    """

    #: The name of the single struct column backing every instance.
    _COLUMN_NAME: ClassVar[str] = "match_claim"

    def __init__(
        self,
        raw: "StructField | None" = None,
        *,
        source_fields: "str | dict[str, Any] | None" = None,
        name: str | None = None,
        metadata: MatchMetadata | None = None,
    ) -> None:
        """Wrap a derived ``MatchClaim`` struct field, carrying shared metadata.

        Prefer the :meth:`from_columns` / :meth:`from_claims` / :meth:`from_dict`
        classmethods; this constructor mirrors the base ``SemanticField`` so
        blueprint construction keeps working, plus a field-level ``metadata``.
        """
        super().__init__(raw, source_fields=source_fields, name=name)
        self._metadata: MatchMetadata | None = metadata

    # ── construction ──────────────────────────────────────────────────────

    @classmethod
    def from_columns(
        cls,
        timeline_a_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        timeline_b_ids: Sequence[str] | pa.Array | pa.ChunkedArray,
        coordinate_a: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
        coordinate_b: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
        *,
        unit_a: TimeUnit | str,
        unit_b: TimeUnit | str,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaimField":
        """Build a field directly from parallel columns (the vectorized path).

        This is the constructor a bulk producer (e.g. an alignment loader)
        uses. It assembles the derived ``MatchClaim`` struct column straight
        from the parallel inputs with :meth:`pyarrow.StructArray.from_arrays`
        and **never** materialises a single :class:`MatchClaim` Python object,
        so it scales to millions of rows.

        Each row is a synchronous instant claim: ``timeline_a_id`` /
        ``timeline_b_id`` and the ``start_anchor`` sub-struct are filled from
        the inputs; ``is_synchronous`` and ``is_explicit`` are ``True``;
        ``end_anchor`` / event ids / names / ``source_coordinate`` /
        ``metadata`` / ``source_claim_id`` / ``id`` are left null (the shared
        metadata is injected on read, the id is generated on materialise).

        Args:
            timeline_a_ids: Timeline-A ids, one per claim.
            timeline_b_ids: Timeline-B ids, one per claim.
            coordinate_a: Coordinate value on timeline A, one per claim.
            coordinate_b: Coordinate value on timeline B, one per claim.
            unit_a: Unit applied to every coordinate on timeline A.
            unit_b: Unit applied to every coordinate on timeline B.
            metadata: Shared provenance for every claim in the field.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If the four inputs do not all have the same length.
        """
        tl_a = cls._string_array(timeline_a_ids)
        tl_b = cls._string_array(timeline_b_ids)
        coord_a = cls._coordinate_struct_array(coordinate_a, unit_a)
        coord_b = cls._coordinate_struct_array(coordinate_b, unit_b)

        lengths = {len(tl_a), len(tl_b), len(coord_a), len(coord_b)}
        if len(lengths) != 1:
            raise ValueError(
                "from_columns requires four equal-length columns; got lengths "
                f"timeline_a_id={len(tl_a)}, timeline_b_id={len(tl_b)}, "
                f"coordinate_a={len(coord_a)}, coordinate_b={len(coord_b)}."
            )
        n = next(iter(lengths))

        struct_array = cls._build_struct_array(n, tl_a, tl_b, coord_a, coord_b)
        return cls._from_struct_array(struct_array, metadata=metadata)

    @classmethod
    def from_claims(
        cls,
        claims: list[MatchClaim],
        *,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaimField":
        """Build a field from existing :class:`MatchClaim` objects.

        Every claim must be synchronous and carry a start anchor. Interval
        claims retain their end anchors.

        If ``metadata`` is ``None`` and all claims share one identical
        :class:`MatchMetadata` (by equality), that metadata is adopted as the
        field-level provenance; otherwise the field's metadata stays ``None``
        (no per-row metadata is ever stored).

        Args:
            claims: The claims to store. Coordinates are pulled from each
                claim's ``start_anchor``.
            metadata: Explicit field-level provenance. When provided it
                overrides any per-claim metadata inference.

        Returns:
            A new :class:`MatchClaimField`.

        Raises:
            ValueError: If any claim is non-synchronous (NOMATCH) or lacks a
                start anchor.
        """
        for index, claim in enumerate(claims):
            if not claim.is_synchronous or claim.start_anchor is None:
                raise ValueError(
                    f"MatchClaimField holds synchronous claims only; "
                    f"claim at index {index} is non-synchronous (NOMATCH) and "
                    f"is out of scope."
                )
        if metadata is None:
            metadata = cls._common_metadata(claims)

        tl_a = cls._string_array([claim.timeline_a_id for claim in claims])
        tl_b = cls._string_array([claim.timeline_b_id for claim in claims])
        coord_a = cls._coordinate_struct_array(
            [claim.start_anchor.coordinate_a.value for claim in claims],
            [claim.start_anchor.coordinate_a.unit for claim in claims],
        )
        coord_b = cls._coordinate_struct_array(
            [claim.start_anchor.coordinate_b.value for claim in claims],
            [claim.start_anchor.coordinate_b.unit for claim in claims],
        )
        end_coord_a = cls._coordinate_struct_array(
            [
                (
                    None
                    if claim.end_anchor is None
                    else claim.end_anchor.coordinate_a.value
                )
                for claim in claims
            ],
            [
                (
                    claim.start_anchor.coordinate_a.unit
                    if claim.end_anchor is None
                    else claim.end_anchor.coordinate_a.unit
                )
                for claim in claims
            ],
        )
        end_coord_b = cls._coordinate_struct_array(
            [
                (
                    None
                    if claim.end_anchor is None
                    else claim.end_anchor.coordinate_b.value
                )
                for claim in claims
            ],
            [
                (
                    claim.start_anchor.coordinate_b.unit
                    if claim.end_anchor is None
                    else claim.end_anchor.coordinate_b.unit
                )
                for claim in claims
            ],
        )
        struct_array = cls._build_struct_array(
            len(claims),
            tl_a,
            tl_b,
            coord_a,
            coord_b,
            end_coord_a,
            end_coord_b,
        )
        return cls._from_struct_array(struct_array, metadata=metadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchClaimField":
        """Deserialize a field from a plain dictionary.

        Inverse of :meth:`to_dict`.

        Args:
            data: A mapping with the four column lists and an optional
                ``metadata`` dict.

        Returns:
            A new :class:`MatchClaimField`.
        """
        metadata_dict = data.get("metadata")
        metadata = (
            MatchMetadata.from_dict(metadata_dict)
            if metadata_dict is not None
            else None
        )
        tl_a = cls._string_array(data["timeline_a_id"])
        tl_b = cls._string_array(data["timeline_b_id"])
        coord_a = cls._coordinate_struct_array(data["coordinate_a"], data["unit_a"])
        coord_b = cls._coordinate_struct_array(data["coordinate_b"], data["unit_b"])
        end_values_a = data["end_coordinate_a"]
        end_values_b = data["end_coordinate_b"]
        end_units_a = data["end_unit_a"]
        end_units_b = data["end_unit_b"]
        end_coord_a = cls._coordinate_struct_array(end_values_a, end_units_a)
        end_coord_b = cls._coordinate_struct_array(end_values_b, end_units_b)
        struct_array = cls._build_struct_array(
            len(tl_a),
            tl_a,
            tl_b,
            coord_a,
            coord_b,
            end_coord_a,
            end_coord_b,
        )
        return cls._from_struct_array(struct_array, metadata=metadata)

    @classmethod
    def _from_struct_array(
        cls,
        struct_array: pa.StructArray,
        *,
        metadata: MatchMetadata | None = None,
    ) -> "MatchClaimField":
        """Wrap a derived ``MatchClaim`` struct array as a live field."""
        pa_field = pa.field(cls._COLUMN_NAME, cls.pa_schema)
        raw = StructField(struct_array, pa_field)
        return cls(raw, metadata=metadata)

    @classmethod
    def _build_struct_array(
        cls,
        n: int,
        timeline_a_id: pa.Array,
        timeline_b_id: pa.Array,
        coordinate_a: pa.StructArray,
        coordinate_b: pa.StructArray,
        end_coordinate_a: pa.StructArray | None = None,
        end_coordinate_b: pa.StructArray | None = None,
    ) -> pa.StructArray:
        """Assemble the derived ``MatchClaim`` struct array, vectorized.

        The struct's child arrays are produced in the exact field order of
        :attr:`pa_schema` (so the resulting type equals ``pa_schema``); unused
        slots are all-null arrays of the right type.
        """
        schema = cls.pa_schema
        start_anchor = cls._build_anchor_array(
            timeline_a_id, timeline_b_id, coordinate_a, coordinate_b
        )
        true_array = pa.array([True] * n, type=pa.bool_())

        per_field: dict[str, pa.Array] = {
            "timeline_a_id": timeline_a_id,
            "timeline_b_id": timeline_b_id,
            "start_anchor": start_anchor,
            "is_synchronous": true_array,
            "is_explicit": true_array,
        }
        if end_coordinate_a is not None and end_coordinate_b is not None:
            per_field["end_anchor"] = cls._build_anchor_array(
                timeline_a_id,
                timeline_b_id,
                end_coordinate_a,
                end_coordinate_b,
                nullable=True,
            )
        arrays: list[pa.Array] = []
        for sub_field in schema:
            arrays.append(
                per_field.get(sub_field.name) or pa.nulls(n, type=sub_field.type)
            )
        return pa.StructArray.from_arrays(arrays, fields=list(schema))

    @classmethod
    def _build_anchor_array(
        cls,
        timeline_a_id: pa.Array,
        timeline_b_id: pa.Array,
        coordinate_a: pa.StructArray,
        coordinate_b: pa.StructArray,
        *,
        nullable: bool = False,
    ) -> pa.StructArray:
        """Build one anchor struct from parallel typed columns."""
        anchor_type = cls.pa_schema.field("start_anchor").type
        sources: dict[str, pa.Array] = {
            "timeline_a_id": timeline_a_id,
            "coordinate_a": coordinate_a,
            "timeline_b_id": timeline_b_id,
            "coordinate_b": coordinate_b,
        }
        children = [sources[field.name] for field in anchor_type]
        return pa.StructArray.from_arrays(
            children,
            fields=list(anchor_type),
            mask=pc.is_null(coordinate_a) if nullable else None,
        )

    @staticmethod
    def _string_array(
        values: Sequence[str] | pa.Array | pa.ChunkedArray,
    ) -> pa.Array:
        """Coerce a sequence/array of ids into a plain ``string`` array."""
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if isinstance(values, pa.Array):
            return pc.cast(values, pa.string())
        return pa.array(list(values), type=pa.string())

    @staticmethod
    def _coordinate_struct_array(
        values: Sequence[int | float | Fraction] | pa.Array | pa.ChunkedArray,
        units: TimeUnit | str | Sequence[TimeUnit | str],
    ) -> pa.StructArray:
        """Build exact-value, per-row-unit coordinate storage."""
        if isinstance(values, pa.ChunkedArray):
            values = values.combine_chunks()
        if isinstance(values, pa.Array):
            raw_values = values.to_pylist()
        else:
            raw_values = list(values)

        if isinstance(units, (TimeUnit, str)):
            raw_units = [units] * len(raw_values)
        else:
            raw_units = list(units)
        if len(raw_units) != len(raw_values):
            raise ValueError("Coordinate values and units must have equal length.")

        if isinstance(units, (TimeUnit, str)):
            try:
                numeric_array = pa.array(raw_values)
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                numeric_array = None
            if numeric_array is not None and (
                pa.types.is_integer(numeric_array.type)
                or pa.types.is_floating(numeric_array.type)
            ):
                unit = units if isinstance(units, TimeUnit) else TimeUnit(units)
                is_integer = pa.types.is_integer(numeric_array.type)
                return pa.StructArray.from_arrays(
                    [
                        pc.cast(numeric_array, pa.float64()),
                        (
                            pc.cast(numeric_array, pa.int64())
                            if is_integer
                            else pa.nulls(len(numeric_array), pa.int64())
                        ),
                        (
                            pa.array([1] * len(numeric_array), pa.int64())
                            if is_integer
                            else pa.nulls(len(numeric_array), pa.int64())
                        ),
                        pa.array([unit.value] * len(numeric_array), pa.string()),
                    ],
                    fields=list(_CLAIM_COORDINATE_TYPE),
                    mask=pc.is_null(numeric_array),
                )

        float_values: list[float | None] = []
        numerators: list[int | None] = []
        denominators: list[int | None] = []
        unit_values: list[str | None] = []
        null_mask: list[bool] = []
        for raw, raw_unit in zip(raw_values, raw_units, strict=True):
            if raw is None:
                float_values.append(None)
                numerators.append(None)
                denominators.append(None)
                unit_values.append(None)
                null_mask.append(True)
                continue
            if hasattr(raw, "item"):
                raw = raw.item()
            float_values.append(float(raw))
            if isinstance(raw, Fraction):
                numerators.append(raw.numerator)
                denominators.append(raw.denominator)
            elif isinstance(raw, int) and not isinstance(raw, bool):
                numerators.append(raw)
                denominators.append(1)
            else:
                numerators.append(None)
                denominators.append(None)
            unit = raw_unit if isinstance(raw_unit, TimeUnit) else TimeUnit(raw_unit)
            unit_values.append(unit.value)
            null_mask.append(False)

        return pa.StructArray.from_arrays(
            [
                pa.array(float_values, type=pa.float64()),
                pa.array(numerators, type=pa.int64()),
                pa.array(denominators, type=pa.int64()),
                pa.array(unit_values, type=pa.string()),
            ],
            fields=list(_CLAIM_COORDINATE_TYPE),
            mask=pa.array(null_mask, type=pa.bool_()),
        )

    @staticmethod
    def _common_metadata(claims: list[MatchClaim]) -> MatchMetadata | None:
        """Return the shared metadata if all claims agree, else ``None``."""
        if not claims:
            return None
        first = claims[0].metadata
        if first is None:
            return None
        for claim in claims[1:]:
            if claim.metadata != first:
                return None
        return first

    # ── access ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        """Number of claims (rows) in the field."""
        data = self._raw.data
        return 0 if data is None else len(data)

    def __getitem__(self, i: int) -> MatchClaim:
        """Materialise the claim at row ``i`` as a :class:`MatchClaim`.

        Supports negative indexing.  The field-level shared metadata is
        injected into the materialised claim (the struct column carries no
        per-row metadata).

        Args:
            i: Zero-based row index (negative counts from the end).

        Returns:
            A synchronous :class:`MatchClaim` carrying the field's shared
            metadata.

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
        if self._metadata is not None:
            claim = claim.model_copy(update={"metadata": self._metadata})
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
        """The backing single-column :class:`pyarrow.Table`.

        The one column is the derived :class:`MatchClaim` struct
        (``MatchClaimField.pa_schema``).
        """
        return pa.table({self._COLUMN_NAME: self._raw.data})

    @property
    def metadata(self) -> MatchMetadata | None:
        """Shared provenance applied to every materialised claim."""
        return self._metadata

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

    def connecting(self, timeline_id: str) -> "MatchClaimField":
        """Return the claims that involve ``timeline_id`` on either side.

        The filter is a vectorized boolean mask over the struct's id
        sub-fields; no claim is materialised. The shared metadata carries
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
        No claim is materialised and the rest of the field is never touched;
        the shared metadata carries over.

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
            start = self._coordinate_comparison_mask(
                "start_anchor", side, query, "equal"
            )
            contained = pc.and_(
                self._coordinate_comparison_mask(
                    "start_anchor", side, query, "less_equal"
                ),
                self._coordinate_comparison_mask(
                    "end_anchor", side, query, "greater_equal"
                ),
            )
            relevant = pc.if_else(
                pc.is_valid(data.field("end_anchor")), contained, start
            )
            masks.append(pc.and_(side_id, relevant))
        mask = pc.or_(*masks)
        # Empty-mask short-circuit: skip building a filtered sub-field.
        if pc.sum(pc.cast(mask, "int64")).as_py() == 0:
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
    ) -> "MatchClaimField":
        """Return a filtered view following the unified filter API.

        Every criterion is a vectorized boolean mask over the struct column;
        no claim is materialised and the shared metadata carries over.  All
        non-default criteria are combined with logical AND.

        ``include_domains`` / ``include_units`` are deliberately absent: they
        need per-timeline metadata (domain, unit) that this store does not
        hold.  A caller that owns that metadata — an
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
            within: Keep rows whose **both** ids are in this set.  This is the
                primitive behind domain- and unit-restricted queries.
            synchronous_only: A no-op — every row in this store is a
                synchronous claim by class invariant.
            nomatch_only: Returns an **empty** field, for the same reason.

        Returns:
            A new :class:`MatchClaimField`.  With no criteria set, a copy of
            the whole field is returned.

        Raises:
            ValueError: If ``synchronous_only`` and ``nomatch_only`` are both
                set.
        """
        if synchronous_only and nomatch_only:
            raise ValueError("synchronous_only and nomatch_only are mutually exclusive")
        if nomatch_only:
            return self._empty()

        data = self._raw.data
        masks: list[pa.Array] = []
        if data is not None and len(data) > 0:
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

        if not masks:
            return self._from_struct_array(data, metadata=self._metadata)
        mask = masks[0]
        for extra in masks[1:]:
            mask = pc.and_(mask, extra)
        return self._filtered(mask)

    def connects_groups(self, group_a_ids: set[str], group_b_ids: set[str]) -> bool:
        """Whether any claim spans the two given sets of timeline ids.

        True when at least one row has one side in ``group_a_ids`` and the
        other side in ``group_b_ids``.  The test is a vectorized mask reduced
        with :func:`pyarrow.compute.any`; no claim is materialised, so it
        answers commensurability over a million-row field in one pass.

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
        return pc.any(mask).as_py() is True

    def max_coordinate(self, timeline_id: str) -> float | None:
        """The largest coordinate this field carries for ``timeline_id``.

        Both anchor sides are scanned vectorized with :func:`pyarrow.compute.max`;
        no claim is materialised.

        Args:
            timeline_id: The timeline to look up.

        Returns:
            The maximum coordinate value carried for that timeline, or
            ``None`` if the timeline does not appear in this field.
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
        the two anchor coordinates of every row, as four parallel Python
        lists pulled straight off the Arrow columns.  Materialising the rows
        as :class:`MatchClaim` objects instead costs orders of magnitude more
        for the same information.

        Each coordinate is read in the representation its own ``unit``
        declares, so an anchor authored at ``5/3`` quarters comes back as
        ``Fraction(5, 3)`` rather than as the double nearest to it.

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

    def _anchor_coordinates(self, side: str) -> list[CoordinateValue]:
        """Read one anchor column in the representation its unit declares.

        A column whose units are all int- or float-canonical is handed over
        as the float member, which is what those units store; a column
        carrying an exact unit is read from its ``numerator``/``denominator``
        members, because on such a unit the ratio IS the value and the double
        beside it is only a mirror. The choice is made once per column from
        the stored units, never guessed from a row's digits.
        """
        anchor = self._raw.data.field("start_anchor").field(f"coordinate_{side}")
        values = anchor.field("value").to_pylist()
        if not _column_declares_exact(anchor.field("unit")):
            return values
        numerators = anchor.field("numerator").to_pylist()
        denominators = anchor.field("denominator").to_pylist()
        return [
            (
                value
                if numerator is None or denominator is None
                else Fraction(numerator, denominator)
            )
            for value, numerator, denominator in zip(
                values, numerators, denominators, strict=True
            )
        ]

    def to_claims(self) -> list[MatchClaim]:
        """Materialise every row into a list of :class:`MatchClaim` objects.

        This is ``O(n)`` and defeats the columnar purpose for large sets; it
        is provided for convenience and round-tripping, not for hot paths.

        Returns:
            A list of synchronous claims, one per row.
        """
        return [self[i] for i in range(len(self))]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary (inverse of :meth:`from_dict`).

        The id and coordinate columns are emitted as plain Python lists (read
        directly from the struct sub-fields); the metadata is emitted via
        :meth:`MatchMetadata.to_dict` (or ``None``).

        Returns:
            A round-trippable mapping.
        """
        data = self._raw.data
        if data is None:
            empty: list[Any] = []
            columns = {
                "timeline_a_id": empty,
                "timeline_b_id": empty,
                "coordinate_a": empty,
                "coordinate_b": empty,
                "unit_a": empty,
                "unit_b": empty,
                "end_coordinate_a": empty,
                "end_coordinate_b": empty,
                "end_unit_a": empty,
                "end_unit_b": empty,
            }
        else:
            anchor = data.field("start_anchor")
            end_anchor = data.field("end_anchor")
            stored_a = anchor.field("coordinate_a").to_pylist()
            stored_b = anchor.field("coordinate_b").to_pylist()
            stored_end_a = end_anchor.field("coordinate_a").to_pylist()
            stored_end_b = end_anchor.field("coordinate_b").to_pylist()

            def storage_value(value: dict[str, Any]) -> CoordinateValue:
                """Read one known claim-coordinate storage cell."""
                numerator = value["numerator"]
                denominator = value["denominator"]
                if numerator is not None and denominator is not None:
                    return Fraction(numerator, denominator)
                return value["value"]

            columns = {
                "timeline_a_id": data.field("timeline_a_id").to_pylist(),
                "timeline_b_id": data.field("timeline_b_id").to_pylist(),
                "coordinate_a": [storage_value(value) for value in stored_a],
                "coordinate_b": [storage_value(value) for value in stored_b],
                "unit_a": [value["unit"] for value in stored_a],
                "unit_b": [value["unit"] for value in stored_b],
                "end_coordinate_a": [
                    None if value is None else storage_value(value)
                    for value in stored_end_a
                ],
                "end_coordinate_b": [
                    None if value is None else storage_value(value)
                    for value in stored_end_b
                ],
                "end_unit_a": [
                    None if value is None else value["unit"] for value in stored_end_a
                ],
                "end_unit_b": [
                    None if value is None else value["unit"] for value in stored_end_b
                ],
            }
        columns["metadata"] = (
            self._metadata.to_dict() if self._metadata is not None else None
        )
        return columns

    def _empty(self) -> "MatchClaimField":
        """Return a zero-row field of this schema, keeping the metadata."""
        data = self._raw.data
        empty = pa.array([], type=self.pa_schema) if data is None else data.slice(0, 0)
        return self._from_struct_array(empty, metadata=self._metadata)

    def _filtered(self, mask: pa.Array) -> "MatchClaimField":
        """Return a new field with the struct array filtered by ``mask``."""
        filtered = self._raw.data.filter(mask)
        return self._from_struct_array(filtered, metadata=self._metadata)

    def _anchor_values(self, side: str) -> pa.Array:
        """The stored anchor coordinate values on ``side`` (``"a"`` / ``"b"``)."""
        anchor = self._raw.data.field("start_anchor")
        return anchor.field(f"coordinate_{side}").field("value")

    def _coordinate_comparison_mask(
        self,
        anchor_name: str,
        side: str,
        query: Coordinate,
        comparison: str,
    ) -> pa.Array:
        """Compare one stored coordinate column to an exact typed query."""
        coordinate = self._raw.data.field(anchor_name).field(f"coordinate_{side}")
        if query.number_type in (NumberType.int, NumberType.fraction):
            exact = Fraction(query.value)
            decimal = pa.decimal256(37, 0)
            stored_numerator = pc.cast(coordinate.field("numerator"), decimal)
            stored_denominator = pc.cast(coordinate.field("denominator"), decimal)
            left = pc.multiply(
                stored_numerator,
                pa.scalar(Decimal(exact.denominator), type=decimal),
            )
            right = pc.multiply(
                stored_denominator,
                pa.scalar(Decimal(exact.numerator), type=decimal),
            )
        else:
            left = coordinate.field("value")
            right = pa.scalar(float(query.value))
        operations = {
            "equal": pc.equal,
            "less_equal": pc.less_equal,
            "greater_equal": pc.greater_equal,
        }
        return operations[comparison](left, right)

    @staticmethod
    def _id_value_set(timeline_ids: set[str]) -> pa.Array:
        """A deterministic ``string`` value set for :func:`pyarrow.compute.is_in`."""
        return pa.array(sorted(timeline_ids), type=pa.string())

    def _involves_mask(self, timeline_id: str) -> pa.Array:
        """Boolean mask: rows where either id sub-field equals ``timeline_id``."""
        data = self._raw.data
        return pc.or_(
            pc.equal(data.field("timeline_a_id"), timeline_id),
            pc.equal(data.field("timeline_b_id"), timeline_id),
        )

    def _involves_any_mask(self, timeline_ids: set[str]) -> pa.Array:
        """Boolean mask: rows involving any id in ``timeline_ids``."""
        data = self._raw.data
        wanted = self._id_value_set(timeline_ids)
        return pc.or_(
            pc.is_in(data.field("timeline_a_id"), value_set=wanted),
            pc.is_in(data.field("timeline_b_id"), value_set=wanted),
        )

    def _within_mask(self, timeline_ids: set[str]) -> pa.Array:
        """Boolean mask: rows whose BOTH id sub-fields are in ``timeline_ids``."""
        data = self._raw.data
        wanted = self._id_value_set(timeline_ids)
        return pc.and_(
            pc.is_in(data.field("timeline_a_id"), value_set=wanted),
            pc.is_in(data.field("timeline_b_id"), value_set=wanted),
        )

    def _between_mask(self, timeline_a_id: str, timeline_b_id: str) -> pa.Array:
        """Boolean mask: rows connecting exactly the two ids, either way round."""
        data = self._raw.data
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
            rows.append(
                f"<tr><td>{i}</td>"
                f"<td>{html_mod.escape(claim.timeline_a_id)}</td>"
                f"<td>{html_mod.escape(_format_coordinate(anchor.coordinate_a))}</td>"
                f"<td>{html_mod.escape(claim.timeline_b_id)}</td>"
                f"<td>{html_mod.escape(_format_coordinate(anchor.coordinate_b))}</td></tr>"
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
