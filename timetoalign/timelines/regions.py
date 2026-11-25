"""Region: an identified TimeInterval for timeline partitioning.

This module provides the Region class, which represents a part of a
timeline defined by a TimeInterval. Regions are NOT timelines themselves -
they cannot hold events or C-maps.

A region is identified by its ``id`` and displayed by its optional ``name``.
Use ``Timeline.create_child_from_region()`` to create a Child timeline
from a Region.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Any, ClassVar

from timetoshare import ATOMIC_SECTION, Catalogue, expand_iri, same_pair

from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.ids import _IdentifiedElement, quote_id
from timetoalign.core.provenance import BeliefValue, Provenance
from timetoalign.core.time import Duration, Interval

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_meta,
    read_name,
    require_object,
)
from .statements import (
    STATE_KEYS,
    NodePath,
    StatementsMixin,
    read_statement_state,
    renamed_state,
)

# region Region


@dataclass(frozen=True)
class Region(_IdentifiedElement, StatementsMixin):
    """An identified part of a timeline defined by a TimeInterval.

    IMPORTANT: Regions are NOT timelines. They cannot hold events or C-maps.
    Use ``Timeline.create_child_from_region(region_id)`` to create a Child
    timeline from a Region.

    A region is frozen: :meth:`with_statement` and its siblings return a
    copy holding other statements (see
    :mod:`~timetoalign.timelines.statements`), and a timeline stores a
    changed copy in the region's place with
    :meth:`Timeline.replace_region <timetoalign.timelines.Timeline.replace_region>`.

    Attributes:
        start: Start coordinate of the region.
        end: End coordinate of the region.
        name: Optional display name (e.g., "Chorus"); names may repeat.
        meta: Additional metadata (e.g., traversal order labels).
        id: The region's id; minted as ``region_{N}`` when not given.

    Examples:
        >>> from timetoalign import Coordinate, TimeUnit
        >>> start = Coordinate(16.0, TimeUnit.quarters)
        >>> end = Coordinate(32.0, TimeUnit.quarters)
        >>> region = Region(start, end, name="Chorus")
        >>> region.id
        'region_1'
        >>> region.duration
        Duration(Fraction(16, 1), quarters)
        >>> region.as_interval
        Interval(start=Coordinate(Fraction(16, 1), quarters), \
end=Coordinate(Fraction(32, 1), quarters))

        A region marks part of a timeline; turning it into a Child timeline
        goes through the timeline that owns it:

        >>> from timetoalign import ContinuousLogicalTimeline
        >>> tl = ContinuousLogicalTimeline(length=64)
        >>> tl.create_region(16, 32, id="chorus", name="Chorus")
        Region('chorus', name='Chorus', 16-32 quarters)
        >>> chorus_child = tl.create_child_from_region("chorus")
        >>> chorus_child.length
        Coordinate(Fraction(16, 1), quarters)
    """

    _id_prefix: ClassVar[str] = "region"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "name",
        "start",
        "end",
        "meta",
        *STATE_KEYS,
    )
    _manifest_path: ClassVar[tuple[str, ...]] = ("timelines", "regions")

    start: Coordinate
    end: Coordinate
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    # The region's statements and provenance as the file writes them, and
    # the prefixes they need beyond the catalogue's; compared.
    _statement_state: Mapping[str, Any] = field(
        default_factory=dict, kw_only=True, repr=False
    )
    _prefixes: Mapping[str, str] = field(default_factory=dict, kw_only=True, repr=False)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Region:
        """Restore a region from its object in a timeline's ``regions``.

        ``start`` and ``end`` are bare values on the timeline's axis, which
        *unit* and *number_type* describe; ``name``, ``meta``,
        ``statements`` and ``provenance`` may be left out. Keys the library
        does not know are kept and written back. The prefixes the statements
        need are the manifest's, which its reader sets.

        Args:
            data: The region object.
            unit: The unit of the timeline holding the region.
            number_type: The number type of that timeline.

        Raises:
            ValueError: If the object is malformed or a value is represented
                otherwise than the axis selects.
        """
        require_object(data, "a region", ("id", "start", "end"))
        region_id = read_id(data["id"], "a region")
        bounds = {
            key: axis_value_from_json(
                data[key], key=key, owner=region_id, unit=unit, number_type=number_type
            )
            for key in ("start", "end")
        }
        what = f"region {region_id!r}"
        region = cls(
            id=region_id,
            name=read_name(data, what),
            meta=read_meta(data, what),
            _statement_state=read_statement_state(data),
            **bounds,
        )
        region._keep_unknown_keys(data)
        return region

    def __post_init__(self) -> None:
        """Mint or validate the id, then validate region bounds and units."""
        super().__post_init__()
        if self.end.value < self.start.value:
            raise ValueError(
                f"Region end ({self.end.value}) cannot be before "
                f"start ({self.start.value})"
            )
        if self.start.unit != self.end.unit:
            raise ValueError(
                f"Region start unit ({self.start.unit}) must match "
                f"end unit ({self.end.unit})"
            )

    @property
    def unit(self) -> TimeUnit:
        """The coordinate unit of this region."""
        return self.start.unit

    @property
    def duration(self) -> Duration:
        """Exact duration of the region (end - start)."""
        return self.as_interval.duration

    @property
    def as_interval(self) -> Interval:
        """The region's extent as a typed :class:`Interval`."""
        return Interval(start=self.start, end=self.end)

    @property
    def is_atomic_section(self) -> bool:
        """Whether the region states the type of an atomic section, unrejected.

        An atomic section is a stretch that the timeline's flow control
        never divides. The type is the format's class ``AtomicSection``
        (``timetoshare.ATOMIC_SECTION``), however the region spells it: as
        the absolute IRI or under a prefix it declares
        (``tts:AtomicSection``). A region whose statement of that type some
        actor believes ``"false"`` is none: that is how a section is kept,
        with its ID and its interval, once the flow control gives its
        stretch no more
        (:meth:`Timeline.create_flow_maps_from_flow_control
        <timetoalign.timelines.Timeline.create_flow_maps_from_flow_control>`).
        """
        stated = self._atomic_section_type()
        if stated is None:
            return False
        context = self._statement_context()
        return not any(
            belief.get("belief") == "false"
            for entry in self.provenance
            if same_pair(
                context, ("type", stated), (entry.get("property"), entry.get("value"))
            )
            for belief in entry.get("beliefs", ())
        )

    def _atomic_section_type(self) -> str | None:
        """The value of ``type`` that names the format's ``AtomicSection``,
        as the region spells it; None where it states none."""
        context = self._statement_context()
        for value in self.statements.get("type", ()):
            if isinstance(value, str) and expand_iri(context, value) == ATOMIC_SECTION:
                return value
        return None

    def contains(
        self, coord: int | float | Fraction | Coordinate | IdCoordinate
    ) -> bool:
        """Check if a coordinate is within this region (left-inclusive).

        Following TTA convention, intervals are [start, end) - left-inclusive,
        right-exclusive.

        Args:
            coord: The coordinate value to check.

        Returns:
            True if start <= coord < end.

        Raises:
            ValueError: If a coordinate object's unit differs from the region's.
        """
        if isinstance(coord, Coordinate):
            if coord.unit is not self.unit:
                raise ValueError(
                    f"Coordinate unit {coord.unit} does not match region unit "
                    f"{self.unit}"
                )
            value = coord.value
        else:
            value = coord
        return self.start.value <= value < self.end.value

    def overlaps(self, other: Region) -> bool:
        """Check if this region overlaps with another.

        Args:
            other: Another Region to check against.

        Returns:
            True if the regions overlap (share any coordinates).
        """
        return self.start.value < other.end.value and other.start.value < self.end.value

    def _dict_fields(self) -> dict[str, Any]:
        """Return the region's keys; ``name`` and ``meta`` are optional,
        ``statements`` and ``provenance`` written only when not empty."""
        return {
            "id": self.id,
            "name": self.name,
            "start": axis_value_to_json(self.start),
            "end": axis_value_to_json(self.end),
            "meta": deepcopy(self.meta),
            **deepcopy(dict(self._statement_state)),
        }

    # region Statements, copy-on-write

    def _statement_holder_name(self) -> str:
        """Return the region as a message names it."""
        return f"region {quote_id(self.id)}"

    def _with_statement_state(
        self,
        state: Mapping[str, Any] | None = None,
        prefixes: Mapping[str, str] | None = None,
    ) -> Region:
        """Return a copy holding *state* and *prefixes* (None keeps them)."""
        return replace(
            self,
            _statement_state=self._statement_state if state is None else dict(state),
            _prefixes=self._prefixes if prefixes is None else dict(prefixes),
        )

    def _with_references_renamed(self, renamed: Mapping[str, str | None]) -> Region:
        """Return the region with the elements its statements name renamed.

        The region itself where its statements name none of them (see
        :func:`~timetoalign.timelines.statements.renamed_state`).
        """
        if not self._statement_state:
            return self
        state = renamed_state(self._statement_state, renamed)
        if state == self._statement_state:
            return self
        return self._with_statement_state(state)

    def with_statement(
        self,
        property: str,
        value: Any,
        *,
        node: NodePath = (),
        provenance: Provenance | Mapping[str, Any] | None = None,
        actor: str | None = None,
        date: str | datetime.date | None = None,
    ) -> Region:
        """Return a copy stating the pair ``(property, value)``; this region is left as it is.

        See :meth:`MutableStatementsMixin.add_statement
        <timetoalign.timelines.MutableStatementsMixin.add_statement>`.

        Examples:
            >>> from timetoalign import Coordinate, TimeUnit
            >>> intro = Region(Coordinate(0, TimeUnit.quarters), Coordinate(12, TimeUnit.quarters))
            >>> described = intro.with_statement("rdfs:comment", "the opening")
            >>> intro.statements, described.statements
            ({}, {'rdfs:comment': ['the opening']})
        """
        return self._with_statement_state(
            self._state_with_statement(
                property,
                value,
                node=node,
                provenance=provenance,
                actor=actor,
                date=date,
            )
        )

    def without_statement(
        self, property: str, value: Any, *, node: NodePath = ()
    ) -> Region:
        """Return a copy without the pair ``(property, value)`` and its entry.

        See :meth:`MutableStatementsMixin.remove_statement
        <timetoalign.timelines.MutableStatementsMixin.remove_statement>`.
        """
        return self._with_statement_state(
            self._state_without_statement(property, value, node=node)
        )

    def with_candidate(
        self,
        property: str,
        value: Any,
        *,
        stated_by: str | Sequence[str],
        method: str,
        date: str | datetime.date | None = None,
        service: str | None = None,
        basis: Sequence[str] = (),
        match: str | None = None,
        note: str | None = None,
        node: NodePath = (),
    ) -> Region:
        """Return a copy offering the pair ``(property, value)`` as a candidate.

        See :meth:`MutableStatementsMixin.add_candidate
        <timetoalign.timelines.MutableStatementsMixin.add_candidate>`.
        """
        return self._with_statement_state(
            self._state_with_candidate(
                property,
                value,
                stated_by=stated_by,
                method=method,
                date=date,
                service=service,
                basis=basis,
                match=match,
                note=note,
                node=node,
            )
        )

    def with_belief(
        self,
        property: str,
        value: Any,
        *,
        actor: str,
        belief: BeliefValue,
        date: str | datetime.date | None = None,
        note: str | None = None,
        node: NodePath = (),
    ) -> Region:
        """Return a copy recording *actor*'s belief in the pair ``(property, value)``.

        See :meth:`MutableStatementsMixin.add_belief
        <timetoalign.timelines.MutableStatementsMixin.add_belief>`.
        """
        return self._with_statement_state(
            self._state_with_belief(
                property,
                value,
                actor=actor,
                belief=belief,
                date=date,
                note=note,
                node=node,
            )
        )

    def with_verb(
        self,
        verb: str,
        system: str,
        value: str,
        *,
        label: str | None = None,
        slots: Mapping[str, Any] | None = None,
        alternative: int = 0,
        provenance: Provenance | Mapping[str, Any] | None = None,
        actor: str | None = None,
        date: str | datetime.date | None = None,
        catalogue: Catalogue | None = None,
    ) -> Region:
        """Return a copy stating a verb of the catalogue in one system's vocabulary.

        See :meth:`MutableStatementsMixin.add_verb
        <timetoalign.timelines.MutableStatementsMixin.add_verb>`.
        """
        return self._with_statement_state(
            self._state_with_verb(
                verb,
                system,
                value,
                label=label,
                slots=slots,
                alternative=alternative,
                provenance=provenance,
                actor=actor,
                date=date,
                catalogue=catalogue,
            )
        )

    def with_prefix(self, name: str, namespace: str) -> Region:
        """Return a copy declaring a prefix its statements may use.

        See :meth:`MutableStatementsMixin.add_prefix
        <timetoalign.timelines.MutableStatementsMixin.add_prefix>`.
        """
        return self._with_statement_state(prefixes=self._prefixes_with(name, namespace))

    # endregion

    def __repr__(self) -> str:
        label = (
            repr(self.id) if self.name is None else f"{self.id!r}, name={self.name!r}"
        )
        return f"Region({label}, {self.start.value}-{self.end.value} {self.unit})"

    def __str__(self) -> str:
        label = self.id if self.name is None else f"{self.name} ({self.id})"
        return f"{label}: [{self.start.value}, {self.end.value}) {self.unit}"


# endregion
