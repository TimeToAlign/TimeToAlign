"""TimelineGroup: an array of timelines sharing one start and one end.

A group is defined by a single property: all of its member timelines share
the same graphical start and end point. Every member's 0 coincides with every
other member's 0, every member's length with every other member's length,
and a position in between converts linearly — position ``p`` on member A is
``p · len_B / len_A`` on member B, computed exactly. When only part of a
timeline takes part, the participant is a child of that timeline, which
carries its exact offset; a member is never placed partially.

IMPORTANT CONCEPTUAL DISTINCTION:
- Perfect Alignment: Bijective coordinate mapping (linear interpolation).
  Does NOT imply the alignment is musically/temporally correct.
- Correct Alignment: A special case where mapping corresponds to reality.

TimelineGroup implements the TimeStampSource protocol and answers with the
same TimeStamp class as Timeline, so a stamp from a group has the identical
structure and methods as a stamp from a timeline.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    IdCoordinate,
    Interval,
    resolve_coordinate_spec,
    resolve_id,
)
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.ids import _OpenObject, mint_id, timeline_id_prefix, validate_id
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    TableFormat,
    dispatch_retrieval,
    format_coordinates,
    is_key_input,
    reject_dataframe_options,
    resolve_coordinate_collection,
    resolve_key_collection,
    validate_coordinate_collection,
    validate_key_collection,
    validate_table_format,
)
from timetoalign.core.time import express_as, struct_to_coordinate
from timetoalign.core.timestamp import (
    ConversionMapsSpec,
    TimeStamp,
    TimestampColumn,
    _format_coordinate_value,
    build_timestamp_table,
    timestamp_table_to_dataframe,
)
from timetoalign.storage import EventData

from .manifest import (
    Manifest,
    matches_any,
    read_id,
    read_meta,
    read_name,
    require_object,
    select,
    write_roots,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from timetoalign.core.enums import ColumnNaming
    from timetoalign.display.ascii import Diagram
    from timetoalign.maps.base import ConversionMap
    from timetoalign.timelines import Timeline
    from timetoalign.timelines.flow import Flow, FlowControllerBase

module_logger = logging.getLogger(__name__)


# region Timeline trees


def _root(timeline: "Timeline") -> "Timeline":
    """Return the root of the manifest tree *timeline* belongs to."""
    while timeline._parent is not None:
        timeline = timeline._parent
    return timeline


def _tree(root: "Timeline") -> list["Timeline"]:
    """Return *root* and every descendant of it."""
    return [root, *(descendant for _, descendant in root._iter_descendants())]


def _held_ids(timelines: Iterable["Timeline"]) -> dict[str, "Timeline"]:
    """Map every id in the manifest trees of *timelines* to its timeline.

    A timeline's tree is the whole manifest it belongs to — its root and all
    of that root's descendants — because that is what a file holding the
    timeline writes.
    """
    held: dict[str, Timeline] = {}
    for timeline in timelines:
        for member in _tree(_root(timeline)):
            held[member.id] = member
    return held


def _nested(first: "Timeline", second: "Timeline") -> bool:
    """Whether one timeline is an ancestor of the other."""

    def ancestors(timeline: "Timeline") -> Iterator["Timeline"]:
        while timeline._parent is not None:
            timeline = timeline._parent
            yield timeline

    return any(ancestor is second for ancestor in ancestors(first)) or any(
        ancestor is first for ancestor in ancestors(second)
    )


def _admit_timeline_tree(
    newcomer: "Timeline",
    held: dict[str, "Timeline"],
    scope: str,
    *,
    reserved: Iterable[str] = (),
    check_only: bool = False,
) -> None:
    """Admit *newcomer*'s manifest tree into a scope of unique timeline ids.

    An id may be held only once in the scope. A tree timeline whose id is
    held by a *different* timeline, or is one of the *reserved* ids (given
    ids of the scope that are no timeline's, such as a group's), is a
    collision: a minted id is minted again, and a given id raises. Nothing
    changes before every collision has been checked, so a raise leaves the
    newcomer untouched.

    Args:
        newcomer: The timeline joining the scope.
        held: Every timeline id of the scope, mapped to its timeline.
        scope: A description of the scope, for the error.
        reserved: Further ids the scope holds that no timeline may take.
        check_only: Raise as admitting would, but mint nothing.

    Raises:
        ValueError: If a given id of the tree is held in the scope.
    """
    reserved = set(reserved)
    tree = _tree(_root(newcomer))
    clashes = [
        timeline
        for timeline in tree
        if (timeline.id in held and held[timeline.id] is not timeline)
        or timeline.id in reserved
    ]
    refused = [timeline for timeline in clashes if not timeline._id_minted]
    if refused:
        holder = "a group" if refused[0].id in reserved else "another timeline"
        raise ValueError(
            f"Timeline {refused[0].id!r} cannot join {scope}: its id is already "
            f"held by {holder} there"
        )
    if check_only:
        return
    taken = set(held) | reserved | {timeline.id for timeline in tree}
    for timeline in clashes:
        minted = mint_id(timeline_id_prefix(timeline.unit), taken)
        taken.add(minted)
        timeline._set_id(minted, minted=True)


# endregion


# region TimelineGroup


@dataclass
class TimelineGroup(_OpenObject):
    """An array of timelines sharing one graphical start and end point.

    By definition every member's 0 coincides with every other member's 0 and
    every member's length with every other member's length; a position in
    between converts linearly. Position ``p`` on member A is
    ``p · len_B / len_A`` on member B, computed exactly — a float enters as
    its exact dyadic ``Fraction`` — and written in B's declared number type.

    When only part of a timeline participates, the participant is a child of
    that timeline: the child carries its exact offset, and the child is the
    member. A member is never placed partially.

    A member may be of unknown length: membership states that it shares the
    group's start and end, wherever that end lies. Carrying a position
    between two members needs both lengths, so such a conversion raises
    ``UnknownLengthError``; one member's length is never carried to another.

    Members keep the order in which they were added.

    Attributes:
        id: Unique identifier for this group, ``group_N`` unless given.
        name: Optional human-readable name.
        meta: Additional metadata dictionary.
        styling: The members of the manifest's ``graphical`` other than
            ``order`` (``styles``, ``statements`` and any other), kept as
            given and never interpreted.

    Examples:
        >>> from fractions import Fraction
        >>> from timetoalign.timelines import TimelineGroup
        >>> from timetoalign.timelines.types import (
        ...     ContinuousLogicalTimeline,
        ...     DiscreteGraphicalTimeline,
        ... )
        >>> score = ContinuousLogicalTimeline(length=Fraction(16))
        >>> image = DiscreteGraphicalTimeline(length=4800)
        >>> group = TimelineGroup(timelines=[score, image])
        >>> group.id, group.timeline_ids
        ('group_1', ['clt_1', 'dgt_1'])
        >>> group.get_coordinate_at(Fraction(1, 3), "clt_1", format="coordinate")
        Coordinate(Fraction(1, 3), quarters)
        >>> group.get_timestamp_at(Fraction(1, 3), "clt_1").get_coordinate_for(
        ...     "dgt_1", format="int"
        ... )
        100
    """

    _dict_keys: ClassVar[tuple[str, ...]] = ("id", "name", "meta", "members")
    _dict_defaults: ClassVar[Mapping[str, Any]] = {"name": None, "meta": {}}

    id: str = field(default="")
    name: str | None = field(default=None)
    meta: dict[str, Any] = field(default_factory=dict)
    styling: dict[str, Any] = field(default_factory=dict)

    # Internal state (not part of dataclass signature)
    _timelines: dict[str, "Timeline"] = field(
        default_factory=dict, init=False, repr=False
    )
    _id_minted: bool = field(default=False, init=False, repr=False)
    # The manifest the group was read from as a whole (its meta and unknown
    # keys), written back around it.
    _manifest: Manifest | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _logger: logging.Logger = field(
        default_factory=lambda: module_logger, init=False, repr=False
    )

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        group_id: str | None = None,
        name: str | None = None,
    ) -> Self:
        """Read a group from a TimeSkeleton manifest.

        - A manifest of one root timeline and no group makes a group of that
          timeline; *group_id* or *name* (or both) must be given.
        - An anonymous bundle with one group reads that group. Arguments
          naming it select it; arguments naming nothing override its id and
          name; given both, they must agree, both naming the group or
          neither.

        In these two cases the manifest is read as a whole: its ``meta``,
        ``graphical`` and unknown keys are kept so that :meth:`to_dict`
        writes them back, around the anonymous bundle a group writes.
        - From a manifest with several groups, or a named one, *group_id* or
          *name* (or both) must be given and select the one group whose id
          and name match them.

        A group is read with the timelines its members need: each member id
        is resolved among the manifest's roots and all their descendants, and
        the members are added in the group's ``graphical.order`` entry (in
        ``members`` order without one). ``graphical``'s other keys are kept as
        :attr:`styling`, unread, when the manifest is read as a whole. Keys
        the library does not know are kept on the object read from the JSON
        object they were found in, and written back.

        Args:
            data: A TimeSkeleton manifest.
            group_id: The group id to select, or the id the group takes.
            name: The group name to select, or the name the group takes.

        Returns:
            The group read.

        Raises:
            ValueError: If *data* is no TimeSkeleton manifest of version
                0.7.0 or is malformed — a member naming no timeline of the
                manifest, an order that is no permutation of its group's
                members, a member that is an ancestor or descendant of
                another member, a group id held by a timeline or by another
                group — or if the selection matches no group or several; the
                message names the problem, or the manifest's groups.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import ContinuousLogicalTimeline
            >>> group = TimelineGroup(
            ...     name="edition",
            ...     timelines=[ContinuousLogicalTimeline(length=Fraction(8))],
            ... )
            >>> data = group.to_dict()
            >>> TimelineGroup.from_dict(data).to_dict() == data
            True
        """
        manifest, content = Manifest.read(data)
        if not content.groups:
            if len(content.roots) != 1:
                raise ValueError(
                    f"the manifest holds no group and {len(content.roots)} timelines: "
                    "only a manifest of one timeline makes a group"
                )
            if group_id is None and name is None:
                raise ValueError(
                    "the manifest holds one timeline and no group: give group_id or "
                    "name to make a group of it"
                )
            if group_id is not None and group_id in content.timelines:
                raise ValueError(f"group id {group_id!r} is also a timeline id")
            # A group writes an anonymous bundle.
            manifest.id = manifest.name = None
            group = cls(id=group_id, name=name)
            # The member keeps the lock the manifest states.
            group._admit_member(content.roots[0])
            return group._keep_manifest(manifest)

        groups = [
            cls._from_group_object(
                group_data,
                content.timelines,
                (
                    content.order.get(group_data.get("id"))
                    if isinstance(group_data, dict)
                    else None
                ),
            )
            for group_data in content.groups
        ]
        ids = [group.id for group in groups]
        repeated = sorted({group_id for group_id in ids if ids.count(group_id) > 1})
        if repeated:
            raise ValueError(f"group id {repeated[0]!r} occurs twice")
        if len(groups) == 1 and manifest.id is None and manifest.name is None:
            (group,) = groups
            if matches_any(groups, id=group_id, name=name):
                select(
                    groups,
                    id=group_id,
                    name=name,
                    what="group",
                    names=[(group.id, group.name)],
                )
                return group._keep_manifest(manifest)
            if group_id is not None:
                if group_id in content.timelines:
                    raise ValueError(f"group id {group_id!r} is also a timeline id")
                group.id = validate_id(group_id)
                group._id_minted = False
            if name is not None:
                group.name = name
            return group._keep_manifest(manifest)
        if group_id is None and name is None:
            raise ValueError(
                "the manifest holds several groups or is named: give group_id or name "
                "to select a group"
            )
        return select(
            groups,
            id=group_id,
            name=name,
            what="group",
            names=[(group.id, group.name) for group in groups],
        )

    @classmethod
    def _from_group_object(
        cls,
        data: Any,
        timelines: dict[str, "Timeline"],
        order: Any = None,
    ) -> Self:
        """Build a group from its ``Group`` object and the manifest's timelines.

        The one reader of the ``Group`` object, shared by :meth:`from_dict`
        and the bundle's reader. ``name`` and ``meta`` may be left out; keys
        the library does not know are kept and written back.

        Args:
            data: ``{"id", "name"?, "meta"?, "members"}``.
            timelines: Every timeline of the manifest, roots and descendants,
                by id.
            order: The members' display order, or None for ``members`` order.

        Raises:
            ValueError: If the object or the order is malformed.
        """
        require_object(data, "a group object", ("id", "members"))
        group_id = read_id(data["id"], "a group")
        if group_id in timelines:
            raise ValueError(f"group id {group_id!r} is also a timeline id")
        what = f"group {group_id!r}"
        members = data["members"]
        if not isinstance(members, list) or not all(
            isinstance(member, str) for member in members
        ):
            raise ValueError(f"{what} members must be an array of ids")
        if len(set(members)) != len(members):
            raise ValueError(f"{what} names a member twice")
        for member in members:
            if member not in timelines:
                raise ValueError(
                    f"{what} names member {member!r}, which is no timeline of "
                    "'timelines'"
                )
        if order is None:
            order = members
        elif (
            not isinstance(order, list)
            or len(order) != len(members)
            or set(order) != set(members)
        ):
            raise ValueError(f"the order of {what} is not a permutation of its members")
        group = cls(id=group_id, name=read_name(data, what), meta=read_meta(data, what))
        # Members keep the lock the manifest states.
        for member in order:
            group._admit_member(timelines[member])
        group._keep_unknown_keys(data)
        return group

    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        timelines: list["Timeline"] | None = None,
        meta: dict[str, Any] | None = None,
        styling: dict[str, Any] | None = None,
    ) -> None:
        """Create a TimelineGroup.

        Args:
            id: Unique identifier. If None, ``group_N`` is minted with N the
                smallest number no timeline of *timelines* holds as an id.
            name: Optional human-readable name.
            timelines: Initial members, in order (optional).
            meta: Additional metadata dictionary.
            styling: The group file's ``styles`` and ``statements``
                (optional), kept as given and never interpreted.

        Raises:
            ValueError: If a given id is invalid, or an initial member
                cannot join (see :meth:`add_timeline`).
        """
        initial = list(timelines or [])
        if id is None:
            self.id = mint_id("group", _held_ids(initial))
            self._id_minted = True
        else:
            self.id = validate_id(id)
            self._id_minted = False
        self.name = name
        self.meta = dict(meta) if meta else {}
        self.styling = dict(styling) if styling else {}
        self._timelines = {}
        self._logger = module_logger.getChild(self.id)

        for tl in initial:
            self.add_timeline(tl)

    # region Timeline Management

    def add_timeline(self, timeline: "Timeline") -> None:
        """Add a member; its whole extent ``[0, length]`` spans the group.

        A member's 0 coincides with every other member's 0 and its length
        with every other member's length. To let only part of a timeline
        participate, add the child of that timeline covering the part.

        Every timeline id in the newcomer's manifest tree (its root and all
        that root's descendants) must be free in the group's trees.

        A newcomer that is an ancestor or a descendant of a member is
        refused: a child sits at its offset inside its parent, so the two
        cannot share one start and end. Every check runs before anything
        changes, so a refused newcomer is left as it was.

        A segment line is locked when it joins, as when it becomes a child.

        Args:
            timeline: The timeline or child to add.

        Raises:
            ValueError: If the timeline is already a member; if it is nested
                with a member; or if a given id of its tree — its own, its
                root's or any descendant's — is held by another timeline of
                the group's trees or is the group's given id (a minted id held
                there is minted again instead, and a minted group id that a
                newcomer's given id takes is minted again).

        Examples:
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousPhysicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> image = DiscreteGraphicalTimeline(length=5000)
            >>> holes = DiscreteGraphicalTimeline(length=2000)
            >>> image.add_child(holes, offset=1000)
            >>> group = TimelineGroup(timelines=[ContinuousPhysicalTimeline(length=8.0)])
            >>> group.add_timeline(holes)
            >>> group.get_coordinate_at(2.0, "cpt_1")
            IdCoordinate(2.0, seconds, 'cpt_1')
            >>> group.get_timestamp_at(2.0, "cpt_1").get_coordinate_for("dgt_2", format="int")
            500
        """
        self._admit_member(timeline)
        if timeline._is_segment_line_instance():
            timeline.lock()

    def _admit_member(self, timeline: "Timeline") -> None:
        """Add a member as :meth:`add_timeline` does, leaving its lock as it is.

        Raises:
            ValueError: As :meth:`add_timeline`.
        """
        if any(member is timeline for member in self._timelines.values()):
            raise ValueError(f"Timeline '{timeline.id}' already in group '{self.id}'")
        self._refuse_nested(timeline)
        held = _held_ids(self._timelines.values())
        _admit_timeline_tree(
            timeline,
            held,
            f"group {self.id!r}",
            reserved=() if self._id_minted else (self.id,),
        )
        # A minted group id that a newcomer's given id now holds is minted
        # again; a given group id was reserved above.
        tree_ids = {member.id for member in _tree(_root(timeline))}
        if self.id in tree_ids:
            self.id = mint_id("group", set(held) | tree_ids)
        self._timelines[timeline.id] = timeline
        self._logger.debug(f"Added timeline '{timeline.id}'")

    def _refuse_nested(self, timeline: "Timeline") -> None:
        """Refuse a newcomer that is an ancestor or descendant of a member.

        Members share one start and one end, while a child sits at its offset
        inside its parent: both in one group would give one position two
        answers.

        Raises:
            ValueError: Naming the newcomer and the member it is nested with.
        """
        for member in self._timelines.values():
            if _nested(member, timeline):
                raise ValueError(
                    f"Timeline {timeline.id!r} cannot join group {self.id!r}: it "
                    f"is nested with member {member.id!r}, and a parent and its "
                    "child cannot share one start and end"
                )

    def remove_timeline(self, timeline_id: str) -> "Timeline":
        """Remove a member from the group.

        Args:
            timeline_id: ID of the timeline to remove.

        Returns:
            The removed timeline.

        Raises:
            KeyError: If timeline_id is not in the group.
        """
        if timeline_id not in self._timelines:
            raise KeyError(
                f"Timeline '{timeline_id}' not in group '{self.id}'. "
                f"Available timelines: {self.timeline_ids}"
            )

        timeline = self._timelines.pop(timeline_id)
        self._logger.debug(f"Removed timeline '{timeline_id}'")
        return timeline

    def _rekey_members(self) -> None:
        """Key the members by their current ids, keeping their order.

        A member's id can change after it joined (a minted id is minted again
        when its tree joins a wider scope); this restores the lookup.
        """
        self._timelines = {
            timeline.id: timeline for timeline in self._timelines.values()
        }

    def get_timeline(self, timeline_id: str) -> "Timeline":
        """Get a timeline by ID.

        Supports partial string and regex matching:
        1. Exact match: If ``timeline_id`` matches an ID exactly, returns it.
        2. Substring match: If ``timeline_id`` is a substring of exactly one ID,
           returns that timeline. If multiple match, returns the first and warns.
        3. Regex match: If ``timeline_id`` is a valid regex, matches via
           ``re.search()``. Same first-match logic with warning.

        Args:
            timeline_id: The timeline's unique identifier, or a partial/regex pattern.

        Returns:
            The Timeline object.

        Raises:
            KeyError: If no timeline matches the pattern.

        Examples:
            >>> group.get_timeline("clt_1")           # Exact match
            >>> group.get_timeline("notes")          # Substring match
            >>> group.get_timeline(r"^score:")       # Regex match
        """
        resolved_id = resolve_id(timeline_id, list(self._timelines.keys()))
        return self._timelines[resolved_id]

    @property
    def n_timelines(self) -> int:
        """Number of timelines in the group."""
        return len(self._timelines)

    @property
    def timeline_ids(self) -> list[str]:
        """IDs of all member timelines."""
        return list(self._timelines.keys())

    def __contains__(self, timeline_id: str) -> bool:
        """Check if a timeline is in the group."""
        return timeline_id in self._timelines

    def __getitem__(self, timeline_id: str) -> "Timeline":
        """Get a timeline by ID.

        Shorthand for ``get_timeline(timeline_id)``. Supports partial
        string and regex matching.

        Args:
            timeline_id: The timeline's unique identifier, or a partial/regex pattern.

        Returns:
            The Timeline object.

        Raises:
            KeyError: If no timeline matches the pattern.

        Examples:
            >>> tl = group["clt_1"]           # Exact match
            >>> tl = group["score"]          # Substring match
        """
        return self.get_timeline(timeline_id)

    def get_events(
        self,
        *,
        timeline_id: str | None = None,
        **kwargs: Any,
    ) -> EventData:
        """Get events from all timelines in the group, concatenated.

        Collects events from all member timelines (or a specific one) and
        concatenates their Arrow tables into a single EventData. Each row includes a
        ``timeline_id`` field identifying the source timeline. A member-provided
        ``timeline_id`` column is discarded because group membership is authoritative.

        Args:
            timeline_id: If provided, only return events from this timeline.
                Supports partial string and regex matching.
            **kwargs: Passed through to each timeline's ``get_events()`` method
                (e.g., ``min_coord``, ``max_coord``, ``event_type``).

        Returns:
            EventData with events from all (or specified) timelines. Includes
            a ``timeline_id`` field and standard event fields (``start``,
            ``end``, ``event_type``, etc.).

        Examples:
            >>> # Get all events from all timelines
            >>> df = group.get_events().to_dataframe()

            >>> # Get events from a specific timeline
            >>> df = group.get_events(timeline_id="clt_1").to_dataframe()

            >>> # Get events with filters
            >>> df = group.get_events(
            ...     event_type="Note", min_coord=0.0, max_coord=100.0
            ... ).to_dataframe()
        """
        if timeline_id is not None:
            timeline = self.get_timeline(timeline_id)
            timelines = [(timeline.id, timeline)]
        else:
            timelines = list(self._timelines.items())

        tables: list[pa.Table] = []
        result_unit = TimeUnit.seconds
        result_number_type = NumberType.float
        for tl_id, timeline in timelines:
            events = timeline.get_events(**kwargs)
            result_unit = events.unit
            result_number_type = events.number_type
            if len(events) == 0:
                continue
            source_ids = pa.array([tl_id] * len(events), type=pa.string())
            member_table = events.table
            if "timeline_id" in member_table.column_names:
                member_table = member_table.drop_columns(["timeline_id"])
            tables.append(member_table.append_column("timeline_id", source_ids))

        if tables:
            column_types: dict[str, pa.DataType] = {}
            for table in tables:
                for schema_field in table.schema:
                    previous = column_types.get(schema_field.name)
                    if previous is None or pa.types.is_null(previous):
                        column_types[schema_field.name] = schema_field.type
                        continue
                    if pa.types.is_null(schema_field.type):
                        continue
                    if previous != schema_field.type:
                        raise ValueError(
                            f"Conflicting Arrow types for column {schema_field.name!r}: "
                            f"{previous} and {schema_field.type}."
                        )
            table = pa.concat_tables(tables, promote_options="permissive")
            return EventData(table, result_unit, result_number_type)

        empty = EventData.empty(result_unit, result_number_type)
        return EventData(
            empty.table.append_column("timeline_id", pa.array([], type=pa.string())),
            result_unit,
            result_number_type,
        )

    # endregion

    # region Timestamps

    def _convert(
        self, value: CoordinateValue, source_id: str, target_id: str
    ) -> CoordinateValue:
        """Carry a position from one member onto another, exactly.

        Position ``p`` on member A is ``p · len_B / len_A`` on member B. The
        arithmetic runs on exact ratios — a float enters as its exact dyadic —
        and the result is written in B's declared number type, which rounds
        half to even onto an integer axis.

        Raises:
            UnknownLengthError: If either member's length is unknown; one
                member's length is never carried to another.
            ValueError: If the source member has length zero and the target
                does not, so that no position can be carried across.
        """
        source = self._timelines[source_id]
        target = self._timelines[target_id]
        operation = f"convert a position from '{source_id}' to '{target_id}'"
        source_length = Fraction(source._require_length(operation).value)
        target_length = Fraction(target._require_length(operation).value)
        if source_length == 0:
            if target_length != 0:
                raise ValueError(
                    f"Timeline '{source_id}' has length 0; no position on it "
                    f"converts to '{target_id}'"
                )
            return express_as(Fraction(0), target.number_type)
        converted = Fraction(value) * target_length / source_length
        return express_as(converted, target.number_type)

    def get_timestamp_at(
        self,
        at: CoordinateSpec,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp:
        """Get a TimeStamp at a specific coordinate.

        This is the primary coordinate resolution API for TimelineGroup.
        Returns a proper TimeStamp object (same as Timeline.get_timestamp_at).
        The queried position is written the way the addressed member writes
        numbers; every other member's coordinate is the exact linear image of
        that position (see :class:`TimelineGroup`), and each member's
        descendants are reached through their exact offsets.

        Args:
            at: The query coordinate. Can be:
                - int/float/Fraction: Raw value, timeline_id required
                - Coordinate: Value with unit, timeline_id required
                - IdCoordinate: Value with unit AND timeline_id (timeline_id param optional)
            timeline_id: Which timeline the coordinate refers to.
                Required unless coordinate is an IdCoordinate.
            conversion_maps: Whether to include C-Map values in timestamp.
                - True (default): C-Maps accessible through typed getters.
                - False/None: Only timeline coordinates

        Returns:
            TimeStamp with axis set to the input coordinate and source_id set to
            the timeline. Access other timelines via `get_coordinate_for()`.
            Access C-Maps via `get_unit()` or `get_conversion_for()`. The
            stamp is interpolated unless the position is the group's common
            start or end.

        Raises:
            KeyError: If timeline_id is not in the group.
            UnknownLengthError: If a position must be converted between two
                members and either length is unknown. A member of unknown
                length bounds its positions below only, so a group of that
                member alone answers every position from 0 on.
            ValueError: If coordinate is outside the timeline's range in the group.
            ValueError: If timeline_id is None and coordinate is not IdCoordinate.

        Examples:
            >>> from timetoalign.core import IdCoordinate, TimeUnit
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousPhysicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> dgt_1 = DiscreteGraphicalTimeline(length=4875, unit="pixels")
            >>> audio = ContinuousPhysicalTimeline(length=150.0, unit="seconds")
            >>> group = TimelineGroup(timelines=[dgt_1, audio])

            >>> ts = group.get_timestamp_at(75.0, "cpt_1")
            >>> ts.axis
            IdCoordinate(75.0, seconds, 'cpt_1')
            >>> ts.get_coordinate_for("dgt_1", format="int")
            2438

            >>> # Using IdCoordinate (timeline_id extracted automatically)
            >>> coord = IdCoordinate(75.0, TimeUnit.seconds, "cpt_1")
            >>> group.get_timestamp_at(coord).axis
            IdCoordinate(75.0, seconds, 'cpt_1')
        """
        resolved = resolve_coordinate_spec(at, timeline_id=timeline_id)
        timeline_id = resolved.timeline_id

        if timeline_id is None:
            raise ValueError(
                "timeline_id is required unless coordinate is an IdCoordinate"
            )

        if timeline_id not in self._timelines:
            raise KeyError(
                f"Timeline '{timeline_id}' not in group. "
                f"Available timelines: {self.timeline_ids}"
            )
        source_timeline = self._timelines[timeline_id]

        if resolved.unit is not None:
            native_coordinate = source_timeline.get_coordinate_at(
                Coordinate(resolved.value, resolved.unit), format="coordinate"
            )
            if not isinstance(native_coordinate, Coordinate):
                raise TypeError(
                    "Member coordinate resolution did not return Coordinate"
                )
            native_value = native_coordinate.value
        else:
            native_value = resolved.value
        # The stamp reports the position the way the addressed timeline writes
        # numbers, whatever Python type the caller happened to pass, and every
        # other member is a projection of that written position.
        axis_value = express_as(native_value, source_timeline.number_type)

        # A member of unknown length bounds its positions below only.
        length = (
            None if source_timeline.length is None else source_timeline.length.value
        )
        if axis_value < 0 or (length is not None and axis_value > length):
            raise ValueError(
                f"Coordinate {_format_coordinate_value(float(axis_value))} "
                f"outside range for '{timeline_id}'"
            )
        # The common start and end are the stored rows; between them, the
        # position is interpolated.
        is_interpolated = not (axis_value == 0 or axis_value == length)

        stored: dict[str, Coordinate] = {}
        for member_id, member in self._timelines.items():
            value = (
                axis_value
                if member_id == timeline_id
                else self._convert(axis_value, timeline_id, member_id)
            )
            stored[member_id] = Coordinate(
                value, member.unit, number_type=member.number_type
            )
        for member_id, member in self._timelines.items():
            member_stamp = member.get_timestamp_at(stored[member_id])
            for (
                descendant_id,
                descendant_coordinate,
            ) in member_stamp.coordinates.items():
                stored.setdefault(descendant_id, descendant_coordinate)
        ordered = {timeline_id: stored.pop(timeline_id), **stored}
        return TimeStamp(
            coordinates=ordered,
            source=self,
            source_id=timeline_id,
            is_interpolated=is_interpolated,
            conversion_maps=conversion_maps,
        )

    def get_timestamps_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> list[TimeStamp]:
        """Get TimeStamps at a collection of coordinates, in input order.

        Args:
            at: Coordinate positions to query.
            timeline_id: Which timeline the coordinates refer to. Required for
                entries without an embedded timeline ID.
            conversion_maps: Whether C-Maps are reachable through the stamps.
                - True (default): all attached C-Maps
                - False/None: only timeline coordinates

        Returns:
            One TimeStamp per input coordinate; an empty collection gives an
            empty list.

        Raises:
            KeyError: If a named timeline is not in the group.
            ValueError: If a position lies outside its timeline's range, or a
                raw value arrives without a ``timeline_id``. A plural query
                answers completely or not at all — one unresolvable position
                raises rather than yielding a partial row.

        Examples:
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousPhysicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> pixels = DiscreteGraphicalTimeline(length=4000, unit="pixels", uid="dgt_1")
            >>> audio = ContinuousPhysicalTimeline(length=10.0, unit="seconds", uid="cpt_1")
            >>> group = TimelineGroup(id="pair", timelines=[pixels, audio])
            >>> stamps = group.get_timestamps_at([0.0, 2.5], "cpt_1")
            >>> [stamp.get_coordinate_for("dgt_1", format="int") for stamp in stamps]
            [0, 1000]

        See Also:
            get_timestamp_table: The same batch as one table or DataFrame.
        """
        stamps, _ = resolve_coordinate_collection(
            at,
            lambda value: self.get_timestamp_at(
                value, timeline_id, conversion_maps=conversion_maps
            ),
        )
        return stamps

    def get_timestamp_table(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection | None = None,
        timeline_id: str | None = None,
        *,
        timeline_filter: set[str] | None = None,
        conversion_maps: ConversionMapsSpec = True,
        format: TableFormat = "table",
        fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
        units: bool | None = None,
    ) -> pa.Table | pd.DataFrame:
        """Get the group's timestamps as a table, one row per cross-section.

        Args:
            at: Positions or event IDs to tabulate, one row each in input
                order. If None, the rows are the group's two stored
                timestamps: the common start and the common end, each member
                at ``0`` and at its length, exact in its declared number type.
            timeline_id: Which timeline the positions refer to. Required for
                entries without an embedded timeline ID.
            timeline_filter: Only include these timeline fields.
            conversion_maps: Whether to include C-Map fields from member timelines.
                - True (default): Include all attached C-Maps from all timelines
                - False/None: No C-Map fields
            format: ``"table"`` (default) for a PyArrow table, ``"dataframe"``
                for a pandas DataFrame.
            fields: How to name the DataFrame fields (``format="dataframe"``):
                - None or ColumnNaming.name (default): Use timeline/cmap name
                - ColumnNaming.id: Use timeline/cmap id
                - Callable: Function taking (name, metadata_dict) -> new_name
                - list[str]: Explicit field names
            units: If True (the DataFrame default), append units to field
                names like "name (unit)".

        Returns:
            One row per timestamp and one field per timeline — coordinate
            structs carrying each number twice — plus C-Map fields when
            ``conversion_maps`` is enabled. An empty group gives an empty
            table.

            A null cell means one thing only: that timeline does not reach
            that position. An unresolvable *query* position is an error, not
            a null, so the two can never be confused.

        Raises:
            KeyError: If a named timeline is not in the group.
            ValueError: On an unknown ``format``, when a DataFrame-shaping
                option is supplied for an Arrow result, or when a queried
                position lies outside its timeline's range. A batch answers
                completely or not at all, on this lane as on the stamp lane.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     ContinuousPhysicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> pixels = DiscreteGraphicalTimeline(length=4000, unit="pixels")
            >>> audio = ContinuousPhysicalTimeline(length=10.0, unit="seconds")
            >>> group = TimelineGroup(timelines=[pixels, audio])
            >>> group.get_timestamp_table().column_names
            ['dgt_1', 'cpt_1']
            >>> group.get_timestamp_table([0.0, 2.5], "cpt_1", format="dataframe")
               dgt_1 (pixels)  cpt_1 (seconds)
            0               0              0.0
            1            1000              2.5

            The stored rows are exact:

            >>> score = ContinuousLogicalTimeline(length=Fraction(5, 3))
            >>> TimelineGroup(timelines=[score]).get_timestamp_table(
            ...     format="dataframe"
            ... )["clt_1 (quarters)"].tolist()
            [Fraction(0, 1), Fraction(5, 3)]
        """
        validate_table_format(format)
        reject_dataframe_options(format, fields=fields, units=units)

        if at is None:
            table = self._stored_timestamp_table(timeline_filter, conversion_maps)
        else:
            table = self._queried_timestamp_table(
                at, timeline_id, timeline_filter, conversion_maps
            )
        if format == "table":
            return table
        return timestamp_table_to_dataframe(
            table=table,
            fields=fields,
            units=True if units is None else units,
        )

    def _stored_timestamp_table(
        self,
        timeline_filter: set[str] | None,
        conversion_maps: ConversionMapsSpec,
    ) -> pa.Table:
        """Tabulate the two stored timestamps: the common start and end.

        Each member stands at ``0`` and at its length, both written in the
        member's declared number type, so a stored row is as exact as the
        member's own length.

        Raises:
            UnknownLengthError: If a member's length is unknown, so that the
                common end has no position on it.
        """
        if not self._timelines:
            return pa.table({})
        ends = {
            member_id: member._require_length("tabulate the group's common end")
            for member_id, member in self._timelines.items()
        }
        rows: list[dict[str, Coordinate]] = [
            {
                member_id: Coordinate(
                    express_as(0, member.number_type),
                    member.unit,
                    number_type=member.number_type,
                )
                for member_id, member in self._timelines.items()
            },
            ends,
        ]
        return self._tabulate(rows, timeline_filter, conversion_maps)

    def _member_columns(
        self, names: list[str], values: dict[str, Any]
    ) -> list[TimestampColumn]:
        """Describe one coordinate column per named timeline."""
        columns: list[TimestampColumn] = []
        for name in names:
            timeline = self._resolve_member(name)
            columns.append(
                TimestampColumn(
                    name=name,
                    values=values[name],
                    unit=timeline.unit,
                    number_type=timeline.number_type,
                    timeline_id=name,
                )
            )
        return columns

    def _resolve_member(self, timeline_id: str) -> "Timeline":
        """Return a member or one of its descendants by ID."""
        member = self._timelines.get(timeline_id)
        if member is not None:
            return member
        for candidate in self._timelines.values():
            for _offset, descendant in candidate.iter_children(include_self=False):
                if descendant.id == timeline_id:
                    return descendant
        raise KeyError(
            f"Timeline '{timeline_id}' not in group. "
            f"Available timelines: {self.timeline_ids}"
        )

    def _cmap_columns(
        self,
        timeline_names: list[str],
        values: dict[str, Any],
        conversion_maps: ConversionMapsSpec,
    ) -> list[TimestampColumn]:
        """Derive one column per (member, numeric C-Map) from member columns."""
        if not conversion_maps:
            return []
        columns: list[TimestampColumn] = []
        for timeline_id in timeline_names:
            timeline = self._timelines.get(timeline_id)
            if timeline is None:
                continue
            for cmap in timeline._conversion_maps.values():
                target_unit = getattr(cmap, "target_unit", None)
                if target_unit is None:
                    # Numeric unit conversions only; a structured map has no
                    # column type here.
                    continue
                try:
                    converted = cmap.convert_array(values[timeline_id])
                except Exception:
                    # Skip C-Maps that fail (e.g., out of bounds)
                    continue
                columns.append(
                    TimestampColumn(
                        name=cmap.name,
                        values=pa.array(converted),
                        unit=target_unit,
                        number_type=cmap.output_number_type,
                        metadata={"cmap_id": cmap.id, "source_timeline": timeline_id},
                    )
                )
        return columns

    def _queried_timestamp_table(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None,
        timeline_filter: set[str] | None,
        conversion_maps: ConversionMapsSpec,
    ) -> pa.Table:
        """Tabulate one row per queried position or event key.

        A position no member can resolve raises, exactly as it does on the
        plural stamp getter. One input may not mean two things depending on
        which exit the caller took, and swallowing it into a null-filled row
        would make a null ambiguous: a reader could no longer tell "this
        member does not reach this position" from "your query was invalid".
        A null in the result therefore means exactly one thing — member not
        reachable — which is what makes it readable.
        """
        rows: list[dict[str, Coordinate]] = []
        if is_key_input(at):
            keys = [at] if isinstance(at, str) else list(at)
            rows = [dict(self.get_timestamp_for(key).coordinates) for key in keys]
        else:
            queried, _ = validate_coordinate_collection(
                at
                if isinstance(at, (list, tuple, np.ndarray, pd.Index, pd.Series))
                else [at]
            )
            rows = [
                dict(
                    self.get_timestamp_at(
                        value, timeline_id, conversion_maps=conversion_maps
                    ).coordinates
                )
                for value in queried
            ]
        return self._tabulate(rows, timeline_filter, conversion_maps)

    def _tabulate(
        self,
        rows: list[dict[str, Coordinate]],
        timeline_filter: set[str] | None,
        conversion_maps: ConversionMapsSpec,
    ) -> pa.Table:
        """Encode cross-sections as one coordinate column per timeline.

        Members come first in member order, then any further timeline a row
        names, each column in its timeline's declared number type.
        """
        names = list(self._timelines)
        for row in rows:
            names.extend(name for name in row if name not in names)
        if timeline_filter is not None:
            names = [name for name in names if name in timeline_filter]

        cells: dict[str, Any] = {
            name: [row.get(name) for row in rows] for name in names
        }
        # The C-Map lane converts whole arrays, so it takes the float mirror of
        # the same cells rather than a second lookup.
        floats: dict[str, Any] = {
            name: np.array(
                [
                    float("nan") if cell is None else float(cell.value)
                    for cell in cells[name]
                ],
                dtype=np.float64,
            )
            for name in names
        }
        return build_timestamp_table(
            self._member_columns(names, cells)
            + self._cmap_columns(names, floats, conversion_maps)
        )

    # endregion

    # region Coordinate Conversion

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Resolve one coordinate on a selected group member axis.

        Args:
            at: Raw, plain, or ID-bearing coordinate input.
            timeline_id: Requested result member.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One coordinate projection or a length-one Series.
        """
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        if isinstance(at, IdCoordinate):
            source_id = at.timeline_id
            result_id = timeline_id or source_id
        else:
            if timeline_id is None:
                raise ValueError(
                    "timeline_id is required for raw or plain Coordinate group queries"
                )
            source_id = timeline_id
            result_id = timeline_id
        if source_id not in self._timelines:
            raise KeyError(
                f"Unknown source timeline ID {source_id!r} in group {self.id!r}"
            )
        if result_id not in self._timelines:
            raise KeyError(
                f"Unknown result timeline ID {result_id!r} in group {self.id!r}"
            )
        if source_id == result_id:
            plain = at.to_coordinate() if isinstance(at, IdCoordinate) else at
            return self._timelines[result_id].get_coordinate_at(
                plain,
                timeline_id=result_id,
                format=format,
                rounding=rounding,
            )
        stamp = self.get_timestamp_at(at)
        return stamp.get_coordinate_for(result_id, format=format, rounding=rounding)

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve a collection of coordinates on one group member axis.

        Args:
            at: Coordinate positions to resolve atomically.
            timeline_id: Requested result member.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        if not values and timeline_id is None:
            raise ValueError("timeline_id is required for an empty group query")
        if timeline_id is not None and timeline_id not in self._timelines:
            raise KeyError(
                f"Unknown result timeline ID {timeline_id!r} in group {self.id!r}"
            )
        results: list[IdCoordinate] = []
        for value in values:
            result = self.get_coordinate_at(
                value,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name=timeline_id
            or (
                results[0].timeline_id
                if results
                and all(
                    result.timeline_id == results[0].timeline_id for result in results
                )
                else "coordinate"
            ),
            empty_number_type=(
                self._timelines[timeline_id].number_type
                if timeline_id is not None
                else None
            ),
        )

    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return a uniquely owned event's start on a selected member axis.

        Args:
            key: Event ID with exactly one owner in the group.
            timeline_id: Requested result member, or the owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One event-start projection or a length-one Series.
        """
        if not isinstance(key, str):
            raise TypeError("get_coordinate_for requires an event-ID string")
        owners = [
            timeline_id_value
            for timeline_id_value, timeline in self._timelines.items()
            if timeline.get_event(key) is not None
        ]
        if not owners:
            raise KeyError(f"Event {key!r} not found in group {self.id!r}")
        if len(owners) > 1:
            raise ValueError(f"Event {key!r} has competing owners {owners}")
        owner = owners[0]
        result_id = timeline_id or owner
        source = self._timelines[owner].get_coordinate_for(
            key, format="id_coordinate", rounding=rounding
        )
        assert isinstance(source, IdCoordinate)
        result = self.get_coordinate_at(
            source,
            timeline_id=result_id,
            format="id_coordinate",
            rounding=rounding,
        )
        assert isinstance(result, IdCoordinate)
        return format_coordinates(
            [result],
            format=format,
            rounding=rounding,
            scalar=True,
            index=pd.Index([key]) if format == "series" else None,
            series_name="coordinate",
        )

    def get_coordinates_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return event-start coordinates for a key collection.

        Args:
            keys: Event IDs to resolve atomically.
            timeline_id: Requested result member, or each owner by default.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_key_collection(keys)
        results: list[IdCoordinate] = []
        for key in values:
            result = self.get_coordinate_for(
                key,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        if format == "series" and index is None:
            index = pd.Index(values)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name="coordinate",
            empty_number_type=(
                self._timelines[timeline_id].number_type
                if timeline_id is not None
                else None
            ),
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a positional or event-key group coordinate query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Requested result member when required.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The selected precise-getter result.
        """
        return dispatch_retrieval(
            self,
            "get_coordinate",
            "get_coordinates",
            at,
            timeline_id,
            format=format,
            rounding=rounding,
        )

    # endregion

    # region Range Queries

    def get_interval_for(self, timeline_id: str) -> Interval:
        """Return a member's extent in the group, ``[0, length]``.

        Every member spans the whole group, so its interval is its own axis
        from 0 to its length, in its unit and declared number type.

        Args:
            timeline_id: The member to query.

        Returns:
            The member's interval ``[0, length]``.

        Raises:
            KeyError: If ``timeline_id`` is not a member.
            UnknownLengthError: If the member's length is unknown.
        """
        if timeline_id not in self._timelines:
            raise KeyError(f"Unknown timeline ID {timeline_id!r} in group {self.id!r}")
        timeline = self._timelines[timeline_id]
        end = timeline._require_length("give the interval of a group member")
        return Interval(
            start=Coordinate(
                express_as(0, timeline.number_type),
                timeline.unit,
                number_type=timeline.number_type,
            ),
            end=end,
        )

    # endregion

    # region Unified Timestamp API (TimeStampSource Protocol)

    def get_timestamp_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp:
        """Get the TimeStamp for a specific event by its ID.

        Searches all timelines in the group for the event and returns
        the corresponding TimeStamp (same structure as
        Timeline.get_timestamp_at).

        Args:
            key: The event's unique identifier.
            timeline_id: Member whose axis the stamp is anchored on; by
                default the member owning the event.
            conversion_maps: Whether C-Maps are reachable through the stamp.

        Returns:
            TimeStamp at the event's coordinate, with access to all
            timelines via `get_coordinate_for()` and C-Maps via `get_unit()`.

        Raises:
            KeyError: If the event is not found in any timeline.

        Examples:
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import ContinuousPhysicalTimeline
            >>> audio = ContinuousPhysicalTimeline(length=150.0, unit="seconds", uid="audio")
            >>> audio.add_events([{"event_type": "Note", "instant": 45.5}])
            >>> group = TimelineGroup(id="my_group", timelines=[audio])

            >>> ts = group.get_timestamp_for("note:000001")
            >>> ts.get_coordinate_for("audio", format="float")
            45.5
            >>> ts.get_unit(TimeUnit.seconds)  # C-Map conversion
            IdCoordinate(45.5, seconds, 'audio')
        """
        if not isinstance(key, str):
            raise TypeError("get_timestamp_for requires an event-ID string")
        for tl_id, tl in self._timelines.items():
            event = tl.get_event(key)
            if event is not None:
                # Read the cell in the representation the member declares --
                # the "value" member alone is the float mirror, and taking it
                # turned an event authored at 5/3 quarters into the dyadic of
                # its double.
                coord = struct_to_coordinate(event["start"], tl.number_type)
                owned = self.get_timestamp_at(
                    coord, tl_id, conversion_maps=conversion_maps
                )
                if timeline_id is None or timeline_id == tl_id:
                    return owned
                # The event's coordinate is a position on ITS OWN axis, so
                # anchoring elsewhere means transferring first -- reading the
                # requested member out of the owner's cross-section -- rather
                # than reinterpreting the same number on another axis.
                return self.get_timestamp_at(
                    owned.get_coordinate_for(timeline_id, format="coordinate"),
                    timeline_id,
                    conversion_maps=conversion_maps,
                )

        raise KeyError(
            f"Event {key!r} not found in any timeline in group. "
            f"Searched timelines: {self.timeline_ids}"
        )

    def get_timestamps_for(
        self,
        keys: KeyCollection,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> list[TimeStamp]:
        """Get timestamps for a collection of event IDs, in input order.

        Args:
            keys: Event IDs to look up.
            timeline_id: Member whose axis every stamp is anchored on; by
                default each event's own owner.
            conversion_maps: Whether C-Maps are reachable through the stamps.

        Returns:
            One TimeStamp per key; an empty collection gives an empty list.

        Raises:
            KeyError: If any event is not found in the group. The batch
                answers completely or not at all.

        Examples:
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import ContinuousPhysicalTimeline
            >>> audio = ContinuousPhysicalTimeline(length=150.0, unit="seconds", uid="audio")
            >>> audio.add_events([{"id": "beat:1", "instant": 45.5}])
            >>> group = TimelineGroup(id="my_group", timelines=[audio])
            >>> stamps = group.get_timestamps_for(["beat:1"])
            >>> stamps[0].get_coordinate_for("audio", format="float")
            45.5
        """
        stamps, _ = resolve_key_collection(
            keys,
            lambda key: self.get_timestamp_for(
                key, timeline_id, conversion_maps=conversion_maps
            ),
        )
        return stamps

    def get_timestamp(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        conversion_maps: ConversionMapsSpec = True,
    ) -> TimeStamp | list[TimeStamp]:
        """Dispatch a positional or event-key timestamp query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Member axis the query is expressed on.
            conversion_maps: Whether C-Maps are reachable through the stamps.

        Returns:
            The selected precise-getter result.

        Raises:
            TypeError: If ``at`` mixes keys and coordinates or is an
                unsupported runtime form.
        """
        return dispatch_retrieval(
            self,
            "get_timestamp",
            "get_timestamps",
            at,
            timeline_id,
            conversion_maps=conversion_maps,
        )

    def _get_unit_map(self, unit: "TimeUnit") -> None:
        """Get a map for unit-based conversion.

        This method is part of the TimeStampSource protocol.
        For TimelineGroup, returns None as groups don't have their own C-Maps.
        Use _get_unit_map_for_timeline() to get C-Maps from member timelines.

        Args:
            unit: Target unit.

        Returns:
            None (use _get_unit_map_for_timeline for member timeline C-Maps).
        """
        return None

    def _get_unit_map_for_timeline(
        self, timeline_id: str, unit: "TimeUnit"
    ) -> "ConversionMap[Any] | None":
        """Get a C-Map for a specific member timeline.

        This method enables TimeStamp.get_unit() to access C-Maps from
        member timelines when the source is a TimelineGroup.

        Args:
            timeline_id: The member timeline to look up C-Map on.
            unit: The target unit.

        Returns:
            The C-Map, or None if not found.
        """
        timeline = self._timelines.get(timeline_id)
        if timeline is not None:
            return timeline._get_unit_map(unit)
        # Member-descendant: locate the owning member and delegate.
        for member in self._timelines.values():
            descendant = member._find_descendant(timeline_id)
            if descendant is not None:
                return descendant._get_unit_map(unit)
        return None

    def _get_unit_maps_for_timeline(
        self, timeline_id: str, unit: "TimeUnit"
    ) -> list[Any]:
        """Get all maps to one unit on a member timeline or descendant."""
        timeline = self._timelines.get(timeline_id)
        if timeline is not None:
            return timeline._get_unit_maps(unit)
        for member in self._timelines.values():
            descendant = member._find_descendant(timeline_id)
            if descendant is not None:
                return descendant._get_unit_maps(unit)
        return []

    def _get_related_timeline_ids(self) -> list[str]:
        """Get IDs of the direct member timelines.

        This method is part of the TimeStampSource protocol. It reports only
        the members; their descendants are reached through
        :meth:`_get_descendant_timeline_ids`.

        Returns:
            List of member timeline IDs.
        """
        return list(self._timelines.keys())

    def _get_descendant_timeline_ids(self) -> list[str]:
        """Get IDs of every member and member-descendant timeline.

        This method is part of the TimeStampSource protocol. Timestamps use it
        to surface conversions from the full subtree of each member.
        """
        ids: list[str] = []
        for member_id, member in self._timelines.items():
            ids.append(member_id)
            ids.extend(member._get_descendant_timeline_ids())
        return ids

    def _get_conversion_maps_for_timeline(
        self, timeline_id: str
    ) -> "list[ConversionMap[Any]]":
        """Get every conversion map attached to a member or its descendants.

        This method is part of the TimeStampSource protocol. It returns C-Maps
        of all kinds, including those with no ``target_unit``, so timestamps
        expose label and structured-value maps across the whole subtree.
        """
        timeline = self._timelines.get(timeline_id)
        if timeline is None:
            for member in self._timelines.values():
                timeline = member._find_descendant(timeline_id)
                if timeline is not None:
                    break
        if timeline is None:
            return []
        return list(timeline._conversion_maps.values())

    def _get_available_units(self) -> list["TimeUnit"]:
        """Get all units available via C-Maps from member timelines.

        This method is part of the TimeStampSource protocol.
        Aggregates all C-Map target units from all member timelines.

        Returns:
            List of all available units from member timeline C-Maps.
        """
        units: set["TimeUnit"] = set()
        for timeline in self._timelines.values():
            units.update(timeline._get_available_units())
        return list(units)

    def _get_unit_for_timeline(self, timeline_id: str) -> "TimeUnit | None":
        """Get the TimeUnit for a timeline in the group.

        This method is part of the TimeStampSource protocol. It enables
        TimeStamp to construct proper Coordinate objects with correct units.

        Args:
            timeline_id: The timeline ID to look up.

        Returns:
            The TimeUnit for the timeline, or None if not found.
        """
        if timeline_id in self._timelines:
            return self._timelines[timeline_id].unit
        for member in self._timelines.values():
            descendant = member._find_descendant(timeline_id)
            if descendant is not None:
                return descendant._unit
        return None

    # endregion

    # region Timeline-like API

    def __len__(self) -> int:
        """Number of timelines."""
        return self.n_timelines

    def __iter__(self) -> Iterator["Timeline"]:
        """Iterate over member timelines."""
        return iter(self._timelines.values())

    def __repr__(self) -> str:
        return f"TimelineGroup(id={self.id!r}, n_timelines={self.n_timelines})"

    def __str__(self) -> str:
        """Return human-readable ASCII diagram of the group.

        Uses the diagram() method to generate a visual representation
        showing all member timelines in a boxed layout.
        """
        return str(self.diagram())

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the ASCII diagram in a monospace pre block so it
        renders correctly in notebook output cells.
        """
        return self.diagram()._repr_html_()

    # endregion

    # region Summary and Serialization

    def summary(self) -> dict[str, Any]:
        """Get a summary of the group.

        Returns:
            Dictionary with group information.
        """
        return {
            "id": self.id,
            "name": self.name,
            "n_timelines": self.n_timelines,
            "timeline_ids": self.timeline_ids,
            "meta": self.meta,
        }

    def to_dict(
        self,
        *,
        minimal: bool = False,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write this group as a TimeSkeleton manifest, an anonymous bundle.

        The group is written with every timeline it needs: ``"timelines"``
        holds the root of each member's tree — a member that is a child
        contributes its root, which holds it — each a timeline object as
        :meth:`Timeline.to_dict <timetoalign.timelines.Timeline.to_dict>`
        writes it, sorted by id. ``"groups"`` holds the group, naming its
        members by id, sorted; ``graphical.order`` keeps their display
        order, which is the order they were added in, under the group's id,
        beside the members of :attr:`styling`. A group read from a manifest
        as a whole writes that manifest's ``meta`` and unknown keys back.

        Args:
            minimal: If True, leave out every key at its default value.
            events: Write every timeline's event rows.
            external_references: Write every timeline's external references.

        Returns:
            ``{"timeskeleton": "0.7.0", ..., "timelines": [...], "groups":
            [{"id", "name", "meta", "members"}], "graphical": {"order":
            {group id: [...]}, ...}}``.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.timelines import TimelineGroup
            >>> from timetoalign.timelines.types import (
            ...     ContinuousLogicalTimeline,
            ...     DiscreteGraphicalTimeline,
            ... )
            >>> image = DiscreteGraphicalTimeline(length=5000)
            >>> part = DiscreteGraphicalTimeline(length=2000)
            >>> image.add_child(part, offset=1000)
            >>> group = TimelineGroup(
            ...     timelines=[part, ContinuousLogicalTimeline(length=Fraction(8))]
            ... )
            >>> data = group.to_dict(minimal=True)
            >>> [timeline["id"] for timeline in data["timelines"]]
            ['clt_1', 'dgt_1']
            >>> data["groups"]
            [{'id': 'group_1', 'members': ['clt_1', 'dgt_2']}]
            >>> data["graphical"]
            {'order': {'group_1': ['dgt_2', 'clt_1']}}
        """
        roots: dict[str, Timeline] = {}
        for member in self._timelines.values():
            root = _root(member)
            roots.setdefault(root.id, root)
        manifest = (
            copy.copy(self._manifest) if self._manifest is not None else Manifest()
        )
        manifest.styling = self.styling
        return manifest.write(
            write_roots(
                list(roots.values()),
                minimal=minimal,
                events=events,
                external_references=external_references,
            ),
            groups=[self._to_group_object(minimal=minimal)],
            order={self.id: self.timeline_ids},
            minimal=minimal,
        )

    def _keep_manifest(self, manifest: Manifest) -> Self:
        """Keep the manifest this group was read from as a whole, and return it.

        Its ``graphical`` styling becomes this group's :attr:`styling`; its
        ``meta`` and unknown keys are written back around the group.
        """
        self.styling, manifest.styling = manifest.styling, {}
        self._manifest = manifest
        return self

    def _to_group_object(self, *, minimal: bool) -> dict[str, Any]:
        """Write the ``Group`` object: id, name, meta and the sorted member ids.

        The one writer of the ``Group`` object, shared by :meth:`to_dict` and
        the bundle's writer.
        """
        return self._dict_form(
            {
                "id": self.id,
                "name": self.name,
                "meta": copy.deepcopy(self.meta),
                "members": sorted(self.timeline_ids),
            },
            minimal=minimal,
        )

    # endregion

    # region Unfolding

    def apply_flow(
        self,
        flow: "Flow",
        flow_controller: "FlowControllerBase",
        reference_timeline_id: str,
        *,
        include_children: bool = True,
        name: str | None = None,
    ) -> "TimelineGroup":
        """Unfold ALL timelines in this group via a single flow.

        Uses the flow controller's repeat structure to compute section
        boundaries in the reference timeline's coordinate space, then
        resolves those boundaries into every other timeline's coordinates
        via the group's linear conversion. Each member is unfolded along its
        own resolved played spans: a new timeline of the member's **same
        concrete type** with one appended child (plus a matching named Region)
        per section, in unfolded coordinates.

        This is the group-level equivalent of
        `timetoalign.timelines.flow.create_unfolded_timeline`, but
        applied to every member at once. It shares the append-children
        assembly via `unfold_via_flowmap`, so each member's children and
        Regions carry the same per-section names as the single-timeline path
        (repeats suffixed ``-rend2``, ``-rend3`` …).

        Args:
            flow: The computed ``Flow`` (from ``controller.compute_flow()``).
            flow_controller: The ``FlowControllerBase`` that produced the flow.
                Required to convert flow sections to quarter-beat coordinates.
            reference_timeline_id: ID of the timeline whose coordinate space
                the flow is defined in (typically the score CLT).  Section
                boundaries are resolved here first, then mapped to all other
                timelines.
            include_children: Whether to recursively slice child timelines
                within each section.
            name: Name for the returned group.  Defaults to
                ``f"{self.name} (unfolded)"``.

        Returns:
            A new ``TimelineGroup`` containing the unfolded timelines. Each
            member keeps its original timeline ID, is the same concrete type as
            its source, and carries one appended child and Region per section.
            A section child is named after its section; its id is minted
            (``clt_2``, ...), so that timeline ids stay unique across the
            group.
            The reference timeline additionally carries a reverse FlowMap
            (id ``"source"``) and a forward FlowMap (id ``f"forward_{flow.id}"``).

        Raises:
            KeyError: If ``reference_timeline_id`` is not in the group.
            ValueError: If the flow controller cannot compute QB sections.

        Examples:
            >>> loader = Ms3Loader.from_file("notes.tsv", "measures.tsv")
            >>> controller = loader.create_flow_controller()
            >>> flow = controller.compute_flow(FlowMode.default)
            >>> score_group = TimelineGroup(
            ...     id="score", timelines=[clt_1, dgt_1, openscore]
            ... )
            >>> unfolded = score_group.apply_flow(flow, controller, "clt_1")
        """
        from timetoalign.timelines.flow import FlowMap, compute_qb_sections

        from .flow.unfolding import unfold_via_flowmap

        if reference_timeline_id not in self._timelines:
            raise KeyError(
                f"Reference timeline '{reference_timeline_id}' not in group. "
                f"Available timelines: {self.timeline_ids}"
            )

        # 1. Section boundaries in QB space, plus the labelled reference map
        #    (its section labels name the children/Regions every member gets).
        qb_sections = compute_qb_sections(flow, flow_controller)
        reference_map = FlowMap.from_qb_sections(flow, qb_sections, id=flow.id)
        labels = [sec.label for sec in reference_map._sections]

        # 2. Resolve each section's boundaries into every member's coordinate
        #    space via the group's timestamps, collecting per-member played
        #    spans as (start, end, label) triples in target order.
        member_spans: dict[str, list[tuple[Any, Any, str | None]]] = {
            tl_id: [] for tl_id in self._timelines
        }
        for i, (qb_start, qb_end) in enumerate(qb_sections):
            start_ts = self.get_timestamp_at(qb_start, reference_timeline_id)
            end_ts = self.get_timestamp_at(qb_end, reference_timeline_id)

            for tl_id in self._timelines:
                try:
                    start_coordinate = start_ts.get_coordinate_for(
                        tl_id, format="coordinate"
                    )
                    end_coordinate = end_ts.get_coordinate_for(
                        tl_id, format="coordinate"
                    )
                except KeyError:
                    raise ValueError(
                        f"Section {i}: cannot resolve boundaries for "
                        f"timeline '{tl_id}' from reference "
                        f"'{reference_timeline_id}'"
                    ) from None
                assert isinstance(start_coordinate, Coordinate)
                assert isinstance(end_coordinate, Coordinate)
                coord_start = start_coordinate.value
                coord_end = end_coordinate.value
                member_spans[tl_id].append((coord_start, coord_end, labels[i]))

        # 3. Unfold each member along its own played spans, reusing the shared
        #    append-children assembly (same-type result, appended children +
        #    Regions), preserving each member's original timeline id.
        result_timelines = []
        for tl_id, tl in self._timelines.items():
            member_map = FlowMap(id=flow.id)
            member_map._build_from_intervals(member_spans[tl_id])
            unfolded = unfold_via_flowmap(
                tl, member_map, uid=tl_id, include_children=include_children
            )
            # A section label names a child; it is no id. Each descendant is
            # given a minted id in the member's tree, so the new group can
            # mint again whatever collides across members.
            taken = {timeline.id for timeline in _tree(unfolded)}
            for _, descendant in list(unfolded._iter_descendants()):
                minted = mint_id(timeline_id_prefix(descendant.unit), taken)
                taken.add(minted)
                descendant._set_id(minted, minted=True)
            result_timelines.append(unfolded)

        # 4. Assemble the new group.
        group_name = name if name is not None else f"{self.name} (unfolded)"
        return TimelineGroup(
            id=f"{self.id}_unfolded" if self.id else None,
            name=group_name,
            timelines=result_timelines,
        )

    # endregion

    # region Display

    def diagram(
        self,
        width: int = 70,
        show_children: bool = True,
        max_children: int = 6,
        unicode: bool = True,
        depth: bool | int = True,
    ) -> "Diagram":
        """Generate ASCII diagram for this group.

        Args:
            width: Total width of the diagram in characters.
            show_children: Whether to expand child timelines.
            max_children: Maximum children per timeline.
            unicode: Use Unicode characters (True) or ASCII fallback (False).
            depth: Child levels to render for each member timeline. ``True``
                renders all levels, ``False`` renders direct children only,
                and a non-negative integer renders at most that many levels.
                In particular, ``0`` renders no child rows.

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).

        Raises:
            ValueError: If ``depth`` is a negative integer.

        Examples:
            >>> print(group.diagram())
            TimelineGroup[my_group] (2 timelines)
            ┌────────────────────────────────────────────────────────────┐
            │ DiscreteGraphicalTimeline[dgt_1:1] (11 events, 5 children)  │
            │ 0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels    │
            │   ├─ system_1     0   ∶∶∶∶∶∶∶                    967       │
            │   └─ ...                                                   │
            └────────────────────────────────────────────────────────────┘
        """
        from timetoalign.display.ascii import group_diagram

        return group_diagram(
            self,
            width=width,
            show_children=show_children,
            max_children=max_children,
            unicode=unicode,
            depth=depth,
        )

    # endregion


# endregion
