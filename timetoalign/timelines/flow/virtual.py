"""Virtual timelines among a group's members, and the route through a group.

A virtual timeline is a statement on the tree it comes from (its
``virtual_timelines`` entry): it has an ID, a unit, a number type and a
length, and holds nothing else. A group admits it as a member through a
:class:`VirtualMember`, which answers for exactly that: an axis from 0 to its
length, without events, children or conversion maps. Its ID belongs to the
manifest of the tree it comes from, so the member's tree is that tree: a group
holding the member holds that tree, and writes it with the group.

A virtual timeline stating ``through`` a group applies the FlowMap of one
member to another member: a position is carried from one to the other by the
group's own conversion between its members.

Examples:
    >>> from timetoalign.timelines import (
    ...     ContinuousLogicalTimeline,
    ...     ContinuousPhysicalTimeline,
    ...     TimelineGroup,
    ... )
    >>> score = ContinuousLogicalTimeline(length=24, uid="score")
    >>> a = score.create_region(0, 16, id="a")
    >>> b = score.create_region(16, 24, id="b")
    >>> played = score.create_virtual_timeline(
    ...     score.create_flow_map([a, a, b]), id="played"
    ... )
    >>> recording = ContinuousPhysicalTimeline(length=80.0, uid="recording")
    >>> group = TimelineGroup(
    ...     id="performance", timelines=[played.as_member(score), recording]
    ... )
    >>> group.timeline_ids
    ['played', 'recording']
    >>> group.get_timestamp_at(20, "played").get_coordinate_for(
    ...     "recording", format="float"
    ... )
    40.0
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, ClassVar

from timetoalign.core import Coordinate, IdCoordinate
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.ids import quote_id

if TYPE_CHECKING:
    from ..base import Timeline
    from ..groups import TimelineGroup, _IdScope
    from .flowmap import CoordinateInput, VirtualTimeline

    # Statically a member answers as a timeline does wherever a group reads
    # its members; at run time only the answers of an axis without content
    # are given (VirtualMember._AXIS_ANSWERS), anything else is no attribute.
    _MemberBase = Timeline
else:
    _MemberBase = object


# region The member


class VirtualMember(_MemberBase):
    """A virtual timeline as a member of a group.

    What :meth:`VirtualTimeline.as_member
    <timetoalign.timelines.flow.VirtualTimeline.as_member>` gives. Inside a
    group a member needs its ID, its unit, its number type, its start 0 and
    its length; a virtual timeline has exactly these and nothing else. The
    member reads them from the virtual timeline its root holds, whenever it
    is asked, so a virtual timeline given a new length answers with it. Its
    positions are those of an axis holding no events, no children and no
    conversion maps: a group's timestamps reach it, a group's table has its
    column, a MatchClaim may name it; nothing can be added to it.

    The member's manifest tree is the tree its virtual timeline comes from,
    which holds its ID: a group or bundle admitting the member admits that
    tree, writes it, and holds the member under the virtual timeline's ID.
    A virtual timeline gives one member object per tree, so it is a member
    of a group once.

    Attributes:
        id: The virtual timeline's ID.

    Examples:
        >>> from timetoalign.timelines import ContinuousLogicalTimeline
        >>> score = ContinuousLogicalTimeline(length=24, uid="score")
        >>> a = score.create_region(0, 16, id="a")
        >>> b = score.create_region(16, 24, id="b")
        >>> played = score.create_virtual_timeline(
        ...     score.create_flow_map([a, a, b]), id="played"
        ... )
        >>> member = played.as_member(score)
        >>> member.id, member.length, member is played.as_member(score)
        ('played', Coordinate(Fraction(40, 1), quarters), True)
        >>> member.get_events().table.num_rows
        0
    """

    _AXIS_ANSWERS: ClassVar[frozenset[str]] = frozenset(
        {
            "_all_conversion_maps",
            "_child_offsets",
            "_children",
            "_conversion_maps",
            "_convert_coordinate_to_self",
            "_find_descendant",
            "_get_available_units",
            "_get_child_coordinate",
            "_get_descendant_timeline_ids",
            "_get_number_type_for_timeline",
            "_get_unit_for_timeline",
            "_get_unit_maps",
            "_holds_position",
            "_iter_descendants",
            "_require_length",
            "domain",
            "get_coordinate",
            "get_coordinate_at",
            "get_coordinate_for",
            "get_coordinates_at",
            "get_coordinates_for",
            "get_event",
            "get_events",
            "get_timestamp",
            "get_timestamp_at",
            "get_timestamp_for",
            "get_timestamps_at",
            "get_timestamps_for",
            "is_continuous",
            "is_discrete",
            "iter_children",
            "iter_regions",
            "n_children",
            "n_conversion_maps",
            "n_events",
            "n_regions",
            "origin",
        }
    )
    """What the member answers as an axis of its unit, number type and length
    answers: a timeline holding nothing (the conversion maps, the children,
    the events, the positions)."""

    _members: ClassVar[weakref.WeakValueDictionary[tuple[int, str], VirtualMember]] = (
        weakref.WeakValueDictionary()
    )
    """The member of each virtual timeline of each live root: a member keeps
    its root alive, so a root's identity is not reused while its member
    lives."""

    _parent: Timeline | None = None
    """A virtual timeline is no child: it has no parent."""

    _virtual_id: str
    _root: Timeline
    _axis_timeline: Timeline | None

    @classmethod
    def of(cls, virtual_id: str, root: Timeline) -> VirtualMember:
        """Return the member of the virtual timeline *virtual_id* of *root*.

        Args:
            virtual_id: The ID of a virtual timeline *root* holds.
            root: The root timeline holding it.

        Raises:
            KeyError: If *root* holds no such virtual timeline.
        """
        root.get_virtual_timeline(virtual_id)
        key = (id(root), virtual_id)
        member = cls._members.get(key)
        if member is None:
            member = cls(virtual_id, root)
            cls._members[key] = member
        return member

    def __init__(self, virtual_id: str, root: Timeline) -> None:
        """Make the member; :meth:`of` gives the one member of a virtual timeline.

        Args:
            virtual_id: The ID of a virtual timeline *root* holds.
            root: The root timeline holding it.
        """
        self._virtual_id = virtual_id
        self._root = root
        self._axis_timeline = None

    def __getattr__(self, name: str) -> Any:
        """Answer as an axis of the member's unit, number type and length would.

        Raises:
            AttributeError: For anything an axis without content does not
                answer: a virtual timeline holds nothing but its axis.
        """
        if name in VirtualMember._AXIS_ANSWERS:
            return getattr(self._axis(), name)
        if name.startswith("__") or "_virtual_id" not in self.__dict__:
            raise AttributeError(name)
        raise AttributeError(
            f"virtual timeline {quote_id(self._virtual_id)} is a member of a group "
            f"as an axis only and has no {name!r}: it holds no events, children "
            "or conversion maps"
        )

    def __repr__(self) -> str:
        return (
            f"VirtualMember({self.id!r}, unit={self.unit}, length={self.length}, "
            f"root={self._root.id!r})"
        )

    def __str__(self) -> str:
        return f"virtual timeline {quote_id(self.id)} [0, {self.length}]"

    @property
    def id(self) -> str:
        """The virtual timeline's ID."""
        return self._virtual_id

    @property
    def virtual(self) -> VirtualTimeline:
        """The virtual timeline, as its root holds it now."""
        return self._root.get_virtual_timeline(self._virtual_id)

    @property
    def name(self) -> str:
        """The virtual timeline's name, else its ID."""
        name = self.virtual.name
        return self._virtual_id if name is None else name

    @name.setter
    def name(self, value: str | None) -> None:
        """Refuse: the name is the virtual timeline's, stated where its root holds it."""
        raise AttributeError(
            f"the name of virtual timeline {quote_id(self._virtual_id)} is set on "
            "the virtual timeline its root holds, not on its member"
        )

    @property
    def _name(self) -> str | None:
        """The virtual timeline's name as stated, or None."""
        return self.virtual.name

    @property
    def unit(self) -> TimeUnit:
        """The unit of the virtual timeline's axis."""
        return self.virtual.unit

    @property
    def number_type(self) -> NumberType:
        """The number type of the virtual timeline's axis."""
        return self.virtual.number_type

    @property
    def length(self) -> Coordinate:
        """The virtual timeline's length: always known."""
        return self.virtual.length

    @length.setter
    def length(self, value: Any) -> None:
        """Refuse: the length is the virtual timeline's, which its map gives."""
        raise AttributeError(
            f"the length of virtual timeline {quote_id(self._virtual_id)} is the "
            "length its map gives, never set on its member"
        )

    @property
    def class_name(self) -> str:
        """The kind of member, as a display names it."""
        return "VirtualTimeline"

    def _is_segment_line_instance(self) -> bool:
        """A virtual timeline is no segment line."""
        return False

    def _tree_root(self) -> Timeline:
        """Return the root of the tree the virtual timeline comes from."""
        return self._root

    def _join_scope(self, container: _IdScope) -> None:
        """Record on the tree the member comes from that it joined *container*."""
        self._root._join_scope(container)

    def _leave_scope(self, container: _IdScope) -> None:
        """Record that the member left *container*.

        The tree it comes from stays in *container* while the root itself or
        another virtual timeline of it is a member there.
        """
        root = self._root
        if not any(
            member is not self
            and member._tree_root() is root
            and (member is root or isinstance(member, VirtualMember))
            for member in container._scope_members()
        ):
            root._leave_scope(container)

    def _drop_references(self, ids: Any) -> None:
        """Drop the elements of *ids* that left the manifest from the
        statements of the tree the member comes from."""
        self._root._drop_references(ids)

    def _axis(self) -> Timeline:
        """The axis the member answers positions on: a timeline holding nothing.

        Built in the member's unit and number type, with its ID and length,
        and built again when the virtual timeline's length changes.
        """
        from ..types import get_timeline_class

        virtual = self.virtual
        axis = self._axis_timeline
        if axis is None or axis.length != virtual.length:
            unit = virtual.unit
            timeline_class = get_timeline_class(
                unit.domain.value, discrete=unit.is_discrete
            )
            axis = timeline_class(
                length=virtual.length,
                unit=unit,
                number_type=virtual.number_type,
                uid=self._virtual_id,
            )
            self._axis_timeline = axis
        return axis


def _member_named(timelines: dict[str, Timeline], member_id: str) -> Timeline:
    """Return the timeline or virtual timeline a group's ``members`` names.

    Args:
        timelines: Every timeline of a manifest, roots and descendants, by
            ID; the roots hold the manifest's virtual timelines.
        member_id: A member of a ``Group`` object.

    Raises:
        KeyError: If neither a timeline nor a virtual timeline has the ID.
    """
    if member_id in timelines:
        return timelines[member_id]
    for timeline in timelines.values():
        if member_id in timeline._virtual_timelines:
            return VirtualMember.of(member_id, timeline)
    raise KeyError(
        f"{quote_id(member_id)} names no timeline and no virtual timeline. "
        f"Available timelines: {list(timelines)}"
    )


# endregion


# region The route through a group


def _through_group(root: Timeline, group_id: str, what: str) -> TimelineGroup:
    """Return the group named *group_id* that a timeline of *root*'s tree is in.

    Args:
        root: The root of the tree a virtual timeline comes from.
        group_id: The ID its ``through`` states.
        what: How a message names the virtual timeline.

    Raises:
        ValueError: If no group of that ID holds a timeline of the tree (a
            conversion map's ID among them: carrying a FlowMap through one is
            not evaluated), or several do.
    """
    from ..groups import TimelineGroup

    found: list[TimelineGroup] = []
    for container in root._scope_containers():
        if isinstance(container, TimelineGroup) and container.id == group_id:
            if not any(group is container for group in found):
                found.append(container)
    if not found:
        raise ValueError(
            f"{what} goes through {quote_id(group_id)}, which is no group holding "
            f"a timeline of the tree of {quote_id(root.id)}: carrying a FlowMap "
            "through anything else (a conversion map) is not supported yet"
        )
    if len(found) > 1:
        raise ValueError(
            f"{what} goes through group {quote_id(group_id)}, and {len(found)} "
            "groups of that ID hold timelines of its tree"
        )
    return found[0]


def _carry_through(
    group: TimelineGroup,
    coordinate: CoordinateInput,
    source_id: str,
    target_id: str,
) -> Coordinate:
    """Carry a position from one timeline of a group to another.

    Each of the two is a member of the group or a descendant of one. The
    position goes from *source_id* to the member holding it by exact offset
    arithmetic, from that member to every other by the group's own
    conversion between its members (:meth:`TimelineGroup.get_timestamp_at
    <timetoalign.timelines.TimelineGroup.get_timestamp_at>`), and is read
    on *target_id*: in its unit and declared number type.

    Args:
        group: The group.
        coordinate: A position on *source_id*: a bare value, a Coordinate in
            its unit or an IdCoordinate naming it.
        source_id: The timeline the position is on.
        target_id: The timeline it is carried to.

    Returns:
        The position on *target_id*.

    Raises:
        KeyError: If *source_id* or *target_id* is neither a member nor
            inside one.
        ValueError: If the position lies outside *source_id*, an
            IdCoordinate names another timeline, or the position does not
            reach *target_id* (outside the child of a member).

    Examples:
        >>> from fractions import Fraction
        >>> from timetoalign.timelines import ContinuousLogicalTimeline, TimelineGroup
        >>> edition = ContinuousLogicalTimeline(length=30, uid="edition")
        >>> score = ContinuousLogicalTimeline(length=24, uid="score")
        >>> group = TimelineGroup(id="scores", timelines=[edition, score])
        >>> _carry_through(group, 5, "edition", "score")
        Coordinate(Fraction(4, 1), quarters)
        >>> _carry_through(group, Fraction(18), "score", "edition")
        Coordinate(Fraction(45, 2), quarters)
    """
    source = group._resolve_member(source_id)
    if isinstance(coordinate, IdCoordinate) and coordinate.timeline_id != source_id:
        raise ValueError(
            f"{coordinate!r} is no position of timeline {quote_id(source_id)}"
        )
    position = source.get_coordinate_at(coordinate, format="coordinate")
    assert isinstance(position, Coordinate)
    members = list(group)
    member = source
    while not any(member is held for held in members):
        parent = member._parent
        assert parent is not None, "a timeline inside a member has the member above it"
        member = parent
    if member is not source:
        position = member.get_coordinate_at(
            IdCoordinate.from_coordinate(position, source.id), format="coordinate"
        )
        assert isinstance(position, Coordinate)
    reached = group.get_timestamp_at(position, member.id, conversion_maps=False)
    if target_id not in reached.coordinates:
        group._resolve_member(target_id)
        raise ValueError(
            f"{position} on {quote_id(member.id)} does not reach "
            f"{quote_id(target_id)} in group {quote_id(group.id)}"
        )
    carried = reached.coordinates[target_id]
    return Coordinate(carried.value, carried.unit, number_type=carried.number_type)


# endregion
