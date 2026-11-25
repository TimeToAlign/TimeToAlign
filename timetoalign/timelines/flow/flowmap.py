"""FlowMaps: one order in which stretches of a root timeline are played.

A :class:`FlowMap` is written as an ordered list of **chunks**: each the ID
of something giving an interval inside the tree of one root timeline (a
Region of the root or of a descendant, a child timeline at any depth, a
virtual timeline of that tree), or a :class:`~timetoalign.core.Gap`, a
stretch nothing of the timeline fills. It needs timelines and nothing else:
no flow control, no measures, no events. A reading, an order of movements
and a set list are FlowMaps written by hand; the flows computed from flow
control are FlowMaps whose chunks are the atomic sections
(:meth:`Timeline.create_flow_maps_from_flow_control
<timetoalign.timelines.Timeline.create_flow_maps_from_flow_control>`).

Resolved against its root (:meth:`FlowMap.resolve`), a FlowMap is a
:class:`ResolvedFlowMap`: the chunks as intervals of the ROOT's axis, by
exact offset arithmetic from parent to child, each placed where the one
before it ends on the **virtual axis**. That axis starts at 0 with the first
chunk, has the root's unit and number type, and is as long as the chunks and
gaps together. A root position is played wherever a chunk covers it
(:meth:`ResolvedFlowMap.unfold`, any number of positions); a virtual position
is the root position played there (:meth:`ResolvedFlowMap.fold`), or none
inside a gap.

A :class:`VirtualTimeline` is the statement that a map was applied to a
timeline, giving a timeline with an ID of its own of which nothing else is
written: no events, no children, no flow control.

Examples:
    >>> from timetoalign.core import Gap
    >>> from timetoalign.timelines import ContinuousLogicalTimeline
    >>> concert = ContinuousLogicalTimeline(length=20, uid="set")
    >>> _ = concert.create_child(8, 0, uid="first")
    >>> _ = concert.create_child(12, 8, uid="second")
    >>> encore = concert.create_flow_map(["second", Gap(4), "first"], name="Encore order")
    >>> played = encore.resolve(concert)
    >>> played.length
    Coordinate(Fraction(24, 1), quarters)
    >>> played.unfold(2), played.fold(2), played.fold(13)
    ([Coordinate(Fraction(18, 1), quarters)], Coordinate(Fraction(10, 1), quarters), None)
    >>> encore.to_dict()
    {'id': 'flow_map_1', 'timeline': 'set', 'chunks': ['second', {'gap': [4, 1]}, 'first'], \
'name': 'Encore order'}
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Self

from timetoalign.core import Coordinate, Duration, Gap, IdCoordinate, Interval
from timetoalign.core.enums import FlowConvention, NumberType, TimeUnit
from timetoalign.core.ids import (
    _IdentifiedElement,
    mint_id,
    quote_id,
    timeline_id_prefix,
    validate_id,
)
from timetoalign.core.provenance import Provenance
from timetoalign.core.time import express_scalar_as, number_from_json, number_to_json

from ..flowcontrol import _read_provenance_entry
from ..manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_name,
    require_object,
)
from .engine import FlowFinding

if TYPE_CHECKING:
    from ..base import Timeline
    from ..regions import Region

_Number = int | float | Fraction
CoordinateInput = _Number | Coordinate
"""A position given to a FlowMap: a bare value on the axis, a
:class:`~timetoalign.core.Coordinate` or an
:class:`~timetoalign.core.IdCoordinate`."""


# region The FlowMap as written


@dataclass(frozen=True)
class FlowMap(_IdentifiedElement):
    """One order in which stretches of one root timeline are played.

    The FlowMap as it is written: it names its chunks and holds no
    interval. :meth:`resolve` reads it against its root timeline.

    Attributes:
        timeline: The ID of the ONE root timeline whose stretches are played.
        chunks: The chunks in the order played. Each is the ID of a Region
            of the root or of a descendant, of a child timeline at any
            depth, or of a virtual timeline of that tree, or a
            :class:`~timetoalign.core.Gap`. A chunk may occur any number of
            times; the root itself is no chunk.
        convention: Which flow of the timeline's flow control the FlowMap
            states: ``default``, ``single`` or ``printed``
            (:class:`~timetoalign.core.enums.FlowConvention`); any other
            word is kept as given. None on a FlowMap that follows none, such
            as one written by hand.
        canonical: Whether this is the canonical FlowMap of its timeline.
        name: Optional display name.
        provenance: How the FlowMap was made and who believes it: any
            number of :class:`~timetoalign.core.provenance.Provenance`
            entries, as the manifest lists them.
        id: The FlowMap's ID; minted as ``flow_map_{N}`` when not given.

    Raises:
        ValueError: If the timeline or a chunk is no valid ID, a chunk names
            the timeline itself or is neither an ID nor a gap, or a
            provenance entry is invalid.

    Examples:
        >>> from timetoalign.core import Gap
        >>> reading = FlowMap("score", ["region_2", Gap(4), "region_1"])
        >>> reading.id, reading.convention, reading.canonical
        ('flow_map_1', None, False)
        >>> FlowMap.from_dict(reading.to_dict()) == reading
        True
    """

    _id_prefix: ClassVar[str] = "flow_map"
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "timeline",
        "chunks",
        "convention",
        "canonical",
        "name",
        "provenance",
    )
    _manifest_path: ClassVar[tuple[str, ...]] = ("flow_maps",)

    timeline: str
    chunks: tuple[str | Gap, ...]
    convention: FlowConvention | str | None = field(default=None, kw_only=True)
    canonical: bool = field(default=False, kw_only=True)
    name: str | None = field(default=None, kw_only=True)
    provenance: tuple[Provenance, ...] = field(default=(), kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowMap:
        """Restore a FlowMap from its object in a manifest's ``flow_maps``.

        IDs are bare, as every object of the library holds them. A chunk is
        an ID or the object ``{gap: length}``, the length read as its JSON
        form says (an integer, a float, a Fraction pair). ``convention``,
        ``canonical``, ``name`` and ``provenance`` may be left out. Keys the
        library does not know are kept, a gap's too, and written back.

        Args:
            data: The FlowMap object.

        Raises:
            ValueError: If the object is malformed.
        """
        require_object(data, "a FlowMap", ("id", "timeline", "chunks"))
        flow_map_id = read_id(data["id"], "a FlowMap")
        what = f"FlowMap {quote_id(flow_map_id)}"
        if not isinstance(data["chunks"], list):
            raise ValueError(f"the chunks of {what} must be an array")
        entries = data.get("provenance", [])
        if not isinstance(entries, list):
            raise ValueError(f"the provenance of {what} must be an array")
        try:
            provenance = tuple(_read_provenance_entry(entry) for entry in entries)
        except (TypeError, ValueError) as error:
            raise ValueError(f"the provenance of {what}: {error}") from error
        flow_map = cls(
            read_id(data["timeline"], f"the timeline of {what}"),
            tuple(_read_chunk(chunk, what) for chunk in data["chunks"]),
            convention=data.get("convention"),
            canonical=data.get("canonical", False),
            name=read_name(data, what),
            provenance=provenance,
            id=flow_map_id,
        )
        flow_map._keep_unknown_keys(data)
        return flow_map

    def __post_init__(self) -> None:
        """Mint or validate the ID, then validate and settle the members."""
        super().__post_init__()
        what = f"FlowMap {quote_id(self.id)}"
        validate_id(self.timeline)
        chunks = tuple(self.chunks)
        for chunk in chunks:
            if isinstance(chunk, Gap):
                continue
            if not isinstance(chunk, str):
                raise ValueError(
                    f"a chunk of {what} is an ID or a Gap, got {chunk!r}: name a "
                    "region, a child timeline or a virtual timeline by its ID"
                )
            validate_id(chunk)
            if chunk == self.timeline:
                raise ValueError(
                    f"{what} names its own timeline {quote_id(chunk)} as a chunk: "
                    "a chunk is a region, a child timeline or a virtual timeline "
                    "inside its tree"
                )
        if not isinstance(self.canonical, bool):
            raise ValueError(f"canonical of {what} is true or false")
        if self.convention is not None and not isinstance(self.convention, str):
            raise ValueError(f"the convention of {what} is a word")
        convention = self.convention
        if convention in FlowConvention.__members__:
            convention = FlowConvention[str(convention)]
        entries = self.provenance
        if isinstance(entries, Provenance | Mapping):
            entries = (entries,)
        object.__setattr__(self, "chunks", chunks)
        object.__setattr__(self, "convention", convention)
        object.__setattr__(
            self,
            "provenance",
            tuple(
                (
                    entry
                    if isinstance(entry, Provenance)
                    else _read_provenance_entry(entry)
                )
                for entry in entries
            ),
        )

    @property
    def is_computed(self) -> bool:
        """Whether the FlowMap states a flow computed from flow control.

        It names one of the three conventions
        (:class:`~timetoalign.core.enums.FlowConvention`) and an entry of its
        provenance gives the method ``inferred``. A FlowMap written by hand,
        stated by a source or following a convention of another name is
        none.
        """
        return isinstance(self.convention, FlowConvention) and any(
            entry.method == "inferred" for entry in self.provenance
        )

    def resolve(self, root: Timeline) -> ResolvedFlowMap:
        """Read the FlowMap against its root timeline.

        Each chunk becomes its interval on the ROOT's axis: a Region its
        extent, a child timeline its whole length, each carried up to the
        root by the offsets of the timelines between (exact offset
        arithmetic); a virtual timeline of the tree the stretches of the
        FlowMap it applies, in their order. A gap keeps its length. The
        stretches are placed one after the other on the virtual axis.

        Args:
            root: The timeline the FlowMap names.

        Returns:
            The FlowMap as placed stretches.

        Raises:
            ValueError: If *root* is a child timeline or not the timeline
                named; a chunk names nothing in the root's tree; a chunk is
                a virtual timeline whose unit is not the root's (every
                region and child of a tree is in the root's unit), or one
                that cannot be read (it applies no FlowMap of the root,
                another timeline than the root's, or goes ``through``
                something); a gap's length is no value of the root's number
                type; or the chunks name one another in a circle.
            UnknownLengthError: If a child timeline named has no known
                length.
        """
        return _resolve(self, root, ())

    def _on_axis(self, root: Timeline) -> FlowMap:
        """Return the FlowMap with every gap in the number type of *root*.

        Raises:
            ValueError: If a gap's length is no value of that number type.
        """
        gaps = {
            index: chunk
            for index, chunk in enumerate(self.chunks)
            if isinstance(chunk, Gap)
        }
        lengths = {
            index: _gap_length(gap, root, self).value for index, gap in gaps.items()
        }
        if all(type(lengths[index]) is type(gap.length) for index, gap in gaps.items()):
            return self
        return replace(
            self,
            chunks=tuple(
                replace(gaps[index], length=lengths[index]) if index in gaps else chunk
                for index, chunk in enumerate(self.chunks)
            ),
        )

    def _with_references_renamed(self, renamed: Mapping[str, str]) -> FlowMap:
        """Return the FlowMap naming its timeline and chunks by their new IDs."""
        timeline = renamed.get(self.timeline, self.timeline)
        chunks = tuple(
            renamed.get(chunk, chunk) if isinstance(chunk, str) else chunk
            for chunk in self.chunks
        )
        if timeline == self.timeline and chunks == self.chunks:
            return self
        return replace(self, timeline=timeline, chunks=chunks)

    def _dict_fields(self) -> dict[str, Any]:
        """Return the FlowMap's keys; ``convention`` and ``provenance`` are
        written only when stated."""
        fields: dict[str, Any] = {
            "id": self.id,
            "timeline": self.timeline,
            "chunks": [
                (
                    chunk
                    if isinstance(chunk, str)
                    else chunk._dict_form(
                        {"gap": number_to_json(chunk.length)}, minimal=False
                    )
                )
                for chunk in self.chunks
            ],
        }
        if self.convention is not None:
            fields["convention"] = str(self.convention)
        fields.update(canonical=self.canonical, name=self.name)
        if self.provenance:
            fields["provenance"] = [entry.to_dict() for entry in self.provenance]
        return fields

    def __repr__(self) -> str:
        details = "".join(
            f", {key}={value!r}"
            for key, value in (
                ("convention", self.convention and str(self.convention)),
                ("name", self.name),
            )
            if value is not None
        )
        return (
            f"FlowMap({self.id!r}, timeline={self.timeline!r}, "
            f"{len(self.chunks)} chunks{details})"
        )


def _read_chunk(data: Any, what: str) -> str | Gap:
    """Read one entry of ``chunks``: an ID, or the object ``{gap: length}``.

    Raises:
        ValueError: If the entry is neither.
    """
    if isinstance(data, str):
        return read_id(data, f"a chunk of {what}")
    if not isinstance(data, dict) or "gap" not in data:
        raise ValueError(f"a chunk of {what} is an ID or {{gap: length}}, got {data!r}")
    try:
        gap = Gap(number_from_json(data["gap"]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"a gap of {what}: {error}") from error
    gap._keep_unknown_keys(data)
    return gap


# endregion


# region The FlowMap as resolved


@dataclass(frozen=True)
class PlacedChunk:
    """A stretch of the root timeline, placed on the virtual axis.

    Attributes:
        source: The stretch played, a half-open interval of the root's axis.
        start: Where it begins on the virtual axis.
    """

    source: Interval
    start: Coordinate

    @property
    def length(self) -> Duration:
        """How long the stretch is, on either axis."""
        return self.source.duration

    @property
    def end(self) -> Coordinate:
        """Where it ends on the virtual axis, exclusive."""
        return self.start + self.length


@dataclass(frozen=True)
class PlacedGap:
    """A stretch of the virtual axis that nothing of the root fills.

    Attributes:
        length: How long the gap is.
        start: Where it begins on the virtual axis.
    """

    length: Duration
    start: Coordinate

    @property
    def end(self) -> Coordinate:
        """Where it ends on the virtual axis, exclusive."""
        return self.start + self.length


@dataclass(frozen=True)
class ResolvedFlowMap:
    """A played order as stretches of a root timeline's axis, placed in order.

    What :meth:`FlowMap.resolve` returns, and, as a :class:`PlayedFlow`,
    what :meth:`Timeline.get_flow <timetoalign.timelines.Timeline.get_flow>`
    computes from flow control without writing anything. The virtual axis
    starts at 0 with the first stretch and has the root's unit and number
    type; every position in and out is a
    :class:`~timetoalign.core.Coordinate` in that number type.

    Attributes:
        timeline: The ID of the timeline whose stretches are played.
        unit: The unit of both axes.
        number_type: The number type of both axes.
        chunks: The stretches in the order played, each placed where the one
            before it ends.
    """

    timeline: str
    unit: TimeUnit
    number_type: NumberType
    chunks: tuple[PlacedChunk | PlacedGap, ...]

    @classmethod
    def from_stretches(
        cls,
        timeline: str,
        stretches: Iterable[Interval | Duration],
        *,
        unit: TimeUnit,
        number_type: NumberType,
        **stated: Any,
    ) -> Self:
        """Place stretches one after the other on a virtual axis.

        Args:
            timeline: The ID of the timeline the intervals lie on.
            stretches: In the order played, an
                :class:`~timetoalign.core.Interval` of the timeline's axis
                for a stretch played and a
                :class:`~timetoalign.core.Duration` for a gap.
            unit: The timeline's unit.
            number_type: The timeline's number type.
            **stated: What a subclass states beside the chunks (the
                ``findings`` of a :class:`PlayedFlow`).
        """
        position = Coordinate(0, unit, number_type=number_type)
        placed: list[PlacedChunk | PlacedGap] = []
        for stretch in stretches:
            chunk = (
                PlacedChunk(stretch, position)
                if isinstance(stretch, Interval)
                else PlacedGap(stretch, position)
            )
            placed.append(chunk)
            position = chunk.end
        return cls(timeline, unit, number_type, tuple(placed), **stated)

    @property
    def length(self) -> Coordinate:
        """The length of the virtual axis: the chunks and gaps together."""
        if not self.chunks:
            return Coordinate(0, self.unit, number_type=self.number_type)
        return self.chunks[-1].end

    def unfold(self, coordinate: CoordinateInput) -> list[Coordinate]:
        """Return every position of the virtual axis a root position is played at.

        Args:
            coordinate: A position on the root's axis: a bare value, a
                Coordinate in the root's unit, or an IdCoordinate naming the
                root.

        Returns:
            The virtual positions in the order played: one per stretch
            covering the position (a stretch is half-open, its end excluded),
            none where it is never played.

        Raises:
            ValueError: If a Coordinate is in another unit, or an
                IdCoordinate names another timeline.
        """
        if (
            isinstance(coordinate, IdCoordinate)
            and coordinate.timeline_id != self.timeline
        ):
            raise ValueError(
                f"{coordinate!r} is no position of timeline {quote_id(self.timeline)}"
            )
        value = self._value(coordinate)
        return [
            self._coordinate(chunk.start.value + (value - chunk.source.start.value))
            for chunk in self.chunks
            if isinstance(chunk, PlacedChunk)
            and chunk.source.start.value <= value < chunk.source.end.value
        ]

    def fold(self, coordinate: CoordinateInput) -> Coordinate | None:
        """Return the root position played at a position of the virtual axis.

        Args:
            coordinate: A position on the virtual axis: a bare value or a
                Coordinate in its unit.

        Returns:
            The position on the root's axis, or None inside a gap.

        Raises:
            ValueError: If the position lies outside the virtual axis, which
                is half-open (``[0, length)``), a Coordinate is in another
                unit, or an IdCoordinate names the root timeline, whose
                positions are not virtual positions.
        """
        if (
            isinstance(coordinate, IdCoordinate)
            and coordinate.timeline_id == self.timeline
        ):
            raise ValueError(
                f"{coordinate!r} is a position of timeline {quote_id(self.timeline)}, "
                "not of the virtual axis: unfold it"
            )
        value = self._value(coordinate)
        for chunk in self.chunks:
            if chunk.start.value <= value < chunk.end.value:
                if isinstance(chunk, PlacedGap):
                    return None
                return self._coordinate(
                    chunk.source.start.value + (value - chunk.start.value)
                )
        raise ValueError(
            f"{self._coordinate(value)} lies outside the virtual axis, which runs "
            f"from 0 to {self.length}, its end excluded"
        )

    def _coordinate(self, value: _Number) -> Coordinate:
        """A coordinate of either axis."""
        return Coordinate(value, self.unit, number_type=self.number_type)

    def _value(self, coordinate: CoordinateInput) -> _Number:
        """Read a position as a bare value in the axes' number type.

        Raises:
            ValueError: If a Coordinate is in another unit.
        """
        if not isinstance(coordinate, Coordinate):
            return self._coordinate(coordinate).value
        if coordinate.unit != self.unit:
            raise ValueError(
                f"{coordinate!r} is in {coordinate.unit}; the axis of timeline "
                f"{quote_id(self.timeline)} is in {self.unit}"
            )
        return express_scalar_as(coordinate, self.number_type).value

    def __repr__(self) -> str:
        return (
            f"ResolvedFlowMap(timeline={self.timeline!r}, {len(self.chunks)} chunks, "
            f"length={self.length})"
        )


@dataclass(frozen=True, repr=False)
class PlayedFlow(ResolvedFlowMap):
    """The flow a timeline's flow control plays, and what playing it found.

    What :meth:`Timeline.get_flow <timetoalign.timelines.Timeline.get_flow>`
    returns: the runs of the flow placed one after the other, through which
    positions convert as through any :class:`ResolvedFlowMap`, together
    with the findings of the engine.

    Attributes:
        findings: What playing the flow control found, in the order found:
            a flow that never ends, a flow ending at a Break, two
            destinations holding together, a stretch never played. Only
            the ``default`` flow has findings.
    """

    findings: tuple[FlowFinding, ...] = ()

    def __repr__(self) -> str:
        count = len(self.findings)
        return (
            f"PlayedFlow(timeline={self.timeline!r}, {len(self.chunks)} chunks, "
            f"length={self.length}, {count} finding{'' if count == 1 else 's'})"
        )


def _resolve(
    flow_map: FlowMap, root: Timeline, visiting: tuple[str, ...]
) -> ResolvedFlowMap:
    """Resolve *flow_map* against *root*; *visiting* holds the FlowMaps being
    resolved around it, so that a circle of chunks is refused."""
    what = f"FlowMap {quote_id(flow_map.id)}"
    if root._parent is not None:
        raise ValueError(
            f"{what} cannot be read on timeline {quote_id(root.id)}, a child of "
            f"{quote_id(root._parent.id)}: a FlowMap names a root timeline"
        )
    if root.id != flow_map.timeline:
        raise ValueError(
            f"{what} is a FlowMap of timeline {quote_id(flow_map.timeline)}, not of "
            f"{quote_id(root.id)}"
        )
    if flow_map.id in visiting:
        raise ValueError(
            f"{what} names itself as a chunk, through the virtual timelines its "
            "chunks name"
        )
    visiting = (*visiting, flow_map.id)
    stretches: list[Interval | Duration] = []
    for chunk in flow_map.chunks:
        if isinstance(chunk, Gap):
            stretches.append(_gap_length(chunk, root, flow_map))
            continue
        interval = _interval_in_tree(root, chunk)
        if interval is not None:
            stretches.append(interval)
            continue
        virtual = root._virtual_timelines.get(chunk)
        if virtual is None:
            raise ValueError(
                f"chunk {quote_id(chunk)} of {what} names no region, child "
                f"timeline or virtual timeline in the tree of {quote_id(root.id)}"
            )
        if virtual.unit != root.unit:
            raise ValueError(
                f"chunk {quote_id(chunk)} of {what} is a virtual timeline in "
                f"{virtual.unit}; the axis of its root {quote_id(root.id)} is in "
                f"{root.unit}"
            )
        stretches.extend(
            placed.source if isinstance(placed, PlacedChunk) else placed.length
            for placed in virtual._resolved(root, visiting).chunks
        )
    return ResolvedFlowMap.from_stretches(
        root.id, stretches, unit=root.unit, number_type=root.number_type
    )


def _gap_length(gap: Gap, root: Timeline, flow_map: FlowMap) -> Duration:
    """The length of *gap* on the axis of *root*, in its number type.

    Raises:
        ValueError: If the length is no value of that number type.
    """
    length = Duration(gap.length, root.unit, number_type=root.number_type)
    if length.value != gap.length:
        raise ValueError(
            f"the gap of {gap.length} in FlowMap {quote_id(flow_map.id)} is no "
            f"{root.number_type} value, the number type of timeline "
            f"{quote_id(root.id)}"
        )
    return length


def _tree_origins(root: Timeline) -> Iterator[tuple[Timeline, _Number]]:
    """Yield every timeline of a tree with where its origin lies on the root's axis.

    The offsets from the root down are added up in the root's number type;
    every timeline of a tree is in the root's unit.
    """
    pending: list[tuple[Timeline, _Number]] = [
        (root, Coordinate(0, root.unit, number_type=root.number_type).value)
    ]
    while pending:
        timeline, origin = pending.pop()
        yield timeline, origin
        pending.extend(
            (
                child,
                origin
                + express_scalar_as(
                    timeline._child_offsets[child_id], root.number_type
                ).value,
            )
            for child_id, child in timeline._children.items()
        )


def _interval_in_tree(root: Timeline, chunk: str) -> Interval | None:
    """The interval on the axis of *root* of the region or child named *chunk*.

    Returns:
        The interval, or None when the tree holds no such region or child.

    Raises:
        UnknownLengthError: If it is a child timeline of unknown length.
    """
    for timeline, origin in _tree_origins(root):
        if timeline is not root and timeline.id == chunk:
            end = timeline._require_length("read it as a chunk of a FlowMap")
            start = timeline.origin
        elif chunk in timeline._regions:
            start, end = timeline._regions[chunk].start, timeline._regions[chunk].end
        else:
            continue
        on_root = (
            Coordinate(
                origin + express_scalar_as(bound, root.number_type).value,
                root.unit,
                number_type=root.number_type,
            )
            for bound in (start, end)
        )
        return Interval(start=next(on_root), end=next(on_root))
    return None


# endregion


# region Atomic sections and computed flows


class ComputedFlows(NamedTuple):
    """What computing a timeline's flows from its flow control wrote.

    Attributes:
        sections: The atomic sections, Regions of the timeline in the order
            of its axis.
        flow_maps: The FlowMaps written: ``default``, then ``single`` and
            ``printed`` where their chunks differ from ``default``'s.
        findings: What playing the flow control found: the findings of the
            ``default`` flow, in the order found.
    """

    sections: tuple[Region, ...]
    flow_maps: tuple[FlowMap, ...]
    findings: tuple[FlowFinding, ...] = ()


class StatedFlow(NamedTuple):
    """What writing the order of measures a source states wrote.

    Attributes:
        regions: The stretches of consecutive measures the order plays,
            Regions of the timeline in the order first played; a stretch
            played more than once is one Region.
        flow_map: The FlowMap listing those Regions in the order stated. It
            follows no convention and its provenance gives the method
            ``imported``.
        differs_at: Where the default flow of the timeline's flow control
            first plays something else than the stated order: the position
            along the played flow up to which the two agree, on the
            timeline's unit. None where they are the same flow.
    """

    regions: tuple[Region, ...]
    flow_map: FlowMap
    differs_at: Coordinate | None


# endregion


# region Virtual timelines


@dataclass(frozen=True)
class VirtualTimeline(_IdentifiedElement):
    """The statement that a map was applied to a timeline, giving a timeline.

    A virtual timeline exists only by applying the map: it has an ID, a
    unit, a number type and a length of its own, and nothing else of it is
    written. It holds no events, no children and no flow control. Its
    positions and those of the timeline it comes from convert through the
    map, in both directions (:meth:`unfold`, :meth:`fold`).

    Attributes:
        map: The ID of the map applied: a FlowMap, or a conversion map.
        timeline: The ID of the timeline the map is applied to, or of
            another virtual timeline.
        length: The end of the virtual timeline on its own axis, which
            states its unit and number type too.
        through: The ID of the group or conversion map that carries the map
            to ``timeline``; kept as read and never evaluated.
        name: Optional display name.
        id: The virtual timeline's ID; minted in the ID kind of its unit
            (``clt_{N}`` on quarters), as a timeline's is.

    Examples:
        >>> from timetoalign.timelines import ContinuousLogicalTimeline
        >>> score = ContinuousLogicalTimeline(length=24, uid="score")
        >>> a = score.create_region(0, 16, id="a")
        >>> b = score.create_region(16, 24, id="b")
        >>> played = score.create_virtual_timeline(score.create_flow_map([a, a, b]))
        >>> played.id, played.length
        ('clt_1', Coordinate(Fraction(40, 1), quarters))
        >>> played.unfold(4, score)
        [Coordinate(Fraction(4, 1), quarters), Coordinate(Fraction(20, 1), quarters)]
        >>> played.fold(20, score)
        Coordinate(Fraction(4, 1), quarters)
    """

    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "map",
        "timeline",
        "unit",
        "number_type",
        "length",
        "through",
        "name",
    )
    _manifest_path: ClassVar[tuple[str, ...]] = ("virtual_timelines",)

    map: str
    timeline: str
    length: Coordinate
    through: str | None = field(default=None, kw_only=True)
    name: str | None = field(default=None, kw_only=True)

    @classmethod
    def from_flow_map(
        cls,
        flow_map: FlowMap,
        root: Timeline,
        *,
        id: str | None = None,
        name: str | None = None,
    ) -> VirtualTimeline:
        """State that *flow_map* was applied to its root timeline.

        The length is the FlowMap's virtual length, the unit and number type
        are the root's.

        Args:
            flow_map: The FlowMap applied.
            root: The root timeline it names.
            id: The virtual timeline's ID; minted when None.
            name: Optional display name.

        Raises:
            ValueError: If the FlowMap cannot be read against *root*
                (:meth:`FlowMap.resolve`).
        """
        return cls(
            flow_map.id, root.id, flow_map.resolve(root).length, id=id, name=name
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VirtualTimeline:
        """Restore a virtual timeline from its object in ``virtual_timelines``.

        IDs are bare. ``length`` is a bare value read as ``number_type``
        decides; ``through`` and ``name`` may be left out. Keys the library
        does not know are kept and written back.

        Args:
            data: The virtual timeline object.

        Raises:
            ValueError: If the object is malformed or its length is
                represented otherwise than its number type selects.
        """
        required = ("id", "map", "timeline", "unit", "number_type", "length")
        require_object(data, "a virtual timeline", required)
        virtual_id = read_id(data["id"], "a virtual timeline")
        what = f"virtual timeline {quote_id(virtual_id)}"
        through = data.get("through")
        virtual = cls(
            read_id(data["map"], f"the map of {what}"),
            read_id(data["timeline"], f"the timeline of {what}"),
            axis_value_from_json(
                data["length"],
                key="length",
                owner=virtual_id,
                unit=TimeUnit(data["unit"]),
                number_type=NumberType[data["number_type"]],
            ),
            through=None if through is None else read_id(through, f"through of {what}"),
            name=read_name(data, what),
            id=virtual_id,
        )
        virtual._keep_unknown_keys(data)
        return virtual

    def __post_init__(self) -> None:
        """Mint the ID in the ID kind of the unit, or validate the one given."""
        if not isinstance(self.length, Coordinate) or self.length.value < 0:
            raise ValueError(
                f"the length of a virtual timeline is a coordinate from 0 on, got "
                f"{self.length!r}"
            )
        if self.id is None:
            object.__setattr__(self, "id", mint_id(self._id_prefix, ()))
            object.__setattr__(self, "_minted", True)
        super().__post_init__()
        for reference in (self.map, self.timeline, self.through):
            if reference is not None:
                validate_id(reference)

    @property
    def _id_prefix(self) -> str:
        """The ID kind a timeline of this unit is minted in."""
        return timeline_id_prefix(self.unit)

    @property
    def unit(self) -> TimeUnit:
        """The unit of the virtual timeline's axis."""
        return self.length.unit

    @property
    def number_type(self) -> NumberType:
        """The number type of the virtual timeline's axis."""
        return self.length.number_type

    def resolve(self, root: Timeline) -> ResolvedFlowMap:
        """Return the played order this virtual timeline is the axis of.

        Args:
            root: The root timeline holding the virtual timeline.

        Raises:
            ValueError: If the virtual timeline is not the FlowMap of
                *root* applied to *root*: it applies another map (a
                conversion map), is applied to another timeline, or goes
                ``through`` a group or conversion map. Such a statement is
                kept and written back, but its positions are not converted.
        """
        return self._resolved(root, ())

    def unfold(self, coordinate: CoordinateInput, root: Timeline) -> list[Coordinate]:
        """Return every position of this virtual timeline a root position is played at.

        See :meth:`ResolvedFlowMap.unfold`.

        Args:
            coordinate: A position on the root's axis.
            root: The root timeline holding the virtual timeline.

        Raises:
            ValueError: As :meth:`resolve` and :meth:`ResolvedFlowMap.unfold`.
        """
        return self.resolve(root).unfold(coordinate)

    def fold(self, coordinate: CoordinateInput, root: Timeline) -> Coordinate | None:
        """Return the root position played at a position of this virtual timeline.

        See :meth:`ResolvedFlowMap.fold`.

        Args:
            coordinate: A position on this virtual timeline's axis: a bare
                value, a Coordinate, or an IdCoordinate naming this virtual
                timeline.
            root: The root timeline holding the virtual timeline.

        Returns:
            The position on the root's axis, or None inside a gap.

        Raises:
            ValueError: As :meth:`resolve` and :meth:`ResolvedFlowMap.fold`,
                or if an IdCoordinate names another timeline.
        """
        if isinstance(coordinate, IdCoordinate) and coordinate.timeline_id != self.id:
            raise ValueError(
                f"{coordinate!r} is no position of virtual timeline {quote_id(self.id)}"
            )
        return self.resolve(root).fold(coordinate)

    def _resolved(self, root: Timeline, visiting: tuple[str, ...]) -> ResolvedFlowMap:
        """:meth:`resolve`, inside the FlowMaps *visiting* being resolved."""
        what = f"virtual timeline {quote_id(self.id)}"
        flow_map = root._flow_maps.get(self.map)
        if flow_map is None:
            raise ValueError(
                f"{what} applies {quote_id(self.map)}, which is no FlowMap of "
                f"timeline {quote_id(root.id)}: its positions are not converted. "
                f"FlowMaps of the timeline: {list(root._flow_maps)}"
            )
        if self.through is not None or self.timeline != flow_map.timeline:
            raise ValueError(
                f"{what} applies FlowMap {quote_id(flow_map.id)} to "
                f"{quote_id(self.timeline)}"
                + (
                    f" through {quote_id(self.through)}"
                    if self.through is not None
                    else ""
                )
                + f", not to the FlowMap's own timeline {quote_id(flow_map.timeline)}: "
                "carrying a FlowMap to another axis is not supported yet"
            )
        return _resolve(flow_map, root, visiting)

    def _with_references_renamed(self, renamed: Mapping[str, str]) -> VirtualTimeline:
        """Return the virtual timeline naming map, timeline and route by their new IDs."""
        changed = {
            key: renamed[value]
            for key in ("map", "timeline", "through")
            if (value := getattr(self, key)) in renamed
        }
        return replace(self, **changed) if changed else self

    def _dict_fields(self) -> dict[str, Any]:
        """Return the virtual timeline's keys; ``through`` is written only
        when stated."""
        fields: dict[str, Any] = {
            "id": self.id,
            "map": self.map,
            "timeline": self.timeline,
            "unit": str(self.unit),
            "number_type": str(self.number_type),
            "length": axis_value_to_json(self.length),
        }
        if self.through is not None:
            fields["through"] = self.through
        fields["name"] = self.name
        return fields

    def __repr__(self) -> str:
        name = "" if self.name is None else f", name={self.name!r}"
        return (
            f"VirtualTimeline({self.id!r}, map={self.map!r}, "
            f"timeline={self.timeline!r}, length={self.length}{name})"
        )


def place_on_roots(
    flow_maps: Sequence[FlowMap],
    virtual_timelines: Sequence[VirtualTimeline],
    timelines: Mapping[str, Timeline],
) -> None:
    """Hand the FlowMaps and virtual timelines a manifest lists to their roots.

    A FlowMap joins the root timeline it names. A virtual timeline joins the
    root of the tree it comes from: the one holding its ``timeline``, found
    virtual timeline by virtual timeline where that names another one.

    Args:
        flow_maps: The manifest's FlowMaps.
        virtual_timelines: The manifest's virtual timelines.
        timelines: Every timeline of the manifest, by ID.

    Raises:
        ValueError: If virtual timelines name one another in a circle, so
            that they come from no timeline.
    """
    for flow_map in flow_maps:
        timelines[flow_map.timeline]._tree_root()._flow_maps[flow_map.id] = flow_map
    comes_from = {virtual.id: virtual.timeline for virtual in virtual_timelines}
    for virtual in virtual_timelines:
        source, passed = virtual.timeline, {virtual.id}
        while source in comes_from:
            if source in passed:
                raise ValueError(
                    f"virtual timeline {quote_id(virtual.id)} comes from no "
                    f"timeline: the virtual timelines {sorted(passed)} name one "
                    "another in a circle"
                )
            passed.add(source)
            source = comes_from[source]
        timelines[source]._tree_root()._virtual_timelines[virtual.id] = virtual


# endregion
