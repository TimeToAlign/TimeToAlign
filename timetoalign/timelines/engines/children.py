"""Provide child timeline and slicing operations for timeline instances."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.core import (
    BoundingBox,
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    NumberType,
    Shape,
    TimeUnit,
)
from timetoalign.core.fields import _filled
from timetoalign.core.ids import (
    IdMinter,
    content_minter,
    minting_from,
    quote_id,
    timeline_id_prefix,
)
from timetoalign.core.time import (
    compare_number_column,
    coordinate_numeric_value,
    express_as,
    number_to_json,
    shift_coordinate,
    shift_number,
    subtract_coordinates,
)
from timetoalign.core.timestamp import ConversionMapsSpec, _format_coordinate_value
from timetoalign.maps import (
    ChainMap,
    ConversionMap,
    FloatingMeasureMap,
    LinearMap,
    MeasureBeatMap,
    MeasureNumberMap,
)
from timetoalign.storage.events import BOX_EVENT_TYPE
from timetoalign.storage.registry import SEGMENT_KIND

from ..manifest import spell
from .regions import _resolve_boundary_names

if TYPE_CHECKING:
    from ..base import Timeline
    from ..structure import MeasureMap
    from .host import TimelineHost as _Host
else:
    _Host = object

SEGMENT_EVENT_TYPE = SEGMENT_KIND
TraversalOrder = Literal["sorted", "depth_first", "breadth_first"]

_MEASURE_VIEWS = (FloatingMeasureMap, MeasureNumberMap, MeasureBeatMap)
"""The conversion maps reading a measure map that a child rebuilds over its slice."""


def placed_on(
    value: CoordinateValue,
    offsets: Sequence[CoordinateValue],
    *,
    as_float: bool,
) -> CoordinateValue:
    """*value*, a position on a descendant, placed on the axis of an ancestor.

    The offsets of the path are added one at a time from the innermost out,
    exactly where they are exact, as ``get_events`` shifts a descendant's rows
    onto the axis queried, so both place a position on the same number.

    Args:
        value: A position on the descendant's own axis.
        offsets: The offsets of the path from the ancestor down to the
            descendant, outermost first.
        as_float: Whether the position is compared as a key, with a float
            query: then its double is returned.

    Returns:
        The position on the ancestor's axis, its double where *as_float*.
    """
    for offset in reversed(offsets):
        value = shift_number(value, offset, subtract=False)
    return float(value) if as_float else value


class _MeasureSlice(NamedTuple):
    """The measure structure a child takes from its parent."""

    measure_map: MeasureMap
    to_quarters: LinearMap | None
    views: list[ConversionMap[Any]]

    def give_to(self, child: Timeline) -> None:
        """Attach the slice, then each view over it, to *child*."""
        child.add_measure_map(self.measure_map, to_quarters=self.to_quarters)
        for view in self.views:
            child.add_conversion_map(view)


class ChildrenMixin(_Host):
    """Provide child timeline and slicing operations for timeline instances."""

    def _spawn_class(self) -> type["Timeline"]:
        """The class used for children and slices of this timeline."""
        return type(self._timeline)

    @property
    def n_children(self) -> int:
        """Number of direct child timelines."""
        return len(self._children)

    def _is_plain_segment(self, child: "Timeline") -> bool:
        """Whether *child* is a plain segment here: no segment line itself.

        Segments are never locked: the segment line's lock governs them. A
        segment that is itself a segment line is locked like any child.
        """
        return (
            self._is_segment_line_instance() and not child._is_segment_line_instance()
        )

    def _kept_lock(self, child: "Timeline") -> bool:
        """The lock a child read or copied into this timeline keeps.

        Its own, except a plain segment's, which is lifted.
        """
        return child.is_locked and not self._is_plain_segment(child)

    def _refuse_segment_lock(self, child: "Timeline") -> None:
        """Refuse to lock *child* where it is a plain segment of this line.

        Raises:
            ValueError: If *child* is a plain segment here.
        """
        if self._is_plain_segment(child):
            raise ValueError(
                f"A segment is never locked: lock its segment line '{self._id}' instead."
            )

    def _child_extent(self, child_id: str) -> CoordinateValue | None:
        """Return how far a child extends on its own axis, or None if unbounded.

        A child of known length extends over its length; a child of unknown
        length is not bounded above.
        """
        length = self._children[child_id].length
        return None if length is None else length.value

    @property
    def _extent(self) -> CoordinateValue | None:
        """This timeline's extent on its own axis, or None if unbounded.

        The length where it is known; otherwise what the parent's structure
        says about this child (a segment ends at its successor's offset).
        """
        if self._length is not None:
            return self._length.value
        parent = self._parent
        return None if parent is None else parent._child_extent(self._id)

    @property
    def _holds_own_end(self) -> bool:
        """Whether this timeline's end is a position on it.

        It is, unless a sibling of nonzero extent starts exactly there: the
        shared boundary belongs to the sibling that starts at it
        (:meth:`_holds_position`). A sibling of zero length takes nothing
        away, since nothing of it lies beyond the boundary, and a timeline of
        zero length holds its end, which is also its start. A root holds its
        end; so does a timeline of unbounded extent, which has none.
        """
        parent = self._parent
        extent = self._extent
        if parent is None or extent is None or extent == 0:
            return True
        end = shift_number(
            parent._child_offsets[self._id].value, extent, subtract=False
        )
        return not any(
            child_id != self._id
            and offset.value == end
            and parent._children[child_id]._extent != 0
            for child_id, offset in parent._child_offsets.items()
        )

    def _holds_position(
        self,
        value: CoordinateValue,
        *,
        offsets: Sequence[CoordinateValue] = (),
        end_held: bool | None = None,
    ) -> bool:
        """Whether a position carried down from a parent lies on this timeline.

        The ONE containment rule for a position carried from a parent onto a
        child: the timeline's stamps (``_get_child_coordinate``), its tables
        (:meth:`_holds_column`, the same rule on a column), its
        :meth:`get_children_at` and a bundle's stamps across a tree all read
        it. A position carried further down reaches a descendant only if
        every timeline on the way holds it. The span is ``[0, extent)``,
        plus the end itself where no sibling starts there
        (:attr:`_holds_own_end`). One boundary position belongs to one of two
        adjoining siblings, the one starting at it, so a position is never
        counted twice across them; and the end of a child nothing follows,
        such as a section ending before its parent does, stays reachable on
        that child. A timeline of unbounded extent holds everything from 0
        on.

        A position written as a float is a key: it is compared with the
        doubles of this timeline's start and end, never subtracted from an
        exact boundary, since a float difference can put a key on a boundary
        just inside it. With *offsets* the key stays on the axis it was
        written on and the boundaries are placed there exactly first.

        Args:
            value: A position on this timeline's own axis, or, with
                *offsets*, on the axis of the ancestor they lead down from.
            offsets: The offsets of the path from that ancestor down to this
                timeline, outermost first; none for a position on this
                timeline's own axis.
            end_held: Whether the end is held; by default as
                :attr:`_holds_own_end` says. A position asked of this timeline
                itself holds its end whatever its siblings do, so the span is
                closed, ``[0, extent]``.

        Returns:
            Whether the position lies on this timeline.
        """
        if value < placed_on(0, offsets, as_float=isinstance(value, float)):
            return False
        extent = self._extent
        if extent is None:
            return True
        end = placed_on(extent, offsets, as_float=isinstance(value, float))
        if value < end:
            return True
        return value == end and (self._holds_own_end if end_held is None else end_held)

    def _holds_column(
        self,
        positions: pa.StructArray,
        number_type: NumberType,
        start: CoordinateValue = 0,
        *,
        end_held: bool | None = None,
    ) -> pa.Array:
        """Apply :meth:`_holds_position` to a column of positions.

        Each row is decided as the scalar rule decides the position it
        holds, exactly on an ``int`` or ``fraction`` column
        (:func:`~timetoalign.core.time.compare_number_column`), never through
        a float that could put a position just inside a boundary on the
        wrong side of it.

        Args:
            positions: Coordinate structs, null where no position arrives.
            number_type: How *positions* declare their numbers.
            start: Where this timeline's 0 lies on the axis of *positions*:
                0 for positions already on this timeline, its offset for
                positions on its parent.
            end_held: Whether the end is held; by default as
                :attr:`_holds_own_end` says. A timeline's own column holds
                its end whatever its siblings do, since the positions were
                asked of it.

        Returns:
            A boolean array, false wherever the position is not held or the
            row is null.
        """
        within = compare_number_column(positions, start, "greater_equal", number_type)
        extent = self._extent
        if extent is not None:
            end = shift_number(start, extent, subtract=False)
            below = compare_number_column(positions, end, "less", number_type)
            if self._holds_own_end if end_held is None else end_held:
                below = pc.or_(
                    below, compare_number_column(positions, end, "equal", number_type)
                )
            within = pc.and_(within, below)
        return _filled(within, pa.scalar(False))

    def _set_length(self, length: Coordinate | None) -> None:
        """Set this timeline's length, which its parent must contain.

        The parent checks the new end (:meth:`_check_child_length`) before
        anything changes; it never grows because a child grew. A length that
        becomes known, or shorter, must leave every flow-control element and
        every place a Jump goes on at within it. Then the length is set and
        the parent's ``Segment`` row follows it.

        Raises:
            ValueError: If the parent cannot contain the new end, or a
                flow-control element or a destination's place would lie
                beyond it.
        """
        parent = self._parent
        if parent is not None:
            parent._check_child_length(self._id, length)
        before = self._length
        if length is not None and (before is None or length.value < before.value):
            self._flow_control._check_length(length)
        self._length = length
        if parent is not None:
            parent._refresh_segment_row(self._id)

    def _check_child_length(self, child_id: str, length: Coordinate | None) -> None:
        """Check that this timeline contains a child's new length.

        Nothing is checked where either end is unknown. This timeline's end
        is its length, or, while that is unknown, what its own parent says
        about it (a segment ends at its successor's offset).

        Raises:
            ValueError: If the child's new end lies beyond this timeline's end.
        """
        bound = self._extent
        if length is None or bound is None:
            return
        end = self._child_offsets[child_id].value + length.value
        if end > bound:
            raise ValueError(
                f"Timeline '{child_id}' cannot end at {_format_coordinate_value(end)} "
                f"on its parent '{self._id}', which ends at "
                f"{_format_coordinate_value(bound)}"
            )

    def _segment_row(self, child_id: str) -> dict[str, Any]:
        """The ``Segment`` row recording a child: its offset, end and duration.

        End and duration are ``None`` while the child's length is unknown.
        """
        offset = self._child_offsets[child_id].value
        length = self._children[child_id].length
        return {
            "id": child_id,
            "name": child_id,
            "temporal_type": "interval",
            "event_type": SEGMENT_EVENT_TYPE,
            "start": offset,
            "end": None if length is None else offset + length.value,
            "duration": None if length is None else length.value,
        }

    def _refresh_segment_row(self, child_id: str) -> None:
        """Rewrite a child's ``Segment`` row in place from its current length."""
        events = self._events
        table = events.table
        is_row = _filled(
            pc.and_kleene(
                pc.equal(table.column("event_type"), pa.scalar(SEGMENT_EVENT_TYPE)),
                pc.equal(table.column("id"), pa.scalar(child_id)),
            ),
            pa.scalar(False),
        )
        position = pc.indices_nonzero(is_row)[0].as_py()
        row = type(events).from_dicts(
            [self._segment_row(child_id)], events.unit, events.number_type
        )
        combined = events.concat(row)
        order = list(range(len(events)))
        order[position] = len(events)
        self._replace_events(events._derived(combined.table.take(order)))

    def validate_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
    ) -> None:
        """Validate that a timeline can be added as a child.

        A timeline can accommodate events and other timelines, called
        Children, as long as they use the same measuring unit.

        For cross-domain relationships (e.g., physical to logical), use
        TimelineGroup instead of parent-child nesting.

        Args:
            child: The timeline to validate.
            offset: The proposed start coordinate.

        A child has its parent's number type too, and one parent: its
        events lie on its root's axis, which a second parent would change.

        Raises:
            TypeError: If child is not a Timeline.
            ValueError: If units or number types don't match, the timeline
                already is a child of this timeline or of another one, or the
                offset is negative, which is not supported yet.
        """
        if not isinstance(child, Timeline):
            raise TypeError(f"Child must be a Timeline, got {type(child).__name__}")

        if child.unit != self._unit:
            raise ValueError(
                f"Child unit '{child.unit}' does not match "
                f"parent unit '{self._unit}'. "
                f"Per TTA specification, Children must share the parent's unit. "
                f"For cross-domain relationships, use TimelineGroup instead."
            )

        if child.number_type is not self._number_type:
            raise ValueError(
                f"Child {quote_id(child.id)} has the number type "
                f"'{child.number_type.name}', its parent {quote_id(self._id)} has "
                f"'{self._number_type.name}': a child has its parent's unit and "
                "number type. Events of another number type reach a timeline only "
                "as Projections through a C-Map (add_projections)."
            )

        if any(existing is child for existing in self._children.values()):
            raise ValueError(f"Child '{child.id}' is already a child of this timeline")

        if child._parent is not None:
            raise ValueError(
                f"Timeline {quote_id(child.id)} is a child of "
                f"{quote_id(child._parent.id)} already: a timeline has one parent "
                "(it may be a member of many groups)"
            )

        offset_coord = self.get_coordinate_at(offset, format="coordinate")
        if offset_coord.value < 0:
            raise ValueError(
                f"A negative child offset is not supported yet: child '{child.id}' "
                f"has offset {spell(number_to_json(offset_coord.value))}."
            )

    def _convert_child_to_parent_unit(
        self,
        child: Timeline,
        use_conversion_map: ConversionMapsSpec,
        seed: int | None = None,
    ) -> Timeline:
        """Convert a child timeline to this timeline's unit via a C-Map.

        The conversion map is resolved from the parent's attached C-Maps.
        We find a map whose ``target_unit`` matches the child's unit, invert
        it, and use it to derive a copy of the child in the parent's unit.

        The derived copy is named ``"{child.name} [{parent.unit}]"``, gets a
        minted id in the parent's unit, and retains all events with converted
        coordinates.

        Args:
            child: The child timeline to convert.
            use_conversion_map: Flexible specification for which C-Map to use.
                Accepts the same formats as the ``conversion_maps`` parameter
                in timestamp functions:
                - ``True``: Auto-select a C-Map whose target unit matches
                  the child's unit.
                - ``str``: Look up by C-Map ID or target unit name.
                - ``TimeUnit``: Find by target unit.
                - ``ConversionMap``: Use directly (must map parent unit to
                  child unit so that its inverse converts child to parent).

        Returns:
            A new Timeline in the parent's unit with converted events.

        Raises:
            ValueError: If no suitable C-Map can be found or inverted.
        """
        # Resolve the conversion map on the parent
        if use_conversion_map is True:
            # Auto-select: find a parent C-Map whose target_unit matches child's unit
            cmap = self.get_conversion_map(child.unit)
            if cmap is None:
                raise ValueError(
                    f"Cannot auto-select conversion map: no C-Map from "
                    f"'{self._unit}' to '{child.unit}' attached to parent "
                    f"'{self._id}'. Attach a C-Map first or specify one "
                    f"explicitly via use_conversion_map."
                )
        elif isinstance(use_conversion_map, ConversionMap):
            cmap = use_conversion_map
        else:
            # Use _resolve_conversion_maps for str / TimeUnit / list
            resolved = self._resolve_conversion_maps(use_conversion_map)
            # Find the one whose target_unit matches child's unit
            matching = [m for m in resolved if m.target_unit == child.unit]
            if not matching:
                raise ValueError(
                    f"None of the resolved conversion maps target the child's "
                    f"unit '{child.unit}'. Resolved maps target: "
                    f"{[m.target_unit for m in resolved]}"
                )
            cmap = matching[0]

        # Invert: we need child_unit -> parent_unit
        if not cmap.is_invertible:
            raise ValueError(
                f"C-Map '{cmap.id}' ({cmap.source_unit} -> {cmap.target_unit}) "
                f"is not invertible. Cannot convert child from "
                f"'{child.unit}' to '{self._unit}'."
            )
        # The name links back to the original; the id is minted, because a
        # label is no identity: from the seed, else from the source's id and
        # the unit it is converted to. With a seed, every id the conversion
        # mints comes from it, the derived child's first.
        prefix = timeline_id_prefix(self._unit)
        with minting_from(seed):
            derived_id = content_minter(prefix, child.id, self._unit.value).mint(prefix)
            inverse_cmap = cmap.inverse()

            # Temporarily add the inverse map to the child so derive() can use it
            child_had_map = bool(child._get_unit_maps(self._unit))
            if not child_had_map:
                child.add_conversion_map(inverse_cmap)

            try:
                derived = child.derive(self._unit, copy_events=True, uid=derived_id)
            finally:
                # Clean up: remove the temporarily added map if we added it.
                # Removal is by identity, so a map the child already carried
                # for the same unit survives.
                if not child_had_map and inverse_cmap.id in child._conversion_maps:
                    del child._conversion_maps[inverse_cmap.id]
                    registered = child._unit_maps.get(self._unit)
                    if registered is not None:
                        registered[:] = [
                            cmap for cmap in registered if cmap is not inverse_cmap
                        ]
                        if not registered:
                            del child._unit_maps[self._unit]

        derived._name = f"{child.name or child.id} [{self._unit}]"

        self._logger.debug(
            f"Converted child '{child.id}' ({child.unit}) to "
            f"'{derived.id}' ({self._unit}) via inverse of '{cmap.id}'"
        )

        return derived

    def add_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
        allow_expansion: bool = False,
        use_conversion_map: ConversionMapsSpec = None,
        *,
        locked: bool | None = None,
        seed: int | None = None,
    ) -> None:
        """Embed a child timeline at the specified offset.

        The child timeline is locked once added unless ``locked=False``, which
        leaves its own state as it is. An unlocked child may change its
        length, but only within this timeline's known end: this timeline
        never grows because a child grew. A segment of a segment line is
        never locked: the line's lock governs it, and the segment's own lock
        is lifted when it joins. A segment line is locked like any child,
        whether it becomes a child or a segment; unlocking it is not
        supported yet, so ``locked=False`` is refused for a segment line that
        is unlocked, and accepted (asking for nothing) for one already locked.
        Parent-child coordinate conversion uses exact offset arithmetic.

        The child's whole subtree joins this timeline's manifest, in which
        every id is unique across all element kinds: an element of the
        subtree whose id, given or minted, is already held here makes the
        call raise without changing anything. Nothing is renamed.

        A timeline can accommodate events and other timelines, called
        Children, as long as they use the same measuring unit.

        When the child uses a different unit than the parent, set
        ``use_conversion_map`` to automatically convert the child to the
        parent's unit via a C-Map. The parent must have a C-Map whose
        ``target_unit`` matches the child's unit, so that inverting it yields
        the ``child_unit -> parent_unit`` conversion. The child's events are
        copied with converted coordinates; the original child is NOT modified.

        The converted child is named ``"{child.name} [{parent.unit}]"`` and
        receives a minted id: from ``IdMinter(seed)`` with *seed*, else from
        the content minter of the source's id and the parent's unit.

        Args:
            child: The timeline to embed.
            offset: The start coordinate on this timeline, in the **parent's**
                unit. When ``use_conversion_map`` is set, the offset must
                already be expressed in the parent's unit (e.g. samples).
            allow_expansion: If True, expand this timeline if needed.
            use_conversion_map: Conversion map specification for unit
                conversion. Accepts the same formats as the
                ``conversion_maps`` parameter in timestamp functions:
                - ``None`` (default): No conversion; units must match.
                - ``True``: Auto-select a parent C-Map whose target unit
                  matches the child's unit.
                - ``str``: Look up by C-Map ID or target unit name.
                - ``TimeUnit``: Find by target unit.
                - ``ConversionMap``: Use directly.
            locked: Whether to lock the child. ``None`` (the default) locks
                every child but a plain segment, which is never locked.
                ``False`` keeps the child's own state: an unlocked child stays
                unlocked, a locked one stays locked; it is refused for an
                unlocked segment line.
            seed: The seed of the minter of a converted child's id; not
                stored.

        Raises:
            TypeError: If child is not a Timeline.
            ValueError: If units don't match (and no conversion map given),
                the offset is negative (not supported yet), the child is
                already a child of this timeline, ``locked=True`` asks to lock
                a plain segment, ``locked=False`` asks to leave a segment line
                unlocked, an id of the child's subtree is already held in
                this manifest, a Jump of the child's subtree that is
                not rejected lies at a position of the root's axis where one
                of this timeline's tree lies (one Jump lies at a position of
                a root's axis), or this timeline is locked and the child
                would end beyond its length without ``allow_expansion``. An
                unlocked timeline expands to fit the child instead, provided
                its own parent contains its new end. A child of unknown
                length has no end, so only its offset must fit; a parent of
                unknown length has no end to exceed, unless it is a segment,
                which ends at its successor's offset.
        """
        # Convert child's unit if a conversion map is specified
        if use_conversion_map is not None and child.unit != self._unit:
            child = self._convert_child_to_parent_unit(child, use_conversion_map, seed)

        if locked:
            self._refuse_segment_lock(child)
        elif (
            locked is False
            and child._is_segment_line_instance()
            and not child.is_locked
        ):
            raise ValueError(
                "Unlocking a segment line that is a child, a segment or a group "
                "member is not supported yet."
            )
        self._embed_child(child, offset, allow_expansion, locked=locked is not False)

    def _embed_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
        allow_expansion: bool = False,
        *,
        locked: bool,
    ) -> None:
        """Embed *child* as :meth:`add_child` does, taking the lock as given.

        ``locked=True`` locks the child, ``False`` keeps its own state; a
        plain segment is always left unlocked. Reading and copying embed
        through here with the lock the child keeps (:meth:`_kept_lock`), so
        a segment line keeps whatever lock the file or the source states.

        Raises:
            TypeError: As :meth:`add_child`.
            ValueError: As :meth:`add_child`, but for the lock requests.
        """
        plain_segment = self._is_plain_segment(child)

        self.validate_child(child, offset)

        offset_coord = self.get_coordinate_at(offset, format="coordinate")
        # A child of unknown length has no end: only its offset must fit.
        child_end = (
            None if child.length is None else offset_coord.value + child.length.value
        )
        # The whole subtree joins this manifest: an id held here raises
        # before anything changes; so does an event id this family holds.
        self._refuse_held(child._subtree_entries())
        # The child roots its own family (it has no parent), whose rows join
        # this one's, shifted onto this family's root.
        family = self._registry._family
        joining = child._registry._family
        family.refuse_joining(joining)
        # The child's Segment row, named by the child's ID, is an event of
        # this timeline: its ID is refused here too, where either family
        # holds it.
        segment_id = pa.array([child.id], pa.string())
        family.refuse_held_ids(self._id, segment_id)
        joining.refuse_held_ids(self._id, segment_id)
        # One Jump per position of the root's axis, over the root's tree.
        _, start = self._root_start()
        self._refuse_repeated_jumps(
            child._jump_positions(start + Fraction(offset_coord.value))
        )

        # Ensure capacity
        self._ensure_capacity(
            offset_coord.value if child_end is None else child_end, allow_expansion
        )

        # Store child reference; the child's family's rows join this
        # family's, shifted once onto its root.
        self._children[child.id] = child
        self._child_offsets[child.id] = offset_coord
        child._parent = self._timeline
        family.attach(joining, self._id, offset_coord.value)
        for timeline in (child, *(t for _, t in child._iter_descendants())):
            timeline._registry = self._registry

        if plain_segment:
            child._locked = False
        elif locked:
            child._locked = True

        # Its ID was refused above where the tree holds it.
        self._append_events(
            self._event_data_of([self._segment_row(child.id)]), checked=True
        )

        self._logger.debug(
            f"Added child '{child.id}' at offset "
            f"{_format_coordinate_value(float(offset_coord.value))}"
        )

    def create_child(
        self,
        length: CoordinateSpec | None,
        offset: CoordinateSpec = 0,
        uid: str | None = None,
        name: str | None = None,
        allow_expansion: bool = False,
        child_class: type["Timeline"] | None = None,
        *,
        measures: bool = False,
        seed: int | None = None,
    ) -> "Timeline":
        """Create and embed a child selected by this timeline's spawn hook.

        By default, this is equivalent to:

            child_type = parent._spawn_class()
            child = child_type(length=length, unit=parent.unit, uid=uid, name=name)
            parent.add_child(child, offset=offset)

        ``child_class=Timeline`` deliberately selects the experimental base class.

        With *measures*, the child holds the slice of this timeline's measure
        map over its span (:meth:`~timetoalign.timelines.MeasureMap.get_slice`),
        read through the same ``LinearMap`` to quarters on an axis in whole
        notes or ticks. The slice is computed before the child is built, so a
        refusal leaves this timeline unchanged.

        Args:
            length: Child length in the parent's unit, or ``None`` when the
                child's length is unknown.
            offset: Child start coordinate. Defaults to zero.
            uid: Unique identifier for the child. Auto-generated if None.
            name: Human-readable name for the child.
            allow_expansion: If True, expand the parent when needed.
            child_class: Explicit class override for the child.
            measures: Give the child the slice of this timeline's measure
                map over its span.
            seed: The seed of the minter of the child's id when *uid* is
                None; not stored.

        Returns:
            The newly created and embedded child Timeline.

        Raises:
            ValueError: If coordinates are invalid or the child exceeds bounds;
                with *measures*, if this timeline has no measure map, the
                child's length is unknown, or the measure map refuses the
                slice.

        Examples:
            >>> from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap
            >>> parent = ContinuousLogicalTimeline(length=8)
            >>> child = parent.create_child(length=4)
            >>> type(child) is ContinuousLogicalTimeline
            True
            >>> bars = MeasureMap.from_rows(
            ...     [{"count": 1, "time_signature": "2/4"}, {"count": 4}]
            ... )
            >>> parent = parent.add_measure_map(bars)
            >>> half = parent.create_child(4, 4, measures=True)
            >>> [measure.id for measure in half.measure_map]
            ['3', '4']
        """
        if length is not None:
            length = self._resolve_axis_value(length)
        measured = self._measure_slice(offset, length) if measures else None

        child = (child_class or self._spawn_class())(
            length=length,
            unit=self._unit,
            number_type=self._number_type,
            uid=uid,
            name=name,
            seed=seed,
        )
        if measured is not None:
            measured.give_to(child)
        self.add_child(child, offset=offset, allow_expansion=allow_expansion)
        return child

    def create_calibrated_child(
        self,
        length: CoordinateSpec | None,
        offset: CoordinateSpec,
        calibration: Sequence[tuple[CoordinateSpec, CoordinateSpec]],
        *,
        value_unit: TimeUnit | str | None = None,
        law: type[ConversionMap[Any]] = LinearMap,
        uid: str | None = None,
        name: str | None = None,
        seed: int | None = None,
    ) -> "Timeline":
        """Create a child over a calibrated segment, with the map it reads.

        A segment of this timeline is calibrated by pairs of a position and
        the value read there in another unit, such as the labelled ticks of
        a drawn axis. The child starts at *offset*, and *law* builds the
        conversion map from the child's unit to the values' unit through the
        pairs, reading each position from *offset* in this timeline's own
        arithmetic (see :meth:`~timetoalign.maps.ConversionMap.from_pairs`).
        The pairs are checked against the law on the positions as given,
        where their rounding happened. The map is built before the child, so
        a refusal leaves this timeline unchanged.

        Args:
            length: Child length in this timeline's unit, or ``None`` when
                the child's length is unknown.
            offset: Child start coordinate.
            calibration: ``(position, value)`` pairs, in any order: positions
                in this timeline's coordinates, values as coordinates
                carrying the one unit they share or as numbers in
                *value_unit*.
            value_unit: The unit of the values. Needed when a value is a
                number; a coordinate value must be in it.
            law: The map class whose ``from_pairs`` reads the pairs.
            uid: Unique identifier for the child. Auto-generated if None.
            name: Human-readable name for the child.
            seed: The seed of one minter of every id the call mints: the
                map's, then the child's when *uid* is None
                (:func:`~timetoalign.core.ids.minting_from`); not stored.

        Returns:
            The newly created and embedded child, carrying the map.

        Raises:
            TypeError: If a value is a number and no *value_unit* is given.
            ValueError: If a position lies outside the child, or *law*
                refuses the pairs (fewer than two, a repeated position,
                values in two units, a pair off the law), or the child does
                not fit as in :meth:`create_child`.

        Examples:
            >>> from timetoalign.core import Coordinate
            >>> from timetoalign.timelines import ContinuousGraphicalTimeline
            >>> parent = ContinuousGraphicalTimeline(length=100, unit="points")
            >>> child = parent.create_calibrated_child(
            ...     80, 10, [(10, Coordinate(0, "seconds")), (90, Coordinate(4, "seconds"))]
            ... )
            >>> child.get_conversion_map("seconds").scalar
            0.05
            >>> child = parent.create_calibrated_child(
            ...     80, 10, [(10, 0), (50, 2), (90, 4)], value_unit="seconds"
            ... )
            >>> child.get_conversion_map("seconds").scalar
            0.05
        """
        start = self._resolve_axis_value(offset)
        end = None if length is None else start + self._resolve_axis_value(length)
        pairs = []
        for index, (position, value) in enumerate(calibration):
            at = self._resolve_axis_value(position)
            if at < start or (end is not None and at > end):
                raise ValueError(
                    f"Calibration pair {index} lies at {at}, outside the child "
                    f"from {start} to {'its unknown end' if end is None else end}."
                )
            pairs.append((at, value))
        with minting_from(seed):
            cmap = law.from_pairs(
                pairs, source_unit=self._unit, value_unit=value_unit, origin=start
            )
            child = self.create_child(length, offset, uid=uid, name=name)
        child.add_conversion_map(cmap)
        return child

    def create_children_from_boundaries(
        self,
        boundaries: Sequence[CoordinateSpec],
        *,
        names: Sequence[str] | None = None,
        name_format: str = "{prefix}_{n}",
        prefix: str = "section",
        allow_expansion: bool = False,
        measures: bool = False,
        seed: int | None = None,
    ) -> list["Timeline"]:
        """Create children spanning consecutive boundary coordinates.

        Source events are not copied into the children.

        Args:
            boundaries: k+1 monotonically increasing coordinates.
            names: Explicit names for the k children.
            name_format: Format with {prefix}, {i}, and {n} placeholders.
            prefix: Prefix for auto-generated names.
            allow_expansion: If True, expand the parent when needed.
            measures: Give each child the slice of this timeline's measure
                map over its span, as :meth:`create_child` does.
            seed: The seed of one minter of every id the call mints (the
                measure maps and views of each child, with *measures*), in
                boundary order (:func:`~timetoalign.core.ids.minting_from`).
                The children's ids are their names. Not stored.

        Returns:
            Child timelines in boundary order.

        Raises:
            ValueError: If the boundaries or number of names are invalid, or,
                with *measures*, as :meth:`create_child` raises.

        Examples:
            >>> from timetoalign.timelines import ContinuousLogicalTimeline, MeasureMap
            >>> score = ContinuousLogicalTimeline(length=8)
            >>> score = score.add_measure_map(
            ...     MeasureMap.from_rows([{"count": 1, "time_signature": "2/4"}, {"count": 4}])
            ... )
            >>> halves = score.create_children_from_boundaries([0, 4, 8], measures=True)
            >>> [[measure.id for measure in half.measure_map] for half in halves]
            [['1', '2'], ['3', '4']]
        """
        coords, child_names = _resolve_boundary_names(
            boundaries,
            names,
            name_format,
            prefix,
            self._resolve_axis_value,
        )
        with minting_from(seed):
            return [
                self.create_child(
                    end - start, start, name, name, allow_expansion, measures=measures
                )
                for start, end, name in zip(
                    coords[:-1], coords[1:], child_names, strict=True
                )
            ]

    def append_child(
        self,
        child: "Timeline",
        *,
        name: str | None = None,
        uid: str | None = None,
    ) -> None:
        """Append a child timeline at the current end of this timeline.

        The child is placed at ``offset = self.length`` (a segment line
        appends after its last segment), so successive calls stack children
        end-to-end and expand this timeline. Because the child
        is a fresh, unlocked timeline, its identity may be set here: *uid*
        becomes the child's ID (the key under which it is stored and
        retrieved) and *name* its human-readable name — both applied before
        :meth:`add_child` embeds it (locking it unless it is a plain segment
        of a segment line). A child :meth:`add_child` refuses keeps the id
        and name it had.

        Args:
            child: The timeline to append. Must be fresh (parentless) and
                share this timeline's unit.
            name: Human-readable name for the child. Applied when given.
            uid: Identifier for the child; also the key for
                :meth:`get_child`. Applied when given.

        Raises:
            TypeError: If child is not a Timeline.
            ValueError: If units don't match.
            RuntimeError: If this timeline is locked.
            UnknownLengthError: If nothing says where to append: this
                timeline's length is unknown, or, on a segment line, the last
                segment's.

        Examples:
            >>> parent = ContinuousLogicalTimeline(length=0)
            >>> parent.append_child(
            ...     ContinuousLogicalTimeline(length=8), uid="A", name="A"
            ... )
            >>> parent.append_child(
            ...     ContinuousLogicalTimeline(length=8), uid="B", name="B"
            ... )
            >>> parent.list_children()
            ['A', 'B']
            >>> float(parent.get_child_offset("B").value)
            8.0
        """
        self._check_not_locked("append child")
        end = self._append_offset()
        given_id, given_name = child.id, child._name
        if uid is not None:
            child._set_id(uid)
        if name is not None:
            child._name = name
        try:
            self.add_child(child, end, allow_expansion=True)
        except Exception:
            # A refused child is left as it was given.
            child._set_id(given_id)
            child._name = given_name
            raise

    def _append_offset(self) -> Coordinate:
        """The offset :meth:`append_child` places a child at: this timeline's end.

        Raises:
            UnknownLengthError: If this timeline's length is unknown.
        """
        return self._require_length("append a child")

    def _rename_child(self, old_id: str, new_id: str) -> None:
        """Re-key a child and its segment row from *old_id* to *new_id*."""
        self._children = {
            (new_id if key == old_id else key): child
            for key, child in self._children.items()
        }
        self._child_offsets = {
            (new_id if key == old_id else key): offset
            for key, offset in self._child_offsets.items()
        }
        self._replace_events(self._events._with_segment_renamed(old_id, new_id))

    def get_child(self, child_id: str) -> Timeline:
        """Retrieve a child timeline by ID.

        Args:
            child_id: The ID of the child to retrieve.

        Returns:
            The child Timeline.

        Raises:
            KeyError: If no child with that ID exists.
        """
        if child_id not in self._children:
            raise KeyError(
                f"No child with ID '{child_id}'. "
                f"Available children: {list(self._children.keys())}"
            )
        return self._children[child_id]

    def get_child_offset(self, child_id: str) -> Coordinate:
        """Get the offset of a child timeline.

        Args:
            child_id: The ID of the child.

        Returns:
            The offset Coordinate.

        Raises:
            KeyError: If no child with that ID exists.
        """
        if child_id not in self._child_offsets:
            raise KeyError(
                f"No child with ID '{child_id}'. "
                f"Available children: {list(self._child_offsets.keys())}"
            )
        return self._child_offsets[child_id]

    def iter_children(
        self,
        order: TraversalOrder = "sorted",
        recursion_limit: int | None = None,
        include_self: bool = False,
    ) -> Iterator[tuple[Coordinate, Timeline]]:
        """Iterate over child timelines.

        Args:
            order: Traversal order - "sorted" (by offset), "depth_first",
                   or "breadth_first".
            recursion_limit: Maximum recursion depth. None for unlimited.
            include_self: If True, yield this timeline first.

        Yields:
            Tuples of (offset_coordinate, child_timeline).
        """
        if include_self:
            yield self._make_coordinate(0), self._timeline

        if recursion_limit is not None and recursion_limit <= 0:
            return

        next_limit = None if recursion_limit is None else recursion_limit - 1

        if order == "sorted":
            # Sort by offset
            sorted_children = sorted(
                self._children.items(),
                key=lambda x: self._child_offsets[x[0]].value,
            )
            for child_id, child in sorted_children:
                offset = self._child_offsets[child_id]
                yield offset, child
                # Recurse
                for sub_offset, sub_child in child.iter_children(
                    order=order, recursion_limit=next_limit, include_self=False
                ):
                    # Adjust offset relative to parent
                    combined_offset = self._make_coordinate(
                        offset.value + sub_offset.value
                    )
                    yield combined_offset, sub_child

        elif order == "breadth_first":
            # Yield all direct children first
            direct_children = []
            for child_id, child in self._children.items():
                offset = self._child_offsets[child_id]
                direct_children.append((offset, child))
                yield offset, child

            # Then recurse
            for offset, child in direct_children:
                for sub_offset, sub_child in child.iter_children(
                    order=order, recursion_limit=next_limit, include_self=False
                ):
                    combined_offset = self._make_coordinate(
                        offset.value + sub_offset.value
                    )
                    yield combined_offset, sub_child

        elif order == "depth_first":
            # Yield child, then its descendants, then next child
            for child_id, child in self._children.items():
                offset = self._child_offsets[child_id]
                yield offset, child
                for sub_offset, sub_child in child.iter_children(
                    order=order, recursion_limit=next_limit, include_self=False
                ):
                    combined_offset = self._make_coordinate(
                        offset.value + sub_offset.value
                    )
                    yield combined_offset, sub_child

    def _get_max_content_coordinate(self) -> CoordinateValue:
        """Get the maximum coordinate that this timeline's length must cover.

        Events count with their ends, children with their ends; a child of
        unknown length counts with its offset only, since its end is not known.

        Returns:
            The maximum coordinate value.
        """
        max_coord: CoordinateValue = 0

        # Check events
        coord_range = self._events.coordinate_range()
        if coord_range:
            max_coord = max(max_coord, coord_range[1])

        # Check children
        for child_id, child in self._children.items():
            offset = self._child_offsets[child_id].value
            extent = child._extent
            child_end = offset if extent is None else offset + extent
            max_coord = max(max_coord, child_end)

        return max_coord

    def create_child_from_region(
        self,
        region_id: str,
        *,
        copy_events: bool = True,
        uid: str | None = None,
        measures: bool = False,
        seed: int | None = None,
    ) -> "Timeline":
        """Create a child timeline from a region (partitioning).

        The child's length = region duration, offset = region start.
        The child's class matches the parent's concrete class, and its name
        is the region's name. If copy_events, the events starting in
        [start, end) are copied whole with adjusted coordinates (the child
        grows to hold one running past the region's end). Where this
        timeline holds Boxes, the child states the part of its area of an
        image (``meta["box"]``, ``meta["image"]``) over the region, as
        :meth:`get_slice` states it, so a Box keeps its shape and its start
        moves with the child's origin.

        A locked timeline gives a child from a region as an unlocked one
        does: the child lies inside the fixed length, which it leaves as it
        is. A region reaching past the end of a locked timeline gives no
        child, since the child would end past that end.

        Args:
            region_id: Id of an existing region.
            copy_events: Copy events within the region to the child.
            uid: Explicit child ID. When None, one is minted.
            measures: Give the child the slice of this timeline's measure
                map over the region, as :meth:`create_child` does.
            seed: The seed of the minter of the child's id when *uid* is
                None; not stored.

        Returns:
            The newly created and attached child timeline.

        Raises:
            KeyError: If region_id not found.
            ValueError: If this timeline is locked and the child would end
                past its length; with *measures*, if this timeline has no
                measure map or the measure map refuses the slice; if this timeline holds
                Boxes and its ``meta["box"]`` is no shape; or if it states
                no ``meta["box"]`` and a Box would move that a descendant
                holds, or one of its own while the region does not start
                at 0.

        Examples:
            >>> movements = tl.create_regions_by_splitting("breaks", prefix="movement")
            >>> mov4 = tl.create_child_from_region(movements[3].id)
            >>> mov4.name
            'movement_4'
        """
        region = self._regions.get(region_id)
        if region is None:
            raise KeyError(
                f"Region '{region_id}' not found on timeline '{self._id}'. "
                f"Available regions: {list(self._regions.keys())}"
            )

        measured = (
            self._measure_slice(region.start, region.duration.value)
            if measures
            else None
        )

        child = self._spawn_class()(
            length=region.duration.value,
            unit=self._unit,
            number_type=self._number_type,
            uid=uid,
            name=region.name,
            seed=seed,
        )
        if measured is not None:
            measured.give_to(child)

        self._copy_events_to_cut(
            child,
            region.start.value,
            region.end.value,
            copy_events=copy_events,
            in_this_tree=True,
        )

        self.add_child(child, offset=region.start)
        return child

    def _measure_slice(self, offset: CoordinateSpec, length: Any) -> _MeasureSlice:
        """Return the measure structure a child would hold.

        The child spans *length* from *offset*, both on this timeline's
        axis. On a quarters axis the span is the slice's range as it is; on
        an axis read through a ``LinearMap`` to quarters, the span goes
        through that map, and the child reads its slice through the same map
        from its own origin (:meth:`LinearMap.with_offset`), in the number
        type that map was given. Every other view this timeline holds of its
        measure map (a :class:`~timetoalign.maps.FloatingMeasureMap`,
        :class:`~timetoalign.maps.MeasureNumberMap` or
        :class:`~timetoalign.maps.MeasureBeatMap` reading a measure map
        equal to it) is built again over the slice, by the same call: the
        child affords the measure views its parent affords, each reading
        the child's own measures.

        Returns:
            The slice, the ``LinearMap`` the child reads it through (None on
            a quarters axis) and the views over the slice.

        Raises:
            ValueError: If this timeline has no measure map, *length* is
                None, or the measure map or a view refuses the slice.
            TypeError: If the measure map is read through another map than
                a ``LinearMap`` to quarters.
        """
        record = self._measure_record_map
        measure_map = self.measure_map
        if record is None or measure_map is None:
            raise ValueError(
                f"Timeline {self._id!r} has no measure map to give a child a slice of"
            )
        if length is None:
            raise ValueError(
                f"A child of unknown length takes no slice of the measure map of "
                f"timeline {self._id!r}"
            )
        start = Fraction(self._resolve_axis_value(offset))
        end = start + Fraction(length)
        to_quarters: LinearMap | None = None
        if isinstance(record, ChainMap):
            first = record.maps[0]
            if not isinstance(first, LinearMap):
                raise TypeError(
                    f"Timeline {self._id!r} reads its measure map through a "
                    f"{type(first).__name__}, not a LinearMap to quarters"
                )
            to_quarters = first
            scalar, shift = Fraction(to_quarters.scalar), Fraction(to_quarters.offset)
            start, end = start * scalar + shift, end * scalar + shift
        sliced = measure_map.get_slice(start, end)
        views: list[ConversionMap[Any]] = [
            type(view)(sliced)
            for view in self._conversion_maps.values()
            if isinstance(view, _MEASURE_VIEWS) and view.measure_map == measure_map
        ]
        return _MeasureSlice(
            sliced, None if to_quarters is None else to_quarters.with_offset(0), views
        )

    def create_children_from_regions(
        self,
        region_ids: Sequence[str] | None = None,
        *,
        copy_events: bool = True,
        measures: bool = False,
        seed: int | None = None,
    ) -> list["Timeline"]:
        """Create children from multiple regions (batch partitioning).

        Each region becomes a child. Regions may overlap — resulting children
        are independent.

        Args:
            region_ids: Region ids. None = all regions in insertion order.
            copy_events: Copy events to children.
            measures: Give each child the slice of this timeline's measure
                map over its region, as :meth:`create_child` does.
            seed: The seed of one minter of the children's ids, minted in
                region order; None mints each from its content.

        Returns:
            List of child timelines in region order.

        Raises:
            KeyError: If any region id is not found.
            RuntimeError: If timeline is locked.
            ValueError: With *measures*, as :meth:`create_child_from_region`
                raises.

        Examples:
            >>> tl.create_regions_by_grouping("@pageIndex",
            ...                               name_format="page_{value}")
            >>> tl.create_children_from_regions()  # All pages as children
        """
        if region_ids is None:
            region_ids = list(self._regions.keys())

        minter = None if seed is None else IdMinter(seed)
        prefix = timeline_id_prefix(self._unit)
        result: list[Timeline] = []
        for region_id in region_ids:
            child = self.create_child_from_region(
                region_id,
                copy_events=copy_events,
                measures=measures,
                uid=None if minter is None else minter.mint(prefix),
            )
            result.append(child)
        return result

    def get_children_at(
        self,
        coord: CoordinateSpec,
    ) -> list["Timeline"]:
        """Return all children that hold the given coordinate.

        A child holds every position from its offset up to its end, and its
        end too unless a sibling starts there: one boundary position belongs
        to the sibling starting at it, while the end of a child nothing
        follows stays on that child. A child of unknown length holds every
        position from its offset on (up to its successor's offset in a
        segment line). This is the rule the timeline's stamps and tables
        follow, so a child listed here is a child the stamp at *coord*
        names.

        Args:
            coord: The coordinate to query.

        Returns:
            List of child Timeline objects, ordered by offset.
            Empty list if no children contain coord.
        """
        value = self._resolve_axis_value(coord)
        matching = [
            (offset.value, self._children[child_id])
            for child_id, offset in self._child_offsets.items()
            if self._children[child_id]._holds_position(
                shift_number(value, offset.value, subtract=True)
            )
        ]
        matching.sort(key=lambda x: x[0])
        return [child for _, child in matching]

    def list_children(self) -> list[str]:
        """List child timeline IDs.

        Returns:
            List of child IDs in insertion order.
        """
        return list(self._children.keys())

    def has_child(self, child_id: str) -> bool:
        """Check if a child with the given ID exists.

        Args:
            child_id: The child ID to check.

        Returns:
            True if such a child exists.
        """
        return child_id in self._children

    def _copy_events_to_cut(
        self,
        target: "Timeline",
        start: Any,
        end: Any,
        *,
        copy_events: bool = True,
        in_this_tree: bool = False,
    ) -> None:
        """Copy the events starting in [start, end) onto *target*, cut there.

        The one copy of a child made from a region and of a segment: each
        event whose start lies in [start, end) (``get_events``, children's
        events included) is taken whole, its ``start``, ``end`` and
        ``instant`` shifted by -start, and added through
        :meth:`~timetoalign.timelines.Timeline.add_events`, so *target* grows
        to hold an event running past *end*; an event starting before
        *start* is not taken. Where this timeline holds Boxes, *target*
        states its place in the image first (:meth:`_move_rows_onto_cut`),
        with or without events. An event ID met twice is taken once, from
        the first row, which is this timeline's own before any child's.

        A copy for a timeline joining this one's family (a child cut from a
        region) is a new event of the family, whose event IDs are unique: it
        is minted an ID of its own (``add_events``' minting), the event it
        copies keeping its own, and the family's registry records which
        event it copies. A child cut earlier from an
        overlapping region holds such copies: a row recorded as a copy of an
        event taken is not taken again, so each event is copied once, from
        the event itself. A descendant's own event is taken whatever it
        holds.

        Args:
            target: The new timeline, standing at *start* of this one.
            start: The cut's start, in this timeline's number type.
            end: The cut's end, likewise.
            copy_events: Whether the events are copied; an area is stated
                either way.
            in_this_tree: Whether *target* joins this timeline's family, so
                that the copies take IDs of their own.

        Raises:
            ValueError: As :meth:`_move_rows_onto_cut` refuses a Box, or as
                ``add_events`` refuses a row; *target* is then incomplete.
        """
        rows: list[dict[str, Any]] = []
        origins: list[str | None] = []
        if copy_events:
            store = self._registry._family
            taken: set[str] = set()
            for event in self.get_events(min_coord=start, max_coord=end):
                # The timeline's own row comes first and is the one taken.
                event_id = event.get("id")
                origin = None if event_id is None else store.origin_of(event_id)
                if event_id is not None:
                    if event_id in taken or origin in taken:
                        continue
                    taken.add(event_id if origin is None else origin)
                row = dict(event)
                for key in ("instant", "start", "end"):
                    if row.get(key) is not None:
                        row[key] = shift_coordinate(row[key], start, subtract=True)
                rows.append(row)
                origins.append(event_id if origin is None else origin)
        minted = self._move_rows_onto_cut(
            target, rows, start, end, stated=False, new_ids=in_this_tree
        )
        if in_this_tree:
            copied = [
                (copy, origin)
                for copy, origin in zip(minted, origins, strict=True)
                if origin is not None
            ]
            target._registry._family.record_copies(
                target.id,
                [copy for copy, _ in copied],
                [origin for _, origin in copied],
            )

    def _holds_boxes(self) -> bool:
        """Whether this timeline holds a Box: among its own events or a
        descendant's, the events :meth:`get_events` gives."""
        table = self._events.table
        if (
            "event_type" in table.column_names
            and pc.any(
                pc.equal(table.column("event_type"), pa.scalar(BOX_EVENT_TYPE))
            ).as_py()
        ):
            return True
        return any(child._holds_boxes() for child in self._children.values())

    def _stated_area(self) -> Shape | None:
        """The area of an image this timeline holding Boxes stands for.

        ``meta`` is free-form; only a timeline holding Boxes
        (:meth:`_holds_boxes`) reads ``meta["box"]`` as its area, the shape
        its Boxes are placed against.

        Returns:
            The shape ``meta["box"]`` states, read on this timeline's axis;
            None where ``meta`` holds no ``box``.

        Raises:
            ValueError: If ``meta["box"]`` is not the dict form of a shape on
                this timeline's axis, naming the timeline.
        """
        box = self._meta.get("box")
        if box is None:
            return None
        refusal = (
            f'Timeline {quote_id(self._id)} holds Boxes, so its meta["box"] '
            f"must be its area of their image, a shape in {self._unit} as "
            f"{self._number_type}"
        )
        if not isinstance(box, Mapping):
            raise ValueError(
                f"{refusal}: it holds {box!r}, not the dict form of a "
                f"BoundingBox or a Quadrilateral."
            )
        try:
            return Shape.from_dict(box, unit=self._unit, number_type=self._number_type)
        except ValueError as error:
            raise ValueError(f"{refusal}: {error}") from None

    def _image_area_of_cut(
        self, area: Shape | None, at: Any, length: Any
    ) -> dict[str, Any]:
        """The ``meta`` entries of a timeline cut from this one at *at*.

        Only for a timeline holding Boxes (:meth:`_holds_boxes`). It stands
        for *area* (:meth:`_stated_area`) of the image ``meta["image"]``
        names. A cut of *length* from *at* measures the same image
        (``image`` carried over) and stands for the part of *area* over its
        x extent. A cut covering this timeline's whole extent stands for
        *area* itself: ``meta["box"]`` is kept as it is written. Any other
        cut states a box at ``x`` = the area's ``x_min`` + *at*, of
        ``width`` *length*, with the area's ``y`` and ``height``.

        Known issue: the area of a proper cut is always a BoundingBox; where
        *area* is a Quadrilateral, it is the cut of the quadrilateral's
        bounding box, not the quadrilateral clipped.

        Without *area* this timeline states no y extent, so the cut states
        no box.

        Args:
            area: This timeline's area, None where it states none.
            at: Where the cut starts on this timeline, in its number type.
            length: The cut's length, likewise.

        Returns:
            The entries, ``box`` and ``image`` where this timeline states
            them.
        """
        entries: dict[str, Any] = {}
        if "image" in self._meta:
            entries["image"] = copy.deepcopy(self._meta["image"])
        if area is None:
            return entries
        if at == 0 and length == self._extent:
            entries["box"] = copy.deepcopy(self._meta["box"])
            return entries
        bounds = area.bounding_box
        entries["box"] = BoundingBox.from_xywh(
            express_as(
                shift_number(bounds.x.value, at, subtract=False), self._number_type
            ),
            bounds.y,
            length,
            bounds.height,
            unit=self._unit,
            number_type=self._number_type,
        ).to_dict()
        return entries

    def _move_rows_onto_cut(
        self,
        target: "Timeline",
        rows: list[dict[str, Any]],
        start: Any,
        end: Any,
        *,
        stated: bool,
        new_ids: bool = False,
    ) -> list[str]:
        """Add *rows*, moved from [start, end) of this timeline, onto *target*.

        The rows are added through ``add_events``, or, with *stated*, as a
        source's rows are read (``_add_stated_events``: no ID minted, no
        growth). A timeline holding no Box (:meth:`_holds_boxes`) gives
        *target* nothing else: its ``meta`` is its own.

        A timeline holding Boxes stands for an area of an image. A Box's
        shape stays in the image's space while its interval moves with the
        cut, so *target* first states its own place in the image
        (:meth:`_image_area_of_cut`): ``start - a`` of each moved Box is then
        ``x_min - o(target)``, the invariant the Box check holds every Box
        to. Where ``add_events`` grew *target*, the area is stated again
        over the grown length.

        Known issue: a timeline holding Boxes that states no area
        (``meta["box"]``) gives the cut no area. One of its own Boxes can
        move only onto a cut at 0 (both stand at x 0). A Box a descendant
        holds (a row whose ``source_timeline`` is not null) stands against
        that descendant's area and cannot move at all: a page made of
        accolades, or a volume of such pages, is a concatenation of areas
        and none itself, so no timeline cut from it has an x origin its
        accolades' Boxes agree with.

        Args:
            target: The new timeline, standing at *start* of this one.
            rows: The rows, their positions already shifted by -start.
            start: The cut's start, in this timeline's number type.
            end: The cut's end, likewise.
            stated: Add as rows read from a source rather than through
                ``add_events``.
            new_ids: Add the rows without their IDs, which ``add_events``
                mints anew (copies joining this timeline's tree); a refusal
                names a row by the ID it was given.

        Returns:
            The IDs ``add_events`` gave the rows, in their order; none for
            *stated* rows.

        Raises:
            ValueError: If this timeline holds Boxes and its ``meta["box"]``
                is no shape (:meth:`_stated_area`); if this timeline states
                no area and a row is a Box that a descendant holds, or one
                of its own while *start* is not 0; or as the Box check or
                ``add_events`` refuses a row. Nothing is added then.
        """
        holds_boxes = self._holds_boxes()
        area = self._stated_area() if holds_boxes else None
        if holds_boxes and area is None:
            for row in rows:
                if row.get("event_type") == BOX_EVENT_TYPE and (
                    start != 0 or row.get("source_timeline") is not None
                ):
                    raise ValueError(
                        f"Box {quote_id(row['id'])} cannot move onto a timeline "
                        f"cut at {_format_coordinate_value(start)} from timeline "
                        f"{quote_id(self._id)}: the timeline states no area "
                        f'(meta["box"]), so the cut\'s area cannot be stated.'
                    )
        extent = end - start
        if holds_boxes:
            target._meta.update(self._image_area_of_cut(area, start, extent))
        added: list[str] = []
        if rows:
            if stated:
                target._add_stated_events(rows)
            else:
                added = target.add_events(
                    [{**row, "id": None} for row in rows] if new_ids else rows
                )
        grown = target.length
        if (
            not stated
            and area is not None
            and grown is not None
            and grown.value > extent
        ):
            target._meta.update(self._image_area_of_cut(area, start, grown.value))
        return added

    def _slice_bounds(
        self, start: CoordinateSpec, end: CoordinateSpec
    ) -> tuple[Any, Any]:
        """Resolve and validate slice bounds in this timeline's number type.

        The end is checked against the length only where the length is
        known: positions on a timeline of unknown length are not bounded
        above, so a slice may end anywhere after its start.

        Raises:
            ValueError: If start >= end, start < 0, or end lies beyond a known
                length.
        """
        start_value = self._resolve_axis_value(start)
        end_value = self._resolve_axis_value(end)
        nt = self._number_type
        if nt == NumberType.fraction:
            s, e = Fraction(start_value), Fraction(end_value)
        elif nt == NumberType.int:
            s, e = int(start_value), int(end_value)
        else:
            s, e = float(start_value), float(end_value)

        if s >= e:
            raise ValueError(
                f"start ({_format_coordinate_value(float(s))}) must be less than "
                f"end ({_format_coordinate_value(float(e))})"
            )
        length = self._length
        bound = (
            "unknown"
            if length is None
            else _format_coordinate_value(float(length.value))
        )
        if s < 0:
            raise ValueError(
                f"start ({_format_coordinate_value(float(s))}) is outside timeline "
                f"bounds [0, {bound})"
            )
        if length is not None and e > length.value:
            raise ValueError(
                f"end ({_format_coordinate_value(float(e))}) is outside timeline bounds "
                f"[0, {bound}]"
            )
        return s, e

    def get_slice(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        *,
        truncate_events: bool = True,
        include_children: bool = True,
        copy_cmaps: bool = True,
        seed: int | None = None,
    ) -> "Timeline":
        """Extract a portion of this timeline as a new, independent timeline.

        Returns a new timeline containing all events within [start, end).
        The returned timeline has its coordinate origin at 0, with all
        coordinates shifted by -start.

        Slicing creates a new timeline that is a structural copy of the
        specified interval of the source.

        A timeline holding Boxes (itself or a descendant) stands for an area
        of an image (``meta["box"]``): the slice states the part of that
        area over its x extent (the area as written where the slice covers
        the whole timeline), and ``meta["image"]`` is carried over, so a Box
        keeps its shape, in the image's space, and its start moves with the
        slice's origin. ``meta`` is otherwise free-form: a timeline holding
        no Box gives the slice none of it.

        Known issues: a Box crossing a slice boundary cannot be clipped (its
        interval would contradict its shape), so with *truncate_events* the
        slice is refused; slice with ``truncate_events=False`` to leave such
        a Box out (the flag holds for the slices of the children too). The
        area of a proper slice of a quadrilateral area is the slice of its
        bounding box. A timeline holding Boxes that states no area moves a
        Box of its own only into a slice starting at 0; its children are
        sliced by this rule in turn, so a page made of accolades is sliced
        accolade by accolade, each slice stating its own area.

        Args:
            start: Start coordinate (inclusive).
            end: End coordinate (exclusive).
            truncate_events: If True (default), interval events straddling
                the slice boundaries are clipped to [start, end). If False,
                events must be fully contained to be included. Children are
                sliced alike.
            include_children: If True (default), child timelines whose span
                overlaps [start, end) are recursively sliced and included.
            copy_cmaps: If True (default), ConversionMaps are bounded-copied
                for the slice range.
            seed: The seed of one minter of every id the slice mints, in
                the order built: the slice, then its children's slices, then
                the copied maps (:func:`~timetoalign.core.ids.minting_from`);
                without one, each from its content. Not stored.

        Returns:
            New Timeline (same concrete subclass) with length = end - start,
            coordinates shifted to [0, end-start).

        Raises:
            ValueError: If start >= end or either is outside timeline bounds
                (the end is not bounded on a timeline of unknown length); if
                a Box crosses a boundary and *truncate_events* is set; if
                this timeline holds Boxes and its ``meta["box"]`` is no
                shape; or if a Box would move while this timeline states no
                ``meta["box"]`` and *start* is not 0.

        Examples:
            >>> source = ContinuousLogicalTimeline(length=100)
            >>> source.add_events([
            ...     {"event_type": "Note", "start": 10, "end": 30},
            ...     {"event_type": "Beat", "instant": 25},
            ... ])
            >>> sliced = source.get_slice(20, 40)
            >>> sliced.length.value  # 40 - 20 = 20
            Fraction(20, 1)
        """
        s, e = self._slice_bounds(start, end)

        slice_length = e - s

        with minting_from(seed):
            sliced = self._empty_slice(slice_length)

            # Copy events with coordinate shifting
            self._copy_events_to_slice(sliced, s, e, truncate_events)

            # Recursively slice children
            if include_children:
                self._copy_children_to_slice(sliced, s, e, truncate_events)

            # Copy conversion maps (bounded to slice range)
            if copy_cmaps:
                self._copy_cmaps_to_slice(sliced, s, e)

        self._close_slice(sliced, slice_length)
        return sliced

    def _empty_slice(self, length: CoordinateValue) -> "Timeline":
        """The timeline :meth:`get_slice` fills: of the spawn class, *length* long."""
        return self._spawn_class()(
            length=length,
            unit=self._unit,
            number_type=self._number_type,
        )

    def _close_slice(self, sliced: "Timeline", length: CoordinateValue) -> None:
        """Settle a filled slice of *length*: nothing here."""

    def _copy_events_to_slice(
        self,
        target: "Timeline",
        start: Any,
        end: Any,
        truncate_events: bool,
    ) -> None:
        """Copy events from [start, end) to target with coordinate shifting.

        All coordinates are shifted by -start. Interval events straddling
        boundaries are either truncated or excluded depending on
        truncate_events.

        Note: In the EventData PyArrow schema, all events store their
        coordinate in 'start' (a struct with 'value', 'numerator',
        'denominator' keys). Instant events have 'start' but no 'end'.
        The 'instant' key is only used as input convenience during
        ``add_events()`` and is converted to 'start' internally.

        The rows are added as a source's rows are read, through the one Box
        check, after *target* states its place in the image
        (:meth:`_move_rows_onto_cut`). A Box is never clipped: one that
        truncating would clip is refused.

        Args:
            target: The target timeline to receive events.
            start: Start coordinate (inclusive) in source coords.
            end: End coordinate (exclusive) in source coords.
            truncate_events: If True, clip straddling intervals. If False,
                only include fully-contained intervals.

        Raises:
            ValueError: If truncating would clip a Box, or as
                :meth:`_move_rows_onto_cut` refuses a row.
        """
        # Get this timeline's own events only. Children are copied separately
        # by _copy_children_to_slice, so pulling their events up here would
        # double-count them (once at the slice's parent level, once inside the
        # recursively-sliced child).
        all_events = self.get_events(include_children=False)

        # *start* and *end* are in this timeline's number type and each cell
        # is read exactly (coordinate_numeric_value), so an exact axis is
        # compared exactly and a float one as its doubles.

        adjusted_events = []
        for event in all_events:
            ev = dict(event)
            temporal = ev.get("temporal_type")

            if temporal == "instant":
                # In EventData, instant events have start={value:...} and end=None
                start_dict = ev.get("start")
                if start_dict is None:
                    continue
                coord = coordinate_numeric_value(start_dict)
                # Left-inclusive, right-exclusive: start <= coord < end
                if start <= coord < end:
                    ev["start"] = shift_coordinate(start_dict, start, subtract=True)
                    adjusted_events.append(ev)

            elif temporal == "interval":
                ev_start_dict = ev.get("start")
                ev_end_dict = ev.get("end")
                if ev_start_dict is None or ev_end_dict is None:
                    continue
                ev_start = coordinate_numeric_value(ev_start_dict)
                ev_end = coordinate_numeric_value(ev_end_dict)

                if truncate_events:
                    # Clip to [start, end)
                    clipped_start = max(ev_start, coordinate_numeric_value(start))
                    clipped_end = min(ev_end, coordinate_numeric_value(end))

                    if clipped_start >= clipped_end:
                        continue  # Fully outside
                    if ev.get("event_type") == BOX_EVENT_TYPE and (
                        clipped_start != ev_start or clipped_end != ev_end
                    ):
                        self._refuse_clipping_box(ev, ev_start, ev_end, start, end)

                    ev["start"] = shift_coordinate(clipped_start, start, subtract=True)
                    ev["end"] = shift_coordinate(clipped_end, start, subtract=True)
                    ev["duration"] = subtract_coordinates(ev["end"], ev["start"])
                    adjusted_events.append(ev)
                else:
                    # Only include fully contained intervals
                    if ev_start >= start and ev_end <= end:
                        ev["start"] = shift_coordinate(
                            ev_start_dict, start, subtract=True
                        )
                        ev["end"] = shift_coordinate(ev_end_dict, start, subtract=True)
                        ev["duration"] = subtract_coordinates(ev["end"], ev["start"])
                        adjusted_events.append(ev)

        self._move_rows_onto_cut(target, adjusted_events, start, end, stated=True)

    def _refuse_clipping_box(
        self,
        row: dict[str, Any],
        box_start: Any,
        box_end: Any,
        start: Any,
        end: Any,
    ) -> None:
        """Refuse to clip the Box *row*, which crosses the slice [start, end).

        Raises:
            ValueError: Always, naming the Box, its span on this timeline and
                the boundary it crosses (*start* where it crosses both).
        """
        boundary = start if box_start < start else end
        raise ValueError(
            f"Box {quote_id(row['id'])} runs from "
            f"{_format_coordinate_value(box_start)} to "
            f"{_format_coordinate_value(box_end)} on timeline {quote_id(self._id)} "
            f"and crosses the slice's boundary at "
            f"{_format_coordinate_value(boundary)}: a slice clips the intervals "
            f"that cross its boundaries, and a clipped Box would contradict its "
            f"shape. Slice with truncate_events=False to leave it out."
        )

    def _copy_children_to_slice(
        self,
        target: "Timeline",
        start: Any,
        end: Any,
        truncate_events: bool,
    ) -> None:
        """Recursively slice child timelines that overlap [start, end).

        For each child at offset o with length l, if [o, o+l) overlaps
        [start, end), recursively slice the child at the overlapping range,
        with the slice's own *truncate_events*. The child's offset in the
        target is max(0, o - start); the slice of a child is locked exactly
        when the child is.

        Args:
            target: The target timeline to receive sliced children.
            start: Start coordinate in source (parent) coords.
            end: End coordinate in source (parent) coords.
            truncate_events: Whether each child's slice clips the intervals
                crossing its boundaries (else leaves them out).
        """
        for child_id, child in self._children.items():
            child_offset = self._child_offsets[child_id].value

            # Check overlap with [start, end); a child of unknown length runs
            # on without bound.
            overlap_start = max(child_offset, start)
            extent = child._extent
            overlap_end = end if extent is None else min(child_offset + extent, end)

            if overlap_start >= overlap_end:
                continue  # No overlap

            # Convert to child-local coordinates
            child_local_start = overlap_start - child_offset
            child_local_end = overlap_end - child_offset

            # Recursively slice the child
            sliced_child = child.get_slice(
                child_local_start,
                child_local_end,
                truncate_events=truncate_events,
                include_children=True,
                copy_cmaps=True,
            )

            # Offset in the target timeline
            target_offset = overlap_start - start
            target._embed_child(
                sliced_child,
                target_offset,
                allow_expansion=True,
                locked=target._kept_lock(child),
            )

    def _copy_cmaps_to_slice(
        self,
        target: "Timeline",
        start: Any,
        end: Any,
    ) -> None:
        """Copy conversion maps to a sliced timeline.

        Currently copies all conversion maps without range bounding.
        Future enhancement: implement bounded C-Map copying for the
        slice coordinate range.

        Args:
            target: The target timeline to receive C-Maps.
            start: Start coordinate in source coords (for future bounded copy).
            end: End coordinate in source coords (for future bounded copy).
        """
        # A slice does not inherit the parent's measure layer.  That layer
        # describes a different extent, measured from a different origin,
        # so handing it over verbatim would state bar positions the slice
        # does not have; the slice answers with its own local structure.
        # What a child should inherit from its parent's
        # structure is a materialization decision, to be made where
        # children are derived from that structure rather than here.
        for cmap_id, cmap in self._conversion_maps.items():
            # The slice holds its own copy under an id of its own, minted
            # from the source map's id: a slice may join a manifest that
            # holds the source's map. Maps nested in it are shared. A
            # measure view's copy keeps the id it holds for its own measure
            # map (``_own_measure_map_id``): the rows it writes equal the
            # source's, so a manifest holding both writes one entry.
            copied = copy.copy(cmap)
            copied._id = content_minter(cmap._id_prefix, cmap.id).mint(cmap._id_prefix)
            try:
                target.add_conversion_map(copied)
            except (ValueError, TypeError):
                # Skip incompatible maps (e.g., unit mismatch after slicing)
                self._logger.debug(
                    f"Skipping C-Map '{cmap_id}' during slice: incompatible"
                )
