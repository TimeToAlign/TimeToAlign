"""Provide child timeline and slicing operations for timeline instances."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

import pyarrow.compute as pc

from timetoalign.core import Coordinate, CoordinateSpec, CoordinateValue, NumberType
from timetoalign.core.ids import mint_id, timeline_id_prefix
from timetoalign.core.time import (
    coordinate_numeric_value,
    number_to_json,
    shift_coordinate,
    subtract_coordinates,
)
from timetoalign.core.timestamp import ConversionMapsSpec, _format_coordinate_value
from timetoalign.maps import ConversionMap

from ..manifest import spell
from ..regions import Region
from .regions import _resolve_boundary_names

if TYPE_CHECKING:
    from ..base import Timeline

SEGMENT_EVENT_TYPE = "Segment"
TraversalOrder = Literal["sorted", "depth_first", "breadth_first"]


class ChildrenMixin:
    """Provide child timeline and slicing operations for timeline instances."""

    def _spawn_class(self) -> type["Timeline"]:
        """The class used for children and slices of this timeline."""
        return type(self)

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

    def _set_length(self, length: Coordinate | None) -> None:
        """Set this timeline's length, which its parent must contain.

        The parent checks the new end (:meth:`_check_child_length`) before
        anything changes; it never grows because a child grew. Then the
        length is set and the parent's ``Segment`` row follows it.

        Raises:
            ValueError: If the parent cannot contain the new end.
        """
        parent = self._parent
        if parent is not None:
            parent._check_child_length(self._id, length)
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
        is_row = pc.fill_null(
            pc.and_kleene(
                pc.equal(table.column("event_type"), SEGMENT_EVENT_TYPE),
                pc.equal(table.column("id"), child_id),
            ),
            False,
        )
        position = pc.indices_nonzero(is_row)[0].as_py()
        row = type(events).from_dicts(
            [self._segment_row(child_id)], events.unit, events.number_type
        )
        combined = events.concat(row)
        order = list(range(len(events)))
        order[position] = len(events)
        self._events = type(events)(
            combined.table.take(order), events.unit, events.number_type
        )

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

        Raises:
            TypeError: If child is not a Timeline.
            ValueError: If units don't match, the timeline already is a child
                of this timeline, or the offset is negative, which is not
                supported yet.
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

        if any(existing is child for existing in self._children.values()):
            raise ValueError(f"Child '{child.id}' is already a child of this timeline")

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
    ) -> Timeline:
        """Convert a child timeline to this timeline's unit via a C-Map.

        The conversion map is resolved from the parent's attached C-Maps.
        We find a map whose ``target_unit`` matches the child's unit, invert
        it, and use it to derive a copy of the child in the parent's unit.

        The derived copy is named ``"{child.name} [{parent.unit}]"``, gets a
        minted id in the parent's unit (minted again in the parent's manifest
        when it joins), and retains all events with converted coordinates.

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
        inverse_cmap = cmap.inverse()

        # Temporarily add the inverse map to the child so derive() can use it
        child_had_map = bool(child._get_unit_maps(self._unit))
        if not child_had_map:
            child.add_conversion_map(inverse_cmap)

        try:
            derived = child.derive(
                self._unit,
                copy_events=True,
            )
        finally:
            # Clean up: remove the temporarily added map if we added it.
            # Removal is by identity, so a map the child already carried for
            # the same unit survives.
            if not child_had_map and inverse_cmap.id in child._conversion_maps:
                del child._conversion_maps[inverse_cmap.id]
                registered = child._unit_maps.get(self._unit)
                if registered is not None:
                    registered[:] = [
                        cmap for cmap in registered if cmap is not inverse_cmap
                    ]
                    if not registered:
                        del child._unit_maps[self._unit]

        # The name links back to the original; the id is minted, because a
        # label is no identity.
        derived._set_id(mint_id(timeline_id_prefix(self._unit), ()), minted=True)
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
        subtree whose minted id is already held here receives a freshly
        minted one, and an element whose given id is already held here makes
        the call raise without changing anything.

        A timeline can accommodate events and other timelines, called
        Children, as long as they use the same measuring unit.

        When the child uses a different unit than the parent, set
        ``use_conversion_map`` to automatically convert the child to the
        parent's unit via a C-Map. The parent must have a C-Map whose
        ``target_unit`` matches the child's unit, so that inverting it yields
        the ``child_unit -> parent_unit`` conversion. The child's events are
        copied with converted coordinates; the original child is NOT modified.

        The converted child is named ``"{child.name} [{parent.unit}]"`` and
        receives a minted id.

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

        Raises:
            TypeError: If child is not a Timeline.
            ValueError: If units don't match (and no conversion map given),
                the offset is negative (not supported yet), the child is
                already a child of this timeline, ``locked=True`` asks to lock
                a plain segment, ``locked=False`` asks to leave a segment line
                unlocked, an id given in the child's subtree is already
                held in this manifest, or this timeline is locked and the child
                would end beyond its length without ``allow_expansion``. An
                unlocked timeline expands to fit the child instead, provided
                its own parent contains its new end. A child of unknown
                length has no end, so only its offset must fit; a parent of
                unknown length has no end to exceed, unless it is a segment,
                which ends at its successor's offset.
        """
        # Convert child's unit if a conversion map is specified
        if use_conversion_map is not None and child.unit != self._unit:
            child = self._convert_child_to_parent_unit(child, use_conversion_map)

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
        id_plan = self._manifest_join_plan(child)

        # Ensure capacity
        self._ensure_capacity(
            offset_coord.value if child_end is None else child_end, allow_expansion
        )

        for owner, old_id, new_id in id_plan:
            owner._reassign_element_id(old_id, new_id)

        # Store child reference
        self._children[child.id] = child
        self._child_offsets[child.id] = offset_coord
        child._parent = self

        if plain_segment:
            child._locked = False
        elif locked:
            child._locked = True

        self._add_events_unchecked([self._segment_row(child.id)])

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
    ) -> "Timeline":
        """Create and embed a child selected by this timeline's spawn hook.

        By default, this is equivalent to:

            child_type = parent._spawn_class()
            child = child_type(length=length, unit=parent.unit, uid=uid, name=name)
            parent.add_child(child, offset=offset)

        ``child_class=Timeline`` deliberately selects the experimental base class.

        Args:
            length: Child length in the parent's unit, or ``None`` when the
                child's length is unknown.
            offset: Child start coordinate. Defaults to zero.
            uid: Unique identifier for the child. Auto-generated if None.
            name: Human-readable name for the child.
            allow_expansion: If True, expand the parent when needed.
            child_class: Explicit class override for the child.

        Returns:
            The newly created and embedded child Timeline.

        Raises:
            ValueError: If coordinates are invalid or the child exceeds bounds.

        Examples:
            >>> parent = ContinuousLogicalTimeline(length=8)
            >>> child = parent.create_child(length=4)
            >>> type(child) is ContinuousLogicalTimeline
            True
        """
        if length is not None:
            length = self._resolve_axis_value(length)

        child = (child_class or self._spawn_class())(
            length=length,
            unit=self._unit,
            number_type=self._number_type,
            uid=uid,
            name=name,
        )
        self.add_child(child, offset=offset, allow_expansion=allow_expansion)
        return child

    def create_children_from_boundaries(
        self,
        boundaries: Sequence[CoordinateSpec],
        *,
        names: Sequence[str] | None = None,
        name_format: str = "{prefix}_{n}",
        prefix: str = "section",
        allow_expansion: bool = False,
    ) -> list["Timeline"]:
        """Create children spanning consecutive boundary coordinates.

        Source events are not copied into the children.

        Args:
            boundaries: k+1 monotonically increasing coordinates.
            names: Explicit names for the k children.
            name_format: Format with {prefix}, {i}, and {n} placeholders.
            prefix: Prefix for auto-generated names.
            allow_expansion: If True, expand the parent when needed.

        Returns:
            Child timelines in boundary order.

        Raises:
            ValueError: If the boundaries or number of names are invalid.
        """
        coords, child_names = _resolve_boundary_names(
            boundaries,
            names,
            name_format,
            prefix,
            self._resolve_axis_value,
        )
        return [
            self.create_child(end - start, start, name, name, allow_expansion)
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
        of a segment line).

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
        if uid is not None:
            child._set_id(uid)
        if name is not None:
            child._name = name
        self.add_child(child, end, allow_expansion=True)

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
        self._events = self._events._with_segment_renamed(old_id, new_id)

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
            yield self._make_coordinate(0), self

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
    ) -> "Timeline":
        """Create a child timeline from a region (partitioning).

        The child's length = region duration, offset = region start.
        The child's class matches the parent's concrete class, and its name
        is the region's name. If copy_events, events in [start, end) are
        copied with adjusted coordinates.

        Args:
            region_id: Id of an existing region.
            copy_events: Copy events within the region to the child.
            uid: Explicit child ID. When None, one is minted.

        Returns:
            The newly created and attached child timeline.

        Raises:
            KeyError: If region_id not found.
            RuntimeError: If timeline is locked.

        Examples:
            >>> tl.create_regions_by_splitting("breaks", prefix="movement")
            >>> mov4 = tl.create_child_from_region("region_4")
            >>> mov4.name
            'movement_4'
        """
        region = self._regions.get(region_id)
        if region is None:
            raise KeyError(
                f"Region '{region_id}' not found on timeline '{self._id}'. "
                f"Available regions: {list(self._regions.keys())}"
            )

        self._check_not_locked("create child from region")

        child = self._spawn_class()(
            length=region.duration.value,
            unit=self._unit,
            number_type=self._number_type,
            uid=uid,
            name=region.name,
        )

        if copy_events:
            self._copy_events_to_child(child, region)

        self.add_child(child, offset=region.start)
        return child

    def create_children_from_regions(
        self,
        region_ids: Sequence[str] | None = None,
        *,
        copy_events: bool = True,
    ) -> list["Timeline"]:
        """Create children from multiple regions (batch partitioning).

        Each region becomes a child. Regions may overlap — resulting children
        are independent.

        Args:
            region_ids: Region ids. None = all regions in insertion order.
            copy_events: Copy events to children.

        Returns:
            List of child timelines in region order.

        Raises:
            KeyError: If any region id is not found.
            RuntimeError: If timeline is locked.

        Examples:
            >>> tl.create_regions_by_grouping("@pageIndex",
            ...                               name_format="page_{value}")
            >>> tl.create_children_from_regions()  # All pages as children
        """
        if region_ids is None:
            region_ids = list(self._regions.keys())

        result: list[Timeline] = []
        for region_id in region_ids:
            child = self.create_child_from_region(region_id, copy_events=copy_events)
            result.append(child)
        return result

    def get_children_at(
        self,
        coord: CoordinateSpec,
    ) -> list["Timeline"]:
        """Return all children whose extent contains the given coordinate.

        A child contains coord if offset <= coord < offset + child.length; a
        child of unknown length contains every coord from its offset on (up
        to its successor's offset in a segment line).

        Args:
            coord: The coordinate to query.

        Returns:
            List of child Timeline objects, ordered by offset.
            Empty list if no children contain coord.
        """
        coord_val = float(self._resolve_axis_value(coord))
        matching: list[tuple[float, Timeline]] = []
        for child_id, child in self._children.items():
            offset = float(self._child_offsets[child_id].value)
            extent = child._extent
            if offset <= coord_val and (
                extent is None or coord_val < offset + float(extent)
            ):
                matching.append((offset, child))
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

    def _copy_events_to_child(
        self,
        child: "Timeline",
        region: Region,
    ) -> None:
        """Copy events within a region to a child, adjusting coordinates.

        Events in [region.start, region.end) are copied with coordinates
        shifted by -region.start so they are relative to the child's origin.

        Args:
            child: The target child timeline.
            region: The region defining the source interval.
        """
        events_in_region = self.get_events(
            min_coord=float(region.start.value),
            max_coord=float(region.end.value),
        )

        adjusted_events = []
        for event in events_in_region:
            adjusted = dict(event)
            for coord_col in ("instant", "start", "end"):
                val = adjusted.get(coord_col)
                if val is not None:
                    adjusted[coord_col] = shift_coordinate(
                        val, region.start.value, subtract=True
                    )
            adjusted_events.append(adjusted)

        if adjusted_events:
            child.add_events(adjusted_events)

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
    ) -> "Timeline":
        """Extract a portion of this timeline as a new, independent timeline.

        Returns a new timeline containing all events within [start, end).
        The returned timeline has its coordinate origin at 0, with all
        coordinates shifted by -start.

        Slicing creates a new timeline that is a structural copy of the
        specified interval of the source.

        Args:
            start: Start coordinate (inclusive).
            end: End coordinate (exclusive).
            truncate_events: If True (default), interval events straddling
                the slice boundaries are clipped to [start, end). If False,
                events must be fully contained to be included.
            include_children: If True (default), child timelines whose span
                overlaps [start, end) are recursively sliced and included.
            copy_cmaps: If True (default), ConversionMaps are bounded-copied
                for the slice range.

        Returns:
            New Timeline (same concrete subclass) with length = end - start,
            coordinates shifted to [0, end-start).

        Raises:
            ValueError: If start >= end or either is outside timeline bounds.
                The end is not bounded on a timeline of unknown length.

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

        # Create new timeline using the source's spawn class
        sliced = self._spawn_class()(
            length=slice_length,
            unit=self._unit,
            number_type=self._number_type,
        )

        # Copy events with coordinate shifting
        self._copy_events_to_slice(sliced, s, e, truncate_events)

        # Recursively slice children
        if include_children:
            self._copy_children_to_slice(sliced, s, e)

        # Copy conversion maps (bounded to slice range)
        if copy_cmaps:
            self._copy_cmaps_to_slice(sliced, s, e)

        return sliced

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

        Args:
            target: The target timeline to receive events.
            start: Start coordinate (inclusive) in source coords.
            end: End coordinate (exclusive) in source coords.
            truncate_events: If True, clip straddling intervals. If False,
                only include fully-contained intervals.
        """
        # Get this timeline's own events only. Children are copied separately
        # by _copy_children_to_slice, so pulling their events up here would
        # double-count them (once at the slice's parent level, once inside the
        # recursively-sliced child).
        all_events = self.get_events(include_children=False)

        # Convert start/end to float for comparison with PyArrow struct values
        start_f = float(start)
        end_f = float(end)

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
                if coord >= start_f and coord < end_f:
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

                    ev["start"] = shift_coordinate(clipped_start, start, subtract=True)
                    ev["end"] = shift_coordinate(clipped_end, start, subtract=True)
                    ev["duration"] = subtract_coordinates(ev["end"], ev["start"])
                    adjusted_events.append(ev)
                else:
                    # Only include fully contained intervals
                    if ev_start >= start_f and ev_end <= end_f:
                        ev["start"] = shift_coordinate(
                            ev_start_dict, start, subtract=True
                        )
                        ev["end"] = shift_coordinate(ev_end_dict, start, subtract=True)
                        ev["duration"] = subtract_coordinates(ev["end"], ev["start"])
                        adjusted_events.append(ev)

        if adjusted_events:
            target._add_events_unchecked(adjusted_events)

    def _copy_children_to_slice(
        self,
        target: "Timeline",
        start: Any,
        end: Any,
    ) -> None:
        """Recursively slice child timelines that overlap [start, end).

        For each child at offset o with length l, if [o, o+l) overlaps
        [start, end), recursively slice the child at the overlapping range.
        The child's offset in the target is max(0, o - start); the slice of
        a child is locked exactly when the child is.

        Args:
            target: The target timeline to receive sliced children.
            start: Start coordinate in source (parent) coords.
            end: End coordinate in source (parent) coords.
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
                truncate_events=True,
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
            # The slice holds its own copy: a slice may join a manifest that
            # holds the source's map, so the copy's id may be minted again.
            copied = copy.copy(cmap)
            copied._id_minted = True
            try:
                target.add_conversion_map(copied)
            except (ValueError, TypeError):
                # Skip incompatible maps (e.g., unit mismatch after slicing)
                self._logger.debug(
                    f"Skipping C-Map '{cmap_id}' during slice: incompatible"
                )
