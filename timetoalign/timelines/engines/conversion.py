"""Provide coordinate conversion operations for timeline instances."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np
import pandas as pd

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    IdCoordinate,
    NumberType,
    TimeUnit,
    express_as,
    resolve_coordinate_spec,
)
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    dispatch_retrieval,
    format_coordinates,
    validate_coordinate_collection,
    validate_key_collection,
)
from timetoalign.core.time import coordinate_numeric_value, exact_coordinate_value
from timetoalign.maps import ConversionMap

if TYPE_CHECKING:
    from ..base import Timeline

SEGMENT_EVENT_TYPE = "Segment"


class ConversionMapsMixin:
    """Provide coordinate conversion operations for timeline instances."""

    @property
    def n_conversion_maps(self) -> int:
        """Number of attached conversion maps."""
        return len(self._conversion_maps)

    def add_conversion_map(self, cmap: ConversionMap[Any]) -> None:
        """Add a ConversionMap to this timeline.

        Any map with a ``target_unit`` is automatically registered in the
        unified timestamp system so that :meth:`get_timestamp` can resolve
        coordinates in that unit. The map is stored as-is: ``TableMap``
        instances honor their own ``kind`` (nearest/previous/next/linear)
        and ``extrapolate`` policy directly, and analytical maps (e.g.
        ``ScalarMap``, ``LinearMap``) are likewise stored directly.

        Several maps may target the same unit — a piece can carry two
        readings of the same axis — and nothing is ever displaced:
        the unit registry holds a list, stamps and tables show every
        reading, and scalar lookup refuses to guess between them
        unless the caller names one.

        The map's id must be unique in this timeline's manifest: a minted id
        already held there is replaced by a freshly minted one (on the map
        itself), a given one raises.

        Args:
            cmap: The ConversionMap to add.

        Raises:
            ValueError: If the map's source unit is incompatible, the map is
                already attached, or its given id is already held in this
                timeline's manifest.
        """
        if cmap.source_unit is not None and cmap.source_unit != self._unit:
            raise ValueError(
                f"Map source unit '{cmap.source_unit}' does not match "
                f"timeline unit '{self._unit}'"
            )
        if any(attached is cmap for attached in self._conversion_maps.values()):
            raise ValueError(
                f"Conversion map '{cmap.id}' is already attached to timeline "
                f"'{self._id}'. Attached map ids: {list(self._conversion_maps)}"
            )
        cmap._id = self._admit_element_id(
            cmap.id,
            minted=cmap._id_minted,
            prefix=cmap._id_prefix,
            element=f"conversion map {cmap.id!r}",
        )
        self._conversion_maps[cmap.id] = cmap

        # Register in the unified timestamp system for unit-based lookup
        if cmap.target_unit is not None:
            self._unit_maps.setdefault(cmap.target_unit, []).append(cmap)

        self._logger.debug(f"Added conversion map '{cmap.id}'")

    def _rename_conversion_map(self, old_id: str, new_id: str) -> None:
        """Store the attached map *old_id* under the minted id *new_id*."""
        cmap = self._conversion_maps[old_id]
        cmap._id = new_id
        self._conversion_maps = {
            (new_id if key == old_id else key): attached
            for key, attached in self._conversion_maps.items()
        }

    def get_conversion_map(
        self, target_unit: TimeUnit | str
    ) -> ConversionMap[Any] | None:
        """Get a conversion map by target unit **or** by name/id.

        When *target_unit* is a valid `TimeUnit` value (or an alias such as
        ``"seconds"``), the method returns the attached map targeting that
        unit — and raises when several do, because picking one of two
        readings of the same axis is the caller's decision, not this
        method's.

        When *target_unit* is a string that does **not** correspond to any
        ``TimeUnit`` member, the method falls back to a name-based lookup:
        it searches first by ``cmap.id``, then by ``cmap.name``.  This is
        useful for maps where source and target units are identical (e.g. a
        ``ShiftMap`` named ``"raw_quarters"`` that maps normalised quarters
        back to raw partitura quarters).

        Args:
            target_unit: A ``TimeUnit`` member, a unit alias string, or a
                conversion-map name/id string.

        Returns:
            A matching ``ConversionMap``, or ``None`` if not found.

        Raises:
            ValueError: If several attached maps target the requested unit.

        Examples:
            >>> timeline.get_conversion_map(TimeUnit.seconds)
            ScalarMap(...)
            >>> timeline.get_conversion_map("raw_quarters")
            ShiftMap(offset=-0.5, ...)
        """
        # Attempt unit-based lookup first
        try:
            target = TimeUnit(target_unit)
        except ValueError:
            pass
        else:
            return self._get_unit_map(target)

        # Fallback: name/id-based lookup (target_unit is a plain string)
        name = str(target_unit)
        # Direct id lookup (O(1) — _conversion_maps is keyed by cmap.id)
        if name in self._conversion_maps:
            return self._conversion_maps[name]
        # Fallback to name attribute scan
        for cmap in self._conversion_maps.values():
            if cmap.name == name:
                return cmap
        return None

    def convert_to(
        self,
        values: CoordinateValue | Coordinate | np.ndarray,
        target_unit: TimeUnit | str,
    ) -> Coordinate | np.ndarray:
        """Convert coordinates to another unit using attached C-Maps.

        Args:
            values: Coordinate value(s) to convert. Can be:
                - Scalar (int, float, Fraction): Returns a Coordinate object
                - Coordinate: Returns a Coordinate object
                - numpy array: Returns a numpy array of converted values
            target_unit: Target unit.

        Returns:
            - For scalar/Coordinate input: Coordinate object in the target unit
            - For array input: numpy array of converted values

        Raises:
            ValueError: If no suitable map is found.

        Examples:
            >>> timeline.add_conversion_map(ScalarMap(scalar=1/300, ...))
            >>> coord = timeline.convert_to(15343, "inches")
            >>> coord
            Coordinate(51.1, inches)
            >>> arr = timeline.convert_to(np.array([100, 200]), "inches")
            >>> arr
            array([0.333, 0.666])
        """
        target = TimeUnit(target_unit)
        cmap = self.get_conversion_map(target)
        if cmap is None:
            raise ValueError(
                f"No conversion map found from '{self._unit}' to '{target}'"
            )
        converted_value = cmap(values)

        # Return array for array input, Coordinate for scalar input
        if isinstance(values, np.ndarray):
            if target.is_discrete:
                return np.vectorize(round, otypes=[int])(converted_value)
            return converted_value
        # The value now lives on the target unit's axis, so it is written the
        # way that axis writes numbers.
        declared = target.default_number_type
        return Coordinate(
            express_as(converted_value, declared), target, number_type=declared
        )

    def derive(
        self,
        target_unit: TimeUnit | str,
        name: str | None = None,
        copy_events: bool = False,
    ) -> "Timeline":
        """Create a derivative timeline in a different unit via C-Map conversion.

        A ConversionMap implies the presence of a derived timeline in the
        target unit. The derive() method makes this implicit timeline explicit.

        The derived timeline:
        - Has coordinates in the target unit
        - Has length equal to the converted source length
        - Automatically has an inverse C-Map back to the source unit
        - Optionally copies and converts events from the source

        This operation creates a NEW timeline, NOT a child timeline.
        The source and derived timelines have different units, so per TTA
        specification, they cannot be parent-child (children must share
        the parent's unit). Use TimelineGroup to connect them.

        Args:
            target_unit: The unit for the derived timeline.
            name: Optional name for the derived timeline.
            copy_events: If True, copy and convert events to the derived timeline.

        Returns:
            A new Timeline in the target unit.

        Raises:
            ValueError: If no C-Map exists for the target unit.
            ValueError: If C-Map is not invertible (needed for roundtrip).

        Examples:
            >>> # Create physical timeline with tempo C-Map
            >>> audio = ContinuousPhysicalTimeline(length=60.0)
            >>> audio.add_conversion_map(LinearMap(2.0, 0.0,
            ...     source_unit=TimeUnit.seconds, target_unit=TimeUnit.quarters))
            >>> # Derive a logical timeline
            >>> score = audio.derive(TimeUnit.quarters, name="score")
            >>> score.unit
            TimeUnit.quarters
            >>> score.length
            Coordinate(120.0, quarters)  # 60 seconds * 2 q/s
        """
        target = TimeUnit(target_unit) if isinstance(target_unit, str) else target_unit

        # Get the C-Map for this conversion
        cmap = self.get_conversion_map(target)
        if cmap is None:
            raise ValueError(
                f"No C-Map from '{self._unit}' to '{target}'. "
                f"Add a ConversionMap with add_conversion_map() first."
            )

        # Convert length
        derived_length = cmap(self._length.value)

        # Determine appropriate Timeline class for target domain
        from ..types import get_timeline_class

        target_domain = target.domain.name.lower()

        try:
            derived_class = get_timeline_class(
                target_domain, discrete=target.is_discrete
            )
        except ValueError:
            # Fallback to base Timeline if domain lookup fails
            derived_class = Timeline

        # Create derived timeline
        derived = derived_class(
            length=derived_length,
            unit=target,
            name=name or f"{self._id}_derived",
        )

        # Add inverse C-Map if available (for roundtrip conversion)
        if cmap.is_invertible:
            inverse = cmap.inverse()
            derived.add_conversion_map(inverse)
        else:
            self._logger.warning(
                f"C-Map '{cmap.id}' is not invertible. "
                f"The derived timeline will not have a C-Map back to '{self._unit}'."
            )

        # Copy and convert events if requested
        if copy_events:
            converted_events = []
            coordinate_values = {
                name: self._events.column_values(name)
                for name in ("instant", "start", "end", "duration")
            }
            for index, event in enumerate(self._events):
                # Skip segment events
                if event.get("event_type") == SEGMENT_EVENT_TYPE:
                    continue

                converted = dict(event)
                for coord_col in ("instant", "start", "end"):
                    val = coordinate_values[coord_col][index]
                    if val is not None:
                        converted[coord_col] = float(cmap(val))

                # Convert duration if present
                duration_val = coordinate_values["duration"][index]
                if duration_val is not None:
                    converted["duration"] = float(cmap(duration_val)) - float(cmap(0))

                converted_events.append(converted)

            if converted_events:
                derived.add_events(converted_events)

        self._logger.debug(
            f"Derived timeline '{derived.id}' in {target} from '{self._id}'"
        )

        return derived

    def _get_child_coordinate(
        self, child_id: str, parent_coord: CoordinateValue
    ) -> CoordinateValue | None:
        """Convert a parent coordinate to a child coordinate via exact offset arithmetic.

        ``child_coord = parent_coord - offset``

        Returns None if *parent_coord* falls outside the child's
        ``[offset, offset + length)`` span.

        Args:
            child_id: The child timeline ID.
            parent_coord: Coordinate on this (parent) timeline.

        Returns:
            Coordinate on the child timeline, or None if out of bounds.
        """
        path = self._descendant_offset_path(child_id)
        if path is None or not path[0]:
            return None
        offsets, child = path
        child_coord = parent_coord
        for offset in offsets:
            child_exact = exact_coordinate_value(child_coord)
            offset_exact = exact_coordinate_value(offset.value)
            child_coord = (
                child_exact - offset_exact
                if child_exact is not None and offset_exact is not None
                else float(coordinate_numeric_value(child_coord)) - float(offset.value)
            )
        child_length = child.length.value
        if child_coord < 0 or (child_length > 0 and child_coord >= child_length):
            return None
        return child_coord

    def _get_parent_coordinate_from_child(
        self, child_id: str, child_coord: CoordinateValue
    ) -> CoordinateValue:
        """Convert a child coordinate to a parent coordinate via exact offset arithmetic.

        ``parent_coord = child_coord + offset``

        Args:
            child_id: The child timeline ID.
            child_coord: Coordinate on the child timeline.

        Returns:
            Coordinate on this (parent) timeline.

        Raises:
            KeyError: If *child_id* is not a descendant of this timeline.
        """
        path = self._descendant_offset_path(child_id)
        if path is None or not path[0]:
            raise KeyError(child_id)
        parent_coord = child_coord
        for offset in reversed(path[0]):
            parent_exact = exact_coordinate_value(parent_coord)
            offset_exact = exact_coordinate_value(offset.value)
            parent_coord = (
                parent_exact + offset_exact
                if parent_exact is not None and offset_exact is not None
                else float(coordinate_numeric_value(parent_coord)) + float(offset.value)
            )
        return parent_coord

    def _get_unit_maps(self, unit: TimeUnit) -> list[ConversionMap[Any]]:
        """Get every map registered for a unit, in attachment order.

        A stamp or table shows all the readings an axis has, while scalar
        lookup must choose between them.

        Args:
            unit: Target unit.

        Returns:
            The attached maps targeting *unit*; empty when there are none.
        """
        return list(self._unit_maps.get(unit, ()))

    def _get_unit_map(
        self, unit: TimeUnit, *, name: str | None = None
    ) -> ConversionMap[Any] | None:
        """Get the one map for unit-based conversion, refusing to guess.

        Returns whichever ``ConversionMap`` was registered by
        :meth:`add_conversion_map` for this unit, called directly regardless
        of its concrete type (``TableMap``, ``ScalarMap``, ``LinearMap``,
        ...).  When several maps target the unit, an unnamed request is
        ambiguous and raises rather than picking by attachment order.

        This method is part of the TimeStampSource protocol.

        Args:
            unit: Target unit.
            name: Which of several same-unit maps to take, by id or by
                name.  Required only when there is more than one.

        Returns:
            A map for conversion, or None if no C-Map available.

        Raises:
            ValueError: If several maps target *unit* and no *name* is given.
            KeyError: If *name* matches no map targeting *unit*.
        """
        candidates = self._unit_maps.get(unit, ())
        if not candidates:
            return None
        if name is None:
            if len(candidates) > 1:
                listed = ", ".join(f"{cmap.id} ({cmap.name})" for cmap in candidates)
                raise ValueError(
                    f"Timeline '{self._id}' has {len(candidates)} conversion maps "
                    f"targeting '{unit}'; name the one you mean. Candidates: {listed}"
                )
            return candidates[0]
        for cmap in candidates:
            if cmap.id == name:
                return cmap
        for cmap in candidates:
            if cmap.name == name:
                return cmap
        raise KeyError(
            f"No conversion map named '{name}' targets '{unit}' on timeline "
            f"'{self._id}'. Available: {[cmap.id for cmap in candidates]}"
        )

    def _get_unit_map_for_timeline(
        self, timeline_id: str, unit: TimeUnit
    ) -> ConversionMap[Any] | None:
        """Get a unit C-Map attached to this timeline or any descendant.

        A timestamp is a cross-section, so a C-Map registered on a child
        (or deeper descendant) is reachable through the ancestor whose
        ``get_timestamp`` produced the stamp.
        """
        timeline = self._find_descendant(timeline_id)
        if timeline is None:
            return None
        return timeline._get_unit_map(unit)

    def _get_unit_maps_for_timeline(
        self, timeline_id: str, unit: TimeUnit
    ) -> list[ConversionMap[Any]]:
        """Get every unit C-Map attached to this timeline or any descendant."""
        timeline = self._find_descendant(timeline_id)
        if timeline is None:
            return []
        return timeline._get_unit_maps(unit)

    def _get_number_type_for_timeline(self, timeline_id: str) -> NumberType | None:
        """Get the numeric representation used by this timeline or a descendant."""
        timeline = self._find_descendant(timeline_id)
        return timeline.number_type if timeline is not None else None

    def _get_related_timeline_ids(self) -> list[str]:
        """Get IDs of the direct child timelines.

        This method is part of the TimeStampSource protocol. It reports only
        the immediate children (the rows a timestamp lists as its
        cross-section); deeper descendants are reached through
        :meth:`_get_descendant_timeline_ids`.

        Returns:
            List of direct child timeline IDs.
        """
        return list(self._children.keys())

    def _iter_descendants(self) -> "Iterator[tuple[str, Timeline]]":
        """Yield ``(id, timeline)`` for every descendant, depth-first."""
        for child_id, child in self._children.items():
            yield child_id, child
            yield from child._iter_descendants()

    def _find_descendant(self, timeline_id: str) -> "Timeline | None":
        """Return this timeline or the descendant with *timeline_id*, else None."""
        path = self._descendant_offset_path(timeline_id)
        return path[1] if path is not None else None

    def _descendant_offset_path(
        self, timeline_id: str
    ) -> tuple[list[Coordinate], Timeline] | None:
        """Find the offset path to a timeline in this subtree.

        Args:
            timeline_id: Timeline ID to find.

        Returns:
            The offsets from this timeline to the owner and the owner itself,
            or None when the ID is unknown.
        """
        if timeline_id == self._id:
            return [], self  # type: ignore[return-value]
        for child_id, child in self._children.items():
            child_path = child._descendant_offset_path(timeline_id)
            if child_path is not None:
                offsets, owner = child_path
                return [self._child_offsets[child_id], *offsets], owner
        return None

    def _get_descendant_timeline_ids(self) -> list[str]:
        """Get IDs of every descendant timeline (children, recursively).

        This method is part of the TimeStampSource protocol. Timestamps use
        it to surface conversions from the full subtree, not just the direct
        children reported by :meth:`_get_related_timeline_ids`.
        """
        return [descendant_id for descendant_id, _ in self._iter_descendants()]

    def _get_conversion_maps_for_timeline(
        self, timeline_id: str
    ) -> list[ConversionMap[Any]]:
        """Get every conversion map attached to a timeline in the subtree.

        This method is part of the TimeStampSource protocol. Unlike
        :meth:`_get_unit_map_for_timeline`, it returns C-Maps of *all* kinds
        -- including those with no ``target_unit`` (label and structured-value
        maps) -- so a timestamp exposes them as a full cross-section.
        """
        timeline = self._find_descendant(timeline_id)
        if timeline is None:
            return []
        return list(timeline._conversion_maps.values())

    def _get_available_units(self) -> list[TimeUnit]:
        """Get all target units available via C-Maps across the subtree.

        This method is part of the TimeStampSource protocol. It aggregates the
        target units of this timeline and every descendant so that unit-based
        conversions surface regardless of which level registered the C-Map.

        Returns:
            List of target units available for conversion.
        """
        units: list[TimeUnit] = list(self._unit_maps.keys())
        for _, descendant in self._iter_descendants():
            for unit in descendant._unit_maps.keys():
                if unit not in units:
                    units.append(unit)
        return units

    def _get_unit_for_timeline(self, timeline_id: str) -> TimeUnit | None:
        """Get the TimeUnit for a timeline anywhere in the subtree.

        This method is part of the TimeStampSource protocol. It enables
        TimeStamp to construct proper Coordinate objects with correct units.

        Args:
            timeline_id: The timeline ID to look up.

        Returns:
            The TimeUnit for the timeline, or None if not found.
        """
        timeline = self._find_descendant(timeline_id)
        return timeline._unit if timeline is not None else None

    def _contains_coordinate(
        self, timeline_id: str, axis: float, source_id: str | None = None
    ) -> bool:
        """Check whether *axis* falls within the span of a child timeline.

        A child embedded at *offset* with *length* spans
        ``[offset, offset + length)`` on this (parent) timeline.

        This method is part of the TimeStampSource protocol.

        Args:
            timeline_id: Child timeline ID.
            axis: Coordinate on the parent (source) timeline.
            source_id: Source timeline ID (ignored for Timeline, used by TimelineGroup).

        Returns:
            True if *axis* is inside the child's span, or if
            *timeline_id* is the source itself.
        """
        if timeline_id == self._id:
            return True
        return self._get_child_coordinate(timeline_id, axis) is not None

    def _convert_coordinate_to_self(self, value: CoordinateSpec) -> Coordinate:
        """Resolve one coordinate into this timeline's canonical axis.

        Args:
            value: Numeric, unit-qualified, or timeline-qualified coordinate.

        Returns:
            A coordinate expressed in this timeline's native unit.

        Raises:
            ValueError: If a timeline ID is unknown or no unit conversion path exists.
        """
        resolved = resolve_coordinate_spec(value)
        owner_id = resolved.timeline_id or self._id
        path = self._descendant_offset_path(owner_id)
        if path is None:
            raise ValueError(
                f"Timeline ID '{owner_id}' is not this timeline "
                f"'{self._id}' or one of its descendants"
            )
        offsets, owner = path
        native_value = resolved.value
        if resolved.unit is not None and resolved.unit != self._unit:
            candidates: list[tuple[int, Timeline]] = [(0, owner)]
            for _, candidate in self._iter_descendants():
                if candidate is owner:
                    continue
                candidate_path = candidate._descendant_offset_path(owner_id)
                if candidate_path is not None:
                    candidates.append((len(candidate_path[0]), candidate))
            if self is not owner:
                candidates.append((len(offsets), self))
            unit_map = None
            for _, candidate in sorted(candidates, key=lambda item: item[0]):
                unit_map = candidate._get_unit_map(resolved.unit)
                if unit_map is not None:
                    break
            if unit_map is None:
                raise ValueError(
                    f"No C-Map available to convert coordinate from unit "
                    f"'{resolved.unit}' to '{self._unit}' on timeline '{self._id}'"
                )
            try:
                inverse_map = unit_map.inverse()
            except (NotImplementedError, ValueError):
                raise ValueError(
                    f"No invertible C-Map available to convert coordinate from unit "
                    f"'{resolved.unit}' to '{self._unit}' on timeline '{self._id}'"
                ) from None
            native_value = inverse_map(resolved.value)

        for offset in reversed(offsets):
            native_exact = exact_coordinate_value(native_value)
            offset_exact = exact_coordinate_value(offset.value)
            native_value = (
                native_exact + offset_exact
                if native_exact is not None and offset_exact is not None
                else float(coordinate_numeric_value(native_value)) + float(offset.value)
            )

        # Expressed on THIS timeline's axis, in the type it declares -- which
        # may differ from its unit's default when the caller chose otherwise.
        return Coordinate(
            express_as(native_value, self._number_type),
            self._unit,
            number_type=self._number_type,
        )

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Resolve one position onto this timeline's canonical axis.

        Args:
            at: Coordinate position to resolve.
            timeline_id: Optional result-axis validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One coordinate projection or a length-one Series.

        Raises:
            KeyError: If the result or embedded source timeline is unknown.
            ValueError: If a unit has no unique invertible conversion path.
            TypeError: If ``at`` is not a scalar coordinate input.
        """
        if timeline_id is not None and timeline_id != self._id:
            raise KeyError(
                f"Unknown result timeline ID {timeline_id!r} on timeline {self._id!r}"
            )
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        try:
            coordinate = self._convert_coordinate_to_self(at)
        except ValueError as exc:
            if isinstance(at, IdCoordinate) and "Timeline ID" in str(exc):
                raise KeyError(
                    f"Unknown source timeline ID {at.timeline_id!r} on "
                    f"timeline {self._id!r}"
                ) from None
            raise
        identified = IdCoordinate.from_coordinate(coordinate, self._id)
        return format_coordinates(
            [identified],
            format=format,
            rounding=rounding,
            scalar=True,
            series_name=self._id,
        )

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve a collection of positions onto this timeline.

        Args:
            at: Coordinate positions to resolve.
            timeline_id: Optional result-axis validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        coordinates: list[IdCoordinate] = []
        for value in values:
            result = self.get_coordinate_at(
                value,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            coordinates.append(result)
        return format_coordinates(
            coordinates,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name=self._id,
            empty_number_type=self._number_type,
        )

    def get_coordinate_for(
        self,
        key: str,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return an event's start coordinate on this timeline.

        Args:
            key: Event ID to find recursively.
            timeline_id: Optional result-axis validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The event-start coordinate projection.
        """
        if not isinstance(key, str):
            raise TypeError("get_coordinate_for requires an event-ID string")
        if timeline_id is not None and timeline_id != self._id:
            raise KeyError(
                f"Unknown result timeline ID {timeline_id!r} on timeline {self._id!r}"
            )
        stamp = self.get_timestamp_for(key)
        if hasattr(stamp, "get_interval"):
            coordinate = stamp.get_interval(self._id).start
        else:
            coordinate = stamp.get_coordinate_for(self._id, format="coordinate")
        assert isinstance(coordinate, Coordinate)
        return format_coordinates(
            [IdCoordinate.from_coordinate(coordinate, self._id)],
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
        """Return event-start coordinates for a collection of event IDs.

        Args:
            keys: Event IDs to retrieve.
            timeline_id: Optional result-axis validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        key_values, index = validate_key_collection(keys)
        coordinates: list[IdCoordinate] = []
        for key in key_values:
            result = self.get_coordinate_for(
                key,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            coordinates.append(result)
        if format == "series" and index is None:
            index = pd.Index(key_values)
        return format_coordinates(
            coordinates,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name="coordinate",
            empty_number_type=self._number_type,
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection | str | KeyCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a positional or event-key coordinate query.

        Args:
            at: Scalar or plural coordinate position or event key.
            timeline_id: Optional result-axis validator.
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

    def _resolve_axis_value(self, coord: CoordinateSpec) -> int | float | Fraction:
        """Resolve a coordinate and return its native numeric value."""
        return self._convert_coordinate_to_self(coord).value
