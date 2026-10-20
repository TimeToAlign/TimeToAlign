"""Timeline: The central object of TimeToAlign!

A Timeline is a positive coordinate axis defined by an origin (zero) and a
measuring unit. It can hold events (via an EventData) and nested child
timelines (segments) that share the same coordinate type.

Design principles:
- Events stored in PyArrow-based EventData (no flyweight pattern)
- Children stored as direct object references AND as segment events
- Unit validation at add-time prevents coordinate type mismatches
- A locked timeline's length is fixed; an embedded child's length changes
  only within its parent's known length
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal

from typing_extensions import Self

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    Domain,
    NumberType,
    TimeUnit,
)
from timetoalign.core.ids import (
    _admit_id,
    _OpenObject,
    _plan_admissions,
    mint_id,
    timeline_id_prefix,
    validate_id,
)
from timetoalign.core.time import (
    express_as,
    express_scalar_as,
    number_from_json,
    number_to_json,
    struct_to_rational,
)
from timetoalign.core.timestamp import _format_coordinate_value
from timetoalign.maps import (
    ChainMap,
    ConversionMap,
    LinearMap,
    MeasureRecordMap,
    ScalarMap,
)
from timetoalign.storage import EventData

from .engines import (
    ChildrenMixin,
    ConversionMapsMixin,
    EventsMixin,
    ExternalReferencesMixin,
    RegionsMixin,
    SegmentsMixin,
    SkeletonsMixin,
    TabularExportMixin,
)
from .engines import children as _children_engine
from .engines import conversion as _conversion_engine
from .engines import (
    empty_external_reference_table,
)
from .flowcontrol import FlowControlRegistry
from .manifest import (
    Manifest,
    axis_value_from_json,
    axis_value_to_json,
    check_children_fit,
    listed,
    matches_any,
    read_id,
    read_list,
    read_meta,
    read_name,
    require_object,
    select,
)
from .regions import Region
from .structure import MeasureMap, MetricHierarchy

if TYPE_CHECKING:
    from timetoalign.display.ascii import Diagram

    from .manifest import MeasureMapEntries

module_logger = logging.getLogger(__name__)

# region Constants

# Event type name for segment events in the EventData
SEGMENT_EVENT_TYPE = "Segment"

# Traversal order options for iterating children
TraversalOrder = Literal["sorted", "depth_first", "breadth_first"]

# endregion


def _is_record_map_on(cmap: ConversionMap[Any], unit: TimeUnit) -> bool:
    """Whether *cmap* reads a timeline's measure map on an axis of *unit*.

    It does as a ``MeasureRecordMap`` on a quarters axis, or as a
    ``ChainMap`` of a ``LinearMap`` from *unit* to quarters and a
    ``MeasureRecordMap``.
    """
    if isinstance(cmap, MeasureRecordMap):
        return unit is TimeUnit.quarters
    if not isinstance(cmap, ChainMap) or len(cmap.maps) != 2:
        return False
    first, last = cmap.maps
    return (
        isinstance(last, MeasureRecordMap)
        and isinstance(first, (LinearMap, ScalarMap))
        and first.source_unit is unit
        and first.target_unit is TimeUnit.quarters
    )


def _refuse_negative_length(length: Coordinate, timeline_id: str) -> Coordinate:
    """Return *length*, raising ``ValueError`` if it is negative."""
    if length.value < 0:
        raise ValueError(
            f"length {_format_coordinate_value(length.value)} of '{timeline_id}' "
            "is negative"
        )
    return length


class UnknownLengthError(ValueError):
    """An operation needs the end of a timeline whose length is unknown.

    A length of ``None`` states that the length is not known. Operations that
    do not need the end work on such a timeline; an operation that needs it
    raises this error before it changes anything.

    Attributes:
        timeline_id: The id of the timeline whose length is unknown.
        operation: What could not be done without the end.
    """

    def __init__(self, timeline_id: str, operation: str) -> None:
        """Name the timeline and the operation that needs its end.

        Args:
            timeline_id: The id of the timeline whose length is unknown.
            operation: What could not be done, as a verb phrase.
        """
        self.timeline_id = timeline_id
        self.operation = operation
        super().__init__(
            f"Cannot {operation}: the length of timeline '{timeline_id}' is unknown"
        )


class Timeline(
    EventsMixin,
    ExternalReferencesMixin,
    ChildrenMixin,
    ConversionMapsMixin,
    TabularExportMixin,
    RegionsMixin,
    SegmentsMixin,
    SkeletonsMixin,
    _OpenObject,
):
    """A positive coordinate axis with events and nested child timelines.

    A Timeline represents a temporal dimension in one of three domains
    (Logical, Physical, Graphical) with either continuous or discrete
    coordinates. It stores events in an EventData and can contain
    nested child timelines (segments) at specified offsets.

    **Intended usage:** This base class provides the full Timeline API but
    does not enforce domain or modality constraints. For typical usage,
    prefer one of the six concrete subclasses or the ``create_timeline()``
    factory function:

    - ``ContinuousLogicalTimeline`` -- beats, quarters, measures (Fraction)
    - ``DiscreteLogicalTimeline`` -- ticks (int)
    - ``ContinuousPhysicalTimeline`` -- seconds, ms, minutes (float)
    - ``DiscretePhysicalTimeline`` -- samples, frames (int)
    - ``ContinuousGraphicalTimeline`` -- cm, inches, points (float)
    - ``DiscreteGraphicalTimeline`` -- pixels (int)

    These subclasses restrict allowed units and number types to prevent
    accidental cross-domain errors and provide sensible defaults.

    Direct instantiation of ``Timeline`` is appropriate for internal use,
    generic algorithms that operate across domains, or advanced scenarios
    where domain constraints are intentionally relaxed.

    If you have a ``Timeline`` instance and need the appropriate typed
    subclass, use :meth:`to_typed`.

    Attributes:
        id: Unique identifier for this timeline.
        unit: The time unit for coordinates (e.g., seconds, quarters, pixels).
        number_type: The numeric type for coordinates (int, float, Fraction).
        domain: The temporal domain (derived from unit).
        origin: The start coordinate (always 0).
        length: The end coordinate, or ``None`` when the length is unknown.
        is_locked: Whether the timeline's length is fixed.
        is_discrete: Whether the timeline uses discrete coordinates.
        is_continuous: Whether the timeline uses continuous coordinates.

    Examples:
        >>> # Preferred: use concrete subclasses
        >>> from timetoalign.timelines import ContinuousPhysicalTimeline
        >>> audio = ContinuousPhysicalTimeline(length=180.0)

        >>> # Or use the factory to auto-select the right subclass
        >>> from timetoalign.timelines import create_timeline
        >>> tl = create_timeline(loader)

        >>> # Direct base class (internal/advanced use)
        >>> from timetoalign.core import TimeUnit
        >>> tl = Timeline(length=100, unit=TimeUnit.seconds)
    """

    # region Class Variables

    # Subclasses can override this to restrict which units they accept.
    # Which NUMBER types a timeline accepts is not a property of the
    # timeline at all -- it follows from the unit (see
    # ``TimeUnit.allowed_number_types``), so there is nothing to declare.
    _allowed_units: ClassVar[frozenset[TimeUnit] | None] = None

    # Default unit (subclasses override)
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds

    # EventData class to use (subclasses can override for domain-specific data)
    _event_data_class: ClassVar[type[EventData]] = EventData

    # The keys of a timeline object, in the order the dictionary form writes
    # them; ``events`` and ``external_references`` only when requested.
    _dict_keys: ClassVar[tuple[str, ...]] = (
        "id",
        "name",
        "unit",
        "number_type",
        "offset",
        "length",
        "locked",
        "is_segment_line",
        "meta",
        "children",
        "conversion_maps",
        "regions",
        "flow_control",
        "events",
        "external_references",
    )
    # A timeline object's optional keys and their values when absent; a
    # number_type left out is the unit's own.
    _dict_defaults: ClassVar[Mapping[str, Any]] = {
        "name": None,
        "locked": False,
        "is_segment_line": False,
        "meta": {},
        "children": [],
        "conversion_maps": [],
        "regions": [],
        "flow_control": [],
    }

    # Where a root's origin lies on its unit's absolute axis (a manifest
    # root's ``offset``), or None. Stored and written back only.
    _root_offset: Coordinate | None = None

    # The manifest this timeline was read from as a whole, written back
    # around it; None for a timeline not read from a manifest.
    _manifest: Manifest | None = None

    @classmethod
    def _from_dict_initial_length(
        cls, length: Coordinate | None
    ) -> CoordinateValue | None:
        """Return the length to construct with while rebuilding a timeline."""
        return None if length is None else length.value

    def _finalize_from_dict(self, length: Coordinate | None) -> None:
        """Restore state that must be set after children are added."""

    # endregion

    # region Initialization

    def __init__(
        self,
        length: CoordinateValue | None = 0,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        uid: str | None = None,
        name: str | None = None,
        locked: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Timeline.

        Args:
            length: The length (end coordinate) of the timeline. Default 0.
                ``None`` states that the length is unknown; positions on such
                a timeline are not bounded above.
            unit: The time unit for coordinates. Defaults to class default.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.
            uid: Explicit identifier, kept verbatim. When None, the id is
                minted from the unit (``clt_1``, ``dpt_1``, ...).
            name: Human-readable name for display (distinct from uid).
            locked: If True, the length is fixed, known or not, and the
                timeline cannot expand. Default False.
            meta: Additional metadata dictionary.

        Raises:
            ValueError: If unit or number_type is not allowed for this class,
                or if *uid* is not a valid id.
            TypeError: If length is not a valid coordinate value.
        """
        # Resolve unit
        if unit is None:
            unit = self._default_unit
        elif isinstance(unit, str):
            unit = TimeUnit(unit)
        # Validate against the concrete runtime class so each leaf type's unit
        # contract is enforced by the shared base constructor.
        type(self)._validate_unit(unit)
        self._unit = unit

        # Resolve number_type -- the unit decides what it can be.
        self._number_type = unit.resolve_number_type(number_type)

        # A given id is kept; otherwise one is minted in an empty scope and
        # re-minted if it collides when this timeline joins a manifest.
        self._id_minted = uid is None
        self._id = (
            mint_id(timeline_id_prefix(unit), ()) if uid is None else validate_id(uid)
        )
        self._parent: Timeline | None = None

        self._length = self._make_length(length)

        # State
        self._locked = locked
        self._name = name
        self._meta = dict(meta) if meta else {}

        # Event storage
        self._events = self._event_data_class.empty(self._unit, self._number_type)

        # Incoming external references (annotations pointing at own events)
        self._external_references = empty_external_reference_table()

        # Child timeline storage
        self._children: dict[str, Timeline] = {}
        self._child_offsets: dict[str, Coordinate] = {}

        # Conversion maps
        self._conversion_maps: dict[str, ConversionMap[Any]] = {}

        # Maps TimeUnit -> the ConversionMap registered for that unit via
        # add_conversion_map(), so that any C-Map with a target_unit is
        # available in the TimeStamp system.
        self._unit_maps: dict[TimeUnit, ConversionMap[Any]] = {}

        # Region storage (named TimeIntervals)
        # A Region is a named part of a timeline defined by a TimeInterval.
        self._regions: dict[str, Region] = {}

        # Authored structure and control events belong to this exact extent.
        # The measure map is held by the conversion map reading it on this
        # axis: a MeasureRecordMap, or a ChainMap of a LinearMap to quarters
        # and a MeasureRecordMap.
        self._measure_record_map: ConversionMap[Any] | None = None
        self._metric_hierarchy: MetricHierarchy | None = None
        self._adopt_flow_control(
            FlowControlRegistry(unit=self._unit, number_type=self._number_type)
        )

        # FlowMap storage (coordinate transformations for flow control)
        # Timelines store FlowMaps (not FlowControllers).
        # FlowMaps enable unfold/fold coordinate conversion for timelines
        # that have flow control (repeats, jumps, etc.).
        self._flow_maps = {}

        # Shared temporal structures enroll timelines through TimeSkeleton.
        self._skeleton_attachments: list[Any] = []

        # Logger
        self._logger = module_logger.getChild(self._id)

    # endregion

    # region Class Methods - Validation

    @classmethod
    def _validate_unit(cls, unit: TimeUnit) -> None:
        """Validate that unit is allowed for this Timeline class.

        Args:
            unit: The unit to validate.

        Raises:
            ValueError: If unit is not in _allowed_units.
        """
        if cls._allowed_units is not None and unit not in cls._allowed_units:
            allowed = ", ".join(sorted(str(u) for u in cls._allowed_units))
            raise ValueError(
                f"{cls.__name__} does not allow unit '{unit}'. "
                f"Allowed units: {allowed}"
            )

    # endregion

    # region Class Methods - Construction

    @classmethod
    def resolve_subclass(
        cls,
        unit: TimeUnit | str,
        number_type: NumberType | str | None = None,
    ) -> type[Timeline]:
        """Return the most specific Timeline subclass for a unit/number_type pair.

        Inspects all subclasses and selects the one whose
        ``_allowed_units`` includes *unit*.  Among candidates the selection
        prefers, in order:

        1. A class whose default unit uses *number_type* (when supplied).
        2. The class with the **smallest** ``_allowed_units`` set (most
           specific domain).

        Further-derived subclasses participate in the lookup and can be
        selected when they are the most specific matching candidate.

        Falls back to the base ``Timeline`` if no subclass claims the unit.

        Args:
            unit: The time unit to look up.
            number_type: Optional number type for disambiguation
                (e.g. ``NumberType.fraction`` selects a fractional logical
                timeline over ``DiscreteLogicalTimeline``).

        Returns:
            The most specific Timeline subclass that accepts *unit*.

        Examples:
            >>> Timeline.resolve_subclass(TimeUnit.quarters, NumberType.fraction)
            <class 'timetoalign.timelines.types.ContinuousLogicalTimeline'>
            >>> Timeline.resolve_subclass(TimeUnit.pixels)
            <class 'timetoalign.timelines.types.DiscreteGraphicalTimeline'>
        """
        if isinstance(unit, str):
            unit = TimeUnit(unit)
        if isinstance(number_type, str):
            number_type = NumberType(number_type)

        def _all_subclasses(base: type[Timeline]) -> Iterator[type[Timeline]]:
            """Yield *base* and all its descendants (breadth-first)."""
            for sub in base.__subclasses__():
                yield sub
                yield from _all_subclasses(sub)

        candidates: list[type[Timeline]] = []
        for sub in _all_subclasses(Timeline):
            allowed = getattr(sub, "_allowed_units", None)
            if allowed is not None and unit in allowed:
                candidates.append(sub)

        if not candidates:
            return Timeline

        # Sort by allowed-units size (smallest = most specific domain)
        candidates.sort(key=lambda c: len(getattr(c, "_allowed_units", frozenset())))

        # If number_type is given, prefer a candidate whose own default unit
        # counts that way -- a fractional request lands on the timeline
        # whose default unit is fractional.
        if number_type is not None:
            for cand in candidates:
                default_unit = getattr(cand, "_default_unit", None)
                if (
                    default_unit is not None
                    and default_unit.default_number_type == number_type
                ):
                    return cand

        # Return the most specific candidate (smallest allowed-units set)
        return candidates[0]

    @classmethod
    def empty(
        cls,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create an empty Timeline with length 0.

        Args:
            unit: The time unit. Defaults to class default.
            number_type: The number type. Defaults to class default.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            A new empty Timeline.
        """
        return cls(length=0, unit=unit, number_type=number_type, **kwargs)

    @classmethod
    def from_events(
        cls,
        rows: list[dict[str, Any]],
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Create a Timeline from event dictionaries.

        The timeline length is automatically set to accommodate all events.

        Args:
            rows: List of event dictionaries with keys:
                - id: unique identifier
                - temporal_type: "instant" or "interval"
                - event_type: class name (e.g., "Note", "Beat")
                - instant: coordinate (for instant events)
                - start, end: coordinates (for interval events)
            unit: The time unit. Defaults to class default.
            number_type: The number type. Defaults to class default.
            **kwargs: Additional arguments passed to __init__.

        Returns:
            A new Timeline containing the events.
        """
        if not rows:
            return cls.empty(unit=unit, number_type=number_type, **kwargs)

        # Calculate max coordinate to determine length
        max_coord: Any = 0
        for row in rows:
            if row.get("instant") is not None:
                max_coord = max(max_coord, cls._coord_to_value(row["instant"]))
            if row.get("end") is not None:
                max_coord = max(max_coord, cls._coord_to_value(row["end"]))
            elif row.get("start") is not None and row.get("duration") is not None:
                start = cls._coord_to_value(row["start"])
                duration = cls._coord_to_value(row["duration"])
                max_coord = max(max_coord, start + duration)

        timeline = cls(length=max_coord, unit=unit, number_type=number_type, **kwargs)
        timeline._add_events_unchecked(rows)
        return timeline

    @classmethod
    def from_event_data(
        cls,
        data: EventData,
        **kwargs: Any,
    ) -> Self:
        """Create a Timeline from an existing EventData.

        Args:
            data: The EventData containing events.
            **kwargs: Additional arguments passed to __init__ (except unit/number_type).

        Returns:
            A new Timeline wrapping the EventData.
        """
        coord_range = data.coordinate_range()
        length = coord_range[1] if coord_range else 0

        timeline = cls(
            length=length,
            unit=data.unit,
            number_type=data.number_type,
            **kwargs,
        )
        timeline._events = data
        return timeline

    # endregion

    # region Properties - Identity

    @property
    def id(self) -> str:
        """Unique identifier for this timeline."""
        return self._id

    @property
    def name(self) -> str:
        """Human-readable name for display.

        Returns the explicit name if set, otherwise falls back to the ID.
        Use this for user-facing displays (e.g., timestamp field headers).
        """
        return self._name if self._name is not None else self._id

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the human-readable name."""
        self._name = value

    @property
    def class_name(self) -> str:
        """The class name of this timeline."""
        return self.__class__.__name__

    def _is_segment_line_instance(self) -> bool:
        """Whether this timeline is a SegmentLine (the dictionary's flag)."""
        return False

    # endregion

    # region Properties - Coordinate Type

    @property
    def unit(self) -> TimeUnit:
        """The time unit for coordinates."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The number type for coordinates."""
        return self._number_type

    @property
    def domain(self) -> Domain:
        """The temporal domain (derived from unit)."""
        return self._unit.domain

    @property
    def is_discrete(self) -> bool:
        """Whether this timeline uses discrete (integer) coordinates.

        Derived from the unit's inherent discreteness. Discrete timelines
        measure time in countable units (ticks, samples, frames, pixels).

        Returns:
            True if the timeline's unit is inherently discrete.

        Examples:
            >>> DiscreteLogicalTimeline(length=1920).is_discrete
            True
            >>> ContinuousPhysicalTimeline(length=10.0).is_discrete
            False
        """
        return self._unit.is_discrete

    @property
    def is_continuous(self) -> bool:
        """Whether this timeline uses continuous (real-valued) coordinates.

        The logical complement of :attr:`is_discrete`. Continuous timelines
        measure time in units that allow arbitrary precision (seconds,
        quarters, centimeters, etc.).

        Returns:
            True if the timeline's unit is not inherently discrete.

        Examples:
            >>> ContinuousLogicalTimeline(length=Fraction(4, 1)).is_continuous
            True
            >>> DiscreteGraphicalTimeline(length=1920).is_continuous
            False
        """
        return not self._unit.is_discrete

    # endregion

    # region Properties - Boundaries

    @property
    def origin(self) -> Coordinate:
        """The start coordinate (always 0)."""
        return self._make_coordinate(0)

    @property
    def length(self) -> Coordinate | None:
        """The end coordinate (length of the timeline), ``None`` when unknown."""
        return self._length

    @length.setter
    def length(self, value: CoordinateSpec | None) -> None:
        """Set the timeline length, or ``None`` to state that it is unknown.

        Args:
            value: New length value, or ``None``.

        A child's new end must lie within its parent's known length, which
        never grows with it; a segment must keep ending at its successor's
        offset, the last one at the line's known length.

        Raises:
            RuntimeError: If timeline is locked.
            ValueError: If new length is less than current content, or the
                parent cannot contain the new end.
        """
        self._check_not_locked("set length")

        if value is None:
            self._set_length(None)
            return

        new_length = self.get_coordinate_at(value, format="coordinate")

        # Check that new length accommodates all content
        max_content = self._get_max_content_coordinate()
        if new_length.value < max_content:
            raise ValueError(
                f"Cannot reduce length to "
                f"{_format_coordinate_value(float(new_length.value))}: "
                f"content extends to {_format_coordinate_value(float(max_content))}"
            )

        self._set_length(new_length)

    @property
    def is_length_known(self) -> bool:
        """Whether this timeline's length is known (not ``None``)."""
        return self._length is not None

    @property
    def start(self) -> Coordinate:
        """Alias for origin."""
        return self.origin

    @property
    def end(self) -> Coordinate | None:
        """Alias for length: ``None`` when the length is unknown."""
        return self._length

    @property
    def reach(self) -> Coordinate:
        """The farthest coordinate on this axis that the content reaches.

        Content means the children's ends, region ends, break and marker
        coordinates, both ends of every jump, and the end of the measure map.
        A child of unknown length contributes its offset plus its own reach.
        Events are not part of it. A timeline without such content reaches
        ``0``. The reach is independent of the length, known or not; displays
        use it as the extent of a timeline whose length is unknown.

        Examples:
            >>> tl = ContinuousLogicalTimeline(length=None)
            >>> tl.create_region(2, 6).end.value
            Fraction(6, 1)
            >>> tl.reach
            Coordinate(Fraction(6, 1), quarters)
        """
        candidates: list[CoordinateValue] = [0]
        for child_id, child in self._children.items():
            offset = self._child_offsets[child_id].value
            extent = child._extent
            candidates.append(
                offset + (child.reach.value if extent is None else extent)
            )
        for region in self._regions.values():
            candidates.append(region.end.value)
        registry = self._flow_control
        candidates.extend(item.coordinate.value for item in registry.breaks)
        candidates.extend(item.coordinate.value for item in registry.markers)
        for jump in registry.jumps:
            candidates.extend((jump.from_coordinate.value, jump.to_coordinate.value))
        measure_map = self.measure_map
        if measure_map is not None:
            for measure in measure_map:
                if measure.qstamp is None:
                    continue
                candidates.append(
                    self._from_quarters(
                        measure.qstamp
                        if measure.actual_length is None
                        else measure.qstamp + measure.actual_length
                    )
                )
        return self._make_coordinate(express_as(max(candidates), self._number_type))

    # endregion

    # region Properties - State

    @property
    def is_locked(self) -> bool:
        """Whether this timeline's length is fixed, known or not."""
        return self._locked

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata dictionary."""
        return dict(self._meta)

    @property
    def root_offset(self) -> Coordinate | None:
        """Where a root's origin lies on its unit's absolute axis, or None.

        A manifest root may carry an ``offset``, negative included: a
        recording whose origin is a point in absolute time, for example.
        :meth:`from_dict` keeps it here and :meth:`to_dict` writes it back;
        no coordinate, conversion, child placement or containment reads it.
        """
        return self._root_offset

    @property
    def measure_map(self) -> MeasureMap | None:
        """Authored measure structure, absent on fresh and derived slices.

        A measure map's positions are quarters; on an axis in other units
        the timeline reaches it through a ``LinearMap`` to quarters (see
        :meth:`add_measure_map`).
        """
        record = self._measure_record_map
        if record is None:
            return None
        if isinstance(record, ChainMap):
            record = record.maps[-1]
        return record.measure_map

    @property
    def metric_hierarchy(self) -> MetricHierarchy | None:
        """Authored metric structure, absent on fresh and derived slices."""
        return self._metric_hierarchy

    @property
    def flow_control(self) -> FlowControlRegistry:
        """Axis-bound breaks, jumps, and markers."""
        return self._flow_control

    # endregion

    # region Manifest Identity

    def _adopt_flow_control(self, registry: FlowControlRegistry) -> None:
        """Make *registry* this timeline's, admitting its ids into the manifest."""
        self._flow_control = registry
        registry._admit = self._admit_element_id

    def _set_id(self, new_id: str, *, minted: bool = False) -> None:
        """Give this timeline a new id, renaming it under its parent."""
        validate_id(new_id)
        if self._parent is not None and new_id != self._id:
            self._parent._rename_child(self._id, new_id)
        self._id = new_id
        self._id_minted = minted

    def _identity_entries(self) -> list[tuple[str, bool, str, str]]:
        """Return ``(id, minted, prefix, description)`` for this timeline and
        every element it holds itself (its children excluded), in the order
        timeline, conversion maps (each followed by the maps nested in it),
        regions, breaks, markers, jumps."""
        owner = f"of timeline {self._id!r}"
        return [
            (
                self._id,
                self._id_minted,
                timeline_id_prefix(self._unit),
                f"timeline {self._id!r}",
            ),
            *(
                entry
                for cmap in self._all_conversion_maps()
                for entry in cmap._identity_entries(f" {owner}")
            ),
            *(
                (
                    element.id,
                    element._minted,
                    element._id_prefix,
                    f"{element._id_prefix} {element.id!r} {owner}",
                )
                for element in (
                    *self._regions.values(),
                    *self._flow_control._elements(),
                )
            ),
        ]

    def _reassign_element_id(self, old_id: str, new_id: str) -> None:
        """Store the element of this timeline with *old_id* under a minted id."""
        held_map = next(
            (
                nested
                for cmap in self._all_conversion_maps()
                for nested in cmap._map_tree()
                if nested.id == old_id
            ),
            None,
        )
        if old_id == self._id:
            self._set_id(new_id, minted=True)
        elif old_id in self._conversion_maps:
            self._rename_conversion_map(old_id, new_id)
        elif held_map is not None:
            held_map._id = new_id
        elif old_id in self._regions:
            self._rename_region(old_id, new_id)
        else:
            self._flow_control._reassign_id(old_id, new_id)

    def _manifest_ids(self) -> dict[str, str]:
        """Map every id of this timeline's manifest to its holder's description.

        The manifest is the root timeline above this one and all its
        descendants; ids are unique within it across all element kinds.
        """
        root = self
        while root._parent is not None:
            root = root._parent
        return {
            element_id: description
            for timeline in (root, *(child for _, child in root._iter_descendants()))
            for element_id, _minted, _prefix, description in timeline._identity_entries()
        }

    def _admit_element_id(
        self, element_id: str, *, minted: bool, prefix: str, element: str
    ) -> str:
        """Return the id under which an element joins this timeline's manifest.

        Raises:
            ValueError: If the element's given id is already taken.
        """
        return _admit_id(
            element_id,
            minted=minted,
            prefix=prefix,
            taken=self._manifest_ids(),
            element=f"{element} of timeline {self._id!r}",
        )

    def _admit_map(self, cmap: ConversionMap[Any]) -> None:
        """Admit *cmap* and every map nested in it into this timeline's manifest.

        Raises:
            ValueError: If a given id of the map's tree is already held in the
                manifest or in the tree, or the tree nests one map twice.
        """
        cmap._admit_ids(self._manifest_ids(), owner=f" of timeline {self._id!r}")

    def _manifest_join_plan(self, child: Timeline) -> list[tuple[Timeline, str, str]]:
        """Plan how *child*'s whole subtree joins this timeline's manifest.

        A minted id that the manifest holds is minted again; the returned
        plan lists ``(owner, old_id, new_id)`` for each. Nothing is changed.

        Raises:
            ValueError: If an id given to an element of the subtree is
                already held in this manifest, naming both elements.
        """
        joining = [
            (timeline, entry)
            for timeline in (child, *(t for _, t in child._iter_descendants()))
            for entry in timeline._identity_entries()
        ]
        plan = _plan_admissions([entry for _, entry in joining], self._manifest_ids())
        return [
            (joining[index][0], joining[index][1][0], new_id)
            for index, new_id in plan.items()
        ]

    # endregion

    # region Authored Structure

    def add_measure_map(
        self, measure_map: MeasureMap, *, to_quarters: LinearMap | None = None
    ) -> Self:
        """Attach the timeline's single authored measure map.

        A measure map's positions are quarters. On a quarters axis the map is
        read as it is; on a ``whole_notes`` axis through the ``LinearMap``
        that multiplies by 4, unless *to_quarters* gives another; on a ticks
        axis through *to_quarters*, which says how many ticks a quarter
        holds. The timeline holds the map in a :class:`MeasureRecordMap`,
        chained after that ``LinearMap`` on an axis other than quarters,
        which its manifest writes among the conversion maps.

        Args:
            measure_map: Immutable measure structure to attach.
            to_quarters: The ``LinearMap`` from this timeline's unit to
                quarters, on an axis in whole notes or ticks.

        Returns:
            This timeline after attachment.

        Raises:
            ValueError: If the timeline already carries a measure map, its
                unit is none of quarters, whole notes and ticks, a ticks
                timeline is given no *to_quarters*, or *to_quarters* is given
                on a quarters axis or does not lead from this unit to
                quarters.
        """
        if self._measure_record_map is not None:
            raise ValueError(f"Timeline {self._id!r} already has a measure map")
        record = MeasureRecordMap(measure_map)
        if self._unit is TimeUnit.quarters:
            if to_quarters is not None:
                raise ValueError(
                    f"Timeline {self._id!r} is on the quarter-note axis: its measure "
                    "map needs no LinearMap to quarters"
                )
            self._set_measure_record_map(record)
            return self
        if to_quarters is None and self._unit is TimeUnit.whole_notes:
            to_quarters = LinearMap(
                scalar=4,
                source_unit=TimeUnit.whole_notes,
                target_unit=TimeUnit.quarters,
            )
        if self._unit not in (TimeUnit.whole_notes, TimeUnit.ticks):
            raise ValueError(
                f"Timeline unit {self._unit.value!r} does not reach the quarter notes "
                "a measure map is anchored on: give it a quarters, whole_notes or "
                "ticks timeline"
            )
        if to_quarters is None:
            raise ValueError(
                f"Timeline {self._id!r} is in ticks: give the LinearMap from ticks to "
                "quarters (to_quarters=) its measure map is read through"
            )
        self._set_measure_record_map(ChainMap([to_quarters, record]))
        return self

    def _set_measure_record_map(self, cmap: ConversionMap[Any]) -> None:
        """Hold *cmap* as the map reading this timeline's measure map.

        Raises:
            ValueError: If *cmap* is neither a ``MeasureRecordMap`` on a
                quarters axis nor a ``ChainMap`` of a ``LinearMap`` from this
                timeline's unit to quarters and a ``MeasureRecordMap``, or its
                given id is already held in the manifest.
        """
        if not _is_record_map_on(cmap, self._unit):
            raise ValueError(
                f"Timeline {self._id!r} reads a measure map through a "
                "MeasureRecordMap on quarters, or a ChainMap of a LinearMap from "
                f"{self._unit.value!r} to quarters and a MeasureRecordMap"
            )
        self._admit_map(cmap)
        self._measure_record_map = cmap

    def _from_quarters(self, value: CoordinateValue) -> CoordinateValue:
        """Return a position of the measure map, in quarters, on this axis."""
        record = self._measure_record_map
        if not isinstance(record, ChainMap):
            return value
        to_quarters = record.maps[0]
        offset = getattr(to_quarters, "offset", 0)
        return (Fraction(value) - offset) / Fraction(to_quarters.scalar)

    def _all_conversion_maps(self) -> list[ConversionMap[Any]]:
        """Every conversion map this timeline holds, its measure map's included."""
        maps = list(self._conversion_maps.values())
        if self._measure_record_map is not None:
            maps.append(self._measure_record_map)
        return maps

    def add_metric_hierarchy(self, metric_hierarchy: MetricHierarchy) -> Self:
        """Attach the timeline's single measure-anchored metric hierarchy.

        Args:
            metric_hierarchy: Hierarchy anchored on this timeline's measure map.

        Returns:
            This timeline after attachment.

        Raises:
            ValueError: If the timeline has no matching measure map, is not on
                the quarter-note axis, or already carries a hierarchy.
        """
        if self._metric_hierarchy is not None:
            raise ValueError(f"Timeline {self._id!r} already has a metric hierarchy")
        if self.measure_map is None:
            raise ValueError("A metric hierarchy requires the timeline's measure map")
        if metric_hierarchy.measure_map != self.measure_map:
            raise ValueError(
                "The metric hierarchy is anchored on a different measure map"
            )
        self._require_quarters_axis("metric hierarchy")
        if isinstance(self._measure_record_map, ChainMap):
            raise ValueError(
                f"Timeline {self._id!r} reads its measure map through a LinearMap: "
                "a metric hierarchy is anchored on a measure map read as it is"
            )
        self._metric_hierarchy = metric_hierarchy
        return self

    def _require_quarters_axis(self, structure: str) -> None:
        """Refuse quarter-anchored structure on any other axis."""
        if self.unit is not TimeUnit.quarters:
            raise ValueError(
                f"Timeline unit {self.unit.value!r} does not match {structure} "
                f"unit {TimeUnit.quarters.value!r}"
            )

    # endregion

    # region Coordinate Factory

    def _make_coordinate(self, value: CoordinateValue | Coordinate) -> Coordinate:
        """Create a Coordinate in this timeline's unit.

        Args:
            value: The coordinate value or an existing Coordinate.

        Returns:
            A Coordinate with this timeline's unit.

        Raises:
            ValueError: If value is a Coordinate with different unit.
        """
        if isinstance(value, Coordinate):
            if value.unit != self._unit:
                raise ValueError(
                    f"Coordinate unit mismatch: got '{value.unit}', "
                    f"expected '{self._unit}'"
                )
            return value
        return Coordinate(value, self._unit, number_type=self._number_type)

    def _make_length(
        self, value: CoordinateValue | Coordinate | None
    ) -> Coordinate | None:
        """Create a length on this axis; ``None`` (unknown) passes through.

        Raises:
            ValueError: If the length is negative.
        """
        if value is None:
            return None
        return _refuse_negative_length(self._make_coordinate(value), self._id)

    def _require_length(self, operation: str) -> Coordinate:
        """Return the length, raising when an operation needs an unknown one.

        Args:
            operation: What needs the end, as a verb phrase for the message.

        Raises:
            UnknownLengthError: If the length is unknown.
        """
        if self._length is None:
            raise UnknownLengthError(self._id, operation)
        return self._length

    def make_coordinate(self, value: CoordinateValue) -> Coordinate:
        """Create a Coordinate in this timeline's unit.

        Public API for creating coordinates compatible with this timeline.

        Args:
            value: The numeric value for the coordinate.

        Returns:
            A Coordinate with this timeline's unit.
        """
        return Coordinate(value, self._unit)

    # endregion

    # region Lock Management

    def lock(self) -> None:
        """Lock this timeline so that it can no longer expand.

        The timeline enters the state ``locked=True`` gives at construction:
        content and children must fit its length, and appending refuses. A
        root that grows by appending, such as a segment line, is locked this
        way once built. Locking a locked timeline changes nothing. A lock
        cannot be lifted. A segment of a segment line is never locked: its
        line's lock governs it.

        Raises:
            ValueError: If this timeline is a segment that is no segment line
                itself.

        Examples:
            >>> from timetoalign.timelines import (
            ...     ContinuousLogicalTimeline,
            ...     SegmentLine,
            ... )
            >>> line = SegmentLine[ContinuousLogicalTimeline](length=0, unit="quarters")
            >>> line.append_segment(ContinuousLogicalTimeline(length=4, unit="quarters"))
            >>> line.append_segment(ContinuousLogicalTimeline(length=2, unit="quarters"))
            >>> line.is_locked
            False
            >>> line.lock()
            >>> line.is_locked
            True
        """
        if self._parent is not None:
            self._parent._refuse_segment_lock(self)
        self._locked = True

    def _check_not_locked(self, operation: str) -> None:
        """Raise RuntimeError if timeline is locked.

        Args:
            operation: Description of the attempted operation.

        Raises:
            RuntimeError: If timeline is locked.
        """
        if self._locked:
            raise RuntimeError(
                f"Cannot {operation} on locked timeline '{self._id}': "
                "its length is fixed."
            )

    # endregion

    # region Event Management

    @staticmethod
    def _coord_to_value(value: Any) -> Any:
        """Return a coordinate value without discarding an exact ratio."""
        if isinstance(value, dict):
            if (
                value.get("numerator") is not None
                and value.get("denominator") is not None
            ):
                return struct_to_rational(value)
            return value["value"]
        return value

    @staticmethod
    def _coord_to_float(value: Any) -> float:
        """Convert a coordinate value to float.

        Handles both raw numeric values and coordinate struct dicts
        (e.g., ``{"value": 4.5, "numerator": 9, "denominator": 2}``)
        as produced by MeasureData and other loader stores.

        Args:
            value: A numeric value or a dict with a ``"value"`` key.

        Returns:
            The coordinate as a float.
        """
        if isinstance(value, dict):
            return float(value["value"])
        return float(value)

    # endregion

    # region Content Inspection

    def summary(self) -> dict[str, Any]:
        """Get a summary of the timeline.

        Returns:
            Dict with timeline information.
        """
        return {
            "id": self._id,
            "class": self.class_name,
            "unit": str(self._unit),
            "number_type": str(self._number_type),
            "domain": str(self.domain),
            "length": None if self._length is None else self._length.value,
            "is_locked": self._locked,
            "n_events": self.n_events,
            "n_children": self.n_children,
            "event_summary": self._events.summary(),
        }

    def _segment_line_structure_error(self) -> str | None:
        """Return the first exact contiguity error, if any.

        Contiguity is checked between known values: a child of unknown length
        ends at its successor's offset, and the last child's end is compared
        with the timeline's length only when both are known.
        """
        if not self._children:
            return f"Timeline '{self._id}' has no children"

        ordered = sorted(
            self._children.items(),
            key=lambda item: self._child_offsets[item[0]].value,
        )
        expected: CoordinateValue | None = 0
        previous_id: str | None = None
        for child_id, child in ordered:
            actual = self._child_offsets[child_id].value
            if expected is not None and actual != expected:
                expected_display = _format_coordinate_value(float(expected))
                actual_display = _format_coordinate_value(float(actual))
                if previous_id is None:
                    return (
                        f"Timeline start and child '{child_id}' are not contiguous: "
                        f"expected offset {expected_display}, got {actual_display}"
                    )
                return (
                    f"Children '{previous_id}' and '{child_id}' are not contiguous: "
                    f"expected offset {expected_display}, got {actual_display}"
                )
            expected = None if child.length is None else actual + child.length.value
            previous_id = child_id

        if expected is None or self._length is None:
            return None
        if expected != self._length.value:
            return (
                f"Child '{previous_id}' and timeline end are not contiguous: "
                f"expected offset {_format_coordinate_value(float(expected))}, "
                f"got {_format_coordinate_value(float(self._length.value))}"
            )
        return None

    def is_segment_line(self) -> bool:
        """Return whether children exactly and contiguously cover this timeline."""
        return self._segment_line_structure_error() is None

    def as_segment_line(self) -> "Timeline":
        """Cast a structurally contiguous hierarchy to a parameterized line.

        The line is a copy holding copies of the children, so a change to it
        never reaches this tree; everything else is carried over as by
        :meth:`to_typed`.
        """
        from .types import SegmentLine, SegmentLineMixin

        if isinstance(self, SegmentLineMixin):
            return self

        structure_error = self._segment_line_structure_error()
        if structure_error is not None:
            raise ValueError(structure_error)

        ordered = sorted(
            self._children.items(),
            key=lambda item: self._child_offsets[item[0]].value,
        )
        child_class = type(ordered[0][1])
        for _, child in ordered[1:]:
            if type(child) is not child_class:
                raise TypeError(
                    "Cannot cast timeline with heterogeneous child classes: "
                    f"expected {child_class.__name__}, got {type(child).__name__}"
                )

        return self._recast(
            SegmentLine[child_class],
            [(child_id, child._copy_tree()) for child_id, child in ordered],
        )

    # endregion

    # region Serialization

    def to_dict(
        self,
        *,
        minimal: bool = False,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Write a TimeSkeleton manifest holding this timeline as its one root.

        The timeline object is the timeline's structure: its axis, children,
        conversion maps, regions and flow control. Its measure map is a
        ``MeasureRecordMap`` among the conversion maps (chained after a
        ``LinearMap`` to quarters on an axis in whole notes or ticks) naming
        an entry of the manifest's ``measure_maps``, which also holds every
        measure map a measure view names; equal measure maps share one entry.
        The metric hierarchy has no manifest form and is not written. Every
        conversion map is written as one of the format's map types (see
        :meth:`ConversionMap.to_dict <timetoalign.maps.ConversionMap.to_dict>`).
        The library's class is not written: the unit decides it. A value on
        the timeline's own axis — ``length``, a child's ``offset``, region
        ``start``/``end``, a flow-control element's ``coordinate`` — is
        written bare, represented as the axis's number type selects (``8``,
        ``8.0``, ``[8, 1]``); a ``length`` of None is written ``null``.
        ``"events"`` and ``"external_references"`` are written only on
        request.

        By default every optional key is written, its default value
        included (the complete form); with ``minimal=True`` every key at its
        default is left out, except ``length``. Keys the library does not
        know, kept when the timeline was read, are written back. A timeline
        read from a manifest as a whole writes that manifest's ``id``,
        ``name``, ``meta``, ``graphical`` and unknown keys back around it.

        Collections are written in one canonical order, independent of the
        order in which their content was added, so equal timelines give equal
        JSON text: ``children`` by offset, then id; ``conversion_maps`` by id;
        ``regions`` by start, then end, then id; ``flow_control`` (breaks,
        markers and jumps in one array, see
        :meth:`FlowControlRegistry.to_dict
        <timetoalign.timelines.flowcontrol.FlowControlRegistry.to_dict>`) by
        coordinate, then id; ``measure_maps`` by id. Values compare exactly and ids by
        Unicode code point. The timeline's own storage is not reordered, and
        :meth:`from_dict` accepts any order.

        Args:
            minimal: If True, leave out every key at its default value.
            events: If True, write an ``"events"`` key on every timeline
                object holding its event rows.
            external_references: If True, write an ``"external_references"``
                key on every timeline object holding the reference table as
                a list of row dicts (``access_points`` as a nested list of
                ``{"uri": ..., "kind": ...}`` dicts), even when it is empty.

        Returns:
            A JSON-serializable manifest ``{"timeskeleton": "0.7.0", ...,
            "timelines": [timeline]}``.

        Raises:
            ValueError: If a conversion map has no TimeSkeleton form or
                cannot be written losslessly, a jump cannot name the marker
                it lands on, or a measure map's given id is held by another
                measure map or by a timeline, group or the manifest.

        Examples:
            >>> from fractions import Fraction
            >>> from timetoalign.timelines import ContinuousLogicalTimeline
            >>> score = ContinuousLogicalTimeline(length=Fraction(32), uid="piece")
            >>> score.to_dict(minimal=True)
            {'timeskeleton': '0.7.0', 'timelines': [{'id': 'piece', 'unit': 'quarters', \
'length': [32, 1]}]}
        """
        manifest = self._manifest if self._manifest is not None else Manifest()
        return manifest.write(
            [self],
            minimal=minimal,
            events=events,
            external_references=external_references,
        )

    def _to_timeline_object(
        self,
        *,
        minimal: bool,
        events: bool,
        external_references: bool,
        offset: Coordinate | None,
        entries: MeasureMapEntries,
    ) -> dict[str, Any]:
        """Write the timeline object: the timeline's own JSON object.

        Args:
            minimal: Leave out every key at its default value.
            events: Write the event rows.
            external_references: Write the external reference table.
            offset: The ``offset`` to write (a child's placement on its
                parent, a root's origin), or None to write none.
            entries: The measure maps of the manifest written, whose ids
                the measure views name.
        """
        children = [
            child._to_timeline_object(
                minimal=minimal,
                events=events,
                external_references=external_references,
                offset=self._child_offsets[child_id],
                entries=entries,
            )
            for child_id, child in sorted(
                self._children.items(),
                key=lambda item: (self._child_offsets[item[0]].value, item[0]),
            )
        ]
        entries.own = self.measure_map
        conversion_maps = [
            cmap._to_manifest(minimal=minimal, entries=entries)
            for cmap in sorted(self._all_conversion_maps(), key=lambda cmap: cmap.id)
        ]
        fields: dict[str, Any] = {
            "id": self._id,
            "name": self._name,
            "unit": str(self._unit),
            "number_type": str(self._number_type),
        }
        if offset is not None:
            fields["offset"] = axis_value_to_json(offset)
        fields.update(
            {
                "length": (
                    None if self.length is None else axis_value_to_json(self.length)
                ),
                "locked": self._locked,
                "is_segment_line": self._is_segment_line_instance(),
                "meta": copy.deepcopy(self._meta),
                "children": children,
                "conversion_maps": conversion_maps,
                "regions": [
                    replace(
                        region,
                        start=express_scalar_as(region.start, self._number_type),
                        end=express_scalar_as(region.end, self._number_type),
                    ).to_dict(minimal=minimal)
                    for region in sorted(
                        self._regions.values(),
                        key=lambda region: (
                            region.start.value,
                            region.end.value,
                            region.id,
                        ),
                    )
                ],
                "flow_control": self._flow_control.to_dict(
                    minimal=minimal, taken=self._manifest_ids()
                ),
            }
        )
        if events:
            fields["events"] = self._event_rows_to_json()
        if external_references:
            fields["external_references"] = self._external_references.to_pylist()
        defaults = {
            **self._dict_defaults,
            "number_type": str(self._unit.default_number_type),
        }
        return self._dict_form(fields, minimal=minimal, defaults=defaults)

    def _event_rows_to_json(self) -> list[dict[str, Any]]:
        """Return the event rows with their coordinates as JSON numbers."""
        event_rows: list[dict[str, Any]] = []
        for stored_event in self._events:
            event = dict(stored_event)
            for coord_col in ("instant", "start", "end", "duration"):
                value = event.get(coord_col)
                if value is None or not isinstance(value, dict):
                    continue
                if self.number_type == NumberType.float:
                    event[coord_col] = number_to_json(value["value"])
                elif self.number_type == NumberType.int:
                    event[coord_col] = number_to_json(int(struct_to_rational(value)))
                else:
                    event[coord_col] = number_to_json(struct_to_rational(value))
            event_rows.append(event)
        return event_rows

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        timeline_id: str | None = None,
        name: str | None = None,
    ) -> Timeline:
        """Read a timeline from a TimeSkeleton manifest.

        Without *timeline_id* and *name*, a manifest holding one root
        timeline and no group is read as a whole: its root is returned, and
        the manifest's own keys are kept so that :meth:`to_dict` writes the
        manifest back. An argument naming a timeline of the manifest, root
        or descendant, selects it; given both, they must name the same
        timeline. Only arguments naming no timeline override the root's id
        and name, and only in a manifest of one root and no group; the new
        id must not be held by any element of the manifest.

        The class comes from the unit (each unit belongs to exactly one
        class) and ``is_segment_line``. Every optional key may be left out
        and reads as its default; a ``number_type`` left out is the unit's
        own (``int`` for ticks, samples, frames and pixels, ``fraction`` for
        quarters and whole_notes, ``float`` otherwise). A value on the
        timeline's own axis is read as its number type decides, whatever its
        lexical form (``2.0`` on an ``int`` axis is 2, ``[1, 8.0]`` on a
        ``fraction`` axis is 1/8); a ``length`` of ``null`` reads as None. A root's ``offset`` is kept
        as :attr:`root_offset`. Keys the library does not know are kept on
        the object read from the JSON object they were found in, and written
        back.

        Args:
            data: A TimeSkeleton manifest.
            timeline_id: The id to select, or, when no timeline has it and
                the manifest holds one root and no group, the id the root
                takes.
            name: The name to select, or, when no timeline has it and the
                manifest holds one root and no group, the name the root
                takes.

        Returns:
            The timeline read.

        Raises:
            ValueError: If *data* is no TimeSkeleton manifest of version
                0.7.0, is malformed, holds a value represented otherwise than
                its axis selects, if the selection matches no timeline or
                several, or if the id the root would take is already held in
                the manifest; the message names the problem, or the
                manifest's timelines.

        Examples:
            >>> from timetoalign.timelines import Timeline
            >>> piece = Timeline.from_dict(
            ...     {
            ...         "timeskeleton": "0.7.0",
            ...         "timelines": [{"id": "piece", "unit": "quarters", "length": [32, 1]}],
            ...     }
            ... )
            >>> type(piece).__name__, piece.length
            ('ContinuousLogicalTimeline', Coordinate(Fraction(32, 1), quarters))
        """
        manifest, content = Manifest.read(data)
        whole = len(content.roots) == 1 and not content.groups
        timelines = list(content.timelines.values())
        if timeline_id is None and name is None:
            if not whole:
                raise ValueError(
                    "the manifest holds several timelines or groups: give timeline_id or "
                    f"name to select one; its timelines: {listed(content.candidates())}"
                )
            timeline = content.roots[0]
        elif whole and not matches_any(timelines, id=timeline_id, name=name):
            timeline = content.roots[0]
            if timeline_id is not None:
                held = timeline._manifest_ids()
                if manifest.id is not None:
                    held[manifest.id] = "the manifest"
                named = [
                    measure_map
                    for member in (
                        timeline,
                        *(child for _, child in timeline._iter_descendants()),
                    )
                    for cmap in member._all_conversion_maps()
                    for measure_map in cmap._measure_maps()
                ]
                for measure_map in (*named, *manifest.measure_maps):
                    held[measure_map.id] = f"measure map {measure_map.id!r}"
                if timeline_id in held:
                    raise ValueError(
                        f"root timeline {timeline.id!r} cannot take the id "
                        f"{timeline_id!r}: it is already held by {held[timeline_id]}"
                    )
                timeline._set_id(timeline_id)
            if name is not None:
                timeline._name = name
        else:
            timeline = select(
                timelines,
                id=timeline_id,
                name=name,
                what="timeline",
                names=content.candidates(),
            )
        if whole and timeline is content.roots[0]:
            timeline._manifest = manifest
        if not isinstance(timeline, cls):
            raise ValueError(
                f"timeline {timeline.id!r} of the manifest is a {type(timeline).__name__}, "
                f"not a {cls.__name__}"
            )
        return timeline

    @classmethod
    def _from_timeline_object(
        cls,
        data: dict[str, Any],
        *,
        is_root: bool = True,
        measure_maps: Mapping[str, MeasureMap] = MappingProxyType({}),
    ) -> Timeline:
        """Rebuild a timeline from its timeline object.

        A ``MeasureRecordMap`` among its conversion maps, or a ``ChainMap``
        of a ``LinearMap`` from the timeline's unit to quarters and a
        ``MeasureRecordMap``, holds the timeline's measure map; the library
        keeps one measure map per timeline.

        Args:
            data: The timeline object.
            is_root: Whether the timeline is a root: a root reads its
                ``offset`` as :attr:`root_offset`, a child's ``offset`` is
                read by its parent.
            measure_maps: The manifest's measure maps by id, which its
                measure views name.

        Raises:
            ValueError: If the object is malformed, or holds more than one
                map reading a measure map as the timeline's.
        """
        from .types import SegmentLine, SegmentLineMixin, get_timeline_class

        require_object(data, "a timeline", ("id", "unit", "length"))
        timeline_id = read_id(data["id"], "a timeline")
        what = f"timeline {timeline_id!r}"
        try:
            unit = TimeUnit(data["unit"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{what} has an unknown unit {data['unit']!r}") from error
        number_type = unit.default_number_type
        if "number_type" in data:
            try:
                number_type = NumberType[data["number_type"]]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"{what} has an unknown number_type {data['number_type']!r}"
                ) from error
        is_segment_line = data.get("is_segment_line", False)
        locked = data.get("locked", False)
        for key, value in (("is_segment_line", is_segment_line), ("locked", locked)):
            if not isinstance(value, bool):
                raise ValueError(f"{what} has a non-boolean {key!r}")
        length = (
            None
            if data["length"] is None
            else _refuse_negative_length(
                axis_value_from_json(
                    data["length"],
                    key="length",
                    owner=timeline_id,
                    unit=unit,
                    number_type=number_type,
                ),
                timeline_id,
            )
        )

        placed: list[tuple[Coordinate, Timeline]] = []
        for child_data in read_list(data, "children", what):
            require_object(child_data, f"a child of {what}", ("offset",))
            child = Timeline._from_timeline_object(
                child_data, is_root=False, measure_maps=measure_maps
            )
            offset = axis_value_from_json(
                child_data["offset"],
                key="offset",
                owner=child.id,
                unit=unit,
                number_type=number_type,
            )
            placed.append((offset, child))

        timeline_class = get_timeline_class(
            unit.domain.value, discrete=unit.is_discrete
        )
        if is_segment_line:
            segment_types = {type(child) for _, child in placed}
            if len(segment_types) > 1:
                names = sorted(segment_type.__name__ for segment_type in segment_types)
                raise ValueError(
                    f"Segment line {what} has segments of mixed classes: {names}"
                )
            segment_type = segment_types.pop() if segment_types else timeline_class
            innermost = segment_type
            while issubclass(innermost, SegmentLineMixin):
                innermost = innermost._segment_type
            if innermost is not timeline_class:
                raise ValueError(
                    f"Segment line {what} is a {timeline_class.__name__} but its "
                    f"segments are {innermost.__name__!r}"
                )
            timeline_class = SegmentLine[segment_type]

        timeline = timeline_class(
            length=timeline_class._from_dict_initial_length(length),
            unit=unit,
            number_type=number_type,
            uid=timeline_id,
            name=read_name(data, what),
            locked=False,
            meta=read_meta(data, what),
        )

        # Event coordinates arrive as JSON numbers; decode them back to the
        # numbers the EventData builder expects. Segment events are recreated
        # by the children.
        events = []
        for row in read_list(data, "events", what):
            if row.get("event_type") == SEGMENT_EVENT_TYPE:
                continue
            event = dict(row)
            for coord_col in ("instant", "start", "end", "duration"):
                if event.get(coord_col) is not None:
                    event[coord_col] = number_from_json(event[coord_col])
            events.append(event)
        if events:
            timeline._add_events_unchecked(events)
        # Validation is skipped: references may legitimately come without events.
        references = data.get("external_references")
        if references:
            timeline.add_external_references(references, validate=False)

        # Reading never changes a length: a child must fit the written one.
        check_children_fit(data, number_type)
        # A segment line's segments in offset order, so that they append
        # contiguously whatever order the object lists them in.
        if is_segment_line:
            placed.sort(key=lambda item: item[0].value)
        for offset, child in placed:
            timeline._embed_child(child, offset, locked=timeline._kept_lock(child))

        records = []
        for map_data in read_list(data, "conversion_maps", what):
            cmap = ConversionMap.from_dict(map_data, measure_maps=measure_maps)
            if _is_record_map_on(cmap, unit):
                records.append(cmap)
            else:
                timeline.add_conversion_map(cmap)
        if len(records) > 1:
            raise ValueError(
                f"{what} holds {len(records)} MeasureRecordMaps "
                f"{[record.id for record in records]}: the library keeps one measure "
                "map per timeline"
            )
        if records:
            timeline._set_measure_record_map(records[0])
        for region_data in read_list(data, "regions", what):
            timeline.add_region(
                Region.from_dict(region_data, unit=unit, number_type=number_type)
            )
        timeline.flow_control._absorb(
            FlowControlRegistry.from_dict(
                read_list(data, "flow_control", what),
                unit=unit,
                number_type=number_type,
            )
        )

        timeline._finalize_from_dict(length)
        timeline._locked = locked
        if is_root and "offset" in data:
            timeline._root_offset = axis_value_from_json(
                data["offset"],
                key="offset",
                owner=timeline_id,
                unit=unit,
                number_type=number_type,
            )
        timeline._keep_unknown_keys(data)
        return timeline

    def _typed_class(self) -> type["Timeline"]:
        """Return the canonical concrete class for this timeline."""
        from .types import get_timeline_class

        return get_timeline_class(self.domain.value, discrete=self._unit.is_discrete)

    def to_typed(self) -> "Timeline":
        """Return this timeline re-instantiated as the appropriate typed subclass.

        Uses the timeline's unit and number type to determine the correct
        concrete subclass (e.g., ``ContinuousPhysicalTimeline`` for seconds/float).
        If the timeline is already an instance of the correct subclass, returns
        ``self`` unchanged.

        This is useful after deserialization (e.g., ``Timeline.from_dict()``) or
        when working with generic ``Timeline`` instances that should carry
        domain-specific type information.

        Events, external references, conversion maps, regions, flow maps,
        metadata, the root offset and the keys the timeline was read with are
        preserved. Children are recursively re-typed copies, so a change to
        the result never reaches the source tree.

        Returns:
            A Timeline instance of the appropriate typed subclass, or ``self``
            if it is already the correct type.

        Examples:
            >>> tl = Timeline(length=10.0, unit=TimeUnit.seconds)
            >>> typed = tl.to_typed()
            >>> type(typed).__name__
            'ContinuousPhysicalTimeline'
            >>> typed.is_continuous
            True

            >>> # Already typed -- returns self
            >>> cpt = ContinuousPhysicalTimeline(length=10.0)
            >>> cpt.to_typed() is cpt
            True
        """
        if self._is_typed_tree():
            return self
        return self._typed_copy()

    def _is_typed_tree(self) -> bool:
        """Whether this timeline and every descendant are of their typed class."""
        return type(self) is self._typed_class() and all(
            child._is_typed_tree() for child in self._children.values()
        )

    def _typed_copy(self) -> "Timeline":
        """Copy this tree, each timeline re-instantiated as its typed class."""
        return self._recast(
            self._typed_class(),
            [
                (child_id, child._typed_copy())
                for child_id, child in self._children.items()
            ],
        )

    def _copy_tree(self) -> "Timeline":
        """Copy this timeline and its descendants, sharing no timeline with them."""
        return self._recast(
            type(self),
            [
                (child_id, child._copy_tree())
                for child_id, child in self._children.items()
            ],
        )

    def _recast(
        self,
        target_class: type["Timeline"],
        children: list[tuple[str, "Timeline"]],
    ) -> "Timeline":
        """Copy this timeline as a *target_class*, embedding *children* instead of its own.

        Everything the timeline holds is carried over: events, external
        references, conversion maps, regions, flow maps, measure map, metric
        hierarchy, a copy of its flow control, its length and lock, its root
        offset, the keys it was read with and the manifest it was read from.
        Each entry of *children* pairs a child's id with the timeline to embed
        in its place, at its offset and with its lock state; the caller passes
        copies, so that a change to the result never reaches this tree.

        Args:
            target_class: The class of the copy.
            children: ``(child id, timeline to embed)`` pairs, a segment
                line's in offset order.

        Returns:
            The copy, without a parent.
        """
        from .types import SegmentLineMixin

        is_line = issubclass(target_class, SegmentLineMixin)
        copied = target_class(
            length=0 if is_line else self._length,
            unit=self._unit,
            number_type=self._number_type,
            uid=self._id,
            name=self._name,
            locked=False,
            meta=dict(self._meta) if self._meta else None,
        )
        events = [
            dict(event)
            for event in self._events
            if event.get("event_type") != SEGMENT_EVENT_TYPE
        ]
        if events:
            copied._add_events_unchecked(events)
        # Arrow tables are immutable, so the table object can be shared.
        copied._external_references = self._external_references
        for child_id, child in children:
            copied._embed_child(
                child,
                self._child_offsets[child_id],
                allow_expansion=is_line,
                locked=copied._kept_lock(child),
            )
        for cmap in self._conversion_maps.values():
            copied.add_conversion_map(cmap)
        copied._regions.update(self._regions)
        copied._flow_maps.update(self._flow_maps)
        copied._measure_record_map = self._measure_record_map
        copied._metric_hierarchy = self._metric_hierarchy
        copied._adopt_flow_control(self._flow_control.copy())
        copied._id_minted = self._id_minted
        copied._length = self._length
        copied._locked = self._locked
        copied._root_offset = self._root_offset
        copied._unknown_keys = copy.deepcopy(dict(self._unknown_keys))
        copied._manifest = self._manifest
        return copied

    # endregion

    # region Display

    def diagram(
        self,
        width: int = 70,
        show_children: bool = True,
        max_children: int = 6,
        unicode: bool = True,
        show: set[str] | None = None,
        depth: bool | int = True,
    ) -> "Diagram":
        """Generate ASCII diagram for this timeline.

        Args:
            width: Total width of the diagram in characters.
            show_children: Whether to show child timelines (one per row).
            max_children: Maximum children to show before truncating.
            unicode: Use Unicode characters (True) or ASCII fallback (False).
            show: Optional set controlling which elements appear. Supported
                values: ``"children"``, ``"regions"``, and ``"cmaps"``
                (attached conversion maps). When ``None``, behaviour is
                exactly as before.
            depth: Child levels to render. ``True`` renders all levels,
                ``False`` renders direct children only, and a non-negative
                integer renders at most that many levels below this timeline.
                In particular, ``0`` renders no child rows.

        Returns:
            Diagram object (displays as ASCII in terminal, rich HTML in Jupyter).

        Raises:
            ValueError: If ``depth`` is a negative integer.

        Examples:
            >>> print(timeline.diagram())
            DiscreteGraphicalTimeline[dgt_1:1] (11 events, 5 children)
            0 ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶ 4835 pixels
              ├─ system_1     0   ∶∶∶∶∶∶∶                        967
              ├─ system_2   967          ∶∶∶∶∶∶∶∶               1934
              └─ ...
        """
        from timetoalign.display.ascii import timeline_diagram

        return timeline_diagram(
            self,
            width=width,
            show_children=show_children,
            max_children=max_children,
            unicode=unicode,
            show=show,
            depth=depth,
        )

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return total number of events (excluding segments)."""
        return self.n_events

    def __repr__(self) -> str:
        """Return string representation."""
        parts = [
            f"id={self._id!r}",
            f"length={'unknown' if self._length is None else self._length.value}",
            f"unit={self._unit}",
            f"events={self.n_events}",
            f"children={self.n_children}",
        ]
        if self.n_conversion_maps > 0:
            parts.append(f"cmaps={self.n_conversion_maps}")
        return f"{self.class_name}({', '.join(parts)})"

    def __str__(self) -> str:
        """Return human-readable ASCII diagram of the timeline.

        Uses the diagram() method to generate a visual representation
        showing the timeline bar and any children.
        """
        return str(self.diagram())

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the ASCII diagram in a monospace pre block so it
        renders correctly in notebook output cells.
        """
        return self.diagram()._repr_html_()

    def __contains__(self, item: str | Region | Timeline) -> bool:
        """Check if a region ID, child ID, or timeline is part of this timeline.

        Checks all noun types uniformly:
        - ``"some_id" in tl`` checks region ids AND child ids.
        - ``some_timeline in tl`` checks if it's a child (by identity).
        - ``some_region in tl`` checks if a region with that id exists.
        """
        if isinstance(item, Region):
            return item.id in self._regions
        if isinstance(item, Timeline):
            return item.id in self._children
        # String: check regions first, then children
        return item in self._regions or item in self._children

    # endregion


# Runtime type binding avoids a circular module dependency.
_children_engine.Timeline = Timeline
_conversion_engine.Timeline = Timeline
