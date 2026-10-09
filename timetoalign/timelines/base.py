"""Timeline: The central object of TimeToAlign!

A Timeline is a positive coordinate axis defined by an origin (zero) and a
measuring unit. It can hold events (via an EventData) and nested child
timelines (segments) that share the same coordinate type.

Design principles:
- Events stored in PyArrow-based EventData (no flyweight pattern)
- Children stored as direct object references AND as segment events
- Unit validation at add-time prevents coordinate type mismatches
- Locking mechanism prevents modification when embedded as child
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Literal

from typing_extensions import Self

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    CoordinateValue,
    Domain,
    NumberType,
    TimeUnit,
    rational_from_dict,
    rational_to_dict,
)
from timetoalign.core.timestamp import _format_coordinate_value
from timetoalign.maps import ConversionMap
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
from .regions import Region
from .structure import MeasureMap, MetricHierarchy

if TYPE_CHECKING:
    from timetoalign.display.ascii import Diagram

module_logger = logging.getLogger(__name__)

# Module-level counter for unique ID generation
_TIMELINE_COUNTER: dict[str, int] = {}

# region Constants

# Event type name for segment events in the EventData
SEGMENT_EVENT_TYPE = "Segment"

# Traversal order options for iterating children
TraversalOrder = Literal["sorted", "depth_first", "breadth_first"]

# endregion


class Timeline(
    EventsMixin,
    ExternalReferencesMixin,
    ChildrenMixin,
    ConversionMapsMixin,
    TabularExportMixin,
    RegionsMixin,
    SegmentsMixin,
    SkeletonsMixin,
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
        length: The end coordinate.
        is_locked: Whether the timeline can be modified.
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

    # Serialized class tag -> timeline subclass. Timeline itself is handled
    # directly; every subclass registers automatically when defined.
    _registry: ClassVar[dict[str, type["Timeline"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register a timeline subclass under its serialized class tag."""
        super().__init_subclass__(**kwargs)
        tag = cls.__name__
        registered = Timeline._registry.get(tag)
        if registered is not None and registered is not cls:
            raise ValueError(
                f"Timeline tag {tag!r} is already registered to {registered!r}; "
                f"cannot register {cls!r}."
            )
        Timeline._registry[tag] = cls

    @classmethod
    def _resolve_serialized_class_hierarchy(cls, class_tag: str) -> type["Timeline"]:
        """Resolve a serialized class tag, materializing parameters recursively."""
        registered = (
            Timeline
            if class_tag == Timeline.__name__
            else Timeline._registry.get(class_tag)
        )
        if registered is not None:
            return registered

        bracket_start = class_tag.find("[")
        if bracket_start == -1:
            raise ValueError(f"Unknown serialized timeline class '{class_tag}'")

        outer_tag = class_tag[:bracket_start]
        if not outer_tag or not class_tag.endswith("]"):
            raise ValueError(f"Unknown serialized timeline class '{class_tag}'")

        depth = 0
        for index, char in enumerate(class_tag[bracket_start:], start=bracket_start):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0 and index != len(class_tag) - 1:
                    raise ValueError(f"Unknown serialized timeline class '{class_tag}'")
                if depth < 0:
                    raise ValueError(f"Unknown serialized timeline class '{class_tag}'")
        if depth != 0:
            raise ValueError(f"Unknown serialized timeline class '{class_tag}'")

        outer_class = Timeline._resolve_serialized_class_hierarchy(outer_tag)
        from .types import SegmentLine

        if outer_class is not SegmentLine:
            raise ValueError(
                f"Serialized timeline class '{outer_class.__name__}' "
                "does not accept type parameters"
            )

        inner_start = bracket_start + 1
        inner_tag = class_tag[inner_start:-1]
        try:
            inner_class = Timeline._resolve_serialized_class_hierarchy(inner_tag)
        except ValueError as error:
            raise ValueError(f"{error} in parameterized tag '{class_tag}'") from error

        return SegmentLine[inner_class]

    @classmethod
    def _from_dict_initial_length(cls, data: dict[str, Any]) -> CoordinateValue:
        """Return the length to use while reconstructing a payload."""
        return rational_from_dict(data["length"])

    def _finalize_from_dict(self, data: dict[str, Any]) -> None:
        """Restore state that must be set after children are added."""

    @classmethod
    def _validate_serialized_payload(cls, data: dict[str, Any]) -> None:
        """Validate compatibility rules specific to serialized payloads."""

    # endregion

    # region Initialization

    def __init__(
        self,
        length: CoordinateValue = 0,
        unit: TimeUnit | str | None = None,
        number_type: NumberType | str | None = None,
        id_prefix: str = "tl",
        uid: str | None = None,
        name: str | None = None,
        locked: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Timeline.

        Args:
            length: The length (end coordinate) of the timeline. Default 0.
            unit: The time unit for coordinates. Defaults to class default.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.
            id_prefix: Prefix for auto-generated ID. Default "tl".
            uid: Explicit unique identifier. Overrides auto-generation.
            name: Human-readable name for display (distinct from uid).
            locked: If True, timeline cannot expand. Default False.
            meta: Additional metadata dictionary.

        Raises:
            ValueError: If unit or number_type is not allowed for this class.
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

        # Generate or use provided ID
        if uid is not None:
            self._id = uid
        else:
            # Use module-level counter for truly unique IDs
            if id_prefix not in _TIMELINE_COUNTER:
                _TIMELINE_COUNTER[id_prefix] = 0
            _TIMELINE_COUNTER[id_prefix] += 1
            self._id = f"{id_prefix}:{_TIMELINE_COUNTER[id_prefix]}"

        # Initialize length as Coordinate
        self._length = self._make_coordinate(length)

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
        self._measure_map: MeasureMap | None = None
        self._metric_hierarchy: MetricHierarchy | None = None
        self._flow_control = FlowControlRegistry(
            unit=self._unit, number_type=self._number_type
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
    def length(self) -> Coordinate:
        """The end coordinate (length of the timeline)."""
        return self._length

    @length.setter
    def length(self, value: CoordinateSpec) -> None:
        """Set the timeline length.

        Args:
            value: New length value.

        Raises:
            RuntimeError: If timeline is locked.
            ValueError: If new length is less than current content.
        """
        self._check_not_locked("set length")

        new_length = self.get_coordinate_at(value, format="coordinate")

        # Check that new length accommodates all content
        max_content = self._get_max_content_coordinate()
        if new_length.value < max_content:
            raise ValueError(
                f"Cannot reduce length to "
                f"{_format_coordinate_value(float(new_length.value))}: "
                f"content extends to {_format_coordinate_value(float(max_content))}"
            )

        self._length = new_length

    @property
    def start(self) -> Coordinate:
        """Alias for origin."""
        return self.origin

    @property
    def end(self) -> Coordinate:
        """Alias for length."""
        return self._length

    # endregion

    # region Properties - State

    @property
    def is_locked(self) -> bool:
        """Whether this timeline is locked (cannot expand)."""
        return self._locked

    @property
    def meta(self) -> dict[str, Any]:
        """Metadata dictionary."""
        return dict(self._meta)

    @property
    def measure_map(self) -> MeasureMap | None:
        """Authored measure structure, absent on fresh and derived slices."""
        return self._measure_map

    @property
    def metric_hierarchy(self) -> MetricHierarchy | None:
        """Authored metric structure, absent on fresh and derived slices."""
        return self._metric_hierarchy

    @property
    def flow_control(self) -> FlowControlRegistry:
        """Axis-bound breaks, jumps, and named control markers."""
        return self._flow_control

    # endregion

    # region Authored Structure

    def add_measure_map(self, measure_map: MeasureMap) -> Self:
        """Attach the timeline's single authored measure map.

        Args:
            measure_map: Immutable measure structure to attach.

        Returns:
            This timeline after attachment.

        Raises:
            ValueError: If the timeline already carries a measure map.
        """
        if self._measure_map is not None:
            raise ValueError(f"Timeline {self._id!r} already has a measure map")
        self._measure_map = measure_map
        return self

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
        if self._measure_map is None:
            raise ValueError("A metric hierarchy requires the timeline's measure map")
        if metric_hierarchy.measure_map != self._measure_map:
            raise ValueError(
                "The metric hierarchy is anchored on a different measure map"
            )
        if self.unit is not TimeUnit.quarters:
            raise ValueError(
                f"Timeline unit {self.unit.value!r} does not match metric hierarchy "
                f"unit {TimeUnit.quarters.value!r}"
            )
        self._metric_hierarchy = metric_hierarchy
        return self

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
        return Coordinate(value, self._unit)

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

    def _check_not_locked(self, operation: str) -> None:
        """Raise RuntimeError if timeline is locked.

        Args:
            operation: Description of the attempted operation.

        Raises:
            RuntimeError: If timeline is locked.
        """
        if self._locked:
            raise RuntimeError(
                f"Cannot {operation} on locked timeline '{self._id}'. "
                "Timelines are locked when embedded as children."
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
                return rational_from_dict(value)
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
            "length": self._length.value,
            "is_locked": self._locked,
            "n_events": self.n_events,
            "n_children": self.n_children,
            "event_summary": self._events.summary(),
        }

    def _segment_line_structure_error(self) -> str | None:
        """Return the first exact contiguity error, if any."""
        if not self._children:
            return f"Timeline '{self._id}' has no children"

        ordered = sorted(
            self._children.items(),
            key=lambda item: self._child_offsets[item[0]].value,
        )
        expected: CoordinateValue = 0
        previous_id: str | None = None
        for child_id, child in ordered:
            actual = self._child_offsets[child_id].value
            if actual != expected:
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
            expected = actual + child.length.value
            previous_id = child_id

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
        """Cast a structurally contiguous hierarchy to a parameterized line."""
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

        segment_line = SegmentLine[child_class](
            length=0,
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
            segment_line._add_events_unchecked(events)
        segment_line._external_references = self._external_references
        for _, child in ordered:
            segment_line.append_segment(child)
        for cmap in self._conversion_maps.values():
            segment_line.add_conversion_map(cmap)
        segment_line._regions.update(self._regions)
        segment_line._flow_maps.update(self._flow_maps)
        segment_line._measure_map = self._measure_map
        segment_line._metric_hierarchy = self._metric_hierarchy
        segment_line._flow_control = self._flow_control.copy()
        segment_line._locked = self._locked
        return segment_line

    # endregion

    # region Serialization

    def to_dict(
        self,
        *,
        events: bool = False,
        external_references: bool = False,
    ) -> dict[str, Any]:
        """Convert timeline to a dictionary for serialization.

        The default output is the timeline's complete structure: its axis,
        children, conversion maps, regions, measure map, metric hierarchy,
        and flow control. ``"events"`` and ``"external_references"`` are
        **absent** unless explicitly requested. This structural payload is
        called the eventless payload: it preserves authored structure without
        carrying event rows. Structure is not inferred for slices, children
        created from regions, or unfolded timelines.

        Coordinate-valued members — ``length`` and every child
        ``offset`` — are emitted as the canonical rational dictionary
        (:func:`~timetoalign.core.rational_to_dict`), so the result is
        JSON-serializable whatever the timeline's number type, and
        ``Fraction`` coordinates survive the round trip exactly.

        Args:
            events: If True, include an ``"events"`` key holding this
                timeline's event rows.
            external_references: If True, include an
                ``"external_references"`` key holding the reference table
                as a list of row dicts (``access_points`` as a nested list
                of ``{"uri": ..., "kind": ...}`` dicts). Included even
                when the table is empty.

        Returns:
            A JSON-serializable dictionary representation of the timeline.

        Examples:
            >>> "events" in tl.to_dict()
            False
            >>> "events" in tl.to_dict(events=True)
            True
        """
        children_data = {}
        for child_id, child in self._children.items():
            children_data[child_id] = {
                "offset": rational_to_dict(self._child_offsets[child_id].value),
                "timeline": child.to_dict(
                    events=events,
                    external_references=external_references,
                ),
            }

        data: dict[str, Any] = {
            "id": self._id,
            "name": self._name,
            "class": self.class_name,
            "unit": str(self._unit),
            "number_type": str(self._number_type),
            "length": rational_to_dict(self._length.value),
            "locked": self._locked,
            "meta": self._meta,
            "children": children_data,
            "conversion_maps": [
                cmap.to_dict() for cmap in self._conversion_maps.values()
            ],
            "regions": [region.to_dict() for region in self._regions.values()],
            "flow_control": self._flow_control.to_dict(),
            "measure_map": (
                self._measure_map.to_dict() if self._measure_map is not None else None
            ),
            "metric_hierarchy": (
                self._metric_hierarchy.to_dict()
                if self._metric_hierarchy is not None
                else None
            ),
        }

        if events:
            data["events"] = list(self._events)
        if external_references:
            data["external_references"] = self._external_references.to_pylist()

        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a Timeline from a dictionary.

        Every rational dictionary in *data* — ``length``, the child
        ``offset``s, and the event coordinate structs — is decoded by
        :func:`~timetoalign.core.rational_from_dict`, so an exact ratio
        comes back as a ``Fraction`` and an inexact one as a ``float``.
        Feeding the result back through :meth:`to_dict` with the same
        flags reproduces the input dictionary.

        The ``"events"`` and ``"external_references"`` keys are optional:
        a dictionary produced without them reconstructs a timeline with
        zero events and an empty reference table. External references are
        restored without event validation, so a payload carrying
        references but no events round-trips intact.

        Args:
            data: Dictionary from to_dict().

        Returns:
            A new Timeline instance.
        """
        class_tag = data.get("class")
        if not isinstance(class_tag, str):
            raise ValueError("Serialized timeline is missing a string 'class' tag")

        timeline_class = Timeline._resolve_serialized_class_hierarchy(class_tag)
        if timeline_class is not cls and issubclass(timeline_class, cls):
            return timeline_class.from_dict(data)

        if timeline_class is not cls:
            raise ValueError(
                f"Serialized timeline class '{class_tag}' does not match "
                f"receiving subclass '{cls.__name__}'"
            )

        cls._validate_serialized_payload(data)

        timeline = cls(
            length=cls._from_dict_initial_length(data),
            unit=data["unit"],
            number_type=data["number_type"],
            uid=data["id"],
            name=data.get("name"),
            locked=False,
            meta=data.get("meta"),
        )

        # Add events (filter out segment events - they'll be recreated).
        # Event coordinates arrive as rational dictionaries; decode them
        # back to the numbers the EventData builder expects.
        events = []
        for e in data.get("events", []):
            if e.get("event_type") == SEGMENT_EVENT_TYPE:
                continue
            event = dict(e)
            for coord_col in ("instant", "start", "end", "duration"):
                if event.get(coord_col) is not None:
                    event[coord_col] = rational_from_dict(event[coord_col])
            events.append(event)

        if events:
            timeline._add_events_unchecked(events)

        # Restore incoming external references. Validation is skipped: the
        # payload may legitimately carry references without events.
        references = data.get("external_references")
        if references:
            timeline.add_external_references(references, validate=False)

        # Add children
        for child_id, child_data in data.get("children", {}).items():
            child = Timeline.from_dict(child_data["timeline"])
            timeline.add_child(child, offset=rational_from_dict(child_data["offset"]))

        # Add conversion maps
        for map_data in data.get("conversion_maps", []):
            cmap = ConversionMap.from_dict(map_data)
            timeline.add_conversion_map(cmap)

        # Restore annotations and authored structures before subclass finalization.
        for region_data in data.get("regions", []):
            timeline.add_region(Region.from_dict(region_data))
        timeline._flow_control = FlowControlRegistry.from_dict(
            data.get("flow_control", {}),
            unit=timeline.unit,
            number_type=timeline.number_type,
        )
        measure_map_data = data.get("measure_map")
        if measure_map_data is not None:
            timeline.add_measure_map(MeasureMap.from_dict(measure_map_data))
        metric_hierarchy_data = data.get("metric_hierarchy")
        if metric_hierarchy_data is not None:
            if timeline.measure_map is None:
                raise ValueError(
                    "A timeline payload with metric_hierarchy requires measure_map"
                )
            timeline.add_metric_hierarchy(
                MetricHierarchy.from_dict(
                    metric_hierarchy_data, measure_map=timeline.measure_map
                )
            )

        timeline._finalize_from_dict(data)
        timeline._locked = data.get("locked", False)

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

        Events, external references, conversion maps, regions, flow maps, and
        metadata are preserved. Children are recursively re-typed.

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
        from .types import SegmentLineMixin

        target_class = self._typed_class()
        typed_children = {
            child_id: child.to_typed() for child_id, child in self._children.items()
        }

        if type(self) is target_class and all(
            typed_children[child_id] is child
            for child_id, child in self._children.items()
        ):
            return self

        initial_length: CoordinateValue = (
            0 if issubclass(target_class, SegmentLineMixin) else self._length.value
        )
        typed = target_class(
            length=initial_length,
            unit=self._unit,
            number_type=self._number_type,
            uid=self._id,
            name=self._name,
            locked=False,
            meta=dict(self._meta) if self._meta else None,
        )

        # Transfer events (use internal method to skip validation overhead)
        if self._events is not None and len(self._events) > 0:
            events = []
            for event in self._events:
                if event.get("event_type") == SEGMENT_EVENT_TYPE:
                    continue
                events.append(dict(event))
            if events:
                typed._add_events_unchecked(events)

        # Transfer external references (Arrow tables are immutable, so the
        # table object can be shared between the two timelines)
        typed._external_references = self._external_references

        # Transfer recursively typed children
        for child_id, child in typed_children.items():
            offset = self._child_offsets[child_id]
            typed.add_child(
                child,
                offset=offset,
                allow_expansion=isinstance(typed, SegmentLineMixin),
            )

        # Transfer conversion maps
        for cmap in self._conversion_maps.values():
            typed.add_conversion_map(cmap)

        # Transfer regions
        for region in self._regions.values():
            typed._regions[region.name] = region

        # Transfer flow maps
        for flow_id, flow_map in self._flow_maps.items():
            typed._flow_maps[flow_id] = flow_map

        typed._measure_map = self._measure_map
        typed._metric_hierarchy = self._metric_hierarchy
        typed._flow_control = self._flow_control.copy()

        typed._length = typed._make_coordinate(self._length.value)
        typed._locked = self._locked
        return typed

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
            DiscreteGraphicalTimeline[dgt1:1] (11 events, 5 children)
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
            f"length={self._length.value}",
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
        """Check if a region name, child ID, or timeline is part of this timeline.

        Checks all noun types uniformly:
        - ``"name" in tl`` checks regions AND children.
        - ``some_timeline in tl`` checks if it's a child (by identity).
        - ``some_region in tl`` checks if a region with that name exists.
        """
        if isinstance(item, Region):
            return item.name in self._regions
        if isinstance(item, Timeline):
            return item.id in self._children
        # String: check regions first, then children
        return item in self._regions or item in self._children

    # endregion


# Runtime type binding avoids a circular module dependency.
_children_engine.Timeline = Timeline
_conversion_engine.Timeline = Timeline
