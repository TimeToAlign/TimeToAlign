"""Domain-specific Timeline subclasses.

This module provides the 6 concrete Timeline types:
- ContinuousLogicalTimeline, DiscreteLogicalTimeline
- ContinuousPhysicalTimeline, DiscretePhysicalTimeline
- ContinuousGraphicalTimeline, DiscreteGraphicalTimeline

Each class restricts valid units to its domain and modality,
and provides sensible defaults.

Additionally provides:
- SegmentLine: A timeline where all children are contiguous (Segments)
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, ClassVar, Iterator

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    NumberType,
    TimeUnit,
)

from .base import Timeline

# region Unit Sets by Domain

# Logical domain units (symbolic/musical time)
LOGICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.whole_note,
        TimeUnit.quarters,
        TimeUnit.floating_measures,
        TimeUnit.ticks,
        TimeUnit.number,
    }
)

# Physical domain units (acoustic/real time)
PHYSICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.seconds,
        TimeUnit.milliseconds,
        TimeUnit.minutes,
        TimeUnit.samples,
        TimeUnit.frames,
    }
)

# Graphical domain units (visual/spatial)
GRAPHICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.pixels,
        TimeUnit.meters,
        TimeUnit.centimeters,
        TimeUnit.millimeters,
        TimeUnit.inches,
        TimeUnit.points,
    }
)

# Continuous vs discrete unit categorization
CONTINUOUS_LOGICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.whole_note,
        TimeUnit.quarters,
        TimeUnit.floating_measures,
        TimeUnit.number,
    }
)
DISCRETE_LOGICAL_UNITS: frozenset[TimeUnit] = frozenset({TimeUnit.ticks})

CONTINUOUS_PHYSICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {TimeUnit.seconds, TimeUnit.milliseconds, TimeUnit.minutes}
)
DISCRETE_PHYSICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {TimeUnit.samples, TimeUnit.frames}
)

CONTINUOUS_GRAPHICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.meters,
        TimeUnit.centimeters,
        TimeUnit.millimeters,
        TimeUnit.inches,
        TimeUnit.points,
    }
)
DISCRETE_GRAPHICAL_UNITS: frozenset[TimeUnit] = frozenset({TimeUnit.pixels})

# endregion


# region Domain Base Classes


class LogicalTimeline(Timeline):
    """A timeline representing logical/musical time.

    Logical timelines measure symbolic musical time such as beats,
    quarter notes, measures, or MIDI ticks. They are used for
    score-based representations of music.

    Allowed units: beats, quarters, measures, ticks, number.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = LOGICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.quarters


class PhysicalTimeline(Timeline):
    """A timeline representing physical/acoustic time.

    Physical timelines measure real-world time in units like
    seconds, milliseconds, or audio samples. They are used for
    performance and audio representations.

    Allowed units: seconds, milliseconds, minutes, samples, frames.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = PHYSICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds


class GraphicalTimeline(Timeline):
    """A timeline representing graphical/spatial coordinates.

    Graphical timelines measure visual positions for score
    visualization and plotting. They can be in pixels or
    physical measurements.

    Allowed units: pixels, meters, centimeters, millimeters, inches, points.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = GRAPHICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.pixels


# endregion


# region Continuous Timeline Types


class ContinuousLogicalTimeline(LogicalTimeline):
    """A logical timeline with continuous coordinates.

    Used for score representations where fractional beat positions
    are meaningful (e.g., a note at beat 2.5 or quarter beat 3/4).

    Default unit: quarters (quarter notes).
    Default number type: Fraction (for exact rhythmic representation).
    Allowed units: beats, quarters, measures, number (NOT ticks).
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = CONTINUOUS_LOGICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.quarters


class ContinuousPhysicalTimeline(PhysicalTimeline):
    """A physical timeline with continuous coordinates.

    Used for acoustic time measurements where arbitrary precision
    is needed (e.g., note onsets at 1.234 seconds).

    Default unit: seconds.
    Default number type: float.
    Allowed units: seconds, milliseconds, minutes (NOT samples/frames).
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = CONTINUOUS_PHYSICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds


class ContinuousGraphicalTimeline(GraphicalTimeline):
    """A graphical timeline with continuous coordinates.

    Used for visualization where real-valued positions are needed
    (e.g., a note head at x=12.75 centimeters).

    Default unit: centimeters.
    Default number type: float.
    Allowed units: meters, centimeters, millimeters, inches, points.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = CONTINUOUS_GRAPHICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.centimeters

    @classmethod
    def _validate_serialized_payload(cls, data: dict[str, Any]) -> None:
        """Explain legacy graphical payloads that predate unit validation."""
        if data.get("unit") == str(TimeUnit.pixels):
            raise ValueError(
                "Serialized ContinuousGraphicalTimeline payload uses unit 'pixels', "
                "which is discrete-only. This payload predates unit enforcement; "
                "use DiscreteGraphicalTimeline instead."
            )


# endregion


# region Discrete Timeline Types


class DiscreteLogicalTimeline(LogicalTimeline):
    """A logical timeline with discrete (integer) coordinates.

    Used for MIDI-based representations where time is measured in
    quantized ticks. Essential for MIDI file parsing and generation.

    Default unit: ticks.
    Default number type: int.
    Allowed units: ticks only.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = DISCRETE_LOGICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.ticks


class DiscretePhysicalTimeline(PhysicalTimeline):
    """A physical timeline with discrete (integer) coordinates.

    Used for audio sample-based representations where time is
    measured in discrete sample indices or video frames.

    Default unit: samples.
    Default number type: int.
    Allowed units: samples, frames.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = DISCRETE_PHYSICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.samples


class DiscreteGraphicalTimeline(GraphicalTimeline):
    """A graphical timeline with discrete (integer) coordinates.

    Used for pixel-based visualization where positions are
    quantized to screen coordinates.

    Default unit: pixels.
    Default number type: int.
    Allowed units: pixels only.
    """

    _allowed_units: ClassVar[frozenset[TimeUnit]] = DISCRETE_GRAPHICAL_UNITS
    _default_unit: ClassVar[TimeUnit] = TimeUnit.pixels


# endregion


# region SegmentLine


class SegmentLineMixin:
    """Enforce and expose contiguous, homogeneously typed child segments."""

    _segment_type: ClassVar[type[Timeline] | None] = None

    def __init__(self, **kwargs: Any) -> None:
        """Initialize segment ordering before accepting children."""
        super().__init__(**kwargs)
        self._segment_order: list[str] = []

    @classmethod
    def _from_dict_initial_length(cls, data: dict[str, Any]) -> int:
        """Start empty so restored segments can be appended contiguously."""
        return 0

    def _finalize_from_dict(self, data: dict[str, Any]) -> None:
        """Restore the serialized length after rebuilding the segments."""
        self._length = self._make_coordinate(Timeline._from_dict_initial_length(data))

    @property
    def segment_type(self) -> type[Timeline] | None:
        """Return the enforced class, inferred on first append for a bare line."""
        return self._segment_type

    @property
    def class_name(self) -> str:
        """Return the dynamic class name or a bare line's inferred dictionary tag."""
        if type(self) is SegmentLine and self._segment_type is not None:
            return f"SegmentLine[{self._segment_type.__name__}]"
        return type(self).__name__

    def validate_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
    ) -> None:
        """Validate the segment class and its exact contiguous offset."""
        super().validate_child(child, offset)

        if self._segment_type is not None:
            if not isinstance(child, self._segment_type):
                raise TypeError(
                    f"SegmentLine expects segments of type "
                    f"{self._segment_type.__name__}, got "
                    f"{type(child).__name__}. All segments in a "
                    f"SegmentLine must be the same Timeline subclass."
                )

        offset_val = self._resolve_axis_value(offset)
        expected_offset = self.length.value if self._segment_order else 0

        if offset_val != expected_offset:
            raise ValueError(
                f"SegmentLine requires contiguous segments. "
                f"Expected offset {expected_offset}, got {offset_val}. "
                f"Use append_segment() for automatic offset calculation."
            )

    def add_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
        allow_expansion: bool = False,
    ) -> None:
        """Add a validated segment and record its insertion order."""
        super().add_child(child, offset, allow_expansion)

        if self._segment_type is None:
            self._segment_type = type(child)
        self._segment_order.append(child.id)

    def _rename_child(self, old_id: str, new_id: str) -> None:
        """Re-key a segment, keeping its place in the segment order."""
        super()._rename_child(old_id, new_id)
        self._segment_order = [
            new_id if segment_id == old_id else segment_id
            for segment_id in self._segment_order
        ]

    def append_segment(
        self,
        segment: Timeline,
        name: str | None = None,
    ) -> None:
        """Append a segment at the current end coordinate."""
        self.append_child(segment, name=name)

    def get_segment_at(
        self,
        coord: CoordinateSpec,
    ) -> tuple[int, Timeline, Any]:
        """Return the index, segment, and local timestamp at a coordinate."""
        ts = self.get_timestamp_at(coord)
        coord_val = float(self._resolve_axis_value(coord))

        for i, seg_id in enumerate(self._segment_order):
            try:
                seg_coord = ts.get_coordinate_for(seg_id, format="coordinate")
            except KeyError:
                continue
            if seg_coord.value >= 0:
                segment = self._children[seg_id]
                if seg_coord.value <= segment.length.value:
                    seg_ts = segment.get_timestamp_at(seg_coord.value)
                    return (i, segment, seg_ts)

        raise ValueError(f"No segment contains coordinate {coord_val}")

    def get_segment_by_index(self, index: int) -> tuple[Coordinate, Timeline]:
        """Return a segment's offset and timeline by zero-based index."""
        if index < 0 or index >= len(self._segment_order):
            raise IndexError(f"Segment index {index} out of range")

        seg_id = self._segment_order[index]
        return (self._child_offsets[seg_id], self._children[seg_id])

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return len(self._segment_order)

    def list_segments(self) -> list[str]:
        """Return segment IDs in insertion order."""
        return list(self._segment_order)

    def has_segment(self, segment_id: str) -> bool:
        """Return whether a segment ID exists."""
        return segment_id in self._segment_order

    def __contains__(self, item: Any) -> bool:
        """Return whether a region, child, or segment belongs to this line."""
        if isinstance(item, str):
            return (
                item in self._regions
                or item in self._children
                or item in self._segment_order
            )
        return super().__contains__(item)

    def iter_segments(self) -> Iterator[tuple[int, Coordinate, Timeline]]:
        """Yield each segment's index, offset, and timeline."""
        for i, seg_id in enumerate(self._segment_order):
            yield (i, self._child_offsets[seg_id], self._children[seg_id])

    def get_slice(
        self,
        start: CoordinateSpec,
        end: CoordinateSpec,
        *,
        truncate_events: bool = True,
        include_children: bool = True,
        copy_cmaps: bool = True,
    ) -> "Timeline":
        """Extract a slice while preserving the parameterized line class."""
        # Coerce to native number type
        start_value = self._resolve_axis_value(start)
        end_value = self._resolve_axis_value(end)
        nt = self._number_type
        if nt == NumberType.fraction:
            s = Fraction(start_value)
            e = Fraction(end_value)
        elif nt == NumberType.int:
            s = int(start_value)
            e = int(end_value)
        else:
            s = float(start_value)
            e = float(end_value)

        # Validate
        if s >= e:
            raise ValueError(f"start ({s}) must be less than end ({e})")
        if s < 0:
            raise ValueError(
                f"start ({s}) is outside timeline bounds " f"[0, {self._length.value})"
            )
        if e > self._length.value:
            raise ValueError(
                f"end ({e}) is outside timeline bounds " f"[0, {self._length.value}]"
            )

        sliced = type(self)(
            length=0,
            unit=self._unit,
            number_type=self._number_type,
        )

        # Copy events (SegmentLines usually have none, but handle it)
        self._copy_events_to_slice(sliced, s, e, truncate_events)

        # Recursively slice children (segments)
        if include_children:
            self._copy_children_to_slice(sliced, s, e)

        # Copy conversion maps
        if copy_cmaps:
            self._copy_cmaps_to_slice(sliced, s, e)

        # If no children were added (slice falls between segments) the
        # length is still 0.  Set it to the expected slice length so
        # the caller gets a correctly-sized timeline.
        expected_length = e - s
        if sliced._length.value < expected_length:
            sliced._length = Coordinate(expected_length, self._unit)

        return sliced

    def _typed_class(self) -> type[Timeline]:
        """Return a line parameterized by the recursively typed child class."""
        if self._children:
            first_child = next(iter(self._children.values()))
            child_class = first_child._typed_class()
            return SegmentLine[child_class]
        if type(self) is not SegmentLine:
            return type(self)
        return Timeline.resolve_subclass(self.unit, self.number_type)

    def to_timeline(self) -> Timeline:
        """Return the parameter timeline type without segment enforcement."""
        target_class = self.segment_type
        if target_class is None:
            target_class = Timeline.resolve_subclass(self.unit, self.number_type)

        initial_length = (
            0 if issubclass(target_class, SegmentLineMixin) else self._length.value
        )
        timeline = target_class(
            length=initial_length,
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
            if event.get("event_type") != "Segment"
        ]
        if events:
            timeline._add_events_unchecked(events)
        timeline._external_references = self._external_references
        for child_id, child in self._children.items():
            timeline.add_child(
                child,
                offset=self._child_offsets[child_id],
                allow_expansion=isinstance(timeline, SegmentLineMixin),
            )
        for cmap in self._conversion_maps.values():
            timeline.add_conversion_map(cmap)
        timeline._regions.update(self._regions)
        timeline._flow_maps.update(self._flow_maps)
        timeline._measure_map = self._measure_map
        timeline._metric_hierarchy = self._metric_hierarchy
        timeline._adopt_flow_control(self._flow_control.copy())
        timeline._id_minted = self._id_minted
        timeline._length = timeline._make_coordinate(self._length.value)
        timeline._locked = self._locked
        return timeline


class SegmentLine(SegmentLineMixin, Timeline):
    """A timeline whose cached parameterized classes inherit the segment API."""

    _parameter_cache: ClassVar[dict[type[Timeline], type["SegmentLine"]]] = {}

    @classmethod
    def __class_getitem__(cls, segment_type: type[Timeline]) -> type["SegmentLine"]:
        """Return the cached dynamic class for a strict Timeline subclass."""
        if cls is not SegmentLine:
            raise TypeError("Only bare SegmentLine can be parameterized")
        if (
            not isinstance(segment_type, type)
            or not issubclass(segment_type, Timeline)
            or segment_type is Timeline
            or segment_type is SegmentLine
        ):
            raise TypeError(
                "SegmentLine parameter must be a strict Timeline subclass "
                "other than bare SegmentLine"
            )

        cached = cls._parameter_cache.get(segment_type)
        if cached is not None:
            return cached

        name = f"SegmentLine[{segment_type.__name__}]"
        attributes = {"_segment_type": segment_type, "__module__": __name__}
        try:
            parameterized = type(name, (SegmentLine, segment_type), attributes)
        except TypeError:
            if not issubclass(segment_type, SegmentLine):
                raise
            parameterized = type(name, (segment_type,), attributes)
        cls._parameter_cache[segment_type] = parameterized
        return parameterized

    @classmethod
    def from_segmentation(
        cls,
        source: Timeline,
        split_coords: list[CoordinateSpec],
        copy_events: bool = True,
    ) -> "SegmentLine":
        """Create a parameterized line from slices of an unchanged source.

        Args:
            source: Timeline to segment (not modified).
            split_coords: Coordinates defining segment boundaries.
                k+1 coordinates create k segments.
            copy_events: If True, copy events to their respective segments.

        Returns:
            New SegmentLine with segments typed to match ``source``.

        Raises:
            ValueError: If source already has children (ambiguous nesting).
            ValueError: If fewer than 2 split coordinates provided.
            TypeError: If a parameterized receiver does not match the source.
        """
        if source.n_children > 0:
            raise ValueError(
                "Cannot segment a timeline that already has children. "
                "This would create conflicting nesting hierarchies."
            )

        if len(split_coords) < 2:
            raise ValueError(
                f"Segmentation requires at least 2 coordinates, got {len(split_coords)}"
            )

        coords = sorted(
            float(source._resolve_axis_value(coord)) for coord in split_coords
        )
        source_class = type(source)

        if cls is SegmentLine:
            line_class = SegmentLine[source_class]
        elif cls._segment_type is not source_class:
            raise TypeError(
                f"{cls.__name__}.from_segmentation() requires source type "
                f"{cls._segment_type.__name__}, got {source_class.__name__}"
            )
        else:
            line_class = cls

        segment_line = line_class(
            length=0,
            unit=source.unit,
            number_type=source.number_type,
        )

        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            length = end - start

            segment = source_class(
                length=length,
                unit=source.unit,
                number_type=source.number_type,
                name=f"segment_{i}",
            )

            if copy_events:
                events_in_range = source.get_events(
                    min_coord=start,
                    max_coord=end,
                )

                adjusted_events = []
                for event in events_in_range:
                    adjusted = dict(event)
                    for coord_col in ("instant", "start", "end"):
                        val = adjusted.get(coord_col)
                        if val is not None:
                            # Read the cell's exact side: taking the "value"
                            # member alone would re-slice an exact event
                            # position as the double nearest to it.
                            adjusted[coord_col] = cls._coord_to_value(val) - start
                    adjusted_events.append(adjusted)

                if adjusted_events:
                    segment.add_events(adjusted_events)

            segment_line.append_segment(segment)

        return segment_line


# endregion


# region Factory Function


def get_timeline_class(
    domain: str,
    discrete: bool = False,
) -> type[Timeline]:
    """Get the appropriate Timeline class for a domain and modality.

    Args:
        domain: One of "logical", "physical", "graphical".
        discrete: If True, return discrete variant; else continuous.

    Returns:
        The appropriate Timeline subclass.

    Raises:
        ValueError: If domain is not recognized.

    Examples:
        >>> get_timeline_class("logical", discrete=False)
        <class 'timetoalign.timelines.types.ContinuousLogicalTimeline'>
        >>> get_timeline_class("physical", discrete=True)
        <class 'timetoalign.timelines.types.DiscretePhysicalTimeline'>
    """
    classes = {
        ("logical", False): ContinuousLogicalTimeline,
        ("logical", True): DiscreteLogicalTimeline,
        ("physical", False): ContinuousPhysicalTimeline,
        ("physical", True): DiscretePhysicalTimeline,
        ("graphical", False): ContinuousGraphicalTimeline,
        ("graphical", True): DiscreteGraphicalTimeline,
    }

    key = (domain.lower(), discrete)
    if key not in classes:
        raise ValueError(f"Unknown domain '{domain}'")

    return classes[key]


# endregion


def __getattr__(name: str) -> Any:
    """Materialize a serialized dynamic class requested by pickle."""
    if name.startswith("SegmentLine[") and name.endswith("]"):
        try:
            resolved = Timeline._resolve_serialized_class_hierarchy(name)
        except (TypeError, ValueError) as error:
            raise AttributeError(name) from error
        if resolved.__name__ == name:
            return resolved
    raise AttributeError(name)
