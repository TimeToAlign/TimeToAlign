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

from typing import Any, ClassVar, Iterator

from timetoalign.core import (
    Coordinate,
    CoordinateSpec,
    TimeUnit,
)
from timetoalign.core.timestamp import _format_coordinate_value

from .base import Timeline, UnknownLengthError

# region Unit Sets by Domain

# Logical domain units (symbolic/musical time)
LOGICAL_UNITS: frozenset[TimeUnit] = frozenset(
    {
        TimeUnit.whole_notes,
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
        TimeUnit.whole_notes,
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
    def _from_dict_initial_length(cls, length: Coordinate | None) -> int:
        """Start empty so restored segments can be appended contiguously."""
        return 0

    def _finalize_from_dict(self, length: Coordinate | None) -> None:
        """Check that the rebuilt segments end exactly at the serialized length.

        The check runs between known values: a serialized length of ``None``
        is restored as unknown, and a last segment of unknown length ends at
        the serialized length, which must not precede its offset.

        Raises:
            ValueError: If the segments end short of, or beyond, the length.
        """
        if length is None:
            self._length = None
            return
        end = length.value
        last_id = self._segment_order[-1] if self._segment_order else None
        if last_id is not None and self._children[last_id].length is None:
            last_offset = self._child_offsets[last_id].value
            if end < last_offset:
                raise ValueError(
                    f"Segment line {self.id!r} has a last segment starting at "
                    f"{_format_coordinate_value(float(last_offset))}, beyond its "
                    f"length {_format_coordinate_value(float(end))}"
                )
            self._length = self._make_coordinate(end)
            return
        if self._length.value != end:
            raise ValueError(
                f"Segment line {self.id!r} has segments ending at "
                f"{_format_coordinate_value(float(self._length.value))}, but its "
                f"length is {_format_coordinate_value(float(end))}"
            )

    def _is_segment_line_instance(self) -> bool:
        """Whether this timeline is a SegmentLine (the dictionary's flag)."""
        return True

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

    def _child_extent(self, child_id: str) -> Any:
        """A segment of unknown length ends at its successor's offset.

        The last one ends at the line's length, unbounded while that is
        unknown too.
        """
        extent = super()._child_extent(child_id)
        if extent is not None:
            return extent
        index = self._segment_order.index(child_id)
        if index + 1 < len(self._segment_order):
            end = self._child_offsets[self._segment_order[index + 1]].value
        elif self._length is not None:
            end = self._length.value
        else:
            return None
        return end - self._child_offsets[child_id].value

    def _check_child_length(self, child_id: str, length: Coordinate | None) -> None:
        """Keep the segments contiguous and ending at the line's known length.

        A segment with a successor must end at the successor's offset, or be
        of unknown length and end there. The last segment must end at the
        line's length, or be of unknown length and end there; a line of
        unknown length has no end to keep. Successors never move, and the
        line never changes its length because a segment did.

        Raises:
            ValueError: If the new length breaks contiguity, or would move the
                last segment's end away from the line's known length.
        """
        index = self._segment_order.index(child_id)
        if length is None:
            return
        end = self._child_offsets[child_id].value + length.value
        if index + 1 < len(self._segment_order):
            successor = self._segment_order[index + 1]
            boundary = self._child_offsets[successor].value
            where = f"its successor '{successor}' starts at"
        elif self._length is not None:
            boundary = self._length.value
            where = f"the line '{self.id}' ends at"
        else:
            return
        if end != boundary:
            raise ValueError(
                f"SegmentLine requires contiguous segments. Segment "
                f"'{child_id}' cannot end at {_format_coordinate_value(end)}: "
                f"{where} {_format_coordinate_value(boundary)}."
            )

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
        if not self._segment_order:
            expected_offset = 0
        else:
            last_id = self._segment_order[-1]
            last_offset = self._child_offsets[last_id].value
            last_length = self._children[last_id].length
            if last_length is None:
                # A segment of unknown length ends at its successor's offset.
                if offset_val < last_offset:
                    raise ValueError(
                        f"SegmentLine requires contiguous segments. Segment "
                        f"'{last_id}' of unknown length starts at {last_offset}, "
                        f"so its successor cannot start at {offset_val}."
                    )
                return
            expected_offset = last_offset + last_length.value

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
        *,
        locked: bool | None = None,
    ) -> None:
        """Add a validated segment and record its insertion order.

        A segment is never locked, unless it is itself a segment line, which
        is locked like any child: ``locked=True`` is refused for a plain
        segment, and ``locked=False`` for a segment that is an unlocked
        segment line, since unlocking one is not supported yet.

        The line ends where its last segment ends, so a growing line whose
        new last segment is of unknown length has an unknown length too.
        """
        super().add_child(child, offset, allow_expansion, locked=locked)

    def _embed_child(
        self,
        child: Timeline,
        offset: CoordinateSpec,
        allow_expansion: bool = False,
        *,
        locked: bool,
    ) -> None:
        """Embed a segment as :meth:`add_child` does, taking the lock as given."""
        super()._embed_child(child, offset, allow_expansion, locked=locked)

        if self._segment_type is None:
            self._segment_type = type(child)
        self._segment_order.append(child.id)
        if child.length is None and (allow_expansion or not self._locked):
            self._set_length(None)

    def _append_offset(self) -> Coordinate:
        """Append after the last segment, at 0 on an empty line.

        The line's own length need not be known.

        Raises:
            UnknownLengthError: If the last segment's length is unknown, so
                that nothing says where it ends.
        """
        if not self._segment_order:
            return self._make_coordinate(0)
        last_id = self._segment_order[-1]
        last = self._children[last_id]
        if last.length is None:
            raise UnknownLengthError(last.id, f"append a segment to '{self.id}'")
        return self._make_coordinate(
            self._child_offsets[last_id].value + last.length.value
        )

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
                if segment.length is None or seg_coord.value <= segment.length.value:
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
        s, e = self._slice_bounds(start, end)

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
            sliced._length = sliced._make_coordinate(expected_length)

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
        """Return the parameter timeline type without segment enforcement.

        The result is a copy holding copies of the segments, so a change to it
        never reaches this line. The plain segments it holds as children are
        unlocked, as they were on the line.
        """
        target_class = self.segment_type
        if target_class is None:
            target_class = Timeline.resolve_subclass(self.unit, self.number_type)
        return self._recast(
            target_class,
            [
                (child_id, child._copy_tree())
                for child_id, child in self._children.items()
            ],
        )


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
    """Materialize a parameterized SegmentLine class requested by pickle.

    Pickle addresses a parameterized class by its name
    (``SegmentLine[DiscreteGraphicalTimeline]``); the parameter is resolved
    among the timeline classes by name, recursively for a nested line.
    """
    if name.startswith("SegmentLine[") and name.endswith("]"):
        try:
            return SegmentLine[
                _timeline_class_named(name.removeprefix("SegmentLine[")[:-1])
            ]
        except (TypeError, LookupError) as error:
            raise AttributeError(name) from error
    raise AttributeError(name)


def _timeline_class_named(name: str) -> type[Timeline]:
    """Return the timeline class named *name*, a parameterized line included.

    Raises:
        LookupError: If no timeline class, or several, have the name.
    """
    if name.startswith("SegmentLine[") and name.endswith("]"):
        return SegmentLine[
            _timeline_class_named(name.removeprefix("SegmentLine[")[:-1])
        ]

    def descendants(base: type[Timeline]) -> list[type[Timeline]]:
        return [
            base,
            *(cls for sub in base.__subclasses__() for cls in descendants(sub)),
        ]

    matches = {cls for cls in descendants(Timeline) if cls.__name__ == name}
    if len(matches) != 1:
        raise LookupError(name)
    return matches.pop()
