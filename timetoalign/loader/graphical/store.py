"""GraphicalStore for graphical timelines.

The store combines image sources, segments, and events into a
cohesive unit for graphical timeline processing and visualization.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timetoalign.core.ids import IdMinter
from timetoalign.storage.store import EventStore

from .paths import Number
from .segment import GraphicalSegment
from .source import ImageSource, _Canvas

if TYPE_CHECKING:
    from timetoalign.timelines import DiscreteGraphicalTimeline

module_logger = logging.getLogger(__name__)


@dataclass
class GraphicalStore(EventStore):
    """Store for graphical timeline data.

    Combines:
    - Image sources (one or more)
    - Segments defining the time axis through images
    - Optional events with pixel coordinates
    - Visualization methods

    The store provides:
    - Coordinate conversion between timeline and image space
    - Segment lookup by coordinate
    - Drawing utilities for visualization
    - Timeline creation

    Attributes:
        sources: List of ImageSource objects.
        segments: List of GraphicalSegment objects (ordered by offset).
        events: Optional event data (to be implemented with EventStore).
        metadata: Additional metadata dictionary.

    Examples:
        >>> store = GraphicalStore(sources=[src1, src2], segments=[seg1, seg2])
        >>> store.total_length
        1000.0
        >>> source_idx, (x, y) = store.timeline_to_image(500.0)
    """

    sources: list[ImageSource] = field(default_factory=list)
    segments: list[GraphicalSegment] = field(default_factory=list)
    events: Any = None  # TODO: RegionEventStore when implemented
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and sort segments."""
        self._logger = module_logger.getChild("GraphicalStore")
        # Sort segments by timeline offset
        self.segments = sorted(self.segments, key=lambda s: s.timeline_offset)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over the optional graphical event table."""
        if self.events is not None:
            yield self.events

    def __len__(self) -> int:
        """Return the number of graphical event tables."""
        return int(self.events is not None)

    def __getitem__(self, name: str) -> Any:
        """Get the optional graphical event table.

        Args:
            name: The event table name, which must be ``"events"``.

        Returns:
            The graphical event table.

        Raises:
            KeyError: If the store has no event table or the name is invalid.
        """
        if name == "events" and self.events is not None:
            return self.events
        raise KeyError(f"Unknown data: {name!r}. Valid: {self.keys()}")

    def __repr__(self) -> str:
        return (
            f"GraphicalStore(sources={len(self.sources)}, "
            f"segments={len(self.segments)}, "
            f"length={self.total_length:.1f})"
        )

    @property
    def n_sources(self) -> int:
        """Number of image sources."""
        return len(self.sources)

    @property
    def n_segments(self) -> int:
        """Number of segments."""
        return len(self.segments)

    @property
    def total_length(self) -> float:
        """Total timeline length in pixels."""
        if not self.segments:
            return 0.0
        return max(s.timeline_end for s in self.segments)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Iterate over the optional graphical event table."""
        if self.events is not None:
            yield ("events", self.events)

    def keys(self) -> tuple[str, ...]:
        """Return the optional graphical event table name."""
        return ("events",) if self.events is not None else ()

    # --- Coordinate Conversion ---

    def get_segment_for_coord(self, coord: float) -> GraphicalSegment | None:
        """Find segment containing a timeline coordinate.

        Args:
            coord: Timeline coordinate.

        Returns:
            The segment containing the coordinate, or None.
        """
        for seg in self.segments:
            if seg.contains_coord(coord):
                return seg
        return None

    def get_segment_index_for_coord(self, coord: float) -> int | None:
        """Find index of segment containing a timeline coordinate.

        Args:
            coord: Timeline coordinate.

        Returns:
            Segment index, or None if not found.
        """
        for i, seg in enumerate(self.segments):
            if seg.contains_coord(coord):
                return i
        return None

    def timeline_to_image(
        self,
        coord: float,
    ) -> tuple[int, tuple[Number, Number]]:
        """Convert timeline coordinate to (source_index, (x, y)).

        Args:
            coord: Timeline coordinate.

        Returns:
            Tuple of (source_index, (x, y)) in image pixel coordinates.

        Raises:
            ValueError: If coordinate is not in any segment.
        """
        seg = self.get_segment_for_coord(coord)
        if seg is None:
            raise ValueError(f"Coordinate {coord} not in any segment")
        return seg.to_image(coord)

    def image_to_timeline(
        self,
        source_index: int,
        x: float,
        y: float,
    ) -> float | None:
        """Convert image coordinate to timeline coordinate.

        Args:
            source_index: Index of the source image.
            x: X coordinate in image.
            y: Y coordinate in image.

        Returns:
            Timeline coordinate, or None if point not on any path.
        """
        for seg in self.segments:
            result = seg.from_image(x, y, source_index=source_index)
            if result is not None:
                return result
        return None

    def get_segments_for_source(self, source_index: int) -> list[GraphicalSegment]:
        """Get all segments associated with a source image.

        Args:
            source_index: Index of the source image.

        Returns:
            List of segments from that source.
        """
        return [s for s in self.segments if s.source_index == source_index]

    # --- Timeline Creation ---

    def create_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
        *,
        minter: IdMinter | None = None,
    ) -> "DiscreteGraphicalTimeline":
        """Create DiscreteGraphicalTimeline from this bundle.

        Args:
            uid: Optional timeline ID.
            name: Optional timeline name.
            minter: The minter of the timeline's ID when no *uid* is given
                (a loader's); without either, the timeline mints its own.

        Returns:
            A DiscreteGraphicalTimeline with length = total_length.
        """
        from timetoalign.core import TimeUnit
        from timetoalign.timelines import DiscreteGraphicalTimeline
        from timetoalign.timelines.factory import mint_timeline_id

        if uid is None and minter is not None:
            uid = mint_timeline_id(TimeUnit.pixels, minter)
        return DiscreteGraphicalTimeline(
            length=int(self.total_length),
            unit="pixels",
            uid=uid,
            name=name,
        )

    # --- Visualization ---

    def draw_segments_on_source(
        self,
        source_index: int,
        color: tuple[int, int, int] = (0, 255, 0),
        line_width: int = 2,
    ) -> ImageSource:
        """Draw all segments for a source on a copy of that source.

        Args:
            source_index: Index of the source image.
            color: RGB color for paths.
            line_width: Line width in pixels.

        Returns:
            New ImageSource with paths drawn.
        """
        if source_index >= len(self.sources):
            raise IndexError(f"Source index {source_index} out of range")

        canvas = _Canvas(self.sources[source_index].copy())
        for seg in self.get_segments_for_source(source_index):
            canvas.path(seg.path, color, line_width, samples=100)
        return canvas.result()

    def draw_interval(
        self,
        start_coord: float,
        end_coord: float,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 3,
    ) -> list[ImageSource]:
        """Draw a timeline interval on all affected source images.

        The interval may span multiple segments and sources.

        Args:
            start_coord: Start of interval on timeline.
            end_coord: End of interval on timeline.
            color: RGB color for the interval.
            line_width: Line width in pixels.

        Returns:
            List of ImageSource objects with interval drawn.
        """
        # One canvas per affected source: every line is drawn on one copy
        canvases: dict[int, _Canvas] = {}

        for seg in self.segments:
            # Check if segment overlaps with interval
            seg_start = seg.timeline_offset
            seg_end = seg.timeline_end

            # Compute overlap
            overlap_start = max(start_coord, seg_start)
            overlap_end = min(end_coord, seg_end)

            if overlap_start >= overlap_end:
                continue  # No overlap

            src_idx = seg.source_index
            if src_idx not in canvases:
                canvases[src_idx] = _Canvas(self.sources[src_idx].copy())

            # Draw the overlapping portion
            # Sample points along the path within the overlap
            samples = max(10, int(overlap_end - overlap_start))
            points = []

            for i in range(samples + 1):
                t = overlap_start + (i / samples) * (overlap_end - overlap_start)
                _, (x, y) = seg.to_image(t)
                points.append((x, y))

            # Draw lines between consecutive points
            canvas = canvases[src_idx]
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                canvas.line(x0, y0, x1, y1, color, line_width)

        # Return all sources (unaffected ones unchanged)
        return [
            canvases[i].result() if i in canvases else src.copy()
            for i, src in enumerate(self.sources)
        ]

    def draw_rectangle_at_coord(
        self,
        coord: float,
        width: float,
        height: float,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
    ) -> ImageSource:
        """Draw a rectangle centered at a timeline coordinate.

        Args:
            coord: Timeline coordinate for rectangle center.
            width: Rectangle width in pixels.
            height: Rectangle height in pixels.
            color: RGB color.
            line_width: Line width.

        Returns:
            ImageSource with rectangle drawn.
        """
        source_idx, (x, y) = self.timeline_to_image(coord)

        # Center the rectangle
        rect_x = x - width / 2
        rect_y = y - height / 2

        return self.sources[source_idx].draw_rectangle(
            rect_x,
            rect_y,
            width,
            height,
            color=color,
            line_width=line_width,
        )

    def visualize_all_segments(
        self,
        output_dir: Path | None = None,
        path_color: tuple[int, int, int] = (0, 255, 0),
    ) -> list[ImageSource]:
        """Create visualization images showing all segment paths.

        Args:
            output_dir: If provided, saves images to this directory.
            path_color: RGB color for segment paths.

        Returns:
            List of ImageSource objects with paths drawn.
        """
        results = []

        for i in range(len(self.sources)):
            img = self.draw_segments_on_source(i, color=path_color)
            results.append(img)

            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                img.save(output_dir / f"segments_source_{i}.png")

        return results

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (sources not included)."""
        return {
            "n_sources": len(self.sources),
            "segments": [s.to_dict() for s in self.segments],
            "total_length": self.total_length,
            "metadata": self.metadata,
        }

    def summary(self) -> dict[str, Any]:
        """Extend the store summary with graphical-level facts.

        Returns:
            Summary with table information and graphical-level facts.

        Examples:
            >>> store = GraphicalStore(sources=[src], segments=[seg])
            >>> store.summary()
            {
                "tables": {},
                "facts": {
                    "n_sources": 1,
                    "n_segments": 1,
                    "total_length": 500.0,
                },
            }
        """
        result = super().summary()
        result["facts"].update(
            {
                "n_sources": len(self.sources),
                "n_segments": len(self.segments),
                "total_length": self.total_length,
                **self.metadata,
            }
        )
        return result
