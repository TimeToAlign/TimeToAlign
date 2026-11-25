"""GraphicalSegment for graphical timelines.

A segment combines an image region with a path through it,
defining how timeline coordinates map to 2D pixel coordinates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeGuard

from timetoalign.core.events import BoundingBox

from .paths import Number, TimeAxisPath

module_logger = logging.getLogger(__name__)


class _XSpanPath(Protocol):
    """A path along a horizontal span, as :class:`HorizontalLinePath` is."""

    x0: Number
    x1: Number
    y: Number


class _EndpointPath(Protocol):
    """A path between two points, as :class:`DiagonalLinePath` is."""

    start: tuple[float, float]
    end: tuple[float, float]


def _has_x_span(path: TimeAxisPath) -> TypeGuard[_XSpanPath]:
    """Whether *path* is described by ``x0``, ``x1`` and ``y``."""
    return hasattr(path, "x0")


def _has_endpoints(path: TimeAxisPath) -> TypeGuard[_EndpointPath]:
    """Whether *path* is described by its ``start`` and ``end``."""
    return hasattr(path, "start")


@dataclass
class GraphicalSegment:
    """A segment of a graphical timeline.

    Combines:
    - Reference to source image (by index)
    - Path defining the time axis through the image
    - Timeline offset (where this segment starts on the full timeline)
    - Optional crop region within the source image

    The segment handles coordinate conversion between the 1D timeline
    and 2D image space, accounting for the path shape and any region offset.

    Attributes:
        source_index: Index of source image in the source list.
        path: The TimeAxisPath defining the time axis geometry.
        timeline_offset: Where this segment starts on the full timeline.
        region: Optional crop region (x, y, width, height) in source image.
        name: Optional human-readable name for the segment.
        metadata: Additional metadata dictionary.

    Examples:
        >>> from timetoalign.loader.graphical.paths import HorizontalLinePath
        >>> path = HorizontalLinePath(x0=10, x1=500, y=100)
        >>> segment = GraphicalSegment(
        ...     source_index=0,
        ...     path=path,
        ...     timeline_offset=0.0,
        ...     name="system_1",
        ... )
        >>> segment.length
        490
        >>> segment.to_image(245.0)
        (0, (255.0, 100))
    """

    source_index: int
    path: TimeAxisPath
    timeline_offset: float = 0.0
    region: tuple[int, int, int, int] | None = None  # (x, y, width, height)
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> Number:
        """Length of this segment in timeline units."""
        return self.path.length

    @property
    def timeline_start(self) -> float:
        """Start coordinate on the full timeline."""
        return self.timeline_offset

    @property
    def timeline_end(self) -> float:
        """End coordinate on the full timeline (exclusive)."""
        return self.timeline_offset + self.length

    def contains_coord(self, coord: float) -> bool:
        """Check if timeline coordinate is within this segment.

        Uses half-open interval [start, end).

        Args:
            coord: Timeline coordinate to check.

        Returns:
            True if coordinate is in this segment.
        """
        return self.timeline_offset <= coord < self.timeline_end

    def to_image(self, coord: float) -> tuple[int, tuple[Number, Number]]:
        """Convert timeline coordinate to (source_index, (x, y)).

        Args:
            coord: Timeline coordinate.

        Returns:
            Tuple of (source_index, (x, y)) in image pixel coordinates.

        Raises:
            ValueError: If coordinate is not within this segment.
        """
        if not self.contains_coord(coord):
            raise ValueError(
                f"Coordinate {coord} not in segment "
                f"[{self.timeline_offset}, {self.timeline_end})"
            )

        # Convert to local coordinate (relative to segment start)
        local_coord = coord - self.timeline_offset

        # Get path coordinates
        x, y = self.path.to_2d(local_coord)

        # Adjust for region offset if segment is cropped from larger image
        if self.region:
            x += self.region[0]
            y += self.region[1]

        return (self.source_index, (x, y))

    def from_image(
        self,
        x: float,
        y: float,
        source_index: int | None = None,
    ) -> float | None:
        """Convert image (x, y) to timeline coordinate.

        Args:
            x: X coordinate in image.
            y: Y coordinate in image.
            source_index: If provided, only match if it equals this segment's source.

        Returns:
            Timeline coordinate, or None if point is not on path.
        """
        # Check source index if provided
        if source_index is not None and source_index != self.source_index:
            return None

        # Adjust for region offset
        local_x = x
        local_y = y
        if self.region:
            local_x -= self.region[0]
            local_y -= self.region[1]

        # Get local coordinate from path
        local_coord = self.path.from_2d(local_x, local_y)
        if local_coord is None:
            return None

        # Convert to global timeline coordinate
        return self.timeline_offset + local_coord

    def get_image_bounds(self) -> BoundingBox:
        """Get the bounding box of the path in image pixels.

        The path is sampled, so its extremes are measured positions; they are
        rounded outward to whole pixels (the floor of each minimum, the ceiling
        of each maximum), so the box holds every sampled point.

        Returns:
            Bounding box in ``pixels``, from the upper-left to the lower-right
            corner.
        """
        # Sample path to find bounds
        x_min = float("inf")
        y_min = float("inf")
        x_max = float("-inf")
        y_max = float("-inf")

        samples = max(100, int(self.length))
        for i in range(samples + 1):
            t = (i / samples) * self.length
            x, y = self.path.to_2d(t)

            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Adjust for region offset
        if self.region:
            x_min += self.region[0]
            y_min += self.region[1]
            x_max += self.region[0]
            y_max += self.region[1]

        return BoundingBox.from_corners(
            math.floor(x_min),
            math.floor(y_min),
            math.ceil(x_max),
            math.ceil(y_max),
            unit="pixels",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Note: Path serialization is basic - complex paths may need
        custom serialization.
        """
        path_data: dict[str, Any] = {
            "type": type(self.path).__name__,
        }

        # Add path-specific attributes
        path = self.path
        if _has_x_span(path):
            path_data["x0"] = path.x0
            path_data["x1"] = path.x1
            path_data["y"] = path.y
        elif _has_endpoints(path):
            path_data["start"] = path.start
            path_data["end"] = path.end

        return {
            "source_index": self.source_index,
            "path": path_data,
            "timeline_offset": self.timeline_offset,
            "region": self.region,
            "name": self.name,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        name_str = f" '{self.name}'" if self.name else ""
        return (
            f"GraphicalSegment{name_str}(source={self.source_index}, "
            f"offset={self.timeline_offset:.1f}, length={self.length:.1f})"
        )
