"""Region: Named TimeInterval for timeline partitioning.

This module provides the Region class, which represents a named part of a
timeline defined by a TimeInterval. Regions are NOT timelines themselves -
they cannot hold events or C-maps.

A Region is a named part of a timeline that is defined by a TimeInterval.
Regions are useful for referring to parts of a timeline by name.

Use ``Timeline.create_child_from_region()`` to create a Child timeline
from a Region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.retrieval import (
    coordinate_from_dict,
    coordinate_to_dict,
)
from timetoalign.core.time import Duration, Interval

# region Region


@dataclass(frozen=True)
class Region:
    """A named part of a timeline defined by a TimeInterval.

    Regions are useful for referring to parts of a timeline by name.

    IMPORTANT: Regions are NOT timelines. They cannot hold events or C-maps.
    Use ``Timeline.create_child_from_region(name)`` to create a Child
    timeline from a Region.

    Attributes:
        name: The region's name (e.g., "Chorus", "Verse").
        start: Start coordinate of the region.
        end: End coordinate of the region.
        meta: Additional metadata (e.g., traversal order labels).

    Examples:
        >>> from timetoalign import Coordinate, TimeUnit
        >>> start = Coordinate(16.0, TimeUnit.quarters)
        >>> end = Coordinate(32.0, TimeUnit.quarters)
        >>> region = Region("Chorus", start=start, end=end)
        >>> region.duration
        Duration(Fraction(16, 1), quarters)
        >>> region.as_interval
        Interval(start=Coordinate(Fraction(16, 1), quarters), \
end=Coordinate(Fraction(32, 1), quarters))

        A region names part of a timeline; turning it into a Child timeline
        goes through the timeline that owns it:

        >>> from timetoalign import ContinuousLogicalTimeline
        >>> tl = ContinuousLogicalTimeline(length=64)
        >>> tl.create_region("Chorus", start=16, end=32)
        Region('Chorus', 16-32 quarters)
        >>> chorus_child = tl.create_child_from_region("Chorus")
        >>> chorus_child.length
        Coordinate(Fraction(16, 1), quarters)
    """

    name: str
    start: Coordinate
    end: Coordinate
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        """Restore a region from its typed coordinate dictionary representation."""
        return cls(
            name=data["name"],
            start=coordinate_from_dict(data["start"]),
            end=coordinate_from_dict(data["end"]),
            meta=dict(data.get("meta", {})),
        )

    def __post_init__(self) -> None:
        """Validate region bounds and units."""
        if self.end.value < self.start.value:
            raise ValueError(
                f"Region end ({self.end.value}) cannot be before "
                f"start ({self.start.value})"
            )
        if self.start.unit != self.end.unit:
            raise ValueError(
                f"Region start unit ({self.start.unit}) must match "
                f"end unit ({self.end.unit})"
            )

    @property
    def unit(self) -> TimeUnit:
        """The coordinate unit of this region."""
        return self.start.unit

    @property
    def duration(self) -> Duration:
        """Exact duration of the region (end - start)."""
        return self.as_interval.duration

    @property
    def as_interval(self) -> Interval:
        """The region's extent as a typed :class:`Interval`."""
        return Interval(start=self.start, end=self.end)

    def contains(
        self, coord: int | float | Fraction | Coordinate | IdCoordinate
    ) -> bool:
        """Check if a coordinate is within this region (left-inclusive).

        Following TTA convention, intervals are [start, end) - left-inclusive,
        right-exclusive.

        Args:
            coord: The coordinate value to check.

        Returns:
            True if start <= coord < end.

        Raises:
            ValueError: If a coordinate object's unit differs from the region's.
        """
        if isinstance(coord, Coordinate):
            if coord.unit is not self.unit:
                raise ValueError(
                    f"Coordinate unit {coord.unit} does not match region unit "
                    f"{self.unit}"
                )
            value = coord.value
        else:
            value = coord
        return self.start.value <= value < self.end.value

    def overlaps(self, other: Region) -> bool:
        """Check if this region overlaps with another.

        Args:
            other: Another Region to check against.

        Returns:
            True if the regions overlap (share any coordinates).
        """
        return self.start.value < other.end.value and other.start.value < self.end.value

    def to_dict(self) -> dict[str, Any]:
        """Return this region as a JSON-safe dictionary."""
        return {
            "name": self.name,
            "start": coordinate_to_dict(self.start),
            "end": coordinate_to_dict(self.end),
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        return f"Region({self.name!r}, {self.start.value}-{self.end.value} {self.unit})"

    def __str__(self) -> str:
        return f"{self.name}: [{self.start.value}, {self.end.value}) {self.unit}"


# endregion
