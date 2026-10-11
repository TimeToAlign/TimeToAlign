"""Region: an identified TimeInterval for timeline partitioning.

This module provides the Region class, which represents a part of a
timeline defined by a TimeInterval. Regions are NOT timelines themselves -
they cannot hold events or C-maps.

A region is identified by its ``id`` and displayed by its optional ``name``.
Use ``Timeline.create_child_from_region()`` to create a Child timeline
from a Region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar

from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.ids import _IdentifiedElement
from timetoalign.core.retrieval import (
    coordinate_from_dict,
    coordinate_to_dict,
)
from timetoalign.core.time import Duration, Interval

# region Region


@dataclass(frozen=True)
class Region(_IdentifiedElement):
    """An identified part of a timeline defined by a TimeInterval.

    IMPORTANT: Regions are NOT timelines. They cannot hold events or C-maps.
    Use ``Timeline.create_child_from_region(region_id)`` to create a Child
    timeline from a Region.

    Attributes:
        start: Start coordinate of the region.
        end: End coordinate of the region.
        name: Optional display name (e.g., "Chorus"); names may repeat.
        meta: Additional metadata (e.g., traversal order labels).
        id: The region's id; minted as ``region_{N}`` when not given.

    Examples:
        >>> from timetoalign import Coordinate, TimeUnit
        >>> start = Coordinate(16.0, TimeUnit.quarters)
        >>> end = Coordinate(32.0, TimeUnit.quarters)
        >>> region = Region(start, end, name="Chorus")
        >>> region.id
        'region_1'
        >>> region.duration
        Duration(Fraction(16, 1), quarters)
        >>> region.as_interval
        Interval(start=Coordinate(Fraction(16, 1), quarters), \
end=Coordinate(Fraction(32, 1), quarters))

        A region marks part of a timeline; turning it into a Child timeline
        goes through the timeline that owns it:

        >>> from timetoalign import ContinuousLogicalTimeline
        >>> tl = ContinuousLogicalTimeline(length=64)
        >>> tl.create_region(16, 32, id="chorus", name="Chorus")
        Region('chorus', name='Chorus', 16-32 quarters)
        >>> chorus_child = tl.create_child_from_region("chorus")
        >>> chorus_child.length
        Coordinate(Fraction(16, 1), quarters)
    """

    _id_prefix: ClassVar[str] = "region"

    start: Coordinate
    end: Coordinate
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        """Restore a region from its typed coordinate dictionary representation."""
        return cls(
            id=data["id"],
            name=data.get("name"),
            start=coordinate_from_dict(data["start"]),
            end=coordinate_from_dict(data["end"]),
            meta=dict(data.get("meta", {})),
        )

    def __post_init__(self) -> None:
        """Mint or validate the id, then validate region bounds and units."""
        super().__post_init__()
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
            "id": self.id,
            "name": self.name,
            "start": coordinate_to_dict(self.start),
            "end": coordinate_to_dict(self.end),
            "meta": self.meta,
        }

    def __repr__(self) -> str:
        label = (
            repr(self.id) if self.name is None else f"{self.id!r}, name={self.name!r}"
        )
        return f"Region({label}, {self.start.value}-{self.end.value} {self.unit})"

    def __str__(self) -> str:
        label = self.id if self.name is None else f"{self.name} ({self.id})"
        return f"{label}: [{self.start.value}, {self.end.value}) {self.unit}"


# endregion
