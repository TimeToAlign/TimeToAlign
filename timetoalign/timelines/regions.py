"""Region: an identified TimeInterval for timeline partitioning.

This module provides the Region class, which represents a part of a
timeline defined by a TimeInterval. Regions are NOT timelines themselves -
they cannot hold events or C-maps.

A region is identified by its ``id`` and displayed by its optional ``name``.
Use ``Timeline.create_child_from_region()`` to create a Child timeline
from a Region.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar

from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.ids import _IdentifiedElement
from timetoalign.core.time import Duration, Interval

from .manifest import (
    axis_value_from_json,
    axis_value_to_json,
    read_id,
    read_meta,
    read_name,
    require_object,
)

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
    _dict_keys: ClassVar[tuple[str, ...]] = ("id", "name", "start", "end", "meta")
    _dict_defaults: ClassVar[Mapping[str, Any]] = {"name": None, "meta": {}}

    start: Coordinate
    end: Coordinate
    name: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        unit: TimeUnit,
        number_type: NumberType,
    ) -> Region:
        """Restore a region from its object in a timeline's ``regions``.

        ``start`` and ``end`` are bare values on the timeline's axis, which
        *unit* and *number_type* describe; ``name`` and ``meta`` may be left
        out. Keys the library does not know are kept and written back.

        Args:
            data: The region object.
            unit: The unit of the timeline holding the region.
            number_type: The number type of that timeline.

        Raises:
            ValueError: If the object is malformed or a value is represented
                otherwise than the axis selects.
        """
        require_object(data, "a region", ("id", "start", "end"))
        region_id = read_id(data["id"], "a region")
        bounds = {
            key: axis_value_from_json(
                data[key], key=key, owner=region_id, unit=unit, number_type=number_type
            )
            for key in ("start", "end")
        }
        region = cls(
            id=region_id,
            name=read_name(data, f"region {region_id!r}"),
            meta=read_meta(data, f"region {region_id!r}"),
            **bounds,
        )
        region._keep_unknown_keys(data)
        return region

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

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Return this region as its JSON object, bounds as bare values.

        Args:
            minimal: Leave out ``name`` and ``meta`` at their defaults.
        """
        return self._dict_form(
            {
                "id": self.id,
                "name": self.name,
                "start": axis_value_to_json(self.start),
                "end": axis_value_to_json(self.end),
                "meta": deepcopy(self.meta),
            },
            minimal=minimal,
        )

    def __repr__(self) -> str:
        label = (
            repr(self.id) if self.name is None else f"{self.id!r}, name={self.name!r}"
        )
        return f"Region({label}, {self.start.value}-{self.end.value} {self.unit})"

    def __str__(self) -> str:
        label = self.id if self.name is None else f"{self.name} ({self.id})"
        return f"{label}: [{self.start.value}, {self.end.value}) {self.unit}"


# endregion
