"""AlignmentAnchor, the coordinate pair, and the axes its coordinates lie on.

- :class:`AlignmentAnchor`: one coordinate on each of two timelines.
- The axis helpers: a timeline's unit and declared number type, and the
  refusal of a coordinate that is not on its timeline's axis.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal, NamedTuple, overload

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from timetoalign.core import Coordinate, IdCoordinate, NumberType, TimeUnit
from timetoalign.core.fields import drop_from_schema, register_value_projector
from timetoalign.core.ids import quote_id
from timetoalign.core.retrieval import (
    CoordinateFormat,
    CoordinateResult,
    Rounding,
    coordinate_from_dict,
    coordinate_to_dict,
    format_coordinates,
)
from timetoalign.core.time import coordinate_cell_projector
from timetoalign.core.timestamp import _format_coordinate


class _Axis(NamedTuple):
    """A timeline's axis: the unit it counts in and the number type it declares."""

    unit: TimeUnit
    number_type: NumberType


def _axis_of(timeline: Any) -> _Axis:
    """The axis of a timeline, or of a ``(unit, number_type)`` pair.

    Args:
        timeline: A timeline (anything with ``unit`` and ``number_type``) or
            a ``(unit, number_type)`` pair.
    """
    if isinstance(timeline, tuple):
        unit, number_type = timeline
        unit = TimeUnit(unit)
        return _Axis(unit, unit.resolve_number_type(number_type))
    return _Axis(TimeUnit(timeline.unit), NumberType(timeline.number_type))


def _axes(timelines: Iterable[Any] | Mapping[str, Any]) -> dict[str, _Axis]:
    """The axes of *timelines*, keyed by timeline id.

    Args:
        timelines: Timelines (each with ``id``, ``unit`` and
            ``number_type``), or a mapping from a timeline id to a timeline
            or to its ``(unit, number_type)`` pair.
    """
    if isinstance(timelines, Mapping):
        return {str(key): _axis_of(value) for key, value in timelines.items()}
    return {timeline.id: _axis_of(timeline) for timeline in timelines}


def _refuse_off_axis(
    unit: TimeUnit,
    number_type: NumberType,
    axis: _Axis,
    *,
    timeline_id: str,
    where: str,
) -> None:
    """Refuse a coordinate that is not on its timeline's own axis.

    The one check of the module: a coordinate of a MatchClaim is in its
    timeline's unit and declared number type, or it is refused, so one
    position can never appear as two coordinates.

    Args:
        unit: The coordinate's unit.
        number_type: The coordinate's number type.
        axis: The timeline's axis.
        timeline_id: The timeline's id, for the message.
        where: Which coordinate it is, for the message.

    Raises:
        ValueError: If the unit or the number type is not the timeline's.
    """
    if unit is axis.unit and number_type is axis.number_type:
        return
    raise ValueError(
        f"{where} is in {unit.value} ({number_type.name}), but timeline "
        f"{quote_id(timeline_id)} counts in {axis.unit.value} "
        f"({axis.number_type.name}): a MatchClaim's coordinates are on its "
        f"timelines' own axes. Convert a position through the timeline's C-Map "
        f"before the MatchClaim is made."
    )


def _on_timeline(value: Any, timeline: Any, where: str) -> Coordinate:
    """Place a factory's coordinate input on *timeline*'s axis.

    A raw number is a position on the timeline: it takes the timeline's unit
    and is written in its declared number type. A coordinate in another unit
    is converted through the timeline's C-Map. A coordinate in the
    timeline's unit is kept as given, so that the check refuses one in
    another number type rather than rewriting it.

    Args:
        value: A raw number, a :class:`Coordinate` or an
            :class:`IdCoordinate` naming the timeline.
        timeline: The timeline.
        where: Which coordinate it is, for the messages.

    Raises:
        ValueError: If an ``IdCoordinate`` names another timeline, or no
            C-Map converts its unit.
    """
    if isinstance(value, IdCoordinate):
        if value.timeline_id != timeline.id:
            raise ValueError(
                f"{where} names timeline {quote_id(value.timeline_id)}, not "
                f"{quote_id(timeline.id)}"
            )
        value = value.to_coordinate()
    if isinstance(value, Coordinate):
        if value.unit is TimeUnit(timeline.unit):
            return value
        converted = timeline.get_coordinate_at(value, format="coordinate")
        if not isinstance(converted, Coordinate):
            raise TypeError(
                f"{where}: timeline {quote_id(timeline.id)} converted "
                f"{value!r} to {converted!r}, not a coordinate"
            )
        return Coordinate(
            converted.value, converted.unit, number_type=converted.number_type
        )
    return Coordinate(value, timeline.unit, number_type=timeline.number_type)


def _coordinate_to_dict(coordinate: Coordinate) -> dict[str, Any]:
    """Serialize a coordinate through the shared representation."""
    return dict(coordinate_to_dict(coordinate))


def _coordinate_from_dict(data: dict[str, Any]) -> Coordinate:
    """Restore a coordinate through the shared representation."""
    return coordinate_from_dict(data)


class AlignmentAnchor(BaseModel):
    """A coordinate pair associating one coordinate on timeline A with one
    coordinate on timeline B.

    A neutral record with no claim semantics. AlignmentAnchors are value
    objects: two anchors with the same coordinates and timeline IDs are
    equivalent regardless of how they were created.

    Each coordinate is kept as given. An anchor knows its timelines only by
    id; the claim holding it checks both coordinates against the two
    timelines (:meth:`MatchClaim.check_on_timelines`), since only the
    timeline knows its declared number type.

    Only synchronous `MatchClaim` instances produce AlignmentAnchors.
    Non-synchronous claims (conceptual matches, NOMATCH) have no anchors.

    Attributes:
        timeline_a_id: First timeline's unique identifier.
        coordinate_a: Coordinate in first timeline.
        timeline_b_id: Second timeline's unique identifier.
        coordinate_b: Coordinate in second timeline.

    Examples:
        >>> anchor = AlignmentAnchor(
        ...     timeline_a_id="score:1",
        ...     coordinate_a=Coordinate(100, TimeUnit.quarters),
        ...     timeline_b_id="recording:1",
        ...     coordinate_b=Coordinate(45.5, TimeUnit.seconds),
        ... )
        >>> anchor.get_coordinate_for("score:1")
        IdCoordinate(Fraction(100, 1), quarters, 'score:1')
    """

    model_config = ConfigDict(frozen=True)

    timeline_a_id: str
    coordinate_a: Coordinate
    timeline_b_id: str
    coordinate_b: Coordinate

    @property
    def timelines(self) -> tuple[str, str]:
        """Return tuple of timeline IDs."""
        return (self.timeline_a_id, self.timeline_b_id)

    @property
    def coordinates(self) -> tuple[Coordinate, Coordinate]:
        """Return tuple of coordinates."""
        return (self.coordinate_a, self.coordinate_b)

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["id_coordinate"] = ...,
        rounding: Rounding = ...,
    ) -> IdCoordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: Literal["coordinate"],
        rounding: Rounding = ...,
    ) -> Coordinate: ...

    @overload
    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat,
        rounding: Rounding = ...,
    ) -> CoordinateResult | pd.Series: ...

    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Get the coordinate for a specific timeline.

        Args:
            timeline_id: The timeline to get coordinate for.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The requested coordinate projection.

        Raises:
            KeyError: If the timeline is not part of this anchor.
        """
        if timeline_id == self.timeline_a_id:
            coordinate = self.coordinate_a
        elif timeline_id == self.timeline_b_id:
            coordinate = self.coordinate_b
        else:
            raise KeyError(f"Timeline {timeline_id!r} is not part of this anchor")
        return format_coordinates(
            [IdCoordinate.from_coordinate(coordinate, timeline_id)],
            format=format,
            rounding=rounding,
            scalar=True,
            index=pd.Index([timeline_id]),
            series_name="coordinate",
        )

    def connects(self, timeline_id: str) -> bool:
        """Check if this anchor connects to a specific timeline.

        Args:
            timeline_id: The timeline to check.

        Returns:
            True if the anchor involves this timeline.
        """
        return timeline_id in (self.timeline_a_id, self.timeline_b_id)

    def connects_both(self, timeline_a_id: str, timeline_b_id: str) -> bool:
        """Check if this anchor connects two specific timelines.

        Args:
            timeline_a_id: First timeline.
            timeline_b_id: Second timeline.

        Returns:
            True if the anchor connects exactly these two timelines.
        """
        return {self.timeline_a_id, self.timeline_b_id} == {
            timeline_a_id,
            timeline_b_id,
        }

    @field_serializer("coordinate_a", "coordinate_b")
    def _serialize_coordinate(self, coordinate: Coordinate) -> dict[str, Any]:
        """Emit a coordinate through the shared JSON representation."""
        return _coordinate_to_dict(coordinate)

    @field_validator("coordinate_a", "coordinate_b", mode="before")
    @classmethod
    def _parse_coordinate(cls, value: Any) -> Any:
        """Accept the serialized coordinate dict as well as a Coordinate."""
        if isinstance(value, dict) and "unit" in value:
            return _coordinate_from_dict(value)
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AlignmentAnchor":
        """Deserialize from dictionary."""
        return cls.model_validate(data)

    def __repr__(self) -> str:
        return (
            f"AlignmentAnchor("
            f"{self.timeline_a_id}@"
            f"{_format_coordinate(self.coordinate_a)}"
            f" <-> {self.timeline_b_id}@"
            f"{_format_coordinate(self.coordinate_b)})"
        )

    def __str__(self) -> str:
        """Return the anchor with exact, unit-bearing coordinates."""
        return repr(self)

    def _repr_html_(self) -> str:
        """Return an HTML representation with exact, unit-bearing coordinates."""
        import html as html_mod

        coordinate_a = _format_coordinate(self.coordinate_a)
        coordinate_b = _format_coordinate(self.coordinate_b)
        return (
            "<div style='font-family: monospace;'>"
            "<strong>AlignmentAnchor</strong>("
            f"{html_mod.escape(self.timeline_a_id)}@{html_mod.escape(coordinate_a)} "
            "&lt;-&gt; "
            f"{html_mod.escape(self.timeline_b_id)}@{html_mod.escape(coordinate_b)})"
            "</div>"
        )


# An anchor's timelines are its row's: the table cell holds its coordinates.
register_value_projector(AlignmentAnchor, "timeline_a_id", drop_from_schema)
register_value_projector(AlignmentAnchor, "timeline_b_id", drop_from_schema)
register_value_projector(AlignmentAnchor, "coordinate_a", coordinate_cell_projector)
register_value_projector(AlignmentAnchor, "coordinate_b", coordinate_cell_projector)
