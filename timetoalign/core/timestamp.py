"""Unified TimeStamp: Cross-section through timeline hierarchies.

This module provides Stamp, TimeStamp, and TimeIntervalStamp, the unified mechanism
for coordinate resolution across both Timeline (with children) and
TimelineGroup (with member timelines).

Design rationale (from unified_timestamp_architecture.md):
- TimeStamp is a lightweight view object that computes coordinates on access
- InterpolationMap gives TimelineGroup members O(log n) coordinate conversion
- Every attached C-Map (whatever its concrete type) is called directly via
  the shared ConversionMap interface, so unit-based conversions honor each
  map's own interpolation kind and extrapolation policy
- Works identically for Timeline.get_timestamp_at() and TimelineGroup.get_timestamp_at()

Key insight: parent<->child coordinates use exact offset arithmetic;
timeline<->group and unit-based conversions are resolved through whichever
ConversionMap was registered for that relationship.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Protocol,
    Sequence,
    runtime_checkable,
)

import pandas as pd

from ..maps.base import ConversionMap
from .enums import NumberType, TimeUnit
from .fields import field_metadata
from .retrieval import (
    CoordinateFormat,
    CoordinateResult,
    KeyCollection,
    Rounding,
    classify_dispatch_input,
    coordinate_to_dict,
    format_coordinates,
    number_type_for_converted_unit,
    validate_key_collection,
)
from .time import (
    Coordinate,
    Duration,
    IdCoordinate,
    IdDuration,
    Interval,
    express_as,
    format_number,
)

if TYPE_CHECKING:
    import pyarrow as pa

    from ..core.enums import ColumnNaming
    from ..maps.interpolation import InterpolationMap
else:
    from ..maps.interpolation import InterpolationMap

module_logger = logging.getLogger(__name__)


# Type alias for flexible conversion_maps parameters.
# Accepts: True (all), single cmap/str, or a list of cmaps/strs.
ConversionMapsSpec = (
    bool
    | str
    | TimeUnit
    | ConversionMap[Any]
    | list[ConversionMap[Any] | str | TimeUnit]
    | None
)


def _conversion_map_enabled_for_spec(
    cmap: "ConversionMap[Any]", spec: "ConversionMapsSpec"
) -> bool:
    """Return whether a ``conversion_maps`` *spec* surfaces *cmap*.

    Pure predicate over a spec and a single map — shared by every
    :class:`Stamp` and by table assembly, which holds a spec but no stamp.
    """
    if spec is True:
        return True
    if spec is False or spec is None:
        return False
    requested = spec if isinstance(spec, list) else [spec]
    for allowed in requested:
        if isinstance(allowed, TimeUnit):
            if cmap.target_unit == allowed:
                return True
        elif isinstance(allowed, str):
            try:
                if cmap.target_unit is not None and TimeUnit(allowed) == (
                    cmap.target_unit
                ):
                    return True
            except ValueError:
                pass
            if cmap.matches_selector(allowed):
                return True
        elif isinstance(allowed, ConversionMap):
            if allowed is cmap or allowed.id == cmap.id:
                return True
    return False


# region Coordinate Formatting


def _names_discrete_unit(unit_str: str) -> bool:
    """Whether *unit_str* names a unit whose values must display as integers.

    Displays receive units as free text — a unit name, an alias like ``px``,
    a C-Map's label, or nothing at all — so the name is resolved through
    :class:`TimeUnit` rather than matched against a list. Anything that is
    not a unit is simply not discrete.
    """
    try:
        return TimeUnit(unit_str.strip()).is_discrete
    except ValueError:
        return False


def _format_coordinate_value(value: int | float | Fraction, unit_str: str = "") -> str:
    """Format a coordinate value without rounding or scientific notation.

    Rules:
    - Discrete units (ticks, samples, pixels, frames): Always integer
    - Exact ratios: their own notation (``2/3``), never a decimal that
      would misrepresent them as terminating
    - Continuous floats: Their shortest lossless representation in fixed-point
      notation
    - Exact integers: Show as integer (no decimal point)

    Args:
        value: The numeric coordinate value.
        unit_str: The unit name (used to detect discrete vs continuous).

    Returns:
        Formatted string, never in scientific notation.
    """
    suffix = f" {unit_str}" if unit_str else ""
    return format_number(value, discrete=_names_discrete_unit(unit_str)) + suffix


def _format_stamp_value(value: Any, unit_str: str = "") -> str:
    """Format any C-Map output for display.

    Numeric outputs reuse :func:`_format_coordinate_value` (optionally
    suffixed with a unit); every other output -- string labels, mappings,
    tuples -- renders via ``str`` so structured conversions surface intact.

    Args:
        value: The C-Map output to format.
        unit_str: Unit appended to numeric values (empty for label maps).

    Returns:
        A display string, never in scientific notation for numeric values.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Fraction):
        suffix = f" {unit_str}" if unit_str else ""
        return f"{value}{suffix}"
    if isinstance(value, (int, float)):
        return _format_coordinate_value(value, unit_str)
    return str(value)


def _format_coordinate(coordinate: Coordinate) -> str:
    """Format a coordinate's value together with its unit.

    Args:
        coordinate: Unit-bearing coordinate to display.

    Returns:
        The coordinate's shared stamp formatting with its unit suffix.
    """
    return _format_stamp_value(coordinate.value, coordinate.unit.value)


# endregion


def _as_float_lane(value: Any) -> Any:
    """Coerce a numeric value to the float lane, passing anything else through.

    Typed retrieval and conversion paths preserve the coordinate's declared
    representation. Internal tabular and interpolation lanes use floats where
    their storage or numerical algorithms require them. Non-numeric conversion
    outputs -- labels and mappings -- pass through untouched.
    """
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, Fraction):
        return float(value)
    return value


# region TimeStampSource Protocol


@runtime_checkable
class TimeStampSource(Protocol):
    """Protocol for objects that can be TimeStamp sources.

    Both Timeline and TimelineGroup implement this protocol,
    allowing TimeStamp to work with either.
    """

    @property
    def id(self) -> str:
        """Unique identifier."""
        ...

    def _get_interpolation_map(
        self, target_id: str, source_id: str | None = None
    ) -> "InterpolationMap | None":
        """Get InterpolationMap for coordinate conversion to target.

        Args:
            target_id: Target timeline/entity ID.
            source_id: Source timeline ID (optional, for multi-source containers).

        Returns:
            InterpolationMap for conversion, or None if not available.
        """
        ...

    def _get_unit_map(self, unit: "TimeUnit") -> "Any":
        """Get a map for unit-based conversion (InterpolationMap or ConversionMap)."""
        ...

    def _get_unit_map_for_timeline(self, timeline_id: str, unit: "TimeUnit") -> "Any":
        """Get a unit map associated with a specific timeline."""
        ...

    def _get_unit_maps_for_timeline(
        self, timeline_id: str, unit: "TimeUnit"
    ) -> "list[Any]":
        """Get every unit map associated with a specific timeline."""
        ...

    def _get_related_timeline_ids(self) -> list[str]:
        """Get IDs of the direct related timelines (children/members)."""
        ...

    def _get_descendant_timeline_ids(self) -> list[str]:
        """Get IDs of every timeline in the subtree (descendants/members).

        Unlike ``_get_related_timeline_ids`` (direct relations only), this
        reaches the full subtree so a timestamp can surface conversions
        registered at any depth.
        """
        ...

    def _get_conversion_maps_for_timeline(
        self, timeline_id: str
    ) -> "list[ConversionMap[Any]]":
        """Get every conversion map attached to a timeline in the subtree.

        Returns C-Maps of all kinds, including those with no ``target_unit``
        (label and structured-value maps), so timestamps expose them.
        """
        ...

    def _get_available_units(self) -> list["TimeUnit"]:
        """Get all units available via C-Maps across the subtree."""
        ...

    def _get_unit_for_timeline(self, timeline_id: str) -> "TimeUnit | None":
        """Get the TimeUnit for a timeline in the hierarchy.

        Args:
            timeline_id: The timeline ID to look up.

        Returns:
            The TimeUnit for the timeline, or None if not found.
        """
        ...

    def _contains_coordinate(
        self, timeline_id: str, axis: float, source_id: str | None = None
    ) -> bool:
        """Check whether *axis* falls within the span of a related timeline.

        For a child embedded at *offset* with *length*, the span on the
        parent is ``[offset, offset + length)``.  Returns ``True`` for
        the source timeline itself and for any related timeline whose
        span includes *axis*.

        Args:
            timeline_id: The related timeline to check.
            axis: The coordinate on the source timeline.
            source_id: The source timeline ID (required for TimelineGroup,
                optional for Timeline where it defaults to the parent).

        Returns:
            True if *axis* is inside the related timeline's span.
        """
        ...


# endregion


# region Stamp


class Stamp(ABC):
    """Shared typed retrieval behavior for synchronized instant stamps."""

    coordinates: dict[str, Coordinate]
    source_id: str
    source: object | None
    is_interpolated: bool

    @property
    def axis(self) -> IdCoordinate:
        """Return the source-axis coordinate with its timeline identity."""
        try:
            coordinate = self.coordinates[self.source_id]
        except KeyError as exc:
            raise ValueError(
                f"Stamp source_id {self.source_id!r} has no stored coordinate"
            ) from exc
        return IdCoordinate.from_coordinate(coordinate, self.source_id)

    @property
    def present_timelines(self) -> list[str]:
        """Return stored timeline IDs in deterministic insertion order."""
        return list(self.coordinates)

    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return one stored coordinate by timeline identity.

        Args:
            timeline_id: Stored timeline identity.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The requested scalar projection or a length-one Series.

        Raises:
            KeyError: If the timeline is absent.
        """
        if not isinstance(timeline_id, str):
            raise TypeError("get_coordinate_for requires a timeline-ID string")
        try:
            coordinate = self.coordinates[timeline_id]
        except KeyError:
            owner = getattr(self.source, "id", self.source_id)
            raise KeyError(
                f"Unknown timeline ID {timeline_id!r} on stamp from {owner!r}"
            ) from None
        identified = IdCoordinate.from_coordinate(coordinate, timeline_id)
        return format_coordinates(
            [identified],
            format=format,
            rounding=rounding,
            scalar=True,
            index=pd.Index([timeline_id]) if format == "series" else None,
            series_name="coordinate",
        )

    def get_coordinates_for(
        self,
        timeline_ids: KeyCollection,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return stored coordinates for a collection of timeline identities.

        Args:
            timeline_ids: Timeline IDs to retrieve in input order.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or one canonical-value Series.
        """
        keys, index = validate_key_collection(timeline_ids)
        identified: list[IdCoordinate] = []
        for key in keys:
            try:
                coordinate = self.coordinates[key]
            except KeyError:
                owner = getattr(self.source, "id", self.source_id)
                raise KeyError(
                    f"Unknown timeline ID {key!r} on stamp from {owner!r}"
                ) from None
            identified.append(IdCoordinate.from_coordinate(coordinate, key))
        if format == "series" and index is None:
            index = pd.Index(keys)
        return format_coordinates(
            identified,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name="coordinate",
        )

    def get_coordinate(
        self,
        at: str | KeyCollection,
        timeline_id: None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a key query to the singular or plural precise getter.

        Args:
            at: One timeline ID or a collection of timeline IDs.
            timeline_id: Must be ``None`` because ``at`` carries the key.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The singular or plural precise-getter result.
        """
        if timeline_id is not None:
            raise TypeError("Stamp.get_coordinate does not accept timeline_id")
        branch = classify_dispatch_input(at, empty_is_keys=True)
        if branch == "key":
            return self.get_coordinate_for(at, format=format, rounding=rounding)
        if branch == "keys":
            return self.get_coordinates_for(at, format=format, rounding=rounding)
        raise TypeError(
            "Stamp.get_coordinate accepts timeline keys; use a positional getter "
            "on the owning timeline or alignment object for coordinate inputs"
        )

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Serialize stored coordinates as typed rational dictionaries.

        Returns:
            Timeline IDs mapped to canonical coordinate dictionaries.
        """
        return {
            timeline_id: coordinate_to_dict(coordinate)
            for timeline_id, coordinate in self.coordinates.items()
        }

    def _on_axis(self, value: Any, timeline_id: str) -> "Coordinate":
        """Express *value* in the declared representation of *timeline_id*.

        Every coordinate this stamp hands out crosses an axis boundary, and
        the axis decides how its numbers are written — the same rule whether
        the value was queried, offset-computed, converted or interpolated.
        Preserving the declared type is what lets a caller reason about a
        fraction-canonical timeline without inspecting each value's kind;
        whether a position was estimated is a separate question, answered by
        ``is_interpolated`` rather than by a number's type.
        """
        unit = self._unit_for(timeline_id)
        declared = self._number_type_enum_for(timeline_id)
        if declared is None:
            declared = unit.default_number_type
        return Coordinate(express_as(value, declared), unit, number_type=declared)

    @staticmethod
    def _on_unit(
        value: Any, unit: TimeUnit, cmap: "ConversionMap[Any] | None" = None
    ) -> Coordinate:
        """Express a conversion result on the axis *unit* names.

        The sibling of :meth:`_on_axis` for a value crossing into a unit
        rather than onto a named timeline. **The target decides alone**: the
        map's own declared output representation where it has one, else the
        target unit's default. Where the value came from does not enter into
        it -- carrying the source axis's representation across a conversion
        is the provenance-in-the-type pattern the boundary rule exists to
        remove, and it showed up as a float-canonical reading rendered as a
        forty-digit exact ratio because the quarters axis feeding it was
        exact. Every lane that evaluates a C-Map comes through here -- the
        typed getters and the display rows alike -- because two readings of
        one map that disagree about the number they carry are
        indistinguishable from a wrong answer.

        Re-expression, not construction: a ratio landing on an integer-valued
        unit is rounded here, where quantizing is what the conversion means,
        rather than refused as a hand-written value would be.
        """
        declared = None if cmap is None else cmap.output_number_type
        if declared is None:
            declared = unit.default_number_type
        return Coordinate(express_as(value, declared), unit, number_type=declared)

    def _number_type_enum_for(self, timeline_id: str) -> "NumberType | None":
        """The declared :class:`NumberType` of *timeline_id*, if it has one."""
        name = self._number_type_for(timeline_id)
        return None if name is None else NumberType(name)

    def _number_type_for(self, timeline_id: str) -> str | None:
        """Name of the declared numeric type of *timeline_id*, if it has one.

        Subclasses that know their source's timelines override this; the
        base has nothing to look in, so it declines.
        """
        return None

    def _unit_for(self, timeline_id: str) -> TimeUnit | None:
        """Return the stored unit for a timeline, when present."""
        coordinate = self.coordinates.get(timeline_id)
        return None if coordinate is None else coordinate.unit

    def get_conversion_for(self, key: str) -> object:
        """Return a structured conversion-map value by selector.

        Args:
            key: Conversion-map selector.

        Raises:
            KeyError: Always for a stamp without conversion-map resolution.
        """
        raise KeyError(f"Unknown conversion selector {key!r} on this stamp")

    def _unit_resolution_enabled(self, unit: TimeUnit) -> bool:
        """Return whether the stamp's conversion-map spec permits a unit."""
        return True

    def _conversion_map_enabled(self, cmap: "ConversionMap[Any]") -> bool:
        """Return whether this stamp's spec surfaces *cmap*."""
        return _conversion_map_enabled_for_spec(
            cmap, getattr(self, "conversion_maps", True)
        )

    @staticmethod
    def _qualify_conversion_rows(
        collected: list[tuple[str, Any, str, str]],
    ) -> list[tuple[str, Any, str]]:
        """Qualify colliding conversion-row labels with their owning timeline id.

        Each *collected* entry is ``(label, value, suffix, owner)``. A label that
        appears once keeps its bare form; a label shared by several owners is
        prefixed ``owner:label`` so displays stay unambiguous.
        """
        counts: dict[str, int] = {}
        for label, _value, _suffix, _owner in collected:
            counts[label] = counts.get(label, 0) + 1
        rows: list[tuple[str, Any, str]] = []
        for label, value, suffix, owner in collected:
            final_label = label if counts[label] == 1 else f"{owner}:{label}"
            rows.append((final_label, value, suffix))
        return rows


# endregion


# region TimeStamp


@dataclass(frozen=True, slots=True)
class TimeStamp(Stamp):
    """A synchronized instant across a timeline hierarchy.

    Lightweight object that computes coordinates on access via InterpolationMaps.
    Works identically for Timeline (with children) and TimelineGroup (with members).

    The TimeStamp represents a cross-section through the timeline structure at
    a specific axis coordinate. Related coordinates are retrieved through the
    typed precise getters and dispatcher.

    Attributes:
        axis: The root/reference coordinate value.
        source: The Timeline or TimelineGroup this timestamp belongs to.
        source_id: ID of the source (for serialization).
        row_index: If from a table row, the index. ``-1`` if interpolated and
            ``None`` for a direct query on the source axis.

    Examples:
        >>> ts = timeline.get_timestamp_at(5.0)
        >>> ts.axis  # The root coordinate with identity
        IdCoordinate(5.0, seconds, 'audio')
        >>> ts.get_coordinate_for("child:1", format="float")
        2.5

        >>> # Convert to different unit
        >>> ts.get_unit(TimeUnit.seconds, format="float")
        10.5
    """

    coordinates: dict[str, Coordinate]
    source_id: str
    source: TimeStampSource | None = field(default=None)
    row_index: int | None = field(default=None)
    is_interpolated: bool = field(default=False)
    conversion_maps: ConversionMapsSpec = field(default=True)

    def __post_init__(self) -> None:
        """Validate canonical typed storage and source identity."""
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("TimeStamp source_id must be a non-empty string")
        if self.source_id not in self.coordinates:
            raise ValueError(
                f"TimeStamp source_id {self.source_id!r} is absent from coordinates"
            )
        normalized: dict[str, Coordinate] = {}
        for timeline_id, coordinate in self.coordinates.items():
            if not isinstance(timeline_id, str) or not timeline_id:
                raise ValueError("TimeStamp coordinate keys must be non-empty strings")
            if type(coordinate) is not Coordinate:
                raise TypeError(
                    "TimeStamp coordinates must contain plain Coordinate values; "
                    f"got {type(coordinate).__name__} for {timeline_id!r}"
                )
            normalized[timeline_id] = Coordinate(
                coordinate.value,
                coordinate.unit,
                number_type=coordinate.number_type,
            )
        object.__setattr__(self, "coordinates", normalized)

    def get_unit(
        self,
        unit: TimeUnit,
        *,
        timeline_id: str | None = None,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Convert one stored coordinate through an eligible unit map.

        Args:
            unit: Requested target unit.
            timeline_id: Optional stored axis to select explicitly.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The converted coordinate projection.

        Raises:
            KeyError: If the selected timeline is absent or no eligible map exists.
        """
        if not isinstance(unit, TimeUnit):
            raise TypeError("get_unit requires a TimeUnit")
        candidates = (
            [timeline_id] if timeline_id is not None else self._surfaceable_ids()
        )
        for candidate in candidates:
            if candidate not in self.coordinates:
                if timeline_id is not None:
                    raise KeyError(
                        f"Unknown timeline ID {candidate!r} on stamp from {self.source_id!r}"
                    )
                continue
            coordinate = self.coordinates[candidate]
            if coordinate.unit == unit:
                converted = Coordinate(
                    coordinate.value,
                    unit,
                    number_type=number_type_for_converted_unit(
                        coordinate.number_type, unit
                    ),
                )
            else:
                if self.source is None or not self._unit_resolution_enabled(unit):
                    continue
                umap = self.source._get_unit_map_for_timeline(candidate, unit)
                if umap is None:
                    continue
                converted = self._on_unit(umap._evaluate(coordinate.value), unit, umap)
            identified = IdCoordinate.from_coordinate(converted, candidate)
            return format_coordinates(
                [identified],
                format=format,
                rounding=rounding,
                scalar=True,
                series_name=candidate,
            )
        raise KeyError(
            f"No eligible conversion to {unit.value!r} on stamp from {self.source_id!r}"
        )

    def _surfaceable_ids(self) -> list[str]:
        """Timeline IDs whose C-Maps this stamp surfaces: source, then subtree.

        The source comes first, followed by every descendant (Timeline) or
        member and member-descendant (TimelineGroup). Deeper relations are
        gathered via ``_get_descendant_timeline_ids`` when the source exposes
        it, otherwise the direct relations are used.
        """
        return list(self.coordinates)

    def _coordinate_on(self, timeline_id: str) -> int | float | Fraction | None:
        """Resolve this stamp's coordinate on *timeline_id* (source or subtree).

        Returns the coordinate in its stored canonical representation so a
        conversion map receives the declared axis type.
        """
        coordinate = self.coordinates.get(timeline_id)
        return None if coordinate is None else coordinate.value

    def _conversion_rows(self) -> list[tuple[str, Any, str]]:
        """Surface every C-Map across the cross-section.

        Returns ``(label, value, suffix)`` for each conversion map attached to
        the source or any descendant present at this axis, evaluated at that
        timeline's coordinate. ``label`` is the map's target-unit name when it
        has one, else the map's name; labels that collide across owners are
        qualified with the owning timeline id. ``suffix`` is the unit appended
        to numeric values in displays (empty for label/structured maps).
        """
        getter = getattr(self.source, "_get_conversion_maps_for_timeline", None)
        if getter is None:
            return []

        collected: list[tuple[str, Any, str, str]] = []  # (label, value, suffix, owner)
        for timeline_id in self._surfaceable_ids():
            coordinate = self.coordinates.get(timeline_id)
            if coordinate is None:
                continue
            for cmap in getter(timeline_id):
                if not self._conversion_map_enabled(cmap):
                    continue
                try:
                    value = cmap._evaluate(coordinate.value)
                    if cmap.target_unit is not None:
                        value = self._on_unit(value, cmap.target_unit, cmap).value
                except Exception:
                    # A display cross-section never propagates a single map's
                    # evaluation failure (e.g. a coordinate outside a map's
                    # domain): that map is simply omitted from the row set.
                    continue
                if cmap.target_unit is not None:
                    label = cmap.target_unit.value
                    suffix = cmap.target_unit.value
                else:
                    label = cmap.name
                    suffix = ""
                collected.append((label, value, suffix, timeline_id))

        return self._qualify_conversion_rows(collected)

    def get_conversion_for(self, key: str) -> object:
        """Get the raw output of a conversion map addressed by name/selector.

        Searches the source and every descendant/member present at this axis
        for a C-Map whose name, id, selector, or target-unit name is *key*, and
        evaluates it at that timeline's coordinate.

        Args:
            key: A conversion-map name, id, selector, or target-unit name.

        Returns:
            The C-Map's output at this instant without projection.

        Raises:
            KeyError: If no enabled map matches ``key``.
        """
        getter = getattr(self.source, "_get_conversion_maps_for_timeline", None)
        if getter is None:
            raise KeyError(f"Unknown conversion selector {key!r}")
        for timeline_id in self._surfaceable_ids():
            coordinate = self.coordinates.get(timeline_id)
            if coordinate is None:
                continue
            for cmap in getter(timeline_id):
                if not self._conversion_map_enabled(cmap):
                    continue
                matches = cmap.matches_selector(key) or cmap.name == key
                if not matches and cmap.target_unit is not None:
                    matches = cmap.target_unit.value == key
                if matches:
                    value = cmap._evaluate(coordinate.value)
                    if cmap.target_unit is None:
                        return value
                    return self._on_unit(value, cmap.target_unit, cmap).value
        raise KeyError(f"Unknown conversion selector {key!r}")

    def _unit_for(self, timeline_id: str) -> "TimeUnit | None":
        """Get the unit associated with a timeline ID."""
        return Stamp._unit_for(self, timeline_id)

    def _unit_resolution_enabled(self, unit: TimeUnit) -> bool:
        """Return whether conversion-map specification permits a unit."""
        spec = self.conversion_maps
        if spec is True:
            return True
        if spec is False or spec is None:
            return False

        requested = spec if isinstance(spec, list) else [spec]
        maps = [
            cmap
            for timeline_id in self._surfaceable_ids()
            for cmap in self.source._get_unit_maps_for_timeline(timeline_id, unit)
        ]

        for allowed in requested:
            if isinstance(allowed, TimeUnit) and allowed == unit:
                return True
            if isinstance(allowed, str):
                try:
                    if TimeUnit(allowed) == unit:
                        return True
                except ValueError:
                    pass
                for cmap in maps:
                    if cmap.matches_selector(allowed):
                        return True
            elif isinstance(allowed, ConversionMap):
                if allowed.target_unit == unit:
                    return True
                if any(cmap.matches_selector(allowed.id) for cmap in maps):
                    return True
        return False

    def _number_type_for(self, timeline_id: str) -> str | None:
        """Get the numeric type name, retaining group compatibility."""
        try:
            number_type = self.source._get_number_type_for_timeline(timeline_id)
        except AttributeError:
            try:
                timelines = self.source._timelines  # type: ignore[attr-defined]
            except AttributeError:
                return None
            timeline = timelines.get(timeline_id)
            number_type = timeline.number_type if timeline is not None else None
        return number_type.name if number_type is not None else None

    def __repr__(self) -> str:
        interp = " (interpolated)" if self.is_interpolated else ""
        conversions = "".join(
            f", {label}={value!r}" for label, value, _suffix in self._conversion_rows()
        )
        return (
            f"TimeStamp(axis={self.axis}, source={self.source_id!r}"
            f"{conversions}{interp})"
        )

    def __str__(self) -> str:
        """Readable cross-section showing all reachable coordinates and units.

        Examples:
            >>> print(timeline.get_timestamp_at(25.0))
            TimeStamp @25 seconds
              audio      25 seconds
              intro      25 seconds
              verse      15 seconds
              chorus     -5 seconds
              milliseconds   25000
              samples  1200000
        """

        lines: list[str] = []

        # Header: axis value with unit
        axis_entry = self.coordinates[self.source_id]
        unit_str = axis_entry.unit.value
        lines.append(
            f"TimeStamp @{_format_coordinate_value(axis_entry.value, unit_str)}"
        )

        # Collect all entries: (label, value_str)
        entries: list[tuple[str, str]] = []

        # Source timeline
        entries.append(
            (
                self.source_id,
                _format_coordinate_value(axis_entry.value, unit_str),
            )
        )

        # Children / related timelines (skip source_id to avoid duplicate)
        for tid, coordinate in self.coordinates.items():
            if tid == self.source_id:
                continue
            entries.append(
                (tid, _format_coordinate_value(coordinate.value, coordinate.unit.value))
            )

        # C-Map conversions (all kinds, across the whole subtree)
        for label, value, suffix in self._conversion_rows():
            entries.append((label, _format_stamp_value(value, suffix)))

        # Align fields
        if entries:
            max_label = max(len(e[0]) for e in entries)
            for label, value_str in entries:
                lines.append(f"  {label:<{max_label}}  {value_str}")

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the timestamp as an HTML table showing all coordinates
        with their units, organized by timeline and C-Map.
        """
        import html

        from timetoalign.display.html import affordance_line

        def _fmt_html(value: int | float | Fraction, unit_name: str = "") -> str:
            """Format for HTML display, keeping unit separate."""
            return _format_coordinate_value(value, unit_name)

        rows = []

        # Add axis coordinate
        axis_entry = self.coordinates[self.source_id]
        axis_unit_name = axis_entry.unit.value
        rows.append(
            f"<tr><td><strong>{html.escape(self.source_id)}</strong></td>"
            f"<td style='text-align: right;'>"
            f"{_fmt_html(axis_entry.value, axis_unit_name)}</td>"
            f"<td><em>axis</em></td></tr>"
        )

        # Add related timeline coordinates (children) - skip source_id to avoid duplicate
        for tid, coordinate in self.coordinates.items():
            if tid == self.source_id:
                continue
            rows.append(
                f"<tr><td>{html.escape(tid)}</td>"
                f"<td style='text-align: right;'>"
                f"{_fmt_html(coordinate.value, coordinate.unit.value)}</td>"
                f"<td><em>child</em></td></tr>"
            )

        # Add C-Map conversions (all kinds, across the whole subtree)
        for label, value, suffix in self._conversion_rows():
            rows.append(
                f"<tr><td style='color: #666;'>{html.escape(label)}</td>"
                f"<td style='text-align: right;'>"
                f"{html.escape(_format_stamp_value(value, suffix))}</td>"
                f"<td style='color: #666;'><em>cmap</em></td></tr>"
            )

        interp_badge = (
            " <span style='background: #ffeb3b; padding: 0 4px; "
            "border-radius: 3px; font-size: 0.8em;'>interpolated</span>"
            if self.is_interpolated
            else ""
        )

        affordances = affordance_line(
            [
                "ts.get_coordinate_for(<tl_id>)",
                "ts.get_unit(<unit>)",
                "ts.get_conversion_for(<key>)",
            ]
        )

        return (
            f"<div style='font-family: monospace;'>"
            f"<strong>TimeStamp</strong>{interp_badge}"
            f"<table style='border-collapse: collapse; margin-top: 4px;'>"
            f"<thead><tr style='border-bottom: 1px solid #ccc;'>"
            f"<th style='text-align: left; padding: 2px 8px;'>ID</th>"
            f"<th style='text-align: right; padding: 2px 8px;'>Coordinate</th>"
            f"<th style='text-align: left; padding: 2px 8px;'>Type</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"{affordances}"
            f"</div>"
        )


# endregion


# region TimeIntervalStamp


@dataclass(frozen=True, slots=True)
class TimeIntervalStamp:
    """A synchronized collection of canonical typed timeline intervals.

    Args:
        intervals: Timeline-ID to interval mapping.
        source_id: Timeline identity defining the interval axis.
        source: Optional object providing conversion maps.
        is_interpolated: Whether any endpoint was estimated.
    """

    intervals: dict[str, Interval]
    source_id: str
    source: TimeStampSource | None = field(default=None)
    is_interpolated: bool = field(default=False)

    def __post_init__(self) -> None:
        """Validate canonical interval storage and source identity."""
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("TimeIntervalStamp source_id must be a non-empty string")
        if self.source_id not in self.intervals:
            raise ValueError(
                f"TimeIntervalStamp source_id {self.source_id!r} is absent from intervals"
            )
        normalized: dict[str, Interval] = {}
        for timeline_id, interval in self.intervals.items():
            if not isinstance(timeline_id, str) or not timeline_id:
                raise ValueError("Interval keys must be non-empty timeline IDs")
            if not isinstance(interval, Interval):
                raise TypeError(
                    "TimeIntervalStamp intervals must contain Interval values; "
                    f"got {type(interval).__name__} for {timeline_id!r}"
                )
            normalized[timeline_id] = interval.model_copy(deep=True)
        object.__setattr__(self, "intervals", normalized)

    @property
    def duration(self) -> Duration:
        """Return the typed duration on the source axis."""
        return self.intervals[self.source_id].duration

    @property
    def present_timelines(self) -> list[str]:
        """Return stored timeline IDs in deterministic insertion order."""
        return list(self.intervals)

    @property
    def axis(self) -> Interval:
        """Return the typed interval on the source axis."""
        return self.intervals[self.source_id]

    @property
    def start(self) -> TimeStamp:
        """Return a derived instant stamp of all interval starts."""
        return TimeStamp(
            coordinates={key: value.start for key, value in self.intervals.items()},
            source_id=self.source_id,
            source=self.source,
            is_interpolated=self.is_interpolated,
        )

    @property
    def end(self) -> TimeStamp:
        """Return a derived instant stamp of all interval ends."""
        return TimeStamp(
            coordinates={key: value.end for key, value in self.intervals.items()},
            source_id=self.source_id,
            source=self.source,
            is_interpolated=self.is_interpolated,
        )

    def get_interval(self, timeline_id: str) -> Interval:
        """Return one stored interval by timeline identity.

        Args:
            timeline_id: Stored timeline identity.

        Returns:
            The canonical typed interval.

        Raises:
            KeyError: If the timeline is absent.
        """
        if not isinstance(timeline_id, str):
            raise TypeError("get_interval requires a timeline-ID string")
        try:
            return self.intervals[timeline_id]
        except KeyError:
            raise KeyError(
                f"Unknown timeline ID {timeline_id!r} on interval stamp "
                f"from {self.source_id!r}"
            ) from None

    def get_intervals(
        self, timeline_ids: KeyCollection | None = None
    ) -> dict[str, Interval]:
        """Return selected stored intervals in requested order.

        Args:
            timeline_ids: Timeline IDs, or ``None`` for every stored interval.

        Returns:
            A new ordered interval dictionary.
        """
        if timeline_ids is None:
            return dict(self.intervals)
        keys, _ = validate_key_collection(timeline_ids)
        return {key: self.get_interval(key) for key in keys}

    def get_duration_for(self, timeline_id: str) -> IdDuration:
        """Return one stored interval duration with timeline identity.

        Args:
            timeline_id: Stored timeline identity.

        Returns:
            The canonical typed ID duration.
        """
        return IdDuration.from_duration(
            self.get_interval(timeline_id).duration, timeline_id
        )

    def get_interval_in(
        self, unit: TimeUnit, *, timeline_id: str | None = None
    ) -> Interval:
        """Convert both endpoints of one stored interval to a target unit.

        Args:
            unit: Requested target unit.
            timeline_id: Optional stored axis to select explicitly.

        Returns:
            A typed interval in the target unit.
        """
        selected = timeline_id
        start = self.start.get_unit(unit, timeline_id=selected, format="coordinate")
        end = self.end.get_unit(unit, timeline_id=selected, format="coordinate")
        if not isinstance(start, Coordinate) or not isinstance(end, Coordinate):
            raise TypeError("Internal interval conversion did not return coordinates")
        return Interval(start=start, end=end)

    def __iter__(self) -> Iterator[TimeStamp]:
        """Iterate as (start, end) pair."""
        return iter((self.start, self.end))

    def __repr__(self) -> str:
        return (
            f"TimeIntervalStamp(start={self.axis.start!r}, end={self.axis.end!r}, "
            f"source={self.source_id!r})"
        )

    def __str__(self) -> str:
        """Return a readable typed interval cross-section."""
        lines = [f"TimeIntervalStamp {self.axis}"]
        for timeline_id, interval in self.intervals.items():
            discrete = interval.unit.is_discrete
            lines.append(
                f"  {timeline_id}  "
                f"{format_number(interval.start.value, discrete=discrete)}  "
                f"{format_number(interval.end.value, discrete=discrete)} "
                f"{interval.unit.value}"
            )
        return "\n".join(lines)


# endregion


# region Timestamp Table Conversion Utilities


@dataclass(frozen=True)
class TimestampColumn:
    """One column of a timestamp table, described before it is encoded.

    Attributes:
        name: Column header.
        values: Cell values — ``Coordinate`` objects, plain numbers, an Arrow
            array, or a NumPy array — with ``None``/null for absent cells.
        unit: The unit the column's numbers are in.
        number_type: The representation the column declares.
        timeline_id: Owning timeline when the column IS a timeline's axis;
            ``None`` for a derived conversion column, which belongs to no
            single axis.
        metadata: Extra field-metadata entries (a conversion map's ID, the
            timeline a derived column was computed from).
    """

    name: str
    values: Any
    unit: TimeUnit
    number_type: NumberType
    timeline_id: str | None = None
    metadata: dict[str, str] | None = None


def build_timestamp_table(columns: Sequence[TimestampColumn]) -> "pa.Table":
    """Encode timestamp columns as semantic coordinate structs.

    Every timestamp table in the library — a timeline's, a group's, a
    bundle's match-stamp table — is assembled here, so all three carry the
    same cell shape: an ``IdCoordinateField`` for a timeline's own axis and a
    ``CoordinateField`` for a derived conversion column, both storing the
    number twice (a float64 and its integer ratio).

    Storing the ratio is what makes the table lossless. A plain ``double``
    column forces every reader to re-derive ``Fraction(value)``, which yields
    the exact dyadic of the double rather than the ratio that was authored —
    so a value written as ``5/3`` quarters comes back as
    ``7505999378950827/4503599627370496``. The struct keeps the authored
    numerator and denominator, and a parquet round trip returns them
    untouched.

    Args:
        columns: Column descriptions in output order.

    Returns:
        A PyArrow table whose fields carry unit, number-type and identity
        metadata in the versioned blob.
    """
    import pyarrow as pa

    from .fields import blob_metadata
    from .time import (
        RATIONAL_STRUCT_TYPE,
        CoordinateField,
        IdCoordinateField,
        build_number_struct_array,
    )

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for column in columns:
        values = _table_cell_values(column.values)
        if (
            isinstance(values, (pa.Array, pa.ChunkedArray))
            and values.type == RATIONAL_STRUCT_TYPE
        ):
            # Already in storage shape: re-encoding it would mean decoding to
            # a number first, and a decode-encode round trip is exactly where
            # an authored ratio turns into the dyadic of its double.
            array = (
                values.combine_chunks()
                if isinstance(values, pa.ChunkedArray)
                else values
            )
        else:
            array = build_number_struct_array(
                values,
                number_type=column.number_type,
                rounding="round",
                on_error="raise",
            )
        if column.timeline_id is not None:
            semantic: Any = IdCoordinateField.from_field(
                array,
                unit=column.unit,
                number_type=column.number_type,
                timeline_id=column.timeline_id,
                name=column.name,
            )
        else:
            semantic = CoordinateField.from_field(
                array,
                unit=column.unit,
                number_type=column.number_type,
                name=column.name,
            )
        pa_field = semantic.to_field()
        if column.metadata:
            payload = field_metadata(pa_field)
            payload.update(column.metadata)
            pa_field = pa_field.with_metadata(blob_metadata(**payload))
        arrays.append(semantic.to_pyarrow())
        fields.append(pa_field)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _table_cell_values(values: Any) -> Any:
    """Unwrap ``Coordinate`` cells so the column builder sees plain numbers."""
    if isinstance(values, (list, tuple)):
        return [
            value.value if isinstance(value, Coordinate) else value for value in values
        ]
    return values


def timestamp_table_to_dataframe(
    table: "pa.Table",
    fields: "ColumnNaming | Callable[[str, dict], str] | list[str] | None" = None,
    units: bool = True,
) -> "pd.DataFrame":
    """Convert a PyArrow timestamp table to a pandas DataFrame with proper formatting.

    This utility function processes timestamp tables (from Timeline.get_timestamp_table()
    or TimelineGroup.get_timestamp_table()) and applies field naming and type conversions.

    The table's cells are coordinate structs carrying the number twice, so
    each column is decoded to **one scalar per cell** in the representation
    the column declares — never the raw struct dictionary.

    Args:
        table: PyArrow table with field-level metadata including 'unit' and
            'timeline_id' or 'cmap_id'.
        fields: How to name the DataFrame fields. Options:
            - None or ColumnNaming.name (default): Use timeline/cmap name property,
              falling back to id if name is not available.
            - ColumnNaming.id: Use timeline/cmap id.
            - Callable[[str, dict], str]: Function taking (field_name, metadata_dict)
              and returning the new field name.
            - list[str]: Explicit list of field names (must match table length).
        units: If True (default), append units to field names like "name (unit)".

    Returns:
        pandas DataFrame with:
        - Fields named according to the ``fields`` parameter
        - Units appended if ``units=True``
        - Each column written in the representation its axis declares:
          whole numbers on an integer-locked axis (nullable ``Int64`` only
          where the column has gaps), exact ratios on a fraction-canonical
          axis, doubles on a float-canonical one

    Examples:
        >>> table = timeline.get_timestamp_table()
        >>> df = timestamp_table_to_dataframe(table, units=True)
        >>> df.columns
        Index(['axis (pixels)', 'dgt1 (pixels)', 'pixels_to_inches (inches)'])

        >>> # Use IDs instead of names
        >>> from timetoalign import ColumnNaming
        >>> df = timestamp_table_to_dataframe(table, fields=ColumnNaming.id)

        >>> # Custom field naming
        >>> df = timestamp_table_to_dataframe(
        ...     table,
        ...     fields=lambda name, meta: meta.get('timeline_id', name)
        ... )
    """
    from .enums import ColumnNaming

    # A table with no rows still has a schema, and the naming below is driven
    # entirely by that schema -- so an empty timeline yields an empty frame
    # with the right columns and dtypes rather than a structureless one.
    # Returning a bare DataFrame() here used to strand every caller that reads
    # df.columns, which is how an event-less timeline could not be exported.
    # Build field name mapping
    new_field_names: list[str] = []

    for i, data_field in enumerate(table.schema):
        field_name = data_field.name
        meta_dict = field_metadata(data_field)

        # Determine base name
        if isinstance(fields, list):
            if i < len(fields):
                base_name = fields[i]
            else:
                base_name = field_name
        elif callable(fields) and not isinstance(fields, ColumnNaming):
            base_name = fields(field_name, meta_dict)
        elif isinstance(fields, ColumnNaming) and str(fields) == "id":
            # Use timeline_id or cmap_id from metadata
            base_name = (
                meta_dict.get("timeline_id") or meta_dict.get("cmap_id") or field_name
            )
        else:  # ColumnNaming.name, None, or default
            # For now, use field name directly (names are already set by Timeline)
            # In future, could look up timeline.name property
            base_name = field_name

        # Append unit if requested
        if units:
            unit = meta_dict.get("unit")
            if unit:
                final_name = f"{base_name} ({unit})"
            else:
                final_name = base_name
        else:
            final_name = base_name

        new_field_names.append(str(final_name))

    # Decode each struct column into one scalar per cell, written the way its
    # axis writes numbers -- the struct is the storage shape, a scalar is what
    # a frame reader expects.
    data: dict[str, pd.Series] = {}
    for i, data_field in enumerate(table.schema):
        data[new_field_names[i]] = _column_on_declared_axis(
            table.column(data_field.name), field_metadata(data_field)
        )
    df = pd.DataFrame(data, columns=new_field_names)
    return df


def _column_on_declared_axis(
    column: "pa.ChunkedArray | pa.Array", metadata: dict[str, Any]
) -> "pd.Series":
    """Decode one timestamp column into the representation its axis declares.

    The axis-level counterpart of :meth:`Stamp._on_axis`, applied to a whole
    column: an int-locked axis yields whole numbers, a fraction-canonical one
    yields exact ratios, a float-canonical one yields doubles. Without it a
    group's frame reported ``12473.0`` pixels where a stamp from the same
    group reported ``12473`` -- one position with two spellings, which is the
    thing the declared type exists to prevent.

    The exact lane reads the stored ``numerator``/``denominator`` members, so
    a ratio that was authored as ``5/3`` comes back as ``5/3`` and not as the
    exact dyadic of its double. It falls back to ``Fraction(value)`` only for
    a cell that carries no ratio at all. The integer lane reads the integer
    side rather than the float mirror, so magnitudes past ``2**53`` survive,
    and uses nullable ``Int64`` only where the column actually has gaps.
    """
    import pyarrow as pa

    declared = metadata.get("number_type")
    unit = metadata.get("unit")
    if declared is None and unit:
        try:
            declared = TimeUnit(unit).default_number_type.name
        except ValueError:
            declared = None
    number_type = None if declared is None else NumberType(declared)

    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
    valid = column.is_valid().to_pylist()
    doubles = column.field("value").to_pylist()

    if number_type is NumberType.fraction:
        numerators = column.field("numerator").to_pylist()
        denominators = column.field("denominator").to_pylist()
        cells: list[Any] = []
        for position, is_valid in enumerate(valid):
            numerator = numerators[position]
            denominator = denominators[position]
            if not is_valid or (numerator is None and doubles[position] is None):
                cells.append(None)
            elif numerator is None or denominator is None:
                cells.append(Fraction(float(doubles[position])))
            else:
                cells.append(Fraction(numerator, denominator))
        return pd.Series(cells, dtype=object)

    if number_type is NumberType.int:
        # The integer side is stored exactly; reading it back through the
        # float mirror would round away magnitudes past 2**53.
        numerators = column.field("numerator").to_pylist()
        denominators = column.field("denominator").to_pylist()
        integers = [
            (
                None
                if not is_valid or numerators[position] is None
                else (
                    numerators[position]
                    if denominators[position] in (None, 1)
                    else round(Fraction(numerators[position], denominators[position]))
                )
            )
            for position, is_valid in enumerate(valid)
        ]
        series = pd.Series(integers, dtype=object)
        return series.astype("Int64" if series.isna().any() else "int64")

    values = [
        doubles[position] if is_valid else None
        for position, is_valid in enumerate(valid)
    ]
    return pd.Series(values, dtype="float64")


# endregion
