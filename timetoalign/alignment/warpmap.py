"""WarpMap: bidirectional coordinate warping from alignment data.

This module implements the WarpMap class, which produces a materialised
copy of a source timeline warped to a target timeline's coordinate
system. It is the final step in the alignment pipeline:

    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine -> **WarpMap**

A WarpMap wraps an `InterpolationMap` and provides:

- ``warp.get_coordinate_at(coord)``: source coordinate to typed target coordinate
- ``warp.convert_array(coords)``: vectorized source -> target conversion
- ``warp.inverse()``: a cached target -> source WarpMap
- ``materialise(source_timeline)``: produce a new Timeline with all
  contents (events, children, regions) warped to the target's
  coordinate system.

Design principle 6 from the overhaul plan:
    "WarpMap materialises a complete timeline copy (events, children,
    regions) from a MatchLine. It IS an InterpolationMap under the hood."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from timetoalign.alignment.claims import AlignmentAnchor
from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    Rounding,
    dispatch_retrieval,
    format_coordinates,
    validate_coordinate_collection,
)
from timetoalign.core.time import Coordinate, IdCoordinate
from timetoalign.maps.interpolation import InterpolationMap

if TYPE_CHECKING:
    from timetoalign.alignment.matchline import MatchLine
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


class AmbiguousWarpMapError(ValueError):
    """Raised when a source coordinate has incompatible target values."""


# Repeated anchors from chords can differ slightly while still describing one
# location. Larger spreads mean that the source coordinate is not a function
# of the target coordinate system (for example, page x positions reused on
# successive systems).
_AMBIGUITY_ABSOLUTE_TOLERANCE = 5.0
_AMBIGUITY_RELATIVE_TOLERANCE = 0.01


@dataclass(frozen=True)
class WarpMap:
    """Bidirectional coordinate warping derived from alignment data.

    A WarpMap converts coordinates from a *source* timeline to a *target*
    timeline (and back) using linear interpolation between anchor points
    extracted from a `timetoalign.MatchLine`.

    Internally it delegates to an `timetoalign.maps.interpolation.InterpolationMap`
    for O(log n) lookup.  The ``materialise()`` method produces a complete
    copy of a source `timetoalign.Timeline` with all coordinates warped
    to the target's coordinate space.

    Attributes:
        source_timeline_id: ID of the source timeline.
        target_timeline_id: ID of the target timeline.
        interpolation_map: The underlying InterpolationMap for coordinate
            conversion.
        source_unit: Canonical unit of the source timeline.
        target_unit: Canonical unit of the target timeline.
        source_number_type: Canonical source-axis number type.
        target_number_type: Canonical target-axis number type.
        n_anchors: Number of anchor points used for interpolation.

    Examples:
        >>> warp = WarpMap.from_match_line(match_line, "audio")
        >>> warp.get_coordinate_at(100.0, format="float")
        45.5
        >>> warp.inverse().get_coordinate_at(45.5, format="float")
        100.0

    See Also:
        `timetoalign.MatchLine`
        `timetoalign.InterpolationMap`
    """

    source_timeline_id: str
    target_timeline_id: str
    interpolation_map: InterpolationMap = field(repr=False)
    source_unit: TimeUnit
    target_unit: TimeUnit
    source_number_type: NumberType
    target_number_type: NumberType
    _inverse_cache: "WarpMap | None" = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate the two canonical axes against interpolation metadata."""
        if not isinstance(self.source_timeline_id, str) or not self.source_timeline_id:
            raise ValueError("WarpMap source_timeline_id must be a non-empty string")
        if not isinstance(self.target_timeline_id, str) or not self.target_timeline_id:
            raise ValueError("WarpMap target_timeline_id must be a non-empty string")
        if not isinstance(self.source_unit, TimeUnit) or not isinstance(
            self.target_unit, TimeUnit
        ):
            raise ValueError(
                "WarpMap source_unit and target_unit must be TimeUnit values"
            )
        if not isinstance(self.source_number_type, NumberType) or not isinstance(
            self.target_number_type, NumberType
        ):
            raise ValueError(
                "WarpMap source_number_type and target_number_type must be NumberType values"
            )
        self.source_unit.resolve_number_type(self.source_number_type)
        self.target_unit.resolve_number_type(self.target_number_type)
        if self.interpolation_map.source_id != self.source_timeline_id:
            raise ValueError("WarpMap source ID conflicts with interpolation metadata")
        if self.interpolation_map.target_id != self.target_timeline_id:
            raise ValueError("WarpMap target ID conflicts with interpolation metadata")
        if (
            self.interpolation_map.source_unit is not None
            and self.interpolation_map.source_unit != self.source_unit
        ):
            raise ValueError(
                "WarpMap source unit conflicts with interpolation metadata"
            )
        if (
            self.interpolation_map.target_unit is not None
            and self.interpolation_map.target_unit != self.target_unit
        ):
            raise ValueError(
                "WarpMap target unit conflicts with interpolation metadata"
            )

    # region Properties

    @property
    def n_anchors(self) -> int:
        """Number of anchor points used for interpolation."""
        return self.interpolation_map.n_anchors

    @property
    def _source_float_array(self) -> NDArray[np.floating[Any]]:
        """Return the private source interpolation array."""
        return self.interpolation_map.source_coords

    @property
    def _target_float_array(self) -> NDArray[np.floating[Any]]:
        """Return the private target interpolation array."""
        return self.interpolation_map.target_coords

    @property
    def is_invertible(self) -> bool:
        """Whether the inverse mapping is defined.

        True if target coordinates are strictly monotonic.
        """
        return self.interpolation_map.is_invertible

    # endregion

    # region Conversion / Inverse

    def _interpolate_float_array(
        self, values: NDArray[np.floating[Any]]
    ) -> NDArray[np.floating[Any]]:
        """Interpolate the private vectorized float lane."""
        return self.interpolation_map.convert_array(values)

    def _interpolate_float(self, value: int | float | Fraction) -> float:
        """Interpolate one value on the private float lane."""
        return float(self.interpolation_map(float(value)))

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Warp one source position to the target canonical axis.

        Args:
            at: Source position.
            timeline_id: Optional target-axis identity validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One target coordinate projection or a length-one Series.
        """
        if timeline_id is not None and timeline_id != self.target_timeline_id:
            raise ValueError(
                f"Result timeline ID {timeline_id!r} does not match WarpMap target "
                f"{self.target_timeline_id!r}"
            )
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        if isinstance(at, IdCoordinate):
            if at.timeline_id != self.source_timeline_id:
                raise ValueError(
                    f"Coordinate source {at.timeline_id!r} does not match WarpMap "
                    f"source {self.source_timeline_id!r}"
                )
            if at.unit != self.source_unit:
                raise ValueError(
                    f"Coordinate unit {at.unit} does not match source unit "
                    f"{self.source_unit}"
                )
            value = at.value
        elif isinstance(at, Coordinate):
            if at.unit != self.source_unit:
                raise ValueError(
                    f"Coordinate unit {at.unit} does not match source unit "
                    f"{self.source_unit}"
                )
            value = at.value
        else:
            value = at
        source = Coordinate(
            value, self.source_unit, number_type=self.source_number_type
        )
        source_float = float(source.value)
        if (
            not self._source_float_array[0]
            <= source_float
            <= self._source_float_array[-1]
        ):
            raise ValueError(
                f"Coordinate {source.value!r} is outside WarpMap source support "
                f"[{self._source_float_array[0]!r}, {self._source_float_array[-1]!r}]"
            )
        target = Coordinate(
            self._interpolate_float(source.value),
            self.target_unit,
            number_type=self.target_number_type,
        )
        result = IdCoordinate.from_coordinate(target, self.target_timeline_id)
        return format_coordinates(
            [result],
            format=format,
            rounding=rounding,
            scalar=True,
            series_name=self.target_timeline_id,
        )

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Warp a collection of source positions to the target axis.

        Args:
            at: Source positions to resolve atomically.
            timeline_id: Optional target-axis identity validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        results: list[IdCoordinate] = []
        for value in values:
            result = self.get_coordinate_at(
                value,
                timeline_id=timeline_id,
                format="id_coordinate",
                rounding=rounding,
            )
            assert isinstance(result, IdCoordinate)
            results.append(result)
        return format_coordinates(
            results,
            format=format,
            rounding=rounding,
            scalar=False,
            index=index,
            series_name=self.target_timeline_id,
            empty_number_type=self.target_number_type,
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a scalar or plural source-position warp query.

        Args:
            at: One source position or a coordinate collection.
            timeline_id: Optional target-axis identity validator.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The selected precise-getter result.
        """
        return dispatch_retrieval(
            self,
            "get_coordinate",
            "get_coordinates",
            at,
            timeline_id,
            positions_only="WarpMap.get_coordinate accepts coordinate inputs only",
            format=format,
            rounding=rounding,
        )

    def inverse(self) -> "WarpMap":
        """Return the cached inverse map with source and target swapped.

        Returns:
            A WarpMap converting target coordinates back to source coordinates.

        Raises:
            ValueError: If the target coordinates are not strictly
                monotonic (map is not invertible).
        """
        if self._inverse_cache is not None:
            return self._inverse_cache

        inverse_map = type(self)(
            source_timeline_id=self.target_timeline_id,
            target_timeline_id=self.source_timeline_id,
            interpolation_map=self.interpolation_map.inverse(),
            source_unit=self.target_unit,
            target_unit=self.source_unit,
            source_number_type=self.target_number_type,
            target_number_type=self.source_number_type,
        )
        object.__setattr__(self, "_inverse_cache", inverse_map)
        object.__setattr__(inverse_map, "_inverse_cache", self)
        return inverse_map

    # endregion

    # region Factory Methods

    @staticmethod
    def _deduplicate_coordinate_pairs(
        source_coords: NDArray[np.floating[Any]],
        target_coords: NDArray[np.floating[Any]],
        *,
        source_timeline_id: str,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Average compatible duplicate sources or reject ambiguous ones."""
        unique_sources, inverse = np.unique(source_coords, return_inverse=True)
        if len(unique_sources) == len(source_coords):
            return source_coords, target_coords

        target_span = float(np.ptp(target_coords))
        tolerance = max(
            _AMBIGUITY_ABSOLUTE_TOLERANCE,
            _AMBIGUITY_RELATIVE_TOLERANCE * target_span,
        )
        deduped_targets = np.empty(len(unique_sources), dtype=np.float64)
        for index, source_coord in enumerate(unique_sources):
            targets = target_coords[inverse == index]
            minimum = float(np.min(targets))
            maximum = float(np.max(targets))
            spread = maximum - minimum
            if spread > tolerance:
                raise AmbiguousWarpMapError(
                    f"Ambiguous WarpMap source timeline '{source_timeline_id}': "
                    f"source coordinate {source_coord:g} maps to materially "
                    f"different target coordinates ({minimum:g} to {maximum:g}; "
                    f"spread {spread:g} exceeds tolerance {tolerance:g}). "
                    "A single interpolated value is undefined; exact graph "
                    "lookups via get_matchstamp_at() remain the supported path."
                )
            deduped_targets[index] = float(np.mean(targets))

        return unique_sources, deduped_targets

    @classmethod
    def from_match_line(
        cls,
        match_line: "MatchLine",
        target_timeline_id: str,
        *,
        source_unit: "TimeUnit | None" = None,
        target_unit: "TimeUnit | None" = None,
        source_number_type: NumberType | None = None,
        target_number_type: NumberType | None = None,
    ) -> "WarpMap":
        """Build a WarpMap from a MatchLine's coordinate pairs.

        Extracts ``(source_coord, target_coord)`` pairs from the
        MatchLine for the given target timeline, deduplicates compatible
        chord coordinates, and constructs the interpolation map. A repeated
        source coordinate with materially different targets is rejected.

        Args:
            match_line: The MatchLine providing ordered stamps.
            target_timeline_id: The target timeline to warp towards.
            source_unit: Explicit source unit, which must match every anchor.
            target_unit: Explicit target unit, which must match every anchor.
            source_number_type: Explicit canonical source representation.
            target_number_type: Explicit canonical target representation.

        Returns:
            A new WarpMap.

        Raises:
            ValueError: If fewer than 2 coordinate pairs are available.
            ValueError: If ``target_timeline_id`` is not in the
                MatchLine's target timelines.
        """
        anchors = match_line.get_alignment_anchors(target_timeline_id)
        if len(anchors) < 2:
            raise ValueError(
                f"WarpMap requires at least 2 coordinate pairs, "
                f"got {len(anchors)} for target '{target_timeline_id}'. "
                f"Available targets: {match_line.target_timeline_ids()}"
            )
        return cls.from_coordinate_pairs(
            source_timeline_id=match_line.source_timeline_id,
            target_timeline_id=target_timeline_id,
            source_coords=anchors,
            source_unit=source_unit,
            target_unit=target_unit,
            source_number_type=source_number_type,
            target_number_type=target_number_type,
        )

    @classmethod
    def from_coordinate_pairs(
        cls,
        source_timeline_id: str,
        target_timeline_id: str,
        source_coords: list[IdCoordinate] | list[AlignmentAnchor],
        target_coords: list[IdCoordinate] | None = None,
        *,
        source_unit: TimeUnit | None = None,
        target_unit: TimeUnit | None = None,
        source_number_type: NumberType | None = None,
        target_number_type: NumberType | None = None,
    ) -> "WarpMap":
        """Build a WarpMap from typed coordinate pairs or alignment anchors.

        Args:
            source_timeline_id: ID of the source timeline.
            target_timeline_id: ID of the target timeline.
            source_coords: Source ``IdCoordinate`` values, or complete
                ``AlignmentAnchor`` values. Duplicate sources are averaged
                only when their targets are compatible.
            target_coords: Target ``IdCoordinate`` values when separate source
                coordinates are supplied; omitted for anchors.
            source_unit: Explicit source unit validator or inferred anchor unit.
            target_unit: Explicit target unit validator or inferred anchor unit.
            source_number_type: Explicit source representation validator or
                inferred homogeneous anchor representation.
            target_number_type: Explicit target representation validator or
                inferred homogeneous anchor representation.

        Returns:
            A new WarpMap.

        Raises:
            TypeError: If coordinate lists use unsupported or mixed scalar forms.
            ValueError: If fewer than two pairs exist, typed identities or axis
                metadata conflict, source values are ambiguous, or remaining
                source coordinates are not strictly increasing.
        """
        if not source_coords:
            raise ValueError("WarpMap requires at least 2 anchor points, got 0")
        if isinstance(source_coords[0], AlignmentAnchor):
            if target_coords is not None:
                raise TypeError(
                    "target_coords must be omitted when source_coords contains anchors"
                )
            if not all(isinstance(value, AlignmentAnchor) for value in source_coords):
                raise TypeError("source_coords must not mix anchors and coordinates")
            anchors = source_coords
            source_coordinates = [
                IdCoordinate.from_coordinate(anchor.coordinate_a, anchor.timeline_a_id)
                for anchor in anchors
            ]
            target_coordinates = [
                IdCoordinate.from_coordinate(anchor.coordinate_b, anchor.timeline_b_id)
                for anchor in anchors
            ]
        else:
            if target_coords is None:
                raise TypeError("target_coords is required for IdCoordinate inputs")
            if len(source_coords) < 2:
                raise ValueError(
                    "WarpMap requires at least 2 anchor points, "
                    f"got {len(source_coords)}"
                )
            source_coordinates = source_coords
            target_coordinates = target_coords
        if not all(isinstance(value, IdCoordinate) for value in source_coordinates):
            raise TypeError("source_coords must contain typed IdCoordinate values")
        if not all(isinstance(value, IdCoordinate) for value in target_coordinates):
            raise TypeError("target_coords must contain typed IdCoordinate values")
        if any(value.timeline_id != source_timeline_id for value in source_coordinates):
            raise ValueError("Source coordinate timeline identities do not match")
        if any(value.timeline_id != target_timeline_id for value in target_coordinates):
            raise ValueError("Target coordinate timeline identities do not match")
        first_source = source_coordinates[0]
        first_target = target_coordinates[0]
        resolved_source_unit = source_unit or first_source.unit
        resolved_target_unit = target_unit or first_target.unit
        if any(value.unit != resolved_source_unit for value in source_coordinates):
            raise ValueError("Source coordinate units do not form one homogeneous axis")
        if any(value.unit != resolved_target_unit for value in target_coordinates):
            raise ValueError("Target coordinate units do not form one homogeneous axis")
        source_types = {value.number_type for value in source_coordinates}
        target_types = {value.number_type for value in target_coordinates}
        if source_number_type is None and len(source_types) != 1:
            raise ValueError(
                "Cannot infer source number type from heterogeneous anchors"
            )
        if target_number_type is None and len(target_types) != 1:
            raise ValueError(
                "Cannot infer target number type from heterogeneous anchors"
            )
        if source_number_type is not None and source_types != {source_number_type}:
            raise ValueError("Explicit source number type conflicts with anchors")
        if target_number_type is not None and target_types != {target_number_type}:
            raise ValueError("Explicit target number type conflicts with anchors")
        resolved_source_type = source_number_type or first_source.number_type
        resolved_target_type = target_number_type or first_target.number_type
        return cls._from_float_arrays(
            source_timeline_id=source_timeline_id,
            target_timeline_id=target_timeline_id,
            source_values=[float(value.value) for value in source_coordinates],
            target_values=[float(value.value) for value in target_coordinates],
            source_unit=resolved_source_unit,
            target_unit=resolved_target_unit,
            source_number_type=resolved_source_type,
            target_number_type=resolved_target_type,
        )

    @classmethod
    def _from_float_arrays(
        cls,
        *,
        source_timeline_id: str,
        target_timeline_id: str,
        source_values: list[float] | NDArray[np.floating[Any]],
        target_values: list[float] | NDArray[np.floating[Any]],
        source_unit: TimeUnit,
        target_unit: TimeUnit,
        source_number_type: NumberType,
        target_number_type: NumberType,
    ) -> WarpMap:
        """Construct a WarpMap from validated private numeric arrays."""
        src_arr = np.asarray(source_values, dtype=np.float64)
        tgt_arr = np.asarray(target_values, dtype=np.float64)
        if len(src_arr) != len(tgt_arr):
            raise ValueError(
                "source_coords and target_coords must have same length, "
                f"got {len(src_arr)} and {len(tgt_arr)}"
            )
        src_arr, tgt_arr = cls._deduplicate_coordinate_pairs(
            src_arr,
            tgt_arr,
            source_timeline_id=source_timeline_id,
        )

        imap = InterpolationMap(
            source_coords=src_arr,
            target_coords=tgt_arr,
            source_id=source_timeline_id,
            target_id=target_timeline_id,
            source_unit=source_unit,
            target_unit=target_unit,
        )

        return cls(
            source_timeline_id=source_timeline_id,
            target_timeline_id=target_timeline_id,
            interpolation_map=imap,
            source_unit=source_unit,
            target_unit=target_unit,
            source_number_type=source_number_type,
            target_number_type=target_number_type,
        )

    # endregion

    # region Materialise

    def materialise(self, source_timeline: "Timeline") -> "Timeline":
        """Produce a new Timeline with all contents warped to target coordinates.

        Creates a complete copy of the source timeline where every
        coordinate (events, children, regions) is converted by calling this
        map. The resulting timeline:

        - Has length equal to the mapped ``source_timeline.length``
        - Preserves the source's unit (unless ``target_unit`` differs)
        - Contains warped copies of all events
        - Contains warped copies of all children (recursively)
        - Contains warped copies of all regions
        - Carries an inverse WarpMap as a ConversionMap for traceability

        Args:
            source_timeline: The timeline to warp.

        Returns:
            A new Timeline with warped coordinates.

        Raises:
            ValueError: If ``source_timeline.id`` does not match
                ``self.source_timeline_id``.
            UnknownLengthError: If the length of the source timeline or of
                any of its descendants is unknown: its extent can be neither
                checked against the support nor warped.
        """
        if source_timeline.id != self.source_timeline_id:
            raise ValueError(
                f"source_timeline.id '{source_timeline.id}' does not match "
                f"WarpMap source '{self.source_timeline_id}'"
            )
        for _, timeline in source_timeline.iter_children(include_self=True):
            timeline._require_length("materialise a warped copy")
        source_start = float(source_timeline.origin.value)
        source_end = float(source_timeline.length.value)
        if (
            source_start < self._source_float_array[0]
            or source_end > self._source_float_array[-1]
        ):
            raise ValueError(
                f"Timeline extent [{source_start!r}, {source_end!r}] is outside "
                "WarpMap source support "
                f"[{self._source_float_array[0]!r}, {self._source_float_array[-1]!r}]"
            )

        from timetoalign.timelines.base import Timeline

        # Determine the target unit and timeline class
        target_unit = self.target_unit
        warped_length = self._interpolate_float(source_timeline.length.value)

        # Choose timeline class: if target unit differs, pick the
        # appropriate typed subclass; otherwise mirror the source type.
        if target_unit != source_timeline.unit:
            from timetoalign.timelines.types import get_timeline_class

            discrete_units = {"ticks", "samples", "frames", "pixels"}
            is_discrete = target_unit.value in discrete_units
            try:
                tl_class = get_timeline_class(
                    target_unit.domain.name.lower(), discrete=is_discrete
                )
            except (ValueError, AttributeError):
                tl_class = Timeline
        else:
            tl_class = type(source_timeline)

        warped = tl_class(
            length=warped_length,
            unit=target_unit,
            number_type=self.target_number_type,
            name=f"{source_timeline.name or source_timeline.id}[warped->{self.target_timeline_id}]",
        )

        # Warp events
        self._warp_events(source_timeline, warped)

        # Warp children (recursively)
        self._warp_children(source_timeline, warped)

        # Warp regions
        self._warp_regions(source_timeline, warped)

        module_logger.debug(
            "Materialised '%s' -> '%s' (length %.3f -> %.3f)",
            source_timeline.id,
            warped.id,
            float(source_timeline.length.value),
            warped_length,
        )

        return warped

    def _warp_events(
        self,
        source: "Timeline",
        target: "Timeline",
    ) -> None:
        """Warp all non-segment events from source to target.

        Args:
            source: Source timeline with original events.
            target: Target timeline to receive warped events.
        """
        from timetoalign.timelines.base import SEGMENT_EVENT_TYPE

        warped_events: list[dict[str, Any]] = []
        for event in source.events:
            if event.get("event_type") == SEGMENT_EVENT_TYPE:
                continue

            warped = dict(event)

            # Warp coordinate fields
            for name in ("instant", "start", "end"):
                val = warped.get(name)
                if val is None:
                    continue
                if isinstance(val, dict) and "value" in val:
                    raw = float(val["value"])
                else:
                    raw = float(val)
                warped[name] = self._interpolate_float(raw)

            # Warp duration from the mapped interval endpoints.
            # This correctly handles non-linear warping.
            if warped.get("duration") is not None:
                start_val = warped.get("start")
                if start_val is not None:
                    # start_raw = float(start_val)
                    dur_raw = event.get("duration")
                    if isinstance(dur_raw, dict) and "value" in dur_raw:
                        dur_raw = float(dur_raw["value"])
                    else:
                        dur_raw = float(dur_raw)
                    # Get original start for correct duration warping
                    orig_start = event.get("start")
                    if isinstance(orig_start, dict) and "value" in orig_start:
                        orig_start = float(orig_start["value"])
                    else:
                        orig_start = float(orig_start)
                    warped["duration"] = self._interpolate_float(
                        orig_start + dur_raw
                    ) - self._interpolate_float(orig_start)
                else:
                    # Instant event with spurious duration — drop it
                    warped["duration"] = 0.0

            warped_events.append(warped)

        if warped_events:
            target.add_events(warped_events, allow_expansion=True)

    def _warp_children(
        self,
        source: "Timeline",
        target: "Timeline",
    ) -> None:
        """Warp all children from source to target (recursively).

        Each child's offset is warped and its length is scaled to
        the difference between the mapped child endpoints. Events
        within each child are warped relative to the child's new
        coordinate system using a derived sub-WarpMap.

        Args:
            source: Source timeline with original children.
            target: Target timeline to receive warped children.
        """
        for child_id, child in source._children.items():
            offset = source._child_offsets[child_id]
            offset_val = float(offset.value)
            child_length = float(child.length.value)

            warped_offset = self._interpolate_float(offset_val)
            warped_end = self._interpolate_float(offset_val + child_length)
            warped_child_length = warped_end - warped_offset

            if warped_child_length <= 0:
                module_logger.warning(
                    "Skipping child '%s': warped length is non-positive "
                    "(%.3f -> %.3f)",
                    child_id,
                    child_length,
                    warped_child_length,
                )
                continue

            # Build a sub-WarpMap that converts child-local coordinates
            # to the warped child-local coordinates.
            # Child-local source coords: [0, child_length]
            # Warped child-local coords: [0, warped_child_length]
            # We need to map through the parent warp:
            #   warped_local = map(local + offset) - map(offset)
            n_points = max(self.n_anchors, 2)
            # Sample the child's coordinate range
            child_sample_coords = np.linspace(0.0, child_length, n_points)
            parent_coords = child_sample_coords + offset_val
            warped_parent = self._interpolate_float_array(
                np.asarray(parent_coords, dtype=np.float64)
            )
            warped_child_local = warped_parent - warped_offset

            # Ensure monotonicity (should hold if parent warp is monotonic)
            if not np.all(np.diff(warped_child_local) > 0):
                module_logger.warning(
                    "Child '%s' sub-warp is not monotonic; skipping.",
                    child_id,
                )
                continue

            child_warp = WarpMap(
                source_timeline_id=child.id,
                target_timeline_id=f"{child.id}[warped]",
                interpolation_map=InterpolationMap(
                    source_coords=child_sample_coords,
                    target_coords=warped_child_local,
                    source_id=child.id,
                    target_id=f"{child.id}[warped]",
                    source_unit=child.unit,
                    target_unit=self.target_unit,
                ),
                source_unit=child.unit,
                target_unit=self.target_unit,
                source_number_type=child.number_type,
                target_number_type=self.target_number_type,
            )

            # Recursively materialise the child
            warped_child = child_warp.materialise(child)

            # Add warped child to target at the warped offset
            target.add_child(warped_child, warped_offset, allow_expansion=True)

    def _warp_regions(
        self,
        source: "Timeline",
        target: "Timeline",
    ) -> None:
        """Warp all regions from source to target.

        Args:
            source: Source timeline with original regions.
            target: Target timeline to receive warped regions.
        """
        from timetoalign.core import Coordinate
        from timetoalign.timelines.regions import Region

        target_unit = target.unit

        for region in source._regions.values():
            warped_start = self._interpolate_float(region.start.value)
            warped_end = self._interpolate_float(region.end.value)

            if warped_end <= warped_start:
                module_logger.warning(
                    "Skipping region '%s': warped interval is degenerate "
                    "(%.3f -> %.3f)",
                    region.id,
                    warped_start,
                    warped_end,
                )
                continue

            warped_region = Region(
                id=region.id,
                name=region.name,
                _minted=region._minted,
                start=Coordinate(
                    warped_start,
                    target_unit,
                    number_type=self.target_number_type,
                ),
                end=Coordinate(
                    warped_end,
                    target_unit,
                    number_type=self.target_number_type,
                ),
                meta=dict(region.meta),
            )
            target.add_region(warped_region)

    # endregion

    # region Serialization

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with timeline IDs, units, and coordinate arrays.
        """
        return {
            "source_timeline_id": self.source_timeline_id,
            "target_timeline_id": self.target_timeline_id,
            "source_coords": self.interpolation_map.source_coords.tolist(),
            "target_coords": self.interpolation_map.target_coords.tolist(),
            "source_unit": (
                self.source_unit.value if self.source_unit is not None else None
            ),
            "target_unit": (
                self.target_unit.value if self.target_unit is not None else None
            ),
            "source_number_type": self.source_number_type.name,
            "target_number_type": self.target_number_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WarpMap":
        """Deserialize from dictionary.

        Args:
            data: Dict as produced by ``to_dict()``.

        Returns:
            A new WarpMap.
        """
        from timetoalign.core.enums import TimeUnit

        if not data.get("source_unit") or not data.get("target_unit"):
            raise ValueError("Serialized WarpMap requires both axis units")
        source_unit = TimeUnit(data["source_unit"])
        target_unit = TimeUnit(data["target_unit"])

        return cls._from_float_arrays(
            source_timeline_id=data["source_timeline_id"],
            target_timeline_id=data["target_timeline_id"],
            source_values=data["source_coords"],
            target_values=data["target_coords"],
            source_unit=source_unit,
            target_unit=target_unit,
            source_number_type=NumberType(data["source_number_type"]),
            target_number_type=NumberType(data["target_number_type"]),
        )

    # endregion

    # region Display

    def __repr__(self) -> str:
        return (
            f"WarpMap(source='{self.source_timeline_id}', "
            f"target='{self.target_timeline_id}', "
            f"n_anchors={self.n_anchors})"
        )

    def __str__(self) -> str:
        return f"WarpMap({self.source_timeline_id} -> " f"{self.target_timeline_id})"

    # endregion
