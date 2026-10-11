"""MatchGraph and MatchStamp classes.

This module implements the mid-level graph structure for alignment:

- MatchStamp: Cross-group timestamp at a single coordinate
- MatchIntervalStamp: Combined endpoint resolutions for an interval query
- MatchGraph: Graph of MatchClaims yielding MatchStamps

The hierarchy is:
    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine

MatchGraph uses networkx to:
1. Build a graph with private ``(timeline_id, canonical_value)`` keys
2. Edges represent synchronous AlignmentAnchors (explicit or inferred)
3. Extend edges via Group membership (implicit claims)
4. Extract MatchStamps from connected components

Design:
    Only synchronous claims produce graph edges. Non-synchronous claims
    (conceptual matches, NOMATCH) are stored as metadata but do not create
    edges. ``extend_to_groups()`` creates implicit ``MatchClaim`` objects
    (case d) and adds their anchors as edges. Each Hendrix M-box
    (M1-M15) is a MatchGraph -- the system is NOT a global graph.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import pandas as pd

from timetoalign.alignment.claims import AlignmentAnchor, MatchClaim
from timetoalign.alignment.filters import ClaimFilter
from timetoalign.core.enums import Domain, NumberType, TimeUnit
from timetoalign.core.retrieval import (
    CoordinateCollection,
    CoordinateFormat,
    CoordinateInput,
    CoordinateResult,
    KeyCollection,
    Rounding,
    coordinate_from_dict,
    coordinate_to_dict,
    dispatch_retrieval,
    format_coordinates,
    number_type_for_converted_unit,
    validate_coordinate_collection,
    validate_key_collection,
)
from timetoalign.core.time import (
    Coordinate,
    CoordinateValue,
    IdCoordinate,
    Interval,
)
from timetoalign.core.timestamp import (
    ConversionMapsSpec,
    Stamp,
    _format_stamp_value,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines import Timeline
    from timetoalign.timelines.groups import TimelineGroup

module_logger = logging.getLogger(__name__)


# Private NetworkX key: (timeline_id, canonical coordinate value).
_GraphNode = tuple[str, CoordinateValue]


# region MatchStamp


@dataclass(frozen=True)
class MatchStamp(Stamp):
    """A synchronized timestamp across multiple timelines.

    A MatchStamp represents a single coordinate (or instant) that has been
    synchronized across multiple timelines via explicit anchors and/or
    inferred group membership.

    Contains one coordinate per participating timeline, where coordinates
    are linked via explicit anchors or inferred group membership.

    Attributes:
        coordinates: Dictionary of timeline ID to canonical coordinate.
        anchor_edges: List of (tl_a, tl_b) pairs that are explicitly anchored.
        inferred_edges: List of (tl_a, tl_b) pairs inferred via groups.
        axis: Identified source coordinate derived from storage.
        source: Bundle that produced the stamp.
        source_id: Timeline ID used for the query.
        is_interpolated: Whether the stamp used interpolated transfer.
        conversion_maps: Conversion maps available to unit lookup and display.
            Opt-in: defaults to ``False``.

    Examples:
        >>> stamp = MatchStamp(
        ...     coordinates={
        ...         "score": Coordinate(100.0, TimeUnit.quarters),
        ...         "audio": Coordinate(45.5, TimeUnit.seconds),
        ...     },
        ...     source_id="score",
        ...     anchor_edges=[("score", "audio")],
        ... )
        >>> stamp.get_coordinate_for("audio", format="float")
        45.5
    """

    coordinates: dict[str, Coordinate] = field(default_factory=dict)
    source_id: str = ""
    anchor_edges: list[tuple[str, str]] = field(default_factory=list)
    inferred_edges: list[tuple[str, str]] = field(default_factory=list)
    source: "AlignmentBundle | None" = None
    is_interpolated: bool = False
    conversion_maps: ConversionMapsSpec = False
    interval_claims: list[MatchClaim] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Isolate mutable containers from callers and serialized data."""
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("MatchStamp source_id must be a non-empty timeline ID")
        if self.source_id not in self.coordinates:
            raise ValueError(
                f"MatchStamp source_id {self.source_id!r} is absent from coordinates"
            )
        normalized: dict[str, Coordinate] = {}
        for timeline_id, coordinate in self.coordinates.items():
            if not isinstance(timeline_id, str) or not timeline_id:
                raise ValueError("MatchStamp coordinate keys must be non-empty strings")
            if type(coordinate) is not Coordinate:
                raise TypeError(
                    "MatchStamp coordinates must be plain Coordinate values"
                )
            normalized[timeline_id] = Coordinate(
                coordinate.value,
                coordinate.unit,
                number_type=coordinate.number_type,
            )
        object.__setattr__(self, "coordinates", normalized)
        object.__setattr__(
            self, "anchor_edges", [tuple(edge) for edge in self.anchor_edges]
        )
        object.__setattr__(
            self, "inferred_edges", [tuple(edge) for edge in self.inferred_edges]
        )
        normalized_claims: list[MatchClaim] = []
        for claim in self.interval_claims:
            if not isinstance(claim, MatchClaim) or not claim.is_interval:
                raise TypeError(
                    "MatchStamp interval_claims must contain interval MatchClaim values"
                )
            normalized_claims.append(claim.model_copy(deep=True))
        object.__setattr__(self, "interval_claims", normalized_claims)

    def _interval_candidate_rows(
        self,
    ) -> list[tuple[MatchClaim, str, Coordinate | None]]:
        """Return every interval claim's mapped counterpart candidate."""
        if not self.interval_claims or self.source is None:
            return []
        query = self.coordinates[self.source_id]
        rows: list[tuple[MatchClaim, str, Coordinate | None]] = []
        for claim in self.interval_claims:
            target_id, coordinate, _ = self.source._interval_claim_candidate(
                claim, self.source_id, query
            )
            rows.append((claim, target_id, coordinate))
        return rows

    def _ambiguous_interval_candidates(
        self,
    ) -> dict[str, list[tuple[MatchClaim, Coordinate | None]]]:
        """Return per-timeline candidates that do not define one coordinate."""
        grouped: dict[str, list[tuple[MatchClaim, Coordinate | None]]] = {}
        for claim, timeline_id, coordinate in self._interval_candidate_rows():
            grouped.setdefault(timeline_id, []).append((claim, coordinate))
        ambiguous: dict[str, list[tuple[MatchClaim, Coordinate | None]]] = {}
        for timeline_id, candidates in grouped.items():
            values = [coordinate for _, coordinate in candidates]
            if any(coordinate is None for coordinate in values):
                ambiguous[timeline_id] = candidates
                continue
            unique = []
            for coordinate in values:
                if coordinate not in unique:
                    unique.append(coordinate)
            if len(unique) != 1:
                ambiguous[timeline_id] = candidates
        return ambiguous

    @staticmethod
    def _claim_candidate_text(
        index: int,
        claim: MatchClaim,
        timeline_id: str,
        coordinate: Coordinate | None,
    ) -> str:
        """Format one claim candidate for display and diagnostics."""
        value = (
            "MISSING (no single coordinate)"
            if coordinate is None
            else _format_stamp_value(coordinate.value, coordinate.unit.value)
        )
        return (
            f"claim {index} {claim.timeline_a_id}<->{claim.timeline_b_id}: "
            f"{timeline_id}={value}"
        )

    def get_coordinate_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Return one coordinate, rejecting per-claim ambiguity atomically."""
        if timeline_id not in self.coordinates:
            candidates = self._ambiguous_interval_candidates().get(timeline_id)
            if candidates:
                details = "; ".join(
                    self._claim_candidate_text(index, claim, timeline_id, coordinate)
                    for index, (claim, coordinate) in enumerate(candidates, 1)
                )
                raise ValueError(
                    f"Timeline {timeline_id!r} has multiple interval-claim "
                    f"candidates: {details}"
                )
        return super().get_coordinate_for(timeline_id, format=format, rounding=rounding)

    def get_coordinates_for(
        self,
        timeline_ids: KeyCollection,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return coordinates after validating every requested timeline."""
        keys, _ = validate_key_collection(timeline_ids)
        ambiguous = self._ambiguous_interval_candidates()
        for timeline_id in keys:
            candidates = ambiguous.get(timeline_id)
            if candidates:
                details = "; ".join(
                    self._claim_candidate_text(index, claim, timeline_id, coordinate)
                    for index, (claim, coordinate) in enumerate(candidates, 1)
                )
                raise ValueError(
                    f"Timeline {timeline_id!r} has multiple interval-claim "
                    f"candidates: {details}"
                )
        return super().get_coordinates_for(
            timeline_ids, format=format, rounding=rounding
        )

    @property
    def n_timelines(self) -> int:
        """Number of timelines in this stamp."""
        return len(self.coordinates)

    @property
    def n_explicit_edges(self) -> int:
        """Number of explicitly anchored pairs."""
        return len(self.anchor_edges)

    @property
    def n_inferred_edges(self) -> int:
        """Number of inferred (via group) pairs."""
        return len(self.inferred_edges)

    def get_unit(
        self,
        unit: TimeUnit,
        *,
        timeline_id: str | None = None,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Get the query coordinate converted to a unit.

        Unit conversion is delegated to the source timeline's owning group so
        the same conversion-map selection rules as ``TimeStamp`` apply.

        Args:
            unit: The target unit.
            timeline_id: Optional stored axis to select explicitly.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The converted coordinate projection.
        """
        if not isinstance(unit, TimeUnit):
            raise TypeError("get_unit requires a TimeUnit")
        candidates = (
            [timeline_id] if timeline_id is not None else list(self.coordinates)
        )
        for candidate in candidates:
            coordinate = self.coordinates.get(candidate)
            if coordinate is None:
                if timeline_id is not None:
                    raise KeyError(f"Unknown timeline ID {candidate!r} on MatchStamp")
                continue
            if coordinate.unit == unit:
                converted = Coordinate(
                    coordinate.value,
                    unit,
                    number_type=number_type_for_converted_unit(
                        coordinate.number_type, unit
                    ),
                )
            else:
                if self.source is None:
                    continue
                cmap = self.source.get_timeline(candidate)._get_unit_map(unit)
                if cmap is None or not self._conversion_map_enabled(cmap):
                    continue
                converted = self._on_unit(cmap._evaluate(coordinate.value), unit, cmap)
            identified = IdCoordinate.from_coordinate(converted, candidate)
            return format_coordinates(
                [identified],
                format=format,
                rounding=rounding,
                scalar=True,
                series_name=candidate,
            )
        raise KeyError(f"No eligible conversion to {unit.value!r} on MatchStamp")

    def _unit_for(self, timeline_id: str) -> TimeUnit | None:
        """Get the unit associated with a timeline ID."""
        coordinate = self.coordinates.get(timeline_id)
        return None if coordinate is None else coordinate.unit

    def _conversion_rows(self) -> list[tuple[str, Any, str]]:
        """Surface every enabled C-Map across the cross-section.

        For each present timeline, evaluate each of its conversion maps at that
        timeline's own coordinate, yielding ``(label, value, suffix)`` triples
        with the same collision-qualification and formatting rules
        :class:`TimeStamp` uses. Empty when no source bundle is attached, no map
        is enabled by ``conversion_maps``, or a map raises at this coordinate.
        """
        if self.source is None:
            return []
        getter = getattr(self.source, "_get_conversion_maps_for_timeline", None)
        if getter is None:
            return []
        collected: list[tuple[str, Any, str, str]] = []
        for timeline_id, coordinate in self.coordinates.items():
            for cmap in getter(timeline_id):
                if not self._conversion_map_enabled(cmap):
                    continue
                try:
                    value = cmap._evaluate(coordinate.value)
                    if cmap.target_unit is not None:
                        value = self._on_unit(value, cmap.target_unit, cmap).value
                except Exception:
                    continue
                if cmap.target_unit is not None:
                    label = cmap.target_unit.value
                    suffix = cmap.target_unit.value
                else:
                    label = cmap.name
                    suffix = ""
                collected.append((label, value, suffix, timeline_id))
        return self._qualify_conversion_rows(collected)

    def has_timeline(self, timeline_id: str) -> bool:
        """Check if timeline is in this stamp."""
        return timeline_id in self.coordinates

    def get_conversion_for(self, key: str) -> object:
        """Return an enabled conversion-map value by selector.

        Args:
            key: Map name, ID, selector, or target-unit name.

        Returns:
            The map result without numeric projection.

        Raises:
            KeyError: If no eligible conversion matches.
        """
        if self.source is None:
            raise KeyError(f"Unknown conversion selector {key!r}")
        getter = getattr(self.source, "_get_conversion_maps_for_timeline", None)
        if getter is None:
            raise KeyError(f"Unknown conversion selector {key!r}")
        for timeline_id, coordinate in self.coordinates.items():
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

    def filter_by_timelines(
        self,
        *,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
    ) -> "MatchStamp":
        """Create filtered stamp with subset of timelines.

        Args:
            timeline_ids: Only include these timelines (None = all).
            id_pattern: Regex filter for timeline IDs.

        Returns:
            New MatchStamp with filtered timelines.
        """
        filt = ClaimFilter.from_kwargs(
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
        )
        filtered_coords = {}
        for tl_id, coord in self.coordinates.items():
            if not filt.matches_timeline(tl_id):
                continue
            filtered_coords[tl_id] = coord

        # Filter edges to only include those between remaining timelines
        remaining = set(filtered_coords.keys())

        filtered_anchor_edges = [
            (a, b) for a, b in self.anchor_edges if a in remaining and b in remaining
        ]
        filtered_inferred_edges = [
            (a, b) for a, b in self.inferred_edges if a in remaining and b in remaining
        ]

        return MatchStamp(
            coordinates=filtered_coords,
            source_id=(
                self.source_id
                if self.source_id in filtered_coords
                else next(iter(filtered_coords))
            ),
            anchor_edges=filtered_anchor_edges,
            inferred_edges=filtered_inferred_edges,
            source=self.source,
            is_interpolated=self.is_interpolated,
            conversion_maps=self.conversion_maps,
            interval_claims=(
                self.interval_claims if self.source_id in filtered_coords else []
            ),
        )

    def to_dict(
        self,
        format: Literal["flat", "prefix", "nested", "graph"] = "flat",
    ) -> dict[str, Any]:
        """Materialize the stamp in a flat, grouped, or graph representation.

        Args:
            format: Output representation. ``"graph"`` preserves the
                MatchGraph storage shape.

        Returns:
            The requested dictionary representation.

        Raises:
            ValueError: If a grouped format is requested without a source
                bundle, or if the format is unknown.
        """
        if format == "graph":
            return {
                "coordinates": {
                    timeline_id: coordinate_to_dict(coordinate)
                    for timeline_id, coordinate in self.coordinates.items()
                },
                "anchor_edges": list(self.anchor_edges),
                "inferred_edges": list(self.inferred_edges),
            }

        def _bundle_uid(timeline_id: str) -> str:
            if self.source is None:
                return timeline_id
            return self.source._timeline_id_to_uid.get(timeline_id, timeline_id)

        if format not in ("flat", "prefix", "nested"):
            raise ValueError(
                f"Unknown format: {format!r}. Use 'flat', 'prefix', "
                "'nested', or 'graph'"
            )
        if format in ("prefix", "nested") and self.source is None:
            raise ValueError(
                f"MatchStamp.to_dict(format={format!r}) requires a source bundle "
                "to resolve timeline groups"
            )

        grouped: dict[str, dict[str, dict[str, object]]] = {}
        for timeline_id, coordinate in self.coordinates.items():
            coordinate_data = coordinate_to_dict(coordinate)
            bundle_uid = _bundle_uid(timeline_id)
            if self.source is None:
                grouped.setdefault(bundle_uid, {})[timeline_id] = coordinate_data
                continue

            group_id = self.source.timeline_to_group.get(bundle_uid, bundle_uid)
            grouped.setdefault(group_id, {})[bundle_uid] = coordinate_data

        if format == "flat":
            return {
                f"{timeline_id} ({coordinate_data['unit']})": coordinate_data
                for timeline_coordinates in grouped.values()
                for timeline_id, coordinate_data in timeline_coordinates.items()
            }

        if format == "nested":
            return {
                group_id: {
                    f"{timeline_id} ({coordinate_data['unit']})": coordinate_data
                    for timeline_id, coordinate_data in timeline_coordinates.items()
                }
                for group_id, timeline_coordinates in grouped.items()
            }

        result: dict[str, dict[str, object]] = {}
        for group_id, timeline_coordinates in grouped.items():
            for timeline_id, coordinate_data in timeline_coordinates.items():
                result[f"{group_id}/{timeline_id} ({coordinate_data['unit']})"] = (
                    coordinate_data
                )
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchStamp":
        """Deserialize a graph-shaped typed coordinate dictionary.

        Args:
            data: Graph-shaped stamp payload.

        Returns:
            A canonical typed match stamp.
        """
        coordinates: dict[str, Coordinate] = {}
        for timeline_id, coordinate_data in data["coordinates"].items():
            if not isinstance(coordinate_data, dict):
                raise TypeError("MatchStamp coordinate leaves must be dictionaries")
            coordinates[timeline_id] = coordinate_from_dict(coordinate_data)
        if not coordinates:
            raise ValueError("MatchStamp payload must contain at least one coordinate")
        return cls(
            coordinates=coordinates,
            source_id=next(iter(coordinates)),
            anchor_edges=[tuple(e) for e in data.get("anchor_edges", [])],
            inferred_edges=[tuple(e) for e in data.get("inferred_edges", [])],
        )

    def __repr__(self) -> str:
        entries = [
            f"{timeline_id}="
            f"{_format_stamp_value(coordinate.value, coordinate.unit.value)}"
            for timeline_id, coordinate in self.coordinates.items()
        ]
        entries.extend(
            self._claim_candidate_text(index, claim, timeline_id, coordinate)
            for index, (claim, timeline_id, coordinate) in enumerate(
                self._interval_candidate_rows(), 1
            )
        )
        return f"MatchStamp({', '.join(entries)})"

    def __str__(self) -> str:
        """Readable cross-section showing all coordinates.

        Examples:
            >>> print(stamp)
            MatchStamp (3 timelines, 2 edges)
              ID              Coordinate   Type
              score:clt_1      0            anchor
              perf:dlt_1       0            anchor
              perf:dlt_2       128          inferred
        """
        n_edges = self.n_explicit_edges + self.n_inferred_edges
        lines: list[str] = [
            f"MatchStamp ({self.n_timelines} timelines, {n_edges} edges)"
        ]

        if not self.coordinates:
            return lines[0]

        # Classify each timeline by edge type
        anchor_tls = set()
        for a, b in self.anchor_edges:
            anchor_tls.add(a)
            anchor_tls.add(b)
        inferred_tls = set()
        for a, b in self.inferred_edges:
            inferred_tls.add(a)
            inferred_tls.add(b)

        entries: list[tuple[str, str, str]] = []
        for tl_id, coord in self.coordinates.items():
            if tl_id in anchor_tls:
                tag = "anchor"
            elif tl_id in inferred_tls:
                tag = "inferred"
            else:
                tag = ""
            entries.append(
                (
                    tl_id,
                    _format_stamp_value(coord.value, coord.unit.value),
                    tag,
                )
            )

        for label, value, suffix in self._conversion_rows():
            entries.append((label, _format_stamp_value(value, suffix), ""))

        for index, (claim, timeline_id, coordinate) in enumerate(
            self._interval_candidate_rows(), 1
        ):
            entries.append(
                (
                    f"{claim.timeline_a_id}<->{claim.timeline_b_id}",
                    (
                        "MISSING"
                        if coordinate is None
                        else _format_stamp_value(
                            coordinate.value, coordinate.unit.value
                        )
                    ),
                    f"interval claim {index} ({timeline_id})",
                )
            )

        if entries:
            max_id = max(len(e[0]) for e in entries)
            max_coord = max(len(e[1]) for e in entries)
            for tl_id, coord_str, tag in entries:
                line = f"  {tl_id:<{max_id}}  {coord_str:>{max_coord}}"
                if tag:
                    line += f"  {tag}"
                lines.append(line)

        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter notebooks.

        Displays the MatchStamp as an HTML table showing all coordinates
        with their edge types, mirroring the TimeStamp HTML display.
        """
        import html as html_mod

        from timetoalign.display.html import affordance_line

        n_edges = self.n_explicit_edges + self.n_inferred_edges

        # Classify timelines
        anchor_tls = set()
        for a, b in self.anchor_edges:
            anchor_tls.add(a)
            anchor_tls.add(b)
        inferred_tls = set()
        for a, b in self.inferred_edges:
            inferred_tls.add(a)
            inferred_tls.add(b)

        rows = []
        for tl_id, coord in self.coordinates.items():
            esc_id = html_mod.escape(tl_id)
            formatted = html_mod.escape(
                _format_stamp_value(coord.value, coord.unit.value)
            )
            if tl_id in anchor_tls:
                tag = "<em>anchor</em>"
                rows.append(
                    f"<tr><td><strong>{esc_id}</strong></td>"
                    f"<td style='text-align: right;'>{formatted}</td>"
                    f"<td>{tag}</td></tr>"
                )
            elif tl_id in inferred_tls:
                tag = "<em style='color: #666;'>inferred</em>"
                rows.append(
                    f"<tr><td style='color: #666;'>{esc_id}</td>"
                    f"<td style='text-align: right;'>{formatted}</td>"
                    f"<td>{tag}</td></tr>"
                )
            else:
                rows.append(
                    f"<tr><td>{esc_id}</td>"
                    f"<td style='text-align: right;'>{formatted}</td>"
                    f"<td></td></tr>"
                )

        for label, value, suffix in self._conversion_rows():
            rows.append(
                f"<tr><td style='color: #666;'>{html_mod.escape(label)}</td>"
                f"<td style='text-align: right;'>"
                f"{html_mod.escape(_format_stamp_value(value, suffix))}</td>"
                f"<td style='color: #666;'><em>cmap</em></td></tr>"
            )

        for index, (claim, timeline_id, coordinate) in enumerate(
            self._interval_candidate_rows(), 1
        ):
            label = html_mod.escape(f"{claim.timeline_a_id}<->{claim.timeline_b_id}")
            value = (
                "MISSING"
                if coordinate is None
                else _format_stamp_value(coordinate.value, coordinate.unit.value)
            )
            rows.append(
                f"<tr><td style='color: #666;'>{label}</td>"
                f"<td style='text-align: right;'>{html_mod.escape(value)}</td>"
                f"<td style='color: #666;'><em>interval claim {index} "
                f"({html_mod.escape(timeline_id)})</em></td></tr>"
            )

        badge = (
            f" <span style='background: #e3f2fd; padding: 0 4px; "
            f"border-radius: 3px; font-size: 0.8em;'>"
            f"{self.n_timelines} timelines, {n_edges} edges</span>"
        )

        affordances = [
            "stamp.get_coordinate_for(<tl_id>)",
            "stamp.get_coordinates_for(<tl_ids>)",
            "stamp.get_unit(<unit>)",
            "stamp.get_conversion_for(<key>)",
        ]
        return (
            f"<div style='font-family: monospace;'>"
            f"<strong>MatchStamp</strong>{badge}"
            f"<table style='border-collapse: collapse; margin-top: 4px;'>"
            f"<thead><tr style='border-bottom: 1px solid #ccc;'>"
            f"<th style='text-align: left; padding: 2px 8px;'>ID</th>"
            f"<th style='text-align: right; padding: 2px 8px;'>Coordinate</th>"
            f"<th style='text-align: left; padding: 2px 8px;'>Type</th>"
            f"</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            f"</table>"
            f"{affordance_line(affordances)}"
            f"</div>"
        )


# endregion


# region MatchIntervalStamp


@dataclass(frozen=True, slots=True)
class MatchIntervalStamp:
    """Two coordinate match stamps combined for an interval query."""

    source_id: str
    interval: Interval
    start_stamp: MatchStamp
    end_stamp: MatchStamp

    def __post_init__(self) -> None:
        """Validate the queried interval and endpoint resolutions."""
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError(
                "MatchIntervalStamp source_id must be a non-empty timeline ID"
            )
        if type(self.interval) is not Interval:
            raise TypeError("MatchIntervalStamp interval must be an Interval")
        if (
            type(self.start_stamp) is not MatchStamp
            or type(self.end_stamp) is not MatchStamp
        ):
            raise TypeError("MatchIntervalStamp endpoints must be MatchStamp values")
        if self.source_id not in self.start_stamp.coordinates:
            raise ValueError("MatchIntervalStamp start does not contain its source")
        if self.source_id not in self.end_stamp.coordinates:
            raise ValueError("MatchIntervalStamp end does not contain its source")
        canonical = Interval(
            self.interval.start,
            self.interval.end,
        )
        object.__setattr__(self, "interval", canonical)

    def __repr__(self) -> str:
        """Render one resolved endpoint pair per present timeline."""
        entries = "; ".join(
            f"{timeline_id}={self._pair_text(timeline_id)}"
            for timeline_id in self.present_timelines
        )
        return f"MatchIntervalStamp({entries})"

    def __str__(self) -> str:
        """Render a readable table of resolved endpoint pairs."""
        lines = [f"MatchIntervalStamp ({len(self.present_timelines)} timelines)"]
        entries = [
            (timeline_id, self._pair_text(timeline_id))
            for timeline_id in self.present_timelines
        ]
        max_id = max(len(timeline_id) for timeline_id, _ in entries)
        for timeline_id, pair in entries:
            lines.append(f"  {timeline_id:<{max_id}}  {pair}")
        return "\n".join(lines)

    @property
    def axis(self) -> Interval:
        """Return the canonical queried interval."""
        return self.interval

    @property
    def start(self) -> MatchStamp:
        """Return the full start-endpoint resolution."""
        return self.start_stamp

    @property
    def end(self) -> MatchStamp:
        """Return the full end-endpoint resolution."""
        return self.end_stamp

    @property
    def present_timelines(self) -> list[str]:
        """Return source-first endpoint timeline order."""
        ordered = [self.source_id]
        ordered.extend(
            timeline_id
            for timeline_id in self.start_stamp.present_timelines
            if timeline_id not in ordered
        )
        ordered.extend(
            timeline_id
            for timeline_id in self.end_stamp.present_timelines
            if timeline_id not in ordered
        )
        return ordered

    def _pair_text(self, timeline_id: str) -> str:
        """Format one timeline's two resolved sides."""
        start = self.start_stamp.coordinates.get(timeline_id)
        end = self.end_stamp.coordinates.get(timeline_id)
        start_text = (
            "MISSING"
            if start is None
            else _format_stamp_value(start.value, start.unit.value)
        )
        end_text = (
            "MISSING" if end is None else _format_stamp_value(end.value, end.unit.value)
        )
        return f"[{start_text}, {end_text}]"

    def get_interval_for(self, timeline_id: str) -> Interval:
        """Return one complete, ordered resolved interval."""
        if timeline_id not in self.present_timelines:
            raise KeyError(
                f"Unknown timeline ID {timeline_id!r} on MatchIntervalStamp. "
                f"Available timelines: {self.present_timelines}"
            )
        start = self.start_stamp.coordinates.get(timeline_id)
        end = self.end_stamp.coordinates.get(timeline_id)
        if start is None:
            raise ValueError(f"Timeline {timeline_id!r} has a missing start endpoint")
        if end is None:
            raise ValueError(f"Timeline {timeline_id!r} has a missing end endpoint")
        if start.value > end.value:
            raise ValueError(
                f"Timeline {timeline_id!r} has reversed endpoints: "
                f"start {start.value!r} exceeds end {end.value!r}"
            )
        return Interval(start, end)

    def to_dict(self) -> dict[str, dict[str, Any | None]]:
        """Serialize endpoint coverage with typed leaves and null missing sides."""
        result: dict[str, dict[str, Any | None]] = {}
        for timeline_id in self.present_timelines:
            start = self.start_stamp.coordinates.get(timeline_id)
            end = self.end_stamp.coordinates.get(timeline_id)
            result[timeline_id] = {
                "start": None if start is None else coordinate_to_dict(start),
                "end": None if end is None else coordinate_to_dict(end),
            }
        return result

    def _repr_html_(self) -> str:
        """Return an HTML table of resolved endpoint pairs."""
        import html as html_mod

        rows = []
        for timeline_id in self.present_timelines:
            start = self.start_stamp.coordinates.get(timeline_id)
            end = self.end_stamp.coordinates.get(timeline_id)
            start_text = (
                "MISSING"
                if start is None
                else _format_stamp_value(start.value, start.unit.value)
            )
            end_text = (
                "MISSING"
                if end is None
                else _format_stamp_value(end.value, end.unit.value)
            )
            rows.append(
                f"<tr><td>{html_mod.escape(timeline_id)}</td>"
                f"<td style='text-align: right;'>{html_mod.escape(start_text)}</td>"
                f"<td style='text-align: right;'>{html_mod.escape(end_text)}</td>"
                f"</tr>"
            )
        badge = (
            f" <span style='background: #e3f2fd; padding: 0 4px; "
            f"border-radius: 3px; font-size: 0.8em;'>"
            f"{len(self.present_timelines)} timelines</span>"
        )
        return (
            f"<div style='font-family: monospace;'>"
            f"<strong>MatchIntervalStamp</strong>{badge}"
            f"<table style='border-collapse: collapse; margin-top: 4px;'>"
            f"<thead><tr style='border-bottom: 1px solid #ccc;'>"
            f"<th style='text-align: left; padding: 2px 8px;'>ID</th>"
            f"<th style='text-align: right; padding: 2px 8px;'>Start</th>"
            f"<th style='text-align: right; padding: 2px 8px;'>End</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )


# endregion


# region MatchGraph


class MatchGraph:
    """A graph of MatchClaims connecting events across timelines/groups.

    The MatchGraph builds a networkx graph where:
    - Nodes: (timeline_id, coordinate) tuples
    - Edges: synchronous AlignmentAnchors (explicit or implicit)

    Only synchronous claims produce graph edges. Non-synchronous claims
    (conceptual matches, NOMATCH) are stored in ``_claims`` but do not
    create nodes or edges.

    Each Hendrix M-box (M1–M15) is a separate MatchGraph. The system
    is NOT a global graph; MatchGraphs are created on demand.

    Attributes:
        claims: List of all MatchClaims in this graph (synchronous and non-synchronous).

    Examples:
        >>> # Build graph from claims
        >>> graph = MatchGraph(claims=[claim1, claim2])

        >>> # Extend via group membership
        >>> extended = graph.extend_to_groups(groups, timeline_to_group)

        >>> # Get synchronized timestamps
        >>> stamps = graph.get_stamps()
    """

    def __init__(
        self,
        claims: list[MatchClaim] | None = None,
        *,
        units: dict[str, str] | None = None,
        axis: int | float | Fraction | None = None,
        source: "AlignmentBundle | None" = None,
        source_id: str | None = None,
    ):
        """Initialize MatchGraph from MatchClaims.

        Args:
            claims: List of MatchClaims to build graph from.
        """
        self._claims: list[MatchClaim] = claims or []
        self._graph: nx.Graph = self._build_graph()
        self._logger = module_logger.getChild("MatchGraph")
        self._units = dict(units or {})
        self._timeline_units = self._collect_timeline_units()
        self._timeline_number_types = self._collect_timeline_number_types()
        self._axis = axis
        self._source = source
        self._source_id = source_id

    @property
    def claims(self) -> list[MatchClaim]:
        """List of all MatchClaims in this graph (synchronous and non-synchronous)."""
        return self._claims

    @property
    def synchronous_claims(self) -> list[MatchClaim]:
        """List of only synchronous MatchClaims (those with anchors/edges)."""
        return [c for c in self._claims if c.is_synchronous]

    @property
    def non_synchronous_claims(self) -> list[MatchClaim]:
        """List of non-synchronous MatchClaims (NOMATCH, conceptual)."""
        return [c for c in self._claims if not c.is_synchronous]

    @property
    def n_claims(self) -> int:
        """Number of claims in this graph."""
        return len(self._claims)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._graph.number_of_nodes()

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return self._graph.number_of_edges()

    timeline_ids = property(
        lambda self: {node[0] for node in self._graph.nodes()},
        doc="Set of all timeline IDs in the graph.",
    )

    def _build_graph(self) -> nx.Graph:
        """Build the anchor graph from claims.

        Only synchronous claims (those with anchors) produce graph edges.
        Non-synchronous claims are stored in ``_claims`` but do not create
        nodes or edges.

        Nodes: (timeline_id, coordinate) tuples
        Edges: Anchors with attributes (explicit, synchronous, claim_id)

        Returns:
            networkx Graph representing the anchor structure.
        """
        G = nx.Graph()

        for claim in self._claims:
            if not claim.is_synchronous or claim.start_anchor is None:
                continue

            # Add start anchor edge
            self._add_anchor_edge(G, claim.start_anchor, claim)

            # Add end anchor edge if interval
            if claim.end_anchor:
                self._add_anchor_edge(G, claim.end_anchor, claim)

        return G

    def _collect_timeline_units(self) -> dict[str, TimeUnit]:
        """Collect the coordinate units recorded by this graph's claims."""
        timeline_units: dict[str, TimeUnit] = {}
        for claim in self._claims:
            for anchor in claim.anchors:
                timeline_units.setdefault(
                    anchor.timeline_a_id, anchor.coordinate_a.unit
                )
                timeline_units.setdefault(
                    anchor.timeline_b_id, anchor.coordinate_b.unit
                )
        return timeline_units

    def _collect_timeline_number_types(self) -> dict[str, NumberType]:
        """Collect declared coordinate representations from claim anchors."""
        result: dict[str, NumberType] = {}
        for claim in self._claims:
            for anchor in claim.anchors:
                result.setdefault(anchor.timeline_a_id, anchor.coordinate_a.number_type)
                result.setdefault(anchor.timeline_b_id, anchor.coordinate_b.number_type)
        return result

    def _add_anchor_edge(
        self,
        G: nx.Graph,
        anchor: AlignmentAnchor,
        claim: MatchClaim,
    ) -> None:
        """Add an anchor as an edge to the graph.

        Args:
            G: The networkx graph to add to.
            anchor: The AlignmentAnchor to add.
            claim: The parent MatchClaim for metadata.
        """
        node_a: _GraphNode = (anchor.timeline_a_id, anchor.coordinate_a.value)
        node_b: _GraphNode = (anchor.timeline_b_id, anchor.coordinate_b.value)

        G.add_edge(
            node_a,
            node_b,
            explicit=claim.is_explicit,
            synchronous=claim.is_synchronous,
            claim_id=claim.id,
        )

    def _get_node_keys_for_timeline(self, timeline_id: str) -> list[_GraphNode]:
        """Return private graph keys for one timeline."""
        return [node for node in self._graph.nodes() if node[0] == timeline_id]

    def get_nodes_for_timeline(self, timeline_id: str) -> list[IdCoordinate]:
        """Get all public graph nodes for a specific timeline.

        Args:
            timeline_id: The timeline to get nodes for.

        Returns:
            Canonical identified coordinates in graph insertion order.

        Raises:
            TypeError: If ``timeline_id`` is not a string.
            KeyError: If the timeline is unknown.
        """
        if not isinstance(timeline_id, str):
            raise TypeError("get_nodes_for_timeline requires a timeline-ID string")
        nodes = self._get_node_keys_for_timeline(timeline_id)
        if not nodes:
            raise KeyError(f"Unknown timeline ID {timeline_id!r} in MatchGraph")
        return [self._identified_coordinate(node[0], node[1]) for node in nodes]

    def get_coordinates_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return the sorted graph coordinate column for one timeline.

        Args:
            timeline_id: The timeline to get coordinates for.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            Typed coordinate projections in sorted order.
        """
        if not isinstance(timeline_id, str):
            raise TypeError("get_coordinates_for requires one timeline-ID string")
        coords = [node[1] for node in self._graph.nodes() if node[0] == timeline_id]
        if not coords:
            raise KeyError(f"Unknown timeline ID {timeline_id!r} in MatchGraph")
        identified = [
            self._identified_coordinate(timeline_id, value) for value in sorted(coords)
        ]
        return format_coordinates(
            identified,
            format=format,
            rounding=rounding,
            scalar=False,
            series_name=timeline_id,
        )

    def _identified_coordinate(
        self, timeline_id: str, value: CoordinateValue
    ) -> IdCoordinate:
        """Build one canonical ID coordinate from a graph node."""
        unit = self._timeline_units.get(timeline_id)
        if unit is None:
            unit_name = self._units.get(timeline_id)
            if unit_name is None:
                raise KeyError(
                    f"Graph has no declared unit for timeline {timeline_id!r}"
                )
            unit = TimeUnit(unit_name)
        number_type = self._timeline_number_types.get(
            timeline_id, NumberType.from_number(value)
        )
        coordinate = Coordinate(value, unit, number_type=number_type)
        return IdCoordinate.from_coordinate(coordinate, timeline_id)

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Resolve one exact graph node to a target component coordinate.

        Args:
            at: Exact graph-node position.
            timeline_id: Requested result timeline, or the unique other node.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            One exact connected-coordinate projection or a length-one Series.
        """
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        if isinstance(at, IdCoordinate):
            source_id = at.timeline_id
            source_value = at.value
            native = self._identified_coordinate(source_id, source_value)
            if at.unit != native.unit:
                raise ValueError(
                    f"Coordinate unit {at.unit} is not native to graph axis {source_id!r}"
                )
        else:
            if timeline_id is None:
                raise ValueError(
                    "timeline_id is required for raw or plain MatchGraph queries"
                )
            source_id = timeline_id
            source_value = at.value if isinstance(at, Coordinate) else at
            native = self._identified_coordinate(source_id, source_value)
            if isinstance(at, Coordinate) and at.unit != native.unit:
                raise ValueError(
                    f"Coordinate unit {at.unit} is not native to graph axis {source_id!r}"
                )
        source_node: _GraphNode = (source_id, native.value)
        if source_node not in self._graph:
            raise KeyError(
                f"No exact graph anchor at {native.value!r} on {source_id!r}"
            )
        component = nx.node_connected_component(self._graph, source_node)
        if timeline_id is None:
            candidates = [node for node in component if node[0] != source_id]
            if not candidates:
                raise KeyError(
                    "Connected component has no coordinate on another timeline"
                )
            if len(candidates) != 1:
                raise ValueError(
                    f"Connected component has competing nodes {candidates}"
                )
        else:
            candidates = [node for node in component if node[0] == timeline_id]
            if not candidates:
                raise KeyError(
                    f"Connected component has no node on timeline {timeline_id!r}"
                )
            if len(candidates) != 1:
                raise ValueError(
                    f"Connected component has competing {timeline_id!r} nodes {candidates}"
                )
        result = self._identified_coordinate(candidates[0][0], candidates[0][1])
        return format_coordinates(
            [result],
            format=format,
            rounding=rounding,
            scalar=True,
            series_name=result.timeline_id,
        )

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve exact graph nodes for a coordinate collection.

        Args:
            at: Exact graph-node positions to resolve atomically.
            timeline_id: Requested result timeline.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        if not values and timeline_id is None:
            raise ValueError("timeline_id is required for an empty MatchGraph query")
        if timeline_id is not None and timeline_id not in self.timeline_ids:
            raise KeyError(f"Unknown timeline ID {timeline_id!r} in MatchGraph")
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
            series_name=timeline_id
            or (
                results[0].timeline_id
                if results
                and all(
                    result.timeline_id == results[0].timeline_id for result in results
                )
                else "coordinate"
            ),
            empty_number_type=(
                self._timeline_number_types.get(timeline_id)
                if timeline_id is not None
                else None
            ),
        )

    def get_coordinate(
        self,
        at: CoordinateInput | CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | list[CoordinateResult] | pd.Series:
        """Dispatch a scalar or plural exact graph-position query.

        Args:
            at: One graph position or a coordinate collection.
            timeline_id: Requested result timeline when required.
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
            positions_only=(
                "MatchGraph.get_coordinate accepts coordinate inputs; use "
                "get_coordinates_for for a timeline column"
            ),
            format=format,
            rounding=rounding,
        )

    def get_connected_nodes(self, node: IdCoordinate) -> list[IdCoordinate]:
        """Get all public nodes connected to a given node.

        Args:
            node: Canonical identified graph coordinate.

        Returns:
            Canonical identified neighboring coordinates.

        Raises:
            TypeError: If ``node`` is not an ``IdCoordinate``.
            ValueError: If its unit is not native to its timeline.
        """
        if not isinstance(node, IdCoordinate):
            raise TypeError("get_connected_nodes requires an IdCoordinate")
        native = self._identified_coordinate(node.timeline_id, node.value)
        if node.unit != native.unit:
            raise ValueError(
                f"Coordinate unit {node.unit} is not native to graph axis "
                f"{node.timeline_id!r}"
            )
        node_key: _GraphNode = (native.timeline_id, native.value)
        if node_key not in self._graph:
            return []
        return [
            self._identified_coordinate(neighbor[0], neighbor[1])
            for neighbor in self._graph.neighbors(node_key)
        ]

    def get_connected_timelines(self, timeline_id: str) -> set[str]:
        """Get all timelines connected to a given timeline.

        Args:
            timeline_id: The timeline to check.

        Returns:
            Set of connected timeline IDs.
        """
        connected = set()
        for node in self._get_node_keys_for_timeline(timeline_id):
            for neighbor in self._graph.neighbors(node):
                connected.add(neighbor[0])
        return connected - {timeline_id}

    def extend_to_groups(
        self,
        groups: dict[str, "TimelineGroup"],
        timeline_to_group: dict[str, str],
        include_inferred: bool = True,
        *,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set[Domain] | None = None,
        include_units: set[TimeUnit] | None = None,
    ) -> "MatchGraph":
        """Extend anchors to full group timestamps via implicit claims.

        For each coordinate in the graph, if it belongs to a Group,
        computes the equivalent coordinate for every other member of that
        Group and adds an implicit ``MatchClaim`` (case d) plus the
        corresponding edge. Filters control which timelines receive
        implicit claims.

        Args:
            groups: Dict of group_id -> TimelineGroup.
            timeline_to_group: Dict of timeline_id -> group_id.
            include_inferred: Whether to add inferred edges.
            timeline_ids: Only extend to these timeline IDs.
            id_pattern: Regex filter for timeline IDs.
            include_domains: Only extend to timelines in these domains.
            include_units: Only extend to timelines with these units.

        Returns:
            New MatchGraph with extended edges (or self if not extending).
        """
        if not include_inferred:
            return self

        # Create a copy of the graph
        extended = nx.Graph(self._graph)
        implicit_claims: list[MatchClaim] = []

        for node in list(self._graph.nodes()):
            timeline_id, coord = node

            # Check if timeline belongs to a group
            group_id = timeline_to_group.get(timeline_id)
            if not group_id:
                continue

            group = groups.get(group_id)
            if not group:
                continue

            # Find the source claim for traceability
            source_claim = self._find_source_claim_for_node(node)

            # Add inferred edges to all other timelines in group
            for other_tl_id in group.timeline_ids:
                if other_tl_id == timeline_id:
                    continue

                # Apply filters
                if not self._passes_filters(
                    other_tl_id,
                    timelines={
                        member_id: group.get_timeline(member_id)
                        for member_id in group.timeline_ids
                    },
                    timeline_ids=timeline_ids,
                    id_pattern=id_pattern,
                    include_domains=include_domains,
                    include_units=include_units,
                ):
                    continue

                # Convert coordinate to other timeline
                try:
                    source_timeline = group.get_timeline(timeline_id)
                    other_coord = group.get_coordinate_at(
                        IdCoordinate(
                            coord,
                            source_timeline.unit,
                            timeline_id,
                            number_type=source_timeline.number_type,
                        ),
                        timeline_id=other_tl_id,
                        format="coordinate",
                    )
                except (KeyError, ValueError):
                    continue
                assert isinstance(other_coord, Coordinate)

                other_value = float(other_coord.value)
                other_node: _GraphNode = (other_tl_id, other_value)

                # Add implicit edge if not already connected
                if not extended.has_edge(node, other_node):
                    # Create an implicit MatchClaim (case d)
                    implicit_claim = MatchClaim.implicit(
                        tl_a_id=timeline_id,
                        coord_a=coord,
                        tl_b_id=other_tl_id,
                        coord_b=other_value,
                        unit_a=group.get_timeline(timeline_id).unit,
                        unit_b=group.get_timeline(other_tl_id).unit,
                        source_claim=source_claim,
                    )
                    implicit_claims.append(implicit_claim)

                    extended.add_edge(
                        node,
                        other_node,
                        explicit=False,
                        synchronous=True,
                        inferred_via="group",
                        group_id=group_id,
                        claim_id=implicit_claim.id,
                    )

        all_claims = list(self._claims) + implicit_claims
        return MatchGraph._from_graph(
            extended,
            all_claims,
            units=self._units,
            axis=self._axis,
            source=self._source,
            source_id=self._source_id,
        )

    def _find_source_claim_for_node(self, node: _GraphNode) -> MatchClaim | None:
        """Find the first explicit synchronous claim that contains this node.

        Args:
            node: The (timeline_id, coordinate) node.

        Returns:
            The source MatchClaim, or None if not found.
        """
        timeline_id, coord = node
        for claim in self._claims:
            if not claim.is_synchronous or not claim.is_explicit:
                continue
            if claim.start_anchor is None:
                continue
            # Check if this claim's anchors touch this node
            for anchor in claim.anchors:
                if (
                    anchor.timeline_a_id == timeline_id
                    and anchor.coordinate_a.value == coord
                ):
                    return claim
                if (
                    anchor.timeline_b_id == timeline_id
                    and anchor.coordinate_b.value == coord
                ):
                    return claim
        return None

    def _passes_filters(
        self,
        timeline_id: str,
        *,
        timelines: dict[str, "Timeline"] | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set[Domain] | None = None,
        include_units: set[TimeUnit] | None = None,
    ) -> bool:
        """Check whether a timeline passes the given filters.

        Args:
            timeline_id: The timeline ID to check.
            timelines: Dict of timeline_id -> Timeline for metadata.
            timeline_ids: Only these timeline IDs pass.
            id_pattern: Regex filter for timeline IDs.
            include_domains: Only timelines in these domains pass.
            include_units: Only timelines with these units pass.

        Returns:
            True if the timeline passes all filters.
        """
        if timelines is None:
            timelines = {
                tl_id: SimpleNamespace(unit=unit)
                for tl_id, unit in self._timeline_units.items()
            }
        return ClaimFilter.from_kwargs(
            timeline_ids=timeline_ids,
            id_pattern=id_pattern,
            include_domains=include_domains,
            include_units=include_units,
        ).matches_timeline(timeline_id, timelines=timelines)

    @classmethod
    def _from_graph(
        cls,
        graph: nx.Graph,
        claims: list[MatchClaim],
        *,
        units: dict[str, str] | None = None,
        axis: int | float | Fraction | None = None,
        source: "AlignmentBundle | None" = None,
        source_id: str | None = None,
    ) -> "MatchGraph":
        """Create MatchGraph from existing networkx graph.

        Internal constructor for creating extended graphs.

        Args:
            graph: The networkx graph.
            claims: All claims (original + implicit).

        Returns:
            New MatchGraph wrapping the graph.
        """
        instance = cls.__new__(cls)
        instance._claims = claims
        instance._graph = graph
        instance._logger = module_logger.getChild("MatchGraph")
        instance._units = dict(units or {})
        instance._timeline_units = instance._collect_timeline_units()
        instance._timeline_number_types = instance._collect_timeline_number_types()
        instance._axis = axis
        instance._source = source
        instance._source_id = source_id
        return instance

    def get_stamps(
        self,
    ) -> list["MatchStamp"]:
        """Get all MatchStamps from the graph.

        Returns one MatchStamp per connected component, each containing
        all coordinates reachable from that component.

        Args:
        Returns:
            List of MatchStamps, one per connected component.
        """
        stamps = []
        for component in nx.connected_components(self._graph):
            stamp = self._build_stamp_from_node(self._component_start_node(component))
            stamps.append(stamp)
        return stamps

    def _component_start_node(self, component: "set[_GraphNode]") -> _GraphNode:
        """Choose a component's starting node so the stamp is the same every run.

        ``connected_components`` hands back sets, and taking whichever node
        came out of one first made the stamp's ``source_id`` -- and therefore
        the position of every entry ordered relative to it -- depend on the
        process's hash seed. Two runs of the same notebook rendered the same
        cross-section with its columns swapped.

        The choice is the one the retrieval order already names: the graph's
        own source timeline where the component contains it, else the
        lexically first node. Everything after the source was already sorted,
        which is why only the first entry moved and the tail looked stable.
        """
        return min(
            component,
            key=lambda node: (node[0] != self._source_id, node[0], node[1]),
        )

    def get_matchstamp(self) -> "MatchStamp":
        """Get the single MatchStamp for this graph.

        One MatchGraph = one MatchStamp. The MatchStamp is the union of
        all coordinates reachable through the graph's edges.

        If the graph contains multiple disconnected components, this
        method raises ``ValueError`` -- each component should be its own
        MatchGraph. Use ``split_components()`` to separate them first, or
        use the multi-component ``get_stamps()`` method.

        Returns:
            Single MatchStamp spanning all timelines in the graph.

        Raises:
            ValueError: If the graph has multiple disconnected components.
            ValueError: If the graph has no synchronous claims (no nodes).

        See Also:
            `split_components`: Split a multi-component graph into separate
                MatchGraph objects.
            `get_stamps`: Multi-component method returning one stamp per component.
        """
        components = list(nx.connected_components(self._graph))
        if not components:
            raise ValueError(
                "MatchGraph has no synchronous claims and therefore no nodes. "
                "Cannot produce a MatchStamp from an empty graph."
            )
        if len(components) > 1:
            raise ValueError(
                f"MatchGraph has {len(components)} disconnected components. "
                f"One graph = one MatchStamp. Use split_components() first, "
                f"or use get_stamps() for the multi-component API."
            )
        return self._build_stamp_from_node(self._component_start_node(components[0]))

    def split_components(self) -> list["MatchGraph"]:
        """Split this graph into one MatchGraph per connected component.

        Each returned MatchGraph represents a single connected component
        and can be queried with ``get_matchstamp()``.

        Returns:
            List of MatchGraph objects, one per connected component.
            Empty list if the graph has no synchronous claims.
        """
        components = list(nx.connected_components(self._graph))
        if not components:
            return []

        result = []
        for component in components:
            subgraph = self._graph.subgraph(component).copy()

            # Find claims whose anchors are entirely within this component
            component_claims = []
            component_tl_ids = {node[0] for node in component}
            for c in self._claims:
                if c.is_synchronous:
                    if (
                        c.timeline_a_id in component_tl_ids
                        and c.timeline_b_id in component_tl_ids
                    ):
                        component_claims.append(c)
                else:
                    # Non-synchronous claims: include if both timelines
                    # are in this component
                    if (
                        c.timeline_a_id in component_tl_ids
                        and c.timeline_b_id in component_tl_ids
                    ):
                        component_claims.append(c)

            result.append(
                MatchGraph._from_graph(
                    subgraph,
                    component_claims,
                    units=self._units,
                    axis=self._axis,
                    source=self._source,
                    source_id=self._source_id,
                )
            )
        return result

    @property
    def n_components(self) -> int:
        """Number of connected components in the graph."""
        return nx.number_connected_components(self._graph)

    def _build_stamp_from_node(self, start_node: _GraphNode) -> "MatchStamp":
        """Build a MatchStamp from a starting node.

        Uses BFS to find all connected nodes and categorize edges.

        Args:
            start_node: The (timeline_id, coordinate) to start from.

        Returns:
            MatchStamp containing all connected coordinates.
        """
        if start_node not in self._graph:
            identified = self._identified_coordinate(start_node[0], start_node[1])
            return MatchStamp(
                coordinates={start_node[0]: identified.to_coordinate()},
                source_id=start_node[0],
                anchor_edges=[],
                inferred_edges=[],
                source=self._source,
            )

        # Find all nodes in the connected component
        component = nx.node_connected_component(self._graph, start_node)

        source_id = (
            self._source_id
            if any(node[0] == self._source_id for node in component)
            else start_node[0]
        )
        ordered_nodes = sorted(
            component,
            key=lambda node: (
                node[0] != source_id,
                "" if node[0] == source_id else node[0],
                node[1],
            ),
        )
        coordinates: dict[str, Coordinate] = {}
        for node in ordered_nodes:
            timeline_id, coord = node
            # If timeline already exists, keep the first coordinate
            # (could be multiple nodes for same timeline in complex graphs)
            if timeline_id not in coordinates:
                coordinates[timeline_id] = self._identified_coordinate(
                    timeline_id, coord
                ).to_coordinate()

        # Categorize edges
        anchor_edges: list[tuple[str, str]] = []
        inferred_edges: list[tuple[str, str]] = []

        # Get subgraph for this component
        subgraph = self._graph.subgraph(component)
        for u, v, data in sorted(
            subgraph.edges(data=True),
            key=lambda edge: (edge[0][0], edge[0][1], edge[1][0], edge[1][1]),
        ):
            edge_pair = (u[0], v[0])
            # Avoid duplicates (edges are undirected)
            reverse_pair = (v[0], u[0])

            if data.get("explicit", True):
                if edge_pair not in anchor_edges and reverse_pair not in anchor_edges:
                    anchor_edges.append(edge_pair)
            else:
                if (
                    edge_pair not in inferred_edges
                    and reverse_pair not in inferred_edges
                ):
                    inferred_edges.append(edge_pair)

        return MatchStamp(
            coordinates=coordinates,
            source_id=source_id,
            anchor_edges=anchor_edges,
            inferred_edges=inferred_edges,
            source=self._source,
        )

    def filter(
        self,
        synchronous_only: bool = False,
        explicit_only: bool = False,
        *,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set[Domain] | None = None,
        include_units: set[TimeUnit] | None = None,
    ) -> "MatchGraph":
        """Create filtered view of the graph.

        Args:
            synchronous_only: Include only synchronous edges.
            explicit_only: Include only explicit edges (no inferred).
            timeline_ids: Only include these timeline IDs.
            id_pattern: Regex filter for timeline IDs.
            include_domains: Only include timelines in these domains.
            include_units: Only include timelines with these units.

        Returns:
            New MatchGraph with filtered edges/nodes.
        """
        # Start with a copy
        filtered = nx.Graph(self._graph)

        # Filter by timeline IDs and domain/unit
        nodes_to_remove = []
        for node in filtered.nodes():
            timeline_id = node[0]
            if not self._passes_filters(
                timeline_id,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            ):
                nodes_to_remove.append(node)
        filtered.remove_nodes_from(nodes_to_remove)

        # Filter edges by attributes
        if synchronous_only or explicit_only:
            edges_to_remove = []
            for u, v, data in filtered.edges(data=True):
                if synchronous_only and not data.get("synchronous", True):
                    edges_to_remove.append((u, v))
                elif explicit_only and not data.get("explicit", True):
                    edges_to_remove.append((u, v))
            filtered.remove_edges_from(edges_to_remove)

        # Remove isolated nodes (nodes with no edges after filtering)
        isolated = list(nx.isolates(filtered))
        filtered.remove_nodes_from(isolated)

        # Filter claims to match remaining edges
        remaining_claim_ids = set()
        for _, _, data in filtered.edges(data=True):
            if "claim_id" in data:
                remaining_claim_ids.add(data["claim_id"])

        # Keep non-synchronous claims that connect remaining timelines
        remaining_tl_ids = {node[0] for node in filtered.nodes()}
        filtered_claims = []
        for c in self._claims:
            if c.id in remaining_claim_ids:
                filtered_claims.append(c)
            elif (
                not c.is_synchronous
                and c.timeline_a_id in remaining_tl_ids
                and c.timeline_b_id in remaining_tl_ids
            ):
                filtered_claims.append(c)

        return MatchGraph._from_graph(
            filtered,
            filtered_claims,
            units=self._units,
            axis=self._axis,
            source=self._source,
            source_id=self._source_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Note: This serializes the claims, not the full graph.
        The graph can be rebuilt from claims.
        """
        return {
            "claims": [c.to_dict() for c in self._claims],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MatchGraph":
        """Deserialize from dictionary."""
        claims = [MatchClaim.from_dict(c) for c in data["claims"]]
        return cls(claims)

    def __repr__(self) -> str:
        return (
            f"MatchGraph(claims={len(self._claims)}, "
            f"nodes={self.n_nodes}, edges={self.n_edges})"
        )


# endregion
