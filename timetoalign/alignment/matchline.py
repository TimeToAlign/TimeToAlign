"""MatchLine: ordered sequence of MatchStamps for WarpMap generation.

This module implements the MatchLine class, which provides the bridge
between MatchGraph and WarpMap.

A MatchLine is an ordered sequence of MatchStamps for a given source
timeline, sorted by coordinate on that timeline. It is the input for
WarpMap generation.

The hierarchy is:
    AlignmentAnchor -> MatchClaim -> MatchGraph -> MatchStamp -> MatchLine

Design:
    MatchLine collects MatchStamps from one or more MatchGraphs, orders
    them by coordinate on a designated source timeline, and provides
    ``get_alignment_anchors()`` for typed public anchor access. The Hendrix
    pattern (M6-M9) is supported via ``from_graphs()``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from timetoalign.alignment.claims import AlignmentAnchor
from timetoalign.alignment.graph import MatchGraph, MatchStamp
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

if TYPE_CHECKING:
    from timetoalign.alignment.claims import MatchClaim
    from timetoalign.alignment.match_format import MatchFileContext
    from timetoalign.core.enums import Domain, TimeUnit
    from timetoalign.timelines.groups import TimelineGroup

module_logger = logging.getLogger(__name__)


class _RawStampExportView:
    """Present raw stamp coordinates to storage writers."""

    def __init__(self, stamp: MatchStamp) -> None:
        self._stamp = stamp

    def _get_float_coordinate_for(self, timeline_id: str) -> float:
        """Return the raw coordinate expected by storage writers."""
        return self._stamp.get_coordinate_for(timeline_id, format="float")


class _RawMatchLineExportView:
    """Present a MatchLine with raw-coordinate stamp views."""

    def __init__(self, match_line: MatchLine) -> None:
        self.source_timeline_id = match_line.source_timeline_id
        self.stamps = [_RawStampExportView(stamp) for stamp in match_line.stamps]
        self._target_timeline_ids = match_line.target_timeline_ids()

    def target_timeline_ids(self) -> set[str]:
        """Return target timeline IDs from the source MatchLine."""
        return self._target_timeline_ids


@dataclass
class MatchLine:
    """Ordered sequence of MatchStamps for a source timeline.

    A MatchLine collects all synchronised timestamps that mention a
    given *source* timeline, orders them by coordinate on that timeline,
    and exposes ``get_alignment_anchors()`` for typed source-to-target pairs.

    Attributes:
        source_timeline_id: The timeline whose coordinates define the
            ordering of the stamps.
        stamps: MatchStamps sorted by coordinate on ``source_timeline_id``.

    Examples:
        >>> line = MatchLine.from_claims(
        ...     claims=claims,
        ...     source_timeline_id="score",
        ... )
        >>> anchors = line.get_alignment_anchors("audio")

    See Also:
        `timetoalign.MatchGraph`
        `timetoalign.MatchStamp`
    """

    source_timeline_id: str
    stamps: list[MatchStamp] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and sort stamps by source coordinate."""
        # Filter out stamps that don't include the source timeline
        valid_stamps = [
            s for s in self.stamps if s.has_timeline(self.source_timeline_id)
        ]
        if len(valid_stamps) != len(self.stamps):
            n_dropped = len(self.stamps) - len(valid_stamps)
            # Routine per-source filtering: a MatchLine is built from all
            # cross-group claims, so components describing OTHER sources are
            # expected here (a hub-and-spoke bridge hits this on every query).
            # Not an error condition, hence debug rather than warning.
            module_logger.debug(
                "MatchLine: dropped %d stamp(s) that do not contain "
                "source timeline '%s'",
                n_dropped,
                self.source_timeline_id,
            )
        # Sort by coordinate on the source timeline
        object.__setattr__(
            self,
            "stamps",
            sorted(
                valid_stamps,
                key=lambda s: s.coordinates[self.source_timeline_id].value,
            ),
        )

    @property
    def n_stamps(self) -> int:
        """Number of stamps in this MatchLine."""
        return len(self.stamps)

    @property
    def source_coordinates(self) -> list[IdCoordinate]:
        """Return sorted typed coordinates on the source timeline."""
        return [
            IdCoordinate.from_coordinate(
                stamp.coordinates[self.source_timeline_id], self.source_timeline_id
            )
            for stamp in self.stamps
        ]

    def target_timeline_ids(self) -> set[str]:
        """All target timelines appearing in at least 2 stamps.

        A target timeline must appear in at least two stamps for
        interpolation (i.e., WarpMap construction) to be meaningful.

        Returns:
            Set of timeline IDs (excluding the source) that appear
            in >= 2 stamps.
        """
        counts: dict[str, int] = {}
        for stamp in self.stamps:
            for tl_id in stamp.present_timelines:
                if tl_id == self.source_timeline_id:
                    continue
                counts[tl_id] = counts.get(tl_id, 0) + 1
        return {tl_id for tl_id, count in counts.items() if count >= 2}

    def get_alignment_anchors(self, target_timeline_id: str) -> list[AlignmentAnchor]:
        """Return typed source-to-target anchors in source order.

        Only stamps that contain both the source and target timelines
        contribute to the result. Pairs are ordered by source coordinate.

        Args:
            target_timeline_id: The timeline to extract target coordinates
                for.

        Returns:
            Typed alignment anchors sorted by source coordinate.

        Raises:
            ValueError: If ``target_timeline_id`` equals
                ``source_timeline_id``.
        """
        if target_timeline_id == self.source_timeline_id:
            raise ValueError(
                f"target_timeline_id '{target_timeline_id}' cannot be the "
                f"same as source_timeline_id '{self.source_timeline_id}'"
            )
        known_targets = {
            timeline_id
            for stamp in self.stamps
            for timeline_id in stamp.present_timelines
        }
        if target_timeline_id not in known_targets:
            raise KeyError(
                f"Unknown target timeline ID {target_timeline_id!r} on MatchLine"
            )
        anchors: list[AlignmentAnchor] = []
        for stamp in self.stamps:
            target_coord = stamp.coordinates.get(target_timeline_id)
            if target_coord is None:
                continue
            anchors.append(
                AlignmentAnchor(
                    timeline_a_id=self.source_timeline_id,
                    coordinate_a=stamp.coordinates[self.source_timeline_id],
                    timeline_b_id=target_timeline_id,
                    coordinate_b=target_coord,
                )
            )
        return anchors

    def _get_float_alignment_pairs(
        self, target_timeline_id: str
    ) -> list[tuple[float, float]]:
        """Return private float anchor pairs for interpolation internals."""
        return [
            (float(anchor.coordinate_a.value), float(anchor.coordinate_b.value))
            for anchor in self.get_alignment_anchors(target_timeline_id)
        ]

    def get_coordinates_for(
        self,
        timeline_id: str,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Return one stored timeline coordinate column.

        Args:
            timeline_id: Timeline column to retrieve.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.

        Raises:
            KeyError: If the timeline does not occur in the line.
        """
        if not isinstance(timeline_id, str):
            raise TypeError("get_coordinates_for requires one timeline-ID string")
        coordinates = [
            IdCoordinate.from_coordinate(stamp.coordinates[timeline_id], timeline_id)
            for stamp in self.stamps
            if timeline_id in stamp.coordinates
        ]
        if not coordinates:
            raise KeyError(f"Unknown timeline ID {timeline_id!r} on MatchLine")
        return format_coordinates(
            coordinates,
            format=format,
            rounding=rounding,
            scalar=False,
            series_name=timeline_id,
        )

    def get_coordinate_at(
        self,
        at: CoordinateInput,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> CoordinateResult | pd.Series:
        """Resolve one exact source anchor to a target coordinate.

        Args:
            at: Exact source position.
            timeline_id: Requested target timeline.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            The exact target coordinate projection.
        """
        if not (
            not isinstance(at, bool)
            and isinstance(at, (int, float, Fraction, Coordinate))
        ):
            raise TypeError("get_coordinate_at requires one scalar coordinate input")
        if not self.stamps:
            raise KeyError("MatchLine contains no source anchors")
        source_axis = self.stamps[0].coordinates[self.source_timeline_id]
        if isinstance(at, IdCoordinate):
            if at.timeline_id != self.source_timeline_id:
                raise ValueError(
                    f"IdCoordinate source {at.timeline_id!r} does not match "
                    f"MatchLine source {self.source_timeline_id!r}"
                )
            source_unit = at.unit
        else:
            if timeline_id is None:
                raise ValueError(
                    "timeline_id is required for raw or plain MatchLine queries"
                )
            source_unit = at.unit if isinstance(at, Coordinate) else None
        if source_unit is not None and source_unit != source_axis.unit:
            raise ValueError(
                f"Coordinate unit {source_unit} does not match source unit "
                f"{source_axis.unit}"
            )
        source = Coordinate(
            at.value if isinstance(at, Coordinate) else at,
            source_axis.unit,
            number_type=source_axis.number_type,
        )
        source_value = source.value
        matches = [
            stamp
            for stamp in self.stamps
            if stamp.coordinates[self.source_timeline_id].value == source_value
        ]
        if not matches:
            raise KeyError(
                f"No exact MatchLine anchor at {source_value!r} on "
                f"{self.source_timeline_id!r}"
            )
        if len(matches) > 1:
            raise ValueError(f"Competing MatchLine anchors at {source_value!r}")
        stamp = matches[0]
        if timeline_id is None:
            candidates = [
                key for key in stamp.coordinates if key != self.source_timeline_id
            ]
            if not candidates:
                raise KeyError("Matched stamp has no non-source coordinate")
            if len(candidates) != 1:
                raise ValueError(f"Matched stamp has competing targets {candidates}")
            timeline_id = candidates[0]
        if timeline_id not in stamp.coordinates:
            raise KeyError(
                f"Matched stamp has no coordinate on timeline {timeline_id!r}"
            )
        result = IdCoordinate.from_coordinate(
            stamp.coordinates[timeline_id], timeline_id
        )
        return format_coordinates(
            [result],
            format=format,
            rounding=rounding,
            scalar=True,
            series_name=timeline_id,
        )

    def get_coordinates_at(
        self,
        at: CoordinateCollection,
        timeline_id: str | None = None,
        *,
        format: CoordinateFormat = "id_coordinate",
        rounding: Rounding = "round",
    ) -> list[CoordinateResult] | pd.Series:
        """Resolve exact source anchors for a coordinate collection.

        Args:
            at: Exact source positions to resolve atomically.
            timeline_id: Requested target timeline.
            format: Requested coordinate output format.
            rounding: Integral projection mode.

        Returns:
            A list of projections or canonical-value Series.
        """
        values, index = validate_coordinate_collection(at)
        if not values and timeline_id is None:
            raise ValueError("timeline_id is required for an empty MatchLine query")
        if timeline_id is not None and not any(
            timeline_id in stamp.coordinates for stamp in self.stamps
        ):
            raise KeyError(f"Unknown target timeline ID {timeline_id!r} on MatchLine")
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
                next(
                    (
                        stamp.coordinates[timeline_id].number_type
                        for stamp in self.stamps
                        if timeline_id is not None and timeline_id in stamp.coordinates
                    ),
                    None,
                )
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
        """Dispatch a scalar or plural exact source-position query.

        Args:
            at: One exact source position or a coordinate collection.
            timeline_id: Requested target timeline when required.
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
                "MatchLine.get_coordinate accepts coordinate inputs; use "
                "get_coordinates_for for a timeline column"
            ),
            format=format,
            rounding=rounding,
        )

    @classmethod
    def from_claims(
        cls,
        claims: list[MatchClaim],
        source_timeline_id: str,
        *,
        groups: dict[str, TimelineGroup] | None = None,
        timeline_ids: set[str] | None = None,
        id_pattern: str | None = None,
        include_domains: set[Domain] | None = None,
        include_units: set[TimeUnit] | None = None,
    ) -> MatchLine:
        """Build a MatchLine from claims via a MatchGraph.

        Constructs a MatchGraph from the supplied claims, optionally
        extends it to groups, extracts MatchStamps, and orders them by
        coordinate on the source timeline.

        Args:
            claims: List of MatchClaims to resolve.
            source_timeline_id: The timeline whose coordinates define
                the ordering.
            groups: Dict of group_id -> TimelineGroup for group extension
                (see :meth:`MatchGraph.extend_to_groups`). If None, no group
                extension is performed.
            timeline_ids: Only extend to these timeline IDs.
            id_pattern: Regex filter for timeline IDs.
            include_domains: Only extend to timelines in these domains.
            include_units: Only extend to timelines with these units.

        Returns:
            A MatchLine with stamps sorted by source coordinate.
        """
        graph = MatchGraph(claims)

        if groups is not None:
            graph = graph.extend_to_groups(
                groups,
                timeline_ids=timeline_ids,
                id_pattern=id_pattern,
                include_domains=include_domains,
                include_units=include_units,
            )

        stamps = graph.get_stamps()
        return cls(
            source_timeline_id=source_timeline_id,
            stamps=stamps,
        )

    @classmethod
    def from_graphs(
        cls,
        graphs: list[MatchGraph],
        source_timeline_id: str,
    ) -> MatchLine:
        """Build a MatchLine from multiple MatchGraphs.

        Merges MatchStamps from several MatchGraphs (the Hendrix M6-M9
        pattern) into a single ordered sequence. Duplicate stamps
        (same source coordinate) are deduplicated, keeping the stamp
        with the most timelines.

        Args:
            graphs: List of MatchGraphs to merge.
            source_timeline_id: The timeline whose coordinates define
                the ordering.

        Returns:
            A MatchLine with merged, deduplicated stamps sorted by
            source coordinate.
        """
        all_stamps: list[MatchStamp] = []
        for graph in graphs:
            all_stamps.extend(graph.get_stamps())

        # Deduplicate: if two stamps have the same source coordinate,
        # keep the one with more timelines (richer information).
        seen: dict[int | float | Fraction, MatchStamp] = {}
        for stamp in all_stamps:
            coordinate = stamp.coordinates.get(source_timeline_id)
            if coordinate is None:
                continue
            coord = coordinate.value
            existing = seen.get(coord)
            if existing is None or stamp.n_timelines > existing.n_timelines:
                seen[coord] = stamp

        return cls(
            source_timeline_id=source_timeline_id,
            stamps=list(seen.values()),
        )

    def save_as(
        self,
        filepath: str | Path,
        *,
        format: str = "match",
        context: MatchFileContext | None = None,
    ) -> Path:
        """Export this MatchLine to a file.

        Args:
            filepath: Output file path.  If the extension matches a known
                format, the format is inferred (e.g. ``.match``).
            format: Export format.  Currently supported: ``"match"``.
            context: Supplementary data for format-specific fields.
                Required for full ``.match`` export; if ``None``, a minimal
                file with coordinate-only placeholder data is produced.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If the format is not supported.

        Examples:
            >>> line.save_as("output.match")  # minimal placeholder export
            >>> line.save_as("output.match", context=ctx)  # rich export
        """
        from timetoalign.alignment.match_format import write_match_file

        filepath = Path(filepath)

        # Infer format from extension if it matches a known format
        ext = filepath.suffix.lower()
        if ext == ".match":
            format = "match"

        if format != "match":
            raise ValueError(
                f"Unsupported export format: {format!r}. "
                f"Currently supported: 'match'."
            )

        return write_match_file(filepath, _RawMatchLineExportView(self), context)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for storage.

        Returns:
            Dict with ``source_timeline_id`` and ``stamps``.
        """
        return {
            "source_timeline_id": self.source_timeline_id,
            "stamps": [s.to_dict(format="graph") for s in self.stamps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchLine:
        """Deserialize from dictionary.

        Args:
            data: Dict as produced by ``to_dict()``.

        Returns:
            A new MatchLine.
        """
        return cls(
            source_timeline_id=data["source_timeline_id"],
            stamps=[MatchStamp.from_dict(s) for s in data["stamps"]],
        )

    def __repr__(self) -> str:
        targets = self.target_timeline_ids()
        target_str = ", ".join(sorted(targets)) if targets else "none"
        return (
            f"MatchLine(source='{self.source_timeline_id}', "
            f"stamps={self.n_stamps}, targets=[{target_str}])"
        )
