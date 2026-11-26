"""MatchStamp and MatchIntervalStamp: aligned positions across timelines.

A :class:`MatchStamp` holds one coordinate per timeline a MatchClaim reaches
at a position; a :class:`MatchIntervalStamp` holds one interval per timeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import pandas as pd

from timetoalign.alignment.claims import MatchClaim
from timetoalign.alignment.filters import ClaimFilter
from timetoalign.core import Coordinate, IdCoordinate, Interval, TimeUnit
from timetoalign.core.retrieval import (
    CoordinateFormat,
    CoordinateResult,
    KeyCollection,
    Rounding,
    coordinate_from_dict,
    coordinate_to_dict,
    validate_key_collection,
)
from timetoalign.core.timestamp import (
    ConversionMapsSpec,
    Stamp,
    _DisplayRow,
    _format_stamp_value,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.maps.base import ConversionMap


@dataclass(frozen=True)
class MatchStamp(Stamp["AlignmentBundle"]):
    """A synchronized timestamp across multiple timelines.

    A MatchStamp represents a single coordinate (or instant) that has been
    synchronized across multiple timelines by the anchors MatchClaims state
    and/or by edges the MatchGraph inferred through group membership.

    Contains one coordinate per participating timeline, each pair of
    timelines linked by a stated or an inferred edge.

    Attributes:
        coordinates: Dictionary of timeline ID to canonical coordinate.
        stated_pairs: The (tl_a, tl_b) timeline pairs a MatchClaim's anchor
            links.
        inferred_pairs: The (tl_a, tl_b) timeline pairs linked by no stated
            anchor: in a bundle's stamps, inferred through a group, a
            parent/child offset or a WarpMap transfer; in a MatchGraph's
            own stamps, its :class:`InferredEdge` s.
        axis: Identified source coordinate derived from storage.
        source: Bundle that produced the stamp.
        source_id: Timeline ID used for the query.
        is_interpolated: Whether the stamp used interpolated transfer.
        conversion_maps: Conversion maps available to unit lookup and display.
            Opt-in: defaults to ``False``.
        interval_claims: The interval claims the stamp's coordinates were
            read through, each once: those covering the queried position, in
            relevance order, then those covering its position on each member
            of the queried timeline's groups; the claims themselves, never
            copies.

    Examples:
        >>> stamp = MatchStamp(
        ...     coordinates={
        ...         "score": Coordinate(100.0, TimeUnit.quarters),
        ...         "audio": Coordinate(45.5, TimeUnit.seconds),
        ...     },
        ...     source_id="score",
        ...     stated_pairs=[("score", "audio")],
        ... )
        >>> stamp.get_coordinate_for("audio", format="float")
        45.5
    """

    _affordances: ClassVar[tuple[str, ...]] = (
        "stamp.get_coordinate_for(<tl_id>)",
        "stamp.get_coordinates_for(<tl_ids>)",
        "stamp.get_unit(<unit>)",
        "stamp.get_conversion_for(<key>)",
    )
    _text_tags: ClassVar[bool] = True

    coordinates: dict[str, Coordinate] = field(default_factory=dict)
    source_id: str = ""
    stated_pairs: list[tuple[str, str]] = field(default_factory=list)
    inferred_pairs: list[tuple[str, str]] = field(default_factory=list)
    source: "AlignmentBundle | None" = None
    is_interpolated: bool = False
    conversion_maps: ConversionMapsSpec = False
    interval_claims: list[MatchClaim] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Isolate mutable containers from callers and serialized data.

        The coordinate dict and the edge and claim lists are the stamp's own;
        the claims in the list are not copied. A claim is frozen (only a
        free claim joining a bundle takes the id the bundle gives it, and
        stays the same claim), so sharing it cannot let the caller change
        what the stamp says; the stamp names the claims that answered, not
        copies of them, which would cost a copy of every claim at every
        retrieval.
        """
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
            self, "stated_pairs", [tuple(edge) for edge in self.stated_pairs]
        )
        object.__setattr__(
            self, "inferred_pairs", [tuple(edge) for edge in self.inferred_pairs]
        )
        normalized_claims: list[MatchClaim] = []
        for claim in self.interval_claims:
            if not isinstance(claim, MatchClaim) or not claim.is_interval:
                raise TypeError(
                    "MatchStamp interval_claims must contain interval MatchClaim values"
                )
            normalized_claims.append(claim)
        object.__setattr__(self, "interval_claims", normalized_claims)

    def _interval_candidate_rows(
        self, *, for_display: bool = False
    ) -> list[tuple[MatchClaim, str, Coordinate | None]]:
        """Return every interval claim's mapped counterpart candidate.

        Each claim is mapped from the timeline the stamp read it on: the
        queried one, or a member of its groups at the query's position there
        (``AlignmentBundle._interval_readings``). The positions are those of
        the bundle's groups now, so a claim may no longer be readable, for
        instance when the groups were edited after the stamp was made.

        Args:
            for_display: Leave out the row of a claim that cannot be read
                (whatever it raises), as :meth:`_conversion_rows` leaves out
                a C-Map failing at the coordinate: the rows then serve
                ``str``, ``repr`` and ``_repr_html_``, which must not raise.
                Without it, as on every getter asking for a value, the
                failure propagates.

        Raises:
            ValueError: Without *for_display*, if a claim holds none of the
                query's current positions.
        """
        if not self.interval_claims or self.source is None:
            return []
        query = self.coordinates[self.source_id]
        try:
            positions = self.source._group_positions(self.source_id, query)
        except Exception:
            if not for_display:
                raise
            positions = {self.source_id: query}
        rows: list[tuple[MatchClaim, str, Coordinate | None]] = []
        for claim in self.interval_claims:
            try:
                read_id, read_at = self.source._interval_claim_reading(claim, positions)
                target_id, coordinate, _ = self.source._interval_claim_candidate(
                    claim, read_id, read_at
                )
            except Exception:
                if not for_display:
                    raise
                continue
            rows.append((claim, target_id, coordinate))
        return rows

    def _displayed_claims(
        self,
    ) -> list[tuple[int, MatchClaim, str, Coordinate | None]]:
        """The candidate rows the displays show, each with its claim's number.

        The number is the claim's place in :attr:`interval_claims`, from 1,
        so a claim left out of the display does not renumber the others.
        """
        number = {
            id(claim): index for index, claim in enumerate(self.interval_claims, 1)
        }
        return [
            (number[id(claim)], claim, timeline_id, coordinate)
            for claim, timeline_id, coordinate in self._interval_candidate_rows(
                for_display=True
            )
        ]

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
    def n_stated_pairs(self) -> int:
        """Number of timeline pairs a MatchClaim's anchor links."""
        return len(self.stated_pairs)

    @property
    def n_inferred_pairs(self) -> int:
        """Number of timeline pairs linked by an inferred edge."""
        return len(self.inferred_pairs)

    def has_timeline(self, timeline_id: str) -> bool:
        """Check if timeline is in this stamp."""
        return timeline_id in self.coordinates

    def _error_name(self) -> str:
        """Errors name the stamp by its class."""
        return "MatchStamp"

    def _unit_maps_on(
        self, timeline_id: str, unit: TimeUnit
    ) -> "list[ConversionMap[Any]]":
        """The bundle timeline's maps into *unit*."""
        if self.source is None:
            return []
        return self.source.get_timeline(timeline_id)._get_unit_maps(unit)

    def filter_by_timelines(
        self,
        *,
        within: set[str] | None = None,
        id_pattern: str | None = None,
    ) -> "MatchStamp":
        """Create filtered stamp with subset of timelines.

        The stamp's source (:attr:`source_id`, the queried timeline) is
        always kept; the filter selects the others.

        Args:
            within: Only these timelines' coordinates beside the source
                (None = all).
            id_pattern: Regex filter for the other timelines' IDs.

        Returns:
            New MatchStamp with filtered timelines.
        """
        return self._narrowed(
            ClaimFilter.from_kwargs(within=within, id_pattern=id_pattern)
        )

    def _narrowed(
        self,
        filt: ClaimFilter,
        *,
        timelines: dict[str, Any] | None = None,
    ) -> "MatchStamp":
        """The stamp holding its source and the other timelines *filt* passes.

        The one narrowing of a stamp's timelines
        (:meth:`ClaimFilter.matches_timeline`), for :meth:`filter_by_timelines`
        and the bundle's matchstamp getters. The source, the queried
        timeline, is always kept, whatever the filter, and with it the
        interval claims; the filter selects the others. The pairs between
        kept timelines are kept.

        Args:
            filt: The filter.
            timelines: The timelines by id, where the filter reads their
                domains or units.
        """
        coordinates = {
            tl_id: coord
            for tl_id, coord in self.coordinates.items()
            if tl_id == self.source_id
            or filt.matches_timeline(tl_id, timelines=timelines)
        }
        return MatchStamp(
            coordinates=coordinates,
            source_id=self.source_id,
            stated_pairs=[
                (a, b)
                for a, b in self.stated_pairs
                if a in coordinates and b in coordinates
            ],
            inferred_pairs=[
                (a, b)
                for a, b in self.inferred_pairs
                if a in coordinates and b in coordinates
            ],
            source=self.source,
            is_interpolated=self.is_interpolated,
            conversion_maps=self.conversion_maps,
            interval_claims=self.interval_claims,
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
                "stated_pairs": list(self.stated_pairs),
                "inferred_pairs": list(self.inferred_pairs),
            }

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

        # A timeline of several groups is listed under each of them; a
        # standalone one under its own id.
        membership = {} if self.source is None else self.source.timeline_to_groups
        grouped: dict[str, dict[str, dict[str, object]]] = {}
        for timeline_id, coordinate in self.coordinates.items():
            coordinate_data = coordinate_to_dict(coordinate)
            for group_id in membership.get(timeline_id, (timeline_id,)):
                grouped.setdefault(group_id, {})[timeline_id] = coordinate_data

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
            stated_pairs=[tuple(e) for e in data.get("stated_pairs", [])],
            inferred_pairs=[tuple(e) for e in data.get("inferred_pairs", [])],
        )

    def __repr__(self) -> str:
        entries = [
            f"{timeline_id}="
            f"{_format_stamp_value(coordinate.value, coordinate.unit.value)}"
            for timeline_id, coordinate in self.coordinates.items()
        ]
        entries.extend(
            self._claim_candidate_text(index, claim, timeline_id, coordinate)
            for index, claim, timeline_id, coordinate in self._displayed_claims()
        )
        return f"MatchStamp({', '.join(entries)})"

    def _coordinate_rows(self) -> list[_DisplayRow]:
        """Every coordinate in storage order, tagged by the pairs naming it.

        ``stated`` where a stated pair names the timeline, else ``inferred``
        where an inferred pair does, else untagged.
        """
        stated = {timeline_id for pair in self.stated_pairs for timeline_id in pair}
        inferred = {timeline_id for pair in self.inferred_pairs for timeline_id in pair}
        rows: list[_DisplayRow] = []
        for timeline_id, coordinate in self.coordinates.items():
            value = _format_stamp_value(coordinate.value, coordinate.unit.value)
            if timeline_id in stated:
                tag, look = "stated", "strong"
            elif timeline_id in inferred:
                tag, look = "inferred", "inferred"
            else:
                tag, look = "", "plain"
            rows.append(_DisplayRow(timeline_id, value, tag, look, tag))
        return rows

    def _claim_rows(self) -> list[_DisplayRow]:
        """One row per interval claim: its counterpart, or ``MISSING``.

        A claim the stamp can no longer read has no row
        (:meth:`_interval_candidate_rows`).
        """
        rows: list[_DisplayRow] = []
        for index, claim, timeline_id, coordinate in self._displayed_claims():
            value = (
                "MISSING"
                if coordinate is None
                else _format_stamp_value(coordinate.value, coordinate.unit.value)
            )
            tag = f"interval claim {index} ({timeline_id})"
            label = f"{claim.timeline_a_id}<->{claim.timeline_b_id}"
            rows.append(_DisplayRow(label, value, tag, "muted", tag))
        return rows

    def _text_header(self) -> str:
        """``MatchStamp (<n> timelines, <m> edges)``."""
        return (
            f"MatchStamp ({self.n_timelines} timelines, "
            f"{self.n_stated_pairs + self.n_inferred_pairs} edges)"
        )

    def _html_badge(self) -> str:
        """The numbers of timelines and edges."""
        n_edges = self.n_stated_pairs + self.n_inferred_pairs
        return (
            f" <span style='background: #e3f2fd; padding: 0 4px; "
            f"border-radius: 3px; font-size: 0.8em;'>"
            f"{self.n_timelines} timelines, {n_edges} edges</span>"
        )


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
