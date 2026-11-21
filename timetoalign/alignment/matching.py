"""Matching: Greedy sequential note matching for producing MatchClaims.

This module provides algorithms for matching note events between two
timelines based on shared attributes (e.g., pitch name + staff number).
The primary use case is aligning performance recordings with score data.

The matching produces :class:`MatchClaim` objects that connect events of two
timelines, enabling cross-group coordinate transfer via WarpMaps in an
:class:`AlignmentBundle`. A side whose rows name no event of its timeline
gets a :class:`~timetoalign.core.Projection` there first.

Algorithm:
    Greedy sequential matching: for each source note (in order), find the
    first target note with matching attributes. Once matched, the target
    note is removed from the candidate pool. This ensures one-to-one matching
    and preserves temporal ordering.

Reference:
    ``dashboard/processing/notebooks/repovizz_parsing.py`` —
    ``match_repovizz_notes_with_abc()``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from timetoalign.alignment.claims import MatchClaim
from timetoalign.core import (
    Coordinate,
    EventReference,
    Projection,
    Provenance,
    TimeUnit,
)
from timetoalign.core.ids import quote_id

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Utilities


def _convert_coordinate_to_float(value: Any) -> float:
    """Convert a coordinate value to float, handling fraction strings.

    Args:
        value: A numeric value, fraction string (e.g. "707/2"), or numeric string.

    Returns:
        The value converted to float.

    Raises:
        ValueError: If the value cannot be converted to float.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Try to parse as a fraction first (e.g., "707/2")
        try:
            return float(Fraction(value))
        except ValueError:
            # Fall back to regular float conversion
            return float(value)
    raise ValueError(f"Cannot convert {type(value).__name__} to float: {value}")


# endregion

# region MatchResult


@dataclass
class MatchResult:
    """Result of a greedy sequential matching operation.

    Attributes:
        matched: DataFrame of matched pairs with columns from both source and target,
            plus ``source_coord`` and ``target_coord`` for the matched coordinates.
        unmatched_source: DataFrame of source rows that found no match.
        unmatched_target: DataFrame of target rows that were not matched.
        match_claims: List of MatchClaim objects, one per matched pair, each
            joining the pair's two events.
    """

    matched: pd.DataFrame
    unmatched_source: pd.DataFrame
    unmatched_target: pd.DataFrame
    match_claims: list[MatchClaim] = field(default_factory=list)

    @property
    def n_matched(self) -> int:
        """Number of matched pairs."""
        return len(self.matched)

    @property
    def n_unmatched_source(self) -> int:
        """Number of unmatched source rows."""
        return len(self.unmatched_source)

    @property
    def n_unmatched_target(self) -> int:
        """Number of unmatched target rows."""
        return len(self.unmatched_target)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict of matching statistics."""
        return {
            "matched": self.n_matched,
            "unmatched_source": self.n_unmatched_source,
            "unmatched_target": self.n_unmatched_target,
            "match_claims": len(self.match_claims),
        }


# endregion


# region Matching Functions


def match_notes_by_attributes(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    match_columns: Sequence[str],
    source_coord_column: str = "start",
    target_coord_column: str = "start",
    *,
    source_timeline: Timeline | None = None,
    target_timeline: Timeline | None = None,
    source_id_column: str | None = None,
    target_id_column: str | None = None,
    source_unit: TimeUnit | str | None = None,
    target_unit: TimeUnit | str | None = None,
    create_claims: bool = True,
    provenance: Provenance | None = None,
) -> MatchResult:
    """Match notes between two DataFrames using greedy sequential matching.

    For each source note (in order), finds the first target note with matching
    values in the specified columns. Once a target note is matched, it is
    removed from the candidate pool.

    This is the core matching algorithm used to align EEP performance notes
    with unfolded ABC score notes by pitch name + staff number.

    Each claim joins two events, one on each timeline. A side given an id
    column names, per row, the event of its timeline with that id (which
    must be one); a side without one states positions only, so each of its
    matched rows becomes a :class:`~timetoalign.core.Projection` added to
    its timeline, standing for the other side's event where that side names
    one (``projection_of``) and for no event otherwise. Every coordinate is
    placed on its timeline's own axis: a value in another unit is converted
    through the timeline's C-Map.

    Args:
        source_df: DataFrame of source notes (e.g., EEP performance notes).
            Must contain the ``match_columns`` and ``source_coord_column``.
        target_df: DataFrame of target notes (e.g., unfolded ABC score notes).
            Must contain the ``match_columns`` and ``target_coord_column``.
        match_columns: Column names to match on (e.g., ``["pitch", "staff"]``).
        source_coord_column: Column in source_df containing the coordinate
            to use in MatchClaims (e.g., onset in seconds); a string such as
            ``"707/2"`` is read as an exact fraction.
        target_coord_column: Column in target_df containing the coordinate
            to use in MatchClaims (e.g., onset in quarterbeats).
        source_timeline: The timeline of the source rows (side A of every
            MatchClaim); required when ``create_claims`` is True.
        target_timeline: The timeline of the target rows (side B).
        source_id_column: Column in source_df naming each row's event on
            ``source_timeline``; without it, the source side gets Projections.
        target_id_column: Column in target_df naming each row's event on
            ``target_timeline``; without it, the target side gets Projections.
        source_unit: Unit of the source coordinates; ``None`` (the default)
            is the source timeline's unit.
        target_unit: Unit of the target coordinates; ``None`` is the target
            timeline's unit.
        create_claims: If True, generate MatchClaim objects for each match
            (and add the Projections they need to the timelines).
        provenance: How the MatchClaims were made, recorded on each; by
            default the method ``"inferred"`` and a note naming the matching
            rule.

    Returns:
        MatchResult with matched pairs, unmatched rows, and MatchClaims.

    Raises:
        ValueError: If required columns are missing from either DataFrame,
            claims are asked for without both timelines, an id column names
            no event of its timeline, or a coordinate cannot be placed on its
            timeline. No Projection is added then.

    Examples:
        >>> # Match EEP notes with ABC notes by pitch and staff
        >>> result = match_notes_by_attributes(
        ...     source_df=eep_notes,
        ...     target_df=abc_notes,
        ...     match_columns=["pitch", "staff"],
        ...     source_coord_column="start",
        ...     target_coord_column="quarterbeats_playthrough",
        ...     source_timeline=performance,
        ...     target_timeline=score,
        ...     source_id_column="id",
        ... )
        >>> result.summary()
        {'matched': 3742, 'unmatched_source': 14, 'unmatched_target': 8}
    """
    # Validate columns exist
    for name in match_columns:
        if name not in source_df.columns:
            raise ValueError(
                f"Match column '{name}' not found in source_df. "
                f"Available: {list(source_df.columns)}"
            )
        if name not in target_df.columns:
            raise ValueError(
                f"Match column '{name}' not found in target_df. "
                f"Available: {list(target_df.columns)}"
            )

    if source_coord_column not in source_df.columns:
        raise ValueError(
            f"Source coordinate column '{source_coord_column}' not found. "
            f"Available: {list(source_df.columns)}"
        )
    if target_coord_column not in target_df.columns:
        raise ValueError(
            f"Target coordinate column '{target_coord_column}' not found. "
            f"Available: {list(target_df.columns)}"
        )
    for name, frame, side in (
        (source_id_column, source_df, "source"),
        (target_id_column, target_df, "target"),
    ):
        if name is not None and name not in frame.columns:
            raise ValueError(
                f"{side.capitalize()} id column '{name}' not found. "
                f"Available: {list(frame.columns)}"
            )
    if create_claims and (source_timeline is None or target_timeline is None):
        raise ValueError(
            "MatchClaims join events of two timelines: pass source_timeline and "
            "target_timeline, or create_claims=False."
        )

    # Convert target to list of records for pop-based matching
    target_records = target_df.reset_index(drop=True).to_dict(orient="records")

    matched_records: list[dict[str, Any]] = []
    unmatched_source_records: list[dict[str, Any]] = []

    for _, source_row in source_df.iterrows():
        source_dict = source_row.to_dict()
        found = False

        for i, target_record in enumerate(target_records):
            # Check if all match columns agree
            if all(
                source_dict.get(name) == target_record.get(name)
                for name in match_columns
            ):
                # Match found — pop the target record
                matched_target = target_records.pop(i)

                # Build matched record with source + target coords
                match_record = {
                    "source_coord": _convert_coordinate_to_float(
                        source_dict[source_coord_column]
                    ),
                    "target_coord": _convert_coordinate_to_float(
                        matched_target[target_coord_column]
                    ),
                }
                # Include match columns for reference
                for name in match_columns:
                    match_record[name] = source_dict[name]
                # Include full source row with "source_" prefix
                for k, v in source_dict.items():
                    match_record[f"source_{k}"] = v
                # Include full target row with "target_" prefix
                for k, v in matched_target.items():
                    match_record[f"target_{k}"] = v

                matched_records.append(match_record)
                found = True
                break

        if not found:
            unmatched_source_records.append(source_dict)

    # Build DataFrames
    matched_df = (
        pd.DataFrame.from_records(matched_records)
        if matched_records
        else pd.DataFrame()
    )
    unmatched_source_df = (
        pd.DataFrame.from_records(unmatched_source_records)
        if unmatched_source_records
        else pd.DataFrame()
    )
    unmatched_target_df = (
        pd.DataFrame.from_records(target_records) if target_records else pd.DataFrame()
    )

    # Generate MatchClaims
    claims: list[MatchClaim] = []
    if create_claims and len(matched_df) > 0:
        assert source_timeline is not None and target_timeline is not None
        if provenance is None:
            provenance = Provenance(
                method="inferred",
                note="greedy sequential matching by " + "+".join(match_columns),
            )
        claims = _claims_between_events(
            matched_df,
            (source_timeline, source_coord_column, source_id_column, source_unit),
            (target_timeline, target_coord_column, target_id_column, target_unit),
            provenance,
        )

    module_logger.info(
        f"Matching complete: {len(matched_df)} matched, "
        f"{len(unmatched_source_df)} unmatched source, "
        f"{len(unmatched_target_df)} unmatched target"
    )

    return MatchResult(
        matched=matched_df,
        unmatched_source=unmatched_source_df,
        unmatched_target=unmatched_target_df,
        match_claims=claims,
    )


def _claims_between_events(
    matched: pd.DataFrame,
    source: tuple[Timeline, str, str | None, TimeUnit | str | None],
    target: tuple[Timeline, str, str | None, TimeUnit | str | None],
    provenance: Provenance,
) -> list[MatchClaim]:
    """Return one MatchClaim per matched pair, between two events.

    Each side is ``(timeline, coordinate column, id column, unit)``, its
    columns read from the ``source_`` / ``target_`` columns of *matched*. A
    side with an id column names its events, refused unless the timeline
    holds them; a side without one gets a Projection per row, standing for
    the other side's event where that side names one. Every check and
    conversion runs before the first Projection is added.
    """
    sides: list[tuple[Timeline, list[Coordinate], list[str] | None]] = []
    for prefix, (timeline, coord_column, id_column, unit) in (
        ("source", source),
        ("target", target),
    ):
        positions = _positions(matched[f"{prefix}_{coord_column}"], timeline, unit)
        ids = None
        if id_column is not None:
            ids = [str(value) for value in matched[f"{prefix}_{id_column}"]]
            _refuse_unknown_events(ids, timeline)
        sides.append((timeline, positions, ids))

    event_ids: list[list[str]] = []
    for side, (timeline, positions, ids) in enumerate(sides):
        if ids is not None:
            event_ids.append(ids)
            continue
        other_timeline, _, other_ids = sides[1 - side]
        projections = [
            Projection(
                start=position,
                projection_of=(
                    None
                    if other_ids is None
                    else EventReference(
                        timeline_id=other_timeline.id, event_id=other_ids[row]
                    )
                ),
            )
            for row, position in enumerate(positions)
        ]
        event_ids.append(timeline.add_events(projections))

    (timeline_a, positions_a, _), (timeline_b, positions_b, _) = sides
    return [
        MatchClaim.from_events(
            {"id": id_a, "start": position_a},
            timeline_a,
            {"id": id_b, "start": position_b},
            timeline_b,
            provenance=provenance,
        )
        for id_a, position_a, id_b, position_b in zip(
            event_ids[0], positions_a, event_ids[1], positions_b, strict=True
        )
    ]


def _positions(
    values: pd.Series, timeline: Timeline, unit: TimeUnit | str | None
) -> list[Coordinate]:
    """Place a column of coordinate values on *timeline*'s own axis.

    A value is read exactly (a fraction string such as ``"707/2"`` as a
    :class:`~fractions.Fraction`). In the timeline's unit it is written in
    the timeline's number type; in another *unit* it is converted through
    the timeline's C-Map.
    """
    unit = TimeUnit(timeline.unit) if unit is None else TimeUnit(unit)
    positions = []
    for value in values:
        if isinstance(value, str):
            exact = Fraction(value)
        else:
            # A NumPy scalar from the frame becomes the Python number it holds.
            exact = value.item() if hasattr(value, "item") else value
        if unit == timeline.unit:
            positions.append(Coordinate(exact, unit, number_type=timeline.number_type))
            continue
        converted = timeline.get_coordinate_at(
            Coordinate(exact, unit), format="coordinate"
        )
        positions.append(
            Coordinate(
                converted.value, converted.unit, number_type=converted.number_type
            )
        )
    return positions


def _refuse_unknown_events(ids: list[str], timeline: Timeline) -> None:
    """Refuse an id naming no event of *timeline* (its own events)."""
    table = timeline.get_events(include_children=False)._table
    held = (
        table.column("id").combine_chunks()
        if "id" in table.column_names
        else pa.array([], type=pa.string())
    )
    named = pa.array(ids, type=pa.string())
    unknown = named.filter(pc.invert(pc.is_in(named, value_set=held)))
    if len(unknown):
        raise ValueError(
            f"{len(unknown)} matched row(s) name no event of timeline "
            f"{quote_id(timeline.id)}, the first {quote_id(unknown[0].as_py())}"
        )


def prepare_eep_notes_for_matching(
    eep_df: pd.DataFrame,
    pitch_column: str = "pitch",
    staff_column: str = "staff",
) -> pd.DataFrame:
    """Prepare EEP notes for matching: filter rests, explode chords.

    Args:
        eep_df: DataFrame from EepNotesLoader with columns including
            ``pitch`` (note name, "rest", or comma-separated chord)
            and ``staff`` (instrument number).
        pitch_column: Column containing pitch information.
        staff_column: Column containing staff/instrument number.

    Returns:
        DataFrame with rests removed and chords exploded into separate rows,
        ready for matching.
    """
    # Filter rests
    notes_only = eep_df[eep_df[pitch_column] != "rest"].copy()

    # Explode chords: "G3,D4,B4,F5" -> separate rows
    notes_only[pitch_column] = notes_only[pitch_column].str.split(",")
    exploded = notes_only.explode(pitch_column).reset_index(drop=True)

    return exploded


def prepare_abc_notes_for_matching(
    abc_df: pd.DataFrame,
    pitch_column: str = "name",
    staff_column: str = "staff",
    tied_column: str = "tied",
    coord_column: str = "quarterbeats_playthrough",
) -> pd.DataFrame:
    """Prepare unfolded ABC notes for matching: keep sounding onsets, parse fractions.

    ms3's tie column holds ``1`` where a note is tied to the following one,
    ``-1`` where it is tied to by the previous one, ``0`` where both hold and
    nothing where the note is not tied. A note sounds anew only when it is
    not tied or starts a tie, so those rows are kept; the continuation and the
    end of a tie prolong a sound already started and are dropped.

    Args:
        abc_df: DataFrame from ms3-loaded unfolded notes TSV with columns
            ``name`` (pitch), ``staff``, ``tied`` (tie indicator), and
            ``quarterbeats_playthrough`` (unfolded coordinate).
        pitch_column: Column containing pitch name.
        staff_column: Column containing staff number.
        tied_column: Column containing the tie indicator (``1``, ``0``,
            ``-1`` or empty, numeric or as text; a text cell that is empty
            or holds only whitespace is empty). Rows holding ``0`` or ``-1``
            are dropped.
        coord_column: Column containing the unfolded coordinate (may contain
            fraction strings like ``"707/2"``).

    Returns:
        DataFrame of the sounding onsets, re-indexed from 0, ready for
        matching. A kept tie start carries its own notated duration, not the
        length of its tie. Includes a ``pitch`` column (renamed from
        pitch_column) for compatibility with :func:`match_notes_by_attributes`.
        The coordinate column values are converted to float.

    Raises:
        ValueError: If the tie column holds a value other than ``1``, ``0``,
            ``-1`` or an empty cell.
    """
    tied = abc_df[tied_column]
    if not pd.api.types.is_numeric_dtype(tied):
        # A TSV read as text holds an empty string where a note is not tied.
        tied = tied.mask(tied.astype("string").str.strip() == "")
    tied = pd.to_numeric(tied)
    unknown = sorted(set(tied.dropna().tolist()) - {-1, 0, 1})
    if unknown:
        raise ValueError(
            f"Column {tied_column!r} holds tie values other than 1, 0, -1: {unknown}"
        )
    sounds_anew = tied.isna() | (tied.fillna(0) == 1)
    note_onsets = abc_df[sounds_anew.to_numpy(dtype=bool)].copy()

    # Rename pitch column to "pitch" for matching compatibility
    if pitch_column != "pitch":
        note_onsets = note_onsets.rename(columns={pitch_column: "pitch"})

    # Convert coordinate column from fraction strings to float
    if not pd.api.types.is_numeric_dtype(note_onsets[coord_column]):
        note_onsets[coord_column] = note_onsets[coord_column].apply(
            lambda x: float(Fraction(str(x)))
        )

    return note_onsets.reset_index(drop=True)


# endregion
