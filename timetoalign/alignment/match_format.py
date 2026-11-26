"""Match format: Export support for the Vienna .match file format (v1.0.0).

This module provides the data structures and formatting functions needed to
write ``.match`` files from :class:`~timetoalign.MatchLine` data.

The Vienna ``.match`` format is specified at
https://cpjku.github.io/docs/match/specification/.

Data structures:

- :class:`SnoteRecord` — frozen dataclass for a score note line.
- :class:`NoteRecord` — frozen dataclass for a performance note line.
- :class:`MatchFileContext` — supplementary metadata and note lookups
  that bridge the coordinate-only :class:`~timetoalign.MatchStamp` data
  with the rich note-level fields required by ``.match``.

Formatting functions:

- :func:`format_snote_line` — ``snote(...)`` string.
- :func:`format_note_line` — ``note(...)`` string.
- :func:`format_match_line` — ``snote(...)-note(...).``
- :func:`format_deletion_line` — ``snote(...)-deletion.``
- :func:`format_insertion_line` — ``insertion-note(...).``
- :func:`format_header` — ``info(...)`` header lines.
- :func:`format_score_properties` — ``scoreprop(...)`` lines.
- :func:`write_match_file` — orchestrate full file output.

See Also:
    timetoalign.MatchLine
    timetoalign.MatchfileLoader
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pandas import DataFrame

    from timetoalign.alignment.matchline import MatchLine
    from timetoalign.alignment.stamps import MatchStamp

module_logger = logging.getLogger(__name__)


# region Record Dataclasses


@dataclass(frozen=True)
class SnoteRecord:
    """Score note record for ``.match`` format export.

    Each field corresponds to a positional element in the ``snote(...)``
    Prolog-style term.  See the Vienna match specification for semantics.

    Attributes:
        id: Score note identifier (e.g. ``"n1"``).
        pitch_name: Note name without accidental (e.g. ``"B"``).
        modifier: Accidental modifier: ``"n"`` (natural), ``"#"`` (sharp),
            ``"b"`` (flat), ``"bb"`` (double flat), ``"x"`` (double sharp).
        octave: Scientific octave number (e.g. 3).
        measure: Measure number (1-indexed).
        beat: Beat number within the measure (1-indexed).
        offset: Symbolic offset within the beat (e.g. ``"0"``, ``"1/16"``).
        duration: Symbolic duration (e.g. ``"1/8"``, ``"1/4"``).
        onset_in_beats: Onset position in quarter-beats (may be negative
            for anacrusis).
        offset_in_beats: End position in quarter-beats.
        attributes: Extra attributes (e.g. ``["v1", "staff1"]``).
    """

    id: str
    pitch_name: str
    modifier: str
    octave: int
    measure: int
    beat: int
    offset: str
    duration: str
    onset_in_beats: float
    offset_in_beats: float
    attributes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class NoteRecord:
    """Performance note record for ``.match`` format export.

    Each field corresponds to a positional element in the ``note(...)``
    Prolog-style term.

    Attributes:
        id: Performance note identifier (e.g. ``"n0"``).
        midi_pitch: MIDI pitch number (0–127).
        onset_tick: MIDI onset tick.
        offset_tick: MIDI offset tick.
        velocity: MIDI velocity (0–127).
        channel: MIDI channel (default 0).
        track: MIDI track (default 0).
    """

    id: str
    midi_pitch: int
    onset_tick: int
    offset_tick: int
    velocity: int
    channel: int = 0
    track: int = 0


# endregion


# region MatchFileContext


@dataclass
class MatchFileContext:
    """Supplementary data required for ``.match`` file export.

    This context object bridges the gap between the coordinate-only
    :class:`~timetoalign.MatchStamp` data and the rich note-level data
    required by the Vienna ``.match`` format.  It is typically constructed
    via :meth:`from_dataframes` or assembled manually.

    Attributes:
        piece: Piece name for the ``info(piece,...)`` header.
        composer: Composer name for the ``info(composer,...)`` header.
        performer: Performer name for the ``info(performer,...)`` header.
        score_filename: Score file name for the header.
        midi_filename: MIDI file name for the header.
        midi_clock_units: MIDI clock units (divisions per quarter note).
        midi_clock_rate: MIDI clock rate in microseconds per quarter note.
        score_notes: Lookup from score coordinate to list of
            :class:`SnoteRecord` objects at that coordinate.
        perf_notes: Lookup from performance coordinate to list of
            :class:`NoteRecord` objects at that coordinate.
        score_properties: Pre-formatted ``scoreprop(...)`` lines.
        deletions: Score notes with no performance match.
        insertions: Performance notes with no score match.
    """

    # Header metadata
    piece: str = ""
    composer: str = ""
    performer: str = ""
    score_filename: str = ""
    midi_filename: str = ""
    midi_clock_units: int = 480
    midi_clock_rate: int = 500000

    # Score note lookup: coordinate -> list[SnoteRecord]
    score_notes: dict[float, list[SnoteRecord]] = field(default_factory=dict)

    # Performance note lookup: coordinate -> list[NoteRecord]
    perf_notes: dict[float, list[NoteRecord]] = field(default_factory=dict)

    # Score properties (pre-formatted scoreprop lines)
    score_properties: list[str] = field(default_factory=list)

    # Deletions (score notes with no performance match)
    deletions: list[SnoteRecord] = field(default_factory=list)

    # Insertions (performance notes with no score match)
    insertions: list[NoteRecord] = field(default_factory=list)

    # region from_dataframes

    @classmethod
    def from_dataframes(
        cls,
        score_df: "DataFrame",
        perf_df: "DataFrame",
        match_result: Any = None,
        *,
        piece: str = "",
        composer: str = "",
        performer: str = "",
        score_filename: str = "",
        midi_filename: str = "",
        midi_clock_units: int = 480,
        midi_clock_rate: int = 500000,
        score_coord_column: str = "quarterbeats_playthrough",
        perf_coord_column: str = "start",
        score_pitch_column: str = "pitch",
        perf_pitch_column: str = "pitch",
        score_staff_column: str = "staff",
        score_duration_column: str = "duration_qb",
        score_measure_column: str = "mc",
        score_beat_column: str = "beat",
        score_offset_column: str = "mc_onset",
    ) -> MatchFileContext:
        """Build a :class:`MatchFileContext` from score and performance DataFrames.

        This factory is designed for the how03 use case: matching EEP
        performance notes against unfolded ABC score notes.  It constructs
        :class:`SnoteRecord` and :class:`NoteRecord` objects from DataFrame
        columns, keyed by their coordinates for lookup during export.

        Args:
            score_df: Score notes DataFrame (e.g. from ``prepare_abc_notes_for_matching``).
                Must contain the ``score_coord_column`` and note attribute columns.
            perf_df: Performance notes DataFrame (e.g. from ``prepare_eep_notes_for_matching``).
                Must contain the ``perf_coord_column`` and ``pitch`` column.
            match_result: Optional :class:`~timetoalign.alignment.matching.MatchResult`.
                If provided, unmatched target notes are recorded as deletions
                and unmatched source notes as insertions.
            piece: Piece name for the header.
            composer: Composer name for the header.
            performer: Performer name for the header.
            score_filename: Score file name for the header.
            midi_filename: MIDI file name for the header.
            midi_clock_units: MIDI divisions per quarter note.
            midi_clock_rate: Microseconds per quarter note.
            score_coord_column: Column in *score_df* with the score coordinate.
            perf_coord_column: Column in *perf_df* with the performance coordinate.
            score_pitch_column: Column in *score_df* with the pitch name.
            perf_pitch_column: Column in *perf_df* with the pitch name.
            score_staff_column: Column in *score_df* with the staff number.
            score_duration_column: Column in *score_df* with note duration
                (in quarter-beats).  Falls back to ``"duration"`` if missing.
            score_measure_column: Column in *score_df* with measure number.
            score_beat_column: Column in *score_df* with beat number.
            score_offset_column: Column in *score_df* with beat offset.

        Known limitation: the note ids come from the frames' index labels.
        A string label is the id itself; an integer label ``i`` gives
        ``n{i + 1}`` for a score note and ``n{i}`` for a performance note.
        Other labels are not checked: a float label gives an id such as
        ``n1.5``, and a tuple label (a ``MultiIndex``) of *score_df* raises
        ``TypeError``.

        Returns:
            A populated :class:`MatchFileContext`.
        """
        score_notes: dict[float, list[SnoteRecord]] = {}
        perf_notes: dict[float, list[NoteRecord]] = {}

        # Build SnoteRecords from score_df
        for idx, row in score_df.iterrows():
            coord = _safe_float(row.get(score_coord_column, 0))
            pitch_full = str(row.get(score_pitch_column, "C4"))

            pitch_name, modifier, octave = _parse_pitch(pitch_full)

            # Measure, beat, offset
            measure = (
                int(row.get(score_measure_column, 1))
                if score_measure_column in score_df.columns
                else 1
            )
            beat = (
                int(row.get(score_beat_column, 1))
                if score_beat_column in score_df.columns
                else 1
            )

            # Offset within beat
            offset_val = (
                row.get(score_offset_column, "0")
                if score_offset_column in score_df.columns
                else "0"
            )
            offset_str = _to_fraction_str(offset_val)

            # Duration
            dur_col = (
                score_duration_column
                if score_duration_column in score_df.columns
                else "duration"
            )
            dur_val = row.get(dur_col, "1/4") if dur_col in score_df.columns else "1/4"
            duration_str = _to_fraction_str(dur_val)

            # Onset/offset in beats
            onset_beats = coord
            dur_float = _safe_float(dur_val)
            offset_beats = onset_beats + dur_float

            # Staff attribute
            staff = (
                int(row.get(score_staff_column, 1))
                if score_staff_column in score_df.columns
                else 1
            )
            attributes = [f"staff{staff}"]

            snote_id = f"n{idx + 1}" if not isinstance(idx, str) else idx

            record = SnoteRecord(
                id=snote_id,
                pitch_name=pitch_name,
                modifier=modifier,
                octave=octave,
                measure=measure,
                beat=beat,
                offset=offset_str,
                duration=duration_str,
                onset_in_beats=onset_beats,
                offset_in_beats=offset_beats,
                attributes=attributes,
            )
            score_notes.setdefault(coord, []).append(record)

        # Build NoteRecords from perf_df
        for idx, row in perf_df.iterrows():
            coord = _safe_float(row.get(perf_coord_column, 0))
            pitch_str = str(row.get(perf_pitch_column, "C4"))

            # Derive MIDI pitch from pitch name
            midi_pitch = _pitch_name_to_midi(pitch_str)

            # For EEP data we typically have seconds, not ticks.
            # Convert to pseudo-ticks using clock rate.
            onset_tick = int(coord * midi_clock_units * 1_000_000 / midi_clock_rate)
            # Estimate duration from DataFrame if available
            dur = _safe_float(row.get("duration", 0.5))
            offset_tick = int(
                (coord + dur) * midi_clock_units * 1_000_000 / midi_clock_rate
            )

            velocity = int(row.get("velocity", 64))

            note_id = f"n{idx}" if not isinstance(idx, str) else idx

            record = NoteRecord(
                id=note_id,
                midi_pitch=midi_pitch,
                onset_tick=onset_tick,
                offset_tick=offset_tick,
                velocity=velocity,
            )
            perf_notes.setdefault(coord, []).append(record)

        # Handle unmatched notes from match_result
        deletions: list[SnoteRecord] = []
        insertions: list[NoteRecord] = []

        if match_result is not None:
            # Unmatched target notes -> deletions (score notes not performed)
            if (
                hasattr(match_result, "unmatched_target")
                and len(match_result.unmatched_target) > 0
            ):
                for i, row in match_result.unmatched_target.iterrows():
                    coord = _safe_float(row.get(score_coord_column, 0))
                    # Try to find an existing SnoteRecord at this coordinate
                    candidates = score_notes.get(coord, [])
                    if candidates:
                        deletions.append(candidates[0])
                    else:
                        pitch_full = str(row.get(score_pitch_column, "C4"))
                        pitch_name, modifier, octave = _parse_pitch(pitch_full)
                        deletions.append(
                            SnoteRecord(
                                id=f"del_{i}",
                                pitch_name=pitch_name,
                                modifier=modifier,
                                octave=octave,
                                measure=1,
                                beat=1,
                                offset="0",
                                duration="1/4",
                                onset_in_beats=coord,
                                offset_in_beats=coord + 1.0,
                                attributes=[],
                            )
                        )

            # Unmatched source notes -> insertions (extra performance notes)
            if (
                hasattr(match_result, "unmatched_source")
                and len(match_result.unmatched_source) > 0
            ):
                for i, row in match_result.unmatched_source.iterrows():
                    coord = _safe_float(row.get(perf_coord_column, 0))
                    candidates = perf_notes.get(coord, [])
                    if candidates:
                        insertions.append(candidates[0])
                    else:
                        pitch_str = str(row.get(perf_pitch_column, "C4"))
                        midi_pitch = _pitch_name_to_midi(pitch_str)
                        onset_tick = int(
                            coord * midi_clock_units * 1_000_000 / midi_clock_rate
                        )
                        insertions.append(
                            NoteRecord(
                                id=f"ins_{i}",
                                midi_pitch=midi_pitch,
                                onset_tick=onset_tick,
                                offset_tick=onset_tick + 480,
                                velocity=64,
                            )
                        )

        return cls(
            piece=piece,
            composer=composer,
            performer=performer,
            score_filename=score_filename,
            midi_filename=midi_filename,
            midi_clock_units=midi_clock_units,
            midi_clock_rate=midi_clock_rate,
            score_notes=score_notes,
            perf_notes=perf_notes,
            deletions=deletions,
            insertions=insertions,
        )

    # endregion


# endregion


# region Formatting Functions


def format_snote_line(snote: SnoteRecord) -> str:
    """Format a single ``snote(...)`` Prolog-style term.

    Args:
        snote: The score note record.

    Returns:
        A string like ``snote(n1,[B,n],3,0:1,0,1/8,-0.5000,0.0000,[v1,staff1])``.
    """
    attrs = ",".join(snote.attributes) if snote.attributes else ""
    return (
        f"snote({snote.id},"
        f"[{snote.pitch_name},{snote.modifier}],"
        f"{snote.octave},"
        f"{snote.measure}:{snote.beat},"
        f"{snote.offset},"
        f"{snote.duration},"
        f"{snote.onset_in_beats:.4f},"
        f"{snote.offset_in_beats:.4f},"
        f"[{attrs}])"
    )


def format_note_line(note: NoteRecord) -> str:
    """Format a single ``note(...)`` Prolog-style term.

    Args:
        note: The performance note record.

    Returns:
        A string like ``note(n0,59,0,261,44,0,0)``.
    """
    return (
        f"note({note.id},"
        f"{note.midi_pitch},"
        f"{note.onset_tick},"
        f"{note.offset_tick},"
        f"{note.velocity},"
        f"{note.channel},"
        f"{note.track})"
    )


def format_match_line(snote: SnoteRecord, note: NoteRecord) -> str:
    """Format a matched pair as ``snote(...)-note(...).``

    Args:
        snote: The score note record.
        note: The performance note record.

    Returns:
        A complete match line ending with a period.
    """
    return f"{format_snote_line(snote)}-{format_note_line(note)}."


def format_deletion_line(snote: SnoteRecord) -> str:
    """Format an unperformed score note as ``snote(...)-deletion.``

    Args:
        snote: The score note record.

    Returns:
        A deletion line ending with a period.
    """
    return f"{format_snote_line(snote)}-deletion."


def format_insertion_line(note: NoteRecord) -> str:
    """Format an extra performance note as ``insertion-note(...).``

    Args:
        note: The performance note record.

    Returns:
        An insertion line ending with a period.
    """
    return f"insertion-{format_note_line(note)}."


def format_header(ctx: MatchFileContext) -> list[str]:
    """Format all ``info(...)`` header lines from a context.

    The order follows the Vienna convention:
    matchFileVersion, piece, scoreFileName, midiFileName,
    composer, performer, midiClockUnits, midiClockRate.

    Args:
        ctx: The export context.

    Returns:
        List of formatted header lines (each ending with ``"."``).
    """
    lines = [
        "info(matchFileVersion,1.0.0).",
    ]
    if ctx.piece:
        lines.append(f"info(piece,{ctx.piece}).")
    if ctx.score_filename:
        lines.append(f"info(scoreFileName,{ctx.score_filename}).")
    if ctx.midi_filename:
        lines.append(f"info(midiFileName,{ctx.midi_filename}).")
    if ctx.composer:
        lines.append(f"info(composer,{ctx.composer}).")
    if ctx.performer:
        lines.append(f"info(performer,{ctx.performer}).")
    lines.append(f"info(midiClockUnits,{ctx.midi_clock_units}).")
    lines.append(f"info(midiClockRate,{ctx.midi_clock_rate}).")
    return lines


def format_score_properties(ctx: MatchFileContext) -> list[str]:
    """Format ``scoreprop(...)`` lines from a context.

    If the context has pre-formatted ``score_properties``, they are
    returned directly.  Otherwise an empty list is returned (score
    properties are optional in the format).

    Args:
        ctx: The export context.

    Returns:
        List of formatted score-property lines.
    """
    return list(ctx.score_properties)


class _RawStampExportView:
    """Present a stamp's coordinates as the floats the ``.match`` writer reads."""

    def __init__(self, stamp: MatchStamp) -> None:
        self._stamp = stamp

    def _get_float_coordinate_for(self, timeline_id: str) -> float:
        """Return the stamp's coordinate on *timeline_id* as a float."""
        return self._stamp.get_coordinate_for(timeline_id, format="float")


class _RawMatchLineExportView:
    """Present a MatchLine to the ``.match`` writer, its stamps as float views."""

    def __init__(self, match_line: MatchLine) -> None:
        self.source_timeline_id = match_line.source_timeline_id
        self.stamps = [_RawStampExportView(stamp) for stamp in match_line.stamps]
        self._target_timeline_ids = match_line.target_timeline_ids()

    def target_timeline_ids(self) -> set[str]:
        """Return target timeline IDs from the source MatchLine."""
        return self._target_timeline_ids


def write_match_file(
    filepath: str | Path,
    match_line: "MatchLine",
    context: MatchFileContext | None = None,
) -> Path:
    """Write a ``.match`` file from a MatchLine and optional context.

    This is the main orchestrator for ``.match`` export.  It assembles
    header, score properties, match/deletion/insertion lines, and writes
    them to disk.

    When *context* is ``None``, a minimal file is produced with
    placeholder note data derived solely from the MatchStamp coordinates.

    Args:
        filepath: Output file path.
        match_line: The MatchLine to export.
        context: Optional :class:`MatchFileContext` with rich note data.

    Returns:
        The resolved output :class:`~pathlib.Path`.
    """
    filepath = Path(filepath)
    view = _RawMatchLineExportView(match_line)

    if context is None:
        context = _build_minimal_context(view)

    lines: list[str] = []

    # Header
    lines.extend(format_header(context))

    # Score properties — partitura requires at least a time signature
    # to parse .match files, so add a default 4/4 if none is provided.
    props = format_score_properties(context)
    if props:
        lines.extend(props)
    else:
        lines.append("scoreprop(timeSignature,4/4,1:1,0,0.0000).")

    # Match lines (from stamps)
    source_id = view.source_timeline_id
    target_ids = view.target_timeline_ids()

    if not target_ids:
        module_logger.warning(
            "MatchLine has no target timelines; "
            "output will contain only header and deletion lines."
        )
        target_id = None
    else:
        # Pick the first (and usually only) target timeline
        target_id = sorted(target_ids)[0]

    for stamp in view.stamps:
        source_coord = stamp._get_float_coordinate_for(source_id)
        if source_coord is None:
            continue

        target_coord = stamp._get_float_coordinate_for(target_id) if target_id else None

        snote = _lookup_snote(context, source_coord)
        if target_coord is not None:
            note = _lookup_note(context, target_coord)
            lines.append(format_match_line(snote, note))
        else:
            # No target coordinate — treat as deletion
            lines.append(format_deletion_line(snote))

    # Deletion lines
    for snote in context.deletions:
        lines.append(format_deletion_line(snote))

    # Insertion lines
    for note in context.insertions:
        lines.append(format_insertion_line(note))

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    module_logger.info("Wrote %d lines to %s", len(lines), filepath)
    return filepath


# endregion


# region Internal Helpers


def _lookup_snote(
    context: MatchFileContext,
    coord: float,
    midi_pitch: int | None = None,
) -> SnoteRecord:
    """Find a SnoteRecord in the context for the given coordinate.

    Falls back to a placeholder record when no match is found.

    Args:
        context: The export context.
        coord: The score coordinate to look up.
        midi_pitch: Optional MIDI pitch for disambiguation in chords.

    Returns:
        The matching :class:`SnoteRecord`, or a placeholder.
    """
    candidates = context.score_notes.get(coord)
    if candidates:
        if midi_pitch is not None and len(candidates) > 1:
            for c in candidates:
                if _pitch_name_to_midi(f"{c.pitch_name}{c.octave}") == midi_pitch:
                    return c
        return candidates[0]

    # Try nearest-coordinate fallback (for floating-point tolerance)
    for stored_coord, records in context.score_notes.items():
        if abs(stored_coord - coord) < 1e-6:
            return records[0]

    return _placeholder_snote(coord)


def _lookup_note(
    context: MatchFileContext,
    coord: float,
    midi_pitch: int | None = None,
) -> NoteRecord:
    """Find a NoteRecord in the context for the given coordinate.

    Falls back to a placeholder record when no match is found.

    Args:
        context: The export context.
        coord: The performance coordinate to look up.
        midi_pitch: Optional MIDI pitch for disambiguation.

    Returns:
        The matching :class:`NoteRecord`, or a placeholder.
    """
    candidates = context.perf_notes.get(coord)
    if candidates:
        if midi_pitch is not None and len(candidates) > 1:
            for c in candidates:
                if c.midi_pitch == midi_pitch:
                    return c
        return candidates[0]

    # Nearest-coordinate fallback
    for stored_coord, records in context.perf_notes.items():
        if abs(stored_coord - coord) < 1e-6:
            return records[0]

    return _placeholder_note(coord)


def _placeholder_snote(coord: float) -> SnoteRecord:
    """Create a placeholder SnoteRecord for coordinate-only export."""
    return SnoteRecord(
        id=f"s{int(coord * 100)}",
        pitch_name="C",
        modifier="n",
        octave=4,
        measure=1,
        beat=1,
        offset="0",
        duration="1/4",
        onset_in_beats=coord,
        offset_in_beats=coord + 1.0,
        attributes=[],
    )


def _placeholder_note(coord: float) -> NoteRecord:
    """Create a placeholder NoteRecord for coordinate-only export."""
    return NoteRecord(
        id=f"p{int(coord * 100)}",
        midi_pitch=60,
        onset_tick=int(coord),
        offset_tick=int(coord) + 480,
        velocity=64,
    )


def _build_minimal_context(view: _RawMatchLineExportView) -> MatchFileContext:
    """Build a minimal MatchFileContext from coordinate data only.

    Creates placeholder SnoteRecords and NoteRecords from the stamps,
    with no real note-level data.

    Args:
        view: The MatchLine to derive context from, as the writer reads it.

    Returns:
        A :class:`MatchFileContext` with placeholder data.
    """
    source_id = view.source_timeline_id
    target_ids = view.target_timeline_ids()
    target_id = sorted(target_ids)[0] if target_ids else None

    score_notes: dict[float, list[SnoteRecord]] = {}
    perf_notes: dict[float, list[NoteRecord]] = {}

    for stamp in view.stamps:
        source_coord = stamp._get_float_coordinate_for(source_id)
        if source_coord is not None and source_coord not in score_notes:
            score_notes[source_coord] = [_placeholder_snote(source_coord)]

        if target_id is not None:
            target_coord = stamp._get_float_coordinate_for(target_id)
            if target_coord is not None and target_coord not in perf_notes:
                perf_notes[target_coord] = [_placeholder_note(target_coord)]

    return MatchFileContext(
        score_notes=score_notes,
        perf_notes=perf_notes,
    )


def _parse_pitch(pitch_str: str) -> tuple[str, str, int]:
    """Parse a pitch string into (name, modifier, octave).

    Handles formats like ``"B3"``, ``"F#4"``, ``"Bb2"``, ``"C♯5"``.

    Args:
        pitch_str: Pitch string to parse.

    Returns:
        Tuple of ``(pitch_name, modifier, octave)``.
    """
    if not pitch_str or pitch_str in ("rest", "Rest", "R"):
        return ("C", "n", 4)

    # Extract note name (first letter)
    name = pitch_str[0].upper()
    rest = pitch_str[1:]

    # Extract modifier and octave
    modifier = "n"
    octave = 4

    if not rest:
        return (name, modifier, octave)

    # Check for accidentals
    if rest.startswith("##") or rest.startswith("x"):
        modifier = "x"
        rest = rest[2:] if rest.startswith("##") else rest[1:]
    elif rest.startswith("#") or rest.startswith("♯"):
        modifier = "#"
        rest = rest[1:]
    elif rest.startswith("bb") or rest.startswith("♭♭"):
        modifier = "bb"
        rest = rest[2:] if rest.startswith("bb") else rest[2:]
    elif rest.startswith("b") or rest.startswith("♭"):
        # Be careful: "b" could be flat or part of octave for note "B"
        # If the next char is a digit, "b" is likely a flat
        if rest.startswith("♭"):
            modifier = "b"
            rest = rest[1:]
        elif len(rest) > 1 and rest[1:].lstrip("-").isdigit():
            modifier = "b"
            rest = rest[1:]
        elif rest == "b":
            # Ambiguous: just "b" at end — assume flat
            modifier = "b"
            rest = ""

    # Extract octave from remaining
    if rest:
        try:
            octave = int(rest)
        except ValueError:
            pass

    return (name, modifier, octave)


# Denominators a Match file writes. These are note values -- halves down to
# sixty-fourths, plus their triplet and sextuplet subdivisions -- so this is
# the format's own vocabulary, not a search for whatever ratio happens to sit
# nearest a number.
_MATCH_DENOMINATORS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def _to_fraction_str(value: Any) -> str:
    """Render a numeric value as a Match-file fraction string.

    An exact input is written exactly -- a triplet eighth arrives as
    ``Fraction(1, 3)`` and leaves as ``"1/3"``.

    A float has no exact note value to write, so it is snapped to the
    nearest denominator the format admits. That is a deliberate,
    format-imposed quantization with a fixed vocabulary, and it is confined
    to this writer: nothing that stays inside the library ever gains a
    denominator it did not earn.

    Args:
        value: A float, int, string, or Fraction.

    Returns:
        A string like ``"0"``, ``"1/8"``, ``"3/16"``.
    """
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, str):
        if "/" in value:
            return value
        try:
            exact = Fraction(value)
        except (ValueError, ZeroDivisionError):
            return value
        return str(exact) if exact != 0 else "0"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    try:
        fval = float(value)
    except (TypeError, ValueError):
        return str(value)
    if fval == 0:
        return "0"
    return str(_snap_to_match_denominator(fval))


def _snap_to_match_denominator(value: float) -> Fraction:
    """Return the closest Match-writable note value to *value*."""
    best = min(
        (Fraction(round(value * d), d) for d in _MATCH_DENOMINATORS),
        key=lambda candidate: abs(float(candidate) - value),
    )
    return best


def _safe_float(value: Any) -> float:
    """Convert a value to float, handling fraction strings.

    Args:
        value: Numeric value, fraction string, or NaN.

    Returns:
        Float value, defaulting to 0.0 on failure.
    """
    if value is None:
        return 0.0
    try:
        if isinstance(value, str):
            return float(Fraction(value))
        result = float(value)
        # Handle NaN
        if result != result:  # noqa: PLR0124 — NaN check
            return 0.0
        return result
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


_PITCH_TO_MIDI_BASE = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

_MODIFIER_SEMITONES = {
    "n": 0,
    "#": 1,
    "b": -1,
    "bb": -2,
    "x": 2,
    "": 0,
}


def _pitch_name_to_midi(pitch_str: str) -> int:
    """Convert a pitch name like ``"B3"`` to a MIDI note number.

    Args:
        pitch_str: Pitch string (e.g. ``"C4"``, ``"F#3"``, ``"Bb2"``).

    Returns:
        MIDI pitch number (0–127).
    """
    name, modifier, octave = _parse_pitch(pitch_str)
    base = _PITCH_TO_MIDI_BASE.get(name, 0)
    mod = _MODIFIER_SEMITONES.get(modifier, 0)
    return (octave + 1) * 12 + base + mod


# endregion
