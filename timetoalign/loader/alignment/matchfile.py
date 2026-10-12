"""MatchfileLoader: Loader for Vienna Match (.match) alignment files.

This module implements the ``MatchfileLoader`` class, which uses partitura
to parse ``.match`` files and produces:

- A shared ``ContinuousLogicalTimeline`` for the score side (quarter-beat
  coordinates).
- One ``DiscreteLogicalTimeline`` per performance (MIDI tick coordinates).
- ``MatchClaim`` objects linking score events to performance events.

The loader follows the standard two-phase pattern:

1. ``loader.load(*match_files)`` — parses files, builds internal state.
2. ``loader.create_bundle()`` — assembles an ``AlignmentBundle``.
3. ``loader.create_timeline(uid)`` — retrieves individual timelines.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.core.backends import suppressed_backend_warnings

with suppressed_backend_warnings():
    import partitura as pt

from timetoalign.alignment.claims import Agent, MatchClaim, MatchMetadata
from timetoalign.core import AgentType, TimeUnit, resolve_id
from timetoalign.loader.base import AlignmentLoader
from timetoalign.maps.linear import ScalarMap, ShiftMap
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    DiscreteLogicalTimeline,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Constants

MATCH_META = MatchMetadata(
    agent=Agent(
        name="vienna_match",
        type=AgentType.software,
        identifier="v1.0.0",
    ),
    certainty=1.0,
)

# endregion


# region MatchfileLoader


def _with_timeline_id(claim: MatchClaim, old_id: str, new_id: str) -> MatchClaim:
    """Return *claim* naming *new_id* wherever it names *old_id*.

    Only the timeline identities change; coordinates and events are kept.
    """
    if old_id == new_id or old_id not in claim.timelines:
        return claim

    def renamed(timeline_id: str) -> str:
        return new_id if timeline_id == old_id else timeline_id

    updates: dict[str, Any] = {
        "timeline_a_id": renamed(claim.timeline_a_id),
        "timeline_b_id": renamed(claim.timeline_b_id),
    }
    for name in ("start_anchor", "end_anchor"):
        anchor = getattr(claim, name)
        if anchor is not None:
            updates[name] = anchor.model_copy(
                update={
                    "timeline_a_id": renamed(anchor.timeline_a_id),
                    "timeline_b_id": renamed(anchor.timeline_b_id),
                }
            )
    return claim.model_copy(update=updates)


class MatchfileLoader(AlignmentLoader):
    """Load Vienna Match (.match) alignment files via partitura.

    A single ``MatchfileLoader`` instance is intended to process **all**
    ``.match`` files that share the same score (e.g. 22 performances of one
    piece). It builds a shared score timeline on the first compatible file
    and verifies subsequent files against it. Incompatible files (snote ID
    present with different coordinates) are rejected with a warning.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(*match_files)`` — parses all files, builds internal state.
    2. ``loader.create_bundle()`` — assembles the result.
    3. ``loader.create_timeline(uid)`` — retrieves individual timelines.

    Produces:

    - A ``ContinuousLogicalTimeline`` for the score side (quarter-beat
      coordinates; MIDI divisions available via the attached
      ``quarters_to_divs`` `ScalarMap` C-Map).
    - One ``DiscreteLogicalTimeline`` per performance (MIDI tick coordinates;
      seconds available via the attached ``ticks_to_seconds`` `ScalarMap`).
    - One `MatchClaim` per alignment record per file:
      - Matched notes produce ``MatchClaim.from_events()`` (synchronous
        interval claim).
      - Deletion records produce ``MatchClaim.nomatch()`` (non-synchronous,
        NOMATCH).

    All MatchClaims carry ``MatchMetadata`` whose agent is
    ``Agent(name="vienna_match", type=AgentType.software,
    identifier="v1.0.0")`` and ``certainty=1.0``.

    **Coordinate system and normalisation:**
    Score coordinates are stored as **normalised TTA coordinates** (shifted
    so the minimum onset is 0.0). A ``ShiftMap`` named ``"raw_quarters"``
    is attached to the score timeline to convert normalised coordinates
    back to raw partitura values. The shift amount is computed from the
    file itself and never hardcoded.

    **External score timeline:**
    Pass a score timeline previously built by ``PartituraLoader`` via the
    ``score_timeline=`` parameter of ``create_bundle()``. The
    loader verifies compatibility (matching snote IDs and coordinates) and
    references the supplied timeline in all MatchClaims. MatchClaims are
    never rebound after creation.

    See Also:
        timetoalign.MatchClaim
        timetoalign.AlignmentBundle

    Args:
        score_unit: Primary unit for the score timeline.
            Default ``TimeUnit.quarters``.
        normalize_anacrusis: If True (default), attach a ``ShiftMap`` that
            maps raw coordinates to normalised (non-negative) coordinates.
            Has no effect on the stored event coordinates, which are always
            the normalised (non-negative) values.
    """

    def __init__(
        self,
        score_unit: TimeUnit = TimeUnit.quarters,
        normalize_anacrusis: bool = True,
    ) -> None:
        super().__init__()
        self._score_unit = score_unit
        self._normalize_anacrusis = normalize_anacrusis
        self._logger = module_logger.getChild("MatchfileLoader")

        # Internal state accumulated across load() calls
        self._score_timeline: ContinuousLogicalTimeline | None = None
        self._performance_timelines: list[DiscreteLogicalTimeline] = []
        self._claims: list[MatchClaim] = []
        self._rejected_files: list[Path] = []
        self._sources: list[Path] = []

        # Score event tracking: snote_id -> (start, end) in TTA coords
        self._score_events: dict[str, tuple[Fraction, Fraction]] = {}

        # Anacrusis offset (computed from first file)
        self._anacrusis_offset: float = 0.0
        self._anacrusis_offset_exact = Fraction(0)

        # Header info from first file
        self._midi_clock_units: int | None = None
        self._midi_clock_rate: int | None = None
        self._piece_name: str | None = None

    # region Properties

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: None,
    ) -> None:
        """Record a parsed match file whose payload is held in loader state."""
        self._sources.append(path)
        self._source_metadata.append(source_meta)

    @property
    def anacrusis_offset(self) -> float:
        """The anacrusis offset applied to raw score coordinates.

        Equals ``-min(raw_onsets)`` from the first loaded file. Zero if
        no anacrusis or if no files have been loaded.
        """
        return self._anacrusis_offset

    @property
    def rejected_files(self) -> list[Path]:
        """Files rejected due to incompatible score coordinates."""
        return list(self._rejected_files)

    @property
    def sources(self) -> list[Path]:
        """All files passed to ``load()`` (including rejected ones)."""
        return list(self._sources)

    # endregion

    # region Loading

    def load(self, *sources: Path | str) -> Self:
        """Load one or more .match files.

        Parses each file via partitura, builds a shared score timeline
        from the first compatible file, and verifies subsequent files
        against it. Incompatible files are logged with a warning and
        tracked internally.

        This method can be called multiple times to add more files
        incrementally.

        Args:
            *sources: Paths to ``.match`` files.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If any source path does not exist.
            ValueError: If any source is not a ``.match`` file.

        Examples:
            >>> loader = MatchfileLoader()
            >>> loader.load("Chopin_op10_no3_p01.match")
            >>> # or load all at once
            >>> loader.load(*sorted(data_dir.glob("*.match")))
        """
        for source in sources:
            path = Path(source)

            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            if path.suffix.lower() != ".match":
                raise ValueError(f"Not a .match file: {path}")

            super().load(path)

        return self

    @staticmethod
    def _to_tta_coord(raw: float, offset: float) -> float:
        """Convert a raw partitura coordinate to a TTA coordinate.

        TTA coordinates are normalised so that the minimum score onset
        is 0.0. The *offset* is the negation of the minimum raw onset
        from the first loaded file.

        Args:
            raw: Raw partitura onset value (may be negative for anacrusis).
            offset: Anacrusis offset (``-min(raw_onsets)``; always >= 0).

        Returns:
            Normalised coordinate (always >= 0).
        """
        return raw + offset

    @staticmethod
    def _read_exact_score_coordinates(
        path: Path,
    ) -> dict[str, tuple[Fraction, Fraction]]:
        """Read exact score onsets and offsets from matchfile text.

        Partitura exposes the score coordinates in its note array as
        ``float32`` values. The matchfile itself stores those beat values as
        decimal text, so parsing the source fields directly retains their
        exact decimal fractions.

        Args:
            path: Matchfile to read.

        Returns:
            Mapping from score-note ID to ``(onset, offset)`` fractions.

        Raises:
            ValueError: If a score-note line cannot be parsed.
        """
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        pattern = re.compile(
            rf"^snote\((?P<id>[^,]+),.*,(?P<onset>{number}),"
            rf"(?P<offset>{number}),\["
        )
        coordinates: dict[str, tuple[Fraction, Fraction]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.startswith("snote("):
                    continue
                match = pattern.match(line.strip())
                if match is None:
                    raise ValueError(f"Could not parse score coordinates in {path}")
                score_id = match.group("id")
                coordinates[score_id] = (
                    Fraction(match.group("onset")),
                    Fraction(match.group("offset")),
                )
        return coordinates

    @staticmethod
    def _to_tta_fraction(raw: Fraction, offset: Fraction) -> Fraction:
        """Convert an exact raw score coordinate to TTA coordinates."""
        return raw + offset

    def _check_or_add_score_event(
        self,
        snote_id: str,
        onset_tta: float,
        end_tta: float,
        source_file: str,
    ) -> bool:
        """Verify or register a score event on the shared timeline.

        Algorithm:
            1. Look up the event by ``snote_id`` in the local cache.
            2a. Found AND start coordinate matches -> compatible.
            2b. Found AND start coordinate does NOT match -> incompatible.
            2c. Not found -> add the event to the timeline and cache.

        Args:
            snote_id: Score note identifier (e.g. ``"n1"``).
            onset_tta: Normalised start coordinate.
            end_tta: Normalised end coordinate.
            source_file: Filename for diagnostic messages.

        Returns:
            ``True`` if compatible (event matched or was added);
            ``False`` if incompatible (coordinate mismatch).
        """
        if snote_id in self._score_events:
            stored_start, _stored_end = self._score_events[snote_id]
            if abs(stored_start - onset_tta) > 1e-10:
                self._logger.warning(
                    "File '%s': snote '%s' has onset %s but timeline "
                    "stores %s — file rejected.",
                    source_file,
                    snote_id,
                    onset_tta,
                    stored_start,
                )
                return False
            return True

        # New event — add to timeline and cache
        self._score_events[snote_id] = (onset_tta, end_tta)
        if self._score_timeline is not None:
            self._score_timeline.add_events(
                [{"id": snote_id, "start": onset_tta, "end": end_tta}],
                allow_expansion=True,
            )
        return True

    def _load_source(self, path: Path) -> None:
        """Parse a single .match file and accumulate internal state.

        On the first compatible file: builds the shared score timeline
        and attaches C-Maps. On subsequent files: verifies score events
        against the shared timeline and rejects the file on first
        incompatibility.

        Args:
            path: Path to the .match file.
        """
        self._logger.debug(f"Loading {path.name}")

        # Parse with partitura
        with suppressed_backend_warnings():
            perf, alignment, score = pt.load_match(str(path), create_score=True)

        # Extract header info from file
        header = self._parse_header(path)
        midi_clock_units = header.get("midiClockUnits", 480)
        midi_clock_rate = header.get("midiClockRate", 500000)
        piece_name = header.get("piece", path.stem.rsplit("_", 1)[0])
        midi_filename = header.get("midiFileName", path.stem + ".mid")
        perf_stem = Path(midi_filename).stem

        # Store header info from first file
        if self._midi_clock_units is None:
            self._midi_clock_units = midi_clock_units
            self._midi_clock_rate = midi_clock_rate
            self._piece_name = piece_name

        # Build note array for score metadata and score IDs. The exact beat
        # coordinates are read from the matchfile source text below because
        # partitura's note array materialises them as float32 values.
        with suppressed_backend_warnings():
            score_na = pt.compute_note_array(score)
        score_by_id = {score_na[i]["id"]: score_na[i] for i in range(len(score_na))}
        exact_score_coords = self._read_exact_score_coordinates(path)
        if set(score_by_id) != set(exact_score_coords):
            raise ValueError(
                f"Score IDs from partitura and source differ in {path}: "
                f"{set(score_by_id) ^ set(exact_score_coords)}"
            )

        # Build lookup for performance notes
        pp = perf[0]  # First (and only) PerformedPart
        perf_notes = pp.notes
        perf_by_id = {n["id"]: n for n in perf_notes}

        # Build alignment lookups
        matches = [a for a in alignment if a["label"] == "match"]
        deletions = [a for a in alignment if a["label"] == "deletion"]

        # Compute anacrusis offset from raw score onsets (first file only)
        is_first_file = self._score_timeline is None
        if is_first_file:
            raw_onsets = [exact_score_coords[sid][0] for sid in score_by_id]
            min_onset = min(raw_onsets)
            self._anacrusis_offset_exact = -min_onset if min_onset < 0 else Fraction(0)
            self._anacrusis_offset = float(self._anacrusis_offset_exact)

        offset = self._anacrusis_offset_exact

        # Step 1: Verify or add score events to the shared timeline
        if is_first_file:
            # Compute timeline length from max offset + duration
            raw_ends = [exact_score_coords[sid][1] for sid in score_by_id]
            max_end = max(raw_ends)
            timeline_length = max_end + offset

            self._score_timeline = ContinuousLogicalTimeline(
                length=timeline_length,
                unit=self._score_unit,
                uid="score:clt_1",
                name=piece_name,
            )

            # Add all score events to the timeline
            score_event_dicts = []
            for sid in score_by_id:
                onset_raw, end_raw = exact_score_coords[sid]
                onset_tta = self._to_tta_fraction(onset_raw, offset)
                end_tta = self._to_tta_fraction(end_raw, offset)
                score_event_dicts.append(
                    {
                        "id": sid,
                        "start": onset_tta,
                        "duration": end_tta - onset_tta,
                        "end": end_tta,
                    }
                )
                self._score_events[sid] = (onset_tta, end_tta)

            self._score_timeline.add_events(score_event_dicts)

            # Attach ShiftMap to convert normalised TTA coordinates back
            # to raw partitura coordinates (for display in timestamps).
            # Since normalised = raw + offset, raw = normalised - offset.
            if self._normalize_anacrusis and offset != 0.0:
                shift_map = ShiftMap(
                    offset=-offset,
                    source_unit=self._score_unit,
                    target_unit=self._score_unit,
                    uid="raw_quarters",
                    name="raw_quarters",
                )
                self._score_timeline.add_conversion_map(shift_map)

            # Attach ScalarMap for quarters_to_divs
            divs_map = ScalarMap(
                scalar=midi_clock_units,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.ticks,
                uid="quarters_to_divs",
                name="quarters_to_divs",
            )
            self._score_timeline.add_conversion_map(divs_map)

            self._logger.debug(
                f"Built score timeline '{self._score_timeline.id}' with "
                f"{len(score_event_dicts)} events, anacrusis_offset={offset}"
            )

        else:
            # Subsequent file: verify compatibility via _check_or_add_score_event
            for sid in score_by_id:
                onset_raw, end_raw = exact_score_coords[sid]
                onset_tta = self._to_tta_fraction(onset_raw, offset)
                end_tta = self._to_tta_fraction(end_raw, offset)

                if not self._check_or_add_score_event(
                    sid, onset_tta, end_tta, path.name
                ):
                    self._rejected_files.append(path)
                    self._logger.warning(
                        "File '%s' rejected due to incompatible " "score coordinates.",
                        path.name,
                    )
                    return

        # Step 2: Build performance timeline
        perf_event_dicts = []
        for note in perf_notes:
            perf_event_dicts.append(
                {
                    "id": note["id"],
                    "start": float(note["note_on_tick"]),
                    "end": float(note["note_off_tick"]),
                }
            )

        # Compute performance timeline length
        if perf_event_dicts:
            perf_max = max(float(e["end"]) for e in perf_event_dicts)
        else:
            perf_max = 0.0

        perf_tl = DiscreteLogicalTimeline(
            length=perf_max,
            unit=TimeUnit.ticks,
            uid=f"perf:{perf_stem}:dlt_1",
            name=perf_stem,
        )
        perf_tl.add_events(perf_event_dicts)

        # Attach ticks_to_seconds ScalarMap
        seconds_per_tick = midi_clock_rate / (midi_clock_units * 1_000_000)
        ticks_to_secs = ScalarMap(
            scalar=seconds_per_tick,
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
            uid=f"ticks_to_seconds:{perf_stem}",
            name="ticks_to_seconds",
        )
        perf_tl.add_conversion_map(ticks_to_secs)

        self._performance_timelines.append(perf_tl)

        self._logger.debug(
            f"Built performance timeline '{perf_tl.id}' with "
            f"{len(perf_event_dicts)} events"
        )

        # Step 3: Build MatchClaims
        score_tl_id = self._score_timeline.id
        perf_tl_id = perf_tl.id

        # Matched notes: synchronous interval claims
        for match_record in matches:
            score_id = match_record["score_id"]
            perf_id = match_record["performance_id"]

            if score_id not in self._score_events:
                self._logger.warning(
                    "Match references unknown score note '%s' in file '%s'",
                    score_id,
                    path.name,
                )
                continue

            if perf_id not in perf_by_id:
                self._logger.warning(
                    "Match references unknown performance note '%s' in file '%s'",
                    perf_id,
                    path.name,
                )
                continue

            score_start, score_end = self._score_events[score_id]
            perf_note = perf_by_id[perf_id]
            perf_start = float(perf_note["note_on_tick"])
            perf_end = float(perf_note["note_off_tick"])

            claim = MatchClaim.from_events(
                event_a={"start": score_start, "end": score_end},
                tl_a_id=score_tl_id,
                event_b={"start": perf_start, "end": perf_end},
                tl_b_id=perf_tl_id,
                unit_a=self._score_unit,
                unit_b=TimeUnit.ticks,
                coord_key="start",
                end_coord_key="end",
                metadata=MATCH_META,
            )
            self._claims.append(claim)

        # Deletion records: non-synchronous NOMATCH claims
        for del_record in deletions:
            score_id = del_record["score_id"]

            if score_id not in self._score_events:
                self._logger.warning(
                    "Deletion references unknown score note '%s' in file '%s'",
                    score_id,
                    path.name,
                )
                continue

            score_start, _ = self._score_events[score_id]
            claim = MatchClaim.nomatch(
                event={"start": score_start},
                source_tl_id=score_tl_id,
                target_tl_id=perf_tl_id,
                unit=self._score_unit,
                metadata=MATCH_META,
            )
            self._claims.append(claim)

        self._logger.debug(
            f"Created {len(matches)} match claims and {len(deletions)} "
            f"NOMATCH claims for '{path.name}'"
        )

    # endregion

    # region Header Parsing

    @staticmethod
    def _parse_header(path: Path) -> dict[str, Any]:
        """Parse ``info(...)`` header lines from a .match file.

        Args:
            path: Path to the .match file.

        Returns:
            Dictionary mapping header keys to values.
        """
        header: dict[str, Any] = {}
        info_pattern = re.compile(r"^info\((\w+),(.+)\)\.\s*$")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("info("):
                    # Headers are at the top; stop at first non-info line
                    if not line.startswith("%"):
                        break
                    continue

                m = info_pattern.match(line)
                if m:
                    key, value = m.group(1), m.group(2)
                    # Try to convert to int
                    try:
                        header[key] = int(value)
                    except ValueError:
                        header[key] = value

        return header

    # endregion

    # region Domain Object Creation

    def _verify_external_score(
        self,
        external: "Timeline",
    ) -> list[str]:
        """Check that internally built score events are compatible with an external timeline.

        Iterates over every score event ID in the local cache and looks
        it up on *external* via `Timeline.get_event`. An event is
        **incompatible** if:

        - It exists on the external timeline but its start coordinate
          differs by more than 1e-10 from the internally stored value.

        Events that are present in the internal cache but absent from
        the external timeline are **not** treated as incompatible — the
        external timeline may simply not contain that event (e.g. grace
        notes omitted by a different parser).

        Args:
            external: A score timeline produced by another loader (e.g.
                ``PartituraLoader``).

        Returns:
            A list of human-readable mismatch descriptions. Empty if
            the timelines are fully compatible.
        """
        mismatches: list[str] = []
        for snote_id, (internal_start, _internal_end) in self._score_events.items():
            ext_event = external.get_event(snote_id)
            if ext_event is None:
                # Event not present on external timeline — tolerated
                continue
            ext_start = ext_event.get("start")
            if ext_start is not None:
                if isinstance(ext_start, dict):
                    ext_val = float(ext_start["value"])
                else:
                    ext_val = float(ext_start)
                if abs(ext_val - internal_start) > 1e-10:
                    mismatches.append(
                        f"snote '{snote_id}': internal={internal_start}, "
                        f"external={ext_val}"
                    )
        return mismatches

    def create_bundle(
        self,
        score_timeline: "Timeline | None" = None,
        verify: bool = True,
    ) -> "AlignmentBundle":
        """Assemble an AlignmentBundle from the loaded data.

        Returns an ``AlignmentBundle`` directly. The bundle contains:

        - The score timeline in its own group (``as_group="score"``)
        - Each performance timeline as a standalone timeline
        - All MatchClaims as cross-group claims

        This method takes NO file arguments. Files must be loaded first
        via ``load()``.

        When an external *score_timeline* is supplied and *verify* is
        ``True`` (the default), the loader cross-checks every internally
        built score event against the external timeline using
        `Timeline.get_event`. A ``ValueError`` is raised if any event
        coordinates are incompatible.

        Args:
            score_timeline: Optional pre-existing score timeline. When
                supplied, the bundle uses this timeline instead of the
                loader's internally built one, whatever its id: the
                MatchClaims name the id of the score timeline the bundle
                holds. The loader's own claim list is left unchanged.
            verify: If ``True`` (default) and *score_timeline* is not
                ``None``, verify that internally built events are
                compatible with the external timeline. Set to ``False``
                to skip verification (e.g. when the external timeline
                is known to be authoritative and you accept any
                coordinate differences).

        Returns:
            ``AlignmentBundle`` with score group + standalone performances
            + MatchClaims.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            ValueError: If *verify* is ``True`` and the external
                timeline has incompatible event coordinates.

        Examples:
            >>> loader = MatchfileLoader()
            >>> loader.load(*match_files)
            >>> bundle = loader.create_bundle()
            >>> len(bundle.timelines)  # score + 22 performances
            23
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if self._score_timeline is None:
            raise RuntimeError(
                "No files loaded yet. Call load() before " "create_bundle()."
            )

        actual_score = (
            score_timeline if score_timeline is not None else self._score_timeline
        )

        # Verify external timeline compatibility
        if score_timeline is not None and verify:
            mismatches = self._verify_external_score(score_timeline)
            if mismatches:
                detail = "; ".join(mismatches[:5])
                suffix = (
                    f" (and {len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
                )
                raise ValueError(
                    f"External score timeline has {len(mismatches)} "
                    f"incompatible event(s): {detail}{suffix}"
                )

        bundle = AlignmentBundle()
        bundle.add_timeline(actual_score, as_group="score")

        for perf_tl in self._performance_timelines:
            bundle.add_timeline(perf_tl)

        # The claims were built against the loader's own score timeline; they
        # name whichever score timeline the bundle actually holds.
        bundle.add_match_claims(
            [
                _with_timeline_id(claim, self._score_timeline.id, actual_score.id)
                for claim in self._claims
            ]
        )

        return bundle

    def create_timeline(self, uid: str) -> "Timeline":
        """Return a single timeline by uid, role string, or partial/regex match.

        Matching precedence:
        1. ``"score"`` — returns the shared score timeline.
        2. Exact match on score timeline uid.
        3. Exact match on performance timeline uid.
        4. ``"perf:N"`` (1-indexed) — returns the N-th performance timeline.
        5. ``"perf:pNN"`` — alternative syntax for performance indexing.
        6. Partial/regex match on any timeline uid.

        Args:
            uid: Timeline uid (e.g. ``"score:clt_1"``), role
                shorthand (``"score"``, ``"perf:1"``), or partial/regex
                pattern.

        Returns:
            The matching Timeline.

        Raises:
            KeyError: If no timeline with the given uid or role exists.
            RuntimeError: If ``load()`` has not been called yet.

        Examples:
            >>> loader.create_timeline("score")               # -> score:clt_1
            >>> loader.create_timeline("perf:1")              # First performance
            >>> loader.create_timeline("perf:p01")            # Same shorthand
            >>> loader.create_timeline(r"^perf:.*p01:dlt_1$")  # Stored-ID regex
        """
        if self._score_timeline is None:
            raise RuntimeError(
                "No files loaded yet. Call load() before create_timeline()."
            )

        # Direct match: "score" role
        if uid == "score":
            return self._score_timeline

        # Direct match: score timeline uid
        if uid == self._score_timeline.id:
            return self._score_timeline

        # Check performance timelines by uid
        for perf_tl in self._performance_timelines:
            if uid == perf_tl.id:
                return perf_tl

        # Check "perf:N" shorthand (1-indexed)
        if uid.startswith("perf:"):
            suffix = uid[5:]
            try:
                idx = int(suffix) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(self._performance_timelines):
                    return self._performance_timelines[idx]
            except ValueError:
                pass

        # Check "perf:pNN" shorthand
        if uid.startswith("perf:p"):
            suffix = uid[6:]
            try:
                idx = int(suffix) - 1  # 1-indexed to 0-indexed
                if 0 <= idx < len(self._performance_timelines):
                    return self._performance_timelines[idx]
            except ValueError:
                pass

        # Fallback: partial/regex match on all timeline IDs
        all_ids = [self._score_timeline.id] + [
            tl.id for tl in self._performance_timelines
        ]
        try:
            resolved_id = resolve_id(uid, all_ids, warn_multiple=True)
            if resolved_id == self._score_timeline.id:
                return self._score_timeline
            for perf_tl in self._performance_timelines:
                if perf_tl.id == resolved_id:
                    return perf_tl
        except KeyError:
            pass

        raise KeyError(
            f"No timeline with uid or role '{uid}'. "
            f"Available: 'score', "
            + ", ".join(f"'{tl.id}'" for tl in self._performance_timelines)
        )

    def create_timelines(self, id_pattern: str | None = None) -> list["Timeline"]:
        """Return all timelines produced from loaded data.

        For MatchfileLoader: ``[score_timeline, perf_p01, ..., perf_pN]``.
        The score timeline is always first.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of Timeline objects. Empty if ``load()`` has not been
            called yet.
        """
        if self._score_timeline is None:
            return []
        timelines = [self._score_timeline] + list(self._performance_timelines)
        return AlignmentLoader._filter_timelines_by_id_pattern(timelines, id_pattern)

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        """Return string representation."""
        n_perf = len(self._performance_timelines)
        n_claims = len(self._claims)
        n_rejected = len(self._rejected_files)
        return (
            f"MatchfileLoader(performances={n_perf}, "
            f"claims={n_claims}, rejected={n_rejected})"
        )

    def __len__(self) -> int:
        """Return total number of loaded performance timelines."""
        return len(self._performance_timelines)

    # endregion


# endregion
