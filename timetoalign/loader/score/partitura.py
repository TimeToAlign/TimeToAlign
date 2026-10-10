"""PartituraLoader: Load scores using partitura."""

from __future__ import annotations

import bisect
from fractions import Fraction
from numbers import Integral
from pathlib import Path
from typing import Any

from timetoalign.core.backends import suppressed_backend_warnings

with suppressed_backend_warnings():
    import partitura as pt
    import partitura.score as pts

from timetoalign.core import NumberType, TimeUnit

from .base import ScoreLoader
from .store import ScoreStore
from .stores import (
    AnnotationEventData,
    ControlEventData,
    MeasureData,
    NoteEventData,
)

# Constant pitch maps (moved out of per-note loop)
_GPC_MAP = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}
_BASE_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}

# Flow control marker types (constant tuple, defined once)
_FLOW_MARKER_TYPES = (
    pts.Ending,
    pts.DaCapo,
    pts.DalSegno,
    pts.Fine,
    pts.Segno,
    pts.Coda,
    pts.ToCoda,
)

# Control event types to capture
_CONTROL_EVENT_TYPES = (
    pts.TimeSignature,
    pts.KeySignature,
    pts.Tempo,
    pts.Direction,
    pts.Slur,
)


def _coordinate_struct(value: Fraction | float) -> dict[str, Any]:
    """Return a coordinate struct without inventing a ratio for a float."""
    if isinstance(value, Fraction):
        return {
            "value": float(value),
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    return {"value": float(value), "numerator": None, "denominator": None}


def _struct_coordinate_value(struct: dict[str, Any]) -> Fraction | float:
    """Read a coordinate pair, falling back to its value-only projection."""
    numerator = struct.get("numerator")
    denominator = struct.get("denominator")
    if (
        isinstance(numerator, Integral)
        and not isinstance(numerator, bool)
        and isinstance(denominator, Integral)
        and not isinstance(denominator, bool)
    ):
        return Fraction(int(numerator), int(denominator))
    return float(struct["value"])


class PartituraLoader(ScoreLoader):
    """Load symbolic scores using partitura.

    Supports MusicXML (fully typed) and MIDI (quantized).
    Returns ScoreStore with category-specific data.

    All TTA timelines start at coordinate 0.  When partitura reports negative
    quarter-beat onsets (anacrusis notes that precede the first full measure),
    this loader shifts every event coordinate so that the earliest onset maps
    to 0.0.  The magnitude of that shift is available as
    :attr:`anacrusis_offset` and is also stored in
    ``ScoreStore.metadata["anacrusis_offset"]`` so that downstream consumers
    (e.g. ``MatchfileLoader``) can apply the same correction when comparing
    raw partitura coordinates against this loader's stored values.

    The shift is equivalent to the offset of a ``ShiftMap`` whose forward
    direction is ``raw_partitura_coord → TTA_coord`` (i.e. adding the offset).
    Its ``InverseMap`` (subtracting the offset) converts TTA coordinates back
    to raw partitura space when needed.
    """

    def __init__(
        self,
        *,
        force_note_ids: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize PartituraLoader.

        Args:
            force_note_ids: If True, force note IDs from partitura.
            **kwargs: Additional arguments passed to parent ScoreLoader.
        """
        super().__init__(**kwargs)
        self._force_note_ids = force_note_ids
        self._anacrusis_offset: float = 0.0

    @property
    def anacrusis_offset(self) -> float:
        """Quarter-beat shift applied to normalise anacrusis coordinates.

        Equal to ``-min(raw_partitura_onset)`` across all notes in the most
        recently loaded source.  Zero when the score has no anacrusis (i.e.
        the first note starts at or after beat 0).

        Use this value to convert a raw partitura coordinate ``c`` to the TTA
        coordinate stored on the resulting timeline: ``tta_coord = c + offset``.
        Conversely, ``raw_coord = tta_coord - offset``.
        """
        return self._anacrusis_offset

    def _load_source(self, source: Path) -> ScoreStore:
        """Load score and return ScoreStore."""
        return self._do_load(source)

    def _do_load(self, source: Path) -> ScoreStore:
        """Core loading logic for a single source file."""
        with suppressed_backend_warnings():
            score = pt.load_score(str(source), force_note_ids=self._force_note_ids)

        # Flatten parts
        parts = []
        if isinstance(score, pts.Score):
            parts = score.parts
        elif isinstance(score, (pts.Part, pts.PartGroup)):
            parts = [score] if isinstance(score, pts.Part) else score.children
        elif isinstance(score, list):
            parts = score

        note_rows = []
        measure_rows = []
        control_rows = []
        annotation_rows = []
        has_rests = False

        for part_idx, part in enumerate(parts):
            part_id = getattr(part, "id", f"P{part_idx+1}")

            # Get quarter beat mapping — use quarter_map, NOT beat_map.
            # beat_map is time-signature-relative in partitura; quarter_map
            # always returns quarter-note positions regardless of time sig.
            beat_map = part.quarter_map  # divs -> quarter beats

            # Partitura's quarter map is a float interpolation, but its input
            # positions and quarter durations are integer division counts.
            # Integrate those exact durations ourselves so rational coordinates
            # do not depend on reconstructing a Fraction from a float.
            quarter_duration_changes: list[tuple[int, int]] | None = []
            for time_value, duration_value in part.quarter_durations():
                if isinstance(duration_value, Integral) and not isinstance(
                    duration_value, bool
                ):
                    exact_duration = int(duration_value)
                elif isinstance(duration_value, float) and duration_value.is_integer():
                    exact_duration = int(duration_value)
                else:
                    quarter_duration_changes = None
                    break
                quarter_duration_changes.append((int(time_value), exact_duration))

            if quarter_duration_changes:
                deduplicated_changes: list[tuple[int, int]] = []
                for change in quarter_duration_changes:
                    if (
                        deduplicated_changes
                        and deduplicated_changes[-1][0] == change[0]
                    ):
                        deduplicated_changes[-1] = change
                    else:
                        deduplicated_changes.append(change)
                quarter_duration_changes = deduplicated_changes

            first_div = int(part.first_point.t)

            def exact_quarter_delta(start: int, end: int) -> Fraction | None:
                """Integrate exact quarter duration between two positions."""
                if not quarter_duration_changes:
                    return None

                def duration_at(time_value: int) -> int | None:
                    current: int | None = None
                    for change_time, change_duration in quarter_duration_changes:
                        if change_time > time_value:
                            break
                        current = change_duration
                    return current

                if end < start:
                    result = exact_quarter_delta(end, start)
                    return -result if result is not None else None

                current = start
                result = Fraction(0)
                duration = duration_at(current)
                if duration is None or duration == 0:
                    return None
                for change_time, change_duration in quarter_duration_changes:
                    if change_time <= current:
                        continue
                    if change_time >= end:
                        break
                    result += Fraction(change_time - current, duration)
                    current = change_time
                    duration = change_duration
                    if duration == 0:
                        return None
                result += Fraction(end - current, duration)
                return result

            first_measure = next(
                (
                    obj
                    for obj in part.iter_all(include_subclasses=True)
                    if isinstance(obj, pts.Measure)
                ),
                None,
            )
            anacrusis_correction = Fraction(0)
            if first_measure is not None and first_measure.end is not None:
                actual_duration = exact_quarter_delta(
                    int(first_measure.start.t), int(first_measure.end.t)
                )
                first_time_signature = next(
                    (
                        time_signature
                        for time_signature in part.time_sigs
                        if time_signature.start.t <= first_measure.start.t
                    ),
                    None,
                )
                if actual_duration is not None and first_time_signature is not None:
                    normal_duration = Fraction(
                        int(first_time_signature.beats) * 4,
                        int(first_time_signature.beat_type),
                    )
                    if actual_duration < normal_duration:
                        anacrusis_correction = actual_duration

            def exact_quarter_position(div_val: int) -> Fraction | None:
                """Return a division position in exact quarter-note units."""
                delta = exact_quarter_delta(first_div, div_val)
                return delta - anacrusis_correction if delta is not None else None

            # Cache beat_map calls: many notes share divisions or measure starts
            beat_map_cache: dict[int, float] = {}

            def cached_beat_map(div_val: int) -> float:
                """Return cached float result of beat_map(div_val)."""
                result = beat_map_cache.get(div_val)
                if result is None:
                    result = float(beat_map(div_val))
                    beat_map_cache[div_val] = result
                return result

            # Cache exact conversions; a float map is retained only as the
            # value-only fallback for a genuinely non-rational Partitura map.
            beat_coordinate_cache: dict[int, Fraction | float] = {}

            def cached_beat_coordinate(div_val: int) -> Fraction | float:
                """Return the exact or value-only quarter position."""
                if div_val not in beat_coordinate_cache:
                    exact = exact_quarter_position(div_val)
                    beat_coordinate_cache[div_val] = (
                        exact if exact is not None else cached_beat_map(div_val)
                    )
                return beat_coordinate_cache[div_val]

            # ===== Single pass: collect all objects by type =====
            measures_list: list[pts.Measure] = []
            notes_and_rests: list[pts.Note | pts.Rest | pts.GraceNote] = []
            flow_markers: list = []
            control_objects: list = []
            annotation_objects: list = []

            for obj in part.iter_all(include_subclasses=True):
                if isinstance(obj, pts.Measure):
                    measures_list.append(obj)
                elif isinstance(obj, (pts.Note, pts.Rest, pts.GraceNote)):
                    notes_and_rests.append(obj)
                elif isinstance(obj, _FLOW_MARKER_TYPES):
                    flow_markers.append(obj)
                elif isinstance(obj, pts.Words):
                    annotation_objects.append(obj)
                elif isinstance(obj, _CONTROL_EVENT_TYPES):
                    control_objects.append(obj)

            # Build measure map for MC lookup
            measures = sorted(measures_list, key=lambda m: m.start.t)
            measure_starts = [m.start.t for m in measures]

            def get_mc(t: int) -> int:
                idx = bisect.bisect_right(measure_starts, t) - 1
                return idx + 1 if idx >= 0 else 1

            # Build measure_number_map cache: MC -> measure number string
            has_mn_map = hasattr(part, "measure_number_map")
            mn_cache: dict[int, str] = {}
            if has_mn_map:
                for i, m in enumerate(measures):
                    mc = i + 1
                    mn_cache[mc] = str(part.measure_number_map(m.start.t))

            def get_mn(mc: int, start_t: int) -> str:
                """Get measure number string, using cache when possible."""
                cached = mn_cache.get(mc)
                if cached is not None:
                    return cached
                if has_mn_map:
                    result = str(part.measure_number_map(start_t))
                    mn_cache[mc] = result
                    return result
                return str(mc)

            # Pre-cache beat_map values for all measure starts
            # (these are reused heavily during note processing)
            measure_start_qb: dict[int, float] = {}
            for m_start in measure_starts:
                measure_start_qb[m_start] = cached_beat_map(m_start)
                # Also warm up the Fraction cache for measure starts
                cached_beat_coordinate(m_start)

            # ===== Extract Flow Control from part.repeats =====
            # repeat_blocks: list of (start_mc, after_mc) tuples.
            # after_mc is the first MC *after* the repeat (partitura's rep.end.t
            # maps to the downbeat of the continuation, not the last bar inside).
            repeat_start_mcs: set[int] = set()
            repeat_end_mcs: set[int] = set()  # last MC inside each repeat
            repeat_blocks: list[tuple[int, int]] = []  # (start_mc, after_mc)

            if hasattr(part, "repeats") and part.repeats:
                for rep in part.repeats:
                    start_mc = get_mc(rep.start.t)
                    repeat_start_mcs.add(start_mc)

                    qb_start = cached_beat_coordinate(rep.start.t)
                    control_rows.append(
                        {
                            "id": f"repeat_start_{start_mc}",
                            "name": "RepeatStart",
                            "temporal_type": "instant",
                            "event_type": "RepeatStart",
                            "quarterbeats": _coordinate_struct(qb_start),
                            "duration_qb": None,
                            "mc": start_mc,
                            "mn": get_mn(start_mc, rep.start.t),
                            "mc_onset": None,
                            "mn_onset": None,
                            "subtype": "RepeatStart",
                            "value": None,
                            "text": "||:",
                            "voice": None,
                            "staff": None,
                            "part_id": part_id,
                        }
                    )

                    if rep.end:
                        # Determine whether rep.end.t falls on a measure *start*
                        # (meaning there is a continuation measure after the repeat)
                        # or on a measure *end* (the piece ends with this repeat).
                        #
                        # bisect_left gives the insertion index for rep.end.t; if
                        # measure_starts[idx] == rep.end.t the time is exactly on
                        # a measure downbeat → the repeat ends before that measure.
                        # Otherwise rep.end.t is a bar-end and the last bar is idx (0-indexed).
                        end_idx = bisect.bisect_left(measure_starts, rep.end.t)
                        if (
                            end_idx < len(measure_starts)
                            and measure_starts[end_idx] == rep.end.t
                        ):
                            # rep.end.t is the downbeat of the continuation measure.
                            after_mc = end_idx + 1  # 1-indexed MC of the continuation
                            last_mc_in_repeat = after_mc - 1
                        else:
                            # rep.end.t is the end of the last bar (piece ends here).
                            last_mc_in_repeat = (
                                end_idx  # 1-indexed: end_idx == last 0-index + 1 → MC
                            )
                            after_mc = last_mc_in_repeat + 1  # virtual "after" MC

                        repeat_end_mcs.add(last_mc_in_repeat)
                        repeat_blocks.append((start_mc, after_mc))

                        qb_end = cached_beat_coordinate(rep.end.t)
                        control_rows.append(
                            {
                                "id": f"repeat_end_{last_mc_in_repeat}",
                                "name": "RepeatEnd",
                                "temporal_type": "instant",
                                "event_type": "RepeatEnd",
                                "quarterbeats": _coordinate_struct(qb_end),
                                "duration_qb": None,
                                "mc": last_mc_in_repeat,
                                "mn": get_mn(last_mc_in_repeat, rep.end.t),
                                "mc_onset": None,
                                "mn_onset": None,
                                "subtype": "RepeatEnd",
                                "value": None,
                                "text": ":||",
                                "voice": None,
                                "staff": None,
                                "part_id": part_id,
                            }
                        )

            # ===== Extract Endings (Volta) and other flow control markers =====
            # ending_mcs: MC -> volta number (first MC of each ending).
            # ending_after_mcs: MC -> first MC after each ending (from Ending.end.t).
            # Navigation directions in implicit X-measures are attached to the
            # preceding playable row in the measure-table convention. Target
            # markers remain on their own row.
            ending_mcs: dict[int, int] = {}  # start MC -> volta number
            ending_after_mcs: dict[int, int] = {}  # start MC -> first MC after ending
            marker_names: dict[int, str] = {}
            jump_bwd: dict[int, str] = {}
            jump_fwd: dict[int, str] = {}
            play_until: dict[int, str] = {}
            fine_mcs: set[int] = set()
            implicit_mcs = {
                i + 1
                for i, measure in enumerate(measures)
                if str(getattr(measure, "name", "")).startswith("X")
            }

            for obj in flow_markers:
                marker_mc = get_mc(obj.start.t)
                qb = cached_beat_coordinate(obj.start.t)

                if isinstance(obj, pts.Ending):
                    ending_num = getattr(obj, "number", 1)
                    try:
                        num = int(ending_num)
                    except (TypeError, ValueError):
                        num = 1
                    ending_mcs[marker_mc] = num
                    if obj.end:
                        ending_after_mcs[marker_mc] = get_mc(obj.end.t)
                elif isinstance(
                    obj,
                    (
                        pts.DaCapo,
                        pts.DalSegno,
                        pts.Fine,
                        pts.Segno,
                        pts.Coda,
                        pts.ToCoda,
                    ),
                ):
                    if isinstance(obj, pts.Segno):
                        marker_names[marker_mc] = "segno"
                    elif isinstance(obj, pts.Coda):
                        marker_names[marker_mc] = "coda"
                    elif isinstance(obj, pts.Fine):
                        fine_mc = (
                            marker_mc - 1 if marker_mc in implicit_mcs else marker_mc
                        )
                        marker_names[fine_mc] = "fine"
                        fine_mcs.add(fine_mc)
                    if isinstance(obj, pts.DaCapo):
                        jump_bwd[marker_mc] = "start"
                    elif isinstance(obj, pts.DalSegno):
                        jump_bwd[marker_mc] = "segno"

                control_rows.append(
                    {
                        "id": f"{type(obj).__name__.lower()}_{marker_mc}",
                        "name": type(obj).__name__,
                        "temporal_type": "instant",
                        "event_type": type(obj).__name__,
                        "quarterbeats": _coordinate_struct(qb),
                        "duration_qb": None,
                        "mc": marker_mc,
                        "mn": get_mn(marker_mc, obj.start.t),
                        "mc_onset": None,
                        "mn_onset": None,
                        "subtype": type(obj).__name__,
                        "value": (
                            float(ending_mcs.get(marker_mc, 1))
                            if isinstance(obj, pts.Ending)
                            else None
                        ),
                        "text": type(obj).__name__,
                        "voice": None,
                        "staff": None,
                        "part_id": part_id,
                    }
                )

            # Partitura drops MusicXML coda identifiers, but retains the
            # implicit X2/X3 measures at those destinations. Associate each
            # such destination with its navigation instruction.
            coda_index = 0
            for marker_mc, target in jump_bwd.items():
                if marker_mc not in implicit_mcs:
                    continue
                coda_index += 1
                coda_name = f"{coda_index + 1}e"
                marker_names[marker_mc] = coda_name
                jump_fwd[marker_mc] = coda_name
                if target == "start":
                    play_until[marker_mc] = "segno"

            # D.S. directions in these scores return to the Fine section.
            for marker_mc, target in jump_bwd.items():
                if target == "segno" and fine_mcs:
                    play_until[marker_mc] = "fine"

            # ===== Derive next[] for each measure from repeat/volta structure =====
            # next_mc_map: MC -> list of possible successor MCs (in visitation order).
            # An absent entry means "next sequential MC" (or -1 for the last bar).
            #
            # Repeat structure remains active when navigation markers are present;
            # the marker fields below let ScoreFlowController combine the two kinds of
            # flow control.
            next_mc_map: dict[int, list[int]] = {}
            num_measures = len(measures)

            # Directions in implicit X-measures execute after the preceding
            # playable measure; other directions stay on their own row.
            jump_rows = {}
            for mc, target in jump_bwd.items():
                row_mc = mc - 1 if mc in implicit_mcs else mc
                if 1 <= row_mc <= num_measures:
                    jump_rows[row_mc] = (target, play_until.get(mc))

            for rep_start_mc, rep_after_mc in repeat_blocks:
                # Collect volta-1 start MCs inside this repeat block.
                volta1_starts = sorted(
                    mc
                    for mc, v in ending_mcs.items()
                    if v == 1 and rep_start_mc <= mc < rep_after_mc
                )
                volta2_starts = sorted(
                    mc
                    for mc, v in ending_mcs.items()
                    if v == 2 and rep_start_mc <= mc < rep_after_mc
                )

                if not volta1_starts:
                    # Simple repeat (no voltas): the last bar inside the repeat
                    # gets next = [rep_start_mc, rep_after_mc].
                    last_in_rep = rep_after_mc - 1
                    if last_in_rep >= 1 and last_in_rep <= num_measures:
                        next_mc_map[last_in_rep] = [rep_start_mc, rep_after_mc]
                else:
                    # Volta repeat: identify the fork bar and the volta-1 end bar.
                    first_v1_mc = volta1_starts[0]
                    first_v2_mc = volta2_starts[0] if volta2_starts else rep_after_mc
                    fork_mc = first_v1_mc - 1

                    # Fork bar: choose volta 1 (first visit) or volta 2 (second visit).
                    if fork_mc >= 1:
                        next_mc_map[fork_mc] = [first_v1_mc, first_v2_mc]

                    # The last MC of volta 1 jumps unconditionally back to repeat start.
                    last_v1_mc = rep_after_mc - 1
                    if last_v1_mc >= 1 and last_v1_mc <= num_measures:
                        next_mc_map[last_v1_mc] = [rep_start_mc]

            # The flow controller uses a non-sequential next edge to identify
            # the origin of a D.S. or D.C. jump.  Keep the typed instruction in
            # the dedicated fields, and mirror its destination here so it is
            # executable during traversal.
            segno_mc = next(
                (mc for mc, name in marker_names.items() if name == "segno"), None
            )
            for mc, (target, _until) in jump_rows.items():
                if mc > num_measures:
                    continue
                if target == "segno" and segno_mc is not None:
                    next_mc_map[mc] = [segno_mc]
                elif target == "start":
                    next_mc_map[mc] = [1]

            # Process Measures
            for i, m in enumerate(measures):
                qb_start = cached_beat_coordinate(m.start.t)
                qb_end = cached_beat_coordinate(m.end.t)
                dur = qb_end - qb_start
                mc = i + 1

                # Resolve next: use pre-computed map, or default sequential.
                if mc in next_mc_map:
                    # A repeat closing the piece continues past the last bar:
                    # the piece ends there, which measure tables spell -1.
                    mc_next: list[int] | None = [
                        successor if successor <= num_measures else -1
                        for successor in next_mc_map[mc]
                    ]
                else:
                    mc_next = None  # default: next sequential MC (handled by ScoreFlowController)

                measure_rows.append(
                    {
                        # mc:NNNNN format based on measure count
                        "id": f"mc:{mc:05d}",
                        "name": str(m.number),
                        "temporal_type": "interval",
                        "event_type": "Measure",
                        "quarterbeats": _coordinate_struct(qb_start),
                        "duration_qb": _coordinate_struct(dur),
                        "mc": mc,
                        "mn": str(m.number),
                        "timesig": None,  # Could extract from part
                        # Flow control fields
                        "start_repeat": mc in repeat_start_mcs,
                        "end_repeat": mc in repeat_end_mcs,
                        "volta": ending_mcs.get(mc),
                        "next": mc_next,
                        "markers": marker_names.get(mc),
                        "jump_bwd": jump_rows.get(mc, (None, None))[0],
                        "jump_fwd": jump_fwd.get(
                            mc + 1
                            if mc + 1 in jump_fwd and mc + 1 in implicit_mcs
                            else mc
                        ),
                        "play_until": jump_rows.get(mc, (None, None))[1],
                        "fine": mc in fine_mcs,
                        "part_id": part_id,
                    }
                )

            # Process Notes/Rests (from pre-collected list)
            for obj in notes_and_rests:
                is_rest = isinstance(obj, pts.Rest)
                if is_rest:
                    has_rests = True

                # Temporal
                start_div = obj.start.t
                dur_div = obj.duration if hasattr(obj, "duration") else 0

                qb_start = cached_beat_coordinate(start_div)
                qb_end = cached_beat_coordinate(start_div + dur_div)
                dur_qb = qb_end - qb_start

                # MC context
                mc = get_mc(start_div)
                mn = get_mn(mc, start_div)

                # mc_onset: offset from measure start
                if mc > 0 and mc <= len(measure_starts):
                    m_start_div = measure_starts[mc - 1]
                    # Use cached float values to compute onset, then convert
                    mc_onset = qb_start - cached_beat_coordinate(m_start_div)
                else:
                    mc_onset = Fraction(0)

                # Pitch
                midi = None
                specific_pitch = None
                octave = None
                tpc = None

                _sp_label = ""
                if not is_rest and hasattr(obj, "midi_pitch"):
                    midi = int(obj.midi_pitch)

                    if hasattr(obj, "step"):
                        step = obj.step
                        alter = int(getattr(obj, "alter", 0) or 0)
                        octave = int(getattr(obj, "octave", 4) or 4)

                        acc_str = ""
                        if alter > 0:
                            acc_str = "\u266f" * alter
                        elif alter < 0:
                            acc_str = "\u266d" * abs(alter)

                        tpc = _BASE_FIFTHS.get(step, 0) + (7 * alter)

                        specific_pitch = {
                            "step": step,
                            "alter": alter,
                            "octave": octave,
                            "cents": 0.0,
                        }
                        _sp_label = f"{step}{acc_str}{octave}"

                # Compute name from specific pitch if available
                note_name = ""
                if _sp_label:
                    note_name = _sp_label
                elif midi is not None:
                    note_name = f"MIDI {midi}"

                note_rows.append(
                    {
                        # ID auto-generated from event_type (Note or Rest)
                        "name": note_name,
                        "temporal_type": "interval" if dur_qb > 0 else "instant",
                        "event_type": "Rest" if is_rest else "Note",
                        "quarterbeats": _coordinate_struct(qb_start),
                        "duration_qb": _coordinate_struct(dur_qb),
                        "mc": mc,
                        "mn": mn,
                        "mc_onset": _coordinate_struct(mc_onset),
                        "mn_onset": _coordinate_struct(mc_onset),
                        "timesig": None,
                        "duration": None,
                        "nominal_duration": None,
                        "scalar": None,
                        "specific_pitch": specific_pitch,
                        "midi": midi,
                        "tpc": tpc,
                        "octave": octave,
                        "velocity": 64,
                        "tied": 0,
                        "gracenote": (
                            "grace" if isinstance(obj, pts.GraceNote) else None
                        ),
                        "chord_id": None,
                        "voice": getattr(obj, "voice", None),
                        "staff": getattr(obj, "staff", None),
                        "part_id": part_id,
                    }
                )

            # Process control events (from pre-collected list)
            for obj in control_objects:
                qb = cached_beat_coordinate(obj.start.t)
                obj_mc = get_mc(obj.start.t)
                control_rows.append(
                    {
                        # ID auto-generated from event_type (class name)
                        "name": obj.__class__.__name__,
                        # Control objects are stored as point events here;
                        # their Partitura end marker is not exported as an
                        # interval coordinate.
                        "temporal_type": "instant",
                        "event_type": obj.__class__.__name__,
                        "quarterbeats": _coordinate_struct(qb),
                        "duration_qb": None,
                        "mc": obj_mc,
                        "mn": get_mn(obj_mc, obj.start.t),
                        "mc_onset": None,
                        "mn_onset": None,
                        "subtype": obj.__class__.__name__,
                        "value": None,
                        "text": str(obj),
                        "voice": getattr(obj, "voice", None),
                        "staff": getattr(obj, "staff", None),
                        "part_id": part_id,
                    }
                )

            # Process annotation events (from pre-collected list)
            for obj in annotation_objects:
                qb = cached_beat_coordinate(obj.start.t)
                obj_mc = get_mc(obj.start.t)
                annotation_rows.append(
                    {
                        # ID auto-generated from event_type
                        "name": getattr(obj, "text", str(obj)),
                        "temporal_type": "instant",
                        "event_type": "Text",
                        "quarterbeats": _coordinate_struct(qb),
                        "duration_qb": None,
                        "mc": obj_mc,
                        "mn": get_mn(obj_mc, obj.start.t),
                        "mc_onset": None,
                        "mn_onset": None,
                        "subtype": "Text",
                        "text": getattr(obj, "text", str(obj)),
                        "staff": getattr(obj, "staff", None),
                        "part_id": part_id,
                    }
                )

        # ── Anacrusis normalisation ───────────────────────────────────────────
        # TTA timelines always start at coordinate 0.  When partitura reports
        # negative quarter-beat onsets (anacrusis notes), we shift all event
        # coordinates so the earliest onset becomes 0.0.
        #
        # The shift is applied uniformly to notes, measures, controls, and
        # annotations.  Its value is exposed via self.anacrusis_offset and
        # stored in the ScoreStore metadata so that any loader that needs to
        # compare raw partitura values against stored TTA coordinates can apply
        # the same correction.
        all_onsets = [
            _struct_coordinate_value(r["quarterbeats"])
            for r in note_rows
            if r.get("quarterbeats") is not None
        ]
        min_qb: Fraction | float = min(all_onsets, default=Fraction(0))
        offset: Fraction | float = -min_qb if min_qb < 0 else Fraction(0)
        self._anacrusis_offset = float(offset)

        if offset:
            for rows in (note_rows, measure_rows, control_rows, annotation_rows):
                for r in rows:
                    old_qb = _struct_coordinate_value(r["quarterbeats"])
                    new_qb = old_qb + offset
                    r["quarterbeats"] = _coordinate_struct(new_qb)

        # Build data
        notes_data = getattr(NoteEventData, "from_" + "dicts")(
            note_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            has_rests=has_rests,
        )

        measures_data = (
            MeasureData.from_dicts(
                measure_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
            if measure_rows
            else MeasureData.empty()
        )

        controls_data = (
            getattr(ControlEventData, "from_" + "dicts")(
                control_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
            if control_rows
            else ControlEventData.empty()
        )

        annotations_data = (
            getattr(AnnotationEventData, "from_" + "dicts")(
                annotation_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
            )
            if annotation_rows
            else AnnotationEventData.empty()
        )

        return ScoreStore(
            notes=notes_data,
            measures=measures_data,
            controls=controls_data,
            annotations=annotations_data,
            metadata={
                "format": "score",
                "parser": "partitura",
                "source": str(source),
                "has_rests": has_rests,
                "anacrusis_offset": float(offset),
            },
        )
