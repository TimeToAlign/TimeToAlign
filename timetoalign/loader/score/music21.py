"""Music21Loader: Load scores using music21."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from fractions import Fraction
from typing import Any

import music21 as m21
from music21.meter.base import TimeSignature

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.time import (
    rational_to_struct,
)
from timetoalign.loader._sources import (
    Source,
    describe_source,
    on_disk,
    readable,
    source_record,
)
from timetoalign.loader.base import LOCATOR, locate

from .base import ScoreLoader
from .store import ScoreStore
from .stores import (
    AnnotationEventData,
    ControlEventData,
    MeasureData,
    NoteEventData,
)

module_logger = logging.getLogger(__name__)


class Music21Loader(ScoreLoader):
    """Load symbolic scores using music21.

    Returns ScoreStore with category-specific data.
    Uses recursive element parsing to extract:
    - Notes/Rests (with chord expansion)
    - Measures (structure)
    - Controls (dynamics, tempo, etc.)
    - Annotations (text)

    All TTA timelines start at coordinate 0.  When music21 reports negative
    quarter-beat offsets (anacrusis notes preceding the first full measure),
    this loader shifts every event coordinate so the earliest onset maps to
    0.0.  The magnitude of that shift is exposed via :attr:`anacrusis_offset`
    and stored in ``ScoreStore.metadata["anacrusis_offset"]``.

    See ``PartituraLoader`` for the full rationale; both loaders apply the
    same convention so that score timelines built from either source are
    directly comparable and compatible with ``MatchfileLoader``.

    Every event names its file's parse and, as its ``locator``, its
    place in music21's reading of the file: ``part:<p> element:<k>``, the
    part's index and the element's in the part's flattened stream
    (``note:<n>`` added for a chord's n-th note), and for a measure
    ``part:<p> measure:<i>``, all from 0. Where the source states a key of
    its own, ``key:<key>`` follows: the ``xml:id`` / ``id`` of a note,
    control or text, a measure's number (``part:0 measure:0 key:1``).

    Known issue: music21 gives a measure without timed content no length
    (the empty measures of an MEI skeleton, which leaves out the measures it
    repeats unchanged), and its MEI reader leaves out ``<ending>`` measures
    holding nothing but measure rests. A source with such a measure states
    no measure positions through music21, and the load raises
    ``ValueError`` naming them.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._anacrusis_offset: Fraction | float = Fraction(0)

    @property
    def anacrusis_offset(self) -> Fraction | float:
        """Quarter-beat shift applied to normalise anacrusis coordinates.

        Equal to ``-min(raw_onset)`` across all notes in the most recently
        loaded source, exact.  Zero when the score has no anacrusis.
        """
        return self._anacrusis_offset

    def _backends(self) -> tuple[str, ...]:
        """music21 reads the score."""
        return ("music21",)

    def _load_source(self, source: Source) -> ScoreStore:
        """Load score and return ScoreStore.

        music21 reads files on disk only: a member of a ZIP archive is read
        from its copy (:func:`~timetoalign.loader.materialize`), and an error
        music21 raises then names the archive and the member.
        """
        with on_disk(source) as path:
            score = m21.converter.parse(str(path), forceSource=True)

        note_rows = []
        measure_rows = []
        control_rows = []
        annotation_rows = []
        has_rests = False

        parts = score.parts if score.hasPartLikeStreams() else [score]

        # ===== MEI-specific pre-processing =====
        # MEI files exported by MuseScore require direct XML parsing to recover
        # two things that music21's MEI parser misses:
        # 1. Sparse skeleton expansion: only boundary measures are stored; gaps
        #    between non-sequential @n values imply empty intermediate measures.
        # 2. Volta (ending) detection: <ending> elements containing volta measures
        #    are not translated to RepeatBracket spanners by music21.
        is_mei = source.suffix.lower() == ".mei"
        mei_measure_info: dict[int, dict[str, Any]] = (
            self._parse_mei_measure_info(source) if is_mei else {}
        )  # mc -> {number, volta, start_repeat, end_repeat}

        for part_idx, part in enumerate(parts):
            # Ensure part_id is always a string (MEI may return int IDs)
            part_id = str(part.id) if part.id else f"P{part_idx+1}"

            # Get measure list for MC lookup
            measure_list = list(part.getElementsByClass(m21.stream.Measure))
            self._refuse_measures_without_length(measure_list, source)
            mei_skeleton_expanded = False

            # ===== MEI sparse skeleton expansion =====
            # MEI files exported by MuseScore use a compact representation where
            # only boundary measures are stored and intermediate empty measures are
            # implied (indicated by non-sequential m.number / MEI @n values).
            # Expand any gaps so that every MC from 1..max_n is present.
            if is_mei and measure_list:
                original_measure_count = len(measure_list)
                measure_list = self._expand_mei_skeleton(measure_list)
                mei_skeleton_expanded = len(measure_list) != original_measure_count

            # Full-score MEI keeps every source measure, including unnumbered
            # boundary measures that carry a repeat start. Sparse MEI skeletons
            # are expanded by ``@n``, so their metadata must use that number.
            mei_info_by_mc = (
                {
                    info["number"]: info
                    for info in mei_measure_info.values()
                    if info["number"] is not None
                }
                if is_mei and mei_skeleton_expanded
                else mei_measure_info
            )

            measure_offsets = [
                (float(m.offset), i + 1, m) for i, m in enumerate(measure_list)
            ]

            def get_mc_and_onset(offset: float) -> tuple[int | None, Fraction]:
                """Find MC and offset from measure start."""
                mc = None
                mc_onset = Fraction(0)
                for m_off, idx, m in reversed(measure_offsets):
                    if m_off <= offset + 1e-5:
                        mc = idx
                        mc_onset = Fraction(offset) - Fraction(m_off)
                        break
                return mc, mc_onset

            # ===== Extract Volta (RepeatBracket) information =====
            # For MEI files, music21 does not parse <ending> elements into
            # RepeatBracket spanners. We use the pre-parsed mei_measure_info instead.
            # For MusicXML, the standard RepeatBracket spanners are used.
            volta_by_offset: dict[float, int] = {}
            if not is_mei:
                for rb in part.flatten().getElementsByClass(m21.spanner.RepeatBracket):
                    volta_num = rb.number
                    if volta_num is not None:
                        for el in rb.getSpannedElements():
                            if isinstance(el, m21.stream.Measure):
                                volta_by_offset[float(el.offset)] = int(volta_num)

            # ===== Extract flow markers from measures =====
            # First pass: collect barline, volta, and expression markers
            measure_info: list[dict[str, Any]] = []
            for i, m in enumerate(measure_list):
                mc = i + 1
                qb = Fraction(m.offset)
                dur = Fraction(m.duration.quarterLength)

                # For MEI files, use pre-parsed XML data (music21's MEI parser
                # drops <ending> elements and doesn't create RepeatBracket spanners).
                if is_mei and mc in mei_info_by_mc:
                    mei_info = mei_info_by_mc[mc]
                    start_repeat = mei_info.get("start_repeat", False)
                    end_repeat = mei_info.get("end_repeat", False)
                    volta = mei_info.get("volta")
                else:
                    # Check left barline for repeat start
                    start_repeat = False
                    left_bl = m.leftBarline
                    if left_bl is not None:
                        if isinstance(left_bl, m21.bar.Repeat):
                            if getattr(left_bl, "direction", None) == "start":
                                start_repeat = True
                        elif left_bl.type == "heavy-light":
                            start_repeat = True

                    # Check right barline for repeat end
                    end_repeat = False
                    right_bl = m.rightBarline
                    if right_bl is not None:
                        if isinstance(right_bl, m21.bar.Repeat):
                            if getattr(right_bl, "direction", None) == "end":
                                end_repeat = True
                        elif right_bl.type == "light-heavy":
                            end_repeat = True

                    # Get volta number
                    volta = volta_by_offset.get(float(m.offset))

                measure_info.append(
                    {
                        "mc": mc,
                        "qb": qb,
                        "dur": dur,
                        "mn": str(m.number),
                        "start_repeat": start_repeat,
                        "end_repeat": end_repeat,
                        "volta": volta,
                    }
                )

            # Build measure rows; ScoreLoader.load derives their successors.
            for i, info in enumerate(measure_info):
                measure_rows.append(
                    {
                        LOCATOR: locate(f"part:{part_idx} measure:{i}", info["mn"]),
                        "name": info["mn"],
                        "temporal_type": "interval",
                        "event_type": "Measure",
                        "quarterbeats": rational_to_struct(info["qb"]),
                        "duration_qb": rational_to_struct(info["dur"]),
                        "mc": info["mc"],
                        "mn": info["mn"],
                        "timesig": None,
                        # Flow control fields
                        "start_repeat": info["start_repeat"],
                        "end_repeat": info["end_repeat"],
                        "volta": info["volta"],
                        "part_id": part_id,
                    }
                )

            # Process Notes/Rests/Controls
            flat_part = part.flatten()

            for k, obj in enumerate(flat_part):
                if isinstance(
                    obj, (m21.stream.Measure, m21.stream.Part, m21.stream.Score)
                ):
                    continue
                place = f"part:{part_idx} element:{k}"

                offset = float(obj.offset)
                duration = float(obj.duration.quarterLength)
                qb = Fraction(offset)
                dur_qb = Fraction(duration)
                mc, mc_onset = get_mc_and_onset(offset)
                mn = (
                    str(obj.measureNumber)
                    if obj.measureNumber is not None
                    else str(mc) if mc else None
                )

                # Notes (Rests are tracked but not added to notes store)
                # Music21 creates implicit rests that don't exist in source.
                # Gold standard (MS3 TSV) and Partitura exclude these.
                if isinstance(obj, m21.note.GeneralNote):
                    if isinstance(obj, m21.note.Rest):
                        # Track that rests exist but don't add to notes store
                        has_rests = True
                        # Skip adding rest to notes store - matches gold standard
                    elif isinstance(obj, m21.note.Note):
                        note_rows.append(
                            self._make_note_row(
                                obj, qb, dur_qb, mc, mn, mc_onset, part_id
                            )
                        )
                        note_rows[-1][LOCATOR] = locate(place, _stated_id(obj))
                    elif isinstance(obj, m21.chord.Chord):
                        for n, note in enumerate(obj.notes):
                            note_rows.append(
                                self._make_note_row(
                                    note,
                                    qb,
                                    dur_qb,
                                    mc,
                                    mn,
                                    mc_onset,
                                    part_id,
                                    chord_id=str(id(obj)),
                                )
                            )
                            note_rows[-1][LOCATOR] = locate(
                                f"{place} note:{n}", _stated_id(note)
                            )

                # Controls
                elif isinstance(
                    obj,
                    (
                        m21.dynamics.Dynamic,
                        m21.tempo.TempoIndication,
                        m21.key.KeySignature,
                        TimeSignature,
                    ),
                ):
                    control_rows.append(
                        {
                            LOCATOR: locate(place, _stated_id(obj)),
                            "id": _stated_id(obj),
                            "name": obj.__class__.__name__,
                            "temporal_type": "instant",
                            "event_type": obj.__class__.__name__,
                            "quarterbeats": rational_to_struct(qb),
                            "duration_qb": None,
                            "mc": mc,
                            "mn": mn,
                            "mc_onset": (
                                rational_to_struct(mc_onset)
                                if mc_onset is not None
                                else None
                            ),
                            "mn_onset": None,
                            "subtype": obj.__class__.__name__,
                            "value": None,
                            "text": str(obj),
                            "voice": None,
                            "staff": None,
                            "part_id": part_id,
                        }
                    )

                # Annotations
                elif isinstance(obj, m21.expressions.TextExpression):
                    annotation_rows.append(
                        {
                            LOCATOR: locate(place, _stated_id(obj)),
                            "id": _stated_id(obj),
                            "name": getattr(obj, "content", str(obj)),
                            "temporal_type": "instant",
                            "event_type": "TextExpression",
                            "quarterbeats": rational_to_struct(qb),
                            "duration_qb": None,
                            "mc": mc,
                            "mn": mn,
                            "mc_onset": (
                                rational_to_struct(mc_onset)
                                if mc_onset is not None
                                else None
                            ),
                            "mn_onset": None,
                            "subtype": "TextExpression",
                            "text": getattr(obj, "content", str(obj)),
                            "staff": None,
                            "part_id": part_id,
                        }
                    )

        m21_offset = self._normalise_anacrusis(
            note_rows, measure_rows, control_rows, annotation_rows
        )
        self._anacrusis_offset = m21_offset
        for rows in (note_rows, measure_rows, control_rows, annotation_rows):
            self._stamp_rows(rows, source, [row[LOCATOR] for row in rows])

        # Build data
        notes_data = NoteEventData.from_dicts(
            note_rows,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            minter=self._id_minter,
            has_rests=has_rests,
        )

        measures_data = (
            MeasureData.from_dicts(
                measure_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                minter=self._id_minter,
            )
            if measure_rows
            else MeasureData.empty()
        )

        controls_data = (
            ControlEventData.from_dicts(
                control_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                minter=self._id_minter,
            )
            if control_rows
            else ControlEventData.empty()
        )

        annotations_data = (
            AnnotationEventData.from_dicts(
                annotation_rows,
                unit=TimeUnit.quarters,
                number_type=NumberType.fraction,
                minter=self._id_minter,
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
                "parser": "music21",
                **source_record(source, "source"),
                "has_rests": has_rests,
                "anacrusis_offset": m21_offset,
            },
        )

    def _make_note_row(
        self,
        obj: Any,
        qb: Fraction,
        dur_qb: Fraction,
        mc: int | None,
        mn: str | None,
        mc_onset: Fraction,
        part_id: str,
        is_rest: bool = False,
        chord_id: str | None = None,
    ) -> dict[str, Any]:
        """Create note row dict."""
        midi = None
        specific_pitch = None
        octave = None
        tpc = None

        _sp_label = ""
        if not is_rest and hasattr(obj, "pitch"):
            midi = int(obj.pitch.midi)

            step = obj.pitch.step
            alter = int(obj.pitch.alter or 0)
            octave = obj.pitch.octave or 4

            acc_str = ""
            if alter > 0:
                acc_str = "♯" * alter
            elif alter < 0:
                acc_str = "♭" * abs(alter)

            base_fifths = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
            tpc = base_fifths.get(step, 0) + (7 * alter)

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

        return {
            # ID auto-generated from event_type (Note or Rest)
            "name": note_name,
            "temporal_type": "interval" if dur_qb > 0 else "instant",
            "event_type": "Rest" if is_rest else "Note",
            "quarterbeats": rational_to_struct(qb),
            "duration_qb": rational_to_struct(dur_qb),
            "mc": mc,
            "mn": mn,
            "mc_onset": rational_to_struct(mc_onset),
            "mn_onset": None,
            "timesig": None,
            "duration": None,
            "nominal_duration": None,
            "scalar": None,
            "specific_pitch": specific_pitch,
            "midi": midi,
            "tpc": tpc,
            "octave": octave,
            "tied": None,
            "gracenote": None,
            "chord_id": chord_id,
            "voice": getattr(obj, "voice", None),
            "staff": getattr(obj, "staff", None),
            "part_id": part_id,
        }

    @staticmethod
    def _refuse_measures_without_length(
        measure_list: list[Any], source: Source
    ) -> None:
        """Raise if music21 gives a measure of *source* no length.

        music21 times a measure by its content: one without timed content
        (an empty layer, such as an MEI skeleton's) has length zero, and its
        MEI reader leaves out ``<ending>`` measures holding nothing but
        measure rests. Such a measure states neither its own length nor the
        position of any measure after it; measures of length zero would put
        flow control at false coordinates (two brackets on one).

        Raises:
            ValueError: Naming the measure numbers music21 gives no length.
        """
        unmeasured = [
            measure.number
            for measure in measure_list
            if measure.duration.quarterLength == 0
        ]
        if unmeasured:
            raise ValueError(
                f"music21 gives {len(unmeasured)} of the {len(measure_list)} measures it "
                f"reads from {source.name} no length (a measure without timed content "
                f"has none): measure numbers {unmeasured}. Neither their lengths nor the "
                "positions of the measures after them are stated, so no measure table "
                "is written"
            )

    @staticmethod
    def _parse_mei_measure_info(source: Source) -> dict[int, dict[str, Any]]:
        """Parse an MEI file directly to extract measure repeat facts.

        music21's MEI parser does not translate ``<ending>`` elements into
        ``RepeatBracket`` spanners. This method parses the raw XML to recover,
        per ``<measure>``:

        - ``number``: its ``@n`` as an integer, or ``None``.
        - ``volta``: ending number (1, 2, …) for measures inside ``<ending>``
          elements.
        - ``start_repeat``: True when the measure has ``left="rptstart"``.
        - ``end_repeat``: True when the measure has ``right="rptend"``.

        Args:
            source: The ``.mei`` file: a path or a ZIP archive member, read
                in place.

        Returns:
            Mapping from source measure position (from 1) to its facts.
        """
        try:
            with readable(source) as document:
                tree = ET.parse(document)
        except ET.ParseError:
            module_logger.warning(
                "Failed to parse MEI XML for %s", describe_source(source)
            )
            return {}

        root = tree.getroot()
        result: dict[int, dict[str, Any]] = {}

        def _walk(element: ET.Element, current_volta: int | None) -> None:
            tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            # Entering an <ending> element: parse its label as volta number
            if tag == "ending":
                label = element.get("label", "")
                try:
                    # Labels can be "1", "2", "1." etc.
                    current_volta = int(label.rstrip("."))
                except ValueError:
                    current_volta = None

            if tag == "measure":
                n_str = element.get("n")
                try:
                    number = int(n_str) if n_str is not None else None
                except ValueError:
                    number = None
                result[len(result) + 1] = {
                    "number": number,
                    "volta": current_volta,
                    "start_repeat": element.get("left", "") == "rptstart",
                    "end_repeat": element.get("right", "") == "rptend",
                }

            for child in element:
                _walk(child, current_volta)

        _walk(root, None)
        return result

    @staticmethod
    def _expand_mei_skeleton(
        measure_list: list[Any],
    ) -> list[Any]:
        """Expand a sparse MEI measure list by filling in implied intermediate measures.

        MuseScore's MEI export uses a compact representation: only boundary measures
        (first/last of a repeat section, volta measures, etc.) are written to the
        file. Intermediate empty measures are implied by gaps in the ``@n`` attribute
        (exposed as ``measure.number`` by music21). For example, if a file contains
        measures with ``n=2`` and ``n=9``, measures 3–8 are implied and must be
        synthesised so that every MC from 1 to the maximum ``@n`` value is present.

        This method only expands the list when the MEI is genuinely sparse, i.e.:

        - All measure numbers are unique (no duplicates from repeated sections).
        - The maximum measure number is substantially greater than the list length,
          indicating that intermediate measures have been omitted.

        When these conditions are not met (e.g., full-score MEI where music21
        assigns the same ``@n`` to different structural occurrences of a bar),
        the original list is returned unchanged.

        Synthetic intermediate measures are plain proxies that inherit the
        duration of the preceding skeleton measure and carry no flow-control
        markers (barlines, repeat signs, etc.).

        Args:
            measure_list: The list of ``music21.stream.Measure`` objects parsed
                from an MEI file.

        Returns:
            An expanded list of measures (real + synthetic) ordered by MC, or
            the original list when no expansion is needed.
        """
        if not measure_list:
            return measure_list

        # Collect measure numbers.
        numbers = [
            int(m.number) if m.number is not None else None for m in measure_list
        ]
        valid_numbers = [n for n in numbers if n is not None]

        if not valid_numbers:
            return measure_list

        # Guard: if there are duplicate measure numbers the file is a full-score
        # MEI (not a skeleton). Do not attempt expansion — deduplication would
        # discard valid measures.
        if len(set(valid_numbers)) < len(valid_numbers):
            return measure_list

        # Guard: if the maximum measure number equals the list length the file
        # is already complete (no gaps). Return as-is.
        max_n = max(valid_numbers)
        min_n = min(valid_numbers)
        full_range_size = max_n - min_n + 1
        if full_range_size == len(measure_list):
            return measure_list

        # The file is sparse: build a {measure_number -> measure} mapping.
        seen: dict[int, Any] = {n: m for n, m in zip(valid_numbers, measure_list)}
        sorted_ns = sorted(seen.keys())

        # For synthetic measures we create a lightweight proxy object that exposes
        # just the attributes the loader reads: number, offset, duration,
        # leftBarline, rightBarline.
        class _SyntheticMeasure:
            """Minimal stand-in for a music21 Measure with no flow-control markers."""

            def __init__(self, number: int, offset: float, duration_ql: float) -> None:
                self.number = number
                self.offset = offset
                self.leftBarline = None
                self.rightBarline = None
                self._duration_ql = duration_ql

            @property
            def duration(self) -> Any:
                """Stand-in for a music21 Duration exposing only ``quarterLength``."""

                class _Dur:
                    def __init__(self, ql: float) -> None:
                        self.quarterLength = ql

                return _Dur(self._duration_ql)

            def __iter__(self):  # type: ignore[override]
                return iter([])  # No child elements — no repeat expressions

        expanded: list[Any] = []
        for i, n in enumerate(sorted_ns):
            real_measure = seen[n]
            expanded.append(real_measure)

            # Fill the gap to the next skeleton measure
            if i < len(sorted_ns) - 1:
                next_n = sorted_ns[i + 1]
                gap = next_n - n - 1
                if gap > 0:
                    real_dur = float(real_measure.duration.quarterLength)
                    real_offset = float(real_measure.offset)
                    for j in range(1, gap + 1):
                        synthetic = _SyntheticMeasure(
                            number=n + j,
                            offset=real_offset + j * real_dur,
                            duration_ql=real_dur,
                        )
                        expanded.append(synthetic)

        return expanded


def _stated_id(obj: Any) -> str | None:
    """The ID the source gives *obj*, or None for one the loader mints.

    music21 keeps a stated ``xml:id`` / ``id`` as a string; an object the
    source gives none answers its memory address, an ``int`` that differs
    between runs and is no ID of the source.
    """
    stated = getattr(obj, "id", None)
    return stated if isinstance(stated, str) and stated else None
