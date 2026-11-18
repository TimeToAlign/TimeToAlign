"""VerovioLoader: read an engraved score through the Verovio toolkit."""

from __future__ import annotations

import bisect
import logging
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import pyarrow as pa
import verovio

from timetoalign.core import NumberType, TimeUnit
from timetoalign.core.enums import ExtrapolationPolicy
from timetoalign.core.time import build_number_struct_array, rational_to_struct

from .base import ScoreLoader
from .events import ScoreEventType
from .store import ScoreStore
from .stores import (
    AnnotationEventData,
    ControlEventData,
    MeasureData,
    NoteEventData,
)

if TYPE_CHECKING:
    from timetoalign.maps import TableMap
    from timetoalign.storage.events import EventData
    from timetoalign.timelines import (
        ContinuousLogicalTimeline,
        ContinuousPhysicalTimeline,
        Timeline,
    )

module_logger = logging.getLogger(__name__)

_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

# ``useFractions`` (exact ``qfrac`` onsets in the timemap) exists from 4.4 on.
_MIN_VERSION = (4, 4)

_TIMEMAP_OPTIONS = {"includeMeasures": True, "includeRests": True}

# Options that choose which ``<mdiv>``s Verovio loads; without one of them the
# loader asks for all.
_MDIV_OPTIONS = ("mdivAll", "mdivXPathQuery")

_STEP_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_BASE_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
# MEI ``data.ACCIDENTAL.WRITTEN`` / ``.GESTURAL`` values in whole semitones.
_ACCID_ALTER = {
    "n": 0,
    "s": 1,
    "f": -1,
    "ss": 2,
    "x": 2,
    "ff": -2,
    "xs": 3,
    "sx": 3,
    "ts": 3,
    "tf": -3,
    "ns": 1,
    "nf": -1,
}
_METER_SYMBOLS = {"common": ("4", "4"), "cut": ("2", "2")}

# Extra note columns, verbatim from the MEI.
_NOTE_EXTRA_TYPES: dict[str, pa.DataType] = {
    "notated_id": pa.string(),
    "dur": pa.string(),
    "dots": pa.int64(),
    "grace": pa.string(),
    "tie": pa.string(),
}
_TEMPO_EXTRA_TYPES: dict[str, pa.DataType] = {"notated_id": pa.string()}
_POSITION_COLUMNS = ("start", "end")
# In-measure offsets are quarters; only the quarters timeline carries them.
_MEASURE_OFFSETS = ("mc_onset",)
_STRUCT_COLUMNS = (*_POSITION_COLUMNS, *_MEASURE_OFFSETS)
# ``<repeatMark @func>`` values naming a place, and those naming a jump back.
_MARKER_FUNCS = ("segno", "coda", "fine")
_JUMP_FUNCS = {"dalSegno": "segno", "daCapo": "start"}


def _local(tag: str) -> str:
    """Return an element tag without its namespace."""
    return tag.rpartition("}")[2]


def _version_tuple(version: str) -> tuple[int, ...]:
    """Return ``(major, minor, patch)`` from ``"6.4.1-fbe8f46"``."""
    return tuple(int(part) for part in version.split("-")[0].split("."))


def _qfrac(entry: dict[str, Any]) -> Fraction:
    """Return the exact quarter-note position of a timemap entry."""
    numerator, denominator = entry["qfrac"]
    return Fraction(numerator, denominator)


def _pitch_label(step: str, alter: int, octave: int) -> str:
    """Spell a pitch with the canonical ``♯``/``♭`` characters."""
    accidental = "♯" * alter if alter > 0 else "♭" * -alter
    return f"{step}{accidental}{octave}"


@dataclass
class _Movement:
    """One ``<mdiv>`` of the MEI tree, in document order."""

    xml_id: str
    n: str | None
    label: str | None
    parent: int | None


@dataclass
class _Measure:
    """One ``<measure>`` of the MEI tree, in document order."""

    xml_id: str
    n: str | None
    left: str | None
    right: str | None
    volta: int | None
    meter: tuple[str, str] | None
    movement: int | None
    marks: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _Timed:
    """One ``<note>``, ``<rest>`` or ``<mRest>`` of the MEI tree."""

    xml_id: str
    kind: str
    measure: int
    staff: int | None
    layer: int | None
    dur: str | None
    dots: int | None
    pname: str | None = None
    oct: str | None = None
    accid: str | None = None
    accid_ges: str | None = None
    grace: str | None = None
    tie: str | None = None
    chord: str | None = None


@dataclass
class _Tempo:
    """One ``<tempo>`` control element."""

    xml_id: str
    measure: int
    text: str
    bpm: float | None
    staff: int | None
    startid: str | None
    tstamp: str | None


@dataclass
class _Score:
    """What the loader reads from Verovio's MEI tree (the notated score)."""

    movements: list[_Movement] = field(default_factory=list)
    measures: list[_Measure] = field(default_factory=list)
    timed: dict[str, _Timed] = field(default_factory=dict)
    tempos: list[_Tempo] = field(default_factory=list)
    ties: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    chord_notes: dict[str, list[str]] = field(default_factory=dict)
    ids: set[str] = field(default_factory=set)
    duplicate_ids: list[str] = field(default_factory=list)
    title: str | None = None


@dataclass
class _RenderedMeasure:
    """One measure of Verovio's timemap, in rendered order."""

    xml_id: str
    notated_id: str
    start: Fraction
    end: Fraction


@dataclass
class _Rendering:
    """What one load keeps for building the timelines.

    Event rows hold absolute positions in rendered quarters; ``movement``
    maps each event id to the innermost ``<mdiv>`` holding it.
    """

    anchors: list[tuple[Fraction, Fraction]]
    rows: list[dict[str, Any]]
    movement: dict[str, int]
    movements: list[_Movement]
    spans: dict[int, tuple[Fraction, Fraction]]
    measures: list[_RenderedMeasure]
    expanded: bool
    printed: list[tuple[_Measure, Fraction, Fraction]]
    end: Fraction


def _single_int(value: str | None, element: ET.Element) -> int | None:
    """Read an attribute that must hold one integer."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"Element {element.get(_XML_ID)!r} states {value!r} where one integer "
            "is expected"
        ) from None


def _volta(ending: ET.Element) -> int | None:
    """Return an ``<ending>``'s number from ``@n``, else from ``@label`` (``"2."``)."""
    for value in (ending.get("n"), ending.get("label")):
        if value is not None and value.rstrip(".").strip().isdigit():
            return int(value.rstrip("."))
    return None


def _check_unique_ids(root: ET.Element, score: _Score) -> None:
    """Collect every ``xml:id`` of the tree and refuse a duplicate join key.

    Verovio's id generator does not test uniqueness, and a source may
    repeat an id. A duplicate held by an element the loader reads (movement,
    measure, chord, note, rest, tempo, tie) would make a join key ambiguous
    and raises; one held only by other elements (sources exported from
    MuseScore repeat ids of ``<harm>`` and ``<dynam>``) is listed in
    ``duplicate_ids``.
    """
    counts = Counter(el.get(_XML_ID) for el in root.iter() if el.get(_XML_ID))
    score.ids = set(counts)
    duplicates = sorted(xml_id for xml_id, count in counts.items() if count > 1)
    read = {
        *(movement.xml_id for movement in score.movements),
        *(measure.xml_id for measure in score.measures),
        *score.chord_notes,
        *score.timed,
        *(tempo.xml_id for tempo in score.tempos),
        *(tie_id for tie_id, _, _ in score.ties),
    }
    ambiguous = [xml_id for xml_id in duplicates if xml_id in read]
    if ambiguous:
        raise ValueError(
            f"{len(ambiguous)} xml:id(s) name more than one element of the score "
            f"Verovio read: {ambiguous[:5]}"
        )
    score.duplicate_ids = duplicates


def _read_score(mei: str) -> _Score:
    """Collect movements, measures, notes, rests, ties, tempos and repeat marks.

    The tree supplies what Verovio has no query for: which movement,
    measure, staff, layer and chord an element belongs to, the ``<tie>``
    elements, the tempo text and each measure's ``<repeatMark>``s (``@func``
    and text). Positions are not computed here; they come from the timemap.
    """
    root = ET.fromstring(mei)
    score = _Score()
    for title in root.iter():
        if _local(title.tag) == "title":
            score.title = " ".join("".join(title.itertext()).split()) or None
            break

    meter: tuple[str, str] | None = None

    def read_meter(element: ET.Element) -> None:
        nonlocal meter
        count, unit = element.get("meter.count"), element.get("meter.unit")
        symbol = element.get("meter.sym")
        if _local(element.tag) == "meterSig":
            count, unit, symbol = (
                element.get("count"),
                element.get("unit"),
                element.get("sym"),
            )
        if count is not None and unit is not None:
            meter = (count, unit)
        elif symbol in _METER_SYMBOLS:
            meter = _METER_SYMBOLS[symbol]

    def walk(element: ET.Element, ctx: dict[str, Any]) -> None:
        tag = _local(element.tag)
        if tag == "mdiv":
            score.movements.append(
                _Movement(
                    xml_id=element.get(_XML_ID),
                    n=element.get("n"),
                    label=element.get("label"),
                    parent=ctx.get("movement"),
                )
            )
            ctx = {**ctx, "movement": len(score.movements) - 1}
        elif tag in ("scoreDef", "staffDef", "meterSig"):
            read_meter(element)
        elif tag == "ending":
            ctx = {**ctx, "volta": _volta(element)}
        elif tag == "measure":
            score.measures.append(
                _Measure(
                    xml_id=element.get(_XML_ID),
                    n=element.get("n"),
                    left=element.get("left"),
                    right=element.get("right"),
                    volta=ctx.get("volta"),
                    meter=meter,
                    movement=ctx.get("movement"),
                )
            )
            ctx = {**ctx, "measure": len(score.measures)}
        elif tag == "staff" and "measure" in ctx:
            ctx = {**ctx, "staff": _single_int(element.get("n"), element)}
        elif tag == "layer":
            ctx = {**ctx, "layer": _single_int(element.get("n"), element)}
        elif tag == "graceGrp":
            ctx = {**ctx, "grace": element.get("grace")}
        elif tag == "tie":
            score.ties.append(
                (
                    element.get(_XML_ID),
                    (element.get("startid") or "").lstrip("#") or None,
                    (element.get("endid") or "").lstrip("#") or None,
                )
            )
        elif tag == "repeatMark" and "measure" in ctx:
            score.measures[ctx["measure"] - 1].marks.append(
                (
                    element.get("func") or "",
                    " ".join("".join(element.itertext()).split()),
                )
            )
            return
        elif tag == "tempo" and "measure" in ctx:
            bpm = element.get("midi.bpm")
            staff = (element.get("staff") or "").split()
            score.tempos.append(
                _Tempo(
                    xml_id=element.get(_XML_ID),
                    measure=ctx["measure"],
                    text=" ".join("".join(element.itertext()).split()),
                    bpm=float(bpm) if bpm is not None else None,
                    staff=int(staff[0]) if staff else None,
                    startid=(element.get("startid") or "").lstrip("#") or None,
                    tstamp=element.get("tstamp"),
                )
            )
            return
        elif tag == "chord":
            ctx = {
                **ctx,
                "chord": element,
                "staff_override": element.get("staff") or ctx.get("staff_override"),
            }
            score.chord_notes[element.get(_XML_ID)] = [
                note.get(_XML_ID)
                for note in element.iter()
                if _local(note.tag) == "note"
            ]
        elif tag in ("note", "rest", "mRest") and "measure" in ctx:
            score.timed[element.get(_XML_ID)] = _timed(element, tag, ctx)
        for child in element:
            walk(child, ctx)

    walk(root, {})
    _check_unique_ids(root, score)
    return score


def _timed(element: ET.Element, tag: str, ctx: dict[str, Any]) -> _Timed:
    """Read one note or rest with what it inherits from its chord."""
    chord: ET.Element | None = ctx.get("chord") if tag == "note" else None
    staff_attr = element.get("staff") or ctx.get("staff_override")
    staff = _single_int(staff_attr, element) if staff_attr else ctx.get("staff")
    dur = element.get("dur") or (chord.get("dur") if chord is not None else None)
    dots = element.get("dots") or (chord.get("dots") if chord is not None else None)
    timed = _Timed(
        xml_id=element.get(_XML_ID),
        kind=tag,
        measure=ctx["measure"],
        staff=staff,
        layer=ctx.get("layer"),
        dur=dur,
        dots=_single_int(dots, element),
    )
    if tag == "note":
        timed.pname = element.get("pname")
        timed.oct = element.get("oct")
        timed.accid = element.get("accid")
        timed.accid_ges = element.get("accid.ges")
        for child in element:
            if _local(child.tag) == "accid":
                timed.accid = timed.accid or child.get("accid")
                timed.accid_ges = timed.accid_ges or child.get("accid.ges")
        timed.grace = (
            element.get("grace")
            or (chord.get("grace") if chord is not None else None)
            or ctx.get("grace")
        )
        timed.tie = element.get("tie") or (
            chord.get("tie") if chord is not None else None
        )
        timed.chord = chord.get(_XML_ID) if chord is not None else None
    return timed


class VerovioLoader(ScoreLoader):
    """Load an engraved score through the Verovio toolkit.

    Any format Verovio reads (MEI, MusicXML, Humdrum, ABC, PAE, …) yields
    one score as Verovio renders it: its timemap is taken verbatim, and
    nothing is computed from the notated rhythm.

    * :meth:`create_timeline` gives the logical timeline: one
      ``ContinuousLogicalTimeline`` in quarters, positions the timemap's exact
      ``qfrac`` (``useFractions``), with a quarters → seconds C-Map.
    * :meth:`create_physical_timeline` gives the same events on Verovio's
      rendered time: one ``ContinuousPhysicalTimeline`` in seconds, positions
      the timemap's integer milliseconds ``tstamp`` as
      ``Fraction(tstamp, 1000)``, with the inverse C-Map. It is time
      synthesised from the score's tempo, not a performance.

    The C-Map is one ``TableMap`` through every timemap entry
    (:meth:`create_quarters_to_seconds_map`). Verovio holds the tempo
    constant within a measure, so linear interpolation between the entries
    reproduces its own time.

    **Movements.** Every ``<mdiv>`` Verovio loads is a child timeline, in
    document order, its id minted by the loader after the root's
    (``score:clt_2``, ``rendered:cpt_2``, …; a number a given root id such
    as ``score:clt_2`` holds is skipped):
    placed at its first position, as long as its measures, named
    by its ``label``, with ``meta`` holding ``count`` (its place among all
    ``<mdiv>``s of the file, from 1) and its ``xml:id``, ``n`` and ``label``
    verbatim. A nested ``<mdiv>`` is a child of its parent's timeline.
    Events live in the innermost movement holding them, at parent − offset;
    the root holds the measures, the flow control and the C-Map. A
    single-movement score has its one child too. All movements are loaded
    unless an ``mdivAll`` or ``mdivXPathQuery`` option chooses.

    **Repeats.** The timemap is Verovio's rendering: with repeats expanded
    (Verovio's default, steered by the ``expand``, ``expandAlways`` and
    ``expandNever`` options), a repeated passage appears once per pass.
    Event ``id`` is the rendered id (``…-rend2`` from the second pass on);
    field ``notated_id`` is the element's id in the notated score (equal to
    ``id`` when nothing is expanded), read with ``getNotatedIdForElement``
    for ids the notated tree does not hold. The store's measure table is the
    printed one (one row per ``<measure>`` in document order, lengths as
    rendered, positions on the folded axis), carrying the repeat barlines,
    endings and ``<repeatMark>``s (segno, coda, Fine, D.S., D.C.), from which
    :meth:`create_flow_controller` derives the repeat structure. Without an
    expansion the timeline carries that table as its ``MeasureMap`` and its
    flow control. With an expansion the
    timeline is already unfolded: its ``MeasureMap`` lists the rendered
    measures (row id = rendered id, name = ``@n``) without flow control, and
    two FlowMaps state the playback order of the printed measures:
    ``"forward_verovio"`` (printed → rendered) and ``"source"`` (its inverse).

    **Ids.** Every event keeps its element's ``xml:id`` verbatim. Verovio
    generates ids only where the source has none, from a random seed unless
    told otherwise; the loader seeds them from the input's checksum
    (``xmlIdChecksum``), so a generated id is stable for one Verovio version
    and one input. A user ``xmlIdSeed`` wins and is logged. After loading,
    every ``xml:id`` of ``getMEI()`` is checked for uniqueness, because
    Verovio's generator does not test it: a duplicate held by an element
    the loader reads raises, any other is logged and listed in the store
    metadata under ``"duplicate_ids"``.

    Sources of each value:

    * **Positions** come from the timemap with ``useFractions``. The
      timemap's float ``qstamp`` is a cross-check: the float nearest to each
      exact position must equal it, else the load raises.
    * **Membership** (movement, measure, staff, layer), ``<tie>`` elements
      and tempo text come from the MEI tree that ``getMEI()`` returns, read
      with the standard library's XML parser.
    * **MIDI pitch** comes from ``getMIDIValuesForElement``.

    Events: one interval event per note (``event_type`` ``"Note"``) and per
    rest (``"Rest"``) from onset to offset, and one instant event per
    ``<tempo>`` and pass (``"tempo indication"``, its ``text`` verbatim).
    ``mc``, ``mn`` name the printed measure; ``mc_onset`` is the position in
    the rendered pass of it, in quarters, on the logical timeline only.
    ``voice`` is the ``<layer>`` number, ``chord_id`` the ``xml:id`` of the
    note's ``<chord>``. A
    tied note's continuation is an event of its own; field ``tie`` (``i``,
    ``m``, ``t``, verbatim from ``@tie`` or from the ``<tie>`` elements
    naming the note) tells attacks from continuations. Field ``grace`` is
    the ``@grace`` of the note, its chord or its ``<graceGrp>``, verbatim.
    ``<space>`` elements are not events. The note ``staff`` is the staff
    the note is drawn on: a cross-staff chord's ``@staff``, else its
    ``<staff>``.

    Known issues:
        Grace notes and arpeggiated notes sit where Verovio performs them,
        not at their notated onsets. An unaccented grace note takes a short
        fixed span before its principal note (or delays the principal note
        at a measure start), an accented one takes a share of the principal
        note and delays it, and the notes of an arpeggio are rolled one
        after the other. The controls are MEI attributes (``@grace``,
        ``@grace.time``, ``graceGrp/@attach``, ``arpeg/@order``); the
        loader does not undo them.
        A ``<tie>`` whose ``endid`` names no element of the file (a tie
        leaving an excerpt) is logged and listed in the store metadata under
        ``"dangling_ties"``; its start note keeps ``tie = "i"`` and nothing
        is invented for the missing end. Verovio prints its own warning to
        the process's standard error; ``getLog()`` stays empty.
        A measure without content has no length in Verovio's timemap and no
        entry of its own; such a score raises, naming the measures.
        Verovio 6.4.1's expansion leaves out the measures of ``<ending>``s
        it cannot place (both endings, and no repeat of their passage, after
        another repeat). Their lengths are read from a second rendering
        with ``expandNever``; their events are not in the rendered
        timelines. They are logged and listed in the store metadata under
        ``"omitted_measures"``.
        Verovio's expansion does not follow D.S. and D.C. marks; the
        printed measure table states them.
        ``expandAlways`` writes the expansion into the MEI tree, which then
        no longer states the printed score; it raises.
        Verovio 6.4.1 crashes the process in ``renderToTimemap`` when it
        expands the repeats of some files; ``expandNever`` reads them.

    Examples:
        >>> loader = VerovioLoader.from_file("score.mei")
        >>> timeline = loader.create_timeline()
        >>> seconds = loader.create_physical_timeline()
    """

    #: One loader reads one score: its timemap and C-Map are the score's.
    _single_source: ClassVar[bool] = True

    def __init__(self, *, options: dict[str, Any] | None = None, **kwargs: Any) -> None:
        """Initialize the loader.

        Args:
            options: Verovio toolkit options, passed to ``setOptions`` before
                the file is loaded and recorded in the store metadata. Those
                that change the timemap: ``expand*``, ``mdiv*``,
                ``appXPathQuery``, ``choiceXPathQuery``, ``substXPathQuery``,
                ``midiNoCue``, ``midiTempoAdjustment``, ``transpose*``.
            **kwargs: Forwarded to `ScoreLoader`.
        """
        super().__init__(**kwargs)
        self._options = dict(options or {})
        self._rendering: _Rendering | None = None

    def _toolkit_options(self) -> dict[str, Any]:
        """Return the user's options with the loader's own defaults."""
        options = dict(self._options)
        if "xmlIdSeed" in options:
            module_logger.info(
                "VerovioLoader: xmlIdSeed=%r given; generated ids follow that seed",
                options["xmlIdSeed"],
            )
        else:
            options.setdefault("xmlIdChecksum", True)
        if not any(name in options for name in _MDIV_OPTIONS):
            options["mdivAll"] = True
        return options

    def _load_source(self, source: Path) -> ScoreStore:
        """Load one file into a ScoreStore through a fresh toolkit."""
        toolkit = verovio.toolkit()
        version = toolkit.getVersion()
        if _version_tuple(version)[:2] < _MIN_VERSION:
            raise ImportError(
                f"VerovioLoader requires verovio>={'.'.join(map(str, _MIN_VERSION))} "
                f"(exact timemap positions); found {version}"
            )
        options = self._toolkit_options()
        if options.get("expandAlways"):
            raise ValueError(
                "expandAlways writes Verovio's expansion into the MEI tree, which then "
                "no longer states the printed score; the default expands the timemap "
                "only"
            )
        _load_file(toolkit, source, options)
        score = _read_score(toolkit.getMEI())
        if score.duplicate_ids:
            module_logger.warning(
                "%d xml:id(s) in %s name more than one element the loader does not "
                "read: %s",
                len(score.duplicate_ids),
                source,
                score.duplicate_ids[:5],
            )
        timemap = _timemap(toolkit, source)

        notated = _notated_ids(timemap, score, toolkit)
        rendered = _rendered_measures(timemap, notated)
        score = _loaded_movements(score, {m.notated_id for m in rendered})
        lengths = _measure_lengths(rendered)
        omitted = [
            m.xml_id for m in score.measures if notated and m.xml_id not in lengths
        ]
        if omitted:
            lengths = _unexpanded_lengths(source, options, lengths, omitted)
        printed = _printed_measures(rendered, score, lengths, expanded=bool(notated))
        omitted_mcs = {
            mc
            for mc, measure in enumerate(score.measures, start=1)
            if measure.xml_id in omitted
        }
        onsets, offsets = _positions(timemap, score, notated, omitted_mcs)

        tie_marks, dangling = _tie_marks(score)
        for tie_id in dangling:
            module_logger.warning(
                "Tie %r in %s names no end element; its start keeps tie='i'",
                tie_id,
                source,
            )

        passes = _Passes(rendered, score)
        note_rows = [
            _timed_row(
                xml_id,
                notated.get(xml_id, xml_id),
                onsets,
                offsets,
                passes,
                tie_marks,
                toolkit,
            )
            for xml_id in onsets
        ]
        note_rows.sort(key=lambda row: row["start"])
        tempo_rows = [
            row
            for tempo in score.tempos
            if tempo.measure not in omitted_mcs
            for row in _tempo_rows(tempo, onsets, passes, toolkit, bool(notated))
        ]
        movement = {
            row["id"]: score.measures[row["mc"] - 1].movement
            for row in (*note_rows, *tempo_rows)
        }
        end = _qfrac(timemap[-1])
        self._rendering = _Rendering(
            anchors=[(_qfrac(e), Fraction(e["tstamp"], 1000)) for e in timemap],
            rows=note_rows + tempo_rows,
            movement=movement,
            movements=score.movements,
            spans=_movement_spans(rendered, score, end),
            measures=rendered,
            expanded=bool(notated),
            printed=printed,
            end=end,
        )

        has_rests = any(row["event_type"] == ScoreEventType.REST for row in note_rows)
        notes = NoteEventData(
            _event_data(NoteEventData, note_rows, _NOTE_EXTRA_TYPES).table,
            TimeUnit.quarters,
            NumberType.fraction,
            has_rests=has_rests,
        )
        controls = (
            _event_data(ControlEventData, tempo_rows, _TEMPO_EXTRA_TYPES)
            if tempo_rows
            else ControlEventData.empty()
        )
        measures = MeasureData.from_dicts(
            [
                {
                    **_measure_row(mc, measure.xml_id, measure, start, length),
                    **_repeat_columns(measure),
                    **navigation,
                }
                for mc, ((measure, start, length), navigation) in enumerate(
                    zip(printed, _navigation(score.measures)), start=1
                )
            ],
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
        )
        return ScoreStore(
            notes=notes,
            measures=measures,
            controls=controls,
            annotations=AnnotationEventData.empty(),
            metadata={
                "format": "score",
                "parser": "verovio",
                "source": str(source),
                "verovio_version": version,
                "verovio_options": options,
                "title": score.title,
                "has_rests": has_rests,
                "dangling_ties": dangling,
                "duplicate_ids": score.duplicate_ids,
                "omitted_measures": omitted,
            },
        )

    def clear(self) -> None:
        """Clear the loaded score."""
        super().clear()
        self._rendering = None

    def _require_rendering(self) -> _Rendering:
        """Return the loaded rendering, or raise if nothing is loaded."""
        rendering = getattr(self, "_rendering", None)
        if rendering is None:
            raise ValueError("No score loaded; call load() first")
        return rendering

    def create_quarters_to_seconds_map(self) -> TableMap:
        """Create the C-Map from Verovio's quarters to its rendered seconds.

        One linear ``TableMap`` with an anchor ``(qfrac, tstamp / 1000)`` at
        every timemap entry, undefined outside the score. Its results are
        written as ``Fraction`` seconds, and so are its inverse's quarters:
        the map is exact both ways.

        Returns:
            The quarters → seconds ``TableMap``.

        Raises:
            ValueError: If no score has been loaded.
        """
        from timetoalign.maps import TableMap

        anchors = self._require_rendering().anchors
        return TableMap(
            x_values=[quarters for quarters, _ in anchors],
            y_values=[seconds for _, seconds in anchors],
            extrapolate=ExtrapolationPolicy.error,
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
            output_number_type=NumberType.fraction,
        )

    def create_timeline(  # type: ignore[override]
        self,
        uid: str | None = None,
        name: str | None = None,
        attach_cmap: bool = True,
    ) -> ContinuousLogicalTimeline:
        """Create the score's logical timeline in quarters.

        The root carries the measures as its ``MeasureMap``, the flow
        control (or, for an expanded rendering, the FlowMaps) and the C-Map;
        each loaded ``<mdiv>`` is a child holding its events.

        Args:
            uid: The timeline id. ``None`` mints ``score:clt_N``.
            name: The timeline's display name.
            attach_cmap: If True, attach the quarters → seconds C-Map.

        Returns:
            A ``ContinuousLogicalTimeline`` in quarters, number type
            ``Fraction``, as long as the timemap.

        Raises:
            ValueError: If no score has been loaded.
        """
        from timetoalign.timelines import ContinuousLogicalTimeline

        rendering = self._require_rendering()
        timeline = self._build_timeline(
            ContinuousLogicalTimeline,
            lambda position: position,
            uid=uid,
            name=name,
            role="score",
            meta=None,
        )
        if rendering.expanded:
            self._attach_rendered_structure(timeline, rendering)
        else:
            self._populate_timeline_structure(timeline)
        if attach_cmap:
            timeline.add_conversion_map(self.create_quarters_to_seconds_map())
        return timeline

    def create_physical_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
        attach_cmap: bool = True,
    ) -> ContinuousPhysicalTimeline:
        """Create the score's timeline in Verovio's rendered seconds.

        The same events, movements and ids as :meth:`create_timeline`, each
        position the timemap's ``tstamp`` as ``Fraction(tstamp, 1000)``
        seconds (a position between entries, such as a tempo indication's,
        through the C-Map). ``meta["provenance"]`` is
        ``"verovio_rendered"``. Measures stay on the logical timeline: a
        quarter-based ``MeasureMap`` has no place on a seconds axis.

        Args:
            uid: The timeline id. ``None`` mints ``rendered:cpt_N``.
            name: The timeline's display name.
            attach_cmap: If True, attach the seconds → quarters C-Map.

        Returns:
            A ``ContinuousPhysicalTimeline`` in seconds, number type
            ``Fraction``.

        Raises:
            ValueError: If no score has been loaded, or if two timemap
                entries share one ``tstamp`` (less than 1 ms apart), which
                leaves their order in seconds unstated.
        """
        from timetoalign.timelines import ContinuousPhysicalTimeline

        rendering = self._require_rendering()
        coinciding = [
            (str(q0), str(q1), str(s1))
            for (q0, s0), (q1, s1) in zip(rendering.anchors, rendering.anchors[1:])
            if s1 <= s0
        ]
        if coinciding:
            raise ValueError(
                "Verovio's timemap places entries at one tstamp, so their order "
                f"in seconds is not stated: (quarters, quarters, seconds) "
                f"{coinciding[:5]}"
            )
        cmap = self.create_quarters_to_seconds_map()
        timeline = self._build_timeline(
            ContinuousPhysicalTimeline,
            lambda position: cmap(position),
            uid=uid,
            name=name,
            role="rendered",
            meta={"provenance": "verovio_rendered"},
        )
        if attach_cmap:
            timeline.add_conversion_map(cmap.inverse())
        return timeline

    def _build_timeline(
        self,
        timeline_cls: type[Timeline],
        convert: Callable[[Fraction], Fraction],
        *,
        uid: str | None,
        name: str | None,
        role: str,
        meta: dict[str, Any] | None,
    ) -> Any:
        """Build the root and one child per movement, events in the leaves.

        Args:
            timeline_cls: The timeline type of the root and its children.
            convert: Maps a position in quarters to the timeline's axis.
            uid: The root's id, or ``None`` to mint ``{role}:{prefix}_N``.
            name: The root's display name.
            role: The role of the minted ids.
            meta: The root's metadata.

        Every timeline of the tree is minted ``{role}:{prefix}_N`` by this
        loader, the root first: its number is drawn even when *uid* names
        it. A given *uid* also takes its own number from the loader's
        minted ids, so no movement is minted the root's id (``score:clt_2``
        given, the movements are ``score:clt_3``, …).
        """
        from timetoalign.timelines.factory import (
            _flag_minted,
            _mint_timeline_id,
            _root_timeline_id,
        )

        rendering = self._require_rendering()
        unit = TimeUnit.quarters if role == "score" else TimeUnit.seconds
        uid, minted = _root_timeline_id(self, unit, uid, role=role)
        by_movement: dict[int | None, list[dict[str, Any]]] = {}
        for row in rendering.rows:
            by_movement.setdefault(rendering.movement[row["id"]], []).append(row)

        def build(
            movement: int | None, first: Fraction, last: Fraction, **kwargs: Any
        ) -> Timeline:
            """Create the timeline of *movement* spanning [first, last)."""
            return _timeline(
                timeline_cls,
                by_movement.get(movement, []),
                unit,
                convert,
                first,
                last - first,
                **kwargs,
            )

        start = convert(Fraction(0))
        root = build(None, start, convert(rendering.end), uid=uid, name=name, meta=meta)
        _flag_minted(root, minted)
        nodes: dict[int | None, tuple[Timeline, Fraction]] = {None: (root, start)}
        for index, movement in enumerate(rendering.movements):
            if index not in rendering.spans:
                continue
            parent, parent_start = nodes[movement.parent]
            span_start, span_end = (convert(p) for p in rendering.spans[index])
            child = build(
                index,
                span_start,
                span_end,
                uid=_mint_timeline_id(self, unit, role=role),
                name=movement.label,
                meta={
                    key: value
                    for key, value in (
                        ("count", index + 1),
                        ("xml_id", movement.xml_id),
                        ("n", movement.n),
                        ("label", movement.label),
                    )
                    if value is not None
                },
            )
            _flag_minted(child, True)
            nodes[index] = (child, span_start)
            parent.add_child(child, offset=span_start - parent_start)
        return root

    def _attach_rendered_structure(
        self, timeline: Timeline, rendering: _Rendering
    ) -> None:
        """Attach the rendered measures and the playback-order FlowMaps."""
        from timetoalign.timelines import MeasureMap
        from timetoalign.timelines.flow import FlowMap

        printed_index = {
            measure.xml_id: index
            for index, (measure, _, _) in enumerate(rendering.printed)
        }
        rows = [
            _measure_row(
                count,
                measure.xml_id,
                rendering.printed[printed_index[measure.notated_id]][0],
                measure.start,
                measure.end - measure.start,
            )
            for count, measure in enumerate(rendering.measures, start=1)
        ]
        timeline.add_measure_map(
            MeasureMap.from_measure_data(
                MeasureData.from_dicts(
                    rows, unit=TimeUnit.quarters, number_type=NumberType.fraction
                )
            )
        )
        spans: list[tuple[Fraction, Fraction]] = []
        for measure in rendering.measures:
            _, start, length = rendering.printed[printed_index[measure.notated_id]]
            if spans and spans[-1][1] == start:
                spans[-1] = (spans[-1][0], start + length)
            else:
                spans.append((start, start + length))
        _, last_start, last_length = rendering.printed[-1]
        printed_end = last_start + last_length
        forward = FlowMap(
            spans,
            id="verovio",
            source_length=printed_end,
            target_length=rendering.end,
        )
        timeline.add_flow_map(forward, id="forward_verovio")
        timeline.add_flow_map(forward.inverse(), id="source")


def _timeline(
    timeline_cls: type[Timeline],
    rows: list[dict[str, Any]],
    unit: TimeUnit,
    convert: Callable[[Fraction], Fraction],
    origin: Fraction,
    length: Fraction,
    **kwargs: Any,
) -> Timeline:
    """Create a timeline of *length* holding *rows*, positions relative to *origin*."""
    notes = [row for row in rows if row["temporal_type"] == "interval"]
    tempos = [row for row in rows if row["temporal_type"] == "instant"]
    parts: list[EventData] = []
    if notes:
        parts.append(
            _event_data(NoteEventData, notes, _NOTE_EXTRA_TYPES, unit, convert, origin)
        )
    if tempos:
        parts.append(
            _event_data(
                ControlEventData, tempos, _TEMPO_EXTRA_TYPES, unit, convert, origin
            )
        )
    if not parts:
        return timeline_cls(
            length=length, unit=unit, number_type=NumberType.fraction, **kwargs
        )
    timeline = timeline_cls.from_event_data(parts[0].concat(*parts[1:]), **kwargs)
    timeline.length = length
    return timeline


def _load_file(toolkit: Any, source: Path, options: dict[str, Any]) -> None:
    """Set the toolkit's options and load *source*, or raise with Verovio's log."""
    toolkit.setOptions(options)
    if not toolkit.loadFile(str(source)):
        raise ValueError(f"Verovio could not load {source}: {toolkit.getLog()}")


def _timemap(toolkit: Any, source: Path) -> list[dict[str, Any]]:
    """Return the toolkit's exact timemap, cross-checked against the float one."""
    timemap = toolkit.renderToTimemap({**_TIMEMAP_OPTIONS, "useFractions": True})
    if not timemap:
        raise ValueError(f"Verovio's timemap of {source} is empty: {toolkit.getLog()}")
    _check_against_qstamps(timemap, toolkit.renderToTimemap(_TIMEMAP_OPTIONS))
    return timemap


def _check_against_qstamps(
    exact: list[dict[str, Any]], rounded: list[dict[str, Any]]
) -> None:
    """Raise unless each exact position rounds to the printed ``qstamp``.

    Verovio prints ``qstamp`` as a double, so the double nearest to the
    exact rational (``float(Fraction)``, correctly rounded) must equal it.
    """
    if len(exact) != len(rounded):
        raise ValueError("Verovio's two timemaps have different lengths")
    for entry, check in zip(exact, rounded):
        position = _qfrac(entry)
        if float(position) != check["qstamp"]:
            raise ValueError(
                f"Timemap position {position} does not round to qstamp {check['qstamp']}"
            )


_ID_KEYS = ("on", "restsOn", "off", "restsOff")


def _notated_ids(
    timemap: list[dict[str, Any]], score: _Score, toolkit: Any
) -> dict[str, str]:
    """Return the notated id of every rendered id the notated tree lacks.

    Only an expansion creates such ids (``…-rend2``), so an empty result
    states that nothing is expanded and no toolkit query is made.
    """
    rendered = {
        xml_id
        for entry in timemap
        for xml_id in (
            *entry.get("on", ()),
            *entry.get("restsOn", ()),
            *([entry["measureOn"]] if "measureOn" in entry else ()),
        )
        if xml_id not in score.ids
    }
    notated = {xml_id: toolkit.getNotatedIdForElement(xml_id) for xml_id in rendered}
    unknown = sorted(r for r, n in notated.items() if n not in score.ids)
    if unknown:
        raise ValueError(
            f"Verovio's timemap names elements of no notated element: {unknown[:5]}"
        )
    return notated


def _rendered_measures(
    timemap: list[dict[str, Any]], notated: dict[str, str]
) -> list[_RenderedMeasure]:
    """Return the timemap's measures in rendered order with their spans."""
    starts = [
        (entry["measureOn"], _qfrac(entry)) for entry in timemap if "measureOn" in entry
    ]
    ends = [start for _, start in starts[1:]] + [_qfrac(timemap[-1])]
    return [
        _RenderedMeasure(xml_id, notated.get(xml_id, xml_id), start, end)
        for (xml_id, start), end in zip(starts, ends)
    ]


def _loaded_movements(score: _Score, rendered: set[str]) -> _Score:
    """Keep the measures, notes, rests and tempos of the movements Verovio loaded.

    An ``mdiv`` option may leave movements out of the timemap; the tree
    still holds them. A movement counts as loaded when one of its measures,
    or of a movement nested in it, is rendered.
    """
    loaded: set[int | None] = {None}
    for measure in score.measures:
        movement = measure.movement if measure.xml_id in rendered else None
        while movement is not None and movement not in loaded:
            loaded.add(movement)
            movement = score.movements[movement].parent
    kept = [
        (mc, measure)
        for mc, measure in enumerate(score.measures, start=1)
        if measure.movement in loaded
    ]
    if len(kept) == len(score.measures):
        return score
    renumber = {old: new for new, (old, _) in enumerate(kept, start=1)}
    score.measures = [measure for _, measure in kept]
    score.timed = {
        xml_id: replace(timed, measure=renumber[timed.measure])
        for xml_id, timed in score.timed.items()
        if timed.measure in renumber
    }
    score.tempos = [
        replace(tempo, measure=renumber[tempo.measure])
        for tempo in score.tempos
        if tempo.measure in renumber
    ]
    return score


def _measure_lengths(rendered: list[_RenderedMeasure]) -> dict[str, Fraction]:
    """Return the length of every rendered printed measure, equal in every pass."""
    lengths: dict[str, Fraction] = {}
    for measure in rendered:
        length = measure.end - measure.start
        if lengths.setdefault(measure.notated_id, length) != length:
            raise ValueError(
                f"Measure {measure.notated_id!r} has rendered lengths "
                f"{lengths[measure.notated_id]} and {length}"
            )
    return lengths


def _unexpanded_lengths(
    source: Path,
    options: dict[str, Any],
    lengths: dict[str, Fraction],
    omitted: list[str],
) -> dict[str, Fraction]:
    """Return the printed lengths of a rendering with ``expandNever``.

    Verovio's expansion can leave printed measures out; a second rendering
    of the notated score states their lengths. Every measure both renderings
    hold must have the same length in each.
    """
    module_logger.warning(
        "Verovio's expansion of %s leaves out %d printed measure(s) %s; their "
        "lengths come from a rendering with expandNever",
        source,
        len(omitted),
        omitted[:5],
    )
    toolkit = verovio.toolkit()
    _load_file(toolkit, source, {**options, "expandNever": True})
    unexpanded = _measure_lengths(_rendered_measures(_timemap(toolkit, source), {}))
    differing = sorted(
        xml_id
        for xml_id, length in lengths.items()
        if unexpanded.get(xml_id, length) != length
    )
    if differing:
        raise ValueError(
            f"Measures {differing[:5]} have other lengths in Verovio's expansion "
            "than in its rendering with expandNever"
        )
    return {**unexpanded, **lengths}


def _printed_measures(
    rendered: list[_RenderedMeasure],
    score: _Score,
    lengths: dict[str, Fraction],
    *,
    expanded: bool,
) -> list[tuple[_Measure, Fraction, Fraction]]:
    """Return ``(measure, start, length)`` of every printed measure, folded.

    The folded positions follow the printed order. Without an expansion the
    rendered order is the printed one and the positions coincide.
    """
    missing = [m.xml_id for m in score.measures if m.xml_id not in lengths]
    order = [m.xml_id for m in score.measures]
    if missing or (not expanded and [m.notated_id for m in rendered] != order):
        raise ValueError(
            "The timemap's measures differ from the MEI tree's: "
            f"{len(lengths)} of {len(order)} measures have an entry; without one: "
            f"{missing[:5]} (Verovio gives a measure without content no length)"
        )
    printed: list[tuple[_Measure, Fraction, Fraction]] = []
    position = rendered[0].start
    for measure in score.measures:
        printed.append((measure, position, lengths[measure.xml_id]))
        position += lengths[measure.xml_id]
    return printed


def _movement_spans(
    rendered: list[_RenderedMeasure], score: _Score, end: Fraction
) -> dict[int, tuple[Fraction, Fraction]]:
    """Return the rendered span of every loaded ``<mdiv>``, nesting included."""
    index = {measure.xml_id: measure for measure in score.measures}
    spans: dict[int, tuple[Fraction, Fraction]] = {}
    for measure in rendered:
        movement = index[measure.notated_id].movement
        while movement is not None:
            start, stop = spans.get(movement, (measure.start, measure.end))
            spans[movement] = (min(start, measure.start), max(stop, measure.end))
            movement = score.movements[movement].parent
    if rendered:
        # The last measure ends where the timemap does.
        last = index[rendered[-1].notated_id].movement
        while last is not None:
            spans[last] = (spans[last][0], max(spans[last][1], end))
            last = score.movements[last].parent
    return spans


def _positions(
    timemap: list[dict[str, Any]],
    score: _Score,
    notated: dict[str, str],
    omitted: set[int],
) -> tuple[dict[str, Fraction], dict[str, Fraction]]:
    """Return the exact onset and offset of every rendered note and rest.

    Notes and rests of the printed measures (MCs) in *omitted*, which the
    rendering leaves out, have none.
    """
    onsets: dict[str, Fraction] = {}
    offsets: dict[str, Fraction] = {}
    for entry in timemap:
        position = _qfrac(entry)
        for key, target in zip(_ID_KEYS, (onsets, onsets, offsets, offsets)):
            for xml_id in entry.get(key, ()):
                target[xml_id] = position
    rendered = {notated.get(xml_id, xml_id) for xml_id in onsets}
    unknown = sorted(rendered - set(score.timed))
    missing = sorted(
        xml_id
        for xml_id, timed in score.timed.items()
        if xml_id not in rendered and timed.measure not in omitted
    )
    unended = sorted(set(onsets) - set(offsets))
    for problem, ids in (
        ("names elements that are no note or rest of the tree", unknown),
        ("has no onset for", missing),
        ("has no offset for", unended),
    ):
        if ids:
            raise ValueError(f"Verovio's timemap {problem}: {ids[:5]}")
    return onsets, offsets


class _Passes:
    """Find the rendered pass of a printed measure that holds a position."""

    def __init__(self, rendered: list[_RenderedMeasure], score: _Score) -> None:
        self.score = score
        self._mc = {measure.xml_id: mc for mc, measure in enumerate(score.measures, 1)}
        self.starts: dict[int, list[Fraction]] = {}
        for measure in rendered:
            self.starts.setdefault(self._mc[measure.notated_id], []).append(
                measure.start
            )

    def context(self, position: Fraction, mc: int) -> dict[str, Any]:
        """Return the measure fields of an event at *position* in printed *mc*.

        The pass is the last one of the measure starting at or before the
        position (the first when an event precedes them all).
        """
        starts = self.starts[mc]
        start = starts[max(bisect.bisect_right(starts, position) - 1, 0)]
        return {
            "mc": mc,
            "mn": self.score.measures[mc - 1].n,
            "mc_onset": position - start,
        }


def _tie_marks(score: _Score) -> tuple[dict[str, str], list[str]]:
    """Return the ``i``/``m``/``t`` mark of every tied note and dangling ties."""
    starts: set[str] = set()
    ends: set[str] = set()
    dangling: list[str] = []
    for tie_id, start, end in score.ties:
        if end is None or end not in score.ids:
            dangling.append(tie_id)
        for target, xml_id in ((starts, start), (ends, end)):
            if xml_id is not None:
                target.update(score.chord_notes.get(xml_id, [xml_id]))
    marks = {xml_id: "i" for xml_id in starts}
    for xml_id in ends:
        marks[xml_id] = "m" if xml_id in starts else "t"
    return marks, dangling


def _timed_row(
    xml_id: str,
    notated_id: str,
    onsets: dict[str, Fraction],
    offsets: dict[str, Fraction],
    passes: _Passes,
    tie_marks: dict[str, str],
    toolkit: Any,
) -> dict[str, Any]:
    """Build the event row of one rendered note or rest."""
    timed = passes.score.timed[notated_id]
    start = onsets[xml_id]
    row: dict[str, Any] = {
        "id": xml_id,
        "name": None,
        "temporal_type": "interval",
        "event_type": (
            ScoreEventType.REST if timed.kind != "note" else ScoreEventType.NOTE
        ),
        "start": start,
        "end": offsets[xml_id],
        **passes.context(start, timed.measure),
        "specific_pitch": None,
        "midi": None,
        "tpc": None,
        "octave": None,
        "chord_id": timed.chord,
        "voice": timed.layer,
        "staff": timed.staff,
        "notated_id": notated_id,
        "dur": timed.dur,
        "dots": timed.dots,
        "grace": None,
        "tie": None,
    }
    if timed.kind == "note":
        step = (timed.pname or "").upper()
        if step not in _STEP_SEMITONES or timed.oct is None:
            raise ValueError(f"Note {notated_id!r} states no pitch name and octave")
        octave = int(timed.oct)
        midi = int(toolkit.getMIDIValuesForElement(xml_id)["pitch"])
        alter = _alter(timed, step, octave, midi)
        row.update(
            name=_pitch_label(step, alter, octave),
            specific_pitch={
                "step": step,
                "alter": alter,
                "octave": octave,
                "cents": 0.0,
            },
            midi=midi,
            tpc=_BASE_FIFTHS[step] + 7 * alter,
            octave=octave,
            grace=timed.grace,
            tie=timed.tie or tie_marks.get(notated_id),
        )
    return row


def _alter(timed: _Timed, step: str, octave: int, midi: int) -> int:
    """Return a note's alteration in semitones.

    The gestural accidental wins over the written one. A note stating
    neither takes its alteration from the key signature or an earlier
    accidental, which Verovio has applied to its MIDI pitch.
    """
    accid = timed.accid_ges or timed.accid
    if accid is not None:
        if accid not in _ACCID_ALTER:
            raise ValueError(f"Note {timed.xml_id!r}: unsupported accidental {accid!r}")
        return _ACCID_ALTER[accid]
    alter = midi - 12 * (octave + 1) - _STEP_SEMITONES[step]
    if not -2 <= alter <= 2:
        raise ValueError(
            f"Note {timed.xml_id!r}: MIDI pitch {midi} is no alteration of {step}{octave}"
        )
    return alter


def _tempo_rows(
    tempo: _Tempo,
    onsets: dict[str, Fraction],
    passes: _Passes,
    toolkit: Any,
    expanded: bool,
) -> list[dict[str, Any]]:
    """Build one instant event per rendered pass of a tempo indication.

    Its place in the measure is its ``@startid`` element's onset (a chord's
    first note), else its ``@tstamp`` in beats of the measure's meter unit,
    else the measure's start; it is the same in every pass. The k-th pass
    gets the k-th of Verovio's expansion ids.
    """
    starts = passes.starts[tempo.measure]
    if tempo.startid is not None:
        startid = passes.score.chord_notes.get(tempo.startid, [tempo.startid])[0]
        if startid not in onsets:
            raise ValueError(
                f"Tempo {tempo.xml_id!r} starts at {tempo.startid!r}, which has no onset"
            )
        offset = onsets[startid] - starts[0]
    elif tempo.tstamp is not None:
        meter = passes.score.measures[tempo.measure - 1].meter
        if meter is None:
            raise ValueError(
                f"Tempo {tempo.xml_id!r} has a tstamp but no meter is in force"
            )
        offset = (Fraction(tempo.tstamp) - 1) * Fraction(4, int(meter[1]))
    else:
        offset = Fraction(0)
    ids = (
        list(toolkit.getExpansionIdsForElement(tempo.xml_id)) if expanded else []
    ) or [tempo.xml_id]
    if len(ids) != len(starts):
        raise ValueError(
            f"Tempo {tempo.xml_id!r} has {len(ids)} rendered ids for "
            f"{len(starts)} passes of its measure"
        )
    return [
        {
            "id": xml_id,
            "name": tempo.text,
            "temporal_type": "instant",
            "event_type": ScoreEventType.TEMPO_INDICATION,
            "start": start + offset,
            **passes.context(start + offset, tempo.measure),
            "subtype": ScoreEventType.TEMPO_INDICATION,
            "value": tempo.bpm,
            "text": tempo.text,
            "staff": tempo.staff,
            "notated_id": tempo.xml_id,
        }
        for xml_id, start in zip(ids, starts)
    ]


def _measure_row(
    mc: int, xml_id: str, measure: _Measure, start: Fraction, length: Fraction
) -> dict[str, Any]:
    """Build the structure row of one measure: *mc*-th, id *xml_id*.

    A printed measure's id is its ``xml:id``; a rendered pass of it has its
    own id and takes everything else from the printed *measure*.
    """
    number = measure.n if measure.n is not None and measure.n.isdigit() else None
    return {
        "id": xml_id,
        "name": measure.n,
        "temporal_type": "interval",
        "event_type": ScoreEventType.MEASURE,
        "quarterbeats": rational_to_struct(start),
        "duration_qb": rational_to_struct(length),
        "mc": mc,
        "mn": measure.n,
        "mn_int": int(number) if number is not None else None,
        "mm_id": xml_id,
        "timesig": "/".join(measure.meter) if measure.meter else None,
    }


def _repeat_columns(measure: _Measure) -> dict[str, Any]:
    """Return a printed measure's repeat barlines, ending number and barline."""
    return {
        "start_repeat": measure.left in ("rptstart", "rptboth"),
        "end_repeat": measure.right in ("rptend", "rptboth"),
        "volta": measure.volta,
        "barline": measure.right,
    }


def _navigation(measures: list[_Measure]) -> list[dict[str, str | None]]:
    """Return the navigation columns of every printed measure.

    They are read from each measure's ``<repeatMark>``s, in the columns'
    shared meaning:

    * ``markers`` names the measure's segno, coda and Fine marks in document
      order, joined by ``" & "``; from the second segno or coda of the score
      on, a mark is named with its count (``coda2``).
    * ``jump_bwd`` is ``segno`` for a ``dalSegno`` (the first segno) and
      ``start`` for a ``daCapo``.
    * ``play_until`` and ``jump_fwd``: a mark whose text says "al Coda"
      plays from the jump's target until the first coda mark there (the "To
      Coda" sign), then continues at the first coda mark after the jump.
      Otherwise a jump plays until the Fine where the score has one: that
      is what a Fine mark states.
    """
    names: list[list[tuple[str, str]]] = []
    counts: Counter = Counter()
    for measure in measures:
        own = []
        for func, _ in measure.marks:
            if func in _MARKER_FUNCS:
                counts[func] += 1
                numbered = func != "fine" and counts[func] > 1
                own.append((func, f"{func}{counts[func]}" if numbered else func))
        names.append(own)

    def first(func: str, start: int, stop: int) -> str | None:
        return next(
            (name for own in names[start:stop] for kind, name in own if kind == func),
            None,
        )

    segno = next(
        (i for i, own in enumerate(names) if any(k == "segno" for k, _ in own)), None
    )
    rows: list[dict[str, str | None]] = []
    for index, measure in enumerate(measures):
        row: dict[str, str | None] = {
            "markers": " & ".join(name for _, name in names[index]) or None,
            "jump_bwd": None,
            "jump_fwd": None,
            "play_until": None,
        }
        for func, text in measure.marks:
            if func not in _JUMP_FUNCS:
                continue
            row["jump_bwd"] = _JUMP_FUNCS[func]
            target = 0 if func == "daCapo" else segno
            if "al coda" in text.lower():
                if target is not None:
                    row["play_until"] = first("coda", target, index + 1)
                    row["jump_fwd"] = first("coda", index + 1, len(measures))
            elif counts["fine"]:
                row["play_until"] = "fine"
        rows.append(row)
    return rows


def _event_data(
    cls: type[EventData],
    rows: list[dict[str, Any]],
    extra: dict[str, pa.DataType],
    unit: TimeUnit = TimeUnit.quarters,
    convert: Callable[[Fraction], Fraction] | None = None,
    origin: Fraction = Fraction(0),
) -> EventData:
    """Build *cls* from rows that all share one set of keys.

    Positions (``start``, ``end``) pass through *convert* and are taken
    relative to *origin*. In-measure offsets are quarters and are kept on a
    quarters axis only.
    """
    schema = cls.get_schema(unit, NumberType.fraction)
    columns: dict[str, Any] = {}
    for name in rows[0]:
        if name in _MEASURE_OFFSETS and unit is not TimeUnit.quarters:
            continue
        values = [row.get(name) for row in rows]
        if name in _POSITION_COLUMNS and convert is not None:
            values = [convert(value) - origin for value in values]
        if name in _STRUCT_COLUMNS:
            columns[name] = build_number_struct_array(
                values, number_type=NumberType.fraction
            )
        elif name in schema.names:
            columns[name] = pa.array(values, type=schema.field(name).type)
        else:
            columns[name] = pa.array(values, type=extra[name])
    return cls.from_arrays(
        columns,
        unit=unit,
        number_type=NumberType.fraction,
        extra_fields=[pa.field(name, data_type) for name, data_type in extra.items()],
    )
