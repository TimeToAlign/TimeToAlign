"""MpmLoader: a multimodal AlignmentBundle from an MPM-Toolbox project.

An MPM-Toolbox project is a sibling triple of XML files describing one
musical work, all sharing a common stem:

* a ``.msm`` (*Musical Sequence Markup*) score — the notated notes, each
  with a tick ``date`` (on a ``pulsesPerQuarter`` grid), a ``midi.pitch``,
  and verbatim spelling attributes (``pitchname`` / ``accidentals`` /
  ``octave``).  The score is plain (non-namespaced) XML.
* a ``.mpm`` (*Music Performance Markup*) modelled performance — one or
  more ``<performance>`` blocks, each overlaying the score with
  expressive markup grouped into *maps*: a ``tempoMap``, a
  ``dynamicsMap``, per-part ``articulationMap``\\ s, an optional
  ``asynchronyMap``, and any other map type the project carries (e.g. an
  ``ornamentationMap``).  The ``.mpm`` uses the CEMFI MPM default
  namespace (``http://www.cemfi.de/mpm/ns/1.0``), so its elements are
  matched by local name.  Tempo / dynamics values may be inline numbers
  or *style names* resolved against the performance's style-definition
  blocks.
* a ``.mpr`` (*MPM-Toolbox project*) glue file — plain XML naming the
  sibling ``.msm`` / ``.mpm`` and carrying an ``<alignment>`` block of
  observed onsets (one ``<note ref midi.pitch milliseconds.date …>`` per
  score note, keyed to the MSM ``xml:id`` by ``ref``).

``MpmLoader`` ingests the project in one call::

    bundle = MpmLoader.from_file(mpr_path).create_bundle()

and produces a single :class:`~timetoalign.alignment.bundle.AlignmentBundle`
expressing the work across all three domains (logical, physical, and
graphical):

* a shared ``"score"`` group holding the score in two logical units — a
  tick :class:`~timetoalign.timelines.types.DiscreteLogicalTimeline`
  (``score:dlt_1``) carrying the notes *and* every MPM markup event
  (tempo / dynamics / articulation / asynchrony / …), and a quarter-note
  :class:`~timetoalign.timelines.types.ContinuousLogicalTimeline`
  (``score:clt_1``) carrying the notes.  ``score:dlt_1`` carries a
  ticks→quarters :class:`~timetoalign.maps.convenience.TicksToQuarters`
  map; ``score:clt_1`` carries a modelled quarters→seconds
  :class:`~timetoalign.maps.table.TableMap` integrated from the
  performance's ``tempoMap``;
* a ``"perf"`` group holding the observed onsets in two physical units —
  a seconds :class:`~timetoalign.timelines.types.ContinuousPhysicalTimeline`
  (``perf:cpt_1``) and a samples
  :class:`~timetoalign.timelines.types.DiscretePhysicalTimeline`
  (``perf:dpt_1``) carrying a
  :class:`~timetoalign.maps.convenience.SamplesToSeconds` map (sample
  rate read from the recording's ``.wav``) — plus, when the project
  carries one, a third graphical timeline: the spectrogram's frame-column
  x-axis as a pixels
  :class:`~timetoalign.timelines.types.DiscreteGraphicalTimeline`
  (``perf:dgt_1``).  It carries no events — it is a graphical axis whose
  length is the spectrogram ``.png``'s width in frame columns — and a
  px→seconds :class:`~timetoalign.maps.linear.ScalarMap`
  (``seconds = px * hopSize / sample_rate``); and
* a cross-group :class:`~timetoalign.alignment.anchors.MatchClaim` per
  score note — a synchronous projection from ``score:clt_1`` (quarters)
  onto ``perf:cpt_1`` (seconds), joined ``xml:id == ref``.

The ``ref`` → ``xml:id`` correspondence is a bijection (every score note
has exactly one observed onset and vice versa), so every claim is
synchronous.

The loader reads the *modelled* performance and the *observed*
alignment; it never runs an aligner and never renders a meico-grade
tempo curve — accelerando / ritardando ramps (the ``transition.to``
attribute) are preserved as an event attribute, not integrated.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
    timetoalign.maps.TicksToQuarters
    timetoalign.maps.SamplesToSeconds
    timetoalign.maps.TableMap
    timetoalign.maps.ScalarMap
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lxml import etree
from typing_extensions import Self

from timetoalign.alignment.claims import Agent, MatchClaim, MatchMetadata
from timetoalign.core import AgentType, TimeUnit
from timetoalign.display.html import code
from timetoalign.loader.base import AlignmentLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.maps.convenience import SamplesToSeconds, TicksToQuarters
from timetoalign.maps.linear import ScalarMap
from timetoalign.maps.table import TableMap
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Constants

#: Score-group timeline uids (type-based ids, role prefix).
_SCORE_CLT_ID = "score:clt_1"
_SCORE_DLT_ID = "score:dlt_1"

#: Performance-group timeline uids.
_PERF_CPT_ID = "perf:cpt_1"
_PERF_DPT_ID = "perf:dpt_1"
_PERF_DGT_ID = "perf:dgt_1"

#: Group ids.
_SCORE_GROUP = "score"
_PERF_GROUP = "perf"

#: The XML-namespace ``xml:id`` attribute key.
_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"

#: The MPM map element local-names this loader recognises specially.
#: Any *other* map type is emitted generically (its entries carry their
#: raw attributes verbatim), so nothing in a project is silently dropped.
_MAP_LOCALNAMES = frozenset(
    {
        "tempoMap",
        "dynamicsMap",
        "articulationMap",
        "asynchronyMap",
        "ornamentationMap",
        "rubatoMap",
        "metricalAccentuationMap",
        "imprecisionMap",
    }
)

#: Fallback sample rate when no ``.wav`` is found next to the project.
_DEFAULT_SAMPLE_RATE = 44100

# endregion


# region Helpers


def _localname(element: Any) -> str:
    """Return an element's namespace-stripped local name."""
    return etree.QName(element).localname


def _snake_case(camel: str) -> str:
    """Convert a camelCase MPM attribute name to snake_case.

    ``absoluteDurationMs`` → ``absolute_duration_ms``;
    ``relativeVelocity`` → ``relative_velocity``.
    """
    out: list[str] = []
    for char in camel:
        if char.isupper():
            out.append("_")
            out.append(char.lower())
        else:
            out.append(char)
    return "".join(out)


def _as_float(text: str | None) -> float | None:
    """Parse an MPM float-string, or ``None`` if it is absent / not numeric."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


# endregion


# region MpmLoader


class MpmLoader(AlignmentLoader):
    """Load an MPM-Toolbox MSM+MPM+MPR triple as one multimodal AlignmentBundle.

    The loader is given the ``.mpr`` project file; it resolves the sibling
    ``.msm`` / ``.mpm`` by the bare filenames the project names, parses the
    score, the selected modelled performance's markup, and the observed
    alignment, and assembles a bundle with a shared logical ``"score"``
    group and a physical ``"perf"`` group — the latter also carrying the
    spectrogram's graphical pixel axis (``perf:dgt_1``) when the project
    ships one — linked by one synchronous :class:`MatchClaim` per score
    note.  The bundle thus spans the logical, physical, and (when a
    spectrogram is present) graphical domains.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(mpr_path)`` — ingest the whole project.
    2. ``loader.create_bundle()`` — assemble the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines.

    By default the *first* ``<performance>`` in the ``.mpm`` is used.  Pass
    ``performance=<name>`` to ``load`` to select a different one.

    The loader reads the modelled performance and the observed alignment;
    it never runs an aligner.
    """

    def __init__(self) -> None:
        super().__init__()

        self._score_clt: ContinuousLogicalTimeline | None = None
        self._score_dlt: DiscreteLogicalTimeline | None = None
        self._perf_cpt: ContinuousPhysicalTimeline | None = None
        self._perf_dpt: DiscretePhysicalTimeline | None = None
        self._perf_dgt: DiscreteGraphicalTimeline | None = None
        self._claims: list[MatchClaim] = []
        self._tempo_map: TableMap | None = None
        self._ppq: int = 720
        self._performance_name: str | None = None
        self._name: str | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Path) -> Any:
        """Not used: an MPM-Toolbox project is a single coherent unit.

        :class:`MpmLoader` ingests an entire project through :meth:`load`;
        the base class's per-source AlignmentStore merge does not apply.
        """
        raise NotImplementedError(
            "MpmLoader ingests a whole project via load(mpr_path); "
            "per-source loading is not used."
        )

    # endregion

    # region Properties

    @property
    def ppq(self) -> int:
        """Pulses per quarter note (read from the MSM/MPM)."""
        return self._ppq

    @property
    def performance_name(self) -> str | None:
        """The name of the selected ``<performance>`` block."""
        return self._performance_name

    @property
    def tempo_map(self) -> TableMap:
        """The modelled quarters→seconds TableMap (from the tempoMap)."""
        if self._tempo_map is None:
            raise RuntimeError("No project loaded yet. Call load() first.")
        return self._tempo_map

    # endregion

    # region Loading

    def load(self, mpr_path: str | Path, *, performance: str | None = None) -> Self:
        """Ingest a whole MPM-Toolbox project.

        Args:
            mpr_path: Path to the ``.mpr`` project file.  The sibling
                ``.msm`` / ``.mpm`` are resolved from the bare filenames
                the project names.
            performance: Name of the ``<performance>`` block to use.  When
                ``None`` (the default), the first performance is selected.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If the ``.mpr`` or a sibling it names is
                missing.
            ValueError: If a named performance is not present in the MPM.
        """
        mpr = Path(mpr_path)
        if not mpr.is_file():
            raise FileNotFoundError(f"Not a file: {mpr}")

        mpr_root = self._parse_xml(mpr)
        msm_path, mpm_path = self._resolve_siblings(mpr, mpr_root)

        self._sources = [mpr, msm_path, mpm_path]
        self._name = mpr.stem

        msm_root = self._parse_xml(msm_path)
        mpm_root = self._parse_xml(mpm_path)

        self._ppq = int(float(msm_root.get("pulsesPerQuarter")))

        score_notes = self._parse_msm_notes(msm_root)
        performance, self._performance_name = self._select_performance(
            mpm_root, performance
        )
        markup_events = self._parse_mpm_markup(performance)
        tempo_entries = self._collect_tempo_entries(performance)
        alignment_notes = self._parse_mpr_alignment(mpr_root)
        sample_rate = self._resolve_sample_rate(mpr)
        spectrogram = self._parse_spectrogram(mpr, mpr_root)

        self._build_score_timelines(score_notes, markup_events, tempo_entries)
        self._build_performance_timelines(alignment_notes, sample_rate)
        self._build_spectrogram_timeline(spectrogram, sample_rate)
        self._build_claims(score_notes, alignment_notes)

        return self

    @staticmethod
    def _parse_xml(path: Path) -> Any:
        """Parse an XML file with id-collection disabled.

        The MSM / MPM / MPR reuse ``xml:id`` values across files, so a
        default id-collecting parser raises ``XMLSyntaxError: ID …
        already defined``.  ``collect_ids=False`` is mandatory.  ``recover``
        is on for robustness against minor markup quirks.
        """
        parser = etree.XMLParser(recover=True, collect_ids=False)
        return etree.parse(str(path), parser).getroot()

    @staticmethod
    def _resolve_siblings(mpr: Path, mpr_root: Any) -> tuple[Path, Path]:
        """Resolve the ``.msm`` / ``.mpm`` siblings the project names.

        The ``.mpr`` carries ``<msm file="bareName.msm">`` /
        ``<mpm file="bareName.mpm">`` pointers; the referenced files are
        siblings of the ``.mpr``, looked up by bare filename.
        """
        msm_name: str | None = None
        mpm_name: str | None = None
        for child in mpr_root:
            tag = _localname(child)
            if tag == "msm":
                msm_name = child.get("file")
            elif tag == "mpm":
                mpm_name = child.get("file")
        if msm_name is None or mpm_name is None:
            raise FileNotFoundError(
                f"Project {mpr.name} does not name both an .msm and an .mpm."
            )
        msm_path = mpr.parent / Path(msm_name).name
        mpm_path = mpr.parent / Path(mpm_name).name
        if not msm_path.is_file():
            raise FileNotFoundError(f"Sibling MSM not found: {msm_path}")
        if not mpm_path.is_file():
            raise FileNotFoundError(f"Sibling MPM not found: {mpm_path}")
        return msm_path, mpm_path

    @staticmethod
    def _msm_note_row(
        note: dict[str, Any],
        start: Any,
        *,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a score-timeline Note row from a parsed MSM note.

        Pitch is afforded faithfully and *only* faithfully.  The MSM
        carries a bare ``midi.pitch`` plus a ``pitchname`` / ``accidentals``
        / ``octave`` spelling whose octave is internally inconsistent with
        ``midi.pitch`` under scientific notation — so a full
        :class:`SpecificPitch` would be inference and is NOT built.  The
        faithful pieces are:

        * the bare number → ``pitch`` as an
          :class:`~timetoalign.core.events.EnharmonicPitch` struct
          (``{midi_number}``): the **default** semantic pitch field
          (most-expressive faithful type for a number-only pitch);
        * ``pitchname`` + ``accidentals`` → ``specific_pitch_class`` as a
          :class:`~timetoalign.core.events.SpecificPitchClass` struct
          (``{step, alter}``): an **additional** afforded field, the
          spelling without the unreliable octave.

        The verbatim ``pitchname`` / ``accidentals`` / ``octave`` are
        carried as non-default raw columns alongside.

        Args:
            note: A parsed MSM note dict (see :meth:`_parse_msm_notes`).
            start: The onset coordinate (ticks for the discrete timeline,
                quarters for the continuous one).
            extra: Optional extra columns to merge (e.g. ``duration``).

        Returns:
            A row dict suitable for ``Timeline.add_events``.
        """
        row: dict[str, Any] = {
            "id": note["xml_id"],
            "start": start,
            "event_type": "Note",
            # Default semantic pitch field: EnharmonicPitch over the bare
            # MIDI number (the most-expressive faithful type here).
            "pitch": {"midi_number": note["pitch"]},
            # Verbatim spelling, kept as non-default raw columns.
            "pitchname": note["pitchname"],
            "accidentals": note["accidentals"],
            "octave": note["octave"],
        }
        # Additional afforded field: SpecificPitchClass from the faithful
        # spelling pieces (step + alter), where a pitchname is present.
        pitchname = note.get("pitchname")
        if pitchname:
            row["specific_pitch_class"] = {
                "step": str(pitchname).upper(),
                "alter": int(note["accidentals"]),
            }
        if extra:
            row.update(extra)
        return row

    def _parse_msm_notes(self, msm_root: Any) -> list[dict[str, Any]]:
        """Parse the score notes from the (plain-XML) MSM.

        Each ``<note xml:id date midi.pitch pitchname accidentals octave
        duration>`` becomes a dict with integer ticks / pitch / octave /
        duration and the verbatim spelling.  No :class:`SpecificPitch` is
        constructed — the MSM octave numbering is inconsistent with
        ``midi.pitch`` under scientific notation, so interpreting it would
        be inference rather than faithful representation; the raw
        ``midi.pitch`` integer is stored as the pitch and the spelling
        attributes are carried verbatim.
        """
        notes: list[dict[str, Any]] = []
        for element in msm_root.iter():
            if _localname(element) != "note":
                continue
            notes.append(
                {
                    "xml_id": element.get(_XML_ID),
                    "date": int(float(element.get("date"))),
                    "pitch": int(float(element.get("midi.pitch"))),
                    "pitchname": element.get("pitchname"),
                    "accidentals": int(float(element.get("accidentals"))),
                    "octave": int(float(element.get("octave"))),
                    "duration": int(float(element.get("duration"))),
                }
            )
        return notes

    @staticmethod
    def _select_performance(mpm_root: Any, name: str | None) -> tuple[Any, str]:
        """Return ``(performance_element, name)`` for the requested block.

        With ``name=None`` the first ``<performance>`` is selected.

        Raises:
            ValueError: If no performances exist, or a named one is absent.
        """
        performances = [
            element
            for element in mpm_root.iter()
            if _localname(element) == "performance"
        ]
        if not performances:
            raise ValueError("MPM contains no <performance> blocks.")
        if name is None:
            chosen = performances[0]
            return chosen, chosen.get("name")
        for performance in performances:
            if performance.get("name") == name:
                return performance, name
        available = [p.get("name") for p in performances]
        raise ValueError(f"No performance named {name!r}. Available: {available}")

    @classmethod
    def _build_style_lookup(cls, performance: Any) -> dict[str, dict[str, dict]]:
        """Index the performance's style definitions by kind → name → attrs.

        The three style families are keyed ``"tempo"`` / ``"dynamics"`` /
        ``"articulation"``.  For tempo / dynamics the stored value is the
        ``*Def``'s ``value`` (a float-string); for articulation the stored
        value is the whole ``articulationDef`` attribute dict (its numeric
        keys, snake-cased, become event columns).
        """
        lookup: dict[str, dict[str, dict]] = {
            "tempo": {},
            "dynamics": {},
            "articulation": {},
        }
        for element in performance.iter():
            tag = _localname(element)
            if tag == "tempoDef":
                lookup["tempo"][element.get("name")] = dict(element.attrib)
            elif tag == "dynamicsDef":
                lookup["dynamics"][element.get("name")] = dict(element.attrib)
            elif tag == "articulationDef":
                lookup["articulation"][element.get("name")] = {
                    _snake_case(key): value
                    for key, value in element.attrib.items()
                    if key != "name"
                }
        return lookup

    @classmethod
    def _iter_map_entries(cls, performance: Any) -> "list[tuple[str, Any, str]]":
        """Yield ``(map_localname, entry_element, style_ref)`` triples.

        Maps appear under both ``performance>global>dated`` and every
        ``performance>part>dated``.  Each map's leading ``<style>`` child
        names the active styleDef and is *not* an entry — it is skipped but
        its ``name.ref`` is attached to every following entry of that map so
        style resolution can find the right def.
        """
        entries: list[tuple[str, Any, str]] = []
        for element in performance.iter():
            if _localname(element) not in _MAP_LOCALNAMES:
                continue
            map_localname = _localname(element)
            style_ref = ""
            for child in element:
                if _localname(child) == "style":
                    style_ref = child.get("name.ref") or ""
                    continue
                entries.append((map_localname, child, style_ref))
        return entries

    def _parse_mpm_markup(self, performance: Any) -> list[dict[str, Any]]:
        """Parse every map entry in the performance into a markup-event dict.

        One dict per entry, keyed for the discrete logical timeline:
        ``start`` (int ticks), ``id`` (the entry's ``xml:id``),
        ``event_type`` (the entry element's capitalised local-name), and
        type-specific resolved columns (see the per-type helpers).  Any
        unrecognised entry type is emitted generically with its raw
        attributes, so nothing is dropped.
        """
        styles = self._build_style_lookup(performance)
        events: list[dict[str, Any]] = []
        for index, (map_localname, element, style_ref) in enumerate(
            self._iter_map_entries(performance)
        ):
            tag = _localname(element)
            event_type = tag[:1].upper() + tag[1:]
            date = int(float(element.get("date")))
            event: dict[str, Any] = {
                "id": element.get(_XML_ID) or f"{event_type.lower()}:{index}",
                "start": date,
                "event_type": event_type,
            }

            if tag == "tempo":
                self._fill_tempo(event, element, styles, style_ref)
            elif tag == "dynamics":
                self._fill_dynamics(event, element, styles, style_ref)
            elif tag == "articulation":
                self._fill_articulation(event, element, styles, style_ref)
            elif tag == "asynchrony":
                event["milliseconds_offset"] = _as_float(
                    element.get("milliseconds.offset")
                )
            else:
                # Generic fallback: carry the raw attributes verbatim
                # (skipping date / xml:id which are already represented).
                self._fill_generic(event, element)

            events.append(event)
        return events

    @staticmethod
    def _resolve_styled_value(
        raw: str | None, defs: dict[str, dict], value_key: str = "value"
    ) -> float | None:
        """Resolve an inline-or-styled MPM value to a float.

        If *raw* parses as a float it is an inline value (audio
        performances store inline numbers).  Otherwise it is a style name
        looked up in *defs*; a name with no matching def yields ``None``.
        """
        if raw is None:
            return None
        inline = _as_float(raw)
        if inline is not None:
            return inline
        styled = defs.get(raw)
        if styled is None:
            return None
        return _as_float(styled.get(value_key))

    @classmethod
    def _fill_tempo(
        cls, event: dict, element: Any, styles: dict, style_ref: str
    ) -> None:
        """Fill a Tempo event: resolved ``bpm``, ``beat_length``, labels."""
        raw_bpm = element.get("bpm")
        defs = styles["tempo"]
        event["bpm"] = cls._resolve_styled_value(raw_bpm, defs)
        event["bpm_label"] = raw_bpm
        beat_length = _as_float(element.get("beatLength"))
        if beat_length is not None:
            event["beat_length"] = beat_length
        transition_to = element.get("transition.to")
        if transition_to is not None:
            event["transition_to"] = transition_to

    @classmethod
    def _fill_dynamics(
        cls, event: dict, element: Any, styles: dict, style_ref: str
    ) -> None:
        """Fill a Dynamics event: resolved ``volume``, label, transition."""
        raw_volume = element.get("volume")
        defs = styles["dynamics"]
        event["volume"] = cls._resolve_styled_value(raw_volume, defs)
        event["volume_label"] = raw_volume
        transition_to = element.get("transition.to")
        if transition_to is not None:
            event["transition_to"] = transition_to

    @classmethod
    def _fill_articulation(
        cls, event: dict, element: Any, styles: dict, style_ref: str
    ) -> None:
        """Fill an Articulation event: ``name``, ``noteid``, resolved def attrs.

        The ``name.ref`` looks up an ``articulationDef``; its snake-cased
        numeric attributes (``relative_duration``, ``absolute_duration_ms``,
        ``absolute_velocity``, ``absolute_velocity_change`` …) become event
        columns.  A ``name.ref`` with no matching def carries the name only
        — the numeric columns stay absent (the loader does not crash).
        """
        name_ref = element.get("name.ref")
        event["name"] = name_ref
        noteid = element.get("noteid")
        if noteid is not None:
            event["noteid"] = noteid.lstrip("#")
        definition = styles["articulation"].get(name_ref)
        if definition is not None:
            for key, value in definition.items():
                numeric = _as_float(value)
                event[key] = numeric if numeric is not None else value

    @staticmethod
    def _fill_generic(event: dict, element: Any) -> None:
        """Carry an unrecognised entry's raw attributes verbatim.

        ``date`` and ``xml:id`` are skipped (already represented); a
        namespaced attribute key is reduced to its local name and a ``.``
        in an attribute name (e.g. ``name.ref``, ``note.order``) is
        snake-cased so the column name is a clean identifier.
        """
        for key, value in element.attrib.items():
            if key == _XML_ID:
                continue
            local = etree.QName(key).localname if key.startswith("{") else key
            if local == "date":
                continue
            column = _snake_case(local.replace(".", "_"))
            numeric = _as_float(value)
            event[column] = numeric if numeric is not None else value

    def _collect_tempo_entries(
        self, performance: Any
    ) -> list[tuple[int, float, float]]:
        """Collect resolved ``(date_ticks, bpm, beat_length)`` tempo triples.

        Sorted by date.  Entries with an unresolvable bpm or beat-length are
        skipped (they cannot contribute a tempo segment).
        """
        styles = self._build_style_lookup(performance)
        defs = styles["tempo"]
        triples: list[tuple[int, float, float]] = []
        for map_localname, element, _style_ref in self._iter_map_entries(performance):
            if _localname(element) != "tempo":
                continue
            bpm = self._resolve_styled_value(element.get("bpm"), defs)
            beat_length = _as_float(element.get("beatLength"))
            if bpm is None or beat_length is None or bpm <= 0 or beat_length <= 0:
                continue
            triples.append((int(float(element.get("date"))), bpm, beat_length))
        triples.sort(key=lambda triple: triple[0])
        return triples

    def _parse_mpr_alignment(self, mpr_root: Any) -> list[dict[str, Any]]:
        """Parse the observed onsets from the (plain-XML) MPR alignment block.

        The ``<alignment>`` element (under ``<audios>/<audio>``) holds
        ``<part>`` children, each with ``<note ref midi.pitch
        milliseconds.date …>`` onsets.  Onsets are collected across all
        parts.  The score-image ``<score><page><note ref x y>`` block is
        intentionally *not* parsed (a later concern); the sibling
        ``<spectrogram>`` x-axis is read separately by
        :meth:`_parse_spectrogram`.
        """
        alignment = None
        for element in mpr_root.iter():
            if _localname(element) == "alignment":
                alignment = element
                break
        if alignment is None:
            raise ValueError("MPR contains no <alignment> block.")

        notes: list[dict[str, Any]] = []
        for element in alignment.iter():
            if _localname(element) != "note":
                continue
            notes.append(
                {
                    "ref": element.get("ref"),
                    "pitch": int(float(element.get("midi.pitch"))),
                    "milliseconds_date": float(element.get("milliseconds.date")),
                    "velocity": int(float(element.get("velocity"))),
                }
            )
        return notes

    @staticmethod
    def _resolve_sample_rate(mpr: Path) -> int:
        """Read the recording's sample rate (Hz) from a sibling ``.wav``.

        The ``.mpr``'s ``<audio>`` points at an ``.mp3``; the ``.wav`` is
        preferred and located by globbing ``*.wav`` under the project
        directory tree.  Falls back to 44100 Hz when no ``.wav`` is found.
        """
        wavs = sorted(mpr.parent.rglob("*.wav"))
        if not wavs:
            return _DEFAULT_SAMPLE_RATE
        return AudioLoader.from_file(wavs[0]).audio_info.sample_rate

    @staticmethod
    def _parse_spectrogram(mpr: Path, mpr_root: Any) -> dict[str, Any] | None:
        """Parse the ``<spectrogram>`` element (under ``<audios>/<audio>``).

        The spectrogram's x-axis is time in frame columns: each column
        advances by ``hopSize`` audio samples.  The XML carries no
        width/height, so the number of frame columns is the width of the
        referenced ``.png`` (read from its PNG IHDR header).

        Returns a dict ``{"hop_size": int, "n_columns": int}`` (the
        timeline length in pixels), or ``None`` when the project carries no
        ``<spectrogram>`` or its ``.png`` is missing — in which case the
        graphical timeline is simply absent (the loader does not crash).
        """
        spectrogram = next(
            (
                element
                for element in mpr_root.iter()
                if _localname(element) == "spectrogram"
            ),
            None,
        )
        if spectrogram is None:
            return None

        hop_size_text = spectrogram.get("hopSize")
        file_attr = spectrogram.get("file")
        if hop_size_text is None or file_attr is None:
            return None

        png_path = mpr.parent / Path(file_attr)
        if not png_path.is_file():
            return None

        n_columns = MpmLoader._read_png_width(png_path)
        if n_columns is None:
            return None

        return {"hop_size": int(float(hop_size_text)), "n_columns": n_columns}

    @staticmethod
    def _read_png_width(png_path: Path) -> int | None:
        """Read a PNG's pixel width from its IHDR header.

        A PNG file begins with the 8-byte signature, then the IHDR chunk
        whose 4-byte length + 4-byte type are followed by the big-endian
        ``uint32`` width and height.  Bytes 16:20 are therefore the width.
        Reading only the header avoids decoding the image (no heavy image
        dependency).  Returns ``None`` if the file is not a valid PNG.
        """
        with png_path.open("rb") as handle:
            header = handle.read(24)
        if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return int.from_bytes(header[16:20], byteorder="big")

    def _build_score_timelines(
        self,
        score_notes: list[dict[str, Any]],
        markup_events: list[dict[str, Any]],
        tempo_entries: list[tuple[int, float, float]],
    ) -> None:
        """Build the two shared logical score timelines.

        ``score:dlt_1`` (ticks) holds the Note events *and* every MPM markup
        event, plus a ticks→quarters :class:`TicksToQuarters` map.
        ``score:clt_1`` (quarters) holds the Note events plus the modelled
        quarters→seconds :class:`TableMap`.
        """
        ppq = self._ppq

        max_note_tick_end = max(
            (note["date"] + note["duration"] for note in score_notes), default=0
        )
        max_markup_tick = max((event["start"] for event in markup_events), default=0)
        dlt_length = max(max_note_tick_end, max_markup_tick)
        max_quarter_end = max(
            (Fraction(note["date"] + note["duration"], ppq) for note in score_notes),
            default=Fraction(0),
        )

        # Discrete logical timeline: notes first, then all markup events.
        score_dlt = DiscreteLogicalTimeline(
            length=dlt_length,
            unit=TimeUnit.ticks,
            uid=_SCORE_DLT_ID,
            name=self._name,
        )
        score_dlt.add_events(
            [
                self._msm_note_row(
                    note, note["date"], extra={"duration": note["duration"]}
                )
                for note in score_notes
            ]
        )
        # Markup events carry a heterogeneous schema (tempo / dynamics /
        # articulation / asynchrony / …); add_events with allow_expansion
        # null-fills the columns absent from the Note batch.
        if markup_events:
            score_dlt.add_events(markup_events, allow_expansion=True)
        score_dlt.add_conversion_map(TicksToQuarters(ppq=ppq))

        # Continuous logical timeline: notes in quarters.
        score_clt = ContinuousLogicalTimeline(
            length=max_quarter_end,
            unit=TimeUnit.quarters,
            uid=_SCORE_CLT_ID,
            name=self._name,
        )
        score_clt.add_events(
            [
                self._msm_note_row(note, Fraction(note["date"], ppq))
                for note in score_notes
            ]
        )
        self._tempo_map = self._build_tempo_table_map(tempo_entries, max_quarter_end)
        if self._tempo_map is not None:
            score_clt.add_conversion_map(self._tempo_map)

        self._score_dlt = score_dlt
        self._score_clt = score_clt

    def _build_tempo_table_map(
        self,
        tempo_entries: list[tuple[int, float, float]],
        max_quarter_end: Fraction,
    ) -> TableMap | None:
        """Integrate the tempoMap into a quarters→seconds :class:`TableMap`.

        Constant tempo per segment (``transition.to`` ramps are ignored
        here — they are preserved only as a Tempo-event attribute).  For
        each tempo entry with resolved bpm ``B`` and beat-length ``L``, the
        segment's seconds-per-quarter is ``(0.25 / L) * (60 / B)``.  The
        entry dates (ticks) are converted to quarters and the segments are
        cumulatively integrated; a final anchor at the score's last quarter
        extends the last segment.

        Returns ``None`` when there are fewer than two anchors (a TableMap
        needs at least two points).
        """
        ppq = self._ppq
        if not tempo_entries:
            return None

        quarters: list[Fraction] = []
        seconds: list[float] = []
        cumulative = 0.0
        for index, (date, bpm, beat_length) in enumerate(tempo_entries):
            current_q = Fraction(date, ppq)
            if index == 0:
                quarters.append(current_q)
                seconds.append(0.0)
                continue
            prev_date, prev_bpm, prev_beat_length = tempo_entries[index - 1]
            prev_q = Fraction(prev_date, ppq)
            prev_spq = (0.25 / prev_beat_length) * (60.0 / prev_bpm)
            cumulative = seconds[-1] + float(current_q - prev_q) * prev_spq
            quarters.append(current_q)
            seconds.append(cumulative)

        # Final anchor at the score's last quarter using the last segment.
        last_date, last_bpm, last_beat_length = tempo_entries[-1]
        last_spq = (0.25 / last_beat_length) * (60.0 / last_bpm)
        if max_quarter_end > quarters[-1]:
            cumulative = seconds[-1] + float(max_quarter_end - quarters[-1]) * last_spq
            quarters.append(max_quarter_end)
            seconds.append(cumulative)

        if len(quarters) < 2:
            return None

        return TableMap(
            x_values=quarters,
            y_values=seconds,
            kind="linear",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )

    def _build_performance_timelines(
        self,
        alignment_notes: list[dict[str, Any]],
        sample_rate: int,
    ) -> None:
        """Build the two physical performance timelines from the onsets.

        ``perf:cpt_1`` (seconds) holds one Note event per observed onset
        (``milliseconds.date / 1000``); ``perf:dpt_1`` (samples) holds the
        same onsets scaled by the sample rate, with a
        :class:`SamplesToSeconds` map.
        """
        seconds = [note["milliseconds_date"] / 1000.0 for note in alignment_notes]
        max_seconds = max(seconds, default=0.0)
        max_samples = int(round(max_seconds * sample_rate))

        perf_cpt = ContinuousPhysicalTimeline(
            length=max_seconds,
            unit=TimeUnit.seconds,
            uid=_PERF_CPT_ID,
            name=self._name,
        )
        perf_cpt.add_events(
            [
                {
                    "id": note["ref"],
                    "start": note["milliseconds_date"] / 1000.0,
                    "event_type": "Note",
                    # Observed onsets carry a bare MIDI number: afford
                    # EnharmonicPitch via its {midi_number} struct.
                    "pitch": {"midi_number": note["pitch"]},
                    "velocity": note["velocity"],
                }
                for note in alignment_notes
            ]
        )

        perf_dpt = DiscretePhysicalTimeline(
            length=max_samples,
            unit=TimeUnit.samples,
            uid=_PERF_DPT_ID,
            name=self._name,
        )
        perf_dpt.add_events(
            [
                {
                    "id": note["ref"],
                    "start": int(
                        round(note["milliseconds_date"] / 1000.0 * sample_rate)
                    ),
                    "event_type": "Note",
                    "pitch": {"midi_number": note["pitch"]},
                    "velocity": note["velocity"],
                }
                for note in alignment_notes
            ]
        )
        perf_dpt.add_conversion_map(SamplesToSeconds(sample_rate=sample_rate))

        self._perf_cpt = perf_cpt
        self._perf_dpt = perf_dpt

    def _build_spectrogram_timeline(
        self,
        spectrogram: dict[str, Any] | None,
        sample_rate: int,
    ) -> None:
        """Build the graphical performance timeline from the spectrogram.

        ``perf:dgt_1`` (pixels) is the spectrogram's frame-column x-axis: a
        :class:`DiscreteGraphicalTimeline` whose length is the number of
        frame columns (the ``.png``'s pixel width).  It carries **no
        events** — the columns are a graphical axis, not events — and a
        px→seconds :class:`ScalarMap` whose scalar is ``hopSize /
        sample_rate`` (each column advances ``hopSize`` audio samples, so
        ``seconds = px * hopSize / sample_rate``).

        When the project carries no spectrogram, the timeline is left
        ``None`` and the bundle simply omits the graphical domain.
        """
        if spectrogram is None:
            self._perf_dgt = None
            return

        perf_dgt = DiscreteGraphicalTimeline(
            length=spectrogram["n_columns"],
            unit=TimeUnit.pixels,
            uid=_PERF_DGT_ID,
            name=self._name,
        )
        perf_dgt.add_conversion_map(
            ScalarMap(
                scalar=spectrogram["hop_size"] / sample_rate,
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.seconds,
            )
        )
        self._perf_dgt = perf_dgt

    def _build_claims(
        self,
        score_notes: list[dict[str, Any]],
        alignment_notes: list[dict[str, Any]],
    ) -> None:
        """Emit one synchronous cross-group MatchClaim per score note.

        The join ``xml_id == ref`` is a bijection, so each score note's
        quarter onset is projected onto its observed seconds onset.  The
        claim source is ``score:clt_1`` (quarters); the target is
        ``perf:cpt_1`` (seconds).
        """
        ppq = self._ppq
        quarter_by_id = {
            note["xml_id"]: Fraction(note["date"], ppq) for note in score_notes
        }
        seconds_by_ref = {
            note["ref"]: note["milliseconds_date"] / 1000.0 for note in alignment_notes
        }
        meta = MatchMetadata(
            agent=Agent(
                name="mpm",
                type=AgentType.software,
                identifier=self._performance_name,
            ),
            certainty=1.0,
        )
        for ref, target_seconds in seconds_by_ref.items():
            source_quarters = quarter_by_id.get(ref)
            if source_quarters is None:
                continue
            self._claims.append(
                MatchClaim.from_projection(
                    event={"start": source_quarters},
                    source_tl_id=_SCORE_CLT_ID,
                    target_tl_id=_PERF_CPT_ID,
                    target_coord=float(target_seconds),
                    source_unit=TimeUnit.quarters,
                    target_unit=TimeUnit.seconds,
                    coord_key="start",
                    metadata=meta,
                )
            )

    # endregion

    # region Domain Object Creation

    def create_bundle(self) -> "AlignmentBundle":
        """Assemble the AlignmentBundle from the loaded project.

        Returns:
            An ``AlignmentBundle`` with a shared ``"score"`` group (two
            logical timelines), a ``"perf"`` group (two physical
            timelines), and one synchronous cross-group MatchClaim per
            score note.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if (
            self._score_clt is None
            or self._score_dlt is None
            or self._perf_cpt is None
            or self._perf_dpt is None
        ):
            raise RuntimeError(
                "No project loaded yet. Call load() before create_bundle()."
            )

        bundle = AlignmentBundle(name=self._name)
        bundle.add_timeline(self._score_clt, uid=_SCORE_CLT_ID, as_group=_SCORE_GROUP)
        bundle.add_timeline(
            self._score_dlt, uid=_SCORE_DLT_ID, grouped_with=_SCORE_CLT_ID
        )
        bundle.add_timeline(self._perf_cpt, uid=_PERF_CPT_ID, as_group=_PERF_GROUP)
        bundle.add_timeline(self._perf_dpt, uid=_PERF_DPT_ID, grouped_with=_PERF_CPT_ID)
        if self._perf_dgt is not None:
            bundle.add_timeline(
                self._perf_dgt, uid=_PERF_DGT_ID, grouped_with=_PERF_CPT_ID
            )
        bundle.add_match_claims(self._claims)
        return bundle

    def create_timelines(self, id_pattern: str | None = None) -> list["Timeline"]:
        """Return all loaded timelines: the two score, then the performance.

        The performance group always contributes ``perf:cpt_1`` and
        ``perf:dpt_1``; ``perf:dgt_1`` (the spectrogram graphical axis) is
        appended when the project carries a spectrogram.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.
        """
        if (
            self._score_clt is None
            or self._score_dlt is None
            or self._perf_cpt is None
            or self._perf_dpt is None
        ):
            return []
        timelines: list["Timeline"] = [
            self._score_clt,
            self._score_dlt,
            self._perf_cpt,
            self._perf_dpt,
        ]
        if self._perf_dgt is not None:
            timelines.append(self._perf_dgt)
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Return a single timeline by its uid.

        Args:
            uid: One of ``"score:clt_1"`` / ``"score:dlt_1"`` / ``"perf:cpt_1"``
                / ``"perf:dpt_1"`` / ``"perf:dgt_1"`` (the last present only
                when the project carries a spectrogram).

        Raises:
            KeyError: If no timeline matches.
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._score_clt is None:
            raise RuntimeError(
                "No project loaded yet. Call load() before create_timeline()."
            )
        mapping: dict[str, "Timeline"] = {
            _SCORE_CLT_ID: self._score_clt,
            _SCORE_DLT_ID: self._score_dlt,
            _PERF_CPT_ID: self._perf_cpt,
            _PERF_DPT_ID: self._perf_dpt,
        }
        if self._perf_dgt is not None:
            mapping[_PERF_DGT_ID] = self._perf_dgt
        if uid in mapping:
            return mapping[uid]
        raise KeyError(
            f"No timeline with uid '{uid}'. Available: "
            + ", ".join(f"'{uid}'" for uid in mapping)
        )

    # endregion

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is cross-group claims, not store events."""
        return ("Claims", str(len(self._claims)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Extend the base rows with the MPM-specific shape.

        The base :class:`AlignmentLoader` count row is replaced by the
        claim count (see :meth:`_repr_count_row`); the data lives in the
        assembled timelines and claims, not the unpopulated per-source
        ``AlignmentStore``.
        """
        rows = super()._repr_rows()
        name = self._name or "(not loaded)"
        performance = self._performance_name or "(not loaded)"
        rows.append(("Project", code(name)))
        rows.append(("Performance", code(performance)))
        if self._score_clt is not None:
            n_timelines = 5 if self._perf_dgt is not None else 4
            rows.append(("Timelines", f"{n_timelines} in 2 group(s) (score, perf)"))
        return rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Number of cross-group MatchClaims (the loader's primary payload).

        The inherited count reads the per-source ``AlignmentStore``, which
        this whole-project loader never populates; the claim count is the
        meaningful size and keeps :meth:`_repr_html_` consistent with
        :meth:`__repr__`.
        """
        return len(self._claims)

    def __repr__(self) -> str:
        n_claims = len(self._claims)
        performance = self._performance_name
        return f"MpmLoader(performance={performance!r}, claims={n_claims})"

    # endregion


# endregion
