"""Ieee1599Loader: one IEEE 1599 document as a multimodal AlignmentBundle.

An IEEE 1599 file is a multi-layer XML encoding of a single musical work.  Its
pivot is the ``<spine>``: a flat list of ``<event>`` elements, each with an
``id``, that every other layer points back at through ``event_ref``.  The spine
is therefore an abstract, unit-less event ordering — a *virtual timing unit*
(VTU) axis — and each layer (symbolic notation, engraved page images, audio
recordings) states where in *its own* coordinate space a given spine event
occurs.  That is exactly an alignment: the spine is the hub, the layers are
projections, and ``event_ref`` is the correspondence.

``Ieee1599Loader`` reads one document in one call::

    bundle = Ieee1599Loader.from_file(path).create_bundle()

and produces a single :class:`~timetoalign.alignment.bundle.AlignmentBundle`:

============================  =========================================
IEEE 1599 construct           TimeToAlign! representation
============================  =========================================
``<spine>``                   ``DiscreteLogicalTimeline`` ``spine:dlt_1``,
                              unit ``ticks``, ``NumberType.int``
``<logic><los>`` notes,       ONE ``DiscreteLogicalTimeline`` ``los:dlt_2``,
rests and lyric syllables     unit ``ticks``
``<notational>`` per          ONE nested ``SegmentLine`` per group: one page
``graphic_instance_group``    child per ``<graphic_instance>``, each holding
                              its ``DiscreteGraphicalTimeline`` accolades,
                              unit ``pixels``
``<audio>`` per ``<track>``   ONE ``ContinuousPhysicalTimeline``,
                              unit ``seconds``
``<structural>`` per          one ``external_references`` row on the
``<segment_event>``           spine timeline, naming the Petri-net place
                              that segment maps to
every ``event_ref``           one synchronous ``MatchClaim``, projection
                              timeline against the spine, all of them in
                              one columnar ``MatchClaimField``
============================  =========================================

**Spine coordinates.**  The ``timing`` and ``hpos`` attributes of a spine
``<event>`` are *relative* integer deltas against the preceding event.  The
stored coordinate is the running sum — the cumulative VTU — so that two events
notated as simultaneous share one coordinate.  ``hpos`` is accumulated the same
way and kept as a single integer event field; the deltas are not stored
alongside, being recoverable by differencing.  Source event ids
(``part_1_voice0_measure1_ev0``, ``event_cow``, …) are preserved verbatim as
spine event ``id`` values, which is what makes ``event_ref`` resolvable on the
way back out.

**Projected layers.**  Every LOS, graphic and track event carries its raw
``event_ref`` as a field *and* contributes one synchronous claim tying its own
coordinate to the referenced spine event's VTU coordinate.  LOS events sit at
the VTU coordinate of the event they reference (that reference *is* their
temporal position); graphic events are integer-pixel interval events in their
edition SegmentLine; track events are instants at ``start_time`` seconds.

**The structural layer.**  An ``<analysis>`` partitions the spine into
``<segment>`` elements, each listing the spine events it covers as
``<segment_event event_ref="…"/>``; a sibling ``<petri_nets>`` block names the
``.pnml`` files that model those segments, a ``<place place_ref="p2"
segment_ref="…"/>`` binding one place of one net to one segment.  That is a
reference *into* an external resource, not a timing statement, so it is carried
as :attr:`~timetoalign.timelines.base.Timeline.external_references` on the
spine: one row per ``(segment_event, place)`` pair, whose ``event_id`` is the
spine event, whose ``external_id`` is the ``place_ref``, whose single access
point is the net's ``file_name`` with kind ``relative_path``, and whose
``comment`` is the segment id.  A segment no place names keeps its row —
``external_id`` the segment id, no access point, ``comment`` ``"segment without
petri-net node"`` — so the segmentation survives whole even where the
Petri-net modelling is incomplete.

**Fidelity rules.**  Notated durations are kept as the verbatim ``num`` /
``den`` integer pair (``duration_num`` / ``duration_den``) rather than a
reduced :class:`~fractions.Fraction`, so that ``4/4`` does not silently become
``1``; the exact rational is ``Fraction(duration_num, duration_den)``.
``<undefined/>`` accidental placeholders are recorded as the string
``"undefined"``, never dropped and never inferred.  Media files are never
opened: ``file_name`` and ``file_format`` are recorded verbatim even when the
file is absent from disk, and even when the format label contradicts the
extension (``video_avi`` naming a ``.mp4``).  ``.pnml`` files are not opened
either — a Petri net's places are cross-referenced from the IEEE 1599 document
itself, so the whole mapping is read there and no path is ever resolved.
Page-image coordinates are measured in pixels (``measurement_unit="pixels"``
on every ``<graphic_instance>``). They are rounded to integer pixels using
round-half-even for the graphical SegmentLine and its child timelines; the
verbatim source box is retained as the float ``source_bbox`` event field when
any of its four coordinates is fractional; it is omitted when all four are
integral. Each graphical event's integer box is a raw ``bbox`` struct with
``ul`` / ``lr`` and ``x`` / ``y`` members. Every page of an edition is its own
image file with its own pixel origin, so an edition nests two levels deep: the
edition line's segments are its pages, and a page's segments are the accolades
engraved on it. A page child carries its ``<graphic_instance>`` attributes in
``meta["page"]``, and each event keeps its own ``file_name`` and
``position_in_group``; the edition's interval-to-constant map names the page
image containing any edition coordinate.

**Known limitations.**

* An ``<analysis>`` that describes timeline events rather than spine segments
  has no representation here: the structural layer is read only through the
  ``<segment>`` / ``<place>`` chain that resolves to spine events.
* ``<staff_list>`` clefs, key signatures and time signatures reference spine
  events but are not note/rest/lyric content; they are kept in the curated
  ``staff_list`` store table and do not become timeline events or claims.
* The ``performance`` layer does not occur in any known specimen and is
  skipped with a debug log line, as is any other unrecognised section.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
    timetoalign.MatchClaimField
    timetoalign.loader.format.xml.XmlLoader
"""

from __future__ import annotations

import logging
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pyarrow as pa

from timetoalign.alignment.claims import (
    Agent,
    MatchClaim,
    MatchClaimField,
    MatchMetadata,
)
from timetoalign.core import AgentType, NumberType, TimeUnit
from timetoalign.core.fields import SemanticField
from timetoalign.display.html import code
from timetoalign.loader.format.xml import XmlLoader
from timetoalign.maps.interval import IntervalToConstantMap
from timetoalign.storage.events import EventData
from timetoalign.timelines.factory import _mint_timeline_id
from timetoalign.timelines.types import (
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    SegmentLine,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Constants

#: Provenance recorded on every claim.  The alignment is not computed here: it
#: is read off the document's own ``event_ref`` cross-references.
_DEFAULT_AGENT = "IEEE 1599"
_AGENT_IDENTIFIER = "ieee1599_event_ref"

#: Document sections this loader represents.  Anything else is logged and
#: skipped (see the module docstring's limitations).
_KNOWN_SECTIONS = frozenset({"general", "logic", "notational", "audio", "structural"})

#: The unit graphic-event boxes declare in the IEEE 1599 document.
_GRAPHIC_UNIT = TimeUnit.pixels

#: Access-point kind of a ``<petri_net>`` ``file_name``: a path relative to
#: the IEEE 1599 document, never resolved against the file system.
_PETRI_NET_ACCESS_KIND = "relative_path"

#: ``comment`` of a segment that no ``<place>`` names.
_UNMAPPED_SEGMENT_COMMENT = "segment without petri-net node"

# endregion


# region Layer specs


@dataclass
class _EditionSpec:
    """One ``<graphic_instance_group>``: an edition of the engraved score."""

    uid: str
    role: str
    description: str
    pages: list[_GraphicPageSpec] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    claim_coordinates: list[int] = field(default_factory=list)


@dataclass
class _GraphicPageSpec:
    """One ``<graphic_instance>``: one page image and the accolades on it."""

    page: dict[str, Any]
    segments: list[_GraphicSegmentSpec] = field(default_factory=list)


@dataclass
class _GraphicSegmentSpec:
    """One contiguous accolade from one graphic-instance page."""

    rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _TrackSpec:
    """One ``<track>``: an audio (or video) rendition of the work."""

    uid: str
    role: str
    file_name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    performers: list[dict[str, Any]] = field(default_factory=list)
    recordings: list[dict[str, Any]] = field(default_factory=list)
    notes: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)


# endregion


# region Parsing helpers


def _int_or_none(value: str | None) -> int | None:
    """Parse an XML attribute as an ``int``, or ``None`` if absent/unparseable."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_none(value: str | None) -> float | None:
    """Parse an XML attribute as a ``float``, or ``None`` if absent/unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _rounded_pixel(value: str | None) -> int | None:
    """Round one XML pixel coordinate to an integer with half-even semantics."""
    if value is None:
        return None
    try:
        return int(Decimal(value).to_integral_value(rounding=ROUND_HALF_EVEN))
    except (InvalidOperation, ValueError):
        return None


def _text_or_none(element: ET.Element) -> str | None:
    """Return an element's stripped text content, or ``None`` if empty."""
    if element.text is None:
        return None
    text = element.text.strip()
    return text or None


def _prune_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop keys that are ``None`` in every row.

    Every row is built with the full key set of its layer so that the layers
    stay readable; a specimen without lyrics or without tuplets should not
    carry all-null columns into its timeline or its store table.

    Args:
        rows: Row dicts sharing one key set.

    Returns:
        The same rows, restricted to the keys that carry at least one value.
    """
    if not rows:
        return rows
    kept = [key for key in rows[0] if any(row[key] is not None for row in rows)]
    return [{key: row[key] for key in kept} for row in rows]


def _table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    """Build a column-oriented ``pa.Table`` from uniformly-keyed row dicts."""
    if not rows:
        return pa.table({})
    return pa.table({key: [row[key] for row in rows] for key in rows[0]})


_BOX_SCHEMA = pa.struct(
    [
        pa.field(
            "ul", pa.struct([pa.field("x", pa.int64()), pa.field("y", pa.int64())])
        ),
        pa.field(
            "lr", pa.struct([pa.field("x", pa.int64()), pa.field("y", pa.int64())])
        ),
    ]
)

_SOURCE_BOX_SCHEMA = pa.struct(
    [
        pa.field(
            "ul", pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        ),
        pa.field(
            "lr", pa.struct([pa.field("x", pa.float64()), pa.field("y", pa.float64())])
        ),
    ]
)

_EMPTY_CURATED_SCHEMAS = {
    "spine": pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("event_type", pa.string()),
            pa.field("instant", pa.int64()),
            pa.field("hpos", pa.int64()),
        ]
    ),
    "los": pa.schema(
        [
            pa.field("event_type", pa.string()),
            pa.field("instant", pa.int64()),
            pa.field("event_ref", pa.string()),
            pa.field("part", pa.string()),
            pa.field("staff", pa.string()),
            pa.field("voice", pa.string()),
            pa.field("measure", pa.string()),
            pa.field("notehead_index", pa.int64()),
            pa.field("duration_num", pa.int64()),
            pa.field("duration_den", pa.int64()),
            pa.field("tuplet_enter_num", pa.int64()),
            pa.field("tuplet_enter_den", pa.int64()),
            pa.field("tuplet_in_num", pa.int64()),
            pa.field("tuplet_in_den", pa.int64()),
            pa.field("augmentation_dots", pa.int64()),
            pa.field("step", pa.string()),
            pa.field("octave", pa.int64()),
            pa.field("actual_accidental", pa.string()),
            pa.field("printed_accidental", pa.string()),
            pa.field("tie", pa.bool_()),
            pa.field("text", pa.string()),
            pa.field("hyphen", pa.string()),
        ]
    ),
    "notational": pa.schema(
        [
            pa.field("timeline_uid", pa.string()),
            pa.field("edition", pa.string()),
            pa.field("event_type", pa.string()),
            pa.field("start", pa.int64()),
            pa.field("end", pa.int64()),
            pa.field("event_ref", pa.string()),
            pa.field("bbox", _BOX_SCHEMA),
            pa.field("source_bbox", _SOURCE_BOX_SCHEMA),
            pa.field("file_name", pa.string()),
            pa.field("position_in_group", pa.int64()),
        ]
    ),
    "audio": pa.schema(
        [
            pa.field("timeline_uid", pa.string()),
            pa.field("track", pa.string()),
            pa.field("event_type", pa.string()),
            pa.field("instant", pa.float64()),
            pa.field("event_ref", pa.string()),
            pa.field("file_name", pa.string()),
        ]
    ),
}


# endregion


# region Ieee1599Loader


class Ieee1599Loader(XmlLoader):
    """Load one IEEE 1599 document as a multimodal AlignmentBundle.

    The loader parses a single IEEE 1599 XML file and represents its spine as
    the hub timeline that every other layer is aligned to, exactly as the
    document's own ``event_ref`` cross-references state.  See the module
    docstring for the full construct-by-construct mapping and the fidelity
    rules.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(path)`` — parse one IEEE 1599 file into curated tables.
    2. ``loader.create_bundle()`` — assemble the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines (built once, then cached).

    The parse phase fills a
    :class:`~timetoalign.storage.store.DictStore` with one curated table per
    layer — ``spine``, ``los``, ``staff_list``, ``notational``, ``audio``,
    ``structural`` — rather than the generic tag-flattening its ``XmlLoader``
    base performs; the layers are too differently shaped for one
    auto-detected schema to describe them.  A layer the document does not
    state gets no table, except that ``spine``, ``los``, ``notational`` and
    ``audio`` are always present with their stable schema, even when empty.

    Examples:
        >>> loader = Ieee1599Loader.from_file("gymnopedie_01.xml")
        >>> bundle = loader.create_bundle()
        >>> bundle.get_matchstamp_table(from_graph=True).num_rows > 0
        True
    """

    _default_unit = TimeUnit.ticks

    def __init__(self) -> None:
        super().__init__(principal_tags=[])
        # The spine is the loader's frame of reference: VTU ticks, integers.
        self._unit = TimeUnit.ticks
        self._number_type = NumberType.int
        self._events = EventData.empty(self._unit, self._number_type)
        self._logger = module_logger.getChild(self.__class__.__name__)

        self._used_roles: set[str] = set()
        self._spine_uid: str | None = None
        self._los_uid: str | None = None
        #: Spine event id -> cumulative VTU coordinate.
        self._spine_coordinates: dict[str, int] = {}
        self._spine_rows: list[dict[str, Any]] = []
        self._los_rows: list[dict[str, Any]] = []
        self._staff_rows: list[dict[str, Any]] = []
        #: External-reference rows the ``<structural>`` layer states.
        self._structural_rows: list[dict[str, Any]] = []
        self._editions: list[_EditionSpec] = []
        self._tracks: list[_TrackSpec] = []
        self._claim_field: MatchClaimField | None = None
        self._timeline_cache: dict[str, Timeline] = {}
        self._name: str | None = None

    # region Loading

    def load(self, *sources: Path | str) -> Self:
        """Ingest exactly one IEEE 1599 document.

        Args:
            *sources: A single path to an IEEE 1599 XML file.  The file may
                carry a UTF-8 BOM and a DOCTYPE declaration; no DTD is
                fetched and no media file is opened.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If more than one source is given, or if this loader
                has already ingested a document.  One IEEE 1599 document is
                one self-contained work with its own spine; two documents are
                two bundles, so they need two loaders.
        """
        if len(sources) != 1:
            raise ValueError(
                "Ieee1599Loader ingests exactly one IEEE 1599 document per "
                f"loader; got {len(sources)} sources."
            )
        self._reject_second_document()
        self._name = Path(sources[0]).stem
        return super().load(*sources)

    def load_string(self, xml_string: str) -> Self:
        """Ingest exactly one IEEE 1599 document from an XML string.

        Args:
            xml_string: A complete ``<ieee1599>`` document.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If this loader has already ingested a document.
        """
        self._reject_second_document()
        return super().load_string(xml_string)

    def load_element(self, root: ET.Element) -> Self:
        """Ingest exactly one already-parsed ``<ieee1599>`` element.

        Args:
            root: The ``<ieee1599>`` root element.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If this loader has already ingested a document.
        """
        self._reject_second_document()
        return super().load_element(root)

    def _reject_second_document(self) -> None:
        """Guard every ingest path against a second document.

        One IEEE 1599 document is one self-contained work with its own spine;
        a second one parsed into the same loader would suffix its timeline
        uids onto the first one's and claim against the wrong spine, so it is
        refused however it arrives — as a path, a string or an element.

        Raises:
            ValueError: If a document is already held.
        """
        if self._spine_uid is None:
            return
        held = self._sources[0].name if self._sources else self._name or "a document"
        raise ValueError(
            f"This Ieee1599Loader already holds {held}. Each IEEE 1599 "
            "document is a self-contained bundle: use a new loader, or call "
            "clear() first."
        )

    def clear(self) -> None:
        """Discard the loaded document and every derived artefact."""
        super().clear()
        self._used_roles.clear()
        self._spine_uid = None
        self._los_uid = None
        self._spine_coordinates.clear()
        self._spine_rows.clear()
        self._los_rows.clear()
        self._staff_rows.clear()
        self._structural_rows.clear()
        self._editions.clear()
        self._tracks.clear()
        self._claim_field = None
        self._timeline_cache.clear()
        self._name = None
        self._minted_timeline_ids.clear()

    def _process_xml(self, root: ET.Element) -> None:
        """Parse one ``<ieee1599>`` root into curated tables and claim fields.

        Overrides the generic ``XmlLoader`` normalisation: the layers of an
        IEEE 1599 document have known, mutually incompatible shapes, so each
        is read into its own curated table instead of being auto-flattened.

        Args:
            root: The ``<ieee1599>`` root element.
        """
        self._read_metadata(root)
        self._read_spine(root)
        self._read_los(root)
        self._read_notational(root)
        self._read_audio(root)
        self._read_structural(root)
        self._log_skipped_sections(root)
        self._store_tables()
        self._build_claim_field()

    def _read_metadata(self, root: ET.Element) -> None:
        """Collect root attributes and ``<general>`` bibliographic metadata."""
        metadata: dict[str, Any] = {
            key: value for key, value in root.attrib.items() if not key.startswith("_")
        }
        description = root.find("./general/description")
        if description is not None:
            for tag in ("main_title", "work_title", "work_number"):
                element = description.find(tag)
                if element is not None:
                    text = _text_or_none(element)
                    if text is not None:
                        metadata["title" if tag == "main_title" else tag] = text
            authors = [
                {"name": _text_or_none(author), "type": author.get("type")}
                for author in description.findall("author")
            ]
            if authors:
                metadata["authors"] = authors
        self._file_metadata.update(metadata)

    def _read_spine(self, root: ET.Element) -> None:
        """Accumulate the spine's relative deltas into absolute coordinates.

        ``timing`` and ``hpos`` are per-event deltas against the previous
        event; both are accumulated, the timing sum becoming the event's VTU
        coordinate and the hpos sum a single integer field.  Events notated as
        simultaneous carry ``timing="0"`` and so share a coordinate.
        """
        self._spine_uid = _mint_timeline_id(
            self, TimeUnit.ticks, role=self._claim_role("spine")
        )
        timing = 0
        hpos = 0
        for event in root.findall("./logic/spine/event"):
            event_id = event.get("id")
            if event_id is None:
                self._logger.debug("Skipping spine <event> without an id.")
                continue
            timing += _int_or_none(event.get("timing")) or 0
            hpos += _int_or_none(event.get("hpos")) or 0
            self._spine_coordinates[event_id] = timing
            self._spine_rows.append(
                {
                    "id": event_id,
                    "event_type": "SpineEvent",
                    "instant": timing,
                    "hpos": hpos,
                }
            )
        self._spine_rows = _prune_columns(self._spine_rows)

    def _read_los(self, root: ET.Element) -> None:
        """Read the ``<los>`` note / rest / lyric content and its staff list.

        Every LOS event is placed at the VTU coordinate of the spine event it
        references.  A ``<chord>`` contributes one event per ``<notehead>``
        (so that each event carries exactly one pitch); the chord is
        recoverable by grouping on ``event_ref`` and ordering by
        ``notehead_index``.
        """
        self._los_uid = _mint_timeline_id(
            self, TimeUnit.ticks, role=self._claim_role("los")
        )
        los = root.find("./logic/los")
        if los is None:
            return

        for staff in los.findall("./staff_list/staff"):
            self._read_staff(staff)

        for part in los.findall("part"):
            part_id = part.get("id")
            staff_by_voice = {
                item.get("id"): item.get("staff_ref")
                for item in part.findall("./voice_list/voice_item")
            }
            for measure in part.findall("measure"):
                measure_number = measure.get("number")
                for voice in measure.findall("voice"):
                    voice_id = voice.get("voice_item_ref")
                    context = {
                        "part": part_id,
                        "staff": staff_by_voice.get(voice_id),
                        "voice": voice_id,
                        "measure": measure_number,
                    }
                    for child in voice:
                        self._read_voice_child(child, context)

        for lyrics in los.findall("lyrics"):
            context = {
                "part": lyrics.get("part_ref"),
                "staff": None,
                "voice": lyrics.get("voice_ref"),
                "measure": None,
            }
            for syllable in lyrics.findall("syllable"):
                self._read_syllable(syllable, context)

        self._los_rows = _prune_columns(self._los_rows)
        self._staff_rows = _prune_columns(self._staff_rows)

    def _read_staff(self, staff: ET.Element) -> None:
        """Record a staff's clefs, key signatures and time signatures.

        These reference spine events but are notational attributes rather than
        note/rest/lyric content, so they are kept in the ``staff_list`` store
        table only — they become neither timeline events nor claims.
        """
        staff_id = staff.get("id")
        line_number = _int_or_none(staff.get("line_number"))
        for child in staff:
            row: dict[str, Any] = {
                "staff": staff_id,
                "line_number": line_number,
                "kind": child.tag,
                "event_ref": child.get("event_ref"),
                "shape": child.get("shape"),
                "staff_step": _int_or_none(child.get("staff_step")),
                "octave_num": _int_or_none(child.get("octave_num")),
                "sharp_num": None,
                "flat_num": None,
                "num": None,
                "den": None,
            }
            sharps = child.find("sharp_num")
            if sharps is not None:
                row["sharp_num"] = _int_or_none(sharps.get("number"))
            flats = child.find("flat_num")
            if flats is not None:
                row["flat_num"] = _int_or_none(flats.get("number"))
            indication = child.find("time_indication")
            if indication is not None:
                row["num"] = _int_or_none(indication.get("num"))
                row["den"] = _int_or_none(indication.get("den"))
            self._staff_rows.append(row)

    def _read_voice_child(self, element: ET.Element, context: dict[str, Any]) -> None:
        """Turn one ``<chord>`` or ``<rest>`` into LOS event rows."""
        if element.tag not in ("chord", "rest"):
            self._logger.debug("Skipping unhandled <voice> child <%s>.", element.tag)
            return

        event_ref = element.get("event_ref")
        coordinate = self._coordinate_of(event_ref, element.tag)
        if coordinate is None:
            return

        base = dict(context)
        base["event_ref"] = event_ref
        base["instant"] = coordinate
        base.update(self._duration_fields(element.find("duration")))
        dots = element.find("augmentation_dots")
        base["augmentation_dots"] = (
            _int_or_none(dots.get("number")) if dots is not None else None
        )

        if element.tag == "rest":
            self._los_rows.append(self._los_row(base, "Rest"))
            return

        for index, notehead in enumerate(element.findall("notehead")):
            row = dict(base)
            row["notehead_index"] = index
            pitch = notehead.find("pitch")
            if pitch is not None:
                row["step"] = pitch.get("step")
                row["octave"] = _int_or_none(pitch.get("octave"))
                row["actual_accidental"] = pitch.get("actual_accidental")
            printed = notehead.find("printed_accidentals")
            if printed is not None:
                # A ``<undefined/>`` placeholder is recorded as such: the
                # document declines to state the printed accidental, which is
                # not the same as there being none.
                row["printed_accidental"] = ";".join(child.tag for child in printed)
            row["tie"] = notehead.find("tie") is not None
            self._los_rows.append(self._los_row(row, "Note"))

    def _read_syllable(self, syllable: ET.Element, context: dict[str, Any]) -> None:
        """Turn one lyric ``<syllable>`` into a LOS event row."""
        event_ref = syllable.get("start_event_ref")
        coordinate = self._coordinate_of(event_ref, "syllable")
        if coordinate is None:
            return
        row = dict(context)
        row["event_ref"] = event_ref
        row["instant"] = coordinate
        row["text"] = _text_or_none(syllable)
        row["hyphen"] = syllable.get("hyphen")
        self._los_rows.append(self._los_row(row, "Syllable"))

    @staticmethod
    def _los_row(values: dict[str, Any], event_type: str) -> dict[str, Any]:
        """Complete one LOS row to the layer's full key set."""
        row: dict[str, Any] = {
            "event_type": event_type,
            "instant": None,
            "event_ref": None,
            "part": None,
            "staff": None,
            "voice": None,
            "measure": None,
            "notehead_index": None,
            "duration_num": None,
            "duration_den": None,
            "tuplet_enter_num": None,
            "tuplet_enter_den": None,
            "tuplet_in_num": None,
            "tuplet_in_den": None,
            "augmentation_dots": None,
            "step": None,
            "octave": None,
            "actual_accidental": None,
            "printed_accidental": None,
            "tie": None,
            "text": None,
            "hyphen": None,
        }
        row.update({key: values[key] for key in values if key in row})
        return row

    @staticmethod
    def _duration_fields(duration: ET.Element | None) -> dict[str, Any]:
        """Read ``<duration>`` verbatim, plus any ``<tuplet_ratio>``.

        The ``num`` / ``den`` pair is kept as two integers rather than a
        reduced :class:`~fractions.Fraction`: ``num="4" den="4"`` is a whole
        note *written as* four quarters and must round-trip as such, and the
        column name ``duration`` is reserved by the event machinery for a
        coordinate-valued interval length.  The exact notated value is
        ``Fraction(duration_num, duration_den)``.
        """
        fields: dict[str, Any] = {
            "duration_num": None,
            "duration_den": None,
            "tuplet_enter_num": None,
            "tuplet_enter_den": None,
            "tuplet_in_num": None,
            "tuplet_in_den": None,
        }
        if duration is None:
            return fields
        fields["duration_num"] = _int_or_none(duration.get("num"))
        fields["duration_den"] = _int_or_none(duration.get("den"))
        tuplet = duration.find("tuplet_ratio")
        if tuplet is not None:
            fields["tuplet_enter_num"] = _int_or_none(tuplet.get("enter_num"))
            fields["tuplet_enter_den"] = _int_or_none(tuplet.get("enter_den"))
            fields["tuplet_in_num"] = _int_or_none(tuplet.get("in_num"))
            fields["tuplet_in_den"] = _int_or_none(tuplet.get("in_den"))
        return fields

    def _read_notational(self, root: ET.Element) -> None:
        """Read one graphical SegmentLine per ``<graphic_instance_group>``.

        An edition's pages contain a document-ordered stream of graphical
        event boxes. Each page begins a new accolade; within a page, a drop in
        ``upper_left_x`` greater than half that page's observed x-span begins
        another accolade. This follows the engraved systems rather than the
        interleaved part order within one system.
        """
        for group in root.findall("./notational/graphic_instance_group"):
            description = group.get("description") or "edition"
            role = self._claim_role(description)
            spec = _EditionSpec(
                uid=_mint_timeline_id(self, TimeUnit.pixels, role=role),
                role=role,
                description=description,
                pages=[],
                rows=[],
            )
            for instance in group.findall("graphic_instance"):
                self._read_graphic_instance(instance, spec)
            self._set_graphic_claim_coordinates(spec)
            spec.rows = _prune_columns(spec.rows)
            if spec.rows:
                keys = spec.rows[0].keys()
                for page_spec in spec.pages:
                    for segment in page_spec.segments:
                        segment.rows = [
                            {key: row[key] for key in keys} for row in segment.rows
                        ]
            self._editions.append(spec)

    @staticmethod
    def _set_graphic_claim_coordinates(spec: _EditionSpec) -> None:
        """Set each graphic claim's edition-wide x coordinate.

        Accolades concatenate within a page and pages concatenate within the
        edition, so one running offset over the document-ordered accolades
        yields the coordinate a claim carries on the edition timeline.
        """
        offset = 0
        for page_spec in spec.pages:
            for segment in page_spec.segments:
                for row in segment.rows:
                    spec.claim_coordinates.append(offset + row["start"])
                offset += max((row["end"] for row in segment.rows), default=0)

    def _read_graphic_instance(self, instance: ET.Element, spec: _EditionSpec) -> None:
        """Read one page and split its graphical events into accolades."""
        page = {
            "file_name": instance.get("file_name"),
            "position_in_group": _int_or_none(instance.get("position_in_group")),
            "encoding_format": instance.get("encoding_format"),
            "file_format": instance.get("file_format"),
            "measurement_unit": instance.get("measurement_unit"),
        }
        page_spec = _GraphicPageSpec(page=page)
        spec.pages.append(page_spec)
        graphics = list(instance.findall("graphic_event"))
        ulx_values = [
            value
            for graphic in graphics
            if (value := _float_or_none(graphic.get("upper_left_x"))) is not None
        ]
        if not ulx_values:
            return
        half_span = (max(ulx_values) - min(ulx_values)) / 2
        segment: _GraphicSegmentSpec | None = None
        previous_ulx: float | None = None

        for graphic in graphics:
            ulx = _float_or_none(graphic.get("upper_left_x"))
            if ulx is None:
                continue
            if previous_ulx is None or previous_ulx - ulx > half_span:
                segment = _GraphicSegmentSpec()
                page_spec.segments.append(segment)
            previous_ulx = ulx
            assert segment is not None

            event_ref = graphic.get("event_ref")
            if self._coordinate_of(event_ref, "graphic_event") is None:
                continue
            row = self._graphic_row(graphic, page)
            if row is None:
                continue
            segment.rows.append(row)
            spec.rows.append(row)

    def _graphic_row(
        self, graphic: ET.Element, page: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build one graphical event with integer and conditional source boxes."""
        ulx = _rounded_pixel(graphic.get("upper_left_x"))
        uly = _rounded_pixel(graphic.get("upper_left_y"))
        lrx = _rounded_pixel(graphic.get("lower_right_x"))
        lry = _rounded_pixel(graphic.get("lower_right_y"))
        if None in (ulx, uly, lrx, lry):
            self._logger.debug(
                "<graphic_event> for %r without a complete box; skipped.",
                graphic.get("event_ref"),
            )
            return None
        assert (
            ulx is not None and uly is not None and lrx is not None and lry is not None
        )
        source_coordinates = tuple(
            _float_or_none(graphic.get(name))
            for name in (
                "upper_left_x",
                "upper_left_y",
                "lower_right_x",
                "lower_right_y",
            )
        )
        assert all(value is not None for value in source_coordinates)
        source_ulx, source_uly, source_lrx, source_lry = source_coordinates
        row: dict[str, Any] = {
            "event_type": "GraphicEvent",
            "start": ulx,
            "end": lrx,
            "event_ref": graphic.get("event_ref"),
            "bbox": {"ul": {"x": ulx, "y": uly}, "lr": {"x": lrx, "y": lry}},
            "source_bbox": None,
            "file_name": page["file_name"],
            "position_in_group": page["position_in_group"],
        }
        if any(value % 1 for value in source_coordinates):
            row["source_bbox"] = {
                "ul": {"x": source_ulx, "y": source_uly},
                "lr": {"x": source_lrx, "y": source_lry},
            }
        return row

    def _read_audio(self, root: ET.Element) -> None:
        """Read one physical timeline per ``<track>``.

        ``track_event`` start times are seconds.  The referenced media file is
        never opened: its name, declared formats and its whole
        ``<track_general>`` description are recorded verbatim, whether or not
        the file exists and whether or not the declared ``file_format``
        matches the extension.
        """
        for track in root.findall("./audio/track"):
            file_name = track.get("file_name") or "track"
            role = self._claim_role(Path(file_name).stem)
            general = track.find("track_general")
            notes = None if general is None else general.find("notes")
            spec = _TrackSpec(
                uid=_mint_timeline_id(self, TimeUnit.seconds, role=role),
                role=role,
                file_name=file_name,
                attributes={
                    "encoding_format": track.get("encoding_format"),
                    "file_format": track.get("file_format"),
                },
                performers=[
                    {"name": performer.get("name"), "type": performer.get("type")}
                    for performer in track.findall("./track_general//performer")
                ],
                recordings=[
                    dict(recording.attrib)
                    for recording in track.findall(
                        "./track_general/recordings/recording"
                    )
                ],
                notes=None if notes is None else _text_or_none(notes),
                rows=[],
            )
            for indexing in track.findall("track_indexing"):
                spec.attributes.setdefault("timing_type", indexing.get("timing_type"))
                for event in indexing.findall("track_event"):
                    event_ref = event.get("event_ref")
                    if self._coordinate_of(event_ref, "track_event") is None:
                        continue
                    start_time = _float_or_none(event.get("start_time"))
                    if start_time is None:
                        self._logger.debug(
                            "<track_event> for %r without a start_time; skipped.",
                            event_ref,
                        )
                        continue
                    spec.rows.append(
                        {
                            "event_type": "TrackEvent",
                            "instant": start_time,
                            "event_ref": event_ref,
                            "file_name": file_name,
                        }
                    )
            spec.rows = _prune_columns(spec.rows)
            self._tracks.append(spec)

    def _read_structural(self, root: ET.Element) -> None:
        """Resolve ``<analysis>`` segmentations against the Petri-net places.

        The document states the mapping in two halves: a ``<segment>`` lists
        the spine events it covers, and a ``<place>`` inside a ``<petri_net>``
        binds a place of that net to a segment.  Joining them on the segment
        id gives, for each spine event, the Petri-net node that models it —
        one external-reference row per ``(segment_event, place)`` pair, so a
        segment two places name yields two rows per event.

        Nothing is read from disk: a ``<petri_net>``'s ``file_name`` becomes
        the row's access-point uri verbatim, and the ``.pnml`` file it names
        is never opened.
        """
        structural = root.find("structural")
        if structural is None:
            return

        places = self._read_petri_net_places(structural)
        for analysis in structural.findall("analysis"):
            for segment in analysis.findall("./segmentation/segment"):
                segment_id = segment.get("id")
                if segment_id is None:
                    self._logger.debug("Skipping <segment> without an id.")
                    continue
                for event in segment.findall("segment_event"):
                    event_ref = event.get("event_ref")
                    if self._coordinate_of(event_ref, "segment_event") is None:
                        continue
                    self._structural_rows.extend(
                        self._structural_row(event_ref, segment_id, place)
                        for place in places.get(segment_id) or [None]
                    )

    def _read_petri_net_places(
        self, structural: ET.Element
    ) -> dict[str, list[tuple[str, str]]]:
        """Map each segment id onto the Petri-net places that name it.

        Args:
            structural: The ``<structural>`` element.

        Returns:
            Segment id -> the ``(place_ref, petri-net file_name)`` pairs whose
            ``segment_ref`` is that segment, in document order.  A
            ``segment_ref`` naming no segment of this document simply never
            gets looked up.
        """
        places: dict[str, list[tuple[str, str]]] = {}
        for net in structural.findall("./petri_nets/petri_net"):
            file_name = net.get("file_name")
            for place in net.findall("place"):
                place_ref = place.get("place_ref")
                segment_ref = place.get("segment_ref")
                if place_ref is None or segment_ref is None or file_name is None:
                    self._logger.debug(
                        "Skipping <place> without place_ref, segment_ref or a petri-net file_name."
                    )
                    continue
                places.setdefault(segment_ref, []).append((place_ref, file_name))
        return places

    @staticmethod
    def _structural_row(
        event_ref: str, segment_id: str, place: tuple[str, str] | None
    ) -> dict[str, Any]:
        """Build one external-reference row for one spine event.

        Args:
            event_ref: The spine event id the ``<segment_event>`` names.
            segment_id: The enclosing ``<segment>``'s id.
            place: The ``(place_ref, file_name)`` pair the segment maps to, or
                ``None`` when no ``<place>`` names it.

        Returns:
            One row of the canonical external-reference schema.
        """
        if place is None:
            return {
                "event_id": event_ref,
                "external_id": segment_id,
                "access_points": [],
                "comment": _UNMAPPED_SEGMENT_COMMENT,
            }
        place_ref, file_name = place
        return {
            "event_id": event_ref,
            "external_id": place_ref,
            "access_points": [{"uri": file_name, "kind": _PETRI_NET_ACCESS_KIND}],
            "comment": segment_id,
        }

    def _log_skipped_sections(self, root: ET.Element) -> None:
        """Log every top-level section this loader does not represent."""
        skipped = [child.tag for child in root if child.tag not in _KNOWN_SECTIONS]
        if skipped:
            self._logger.debug(
                "Skipped unrepresented IEEE 1599 sections: %s", ", ".join(skipped)
            )

    def _coordinate_of(self, event_ref: str | None, origin: str) -> int | None:
        """Resolve an ``event_ref`` to its spine VTU coordinate.

        Args:
            event_ref: The referenced spine event id.
            origin: The referring element's tag, for the log message.

        Returns:
            The cumulative VTU coordinate, or ``None`` when the reference is
            missing or dangling (which is logged and skipped, never guessed).
        """
        if event_ref is None:
            self._logger.debug("<%s> without event_ref; skipped.", origin)
            return None
        coordinate = self._spine_coordinates.get(event_ref)
        if coordinate is None:
            self._logger.debug(
                "<%s> references unknown spine event %r; skipped.", origin, event_ref
            )
        return coordinate

    def _claim_role(self, label: str) -> str:
        """Sanitise *label* into a unique timeline-uid role.

        Accents are folded to their base letters, the result is lowercased,
        every run of characters outside ``[a-z0-9_]`` becomes a single ``_``,
        and a numeric suffix disambiguates a role already in use.
        """
        normalised = unicodedata.normalize("NFKD", label)
        folded = "".join(ch for ch in normalised if not unicodedata.combining(ch))
        role = re.sub(r"[^a-z0-9_]+", "_", folded.lower()).strip("_") or "unnamed"
        candidate = role
        suffix = 1
        while candidate in self._used_roles:
            suffix += 1
            candidate = f"{role}_{suffix}"
        self._used_roles.add(candidate)
        return candidate

    # endregion

    # region Curated tables

    def _store_tables(self) -> None:
        """Publish one curated table per layer into the ``DictStore``."""
        self._add_table(
            "spine",
            self._spine_rows,
            TimeUnit.ticks,
            NumberType.int,
            empty_schema=_EMPTY_CURATED_SCHEMAS["spine"],
        )
        self._add_table(
            "los",
            self._los_rows,
            TimeUnit.ticks,
            NumberType.int,
            empty_schema=_EMPTY_CURATED_SCHEMAS["los"],
        )
        self._add_table("staff_list", self._staff_rows, TimeUnit.ticks, NumberType.int)
        self._add_table(
            "notational",
            [
                {"timeline_uid": spec.uid, "edition": spec.description, **row}
                for spec in self._editions
                for row in spec.rows
            ],
            _GRAPHIC_UNIT,
            NumberType.int,
            empty_schema=_EMPTY_CURATED_SCHEMAS["notational"],
        )
        self._add_table(
            "audio",
            [
                {"timeline_uid": spec.uid, "track": spec.file_name, **row}
                for spec in self._tracks
                for row in spec.rows
            ],
            TimeUnit.seconds,
            NumberType.float,
            empty_schema=_EMPTY_CURATED_SCHEMAS["audio"],
        )
        # The structural layer states references, not coordinates; the table
        # is unit-less in substance and carries the spine's own unit only
        # because ``EventData`` demands one.
        self._add_table(
            "structural", self._structural_rows, TimeUnit.ticks, NumberType.int
        )

    def _add_table(
        self,
        name: str,
        rows: list[dict[str, Any]],
        unit: TimeUnit,
        number_type: NumberType,
        empty_schema: pa.Schema | None = None,
    ) -> None:
        """Wrap one curated row set as ``EventData`` under *name*."""
        if not rows:
            if empty_schema is None:
                return
            self._store.add(
                name,
                EventData(
                    pa.Table.from_pylist([], schema=empty_schema), unit, number_type
                ),
            )
            return
        self._store.add(name, EventData(_table_from_rows(rows), unit, number_type))

    # endregion

    # region Claims

    def _build_claim_field(self) -> None:
        """Build the document's single columnar claim field.

        Every projected event states where the spine event it references falls
        in that layer's own coordinate space, which is one synchronous claim
        ``projection@own_coordinate <-> spine@vtu``.  All three layers go into
        one field: the alignment they express is one alignment, hub-and-spoke
        around the spine, and the coordinate storage carries a unit per row,
        so ``ticks``, page-image coordinates and ``seconds`` sit side by side
        without a per-layer split.

        Claim counts run into the tens of thousands, so the field is assembled
        straight from parallel columns with
        :meth:`MatchClaimField.from_columns`; no :class:`MatchClaim` object is
        ever constructed.
        """
        metadata = MatchMetadata(
            agent=Agent(
                name=str(self._file_metadata.get("creator") or _DEFAULT_AGENT),
                type=AgentType.software,
                identifier=_AGENT_IDENTIFIER,
            ),
            certainty=1.0,
        )
        blocks: list[
            tuple[str | None, list[dict[str, Any]], list[int | float], TimeUnit]
        ] = [
            (
                self._los_uid,
                self._los_rows,
                [row["instant"] for row in self._los_rows],
                TimeUnit.ticks,
            ),
            *(
                (spec.uid, spec.rows, spec.claim_coordinates, _GRAPHIC_UNIT)
                for spec in self._editions
            ),
            *(
                (
                    spec.uid,
                    spec.rows,
                    [row["instant"] for row in spec.rows],
                    TimeUnit.seconds,
                )
                for spec in self._tracks
            ),
        ]

        timeline_ids: list[str] = []
        units: list[TimeUnit] = []
        projection: list[float] = []
        spine: list[int] = []
        for uid, rows, coordinates, unit in blocks:
            if uid is None or not rows:
                continue
            timeline_ids.extend([uid] * len(rows))
            units.extend([unit] * len(rows))
            projection.extend(coordinates)
            spine.extend(self._spine_coordinates[row["event_ref"]] for row in rows)
        if not timeline_ids:
            return

        # ``unit_a`` is a per-row sequence: the claim coordinate storage keeps
        # one unit per row, which is what lets the three layers share a field.
        self._claim_field = MatchClaimField.from_columns(
            timeline_ids,
            [self._spine_uid] * len(timeline_ids),
            projection,
            np.asarray(spine, dtype=np.int64),
            unit_a=units,
            unit_b=TimeUnit.ticks,
            metadata=metadata,
        )

    def get_field(
        self,
        selector: type[MatchClaim] | type[SemanticField[Any]],
    ) -> MatchClaimField:
        """Return the spine-referenced claim field of the whole document.

        The loader's alignment is reached through the uniform field API, the
        same way any
        :class:`~timetoalign.storage.mixins.SemanticFieldAccessMixin` surfaces
        a semantic view::

            >>> field = loader.get_field(MatchClaim)
            >>> isinstance(field, MatchClaimField)
            True

        The selector may be the :class:`MatchClaim` scalar class or its paired
        :class:`MatchClaimField` class; both resolve to the single
        ``MatchClaimField`` this loader builds, which holds every LOS, graphic
        and track event's claim against the spine.

        Args:
            selector: ``MatchClaim`` or ``MatchClaimField``.

        Returns:
            The :class:`MatchClaimField` of every ``event_ref`` in the
            document.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            TypeError: If *selector* is not ``MatchClaim`` /
                ``MatchClaimField``.
        """
        if self._claim_field is None:
            raise RuntimeError(
                "No document loaded yet. Call load() before get_field()."
            )
        if selector is MatchClaim or selector is MatchClaimField:
            return self._claim_field
        selector_name = getattr(selector, "__name__", repr(selector))
        raise TypeError(
            f"Ieee1599Loader.get_field() resolves only MatchClaim / "
            f"MatchClaimField; got {selector_name}."
        )

    # endregion

    # region Properties

    @property
    def claim_field(self) -> MatchClaimField | None:
        """The document's claim field, or ``None`` if nothing is loaded."""
        return self._claim_field

    @property
    def spine_uid(self) -> str | None:
        """The spine timeline's uid (``spine:dlt_1``), or ``None`` if unloaded."""
        return self._spine_uid

    @property
    def los_uid(self) -> str | None:
        """The LOS timeline's uid (``los:dlt_2``), or ``None`` if unloaded."""
        return self._los_uid

    @property
    def edition_uids(self) -> list[str]:
        """The graphical timeline uids, in document order."""
        return [spec.uid for spec in self._editions]

    @property
    def track_uids(self) -> list[str]:
        """The audio timeline uids, in document order."""
        return [spec.uid for spec in self._tracks]

    @property
    def timeline_uids(self) -> list[str]:
        """Every timeline uid: spine, LOS, editions, tracks."""
        uids = [uid for uid in (self._spine_uid, self._los_uid) if uid is not None]
        return uids + self.edition_uids + self.track_uids

    # endregion

    # region Domain Object Creation

    def create_bundle(self, **kwargs: Any) -> AlignmentBundle:
        """Assemble the AlignmentBundle from the parsed document.

        Each timeline occupies its own singleton TimelineGroup. The claims
        relate those groups through the spine and are added columnar with
        :meth:`~timetoalign.alignment.bundle.AlignmentBundle.add_match_claim_field`
        rather than exploded into claim objects.

        Returns:
            An ``AlignmentBundle`` with one group per timeline and the
            projection claims.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if self._spine_uid is None:
            raise RuntimeError(
                "No document loaded yet. Call load() before create_bundle()."
            )

        bundle = AlignmentBundle(name=self._name)
        for timeline in self.create_timelines():
            bundle.add_timeline(timeline, uid=timeline.id, as_group=timeline.id)
        if self._claim_field is not None:
            bundle.add_match_claim_field(self._claim_field)
        return bundle

    def create_timelines(self, id_pattern: str | None = None) -> list[Timeline]:
        """Return every timeline: spine, LOS, editions, tracks.

        Args:
            id_pattern: Optional regular expression filtering timeline uids.
        """
        timelines = [self.create_timeline(uid) for uid in self.timeline_uids]
        return [
            timeline
            for timeline in timelines
            if id_pattern is None or re.search(id_pattern, timeline.id)
        ]

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> Timeline:
        """Return one timeline by uid, building it once and caching it.

        Args:
            uid: A timeline uid.  Defaults to the spine's.

        Returns:
            The built `Timeline`.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            KeyError: If *uid* names no timeline of this document.
        """
        if self._spine_uid is None:
            raise RuntimeError(
                "No document loaded yet. Call load() before create_timeline()."
            )
        if uid is None:
            uid = self._spine_uid
        if uid in self._timeline_cache:
            return self._timeline_cache[uid]

        timeline = self._build_timeline(uid)
        self._timeline_cache[uid] = timeline
        return timeline

    def _build_timeline(self, uid: str) -> Timeline:
        """Build the timeline identified by *uid* from its curated rows."""
        if uid == self._spine_uid:
            return self._build_spine(uid)
        if uid == self._los_uid:
            return self._build_logical(
                uid, "Logically organised symbols", self._los_rows
            )
        for spec in self._editions:
            if spec.uid == uid:
                return self._build_graphical(spec)
        for spec in self._tracks:
            if spec.uid == uid:
                return self._build_physical(spec)
        raise KeyError(
            f"No timeline with uid {uid!r}. Available: {self.timeline_uids}."
        )

    def _build_spine(self, uid: str) -> DiscreteLogicalTimeline:
        """Build the spine timeline and attach its external references.

        The ``<structural>`` layer points *at* spine events, so its rows ride
        on the spine wherever it is built — through ``create_timeline`` or
        through ``create_bundle``, which reaches the same cached timeline.
        Every ``event_id`` is a spine event id resolved during the parse, so
        the addition is validated rather than trusted.
        """
        timeline = self._build_logical(uid, "Spine", self._spine_rows)
        if self._structural_rows:
            timeline.add_external_references(self._structural_rows)
        return timeline

    def _build_logical(
        self, uid: str, name: str, rows: list[dict[str, Any]]
    ) -> DiscreteLogicalTimeline:
        """Build a VTU (ticks) timeline from instant rows."""
        length = max((row["instant"] for row in rows), default=0)
        timeline = DiscreteLogicalTimeline(
            length=length,
            unit=TimeUnit.ticks,
            number_type=NumberType.int,
            uid=uid,
            name=name,
        )
        if rows:
            timeline.add_events(rows)
        return timeline

    def _build_graphical(
        self, spec: _EditionSpec
    ) -> SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]:
        """Build an edition as pages of contiguous, integer-pixel accolades.

        The edition is two levels deep: one page child per
        ``<graphic_instance>``, each holding that page's accolades. Every
        accolade keeps its source page's pixel coordinates, and each level
        concatenates its children's coordinate spaces, so graphical claims use
        the resulting edition coordinate. Because a page owns its own image
        file and pixel origin, its child accolade coordinates resolve back to
        that image without consulting any sibling page. A page-file interval
        map on the edition resolves an edition coordinate to the page image
        that contains it.
        """
        timeline: SegmentLine[SegmentLine[DiscreteGraphicalTimeline]] = SegmentLine[
            SegmentLine[DiscreteGraphicalTimeline]
        ](
            length=0,
            unit=_GRAPHIC_UNIT,
            number_type=NumberType.int,
            uid=spec.uid,
            name=spec.description,
            meta={
                "description": spec.description,
                "pages": [page_spec.page for page_spec in spec.pages],
            },
        )
        boundaries: list[int] = []
        file_names: list[str | None] = []
        for page_index, page_spec in enumerate(spec.pages, start=1):
            accolade_specs = [
                segment_spec for segment_spec in page_spec.segments if segment_spec.rows
            ]
            if not accolade_specs:
                continue
            page_uid = f"{spec.uid}_page{page_index}"
            page: SegmentLine[DiscreteGraphicalTimeline] = SegmentLine[
                DiscreteGraphicalTimeline
            ](
                length=0,
                unit=_GRAPHIC_UNIT,
                number_type=NumberType.int,
                uid=page_uid,
                name=f"page_{page_index}",
                meta={"page": page_spec.page},
            )
            for accolade_index, segment_spec in enumerate(accolade_specs, start=1):
                accolade = DiscreteGraphicalTimeline(
                    length=max(row["end"] for row in segment_spec.rows),
                    unit=_GRAPHIC_UNIT,
                    number_type=NumberType.int,
                    uid=f"{page_uid}_accolade{accolade_index}",
                    name=f"accolade_{accolade_index}",
                )
                accolade.add_events(segment_spec.rows)
                page.append_segment(accolade)
            boundaries.append(timeline.length.value)
            file_names.append(page_spec.page["file_name"])
            timeline.append_segment(page)
        if boundaries:
            timeline.add_conversion_map(
                IntervalToConstantMap(
                    boundaries=boundaries,
                    values=file_names,
                    source_unit=_GRAPHIC_UNIT,
                    name="file_name",
                )
            )
        return timeline

    def _build_physical(self, spec: _TrackSpec) -> ContinuousPhysicalTimeline:
        """Build one track's seconds timeline from its track events.

        The track's ``<track_general>`` description — its performers, its
        ``<recordings>`` attributes and its free-text ``<notes>`` — is carried
        verbatim in ``meta``; a key is present only when the document states
        it.
        """
        length = max((row["instant"] for row in spec.rows), default=0.0)
        meta: dict[str, Any] = {
            "file_name": spec.file_name,
            "performers": spec.performers,
            **spec.attributes,
        }
        if spec.recordings:
            meta["recordings"] = spec.recordings
        if spec.notes is not None:
            meta["notes"] = spec.notes
        timeline = ContinuousPhysicalTimeline(
            length=length,
            unit=TimeUnit.seconds,
            number_type=NumberType.float,
            uid=spec.uid,
            name=Path(spec.file_name).stem,
            meta=meta,
        )
        if spec.rows:
            timeline.add_events(spec.rows)
        return timeline

    # endregion

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is spine-referenced claims."""
        return ("Claims", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Extend the base rows with the IEEE 1599 layer shape."""
        rows = super()._repr_rows()
        rows.append(("File", code(self._name or "(not loaded)")))
        title = self._file_metadata.get("title")
        if title:
            rows.append(("Title", str(title)))
        rows.append(("Spine events", str(len(self._spine_rows))))
        rows.append(("LOS events", str(len(self._los_rows))))
        rows.append(("Editions", str(len(self._editions))))
        rows.append(("Tracks", str(len(self._tracks))))
        if self._claim_field is not None:
            rows.append(("Claims", str(len(self._claim_field))))
        return rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Total number of spine-referenced claims across all layers."""
        return 0 if self._claim_field is None else len(self._claim_field)

    def __repr__(self) -> str:
        if self._spine_uid is None:
            return "Ieee1599Loader(not loaded)"
        return (
            f"Ieee1599Loader(spine={len(self._spine_rows)}, "
            f"los={len(self._los_rows)}, editions={len(self._editions)}, "
            f"tracks={len(self._tracks)}, claims={len(self)})"
        )

    # endregion


# endregion
