"""MtdLoader: themes of the Musical Theme Dataset (MTD) as one AlignmentBundle.

The Musical Theme Dataset (Zalkow, Balke, Arifi-Müller and Müller, TISMIR
2020, doi:10.5334/tismir.68) gives, for each of 2,067 themes (by MTD ID)
from Barlow and Morgenstern's *A Dictionary of Musical Themes*, several
encodings of the theme, an excerpt of a commercial recording of its
movement, and an alignment between the two. One theme
``MTDxxxx_<ComposerID>_<WorkID>`` is sixteen files of the archive's ``MTD/``
folder:

* ``data_META/<stem>.json``: the theme's identifiers (``MTDID``, ``BMID``),
  its movement (``ComposerID``, ``WorkID``), the recording (the MusicBrainz
  release ``MusicBrainzID``, the disc ``NameCD`` and track ``NameTrack`` in
  the collection ``CollectionID``, the performer ``PerformanceID``), the
  excerpt's place in that track (``StartTime``, ``EndTime``, written
  ``mm:ss.s``) and ``MidiTransposition``, the semitones by which the
  recording sounds transposed against the symbolic encodings;
* ``data_ALIGNMENT/<stem>.csv``: ``"MID";"WAV"`` rows, MID in quarter notes
  of ``data_EDM-corr_MID`` written with five decimals, WAV in seconds from
  the excerpt's start; every row but the last is the onset of a note, the
  last the end of the last note;
* ``data_AUDIO/<stem>.wav``: the excerpt, ``EndTime - StartTime`` long;
* four encodings of the theme (:data:`THEME_FILES`): EDM-orig, the theme as
  the Electronic Dictionary of Musical Themes (EDM) gives it, and EDM-corr,
  MTD's correction of it, each as MIDI, CSV note list and PDF; EDM-alig,
  EDM-corr's MIDI aligned to the excerpt by the alignment (its quarter notes
  are the excerpt's seconds), as MIDI and CSV; SCORE, the engraved theme,
  as a Sibelius file and its exports to MusicXML, MIDI, CSV and PDF.

The themes of one movement (one ``ComposerID`` and ``WorkID``) lie in one CD
track. ``MtdLoader`` reads the archive in place, or an extracted folder, and
makes, for the movements or themes selected::

    bundle = MtdLoader.from_file(archive, movements=["Beethoven_Op002No1-01"]).create_bundle()

* per CD track, a seconds
  :class:`~timetoalign.timelines.types.ContinuousPhysicalTimeline`
  (``cpt_…``, number type float) of unknown length, unless the caller gives
  it (MTD does not state it), with statements of what MTD says of the
  recording;
* per theme, one timeline per file but META and the alignment, each naming
  its file: the excerpt, a child of its track at ``StartTime``; per MIDI
  file the ticks timeline
  :class:`~timetoalign.loader.midi.ScoreMidiLoader` builds of it
  (:class:`~timetoalign.timelines.types.DiscreteLogicalTimeline`,
  ``dlt_…``, number type int), holding its notes, as long as its end of
  track and locked, its quarters, seconds and bars the C-Maps and measure
  map the file states (the ticks → quarters C-Map at its resolution, the
  ticks → seconds C-Map of its tempo map, a measure map of its time
  signatures); a quarters
  :class:`~timetoalign.timelines.types.ContinuousLogicalTimeline`
  (``clt_…``, number type fraction) per MusicXML file and for the Sibelius
  file; a seconds timeline per CSV note list; a points
  :class:`~timetoalign.timelines.types.ContinuousGraphicalTimeline`
  (``cgt_…``, number type float) per PDF;
* per theme, four :class:`~timetoalign.timelines.TimelineGroup` s, one per
  encoding: EDM-orig, EDM-corr, SCORE, and EDM-alig with the excerpt;
* per alignment row but the last, a synchronous
  :class:`~timetoalign.alignment.claims.MatchClaim` from each note at that
  onset of the EDM-corr MIDI timeline to :data:`~timetoalign.core.ids.CHUNK`
  on the excerpt, anchored at the onset in ticks and the row's seconds: the
  row's quarters reach the MIDI file's ticks through its C-Map.

Each timeline is as long as its file states: a MIDI file its end of track,
the MusicXML file the end of its last measure, the excerpt
``EndTime - StartTime``. A CSV note list, a PDF and the Sibelius file state
no end: their length is unknown. The members of a group share its start and
end; where two members of known length differ, the longer takes part
through a child of it, starting at 0 and as long as the shorter one, so
that nothing is stretched. This is the case of the EDM-alig MIDI file,
which ends after the excerpt (its last note ends 0.05 seconds after the
excerpt's end, and its track one tick later), and of the MusicXML file of
some themes, whose last measure ends after its MIDI export's end of track.

Where MTD's paper states that one file was made from another, the file's
timeline says so (``dcterms:source`` naming the other's timeline): the
SCORE exports derive from the Sibelius file, an EDM encoding's CSV and PDF
from its MIDI file, EDM-corr's MIDI file from EDM-orig's, and EDM-alig's
files from EDM-corr's MIDI file. The theme's identifiers and its movement
are stated on the timelines nothing is derived from: EDM-orig's MIDI file
and the Sibelius file.

The loader reads what MTD states; it fetches nothing. Track lengths, and
MusicBrainz recording and track IDs, are the caller's to add. The notes
keep their encoding's pitches; ``MidiTransposition`` is stated as a comment
on the excerpt.

Known issues of the data, stored as MTD gives them: in MTD 1.0.0, 25
alignment rows that become claims, in 25 themes, state a second outside
their excerpt, 19 before its start (MTD1275: ``0.00000;-0.30000``) and 6
after its end (MTD1063: ``7.75000;3.32070`` on an excerpt 3.3 seconds
long). The claim holds that second on the excerpt's axis; since the excerpt
is a child of its track and a child answers only within its span, in a
bundle holding the claims without the groups a matchstamp at such a claim
names the excerpt's coordinate and no coordinate of the track.

Matchstamps of the bundle :meth:`MtdLoader.create_bundle` returns raise
:exc:`~timetoalign.timelines.UnknownLengthError`, at every claim: each of
a theme's four groups holds a member of unknown length (a CSV note list, a
PDF or the Sibelius file), and a group converts no position while a
member's length is unknown; ``within=`` does not avoid it. The claims are
read through ``get_field(MatchClaim)`` or the bundle's MatchClaim table, and
matchstamps through a bundle holding the timelines and claims without the
groups.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import Self

from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import CHUNK, NumberType, Provenance, TimeUnit
from timetoalign.core.fields import SemanticField
from timetoalign.core.ids import GROUP_PREFIX, minting_from, seed_from
from timetoalign.core.time import express_as
from timetoalign.display.html import code
from timetoalign.loader._sources import (
    Source,
    SourceLike,
    as_directory,
    as_source,
    describe_source,
    find_sources,
    join_source,
    source_record,
)
from timetoalign.loader.base import AlignmentLoader
from timetoalign.maps.convenience import QuartersToTicks, TicksToQuarters
from timetoalign.timelines.factory import mint_timeline_id
from timetoalign.timelines.groups import TimelineGroup
from timetoalign.timelines.types import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteLogicalTimeline,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.loader.midi import ScoreMidiLoader
    from timetoalign.timelines.base import Timeline


# region Constants

#: The paper describing the dataset, the basis of every statement and claim.
MTD_DOI = "https://doi.org/10.5334/tismir.68"

#: The release of the MTD whose layout the loader reads, as provenance names
#: it (the archive states no version of its own).
MTD_VERSION = "MTD 1.0.0"

#: The archive of that release, which every file a timeline names is part of.
MTD_ARCHIVE_URL = "https://www.audiolabs-erlangen.de/resources/MIR/MTD/MTD_1.0.0.zip"

#: The folder of the theme encoding whose quarter notes the alignment counts.
DEFAULT_THEME_MIDI = "data_EDM-corr_MID"

#: The MIDI folders the alignment can be matched against, with how a message
#: names the encoding.
THEME_MIDI_FOLDERS: dict[str, str] = {
    "data_EDM-corr_MID": "MTD's corrected EDM encoding of the theme",
    "data_SCORE_MID": "MTD's encoding of the engraved theme",
}

#: The copyright status of a commercial recording (rightsstatements.org).
IN_COPYRIGHT = "http://rightsstatements.org/vocab/InC/1.0/"

#: COAR access right of a recording identified but not shared.
METADATA_ONLY_ACCESS = "http://purl.org/coar/access_right/c_14cb"

_MUSICBRAINZ_RELEASE = "http://musicbrainz.org/release/"

#: ``mm:ss`` with optional decimals, as ``StartTime`` / ``EndTime`` write it.
_TIME = re.compile(r"(\d+):(\d{2}(?:\.\d+)?)")

#: A theme file's stem: ``MTD<id>_<ComposerID>_<WorkID>``.
_STEM = re.compile(r"(MTD[^_]+)_.+")


@dataclass(frozen=True)
class MtdFile:
    """One file MTD ships per theme, as the loader makes a timeline of it.

    Attributes:
        folder: Its folder in ``MTD/`` (``data_SCORE_XML``).
        extension: Its file name extension (``xml``).
        encoding: The encoding it is part of: ``EDM-orig``, ``EDM-corr``,
            ``SCORE`` or ``EDM-alig``; ``excerpt`` for the recording.
        kind: What the file is: ``MIDI``, ``CSV``, ``PDF``, ``MusicXML``,
            ``Sibelius`` or ``WAV``.
    """

    folder: str
    extension: str
    encoding: str
    kind: str

    @property
    def label(self) -> str:
        """The file as names say it: ``SCORE MusicXML``; the WAV ``excerpt``."""
        return self.encoding if self.kind == "WAV" else f"{self.encoding} {self.kind}"

    def member(self, stem: str) -> str:
        """The file of the theme *stem*, relative to ``MTD/``."""
        return f"{self.folder}/{stem}.{self.extension}"


#: The excerpt's recording, a child of its CD track.
EXCERPT_FILE = MtdFile("data_AUDIO", "wav", "excerpt", "WAV")

#: The thirteen files of a theme beside META, the alignment and the excerpt,
#: in the order :meth:`MtdLoader.create_timelines` returns their timelines:
#: encoding by encoding, as :data:`ENCODINGS` orders them.
THEME_FILES: tuple[MtdFile, ...] = (
    MtdFile("data_EDM-orig_MID", "mid", "EDM-orig", "MIDI"),
    MtdFile("data_EDM-orig_CSV", "csv", "EDM-orig", "CSV"),
    MtdFile("data_EDM-orig_IMG", "pdf", "EDM-orig", "PDF"),
    MtdFile("data_EDM-corr_MID", "mid", "EDM-corr", "MIDI"),
    MtdFile("data_EDM-corr_CSV", "csv", "EDM-corr", "CSV"),
    MtdFile("data_EDM-corr_IMG", "pdf", "EDM-corr", "PDF"),
    MtdFile("data_SCORE_SIB", "sib", "SCORE", "Sibelius"),
    MtdFile("data_SCORE_XML", "xml", "SCORE", "MusicXML"),
    MtdFile("data_SCORE_MID", "mid", "SCORE", "MIDI"),
    MtdFile("data_SCORE_CSV", "csv", "SCORE", "CSV"),
    MtdFile("data_SCORE_IMG", "pdf", "SCORE", "PDF"),
    MtdFile("data_EDM-alig_MID", "mid", "EDM-alig", "MIDI"),
    MtdFile("data_EDM-alig_CSV", "csv", "EDM-alig", "CSV"),
)

#: The files by label.
_BY_LABEL: dict[str, MtdFile] = {
    file.label: file for file in (EXCERPT_FILE, *THEME_FILES)
}

#: The encodings, in the order of a theme's groups; the EDM-alig group holds
#: the excerpt first, then EDM-alig's files.
ENCODINGS: tuple[str, ...] = ("EDM-orig", "EDM-corr", "SCORE", "EDM-alig")

#: The axis of each kind of file: timeline class, unit, number type.
_AXES: dict[str, tuple[type[Timeline], TimeUnit, NumberType]] = {
    "MIDI": (DiscreteLogicalTimeline, TimeUnit.ticks, NumberType.int),
    "MusicXML": (ContinuousLogicalTimeline, TimeUnit.quarters, NumberType.fraction),
    "Sibelius": (ContinuousLogicalTimeline, TimeUnit.quarters, NumberType.fraction),
    "CSV": (ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float),
    "WAV": (ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float),
    "PDF": (ContinuousGraphicalTimeline, TimeUnit.points, NumberType.float),
}

_SCORE_EXPORT = "exported from the Sibelius file, as MTD's paper states"
_EDM_RENDERING = "made from the encoding's MIDI file, as MTD's paper states"

#: Which file each file was made from, by label, and the statement of MTD's
#: paper saying so; no other relation between encodings is stated.
_DERIVED_FROM: dict[str, tuple[str, str]] = {
    "SCORE MusicXML": ("SCORE Sibelius", _SCORE_EXPORT),
    "SCORE MIDI": ("SCORE Sibelius", _SCORE_EXPORT),
    "SCORE CSV": ("SCORE Sibelius", _SCORE_EXPORT),
    "SCORE PDF": ("SCORE Sibelius", _SCORE_EXPORT),
    "EDM-orig CSV": ("EDM-orig MIDI", _EDM_RENDERING),
    "EDM-orig PDF": ("EDM-orig MIDI", _EDM_RENDERING),
    "EDM-corr MIDI": (
        "EDM-orig MIDI",
        "MTD's correction of the EDM encoding, as MTD's paper states",
    ),
    "EDM-corr CSV": ("EDM-corr MIDI", _EDM_RENDERING),
    "EDM-corr PDF": ("EDM-corr MIDI", _EDM_RENDERING),
    "EDM-alig MIDI": (
        "EDM-corr MIDI",
        "aligned to the excerpt by data_ALIGNMENT, as MTD's paper states",
    ),
    "EDM-alig CSV": (
        "EDM-corr MIDI",
        "aligned to the excerpt by data_ALIGNMENT, as MTD's paper states",
    ),
}

#: The timelines stating the theme's identifiers and movement: those no
#: other file is derived from.
_THEME_STATED_ON: tuple[str, ...] = ("EDM-orig MIDI", "SCORE Sibelius")

#: The pairs of group members whose known lengths are compared; the longer
#: takes part through a child as long as the shorter.
_COMPARED: tuple[tuple[str, str], ...] = (
    ("excerpt", "EDM-alig MIDI"),
    ("SCORE MusicXML", "SCORE MIDI"),
)

# endregion


# region Parsed data


@dataclass(frozen=True)
class _File:
    """One file of a theme as the loader holds it after ``load``.

    ``length`` is the file's extent on its axis (``None`` when the file
    states none). A MIDI file's ``midi`` is the
    :class:`~timetoalign.loader.midi.ScoreMidiLoader` that read it, which
    builds its timeline on ticks, as long as its end of track and locked,
    with the C-Maps the file states; ``length`` is that end of track,
    ``ppq`` its ticks per quarter, and ``cmap`` its ticks → quarters C-Map
    as the MIDI loader states it, attached nowhere: it gives the file's
    extent in quarters and places extents on the ticks.
    """

    file: MtdFile
    id: str
    length: Fraction | float | int | None
    ppq: int | None = None
    midi: ScoreMidiLoader | None = None
    cmap: TicksToQuarters | None = None

    @property
    def to_ticks(self) -> QuartersToTicks:
        """The inverse of :attr:`cmap`: quarters → the MIDI file's ticks.

        It places a value on the ticks and is attached nowhere, so it is
        built under a minter of its own and draws no ID from the loader's.
        """
        assert self.cmap is not None
        with minting_from(0):
            return self.cmap.inverse()

    @property
    def quarters(self) -> Fraction | None:
        """A MIDI file's extent in quarters, through its C-Map."""
        if self.cmap is None or self.length is None:
            return None
        return Fraction(self.cmap(self.length))


@dataclass(frozen=True)
class _Part:
    """The child through which the longer of two group members takes part.

    ``whole`` is the label of the member it is a child of, ``covers`` the
    label of the shorter member whose span it covers.
    """

    whole: str
    covers: str
    id: str
    length: Fraction | float


@dataclass(frozen=True)
class _Theme:
    """One theme as the loader holds it after ``load``."""

    stem: str
    meta: dict[str, str]
    start: Decimal
    end: Decimal
    files: dict[str, _File]
    parts: tuple[_Part, ...]
    group_ids: dict[str, str]
    claimed: str
    rows: tuple[tuple[int, float], ...]

    @property
    def movement(self) -> str:
        """The movement, ``<ComposerID>_<WorkID>``."""
        return f"{self.meta['ComposerID']}_{self.meta['WorkID']}"

    @property
    def track(self) -> tuple[str, str, str]:
        """The CD track: MusicBrainz release, disc and track as MTD names them."""
        return (self.meta["MusicBrainzID"], self.meta["NameCD"], self.meta["NameTrack"])

    @property
    def name(self) -> str:
        """The theme as names say it: ``B689 (MTD 0807)``."""
        return f"{self.meta['BMID']} (MTD {self.meta['MTDID']})"

    @property
    def excerpt(self) -> _File:
        """The excerpt."""
        return self.files[EXCERPT_FILE.label]

    def part_of(self, label: str) -> _Part | None:
        """The child through which the member *label* takes part, if any."""
        return next((part for part in self.parts if part.whole == label), None)


def _seconds(text: str, field: str, stem: str) -> Decimal:
    """The seconds a ``mm:ss.s`` string of META writes, as an exact decimal.

    Raises:
        ValueError: If *text* is not of that form.
    """
    match = _TIME.fullmatch(text.strip())
    if match is None:
        raise ValueError(
            f"{field} {text!r} of MTD theme {stem} is not a time written mm:ss or mm:ss.s"
        )
    return Decimal(match.group(1)) * 60 + Decimal(match.group(2))


def _mtd_id(stem: str) -> str:
    """The MTD ID a theme's file name starts with: ``MTD0807`` of ``MTD0807_Beethoven_…``."""
    return stem.split("_", 1)[0]


def _movement_label(movement: str) -> str:
    """``Beethoven_Op002No1-01`` as a person reads it: ``Beethoven Op002No1-01``."""
    return movement.replace("_", " ", 1)


def _movements_label(movements: Iterable[str]) -> str:
    """Movements as a person reads them: ``A``, ``A and B``, ``A, B and C``."""
    labels = [_movement_label(movement) for movement in dict.fromkeys(movements)]
    if len(labels) == 1:
        return labels[0]
    return f"{', '.join(labels[:-1])} and {labels[-1]}"


def _number_or_text(text: str) -> str:
    """A disc or track number as a person reads it: ``049`` → ``49``, ``Track01`` → ``1``."""
    digits = text.removeprefix("Track")
    return str(int(digits)) if digits.isdigit() else text


def _identifier(scheme: str, value: str) -> dict[str, Any]:
    """An identifier node in MeLODy's form, its scheme as label."""
    return {
        "type": ["melod:Identifier"],
        "rdfs:label": [scheme],
        "owl:hasValue": [value],
    }


def _item(member: str) -> dict[str, Any]:
    """The item a timeline is exemplified by: *member* of the MTD archive."""
    return {
        "type": ["melod:Item"],
        "label": f"MTD/{member}",
        "schema:isPartOf": [{"id": MTD_ARCHIVE_URL}],
    }


def _transposition(text: str) -> str:
    """MTD's ``MidiTransposition`` as the excerpt's comment states it."""
    unit = "semitone" if text.strip().lstrip("+-") == "1" else "semitones"
    return (
        f"MTD's MidiTransposition: {text.strip()} {unit}, the recording against "
        "the symbolic encodings"
    )


# endregion


# region MtdLoader


class MtdLoader(AlignmentLoader):
    """Load themes of the Musical Theme Dataset as one AlignmentBundle.

    The loader reads the MTD archive in place (its path, or a
    ``zipfile.Path`` of it) or an extracted copy (the folder holding
    ``MTD/``, or ``MTD/`` itself), the movements or themes selected only.
    Per CD track it makes a seconds timeline with an excerpt child per
    theme; per theme a timeline per file of :data:`THEME_FILES`, each naming
    its file, grouped by encoding (:data:`ENCODINGS`, the excerpt with
    EDM-alig); and per alignment row but the last a MatchClaim from each
    note at that onset of the EDM-corr MIDI timeline to
    :data:`~timetoalign.core.ids.CHUNK` on the excerpt.

    The notes keep their encoding's pitches; MTD's ``MidiTransposition``
    between the encodings and the recording is a comment on the excerpt. A
    claim whose second lies outside its excerpt (25 in MTD 1.0.0, see the
    module) is stored as given.

    The bundle's groups each hold a member of unknown length, so every
    matchstamp of the bundle :meth:`create_bundle` returns raises
    :exc:`~timetoalign.timelines.UnknownLengthError`; the claims are read
    through ``get_field(MatchClaim)`` or the bundle's MatchClaim table. In a
    bundle holding the timelines and claims without the groups, a
    matchstamp at a claim outside its excerpt names no coordinate of the
    track.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(source, movements=..., themes=...)`` — read the
       selected themes.
    2. ``loader.create_bundle(track_lengths=...)`` — assemble the
       AlignmentBundle; ``create_timelines()`` / ``create_timeline(uid)``
       give the timelines alone, ``create_groups(timelines)`` their groups.

    Every ID is minted at ``load`` from the bytes of every file of the
    themes read, so a fresh loader reading the same themes mints the same
    IDs, from the archive and from an extracted folder alike.

    A loader reads one source; ``clear()`` before reading another.
    """

    _single_source: ClassVar[bool] = True

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._themes: list[_Theme] = []
        self._track_ids: dict[tuple[str, str, str], str] = {}
        self._theme_midi: str = DEFAULT_THEME_MIDI
        self._claim_field: MatchClaimField | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Source) -> Any:
        """Not used: the MTD is read through :meth:`load`, theme by theme."""
        raise NotImplementedError(
            "MtdLoader reads the MTD through load(source, movements=..., themes=...); "
            "per-source loading is not used."
        )

    # endregion

    # region Loading

    def load(
        self,
        *sources: SourceLike,
        movements: Iterable[str] | None = None,
        themes: Iterable[str] | None = None,
        theme_midi: str = DEFAULT_THEME_MIDI,
    ) -> Self:
        """Read the selected themes of the MTD.

        Without a selection every theme is read. A theme is selected when
        its movement (``<ComposerID>_<WorkID>`` as its META file states
        them, e.g. ``Beethoven_Op002No1-01``) is among *movements* or its
        MTD ID (the file name's ``MTD0807``) among *themes*. The META files
        of every theme are read to find the movements; the other files only
        of the themes selected, each of which must hold all sixteen.

        The four MIDI files are read with
        :class:`~timetoalign.loader.midi.ScoreMidiLoader` (``parser="mido"``):
        their notes and their end of track. The MusicXML file is read with
        :class:`~timetoalign.loader.score.PartituraLoader`, which reads a
        measure's length exactly from the file's divisions: the end of its
        last measure is the sum of the lengths of its measures. The CSV
        note lists, the PDFs, the Sibelius file and the WAV are named, not
        parsed: their bytes enter the seed of the IDs only.

        Each alignment row but the last is matched to the onset it writes
        in the MIDI file of *theme_midi*: the onset within half a tick of
        the row's five-decimal quarters (``6.66667`` is ``20/3``) times the
        file's ticks per quarter, compared exactly, so a row lying exactly
        half-way between two onsets meets both and is refused. The last
        row is checked to be the end of the last note within half a tick;
        the library has no place for the aligned span's end, so it is not
        kept (its second lies 0.05 seconds past the excerpt's end in all but
        one theme of MTD 1.0.0).

        :meth:`from_file` passes *movements*, *themes* and *theme_midi* to
        this method and ``seed`` to the constructor.

        Args:
            *sources: One source: the MTD archive (a path, or a
                ``zipfile.Path`` of it), the folder holding ``MTD/``, or
                ``MTD/`` itself.
            movements: Movements to read, ``<ComposerID>_<WorkID>``.
            themes: Themes to read, by MTD ID (``MTD0807``).
            theme_midi: The MIDI folder the alignment is matched against,
                one of :data:`THEME_MIDI_FOLDERS`; by default
                ``data_EDM-corr_MID``, whose quarter notes the alignment
                counts. The claims name the notes of that file's timeline.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If *source* holds no ``MTD/data_META`` folder,
                or a selected theme lacks one of its files (every missing
                file is named).
            KeyError: If a movement or MTD ID asked for is not in *source*.
            ValueError: If *theme_midi* is no theme encoding folder; if not
                one source is given, or the loader holds one already; if a
                time of META is not ``mm:ss.s`` or the excerpt ends before
                it starts; if a MIDI file states no ticks per quarter; if an
                alignment row meets no onset of the MIDI file matched
                against, or its last row is not the end of the last note; if
                two themes of one CD track name different performers.
        """
        if theme_midi not in THEME_MIDI_FOLDERS:
            raise ValueError(
                f"theme_midi {theme_midi!r} is no theme encoding folder of the MTD. "
                f"Available: {list(THEME_MIDI_FOLDERS)}"
            )
        self._refuse_further_source(*sources)
        held = as_source(sources[0])
        root = self._mtd_folder(held)
        metas = self._read_metas(root)
        selected = self._select(metas, movements, themes, held)

        members: list[Source] = []
        for stem in selected:
            members += [
                join_source(root, f"data_META/{stem}.json"),
                join_source(root, f"data_ALIGNMENT/{stem}.csv"),
                join_source(root, EXCERPT_FILE.member(stem)),
            ]
            members += [join_source(root, file.member(stem)) for file in THEME_FILES]
        missing = [
            describe_source(member) for member in members if not member.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"MTD files missing: {missing}")
        self._start_minter(theme_midi, *members)

        self._theme_midi = theme_midi
        self._themes = [self._read_theme(root, stem, metas[stem]) for stem in selected]
        self._track_ids = {}
        for theme in self._themes:
            if theme.track not in self._track_ids:
                self._track_ids[theme.track] = mint_timeline_id(
                    TimeUnit.seconds, self._id_minter
                )
        self._refuse_two_performers()
        self._claim_field = self._build_claim_field()
        self._sources = [held]
        self._source_metadata = [
            {
                **source_record(held),
                "themes": [_mtd_id(stem) for stem in selected],
                "theme_midi": theme_midi,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        return self

    @staticmethod
    def _mtd_folder(source: Source) -> Source:
        """The ``MTD/`` folder of *source*: archive, folder holding it, or itself.

        Raises:
            FileNotFoundError: If *source* holds no ``MTD/data_META``.
        """
        folder = as_directory(source)
        for candidate in (join_source(folder, "MTD"), folder):
            if join_source(candidate, "data_META").is_dir():
                return candidate
        raise FileNotFoundError(
            f"{describe_source(source)} holds no MTD/data_META folder: neither an MTD "
            "archive, nor a folder holding MTD/, nor MTD/ itself."
        )

    @staticmethod
    def _read_metas(root: Source) -> dict[str, dict[str, str]]:
        """Every theme's META file, by stem."""
        metas: dict[str, dict[str, str]] = {}
        for member in find_sources(join_source(root, "data_META"), "*.json"):
            stem = member.name.removesuffix(".json")
            if _STEM.fullmatch(stem) is None:
                continue
            with member.open("rb") as handle:
                metas[stem] = json.load(handle)
        return metas

    @staticmethod
    def _select(
        metas: Mapping[str, Mapping[str, str]],
        movements: Iterable[str] | None,
        themes: Iterable[str] | None,
        source: Source,
    ) -> list[str]:
        """The stems of the themes selected, in file-name order.

        Raises:
            KeyError: If a movement or MTD ID asked for is not in *metas*.
        """
        if movements is None and themes is None:
            return sorted(metas)
        by_movement: dict[str, list[str]] = {}
        by_id: dict[str, str] = {}
        for stem, meta in metas.items():
            by_movement.setdefault(f"{meta['ComposerID']}_{meta['WorkID']}", []).append(
                stem
            )
            by_id[_mtd_id(stem)] = stem
        wanted_movements = list(movements or ())
        wanted_ids = list(themes or ())
        unknown_movements = [m for m in wanted_movements if m not in by_movement]
        unknown_ids = [i for i in wanted_ids if i not in by_id]
        if unknown_movements or unknown_ids:
            raise KeyError(
                f"{describe_source(source)} holds no "
                + "; no ".join(
                    part
                    for part in (
                        f"movement {unknown_movements}" if unknown_movements else "",
                        f"theme {unknown_ids}" if unknown_ids else "",
                    )
                    if part
                )
                + f". It holds {len(by_movement)} movements (e.g. "
                f"{sorted(by_movement)[:3]}) and {len(by_id)} themes (e.g. "
                f"{sorted(by_id)[:3]})."
            )
        chosen = {
            stem for movement in wanted_movements for stem in by_movement[movement]
        }
        chosen |= {by_id[mtd_id] for mtd_id in wanted_ids}
        return sorted(chosen)

    def _read_theme(self, root: Source, stem: str, meta: dict[str, str]) -> _Theme:
        """Read one theme's files and alignment; mint its IDs.

        The MIDI files are read by their own loaders (:meth:`_read_midi`),
        which mint their notes' IDs. This loader mints, in this order: the
        excerpt, the timelines of :data:`THEME_FILES` (in that order), the
        children through which a longer member takes part (in the order of
        :data:`_COMPARED`), the groups (in the order of :data:`ENCODINGS`).
        The maps of the MIDI files' timelines are minted when
        :meth:`create_timelines` builds them.
        """
        start = _seconds(meta["StartTime"], "StartTime", stem)
        end = _seconds(meta["EndTime"], "EndTime", stem)
        if end <= start:
            raise ValueError(
                f"MTD theme {stem} ends ({meta['EndTime']}) before it starts "
                f"({meta['StartTime']})"
            )
        midis = {
            file.label: self._read_midi(
                join_source(root, file.member(stem)), file.member(stem)
            )
            for file in THEME_FILES
            if file.kind == "MIDI"
        }
        score_length = self._read_score_length(
            join_source(root, _BY_LABEL["SCORE MusicXML"].member(stem))
        )

        def minted(file: MtdFile) -> str:
            return mint_timeline_id(_AXES[file.kind][1], self._id_minter)

        files = {
            EXCERPT_FILE.label: _File(
                EXCERPT_FILE, minted(EXCERPT_FILE), float(end - start)
            )
        }
        for file in THEME_FILES:
            if file.kind == "MIDI":
                midi = midis[file.label]
                files[file.label] = _File(
                    file,
                    minted(file),
                    midi.end_of_track,
                    midi.ticks_per_beat,
                    midi,
                    _quarters_map(midi),
                )
            elif file.kind == "MusicXML":
                files[file.label] = _File(file, minted(file), score_length)
            else:
                files[file.label] = _File(file, minted(file), None)

        # Exact extents, in quarters or seconds (EDM-alig's quarters are the
        # excerpt's seconds): the excerpt's as META writes its decimals,
        # never its float; a MIDI file's through its C-Map. They decide
        # which member is longer, and the child of the longer takes the
        # shorter's exact extent on the longer's axis: a MIDI file's in
        # ticks through the C-Map, the excerpt's in its float (2.8 seconds
        # is 1344 ticks at 480 per quarter, never the float's dyadic).
        extents: dict[str, Fraction | None] = {
            label: (
                held.quarters
                if held.midi is not None
                else None if held.length is None else Fraction(held.length)
            )
            for label, held in files.items()
        }
        extents[EXCERPT_FILE.label] = Fraction(end - start)
        parts: list[_Part] = []
        for first, second in _COMPARED:
            one, other = extents[first], extents[second]
            if one is None or other is None or one == other:
                continue
            longer, shorter = (first, second) if one > other else (second, first)
            whole = files[longer]
            parts.append(
                _Part(
                    whole=longer,
                    covers=shorter,
                    id=mint_timeline_id(_AXES[whole.file.kind][1], self._id_minter),
                    length=_span(whole, min(one, other)),
                )
            )
        group_ids = {
            encoding: self._id_minter.mint(GROUP_PREFIX) for encoding in ENCODINGS
        }

        claimed = next(
            file.label for file in THEME_FILES if file.folder == self._theme_midi
        )
        alignment = join_source(root, f"data_ALIGNMENT/{stem}.csv")
        rows = self._match_rows(
            self._read_alignment(alignment),
            files[claimed],
            alignment,
            join_source(root, files[claimed].file.member(stem)),
            self._theme_midi,
        )
        return _Theme(
            stem=stem,
            meta=meta,
            start=start,
            end=end,
            files=files,
            parts=tuple(parts),
            group_ids=group_ids,
            claimed=claimed,
            rows=rows,
        )

    def _read_midi(self, midi: Source, member: str) -> ScoreMidiLoader:
        """The :class:`~timetoalign.loader.midi.ScoreMidiLoader` that read a MIDI file.

        The file is read with ``parser="mido"``, which pairs note-ons with
        note-offs on the file's own axis, ticks, and states its end of
        track; the loader mints the notes' IDs from the file, or, where
        this loader was given a ``seed``, from a seed of it and *member*,
        the file's path below ``MTD/`` (``data_EDM-orig_MID/<stem>.mid``),
        so the IDs do not depend on where the dataset lies. The MIDI
        loader's extraction of the file is this loader's too.

        Raises:
            ValueError: If the file states no ticks per quarter, or holds a
                note without onset, end or pitch.
        """
        from timetoalign.loader.midi import ScoreMidiLoader

        seed = None if self._seed is None else seed_from(self._seed, member)
        midi_loader = ScoreMidiLoader(parser="mido", seed=seed).load(midi)
        if not midi_loader.ticks_per_beat or midi_loader.end_of_track is None:
            raise ValueError(f"{describe_source(midi)} states no ticks per quarter")
        table = _notes(midi_loader)
        if any(table.column(name).null_count for name in ("start", "end", "pitch")):
            raise ValueError(
                f"{describe_source(midi)} holds a note without onset, end or pitch"
            )
        self._adopt_extractions(midi_loader)
        return midi_loader

    @staticmethod
    def _read_score_length(musicxml: Source) -> Fraction | None:
        """The end of the MusicXML file's last measure, in exact quarters.

        Read with :class:`~timetoalign.loader.score.PartituraLoader`, whose
        timeline's measure map states each measure's actual length from the
        file's divisions; their sum is where the last measure ends. A file
        whose measures leave a length unstated gives ``None``.
        """
        from timetoalign.loader.score import PartituraLoader

        timeline = PartituraLoader().load(musicxml).create_timeline()
        measure_map = timeline.measure_map
        if measure_map is None:
            return None
        length = measure_map.total_actual_length
        return None if length is None else Fraction(length)

    @staticmethod
    def _read_alignment(alignment: Source) -> list[tuple[str, str]]:
        """The ``(MID, WAV)`` strings of an alignment file, header dropped."""
        with alignment.open("rb") as handle:
            text = io.TextIOWrapper(handle, encoding="utf-8").read()
        lines = list(csv.reader(text.splitlines(), delimiter=";"))
        if not lines or [cell.strip() for cell in lines[0]] != ["MID", "WAV"]:
            raise ValueError(
                f'{describe_source(alignment)} does not start with the header "MID";"WAV"'
            )
        return [(row[0], row[1]) for row in lines[1:] if row]

    @staticmethod
    def _match_rows(
        rows: list[tuple[str, str]],
        midi_file: _File,
        alignment: Source,
        midi: Source,
        theme_midi: str,
    ) -> tuple[tuple[int, float], ...]:
        """Each row but the last as ``(onset in ticks, seconds)``.

        A row's quarters, read as the exact decimal it writes, times the
        file's ticks per quarter (the scalar of its C-Map, unrounded: the
        product is compared, never placed on the ticks) meet the onset
        within half a tick (``|tick - quarters * ppq| <= 1/2``), so a row
        exactly half-way between two onsets meets both. The last row is the
        end of the last note within half a tick.

        Raises:
            ValueError: If a row meets no onset or two, or the last row is
                not the end of the last note; the message names the folder
                the encoding was read from.
        """
        reading = (
            f" The alignment counts the quarters of {DEFAULT_THEME_MIDI}; this theme "
            f"encoding was read from {theme_midi}."
        )
        ppq = midi_file.ppq
        assert ppq is not None and midi_file.midi is not None
        if len(rows) < 2:
            raise ValueError(f"{describe_source(alignment)} holds fewer than two rows")
        notes = _notes(midi_file.midi)
        if not notes.num_rows:
            raise ValueError(f"{describe_source(midi)} holds no note." + reading)
        onsets = np.unique(
            pc.struct_field(notes.column("start"), "numerator").to_numpy()
        )
        matched: list[tuple[int, float]] = []
        for mid, wav in rows[:-1]:
            ticks = _decimal(mid, alignment) * ppq
            # |onset - n/d| <= 1/2, in integers.
            near = 2 * np.abs(onsets * ticks.denominator - ticks.numerator)
            hits = onsets[near <= ticks.denominator]
            if len(hits) != 1:
                raise ValueError(
                    f"The alignment row {mid};{wav} of {describe_source(alignment)} meets "
                    f"{len(hits)} onsets of {describe_source(midi)} within half a tick "
                    f"({ppq} ticks per quarter): "
                    f"{[str(Fraction(int(t), ppq)) for t in hits]}." + reading
                )
            _decimal(wav, alignment)
            matched.append((int(hits[0]), float(wav)))
        last_mid = rows[-1][0]
        last_end = pc.max(pc.struct_field(notes.column("end"), "numerator")).as_py()
        last = _decimal(last_mid, alignment) * ppq
        if abs(last_end - last) > Fraction(1, 2):
            raise ValueError(
                f"The last alignment row {last_mid} of {describe_source(alignment)} is not "
                f"the end of the last note of {describe_source(midi)}, "
                f"{Fraction(last_end, ppq)} quarters, within half a tick." + reading
            )
        return tuple(matched)

    def _refuse_two_performers(self) -> None:
        """Refuse two themes of one CD track naming different performers."""
        performers: dict[tuple[str, str, str], tuple[str, str]] = {}
        for theme in self._themes:
            named = performers.setdefault(
                theme.track, (theme.stem, theme.meta["PerformanceID"])
            )
            if named[1] != theme.meta["PerformanceID"]:
                raise ValueError(
                    f"MTD themes {named[0]} and {theme.stem} lie in one CD track {theme.track} "
                    f"but name the performers {named[1]!r} and "
                    f"{theme.meta['PerformanceID']!r}"
                )

    def _build_claim_field(self) -> MatchClaimField:
        """One synchronous claim per note at an aligned onset, columnar.

        Per theme the notes of the MIDI file matched against, in the order
        its loader holds them, are joined to the alignment rows by onset
        tick, row by row; each claim names the note at its onset in ticks
        and the row's second on the excerpt.
        """
        encoding_ids: list[str] = []
        excerpt_ids: list[str] = []
        note_ids: list[pa.Array] = []
        ticks: list[pa.Array] = []
        seconds: list[pa.Array] = []
        axes: dict[str, tuple[TimeUnit, NumberType]] = {}
        for theme in self._themes:
            encoding = theme.files[theme.claimed]
            excerpt = theme.excerpt
            assert encoding.midi is not None
            axes[encoding.id] = (TimeUnit.ticks, NumberType.int)
            axes[excerpt.id] = (TimeUnit.seconds, NumberType.float)
            notes = _notes(encoding.midi)
            held = pa.Table.from_arrays(
                [
                    notes.column("id").combine_chunks(),
                    pc.struct_field(
                        notes.column("start"), "numerator"
                    ).combine_chunks(),
                    pa.array(np.arange(notes.num_rows), pa.int64()),
                ],
                names=["id", "tick", "note"],
            )
            aligned = pa.Table.from_arrays(
                [
                    pa.array([tick for tick, _ in theme.rows], pa.int64()),
                    pa.array([second for _, second in theme.rows], pa.float64()),
                    pa.array(np.arange(len(theme.rows)), pa.int64()),
                ],
                names=["tick", "second", "row"],
            )
            joined = held.join(aligned, "tick", join_type="inner").sort_by(
                [("row", "ascending"), ("note", "ascending")]
            )
            encoding_ids += [encoding.id] * joined.num_rows
            excerpt_ids += [excerpt.id] * joined.num_rows
            note_ids.append(joined.column("id").combine_chunks())
            ticks.append(joined.column("tick").combine_chunks())
            seconds.append(joined.column("second").combine_chunks())
        return MatchClaimField.from_columns(
            encoding_ids,
            excerpt_ids,
            pa.concat_arrays(note_ids) if note_ids else pa.array([], pa.string()),
            pa.repeat(pa.scalar(CHUNK), len(encoding_ids)),
            pa.concat_arrays(ticks) if ticks else pa.array([], pa.int64()),
            pa.concat_arrays(seconds) if seconds else pa.array([], pa.float64()),
            timelines=axes,
            provenance=Provenance(
                method="imported",
                basis=(MTD_DOI,),
                note=f"{MTD_VERSION} data_ALIGNMENT",
            ),
            minter=self._id_minter,
        )

    # endregion

    # region Field access

    def get_field(
        self,
        selector: type[MatchClaim] | type[SemanticField[Any]],
    ) -> MatchClaimField:
        """Return the field of the themes' MatchClaims.

        Args:
            selector: ``MatchClaim`` or ``MatchClaimField``.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            TypeError: If *selector* is neither.
        """
        claims = self._loaded()
        if selector is MatchClaim or selector is MatchClaimField:
            return claims
        name = getattr(selector, "__name__", repr(selector))
        raise TypeError(
            f"MtdLoader.get_field() resolves only MatchClaim / MatchClaimField; got {name}."
        )

    def _loaded(self) -> MatchClaimField:
        """The claim field; refuses a loader that has read nothing."""
        if self._claim_field is None:
            raise RuntimeError("No MTD themes loaded yet. Call load() first.")
        return self._claim_field

    # endregion

    # region Properties

    @property
    def movements(self) -> list[str]:
        """The movements read, ``<ComposerID>_<WorkID>``, in order of their first theme."""
        return list(dict.fromkeys(theme.movement for theme in self._themes))

    @property
    def themes(self) -> list[str]:
        """The MTD IDs of the themes read (``MTD0807``), in file-name order."""
        return [_mtd_id(theme.stem) for theme in self._themes]

    @property
    def tracks(self) -> list[tuple[str, str, str]]:
        """The CD tracks read: ``(MusicBrainzID, NameCD, NameTrack)`` as MTD writes them.

        These are the keys of ``track_lengths`` in :meth:`create_bundle`.
        """
        return list(self._track_ids)

    @property
    def theme_midi(self) -> str:
        """The MIDI folder the alignment was matched against."""
        return self._theme_midi

    # endregion

    # region Domain Object Creation

    def create_bundle(
        self,
        *,
        track_lengths: Mapping[tuple[str, str, str], float | None] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> "AlignmentBundle":
        """Assemble the AlignmentBundle of the themes read.

        The bundle holds every timeline of :meth:`create_timelines` (the CD
        tracks with their excerpts as children, then each theme's file
        timelines), every group of :meth:`create_groups` over them, and the
        MatchClaims of the alignments, columnar.

        Note:
            Every group holds a member of unknown length, so every
            matchstamp of this bundle raises
            :exc:`~timetoalign.timelines.UnknownLengthError`, ``within=``
            or not. The claims are read through ``get_field(MatchClaim)``
            or the bundle's MatchClaim table; a bundle holding the
            timelines and claims without the groups gives matchstamps.

        Args:
            track_lengths: Lengths in seconds of CD tracks, keyed as
                :attr:`tracks` lists them; a track not named, or named with
                ``None``, has an unknown length.
            name: The bundle's name; by default the movement's
                (``Beethoven Op002No1-01``), or ``MTD, <n> movements``.
            **kwargs: None is taken; any is refused.

        Returns:
            The AlignmentBundle.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            KeyError: If *track_lengths* names a track not read.
            TypeError: If another keyword is given.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if kwargs:
            raise TypeError(
                f"MtdLoader.create_bundle() takes track_lengths and name; got {sorted(kwargs)}"
            )
        claims = self._loaded()
        if name is None:
            movements = self.movements
            name = (
                _movement_label(movements[0])
                if len(movements) == 1
                else f"MTD, {len(movements)} movements"
            )
        bundle = AlignmentBundle(name=name)
        timelines = self.create_timelines(track_lengths=track_lengths)
        for timeline in timelines:
            bundle.add_timeline(timeline)
        for group in self.create_groups(timelines):
            bundle.add_group(group)
        bundle.add_match_claim_field(claims)
        return bundle

    def create_timelines(
        self,
        id_pattern: str | None = None,
        *,
        track_lengths: Mapping[tuple[str, str, str], float | None] | None = None,
    ) -> list["Timeline"]:
        """The root timelines: the CD tracks', then per theme its files'.

        A CD track holds its themes' excerpts as children. Per theme follow
        the timelines of :data:`THEME_FILES`, in that order (13 per theme).
        Where the longer of two group members takes part through a child,
        that child is a child of its member's timeline (an excerpt's child
        of the excerpt). Call :meth:`create_groups` with the timelines
        returned to group them.

        A MIDI file's timeline is the one its MIDI loader's store builds,
        as long as the file's end of track and locked. The IDs of its maps
        (the ticks → quarters and ticks → seconds C-Maps, the measure map,
        and the ``ChainMap`` reading it with its two links) are drawn from
        this loader's minter when the timeline is built, six per MIDI
        file, after every ID ``load`` mints; a second call draws new ones,
        as a second
        :meth:`ScoreMidiLoader.create_timeline
        <timetoalign.loader.midi.MidiLoader.create_timeline>` does.

        A timeline derived from another (``dcterms:source``) refuses
        ``to_dict()`` when written alone: its derivation names a timeline
        outside its one-timeline manifest. The same timeline taken from
        the bundle of :meth:`create_bundle` writes while that bundle is
        held, the reference left out.

        Args:
            id_pattern: Optional regular expression the IDs must match.
            track_lengths: Lengths in seconds of CD tracks (see
                :meth:`create_bundle`).

        Returns:
            The CD tracks' timelines, then the file timelines theme by
            theme.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            KeyError: If *track_lengths* names a track not read.
        """
        self._loaded()
        lengths = dict(track_lengths or {})
        unknown = [track for track in lengths if track not in self._track_ids]
        if unknown:
            raise KeyError(
                f"track_lengths names tracks not read: {unknown}. Available: {self.tracks}"
            )
        timelines: list[Timeline] = [
            self._make_track(track, lengths.get(track)) for track in self._track_ids
        ]
        for theme in self._themes:
            timelines += [self._make_file(theme, file) for file in THEME_FILES]
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_groups(
        self, timelines: Sequence["Timeline"] | None = None
    ) -> list[TimelineGroup]:
        """Group each theme's timelines by encoding, members as their timelines.

        Per theme, in the order of :data:`ENCODINGS`, one group named after
        the theme and the encoding (``B689 (MTD 0807), EDM-corr``): EDM-orig
        {MIDI, CSV, PDF}, EDM-corr {MIDI, CSV, PDF}, SCORE {Sibelius,
        MusicXML, MIDI, CSV, PDF} and EDM-alig {excerpt, MIDI, CSV}. A member
        is its file's timeline, or the child through which that timeline
        takes part where it is the longer of two members of known length.

        Each group holds a member of unknown length (a CSV note list, a PDF
        or the Sibelius file), so a group converts no position, and a
        matchstamp of a bundle holding these groups raises
        :exc:`~timetoalign.timelines.UnknownLengthError`; the claims are
        read through ``get_field(MatchClaim)``. A group taken alone whose
        member is derived from a timeline outside it (EDM-corr's MIDI from
        EDM-orig's, EDM-alig's files from EDM-corr's MIDI) refuses
        ``to_dict()``, its derivation naming a timeline outside its
        manifest; the bundle writes it.

        Args:
            timelines: The timelines of :meth:`create_timelines` to group
                (their members are taken from them, so that groups and
                timelines are the same objects). Without them, the
                timelines are made afresh, every CD track of unknown length.

        Returns:
            The groups, theme by theme.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            KeyError: If *timelines* lack a timeline a group holds.
        """
        self._loaded()
        if timelines is None:
            timelines = self.create_timelines()
        roots = {timeline.id: timeline for timeline in timelines}

        def root(uid: str) -> Timeline:
            if uid not in roots:
                raise KeyError(
                    f"The timelines given lack {uid!r}, which a group holds. "
                    f"Available: {list(roots)}"
                )
            return roots[uid]

        groups: list[TimelineGroup] = []
        for theme in self._themes:
            for encoding in ENCODINGS:
                members: list[Timeline] = []
                labels = [
                    file.label for file in THEME_FILES if file.encoding == encoding
                ]
                if encoding == "EDM-alig":
                    labels.insert(0, EXCERPT_FILE.label)
                for label in labels:
                    if label == EXCERPT_FILE.label:
                        timeline = root(self._track_ids[theme.track]).get_child(
                            theme.excerpt.id
                        )
                    else:
                        timeline = root(theme.files[label].id)
                    part = theme.part_of(label)
                    members.append(
                        timeline if part is None else timeline.get_child(part.id)
                    )
                groups.append(
                    TimelineGroup(
                        id=theme.group_ids[encoding],
                        name=f"{theme.name}, {encoding}",
                        timelines=members,
                    )
                )
        return groups

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """One timeline of :meth:`create_timelines` by its ID.

        Args:
            uid: A CD track's or a theme file's timeline ID.
            **kwargs: ``track_lengths``, as :meth:`create_timelines` takes
                it; any other keyword is refused.

        Returns:
            The timeline with the ID, made afresh. Written alone, a
            timeline derived from another refuses ``to_dict()`` (see
            :meth:`create_timelines`).

        Raises:
            KeyError: If no timeline has the ID.
            RuntimeError: If ``load()`` has not been called yet.
            TypeError: If another keyword is given.
        """
        unknown = sorted(set(kwargs) - {"track_lengths"})
        if unknown:
            raise TypeError(
                f"MtdLoader.create_timeline() takes uid and track_lengths; got {unknown}"
            )
        timelines = self.create_timelines(track_lengths=kwargs.get("track_lengths"))
        for timeline in timelines:
            if timeline.id == uid:
                return timeline
        raise KeyError(
            f"No timeline with uid {uid!r}. Available: {[t.id for t in timelines]}"
        )

    def _make_track(
        self, track: tuple[str, str, str], length: float | None
    ) -> ContinuousPhysicalTimeline:
        """A CD track's seconds timeline: its statements, its excerpts as children."""
        themes = [theme for theme in self._themes if theme.track == track]
        first = themes[0].meta
        release, disc, number = track
        timeline = ContinuousPhysicalTimeline(
            length=length,
            unit=TimeUnit.seconds,
            number_type=NumberType.float,
            uid=self._track_ids[track],
            name=(
                f"{_movements_label(theme.movement for theme in themes)}, "
                f"{first['PerformanceID']} "
                f"(CD {_number_or_text(disc)}, track {_number_or_text(number)})"
            ),
        )
        provenance = Provenance(
            method="imported",
            basis=(MTD_DOI,),
            note=f"{MTD_VERSION} data_META of "
            + ", ".join(_mtd_id(theme.stem) for theme in themes),
        )
        pairs: list[tuple[str, Any]] = [("type", "schema:MusicRecording")]
        pairs += [
            ("schema:recordingOf", {"label": _movement_label(movement)})
            for movement in dict.fromkeys(theme.movement for theme in themes)
        ]
        pairs += [
            ("schema:inAlbum", {"id": f"{_MUSICBRAINZ_RELEASE}{release}"}),
            (
                "melod:hasIdentifier",
                _identifier("MTD CollectionID", first["CollectionID"]),
            ),
            ("melod:hasIdentifier", _identifier("MTD NameCD", disc)),
            ("melod:hasIdentifier", _identifier("MTD NameTrack", number)),
            ("rel:prf", {"label": first["PerformanceID"]}),
            ("dcterms:rights", {"id": IN_COPYRIGHT}),
            ("dcterms:accessRights", {"id": METADATA_ONLY_ACCESS}),
        ]
        for property, value in pairs:
            timeline.add_statement(property, value, provenance=provenance)
        for theme in themes:
            timeline.add_child(self._make_file(theme, EXCERPT_FILE), float(theme.start))
        return timeline

    def _make_file(self, theme: _Theme, file: MtdFile) -> Timeline:
        """The timeline of one file of a theme, with the child it takes part through.

        Every such timeline names its file and states what it was made
        from; a MIDI file's holds its notes; the derivation roots state
        the theme; the excerpt states MTD's transposition and the
        recording's rights.
        """
        held = theme.files[file.label]
        cls, unit, number_type = _AXES[file.kind]
        name = (
            f"Theme {theme.name}"
            if file is EXCERPT_FILE
            else f"{theme.name}, {file.label}"
        )
        if held.midi is None:
            timeline = cls(
                length=held.length,
                unit=unit,
                number_type=number_type,
                uid=held.id,
                name=name,
            )
        else:
            # The MIDI loader's store builds the file's timeline on its
            # ticks, as long as its end of track and locked, with the C-Maps
            # (quarters, seconds) and the measure map the file states. Their
            # IDs come from this loader's minter: two MIDI files of one
            # selection may be equal byte for byte (MTD8510 and MTD8522),
            # and their own loaders, seeded from their content, would give
            # the maps of both one ID, which a bundle refuses.
            timeline = held.midi.store.create_timeline(
                uid=held.id, minter=self._id_minter
            )
            timeline.name = name
        member = file.member(theme.stem)
        timeline.add_statement(
            "lrmoo:R7i_is_exemplified_by",
            _item(member),
            provenance=Provenance(
                method="imported",
                basis=(MTD_DOI,),
                note=f"{MTD_VERSION} MTD/{member}",
            ),
        )
        if file.label in _DERIVED_FROM:
            source, statement = _DERIVED_FROM[file.label]
            timeline.add_statement(
                "dcterms:source",
                {"id": f"#{theme.files[source].id}"},
                provenance=Provenance(
                    method="imported", basis=(MTD_DOI,), note=statement
                ),
            )
        meta = Provenance(
            method="imported",
            basis=(MTD_DOI,),
            note=f"{MTD_VERSION} data_META/{theme.stem}.json",
        )
        pairs: list[tuple[str, Any]] = []
        if file.label in _THEME_STATED_ON:
            pairs += [
                ("schema:isPartOf", {"label": _movement_label(theme.movement)}),
                ("melod:hasIdentifier", _identifier("MTD", theme.meta["MTDID"])),
                (
                    "melod:hasIdentifier",
                    _identifier("Barlow and Morgenstern", theme.meta["BMID"]),
                ),
            ]
        if file is EXCERPT_FILE:
            pairs += [
                ("rdfs:comment", _transposition(theme.meta["MidiTransposition"])),
                ("dcterms:rights", {"id": IN_COPYRIGHT}),
                ("dcterms:accessRights", {"id": METADATA_ONLY_ACCESS}),
            ]
        for property, value in pairs:
            timeline.add_statement(property, value, provenance=meta)
        part = theme.part_of(file.label)
        if part is not None:
            covered = theme.files[part.covers].file
            child = cls(
                length=part.length,
                unit=unit,
                number_type=number_type,
                uid=part.id,
                name=f"{timeline.name}, span of the {covered.label}",
            )
            timeline.add_child(child, express_as(0, number_type))
        return timeline

    # endregion

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is MatchClaims, not store events."""
        return ("Claims", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """The base rows, then the movements, themes and tracks read."""
        rows = super()._repr_rows()
        rows.append(
            ("Movements", ", ".join(code(m) for m in self.movements) or "(none)")
        )
        rows.append(("Themes", str(len(self._themes))))
        rows.append(("CD tracks", str(len(self._track_ids))))
        rows.append(("Claims matched against", code(self._theme_midi)))
        return rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """The number of MatchClaims read."""
        return len(self._claim_field) if self._claim_field is not None else 0

    def __repr__(self) -> str:
        return (
            f"MtdLoader(movements={len(self.movements)}, themes={len(self._themes)}, "
            f"claims={len(self)})"
        )

    # endregion


def _notes(midi: ScoreMidiLoader) -> pa.Table:
    """The Note rows of the events a MIDI loader read, in its order."""
    table = midi.events.table
    return table.filter(pc.equal(table.column("event_type"), pa.scalar("Note")))


def _quarters_map(midi: ScoreMidiLoader) -> TicksToQuarters:
    """The ticks → quarters C-Map a MIDI loader states of its file, unattached.

    The map its timeline carries, built anew under a minter of its own, so
    that it draws no ID from any loader's minter; the store is built
    before, outside that minter.
    """
    store = midi.store
    with minting_from(0):
        cmap = store.get_cmaps()["quarters"]
    assert isinstance(cmap, TicksToQuarters)
    return cmap


def _span(whole: _File, extent: Fraction) -> Fraction | float | int:
    """*extent* (quarters or seconds) as a length on *whole*'s axis.

    On a MIDI file's ticks through the inverse of its C-Map (quarters →
    ticks), which places the extent on the ticks: an extent ending between
    two ticks takes the nearer one, a tie the even one (``3.33`` quarters
    at 480 per quarter, ``1598.4`` ticks, is ``1598``); on the excerpt its
    float; on MusicXML the exact quarters.
    """
    if whole.ppq is None:
        return express_as(extent, _AXES[whole.file.kind][2])
    return int(whole.to_ticks(extent))


def _decimal(text: str, source: Source) -> Fraction:
    """The exact value of a decimal number an alignment file writes.

    Raises:
        ValueError: If *text* is no decimal number.
    """
    try:
        return Fraction(Decimal(text.strip()))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(
            f"{text!r} in {describe_source(source)} is no number"
        ) from error


# endregion
