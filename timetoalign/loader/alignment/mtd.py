"""MtdLoader: themes of the Musical Theme Dataset (MTD) as one AlignmentBundle.

The Musical Theme Dataset (Zalkow, Balke, Arifi-Müller and Müller, TISMIR
2020, doi:10.5334/tismir.68) gives, for each of 2,067 themes (by MTD ID)
from Barlow and Morgenstern's *A Dictionary of Musical Themes*, an encoding of
the theme, an excerpt of a commercial recording of its movement, and an
alignment between the two. One theme ``MTDxxxx_<ComposerID>_<WorkID>`` is three files of the
archive's ``MTD/`` folder:

* ``data_META/<stem>.json``: the theme's identifiers (``MTDID``, ``BMID``),
  its movement (``ComposerID``, ``WorkID``), the recording (the MusicBrainz
  release ``MusicBrainzID``, the disc ``NameCD`` and track ``NameTrack`` in
  the collection ``CollectionID``, the performer ``PerformanceID``) and the
  excerpt's place in that track (``StartTime``, ``EndTime``, written
  ``mm:ss.s``);
* ``data_ALIGNMENT/<stem>.csv``: ``"MID";"WAV"`` rows, MID in quarter notes
  of the theme encoding written with five decimals, WAV in seconds from the
  excerpt's start; every row but the last is the onset of a note, the last
  the end of the last note;
* the theme encoding as a MIDI file: ``data_EDM-corr_MID/<stem>.mid``, MTD's
  corrected encoding of the theme from the Electronic Dictionary of Musical
  Themes (EDM), whose quarter notes the alignment counts. The engraved
  theme's MIDI, ``data_SCORE_MID/<stem>.mid``, holds other notes for some
  themes (ornaments written out, other note values), so its onsets do not
  always meet the alignment's rows; it can be chosen with
  ``theme_midi="data_SCORE_MID"``, which refuses 616 of the 2,067 themes of
  MTD 1.0.0 (about 30 %: 498 with a row meeting no onset, 118 with a last
  row that is not the end of the last note). One refused theme refuses the
  whole ``load``, so read those themes from the default folder.

The themes of one movement (one ``ComposerID`` and ``WorkID``) lie in one CD
track. ``MtdLoader`` reads the archive in place, or an extracted folder, and
makes, for the movements or themes selected::

    bundle = MtdLoader.from_file(archive, movements=["Beethoven_Op002No1-01"]).create_bundle()

* per CD track, a seconds
  :class:`~timetoalign.timelines.types.ContinuousPhysicalTimeline`
  (``cpt_…``, number type float) of unknown length, unless the caller gives
  it (MTD does not state it), with statements of what MTD says of the
  recording;
* per theme, the excerpt: a child timeline of its track at ``StartTime``,
  ``EndTime - StartTime`` long;
* per theme, the theme encoding: a quarters
  :class:`~timetoalign.timelines.types.ContinuousLogicalTimeline`
  (``clt_…``, number type fraction) holding its notes, each tick an exact
  fraction of a quarter, with statements of the movement it is part of and
  its identifiers; it is a timeline of its own, never a child of the
  excerpt (the two are two manifestations of the theme, related by the
  alignment);
* per alignment row but the last, a synchronous
  :class:`~timetoalign.alignment.claims.MatchClaim` from each note at that
  onset to :data:`~timetoalign.core.ids.CHUNK` on the excerpt, anchored at
  the onset in quarters and the row's seconds.

The loader reads what MTD states; it fetches nothing. Track lengths, and
MusicBrainz recording and track IDs, are the caller's to add. The pitches
are the encoding's: META's ``MidiTransposition``, the interval by which the
recording sounds transposed against the encoding (774 of the 2,067 themes,
mostly an octave down), is not carried.

Known issues of the data, stored as MTD gives them: in MTD 1.0.0, 25
alignment rows that become claims, in 25 themes, state a second outside
their excerpt, 19 before its start (MTD1275: ``0.00000;-0.30000``) and 6
after its end (MTD1063: ``7.75000;3.32070`` on an excerpt 3.3 seconds
long). The claim holds that second on the excerpt's axis; since the excerpt
is a child of its track and a child answers only within its span, a
matchstamp at such a claim names the excerpt's coordinate and no coordinate
of the track.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow as pa
import pyarrow.compute as pc
from typing_extensions import Self

from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import CHUNK, NumberType, Provenance, TimeUnit
from timetoalign.core.fields import SemanticField
from timetoalign.display.html import code
from timetoalign.loader._ids import give_event_ids
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
from timetoalign.timelines.factory import mint_timeline_id
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline


# region Constants

#: The paper describing the dataset, the basis of every statement and claim.
MTD_DOI = "https://doi.org/10.5334/tismir.68"

#: The release of the MTD whose layout the loader reads, as provenance names
#: it (the archive states no version of its own).
MTD_VERSION = "MTD 1.0.0"

#: The folder of the theme encoding whose quarter notes the alignment counts.
DEFAULT_THEME_MIDI = "data_EDM-corr_MID"

#: The folders a theme encoding can be read from, with how its statements
#: name the encoding.
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

# endregion


# region Parsed data


@dataclass(frozen=True)
class _Note:
    """A note of a theme encoding, in ticks."""

    id: str
    start: int
    end: int
    pitch: int
    velocity: int | None


@dataclass(frozen=True)
class _Theme:
    """One theme as the loader holds it after ``load``."""

    stem: str
    meta: dict[str, str]
    start: Decimal
    end: Decimal
    ppq: int
    notes: tuple[_Note, ...]
    rows: tuple[tuple[Fraction, float], ...]
    score_id: str
    excerpt_id: str

    @property
    def movement(self) -> str:
        """The movement, ``<ComposerID>_<WorkID>``."""
        return f"{self.meta['ComposerID']}_{self.meta['WorkID']}"

    @property
    def track(self) -> tuple[str, str, str]:
        """The CD track: MusicBrainz release, disc and track as MTD names them."""
        return (self.meta["MusicBrainzID"], self.meta["NameCD"], self.meta["NameTrack"])


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


# endregion


# region MtdLoader


class MtdLoader(AlignmentLoader):
    """Load themes of the Musical Theme Dataset as one AlignmentBundle.

    The loader reads the MTD archive in place (its path, or a
    ``zipfile.Path`` of it) or an extracted copy (the folder holding
    ``MTD/``, or ``MTD/`` itself), the movements or themes selected only.
    Per CD track it makes a seconds timeline with an excerpt child per
    theme, per theme a quarters timeline of the theme encoding, and per
    alignment row but the last a MatchClaim from each note at that onset to
    :data:`~timetoalign.core.ids.CHUNK` on the excerpt.

    The notes keep the encoding's pitches; MTD's ``MidiTransposition``
    between encoding and recording is not carried. A claim whose second
    lies outside its excerpt (25 in MTD 1.0.0, see the module) is stored as
    given; a matchstamp there names no coordinate of the track.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(source, movements=..., themes=...)`` — read the
       selected themes.
    2. ``loader.create_bundle(track_lengths=...)`` — assemble the
       AlignmentBundle; ``create_timelines()`` / ``create_timeline(uid)``
       give the timelines alone.

    Every ID is minted at ``load`` from the bytes of the files read, so a
    fresh loader reading the same themes mints the same IDs, from the
    archive and from an extracted folder alike.

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
        of every theme are read to find the movements; the alignment and the
        MIDI file only of the themes selected.

        Each alignment row but the last is matched to the onset it writes:
        the onset of the theme encoding within half a tick of the row's
        five-decimal quarters (``6.66667`` is ``20/3``). The last row is
        checked to be the end of the last note within half a tick; the
        library has no place for the aligned span's end, so it is not kept
        (its second lies 0.05 seconds past the excerpt's end in all but one
        theme of MTD 1.0.0).

        :meth:`from_file` passes *movements*, *themes* and *theme_midi* to
        this method and ``seed`` to the constructor.

        Args:
            *sources: One source: the MTD archive (a path, or a
                ``zipfile.Path`` of it), the folder holding ``MTD/``, or
                ``MTD/`` itself.
            movements: Movements to read, ``<ComposerID>_<WorkID>``.
            themes: Themes to read, by MTD ID (``MTD0807``).
            theme_midi: The folder of the theme encodings, one of
                :data:`THEME_MIDI_FOLDERS`; by default
                ``data_EDM-corr_MID``, whose quarter notes the alignment
                counts.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If *source* holds no ``MTD/data_META`` folder,
                or a selected theme lacks its alignment or MIDI file.
            KeyError: If a movement or MTD ID asked for is not in *source*.
            ValueError: If *theme_midi* is no theme encoding folder; if not
                one source is given, or the loader holds one already; if a time of META is not
                ``mm:ss.s`` or the excerpt ends before it starts; if an
                alignment row meets no onset of the theme encoding, or its
                last row is not the end of the last note; if two themes of
                one CD track name different performers.
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
                join_source(root, f"{theme_midi}/{stem}.mid"),
            ]
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
        """Read one theme's alignment and notes; mint its IDs."""
        start = _seconds(meta["StartTime"], "StartTime", stem)
        end = _seconds(meta["EndTime"], "EndTime", stem)
        if end <= start:
            raise ValueError(
                f"MTD theme {stem} ends ({meta['EndTime']}) before it starts "
                f"({meta['StartTime']})"
            )
        midi = join_source(root, f"{self._theme_midi}/{stem}.mid")
        ppq, notes = self._read_notes(midi)
        alignment = join_source(root, f"data_ALIGNMENT/{stem}.csv")
        rows = self._read_alignment(alignment)
        matched = self._match_rows(rows, ppq, notes, alignment, midi, self._theme_midi)
        excerpt_id = mint_timeline_id(TimeUnit.seconds, self._id_minter)
        score_id = mint_timeline_id(TimeUnit.quarters, self._id_minter)
        return _Theme(
            stem=stem,
            meta=meta,
            start=start,
            end=end,
            ppq=ppq,
            notes=notes,
            rows=matched,
            score_id=score_id,
            excerpt_id=excerpt_id,
        )

    def _read_notes(self, midi: Source) -> tuple[int, tuple[_Note, ...]]:
        """The ticks per quarter and the notes of a theme encoding, IDs minted.

        The MIDI file is read by :class:`~timetoalign.loader.midi.ScoreMidiLoader`
        (``parser="mido"``), which pairs note-ons with note-offs; the notes
        are ordered by onset, then pitch.
        """
        from timetoalign.loader.midi.score import ScoreMidiLoader

        midi_loader = ScoreMidiLoader(parser="mido").load(midi)
        ppq = midi_loader.ticks_per_beat
        if not ppq:
            raise ValueError(f"{describe_source(midi)} states no ticks per quarter")
        table = midi_loader.events.table
        kinds = table.column("event_type").to_pylist()
        starts = pc.struct_field(table.column("start"), "numerator").to_pylist()
        ends = pc.struct_field(table.column("end"), "numerator").to_pylist()
        pitches = table.column("pitch").to_pylist()
        velocities = table.column("velocity").to_pylist()
        found: list[tuple[int, int, int, int | None]] = []
        for kind, start, end, pitch, velocity in zip(
            kinds, starts, ends, pitches, velocities, strict=True
        ):
            if kind != "Note":
                continue
            if start is None or end is None or pitch is None:
                raise ValueError(
                    f"{describe_source(midi)} holds a note without onset, end or pitch"
                )
            found.append((int(start), int(end), int(pitch), velocity))
        found.sort(key=lambda note: (note[0], note[2]))
        rows: list[dict[str, Any]] = [{"event_type": "Note"} for _ in found]
        give_event_ids(rows, self._id_minter)
        notes = tuple(
            _Note(row["id"], start, end, pitch, velocity)
            for row, (start, end, pitch, velocity) in zip(rows, found, strict=True)
        )
        return ppq, notes

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
        ppq: int,
        notes: tuple[_Note, ...],
        alignment: Source,
        midi: Source,
        theme_midi: str,
    ) -> tuple[tuple[Fraction, float], ...]:
        """Each row but the last as ``(exact onset in quarters, seconds)``.

        A row's quarters, read as the exact decimal it writes, meet the onset
        within half a tick (``|tick - quarters * ppq| <= 1/2``). The last row
        is the end of the last note within half a tick.

        Raises:
            ValueError: If a row meets no onset or two, or the last row is
                not the end of the last note; the message names the folder
                the encoding was read from.
        """
        reading = (
            f" The alignment counts the quarters of {DEFAULT_THEME_MIDI}; this theme "
            f"encoding was read from {theme_midi}."
        )
        if len(rows) < 2:
            raise ValueError(f"{describe_source(alignment)} holds fewer than two rows")
        onsets = sorted({note.start for note in notes})
        matched: list[tuple[Fraction, float]] = []
        for mid, wav in rows[:-1]:
            ticks = _decimal(mid, alignment) * ppq
            hits = [tick for tick in onsets if abs(tick - ticks) <= Fraction(1, 2)]
            if len(hits) != 1:
                raise ValueError(
                    f"The alignment row {mid};{wav} of {describe_source(alignment)} meets "
                    f"{len(hits)} onsets of {describe_source(midi)} within half a tick "
                    f"({ppq} ticks per quarter): {[str(Fraction(t, ppq)) for t in hits]}."
                    + reading
                )
            _decimal(wav, alignment)
            matched.append((Fraction(hits[0], ppq), float(wav)))
        last_mid = rows[-1][0]
        last_end = max(note.end for note in notes)
        if abs(last_end - _decimal(last_mid, alignment) * ppq) > Fraction(1, 2):
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
        """One synchronous claim per note at an aligned onset, columnar."""
        score_ids: list[str] = []
        excerpt_ids: list[str] = []
        note_ids: list[str] = []
        quarters: list[Fraction] = []
        seconds: list[float] = []
        axes: dict[str, tuple[TimeUnit, NumberType]] = {}
        for theme in self._themes:
            axes[theme.score_id] = (TimeUnit.quarters, NumberType.fraction)
            axes[theme.excerpt_id] = (TimeUnit.seconds, NumberType.float)
            for onset, wav in theme.rows:
                for note in theme.notes:
                    if Fraction(note.start, theme.ppq) == onset:
                        score_ids.append(theme.score_id)
                        excerpt_ids.append(theme.excerpt_id)
                        note_ids.append(note.id)
                        quarters.append(onset)
                        seconds.append(wav)
        return MatchClaimField.from_columns(
            score_ids,
            excerpt_ids,
            note_ids,
            pa.repeat(pa.scalar(CHUNK), len(note_ids)),
            quarters,
            seconds,
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
        """The folder the theme encodings were read from."""
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

        The bundle holds, ungrouped, every CD track's timeline (its
        excerpts as children) and every theme encoding's timeline, and the
        MatchClaims of the alignments, columnar.

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
        for timeline in self.create_timelines(track_lengths=track_lengths):
            bundle.add_timeline(timeline)
        bundle.add_match_claim_field(claims)
        return bundle

    def create_timelines(
        self,
        id_pattern: str | None = None,
        *,
        track_lengths: Mapping[tuple[str, str, str], float | None] | None = None,
    ) -> list["Timeline"]:
        """The CD tracks' timelines (excerpts as children), then the themes'.

        Args:
            id_pattern: Optional regular expression the IDs must match.
            track_lengths: Lengths in seconds of CD tracks (see
                :meth:`create_bundle`).

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
        timelines += [self._make_theme_score(theme) for theme in self._themes]
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """One timeline of :meth:`create_timelines` by its ID.

        Args:
            uid: A CD track's or a theme encoding's timeline ID.
            **kwargs: ``track_lengths``, as :meth:`create_timelines` takes
                it; any other keyword is refused.

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
            excerpt = ContinuousPhysicalTimeline(
                length=float(theme.end - theme.start),
                unit=TimeUnit.seconds,
                number_type=NumberType.float,
                uid=theme.excerpt_id,
                name=f"Theme {theme.meta['BMID']} (MTD {theme.meta['MTDID']})",
            )
            timeline.add_child(excerpt, float(theme.start))
        return timeline

    def _make_theme_score(self, theme: _Theme) -> ContinuousLogicalTimeline:
        """A theme encoding's quarters timeline: its notes and statements."""
        ppq = theme.ppq
        timeline = ContinuousLogicalTimeline(
            length=Fraction(max(note.end for note in theme.notes), ppq),
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            uid=theme.score_id,
            name=f"{theme.meta['BMID']} (MTD {theme.meta['MTDID']})",
        )
        timeline.add_events(
            [
                {
                    "id": note.id,
                    "start": Fraction(note.start, ppq),
                    "end": Fraction(note.end, ppq),
                    "event_type": "Note",
                    "pitch": {"midi_number": note.pitch},
                    "velocity": note.velocity,
                }
                for note in theme.notes
            ]
        )
        provenance = Provenance(
            method="imported",
            basis=(MTD_DOI,),
            note=f"{MTD_VERSION} data_META/{theme.stem}.json",
        )
        pairs: list[tuple[str, Any]] = [
            ("schema:isPartOf", {"label": _movement_label(theme.movement)}),
            ("melod:hasIdentifier", _identifier("MTD", theme.meta["MTDID"])),
            (
                "melod:hasIdentifier",
                _identifier("Barlow and Morgenstern", theme.meta["BMID"]),
            ),
            (
                "dcterms:source",
                {
                    "label": f"{THEME_MIDI_FOLDERS[self._theme_midi]}, "
                    f"{MTD_VERSION} {self._theme_midi}/{theme.stem}.mid"
                },
            ),
        ]
        for property, value in pairs:
            timeline.add_statement(property, value, provenance=provenance)
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
        rows.append(("Theme encodings", code(self._theme_midi)))
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
