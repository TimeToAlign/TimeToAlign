"""ParangonadaLoader: a multimodal AlignmentBundle from parangonada exports.

A *parangonada* export is the CSV form produced by parangonar's
parangonada visualisation.  For a single musical work it bundles, for
every recorded performance, a triple of CSV files under a per-performer
subdirectory of ``match/match_transkun/``:

* ``part.csv`` — the score notes.  Header
  ``onset_beat,duration_beat,onset_quarter,duration_quarter,onset_div,
  duration_div,pitch,voice,id``.  The score is identical across every
  performance in the dataset, so it is parsed once.  ``onset_quarter``
  is an exact rational (parsed as a :class:`~fractions.Fraction`),
  ``onset_div`` an integer in the score's division grid, and ``pitch``
  a MIDI integer.  Two arithmetic invariants tie the two grids:
  ``onset_div == 32 * onset_quarter + 16`` and
  ``duration_div == 32 * duration_quarter``.
* ``ppart.csv`` — the performed notes, timed in seconds.  Header
  ``onset_sec,duration_sec,pitch,velocity,track,channel,id``.
* ``align.csv`` — the note-level correspondences.  Header
  ``idx,matchtype,partid,ppartid``.  ``matchtype`` is ``0`` for a match
  (both ids present), ``1`` for a score-only insertion
  (``ppartid == "undefined"``), and ``2`` for a performance-only
  deletion (``partid == "undefined"``).

Each performance also carries a pair of **measured beat-feature**
files under the dataset's ``features/`` directory, tab-separated and
discovered by globbing ``<key>_measure*.{beats,dyn}`` (the file stems
are non-uniform — one performer's are ``..._measure_fix`` while the
rest are ``..._measure``):

* ``<key>_measure*.beats`` — one row per measured beat.  Header
  ``measure_number  beat  onset_beat  onset_sec  BPM  interp  swap``.
* ``<key>_measure*.dyn`` — one row per measured beat.  Header
  ``measure_number  beat  onset_beat  velocity_mean  velocity_max
  interp`` (no ``onset_sec`` column).

These are *measured* per-beat features (variable tempo and dynamics
extracted from the recording), not a synthesised tempo curve; they are
added as ``Beat`` and ``Dynamics`` events to the performance's seconds
timeline (``perf:<key>:cpt_1``), alongside its ``Note`` events.  The
``.dyn`` rows have no onset of their own, so each is joined to the
``.beats`` row sharing its ``(measure_number, beat)`` key (the two
files are 1:1 on that key) to recover its ``onset_sec``.

``ParangonadaLoader`` ingests the whole dataset directory in one call::

    bundle = ParangonadaLoader.from_file(dataset_dir).create_bundle()

and produces a single :class:`~timetoalign.alignment.bundle.AlignmentBundle`
expressing the work across the logical and physical domains:

* a shared ``"score"`` group with the same notes in two logical units —
  a quarter-note :class:`~timetoalign.timelines.types.ContinuousLogicalTimeline`
  (``score:clt_1``) and a division-grid
  :class:`~timetoalign.timelines.types.DiscreteLogicalTimeline`
  (``score:dlt_1``) carrying a divs→quarters
  :class:`~timetoalign.maps.linear.LinearMap`;
* one ``perf:<key>`` group per performance with the performed notes in
  two physical units — a seconds
  :class:`~timetoalign.timelines.types.ContinuousPhysicalTimeline`
  (``perf:<key>:cpt_1``) and a samples
  :class:`~timetoalign.timelines.types.DiscretePhysicalTimeline`
  (``perf:<key>:dpt_1``) carrying a
  :class:`~timetoalign.maps.convenience.SamplesToSeconds` map; and
* a cross-group :class:`~timetoalign.alignment.anchors.MatchClaim` per
  ``align.csv`` row — a synchronous claim for a match and a NOMATCH
  claim for an insertion or a deletion.

The loader reads the *existing* alignment; it never runs an aligner.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
    timetoalign.maps.LinearMap
    timetoalign.maps.SamplesToSeconds
"""

from __future__ import annotations

import csv
import logging
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.alignment.claims import Agent, MatchClaim, MatchMetadata
from timetoalign.core import AgentType, TimeUnit
from timetoalign.display.html import code
from timetoalign.loader.base import AlignmentLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.maps.convenience import SamplesToSeconds
from timetoalign.maps.linear import LinearMap
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
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

#: Group id for the shared score timelines.
_SCORE_GROUP = "score"

#: Required CSV files in every performer subdirectory.
_REQUIRED_FILES = ("part.csv", "ppart.csv", "align.csv")

#: Dataset subdirectory holding the per-performer beat-feature files.
_FEATURES_DIR = "features"

#: Trailing suffix some performer directories carry; stripped to form the key.
_PARANGONADA_SUFFIX = "_parangonada"

#: Sample rate of every recording in the dataset (44100 Hz stereo ``.wav``).
#: Read per performer from the audio file rather than hard-coded.
_DEFAULT_SAMPLE_RATE = 44100

#: ``align.csv`` matchtype codes.
_MATCH = "0"
_SCORE_ONLY = "1"
_PERF_ONLY = "2"

# endregion


# region ParangonadaLoader


class ParangonadaLoader(AlignmentLoader):
    """Load a parangonada CSV export as one multimodal AlignmentBundle.

    The loader discovers every performance under the dataset's
    ``match/match_transkun/`` directory, parses the shared score
    ``part.csv`` once, each performance's ``ppart.csv`` and ``align.csv``,
    and binds each recording's ``.wav`` for its sample rate.  It then
    assembles a single bundle with one shared score group and one group
    per performance, linked by cross-group :class:`MatchClaim` instances.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(dataset_dir)`` — ingest the whole dataset directory.
    2. ``loader.create_bundle()`` — assemble the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines.

    The loader reads existing alignments; it never runs an aligner.
    """

    def __init__(self) -> None:
        super().__init__()

        self._score_clt: ContinuousLogicalTimeline | None = None
        self._score_dlt: DiscreteLogicalTimeline | None = None
        self._perf_cpt: dict[str, ContinuousPhysicalTimeline] = {}
        self._perf_dpt: dict[str, DiscretePhysicalTimeline] = {}
        self._claims: list[MatchClaim] = []
        self._name: str | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Path) -> Any:
        """Not used: a parangonada dataset is a single coherent unit.

        :class:`ParangonadaLoader` ingests an entire dataset directory
        through :meth:`load`; the base class's per-source AlignmentStore
        merge does not apply.
        """
        raise NotImplementedError(
            "ParangonadaLoader ingests a whole dataset directory via "
            "load(dataset_dir); per-source loading is not used."
        )

    # endregion

    # region Loading

    def load(self, dataset_dir: str | Path) -> Self:
        """Ingest a whole parangonada dataset directory.

        Args:
            dataset_dir: Path to the dataset directory (the one that
                contains the ``match/`` and ``audio/`` subdirectories).

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If the directory or a required subdirectory
                is missing.
        """
        directory = Path(dataset_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

        match_dir = directory / "match" / "match_transkun"
        if not match_dir.is_dir():
            raise FileNotFoundError(
                f"No 'match/match_transkun/' subdirectory in dataset: {directory}"
            )
        audio_dir = directory / "audio"
        features_dir = directory / _FEATURES_DIR

        performers = self._discover_performers(match_dir)
        self._sources = [match_dir]
        self._name = directory.name

        # The score is identical across performances; parse it once from
        # the first performer's part.csv.
        _first_key, first_dir = performers[0]
        score_rows = self._read_csv(first_dir / "part.csv")
        self._build_score_timelines(score_rows)

        score_q_by_id = {
            row["id"]: Fraction(str(row["onset_quarter"])) for row in score_rows
        }

        for performer_key, performer_dir in performers:
            self._build_one_performer(
                performer_key, performer_dir, audio_dir, features_dir, score_q_by_id
            )

        return self

    @staticmethod
    def _discover_performers(match_dir: Path) -> list[tuple[str, Path]]:
        """Find every performer subdirectory holding the required CSV triple.

        Performers are the immediate *subdirectories* of *match_dir*
        containing ``part.csv`` / ``ppart.csv`` / ``align.csv``; the
        sibling ``.csv`` / ``.match`` files in *match_dir* are ignored.
        The key is the directory name with a trailing
        ``"_parangonada"`` removed; the result is sorted by key.

        Args:
            match_dir: The ``match/match_transkun/`` directory.

        Returns:
            A list of ``(performer_key, performer_dir)`` pairs, sorted by
            key.

        Raises:
            FileNotFoundError: If no performer subdirectory is found.
        """
        performers: list[tuple[str, Path]] = []
        for child in match_dir.iterdir():
            if not child.is_dir():
                continue
            if not all((child / name).is_file() for name in _REQUIRED_FILES):
                continue
            key = child.name.removesuffix(_PARANGONADA_SUFFIX)
            performers.append((key, child))

        if not performers:
            raise FileNotFoundError(
                f"No performer subdirectory with {list(_REQUIRED_FILES)} "
                f"found under {match_dir}"
            )

        performers.sort(key=lambda pair: pair[0])
        return performers

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        """Read a header-based CSV into a list of row dicts."""
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _read_tsv(path: Path) -> list[dict[str, str]]:
        """Read a header-based TAB-separated file into a list of row dicts."""
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    def _build_score_timelines(self, score_rows: list[dict[str, str]]) -> None:
        """Build the two shared logical score timelines from ``part.csv``.

        The continuous timeline carries quarter-note onsets (exact
        ``Fraction``); the discrete timeline carries division-grid onsets
        (int) plus a divs→quarters :class:`LinearMap`.  Both hold the same
        notes, each carrying the faithful MIDI pitch and voice.
        """
        # Length bounds large enough to hold every note's full extent.
        max_quarter_end = max(
            Fraction(str(row["onset_quarter"])) + Fraction(str(row["duration_quarter"]))
            for row in score_rows
        )
        max_div_end = max(
            int(row["onset_div"]) + int(row["duration_div"]) for row in score_rows
        )

        score_clt = ContinuousLogicalTimeline(
            length=max_quarter_end,
            unit=TimeUnit.quarters,
            uid=_SCORE_CLT_ID,
            name=self._name,
        )
        score_clt.add_events(
            [
                {
                    "id": row["id"],
                    "start": Fraction(str(row["onset_quarter"])),
                    "event_type": "Note",
                    # Number-only source: the most-expressive faithful pitch
                    # type is EnharmonicPitch, carried as its {midi_number}
                    # struct so the timeline affords the EP semantic field.
                    "pitch": {"midi_number": int(row["pitch"])},
                    "voice": int(row["voice"]),
                }
                for row in score_rows
            ]
        )

        score_dlt = DiscreteLogicalTimeline(
            length=max_div_end,
            unit=TimeUnit.ticks,
            uid=_SCORE_DLT_ID,
            name=self._name,
        )
        score_dlt.add_events(
            [
                {
                    "id": row["id"],
                    "start": int(row["onset_div"]),
                    "event_type": "Note",
                    "pitch": {"midi_number": int(row["pitch"])},
                    "voice": int(row["voice"]),
                }
                for row in score_rows
            ]
        )
        # divs → quarters: quarters = div / 32 - 1/2 (the score's anacrusis
        # places div 0 at quarter -1/2; 32 divs per quarter).
        score_dlt.add_conversion_map(
            LinearMap(
                scalar=Fraction(1, 32),
                offset=Fraction(-1, 2),
                source_unit=TimeUnit.ticks,
                target_unit=TimeUnit.quarters,
            )
        )

        self._score_clt = score_clt
        self._score_dlt = score_dlt

    def _build_one_performer(
        self,
        performer_key: str,
        performer_dir: Path,
        audio_dir: Path,
        features_dir: Path,
        score_q_by_id: dict[str, Fraction],
    ) -> None:
        """Build a performance's two physical timelines and its MatchClaims.

        The continuous timeline holds one event per ``ppart.csv`` row in
        seconds, plus the measured ``Beat`` / ``Dynamics`` feature events
        from the ``features/`` directory; the discrete timeline holds the
        same notes converted to sample indices, with a
        :class:`SamplesToSeconds` C-Map.  Claims are emitted faithfully —
        one per ``align.csv`` row, with no deduplication.
        """
        sample_rate = self._resolve_sample_rate(performer_key, audio_dir)

        ppart_rows = self._read_csv(performer_dir / "ppart.csv")
        perf_sec_by_id = {row["id"]: float(row["onset_sec"]) for row in ppart_rows}

        # Physical timeline lengths from the recording's audio.
        audio_info = self._resolve_audio_info(performer_key, audio_dir)
        seconds_length = (
            audio_info.duration_seconds
            if audio_info is not None
            else (max(perf_sec_by_id.values()) if perf_sec_by_id else 0.0)
        )
        samples_length = (
            audio_info.n_samples
            if audio_info is not None
            else int(round(seconds_length * sample_rate))
        )

        cpt_id = f"perf:{performer_key}:cpt_1"
        dpt_id = f"perf:{performer_key}:dpt_1"

        perf_cpt = ContinuousPhysicalTimeline(
            length=seconds_length,
            unit=TimeUnit.seconds,
            uid=cpt_id,
            name=performer_key,
        )
        perf_cpt.add_events(
            [
                {
                    "id": row["id"],
                    "start": float(row["onset_sec"]),
                    "event_type": "Note",
                    "pitch": {"midi_number": int(row["pitch"])},
                    "velocity": int(row["velocity"]),
                }
                for row in ppart_rows
            ]
        )

        # Measured per-beat features (variable tempo + dynamics) extend
        # the seconds timeline with Beat / Dynamics events.  The schema
        # grows (new columns) per successive add_events call; absent
        # columns are null-filled.
        self._add_feature_events(performer_key, features_dir, perf_cpt)

        perf_dpt = DiscretePhysicalTimeline(
            length=samples_length,
            unit=TimeUnit.samples,
            uid=dpt_id,
            name=performer_key,
        )
        perf_dpt.add_events(
            [
                {
                    "id": row["id"],
                    "start": int(round(float(row["onset_sec"]) * sample_rate)),
                    "event_type": "Note",
                    "pitch": {"midi_number": int(row["pitch"])},
                    "velocity": int(row["velocity"]),
                }
                for row in ppart_rows
            ]
        )
        perf_dpt.add_conversion_map(SamplesToSeconds(sample_rate=sample_rate))

        self._perf_cpt[performer_key] = perf_cpt
        self._perf_dpt[performer_key] = perf_dpt

        self._build_claims(
            performer_key, performer_dir, score_q_by_id, perf_sec_by_id, cpt_id
        )

    @classmethod
    def _add_feature_events(
        cls,
        performer_key: str,
        features_dir: Path,
        perf_cpt: ContinuousPhysicalTimeline,
    ) -> None:
        """Add measured ``Beat`` / ``Dynamics`` events to the seconds timeline.

        Both feature files are discovered by globbing
        ``<key>_measure*.beats`` / ``<key>_measure*.dyn`` under
        *features_dir* (the stems are non-uniform — one performer uses
        ``_measure_fix``, the rest ``_measure``).

        Each ``.beats`` row becomes a ``Beat`` event timed at its
        ``onset_sec``.  The ``.dyn`` file carries no onset, so each
        ``.dyn`` row is joined to the ``.beats`` row sharing its
        ``(measure_number, beat)`` key (1:1) to recover the onset for a
        ``Dynamics`` event.  Events are faithful measured features — no
        tempo curve is synthesised.

        Args:
            performer_key: The stripped performer key (the feature-file
                stem prefix).
            features_dir: The dataset's ``features/`` directory.
            perf_cpt: The performer's seconds (``cpt_1``) timeline, already
                holding its ``Note`` events; feature events are appended in
                place via ``add_events(..., allow_expansion=True)``.
        """
        beats_path = cls._glob_feature_file(features_dir, performer_key, "beats")
        dyn_path = cls._glob_feature_file(features_dir, performer_key, "dyn")

        beats_rows = cls._read_tsv(beats_path)
        dyn_rows = cls._read_tsv(dyn_path)

        # Onset lookup for the .dyn join: (measure_number, beat) -> onset_sec.
        onset_by_key: dict[tuple[str, str], float] = {
            (row["measure_number"], row["beat"]): float(row["onset_sec"])
            for row in beats_rows
        }
        # The two files are 1:1 on (measure_number, beat): every .dyn row
        # must find exactly one .beats onset.
        assert len(onset_by_key) == len(beats_rows), (
            f"{performer_key}: .beats (measure_number, beat) keys are not "
            f"unique ({len(onset_by_key)} keys for {len(beats_rows)} rows)"
        )
        matched = sum(
            1
            for row in dyn_rows
            if (row["measure_number"], row["beat"]) in onset_by_key
        )
        assert matched == len(dyn_rows), (
            f"{performer_key}: .dyn ↔ .beats join is not 1:1 "
            f"({matched} of {len(dyn_rows)} .dyn rows matched a .beats onset)"
        )

        beat_events = [
            {
                "id": f"{performer_key}:beat:{i}",
                "start": float(row["onset_sec"]),
                "event_type": "Beat",
                "measure_number": int(row["measure_number"]),
                "beat": int(row["beat"]),
                "bpm": float(row["BPM"]),
                "interp": int(row["interp"]),
                "swap": int(row["swap"]),
            }
            for i, row in enumerate(beats_rows)
        ]
        dyn_events = [
            {
                "id": f"{performer_key}:dyn:{i}",
                "start": onset_by_key[(row["measure_number"], row["beat"])],
                "event_type": "Dynamics",
                "measure_number": int(row["measure_number"]),
                "beat": int(row["beat"]),
                "velocity_mean": float(row["velocity_mean"]),
                "velocity_max": float(row["velocity_max"]),
                "interp": int(row["interp"]),
            }
            for i, row in enumerate(dyn_rows)
        ]

        perf_cpt.add_events(beat_events, allow_expansion=True)
        perf_cpt.add_events(dyn_events, allow_expansion=True)

    @staticmethod
    def _glob_feature_file(features_dir: Path, performer_key: str, suffix: str) -> Path:
        """Return the single ``<key>_measure*.<suffix>`` file for a performer.

        Args:
            features_dir: The dataset's ``features/`` directory.
            performer_key: The stripped performer key.
            suffix: The feature-file extension without a dot (``beats`` /
                ``dyn``).

        Raises:
            FileNotFoundError: If no matching file (or more than one) is
                found.
        """
        matches = sorted(features_dir.glob(f"{performer_key}_measure*.{suffix}"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one '{performer_key}_measure*.{suffix}' file "
                f"under {features_dir}, found {len(matches)}: "
                + ", ".join(m.name for m in matches)
            )
        return matches[0]

    def _build_claims(
        self,
        performer_key: str,
        performer_dir: Path,
        score_q_by_id: dict[str, Fraction],
        perf_sec_by_id: dict[str, float],
        cpt_id: str,
    ) -> None:
        """Emit one cross-group MatchClaim per ``align.csv`` row (faithfully).

        A matchtype-0 row yields a synchronous score→performance
        projection claim; matchtype-1 (score-only) and matchtype-2
        (performance-only) rows yield NOMATCH claims, oriented so the
        present side is the claim's source.  Duplicated rows in the source
        data are preserved as duplicate claims.
        """
        meta = MatchMetadata(
            agent=Agent(
                name="parangonada",
                type=AgentType.software,
                identifier=performer_key,
            ),
            certainty=1.0,
        )
        for row in self._read_csv(performer_dir / "align.csv"):
            matchtype = row["matchtype"]
            partid = row["partid"]
            ppartid = row["ppartid"]

            if matchtype == _MATCH:
                self._claims.append(
                    MatchClaim.from_projection(
                        event={"start": score_q_by_id[partid]},
                        source_tl_id=_SCORE_CLT_ID,
                        target_tl_id=cpt_id,
                        target_coord=float(perf_sec_by_id[ppartid]),
                        source_unit=TimeUnit.quarters,
                        target_unit=TimeUnit.seconds,
                        coord_key="start",
                        metadata=meta,
                    )
                )
            elif matchtype == _SCORE_ONLY:
                self._claims.append(
                    MatchClaim.nomatch(
                        event={"start": score_q_by_id[partid]},
                        source_tl_id=_SCORE_CLT_ID,
                        target_tl_id=cpt_id,
                        unit=TimeUnit.quarters,
                        metadata=meta,
                    )
                )
            elif matchtype == _PERF_ONLY:
                self._claims.append(
                    MatchClaim.nomatch(
                        event={"start": float(perf_sec_by_id[ppartid])},
                        source_tl_id=cpt_id,
                        target_tl_id=_SCORE_CLT_ID,
                        unit=TimeUnit.seconds,
                        metadata=meta,
                    )
                )

    @staticmethod
    def _resolve_audio_info(performer_key: str, audio_dir: Path) -> Any:
        """Return the :class:`AudioInfo` for a performer's ``.wav``, or None.

        The recording shares the performer's original directory name, so
        it is found by globbing ``<key>*.wav`` under the audio directory.
        Returns ``None`` when no audio directory or file is present.
        """
        if not audio_dir.is_dir():
            return None
        matches = sorted(audio_dir.glob(f"{performer_key}*.wav"))
        if not matches:
            return None
        return AudioLoader.from_file(matches[0]).audio_info

    @classmethod
    def _resolve_sample_rate(cls, performer_key: str, audio_dir: Path) -> int:
        """Return a performer recording's sample rate (Hz).

        Falls back to the dataset's known 44100 Hz when no audio file is
        present.
        """
        info = cls._resolve_audio_info(performer_key, audio_dir)
        return info.sample_rate if info is not None else _DEFAULT_SAMPLE_RATE

    # endregion

    # region Properties

    @property
    def performer_keys(self) -> list[str]:
        """The discovered performer keys, in sorted (chronological) order."""
        return list(self._perf_cpt.keys())

    # endregion

    # region Domain Object Creation

    def create_bundle(self) -> "AlignmentBundle":
        """Assemble the AlignmentBundle from the loaded dataset.

        Returns:
            An ``AlignmentBundle`` with one shared ``"score"`` group (two
            logical timelines), one ``perf:<key>`` group per performance
            (two physical timelines), and all cross-group MatchClaims.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if self._score_clt is None or self._score_dlt is None:
            raise RuntimeError(
                "No dataset loaded yet. Call load() before create_bundle()."
            )

        bundle = AlignmentBundle(name=self._name)
        bundle.add_timeline(self._score_clt, as_group=_SCORE_GROUP)
        bundle.add_timeline(self._score_dlt, grouped_with=self._score_clt.id)

        for performer_key in self._perf_cpt:
            cpt = self._perf_cpt[performer_key]
            dpt = self._perf_dpt[performer_key]
            group_id = f"perf:{performer_key}"
            bundle.add_timeline(cpt, as_group=group_id)
            bundle.add_timeline(dpt, grouped_with=cpt.id)

        bundle.add_match_claims(self._claims)
        return bundle

    def create_timelines(self, id_pattern: str | None = None) -> list["Timeline"]:
        """Return all timelines: the two score timelines then each performer's.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.
        """
        if self._score_clt is None or self._score_dlt is None:
            return []
        timelines: list[Timeline] = [self._score_clt, self._score_dlt]
        for performer_key in self._perf_cpt:
            timelines.append(self._perf_cpt[performer_key])
            timelines.append(self._perf_dpt[performer_key])
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Return a single timeline by its uid.

        Args:
            uid: A timeline uid — ``"score:clt_1"``, ``"score:dlt_1"``, or a
                performer timeline uid (``"perf:<key>:cpt_1"`` /
                ``"perf:<key>:dpt_1"``).

        Raises:
            KeyError: If no timeline matches.
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._score_clt is None or self._score_dlt is None:
            raise RuntimeError(
                "No dataset loaded yet. Call load() before create_timeline()."
            )

        if uid == _SCORE_CLT_ID:
            return self._score_clt
        if uid == _SCORE_DLT_ID:
            return self._score_dlt
        for performer_key, cpt in self._perf_cpt.items():
            if uid == cpt.id:
                return cpt
            if uid == self._perf_dpt[performer_key].id:
                return self._perf_dpt[performer_key]

        available = [_SCORE_CLT_ID, _SCORE_DLT_ID]
        for performer_key in self._perf_cpt:
            available.append(f"perf:{performer_key}:cpt_1")
            available.append(f"perf:{performer_key}:dpt_1")
        raise KeyError(
            f"No timeline with uid '{uid}'. Available: "
            + ", ".join(f"'{uid}'" for uid in available)
        )

    # endregion

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is cross-group claims, not store events."""
        return ("Claims", str(len(self._claims)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Extend the base rows with the parangonada-specific shape.

        The base :class:`AlignmentLoader` count row is replaced by the
        claim count (see :meth:`_repr_count_row`); the data lives in the
        assembled timelines and claims, not the unpopulated per-source
        ``AlignmentStore``.
        """
        rows = super()._repr_rows()
        n_perf = len(self._perf_cpt)
        name = self._name or "(not loaded)"
        rows.append(("Dataset", code(name)))
        rows.append(("Performers", str(n_perf)))
        if self._score_clt is not None:
            # 2 shared score timelines + 2 per performer; 1 score group + 1
            # per performer.
            n_timelines = 2 + 2 * n_perf
            n_groups = 1 + n_perf
            rows.append(("Timelines", f"{n_timelines} in {n_groups} group(s)"))
            keys = ", ".join(code(k) for k in self._perf_cpt)
            rows.append(("Performer keys", keys))
        return rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Number of cross-group MatchClaims (the loader's primary payload).

        The inherited count reads the per-source ``AlignmentStore``, which
        this whole-directory loader never populates; the claim count is the
        meaningful size and keeps :meth:`_repr_html_` consistent with
        :meth:`__repr__`.
        """
        return len(self._claims)

    def __repr__(self) -> str:
        n_perf = len(self._perf_cpt)
        n_claims = len(self._claims)
        return f"ParangonadaLoader(performers={n_perf}, claims={n_claims})"

    # endregion


# endregion
