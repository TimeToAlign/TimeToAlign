"""PerformancePrecisionLoader: score-to-audio alignment from CAAMP exports.

A Performance Precision specimen is a directory exported by an
audio-to-score alignment tool.  It bundles three artifacts for a single
musical work:

* a ``.solo`` score file (a header-less tab-separated note table — see
  :class:`~timetoalign.loader.tabular.solo.SoloLoader`),
* a Verovio timemap ``.json`` file giving each measure's absolute
  quarter-note position, and
* an ``Alignments/`` directory holding, for every recorded performance,
  three alignment CSVs at the *note*, *bar*, and *beat* granularities.

Each alignment CSV has a ``LABEL,TIME,FRAME`` header.  ``LABEL`` is a
score position written ``"<measure>+<offset>"`` (the same convention as
the ``.solo`` file's first column); ``TIME`` is the onset in seconds (or
the literal ``"N"`` for a score position the aligner could not locate in
the recording); ``FRAME`` is the audio sample index (ignored here).

``PerformancePrecisionLoader`` ingests the whole directory in one call::

    bundle = PerformancePrecisionLoader.from_file(specimen_dir).create_bundle()

and produces an :class:`~timetoalign.alignment.bundle.AlignmentBundle`
with:

* one logical score timeline in quarters (absolute positions resolved
  against a :class:`~timetoalign.maps.meter.MetricMap` built from the
  Verovio timemap), holding all ``.solo`` notes;
* one physical performance timeline per recording, in seconds; and
* a :class:`~timetoalign.alignment.anchors.MatchClaim` per alignment row
  per granularity — synchronous claims for located onsets and NOMATCH
  claims for the dangling (``"N"``) score positions.

The score's measure → quarter conversion runs through a
:class:`~timetoalign.maps.meter.MetricalPositionMap`, which the loader
attaches to the score timeline and also exposes via the
:attr:`metrical_position_map` property.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
    timetoalign.maps.MetricMap
"""

from __future__ import annotations

import csv
import json
import logging
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.alignment.claims import Agent, MatchClaim, MatchMetadata
from timetoalign.core import AgentType, TimeUnit
from timetoalign.loader.base import AlignmentLoader
from timetoalign.loader.tabular.solo import SoloLoader
from timetoalign.maps.meter import MetricalPositionMap, MetricMap
from timetoalign.timelines.types import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Constants

#: Score timeline uid (type-based id with role prefix).
_SCORE_TL_ID = "score:clt_1"

#: Sentinel for an alignment row the aligner could not locate in audio.
_DANGLING = "N"

#: Granularities present in the ``Alignments/`` directory, mapped to the
#: filename-stem suffix that distinguishes them.  The note level is the
#: bare stem (no suffix).
_GRANULARITIES = {
    "note": "",
    "bar": "_bar",
    "beats": "_beats",
}

# endregion


# region PerformancePrecisionLoader


class PerformancePrecisionLoader(AlignmentLoader):
    """Load a Performance Precision specimen directory as an AlignmentBundle.

    The loader composes :class:`SoloLoader` internally for the ``.solo``
    score and :meth:`MetricMap.from_verovio_timemap` for the measure
    structure, then resolves every ``"<measure>+<offset>"`` label — both
    the score notes and the per-performer alignment rows — into absolute
    quarter-note positions.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(specimen_dir)`` — ingests the whole directory.
    2. ``loader.create_bundle()`` — assembles the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines.

    Args:
        score_unit: Unit for the score timeline. Default
            ``TimeUnit.quarters``.
        media_unit: Unit for the performance timelines. Default
            ``TimeUnit.seconds``.
    """

    def __init__(
        self,
        *,
        score_unit: TimeUnit = TimeUnit.quarters,
        media_unit: TimeUnit = TimeUnit.seconds,
    ) -> None:
        super().__init__()
        self._score_unit = score_unit
        self._media_unit = media_unit

        self._score_timeline: ContinuousLogicalTimeline | None = None
        self._performance_timelines: dict[str, ContinuousPhysicalTimeline] = {}
        self._claims: list[MatchClaim] = []
        self._metric_map: MetricMap | None = None
        self._metrical_position_map: MetricalPositionMap | None = None
        self._name: str | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Path) -> Any:
        """Not used: a specimen directory is a single coherent unit.

        :class:`PerformancePrecisionLoader` ingests an entire specimen
        directory through :meth:`load`; the base class's per-source
        AlignmentStore merge does not apply.
        """
        raise NotImplementedError(
            "PerformancePrecisionLoader ingests a whole specimen directory "
            "via load(specimen_dir); per-source loading is not used."
        )

    # endregion

    # region Properties

    @property
    def metric_map(self) -> MetricMap:
        """The MetricMap of measure boundaries (built from the timemap)."""
        if self._metric_map is None:
            raise RuntimeError("No specimen loaded yet. Call load() first.")
        return self._metric_map

    @property
    def metrical_position_map(self) -> MetricalPositionMap:
        """The MetricalPositionMap linking quarters ↔ measure/beat."""
        if self._metrical_position_map is None:
            raise RuntimeError("No specimen loaded yet. Call load() first.")
        return self._metrical_position_map

    # endregion

    # region Loading

    def load(self, specimen_dir: str | Path) -> Self:
        """Ingest a whole Performance Precision specimen directory.

        Args:
            specimen_dir: Path to the directory containing the ``.solo``
                file, the Verovio timemap ``.json``, and the
                ``Alignments/`` subdirectory.

        Returns:
            Self, for method chaining.

        Raises:
            FileNotFoundError: If the directory or a required artifact is
                missing.
            ValueError: If more than one ``.solo`` or timemap is present.
        """
        directory = Path(specimen_dir)
        if not directory.is_dir():
            raise FileNotFoundError(f"Not a directory: {directory}")

        solo_path = self._resolve_single(directory, "*.solo")
        timemap_path = self._resolve_single(directory, "*.json")
        alignments_dir = directory / "Alignments"
        if not alignments_dir.is_dir():
            raise FileNotFoundError(
                f"No 'Alignments/' subdirectory in specimen: {directory}"
            )

        self._sources = [solo_path, timemap_path, alignments_dir]
        self._name = solo_path.stem

        # Measure structure first — needed to resolve every label.
        self._metric_map = MetricMap.from_verovio_timemap(timemap_path)
        self._metrical_position_map = MetricalPositionMap(self._metric_map)
        first_meter_quarters = self._first_meter_quarters(timemap_path)
        starts, m0_downbeat = self._build_measure_lookup(
            self._metric_map, first_meter_quarters
        )

        self._build_score_timeline(solo_path, starts, m0_downbeat)
        self._build_performers(alignments_dir, starts, m0_downbeat)

        return self

    @staticmethod
    def _resolve_single(directory: Path, pattern: str) -> Path:
        """Return the single file in *directory* matching *pattern*."""
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No file matching '{pattern}' in specimen: {directory}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Expected exactly one '{pattern}' in {directory}, "
                f"found {len(matches)}: {[m.name for m in matches]}"
            )
        return matches[0]

    @staticmethod
    def _first_meter_quarters(timemap_path: Path) -> Fraction:
        """Nominal length, in quarters, of the first measure.

        Read from the first ``measureOn`` entry's ``meterSig`` time
        signature (``"<mn> <num>/<den>"``).  A ``12/8`` signature is
        ``12/8`` of a whole note, i.e. ``12/8 * 4 == 6`` quarters.  This
        is the *nominal* bar length, distinct from the MetricMap's first
        stored length, which is only the gap to the first real boundary
        and is shortened by the anacrusis.
        """
        timemap = json.loads(Path(timemap_path).read_text(encoding="utf-8"))
        for entry in timemap:
            if "measureOn" in entry:
                timesig = str(entry["meterSig"]).split()[1]
                numerator, denominator = timesig.split("/")
                return Fraction(int(numerator), int(denominator)) * 4
        raise ValueError(f"No measure boundaries in timemap: {timemap_path}")

    @staticmethod
    def _build_measure_lookup(
        metric_map: MetricMap,
        first_meter_quarters: Fraction,
    ) -> tuple[list[Fraction], Fraction]:
        """Build the LABEL-measure → quarter-start lookup.

        The ``.solo`` / alignment LABEL convention numbers measures from
        ``0`` (the anacrusis) upward.  The MetricMap's measure starts are
        the *first* sounding boundary of each bar, so LABEL measure ``M``
        for ``M >= 1`` maps directly to ``starts[M]``.

        The anacrusis (LABEL measure ``0``) is special: its offsets are
        measured from a *virtual full-bar downbeat* that precedes the
        first sounding note.  That virtual downbeat sits one nominal bar
        length before the first real boundary, i.e.
        ``starts[1] - <first measure's nominal meter length in quarters>``.

        Args:
            metric_map: The MetricMap built from the timemap.
            first_meter_quarters: Nominal length of the first measure in
                quarters (from the timemap's first meter signature).

        Returns:
            A ``(starts, m0_downbeat)`` pair, where ``starts`` is the list
            of measure-start quarter positions (length ``n_measures``) and
            ``m0_downbeat`` is the virtual downbeat of LABEL measure 0.
        """
        starts = list(metric_map._starts_frac)
        # The bar the anacrusis belongs to nominally begins one full meter
        # length before the first real sounding boundary (``starts[1]``).
        reference = starts[1] if len(starts) > 1 else starts[0]
        m0_downbeat = reference - first_meter_quarters
        return starts, m0_downbeat

    @classmethod
    def _resolve_label(
        cls,
        measure: int,
        offset_wn: Fraction,
        starts: list[Fraction],
        m0_downbeat: Fraction,
    ) -> Fraction:
        """Resolve a single LABEL into absolute quarters.

        ``abs_quarters = measure_start[measure] + offset_wn * 4`` where
        ``offset_wn`` is the within-measure offset expressed in whole
        notes (``* 4`` converts whole notes → quarters).

        Args:
            measure: LABEL measure number (0 = anacrusis).
            offset_wn: Within-measure offset in whole notes.
            starts: Measure-start quarter positions.
            m0_downbeat: Virtual downbeat of LABEL measure 0.
        """
        measure_start = m0_downbeat if measure == 0 else starts[measure]
        return measure_start + offset_wn * 4

    def _build_score_timeline(
        self,
        solo_path: Path,
        starts: list[Fraction],
        m0_downbeat: Fraction,
    ) -> None:
        """Compose SoloLoader and resolve every note onset to quarters.

        Resolution is array-based: the ``measure_number`` and ``mn_onset``
        struct columns are pulled once from the SoloLoader's PyArrow table
        and resolved column-wise into exact ``Fraction`` quarter positions
        — no per-event scalar materialisation on the hot path.
        """
        solo = SoloLoader.from_file(solo_path)
        table = solo.events._table

        measures = [int(r["mn"]) for r in table.column("measure_number").to_pylist()]
        onsets = [
            Fraction(r["numerator"], r["denominator"])
            for r in table.column("mn_onset").to_pylist()
        ]
        # ``pitch`` is the SoloLoader's EnharmonicPitch struct {midi_number}.
        # Carry it through verbatim so the score timeline keeps the
        # EnharmonicPitch affordance (a number-only source's most-expressive
        # faithful pitch type) instead of degrading it to a raw int.
        pitches = table.column("pitch").to_pylist()
        velocities = table.column("velocity").to_pylist()
        note_ids = [r["value"] for r in table.column("note_id").to_pylist()]

        abs_quarters = [
            self._resolve_label(m, off, starts, m0_downbeat)
            for m, off in zip(measures, onsets)
        ]

        score_tl = ContinuousLogicalTimeline(
            length=self._metric_map.total_length,
            unit=self._score_unit,
            uid=_SCORE_TL_ID,
            name=self._name,
        )

        rows: list[dict[str, Any]] = []
        for i, q in enumerate(abs_quarters):
            rows.append(
                {
                    "id": f"score:{i}",
                    "start": q,
                    "event_type": "Note",
                    "pitch": pitches[i],
                    "velocity": velocities[i],
                    "note_id": note_ids[i],
                    "measure_number": measures[i],
                }
            )
        score_tl.add_events(rows)

        # Expose the measure/beat conversion on the timeline.  The
        # MetricalPositionMap is a multi-output CombinationMap (no single
        # target unit), so it is stored as an attached map for inspection
        # rather than registered for unit-based timestamp resolution.
        score_tl.add_conversion_map(self._metrical_position_map)

        self._score_timeline = score_tl

    def _build_performers(
        self,
        alignments_dir: Path,
        starts: list[Fraction],
        m0_downbeat: Fraction,
    ) -> None:
        """Build one physical timeline + claim set per performer."""
        # Group the CSVs by performer key (the note-level stem).
        note_files = sorted(
            p
            for p in alignments_dir.glob("*.csv")
            if not p.stem.endswith(("_bar", "_beats"))
        )

        for note_file in note_files:
            performer_key = note_file.stem
            self._build_one_performer(performer_key, note_file, starts, m0_downbeat)

    def _build_one_performer(
        self,
        performer_key: str,
        note_file: Path,
        starts: list[Fraction],
        m0_downbeat: Fraction,
    ) -> None:
        """Build a single performer's timeline and MatchClaims.

        The physical timeline holds one event per *aligned* note-level row
        (dangling ``"N"`` rows have no audio coordinate).  Claims are
        emitted for all three granularities; bar/beat onsets are score-time
        subsets and need not appear as physical events, since claims carry
        their own target coordinates.
        """
        perf_tl_id = f"perf:{performer_key}:cpt_1"
        stem = note_file.stem

        # Read all three granularity files up front.
        granularity_rows: dict[str, list[tuple[int, Fraction, str]]] = {}
        for gran, suffix in _GRANULARITIES.items():
            path = note_file.with_name(f"{stem}{suffix}.csv")
            granularity_rows[gran] = self._read_alignment_csv(path)

        # Physical timeline length = max aligned TIME of the note level.
        aligned_times = [
            float(time_str)
            for _m, _off, time_str in granularity_rows["note"]
            if time_str != _DANGLING
        ]
        perf_length = max(aligned_times) if aligned_times else 0.0

        perf_tl = ContinuousPhysicalTimeline(
            length=perf_length,
            unit=self._media_unit,
            uid=perf_tl_id,
            name=performer_key,
        )
        perf_events = [
            {
                "id": f"{performer_key}:{i}",
                "start": float(time_str),
                "event_type": "Note",
            }
            for i, (_m, _off, time_str) in enumerate(granularity_rows["note"])
            if time_str != _DANGLING
        ]
        perf_tl.add_events(perf_events)
        self._performance_timelines[performer_key] = perf_tl

        # MatchClaims for every granularity.
        for gran, rows in granularity_rows.items():
            meta = MatchMetadata(
                agent=Agent(
                    name="performance_precision",
                    type=AgentType.software,
                    identifier=gran,
                ),
                certainty=1.0,
            )
            for measure, offset_wn, time_str in rows:
                quarters = self._resolve_label(measure, offset_wn, starts, m0_downbeat)
                if time_str == _DANGLING:
                    self._claims.append(
                        MatchClaim.nomatch(
                            event={"start": quarters},
                            source_tl_id=_SCORE_TL_ID,
                            target_tl_id=perf_tl_id,
                            unit=self._score_unit,
                            metadata=meta,
                        )
                    )
                else:
                    self._claims.append(
                        MatchClaim.from_projection(
                            event={"start": quarters},
                            source_tl_id=_SCORE_TL_ID,
                            target_tl_id=perf_tl_id,
                            target_coord=float(time_str),
                            source_unit=self._score_unit,
                            target_unit=self._media_unit,
                            coord_key="start",
                            metadata=meta,
                        )
                    )

    @staticmethod
    def _read_alignment_csv(path: Path) -> list[tuple[int, Fraction, str]]:
        """Read a ``LABEL,TIME,FRAME`` alignment CSV.

        Returns:
            A list of ``(measure, offset_in_whole_notes, time_str)`` tuples,
            one per data row.  ``time_str`` is kept as a string so the
            dangling ``"N"`` sentinel survives.
        """
        rows: list[tuple[int, Fraction, str]] = []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)  # header
            for record in reader:
                if not record:
                    continue
                label, time_str = record[0], record[1]
                measure_str, offset_str = label.split("+")
                numerator, denominator = offset_str.split("/")
                rows.append(
                    (
                        int(measure_str),
                        Fraction(int(numerator), int(denominator)),
                        time_str,
                    )
                )
        return rows

    # endregion

    # region Domain Object Creation

    def create_bundle(self) -> "AlignmentBundle":
        """Assemble an AlignmentBundle from the loaded specimen.

        Returns:
            An ``AlignmentBundle`` with the score timeline in its own
            ``"score"`` group, each performer timeline standalone, and all
            MatchClaims as cross-group claims.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if self._score_timeline is None:
            raise RuntimeError(
                "No specimen loaded yet. Call load() before create_bundle()."
            )

        bundle = AlignmentBundle(name=self._name)
        bundle.add_timeline(self._score_timeline, uid=_SCORE_TL_ID, as_group="score")
        for perf_tl in self._performance_timelines.values():
            bundle.add_timeline(perf_tl, uid=perf_tl.id)
        bundle.add_match_claims(self._claims)
        return bundle

    def create_timelines(self, id_pattern: str | None = None) -> list["Timeline"]:
        """Return all timelines: ``[score, performer_1, ..., performer_n]``.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.
        """
        if self._score_timeline is None:
            return []
        timelines = [self._score_timeline, *self._performance_timelines.values()]
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Return a single timeline by uid, role, or performer key.

        Args:
            uid: ``"score"`` (or the score uid) for the score timeline; a
                performer key (``"Chopin_Ashkenazy"``) or its uid
                (``"perf:Chopin_Ashkenazy:cpt_1"``) for a performance timeline.

        Raises:
            KeyError: If no timeline matches.
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._score_timeline is None:
            raise RuntimeError(
                "No specimen loaded yet. Call load() before create_timeline()."
            )

        if uid in ("score", _SCORE_TL_ID):
            return self._score_timeline

        if uid in self._performance_timelines:
            return self._performance_timelines[uid]

        for performer_key, perf_tl in self._performance_timelines.items():
            if uid == perf_tl.id:
                return perf_tl

        raise KeyError(
            f"No timeline with uid or role '{uid}'. Available: 'score', "
            + ", ".join(f"'{k}'" for k in self._performance_timelines)
        )

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        n_perf = len(self._performance_timelines)
        n_claims = len(self._claims)
        return f"PerformancePrecisionLoader(performers={n_perf}, " f"claims={n_claims})"

    # endregion


# endregion
