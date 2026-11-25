"""MapsLoader: MAPS score-to-performance alignments as MatchClaims.

A MAPS file (the alignment JSON of the TROMPA project's ``trompa-align``) is a
list of observations, one per performance onset::

    [
        {"obs_mean_onset": 0.5, "xml_id": ["n1", "n2"], "obs_num": 1},
        {"obs_mean_onset": 1.25, "xml_id": ["n3"], "obs_num": 2},
    ]

Each row pairs a set of simultaneous score notes, named by their MEI
``xml:id``, with one time in seconds: the onset observed for those notes in a
recording. ``obs_num`` numbers the rows from 1.

trompa-align (https://github.com/trompamusic/trompa-align,
``scripts/trompa_align.py`` and ``scripts/trompa-align.R``) writes two more
kinds of rows, both stating a note on one side only:

* **A score note not performed.** The note-alignment file it reconciles marks
  a missing performance note with ``*``, which it reads as ``-1``; such notes
  are grouped like any other, under ``obs_mean_onset`` ``-1``.
* **A performed note not in the score.** Its id is
  ``trompa-align_inserted_`` followed by the note's spelled pitch (``#``
  written ``s``), so two inserted notes of one pitch share an id; its onset is
  the note's time in the recording. The Python writer gives this id as a bare
  string rather than a one-item list.

The loader keeps the rows verbatim and states each as
:class:`~timetoalign.alignment.claims.MatchClaim` objects between the events
of a score timeline whose events carry those ``xml:id`` values (as
:class:`~timetoalign.loader.score.VerovioLoader` gives) and the Projections
it adds to a performance timeline::

    bundle = MapsLoader.from_file("alignment.json").create_bundle(score, performance)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from typing import TYPE_CHECKING, Any, ClassVar

from timetoalign.alignment.claims import ClaimSide, MatchClaim, _claims_between_events
from timetoalign.core import Provenance, TimeUnit
from timetoalign.core.time import Coordinate
from timetoalign.loader._sources import Source, describe_source
from timetoalign.loader.base import AlignmentLoader

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

_KEYS = ("obs_mean_onset", "xml_id", "obs_num")

#: Prefix of the ids trompa-align writes for a performed note not in the score.
_INSERTED_ID_PREFIX = "trompa-align_inserted_"


class MapsLoader(AlignmentLoader):
    """Load a MAPS alignment file and state it as MatchClaims.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(path)`` — read and validate the rows of one MAPS file.
    2. ``loader.create_bundle(score, performance)`` — both timelines and the
       claims in one bundle; ``create_match_claims`` gives the claims alone.

    Every row is kept as stated, one claim per listed id. The recording has
    no events of its own, so an onset enters the performance timeline as a
    :class:`~timetoalign.core.Projection` on its own axis:

    * a non-negative onset: a synchronous claim between the score note and
      its Projection at the onset;
    * a negative onset (trompa-align's ``-1`` for a note not performed): a
      NOMATCH claim from the score note;
    * an id beginning ``trompa-align_inserted_`` (a performed note not in the
      score): a NOMATCH claim from a Projection at the onset that stands for
      no score note, the id kept.

    An id listed in two rows (say, a repeat played against an unrepeated
    score) gives one claim per row, and ``load`` logs it. ``load`` logs the
    count of each kind, and refuses a file it cannot read as stated: a
    missing key, ``obs_num`` not consecutive from 1, an onset that is not a
    JSON number, an ``xml_id`` that is neither a non-empty string nor a
    non-empty list of them, and an inserted id at a negative onset, which
    states a note on neither side.

    One loader reads one file: a MAPS file aligns one performance. Its
    payload is the rows, not an ``AlignmentStore``, so ``store`` stays
    ``None``.
    """

    _single_source: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] | None = None

    # region Loading

    # The payload is the rows; AlignmentLoader's hooks are typed for a store.
    def _load_source(self, source: Source) -> list[dict[str, Any]]:  # type: ignore[override]
        """Read and validate the rows of one MAPS file.

        Args:
            source: The MAPS JSON file: a path or a ZIP archive member.

        Returns:
            The rows, verbatim.

        Raises:
            ValueError: If the file does not hold valid MAPS rows (see the
                class docstring).
        """
        with source.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        _validate_rows(rows, source)
        return rows

    # The payload is the rows; AlignmentLoader's hooks are typed for a store.
    def _accept_source(  # type: ignore[override]
        self, path: Source, source_meta: dict[str, Any], payload: list[dict[str, Any]]
    ) -> None:
        """Keep the rows of the one file this loader reads, and log their kinds."""
        self._rows = payload
        self._sources.append(path)
        self._source_metadata.append(source_meta)
        kinds: Counter[str] = Counter()
        rows_by_id: dict[str, list[int]] = {}
        for row in payload:
            for xml_id in _ids(row):
                if xml_id.startswith(_INSERTED_ID_PREFIX):
                    kinds["inserted"] += 1
                    continue
                kinds["unperformed" if row["obs_mean_onset"] < 0 else "aligned"] += 1
                rows_by_id.setdefault(xml_id, []).append(row["obs_num"])
        self._logger.info(
            "%s: %d score note(s) at a time of the recording, %d score note(s) "
            "not performed (negative onset), %d performed note(s) not in the score",
            path.name,
            kinds["aligned"],
            kinds["unperformed"],
            kinds["inserted"],
        )
        repeated = {
            xml_id: rows for xml_id, rows in rows_by_id.items() if len(rows) > 1
        }
        if repeated:
            self._logger.warning(
                "%s lists %d xml:id(s) in more than one row; each row's claim is "
                "kept: %s",
                path.name,
                len(repeated),
                "; ".join(
                    f"{xml_id!r} in rows {', '.join(map(str, rows))}"
                    for xml_id, rows in repeated.items()
                ),
            )

    def clear(self) -> None:
        """Forget the loaded file."""
        super().clear()
        self._rows = None

    # endregion

    # region Properties

    @property
    def rows(self) -> list[dict[str, Any]]:
        """The rows of the loaded file, as it states them."""
        return list(self._require_rows())

    # endregion

    # region Domain Object Creation

    def create_match_claims(
        self,
        score: Timeline,
        performance: Timeline,
        *,
        provenance: Provenance | None = None,
    ) -> list[MatchClaim]:
        """Return one claim per listed id of every row, as the row states it.

        Every claim names its events. The recording holds no event of its
        own, so each onset is a :class:`~timetoalign.core.Projection` on
        ``performance``, its position the onset converted onto the
        performance timeline's axis (through its C-Map where that is not in
        seconds: a samples timeline holds samples, never seconds); the
        claims are made by the library's one projection operation
        (:meth:`AlignmentBundle.create_match_claims
        <timetoalign.alignment.bundle.AlignmentBundle.create_match_claims>`
        takes the same sides):

        * a note at a non-negative onset: a Projection of the score note
          (``projection_of`` naming it on ``score``) and a synchronous claim
          between the score note, at its start coordinate on ``score``, and
          that Projection;
        * a note at a negative onset: a NOMATCH claim from the score note;
          nothing is added to ``performance``;
        * an inserted id: a Projection standing for a note the score does not
          hold (``projection_of`` null) and a NOMATCH claim from it. The
          Projection keeps the file's id; where trompa-align gives two
          inserted notes one id (it spells the pitch only), each Projection
          is minted an id of its own and the claim keeps the file's id as
          the event's name.

        ``performance`` keeps one Projection per score note and onset: a
        second call on the same timelines reuses every Projection the first
        added (an inserted note under its kept id is then the event the
        timeline holds), so it adds nothing to ``performance``.

        Args:
            score: The timeline holding the events the ids name (searched
                with its children); the claims name it and give each note's
                coordinate on its axis.
            performance: The timeline of the recording, which receives the
                Projections.
            provenance: Provenance recorded on every claim.

        Returns:
            The claims, in row order and, within a row, in the file's id
            order.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If an id of a score note names no event of ``score``;
                nothing is added to ``performance`` then.
            ValueError: If ``performance`` is not in seconds and has no C-Map
                to seconds, or an onset does not fit on it; nothing is added
                to ``performance`` then.
        """
        rows = self._require_rows()
        _require_seconds(performance, rows)
        inserted = Counter(
            xml_id
            for row in rows
            for xml_id in _ids(row)
            if xml_id.startswith(_INSERTED_ID_PREFIX)
        )
        pairs: list[tuple[ClaimSide, Timeline, ClaimSide, Timeline]] = []
        for row in rows:
            onset = Coordinate(row["obs_mean_onset"], TimeUnit.seconds)
            for xml_id in _ids(row):
                if not xml_id.startswith(_INSERTED_ID_PREFIX):
                    performed = None if onset.value < 0 else {"start": onset}
                    pairs.append((xml_id, score, performed, performance))
                elif inserted[xml_id] == 1:
                    pairs.append(
                        ({"id": xml_id, "start": onset}, performance, None, score)
                    )
                else:
                    stated = {"start": onset, "name": xml_id}
                    pairs.append((stated, performance, None, score))
        return _claims_between_events(pairs, provenance=provenance)

    def create_bundle(
        self,
        score: Timeline,
        performance: Timeline,
        *,
        provenance: Provenance | None = None,
    ) -> AlignmentBundle:
        """Return a bundle of both timelines, related by the file's claims.

        The score joins as group ``"score"``, the performance as group
        ``"performance"``, and then :meth:`create_match_claims` gives the
        claims: the Projections are added while both timelines are in the
        bundle, so their IDs are minted unique among all Projections of the
        bundle (above any the score holds) and the bundle's package can be
        written. The bundle is named after the file.

        Args:
            score: The timeline holding the events the ids name.
            performance: The timeline of the recording.
            provenance: Provenance recorded on every claim.

        Returns:
            The bundle.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If an id of a score note names no event of ``score``.
            ValueError: If ``performance`` is not in seconds and has no C-Map
                to seconds, or the two timelines cannot join one bundle
                (:meth:`AlignmentBundle.add_timeline
                <timetoalign.alignment.bundle.AlignmentBundle.add_timeline>`).
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        _require_seconds(performance, self._require_rows())
        bundle = AlignmentBundle(name=self._sources[0].stem)
        bundle.add_timeline(score, as_group="score")
        bundle.add_timeline(performance, as_group="performance")
        bundle.add_match_claims(
            self.create_match_claims(score, performance, provenance=provenance)
        )
        return bundle

    # endregion

    def _require_rows(self) -> list[dict[str, Any]]:
        """Return the loaded rows, or raise if nothing is loaded."""
        if self._rows is None:
            raise RuntimeError("No MAPS file loaded yet. Call load() first.")
        return self._rows

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is rows, not store events."""
        return ("Rows", str(len(self)))

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippet for the ``Try`` row."""
        return [
            "create_bundle(score, performance)",
            "create_match_claims(score, performance)",
        ]

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Number of rows in the loaded file."""
        return len(self._rows) if self._rows is not None else 0

    def __repr__(self) -> str:
        return f"MapsLoader(sources={len(self._sources)}, rows={len(self)})"

    # endregion


def _ids(row: dict[str, Any]) -> list[str]:
    """Return a row's ids; a bare string is one id."""
    ids = row["xml_id"]
    return [ids] if isinstance(ids, str) else ids


def _validate_rows(rows: Any, source: Source) -> None:
    """Raise unless *rows* are MAPS rows this loader reads as stated."""
    if not isinstance(rows, list):
        raise ValueError(
            f"A MAPS file holds a JSON list of rows; {describe_source(source)} does not."
        )
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or any(key not in row for key in _KEYS):
            raise ValueError(
                f"MAPS row {position} of {describe_source(source)} must be an object with the keys "
                f"{', '.join(_KEYS)}; got {row!r}."
            )
        obs_num = row["obs_num"]
        if isinstance(obs_num, bool) or obs_num != position:
            raise ValueError(
                f"MAPS rows of {describe_source(source)} must be numbered 1, 2, … in order; row "
                f"{position} has obs_num {obs_num!r}."
            )
        onset = row["obs_mean_onset"]
        if (
            isinstance(onset, bool)
            or not isinstance(onset, (int, float))
            or not math.isfinite(onset)
        ):
            raise ValueError(
                f"MAPS row {position} of {describe_source(source)}: obs_mean_onset must be a number; "
                f"got {onset!r}."
            )
        ids = row["xml_id"]
        if (
            not isinstance(ids, (str, list))
            or not ids
            or not all(isinstance(xml_id, str) and xml_id for xml_id in _ids(row))
        ):
            raise ValueError(
                f"MAPS row {position} of {describe_source(source)}: xml_id must be a non-empty string "
                f"or a non-empty list of them; got {ids!r}."
            )
        inserted = [
            xml_id for xml_id in _ids(row) if xml_id.startswith(_INSERTED_ID_PREFIX)
        ]
        if onset < 0 and inserted:
            raise ValueError(
                f"MAPS row {position} of {describe_source(source)} states the inserted note(s) "
                f"{', '.join(inserted)} at the negative onset {onset}: a note on "
                "neither the score nor the recording."
            )


def _require_seconds(performance: Timeline, rows: list[dict[str, Any]]) -> None:
    """Raise unless a position in seconds resolves on *performance*.

    The probe is the first non-negative onset; with none, no claim places
    anything on *performance*.
    """
    onsets = [row["obs_mean_onset"] for row in rows if row["obs_mean_onset"] >= 0]
    if performance.unit == TimeUnit.seconds or not onsets:
        return
    try:
        performance.get_coordinate_at(
            Coordinate(onsets[0], TimeUnit.seconds), format="coordinate"
        )
    except ValueError as error:
        raise ValueError(
            f"MAPS onsets are in seconds, and performance timeline {performance.id!r} "
            f"in {performance.unit.value} has no C-Map to seconds; attach one (for "
            "samples, SamplesToSeconds(sample_rate))."
        ) from error
