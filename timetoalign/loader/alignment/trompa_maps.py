"""MapsLoader: MAPS score-to-performance alignments as MatchClaims.

A MAPS file (the alignment JSON of the TROMPA project's ``trompa-align``) is a
list of observations, one per score onset::

    [
        {"obs_mean_onset": 0.5, "xml_id": ["n1", "n2"], "obs_num": 1},
        {"obs_mean_onset": 1.25, "xml_id": ["n3"], "obs_num": 2},
    ]

Each row pairs a set of simultaneous score notes, named by their MEI
``xml:id``, with one time in seconds: the mean onset observed for those notes
in a recording. ``obs_num`` numbers the rows from 1.

The loader keeps the rows verbatim and turns them into synchronous projection
:class:`~timetoalign.alignment.claims.MatchClaim` objects between a score
timeline whose events carry those ``xml:id`` values (as
:class:`~timetoalign.loader.score.VerovioLoader` gives) and a performance
timeline::

    loader = MapsLoader.from_file("alignment.json")
    claims = loader.create_match_claims(score, performance)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from timetoalign.alignment.claims import MatchClaim, MatchMetadata
from timetoalign.core import TimeUnit
from timetoalign.core.time import Coordinate
from timetoalign.loader.base import AlignmentLoader

if TYPE_CHECKING:
    from timetoalign.timelines.base import Timeline

_KEYS = ("obs_mean_onset", "xml_id", "obs_num")

#: Prefix of the ids trompa-align writes for what is no element of the score.
_NON_SCORE_ID_PREFIX = "trompa"


class MapsLoader(AlignmentLoader):
    """Load a MAPS alignment file and state it as MatchClaims.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(path)`` — read and validate the rows of one MAPS file.
    2. ``loader.create_match_claims(score, performance)`` — one claim per
       listed ``xml:id``.

    ``load`` refuses a file it cannot read as stated: a missing key,
    ``obs_num`` not consecutive from 1, an onset that is not a JSON number,
    an ``xml_id`` that is not a non-empty list of non-empty strings, and an id
    listed in two rows (two performance times for one note). It also refuses
    two kinds of rows trompa-align can write, because neither pairs a score
    note with a time of the recording: a **negative onset**, and an id
    beginning ``trompa``, which is not an ``xml:id`` of the score. They are
    refused rather than skipped, so a file is never read as more complete
    than it is; remove them to read the rest.

    One loader reads one file: a MAPS file aligns one performance.
    """

    _single_source = True

    def __init__(self) -> None:
        super().__init__()
        self._rows: list[dict[str, Any]] | None = None

    # region Loading

    def _load_source(self, source: Path) -> list[dict[str, Any]]:  # type: ignore[override]
        """Read and validate the rows of one MAPS file.

        Args:
            source: Path to the MAPS JSON file.

        Returns:
            The rows, verbatim.

        Raises:
            ValueError: If the file does not hold valid MAPS rows (see the
                class docstring).
        """
        with open(source, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        _validate_rows(rows, source)
        return rows

    def _accept_source(  # type: ignore[override]
        self, path: Path, source_meta: dict[str, Any], payload: list[dict[str, Any]]
    ) -> None:
        """Keep the rows of the one file this loader reads."""
        self._rows = payload
        self._sources.append(path)
        self._source_metadata.append(source_meta)

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
        metadata: MatchMetadata | None = None,
    ) -> list[MatchClaim]:
        """Return one synchronous claim per ``xml:id`` of every row.

        A claim relates the event named by the id on ``score``, at its start
        coordinate there, to ``performance`` at the row's onset in seconds, as
        the file states it. A performance timeline in another unit reaches
        seconds through its C-Map; an
        :class:`~timetoalign.alignment.bundle.AlignmentBundle` holding both
        timelines converts the claims onto its axis.

        Args:
            score: The timeline holding the events the ids name (searched
                with its children).
            performance: The timeline of the recording.
            metadata: Provenance recorded on every claim.

        Returns:
            The claims, in row order and, within a row, in the file's id
            order.

        Raises:
            RuntimeError: If no file has been loaded.
            KeyError: If an id names no event of ``score``.
            ValueError: If ``performance`` is not in seconds and has no C-Map
                to seconds.
        """
        rows = self._require_rows()
        _require_seconds(performance, rows)
        claims: list[MatchClaim] = []
        for row in rows:
            onset = Coordinate(row["obs_mean_onset"], TimeUnit.seconds)
            for xml_id in row["xml_id"]:
                try:
                    start = score.get_coordinate_for(xml_id, format="coordinate")
                except KeyError:
                    raise KeyError(
                        f"MAPS row {row['obs_num']} names xml:id {xml_id!r}, which is no "
                        f"event of score timeline {score.id!r}; is it the score the file "
                        "was made for?"
                    ) from None
                claims.append(
                    MatchClaim.from_projection(
                        {"id": xml_id, "start": start},
                        score.id,
                        performance.id,
                        onset,
                        source_unit=score.unit,
                        metadata=metadata,
                    )
                )
        return claims

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
        return ["create_match_claims(score, performance)"]

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Number of rows in the loaded file."""
        return len(self._rows) if self._rows is not None else 0

    def __repr__(self) -> str:
        return f"MapsLoader(sources={len(self._sources)}, rows={len(self)})"

    # endregion


def _validate_rows(rows: Any, source: Path) -> None:
    """Raise unless *rows* are MAPS rows this loader reads as stated."""
    if not isinstance(rows, list):
        raise ValueError(f"A MAPS file holds a JSON list of rows; {source} does not.")
    refused: list[str] = []
    seen: dict[str, int] = {}
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or any(key not in row for key in _KEYS):
            raise ValueError(
                f"MAPS row {position} of {source} must be an object with the keys "
                f"{', '.join(_KEYS)}; got {row!r}."
            )
        obs_num = row["obs_num"]
        if isinstance(obs_num, bool) or obs_num != position:
            raise ValueError(
                f"MAPS rows of {source} must be numbered 1, 2, … in order; row "
                f"{position} has obs_num {obs_num!r}."
            )
        onset = row["obs_mean_onset"]
        if (
            isinstance(onset, bool)
            or not isinstance(onset, (int, float))
            or not math.isfinite(onset)
        ):
            raise ValueError(
                f"MAPS row {position} of {source}: obs_mean_onset must be a number; "
                f"got {onset!r}."
            )
        ids = row["xml_id"]
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(xml_id, str) and xml_id for xml_id in ids)
        ):
            raise ValueError(
                f"MAPS row {position} of {source}: xml_id must be a non-empty list of "
                f"non-empty strings; got {ids!r}."
            )
        non_score = [
            xml_id for xml_id in ids if xml_id.startswith(_NON_SCORE_ID_PREFIX)
        ]
        if onset < 0:
            refused.append(f"row {position} (onset {onset})")
        if non_score:
            refused.append(f"row {position} (ids {', '.join(non_score)})")
        if onset < 0 or non_score:
            continue
        for xml_id in ids:
            if xml_id in seen:
                raise ValueError(
                    f"MAPS rows {seen[xml_id]} and {position} of {source} both name "
                    f"xml:id {xml_id!r}: two performance times for one score note."
                )
            seen[xml_id] = position
    if refused:
        raise ValueError(
            f"{source} holds rows that pair no score note with a time of the "
            "recording: a negative onset is no time of it, and an id beginning "
            f"{_NON_SCORE_ID_PREFIX!r} is no xml:id of the score. Remove them to read "
            f"the file: {'; '.join(refused)}."
        )


def _require_seconds(performance: Timeline, rows: list[dict[str, Any]]) -> None:
    """Raise unless a position in seconds resolves on *performance*."""
    if performance.unit == TimeUnit.seconds or not rows:
        return
    try:
        performance.get_coordinate_at(
            Coordinate(rows[0]["obs_mean_onset"], TimeUnit.seconds), format="coordinate"
        )
    except ValueError as error:
        raise ValueError(
            f"MAPS onsets are in seconds, and performance timeline {performance.id!r} "
            f"in {performance.unit.value} has no C-Map to seconds; attach one (for "
            "samples, SamplesToSeconds(sample_rate))."
        ) from error
