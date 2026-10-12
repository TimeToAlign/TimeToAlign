"""ListenHereLoader: a dense audio-to-audio AlignmentBundle from one JSON file.

A *Listen Here!* alignment export is a single JSON file describing how many
recordings of one musical work line up in time.  Every recording is warped onto
a shared, equidistant **reference grid**: the file stores, per recording, a
``times`` array whose ``i``-th entry is that recording's clock-time (in seconds)
at reference-grid column ``i``.  Because every recording is sampled against the
same grid, the per-recording arrays are **parallel and equal length** — together
they form a dense alignment matrix of shape ``(recordings × grid columns)``.

Unlike a pairwise score-to-performance export (one score against one
performance), a single Listen Here! file already encodes the alignment of *all*
recordings against one another.  The natural TimeToAlign! reading is therefore
a **complete pairwise topology**: at every grid column, every unordered pair of
recordings ``(a, b)`` is directly related by a synchronous instant
:class:`~timetoalign.alignment.claims.MatchClaim`
(``a@times_a[i] ↔ b@times_b[i]``).  For ``R`` recordings and ``N`` grid columns
this is ``C(R, 2) × N`` claims — a very large set for a whole work — so the
claims are held columnar in a
:class:`~timetoalign.alignment.claims.MatchClaimField` and assembled
**vectorized** (never one Python claim object per row).

``ListenHereLoader`` ingests one alignment JSON file in one call::

    bundle = ListenHereLoader.from_file(alignment_json).create_bundle()

and produces a single :class:`~timetoalign.alignment.bundle.AlignmentBundle`:

* one samples
  :class:`~timetoalign.timelines.types.DiscretePhysicalTimeline`
  (``<stem>:dpt_1``) per recording, each in its **own** group, with
  ``length`` converted from that recording's stored ``duration``.  The recordings
  carry one point event for each source-grid time.  Its integer sample
  coordinate supports the alignment, while its ``seconds`` field preserves the
  verbatim JSON value; all pairwise alignment lives in the cross-group claim
  field; and
* the complete-topology pairwise claims, reached through the uniform field
  API ``loader.get_field(MatchClaim)`` -> :class:`MatchClaimField`.

The reference recording named by ``header.ref`` fixes the grid origin but is
**not** a privileged hub: it is just another recording, related to every other
by direct pairwise claims like any other pair.

The loader reads the *existing* alignment; it never runs an aligner.

A recording's early grid columns may carry **negative** seconds — a recording
that begins after the reference's grid origin is extrapolated backwards before
its first onset.  These negative coordinates are stored faithfully, neither
clamped nor dropped.

See Also:
    timetoalign.AlignmentBundle
    timetoalign.MatchClaim
    timetoalign.MatchClaimField
"""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import Self

from timetoalign.alignment.claims import (
    Agent,
    MatchClaim,
    MatchClaimField,
    MatchMetadata,
)
from timetoalign.core import AgentType, TimeUnit
from timetoalign.core.fields import SemanticField
from timetoalign.display.html import code
from timetoalign.loader.base import AlignmentLoader
from timetoalign.loader.physical.audio import AudioLoader
from timetoalign.maps import SamplesToSeconds
from timetoalign.timelines.types import DiscretePhysicalTimeline

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)


# region Constants

#: Provenance recorded on every materialised claim.  The agent is read from the
#: file's ``header.createdBy`` (falling back to a generic label); the agent
#: identifier names the alignment method Listen Here! uses (chroma-feature DTW).
_DEFAULT_AGENT = "Listen Here!"
_AGENT_IDENTIFIER = "dtw_chroma_alignment"

#: Fallback for exports whose recordings are not available beside the JSON.
_DEFAULT_SAMPLE_RATE = 44100

# endregion


# region ListenHereLoader


class ListenHereLoader(AlignmentLoader):
    """Load a Listen Here! alignment JSON file as one AlignmentBundle.

    The loader parses a single alignment JSON file describing many recordings
    of one work warped onto a shared equidistant reference grid, and assembles
    a bundle in which every recording is a samples timeline in its own group,
    related to every other recording by a complete set of pairwise synchronous
    instant :class:`MatchClaim` rows held columnar in a :class:`MatchClaimField`.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(alignment_json)`` — parse one alignment JSON file.
    2. ``loader.create_bundle()`` — assemble the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines.

    The loader reads the existing alignment; it never runs an aligner.
    """

    def __init__(self) -> None:
        super().__init__()

        # Sorted audio keys (file names / URLs from ``body.audio``).
        self._keys: list[str] = []
        # Per key: the stem (basename without extension) and the recording's
        # duration in seconds.
        self._stems: dict[str, str] = {}
        self._durations: dict[str, float] = {}
        self._sample_rates: dict[str, int] = {}
        self._sample_rate_provenance: dict[str, str] = {}
        self._times_by_key: dict[str, list[float]] = {}
        self._claim_field: MatchClaimField | None = None
        self._name: str | None = None
        # The reference recording key (``header.ref``): the grid origin.
        self._ref: str | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Path) -> Any:
        """Not used: a Listen Here! export is a single coherent file.

        :class:`ListenHereLoader` ingests one alignment JSON file through
        :meth:`load`; the base class's per-source AlignmentStore merge does
        not apply.
        """
        raise NotImplementedError(
            "ListenHereLoader ingests one alignment JSON file via "
            "load(alignment_json); per-source loading is not used."
        )

    # endregion

    # region Loading

    def load(self, source: str | Path) -> Self:
        """Ingest one Listen Here! alignment JSON file.

        Args:
            source: Path to the ``alignment.json`` file.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``header.ref`` is missing or is not one of the
                ``body.audio`` keys; if fewer than two recordings are present;
                or if the per-recording ``times`` arrays are not all the same
                length.
        """
        path = Path(source)
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        header = data.get("header", {})
        audio = data.get("body", {}).get("audio", {})

        keys = sorted(audio.keys())
        if len(keys) < 2:
            raise ValueError(
                f"A Listen Here! alignment needs at least 2 recordings; "
                f"found {len(keys)} in {path}."
            )

        ref = header.get("ref")
        if ref not in audio:
            raise ValueError(
                f"header.ref {ref!r} is not one of the body.audio keys {keys} in {path}."
            )

        # Normalise each entry to (times, duration) and validate equal lengths.
        times_by_key: dict[str, list[float]] = {}
        for key in keys:
            times, duration = self._normalise_entry(audio[key])
            times_by_key[key] = times
            self._durations[key] = duration
            self._stems[key] = os.path.splitext(os.path.basename(key))[0]
            sample_rate, provenance = self._resolve_sample_rate(path, key)
            self._sample_rates[key] = sample_rate
            self._sample_rate_provenance[key] = provenance

        lengths = {key: len(times_by_key[key]) for key in keys}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All recordings' 'times' arrays must have the same length "
                f"(they index a shared reference grid); got {lengths} in {path}."
            )

        self._keys = keys
        self._times_by_key = times_by_key
        self._name = path.stem
        self._ref = ref
        agent = header.get("createdBy", _DEFAULT_AGENT)
        self._claim_field = self._build_claim_field(
            keys, times_by_key, self._sample_rates, agent
        )
        self._sources = [path]
        return self

    @staticmethod
    def _normalise_entry(entry: Any) -> tuple[list[float], float]:
        """Normalise one ``body.audio`` value to ``(times, duration)``.

        A value may be either an object ``{"times": [...], "peaks": [...],
        "duration": <float>}`` or a bare array of times.  ``peaks`` (a
        waveform-display payload) is ignored.  For the bare-array form the
        duration is taken as the maximum time.

        Args:
            entry: The ``body.audio[key]`` value (object or bare list).

        Returns:
            A ``(times, duration)`` pair.
        """
        if isinstance(entry, dict):
            times = [float(t) for t in entry["times"]]
            duration = float(entry["duration"])
            return times, duration
        times = [float(t) for t in entry]
        duration = max(times) if times else 0.0
        return times, duration

    @staticmethod
    def _resolve_sample_rate(source: Path, key: str) -> tuple[int, str]:
        """Return a recording's rate and whether it came from the file.

        Audio paths in Listen Here! JSON are conventionally relative to the
        export.  An absolute path is also accepted.  Missing or unreadable
        recordings retain usable coordinates via the documented 44100 Hz
        fallback.
        """
        key_path = Path(key)
        candidate = key_path if key_path.is_absolute() else source.parent / key_path
        if candidate.is_file():
            try:
                return AudioLoader.from_file(candidate).sample_rate, "file"
            except (OSError, ValueError):
                module_logger.debug(
                    "Could not read audio metadata from %s; assuming %d Hz.",
                    candidate,
                    _DEFAULT_SAMPLE_RATE,
                )
        return _DEFAULT_SAMPLE_RATE, "assumed"

    def _build_claim_field(
        self,
        keys: list[str],
        times_by_key: dict[str, list[float]],
        sample_rates: dict[str, int],
        agent: str,
    ) -> MatchClaimField:
        """Build the complete-topology pairwise claim field, vectorized.

        For every unordered pair of recordings ``(a, b)`` and every reference
        grid column ``i``, a synchronous instant claim relates ``a`` at
        ``times_a[i]`` to ``b`` at ``times_b[i]``.  The four parallel columns
        are concatenated across all pairs with numpy and handed to
        :meth:`MatchClaimField.from_columns`, so no :class:`MatchClaim` object
        is constructed.

        Args:
            keys: The sorted audio keys.
            times_by_key: Each key's per-grid-column times (seconds).
            sample_rates: The resolved sample rate for each recording.
            agent: The provenance agent (``header.createdBy``).

        Returns:
            A :class:`MatchClaimField` of ``C(R, 2) × N`` synchronous claims.
        """
        coord_arrays = {
            key: np.rint(
                np.asarray(times_by_key[key], dtype=np.float64) * sample_rates[key]
            ).astype(np.int64)
            for key in keys
        }
        uid_by_key = {key: self._timeline_uid(key) for key in keys}
        ncols = len(next(iter(coord_arrays.values()))) if keys else 0

        a_id_blocks: list[list[str]] = []
        b_id_blocks: list[list[str]] = []
        coord_a_blocks: list[np.ndarray] = []
        coord_b_blocks: list[np.ndarray] = []

        for key_a, key_b in itertools.combinations(keys, 2):
            a_id_blocks.append([uid_by_key[key_a]] * ncols)
            b_id_blocks.append([uid_by_key[key_b]] * ncols)
            coord_a_blocks.append(coord_arrays[key_a])
            coord_b_blocks.append(coord_arrays[key_b])

        timeline_a_ids = [uid for block in a_id_blocks for uid in block]
        timeline_b_ids = [uid for block in b_id_blocks for uid in block]
        coordinate_a = (
            np.concatenate(coord_a_blocks)
            if coord_a_blocks
            else np.empty(0, dtype=np.float64)
        )
        coordinate_b = (
            np.concatenate(coord_b_blocks)
            if coord_b_blocks
            else np.empty(0, dtype=np.float64)
        )

        metadata = MatchMetadata(
            agent=Agent(
                name=agent,
                type=AgentType.software,
                identifier=_AGENT_IDENTIFIER,
            ),
            certainty=1.0,
        )
        return MatchClaimField.from_columns(
            timeline_a_ids,
            timeline_b_ids,
            coordinate_a,
            coordinate_b,
            unit_a=TimeUnit.samples,
            unit_b=TimeUnit.samples,
            metadata=metadata,
        )

    @staticmethod
    def _timeline_uid(key: str) -> str:
        """The samples-timeline uid for an audio key: ``<stem>:dpt_1``."""
        stem = os.path.splitext(os.path.basename(key))[0]
        return f"{stem}:dpt_1"

    @staticmethod
    def _human_name(stem: str) -> str:
        """A human-readable timeline name from a stem (``-`` → space)."""
        return stem.replace("-", " ")

    # endregion

    # region Field access

    def get_field(
        self,
        selector: type[MatchClaim] | type[SemanticField[Any]],
    ) -> MatchClaimField:
        """Return the complete-topology pairwise claim field.

        The loader's alignment is reached through the uniform field API, the
        same way any :class:`~timetoalign.storage.mixins.SemanticFieldAccessMixin`
        surfaces a semantic view:

            >>> field = loader.get_field(MatchClaim)
            >>> isinstance(field, MatchClaimField)
            True

        The selector may be the :class:`MatchClaim` scalar class or its paired
        :class:`MatchClaimField` class; both resolve to the single
        ``MatchClaimField`` this loader builds.

        Args:
            selector: ``MatchClaim`` or ``MatchClaimField``.

        Returns:
            The :class:`MatchClaimField` of ``C(R, 2) × N`` synchronous claims.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
            TypeError: If ``selector`` is not ``MatchClaim`` /
                ``MatchClaimField``.
        """
        if self._claim_field is None:
            raise RuntimeError(
                "No alignment loaded yet. Call load() before get_field()."
            )
        if selector is MatchClaim or selector is MatchClaimField:
            return self._claim_field
        name = getattr(selector, "__name__", repr(selector))
        raise TypeError(
            f"ListenHereLoader.get_field() resolves only MatchClaim / MatchClaimField; got {name}."
        )

    # endregion

    # region Properties

    @property
    def reference(self) -> str | None:
        """The reference recording (``header.ref``) — the grid origin.

        ``None`` until :meth:`load` has been called.
        """
        return self._ref

    @property
    def recording_keys(self) -> list[str]:
        """The recording stems, in sorted order."""
        return [self._stems[key] for key in self._keys]

    # endregion

    # region Domain Object Creation

    def create_bundle(self) -> "AlignmentBundle":
        """Assemble the AlignmentBundle from the loaded alignment.

        Each recording becomes a samples
        :class:`DiscretePhysicalTimeline` in its own group, and the complete
        pairwise claim field is added to the bundle **columnar**.

        The field is handed to the bundle with
        :meth:`~timetoalign.alignment.bundle.AlignmentBundle.add_match_claim_field`,
        not exploded into a list of :class:`MatchClaim` objects.  The bundle
        answers ``get_matchstamp_at`` by filtering the field's struct column
        vectorized and materialising only the handful of claims at the queried
        coordinate, so a whole-work file (on the order of a million claims)
        never builds a million Python claims.

        Returns:
            An ``AlignmentBundle`` with one group per recording and the
            cross-group pairwise claim field.

        Raises:
            RuntimeError: If ``load()`` has not been called yet.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if self._claim_field is None:
            raise RuntimeError(
                "No alignment loaded yet. Call load() before create_bundle()."
            )

        bundle = AlignmentBundle(name=self._name)
        for key in self._keys:
            timeline = self._make_timeline(key)
            stem = self._stems[key]
            bundle.add_timeline(timeline, as_group=stem)

        bundle.add_match_claim_field(self._claim_field)
        return bundle

    def create_timelines(self, id_pattern: str | None = None) -> list["Timeline"]:
        """Return one samples timeline per recording, in sorted order.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.
        """
        timelines = [self._make_timeline(key) for key in self._keys]
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Return a single recording's samples timeline by its uid.

        Args:
            uid: A timeline uid (``"<stem>:dpt_1"``).

        Raises:
            KeyError: If no recording matches.
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._claim_field is None:
            raise RuntimeError(
                "No alignment loaded yet. Call load() before create_timeline()."
            )

        for key in self._keys:
            if uid == self._timeline_uid(key):
                return self._make_timeline(key)

        available = [self._timeline_uid(key) for key in self._keys]
        raise KeyError(
            f"No timeline with uid '{uid}'. Available: "
            + ", ".join(f"'{uid}'" for uid in available)
        )

    def _make_timeline(self, key: str) -> DiscretePhysicalTimeline:
        """Build one samples timeline with faithful source-time events."""
        stem = self._stems[key]
        sample_rate = self._sample_rates[key]
        timeline = DiscretePhysicalTimeline(
            length=int(round(self._durations[key] * sample_rate)),
            unit=TimeUnit.samples,
            uid=self._timeline_uid(key),
            name=self._human_name(stem),
            meta={
                "sample_rate": sample_rate,
                "sample_rate_provenance": self._sample_rate_provenance[key],
            },
        )
        timeline.add_events(
            [
                {
                    "event_type": "ListenHerePoint",
                    "instant": int(round(seconds * sample_rate)),
                    "seconds": np.float64(seconds),
                }
                for seconds in self._times_by_key[key]
            ]
        )
        timeline.add_conversion_map(SamplesToSeconds(sample_rate=sample_rate))
        return timeline

    # endregion

    # region HTML Representation

    def _repr_count_row(self) -> tuple[str, str]:
        """This loader's payload is pairwise claims, not store events."""
        return ("Claims", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Extend the base rows with the Listen Here!-specific shape.

        The base :class:`AlignmentLoader` count row is replaced by the
        claim count (see :meth:`_repr_count_row`); the data lives in the
        assembled timelines and the claim field, not the unpopulated
        per-source ``AlignmentStore``.  One samples timeline per recording,
        each in its own group.
        """
        rows = super()._repr_rows()
        n_recordings = len(self._keys)
        name = self._name or "(not loaded)"
        ref = self._ref or "(not loaded)"
        rows.append(("File", code(name)))
        rows.append(("Recordings", str(n_recordings)))
        rows.append(("Reference", code(ref)))
        if self._claim_field is not None:
            rows.append(("Timelines", f"{n_recordings} in {n_recordings} group(s)"))
            stems = ", ".join(code(s) for s in self.recording_keys)
            rows.append(("Recording keys", stems))
        return rows

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Number of pairwise claims in the loaded claim field.

        The inherited count reads the per-source ``AlignmentStore``, which this
        single-file loader never populates; the claim count is the meaningful
        size and keeps :meth:`_repr_html_` consistent with :meth:`__repr__`.
        """
        return len(self._claim_field) if self._claim_field is not None else 0

    def __repr__(self) -> str:
        n_recordings = len(self._keys)
        n_claims = len(self._claim_field) if self._claim_field is not None else 0
        ref = self._ref if self._ref is not None else "(not loaded)"
        return f"ListenHereLoader(recordings={n_recordings}, reference={ref!r}, claims={n_claims})"

    # endregion


# endregion
