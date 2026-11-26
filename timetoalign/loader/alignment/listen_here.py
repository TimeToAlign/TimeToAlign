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

The file states positions on a grid and nothing else: no note, beat or other
event of any recording.  Every claim therefore names the sentinel
:data:`~timetoalign.core.ids.CHUNK` on both sides — a position on a timeline
that holds no event for it — and its anchor holds the two positions, in
seconds, the file's numbers exactly.  The recordings' timelines hold no
events.

``ListenHereLoader`` ingests one alignment JSON file in one call::

    bundle = ListenHereLoader.from_file(alignment_json).create_bundle()

and produces a single :class:`~timetoalign.alignment.bundle.AlignmentBundle`:

* one seconds
  :class:`~timetoalign.timelines.types.ContinuousPhysicalTimeline`
  (``cpt_…``, number type float, named after the recording's stem) per
  recording, each in its **own** group, its ``length`` that recording's
  stored ``duration`` in seconds; no events; and
* the complete-topology pairwise claims, ``CHUNK`` on both sides, reached
  through the uniform field API ``loader.get_field(MatchClaim)`` ->
  :class:`MatchClaimField`.

The loader represents what the file gives: the seconds of the ``times``
arrays, unchanged.  It reads no recording, so it states no sample rate and
no samples timeline.

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
import os
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
from typing_extensions import Self

from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import CHUNK, NumberType, Provenance, TimeUnit
from timetoalign.core.fields import SemanticField
from timetoalign.display.html import code
from timetoalign.loader._sources import (
    Source,
    SourceLike,
    as_source,
    describe_source,
)
from timetoalign.loader.base import AlignmentLoader
from timetoalign.timelines.factory import mint_timeline_id
from timetoalign.timelines.types import ContinuousPhysicalTimeline

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline


# region Constants

#: Provenance recorded on every materialised claim: imported, with a note
#: naming the file's ``header.createdBy`` (falling back to a generic label) and
#: the alignment method Listen Here! uses (chroma-feature DTW).
_DEFAULT_AGENT = "Listen Here!"
_AGENT_IDENTIFIER = "dtw_chroma_alignment"


# endregion


# region ListenHereLoader


class ListenHereLoader(AlignmentLoader):
    """Load a Listen Here! alignment JSON file as one AlignmentBundle.

    The loader parses a single alignment JSON file describing many recordings
    of one work warped onto a shared equidistant reference grid, and assembles
    a bundle in which every recording is a seconds timeline in its own group,
    related to every other recording by a complete set of pairwise synchronous
    instant :class:`MatchClaim` rows held columnar in a :class:`MatchClaimField`.
    The file states grid positions, not events, so every claim is
    :data:`~timetoalign.core.ids.CHUNK` on both sides and the timelines hold
    no events.

    **Usage follows the standard loader two-phase pattern:**

    1. ``loader.load(alignment_json)`` — parse one alignment JSON file.
    2. ``loader.create_bundle()`` — assemble the AlignmentBundle.
    3. ``loader.create_timeline(uid)`` / ``create_timelines()`` — retrieve
       individual timelines.

    The loader reads the existing alignment; it never runs an aligner.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)

        # Sorted audio keys (file names / URLs from ``body.audio``).
        self._keys: list[str] = []
        # Per key: the stem (basename without extension) and the recording's
        # duration in seconds.
        self._stems: dict[str, str] = {}
        self._durations: dict[str, float] = {}
        self._claim_field: MatchClaimField | None = None
        # IDs minted once per load, so that every timeline built from the
        # loader holds them.
        self._uid_by_key: dict[str, str] = {}
        self._name: str | None = None
        # The reference recording key (``header.ref``): the grid origin.
        self._ref: str | None = None

    # region Abstract-method satisfaction

    def _load_source(self, source: Source) -> Any:
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

    def load(self, source: SourceLike) -> Self:
        """Ingest one Listen Here! alignment JSON file.

        Args:
            source: Path to the ``alignment.json`` file, or its member in a
                ZIP archive.

        Returns:
            Self, for method chaining.

        Raises:
            ValueError: If ``header.ref`` is missing or is not one of the
                ``body.audio`` keys; if fewer than two recordings are present;
                or if the per-recording ``times`` arrays are not all the same
                length.
        """
        path = as_source(source)
        self._start_minter(path)
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        header = data.get("header", {})
        audio = data.get("body", {}).get("audio", {})

        keys = sorted(audio.keys())
        if len(keys) < 2:
            raise ValueError(
                f"A Listen Here! alignment needs at least 2 recordings; "
                f"found {len(keys)} in {describe_source(path)}."
            )

        ref = header.get("ref")
        if ref not in audio:
            raise ValueError(
                f"header.ref {ref!r} is not one of the body.audio keys {keys} in {describe_source(path)}."
            )

        # Normalise each entry to (times, duration) and validate equal lengths.
        times_by_key: dict[str, list[float]] = {}
        for key in keys:
            times, duration = self._normalise_entry(audio[key])
            times_by_key[key] = times
            self._durations[key] = duration
            self._stems[key] = os.path.splitext(os.path.basename(key))[0]

        lengths = {key: len(times_by_key[key]) for key in keys}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "All recordings' 'times' arrays must have the same length "
                f"(they index a shared reference grid); got {lengths} in {describe_source(path)}."
            )

        self._keys = keys
        minter = self._id_minter
        self._uid_by_key = {
            key: mint_timeline_id(TimeUnit.seconds, minter) for key in keys
        }
        self._name = path.stem
        self._ref = ref
        agent = header.get("createdBy", _DEFAULT_AGENT)
        self._claim_field = self._build_claim_field(keys, times_by_key, agent)
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

    def _build_claim_field(
        self,
        keys: list[str],
        times_by_key: dict[str, list[float]],
        agent: str,
    ) -> MatchClaimField:
        """Build the complete-topology pairwise claim field, vectorized.

        For every unordered pair of recordings ``(a, b)`` and every column
        ``i``, a synchronous instant claim joins ``times_a[i]`` on ``a`` to
        ``times_b[i]`` on ``b``, both in seconds as the file states them,
        with :data:`CHUNK` on both
        sides (the file names no event).  The parallel columns are
        concatenated across all pairs with numpy and handed to
        :meth:`MatchClaimField.from_columns`, so no :class:`MatchClaim` object
        is constructed.

        Args:
            keys: The sorted audio keys.
            times_by_key: Each key's per-grid-column times (seconds).
            agent: Who created the export (``header.createdBy``), named in the
                provenance note.

        Returns:
            A :class:`MatchClaimField` of ``C(R, 2) × N`` synchronous claims.
        """
        coord_arrays = {
            key: np.asarray(times_by_key[key], dtype=np.float64) for key in keys
        }
        uid_by_key = self._uid_by_key
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
        # The file states positions, no events: CHUNK on both sides.
        event_ids = pa.repeat(pa.scalar(CHUNK), len(timeline_a_ids))

        provenance = Provenance(method="imported", note=f"{agent} {_AGENT_IDENTIFIER}")
        return MatchClaimField.from_columns(
            timeline_a_ids,
            timeline_b_ids,
            event_ids,
            event_ids,
            pa.array(coordinate_a),
            pa.array(coordinate_b),
            timelines={
                uid: (TimeUnit.seconds, NumberType.float) for uid in uid_by_key.values()
            },
            provenance=provenance,
            minter=self._id_minter,
        )

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

        Each recording becomes a seconds
        :class:`ContinuousPhysicalTimeline` in its own group, and the complete
        pairwise claim field is added to the bundle **columnar**.

        The field is handed to the bundle with
        :meth:`~timetoalign.alignment.bundle.AlignmentBundle.add_match_claim_field`,
        not exploded into a list of :class:`MatchClaim` objects.  The bundle
        answers ``get_matchstamp_at`` by filtering the field's claim columns
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
        """Return one seconds timeline per recording, in sorted order.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.
        """
        timelines: list[Timeline] = [self._make_timeline(key) for key in self._keys]
        return self._filter_timelines_by_id_pattern(timelines, id_pattern)

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Return a single recording's seconds timeline by its uid.

        Args:
            uid: A timeline uid (``cpt_…``), as :meth:`create_timelines`
                lists them.

        Raises:
            KeyError: If no recording matches.
            RuntimeError: If ``load()`` has not been called yet.
        """
        if self._claim_field is None:
            raise RuntimeError(
                "No alignment loaded yet. Call load() before create_timeline()."
            )

        for key in self._keys:
            if uid == self._uid_by_key[key]:
                return self._make_timeline(key)

        available = [self._uid_by_key[key] for key in self._keys]
        raise KeyError(
            f"No timeline with uid '{uid}'. Available: "
            + ", ".join(f"'{uid}'" for uid in available)
        )

    def _make_timeline(self, key: str) -> ContinuousPhysicalTimeline:
        """Build one recording's seconds timeline: no events, its stored length."""
        return ContinuousPhysicalTimeline(
            length=self._durations[key],
            unit=TimeUnit.seconds,
            number_type=NumberType.float,
            uid=self._uid_by_key[key],
            name=self._human_name(self._stems[key]),
        )

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
        per-source ``AlignmentStore``.  One seconds timeline per recording,
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
