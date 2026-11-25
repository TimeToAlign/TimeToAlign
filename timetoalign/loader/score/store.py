"""ScoreStore: Container for category-specific EventData classes."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from timetoalign.core.time import number_from_json, number_to_json
from timetoalign.loader.score.stores.annotations import AnnotationEventData
from timetoalign.loader.score.stores.controls import ControlEventData
from timetoalign.loader.score.stores.measures import MeasureData
from timetoalign.loader.score.stores.notes import NoteEventData
from timetoalign.storage.events import EventData
from timetoalign.storage.store import EventStore

if TYPE_CHECKING:
    from timetoalign.maps import ConversionMap
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)

# Store names in canonical order
STORE_NAMES: tuple[str, ...] = ("notes", "measures", "controls", "annotations")

_FacetT = TypeVar(
    "_FacetT", NoteEventData, MeasureData, ControlEventData, AnnotationEventData
)


def _read_facet(path: Path, klass: type[_FacetT]) -> _FacetT:
    """Read one facet from *path*, or return it empty when the file is absent."""
    if path.exists():
        return klass.from_parquet(path)
    return klass.empty()


@dataclass
class ScoreStore(EventStore):
    """Container for score data organized by category.

    A ScoreLoader returns a ScoreStore containing separate EventData
    for each category (notes, measures, controls, annotations).

    Attributes:
        notes: NoteEventData with note/rest/chord events.
        measures: MeasureData with measure boundaries.
        controls: ControlEventData with dynamics, tempo, etc.
        annotations: AnnotationEventData with text annotations.
        metadata: Source metadata (format, parser, source, has_rests,
            anacrusis_offset).
    """

    notes: NoteEventData
    measures: MeasureData
    controls: ControlEventData
    annotations: AnnotationEventData
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def anacrusis_offset(self) -> Fraction | float:
        """Quarter-beat shift applied by the loader to normalise anacrusis.

        Equal to ``-min(raw_partitura_onset)`` (or the music21 equivalent)
        across all notes in the source file, as exact as the loader read the
        onsets: a ``Fraction``, or a ``float`` where the source states a
        float.  Zero when the score has no anacrusis (i.e. the first note
        starts at or after beat 0).

        This value is stored in :attr:`metadata` under the key
        ``"anacrusis_offset"`` by both ``PartituraLoader`` and
        ``Music21Loader``.  ``MatchfileLoader`` reads it to convert raw
        partitura coordinates from ``.match`` files to the TTA coordinate
        space before comparing them against stored event coordinates.

        The shift is the offset of the ``ShiftMap`` that ``MatchfileLoader``
        should attach to any score timeline it builds: the forward direction
        is ``raw → TTA`` (add offset); the ``InverseMap`` converts back
        (subtract offset).
        """
        return self.metadata.get("anacrusis_offset", Fraction(0))

    @classmethod
    def empty(cls) -> ScoreStore:
        """Create an empty ScoreStore with empty data."""

        return cls(
            notes=NoteEventData.empty(),
            measures=MeasureData.empty(),
            controls=ControlEventData.empty(),
            annotations=AnnotationEventData.empty(),
            metadata={},
        )

    def extend(self, other: ScoreStore) -> None:
        """Extend this store with another store's data.

        Args:
            other: The ScoreStore to add.
        """
        self.notes.extend(other.notes)
        self.measures.extend(other.measures)
        self.controls.extend(other.controls)
        self.annotations.extend(other.annotations)
        # Metadata aggregation strategy: keep last or list?
        # Loader handles source metadata separately.
        # We can merge non-source metadata if needed.
        self.metadata.update(other.metadata)

    # region Serialization

    def to_parquet(self, directory: Path | str) -> None:
        """Save all facets to a directory of Parquet files.

        Creates one ``<facet>.parquet`` file for each non-empty facet
        (notes, measures, controls, annotations) and a ``metadata.json``
        containing store-level metadata such as ``anacrusis_offset``.

        Args:
            directory: Directory to write into.  Created if it does not
                exist.

        Examples:
            >>> store.to_parquet("/tmp/my_score")
            >>> # creates notes.parquet, measures.parquet, …, metadata.json
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        for name, data in self.items():
            if len(data) > 0:
                data.to_parquet(directory / f"{name}.parquet")

        # Persist store-level metadata (e.g. anacrusis_offset)
        meta_path = directory / "metadata.json"
        # Serialise only JSON-safe values from self.metadata; the anacrusis
        # offset in its JSON number form, so an exact offset stays exact.
        serialisable = {}
        for k, v in self.metadata.items():
            if k == "anacrusis_offset":
                v = number_to_json(v)
            try:
                json.dumps(v)
                serialisable[k] = v
            except (TypeError, ValueError):
                module_logger.debug("Skipping non-serialisable metadata key %r", k)
        meta_path.write_text(json.dumps(serialisable, indent=2))

    @classmethod
    def from_parquet(cls, directory: Path | str) -> ScoreStore:
        """Load a ScoreStore from a directory of Parquet files.

        Expects the layout produced by :meth:`to_parquet`: one
        ``<facet>.parquet`` per facet and an optional ``metadata.json``.

        Args:
            directory: Directory containing the Parquet files.

        Returns:
            A reconstructed ScoreStore.

        Raises:
            FileNotFoundError: If *directory* does not exist.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Expected a directory for ScoreStore deserialization: {directory}"
            )

        notes = _read_facet(directory / "notes.parquet", NoteEventData)
        measures = _read_facet(directory / "measures.parquet", MeasureData)
        controls = _read_facet(directory / "controls.parquet", ControlEventData)
        annotations = _read_facet(
            directory / "annotations.parquet", AnnotationEventData
        )

        # Load metadata
        meta_path = directory / "metadata.json"
        metadata: dict[str, Any] = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text())
        if "anacrusis_offset" in metadata:
            metadata["anacrusis_offset"] = number_from_json(
                metadata["anacrusis_offset"]
            )

        return cls(
            notes=notes,
            measures=measures,
            controls=controls,
            annotations=annotations,
            metadata=metadata,
        )

    # endregion

    def summary(self) -> dict[str, Any]:
        """Extend the store summary with score-level facts."""
        result = super().summary()
        result["facts"].update(
            {
                "has_rests": (
                    self.notes.has_rests if hasattr(self.notes, "has_rests") else None
                ),
                **self.metadata,
            }
        )
        return result

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
        ppq: int | None = None,
        attach_cmaps: bool = True,
    ) -> "Timeline":
        """Create a Timeline from this ScoreStore.

        Overrides base class to automatically attach the conversion maps
        (C-Maps) of :meth:`get_cmaps` and the anacrusis ShiftMap when
        applicable.

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline.
            ppq: Pulses per quarter note. Given, a quarters → ticks C-Map at
                this resolution is attached; ``None`` (default) attaches none,
                because a score states no tick resolution.
            attach_cmaps: If True (default), attach the C-Maps of
                :meth:`get_cmaps` (seconds from tempo markings, floating
                measures from the measure table, ticks when ``ppq`` is
                given), plus a ShiftMap for the anacrusis offset.

        Returns:
            A Timeline with attached C-Maps.
        """
        from timetoalign.timelines.factory import create_timeline_from_bundle

        timeline = create_timeline_from_bundle(
            self,
            uid=uid,
            store_filters=store_filters,
            include_stores=include_stores,
            exclude_stores=exclude_stores,
            flatten=flatten,
        )

        if attach_cmaps:
            cmaps = self.get_cmaps(ppq=ppq)
            for cmap in cmaps.values():
                timeline.add_conversion_map(cmap)

            # Attach ShiftMap for anacrusis if present
            offset = self.anacrusis_offset
            if offset != 0:
                from timetoalign.maps import ShiftMap

                shift_map = ShiftMap(
                    offset=-offset,  # TTA coord - offset = raw coord
                    source_unit="quarters",
                    target_unit="quarters",
                    uid="raw_quarters",
                    name="raw_quarters",
                )
                timeline.add_conversion_map(shift_map)

        return timeline

    def get_cmaps(self, ppq: int | None = None) -> dict[str, ConversionMap]:
        """Get the ConversionMaps the score's data states.

        Returns C-Maps based on what the source states:
        - "seconds": quarters -> seconds (if tempo markings available)
        - "measures": quarters -> floating measures (if measure data available)
        - "ticks": quarters -> ticks, only when ``ppq`` is passed

        A score states no tick resolution: a DCML measures table, a
        MusicXML, MEI or kern file and a measure map count in quarters, and
        MusicXML's ``divisions`` may change inside one file. No ticks map is
        therefore derived from the score; a caller who wants ticks names the
        resolution.

        Args:
            ppq: Pulses per quarter note. Given, the result holds a
                quarters -> ticks ``ScalarMap`` at this resolution; ``None``
                (default) adds none.

        Returns:
            Dict mapping target unit name to ConversionMap.

        Examples:
            >>> bundle = score_loader.load("score.musicxml")
            >>> cmaps = bundle.get_cmaps()
            >>> "ticks" in cmaps
            False
            >>> ticks_map = bundle.get_cmaps(ppq=960)["ticks"]
            >>> ticks_map(2)  # two quarters at 960 PPQ
            1920

            >>> # Get measure position
            >>> measure_map = cmaps["measures"]
            >>> measure_map(4.0)  # e.g., 2.0 (start of measure 2)
        """
        from timetoalign.maps import ScalarMap
        from timetoalign.maps.interval import QuartersToFloatingMeasures
        from timetoalign.timelines.structure import MeasureMap

        cmaps: dict[str, ConversionMap] = {}

        # Ticks only at a resolution the caller names: a score states none.
        if ppq is not None:
            cmaps["ticks"] = ScalarMap(
                scalar=ppq,
                source_unit="quarters",
                target_unit="ticks",
            )

        # Check for tempo markings to build quarters -> seconds map
        tempo_events = self._extract_tempo_markings()
        if tempo_events:
            cmaps["seconds"] = self._create_tempo_map(tempo_events)

        # Floating-measure conversion is derived from the structure described
        # by the rows, never from printed labels.
        if len(self.measures) > 0:
            try:
                measure_map = MeasureMap.from_measure_data(self.measures)
                cmaps["measures"] = QuartersToFloatingMeasures.from_measure_map(
                    measure_map
                )
            except ValueError:
                # A structure the fm lattice cannot express — split bars
                # share one notated downbeat and cannot anchor two
                # ordinals. Leave the conversion absent (documented above):
                # an fm reading these measures do not support is worse than
                # no fm reading.
                pass

        return cmaps

    def _extract_tempo_markings(self) -> list[tuple[float, float]]:
        """Extract tempo markings from controls store.

        Returns:
            List of (quarter_position, bpm) tuples, sorted by position.
            Empty list if no tempo markings available.
        """
        # Check metadata first for pre-extracted tempo events
        if "tempo_events" in self.metadata:
            return self.metadata["tempo_events"]

        # Extract from controls store - look for Tempo subtypes
        tempo_events: list[tuple[float, float]] = []

        for event in self.controls:
            subtype = event.get("subtype")
            if subtype == "Tempo":
                # Get position (start coordinate) and BPM value
                start = event.get("start")
                bpm = event.get("value")
                if start is not None and bpm is not None:
                    # Handle Fraction or dict representation of start
                    if isinstance(start, dict):
                        start = start["value"]
                    tempo_events.append((float(start), float(bpm)))

        # Sort by position
        tempo_events.sort(key=lambda x: x[0])

        return tempo_events

    def _create_tempo_map(
        self, tempo_events: list[tuple[float, float]]
    ) -> ConversionMap:
        """Create a quarters -> seconds ConversionMap from tempo markings.

        Args:
            tempo_events: List of (quarter_position, bpm) tuples.

        Returns:
            A TableMap for quarters -> seconds conversion.
        """
        from timetoalign.maps import TableMap

        if not tempo_events:
            raise ValueError("No tempo events provided")

        # Build cumulative time map
        quarter_positions: list[float] = []
        second_positions: list[float] = []

        current_seconds = 0.0

        for i, (quarter, bpm) in enumerate(tempo_events):
            quarter_positions.append(quarter)
            second_positions.append(current_seconds)

            # Calculate time to next tempo change
            if i < len(tempo_events) - 1:
                next_quarter = tempo_events[i + 1][0]
                duration_quarters = next_quarter - quarter
                seconds_per_quarter = 60.0 / bpm
                current_seconds += duration_quarters * seconds_per_quarter

        # Add a final point to allow extrapolation
        # Extend 4 quarters beyond the last tempo change
        final_quarter = quarter_positions[-1] + 4.0
        seconds_per_quarter = 60.0 / tempo_events[-1][1]
        final_seconds = current_seconds + 4.0 * seconds_per_quarter

        quarter_positions.append(final_quarter)
        second_positions.append(final_seconds)

        return TableMap(
            x_values=quarter_positions,
            y_values=second_positions,
            source_unit="quarters",
            target_unit="seconds",
        )

    def __repr__(self) -> str:
        return (
            f"ScoreStore(notes={len(self.notes)}, measures={len(self.measures)}, "
            f"controls={len(self.controls)}, annotations={len(self.annotations)})"
        )

    # region Iterator Protocol

    def __iter__(self) -> Iterator[EventData]:
        """Iterate over data.

        Yields:
            EventData in canonical order: notes, measures, controls, annotations.

        Examples:
            >>> for data in store:
            ...     timeline = Timeline.from_event_data(data)
        """
        yield self.notes
        yield self.measures
        yield self.controls
        yield self.annotations

    def __len__(self) -> int:
        """Return the number of stores (always 4)."""
        return len(STORE_NAMES)

    def __getitem__(self, name: str) -> EventData:
        """Get data by name.

        Args:
            name: Data name (notes, measures, controls, annotations).

        Returns:
            The EventData for that category.

        Raises:
            KeyError: If name is not a valid data name.

        Examples:
            >>> notes_data = store["notes"]
        """
        if name == "notes":
            return self.notes
        elif name == "measures":
            return self.measures
        elif name == "controls":
            return self.controls
        elif name == "annotations":
            return self.annotations
        else:
            raise KeyError(f"Unknown store name: {name!r}. Valid: {STORE_NAMES}")

    def __contains__(self, name: object) -> bool:
        """Check if a data name is valid.

        Args:
            name: Data name to check.

        Returns:
            True if name is a valid data name.
        """
        return name in STORE_NAMES

    def keys(self) -> tuple[str, ...]:
        """Return data names.

        Returns:
            Tuple of data names in canonical order.
        """
        return STORE_NAMES

    def values(self) -> Iterator[EventData]:
        """Iterate over data.

        Yields:
            EventData in canonical order.

        Examples:
            >>> for data in store.values():
            ...     print(len(data))
        """
        yield self.notes
        yield self.measures
        yield self.controls
        yield self.annotations

    def items(self) -> Iterator[tuple[str, EventData]]:
        """Iterate over (name, data) pairs.

        Yields:
            Tuples of (name, EventData) in canonical order.

        Examples:
            >>> for name, data in store.items():
            ...     timeline = Timeline.from_event_data(data, uid=name)
        """
        yield ("notes", self.notes)
        yield ("measures", self.measures)
        yield ("controls", self.controls)
        yield ("annotations", self.annotations)

    # endregion
