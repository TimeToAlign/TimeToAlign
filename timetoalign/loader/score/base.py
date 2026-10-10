"""ScoreLoader: Base loader for symbolic scores."""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from timetoalign.core import TimeUnit
from timetoalign.loader.base import EventLoader

from .store import ScoreStore
from .stores.notes import NoteEventData

if TYPE_CHECKING:
    from timetoalign.timelines import MeasureMap, Timeline
    from timetoalign.timelines.flow import ScoreFlowController, SegmentNameGenerator

module_logger = logging.getLogger(__name__)


class ScoreLoader(EventLoader):
    """Base loader for symbolic music scores (MusicXML, MIDI-Score, TSV).

    All score loaders normalise coordinates to quarter-note positions using
    ``Fraction`` precision.  The reported metadata reflects this:
    ``unit='quarters'``, ``number_type='fraction'``.
    """

    _default_unit = TimeUnit.quarters
    _event_data_class = NoteEventData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._store = ScoreStore.empty()

    @property
    def store(self) -> ScoreStore:
        """The ScoreStore containing all loaded score elements."""
        return self._store

    @property
    def events(self) -> NoteEventData:
        """The primary event data (notes), for compatibility."""
        return self._store.notes

    def get_events(
        self, properties: bool | str | tuple[str, ...] = True
    ) -> NoteEventData:
        """Return note events with field control.

        Overrides the base ``Loader.get_events`` because ``ScoreLoader``
        stores events in a ``ScoreStore``, not in ``self._events``.

        Args:
            properties: Controls which non-field-spec fields to include.
                ``True``: all fields.  ``False``: spec + core fields only.
                Single string: shorthand for a one-element tuple.
                Tuple of strings: named property fields only.
        """
        if isinstance(properties, str):
            properties = (properties,)
        self._events = self._store.notes
        return self._assemble_events_table(properties)

    @abstractmethod
    def _load_source(self, source: Path) -> ScoreStore:
        """Load a single source file into a ScoreStore.

        Args:
            source: Path to the source file.

        Returns:
            A ScoreStore containing the loaded data.
        """
        ...

    def load(self, *sources: Path | str) -> Self:
        """Load sources into the store.

        Args:
            *sources: Paths to source files.

        Returns:
            Self, for method chaining.
        """
        for source in sources:
            path = Path(source)

            # Load store from source
            loaded_store = self._load_source(path)

            # Update metadata
            meta = loaded_store.metadata.copy()
            meta["path"] = str(path)
            meta["loaded_at"] = datetime.now(timezone.utc).isoformat()

            self._sources.append(path)
            self._source_metadata.append(meta)

            # Extend internal store
            self._store.extend(loaded_store)

        return self

    def clear(self) -> None:
        """Clear all loaded sources and store data."""
        super().clear()
        self._store = ScoreStore.empty()

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> Timeline:
        """Create a score timeline with measure and flow-control structure."""
        timeline = super().create_timeline(uid=uid, **kwargs)
        self._populate_timeline_structure(timeline)
        return timeline

    def _measure_map(self) -> MeasureMap:
        """Return the measure map the loaded measure table describes."""
        from timetoalign.timelines import MeasureMap

        return MeasureMap.from_measure_data(self._store.measures)

    def _populate_timeline_structure(self, timeline: Timeline) -> None:
        """Attach structure derivable from a non-empty measure store.

        The measure store must describe one measure map in printed order. A
        store whose measures cannot form that map is a defect of the source,
        not a lossy reading of it: an unfolded measures table repeats printed
        measure counts and therefore measure ids. Such a store raises instead
        of producing a timeline without its measure structure.

        Raises:
            ValueError: If the stored measures cannot form a ``MeasureMap``.
        """
        if len(self._store.measures) == 0:
            return

        timeline.add_measure_map(self._measure_map())

        registry = self.create_flow_controller().get_flow_control_registry()
        for brk in registry.breaks:
            timeline.flow_control.add_break(brk)
        for jump in registry.jumps:
            timeline.flow_control.add_jump(jump)
        for name, coordinate in registry.markers.items():
            timeline.flow_control.add_marker(name, coordinate)

    # region Flow Control

    def create_flow_controller(
        self,
        *,
        name_generator: "SegmentNameGenerator | None" = None,
    ) -> "ScoreFlowController":
        """Create a `timetoalign.ScoreFlowController` from the loaded measure data.

        The flow controller derives the repeat structure (repeats, voltas,
        jumps, D.S., D.C.) from the measure data and computes the
        traversal order.  Requires measures to have been loaded; raises
        ``ValueError`` if no measure data is available.

        Args:
            name_generator: Strategy for labelling atomic sections, forwarded
                to the ``ScoreFlowController``.  ``None`` uses the default
                naming (standard alphabet with the volta-suffix rule).

        Returns:
            A configured ``ScoreFlowController`` ready to compute flows.

        Raises:
            ValueError: If no measure data has been loaded.

        Examples:
            >>> loader = Ms3Loader.from_file("notes.tsv", "measures.tsv")
            >>> controller = loader.create_flow_controller()
            >>> flow = controller.compute_flow(FlowMode.default)
        """
        from timetoalign.timelines.flow import ScoreFlowController

        if self._store.measures is None or len(self._store.measures) == 0:
            raise ValueError(
                "Cannot create a ScoreFlowController: no measure data has been "
                "loaded.  Load a measures file first (e.g. *.measures.tsv)."
            )
        return ScoreFlowController(self._store.measures, name_generator=name_generator)

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the loaded score data to Parquet files.

        For ``ScoreLoader`` subclasses, data lives in a multi-facet
        ``ScoreStore`` (notes, measures, controls, annotations).  Each
        facet is written as a separate ``.parquet`` file inside a directory
        at *path*, together with a ``metadata.json`` for store-level
        metadata.

        Args:
            path: Directory to write into.  Created if it does not exist.
        """
        self._store.to_parquet(path)

    @classmethod
    def from_parquet(cls, path: Path | str) -> Self:
        """Load a ScoreLoader from a directory of Parquet files.

        Reconstructs the internal ``ScoreStore`` from the facet files
        produced by :meth:`to_parquet`.

        Args:
            path: Directory containing the facet Parquet files.

        Returns:
            A new ScoreLoader with events loaded from the files.
        """
        store = ScoreStore.from_parquet(path)
        loader = cls.__new__(cls)
        # Set attributes that __init__ would normally create
        loader._unit = cls._default_unit
        loader._number_type = loader._unit.default_number_type
        loader._sources = []
        loader._source_metadata = []
        loader._events = cls._event_data_class.empty(loader._unit, loader._number_type)
        loader._store = store
        return loader

    # endregion
