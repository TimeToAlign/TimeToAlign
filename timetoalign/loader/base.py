"""Loader: Base classes for loading music representations into TimeToAlign!

This module provides the loader hierarchy for TimeToAlign!:

- **Loader (ABC)**: Returns EventStore (events only)
- **ManifestLoader (ABC)**: Returns ManifestData (dimensions/metadata only)
- **AlignmentLoader (ABC)**: Returns AlignmentStore (events + C-maps + Matches)

Design principles:
- Multi-source: One Loader can aggregate multiple files
- Metadata in table: Deterministic, stored in PyArrow schema metadata
- Polymorphic returns: Different loader categories return different types
- Subclassable: Domain-specific loaders extend these ABCs
"""

from __future__ import annotations

import html
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import numpy as np
import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import (
    IntervalPolicy,
    NumberType,
    TimeUnit,
)
from timetoalign.core.fields import TIMETOALIGN_METADATA_KEY, field_metadata
from timetoalign.core.time import struct_to_coordinate
from timetoalign.display.html import affordance_html, code
from timetoalign.storage import EventData

# Event loaders return source metadata and a vectorized field dictionary. Other
# payload families return their payload directly and use ``_accept_source``.
LoadSourceResult = tuple[
    dict[str, Any], dict[str, np.ndarray | pa.Array | pa.ChunkedArray]
]

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.storage.store import AlignmentStore, EventStore
    from timetoalign.timelines.base import Timeline
    from timetoalign.timelines.groups import TimelineGroup

module_logger = logging.getLogger(__name__)

# Type variable for polymorphic loader payloads.
PayloadT = TypeVar("PayloadT")


class Loader(ABC, Generic[PayloadT]):
    """Shared lifecycle for all loader families.

    The root owns source tracking, the ``load`` template, convenience
    construction, and timeline-creation surfaces. Payload-specific state and
    APIs belong to subclasses such as :class:`EventLoader`,
    :class:`ManifestLoader`, and :class:`AlignmentLoader`.
    """

    def __init__(self) -> None:
        """Initialize shared loader state."""
        self._sources: list[Path] = []
        self._source_metadata: list[dict[str, Any]] = []
        self._minted_timeline_ids: set[str] = set()

    @abstractmethod
    def _load_source(self, source: Path) -> PayloadT:
        """Parse one source and return the family-specific payload."""
        ...

    def _unpack_source_result(
        self, result: PayloadT
    ) -> tuple[dict[str, Any], PayloadT]:
        """Return metadata and payload from a source parser result."""
        return {}, result

    def _accept_source(
        self, path: Path, source_meta: dict[str, Any], payload: PayloadT
    ) -> None:
        """Record a parsed payload.

        Generic payload loaders override this hook to retain their parsed
        object. The default records source lifecycle information only.
        """
        self._sources.append(path)
        self._source_metadata.append(source_meta)

    def load(self, *sources: Path | str) -> Self:
        """Load and accept one or more sources through the shared template."""
        for source in sources:
            path = Path(source)
            source_meta, payload = self._unpack_source_result(self._load_source(path))
            source_meta["path"] = str(path)
            source_meta["loaded_at"] = datetime.now(timezone.utc).isoformat()
            self._accept_source(path, source_meta, payload)
        return self

    @classmethod
    def from_file(cls, *paths: Path | str, **kwargs: Any) -> Self:
        """Instantiate the loader, load *paths*, and return it."""
        loader = cls(**kwargs)
        loader.load(*paths)
        return loader

    @property
    def sources(self) -> list[Path]:
        """List of loaded source paths."""
        return list(self._sources)

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated source metadata."""
        return {
            "loader_class": self.__class__.__name__,
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    def clear(self) -> None:
        """Clear shared source state."""
        self._sources.clear()
        self._source_metadata.clear()

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Create a timeline from the loaded payload.

        Concrete loader families implement payload-specific construction.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline()."
        )

    def create_timelines(self, id_pattern: str | None = None) -> "list[Timeline]":
        """Create all timelines, optionally filtered by an implementation."""
        return [self.create_timeline()]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a timeline group when supported."""
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an alignment bundle when supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    def __repr__(self) -> str:
        """Return a concise shared-lifecycle representation."""
        return f"{self.__class__.__name__}(sources={len(self._sources)})"


class EventLoader(Loader[LoadSourceResult]):
    """Abstract base class for event-producing loaders.

    Loader provides a unified interface for loading one or more source files
    and aggregating their events into an EventData. Metadata about the sources
    is stored in the PyArrow table's schema metadata for determinism.

    Subclasses must implement:
    - _load_source(): Parse a single source file into event rows
    - _default_unit: The default time unit for this loader type

    Attributes:
        events: The EventData containing all loaded events.
        sources: List of loaded source file paths.
        unit: The time unit for coordinates.
        number_type: The number type for coordinates.

    Examples:
        >>> # Subclass implementation
        >>> class MidiLoader(Loader):
        ...     _default_unit = TimeUnit.ticks
        ...     _event_data_class = EventData
        ...
        ...     def _load_source(self, path):
        ...         # Parse MIDI file, return (metadata_dict, event_rows)
        ...         return {"format": "midi"}, [{"id": "n1", ...}]
        >>>
        >>> loader = MidiLoader()
        >>> loader.load("piece.mid")
        >>> print(loader.event_summary())
    """

    # Class-level configuration (subclasses override)
    _default_unit: ClassVar[TimeUnit] = TimeUnit.seconds
    _event_data_class: ClassVar[type[EventData]] = EventData

    # Canonical ordering for RawField fields in assembled events tables.
    # Fields matching these names appear first (in this order), followed by
    # non-canonical RawFields (in field_specs order), then property fields.
    _CANONICAL_FIELD_ORDER: ClassVar[list[str]] = ["start", "end", "duration"]

    def __init__(
        self,
        unit: TimeUnit | None = None,
        number_type: NumberType | None = None,
        interval_policy: IntervalPolicy | str = IntervalPolicy.warn,
    ) -> None:
        """Initialize the Loader.

        Args:
            unit: The time unit for coordinates. Defaults to class's _default_unit.
            number_type: The number type for coordinates. Defaults to the
                one the unit itself uses.
            interval_policy: How to handle end/duration inconsistencies in
                interval events.  Accepts an `IntervalPolicy` enum member or
                a string (``"warn"``, ``"prefer_end"``, ``"prefer_duration"``,
                ``"strict"``).  Default: ``"warn"`` -- prefer ``end``, warn
                on discrepancies.
        """
        super().__init__()
        self._unit = unit or self._default_unit
        self._number_type = self._unit.resolve_number_type(number_type)
        self._interval_policy = IntervalPolicy(interval_policy)
        self._events: EventData = self._event_data_class.empty(
            self._unit, self._number_type
        )

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Path) -> LoadSourceResult:
        """Load a single source file.

        Subclasses implement this to parse their specific format.

        Event loaders return ``(metadata_dict, field_dict)`` where
        ``field_dict`` contains numpy or PyArrow arrays for each field. Other
        loader families may return their payload directly.

        Args:
            source: Path to the source file.

        Returns:
            A source result. Event loaders return a tuple of file metadata and
            vectorized field arrays; payload loaders return their payload.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    # region Properties

    @property
    def events(self) -> EventData:
        """The EventData containing all loaded events."""
        return self._events

    @property
    def sources(self) -> list[Path]:
        """List of loaded source file paths."""
        return list(self._sources)

    @property
    def unit(self) -> TimeUnit:
        """The time unit for coordinates."""
        return self._unit

    @property
    def number_type(self) -> NumberType:
        """The number type for coordinates."""
        return self._number_type

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated metadata from all sources.

        Returns:
            Dict with loader info and per-source metadata.
        """
        return {
            "loader_class": self.__class__.__name__,
            "unit": str(self._unit),
            "number_type": str(self._number_type),
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    @property
    def store(self) -> "EventStore":
        """Return an EventStore wrapping the loader's events.

        Subclasses may override to return specialized stores
        (e.g., ScoreStore, MidiStore). Default implementation
        wraps self.events in a SingleStore.

        Returns:
            An EventStore providing uniform access to loaded data.
        """
        from timetoalign.storage.store import SingleStore

        return SingleStore(self._events, name="events")

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
    ) -> "Timeline":
        """Create a Timeline from the loaded events.

        Convenience method that delegates to self.store.create_timeline().

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation. Example: {"notes": {"event_type": "Note"}}.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline.

        Returns:
            A Timeline containing the loaded events.

        Examples:
            >>> loader = Ms3Loader()
            >>> loader.load("notes.tsv")
            >>> timeline = loader.create_timeline(uid="my_score")
        """
        from timetoalign.timelines.factory import _flag_minted, _mint_timeline_id

        minted = uid is None
        if minted:
            uid = _mint_timeline_id(self, self._unit)

        return _flag_minted(
            self.store.create_timeline(
                uid=uid,
                store_filters=store_filters,
                include_stores=include_stores,
                exclude_stores=exclude_stores,
                flatten=flatten,
            ),
            minted,
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        The default implementation returns a single-element list with
        ``create_timeline()``. Subclasses with multi-timeline output
        (e.g., ``TiliaJsonLoader``, ``MatchfileLoader``) override this.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.
        """
        return [self.create_timeline()]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a `TimelineGroup`. Override in subclasses that support groups.

        Returns:
            A `TimelineGroup`, or ``None`` if the loader does not produce groups.
        """
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an `AlignmentBundle`. Override in subclasses that support bundles.

        Raises:
            NotImplementedError: If the concrete loader does not produce bundles.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    # endregion

    # region Item Access

    def __getitem__(self, key: str) -> "EventData":
        """Access a named EventData table from the store.

        Delegates to ``self.store[key]``.

        Args:
            key: The name of the EventData table.

        Returns:
            The `EventData` for the given key.

        Raises:
            KeyError: If the key is not found in the store.
        """
        return self.store[key]

    # endregion

    # region Events Assembly

    def get_events(self, properties: bool | str | tuple[str, ...] = True) -> EventData:
        """Return an EventData assembled from loaded data.

        This is the primary access method for the loader-first pipeline.
        It assembles EventData from the loaded data, with control over
        which property fields are included.

        Args:
            properties: Controls which non-field-spec data fields to include.
                - ``True``: Include all property fields (default).
                - ``False``: Only include semantic-field fields (start, end,
                  duration, pitch, etc.) and core fields (id, name,
                  event_type, temporal_type).
                - Single string: shorthand for a one-element tuple — include
                  only the named property field.
                - Tuple of strings: Include only the named property fields.

        Returns:
            An ``EventData`` containing the assembled events.
        """
        if isinstance(properties, str):
            properties = (properties,)
        return self._assemble_events_table(properties)

    def _assemble_events_table(
        self, properties: bool | str | tuple[str, ...] = True
    ) -> EventData:
        """Assemble an EventData from internal state, with field ordering.

        Field ordering follows the canonical field order:
        1. Core fields (id, name, event_type, temporal_type)
        2. Canonical fields (start, end, duration, pitch -- in order, if present)
        3. Non-canonical fields (in field_specs order)
        4. Property fields (controlled by the *properties* parameter)

        The default implementation returns ``self._events`` unmodified
        when ``properties=True``.  Subclasses with ``field_specs`` can
        override to produce filtered/reordered tables.

        Args:
            properties: Field filter (see ``get_events``).

        Returns:
            An ``EventData`` with the requested fields.
        """
        if isinstance(properties, str):
            properties = (properties,)
        if properties is True:
            return self._events

        # Determine which fields to keep
        events = self._events
        if len(events) == 0:
            return events

        table = events._table
        all_names = set(table.column_names)

        # Core fields always included
        core_names = {"id", "name", "event_type", "temporal_type"}

        # Canonical-spec fields always included (when present)
        spec_names = set()
        for name in self._CANONICAL_FIELD_ORDER:
            if name in all_names:
                spec_names.add(name)

        # Also include any fields with timetoalign metadata (semantic fields)
        for i in range(len(table.schema)):
            pa_field = table.schema.field(i)
            if pa_field.metadata and TIMETOALIGN_METADATA_KEY in pa_field.metadata:
                spec_names.add(pa_field.name)

        keep = core_names | spec_names

        if properties is False:
            # Only fields + core
            pass
        elif isinstance(properties, tuple):
            # Add requested property fields
            for name in properties:
                if name in all_names:
                    keep.add(name)

        # Build ordered field list preserving table order
        ordered = [c for c in table.column_names if c in keep]
        if set(ordered) == all_names:
            return events

        filtered_table = table.select(ordered)
        return EventData(filtered_table, events.unit, events.number_type)

    # endregion

    # region Loading

    def _unpack_source_result(
        self, result: LoadSourceResult
    ) -> tuple[dict[str, Any], dict[str, np.ndarray | pa.Array | pa.ChunkedArray]]:
        """Split an event parser result into metadata and field arrays."""
        return result

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: dict[str, np.ndarray | pa.Array | pa.ChunkedArray],
    ) -> None:
        """Accept one loaded source using the event payload implementation."""
        if not isinstance(payload, dict):
            raise TypeError(
                f"{self.__class__.__name__} returned an unsupported payload "
                f"type: {type(payload).__name__}"
            )

        self._sources.append(path)
        self._source_metadata.append(source_meta)
        if not payload:
            return

        extra_fields = getattr(self, "_extra_schema_fields", None)
        new_data = self._event_data_class.from_arrays(
            payload,
            self._unit,
            self._number_type,
            validate=getattr(self, "_validate_vectorized", True),
            extra_fields=extra_fields,
            interval_policy=self._interval_policy,
        )
        if len(self._events) == 0:
            self._events = new_data
        else:
            self._events.extend(new_data)

    def clear(self) -> None:
        """Clear all loaded sources and events."""
        super().clear()
        self._events = self._event_data_class.empty(self._unit, self._number_type)

    # endregion

    # region Stats (delegated to EventData)

    def event_summary(self) -> dict[str, Any]:
        """Get a summary of loaded events.

        Returns:
            Dict with event counts, types, coordinate range, etc.
        """
        summary = self._events.summary()
        summary["sources"] = [str(s) for s in self._sources]
        return summary

    def count_events_by_type(self) -> dict[str, int]:
        """Count events grouped by event_type.

        Returns:
            Dict mapping event type names to counts.
        """
        return self._events.count_by("event_type")

    def count_events_by_temporal_type(self) -> dict[str, int]:
        """Count events grouped by temporal_type (instant/interval).

        Returns:
            Dict mapping "instant"/"interval" to counts.
        """
        return self._events.count_by("temporal_type")

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the total number of events across all store tables."""
        return self.store.event_count

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"sources={len(self._sources)}, "
            f"events={self.store.event_count}, "
            f"unit={self._unit})"
        )

    # endregion

    # region C-Map Creation

    def create_cmap(
        self,
        source_field: str,
        target_field: str,
        *,
        map_type: type | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a ConversionMap from two coordinate fields.

        This method creates a C-Map from loaded coordinate data, enabling
        conversion between different coordinate systems (e.g., seconds to pixels).

        Both fields must contain coordinate data (either core coordinates like
        'start'/'end', or CoordinateField extra fields).

        Args:
            source_field: Name of the source coordinate field.
            target_field: Name of the target coordinate field.
            map_type: The map class to create. Defaults to TableMap.
                Supported: TableMap, LinearMap, ScalarMap.
            **kwargs: Additional arguments passed to the map constructor.
                For TableMap: kind, extrapolate
                For LinearMap: (computed automatically from data)

        Returns:
            A ConversionMap instance.

        Raises:
            ValueError: If fields don't exist or aren't coordinate fields.
            ValueError: If insufficient data points for the map type.

        Examples:
            >>> # Load data with dual coordinates
            >>> loader.load("data.tsv")

            >>> # Create TableMap (default) from start -> x_pixels
            >>> cmap = loader.create_cmap("start", "x_pixels")

            >>> # Create LinearMap (fits y = ax + b to data)
            >>> cmap = loader.create_cmap("start", "x_pixels", map_type=LinearMap)

            >>> # TableMap with custom interpolation
            >>> cmap = loader.create_cmap("start", "x_pixels", kind="cubic")
        """
        from timetoalign.maps import LinearMap, ScalarMap, TableMap

        # Default to TableMap
        if map_type is None:
            map_type = TableMap

        # Extract coordinate values from fields
        source_values = self._extract_coordinate_values(source_field)
        target_values = self._extract_coordinate_values(target_field)

        if len(source_values) != len(target_values):
            raise ValueError(
                f"Field length mismatch: {source_field} has {len(source_values)} "
                f"values, {target_field} has {len(target_values)}"
            )

        if len(source_values) < 2:
            raise ValueError(
                f"Need at least 2 data points to create a C-Map, got {len(source_values)}"
            )

        # Get units from field metadata
        source_unit = self._get_field_unit(source_field)
        target_unit = self._get_field_unit(target_field)

        # Create the appropriate map type
        if map_type is TableMap:
            return TableMap(
                x_values=source_values,
                y_values=target_values,
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )
        elif map_type is LinearMap:
            # Fit linear regression: y = ax + b
            x = np.array(source_values, dtype=np.float64)
            y = np.array(target_values, dtype=np.float64)
            # Use numpy's polyfit for simple linear regression
            coeffs = np.polyfit(x, y, 1)
            scalar, offset = coeffs[0], coeffs[1]
            return LinearMap(
                scalar=float(scalar),
                offset=float(offset),
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )
        elif map_type is ScalarMap:
            # Fit pure scaling: y = ax (no offset)
            x = np.array(source_values, dtype=np.float64)
            y = np.array(target_values, dtype=np.float64)
            # Least squares fit through origin: scalar = sum(x*y) / sum(x*x)
            scalar = np.sum(x * y) / np.sum(x * x)
            return ScalarMap(
                scalar=float(scalar),
                source_unit=source_unit,
                target_unit=target_unit,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unsupported map_type: {map_type}. "
                f"Supported: TableMap, LinearMap, ScalarMap"
            )

    def _extract_coordinate_values(
        self, field_name: str
    ) -> list[int | float | Fraction]:
        """Extract a coordinate field's values in their own representation.

        A C-Map built from an exact column deserves exact anchors: a tick
        grid read off a score should convert back to the ratios it came
        from, not to whatever the float projection rounds to. So this reads
        each cell in the representation the field declares rather than
        reaching straight for the float side.

        Args:
            field_name: The field name to extract.

        Returns:
            List of coordinate values.

        Raises:
            ValueError: If field doesn't exist or isn't a coordinate field.
        """
        table = self._events._table

        if field_name not in table.column_names:
            raise ValueError(
                f"Field '{field_name}' not found. " f"Available: {table.column_names}"
            )

        arr = table.column(field_name)

        # Check if it's a coordinate struct (has 'value' field)
        if pa.types.is_struct(arr.type):
            inner_names = [f.name for f in arr.type]
            if "value" in inner_names:
                number_type = self._get_field_number_type(field_name)
                decoded: list[int | float | Fraction] = []
                for cell in arr.to_pylist():
                    if cell is None:
                        decoded.append(0.0)
                        continue
                    try:
                        decoded.append(struct_to_coordinate(cell, number_type))
                    except (TypeError, ValueError, ZeroDivisionError):
                        decoded.append(float(cell.get("value") or 0.0))
                return decoded
            else:
                raise ValueError(
                    f"Field '{field_name}' is a struct but doesn't have a 'value' "
                    f"field. Inner fields: {inner_names}"
                )
        elif pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type):
            # Plain numeric field
            return [float(v) for v in arr.to_pylist() if v is not None]
        else:
            raise ValueError(
                f"Field '{field_name}' is not a coordinate field. " f"Type: {arr.type}"
            )

    def _get_field_unit(self, field_name: str) -> str | None:
        """Get the unit metadata from a coordinate field.

        Args:
            field_name: The field name.

        Returns:
            The unit string, or None if not specified.
        """
        table = self._events._table
        schema = table.schema

        field_idx = schema.get_field_index(field_name)
        if field_idx < 0:
            return None

        unit = field_metadata(schema.field(field_idx)).get("unit")
        if unit:
            return unit

        # For core coordinate fields (start, end, duration), use loader's unit
        if field_name in ("start", "end", "duration"):
            return str(self._unit.value) if self._unit else None

        return None

    def _get_field_number_type(self, field_name: str) -> NumberType:
        """Return the representation a coordinate field declares.

        Falls back to the loader's own, which is what an unannotated core
        coordinate column is written in.
        """
        schema = self._events._table.schema
        field_idx = schema.get_field_index(field_name)
        if field_idx >= 0:
            declared = field_metadata(schema.field(field_idx)).get("number_type")
            if declared:
                return NumberType(declared)
        return self._number_type

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Built from the overridable :meth:`_repr_rows` /
        :meth:`_repr_count_row` / :meth:`_repr_affordances` hooks so the
        whole loader hierarchy shares one rendering path (no hand-rolled
        ``<table>`` markup per subclass).
        """
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_count_row(self) -> tuple[str, str]:
        """Return the payload-count row shown in :meth:`_repr_html_`.

        Default is ``("Events", str(len(self)))`` — ``len(self)`` (NOT
        ``self.store.event_count``) so whole-directory / single-file
        loaders that keep their payload in claims or timelines report the
        meaningful count via their ``__len__`` override.  Loaders whose
        payload is better named otherwise (e.g. alignment loaders counting
        claims) override this hook rather than appending a second row.
        """
        return ("Events", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Default rows: Sources (count + first up-to-5 names), the
        :meth:`_repr_count_row` payload-count row, and Unit.  Subclasses
        extend by calling ``super()._repr_rows()`` and adding their own.
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        return [
            ("Sources", sources_value),
            self._repr_count_row(),
            ("Unit", html.escape(str(self._unit.value))),
        ]

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row.

        Detects the create methods this loader overrides plus the
        universal ``get_events(...)`` accessor and, when the class defines
        a ``diagram`` method, ``diagram()``.
        """
        snippets = ["create_timeline()"]
        if self.create_group.__func__ is not EventLoader.create_group:
            snippets.append("create_group()")
        if self.create_bundle.__func__ is not EventLoader.create_bundle:
            snippets.append("create_bundle()")
        snippets.append("get_events(...)")
        if hasattr(type(self), "diagram"):
            snippets.append("diagram()")
        return snippets

    # endregion

    # region Serialization

    def to_parquet(self, path: Path | str) -> None:
        """Save the loaded events to a Parquet file.

        The metadata (including source info) is preserved in the file.

        Args:
            path: Path to write the Parquet file.
        """
        self.events.to_parquet(path)

    @classmethod
    def from_parquet(cls, path: Path | str) -> Self:
        """Load a Loader from a Parquet file.

        Note: This creates a new Loader with the EventData loaded,
        but source paths may not be accessible for re-loading.

        Args:
            path: Path to the Parquet file.

        Returns:
            A new Loader with events loaded from the file.
        """
        events = cls._event_data_class.from_parquet(path)
        loader = cls.__new__(cls)
        loader._unit = events.unit
        loader._number_type = events.number_type
        loader._events = events
        loader._sources = []
        loader._source_metadata = []
        return loader

    # endregion


# region ManifestData


@dataclass
class ManifestData:
    """Container for manifest/metadata from ManifestLoader.

    ManifestData holds structural information about a source without events.
    This includes dimensions, sample rates, page counts, etc.

    Attributes:
        dimensions: Dict of named dimensions (e.g., {"width": 1920, "height": 1080}).
        metadata: Additional metadata about the source.
        source_path: Path to the source file.
        source_type: Type of source (e.g., "audio", "image", "pdf").

    Examples:
        >>> # Audio manifest
        >>> manifest = ManifestData(
        ...     dimensions={"duration_samples": 44100 * 60, "duration_seconds": 60.0},
        ...     metadata={"sample_rate": 44100, "channels": 2},
        ...     source_type="audio",
        ... )

        >>> # PDF manifest
        >>> manifest = ManifestData(
        ...     dimensions={"pages": 10, "width": 612, "height": 792},
        ...     metadata={"title": "Score", "dpi": 72},
        ...     source_type="pdf",
        ... )
    """

    dimensions: dict[str, int | float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None
    source_type: str = "unknown"

    def __repr__(self) -> str:
        dims_str = ", ".join(f"{k}={v}" for k, v in self.dimensions.items())
        return f"ManifestData({self.source_type}, {dims_str})"


# endregion


# region ManifestLoader


class ManifestLoader(Loader[ManifestData]):
    """Abstract base class for loaders that return ManifestData.

    ManifestLoader is for sources that provide dimensional/structural
    information without timeline events:
    - Audio files -> duration, sample rate, channels
    - PDF files -> pages, dimensions
    - Images -> width, height, DPI

    Subclasses must implement:
    - _load_source(): Parse a single source into ManifestData

    The loaded ManifestData can be used to create empty timelines with
    correct dimensions, or to configure subsequent event loading.

    Examples:
        >>> class AudioManifestLoader(ManifestLoader):
        ...     def _load_source(self, source: Path) -> ManifestData:
        ...         # Read audio header
        ...         return ManifestData(
        ...             dimensions={"duration_samples": n_samples},
        ...             metadata={"sample_rate": sr},
        ...             source_type="audio",
        ...         )
    """

    def __init__(self) -> None:
        """Initialize the ManifestLoader."""
        super().__init__()
        self._manifests: list[ManifestData] = []
        self._logger = module_logger.getChild(self.__class__.__name__)

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Path) -> ManifestData:
        """Load a single source file into ManifestData.

        Subclasses implement this to parse their specific format.

        Args:
            source: Path to the source file.

        Returns:
            ManifestData containing dimensions and metadata.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: ManifestData,
    ) -> None:
        """Record one manifest payload from the root lifecycle."""
        payload.source_path = path
        self._sources.append(path)
        self._source_metadata.append(source_meta)
        self._manifests.append(payload)
        self._logger.debug(f"Loaded manifest from {path}")

    # region Properties

    @property
    def sources(self) -> list[Path]:
        """List of loaded source file paths."""
        return list(self._sources)

    @property
    def manifests(self) -> list[ManifestData]:
        """List of loaded ManifestData objects."""
        return list(self._manifests)

    @property
    def manifest(self) -> ManifestData | None:
        """The first (or only) manifest, for convenience."""
        return self._manifests[0] if self._manifests else None

    # endregion

    def clear(self) -> None:
        """Clear all loaded sources and manifests."""
        super().clear()
        self._manifests.clear()

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        **kwargs: Any,
    ) -> "Timeline":
        """Create a Timeline from the loaded manifest.

        Subclasses override this to create the appropriate timeline type
        (e.g., ``DiscretePhysicalTimeline`` for audio, ``DiscreteGraphicalTimeline``
        for images).

        Args:
            uid: Unique identifier for the timeline. Auto-generated if None.
            **kwargs: Additional arguments for subclass implementations.

        Returns:
            A Timeline representing the loaded source.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline(). "
            "Subclasses should override this method."
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        Default implementation returns a single-element list.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.
        """
        return [self.create_timeline()]

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks."""
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Sources, the manifest count, then one row per dimension of the
        first manifest (the manifest's structural shape).
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        rows: list[tuple[str, str]] = [
            ("Sources", sources_value),
            ("Manifests", str(len(self))),
        ]
        if self._manifests:
            for k, v in self._manifests[0].dimensions.items():
                rows.append((str(k), html.escape(str(v))))
        return rows

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        return ["create_timeline()", "create_timelines()"]

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return the number of loaded manifests."""
        return len(self._manifests)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(sources={len(self._sources)})"

    # endregion


# endregion


# region AlignmentLoader


class AlignmentLoader(Loader["AlignmentStore"]):
    """Abstract base class for loaders that return AlignmentStore.

    AlignmentLoader is for formats that encode aligned multimodal data:
    - TiLiA JSON -> hierarchical annotations with alignments
    - Match files -> score-to-performance alignment
    - MPM, parangonada, Listen Here!, and PerformancePrecision exports

    Subclasses must implement:
    - _load_source(): Parse a single source into AlignmentStore

    AlignmentStore contains:
    - EventData tables for each domain/layer
    - ConversionMaps between coordinate systems
    - Match objects representing alignment claims

    Examples:
        >>> loader = MatchfileLoader()
        >>> loader.load("performance.match")
        >>> bundle = loader.create_bundle()
    """

    def __init__(self) -> None:
        """Initialize the AlignmentLoader."""
        super().__init__()
        self._store: AlignmentStore | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    # region Abstract Methods

    @abstractmethod
    def _load_source(self, source: Path) -> "AlignmentStore":
        """Load a single source file into AlignmentStore.

        Subclasses implement this to parse their specific format.

        Args:
            source: Path to the source file.

        Returns:
            AlignmentStore containing events, C-maps, and matches.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the source file is invalid.
        """
        ...

    # endregion

    def _accept_source(
        self,
        path: Path,
        source_meta: dict[str, Any],
        payload: "AlignmentStore",
    ) -> None:
        """Store or merge one alignment payload from the root lifecycle."""
        self._sources.append(path)
        self._source_metadata.append(source_meta)
        if self._store is None:
            self._store = payload
        else:
            self._store.extend(payload)
        self._logger.debug(f"Loaded alignment from {path}")

    # region Properties

    @property
    def sources(self) -> list[Path]:
        """List of loaded source file paths."""
        return list(self._sources)

    @property
    def store(self) -> "AlignmentStore | None":
        """The AlignmentStore containing all loaded data."""
        return self._store

    @property
    def metadata(self) -> dict[str, Any]:
        """Aggregated metadata from all sources."""
        return {
            "loader_class": self.__class__.__name__,
            "source_count": len(self._sources),
            "sources": self._source_metadata,
        }

    # endregion

    def clear(self) -> None:
        """Clear all loaded sources and store."""
        super().clear()
        self._store = None

    # region Timeline & Bundle Creation

    def create_timeline(self, uid: str | None = None, **kwargs: Any) -> "Timeline":
        """Create a single Timeline by uid.

        Subclasses override this to assemble timelines from loaded data.

        Args:
            uid: Timeline identifier. Interpretation is subclass-specific.
            **kwargs: Additional arguments for subclass implementations.

        Returns:
            A Timeline.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timeline(). "
            "Subclasses should override this method."
        )

    def create_timelines(
        self,
        id_pattern: str | None = None,
    ) -> "list[Timeline]":
        """Create all timelines, optionally filtered by regex pattern.

        Args:
            id_pattern: Optional regex pattern to filter timeline IDs.

        Returns:
            List of `Timeline` objects.

        Raises:
            NotImplementedError: If the subclass does not provide an implementation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_timelines(). "
            "Subclasses should override this method."
        )

    @staticmethod
    def _filter_timelines_by_id_pattern(
        timelines: list["Timeline"], id_pattern: str | None
    ) -> list["Timeline"]:
        """Filter timelines by an optional regular expression on their IDs."""
        return [
            tl for tl in timelines if id_pattern is None or re.search(id_pattern, tl.id)
        ]

    def create_group(self, **kwargs: Any) -> "TimelineGroup | None":
        """Create a `TimelineGroup`. Override in subclasses that support groups.

        Returns:
            A `TimelineGroup`, or ``None`` if not supported.
        """
        return None

    def create_bundle(self, **kwargs: Any) -> "AlignmentBundle":
        """Create an `AlignmentBundle`. Override in subclasses that support bundles.

        Raises:
            NotImplementedError: If the concrete loader does not produce bundles.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_bundle()."
        )

    # endregion

    # region HTML Representation

    def _repr_html_(self) -> str:
        """Rich HTML affordance card for Jupyter notebooks.

        Built from the overridable :meth:`_repr_rows` /
        :meth:`_repr_count_row` / :meth:`_repr_affordances` hooks so every
        alignment loader shares one rendering path.
        """
        return affordance_html(
            type(self).__name__,
            self._repr_rows(),
            affordances=self._repr_affordances(),
        )

    def _repr_count_row(self) -> tuple[str, str]:
        """Return the payload-count row shown in :meth:`_repr_html_`.

        Default ``("Events", str(len(self)))``.  Whole-directory /
        single-file alignment loaders whose payload is claims override
        this to ``("Claims", …)`` rather than appending a second row, so
        the count row never doubles up.
        """
        return ("Events", str(len(self)))

    def _repr_rows(self) -> list[tuple[str, str]]:
        """Return the ``(label, value)`` rows for :meth:`_repr_html_`.

        Default rows: Sources (count + first up-to-5 names) and the
        :meth:`_repr_count_row` payload-count row.  Subclasses extend by
        calling ``super()._repr_rows()`` and adding their own.
        """
        n_sources = len(self._sources)
        names = ", ".join(code(s.name) for s in self._sources[:5])
        if n_sources > 5:
            names += f", … (+{n_sources - 5} more)"
        sources_value = f"{n_sources} file(s)" + (f": {names}" if names else "")
        return [
            ("Sources", sources_value),
            self._repr_count_row(),
        ]

    def _repr_affordances(self) -> list[str]:
        """Return the invocation snippets for the ``Try`` row."""
        snippets = ["create_timeline()", "create_timelines()"]
        if self.create_group.__func__ is not AlignmentLoader.create_group:
            snippets.append("create_group()")
        if self.create_bundle.__func__ is not AlignmentLoader.create_bundle:
            snippets.append("create_bundle()")
        if hasattr(type(self), "diagram"):
            snippets.append("diagram()")
        return snippets

    # endregion

    # region Magic Methods

    def __len__(self) -> int:
        """Return total event count across all stores."""
        if self._store is None:
            return 0
        return self._store.event_count

    def __repr__(self) -> str:
        """Return string representation."""
        n_events = len(self) if self._store else 0
        return (
            f"{self.__class__.__name__}("
            f"sources={len(self._sources)}, "
            f"events={n_events})"
        )

    # endregion


# endregion
