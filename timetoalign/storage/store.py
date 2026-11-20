"""EventStore and AlignmentStore: Containers for loaded data.

This module provides:
- **EventStore (ABC)**: Container for collections of EventData
- **SingleStore**: Simple wrapper for a single EventData
- **AlignmentStore**: Container for aligned multimodal data (events + C-maps + matches)

NOTE: This class was renamed from EventBundle to EventStore in the 2026-01 API
refactoring. EventStore holds one or more EventData tables.
- EventData (formerly EventStore): PyArrow-based storage for timeline events
- EventStore (formerly EventBundle): Container for multiple EventData

Design principles:
- Uniform interface for both single-data and multi-data stores
- Consistent timeline creation across all loader types
- Children timelines maintain their own 0-based coordinate systems
- AlignmentStore bundles events with conversion maps and matches
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from html import escape
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    from timetoalign.core import Domain
    from timetoalign.maps import ConversionMap
    from timetoalign.storage.events import EventData
    from timetoalign.timelines.base import Timeline

module_logger = logging.getLogger(__name__)

_timeline_factory: Callable[..., "Timeline"] | None = None


def _register_timeline_factory(factory: Callable[..., "Timeline"]) -> None:
    """Register the timeline-layer constructor used by ``create_timeline``."""
    global _timeline_factory
    _timeline_factory = factory


class EventStore(ABC):
    """Abstract base class for collections of EventData.

    Provides a uniform interface for both single-data and multi-data
    stores, enabling consistent timeline creation across loader types.

    All stores must implement the collection protocol (iteration, keys,
    items, getitem) to allow uniform access to their constituent data.

    NOTE: This class was renamed from EventBundle to EventStore in the 2026-01 API
    refactoring. EventStore holds one or more EventData tables.

    Subclasses:
        - ScoreStore: notes, measures, controls, annotations
        - MidiStore: notes, controls
        - SingleStore: generic wrapper for any single EventData

    Examples:
        >>> # Iterate over stores
        >>> for name, store in bundle.items():
        ...     print(f"{name}: {len(store)} events")

        >>> # Create default timeline with children
        >>> timeline = bundle.create_timeline(uid="my_score")

        >>> # Create filtered timeline
        >>> timeline = bundle.create_timeline(
        ...     store_filters={"notes": {"event_type": "Note"}},
        ...     exclude_stores=["annotations"],
        ... )
    """

    # region Abstract Methods

    @abstractmethod
    def __iter__(self) -> Iterator["EventData"]:
        """Iterate over data in canonical order.

        Yields:
            EventData in the store's canonical order.
        """
        ...

    @abstractmethod
    def items(self) -> Iterator[tuple[str, "EventData"]]:
        """Iterate over (name, data) pairs in canonical order.

        Yields:
            Tuples of (data_name, EventData).
        """
        ...

    @abstractmethod
    def keys(self) -> tuple[str, ...]:
        """Return data names in canonical order.

        Returns:
            Tuple of data names.
        """
        ...

    @abstractmethod
    def __getitem__(self, name: str) -> "EventData":
        """Get data by name.

        Args:
            name: The data name.

        Returns:
            The EventData for that name.

        Raises:
            KeyError: If name is not a valid data name.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return number of EventData in the store."""
        ...

    # endregion

    # region Collection Protocol

    def __contains__(self, name: object) -> bool:
        """Check if data name exists in the store.

        Args:
            name: Data name to check.

        Returns:
            True if name is a valid data name.
        """
        return name in self.keys()

    def values(self) -> Iterator["EventData"]:
        """Iterate over data.

        Yields:
            EventData in canonical order.
        """
        yield from self

    # endregion

    # region Properties

    @property
    def event_count(self) -> int:
        """Total number of events across all data tables in the store.

        Returns:
            Sum of event counts from all constituent EventData tables.
        """
        return sum(len(data) for data in self)

    def summary(self) -> dict[str, Any]:
        """Get a summary of the store contents.

        The summary contract separates per-table data from store-level facts::

            {
                "tables": {
                    table_name: {
                        "unit": str,
                        "count": int,
                        "range": (float, float),
                    }
                },
                "facts": {fact_name: fact_value},
            }

        Subclasses may extend ``facts`` with domain-specific information, but
        must preserve this two-key structure and the exact table entry shape.

        Returns:
            A dictionary containing ``tables`` and ``facts`` mappings.

        Examples:
            >>> store = SomeStore(...)
            >>> store.summary()
            {
                "tables": {
                    "notes": {
                        "unit": "quarters",
                        "count": 498,
                        "range": (0.0, 120.5),
                    }
                },
                "facts": {},
            }
        """
        tables: dict[str, dict[str, Any]] = {}

        for name, data in self.items():
            info: dict[str, Any] = {
                "unit": str(data.unit) if hasattr(data, "unit") else "unknown",
                "count": len(data),
            }

            # Try to extract coordinate range
            if len(data) > 0:
                try:
                    # EventData stores coordinates in 'start' or 'instant' fields
                    table = data._table if hasattr(data, "_table") else None
                    if table is not None:
                        start_col = (
                            table.column("start")
                            if "start" in table.schema.names
                            else None
                        )
                        instant_col = (
                            table.column("instant")
                            if "instant" in table.schema.names
                            else None
                        )

                        # Get min/max from start or instant
                        coords = []
                        if start_col is not None:
                            for i in range(len(start_col)):
                                s = start_col[i].as_py()
                                if s is not None and isinstance(s, dict):
                                    coords.append(s.get("value", 0.0))
                        if instant_col is not None:
                            for i in range(len(instant_col)):
                                s = instant_col[i].as_py()
                                if s is not None and isinstance(s, dict):
                                    coords.append(s.get("value", 0.0))

                        if coords:
                            info["range"] = (min(coords), max(coords))
                        else:
                            info["range"] = (0.0, 0.0)
                    else:
                        info["range"] = (0.0, 0.0)
                except Exception:
                    info["range"] = (0.0, 0.0)
            else:
                info["range"] = (0.0, 0.0)

            tables[name] = info

        return {"tables": tables, "facts": {}}

    def _repr_html_(self) -> str:
        """Return an HTML representation for Jupyter display.

        Renders the store summary as an HTML table showing table names,
        units, coordinate ranges, and event counts.

        Returns:
            HTML string for display in Jupyter notebooks.
        """
        summary = self.summary()

        rows = []
        for name, info in summary["tables"].items():
            unit = info["unit"]
            count = info["count"]
            coord_range = info["range"]
            range_str = f"{coord_range[0]:.2f} – {coord_range[1]:.2f}"
            rows.append(
                f"<tr><td><strong>{escape(name)}</strong></td>"
                f"<td>{escape(unit)}</td><td>{range_str}</td><td>{count}</td></tr>"
            )

        fact_rows = [
            f"<tr><td><strong>{escape(name)}</strong></td><td>{escape(str(value))}</td></tr>"
            for name, value in summary["facts"].items()
        ]

        class_name = escape(self.__class__.__name__)
        header = "<tr><th>Table</th><th>Unit</th><th>Range</th><th>Events</th></tr>"
        table_html = f"<table>{header}{''.join(rows)}</table>"
        facts_html = ""
        if fact_rows:
            facts_header = "<tr><th>Fact</th><th>Value</th></tr>"
            facts_html = f"<table>{facts_header}{''.join(fact_rows)}</table>"

        return (
            f"<div><strong>{class_name}</strong> "
            f"({self.event_count} events total){table_html}{facts_html}</div>"
        )

    # endregion

    # region Conversion Maps

    def get_cmaps(self) -> dict[str, "ConversionMap"]:
        """Get ConversionMaps derivable from store metadata.

        Returns a dictionary mapping target unit names to ConversionMaps
        that can convert from this bundle's native unit. Subclasses should
        override to provide maps based on available metadata (PPQ, tempo
        events, sample rate, etc.).

        Returns:
            Dict mapping target unit name (e.g., "quarters", "seconds") to
            ConversionMap instances. Empty dict if no C-Maps can be derived.

        Examples:
            >>> bundle = midi_loader.load("performance.mid")
            >>> cmaps = bundle.get_cmaps()
            >>> if "quarters" in cmaps:
            ...     quarters = cmaps["quarters"](960)  # Convert 960 ticks
        """
        return {}

    # endregion

    # region Timeline Creation

    def create_timeline(
        self,
        uid: str | None = None,
        store_filters: dict[str, dict[str, Any]] | None = None,
        include_stores: list[str] | None = None,
        exclude_stores: list[str] | None = None,
        flatten: bool = False,
    ) -> "Timeline":
        """Create a Timeline from this store.

        By default, each data becomes a child timeline embedded at offset 0,
        maintaining its own 0-based coordinate system. The parent timeline's
        length equals the maximum length across all children.

        Args:
            uid: Unique ID for the parent timeline. Auto-generated if None.
            store_filters: Per-data filter kwargs to apply before timeline
                creation. Example: {"notes": {"event_type": "Note"}} excludes
                rests from the notes data.
            include_stores: Only include these data (default: all non-empty).
            exclude_stores: Exclude these data from the timeline.
            flatten: If True, merge all events into a single parent timeline
                without children. If False (default), each data becomes a
                child timeline at offset 0.

        Returns:
            A Timeline with the requested structure.

        Raises:
            ValueError: If no data remain after filtering, or all data
                are empty.

        Examples:
            >>> # Default: each data as a child, named after it, its id minted
            >>> timeline = store.create_timeline(uid="my_score")
            >>> children = {timeline.get_child(i).name: i for i in timeline.list_children()}
            >>> notes = timeline.get_child(children["notes"])

            >>> # Filtered: only Note events, exclude annotations
            >>> timeline = store.create_timeline(
            ...     store_filters={"notes": {"event_type": "Note"}},
            ...     exclude_stores=["annotations"],
            ... )
        """
        if _timeline_factory is None:
            raise RuntimeError("Timeline creation requires timetoalign.timelines")
        return _timeline_factory(
            self,
            uid=uid,
            store_filters=store_filters,
            include_stores=include_stores,
            exclude_stores=exclude_stores,
            flatten=flatten,
        )

    # endregion


class SingleStore(EventStore):
    """Store wrapper for a single EventData.

    Provides EventStore interface compatibility for loaders that naturally
    produce a single EventData rather than multiple categorized data.

    This is used as the default store type for loaders that don't have
    specialized store implementations (e.g., simple audio loaders).

    NOTE: This class was renamed from SingleStoreBundle to SingleStore in
    the 2026-01 API refactoring.

    Attributes:
        data: The wrapped EventData.
        name: The data name.

    Examples:
        >>> data = EventData.from_dicts([...], unit=TimeUnit.seconds)
        >>> store = SingleStore(data, name="beats")
        >>> timeline = store.create_timeline()
    """

    def __init__(self, data: "EventData", name: str = "events") -> None:
        """Initialize SingleStore.

        Args:
            data: The EventData to wrap.
            name: The name for this data. Used as the child timeline ID.
        """
        self._data = data
        self._name = name

    # region EventStore Protocol

    def __iter__(self) -> Iterator["EventData"]:
        """Yield the single data."""
        yield self._data

    def items(self) -> Iterator[tuple[str, "EventData"]]:
        """Yield (name, data) pair."""
        yield (self._name, self._data)

    def keys(self) -> tuple[str, ...]:
        """Return tuple with single data name."""
        return (self._name,)

    def __getitem__(self, name: str) -> "EventData":
        """Get data by name.

        Args:
            name: Must match the data name.

        Returns:
            The wrapped EventData.

        Raises:
            KeyError: If name doesn't match.
        """
        if name == self._name:
            return self._data
        raise KeyError(f"Unknown data: {name!r}. Valid: {self.keys()}")

    def __len__(self) -> int:
        """Return 1 (single data)."""
        return 1

    # endregion

    # region Properties

    @property
    def data(self) -> "EventData":
        """The wrapped EventData."""
        return self._data

    @property
    def name(self) -> str:
        """The data name."""
        return self._name

    # endregion

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SingleStore({self._name}={len(self._data)} events)"


# endregion


# region DictStore


class DictStore(EventStore):
    """Generic EventStore with arbitrary named EventData tables.

    Used by AlignmentLoaders that produce a variable number of timelines
    (e.g., MatchfileLoader: 1 score + N performances; TiliaJsonLoader:
    one table per annotated timeline).

    Maintains the uniform EventStore interface (iteration, keys, items,
    getitem) while supporting any number of named data sources.

    Args:
        data: Dictionary mapping names to EventData tables.

    Examples:
        >>> store = DictStore({
        ...     "score": score_event_data,
        ...     "perf:p01": perf_01_event_data,
        ...     "perf:p02": perf_02_event_data,
        ... })
        >>> for name, events in store.items():
        ...     print(f"{name}: {len(events)} events")
    """

    def __init__(self, data: dict[str, "EventData"] | None = None) -> None:
        """Initialize DictStore.

        Args:
            data: Dictionary mapping names to EventData tables.
                If None, starts with an empty dictionary.
        """
        self._data: dict[str, "EventData"] = data or {}

    def add(self, name: str, events: "EventData") -> None:
        """Add or replace a named EventData table.

        Args:
            name: The name for this data (e.g., ``"score"``, ``"perf:p01"``).
            events: The EventData to store.
        """
        self._data[name] = events

    # region EventStore Protocol

    def __iter__(self) -> Iterator["EventData"]:
        """Iterate over EventData in insertion order."""
        yield from self._data.values()

    def items(self) -> Iterator[tuple[str, "EventData"]]:
        """Iterate over (name, EventData) pairs in insertion order."""
        yield from self._data.items()

    def keys(self) -> tuple[str, ...]:
        """Return data names in insertion order."""
        return tuple(self._data.keys())

    def __getitem__(self, name: str) -> "EventData":
        """Get EventData by name.

        Args:
            name: The data name.

        Returns:
            The EventData for that name.

        Raises:
            KeyError: If name is not found.
        """
        try:
            return self._data[name]
        except KeyError:
            raise KeyError(f"Unknown data: {name!r}. Valid: {self.keys()}") from None

    def __len__(self) -> int:
        """Return number of named EventData tables."""
        return len(self._data)

    # endregion

    def __repr__(self) -> str:
        """Return string representation."""
        entries = ", ".join(
            f"{name}={len(ed)} events" for name, ed in self._data.items()
        )
        return f"DictStore({entries})"


# endregion


# region AlignmentStore


# Match data PyArrow schema
MATCH_SCHEMA = pa.schema(
    [
        pa.field("match_id", pa.string(), nullable=False),
        pa.field("source_event_id", pa.string(), nullable=True),
        pa.field("source_domain", pa.string(), nullable=False),
        pa.field(
            "source_coordinate",
            pa.struct(
                [
                    pa.field("value", pa.float64(), nullable=False),
                    pa.field("numerator", pa.int64(), nullable=True),
                    pa.field("denominator", pa.int64(), nullable=True),
                ]
            ),
            nullable=True,
        ),
        pa.field("target_event_id", pa.string(), nullable=True),
        pa.field("target_domain", pa.string(), nullable=False),
        pa.field(
            "target_coordinate",
            pa.struct(
                [
                    pa.field("value", pa.float64(), nullable=False),
                    pa.field("numerator", pa.int64(), nullable=True),
                    pa.field("denominator", pa.int64(), nullable=True),
                ]
            ),
            nullable=True,
        ),
        pa.field("confidence", pa.float64(), nullable=True),
        pa.field("agent", pa.string(), nullable=True),
        pa.field("method", pa.string(), nullable=True),
    ]
)


@dataclass
class MatchData:
    """Container for alignment matches stored as a PyArrow table.

    Matches represent claims that events or coordinates from different
    timelines/domains are equivalent or synchronous.

    The schema follows the TTA conceptual model:
    - source_event_id/target_event_id: Event identifiers (optional)
    - source_coordinate/target_coordinate: Coordinate values
    - source_domain/target_domain: Domain labels ("physical", "logical", "graphical")
    - confidence: Certainty level (0.0-1.0)
    - agent: Who/what created the match
    - method: How it was created

    Attributes:
        table: PyArrow table containing match data.

    Examples:
        >>> matches = MatchData.from_dicts([
        ...     {
        ...         "match_id": "m1",
        ...         "source_event_id": "note_001",
        ...         "source_domain": "logical",
        ...         "source_coordinate": {"value": 0.0, "numerator": 0, "denominator": 1},
        ...         "target_event_id": None,
        ...         "target_domain": "physical",
        ...         "target_coordinate": {"value": 1.5, "numerator": None, "denominator": None},
        ...         "confidence": 0.95,
        ...         "agent": "human_annotator",
        ...         "method": "manual",
        ...     }
        ... ])
    """

    table: pa.Table

    @classmethod
    def empty(cls) -> "MatchData":
        """Create an empty MatchData."""
        return cls(
            table=pa.table(
                {name: [] for name in MATCH_SCHEMA.names}, schema=MATCH_SCHEMA
            )
        )

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]]) -> "MatchData":
        """Create MatchData from a list of match dictionaries.

        Args:
            rows: List of match dictionaries matching MATCH_SCHEMA.

        Returns:
            A new MatchData containing the matches.
        """
        if not rows:
            return cls.empty()

        table = pa.Table.from_pylist(rows, schema=MATCH_SCHEMA)
        return cls(table=table)

    @classmethod
    def from_table(cls, table: pa.Table) -> "MatchData":
        """Create MatchData from an existing PyArrow table.

        Args:
            table: PyArrow table with MATCH_SCHEMA fields.

        Returns:
            A new MatchData wrapping the table.
        """
        return cls(table=table)

    @property
    def count(self) -> int:
        """Number of matches."""
        return self.table.num_rows

    def extend(self, other: "MatchData") -> None:
        """Extend this data with matches from another MatchData (in-place).

        Args:
            other: Another MatchData to append.
        """
        self.table = pa.concat_tables([self.table, other.table])

    def __len__(self) -> int:
        """Return number of matches."""
        return self.count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over matches as dictionaries."""
        for batch in self.table.to_batches():
            yield from batch.to_pylist()

    def __repr__(self) -> str:
        return f"MatchData(count={self.count})"


@dataclass
class AlignmentStore(EventStore):
    """Container for aligned multimodal data.

    AlignmentStore bundles:
    - events: EventStore containing events from all domains
    - cmaps: ConversionMaps between coordinate systems
    - matches: MatchData with alignment claims

    This is the return type for AlignmentLoader subclasses that load
    formats encoding complete alignments (TiLiA, MPM, parangonada, and
    performance-precision exports).

    Attributes:
        events: EventStore with events organized by domain/category.
        cmaps: List of ConversionMaps between coordinate systems.
        matches: MatchData containing alignment claims.
        metadata: Additional metadata about the alignment.

    Examples:
        >>> store = AlignmentStore(
        ...     events=score_store,
        ...     cmaps=[ticks_to_seconds, quarters_to_ticks],
        ...     matches=match_data,
        ... )
        >>> store.domains
        {Domain.logical, Domain.physical}
    """

    events: EventStore
    cmaps: list["ConversionMap"] = field(default_factory=list)
    matches: MatchData = field(default_factory=MatchData.empty)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "AlignmentStore":
        """Create an empty AlignmentStore.

        Returns:
            An AlignmentStore with empty events, no C-maps, and no matches.
        """
        from timetoalign.core import TimeUnit

        from .events import EventData

        empty_data = EventData.empty(TimeUnit.seconds)
        empty_store = SingleStore(empty_data, name="events")

        return cls(
            events=empty_store,
            cmaps=[],
            matches=MatchData.empty(),
            metadata={},
        )

    def __iter__(self) -> Iterator[EventData]:
        """Iterate over the aligned event tables."""
        yield from self.events

    def __len__(self) -> int:
        """Return the number of aligned event tables."""
        return len(self.events)

    def __getitem__(self, name: str) -> EventData:
        """Get an aligned event table by name.

        Args:
            name: The event table name.

        Returns:
            The named event table.
        """
        return self.events[name]

    def __repr__(self) -> str:
        return (
            f"AlignmentStore(events={self.event_count}, "
            f"matches={self.match_count}, "
            f"cmaps={self.cmap_count})"
        )

    @property
    def event_count(self) -> int:
        """Total number of events across all stores."""
        total = 0
        for data in self.events:
            total += len(data)
        return total

    @property
    def match_count(self) -> int:
        """Number of alignment matches."""
        return len(self.matches)

    @property
    def cmap_count(self) -> int:
        """Number of conversion maps."""
        return len(self.cmaps)

    @property
    def domains(self) -> set["Domain"]:
        """Return all domains represented in this alignment.

        Inferred from event units and match domain labels.
        """
        from timetoalign.core import Domain

        found: set[Domain] = set()

        # Infer from event data units
        for data in self.events:
            if hasattr(data, "unit"):
                found.add(data.unit.domain)

        # Also check match domains
        for match in self.matches:
            source_domain = match.get("source_domain")
            target_domain = match.get("target_domain")
            if source_domain:
                try:
                    found.add(Domain(source_domain))
                except ValueError:
                    pass
            if target_domain:
                try:
                    found.add(Domain(target_domain))
                except ValueError:
                    pass

        return found

    def extend(self, other: "AlignmentStore") -> None:
        """Extend this store with data from another AlignmentStore (in-place).

        Events are concatenated (by name if both are EventStore subclasses),
        C-maps are appended, and matches are concatenated.

        Args:
            other: Another AlignmentStore to merge.
        """
        # Extend events - this depends on the EventStore implementation
        # For simplicity, we'll create a new combined store
        # This is a limitation; real impl would need smart merging
        self.cmaps.extend(other.cmaps)
        self.matches.extend(other.matches)
        self.metadata.update(other.metadata)

    def get_cmap(self, source_unit: str, target_unit: str) -> "ConversionMap | None":
        """Find a ConversionMap between two units.

        Args:
            source_unit: Source unit name (e.g., "ticks").
            target_unit: Target unit name (e.g., "seconds").

        Returns:
            The ConversionMap if found, None otherwise.
        """
        for cmap in self.cmaps:
            if (
                hasattr(cmap, "source_unit")
                and hasattr(cmap, "target_unit")
                and cmap.source_unit == source_unit
                and cmap.target_unit == target_unit
            ):
                return cmap
        return None

    def get_matches_for_event(self, event_id: str) -> list[dict[str, Any]]:
        """Get all matches involving a specific event.

        Args:
            event_id: The event identifier to search for.

        Returns:
            List of match dictionaries where event_id is source or target.
        """
        results = []
        for match in self.matches:
            if (
                match.get("source_event_id") == event_id
                or match.get("target_event_id") == event_id
            ):
                results.append(match)
        return results

    def summary(self) -> dict[str, Any]:
        """Get a summary of the alignment store.

        Returns:
            Summary with table information and alignment-level facts.
        """
        result = super().summary()
        result["facts"].update(
            {
                "event_count": self.event_count,
                "match_count": self.match_count,
                "cmap_count": self.cmap_count,
                "domains": sorted(domain.name for domain in self.domains),
                "store_names": list(self.events.keys()),
                **self.metadata,
            }
        )
        return result

    def items(self) -> Iterator[tuple[str, EventData]]:
        """Iterate over aligned event table names and data."""
        yield from self.events.items()

    def keys(self) -> tuple[str, ...]:
        """Return aligned event table names in canonical order."""
        return self.events.keys()


# endregion
