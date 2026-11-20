"""TiliaJsonLoader: Loader for TiLiA JSON annotation exports.

This module implements the ``TiliaJsonLoader`` class, which parses TiLiA
``.tla`` / ``.json`` exports and produces:

- One ``ContinuousPhysicalTimeline`` per annotated timeline (timelines in
  the JSON's ``"timelines"`` array), with events derived from each
  timeline's ``"components"`` array.
- A ``TimelineGroup`` containing all timelines, with timestamps at 0 and
  at the media length.
- Optionally an ``AlignmentBundle`` wrapping the group (no cross-group
  claims).

TiLiA encodes analytical annotations on audio recordings. Each timeline
in a TiLiA export carries a ``"kind"`` field (``HIERARCHY_TIMELINE``,
``MARKER_TIMELINE``, ``BEAT_TIMELINE``, ``HARMONY_TIMELINE``,
``PDF_TIMELINE``, etc.) and a ``"components"`` array whose elements have
kind-dependent schemas.

The loader follows the standard two-phase pattern:

1. ``loader.load(*sources)`` -- parse TiLiA JSON files.
2. ``loader.create_group()`` -- returns a ``TimelineGroup``.
3. ``loader.create_timeline(uid)`` -- returns a single timeline.
4. ``loader.create_bundle()`` -- convenience wrapper.

The loader's ``.store`` property returns a `TiliaDictStore` with helper
properties for each TiLiA timeline type:

    - ``loader.store.beat`` -- concatenation of all beat timeline tables
    - ``loader.store.harmony`` -- concatenation of all harmony timeline tables
    - ``loader.store.hierarchy`` -- concatenation of all hierarchy timeline tables
    - ``loader.store.marker`` -- concatenation of all marker timeline tables
    - ``loader.store.pdf`` -- concatenation of all PDF timeline tables

See Also:
    timetoalign.loader.format.json.JsonLoader
    timetoalign.timelines.groups.TimelineGroup
    timetoalign.alignment.bundle.AlignmentBundle
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pyarrow as pa

from timetoalign.core import TimeUnit, resolve_id
from timetoalign.loader.format.json import JsonLoader, _normalise_array
from timetoalign.storage.events import EventData
from timetoalign.storage.store import DictStore
from timetoalign.timelines.factory import _mint_timeline_id
from timetoalign.timelines.types import ContinuousPhysicalTimeline

if TYPE_CHECKING:
    from timetoalign.alignment.bundle import AlignmentBundle
    from timetoalign.timelines.base import Timeline
    from timetoalign.timelines.groups import TimelineGroup

module_logger = logging.getLogger(__name__)


# region Component-to-event mapping


def _hierarchy_to_events(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert HIERARCHY components to interval events.

    Hierarchy components have ``start``, ``end``, ``label``, ``level``
    and other fields.  We map them to interval events with ``start`` and
    ``end`` coordinates plus metadata.  All JSON fields are preserved as
    event properties, with ``label`` mapped to ``name``.

    Args:
        components: List of HIERARCHY component dicts.

    Returns:
        List of event dicts suitable for ``Timeline.add_events()``.
    """
    events: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        event: dict[str, Any] = {
            "id": f"h{i:04d}",
            "start": float(comp["start"]),
            "end": float(comp["end"]),
            "event_type": "Hierarchy",
        }
        # Map label → name (the canonical event name field)
        if "label" in comp:
            event["name"] = comp["label"]
        # Include ALL other fields from the JSON component
        for key, value in comp.items():
            if key not in ("start", "end", "label"):
                event[key] = value
        events.append(event)
    return events


def _marker_to_events(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert MARKER components to instant events.

    Marker components have ``time``, ``label``, and optional
    ``measure``/``beat`` fields.  All JSON fields are preserved as
    event properties, with ``label`` mapped to ``name``.

    Args:
        components: List of MARKER component dicts.

    Returns:
        List of event dicts suitable for ``Timeline.add_events()``.
    """
    events: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        event: dict[str, Any] = {
            "id": f"m{i:04d}",
            "instant": float(comp["time"]),
            "event_type": "Marker",
        }
        # Map label → name (the canonical event name field)
        if "label" in comp:
            event["name"] = comp["label"]
        # Include ALL other fields from the JSON component
        for key, value in comp.items():
            if key not in ("time", "label"):
                event[key] = value
        events.append(event)
    return events


def _beat_to_events(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert BEAT components to instant events.

    Beat components have ``time``, ``measure``, and ``beat`` fields.
    All JSON fields are preserved as event properties, with ``label``
    mapped to ``name`` if present.

    Args:
        components: List of BEAT component dicts.

    Returns:
        List of event dicts suitable for ``Timeline.add_events()``.
    """
    events: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        event: dict[str, Any] = {
            "id": f"b{i:04d}",
            "instant": float(comp["time"]),
            "event_type": "Beat",
        }
        # Map label → name (the canonical event name field) if present
        if "label" in comp:
            event["name"] = comp["label"]
        # Include ALL other fields from the JSON component
        for key, value in comp.items():
            if key not in ("time", "label"):
                event[key] = value
        events.append(event)
    return events


def _pdf_marker_to_events(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert PDF_MARKER components to instant events.

    PDF marker components have ``time``, ``page_number``, and optional
    ``measure``/``beat`` fields.  All JSON fields are preserved as
    event properties, with ``label`` mapped to ``name`` if present.

    Args:
        components: List of PDF_MARKER component dicts.

    Returns:
        List of event dicts suitable for ``Timeline.add_events()``.
    """
    events: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        event: dict[str, Any] = {
            "id": f"p{i:04d}",
            "instant": float(comp["time"]),
            "event_type": "PdfMarker",
        }
        # Map label → name (the canonical event name field) if present
        if "label" in comp:
            event["name"] = comp["label"]
        # Include ALL other fields from the JSON component
        for key, value in comp.items():
            if key not in ("time", "label"):
                event[key] = value
        events.append(event)
    return events


def _generic_to_events(
    components: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fallback converter for unknown component kinds.

    Attempts to detect instant vs interval events by looking for
    ``time`` (instant), or ``start``/``end`` (interval) keys.
    All JSON fields are preserved as event properties, with ``label``
    mapped to ``name`` if present.

    Args:
        components: List of component dicts.

    Returns:
        List of event dicts suitable for ``Timeline.add_events()``.
    """
    events: list[dict[str, Any]] = []
    for i, comp in enumerate(components):
        if "start" in comp and "end" in comp:
            event: dict[str, Any] = {
                "id": f"g{i:04d}",
                "start": float(comp["start"]),
                "end": float(comp["end"]),
                "event_type": comp.get("kind", "Unknown"),
            }
            # Map label → name (the canonical event name field) if present
            if "label" in comp:
                event["name"] = comp["label"]
            # Include ALL other fields from the JSON component
            for key, value in comp.items():
                if key not in ("start", "end", "kind", "label"):
                    event[key] = value
            events.append(event)
        elif "time" in comp:
            event = {
                "id": f"g{i:04d}",
                "instant": float(comp["time"]),
                "event_type": comp.get("kind", "Unknown"),
            }
            # Map label → name (the canonical event name field) if present
            if "label" in comp:
                event["name"] = comp["label"]
            # Include ALL other fields from the JSON component
            for key, value in comp.items():
                if key not in ("time", "kind", "label"):
                    event[key] = value
            events.append(event)
        else:
            module_logger.warning(
                "Component %d has no 'time', 'start', or 'end' key; skipping.",
                i,
            )
    return events


# Dispatcher mapping timeline kinds to converter functions
_EVENT_CONVERTERS: dict[str, Any] = {
    "HIERARCHY_TIMELINE": _hierarchy_to_events,
    "MARKER_TIMELINE": _marker_to_events,
    "BEAT_TIMELINE": _beat_to_events,
    "PDF_TIMELINE": _pdf_marker_to_events,
}

# Mapping from TiLiA kind strings to short type names for store properties
_KIND_TO_TYPE: dict[str, str] = {
    "BEAT_TIMELINE": "beat",
    "HARMONY_TIMELINE": "harmony",
    "HIERARCHY_TIMELINE": "hierarchy",
    "MARKER_TIMELINE": "marker",
    "PDF_TIMELINE": "pdf",
}

# All recognised TiLiA timeline type names
TILIA_TYPES: tuple[str, ...] = ("beat", "harmony", "hierarchy", "marker", "pdf")


# endregion


# region TiliaDictStore


class TiliaDictStore(DictStore):
    """``DictStore`` subclass with convenience properties for TiLiA timeline types.

    Each property returns the concatenation of all ``EventData`` tables
    whose TiLiA kind matches the requested type.  If no tables of that
    type exist, an empty ``EventData`` is returned.

    The type of each entry is tracked via a ``kind_map`` that maps store
    keys to short type names (``beat``, ``harmony``, ``hierarchy``,
    ``marker``, ``pdf``).

    Examples:
        >>> loader = TiliaJsonLoader()
        >>> loader.load("Bruckner5_Scherzo.json")
        >>> loader.store.hierarchy  # all hierarchy tables concatenated
        >>> loader.store.marker     # all marker tables concatenated
        >>> loader.store.beat       # all beat tables concatenated

    See Also:
        timetoalign.storage.store.DictStore
    """

    def __init__(
        self,
        data: dict[str, EventData] | None = None,
        kind_map: dict[str, str] | None = None,
    ) -> None:
        """Initialize TiliaDictStore.

        Args:
            data: Dictionary mapping names to EventData tables.
            kind_map: Dictionary mapping store keys to TiLiA type names
                (``"beat"``, ``"harmony"``, ``"hierarchy"``, ``"marker"``,
                ``"pdf"``).
        """
        super().__init__(data)
        self._kind_map: dict[str, str] = kind_map or {}

    def add(self, name: str, events: EventData, kind: str | None = None) -> None:  # type: ignore[override]
        """Add or replace a named EventData table with optional kind tracking.

        Args:
            name: The name for this data (e.g., ``"HIERARCHY_TIMELINE_0"``).
            events: The EventData to store.
            kind: The TiLiA type name (``"beat"``, ``"hierarchy"``, etc.).
                If ``None``, the kind is inferred from the name if possible.
        """
        super().add(name, events)
        if kind is not None:
            self._kind_map[name] = kind
        elif name in self._kind_map:
            pass  # already tracked
        else:
            # Try to infer from the key name (e.g., "BEAT_TIMELINE_3" -> "beat")
            for kind_str, type_name in _KIND_TO_TYPE.items():
                if name.startswith(kind_str):
                    self._kind_map[name] = type_name
                    break

    def _get_tables_by_type(self, type_name: str) -> EventData | None:
        """Get concatenation of all tables matching a TiLiA type.

        Args:
            type_name: The short type name (``"beat"``, ``"hierarchy"``, etc.).

        Returns:
            Concatenated ``EventData``, or ``None`` if no tables match.
        """
        matching: list[EventData] = []
        for key, kind in self._kind_map.items():
            if kind == type_name and key in self._data:
                matching.append(self._data[key])

        if not matching:
            return None

        if len(matching) == 1:
            return matching[0]

        # Concatenate all matching tables
        tables = [ed._table for ed in matching]
        merged = pa.concat_tables(tables, promote_options="default")
        return EventData(merged, matching[0]._unit, matching[0]._number_type)

    @property
    def beat(self) -> EventData:
        """All beat timeline tables concatenated.

        Returns:
            ``EventData`` with all beat events, or empty ``EventData``.
        """
        result = self._get_tables_by_type("beat")
        if result is None:
            return EventData(pa.table({}), TimeUnit.seconds)
        return result

    @property
    def harmony(self) -> EventData:
        """All harmony timeline tables concatenated.

        Returns:
            ``EventData`` with all harmony events, or empty ``EventData``.
        """
        result = self._get_tables_by_type("harmony")
        if result is None:
            return EventData(pa.table({}), TimeUnit.seconds)
        return result

    @property
    def hierarchy(self) -> EventData:
        """All hierarchy timeline tables concatenated.

        Returns:
            ``EventData`` with all hierarchy events, or empty ``EventData``.
        """
        result = self._get_tables_by_type("hierarchy")
        if result is None:
            return EventData(pa.table({}), TimeUnit.seconds)
        return result

    @property
    def marker(self) -> EventData:
        """All marker timeline tables concatenated.

        Returns:
            ``EventData`` with all marker events, or empty ``EventData``.
        """
        result = self._get_tables_by_type("marker")
        if result is None:
            return EventData(pa.table({}), TimeUnit.seconds)
        return result

    @property
    def pdf(self) -> EventData:
        """All PDF timeline tables concatenated.

        Returns:
            ``EventData`` with all PDF events, or empty ``EventData``.
        """
        result = self._get_tables_by_type("pdf")
        if result is None:
            return EventData(pa.table({}), TimeUnit.seconds)
        return result

    @property
    def kind_map(self) -> dict[str, str]:
        """Mapping from store keys to TiLiA type names."""
        return dict(self._kind_map)

    def __repr__(self) -> str:
        """Return string representation."""
        type_counts: dict[str, int] = {}
        for kind in self._kind_map.values():
            type_counts[kind] = type_counts.get(kind, 0) + 1
        parts = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        return f"TiliaDictStore({parts})"


# endregion


# region TiliaJsonLoader


class TiliaJsonLoader(JsonLoader):
    """Loader for TiLiA JSON annotation exports.

    Parses a TiLiA ``.tla`` or ``.json`` file and produces one
    `ContinuousPhysicalTimeline` (seconds) per annotated timeline in
    the file.  The primary output is a `timetoalign.timelines.groups.TimelineGroup`
    containing all timelines, accessible via ``create_group()``.

    Internally, the ``"timelines"`` array is parsed so that each element
    (which has its own ``"kind"`` and ``"components"``) becomes a
            separate ``pa.Table`` keyed by its source identifier. The underlying
            ``JsonLoader`` machinery is
    used for normalising each timeline's components into a flat table.

    The ``.store`` property returns a `TiliaDictStore` with helper
    properties for each TiLiA timeline type::

        loader.store.hierarchy  # all hierarchy tables concatenated
        loader.store.marker     # all marker tables concatenated
        loader.store.beat       # all beat tables concatenated

    **Two-phase usage:**

    1. ``loader.load("Bruckner5_Scherzo.json")``
    2. ``group = loader.create_group()``
        3. ``tl = loader.create_timeline("BEAT_TIMELINE_3")``
    4. ``bundle = loader.create_bundle()``

    Args:
        media_unit: The ``TimeUnit`` for all timelines.  Default
            ``TimeUnit.seconds`` (TiLiA annotations are time-based).

    Examples:
        >>> loader = TiliaJsonLoader()
        >>> loader.load("Bruckner5_Scherzo.json")
        >>> group = loader.create_group()
        >>> group.n_timelines
        7
        >>> loader.create_timeline("BEAT_TIMELINE_3").n_events
        1146
        >>> loader.store.hierarchy._table.num_rows
        33

    See Also:
        timetoalign.loader.format.json.JsonLoader
        timetoalign.timelines.groups.TimelineGroup
        TiliaDictStore
    """

    def __init__(
        self,
        *,
        media_unit: TimeUnit = TimeUnit.seconds,
        **kwargs: Any,
    ) -> None:
        # TiliaJsonLoader does not use JsonLoader's principal_keys /
        # auto-detect mode.  Instead it overrides _process_json to
        # handle the "timelines" array specially.
        super().__init__(principal_keys=None, sep=".", resolve_lookups=False, **kwargs)
        self._media_unit = media_unit
        self._timeline_specs: list[dict[str, Any]] = []
        self._timelines_cache: dict[str, ContinuousPhysicalTimeline] = {}
        self._media_length: float = 0.0
        # Replace the plain DictStore from JsonLoader with a TiliaDictStore.
        # TiliaDictStore is a subclass of DictStore, so this is compatible.
        self._store: TiliaDictStore = TiliaDictStore()  # type: ignore[assignment]
        self._logger = module_logger.getChild("TiliaJsonLoader")

    # region Properties

    @property
    def store(self) -> TiliaDictStore:
        """The ``TiliaDictStore`` containing all normalised timeline tables.

        Provides convenience properties for accessing tables by TiLiA
        timeline type (beat, harmony, hierarchy, marker, pdf).

        Returns:
            A ``TiliaDictStore`` with one entry per parsed timeline.
        """
        return self._store

    @property
    def media_length(self) -> float:
        """Media length in seconds (from ``media_metadata.media length``)."""
        return self._media_length

    @property
    def timeline_ids(self) -> list[str]:
        """Type-based identifiers for all parsed timelines.

        IDs are ``cpt_1``, ``cpt_2``, ... in source order. Source IDs and
        titles remain accepted by :meth:`create_timeline`.
        """
        return [spec["id"] for spec in self._timeline_specs]

    @property
    def timeline_specs(self) -> list[dict[str, Any]]:
        """Metadata dicts for each parsed timeline.

        Each dict contains ``id``, ``kind``, ``name``, ``n_components``,
        and ``ordinal``.
        """
        return list(self._timeline_specs)

    # endregion

    # region Loading override

    def _process_json(self, data: dict[str, Any]) -> None:
        """Override: parse TiLiA-specific structure.

        Instead of auto-detecting top-level arrays, this method
        specifically handles the ``"timelines"`` array where each
        element is a timeline object with its own ``"components"``.

        Args:
            data: The root JSON object.

        Raises:
            TypeError: If *data* is not a dict.
            KeyError: If ``"timelines"`` key is missing.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object (dict), got {type(data).__name__}")

        if "timelines" not in data:
            raise KeyError(
                "TiLiA JSON must contain a 'timelines' key. "
                f"Found keys: {list(data.keys())}"
            )

        # Extract media metadata
        media_meta = data.get("media_metadata", {})
        if isinstance(media_meta, dict):
            ml = media_meta.get("media length")
            if ml is not None:
                self._media_length = float(ml)
            self._file_metadata.update(
                {k: v for k, v in media_meta.items() if not isinstance(v, (dict, list))}
            )

        media_path = data.get("media_path")
        if media_path is not None:
            self._file_metadata["media_path"] = media_path

        # Process each timeline in the array (v1 list) or dict (v2)
        timelines_raw = data["timelines"]
        if isinstance(timelines_raw, dict):
            # v2 format: dict with ID keys
            timelines_array = [{"_tilia_id": k, **v} for k, v in timelines_raw.items()]
        elif isinstance(timelines_raw, list):
            timelines_array = timelines_raw
        else:
            raise TypeError(
                f"'timelines' must be a list or dict, got {type(timelines_raw).__name__}"
            )

        self._timeline_specs = []
        self._timelines_cache = {}
        self._minted_timeline_ids: set[str] = set()

        for idx, tl_obj in enumerate(timelines_array):
            if not isinstance(tl_obj, dict):
                self._logger.warning("timelines[%d] is not a dict; skipping.", idx)
                continue

            kind = tl_obj.get("kind", "UNKNOWN")
            name = tl_obj.get("name", "")
            ordinal = tl_obj.get("ordinal", idx)
            components_raw = tl_obj.get("components", [])

            # v2 format: components is a dict with ID keys; convert to list
            if isinstance(components_raw, dict):
                components = [{"_comp_id": k, **v} for k, v in components_raw.items()]
                # Update tl_obj so that spec["raw"] has the list format
                tl_obj["components"] = components
            else:
                components = components_raw

            # Use tilia_id if available (from v2 format conversion)
            source_id = tl_obj.get("_tilia_id", f"{kind}_{idx}")
            tl_id = _mint_timeline_id(self, TimeUnit.seconds)
            timeline_name = name or source_id

            # Normalise components into a pa.Table
            if components and isinstance(components, list):
                if isinstance(components[0], dict):
                    table = _normalise_array(components, sep=self._sep)
                else:
                    table = pa.table({"value": components})
            else:
                table = pa.table({})

            # Determine the TiLiA type for the store
            tilia_type = _KIND_TO_TYPE.get(kind)

            # Store in the TiliaDictStore
            self._store.add(source_id, self._wrap_table(table), kind=tilia_type)

            self._timeline_specs.append(
                {
                    "id": tl_id,
                    "source_id": source_id,
                    "kind": kind,
                    "name": timeline_name,
                    "n_components": len(components),
                    "ordinal": ordinal,
                    "index": idx,
                    "raw": tl_obj,
                }
            )

            self._logger.debug(
                "Timeline %d: %s (%s) with %d components",
                idx,
                source_id,
                timeline_name,
                len(components),
            )

    # endregion

    # region Domain Object Creation

    def create_timeline(self, uid: str | int) -> "Timeline":
        """Create a single ``ContinuousPhysicalTimeline`` by uid.

        The *uid* may be a type-based ID from ``timeline_ids`` (for example,
        ``"cpt_4"``), the original TiLiA source ID/title, or an integer index.

        Args:
            uid: Timeline identifier or integer index.

        Returns:
            A ``ContinuousPhysicalTimeline`` with events from the
            timeline's components.

        Raises:
            KeyError: If no timeline with *uid* exists.
            RuntimeError: If ``load()`` has not been called.
        """
        if not self._timeline_specs:
            raise RuntimeError("No data loaded. Call load() before create_timeline().")

        # Allow integer indexing
        spec = self._find_spec(uid)
        tl_id = spec["id"]

        if tl_id in self._timelines_cache:
            return self._timelines_cache[tl_id]

        tl = self._build_timeline(spec)
        self._timelines_cache[tl_id] = tl
        return tl

    def create_timelines(self, uids: list[str] | None = None) -> list["Timeline"]:
        """Create multiple timelines.

        Args:
            uids: List of timeline identifiers to create.  If ``None``
                (the default), all timelines are created.

        Returns:
            List of ``ContinuousPhysicalTimeline`` objects.
        """
        if not self._timeline_specs:
            raise RuntimeError("No data loaded. Call load() before create_timelines().")

        if uids is None:
            uids = self.timeline_ids

        return [self.create_timeline(tid) for tid in uids]

    def create_group(self, uids: list[str] | None = None) -> "TimelineGroup":
        """Create a ``TimelineGroup`` containing all (or selected) timelines.

        This is the primary output method for TiliaJsonLoader.  The
        group is built with all member timelines mapped to the same
        physical time axis (seconds).

        Args:
            uids: Timeline identifiers to include.  ``None`` (default)
                means all.

        Returns:
            A ``TimelineGroup`` with one member per timeline.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        from timetoalign.timelines.groups import TimelineGroup

        if not self._timeline_specs:
            raise RuntimeError("No data loaded. Call load() before create_group().")

        timelines = self.create_timelines(uids)

        # Use the source filename (if available) as the group name
        group_name = None
        if self._sources:
            group_name = self._sources[-1].stem

        group = TimelineGroup(
            id=f"tilia:{group_name or 'group'}",
            name=group_name,
            timelines=timelines,
        )

        return group

    def create_bundle(self) -> "AlignmentBundle":
        """Create an ``AlignmentBundle`` wrapping all timelines in one group.

        This is a convenience method.  The bundle contains a single
        ``TimelineGroup`` with no cross-group ``MatchClaim`` objects.

        Returns:
            An ``AlignmentBundle`` with one group and no claims.

        Raises:
            RuntimeError: If ``load()`` has not been called.
        """
        from timetoalign.alignment.bundle import AlignmentBundle

        if not self._timeline_specs:
            raise RuntimeError("No data loaded. Call load() before " "create_bundle().")

        group = self.create_group()
        bundle = AlignmentBundle(name=group.name)
        bundle.add_group(group)

        return bundle

    # endregion

    # region Clear

    def clear(self) -> None:
        """Clear all loaded data, including timeline specs and cache."""
        super().clear()
        self._timeline_specs = []
        self._minted_timeline_ids = set()
        self._timelines_cache = {}
        self._media_length = 0.0
        # Re-create a fresh TiliaDictStore
        self._store = TiliaDictStore()

    # endregion

    # region Internal Helpers

    def _find_spec(self, uid: str | int) -> dict[str, Any]:
        """Look up a timeline spec by uid, name, or index.

        Matching precedence:
        1. Integer index (0-indexed).
        2. String-encoded integer index (e.g. ``"3"``).
        3. Exact match on spec ``"id"`` field.
        4. Exact match on the source ID.
        5. Exact match on spec ``"name"`` field.
        6. Partial/regex match on stored ID or source ID.
        7. Partial/regex match on spec ``"name"`` field.

        Args:
            uid: Timeline identifier, name, or integer index.

        Returns:
            The spec dict.

        Raises:
            KeyError: If not found.
        """
        # Try integer index
        if isinstance(uid, int):
            if 0 <= uid < len(self._timeline_specs):
                return self._timeline_specs[uid]
            raise KeyError(
                f"Timeline index {uid} out of range "
                f"(0-{len(self._timeline_specs) - 1})"
            )

        # Try string index (e.g. "3")
        try:
            idx = int(uid)
            if 0 <= idx < len(self._timeline_specs):
                return self._timeline_specs[idx]
        except ValueError:
            pass

        # Try exact match by id
        for spec in self._timeline_specs:
            if spec["id"] == uid:
                return spec

        # Try exact match by source id
        for spec in self._timeline_specs:
            if spec["source_id"] == uid:
                return spec

        # Try exact match by name
        for spec in self._timeline_specs:
            if spec["name"] == uid:
                return spec

        # Try partial/regex match by id
        all_ids = [s["id"] for s in self._timeline_specs]
        all_source_ids = [s["source_id"] for s in self._timeline_specs]
        try:
            resolved_id = resolve_id(uid, all_ids + all_source_ids, warn_multiple=True)
            for spec in self._timeline_specs:
                if resolved_id in (spec["id"], spec["source_id"]):
                    return spec
        except KeyError:
            pass

        # Try partial/regex match by name
        all_names = [s["name"] for s in self._timeline_specs]
        try:
            resolved_name = resolve_id(uid, all_names, warn_multiple=True)
            for spec in self._timeline_specs:
                if spec["name"] == resolved_name:
                    return spec
        except KeyError:
            pass

        raise KeyError(
            f"No timeline with uid or name matching '{uid}'. "
            f"Available UIDs: {all_ids}; source IDs: {all_source_ids}"
        )

    def _build_timeline(self, spec: dict[str, Any]) -> ContinuousPhysicalTimeline:
        """Build a ``ContinuousPhysicalTimeline`` from a timeline spec.

        Args:
            spec: The timeline spec dict from ``_timeline_specs``.

        Returns:
            A ``ContinuousPhysicalTimeline`` with events.
        """
        tl_id = spec["id"]
        kind = spec["kind"]
        name = spec["name"]
        raw = spec["raw"]
        components = raw.get("components", [])

        # Determine timeline length
        # Use media_length if available, otherwise compute from components
        length = self._media_length if self._media_length > 0 else 0.0

        if not length and components:
            # Compute from component coordinates
            max_coord = 0.0
            for comp in components:
                if "end" in comp:
                    max_coord = max(max_coord, float(comp["end"]))
                elif "time" in comp:
                    max_coord = max(max_coord, float(comp["time"]))
            length = max_coord

        tl = ContinuousPhysicalTimeline(
            length=length,
            unit=self._media_unit,
            uid=tl_id,
            name=name,
        )
        # The loader minted the id, so a scope that already holds it mints
        # it again rather than refusing the timeline.
        tl._set_id(tl_id, minted=True)

        # Convert components to events using the appropriate converter
        converter = _EVENT_CONVERTERS.get(kind, _generic_to_events)
        events = converter(components)

        if events:
            tl.add_events(events)

        return tl

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        if not self._timeline_specs:
            return "TiliaJsonLoader(not loaded)"
        entries = ", ".join(
            f"{s['id']}={s['n_components']} components" for s in self._timeline_specs
        )
        return f"TiliaJsonLoader({entries})"

    # endregion


# endregion
