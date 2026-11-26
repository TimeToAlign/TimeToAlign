"""JsonLoader: Configurable JSON normaliser producing flat PyArrow tables.

This module implements the ``JsonLoader`` class, a format-level loader that
parses JSON files (or Python dicts) and normalises nested structures into
flat ``pa.Table`` objects -- one table per *principal key*.

The design mirrors what ``pandas.json_normalize`` does recursively, but uses
PyArrow throughout for efficient columnar storage.  The user (or a subclass)
specifies one or more *principal keys* -- JSON object keys whose values are
arrays of objects.  Each such array is flattened into its own ``pa.Table``
with one row per element.  Nested sub-objects are flattened with
dot-separated field names; nested arrays become list-typed fields.

When no principal keys are specified, the loader auto-detects all top-level
keys whose values are arrays of objects.

A `timetoalign.storage.store.DictStore` is used as the internal storage,
keyed by principal key name.  Each normalised ``pa.Table`` is wrapped in an
`timetoalign.storage.store.EventData` for uniform access through the
``EventStore`` interface.

Usage::

    loader = JsonLoader()
    loader.load("data.json")

    # Access via store
    for name in loader.store.keys():
        table = loader.get_table(name)

    # Or specify principal keys
    loader = JsonLoader(principal_keys=["audio"])
    loader.load("dj_studio_data.json")
    table = loader.get_table("audio")  # 3 rows

The loader follows the standard two-phase pattern:
``loader.load(*sources)`` then ``loader.get_table(key)`` or
``loader.store`` for the ``DictStore``.

See Also:
    timetoalign.storage.store.DictStore
    timetoalign.loader.base.Loader
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader._sources import Source
from timetoalign.loader.base import LoadSourceResult
from timetoalign.storage.store import DictStore

from ._tables import NormalisedTablesLoader

module_logger = logging.getLogger(__name__)


# region Normalisation helpers


def _flatten_dict(
    obj: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> dict[str, Any]:
    """Flatten a nested dict into a single-level dict with compound keys.

    Nested dicts are expanded with *sep*-separated keys.  Nested lists
    and scalar values are kept as-is.

    Args:
        obj: The dict to flatten.
        parent_key: Prefix for all keys (used in recursion).
        sep: Separator between parent and child key names.

    Returns:
        A flat dict.
    """
    items: list[tuple[str, Any]] = []
    for key, value in obj.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def _normalise_array(
    rows: list[dict[str, Any]],
    sep: str = ".",
) -> pa.Table:
    """Normalise a list of (possibly nested) dicts into a ``pa.Table``.

    Each dict is flattened via :func:`_flatten_dict`, then the union of
    all keys across rows forms the field set.  PyArrow infers types
    automatically.

    Args:
        rows: List of dict objects to normalise.
        sep: Separator for nested key names.

    Returns:
        A ``pa.Table`` with one row per input dict.
    """
    if not rows:
        return pa.table({})

    flat_rows = [_flatten_dict(row, sep=sep) for row in rows]

    # Collect all field names in insertion order
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in flat_rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build field arrays
    field_lists: dict[str, list[Any]] = {k: [] for k in all_keys}
    for row in flat_rows:
        for k in all_keys:
            field_lists[k].append(row.get(k))

    # Let PyArrow infer types from the Python lists
    arrays: dict[str, pa.Array] = {}
    for k, vals in field_lists.items():
        try:
            arrays[k] = pa.array(vals)
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fall back to string representation for mixed types
            arrays[k] = pa.array([str(v) if v is not None else None for v in vals])

    return pa.table(arrays)


def _collect_nested_rows(
    data: Any,
    key_path: list[str],
    parent_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Walk a nested JSON structure and collect rows for a given key path.

    For a key path like ``["audio", "cueData", "hotCuePoints"]``, this
    walks ``data["audio"]`` (which must be a list), and for each element
    walks ``element["cueData"]["hotCuePoints"]`` (also a list), yielding
    one flattened dict per leaf element.  Parent scalar fields are
    propagated downward so that, e.g., the audio item's ``name`` appears
    on every hotCuePoint row.

    Args:
        data: The root JSON object (dict or list).
        key_path: Sequence of keys to walk.
        parent_context: Inherited scalar fields from parent objects.

    Returns:
        List of flat dicts, one per leaf-level element.
    """
    if parent_context is None:
        parent_context = {}

    if not key_path:
        # We've consumed all keys -- ``data`` should be a list of objects
        if isinstance(data, list):
            results = []
            for item in data:
                if isinstance(item, dict):
                    row = dict(parent_context)
                    row.update(_flatten_dict(item))
                    results.append(row)
                else:
                    row = dict(parent_context)
                    row["value"] = item
                    results.append(row)
            return results
        elif isinstance(data, dict):
            row = dict(parent_context)
            row.update(_flatten_dict(data))
            return [row]
        else:
            return [dict(parent_context, value=data)]

    head, *tail = key_path
    if isinstance(data, dict):
        child = data.get(head)
        if child is None:
            return []
        if isinstance(child, list):
            # Array at this level -- iterate, propagate parent scalars
            results = []
            # Gather parent scalars from current dict
            ctx = dict(parent_context)
            for k, v in data.items():
                if k == head:
                    continue
                if isinstance(v, (dict, list)):
                    continue  # skip non-scalar
                ctx[k] = v
            for item in child:
                results.extend(_collect_nested_rows(item, tail, parent_context=ctx))
            return results
        else:
            # Not a list -- descend
            return _collect_nested_rows(child, tail, parent_context)
    elif isinstance(data, list):
        results = []
        for item in data:
            results.extend(_collect_nested_rows(item, key_path, parent_context))
        return results
    else:
        return []


# endregion


# region Lookup resolution


def _resolve_lookups(
    table: pa.Table,
    lookup_tables: dict[str, pa.Table],
    *,
    suffix: str = "_id",
) -> pa.Table:
    """Replace ``*_id`` foreign-key fields with resolved values from lookup tables.

    For each field whose name ends with *suffix* (e.g. ``image_id``),
    look for a matching lookup table (``images``) in *lookup_tables*.
    If found and the lookup table has an ``id`` field, perform a left
    join so that the foreign-key field is augmented with the lookup's
    other fields (prefixed by the lookup name).

    This is a best-effort convenience; unresolvable fields are left
    untouched.

    Args:
        table: The table to enrich.
        lookup_tables: Dict mapping key names to ``pa.Table`` objects
            (typically the non-principal top-level keys).
        suffix: The foreign-key suffix to detect.

    Returns:
        A new table with resolved fields appended.
    """
    for field_name in table.column_names:
        if not field_name.endswith(suffix):
            continue
        # Derive the lookup key name: image_id -> images, category_id -> categories
        base = field_name[: -len(suffix)]
        # Try plural forms (including y -> ies)
        candidates = [base + "s", base + "es", base]
        if base.endswith("y"):
            candidates.insert(0, base[:-1] + "ies")
        for candidate in candidates:
            if candidate in lookup_tables:
                lut = lookup_tables[candidate]
                if "id" not in lut.column_names:
                    continue
                # Perform the join: bring in all non-id fields from the LUT
                # prefixed with the base name
                lut_names = [c for c in lut.column_names if c != "id"]
                if not lut_names:
                    continue
                # Build a Python dict lookup for efficiency
                lut_dict: dict[Any, dict[str, Any]] = {}
                for row in lut.to_pylist():
                    lut_dict[row["id"]] = {f"{base}.{c}": row[c] for c in lut_names}
                # Resolve each foreign key
                fk_values = table.column(field_name).to_pylist()
                resolved: dict[str, list[Any]] = {f"{base}.{c}": [] for c in lut_names}
                for fk in fk_values:
                    match = lut_dict.get(fk)
                    for rc in resolved:
                        resolved[rc].append(match[rc] if match else None)
                # Append resolved fields to the table
                for rc, vals in resolved.items():
                    try:
                        table = table.append_column(rc, pa.array(vals))
                    except (pa.ArrowInvalid, pa.ArrowTypeError):
                        table = table.append_column(
                            rc,
                            pa.array([str(v) if v is not None else None for v in vals]),
                        )
                break  # resolved this field
    return table


# endregion


# region JsonLoader


class JsonLoader(NormalisedTablesLoader):
    """Configurable JSON normaliser producing flat ``pa.Table`` objects.

    ``JsonLoader`` parses one or more JSON files and normalises their
    nested structures into flat PyArrow tables stored in a
    `timetoalign.storage.store.DictStore`.

    **Principal keys** determine which top-level arrays become tables:

    - If *principal_keys* is given (a list of strings, possibly dotted
      for nested paths like ``"cueData.hotCuePoints"``), only those keys
      are normalised.
    - If *principal_keys* is ``None`` (the default), every top-level key
      whose value is a list of dicts is auto-detected and normalised.

    Non-principal top-level keys whose values are lists of dicts are kept
    as *lookup tables* and used to resolve foreign-key fields (fields
    ending in ``_id``) automatically.  For example, if ``annotations``
    rows contain ``image_id`` and there is a top-level ``images`` array,
    the loader appends ``image.file_name``, ``image.width``, etc. to the
    annotations table.

    Scalar top-level keys (like ``"width": 604``) are stored as table-
    level metadata on every resulting ``pa.Table``.

    Follows the standard two-phase pattern:

    1. ``loader.load(*sources)`` -- parse JSON files.
    2. ``loader.get_table(key)`` -- retrieve a normalised table.

    Args:
        principal_keys: List of keys to normalise.  ``None`` for auto-detect.
        sep: Separator for flattened nested key names.  Default ``"."``.
        resolve_lookups: Whether to resolve ``*_id`` foreign keys.  Default ``True``.

    Examples:
        Auto-detect all array keys::

            loader = JsonLoader()
            loader.load("Wagner_WWV086B_140.json")
            for name in loader.store.keys():
                print(name, loader.get_table(name).num_rows)

        Specify principal key::

            loader = JsonLoader(principal_keys=["audio"])
            loader.load("dj_studio_data.json")
            assert loader.get_table("audio").num_rows == 3

        Nested principal key::

            loader = JsonLoader(principal_keys=["hotCuePoints"])
            loader.load("dj_studio_data.json")
            # Collects hotCuePoints from audio[*].cueData.hotCuePoints

    See Also:
        timetoalign.storage.store.DictStore
        timetoalign.loader.base.Loader
    """

    _default_unit = TimeUnit.seconds

    def __init__(
        self,
        principal_keys: list[str] | None = None,
        *,
        sep: str = ".",
        resolve_lookups: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(unit=TimeUnit.seconds, number_type=NumberType.float, **kwargs)
        self._principal_keys = principal_keys
        self._sep = sep
        self._resolve_lookups = resolve_lookups
        self._store: DictStore = DictStore()
        self._file_metadata: dict[str, Any] = {}
        self._raw_data: dict[str, Any] | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    # region Properties

    @property
    def store(self) -> DictStore:
        """The ``DictStore`` containing all normalised tables.

        Each principal key maps to an ``EventData`` wrapping the
        normalised ``pa.Table``.

        Returns:
            A ``DictStore`` with one entry per principal key.
        """
        return self._store

    @property
    def tables(self) -> dict[str, pa.Table]:
        """Dict mapping key names to normalised ``pa.Table`` objects.

        Convenience accessor that unwraps the raw PyArrow tables from
        the ``EventData`` wrappers in the store.
        """
        return {name: ed.table for name, ed in self._store.items()}

    @property
    def raw_data(self) -> dict[str, Any] | None:
        """The raw parsed JSON data from the last loaded file."""
        return self._raw_data

    @property
    def file_metadata(self) -> dict[str, Any]:
        """Scalar top-level metadata from the JSON file."""
        return dict(self._file_metadata)

    # endregion

    # region Loading

    def _load_source(self, source: Source) -> LoadSourceResult:
        """Parse one JSON file into the loader's tables.

        ``load()`` reads each source through this hook. For multiple files
        the tables are concatenated (same keys) or added (new keys). The
        tables are the loader's store, so no event arrays are returned.

        Args:
            source: The JSON file: a path or a ZIP archive member.

        Returns:
            No metadata and no event arrays.
        """
        with source.open(encoding="utf-8") as f:
            data = json.load(f)
        self._raw_data = data
        self._process_json(data)
        self._logger.debug("Loaded %s: %s", source.name, list(self._store.keys()))
        return {}, {}

    def load_dict(self, data: dict[str, Any]) -> Self:
        """Load from an already-parsed Python dict.

        Convenient for programmatic use or testing.

        The loader's ID minter is seeded from the dict's JSON with sorted
        keys and no spaces (see ``Loader.load``).

        Args:
            data: A parsed JSON structure (dict).

        Returns:
            Self, for method chaining.
        """
        self._start_minter(json.dumps(data, sort_keys=True, separators=(",", ":")))
        self._raw_data = data
        self._process_json(data)
        return self

    def clear(self) -> None:
        """Clear all loaded data."""
        super().clear()
        self._store = DictStore()
        self._file_metadata.clear()
        self._raw_data = None

    # endregion

    # region Access

    def get_table(self, key: str) -> pa.Table:
        """Retrieve a normalised table by key name.

        Args:
            key: The principal key name.

        Returns:
            A ``pa.Table`` with one row per element.

        Raises:
            KeyError: If *key* was not normalised.
        """
        return self._table(key, "key")

    # endregion

    # region Internal

    def _process_json(self, data: dict[str, Any]) -> None:
        """Process a parsed JSON dict into normalised tables.

        Args:
            data: The root JSON object.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object (dict), got {type(data).__name__}")

        # Separate scalar metadata from structured data
        scalar_meta: dict[str, Any] = {}
        array_keys: dict[str, list[Any]] = {}
        for key, value in data.items():
            if isinstance(value, list):
                array_keys[key] = value
            elif isinstance(value, dict):
                # Nested dict at top level -- skip for now (not an array)
                scalar_meta[key] = value
            else:
                scalar_meta[key] = value

        self._file_metadata.update(
            {k: v for k, v in scalar_meta.items() if not isinstance(v, dict)}
        )

        # Determine principal keys
        principal = self._determine_principal_keys(data, array_keys)

        # Build lookup tables from non-principal array keys
        lookup_tables: dict[str, pa.Table] = {}
        for key, arr in array_keys.items():
            if key not in principal and arr and isinstance(arr[0], dict):
                lookup_tables[key] = _normalise_array(arr, sep=self._sep)

        # Normalise each principal key
        for key in principal:
            table = self._normalise_principal_key(data, key, array_keys)
            if table is not None and table.num_rows > 0:
                # Resolve lookups if enabled
                if self._resolve_lookups and lookup_tables:
                    table = _resolve_lookups(table, lookup_tables)

                # Store metadata on the table
                meta_bytes = {
                    k.encode(): str(v).encode()
                    for k, v in scalar_meta.items()
                    if not isinstance(v, (dict, list))
                }
                if meta_bytes:
                    existing = table.schema.metadata or {}
                    existing.update(meta_bytes)
                    table = table.replace_schema_metadata(existing)

                # Accumulate or replace
                if key in self._store:
                    existing_table = self._store[key].table
                    merged = pa.concat_tables(
                        [existing_table, table],
                        promote_options="default",
                    )
                    self._store.add(key, self._wrap_table(merged))
                else:
                    self._store.add(key, self._wrap_table(table))

    def _determine_principal_keys(
        self,
        data: dict[str, Any],
        array_keys: dict[str, list[Any]],
    ) -> list[str]:
        """Determine which keys to normalise.

        If ``self._principal_keys`` is set, use those.  Otherwise
        auto-detect all top-level keys whose values are arrays of dicts.

        Args:
            data: The root JSON object.
            array_keys: Top-level keys with list values.

        Returns:
            List of principal key names.
        """
        if self._principal_keys is not None:
            return list(self._principal_keys)

        # Auto-detect: keys whose values are lists of dicts
        detected = []
        for key, arr in array_keys.items():
            if arr and isinstance(arr[0], dict):
                detected.append(key)
        return detected

    def _normalise_principal_key(
        self,
        data: dict[str, Any],
        key: str,
        array_keys: dict[str, list[Any]],
    ) -> pa.Table | None:
        """Normalise a single principal key into a ``pa.Table``.

        Handles both top-level keys (``"audio"``) and nested paths
        (``"hotCuePoints"`` found by searching inside ``audio[*]``).

        Args:
            data: The root JSON object.
            key: The principal key name (possibly nested via dot notation or
                 a simple name that must be discovered).
            array_keys: Top-level keys with list values.

        Returns:
            A ``pa.Table``, or ``None`` if the key could not be resolved.
        """
        # Case 1: direct top-level key
        if key in array_keys:
            arr = array_keys[key]
            if arr and isinstance(arr[0], dict):
                return _normalise_array(arr, sep=self._sep)
            return None

        # Case 2: dotted path (e.g., "cueData.hotCuePoints")
        if self._sep in key:
            parts = key.split(self._sep)
            head = parts[0]
            tail = parts[1:]
            if head in array_keys:
                rows = _collect_nested_rows(array_keys[head], tail)
                if rows:
                    return _normalise_array(rows, sep=self._sep)
            return None

        # Case 3: search for the key recursively in all array values
        # E.g., "hotCuePoints" found in audio[*].cueData.hotCuePoints
        rows = self._search_for_key(data, key)
        if rows:
            return _normalise_array(rows, sep=self._sep)

        self._logger.warning(f"Principal key {key!r} not found in JSON structure.")
        return None

    def _search_for_key(
        self,
        data: Any,
        target_key: str,
        parent_context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Recursively search for a key in a nested JSON structure.

        Collects all values found under *target_key* across all nesting
        levels, propagating parent scalar fields downward.

        Args:
            data: Current JSON node.
            target_key: The key name to find.
            parent_context: Inherited scalar fields from parent objects.

        Returns:
            List of flat row dicts.
        """
        if parent_context is None:
            parent_context = {}
        results: list[dict[str, Any]] = []

        if isinstance(data, dict):
            # Gather scalar context from this dict
            ctx = dict(parent_context)
            for k, v in data.items():
                if k == target_key:
                    continue
                if not isinstance(v, (dict, list)):
                    ctx[k] = v

            if target_key in data:
                value = data[target_key]
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            row = dict(ctx)
                            row.update(_flatten_dict(item, sep=self._sep))
                            results.append(row)
                        else:
                            results.append(dict(ctx, value=item))
                elif isinstance(value, dict):
                    row = dict(ctx)
                    row.update(_flatten_dict(value, sep=self._sep))
                    results.append(row)
                else:
                    results.append(dict(ctx, value=value))
            else:
                # Recurse into sub-dicts and sub-lists
                for k, v in data.items():
                    if isinstance(v, (dict, list)):
                        results.extend(self._search_for_key(v, target_key, ctx))
        elif isinstance(data, list):
            for item in data:
                results.extend(self._search_for_key(item, target_key, parent_context))

        return results

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{k}={ed.table.num_rows} rows" for k, ed in self._store.items()
        )
        return f"JsonLoader({entries})"

    # endregion


# endregion
