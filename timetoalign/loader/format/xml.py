"""XmlLoader: Configurable XML normaliser producing flat PyArrow tables.

This module implements the ``XmlLoader`` class, a format-level loader that
parses XML files and normalises nested element structures into flat
``pa.Table`` objects -- one table per *principal element tag*.

The design mirrors ``JsonLoader`` but for XML: the user (or a subclass)
specifies one or more *principal tags* -- XML element tags whose instances
form arrays of objects.  Each such collection is flattened into its own
``pa.Table`` with one row per element.  Nested child elements are flattened
with dot-separated field names; element attributes become fields.

When no principal tags are specified, the loader auto-detects all element
tags that appear multiple times as children of any parent.

A `timetoalign.storage.store.DictStore` is used as the internal storage,
keyed by element tag name.  Each normalised ``pa.Table`` is wrapped in an
`timetoalign.storage.events.EventData` for uniform access through the
``EventStore`` interface.

Usage::

    loader = XmlLoader()
    loader.load("data.xml")

    # Access via store
    for name in loader.store.keys():
        table = loader.get_table(name)

    # Or specify principal tags
    loader = XmlLoader(principal_tags=["Signal", "Audio"])
    loader.load("repovizz_manifest.xml")
    table = loader.get_table("Signal")  # N rows, one per Signal element

The loader follows the standard two-phase pattern:
``loader.load(*sources)`` then ``loader.get_table(tag)`` or
``loader.store`` for the ``DictStore``.

See Also:
    timetoalign.storage.store.DictStore
    timetoalign.loader.base.Loader
    timetoalign.loader.format.json.JsonLoader
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import pyarrow as pa
from typing_extensions import Self

from timetoalign.core import NumberType, TimeUnit
from timetoalign.loader._sources import Source, readable
from timetoalign.loader._xml import XML_ID, element_paths
from timetoalign.loader.base import LoadSourceResult, locate
from timetoalign.storage.store import DictStore

from ._tables import NormalisedTablesLoader

module_logger = logging.getLogger(__name__)


# region Normalisation helpers


def _element_to_dict(
    elem: ET.Element,
    sep: str = ".",
    include_text: bool = True,
    text_key: str = "_text",
) -> dict[str, Any]:
    """Convert an XML element to a flat dictionary.

    Attributes become keys directly.  Nested child elements become
    dot-separated keys (recursive flattening).  Element text content
    is stored under the *text_key* if non-empty.

    Args:
        elem: The XML element to convert.
        sep: Separator for nested key names.
        include_text: Whether to include element text as a field.
        text_key: Key name for element text content.

    Returns:
        A flat dict with string keys and primitive/list values.
    """
    result: dict[str, Any] = {}

    # Add all attributes
    for attr_name, attr_value in elem.attrib.items():
        # Skip internal attributes (starting with _)
        if not attr_name.startswith("_"):
            result[attr_name] = _parse_value(attr_value)

    # Add text content if present and non-whitespace
    if include_text and elem.text and elem.text.strip():
        result[text_key] = elem.text.strip()

    # Recursively flatten child elements
    child_counts: dict[str, int] = {}
    for child in elem:
        tag = child.tag
        child_counts[tag] = child_counts.get(tag, 0) + 1

    # Track which children we've seen for indexing
    child_indices: dict[str, int] = {}
    for child in elem:
        tag = child.tag
        child_dict = _element_to_dict(child, sep, include_text, text_key)

        # If this child tag appears multiple times, index them
        if child_counts[tag] > 1:
            idx = child_indices.get(tag, 0)
            child_indices[tag] = idx + 1
            prefix = f"{tag}{sep}{idx}"
        else:
            prefix = tag

        # Add all flattened child keys with prefix
        for key, value in child_dict.items():
            result[f"{prefix}{sep}{key}"] = value

    return result


def _parse_value(value: str) -> Any:
    """Parse a string value to an appropriate Python type.

    Tries int, float, bool, then falls back to string.

    Args:
        value: The string value to parse.

    Returns:
        Parsed value as int, float, bool, or str.
    """
    # Try int
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Try bool
    if value.lower() in ("true", "yes", "1"):
        return True
    if value.lower() in ("false", "no", "0"):
        return False

    return value


def _collect_elements_by_tag(
    root: ET.Element,
    tag: str,
    parent_context: dict[str, Any] | None = None,
    sep: str = ".",
    include_text: bool = True,
    text_key: str = "_text",
) -> list[tuple[ET.Element, dict[str, Any]]]:
    """Collect all elements with a given tag, flattening each to a dict.

    Walks the entire tree and collects all elements matching *tag*,
    propagating scalar attributes from ancestor elements downward.

    Args:
        root: The root element to search.
        tag: The element tag to collect.
        parent_context: Inherited attributes from parent elements.
        sep: Separator for nested key names.
        include_text: Whether to include element text as a field.
        text_key: Key name for element text content.

    Returns:
        Per matching element, in document order, the element and its flat
        dict.
    """
    if parent_context is None:
        parent_context = {}

    results: list[tuple[ET.Element, dict[str, Any]]] = []

    def _walk(elem: ET.Element, context: dict[str, Any]) -> None:
        # Build context from this element's attributes
        ctx = dict(context)
        for attr_name, attr_value in elem.attrib.items():
            if not attr_name.startswith("_"):
                ctx[f"parent_{elem.tag}_{attr_name}"] = _parse_value(attr_value)

        if elem.tag == tag:
            # This element matches -- flatten it
            elem_dict = _element_to_dict(elem, sep, include_text, text_key)
            # Merge parent context (parent attrs appear first)
            row = dict(ctx)
            row.update(elem_dict)
            results.append((elem, row))
        else:
            # Recurse into children
            for child in elem:
                _walk(child, ctx)

    _walk(root, parent_context)
    return results


def _normalise_elements(
    rows: list[dict[str, Any]],
) -> pa.Table:
    """Normalise a list of element dicts into a ``pa.Table``.

    Union of all keys across rows forms the field set.  PyArrow infers
    types automatically.

    Args:
        rows: List of flat dicts to normalise.

    Returns:
        A ``pa.Table`` with one row per input dict.
    """
    if not rows:
        return pa.table({})

    # Collect all field names in insertion order
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    # Build field arrays
    field_lists: dict[str, list[Any]] = {k: [] for k in all_keys}
    for row in rows:
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


def _detect_principal_tags(root: ET.Element, min_count: int = 2) -> list[str]:
    """Auto-detect principal tags (tags appearing multiple times).

    Walks the tree and counts occurrences of each tag.  Tags that appear
    at least *min_count* times are considered principal.

    Args:
        root: The root element.
        min_count: Minimum occurrences to consider principal.

    Returns:
        List of principal tag names, sorted by frequency (descending).
    """
    counts: dict[str, int] = {}

    def _count(elem: ET.Element) -> None:
        counts[elem.tag] = counts.get(elem.tag, 0) + 1
        for child in elem:
            _count(child)

    _count(root)

    # Filter and sort by count descending
    principal = [(tag, count) for tag, count in counts.items() if count >= min_count]
    principal.sort(key=lambda x: -x[1])

    return [tag for tag, _ in principal]


# endregion


# region XmlLoader


class XmlLoader(NormalisedTablesLoader):
    """Configurable XML normaliser producing flat ``pa.Table`` objects.

    ``XmlLoader`` parses one or more XML files and normalises their
    nested element structures into flat PyArrow tables stored in a
    `timetoalign.storage.store.DictStore`.

    **Principal tags** determine which element types become tables:

    - If *principal_tags* is given (a list of tag names), only elements
      with those tags are collected and normalised.
    - If *principal_tags* is ``None`` (the default), tags appearing at
      least twice in the document are auto-detected as principal.

    Each element's attributes become fields.  Nested child elements are
    flattened with dot-separated field names.  Ancestor attributes are
    propagated downward with ``parent_<tag>_<attr>`` prefixes. A row's
    ``locator`` is its element's path in the document (``/root/item[2]``:
    local names, a position among same-named siblings where there are
    several), followed by ``key:<key>`` where the element states a key of
    its own, its ``xml:id``, else its ``id`` (``/root/item[2] key:i2``); a
    row read from a file names that file's parse (:attr:`parses`),
    one read from a string or an element none.

    Follows the standard two-phase pattern:

    1. ``loader.load(*sources)`` -- parse XML files.
    2. ``loader.get_table(tag)`` -- retrieve a normalised table.

    Args:
        principal_tags: List of element tags to normalise.  ``None`` for
            auto-detect.
        sep: Separator for flattened nested key names.  Default ``"."``.
        include_text: Whether to include element text as ``_text`` field.
            Default ``True``.
        propagate_ancestors: Whether to propagate ancestor attributes.
            Default ``True``.
        seed: The seed of the loader's ID minter (see ``Loader.load``).

    Examples:
        Auto-detect all repeated tags::

            loader = XmlLoader()
            loader.load("data.xml")
            for tag in loader.store.keys():
                print(tag, loader.get_table(tag).num_rows)

        Specify principal tags::

            loader = XmlLoader(principal_tags=["Signal", "Audio"])
            loader.load("manifest.xml")
            assert loader.get_table("Signal").num_rows > 0

    See Also:
        timetoalign.storage.store.DictStore
        timetoalign.loader.base.Loader
        timetoalign.loader.format.json.JsonLoader
    """

    _default_unit = TimeUnit.seconds

    def __init__(
        self,
        principal_tags: list[str] | None = None,
        *,
        sep: str = ".",
        include_text: bool = True,
        propagate_ancestors: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(unit=TimeUnit.seconds, number_type=NumberType.float, seed=seed)
        self._principal_tags = principal_tags
        self._sep = sep
        self._include_text = include_text
        self._propagate_ancestors = propagate_ancestors
        self._store: DictStore = DictStore()
        self._file_metadata: dict[str, Any] = {}
        self._raw_root: ET.Element | None = None
        # The file being read, while it is (None for in-memory XML).
        self._reading: Source | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    def _seed_config(self) -> dict[str, object]:
        """The normalisation options, which shape the tables, for the seed."""
        return {
            **super()._seed_config(),
            "principal_tags": self._principal_tags,
            "sep": self._sep,
            "include_text": self._include_text,
            "propagate_ancestors": self._propagate_ancestors,
        }

    # region Properties

    @property
    def store(self) -> DictStore:
        """The ``DictStore`` containing all normalised tables.

        Each principal tag maps to an ``EventData`` wrapping the
        normalised ``pa.Table``.

        Returns:
            A ``DictStore`` with one entry per principal tag.
        """
        return self._store

    @property
    def tables(self) -> dict[str, pa.Table]:
        """Dict mapping tag names to normalised ``pa.Table`` objects.

        Convenience accessor that unwraps the raw PyArrow tables from
        the ``EventData`` wrappers in the store.
        """
        return {name: ed.table for name, ed in self._store.items()}

    @property
    def raw_root(self) -> ET.Element | None:
        """The raw parsed XML root element from the last loaded file."""
        return self._raw_root

    @property
    def file_metadata(self) -> dict[str, Any]:
        """Metadata from the root element's attributes."""
        return dict(self._file_metadata)

    # endregion

    # region Loading

    def _load_source(self, source: Source) -> LoadSourceResult:
        """Parse one XML file into the loader's tables.

        ``load()`` reads each source through this hook. For multiple files
        the tables are concatenated (same tags) or added (new tags). The
        tables are the loader's store, so no event arrays are returned.

        Args:
            source: The XML file: a path or a ZIP archive member.

        Returns:
            No metadata and no event arrays.
        """
        with readable(source) as document:
            root = ET.parse(document).getroot()
        self._raw_root = root
        self._reading = source
        try:
            self._process_xml(root)
        finally:
            self._reading = None
        self._logger.debug("Loaded %s: %s", source.name, list(self._store.keys()))
        return {}, {}

    def load_string(self, xml_string: str) -> Self:
        """Load from an XML string.

        Convenient for programmatic use or testing.

        The loader's ID minter is seeded from the string's text (see
        ``Loader.load``).

        Args:
            xml_string: A valid XML string.

        Returns:
            Self, for method chaining.
        """
        self._start_minter(xml_string)
        root = ET.fromstring(xml_string)
        self._raw_root = root
        self._process_xml(root)
        return self

    def load_element(self, root: ET.Element) -> Self:
        """Load from an already-parsed Element tree.

        Convenient for programmatic use or testing.

        The loader's ID minter is seeded from the element serialized
        (``xml.etree.ElementTree.tostring``; see ``Loader.load``).

        Args:
            root: An XML Element.

        Returns:
            Self, for method chaining.
        """
        self._start_minter(ET.tostring(root))
        self._raw_root = root
        self._process_xml(root)
        return self

    def clear(self) -> None:
        """Clear all loaded data."""
        super().clear()
        self._store = DictStore()
        self._file_metadata.clear()
        self._raw_root = None

    # endregion

    # region Access

    def get_table(self, tag: str) -> pa.Table:
        """Retrieve a normalised table by element tag.

        Args:
            tag: The principal tag name.

        Returns:
            A ``pa.Table`` with one row per element.

        Raises:
            KeyError: If *tag* was not normalised.
        """
        return self._table(tag, "tag")

    # endregion

    # region Internal

    def _process_xml(self, root: ET.Element) -> None:
        """Process a parsed XML root into normalised tables.

        Args:
            root: The root XML element.
        """
        # Extract root metadata
        self._file_metadata.update(
            {
                k: _parse_value(v)
                for k, v in root.attrib.items()
                if not k.startswith("_")
            }
        )

        # Determine principal tags
        principal = self._determine_principal_tags(root)

        # Normalise each principal tag; a row's locator is its element's
        # path and its key, the xml:id, else the id it states.
        paths: dict[int, str] | None = None
        for tag in principal:
            if self._propagate_ancestors:
                found = _collect_elements_by_tag(
                    root, tag, sep=self._sep, include_text=self._include_text
                )
            else:
                found = [
                    (elem, _element_to_dict(elem, self._sep, self._include_text))
                    for elem in root.iter(tag)
                ]

            if found:
                if paths is None:
                    paths = element_paths(root)
                table = self._stamped(
                    _normalise_elements([row for _, row in found]),
                    [
                        locate(paths[id(elem)], elem.get(XML_ID) or elem.get("id"))
                        for elem, _ in found
                    ],
                )

                # Store metadata on the table
                meta_bytes = {
                    k.encode(): str(v).encode() for k, v in self._file_metadata.items()
                }
                if meta_bytes:
                    existing = table.schema.metadata or {}
                    existing.update(meta_bytes)
                    table = table.replace_schema_metadata(existing)

                # Accumulate or replace
                if tag in self._store:
                    existing_table = self._store[tag].table
                    merged = pa.concat_tables(
                        [existing_table, table],
                        promote_options="default",
                    )
                    self._store.add(tag, self._wrap_table(merged))
                else:
                    self._store.add(tag, self._wrap_table(table))

    def _determine_principal_tags(self, root: ET.Element) -> list[str]:
        """Determine which tags to normalise.

        If ``self._principal_tags`` is set, use those.  Otherwise
        auto-detect tags appearing at least twice.

        Args:
            root: The root XML element.

        Returns:
            List of principal tag names.
        """
        if self._principal_tags is not None:
            return list(self._principal_tags)

        # Auto-detect: tags appearing at least twice
        return _detect_principal_tags(root, min_count=2)

    # endregion

    # region Magic Methods

    def __repr__(self) -> str:
        entries = ", ".join(
            f"{k}={ed.table.num_rows} rows" for k, ed in self._store.items()
        )
        return f"XmlLoader({entries})"

    # endregion


# endregion
