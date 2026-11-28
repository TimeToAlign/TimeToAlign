"""The table access shared by the configurable format loaders."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from timetoalign.loader._sources import Source
from timetoalign.loader.base import LOCATOR, EventLoader
from timetoalign.storage.events import EventData
from timetoalign.storage.store import DictStore


class NormalisedTablesLoader(EventLoader):
    """Base of the loaders that normalise a document into named tables.

    A subclass fills ``self._store``, a ``DictStore`` holding one
    ``EventData`` per table name (a principal JSON key or XML tag), and
    wraps each table it adds with :meth:`_wrap_table`. Its ``get_table``
    names the kind of name it takes and asks :meth:`_table`.

    Each row of a table holds its ``locator``, the place in the document it
    was read from (the subclass's rule), and, read from a file, the
    ``extraction_id`` of that file's extraction (:meth:`_stamped`). A field
    of the document named ``locator`` or ``extraction_id`` is replaced by
    these: the names are the registry's.
    """

    _store: DictStore

    #: The file being read, while it is; None for in-memory content.
    _reading: Source | None

    def __len__(self) -> int:
        """Return the number of normalised tables."""
        return len(self._store)

    def keys(self) -> list[str]:
        """Return the names of the available tables."""
        return list(self._store.keys())

    def _table(self, name: str, kind: str) -> pa.Table:
        """Return the table *name*, or raise naming *kind* and the available names.

        Raises:
            KeyError: If no table is named *name*.
        """
        try:
            return self._store[name].table
        except KeyError:
            raise KeyError(
                f"No table for {kind} {name!r}. Available: {list(self._store.keys())}"
            ) from None

    def _stamped(self, table: pa.Table, locators: Sequence[str]) -> pa.Table:
        """*table* with each row's *locator* and, read from a file, its extraction.

        Args:
            table: The rows read, in document order.
            locators: Row for row, its place in the document.
        """
        located = pa.array(locators, pa.string())
        columns = (
            {LOCATOR: located}
            if self._reading is None
            else self._stamps(self._reading, located)
        )
        for name, column in columns.items():
            at = table.schema.get_field_index(name)
            table = (
                table.append_column(name, column)
                if at < 0
                else table.set_column(at, name, column)
            )
        return table

    def _wrap_table(self, table: pa.Table) -> EventData:
        """Wrap a raw ``pa.Table`` in an ``EventData`` for store compatibility.

        The ``EventData`` carries the extractions its rows name.

        Args:
            table: The normalised PyArrow table.

        Returns:
            An ``EventData`` wrapping the table.
        """
        return self._carrying(EventData(table, self._unit, self._number_type))
