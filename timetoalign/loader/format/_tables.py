"""The table access shared by the configurable format loaders."""

from __future__ import annotations

import pyarrow as pa

from timetoalign.loader.base import EventLoader
from timetoalign.storage.events import EventData
from timetoalign.storage.store import DictStore


class NormalisedTablesLoader(EventLoader):
    """Base of the loaders that normalise a document into named tables.

    A subclass fills ``self._store``, a ``DictStore`` holding one
    ``EventData`` per table name (a principal JSON key or XML tag), and
    wraps each table it adds with :meth:`_wrap_table`. Its ``get_table``
    names the kind of name it takes and asks :meth:`_table`.
    """

    _store: DictStore

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
            return self._store[name]._table
        except KeyError:
            raise KeyError(
                f"No table for {kind} {name!r}. Available: {list(self._store.keys())}"
            ) from None

    def _wrap_table(self, table: pa.Table) -> EventData:
        """Wrap a raw ``pa.Table`` in an ``EventData`` for store compatibility.

        Args:
            table: The normalised PyArrow table.

        Returns:
            An ``EventData`` wrapping the table.
        """
        return EventData(table, self._unit, self._number_type)
