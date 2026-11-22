"""Tests for :meth:`Loader.get_events`'s ``properties=`` parameter.

The ``properties=`` argument controls which *raw, unconsumed source
columns* survive as property columns alongside the fields produced by
``column_specs`` / ``field_specs``:

* Column-spec emissions are **fields** (they carry ``b"timetoalign"``
  metadata) and are always preserved — they are NOT property columns.
* Source-DataFrame columns that no ``column_specs`` entry consumed are
  property columns; they survive only when ``properties`` includes
  them.

The ``properties=`` argument accepts four shapes:

* ``True`` (default) — every property column included.
* ``False`` — every property column dropped; only fields + core
  columns (``id``, ``name``, ``event_type``, ``temporal_type``) and
  canonical coordinate columns (``start``, ``end``, ``duration``)
  remain.
* A single string OR a tuple of strings — restrict to the named
  property columns.  The single-string form is a shorthand for a
  one-element tuple.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from timetoalign.loader.tabular import CsvLoader


class _MixedLoader(CsvLoader):
    """Loader that emits two column-spec fields and leaves two raw extras.

    Source CSV columns: ``id,start,end,pitch,label,scribbled,page``.
    ``pitch`` and ``label`` are claimed by ``column_specs`` → emitted as
    fields.  ``scribbled`` and ``page`` are unconsumed → property
    columns.
    """

    id_column = "id"
    start_column = "start"
    end_column = "end"
    column_specs = {"pitch": int, "label": str}


@pytest.fixture
def csv_file_mixed(tmp_path: Path) -> Path:
    path = tmp_path / "events.csv"
    path.write_text(
        "id,start,end,pitch,label,scribbled,page\n"
        "e1,0.0,1.0,60,intro,one,1\n"
        "e2,1.0,2.0,62,verse,two,2\n"
    )
    return path


class TestPropertiesArgument:

    def test_properties_true_includes_fields_and_properties(
        self, csv_file_mixed: Path
    ) -> None:
        loader = _MixedLoader.from_file(csv_file_mixed)
        events = loader.get_events(properties=True)
        cols = set(events.table.column_names)
        # Column-spec fields survive.
        assert {"pitch", "label"}.issubset(cols)
        # Raw unconsumed source columns survive too under properties=True.
        assert {"scribbled", "page"}.issubset(cols)

    def test_properties_true_includes_unconsumed_columns(
        self, csv_file_mixed: Path
    ) -> None:
        # Standalone check: unconsumed source columns are propagated
        # into the EventData table at all (the bug this test targets
        # was that they were silently dropped before reaching the
        # filter).
        loader = _MixedLoader.from_file(csv_file_mixed)
        events = loader.get_events(properties=True)
        cols = set(events.table.column_names)
        assert "scribbled" in cols
        assert "page" in cols
        # The raw data round-trips intact.
        assert events.table["scribbled"].to_pylist() == ["one", "two"]
        assert events.table["page"].to_pylist() == [1, 2]

    def test_properties_false_drops_only_raw_properties(
        self, csv_file_mixed: Path
    ) -> None:
        # Per the spec, ``properties=False`` MUST preserve column-spec
        # emissions (they are fields, not properties) and MUST drop
        # only the unconsumed source columns.
        loader = _MixedLoader.from_file(csv_file_mixed)
        events = loader.get_events(properties=False)
        cols = set(events.table.column_names)
        # Column-spec fields are preserved.
        assert {"pitch", "label"}.issubset(cols)
        # Core + canonical columns are preserved.
        assert {"id", "event_type", "temporal_type", "start"}.issubset(cols)
        # Raw unconsumed source columns are dropped.
        assert {"scribbled", "page"}.isdisjoint(cols)

    def test_properties_tuple_keeps_named_property(self, csv_file_mixed: Path) -> None:
        loader = _MixedLoader.from_file(csv_file_mixed)
        events = loader.get_events(properties=("scribbled",))
        cols = set(events.table.column_names)
        # Column-spec fields always survive regardless of properties=.
        assert {"pitch", "label"}.issubset(cols)
        # Only the named raw property survives.
        assert "scribbled" in cols
        assert "page" not in cols

    def test_properties_string_keeps_only_named_unconsumed(
        self, csv_file_mixed: Path
    ) -> None:
        # The single-string shorthand normalises to a one-element tuple.
        loader = _MixedLoader.from_file(csv_file_mixed)
        events_str = loader.get_events(properties="page")
        cols_str = set(events_str.table.column_names)
        # Column-spec fields survive.
        assert {"pitch", "label"}.issubset(cols_str)
        # Only the named raw property survives.
        assert "page" in cols_str
        assert "scribbled" not in cols_str
        # And the result agrees with the explicit one-tuple form.
        events_tuple = loader.get_events(properties=("page",))
        assert set(events_tuple.table.column_names) == cols_str
