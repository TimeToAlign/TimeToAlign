"""Tests for the event pipeline on loader/base.py EventLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from timetoalign.core import NumberType, TimeUnit
from timetoalign.storage import EventData

from .conftest import DummyLoader


class TestLoaderCreation:
    """Tests for Loader creation."""

    def test_create_with_defaults(self) -> None:
        """Can create loader with default settings."""
        loader = DummyLoader()
        assert loader.unit == TimeUnit.ticks  # DummyLoader._default_unit
        # ticks are counted, so the unit's own representation is int.
        assert loader.number_type == NumberType.int
        assert len(loader) == 0

    def test_create_with_unit(self) -> None:
        """Can specify unit at creation."""
        loader = DummyLoader(unit=TimeUnit.seconds)
        assert loader.unit == TimeUnit.seconds

    def test_create_with_number_type(self) -> None:
        """Can specify number_type at creation."""
        loader = DummyLoader(unit=TimeUnit.quarters, number_type=NumberType.fraction)
        assert loader.number_type == NumberType.fraction


class TestLoaderProperties:
    """Tests for Loader properties."""

    def test_events_property(self, dummy_loader: DummyLoader) -> None:
        """events property returns EventData."""
        assert isinstance(dummy_loader.events, EventData)

    def test_sources_property_empty(self, dummy_loader: DummyLoader) -> None:
        """sources property returns empty list initially."""
        assert dummy_loader.sources == []

    def test_metadata_property(self, dummy_loader: DummyLoader) -> None:
        """metadata property returns dict."""
        meta = dummy_loader.metadata
        assert meta["loader_class"] == "DummyLoader"
        assert meta["unit"] == "ticks"
        assert meta["source_count"] == 0


class TestLoaderLoading:
    """Tests for Loader.load method."""

    def test_load_single_source(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """load() loads a single source."""
        dummy_loader.load(temp_source_file)

        assert len(dummy_loader.sources) == 1
        assert temp_source_file in dummy_loader.sources
        assert len(dummy_loader) == 2  # DummyLoader creates 2 events per file

    def test_load_multiple_sources(
        self, dummy_loader: DummyLoader, tmp_path: Path
    ) -> None:
        """load() can load multiple sources at once."""
        files = []
        for i in range(3):
            p = tmp_path / f"source_{i}.dummy"
            p.write_bytes(b"dummy")
            files.append(p)

        dummy_loader.load(*files)

        assert len(dummy_loader.sources) == 3
        assert len(dummy_loader) == 6  # 2 events per file

    def test_load_source_no_events(
        self, dummy_loader: DummyLoader, tmp_path: Path
    ) -> None:
        """load() handles sources returning no events."""

        # Subclass that returns empty events
        class EmptySourceLoader(DummyLoader):
            def _load_source(
                self, source: Path
            ) -> tuple[dict[str, Any], dict[str, list[Any]]]:
                return {"format": "empty"}, {}

        path = tmp_path / "empty.empty"
        path.write_bytes(b"")
        loader = EmptySourceLoader()
        loader.load(path)

        assert len(loader.sources) == 1
        assert len(loader) == 0

    def test_load_returns_self(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """load() returns self for chaining."""
        result = dummy_loader.load(temp_source_file)
        assert result is dummy_loader

    def test_load_missing_file_raises(self, dummy_loader: DummyLoader) -> None:
        """load() raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            dummy_loader.load(Path("/nonexistent/file.dummy"))

    def test_load_incremental(
        self, dummy_loader: DummyLoader, temp_source_file: Path, tmp_path: Path
    ) -> None:
        """Multiple load() calls accumulate events."""
        dummy_loader.load(temp_source_file)
        assert len(dummy_loader) == 2

        # Create another file
        file2 = tmp_path / "second.dummy"
        file2.write_bytes(b"dummy2")

        dummy_loader.load(file2)
        assert len(dummy_loader) == 4
        assert len(dummy_loader.sources) == 2


class TestLoaderClear:
    """Tests for Loader.clear method."""

    def test_clear(self, dummy_loader: DummyLoader, temp_source_file: Path) -> None:
        """clear() removes all sources and events."""
        dummy_loader.load(temp_source_file)
        assert len(dummy_loader) == 2

        dummy_loader.clear()
        assert len(dummy_loader) == 0
        assert len(dummy_loader.sources) == 0


class TestLoaderStats:
    """Tests for Loader stats methods."""

    def test_event_summary(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """event_summary() returns comprehensive stats."""
        dummy_loader.load(temp_source_file)
        summary = dummy_loader.event_summary()

        assert summary["count"] == 2
        assert "temporal_types" in summary
        assert "sources" in summary

    def test_count_events_by_type(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """count_events_by_type() groups by event_type."""
        dummy_loader.load(temp_source_file)
        counts = dummy_loader.count_events_by_type()

        # DummyLoader creates 1 Beat + 1 Note per file
        assert counts["Beat"] == 1
        assert counts["Note"] == 1

    def test_count_events_by_temporal_type(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """count_events_by_temporal_type() groups by temporal_type."""
        dummy_loader.load(temp_source_file)
        counts = dummy_loader.count_events_by_temporal_type()

        assert counts.get("instant", 0) == 1
        assert counts.get("interval", 0) == 1


class TestLoaderRepr:
    """Tests for Loader __repr__."""

    def test_repr_empty(self, dummy_loader: DummyLoader) -> None:
        """__repr__ for empty loader."""
        r = repr(dummy_loader)
        assert "DummyLoader" in r
        assert "sources=0" in r
        assert "events=0" in r

    def test_repr_with_data(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """__repr__ for loader with data."""
        dummy_loader.load(temp_source_file)
        r = repr(dummy_loader)

        assert "sources=1" in r
        assert "events=2" in r


class TestLoaderSerialization:
    """Tests for Loader Parquet serialization."""

    def test_to_parquet(
        self,
        dummy_loader: DummyLoader,
        temp_source_file: Path,
        temp_parquet_path: Path,
    ) -> None:
        """to_parquet() saves events."""
        dummy_loader.load(temp_source_file)
        dummy_loader.to_parquet(temp_parquet_path)

        assert temp_parquet_path.exists()

    def test_from_parquet(
        self,
        dummy_loader: DummyLoader,
        temp_source_file: Path,
        temp_parquet_path: Path,
    ) -> None:
        """from_parquet() loads events."""
        dummy_loader.load(temp_source_file)
        dummy_loader.to_parquet(temp_parquet_path)

        loaded = DummyLoader.from_parquet(temp_parquet_path)

        assert len(loaded) == 2
        assert loaded.unit == TimeUnit.ticks


class TestLoaderMetadata:
    """Tests for Loader metadata handling."""

    def test_source_metadata_recorded(
        self, dummy_loader: DummyLoader, temp_source_file: Path
    ) -> None:
        """Each source's metadata is recorded."""
        dummy_loader.load(temp_source_file)

        meta = dummy_loader.metadata
        assert meta["source_count"] == 1
        assert len(meta["sources"]) == 1

        source_meta = meta["sources"][0]
        assert "path" in source_meta
        assert "loaded_at" in source_meta
        assert "format" in source_meta


class SingleSourceDummyLoader(DummyLoader):
    """A DummyLoader that reads one source."""

    _single_source = True


class TestSingleSource:
    """One rule refuses a further source on every single-source loader."""

    @pytest.fixture
    def sources(self, tmp_path: Path) -> tuple[Path, Path]:
        first, second = tmp_path / "first.dummy", tmp_path / "second.dummy"
        first.write_bytes(b"dummy")
        second.write_bytes(b"dummy")
        return first, second

    def test_two_in_one_call_are_refused_before_parsing(
        self, sources: tuple[Path, Path]
    ) -> None:
        loader = SingleSourceDummyLoader()
        with pytest.raises(ValueError) as raised:
            loader.load(*sources)
        assert str(raised.value) == (
            f"SingleSourceDummyLoader reads one source; got 2: {sources[0]}, "
            f"{sources[1]}."
        )
        assert loader.sources == []
        assert len(loader) == 0

    def test_a_second_call_is_refused(self, sources: tuple[Path, Path]) -> None:
        first, second = sources
        loader = SingleSourceDummyLoader().load(first)
        with pytest.raises(ValueError) as raised:
            loader.load(second)
        assert str(raised.value) == (
            f"SingleSourceDummyLoader reads one source and already holds {first}; "
            f"use a new loader for {second}, or call clear() first."
        )
        assert loader.sources == [first]
        assert len(loader) == 2

    def test_clear_releases_the_loader(self, sources: tuple[Path, Path]) -> None:
        first, second = sources
        loader = SingleSourceDummyLoader().load(first)
        loader.clear()
        assert loader.load(second).sources == [second]
        assert len(loader) == 2

    def test_unflagged_loaders_read_many(self, sources: tuple[Path, Path]) -> None:
        first, second = sources
        loader = DummyLoader().load(first).load(second)
        assert loader.sources == [first, second]
        assert len(loader) == 4
