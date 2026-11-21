"""Tests for base loader classes: ManifestLoader, AlignmentLoader, AlignmentStore."""

from __future__ import annotations

from pathlib import Path

from timetoalign.core import TimeUnit
from timetoalign.loader import (
    AlignmentLoader,
    EventLoader,
    Loader,
    ManifestData,
    ManifestLoader,
)
from timetoalign.storage import AlignmentStore, EventData, SingleStore

# region ManifestData Tests


class TestManifestData:
    """Tests for ManifestData."""

    def test_create_basic(self):
        """Test basic ManifestData creation."""
        manifest = ManifestData(
            dimensions={"width": 1920, "height": 1080},
            metadata={"format": "mp4"},
            source_type="video",
        )

        assert manifest.dimensions["width"] == 1920
        assert manifest.dimensions["height"] == 1080
        assert manifest.source_type == "video"

    def test_repr(self):
        """Test string representation."""
        manifest = ManifestData(
            dimensions={"duration": 60.0},
            source_type="audio",
        )

        repr_str = repr(manifest)
        assert "audio" in repr_str
        assert "duration" in repr_str


# endregion


# region ManifestLoader Tests


def test_loader_hierarchy_separates_event_pipeline() -> None:
    """Only EventLoader descendants expose event assembly APIs."""
    assert issubclass(EventLoader, Loader)
    assert issubclass(ManifestLoader, Loader)
    assert issubclass(AlignmentLoader, Loader)
    assert hasattr(EventLoader, "get_events")
    for loader_class in (ManifestLoader, AlignmentLoader):
        assert not hasattr(loader_class, "events")
        assert not hasattr(loader_class, "get_events")


class TestManifestLoader:
    """Tests for ManifestLoader ABC."""

    def test_concrete_implementation(self, tmp_path):
        """Test a concrete ManifestLoader implementation."""

        class TestManifestLoader(ManifestLoader):
            def _load_source(self, source: Path) -> ManifestData:
                return ManifestData(
                    dimensions={"lines": 100},
                    metadata={"encoding": "utf-8"},
                    source_type="text",
                )

        path = tmp_path / "test.txt"
        path.write_text("test")

        loader = TestManifestLoader()
        loader.load(path)

        assert len(loader) == 1
        assert loader.manifest is not None
        assert loader.manifest.dimensions["lines"] == 100

    def test_multiple_sources(self, tmp_path):
        """Test loading multiple sources."""

        class CountingLoader(ManifestLoader):
            def __init__(self):
                super().__init__()
                self._count = 0

            def _load_source(self, source: Path) -> ManifestData:
                self._count += 1
                return ManifestData(
                    dimensions={"index": self._count},
                    source_type="test",
                )

        paths = []
        for i in range(3):
            p = tmp_path / f"test_{i}.txt"
            p.write_text(f"test {i}")
            paths.append(p)

        loader = CountingLoader()
        loader.load(*paths)

        assert len(loader) == 3
        assert len(loader.manifests) == 3
        assert loader.manifests[0].dimensions["index"] == 1
        assert loader.manifests[2].dimensions["index"] == 3

    def test_clear(self, tmp_path):
        """Test clearing loaded data."""

        class SimpleLoader(ManifestLoader):
            def _load_source(self, source: Path) -> ManifestData:
                return ManifestData(source_type="test")

        path = tmp_path / "test.txt"
        path.write_text("test")

        loader = SimpleLoader()
        loader.load(path)
        assert len(loader) == 1

        loader.clear()
        assert len(loader) == 0
        assert loader.manifest is None


# endregion


# region AlignmentStore Tests


class TestAlignmentStore:
    """Tests for AlignmentStore."""

    def test_empty(self):
        """Test creating empty AlignmentStore."""
        store = AlignmentStore.empty()

        assert store.event_count == 0
        assert store.cmap_count == 0

    def test_with_events(self):
        """Test AlignmentStore with events."""
        events = EventData.from_dicts(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "start": 0.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "start": 1.0,
                },
            ],
            unit=TimeUnit.seconds,
        )

        store = AlignmentStore(
            events=SingleStore(events, name="beats"),
            cmaps=[],
        )

        assert store.event_count == 2

    def test_summary(self):
        """Test summary generation."""
        store = AlignmentStore.empty()
        summary = store.summary()

        assert summary == {
            "tables": {
                "events": {
                    "unit": "seconds",
                    "count": 0,
                    "range": (0.0, 0.0),
                }
            },
            "facts": {
                "event_count": 0,
                "cmap_count": 0,
                "domains": ["physical"],
                "store_names": ["events"],
            },
        }


# endregion


# region AlignmentLoader Tests


class TestAlignmentLoader:
    """Tests for AlignmentLoader ABC."""

    def test_concrete_implementation(self, tmp_path):
        """Test a concrete AlignmentLoader implementation."""

        class TestAlignmentLoader(AlignmentLoader):
            def _load_source(self, source: Path) -> AlignmentStore:
                events = EventData.from_dicts(
                    [
                        {
                            "id": "e1",
                            "temporal_type": "instant",
                            "event_type": "Beat",
                            "start": 0.0,
                        }
                    ],
                    unit=TimeUnit.seconds,
                )
                return AlignmentStore(
                    events=SingleStore(events, name="events"),
                    cmaps=[],
                )

        path = tmp_path / "test.xml"
        path.write_text("<alignment/>")

        loader = TestAlignmentLoader()
        loader.load(path)

        assert len(loader) == 1
        assert loader.store is not None
        assert loader.store.event_count == 1

    def test_clear(self, tmp_path):
        """Test clearing loaded data."""

        class SimpleAlignmentLoader(AlignmentLoader):
            def _load_source(self, source: Path) -> AlignmentStore:
                return AlignmentStore.empty()

        path = tmp_path / "test.xml"
        path.write_text("<alignment/>")

        loader = SimpleAlignmentLoader()
        loader.load(path)
        assert loader.store is not None

        loader.clear()
        assert loader.store is None
        assert len(loader.sources) == 0


# endregion
