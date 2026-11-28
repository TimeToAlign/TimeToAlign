"""Tests for JsonLoader: Configurable JSON normaliser.

Test specimens:
    1. dj_studio_data.json — DJ Studio export with nested audio/cueData/hotCuePoints
    2. Wagner_WWV086B_140.json — Audiolabs OMR bounding boxes (auto-detect 3 tables)
    3. all_annotations.json — COCO-style annotations with foreign-key resolution

See Also:
    timetoalign.loader.format.json.JsonLoader
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from timetoalign.loader.format.json import JsonLoader
from timetoalign.storage.store import DictStore
from timetoalign.testdata import ensure_data

# region Test data paths

AUDIO_DIR, OMR_DIR = ensure_data("audio", "audiolabs_omr")

DJ_STUDIO_PATH = AUDIO_DIR / "hard_techno" / "dj_studio_data.json"
WAGNER_PATH = OMR_DIR / "Wagner_WWV086B-3" / "json" / "Wagner_WWV086B_140.json"
ANNOTATIONS_PATH = OMR_DIR / "all_annotations.json"

# endregion


# region Fixtures


@pytest.fixture
def dj_audio_loader() -> JsonLoader:
    """JsonLoader configured for DJ Studio 'audio' principal key."""
    loader = JsonLoader(principal_keys=["audio"])
    loader.load(DJ_STUDIO_PATH)
    return loader


@pytest.fixture
def dj_hotcue_loader() -> JsonLoader:
    """JsonLoader configured for DJ Studio 'hotCuePoints' principal key."""
    loader = JsonLoader(principal_keys=["hotCuePoints"])
    loader.load(DJ_STUDIO_PATH)
    return loader


@pytest.fixture
def wagner_loader() -> JsonLoader:
    """JsonLoader with auto-detection on Wagner OMR bounding boxes."""
    loader = JsonLoader()
    loader.load(WAGNER_PATH)
    return loader


@pytest.fixture
def annotations_loader() -> JsonLoader:
    """JsonLoader for COCO-style annotations with lookup resolution."""
    loader = JsonLoader(principal_keys=["annotations"])
    loader.load(ANNOTATIONS_PATH)
    return loader


# endregion


# region DJ Studio: audio principal key


class TestDjStudioAudio:
    """DJ Studio JSON: principal_keys=['audio'] -> 3 rows."""

    def test_file_exists(self) -> None:
        assert DJ_STUDIO_PATH.exists(), f"Test data not found: {DJ_STUDIO_PATH}"

    def test_audio_row_count(self, dj_audio_loader: JsonLoader) -> None:
        table = dj_audio_loader.get_table("audio")
        assert table.num_rows == 3

    def test_audio_is_pyarrow_table(self, dj_audio_loader: JsonLoader) -> None:
        table = dj_audio_loader.get_table("audio")
        assert isinstance(table, pa.Table)

    def test_audio_keys(self, dj_audio_loader: JsonLoader) -> None:
        assert dj_audio_loader.keys() == ["audio"]

    def test_loader_len(self, dj_audio_loader: JsonLoader) -> None:
        assert len(dj_audio_loader) == 1

    def test_sources_tracked(self, dj_audio_loader: JsonLoader) -> None:
        assert len(dj_audio_loader.sources) == 1
        assert dj_audio_loader.sources[0] == DJ_STUDIO_PATH


# endregion


# region DJ Studio: hotCuePoints (nested search)


class TestDjStudioHotCuePoints:
    """DJ Studio JSON: principal_keys=['hotCuePoints'] -> 24 rows."""

    def test_hotcue_row_count(self, dj_hotcue_loader: JsonLoader) -> None:
        table = dj_hotcue_loader.get_table("hotCuePoints")
        assert table.num_rows == 24

    def test_hotcue_is_pyarrow_table(self, dj_hotcue_loader: JsonLoader) -> None:
        table = dj_hotcue_loader.get_table("hotCuePoints")
        assert isinstance(table, pa.Table)

    def test_hotcue_has_parent_context(self, dj_hotcue_loader: JsonLoader) -> None:
        """Parent scalar fields (from audio items) should be propagated."""
        table = dj_hotcue_loader.get_table("hotCuePoints")
        col_names = table.column_names
        # Parent audio items have a "name" field that should propagate
        assert (
            "name" in col_names
        ), f"Expected 'name' from parent audio item in columns: {col_names}"


# endregion


# region Wagner OMR: auto-detection


class TestWagnerAutoDetect:
    """Wagner OMR: auto-detect 3 tables with exact row counts."""

    def test_file_exists(self) -> None:
        assert WAGNER_PATH.exists(), f"Test data not found: {WAGNER_PATH}"

    def test_auto_detects_three_tables(self, wagner_loader: JsonLoader) -> None:
        keys = sorted(wagner_loader.keys())
        assert len(keys) == 3

    def test_system_measures_count(self, wagner_loader: JsonLoader) -> None:
        table = wagner_loader.get_table("system_measures")
        assert table.num_rows == 15

    def test_stave_measures_count(self, wagner_loader: JsonLoader) -> None:
        table = wagner_loader.get_table("stave_measures")
        assert table.num_rows == 30

    def test_staves_count(self, wagner_loader: JsonLoader) -> None:
        table = wagner_loader.get_table("staves")
        assert table.num_rows == 10

    def test_all_tables_are_pyarrow(self, wagner_loader: JsonLoader) -> None:
        for key in wagner_loader.keys():
            assert isinstance(wagner_loader.get_table(key), pa.Table)

    def test_table_key_names(self, wagner_loader: JsonLoader) -> None:
        expected = {"stave_measures", "staves", "system_measures"}
        assert set(wagner_loader.keys()) == expected


# endregion


# region COCO Annotations: foreign-key resolution


class TestCocoAnnotations:
    """COCO all_annotations.json: annotations with lookup resolution."""

    def test_file_exists(self) -> None:
        assert ANNOTATIONS_PATH.exists(), f"Test data not found: {ANNOTATIONS_PATH}"

    def test_annotations_row_count(self, annotations_loader: JsonLoader) -> None:
        table = annotations_loader.get_table("annotations")
        assert table.num_rows == 6345

    def test_image_id_resolved(self, annotations_loader: JsonLoader) -> None:
        """image_id field should be resolved -> image.file_name exists."""
        table = annotations_loader.get_table("annotations")
        col_names = table.column_names
        assert (
            "image.file_name" in col_names
        ), f"Expected 'image.file_name' from lookup resolution in: {col_names}"

    def test_category_id_resolved(self, annotations_loader: JsonLoader) -> None:
        """category_id field should be resolved -> category.name exists."""
        table = annotations_loader.get_table("annotations")
        col_names = table.column_names
        assert (
            "category.name" in col_names
        ), f"Expected 'category.name' from lookup resolution in: {col_names}"

    def test_image_width_resolved(self, annotations_loader: JsonLoader) -> None:
        """image.width should be present from lookup resolution."""
        table = annotations_loader.get_table("annotations")
        assert "image.width" in table.column_names

    def test_image_height_resolved(self, annotations_loader: JsonLoader) -> None:
        """image.height should be present from lookup resolution."""
        table = annotations_loader.get_table("annotations")
        assert "image.height" in table.column_names


# endregion


# region Store integration


class TestJsonLoaderStore:
    """JsonLoader.store returns a DictStore wrapping the normalised tables."""

    def test_store_is_dict_store(self, wagner_loader: JsonLoader) -> None:
        assert isinstance(wagner_loader.store, DictStore)

    def test_store_keys_match_loader_keys(self, wagner_loader: JsonLoader) -> None:
        assert set(wagner_loader.store.keys()) == set(wagner_loader.keys())

    def test_store_len_matches_loader_len(self, wagner_loader: JsonLoader) -> None:
        assert len(wagner_loader.store) == len(wagner_loader)

    def test_store_getitem_returns_event_data(self, wagner_loader: JsonLoader) -> None:
        """Store values should be EventData wrapping pa.Table."""
        from timetoalign.storage.events import EventData

        for key in wagner_loader.keys():
            data = wagner_loader.store[key]
            assert isinstance(data, EventData)

    def test_store_table_matches_get_table(self, wagner_loader: JsonLoader) -> None:
        """EventData._table should be the same pa.Table as get_table()."""
        for key in wagner_loader.keys():
            store_table = wagner_loader.store[key]._table
            direct_table = wagner_loader.get_table(key)
            assert store_table.equals(direct_table)

    def test_store_iteration(self, wagner_loader: JsonLoader) -> None:
        """Should be able to iterate over store items."""
        count = 0
        for name, data in wagner_loader.store.items():
            assert isinstance(name, str)
            count += 1
        assert count == 3

    def test_store_contains(self, wagner_loader: JsonLoader) -> None:
        assert "staves" in wagner_loader.store
        assert "nonexistent" not in wagner_loader.store

    def test_tables_property_unwraps(self, wagner_loader: JsonLoader) -> None:
        """tables property should return raw pa.Table objects."""
        tables = wagner_loader.tables
        for key, table in tables.items():
            assert isinstance(table, pa.Table)


# endregion


# region Edge cases


class TestJsonLoaderEdgeCases:
    """Edge cases and error handling."""

    def test_get_table_missing_key_raises(self, wagner_loader: JsonLoader) -> None:
        with pytest.raises(KeyError, match="No table for key"):
            wagner_loader.get_table("nonexistent")

    def test_load_nonexistent_file_raises(self) -> None:
        loader = JsonLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.json")

    def test_clear_resets_state(self, wagner_loader: JsonLoader) -> None:
        # The file's three list-valued keys are three tables.
        assert len(wagner_loader) == 3
        wagner_loader.clear()
        assert len(wagner_loader) == 0
        assert wagner_loader.raw_data is None

    def test_clear_resets_store(self, wagner_loader: JsonLoader) -> None:
        assert len(wagner_loader.store) == 3
        wagner_loader.clear()
        assert len(wagner_loader.store) == 0

    def test_load_dict_api(self) -> None:
        """load_dict() should work with an already-parsed dict."""
        data = {
            "items": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
            ]
        }
        loader = JsonLoader()
        loader.load_dict(data)
        table = loader.get_table("items")
        assert table.num_rows == 2

    def test_load_dict_populates_store(self) -> None:
        """load_dict() should populate the store."""
        data = {
            "items": [
                {"id": 1, "name": "a"},
                {"id": 2, "name": "b"},
            ]
        }
        loader = JsonLoader()
        loader.load_dict(data)
        assert "items" in loader.store
        assert len(loader.store) == 1

    def test_repr(self, wagner_loader: JsonLoader) -> None:
        r = repr(wagner_loader)
        assert "JsonLoader(" in r

    def test_file_metadata_extracted(self, wagner_loader: JsonLoader) -> None:
        """Scalar top-level metadata should be captured."""
        meta = wagner_loader.file_metadata
        assert isinstance(meta, dict)

    def test_resolve_lookups_disabled(self) -> None:
        """When resolve_lookups=False, no *_id fields are resolved."""
        loader = JsonLoader(principal_keys=["annotations"], resolve_lookups=False)
        loader.load(ANNOTATIONS_PATH)
        table = loader.get_table("annotations")
        # Should NOT have resolved fields
        assert "image.file_name" not in table.column_names


# endregion


def test_source_recorded_in_metadata() -> None:
    """A filesystem source is recorded as its path in ``metadata["sources"]``."""
    loader = JsonLoader.from_file(DJ_STUDIO_PATH)
    (entry,) = loader.metadata["sources"]
    assert sorted(entry) == ["loaded_at", "path"]
    assert entry["path"] == str(DJ_STUDIO_PATH)
    assert loader.metadata["source_count"] == 1


LOCATED = {
    "title": "x",
    "audio": [
        {
            "id": "a1",
            "name": "a",
            "cueData": {"hotCuePoints": [{"id": "c1", "t": 1}, {"t": 2}]},
        },
        {"name": "b", "cueData": {"hotCuePoints": [{"t": 3}]}},
    ],
    "a/b": [{"x": 1}],
}
CUE_LOCATORS = [
    "/audio/0/cueData/hotCuePoints/0 key:c1",
    "/audio/0/cueData/hotCuePoints/1",
    "/audio/1/cueData/hotCuePoints/0",
]


class TestLocators:
    """A row's locator: the JSON Pointer of its object, then the object's own ``id``."""

    @pytest.fixture
    def located(self, tmp_path: Path) -> Path:
        path = tmp_path / "located.json"
        path.write_text(json.dumps(LOCATED), encoding="utf-8")
        return path

    def test_a_top_level_key(self, located: Path) -> None:
        loader = JsonLoader(principal_keys=["audio"]).load(located)
        table = loader.get_table("audio")
        assert table.column("locator").to_pylist() == ["/audio/0 key:a1", "/audio/1"]
        (extraction,) = loader.extractions
        assert extraction.source == located.as_posix()
        assert table.column("extraction_id").to_pylist() == [extraction.id] * 2

    @pytest.mark.parametrize("key", ["hotCuePoints", "audio.cueData.hotCuePoints"])
    def test_a_nested_key(self, located: Path, key: str) -> None:
        table = JsonLoader(principal_keys=[key]).load(located).get_table(key)
        assert table.column("locator").to_pylist() == CUE_LOCATORS

    def test_a_slash_is_escaped(self, located: Path) -> None:
        table = JsonLoader(principal_keys=["a/b"]).load(located).get_table("a/b")
        assert table.column("locator").to_pylist() == ["/a~1b/0"]

    def test_a_dict_names_no_extraction(self) -> None:
        loader = JsonLoader(principal_keys=["audio"]).load_dict(LOCATED)
        table = loader.get_table("audio")
        assert table.column("locator").to_pylist() == ["/audio/0 key:a1", "/audio/1"]
        assert "extraction_id" not in table.column_names
        assert loader.extractions == []


NAMED_LIKE_THE_REGISTRY = {
    "events": [{"locator": "bar 3", "extraction_id": "run 1", "t": 1}, {"t": 2}]
}


class TestSourceFieldsNamedLikeTheRegistry:
    """A document's own ``locator`` and ``extraction_id`` are kept beside the registry's."""

    def test_read_from_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "named.json"
        path.write_text(json.dumps(NAMED_LIKE_THE_REGISTRY), encoding="utf-8")
        loader = JsonLoader(principal_keys=["events"]).load(path)
        table = loader.get_table("events")
        (extraction,) = loader.extractions
        assert table.column("source_locator").to_pylist() == ["bar 3", None]
        assert table.column("source_extraction_id").to_pylist() == ["run 1", None]
        assert table.column("locator").to_pylist() == ["/events/0", "/events/1"]
        assert table.column("extraction_id").to_pylist() == [extraction.id] * 2

    def test_read_from_a_dict(self) -> None:
        loader = JsonLoader(principal_keys=["events"]).load_dict(
            NAMED_LIKE_THE_REGISTRY
        )
        table = loader.get_table("events")
        assert table.column("source_extraction_id").to_pylist() == ["run 1", None]
        assert "extraction_id" not in table.column_names

    def test_a_name_held_already(self) -> None:
        document = {"events": [{"locator": "a", "source_locator": "b"}]}
        table = (
            JsonLoader(principal_keys=["events"])
            .load_dict(document)
            .get_table("events")
        )
        assert table.column("source_locator").to_pylist() == ["b"]
        assert table.column("source_source_locator").to_pylist() == ["a"]
        assert table.column("locator").to_pylist() == ["/events/0"]
