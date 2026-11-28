"""Tests for XmlLoader: Configurable XML normaliser.

Test specimens:
    1. StringQuartetEEP_I_Normal.xml — RepoVizz manifest with nested Audio/Signal elements
    2. Synthetic XML data for edge cases

See Also:
    timetoalign.loader.format.xml.XmlLoader
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from timetoalign.loader.format.xml import XmlLoader
from timetoalign.storage.store import DictStore
from timetoalign.testdata import ensure_data

# region Test data paths

# RepoVizz XML manifest of the Beethoven Op. 18 No. 4 iv multimodal specimen
XML_PATH = (
    ensure_data("score")
    / "beethoven_op18-4iv_multimodal"
    / "StringQuartetEEP_I_Normal"
    / "StringQuartetEEP_I_Normal.xml"
)

# endregion


# region Synthetic XML for unit tests

SIMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root id="ROOT0">
    <item id="item1" name="First" value="10"/>
    <item id="item2" name="Second" value="20"/>
    <item id="item3" name="Third" value="30"/>
</root>
"""

NESTED_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <group name="GroupA">
        <item id="a1" value="100"/>
        <item id="a2" value="200"/>
    </group>
    <group name="GroupB">
        <item id="b1" value="300"/>
    </group>
</root>
"""

ATTRIBUTES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <signal id="sig1" samplerate="44100" numsamples="1000" filename="audio.wav"/>
    <signal id="sig2" samplerate="22050" numsamples="500" filename="voice.wav"/>
</root>
"""

TEXT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <description id="d1">This is some text content</description>
    <description id="d2">Another description</description>
</root>
"""

MIXED_TYPES_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <element id="e1" int_val="42" float_val="3.14" bool_val="true" str_val="hello"/>
    <element id="e2" int_val="-5" float_val="2.718" bool_val="false" str_val="world"/>
</root>
"""

SINGLE_ELEMENT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<root>
    <item id="only_one" value="42"/>
</root>
"""

# endregion


# region Fixtures


@pytest.fixture
def simple_loader() -> XmlLoader:
    """XmlLoader with simple synthetic XML."""
    loader = XmlLoader(principal_tags=["item"])
    loader.load_string(SIMPLE_XML)
    return loader


@pytest.fixture
def nested_loader() -> XmlLoader:
    """XmlLoader with nested synthetic XML."""
    loader = XmlLoader(principal_tags=["item"])
    loader.load_string(NESTED_XML)
    return loader


@pytest.fixture
def auto_detect_loader() -> XmlLoader:
    """XmlLoader with auto-detection on nested XML."""
    loader = XmlLoader()
    loader.load_string(NESTED_XML)
    return loader


@pytest.fixture
def repovizz_loader() -> XmlLoader:
    """XmlLoader for RepoVizz manifest."""
    loader = XmlLoader(principal_tags=["Audio", "Signal", "Annotation"])
    loader.load(XML_PATH)
    return loader


@pytest.fixture
def repovizz_auto_loader() -> XmlLoader:
    """XmlLoader with auto-detection on RepoVizz manifest."""
    loader = XmlLoader()
    loader.load(XML_PATH)
    return loader


# endregion


# region Simple XML tests


class TestSimpleXml:
    """Simple XML: principal_tags=['item'] -> 3 rows."""

    def test_item_row_count(self, simple_loader: XmlLoader) -> None:
        table = simple_loader.get_table("item")
        assert table.num_rows == 3

    def test_item_is_pyarrow_table(self, simple_loader: XmlLoader) -> None:
        table = simple_loader.get_table("item")
        assert isinstance(table, pa.Table)

    def test_item_keys(self, simple_loader: XmlLoader) -> None:
        assert simple_loader.keys() == ["item"]

    def test_loader_len(self, simple_loader: XmlLoader) -> None:
        assert len(simple_loader) == 1

    def test_attributes_become_columns(self, simple_loader: XmlLoader) -> None:
        table = simple_loader.get_table("item")
        col_names = table.column_names
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names

    def test_attribute_values(self, simple_loader: XmlLoader) -> None:
        table = simple_loader.get_table("item")
        ids = table.column("id").to_pylist()
        assert ids == ["item1", "item2", "item3"]

    def test_value_parsing(self, simple_loader: XmlLoader) -> None:
        """Values should be parsed to appropriate types."""
        table = simple_loader.get_table("item")
        values = table.column("value").to_pylist()
        assert values == [10, 20, 30]


# endregion


# region Nested XML tests


class TestNestedXml:
    """Nested XML: item elements inside group elements."""

    def test_item_row_count(self, nested_loader: XmlLoader) -> None:
        table = nested_loader.get_table("item")
        assert table.num_rows == 3

    def test_parent_attributes_propagated(self, nested_loader: XmlLoader) -> None:
        """Parent group's 'name' should be propagated."""
        table = nested_loader.get_table("item")
        col_names = table.column_names
        # Should have parent_group_name
        assert "parent_group_name" in col_names

    def test_parent_values(self, nested_loader: XmlLoader) -> None:
        """Items should know which group they came from."""
        table = nested_loader.get_table("item")
        parent_names = table.column("parent_group_name").to_pylist()
        assert parent_names == ["GroupA", "GroupA", "GroupB"]


# endregion


# region Auto-detection tests


class TestAutoDetection:
    """XML auto-detection of principal tags."""

    def test_auto_detects_group_and_item(self, auto_detect_loader: XmlLoader) -> None:
        keys = auto_detect_loader.keys()
        # Both 'group' (2) and 'item' (3) appear at least twice
        assert "group" in keys
        assert "item" in keys

    def test_auto_detect_group_count(self, auto_detect_loader: XmlLoader) -> None:
        table = auto_detect_loader.get_table("group")
        assert table.num_rows == 2

    def test_auto_detect_item_count(self, auto_detect_loader: XmlLoader) -> None:
        table = auto_detect_loader.get_table("item")
        assert table.num_rows == 3


# endregion


# region Type parsing tests


class TestTypeParsing:
    """Test value type parsing (int, float, bool, string)."""

    def test_mixed_types(self) -> None:
        loader = XmlLoader(principal_tags=["element"])
        loader.load_string(MIXED_TYPES_XML)
        table = loader.get_table("element")

        int_vals = table.column("int_val").to_pylist()
        assert int_vals == [42, -5]

        float_vals = table.column("float_val").to_pylist()
        assert float_vals == [3.14, 2.718]

        bool_vals = table.column("bool_val").to_pylist()
        assert bool_vals == [True, False]

        str_vals = table.column("str_val").to_pylist()
        assert str_vals == ["hello", "world"]


# endregion


# region Text content tests


class TestTextContent:
    """Test element text content extraction."""

    def test_text_content_extracted(self) -> None:
        loader = XmlLoader(principal_tags=["description"], include_text=True)
        loader.load_string(TEXT_XML)
        table = loader.get_table("description")

        assert "_text" in table.column_names
        texts = table.column("_text").to_pylist()
        assert texts == ["This is some text content", "Another description"]

    def test_text_content_disabled(self) -> None:
        loader = XmlLoader(principal_tags=["description"], include_text=False)
        loader.load_string(TEXT_XML)
        table = loader.get_table("description")

        assert "_text" not in table.column_names


# endregion


# region RepoVizz XML tests


class TestRepoVizzXml:
    """RepoVizz XML manifest: Audio, Signal, Annotation elements."""

    def test_file_exists(self) -> None:
        assert XML_PATH.is_file()

    def test_audio_elements_count(self, repovizz_loader: XmlLoader) -> None:
        """Should find 6 Audio elements (2 ambient + 4 pickups)."""
        table = repovizz_loader.get_table("Audio")
        assert table.num_rows == 6

    def test_signal_elements_count(self, repovizz_loader: XmlLoader) -> None:
        """Should find 318 Signal elements (descriptors + MoCap)."""
        table = repovizz_loader.get_table("Signal")
        assert table.num_rows == 318  # the file holds 318 Signal elements

    def test_annotation_elements_count(self, repovizz_loader: XmlLoader) -> None:
        """Should find 4 Annotation elements (.notes files)."""
        table = repovizz_loader.get_table("Annotation")
        assert table.num_rows == 4

    def test_audio_has_expected_columns(self, repovizz_loader: XmlLoader) -> None:
        table = repovizz_loader.get_table("Audio")
        col_names = table.column_names
        assert "Filename" in col_names
        assert "SampleRate" in col_names
        assert "NumSamples" in col_names

    def test_signal_has_expected_columns(self, repovizz_loader: XmlLoader) -> None:
        table = repovizz_loader.get_table("Signal")
        col_names = table.column_names
        assert "samplerate" in col_names or "SampleRate" in col_names
        assert "numsamples" in col_names or "NumSamples" in col_names

    def test_audio_sample_rate_values(self, repovizz_loader: XmlLoader) -> None:
        """Audio elements should have SampleRate=44100."""
        table = repovizz_loader.get_table("Audio")
        sample_rates = table.column("SampleRate").to_pylist()
        assert all(sr == 44100 for sr in sample_rates)

    def test_audio_numsamples_values(self, repovizz_loader: XmlLoader) -> None:
        """Audio elements should have NumSamples=11753638."""
        table = repovizz_loader.get_table("Audio")
        num_samples = table.column("NumSamples").to_pylist()
        assert all(ns == 11753638 for ns in num_samples)


# endregion


# region Store integration tests


class TestXmlLoaderStore:
    """XmlLoader.store returns a DictStore wrapping the normalised tables."""

    def test_store_is_dict_store(self, simple_loader: XmlLoader) -> None:
        assert isinstance(simple_loader.store, DictStore)

    def test_store_keys_match_loader_keys(self, simple_loader: XmlLoader) -> None:
        assert set(simple_loader.store.keys()) == set(simple_loader.keys())

    def test_store_len_matches_loader_len(self, simple_loader: XmlLoader) -> None:
        assert len(simple_loader.store) == len(simple_loader)

    def test_store_getitem_returns_event_data(self, simple_loader: XmlLoader) -> None:
        """Store values should be EventData wrapping pa.Table."""
        from timetoalign.storage.events import EventData

        for key in simple_loader.keys():
            data = simple_loader.store[key]
            assert isinstance(data, EventData)

    def test_store_table_matches_get_table(self, simple_loader: XmlLoader) -> None:
        """EventData._table should be the same pa.Table as get_table()."""
        for key in simple_loader.keys():
            store_table = simple_loader.store[key]._table
            direct_table = simple_loader.get_table(key)
            assert store_table.equals(direct_table)

    def test_store_iteration(self, auto_detect_loader: XmlLoader) -> None:
        """Should be able to iterate over store items."""
        count = 0
        for name, data in auto_detect_loader.store.items():
            assert isinstance(name, str)
            count += 1
        # NESTED_XML auto-detects exactly two principal tags: 'group' and 'item'.
        assert count == 2

    def test_store_contains(self, simple_loader: XmlLoader) -> None:
        assert "item" in simple_loader.store
        assert "nonexistent" not in simple_loader.store

    def test_tables_property_unwraps(self, simple_loader: XmlLoader) -> None:
        """tables property should return raw pa.Table objects."""
        tables = simple_loader.tables
        for key, table in tables.items():
            assert isinstance(table, pa.Table)


# endregion


# region Edge cases


class TestXmlLoaderEdgeCases:
    """Edge cases and error handling."""

    def test_get_table_missing_tag_raises(self, simple_loader: XmlLoader) -> None:
        with pytest.raises(KeyError, match="No table for tag"):
            simple_loader.get_table("nonexistent")

    def test_load_nonexistent_file_raises(self) -> None:
        loader = XmlLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.xml")

    def test_clear_resets_state(self, simple_loader: XmlLoader) -> None:
        assert len(simple_loader) == 1  # one principal tag, one table
        simple_loader.clear()
        assert len(simple_loader) == 0
        assert simple_loader.raw_root is None

    def test_clear_resets_store(self, simple_loader: XmlLoader) -> None:
        assert len(simple_loader.store) == 1
        simple_loader.clear()
        assert len(simple_loader.store) == 0

    def test_load_string_api(self) -> None:
        """load_string() should work with an XML string."""
        loader = XmlLoader(principal_tags=["item"])
        loader.load_string(SIMPLE_XML)
        table = loader.get_table("item")
        assert table.num_rows == 3

    def test_load_string_populates_store(self) -> None:
        """load_string() should populate the store."""
        loader = XmlLoader(principal_tags=["item"])
        loader.load_string(SIMPLE_XML)
        assert "item" in loader.store
        assert len(loader.store) == 1

    def test_repr(self, simple_loader: XmlLoader) -> None:
        r = repr(simple_loader)
        assert "XmlLoader(" in r

    def test_file_metadata_extracted(self, simple_loader: XmlLoader) -> None:
        """Root element attributes should be captured in file_metadata."""
        meta = simple_loader.file_metadata
        assert isinstance(meta, dict)
        assert "id" in meta
        assert meta["id"] == "ROOT0"

    def test_single_element_not_detected(self) -> None:
        """Single elements (count < 2) should not be auto-detected."""
        loader = XmlLoader()
        loader.load_string(SINGLE_ELEMENT_XML)
        # 'item' appears only once, so should not be auto-detected
        keys = loader.keys()
        assert "item" not in keys

    def test_single_element_with_explicit_tag(self) -> None:
        """Single elements can still be loaded with explicit principal_tags."""
        loader = XmlLoader(principal_tags=["item"])
        loader.load_string(SINGLE_ELEMENT_XML)
        table = loader.get_table("item")
        assert table.num_rows == 1

    def test_propagate_ancestors_disabled(self) -> None:
        """When propagate_ancestors=False, no parent attrs are added."""
        loader = XmlLoader(principal_tags=["item"], propagate_ancestors=False)
        loader.load_string(NESTED_XML)
        table = loader.get_table("item")
        col_names = table.column_names
        # Should NOT have parent_group_name
        assert "parent_group_name" not in col_names


# endregion


# region load_element tests


class TestLoadElement:
    """Tests for load_element() method."""

    def test_load_element_works(self) -> None:
        """load_element() should work with an Element tree."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(SIMPLE_XML)
        loader = XmlLoader(principal_tags=["item"])
        loader.load_element(root)
        table = loader.get_table("item")
        assert table.num_rows == 3

    def test_raw_root_set(self) -> None:
        """raw_root should be set after load_element()."""
        import xml.etree.ElementTree as ET

        root = ET.fromstring(SIMPLE_XML)
        loader = XmlLoader()
        loader.load_element(root)
        assert loader.raw_root is not None
        assert loader.raw_root.tag == "root"


# endregion


def test_source_recorded_in_metadata() -> None:
    """A filesystem source is recorded as its path in ``metadata["sources"]``."""
    loader = XmlLoader.from_file(XML_PATH)
    (entry,) = loader.metadata["sources"]
    assert sorted(entry) == ["loaded_at", "path"]
    assert entry["path"] == str(XML_PATH)
    assert loader.metadata["source_count"] == 1


LOCATED_XML = (
    '<root><item xml:id="i1" v="1"/><item v="2"/><group><item v="3"/></group></root>'
)
ITEM_LOCATORS = ["i1", "/root/item[2]", "/root/group/item"]


class TestLocators:
    """Each row's locator is its element's xml:id, else the element's path."""

    @pytest.mark.parametrize("propagate", [True, False])
    def test_the_stated_id_else_the_path(self, tmp_path: Path, propagate: bool) -> None:
        path = tmp_path / "located.xml"
        path.write_text(LOCATED_XML, encoding="utf-8")
        loader = XmlLoader(principal_tags=["item"], propagate_ancestors=propagate)
        table = loader.load(path).get_table("item")
        assert table.column("locator").to_pylist() == ITEM_LOCATORS
        (extraction,) = loader.extractions
        assert table.column("extraction_id").to_pylist() == [extraction.id] * 3

    def test_a_string_names_no_extraction(self) -> None:
        loader = XmlLoader(principal_tags=["item"]).load_string(LOCATED_XML)
        table = loader.get_table("item")
        assert table.column("locator").to_pylist() == ITEM_LOCATORS
        assert "extraction_id" not in table.column_names
