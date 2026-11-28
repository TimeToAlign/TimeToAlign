# Format Loader Tests

Tests for the format-level loaders in `timetoalign.loader.format`.

## Test Files

### `test_json_loader.py` (45 tests)

Tests for `JsonLoader`, the configurable JSON normaliser that produces flat
PyArrow tables from nested JSON structures.

**Test Specimens:**

| Specimen | Principal Keys | Expected Rows | Validation Focus |
|----------|---------------|---------------|------------------|
| `dj_studio_data.json` | `["audio"]` | 3 | Direct top-level array |
| `dj_studio_data.json` | `["hotCuePoints"]` | 24 | Recursive nested search, parent context propagation |
| `Wagner_WWV086B_140.json` | auto-detect | 15 + 30 + 10 (3 tables) | Auto-detection of all top-level arrays |
| `all_annotations.json` | `["annotations"]` | 6345 | Foreign-key resolution (`image_id` -> images, `category_id` -> categories) |

**Specimen Provenance:**

- `dj_studio_data.json`: DJ Studio export (audio metadata with nested cue points).
  Located at `tests/data/audio/hard_techno/dj_studio_data.json`.
- `Wagner_WWV086B_140.json`: Audiolabs OMR bounding boxes for Wagner WWV086B page 140.
  Located at `tests/data/audiolabs_omr/Wagner_WWV086B-3/json/Wagner_WWV086B_140.json`.
- `all_annotations.json`: COCO-style object detection annotations for Audiolabs OMR.
  Located at `tests/data/audiolabs_omr/all_annotations.json`.

**Key Validations:**

- Exact row counts (ZERO TOLERANCE)
- Foreign-key resolution: `image_id` resolves via `images` table, `category_id`
  resolves via `categories` table (y->ies pluralisation)
- Parent context propagation in nested search (audio item scalars appear on hotCuePoints rows)
- Auto-detection correctly identifies all top-level keys with list-of-dict values

**Locators** (`TestLocators`, a document the test writes into `tmp_path`):
`{"title": "x", "audio": [{"name": "a", "cueData": {"hotCuePoints":
[{"t": 1}, {"t": 2}]}}, {"name": "b", "cueData": {"hotCuePoints": [{"t":
3}]}}], "a/b": [{"x": 1}]}`. Each row's `locator` is the JSON Pointer of
the object it was read from, written out by hand from the document:

- principal key `audio`: `/audio/0`, `/audio/1`;
- `hotCuePoints`, found by the search, and the dotted path
  `audio.cueData.hotCuePoints` alike: `/audio/0/cueData/hotCuePoints/0`,
  `/audio/0/cueData/hotCuePoints/1`, `/audio/1/cueData/hotCuePoints/0`;
- the key `a/b` is escaped as RFC 6901 writes it: `/a~1b/0`.

Read from the file, every row names the loader's one extraction (its source
the file); read with `load_dict`, the rows hold the locators and no
`extraction_id` column, and the loader lists no extraction.

---

### `test_xml_loader.py` (49 tests)

Tests for `XmlLoader`, a generic XML-to-flat-table loader using `xml.etree.ElementTree`.

**Test Classes:**

| Class | Tests | Focus |
|-------|-------|-------|
| `TestSimpleXml` | 7 | Core functionality: row counts, attribute-to-column extraction, value parsing |
| `TestNestedXml` | 3 | Parent-attribute propagation onto child rows |
| `TestAutoDetection` | 3 | Auto-detection of principal tags (elements appearing at least twice) |
| `TestTypeParsing` | 1 | int / float / bool / string attribute value parsing |
| `TestTextContent` | 2 | Element text extraction as the `_text` field |
| `TestRepoVizzXml` | 8 | Real RepoVizz manifest: 6 Audio, 4 Annotation, 318 Signal elements (counted in the file with `ElementTree`, independently of the loader) |
| `TestXmlLoaderStore` | 8 | `store` DictStore integration and iteration |
| `TestXmlLoaderEdgeCases` | 11 | Error handling, empty/single-element XML, clear/repr, ancestor toggle |
| `TestLoadElement` | 2 | `load_element()` from an `ElementTree` |
| `TestLocators` | 3 | Each row's `locator` and extraction (below) |

**Test Specimens:**

| Specimen | Principal Tags | Validation Focus |
|----------|---------------|------------------|
| RepoVizz XML manifest | Audio, Signal, Annotation | Hierarchical catalogue parsing |
| Inline test XML | Various | Attribute extraction, text content |

**Key Validations:**

- Ancestor attribute propagation (parent attributes appear on child rows)
- Principal tag filtering (extract only specified element types)
- Auto-detection of principal tags from XML structure. `TestXmlLoaderStore.test_store_iteration`
  pins the auto-detection of `NESTED_XML` at **exactly two** principal tags
  (`group` and `item`) — an exact `count == 2`, not a `>= 2` lower bound.
- Text content extraction as `_text` field
- Error handling for malformed XML

**Locators** (`TestLocators`). The document `<root><item xml:id="i1"
v="1"/><item v="2"/><group><item v="3"/></group></root>`, written into
`tmp_path`, principal tag `item`: the locators are `i1` (the stated
`xml:id`), `/root/item[2]` and `/root/group/item` (the elements' paths,
written out by hand: `root` holds two `item`s, `group` one), in document
order, with ancestors propagated or not alike. Read from the file, every
row names the loader's one extraction; read with `load_string`, the rows
hold the locators and no `extraction_id` column.

---

## Test Results

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_json_loader.py` | 45 | All pass |
| `test_xml_loader.py` | 49 | All pass |
| **Total** | **94** | **All pass** |

All tests pass in parallel mode (`pytest-xdist -n auto`).

**Clearing.** A clear test first pins the exact number of tables it removes:
the Wagner file's three list-valued keys (`stave_measures`, `staves`,
`system_measures`) are three tables in the loader and in its store; the
synthetic `SIMPLE_XML` has one principal tag, `item`, hence one table.
