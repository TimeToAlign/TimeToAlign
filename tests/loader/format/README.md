# Format Loader Tests

Tests for the format-level loaders in `timetoalign.loader.format`.

## Test Files

### `test_json_loader.py` (48 tests)

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
`{"title": "x", "audio": [{"id": "a1", "name": "a", "cueData":
{"hotCuePoints": [{"id": "c1", "t": 1}, {"t": 2}]}}, {"name": "b",
"cueData": {"hotCuePoints": [{"t": 3}]}}], "a/b": [{"x": 1}]}`. Each row's
`locator` is the JSON Pointer of the object it was read from, followed by
` key:<id>` where that object states an `id` member of its own, written out
by hand from the document:

- principal key `audio`: `/audio/0 key:a1`, `/audio/1`;
- `hotCuePoints`, found by the search, and the dotted path
  `audio.cueData.hotCuePoints` alike: `/audio/0/cueData/hotCuePoints/0
  key:c1`, `/audio/0/cueData/hotCuePoints/1`,
  `/audio/1/cueData/hotCuePoints/0`. The second cue point states no `id`:
  the search hands it its parent's fields (`id` `a1` among them), and its
  locator still names no key, because the key is the object's own;
- the key `a/b` is escaped as RFC 6901 writes it: `/a~1b/0`.

Read from the file, every row names the loader's one extraction (its source
the file); read with `load_dict`, the rows hold the locators and no
`extraction_id` column, and the loader lists no extraction.

**The document's own `locator` and `extraction_id`** (`TestSourceFieldsNamedLikeTheRegistry`).
The two names are the registry's, so a document's fields of these names
are kept beside them, never overwritten and never refused: the document
`{"events": [{"locator": "bar 3", "extraction_id": "run 1", "t": 1},
{"t": 2}]}` gives the columns `source_locator` `["bar 3", None]` and
`source_extraction_id` `["run 1", None]` beside the registry's `locator`
(`/events/0`, `/events/1`) and `extraction_id` (the loader's one
extraction, twice); read with `load_dict`, `source_extraction_id` holds
`"run 1"` too and no `extraction_id` column exists. A document stating
`source_locator` as well (`{"events": [{"locator": "a", "source_locator":
"b"}]}`) keeps both: `source_locator` `["b"]`, `source_source_locator`
`["a"]`.

---

### `test_xml_loader.py` (50 tests)

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
| `TestLocators` | 4 | Each row's `locator` and extraction; an attribute named `locator` kept (below) |

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
v="1"/><item id="i2" v="2"/><group><item v="3"/></group></root>`, written
into `tmp_path`, principal tag `item`: the locators are each element's path
(written out by hand: `root` holds two `item`s, `group` one), followed by
` key:<key>` where the element states its own key, its `xml:id`, else its
`id`: `/root/item[1] key:i1`, `/root/item[2] key:i2`, `/root/group/item`,
in document order, with ancestors propagated or not alike. Read from the
file, every row names the loader's one extraction; read with
`load_string`, the rows hold the locators and no `extraction_id` column.
The element's attribute `locator` is kept as the column `source_locator`
(`<root><item locator="here" v="1"/><item v="2"/></root>`: `["here",
None]`), the registry's `locator` beside it (`/root/item[1]`,
`/root/item[2]`).

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
