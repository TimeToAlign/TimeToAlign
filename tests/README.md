# Test suite

Run `pytest` for the everyday development loop. The default lane runs in
parallel through pytest-xdist and does not collect coverage.

Long-running corpus-scale integration tests carry the `slow` marker and are
skipped by default. Run `pytest --runslow` to execute the complete suite.
The complete suite is required before any release and before changes to
loaders, unfolding, or alignment internals.

Coverage is opt-in:

```text
pytest --cov=timetoalign --cov-report=term-missing
```

To run one test file, use for example:

```text
pytest tests/path/to/test_file.py
```

## TimeSkeleton conformance

The eventless manifest is proven against the schema version published by the
installed `timeskeleton` package. The conformance module builds a catalogue of
timelines, wraps each result in a document, and passes it to `validate`, so the
assertion checks the emitted representation rather than assumptions made by a
reader. The catalogue directly constructs all six timeline classes with two
children, a region, registered conversion maps attached across its members,
one break, one jump, one marker, and, on the quarter-note timeline only, a
Measure Map built from Measure Map rows (a measure map is anchored in quarter
notes; the schema refuses one on any other unit and so does `add_measure_map`).
It also uses `ensure_data` at module level for the same `Ms3Loader`, MIDI,
`AudioLoader`, discrete-physical, and graphical specimens listed by the
fixture corpus, proving loader output has the same structural contract without
depending on a relative data path.

Every bundled fixture for every `SUPPORTED_VERSIONS` entry from
`fixture_paths(version)` must round-trip exactly from
`json.loads(json.dumps(Timeline.from_dict(m).to_dict())) == m` for each
top-level manifest. Older fixtures are passed through `timeskeleton.migrate`
first, so this demonstrates that migration leaves the library with a complete,
lossless current document rather than merely an accepted approximation.

Two row-level rules of the Measure Map lie beyond what JSON Schema can
express, so they are pinned here rather than among the shared rejections: a
document whose measure map has a `count` that does not increase, and one whose
`next` names a count beyond the last row, are each accepted by `validate` and
refused by `Timeline.from_dict` with a `ValueError` naming the count. Both
readers are needed to accept a document; the test proves the library is the
stricter one exactly where the schema cannot speak.

JavaScript numbers do not distinguish `480.0` from `480`, so a JavaScript
writer cannot produce `480.0`. A coordinate `{"number_type": "int", "value":
480.0}` is accepted by `validate` and must restore as the integer `480`; one
holding `480.5` is refused by both, with the library naming the declared
number type ("does not match declared number_type 'int'"). A value that is no
JSON number at all, such as the string `"480"`, is a different failure and
gets a different message ("invalid coordinate value for key 'value'"). The mirror case `{"number_type": "float",
"value": 2}` restores as the float `2.0`. This proves the two readers agree on
every value either accepts.

The integer boundary asserts that `number_to_json(2**63 - 1)` remains that
integer, `number_to_json(2**63)` raises `ValueError`, and
`number_from_json(9007199254740993)` returns that exact `int`. A
`DiscreteLogicalTimeline(length=9007199254740993, unit="ticks")` must emit
`{"unit": "ticks", "number_type": "int", "value": 9007199254740993}` in a
document `validate` accepts, proving coordinates do not lose integers above
the floating-point safe range. Finally, a fresh-interpreter subprocess imports
`timetoalign` and checks `sys.modules`; the absence of `timeskeleton` proves
schema validation stays a test-only dependency rather than becoming runtime
library behavior.

## Layout

Tests live in a directory per subject area, each with a README documenting
what its assertions check and why those are the right values.

| Path | Subject |
|---|---|
| `alignment/` | Claims, graphs, match lines, warp maps, bundles |
| `core/` | Scalars, fields, number storage, stamps, display |
| `display/` | Shared rendering helpers |
| `integration/` | End-to-end corpus walkthroughs |
| `loader/` | Every loader family, and loader parity |
| `maps/` | Conversion maps |
| `storage/` | Event storage and schemas |
| `timelines/` | Timelines, nesting, regions, flow, unfolding |

Two suites sit at the top level rather than in any of them:

| File | Subject |
|---|---|
| `test_number_type_preservation.py` | A declared `number_type` is preserved at every coordinate boundary, whichever object answers the query. Its validation logic is documented in `core/README.md` under "Number type is preserved everywhere". |
| `test_stamp_retrieval_surface.py` | Every stamp-producing object answers the same four precise questions under the same four names, and every table speaks the same two output formats. Its validation logic is documented in `core/README.md` under "One stamp surface, one table surface". |

They stay at the top level because they are cross-cutting rules rather than
subject areas: each asserts the same property of timelines, groups, bundles,
maps and stamps together, and filing either under any one of them would
invite the next person to check only that one.

`helpers.py` beside them holds the one utility both need: `table_column`,
which reads a timestamp-table column through the library's own decoding
boundary. Timestamp-table cells are coordinate structs, so a bare
`column.to_pylist()` yields dicts rather than positions; tests read what a
user reads.
