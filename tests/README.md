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

`test_timeskeleton_conformance.py` proves the library's manifests against
the schema version published by the installed `timeskeleton` package. The
package's own functions are the oracle: `validate`, `unknown_keys`,
`complete`, `minimal`, `canonical_order` and `dumps` are written
independently of the library, in Python and JavaScript, so a test that
passes cannot be the library agreeing with itself.

**The fixtures, both ways.** `fixture_paths()` lists every manifest under
the package's `fixtures/0.7.0/` (drawings excluded): each fixture in the
complete form and its minimal twin `<name>.minimal.json` beside it. The
pairing is exact, every fixture has its twin and every twin its fixture,
and the counts are pinned: 36 manifests of one root directly in the
directory, 10 in `groups/`, 9 in `bundles/`, each with its twin. For every
pair:

- both validate, and `unknown_keys` returns `[]` for both: the library
  writes no key the schema does not declare;
- `dumps(minimal(fixture))` is the twin's text and `dumps(complete(twin))`
  the fixture's: the library's two forms are exactly the package's two
  forms, compared as text, so `8` and `8.0` or a missing `null` differ;
- the class that reads the directory's manifests as a whole
  (`Timeline.from_dict` directly in the directory, `TimelineGroup.from_dict`
  in `groups/`, `AlignmentBundle.from_dict` in `bundles/`) reads fixture and
  twin to objects of the same class that write the same manifests: the
  defaults the twin leaves out are supplied by the library's reader as the
  package supplies them;
- each object writes back byte for byte: `dumps(to_dict())` is the
  fixture's text and `dumps(to_dict(minimal=True))` the twin's. Because
  `dumps` puts arrays in canonical order itself, the library's own output
  is also compared with keys sorted and arrays as they stand: the library
  must write children, conversion maps, regions, flow control, timelines,
  groups and members in canonical order without help;
- `AlignmentBundle.from_dict`, which reads every manifest, reads each
  fixture and each twin and writes it back byte for byte in its own form.

The fixtures are written by timetoalign through the fixture script, so a
fixture that the library cannot reproduce from itself would fail here;
the package's functions decide what the forms must be.

**The shared rejection cases.** `rejection_paths()` lists the package's
hand-written rejection cases, one rule each. Every case is read by the class
that would read its manifest as a whole (`Timeline.from_dict` for one root
and no group, `TimelineGroup.from_dict` for one anonymous group, otherwise
`AlignmentBundle.from_dict`) and by `AlignmentBundle.from_dict`, which reads
every manifest, and each must raise `ValueError`. A reader that accepted a
manifest the validators refuse would let an invalid document through one of
two doors. The cases are listed from the package, not copied, so a case
added there is covered here without an edit. The one exception is the
rules about `graphical.statements` (selectors and declarations): the
library keeps the drawing's statements as given and never interprets them,
so it has nothing to check them against. Those cases are listed by name in
the test with that reason, and the test fails if a listed case is refused
by the library or no longer exists, so the list cannot outgrow the truth.

**The shared warning cases.** `warning_paths()` lists manifests carrying
unknown keys at many depths, each with the `unknown_keys` the validators
report. `AlignmentBundle.from_dict` reads every case, and the bundle writes
it back in the form the case is written in (complete or minimal) byte for
byte with `dumps`, its `unknown_keys` equal to the case's list. One case,
`several_depths.json`, writes its measure-map row as a hand would, without
the keys the library writes in every row it keeps (`name`, `start_repeat`,
`end_repeat`) and with JSON numbers where the library writes exact pairs; the
expected text is the case with that one row as the library writes it,
spelled out in the test, unknown `tempo` included.

**Manifests the library builds.** A catalogue builds manifests directly and
through the loaders, so the check covers the emitted representation rather
than assumptions made by a reader: all six timeline classes, each with two
children, a region, one break, one jump, one marker, and on the quarter-note
timeline a Measure Map built from rows (a measure map is anchored in quarter
notes; the schema refuses one on any other unit and so does
`add_measure_map`); every registered conversion-map class attached across
four timelines; the timelines of `Ms3Loader`, `ScoreMidiLoader`,
`AudioLoader`, `ParangonadaLoader` and `IIIFManifestLoader`; a group of
three timelines whose number types differ; a group whose member is a child;
a bundle of two groups, an ungrouped root and a grouped child whose root is
ungrouped; the bundles of `ParangonadaLoader`, `MatchfileLoader`,
`MpmLoader`, `Ieee1599Loader` (segment lines of pages and accolades),
`ListenHereLoader` (a two-recording export written to a temporary
directory, since no corpus holds one) and `TiliaJsonLoader`, and the group
of `TiliaJsonLoader`. The corpus paths come from `ensure_data` at module
level. For each object, `to_dict()` and `to_dict(minimal=True)` validate
with no unknown keys, `minimal` and `complete` map each onto the other as
text, and `canonical_order` changes neither. MatchClaims are not part of a
manifest and take no part.

**Rules beyond the schema.** Two row rules of the Measure Map lie beyond
what JSON Schema can express, so they are pinned here rather than among the
shared rejections: a measure map whose `count` does not increase, and one
whose `next` names a count beyond the last row, are each accepted by
`validate` and refused by `Timeline.from_dict` with a `ValueError` naming
the count. Both readers are needed to accept a manifest; the library is the
stricter one exactly where the schema cannot speak.

**Representation, read alike by both.** A bare value on a timeline's own
axis is represented as the axis's number type selects. JavaScript numbers
do not distinguish `480.0` from `480`, so a JavaScript writer cannot write
`480.0`: a tick length of `480.0` is accepted by `validate` and read by the
library as the integer `480`, and a seconds length of `2` is accepted and
read as the float `2.0`. A tick length of `480.5` (a float on an int axis)
and of `[480, 1]` (a fraction on an int axis) are refused by both readers
with the same message, `validate` at the pointer `#/timelines/0/length`;
the string `"480"` is refused by both, as no number at all. A seconds
length of `9007199254740993` (2^53 + 1, an integer no double holds) is
refused by both with `length 9007199254740993 of 'take' is an int on a float
axis: write a float`. This proves the two readers agree on every value
either accepts.

**A child beyond its parent, refused alike.** Reading never changes a
length, so a child beyond its parent's known length is refused by both
readers in the same words, the values spelled as written:
`offset 75.0 of 'c' is beyond the length 60.0 of its parent 'r'` for an
offset, `child 'c' ends at <end>, beyond the length <length> of its parent
'r'` for an end. The end is computed, so its spelling is pinned on every
axis kind: `12` on ticks, `12.0` on seconds whose length is written `10`,
`[11, 3]` for `1/3 + 10/3` on quarters. In a segment line a segment of
unknown length ends at its successor's offset, so `a` (unknown, at 0)
followed by `b` at 61 in a line of length 60 is refused at `a`, `ends at
61.0`, before `b` is looked at. `validate`'s pointer is pinned with it.

**Integers stay exact.** `number_to_json(2**63 - 1)` remains that integer,
`number_to_json(2**63)` raises `ValueError`, and
`number_from_json(9007199254740993)` returns that exact `int`; a tick
timeline of length 9007199254740993 writes `"length": 9007199254740993` in
a manifest `validate` accepts, so integers above the floating-point safe
range survive.

**Test-only dependency.** A fresh interpreter imports `timetoalign` and
checks `sys.modules`; the absence of `timeskeleton` proves schema
validation stays a test-only dependency.

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
