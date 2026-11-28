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

## TimeToShare conformance

`test_timetoshare_conformance.py` proves the library's manifests against
the schema version published by the installed `timetoshare` package. The
package's own functions are the oracle: `validate`, `unknown_keys`,
`complete`, `minimal`, `canonical_order` and `dumps` are written
independently of the library, in Python and JavaScript, so a test that
passes cannot be the library agreeing with itself. Every manifest the
library writes and the package checks is in the file form, every ID
written `#` followed by the ID; the few manifests written inline here are
turned into it by `tests/helpers.py`'s `filed` or spell the `#`
themselves.

**The fixtures, both ways.** `fixture_paths()` lists every manifest under
the package's `fixtures/0.2.0/` (drawings excluded): each fixture in the
complete form and its minimal twin `<name>.minimal.yaml` beside it, both
in canonical YAML and read with the package's `loads`. The
pairing is exact, every fixture has its twin and every twin its fixture,
and the fixtures lie in exactly the five directories a reader is named
for (directly in the directory, `groups/`, `bundles/`, `statements/`,
`flow/`), none of them empty. How many fixtures each holds is the package's catalogue to decide,
so no count is pinned here: a fixture added there is proved here without a
change. For every pair:

- both validate, and `unknown_keys` returns `[]` for both: the library
  writes no key the schema does not declare;
- `dumps(minimal(fixture))` is the twin's text and `dumps(complete(twin))`
  the fixture's: the library's two forms are exactly the package's two
  forms, compared as text, so `8` and `8.0` or a missing `null` differ;
- the class that reads the directory's manifests as a whole
  (`Timeline.from_dict` directly in the directory and in `flow/`, whose
  manifests hold one root with its flow control, `TimelineGroup.from_dict`
  in `groups/`, `AlignmentBundle.from_dict` in `bundles/` and
  `statements/`) reads fixture and
  twin to objects of the same class that write the same manifests: the
  defaults the twin leaves out are supplied by the library's reader as the
  package supplies them;
- each object writes back byte for byte: `dumps(to_dict(minimal=False))`
  is the fixture's text and `dumps(to_dict())` the twin's. Because
  `dumps` puts arrays in canonical order itself, the library's own output
  is also compared with keys sorted and arrays as they stand: the library
  must write children, conversion maps, regions, flow control, the
  destinations of every jump, timelines, groups and members in canonical
  order without help. The fixtures of `flow/` hold a manifest's FlowMaps
  and virtual timelines beside the timelines; the library does not read
  them as objects yet and must write them back as they were read, in both
  forms, which the byte comparison proves;
- `AlignmentBundle.from_dict`, which reads every manifest, reads each
  fixture and each twin and writes it back byte for byte in its own form,
  and the bundle's minimal form is `minimal` of its complete form, as text
  and as a value: the library's minimal form is the package's, whichever
  form was read.

The fixtures are written by timetoalign through the fixture script, so a
fixture that the library cannot reproduce from itself would fail here;
the package's functions decide what the forms must be.

**The shared rejection cases.** `rejection_paths()` lists the package's
hand-written rejection cases, one rule each. Every case is read by the class
that would read its manifest as a whole (`Timeline.from_dict` for one root
and no group, `TimelineGroup.from_dict` for one anonymous group, otherwise
`AlignmentBundle.from_dict`) and by `AlignmentBundle.from_dict`, which reads
every manifest. The library validates every manifest it reads with the
package's `validate` and passes its refusal on unchanged, so each reader
must raise the case's own code at the case's own path (the version case:
the package's `UnsupportedVersionError`). A reader that accepted a manifest
the validator refuses would let an invalid document through one of two
doors; one that refused it for another reason first would answer with
another code. The cases are listed from the package, not copied, so a case
added there is covered here without an edit, and there is no exception:
the rules about `graphical.rules`, band orders, named timestamps, named
MatchStamps and groupstamps, which the library keeps as given, are the
validator's like every other. Where both refusals are compared on one
value (a representation, an integer no double holds, a child beyond its
parent, a string length), the library's code, path and message equal the
validator's.

**The shared warning cases.** `warning_paths()` lists manifests carrying
unknown keys at many depths, each with the `unknown_keys` the validators
report. `AlignmentBundle.from_dict` reads every case, and the bundle writes
it back in the form the case is written in byte for byte with `dumps`, its
`unknown_keys` equal to the case's list: a case in the minimal form as it
stands, any other in the complete form as the package's `complete` writes
it (the hand-written cases predate the defaults `resources: []` and
`graphical.matchstamps: []`, which the complete form states). One case,
`several_depths.yaml`, writes its measure-map row as a hand would, without
the keys the library writes in every row it keeps (`name`, `start_repeat`,
`end_repeat`) and with JSON numbers where the library writes exact pairs; the
expected text is the case with that one row as the library writes it,
spelled out in the test, unknown `tempo` included.

**The package's tables, both ways.** `package_fixture_path("acceptances")`
lists the package's acceptance packages (the shared package with one change
each that its package check accepts: a recording resource, a timeline in
two manifests, a single-file registry, an event only re-parsable, an
external location), each beside an `expected.yaml` of its resolutions. For
every case, `Package.from_file` opens it and every manifest it lists reads
through `Package.bundle`, the events of every timeline read (its registry
rows installed, each pre-parsed content read by ID); for every resolution
the case states, the library's `resolve` gives one object per row the
package's `resolve` finds, each of the ID asked for. The case's oracle is
the package's own data, so a reader that refused a valid package, or
answered an ID with another count, fails. A bundle built here (a root with
a child holding a Note and a Projection, a recording holding a Box)
written with `to_file(events=True)` gives a registry, an Extraction table
and a Projection table that the package's table checks accept against the
manifest written (`registry_problems`, `extraction_problems`,
`projection_problems` all `[]`), and a registry whose index is the
package's `index_rows` of the manifest read as a package of one manifest.

**The unit table is the package's.** For every `TimeUnit`, its domain,
modality (`is_discrete`), default number type and admitted number types
equal the row of the package's `UNITS` exactly, and the two hold the same
units: the library knows every unit the schema names, no other, and gives
each the domain, the default number type and the number types the
validators apply, so a unit added or changed on one side only fails here.

**Manifests the library builds.** A catalogue builds manifests directly and
through the loaders, so the check covers the emitted representation rather
than assumptions made by a reader: all six timeline classes, each with two
children, a region, one break, one repeat start and one repeat end landing
on it (the continuous graphical class once in each of `points`,
`css_pixels` and `user_units`), and on the quarter-note timeline a Measure
Map built from rows;
every conversion-map class with a manifest form attached across four
timelines, and the three measure views over the quarter-note timeline's
measure map (the periodic maps have none, and a label step function is
given its end); the timelines of `Ms3Loader`, `ScoreMidiLoader`,
`AudioLoader` (in samples and in seconds), `ParangonadaLoader` and
`IIIFManifestLoader`; a group of
three timelines whose number types differ; a group whose member is a child;
a bundle of two groups, an ungrouped root and a grouped child whose root is
ungrouped; the bundles of `ParangonadaLoader`, `MatchfileLoader`,
`MpmLoader`, `Ieee1599Loader` (segment lines of pages and accolades),
`ListenHereLoader` (a two-recording export written to a temporary
directory, since no corpus holds one) and `TiliaJsonLoader`, and the group
of `TiliaJsonLoader`. The corpus paths come from `ensure_data` at module
level. For each object, `to_dict(minimal=False)` and `to_dict()` validate
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
