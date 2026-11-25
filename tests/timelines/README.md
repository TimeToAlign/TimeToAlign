# Timeline Tests

This directory contains comprehensive tests for the `timetoalign.timelines` module,
which implements the central Timeline class and its 6 domain-specific subclasses.

## Test Coverage Summary

| Module | Coverage | Status |
|--------|----------|--------|
| `timelines/base.py` | 94% | Excellent |
| `timelines/types.py` | 100% | Complete |
| `timelines/beatgrid.py` | 100% | Complete |
| `timelines/regions.py` | 94% | Excellent |

### Measure Map compressed rows

Measure-map documents are compressed row lists rather than tagged measure
objects. The tests prove this is a lossless codec, not merely a shorter
rendering: every map ``MeasureMapLoader`` builds from a folded ``.mm.json``
score specimen (their anacrusis constituents included) and every map built by
``Ms3Loader`` from the folded measure tables already exercised by
the suite satisfies both ``MeasureMap.from_rows(m.to_rows()) == m`` under strict
equality (same concrete class and every field equal, ``offset_within_measure``
included) and the row fixpoint
``MeasureMap.from_rows(m.to_rows()).to_rows() == m.to_rows()``. Equality has no
exclusions, so a codec that dropped an offset or reclassified a measure fails
the first assertion even when the rows happen to be a fixpoint.

The reference implementation independently determines which rows carry new
information. For the same specimens, tests expand the library rows, convert the
exact values to floats for the reference input, and require identical retained
counts and shared values after conversion (the specimens use default IDs, so a
library ``next`` of IDs is compared with the reference's counts through
``int``). The pinned corpus results are:
``c05n05_musete`` has 8 of 58 rows at counts ``[1, 5, 6, 16, 17, 31, 32, 58]``;
``c11n08_Rondeau`` has 8 of 60 at ``[1, 9, 10, 18, 19, 27, 28, 60]``;
``Bruckner_WAB.105_3a_Scherzo.measures`` has 2 of 382 at ``[1, 382]``;
``Piano_Concerto_No._2_Opus_18_1st_Movement__Rachmaninoff.measures`` has 4 of
374 at ``[1, 8, 337, 374]``; and ``WoO71.measures`` has 52 of 397. Agreement
with the ``pymeasuremap`` reference implementation establishes that retained
rows follow the standard's default successor rule rather than an internal
compression choice.

The standard example ``[{"time_signature": "2/2"}, {"number": 563}]``
expands to 563 measures and recompresses to two rows at counts 1 and 563. Its
long omitted run proves successor arithmetic, rather than merely checking a
local change. Unfolded score files are deliberately excluded because their
``count`` values restart and therefore cannot describe one loadable map.

The document test requires a list whose first row has ``count``, ``qstamp``,
``number``, ``name``, ``actual_length``, ``start_repeat``, and ``end_repeat``;
it has no ``ID`` (the default ``str(count)`` is never written) and no null
value. The former ``{"measures": [...]}`` wrapper is refused. A timeline
carrying a corpus map survives a dictionary round trip, which proves the
enclosing serializer uses the same compressed codec.

**Row identity.** A measure's ``id`` is its row ``ID``: ``str(count)`` unless
the row states another, and ``next`` names successors by these IDs, so there is
one spelling of a measure's identity. A map authored with IDs ``a``, ``b``,
``c`` (each row complete) must reproduce its own rows byte for byte: the first
row carries ``"ID": "a"`` and ``"next": ["b"]``, and the measures' ids are
exactly ``("a", "b", "c")``. ``next`` read as the standard's ``count``
integers and as its ``ID`` strings must build equal maps, an empty ``next``
ending the piece. ``-1``, the measure tables' spelling of the end of the
piece, is no row count: the row lane refuses it like any count that names no
row (``from_measure_data`` and the ``.mm.json`` loader, which read measure-table
conventions, drop it before rows exist). A ``next`` count beyond the last row,
``0``, ``-1`` and an ``ID`` no row states are refused by name, as is an ``ID``
stated twice. A map states every successor: a measure built without ``next`` is
followed by the next measure in printed order and the last one by none, so a
map built from ``Measure`` objects and the same map read back from its rows
compare equal.

**One classifier.** Rows and measure tables are classified by one function,
``Measure.from_values``. The five outcomes are pinned on one row each: no
nominal length gives ``Measure``; a ``cadenza`` signature ``CadenzaMeasure``;
a stated ``offset_within_measure`` ``MeasureConstituent`` carrying exactly that
offset; actual equal to nominal ``RegularMeasure``; anything else
``IrregularMeasure``. A stated offset on a measure without nominal length is
refused, because it could not survive the round trip. For the same reason a
map holding a ``SplitRegularMeasure`` or ``SplitIrregularMeasure`` cannot be
written as rows: ``to_rows`` raises a ``ValueError`` naming that measure's ID
and class and returns nothing, instead of writing a row that would read back
as a different class. Two short rows sharing a
``number`` (the standard's ``16a``/``16b`` endings) stay ``IrregularMeasure``:
a shared number is never read as a split measure. ``to_rows`` writes
``offset_within_measure`` exactly when it is non-zero and a row stating ``0``
is refused (the schema's rule), so a constituent and an irregular measure of
equal lengths produce different rows.

**One reader.** Row numbers are read only by ``number_from_json``: a JSON
integer, a JSON float (its exact binary expansion, never a guessed ratio), or a
Fraction pair. A string such as ``"1/2"`` is refused, as is a Python
``Fraction`` object, which is not a document value.

**Row-level rules.** The standard's compression omits every row equal to its
predecessor's default successor, so a jump in ``count`` (1 then 3) is an
omitted run and expands to the default successors it stands for: three
measures. A ``count`` that does not strictly increase cannot arise from
compression and is refused with an error naming that count, whether it goes
backwards (3 then 2) or repeats (2 then 2). ``volta`` carries no constraint: a row outside any repeat may carry
``volta: 7`` and the value survives the round trip. A ``volta`` is a pass
number or a list of them, and a map keeps the form its rows state: rows
stating ``1``, ``[1]``, ``[1, 3]`` and ``[3, 1]`` give measures stating ``1``,
``(1,)``, ``(1, 3)`` and ``(3, 1)``, write the same four values back
compressed and in full, and keep them in a slice; an empty list and a list
naming a number twice are refused. A stated ``qstamp`` that
contradicts the position derived from preceding actual lengths emits exactly
one ``UserWarning`` naming both values and the derived position is kept; the
same message is emitted by ``MeasureMap`` construction from measures, because
both go through one helper.

**Axis.** A measure map is anchored in quarter notes: a quarters timeline
holds it as it is, a ``whole_notes`` or ticks timeline through a
``LinearMap`` to quarters (see *Measure maps in a manifest*), and
``add_measure_map`` on any other unit raises; ``add_metric_hierarchy`` stays
bound to a quarters timeline holding its measure map as it is.

**Corpus measure classes.** Every folded corpus specimen was probed before
and after the classifier was unified; the only differences are the ones the
classifier specifies:

- twelve folded ``.mm.json`` specimens open with a row numbered 0 that is
  shorter than nominal; that row was an ``IrregularMeasure`` and is now a
  ``MeasureConstituent`` at ``nominal - actual`` (the loader's anacrusis
  convention): ``n04op18-4_04`` and its ``_flow``, ``sq8913219`` and its
  ``_flow``, ``Beethoven_Op018No4-04`` and its ``_flow`` (offset 3 each),
  ``WoO71`` and ``original_file_before_cleaning`` (1), ``c05n05_musete``
  (3/2), ``c11n08_Rondeau`` (1), ``out_of_the_flow_experience-flow_only`` (3),
  ``-full`` and ``-polyrhythm_only`` (5). Every other ``.mm.json`` row keeps its
  class; the inner split bars that the corresponding measure tables mark with
  ``mc_offset`` stay ``IrregularMeasure`` in the ``.mm.json`` lane, because
  those files state no offset and a shared number is not one;
- the ``Music21Loader`` and ``PartituraLoader`` extract no time signature into
  their measure tables (the ``timesig`` column is empty), so the nominal length
  is unknown: every measure of the 13 Music21 and 19 Partitura specimens that
  load was a ``RegularMeasure`` (the former loader lane set nominal to actual)
  and is now a plain ``Measure``, the class of a measure whose nominal length is
  unknown;
- Partitura closed a repeat that ends the piece with a successor one past the
  last bar (``c11n08_Rondeau`` and its ``_reviewed`` revision: count 61 of 60).
  Such a ``next`` is now refused, and the loader states the end of the piece as
  ``-1`` instead, so both specimens load as before;
- no ``Ms3Loader`` measure table and no Rekordbox track changes class or
  offset.

### Measure map slices (`test_structure.py`, `test_nesting.py`)

``MeasureMap.get_slice(start, end)`` returns the measures holding
``[start, end)`` (quarters), re-based to ``start``; ``create_child(...,
measures=True)`` gives a child that slice of its parent's map. A slice is
proved right by evidence that does not come from the slicer itself:

1. **Rows written by hand.** The slice's ``to_rows()`` equals rows written
   out by hand from the parent's measures, re-based: never the slicer's own
   output fed back. The fixture is a 9/8 map of 72 measures (324 quarters,
   IDs ``str(count)``) read from its two compressed rows. ``[0, 27)`` is two
   rows (measures 1 and 6, the last without ``next``, because measure 7 is no
   longer in the map); ``[27, 54)`` is six rows, each stating its ``ID``
   ``"7"`` … ``"12"`` beside counts 1–6, since an ID kept from the parent is
   no longer ``str(count)``, and each ``next`` but the last's names the
   following ID. ``[9/4, 27)`` cuts measure 1 at its middle: it becomes a
   ``MeasureConstituent`` at offset ``9/4`` with actual length ``9/4``, the
   map's total actual length is ``99/4`` and the last row's qstamp ``81/4``.
   A second fixture holds what a uniform map lacks: seven 4/4 bars (28
   quarters, IDs ``str(count)``) with a plain repeat (bar 2 ``start_repeat``,
   bar 3 ``end_repeat`` with ``next: ["2", "4"]``), a repeat with two voltas
   (bar 4 ``start_repeat``; bar 5 ``volta: 1``, ``end_repeat``, ``next:
   ["4"]``; bar 6 ``volta: 2``, ``next: ["7"]``) and an unknown row key
   (``tempo: "Allegro"`` on bar 6, read with ``keep_unknown_keys``). Four
   slices, rows written by hand: ``[6, 22)`` is cut at both ends (bar 2 a
   ``MeasureConstituent`` at offset 2 with actual length 2, bar 6 clipped to
   actual length 2 while its nominal length stays 4, hence an
   ``IrregularMeasure``), keeps both successors of bar 3, both repeat signs,
   both voltas and ``tempo``, and bar 6 loses its successor ``"7"``; ``[10,
   22)`` drops the backward ``"2"`` of bar 3 (``next: ["4"]``); ``[16, 22)``
   leaves bar 5, mid-slice, with ``next: []`` since its only successor lies
   before the slice; slicing ``[6, 22)`` again at ``[1, 6)`` cuts the
   constituent further, its offset the sum ``2 + 1 = 3``.
2. **Same measure at the same position.** For positions of the child the
   child's ``MeasureRecordMap`` answers the measure (by ``id``) that the
   parent's answers at offset + position: ``"1"``, ``"2"``, ``"6"`` at 0,
   9/2, 26 of a child at offset 0; ``"7"`` at 0 of a child at offset 27. A
   ticks parent (480 ticks per quarter, its map attached through a
   ``LinearMap`` to quarters) proves the span is read in quarters and the
   child reaches its slice through the same scalar: the child at ticks 12960
   of length 12960 holds the ``[27, 54)`` rows and answers ``"7"`` at tick 0.
   The child's map is the parent's map from the child's origin, in the
   number type the parent's map was given: a parent map given
   ``output_number_type=float`` gives a child map answering ``float``
   (``1.0`` at tick 480, the parent's answer), with the same ``[27, 54)``
   rows. A whole-notes parent (81 whole notes, its map read through the
   default ``LinearMap`` multiplying by 4) gives the child at ``27/4`` of
   length ``27/4`` exactly the quarters child's rows.
   ``create_child_from_region`` over a region ``[27, 54)`` and
   ``create_children_from_boundaries([27, 54])`` hold the same rows: one
   slice, whichever call derives the child. ``create_children_from_regions``
   over the regions ``[0, 27)`` and ``[27, 54)`` with ``measures=True`` gives
   two children holding exactly ``BARS_1_TO_6`` and ``BARS_7_TO_12``, in
   region order; without it, neither holds a measure map.
3. **Refusals change nothing.** ``get_slice`` refuses ``start >= end``, a
   range outside ``[0, total actual length]`` and a map with a measure lacking
   ``actual_length`` (``ValueError``), and leaves its source map equal to a
   fresh copy. ``create_child(..., measures=True)`` on a parent without a
   measure map, or with an unknown child length, raises ``ValueError`` and
   the parent's children are unchanged, because the slice is computed before
   the child exists. The default, ``measures=False``, gives no measure map.
4. **The child's map survives a manifest.** A bundle holding the parent with
   its child, written by ``to_dict()`` and read by ``from_dict()``, holds two
   ``measure_maps`` entries, and the child read back carries a measure map
   equal to the slice.
5. **The child affords its parent's measure views.** A view of the measure
   map (``FloatingMeasureMap``, ``MeasureNumberMap``, ``MeasureBeatMap``) is
   built from the measure map alone, so the child gets each view its parent
   holds, built over its slice by the same constructor. A quarters parent
   holding a ``FloatingMeasureMap`` and a ``MeasureNumberMap`` of its map
   gives the child at ``[27, 54)`` both, reading the child's own measure map:
   floating measures ``1.0``, ``2.0``, ``7.0`` at 0, ``9/2``, 27 (the
   lattice counts the slice's measures from 1, as its ``count`` does), bar
   numbers ``7`` at 0 and ``12`` at 26 (numbers are kept). The child at
   ``[9/4, 27)`` is cut in bar 1, which the lattice reads as an anacrusis,
   the shape it has: ``0.5`` at 0 and ``1.0`` at ``9/4``, where bar 2
   begins. A parent without
   views gives a child without views (only its ``MeasureRecordMap``). The
   bundle written from the parent with its two children holds three
   ``measure_maps`` entries (the views share their timeline's entry), and
   the child read back holds a ``FloatingMeasureMap`` reading a measure map
   equal to its slice.

### Merged Score Event Coordinates

The score-timeline regression inspects the flattened Partitura event table
directly.  It counts populated ``start``, ``end``, and ``duration`` structs,
checks that every populated struct has an exact pair, and verifies that all
instant control and annotation rows keep null ``end`` and ``duration`` cells.

## Native Timeline Structure and Flow-Control Validation

Timeline structure is validated as one hierarchy-wide storage contract. A fresh
instance of the base timeline, each of the six domain/modality classes, and a
parameterized segment line must expose an axis-bound flow-control registry while
both optional measure structures are exactly ``None``. Adding either structure a
second time must raise, because replacing the authored structure would describe a
different timeline.

The registry's binding to its axis (units, number types, unbound storage,
copies) is described under *Flow control* below.

A region answers containment for raw numbers by value on its own axis. A
coordinate object in another unit is rejected with both units named instead of
comparing values that measure different quantities.

Flow-control elements and regions write every coordinate as a bare value on
the timeline's own axis, represented as its number type selects, and read it
back against that axis. The registry is written as one array of elements (see
*Flow control*); a JSON encode/decode cycle cannot change it.

A structure-bearing timeline combines a measure map containing regular,
irregular, and constituent measures, each with the number and nominal length
its class requires (the constituent at offset ``1/3``, so a codec dropping the
offset fails strict map equality); a metric hierarchy; a region; one break;
one jump; one jump target; one child; and one conversion map. Its default timeline
object has exactly ``id``, ``name``, ``unit``, ``number_type``, ``length``,
``locked``, ``is_segment_line``, ``meta``, ``children``, ``conversion_maps``,
``regions`` and ``flow_control``; ``events`` remains absent, and the measure
map is a ``MeasureRecordMap`` among the conversion maps naming the manifest's
one ``measure_maps`` entry. The metric hierarchy has no manifest form: it is
not written, and the timeline read back has none. JSON must be a fixpoint,
and restoration must reproduce each written constituent by exact equality. Same-extent retyping and conversions to and from a segment
line preserve both structure slots and copy the registry into a distinct object.
A slice carries none of those three objects, pinning the current derivation
boundary: slicing, child derivation, and unfolding do not infer how authored
structure should be partitioned.

### Manifests (`test_manifest.py`)

`Timeline`, `TimelineGroup` and `AlignmentBundle` write and read one shape of
TimeToShare manifest: `{"timetoshare": "0.2.0", "timelines": [...]}` with an
optional `id`, `name`, `meta`, `groups`, `measure_maps`, `resources`,
`graphical`, `@context`, `statements` and `provenance` (*Statements*
below). The library does not read `resources` yet: a manifest's resource
entries are kept as read and written back unchanged, and the complete form
states their default `[]` as TimeToShare's `complete` does. A
reader that only
reads what its own writer produced proves nothing about the format, so the
tests read manifests written by hand from the format's rules, and compare what
the library writes back against the same literal text.

- **The format's validator decides at the boundary.** Every manifest the
  library reads (`from_dict`, `from_file`, whatever class reads it) and
  every manifest it writes (`to_dict`, `to_file`) goes through
  TimeToShare's `validate` whole, and a refusal reaches the caller
  unchanged: a `timetoshare.SchemaValidationError` whose `code`, `path`
  (as a JSON pointer) and `message` the tests assert exactly
  (`tests/helpers.py`'s `refusal` returns the three). The path is where the
  fault lies, and a message often does not say it: an ID held twice is
  refused at its second holder (the message names the first, the path the
  second), and a reference naming nothing (`'clt_9' names no timeline of
  this manifest`) is placed only by its path; a test asserting code and
  message alone would pass a validator blaming the wrong element. The
  validator's warnings (unknown keys) raise nothing. What proves it is a
  rule the library never checked itself: a named timestamp, which the
  library keeps unread with the styling. A timestamp on `clt_9`, a
  timeline the manifest lacks, is refused on reading with
  `timestamp_timeline`, `'clt_9' names no timeline of this manifest`; one
  at `[13, 1]` on `clt_1` of length `[12, 1]` with `timestamp_coordinate`,
  `coordinate [13, 1] of 'timestamp_1' is beyond the length [12, 1] of its
  timeline 'clt_1'`, through `Timeline` and `AlignmentBundle` alike. The
  writing side: a valid manifest whose timestamp sits at `[11, 1]` is read,
  `length` set to 10, and `to_dict` refuses with `coordinate [11, 1] of
  'timestamp_1' is beyond the length [10, 1] of its timeline 'clt_1'`;
  `to_file` refuses the same and leaves the directory empty, and a bundle
  holding MatchClaims and Projections given such a timestamp writes
  neither its manifest nor its two tables: the refusal comes before any
  file. The serialization refuses before any file too, because the
  manifest is serialized before a table is written: the same bundle
  holding a `DiscretePhysicalTimeline` of length `2**53 + 1` written with
  `serialization="jcs"` raises the serialization's `ValueError` (`the
  integer 9007199254740993 cannot be written in RFC 8785 JSON, whose
  numbers are doubles`), and an unknown serialization name (`"rfc8785"`)
  raises its own; in both cases the directory stays empty and the
  bundle's `resources` are unchanged (a writer that wrote the tables
  first leaves `match_claims.parquet` and `projections.parquet` behind and
  renames the resources). Into a directory already holding the bundle's
  package, written in YAML, and then given a belief that changes its
  MatchClaim table, the refused `jcs` writing leaves every file with its
  bytes and the `resources` as they were. Every other refusal pinned below in the validator's words is the
  same rule seen from another case; a valid manifest reads and writes
  byte for byte as before (the round trips of this file,
  `test_manifest_files.py` and the conformance tests are unchanged).
- **Rules of the library alone.** A manifest the format admits may still
  be refused by the library where it cannot hold what is written; these
  checks stay the library's, with its own words, and are pinned where
  they are tested: a negative child offset (below), a region ending
  before it starts, measure-map rows whose `count` does not increase, a
  `next` beyond the last row, a measure view over rows without lengths, a
  `LinearMap` of scalar 0, a `ChainMap` whose links' units do not meet,
  and a timeline holding two maps that read a measure map as its own.
  The readers of one element outside a manifest (a measure-map entry, a
  flow-control array, a region, a conversion map) see no validator, so
  they keep the format's rules they check: the measure-map entry and
  flow-control tests read the bare element and assert the library's
  words.
- **IDs in the file form.** A manifest dict, what `to_dict` returns and
  `from_dict` reads, writes every ID as `#` followed by the ID, while every
  object holds IDs bare. Most literals are written with bare IDs, as the
  objects hold them, and reach a reader through `tests/helpers.py`'s
  `filed` (or a reader wrapped by `reading`); what a writer returns is
  compared after `bare`. Both are TimeToShare's `with_file_ids` and
  `with_bare_ids`, which the package proves, applied to a whole manifest
  only. Literals that stand for file text (`test_manifest_files.py`) and the
  assertions on a written manifest's own keys (`test_base.py`,
  `test_maps_integration.py`, the doctests) spell the `#` themselves. A
  literal whose empty bundle `id` becomes `#`, or a bare ID in a manifest,
  is refused by the validator (`schema`).
- **The version marker.** Without `timetoshare` each reader raises the
  validator's `not_a_manifest`, `not a TimeToShare manifest`;
  `"timetoshare": "0.1.0"` raises its `UnsupportedVersionError`,
  `unsupported TimeToShare version "0.1.0"; this package reads 0.2.0
  only`. Nothing else is read first, so the refusal cannot depend on the
  rest of the content. Writing stamps the format's own
  `timetoshare.SCHEMA_VERSION` (`0.2.0`, asserted), never a copy of it, so
  the library cannot write a version its validator no longer reads.
- **Defaults are supplied.** A minimal manifest (only `id`, `unit` and
  `length` per timeline, only `id`, `role` and `coordinate` per section
  break, and so on)
  and its complete twin, both literal, read to objects whose complete and
  minimal forms are identical; the complete form written back equals the
  complete literal and the minimal form the minimal literal. For each of the
  eighteen units a minimal timeline reads with the unit's own number type —
  `int` for ticks, samples, frames and pixels, `fraction` for quarters and
  whole_notes, `float` otherwise — asserted per unit, so a reader that
  assumed one default for all would fail on two of the three groups.
- **The class comes from the unit.** A timeline object carrying a stale
  `"class": "ContinuousGraphicalTimeline"` beside `"unit": "seconds"` reads as
  a `ContinuousPhysicalTimeline`; the `class` key is not read, and is kept and
  written back as an unknown key.
- **The number type governs, the lexical form never.** One refused value per bare key —
  `length`, a root's `offset`, a child's `offset`, region `start` and `end`,
  the `coordinate` of a flow-control element (a section break, a jump
  target and a jump), the `by` of a jump's destination —
  each raises the exact message naming the key, the value, the id of the
  object holding the key (the timeline for `length` and `offset`, a child's
  own id for its offset, the region or flow-control element for its keys, the
  jump for its destination's `by`) and
  what to write instead, e.g. `length 32 of 'score' is an int on a fraction
  axis: write [32, 1]`. The rewritten value is named only when the rewriting
  is exact, otherwise the axis's number type: `[5, 2]` on a float axis says
  `write 2.5`, `[1, 3]` says `write a float`, `[1, 2]` on an int axis says
  `write an int`, and `0.1` on a fraction axis names its exact pair
  `[3602879701896397, 36028797018963968]`. These are the validator's
  refusals, code `representation`, its message asserted exactly. What a JSON round trip may
  respell is read as the number type decides, each case asserting the value
  AND its Python type: `480.0` on an int axis is the int `480`, `8` on a
  float axis the float `8.0`, `[1, 8.0]` on a fraction axis the Fraction
  `1/8` (a JavaScript writer may respell any member). The respelling never
  admits a value that no reading of the axis allows: `[1, 8.5]` (a
  non-integral member) is refused by the schema (`8.5 is not of type
  'integer'`), `"8"` likewise (`'8' is not of type 'number'`), and
  `[480, 1.0]` on an int axis is still a fraction on an int axis
  (`write 480`).
- **Unknown keys are kept where they were found.** One literal manifest
  carries an `x-note` key on the manifest, a root and a child timeline, a
  region, a jump target, a jump, one of its destinations and that
  destination's `after`, a break and its `ends`, a section break, a
  conversion map and a map nested in it, a `measure_maps` entry, a group and `graphical`, each
  with a different value. `AlignmentBundle.from_dict` followed by `to_dict` equals the literal,
  and `Timeline.from_dict` and `TimelineGroup.from_dict` keep the keys of the
  parts they read. Distinct values at every depth catch a reader that moves a
  key to the wrong object.
- **Measure-map rows keep unknown keys too.** A `measure_maps` entry's
  `rows` are `[{"count": 1, "time_signature": "4/4"},
  {"count": 2, "x-row": "kept"}, {"count": 3, "tempo": 96}]`. Row 2 is the
  default successor of row 1, so without its unknown key the writer would
  leave it out; with it, the row is written, all three rows in full, `x-row`
  on row 2 and `tempo` on row 3. A JSON `tempo` is a key the row form does
  not define (a document cannot hold a tempo object), so it is kept, not
  read. Reading the written form writes it again unchanged, and the map
  equals the one read from the same rows without the unknown keys: they
  take no part in equality.
- **Measure-map rows a manifest refuses.** The rows of a manifest follow the
  schema: empty `rows`, a row with neither `count` nor `ID`, a negative
  `qstamp` and an `actual_length` of `-1` are each refused by the
  validator (`schema`, its message exactly), and so is a conversion map
  lacking a key its type requires (a `TableMap` without `x_values`). The
  reader of one measure-map entry, `MeasureMap.from_dict`, sees no
  validator and refuses the same four rows in its own words, naming the
  row's position.
- **Reading never changes a length.** A root `score` (quarters, length
  `[10, 1]`) holding a child `coda` at offset 8 with length 4 is refused,
  locked or not, by the validator (`child_end`): `child 'coda' ends at
  [12, 1], beyond the length [10, 1] of its parent 'score'`; a child `late`
  of unknown length at offset 12 with `child_offset`, `offset [12, 1] of
  'late' is beyond the length [10, 1] of its parent 'score'`. A child ending
  exactly at 10 is read, and the parent writes `[10, 1]` back in both forms.
  The parent is never grown to the child's end, which is what these pin.
  On a seconds root of length `60.0`, an end of `62.5`, an offset of
  `75.0` and a segment of unknown length whose successor starts at `61.0`
  are refused with the values spelled as written (`ends at 62.5`, `offset
  75.0`, `ends at 61.0`). The setters keep the library's own message, since
  no JSON spelling exists there.
- **A child's end is the sum a reader computes.** The rule is TimeToShare's,
  applied by the format's validator before the library reads anything, so
  the reader and the validators cannot disagree: where the parent and the child are both `float`, the end is the
  double sum of the two JSON numbers, exact otherwise. A `points` page of
  length `932.2326510101616` holding a child at offset `76.70371459980326`
  of length `855.5289364103584` ends, in doubles, exactly at the page's
  length, while the exact sum lies one rounding beyond it: the page is read,
  and so is a segment line whose second segment ends there. The same values
  as Fraction pairs on a `fraction` page, and the same float page holding a
  `fraction` child, are refused with the exact end
  `[65600040932999587, 70368744177664]`; a `ticks` parent of length `2**53`
  holding a child at offset 1 of length `2**53` is refused with `ends at
  9007199254740993`, although the double sum would round back to `2**53`.
  A reader summing exactly everywhere fails the first two cases, one summing
  in doubles everywhere the last three.
- **Ids are free; a length is not negative.** An id is any non-empty
  string without a control character, so the words that name the parts and
  facets of a drawn timeline are ids like any other: `start` as a root id,
  `end` as a region id, `card` as the manifest id, `regions_1` as a group id,
  `label` as a conversion-map id and `markers_10` as a child id are read,
  and the manifest writes back equal to what was read; so are the
  overrides `Timeline.from_dict(..., timeline_id="start")` and
  `TimelineGroup.from_dict(..., group_id="regions_1")`, which give the root
  and the group these ids. (Free ids with slashes, spaces and URIs are
  proved under "Manifest identity" below.) A negative length is
  refused on reading by the schema (`[-12, 1]` on quarters: `-12 is less
  than the minimum of 0`; `-8.0` on seconds: `-8.0 is less than the
  minimum of 0`) and on construction in the library's words (`length -12
  of 'x' is negative`).
- **A negative child offset is not supported yet.** The format admits a
  child at a negative offset; the library does not place one yet. A child
  `neg` at `[-5, 2]` on a quarters root, at `-1` on a samples root and at
  `-2.5` on a seconds root is refused by every reader (timeline, group,
  bundle) with `A negative child offset is not supported yet: child 'neg'
  has offset [-5, 2].` (`-1` on samples, `-2.5` on seconds), the value
  spelled as the TimeToShare writer writes it and its validators quote it.
  A root's own negative `offset` stays readable.
- **A group made from one timeline keeps the manifest's keys.** A one-root
  manifest with `id`, `name`, `meta`, `graphical.styles` and an unknown
  top-level key, read by `TimelineGroup.from_dict(..., group_id="g")`,
  writes the anonymous bundle of its group with the same `meta`,
  `styles` and unknown key: only `id` and `name` are gone, because a group
  writes an anonymous bundle.
- **A `graphical` object read is written back.** `"graphical": {}` on a
  one-root manifest is written back as `{}` in the minimal form and as
  `{"elisions": [], "order": {}, "rules": [], "styles": {},
  "timestamps": [], "matchstamps": []}` in the complete form (every member
  of `graphical` with a default, as TimeToShare's `complete` writes it),
  through `Timeline` and `AlignmentBundle` alike; a manifest read without
  `graphical` is written without it. Presence is part of the round trip.
- **A stored elision is written back unchanged.** `graphical.elisions` and
  `graphical.rules` are styling members the library keeps without reading.
  A one-root manifest whose `graphical` holds two elisions on the root,
  given out of the canonical order (the later one first), is read through
  `Timeline` and `AlignmentBundle` alike and written back in both forms.
  The first elision states every optional member: a `width` of the `ratio`
  form, a `name` and a `display` of all five words (`ellipsis`, `name`,
  `id`, `coordinate`, `continuation`) in canonical order; the second states
  none. Two rules addressed to the root's cards carry the four elision
  properties, once with values (`elision-mode: proportional`,
  `elision-proportion: 0.5`, `elision-depths: "1 2"`, `elision-display:
  "ellipsis name"`) and once with the words `auto` and `none`. The minimal
  form equals the literal read (same order, every member kept, nothing
  added); the complete form equals it with the second elision's default
  `display` stated (the format's own `timetoshare.DEFAULTS`, so the test
  does not restate it) and the other members of `graphical` at their
  defaults, as `timetoshare.complete` writes them. The elision model is
  TimeToShare's: the library neither orders, checks nor interprets an
  elision or a rule, so a test of its own can only prove that nothing of
  the stored values is lost, reordered, rewritten or invented on the way
  through. Elisions are never part of the timelines the library builds.
- **A stored elision survives a file round trip.** The same manifest is
  written with `to_file` to YAML and to JSON and read again with
  `from_file`, through both classes. The file lists the elisions in the
  format's canonical order (by ID), so the minimal form read back equals
  the manifest with the elisions sorted by ID, rules and every value
  unchanged; writing the read object again gives the same bytes.
- **A styling member at its default is dropped as TimeToShare drops it.**
  The library keeps the styling read (every member of `graphical` but
  `order`) in TimeToShare's minimal form and has no default table of its
  own: `styles: {}`, `rules: []`, `timestamps: []` and `matchstamps: []`
  are each read to no member, a named timestamp's default `display` and
  its defaults `"matchstamp": false` and `"groupstamp": false` to none and a
  non-default `display` to its canonical order. Each case is asserted
  twice, against a literal written from the format's defaults and against
  `timetoshare.minimal` of a manifest holding only that `graphical`, so the
  library and TimeToShare cannot drift apart. A member whose value Python's
  `==` confuses with another type (`0` and `False`, in an unknown member
  and inside a style's declarations) is kept with its own type, as
  TimeToShare compares a boolean only with a boolean.
- **A named timestamp's `display` has the format's default.** The library
  keeps `graphical.timestamps` unread but for `display`, the parts drawn of
  a timestamp, whose default `["wedge", "name", "cross_section"]` the
  complete form states and the minimal form leaves out, and whose words are
  written in the canonical order `wedge`, `name`, `id`, `coordinate`,
  `cross_section`; a word the format lacks (`flag`) is refused by the
  validator (`schema`, `'flag' is not one of ['wedge', 'name', 'id',
  'coordinate', 'cross_section']`). Read from a
  one-root manifest through `Timeline` and `AlignmentBundle` alike: a
  timestamp without `display` is written with the default in the complete
  form and without it in the minimal one; the complete form of every one
  of these ordinary timestamps also states `"matchstamp": false` and
  `"groupstamp": false`, the format's defaults (TimeToShare's schema gives
  both the default `false`), which the minimal form leaves out; one stating the default as
  `["cross_section", "wedge", "name"]` is written exactly as the first in
  both forms; one stating `["name", "wedge"]` is written `["wedge",
  "name"]` in both; one stating `["coordinate", "id", "wedge"]` is
  written `["wedge", "id", "coordinate"]`; every other member of a
  timestamp, an unknown one included, is written as given, and the input
  is never changed.
- **A groupstamp lies in a group.** A named timestamp with `"groupstamp":
  true` draws its line across the frame of the groups its timeline lies in,
  so the format refuses one whose timeline lies in none, and so does the
  library, on reading and on writing, through the validator. Read: a one-root manifest without
  groups holding a groupstamp on its root raises, through `Timeline` and
  `AlignmentBundle` alike, `#/graphical/timestamps/0/groupstamp:
  'timestamp_1' is a groupstamp, but its timeline 'r' is a member of no
  group and lies in none`; the same groupstamp on a child `c` of a root
  `r` that is a group's member is read and written back with
  `"groupstamp": true`, since a member's descendants lie in its group.
  Written: a bundle read with a group of `r` and `s` and a groupstamp on
  `s`, whose group then loses `s`, raises the same refusal naming
  `timestamp_1` and `s` on `to_dict`, so a manifest that breaks the rule
  is never emitted. (The shared rejection cases cover the parent of a
  member and a child of an ungrouped root.)
- **A match timestamp may be a groupstamp.** The rule above holds for a
  match timestamp as for any named timestamp, and nothing else refuses
  one. A bundle of the roots `r` and `s` (4 quarters each), both members
  of the group `g`, whose MatchStamp `matchstamp_1` joins the match
  timestamp `timestamp_1` at 1 on `r`, a groupstamp, to `timestamp_2` at 2
  on `s` by the anchor `anchor_1`, is read and written back with its
  timestamps and MatchStamps unchanged, in the complete form; the same
  bundle without the group raises `#/graphical/timestamps/0/groupstamp:
  'timestamp_1' is a groupstamp, but its timeline 'r' is a member of no
  group and lies in none` on reading.
- **A root's offset.** A seconds root with `"offset": -5.5` and a child at
  offset 2 round-trips with `root_offset == Coordinate(-5.5, seconds)`, and
  its timestamps and child positions equal those of the same root without
  the offset, so the offset is stored and changes nothing computed.
- **Selection.** A literal manifest with roots `score` (named `Score`, child
  `part` also named `Score`) and `perf` (named `Performance`), and groups
  `g1` (named `pair`) and `g2` (named `pair`):
  `Timeline.from_dict` selects `part` by `timeline_id`, `perf` by name, and
  raises for the name `Score` (two matches) and for the id `nope` (none) with
  the exact message listing the manifest's timelines; without an argument it
  raises. `TimelineGroup.from_dict` selects `g1` by id, raises for the name
  `pair` (two) and the id `nope` (none), each listing the groups; a manifest
  of one timeline and no group makes a group only when `group_id` or `name`
  is given; an anonymous one-group manifest is read with arguments naming
  nothing overriding its id and name; a named one-group manifest needs a
  selection.
- **A match selects before anything is overridden.** `Timeline.from_dict`
  reads a manifest of one root and no group as a whole when no argument is
  given; an argument that matches a timeline of the manifest, root or
  descendant, selects it; only an argument that matches none overrides the
  root's id or name, and only in a manifest of one root and no group. The
  failure this guards against is the override taken where a selection was
  meant: the root renamed to its own child's id, so that the tree holds the
  id twice. The literal is one root `ex` (named `Example`) with two
  children, `ex_child` (named `Child`, holding the region `intro`) and
  `ex_coda` (named `Example`). Each assertion separates the two readings:
  - `timeline_id="ex_child"`, `name="Child"`, and both together return the
    child — length 8, parent `ex` — and the root keeps the id `ex`, so the
    tree's timeline ids are `ex`, `ex_child`, `ex_coda`, each once. An
    override would have answered a root of length 16.
  - `timeline_id="edition"` and `name="Edition"`, matching nothing, rename
    the root and leave both children as they were; the minimal form written
    back is the literal with the root's id (or name) replaced, and nothing
    else.
  - An override is refused, with the exact message naming the holder,
    before the root is renamed, when the new id is already held in the
    manifest: by the child's region `intro`, or by the manifest's own id.
  - `name="Example"` matches the root and `ex_coda`: it raises listing the
    manifest's timelines, rather than renaming anything.
    `timeline_id="ex_child"` with `name="Edition"` matches the child by id
    and nothing by name: both must name the same timeline, so it raises.
  - An unmatched id raises in a manifest of two roots and no group, and in
    a manifest of one root and one group, each listing the timelines: the
    override is bound to both conditions, one at a time. Without an
    argument, a manifest that cannot be read as a whole raises listing its
    timelines.
  - `TimelineGroup.from_dict` overrides a group's id only in a manifest of
    one timeline and no group, or of one anonymous group. In both, a
    `group_id` held by a descendant timeline (`part`) raises naming it, so
    the group cannot take an id the manifest's tree already holds; a
    `group_id` equal to the group's own id reads the group unchanged.
  - Mixed arguments raise in both readers. As for timelines, a group's
    arguments must agree: `group_id="g"` naming the one group with
    `name="Other"` naming nothing raises `the manifest holds no group with
    id 'g', name 'Other'` listing the groups, instead of renaming a group
    that was selected.
- **Identity of the manifest.** An `AlignmentBundle()` writes no `id`; read
  from an anonymous bundle it holds a minted `bundle_1` that it never
  writes, and a manifest with a `name` and no `id` writes back exactly that.
  A given bundle id is written. A legacy `"support_policy": "clamp"` beside
  the groups is an unknown key: kept and written back, never read, so the
  bundle's `support_policy` stays `omit`.
- **`graphical.order`** is the object keyed by a group's id or by a
  timeline's id, of a root or a descendant. The list form is refused by
  the validator with `schema`, `[['score'], ['perf']] is not of type
  'object'`, and a key naming neither with `order_key`, `'g3' names no
  group or timeline of this manifest`. The keys `score` (a root) and `part` (a child) of the band-order
  manifests below are read, so a reader that looked among the roots alone
  would fail.
- **Band orders are kept, never read.** A timeline's entry of
  `graphical.order` is its band order, graphical data the library keeps as
  given: it does not check, reorder or interpret it, since both TimeToShare
  packages check it. It is kept where the manifest's styling is kept, with
  the manifest read as a whole — `Timeline.from_dict` of one root and no
  group, `TimelineGroup.from_dict` of one root or of one anonymous group,
  `AlignmentBundle.from_dict` of any manifest — and written back, unchanged,
  whenever its timeline is written, beside the group orders the writer
  computes from its groups. The literals: a root `score` (quarters, length
  `[8, 1]`) with a child `part` (offset `[2, 1]`, length `[4, 1]`), alone in
  a one-root manifest, and beside a root `perf` (seconds, `20.0`) in a
  manifest with one anonymous group `g` of `part` and `perf` displayed
  `["perf", "part"]`. Each timeline's band order is a permutation of `line`
  and the five facets that no default produces and no sort reproduces:
  `score` the format README's example (`children`, `markers`, `line`,
  `conversion_maps`, `regions`, `flow_control`), `part` every band above
  its line, in reverse default order, and `perf` every band below it, in
  reverse default order. A root's band order (`score`, `perf`) and a
  descendant's (`part`), and a member's beside a non-member's, are each
  exercised. Every reader that reads a literal as a whole (timeline and
  bundle on the one-root manifest, group and bundle on the grouped one)
  reads each literal in each form, complete and minimal, and writes it back
  in the same form equal to the literal, with the lists compared in order,
  and in the other form equal to the other literal. Mutating the input's
  lists after reading changes nothing written, so the library holds its own
  copy. A band order that is no permutation of the bands (`["line",
  "markers"]`), which the format admits, is read and written back
  unchanged.
- **A band order without its timeline is left out.** An entry is written
  when a timeline of its key's id is written, and left out otherwise. A
  key naming no written timeline makes the manifest invalid, and the
  library never rewrites an entry it keeps unread; leaving such an entry
  out keeps the written manifest valid and changes nothing else. A group read
  from the grouped literal that removes its member `perf` writes neither
  `perf` nor its band order, and keeps `score`'s and `part`'s beside the
  order `["part"]` of `g`. `Timeline.from_dict(..., timeline_id="edition")`
  of the one-root literal gives its root a new id on load: the manifest
  written holds `part`'s band order alone, neither the entry of `score` nor
  one of `edition`. A timeline selected out of a larger manifest keeps none
  of its `graphical`, styling and band orders alike:
  `Timeline.from_dict(..., timeline_id="perf")` of the grouped literal
  writes no `graphical`.
- **An unknown length.** `"length": null` reads as `length is None` and is
  written back as `null` in both forms: `length` is the one key the minimal
  form never leaves out.
- **Minimal by default.** Every writer writes the minimal form unless asked
  for the complete one: `to_dict()` of each of the three classes, and of each
  element of the one-root literal read in its complete form — the region,
  the conversion map, the jump target, the jump, the section break and the
  flow-control registry — equals that element in the minimal literal, and
  `to_dict(minimal=False)` equals it in the complete literal. The tests that
  pin a complete form ask for it explicitly.
- **The minimal form is the package's.** A default compares by value, as
  `timetoshare.minimal` compares it, never by type: a seconds timeline
  holding a float `LinearMap` of scalar `2.0` and offset `0.0` writes the
  map without `offset` in the minimal form, through the timeline's
  `to_dict()` and the map's own, and with `"offset": 0.0` in the complete
  form; the timeline's minimal form equals `minimal` of its complete form.
  The map's offset is `0.0` and its default `0`, so a writer comparing type
  and value keeps the key the package leaves out. The minimal form read
  back supplies the default as the package does: its complete form writes
  `"offset": 0`, so this map's complete form survives the minimal one in
  value, not in spelling.
- **A group is put in its form with the whole manifest.** A group object is
  written once, in the complete form, and the manifest written around it
  takes TimeToShare's form as a whole. A group without a name or meta and a
  group with both, written by `TimelineGroup.to_dict` and by
  `AlignmentBundle.to_dict`, give a minimal form equal to
  `timetoshare.minimal` of the complete form and a complete form equal to
  `timetoshare.complete` of the minimal one, and each group object equals
  its literal: `{"id", "members"}` minimal and `{"id", "name": null,
  "meta": {}, "members"}` complete for the first, every key in both forms
  for the second.

### Manifest files (`test_manifest_files.py`)

`from_file` and `to_file`, one implementation shared by `Timeline`,
`TimelineGroup` and `AlignmentBundle`, exist for one workflow: read a
manifest file, enrich it, write it back, and review the change as a diff. A
file read and written back unchanged must therefore keep its bytes, and a
change must show as its own lines. The reference texts are never the
library's own output: each is `timetoshare.dumps` of a manifest written by
hand (or of a package fixture), in the serialization and form under test,
so a writer that agreed only with itself could not pass.

The manifest's `resources` and the tables a bundle writes and reads beside
it (the MatchClaim table, the Projections table) are tested in
`../alignment/test_bundle_package.py` (its README section "A Package: the
Manifest and Its Tables"); here the fixture `bundle_resources.yaml`, whose
tables are absent, is one more byte-identical round trip.

- **Byte-identical round trips.** Every package fixture and its minimal twin,
  read by the class its directory names, is written to text with
  `timetoshare.dumps` in each of the three serializations; `from_file`
  followed by `to_file` without arguments reproduces each file byte for
  byte. Twelve texts per fixture (two forms, three serializations, with and
  without `graphical` where the fixture has one) cover the form, the styling
  and the serialization being kept independently: a writer that kept one of
  the three by accident (always YAML, always minimal, always styling) fails
  on the texts that differ in it.
- **The form is inferred from the keys at their defaults.** The form,
  styling and serialization are TimeToShare's `file_form`: a text is minimal
  when `dumps(minimal(value))` equals `dumps(value)`, so integers and floats
  are told apart (before, the two values were compared with `==`, where `0`
  equals `0.0`; no package fixture, nor any of them with one integral number
  swapped between integer and float, reads differently under the two rules,
  because canonical order respells the numbers of both sides alike). A
  minimal manifest with one key stated at its default (`"meta": {}` on a
  timeline) is not minimal: it is written back in the complete form,
  `dumps(complete(manifest))`, while the same manifest without that key is
  written back as it was. A default stated as a float of the integer default
  (`offset: 0.0` on a `LinearMap`, whose default is `0`) is a key at its
  default all the same: the text reads as complete and is written back as
  `dumps(complete(manifest))`, the float kept.
- **Writing goes through TimeToShare's `dumps_file`** with the complete
  `to_dict(minimal=False)`; the byte-identical round trips of the minimal
  twins hold only because `minimal` of the complete manifest equals the
  library's own minimal manifest for every fixture.
- **Styling is the `graphical` object.** A one-group manifest file without
  `graphical` read as a `TimelineGroup` is written back without it, although
  `to_dict()` writes the group's display order there; the same file with
  `graphical` keeps it.
- **A text that is not canonical** — JSON on one line with spaces after its
  colons — is read as `"json"` and written back as pretty JSON, the bytes
  `dumps(manifest, serialization="json")`.
- **Each argument overrides one inference.** On an object read from a
  minimal canonical YAML file with `graphical`: `minimal=False` writes
  `dumps(complete(manifest))`, `styling=False` the manifest without
  `graphical`, `serialization="jcs"` its RFC 8785 text; each leaves the
  other two as read.
- **An object built in code** is written minimal, its styling included, in
  canonical YAML: a bundle with a group writes `dumps(bundle.to_dict())`
  with its `graphical`.
- **Event rows and external references are kept as read.** A hand-written
  bundle of a recording holding two event rows (an instant and an interval)
  and one external-reference row naming the first, beside a score holding
  both keys empty, is written back byte for byte in each of the three
  serializations, and so are its two variants holding only one of the keys:
  the file's keys decide what is written, each independently, so a writer
  that dropped the rows, or wrote both keys whenever it found one, fails.
  `to_file(events=False)` on the full literal writes it without `events`.
  An object built in code with the same timelines, events and references
  writes neither key by default (the literal without them) and, with
  `events=True, external_references=True`, the full literal's bytes.
- **A drawing written is a drawing read.** A `points` page of length
  `932.2326510101616` holding a child at offset `76.70371459980326` of length
  `855.5289364103584` (built in code, accepted there because the child ends
  at the page's length in doubles) is written with `AlignmentBundle.to_file`
  and read back with `from_file`; the bundle read writes the same manifest.
  The reader used to sum exactly and refuse its own output.
- **Selection passes through.** `Timeline.from_file(path,
  timeline_id="take_1")` reads the one timeline as `Timeline.from_dict`
  selects it; `TimelineGroup.from_file(path, group_id=...)` likewise.
- **The editing workflow.** A small bundle in canonical YAML (a score, a
  recording, both ungrouped) is read with `AlignmentBundle.from_file`, a
  second recording is added, and the file is written back. The unified diff
  of the two texts (`difflib.unified_diff`) is asserted exactly: one hunk
  adding the four lines of the new timeline, nothing else changed.
- `to_file` returns the path it wrote.
- **A file is validated as it is written.** The fixture
  `discrete_logical.minimal.yaml` styles named timestamps at 0 and 480
  ticks; read and shortened to length 400, its styling names a coordinate
  beyond the end. `to_file` with the styling (as read) is refused by the
  validator (`timestamp_coordinate` at
  `#/graphical/timestamps/1/coordinate`, the message naming
  `'timestamp_2'`) and writes no file; `to_file(styling=False)` writes a
  manifest without `graphical` (keys `@context`, `meta`, `timelines`,
  `timetoshare`), which reads back with length 400. A writer validating
  the whole manifest whatever the file holds refuses the second; one
  validating nothing writes the first.

- **The Projections table is read from what Parquet reads.**
  `read_projection_table` takes the package fixture `projections.parquet`
  as a `str`, a `Path`, an `io.BytesIO` of its bytes and a
  `pyarrow.BufferReader` of them; each gives the same Projections by
  timeline as the `str` (one on `clt_2`, two on `cpt_1`, eight on
  `dpt_1`, their IDs in row order). A reader that turned every argument
  into a path string refuses the two file objects with a `TypeError`.

### Measure maps in a manifest (`test_manifest_measure_maps.py`)

A manifest holds its measure maps in the top-level `measure_maps` array, each
`{id, name?, meta?, rows}`, written sorted by id; a timeline reaches one
through a `MeasureRecordMap` among its conversion maps, and every measure
view names one by id. The library keeps one measure map per timeline; the
tests prove that the writer lifts it into `measure_maps`, that the reader
puts it back, and that every legacy measure-derived map names the right
entry, with literal manifests written from the format's rules.

- **A measure map writes its minimal entry by default.** The two-bar map
  written on its own is `{"id": "measuremap_1", "rows": [...]}`; with
  `minimal=False` it states `"name": null` and `"meta": {}` as well.
- **Writing lifts the timeline's measure map.** A quarters timeline holding
  a measure map writes one entry `measuremap_1` and a `MeasureRecordMap`
  `{"type": "MeasureRecordMap", "id": <minted cmap id>, "source_unit":
  "quarters", "target_unit": null, "measure_map": "measuremap_1"}` in its
  `conversion_maps`, sorted with the others by id; the timeline object has
  no `measure_map` and no `metric_hierarchy` key. A `whole_notes` timeline
  writes its record map inside a `ChainMap` after `LinearMap(scalar=4)` from
  `whole_notes` to `quarters`, the chain `cmap_1` and the record map
  `cmap_2` (a nested map holds an id of its own), and a ticks timeline after the `LinearMap` it
  was given (`scalar` `[1, 480]`); `add_measure_map` refuses a ticks timeline
  without one, and any unit but quarters, whole_notes and ticks, with the
  exact message.
- **Reading puts it back.** The literal manifests of both forms read to
  timelines whose `measure_map` equals the map built from the same rows,
  each written back equal to its literal. A record map given its own id
  (`bars`) and an entry given an id, a name and meta keep all of them; a
  chained record map keeps the chain's and the `LinearMap`'s ids, so a
  manifest written by hand is a fixpoint, not only one the library wrote.
- **One measure map per timeline.** A timeline holding two
  `MeasureRecordMap`s (two direct, or one direct and one as the last member
  of a chain from the timeline's unit to quarters) is refused by every
  reader with `timeline 'score' holds 2 MeasureRecordMaps ['bars',
  'bars2']: the library keeps one measure map per timeline`. A record map
  nested elsewhere (in a `CombinationMap`) is a view like any other and is
  kept as a conversion map.
- **A map converts from its timeline's unit.** A `MeasureRecordMap`
  (source `quarters`) placed directly among the conversion maps of a ticks
  timeline is refused by every reader with `Map source unit 'quarters' does
  not match timeline unit 'ticks'`, the message `add_conversion_map` gives,
  since the reader attaches every map through it; a ticks timeline reaches
  its measure map only through the chain above.
- **Several timelines may share one entry.** Two roots holding the same
  `MeasureMap` object, and two holding equal maps with minted ids, write one
  entry both record maps name; two holding unequal maps write
  `measuremap_1` and `measuremap_2`. Read back, two timelines naming one
  entry hold the same `MeasureMap` object. An entry no timeline names is
  kept and written back where the manifest is read as a whole.
- **Ids are the manifest's.** A measure map's id is unique together with
  every other id of the manifest: an entry whose id is a timeline's, a
  group's, the manifest's or a region's, and two entries of one id, are
  refused on reading with `the id '<id>' is held twice in the manifest: by
  <first> and by <second>` (the manifest first, then the measure maps, the
  trees, the groups); writing a measure map whose given id a timeline holds
  is refused with `measure map 'score' cannot be written: its id is already
  held by timeline 'score'`; a minted id is minted again around every id of
  the manifest (`measuremap_1` held by a timeline makes the entry
  `measuremap_2`). A view naming an id no entry has is refused naming it.
- **Measure-derived maps name the measure map they derive from.** On a score
  timeline, the floating-measure map the loader attaches (a
  `FloatingMeasureMap` over an equal, separately built map) names the
  timeline's own entry, so the manifest holds one entry. A
  `QuartersToFloatingMeasures`, `QuartersToMeasureNumber` and `MetricMap`
  built from the timeline's measures name the own entry too; ones built from
  other data each write an entry of their own built from their data, which
  reads back to a view converting every probe exactly as the written map
  (`MetricMap`: rows with `number` the measure count, `name` the label,
  `qstamp` the start and `actual_length` the length). What is not lossless is
  refused (see `tests/maps/README.md`).
- **The views read the measure maps they name.** Read back, every view of
  the literal manifest is rebuilt over the entry it names, the same object
  as the timeline's measure map where it names that entry, and answers the
  exact values `tests/maps/README.md` pins.
- **Unknown keys of an entry are kept.** An entry's unknown key and a row's
  unknown key are written back where they were found.
- **A volta list is a row value like any other.** A timeline of 16 quarters
  whose measure map states the brackets `[1, 3]` and `[2]` writes the
  literal entry with `"volta": [1, 3]` and `"volta": [2]` in its rows; the
  manifest is valid for the format's validator, reads back to a timeline
  whose measures state `(1, 3)` and `(2,)`, and is written back equal to
  itself. The same holds with `"volta": 1` and `"volta": 2`, which stay
  numbers: neither form is written as the other.

### Flow control (`test_flowcontrol.py`, `test_manifest_flow_control.py`)

`flow_control` is one array of elements `{id, role, coordinate, …, name?,
label?, meta?, provenance?}`, written in canonical order by coordinate and
then ID; `provenance` is a list of provenance entries. The `role` names the class: `jump` (`Jump`, with `targets`),
`jump_target` (`JumpTarget`), `break` (`Break`, with `ends`),
`section_break` (`SectionBreak`) and `barline` (`Barline`); the last four
may state a `manifestation`. A destination in `targets` is an object with
the keys `to`, `by`, `passes`, `repeat_count`, `after`, `determined`,
`reset`, `manifestation`, `label`, all optional; `ends` is an object with
`passes` and `after`. What proves the classes right is that the dictionary
form and the object are the same statement, so the tests are written from
literals of the format and never from what the writer happens to produce.

- **Every class and every key round-trips.** One literal array holds each
  of the five roles, a destination stating every key (in two destinations,
  since `to` and `by` exclude each other), both forms of `passes` (a list
  and `{from: n}`), an `after` with `to`, one with `by` and one with
  neither, an `ends` with both conditions, `provenance` of two entries, one
  of them with a belief, `name`, `label`, `meta`, and an unknown key on an
  element, on a destination, on an `after` and on an `ends`. Reading it and
  writing it in the minimal form gives the literal back; the complete form
  adds exactly `name: null`, `label: null`, `meta: {}` to each element that
  left them out and `determined: true`, `reset: true`, `label: null` to
  each destination that left them out, and nothing inside an `after` or an
  `ends`, which have no defaults. The expected complete form is the
  literal with exactly those keys added, and the format's own functions
  are the second witness: `timetoshare.complete` of a manifest holding the
  minimal array gives the array the library writes in the complete form,
  and `timetoshare.minimal` of that one the literal. A destination written
  with `determined: true`, `reset: true` and `label: null` reads as one
  that left them out, and writes them in the complete form only.
- **Reading builds the objects.** The same literal read gives, by exact
  value, each element's class, coordinate on the axis's number type,
  manifestation as a `FlowManifestation`, each destination's attributes,
  each distance as a `Duration` of the axis, `passes` as a tuple or a
  `PassesFrom`, and the provenance as a tuple of `Provenance`, one per
  entry of the list. One entry given to the constructor (a `Provenance` or
  its dictionary) is held as the list of that entry. `provenance: []`, a
  single object in place of the list and an entry with an unknown method
  are refused, the element named.
- **A manifestation outside the vocabulary is kept.** `"x-wavy-line"` on a
  jump target, a break, a section break, a barline and a destination is
  read, held as the string and written back unchanged. A word of the
  vocabulary on the wrong class (a break that is a `segno`, a destination
  that is a `coda`, a section break that is a `fine`) is refused with the
  class's own words listed.
- **The old form is refused, naming the new one.** An element with `type`
  (each of `jump_from`, `jump_to`, `flow_marker`), one with those words as
  its `role`, one with `target` and one with its own `repeat_count` is
  each refused with a message that names `role`, `targets` or the
  destination; nothing is converted. An unknown role is refused listing
  the five.
- **What cannot be played is refused**, each case breaking one rule only,
  the whole message asserted: a second Jump at a coordinate; two
  destinations of one Jump with the same `to`, the same `by` (`[1, 1]` and
  `1.0` on a float axis are one distance) or both without place; a `to`
  naming nothing; an `after` naming no Jump, the holder's own Jump, a place
  no destination of that Jump has, and no place where that Jump has two
  placed destinations; `ends: {}`; a `repeat_count` of 0, of `true` and of
  `1.5`; `passes` empty, with 0, with a repeated number, `{from: 0}`; a
  destination with both `to` and `by`; a Jump without destination. After
  each refusal by the registry its elements are what they were.
- **Nothing lies outside the axis.** A run of a flow is a stretch of the
  axis, so an element or a place a Jump goes on at that lies before 0 or
  beyond the timeline's length can only be an error; the format's
  validator refuses both (`flow_coordinate`). What is proved is that such
  flow control never comes to stand, by whichever way it arrives, each
  refusal with its whole message and the registry unchanged after it. On a
  timeline of length 12: a Jump at 4 going on `by: -8` (place −4), one
  going on `by: 20` (place 24), a Break at 20, a JumpTarget at −1, and a
  Jump whose `to` names a Region starting at 14 are each refused when
  added; the bounds themselves are places (`by` landing on 0 and on 12, an
  element at 12) and are accepted; a rejected Break at 20 is accepted,
  since a rejected element is out of every rule. On a timeline of unknown
  length only 0 bounds: the Break at 20 and the Jump reaching 24 are added
  and `reach` is 24, the Jump reaching −4 is refused. The length then
  becomes known: 12 is refused naming the first element off the axis and
  the length stays unknown; 24 is accepted (20 and 24 lie within it); a
  later 22 is refused and the length stays 24. A registry that belongs to
  no timeline knows no length: it refuses what lies before 0 and accepts
  the Break at 20, which the engine refuses when it is given a length
  (`test_flow_engine.py`).
- **Elements naming one another are added together.** A Fine naming a D.C.
  and a To Coda naming it are refused one by one in the wrong order and
  accepted by `add_all` in any order. Two elements arriving with the same
  minted ID are stored as `jump_target_1` and `jump_target_2`, and a
  registry absorbed into a timeline whose manifest holds `jump_target_1`
  has that target stored as `jump_target_2` with the destination naming it
  following.
- **Canonical order.** Elements added in reverse are held and written by
  coordinate and then ID, an integer, a fraction and a float axis each
  with values whose text order differs from their numeric order. The
  destinations of a Jump carry no order in the format, which writes them
  by place: the one without place, then those with `by`, by value, then
  those with `to`, by ID. One Jump with seven destinations (no place; `by`
  −2/3, 0 and 7/2, whose text order is not their numeric one; `to` `B`,
  `a` and `b`, an uppercase letter before every lowercase one) is built
  from every rotation of one given order and from its reverse: each is
  equal to the first, holds the places in the order spelled out, and
  writes the same object. On a float axis three `by` values are ordered
  numerically. The literal of the round trip is written in this order,
  and the conformance test proves it the format's (`canonical_order`).
- **Places.** `place_of` is asserted for every kind of place: a jump
  target; `by` forwards, backwards and 0, on an integer, a fraction (9/2 −
  1/2 = 4 exactly) and a float axis; a Region of the timeline (its start),
  a child timeline (its offset) and the timeline itself (0), all three
  through a real timeline, with a Region renamed afterwards followed by
  the destination naming it; no place without and with section breaks
  before the Jump (the nearest one before it; one at the Jump's own
  coordinate does not count).
- **The axis.** A bound registry refuses another unit naming both, and
  stores a float-authored `1.5` on a fraction axis as `Fraction(3, 2)` and
  `Fraction(1, 3)` on a float axis as `0.3333333333333333`, for the
  element's coordinate and for a destination's `by` alike. An unbound
  registry stores the identical object. A copy is equal and independent.
- **Coordinates given to the registry.** `jump_at`, the one query by
  position, answers the Jump at 9/2 of a fraction axis for the bare values
  `4.5` and `Fraction(9, 2)`, for a `Coordinate` and for an `IdCoordinate`
  in quarters, answers None at 4, and refuses a coordinate in seconds
  naming both units. The element constructors take a `Coordinate` only: a
  bare number states no unit, and `Jump(4, …)` raises `TypeError` saying
  so.
- **Exports.** The registry, the five element classes, the values a Jump
  and a Break are built from, the engine's two functions and its result
  types, and the flow `Timeline.get_flow` returns are each the same object
  in `timetoalign.timelines.flow`, `timetoalign.timelines` and
  `timetoalign`, and listed in the `__all__` of all three; the draft the
  score loaders write through is public in `timetoalign.timelines.flow`.
- **The collection is read-only.** A repair never deletes, and a rule
  checked on adding must not be bypassed by writing into the list.
  `elements` and the five views are tuples; assigning to `elements` raises
  `AttributeError`; there is no method that removes (`clear` is gone). The
  collection changes through `add`, `add_all`, `with_belief` and
  `with_beliefs` only, which the other bullets prove one by one.
- **Rejection.** An element is rejected when any entry of its provenance
  holds a belief `"false"`, whoever holds it and whichever entry: the same
  reading as the format's validators, which the test checks against
  `timetoshare.validation.flow.rejected` on the written elements. Four
  jump targets (no provenance; `unlikely` by the editor; `false` by the
  editor in the first entry; `false` by the reviewer in the second of two
  entries): `active_elements()` keeps the first two;
  `active_elements(believer=editor)` keeps all but the editor's rejection,
  so the reviewer's rejection stands for the editor; `believer=reviewer`
  keeps all but the reviewer's; `beliefs=["false"]` with the editor keeps
  exactly the one the editor rejects, `["true"]` the one the editor
  believes and `["undecided"]` the one without provenance; a word that is
  no belief state is refused with or without a believer. An element in
  which one actor holds
  `true` in one entry and `false` in another is read as rejected by that
  actor (the more critical belief).
- **A rejected element is out of every rule.** A rejected Jump and its
  replacement lie at one coordinate, in either order of adding, and
  `jump_at` answers the replacement; a third Jump there that is not
  rejected is refused. A rejected Jump may name anything (a `to` that
  names nothing), since nothing reads it. A Jump that is not rejected is
  refused when its `to` names a rejected jump target, and a Break when its
  `ends.after` names a rejected Jump, the message saying that the element
  named is rejected.
- **Recording a belief.** `registry.with_belief(id, actor, belief)` puts
  the element's copy in the element's place: same ID, coordinate and keys,
  the provenance alone changed, the object held before untouched. On an
  element without provenance the result is one entry holding the belief
  alone, `[{beliefs: [{actor, belief, date}]}]`, the form a MatchClaim
  stated without provenance gets; on an element with an entry the belief
  joins that entry beside its paradata; a second belief of the same actor
  replaces the first in place and leaves another actor's standing; with
  two entries, the entry already holding the actor's belief takes the new
  one. The date is the one given, as a string or a `datetime.date`, and an
  invalid belief or date is refused. An unknown ID raises `KeyError`
  listing the IDs.
- **What an element names is not rejected under it.** A `to` or an `after`
  naming a rejected element names nothing, and the format refuses such a
  manifest; so the rejection itself is refused, at the call that would
  cause it, naming every element that names the one to be rejected. The
  case is a repeat with a Fine: a jump target `start`, a Jump `printed`
  going on at it, a Break `fine` ending after `printed`. Rejecting `start`
  alone is refused naming `printed`; rejecting `printed` alone is refused
  naming `fine`; rejecting `start` and `printed` in one step is still
  refused naming `fine`; the registry is unchanged each time. All three in
  one step (`with_beliefs`) are rejected, returned in the order given, the
  canonical order of the registry untouched. A target two Jumps go on at
  is refused naming both. An element named only by a rejected element is
  rejected alone. Beliefs that reject nothing (`unlikely` on several
  elements in one step) are recorded without any check failing. Then the
  repair is finished by adding the replacements, and taking a rejection
  back is checked like an addition: `printed` believed `true` again while
  its replacement lies at the coordinate is refused (a second Jump), and
  `fine` believed `true` again is refused because the Jump it names is
  still rejected; the registry holds what it held. An unknown ID among
  several raises `KeyError` and records nothing; a single string given to
  `with_beliefs` is refused as no list of IDs.
- **Provenance entries as the format admits them.** The format's entry is
  an open object whose keys are all optional. Three entries that
  `timetoshare.validate` accepts and that a reader built for the known
  keys alone would refuse lie on one jump target of a whole manifest: an
  entry stating nothing (`{}`), an entry with keys the library does not
  know (one of them null), and a belief with a key it does not know. The
  manifest is read by `Timeline.from_dict`; the element holds three
  entries, its known keys read as on any entry (the method, the actor's
  belief `unlikely`, which does not reject it); written back, the
  `flow_control` array equals the one read, and the manifest written
  validates. A belief the editor records afterwards replaces the editor's
  own in the third entry (the belief replaced goes with its keys) and the
  other two entries stand as read. What the library knows stays checked:
  an entry with an unknown key and an invalid method is refused naming the
  method. The entries of a MatchClaim are untouched by this: the same
  three entries are refused by `Provenance.from_dict`, which the test
  asserts beside it.
- **Score loaders.** The elements a loader writes from the ms3 measures of
  Couperin's `c05n05_musete` (class, ID, coordinate and name of each of the
  nine) are written as the `flow_control` array and read back to an equal
  registry. What each element states, and from which column of the table,
  is the subject of `tests/loader/score/test_flow_gold.py`.

Whole manifests holding flow control are proved by
`tests/test_timetoshare_conformance.py`: every fixture of the format, the
ten of its directory `flow/` among them, is read by `Timeline.from_dict` or
the class of its directory and written back byte for byte in both forms,
and every rejection case of the format is refused with its code and path.

### The flow engine (`test_flow_engine.py`)

`play(registry, length, convention=…)` returns the flow as runs of the
axis and its findings; `atomic_sections` the stretches flow control never
divides. The engine reads places and conditions only, so a test proves it
right by stating a piece of music as elements, deriving by hand what a
player plays, and comparing run for run. No expected value comes from the
engine: each one below is derived from the rules in the order the walk
applies them, and where a published flow exists (the gold files in
`target_flows`) the engine is compared with the file. Synthetic cases use
an integer axis and name stretches by letters of four units (A = [0, 4),
B = [4, 8), …) so that the played order can be read off the runs.

**One rule per case** (runs in played order; a run that goes on where the
one before stopped is merged with it):

| Case | Elements | Runs |
|---|---|---|
| repeat without start | Jump 8 `{repeat_count: 1}`; length 12 | [0,8) [0,12) |
| repeat with start | target 4; Jump 8 to it, once | [0,8) [4,12) |
| `repeat_count: 3` | Jump 4, length 8 | [0,4) [0,4) [0,4) [0,8) |
| nested repeats | target 4; Jump 8 to it once; Jump 12 `{repeat_count: 1}`; length 16 | [0,8) [4,12) [0,8) [4,16): the outer repeat restores the inner Jump, which lies wholly inside [0, 12) |
| brackets 1, 2 | Jump 4 `{by: 0, passes: [1]}`, `{to: 8, passes: [2]}`; Break 8; Jump 8 once; length 12 | [0,8) [0,4) [8,12) |
| three brackets | Jump 4 with passes 1, 2, 3 to 4, 8, 12; Breaks 8, 12; Jumps 8 and 12 once; length 16 | [0,8) [0,4) [8,12) [0,4) [12,16) |
| brackets (1, 3), (2) | Jump 4 `{by: 0, passes: [1, 3]}`, `{to: 8, passes: [2]}`; Break 8; Jump 8 `{repeat_count: 2}`; Jump 12 once; length 16 | [0,8) [0,4) [8,12) [0,8) [0,4) [8,16), see below |
| first bracket without sign | brackets 1, 2 without the Jump at 8 | [0,8); `flow_ends_at_break` at 8, `never_played` [8,12) |
| brackets (1, 3), (2) ending with a Fine | Jump 4 `{by: 0, passes: [1, 3]}`, `{to: 8, passes: [2]}`; Break 8; Jump 8 once; Jump 12 `{repeat_count: 1, reset: false}`; a second Break at 8 ending after the Jump at 12; length 12 | [0,8) [0,4) [8,12) [0,8), no finding, see below |
| D.C. al Fine | Break 4 ending after the Jump at 8 `{repeat_count: 1, reset: false}`; length 8 | [0,8) [0,4) |
| D.S. al Coda | segno 4; Jump 8 to the coda at 16 after the D.S.; Jump 16 to the segno, once, no reset; length 20 | [0,16) [4,8) [16,20) |
| repeat end and D.S. on one barline | segno 4; Jump 8 `{repeat_count: 1}` and `{to: segno, repeat_count: 1, reset: false}` | [0,8) [0,8) [4,8): the restoring destination first |
| `A Fine B :‖ D.C. al Fine` on one barline | repeat start 0; Jump 8 to it once and `{repeat_count: 1, reset: false}`; Break 4 ending after that Jump's placeless destination | [0,8) [0,8) [0,4) |
| D.C. with `reset` at its default | Jump 4 once; Jump 8 `{repeat_count: 1}` | [0,4) [0,8) [0,4) [0,8): the repeat is played again |
| loop on a seconds axis | Jump 2.5 `{repeat_count: 127}`; length 4.0 | 127 × [0.0, 2.5), then [0.0, 4.0), every bound a float |
| ossia | Jump 4 `{by: 0}` and `{to: 8, determined: false}`; Jump 8 to 12; Jump 12 to 12; length 16 | [0,8) [12,16); `undetermined_target` at 4, `never_played` [8,12) |
| endless loop | Jump 4 `{}`; length 8 | [0,4) [0,4); `endless_loop` at 4, `never_played` [4,8): the second take meets the state of the first |
| two sections, D.C. in the second | SectionBreak 8; Jump 16 `{repeat_count: 1, reset: false}`; length 16 | [0,16) [8,16) |
| a Fine ending the first section | Break 4 ending after the Jump at 8; Jump 8 `{repeat_count: 1, reset: false}`; SectionBreak 8; length 12 | [0,8) [0,4) [8,12): the D.C. lands at 0, since the section break at its own coordinate ends its section, and the flow begins again at the section break behind the Fine |
| barline | Barline 4; length 8 | [0,8), one atomic section |
| two destinations of one kind | Jump 4 to targets at 6 and 8, unconditional; length 10 | [0,4) [6,10); `ambiguous_targets` at 4, `never_played` [4,6) |

The brackets (1, 3), (2), by the rules as they stand: pass 1 plays [0, 8)
and the repeat at 8 is taken (1 of 2); pass 2 goes to the second bracket,
whose repeat at 12 goes back to 0 and restores every Jump and Break lying
wholly inside [0, 12): the bracket Jump at 4 (its places 4 and 8) and the
Jump at 8. The bracket Jump is therefore arrived at for the first time
again, plays the first bracket, the restored repeat at 8 is taken, and the
fourth time through enters the second bracket and goes on to the end. A
"third pass" numbered 3 is never reached. The test asserts this reading and
is the place to change when the rule changes.

The same brackets ending with a Fine are the encoding that reaches pass 3:
the repeat sign at the end of the second bracket does not restore
(`reset: false`), so the bracket Jump keeps its count. Pass 1 plays [0, 8)
and the repeat at 8 is taken; pass 2 goes to the second bracket [8, 12),
whose sign goes back to 0 and raises the flag the Fine waits for; pass 3
goes on into the first bracket and plays up to 8, where the repeat is used
up and two Breaks lie: the one between the brackets, which has no `ends`,
and the Fine, whose `ends` holds. The flow ends there, and because a Break
with `ends` holds at that coordinate it ends regularly: no
`flow_ends_at_break`. The same elements without the Fine end at the same
place with `flow_ends_at_break` at 8, which is asserted beside it, so the
finding is shown to depend on the Fine alone. `single` arrives at the
bracket Jump on pass 2 (the first bracket's repeat restores, the second's
does not count) and plays [0,4) [8,12).

`single` and `printed` are asserted for the three bracket cases: `printed`
is [0, length) for each. `single` arrives at the bracket Jump on pass 1 +
the `repeat_count` of every restoring destination whose stretch contains
it: pass 2 for brackets 1, 2 ([0,4) [8,12)); pass 3 for three brackets
([0,4) [12,16)); pass 4 for (1, 3), (2), which is beyond the highest pass
named, so the destination naming 3 holds, which is going on: [0,8), ending
at the Break at 8. Findings are a result of `default` alone: `single` and
`printed` return none in every case, also where `single` ends at a Break
without condition as here.

**Three pieces**, on an integer axis where position = measure count − 1
for the first two:

1. *Musette* (length 58; segno 5; repeats ending at 16, 31, 58; a D.S. at
   31 and at 58; at 16 a Fine after the second D.S. and a jump to the
   second ending after the first). Trace: [0,16), repeat to 0, [0,16) and
   on to 31; repeat to 16 (the Fine is restored, the Jump at 16 is not: its
   places 0 and 31 lie outside), [16,31); first D.S. to 5, [5,16), where
   the destination after the first D.S. holds and goes to 31; [31,58),
   repeat to 31, [31,58); second D.S. to 5, [5,16), where the Fine now
   ends the flow. Runs [0,16) [0,31) [16,31) [5,16) [31,58) [31,58) [5,16),
   no finding. `single`: [0,58). Atomic sections [0,5) [5,16) [16,31)
   [31,58).
2. *Rondeau* (length 60; the refrain [0, 9) played after each couplet):
   runs [0,9) [0,18) [9,18) [0,9) [18,27) [18,27) [0,9) [27,60) [27,60);
   `single` [0,60).
3. *The stress score* (quarters, length 37, fifteen measures starting at 0,
   1, 5, 6, 7, 9, 10, 12, 16, 20, 24, 28, 30, 32, 34), stated twice.
   (a) As the source prints it: the repeat ends at 16 and 34 have no
   start, and nothing stands at the end of the second of three brackets.
   The flow plays [0,6), repeats, plays on to 9 through the first bracket,
   repeats from 6, enters the second bracket [9,10) and meets the Break at
   10 with no Jump: runs [0,6) [0,9) [6,7) [9,10), `flow_ends_at_break` at
   10, `never_played` [10,37). (b) With three repair elements stated as
   such (jump targets at 12 and 30 that the repeat ends at 16 and 34 name,
   and a Jump at 10 to the target at 12): the default flow equals the
   `default` rows of the gold file
   `out_of_the_flow_experience-flow_only.flow.csv` measure for measure
   (1-3, 1-5, 4, 6, 8, 8-10, 10-11, 9-10, 10-12, 9, 13-14, 13, 15, 1-4, 7),
   `single` equals its `single` rows (1-4, 6, 8-13, 15), and the atomic
   sections equal its `atomic` rows, with no finding. The gold rows are
   read from the file and turned into quarters by the measure starts
   above; nothing is copied from the engine.

**A Break's passes are exactly those arrivals.** Among the destinations of
a Jump the one naming the highest pass listed also holds at every later
arrival; a Break has no such rule, or a Break ending the first arrival
only could not be stated. The case that tells the two readings apart must
bring the flow to the Break a second time without restoring it: a jump
target at 0, a Break at 4 with `ends: {passes: [1]}`, a section break at
6, a Jump at 8 going back to 0 once with `reset: false`; length 12.
Arrival 1 ends the flow at 4, which begins again at the section break:
[0,4) [6,8). The Jump goes back; arrival 2 is not listed, so the flow
goes on across the Break to the end, the Jump being used up: [0,12). Runs
[0,4) [6,8) [0,12), no finding (a Break with `ends` ended the flow, and
[4,6) is played on the way back). Read with the Jump's rule the Break
would hold again and the runs would be [0,4) [6,8) [0,4) [6,12). `single`
arrives once: [0,4) [6,12). With `ends: {passes: {from: 2}}` in the same
piece the first arrival goes on and the second ends: [0,8) [0,4) [6,12);
`single` arrives once and goes on: [0,12). The Jump's rule is asserted
beside it on the same list: a Jump at 4 going on at a target at 8 with
`passes: [1]`, and a Jump at 12 going back to 0 once with `reset: false`;
length 16. Arrival 1 jumps to 8, the flow goes back at 12, and arrival 2,
beyond the highest pass listed, jumps again: [0,4) [8,12) [0,4) [8,16),
`never_played` [4,8).

**Nothing is played outside the axis.** The engine is given a length and
refuses, before walking, an element read that lies beyond it and a
destination whose place does, in the registry's words, the whole message
asserted for `play` and for `atomic_sections`: on a registry that belongs
to no timeline (which knows no length when elements are added), a Jump at
4 going on `by: 20` with length 12, a Break at 20 with length 12, and a
Jump going on at a jump target at 20 with length 12. With length 24 the
first is played: [0,4), the place 24 being the end, `never_played`
[4,24). A rejected Break at 20 is not read and does not stand in the way.

**Rejected elements.** Without a believer the engine plays the elements no
actor rejects: a rejected Jump once at 8 and its replacement twice at 8
play [0,8) [0,8) [0,12), with or without naming the believer who rejected
it. For a believer who rejects neither, both Jumps are read and the walk
is refused in the registry's words for a second Jump at a coordinate. The
registry never lets an element stand that names a rejected one, so a name
can dangle only in a reading narrowed by a belief filter: read for the
editor's `true` beliefs alone, a Jump the editor believes whose `to` names
a jump target the editor has not decided on, and a Break the editor
believes whose `ends.after` names such a Jump, name nothing. `play` and
`atomic_sections` raise `ValueError` with the message the registry
refuses such an element with when it is added, the whole message
asserted, ending in the statement that the element named is left out by
the belief filter; read without a believer both flows are played.

**The length given.** `play` and `atomic_sections` take the length as a
bare value, a `Coordinate` or an `IdCoordinate` on the registry's axis and
play the same runs for each.

### The format's expected flows (`test_flow_expected.py`)

TimeToShare ships, for each of its flow fixtures with flow control, the
flow that follows from it: `fixtures/flows/<version>/<name>.yaml` beside
the trace it was derived by hand with. The files name a manifest and a
timeline and state the atomic sections and, for `default`, `single` and
`printed`, the runs and the letters of the sections they cover; for
`default` also the findings. They were written by another hand than the
engine, from the same rules, so agreement is evidence for both and a
difference is a defect of one of them, to be decided by the rules.

The test reads each named manifest with `AlignmentBundle.from_dict`, takes
the named timeline's flow control and length and compares, exactly:

- the atomic sections (name, start, end, in the file's spelling of the
  axis's number type);
- for each convention the runs. The files list one run per stretch up to a
  Jump, a Break or the length; the engine joins a run to the one before
  where it goes on from it, so the file's runs are joined by that one rule
  before comparing. So that the joining cannot hide a difference, the
  letters of the atomic sections the engine's runs cover, in order, are
  compared with the file's `sections` as well;
- for `default` the findings as kind and coordinate, in the file's order.
  The files state no findings for `single` and `printed`, and none is
  compared there.

The nine case names are asserted, so that a directory not found cannot
pass as nine cases agreeing. A case whose file is known to differ from the
rules is listed with the exact difference and runs as a strict expected
failure, never skipped: at present none.

### Measures and flow control (`test_flow_measures.py`)

`timelines/flow/measures.py` holds the one translation between a measure
map and flow control, the statement of a flow in measure counts, the reader
of `.flow.csv` files and one check of the measures.

**What would prove the translation right.** The translation is a set of
rules, each of which says what one fact of a row becomes. A test that ran a
real score through it would prove only that the rules agree with themselves
on that score. So every rule gets a measure map made for it, small enough
that the expected elements can be written down by hand from the rule and
the bar positions alone: 4/4 bars, measure count *n* starting at `4 (n - 1)`
quarters. The expected value is the whole `to_dict()` of every element, IDs
included, so an element too many, a key too many or a default written out
fails as surely as a wrong place. The cases:

| Case | Proves |
|---|---|
| no sign | nothing is invented |
| a repeat start | a `jump_target` (`repeat_start`) at the row's start |
| start and end | the destination `to` the start, `repeat_count: 1`, at the row's END |
| an end without a start | no place, and no repeat start added |
| nested repeats | the nearest open start is closed first |
| `start … end … end` | a start once closed serves no later repeat end |
| start and end on one measure | the measure repeats itself |
| brackets 1, 2 | ONE Jump at the first bracket's start (`by: 0`, then `to`), one landing, one break, the repeat end on its own Jump |
| three brackets, ends on 1 and 2 | both ends name one start; the second is `reset: false`, the first is not |
| a bracket of two measures | adjacent rows of equal `volta` are one bracket: one break, behind it |
| `volta` as a list (a measure map, and the rows of `MeasureMap.expand_rows`) | `passes` is the list; `[1, 3]` and `[2]` are one run; map and rows give the same elements |
| brackets 1, 2, 1, 2 | a bracket naming a pass again begins the next run: two Jumps |
| a repeat end where the brackets begin | signs on one coordinate are destinations of one Jump, in the order of the rows |
| 3/8 bars | coordinates are the exact positions, `[3, 2]` and `[9, 2]` |
| the elements in a registry, played | the IDs given serve as references; A B, A C comes out |
| an abstract map; a `volta` that is no number | refused, with the count named |

The list form is tested on both inputs the function reads: a measure map,
whose measures answer with their pass numbers whichever form they state,
and full rows.

**The reverse** (`measure_keys_from_flow_control`) is proved by going there
and back: a map with a repeat, three brackets and a second repeat gives its
own three keys again for every measure. Two tests show what cannot come
back. Where the last bracket ends is no fact of flow control: with rows that
state a two-measure second bracket the keys cover both measures, with the
same positions and no `volta` they cover the first only. Several passes
come back as a list (`[1, 3]`), one pass as a number: flow control states
passes, not whether a source wrote one pass as `2` or as `[2]`, so a map
whose second bracket states `[2]` reads back `2` there. And the keys are
read from manifestations: a jump target without one, a destination without
one and a repeat start lying inside a measure state nothing, so a repaired
or hand-written registry does not turn into notation it does not state.

**Measure counts.** A played flow and its atomic sections become ranges
with the letters of the sections they cover (`A B`, `A`, `C` for brackets
1, 2); a run is divided at the coordinates asked for and nowhere else,
which is what the rows of a `.flow.csv` file do at a section break; a
stretch beginning or ending inside a measure, or outside the measures, has
no counts and is refused with the measure named.

**The file reader** is tested on a file written by the test (two modes this
library plays, one of another tool, one `ERROR` row, a seventh column) and
on the stress score's file, whose `ms3` and `music21` rows must stay
readable though nothing here computes them. A file lacking a column is
refused naming the columns it has.

**The order a source states.** `measure_order_from_successors` reads the
successors a measure table lists for each measure (`next`) as their format
defines them: the n-th arrival at a measure goes on to its n-th successor,
the last one once they are used up, and the walk ends at a measure without
successors or at a successor that is no measure (`-1`). The rule is proved
on successors written by hand, each with the order worked out on paper: a
repeat (`1 → 2`, `2 → 1, 3`: 1 2 1 2 3), brackets (`1 → 2, 3`, `2 → 1`: 1 2
1 3), a da capo al fine (`1 → 2, -1`, `2 → 1`: 1 2 1), the last successor
serving again (`3 → 2, -1` after a repeat of 1 2: 1 2 1 2 3 2 3), a walk
beginning at another measure than the first, and printed order. Successors
that never end (`1 → 2`, `2 → 1`) are refused naming the measure; so is a
first measure that is none. An arbitrary step limit would pass or cut a
long piece by its size; the refusal rests on the state instead (a measure
arrived at again with its successors used up and no other successor used
up since).

`Timeline.create_flow_map_from_measure_order(order)` writes such an order
as a FlowMap. On three 4/4 bars with a repeat end at 8 (the engine's
default flow is [0, 8) [0, 12)):

| Order | Regions | Chunks | `differs_at` |
|---|---|---|---|
| 1 2 1 2 3 | [0, 8), [0, 12) | both, in order | None |
| 1 2 1 2 | [0, 8) | the one Region twice | 16 (the default flow goes on with bar 3) |
| 1 2 3 | [0, 12) | one | 8 (the default flow goes back) |
| 1 3 | [0, 4), [8, 12) | both | 4 |
| 3 1 | [8, 12), [0, 4) | both | 0 (the first stretch begins elsewhere) |

A stretch is a run of measures consecutive in the measure map; a stretch
played twice is one Region. The FlowMap names no convention, its
provenance is `[{method: imported}]`, and it is not a computed one
(`is_computed` false), so computing the flows from flow control afterwards
leaves it standing. Where `differs_at` is a position, exactly one warning
is logged, its text asserted; where it is None, none. The position is
where the two flows, read as stretches played end to end, stop being the
same, so it is checkable by adding lengths. Refused, with the timeline
holding the Regions and FlowMaps it held before: a count that is no
measure, an empty order, a child timeline, a timeline without a measure
map, and a given FlowMap ID that is already held (the Regions written for
it are taken out again).

**The finding.** `repeat_end_findings` returns a repeat end without a
printed start when the last sign before it that closes anything is a last
bracket. One positive case (`|: A |1 B :|2 C | D :|`: count 4) and five
that must stay silent or find only the first: a printed start, a repeat end
before it, nothing before it, the repeat end of its own bracket, and a
second repeat end behind the first found one. The positive case also
asserts that the translation writes the repeat end without a place: a
finding changes nothing.

### Manifest identity (`test_manifest_ids.py`)

Every element of a manifest (a timeline, a conversion map, a region, a
flow-control element of any of the five roles) carries an `id` for
identification and a `name` for display. The tests prove four properties, each against literal expected ids.

- **One form, one generator, no process state.** An element constructed
  without an id is minted `{prefix}_{N}` with N above the highest number
  an id of its scope holds under that prefix (never a gap's number, so no
  id is given again). A standalone element's scope is empty, so each
  class answers `_1`: the six timeline classes `clt_1`, `dlt_1`, `cpt_1`,
  `dpt_1`, `cgt_1`, `dgt_1`; the base `Timeline` and a segment line by their
  unit (`Timeline(unit="seconds")` is `cpt_1`, ticks `dlt_1`, pixels `dgt_1`);
  `LinearMap` `lmap_1`, `TableMap` and its subclasses `tmap_1`, `PiecewiseMap`
  `pmap_1`, every other map `cmap_1`; `region_1`; and a flow-control
  element by its `role`: a `Break` `break_1`, a `SectionBreak`
  `section_break_1`, a `Barline` `barline_1`, a `Jump` `jump_1`, a
  `JumpTarget` `jump_target_1`. Determinism is proved by building one reference manifest
  (a quarters root, two children made by `create_child`, two regions, two
  breaks, a jump target and a jump naming it, a `LinearMap`) twice, the second
  time after constructing five unrelated timelines and four maps, and once
  more with its collections added in the reverse order. The first two must
  serialize to identical `to_dict` payloads, and all three must carry
  exactly the ids spelled out literally (`clt_1`; children `clt_2`,
  `clt_3`; `region_1`, `region_2`; `break_1`, `break_2`;
  `jump_target_1`; `jump_1` going on at `jump_target_1`; `lmap_1`). A
  process-global counter gives
  different ids in the second build; insertion order may decide which
  element holds which id, never which ids exist. Minting above the highest
  is pinned by a given child id `clt_3` on a `clt_1` root: the next two
  `create_child` calls mint `clt_4`, then `clt_5`, never `clt_2`. No minted timeline id
  starts with `tl:` and none lacks the `_` separator.
- **Whatever is given is the id.** Hand-chosen ids (`performance`, `a`,
  `Bass Kick`, `score:clt_7`, `score/1`, `part 2`,
  `https://example.org/op35#m1`) are kept verbatim on timelines, maps,
  regions, breaks, jumps and jump targets, and survive `to_dict` → JSON →
  `from_dict` → `to_dict` text-identically; a minted manifest round-trips
  its minted ids verbatim too. A manifest whose own id, roots, child,
  group, measure map, conversion map, region and three flow-control
  elements carry ids with slashes, spaces, colons and URIs is read by
  `AlignmentBundle.from_dict`, every element answers its id, and the
  manifest is written back equal to what was read and text-identically
  through JSON. Refused ids raise `ValueError` on every kind: the empty
  string and any containing a control character (`"a\tb"`, `"a\nb"`), on
  each of the five flow-control classes too;
  `a/b` and `Bass Kick` are accepted.
- **Collisions within one manifest, across all kinds.** A manifest is a root
  timeline and all its descendants. When an element joins it, a minted id
  that is taken is re-minted in the manifest's scope, and a given id that is
  taken raises `ValueError` naming both elements. Specimens: two standalone
  quarters timelines, each `clt_1` with a region `region_1`; adding the
  second as a child of the first renames the child `clt_2` (its key in
  `list_children()` and its segment row both say `clt_2`) and its region
  `region_2`, while the root keeps `clt_1` and `region_1`. A grandchild is
  renamed under its own parent's key as well (the nested specimen: the child
  carries a minted child `clt_2` of its own, which becomes `clt_3`). A child
  whose jump target `jump_target_1` collides is re-minted `jump_target_2`
  and the `to` of its jump's destination follows to `jump_target_2`. Given ids: a child `score` under a root
  `score` raises naming `timeline 'score'` twice; a child timeline `intro`
  under a root holding a region `intro` raises naming `region 'intro'`; a
  region `intro` added beside a child `intro` raises; adding a given map id
  already held by a child's map raises. Each raise is atomic: the parent
  keeps its children, regions and maps, and the joining child keeps its id.
- **Every id of a manifest read is unique across the manifest.** Within one
  tree the joins above refuse a repeated id while the tree is read; across
  trees, groups, measure maps and the manifest's own id, `Manifest.read`
  refuses the second holder of an id with exactly `the id '<id>' is held
  twice in the manifest: by <first> and by <second>`, holders in reading
  order (the manifest, the measure maps, each root's tree, the groups).
  One specimen per pair of kinds, in two roots `score` and `perf` each of
  which `Timeline.from_dict` reads on its own, so that only the pair is
  refused:
  - a region `intro` in each root: `by region 'intro' of timeline 'score'
    and by region 'intro' of timeline 'perf'`;
  - a conversion map `m` on `score` and a break `m` on `perf`: `by
    conversion map 'm' of timeline 'score' and by break 'm' of
    timeline 'perf'`;
  - a root `perf` and a child `perf` of `score`: `by timeline 'perf' and by
    timeline 'perf'`;
  - a group `g` and a region `g` of `perf`: `by region 'g' of timeline
    'perf' and by group 'g'`;
  - a measure map `bars` and a jump target `bars` of `perf`: `by measure map
    'bars' and by jump_target 'bars' of timeline 'perf'`;
  - the manifest id `op35` and a map `op35` nested in a `ChainMap` `chain`
    of `score`: `by the manifest and by conversion map 'op35' nested in
    conversion map 'chain' of timeline 'score'`.
- **Trees joining a group or a bundle, every element kind.** Two trees
  built alike — a root of given id (`a`, `b`) holding a minted region
  `region_1`, a minted `LinearMap` `lmap_1`, a minted jump target
  `jump_target_1` and a given jump naming it — join one scope: in
  `TimelineGroup(timelines=[a, b])` and in an `AlignmentBundle` holding
  both ungrouped, `a` keeps every id and `b`'s become `region_2`, `lmap_2`
  and `jump_target_2`, the `to` of its jump's destination following to
  `jump_target_2`. A given id
  refuses: `b` holding a region `intro` where `a` holds a map `intro` is
  refused with `region 'intro' of timeline 'b' cannot join group
  'group_1': its id 'intro' is already held by conversion map 'intro' of
  timeline 'a'` (`bundle 'bundle_1'` in a bundle), and atomically — `b`'s
  minted `region_1`, which would have been minted again, is still
  `region_1`, and the group or bundle still holds `a` alone. A bundle's
  given id is the manifest's: a timeline `study` joining the bundle
  `study` is refused with `timeline 'study' cannot join bundle 'study':
  its id 'study' is already held by bundle 'study'`, and a group of given
  id `g` joining a bundle where a region `g` lies is refused with `group
  'g' cannot join bundle 'bundle_1': its id 'g' is already held by region
  'g' of timeline 'a'`.
- **The scope holds while the group or bundle lives.** In a group of `a`
  (holding `region_1`) and `b`, `b.create_region(...)` mints `region_2`;
  `b.add_region(Region(..., id="intro"))` where `a` holds a region `intro`
  is refused with `region 'intro' of timeline 'b' cannot join: its id
  'intro' is already held by region 'intro' of timeline 'a'`, a given map
  `intro` with `conversion map 'intro' of timeline 'b' cannot join: …` and
  a given break `intro` with `break 'intro' of timeline 'b' cannot
  join: …`, each message compared whole; `b` holds no region, map or break
  `intro` afterwards. The same holds in a bundle holding `a` and `b`
  ungrouped. Once `b` is removed from the group (`remove_timeline`), or
  the group is dropped and collected, `b.create_region(...)` mints
  `region_1` again: the group no longer joins the two trees. A group
  pickled and unpickled keeps the scope (`b.create_region` on the
  restored members mints `region_2`). A given id that takes a minted group
  id mints the group again: in a group `group_1` of `a` and `b`,
  `b.create_region(..., id="group_1")` keeps the region's id and the group
  becomes `group_2`; in a bundle holding that group it is re-keyed
  (`list(bundle.groups) == ["group_2"]`).
- **A group joined to a bundle admits its newcomers into the bundle.** A
  bundle holds every member of its groups, so a timeline added to a group
  afterwards (`TimelineGroup.add_timeline`) joins the bundle as well, its
  tree admitted into the group and the bundle together. Specimen: a bundle
  holding `d` ungrouped, then a group of `a` and `b` (`add_group`), then
  `c` added to the group, where `c` and `d` each hold a region. With both
  regions given `r`, the add is refused with `region 'r' of timeline 'c'
  cannot join group 'group_1': its id 'r' is already held by region 'r' of
  timeline 'd'`, and atomically: the group still holds `a`, `b`, the
  bundle `a`, `b`, `d`, and `c` keeps its region `r` and joins no scope:
  a region of id `d`, the id of the bundle's timeline, is created on `c`
  afterwards without refusal. With both regions minted (`region_1`), `c`'s is minted again
  `region_2`, `d` keeps `region_1`, the bundle registers `c`, and
  `bundle.to_dict()` writes the roots `a`, `b`, `c`, `d` and the group
  with members `a`, `b`, `c`, which the TimeToShare validator accepts.
  Without the bundle's admission the manifest would name `c` as a member
  of no root and hold `region_1` twice. The bundle is found from the
  group itself, not from its members, so an **empty** group behaves
  alike: a bundle holding `d` (region `r`) and the empty group `group_1`
  refuses `c` (region `r`) with the same message, the group staying empty
  and the bundle holding `d` alone; with minted regions `c`'s becomes
  `region_2` and `bundle.to_dict()` writes the roots `c`, `d` and the
  group with member `c`, valid. A bundle pickled and unpickled holds its
  group again: the restored group refuses the restored `c` alike. Once the
  bundle no longer holds the group (removed from `bundle.groups`) or is
  deleted and collected, the group consults it no more: `c` with region
  `r` joins the group and keeps `r`, and the bundle does not register it.
- **Messages name an element by its id in single quotes.** An id is
  quoted as it is, never as its `repr`: a region `Bach's` given twice on
  one timeline `t` is refused with `region 'Bach's' of timeline 't' cannot
  join: its id 'Bach's' is already held by region 'Bach's' of timeline
  't'` (a `repr` would write `"Bach's"`).
- **Maps nested in maps share the scope.** A `ChainMap`, `PiecewiseMap`,
  `CombinationMap` (and an `IntervalToConstantMap` with an end, whose form
  nests a `ConstantMap` per interval) writes its nested maps with ids of
  their own, so a nested id is held like any other: within the composite's
  own tree when it is built, within the manifest when it joins. The ids
  claim in written order (the composite, then its nested maps depth first);
  a minted id another holds is minted again, a given one is kept or
  refused. What the tests prove, each against literal ids:
  - a `ChainMap` of two unnamed `LinearMap`s writes `cmap_1` holding
    `lmap_1`, `lmap_2`; added to a timeline already holding a `LinearMap`
    `lmap_1`, its members become `lmap_3`, `lmap_2` (the first was taken,
    the second was not), and the manifest writes each id once;
  - a `CombinationMap` of a `ChainMap` (a whole-notes-to-quarters
    `LinearMap`, then a `MeasureRecordMap`) under `measure` and a
    `ConstantMap` under `section`: built, the record map is `cmap_2` (the
    chain holds `cmap_1`), the chain `cmap_3` and the constant `cmap_4` (the
    combination holds `cmap_1`); added to a whole-notes timeline already
    holding a `ConstantMap` `cmap_1`, the combination becomes `cmap_5` and
    the nested maps keep theirs. Its manifest round-trips text-identically
    through JSON, so the reader keeps every nested id it reads;
  - an `IntervalToConstantMap` `cmap_1` with an end writes its two
    intervals' `ConstantMap`s as `cmap_2`, `cmap_3`;
  - a child whose `ChainMap` holds `lmap_1`, `lmap_2` joins a root holding
    `lmap_1`: the nested `lmap_1` is re-minted `lmap_3` in the child;
  - refusals, each atomic (the timeline's maps and the joining map's ids
    unchanged): two nested maps given one id; a nested given id that a top
    map of the manifest holds; one map object nested in two places (its one
    id would be written twice); and a manifest whose `ChainMap` nests two
    maps `lmap_1`, which `Timeline.from_dict` refuses as the TimeToShare
    validators do.
- **Regions and jump targets are keyed by id; a destination names a jump
  target by its id.** Two
  regions named `Chorus` coexist as `region_1` and `region_2`; `get_region`,
  `has_region`, `list_regions`, `create_child_from_region` and
  `create_children_from_regions` take ids, and `get_region("Chorus")` raises
  `KeyError` listing the available ids. `create_child_from_region` names the
  child after the region (`name == "Chorus"`) and mints its id (`clt_2`),
  never reusing the region's id. Adding a jump whose destination's `to`
  names no jump target of the registry and no place of the timeline is
  refused, the whole message asserted, and nothing is added. A score loader
  resolves a D.S. to the segno's id: the ms3 measures of Couperin's
  `c05n05_musete` (corpus `score`, `couperin_concerts`) load the segno as
  the jump target `jump_target_3`, named `segno`, at quarter 12 (IDs are
  minted in the order written: the two printed repeat starts first, then
  the table's markers `segno`, `1e`, `2e`); its Fine is a break at the end
  of its bar, 42, and no jump target; and both D.S. al Fine destinations
  name `jump_target_3`. In the Rondeau `c11n08_Rondeau` the two da capo
  destinations (at 24 and 36) and the first repeat end (at 12, which has
  no printed start) name no place, each other repeat end names the repeat
  start printed for it (`jump_target_1` to `jump_target_3`), and the two
  destinations that go on to a couplet after a da capo (at 12, without a
  manifestation) name the markers `2e` and `3e` (`jump_target_5`,
  `jump_target_6`).
- **A manifest's ids are read once per write, never once per root**
  (`TestIdsReadOncePerManifest`). Reading every id a group or bundle holds
  is one pass over all its trees (`groups._held_ids`); asking it once per
  root made writing and reading a bundle of N roots cost N passes, each
  over all N trees. The test counts the passes, so it fails on the
  quadratic shape whatever the machine's speed: for a flat bundle of 3 and
  of 12 roots (each with one `LinearMap` and one region), `to_dict` makes
  exactly one pass, and `AlignmentBundle.from_dict` of that manifest makes
  none (the reader has found every id of the manifest unique, so no root
  needs admitting). That the ids read in one pass are the ids read root by
  root is proved on a manifest where the two could differ: roots of a
  bundle holding a group with a given id, an ungrouped root, and a root
  also held by a second bundle with a given id; the ids of all roots read
  together equal the union of `Timeline._manifest_ids` of each root, id
  for id and holder description for holder description. The bytes written
  stay pinned by the round-trip, twin and canonical-order tests, unchanged.

Loader parity (`test_manifest_ids.py::TestLoaderParity`): for one specimen
each of the ms3 (with and without D.S. jumps), score-MIDI, lab, audio (its
samples timeline and its seconds twin), Parangonada and IIIF loaders the test walks the whole
`to_dict` tree and requires every timeline, child, map, region and
flow-control element to carry a valid id (non-empty, no control character),
all ids within one manifest to be distinct, and no id to start with `tl:`.

### Statements (`test_statements.py`, `test_statements_references.py`)

A bundle (the manifest), every timeline at any depth and every region hold
`statements` and `provenance` exactly as the file writes them; the manifest
written carries an inline JSON-LD `@context`. The library's API is a
delegation: matching, adding, removing, candidates, beliefs,
the rectangles a catalogue reads and the prefixes a document uses are
TimeToShare's (`timetoshare.statements`, `timetoshare.catalogue`), run on a
one-subject document built from the holder's own state. Re-testing those
rules here would only test the package twice. Each case therefore proves
what the library adds, and where the package decides, its own messages are
the oracle (the expected `ValueError` text is the package's, or is computed
by calling the package):

- **Storage and the three faces agree.** One parametrised suite runs every
  case over the three holders: a timeline (`ContinuousPhysicalTimeline`
  `cpt_1` of unknown length), a bundle (`AlignmentBundle(id="rec")`) and a
  region (`region_1`, 0.0 to 10.0 seconds) through an adapter that calls the
  region's copy-on-write method, stores the copy with `replace_region` and
  reads it back with `get_region`. A case that passes on one face and not
  another shows a face that does not delegate the same way. The pair `type
  schema:VideoObject` and the verb `performed_by` in Wikidata's vocabulary
  (`wd:Q235650`, label "Maria João Pires") give the exact `statements`
  dict, keys in code-point order, and `provenance == []`: no entry is
  written unless asked for, since there is no default assertion. The same
  pair again, and with the thing written as its absolute IRI
  (`http://www.wikidata.org/entity/Q235650`), raise the package's `'wdt:P175'
  already has this value`: the holder's context includes the catalogue's
  prefixes, otherwise the absolute IRI would not match. Removing both pairs
  leaves `{}` and `[]`.
- **The prefix rule.** A pair using `foaf:` raises `the prefix 'foaf' is
  declared neither by the catalogue nor on <holder>`, the holder named as
  `timeline 'cpt_1'`, `bundle 'rec'` or `region 'region_1'`, and the
  holder's state is unchanged (the statements compared before and after).
  `add_prefix("foaf", "http://xmlns.com/foaf/0.1/")` then admits the pair and
  `prefixes` is exactly that one entry. `add_prefix("unit", ...)` is refused
  with `cannot declare the prefix 'unit' as '<namespace>': <problem>`, the
  problem computed by calling `timetoshare.statement_problems` on the same
  context, so the test follows the package's name rules rather than copying
  them; `foaf` redeclared with another namespace raises `the prefix 'foaf'
  already stands for 'http://xmlns.com/foaf/0.1/' on <holder>`, and the same
  declaration again changes nothing.
- **Candidates and beliefs agree on every face.** On each holder a
  candidate (`add_candidate`, `stated_by` `TILIE` given as a string,
  `method="inferred"`) followed by the belief `"false"` of `EDITOR` gives
  the same whole entry (`stated_by` a one-item list, the belief's
  `actor`, `belief`, `date`), `statements == {}`, `get_candidates()` that
  entry and `get_candidates(believer=EDITOR) == []`: a face whose
  copy-on-write method did not delegate the same way would differ.
- **Provenance.** On `cpt_1`, every entry is compared whole, so each
  member kept, replaced or added is visible:
  - a pair stated with `provenance=` as a dictionary and a `type` pair
    stated with a `Provenance` object write the same members (the `type`
    entry's property `rdf:type`, its value `{"id": ...}`); removing both
    leaves `{}` and `[]`;
  - a candidate resting on `basis: ["#cpt_2"]` is held in `provenance`
    alone: `statements == {}`, `get_candidates() == provenance`;
    offering it again with `date` omitted and the module's `_today` fixed
    to 2026-11-21 replaces the entry in place, dated `"2026-11-21"`: the
    date is the library's, the rest the package's;
  - `EDITOR` believing it `"false"` leaves it out of
    `get_candidates(believer=EDITOR)` and in `get_candidates()` and
    `get_candidates(believer=REVIEWER)`: one actor's belief filters for
    that actor alone;
  - `EDITOR` stating it with `actor=EDITOR` and a `datetime.date` puts the
    pair (with its label) in `statements`, keeps the candidate's members,
    and leaves EDITOR one belief, `"true"`, dated `"2026-11-20"`: the
    later belief replaces the earlier one; `get_candidates() == []` and
    `get_provenance(believer=EDITOR, beliefs=["true"])` is the entry;
  - `REVIEWER` believing the stated pair `"false"` (with a note) leaves
    `statements` unchanged and appends REVIEWER's belief after EDITOR's:
    believing `"false"` never removes a pair, and the entry then leaves
    `get_provenance(believer=REVIEWER)` (the default states, all but
    `false`) while `get_provenance(believer=REVIEWER, beliefs=["false"])`
    and `get_provenance(believer=EDITOR, beliefs=["true"])` keep it;
  - `REVIEWER` changing to `"unlikely"` passes the default filter
    (`get_provenance(believer=REVIEWER)` is the entry) and
    `beliefs=["unlikely"]`, not `beliefs=["true"]`; for `TILIE`, who
    proposed the candidate and holds no belief, the entry is `undecided`
    (`beliefs=["undecided"]` keeps it).
- **Actors compare as IRIs.** A belief recorded by the compact actor
  `wd:Q4700` filters for the believer written
  `http://www.wikidata.org/entity/Q4700`, and the other way round, as
  TimeToShare's `candidates(..., actor=)` compares them; the filter is the
  library's one `belief_mask`.
- **Refusals change nothing.** The belief `"maybe"` is refused by the
  library's `Belief` scalar (a `ValueError` whose one error is the
  `literal_error` at `belief`); the filter states `["not_false"]` with the
  message of TimeToShare's `check_belief_filter`, which `belief_mask` asks,
  `'not_false' is no belief state: write 'true', 'undecided', 'unlikely'
  or 'false'` (even without a believer); a `date=` without `actor=` with `a
  date dates the belief of the actor stating the pair; give actor=`; and
  a candidate of a stated pair with the package's `'wdt:P2550' already has
  this value; a stated pair is no candidate`. The holder's `statements`
  and `provenance` are compared before and after each.
- **Node paths.** A `wdt:P629` node holding its own `schema:sameAs` pair; a
  candidate of a second `schema:sameAs` at `node=("wdt:P629", 0)` and
  `EDITOR`'s belief `"false"` in it (with a note) leave the node's
  `provenance` exactly one entry (method, service, date, `stated_by` and
  the belief as given), `get_candidates(node=..., believer=EDITOR) == []`
  and `get_pairs(node=...)` the one pair stated, so the path reaches the
  node and not the holder, whose `provenance` stays `[]`.
- **Copy-on-write.** A region's `with_statement` leaves the original's
  statements `{}`; `replace_region` stores the copy, and `get_region` reads
  it back. Regions with different statements are unequal.
- **Rectangles.** `get_rectangles()` of `cpt_1` read through
  `AlignmentBundle.from_dict` of the package's second example manifest (a
  score encoding `clt_6` and its video `cpt_1`) gives the package's three
  rectangles for that timeline, `is_a`, `performed_by` and `same_as`, with
  their things and labels, read with the shipped catalogue.
- **The manifest boundary.** The second example manifest, completed with
  `timetoshare` and `id`, reads; its statements land on `clt_6` and `cpt_1`
  (compared whole), `to_dict()["@context"]` equals the manifest's own and
  the round trip is equal. A plain timeline writes exactly
  `{"@context": FORMAT_CONTEXT, "timetoshare": "0.2.0", "timelines": [...]}`.
  A prefix declared but unused (`foaf`) is not written. A prefix the
  catalogue lacks (`mbw`) is kept as `prefixes` on the holder using it and
  only there (`{}` elsewhere), and written back; a prefix declared with
  another namespace than the catalogue's (`schema: "https://schema.org/"`)
  is kept on the holder too. Two timelines read from two manifests whose
  `ex` stands for two namespaces refuse to be written in one bundle with
  `the prefix 'ex' stands for 'http://a.example/' on timeline 'clt_1' and for
  'http://b.example/' on timeline 'clt_2'`. Statements without `@context`
  are refused on reading with the package's message, and a timeline stating
  `dcterms:source: [{"id": "#ghost"}]` is refused on writing with the
  package's `'#ghost' names no element of this manifest` at its path. A
  timeline read whole writes the manifest's own statements back; a manifest
  without `@context` reads and is written with the format's terms.

The reference plumbing (`test_statements_references.py`) keeps what a value
names true when IDs change and when part of a manifest is written. A part of
a manifest naming another is a pair `dcterms:source: [{"id": "#X"}]` with a
provenance entry whose `basis` is `["#X"]`, so every case checks the three
places a reference can stand, value, entry value and basis item:

- **Renamed.** An element given a new ID — a timeline or region minted again
  on joining a group or a bundle, on `add_child` or in
  `AlignmentBundle.from_bundles`, a root given an ID on reading, a group
  minted again by `from_bundles` — is named by its new ID in the value, the
  entry and the basis, and the manifest written validates. The member
  already in the scope keeps its own references: renames run before the
  newcomer joins, so a holder naming its own element by the ID the newcomer
  had is untouched. A conversion map and a flow-control element minted
  again with their subtree are renamed in a value naming them. A map added
  to a timeline that already holds a map of its minted ID takes a new ID,
  and a statement naming the existing map still names it: only an element
  that was named before it moved is renamed.
- **A group given a new ID.** A statement naming a group by `#group_N`
  follows the group wherever its minted ID is minted again, and only
  where the old ID named that group:
  - `add_group` of a group minted `group_1` into a bundle whose own
    `group_1` holds `p` mints it `group_2`; the newcomer member `s` naming
    `#group_1` names `group_2`, while `p`, a member of both groups whose
    tree the bundle held before, keeps naming the bundle's `group_1`.
  - `from_bundles` of a bundle holding `group_1` (member `f`) and a source
    bundle whose minted groups `group_1` (`m`) and `group_2` (`n`) become
    `group_2` and `group_3`: every tree carried from the source, the
    ungrouped root `u` included, is renamed at once, so `u` naming both
    groups names `group_2`, `group_3` (each reference renamed once, never
    twice), `m` naming the source's `group_2` names `group_3`, and `f`
    still names `group_1`. The trees are shared with the source bundle,
    so the source no longer writes: its `to_dict` raises the check's
    `'#group_3' names no element of this manifest`. This is a documented
    limit of `from_bundles`.
  - A minted `group_1` giving way to a given ID `group_1` — a timeline
    joining the bundle holding the group, a timeline joining the group,
    a region added to a member — becomes `group_2`; the member and the
    bundle naming `#group_1` name `group_2`, while the joining timeline,
    whose `#group_1` names itself, keeps it.
  Each manifest written validates.
- **Dropped.** An element leaving the manifest — a member removed from a
  group, a timeline or a group selected from a larger manifest, a partial
  write (a timeline of a bundle written on its own) — takes its pair, its
  entry and its basis item with it; the manifest written validates and
  equals the expected literal. A bundle holding the group still holds the
  removed member, so nothing it holds names less.
- **Prefixes travel with their holders.** `from_bundles` carries a
  timeline's own prefixes, and two namespaces for one prefix refuse to be
  written as above.
- **What starts empty, what copies.** `create_child_from_region` and
  `get_slice` give timelines without statements (a region's statements stay
  on the region); `to_typed` copies statements and prefixes, and a change
  to the copy leaves the source as it was.

### Canonical order of a serialized timeline (`test_canonical_order.py`)

`Timeline.to_dict` writes every unordered collection in one canonical order,
so that two timelines with the same content serialize to the same text however
their content was added: `children` by offset, then id; `conversion_maps` by
id; `regions` by start, then end, then id; the `flow_control` elements
(every role in one array) by coordinate, then id; the `targets` of a jump
by place (the destination without one, then those with `by`, by value,
then those with `to`, by id). Ids are unique within a manifest, so no two
entries of one list tie on every member. Coordinates compare by their exact numeric value, never by their JSON
spelling, and strings by Unicode code point.

What proves it:

- **Invariance.** One fixed content set is built on three axes, `ticks`
  (`int`), `quarters` (`fraction`) and `seconds` (`float`): four children,
  two of them at the same offset; three conversion maps; four regions, three
  sharing a start and two of those sharing start and end; three breaks (a
  Fine and two section breaks), two at one coordinate; three jumps, one at
  each position, since a coordinate holds one jump; five destinations of
  the last jump (none, two distances back, two markers); four jump targets,
  two at one coordinate.
  Every element carries a given id (its kind and label, e.g. `break-fine`),
  because minted ids follow insertion order. For every permutation of
  the insertion order of each collection, the payload must equal the payload
  of the reference insertion order under `==`, and its `json.dumps` text must
  be identical. Equality of dicts ignores key order, so the text comparison is
  what catches a differing `children` or `targets` order. A serializer that
  kept insertion order fails on the first non-identity permutation.
- **The order itself.** Invariance alone would also hold for an arbitrary but
  fixed order, so the expected sequence of child ids, map ids, region ids
  and flow-control element ids is spelled out literally per axis. The values are chosen so that numeric order and JSON-text order
  disagree: on the `ticks` axis `9` precedes `10`, on the `quarters` axis
  `5/8` precedes `2/3` although `[2, 3]` sorts first as text, and on the
  `seconds` axis `9.25` precedes `10.5`. The two jump targets at one
  coordinate, `marker-Coda` and `marker-coda`, pin code-point order of the
  id tie-break (an uppercase letter precedes every lowercase one). The two
  distances of the last jump are spelled out per axis: back to the lowest
  position lies further than back to the middle one, `-11` before `-10`
  ticks, `[-23, 8]` before `[-17, 6]` quarters, `-11.5` before `-10.25`
  seconds, each against its text order; the two destinations naming a
  marker follow in the order `marker-Coda`, `marker-segno`.
- **Ties.** Children at one offset fall back to id; regions at one start fall
  back to end, then id; flow-control elements at one coordinate (two breaks,
  a jump and two jump targets at one position) fall back to their id, whose
  expected order is spelled out from the literal entries.
- **Reading accepts any order.** For every permutation of each collection in
  the canonical payload (the lists reordered; for breaks, jumps and jump
  targets, the elements of those roles moved to the front of the
  `flow_control` array in the permuted order, so that jumps come before the
  jump targets they name; for the destinations, the `targets` of the last
  jump reordered), `Timeline.from_dict` must restore a timeline whose
  `to_dict` is the canonical payload, text-identical.
- **What is stored.** `children` keep their insertion order after
  `to_dict`: there the order is a property of the serialized form only.
  Flow control is held in the canonical order itself: the registry's
  elements and the destinations of a jump read back in the order spelled
  out, on the object and on its copy.

### Measure and hierarchy structure

The hierarchy specimens prove that every metrical claim is anchored to the
exact quarter-note extents of one measure map. A default reading is reconstructed
from signatures rather than serialized: regular bars cover their beats and
compound beats cover their divisions, while irregular bars without a stated
window remain unpartitioned. Constituent bars clip that same nominal grid, so
an anacrusis pulse means the represented fragment and never an invented full
beat. Missing levels, positions, gaps, overlaps, and children at or above their
parent's level all fail validation.

The Beethoven rows exercise compressed-row expansion across four number
restarts. Exact counts, qstamps, lengths, classes, tempo positions, pulse
lengths, levels, and derived policies prove that a tempo indication's beat unit
selects the pulse without turning its rate or text into structure. Changing
only rate or text preserves forest equality; changing the beat unit can move
the pulse and therefore changes equality. Abstract rows remain abstract, and
unknown keys, contradictory counts or qstamps, unreadable signatures, invalid
cadenzas, and non-tempo extension values cannot be silently repaired.
A JSON float length enters as its exact binary expansion, never rounded and
never guessed, because the Measure Map standard writes lengths as floats.

An off-grid beat unit produces a separate deterministic reading only across a
stretch that complete rational least-common-multiple groups can tile. The
tests distinguish a valid four-bar reinterpretation from an invalid three-bar
remainder and prove that a group coinciding with a bar packs as an alternative
expansion of the same node. Authored hypermeasures similarly pack shared bar
identities under their stated parents while leaving the default bar partition
attributed only to the default reading. Reserved reading ids and falsely
attributed expansions are rejected.

Tempo observations retain their four source states, but every tempo entry has
a logical position. Typed coordinates and exact spans survive JSON as a
fixpoint; derived readings and the owning measure map do not enter the hierarchy
payload. ``MetricHierarchy.from_dict`` is given that map, and a timeline
rejects a hierarchy without one or anchored on a different map. A manifest
carries no hierarchy: ``to_dict`` writes none, and a ``metric_hierarchy`` key
on a timeline object is an unknown key, kept and written back, never read. These checks rule out detached
forests, positionless tempo facts, ambiguous simultaneous pulse units, and
accidental mixing of symbolic and physical axes.

Measure serialization is tested with all six concrete subclasses and
non-dyadic fractions such as ``Fraction(1, 3)``. The exact class name and every
field must survive JSON; a float detour would change at least one denominator and
fail. An unknown or missing class name is rejected with the complete known-class
vocabulary. A payload key the tagged class does not declare is rejected by name:
a ``RegularMeasure`` payload carrying ``offset_within_measure`` would otherwise
lose the offset silently, so accepting it would not be a faithful restoration.
A measure-map round trip uses the public constructor, emits no deviation warning,
and compares its type-aware measure sequence exactly. Two independently built
maps of the same concrete measure classes are equal; changing only one element
from ``RegularMeasure`` to ``IrregularMeasure`` makes the maps unequal even when
all field values match.

Metric dictionary tests cover a whole-note beat size, a quarters beat size, and a policy
authored from division alone. Each reconstructed policy must equal the original
full pydantic scalar, including its name and the authored beat-size unit. A metric
hierarchy round trip preserves the recursive forest, attributed reading ids,
reading provenance, and every independent tempo map as an exact JSON fixpoint.

Equivalence is deliberately narrower than the dictionary contract. Alternative child
partitions and their asserting reading ids define structural identity. Tempo
maps, verbal tempo text, reading authorship, reinterpretation links, and local
level vocabulary remain queryable but do not affect equality.

### `test_groups.py` and `test_group_definition.py` - Timeline Groups

**A C-Map column only where the map reaches every row
(`TestConversionColumnErrors`).** A group's timestamp table converts each
member's column through each numeric C-Map as one array; a map raising
`ValueError` (a row outside its domain) gets no column, and any other error
is a defect that propagates. Two tests, as the bundle's catches are tested
(`tests/alignment/test_bundle.py::TestUnexpectedErrorsPropagate`): with
`TableMap._convert_array` raising `RuntimeError("boom")`,
`get_timestamp_table([25.0], "clock", conversion_maps=True)` raises it, and
the traceback passes `conversion_map_column` (the one column rule the
timeline's table applies too) at the statement `converted =
cmap.convert_array(values)`; and on a 100-second `clock`
carrying a milliseconds map over `[0, 100]` and a frames map over `[0, 50]`
with `extrapolate="error"`, the rows `[25.0]` give the columns `clock`,
`seconds_to_milliseconds`, `seconds_to_frames` (1250 frames), the rows
`[25.0, 75.0]` only `clock` and `seconds_to_milliseconds` (25000 and 75000).

A `TimelineGroup` is an array of member timelines that, by definition, share
one graphical start and one graphical end: every member's 0 coincides with
every other member's 0, every member's length with every other member's
length, and a position in between converts linearly. When only part of a
timeline takes part, the participant is a **child** of that timeline, which
carries the part's exact offset; a member is never placed partially. The tests
are written to prove that definition and nothing richer.

**What proves the definition.** The conversion is pinned by exact values that
only the linear map `p · len_B / len_A`, computed on exact ratios, produces.
The fixture mixes the three number types so that each re-expression rule is
exercised: A is a `ContinuousLogicalTimeline` of `Fraction(16)` quarters
(fraction-canonical), B a `ContinuousPhysicalTimeline` of `40.0` seconds
(float-canonical), C a `DiscreteGraphicalTimeline` of `4800` pixels
(int-canonical):

| From | Position | To B (seconds) | To C (pixels) | To A (quarters) |
|---|---|---|---|---|
| A | `8` | `20.0` | `2400` | — |
| A | `Fraction(1, 3)` | `0.8333333333333334` (= `float(Fraction(5, 6))`) | `100` | — |
| A | `16` | `40.0` | `4800` | — |
| C | `1` | `float(Fraction(1, 120))` | — | `Fraction(1, 300)` |
| A | `Fraction(1, 7)` | — | `43` | — |
| A | `Fraction(1, 600)` | — | `0` | — |
| A | `Fraction(1, 200)` | — | `2` | — |

`A 1/3 → B` is the case a float pipeline gets wrong in the last place: the
exact image is `5/6`, and the expected float is `5/6` rounded once. A float
query enters as its exact dyadic `Fraction`, so the arithmetic never rounds
before the target axis writes the result.

**The int-canonical rule.** A result on an int-canonical member is written
under the library's boundary rule for a *computed* value (`express_as`, the
re-expression documented in "Rounding and exactness"): an exact non-integral
ratio is rounded to the nearest integer, ties to even. Construction refuses an
exact non-integral value; re-expression of a conversion result rounds, because
a conversion into pixels is a quantization by intent. `300/7 = 42.857…` names
pixel `43`; the two tie cases pin the tie rule itself: `1/600 · 300 = 0.5`
names pixel `0` and `1/200 · 300 = 1.5` names pixel `2`. Each expected value
is also asserted with its Python type (`float`, `int`, `Fraction`), so a
correct number in the wrong representation fails.

**A group stamp declares each member's number type.** A stamp from a group
holds one coordinate per member, and each coordinate states its member's
declared number type, whichever member was queried and whether the position is
a stored row or a computed one. Values of the right Python type are not enough
to prove it: a coordinate could carry a correct value while naming a
different number type, and the declared type is what a reader of `to_dict()`
sees. So `test_group_definition.py::test_group_stamp_declares_member_number_types`
queries the fixture above three times — A `Fraction(1, 3)` (computed), A `16` (the stored end row)
and C `1` (computed, from the int member) — and asserts, for every member,
the `number_type` of `get_coordinate_for(member, format="coordinate")`
(`NumberType.fraction` for A, `NumberType.float` for B, `NumberType.int` for C)
and the `number_type` written by `to_dict()` (`"fraction"`, `"float"`,
`"int"`), beside the exact values of the table above.

**A child member.** Root R is a `DiscreteGraphicalTimeline` of 5000 px with a
child K of 2000 px at offset 1000; the group holds A′ (`Fraction(8)` quarters)
and K. A′ `2` is a quarter of the way, so K `500`. R is no member, so the group
stamp holds A′ and K only; R's own stamp at `1500` holds K at `500`, which is
what places the part inside its root. The manifest writes the roots A′ and R
(sorted by id) and names the members A′ and K (sorted by id); the member order,
kept in `graphical.order` under the group's id, is the order of addition.

**Stored rows and intervals.** The stored rows are the common start and end:
each member at `0` and at its own length, exact in its declared number type.
The fraction member of that test is `Fraction(5, 3)` quarters long, a length
no float holds, so its end row must read `Fraction(5, 3)` exactly; the float
and int members read `40.0` and `4800`. `get_interval_for(member)` is
`[0, length]` of that member. A position outside `[0, length]` on the queried
member raises `ValueError`; there is no other unreachable position, because
every member spans the whole group. The only null a group table can carry is
in a descendant column, where a member's child does not reach a queried row.

**Round trips.** For every group built in these tests:
`TimelineGroup.from_dict(g.to_dict()).to_dict() == g.to_dict()`, and the
rebuilt group answers `get_timestamp_at` at `0`, a third, a half and the length
of every member with stamps whose every coordinate is identical to the
original's (value, Python type and unit). Equality of the
dictionaries alone would not prove the rebuilt group *behaves* the same, so the
queries are asserted as well.

**Styling is carried, never read.** A manifest's `graphical` may hold
`styles` and `rules` beside `order`: presentation data the library keeps
on `TimelineGroup.styling` without interpreting it. What proves it is carried
unchanged is a styling block the library could not have produced itself — a
named style, a rule setting that style by name and a rule setting a
declaration directly, nested three levels deep — surviving `from_dict` →
`to_dict` equal to the input dictionary, beside the unchanged `order`; and the
rebuilt group's `styling` must not share an object with the input, so a later
edit of the input dictionary cannot reach the group. Each member alone
(`styles` only, `rules` only) round-trips alone in the minimal form, so
neither is invented when absent; a group without styling writes `graphical`
as `{"order": {...}}` in the minimal form and adds `styles: {}`,
`rules: []`, `timestamps: []`, `matchstamps: []` and `elisions: []` in the
complete one. A `graphical` holding styling but no
`order` reads its members in `members` order, as an absent `graphical` does.
Any other key of `graphical` (`frame`) is kept and written back, never read.
The band orders of `graphical.order`, keyed by a timeline's id, are carried
beside the styling; their tests are those of "Manifests" above.

**Identity.** The default id is `group_1` (minted statelessly, above the
highest number held, so two fresh groups in one process are both `group_1`). The
members' trees share one manifest, in which every id is unique whatever
element holds it (the other kinds are proved under "Manifest identity"). A
newcomer whose given id, or any given descendant id, is held by another
timeline of the group's trees is refused with `ValueError` (`timeline 'B'
cannot join group 'group_1': its id 'B' is already held by timeline 'B'`)
and leaves the group unchanged; a minted id of the newcomer's tree that is held is minted
again — the newcomer's own (`clt_2`) and a descendant's alike (a minted
`clt_1` child of a newcomer with a given id becomes `clt_2` under its
unchanged parent). A refused newcomer keeps its ids: the collision checks run
before any id of its tree is minted again, so a newcomer whose minted child
would be re-minted but whose given root id is refused still carries its
child's original minted id afterwards. A timeline whose given id is the
group's own given id is refused (`timeline 'shared' cannot join group
'shared': its id 'shared' is already held by group 'shared'`); a *minted*
group id that a newcomer's given
id takes is minted again instead (`group_1` becomes `group_2`), because only
a given id is a commitment.

**No nesting inside one group.** Members share their start and end, while a
child sits at its offset inside its parent, so a parent and its child in one
group would give one position two answers. Root R is a
`DiscreteGraphicalTimeline` of 5000 px with a child K of 2000 px at offset
1000: a group of R and K is refused with a `ValueError` naming both ids,
whichever is added first, and `from_dict` refuses a group object naming both,
leaving nothing built. A parent and its child in *different* groups of one
bundle remain allowed (the bundle tests pin it): each group converts within
itself, and the offset relates the two.

**One timeline, several groups** (`test_group_definition.py`). Membership is
no partition: one timeline object may be a member of several groups at once.
From a manifest whose groups `g1` (`t`, `a`) and `g2` (`t`, `b`) share `t`,
`TimelineGroup.from_dict` selecting `g1` reads the members `t`, `a` and
selecting `g2` reads `t`, `b`, each in that order, and each group converts
`t`'s position by its own members' lengths. The bundle's handling of a shared
member is tested in `tests/alignment/README.md`, "A Timeline in Several
Groups".

**A segment-line member is locked.** A segment line that becomes a member is
locked by default, as it is when it becomes a child; lifting that lock is
not supported yet. A plain member is left as it is. What proves it: a
`SegmentLine[ContinuousLogicalTimeline]` of two segments (4 and 2 quarters),
built unlocked, is locked after `TimelineGroup(timelines=[line, plain])`, after
`group.add_timeline(line)` and after `AlignmentBundle.add_timeline(line,
grouped_with=...)`, while the unlocked plain member stays unlocked and the
line's segments stay unlocked; the group writes `locked: true` on the line in
both forms. Reading is not construction: `TimelineGroup.from_dict` and
`AlignmentBundle.from_dict` of a manifest whose member line states
`locked: false` keep it unlocked and write `false` back, and one stating
`true` keeps it locked. So does `TimelineGroup.from_dict(data, group_id="g")`
of a manifest without groups whose one root is that line: the group holds
`line` alone, unlocked when the file states `false` (written `false`, left
out of the minimal form) and locked when it states `true` (written `true`
in both forms).

`from_dict` refuses each malformed input with the format validator's
refusal (a `ValueError`): a missing required key, a member naming no
timeline, a timeline id written twice, an order that is not a permutation
of the members, a member nested with another member, a group id that equals
a timeline id, and one holding a control character. Each case asserts the
code and the JSON pointer of the fault exactly and, of the message, the part
naming the fault (`'members'`, `'Z'`, the first holder `#/timelines/0/id`,
`U+0009`, both `'R'` and `'K'`): the wording is TimeToShare's and changes
there without any change in what is refused. A root
that holds no member is not refused: a group is read with the timelines its
members need, and such a root is left out.

Event aggregation tests assert `EventData` results because the group preserves
Arrow schemas, null-fills fields that are absent from some member timelines,
and adds a constant `timeline_id` provenance column. A member's own
`timeline_id` is overwritten by authoritative group membership. Columns
present in multiple members must retain the same Arrow type; incompatible
types raise an exact `ValueError` naming the column and both types, while
genuinely missing columns remain null-filled. DataFrame views are tested only
through `EventData.to_dataframe()`.

**A half-pixel query is not a half-pixel coordinate.** `4875` is odd, so its
midpoint `2437.5` falls between two pixels — and a pixel axis is int-canonical,
so that query names pixel `2438` and the group answers with *that pixel's*
position, `2438 · 150 / 4875 = 75.01538461538462 s`. Every projection of a
stamp is a projection of the one position its axis names, so the rounding is
visible in the answer instead of only in the pixel column. The same applies to
`test_thoresen_poc.py`, whose width `4835` puts its midpoint at pixel `2418`.

`test_three_timeline_group` places an image over audio seconds 0–150 and a
score over audio seconds 45–135: each part is a child of the audio, grouped
with the timeline spanning it, and the score's `100 · 30/90` is the float
`33.333333333333336`, asserted exactly. `test_round_trip_conversion` closes an
audio → pixel → audio loop: 67.5 s is pixel `2193.75`, which names `2194`,
whose seconds are `67.50769230769231`.

### `test_unknown_length.py` - Timelines of Unknown Length

A length of `None` states that a timeline's length is not known. It is a
fact about the timeline, not a defect: it is never inferred, never refused,
and any timeline may have it (root, child, locked, group member, segment).
The tests prove one rule and its consequences: **an operation that needs the
end raises `UnknownLengthError` before it changes anything; an operation
that does not need it works, and positions on the timeline are not bounded
above.**

**What proves it.** Every raising operation is pinned with its exact message
(the message names the operation and the timeline whose length is unknown)
and with the state checked after the raise, which must be what it was
before: an unchanged length, child list, segment order, region count or
flow-map list, whichever the operation would have touched. Each
end-independent operation is pinned by the exact value it returns on a
timeline of unknown length, so a test cannot pass by raising early or by
quietly treating the unknown end as some number. In particular, a position
far beyond all content (for example `100` on a timeline whose content ends at
`8`) must resolve, and an unknown length must still read as `None` after
content is added. That last check is what rules out an implementation that
infers a length from content.

| Operation | Needs the end? | Expected |
|---|---|---|
| constructor, `create_child(length=None)`, `locked=True` | no | `length is None`, `end is None`, `is_length_known is False` |
| `length = None`, `length = 8` from `None` | no | ordinary edits; locked raises `RuntimeError`; a value below the content raises `ValueError` |
| adding events or children | no | the length stays `None` |
| child containment | partly | `0 ≤ offset` always; `offset ≤ parent length` when the parent's is known; `offset + length ≤ parent length` when both are |
| position queries, slicing to an explicit end | no | resolve past all content |
| timestamp table, boundary table | no | an unknown member is bounded below only; it has a start row but no end row |
| group conversion between members | yes | raises, naming the member whose length is unknown |
| group position check | no | the member bounds its positions below only: a group of one unknown member answers `1000` and refuses `-1` |
| group stored table, `get_interval_for` | yes | raises |
| matchstamp through a group with an unknown member | yes | raises (not swallowed) |
| `append_child` on an unknown parent, `append_segment` after an unknown segment | yes | raises |
| `append_segment` on a line of unknown length, empty or after a known segment | no | appends at `0`, then at the known segment's end; the line stays unknown |
| splitting with `include_after_last=True` | yes | raises; with `False` it works |
| a child timeline of unknown length named as a chunk of a FlowMap | yes | raises when the FlowMap joins the timeline; a FlowMap over Regions of a timeline of unknown length resolves |
| `WarpMap.materialise` on an unknown source or descendant | yes | raises |
| WarpMap queries | no | the support ends at the last anchor whatever the timeline's length |

**Segments.** Contiguity is checked between known values. A segment of
unknown length ends at its successor's offset, so its successor may start
anywhere at or after its own offset, and a query between the two reaches
only the earlier segment. Built from `[0, 4)`, an unknown segment at `4` and
`[10, 12)`: position `7` reaches the unknown segment at `3`, position `11`
reaches only the last segment, at `1`. The line itself has an unknown length
until one is set (`12`). A growing line whose new last segment is of unknown
length becomes of unknown length too, because a line ends where its last
segment ends.

A segment of unknown length ends where its parent ends it, so content added
into it may not pass that point. Segment `x` (unknown, at `0`) with its
successor at `5`: a Note `[1, 5]` is accepted and leaves `x` unknown, a Note
`[8, 9]` raises `Adding content ending at 9 to timeline 'x' passes 5, where
its parent 'line' ends it while its length is unknown`, and `x` still holds
one event. The last segment ends at the line's known length: in a line of
length `9` whose last segment `b` (unknown) starts at `4`, `[0, 5]` is
accepted and `[0, 6]` raises (`passes 5`). The bound is the length's absence
read through the parent, so it cannot come from anything the segment itself
holds.

**Reach.** A display needs an extent even when the length is unknown. The
reach is the farthest own-axis coordinate the content reaches: children's
ends (a child of unknown length contributing its offset plus its own reach),
region ends, the coordinate of every flow-control element, the place a
jump's destination reaches by its distance (`by`), and the end of the
measure map. Events are not part of it. The nested example is
computed by hand:

| Timeline | Content | Reach |
|---|---|---|
| `V` (unknown) | jump target at `7` | `7` |
| `U` (unknown) | region `[1, 3]`, `V` at offset `5` → `5 + 7 = 12` | `12` |
| `R` (unknown) | child `A` `[2, 6)`; `U` at `10` → `10 + 12 = 22`; region `[0, 15]`; break at `18`; jump at `20` going on `16` before it | `22` |

Each content kind also wins once on its own: a lone measure map of five
3/4 bars reaches `15`, a jump at `9` going on `21` behind it reaches `30`, and a timeline without
content reaches `0`. The diagram of `R` scales to `22`, labels each unknown
end `?`, and never prints `0` or `nan` for it. `repr()` says `length=unknown`.

**The retired sentinel.** A length of `0` used to mean "unbounded" when
deciding whether a parent position reaches a child. A zero-length child at
offset `3` in a parent of length `10` must now be absent from the parent's
stamp at `5`; the old reading would have reported it at `2`. A zero extent in
the diagram puts every bar at the origin rather than filling the width.

### `test_unlocked_child.py` - Unlocked Children

`locked` states whether a timeline's length is fixed. It defaults to `false`
on every timeline, a child included, and a child's `locked` is read and
written as it stands. Embedding means boundary containment only: an unlocked
child behaves like any unlocked timeline whose boundaries its parent must
contain. `add_child` still locks the child by default (programmatic use and
the loaders rely on it); `add_child(..., locked=False)` leaves the child's
own state as it is, so an unlocked child stays unlocked and a locked one
stays locked (a lock is never lifted by a call). A segment line is the
exception: it is locked as a child, see below.

**Segments are never locked.** A plain segment, a segment of a segment line
that is no segment line itself, has no lock of its own: the segment line's
lock governs it. Its own `locked` means nothing, so it is `false` however
the segment came to be — appended, read, cast or copied — and a lock the
segment carried before it joined is lifted. Asking for it is refused: an
explicit `add_child(..., locked=True)` or `lock()` on a plain segment raises
`ValueError("A segment is never locked: lock its segment line '<line>'
instead.")`. A segment line that becomes a child, a segment or a group member
is locked, like any child; lifting that lock is not supported yet. So
`add_child(line, ..., locked=False)` on a segment line that is unlocked is
refused before anything changes with `ValueError("Unlocking a segment line
that is a child, a segment or a group member is not supported yet.")`; on a
segment line already locked it asks for nothing and is accepted. Reading and
copying are not construction: a file's own `locked` on such a segment line
is read as it stands, `false` included.

The tests prove one rule: **a change of a timeline's length must stay within
its parent's known length, and a parent never grows because a child grew.**
Content or a child added into an unlocked timeline grows that timeline only,
and the addition raises when that growth would leave the timeline's own
parent. Every refusal is a `ValueError` raised before anything changes,
naming the child, the end it would have and the parent's end:
`Timeline 'C' cannot end at 12 on its parent 'P', which ends at 10`. A parent
of unknown length checks nothing; a segment of unknown length ends at its
successor's offset, and the last segment of a line at the line's length.

**What proves it.** Every value is computed by hand from offsets and lengths,
so a test cannot pass by restating the implementation:

- **Round trip.** A manifest root `rec` (seconds, length 30) with children
  `a` (offset 0, length 4, `locked` absent), `b` (offset 4, length 4,
  `locked: false`) and `c` (offset 8, length 4, `locked: true`) reads to
  `is_locked` = `False`, `False`, `True`. The complete form writes `locked`
  = `false`, `false`, `true`; the minimal form omits it on `a` and `b` and
  writes `true` on `c`. Reading the written form again writes the same form
  back. A library that forced every child locked would write `true` three
  times, which is the defect this pins. `to_typed` keeps an unlocked child
  unlocked.
- **`add_child`.** Embedded with the default, a child is locked; with
  `locked=False` an unlocked child stays unlocked (its length can then be
  set: 3 at offset 2 writes the row end 5, duration 3) and a child built
  locked stays locked.
- **Unlocking a segment line is refused.** An unlocked segment line `line`
  (segments 4 s and 2 s) given `locked=False` is refused with the exact
  message above twice: as a child of a plain 20 s parent `P`, and as a
  segment of an empty edition line `edition`. Each time the parent is left
  without children (and the edition at length 0) and `line` still unlocked.
  The same line locked first is accepted with `locked=False` and stays
  locked; the parent then holds it. A child line read from a file stating
  `false` keeps `False` (pinned under "Reading normalises a plain segment").
- **Plain segments are unlocked on every construction path.** Each of the
  four ways a segment line is built — `append_segment` on a
  `SegmentLine[ContinuousPhysicalTimeline]` (segments 4 s and 2 s),
  `append_child` on the same line, `create_segment_line([0, 4, 6])` and
  `SegmentLine.from_segmentation(source, [0, 4, 6])` on a 6 s timeline —
  gives two segments with `is_locked` `False`, and the complete form writes
  `locked: false` on both while the minimal form leaves the key out. A
  segment built `locked=True` before it is appended is unlocked once
  appended. A `lock()` on the line afterwards locks the line only.
- **A segment that is a segment line is locked.** An edition line of two
  page lines (each a line of plain segments) holds both pages with
  `is_locked` `True` and their plain segments with `False`; the minimal form
  writes `locked: true` on each page and nothing on the plain segments.
- **Asking to lock a plain segment is refused.** `add_child(segment, 0,
  locked=True)` on an empty line raises the exact message above naming the
  line, and leaves the line without segments and the segment unlocked;
  `lock()` on an appended plain segment raises the same message and
  leaves it unlocked. `locked=False` on a plain segment is accepted (it asks
  for nothing). `lock()` on a nested page line, a segment line itself,
  succeeds.
- **Reading normalises a plain segment.** A manifest stating `locked: true`
  on both segments of a line reads them unlocked, writes `locked: false` in
  the complete form and leaves the key out of the minimal form. A nested
  page line stated `locked: false` keeps `False`, and one stated `true`
  keeps `True`: the file decides for a segment line.
- **An unlocked segment still keeps the line contiguous.** The line `line`
  (4 s `A`, 2 s `B`, so 6 s), locked or not: an event `[0, 5]` added to `A`
  raises `SegmentLine requires contiguous segments. Segment 'A' cannot end
  at 5: its successor 'B' starts at 4.`, an event `[0, 3]` added to `B`
  raises `... Segment 'B' cannot end at 7: the line 'line' ends at 6.`, and
  `append_child` of a 1 s child to `A` raises the first message; after each
  raise the lengths are 4, 2 and 6 and the segment holds no new event or
  child. Events that end inside the segment (`[0, 4]` on `A`, `[0, 2]` on
  `B`) are added and change no length. The last segment of a line of
  unknown length has no end to keep: an event `[0, 7]` added to `B` makes
  `B` 7 long and the line stays of unknown length, since growth never
  pushes the parent; `A` in that line is still refused at 5.
- **Growth inside the parent.** `G` (length 15, a root) holds `P` (length 10)
  at offset 5, which holds `C` (length 4) at offset 3, both embedded
  unlocked. An event `[0, 6]` added to `C` makes `C` 6 long, ending at
  `3 + 6 = 9` within `P`'s 10: `P` and `G` keep 10 and 15, and `C`'s
  `Segment` row in `P` reads start 3, end 9, duration 6. Setting `C`'s
  length to 7 ends it exactly at 10, which is still inside.
- **Growth beyond the parent, atomically.** The event `[0, 9]` would end `C`
  at 12 on `P` (length 10) and raises the message above; setting `C`'s
  length to 8 raises with `cannot end at 11`. It makes no difference whether
  `P` is locked: `P` unlocked and `P` locked give the same message. After
  each raise every length (`C` 4, `P` 10, `G` 15), `C`'s event count (0) and
  both rows are what they were before, which rules out an implementation
  that checks after mutating or grows the parent.
- **Growth of the parent itself.** Content added into the unlocked `P`
  grows `P` only. With `G` of length 20, an event `[0, 12]` added to `P`
  makes `P` 12 long, ending at `5 + 12 = 17`: `G` keeps 20, and `P`'s row in
  `G` reads start 5, end 17, duration 12. With `G` of length 15 the same
  event raises `Timeline 'P' cannot end at 17 on its parent 'G', which ends
  at 15`, and a child of length 4 added to `P` at offset 8 raises the same
  way; afterwards `P` is 10 long, holds no event and no new child, and both
  rows are unchanged.
- **Unknown length.** `C` (offset 2, length 4) in a root `P` of length 20:
  setting `C`'s length to `None` leaves `P` at 20 and writes end and duration
  `None` in `C`'s row; setting it back to 6 writes end 8, duration 6;
  setting it to 25 raises (`cannot end at 27`, `P` ends at 20). A parent of
  unknown length checks nothing: `C` set to 100 in such a parent leaves the
  parent's length `None` and writes the row end 102.
- **Segments keep their boundaries.** A segment line read from a manifest
  (length 6, unlocked segments `A` = `[0, 4)` and `B` = `[4, 6)`) never moves
  a successor or its own end. Setting `A` to 5 raises (its successor starts
  at 4) with every length and row unchanged; setting `A` to `None` is
  contiguous (a segment of unknown length ends at its successor's offset)
  and `A` back to 4 is contiguous again. Setting `B` to 3 or to 1 raises,
  because the line ends at 6:
  `SegmentLine requires contiguous segments. Segment 'B' cannot end at 7:
  the line 'line' ends at 6.` Lengths and rows stay as they were. Setting
  `B` to `None` leaves the line at 6, and `B` back to 2 ends it at 6 again.
  The lock of the line makes no difference to any of this.
- **Copies share no timeline with their source.** `to_typed`,
  `as_segment_line` and `SegmentLine.to_timeline` return copies holding
  copies of the children. The source `P` (4 s) holds the unlocked children
  `A` `[0, 2)` and `B` `[2, 4)` (base `Timeline`s for `to_typed`, which
  returns an already typed tree itself). For each cast, the result's `A` is
  not the source's `A`, each has its own timeline as parent, and setting the
  result's `A` to an unknown length leaves the source's `A` at 2 with its
  `Segment` row at start 0, end 2, duration 2, while the result's row reads
  end and duration `None`. Before, the child object was shared, so its
  parent pointer moved to the copy and a change on one side reached the
  other.
- **Casts keep what a read timeline holds.** A root read with
  `"offset": -2.5` and an unknown key `x-root` keeps `root_offset` and the
  key through `as_segment_line` and back through `to_timeline`, and both
  casts write the manifest's `meta` back, because the root was read whole.
- **Slices keep each child's lock.** Slicing a parent whose children are one
  unlocked and one locked keeps `is_locked` `False` and `True` on the
  slices of the two.

### `test_locked_segment.py` - A Locked Timeline Placed in Another

A timeline whose length its source states arrives locked (a loaded
recording, a score whose measures state every length). The question is what
becomes of that fixed length when the timeline is placed in another one.
Two rules answer it, and the tests prove each on the route that applies:

- **As a child, the lock is kept.** Every route that embeds a timeline as a
  child of a timeline that is no segment line — `add_child` with the default
  `locked=None`, `add_child(..., locked=False)` (which keeps the child's own
  state) and `append_child` — leaves a locked timeline locked. The lock is
  then observable in three ways, each computed by hand: `is_locked` is
  `True` and the length is the one it had; content past the end is refused
  by the timeline's own lock, `Adding content ending at 5 exceeds timeline
  length 4. Timeline is locked. Use allow_expansion=True to override.`; and a
  MatchClaim past the end is refused by `create_match_claims` with the
  message that names the lock (`... past the end of timeline 'rec' at 4
  seconds. The timeline is locked: ...`). After each refusal the length is
  still 4 and nothing was added. Writing the parent and reading it again
  gives a child that is still locked and 4 long. A child left unlocked
  stays unlocked and round-trips unlocked; that is pinned under
  `test_unlocked_child.py` ("Reading and writing `locked`", "`add_child`").
- **As a plain segment, the line's boundaries fix the length.** A segment
  of a segment line that is no segment line itself is never locked (see
  "Segments are never locked" above): a locked 4 s recording `A` and a
  locked 2 s recording `B` appended with `append_segment` to an empty line
  read `is_locked` `False`. Their lengths are nevertheless fixed while the
  line's length is known, and more firmly than by a lock: content past the
  end of `A` (`[0, 5]`) and of `B` (`[0, 3]`) is refused by contiguity
  (`Segment 'A' cannot end at 5: its successor 'B' starts at 4.`, `Segment
  'B' cannot end at 7: the line 'line' ends at 6.`) even with
  `allow_expansion=True`, which would override a lock; and a MatchClaim at
  5 s on `A` or 3 s on `B` is refused because the timeline cannot grow
  inside its line, with the contiguity message inside. After each refusal
  the lengths are 4, 2 and 6 and the bundle holds no MatchClaim. The last
  segment of a line of unknown length is the one case where this does not
  hold; it is pinned, as it stands, under "An unlocked segment still keeps
  the line contiguous" above.

### `test_locked_region_child.py` - Children from Regions of a Locked Timeline

A lock fixes a timeline's length and nothing more. A child made from one of
its Regions lies inside that length, so making it changes nothing the lock
protects, exactly as `add_child` of a child within the length is allowed on
a locked timeline. What proves it:

- **The child is made.** A locked quarters timeline of length 16 (exact
  `Fraction` axis) holding a Region `[5/2, 10)` gives a child whose offset
  is `5/2` and whose length is `15/2`, both exact, the child's events
  shifted by the offset (a note at `3` reads `1/2` in the child). The
  parent's length stays `16` and `is_locked` stays `True`; the child is
  locked as every child `add_child` embeds by default. The same holds with
  `measures=True`: a measure map of four 4/4 bars sliced over a Region
  `[4, 12)` gives the child two measures of 4 quarters.
- **A Region reaching past the locked end gives no child.** Regions are
  annotations and are not bounded by the length (a Region `[12, 20)` on the
  locked 16 is created). A child from it would end at 20, past the fixed
  end, so `add_child`'s capacity check refuses it with `ValueError`
  ("exceeds timeline length 16. Timeline is locked."); the parent is left
  as it was: no child, the same Segment rows, length 16, still locked.
  An unlocked timeline holding the same Region grows to 20 (the contrast
  that shows the refusal comes from the lock).
- **The other guards stay.** `append_child` on a locked timeline raises
  `RuntimeError` (appending lengthens), and setting the length does too.

## Test Files

### `test_coordinate_resolution.py` - Public Coordinate Resolution

These tests pass foreign-unit coordinates through timeline entry points that
read or update events, regions, children, lengths, and slices. Each entry point
must delegate to the same coordinate resolver and preserve its exact missing
C-Map error, demonstrating that coordinate normalization is consistent across
the public timeline surface.

A C-Map declared to write fractions computes exactly and refuses a float, a
ratio the input never stated, raising `OutputNumberTypeError`. A query
reaching such a map is refused at the timeline as a plain `ValueError` whose
message adds the timeline, with the map's refusal as its cause; the
timeline recognises the refusal by its class alone. A `score` in
quarters (8, `fraction`) carries `q2s` (quarters to seconds, scalar `1/2`)
and `q2n` (quarters to `number`, scalar 2), both declared `fraction`.
`get_coordinate_at` and `get_timestamp_at` with the float `1.5` seconds, and
with the float `3.0` in `number`, raise `ValueError` with the exact message
naming the map, its units, its declaration, the float and the timeline. The
same getters answer exactly with `Fraction(3, 2)` seconds (3 quarters, the
stamp reading `3/2` seconds and `6` in `number`) and with the int `3` in
`number` (`3/2` quarters, `3/4` seconds). Seconds hold no int, so the int
input goes through `number`, which admits both. A converter that let the
float through would answer `3.0` rather than refuse; one that kept the map's
own error fails the exception type, which must be `ValueError` itself.

Every other error a map raises passes through the timeline unchanged, the
same object, even for a float input: a timeline that relabelled any error
on a float as the fraction refusal would report a cause that is not the
cause. A `LinearMap` subclass declared to write floats, whose evaluation
raises a `TypeError` or a `ValueError` of its own, is queried with the float
`1.5` seconds; `get_coordinate_at` raises that very error object.

### `test_offset_arithmetic.py` - Nested Offset Arithmetic

**Design:** Parent–child coordinate transfer uses exact offset arithmetic
(`child_coord = parent_coord - offset`), so no floating-point drift enters a
transfer and rational offsets compose exactly. Parent-to-descendant lookup
checks every embedding span, while descendant-to-parent lookup and
`get_coordinate()` compose all offsets into the ancestor coordinate system.
Integer and `Fraction` cases assert exact results; unknown IDs retain their
established return or exception behavior.

**One containment rule, applied along the path.** A child holds
`[offset, offset + length)` plus its end where no sibling of nonzero length
starts there (one predicate, `_holds_position`): a lone child's end is on the
child, and a boundary two siblings share is on the one starting there only.
A position carried further down reaches a descendant only if every timeline
on the way holds it, so a grandchild never shows a position its parent leaves
to a sibling. Every reader follows that one rule: the timeline's stamps, its
tables, `get_children_at`, `get_events_at` and a bundle's stamps across the
tree. A test proves it only if all of them give the same answer on cases where
a lane deciding otherwise would differ:

- **The grandchild at a shared boundary.** `P` (length 8) with `A` at 0 and
  `B` at 4 (length 4 each), `A1` filling `A`, `B1` filling `B`. At `P` 4 the
  stamp names exactly `P` 4, `B` 0, `B1` 0; the table row has `A` and `A1`
  null and the same three values; the bundle's stamp is the same set from
  `P` 4, `B` 0 and `B1` 0. `A1` alone holds its end inside `A` (nothing in `A`
  starts at 4), so a lane that checked only the target, not the path, shows
  `A1` without `A` — the defect this pins. Queried on `A` at 4, `A`'s own stamp
  and table hold `A1` 4: a timeline's own positions are asked of it, so its
  end is its own whatever its siblings do; the bundle from `A` 4 adds the
  ancestor `P` 4 and what `P` 4 reaches (`B`, `B1`).
- **Exact tables decide exactly.** On an exact axis the table compares the
  positions as ratios (cross-multiplied), never through their floats. Two
  cases where floats decide wrong: a child at `7/10` of length `1/10` at
  `4/5` (`7/10 + 1/10` is `4/5` exactly, but `0.7 + 0.1` is not `0.8`): the
  stamp and the table both hold the child at `1/10`, its end, which nothing
  follows; and `A` at `1/10` (length `1/5`) beside `B` at `3/10`, at `3/10`:
  `B` alone, at 0, in the stamp and the table, `A` null.
- **Lanes agree.** On a three-level tree with siblings at shared and unshared
  boundaries, every position from 0 to 1 in steps of `1/100` gives the same
  set of timelines in the root's stamp, its table row and the bundle's stamp,
  on an exact (quarters), a float (seconds) and an int (samples) axis, and the
  table's values equal the stamp's.
- **A zero-length child** is held at its offset, which is its end, even where
  a sibling starts there too; it takes no position from the child ending
  there either: `Z` (length 0) and `C` (length 4) at 4 with `A` (length 4) at
  0: at 4 the stamp holds `C` 0 and `Z` 0, not `A`; with `C` removed it holds
  `A` 4 and `Z` 0.
- **`get_children_at`** lists exactly the children the stamp names: `B` alone
  at 4 in the first case; a lone child ending at 4 inside a parent of length 8
  at 4 (formerly left out, its end then on no child).

- **A timeline placed in two parents is answered through the tree
  queried.** `add_child` still accepts a timeline that already has a
  parent; it then sits in both parents' children while naming the second
  as its parent. `CH` (length 2, holding `G` of length 1 at 0) placed in
  `P1` (length 8) at 1 and then in `P2` (length 8) at 3: `P1` at 2 names
  `P1` 2, `CH` 1, `G` 1, and `P2` at `7/2` names `P2` `7/2`, `CH` `1/2`,
  `G` `1/2`, each stamp equal to its table row. A stamp that found the way
  from `CH` up through its parent reference left `P1`'s tree at `P2` and
  failed; one walking down the queried tree cannot. Whether `add_child` should refuse such a timeline is not
  decided here.

| Test | Validates |
|------|-----------|
| `test_exact_parent_to_child` | `_get_child_coordinate()` returns the exact result |
| `test_exact_child_to_parent` | `_get_parent_coordinate_from_child()` returns the exact result |
| `test_out_of_bounds_returns_none` | `None` for a coordinate outside the child span; the end of a lone child is held |
| `test_shared_boundary_belongs_to_the_later_sibling` | Where one child ends and another starts, only the later one holds the position, in stamps and tables alike |
| `test_unknown_child_returns_none` / `test_unknown_child_raises_keyerror` | Unknown ids keep their established return or exception |
| `test_recursive_offset_exact` | Offsets compose through a grandchild |
| `test_recursive_fraction_offset_exact` | Rational offsets compose without float coercion |
| `test_timestamp_uses_offset_arithmetic` | A `TimeStamp` resolves a child through its offset |
| `test_timestamp_out_of_bounds_returns_none` | A stamp outside the child span raises `KeyError` for the child |
| `test_timestamp_boundary_values` | Start inclusive; the end of a child nothing follows inclusive |
| `test_no_interpolation_maps_attribute` | A timeline holds no interpolation maps |
| `TestContainmentAlongThePath` | The grandchild at a shared boundary, in stamp, table, bundle and `get_children_at`; a timeline's own end |
| `TestExactTableContainment` | The two exact-axis cases |
| `test_lanes_agree_at_every_position` | Stamp, table and bundle agree on three axes, values included |
| `TestZeroLengthChild` | A zero-length child is held at its offset and takes no end away |
| `test_a_timeline_in_two_parents_is_answered_through_the_tree_queried` | The stamp and table of each parent follow that parent's own offsets |

### `test_fraction_generation.py` - Exact Logical Coordinate Generation

These tests author fraction-typed metrical events on a logical timeline and
inspect the stored PyArrow coordinate structs. Beat and measure event
coordinates must retain the exact numerator/denominator pairs derived from the
musical positions, including fractional positions such as `Fraction(3, 2)`.
Eighth-note beats are the load-bearing choice: `1/2` has no exact float, so a
lane that rounded on the way into storage shows up in the stored pair. The
tests also verify that `Timeline.from_events()` preserves an exact maximum
coordinate as the timeline length. These assertions protect the rational fields
that downstream alignment and serialization use; the convenience float field
alone is not sufficient.

Two earlier tests here drove the same storage lane through timeline-side
metrical *generators* that no longer exist. Their subject was the generator,
not the storage, and the surviving test covers the storage claim in one place
rather than twice; the beat lattice's own exactness is pinned in
`test_beatgrid.py`.

### `test_fraction_engines.py` - Exact Coordinate Propagation

These tests exercise exact coordinate handling across nested timelines and
segment-based copies. Event coordinates entered as `Fraction` values are
stored as Arrow coordinate structs with a float projection and an authoritative
numerator/denominator pair. When an event is shifted by a child offset or a
segment start, the tests assert the resulting `Fraction` values and the exact
stored pairs, so the float projection and rational pair are updated together.

The event checks also reject any returned coordinate struct whose float value
does not equal the value represented by its pair. This protects both the
single-event lookup path and the bulk event path, including instant and
interval coordinate fields. Segment extraction and region-based child creation
are checked independently: each must subtract the exact segment or region
start while preserving the resulting rational representation in the copied
event data.

The timestamp checks distinguish the typed default from explicit projections:
`get_coordinate_for()` returns an `IdCoordinate`, while
`format="coordinate"` returns the canonical `Coordinate` and
`format="float"` performs a requested numeric projection. An interpolated
timestamp remains marked with `is_interpolated=True` and is re-expressed in
the axis's declared canonical number type.

**Boundaries reach the event paths in the axis's number type**
(`TestBoundariesInTheAxisType`). `get_events(min_coord=, max_coord=)`
compares exactly on an exact axis, so a library caller that turns an exact
boundary into a float first moves it: `float(Fraction(1, 10))` is a little
more than `1/10`, `float(Fraction(3, 10))` and `float(Fraction(7, 10))` a
little less. Three callers did, and their parts lost or gained the event at
the boundary; the shifted child-local starts and segment lengths were the
doubles' dyadics (`-1/90071992547409920` and the like) rather than tenths.
The specimen is ten Notes `n0`…`n9` on `[k·s, (k+1)·s)`, boundaries at
`1·s`, `3·s`, `7·s`, with `s = 1/10` on a `fraction` quarters axis, where
the defect shows, and `s = 10` on an `int` ticks axis and `s = 0.5` on a
`float` seconds axis, whose boundaries no float conversion moves and whose
results are therefore those of before (each axis case run against the code
before the fix: the fraction cases fail there, the int and float ones pass).
For each axis every part is asserted on its exact event IDs, the child-local
start of each event (`j·s`), its offset and its length (`1·s`, `2·s`, `4·s`,
`3·s`), each value with its Python type, so a length that equals the
expected value but arrives as a float on an exact axis fails:

| Call | Parts asserted |
|---|---|
| `create_child_from_region` on `[1·s, 3·s)` | one child: `n1`, `n2` at `0`, `1·s`; offset `1·s`, length `2·s` |
| `create_segment_line_by_splitting` at the ends of `n0`, `n2`, `n6` | `[n0]`, `[n1, n2]`, `[n3 … n6]`, `[n7, n8, n9]` |
| `SegmentLine.from_segmentation([0, 1·s, 3·s, 7·s, 10·s])` | the same four parts |

Two further readers converted the same way and are pinned on the fraction
axis, each with the boundary the double moves to the wrong side:
`get_slice(0, 1/10)` over Beats at `k/10` holds `b0` alone (the Beat at
`1/10` is the slice's end, exclusive), `get_slice(3/10, 7/10,
truncate_events=False)` over the Notes holds `n3`…`n6` (each fully inside,
`n6` ending at the slice's end), and `get_regions_at(3/10)` with regions
`[1/10, 3/10)` and `[3/10, 7/10)` names the second one only.

**A float query compares as a key; an exact one exactly**
(`TestQueriesOnExactAxes`). A position written as a float names the value
it denotes: it is compared with the float side of the stored cells (the
nearest double to each stored value), so a decimal finds the stored ratio
it was written for. A position written exactly (an `int`, a `Fraction`)
is compared exactly with the ratios and never rounded onto the axis. The
two readers are `get_events(min_coord=, max_coord=)` and `get_events_at`;
both are asserted on event IDs. What would break each row: a float
re-expressed as its double's exact ratio misses `1/10` at `0.1`; a float
rounded onto an `int` axis puts `10.5` at `10`; an exact query compared as a
float finds `1/10` at `Fraction(0.1)`, the double's ratio, which is not
`1/10`.

On a `fraction` quarters timeline with Beats `tenth` at `1/10` and
`third` at `1/3`:

| Query | Asserted |
|---|---|
| `get_events(min_coord=0.1)` | `tenth`, `third` |
| `get_events(max_coord=0.1)` | nothing (`tenth` is not below `0.1`) |
| `get_events(min_coord=Fraction(1, 10))` | `tenth`, `third` |
| `get_events(min_coord=Fraction(0.1))` | `third` (the double's exact ratio lies above `1/10`) |
| `get_events_at(0.1)` | `tenth` |
| `get_events_at(1 / 3)` | `third` |
| `get_events_at(Fraction(1, 10))` | `tenth` |
| `get_events_at(Fraction(0.1))` | nothing |

On a `fraction` quarters timeline with Notes `b` on `[1/3, 2/3)` and `c` on
`[1/2, 1)`, `get_events_at(2 / 3)` finds `c` alone: `b`'s end is the
double `0.6666666666666666`, equal to the query, and an interval holds its
start, not its end. `get_events_at(Fraction(2, 3))` finds `c` alone too,
exactly. On an `int` ticks timeline with Notes `x` on `[10, 11)` and `y` on
`[11, 12)`: `min_coord=10.5` keeps `y` only, `max_coord=10.5` keeps `x`
only, `get_events_at(11.5)` finds `y`, `min_coord=Fraction(21, 2)` keeps `y`
and `get_events_at(Fraction(23, 2))` finds `y` (exact, not rounded to 12,
where `y` would not hold).

**A float key meets a child's exact boundaries on the axis it was written
on** (`TestFloatKeysAcrossChildren`). A float position is never carried
down by float subtraction: `0.3 - 0.1` is `0.19999999999999998`, which lies
inside a child of length `1/5` that ends at `3/10`, and `0.4 - 0.3 - 0.1`
is not 0, so a query subtracting doubles counts one boundary in two siblings
and misses an instant at a grandchild's start. `get_events_at` carries the
exact offsets of the path instead and compares the key with the double of
each exact position on the queried axis: a child holds the key where
`float(o) <= key < float(o + extent)` (`o` the child's exact position on the
queried axis), its end too where no sibling starts there (the rule of
`_holds_position`), and a row is compared at `float(o + start)`, the very
double `get_events(min_coord=, max_coord=)` compares, since it shifts each
child row exactly onto the queried axis first. So both readers agree on
every boundary: a position is a child's boundary in one reader exactly
when it is in the other. On a `fraction` quarters parent of length 1, with
Notes as named:

- `A` at `1/10` (length `1/5`, Note `a_last` on its `[1/10, 1/5)`, i.e.
  `[1/5, 3/10)` on the parent) beside `B` at `3/10` (length `7/10`, Note
  `b_first` on its `[0, 1/10)`), `B` holding `B1` at `1/10` (length
  `1/10`, a Beat `b1_start` at its 0, i.e. `2/5` on the parent):
  `get_events_at(0.3)` and `get_events_at(Fraction(3, 10))` find
  `{B: [b_first]}` (the boundary is `B`'s alone, `A` does not hold it);
  `get_events_at(0.4)` and `Fraction(2, 5)` find `{B1: [b1_start]}`
  (`b_first` ends there); `get_events(min_coord=0.3)` holds `b_first` and
  `b1_start` and `get_events(max_coord=0.3)` holds `a_last` alone, the
  same side of `0.3` as `get_events_at` places them.
- `C` at 0 (length `1/10`, the parent's only child, Note `c_note` on
  `[0, 1/10)` and a Beat `c_end` at `1/10`, its end): `get_events_at(0.1)`
  and `get_events_at(Fraction(1, 10))` find `{C: [c_end]}`, since nothing
  follows `C` and it holds its end; `get_events(min_coord=0.1)` holds
  `c_end` alone and `get_events(max_coord=0.1)` `c_note` alone.
- A child at offset `1/10` with a Beat at its 0 is found at
  `get_events_at(0.1)`, under the child's ID.

**Infinite and NaN positions** (`TestNonFinitePositions`). An infinite
float is an ordinary double compared as a key: on a seconds (`float`), a
quarters (`fraction`) and a ticks (`int`) timeline with Notes at 1 and 2,
`get_events(max_coord=math.inf)` and `get_events(min_coord=-math.inf)`
return both, `get_events(min_coord=math.inf)` and `max_coord=-math.inf`
none, and `get_events_at(math.inf)` finds nothing. NaN names no position:
`get_events(min_coord=math.nan)`, `max_coord=math.nan` and
`get_events_at(math.nan)` raise `ValueError` `NaN names no position: it
compares with no cell` on every axis, instead of answering an empty result
nobody asked for.

**A timeline's events are not handed out** (`TestGetEventsIsACopy`).
`get_events()` without a filter returns an `EventData` of its own: `is` the
timeline's `events` is false, and `extend` on it with a Note at `[10, 11)`
leaves a timeline of length 4 holding its one event, length 4.

### `test_base.py` - Core Timeline Functionality

**Purpose:** Validates the fundamental Timeline contract and behavior.

**Test Categories:**

1. **Construction Tests** (12 tests)
   - Empty timeline creation with defaults
   - Timeline with explicit length, unit, number_type
   - Custom ID vs auto-generated ID
   - Locked state on creation
   - Factory methods: `from_events()`, `from_event_store()`
   - The experimental base class remains available as an explicit
     `create_child(..., child_class=Timeline)` override

2. **Coordinate Factory Tests** (6 tests)
   - `make_coordinate()` preserves value types (int, float, Fraction)
   - Origin is always zero
   - Start/end property aliases

3. **Event Management Tests** (9 tests)
   - Adding instant events, interval events, mixed events
   - `get_events()` filtering by temporal_type, event_type
   - Segment events excluded by default

   **`add_events` stores positions on the timeline's axis and IDs unique on
   it (`TestAddEventsAxisAndIds`).** What proves it, on a samples timeline
   (`int`) with a samples→seconds `LinearMap` of scalar `1/44100`:
   - a `Duration` in seconds added to a start in seconds is the span from
     that start: start `Coordinate(1, s)`, duration `Duration(1/2, s)` store
     start 44100, duration 22050 and end 66150 samples, each an `int` cell
     (numerator over denominator 1); before, a duration was never converted;
   - a `Duration` already in samples is stored as its value (`Duration(10,
     samples)` from start 0: duration 10, end 10);
   - a duration in seconds added to a start given as a bare number is
     refused with the message naming both units, since the span has no
     start in its own unit; nothing is added (the table length is
     unchanged);
   - an ID already held by an event of the timeline is refused naming the ID
     and the timeline, and so is one given twice in one call; nothing is
     added either time, so the refusal is atomic;
   - two calls without IDs mint `note_1` and then `note_2`, never the same
     ID twice (a per-call counter minted the first ID twice);
   - a minted ID continues above the highest held: a timeline holding the
     given `note_5` mints `note_6` for the next Note, never `note_1` below
     it (an ID is never given again);
   - `add_events` returns the IDs of the rows it added, in row order, the
     minted ones included (`beat_1`, `given`, `beat_2`).

4. **Length and Expansion Tests** (7 tests)
   - Length can expand (unlocked)
   - Length cannot contract below content
   - Auto-expansion behavior
   - `allow_expansion` override for locked timelines

5. **Locking Tests** (9 tests)
   - Default unlocked state
   - Children become locked
   - Locked timelines reject expansion
   - `lock()` locks a timeline once built, so a root that grew by appending
     (a segment line above all, whose segments cannot be placed into a line
     locked at construction) can end locked. What proves it: a root built
     unlocked reports `is_locked` False and True after `lock()`; the locked
     root refuses growth — `append_child` raises `RuntimeError`, `add_child`
     of a child ending beyond the length raises `ValueError` — while a child
     that fits is still added; `to_dict()["locked"]` is True and
     `Timeline.from_dict(to_dict())` is locked with an equal `to_dict()`; a
     `SegmentLine[ContinuousLogicalTimeline]` of three segments (4, 2 and 6
     quarters) locked after building keeps its segment order, its length 12
     and `is_segment_line`, refuses `append_segment` with `RuntimeError` and
     round-trips through its dict form, rebuilt locked; `lock()` on a child,
     locked already, and a second `lock()` on a root leave it locked and
     raise nothing. There is no `unlock`.

6. **Serialization Tests** (5 tests)
   - `to_dict()` / `from_dict()` roundtrip; `to_dict()` omits events by
     default, so the event-bearing cases pass `events=True` explicitly
     (see `test_external_references.py` for the full flag matrix)
   - Children serialized recursively
   - Segment events regenerated on deserialization
   - The class is read from the unit: `Timeline.from_dict` returns each
     concrete timeline type for its unit, and a call through a concrete
     subclass the unit does not name is refused
   - `length` and every child `offset` are written as bare values on the
     timeline's own axis, represented as its number type selects; the same
     JSON number representation carries event coordinates, so a `Fraction`
     timeline serializes to JSON and comes back exact. The format and its
     fixpoint guarantee are specified in `tests/core/README.md`
     (`test_number_json.py`), which also holds the exact-value tests for
     timeline lengths and offsets.

7. **Magic Methods Tests** (5 tests)
   - `__len__`, `__repr__`, `__str__`, `__contains__`

8. **Future API Stubs Tests** (2 tests)
   - `add_conversion_map()`, `convert_to()` raise `NotImplementedError`
9. **Scale Tests** (1 test)
   - Adding 10,000 instants in one call holds exactly 10,000 events (the
     input's count). No wall-clock threshold: a time budget fails under load
     and proves nothing about the result.

**Validity Rationale:**

These tests verify the fundamental Timeline contract:
- A Timeline is a positive coordinate axis with an origin (zero) and measuring unit
- Events are stored efficiently in an EventStore (PyArrow-backed)
- Timelines can expand (if unlocked) but cannot contract below content
- Coordinates are validated against the timeline's unit

---

### `test_projections.py` - Projections added to a timeline

**Purpose:** A Projection is an event standing on a timeline for an event
not held there (the scalar is tested in `tests/core/test_projection.py`). It
is added like any event, through `Timeline.add_events` with `event_type`
`"Projection"`, and is stored on the timeline's own axis: a position given in
a unit the timeline converts is converted when the Projection is added, never
kept in the other unit. These tests prove that a timeline's events hold
Projections under that one mechanism, with exact positions, their reference
kept as an object, and IDs minted by the ID scheme.

**What proves it** (a samples timeline, `int`, length 1,000,000, with a
samples→seconds `LinearMap` of scalar `1/44100`; and a quarters timeline,
`fraction`, with a quarters→seconds `LinearMap` of scalar `1/2` declared to
write fractions):

- **Instant, converted.** `Coordinate(1/2, seconds)` is stored at sample
  22050 (cell `{22050.0, 22050, 1}`) with null `end` and `duration` and
  `temporal_type` `"instant"`; the float `Coordinate(19.278367347,
  seconds)` of the LayerIt alignment lands on sample 850176. On the
  quarters timeline `Coordinate(3/4, seconds)` is stored at 3/2 quarters
  (cell numerator 3, denominator 2): the number type of the timeline is
  kept, never a float.
- **Interval, converted.** Start `Coordinate(1, s)` with
  `Duration(1/2, s)` is stored as start 44100, duration 22050, end 66150,
  `temporal_type` `"interval"`.
- **The reference is an object.** `projection_of` given as a dict
  `{timeline_id, event_id}` or as an `EventReference` is stored in the
  struct column `projection_of` (`{timeline_id: string, event_id:
  string}`, its field metadata naming `EventReferenceField`), never as a
  joined string; `None` is a null cell. A batch of Projections all without
  a reference still types the column as that struct, so a later batch with
  one concatenates (the column of the second Projection reads back its
  dict).
- **IDs.** A source's ID is kept (`x1azmn68`); a missing one is minted
  `projection_N`, N above the highest held among the timeline's events of
  every type, so no ID is given twice and a gap is never filled: the first
  call mints `projection_1`, `projection_2`; with `projection_2` held by a
  Note, the next call mints `projection_3` and `projection_4` for two rows
  (the call returns them in row order), never `projection_1` below it.
- **A new Projection's ID is unique in its manifest** (the timeline's tree
  and every tree of the groups and bundles holding it, as a bundle writes
  one Projections table for all of them; TimeToShare's table check keys
  the ID alone). In a bundle holding `dpt_1` and `clt_6` (both roots),
  `add_projections` of two Projections on `dpt_1` mints `projection_1`,
  `projection_2`; one Projection then added to `clt_6` through
  `add_events` is minted `projection_3`, never `projection_1` again. In
  one tree without a bundle the same holds: with `projection_1` on
  `dpt_1`, a Projection on its child `part` is minted `projection_2`. A
  timeline in no manifest with them mints on its own: a free `clt_6`
  holding nothing mints `projection_1` (the scope is the manifest, not
  the process). A given Projection ID another timeline of the manifest
  holds as a Projection is refused, atomically (`clt_6` still holds 0
  events), naming both timelines — through `add_events`, `Projection ID
  'projection_1' names a Projection on the timeline 'dpt_1' and one on
  'clt_6': a Projection's ID is unique among all Projections of a
  manifest and its package`, and through `add_projections` (get-or-add
  reuses a Projection of its own timeline only), the same message. Nothing
  here mints again: a test that passed with per-timeline minting would
  see `clt_6` mint `projection_1` and accept the given one.
- **A child joins whatever Projection IDs it holds.** The rule binds the
  creation of a Projection, not a join: a timeline that created its
  Projections before joining keeps their IDs (a MatchClaim names an event
  by its timeline and ID, so it stays unambiguous in memory; the package
  writer refuses the shared ID, tested in `tests/alignment/`). `dpt_1`
  holding `projection_1` takes a free `part` whose own `projection_1` was
  minted alone as a child at offset 0: `dpt_1` then has one child, `part`
  has `dpt_1` as its parent and still holds `projection_1` (never minted
  again). The manifest now holds `projection_1` twice, and that refuses
  nothing on creation: a Projection then added to `dpt_1` without an ID
  is minted `projection_2` (above every Projection of the manifest), one
  given the free ID `onset_a` on `part` is added, and one given
  `projection_2` on `part` is refused, `Projection ID 'projection_2' names
  a Projection on the timeline 'dpt_1' and one on 'part': ...`, `part`
  keeping its two events. A check that compared the manifest's IDs among
  themselves would refuse both of the first two. In a bundle, `dpt_1`
  takes `part` likewise although `clt_6`, another root of the bundle,
  holds `projection_1`.
- **A Projection object stands for its row.** `add_events` takes a
  `Projection` alone or in a list beside dicts: `Projection(start=
  Coordinate(1/2, seconds), projection_of=EventReference(clt_6, x1azmn68))`
  without an ID is minted `projection_1` and stored at sample 22050 with its
  reference, exactly as the same dict; with `duration=Duration(0.5,
  seconds)` from start `Coordinate(1.0, seconds)` (one axis, both float) it
  is stored as start 44100, duration 22050; a Projection given its ID keeps it. A Note dict
  beside it in the same list mints `note_1`.
- **Refusals**, each atomic (the number of events unchanged):
  - an ID held by any event of the timeline (a Note's ID reused by a
    Projection);
  - a row giving a property the Projection does not have (`pitch`):
    `A Projection holds its ID, its position and projection_of, nothing else
    of the event it stands for; the row also gives ['pitch']`;
  - a `projection_of` that is not an event reference (the joined string
    `"clt_6:x1azmn68"`);
  - a `projection_of` naming the timeline the Projection is added to (a
    Projection stands for an event of another timeline, as TimeToShare's
    table check requires): `A Projection added to timeline 'dpt_1' stands
    for an event of another timeline; its projection_of names 'dpt_1'
    itself`.

**Get-or-add: `Timeline.add_projections`.** A timeline keeps one Projection
per thing it stands for. What proves it is the identity of the returned IDs
and the number of rows the timeline holds afterwards, never a count of calls:

- **A second call reuses every Projection.** On the samples timeline,
  `add_projections` of a Projection of `clt_6:n1` at `Coordinate(1/2, s)`,
  one of `clt_6:n2` at the same position, one of no event at
  `Coordinate(1, s)` and one of no event from `Coordinate(1, s)` for
  `Duration(1/2, s)` returns `projection_1` … `projection_4`; the same call
  again returns the same four IDs and the timeline still holds four events.
  The position is compared after conversion, exactly: `22050` given as a
  bare sample number is `projection_1` again.
- **Equal rows of one call are one Projection.** Three rows standing for
  no event at sample 22050 and one at 44100 give `['projection_1',
  'projection_1', 'projection_1', 'projection_2']` and two events.
- **What tells Projections apart.** The event stood for (a Projection of
  `n1` and one of no event at one position are two), the start (`n1` at two
  positions, a repeat played against one score note, are two), and the
  duration (an instant and a span from one start are two).
- **One value, one key, however the cell is spelled.** The keys compare a
  cell's members, so they compare values only because memory holds one
  spelling of a value. On the quarters timeline a Projection row given its
  start as the whole cell `{0.5, 2, 4}` is stored as `{0.5, 1, 2}` and
  added as `projection_1`; `Projection(start=Coordinate(1/2, quarters))`
  then returns `projection_1` and the timeline holds one event. The same
  holds for an event table given whole: `EventData.from_arrays` with a
  `start` struct column `[{0.5, 2, 4}, {0.75, 3, 4}]` stores
  `[{0.5, 1, 2}, {0.75, 3, 4}]`.
- **A given ID.** A Projection given `x1azmn68` is added under it; the same
  Projection given again returns `x1azmn68` and adds nothing. Refused,
  atomically: the ID held by a Note, the ID held by a Projection at another
  position (`Event id 'x1azmn68' is already held by another event of
  timeline 'dpt_1'`), the ID given twice in one call for two positions
  (`Projection id 'p' is given twice for different positions in one call on
  timeline 'dpt_1'`), and a row that is no Projection (`add_projections adds
  Projections, not a row of type 'Note'`).

---

### `test_boxes.py` - Boxes added to a timeline

**Purpose:** A Box is an area of an image held as an interval event of a
timeline running along the image's x axis (the scalar is tested in
`tests/core/test_box.py`). It holds two statements of one x extent: its
shape, in the image's pixel space, and its interval on the timeline. They
agree by one invariant, `start = x_min − o(T)` and `duration = x_max −
x_min`, where `o(T)` is the x at which the timeline stands in its image: the
`x_min` of the shape its `meta["box"]` states, 0 where it states none.

**One check, every entry path.** A Box reaches a timeline's events as a
`Box` scalar, as a dict row (`add_events`, `Timeline.from_events`), as an
event row of a manifest (`Timeline.from_dict`, which an `AlignmentBundle`
read uses too) and as a row of a package's Box table (`boxes.parquet`). Every
one of these paths passes the same rules, in this order, for all rows before
any is added: exactly one shape; a duration; `start = x_min − o(T)`;
`duration = x_max − x_min`. The Box's own rules (one shape, its duration) are
refused with the `Box` scalar's own message, whichever path brought the row,
so a dict and a manifest cannot be refused differently from the scalar. A
shape is never converted or rounded onto an axis: a shape in another unit or
number type than the timeline is refused before any position is converted,
by the one rule `EventData.from_dicts` applies (its message is the table's,
`tests/storage/README.md`). What would show it broken: a Box accepted whose
interval contradicts its shape on any one path (a reader of the interval and
a reader of the shape would then name two areas, and a package written from
it would be refused on reading), a refusal that leaves part of a batch
added, or one rule worded two ways.

**One dict form.** `Timeline.to_dict(events=True)` writes every number of a
Box's shape as the contract's JSON numbers (`number_to_json`: an int an
integer, a float a number, a Fraction a `[numerator, denominator]` pair),
the shape's own `to_dict()`; no memory cell (`{value, numerator,
denominator}`) is written, so no number is written twice and no written
number can contradict itself. Reading takes only that form
(`Shape.from_dict` on the timeline's axis); a memory cell in a manifest is
refused. What proves it: the JSON text of the written row, asserted
literally per number type (the text tells `385` from `385.0`, which a Python
equality would not), and the exact round trip `to_dict → from_dict →
to_dict` with equal event cells. What would show it broken: a float beside a
Fraction in the written shape, or a written dict the reader refuses.

**What proves it** (values read from the sources, so the test is a check of
the sources' own arithmetic, never a restatement of the implementation):

- **Thoresen's `rect_h2`, accepted on its crop.** The Thoresen TSV places
  `rect_h2` on an 875 × 270 image at `{x: 385, y: 46, width: 139, height:
  20}`; the 2010 example crops that image at x0 = 7 to x1 = 874 as its
  second segment. `seg_2` (`DiscreteGraphicalTimeline`, pixels, int, length
  867, `meta={"box": {"x": 7, "y": 0, "width": 867, "height": 270}}`) takes
  `Box.from_shape(BoundingBox.from_xywh(385, 46, 139, 20, unit="pixels"),
  origin=7, id="rect_h2")`: `add_events` returns `["rect_h2"]`, the stored
  `start` cell is `{378.0, 378, 1}`, `duration` `{139.0, 139, 1}`, `end`
  `{517.0, 517, 1}`, `bounding_box.x` `{385.0, 385, 1}` and `quadrilateral`
  null. As the second segment of the segment line `dgt_2`, after `seg_1`
  (length 866, the first crop 8 to 874), `dgt_2.get_timestamp_for("rect_h2")`
  spans [1244, 1383) (866 + 378, 866 + 517), the How-to's existing check.
- **The same Box without `meta["box"]` is refused.** On a `seg_2` stating
  no box the timeline stands at x 0, so the Box's start must be 385:
  `Box 'rect_h2' starts at 378 on timeline 'seg_2', but its shape starts at
  x 385 and the timeline stands at x 0 of its image (meta["box"]): its
  start is 385.`, and `seg_2` holds no event and keeps its length 867.
- **Off the axis, refused before any conversion.** The same rectangle in
  `css_pixels` as float (`BoundingBox.from_xywh(385.0, 46.0, 139.0, 20.0,
  unit="css_pixels")`, origin 7.0) on `seg_2`, whose start is a
  `Coordinate` in `css_pixels` that `seg_2` has no C-Map for: the shape is
  refused, not the conversion, `The event's bounding_box is in css_pixels as
  float, the events are in pixels as int; a shape is never converted onto
  the events' axis.`, and `seg_2` holds no event.
- **A Box made without an ID is named `A Box`** in the start message (`A
  Box starts at 378 on timeline 'seg_2', ...`) and in the Box's own
  duration message (`A Box: its duration is ...`): the ID the timeline would
  mint is never shown.
- **A quadrilateral on a child timeline.** DMOS's *Aimons-nous* page 1
  (`page`, pixels, length 4512, meta box `(0, 0, 4512, 6174)`) holds
  system 1 as its child `sys` at offset 1221, length 2880 (4101 − 1221),
  whose `meta["box"]` is the system's quadrilateral (corners (1221, 1589),
  (4097, 1589), (4101, 2266), (1221, 2266)), so `o(sys)` = 1221. Measure 1
  (corners (1221, 1589), (2151, 1589), (2161, 2266), (1221, 2266)) and
  measure 2 ((2151, 1589), (2766, 1589), (2775, 2266), (2161, 2266)), as
  `Box.from_shape(q, origin=1221, id=...)`, are accepted with starts 0 and
  930, durations 940 and 624; their `quadrilateral` cells' `bottom_right`
  are (2161, 2266) and (2775, 2266), `bounding_box` null.
- **Atomic in a batch.** `[m1 at origin 1221, m2 at origin 0]` on a fresh
  `sys`: the second row is refused (`Box 'm2' starts at 2151 on timeline
  'sys', but its shape starts at x 2151 and the timeline stands at x 1221
  of its image (meta["box"]): its start is 930.`) and `sys` holds no event,
  the first row included.
- **Exact on a fraction axis.** `uu` (`ContinuousGraphicalTimeline`,
  `user_units`, fraction, length 2) takes the Box `f` of
  `BoundingBox.from_xywh(1/3, 0, 2/3, 1)` at origin 0: start cell `{1/3, 1,
  3}`, duration `{2/3, 2, 3}`, `bounding_box.width` `{2/3, 2, 3}`. With
  `meta["box"] = {"x": [1, 3], "y": [0, 1], "width": [5, 3], "height": [1,
  1]}` (the fraction pairs `Shape.to_dict` writes), `o(uu)` = 1/3: `f` at
  origin 1/3 starts at 0 and is accepted; `f` at origin 0 is refused, `Box
  'f' starts at 1/3 on timeline 'uu', but its shape starts at x 1/3 and the
  timeline stands at x 1/3 of its image (meta["box"]): its start is 0.`
  The comparison is exact: a float detour would still agree here, so the
  float-free path is pinned by the cells holding numerator and denominator.
- **Other events pass unchanged.** A Note dict beside a Box in one call is
  added with it (`["note_1", "rect_h2"]`); a timeline holding no Box never
  reads its `meta["box"]` (a malformed one, `{"x": "left"}`, refuses
  nothing when a Note is added).

**Every path refuses alike** (`TestOneCheckEveryPath`; on `seg_2`, whose
`o` is 7, so `rect_h2` starts at 378 and spans 139, x 385 to 524; each
refusal leaves `seg_2` with no event and its length 867):

- **The duration.** The scalar `Box(id="a", start=378 px, duration=10 px,
  bounding_box=rect_h2)` is refused at construction with `Box 'a': its
  duration is 10 and its shape spans 139 in x (385 to 524); a Box's
  duration is its shape's x extent.` (inside pydantic's report); the dict
  row of the same numbers given to `add_events` is refused with exactly that
  sentence, and so is the dict row stating `end` 400 in place of a duration
  (`its duration is 22`, the 400 − 378 its interval states). A dict row
  stating no duration and no end: `Box 'a': it has no duration; a Box's
  duration is its shape's x extent.`
- **One shape.** A dict row holding no shape, and one holding both
  `rect_h2` and its quadrilateral (corners (385, 46), (524, 46), (524, 66),
  (385, 66)), are refused with the Box's own `A Box holds exactly one shape:
  give bounding_box or quadrilateral.`
- **Atomic.** A batch of a valid Note and the duration-10 Box adds nothing.
- **`Timeline.from_events`** takes dict rows of a source as they stand and
  refuses the duration-10 Box with the same sentence.
- **A manifest.** `seg_2` holding `rect_h2`, written by
  `to_dict(events=True)` and edited, is refused by `Timeline.from_dict`:
  its shape's `width` set to 10 (the row's interval kept, 378 to 517), `Box
  'rect_h2': its duration is 139 and its shape spans 10 in x (385 to 395);
  ...`; its `start`, `end` set to 1, 140 (an interval of 139 still),
  `Box 'rect_h2' starts at 1 on timeline 'seg_2', but its shape starts at x
  385 and the timeline stands at x 7 of its image (meta["box"]): its start
  is 378.`; its `bounding_box` set to null, the one-shape sentence.
- **A package.** A bundle holding `seg_2` writes `boxes.parquet` beside its
  manifest; the table's `start` of `rect_h2` changed to 1 (the package
  form of an int, numerator 1, denominator and value null) is refused by
  `AlignmentBundle.from_file` with the start sentence above for start 1,
  and its `duration` changed to 10 with the duration sentence (inside
  pydantic's report: the package reader builds the `Box` itself).

**Written and read back** (`TestDictForm`; the JSON text of the written
Box row, and `Timeline.from_dict(d).to_dict(events=True) == d` with equal
event cells):

- **int.** `rect_h2` on `seg_2`, added as a dict row whose `bounding_box` is
  the `BoundingBox` scalar: `"start": 378, "end": 517, "duration": 139,
  "bounding_box": {"x": 385, "y": 46, "width": 139, "height": 20}` and no
  `quadrilateral` key (no row of the table gave that column).
- **fraction.** `f` (`BoundingBox.from_xywh(1/3, 0, 2/3, 1)`) on `uu`:
  `"start": [1, 3], "end": [1, 1], "duration": [2, 3], "bounding_box":
  {"x": [1, 3], "y": [0, 1], "width": [2, 3], "height": [1, 1]}`: a
  Fraction is a pair even where it is whole.
- **float.** `q`, the quadrilateral (10.5, 1.0), (20.25, 1.0), (20.25,
  5.5), (10.5, 5.5) in `css_pixels` on `fl` (`ContinuousGraphicalTimeline`,
  float, length 900.0): `"start": 10.5, "end": 20.25, "duration": 9.75,
  "quadrilateral": {"top_left": {"x": 10.5, "y": 1.0}, "top_right": {"x":
  20.25, "y": 1.0}, "bottom_right": {"x": 20.25, "y": 5.5}, "bottom_left":
  {"x": 10.5, "y": 5.5}}`: `1.0`, never `1`.
- **A child and its package.** The DMOS page with `sys` holding `m1` reads
  back from its dict with `m1`'s quadrilateral `{"top_left": {"x": 1221,
  "y": 1589}, "top_right": {"x": 2151, "y": 1589}, "bottom_right": {"x":
  2161, "y": 2266}, "bottom_left": {"x": 1221, "y": 2266}}`, and a bundle of
  the page written by `to_file` and read by `from_file` holds `sys`'s event
  rows equal, cell for cell.
- **A memory cell in a manifest is refused.** The written `bounding_box`
  replaced by its memory cells (`{"x": {"value": 385.0, "numerator": 385,
  "denominator": 1}, ...}`, as memory holds them): `Box 'rect_h2' of
  timeline 'seg_2': its bounding_box is not the dict form of a BoundingBox
  in pixels as int`, followed by the shape's own refusal in parentheses
  (asserted as a prefix: that part is the shape's).

### `test_boxes_rebased.py` - Boxes moved onto a timeline cut from theirs

**Purpose:** Four operations make a timeline from a part of another and move
the source's events onto it, each event's start re-based by the cut's start
`a` (offset arithmetic, `start − a`): `get_slice`, `create_child_from_region` / `create_children_from_regions`,
`create_segment_line` (hence `_from_regions`, `_by_grouping`,
`_by_splitting`) and `SegmentLine.from_segmentation`. A Box holds its
shape in the image's space, so the shape never moves; what moves is the
timeline's place in the image.

**Which timelines this concerns.** `meta` is free-form. A timeline *holds
Boxes* when a Box is among the events `get_events()` gives it: its own or a
descendant's (a page whose systems hold the measures holds them). Only such
a timeline reads `meta["box"]`, as its area of an image. A timeline holding
no Box is cut exactly as before Boxes existed: the new timeline's `meta` is
`{}`, whatever the source's `meta["box"]` holds (a string, a dict of another
form, a valid shape), and nothing is refused for it. A timeline holding
Boxes whose `meta["box"]` is not a shape on its axis is refused, the message
naming the timeline (such a `meta["box"]` cannot reach a timeline holding
Boxes of its own through `add_events`, only one whose descendants hold
them).

**The rule.** The new timeline cut from a timeline holding Boxes states its
own area: its `meta["box"]` is the source's area cut to the new x extent,
`x = o(S) + a`, `width` = the new timeline's length, `y` and `height` the
source's (of its bounding box where the source states a quadrilateral), and
`meta["image"]` is carried over. Then `start − a = x_min − (o(S) + a)` is
exactly the invariant `start = x_min − o(T)` on the new timeline, so every
operation adds the moved rows through the checked path (the slice through
the one Box check of a source's rows, the others through `add_events`) and a
Box contradicting its shape cannot arise. A cut covering the source's whole
extent (`a = 0`, length = the source's extent) stands for the source's area
itself: its `meta["box"]` is the source's as written, a quadrilateral
included.

**What each operation does with an interval crossing the cut, applied to
Boxes unchanged:** the three cut operations take an event when its start
lies in `[a, b)` and keep it whole, the new timeline growing to hold it (a
Box's shape is then whole too, and the area stated is the grown length); an
event starting before `a` is not taken. `get_slice` clips a crossing
interval to `[a, b)` (`truncate_events=True`, the default) or leaves it out
(`False`), and slices every child it reaches with the same flag. A clipped
Box would contradict its shape, so with `truncate_events=True` `get_slice`
refuses a Box crossing either boundary, on the timeline itself or on any
descendant, naming the Box, its timeline, the boundary on that timeline's
axis and `truncate_events=False`, which then leaves it out.

**Known issues, each with its own test.**

- A slice with `truncate_events=True` through a Box is refused (above).
- A proper cut of a timeline whose area is a quadrilateral states the cut
  of the quadrilateral's bounding box (a `BoundingBox`), not the
  quadrilateral clipped.
- A timeline holding Boxes that states no `meta["box"]` gives no y extent,
  and the design has no form for an area without one: a cut at 0 keeps
  every Box valid (both stand at x 0) and states no `meta["box"]`; a cut
  elsewhere that would move a Box is refused, naming the Box, the cut, the
  source and what is missing. A segment line of pages, each stating its own
  area but the line none, is such a timeline: a child cut from it is
  refused as soon as it would move a Box.

**What would show it broken:** a Box whose shape changed; a moved Box whose
start disagrees with its new timeline's `meta["box"]` (the timeline then
cannot be written and read back: the round trip is the proof); a clipped
Box; a slice refused with `truncate_events=False`; a quadrilateral area
rewritten by a cut covering it whole; a timeline without Boxes whose cut
gains or refuses anything (its results were compared with those of the
library before Boxes existed, field for field, and are pinned below as
literals; the existing test files of the four operations run unchanged).

**Values.** `page` (`DiscreteGraphicalTimeline`, pixels, int, length 4564,
`meta = {"box": {"x": 0, "y": 0, "width": 4564, "height": 6174}, "image":
"page_1.jpg"}`) holds the Boxes `b1` = `BoundingBox.from_xywh(1221, 1589,
940, 677)` (x 1221 to 2161) and `b2` = `(2161, 1589, 800, 677)` (x 2161 to
2961), starts 1221 and 2161 (origin 0). Every shape below is asserted equal
to the shape added (`bounding_box` cell `{x 1221, y 1589, width 940, height
677}` for `b1`), and every new timeline round-trips:
`type(t).from_dict(t.to_dict(events=True)).to_dict(events=True) ==
t.to_dict(events=True)` (a segment line and a parent with its child read
back whole, which runs the Box check on every timeline of the tree).

The DMOS page: a page holding no Box of its own (length 4564, the `meta`
above) whose child `sys` (offset 1221, length 2880, `meta["box"]` DMOS's
system 1 quadrilateral `{"top_left": {"x": 1221, "y": 1589}, "top_right":
{"x": 4097, "y": 1589}, "bottom_right": {"x": 4101, "y": 2266},
"bottom_left": {"x": 1221, "y": 2266}}`, x 1221 to 4101, y 1589 to 2266)
holds the measures `m1` (x 1221 to 2161) and `m2` (x 2151 to 2775) at
starts 0 and 930, durations 940 and 624.

- **Slice [1000, 3000).** Length 2000, `meta == {"box": {"x": 1000, "y": 0,
  "width": 2000, "height": 6174}, "image": "page_1.jpg"}`; `b1` start 221
  (1221 − 1000), duration 940; `b2` start 1161, duration 800.
- **Child from the region [1000, 3000).** The child at offset 1000, length
  2000, the same `meta`, `b1` at 221 and `b2` at 1161 (its own rows); the
  page reads back whole.
- **Segment line at [0, 1000, 3000, 4564]** (`create_segment_line` and
  `SegmentLine.from_segmentation` alike): three segments at offsets 0,
  1000, 3000, lengths 1000, 2000, 1564, areas `x` 0, 1000, 3000 with those
  widths (`y` 0, `height` 6174, image carried); the middle one holds `b1`
  at 221 and `b2` at 1161, the others no event.
- **A quadrilateral source, through a child.** The DMOS page's slice
  [1000, 3000) states `_area(1000, 2000)` (the page holds Boxes through
  `sys`) and holds the slice of `sys` at offset 221 (1221 − 1000), length
  1779 (3000 − 1221), `meta["box"] = {"x": 1221, "y": 1589, "width": 1779,
  "height": 677}` (the cut starts at `sys`'s 0, so `x` stays 1221; `y`,
  `height` from the quadrilateral's bounding box), `m1` at 0, `m2` at 930,
  quadrilaterals unchanged.
- **Crossing, slice.** `page.get_slice(1500, 3000)` is refused, `b1` (1221
  to 2161) crossing 1500: `Box 'b1' runs from 1221 to 2161 on timeline
  'page' and crosses the slice's boundary at 1500: a slice clips the
  intervals that cross its boundaries, and a clipped Box would contradict
  its shape. Slice with truncate_events=False to leave it out.` The end
  boundary likewise: `get_slice(1000, 2500)` names `b2` (2161 to 2961) and
  2500. With `truncate_events=False`, `[1500, 3000)` leaves `b1` out and
  holds `b2` at 661, area `x` 1500, width 1500.
- **Crossing, a child's Box.** The DMOS page's `get_slice(1500, 3500)`
  slices `sys` over its own [279, 2279) (1500 − 1221, 3500 − 1221), where
  `m1` (0 to 940) crosses 279: refused, `Box 'm1' runs from 0 to 940 on
  timeline 'sys' and crosses the slice's boundary at 279: ...` (the same
  sentence as above). With `truncate_events=False` the slice (length 2000,
  `_area(1500, 2000)`) holds the slice of `sys` at offset 0, length 2000,
  `meta["box"] = {"x": 1500, "y": 1589, "width": 2000, "height": 677}`,
  holding `m2` alone, at 651 (930 − 279); it round-trips.
- **Crossing, a segment line's segment.** `vol`, a `SegmentLine` (pixels)
  of the pages `p1` and `p2` (each length 1100, `meta` `{"box": {"x": 0,
  "y": 0, "width": 1100, "height": 1500}, "image": "<id>.jpg"}`), `p1`
  holding `a1` = `(200, 100, 300, 200)` and `p2` holding `b2` = `(400, 100,
  400, 200)`. `vol.get_slice(500, 1700)` reaches `p2` over [0, 600), where
  `b2` (400 to 800) crosses 600: `Box 'b2' runs from 400 to 800 on timeline
  'p2' and crosses the slice's boundary at 600: ...`. With
  `truncate_events=False` the slice (length 1200) holds the slice of `p1`
  at 0 (length 600, `meta` `{"box": {"x": 500, "y": 0, "width": 600,
  "height": 1500}, "image": "p1.jpg"}`, no Box: `a1` lies before 500) and
  of `p2` at 600 (length 600, `{"box": {"x": 0, "y": 0, "width": 600,
  "height": 1500}, "image": "p2.jpg"}`, no Box: `b2` left out).
- **Crossing, child and segments.** The child from the region [1500, 2500)
  does not take `b1` (it starts at 1221) and takes `b2` whole (it starts at
  2161): `b2` at 661, the child grown from 1000 to 1461 (2961 − 1500), area
  `x` 1500, width 1461. The segment line at [0, 1500, 2500, 4564]: `b1` in
  the first segment at 1221 (grown to 2161, area width 2161), `b2` in the
  second at 661 (grown to 1461, area `x` 1500, width 1461), the third empty
  (length 2064, area `x` 2500), the segments at offsets 0, 2161, 3622 (the
  line places each after the grown one before it, as for any interval).
- **A cut covering the whole extent keeps the area as written.** The DMOS
  page's `get_slice(0, 4564)`: its `meta` equals the page's (`{"box": {"x":
  0, "y": 0, "width": 4564, "height": 6174}, "image": "page_1.jpg"}`), and
  the slice of `sys` (offset 1221, [0, 2880), its whole extent) has `meta
  == {"box": <the system 1 quadrilateral above>}`, all eight numbers kept;
  `sys.get_slice(0, 2880)` and the child of `sys` from its region [0, 2880)
  likewise, the child holding `m1` and `m2` as `sys` does (`sys` built
  alone here and below: a child timeline is locked and makes no child).
- **Known issue: a proper cut of a quadrilateral area.**
  `sys.get_slice(0, 1600)`: `meta == {"box": {"x": 1221, "y": 1589,
  "width": 1600, "height": 677}}`, the bounding box of the quadrilateral
  (y 1589 to 2266) cut to x 1221 to 2821, a `BoundingBox`; the child of
  `sys` from its region [930, 2880): `{"box": {"x": 2151, "y": 1589,
  "width": 1950, "height": 677}}`, holding `m2` at 0.
- **Known issue: a source holding Boxes without `meta["box"]`.** The same
  page stating no meta: `get_slice(0, 3000)` keeps `b1` at 1221 and `b2` at
  2161 and its `meta` is `{}`; `get_slice(1000, 3000)`, the child from the
  region [1000, 3000) and the segment line at [0, 1000, 3000, 4564] are
  refused with `Box 'b1' cannot move onto a timeline cut at 1000 from
  timeline 'page': the timeline states no area (meta["box"]), so the cut's
  area cannot be stated.`, and the page keeps no child. `vol` above (no
  `meta` of its own), its child from the region [150, 1000): `Box 'a1'
  cannot move onto a timeline cut at 150 from timeline 'vol': the timeline
  states no area (meta["box"]), so the cut's area cannot be stated.`, and
  `vol` keeps its two segments only.
- **`meta` is free-form on a timeline without Boxes.** `t`
  (`ContinuousLogicalTimeline`, quarters, fraction, length 20, uid `t`,
  `meta = {"box": <box>, "image": "scan.png"}`) holding the Notes `n1`
  [0, 5), `n3` [8, 14), `n2` [12, 15), for each `<box>` of `"cardboard"`,
  `{"left": 1, "top": 2}` and the valid quarters shape `{"x": [0, 1], "y":
  [0, 1], "width": [20, 1], "height": [1, 1]}`; each result was compared
  with the library before Boxes existed (`to_dict(events=True)` of every
  result and of the source, equal):
  - `get_slice(10, 20)`: length 10, `meta == {}`, `n3` [0, 4) (clipped),
    `n2` [2, 5); with `truncate_events=False`: `n2` [2, 5) alone;
  - the child from the region [10, 20): length 10, `meta == {}`, `n2`
    [2, 5);
  - `create_segment_line([0, 10, 20])` and
    `SegmentLine.from_segmentation(t, [0, 10, 20])`: segments of lengths
    14 (grown to hold `n3`) and 10, `meta` `{}` and `{}`, holding `n1`
    [0, 5), `n3` [8, 14) and `n2` [2, 5);
  - `t.meta` unchanged after each.
- **A timeline holding Boxes refuses a `meta["box"]` that is no shape.** The
  DMOS page with `meta = {"box": <box>}` (it holds Boxes through `sys`):
  `get_slice(0, 3000)`, the child from the region [0, 3000) and
  `create_segment_line([0, 3000, 4564])` are refused with `Timeline 'page'
  holds Boxes, so its meta["box"] must be its area of their image, a shape
  in pixels as int: ` followed by, for `"cardboard"`: `it holds
  'cardboard', not the dict form of a BoundingBox or a Quadrilateral.`; for
  `{"left": 1, "top": 2}`: the shape's own refusal `A shape's dict holds
  ['left', 'top']: a BoundingBox's keys are x, y, width and height, a
  Quadrilateral's top_left, top_right, bottom_right and bottom_left.`

### `test_boxes_read_back.py` - A Box is read back by the calls that read any event

**Purpose:** A Box is an event type like any other: what a timeline holds
comes back by the calls that read every event type, and a Box added comes
back equal to the Box added, through every form the timeline is written
in. No call or parameter exists for Boxes alone.

**The calls, established before any code.** From a timeline a user reaches
its events by `get_events(event_type=, include_children=, min_coord=,
max_coord=, **field_filters)` (an `EventData`; `events` is the timeline's
own), `get_event(id)` (one row as a dict) and `get_events_at(position)`
(rows as dicts, per timeline). An `EventData` gives rows (iteration,
`to_dataframe`) and typed scalars by `get_field(<scalar class>)`, by its
paired Field class, or by a column's name. Before this change none of these
gave back a `Box` (or a `Note`) object: `get_field(Box)` and
`get_field(Note)` both raised `No field matching ...`, and the only way from
a row to the object was the scalar's `from_row` (`Note.from_row(row)`, which
reads every position as quarters whatever the axis;
`Box.from_row(row, unit=, number_type=)`, which needs the axis the row does
not carry). A row of `get_events_at` holds bare numbers, which neither
`from_row` reads. So the one call that can read any event type back on its
axis is the field: `get_field(<event type>)` is now composed from the
columns (`tests/storage/README.md`, "An event type's field").

**What proves "the same calls"** (`TestTheSameCalls`, parametrised over
`Box`, `Projection`, `Note` and `ScoreNote`; each case adds one event on a
child at offset 866 of a `pixels` page, `seg_2` of the Thoresen example
(`meta["box"]` x 7), and one Beat `b1` to the page itself): every
assertion is written once and run for every type. The Note is the row
`{event_type: Note, id: n_378, start: 378, duration: 4, pitch:
{midi_number: 60}}`, the ScoreNote the same row with `staff` 1 and `voice`
2 (rows keep `event_type` `Note`; the ScoreNote is read on them because the
table states its `staff` and `voice`); each comes back as the Note built
from those numbers on `pixels`.

- `seg_2.get_events(event_type=T).get_field(T)` is `[added]`: the event
  added (its minted or given ID included) comes back equal, `start` 378.
- `seg_2.events.get_field(T)` gives the same list (the timeline's own
  events).
- On the page's `get_events()` (Note, then the child's row; the child's
  row shifted by its offset as every event of a child is) `get_field(T)` is
  `[None, event]` with `start` 1244 (866 + 378), every other property
  equal to the event added: the field is row-aligned with the rows the
  page reports.
- `page.get_events(id=<its id>).get_field(T)` (a field filter) holds the
  one event; `get_field(TField)` is the same object as `get_field(T)`.
  For a Note the child's row is shifted whole: `start` 1244, `end` 1248.

**Scores read back** (`TestScoresReadBack`). The expected notes are built
from the sources, never from the loader's output, and compared whole
(`==` on the scalars, so pitch, positions, `staff` and `voice` at once):

- *ms3, Chopin Op. 10 No. 3* (`vienna_1x22`, `ms3/chopin_op10_no3.notes.tsv`,
  498 rows). Each TSV row read with `csv`: `start` the `quarterbeats`
  fraction, `duration` the `duration` fraction (whole notes) times 4, `end`
  their sum, all `Fraction` on `quarters` (the four grace notes, duration
  0, rows 158, 170, 492, 493, are instants: no end, no duration); the
  pitch `cents` 0.0, as the score's pitch column writes it; the pitch
  `SpecificPitch(step, alter, octave)` from `name` (letter, then `#` +1 and
  `b` -1 each) and `octave`; `staff`, `voice` the columns. `Ms3Loader`'s
  timeline, `get_events(event_type="Note").get_field(ScoreNote)`: the 498
  notes in that order, none a rest; `get_field(Note)` the same notes as
  plain `Note`s. First note: B3 at 0, duration 1/2, staff 1, voice 1.
- *MEI through Verovio, Reger Op. 13 No. 4* (`mpm_toolbox`, 106 `<note>`s,
  25 of them tied, no grace note). From the XML: per note `(step, alter,
  octave, staff, layer)`, `step` the upper-cased `pname`, `alter` from
  `accid` or else `accid.ges` (`f` -1, `s` +1, `n` 0, absent 0), `octave`
  `oct`, `staff` and `layer` the `n` of the enclosing `<staff>` and
  `<layer>`. The read back field (`get_field(ScoreNote)` on the timeline's
  events) holds 106 notes and one null row (the tempo indication), its
  `(step, alter, octave, staff, voice)` multiset that of the XML, every
  position a `Fraction` on `quarters`; the first two notes as their `dur`
  states them: `n1` E♭5 at 0 lasting 1/4 (a sixteenth), `n4` A♭4 at 0
  lasting 3/4 (a dotted eighth), on staff 1, voices 1 and 2.

A reader assuming quarters, dropping the pitch, or reading a row of
another type as a note fails these.

**A field only where the table holds the type's data** (`TestTypesNotHeld`).
`get_field(<event type>)` composes a field only where every property the
scalar cannot do without (a required property) is a column of the table,
or, for a scalar that can do without every property (`Measure`,
`MidiEvent`), where every property is; an optional property whose column
is absent (no row states it: a Box's second shape, `projection_of`) is
null. Otherwise there is no field and the call raises the `KeyError` every
class without a matching column raises. A field of nulls would read present
data as absent, so these are the cases that would show it broken:

- *TiLiA, Bruckner 5, Scherzo* (`score`,
  `bruckner5_scherzo/harnoncourt/Bruckner5_Scherzo.json`), timeline
  `cpt_4` ("Takte"): its events hold 1,146 rows, every one typed `Beat`,
  and no `index` column (the beat's number is in `beat` and `measure`).
  `get_field(Beat)` raises `KeyError` starting `"No field matching
  'BeatField' found in table"`, `has_field(BeatField)` is `False`.
- *A Measure row*: a `quarters` timeline holding the row `{event_type:
  Measure, id: m1, start: 0, end: 4}`. `get_field(Measure)` raises
  `KeyError` starting `"No field matching 'MeasureField' found in
  table"`: `Measure` requires none of its properties, and `count`,
  `qstamp`, `number` and the others are no columns (`id` and `name` are,
  as on every event table, which proves nothing).
- *Side by side, one table*: a `pixels` timeline holding, in this order, a
  Beat (`b1` at 0), the Box `rect_h2`, the Projection `p1` at 378, an Event
  `e1` at 10 lasting 5, the Note `n_378` and a Measure `m1` at 30.
  `get_field(T)` for `Box`, `Projection`, `Event`, `Note` is a field of 6
  rows, the event added on its own row (index 1 to 4, equal to the event
  added) and `None` on every other; on the same table `get_field(Beat)`
  and `get_field(Measure)` raise the `KeyError` above. The Box holds a
  bounding box and no quadrilateral, the Projection no `projection_of`:
  optional properties, null where no row states them.
- *A defect is no absence*: where the field's own construction fails
  (`BoxField.from_field` made to raise `ValueError("broken")` for the
  test), `get_field(Box)` raises that `ValueError`, never `No field
  matching ...`.
- *Notes whose pitch the scalar cannot read*. A property the field's
  struct leaves out (a Note's pitch, represented once in a column of its
  own) needs a column the scalar reads it from, required or not; else
  every note would come back a rest. Two loaded tables state pitches only
  in such forms, and answer `get_field(Note)` and `get_field(ScoreNote)`
  as every `get_field(Note)` answered before an event type's field was
  composed: `KeyError` starting `"No field matching 'NoteField' found in
  table"` (`'ScoreNoteField'` for the ScoreNote), `has_field(NoteField)`
  `False`. *EEP*, `score`, `beethoven_op18-4iv_multimodal/
  StringQuartetEEP_I_Normal/StringQuartetEEP_I_Normal_align_vln1.notes`
  (`EepNotesLoader`): 1,266 Note rows, `pitch` a string column (the first
  row's value read from the file's first note line); *IEEE 1599*,
  `ieee1599/bach_artefuga_01.xml` (`Ieee1599Loader`), timeline
  `los:dlt_2`: 1,038 Note rows, no `pitch`, `specific_pitch` or `midi`
  column, the spelling in the raw `step`, `octave` and
  `actual_accidental` columns. What would show it broken: a field of
  1,266 or 1,038 notes, every one a rest.

**Round trips with exact values from real pages** (`TestRoundTrips`). The
expected Boxes are built from the sources' own numbers, never from the
loader's output: the two measure zones of the Anhalt chorale facsimile
(MEI, `ulx, uly, lrx, lry` = `490, 344, 973, 877` and `944, 368, 1523,
864`, one accolade from x 490: Boxes `BoundingBox.from_corners(...)` with
`start` 0 and 454, `duration` 483 and 579, the zones' `xml:id`s as IDs)
and the two measures of system 1 of the DMOS page (*Aimons-nous*, system
from x 1221: quadrilaterals `start` 0 and 930, `duration` 940 and 624,
minted IDs `box_1`, `box_2`), and a fraction case (the thirds Box on a
`user_units` timeline). For each, the accolade's
`get_events(event_type="Box").get_field(Box)` equals the expected Boxes,
each `start.value`, `duration.value` and shape number of the type the axis
declares (`int` on pixels, `Fraction` on fractions), and so do the Boxes
read back after (a) the timeline's dict form, `Timeline.from_dict(
page.to_dict(events=True))`, and (b) the package: an `AlignmentBundle`
holding the page written with `to_file` (`boxes.parquet` beside the
manifest) and read with `AlignmentBundle.from_file`. What would show it
broken: a Box read back unequal to the one added (an ID, a number or its
type changed on the way), or a form that loses the Boxes.

### `test_boxes_pages.py` - Cuts of pages and volumes

**Purpose:** A scanned page is a segment line of accolades: each accolade
stands for an area of the page's image (`meta["box"]`, `meta["image"]`) and
holds its Boxes at `start = x_min −` the `x` of that area. The page states
`meta["image"]` and no `meta["box"]`: its axis is the accolades laid end to
end, a concatenation of areas and no area itself. A volume is a segment
line of such pages and states no area either. The four cut operations of
the section above meet such a timeline in two ways.

**Reasoning.** A Box's shape never moves, so a Box may only stand on a
timeline whose x origin agrees with it (`start = x_min − origin`). On the
page's axis the measure `m3` of the second accolade starts at 1100 while
its shape starts at x 220: no timeline cut from the page has an origin both
accolades' Boxes agree with, at any cut start, 0 included. Hence:

- **A slice never moves a Box off its accolade.** It copies each timeline's
  own rows and slices each child in turn, so each accolade is cut by the
  rule of the section above (its slice states its own area) and the page's
  and the volume's slices carry `meta["image"]` and state no box. This is
  the cut a page affords.
- **A cut that lifts descendants' rows onto one new timeline is refused**
  wherever one of those rows is a Box: a child from a region and a segment
  of `create_segment_line` taking events along. The refusal is the one a
  timeline without area already gives (`... the timeline states no area
  (meta["box"]), so the cut's area cannot be stated.`), now at every cut
  start. What tells such a row from a timeline's own is `source_timeline`
  of the rows `get_events(min_coord=, max_coord=)` gives: null for an own
  row, the holding descendant's id otherwise. The tests assert that reading
  first, on the page and on the volume, since the rule rests on it.
- **A timeline's own Boxes are untouched by this.** A hand-made timeline
  holding Boxes itself without `meta["box"]` stands at x 0: its own Boxes
  move onto a cut at 0 and nowhere else, as in the section above. Where it
  also has an accolade child, only that child's Boxes are refused at 0.
- **Without a Box nothing changes.** A timeline whose tree holds no Box is
  cut as before, descendants' rows included, whatever `meta` says.

**What would show it broken:** a page's slice stating a box, or losing its
image; an accolade's slice whose area or Box starts differ from the
accolade cut alone; a child or segment of a page holding a Box (it could
not be written and read back) or refused with the Box check's sentence
instead of the cut's; a refused cut leaving a child behind; an own Box
refused at 0; a tree without Boxes refused or given `meta`. Every timeline
a cut returns round-trips (`from_dict(to_dict(events=True))`), which runs
the Box check on each timeline of its tree, and every Box's shape cell is
compared with the one its accolade held.

**Values** (pixels, int). Accolade `a1`: area `{x 100, y 200, width 1000,
height 300}`, length 1000, Boxes `m1` = `(150, 200, 300, 300)` at start 50
and `m2` = `(600, 200, 300, 300)` at 500, each of duration 300. Accolade
`a2`: area `{x 120, y 700, width 900, height 300}`, length 900, `m3` =
`(220, 700, 300, 300)` at 100 and `m4` = `(700, 700, 300, 300)` at 580.
`page` (`SegmentLine[DiscreteGraphicalTimeline]`, `meta == {"image":
"page_1.jpg"}`) holds `a1` at 0 and `a2` at 1000, length 1900. `vol`
(`SegmentLine` of such pages, `meta == {"image": "volume"}`) holds `p1` at
0 and `p2` at 1900 (ids prefixed `p1_`, `p2_`; images `page_1.jpg`,
`page_2.jpg`), length 3800.

- **Rows of descendants.** `page.get_events(min_coord=0, max_coord=1900)`:
  `(id, start, source_timeline)` = `(m1, 50, a1)`, `(m2, 500, a1)`, `(m3,
  1100, a2)`, `(m4, 1580, a2)`. `vol.get_events(min_coord=1000,
  max_coord=2000)`: `(p1_m3, 1100, p1_a2)`, `(p1_m4, 1580, p1_a2)`,
  `(p2_m1, 1950, p2_a1)` (the holder, two levels down).
  `page.get_events(include_children=False)` is empty.
- **Slice of the page, [500, 1500).** Length 1000, `meta == {"image":
  "page_1.jpg"}`; two segments at 0 and 500, each of length 500: the first
  with `meta["box"] == {x 600, y 200, width 500, height 300}` (100 + 500)
  holding `m2` at 0; the second `{x 120, y 700, width 500, height 300}`
  holding `m3` at 100; both carry the image. The whole page, [0, 1900):
  both accolades' `meta` as written, the four Boxes at their starts.
- **A Box crossing a slice boundary.** `page.get_slice(200, 1900)` cuts
  `a1` at 200, which `m1` (50 to 350) crosses: `Box 'm1' runs from 50 to
  350 on timeline 'a1' and crosses the slice's boundary at 200: a slice
  clips the intervals that cross its boundaries, and a clipped Box would
  contradict its shape. Slice with truncate_events=False to leave it out.`
  With `truncate_events=False`: length 1700, segments at 0 (length 800,
  area `x` 300, width 800, `m2` at 300) and 800 (`a2` whole: area as
  written, `m3` at 100, `m4` at 580).
- **Slice of the volume, [500, 2400).** Length 1900, `meta == {"image":
  "volume"}`; segments at 0 (length 1400, `{"image": "page_1.jpg"}`) and
  1400 (length 500, `{"image": "page_2.jpg"}`). The first holds the slice
  of `p1_a1` at 0 (length 500, area `x` 600, width 500, `p1_m2` at 0) and
  `p1_a2` whole at 500 (length 900, `p1_m3` at 100, `p1_m4` at 580); the
  second the slice of `p2_a1` at 0 (length 500, area `{x 100, y 200, width
  500, height 300}`, `p2_m1` at 50).
- **Child from a region, refused.** `page`, region [0, 1000): `Box 'm1'
  cannot move onto a timeline cut at 0 from timeline 'page': the timeline
  states no area (meta["box"]), so the cut's area cannot be stated.`;
  region [500, 1500): the same sentence with `'m2'` and `500`. `vol`,
  region [0, 1900): `'p1_m1'`, `0`, `'vol'`; region [1900, 3800):
  `'p2_m1'`, `1900`. The page keeps `a1`, `a2` alone, the volume `p1`,
  `p2`.
- **Segments, refused.** `page.create_segment_line([0, 1000, 1900])`:
  the sentence with `'m1'`, `0`, `'page'`;
  `vol.create_segment_line([0, 1900, 3800])`: `'p1_m1'`, `0`, `'vol'`.
  With `copy_events=False` no Box moves and nothing is refused: the page
  gives two segments of lengths 1000 and 900, each with `meta == {"image":
  "page_1.jpg"}` and no event. `SegmentLine.from_segmentation` never
  reaches a cut here: it refuses every timeline that has children (`Cannot
  segment a timeline that already has children. This would create
  conflicting nesting hierarchies.`), as before.
- **A parent that is no segment line.** `sheet`
  (`DiscreteGraphicalTimeline`, length 3000, `meta == {"image":
  "page_1.jpg"}`) holding `a1` at offset 1500. Its child from the region
  [0, 1000) takes no Box and is made: length 1000, `meta == {"image":
  "page_1.jpg"}`, no event. From [0, 2000) (`m1` at 1550): refused,
  `'m1'`, `0`, `'sheet'`; `sheet` keeps `a1` alone.
- **Own Boxes beside an accolade.** `sheet` also holding its own Box `own`
  = `(200, 50, 400, 100)` at 200 (no area: x 0). The child from [0, 1000):
  `own` at 200, shape unchanged, `meta == {"image": "page_1.jpg"}`. From
  [100, 1000): refused, `'own'`, `100`. From [0, 2000): refused for the
  accolade's `'m1'` at `0`, and no child is made. `get_slice(0, 3000)`:
  `own` at 200 and the slice of `a1` at 1500 with its area as written,
  `m1` at 50, `m2` at 500.
- **No Box anywhere.** `sheet` (same `meta`) holding the Note `e1`
  [200, 600) and the child `strip` (offset 1500, length 1000, `meta`
  `{"box": {x 100, y 200, width 1000, height 300}, "image":
  "page_1.jpg"}`) holding the Note `e2` [50, 350). The child from the
  region [100, 2000): length 1900, `meta == {}`, `e1` at 100 and `e2` at
  1450 (1500 + 50 − 100; the descendant's row moves). `get_slice(100,
  3000)`: `meta == {}`, `e1` at 100, the slice of `strip` at 1400 with
  `meta == {}` and `e2` at 50. `create_segment_line([0, 1000, 3000])`:
  segments of lengths 1000 and 2000, `meta` `{}` and `{}`, holding `e1` at
  200 and `e2` at 550.

### `test_external_references.py` - Incoming External References and `to_dict()` Flags

**Purpose:** Validates the per-timeline table of *incoming* external
references — annotations living outside TimeToAlign that point **at** this
timeline's events — and the serialization flags that decide whether events
and references appear in `to_dict()` output.

**Test Categories:** schema (12 tests), adding and normalization (12),
malformed-row rejection (12), event-id validation (10), the `to_dict()` flag
matrix (12), child forwarding (6), `from_dict` tolerance (5), and round-trip
(7) — 76 tests in total.

#### Canonical Arrow schema

`Timeline.external_references` always returns a `pa.Table`, never `None`.
An untouched timeline returns a zero-row table carrying exactly this schema:

| column | Arrow type | nullable |
|---|---|---|
| `event_id` | `string` | no |
| `external_id` | `string` | no |
| `access_points` | `list<struct<uri: string, kind: string>>` | no |
| `comment` | `string` | yes |

`event_id` names an event of *this* timeline; `external_id` is the identifier
the annotation carries inside the external resource; `access_points` holds the
locators that resolve that resource, with `kind` an open vocabulary
(`"relative_path"` and `"url"` are the usual values, but any string is
accepted). The access-point list may be empty, but the cell is never null.
Both struct fields are non-nullable. The tests assert the schema field by
field, including nullability, so a silent type drift fails.

#### `add_external_references()` normalization

Rows may be given as dicts or as a `pa.Table`; a table is read through
`to_pylist()` so both inputs follow one validation path. Per row:

- `access_points` omitted or `None` → stored as `[]`.
- `comment` omitted → stored as `None`.
- Unknown columns raise `ValueError` naming every unknown key. Unknown keys
  inside an access point raise `ValueError` the same way.
- `event_id` and `external_id` must be `str`; `comment` must be `str` or
  `None`; `uri` and `kind` must both be `str`. Anything else raises
  `ValueError` naming the row index and the offending field.
- A non-mapping row, or a non-mapping access point, raises `TypeError`.

Calls append: two calls of one row each and one call of two rows produce the
same two-row table. The method returns `self`, so calls chain.

#### Validation `KeyError` contract

With `validate=True` (the default) every `event_id` must appear in this
timeline's **own** event table (`Timeline.events`) — child events do not
count, because the column is defined as an event of this timeline. When one
or more ids are absent the call raises `KeyError` whose message names **all**
missing ids, sorted and quoted, in a single message of the form:

```
Timeline 'spine' has no event(s) with id: 'ghost_a', 'ghost_b'
```

Validation runs before anything is appended, so a rejected call leaves the
table byte-for-byte unchanged. `validate=False` skips the check entirely and
accepts ids that name no event — the escape hatch deserialization uses.

#### `to_dict()` flag matrix

`to_dict(*, minimal: bool = True, events: bool = False, external_references:
bool = False)` writes a manifest whose one timeline object carries the flags'
keys; the matrix is asserted on the complete form (`minimal=False`), whose
keys do not depend on the timeline's content. Both keys are **absent** unless requested; requesting `external_references` emits
the key even when the table is empty. The four combinations are asserted
exactly:

| `events` | `external_references` | `"events"` key | `"external_references"` key |
|---|---|---|---|
| `False` | `False` | absent | absent |
| `True` | `False` | present | absent |
| `False` | `True` | absent | present (may be `[]`) |
| `True` | `True` | present | present |

The remaining keys of the timeline object are identical in all four cases:
`id`, `name`, `unit`, `number_type`, `length`, `locked`, `is_segment_line`,
`meta`, `children`, `conversion_maps`, `regions` and `flow_control`. The
tests assert that exact key set for the complete form without flags.

Reference rows serialize as plain dicts with `access_points` a nested list of
`{"uri": ..., "kind": ...}` dicts, so the payload is JSON-safe; JSON safety of
the whole timeline payload is covered by `tests/core/test_number_json.py`.

#### Child flag forwarding

`children` recursion forwards **both** flags unchanged. A parent serialized
with `events=True, external_references=False` yields children that carry
`"events"` and lack `"external_references"`, and so on for every combination.
The tests check a two-level hierarchy (parent → child → grandchild) so
forwarding is verified through more than one recursion step. Each level emits
its own rows: a reference added to the child appears in the child payload
only. An `"events"` list also carries the `Segment` bookkeeping event that
`add_child()` records — the expected ids spell it out, since `from_dict()`
skips and regenerates it.

#### Round-trip

`from_dict()` tolerates absent `"events"` and `"external_references"` keys: a
default payload reconstructs a timeline with zero events and an empty
reference table, and the structural fields (id, length, children, offsets,
conversion maps) still round-trip. When `"external_references"` is present it
is restored with `validate=False`, so a payload carrying references but no
events round-trips intact — the restored table equals the original via
`pa.Table.equals`. Re-serializing with the same flags reproduces the input
dictionary (the fixpoint guarantee stated in `tests/core/README.md`).

The coordinate form for `length`, child `offset`s, and event coordinates is
untouched by these flags.

`to_typed()` carries the reference table over to the re-typed instance; Arrow
tables are immutable, so the same table object is shared.

**Validity Rationale:**

The direction is deliberately incoming: a timeline records what points at it,
never what it points to, so any number of independent external tools can
annotate the same events without the timeline knowing about them. Keeping the
table on the Timeline base class (an engine mixin, like events and regions)
means every timeline in a hierarchy carries its own references, addressed by
its own event ids. Slimming `to_dict()` keeps structural serialization cheap
for large hierarchies, where the event rows dominate the payload; callers that
need contents opt in.

---

### `test_nesting.py` - Child Timeline (Segment) Tests

**Purpose:** Validates hierarchical timeline nesting and traversal.

**Test Categories:**

1. **Child Validation Tests** (5 tests)
   - Matching units accepted
   - Mismatched units rejected
   - Non-Timeline objects rejected
   - Duplicate children rejected
   - A negative offset is refused as not supported yet, with the exact
     message `A negative child offset is not supported yet: child '<id>' has
     offset -1.0.` on a seconds parent, the offset spelled in its JSON form
     (a root's own offset may be negative; see "Manifests")

2. **Adding and Appending Children Tests** (13 tests)
   - Basic child addition
   - Offset storage and retrieval
   - Child locking upon embedding (a plain segment of a segment line
     excepted; see `test_unlocked_child.py`)
   - Segment event creation in parent's EventStore
   - Auto-expansion of parent
   - Multiple children
   - `append_child()` requires an unlocked parent, reports the exact standard
     lock error, preserves keyword-only identity arguments, and places a named
     child at the previous end while expanding the parent
   - `append_segment()` uses the same lock gate when its SegmentLine is already
     embedded as a child

3. **Get Child Tests** (4 tests)
   - Retrieve child by ID
   - Retrieve offset by ID
   - KeyError for nonexistent children

4. **Iteration Tests** (9 tests)
   - Empty timeline iteration
   - Sorted order (by offset)
   - Breadth-first traversal
   - Depth-first traversal
   - `include_self` parameter
   - `recursion_limit` parameter
   - Absolute offset calculation

5. **Unit Validation Tests** (4 tests)
   - Logical cannot contain physical
   - Physical cannot contain logical
   - Same domain, different units rejected

6. **Scale Tests** (2 tests)
   - Adding 1,000 children holds exactly 1,000.
   - Depth-first iteration of a chain 100 deep yields `depth_1` … `depth_99`
     in order, the timeline at depth *i* at root offset *i* (each sits at offset 1 on
     its parent).

7. **Cross-Unit Child Addition via Conversion Map** (11 tests)
   - `use_conversion_map=True` auto-selects parent's C-Map (e.g., SamplesToSeconds)
   - Converted child is named `{original name} [{parent_unit}]` (e.g., `notes [samples]`); its id is minted in the parent's unit (`dpt_1`), because a label is no identity
   - Event coordinates correctly converted (seconds → samples via inverse C-Map)
   - Original child is NOT modified or locked (only the derived copy is locked)
   - Lookup by string target unit name (`use_conversion_map="seconds"`)
   - Direct `ConversionMap` object accepted
   - Without `use_conversion_map`, unit mismatch still raises `ValueError`
   - `use_conversion_map=True` raises when no matching C-Map attached
   - Same-unit child passes through unchanged (no `[unit]` suffix)
   - `allow_expansion=True` works with unit conversion
   - Converted copy is locked, original is not

8. **Segment Duration Exactness Tests** (2 tests)
   - `add_child` records the segment event's `duration` from the child's
     `length`. When both the segment's `start` and `end` carry an exact
     ratio (`numerator`/`denominator` populated, i.e. an int or `Fraction`
     coordinate), `duration` is recomputed via exact `Fraction` arithmetic
     rather than a float `end - start` subtraction, so an integral length
     (`Fraction(2, 1)`) and a non-integral one (`Fraction(7, 3)`) both
     round-trip through `parent.to_dict()["events"][0]["duration"]` as the
     exact Fraction pair (`number_to_json`) for the child's length — not a
     rounded float projection.

9. **Exact child shift** (2 tests)
   - On a quarters/`fraction` parent, a child at offset `0.1` (held as its
     exact dyadic `3602879701896397/2⁵⁵`) receives the parent positions
     `0.7` and `1/3` as `21617278211378379/2⁵⁵` and
     `25220157913274777/(3·2⁵⁵)` in `get_timestamp_table(at=..., format="table")`
     (`0.7 − 0.1` cross-multiplies past `2⁶³`); the expected ratios are
     Python `Fraction` arithmetic, written out.
   - The other direction: a child of length `0.7` at offset `0.1` puts its
     end on the root axis at exactly `28823037615171173/2⁵⁵`
     (`0.7999999999999999`), the default table's last row.

9. **The holder of an event, at every depth** (`TestEventHolder`)
   - `get_events(include_children=True)` tags every row a child reports with
     the timeline holding the event (`source_timeline`): the child for its
     own events, a deeper descendant's name kept; only the timeline's own
     rows stay untagged. A child that has a descendant holding events
     reports its own rows beside the descendant's tagged ones, so a rule
     that tags only rows lacking the key leaves them untagged and the
     parent reads them as its own (what the MatchClaim factory relies on,
     `tests/alignment/README.md`, `TestCreateMatchClaims`). Evidence on
     `score` (quarters, 8) ⊃ `c` (offset 2, length 4) ⊃ `g` (offset 1 in
     `c`, length 2), `score` holding `s1` at 0, `c` `n1` at 1, `g` `x9` at
     1: `score.get_events(include_children=True)` gives exactly
     `[("s1", None, 0), ("n1", "c", 3), ("x9", "g", 4)]` as `(id,
     source_timeline, start)`, `c.get_events(include_children=True)` gives
     `[("n1", None, 1), ("x9", "g", 2)]`; `score.get_event` names `None`
     for `s1`, `"c"` for `n1`, `"g"` for `x9`.

10. **Child Construction and Boundary Tests**
   - `create_child()` defaults its offset to zero and returns the parent's
     exact concrete class for all six concrete timeline types
   - Coordinate objects remain valid for child lengths and offsets
   - `create_children_from_boundaries()` creates named, contiguous children
     that tile the parent exactly, supports explicit names, and reports the
     same boundary-validation errors as `create_regions_from_boundaries()`

11. **A child's slice of its parent's measure map** (`TestChildMeasureMap`)
   - `create_child(..., measures=True)`, on a quarters and on a ticks parent,
     with the refusals and the manifest round trip; the strategy is in
     "Measure map slices" above

**Validity Rationale:**

The TTA model specifies that timelines can contain nested "children" (segments)
that share the same coordinate type. These tests verify:
- Children must have matching units (type safety)
- Children are locked upon embedding (immutability), except a plain segment,
  whose segment line's lock governs it
- Children appear as interval events in the parent's EventStore
- Traversal orders work correctly for hierarchical access
- Cross-unit nesting works via C-Map inversion (e.g., EEP notes in seconds
  added as child of audio DPT in samples, using `SamplesToSeconds` inverse)

---

### `test_calibrated_child.py` - Children Calibrated by Position/Value Pairs

**Purpose:** `create_calibrated_child()` turns a calibrated segment of a
timeline — calibration pairs of a parent position and a value in another
unit, read under a law — into a child at the segment's start carrying the
law's conversion map. A drawn axis with labelled ticks is the motivating
case.

**Validation logic:**

- **A drawn seconds axis with five labelled ticks.** The ticks (0, 10, 20,
  30, 40 s at 76.70371459980326 … 932.2326510101616 pt) are calibrated on a
  float `points` parent. The child's offset and length are the ones given;
  the map reads each position from the offset in float arithmetic, so the
  first tick reads 0.0 and the last 855.5289364103584. The map's scalar is
  then 40.0 / 855.5289364103584 = 0.04675470144567245 and its offset 0.0,
  computed independently of the library from the two-point formula. The map
  is probed at an inner tick (213.88223410258962 → 10.000000000000002, the
  float the line gives, one ulp above the label) and at the end (→ 40.0).
  These are the values of plain float arithmetic; a library that rounded,
  fitted or guessed a ratio would give other bits and fail.
- **The inner ticks are checked on the parent's positions.** The check runs
  on the positions as given, before the offset is subtracted, where their
  rounding happened (see `tests/maps/README.md`): against exact labels the
  inner ticks lie 0.9375, 0.375 and 0.15625 of their ulps off the exact line
  through the end ticks and pass, building the same map. The fourth tick
  moved up by four floats (718.3504169075725, 4.16 of its ulps off, beyond
  the 3.40 ulps a computed position may miss there) is refused and the
  parent keeps no child. Refusals name positions as given,
  in the parent's coordinates.
- **Exact axes stay exact.** On a `fraction` parent, `Fraction` positions and
  `quarters` values give a `Fraction` scalar and offset and an exact
  conversion.
- **A refusal changes nothing.** Moving the label "10" to 300.0 takes it off
  the line through the end ticks, which reaches 10 at 290.58594870239284:
  the call raises, and the parent then has
  no child, no new segment event and the same manifest ids as before, so the
  next child it creates gets the id it would have had anyway.
- **Order is irrelevant.** Unsorted calibration gives a map equal to the
  sorted one.
- **The scalar survives serialization bit for bit.** `to_dict()` → JSON text
  → `from_dict()` of the child restores the scalar with the same
  `float.hex()` and the offset 0.0. The map's declared `float` output is a
  library setting the manifest does not carry; the map read back writes
  floats because seconds do by default.
- **Every refusal names the offending pair.** Fewer than two pairs, a
  repeated position, values in two units, an inner pair off the line, a
  position before the offset or after its end — each raises `ValueError`
  with a message naming the pair; a number given as a value without
  `value_unit` raises `TypeError` naming that keyword, since without a unit
  there is no target axis.
- **Values come as coordinates or as numbers in `value_unit`.** Numbers with
  `value_unit` build the same map as the equal coordinates without it, and a
  `Coordinate` value in another unit than `value_unit` is refused.
- **Every timeline class has it.** One test runs over the six leaf classes
  (continuous and discrete, in the three domains) with their default units
  and number types: integer positions and lengths on the discrete parents,
  floats on the float parents, `Fraction`s on the `quarters` parent. The
  pairs (10 → 0, 50 → 2, 90 → 4, child from 10 to 90) read in `quarters`, or
  in `seconds` for the logical parents, so the scalar is 4/80: the exact
  `Fraction(1, 20)` where positions and values are both exact, the float
  0.05 where either is a float. The child is the parent's class, in its unit
  and number type, and the map reads the inner pair's value exactly.
- **Identity.** The child gets the given id and name.

---

### `test_types.py` - Domain-Specific Timeline Subclasses

**Purpose:** Validates the 6 timeline types (3 domains x 2 modalities), the
dynamic `SegmentLine[T]` class family, and timeline serialization dispatch.

#### An empty timeline still has columns

`to_dataframe()` on a timeline with no events returns an empty frame **with
the schema's columns and dtypes**, not a structureless `DataFrame()`. The
latter stranded every caller that reads `df.columns` — which is what made
`export_to_csv(format="default")` unusable on any event-less timeline.

Relatedly, only conversion maps with a numeric `target_unit` become timestamp
columns. A structured map such as `MetricalPositionMap` answers
`{"mc": ..., "beat": ...}` and has no column type; it used to be forced into
the float64 schema, where the cast failed on the field name `'beat'` itself.
This matches the rule the matchstamp table already applies, so the two table
builders cannot disagree about which maps are columnar.

**Timeline Types:**

| Type | Domain | Modality | Default Unit | Default Number Type |
|------|--------|----------|--------------|---------------------|
| `ContinuousLogicalTimeline` | Logical | Continuous | quarters | Fraction |
| `DiscreteLogicalTimeline` | Logical | Discrete | ticks | int |
| `ContinuousPhysicalTimeline` | Physical | Continuous | seconds | float |
| `DiscretePhysicalTimeline` | Physical | Discrete | samples | int |
| `ContinuousGraphicalTimeline` | Graphical | Continuous | centimeters | float |
| `DiscreteGraphicalTimeline` | Graphical | Discrete | pixels | int |

**Test Categories:**

1. **Parametrized Type Tests** (24 tests, 4 per type)
   - Default unit is valid
   - Default number_type is valid
   - Domain derived correctly
   - Can hold events

2. **Logical Timeline Tests** (8 tests)
   - Allowed units (beats, quarters, measures, ticks, number)
   - Continuous accepts float/Fraction, rejects int
   - Discrete accepts int only, rejects float
   - Continuous rejects ticks, Discrete rejects quarters

3. **Physical Timeline Tests** (8 tests)
   - Allowed units (seconds, milliseconds, minutes, samples, frames)
   - Continuous accepts float/Fraction, rejects int
   - Discrete accepts int only
   - Unit restrictions enforced

4. **Graphical Timeline Tests** (8 tests)
   - Allowed units (pixels, meters, cm, mm, inches, points, css_pixels,
     user_units)
   - Similar pattern to other domains
   - Every legal leaf unit round-trips with each number type it admits;
     `css_pixels` and `user_units` with `float` and `fraction`, as `points`

5. **Unit sets come from the unit.** Every `TimeUnit` is allowed by
   exactly one of the six leaf classes, and that class's domain and
   modality are the unit's own `domain` and `is_discrete`; the domain
   classes allow exactly the units of their domain. A unit added to
   `TimeUnit` is therefore placed without a second list to update, and a
   leaf class listing a unit by hand in the wrong place fails. In
   particular `Timeline.resolve_subclass("css_pixels")` and
   `Timeline.resolve_subclass("user_units")` are `ContinuousGraphicalTimeline`,
   and a `Timeline` in either unit is minted `cgt_1`.

6. **Domain Property Tests** (3 tests)
   - Each domain class returns correct Domain enum

7. **Inheritance Tests** (6 tests)
   - All concrete types inherit from correct base classes

8. **Cross-Domain Compatibility Tests** (3 tests)
   - Logical cannot contain physical
   - Physical cannot contain graphical
   - Graphical cannot contain logical

9. **Parameterized SegmentLine and Serialization Tests**
   - Repeated parameterization returns the identical cached class
   - Dynamic classes inherit the APIs and restrictions of both `SegmentLine`
     and their timeline parameter
   - Invalid and repeated parameterization raises `TypeError`
   - Pickle reconstructs dynamic classes by module reference (the dynamic
     class names are a pickle concern only; the dictionary never writes them)
   - `to_typed()` recursively re-types children and retains exact classes

   **The dictionary form of a SegmentLine** (`test_segment_line_dict.py`).
   Every timeline object at every depth carries the boolean
   `is_segment_line` in the complete form, false on a plain timeline,
   exactly like `locked`; a timeline object writes no class, because the
   unit decides it. A SegmentLine writes `is_segment_line: true`; its
   children are its segments. What proves it:
   - **Exact dictionaries.** A `SegmentLine[DiscreteGraphicalTimeline]` of
     two segments (lengths 300 and 200 px, so the line is 500 px) is pinned
     key for key: flag true, both children with flag false at offsets 0 and
     300, no `class` anywhere. A nested, IEEE-shaped line — a page line of
     two accolade lines (accolades 100+150 px and 120 px) — is pinned too:
     the outer and both page lines carry true, the accolades false. A plain
     timeline writes false. `locked` is pinned on every object: a plain
     segment (both flat segments, every accolade) writes `false`, because a
     segment is never locked; a segment that is itself a segment line (both
     pages) writes `true`; the root lines write `false`.
   - **Round trips that rebuild the type.** `from_dict` returns
     `SegmentLine[DiscreteGraphicalTimeline]` for the flat line and
     `SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]` for the nested
     one (the segment type is the children's class, a SegmentLine itself
     when the children carry the flag, and the class of the unit for a line
     without segments), and writing the result again gives the same
     dictionary. Children given out of offset order are rebuilt in offset
     order, so a reordered file reads as the same line.
   - **Refusals.** `from_dict` raises `ValueError` for a flagged timeline
     whose children disagree on the flag (mixed classes), whose segments
     leave a gap or overlap, whose segments end short of its length, whose
     last segment ends beyond it (refused as any child ending beyond its
     parent, before the segments are placed), and for a non-boolean flag.
   - **Every line can be written.** A SegmentLine with no segments writes
     its unit and the flag, and rebuilds as `SegmentLine[X]` for X the
     class of its unit; a bare, never-typed `SegmentLine()` on pixels
     therefore reads back as `SegmentLine[DiscreteGraphicalTimeline]`. The
     test pins that type, since an identity carried by no key cannot be
     restored.

**Validity Rationale:**

TimeToAlign defines 6 timeline types across 3 domains (Logical, Physical,
Graphical) and 2 modalities (Continuous, Discrete). Each type has:
- Restricted allowed units (domain-specific)
- Appropriate default unit and number_type
- Consistent behavior with base Timeline

---

### `test_beatgrid.py` - Beat Grids

**Purpose:** Validates `BeatGridSegment`, `GridBeat` and `BeatGrid` — the
tempo-segment model that generates a labelled lattice of beats and answers
measure/beat questions on a seconds axis.

**What the object is.** A `BeatGridSegment` states only source facts: the
anchor beat's second (`start`), the tempo (`bpm`), how lattice beats group
into bars (`policy`), and which beat of a bar the anchor is (`battito`). A
`BeatGrid` sorts the segments, bounds each by its successor's `start` and the
last by the grid's `extent`, and generates beats forever from each one. Beat
and bar lengths are derived, never stored beside the facts, so no test may
assert a stored duration — it asserts the derivation.

A `GridBeat` **stores** its position as `seconds`, the exact `Fraction` the
grid's arithmetic produced, and **publishes** it as `instant`, a derived
seconds `Coordinate`. One stored field, one derived view: exactness is what the
grid computes with — the loader's fm anchors and the tempo integration read
`seconds` — while `instant` is what a caller sees, built exactly as
`seconds_at` builds its answer, on the float lane the seconds axis declares.

That is the point of the split, and the tests pin it as such: **the two
directions of the grid must compare equal**, so
`grid.position_at(x).instant == grid.seconds_at(m, b)` holds whenever both name
the same moment. A `fraction`-typed instant against a `float`-typed
`seconds_at` never compared equal, which made the obvious cross-getter
assertion fail on values that were in fact identical. The equality is asserted
at 30.9375 s — a value every lane represents exactly, so it pins the equality
rather than a shared rounding — and again at 7.01 s, where no exact float
exists but both directions convert the same ratio and therefore still land
together. A negative case (`position_at(30).instant != seconds_at(16)`) keeps
the assertion from passing vacuously.

#### The boundary rule is the whole reason the grid exists

The rule under test: **a beat of segment *k* is dropped when it falls within
HALF of segment *k*'s beat duration before segment *k+1*'s start; at exactly
half a beat it is kept.** The rule applies at segment boundaries only, never
at the end of the grid, where the extent alone bounds the last segment
(`instant < extent`): the phantom grid's last beat, 29.81 s, is kept 19/100 s
before its extent 30, within half a 120 BPM beat (1/4 s).

**Why a rule is needed at all.** A DJ-software export rounds each grid's
anchor to three decimals, and when the analyst nudges a later grid's anchor
to the right it freezes the beats to the left of it without inserting a
fill-in. The beat the earlier segment would generate one or two hundredths of
a second before the next anchor is therefore *the same beat*, displaced —
counting both produces a phantom bar and shifts every measure number after
it. Exact-equality precedence cannot see this: `10.5 != 10.51`.

**What proves it.** Three cases, each hand-derived, because a rule stated as
an inequality fails at exactly one place — its boundary:

| Case | Gap to the next segment | Half a beat | Expected |
|---|---|---|---|
| below | 0.01 s | 0.25 s | dropped |
| exactly at | 0.5 s | 0.5 s | **kept** |
| above | 0.30 s | 0.25 s | kept |

The knife edge is what a `<=`/`<` slip breaks and nothing else catches, so it
is asserted directly: a 60 BPM segment (beat 1 s, half-beat 0.5 s) whose next
segment opens at 3.5 s generates a beat at 3.0 s, exactly half a beat before
the boundary, and that beat is kept. A test suite that only asserted the
"obviously phantom" 0.01 s case would pass with either comparison.

The rule is a **heuristic** and the tests say so: what the source program does
for anchor displacements beyond half a beat has not been studied, so no test
asserts behaviour there.

#### Numbering: two readings of one lattice

Every generated beat carries its in-measure index
`((battito - 1 + step) mod n_beats) + 1`; index 1 is a downbeat. Measures are
counted by downbeats, in two schemes that must be produced by the same walk:

- **set** — one counter for the whole grid; the beats before the grid's first
  downbeat are measure `0`, the first downbeat opens measure `1`. This is the
  numbering the floating-measure lattice reads, where a pickup is bar 0.
- **segment** — the same rule restarted at each segment, so a segment whose
  anchor is mid-bar opens with local measure `0`, and its first downbeat is
  its local measure 1.

The two must be able to disagree, or the test proves nothing: a grid whose
second segment anchors on beat 3 has an anchor beat with set measure 2 (it
continues the measure the first segment opened) and segment measure 0. That
one beat is asserted in both readings.

`n_measures` counts distinct measure labels, so a grid opening off the
downbeat has one more measure than it has downbeats.

#### The four conversion directions

Each direction is asserted with values derivable from the tempo by hand,
because a wrong linear placement still returns plausible monotonic numbers and
passes every internal-consistency check.

1. `seconds_at(measure, beat)` — the instant of a labelled beat. A fractional
   beat interpolates between the two beats around it: at 120 BPM in 4/4, beat
   2.5 of measure 1 is `0.5 + 0.5 * 0.5 = 0.75` s.
2. `segment_seconds_at(...)` — the same instant minus the containing
   segment's start, so the two answers must differ by exactly that start.
3. `position_at(seconds)` — floor semantics: the last beat at or before the
   position, carrying both numberings. Asserted at a position strictly
   between two beats, so a rounding implementation lands on the wrong one.
4. `get_beat_table()` — the tabular rendering of the same walk. Its row count
   must equal the number of generated beats and its values must match the
   scalar getters, since one is defined as a rendering of the other.

**Policy overrides.** A caller policy re-reads the beat index as a quarter-note
offset from the downbeat and converts it back through the segments' tempi.
`BeatPolicy.uniform(Fraction(1, 2), 8)` on a 4/4 segment counts eighths, so its
beat 3 is quarter offset `2 x 1/2 = 1`; at 120 BPM in 4/4 (2 quarters per
second) that is 0.5 s after the downbeat. The inverse reading is pinned beside
it, at an instant where the two countings disagree: second 1.0 is quarter
offset 2, which the eighth policy calls beat 5 and the grid's own lattice calls
beat 3.

#### `quarters_between` integrates, it does not average

Tempo changes make an average wrong. The test spans a segment boundary: 60 BPM
in 4/4 from 4 s to 6.01 s is `2.01` quarters and 120 BPM from 6.01 s to 7.01 s
is `2.0`, so the interval `[4, 7.01)` is exactly `401/100` quarters — a value
no single tempo produces. All decimal inputs are read as exact decimal
rationals; a float input is documented to contribute its exact binary value,
which is why the tests spell decimals as strings or `Fraction`s.

#### One address, one beat

`position_at` and `get_beat_table` read *instants*, which are unambiguous: a
second belongs to exactly one beat. The reverse direction is not symmetric. A
measure stays open until the next downbeat, and a segment that re-anchors
mid-bar restarts its own beat count immediately — so one measure can state one
beat index twice, and `(measure, beat)` then names no single instant.

**The fixture that produces it.** Segments `(0 s, 120 bpm, 4/4, battito 1)` and
`(30 s, 128 bpm, 4/4, battito 3)` over 32 seconds. At 120 BPM a beat is 0.5 s,
so measure 15 opens at `56 × 0.5 = 28.0` and its four beats fall at 28.0, 28.5,
29.0, 29.5; the beat at 30.0 would be the next one, but the extent of segment 1
stops it (and the half-beat rule leaves 29.5 alone — the gap is a full 0.5 s
against a 0.25 s half-beat). Segment 2 anchors on **beat 3** at 30.0 with a
`60/128 = 0.46875` s beat, so 30.0 is beat 3 again and 30.46875 is beat 4 again;
its first downbeat is `30.0 + 2 × 0.46875 = 30.9375`, which opens measure 16.
Measure 15 therefore holds six beats — 1, 2, 3, 4, 3, 4.

**The rule.** `seconds_at` and `segment_seconds_at` raise `ValueError` naming
the candidate instants when the queried index is stated more than once —
`seconds_at(15, 3)` lists 29 and 30 — because picking either one would answer a
question the caller did not ask. A fractional beat interpolates between the
*unique* beats ⌊b⌋ and ⌊b⌋+1 whichever segments they come from, and raises the
same way when either index is missing or duplicated. Unaffected and asserted
alongside: `seconds_at(15, 2)` still answers 28.5, and `get_beat_table()` lists
all six rows of measure 15, because neither reading is ambiguous.

#### Segments abut, and a stated end must say so

A grid bounds each segment by the next one's `start` and the last by `extent`.
A segment arriving with `end=None` is filled in; a segment arriving with an
`end` of its own must state exactly that bound. Silently rebounding a stated
end would discard a caller's declared fact, and honouring a different one would
leave a gap or an overlap the lattice cannot represent — so a disagreement
raises. Pinned in both directions: a segment stating the correct bound survives
`__init__` unchanged, and one stating any other (including an `end` on the last
segment of an unbounded grid) raises.

#### `quarters_between` has a domain, and it is not the whole line

The seconds direction (`_seconds_after`, reached through a caller policy)
already raised past the extent. The quarters direction silently extrapolated,
so one direction answered where the other refused. Both now share the domain
**`[0, extent]`**, closed at both ends — `[0, ∞)` when the grid is unbounded.

Closed at the extent is deliberate: no beat sounds there, but the grid does
state a tempo up to it, and the final measure's length is exactly
`quarters_between(last_downbeat, extent)`. Open at zero is equally deliberate:
a grid whose first segment opens at 0.5 s still reads `[0, 0.5)` at that
segment's tempo, because the floating-measure lattice hangs its `t = 0` anchor
on precisely that stretch. Outside the domain the grid states no tempo, and
both directions raise.

#### Atomic raises

Per the retrieval contract, an unresolvable **query** raises rather than
answering approximately. Pinned: a position before the grid's first segment; a
position at or after the extent (the half-open convention, so the extent
itself raises); a `(measure, beat)` pair the grid does not state; a fractional
beat whose upper neighbour is missing; a `(measure, beat)` the grid states
twice; a `quarters_between` end outside `[0, extent]`; a coordinate carrying a
unit other than seconds; and tabulating or counting measures of an unbounded
grid, which generates beats without end.

#### Serialization

The grid has none this iteration, and the tests assert none. A grid is
reconstructed from its segments, which are the source facts; a second dictionary
format for them would be a second source of truth.

---

### `test_maps_integration.py` - Conversion Maps on Timelines

**Source unit (`TestMapIntegration`):** a map is a function from a position
on its timeline's axis, so `add_conversion_map` refuses a map whose
non-null source unit differs from the timeline's unit, with the exact
message `Map source unit 'ticks' does not match timeline unit 'seconds'`,
and leaves the timeline without the map; a map whose source unit is null
(`ScalarMap(scalar=2)`) attaches to a timeline of any unit. This one check
is the attach path of every map, the readers' included.

**TableMap honesty (`TestTableMapHonesty`):** `add_conversion_map` used to
wrap every `TableMap` in an `InterpolationMap`, silently converting its
interpolation `kind` to linear and every `extrapolate` policy to linear
extrapolation — a timestamp could contradict the very map attached to the
timeline. Maps are now stored directly, so the tests pin the honest
behaviour end-to-end: `kind='previous'` step values survive to
`get_unit()`; `extrapolate='error'` raises for out-of-bounds coordinates;
`extrapolate='constant'` clips to the exact boundary value. All assertions
are exact — no `pytest.approx`.

**Timeline + MetricMap round trip
(`TestTimelineSerializationWithMeterMap`):** `ConversionMap.from_dict`
previously dispatched through a hand-written table that omitted the meter
maps, so `Timeline.from_dict()` raised `ValueError: Unknown map type:
MetricMap` for any timeline carrying one. The registry is now populated by
`__init_subclass__`; the test attaches a `MetricMap`, round-trips through
`Timeline.to_dict`/`from_dict`, and asserts `meter(4.0) == 2` exactly.

**One column rule in both tables
(`test_both_tables_refuse_a_map_without_output_number_type`):** a
timeline's and a group's timestamp table derive a conversion map's column
by one rule, so they cannot disagree. A public `LinearMap` subclass naming
its target unit (`seconds`) through the `target_unit` property alone
declares no output number type; on a quarters timeline, both
`get_timestamp_table([0, 1])` and the group's
`get_timestamp_table([0, 1], "clt_1")` raise the same `ValueError`,
`Conversion map 'lmap_1' names the target unit 'seconds' but no output
number type`. One table leaving the column out while the other refuses,
or either failing with an assertion, fails the test.

---

### `test_integration.py` - Loader Integration Tests

**Purpose:** Validates Timeline integration with EventStore and loaders.

**Test Categories:**

1. **EventStore Integration** (3 tests)
   - Timeline from empty EventStore
   - Timeline from populated EventStore
   - Unit/number_type preservation

2. **MIDI Loader Integration** (4 tests)
   - Performance MIDI to Timeline
   - Score MIDI to Timeline
   - Nested MIDI timelines
   - A discrete timeline from MIDI-shaped event data

3. **Score Loader Integration** (2 tests)
   - Note events from score
   - Combined score data

4. **Complex Hierarchy Tests** (2 tests)
   - Multi-level nesting with events at each level
   - Serialization roundtrip of complex hierarchies

5. **Scale Integration** (1 test)
   - 50,000 interval events become a timeline holding exactly 50,000. Real
     MIDI loading is pinned with exact note counts in
     `tests/integration/test_loader_to_timeline.py`.

6. **Edge Cases** (4 tests)
   - Empty event list
   - Instant-only events
   - Interval-only events
   - Zero-duration intervals

**Validity Rationale:**

The Timeline class must integrate seamlessly with the loader infrastructure:
- Events from EventStore can populate Timelines
- Multiple sources can be combined hierarchically
- Real-world test data produces valid timelines

---

## Running Tests

```bash
# Run all timeline tests
pytest tests/timelines/ -v

# Run with coverage
pytest tests/timelines/ --cov=timetoalign.timelines --cov-report=term-missing

# Run specific test file
pytest tests/timelines/test_base.py -v

# Run performance tests only
pytest tests/timelines/ -v -k "performance"
```

---

### `test_timeline_relationships.py` - Unified Verb×Noun API, Regions, SegmentLine, derive()

**Purpose:** Validates the unified verb×noun Timeline API (Phases A-D) plus TTA
architecture harmonization features that distinguish between different timeline
relationship concepts.

**~170 tests**, including 13 real-data tests using Wagner Walküre Act III
measures.

**TimeToAlign Concepts Tested:**

| Concept | Definition | Test Category |
|---------|------------|---------------|
| **Region** | A named TimeInterval (NOT a timeline) | `TestRegionDataclass`, `TestTimelineRegionManagement` |
| **Child** | A timeline nested in a parent (same unit) | `TestCreateChildFromRegion`, `TestCreateChildrenFromRegions` |
| **Segment** | A Child that is contiguous with siblings | `TestSegmentLineBasics` |
| **SegmentLine** | A parent where ALL children are Segments | `TestCreateSegmentLine`, `TestCreateSegmentLineFromRegions` |
| **Derivative** | A new timeline created via C-map (different unit) | `TestTimelineDerive` |

**Unified Verb×Noun API Methods (new):**

| Verb | Region | Child | SegmentLine |
|------|--------|-------|-------------|
| `create_` | `create_region()`, `create_regions_from_boundaries()`, `create_regions_by_grouping()`, `create_regions_by_splitting()` | `create_child_from_region()`, `create_children_from_regions()` | `create_segment_line()`, `create_segment_line_from_regions()`, `create_segment_line_by_grouping()`, `create_segment_line_by_splitting()` |
| `get_` | `get_region()`, `get_regions_at()` | `get_children_at()` | — |
| `has_` | `has_region()` | `has_child()` | `has_segment()` |
| `list_` | `list_regions()` | `list_children()` | `list_segments()` |
| `add_` | `add_region()` (overloaded) | — | — |

**Removed Methods:** `partition()`, `region_to_child()`, `get_region_object()`.

**Changed Methods:** `get_region()` now returns `Region` object (was dict), raises
`KeyError` if not found (was `None`). `__contains__` now checks regions AND children.

**Test Categories:**

1. **Region Dataclass Tests** (9 tests)
   - Creation with name, start, end, meta
   - Duration computation (end - start)
   - as_interval property (tuple)
   - contains() follows [start, end) convention (left-inclusive, right-exclusive)
   - overlaps() detection
   - Rejects end < start
   - Rejects mismatched units
   - Immutability: mutating a field raises the concrete
     `dataclasses.FrozenInstanceError` (asserted by class, not a bare
     `Exception`), since `Region` is a frozen dataclass

2. **Timeline Region Management Tests** (12 tests)
   - add_region() returns Region object
   - get_region() returns Region object (updated from dict)
   - get_region() raises KeyError if not found (updated from None)
   - has_region(), iter_regions(), list_regions()
   - n_regions property
   - Duplicate name rejection
   - **Region creation succeeds on a locked timeline.** Length-locking guards
     a timeline's own coordinate contract with its parent (`set length`), not
     the addition of named annotations within the existing coordinate space.
     Regions are pure annotations that change nothing structural — they are
     strictly less structural than children, and child creation is already
     unguarded on a locked timeline. So `add_region`, `create_region`,
     `create_regions_from_boundaries`, and `create_regions_by_splitting` all
     succeed on a `locked=True` timeline with exact bounds/names: `add_region`
     and `create_region` produce the requested `[start, end)` region;
     `create_regions_from_boundaries([0, 40, 100], prefix="sec")` yields
     `sec_1 [0, 40)` and `sec_2 [40, 100)`; and
     `create_regions_by_splitting("breaks", prefix="mov")` on an event-free
     timeline yields the single whole-span region `mov_1 [0, 100)`. A child
     from such a region is made on a locked timeline too, as long as it ends
     within the length (see the create_child_from_region group below).

3. **Region creation — create_region() / add_region()** (5 tests)
   - `create_region(start, end, id=…, name=…)` creates and registers a Region
   - `add_region(Region)` attaches an existing Region object
   - a given region id already held raises; a repeated name does not

4. **Region creation — create_regions_from_boundaries()** (5 tests)
   - Creates N-1 regions from N boundary coordinates
   - Names follow configurable format pattern
   - Validates sorted, non-negative boundaries

5. **Region creation — create_regions_by_grouping()** (6 tests)
   - Groups adjacent events by field value
   - Recurring group values (e.g., 4/4 → 3/4 → 4/4) repeat the name;
     minted ids (`region_1`, `region_2`, `region_3`) tell the runs apart
   - Supports custom `name_format` with `{value}` and `{run}` placeholders
   - Raises ValueError for missing field

6. **Region creation — create_regions_by_splitting()** (5 tests)
   - Splits at events matching a predicate (field name or dict filter)
   - Supports `include_before_first` and `include_after_last`
   - Deduplicates split points at timeline boundaries

7. **Children and regions — create_child_from_region()** (7 tests)
   - Creates child at region's offset
   - Copies events within region to child
   - Adjusts event coordinates relative to child origin
   - copy_events=False creates empty child
   - Raises KeyError for nonexistent region
   - On a locked timeline of 100 s, a region `[0, 10)` gives a child of
     length 10; the timeline keeps 100 and its lock (the full account in
     `test_locked_region_child.py`)

8. **Children and regions — create_children_from_regions()** (3 tests)
   - Creates children for all or specified regions
   - Returns list of created children

9. **Children and regions — get_regions_at() / get_children_at()** (6 tests)
   - `get_regions_at(coord)` returns all regions containing that coordinate
   - `get_children_at(coord)` returns all children whose span includes that coordinate

10. **Children and regions — list_children() / has_child()** (4 tests)
    - `list_children()` returns sorted list of child IDs
    - `has_child(id_or_obj)` checks by string or Timeline object

11. **Segment lines — create_segment_line()** (4 tests)
    - Creates SegmentLine from boundary coordinates
    - Does NOT modify source timeline
    - Returns `SegmentLine[type(source)]` with the source timeline API

12. **Segment lines — create_segment_line_from_regions()** (3 tests)
    - Creates SegmentLine from contiguous regions
    - Validates contiguity

13. **Segment lines — create_segment_line_by_grouping()** (3 tests)
    - Shortcut: groups events and creates SegmentLine directly
    - Does NOT add intermediate regions to source

14. **Segment lines — create_segment_line_by_splitting()** (3 tests)
    - Shortcut: splits and creates SegmentLine directly

15. **Segment lines — SegmentLine list_segments() / has_segment()** (5 tests)
    - `list_segments()` returns ordered segment IDs
    - `has_segment()` checks by ID or object
    - `__contains__` override on SegmentLine checks segments

16. **Membership — __contains__ checks regions AND children** (4 tests)
    - String `in tl` checks region ids AND child ids
    - Region object `in tl` checks by id
    - Timeline object `in tl` checks by identity

17. **SegmentLine Basics and Casting Tests**
    - Empty creation
    - append_segment() adds contiguous children
    - Segment offsets form contiguous sequence
    - Rejects non-contiguous offsets
    - First segment must start at 0
    - `add_child` on a segment line takes every parameter of
      `Timeline.add_child`, `use_conversion_map` included: on a line of one
      4 s segment carrying a `ScalarMap` seconds → milliseconds, a 2000 ms
      timeline added at 4 s with `use_conversion_map=True` becomes the second
      segment, at 4.0 s, of length 2.0 s in seconds, the line 6 s long, the
      timeline given unchanged (2000 ms, not a child of the line); a further
      1 s segment added with `use_conversion_map=None` makes the line 7 s.
      An override dropping the parameter refuses both calls with a
      `TypeError`.
    - get_segment_by_index()
    - get_segment_at() finds segment by coordinate
    - `is_segment_line` uses exact structural coverage, including
      zero-length children, and rejects an empty child collection
    - `as_segment_line()` preserves content while enforcing contiguity and a
      homogeneous child class; its segments are copies of the children,
      never the children themselves
    - `to_timeline()` removes segment enforcement without losing content,
      holding copies of the segments
    - Boundary-created children make the parent structurally segmentable;
      casting returns the parent's concrete dynamic SegmentLine class with the
      exact child count, names, and offsets
    - A SegmentLine nested beneath a parent renders its segments as further
      rows like any other child: the diagram's `depth` argument alone decides
      how far the rendering descends (`depth=1` stops at the segment line)

18. **SegmentLine from_segmentation Tests**
    - Creates correct number of segments
    - Segments have correct lengths
    - Copies events to respective segments
    - Requires at least 2 split coordinates
    - Fails if source has existing children
    - Rejects invocation through a parameterized class whose parameter differs
      from the source's exact class

19. **Timeline.derive() Tests** (8 tests)
    - Creates timeline in target unit
    - Creates correct Timeline subclass for domain
    - Attaches inverse C-map for roundtrip; the exact 2× tempo inverse
      (`60 / 2 = 30`) is asserted with `==`
    - Roundtrip accuracy verification: `test_derive_roundtrip_accuracy` scales
      by a 1.5× tempo and inverts with `÷ 1.5`. For its fixed inputs
      (0, 25, 50, 75, 100) each product `x·1.5` and its division back are
      exactly representable, so the round trip recovers every value
      **bit-exactly** and is asserted with `==`. (1.5 is a rational, in fact
      dyadic, scale — not irrational; the earlier claim that this needed a
      tolerance was wrong. `test_timeline_relationships.py` now retains no
      `pytest.approx`; the only remaining tolerances in this directory are the
      two load-bearing sites in `test_groups.py` described above.)
    - Raises ValueError without C-map
    - Uses custom name if provided
    - copy_events=True copies and converts events
    - copy_events=False (default) creates empty timeline

20. **get_timeline_class() Tests** (7 tests)
    - Returns correct class for all 6 domain/modality combinations
    - Raises ValueError for unknown domain

21. **get_events_at() Tests** (10 tests)
    - Returns instant events at exact coordinate
    - Uses tolerance for instant matching
    - Interval left-inclusive (start included)
    - Interval right-exclusive (end excluded)
    - Interval middle coordinates included
    - Includes events from children
    - include_children=False excludes child events
    - Returns dict keyed by timeline ID
    - Returns empty dict when no match
    - Accepts Coordinate object input

22. **Real Data Tests (Ms3Loader)** (13 tests)
    - See "Real Data Validation" section below

23. **Integration Tests** (3 tests)
    - Regions used to create SegmentLine structure
    - Derived timeline can have children added
    - SegmentLine with individual segment C-maps

**Validity Rationale:**

TimeToAlign concepts validated here:
- A Region is a named part of a timeline that is defined by a TimeInterval.
- When all Children of the same parent timeline are contiguous, they are
  Segments and the parent is a SegmentLine.
- A ConversionMap implies the presence of a derived timeline in the target unit.

These tests ensure:
1. Region is NOT a Timeline (immutable dataclass, no events/C-maps)
2. SegmentLine enforces strict contiguity (no gaps/overlaps)
3. derive() creates proper cross-domain relationships via C-maps
4. Inverse C-maps enable accurate roundtrip conversions
5. get_events_at() enables point-in-time queries following [start, end) interval semantics
6. Coordinate errors propagate across event range filters, point queries,
   region and child lookups, length assignment, region creation, and
   `SegmentLine.get_slice()`
7. The unified verb×noun API provides consistent naming across all noun types
8. Real data from Ms3Loader validates the API against production musicological data
9. Dynamic SegmentLine classes preserve both segment behavior and the complete
   API of their parameter timeline class

---

#### `is_segment_line` is a read-only property

`Timeline.is_segment_line` is a property, like the other `is_*` answers of a
timeline (`is_locked`, `is_length_known`) and like the `is_segment_line` key of
the dictionary form. Written as a method, `timeline.is_segment_line` without a
call was a bound method and therefore always true, so the tests prove the
property by comparing with `is True` / `is False`: a bound method is neither,
and every such assertion fails if the attribute ever becomes a method again.
`TestSegmentLineCasting.test_is_segment_line_is_a_read_only_property` adds
three facts the comparisons cannot show: the class attribute is a `property`;
a timeline whose children leave a gap answers `False` without being called;
and assigning to the attribute raises `AttributeError`, since the answer is
derived from the children and never stored. The answer is structural for
every class: a plain `Timeline` that tiles answers `True`, and the
`SegmentLine` cast of it (`as_segment_line()`) answers the same.

### Real Data Validation (`TestUnifiedAPIWithRealData`)

**Purpose:** Validates the unified verb×noun API using real musicological data from
Wagner's Walküre Act III (measures.tsv loaded via Ms3Loader).

**Test Data Provenance:**

| Property | Value | Source |
|----------|-------|--------|
| File | `Wagner_WWV086B-3.measures.tsv` | DCML Ms3 corpus (MuseScore annotation) |
| Location | `tests/data/score/wagner_walkure/01_RawData/score_musescore/` | |
| Total Measures | 1733 | Exact count from TSV rows |
| Timeline Length | 6699.5 quarter beats | Computed from last event end coordinate |
| Unique Time Signatures | 8 (`9/8`, `3/4`, `2/2`, `12/8`, `2/4`, `4/4`, `6/4`, `6/8`) | TSV `timesig` column |
| Adjacent Timesig Runs | 22 | Adjacent-grouping count |
| Break Events | 405 (347 line + 58 page, 0 section) | TSV `breaks` column |
| All Break Coordinates | 405 unique | Verified no duplicate end coordinates |

**Fixture Strategy:**

The `wagner_timeline` fixture uses `MeasureData.create_timeline()` which directly
assigns the MeasureData as the timeline's event store. This preserves all 32 fields
(including `timesig`, `breaks`, `keysig`) that the base `EventData.from_dicts()`
would discard (it only keeps the 7 base schema fields). Each test invocation
creates a fresh `Ms3Loader` instance, ensuring no shared state between tests.

**Gold Standard Counts (all EXACT, no approximations):**

| Test | Assertion | Value | Derivation |
|------|-----------|-------|------------|
| `test_event_count` | `n_events` | 1733 | TSV row count |
| `test_create_regions_by_grouping_timesig` | region count | 22 | Adjacent-run counting on timesig field |
| `test_create_regions_by_splitting_breaks` | region count | 406 | 405 unique split points + 1 |
| `test_create_regions_by_splitting_page_breaks_only` | region count | 59 | 58 unique page break coords + 1 |
| `test_create_regions_by_splitting_section_breaks` | region count | 1 | 0 section breaks → 1 region (whole timeline) |
| `test_create_segment_line_by_grouping_timesig` | segment count | 22 | Same as timesig runs |
| `test_create_segment_line_by_splitting_page` | segment count | 59 | Same as page break regions |
| `test_create_child_from_region_with_real_data` | child length | 968.0 | First 9/8 region span (0.0–968.0) |
| `test_create_child_from_region_with_real_data` | child events | 216 | Measures in 0.0–968.0 range |
| `test_create_children_from_regions_all` | children count | 22 | One child per timesig region |

---

### FlowMaps (`test_flowmap.py`)

A `FlowMap` is written as the ID of ONE root timeline and an ordered list of
chunks, each the ID of a Region (of the root or of a descendant), of a child
timeline at any depth or of a virtual timeline of that tree, or a `Gap` of
positive length. Read against its root it is a list of intervals of the
ROOT's axis placed one after the other on a virtual axis that starts at 0.
Everything a FlowMap answers is arithmetic on intervals that the test
states itself, so every expected value is worked out by hand from the
stated offsets and bounds and written as a literal; none is taken from the
code. The timelines hold no flow control, no measures and no events: a
FlowMap needs none.

**The quarters case** (fraction axis). Root `set`, length 40. Children
`first` (offset 0, length 8) and `second` (offset 8, length 12); `second`
holds the child `trio` (offset 4, length 6) and the Region `coda`
([10, 12) on its own axis). The root holds the Regions `intro` [0, 3) and
`tail` [30, 40). On the root's axis, by adding the offsets from the root
down: `first` = [0, 8), `second` = [8, 20), `trio` = [12, 18), `coda` =
[18, 20).

The FlowMap `second, Gap(4), first, trio, coda, intro`:

| Chunk | Root interval | Virtual interval |
|---|---|---|
| `second` | [8, 20) | [0, 12) |
| gap 4 | none | [12, 16) |
| `first` | [0, 8) | [16, 24) |
| `trio` | [12, 18) | [24, 30) |
| `coda` | [18, 20) | [30, 32) |
| `intro` | [0, 3) | [32, 35) |

Length 35. `unfold` returns one virtual position per chunk covering the
root position, in played order (a chunk is half-open): 2 → [18, 34]
(`first`, `intro`); 13 → [5, 25] (`second`, `trio`); 19 → [11, 31]
(`second`, `coda`); 20 → [] (the end of `second` is excluded and nothing
else covers it); 25 → [] (never played). `fold` returns the root position
of a virtual position: 0 → 8, 11 → 19, 16 → 0, 24 → 12, 31 → 19, 34 → 2;
12 and 15 → None (inside the gap); 35 and -1 are refused (the virtual axis
is [0, 35)).

**The seconds case** (float axis; every number a multiple of 0.25, so the
float arithmetic is exact). Root `audio`, length 30.0, Regions `verse`
[0.5, 10.25) and `chorus` [10.25, 20.0). FlowMap `chorus, verse, Gap(1.5),
chorus`: `chorus` at [0, 9.75), `verse` at [9.75, 19.5), the gap at
[19.5, 21.0), `chorus` at [21.0, 30.75); length 30.75. `unfold(12.0)` =
[1.75, 22.75]; `unfold(0.25)` = []; `fold(10.0)` = 0.75; `fold(20.0)` =
None; `fold(30.0)` = 19.25. Every result is a `Coordinate` holding a float
on a float axis and a `Fraction` on the fraction axis: the number type is
the root's, whatever form the question had.

**Positions in and out.** A position is given as a bare number, a
`Coordinate` or an `IdCoordinate`, and every answer is a `Coordinate`.
Refused: a Coordinate in another unit; an IdCoordinate naming another
timeline than the root on `unfold`; one naming the root on `fold` (a root
position is no virtual position).

**Gaps keep their length in the axis's number type.** `Gap(4)` on the
fraction axis is stored and written as `[4, 1]`, on a float axis as `4.0`;
a gap whose length is no value of the axis's number type (`Gap(4.5)` on a
samples axis, `Gap(Fraction(1, 3))` on a float axis) is refused, since no
value is rounded. `Gap(0)` and a negative length are refused by `Gap`
itself.

**Refusals**, each leaving the timeline's FlowMaps unchanged: a chunk
naming nothing in the root's tree (a Region of another root, an unknown
ID); the root itself as a chunk (refused when the FlowMap is constructed);
a FlowMap naming another timeline than the one it is given to; a FlowMap
given to, or read against, a child timeline; a child timeline of unknown
length as a chunk (`UnknownLengthError`); a second canonical FlowMap of
one timeline; a given ID the manifest already holds; a chunk that is
neither an ID nor a `Gap`. A root that holds FlowMaps and is then placed
in another timeline as a child keeps them, and the manifest is refused when
written, in the format's words (`'set' is a child timeline; a FlowMap names
a root timeline`): a FlowMap is never dropped silently.

**A virtual timeline as a chunk** stands for the stretches of the FlowMap
it applies. With `twice` = `first, first` (length 16) applied as the
virtual timeline `z`, the FlowMap `z, Gap(1), second` plays [0, 8), [0, 8),
a gap of 1, [8, 20): length 29, `unfold(2)` = [2, 10], `fold(16)` = None,
`fold(17)` = 8. A FlowMap whose chunk is a virtual timeline applying that
same FlowMap names itself. The format refuses such a manifest (no length
the virtual timeline could state is the sum of chunks that include
itself), which is asserted in its words; the library's own refusal is
asserted on the same two objects held in memory, where no validator
stands before it. A chunk naming a virtual timeline that states another
unit than the root's (the one way a chunk can lie on an axis in another
unit, since the regions and children of a tree share the root's unit) is
refused when read against its root; it is built as a manifest, which the
format admits.

**The dictionary form** is the manifest's, IDs bare as in every object of
the library: `{id, timeline, chunks, convention?, canonical?, name?,
provenance?}`, a gap written `{gap: <Number>}`. The tests read literals
and compare objects, and write objects and compare literals: the minimal
form leaves `canonical: false` and `name: null` out and the complete form
writes them; `convention` and `provenance` are written only when stated,
in both forms; `default`, `single` and `printed` are read as the
enumeration's members and any other word is kept as given; a key the
library does not know is kept and written back, on the FlowMap and on a
gap.

**IDs and references stay true.** A FlowMap ID is minted `flow_map_{N}`
in the timeline's manifest. When two trees whose minted IDs collide join
one bundle, the second tree's Region, FlowMap and virtual timeline are
minted again and the FlowMap's chunks, its `timeline` and the virtual
timeline's `map` name the new IDs; the test states the expected IDs from
the minting rule (the number above the highest held).

### Atomic sections and computed FlowMaps (`test_flow_maps_from_flow_control.py`)

`Timeline.create_flow_maps_from_flow_control()` plays the timeline's flow
control with the engine and writes the result: the atomic sections as
Regions and the computed flows as FlowMaps over them. `Timeline.get_flow()`
computes one flow and writes nothing. The engine is proven in
`test_flow_engine.py`; what is proven here is that its result is written as
the format says, so the expected values are those of the first movement of
Beethoven's Op. 106, each derived by hand below.

**The Op. 106/i shape.** A quarters timeline of length 1644 (fraction axis)
with four elements: a Jump at 953/2 with the destinations `{by: 0, passes:
[1]}` and `{to: jump_target_b, passes: [2]}` (the two brackets), a Jump at
500 `{repeat_count: 1}` (the repeat sign), a Break at 500 between the
brackets and the JumpTarget `jump_target_b` at 500.

| Quantity | Expected | Why |
|---|---|---|
| atomic sections | A [0, 953/2), B [953/2, 500), C [500, 1644) | the boundaries are 0, 953/2, 500 and 1644 |
| each section | a Region named by its letter, `statements == {"type": ["tts:AtomicSection"]}`, `provenance == [{"property": "rdf:type", "value": {"id": "tts:AtomicSection"}, "method": "inferred"}]`, the prefix `tts` declared as the format's namespace, `is_atomic_section` true | the form of the format's fixture `flow/sonata_exposition` |
| `default` | A B A C | first pass through the first bracket, back, second pass into the second |
| `single` | A C | once through, on the pass after the repeat |
| `printed` | A B C | the axis as it stands |
| each FlowMap | its `convention`, provenance `[{method: inferred}]`, not canonical, unnamed | |
| length of `default` | 953/2 + 47/2 + 953/2 + 1144 = 4241/2 | |
| length of `single` | 953/2 + 1144 = 3241/2 | |
| `default`, `unfold(393/2)` | [393/2, 1393/2] | A is played at 0 and at 500 |
| `default`, `unfold(961/2)` | [961/2] | B is played once, where it stands |
| `default`, `unfold(504)` | [1961/2] | C begins at 1953/2 |
| `default`, `fold(1393/2)` | 393/2 | |

`Region.is_atomic_section` reads the type however the Region spells it
(the absolute IRI, or any prefix declared for the format's namespace) and
is false for a Region without it or with another type. It is false, too,
for a Region whose statement of the type holds a belief `false` of any
actor, under either spelling, while a belief `unlikely` or `true` leaves
it an atomic section: rejection is read as the format reads it on a
flow-control element.

The virtual timeline of `default` states the length 4241/2, quarters and
`fraction`; its minted ID is of the timeline's kind (`clt_1`, the root
having a given ID); its `unfold` and `fold` answer as the FlowMap's do; a
position given as an IdCoordinate of another timeline is refused, and so
is a virtual timeline stating another length than its FlowMap gives.
`get_flow` returns the same lengths and positions for the three
conventions and leaves the timeline without Regions and FlowMaps.

**Findings come back with what is computed.** A caller must be able to see
that flow control is broken without reading a log, so both calls return
the engine's findings as values. The timeline `Jump 4 {}` of length 8 goes
round for ever: the engine stops at the second take with `endless_loop` at
4 and `never_played` [4, 8) (the case "endless loop" of
`test_flow_engine.py`). `get_flow()` of that timeline returns the two runs
[0, 4) [0, 4) and, in `findings`, exactly those two findings by kind,
coordinate and end; `get_flow("single")` and `get_flow("printed")` return
none, since findings are a result of `default` alone;
`create_flow_maps_from_flow_control()` returns the same two findings
beside the sections and FlowMaps it writes. The flow returned is still a
resolved flow (`unfold`, `fold`, `length` answer as before), and a
timeline whose flow control plays through has `findings == ()` in both
results.

**`single` and `printed` are written only where they differ from
`default`.** A timeline whose only element is a section break has two
atomic sections and one flow: one FlowMap, `default`.

**Nothing is written without flow control**: the call returns None and the
timeline holds no Region and no FlowMap. On a timeline of unknown length
it raises `UnknownLengthError`.

**A child's flows are written on its root.** A FlowMap names one root
timeline and lists anything giving an interval inside that root's tree, a
child's Regions included; so the movements of a work, each a child with
its own flow control, have their sections as Regions of the child and
their FlowMaps on the root, side by side. The proof must show that the
Regions lie on the child's axis (a FlowMap of the root resolves them at
the child's offset), that two children's FlowMaps do not take each other
for their own when written again, and that the format accepts the result.
The root `sonata` (length 40) holds `movement` ("Allegro", length 12 at
offset 4, a repeat sign at 8) and `finale` ("Finale", length 12 at offset
20, a repeat sign at 6). Written for `movement`: `region_1` A [0, 8) and
`region_2` B [8, 12) on `movement`, none on the root; on the root
`flow_map_1` `default` (1, 1, 2) named "Allegro, default",
`flow_map_2` `single` (1, 2) and `flow_map_3` `printed` (1, 2), each with
`timeline` `sonata`. `flow_map_1` resolved on the root places [4, 12),
[4, 12), [12, 16): length 20; a virtual timeline of the root applying it
has length 20. Then written for `finale`: `region_3` A [0, 6), `region_4`
B [6, 12) on `finale`; `flow_map_4` `default` (3, 3, 4) "Finale,
default", `flow_map_5`, `flow_map_6`; the movement's three FlowMaps are
unchanged. The manifest written (`to_dict`, which the format's validator
checks) and read back holds the same Regions and FlowMaps.

Written again on the child, the guarantees of a root hold. A repeat sign
added at 12 (a repeat end without place goes back to the start of the
timeline, restoring the repeat inside) adds no boundary: the Regions are
kept, `flow_map_1` keeps its ID and name and lists (1, 1, 2, 1, 1, 2),
and the virtual timeline takes the new length 40. A JumpTarget added at 4
in `movement` instead divides A: `region_1` stays and is rejected by the
actor, `region_5` A [0, 4), `region_6` B [4, 8), `region_2` C [8, 12);
`flow_map_1` keeps its ID and lists (5, 6, 5, 6, 2); the finale's
FlowMaps are unchanged. Without an actor that call raises and changes
nothing.

**A timeline is played with its tree** (`TestTree`). Flow control stated
on a root or on any descendant holds across descendants: a timeline is
played with its own elements and those of every descendant, each at its
position on the played timeline's axis by offset arithmetic
(`Timeline.tree_flow_control`). What would prove it is that where an
element is stated makes no difference to the flow, only where it lies on
the played axis; that a child is played with its subtree alone; and that
the rule "one jump at a coordinate" holds on the root's axis over the
whole tree, refused where a Jump or a child is added, as the format's
validator refuses such a manifest. The
values below follow from the walk of the engine (`test_flow_engine.py`)
applied to the positions on the played axis, nothing else.

The tree: the root `work` (length 40) holds `allegro` (length 12 at
offset 4) and `finale` (length 12 at offset 20); `allegro` holds
`codetta` (length 4 at its offset 8). `allegro` states the JumpTarget
`start` at 0 and the Jump `repeat` at 8 `{to: start, repeat_count: 1}`;
`codetta` the Barline `codetta_end` at 4; the root the Jump `dc` at 32
`{to: work, repeat_count: 1, reset: false}` (a da capo: the root's own
start is a place).

| Quantity | Expected | Why |
|---|---|---|
| the root's view | `start` 4, `repeat` 12, `codetta_end` 16, `dc` 32, IDs unchanged | 0 + 4, 8 + 4, 4 + 8 + 4, 32 |
| `allegro`'s view | `start` 0, `repeat` 8, `codetta_end` 12; no `dc` | its subtree on its axis |
| `finale`'s view | empty | no element in its subtree |
| storage | `work.flow_control` holds `dc` alone, `allegro`'s `start` and `repeat` | the view writes nothing |
| root `default` | runs [0, 12) [4, 32) [0, 40), length 80, no finding | back to 4 at 12; on at 32 to 0; the repeat is not taken again (taken once, and the da capo restores nothing) |
| root `single`, `printed` | [0, 40), length 40 | nothing before its Jump is taken once through |
| the same four elements stated on the root at 4, 12, 16, 32 | the same runs and lengths in all three conventions | only positions on the played axis count |
| `allegro`'s `default` | runs [0, 8) [0, 12), length 20 | equal to a timeline of length 12 holding `start` at 0 and `repeat` at 8 |

**A destination without place.** It goes on at the nearest section
break before its Jump on the played axis, stated by any timeline of the
tree, but never before the start of the timeline stating the Jump, which
counts as a section boundary for it; at that start where no section break
lies from it up to the Jump. What proves the rule is a child's repeat
sign played within the root and alone, with section breaks stated before
the child, inside its stretch by the root, and inside it by the child
itself; each value follows from the sentence alone. The root `work`
(length 40) holds `allegro` (length 12 at offset 4), which states the
Jump `repeat` at 8 (12 on the root) without place, `repeat_count: 1`:

| Section breaks | Root's flow | Why |
|---|---|---|
| none | [0, 12) [4, 40), length 48 | back to `allegro`'s start, 4, not to the root's 0 |
| the root at 2 | [0, 12) [4, 40) | 2 lies before `allegro`'s start |
| the root at 6 | [0, 12) [6, 40) | 6 lies inside, from 4 up to 12 |
| `allegro` at 3 (7 on the root) | [0, 12) [7, 40) | stated by the child, read on the root's axis |
| the root at 6, `allegro` at 3 | [0, 12) [7, 40) | the nearest before 12 |
| the root at 12 | [0, 12) [4, 40) | a section break at the Jump lies not before it |

`allegro` played alone gives [0, 8) [0, 12) without a section break of
its own and [0, 8) [3, 12) with its break at 3: a root's section break
does not count when the child is played alone, since ancestors are not
part of its tree. The same Jump stated on the root at 12 goes back to
the root's 0: [0, 12) [0, 40) (length 52).

**Where the sections are written.** A root whose flow control lies only
in descendants gets its atomic sections as Regions of the root: without
`dc`, `work.create_flow_maps_from_flow_control()` writes `region_1` A
[0, 4), `region_2` B [4, 12), `region_3` C [12, 40) on `work`, none on
`allegro`; `flow_map_1` `default` (1, 2, 2, 3), `flow_map_2` `single`
(1, 2, 3), `flow_map_3` `printed` (1, 2, 3), unnamed, on `work`; the
manifest written passes the format's validator.

**One jump at a position of the root's axis.** A tree that the library
holds must be one it can write: the format's validator refuses two Jumps
not rejected at one position of a root's axis (`jump_repeated`), so the
library refuses the step that would bring them together, whichever it
is, and nothing changes then. The tree is the one above without `dc`.

| Step | Result |
|---|---|
| the root adds `skip` at 12 `{by: 4}` beside `allegro`'s `repeat` at 8 (12 on the root) | `ValueError`: "jump 'skip' is a second jump at 12 quarters of timeline 'work', after jump 'repeat' of timeline 'allegro' at the same position on their root's axis: several signs at one place are destinations of one jump"; the manifest is unchanged |
| the same through `add_all` beside a Barline | the same refusal; neither element is added |
| `codetta` (offset 8 in `allegro`) adds a Jump `on` at 2 `{by: 2}` (14 on the root) | stored: no Jump lies at 14 |
| `codetta` adds a Jump at 0 (12 on the root) | refused, naming `codetta` and `allegro`'s `repeat` |
| the root adds `skip` at 12 rejected (`provenance` with the belief `false`) | stored: a rejected Jump is out of the rule; the root plays [0, 12) [4, 14) [16, 40), `skip` read as absent and `on` taken |
| a root of length 40 holding `skip` at 12 adds a child of length 12 at 4 that states a Jump at 8 | `add_child` refuses, naming the child's Jump second; the root holds no child, the child no parent |
| the same child added at 5 | the child is added (its Jump lies at 13) |
| the manifest written after each accepted step | passes the format's validator |

Playing still reads the rule on the played axis, for what a float axis
cannot separate (below).

**Positions are summed exactly.** The validator reads a child's
position as the exact sum of the offsets down to it and of the
coordinate, a float as the exact binary value it is; the library sums
the same way and expresses the sum in the axis's number type once.
Floating-point addition gives other answers, so the cases are chosen
where it does, on a float axis (`seconds`):

| Tree | Exact reading | Library |
|---|---|---|
| root (length 2.0) → `ch` at 0.2 → `g` at 0.5 states a Jump at 0.2; the root adds a Jump at 0.9 | 0.2 + 0.5 + 0.2 is exactly the double 0.9: one position, `jump_repeated` | refused at `add`, as the validator does; the gathered Jump of `g` lies at 0.9 (summed in floating point from the root down, 0.2 + 0.5 + 0.2 gives 0.8999999999999999) |
| root (length 1.0) → `ch` at 0.1 states a Jump at 0.2; the root adds a Jump at 0.30000000000000004 | 0.1 + 0.2 is exactly 0.3000000000000000166533453693773481063544750213623046875, not the double 0.30000000000000004: valid | stored, and written and validated; playing the root refuses, since both Jumps lie at the double 0.30000000000000004 of the played float axis, which cannot separate them ("jump … lies at 0.30000000000000004 seconds, where jump … lies already") |

**The Op. 106 manifest.** The timelines and flow control of the sonata as
its manifest states them (`beethoven_piano_sonatas`,
`sonata_29_op106/manifest.json`): the root `clt_1` (15833/4) holds the
four movements `clt_2` (1644 at 0), `clt_3` (516 at 1644), `clt_4` (561
at 2160) and `clt_5` (4949/4 at 2721); `clt_2` states the brackets (a
Jump at 953/2, `{by: 0, passes: [1]}` and `{to: jump_target_1, passes:
[2]}`), the repeat sign at 500, the Break and the JumpTarget at 500 and
two barlines, the others barlines only. The root's `default` flow is the
first movement unfolded (A B A C on its axis, 4241/2) followed by the
other three whole: runs [0, 500) [0, 953/2) [500, 15833/4), length
15833/4 − 1644 + 4241/2 = 17739/4, no finding.

**Written again: nothing is taken away, and no ID comes to name another
stretch.** Whatever names a Region (a FlowMap written by hand, a statement
on another element, a Jump going on at it) must find the same stretch
under that ID after the sections were computed again. The proof is
therefore made on a timeline in which each of those three names a section
that the change takes away, and every value below follows from the rule
alone: a section whose stretch is still a section keeps its Region; a new
stretch gets a new Region; a Region whose stretch is a section no longer
stays as it is and its statement `type: tts:AtomicSection` takes the
recomputing actor's belief `false`; the letters are those of the new
sectioning.

The timeline is `|: 8 :| 12` (a repeat sign at 8, length 12). First call:
`region_1` A [0, 8), `region_2` B [8, 12); `flow_map_1` `default`
(1, 1, 2), `flow_map_2` `single` (1, 2), `flow_map_3` `printed` (1, 2).
Then are added: a Region `theme` [2, 6) stating `dcterms:relation`
`region_1`; a FlowMap by hand `backwards` (`region_2`, `region_1`); a
virtual timeline `played` applying `flow_map_1` (length 8 + 12 = 20).
Each row of the table is one change made to that state, on its own.

| Change | Sections after | What happened to the old Regions | Computed FlowMaps |
|---|---|---|---|
| a JumpTarget at 4 | `region_3` A [0, 4), `region_4` B [4, 8), `region_2` C [8, 12) | `region_1` stays [0, 8), still named A, rejected; `region_2` kept and renamed from B to C | `flow_map_1` (3, 4, 3, 4, 2); `flow_map_2` and `flow_map_3` (3, 4, 2) |
| a Break at 10 | `region_1` A [0, 8), `region_3` B [8, 10), `region_4` C [10, 12) | `region_2` stays [8, 12), named B, rejected; `region_1` kept | `flow_map_1` (1, 1, 3): the flow ends at the Break; `flow_map_2` (1, 3); `flow_map_3` (1, 3, 4) |

The new IDs are `region_3` and `region_4` in both rows because the
timeline holds `region_1` and `region_2` throughout: an ID is minted above
the highest held and never again for another stretch. Asserted after the
first row: `region_1` has its interval [0, 8) and is no atomic section any
more (`is_atomic_section` false), and its provenance entry for the type
holds the belief `{actor, belief: false, date}` with the actor and the
date given to the call; `theme` still states `region_1`; `backwards`
still resolves, to [8, 12) then [0, 8), length 12; `played` keeps its
length 20, since the default flow is as long as before. After the second
row `played` states 18 (8 + 8 + 2). The manifest written validates in both
rows. A further call with nothing changed needs no actor and changes
nothing: the rejected Region is no section and is not rejected a second
time. A stretch that comes back is a new stretch: after the first row the
JumpTarget at 4 is rejected and the sections are computed again by a
second actor. [0, 8) is a section again and gets `region_5`, not the
rejected `region_1`, which stays exactly as it was; `region_3` and
`region_4` are rejected by the second actor; `region_2` is kept and named
B again; the FlowMaps keep their IDs with (5, 5, 2), (5, 2) and (5, 2).

**The actor is asked for only when something is rejected.** With unchanged
flow control the second call writes the same sections and FlowMaps under
the same IDs and returns the same result, without `actor`. Where a
section's stretch is gone and no actor is given, the call raises naming
the sections and `actor=`, and has changed nothing (Regions, FlowMaps,
virtual timelines and the manifest written are compared before and after).
An invalid date is refused the same way.

**A FlowMap written before for `single` or `printed` stays** where its
chunks come to equal `default`'s. In `|: 8 :| 12` the repeat is rejected
and a section break added at 8: the sections are the same two (no actor
needed), every flow is (1, 2), and `flow_map_2` and `flow_map_3` are
still held, with those chunks, because a virtual timeline may apply them;
`played` states 12.

**FlowMaps of another origin stand untouched beside the computed ones.**
Three cases: the FlowMap by hand above; a FlowMap stating `inferred` under
a convention word of its own (`as recorded`), which is none of the three
computed ones and keeps its chunks; and FlowMaps that name a convention
but state no provenance, so that nothing says they were computed: the
manifest of `|: 8 :| 12` with its flows written is read again with the
`provenance` of its three FlowMaps removed; the two sections are kept as
they are, those three FlowMaps are untouched, and the computed FlowMaps
are written beside them as `flow_map_4` (1, 1, 2), `flow_map_5` (1, 2)
and `flow_map_6` (1, 2). The format's own fixture `flow/sonata_exposition`
states its sections and its FlowMaps as computed (`inferred`): computing
them again from its flow control gives the same three sections ([0, 12),
[12, 16), [16, 24)) and the same three FlowMaps under the same IDs
((1, 2, 1, 3), (1, 3), (1, 2, 3)) and changes nothing in the timeline,
which shows that what the library writes is the form the format states.
A Jump going on at a section (`to: region_2`, added after the first call)
keeps landing at 8 after the sections were computed again.

### FlowMaps and virtual timelines in a manifest (`test_manifest_flow_maps.py`)

`flow_maps` and `virtual_timelines` are lists at the top level of a
manifest. In memory each FlowMap is held by the root timeline it names and
each virtual timeline by the root of the tree it comes from, so they travel
with the timeline into a group or a bundle and are written wherever it is
written.

- **The format's fixtures are read and written byte for byte**:
  `flow/sonata_exposition` and `flow/flow_map_by_hand`, each in the
  complete and the minimal form, through `Timeline` and through
  `AlignmentBundle`. That alone would also hold for keys kept unread, so
  the objects read are compared with literals as well: the three FlowMaps
  of the exposition with their conventions, their chunks and the
  provenance `inferred` that says they were computed from its flow control
  (`is_computed`), its virtual timeline
  (length 36: A 12 + B 4 + A 12 + C 8), and the by-hand FlowMap with its
  gap of `Fraction(4)`, its name and its provenance entry.
- **Built in memory, written, read, equal.** A bundle of two root
  timelines (a score with computed flows and a virtual timeline, a set list
  with a FlowMap over children and a gap) is written with `to_dict`, read
  with `from_dict`, and holds equal FlowMaps and virtual timelines; written
  again it gives the same manifest. In the manifest written, `flow_maps`
  and `virtual_timelines` are sorted by ID across both roots, every ID is
  spelled `#id`, and a manifest of timelines without FlowMaps holds neither
  key.
- **Prefixes of a FlowMap's provenance.** An actor written `ex:editor` is
  read under the manifest's `@context`. A manifest written carries the
  prefix where the manifest (a bundle, or a timeline read from a manifest as
  a whole) declares it or the catalogue has it; a timeline built in memory
  holds no manifest, so an undeclared prefix is refused when written, in
  the format's words. A manifest stating such a prefix is read and written
  back equal.
- **Kept as read.** A virtual timeline that applies a conversion map, or
  states `through`, is no FlowMap applied to its root: it is read, written
  back unchanged and refuses to convert positions.
- **Provenance entries as the format admits them.** A FlowMap of a whole
  manifest carrying an empty entry, an entry with keys the library does
  not know and a belief with such a key is read, holds three entries and
  is written back with `flow_maps` equal to what was read (the reasoning
  is in "Flow control" above).

---

### `test_timestamps.py` - Cross-Section Timestamp Tables

**Purpose:** Validates `get_timestamp_table()` (both `format="table"` and
`format="dataframe"`) and the helpers that build the timestamp axis
(event-coordinate collection, boundary collection, local-coordinate
computation) across a timeline hierarchy.

**Fraction-length regression (`TestFractionLengthTimestamps`):**

A logical timeline (quarters/beats) carries `number_type = Fraction`, so its
length is a `Coordinate` whose `value` is a `Fraction`. The bounds check inside
`_compute_local_coordinates` masks out-of-range coordinates by comparing the
coordinate structs against the length exactly, on their ratio side
(`compare_number_column`, cross-multiplied in decimals), as a single position
is compared in a stamp; a float comparison would put a position just inside a
boundary on the wrong side of it (`test_offset_arithmetic.py`,
`TestExactTableContainment`). The length therefore never reaches a kernel as a
raw `Fraction`, and the coordinates kept stay exact.

| Test | Construction | Exact assertion |
|------|--------------|-----------------|
| `test_compute_local_coordinates_with_fraction_length` | CLT length `9/2`, probe `[0.0, 2.0, 4.5, 5.0]` | local = `[0, 2, 9/2, None]` (5 > 9/2 → null) |
| `test_get_timestamp_table_with_fraction_length` | CLT length `9/2`, events at `0, 3/2, 3` | `num_rows == 3`; the timeline's column = `[0, 3/2, 3]` as Fractions |
| `test_dataframe_with_fraction_length` | same timeline, `format="dataframe"` | shape `(3, 1)`; the column = `[Fraction(0), Fraction(3, 2), Fraction(3)]` (the axis's number type) |

**Validity Rationale:** The timestamp axis is the spine of every cross-domain
alignment view. A quarters/beats timeline — the most common logical timeline —
must produce timestamp tables and frames without raising. The asserted values
are the exact in-bounds coordinates: the out-of-bounds probe coordinate becomes
`null`, and the event coordinates pass through unchanged and exact.

---

### `test_get_slice.py` - Slicing a Timeline

**Purpose:** Validates `Timeline.get_slice()`, the primitive that cuts a
half-open stretch out of a timeline, and the assembly of slices into a
`SegmentLine`.

| Class | Purpose |
|-------|---------|
| `TestGetSlice` | Unit tests for `Timeline.get_slice()` |
| `TestSegmentLineAssembly` | Integration: slice + concatenate into a SegmentLine (this module's own manual helper) |

1. **get_slice() Unit Tests** (16 tests)
   - Basic slicing with coordinate shifting (`[start, end)` semantics)
   - Interval event truncation (`truncate_events=True/False`)
   - Child timeline recursive slicing
   - Number type preservation (Fraction stays Fraction)
   - Concrete class preservation (CLT returns CLT)
   - Parameterized SegmentLine slicing preserves its dynamic class
   - Discrete and physical timeline support
   - Error cases (invalid range, out of bounds)

2. **SegmentLine Assembly Integration Tests** (5 tests)
   - Contiguity enforcement (each segment starts where previous ended)
   - Total length = sum of slice lengths
   - Events preserved in assembled segments
   - Segment type matches source class
   - The assembled line is an instance of both SegmentLine and the source class
   - Repeated section assembly (same range played twice)

**Coordinate Handling:** EventData stores instant events under `start` (struct dict),
not `instant`. Use `struct_to_coordinate()` for reading; never cast to `float`
unnecessarily (loses Fraction precision).

---

### `test_unfolding.py` - The played flow against ms3's unfolded tables

**Purpose:** Validates the default flow that a score's flow control plays
against ms3's own unfolded measure tables (`TestUnfoldingGoldStandard`,
49 tests: 7 tests × 7 specimens).

- Each specimen's measures table is loaded, the flow control on its timeline is
  played by the engine (`default`), and the runs are stated in measure counts
- EXACT row count, MC sequence, mc_playthrough, mn_playthrough with suffixes
- EXACT quarterbeats as Fraction (not float), total unfolded length: the lengths of
  the runs sum up to it
- `flow_only` is played with the repair its target flow needs (two repeat
  starts, a jump past the third bracket); as its source stands the flow ends at the
  second of three brackets (`tests/loader/score/README.md`, "Gold flows")

**Gold Standard Exact Values:**

| Specimen | Folded | Unfolded | Total QB | Challenge |
|----------|--------|----------|----------|-----------|
| rachmaninoff | 374 | 374 | 2997/2 | No flow control (baseline) |
| polyrhythm_only | 14 | 14 | 45 | Line breaks only |
| musete | 58 | 138 | 384 | D.S. al Fine, anacrusis, 6/8 |
| rondeau | 60 | 138 | 195 | Rondeau form (D.S.) |
| op18_no4_mov4 | 226 | 291 | 1116 | Repeats + volta brackets |
| woo71 | 397 | 505 | 1078 | Complex split bars |
| flow_only | 15 | 31 | 81 | D.S./D.C. + voltas |

**Known Edge Cases:**
- **Anacrusis** (Musete): Incomplete first measure (1.5 QB instead of 3.0). Repeated instance preserves short duration.
- **Split bars** (WoO71): Single notated measure divided into two MCs. Preserved exactly through unfolding.
- **Volta brackets** (Op.18): Different MCs played on different passes. Correct volta selected per pass.
- **Rondeau form** (c11n08): ABACADA return pattern via D.S.-like mechanism.

---

## Validation Methodology

## Timestamp, Export, Factory, and Subclass Resolution Validation

`test_groups.py` verifies event-ID timestamp lookup with a two-member group.
The event coordinate remains the timestamp axis, the source ID identifies the
member containing the event, and the present-timeline sequence reflects every
coordinate surfaced by the timestamp. Bulk lookup preserves request order and
keeps an explicit missing-event row.

`test_beatgrid.py` exports a grid in both supported annotation formats. The
expected files include headers, row order, labels, and boolean spellings so
format-specific consumers receive a stable dictionary layout.

`test_factory.py` exercises `EventStore.create_timeline()` for direct events,
multi-store children, and store selection. It pins child IDs and event counts
to distinguish direct placement from child construction.

Stores of one unit and two number types make one timeline
(`TestStoresOfTwoNumberTypes`): a store holding a `fraction` quarters table
(`a` on `[1/3, 2/3)`) and then a `float` one (a Beat `b` at `0.75`) gives,
with `flatten=True`, a `fraction` timeline `t` of length `3/4` holding
`t:a` at `1/3` and `t:b` at `3/4` (the double `0.75` as its exact ratio),
and with `group_by="event_type"` a `fraction` timeline whose children
`Beat` and `Note` hold `b` at `3/4` and `a` at `1/3`. Both joins pass
through `EventData.concat`, which re-expresses the second table's rows in
the first's number type (`tests/storage/README.md`, "One axis per table").

A timeline made from an `EventData` holds its events in an `EventData` of
its own, over the same table (tables are immutable, so sharing one costs
nothing and changes nothing). Before, the timeline held the very object it
was given: two timelines made from one `EventData` and the `EventData`
itself were one object, so `a.add_events(...)` added the row to all three.
The test makes `a` and `b` from one source of one Beat (by
`EventData.create_timeline()` and by `Timeline.from_event_data()`), adds a
Beat to `a`, and asserts the counts `a` 2, `b` 1, source 1, and that no two
of the three are the same object. What `events` returns and what `extend`
does are unchanged: `extend` still replaces the table of the object it is
called on, which is now only that timeline's.

The timeline's `EventData` is a copy of the source's, of its class and its
state, not a new instance of the class: a `NoteEventData` holding
`has_rests=True` gave each timeline a `NoteEventData` with `has_rests`
`False`, the constructor's default. The test makes timelines of such a
source both ways and asserts `type(events) is NoteEventData` and
`events.has_rests is True` on each, and the same of `source.filter(...)`,
since a filter derives its table through the same copy.

`test_types.py` resolves continuous, discrete, string-unit, and unmatched-unit
inputs to the most specific matching classes. Further-derived classes also
participate in resolution. The unmatched input verifies the documented base
`Timeline` fallback when no registered subclass accepts a unit.

All tests follow these principles:

1. **Positive Tests**: Verify correct behavior under normal conditions
2. **Negative Tests**: Verify errors are raised for invalid input
3. **Boundary Tests**: Test edge cases (empty, zero, max values)
4. **Scale Tests**: Large inputs give exact results; no wall-clock thresholds
5. **Roundtrip Tests**: Verify serialization preserves data

Each test class includes a docstring explaining its validity rationale,
linking back to specific requirements from the TimeToAlign model and the
project's engineering standards.
