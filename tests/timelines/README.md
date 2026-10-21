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
``volta: 7`` and the value survives the round trip. A stated ``qstamp`` that
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

The number-type boundary is proved in both directions and through every storing
entry point: a break coordinate, both coordinates of a jump, and a marker. A
quarters registry bound to fractions receives float-authored coordinates at
``1.5`` and must store the exact ``Fraction(3, 2)``, because ``1.5`` is a dyadic
double whose exact value is three halves; a seconds registry bound to floats
receives ``Fraction(1, 3)`` and must store exactly Python's
``0.3333333333333333``, the nearest double to one third. Break, jump, and marker
unit mismatches are each rejected with both units named. A query coordinate
object whose unit differs from a bound registry's unit is rejected with both
units named as well, since comparing its bare value with stored values of another
unit would answer a different question; a raw number query still compares by
value, and an unbound registry compares values as it always has.

An unbound registry retains the input objects unchanged: an added break, jump, or
marker whose id is free is stored as the identical object, so a marker at an
``IdCoordinate`` keeps that ``IdCoordinate``. A raw marker number names no unit,
so an unbound registry rejects it. Constructor content passes through the same
add methods, but the registry copies the given containers first. The caller's
list therefore remains a distinct object in its original order, even though the
registry sorts its own copy by coordinate; of two breaks constructed without an
id (both minted ``flow_marker_1``), the one added second is stored as ``flow_marker_2``. Copying a registry must reproduce its
binding and exact content while subsequent mutation of either registry leaves the
other unchanged.

A region answers containment for raw numbers by value on its own axis. A
coordinate object in another unit is rejected with both units named instead of
comparing values that measure different quantities.

Flow-control elements and regions write every coordinate as a bare value on
the timeline's own axis, represented as its number type selects, and read it
back against that axis. Their round trips pin every manifestation, label/name
field, repeat count, and metadata dictionary. The registry is written as one
array of elements (see *Flow control in a manifest*); a JSON encode/decode
cycle cannot change it.

A structure-bearing timeline combines a measure map containing regular,
irregular, and constituent measures, each with the number and nominal length
its class requires (the constituent at offset ``1/3``, so a codec dropping the
offset fails strict map equality); a metric hierarchy; a region; one break;
one jump; one marker; one child; and one conversion map. Its default timeline
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
TimeToShare manifest: `{"timetoshare": "0.1.0", "timelines": [...]}` with an
optional `id`, `name`, `meta`, `groups` and `graphical`. A reader that only
reads what its own writer produced proves nothing about the format, so the
tests read manifests written by hand from the format's rules, and compare what
the library writes back against the same literal text.

- **The version marker.** Without `timetoshare` each reader raises exactly
  `not a TimeToShare manifest`; `"timetoshare": "0.2.0"` raises a message
  naming both `'0.2.0'` and `0.1.0`. Nothing else is read first, so the
  refusal cannot depend on the rest of the content.
- **Defaults are supplied.** A minimal manifest (only `id`, `unit` and
  `length` per timeline, only `id` and `coordinate` per marker, and so on)
  and its complete twin, both literal, read to objects whose complete and
  minimal forms are identical; the complete form written back equals the
  complete literal and the minimal form the minimal literal. For each of the
  sixteen units a minimal timeline reads with the unit's own number type —
  `int` for ticks, samples, frames and pixels, `fraction` for quarters and
  whole_notes, `float` otherwise — asserted per unit, so a reader that
  assumed one default for all would fail on two of the three groups.
- **The class comes from the unit.** A timeline object carrying a stale
  `"class": "ContinuousGraphicalTimeline"` beside `"unit": "seconds"` reads as
  a `ContinuousPhysicalTimeline`; the `class` key is not read, and is kept and
  written back as an unknown key.
- **The number type governs, the lexical form never.** One refused value per bare key —
  `length`, a root's `offset`, a child's `offset`, region `start` and `end`,
  the `coordinate` of a flow-control element —
  each raises the exact message naming the key, the value, the id of the
  object holding the key (the timeline for `length` and `offset`, a child's
  own id for its offset, the region or flow-control element for its keys) and
  what to write instead, e.g. `length 32 of 'score' is an int on a fraction
  axis: write [32, 1]`. The rewritten value is named only when the rewriting
  is exact, otherwise the axis's number type: `[5, 2]` on a float axis says
  `write 2.5`, `[1, 3]` says `write a float`, `[1, 2]` on an int axis says
  `write an int`, and `0.1` on a fraction axis names its exact pair
  `[3602879701896397, 36028797018963968]`. These are the messages of the
  TimeToShare validators, word for word. What a JSON round trip may
  respell is read as the number type decides, each case asserting the value
  AND its Python type: `480.0` on an int axis is the int `480`, `8` on a
  float axis the float `8.0`, `[1, 8.0]` on a fraction axis the Fraction
  `1/8` (a JavaScript writer may respell any member). The respelling never
  admits a value that no reading of the axis allows: `[1, 8.5]` (a
  non-integral member) is refused as no number, and `[480, 1.0]` on an int
  axis is still a fraction on an int axis (`write 480`).
- **Unknown keys are kept where they were found.** One literal manifest
  carries an `x-note` key on the manifest, a root and a child timeline, a
  region, a `flow_marker`, a `jump_from`, a `jump_to`, a conversion map and a
  map nested in it, a `measure_maps` entry, a group and `graphical`, each
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
  `qstamp` and an `actual_length` of `-1` each raise the exact message naming
  the row's position. A conversion map lacking a key its type writes and has
  no default for (a `TableMap` without `x_values`) raises naming the key.
- **Reading never changes a length.** A root `score` (quarters, length
  `[10, 1]`) holding a child `coda` at offset 8 with length 4 is refused,
  locked or not, with the TimeToShare validators' words: `child 'coda'
  ends at [12, 1], beyond the length [10, 1] of its parent 'score'`; a child
  `late` of unknown length at offset 12 is refused with `offset [12, 1] of
  'late' is beyond the length [10, 1] of its parent 'score'`. A child ending
  exactly at 10 is read, and the parent writes `[10, 1]` back in both forms.
  The library used to grow the parent to 12 instead, which is what these
  pin. On a seconds root of length `60.0`, an end of `62.5`, an offset of
  `75.0` and a segment of unknown length whose successor starts at `61.0`
  are refused with the values spelled as written (`ends at 62.5`, `offset
  75.0`, `ends at 61.0`). The setters keep the library's own message, since
  no JSON spelling exists there.
- **Ids and lengths the format reserves.** The part words `start`, `end`,
  `name`, `label` and `card` follow an address as its parts and are no id:
  `start` as a root id, `end` as a region id, `card` as the manifest id,
  `name` as a group id and `label` as a conversion-map id are each refused
  with `Invalid id '<word>': it is reserved for the parts of an address`,
  and so are the overrides `Timeline.from_dict(..., timeline_id="start")`
  and `TimelineGroup.from_dict(..., group_id="end")`. The ids of a
  timeline's five bands follow its address too, so every id of the form
  `<kind>_<N>`, a kind among `markers`, `conversion_maps`, `children`,
  `regions` and `flow_control` and N a number from 1 without a leading zero, is
  no id either: `markers_1` as a root id, `children_1` as a child id,
  `regions_2` as a region id, `conversion_maps_3` as a map id,
  `flow_control_1` as a group id and `markers_10` as the manifest id are
  each refused with `Invalid id '<id>': it is reserved for the bands of a
  timeline`, one place per kind so that no kind word is left unchecked, and
  so are the overrides `Timeline.from_dict(..., timeline_id="children_1")`
  and `TimelineGroup.from_dict(..., group_id="regions_1")`. A negative length is
  refused on reading (`[-12, 1]` on quarters, `-8.0` on seconds:
  `length -12 of 'x' is negative`) and on construction.
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
  `{"order": {}, "statements": [], "styles": {}}` in the complete form,
  through `Timeline` and `AlignmentBundle` alike; a manifest read without
  `graphical` is written without it. Presence is part of the round trip.
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
  timeline's id, of a root or a descendant. The list form is refused with
  `'graphical.order' must map group or timeline ids to orders`, and a key
  naming neither with `'graphical.order' names no group or timeline:
  ['g3']`. The keys `score` (a root) and `part` (a child) of the band-order
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
  `["perf", "part"]`. Each timeline's band order is a permutation no
  default produces and no sort reproduces: `score` the README's example
  (`score/children_1`, `score/markers_1`, `score`, `score/conversion_maps_1`,
  `score/regions_1`, `score/flow_control_1`), `part` every band above its
  line, in reverse default order, and `perf` every band below it, in
  reverse default order. A root's band order (`score`, `perf`) and a
  descendant's (`part`), and a member's beside a non-member's, are each
  exercised. Every reader that reads a literal as a whole (timeline and
  bundle on the one-root manifest, group and bundle on the grouped one)
  reads each literal in each form, complete and minimal, and writes it back
  in the same form equal to the literal, with the lists compared in order,
  and in the other form equal to the other literal. Mutating the input's
  lists after reading changes nothing written, so the library holds its own
  copy. A band order that is no permutation of the bands (`["part",
  "part/markers_1"]`) is read and written back unchanged, which is what
  leaving the check to the validators means.
- **A band order without its timeline is left out.** An entry is written
  when a timeline of its key's id is written, and left out otherwise. An
  entry names the bands by its timeline's id (`score/markers_1`), so it
  cannot follow a timeline to a new id unchanged, and a key naming no
  written timeline makes the manifest invalid; leaving such an entry out
  keeps the written manifest valid and changes nothing else. A group read
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

### Measure maps in a manifest (`test_manifest_measure_maps.py`)

A manifest holds its measure maps in the top-level `measure_maps` array, each
`{id, name?, meta?, rows}`, written sorted by id; a timeline reaches one
through a `MeasureRecordMap` among its conversion maps, and every measure
view names one by id. The library keeps one measure map per timeline; the
tests prove that the writer lifts it into `measure_maps`, that the reader
puts it back, and that every legacy measure-derived map names the right
entry, with literal manifests written from the format's rules.

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
  the timeline, group and manifest ids: an entry whose id is a timeline's,
  a group's or the manifest's, and two entries of one id, are refused with
  the exact message; a minted id is minted again around them
  (`measuremap_1` held by a timeline makes the entry `measuremap_2`). A view
  naming an id no entry has is refused naming it.
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

### Flow control in a manifest (`test_manifest_flow_control.py`)

`flow_control` is one array of elements `{id, type, manifestation,
coordinate, target?, repeat_count?, name?, label?, meta?}`, written in
canonical order by coordinate and then id. A `Break` writes a `flow_marker`
(`fine`, `section_break`, `double_barline`, `final_barline`), a `Jump` a
`jump_from` (`repeat_end`, the six D.C./D.S. manifestations, `to_coda`) and
a `Marker` a `jump_to` (`repeat_start`, `segno`, `coda`); the element's
manifestation is the library element's `control_type`, which each class
restricts to its own set (constructing a `Break` of `segno` raises).
`Marker.control_type` is required: the library states whether a marker is a
segno, a coda or a repeat start rather than guessing it from a name.

- **The literal round-trips.** A quarters timeline with a section break, a
  fine, a segno, a coda, a repeat start, a repeat end (`repeat_count` 2)
  targeting the repeat start, a D.S. al Coda targeting the segno and a
  `to_coda` targeting the coda reads from its complete and minimal literal
  and writes each back equal to the literal; every element's id, name,
  label, meta and unknown key survives.
- **The target is a `jump_to`, and the jump's destination its coordinate.**
  Reading sets a jump's `to_coordinate` to its target's coordinate, or 0 for
  a `da_capo*` or `repeat_end` without target. Writing a jump whose target
  marker lies elsewhere than its `to_coordinate` is refused naming both.
- **A destination without a marker gets one.** A `repeat_end` to 8, a
  `dal_segno` to 4, a `to_coda` to 12 and a `da_capo_al_coda` to 10, none
  with a marker, write a `jump_to` at each destination whose manifestation
  follows from the jump (`repeat_start`, `segno`, `coda`, `coda`), with ids
  minted in the timeline's tree (`jump_to_1`, ... around the ids it holds,
  among them a break given the id `jump_to_1`, which the minting steps
  around);
  two jumps to one destination of one manifestation share one. A
  `repeat_end` and a `da_capo` to 0 write no target. A `da_capo` to another
  position than 0 is refused.
- **When an element acts follows from its manifestation.** The library's
  `condition` is derived, not stored: `repeat_end` acts on its first
  `repeat_count` passes (`first_n`), `to_coda` only after a D.C. or D.S.
  (`after_dc_ds`), every other element always. `repeat_count` is written on
  `repeat_end` only; a `Jump` of another manifestation with a
  `repeat_count` other than 1 is refused at construction, and reading a
  `repeat_count` elsewhere, a target on a `jump_to` or `flow_marker`, a
  target naming no `jump_to`, a missing target on a `dal_segno`, an unknown
  `type` and a manifestation of another type are each refused with the
  exact message.
- **Score loaders keep their traversal.** The flow tests (`test_flow.py`)
  are the oracle and stay green: the traversal reads the controller, not
  the registry. The registry of the ms3 measures of Couperin's
  `c05n05_musete` (repeats and two D.S. al Fine) writes and reads back to
  the same breaks and the same jumps (id, manifestation, from, to,
  `repeat_count`, condition) with exact values; the two repeat ends landing
  at 42 and 84, where no marker lies, gain the `repeat_start` markers
  `jump_to_2` and `jump_to_3` written there, and the read registry is a
  fixpoint of writing.

### Manifest identity (`test_manifest_ids.py`)

Every element of a manifest (a timeline, a conversion map, a region, a break,
a jump, a marker) carries an `id` for identification and a `name` for
display. The tests prove four properties, each against literal expected ids.

- **One form, one generator, no process state.** An element constructed
  without an id is minted `{prefix}_{N}` with N the smallest positive integer
  no id in its scope holds. A standalone element's scope is empty, so each
  class answers `_1`: the six timeline classes `clt_1`, `dlt_1`, `cpt_1`,
  `dpt_1`, `cgt_1`, `dgt_1`; the base `Timeline` and a segment line by their
  unit (`Timeline(unit="seconds")` is `cpt_1`, ticks `dlt_1`, pixels `dgt_1`);
  `LinearMap` `lmap_1`, `TableMap` and its subclasses `tmap_1`, `PiecewiseMap`
  `pmap_1`, every other map `cmap_1`; `region_1`; and a flow-control
  element by its manifest `type`: a `Break` (a `flow_marker`)
  `flow_marker_1`, a `Jump` (a `jump_from`) `jump_from_1`, a `Marker` (a
  `jump_to`) `jump_to_1`. Determinism is proved by building one reference manifest
  (a quarters root, two children made by `create_child`, two regions, two
  breaks, a marker and a jump targeting it, a `LinearMap`) twice, the second
  time after constructing five unrelated timelines and four maps, and once
  more with its collections added in the reverse order. The first two must
  serialize to identical `to_dict` payloads, and all three must carry
  exactly the ids spelled out literally (`clt_1`; children `clt_2`,
  `clt_3`; `region_1`, `region_2`; `flow_marker_1`, `flow_marker_2`;
  `jump_to_1`; `jump_from_1` targeting `jump_to_1`; `lmap_1`). A
  process-global counter gives
  different ids in the second build; insertion order may decide which
  element holds which id, never which ids exist. Smallest-free
  is pinned by a given child id `clt_3` on a `clt_1` root: the next two
  `create_child` calls mint `clt_2`, then `clt_4`. No minted timeline id
  starts with `tl:` and none lacks the `_` separator.
- **Whatever is given is the id.** Hand-chosen ids (`performance`, `a`,
  `Bass Kick`, `score:clt_7`) are kept verbatim on timelines, maps, regions,
  breaks, jumps and markers, and survive `to_dict` → JSON → `from_dict` →
  `to_dict` text-identically; a minted manifest round-trips its minted ids
  verbatim too. Refused ids raise `ValueError` on every kind: the empty
  string, any id containing `/`, and any containing a control character
  (`"a\tb"`, `"a\nb"`); `Bass Kick` (a space) is accepted.
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
  whose marker `jump_to_1` collides is re-minted `jump_to_2` and its jump's
  `target` follows to `jump_to_2`. Given ids: a child `score` under a root
  `score` raises naming `timeline 'score'` twice; a child timeline `intro`
  under a root holding a region `intro` raises naming `region 'intro'`; a
  region `intro` added beside a child `intro` raises; adding a given map id
  already held by a child's map raises. Each raise is atomic: the parent
  keeps its children, regions and maps, and the joining child keeps its id.
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
- **Regions and markers are keyed by id; jumps target marker ids.** Two
  regions named `Chorus` coexist as `region_1` and `region_2`; `get_region`,
  `has_region`, `list_regions`, `create_child_from_region` and
  `create_children_from_regions` take ids, and `get_region("Chorus")` raises
  `KeyError` listing the available ids. `create_child_from_region` names the
  child after the region (`name == "Chorus"`) and mints its id (`clt_2`),
  never reusing the region's id. `add_jump` refuses a `target` that names no
  marker of the registry. A score loader resolves D.S. targets to marker
  ids: the ms3 measures of Couperin's `c05n05_musete` (corpus `score`,
  `couperin_concerts`) load one marker, `jump_to_1`, a `segno` named `segno`
  at quarter 12 (its Fine is a break at the end of its bar, 42, and no jump
  target), and both D.S. al Fine jumps target `jump_to_1`; the da capo and
  repeat jumps of the Rondeau `c11n08_Rondeau` target `None`.

Loader parity (`test_manifest_ids.py::TestLoaderParity`): for one specimen
each of the ms3 (with and without D.S. jumps), score-MIDI, lab, audio,
Parangonada and IIIF loaders the test walks the whole
`to_dict` tree and requires every timeline, child, map, region, break, jump
and marker to carry a valid id (non-empty, no `/`, no control character),
all ids within one manifest to be distinct, and no id to start with `tl:`.

### Canonical order of a serialized timeline (`test_canonical_order.py`)

`Timeline.to_dict` writes every unordered collection in one canonical order,
so that two timelines with the same content serialize to the same text however
their content was added: `children` by offset, then id; `conversion_maps` by
id; `regions` by start, then end, then id; the `flow_control` elements
(breaks, jumps and markers in one array) by coordinate, a jump's being its
from coordinate, then id. Ids are unique within a manifest, so no two
entries of one list tie on every member. Coordinates compare by their exact numeric value, never by their JSON
spelling, and strings by Unicode code point.

What proves it:

- **Invariance.** One fixed content set is built on three axes, `ticks`
  (`int`), `quarters` (`fraction`) and `seconds` (`float`): four children,
  two of them at the same offset; three conversion maps; four regions, three
  sharing a start and two of those sharing start and end; three breaks, two
  at one coordinate; four jumps, three sharing a from coordinate; four
  markers, two at one coordinate, each jump landing on a marker.
  Every element carries a given id (its kind and label, e.g. `break-fine`),
  because minted ids follow insertion order. For every permutation of
  the insertion order of each collection, the payload must equal the payload
  of the reference insertion order under `==`, and its `json.dumps` text must
  be identical. Equality of dicts ignores key order, so the text comparison is
  what catches a differing `children` or `markers` order. A serializer that
  kept insertion order fails on the first non-identity permutation.
- **The order itself.** Invariance alone would also hold for an arbitrary but
  fixed order, so the expected sequence of child ids, map ids, region ids
  and flow-control element ids is spelled out literally per axis. The values are chosen so that numeric order and JSON-text order
  disagree: on the `ticks` axis `9` precedes `10`, on the `quarters` axis
  `5/8` precedes `2/3` although `[2, 3]` sorts first as text, and on the
  `seconds` axis `9.25` precedes `10.5`. The two markers at one coordinate,
  `marker-Coda` and `marker-coda`, pin code-point order of the id tie-break
  (an uppercase letter precedes every lowercase one).
- **Ties.** Children at one offset fall back to id; regions at one start fall
  back to end, then id; flow-control elements at one coordinate (two breaks,
  a jump and two markers at one position) fall back to their id, whose
  expected order is spelled out from the literal entries.
- **Reading accepts any order.** For every permutation of each collection in
  the canonical payload (the lists reordered; for breaks, jumps and markers,
  the elements of that type moved to the front of the `flow_control` array in
  the permuted order, so that jumps come before the markers they name),
  `Timeline.from_dict` must restore a timeline whose `to_dict` is the
  canonical payload, text-identical.
- **Storage is untouched.** The order is a property of the serialized form
  only: a registry's `breaks` keep their stored sequence (sorted by position,
  ties in insertion order) and `children` keep their insertion order after
  `to_dict`, proven by reading both back after serializing a permutation
  whose stored order differs from the canonical one.

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

The same three-section hierarchy is constructed from nested measure lists, the
counts ``[78, 65, 60]``, and the named counts
``{"I": 78, "II": 65, "III": 60}``. All compare equal because their ordered
leaf counts are exactly ``(78, 65, 60)`` and their total is exactly ``203``;
at three quarters per ``3/4`` bar, the per-leaf quarter spans are exactly
``(234, 195, 180)``. Display names do not alter structure. Building from an existing map must retain
that exact map object. Counts partition it into consecutive normalized slices,
and any count sum other than the map length raises instead of dropping or
inventing measures.

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
`styles` and `statements` beside `order`: presentation data the library keeps
on `TimelineGroup.styling` without interpreting it. What proves it is carried
unchanged is a styling block the library could not have produced itself — a
named style, a statement setting that style by name and a statement setting a
declaration directly, nested three levels deep — surviving `from_dict` →
`to_dict` equal to the input dictionary, beside the unchanged `order`; and the
rebuilt group's `styling` must not share an object with the input, so a later
edit of the input dictionary cannot reach the group. Each member alone
(`styles` only, `statements` only) round-trips alone in the minimal form, so
neither is invented when absent; a group without styling writes `graphical`
as `{"order": {...}}` in the minimal form and adds `styles: {}` and
`statements: []` in the complete one. A `graphical` holding styling but no
`order` reads its members in `members` order, as an absent `graphical` does.
Any other key of `graphical` (`frame`) is kept and written back, never read.
The band orders of `graphical.order`, keyed by a timeline's id, are carried
beside the styling; their tests are those of "Manifests" above.

**Identity.** The default id is `group_1` (minted statelessly: the smallest
free number, so two fresh groups in one process are both `group_1`). A
newcomer whose given id, or any given descendant id, is held by another
timeline of the group's trees is refused with `ValueError` and leaves the
group unchanged; a minted id of the newcomer's tree that is held is minted
again — the newcomer's own (`clt_2`) and a descendant's alike (a minted
`clt_1` child of a newcomer with a given id becomes `clt_2` under its
unchanged parent). A refused newcomer keeps its ids: the collision checks run
before any id of its tree is minted again, so a newcomer whose minted child
would be re-minted but whose given root id is refused still carries its
child's original minted id afterwards. A timeline whose given id is the
group's own given id is refused; a *minted* group id that a newcomer's given
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

`from_dict` refuses each malformed input with a `ValueError` naming the
problem: a missing required key, a member naming no timeline, a timeline id
written twice, an order that is not a permutation of the members, a member
nested with another member, and a group id that equals a timeline id. A root
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

**Group unfolding.** `apply_flow` cuts every member at the same section
labels. The labels name the section children, never identify them: the test
pins the eleven names on each member, requires every descendant id to be a
minted `clt_N`, and requires the group's trees to hold as many distinct ids as
they hold timelines.

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
| `TimelineGroup.apply_flow` with an unknown member | yes | raises at the first boundary converted, before anything is built |
| `append_child` on an unknown parent, `append_segment` after an unknown segment | yes | raises |
| `append_segment` on a line of unknown length, empty or after a known segment | no | appends at `0`, then at the known segment's end; the line stays unknown |
| splitting with `include_after_last=True` | yes | raises; with `False` it works |
| a timeline of unknown length used as a flow interval | yes | raises |
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
region ends, break and marker coordinates, both ends of each jump, and the
end of the measure map. Events are not part of it. The nested example is
computed by hand:

| Timeline | Content | Reach |
|---|---|---|
| `V` (unknown) | marker at `7` | `7` |
| `U` (unknown) | region `[1, 3]`, `V` at offset `5` → `5 + 7 = 12` | `12` |
| `R` (unknown) | child `A` `[2, 6)`; `U` at `10` → `10 + 12 = 22`; region `[0, 15]`; break at `18`; jump `20 → 4` | `22` |

Each content kind also wins once on its own: a lone measure map of five
3/4 bars reaches `15`, a jump `9 → 30` reaches `30`, and a timeline without
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

## Test Files

### `test_coordinate_resolution.py` - Public Coordinate Resolution

These tests pass foreign-unit coordinates through timeline entry points that
read or update events, regions, children, lengths, and slices. Each entry point
must delegate to the same coordinate resolver and preserve its exact missing
C-Map error, demonstrating that coordinate normalization is consistent across
the public timeline surface.

### `test_offset_arithmetic.py` - Nested Offset Arithmetic

These tests validate exact coordinate transfer between parents, children, and
deeper descendants. Parent-to-descendant lookup checks every embedding span,
while descendant-to-parent lookup and `get_coordinate()` compose all offsets
into the ancestor coordinate system. Integer and `Fraction` cases assert exact
results, unknown IDs retain their established return or exception behavior, and
timestamp cross-sections keep their left-inclusive, right-exclusive bounds.

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
9. **Performance Tests** (2 tests)
   - Adding 10,000 events: < 5 seconds
   - Creating 1,000 timelines: < 2 seconds

**Validity Rationale:**

These tests verify the fundamental Timeline contract:
- A Timeline is a positive coordinate axis with an origin (zero) and measuring unit
- Events are stored efficiently in an EventStore (PyArrow-backed)
- Timelines can expand (if unlocked) but cannot contract below content
- Coordinates are validated against the timeline's unit

---

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

`to_dict(*, minimal: bool = False, events: bool = False, external_references:
bool = False)` writes a manifest whose one timeline object carries the flags'
keys. Both keys are **absent** unless requested; requesting `external_references` emits
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
tests assert that exact key set for the default call.

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

6. **Performance Tests** (2 tests)
   - Adding 1,000 children: < 10 seconds
   - Iterating depth-100 hierarchy: < 1 second

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

9. **Child Construction and Boundary Tests**
   - `create_child()` defaults its offset to zero and returns the parent's
     exact concrete class for all six concrete timeline types
   - Coordinate objects remain valid for child lengths and offsets
   - `create_children_from_boundaries()` creates named, contiguous children
     that tile the parent exactly, supports explicit names, and reports the
     same boundary-validation errors as `create_regions_from_boundaries()`

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
   - Allowed units (pixels, meters, cm, mm, inches, points)
   - Similar pattern to other domains

5. **Domain Property Tests** (3 tests)
   - Each domain class returns correct Domain enum

6. **Inheritance Tests** (6 tests)
   - All concrete types inherit from correct base classes

7. **Cross-Domain Compatibility Tests** (3 tests)
   - Logical cannot contain physical
   - Physical cannot contain graphical
   - Graphical cannot contain logical

8. **Parameterized SegmentLine and Serialization Tests**
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
(`instant < extent`).

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

#### `create_beatgrid()` on the structural view

A structure read from a tempo-bearing source carries the grid that generated
its measures, and `create_beatgrid()` returns that same object (identity, not
a rebuild). A structure with no tempo segments raises: measure lengths alone
do not state a tempo, and inventing one would fabricate a fact. Structural
equality includes the grid, so two otherwise identical structures with
different tempi are not equal.

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
   - Real file benchmarks

3. **Score Loader Integration** (2 tests)
   - Note events from score
   - Combined score data

4. **Complex Hierarchy Tests** (2 tests)
   - Multi-level nesting with events at each level
   - Serialization roundtrip of complex hierarchies

5. **Performance Integration** (2 tests)
   - 50,000 events to Timeline: < 5 seconds
   - Real MIDI file loading benchmark

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

## Profiling Results

Performance benchmarks are collected via the `profiler` fixture and reported
at the end of test sessions. Key benchmarks:

| Operation | Target | Actual |
|-----------|--------|--------|
| Add 10,000 events | < 5s | ~0.5s |
| Create 1,000 timelines | < 2s | ~0.1s |
| Add 1,000 children | < 10s | ~3s |
| Iterate depth-100 | < 1s | ~0.01s |
| Timeline from 50k events | < 5s | ~1s |

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
     timeline yields the single whole-span region `mov_1 [0, 100)`. (Distinct
     from `create_child_from_region`, which remains lock-guarded — see the
     create_child_from_region group below.)

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
   - Raises RuntimeError on locked timeline

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
    - get_segment_by_index()
    - get_segment_at() finds segment by coordinate
    - `is_segment_line()` uses exact structural coverage, including
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

### `test_flow.py` - Flow Control and Unfolding

**Purpose:** Validates the Flow API that computes unfolded measure sequences from flow control data (repeats, voltas, D.S., D.C., etc.).

**Architecture (Feb 2026 redesign):**

The Flow API uses a **MeasureUnit-based architecture** with FlowControlElement integration:

| Class | Purpose |
|-------|---------|
| `MeasureUnit` | Fundamental building block with FlowControlElement fields |
| `AtomicSection` | Smallest traversal unit (future: `typed_measures`, `groups`) |
| `PlaythroughSection` | Contiguous atomics in traversal (future: `typed_measures`, `groups`) |
| `Flow` | Sequence of PlaythroughSections, delegates iteration to controller |

**MeasureUnit FlowControlElement Fields:**
- `timesig_duration_qb`: Expected duration from time signature
- `jump_from`: True if MC is a jump origin (D.C., D.S., multiple next)
- `jump_to`: True if MC is a jump target (segno, coda, non-adjacent next target)
- `segno`, `coda`, `fine`, `section_break`: Marker fields
- `flow_control_types`: Tuple of FlowControlElement.value strings for serialization
- `to_dict()` / `from_dict()`: Round-trip serialization support

**Naming Rationale**: Named "Section" (not "Segment") to avoid confusion with TimeToAlign's `Segment` concept (a child timeline contiguous with siblings). These are flow control concepts, not timeline children.

**Interval Convention**: Uses **right-open** `[mc_start, mc_end)` intervals, aligning with:
- TimeToAlign TimeInterval definition (left-inclusive, right-exclusive)
- Python `range()` semantics
- partitura convention

| Old (Inclusive) | New (Right-open) | Meaning |
|-----------------|------------------|---------|
| `mc_start=1, mc_end=5` | `mc_start=1, mc_end=6` | MCs 1,2,3,4,5 (5 MCs) |
| `mc_count = end - start + 1` | `mc_count = end - start` | Direct subtraction |

**Current state (Feb 2026):**

- `MeasureUnit` replaces `FlowStep` as the fundamental building block
- `ScoreFlowController.iter_units()` iterates over MeasureUnits
- `ScoreFlowController.iter_sections(mode=None)` defaults to AtomicSections
- `Flow._controller_ref` links Flow to its controller for `iter_units()` access
- `Flow.steps` removed entirely (section-based only)
- FlowControlElement fields on `MeasureUnit` (jump_from, jump_to, segno, coda, fine, etc.)
- `MeasureUnit.from_dict()` for DataFrame round-trip serialization
- Old `get_atomic_sections()` deprecated with DeprecationWarning
- `iter_mcs()` removed (derive from sections instead)
- `_build_units()` uses helper methods for FlowControl detection
- Typed subclasses (`IncompleteMeasure`, `CompleteMeasure`, `OverlengthMeasure`)
  produced by the Typing step
- MeasureGroup hierarchy (`MeasureGroup`, `SplitMeasure`, `VoltaGroup`,
  `CompleteMeasureGroup`, `IncompleteGroup`, `OverlengthGroup`) produced by the
  Grouping step

**Test Categories:**

1. **Dataclass Tests** (22 tests)
   - `TestMeasureUnit`: Creation, immutability, `to_dict()`, `from_dict()`, round-trip, FlowControlElement fields
   - `TestAtomicSection`: Creation, `mc_range`, `mc_count`, frozen, validation
   - `TestPlaythroughSection`: Creation, `mc_range`, `mc_count`, frozen, `to_mc_sequence()`

2. **Typed Measure Tests** (13 tests)
   - `TestTypedMeasures`: Creation and inheritance of typed subclasses
     - `IncompleteMeasure`: Creation with `position` field, is MeasureUnit subclass
     - `CompleteMeasure`: Creation, is MeasureUnit subclass
     - `OverlengthMeasure`: Creation, is MeasureUnit subclass
     - FlowControlElement field preservation in typed measures
     - `IncompletePosition` enum values
   - `TestAtomicSectionTypedMeasures`: `typed_measures` field on AtomicSection
   - `TestPlaythroughSectionTypedMeasures`: `typed_measures` field on PlaythroughSection

3. **MeasureGroup Tests** (16 tests)
   - `TestMeasureGroup`: MeasureGroup base class and subclasses (12 tests)
     - `MeasureGroup`: Base frozen dataclass with `members`, `mc_start`, `mc_end`, `total_duration_qb`
     - `SplitMeasure`: IncompleteMeasures that sum to time signature
     - `IncompleteGroup`: Isolated IncompleteMeasures (anacrusis, final)
     - `VoltaGroup`: Measures under same volta bracket with `volta_number`
     - `CompleteMeasureGroup`: Adjacent CompleteMeasures
     - `OverlengthGroup`: OverlengthMeasures grouped together
   - `TestBuildGroups`: ScoreFlowController grouping algorithm (4 tests)
      - `_build_groups()`: Grouping-step algorithm
     - `_group_voltas()`, `_group_splits()`, `_group_incompletes()`
     - `_group_overlengths()`, `_group_completes()`
     - `groups` field on AtomicSection and PlaythroughSection

4. **Flow Serialization Tests** (17 tests)
   - `TestFlow`: Empty flow, simple flow, flow with repeats, `to_dataframe()`
   - `TestFlowSectionBased`: section and record constructors, record/CSV exports, equivalence, measure-count and atomic sequences, flow diffs, and unfolded length
   - `TestFlowCSVLoading`: `Flow.from_csv()`, invalid mode raises, `load_valid_flows()`

5. **ScoreFlowController Tests** (6 tests)
   - `iter_units()`: Iterate over MeasureUnits
   - `iter_sections(mode=None)`: AtomicSections by default
   - `get_sections()`: Unified API (replaces get_atomic_sections)
   - `get_sections(mode)`: Returns PlaythroughSections for specified mode
   - `from_atomic_sections()` class method
   - `compute_flow()` populates sections with controller ref

6. **Integration Tests** (10 tests)
   - `TestExample1Rachmaninoff`: No flow control baseline (3 tests)
   - `TestExample3CouperinMusete`: D.S. al Fine (4 tests, all pass)
   - `TestFlowMap`: FlowMap creation
   - `TestPrintedMode`: PRINTED mode returns folded count

**Test Status: 84 passed** (Feb 2026)

**New Methods (Feb 2026):**

- `Flow.to_atomic_sequence()`: Returns flattened list of atomic section IDs from all sections
- `Flow.diff_flows()`: Shows differences between flows using sequence alignment (difflib)

**c05n05_musete Fix (Feb 2026):**

The gold standard CSV was updated to include all 7 section occurrences in playthrough order:

| # | MC Range | Atomic IDs | Notes |
|---|----------|------------|-------|
| 1 | (1-17) | A;B | First refrain |
| 2 | (1-32) | A;B;C | Second refrain + 1e couplet |
| 3 | (17-32) | C | Repeat of 1e couplet |
| 4 | (6-17) | B | D.S. back to segno |
| 5 | (32-59) | D | First 2e couplet |
| 6 | (32-59) | D | Repeat of 2e couplet |
| 7 | (6-17) | B | Final D.S. al Fine |

**Validity Rationale:**

Following the project's ZERO TOLERANCE validation policy, all tests use **exact value comparisons**:
- EXACT section counts (no ranges or minimums)
- EXACT MC ranges: `(mc_start, mc_end)` pairs must match exactly (right-open)
- EXACT section order (positional comparison)

#### Volta-follows-volta invariant (`TestFlowInvariants`)

**Purpose.** Validate `ScoreFlowController.check_invariants()`, a
detect-and-report structural-invariant check over the atomic flow graph. It
never raises; it returns a list of `FlowDiagnostic` describing each violation
and an empty list when the flow is well-formed.

**The invariant.** In the atomic flow graph a volta section can never have a
`to` edge to another volta section. A prima volta's only out-edge is the repeat
back-edge (to the repeat-start, a non-volta section); a seconda volta is
reached only from the repeat-start and continues into the music after the
bracket (also non-volta). Two flow-adjacent voltas therefore indicate a
malformed `next` array — most often a jump target that resolved to the wrong
ending. This is the `to` (flow) edge relation, NOT score-order adjacency: volta
sections ARE naturally adjacent in MC order, which is correct; they must merely
never be connected by a `to` edge. A section "is a volta" iff its first
measure's `volta is not None` — the same criterion the label generator's
volta-flag and the ASCII diagram's `┌N` corner use.

**Fixture — clean score → no diagnostics.** A repeat with a prima/seconda
volta, then a non-volta section, with a section break two measures after the
seconda volta. Eight MCs (4/4 throughout):

| MC | mn | volta | next | section_break | role |
|----|----|-------|------|---------------|------|
| 1 | 1 | None | [2] | — | repeat-start body |
| 2 | 2 | None | [3] | — | body |
| 3 | 3 | None | [4, 5] | — | branch into the two endings |
| 4 | 3 | 1 | [1] | — | prima volta (repeats back to MC 1) |
| 5 | 4 | 2 | [6] | — | seconda volta (continues sequentially) |
| 6 | 5 | None | [7] | — | continuation after the bracket |
| 7 | 6 | None | [8] | yes | continuation, with section break |
| 8 | 7 | None | [-1] | — | final section |

This is a well-formed score. The atomic partition is `A [1,4)` (to A1, A2),
`A1 [4,5)` prima (to A), `A2 [5,8)` seconda (to B), `B [8,9)` final. The seconda
volta absorbs its continuation up to the next genuine boundary (the section
break at MC 7 puts a boundary at MC 8) — the correct behaviour, matching how a
seconda volta continues in real scores (see the design note below). The only
volta sections are A1 (`to=(A,)` → A non-volta) and A2 (`to=(B,)` → B
non-volta), so `controller.check_invariants() == []`.

**Fixture — malformed score → exactly one diagnostic.** A seven-MC fixture that
mis-resolves the prima volta's `next` so it points forward to the seconda volta
(`MC 4`, volta 1, `next=[5]`) instead of back to the repeat-start (`next=[1]`).

| MC | volta | next | resulting section | `to` |
|----|-------|------|-------------------|------|
| 1–3 | None / None / None | [2]/[3]/[4,5] | A `[1,4)` | (A1, A2) |
| 4 | 1 | [5] | A1 `[4,5)` | (A2,) ← violation |
| 5 | 2 | [6] | A2 `[5,8)` | () |
| 6–7 | None / None | [7]/[-1] | (absorbed into A2) | — |

The prima-volta section `A1` then carries `to=(A2,)` where `A2` is the
seconda-volta section. `check_invariants()` returns exactly one
`FlowDiagnostic(kind="volta_follows_volta", section_id="A1", mc=4)`; its message
names both section ids `'A1'`/`'A2'` and hints that the source section's
next/jump target is mis-resolved.

Following the project's ZERO TOLERANCE policy, all assertions are exact:
diagnostic count, the `to` edges, the volta values at each section start, and
the diagnostic's `kind` / `section_id` / `mc` plus the substring identities in
its message.

**Design note — a closing volta does NOT force an atomic boundary.** The
volta-boundary rule adds a boundary at the *onset* of a volta bracket (a
`volta` value change *into* a non-None ending), NOT at the measure where a
bracket closes (`volta n → None`). A seconda (or final) volta therefore
continues into the music that follows it, absorbing it up to the next genuine
boundary — a later jump target or `section_break`. This is intentional and
matches every `.flow.csv` gold standard: e.g. Op.18 No.4 iv's seconda voltas
are the atomic sections `(45, 78)` and `(103, 227)`, each spanning far beyond
the single volta measure. Adding a boundary at the volta-close transition would
split those sections (`(45, 46)` + `(46, 78)`, …), changing the atomic
partition, the section labels (`A, B, C, D, D1, D2, E, F, F1, F2, G, G1, G2`
would gain extra letters), and the unfolded flow — breaking the gold for Op.18
and the segment-naming integration test. The detect-and-report invariant above
is the right home for "this volta arrangement looks wrong": a malformed flow is
surfaced as a `FlowDiagnostic`, not silently re-partitioned.

#### Guarded default-flow traversal (`TestGuardedFlowTraversal`)

**Purpose.** Pin the default-flow traversal as a *guarded transition system*.
At each visited MC a fixed total order over edge kinds decides which eligible
out-edge fires, the traversal reports rather than raises on a flow that cannot
terminate, coda destinations are selected per pass, and repeat-ends resolve
through a nearest-first candidate generator. All fixtures are built from
synthetic measure tables through the public `ScoreFlowController` constructor
(a PyArrow table wrapped in the shared `_MockMeasureData`), so they exercise the
real column-to-`MeasureUnit` pipeline without any specimen file. Every
assertion is exact.

The traversal moves in score (MC) order; a `next` array that points elsewhere
only marks a measure as a jump origin/target and shapes the atomic partition —
it is never followed as the traversal path. The fixtures below are laid out
with that in mind (e.g. a coda destination that must be revisited is placed
*before* its jump origin in MC order).

**Edge-kind total order — repeat back-edge outranks an un-armed coda jump.**
The out-edges at an MC are ranked: (1) an armed Fine stop, (2) an unexhausted
repeat back-edge, (3) an armed coda/segno play-until jump, (4) a D.S./D.C.
jump, (5) the sequential successor. The Fine is a hard stop and outranks a
still-owed repeat (an al-Fine pass ends at the Fine); below it, a section still
owing a repeat plays it before an al-coda exit fires. Fixture (six MCs, 4/4): a
segno block `MC 2` opening an inner
repeat `[MC 2, MC 4]`, a D.S.-al-coda origin at `MC 5` (`jump_bwd=segno`,
`play_until=coda`, `jump_fwd=codab`) that resets the block, and a `codab`
tail at `MC 6`. `MC 4` is both the inner repeat-end and — via the section-end
"to coda" convention, since the score carries no explicit `coda` marker — the
armed trigger point.

- Old behaviour fired the coda jump first, so on the post-jump pass the inner
  repeat was skipped: `[1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 6]`.
- New behaviour runs the owed repeat before the coda exit:
  `[1, 2, 3, 4, 2, 3, 4, 5, 2, 3, 4, 2, 3, 4, 6]` (the post-jump pass now
  contains the inner `2, 3, 4` a second time). This fails on the old code and
  passes on the new.

**Lasso cycle detection — report, never spin.** The old traversal bounded
itself with `len(units) * 30` iterations and then *silently truncated*. It is
replaced by a lasso check over the whole traversal state (position, per-block
pass counts, and armed-jump configuration): revisiting a state means a
guard-disabled edge closes a loop, so the traversal stops and appends a
`FlowDiagnostic(kind="flow_cycle", …)` naming the looping section(s) and the
edge that closes them. Fixture (two MCs): `MC 1 → MC 2`, and `MC 2` is a bare
D.C. (`jump_bwd=start`, `next=[1]`) with no Fine or Coda to ever end it.

- Old behaviour produced 60 rows (`2 * 30`) of `1, 2, 1, 2, …` and *no*
  diagnostic.
- New behaviour stops after `[1, 2]` and reports exactly one
  `flow_cycle` diagnostic (`section_id="A"`, `mc=1`) whose message names the
  `D.S./D.C. jump` edge at `MC 2` re-entering `MC 1`. The test asserts the old
  code emits no such diagnostic and the row count is bounded.

**Scalar coda targets and pass-indexed arming.** `jump_fwd` is one marker name,
including when the name itself contains a comma. A six-MC fixture gives the
destination marker the literal name `codab,varcoda` and asserts that the
al-coda exit reaches it. A second fixture arms the same single-target `codab`
exit twice; both arming passes resolve the same scalar destination, and the
per-origin arming counter remains part of the guarded traversal state.

**Candidate repeat-end resolution.** `_repeat_end_resolution()` maintains the
open explicit repeat-start scopes. Segno and coda markers are jump targets for
their own instructions and never repeat-end candidates; the conventional piece
start is considered only when no explicit repeat start is open and the source
encodes an edge back to the first MC. An end with no open explicit scope and no
backward edge is dangling. Loader-specific one-bar and coda-volta execution
inferences remain outside the registry and cannot become repeat candidates.

- **exactly one explicit scope** → resolve without a diagnostic;
- **several explicit scopes** → resolve to the nearest and record the other
  open scopes in an `ambiguous_repeat_end` diagnostic;
- **no explicit scope, edge to the first MC present** → use the implicit piece
  start;
- **no explicit scope or backward edge** → emit `dangling_repeat_end` and no
  registry jump.

The default traversal and `FlowControlRegistry` both consume this one resolved
repeat-end map. `TestRegistryTraversalAgreement` pins the exact canonical maps
for Musete (`16→1`, `31→17`, `58→32`), Rondeau (`9→1`, `18→10`,
`27→19`, `60→28`), and op18 (`9→1`, `18→10`, `27→19`, `44→28`,
`84→78`, `93→85`, `102→95`), then checks each first executable repeat
edge against its registry target.

`flow_diagnostics(mode)` computes the requested mode and stores traversal
diagnostics separately for each mode. Repeat counters are saturated at their
exhaustion bound before lasso-state comparison, so an exhausted repeat cannot
hide a cycle while long finite chains of legal repeats remain complete.

---

### `test_flow_intervals.py` - Interval-Built FlowMaps

**Purpose:** Validates constructing a `FlowMap` directly from anything that
describes an interval — a singleton or a collection — through the single
polymorphic positional argument of `FlowMap(source)`, and the
`Timeline.create_flow_map(intervals)` convenience that constructs, attaches,
and returns such a map. There are no new `from_*` constructors: one argument
accepts singleton-or-collection via coercion, plus keyword-only options
(`id`, `resolve`), matching `create_regions_from_boundaries(boundaries, *, …)`.

**Motivating case.** A performance that skips score measures 42 and 43. The
child timeline lives in quarterbeat (QB) space and keeps two played spans,
`[0, 123)` and `[129, length)`. Measures 42–43 occupy QB `123–129`; they fall
in the gap and map to nothing. The two played spans concatenate in the
unfolded (target) axis, so QB `129` in the source becomes QB `123` in the
target.

#### Coercion primitives (`_interval_value`, `_as_interval`, `_coerce_intervals`)

The module-private helpers in `timetoalign/timelines/flow/sections.py` are
duck-typed (no hard imports of `Timeline`/`Region`/`Coordinate`). `_as_interval`
recognises descriptors in strict priority order and each variant yields an
**exact** `(Fraction, Fraction, label)` triple. The `label` is a best-effort
identity — a region id for a `str` or a `Region`, the `id` for a `Timeline`,
and `None` for a bare coordinate pair or an id-less event — carried through so
unfolding can identify the child timeline it derives from each span (the
identity, never the display name: a region `A8_1` named `Opening` labels its
span `A8_1`; a timeline `clt_1` named `Score` labels its span `clt_1`):

| Input | Recognised as | Expected `(start, end, label)` |
|-------|---------------|--------------------------------|
| `(0, 123)` (raw ints) | 2-element sequence | `(Fraction(0), Fraction(123), None)` |
| `(Coordinate(0), Coordinate(129))` | 2-element sequence | `(Fraction(0), Fraction(129), None)` |
| `Region(0, 123, id="A8_1", name="Opening")` | has `.start`/`.end` | `(Fraction(0), Fraction(123), "A8_1")` |
| a `Timeline` of length 200, id `clt_1`, name `Score` | has `.origin`/`.length` | `(Fraction(0), Fraction(200), "clt_1")` |
| `Note(start=0, end=123)` interval event | has `.start`/`.end` | `(Fraction(0), Fraction(123), None)` |
| `"A8_1"` with a `resolve` callable | `str` → resolved region | `(Fraction(0), Fraction(123), "A8_1")` |
| `"A8_1"` with `resolve=None` | `str`, no resolver | raises `ValueError` |

`_coerce_intervals` disambiguates singleton-vs-collection by trying the whole
spec as one interval first, and only iterating on failure:

- A `str` is **always** a singleton — never iterated character by character
  (`_coerce_intervals("A8_1", resolve=…)` → one interval, not five).
- `[0, 123]` is one coordinate pair (singleton), yielding one interval.
- `[region1, region2]` and `["A8_1", "A8_2"]` are collections, yielding two
  intervals each (the singleton attempt fails because the first element is not
  a coordinate value, so the fallback iterates).
- `[(0, 123), (129, 200)]` yields two intervals.
- Any interval with `end < start` raises `ValueError`.

All assertions are exact `Fraction` equality — no ranges, no `pytest.approx`.

#### Interval-built `FlowMap`

Sections are built directly from the coerced `(start, end, label)` ranges with
a **cumulative target position** (identical math to `from_qb_sections`), each
section carrying its `label`, and `flow` is `None`. For the two motivating
spans `[0, 123)` and `[129, 252)` (length 252):

| Query | Method | Exact result | Derivation |
|-------|--------|--------------|------------|
| `unfold_coordinate(50)` | source → target | `[Fraction(50)]` | span 1: `0 + (50-0)` |
| `unfold_coordinate(125)` | source → target (gap) | `[]` | QB 125 is in the skipped gap |
| `unfold_coordinate(150)` | source → target | `[Fraction(144)]` | span 2: `123 + (150-129)` |
| `fold(144)` | target → source | `Fraction(150)` | inverse of the above |
| `fold(50)` | target → source | `Fraction(50)` | span 1 |
| `total_target_length` | — | `Fraction(246)` | `123 + (252-129)` |
| `n_sections` | — | `2` | two played spans |

`inverse()` round-trips (`inverse().unfold_coordinate(144) == [Fraction(150)]`) and does
not require a non-`None` `flow`. `__repr__` reads the `id`, so an interval map
built with `id="A8"` reprs as `FlowMap(A8: 2 sections)` and its inverse as
`FlowMap(A8_inverse: 2 sections)`.

#### Regression — the `Flow` and `from_qb_sections` paths are unchanged

`FlowMap(flow_obj)` still builds MC-space sections from `flow.sections`
(`unfold_coordinate(3)` over two `[1, 5)` playthrough sections gives
`[Fraction(2), Fraction(6)]`, since MC 3 is visited twice), `id` defaults to
`flow.mode.value`, and `from_qb_sections` is untouched. The polymorphic
positional parameter is named `source`, so `FlowMap(source=flow_obj)` works by
keyword; the historical `FlowMap(flow=flow_obj)` keyword no longer exists (it
raises `TypeError`) — every construction site was migrated to positional and no
compatibility alias is retained.

#### `Timeline.create_flow_map`

`create_flow_map(intervals, *, id="default")` constructs the FlowMap
(resolving region-name strings via `self.get_region`), attaches it under `id`,
and returns it — the `create` verb×noun contract. The four equivalent target
calls all build the same two-span map on a length-252 child:
`create_flow_map(["A8_1", "A8_2"], id="A8")`,
`create_flow_map([region1, region2], id="A8")`,
`create_flow_map([(0, 123), (129, length)], id="A8")`, and the singleton
`create_flow_map(region1, id="X")`. After construction `get_flow_map(id)`
returns the same object, and the timeline's `unfold_coordinate`/`fold`
convenience methods delegate with the exact values above
(`tl.unfold_coordinate(150, "A8") == [144.0]`,
`tl.unfold_coordinate(125, "A8") == []`, `tl.fold(144, "A8") == 150.0`).

#### `Timeline.apply_flow` — the unfolded timeline

`unfold_coordinate` maps a single folded coordinate to a list of unfolded
coordinates; `apply_flow(id)` instead yields the whole unfolded **timeline**.
It slices the source at each section of the attached FlowMap via
`Timeline.get_slice` (which rebases each slice by `-start`) and **appends** the
slices, in target order, as children of a new timeline of the source's **same
concrete type**, whose child offsets accumulate as `0` then `123`. Each section
also becomes a matching named Region in unfolded (target) coordinates. Section
events live in the appended children; the flattened coordinates remain reachable
via `get_events(include_children=True)`. The result carries a reverse FlowMap
(id `"source"`) and the forward FlowMap (id `f"forward_{id}"`). Each child and
Region takes the section's name (a repeat is suffixed `-rend2`, `-rend3`, …).

The fixture is a length-252 source with played spans `A8_1` `[0, 123)` and
`A8_2` `[129, 252)`, instants `e10` at absolute 10 (span 1) and `e140` at
absolute 140 (span 2), and a nested child at offset 40 (span `[40, 60)`, inside
span 1) carrying instant `c5` at local 5. `r = source.apply_flow("A8")` gives
the exact values:

| Query | Exact result | Derivation |
|-------|--------------|------------|
| `type(r) is ContinuousLogicalTimeline` | `True` | same concrete type as source |
| `r.length.value` | `Fraction(246)` | `123 + (252 - 129)` |
| `r.n_children` | `2` | one appended child per played span |
| `r.list_children()` | `["A8_1", "A8_2"]` | children named after the spans |
| child `A8_1` / `A8_2` offset | `Fraction(0)` / `Fraction(123)` | cumulative span lengths |
| child `A8_1` / `A8_2` length | `Fraction(123)` / `Fraction(123)` | span lengths |
| `e10` in child `A8_1` | `Fraction(10)` | slice `[0, 123)`: `10 - 0` |
| `e140` in child `A8_2` | `Fraction(11)` | slice `[129, 252)`: `140 - 129` |
| `A8_1` sub-child offset | `Fraction(40)` | `40 - 0`; `A8_2` has no sub-children |
| sub-child `c5` local coord | `Fraction(5)` | slice `[0, 20)`: `5 - 0` |
| `r.list_regions()` | `["A8_1", "A8_2"]` | one Region per section |
| region `A8_1` / `A8_2` | `[0, 123)` / `[123, 246)` | contiguous target coords |
| `r.get_flow_map("source")` labels | `["A8_1", "A8_2"]` | reverse FlowMap carries labels |
| `r.has_flow_map("forward_A8")` | `True` | forward FlowMap attached |
| `get_events(include_children=True)` | `e10@10`, `c5@45`, `e140@134` (each once) | flattened parent-space coords |
| `["A8_1", "A8_1"]` flow → children/regions | `["A8_1", "A8_1-rend2"]` | repeat gets a `-rend2` suffix |
| `source.apply_flow("nope")` | raises `ValueError` | no FlowMap attached with that id |

---

### `test_segment_naming.py` - Customizable Atomic-Section Labelling

**Purpose:** Validates `SegmentNameGenerator`, the strategy object that the
`ScoreFlowController` uses to label atomic sections, and its integration into
the controller's section-building pass.

**What the generator does.** `generate(volta_flags)` turns a list of
per-section booleans (`volta_flags[i]` is `True` when atomic section `i` opens
a volta bracket) into one label per section. Two policies are configurable at
construction:

- **Alphabet** (`alphabet=`, default `_SECTION_ALPHABET`): base sections walk
  the alphabet; once exhausted it repeats with a numeric suffix (`A2`, `B2`,
  …). The generator never reaches into punctuation or control characters.
- **Volta suffix** (`volta_suffix=`, default `True`): a section that opens a
  volta bracket inherits the preceding base section's label plus a *positional*
  numeric suffix (`1`, `2`, …) — section `B` followed by two endings reads `B,
  B1, B2`, not `B, C, D`. The suffix is positional and independent of the
  volta's own ending number. A non-volta section resets the counter, so two
  independent volta groups read `B, B1, B2` then `C, C1, C2` (never `C3, C4`).
  A leading volta with no preceding base falls back to a base letter (never
  `None1`). With `volta_suffix=False` every section consumes the next base
  label, the historical pure-sequential behaviour.

**Generator unit tests (exact strings):**

| Case | `volta_flags` | Generator config | Expected labels |
|------|---------------|------------------|-----------------|
| Default sequential | all `False` | default | `A, B, C` |
| Alphabet overflow | one flag beyond alphabet length | tiny `alphabet=` | tail label carries the `2` suffix (`…, A2`) |
| Volta suffix | `[False, True, True]` after a base | default | `A, A1, A2` |
| Legacy sequential | `[False, True, True]` | `volta_suffix=False` | `A, B, C` |
| Custom alphabet | all `False` | `alphabet=["X", "Y", "Z"]` | `X, Y, Z` |
| First-section volta | `[True, False, True]` | default | `A, B, B1` (no `None1`) |
| Two volta groups | `[False, True, True, False, True]` | default | `A, A1, A2, B, B1` (counter resets) |

**Integration test (Op.18 No.4 iv specimen).** The folded measures TSV at
`beethoven_op18-4iv_multimodal/op18_no4_mov4_flow/` carries three volta groups.
Building a `ScoreFlowController` over it and reading the atomic sections must
yield, in order:

```
A, B, C, D, D1, D2, E, F, F1, F2, G, G1, G2
```

(13 sections — count unchanged from the pre-volta-suffix labelling). The test
asserts EXACT section ids AND that every `to[]` graph edge references those new
labels (the edges are derived from the same label list, so e.g. `D` points to
`('D1', 'D2')` and `D1` back to `('D',)`). It also confirms section count and
MC ranges are untouched by the relabelling. Passing `volta_suffix=False` to the
controller reproduces the legacy sequential ids `A, B, … M`.

Following the project's ZERO TOLERANCE policy, all assertions are exact string
and exact count comparisons; the data path is resolved relative to the test
file (the pattern already used by `test_unfolding.py` for this specimen).

---

### `test_timestamps.py` - Cross-Section Timestamp Tables

**Purpose:** Validates `get_timestamp_table()` / `to_dataframe()` and the
helpers that build the timestamp axis (event-coordinate collection, boundary
collection, local-coordinate computation) across a timeline hierarchy.

**Fraction-length regression (`TestFractionLengthTimestamps`):**

A logical timeline (quarters/beats) carries `number_type = Fraction`, so its
length is a `Coordinate` whose `value` is a `Fraction`. The bounds check inside
`_compute_local_coordinates` masks out-of-range coordinates by comparing the
local-coordinate array against the length scalar through a PyArrow compute
kernel (`pc.greater`). That kernel rejects a raw `Fraction` argument, so the
length must be coerced to `float` at the kernel boundary — consistent with the
sibling `pc.less(local, 0.0)` and the float64 arrays this internal helper
already operates on. Empirically, only the `pc.greater` comparison broke: the
boundary collector builds a `pa.array(..., type=pa.float64())`, which coerces a
`Fraction` via `__float__`, and the child-offset accumulation starts from a
`float` default so it never reaches a kernel as a raw `Fraction`.

| Test | Construction | Exact assertion |
|------|--------------|-----------------|
| `test_compute_local_coordinates_with_fraction_length` | CLT length `9/2`, probe `[0.0, 2.0, 4.5, 5.0]` | local = `[0.0, 2.0, 4.5, None]` (5.0 > 4.5 → null) |
| `test_get_timestamp_table_with_fraction_length` | CLT length `9/2`, events at `0, 3/2, 3` | `num_rows == 3`; axis = `[0.0, 1.5, 3.0]`; root-local = `[0.0, 1.5, 3.0]` |
| `test_to_dataframe_with_fraction_length` | same timeline | shape `(3, 2)`; axis column = `[Fraction(0), Fraction(3,2), Fraction(3)]` (auto-rendered as Fractions for a Fraction-type timeline) |

**Validity Rationale:** The timestamp axis is the spine of every cross-domain
alignment view. A quarters/beats timeline — the most common logical timeline —
must produce timestamp tables and frames without raising. The asserted values
are the exact in-bounds coordinates: the out-of-bounds probe coordinate becomes
`null`, the event coordinates pass through unchanged, and the root timeline's
local coordinates equal the axis because its offset is zero.

---

### `test_timestamps.py` - Cross-Section Timestamp Tables

**Purpose:** Validates `get_timestamp_table()` / `to_dataframe()` and the
helpers that build the timestamp axis (event-coordinate collection, boundary
collection, local-coordinate computation) across a timeline hierarchy.

**Fraction-length regression (`TestFractionLengthTimestamps`):**

A logical timeline (quarters/beats) carries `number_type = Fraction`, so its
length is a `Coordinate` whose `value` is a `Fraction`. The bounds check inside
`_compute_local_coordinates` masks out-of-range coordinates by comparing the
local-coordinate array against the length scalar through a PyArrow compute
kernel (`pc.greater`). That kernel rejects a raw `Fraction` argument, so the
length must be coerced to `float` at the kernel boundary — consistent with the
sibling `pc.less(local, 0.0)` and the float64 arrays this internal helper
already operates on. Empirically, only the `pc.greater` comparison broke: the
boundary collector builds a `pa.array(..., type=pa.float64())`, which coerces a
`Fraction` via `__float__`, and the child-offset accumulation starts from a
`float` default so it never reaches a kernel as a raw `Fraction`.

| Test | Construction | Exact assertion |
|------|--------------|-----------------|
| `test_compute_local_coordinates_with_fraction_length` | CLT length `9/2`, probe `[0.0, 2.0, 4.5, 5.0]` | local = `[0.0, 2.0, 4.5, None]` (5.0 > 4.5 → null) |
| `test_get_timestamp_table_with_fraction_length` | CLT length `9/2`, events at `0, 3/2, 3` | `num_rows == 3`; axis = `[0.0, 1.5, 3.0]`; root-local = `[0.0, 1.5, 3.0]` |
| `test_to_dataframe_with_fraction_length` | same timeline | shape `(3, 2)`; axis column = `[Fraction(0), Fraction(3,2), Fraction(3)]` (auto-rendered as Fractions for a Fraction-type timeline) |

**Validity Rationale:** The timestamp axis is the spine of every cross-domain
alignment view. A quarters/beats timeline — the most common logical timeline —
must produce timestamp tables and frames without raising. The asserted values
are the exact in-bounds coordinates: the out-of-bounds probe coordinate becomes
`null`, the event coordinates pass through unchanged, and the root timeline's
local coordinates equal the axis because its offset is zero.

---

### `test_unfolding.py` - Unfolding via Slicing

**Purpose:** Validates the slice-based unfolding pipeline that replaces the buggy MC-space
FlowMap approach with structural slicing in QB-space.

**Test classes** validating the pipeline:

| Class | Purpose |
|-------|---------|
| `TestGetSlice` | Unit tests for `Timeline.get_slice()` primitive |
| `TestComputeQBSections` | QB boundary computation from Flow + ScoreFlowController |
| `TestSegmentLineAssembly` | Integration: slice + concatenate into a SegmentLine (this module's own manual helper) |
| `TestCreateUnfoldedTimelineIdentity` | Explicit uid on the same-type unfolded timeline |
| `TestUnfoldingGoldStandard` | End-to-end validation against 7 ms3 gold standard specimens |
| `TestGroupUnfolding` | Cross-domain unfolding through the group's `TimeStamp`s, which convert linearly between members |

**Testing Pyramid:**

```
    ┌─────────────────────────────────┐
    │  Group Unfolding Tests          │  ← 3 timelines × 1 FlowMap,
    │  (TestGroupUnfolding)           │     cross-domain (Logical+Graphical)
    ├─────────────────────────────────┤
    │  Gold Standard Tests            │  ← 7 specimens, EXACT match
    │  (TestUnfoldingGoldStandard)    │     against ms3 TSV files
    ├─────────────────────────────────┤
    │  Integration Tests              │  ← SegmentLine assembly from
    │  (TestSegmentLineAssembly)      │     slices, contiguity, length
    ├─────────────────────────────────┤
    │  Unit Tests                     │  ← get_slice() primitive
    │  (TestGetSlice +                │     compute_qb_sections() helper
    │   TestComputeQBSections)        │
    └─────────────────────────────────┘
```

**Test Categories:**

1. **get_slice() Unit Tests** (16 tests)
   - Basic slicing with coordinate shifting (`[start, end)` semantics)
   - Interval event truncation (`truncate_events=True/False`)
   - Child timeline recursive slicing
   - Number type preservation (Fraction stays Fraction)
   - Concrete class preservation (CLT returns CLT)
   - Parameterized SegmentLine slicing preserves its dynamic class
   - Discrete and physical timeline support
   - Error cases (invalid range, out of bounds)

2. **compute_qb_sections() Unit Tests** (8 tests)
   - Sequential score (Rachmaninoff): single pass, total QB = 2997/2
   - D.S. al Fine (Musete): repeated sections, total QB = 384
   - Rondeau form, volta brackets, split bars, D.S./D.C.
   - QB boundaries match folded TSV `quarterbeats` column exactly
   - All sections have strictly positive duration

3. **SegmentLine Assembly Integration Tests** (5 tests)
   - Contiguity enforcement (each segment starts where previous ended)
   - Total length = sum of slice lengths
   - Events preserved in assembled segments
   - Segment type matches source class
   - The assembled line is an instance of both SegmentLine and the source class
   - Repeated section assembly (same range played twice)

4. **Standalone Unfolded Timeline Identity Tests** (2 tests)
   - Explicit `uid` is retained by flat timeline results
   - Explicit `uid` is retained by SegmentLine results

5. **Gold Standard End-to-End Tests** (49 tests, parametrized × 7 specimens)
   - EXACT row count, MC sequence, mc_playthrough, mn_playthrough with suffixes
   - EXACT quarterbeats as Fraction (not float), total unfolded length

6. **Group Unfolding Tests** (10 tests)
   - Uses Beethoven Op.18 No.4 iv multimodal score group (CLT1 + DGT1 + OpenScore)
   - Validates that one FlowMap unfolds ALL timelines regardless of domain
   - Resolves section boundaries through the group's `TimeStamp`s (members convert linearly between their common start and end)
   - Tests: segment counts (11), contiguity, type preservation, length consistency

**Gold Standard Exact Values:**

| Specimen | Folded | Unfolded | Total QB | Challenge |
|----------|--------|----------|----------|-----------|
| rachmaninoff | 374 | 374 | 2997/2 | No flow control (baseline) |
| polyrhythm_only | 14 | 14 | 45 | Line breaks only |
| musete | 58 | 138 | 384 | D.S. al Fine, anacrusis, 6/8 |
| rondeau | 60 | 138 | 195 | Rondeau form (D.S.) |
| op18_no4_mov4 | 226 | 291 | 1116 | Repeats + volta brackets |
| woo71 | 397 | 505 | 1078 | Complex split bars |
| flow_only | 15 | 30 | 75 | D.S./D.C. + voltas |

**Known Edge Cases:**
- **Anacrusis** (Musete): Incomplete first measure (1.5 QB instead of 3.0). Repeated instance preserves short duration.
- **Split bars** (WoO71): Single notated measure divided into two MCs. Preserved exactly through unfolding.
- **Volta brackets** (Op.18): Different MCs played on different passes. Correct volta selected per pass.
- **Rondeau form** (c11n08): ABACADA return pattern via D.S.-like mechanism.

**Bug Fixes Applied:**
- `Ms3Loader._resolve_quarterbeats()`: Prefers `quarterbeats_all_endings` over `quarterbeats` (fixes 10.5q error for volta first-endings).
- `SegmentLine.get_slice()`: Override creates with `length=0` and sets final length after children, fixing contiguity validation.

**Coordinate Handling:** EventData stores instant events under `start` (struct dict),
not `instant`. Use `struct_to_coordinate()` for reading; never cast to `float`
unnecessarily (loses Fraction precision).

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

`test_types.py` resolves continuous, discrete, string-unit, and unmatched-unit
inputs to the most specific matching classes. Further-derived classes also
participate in resolution. The unmatched input verifies the documented base
`Timeline` fallback when no registered subclass accepts a unit.

All tests follow these principles:

1. **Positive Tests**: Verify correct behavior under normal conditions
2. **Negative Tests**: Verify errors are raised for invalid input
3. **Boundary Tests**: Test edge cases (empty, zero, max values)
4. **Performance Tests**: Ensure operations complete within time budgets
5. **Roundtrip Tests**: Verify serialization preserves data

Each test class includes a docstring explaining its validity rationale,
linking back to specific requirements from the TimeToAlign model and the
project's engineering standards.
