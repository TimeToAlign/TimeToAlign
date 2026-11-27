# Storage Tests

This directory contains tests for `timetoalign.storage`, the PyArrow-backed
event storage layer (`EventData`, `EventStore` and its subclasses).

## Fraction Fidelity Validation

Interval completion is checked with exact rational expectations. Tests cover
both row-oriented `from_dicts` construction and the vectorized `from_arrays`
path, including addition and subtraction results such as `7/2 + 3/4 = 17/4`.

Every non-null coordinate cell carries the number on **both** sides of the
struct — a float64 and an integer ratio — so the tests assert both, plus the
agreement between them. There is no longer any such thing as a "null pair
signals float": which side is authoritative is declared once in the field's
`number_type` metadata, so a computed `duration` in a float-typed column has a
float canonical side and an exact dyadic mirror, and one in a fraction-typed
column has the reverse. Asserting only the side a caller happens to read would
miss a cell whose two halves disagree.

### Interval completion runs in the axis's number type

Completion fills `duration = end − start` where a row has no duration (or the
policy recomputes it) and `end = start + duration` where it has no end; the
tabular loader's `duration_column` takes the second form. A row without a
`start` is not completed. The arithmetic runs in the number type the table
declares and nowhere else: a float axis subtracts its doubles and mirrors the
result once, an exact axis (`int`, `fraction`) computes on its ratios with
arithmetic that cannot overflow. What proves it, in `test_events.py`
(`TestIntervalCompletion`):

- **The struct invariant on every completed cell.** For every completed cell:
  `denominator > 0`; on a float axis `Fraction(numerator, denominator) ==
  Fraction(value)` exactly (the ratio is the exact mirror of the stored
  double, never a ratio computed beside it); on an exact axis `value ==
  float(Fraction(numerator, denominator))`.
- **Float axes give the float result.** Seconds with three decimals, the shape
  of every real annotation file: the completed `value` is the IEEE result of
  the operation on the stored doubles, stated as a literal
  (`2.718 - 1.234 == 1.484`, `3.1 - 2.718 == 0.3820000000000001`, `1.234 +
  1.484 == 2.718`), on both construction paths. Two rows span magnitudes,
  `1.5 - 0.003 == 1.497` and `123.457 - 0.001 == 123.45599999999999`: the
  exact difference of their dyadic mirrors needs more than 53 bits, so a
  completion that computed it would store a ratio that is not the mirror of
  its double (the first) or fail on a numerator past int64 (the second).
- **Scale.** 10,000 rows of three-decimal seconds (seeded generator): zero
  cells with a non-positive denominator, every ratio the mirror of its value,
  every value equal to `end − start` computed by numpy on the same doubles,
  and `to_parquet` / `from_parquet` return identical cells.
- **Exact axes beyond int64 cross-products.** On quarters, `end =
  (3·2⁴⁰ + 1)/(3·2⁴⁰)` and `start = 1/(5·2⁴⁰)`:
  the denominators' product is past `2⁶³`, the result is not:
  `duration == Fraction(15·2³⁹ + 1, 15·2³⁹)` exactly, both directions (`end`
  from `start + duration` gives back the `end`). On ticks (`int`) the result is
  the integer difference / sum with denominator 1.
- **Consistency is decided in the axis's number type.** A row stating `start`,
  `end` and `duration` is consistent when `duration == end − start`. On an
  exact axis the test is exact, so a near miss no float tolerance would see is
  caught: on quarters, `start = 0`, `end = 1/3`, `duration = 1/3 + 1/2⁴⁰`
  (a twelve-digit decimal duration beside an exact end has this shape). Under
  `warn` the stored duration is `1/3` and one warning names the row; under
  `strict` construction raises `Interval inconsistency`; under `prefer_end`
  the duration is `1/3` without a warning; under `prefer_duration` the
  duration stays `1/3 + 1/2⁴⁰` and `end` becomes `1/3 + 1/2⁴⁰`. Every case runs
  on both paths with the same cells. On ticks the same holds for `start = 0`,
  `end = 4`, `duration = 3` (`strict` raises, `warn` stores `4`). On a float
  axis the comparison is a float comparison within `1e-9`: seconds `start =
  0.1`, `end = 0.3`, `duration = 0.2` (whose double difference is
  `0.19999999999999998`) is consistent under `strict` and keeps `0.2`.
- **Exact axes compute exactly on both paths.** The column path computes an
  `int` or `fraction` axis on integers, never on doubles, and no
  intermediate product can overflow. Ticks `start = 2⁵³ + 1`, `end = 2⁵³ + 4`
  give `duration = 3` on both paths, and `start = 1`, `duration = 2⁵³ − 1`
  give `end = 2⁵³` (numerator `9007199254740992`). On quarters, float ends
  and starts whose exact dyadic ratios cross-multiply past `2⁶³`
  (`end = 0.3`, `start = 0.1`; `end = 0.9`, `start = 0.7`) give the exact
  durations `7205759403792793/2⁵⁵` and `1801439850948199/2⁵³` on both paths,
  with the floats `0.19999999999999998` and `0.20000000000000007` — no
  tolerance, no rounded difference. Only a result whose lowest terms do not
  fit int64 is refused: quarters `start = [0, 1129/12]`,
  `duration = [1, 0.1]` through `from_arrays` raises a `ValueError` whose
  whole message is asserted — it names the computation and the axis
  (`Completing the events' end on a fraction axis`), the row (`row 1`, the
  row of the table, not of the rows being completed), the operands and the
  reduced result.
- **One rule for both construction paths.** The same rows through `from_dicts`
  (row path) and `from_arrays` (column path) give identical `end` and
  `duration` cells, on float, int and fraction axes.
- **The tabular loader.** Covered in `tests/loader/tabular/README.md`
  (LabLoader) and `test_struct_columns.py`.

The interval-normalisation regression also checks the explicit temporal type:
an instant row may have a populated `start` coordinate, but its `end` and
`duration` cells remain null — genuinely null rows are the only place a null
sub-field appears. Interval rows still receive both coordinates, and their
generated pairs are checked with Fraction arithmetic. This is the same
vectorized path used while assembling merged score tables.

`EventData.from_dicts` also normalizes integer-valued float ratio members in
canonical coordinate-shaped dictionaries to Arrow `int64` children. This
applies to both base coordinates and carried extra coordinate structs, so the
library does not reproduce the legacy float-member representation it accepts
when reading older artifacts. Fractional float ratio members remain invalid.

A cell that carries only a `value` — hand-built, or written before both sides
were mandatory — still decodes: the float side is the only place the number
lives, and its own exact ratio is the honest answer. Tests cover that path
explicitly, because it is what keeps older artifacts and hand-written fixtures
readable without reintroducing the guess-from-the-row habit.

That read tolerance is only safe while nothing in the library *writes* a
partial cell — otherwise a writer bug would emit them indefinitely and the
tolerant reader would absorb them in silence. So the invariant is asserted on
the writing side too: `tests/core/test_number_storage.py` walks the builder's
output across every representation and input shape and asserts no non-null row
has a null member, and does the same for `Coordinate.to_dict()` /
`Duration.to_dict()`. Tolerance on read, strictness on write.

## Test Files

### `test_events.py` - `EventData.create_timeline()`

`EventData.create_timeline()` selects the concrete timeline class from the
data unit and number type, places the selected rows directly on that timeline,
and applies EventData filters before assignment. The tests pin the logical
timeline class and the exact unfiltered and filtered event counts.

### `test_events.py` - A measure's volta is carried as stated

A table carries a measure's `volta` in two columns, so that a number and a
list stay apart: `volta` (`int64`) holds the number, `volta_list`
(`list<int64>`) the list of a bracket stating several passes. Rows giving
`volta` as `2`, `[1, 3]`, `[1]` and `None` make the columns `[2, None,
None, None]` and `[None, [1, 3], [1], None]`. Both columns have their type
whether a table's rows hold a value or none, so a table whose rows state no
bracket joins one whose rows do: extended by a table holding `[1, 3]`, a
table of one row without bracket reads `[None, [1, 3]]` in `volta_list`.
A table without the list column reads as before: `volta` alone, an
`int64` column of `[None, 2]` (the type `tests/timelines/test_groups.py`
pins).

### `test_events.py` - `EventData.column_values()`

**Purpose:** Documents and validates the public, decoded column-read
affordance that lets callers outside `timetoalign.storage` read a column's
values without reaching into `EventData`'s private `_table`.

**Contract:**

- `column_values(name, *, default=None)` returns one plain Python value per
  event (row), in row order.
- A **rational-shaped** column -- the canonical coordinate struct
  `{value, numerator, denominator}` used by `start`, `end`, `duration`, and
  any extra field sharing that shape -- decodes to an exact `Fraction` via
  `timetoalign.core.struct_to_rational`.
- Integer-valued float ratio members from earlier persisted artifacts decode
  losslessly, including through `to_dataframe()`; fractional members raise
  rather than being rounded.
- When a row's struct carries no exact ratio (missing or non-integral
  `numerator`/`denominator`), the float `value` member is used instead,
  wrapped in a `Fraction`.
- A **null coordinate struct cell** yields `default`.
- Every other (non rational-shaped) column type is returned via
  `to_pylist()` unchanged -- ints, strings, bools, lists pass through
  as-is, including `None` for a null cell (`default` does not apply here;
  it only substitutes for a null coordinate struct or an absent column).
- When `name` is not a column of the table at all, the method returns
  `[default] * len(self)` rather than raising `KeyError`.

**Test Categories:**
- **Rational decoding**: a `duration` column with exact numerator/
  denominator decodes to the exact `Fraction`, not a rounded float.
- **Fallback decoding**: a coordinate struct with only `value` populated
  (no numerator/denominator) falls back to `Fraction(value)`.
- **Null coordinate handling**: a null coordinate-struct cell yields the
  caller-supplied `default`.
- **Null plain-column handling**: a null cell in a non-coordinate column
  passes through as `None`, unaffected by `default`.
- **Missing column**: requesting an absent column name returns a
  `default`-filled list sized to the event count, not an exception.
- **Plain passthrough**: non-coordinate columns (e.g. `event_type`) are
  returned unchanged from `to_pylist()`.

### `test_events.py` - `EventData.head()`

**Purpose:** Validates the pandas-style leading-row preview returned by
`EventData.head()`.

**Contract:**

- `head(n=5)` returns the first `n` events as a pandas DataFrame, using the
  same coordinate conversion as `to_dataframe()` (native numbers, not raw
  struct dicts).
- `n` larger than the event count returns every event; `n <= 0` returns an
  empty frame.
- `head(n)` equals the leading `n` rows of `to_dataframe()` (checked with
  `pandas.testing.assert_frame_equal`).

**Test Categories:**
- **Default arity**: `head()` returns exactly five rows.
- **Explicit arity**: `head(n)` returns `n` rows with exact `Fraction`
  coordinates.
- **Over-count / non-positive `n`**: saturates to the full table / yields
  an empty frame.
- **Conversion equivalence**: `head(n)` matches `to_dataframe().head(n)`.
- **Numeric rendering**: coordinate columns decode to `Fraction`, never
  raw struct dicts.

### `test_events.py` - Minted event IDs, Projections as rows and the declared `projection_of` column

**Purpose:** Every path that builds an `EventData` mints a missing event ID
through `mint_event_ids`, which draws from the library's random-ID minter
(`core/ids.py::IdMinter`), a Projection object stands for its event row, and
a Projection's reference column is typed by declaration rather than by the
rows of one batch.

**Contract and what proves it** (`TestMintedIds`). A minted ID is only
proven by the minter that drew it, so each expected ID below is computed by
an `IdMinter` of the same seed in the order the contract states, never
typed from a run: a test that restated the output of the function under
test would prove nothing.

- `mint_event_ids(event_types, minter)` (public, also exported from
  `timetoalign.storage`, the same function) mints one ID per type: the
  prefix is the event type in lower case (`event` for a missing or empty
  type), the suffix drawn from *minter*, one batch per prefix in the order
  of each prefix's first row, the rows of a prefix in row order. So
  `["Note", "Projection", "Note", None, ""]` with `IdMinter(5)` gives the
  first two IDs of a second `IdMinter(5)` (minted under `note`) for rows 0
  and 2, its third (`projection`) for row 1, and its fourth and fifth
  (`event`) for rows 3 and 4. Without a minter the IDs come from the content
  minter of the event types: two calls with equal types in one test give
  distinct IDs, and after `reset_content_minters()` the first call's IDs
  again (the process replays).
- `EventData.from_dicts` and `from_arrays` keep a given ID and mint the
  missing ones: with `seed=3`, the rows without an ID hold exactly the IDs
  `mint_event_ids` draws from `IdMinter(3)` for their event types, in row
  order of the missing rows; with `minter=` they continue that minter's
  counter (a loader's one minter: the second table starts where the first
  stopped); two calls with one seed give equal tables; two calls without
  either give distinct IDs. `from_arrays` mints both for an absent `id`
  column and for a null in a given one. `from_dataframe` passes `seed=` on.
  Giving both `seed` and `minter` raises `TypeError`. No table built by a
  constructor holds a null ID.
- `EventData.from_dicts` takes a `Projection` object, alone or in the list
  beside dicts, as its event row: `Projection(start=Coordinate(2, samples),
  duration=Duration(3, samples), projection_of=EventReference(clt_6, x1),
  id=p1)` and a Note dict with the ID `n1` give the IDs `p1`, `n1`, the event type
  `Projection`, the cells `{2.0, 2, 1}` and `{3.0, 3, 1}` and the reference
  as its two values; a Projection given its ID keeps it. A position given as
  a `Coordinate` in the table's unit is written in the table's number type
  (`Coordinate(1.5, quarters)` on a fraction table: the cell `{1.5, 3, 2}`);
  one in another unit is refused with a message pointing to a timeline,
  which converts.
- A batch whose rows hold `projection_of` only as `None` still has the
  column typed `struct<timeline_id: string, event_id: string>` with the
  `EventReferenceField` field type in its metadata, so `extend` with a batch
  holding a reference concatenates and the column reads `[None,
  {"timeline_id": "clt_6", "event_id": "x1azmn68"}]`.

### `test_events.py` - `EventData.from_dataframe()` refuses a non-string column label

**Purpose:** a table's column names are strings, so a DataFrame whose column
labels are not all strings cannot become an `EventData`. It was refused deep
in PyArrow with `TypeError: expected bytes, int found`, a message that names
neither the cause nor the label; it is now refused before any row is read,
with a `TypeError` (the same exception type) naming the offending labels.

**Contract and what proves it** (`TestFromDataFrameLabels`):

- `{"start": [0.0], 1: [2], 2.5: [3]}` raises `TypeError` with exactly
  `Column labels must be strings: [1, 2.5]` (every non-string label, in
  column order; `start` is not named).
- `{"start": [0.0], "x": [2]}` still builds: one row, the column `x` reads
  `[2]` (the check refuses only non-string labels).
- An empty DataFrame with an integer label still gives an empty `EventData`
  (no row to read, so nothing is refused; this answer is unchanged).

### `test_events.py` - Boxes as rows and the declared shape columns

**Purpose:** A `Box` stands for its event row as a Projection does, and its
two shape columns, `bounding_box` and `quadrilateral`, are typed by
declaration and written on the table's axis, as the coordinate columns are,
so a shape is never converted or rounded on its way into a table.

**Contract and what proves it** (`TestBoxRows`):

- `EventData.from_dicts([Box.from_shape(m1, origin=1221,
  id="m1").to_event_row()], pixels, int)` (the DMOS measure of
  `tests/core/README.md`) holds the columns `bounding_box` (one null row)
  and `quadrilateral`, typed exactly `BoundingBoxField.pa_schema` and
  `QuadrilateralField.pa_schema`, their metadata declaring the field type,
  `pixels` and `int`; `get_field(Quadrilateral)[0] == m1`, and the
  `start` and `duration` cells are `{0.0, 0, 1}` and `{940.0, 940, 1}`.
- A row whose shape is in another unit or written in another number type
  than the table's is refused whole, before any table is made (where a
  position would be re-expressed in the table's number type, a shape is
  not: geometry is never rounded): a row holding the `pixels` measure on a
  `css_pixels` float table, `The event's quadrilateral is in pixels as
  int, the events are in css_pixels as float; a shape is never converted
  onto the events' axis.`; a Box of fractions on a `user_units` float
  table, `The event's bounding_box is in user_units as fraction, the
  events are in user_units as float; a shape is never converted onto the
  events' axis.`
- **The two columns are a Box's.** Design: in a timeline's events,
  `bounding_box` and `quadrilateral` are declared columns holding a Box's
  shape; a Box row leaves the other one null. So a row of another event
  type giving either is refused with a message the library composes,
  naming the column and the event type, never a raw `ArrowTypeError`: a
  Note giving `bounding_box` `{"x": 1, "y": 2, "width": 3, "height": 4}`,
  `"left"` or a `BoundingBox` scalar, `An event of type 'Note' gives
  bounding_box: the columns bounding_box and quadrilateral hold the shape
  of a Box and belong to Box events only.` (A Note row giving neither is
  unchanged; every other test of this file adds such rows.) What would show
  it broken: a pyarrow conversion error reaching the caller, or a shape
  column accepted on a Note. A row without `event_type` giving
  `bounding_box` is named as such, never as a type `'None'`: `An event
  without an event type gives bounding_box: ...` (the same sentence after
  the subject).
- **A Box's shape is a shape or its memory cell.** A Box row's
  `bounding_box` given as the dict form `{"x": 1, "y": 2, "width": 3,
  "height": 4}` is refused, `A Box's bounding_box is a BoundingBox, not
  {'x': 1, 'y': 2, 'width': 3, 'height': 4}. Read a dict form with
  BoundingBox.from_dict.`; the rows of a table holding a Box, read back
  (`list(data)`, memory cells) and given to `from_dicts` again, give an
  equal table (the internal paths that rebuild rows, such as a parent's
  `get_events` over its children, rely on it).
- **A memory cell passes the rule a shape's numbers pass.** Design: every
  number enters a shape as given, never rounded (on an `int` axis only
  whole numbers, on a `fraction` axis exact ratios), and the package writes
  each number once, on the side its number type names. A memory cell has
  two sides, so a cell taken as it stands could hold a number the axis
  cannot hold (the third `1/3` on `pixels`, `int`) or two numbers (float
  side 46.0, ratio 47/1); both would be accepted in memory and written
  otherwise (the package writes the numerator of an int: `1`, `47`). So a
  cell is read as the shape it states, each number on the side the axis
  reads (the ratio on `int`/`fraction`, the float on `float`) and handed to
  the shape's constructor, the one entry of every number into a shape; its
  cells are then the shape's. Proved on `rect_h2` (`x` 385, `width` 139,
  `height` 20, cells `{385.0, 385, 1}` ...) on a `pixels` `int` table:
  - `y` the cell `{1/3, 1, 3}`: refused with exactly the message
    `BoundingBox.from_xywh(385, Fraction(1, 3), 139, 20, unit="pixels")`
    raises, `A shape's y is 1/3, which pixels as int cannot hold exactly:
    geometry is never rounded.` (one rule, one sentence, whichever form
    the number came in);
  - `y` the cell `{46.0, 47, 1}`: refused, `Box 'a': its bounding_box.y is
    the cell {'value': 46.0, 'numerator': 47, 'denominator': 1}, whose
    sides hold two numbers (46.0 and 47/1); a cell holds one number on
    both sides.`; a corner of a quadrilateral is named by its path: the DMOS
    measure `m1` with `top_left.x` `{1221.0, 1222, 1}`, `Box 'm1': its
    quadrilateral.top_left.x is the cell {'value': 1221.0, 'numerator':
    1222, 'denominator': 1}, whose sides hold two numbers (1221.0 and
    1222/1); ...`;
  - the thirds box `(1/3, 0, 2/3, 1)` on a `user_units` `fraction` table,
    its `x` given as `{1/3, 2, 6}` (one value in another spelling): accepted,
    the table equal to the one `from_dicts` makes of the Box scalar (`x`
    held `{1/3, 1, 3}`).
  What would show it broken: a refusal naming the cell's decoder instead of
  the shape's rule, or an accepted cell whose table differs from the
  scalar's.
- **A Box row states its axis as the package's Box table does, or not at
  all.** Design: a Box enters only on its timeline's axis; the package's
  Box table states `timeline_id`, `unit` and `number_type` once per row,
  and nothing else of the events holds them. So a Box dict row giving
  `unit` or `number_type` is checked as a shape scalar is: `unit "seconds"`,
  `number_type "float"` on a `pixels` `int` table is refused with the
  sentence the scalar path gives, `The event's bounding_box is in seconds as
  float, the events are in pixels as int; a shape is never converted onto
  the events' axis.`; `unit "seconds"` alone reads the number type as the
  table's (`... is in seconds as int, ...`). Equal keys (`"pixels"`,
  `"int"`, or the enums) are accepted and not stored: the table equals the
  one of the row without them (no `unit` or `number_type` column). A table
  of `from_dicts` belongs to no timeline, so a Box row giving `timeline_id`
  is refused there: `Box 'a' names the timeline 'seg_2' (timeline_id), but
  events read without a timeline hold no timeline ID: add the Box to its
  timeline with Timeline.add_events.` (`Timeline.add_events` checks it
  against its own ID: `tests/alignment/README.md`, "Box rows as the Box
  table states them").
- The shape leaves are reduced at the doors like every coordinate cell: on
  a `user_units` table written as fractions, a quadrilateral corner `x`
  cell stating `2442/2` is held as `1221/1` after `EventData(table)`, the
  column's metadata kept.

### `test_events.py` - An event type's field, composed from the table's columns

**Purpose:** An event table holds each event as one row whose properties
are spread over its columns (a Box's `id`, `start`, `duration`,
`bounding_box`, `quadrilateral`). `get_field(<event type>)` (the scalar or
its paired Field class, the call that reads any scalar back) gives the
events of that type as a field: one row per table row, the event where the
row's `event_type` is the type's name, null elsewhere, each number read on
the table's axis. One mechanism serves every event type whose data the
table holds: every property the scalar requires is a column (for a scalar
requiring none, every property is), an optional property whose column is
absent being null (`Box`, `Projection`, `Event`, `Note`, `ScoreNote`, a
`Tempo` stating `bpm`; not a `Beat` without `index`, nor a `Measure` row
without the measure's columns: `tests/timelines/README.md`, "A field only
where the table holds the type's data"); nothing in it names a type.

**What proves it** (`TestEventTypeFields`; the shapes of
`tests/core/README.md`):

- **Row-aligned, typed, exact.** A table of three rows (`from_dicts`, on
  `user_units` as `fraction`): the thirds Box (`(1/3, 0, 2/3, 1)`, `start`
  1/3, `duration` 2/3), a Note at 0 lasting 1, a Projection at 1/2.
  `get_field(Box)` is a `BoxField` named `box`, on `user_units` and
  `fraction`, `[box, None, None]` with `box` equal to the Box added and
  its `start.value` the `Fraction(1, 3)`; `get_field(Projection)` is a
  `ProjectionField` `[None, None, projection]`, the Projection's start
  `Fraction(1, 2)` on `user_units`. A float table (`css_pixels`, the box
  `(1.5, 0, 2.5, 1)`) gives `start.value` the float `1.5`, never a ratio.
  What would show it broken: a field shorter than the table (the rows of
  another type dropped, so index `i` no longer names row `i`), a number
  in another number type than the table's, or a Box whose shape is
  rounded.
- **One field, every spelling of the call.** `get_field(BoxField)` and
  `get_field(Box, name="box")` return the same object as `get_field(Box)`
  (the field cache), and `has_field(BoxField)` is `True`.
- **No rows of the type: no field, as before.** On a table without a Box
  row `has_field(BoxField)` is `False` and `get_field(Box)` raises the
  `KeyError` every class without a matching column raises (`No field
  matching 'BoxField' found in table. Available columns: [...]`).
- **A type keeping a property outside its struct is composed too**: the
  next section.

### `test_events.py` - A Note's field: the pitch from its own column, a ScoreNote on a Note's rows

**Purpose:** A Note is read back by the call that reads every event type,
`get_field(Note)`, though its pitch is represented once, in a column of its
own (`specific_pitch` on a spelled score, a bare `pitch` number, an
`EnharmonicPitch` struct), which `NoteField`'s struct leaves out. One rule,
stated for fields in general: a property the field's struct leaves out is
read from the columns holding a field of the property's type (each arm of
its annotation with a paired field, by every discovery strategy), else from
the column of the property's own name where its type is one the scalar
reads (a struct casting to the struct of one of the arms, or a number
standing for an arm whose struct holds one number); those columns ride in
the composed struct under their own names, so the scalar's `from_row` reads
them, and the stored table still holds the pitch once. Such a property
needs a representing column whether the scalar requires it or not: a table
of Notes without one states its pitches in a form the scalar cannot read
(a pitch name string, a spelling split over raw columns) or not at all,
and gives no field, as before the composition existed; read as Notes, every
one of its notes would come back a rest. A subtype is read on the rows
of its base's name (`ScoreNote` on rows typed `Note`) when the table
states every property it adds, read as the field's types; nothing in the
rule names a type.

**What proves it** (`TestANoteField`):

- **The pitch from its own column, the positions on the table's axis.** A
  `seconds`/`float` table of three rows (`from_dicts`): a Note at 0.5
  lasting 0.25 with the pitch `{midi_number: 60}`, a Rest at 0.75, a Note
  at 1.0 lasting 0.5 with `{midi_number: 62}`. `get_field(Note)` is a
  `NoteField` on `seconds`/`float`, `[Note(0.5, end 0.75, duration 0.25,
  EnharmonicPitch(60)), None, Note(1.0, end 1.5, duration 0.5,
  EnharmonicPitch(62))]`, every position a `float` in `seconds` (a reader
  assuming quarters gives `quarters`; one that dropped the pitch gives
  rests). The table's columns are the same before and after the call (no
  second pitch column is stored), the field's struct holds `pitch` once,
  after the Note's members.
- **No column the scalar reads a pitch from: no field.** Each table below
  answers `get_field(Note)` with the `KeyError` every field the table
  cannot give raises (`No field matching 'NoteField' found in table.
  Available columns: [...]`) and `has_field(NoteField)` with `False`,
  exactly as `get_field(Note)` answered on every table before an event
  type's field was composed: the three-row table of the previous section
  (its Note row has no pitch column at all); one Note row whose `pitch` is
  the name string `"E♭5"` (as the EEP note files state it); one Note row
  with a bare `midi` integer on a base `EventData`, which declares no view
  of a `midi` column (only a score's `NoteEventData` does); one Note row
  spelling its pitch in raw `step`, `octave` and `actual_accidental`
  columns (as IEEE 1599 states it). What would show it broken: a
  `NoteField` whose items are all rests. The same row with `pitch` the
  integer 63 gives `[Note(..., EnharmonicPitch(63))]`: the own-name
  column counts where it is a number. The corpus tables these shapes come
  from are read in `tests/timelines/README.md`, "A field only where the
  table holds the type's data".
- **A ScoreNote on the rows of a Note.** A `quarters`/`fraction` table of
  one Note row (start 0, duration 1/2, pitch `{midi_number: 60}`, `staff`
  1, `voice` 2): `get_field(ScoreNote)` is a `ScoreNoteField` holding
  `ScoreNote(..., voice=2, staff=1)`; `get_field(Note)` on the same table
  the plain `Note` (staff and voice not read, as `from_row` ignores the
  columns a type does not declare).
- **What is no ScoreNote.** Each answers as `get_field` answers for any
  field the table cannot give (`KeyError` `No field matching
  'ScoreNoteField' found in table. ...`, `has_field` `False`), while
  `get_field(Note)` reads the notes: a table holding `staff` alone (the
  shape of a performance whose `staff` numbers an instrument, EEP, RepoVizz:
  no `voice`); a table whose `staff` and `voice` are text identifiers
  (`"P1_staff1"`, as IEEE 1599 writes them: they do not cast to the
  integer members); a table whose `staff` and `voice` columns are null on
  every Note row.
- **A base type's row is no subtype's event.** A table holding one
  `Event` row (`Event(id="e", start=0, duration=1)`, its shape columns
  null): `get_field(Box)` raises the `KeyError`; a Box adds a shape, which
  the row does not state.

### `test_events.py` - One axis per table: number type kept, cells reduced

**Purpose:** An `EventData` reads every cell in its one unit and number type,
and holds one spelling of each value, whatever way its rows came in.

**Contract and what proves it** (`TestOneAxisPerTable`):

- **A further coordinate column keeps its number type through Parquet.**
  `from_arrays` given a coordinate column outside the schema (`onset`,
  holding `1/3` on a `fraction` quarters table) wrote only `{unit}` beside
  it, so after `to_parquet` / `from_parquet` its `CoordinateField` declared
  `float` and read `0.3333333333333333`. It is now written by the builder
  that writes the base columns (unit and number type in the metadata blob):
  the blob reads `{"unit": "quarters", "number_type": "fraction", "version":
  1}`, the same as `start`'s, and the field read back declares `fraction` and
  holds `Fraction(1, 3)`. A file
  written before, whose column carries only the bare `unit` key, still reads
  as before (`float`, `0.3333333333333333`): that branch is the reader's
  fallback, asserted on a table written with the old metadata by hand.
- **`concat` keeps every row's number type; the cast is on request**
  (`TestJoiningKeepsNumberTypes`). A join never converts a value: tables of
  one axis give a table of that axis as they stand (no `timeline_id`
  column, the inputs unchanged); tables of different axes give a table
  spanning them, each row keyed by its timeline in the dictionary-encoded
  column `timeline_id` (a table standing alone that lies on no timeline by
  its axis, `<quarters fraction>`, `<quarters float>`) and read in that
  key's unit and number type, which the table maps once per key in memory.
  What proves it is the cells, read member by member, and the rows each
  number type selects (`filter(include_number_types=)`), for each
  direction:
  - a `fraction` quarters table (`a` at `[1/3, 2/3)`) joined by a `float`
    one (`b` at `0.1` lasting `0.2`, an instant `c` at `0.5`): every cell of
    the result equals the cell of the table it came from, member for member
    (`b`'s start stays the float cell `0.1`, not the fraction
    `3602879701896397/36028797018963968`), the keys are `<quarters
    fraction>`, `<quarters float>`, `<quarters float>`, the `fraction` rows
    are `a`, the `float` rows `b`, `c`, and the two inputs keep 1 and 2 rows;
  - the old re-expressions are now the cast, `to_number_type`, with the old
    values: on that join, `to_number_type("fraction")` gives `b`'s start
    `3602879701896397/36028797018963968` (`Fraction(0.1)`), its duration
    `3602879701896397/18014398509481984` (`Fraction(0.2)`), its end their
    exact sum `10808639105689191/36028797018963968` (`Fraction(0.1) +
    Fraction(0.2)`, not `Fraction(0.1 + 0.2)`: start and duration are the
    stored pair of an event and the end is derived from them, so a row cast
    onto an exact type ends at its start plus its duration, exactly), `c` an
    instant at `1/2`, every coordinate field and every row `fraction`; the
    `float` table joined by the `fraction` one, cast to `float`, gives `a`'s
    cells `0.3333333333333333` and `0.6666666666666666`, each mirrored by
    its double's exact ratio (`6004799503160661/18014398509481984`); an
    `int` table joined by `fraction` rows `[5/2, 7/2)` and `[3/2, 9/2)` and a
    `float` row `[2.5, 3.5)` keeps them as stated (`5/2`, `3/2`, `2.5`, the
    rows of each type selected by each type) and, cast to `int`, gives
    start and duration nearest with ties to the even neighbour and the end
    their sum: `5/2` → start 2, duration 1, end 3; `3/2` → start 2,
    duration 3, end 5; `2.5` → start 2, duration 1, end 3, every
    `denominator` 1; a `fraction` table joined by `int` rows, cast to
    `fraction`, keeps them as `n/1`. A row already of the target type is
    kept as it is (`a`, `x`).
  - a table of another unit joins on its own axis (`<seconds float>`), its
    cell `1.0` unchanged: the join's single-unit contract is lifted.
  - `to_number_type(..., columns=[...])` on a table standing alone casts
    the columns named and declares the type on them only (`start`
    `fraction`, `end` and `duration` `float`, the table `float`); on a
    table spanning timelines, whose rows read every coordinate column in
    their timeline's type, a partial cast is refused naming the columns it
    would leave out.
  - `filter(include_number_types=)` on a table standing alone is all or
    nothing: its declared type keeps every row, another none.
  The joins the factory makes (`flatten=True`, `group_by=`) cast the stores
  joining to the first store's number type, as they did; that is proven end
  to end in `tests/timelines/test_factory.py` (`TestStoresOfTwoNumberTypes`).
- **Bounds of `filter`** (`TestFilterBounds`). An exact bound (`int`,
  `Fraction`) compares exactly on an `int` or `fraction` table; a bound
  written as a float compares as a key with the float side of the cells, so
  `min_coord=0.1` keeps a start at `1/10` on a `fraction` table and
  `min_coord=10.5` on an `int` table keeps a start at 11 and leaves out one
  at 10. An infinite float bound is an ordinary double: `max_coord=inf`
  keeps every row and `min_coord=inf` none, on `float`, `fraction` and
  `int` tables alike. NaN names no position and is refused (`NaN names no
  position: it compares with no cell`) rather than answering an empty
  table nobody asked for.
- **Cells are reduced on entry.** A ratio stated as `2/4` is accepted and
  held as `1/2` at every door that takes whole cells: `EventData(table)`,
  `EventData.from_parquet`; `concat` joins tables that entered
  through those doors, so their rows stay reduced without a second pass. Each is
  asserted on the cells read back (`(1, 2)`, never `(2, 4)`), never by
  refusing the input. A table whose `start` cells spell one value three
  ways, `2/4`, `4/8` and `1/2`, holds three equal cells after entry, and
  `to_parquet` writes `(1, 2)` three times: the bytes on disk are the
  reduced spelling, read with `pyarrow.parquet` directly, not through the
  library. The defect this closes, end to end: a Projection at `2/4` in a
  Parquet file, read with `from_parquet` and placed on a timeline, then
  `add_projections` of the same Projection at `1/2`, gave a second
  Projection (`projection_2`), since Projections are matched by their cells;
  now it is the same Projection (the ID the first call gave, one event).
- **Cells are reduced at the doors only** (`TestReducedAtTheDoorsOnly`). A
  table derived from tables already held (rows selected by `filter` or
  `where`, tables joined by `concat`, a column retyped, a
  timeline's own copy, the rows `get_events` rebuilds from a timeline and
  its children) holds cells that were reduced when they entered, so
  reducing them again costs a pass over every coordinate column and changes
  nothing. The proof counts the calls of the one reducing function,
  `storage.events._in_lowest_terms`, replaced for the test by a counting
  wrapper around it: `EventData(table)` and `EventData.from_parquet` call it
  once each (the doors); `filter`, `concat`, a timeline made
  by `create_timeline`, and `get_events(temporal_type=, event_type=,
  min_coord=, max_coord=)` on a flat timeline and on a parent with a child
  call it zero times. A table of no rows is not given to it at all, as it
  holds no cell to reduce (a new timeline starts with one). The results are
  asserted too (IDs and the reduced
  cells), so a path that skipped the reduction and then returned wrong rows
  would not pass on the count alone; the reduction tests above pass
  unmodified, which is what shows the doors still reduce.

### `test_tree_store.py` - The tree store: one table per timeline tree

**Purpose:** The root of a timeline tree holds the events of itself and every
descendant (`TreeStore`, `storage/tree.py`), each row lying in its own
timeline's coordinates, unit and number type; the tree's table is the
concatenation of every timeline's rows, each row naming its timeline, and a
timeline reads its own rows as they lie in that table, copying nothing.

**Contract and what proves it** (one tree: a `fraction` quarters root
`clt_root` of length 8 holding `r1` at `1/2`; a `fraction` child `clt_a` at
offset 0 holding `a1` at `1/3` and `a2` at `0`, added in that order; a
`float` child `clt_b` at offset 4 holding `b1` at `0.1`):

- **One table for the tree, in the order rows came.** The three timelines
  refer to one store, and the children's own stores are emptied when they
  are attached (their `timeline_ids` read `()`). The table holds 6 rows: the
  root's `r1` and the two `Segment` rows recording its children (still
  among its events), `a1`, `a2`, `b1`. Its `timeline_id` column is
  dictionary-encoded and reads `clt_root` three times, `clt_a` twice,
  `clt_b` once: the timelines in the order they joined the tree, the root
  first, and each timeline's rows in the order they were added, never
  re-sorted by start. So the IDs read `r1`, `clt_a`, `clt_b`, `a1`, `a2`,
  `b1` (`a1` before `a2` although it starts later), and `clt_a`'s own rows
  `a1`, `a2`. A format round trip depends on it: rows read from a file come
  back in the file's order. Positions stay child-local: `b1`'s start is the
  float cell `0.1`, not `4.1`.
- **The map and the row-type mask.** The store maps `clt_root` and `clt_a`
  to `(quarters, fraction)`, `clt_b` to `(quarters, float)`, once, in
  memory; no column of the table states a number type. The row-type mask
  of the tree's one table (`row_type_codes(store.tables[0], store.axes)`)
  is `[1, 1, 1, 1, 1, 2]` (`NUMBER_TYPE_CODES`: fraction 1, float 2).
- **A timeline's rows are copied by nothing.** The buffer holding the float
  side of `clt_b`'s `start` cells is the buffer of one chunk of the tree
  table's `start` column (same address, same size): the tree's table is
  built from the timelines' rows without a copy. The rows' `start` field
  declares `quarters` and `float`, `clt_a`'s `fraction`, and their
  `EventData` say so (`number_type`); they hold the timeline's own columns,
  never `timeline_id`. A timeline's rows are the same object until they
  change; rows added to the root or a child attached to it leave `clt_a`'s
  object as it was (the identity is the proof that no other timeline's
  change rebuilds or re-sorts them, so no old copy of the tree is held by
  a timeline's rows).
- **Rows change only through the store.** `Timeline._events` and
  `Timeline.events` are read-only (assigning either raises
  `AttributeError`); `add_events` on a child adds its rows to the root's
  table (7 rows after one more event on `clt_a`, which reads `a1`, `a2`,
  `a3`).
- **Attaching moves rows.** Before `add_child`, `clt_b`'s rows are in its
  own store; after it they are in the root's (`clt_b`'s segment row, then
  `b1`), and its own store holds none.
- **Timelines typing one column differently share the tree.** A child
  whose `pitch` is the struct `{midi_number, name}` under a root whose
  `pitch` is `{midi_number}` is attached, and the tree's table types
  `pitch` with both members: the root's cell reads `name` null, the child's
  keeps `C4`; each timeline reads its own rows in its own type (the root's
  `pitch` still has one member). A child whose `velocity` is a string
  under a root whose `velocity` is an integer is attached too: the tree
  then has two tables (`tables`), the root's rows in the first, the
  child's in a second, each cell as it was written (`80`, `"ff"`), every
  row of the tree in one of them (3 rows). The store offers no single
  table or `EventData` of the tree (`table` and `data` are not attributes
  of it), so no reader can take the first table for the tree and leave
  the child's row out unnoticed. No cell is rewritten.
- **A refused child changes nothing.** A child holding an event ID of the
  tree, and a child holding an event whose ID is the child's own (its
  `Segment` row, named by its ID, would join the parent), are refused
  before anything changes: the parent keeps its children, the tree's rows
  and its store, and the child keeps its parent (none), its store and its
  rows.
- **A join keeps each row's type** (the join converts nothing):
  `clt_a`'s events concatenated with `clt_b`'s key the rows by their
  timelines (`clt_a`, `clt_a`, `clt_b`), `b1`'s start stays the float cell
  `0.1` with the ratio side as stored, and the float filter keeps `b1`
  alone; `to_number_type("fraction")` gives `b1` the exact ratio of the
  double, `3602879701896397/36028797018963968`, its field declaring
  `fraction`.
- **A join is never refused for the number type**, even where a timeline's
  rows meet a cast of themselves. A cast that changes the type no longer
  lies on its timeline: it is keyed by its axis. `clt_b`'s rows joined
  with their own `fraction` cast (both orders: rows first, cast first)
  give a table of 2 rows, keyed `clt_b` and `<quarters fraction>` (or
  `<quarters fraction>` and `clt_b`), its map holding `clt_b` as `float`
  and the key `<quarters fraction>` as `fraction`; the float filter keeps
  the row of `clt_b`, the fraction filter the cast. `clt_a`'s rows joined
  with their own `float` cast likewise give `clt_a` twice and the key
  `<quarters float>` twice, in the order joined, and so does the cast of
  `clt_a`'s rows read through `get_events` joined with them, the cast
  first. A table spanning timelines cast to a number type keys the rows
  whose type changes by their new axis, as a table standing alone does:
  the table spanning `clt_a` and `clt_b` cast to `float` is keyed
  `<quarters float>`, `<quarters float>`, `clt_b` (`clt_b`'s rows were
  `float` already and keep their timeline), its map `<quarters float>`
  and `clt_b`, both `float`; joined with `clt_a`'s own `fraction` rows,
  the keys read `<quarters float>`, `<quarters float>`, `clt_b`, `clt_a`,
  `clt_a`, the map adds `clt_a` as `fraction`, and the fraction filter
  keeps `a1`, `a2`. A cast to the type the rows have already keeps its
  timeline (`clt_b`'s rows cast to `float` and joined with themselves are
  one axis, joined as they stand).
- **A cast of a spanning table joined again keeps every row's type**
  (`TestACastJoinedAgainKeepsEachRowsType`). The defect this closes: the
  cast kept a key whose axis it changed, mapped to the new type, so the
  rows it named, joined again with the uncast table, were read in the
  cast's type, and a `fraction` row at `1/10` came back from a cast to
  `fraction` as `3602879701896397/36028797018963968`. Each key whose axis
  changes is keyed by its new axis now, so every key of a table names
  the axis its rows lie on, and the re-join cannot meet a key held on
  another axis (`concat` asserts it). Two spellings, each joined as
  `span.to_number_type("float").concat(span)`:
  - **keys of axes:** a `fraction` table holding `a` at `1/10` joined with
    a `float` table holding `b` at `0.1` (`span`, keyed `<quarters
    fraction>`, `<quarters float>`). The cast is keyed `<quarters float>`
    twice, its map `<quarters float>` alone, `float`. The re-join is keyed
    `<quarters float>`, `<quarters float>`, `<quarters fraction>`,
    `<quarters float>`, its map `<quarters float>` `float` and `<quarters
    fraction>` `fraction`; the fraction filter keeps `a` alone, the float
    filter `a`, `b`, `b`; cast to `fraction`, the rows read
    `3602879701896397/36028797018963968` (the cast of `a`, a float now),
    the same for `b`, `1/10` (`a` as it was), and
    `3602879701896397/36028797018963968` (`b`).
  - **keys of timelines:** a `fraction` quarters root `R` holding `a` at
    `1/10` with a `float` child `C` at offset 0 holding `b` at `0.1`;
    `span` is `R`'s rows (`a`, the `Segment` row `C` at `0`) joined with
    `C`'s (`b`), keyed `R`, `R`, `C`. The re-join is keyed `<quarters
    float>`, `<quarters float>`, `C`, `R`, `R`, `C`, its map `<quarters
    float>` and `C` `float`, `R` `fraction`; the fraction filter keeps `a`,
    `C` (`R`'s own rows); cast to `fraction`, the rows read
    `3602879701896397/36028797018963968` (the cast of `a`), `0/1` (the
    cast of the `Segment` row), `3602879701896397/36028797018963968`
    (`b`), `1/10` (`a` as it was), `0/1`, and
    `3602879701896397/36028797018963968` (`b`).
- **One rule joins columns, in `concat` and in the tree**
  (`TestOneRuleJoinsColumns`, `storage.events.unified_type`). The same two
  timelines gave a tree and were refused by `concat`, which used Arrow's
  own promotion; both now join by the one rule. A root whose `tag` is the
  struct `{midi}` and a child whose `tag` is `{midi, name}`:
  `root.events.concat(a.events)` types `tag` as the tree's table does
  (`{midi: int64, name: string}`) and holds the same cells
  (`{midi: 60, name: None}`, `None` for the `Segment` row,
  `{midi: 60, name: "C4"}`). A table whose `tag` column is
  dictionary-encoded (`"x"`) joined with one whose `tag` is plain
  (`"y"`) gives a plain `string` column `["x", "y"]` (Arrow refused it);
  so does that table joined with a `float` table whose `tag` is plain
  (`"z"`), the two rows keyed `<quarters fraction>`, `<quarters float>`.
  A column holding an `int` beside a `string` (a root's `velocity` 80, a
  child's `"ff"`) has no one type: the tree keeps the child in a table of
  its own (above), and `concat`, which returns one table, refuses it,
  naming the column and the types (`the column 'velocity' holds values of
  type int64 in one table and of type string in another, which no one type
  reads; one table holds one type per column`), never for the number
  type.
- **The extremes of a spanning table are read row by row.** A `fraction`
  table holding a start at `1/2` joined with a `float` table holding a
  start at `0.7` spans both types; `coordinate_range()` answers
  `(Fraction(1, 2), 0.7)`, the high end the `float` the row holds (type
  `float`), never the exact ratio of that double
  (`3152519739159347/4503599627370496`), which reading every row in the
  first table's type gave.
- **The number-type filter resolves from the map.** On the tree's
  timelines' rows joined (`root.events.concat(a.events, b.events)`)
  `include_number_types={NumberType.float}` keeps `b1`,
  `{NumberType.fraction}` every other row; `root.get_events(...)` with the
  same filter gives `["b1"]` and `["r1", "a1", "a2"]` (its own rows and its
  children's, each named by `source_timeline`, resolved through the map,
  never from a cell).
- **Event IDs are unique across the tree**, with nothing renamed, and one
  message names the ID, its holder and the timeline it would join:
  `r1` added to `clt_a` is refused (`Event id 'r1' is already held by an
  event of timeline 'clt_root', so it cannot join timeline 'clt_a': event
  ids are unique across a timeline tree`); a child `clt_c` holding `r1`
  attached to the root is refused with the same message for `clt_c`, and
  the tree is as it was: two children, 6 rows, `clt_c` holding its row in
  its own store. Two rows given one ID are refused naming the ID, their
  number and the timeline (`Event id 'x' is given to 2 events joining
  timeline 'clt_a'; event ids are unique across a timeline tree`).
- **A timeline's new ID renames its rows without rewriting one.** After the
  child `clt_a` takes the ID `clt_x`, the store's timelines read
  `clt_root`, `clt_x`, `clt_b`, the child reads its two rows, and the
  store names `clt_x` as the holder of `a1`.
- **A refused new ID changes nothing.** The child `clt_a` given the ID of
  another timeline of the tree (`clt_b`, `clt_root`), or of an event of
  the tree (`r1`, which its parent's `Segment` row would then hold a
  second time), is refused before its parent changes: the parent's
  children read `clt_a`, `clt_b` with their offsets 0 and 4, its rows
  read `r1`, `clt_a`, `clt_b`, the store's timelines `clt_root`, `clt_a`,
  `clt_b`, and the child keeps its ID. Before, the parent's children
  were re-keyed (a sibling's entry overwritten) before the store refused.

### `test_coordinate_range.py` - `EventData.coordinate_range()`

**Purpose:** `coordinate_range()` (read by every `create_timeline`) finds the
smallest and largest number among the `start` and `end` cells with array
kernels instead of decoding one Python number per cell, and answers exactly
what the per-cell reading answered.

**Why a comparison proves it:** the per-cell reading is correct by
construction (every non-null cell decoded with `struct_to_coordinate` in the
axis's number type, then Python's `min` and `max`), so the test keeps it,
in the test file only, as the reference, and asserts that the two agree on
every table: equal results, the same Python type for each end (`int`,
`float` or `Fraction` as the axis declares), and the same `repr` (so the
sign of a zero is compared too: `min` and `max` give the first of equal
values in column order). Where the extremes can be stated they are also
asserted as literals.

**What could go wrong, and the table that would show it:**

- Two ratios sharing one double but differing exactly. The kernel finds
  the extreme double; the exact extreme must still win among the cells
  holding it. Seeded columns of neighbours `n/(3·2⁶⁰)` around 1 and -1
  and `n/(3·2⁵⁰)` around 2⁹ (dozens share each double), on both columns,
  with nulls between.
- Integers past `2⁵³`, where neighbouring integers share a double: ticks
  `2⁶² − j` and `−2⁶² + j`, shuffled.
- Floats: large and small magnitudes of both signs, and `0.0` / `-0.0`
  in both orders (the first in column order is the answer).
- The extreme lying in `end` only, or in `start` only; `end` entirely null;
  columns of several chunks; a single row; every cell null (`None`); an
  empty table (`None`).
- Cells holding only a `value` (older artifacts): read as their double's
  exact ratio, as before.
- Dictionary-encoded `start` / `end` columns (a dictionary over the number
  cells, which the constructor accepts): the kernels cannot read a member
  of such a column, so the range must decode it first. The reference reads
  them as it reads any column (`to_pylist` decodes), so agreement plus the
  literal extremes prove the decoding. Cases: one chunk on each axis
  (fractions `1/3` and `5/2` picked by indices `[1, 0, 1]`, range
  `(1/3, 5/2)`); several chunks, each with its own dictionary, one of them
  empty, and nulls among the indices; a sliced dictionary column, whose
  extremes left out by the slice must not count; a dictionary `start` beside
  a plain `end`; and `create_timeline()` of the one-chunk table, which reads
  the range, building a timeline of length `5/2`.
- The event tables the storage and core tests build (decimal durations on
  quarters, completions past int64 cross-products, ticks past `2⁵³`,
  three-decimal seconds, Projections) through `from_dicts` and
  `from_arrays`.

### `test_store_summary_contract.py` - store summaries and cards

Every store a loader exposes (single, dict, score, MIDI, graphical,
alignment) returns the exact `summary()` dict pinned for its specimen, and a
non-empty notebook card. The alignment store's facts are its events, its
C-Maps, its domains and its table names (`event_count`, `cmap_count`,
`domains`, `store_names`): it holds no MatchClaims, which live in the
MatchClaim table, so no match count appears among them.

### `test_timeline_factory_import.py` - `create_timeline()` after `import timetoalign`

**What would prove it:** the storage layer builds timelines through a factory
that the timeline layer registers when it is imported, and `import timetoalign`
loads the timeline layer lazily, so the defect shows only in an interpreter
that has imported nothing else. A test inside the pytest process cannot see
it: some other test has always imported `timetoalign.timelines` already.
Each check therefore runs a fresh interpreter (`sys.executable -c`) that does
`import timetoalign` and nothing more, first asserts that
`timetoalign.timelines` is NOT yet in `sys.modules` (otherwise the check would
pass whether or not the fix exists), then calls the entry point and prints
what it made.

- `EventData.from_dicts` of one `Note` `n1`, `[0, 1)` quarters, then
  `create_timeline(uid="notes")`: a `ContinuousLogicalTimeline` with id
  `notes` holding one event.
- `SingleStore` of the same data, then `create_timeline(uid="score")`: a
  `ContinuousLogicalTimeline` with id `score` whose events are `["n1"]` (an
  event keeps its ID on the timeline).

The child's standard output must be exactly those lines, and its exit code 0;
before the fix both calls raised `RuntimeError: Timeline creation requires
timetoalign.timelines`.

### `test_field_type_map.py` - the two maps stored columns and scalars are read back through

**What would prove it:** a stored column names its paired field class in its
metadata (`field_type`), and `storage.mixins._get_field_type_map()` turns that
name back into the class; its iteration order also decides which class claims
a struct column by shape and the order of `get_fields_satisfying`. The second
map, `_get_scalar_to_field_map()`, turns a scalar class back into its paired
field: `EventData.get_field(<scalar class>)` and the union arms of a scalar's
property (`_columns_representing`) go through it. Every paired
`XField(SemanticField[X])` declared anywhere in the package registers itself
in `core.fields` when its class is created. The field-type map is a live view
of that registry after a fixed head of core classes, and the scalar map is
derived from the field-type map: each field class keyed by its scalar. A pair
declared after the first lookup (`MatchClaimField`, declared when
`timetoalign.alignment` loads) therefore joins both maps at the next lookup.
Whether a map is complete depends on which modules an interpreter has loaded,
never on which lookups ran before. The scalar map once broke this: it was
built on its first call and kept, so an interpreter whose first scalar lookup
preceded `import timetoalign.alignment` never found `MatchClaim`
(`KeyError: No paired SemanticField registered for scalar 'MatchClaim'`),
while one that loaded alignment first did.

A test inside the pytest process cannot see any of this, since some other
test has always imported every subpackage already, so each check runs a fresh
interpreter (`sys.executable -c`) and prints each map as lines in iteration
order: `name module.Class` for the field-type map, `module.Scalar
module.Field` for the scalar map. Both are pinned. The scalar map is the
field-type map's 43 classes, in the same order, each keyed by its scalar;
every scalar has exactly one paired field.

- After `import timetoalign` and nothing else: both maps hold the 42 pinned
  entries without `MatchClaim` / `MatchClaimField` (the child first asserts
  that `timetoalign.alignment` is not loaded, otherwise the check would prove
  nothing). After then importing `timetoalign.alignment`, the same lookups
  yield all 43, `MatchClaim` / `MatchClaimField` last, and
  `_get_field_class_for_scalar(MatchClaim)` answers `MatchClaimField`. This
  is the sequence that raised `KeyError` while the scalar map was kept from
  its first call.
- For seven first imports (`timetoalign`, `timetoalign.storage`,
  `timetoalign.storage.mixins`, `timetoalign.core.fields`,
  `timetoalign.core.events`, `timetoalign.alignment.claims`,
  `timetoalign.timelines`), each run twice: once with both maps read right
  after that import, once without that early read. Then every subpackage of
  `timetoalign` is imported (in reverse order, so the first import is not
  always the same one again) and both maps are read again. The late read
  gives exactly the pinned 43 entries of each map, in the pinned order, every
  time; where there was an early read, the late field-type map is the same
  dictionary object, and the early read gave the pinned entries without
  `MatchClaim` / `MatchClaimField` exactly where `timetoalign.alignment` was
  not yet loaded (the child prints whether it was), else all of them.

What the first imports can and cannot vary: every `timetoalign.*` import runs
the package root first, and the root imports `timetoalign.core` before
`timetoalign.storage`, so no first import can make the storage layer look the
registry up before the core has registered its pairs. The seven first imports
vary which subpackages follow the core, and whether the first lookup comes
before alignment loads; the early-read variant is what discriminates a map
kept from its first call from a live one.

The pinned lists are the maps as captured on the code base before the core
stopped patching the field-type map from outside the storage layer (that
refactor had to leave it identical) and, for the scalar map, as it was built
when its first call came after every module was loaded. Every field-type key
equals its class's name (`field_type()`).
