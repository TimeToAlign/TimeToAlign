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
  A timeline is one axis: the factory refuses stores of two number types
  instead of casting them, and a timeline takes no table spanning axes
  (`tests/timelines/test_factory.py`, `TestStoresOfTwoNumberTypes`;
  `tests/storage/test_registry.py`, `TestRefusals`).
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
- **Further fields of `filter`** (`TestFilterFields`). A keyword beside the
  named criteria is an exact match on a field. On a base `EventData` of two
  Notes `n1` (pitch 60) and `n2` (pitch 62), whose class declares no
  `pitch`, `filter(pitch=62)` keeps `n2` only: the table's column is a
  field. A field the class declares that the table lacks holds no value, so
  no row is kept: an `AnnotationEventData` of one Harmony `h1` without its
  `subtype` column, `filter(subtype="Harmony")`, keeps none. A field that is
  neither is refused, never passed over: `filter(colour="red")` raises
  `KeyError` (`No field 'colour' to filter events by. Available fields:
  ['id', 'name', 'temporal_type', 'event_type', 'start', 'end', 'duration',
  'pitch']`). What would show it broken: a criterion that keeps every row
  because its name was not recognised.
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

### `test_registry.py` - The event registry: root positions, local events, views

**Purpose:** The events of a family of timelines (a root and every
descendant, one unit and one number type) are held by the root's
`EventRegistry` (`storage/registry.py`): one registry row per event, its
position on the **root's** axis, and per timeline its events in its **own**
coordinates (`Timeline.events`). Both are read back, and agree, on every
number type the plan states; everything a ruling refuses is refused before
anything changes, and a refusal leaves the family as it was (its rows, its
events and its extractions).

**The family** (`_family()`, used throughout): a `fraction` quarters root
`clt_root` (length 8) holding the Beat `r1` at `1/2`; a child `clt_a`
(length 4) holding the Beats `a1` at `1/3` and `a2` at `0`, added in that
order, which first takes a grandchild `clt_g` (length 2) at offset 1 holding
the Note `g1` at `1/4` lasting `1/2` with `pitch` 60; `clt_a` then joins the
root at offset 2, and a child `clt_b` (length 4) holding the Beat `b1` at
`1/10` joins at 4.

**Contract and what proves it:**

- **Root storage** (`TestRootPositions`). The root's registry view holds
  its 5 events in the columns of `REGISTRY_SCHEMA` (`id`, `kind`,
  `location`, `timeline_id`, `unit`, `number_type`, `start`, `duration`,
  `name`, `extraction_id`, `locator`, `vanished_in`), the timelines in
  joining order, each timeline's rows in the order they came: IDs `r1`,
  `a1`, `a2`, `g1`, `b1`; kinds `Beat`, `Beat`, `Beat`, `Note`, `Beat`;
  timelines `clt_root`, `clt_a`, `clt_a`, `clt_g`, `clt_b`; every `unit`
  `quarters`, every `number_type` `fraction`. The starts, as exact ratios of
  the cells, are the root's: `1/2`, `7/3` (`2 + 1/3`), `2`, `13/4` (`1/4 + 1
  + 2`), `41/10`; durations null but `g1`'s `1/2`. A child's place is a
  `Segment` row of its parent, held by the family for the timelines and
  shown by no view: the family's own table (`_family.table()`) holds 8 rows,
  `r1`, `clt_a`, `clt_b`, `a1`, `a2`, `clt_g`, `g1`, `b1`, the Segments at
  `2`, `4`, `3` lasting `4`, `4`, `2`. Every event row names one extraction,
  the family's tool extraction (parser `timetoalign`, the library's version,
  no source); a `Segment` row names none (`extraction_id` null);
  `location`, `locator`, `vanished_in` are null. An event stating no event
  type (`{"id": "z1", "start": 1}` added to a root) is of kind `Event`, the
  `Event` base's (the column is never null), its own `event_type` left as
  held (null).
- **Local positions derived, exact for int and fraction.** The timelines
  read their own events locally: `clt_a`'s starts `1/3`, `0`, `1` (its
  Segment row for `clt_g`), `clt_g`'s `1/4`, `clt_b`'s `1/10`; on a ticks
  root (`int`) with a child at 7 holding an event at 3, the registry start
  is `10` (an `int` cell, denominator 1) and the child's own start `3`.
  Read from a package (`add_rows`): a `fraction` root with a child at 2 and
  no events takes a registry row `p1` (a Note at `5/2` lasting `1/2`) and
  the content `{id: p1, pitch: 62}`: the child's events read `p1` at `1/2`,
  ending at `1`, lasting `1/2`, an `interval` `Note` with `pitch` 62; on the
  root's axis the row stays at `5/2`. On a ticks root the row at `10` gives
  the child at 7 the start `3`.
- **Rows read from a package keep their order.** Registry rows `n1` (a Note,
  locator `L1`), `b1` (a Beat, `L2`), `n2` (a Note, `L3`) at `1`, `2`, `3`,
  with the contents `{n1: 60, n2: 64}` (pitch) and `{b1: 1}` (strength),
  taken by `add_rows` (not registered): the timeline's events read `n1`,
  `b1`, `n2`, as its registry rows do, though the kinds' contents are read
  kind by kind. A second run of the same source (another extraction) giving
  `L1` with pitch 61 and `L3` with pitch 65 updates it: events and registry
  both `n1`, `b1`, `n2`, pitches `61`, `None`, `65`, `b1` (not found)
  vanished in the second run and kept.
- **The float case as the plan states it.** On a `float` seconds root with
  a child at offset `0.2` holding an event at `0.1`, built in code, the
  registry holds `0.1 + 0.2` in binary64 (`0.30000000000000004`) and the
  child its own `0.1`: both as computed. A registry row read from a package
  at `0.3` for that child gives the child `0.3 - 0.2` in binary64
  (`0.09999999999999998`): what binary64 gives, the root's `0.3` kept.
- **The child shift is column-wise** (`TestColumnWise`). With the per-row
  shift of `core/time.py` (`shift_coordinate`, which `get_event` uses)
  replaced by a function that raises, the root's registry rows and
  `get_events()` over its children are built all the same and give the
  starts above (`get_events` names each row's holder in `source_timeline`:
  null for `r1`, `clt_a` for `a1`, `a2`, `clt_g` for `g1`, `clt_b` for
  `b1`, its starts `1/2`, `7/3`, `2`, `13/4`, `41/10`).
- **Views and counts** (`TestViews`). A view reads when asked, so it
  follows the rows: `clt_a.registry` (itself and its descendants) reads
  `a1`, `a2`, `g1`, and after `a3` is added to `clt_a`, the same view object
  also `a3` (`a1`, `a2`, `a3`, `g1`). `filter(kinds="Beat")` gives `r1`,
  `a1`, `a2`, `b1`; chained with `filter(within="clt_a")` (the unified
  filter parameter naming a set of timelines), `a1`, `a2`;
  `filter(min_coord=2, max_coord=4)` (the root's axis, start inclusive, end
  exclusive) `a1`, `a2`, `g1`. **A bound's unit selects the families in that
  unit**: a bound in seconds keeps no row of the quarters family (no row
  lies at a position of another unit; nothing is refused); over a bundle
  of the root and a `seconds` timeline holding `o1` at `1.0`,
  `filter(min_coord=Coordinate(1, quarters))` reads the quarters rows from 1
  on (`a1`, `a2`, `g1`, `b1`) and no seconds row, and
  `filter(min_coord=Coordinate(1.0, seconds))` reads `o1` only. **An
  `IdCoordinate` bound lies on the timeline it names**, moved onto that
  timeline's root by its ancestors' offsets, and keeps rows of that family
  only: `min_coord=IdCoordinate(1/3, quarters, "clt_a")` (root `7/3`)
  keeps `a1`, `g1`, `b1`; `min_coord=IdCoordinate(1, quarters, "clt_g")`
  (root `4`, `1 + 1 + 2`) keeps `b1`; `max_coord=IdCoordinate(1/3,
  quarters, "clt_a")` keeps `r1`, `a2`; on the bundle,
  `min_coord=IdCoordinate(0, quarters, "clt_a")` keeps no seconds row, and
  one naming a timeline no family holds (`clt_nowhere`) keeps no row.
  `counts()` gives one row per kind, sorted: `Beat` 4, `Note` 1 (the total
  `len(registry)`, 5); `counts("timeline_id")` `clt_a` 2, `clt_b` 1,
  `clt_g` 1, `clt_root` 1; an unknown or position column is refused
  (`KeyError`, naming the columns). `EventRegistry.concat` keeps each row
  once, in its family's order: `concat(clt_a.registry,
  root.registry.filter(kinds="Note"))` reads `a1`, `a2`, `g1`. A bundle's
  registry concatenates its families': with the root and the `seconds`
  timeline it reads the 5 rows then `o1`, whose `unit` is `seconds` and
  `number_type` `float`. **A group's registry is its members and their
  descendants**: a group of `clt_a` and the seconds timeline reads `a1`,
  `a2`, `g1`, `o1`, never the root's `r1` or the sibling's `b1`.
  `content("Note")` is the Note's own columns keyed by `id`: `id` `g1`,
  `pitch` 60, no position, type, name or registry column; `content("Beat",
  ids=["a1"])` the one column `id`, `a1`; `content("Beat")` reads `r1`,
  `a1`, `a2`, `b1` in the view's order. `extractions` lists the tool
  extraction once.
- **The tool extraction only where an event names it.** A root with a
  child at 1 whose one event comes from a source file (an
  `Extraction.of_source`) lists that extraction only: the child's place, a
  `Segment` row, names no extraction, so the family makes no tool
  extraction for it.
- **Tables span timelines** (`TestTablesSpanTimelines`, T136–T138 for
  tables standing alone). A timeline's events lie on its timeline: `b1`'s
  events (`fraction` quarters) joined with the seconds timeline's (`float`)
  key their rows by their timelines, the column `timeline_id` reading
  `clt_b`, `audio`, each row in its own number type (`b1`'s start the exact
  `1/10`, `o1`'s the float `1.0`); `filter(include_number_types={float})`
  keeps `o1`. A timeline is one axis, so such a table joins no timeline: a
  table spanning `fraction` and `float` quarters added to `clt_a` is
  refused (`the rows of timeline 'clt_a' lie on its axis, quarters in
  fraction; these lie on quarters in float and quarters in fraction`),
  nothing added; cast to `fraction` (`to_number_type`), every row on the
  timeline's axis, it joins: its rows are added and the timeline's events
  hold no `timeline_id` column. `get_events(include_number_types=)` is all
  or nothing on a timeline: `{fraction}` keeps the root's 5 events,
  `{float}` none.
- **Merge on attach** (`TestMergeOnAttach`). A view of `clt_a` taken
  before it joins the root reads `a1` at `1/3` and `g1` at `5/4` (its own
  family's root, `clt_a`); after the join the same view reads them at `7/3`
  and `13/4`, shifted once onto the new root, and every event row of the
  merged family names the root's tool extraction (the child's tool
  extraction is dropped, its rows renamed to the root's), `extractions`
  one. A root of no events taking such a child adopts the child's tool
  extraction. Rows added to a child after it joined are shifted likewise
  (`a3` at `2` on `clt_a` is `4` on the root).
- **The writers' read** (`TestWritten`). `written()` over the family of
  `_family()` with a user's `Segment` event `s1` (an event of kind
  `Segment` whose ID names no timeline) added to `clt_b` at 1: its rows are
  every row of the family but the three recording a child's place (the
  `Segment` rows `clt_a`, `clt_g`, `clt_b`), in the family's order, `r1`,
  `a1`, `a2`, `g1`, `b1`, `s1`, each naming the family's tool extraction
  (`s1` too, which names none in memory) and lying on the root's axis
  (`1/2`, `7/3`, `2`, `13/4`, `41/10`, `5`); its contents are keyed
  `(timeline, kind)`, the family's timelines in order and a timeline's
  kinds sorted: `(clt_root, Beat)`, `(clt_a, Beat)`, `(clt_g, Note)`,
  `(clt_b, Beat)`, `(clt_b, Segment)`, each `id` first then the columns in
  which one of its events holds a value, no position, type or name
  (`(clt_g, Note)` is `id` `g1`, `pitch` 60; `(clt_a, Beat)` the one column
  `id`, `a1`, `a2`); its extractions the tool extraction alone; its one
  family `clt_root` on quarters in `fraction`, the timelines `clt_root`,
  `clt_a`, `clt_g`, `clt_b`, and `family_of("clt_g")` that family (an
  unknown timeline `KeyError`, naming the available ones).
  `written("Note")` keeps `g1` and `(clt_g, Note)` only. A view of
  `clt_a` writes the same rows: a package holds a family whole. A
  bundle of the root and the seconds timeline writes both families, the
  seconds one after (`o1`, `audio` on seconds in `float`).
- **Update by locator** (`TestUpdate`). A timeline holds three events of
  one source file (an `Extraction.of_source` of a file of the bytes `abc`,
  SHA-256 `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad`)
  with the locators `m0`, `m1`, `m2` (`n1` at 0, `n2` at 1, `n3` at 2,
  pitch 60, 62, 64), the column `locator` moved into the registry rows (the
  events hold none). A second run over the file (another `Extraction` of
  the same source) gives `m0` with pitch 61, `m2` at `5/2` and a new `m3`
  at 3, under IDs of its own (`x0`, `x2`, `x3`): the rows read `n1`, `n2`,
  `n3`, `x3`: `n1` and `n3` keep their IDs and take the new content (pitch
  61, start `5/2`) and the new extraction; `n2`, no longer found, is kept,
  its `vanished_in` the new extraction, its extraction the old; `x3` is
  added, naming the new extraction. A row of another source keeps its place
  and is never flagged. `filter(vanished=True)` reads `n2`. Data without a
  `locator` column is refused (`an update matches rows by their locator:
  the rows give none (a column 'locator')`).
- **Update through a view** (`TestUpdate`). `update(data, extraction,
  timeline_id=None)` is taken by any view and returns the timeline it
  updates. A root `clt_root` with the child `clt_u` (at offset 2) whose
  rows `n1`, `n2`, `n3` the first run made: a second run (`m0` pitch 61,
  `m2` at `5/2`, the new `m3` at 3) through `root.registry`, naming no
  timeline, updates `clt_u` (the locators match its rows) and returns
  `"clt_u"`: `n1`, `n2`, `n3`, `x3` as above, `x3` at 3 on `clt_u`, 5 on the
  root. `update_target` answers the same without changing anything. The
  target is the timeline holding most of the matched rows: with a sibling
  `clt_v` holding rows of the same source under `m0` and `m1` and `clt_u`
  holding `m2` only, a run giving `m0`, `m1`, `m2` updates `clt_v`. Rows
  of another source never match: a run of the bytes `xyz` (another file)
  giving `m0` matches none and, given `timeline_id="clt_u"`, adds its row to
  `clt_u` under its own ID (the fallback by ID); without a timeline it is
  refused (`the rows of extraction '<ID>' (<source>) match no timeline of
  the registry, by locator or by ID (None); nothing is updated`), nothing
  changed, and `update_target` answers None.
- **A locator names one event of its source** (`TestUpdate`). A second run
  giving `m0` to two new rows (`x1`, `x2`) is refused (`locator 'm0' is
  given to 2 events of extraction 'extraction_second'; a locator names one
  event of its source`), the timeline's three events as they were; adding
  a row `y1` with the locator `m0` of the first extraction, which `n1`
  holds, is refused likewise (`... given to 2 events of extraction
  'extraction_first' ...`), and so are registry rows (`add_rows`) repeating
  a held pair; rows of another extraction may take `m0`.
- **Extractions** (`TestExtractions`). `Extraction.of_source` reads the
  file's SHA-256, its `to_row()` / `from_row()` round trip gives an equal
  extraction, its `parameters` canonical JSON text (`{"a":1,"b":[2]}` for
  `{"b": [2], "a": 1}`), and two runs over one file with one parser mint one
  ID. **The ID never depends on where the file lies**: the same bytes at two
  paths (`one/score.txt`, `two/copy.txt`) give one ID, with or without a
  `source` (the reference recorded; the path given, POSIX-spelled, when
  none): `source="score.txt"` is recorded as given. Equal extractions hash
  alike (`hash(extraction) == hash(extraction.id)`, so a set holds them). A
  source without its digest is refused. `extraction_table` has the columns
  of `EXTRACTION_SCHEMA`. **A member of a ZIP archive is read in place**: the
  bytes `abc` as the member `inner/score.txt` of an archive written by the
  test give the digest of `abc` and the ID of the same bytes on disk
  (`one/score.txt`), and the recorded source is the member's `jar:` URI,
  `jar:<the archive's file URI>!/inner/score.txt` (the archive's URI by
  `Path.as_uri()`), unless a `source` is given; a member of an archive
  opened from a file object (`io.BytesIO` of its bytes), without a
  `source`, is refused (it has no URI) and with one is read.
- **A table carries its extractions to the timeline made of it.** An
  `EventData` of two Notes naming `extraction_id` `e1`'s ID with the
  locators `m0`, `m1`, carrying `e1` (`with_extractions(e1)`), made a
  timeline by `create_timeline()`: the timeline's registry lists `e1` only,
  its rows name `e1` and hold the locators, its events hold neither column;
  `concat` of two tables carrying `e1` and `e2` carries `e1`, `e2`, each
  once. A row naming an extraction the family neither holds nor is given
  (`extraction_nowhere`) is refused (`the rows name the extraction
  'extraction_nowhere', which this family does not hold: give it with the
  rows (EventData.with_extractions)`), whether through the factory or
  `add`, nothing added.
- **Rows read from a package, through a view** (`TestAddRows`).
  `add_rows` is taken by any view: rows joining a root `clt_p` (length 4)
  with a child `clt_pc` at 1 (length 2) and a seconds timeline `cpt_s`
  (length 1.0), through the bundle of both: `p1` (a Note on `clt_pc` at
  root `5/2`, duration `1/2`, pitch 62) and `o1` (an Onset on `cpt_s` at
  2.5) join their families, `clt_pc` reading `p1` at `3/2` with its pitch,
  `cpt_s` `o1` at 2.5. **Lengths are held as for rows a timeline adds**:
  `clt_pc` (locked, as a child is) holds its row's local end, `3/2 + 1/2 =
  2`, and stays 2; `cpt_s` (unlocked) grows from 1.0 to 2.5 (the instant's
  start); a root `clt_l` locked at length 2
  refuses a row ending at 3 (start `5/2`, duration `1/2`: `Adding content
  ending at 3 exceeds timeline length 2. Timeline is locked. Use
  allow_expansion=True to override.`), its registry unchanged. **A row the family states already is
  passed over**: the same `p1` given again (same timeline, kind, `start`
  and `duration` cells) leaves the registry as it was (one `p1`), while a
  `p1` at another start is refused as an ID held (`Event id 'p1' is already
  held by an event of timeline 'clt_pc'`). **Atomic over families**: rows of
  `cpt_s` then of `clt_p` where the `clt_p` row names an extraction not
  given leave `cpt_s` without `o1`. A row naming a timeline no family of
  the view holds is refused (`KeyError`, naming the available timelines).
  The extractions given are held by each family taking rows.
- **A Box read is checked against its shape** (`TestAddRows`). A Box row
  read for a graphical timeline (`seg_2`, the timeline at x 7 of its image)
  whose `start` is 1 while its shape's `x_min` is 385 is refused with the
  Box check's message (`Box 'rect_h2' starts at 1 on timeline 'seg_2', but
  its shape starts at x 385 and the timeline stands at x 7 of its image
  (meta["box"]): its start is 378.`), and the registry and the events are
  as before (no `rect_h2`); with `start` 378 it is read (`seg_2` holds
  `rect_h2`). A registry standing alone (no host timeline) holds no length
  and checks no Box.
- **Extractions given with the rows a timeline adds** (`TestAddEvents`).
  `Timeline.add_events(rows, extractions=[e1])` with two Notes naming
  `e1`'s ID and the locators `m0`, `m1`: the timeline's registry lists
  `e1`, its rows name it and hold the locators, its events hold neither
  column; a further extraction `e2` given but named by no row is not held.
  Rows refused (the second call repeating `n1`) hold no extraction given
  with them: `e3` given with the refused rows is not held after. A row
  naming an extraction neither given nor held is refused (`the rows name
  the extraction 'extraction_nowhere', which this family does not hold:
  give it with the rows (EventData.with_extractions)`).
- **One `jar:` URI** (`TestExtractions`). `storage.registry.member_uri` is
  the one function: `loader._sources.member_uri` is the same object, and
  the URI of `inner/sc ore.txt` in an archive is
  `jar:<the archive's absolute file URI>!/inner/sc%20ore.txt`, a member
  named with a closing `/` without it.
- **Lazy content.** A content given as a callable is not called by
  `add_rows`, is called once when the timeline's events are first read and
  not again on the second read.
- **Every refusal, nothing changed after it** (`TestRefusals`):
  - a child of another number type (`Child 'clt_f' has the number type
    'float', its parent 'clt_root' has 'fraction': a child has its parent's
    unit and number type. Events of another number type reach a timeline
    only as Projections through a C-Map (add_projections).`), the root
    keeping its two children;
  - a timeline with a parent, events and all, moved to a second parent
    (`Timeline 'clt_b' is a child of 'clt_root' already: a timeline has one
    parent (it may be a member of many groups)`);
  - a family read from a package (`add_rows`, registered) attached under a
    parent (`timeline 'clt_p' holds events read from a package, positioned
    on its own axis: it cannot become a child (that would move every
    position); relate the two through a group or MatchClaims`); rows added
    with `registered=False` attach;
  - rows on another axis than the family's (`the rows of timeline 'clt_a'
    lie on its axis, quarters in fraction; these lie on quarters in float`);
  - registry rows naming a timeline the family does not hold (`KeyError`),
    another axis, an extraction not given, an ID the family holds; the
    family's extractions unchanged after each;
  - an event added with an extraction whose column the timeline's does not
    read (`velocity` `"ff"` beside `80`): refused, and the extraction is not
    held afterwards;
  - a child whose family holds an extraction under an ID the parent's holds
    with other content (`extraction_x`, parser `A` and `B`): refused
    (`extraction 'extraction_x' is held already, of other content`), the
    child still its own family's root, the parent's rows and extractions
    unchanged;
  - rows added (`add`) through a view (`TypeError`: rows join the registry
    of one family, not a view of it; `add_rows` and `update` take a view);
  - event IDs unique among a root and its descendants, one message naming
    the ID, its holder and the timeline it would join: `r1` added to
    `clt_a` (`Event id 'r1' is already held by an event of timeline
    'clt_root', so it cannot join timeline 'clt_a': event ids are unique
    among a root timeline and its descendants`), a child holding `r1` or
    its own ID as an event attached to the root (the root keeps its rows
    and children, the child its family), two rows given one ID (`Event id
    'x' is given to 2 events joining timeline 'clt_a'; event ids are unique
    among a root timeline and its descendants`);
  - a family asked for a timeline it does not hold (`No timeline 'clt_2' in
    this family. Available timelines: ['clt_1']`).
- **One rule joins columns** (`TestOneRuleJoinsColumns`,
  `storage.events.unified_type`), in `concat` and in a kind's content read
  over timelines: a root whose `tag` is the struct `{midi}` and a child
  whose `tag` is `{midi, name}` give, through `root.events.concat(a.events)`
  and through `root.registry.content("Beat")`, `tag` typed `{midi: int64,
  name: string}` (the cells `{midi: 60, name: None}`, `None` for the
  `Segment` row in the join, `{midi: 60, name: "C4"}`), each timeline still
  reading its own type; a dictionary-encoded `tag` (`"x"`) joined with a
  plain one (`"y"`) is a plain `string` column `["x", "y"]`; an `int`
  `velocity` (80) beside a `string` one (`"ff"`) is refused by `concat`,
  naming the column and the types (`the column 'velocity' holds values of
  type int64 in one table and of type string in another, which no one type
  reads; one table holds one type per column`), and so is the content of
  their kind, while each timeline keeps its own rows.
- **Rows change only through the registry.** `Timeline._events` and
  `Timeline.events` are read-only (`AttributeError`).
- **A timeline's new ID renames its rows** (`TestRename`): after `clt_a`
  takes `clt_x`, the registry's `timeline_id` reads `clt_x` for its rows
  (`a1`, `a2`; its Segment row for `clt_g` too, on the family's table) and
  the family names it as the holder of `a1`; a new ID held by another
  timeline or an event of the family (`clt_b`, `clt_root`, `r1`) is
  refused before the parent's children or rows change.

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
