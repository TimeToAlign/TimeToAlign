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
through the library's one minter (`core/ids.py::mint_ids`), a Projection
object stands for its event row, and a Projection's reference column is typed
by declaration rather than by the rows of one batch.

**Contract and what proves it:**

- `mint_event_ids(event_types, taken)` (public, also exported from
  `timetoalign.storage`, the same function) mints one ID per type, in order: the
  prefix is the event type in lower case (a missing type counts as
  `Event`), N continues above the highest number held under that prefix,
  so a gap is never filled. `["Note", "Projection", "Note", None]` against
  `{"note_1", "projection_2", "note:000005"}` gives exactly `["note_2",
  "projection_3", "note_3", "event_1"]` (`projection_1` is free but lies
  below `projection_2`, so it is not given; `note:000005` is no minted ID).
- `EventData.from_dicts` mints above the IDs its rows give: a row given
  `note_4` and one without an ID end with `note_4` and `note_5` (no
  repeated ID within the table); `from_arrays` without an `id` column mints
  `projection_1`, `projection_2`, `note_1` for the types `Projection,
  Projection, Note`.
- `EventData.from_dicts` takes a `Projection` object, alone or in the list
  beside dicts, as its event row: `Projection(start=Coordinate(2, samples),
  duration=Duration(3, samples), projection_of=EventReference(clt_6, x1))`
  and a Note dict give the IDs `projection_1`, `note_1`, the event type
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
the table's axis. One mechanism serves every event type whose field holds
all of its scalar's properties (`Box`, `Projection`, `Event`); nothing in
it names a type.

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
the column of the property's own name; those columns ride in the composed
struct under their own names, so the scalar's `from_row` reads them, and
the stored table still holds the pitch once. A subtype is read on the rows
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
- **A Note without a pitch column is a rest.** The three-row table above
  (`user_units`, `fraction`): `get_field(Note)` is `[None, Note(start 0,
  end 1, duration 1), None]`, each number a `Fraction`, `pitch` `None`.
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
- **`extend` and `concat` join rows of another number type, re-expressed
  in the receiving table's** (`TestJoiningNumberTypes`). The rule is the one
  every value landing on an axis obeys: float → fraction as the double's
  exact ratio (`Fraction(x)`, no denominator limiting), fraction → float as
  the nearest double (`float(r)`), onto `int` nearest with ties to even
  (`express_as`). What proves it is the cells, read member by member, and
  the joined table's declared number type and field metadata, for each
  direction:
  - a `fraction` quarters table (`a` at `[1/3, 2/3)`) absorbing a `float`
    one (`b` at `0.1` lasting `0.2`, an instant `c` at `0.5`): `b`'s start
    is `3602879701896397/36028797018963968` (`Fraction(0.1)`), its
    duration `3602879701896397/18014398509481984` (`Fraction(0.2)`), its
    end their exact sum, `10808639105689191/36028797018963968`
    (`Fraction(0.1) + Fraction(0.2)`, not `Fraction(0.1 + 0.2)`: start and
    duration are the stored pair of an event and the end is derived from
    them, so on an exact table the joined row's end is its start plus its
    duration, exactly, and the row stays consistent under the exact checks
    the table reads it through); `c` stays an instant at `1/2`; every
    coordinate field declares `fraction`;
  - the reverse, the `float` table absorbing the `fraction` one: `a`'s cells
    are `0.3333333333333333` and `0.6666666666666666`, each mirrored by its
    double's exact ratio (`6004799503160661/18014398509481984`), the
    duration `0.3333333333333333` (a float table keeps each value
    re-expressed, the doubles agreeing within its tolerance);
  - an `int` table absorbing `fraction` rows `[5/2, 7/2)` and `[3/2, 9/2)`
    and a `float` row `[2.5, 3.5)`: start and duration are re-expressed,
    ties to the even neighbour, and the end is their sum, so the rows keep
    their durations: `5/2` → start 2, duration 1, end 3; `3/2` → start 2,
    duration 3, end 5; `2.5` → start 2, duration 1, end 3; every cell
    `denominator` 1 (re-expressing the ends instead would give `[2, 4)`
    three times, durations 2, and lose the stated durations 1 and 3); and a
    `fraction` table absorbing the `int` rows keeps them as `n/1`.
  `extend` and `concat` give the same table; the receiving table's own rows
  are unchanged. A table of another unit is still refused (`Unit mismatch:
  quarters vs seconds`), as no number can be re-expressed across units
  without a C-Map. The joins the factory makes (`flatten=True`,
  `group_by=`) are proven end to end in `tests/timelines/test_factory.py`
  (`TestStoresOfTwoNumberTypes`).
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
  `EventData.from_parquet`; `concat` and `extend` join tables that entered
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
  now it is the same Projection (`projection_1`, one event).
- **Cells are reduced at the doors only** (`TestReducedAtTheDoorsOnly`). A
  table derived from tables already held (rows selected by `filter` or
  `where`, IDs prefixed, tables joined by `concat`, a column retyped, a
  timeline's own copy, the rows `get_events` rebuilds from a timeline and
  its children) holds cells that were reduced when they entered, so
  reducing them again costs a pass over every coordinate column and changes
  nothing. The proof counts the calls of the one reducing function,
  `storage.events._in_lowest_terms`, replaced for the test by a counting
  wrapper around it: `EventData(table)` and `EventData.from_parquet` call it
  once each (the doors); `filter`, `prefix_ids`, `concat`, a timeline made
  by `create_timeline`, and `get_events(temporal_type=, event_type=,
  min_coord=, max_coord=)` on a flat timeline and on a parent with a child
  call it zero times. A table of no rows is not given to it at all, as it
  holds no cell to reduce (a new timeline starts with one). The results are
  asserted too (IDs and the reduced
  cells), so a path that skipped the reduction and then returned wrong rows
  would not pass on the count alone; the reduction tests above pass
  unmodified, which is what shows the doors still reduce.

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

- `EventData.from_dicts` of one `Note` `[0, 1)` quarters, then
  `create_timeline(uid="notes")`: a `ContinuousLogicalTimeline` with id
  `notes` holding one event.
- `SingleStore` of the same data, then `create_timeline(uid="clt_1")`: a
  `ContinuousLogicalTimeline` with id `clt_1` whose events are
  `["clt_1:note_1"]` (the store prefixes the event ID with the timeline's).

The child's standard output must be exactly those lines, and its exit code 0;
before the fix both calls raised `RuntimeError: Timeline creation requires
timetoalign.timelines`.
