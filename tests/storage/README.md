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

### `test_events.py` - One axis per table: number type kept, cells reduced

**Purpose:** An `EventData` reads every cell in its one unit and number type,
and holds one spelling of each value. Three ways in broke that.

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
- **`extend` and `concat` refuse rows of another number type.** They
  checked the unit only, so a `fraction` table absorbed a `float` table's
  rows and read their dyadics as authored ratios. Both now raise the
  `ValueError` they raise for a unit mismatch, with the message `Number type
  mismatch: fraction vs float`, and the table is unchanged (one row). A
  table of the same unit and number type still joins (two rows).
- **Cells are reduced on entry.** A ratio stated as `2/4` is accepted and
  held as `1/2` at every door that takes whole cells: `EventData(table)`,
  `EventData.from_parquet`, and therefore `concat` (which builds through the
  constructor) and `extend` (whose argument was built through it). Each is
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
