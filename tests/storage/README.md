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

### `test_store_summary_contract.py` - store summaries and cards

Every store a loader exposes (single, dict, score, MIDI, graphical,
alignment) returns the exact `summary()` dict pinned for its specimen, and a
non-empty notebook card. The alignment store's facts are its events, its
C-Maps, its domains and its table names (`event_count`, `cmap_count`,
`domains`, `store_names`): it holds no MatchClaims, which live in the
MatchClaim table, so no match count appears among them.
