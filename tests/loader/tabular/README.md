# Tabular Loader Tests

This directory contains comprehensive tests for the `timetoalign.loader.tabular` subpackage, validating the **vectorized** loading pipeline.

## Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `test_tabular.py` | 11 | Core TabularLoader, CsvLoader, TsvLoader functionality |
| `test_vectorized.py` | 13 | Vectorized pipeline integration tests |
| `test_lab_loader.py` | 30 | `LabLoader` (.lab audio-region files) — see §LabLoader below |
| `test_correctness.py` | 11 | ZERO TOLERANCE validation with real specimens |
| `test_error_handling.py` | 15 | Graceful degradation and error message clarity |
| `test_struct_columns.py` | 21 | `Field`, `ComputedField`, `ConvertedField` unit tests and JSON-to-struct parsing helper |
| `test_solo_loader.py` | `SoloLoader` round-trip against the Chopin Nocturne ``.solo`` specimen — see §SoloLoader below |
| `profile_vectorized.py` | - | Performance profiling script (not a test file) |

---

## Test Specimens

### Source/Provenance

Tests use specimens from `tests/data/score/` which are **gold standard** files from established music datasets:

| Specimen | Source | Format | Events | Description |
|----------|--------|--------|--------|-------------|
| `beethoven_woo71/WoO71.notes.tsv` | ms3 parser | ms3 TSV | 4,753 | Beethoven WoO 71 piano piece, all notes |
| `beethoven_woo71/WoO71.measures.tsv` | ms3 parser | ms3 TSV | 397 | Measure annotations for WoO 71 |
| `rachmaninoff_concerto2/.../notes.tsv` | ms3 parser | ms3 TSV | 14,315 | Rachmaninoff Piano Concerto No. 2, 1st mvt |
| `beethoven_woo71/original_file_before_cleaning.notes.tsv` | ms3 parser | ms3 TSV | 4,750 | Original uncleaned version |

### ms3 TSV Format

The ms3 (MuseScore 3) parser produces TSV files with the following columns:

```
mc  mn  quarterbeats  quarterbeats_all_endings  duration_qb  volta  mc_onset  mn_onset  timesig  staff  voice  duration  gracenote  nominal_duration  scalar  tied  tpc  midi  name  octave  chord_id
```

**Key columns for TabularLoader:**
- `quarterbeats`: Start coordinate as fraction strings ("0", "1/2", "3/4")
- `duration_qb`: Duration in quarter beats (float)
- `name`: Note name ("A4", "C#5", etc.)
- `midi`: MIDI note number (integer)

### Generated Test Data

Some tests use **dynamically generated temporary files** for isolation:

| Fixture | Format | Columns | Purpose |
|---------|--------|---------|---------|
| `csv_file` | CSV | id, start, end, event_type, name | Basic CSV with mixed instant/interval events |
| `tsv_file` | TSV | id, start, end, event_type, name | Same structure, tab-delimited |
| `minimal_csv` | CSV | start | Minimal file with only required column |

All generated test files use pytest's `tmp_path` fixture for automatic cleanup.
No manual `os.unlink()` or `tempfile.NamedTemporaryFile(delete=False)` patterns remain.

---

## Validation Logic

### Coordinate ranges are exact

`test_vectorized.py` asserts `coordinate_range()` endpoints with `==`, not
`pytest.approx`: the `np.linspace(0.1, 100.1, 10_000)` large-file case pins the
max at exactly `100.1` (NumPy's `linspace` sets the last sample to the `stop`
value, so it is the same IEEE-754 double), and the float-coordinate CSV pins
`[0.0, 4.75]` (both exactly representable). No approx assertions remain in this
file.

### Exact Coordinate Completion

A `duration_column` is completed as `end = start + duration` in the number
type the loader declares, and every computed cell carries both sides. In
`test_vectorized.py`: a fraction axis given a float duration (`7/2 + 0.75`)
gives the exact `17/4`; a seconds axis gives the IEEE sums of the stored
doubles as literals (`1.234 + 1.484 == 2.718`, `2.718 + 0.382 == 3.1`, `0.001 +
123.456 == 123.45700000000001`) with every ratio the exact mirror of its
double (`Fraction(numerator, denominator) == Fraction(value)`, denominator
positive); a quarters axis whose operands' denominators multiply past `2⁶³`
(`1/(5·2⁴⁰) + (15·2³⁹ + 1)/(15·2³⁹)`) gives exactly `(3·2⁴⁰ + 1)/(3·2⁴⁰)`.
A duration a `column_specs` producer synthesises as number cells is taken
whole, each side in the role the axis's number type gives it: on seconds, a
sub-millisecond `0.000123` (whose mirror is capped at a `2⁶²` denominator, so
`float(mirror) != value`) keeps `duration == 0.000123` and gives `end ==
0.000123` from `start = 0` and `end == 1.5 + 0.000123` from `start = 1.5`, the
IEEE sums. Reading the mirror as the number would store
`0.00012300000000000006`.

The how-to's configuration (a `TsvLoader` with `start_column =
"quarterbeats"` written as ratios, `duration_column = "duration_qb"` written
as decimals, a quarters/`fraction` axis, `midi`/`staff`/`voice` as `int`) runs
in `test_vectorized.py` on a five-row TSV written to `tmp_path`. Its starts
`0`, `1/2`, `0.7`, `1/3`, `3/2` and durations `1.0`, `0.5`, `0.2`, `0.25`,
`0.1` give the ends `1`, `1`, `16212958658533785/2⁵⁴`, `7/12` and
`57646075230342349/2⁵⁵` exactly: `0.7 + 0.2` cross-multiplies past `2⁶³`
and must still be exact. A sixth row `1129/12 + 0.1`, whose exact end has a
numerator past `2⁶³ − 1` in lowest terms, makes `load` raise a `ValueError`
whose whole message is asserted: it names the duration column, the axis
(`fraction`, `quarters`), the row (`row 5`), the operands and the result.

### Computed-field formulas read numeric literals as numbers

A `ComputedField` formula is a left-to-right chain of operands joined by
`+ - * /`, an operand being a column reference (`n`, `rect.x`) or a number.
`TestComputedFieldNumbers` checks that a number in a formula is that number
and never a column name, on the table `n = [3, 4]` (int64), `x = [1.0, 2.0]`
(double) and a struct column `rect = [{x: 10}, {x: 20}]`:

- `n * 2` gives `[6, 8]` as an int64 array: an integer literal is an
  integer, so it does not turn an integer column into doubles (the number
  type of the column survives the formula).
- `n / 2.5` gives `[1.2, 1.6]` and `n * 1.5` `[4.5, 6.0]`, double arrays: a
  decimal literal is a double.
- `2 * n` gives `[6, 8]`: a number may come first.
- `/` is PyArrow's `divide`, which divides integers as integers: `n / 2`
  gives `[1, 2]`, exactly as the column formula `n / m` over `m = [2, 2]`
  already did; a decimal literal (`n / 2.0`) divides in doubles.
- `rect.x - 1` gives `[9, 19]`: a struct path followed by a literal.
- The column references are read as before: `n * n` gives `[9, 16]`,
  `n + x` `[4.0, 6.0]`, and a column whose name starts with a digit
  (`2nd = [5, 6]`) is still a column: `2nd + n` gives `[8, 10]`.
- Operands and operators alternate, starting and ending with an operand.
  `n % 2`, `n % n`, `n 2`, `- n`, `n * * 2` and `n +` are refused with a
  `ValueError` naming the formula. `%` is no operator, and the tokenizer
  skips a character it does not know, so without this check `n % 2` would
  answer `n` itself, silently dropping the `2`; `n % n` did answer `n`
  before, and `- n`, `n +` and `n * * 2` failed with an `IndexError`. No
  formula of the documented syntax is affected.
- A formula without any column reference (`2`, `2 * 3`) names nothing to
  compute over and is refused with a `ValueError` naming the formula; it
  raised a `KeyError` (`Column '2' not found`) before, so no formula that
  worked changes its answer.

Before the fix the token pattern matched every digit run as a column name,
so `n * 2` raised `KeyError: "Column '2' not found ..."` and `n * 1.5`
`"Column '1' ..."`; the exact values above are what the arithmetic gives
and only a formula reading the literal as a number reaches them.

### How We Know the Parser is Correct

1. **EXACT COUNT VALIDATION**: Event counts are validated against the source file line counts minus header. For example:
   - `WoO71.notes.tsv` has 4,754 lines -> 4,753 events (EXACT)
   - `WoO71.measures.tsv` has 398 lines -> 397 events (EXACT)

2. **COORDINATE RANGE VALIDATION**: Start/end coordinates are validated against known musical structure:
   - First note starts at quarterbeat 0 (EXACT)
   - Coordinate range matches the piece length in quarter beats

3. **FRACTION ROUND-TRIP**: For fraction coordinates, we verify:
   ```python
   Fraction(numerator, denominator) == value  # Within float precision
   ```

4. **TEMPORAL TYPE INFERENCE**:
   - Events with valid `duration_qb` -> "interval"
   - Events with null `duration_qb` or null `quarterbeats` -> "instant"
   - Validated: all 4,753 events are intervals and none is an instant (EXACT, `test_correctness.py`; every row of the TSV has a `duration_qb`)

5. **ZERO ITERATION GUARANTEE**: Validated via monkey-patching:
   ```python
   pd.DataFrame.__iter__ = lambda self: raise AssertionError("ITERATION DETECTED!")
   ```

### ZERO TOLERANCE Validation Policy

Per the ZERO TOLERANCE validation policy, all tests use **EXACT** assertions:

```python
# CORRECT
assert len(loader.events) == 4753  # Exact count from gold standard

# WRONG - FORBIDDEN
assert len(loader.events) >= 4000  # Approximate count
assert len(loader.events) > 0      # "At least some events"
```

---

## Performance Benchmarks

### Profiling Results (2026-01-31)

| Specimen | Events | Load Time | Throughput | Zero-Iter |
|----------|--------|-----------|------------|-----------|
| WoO71.notes.tsv | 4,753 | 0.029s | 166,074/s | PASS |
| WoO71.measures.tsv | 397 | 0.008s | 49,966/s | PASS |
| Rachmaninoff.notes.tsv | 14,315 | 0.077s | 185,830/s | PASS |
| original_file_before_cleaning.notes.tsv | 4,750 | 0.029s | 161,816/s | PASS |

**Summary:**
- **Average throughput:** 140,921 events/sec
- **Target (20,000 events/sec):** ACHIEVED (7x over target)
- **Zero iteration:** ALL PASS

### Running the Profiler

```bash
cd timetoalign
python -m tests.loader.tabular.profile_vectorized
```

**Note:** The profiler loads specimens from `tests/data/score/` (same as the test suite).

See `PROFILING_REPORT.md` for full details.

---

## Known Limitations

## ms3 coverage ownership

The tabular package no longer owns an ms3 loader. The exact ms3 gold-vector
tests that remain here use a local `TsvLoader` subclass with explicit generic
column configuration; score-facet parsing and pitch enrichment are tested by
the score package's `Ms3Loader` suite.

### Configured TsvLoader

1. **Null start coordinates stay null**: If `quarterbeats` contains empty values, the fraction parser preserves those positions as null coordinate structs. The corresponding events are treated as instants when no valid interval endpoint can be computed.

2. **Sibling-file variants silently skip missing columns**: `column_specs` lookups against the source DataFrame return `None` when a named column is absent (a measures.tsv lacks the `staff` / `voice` / `chord_id` columns that notes.tsv carries).  Positional specs (integer source keys) are NOT skipped — a missing positional index is a hard `IndexError`.

3. **Fraction parsing requires specific format**: Accepts "num/den" strings or pure integers ("0", "1", "42"). Does NOT accept:
   - Decimal strings ("0.5")
   - Multiple slashes ("1/2/3")
   - Non-numeric strings ("invalid")

### CsvLoader / TsvLoader

1. **Delimiter must match file format**: Using CsvLoader on a TSV file (or vice versa) will fail with "column not found" error because the entire line becomes a single column.

2. **Empty files return empty EventData**: Loading a header-only file succeeds with 0 events (does not raise error).

---

## Discrepancies Between Loaders

### Configured TsvLoader vs Generic TsvLoader

| Aspect | Configured TsvLoader | TsvLoader |
|--------|-----------|-----------|
| Start column | `quarterbeats` | `start` |
| End column | Computed from `duration_qb` | `end` |
| Coordinate type | `NumberType.fraction` | `NumberType.float` |
| Time unit | `TimeUnit.quarters` | `TimeUnit.seconds` |
| Default event type | `"Note"` | `"Event"` |

The score package's `Ms3Loader` owns ms3 facet dispatch. Generic tabular users
can provide equivalent column mappings through `TsvLoader` configuration.

### Null Handling

| Scenario | Behavior |
|----------|----------|
| Null `quarterbeats` (start) | Preserved as a null coordinate; the event is instant when no endpoint is available |
| Null `duration_qb` (duration) | Event becomes "instant" (no end coordinate) |
| Null `end` column | Event becomes "instant" |

---

## Test Coverage Requirements

Each loader class requires **15+ tests** covering:

1. **Unit Tests (5+):** Happy path, empty input, single element, nulls, invalid format
2. **Integration Tests (3+):** Real specimen loading with EXACT counts
3. **Performance Tests (2+):** Small and large file benchmarks
4. **Error Handling Tests (3+):** Missing columns, malformed data, type mismatches
5. **Edge Case Tests (2+):** Mixed coordinate types, temporal type inference

Current coverage for `TabularLoader` family: **47 tests** (exceeds requirement)

---

## Running Tests

```bash
# Run all tabular tests
cd timetoalign
python -m pytest tests/loader/tabular/ -v

# Run only correctness tests
python -m pytest tests/loader/tabular/test_correctness.py -v

# Run only error handling tests
python -m pytest tests/loader/tabular/test_error_handling.py -v

# Run profiling (not a pytest test)
python -m tests.loader.tabular.profile_vectorized
```

---

## Adding New Tests

When adding tests for new specimens:

1. **Document provenance**: Add the specimen to the table above with source and expected event count
2. **Determine EXACT counts**: Count lines in file (`wc -l file.tsv`) and subtract 1 for header
3. **Use ZERO TOLERANCE assertions**: `assert len(events) == EXACT_COUNT`
4. **Update this README**: Add any new limitations or discrepancies discovered

When adding tests for new loaders:

1. Follow the test coverage requirements (15+ tests)
2. Profile with real specimens and document results
3. Add to the "Discrepancies Between Loaders" section if behavior differs

---

## LabLoader

### Specimens

Three synthetic tab-separated `.lab` files in
`tests/data/fixtures/lab/` (audio-region label files, `start<TAB>end<TAB>label`
per line, no header):

| Specimen | Events | Kind | Coordinate range |
|----------|--------|------|------------------|
| `regions.lab` | **6** | intervals (Intro, Verse, Chorus, Verse, Chorus, Outro) | `[0.0, 15.2]` |
| `beats.lab` | **6** | intervals all labelled `Beat` | `[0.0, 3.0]` |
| `instants.lab` | **4** | `start == end` events labelled `Downbeat` | `[0.0, 3.0]` |

### Validation logic (zero tolerance)

Event counts are **exact**, per the ZERO TOLERANCE policy — the loading smoke
tests assert `len(loader) == 6` / `6` / `4` (never `len(loader) > 0`), agreeing
with the dedicated `TestRegionsLab` / `TestBeatsLab` / `TestInstantsLab`
count assertions and with `test_from_file_convenience` (`== 6`). Coordinate
ranges are asserted as exact `Fraction`/float endpoints (`15.2`, `3.0`, `0.0`).
The default unit is `seconds` and the default event type is `Region`.

**Completed durations on a seconds axis.** The fixtures above hold values whose
ratios are small; real annotation files hold three-decimal seconds, whose
mirrors carry denominators up to `2⁶²` (`0.001`). `TestNonDyadicSeconds` writes the three
lines `0.000 1.234 N`, `1.234 2.718 C:maj`, `2.718 3.1 G:maj` under `tmp_path`
and loads them with `LabLoader.from_file`. For every row of `start`, `end` and
`duration`: `denominator > 0` and `Fraction(numerator, denominator) ==
Fraction(value)` exactly; the durations are the IEEE differences of the stored
doubles, as literals: `1.234`, `1.484`, `0.3820000000000001`.

## SoloLoader

### Specimen

`tests/data/performance_precision/Chopin Nocturne Op. 9 No. 2.solo` — a
header-less tab-separated file of 2494 lines (one note-on or note-off
event per line) produced by a performance-analysis tool.  Six columns:

| Index | Name             | Type         | Example  |
|-------|------------------|--------------|----------|
| 0     | `position`       | "M+frac"     | `1+3/8`  |
| 1     | `duration`       | rational     | `1/2`    |
| 2     | `channel`        | int          | `90`     |
| 3     | `pitch`          | int          | `63`     |
| 4     | `velocity`       | int          | `80`     |
| 5     | `note_id`        | alphanumeric | `nwp1wrk`|

### Validation logic (decided in advance — document-before-implement)

1. **EXACT row count**: 2494 events, matching `wc -l` on the source.
2. **Sentinel rows asserted exactly**:
   - Row 0: `position == "0+11/8"`, `duration == "0/1"`, `channel == 90`,
     `pitch == 70`, `velocity == 80`, `note_id == "n1b8xktz"`.
   - Row 1: `position == "1+0/1"`, `duration == "1/8"`, `pitch == 70`,
     `note_id == "n1b8xktz"` (the note-off paired with row 0).
   - Row 12: `position == "1+3/8"`, `duration == "1/2"`, `pitch == 58`,
     `velocity == 0`, `note_id == "n1fst1u3"` (a note-off pairing).
3. **`column_specs` shape decision**: the composite `position` column
   splits into two parts named `measure_number` (default name derived
   from `MeasureNumberField`'s class name minus the `Field` suffix in
   snake_case) and `mn_onset` (explicit `name="mn_onset"`).  The
   loader surfaces both at the top level of the EventData table so
   `start_column = "mn_onset"` resolves correctly.  The composite
   `position` column also remains as an opaque struct for users who
   need the original packing.
4. **Step 2 promotion**: the `pitch` column is decorated with
   `EnharmonicPitchField` metadata; it is repacked from `int64` into
   the `{midi_number: int64}` struct shape expected by
   `EnharmonicPitchField`.  `note_id` is decorated with `IdField`
   metadata and repacked into `{value: string}`.
5. **`get_field` round-trip**: `events.get_field(EnharmonicPitch)`
   returns an `EnharmonicPitchField` of length 2494; indexing into
   it yields `EnharmonicPitch` scalars (e.g. `[0] -> EnharmonicPitch(B♭4)`
   for MIDI 70).  `events.get_field(Id)[0]` yields
   `Id(value='n1b8xktz')`.
6. **The locator is the data row and the note's `note_id`**: row 0
   `row:0 key:n1b8xktz`, row 1 (its note-off, the same `note_id`)
   `row:1 key:n1b8xktz`, row 12 `row:12 key:n1fst1u3`; the 2494 locators
   are distinct although every `note_id` is written twice, because the
   row leads.
7. **Out-of-scope** (explicitly NOT asserted): the canonical `start`
   coordinate.  Measures are second-order time units; the loader
   currently routes `start_column = "mn_onset"` and consumers must
   reconcile measure-number into flat-quarter coordinates against a
   `MetricMap` separately.

### Composite-part naming convention

When the iterable form of `CompositeFieldParser(parts=[...])` is given
an unnamed entry that resolves to a paired SemanticField subclass
(e.g. `MeasureNumberField`), the resulting sub-field's default name is
the class name minus the `Field` suffix, snake-cased:
`MeasureNumberField` → `measure_number`.  When the resolved entry is a
DataField blueprint or a `FieldParser` instance with an explicit
`name=`, that name is used directly.  Otherwise the fallback is
`f"part_{i}"`.  This is exercised end-to-end in
`test_solo_loader.py` (the `position` column splits into
`measure_number` + `mn_onset`).
