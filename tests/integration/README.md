# Integration Tests: Loader to Timeline Pipeline

This directory contains integration tests validating the complete pipeline from
loading MusicXML/MIDI files through EventStores to Timeline creation.

## Validation Policy

**ZERO TOLERANCE VALIDATION POLICY**

1. **EXACT COUNTS REQUIRED**: All test assertions use exact expected values
2. **GOLD STANDARD IS AUTHORITATIVE**: MS3 TSV files are the ground truth
3. **LOADER PARITY IS REQUIRED**: Different loaders parsing the same file must produce identical counts
4. **NO TOLERANCE WITHOUT ROOT CAUSE**: Any mismatch indicates a bug

## Gold Standard Reference

### Chopin Op.10 No.3

| Metric | Exact Count | Source File |
|--------|-------------|-------------|
| Notes | **498** | `tests/data/vienna_1x22/ms3/chopin_op10_no3.notes.tsv` (499 lines - 1 header) |
| Measures | **22** | `tests/data/vienna_1x22/ms3/chopin_op10_no3.measures.tsv` (23 lines - 1 header) |

These counts are derived from human-verified MuseScore annotations and serve as
the authoritative reference for all loader implementations.

## Test Categories

### 1. Structural Validation (`test_*_structure`)

**What we test:** Timeline hierarchy is correctly created.

- Parent timeline has expected ID
- Children exist for each non-empty store
- All children are at offset 0
- Children are locked after being added

**Why this matters:** The TTA model specifies that children maintain their own
0-based coordinate system. Structural correctness is prerequisite for any
downstream operations.

### 2. Event Count Validation (`test_*_count_exact`)

**What we test:** Event counts match gold standard EXACTLY.

- `test_partitura_notes_count_exact`: 498 notes
- `test_partitura_measures_count_exact`: 22 measures
- `test_music21_notes_count_exact`: 498 notes

**Why this matters:** Event counts directly impact research results. A count
mismatch of even 1 event could indicate:
- Missing grace notes
- Duplicated tied notes
- Parsing errors

**Validation method:** Direct comparison with `wc -l` count on MS3 TSV files.

### 3. Cross-Loader Consistency (`TestCrossLoaderConsistency`)

**What we test:** Different loaders produce identical results for core data.

- `test_partitura_vs_music21_note_count_identical`: Both must equal 498
- `test_partitura_vs_music21_core_children_identical`: Core children (notes, measures, controls) match

**Why this matters:** Users should get consistent results regardless of which
loader they use. Any difference in core data indicates a bug.

**Documented differences:**

| Aspect | Partitura | Music21 | Explanation |
|--------|-----------|---------|-------------|
| Annotations | 0 (not parsed) | 5 (TextExpressions) | Different parsing scope |
| Implicit rests | Not created | Tracked but not stored | Music21 creates rests; we filter them |

### 4. Filter Validation (`TestScoreLoaderWithFilters`)

**What we test:** `include_stores` and `exclude_stores` work correctly.

- `test_filter_notes_only`: Only notes child created
- `test_exclude_controls_and_annotations`: Controls and annotations excluded

**Why this matters:** Custom timeline creation relies on filters working correctly.

**Child identity.** A store-built timeline names each child after its store
(`notes`, `measures`, …) and mints the child's id in its own manifest (`clt_2`,
`dlt_2`, …): a store name is a label, not an id. These tests therefore look a
child up by its name, and the event ids keep the store name as their prefix.

### 5. MIDI Timeline Structure (`TestMidiLoaderToTimeline`)

**What we test:** MIDI loaders build the expected timeline structure.

- `test_performance_midi_to_timeline`: `PerformanceMidiLoader` on
  `supra_raw.mid` yields **exactly 2** children — `notes` and `controls`
  (`assert timeline.n_children == 2`, an exact count, not `>= 1`).
- `test_performance_midi_notes_child_has_events`: the `notes` child is
  non-empty. Its exact event count is specimen-dependent and intentionally left
  as `> 0` (the piano-roll note total is not part of the gold standard).

## Beethoven WoO71 Gold Standard (`test_beethoven_woo71.py`)

A second, larger gold specimen (Beethoven Piano Trio WoO71) with exact TSV
counts: **4753** notes, **397** measures, and **8** volta notes with null
`quarterbeats`. All of these are asserted with exact `==`.

**Retained `pytest.approx` — documentary scale ratio.** One assertion,
`test_beethoven_has_more_notes_than_chopin`, records the *order-of-magnitude*
size ratio versus Chopin as `ratio == pytest.approx(9.54, rel=0.1)`. The gold
`9.54` is a rounded human-readable figure, **not** the bit-exact quotient
`4753 / 498 = 9.5441…`; the loose `rel=0.1` tolerance is the point of the check
(the real correctness gate is the exact `4753` count and the `ratio > 5`
assertion just above it). It is therefore kept as `approx` rather than pinned
to a false `== 9.54`.

## Test Data Specimens

| File | Type | Source | Purpose |
|------|------|--------|---------|
| `vienna_1x22/Chopin_op10_no3.musicxml` | MusicXML | Vienna 4x22 | Primary validation |
| `data/midi/score/beethoven_op18.mid` | Score MIDI | OMR Groundtruth | MIDI loader testing |
| `data/midi/performance/supra_raw.mid` | Performance MIDI | Supra Rolls | Performance MIDI testing |

## How to Run

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run with specific marker
pytest tests/integration/ -v -k "partitura"

# Run with coverage
pytest tests/integration/ --cov=timetoalign.timelines.factory --cov=timetoalign.loader.bundle
```

## Known Loader Differences

### Music21 vs Partitura

| Behavior | Music21 | Partitura | Resolution |
|----------|---------|-----------|------------|
| Implicit rests | Creates them | Does not | Music21 loader filters them out (2025-01-26) |
| Text annotations | Parses TextExpressions | Does not parse | Documented as scope difference, not a bug |

These are **not bugs** but documented differences in parsing scope. Core event
data (notes, measures, controls) must still match exactly.

## Adding New Tests

When adding tests for new specimens:

1. **Establish gold standard**: Use authoritative source (MS3, manual count)
2. **Document exact counts**: Add to the table above
3. **Use exact assertions**: `assert count == EXACT_VALUE`, never `assert count >= X`
4. **Explain validation**: Document how the gold standard was derived
