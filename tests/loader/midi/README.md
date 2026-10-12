# MIDI loader tests

The MIDI tests cover the two supported interpretations of MIDI data:

- `PerformanceMidiLoader` uses mido for event-stream data and returns `MidiEventData`.
- `ScoreMidiLoader` uses Partitura for quantized score data and returns `ScoreMidiEventData`.

`MidiEventData` adds `pitch`, `velocity`, `channel`, `track`, `control`, `value`, and `program`.
`ScoreMidiEventData` additionally adds `voice`, `staff`, and `part_id`. Performance stores must not
contain those three score-only columns.

## Test files

| File | Remaining coverage |
|---|---|
| `test_performance.py` | Performance-file event-type and extent golds, plus synthetic loader coverage. |
| `test_harmonization.py` | Canonical same-file comparisons between performance and score loaders. |
| `test_score.py` | Beethoven score loading, empty-file failure, and the wider score schema. |
| `test_store.py` | Base and score event-data schemas, nullability, construction, and inheritance. |
| `test_bundle.py` | `MidiStore` splitting, empty stores, extension, mapping protocol, summaries, and timelines. |
| `conftest.py` | Downloaded MIDI specimen paths and synthetic shared event data. |

## Performance loader

`test_performance.py` also pins unique raw-performance-file behavior that score parsing cannot
cover.

- A C4 note-on at tick 0 and note-off 480 ticks later becomes one note with pitch 60 and duration
  480.
- With controls enabled, the concrete store type is exactly `MidiEventData`, contains all seven
  performance columns, and omits `voice`, `staff`, and `part_id`.
- A program change at 0, note at 240, control change at 480, and note end at 960 produce starts
  `[0.0, 240.0, 480.0]`, ends `[None, None, 960.0]`, coordinate range `(0.0, 960.0)`, timeline
  length 960, and group ID `perf:dlt_1`.
- `supra_raw.mid` produces 30,096 total events: 30,092 notes and 4 control changes. Its coordinate
  range is `(0.0, 277776.0)`, its created timeline has length 277776, and it constructs a
  `TimelineGroup` named `supra`.

## Same-file harmonization

`test_harmonization.py` is authoritative for downloaded performance files because it loads each
input with both loader implementations.

| Input | Exact assertions |
|---|---|
| `supra_raw.mid` | Both loaders produce 30,092 notes and maximum end tick 277,776.0. |
| `Chopin_op10_no3_p01.mid` | Both loaders produce 451 notes and identical pitch histograms; MIDI 59 is most frequent with 50 occurrences. |

Performance MIDI can contain controls that score parsing omits. Harmonization therefore compares
note events rather than total raw message counts.

## Score mido fast path

The optional ``ScoreMidiLoader(parser="mido")`` path parses note intervals directly from the MIDI
event stream when structural score analysis is not requested. Its tests pin the raw-file note count,
first and last note start ticks, and pitch sum against ``PerformanceMidiLoader`` with controls and
program changes disabled, because both paths then apply the same message-level pairing rules. A
quantized score MIDI test separately pins each parser's note count, pitch multiset, and first-note
coordinates: pitch content must agree, while partitura may use structurally derived coordinates.
The tests also pin the mido tick resolution and exact narrow ``MidiEventData`` column list,
explicitly excluding voice, staff, and part identifiers. Structural parser options are rejected
for this path.

## Score loader and stores

The Beethoven score test pins 3,751 events, non-null tick resolution, Partitura metadata with four
parts, and populated pitch data. An empty MIDI file must raise `EOFError`. `ScoreMidiEventData`
must contain the three score-only fields, while both event-data classes retain their declared
nullable fields and can be constructed from dictionaries.

`MidiStore` tests require notes and controls to split into the correct child stores while retaining
metadata. Empty stores have empty data and default metadata. Extension merges each category and
updates metadata. Mapping keys, iteration, items, lookup, membership, and length use the canonical
`notes`, `controls` order. Summary counts are exact. Timeline conversion creates the two expected
children at offset zero with the original note/control counts; each child is named after its store
and carries a minted id, so the tests find it by name.

All exact counts, ticks, schemas, IDs, and extents use equality; there are no `pytest.approx` calls
in the MIDI-loader tests.

## Running the tests

From the repository root:

```bash
/home/laser/miniconda3/envs/timetoalign/bin/python -m pytest --runslow tests/loader/midi
```
