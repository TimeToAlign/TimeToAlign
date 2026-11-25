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

## Score velocities

A MIDI note states its velocity in its note-on message, and both score parsers write that value
into `velocity`; no velocity is invented, and none is left out where the file states one.
Partitura's score parse pairs note-ons with note-offs but drops the velocity, so the default path
reads it from the same file in the same pairing and joins it to Partitura's notes by note number
(Partitura numbers its notes `n0`, `n1`, … in its own pairing order), checking that the note of
that number has the row's onset, pitch and duration.

The expected values come from the bytes, never from the loader: the test reads each file with
`mido`, pairs every note-on (velocity above 0) with the next note-off or zero-velocity note-on of
the same track, channel and pitch, and takes `(start tick, pitch, duration in ticks, velocity)`
of every note. Each parser's notes, read as the same tuple, must equal that multiset exactly, so
a velocity joined to the wrong note fails as surely as a missing one. The velocities themselves
are also pinned as written in the files:

| File | Notes | Velocities |
|---|---|---|
| `beethoven_op18.mid` | 3,751 | 100 on every note |
| `beethoven_mtd.mid` | 16 | 80 on every note |
| `rachmaninoff_piano.mid` | 111 | 112 on 98 notes, 127 on 13 |

`rachmaninoff_piano.mid` is the one that tells a correct join from a wrong one: its two values are
spread over notes of different pitches and positions. The Beethoven quartet is checked on both
parsers (`parser="mido"` reads the event stream directly); the two small files on the default
parser.

**Quantized times.** With `quantization_unit`, Partitura moves every note-on and note-off time to
the nearest multiple of the unit (ties to the even multiple) before pairing, so a note's onset is
its quantized note-on time and its duration the quantized note-off time minus the quantized onset.
The join must apply the same quantization, or no quantized note would find its number. The test
loads `rachmaninoff_piano.mid` with `quantization_unit=10` and quantizes the `mido` reading in the
test itself (`10 * round(tick / 10)` on each message time, Python's `round` ties to even as
Partitura's does): the loaded `(onset, pitch, duration, velocity)` multiset must equal that one
exactly, velocities 112 on 98 notes and 127 on 13 as unquantized.

**A fractional unit is quantized as Partitura quantizes it.** Partitura accepts a unit that is not
an integer and then returns float times (`2.5 * round(tick / 2.5)`, e.g. 12.5), which its note
array stores as integer divisions. The join compares the file's notes at the times `quantize`
returns, unaltered: the loader does not make them integral itself, so a note whose quantized onset
or duration is fractional does not equal Partitura's integer row and keeps an empty velocity, with
the one warning. Truncating the times to integers in the loader would instead join such notes by a
rule Partitura does not state. The test loads `vienna_1x22/Chopin_op10_no3_p01.mid` with
`quantization_unit=2.5` and pins the result: 288 of the 451 velocities are null and exactly one
`UserWarning` names `288 of 451 notes` (truncation gives 99 of 451).

**A note that cannot be joined is read without a velocity.** The file is valid MIDI and Partitura
reads it; the join only reproduces Partitura's numbering, which a Partitura release could change.
Such a change must not make the file unreadable, nor give a note another note's velocity: every
note whose number does not name a note of equal onset, pitch and duration keeps an empty velocity,
and one `UserWarning` per file names the first such note and points to `parser="mido"`. The test
simulates a renumbering by replacing the file's reading with the same list rotated by one place
(note `n{i}` then meets the file's note `i + 1`) on `rachmaninoff_piano.mid`, where no two
consecutive notes agree in onset, pitch and duration. It asserts:

- every one of the 111 velocities is null, while `(start, pitch, duration)` of every note still
  equals the `mido` reading (the file is read whole, only the velocities are left out);
- exactly one warning is raised, of class `UserWarning`, with the exact text naming the count
  (`111 of 111 notes`), the file's path and Partitura's first note `'n0'` (onset 0, pitch 24,
  duration 106); the loaded row `n0` is asserted to have these three values, so the literal is
  tied to the file and not to the loader's message alone.

## Score loader and stores

The Beethoven score test pins 3,751 events, the tick resolution 480 the file header states, Partitura metadata with four
parts, and populated pitch data. An empty MIDI file must raise `EOFError`. `ScoreMidiEventData`
must contain the three score-only fields, while both event-data classes retain their declared
nullable fields and can be constructed from dictionaries.

`MidiStore` tests require notes and controls to split into the correct child stores while retaining
metadata. Empty stores have empty data and default metadata. Extension merges each category and
updates metadata. Mapping keys, iteration, items, lookup, membership, and length use the canonical
`notes`, `controls` order. Summary counts are exact. Timeline conversion creates the two expected
children at offset zero with the original note/control counts; each child is named after its store
and carries a minted id, so the tests find it by name.

The ticks→quarters map a `MidiStore` offers divides by the file's tick resolution exactly:
at 480 PPQ, 240 ticks answer `Fraction(1, 2)` and 960 ticks `Fraction(2, 1)`. The answer type is
asserted, because a map built on the float `1 / 480` computes in floats and answers `0.5`, which
compares equal to `Fraction(1, 2)`.

All exact counts, ticks, schemas, IDs, and extents use equality; there are no `pytest.approx` calls
in the MIDI-loader tests.

## Running the tests

From the repository root:

```bash
python -m pytest --runslow tests/loader/midi
```
