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
| `test_score.py` | Beethoven score loading, empty-file failure, the wider score schema, and the end of track of both parsers. |
| `test_store.py` | Base and score event-data schemas, nullability, construction, and inheritance. |
| `test_tempo_and_bars.py` | The tempo map, the bars and the stated facts of synthetic and real files, on all three loaders. |
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
  duration 106); the first loaded row, Partitura's `n0` (rows keep Partitura's order), is
  asserted to have these three values, so the literal is tied to the file and not to the
  loader's message alone. Every row's ID is minted (`note_…`): a MIDI file states no note IDs,
  and Partitura's numbering serves the velocity join only, whichever parser reads the file.

## End of track

A MIDI file states its extent in the `end_of_track` message closing each track: the tick at which
the track ends, the sum of its delta times. The file ends where its latest track ends, which may
lie after the last note's end (many files close a track one tick after it) and never before it.
Every MIDI loader exposes that tick as `end_of_track`, and each source's metadata records it.

The test writes a type-1 file with mido whose two tracks end at ticks the test chooses: a meta
track closed at tick 1, a note track whose only note (tick 0 to 1,440) is followed by its
`end_of_track` one tick later, at 1,441. Both `ScoreMidiLoader` parsers (Partitura and mido) and
`PerformanceMidiLoader` must answer 1,441, not the last note's end (1,440) and not the first
track's end (1); the source metadata says 1,441 too. A loader that has read nothing answers
`None`. A second file whose meta track ends after its notes (at 2,000, the note again ending at
1,440) answers 2,000: the latest track decides, whatever it holds.

`clear()` forgets the file read last: each of the three loaders, having read the first file
(1,441, 480 ticks per quarter), answers `None` for both `end_of_track` and `ticks_per_beat` after
`clear()`, as one that has read nothing, and the file read next (the 2,000 one) answers its own
2,000 and 480. A value surviving `clear()` would be the previous file's, stated of none.

## Tempo map and bars

A Standard MIDI File states three facts about its time axis besides its end of track: its
resolution (ticks per quarter, the header), its tempo map (`set_tempo`, microseconds per quarter)
and its bars (`time_signature`). Every MIDI loader records the last two per source, as the file
states them: `tempo_events`, `(tick, microseconds per quarter)` pairs, and `time_signatures`,
`(tick, numerator, denominator)` triples, integers both, sorted by tick, a later message at the
same tick (in track order, then message order) replacing an earlier one. The timeline a MIDI
loader creates stays on ticks and reaches the other units through what the file states: the
exact ticks→quarters `LinearMap`, a ticks→seconds `TableMap` and a measure map. All three
loaders (`ScoreMidiLoader` with either parser, `PerformanceMidiLoader`) must state the same axis
for the same file, so every synthetic case runs on all three.

What proves it is the arithmetic of the standard, done by hand on files the test writes with
mido in `tmp_path`, and the fact that what is read is written back.

**Tempo map.** The seconds at a tick are the sum over the tempo segments before it of
Δticks · µs / (ppq · 10⁶). The map's knots are tick 0, every tempo change and the end of track,
its seconds exact `Fraction`s, so the test compares the knots with `==` on `Fraction` literals,
and a float slipping in anywhere fails on type as well as value.

| File | Knots (tick, seconds) | Checks |
|---|---|---|
| ppq 480, 500,000 µs at 0, 250,000 at 960, end 1,920 | (0, 0), (960, 1), (1920, 3/2) | 480 ticks → 1/2 s, 1,440 → 5/4 s (inside the second segment: 1 + 480 · 250000 / 480·10⁶) |
| ppq 480, no `set_tempo`, end 960 | (0, 0), (960, 1) | the SMF default 500,000 µs per quarter: 960 · 500000 / 480·10⁶ = 1 |

Past the end of track the map extrapolates linearly with the tempo in force there: 2,400 ticks →
3/2 + 480 · 250000 / 480·10⁶ = 7/4 s on the first file. The public call writes the seconds
axis's own number type (`float`: 0.5, 1.25, 1.75, each exact in binary); the exact values are
the anchors, which the manifest writes as `[numerator, denominator]` pairs (`[3, 2]`), and the
test reads them there, and the timestamp at 1,440 ticks answers `Fraction(5, 4)` seconds when
asked for a fraction.

**Bars.** Bars run from tick 0, each the length of its signature (numerator · 4 / denominator
quarters); every time signature message starts a new bar, so a bar it interrupts keeps the length
it actually has; the last bar may be cut short by the end of track, and a bar that would start at
the end of track is not counted. A file without `time_signature` has 4/4, the SMF default. The
measure map counts bars from 1 and states no printed number (an SMF states none, so the Measure
Map standard's default applies). Expected rows, by hand:

- ppq 480, 4/4 at 0, 3/4 at 1,920, 2/4 at 2,880 (inside the 3/4 bar), end 4,800, that is
  quarters 0, 4, 6 and 10: four bars, `(count, qstamp, signature, nominal, actual)` =
  `(1, 0, 4/4, 4, 4)`, `(2, 4, 3/4, 3, 2)`, `(3, 6, 2/4, 2, 2)`, `(4, 8, 2/4, 2, 2)`.
- ppq 480, no `time_signature`, end 2,640 (5.5 quarters): `(1, 0, 4/4, 4, 4)`,
  `(2, 4, 4/4, 4, 3/2)`; and a 3/4 at the end of track itself adds no bar.
- 6/8 and 19/16 stay as written: 6/8 at 0, 19/16 at 1,440, end 3,720 (ppq 480): `(1, 0, 6/8, 3,
  3)`, `(2, 3, 19/16, 19/4, 19/4)`.

The measure map is read through the timeline (`timeline.measure_map`), whose ticks axis reaches
it through a ticks→quarters `LinearMap` chained before the `MeasureRecordMap`, so the timeline
writes `measure_maps` and the `ChainMap` in its manifest (on the bars file above, with the two
tempi: the chain's `LinearMap` has scalar `[1, 480]`, its `MeasureRecordMap` names the one
measure-map entry, and the tempo knots are (0, 0), (960, 1), (4800, 3)). `Timeline.from_dict` of the written
manifest must give an equal measure map, and writing the restored timeline must give the same
dictionary: nothing the file states is lost on the way.

**Several sources.** A loader holding several files puts their events on one ticks axis, so the
timeline states a fact of that axis only where every file states it alike; the end of track is
the latest. Two files at 480 ticks per quarter, both at 250,000 µs from tick 0, ending at 960 and
1,920, give the knots (0, 0), (1920, 1) and one bar (1,920 ticks are four quarters), without a
warning. Two files that state 250,000 and 500,000 µs give no seconds map at all (falling back to
the default tempo would state a tempo neither file states) and exactly one `UserWarning`, whose
text names the loader, the two sources and `tempo_events`; the facts they share still hold: the
quarters map (480 ticks → 1 quarter) and the one 4/4 bar, cut to 2 quarters by the end of
track at 960.

**Real file.** `score/beethoven_op18.mid` (480 ticks per quarter) is checked against values read
from its bytes, not from the loader: the test scans the file's messages with mido itself and the
literals below were read off that scan. It states 14 `set_tempo` messages, at ticks 0, 15,360,
37,440, 37,920, 52,800, 53,280, 191,040, 191,520, 306,240, 306,720, 340,320, 340,560, 430,560
and 532,800, and its tracks end at 534,240. The tempo map's knot ticks must be these 14 and the
end of track (15 knots), and on every segment the slope must be exactly the segment's
microseconds over 480 · 10⁶ (352,941 µs on the first). Its bars, counted by hand from the
signatures the scan lists (a pickup of 1/4, then 2/2, 3/4, 1/4, 7/8 and 2/16 sections, each
section a whole number of bars), are 290, every one of its signature's full length. Three
signatures at one tick (1/4, 2/2, 1/4 at 30,720, quarter 64) leave the last in force: bar 19 is
`(19, 64, 1/4, 1, 1)`; bar 121 is `(121, 897/2, 2/16, 1/2, 1/2)`; and the 3/4 at its end of track
(quarter 1,113) adds no bar, so the last is `(290, 1109, 2/2, 4, 4)`. `score/beethoven_mtd.mid` (1/8 at 0, 4/4 at 240, 500,000 µs at 0, end 6,960) gives
by hand five bars, `(1, 0, 1/8, 1/2, 1/2)`, `(2, 1/2, 4/4, 4, 4)`, `(3, 9/2, …)`,
`(4, 17/2, …)`, `(5, 25/2, 4/4, 4, 2)`, and a seconds map of two knots, (0, 0) and
(6960, 29/4).

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
