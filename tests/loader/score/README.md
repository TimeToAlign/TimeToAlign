# Symbolic score loader tests

These tests validate the symbolic score loaders against exact source-derived values. TSV data
loaded by `Ms3Loader` is authoritative whenever loaders are compared.

## Test files

| File | Remaining coverage |
|---|---|
| `conftest.py` | Seven specimen definitions, source lookup, size guards, `.flow.csv` parsing, a timeline's flow in the form of that file, and the stress score's repair. |
| `test_score_parsing_matrix.py` | CSV modes, folded counts, loader/format combinations, exact flow-control profiles, each loader's played default flow against the target, cross-loader counts, and pinned parser differences. |
| `test_flow_gold.py` | The gold flows played from the ms3 measures tables (see "Gold flows"), and the signs a loader writes as flow control (see "Signs written as flow control"). |
| `test_loaders.py` | MS3 coordinate and duration semantics, pitch schema, Partitura measure/MIDI regressions, shared `mc_onset`, anacrusis offsets, and parquet round trips. |
| `test_cross_validation.py` | Exact Chopin note-count, MIDI-pitch and pitch-name agreement across MS3, Partitura, and Music21; measure onsets only where the source states them (see "Cross-loader parity" below). |
| `test_measuremap_loader.py` | MeasureMap expansion, validation, schema, and comparison with MS3. |
| `test_flow_csv_validation.py` | Right-open ranges, exact atomic boundaries, and live Partitura boundaries. |
| `test_partitura_exactness.py` | Exact Partitura coordinate pairs, instant nulls, flattened timelines, and agreement with Partitura's float map. |
| `test_ms3_facets.py` | `Ms3Loader`'s companion discovery in both corpus layouts and its reading of empty note cells, and its refusal of a measure without a count or a length (see "MS3 facet discovery", "MS3 empty note cells" and "MS3 measures without a count or a length" below). |
| `test_verovio_loader.py` | `VerovioLoader` on MEI fixtures and specimens: exact onsets, kept ids, measures as structure, ties, grace notes, repeats, movements, the physical timeline, music21 parity, Verovio's expansion against the played flow control (see below). |
| `test_verovio_ids.py` | `VerovioLoader`'s minted timeline ids when the root's id is given (see "Ids of a given root" below). |

## Parsing matrix

The matrix covers these specimens and exact folded/unfolded measure counts:

| Specimen | Folded | Unfolded | Atomic sections |
|---|---:|---:|---:|
| `rachmaninoff` | 374 | 374 | 1 |
| `polyrhythm_only` | 14 | 14 | 1 |
| `c05n05_musete` | 58 | 138 | 4 |
| `c11n08_Rondeau` | 60 | 138 | 4 |
| `op18_no4_mov4_flow` | 226 | 291 | 13 |
| `flow_only` | 15 | 30 for MS3 | 13 |
| `WoO71` | 397 | 505 | 26 |

For every available source, the matrix checks TSV and MeasureMap folded counts, Music21
MusicXML and MEI folded counts, Partitura MusicXML and MEI folded counts, and same-loader
MEI/MusicXML equivalence where both formats exist. Files larger than 500,000 bytes use the
documented size skip because third-party parsing is too slow for this integration suite.

**The flow of a loader is what its timeline's flow control plays.** A loader's timeline is
created, the engine plays its flow control (`default`), and the runs are stated as the
`.flow.csv` file states a flow: right-open ranges of measure counts, each listing the atomic
sections it covers, divided at the section breaks (`conftest.played_ranges`). The comparison
with the file's `default` rows is exact, letters included, and the walk must report no
finding. This proves a loader end to end, source to signs to flow, against rows nobody
computed with this library's rules; it cannot be satisfied by a loader and a test agreeing
on a wrong reading, since the file is the third party.

- **Music21 MusicXML**: `op18_no4_mov4_flow` plays the target flow. `polyrhythm_only`,
  `c05n05_musete` and `c11n08_Rondeau` are strict `xfail`s (`ValueError`): a MusicXML score of
  several parts gives the loader one measure row per part, which form no measure map, so no
  timeline is created. `flow_only` is pinned below.
- **Partitura MusicXML**: `c05n05_musete` (two D.S. al Fine), `c11n08_Rondeau` (two D.C. that
  play until the segno) and `op18_no4_mov4_flow` play the target flow and have the target's
  atomic sections. `polyrhythm_only` is the same strict `xfail`; `flow_only` is pinned below.
- **Music21 MEI against MusicXML**: music21 reads repeat signs and endings and nothing else, so
  the three columns the flow control is written from (`start_repeat`, `end_repeat`, `volta`)
  are compared row by row between the two formats.
- **MeasureMap**: a Measure Map row has no key for a D.S., a D.C., a segno or a Fine, and the
  corpus files state no `volta`. For every specimen the repeat signs read back from the
  timeline's flow control (`measure_keys_from_flow_control`) must equal those of the ms3
  table, measure by measure. The two Couperin pieces pin what such a timeline plays: the
  repeats and nothing else, `[(1, 17), (1, 32), (17, 59), (32, 59)]` and
  `[(1, 10), (1, 19), (10, 28), (19, 61), (28, 61)]`, and its elements are exactly the
  translation of its own measure map. What the file knows beyond the repeats is its `next`,
  which stays with the measures (`MeasureMap` and the measure table) and is no flow control.

`TestVerovioLoaderMEI` reads each matrix score's MEI with `VerovioLoader` (`expandNever`: the
printed measure table is the same either way, and Verovio crashes on expanding the
flow-control files) and requires the folded count and the target `default` flow from the
timeline's flow control. `polyrhythm_only` matches exactly. `flow_only` has a test of its own:
the MEI's marks give the signs the ms3 table gives (every jump target, break and section
break at the same coordinate with the same manifestation; Verovio names the second coda
`coda2`, the table `codab`), so the flow ends at the second of three brackets as the table's
does, the atomic sections are the target's, and with the same repair the flow is the
target's `default` flow (see "Gold flows"). Each residue is a strict `xfail` naming what the
MEI states:

- `c05n05_musete`: a `daCapo` at MC 31, where the target flow jumps to the segno.
- `c11n08_Rondeau`: `daCapo`s at MCs 18 and 27 with neither a Fine nor an "al …" text, so
  nothing states where they end.
- `WoO71` (slow): no section breaks, which cut the target's sections. The MC sequence itself is
  the target's, all 505 visits, asserted by `test_woo71_mc_sequence`.
- `op18_no4_mov4_flow`: a skeleton whose measures have no content, hence no timemap entry; the
  load raises `ValueError`.

Rachmaninoff is not in the Verovio matrix: Verovio 6.4.1 crashes the process on its MEI.

Every `.flow.csv` must exist and contain `atomic` and `default` modes. Its number of atomic rows
must equal the specimen value above. TSV and MeasureMap must have identical folded counts.

## Exact flow-control profiles

`TestFlowControlProfiles` contains the former standalone parity assertions. It derives repeat
counts from either `start_repeat`/`end_repeat` booleans or MS3's `repeats` values, counting
`startend` as both a start and an end.

| Specimen and loader | Measures | Starts | Ends | Other exact values |
|---|---:|---:|---:|---|
| WoO71 TSV | 397 | 11 | 11 | 12 section breaks; 4 double barlines |
| WoO71 MeasureMap | matrix count | 11 | 11 | flow control present |
| WoO71 Music21 | matrix count | 11 | 11 | flow control present |
| WoO71 Partitura | matrix count | region model | region model | flow control present |
| `flow_only` TSV | 15 | 3 | 6 | flow control present |
| `flow_only` MeasureMap | matrix count | 3 | 6 | flow control present |
| `flow_only` Music21 | 15 | 3 | 6 | flow control present |
| `flow_only` Partitura | 15 | 7 | 7 | inferred region boundaries; flow control present |

The marker-based loaders therefore agree exactly on repeat counts. Partitura is intentionally
different on `flow_only`: it completes repeat regions by inferring boundaries.

## Pinned parser differences

`TestDocumentedDeviations` requires intentional updates when third-party parser behavior changes:

- Music21 reads no navigation mark (slow): on `c05n05_musete` and `c11n08_Rondeau` its measure
  table states the repeat starts (MCs 17, 32; MCs 10, 19, 28) and repeat ends (MCs 16, 31, 58;
  MCs 9, 18, 27, 60) and leaves `markers`, `jump_bwd`, `jump_fwd` and `play_until` empty in
  every row, so no D.S. and no D.C. can be written.
- Music21's `flow_only` timeline holds repeats and brackets alone (the manifestations of its
  destinations are `repeat_end` and `volta`). Its flow is `[(1, 4), (1, 6), (4, 5), (6, 7)]`,
  ending at the second of three brackets as the score's own source has it, and its eleven
  atomic sections lack the boundaries the segno and the codas give at MCs 8 and 13.
- Partitura's `flow_only` source rows have starts at MCs `[1, 4, 6, 9, 10, 11, 12]`, ends at
  `[3, 5, 8, 9, 10, 11, 14]`, and voltas
  `[(5, 1), (6, 2), (7, 3), (14, 1), (15, 2)]`.
- The `flow_only` CSV must retain both `default` and `ms3` interpretations.
- Music21 reads only the 10 measures of the `flow_only` MEI that state an `@n` (of 15): its
  MEI folded count and MEI-against-MusicXML flow are strict `xfail`s.
- Partitura's MEI importer (1.7.0) raises before the loader sees a measure, so the Partitura MEI
  folded count is a strict `xfail` on three scores: on `polyrhythm_only` it raises `Exception`
  in `_handle_metersig`, because staff 3 states its meter only as `meter.sym="cut"`; on
  `c05n05_musete` and `c11n08_Rondeau` it raises `TypeError` in `_find_dir_positions`, because a
  `<dir>` with a `@type` is placed by `@startid` and states no `@tstamp`.

## Cross-loader parity

`test_cross_validation.py` loads Chopin Op. 10 No. 3 three times: the ms3 notes TSV and the
MusicXML through Partitura and through Music21. Each loader parses with a different library, so
an agreement is evidence that all three read the score, and a disagreement on a column they all
write is a defect of one of them.

- **Note count and MIDI pitch.** 498 notes each; the MIDI pitches agree exactly.
- **Pitch names.** The name a loader gives a note is its spelling, so the three must give every
  note the same name, written with `♯`/`♭`. ms3 writes ASCII (`G#4`), so this holds only when
  `Ms3Loader` respells at the loader. The rows of the three loaders are not in one order (the
  sort is by MIDI pitch, and ties keep whatever order each parser produced), so the comparison
  is of the multisets of `(MIDI pitch, name)` pairs, which must be equal; that the pairs are
  equal and not just the names also proves that each loader attaches its name to the right
  pitch. On this score 217 of the ms3 names carry a sharp, two of them `B♯4`, and no name holds
  `#` or `b`.
- **Pitch names from Verovio** (slow). Chopin has no MEI, so Verovio is compared on Beethoven
  Op. 18 No. 4/iv, whose `ABC` directory holds the MEI and the ms3 notes TSV of one score.
  Verovio renders its own expansion, which leaves out the endings at MCs 44-45, 93-94 and
  102-103, so the note counts differ and the comparison is of the sets of distinct `(MIDI
  pitch, name)` pairs: Verovio's 64 pairs must all be among ms3's 65. The one pair only ms3
  gives, `(42, "F♯2")`, belongs to the score's only F♯2, in MC 45, an ending Verovio leaves
  out. A loader
  spelling one name otherwise (`F#2`, `Gb2`) adds a pair outside ms3's and fails.
- **Measure onsets are stated, never copied.** `mn_onset` is a note's position in its notated
  measure, `mc_onset` its position in the measure count; they differ wherever a notated measure
  is split, for instance at an anacrusis. The TSV states both (its first note: `mc_onset` 0,
  `mn_onset` 3/8), so `Ms3Loader` keeps them. MusicXML states neither as such and Partitura and
  Music21 derive only `mc_onset`, so their `mn_onset` is null on every note, control and
  annotation row: a copy of `mc_onset` there would claim a fact the source does not state, and
  the TSV shows that it would be wrong.
- **A column of nulls keeps its type.** Every store declares `mn_onset` as a rational struct
  (`struct<value: double, numerator: int64, denominator: int64>`), and the stores of Partitura and
  Music21 keep that type though every cell is null. The timeline's `get_events()` rebuilds its
  rows into a new table, which took each further column's type from its values, so a column of
  nulls came out as `string`. The test asserts, for both loaders, that the `mn_onset` field of
  `create_timeline().get_events()` has the store's struct type and its metadata, and that every
  one of its cells is null (547 Partitura events, 536 Music21 events). The timeline holds one
  child per store and `get_events()` returns their events: 498 notes, 22 measures (the score's
  22 bars), the controls (Partitura 22: six slurs, fourteen loudness directions, one time and one
  key signature; Music21 11: six dynamics, three metronome marks, one key and one time
  signature) and 5 text annotations, so 547 = 498 + 22 + 22 + 5 and 536 = 498 + 22 + 11 + 5.
  The per-facet counts asserted beside them (`notes` 498, `controls` 22 or 11, `annotations`
  5) are the same stores' rows.

## Loader-specific assertions

`test_loaders.py` keeps behavior that is not part of the parsing matrix:

- WoO71 MS3 measures begin with exact `(start, duration, end)` Fractions `(0, 1, 1)`,
  `(1, 2, 3)`, and `(3, 2, 5)`.
- Symbolic triplet duration is exactly `1/3`; native `0.5` is exactly `1/2`; a derived
  `0.3333333333333333` without a symbolic source keeps its double, mirrored by that
  double's exact dyadic `6004799503160661/18014398509481984`. No cell is ever
  pair-less — the assertion checks the mirror is the dyadic and **not** `1/3`, since
  tidying it to `1/3` is the fabrication the exactness rules exist to forbid.
- **Every facet reads the symbolic duration column, not just notes and measures.** An
  ms3 TSV states a duration twice — exactly, as a whole-note fraction in `duration`,
  and derived, as a float in `duration_qb` — and which one a facet reads is not a
  matter of taste. Reading the float discards a representation the source actually
  carries, and the loss is not confined to the last decimal: the exact dyadic of
  `0.0833333333333333` has denominator `2**56`, so adding it to a position of any
  musical size needs a numerator wider than the int64 the coordinate struct stores,
  and the load fails outright. The chords and harmonies facets read the float and
  broke on ordinary triplet durations; both now resolve the same way notes do.

  Pinned on a chord row whose source cells are `quarterbeats = 695/4` and
  `duration = 1/48` (whole notes), beside a `duration_qb` of `0.0833333333333333`:

  | Coordinate | Exact value | Read from |
  |---|---|---|
  | start | `695/4` | `quarterbeats` |
  | duration | `1/12` | `duration` × 4 — *not* the float column |
  | end | `1043/6` | exact sum |

  Taken from the float instead, the same duration is `6004799503160659/2**56` and the
  end is `12526011763593139539/2**56`, whose numerator does not fit. The test asserts
  the exact triple, so a facet quietly falling back to `duration_qb` fails on the
  value rather than only on corpora large enough to overflow.
- **A harmony label's span is stated by the label after it.** A DCML `harmonies`
  facet has no symbolic duration column at all, so the rule above has nothing to
  read. It does carry `duration_qb` — but that column *is* the label-to-label
  subtraction already, carried out in float, and it keeps the residue: the label at
  `1003/2` records `0.16666666666668561`, which is not even the nearest double to
  `1/6`. Redoing the subtraction on the exact `quarterbeats` cells recovers what the
  source encodes (`1505/3 − 1003/2 = 1/6`), so this is the source's own derivation
  minus its rounding, not a semantics the loader invented. Snapping a float residue
  to a tidy ratio stays forbidden — the value here is *read*, never guessed.

  The final label has no successor, so its span runs to the end of the piece, taken
  from the companion `measures` facet as the furthest measure position plus that
  measure's `act_dur` × 4. Pinned on the Meistersinger prelude: the last label starts
  at `1759/2` and the measures facet ends at `884 + 4 = 888`, giving `17/2` — the same
  `8.5` ms3 derived, now exact. Where either end of a span is missing, out of order or
  inexact, the loader says nothing and falls back to the derived column rather than
  asserting a span the source does not support.

- **The whole score corpus is loaded, not a chosen file.** All 46 `.chords.tsv` and
  `.harmonies.tsv` files in the `score` corpus, and all 4 in `supra`, are loaded in
  one slow-marked sweep. This is deliberately a sweep rather than a specimen: the
  defect above sat behind a green suite because the affected files were in no test's
  load set, and a hand-picked replacement specimen would reproduce exactly that gap.
  `supra` is included because its Meistersinger harmonies are the only corpus files
  exercising the label-span derivation at a magnitude where the float column
  overflows. Marked slow; runs under `--runslow`.
- All 4,753 populated WoO71 note coordinates carry exact numerator/denominator pairs; instant
  rows keep null ends and durations.
- MS3 exposes spelled pitch once through `specific_pitch`, retains raw MIDI 59 for the first B3,
  and affords the same number through `EnharmonicPitch`.
- Partitura extracts 22 Chopin measures and loads the Beethoven MIDI regression as 4,186 notes.
- MS3, Partitura, and Music21 populate `mc_onset`.
- Chopin's raw Partitura onset is exactly `-0.5` quarter beats and its normalization offset is
  exactly `0.5`; Music21's offset is exactly `0.0`. Metadata and store properties carry those
  exact values.
- **The anacrusis offset is as exact as the onsets it is read from.** Partitura states the
  Chopin onsets as exact ratios, so the offset is `Fraction(1, 2)` — its type is asserted,
  not only its value, because `0.5 == Fraction(1, 2)` would pass a float. The `raw_quarters`
  ShiftMap the store attaches then reads the timeline's first note, at `1/2`, as
  `Fraction(0)` and the quarter at `3/2` as `Fraction(1)`: a float offset would answer the
  floats `0.0` and `1.0`, so the map's answer type is what proves the parameter exact. The
  offset survives `to_parquet` / `from_parquet` as the Fraction pair `[1, 2]` in
  `metadata.json`, read back as `Fraction(1, 2)`; a float offset would be written as a
  float and read back as one.
- The measures are shifted with the notes: the MusicXML's measure 1 holds one note of 8 divisions
  at 16 per quarter (an anacrusis of `1/2`), so after the offset measure 1 starts at `0` and
  measure 2 at `1/2`.
- `ScoreStore` and loader parquet tests pin files, note/measure counts, metadata, units, empty
  facets, missing paths, and event-property round trips. The loader's `notes.parquet` holds
  the notes TSV's 498 rows (the defect it guards wrote an empty file).

The 498-note Chopin gold value is asserted once per loader in
`TestNoteCountConsistency.test_note_counts_exact_match`. The same module requires exact MIDI
pitch equality and checks both `EnharmonicPitch` and `MidiPitch` views for all 498 events.

## MS3 facet discovery

`Ms3Loader(auto_discover=True)` promises the piece's other facets from one file. Each facet
file is named `<piece>.<facet>.tsv`; a companion keeps the piece name and swaps the facet, and
is looked for beside the file (the ms3 layout) and, when the file lies in a directory named
after its facet, in the sibling directory of the other facet (the DCML corpus layout:
`notes/p.notes.tsv`, `measures/p.measures.tsv`, `harmonies/p.harmonies.tsv`). Discovery used
to look for `measures/p.notes.tsv`, so on a DCML corpus it found nothing and the notes loaded
alone, silently: Debussy's Clair de lune came out 645/2 quarters long (its last note's end)
instead of the notated 324, with no measure map and no harmony labels.

The proof is a small piece written into `tmp_path` in each layout: three 4/4 measures (12
quarters), one note at quarter 9 lasting one quarter (it ends at 10, short of the notated end),
two harmony labels at 0 and 6 and no chords facet. The last label's `duration_qb` cell is left
empty on purpose: its span can then come only from the measures facet (end 12, so 6), which the
harmonies reader finds through the same discovery. From the notes file, in both layouts:

| Assertion | Value | What it proves |
|---|---|---|
| loaded sources | notes, measures, harmonies, in that order, resolved | every present facet found, the absent chords skipped |
| timeline length | exactly `Fraction(12)` | the measures facet was read; the notes alone give 10 |
| measure map | 3 measures | the measures facet became structure |
| labels and their spans | 2; `6`, `6` (exact) | harmonies found, and the last span reached the measures facet |
| warnings | none | a complete discovery is silent |

Both layouts give identical sources (up to directory), lengths and spans. Starting from the
measures file of the DCML layout finds the notes and harmonies. An `_unfolded` measures file
beside the folded one in `measures/` is never taken for the folded notes, and the reverse.
Where a facet exists in both places the file beside the source is taken, the order the
docstring states.

When nothing is found the loader still loads the file alone, but warns: a notes file alone in
`notes/` raises one `UserWarning` naming the source and every path it looked at (both
candidates of each of the three other facets); a file whose name states no facet warns that it
states none. Without `auto_discover` nothing is looked for and nothing warns.

The real corpus is checked by hand only (CC BY-NC-SA, never copied here): from
`notes/l075-03_suite_clair.notes.tsv` discovery loads all four facets, length 324 quarters, 72
measures, 1680 notes, 150 labels.

## MS3 empty note cells

An empty cell states nothing, so it reads as null, never as a value. In ms3's `tied` column `1`
starts a tie, `0` continues one (tied from the previous note and to the next), `-1` ends one,
and an empty cell is a note without a tie. The loader used to read the empty cell as `0`, which
turned every untied note into a tie middle: in mm. 1–6 of Clair de lune it gave `{0: 38, 1: 20,
-1: 19}` where the TSV holds 36 empty cells and two `0`s. The same reading filled an empty
`octave` with `4` and built a specific pitch on it, an octave the source never states.

The proof is one notes TSV in `tmp_path` with four notes whose `tied` cells are, in row order,
empty, `0`, `1` and `-1`; the loaded `tied` column must be exactly `[None, 0, 1, -1]`, so a
default of either `0` or any other value fails on the first row, and a reader that shifted
values fails on the others. A fifth note spelled `C` with an empty `octave` must load with a
null `octave` and a null `specific_pitch`, while its MIDI pitch `60` stays; the notes with an
octave keep their specific pitch, so the check is not passed by dropping spellings wholesale.

Every other optional integer cell of the notes facet (`mc`, `midi`, `tpc`, `voice`, `staff`,
`chord_id`) already reads as null. Music21Loader and PartituraLoader read no ties; they wrote
`0` into the shared `tied` column, which under ms3's values claims a tie middle on every note,
and now write null. By hand on Clair de lune: mm. 1–6 give start 20, end 19, middle 2, none 36;
the whole movement 118, 118, 3 and 1441, exactly the TSV's counts.

## MS3 pitch names

ms3 spells accidentals in ASCII (`Bb3`, `F#4`); the loader writes the `name` column in the
library's spelling, `♯` and `♭` after the step letter (`canonical_pitch_name`, see
`tests/core/README.md`, "Pitch names in one spelling"), as the other score loaders do. An empty
`name` cell states no name and reads as null, never as the string `"nan"`, a name no note has.
The octave of MIDI pitches 0 to 11 is `-1`, and its `-` is no flat: `C-1` is C in octave -1,
not C♭.

The proof is one notes TSV in `tmp_path` with five notes whose `name` cells are, in row
order, `Bb3`, `F#4`, `C-1`, `F##4` and empty (the last with MIDI 60 and octave 4). The loaded
`name` column must be exactly `["B♭3", "F♯4", "C-1", "F♯♯4", None]` and the specific pitches'
`(fifths, octave)` exactly `(-2, 3)`, `(6, 4)`, `(0, -1)`, `(13, 4)` and none for the nameless
note (no spelling, no specific pitch), while its MIDI pitch 60 stays. A spelling is a position on
the line of fifths (C 0, G 1, … B 5, F -1; each sharp adds 7, each flat takes 7 away), so B♭ is
5 - 7 = -2, F♯ is -1 + 7 = 6 and F♯♯ is -1 + 14 = 13. Each stored pitch is read back as a
`SpecificPitch` (`SpecificPitch.from_row`) and compared by its `fifths` and `octave`, never by
the members the column happens to store.

## MS3 measures without a count or a length

A measures table numbers every measure in `mc` and states its length in `act_dur` (whole
notes, exact) or `duration_qb` (quarters, derived). The loader used to invent what a row
left out: an empty `mc` became the row's position plus one, so a gap in the numbering moved
every following measure without a word, and a row stating neither length became a measure of
length `0`. Neither is a reading of the source; both now refuse, and no other column states
either value exactly (`timesig` states the nominal length, not the actual one, and the next
row's `quarterbeats` is not this measure's end where endings or jumps follow it).

The proof is a measures TSV in `tmp_path` of three 4/4 measures, one cell removed per case;
`Ms3Loader.from_file` must raise `ValueError` with exactly:

| Removed | Message |
|---|---|
| `mc` of the second row | `The measures table 'p.measures.tsv' states no mc in row 2` |
| `act_dur` of the second row, no `duration_qb` column | `The measures table 'p.measures.tsv' states no length of mc 2 (row 2): neither act_dur nor duration_qb holds a value` |
| both length cells of the second row, both columns present | the same message |
| `act_dur` of the second row, its `duration_qb` the text `four` | `The measures table 'p.measures.tsv' states no readable length of mc 2 (row 2)` |

The complete table loads with lengths `[4, 4, 4]` as exact `Fraction`s and counts `[1, 2, 3]`,
so the refusals are not passed by refusing every table. A row giving `duration_qb` alone still
loads: with `act_dur` empty and `duration_qb` `3.0` its measure is `Fraction(3)` long, the
length the source states.

The last case states a length the loader cannot read: the cell holds a value, so the row is
not "without a length", but no number can be read from it. Reading it as anything (`0`, the
nominal length, the next row's position) would invent the length again, so it is refused
under its own message, which says the length is unreadable rather than absent.

## Coordinate exactness

Temporal coordinates use structs with `value`, `numerator`, and `denominator`. Populated
coordinates must retain their pair; instant events must keep null `end` and `duration` values.
The flattened Chopin timeline contains exactly 547 rows: 31 instants and 516 intervals. Known
Partitura divisions include start `11/2` with duration `3/4` and start `3/4` with duration `1/4`.

There are no `pytest.approx` calls and no tolerances in the score-loader tests. Exact integers,
Fractions, binary halves, zero offsets, metadata values, and coordinate extents use equality.
`test_partitura_exactness.py` compares every Partitura start with its division count over the
divisions per quarter the file states (16 for Chopin Op. 10 No. 3, 72 for WoO71): a note at
division `t` starts at exactly `t/16` (or `t/72`) quarters, the first note at division 0. The
floating-point `quarter_map` is not the oracle, since it cannot state `t/72` exactly.

## MeasureMap and flow CSV contracts

### Timeline-native structure on the Rondeau specimen

The Couperin Rondeau expectations are derived directly from
``c11n08_Rondeau.measures.tsv`` resolved through the score corpus. Its 60 rows
carry repeat ends at MC ``9``, ``18``, ``27``, and ``60`` and repeat starts at
MC ``10``, ``19`` and ``28``. Each repeat end is a destination (``repeat_end``)
of the Jump at its measure's exact ``qstamp + actual_length``, and the place it
goes on at (``FlowControlRegistry.place_of``) must be the exact ``qstamp`` of MC
``1``, ``10``, ``19`` and ``28`` respectively. The last three name their repeat
start. The first has no printed start: its destination states no place
(``{repeat_count: 1, manifestation: repeat_end}`` and nothing else), and the
place is the start of its section, the piece's. The expected coordinates are
derived from independently parsed measure facts.

The ``jump_bwd`` column adds two da capo destinations. It holds ``start`` on MC
``18`` and MC ``27`` and is empty everywhere else. A backward target of ``start``
names the beginning, so each is a destination without place on the Jump at its
measure's end. Prefix sums of ``act_dur`` × 4 over the timeline's own measure map
give the end of MC ``18`` as ``23 + 1 = 24`` quarters and the end of MC ``27`` as
``35 + 1 = 36`` quarters, so the two go on from ``24`` and ``36`` at ``0``. Both
lie on the Jumps of the repeat ends of those measures: a barline has one Jump,
and the registry holds exactly ``4``. The ``play_until`` cells (``segno``) are
neither ``fine`` nor a coda, so the manifestation is plain ``da_capo``.

The TSV has no ``breaks`` cell containing ``section``, and what its two D.C. play
until is followed by a jump (``jump_fwd`` ``2e`` and ``3e``), so the registry
holds no break and no section break. The markers ``1e``, ``2e`` and ``3e`` are
jump targets at the starts of MC ``10``, ``19`` and ``28`` (``12``, ``24`` and
``36`` quarters) that keep their names and have no manifestation. The ``segno``
of MC ``9`` is named by ``play_until`` alone: it is where the two D.C. stop and
go on, the end of that measure, and no jump target.

Both normal and flattened timeline creation must store one ``MeasureMap`` and the
same flow-control facts: the timeline holds its map as one object, returned
identically on every access, with exactly ``60`` measures (one per row of the
TSV), and flattening merges events only, never dropping the map or a flow-control
fact. The registry's ``flow_control`` array is serialized through JSON and read
back to an equal array holding ``4`` jumps. The whole timeline is serialized
through JSON and restored as well: the restored payload must equal the original
payload, which is a fixpoint over every structural key, the restored registry
must hold exactly ``4`` jumps, and the restored measure map must equal the
original under type-aware measure equality.

An unfolded measures table cannot be a measure map. The map lists measures in
printed order, and each measure id is its Measure Map row ``ID``, which for a
measures table is the printed measure count (``mc``) as a string. ``c11n08_Rondeau_unfolded.measures.tsv`` has ``138`` rows that replay
the printed measures in performance order: its ``mc`` column reads
``1..9, 1..18, 10..18, 1..9, 19..27, ...``. The count ``1`` therefore occurs
more than once, and so does the id ``"1"``. Creating a timeline from that table
must raise ``ValueError`` stating that measure IDs must be unique within a
``MeasureMap``. The rows are a defect of the source, not a lossy reading of it,
so the loader must not return a timeline without its measure structure. The
uniqueness check runs before any per-measure count or ``qstamp`` comparison, so
it must emit no warning at all.

The `.mm.json` loader applies the Measure Map standard's one anacrusis
convention before any measure is built, and nothing else. A file whose first row
is numbered 0, in 4/4, with actual length 1 must load as a `MeasureConstituent`
at offset exactly 3 (nominal 4 minus actual 1), and its document rows must carry
`"offset_within_measure": [3, 1]`. The same row numbered 1 is an
`IrregularMeasure` with no offset: the convention names number 0 only. A first
row that states its own `offset_within_measure` keeps that offset. A later row
numbered 0 and shorter than nominal stays an `IrregularMeasure`, because only
the first row is an anacrusis by convention and a shared number never marks a
split. The loader also reads the measure-table spelling `-1` in `next` (the end of the
piece). The row form refuses it as a count naming no row, so the rows are read
without it and the measure map leaves it out; the loader's own expanded data
and measure table keep it where the file states it. Every folded corpus
`.mm.json` file states `-1`, so loading `WoO71.measures.mm.json` at all pins the
conversion. Measures are
built from the expanded rows themselves, never from the measure table the loader
also fills. On the corpus, `WoO71.measures.mm.json` opens with a `MeasureConstituent`
at offset 1, exactly the offset `WoO71.measures.tsv` states in its `mc_offset`
column for MC 1.

MeasureMap tests require unique MC values, monotonic `qstamp`, valid `next` references,
397 folded WoO71 measures, and matching MS3 MCs and repeat presence. Its schema must include flow-control and identity fields. The synthetic summary
contains three measures, one repeat start, and one repeat end.

Flow CSV tests enforce right-open contiguous atomic ranges. Exact samples are
`c05n05_musete`: `A=(1, 6)`, `B=(6, 17)`, `C=(17, 32)`, `D=(32, 59)`;
`polyrhythm_only`: `A=(1, 15)`; and `rachmaninoff`: `A=(1, 375)`. Live Partitura segments are
compared against the corresponding CSV rows.

## Gold flows

`test_flow_gold.py` plays the seven pieces of `target_flows/` from their ms3 measures tables.

**What proves the flow right.** A `.flow.csv` file was written by hand as the flow a musician
plays, and it is the only statement of that flow that does not come from this library. So the
test takes nothing from the library but its public path: `Ms3Loader` creates the timeline, the
engine plays the timeline's own flow control, `measure_ranges` states the runs in measure
counts. The result must equal the file's rows exactly: the same number of rows, the same
`[mc_start, mc_end)` of each, and the same letters of the atomic sections each covers, for
`default` and for `single`; and the atomic sections must equal the `atomic` rows. A finding
of the walk is a failure too, since a correct score plays through. Nothing is patched between
the three steps, so a wrong sign, a wrong rule and a wrong conversion each show.

- **`c05n05_musete`, `c11n08_Rondeau`, `op18_no4_mov4_flow`, `WoO71`**: `default`, `single`
  and `atomic` equal the file; no finding; `repeat_end_findings` is empty.
- **`rachmaninoff`, `polyrhythm_only`**: no jump, break or jump target. The file states one
  atomic section and one `default` row, both the whole piece, and no `single` rows; `default`
  and `single` are both played and both must be that one row.
- **The rows of a file never span a section break.** WoO 71 has twelve; its `default` rows
  are 25, its runs 13 (the first three `[1, 22)`, `[12, 43)`, `[33, 94)` against the file's
  `[1, 22)`, `[12, 22)`, `[22, 43)`, `[33, 43)`, `[43, 73)`). The form of the file therefore
  divides a run at every section break it crosses, which `measure_ranges(divide_at=…)` does;
  the engine's runs are not divided. A test pins both numbers so that the division stays a
  stated step.
- **`flow_only`, the stress score, as its source stands.** The second of three brackets has
  no sign at its end. The flow is the file's first four rows, `[1, 4)`, `[1, 6)`, `[4, 5)`,
  `[6, 7)`, and ends at the break between the second and the third bracket (10 quarters): the
  walk reports `flow_ends_at_break` there and `never_played` for `[10, 37)`. The atomic
  sections are the file's thirteen all the same. `repeat_end_findings` returns the repeat end
  of MC 8, whose repeat would begin right behind the last bracket.
- **`flow_only`, repaired.** `conftest.repaired_stress_score` is the one worked repair of
  the suite, done on a copy of the loaded registry as an editor would do it. A repair adds
  elements and rejects elements; none is deleted or changed in place.
  1. *Reject.* The Jumps at the ends of MC 8 and MC 14 each hold one destination, a repeat
     end without a place (no repeat start is printed for it). The editor records the belief
     `false` on both with `registry.with_belief(id, actor, "false")`: they stay in the
     registry and in what it writes, and are left out of every flow.
  2. *Add the places.* Two JumpTargets (`repeat_start`) at the starts of MC 8 and MC 13,
     provenance method `manual`, stated by the editor.
  3. *Replace.* At each of the two coordinates one new Jump holding the destination of the
     rejected one (`repeat_end`, `repeat_count: 1`) with `to` naming its new start. A
     rejected Jump and its replacement share a coordinate.
  4. *Add the way past the third bracket.* A Jump at the end of MC 6 (the second of three
     brackets, which has no sign at its end) `to` the start of MC 8.

  The test asserts the bookkeeping (five elements more than the loader wrote, exactly two
  of them rejected, by the editor; three more elements read by the engine than before; the
  five added ones `manual`; the loaded timeline's own registry untouched) and then the
  result: `default` and `single` equal the file's rows without a finding, and the atomic
  sections equal its `atomic` rows. The file's `ms3` and `music21` rows are another tool's
  unfolding and are not computed.

## Signs written as flow control

A score loader writes repeat signs and brackets through the one translation of its measure map
(`tests/timelines/README.md`, "Measures and flow control"). Every other sign comes from the
columns of the measure table that each loader fills from its source: `markers`, `jump_bwd`,
`jump_fwd`, `play_until`, `breaks`, `barline`. `test_flow_gold.py` proves that writing on two
kinds of table.

**On the corpus**, where the expected elements are read off the table by hand:

- The Musette's whole `flow_control` array, nine elements, each commented with the cell it
  comes from. It shows the three readings of a marker: `segno` (named by `jump_bwd`) and `2e`
  (named by `jump_fwd`) are jump targets at their measures' starts, `1e` (named by no row) is
  one too, and `fine` (named by `play_until` only) is none: at the END of its measure it is a
  break ending the flow after the D.S. of MC 58, which states no `jump_fwd`, and a
  destination `to` the `2e` target after the D.S. of MC 31, which states one.
- The Rondeau: its `segno` is played until by both D.C. and is no jump target; the Jump at
  its measure's end holds the repeat end and the two destinations to `2e` and `3e`, each
  `after` its D.C., which is named without place keys.
- WoO 71: the twelve section breaks lie at the ends of MCs 21, 42, 72, 93, 114, 135, 156,
  177, 198, 219, 240, 261 and the five barlines (four `double`, one `end`) at the ends of
  MCs 53, 72, 261, 329, 371.
- `next`: the Musette's MC 16 states `1, 17, 32, -1`. The measure table keeps the cell and
  the measure map the successors `("1", "17", "32")`; nothing of it is flow control.

**On tables written by the test** (4/4 bars through `Ms3Loader`, so the real reading of
the columns is part of what is tested), one sign combination each, the whole array and the
runs it plays: a D.S. al Coda over a repeat (the To Coda destination shares the Jump of the
repeat end and names the leap with its place), a D.C. al Fine, two leaps playing until one
Fine (two breaks at one coordinate, each naming its leap), `repeats: startend`, `breaks`
cells with and without `section`, and markers no row names.

**What cannot be written is left out and logged.** Eight tables each state one sign the
elements cannot express, and each must give exactly one warning with its text and exactly the
remaining elements: a D.S. whose segno no measure carries; a Fine no D.C. or D.S. plays until
(nothing says when it would end the flow); a `jump_fwd` on a row without `jump_bwd`; a D.C.
on the barline of a repeat end without a printed start (both would go on at the section
start, and one Jump has one destination per place); a `play_until` and a `jump_fwd` naming a
marker no measure carries; a `jump_fwd` without `play_until`; and a To Coda in the measure of
its own leap.

## VerovioLoader

`test_verovio_loader.py` loads small MEI files written by the test module itself (into
`tmp_path`), so that every expected value follows from the encoding by the reasoning below, plus
provisioned specimens from the corpora `score`, `vienna_1x22` and `parangonar`. Values that only
Verovio decides (generated ids, grace-note placement) were established with **Verovio 6.4.1**
and are pinned to it: those tests skip on another version. The module is skipped when `verovio`
is not installed. Unless an option says otherwise Verovio plays 120 quarters per minute, so a
position of *q* quarters lies at *q*/2 seconds; every value below is a whole number of
milliseconds, which Verovio's integer `tstamp` states exactly.

### Fixtures and what they prove

**`basic.mei`**: two measures of 9/8 (`m1`, `m2`), two staves, one `<mdiv xml:id="mdiv1">`
without `@label`.

| Staff 1 | Staff 2 |
|---|---|
| m1: rest `r1` (8th) · chord `c1` (quarter, `@staff="2"`: `c1a` C4, `c1b` A3 with `accid.ges="f"`) · duplet `tu1` (`num=2 numbase=3`: `d1` E5, `d2` F5, 8ths) · `t1` G4 dotted quarter, tied to `t2` | m1: `b1` C3 dotted half · `b2` G2 dotted quarter |
| m2: `t2` G4 dotted quarter · `t3` A4 dotted half, tied to `#outside` (no such element) | m2: rest `r2` dotted quarter · one F♯3 dotted half **without `xml:id`** |

A `<tempo xml:id="tempo1" tstamp="1">Andante très expressif</tempo>` sits in m1.

- **Exact onsets.** An 8th is 1/2 quarter; a duplet 8th is 1/2 × 3/2 = 3/4. Hence `r1` 0–1/2,
  `c1a`/`c1b` 1/2–3/2, `d1` 3/2–9/4, `d2` 9/4–3, `t1` 3–9/2, `b1` 0–3, `b2` 3–9/2, `t2` 9/2–6,
  `t3` 6–9, `r2` 9/2–6, the generated note 6–9. The stored cells of `d2` hold numerator 9 and
  denominator 4; the timeline's number type is `Fraction`. All of these are dyadic, so a float
  reading of the timemap would store the very same cells: they prove the values, not that the
  loader reads exactly. `startid.mei` below holds the triplet that does.
- **In-measure offsets.** `mc_onset` is the position in the measure: `t2` 0, `t3` 3/2. `mn_onset`
  is not written (it would repeat `mc_onset`) and stays null.
- **Ids.** Note ids are the fixture's `<note>` `xml:id`s (read with the standard library from the
  fixture) plus one generated id; rests are `r1`, `r2`; the tempo indication is `tempo1` at 0 with
  its text verbatim. `notated_id` equals `id` on every event (nothing is expanded). The generated
  id is the same in two loads, and with 6.4.1 it is `s3clx4n` (seeded by `xmlIdChecksum`).
- **Structure.** The root is one `ContinuousLogicalTimeline` `score:clt_1` in quarters, length
  9. The `<mdiv>` has no name, so it makes no child (see "Named and unnamed `<mdiv>`s" below):
  the root has no children and holds all 13 events itself (10 notes, 2 rests, 1 tempo
  indication): its own events are every event of the tree. The
  `MeasureMap` has `m1`, `m2` named `1`, `2`, at 0 and 9/2, each 9/8 with actual length 9/2;
  neither measure id is an event. No flow control, no FlowMap.
- **Ties.** `t1` is `i`, `t2` `t`, `t3` `i`; every other note has none. The dangling tie `tie2`
  is listed under `dangling_ties`, logged once as a warning, and nothing is invented: `t3` ends at
  9, where the timemap ends it.
- **Staff and pitch.** The chord notes have staff 2 (the chord's `@staff`), voice 1 (the
  `<layer>` number, stored once, in `voice`) and `chord_id` `c1`; `r1` has staff 1, voice 1; no
  other event has a `chord_id`. `c1b` is `A♭3`, specific pitch `(A, -1, 3)`, MIDI pitch 56; the generated note is
  `F♯3`, MIDI pitch 54. For every note the MIDI pitch Verovio reports equals
  `12 × (octave + 1) + step + alter` of the stored spelling (two independent readings agree).
- **Physical timeline.** `create_physical_timeline()` is a `ContinuousPhysicalTimeline`
  `rendered:cpt_1` in seconds, number type `Fraction`, length 9/2, `meta["provenance"] ==
  "verovio_rendered"`, no child. Every event lies at half its quarter
  position (`d2` 9/8–3/2, `t3` 3–9/2, `tempo1` 0). Its events carry `mc` (1 and 2) but no
  in-measure offset: those are quarters, which have no place on a seconds axis. The logical timeline's C-Map, built with output number type `fraction`, maps
  9/4 quarters to exactly `Fraction(9, 8)` seconds, and the physical timeline's C-Map maps 9/8 back
  to exactly `Fraction(9, 4)`. `attach_cmap=False` attaches none.
- **Two-phase contract and options.** `from_file` and `load` give the same events;
  `create_timeline(uid="clt_1", name="Fixture")` keeps both and still has no child. The store
  metadata records the version and the options the toolkit ran with,
  `{"xmlIdChecksum": True, "mdivAll": True}`. A second source raises.

**`grace.mei`** (4/4, one staff): m1 `a1` C4 quarter · unaccented grace `g1` · `p1` D4 quarter ·
accented grace `g2` · `p2` F4 half; m2 unaccented grace `g3` at the measure start · `p3` C5 half ·
chord `ch1` (E4 G4 C5, half) under `<arpeg>`. Verovio's rules (`InitTimemapAdjustNotesFunctor`,
Verovio ≥ 4.4): an unaccented grace lasts 120/2048 = 15/256 quarter and ends at its principal's
notated onset, unless that would cross the measure start, where the principal is delayed instead;
an accented grace takes `@grace.time` (default 50 %) of its principal and delays it; an arpeggio
is rolled by the same 15/256 per note. Expected: `g1` 241/256–1, `p1` 1; `g2` 2–3, `p2` 3–4;
`g3` 4–1039/256, `p3` 1039/256–6; `ch1a` 6, `ch1b` 1551/256, `ch1c` 783/128, all ending at 8.
Field `grace` is `unacc`/`acc` verbatim from the `<graceGrp>`; principal notes have none. On
the physical timeline `g1` starts at 471/1000 s (241/256 quarter at 500 ms per quarter is 470.7 ms,
which Verovio rounds to the millisecond) and `p3` at 2029/1000 s: exact, not a binary fraction. Known
issue: these are Verovio's performed positions, not the notated onsets; the loader does not undo
them.

**`repeat.mei`** (2/4): `m1` (`left="rptstart"`, note `m1n`, tempo `tp1` at `tstamp="2"`),
`m2` (`right="rptend"`, `m2n`), `m3` (`m3n`). Verovio's default expansion plays m1 m2 m1 m2 m3:

- Events `m1n` 0, `m2n` 2, `m1n-rend2` 4, `m2n-rend2` 6, `m3n` 8 (length 10); the second-pass
  events have `notated_id` `m1n`, `m2n`. The tempo indication appears once per pass: `tp1` at 1
  and `tp1-rend2` at 5 (beat 2 of 2/4 is one quarter after the measure start).
- The timeline is the unfolded one: its `MeasureMap` lists `m1`, `m2`, `m1-rend2`, `m2-rend2`,
  `m3` named `1`, `2`, `1`, `2`, `3`, all 2/4, without flow control. FlowMap `forward_verovio` has the
  sections `[0, 4) → 0` and `[0, 6) → 4` on the printed axis (folded position 1 unfolds to 1 and
  5); FlowMap `source` is its inverse (5 folds to 1).
- The store's measure table is the printed one: three rows at 0, 2, 4, `m1` opening and `m2`
  closing the repeat. The flow control written from it plays MC 1 2 1 2 3, Verovio's order.
- With `options={"expandNever": True}` nothing is expanded: three events, the printed
  `MeasureMap` on the timeline, the repeat as one jump in its flow control, no FlowMap.

**`repeat_endings.mei`** (2/4): `m1`–`m2` a plain repeat, then `m3` (`rptstart`), first ending
`m4` (`rptend`), second ending `m5`. Verovio 6.4.1 renders m1 m2 m1 m2 m3: no second pass of m3
and neither ending. The expansion states no length for the printed measures `m4`, `m5`, so the
loader reads the printed lengths from a second Verovio rendering with `expandNever`; nothing is
computed by the loader. Expected: one warning naming the two measures; store metadata
`omitted_measures == ["m4", "m5"]`; the printed table `m1`…`m5` at 0, 2, 4, 6, 8, each 2 long,
identical to the table of a load with `expandNever` (whose `omitted_measures` is empty); the
rendered timeline of length 10 with the `MeasureMap` `m1`, `m2`, `m1-rend2`, `m2-rend2`, `m3`
and the events `m1n`, `m2n`, `m1n-rend2`, `m2n-rend2`, `m3n` (`m4n`, `m5n` are not rendered). The
printed table's default flow is MC 1 2 1 2 3 4 3 5, which is not Verovio's order: that
difference is the defect, and the hardening experiment below is where it is recorded.
`expandAlways` raises (it writes the expansion into the MEI tree, which then no longer states
the printed score).

**`two_movements.mei`**: `mdiv` `mov1` (`n=1`, `label=I`, 2/4: `n1` C4 and `n2` D4 quarters,
`n3` E4 half) and `mov2` (`n=2`, `label=II`, 3/4: `n4` F4 dotted half, `<tempo midi.bpm="60">`
`t2`). Verovio continues the second movement where the first ends: one timemap entry at 4 holds
the end of `n3`, the measure start of `m3` and the onset of `n4`; seconds continue (2 s).
Expected: root length 7; both `<mdiv>`s are named, so the children are `score:clt_2` [0, 4) and
`score:clt_3` [4, 7) (the same with `uid="score"`), named `I`, `II`, with `meta` holding `count`
(1, 2), `xml_id`, `n`, `label` verbatim and `element` `mdiv`. `n4` is at 0 in its child and
at 4 on the root; `t2` belongs to the second child. The root's `MeasureMap` (measures at 0, 2, 4)
and C-Map serve the children through the root: the root's timestamp table gives `n4` 2 seconds;
a child carries neither. On the physical timeline the second child starts at 2 s and lasts 3 s
(three quarters at 60 per minute); the children are `rendered:cpt_2` and `rendered:cpt_3`. Both C-Maps
are exact between anchors as well: 10/3 quarters are exactly 5/3 seconds and 13/3 quarters 7/3
seconds (the second across the tempo change), and back. With `options={"mdivXPathQuery": "./mdiv[2]"}` only the second
movement is loaded: one child `score:clt_2` whose `meta["count"]` is still 2 (`count` numbers every
`<mdiv>` of the file), at 0 with length 3.

**Named and unnamed `<mdiv>`s.** An `<mdiv>` is a child timeline only when it has a name, its
`@label`; an `<mdiv>` without `@label`, or with an empty or blank one, makes no timeline (`@n`
is a number, not a name). What an unnamed `<mdiv>` holds, events and measures, belongs to the
nearest named ancestor `<mdiv>`'s child, else to the root, and a named `<mdiv>` nests under the
child of its nearest named ancestor, else under the root. A named child's `name` is the label
verbatim and its `meta` keeps `count` (its place among every `<mdiv>` of the file, unnamed ones
included), `xml_id`, `n`, `label`, adding `"element": "mdiv"`; its id is minted as before. What
proves the rule: the tree (which timelines exist, their names, meta, offsets, lengths, and which
timeline holds each event) changes with the labels alone, while every event's position on the
root stays the one the timemap states, so the holder moves and no coordinate does. Each fixture
is 2/4, one half note `{measure}n` per measure, so measure `mK` starts at `2(K-1)` quarters:

- `basic.mei` (one unnamed `<mdiv>`): no child; the root holds every event (above).
- `named.mei`: one `<mdiv xml:id="solo" label="Rondo">` with `m1`, `m2`. Root length 4, one child
  `score:clt_2` named `Rondo`, offset 0, length 4, meta `{"count": 1, "xml_id": "solo", "label":
  "Rondo", "element": "mdiv"}`, holding `m1n` at 0 and `m2n` at 2; the root holds no event. The
  physical timeline has the same tree, `rendered:cpt_2` of length 2 seconds.
- `siblings.mei`: `<mdiv xml:id="first" label="I">` (`m1`, `m2`), then an unnamed `<mdiv
  xml:id="second">` (`m3`). Root length 6, one child `score:clt_2` named `I` at offset 0, length 4
  (its own measures only, not the unnamed sibling's), meta count 1, holding `m1n`, `m2n` at 0 and
  2; the root holds `m3n` at 4. On the root, `m1n`, `m2n`, `m3n` are at 0, 2, 4.
- `nested.mei`: an unnamed `<mdiv xml:id="work">` holds a named `<mdiv xml:id="set"
  label="Variationen">` and an unnamed `<mdiv xml:id="coda">` (`m4`); `set` holds an unnamed
  `<mdiv xml:id="group">`, which holds a named `<mdiv xml:id="v1" label="Var. 1">` (`m1`, `m2`)
  and an unnamed `<mdiv xml:id="v2">` (`m3`). Document order counts `work` 1, `set` 2, `group`
  3, `v1` 4, `v2` 5, `coda` 6. Expected: root length 8 with one child `score:clt_2`
  `Variationen` (count 2) at offset 0, length 6; that child has one child `score:clt_3` `Var. 1`
  (count 4) at offset 0, length 4: the unnamed `work` and `group` are skipped, and `v1` nests
  under `set`. `Var. 1` holds `m1n`, `m2n` (0, 2); `Variationen` holds `m3n` (at 4 in it, from the
  unnamed `v2` within the unnamed `group`); the root holds `m4n` (at 6, from the unnamed `coda`
  under the unnamed `work`). On the root the four notes are at 0, 2, 4, 6.

**`fast.mei`**: three 256th notes at `midi.bpm="2000"`, 1/64 quarter apart, then rests to the end
of the 2/4 measure; Verovio rounds the first four entries to the milliseconds 0, 0, 1, 1.
`create_timeline()` works (length 2; the C-Map only needs increasing quarters); `create_physical_timeline()` raises, naming the coinciding entries, because their
order in seconds is not stated.

**Measures of `<mSpace>`** (`space_first.mei`, `space_middle.mei`, `space_last.mei`). An
`<mSpace>` is a measure-long space: valid MEI that states no event, only that the measure passes.
Verovio times a measure by its timed elements, so it gives a measure holding nothing but
`<mSpace>` no timemap entry and no time: in `space_middle.mei` below, Verovio 6.4.1 starts the
third measure at 3, where the first ends, and plays its notes at 3/2 s. Read verbatim, the measure has no length, and a zero-length measure is no measure of the
format (its `actual_length` is strictly positive). What the source states is the meter, and an
`<mSpace>` fills its measure as a measure rest does, so the loader renders Verovio's MEI output
once more with each `<mSpace>` of such a measure written `<mRest>` (same `xml:id`): Verovio then
times the measure itself, at its meter's length and the tempo in force there, in every pass of an
expansion, and nothing is computed by the loader. The rests are no events. Where the measure
stands does not matter: first, in the middle or last (where Verovio opens it at the timemap's end
with no time), it is rendered in its place. Each fixture is 3/4
on two staves, three measures `m1`, `m2`, `m3` (named `1`–`3`), one of them holding only
`<mSpace>` in each layer; the other two hold, in order, staff 1 `x1` quarter + `x2` half, staff 2
`y1` dotted half (the first), and staff 1 `z1` dotted half, staff 2 `w1` half + `w2` quarter (the
second). Expected, for each: root length 9 quarters, the `MeasureMap` `m1`, `m2`, `m3` at 0, 3, 6,
each 3/4 with actual length 3; exactly 6 events, all notes, no rest, none at the `<mSpace>`'s
`xml:id`:

| Fixture | `<mSpace>` in | Onsets of `x1 x2 y1 z1 w1 w2` | First note after the space, in seconds |
|---|---|---|---|
| `space_first.mei` | `m1` | 3, 4, 3, 6, 6, 8 | `x1` at 3/2 |
| `space_middle.mei` | `m2` | 0, 1, 0, 6, 6, 8 | `z1` at 3 |
| `space_last.mei` | `m3` | 0, 1, 0, 3, 3, 5 | — (the physical timeline ends at 9/2) |

The physical timeline is 9/2 s long in each case (nine quarters at 120 per minute), and the C-Map
maps 6 quarters to exactly 3 s. Store metadata `omitted_measures` is empty (nothing is
expanded). `space_empty.mei`, `space_middle.mei` with `m2`'s layers left empty, still raises
naming `m2`: a measure without any content, not even an `<mSpace>`, states no length.

What would fail if the measure were timed by the loader instead of by Verovio, the two cases
below: a repeat (the measure must be played as often as the expansion plays it) and a tempo
change around it (the measure must last at its own tempo, not the next measure's). Each is 2/4
on one staff, a measure `mK` holding a half note `mKn` (C4, D4, E4 by measure) or only
`<mSpace xml:id="mKsp">`; a tempo indication `tK` stands at beat 1 of its measure with its
`midi.bpm`.

- **`space_repeat.mei`**: `m1` (`rptstart`, `m1n`), `m2` (`<mSpace>` only, `rptend`), `m3`
  (`m3n`). Verovio plays m1 m2 m1 m2 m3, so the expanded reading is the unexpanded one unfolded:
  root length 10 quarters; `MeasureMap` `m1`, `m2`, `m1-rend2`, `m2-rend2`, `m3`, named `1`, `2`,
  `1`, `2`, `3`, at 0, 2, 4, 6, 8, each 2/4 with actual length 2; exactly 3 events, the notes
  `m1n` 0, `m1n-rend2` 4, `m3n` 8 (no rest, nothing at `m2sp`); store metadata
  `omitted_measures` empty; the printed table `m1`, `m2`, `m3` at 0, 2, 4, each 2 long, with
  default flow MC 1 2 1 2 3. At 120 per minute the physical timeline is 5 s long and `m3n` starts
  at 4 s. With `expandNever` the same file reads as printed: length 6, `MeasureMap` `m1`, `m2`,
  `m3` at 0, 2, 4, each 2, events `m1n` 0 and `m3n` 4, the same printed table.
- **`space_tempo.mei`**: `m1` (`m1n`, `t1` 60), `m2` (`<mSpace>` only), `m3` (`m3n`, `t3`
  120). `m2` is played at 60, the tempo before it: 2 quarters last 2 s, and `t3` applies from
  `m3` on. Expected: length 6 quarters, `MeasureMap` at 0, 2, 4 each 2; `m3n` and `t3` at 4
  quarters. In seconds `m1n` and `t1` 0, `m3n` and `t3` 4 (not 3, which the next measure's tempo
  would give), physical length 5; the C-Map maps 2 quarters to 2 s and 4 to 4 s.
- **`space_tempo_inside.mei`**: as `space_tempo.mei`, with `t2` 30 in `m2`. A tempo written in the
  silent measure is its tempo: 2 quarters at 30 per minute last 4 s. Expected: `t2` at 2 quarters
  and 2 s, `m3n` and `t3` at 4 quarters and 6 s, physical length 7.
- **`space_first_tempo.mei`**: the score's initial tempo `midi.bpm="60"` on the `<scoreDef>`,
  `m1` (`<mSpace>` only), `m2` (`m2n`, `t2` 120). The first measure is played at the initial
  tempo: `m2n` and `t2` at 2 quarters and 2 s, physical length 3.

All seconds here are whole milliseconds, so no value depends on Verovio's rounding.

**`startid.mei`** (2/4): m1 `a1` C4 quarter, chord `ch1` (`ch1a` E4, `ch1b` G4, quarter) with
`<tempo xml:id="tc" startid="#ch1">Allegro</tempo>`; m2 a triplet `tu` (`num=3 numbase=2`: `u1`,
`u2`, `u3`, 8ths) and `a2` F4 quarter, with `<tempo xml:id="tn" startid="#u2">Meno mosso</tempo>`.
A triplet 8th is 1/2 × 2/3 = 1/3 quarter, so `u1` 2–7/3, `u2` 7/3–8/3, `u3` 8/3–3. `u2`'s start
cell holds numerator 7 and denominator 3, and its `mc_onset` 1/3: no binary fraction equals a
third, so only an exact reading of `qfrac` stores these cells. A tempo indication naming a chord
sits at the chord's first note (`tc` at 1, where `ch1a` starts); one naming a note sits at that
note (`tn` at 7/3, `mc_onset` 1/3). `ch1a` and `ch1b` have `chord_id` `ch1`.

**Duplicate ids.** Two `<dynam xml:id="dd">` in `repeat.mei` load (listed under
`duplicate_ids`); two notes sharing `xml:id="m1n"` raise, naming `m1n`.

### Provisioned specimens

- **Nested movements** (`parangonar/Beethoven_Eroica_op35-cpjku/score.mei`): `<mdiv
  xml:id="Eroica-Variationen">`, which has no `@label`, holds `<mdiv xml:id="Var_14" n="18"
  label="Var. XIV.">`. The outer `<mdiv>` makes no child; the inner one is the root's one child
  `score:clt_2`, named `Var. XIV.`, meta `{"count": 2, "xml_id": "Var_14", "n": "18", "label":
  "Var. XIV.", "element": "mdiv"}`, spanning [0, 129/2) like the root. The 297 notes, 36 rests
  and 2 tempo indications live in that child, none in the root. The `MeasureMap` holds 33
  measures. No provisioned file has two sibling movements; `two_movements.mei` covers that case.
- **Repeat facts** (`score/flow_control/out_of_the_flow_experience-flow_only.mei`, read with
  `expandNever`): the measure rows carry the same facts as the hand-checked
  `flow_only/out_of_the_flow_experience-flow_only-canonical.measures.tsv` (15 measures;
  `start_repeat` at MCs 4, 9, 10; `end_repeat` at 3, 5, 8, 10, 11, 14; voltas (5, 1), (6, 2),
  (7, 3), (14, 1), (15, 2)). Known issue: Verovio 6.4.1 crashes the process in
  `renderToTimemap` when it expands this file's repeats, so no test loads it expanded.
- **Repeat marks** (same file). Each `<repeatMark>` is read into the navigation columns, with
  the meaning they have for every score loader: `markers` names the segno, coda and Fine marks
  of a measure (joined by ` & `, a second segno or coda of the score numbered, `coda2`);
  `jump_bwd` is `segno` for `func="dalSegno"`, `start` for `func="daCapo"`; "al Coda" in the
  text plays from the jump's target to the first coda mark there (the "To Coda" sign) and
  continues at the first coda mark after the jump; any other jump plays until the Fine where the
  score has one, which is what a Fine states. The file states a Fine in MC 7, a segno and a "To"
  coda in MC 9, `D.S. al Coda` in MC 12, a coda in MC 13 and `D.C. al Fine` in MC 15, so the
  columns `(markers, jump_bwd, jump_fwd, play_until)` are exactly: MC 7 `fine`; MC 9
  `segno & coda`; MC 12 `segno`, `coda2`, `coda`; MC 13 `coda2`; MC 15 `start`, `fine`. The
  MEI states no successors, so `next` is empty. The flow these signs play is the subject of
  the parsing matrix.
- **The Fine rule** (`couperin_concerts/c05n05_musete.mei`, `expandNever`). Segnos in MCs 6, 17
  and 32 (`segno`, `segno2`, `segno3`), a Fine in MC 16, a `daCapo` in MC 31 and a `dalSegno` in
  MC 58, both with the text "au renvoy" (no "al …"). Both jumps play until the Fine: MC 31
  `start`, `fine`; MC 58 `segno`, `fine` (the first segno).
- **Acceptance against music21** (`vienna_1x22/Chopin_op10_no3.musicxml`): `VerovioLoader` and
  `Music21Loader` must give the same multiset of `(onset, MIDI pitch)` over the non-grace notes,
  once Verovio's placement of principal notes is undone by its own rule: a non-grace note that
  starts where a grace note of the same staff and layer ends is a principal; after an accented
  group its notated onset is the onset of the group's first grace note, after an unaccented one
  its own onset. Grace notes are excepted on both sides (music21 states them without duration).
  The file has 4 grace notes and no arpeggio.
- **Hardening: Verovio's expansion against the played flow control** (slow). For each specimen
  the flow control written from the printed measure table (`store.measures`) is played by the
  engine (`default`), and the resulting
  MC sequence must equal Verovio's rendered measure order, read through the public surface: each
  rendered measure's start, folded through FlowMap `source`, is a printed measure start. On the
  `repeat.mei` fixture they are equal (above). Every provisioned score differs, each a strict
  `xfail` with its cause:
  - `couperin_concerts/c05n05_musete.mei` and `couperin_concerts/c11n08_Rondeau.mei`:
    **Verovio**'s expansion does not follow D.S. and D.C. marks (it renders 116 and 120
    measures, the repeats only), while the measure table states them (a D.C. in MC 31 and a D.S.
    in MC 58; D.C.s in MCs 18 and 27).
  - `beethoven_woo71/WoO71.mei`: **Verovio** leaves out both endings numbered bar 19 (MCs 260,
    261) and does not repeat their passage: 495 rendered measures, where the printed table
    unfolds to 505. The load reads the two lengths from the rendering with `expandNever`.
  - `beethoven_op18-4iv_multimodal/ABC/n04op18-4_04.mei`: **Verovio** leaves out the endings at
    MCs 44–45, 93–94 and 102–103: 254 rendered measures, where the printed table unfolds to 291.
  - `beethoven_op18-4iv_multimodal/recordings/Beethoven_Op018No4-04.mei`: **the source** opens a
    first `<ending>` at MC 44 that closes only at the end of the score, with the second ending
    nested in it (Verovio logs `Adding 'ending' to a 'ending'`). The measure table therefore
    states volta 1 from MC 44 on, one bracket to the end of the score, which the played flow
    control and Verovio's rendering read differently.
  - Known issue, not loaded: Verovio 6.4.1 crashes the process (segmentation fault) in
    `renderToTimemap` when it expands `OpenScoreSQ/sq8913219.mei`; with `expandNever` it reads.
  - Not usable: the `*_flow.mei` skeletons have measures without content, which Verovio gives no
    timemap entry; the load raises naming them.
  - Known issue, not loaded: Verovio 6.4.1 crashes the process on the Rachmaninoff MEI
    (`rachmaninoff_concerto2/score/…Rachmaninoff.mei`), with `expandNever` too.
- **Omitted endings in provisioned scores** (slow, Verovio 6.4.1). WoO71 and `n04op18-4_04.mei` load with
  Verovio's default expansion: `omitted_measures` names exactly the measures at MCs 260, 261 and
  at MCs 44, 45, 93, 94, 102, 103, and the rendered `MeasureMap`s hold 495 and 254 measures.
- **Id check overhead.** Measured on the largest provisioned MEI (`wagner_walkure/…/
  Wagner_WWV086B-3.mei`, 8.8 MB, 76 623 ids, Verovio 6.4.1): `loadFile` 3.9 s and the timemap
  3.1 s; `getMEI` 0.09 s and its parse 0.21 s, which the loader needs anyway for membership; the
  uniqueness count itself 0.02 s. In that file 15 ids name more than one element (its MuseScore
  export repeats ids of `<dynam>`, `<dir>` and `<fermata>` elements); they are listed, not refused.

### Ids of a given root

`test_verovio_ids.py` writes one MEI with two named `<mdiv>`s (one 2/4 measure
each, `@label` `I` and `II`, so each is a child) and asks for the ids of the root and its two movement children. The loader
mints `{role}:{prefix}_N`, the root's number first. A given root id of that
same shape names a number the movements could be minted; the loader takes it
from them, so:

* `create_timeline()` gives the root `score:clt_1` and movements
  `score:clt_2`, `score:clt_3`;
* `create_timeline(uid="score:clt_2")` gives the root `score:clt_2` and
  movements `score:clt_3`, `score:clt_4`. Without the reservation the first
  movement is minted `score:clt_2`, collides with the root and is re-minted
  on joining as `clt_1`, losing its role; the exact list tells the two apart;
* a given id led by another role (`uid="perf:clt_2"`) reserves its number
  too, since the number counts every timeline of the prefix whatever its
  role: the movements are `score:clt_3`, `score:clt_4`. The rule strips any
  role, not only the loader's own; it is the one rule `ScoreWarpLoader`
  follows as well (`tests/loader/README.md`, "`ScoreWarpLoader` validation
  logic");
* a given id of another shape (`uid="score"`) reserves nothing a movement
  could take: the movements stay `score:clt_2`, `score:clt_3`;
* `create_physical_timeline(uid="rendered:cpt_2")` gives movements
  `rendered:cpt_3`, `rendered:cpt_4`: the same rule on the seconds tree.

Each case uses a fresh loader, since a loader's minted ids count across its
calls.

### Encodings without music

An MEI file can hold a header (`<meiHead>`) that describes a work and no music: the six
header-only reference files of the CollabScore corpus end in `<music xml:id="g-3"/>`
(none of them is test data). Verovio 6.4.1 declines most such shapes, but its Python binding
writes the reason to standard error, so `getLog()` is empty and the refusal used to end after
its colon; a `<score>` holding only a `<scoreDef>` loads and then crashes the process in
`renderToTimemap`. What proves the refusal is useful and the reader still accepts music:

- **Every shape names its element.** One header-only MEI per shape is written into
  `tmp_path`: no `<music>`, `<music/>` empty, `<body/>` empty, `<mdiv/>` empty, a `<score>`
  with a `<scoreDef>` and no `<section>`, an empty `<section>`. Each raises `ValueError`
  whose whole message is pinned: the file holds an MEI header and no music, the named element
  (`it has no <music> element`, `its <music> holds no <body>`, `its <body> holds no <mdiv>`,
  `its <mdiv> holds no <score>`, `its <score> holds no <section>`, `its <section> holds no
  <measure> and no <staff>`), and that such an encoding describes a work but can give no
  timeline. A generic message, or one naming the wrong element, fails. The `<scoreDef>` case
  also proves the file is read before Verovio renders it: otherwise the worker crashes.
- **Without a header** (`<mei><music/></mei>`) the message says "holds no music", not "an MEI
  header and no music": the loader states only what the file holds.
- **A refusal Verovio gives without a reason** says so instead of ending after the colon: an
  XML file Verovio cannot identify (`<foo/>`) raises `Verovio could not load <path>: Verovio
  gave no reason (its messages, if any, are on standard error)`, through the real toolkit.
  The same wording ends an empty timemap's message; a toolkit stub whose log is empty and whose
  timemap is `[]` pins it, and one whose log holds a line pins that the line is passed on.
- **Music still loads.** The fixtures above hold measures and load unchanged; the check stops
  at the first `<measure>` or `<staff>` inside `<music>`, and a file that is not MEI XML
  (MusicXML, Humdrum) is left to Verovio, which the provisioned specimens exercise.

### Guards

Without `verovio` the lazy export raises `ImportError` naming `timetoalign[verovio]`. A toolkit
older than 4.4 (no exact timemap positions) raises `ImportError` naming the version. A float
`qstamp` that is not the nearest double to the exact position makes the cross-check raise.
`create_timeline()` and `create_physical_timeline()` before `load()` raise `ValueError`.

## Running the tests

From the repository root:

```bash
python -m pytest --runslow tests/loader/score
```
