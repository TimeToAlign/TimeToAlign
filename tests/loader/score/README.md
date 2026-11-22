# Symbolic score loader tests

These tests validate the symbolic score loaders against exact source-derived values. TSV data
loaded by `Ms3Loader` is authoritative whenever loaders are compared.

## Test files

| File | Remaining coverage |
|---|---|
| `conftest.py` | Seven specimen definitions, source lookup, size guards, and `.flow.csv` parsing. |
| `test_score_parsing_matrix.py` | CSV modes, folded and unfolded counts, loader/format combinations, exact flow-control profiles, target-flow reproduction, cross-loader counts, and pinned parser differences. |
| `test_loaders.py` | MS3 coordinate and duration semantics, pitch schema, Partitura measure/MIDI regressions, shared `mc_onset`, anacrusis offsets, and parquet round trips. |
| `test_cross_validation.py` | Exact Chopin note-count, MIDI-pitch and pitch-name agreement across MS3, Partitura, and Music21; measure onsets only where the source states them (see "Cross-loader parity" below). |
| `test_measuremap_loader.py` | MeasureMap expansion, validation, traversal, schema, and comparison with MS3. |
| `test_flow_csv_validation.py` | Right-open ranges, exact atomic boundaries, live Partitura boundaries, and CSV serialization. |
| `test_partitura_exactness.py` | Exact Partitura coordinate pairs, instant nulls, flattened timelines, and agreement with Partitura's float map. |
| `test_ms3_facets.py` | `Ms3Loader`'s companion discovery in both corpus layouts and its reading of empty note cells, and its refusal of a measure without a count or a length (see "MS3 facet discovery", "MS3 empty note cells" and "MS3 measures without a count or a length" below). |
| `test_successors.py` | The `next` column derived from the measure columns: endings of more than one measure, Music21/Partitura parity, agreement with the ms3 TSVs (see below). |
| `test_verovio_loader.py` | `VerovioLoader` on MEI fixtures and specimens: exact onsets, kept ids, measures as structure, ties, grace notes, repeats, movements, the physical timeline, music21 parity, Verovio's expansion against the flow controller (see below). |
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

For every available source, the matrix checks TSV and MeasureMap folded counts, MeasureMap
default traversal length, Music21 MusicXML and MEI folded counts, Partitura MusicXML and MEI
folded counts, and same-loader MEI/MusicXML equivalence where both formats exist. Files larger
than 500,000 bytes use the documented size skip because third-party parsing is too slow for
this integration suite.

`TestVerovioLoaderMEI` reads each matrix score's MEI with `VerovioLoader` (`expandNever`: the
printed measure table is the same either way, and Verovio crashes on expanding the
flow-control files) and requires the folded count and the target `default` flow from the
printed table. `polyrhythm_only` and `flow_only` match exactly (`flow_only`: 31 visits, D.S. al
Coda, a To-Coda jump, voltas 1–3 and D.C. al Fine, all from the MEI's own marks). Each residue is
a strict `xfail` naming what the MEI states:

- `c05n05_musete`: a `daCapo` at MC 31, where the target flow jumps to the segno.
- `c11n08_Rondeau`: `daCapo`s at MCs 18 and 27 with neither a Fine nor an "al …" text, so
  nothing states where they end; the traversal stops at the cycle.
- `WoO71` (slow): no section breaks, which cut the target's sections. The MC sequence itself is
  the target's, all 505 visits, asserted by `test_woo71_mc_sequence`.
- `op18_no4_mov4_flow`: a skeleton whose measures have no content, hence no timemap entry; the
  load raises `ValueError`.

Rachmaninoff is not in the Verovio matrix: Verovio 6.4.1 crashes the process on its MEI.

Every `.flow.csv` must exist and contain `atomic` and `default` modes. Its number of atomic rows
must equal the specimen value above. MS3 must reproduce one documented valid default flow and
the exact atomic boundaries. TSV and MeasureMap must have identical folded counts.

### Successors derived from the measure columns

A source that lists each measure's successors (an ms3 TSV, a Measure Map) keeps them. For every
other score loader `ScoreLoader.load()` derives the `next` column, per part, from the repeat,
volta and navigation columns of the measure table (`ScoreLoader._successors`), so one rule
serves Music21, Partitura and Verovio. An ending is a run of measures that all carry its volta
number. A repeat end goes back to its repeat start and, outside an ending, on to the following
measure. A D.S. or D.C. replaces the onward successors with its target and keeps a repeat end's
way back; a Fine adds `-1`, the end of the piece.

`test_successors.py` proves the rule against two references:

- **An ending of two measures.** The test writes a MusicXML file of seven 4/4 measures: m1 opens
  a repeat, the first ending is m4–m5 with the repeat end at m5, the second ending is m6, m7
  closes. Each loader must write volta `1` on m4 *and* m5 and `2` on m6. Then `next` is `4, 6`
  on m3 (the way into each ending), `1` on m5 (back to the repeat start; the way on is the next
  ending), null (the following measure) everywhere else, and the default flow is
  `1 2 3 4 5 1 2 3 6 7`. Music21Loader and PartituraLoader must both give exactly this, hence
  identical `next` columns. Partitura used to write the volta on an ending's first measure only,
  which cut the ending after m4 and lost the repeat (m3 `4, 5`, m5 `6, 7`). Music21's former
  derivation sent every measure of a first ending without a repeat end to the exit (m4 `7`);
  under the shared rule m4 follows on to m5.
- **Agreement with ms3.** The ms3 TSV is the source format of `next` and lists every successor,
  so it is the reference. On `c05n05_musete` and `c11n08_Rondeau`, the derived `next` of every MC,
  read as a set (null meaning the following MC, `-1` after the last), must equal the TSV's. The
  sources are the two that carry navigation marks: PartituraLoader on the MusicXML and
  VerovioLoader on the MEI (Music21 reads none). This pins the jump rule: `c05n05_musete` MC 31
  is `{17, 6}` and MC 58 `{32, 6}` (back to the repeat start, then the D.S. to the segno at MC 6,
  never on to the following measure); `c11n08_Rondeau` MC 18 is `{10, 1}` and MC 27 `{19, 1}`
  (the D.C.); and `c05n05_musete` MC 16, a repeat end with the Fine, holds `-1` beside `1, 17`.
  Before, a D.S. or D.C. replaced the repeat's edges (MC 31 was `6`).

Where a source cannot state what the TSV lists, the comparison of that MC is a strict `xfail`
naming the reason, and the remaining MCs of the score still have to agree:

- `c05n05_musete` MC 31 on the MEI: the MEI states a `daCapo` where the TSV has a D.S., so the
  derived set is `{17, 1}`.
- `c05n05_musete` MC 16 (both sources): the TSV continues after the D.S. al Fine at the next
  section, MC 32; the derived set is `{1, 17, -1}`. Neither the MEI nor the rule states a
  continuation after the Fine.
- `c11n08_Rondeau` MC 9 (both sources): the TSV continues after each D.C. al segno at the next
  section, MCs 19 and 28; the derived set is `{1, 10}`, for the same reason.

The derived successors are an input of the flow controller, so each loader's default flow on
every score of the matrix, and each pinned deviation below, must stay exactly what it is. The rule
writes edges the two former loader-specific derivations did not (Music21 wrote none when a score
had D.S., D.C., segno, coda or Fine marks; Partitura wrote no third volta and no exit after a
non-final volta without a repeat end), and none of them changes a flow on these scores: the
traversal routes voltas by pass, and a D.S. or D.C. is executed only where the measure table
names its target.

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

- Music21 produces 116 visits for `c05n05_musete`, with sections
  `[(1, 17), (1, 32), (17, 59), (32, 59)]`.
- Music21 produces 120 visits for `c11n08_Rondeau`, with sections
  `[(1, 10), (1, 19), (10, 28), (19, 61), (28, 61)]`.
- Music21's exact `flow_only` MC sequence and nine right-open sections are asserted directly.
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
- **Measure onsets are stated, never copied.** `mn_onset` is a note's position in its notated
  measure, `mc_onset` its position in the measure count; they differ wherever a notated measure
  is split, for instance at an anacrusis. The TSV states both (its first note: `mc_onset` 0,
  `mn_onset` 3/8), so `Ms3Loader` keeps them. MusicXML states neither as such and Partitura and
  Music21 derive only `mc_onset`, so their `mn_onset` is null on every note, control and
  annotation row: a copy of `mc_onset` there would claim a fact the source does not state, and
  the TSV shows that it would be wrong.

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
- `ScoreStore` and loader parquet tests pin files, note/measure counts, metadata, units, empty
  facets, missing paths, and event-property round trips.

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

There are no `pytest.approx` calls in the score-loader tests. Exact integers, Fractions, binary
halves, zero offsets, metadata values, and coordinate extents use equality. The only tolerance is
the explicit `< 1e-6` comparison in `test_partitura_exactness.py`; it compares exact rational
coordinates with values returned by Partitura's genuinely floating-point `quarter_map`.

## MeasureMap and flow CSV contracts

### Timeline-native structure on the Rondeau specimen

The Couperin Rondeau expectations are derived directly from
``c11n08_Rondeau.measures.tsv`` resolved through the score corpus. Its 60 rows
carry repeat ends at MC ``9``, ``18``, ``27``, and ``60``. The explicit repeat
starts and piece-start fallback therefore pair those ends with MC ``1``, ``10``,
``19``, and ``28`` respectively. The timeline registry must contain those exact
repeat jumps, with each source coordinate equal to that measure's exact
``qstamp + actual_length`` and each destination equal to the target measure's
exact ``qstamp``. This derives expected coordinates from independently parsed
measure facts rather than copying controller output.

The ``jump_bwd`` column adds two da capo jumps. It holds ``start`` on MC ``18``
and MC ``27`` and is empty everywhere else. A backward target of ``start`` names
the beginning of the piece, so each jump leaves the end of its measure and lands
on the start of MC ``1``. Prefix sums of ``act_dur`` × 4 over the timeline's own
measure map give the end of MC ``18`` as ``23 + 1 = 24`` quarters and the end of
MC ``27`` as ``35 + 1 = 36`` quarters. MC ``1`` starts at ``0``, so the jumps are
exactly ``24 → 0`` and ``36 → 0``, in quarters. The ``play_until`` cells
(``segno``) do not appear in the ``coda`` or ``fine`` spellings that would make
them da capo al coda or al fine jumps. The registry therefore holds exactly
``6`` jumps: ``4`` repeat ends and ``2`` da capo jumps.

The TSV has no ``breaks`` cell containing ``section`` and no ``fine`` marker, so
the exact break count is ``0``. Its sole flow target marker in the supported
segno/coda/fine vocabulary is ``segno`` at MC ``9``; the exact marker coordinate
is the start of that measure, ``Fraction(11)`` quarters. The other ``markers``
cells (``1e``, ``2e``, and ``3e``) name ending boundaries, not segno/coda/fine
targets, and therefore do not enter the registry. These values are pinned from
the source columns and the documented mapping: section and fine breaks occur at
measure ends, while target markers occur at measure starts.

Both normal and flattened timeline creation must store one ``MeasureMap`` and the
same flow-control facts: the timeline holds its map as one object, returned
identically on every access, with exactly ``60`` measures (one per row of the
TSV), and flattening merges events only, never dropping the map or a flow-control
fact. The normal timeline is serialized through JSON
and restored. The restored payload must equal the original payload, which is a
fixpoint over every structural key. The restored registry must hold exactly
``6`` jumps. The restored measure map must equal the original under type-aware
measure equality.

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
without it and the measure map leaves it out; the loader's own expanded data,
measure table and default traversal keep it where the file states it, since an
ending bar such as `c05n05_musete`'s with successors `1` and `-1` ends the piece
on its second visit. Every folded corpus `.mm.json` file states `-1`, so loading
`WoO71.measures.mm.json` at all pins the conversion, and the exact traversal
counts (138 for both Couperin pieces, 505 for WoO71) pin that the end survives. Measures are
built from the expanded rows themselves, never from the measure table the loader
also fills. On the corpus, `WoO71.measures.mm.json` opens with a `MeasureConstituent`
at offset 1, exactly the offset `WoO71.measures.tsv` states in its `mc_offset`
column for MC 1.

MeasureMap tests require unique MC values, monotonic `qstamp`, valid `next` references, exact
linear/repeat traversals, 397 folded and 505 unfolded WoO71 measures, and matching MS3 MCs and
repeat presence. Its schema must include flow-control and identity fields. The synthetic summary
contains three measures, one repeat start, and one repeat end.

Flow CSV tests enforce right-open contiguous atomic ranges. Exact samples are
`c05n05_musete`: `A=(1, 6)`, `B=(6, 17)`, `C=(17, 32)`, `D=(32, 59)`;
`polyrhythm_only`: `A=(1, 15)`; and `rachmaninoff`: `A=(1, 375)`. Live Partitura segments are
compared against the corresponding CSV rows, and `Flow.from_csv` must survive a records round trip.

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
  closing the repeat. Its `ScoreFlowController` default flow is MC 1 2 1 2 3, Verovio's order.
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
  columns `(markers, jump_bwd, jump_fwd, play_until, next)` are exactly: MC 7 `fine`, next
  `8, -1` (the Fine also ends the piece); MC 9
  `segno & coda`; MC 12 `segno`, `coda2`, `coda`, next `9`; MC 13 `coda2`, next `14, 15` (the
  measure before the endings); MC 15 `start`, `fine`, next `1`. Unfolded by the flow
  controller, this table gives exactly the target default flow (parsing matrix).
- **The Fine rule** (`couperin_concerts/c05n05_musete.mei`, `expandNever`). Segnos in MCs 6, 17
  and 32 (`segno`, `segno2`, `segno3`), a Fine in MC 16, a `daCapo` in MC 31 and a `dalSegno` in
  MC 58, both with the text "au renvoy" (no "al …"). Both jumps play until the Fine: MC 31
  `start`, `fine`, next `17, 1`; MC 58 `segno`, `fine`, next `32, 6` (the first segno). Both
  measures end a repeat, which keeps its way back; MC 16, the Fine and a repeat end, has next
  `1, 17, -1`.
- **Acceptance against music21** (`vienna_1x22/Chopin_op10_no3.musicxml`): `VerovioLoader` and
  `Music21Loader` must give the same multiset of `(onset, MIDI pitch)` over the non-grace notes,
  once Verovio's placement of principal notes is undone by its own rule: a non-grace note that
  starts where a grace note of the same staff and layer ends is a principal; after an accented
  group its notated onset is the onset of the group's first grace note, after an unaccented one
  its own onset. Grace notes are excepted on both sides (music21 states them without duration).
  The file has 4 grace notes and no arpeggio.
- **Hardening: Verovio's expansion against `ScoreFlowController`** (slow). For each specimen the
  printed measure table (`store.measures`) is unfolded with `FlowMode.default`, and the resulting
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
    states volta 1 from MC 44 on; the flow controller repeats MCs 29–44 and skips the rest of the
    "ending" (232 MCs), while Verovio plays MCs 44–223 once with no further repeats (250 MCs).
    They first differ at the 72nd rendered measure.
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

### Guards

Without `verovio` the lazy export raises `ImportError` naming `timetoalign[verovio]`. A toolkit
older than 4.4 (no exact timemap positions) raises `ImportError` naming the version. A float
`qstamp` that is not the nearest double to the exact position makes the cross-check raise.
`create_timeline()` and `create_physical_timeline()` before `load()` raise `ValueError`.

## Running the tests

From the repository root:

```bash
/home/laser/miniconda3/envs/timetoalign/bin/python -m pytest --runslow tests/loader/score
```
