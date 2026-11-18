# Symbolic score loader tests

These tests validate the symbolic score loaders against exact source-derived values. TSV data
loaded by `Ms3Loader` is authoritative whenever loaders are compared.

## Test files

| File | Remaining coverage |
|---|---|
| `conftest.py` | Seven specimen definitions, source lookup, size guards, and `.flow.csv` parsing. |
| `test_score_parsing_matrix.py` | CSV modes, folded and unfolded counts, loader/format combinations, exact flow-control profiles, target-flow reproduction, cross-loader counts, and pinned parser differences. |
| `test_loaders.py` | MS3 coordinate and duration semantics, pitch schema, Partitura measure/MIDI regressions, shared `mc_onset`, anacrusis offsets, and parquet round trips. |
| `test_cross_validation.py` | Exact Chopin note-count and MIDI-pitch agreement across MS3, Partitura, and Music21. |
| `test_measuremap_loader.py` | MeasureMap expansion, validation, traversal, schema, and comparison with MS3. |
| `test_flow_csv_validation.py` | Right-open ranges, exact atomic boundaries, live Partitura boundaries, and CSV serialization. |
| `test_partitura_exactness.py` | Exact Partitura coordinate pairs, instant nulls, flattened timelines, and agreement with Partitura's float map. |
| `test_verovio_loader.py` | `VerovioLoader` on MEI fixtures and specimens: exact onsets, kept ids, measures as structure, ties, grace notes, repeats, movements, the physical timeline, music21 parity, Verovio's expansion against the flow controller (see below). |

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

Every `.flow.csv` must exist and contain `atomic` and `default` modes. Its number of atomic rows
must equal the specimen value above. MS3 must reproduce one documented valid default flow and
the exact atomic boundaries. TSV and MeasureMap must have identical folded counts.

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
- `ScoreStore` and loader parquet tests pin files, note/measure counts, metadata, units, empty
  facets, missing paths, and event-property round trips.

The 498-note Chopin gold value is asserted once per loader in
`TestNoteCountConsistency.test_note_counts_exact_match`. The same module requires exact MIDI
pitch equality and checks both `EnharmonicPitch` and `MidiPitch` views for all 498 events.

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
same flow-control facts. The normal timeline also has a skeleton whose section
hierarchy holds that identical map object; flattening opts out only of section
children and skeleton attachment. The normal timeline is serialized through JSON
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

**`basic.mei`**: two measures of 9/8 (`m1`, `m2`), two staves, one `<mdiv xml:id="mdiv1">`.

| Staff 1 | Staff 2 |
|---|---|
| m1: rest `r1` (8th) · chord `c1` (quarter, `@staff="2"`: `c1a` C4, `c1b` A3 with `accid.ges="f"`) · duplet `tu1` (`num=2 numbase=3`: `d1` E5, `d2` F5, 8ths) · `t1` G4 dotted quarter, tied to `t2` | m1: `b1` C3 dotted half · `b2` G2 dotted quarter |
| m2: `t2` G4 dotted quarter · `t3` A4 dotted half, tied to `#outside` (no such element) | m2: rest `r2` dotted quarter · one F♯3 dotted half **without `xml:id`** |

A `<tempo xml:id="tempo1" tstamp="1">Andante très expressif</tempo>` sits in m1.

- **Exact onsets.** An 8th is 1/2 quarter; a duplet 8th is 1/2 × 3/2 = 3/4. Hence `r1` 0–1/2,
  `c1a`/`c1b` 1/2–3/2, `d1` 3/2–9/4, `d2` 9/4–3, `t1` 3–9/2, `b1` 0–3, `b2` 3–9/2, `t2` 9/2–6,
  `t3` 6–9, `r2` 9/2–6, the generated note 6–9. The stored cells of `d2` hold numerator 9 and
  denominator 4; the timeline's number type is `Fraction`.
- **Ids.** Note ids are the fixture's `<note>` `xml:id`s (read with the standard library from the
  fixture) plus one generated id; rests are `r1`, `r2`; the tempo indication is `tempo1` at 0 with
  its text verbatim. `notated_id` equals `id` on every event (nothing is expanded). The generated
  id is the same in two loads, and with 6.4.1 it is `s3clx4n` (seeded by `xmlIdChecksum`).
- **Structure.** The root is one `ContinuousLogicalTimeline` `score:clt_1` in quarters, length
  9. Its one child is `score:clt_1_mdiv1` at offset 0, length 9, `meta == {"xml_id": "mdiv1"}`,
  holding all 13 events (10 notes, 2 rests, 1 tempo indication); the root holds none. The
  `MeasureMap` has `m1`, `m2` named `1`, `2`, at 0 and 9/2, each 9/8 with actual length 9/2;
  neither measure id is an event. No flow control, no FlowMap.
- **Ties.** `t1` is `i`, `t2` `t`, `t3` `i`; every other note has none. The dangling tie `tie2`
  is listed under `dangling_ties`, logged once as a warning, and nothing is invented: `t3` ends at
  9, where the timemap ends it.
- **Staff and pitch.** The chord notes have staff 2 and layer 1 (the chord's `@staff`); `r1`
  has staff 1. `c1b` is `A♭3`, specific pitch `(A, -1, 3)`, MIDI pitch 56; the generated note is
  `F♯3`, MIDI pitch 54. For every note the MIDI pitch Verovio reports equals
  `12 × (octave + 1) + step + alter` of the stored spelling (two independent readings agree).
- **Physical timeline.** `create_physical_timeline()` is a `ContinuousPhysicalTimeline`
  `rendered:cpt_1` in seconds, number type `Fraction`, length 9/2, `meta["provenance"] ==
  "verovio_rendered"`, one child `rendered:cpt_1_mdiv1` of length 9/2. Every event lies at half
  its quarter position (`d2` 9/8–3/2, `t3` 3–9/2, `tempo1` 0). The logical timeline's C-Map maps
  9/4 quarters to 9/8 seconds, which a map answers as the float `1.125` because the seconds axis
  writes floats (the physical timeline itself is built from the exact values); the physical
  timeline's C-Map maps 9/8 back to exactly `Fraction(9, 4)`. `attach_cmap=False` attaches none.
- **Two-phase contract and options.** `from_file` and `load` give the same events;
  `create_timeline(uid="debussy", name="CdL")` keeps both, and the child becomes
  `debussy_mdiv1`. The store metadata records the version and the options the toolkit ran with,
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
and neither ending. The printed measures `m4`, `m5` then have no stated length, and the load
raises naming them and pointing to `expandNever`. With `expandNever` it loads, and the flow
controller's default flow is MC 1 2 1 2 3 4 3 5. `expandAlways` raises (it writes the expansion
into the MEI tree, which then no longer states the printed score).

**`two_movements.mei`**: `mdiv` `mov1` (`n=1`, `label=I`, 2/4: `n1` C4 and `n2` D4 quarters,
`n3` E4 half) and `mov2` (`n=2`, `label=II`, 3/4: `n4` F4 dotted half, `<tempo midi.bpm="60">`
`t2`). Verovio continues the second movement where the first ends: one timemap entry at 4 holds
the end of `n3`, the measure start of `m3` and the onset of `n4`; seconds continue (2 s).
Expected: root length 7; children `score:clt_1_mdiv1` [0, 4) and `score:clt_1_mdiv2` [4, 7),
named `I`, `II`, with `meta` `{"xml_id", "n", "label"}` verbatim. `n4` is at 0 in its child and
at 4 on the root; `t2` belongs to the second child. The root's `MeasureMap` (measures at 0, 2, 4)
and C-Map serve the children through the root: the root's timestamp table gives `n4` 2 seconds;
a child carries neither. On the physical timeline the second child starts at 2 s and lasts 3 s
(three quarters at 60 per minute). With `options={"mdivXPathQuery": "./mdiv[2]"}` only the second
movement is loaded: one child, still `score:clt_1_mdiv2` (``k`` counts every `<mdiv>` of the
file), at 0 with length 3.

**`fast.mei`**: three 256th notes at `midi.bpm="2000"`, 1/64 quarter apart, then rests to the end
of the 2/4 measure; Verovio rounds the first four entries to the milliseconds 0, 0, 1, 1.
`create_timeline()` works (length 2; the C-Map only needs increasing quarters); `create_physical_timeline()` raises, naming the coinciding entries, because their
order in seconds is not stated.

**Duplicate ids.** Two `<dynam xml:id="dd">` in `repeat.mei` load (listed under
`duplicate_ids`); two notes sharing `xml:id="m1n"` raise, naming `m1n`.

### Provisioned specimens

- **Nested movements** (`parangonar/Beethoven_Eroica_op35-cpjku/score.mei`): `<mdiv
  xml:id="Eroica-Variationen">` holds `<mdiv xml:id="Var_14" n="18" label="Var. XIV.">`. The
  root (length 129/2) has one child `score:clt_1_mdiv1` (meta `{"xml_id":
  "Eroica-Variationen"}`), which has one child `score:clt_1_mdiv2` (meta with `n` `18` and label
  `Var. XIV.`); both span [0, 129/2). The 297 notes, 36 rests and 2 tempo indications live in the
  inner child. The `MeasureMap` holds 33 measures. No provisioned file has two sibling movements;
  `two_movements.mei` covers that case.
- **Repeat facts** (`score/flow_control/out_of_the_flow_experience-flow_only.mei`, read with
  `expandNever`): the measure rows carry the same facts as the hand-checked
  `flow_only/out_of_the_flow_experience-flow_only-canonical.measures.tsv` (15 measures;
  `start_repeat` at MCs 4, 9, 10; `end_repeat` at 3, 5, 8, 10, 11, 14; voltas (5, 1), (6, 2),
  (7, 3), (14, 1), (15, 2)). Known issue: Verovio 6.4.1 crashes the process in
  `renderToTimemap` when it expands this file's repeats, so no test loads it expanded.
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
  rendered measure's start, folded through FlowMap `source`, is a printed measure start. Equal on
  `couperin_concerts/c05n05_musete.mei` (58 printed, 116 rendered) and
  `couperin_concerts/c11n08_Rondeau.mei` (60, 120). Each divergence is its own strict `xfail`:
  - `beethoven_woo71/WoO71.mei`: **Verovio** leaves out both endings numbered bar 19 (MCs 260,
    261) and does not repeat their passage; the load raises (`ValueError`).
  - `beethoven_op18-4iv_multimodal/ABC/n04op18-4_04.mei`: **Verovio** leaves out the endings at
    MCs 44–45, 93–94 and 102–103; the load raises.
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
- **Id check overhead.** Measured on the largest provisioned MEI (`wagner_walkure/…/
  Wagner_WWV086B-3.mei`, 8.8 MB, 76 623 ids, Verovio 6.4.1): `loadFile` 3.9 s and the timemap
  3.1 s; `getMEI` 0.09 s and its parse 0.21 s, which the loader needs anyway for membership; the
  uniqueness count itself 0.02 s. In that file 15 ids name more than one element (its MuseScore
  export repeats ids of `<dynam>`, `<dir>` and `<fermata>` elements); they are listed, not refused.

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
