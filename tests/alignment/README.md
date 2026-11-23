# Alignment Module Tests - Validation Strategy

This document explains **why** the test suite provides evidence that the alignment code is correct, following the TimeToAlign! Zero Tolerance Validation Policy.

## Test Philosophy

The alignment module implements TimeToAlign's multi-level hierarchy:

```
AlignmentAnchor (atomic) -> MatchClaim (low) -> MatchGraph (mid) -> MatchLine (high)
                                 |
                                 v
                  TimelineGroup (members share one start and one end)
```

**NOTE:** A TimelineGroup is an array of timelines sharing one start and one
end; a part of a timeline takes part as a child of it. Its tests live in
`tests/timelines/test_groups.py` and `tests/timelines/test_group_definition.py`;
alignment tests cover consumers of groups.

Each test validates a **specific claim** from the TimeToAlign model. Tests are not exploratory--they verify exact behaviors required by the model.

``WarpMap`` follows the same value-facing vocabulary as the conversion-map
family: calling the map or ``convert_array`` converts values, while
``inverse()`` returns a cached map with swapped source and target identities.
The tests therefore validate inverse values through the returned map and also
assert reciprocal cache identity, rather than treating ``inverse`` as a
value-conversion method.

---

## Shared Test Fixtures (`conftest.py`)

All Thoresen test data constants and shared fixtures are centralised in `tests/alignment/conftest.py`. Individual test files import from conftest rather than defining their own copies.

### Centralised Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `DGT1_SEGMENT_LENGTH` | 967 | Pixels per DGT1 segment (x1 - x0 = 969 - 2) |
| `DGT1_TOTAL_WIDTH` | 4835 | Total DGT1 width (5 × 967) |
| `DGT2_SEGMENT_LENGTHS` | [866, 867, 867, 864, 864] | Per-segment widths for DGT2 |
| `DGT2_TOTAL_WIDTH` | 4328 | Total DGT2 width |
| `AUDIO_DURATION_SECONDS` | 150.0 | Shared audio reference duration |
| `THORESEN_TEST_EVENTS` | 11 events | All events from `thoresen_test.tsv` |

### Centralised Fixtures

| Fixture | Type | Description |
|---------|------|-------------|
| `dgt1_timeline` | `DiscreteGraphicalTimeline` | DGT1 (2009): 4835 px, uid="dgt_1" |
| `dgt2_timeline` | `DiscreteGraphicalTimeline` | DGT2 (2010): 4328 px, uid="dgt_2" |
| `audio_timeline` | `ContinuousPhysicalTimeline` | 150 seconds, uid="audio" |
| `thoresen_segment_claims` | `list[MatchClaim]` | 5 interval claims mapping DGT1↔DGT2 segments; each names the segment events it matches (`system_N` on DGT1, `page_N` on DGT2, the names the graphical fixtures give the segments), its coordinates are integer pixels on both axes, its provenance `Provenance(method="manual", note="thoresen_analysis segment_correspondence")` |
| `dgt1_bundle` / `dgt2_bundle` | `GraphicalBundle` | Full graphical bundles (requires PyMuPDF) |

`test_thoresen_poc.py::TestThoresenSegmentClaims` pins the claim fixture itself: five interval claims, each connecting `dgt_1` and `dgt_2`, contiguous on both sides from 0 to the total width (4835, 4328), with exactly that provenance and the event pairs `(system_1, page_1)` … `(system_5, page_5)` in order.

### No ID Reset

Every id the alignment module mints is minted statelessly (`group_N`, `bundle_N`, `claim_N`, N above the highest number its holder holds), so no test resets a generator and no test depends on what an earlier test built.

---

## TimelineGroup Architecture

### What a group is

Every member of a group spans the whole group: its 0 is every other member's
0 and its length every other member's length, and a position converts
linearly, `p · len_B / len_A`, exactly. A part of a timeline is a child of it:

```
dgt1_image (0 - 299,400 px)
  └── dgt1_holes, child at offset 15,343 (0 - 277,776 px)
        ═══ group ═══ dlt1_raw (0 - 871,800 ticks)
```

The holes child and the MIDI timeline form the group; the image reaches both
through the child's exact offset.

### Coordinate and Filter Canonicalisation

Alignment coordinate-entry APIs preserve ``Coordinate`` and ``IdCoordinate``
units until the receiving timeline resolves them. Native-unit inputs retain
their value, C-Map-afforded units convert to the timeline's native unit, and
unafforded units raise ``ValueError`` rather than being silently reduced to a
number. An ``IdCoordinate`` must agree with an explicitly supplied timeline
ID. Graph, match-line, and stamp timeline filters use the common
``timeline_ids`` and ``id_pattern`` vocabulary; both constraints must pass.
The tests cover failed and C-Map-backed bundle queries, ID conflicts, and
equivalent graph, group-extension, match-line, and stamp filtering results.

### Key Changes from PerfectAlignment

| Before (deprecated) | After |
|---------------------|-------------------|
| `PerfectAlignment(source_start=0, source_end=277776, ref_start=15343, ref_end=293119)` | `image.add_child(holes, offset=15343)`, then `TimelineGroup(timelines=[holes, midi])` |
| Per-timeline alignment objects | Members sharing one start and one end |
| `group.reference_timeline_id` | No reference: every member spans the group |

---

## Transitive Cross-Group Union & Support Policy (`test_transitive_support.py`)

### What We're Validating

`AlignmentBundle.get_matchstamp_at` assembles the **transitive cross-group
union** reachable from the query, and governs out-of-support transfers with a
`support_policy` (`omit` / `clamp` / `extrapolate`, default `omit`). Two
properties are proven with a synthetic fixture that owns no corpus data:

1. A query reaches **every** timeline of **both** merged bundles, including
   timelines that are only reachable through a bridge timeline. A bridging
   position stated in a derived unit (seconds on a samples timeline) is
   placed on the bridge's own axis through its C-Map when the claim is made,
   so the WarpMap behind the warp is built from coordinates already on the
   axis; the bundle converts nothing when it reads.
2. A coordinate below the first alignment anchor produces **no negative
   coordinate under any policy**, and the three policies produce exact,
   distinct results.

Merging keeps every claim id, and a MatchClaim is addressed by its id
alone: the two source bundles hold different ids (bundle A states its
claims as `star_1` … `star_6`, bundle B mints `claim_1` … `claim_8`), since
a merge refuses two sources holding one id, and the claim the merged bundle
adds afterwards is minted above the highest it holds. The test pins the ids
of the merged MatchClaim table, so a merge that re-mints an id fails.

### Synthetic Fixture Topology

Two bundles are built, then merged with `AlignmentBundle.from_bundles`, then
bridged with `create_match_claims`. Every claim names an event on each side.

- **Bundle A — a star of single claims.** `a1` (score, quarters, exact axis,
  len 100), `a2` (seconds, len 100), `a3` (seconds, len 100), each in its own
  group. Six claims between fixture events (`helpers.match_claim`) at `a1`
  coordinates `{0, 50, 100}`: `a2 = a1` and `a3 = a1 / 2`, given ids
  `star_1` … `star_6`. A query at `a1 = 50` reaches `a2 = 50`, `a3 = 25` by exact anchor.
- **Bundle B — a WarpMap-able columnar `MatchClaimField`.** `b_bridge`
  (samples, int axis, len 1000) carrying a `ScalarMap(0.01, samples→seconds)`
  C-Map, plus `b1` (samples, len 3000) and `b2` (samples, len 3000), each in
  its own group. The field (`MatchClaimField.from_columns`, ids `claim_1` …
  `claim_8`) holds native-samples instant anchors between named events:

  | `b_bridge` (samples) | `b1` (samples) | `b2` (samples) |
  |---|---|---|
  | 200 | 100 | 300 |
  | 400 | 900 | 600 |
  | 600 | 1700 | 900 |
  | 800 | 2500 | 1200 |

  The field's `b_bridge` hull is therefore `[200, 800]`.
- **Bridge A↔B.** `create_match_claims` pairs the position `a1 = 50` with the
  position `b_bridge = 5 seconds`. Each side becomes a Projection event
  (`projection_1` on `a1`, `projection_2` on `b_bridge`: a Projection's ID
  is unique in the bundle); the bridge C-Map places 5 seconds at
  exactly `500 samples` (`5 / 0.01`), so the claim's `b_bridge` anchor is
  `500` samples. The merged bundle mints it `claim_9`.
- **Mixed units on one timeline.** A separate bundle (`source`, samples with
  the same C-Map; `target`, seconds) gets three claims from positions
  `source = 2.0 s, 400 samples, 6.0 s` to `target = 20.0, 40.0, 60.0`. Each
  source position is placed on the samples axis when its claim is made:
  `200, 400, 600` samples, the Projections `projection_1` … `projection_3`.
  The WarpMap's source array is `[200, 400, 600]` and `transfer(300)` gives
  `30.0`; no range or magnitude inference participates.

### Exact Expected Values

**Merged MatchClaim table:** ids `star_1 … star_6`, `claim_1 … claim_8`,
`claim_9`.

**Union query — `merged.get_matchstamp_at(50, "a1:clt_1")` (default `omit`):**

`is_interpolated is False` (the query carries an exact anchor); 6 timelines,
no negatives:

| timeline | coordinate | how |
|---|---|---|
| `a1:clt_1` | 50 | query |
| `a2:cpt_1` | 50 | exact anchor |
| `a3:cpt_2` | 25 | exact anchor |
| `b_bridge:dpt_1` | 500 | exact bridge anchor, placed at 500 samples when made |
| `b1:dpt_2` | 1300 | warp(500) |
| `b2:dpt_3` | 750 | warp(500) |

`b_bridge = 500` lies in the field hull `[200, 800]`, and `np.interp(500, …)`
gives `b1 = 1300` (`900 + ½·800`), `b2 = 750` (`600 + ½·300`).

**Parity.** `b_bundle.get_matchstamp_at(500, "b_bridge:dpt_1")` yields
`b1 = 1300`, `b2 = 750` — identical to the union's B portion for the
transferred timelines.

**Out-of-support query — `merged.get_matchstamp_at(50, "b_bridge:dpt_1")`** (50
samples, below the hull):

| policy | timelines | `b1:dpt_2` | `b2:dpt_3` |
|---|---|---|---|
| `omit` | 1 (`b_bridge` only) | — | — |
| `clamp` | 3 | 100 (`warp(200)`) | 300 (`warp(200)`) |
| `extrapolate` | 3 | 0 (`warp(50) = -500`, floored) | 75 (`warp(50)`, kept) |

`b_bridge = 50` stays present and unaltered under every policy. `a1/a2/a3` are
unreachable from `b_bridge` (the single bridge pair cannot form a WarpMap, which
needs at least two anchors), which keeps the out-of-support counts exact.

---

## Ungrouped Timelines Warp (`test_ungrouped_warp.py`)

### What We're Validating

A timeline that is in no group takes part in matchstamp warping exactly as a
group of that one timeline would: it is a warp source and a warp target. Before
this was true, a query between two anchors on an ungrouped score returned only
the score, even though the bundle's own WarpMap for that pair answers the
position. A test proves the rule only if it holds in the following cases:

1. **Between anchors**, the stamp carries the value the bundle's WarpMap gives
   for the pair, on every retrieval lane: `get_matchstamp_at`,
   `get_matchstamps_at` and `get_matchstamp_table` (struct cells and
   `format="dataframe"` scalars). The value is pinned twice, to a constant
   worked out by hand and to `_get_or_build_warp_map(...).get_coordinate_at`.
   That rules out both a wrong map and a lane that skips the map.
2. **On an anchor**, the stamp still gives the anchor and is not interpolated.
3. **Outside the anchors**, under the default `omit` policy, the recording does
   not appear: the stamp holds the score alone, and the table cell is null.
   A position before the first anchor and one after the last are both checked.
4. **Both directions.** A query on the ungrouped recording reaches the ungrouped
   score, so the ungrouped timeline is a warp source as well as a target.
5. **Parity.** The same bundle answers identically, on the same stamp
   coordinates and `is_interpolated`, in three forms: nothing grouped; the score
   in a group of its own; both timelines each in a group of their own. The last
   form already warped before the fix, so it is the reference.
6. **No claims, no answer.** An ungrouped timeline with no claims to anything
   stays unresolved. In the claimed bundle it never appears, and a bundle of two
   ungrouped timelines with no claims at all answers the queried timeline only.
7. **Interval ambiguity survives the warp.** Two interval claims mapping one
   boundary of `mix` to two different positions of `loop` leave `loop` out of
   the stamp (see "Interval-Claim Position Retrieval"). The closure must not
   fill it from the WarpMap those same claims build, in the ungrouped form or in
   the grouped one. Both live in `test_interval_matchstamp.py`:
   `test_loop_boundary_ambiguity_is_per_claim_and_atomic` (ungrouped) and
   `test_loop_boundary_ambiguity_survives_grouping` (each timeline in a group of
   its own). An interior position of either claim still resolves `loop`: mix 15
   → loop 5 and mix 25 → loop 105.

### Fixture

- `score`: quarters, length 16, exact (`Fraction`) axis.
- `recording`: samples, length 441000, with `SamplesToSeconds(44100)`.
- Four instant claims made with `create_match_claims` from positions, score
  quarters → recording **seconds**: each side becomes a Projection event, and
  each seconds position is placed on the recording's sample axis through the
  C-Map when the claim is made, so the claims hold samples
  (`test_claims_hold_the_recording_samples`). Each `seconds × 44100` is whole:

  | score (quarters) | recording (seconds) | recording (samples) |
  |---|---|---|
  | 1 | 0.5 | 22050 |
  | 3 | 1.5 | 66150 |
  | 5 | 2.0 | 88200 |
  | 9 | 4.0 | 176400 |

- `other` (seconds, length 10): a third ungrouped timeline with no claims.

### Exact Expected Values

Linear interpolation between neighbouring anchors:

| score query | recording | how |
|---|---|---|
| 2 | 44100 | `22050 + (1/2)·44100` |
| 7/3 | 51450 | `22050 + (2/3)·44100` |
| 4 | 77175 | `66150 + (1/2)·22050` |
| 7 | 132300 | `88200 + (1/2)·88200` |
| 3 (anchor) | 66150 | anchor, `is_interpolated is False` |
| 1/2, 10 | absent | outside the hull `[1, 9]`, `omit` |

The reverse direction: recording `44100` → score `2`, and recording `77175` →
score `4`. Both are whole quarters, so the float WarpMap lane meets the exact
axis without rounding.

---

## Interval-Claim Position Retrieval (`test_interval_matchstamp.py`)

### What We're Validating

The runtime form of the query determines the stamp form. A coordinate query
always produces a `MatchStamp`; every relevant interval claim maps that position
affinely from the queried side to the counterpart side. An `Interval` query
always produces a `MatchIntervalStamp` by resolving its start and end as two full
coordinate stamps and combining their timeline coverage. Synthetic standalone
timelines isolate claim geometry from group interpolation and corpus data.

The validation covers these exact geometries:

- Closed containment selects an interval claim at both anchors and throughout
  its interior. Anchor queries map to the stored counterpart anchors and are not
  interpolated; interior queries map proportionally and are interpolated.
- Overlapping claims to different counterpart timelines contribute both mapped
  coordinates in first-relevance order. The DJ-set crossfade geometry is checked
  for both a coordinate query and an interval query.
- Adjacent claims mapping one shared source boundary to different values on the
  same counterpart timeline retain both per-claim candidates. The classic
  coordinate mapping omits that timeline, singular retrieval raises with both
  claims and values, and both display forms expose the alternatives.
- Unequal interval extents use proportional mapping. A fraction-canonical case
  pins an exact rational result that binary64 interpolation cannot reproduce.
- Unambiguous interval-derived coordinates participate in transitive closure in
  the same way as other reached coordinates.
- An interval query whose endpoints both reach a timeline yields its exact
  `Interval`. A timeline reached by only one endpoint retains the reached side,
  serializes the other as null, displays the missing side, and refuses interval
  retrieval with the missing side named.
- When both endpoints reach every timeline, each dictionary pair is complete and
  agrees exactly with the constituent `MatchStamp` coordinate projections;
  plural timeline-coordinate retrieval rejects conflicting per-claim
  alternatives before projecting any requested coordinate, and plural position
  retrieval rejects an unsupported element before resolving a preceding valid
  element.
- Raw values, `Coordinate`, `IdCoordinate`, `Interval`, plural position getters,
  and the dispatcher preserve query-form result selection. A mixed collection
  resolves each element independently. Raw and interval inputs without a source
  timeline fail atomically.
- A fraction-canonical source axis compares a converted query and native claim
  boundaries in the exact domain, and the columnar decimal comparison
  distinguishes exact `1/3` from a different fraction with the same binary64
  value.
- Coordinate-query tables materialize unambiguous mapped cells exactly. Interval
  queries have no table form and raise `NotImplementedError`; ambiguous queried
  cells raise instead of becoming null.
- Two instant claims forming `A -> B -> C` retain standalone transitive closure,
  direct-edge transfer fallback remains available, public exports remain stable,
  and a position outside every claim produces a source-only `MatchStamp`.

Every claim names a fixture event on each side (`helpers.match_claim`), its
coordinates on the timelines' own axes; the DJ-set claims are two interval
claims built that way.

Every assertion uses exact `Coordinate`, `Interval`, `Fraction`, dictionary, and
ordering expectations; no tolerance or approximate comparison is used.

---

## A Claim's Bundle Is Never Copied With It (`test_claim_back_reference.py`)

### What We're Validating

A bundle holds its claims as rows of one MatchClaim table
(`match_claim_table`) and materialises a `MatchClaim` when one is read
(`get_match_claims`, a stamp's `interval_claims`). A claim read from a bundle
records that bundle (`MatchClaim.bundle`), so that `get_matchstamp` can be
called without passing it. That record is where the claim is, not part of
what it claims. Copying a claim must therefore never copy the bundle: a deep
copy of one claim that dragged its bundle along would copy the whole
alignment, so retrieving `k` claims and copying them would cost `k` times the
bundle.

Wall time cannot prove this (it depends on the machine and is not an exact
value). What proves it is the number of claims copied, counted by wrapping
`MatchClaim.__deepcopy__`: it is the number of claim objects the operation
actually duplicated, independent of speed.

- **A claim read records its bundle.** Every claim `get_match_claims` returns
  has `bundle is` the bundle; their ids are the ones the bundle minted in
  order, `claim_1` … `claim_303`.
- **One claim, one copy.** Deep-copying one claim read from the bundle
  (`copy.deepcopy` and `model_copy(deep=True)` alike) copies exactly that
  claim: the copy count is 1, the copied object the claim itself.
- **A copy is held by no bundle.** The copy's `bundle` is `None`: the bundle
  is not copied with it, and a copy is not a row of the bundle's table, so
  naming that bundle would state a membership that does not exist. The
  original still records the bundle. The copy equals the original (`__eq__`
  reads the fields only) and is a different object.
- **A copied bundle copies its table, and its claims record the copy.**
  Deep-copying the whole bundle copies its table (a different
  `match_claim_table` object, `match_claim_table.table` equal row for row,
  303 rows) and
  no claim object, since the bundle holds none: 0 claim copies. Every claim
  read from the copy records the copy; every claim read from the original
  still records the original; the two lists are equal.
- **Retrieval copies no claim.** `get_matchstamp_at` at a position four
  interval claims cover makes zero claim copies; the stamp's
  `interval_claims` are those four claims, read from the table in table order
  (ids and equality to the claims `get_match_claims` returns at the same
  rows), and they record the bundle. A `MatchStamp` copies its list, not the
  claims in it: a caller appending to the list it passed leaves the stamp's
  list unchanged, and the stamp holds the very claim object it was given.

### Fixture

Two standalone float-canonical `floating_measures` timelines, `score` and
`perf`, length 1200. Claim `i` (`i = 0 … 299`, id `claim_{i + 1}`) is an
interval claim between two fixture events (`helpers.match_claim`) mapping
`score [i, i + 1]` to `perf [2i, 2i + 2]`; three further claims `score [10,
12]`, `score [10.25, 10.75]`, `score [9, 11]` (to `perf [20, 24]`, `[20.5,
21.5]`, `[18, 22]`; ids `claim_301` … `claim_303`) overlap position `10.5`,
which claim 10 (`claim_11`) also covers. The query at `score 10.5` is
therefore covered by exactly four claims; they all map it to `perf 21`.

### Exact Expected Values

| Operation | Claims copied | Back-reference |
|---|---|---|
| `get_match_claims()` | 0 | every claim's `bundle is` the bundle |
| `copy.deepcopy(claim read)` | 1 (that claim) | copy `None`, original the bundle |
| `claim.model_copy(deep=True)` | 1 (that claim) | copy `None`, original the bundle |
| `copy.deepcopy(bundle)` | 0 | every claim read from the copy records the copy |
| `get_matchstamp_at(10.5, "score")` | 0 | `interval_claims` are `claim_11`, `claim_301`, `claim_302`, `claim_303`, recording the bundle |

The stamp's `perf` coordinate is `21` floating measures, a float on the float
axis.

---

## A Matchstamp Reads Only the Claims Naming Each Timeline (`test_closure_work.py`)

### What We're Validating

A matchstamp's closure visits every timeline of the bundle's trees that holds
the position: a score with loader category children or unfolded spans, and
recordings with feature and track children, put dozens to hundreds of
timelines on that walk while claims name only a handful of them. The bundle
holds its claims as rows of one MatchClaim table, so two pieces of work must
scale with the claims naming a timeline, never with the table's length times
the number of timelines visited:

- **Relevance.** Only a claim naming a timeline can be relevant to a position
  on it. Each visited timeline cuts the table once, a vectorized cut
  (`MatchClaimField.connecting`, through `_claims_naming`) cached per timeline
  and table state, however often the timeline is asked; only the rows of that
  cut holding the position are materialised as claims (`_materialized`). A
  timeline no claim names cuts zero rows and materialises none.
- **MatchLines.** A MatchLine is built from the claims naming its source only,
  and a source no claim names has no line: nothing is materialised for it.

Wall time cannot prove either (it depends on the machine and is not an exact
value). What proves it is the amount of work, counted by wrapping
`MatchClaimField.connecting` (cuts per timeline, rows per cut),
`AlignmentBundle._materialized` (claims materialised per call),
`MatchLine.from_claims` (lines built, with their source) and
`AlignmentBundle._transfer_to_group` (warps attempted, with their source).
These counts are exact for the fixture and independent of speed; a cut
repeated per request, a cut that is not narrowed to its timeline, or a line
built from the whole table changes them. The stamp itself is pinned too, so
the counts cannot fall by reaching fewer timelines.

### Fixture

Quarters and seconds: exact (fraction) quarters, float seconds, every claim a
synchronous instant claim between two fixture events (`helpers.match_claim`).

- Group `score_group`, members in order `score`, `facsimile`, `edition`, all
  quarters of length 40. `score` has 20 children `part_1` … `part_20`, each of
  length 40 at offset 0 (category children spanning the whole score).
- Groups `perf_n_group` (`n = 1, 2, 3`), each holding one seconds timeline
  `perf_n` of length 120 with 10 children `perf_n_track_1` … `perf_n_track_10`, each of
  length 120 at offset 0.
- Group `recording_group` holding the seconds timeline `recording`, length 50.
- Claims, in this order: for each `n`, 21 instant claims `score q` ↔
  `perf_n n·q` for `q = 0, 2, …, 40` (63 claims); then 2 claims `edition 0` ↔
  `recording 5.0` and `edition 40` ↔ `recording 45.0`. 65 claims in all.

Query: `get_matchstamp_at(11, "score")`, between two anchors, so every other
timeline is reached by a warp, the group or the tree.

### Derivation of the counts

- **Cuts.** The closure visits 57 timelines (the stamp's). `score` is asked
  for its relevant claims three times (the interval-claim scan, the
  exact-anchor MatchGraph, the closure) and cut once; every other visited
  timeline is asked and cut once. 57 cuts, one per timeline. The rows cut are
  the claims naming each: `score` 63, `perf_1`–`perf_3` 21 each, `edition` 2,
  `recording` 2, the 51 others (`facsimile`, the parts, the tracks) 0:
  63 + 63 + 2 + 2 = **130** rows.
- **Relevance materialises nothing.** No visited timeline holds an anchor at
  its position: `score` 11 and `edition` 11 (anchors at even quarters, at 0
  and 40), `perf_1` 11.0, `perf_2` 22.0, `perf_3` 33.0 (anchors at `n·q`, `q`
  even: 22 would need `q = 11`, 11 and 33 are odd multiples), `recording`
  16.0 (anchors 5.0, 45.0). Each of the 59 relevance reads (3 on `score`, 56
  more timelines) materialises 0 claims.
- **MatchLines.** The closure pops `score` first. Its group adds `facsimile`
  and `edition`; then it warps into the four other groups, 4 calls of
  `_transfer_to_group`, all from `score`: `perf_1`–`perf_3` directly through
  `score`'s MatchLine (built from the 63 claims naming `score`), `recording`
  indirectly — `score`'s line does not reach it, `facsimile` names no claim
  (no line, nothing materialised), `edition`'s line (built from its 2 claims)
  does, at `11 + 5.0 = 16.0`. Every group is then reached, so no later node
  warps. MatchLines: `["score", "edition"]`.
- **Materialised, in call order:** the 3 relevance reads on `score` (0 each),
  `score`'s line (63), `edition`'s line (2), the 56 other relevance reads (0
  each): `[0, 0, 0, 63, 2] + [0] * 56`, 65 claims in all — the claims of the
  two lines, never the table times the timelines visited (57 × 65 = 3705).

### Exact Expected Values

| Measure | Value |
|---|---|
| table cuts | 57, one per visited timeline |
| rows cut | 130 (`score` 63, `perf_1`–`perf_3` 21 each, `edition` 2, `recording` 2) |
| claims materialised, per call | `[0, 0, 0, 63, 2] + [0] * 56` |
| MatchLines built, by source | `["score", "edition"]` |
| `_transfer_to_group` calls, by source | `["score"] * 4` |
| timelines in the stamp | 57 (3 score group, 20 parts, 3 performances, 30 tracks, `recording`) |
| `perf_1`, `perf_2`, `perf_3` | `11.0`, `22.0`, `33.0` seconds |
| `edition`, `part_7` | `11` quarters each |
| `recording` | `16.0` seconds |

---

## Every Timeline of a Bundle's Trees (`test_nested_addressing.py`)

### What We're Validating

A bundle answers for every timeline of the trees it holds, whatever its depth
and whether or not it was added, registered or grouped. Registering is how a
timeline enters a bundle (its root comes along); it is not a condition for
being found. Each timeline of a tree reaches every other one of that tree by the
exact parent-child offsets, the first transfer mechanism, and a timeline appears
in a stamp exactly when it holds the position by the timeline's own
parent-to-child rule: `[0, length)`, plus its end where no sibling starts there,
applied along the path — a position carried down reaches a timeline only if
every child on the way holds it, as in the root's own stamps and tables
(`../timelines/README.md`, `test_offset_arithmetic.py`). The walk up starts
only from a position on the queried timeline, closed `[0, length]`: its own end
is its own whatever its siblings do, since the position was asked of it. An
ancestor contains that span, end included, so no span of its own is checked. A
position outside the queried timeline lies on no ancestor and reaches no
relative through the tree; the stamp then holds what the other mechanisms give
(the queried timeline alone where nothing else applies), exactly as a
timeline's own stamp and a group's answer outside their span, without raising.

Before this held, a timeline between a registered root and a grouped
descendant (a "middle ancestor") was invisible: querying it raised `KeyError`,
and stamps queried from the root, the descendant or the descendant's group
partner left it out while including the timelines on either side of it. The
same was true of every unregistered sibling. A test proves the rule only if it
holds in the following cases:

1. **Lookup.** `get_timeline` returns the nested timelines themselves (identity),
   at depth one and two, while the registry is unchanged: `timeline_ids` still
   lists only what was registered. Addressing must not enrol anything, or the
   dictionary form would change.
2. **The middle ancestor as query.** A query on it returns every timeline
   reaching the position: its child, the root, the child's group partner and the
   partner's claimed recording, each at its exact value, in a fixed order.
3. **The middle ancestor in others' stamps.** Queries at the same instant from the
   root, the grouped descendant and its group partner all return the same
   coordinate set, the middle ancestor included. Four sources agreeing rules out
   a rule that depends on where the query starts.
4. **Only the timelines holding the position appear.** At the boundary between
   two unregistered siblings only the one starting there appears (`clt_4` at 0,
   not `clt_3` at 312): one position on the root names one movement, as it does
   in the root's own stamps. The end of a child nothing follows stays on it:
   `clt_7` ends at 27 inside `clt_4`, which runs on, so `clt_4` 27 still reaches
   `clt_7` 27 and, through its group and the claim there, `cpt_1` 9.0. A
   position covered by one child only shows that child and the root. An
   unregistered branch with no group and no claim is reached by offsets alone.
5. **The walk up starts on the queried timeline.** A small tree pins it (fixture
   below): a position past the queried child's end and one before its start
   reach no ancestor, so neither the parent nor the sibling the parent would
   carry the position into appears; the queried child's own end does reach its
   parent, and from there only the sibling starting at it (and that sibling's
   child), while the queried child's own child keeps its lone end. Before the
   rule, `A` 6 came back with `P` 6, `B` 2 and `B1` 2, and `A` -1 with `P` -1:
   positions the queried timeline does not have, carried onto timelines that
   do.
6. **Every lane follows.** The plural getter, the key getter (`_for`), the
   position table and the coordinate getter all give the same values as the
   singular stamp; they share one assembly path, and the test pins each of them.
7. **Filters select among every known timeline.** `within`, `id_pattern`
   and `include_units` keep exactly the nested timelines they name, beside
   the queried `clt_1`, which a stamp always keeps: `within={"clt_4",
   "cpt_1"}` → `clt_1`, `cpt_1`, `clt_4`; `id_pattern=r"^clt_[47]$"` →
   `clt_1`, `clt_7`, `clt_4`; `include_units={seconds}` → `clt_1`, `cpt_1`.
8. **An unknown id** still raises `KeyError`, naming every timeline the bundle
   knows: the registered ones in registration order, then each root's other
   descendants depth-first.

### Fixture

- `clt_1`: root, quarters, length 1616, children at exact offsets:

  | child | offset | length | span on `clt_1` |
  |---|---|---|---|
  | `clt_2` | 0 | 356 | `[0, 356]` |
  | `clt_3` | 356 | 312 | `[356, 668]` |
  | `clt_4` | 668 | 324 | `[668, 992]` |
  | `clt_5` | 992 | 624 | `[992, 1616]` |

- `clt_7`: child of `clt_4` at offset 0, length 27, so `[668, 695]` on `clt_1`.
- `clt_6`: quarters, length 27, grouped with `clt_7` (one group, `score`).
- `cpt_1`: seconds, length 10, ungrouped, with two instant claims to `clt_6`
  between fixture events (`helpers.match_claim`): `0 ↔ 0.0` and `27 ↔ 9.0`.
- Registered: `clt_7` (bringing its root `clt_1`), `clt_6`, `cpt_1`.
  `clt_2`..`clt_5` are neither registered nor grouped.

### Exact Expected Values

| query | expected coordinates (in this order) |
|---|---|
| `clt_4` 27/2 | `clt_4` 27/2, `clt_7` 27/2, `clt_1` 1363/2, `clt_6` 27/2, `cpt_1` 4.5 |
| `clt_1` 1363/2, `clt_7` 27/2, `clt_6` 27/2 | the same five values |
| `clt_4` 27 | `clt_4` 27, `clt_7` 27, `clt_1` 695, `clt_6` 27, `cpt_1` 9.0 |
| `clt_1` 668 | `clt_1` 668, `clt_7` 0, `clt_6` 0, `cpt_1` 0.0, `clt_4` 0 |
| `clt_3` 312 | `clt_3` 312, `clt_7` 0, `clt_1` 668, `clt_6` 0, `cpt_1` 0.0, `clt_4` 0 |
| `clt_4` 0 | `clt_4` 0, `clt_7` 0, `clt_1` 668, `clt_6` 0, `cpt_1` 0.0 |
| `clt_1` 1616 | `clt_1` 1616, `clt_5` 624 |
| `clt_1` 100 | `clt_1` 100, `clt_2` 100 |
| `clt_5` 300 | `clt_5` 300, `clt_1` 1292 |

`1363/2 = 668 + 27/2`; `cpt_1` 4.5 is the WarpMap between the two claims,
`(27/2) / 27 · 9`, and 0.0 and 9.0 are the claims' own anchors. `clt_3` 312
is the root's 668, which `clt_4` holds and `clt_3` does not; the root's end 1616
is the end of `clt_5`, the last movement, which nothing follows. The order is the
query timeline, then the registered timelines in registration order, then the
other descendants depth-first; a filter keeps that order. Quarters values are
`Fraction`s on every lane, seconds values floats.

The key getter is pinned with an event stored on `clt_4` at 27/2: the root
finds it among its children, so the root owns it, and the stamp is the root's
query at 1363/2, `clt_4` included. The table at `clt_4` 27/2 has the columns
`clt_1`, `clt_4`, `clt_6`, `clt_7`, `cpt_1`, the `clt_4` column naming its
timeline in the field metadata.

### The Walk Up Starts on the Queried Timeline

- `P`: root, quarters, length 8, registered alone; children `A` at 0 and `B` at
  4, each of length 4; `A1` in `A` and `B1` in `B`, each at 0 with length 4.

| query | expected coordinates (in this order) |
|---|---|
| `A` 6 | `A` 6 |
| `A` -1 | `A` -1 |
| `A` 4 | `A` 4, `P` 4, `A1` 4, `B` 0, `B1` 0 |
| `A` 0 | `A` 0, `P` 0, `A1` 0 |

`A` 6 would be `P` 6, which `B` holds at 2; it is not on `A`, so the tree is not
entered. `A` 4 is `A`'s own end: `P` 4 is the boundary `B` starts at, so `B` and
`B1` take it, and `A1`, which ends with `A` and is followed by nothing inside
it, holds its end. The queried timeline's coordinate is kept as asked in every
row. Values are `Fraction`s.

### Claims on an Unregistered Nested Timeline

A claim names the timeline it was made on. When that timeline is a child no one
registered, the bundle must still make the claim, and warp through it, exactly
as it would for a registered timeline. The root's offsets carry a position onto
the child, but they cannot warp it on: the claims build the child's `MatchLine`,
never the root's. So the child must be a warp unit of its own, like every
ungrouped registered timeline. What proves it:

8. **Claims are made on it.** `create_match_claims` resolves event ids on the
   unregistered child, where a registry lookup would raise `KeyError`; an id
   naming no timeline of the trees still raises `KeyError`, naming every known
   timeline.
9. **The warp runs through it between the anchors.** A position between the two
   claims, queried on the child, on its root and on the recording, gives the same
   three coordinates each time. Before, only the two anchors themselves reached the
   recording (an exact-anchor edge needs no warp); every position between them
   stopped at the tree. Three sources agreeing shows the warp runs in both
   directions.
10. **Only through the timeline that holds the claims.** A root position the
    child does not cover reaches nothing else: the root holds no claim, so no warp
    runs from it.
11. **The private coordinate transfer agrees**, in both directions, and is no
    longer refused for an unregistered source or target.
12. **Only a claimed descendant is a unit of its own.** A descendant no claim
    names warps nothing, so it is no unit: a registered root (quarters, length
    100) with 100 unregistered children of length 1, grouped alone, beside a
    grouped recording (seconds, length 100.0) with claims `0 ↔ 0.0` and
    `100 ↔ 100.0` on the root (made by `create_match_claims` from positions,
    a Projection on each side), has exactly 2 units, and the stamp at the
    root's 50.5 holds the root 50.5, its child at 50 holding 1/2, and the
    recording 50.5 after exactly 1 transfer (102 units and 298 transfers
    before). In the fixture above, the units are the registered `clt_1` and
    `cpt_1` and the claimed `clt_2`, and the queries of 9–10 give the values
    below.

Fixture (separate from the one above): `clt_1` root, quarters, length 100,
registered; `clt_2` its child at offset 20, length 40, not registered, events
`n1` at 0 and `n2` at 40; `cpt_1` seconds, length 30, registered, events `a1` at
5.0 and `a2` at 25.0. Claims made by `create_match_claims`: `n1 ↔ a1`,
`n2 ↔ a2`.

| query | expected coordinates (in this order) |
|---|---|
| `clt_2` 20 | `clt_2` 20, `clt_1` 40, `cpt_1` 15.0 |
| `clt_1` 40 | `clt_1` 40, `cpt_1` 15.0, `clt_2` 20 |
| `cpt_1` 15.0 | `cpt_1` 15.0, `clt_1` 40, `clt_2` 20 |
| `clt_1` 10 | `clt_1` 10 |

`15.0 = 5.0 + (20 / 40) · (25.0 − 5.0)`; `40 = 20 + 20`. The claims' anchors are
`clt_2` 0 ↔ `cpt_1` 5.0 and `clt_2` 40 ↔ `cpt_1` 25.0, both synchronous, between
the events `n1 ↔ a1` and `n2 ↔ a2` (read back with `get_match_claims`).
`transfer(20, "clt_2", "cpt_1")` is `15.0` and `transfer(15.0, "cpt_1",
"clt_2")` is `20.0`.

### One Step per Edge

A position is carried from the queried timeline along the tree's edges, one
offset addition (child to parent) or subtraction (parent to child) per edge,
exactly as a timeline converts between itself and its children. Going through the
root instead adds the queried timeline's origin and subtracts the other's: on a
float axis that is two roundings, so a child at offset 0 got a value differing
from its parent's in the last digit, and two lanes of one stamp named different
positions. What proves the walk:

12. **A zero offset is the identity on a float axis.** In a drawing with a nested
    child at offset 0, a position on the child and the same position on its parent
    are the same float, whichever of the two is queried; and from the root both
    get one value. The values are a real specimen's (a page drawing in CSS
    pixels), where the sum through the root does not round back: `101.68572501334543
    + 76.70371459980326 = 178.38943961314868`, and `178.38943961314868 −
    76.70371459980326 = 101.68572501334542`.
13. **An exact tree stays exact** across a path that goes up and down again
    (grandchild → its parent → the root → an uncle), with `Fraction` results.

Float fixture: `cgt_1` (css_pixels, float, length 932.2326510101616), its child
`cgt_2` at 76.70371459980326 (length 855.5289364103584), and `cgt_2`'s child
`cgt_3` at 0 (length 713.3617548979767); `cgt_1` registered.

| query | expected coordinates (in this order) |
|---|---|
| `cgt_3` 101.68572501334543 | `cgt_3` 101.68572501334543, `cgt_1` 178.38943961314868, `cgt_2` 101.68572501334543 |
| `cgt_2` 101.68572501334543 | `cgt_2` 101.68572501334543, `cgt_1` 178.38943961314868, `cgt_3` 101.68572501334543 |
| `cgt_1` 178.38943961314868 | `cgt_1` 178.38943961314868, `cgt_2` 101.68572501334542, `cgt_3` 101.68572501334542 |

From the root the value on `cgt_2` is the one subtraction the edge prescribes,
and `cgt_3` repeats it exactly.

Exact fixture: `clt_1` (quarters, length 10) with children `clt_2` at 1/3
(length 4) and `clt_3` at 10/3 (length 2); `clt_3`'s child `clt_4` at 1/2
(length 1); `clt_1` registered. Querying `clt_4` at 1/4 gives `clt_4` 1/4,
`clt_1` 49/12, `clt_2` 15/4, `clt_3` 3/4 (`3/4 = 1/4 + 1/2`, `49/12 = 3/4 +
10/3`, `15/4 = 49/12 − 1/3`), every value a `Fraction`.

### Grouping a Nested Timeline

A group may hold any timeline of the bundle's trees, so every way of founding
or joining a group must find a nested one through the same lookup as the
getters, and must register it as a member is registered (a group member is
always in `timeline_ids`). Before, `add_timeline(t, grouped_with=X)` looked `X`
up in the registry alone and raised `KeyError` for a child no one had
registered. What proves it:

14. **`grouped_with` names a nested timeline.** In the main fixture's tree with
    only the root `clt_1` registered, `add_timeline(clt_6, grouped_with="clt_7")`
    founds one group of `clt_7` and `clt_6` (in that order), registers both
    (`timeline_ids` is `clt_1`, `clt_7`, `clt_6`), and a stamp at `clt_6` 27/2
    reaches `clt_7` 27/2, `clt_4` 27/2 and `clt_1` 1363/2.
15. **`add_group` with a nested member** gives the same membership and the same
    stamp, whether the bundle held the tree before (`clt_1`, `clt_7`, `clt_6`)
    or the group brings it (`clt_7`, `clt_1`, `clt_6`: a member is registered
    with its root right after it).
16. **A name of no timeline** still raises `KeyError`, now naming every timeline
    the bundle knows, nested ones included:
    `Timeline to group with 'clt_9' not in bundle. Available timelines: ['clt_1',
    'clt_2', 'clt_3', 'clt_4', 'clt_7', 'clt_5']`.

---

## TimelineGroup Integration

Group behaviour — the definition, exact linear conversion, child members,
stored rows, identity and the dictionary form — is validated in
`tests/timelines/` (see "Timeline Groups" in `tests/timelines/README.md`).
Alignment tests exercise groups only as parts of bundles.

---

## Group TimeStamp Tests (`../timelines/test_groups.py::TestTimelineGroupTimestampAt`)

### What We're Validating

`Timeline` (with children) and `TimelineGroup` (with members) answer one
`TimeStamp` class through the `TimeStampSource` protocol. A timeline reaches
its descendants by exact offset arithmetic; the members of a group share one
start and one end, so a position on member A is `p · len_B / len_A` on member
B, computed on exact ratios and written in B's declared number type. There is
no group-specific stamp class and no interpolation map inside a group.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_get_timestamp_at_basic` | The stamp's axis is the queried coordinate on the queried member |
| `test_get_timestamp_at_coordinate_conversion` | A member's coordinate is the linear image of the axis |
| `test_get_timestamp_at_bidirectional` | The conversion answers in both directions |
| `test_get_timestamp_at_unknown_timeline_raises` | `KeyError` for an unknown timeline id |
| `test_get_timestamp_at_empty_group_raises` | `KeyError` for a group without members |
| `test_interval_stamp_from_get_timestamp_at` | A `TimeIntervalStamp` composes from two stamps |
| `test_timestamp_with_three_timelines` | Every member of a three-member group is answered |
| `test_timestamp_same_timeline_returns_axis` | The queried member's own coordinate is the axis |
| `test_implements_timestamp_source_protocol` | `TimelineGroup` implements `TimeStampSource` |
| `test_get_related_timeline_ids` | `_get_related_timeline_ids()` returns the member ids |
| `test_get_available_units_returns_empty` | A group carries no C-Maps of its own |

---

## AlignmentAnchor Tests (`test_anchors.py::TestAlignmentAnchor`)

### What We're Validating

**Design:** AlignmentAnchor is a **pure coordinate pair** — a neutral record associating one coordinate on timeline A with one coordinate on timeline B. It contains no claim semantics (no `is_synchronous`, no `id`). Claim semantics live exclusively on `MatchClaim`. In memory an anchor names its two timelines; in the MatchClaim table and in a claim's dictionary it is its two coordinates alone, its timelines being the row's (`timeline_a_id`, `timeline_b_id`), stated once.

`AlignmentAnchor` is a **frozen pydantic v2 `BaseModel`** whose coordinates are
unit-bearing `Coordinate` values, **kept as given**. An anchor knows its
timelines by id only, so it cannot know the number type a timeline declares
(a `quarters` timeline may declare `float`); rewriting a coordinate onto its
unit's default would put a `fraction` where the timeline counts in `float`,
and the claim's check against the timeline (below) would then refuse what the
caller wrote correctly. The evidence is therefore that the anchor changes
nothing: `Fraction(25, 2)` seconds stays `25/2 seconds` (a `fraction`), and
`1.5` quarters written as `float` stays `1.5 quarters` (a `float`), in the
coordinate, the repr and the dictionary.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_basic_creation` | Anchor stores two timeline IDs and two required `Coordinate` scalars (no raw-float coercion or semantic flags) |
| `test_connects` / `test_connects_both` | Query methods correctly identify connected timelines |
| `test_get_coordinate_for` | Coordinate retrieval preserves the exact value and unit |
| `test_from_dict_roundtrip` | Nested coordinate dictionaries round-trip exactly (no legacy flat-float shape) |
| `test_coordinates_kept_as_given` | a `fraction` on seconds and a `float` on quarters keep their number types through construction, `repr`, `to_dict` and `from_dict` |
| `test_frozen_model` | Immutability enforced (frozen pydantic model) |

### Why Immutability Matters

```python
def test_frozen_model(self):
    with pytest.raises(ValidationError):
        basic_anchor.coordinate_a = 200.0
```

Anchors are value objects — identified entirely by their coordinates. Immutability prevents subtle bugs where coordinate modifications propagate unexpectedly through a MatchGraph. (A frozen pydantic model raises `pydantic.ValidationError` on attribute assignment.)

---

## MatchClaim Tests (`test_anchors.py::TestMatchClaim`)

### What We're Validating

**Design:** MatchClaim has `timeline_a_id` and `timeline_b_id` as **top-level
fields** (not derived from anchors). Anchors are `Optional` — only synchronous
claims have them. A MatchClaim **connects events**: it names an event on each
side, or one event and the NOMATCH sentinel (`None`) on the other; it never
states a bare coordinate. Two case-specific constructors:

| Constructor | Case | Synchronous | Anchors |
|-------------|------|-------------|---------|
| `from_events(event_a, timeline_a, event_b, timeline_b)` | Two events correspond | Yes (default); `is_synchronous=False` gives a conceptual claim | From the events' coordinates, on the two timelines' axes |
| `nomatch(event, timeline, other_timeline)` | An event has no counterpart | No | None, and no coordinate at all |

What a MatchGraph infers through groups is no MatchClaim (`MatchGraph`
section, `InferredEdge`). The properties `id_minted`, `is_explicit`,
`source_claim_id` and `source_coordinate` are gone, and a claim refuses any
property it does not have (`extra="forbid"`): passing one of the four is a
`ValidationError` (`Extra inputs are not permitted`), so a caller still
writing `is_explicit=False` is told, never silently given a stated claim.

**Events are required.** The validator refuses a synchronous claim naming
fewer than two events (one named, and none named, each refused) and a
non-synchronous claim naming none, so no path — direct construction,
`from_dict`, a factory — makes an event-less claim. `from_events` and
`nomatch` refuse an event dict without `id`, naming the side. The messages
are asserted exactly, because they tell the caller what to do (give the side
a Projection event).

**Coordinates are on the timelines' axes.** `check_on_timelines(timeline_a,
timeline_b)` is the one check: every coordinate of the claim (both of each
anchor) must be in its timeline's unit **and** declared number type, else
`ValueError` with the message naming the coordinate, its unit and number
type, and the timeline's. It takes a timeline or its `(unit, number_type)`
pair, refuses a timeline whose id is not the one the claim names, and
returns the claim. What proves it is a refusal for each half of the rule —
an anchor in `seconds` on a `samples` timeline (unit), and a `fraction` on a
`quarters` timeline declared `float` (number type) — beside acceptance of
the same claim against the right axes, and of a `(unit, number_type)` pair.

`from_events` and `nomatch` receive the timelines and run that check on the
claim they make. A raw number from an event is a position on the timeline:
it takes the timeline's unit and is written in its declared number type
(`8` on `quarters` is `Fraction(8, 1)`, `45.5` on `seconds` a float, `96` on
`ticks` an int; a storage cell `{value 2.5, numerator 5, denominator 2}` on
a `float` seconds timeline is `2.5`). A `Coordinate` in another unit is
converted through the timeline's C-Map: LayerIt's first onset,
`3.16952381` seconds on a 44,100 Hz samples timeline, becomes exactly
`139776` samples (the integer the axis declares, not `139776.000021`); an
`IdCoordinate` naming another timeline is refused, and so is a unit no C-Map
converts. A `Coordinate` already in the timeline's unit is **not** rewritten:
a `float` quarters coordinate on a `fraction` timeline is refused by the
check, which is what keeps one position from appearing as two coordinates.

**The rules of TimeToShare's MatchClaim table hold for the scalar.** A
MatchClaim connects two timelines: one naming the same timeline twice is
refused (`A MatchClaim connects two timelines; both sides are 'tl1'.`). Each
anchor joins the claim's timeline A and timeline B **in that order**, the
order of the table's cells: an anchor written the other way round, and an
end anchor on another pair of timelines, are refused with the message naming
both pairs.

**Certainty and provenance.** `certainty` is the claim's own property,
`1.0` by default, refused outside `[0, 1]` (`Certainty must be in [0, 1],
got -0.1`), the boundaries `0.0` and `1.0` accepted, and `None` accepted as
a certainty not stated (TimeToShare's table allows a null cell). `provenance` holds a
`Provenance` (tests of the entry itself: `tests/core/README.md`,
`test_provenance.py`); `to_dict` writes it as `Provenance.to_dict()` (absent
keys left out) and `from_dict` reads it back equal, asserted on a whole
literal dictionary of one claim. That dictionary's keys are the MatchClaim
table's columns in its order (`timeline_a_id`, `timeline_b_id`,
`start_anchor`, `end_anchor`, `is_synchronous`, `provenance`, `certainty`,
`event_a_id`, `event_a_name`, `event_b_id`, `event_b_name`, `id`), and an
anchor is `{coordinate_a, coordinate_b}` (coordinate dicts): `from_dict`
rebuilds the `AlignmentAnchor` on the claim's two timelines, and refuses an
anchor mapping holding anything else (`timeline_a_id` included), naming the
keys it must hold.

**The matchstamp query uses the claim's own coordinate.**
`get_matchstamp()` asks the bundle at the start anchor's coordinate **for the
claim's timeline A**, as an `IdCoordinate`: the stamp of a `score` (8
quarters) ↔ `perf` (20.0 seconds) claim reads `score` 8 quarters and `perf`
20.0 seconds, from the bundle's graph and reduced alike; the same claim with
its anchor written the other way round is refused before any query.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_instant_creation` | Single anchor → `is_interval == False`, timeline IDs are top-level |
| `test_interval_creation` | Two anchors → `is_interval == True`; both ends read back per timeline |
| `test_auto_generated_and_explicit_id` | a claim made without an id holds `None`; a given id is kept |
| `test_removed_properties_are_refused` | `id_minted`, `is_explicit`, `source_claim_id`, `source_coordinate` are no fields and each is refused as an extra input |
| `test_mismatched_anchors_raises` | **Critical invariant**: an end anchor on another pair of timelines is refused, naming both pairs |
| `test_anchor_timeline_mismatch_raises` | an anchor on other timelines, and one in the wrong order, are refused with the message naming both pairs |
| `test_one_timeline_twice_refused` | a claim naming one timeline on both sides is refused |
| `test_synchronous_requires_anchor` | a synchronous claim without anchors is refused |
| `test_non_synchronous_rejects_anchor` | a non-synchronous claim with anchors is refused |
| `test_claim_names_its_events` | a synchronous claim naming one event, or none, and a non-synchronous claim naming none are refused with the exact messages |
| `test_from_events_places_numbers_on_the_axes` | raw numbers on `quarters` / `seconds` / `ticks` timelines, and a storage cell on `seconds`, take the timeline's unit and number type |
| `test_from_events_converts_another_unit` | `3.16952381` seconds → `139776` samples through the C-Map; an `IdCoordinate` of another timeline and an unconvertible unit are refused |
| `test_from_events_refuses_another_number_type` | a `float` quarters coordinate on a `fraction` timeline is refused with the check's exact message |
| `test_from_events_requires_event_ids` | an event dict without `id` is refused, naming `event_a` / `event_b` |
| `test_check_on_timelines` | the unit half and the number-type half each refused; the same claim accepted against `(unit, number_type)` pairs that state its coordinates; a timeline of another id refused |
| `test_a_claim_without_anchors_holds_no_coordinate` | a NOMATCH claim passes the check whatever axes are given, none included: it holds no coordinate |
| `test_nomatch_factory` | `nomatch()` has no anchors, `is_synchronous=False`, no coordinate in its dictionary (the event's `start` is the event's); the other timeline may be given by id |
| `test_certainty_validation` / `test_certainty_boundaries` | the range rule, its two boundaries and `None` |
| `test_to_dict_literal_with_provenance` | one claim with a given id, a certainty and a provenance holding a belief, as a whole literal, keys in the table's column order, anchors of two coordinates; `from_dict` gives it back equal, its anchor on the claim's timelines; an anchor stating a timeline id is refused |
| `test_matchstamp_query_reads_the_claim_s_timeline` | the claim's stamp, and the refusal of its swapped anchor, as above |

### NOMATCH Claims & Repr

A NOMATCH claim holds no coordinate: the position of its event is the
event's, read from its timeline, and stating it again in the claim would be
a second copy that could disagree (minimal schema). `nomatch()` reads only
the event's `id` and `name`; a `start` in the event dict is not kept.
`to_dict()`/`from_dict()` round-trip the claim.

The `__repr__` of a NOMATCH claim names the two timelines and tags the claim
with the flag word `NOMATCH` (not the internal field name
"non-synchronous"); its `str` names the timelines and the event, no
coordinate.

| Test | Validates |
|------|-----------|
| `test_repr_non_synchronous` | NOMATCH claim repr carries the `NOMATCH` flag and never the literal "non-synchronous" |
| `test_repr_nomatch_names_the_timelines` | Exact strings: repr `MatchClaim(score:clt_1 <-> perf:Chopin_Ashkenazy [NOMATCH])` for an event stating `start` 188.8; `str` the header `MatchClaim (NOMATCH)`, timeline A, event A, timeline B |
| `test_nomatch_roundtrip` | `from_dict(to_dict())` equals the original |
| `test_repr_synchronous_instant_with_units` | a synchronous instant repr carries no badge |
| `test_str_shows_certainty_provenance_and_beliefs` | the whole multi-line `str` of a claim with certainty `0.9`, paradata and a belief |

These assertions are exact strings (ZERO TOLERANCE).

### The Mismatch Test in Detail

```python
def test_mismatched_anchors_raises(self):
    start = AlignmentAnchor(timeline_a_id="tl1", ..., timeline_b_id="tl2", ...)
    end = AlignmentAnchor(timeline_a_id="tl1", ..., timeline_b_id="tl3", ...)  # Different!

    with pytest.raises(ValueError, match="An anchor of a MatchClaim joins ..."):
        MatchClaim(timeline_a_id="tl1", timeline_b_id="tl2",
                   start_anchor=start, end_anchor=end)
```

This prevents creating semantically invalid claims. An interval match between `(tl1, tl2)` and `(tl1, tl3)` would represent... what? The constraint catches this at construction time.

---

## Claim Kinds (`test_claim_type.py`)

### What We're Validating

`MatchClaim.claim_type` is derived from two structural facts and never
stored: whether the claim is synchronous, and how many of `event_a_id` /
`event_b_id` are set. Since a claim always names its events, three kinds
remain: `event_match` (synchronous, an event on each side), `nomatch`
(non-synchronous, one event, the other side the NOMATCH sentinel) and
`conceptual` (non-synchronous, two events). The kinds for a side, or both,
without an event (`projection`, `anchor`) and the kind of a correspondence a
MatchGraph infers (`implicit`, now the graph's own `InferredEdge`, never a
MatchClaim) are not members of `ClaimType`: `ClaimType("projection")`,
`ClaimType("anchor")` and `ClaimType("implicit")` raise, the member list is
exactly the three, and `is_anonymous` is gone. The evidence for each kind is
the path that makes it (`from_events`, `from_events(...,
is_synchronous=False)`, `nomatch`, a `MatchClaimField` row), and the
display: `repr` badges every kind but a plain `event_match` (a NOMATCH claim
reads `MatchClaim(score:1 <-> recording:1 [NOMATCH])`, no coordinate: it
holds none), and a conceptual claim never reads NOMATCH.

---

## MatchClaimField Tests (`test_match_claim_field.py`)

### What We're Validating

`MatchClaimField` is the **MatchClaim table**: a genuine
`SemanticField[MatchClaim]`, the Field paired with the `MatchClaim` scalar.
It holds a very large set of claims as columns derived from the scalar instead
of one frozen `MatchClaim` object per claim. Dense audio-to-audio alignments
push this into the millions of claims, so the store must build vectorized
(never one Python object per row) and materialise individual `MatchClaim`
instances only on demand.

The field's struct schema is **derived from the scalar** by
`derive_arrow_struct(MatchClaim)` and cached on the class as
`MatchClaimField.pa_schema`, like every other paired `SemanticField`.

A provenance shared by every row is a single `Provenance | None` held **once
at field level** and **injected on read**: the `provenance` column is left
null, and `__getitem__` gives each materialised claim the field's
provenance. Each coordinate is stored as the library's one coordinate cell
(`core/time.py`, `COORDINATE_CELL_TYPE`, which is TimeToShare's
`COORDINATE_TYPE`) `{value, numerator, denominator, unit, number_type}`,
both sides of the number filled on every cell: a float's ratio is its exact
dyadic (`0.1` seconds holds `3602879701896397/36028797018963968`, `21.0`
holds `21/1`), never a guessed ratio. The number type decides which side is
read, so a coordinate comes back in the number type it was written in, never
the unit's default — a `quarters` timeline declaring `float` keeps its
floats.

**The table is TimeToShare's table type.** The columns, their types and
their order are TimeToShare's `MATCH_CLAIM_SCHEMA` (`provenance` before
`certainty`; no `id_minted`, `is_explicit`, `source_claim_id`,
`source_coordinate`), asserted equal schema to schema; `from_table` reads the
columns by name, so a table holding them in another order is read the same.
The shared provenance travels under TimeToShare's key
`timetoshare:provenance`. The conformance test (below) reads TimeToShare's
own fixture table and writes it back unchanged.

### Every kind of MatchClaim is a row

The table holds every kind of claim: synchronous claims with their anchors
(interval claims with both), **NOMATCH and conceptual claims without** (null
anchors). `metadata` became `provenance`, and `certainty` is the claim's own
column. **An anchor cell holds `coordinate_a` and `coordinate_b` only**: its
timelines are the row's `timeline_a_id` and `timeline_b_id`, stated once,
so an anchor can no longer name other timelines than its row, and the rule
that it joins them in that order holds by construction (the table has no
cell that could break it). Reading rebuilds the in-memory `AlignmentAnchor`
on the row's two timelines. Readers refuse a row no `MatchClaim` could be,
vectorized, naming the first offending row (`MatchClaim table row {i} ...`):
a synchronous row without a start anchor, a non-synchronous row with an
anchor, a synchronous row not naming an event on each side, a
non-synchronous row naming none, a row naming one timeline twice, a
certainty outside `[0, 1]` (a null certainty is read, as TimeToShare allows
it), an anchor holding a null coordinate (a non-finite number becomes a null
cell: `MatchClaim table row 0 start_anchor.coordinate_a: the coordinate is
null.`, TimeToShare's `invalid_coordinate`). The last is refused by every
reader, `from_claims` included: a claim made of `Coordinate(nan, seconds)`
on side A of its start anchor, or `Coordinate(nan, quarters)` on side B of
its end anchor (`... row 1 end_anchor.coordinate_b: ...`), is refused by
`from_claims`, `from_dict` of that row, `from_table` and `from_parquet` of a
table holding the null cell, `from_package_table` and `from_columns` (a NaN
coordinate column), so no MatchClaim table TimeToShare refuses is ever
written; a NOMATCH row (no anchor) is no problem. Across rows, the coordinates of one
timeline are on one axis: a table holding `audio` in `seconds (float)` in
one row and in `samples (int)` in another is refused, naming the timeline
and both axes, though no timelines are given. These are the rules
TimeToShare's table check applies, so a table the library reads is one
TimeToShare accepts. What proves the kinds are first-class
is the round trip of a field holding all of them (an event match, an interval
claim, a NOMATCH claim, a conceptual claim) claim by claim, through the
dictionary and through Parquet,
and the queries that read them: `filter(synchronous_only=True)` keeps the
anchored rows, `filter(nomatch_only=True)` the NOMATCH row alone,
`at(...)` and `connects_groups` never return or count an anchorless row.

### `from_claims` keeps every claim whole

`from_claims` writes each claim's every property into the row, so
`from_claims(claims).to_claims() == claims` — equality over every field, the
claim `id` included. Provenance follows one rule: one provenance shared by
every claim becomes the field's and the rows carry none; claims whose
provenance differs (`None` included) leave the field's `None` and each row
carries its own; an explicit `provenance=` replaces every claim's own.

**Equality ignores the bundle back-reference.** A claim added to a bundle
records it (`claim.bundle`); a materialised row has none. Equality is a
statement about the claim, so the back-reference takes no part, asserted on
a claim and its copy held by a bundle.

**Table equality is row equality.** Two MatchClaim tables are equal when
they hold equal rows in the same order, every column compared, provenance
included; a provenance the field holds once counts as every row's. A bundle
compares its table this way, so this equality decides bundle equality too.
Evidence: a field equals its dictionary round trip; the gold vector (shared
provenance) equals the same rows each carrying it; one row without it, one
belief added to one claim, another set of rows each make the tables
unequal; two empty tables are equal.

### `from_columns`: event matches on their timelines

The vectorized constructor takes the two timeline-id columns, the two
**event-id** columns and the two coordinate columns, optional end columns,
and `timelines=` (timelines, or a mapping from an id to a timeline or its
`(unit, number_type)` pair). A coordinate is a number on its row's timeline,
written in that timeline's unit and declared number type, and the field is
checked against the timelines before it is returned. A row naming no event
(a null event id) is refused (`Row 1 names no event on side b: ...`), and so
is a row naming a timeline not given; there is no coordinate-only form. The
gold vector below exercises the per-row axes (a `fraction` quarters timeline
beside two `float` ones) and the end columns make one row an interval claim.

**Only a row without an end on either side is an instant claim.** An end
given on one side alone, or one that is no number a cell holds (a NaN),
must not turn an interval claim into an instant claim that is then written
without complaint: the end anchor keeps its null coordinate and the reader
refuses it. Evidence (`TestNullCoordinateRefused`), on `score` (quarters)
↔ `audio` (seconds), row 0 an interval claim with both ends and row 1 the
case: an end on side A only (`[2/3, 1]`, `[0.75, None]`) is refused with
`MatchClaim table row 1 end_anchor.coordinate_b: the coordinate is null.`,
an end on side B only (`[2/3, None]`, `[0.75, 2.0]`) with `... row 1
end_anchor.coordinate_a: ...`, a NaN end on side B (`[2/3, 1]`, `[0.75,
nan]`) with `... row 1 end_anchor.coordinate_b: ...`, and NaN on both sides
(`[2/3, nan]`, `[0.75, nan]`) with `... row 1 end_anchor.coordinate_a:
...`; a null end on both sides stays the instant claim of
`test_from_columns_end_columns`.

### One spelling of a value in memory (`TestOneSpelling`)

Any fraction is valid, `2/4` included, and compares by value. Three places
compare a table's cells member by member: field equality, the succession
check of a file written over (its leaf columns) and the Projection keys. They
compare values because memory holds one spelling of each value: every entry
of whole cells reduces their ratio (`in_lowest_terms`). What would show it
broken: a field built from `2/4` unequal to the one built from `1/2`, a
field refused when written over its own file or over the `1/2` file, a file
read back unequal to the field written. Evidence, on `score` (quarters,
fraction) ↔ `audio` (seconds, float), one claim `n1` ↔ `o1` at `1/2` ↔
`1.0`:

- `from_columns` with `coordinate_a` a number-struct column holding
  `{0.5, 2, 4}` equals the field built from `Fraction(1, 2)`, and its cell
  holds `{0.5, 1, 2, "quarters", "fraction"}`;
- `from_table` of that field's table with the cell rewritten to `(2, 4)`
  equals it;
- through a file (the fields held by a bundle, which mints `claim_1`): the
  `2/4` field written, written again over its own file, then the `1/2`
  field written over it, all accepted; and the other way round, the `1/2`
  field written first and the `2/4` field over it; `from_parquet` of the
  file equals both fields.

### The belief filter

`filter(believer=..., beliefs=...)` is the belief filter of the unified
filter API, decided by `belief_mask` (one predicate, tested for itself in
`tests/core/test_provenance.py`). On the table it is applied to the rows'
provenance, or, when the field holds one shared provenance, to that one for
every row. The field of three claims — no provenance, `jo` believes `false`,
`jo` believes `true` — gives: no believer, all three; `jo` with the default
states (all but `false`), rows 0 and 2; `jo` with `["true"]`, row 2; `jo`
with `["undecided"]`, row 0; `jo` with `["false"]`, row 1; `ana`, all three;
`ana` with `["true"]`, none. A shared provenance in which `jo` believes
`false` keeps no row for `jo` and every row for `ana`. A word that is no
state is refused, a believer given or not.

### Coordinates on the timelines, vectorized

`MatchClaimField.check_on_timelines(timelines)` is the vectorized form of the
scalar check: every coordinate cell is compared with the unit and number type
of the timeline it is on (`coordinate_a` on the row's `timeline_a_id`,
`coordinate_b` on its `timeline_b_id`), and the first offending cell
is refused with the scalar check's message, prefixed by its row and path.
Evidence: a field of a claim in `seconds` on the `samples` timeline refused
(`Row 1 start_anchor.coordinate_b is in seconds (float), but timeline 'pcm'
counts in samples (int): ...`), the same field accepted against `(unit,
number_type)` pairs that state seconds, a row on a timeline not given
refused; `from_parquet(..., timelines=...)` runs the same check.

### Dictionary round trip

`to_dict()` is the table in the one JSON form of the project, not a second
layout: `{"provenance": <Provenance.to_dict() or None>, "rows": [...]}` with
one mapping per claim whose keys are the table's columns, in their order.
Every cell is the table's cell as stored (a nested property a nested mapping,
the beliefs a list of mappings, an anchor `{coordinate_a, coordinate_b}`)
except the coordinate cells (both anchors' `coordinate_a` / `coordinate_b`),
which are coordinate
dicts `{unit, number_type, value}`, a Fraction value as `[numerator,
denominator]` — the form `MatchClaim.to_dict()` writes. What proves the form:

- **The literal dictionary** of a two-claim field (`score` quarters `1/3` ↔
  `audio` seconds `0.5`; `midi` ticks `96` ↔ `pcm` samples `44100`), asserted
  whole: nothing in it holds `numerator`/`denominator`, no anchor holds a
  timeline id, and its rows hold no id (`id` null: a field built from
  columns mints none).
- **The same values as the claim**: each anchor of each row equals the anchor
  of `field[i].to_dict()`, and the cells that are not coordinates equal
  `field.table.to_pylist()`'s.
- **Round trip**: `from_dict(to_dict())` gives claims equal to the originals
  and a table equal to the original's, schema metadata included; the
  dictionary through `timetoshare.dumps(..., "yaml")` / `loads` is equal and
  reads back to the original claims.
- **An edit is read**: changing an exact cell's `value` from `[1, 3]` to
  `[2, 3]` reads `2/3` quarters, a float cell's `0.5` to `0.75` reads `0.75`
  seconds.

`from_dict` refuses a mapping whose keys are not exactly those it must hold,
at every depth: the mapping itself (`provenance`, `rows`), a row (the table's
columns), an anchor, a provenance, each belief of its list, and a coordinate
dict — a missing and an unknown key each named with the path of the mapping
(`row 0 provenance.beliefs.0`), since Arrow would otherwise fill a missing key
with null or drop an unknown one silently.

### Arrow table and Parquet file (`TestTableRoundTrip`, `TestParquetFile`)

**Layout rule.** A Field's table, as written to Parquet, has one column per
top-level property of its scalar, each column nested as the property is. The
test derives the expected names and types from the derived struct and checks
one literal name list besides, so an accidental change to `MatchClaim`'s
fields shows up as a failing test.

`MatchClaimField.from_table(table)` is the inverse of `.table`. The shared
provenance lives outside the claim columns, in the Arrow schema metadata
under `timetoshare:provenance` (the JSON of `Provenance.to_dict()`, asserted
byte-exact); no key means `None`. A table stating provenance both there and
in rows is refused with its count of rows.

**Further columns are kept, never read.** The MatchClaim table of the
format may hold further columns, which a reader keeps and never checks. A
table lacking a property's column is refused naming what it lacks (the
gold field's table without `id`: `A MatchClaim table holds one column per
MatchClaim property, [...]; this table lacks ['id'].`; a table of one
struct column `match_claim` lacks all twelve). The gold field's table with
a further column `extra` (`[1, 2, 3]`) is read: its `table` is the input
with `extra` last, its `package_table` likewise; `filter(timeline_id="C")`
keeps `extra` for its two rows (`[2, 3]`); the field read is not equal to
the gold field (a further column takes part in equality). The carrying
through a belief, a bundle's table and a write over the file is pinned in
`test_bundle_package.py` ("A further column is kept").

**The file holds statements.** Every row is a stated MatchClaim (the table
has no column that could mark one derived: what a `MatchGraph` infers is
its own, never a row). A failed write leaves no temporary file beside it.

`to_parquet(path)` / `from_parquet(path)` write and read the canonical
`match_claims.parquet` (a directory takes that file name). The file holds
the package form (`package_table`; `from_package_table` reads it): every
anchor coordinate cell writes its number once, in the member its
`number_type` names, the others null (`test_coordinate_cell.py`'s
"package form"); memory keeps both sides. So the file's schema equals
`package_table`'s, not `pa_schema`, and the cells written for the field
below, as `(numerator, denominator, value)` of each start anchor's
coordinates A and B, are exactly: `c1` `(7, 3, null)` and `(null, null,
1.25)` (its end anchor `(10, 3, null)` and `(null, null, 2.5)`), `midi
96` `(96, null, null)` and `(44100, null, null)`, `c3` `(null, null, 1.5)`
on the float-declared quarters and `(null, null, 0.75)`; the NOMATCH and
conceptual rows' anchors are null. A file whose int cell keeps the
denominator `1` (the `midi 96` row's `coordinate_b` rewritten to `(44100,
1, null)`) is refused on reading, with the message naming `MatchClaim
table start_anchor.coordinate_b row 1` and what the cell holds. A round trip
proves losslessness only when compared **claim by claim against the original
claims** — a table-to-table comparison would pass a column both sides drop
alike — so the field holds one claim per stored representation: exact
`Fraction` coordinates on `quarters` (`7/3`, `10/3`), `int` coordinates on
`ticks` / `samples` (`96`, `44100`), `float` on `seconds` (`1.25`, `2.5`), a
`float` on a `quarters` timeline declaring `float` (`1.5`), an interval claim,
a NOMATCH claim, a conceptual claim, row provenance with beliefs and a
shared provenance.

**The succession rule.** Writing over an existing file is allowed only when
the new table succeeds it: every id of the old table is in the new one with
every value equal but its provenance. Evidence, each through an actual file:
the same table with a belief added to one claim's provenance, and with one
new claim, are written; a table lacking a claim of the old one is refused
naming the claim (`... lacks the MatchClaim 'c3' of the existing one: a
MatchClaim is never removed ...`), and a table changing a claim's certainty,
or an anchor coordinate, is refused naming the claim and the leaf column
(`'certainty'`, `'start_anchor.coordinate_b.value'`); after a refusal the
file holds what it held.

### Canonical gold vector

```python
field = MatchClaimField.from_columns(
    timeline_a_ids=["A", "A", "B"],
    timeline_b_ids=["B", "C", "C"],
    event_a_ids=["a0", "a0", "b1"],
    event_b_ids=["b0", "c0", "c1"],
    coordinate_a=[0, 0, 1],
    coordinate_b=[10.0, 20.0, 21.0],
    timelines={
        "A": ("quarters", "fraction"),
        "B": ("seconds", "float"),
        "C": ("seconds", "float"),
    },
    provenance=Provenance(service="https://example.org/aligner", method="inferred"),
)
```

Row 2's `B` coordinate is a float (`1.0` seconds) on side a while row 0's
`B` coordinate is on side b: the per-row axes are read from each row's
timeline.

### Key Evidence (exact, zero-tolerance)

| Test | Validates |
|------|-----------|
| `test_from_columns_length` | `len(f) == 3` |
| `test_is_semantic_field` | `issubclass(MatchClaimField, SemanticField)` |
| `test_pa_schema_is_derived` | `pa_schema == derive_arrow_struct(MatchClaim)`; the coordinate cell is `COORDINATE_CELL_TYPE`, `{value, numerator, denominator, unit, number_type}` |
| `test_table_one_column_per_property` | the column names and types are the derived struct's fields, in order (and the literal name list, `provenance` before `certainty`) |
| `test_float_cells_carry_their_exact_ratio` | the gold vector's float cells hold `10/1`, `20/1`, `21/1`, and `0.1` seconds its exact dyadic |
| `test_timeline_ids` | `f.timeline_ids == {"A", "B", "C"}` |
| `test_from_columns_writes_each_row_on_its_axis` | row 0 `A` is `Fraction(0)` quarters, row 2 `B` `1.0` seconds, the events named, `certainty` 1.0 |
| `test_from_columns_end_columns` | an end on both sides makes the row an interval claim, a null end an instant |
| `test_from_columns_refusals` | unequal lengths, a null event id, a timeline not given, one end column alone |
| `TestNullCoordinateRefused` | a null coordinate refused by every reader; an end on one side only or a NaN end refused by `from_columns`, as specified above |
| `TestOneSpelling` | an unreduced fraction is held reduced, so fields compare and succeed by value, as specified above |
| `test_getitem_first_row` | `f[0]` is a `MatchClaim`, `A`↔`B`, synchronous instant, events `a0`/`b0`, provenance identity `is` the field's |
| `test_getitem_injects_provenance` | the `provenance` column is null and a materialised claim carries the field's |
| `test_getitem_out_of_range` | `f[3]` / `f[-4]` raise `IndexError` |
| `TestFilters`, `TestAt`, `TestConnectsGroups` | the vectorized primitives, below |
| `TestEveryKind` | the field of every kind, as specified under "Every kind of MatchClaim is a row" |
| `TestBeliefFilter` | the belief filter, as specified above |
| `TestCheckOnTimelines` | the vectorized check, as specified above |
| `TestDictRoundTrip` | the dictionary form, as specified above |
| `TestTableRoundTrip`, `TestParquetFile` | the table, the file and the succession rule |
| `TestEquality` | a claim equals its copy held by a bundle; the `from_claims` round trip holds; table equality as specified above |
| `test_repr` | `repr(f) == "MatchClaimField(claims=3, timelines=3)"` |
| `test_repr_html_contains_summary` | `_repr_html_()` of the gold vector, exact (a NOMATCH row's, with `—` for its coordinates: `TestEveryKind`) |
| `test_top_level_export` | `MatchClaimField`, `Provenance` and `Belief` importable from `timetoalign` and `timetoalign.alignment` |

### Vectorized query primitives

Every bundle-level claim query is composed from primitives that live **on the
field**, so the answer is computed over Arrow columns and no `MatchClaim` is
built unless the caller asks for one. These are the primitives, tested against
the canonical gold vector above (three rows, `A↔B`, `A↔C`, `B↔C`).

| Test | Validates |
|------|-----------|
| `test_at_matches_a_side` | `f.at("A", 0)` has `len == 2` (rows 0 and 1) |
| `test_at_matches_b_side` | `f.at("C", 20.0)` has `len == 1`, the `A↔C` row |
| `test_at_exact_equality_only` | `f.at("C", 20.0000001)` is empty — no tolerance, no nearest-value fallback |
| `test_at_unknown_timeline_empty` | `f.at("Z", 0)` is empty; the provenance still carries over |
| `test_filter_id_pattern` | `filter(id_pattern=r"^A$")` has `len == 2` (rows 0 and 1) |
| `test_filter_between_order_independent` | `filter(between=("A","B"))` == `filter(between=("B","A"))`, `len == 1` |
| `test_filter_within_requires_both_sides` | `filter(within={"A","B"})` has `len == 1` |
| `test_filter_mutually_exclusive_raises` | `synchronous_only` and `nomatch_only` together raise `ValueError` |
| `test_connects_groups` | `{"A"}`/`{"C"}` True either way round, `{"A"}`/`{"Z"}` False, False on an empty field |

`within` is the vectorized primitive behind domain- and unit-restricted
queries. `include_domains` / `include_units` cannot live on the field itself —
they need each timeline's domain and unit, which the store does not hold — so
the bundle resolves them into the set of timeline IDs that pass and hands that
set to `within`.

### Timeline ids, read once per table (`TestTimelineIdsReadOnce`)

`timeline_ids` is asked on every position query (the bundle's warp units),
so it is read from the two id columns once per struct array and kept beside
it. Two things must hold, and each has its own proof:

1. **The answer is the one the row-by-row reading gave**, as a set and in
   its iteration order: for every field the alignment fixtures build (the
   gold vector, every claim kind from `_every_kind`, the Thoresen segment
   claims of `conftest.py`, a field whose B column names an id before its
   A column does, a filtered field, an empty one) `list(ids)` equals
   `list` of a set filled from `timeline_a_id` then `timeline_b_id` row by
   row (the former expression, kept in the test as the oracle). Equal
   insertion sequences make equal sets with equal iteration order, so the
   comparison is exact in one process.
2. **It is read once per table state**: the read is counted (a wrapper
   around `MatchClaimField._distinct_timeline_ids` recording the array it
   was given). One field asked five times reads once. In a bundle, five
   `get_matchstamp_at` queries read the bundle's MatchClaim table exactly
   once; a MatchClaim added makes a new table, read once more for the next five
   queries (two reads of a table in all, one per state). Assigning another
   field to `match_claim_table` answers that field's ids at once (a field
   never answers for another array), and a caller adding to the returned
   set changes nothing the next call returns.

---

## Claim Ids (`test_claim_ids.py`)

### What We're Validating

A claim id addresses one claim, alone (no scope, no prefix joined to it),
in the table or bundle that holds it; it **never changes** once given, and
the same input gives the same ids in every process. Claims
persist (the MatchClaim table written to Parquet keeps its ids), and a stored
id is what a manifest's named MatchStamp records, so renaming a claim when it
joins a table would break every reference to it. The rule tested:

- **A free claim** made without an id holds `None` until the first bundle
  it joins mints one; a given id is kept verbatim; an id with a control
  character is refused by `validate_id`. Nothing records whether an id was
  given or minted: an id never changes once given, so no reader needs to
  know (no flag, no re-minting that would read one).
- **A field holds each id once and mints nothing** (every constructor:
  `from_columns`, `from_claims`, `from_dict`, `from_table`, `from_parquet`):
  an id held by two rows is refused, **whether given or minted** (`claim id
  'a' is held by 2 claims`), before anything is built, and a row without an
  id keeps none. Were a field to mint, two loaders'
  fields would both hold `claim_1` and could never join one bundle; the
  bundle they join is the one holder that mints. A field whose rows hold no
  id is not written to a file (TimeToShare's table requires every id):
  `MatchClaim table row 0 holds no id, and every MatchClaim of the file
  holds one: ...`.
- **The id is stored**: a row read back keeps its id.
- **A bundle is the holder that mints** for the claims added to it
  (`add_match_claims`, `add_match_claim_field`): a row or claim without an
  id is minted `claim_N`, N above the highest number held (minting gives a
  first id, it renames nothing, and a number is never given twice), a claim
  object in place (its first id); a field's rows in the bundle's table (the
  field passed in keeps its nulls). An id the bundle holds is refused,
  never re-minted. **A merge keeps every id and refuses a shared one**
  (`from_bundles`): a MatchClaim is addressed by its id alone, so two
  sources both holding `claim_1` are refused before anything is merged,
  naming the id and both bundles (the accepted cost: one of them is stated
  again under other ids first); claims added to the merge afterwards are
  minted above the highest id it holds.

### Why this proves "beyond one process"

The minted id depends only on the ids already held, never on what
was built before in the process. One test builds and stores a field in a
subprocess and asserts its ids and table equal the in-process build, so the
determinism across processes is observed, not inferred.

### Field part: Exact Expected Values

| Test | Validates |
|------|-----------|
| `test_free_claim_holds_no_id` | two fresh claims both hold `None`; a given `"onset 7"` is kept; `"a\tb"` raises |
| `test_field_holds_each_id_once` | `from_claims` of claims given `s0`–`s2` keeps them; two free claims are two rows without an id, the objects still `None`; two claims given `"a"` are refused (`claim id 'a' is held by 2 claims`) |
| `test_from_columns_mints_no_id` | the gold vector's ids are `None` ×3; the bundle it joins holds them as `claim_1`–`claim_3`, and the field passed in still holds `None` ×3 |
| `test_two_fields_join_one_bundle` | the `STORED` field (`score ↔ audio`, 4 rows) and a field of the two `NEW` pairs, both from `from_columns`, join one bundle in that order as `claim_1`–`claim_4` and `claim_5`, `claim_6` |
| `test_null_ids_are_minted_around_held_ones` | a table whose rows hold `claim_2`, null, `x`, null reads back as is (`claim_2`, `None`, `x`, `None`); the bundle it joins mints `claim_3` and `claim_4` (above the highest held, `claim_1` never given) |
| `test_a_field_without_ids_is_not_written` | `to_parquet` of the `from_columns` field is refused (`MatchClaim table row 0 holds no id, ...`) and no file is left |
| `test_two_runs_store_the_same_ids` | a subprocess and this process add the same `from_columns` field to a bundle and store its MatchClaim table to Parquet: ids `claim_1`–`claim_4` and the tables equal |

### Bundle part: Exact Expected Values

Fixture: `score` and `score_copy` (quarters, fraction) in one group, `audio`
(seconds, float) in another, or `score` and `audio` ungrouped (`_holding`);
the claims are `from_events` between named notes and onsets on those axes.

| Test | Validates |
|------|-----------|
| `test_two_free_claims_become_distinct` | two free claims, `score ↔ audio` (0, 0.0) and `score ↔ score_copy` (4, 4), both hold `None`: a graph built from them filters by object, so `filter(within={"score", "audio"})` keeps the first alone and `{"score", "score_copy"}` the second alone, both edges carrying `claim_id is None`; `from_claims` stores them without ids and leaves the objects unchanged; a bundle gives the objects `claim_1`, `claim_2` in place, and a graph of the bundle's claims keeps `claim_2` for `{"score", "score_copy"}` |
| `test_stored_and_new_claims_are_distinct` | the stored field (a bundle's MatchClaim table of the four `STORED` claims, written and read back) keeps `claim_1`–`claim_4`; the same four built again plus the two new ones, all without an id, added to the bundle, are given `claim_5`–`claim_10` in order; the caller's objects carry them and `claim.bundle is bundle`; the bundle's claims read back are `claim_1`–`claim_10`, the last six equal to the caller's objects, each with `bundle` the bundle |
| `test_graph_filter_unaffected` | a bundle of the stored field and the two new claims (`claim_1`–`claim_6`): `MatchGraph(bundle.get_match_claims()).extend_to_groups(bundle.groups)` holds exactly those six claims (an inferred edge is no claim and takes no id) and six inferred edges, each on `score`→`score_copy`, its `source_claim` one of the six and its `score` coordinate that claim's; the six stated edges' `claim_id` equal their claims' ids, all distinct, and the other six edges carry the inferred-edge records, in the graph's order; `filter(stated_only=True)` and `filter(within={"score", "audio"})` keep exactly `claim_1`–`claim_6` and no inferred edge |
| `test_held_ids_are_refused_never_reminted` | `from_claims` of a claim given `"anchor"` and a free claim stores `"anchor"` and no id; a bundle holding `claim_1`, `claim_2` takes that field as `"anchor"`, `claim_3`; another bundle's MatchClaim table (`claim_1`, `claim_2`, minted) is refused by it (`claim id 'claim_1' is already held by another claim of this bundle`), which still holds its four; a bundle holding `"anchor"` and a minted `claim_1` gives a free claim `claim_2` |
| `test_two_runs_merged` | a subprocess and this process store the same bundle's MatchClaim table: ids equal (`claim_1`–`claim_4`), tables equal; one bundle takes the first and refuses the second (`claim id 'claim_1' is already held ...`), keeping 4 rows; two bundles, one run each, are refused by `from_bundles` with the exact message `MatchClaim id 'claim_1' is held by bundle '<first>' and by bundle '<second>' (bundles 0 and 1 of the merge): a MatchClaim is addressed by its id alone, so the two merge once one of them states its MatchClaims under other ids.`, each still holding its run's table |
| `test_given_id_refused_atomically` | a bundle holding a given `"a"` refuses a claim and a field with a given `"a"` (`claim id 'a' is already held by another claim of this bundle`); after each refusal the bundle's claims are exactly `[held]` and the newcomer keeps its id (`None` for the one made without) and no bundle |
| `test_merge_keeps_ids_and_refuses_a_shared_one` | `from_bundles` of two bundles each holding `claim_1`, `claim_2` is refused with the message above (naming `claim_1` and both bundles), the sources unchanged (`claim_1`, `claim_2`, each claim recording its source); two bundles holding a minted `claim_1` and a given `claim_2` merge: the merge's claims are `claim_1`, `claim_2`, equal to the two added, each recording the merge, and its table holds the twelve MatchClaim columns alone; a free claim added to the merge is given `claim_3`; a claim another bundle holds, added to a third bundle, joins as a copy: the original keeps its bundle, the third bundle's row records the third |
| `test_merge_refuses_an_id_two_distant_sources_hold` | the merge's one check, made before anything is built, compares every pair of sources (so nothing checks the ids again while the merge appends each source's rows): of three sources holding `claim_1`, `claim_2` and `claim_1`, `from_bundles` is refused naming the first and the third (`... (bundles 0 and 2 of the merge) ...`, the message above), every source unchanged (its ids, each claim recording it) |

---

## The MatchClaim table is TimeToShare's (`test_timetoshare_conformance.py`)

### What We're Validating

The library writes and reads the MatchClaim table that TimeToShare defines
(`timetoshare.tables`): one table type in both repositories, so a file the
library writes passes TimeToShare's check and a file TimeToShare holds is
read by the library value for value. A comparison with the library's own
expectations would prove nothing here — the two could agree with each other
and both differ from the format — so every assertion is against
TimeToShare's own artifacts:

- **Schema.** What the library writes, an empty field's `package_table`
  schema, equals TimeToShare's `MATCH_CLAIM_SCHEMA` (names, types and
  order); the package coordinate cell equals its `COORDINATE_TYPE`; the
  provenance key is its `PROVENANCE_METADATA_KEY`. In memory the cells
  keep both sides of a number, so `pa_schema` is the same columns with
  `COORDINATE_CELL_TYPE` cells.
- **The fixture table, read and written back.** TimeToShare's fixture
  `fixtures/tables/<version>/match_claims.parquet` is read with
  `MatchClaimField.from_parquet`: fifteen claims, `claim_1`–`claim_15`,
  thirteen `event_match`, one `nomatch` (`claim_12`, naming `note_10` and
  the sentinel, no anchor), one `conceptual` (`claim_14`); rows with a null
  `beliefs` read. Every anchor read back stands on its row's two timelines
  (the table's anchor cell holds the two coordinates alone): `claim_2`'s is
  `clt_1` 1 quarter ↔ `dpt_1` 44100 samples. Written back with `to_parquet`, the file read again
  `equals` the fixture's table, schema metadata included; the claims
  materialised and stored again with `from_claims` give the same table. Both
  are content identity, so a dropped, retyped or rewritten cell fails. The
  claims stored again give a `package_table` equal to the fixture table.
- **Columns by name.** The fixture's columns in reverse order read
  (`from_package_table`) to the same claims.
- **What the library writes, TimeToShare accepts.** TimeToShare's table
  requires every id, and a field built from columns or claims mints none
  (only the bundle mints), so what is written is a
  bundle's `match_claim_table`. A field from `from_columns` over a fraction
  `quarters`, an int `samples` and a float `seconds` timeline (a float `0.1`
  among the cells), with a shared provenance, holds three null ids; added to
  a bundle of those three timelines it becomes rows `claim_1`–`claim_3`, and
  the bundle's table written with `to_parquet` gives
  `match_claim_problems(...) == []`, each cell's number written once by its
  number type: `(numerator, denominator, value)` of the start anchors'
  cells A and B are `(1, 3, null)` and `(null, null, 0.1)`, `(4, 1, null)`
  and `(88200, null, null)`, `(44100, null, null)` and `(null, null, 1.0)`
  (the float `0.1` never written as a ratio, the ints without
  denominator), and TimeToShare's `row_provenance` reads the shared
  provenance on every row. An interval claim (`c1`), a NOMATCH claim (`c2`) and a
  conceptual claim without an id, stored with `from_claims`, hold the ids
  `["c1", "c2", None]`; added to a bundle, its table holds
  `["c1", "c2", "claim_1"]` (minted above the highest held number, none) and
  its `package_table` gives `match_claim_problems(...) == []`.

**Where the fixture comes from.** The installed `timetoshare` package ships
it: the test reads `timetoshare.tables.fixture_path("match_claims.parquet")`.
Nothing is skipped; a missing fixture fails.

---

## Beliefs in Held MatchClaims (`test_claim_beliefs.py`)

### What We're Validating

A MatchClaim is never removed and never changed; who believes it is the
one thing that may change, in its provenance.
`AlignmentBundle.add_match_claim_belief(claim_id, *, actor, belief,
date=None, note=None)` records a belief in a claim the bundle
holds, `MatchClaimField.with_belief` and
`MatchClaim.with_belief` are its copy-on-write forms, and
`create_match_claims(..., actor=, date=)` records the stating person's
belief `true`. Every query of the bundle reads the belief filter, the
coordinate getters and `transfer` included. What would show it broken: a
belief that changes another column or another row, an actor holding two
beliefs because one spelling is compact and the other absolute, a belief
the filter does not see (a cache built before it), a file the succession
rule refuses after a belief, a coordinate getter that still moves through a
claim its believer holds `false`, a bundle that can no longer record a belief
in a pair of its own statements (the statements' `add_belief(property,
value, ...)`, which every holder of statements has, shadowed by the
MatchClaim method), an actor spelled with a prefix the manifest declares
only for the MatchClaim table's actors that does not expand.

### Fixture

`score` (quarters, fraction, length 16; notes `n1` at 0, `n2` at 2, `n3`
at 4) and `pcm` (samples, int, length 705600; notes `o1` at 0, `o2` at
100000, `o3` at 176400), both ungrouped roots of the bundle `beliefs`;
`create_match_claims` of `n1`↔`o1`, `n2`↔`o2`, `n3`↔`o3` with one shared
provenance (`method` `imported`, `service` `https://example.org/aligner`)
gives `claim_1`–`claim_3`. The actor is Wikidata's `Q42`, written absolutely
(`http://www.wikidata.org/entity/Q42`) or compactly (`wd:Q42`, a prefix of
the catalogue, hence of every bundle's `@context`). `claim_2` is not on the
line through the other two: 2 quarters lie halfway between 0 and 4, so
without `claim_2` the WarpMap of `claim_1` and `claim_3` places quarter 2 at
176400 / 2 = 88200 samples, and with it at 100000.

### Exact Expected Values

- **The belief is recorded, nothing else changes.** `add_match_claim_belief("claim_2",
  actor=Q42, belief="false", date="2026-11-21")` returns the bundle; the
  provenance of `claim_2` is the shared entry plus that one belief,
  `claim_1` and `claim_3` keep the shared entry, every column but
  `provenance` of the table is equal to the one before, and the table holds
  no shared provenance any more (each row carries its own, since one
  differs).
- **One belief per actor, compared as IRIs.** A second `add_match_claim_belief` on
  `claim_2` by `wd:Q42`, `true`, 2026-11-22, replaces the first: the claim
  holds exactly that one belief, as written (`wd:Q42`).
- **The filter reads it at once.** Before any belief, `get_match_claims(
  believer=Q42)` holds the three; after `false` it holds `claim_1`,
  `claim_3`, with `beliefs=["true"]` none, with `["false"]` `claim_2`, with
  `["undecided"]` `claim_1`, `claim_3`; after the compact `true`,
  `believer=Q42, beliefs=["true"]` holds `claim_2` alone.
- **The coordinate getters take the belief filter.** Quarter 2 on `score`
  onto `pcm`: `get_coordinate_at` gives 100000 without a believer and with
  `Q42` before the belief; after `Q42` believes `claim_2` false it gives
  88200 for `Q42` (absolute or compact) and 100000 without a believer
  (the cache built before the belief is not read). The same holds for
  `get_coordinate_for("n2", "pcm")` (88200), `get_coordinates_at` of
  quarters 2 and 3 (88200, 132300), the dispatcher `get_coordinate`
  (88200), `transfer(2, "score", "pcm")` (88200.0 for `Q42`, 100000.0
  without) and `transfer_interval(1, 3, ...)` ((44100.0, 132300.0) for
  `Q42`, (50000.0, 138200.0) without: 1 and 3 quarters lie at a quarter and
  three quarters of the segments either way). A word that is no belief
  state (`beliefs=["maybe"]`) is refused by each getter, by
  `get_match_claims` and by `get_matchstamp_at` (`The belief filter keeps
  the states ['true', 'undecided', 'unlikely', 'false'], not ['maybe']`).
- **A MatchLine built from a bundle's claims compares actors as the bundle
  does.** After `Q42` believes `claim_2` false,
  `MatchLine.from_claims(bundle.get_match_claims(), "score",
  believer=WD_Q42, source=bundle)` holds two stamps, at 0 and 4 quarters:
  under the bundle's `@context` the compact believer is the absolute actor.
  The same call without `source` declares no prefix, so `wd:Q42` names
  nobody who believes anything and all three stamps (0, 2, 4) remain.
- **Refusals change nothing.** A claim the bundle does not hold (`Bundle
  'beliefs' holds no MatchClaim 'claim_9' (3 MatchClaims).`) and a belief
  word outside the three raise, the table equal to the one before. `ClaimFilter(beliefs=["maybe"])`
  and `believed([], None, ["maybe"])` refuse the word as the getters do.
- **A statement and a MatchClaim each take their belief.** On the fixture
  bundle, `add_statement("dcterms:title", "Beliefs")` then
  `add_belief("dcterms:title", "Beliefs", actor=Q42, belief="unlikely",
  date="2026-11-21")` (the statements' method) and
  `add_match_claim_belief("claim_2", actor=Q42, belief="false",
  date="2026-11-21")`: the bundle's provenance is exactly one entry
  `{"property": "dcterms:title", "value": "Beliefs", "beliefs": [{"actor":
  Q42, "belief": "unlikely", "date": "2026-11-21"}]}`, its statements still
  `{"dcterms:title": ["Beliefs"]}`, and the MatchClaims hold the one belief
  in `claim_2` (`claim_1` and `claim_3` keep the shared provenance). Neither
  belief reaches the other's holder.
- **The actor context is every prefix the manifest declares.** The fixture
  bundle's manifest (`to_dict()`) with `"orcid": "https://orcid.org/"`
  added to its `@context`, a prefix no statement uses and the catalogue
  lacks, read by `from_dict`: `bundle.prefixes == {"orcid":
  "https://orcid.org/"}` and `belief_context(bundle)["orcid"] ==
  "https://orcid.org/"`. A MatchClaim between quarter 0 and sample 0 stated
  with `actor="orcid:0000-0001"` (date 2026-11-21) is kept by
  `get_match_claims(believer="https://orcid.org/0000-0001", beliefs=["true"])`
  (one claim, `claim_1`) — under a context holding only the prefixes the
  statements use, the compact actor would stay as written and the filter
  would keep none.
- **The date defaults to today**, read where statements read it: with
  `timetoalign.timelines.statements._today` patched to 2026-11-21, a belief
  without `date` is dated `2026-11-21`.
- **A claim is addressed by its id alone.** `from_bundles` of `source_a`
  holding `claim_1` and `source_b` holding a given `claim_2` (each n1↔o1 on
  its own timelines): `add_match_claim_belief("claim_2", ...)` changes the
  row of `source_b` alone; `get_match_claims(believer=Q42)` keeps the row of
  `source_a` and drops that of `source_b`.
- **The package succeeds itself.** The bundle written with `to_file`, then
  given a belief and written over its own files, is accepted by the
  succession rule; read back, `claim_2` holds the belief and the table
  equals the bundle's.
- **`create_match_claims(..., actor=)`.** With `actor=wd:Q42,
  date="2026-11-21"` and no provenance, the claim's provenance is exactly
  `Provenance(beliefs=(Belief("wd:Q42", "true", "2026-11-21"),))`; given a
  provenance in which the absolute `Q42` believes `false`, the claim holds
  the paradata and one belief, the compact `true`, in its place. Without
  `actor`, the given provenance stands unchanged (a loader states no
  belief). `date` without `actor` is refused (`a date dates the belief of
  the actor stating the MatchClaims; give actor=`) and nothing is added.
- **Copy-on-write forms.** `MatchClaimField.with_belief("claim_2", belief)`
  over the bundle's table returns a field whose `claim_2` holds the belief
  and leaves the table unchanged; an id the field lacks raises `KeyError`
  (`The MatchClaim table holds no MatchClaim 'claim_9' (3 MatchClaims).`);
  a one-row field holding a shared provenance keeps it shared, with the
  belief. `MatchClaim.with_belief` returns a copy holding the belief and no
  bundle, the claim read from the bundle unchanged and still its bundle's;
  a claim stated without provenance gets an entry holding the belief alone.

---

## A Package: the Manifest and Its Tables (`test_bundle_package.py`)

### What We're Validating

A manifest and the files it names make a package. The manifest's
`resources` is a modelled list (`timelines/manifest.py::Resource`, with
`Integrity`), read and written by the one manifest module, and
`AlignmentBundle.to_file` writes the bundle's two tables beside the manifest
— the MatchClaim table (`match_claims.parquet`) and the Projections table
(`projections.parquet`, every Projection of every timeline of the bundle's
trees with its `timeline_id`) — naming each in `resources`;
`AlignmentBundle.from_file` reads them back through those entries. What
would show the feature broken: a resource read as an opaque blob (an unknown
key would be lost, a malformed one accepted), a table written but not named
or named with a size or digest that is not its file's, a round trip that
changes a value, a Projection added twice, a table written over another
that drops a MatchClaim, a manifest that names a table that is absent and
then fails to load. Each test pins one of these with exact values.

- **Resources are read and refused as the format says.** TimeToShare's
  fixture `bundles/bundle_resources.yaml` reads to `bundle.resources ==
  [Resource("resource_1", "MatchClaim", ("score", "take_1"),
  "match_claims.parquet", "application/vnd.apache.parquet"),
  Resource("resource_2", "Projection", ("take_1",), "projections.parquet")]`
  (IDs bare). An unknown key of a resource and of its `integrity` is kept
  and written back. A one-root manifest read by `Timeline.from_dict` writes
  its resource back. `Manifest.read` refuses, each with its exact message:
  a resource naming a timeline the manifest lacks, a second resource of the
  type `MatchClaim`, an integrity algorithm outside PREMIS's five, a
  resource ID a timeline holds too. A resource ID of any other shape
  (`table_1`) is an ID like any other: it reads and is written back as it
  stands (minted IDs will not be numbered, so nothing assumes
  `resource_N`). (The four
  rejection cases of TimeToShare are also in
  `test_timetoshare_conformance.py`'s refusal test.)
- **The synthetic package, written.** Three roots — `score` (quarters,
  fraction, 16, its child `part` at 8, length 4, the notes `n1` at 0 with
  duration 1 and `n2` at 3/2; `part` holds `p1` at 1), `pcm` (samples, int,
  705600) and `audio` (seconds, float, 16.0) — and, through
  `create_match_claims` with one shared provenance (`method` `imported`):
  `claim_1` `n1` ↔ a span 0–44100 on `pcm` (an interval claim),
  `claim_2` `n2` ↔ 66150 on `pcm`, `claim_3` the position 0.5 s named
  `onset_a` on `audio` ↔ 22050 on `pcm` (both sides positions),
  `claim_4` `p1` (the child's note) ↔ 9.0 s on `audio`, `claim_5` a NOMATCH
  of `n2` towards `audio`. `to_file(dir / "package.yaml")` leaves exactly
  three files. The Projections table holds five rows, timelines sorted by
  ID and each timeline's Projections in the order they were added:
  `onset_a` on `audio` (no source), `projection_4` on `audio` (of
  `part`/`p1`), `projection_1` on `pcm` (of `score`/`n1`), `projection_2`
  (of `score`/`n2`), `projection_3` (no source). A Projection's ID is
  unique among all Projections of the bundle, so each is minted above
  every Projection ID the bundle's timelines hold when it is created:
  `create_match_claims` adds `pcm`'s three first (the timeline a pair
  first gives a Projection), then `audio`'s, which is `projection_4`, never
  a second `projection_1`. `claim_4` names `projection_4` as its event B.
  Each
  row states its axis once, `unit` and `number_type` (`seconds`/`float`
  on `audio`, `samples`/`int` on `pcm`), and `start` and `duration` their
  number once, as `(numerator, denominator, value)`: `onset_a` start
  `(null, null, 0.5)` (no mirror), `projection_4` `(null, null,
  9.0)`, `projection_1` start `(0, null, null)` and duration `(44100, null,
  null)`, `projection_2` `(66150, null, null)`, `projection_3` `(22050,
  null, null)`; every instant's duration null. The
  MatchClaim table passes TimeToShare's check against the manifest written
  (`check_match_claims`), and so does the Projections table
  (`projection_problems` is empty). `bundle.resources` and the
  manifest name `resource_1` (`MatchClaim`, timelines `audio`, `part`,
  `pcm`, `score`) and `resource_2` (`Projection`, `audio`, `pcm`), each
  with the Parquet media type, and `size` and `integrity` equal to the
  file's byte count and the SHA-256 of its bytes, computed in the test from
  the file itself (a Parquet file's bytes depend on the writing library's
  version, so the test pins the relation, not a digest literal).
- **Read back exactly.** `from_file` gives a bundle whose
  `match_claim_table` equals the written one row for row (provenance
  included), whose timelines hold the same Projections (`projection_table`
  of the bundle read equals the file's table), and whose `to_dict()` equals
  the writer's. Written again into a fresh directory, all three files are
  byte for byte the first ones.
- **The actor context survives the package.** The synthetic package with
  `add_prefix("orcid", "https://orcid.org/")` on the bundle and the belief
  `true` of `orcid:0000-0001` recorded in `claim_2`
  (`add_match_claim_belief`, date 2026-11-21): no statement uses `orcid`,
  only a MatchClaim table actor does. The manifest written holds `"orcid":
  "https://orcid.org/"` in its `@context` beside the format's terms and
  nothing else; read
  back, `bundle.prefixes == {"orcid": "https://orcid.org/"}`, the belief is
  still written `orcid:0000-0001` in the table, and
  `get_match_claims(believer="https://orcid.org/0000-0001", beliefs=["true"])`
  keeps `claim_2` alone, as it does on the bundle written. A writer keeping
  only the prefixes statements use would drop the declaration: the compact
  actor would then expand to nothing after the reload and the filter would
  keep no claim. Written again, the files are byte for byte the first ones.
- **A prefix neither the statements nor the table use is not written.**
  The written `@context` holds what its readers need, no more: a prefix
  the statements use or the MatchClaim table's provenance uses (its
  actors), whether the bundle declares it or the catalogue does. The
  synthetic package with `add_prefix("ex", "http://example.org/ns/")`
  (used by nothing) and `add_prefix("orcid", "https://orcid.org/")`, the
  belief `true` of `orcid:0000-0001` in `claim_2` and of `wd:Q1` (a
  catalogue prefix no statement uses) in `claim_3`, written with its table:
  the `@context` is exactly the format's terms, `orcid` and `wd` with the
  catalogue's namespace; `ex` is not written. Read back, `prefixes` is
  `{"orcid": "https://orcid.org/"}` (`wd` is the catalogue's, so not the
  bundle's own). A writer keeping every declared prefix would write `ex`;
  one keeping only what statements use would drop `orcid` and `wd`.
- **The succession rule binds the writer.** Writing the same bundle over
  its own files succeeds and changes no byte. A bundle holding only
  `claim_1`–`claim_4` of those claims, written over them, is refused (`The
  new MatchClaim table lacks the MatchClaim 'claim_5' of the existing
  one: ...`) before anything is written: the three files keep their bytes.
- **No claims and no Projections, no table.** A bundle with timelines and
  neither MatchClaims nor Projections writes the manifest alone, which
  holds no `resources`.
- **Projections without claims.** A bundle whose `pcm` holds one
  Projection (`Timeline.add_projections`, 22050, no source) and no
  MatchClaim writes the manifest and `projections.parquet`, named by
  `resource_1` (`Projection`, `pcm`); read back, `pcm` holds that
  Projection (the bundle's Projections table equals the file's).
- **A named table that is absent.** The fixture `bundle_resources.yaml`
  copied alone into a directory loads (a manifest is read without its
  tables), holds no claims, keeps both entries, and written back gives
  the same bytes and no table file. Written into a directory where neither
  file lies, after a Projection is added to `take_1` (no claim), the
  Projections table is written under its entry (`resource_2`) and the
  MatchClaim entry, whose file is neither written nor lying there, goes.
- **No event row beside the tables.** A manifest never names an event's
  ID: `to_file(..., events=True)` of the synthetic package is refused
  before any file is written (`A package's manifest names no event ID: ...
  write it with events=False.`), and so is a bundle holding Projections
  alone. A manifest whose event rows name a Projection the Projections
  table holds too (written by hand: the package's manifest with the rows of
  `pcm` and `audio` added) reads each Projection once (`pcm` 3 events,
  `audio` 2), the table's row reusing the Projection held under its ID.
- **Placement.** An entry read with the manifest keeps its ID and URL:
  with `bundle.resources = [Resource("resource_7", "MatchClaim",
  ("score",), "tables/claims.parquet")]` the table is written to
  `tables/claims.parquet`, the entry keeps `resource_7` and its URL and
  names the four timelines the claims name; the Projections table, new,
  gets `resource_8` beside the manifest. A URL with a scheme
  (`https://example.org/claims.parquet`) is refused.
- **Merged claims keep their ids.** `from_bundles` of `source_a` and
  `source_b`, each holding a score (quarters) and a samples timeline with
  the Note `n1` and the onset `o1` at their origins and one claim
  `n1` ↔ `o1`: `source_a`'s minted `claim_1`, `source_b`'s given `claim_2`;
  `merged` then creates `claim_3` between `n2` (1 quarter on `a_score`)
  and `o2` (44100 samples on `b_pcm`). No claim needs a Projection, so the
  package holds no Projections table. `to_file` writes the MatchClaim table
  with exactly the twelve MatchClaim columns, ids `claim_1`, `claim_2`,
  `claim_3`, which TimeToShare's check accepts against the manifest
  written; read back, the MatchClaim table and `to_dict()` equal the
  writer's, and a second write gives the same bytes. Written over its own
  files the merged bundle changes no byte; the same merge with
  `source_b`'s claim given `claim_7` instead, written over them, lacks
  `claim_2` and is refused (`The new MatchClaim table lacks the MatchClaim
  'claim_2' of the existing one: ...`), the files keeping their bytes: a
  MatchClaim is its ID.
- **A further column is kept.** The merged package's MatchClaim table
  (`claim_1`, `claim_2`) rewritten by hand with a `scope` column appended
  (`source_a`, null), as a package written by an earlier tool holds one:
  `MatchClaimField.from_parquet` reads it, the column last in `table`, its
  values as written, and `package_table` equals the file's table;
  `AlignmentBundle.from_file` holds the same field, and its
  `filter(timeline_id="b_score")` keeps the row's `scope` (null). Written
  to another directory, the MatchClaim table equals the hand-written one;
  after a claim `n2` ↔ `o2` is added (minted `claim_3`) and `claim_1` is
  believed `false`, the rewritten table's `id`/`scope` rows are `claim_1`
  `source_a`, `claim_2` null, `claim_3` null: the column is carried row
  for row, null for a row added, and never compared by the succession
  rule.
- **An unreduced fraction in the package's tables** (the format's rule that "any fraction
  is valid" at the file boundary, through the whole path). A package of
  `score` (quarters, fraction, 8; `n1` at 1/2) and `audio` (seconds, float,
  60.0; `o1` at 1.0) with `claim_1` `n1` ↔ `o1` and `claim_2` the position
  3/2 on `score` ↔ `o1` (a Projection `projection_1` on `score` at 3/2, of
  `audio`/`o1`) is written, then both files are rewritten by hand with every
  fraction spelled doubled: the MatchClaim cells `(2, 4, null)` and
  `(6, 4, null)`, the Projection's start `(6, 4, null)`. `from_file` reads
  it: its `match_claim_table` equals the writer's and `score` holds one
  Projection, `projection_1` at `Coordinate(3/2, quarters)` (cell
  `{1.5, 3, 2}`). `add_projections` of the Projection of `audio`/`o1` at
  `Coordinate(3/2, quarters)` on the `score` read (the package holds no
  Note: a manifest names no event, so the Projection is its one event)
  returns `["projection_1"]` and `score` still holds one event: get-or-add
  matches a cell read as `6/4` with one placed as `3/2`. Written
  back over the rewritten files (the succession rule compares values, so
  the file's `6/4` is the bundle's `3/2`), the three files are byte for byte
  the ones first written (a writer respells reduced).
- **A Projections table that breaks the format** (a row naming the
  timeline `ghost`) makes `from_file` raise TimeToShare's `TableError`.
- **Projection IDs are unique across the package** (TimeToShare's check
  keys the Projections table on the ID alone). Reading: the file's row 3
  (`projection_2` on `pcm`) rewritten to `projection_1` is refused by that
  check, which the reader raises as it stands, a `TableError` `the ID
  'projection_1' is given twice`; so is row 1 (`projection_4` on `audio`)
  rewritten to `projection_1`, the same ID on two timelines, with the
  same message. Writing: IDs are minted unique when a Projection is
  created in its manifest, and nothing is minted again on writing (an ID
  once given is stable, a MatchClaim may name it). Timelines that created
  their Projections apart and joined afterwards may each hold one ID, so
  the writer refuses such a table before any file is written: a free
  `part` given `projection_1` at sample 5 joins `pcm` of a bundle whose
  `audio` holds `projection_1` (minted there at 0.5 s) as a child (the
  join succeeds: `pcm` has one child, `part` keeps `projection_1`), and
  `projection_table` of `audio` and `part` raises `Projection ID
  'projection_1' names a Projection on the timeline 'audio' and one on
  'part': a Projection's ID is unique among all Projections of a manifest
  and its package. Create the MatchClaims once the timelines share the
  bundle (for instance with AlignmentBundle.create_match_claims), so that
  their Projection IDs are minted unique` (the timelines in the table's
  order). A writer that let it pass would write a table TimeToShare's
  check refuses; one that minted again would change an ID a MatchClaim may
  name.
- **A join keeps a Projection ID the manifest holds; the writer refuses
  it** (`TestProjectionIdsAtAJoin`). Matching before bundling is a valid
  workflow (each timeline of a separate group mints its own Projections),
  and a MatchClaim addresses an event by its timeline and ID, so a join
  is never refused for a Projection ID. Two timelines given Projections
  apart each mint `projection_1` (`pcm` at 22050 samples, `audio` at
  0.5 s). Every way a timeline joins a bundle's manifest passes one
  admission, so each is tried once with the bundle holding `pcm`:
  `add_timeline(audio)`, `add_timeline(audio, grouped_with="pcm")` (the
  group, held by the bundle, admits it), `add_group` of a group holding
  `audio`, and `from_bundles` of a bundle holding `pcm` and one holding
  `audio` (the child join, `add_child`, is the item above). Each join
  changes exactly what it changes otherwise: the bundle's timelines are
  `['pcm', 'audio']`, its groups' members as the join makes them (none;
  `[['pcm', 'audio']]`; `[['audio']]`; none), and each timeline keeps its
  one Projection `projection_1` at its position (never minted again). A
  Projection then added without an ID to `audio` is minted
  `projection_2`, above every Projection of the manifest. Writing the
  package raises `ValueError` before any file is written (the directory
  stays empty), naming the ID and both timelines in the table's order
  (sorted by timeline ID: `audio`, then `pcm`) and the remedy, the message
  of the item above. The control: `audio`'s Projection given the ID
  `onset_a` instead joins, and the package holds `onset_a` on `audio` and
  `projection_1` on `pcm`. A bundle whose MatchClaims were created inside
  it writes and reads back with its exact Projection IDs (the loader tests
  of the Performance Precision and MAPS bundles).
- **Boxes in the package** (`TestBoxes`). One writer serves every event
  type: `event_table(timelines, event_type)` writes `id`, `timeline_id`,
  `unit`, `number_type`, then every property of the type after `id`, each
  number written once by its row's number type, at whatever depth it lies
  in a shape; `read_event_table` is its inverse. The Projections table is
  that writer for `Projection`: `event_table_schema(Projection)` equals
  TimeToShare's `PROJECTION_SCHEMA`, and the byte-equal tests above
  (`test_the_package_is_written_back`, the fixture package) stay green, so
  the generalisation changed no byte. `event_table_schema(Box)` is `id`,
  `timeline_id`, `unit`, `number_type` (strings), `start`, `duration` (the
  package number), `bounding_box` (`x`, `y`, `width`, `height`, each the
  package number) and `quadrilateral` (`top_left`, `top_right`,
  `bottom_right`, `bottom_left`, each a struct of `x` and `y` package
  numbers). The package bundle holds three roots:
  - `page` (`DiscreteGraphicalTimeline`, pixels, int, length 4512,
    `meta["box"]` `(0, 0, 4512, 6174)`, `meta["image"]` the Gallica URL
    `https://gallica.bnf.fr/iiif/ark:/12148/bpt6k11620688/f2`), its child
    `sys` at offset 1221, length 2880, `meta["box"]` the quadrilateral of
    DMOS system 1, holding `m1` (start 0, duration 940) and `m2` (930,
    624), both quadrilaterals (values in `tests/timelines/README.md`,
    `test_boxes.py`);
  - `seg_2` (pixels, length 867, `meta["box"]` x 7) with `rect_h2`
    (start 378, duration 139, the bounding box (385, 46, 139, 20));
  - `uu` (`ContinuousGraphicalTimeline`, `user_units`, fraction, length
    2) with `f`, the bounding box (1/3, 0, 2/3, 1), start 1/3, duration
    2/3.

  It holds neither MatchClaims nor Projections, and `to_file` writes
  exactly `boxes.parquet` beside the manifest. Its rows, timelines sorted
  by ID (`page` holds no Box): `rect_h2` on `seg_2`, `m1`, `m2` on `sys`,
  `f` on `uu`, with `unit`/`number_type` `pixels`/`int` ×3 and
  `user_units`/`fraction`. `rect_h2.start` is `{numerator: 378,
  denominator: null, value: null}` (an int is its numerator alone);
  `f.start` is `{1, 3, null}`, `f.bounding_box.y` `{0, 1, null}` and
  `f.bounding_box.height` `{1, 1, null}` (a fraction keeps its
  denominator, 1 included); `m1.quadrilateral.bottom_right` is `{x: {2161,
  null, null}, y: {2266, null, null}}`; `rect_h2.quadrilateral` and
  `m1.bounding_box` are null. The resource is `resource_1`, type `"Box"`,
  timelines `["seg_2", "sys", "uu"]`, URL `boxes.parquet`, the Parquet
  media type, its size and SHA-256 those of the file. `from_file` reads
  back the four Boxes equal to the originals (each timeline's Box rows
  read by `Box.from_row` on its axis), the timelines' `meta` as written;
  written again into a fresh directory, both files are byte for byte the
  first ones. A writer that wrote numbers through a float would lose `f`'s
  thirds; one that dropped nested leaves' number types would write `{1,
  1}` as an int.
- **A Box table contradicting its shapes is refused on reading.** The
  table rewritten with `m1.start` `{1, null, null}`: `from_file` raises
  the invariant message of `Timeline.add_events`, `Box 'm1' starts at 1 on
  timeline 'sys', but its shape starts at x 1221 and the timeline stands
  at x 1221 of its image (meta["box"]): its start is 0.` Every Box read
  passes `add_events`, so a table is never trusted over its shapes. A
  Box's own rule is refused in the same words on every path: the table
  rewritten with `m1.duration` `{10, null, null}` gives exactly the
  sentence `add_events` gives for that row as a dict, added to an empty
  timeline standing where `sys` does (its length and `meta`) (`Box 'm1':
  its duration is 10 and its shape spans 940 in x (1221 to 2161); a Box's
  duration is its shape's x extent.`), not pydantic's report around it;
  the test asserts both the literal text and the equality with the
  `add_events` refusal, so a reader wrapping the sentence fails.
- **Boxes beside event rows.** As for Projections, a bundle holding Boxes
  writes them in `boxes.parquet` alone: `to_file(..., events=True)` is
  refused before any file is written (`A package's manifest names no event
  ID: the bundle writes its MatchClaim table and its event tables beside
  the manifest, so its event rows are not written into it; write it with
  events=False.`).
- **Event tables of timelines holding other events**
  (`TestEventTablesBesideOtherEvents`). A table of one event type holds
  exactly the events of that type of each timeline, whatever other events
  the timeline holds and whatever columns they bring (a score's Notes, with
  a `midi` column and no `projection_of`): a writer reading a column the
  Projections lack, or missing one they have, fails or writes a Note. A
  column the timeline's table lacks is written as typed nulls of the
  package schema. Each case writes the package, reads it back and writes it
  again into a fresh directory, byte for byte the first one.
  - *One Projection beside Notes.* `score` (`ContinuousLogicalTimeline`,
    quarters, fraction, length 16) holds `n1` (start 0, duration 1, `midi`
    60) and `n2` (start 3/2, `midi` 62), then one Projection added by
    `add_projections` (start 1/3, duration 2/3; minted `projection_1`). The
    bundle `one` holds `score` alone; `to_file` writes exactly
    `one.yaml` and `projections.parquet`, whose one row is `{id:
    projection_1, timeline_id: score, unit: quarters, number_type:
    fraction, start: {1, 3, null}, duration: {2, 3, null}, projection_of:
    null}`; the resource is `resource_1`, type `"Projection"`, timelines
    `["score"]`, URL `projections.parquet`. Read back, `score` holds that
    Projection alone (start 1/3, duration 2/3, `projection_of` null: the
    manifest names no Note).
  - *Two Boxes beside a Note.* `seg_2` (pixels, int, length 867,
    `meta["box"]` x 7) holds the Note `note_1` (start 0, duration 10,
    `midi` 60), `rect_h2` (the bounding box (385, 46, 139, 20): start 378,
    duration 139) and `rect_b` (the bounding box (400, 100, 50, 30): start
    393, duration 50). `to_file` writes exactly the manifest and
    `boxes.parquet`, whose rows are `rect_h2` and `rect_b` on `seg_2`,
    `pixels`/`int`, starts `{378, null, null}` and `{393, null, null}`,
    durations `{139, null, null}` and `{50, null, null}`, bounding boxes as
    given, `quadrilateral` null; read back, `seg_2` holds those two Boxes,
    equal to the originals.
  - *Two timelines, one holding a Projection.* `score` holds `n1` (as
    above) and `pcm` (`DiscretePhysicalTimeline`, samples, int, length
    705600) holds the Note `o1` at 22050. One MatchClaim states the score
    side as the position `{start: 0, end: 4}` and the performance side as
    the event `o1`: it adds the Projection `projection_1` on `score` and
    none on `pcm`. `projections.parquet` holds exactly one row, `{id:
    projection_1, timeline_id: score, unit: quarters, number_type:
    fraction, start: {0, 1, null}, duration: {4, 1, null}, projection_of:
    {timeline_id: pcm, event_id: o1}}`; the Projection resource names
    `["score"]` alone. Read back, `score` holds that Projection with its
    reference, and the bundle holds the one MatchClaim, `claim_1`, from
    `projection_1` on `score` to `o1` on `pcm`.
  - The byte-equal fixture tests above (`test_the_package_is_written_back`,
    `TestTimeToShareFixturePackage`), whose timelines hold Projections
    alone, are unchanged and stay green.

### Conformance with TimeToShare's fixture package

TimeToShare's `fixtures/tables/<version>/` is a package: `manifest.yaml`
naming `match_claims.parquet` and `projections.parquet`, and
`expected.yaml` holding, for seven belief filters, each a `believer` and its
`beliefs` (states of the four: none with the default; the actor
`0000-0002-1825-0097` with the default `["true", "undecided",
"unlikely"]`, with `["true"]`, with `["undecided"]` and with
`["unlikely", "false"]`; the actor `0000-0001-5109-3700` with `["true"]`;
the first actor written compact, `orcid:0000-0002-1825-0097`, with
`["true"]`, which must keep what its absolute IRI keeps),
the expected unique alignment anchors
with the MatchClaims stating each and the expected potential MatchStamps.
Those values are literal data written by hand in TimeToShare's emission
script, never computed by a derivation, so they are an independent oracle.

- `AlignmentBundle.from_file(manifest.yaml)` holds the fifteen MatchClaims
  and the eleven Projections (eight on `dpt_1`, two on `cpt_1`, one on
  `clt_2`; `clt_1` holds none), so every timeline's events are its
  Projections alone; the bundle's Projections table equals the fixture's
  rows (the table is read from the bundle's timelines sorted by ID, each
  timeline in its event order: `clt_2`, `cpt_1` ×2, `dpt_1` ×8, the order
  the fixture holds them in). The Projections file is the fixture's byte
  for byte, as the MatchClaim file is.
- Written to a fresh directory, `match_claims.parquet` and
  `projections.parquet` are the fixture's byte for byte, and both resource
  entries equal the fixture's (size and digest included).
- For each case, `MatchGraph(bundle.get_match_claims(), believer=,
  beliefs=, source=bundle)`, the case's `believer` and `beliefs` read as
  they stand (the fixture holds exactly the seven cases): the unique anchors (each synchronous claim's
  start and end anchors, both coordinates sorted, with the IDs of the claims
  stating them) equal the expected anchors, and the graph has one edge per
  unique anchor; the coordinates of each connected component
  (`split_components`) with the IDs of the claims stating its anchors equal
  the expected potential MatchStamps. Both are compared as TimeToShare's
  `dumps` text of the sorted lists, so a fraction must be `[n, d]`, an
  integer an integer and a float a float. End anchors are included
  (`claim_2`, `claim_9`). The bundle's own belief filter
  (`get_match_claims(believer=, beliefs=)`) keeps the same claims as the
  graph's, and so does the MatchClaim table filtered under the fixture
  manifest's own `@context`, read from the file
  (`match_claim_table.filter(believer=, beliefs=, context=)`): the seventh
  case names its believer by the compact IRI `orcid:…`, which only that
  context expands, so every case is read under it, as TimeToShare reads
  them.
- The fixtures are TimeToShare's own, found through its package
  (`timetoshare.tables.fixture_path`, `timetoshare.fixtures.fixture_paths`),
  which every installation of `timetoshare` carries: the tests never skip.

---

## AlignmentBundle Dictionary Form and Identity (`test_bundle_dict.py`)

### What We're Validating

A bundle is its timelines and its groups. Its dictionary form is a
TimeToShare manifest: the bundle's `id` only when one was given, its `name`
and `meta`, every root timeline once (each the timeline object
`Timeline.to_dict()` writes, sorted by id), the groups as `Group` objects
naming their members by id (sorted by id), every group's member order
under `graphical.order`. MatchClaims and the support policy (a setting of the
bundle, not of the manifest) are not part of the form; claims are excluded
from every assertion here.

### What proves it

1. **Round trips that behave.** For every bundle built in the tests and the
   bundle of every alignment loader with a specimen — Parangonada (Eroica,
   five performers), Matchfile (all 22 Vienna performances), MPM (both
   specimens), IEEE 1599 (`gymnopedie`), ListenHere (the synthetic canonical
   specimen of `test_listen_here_loader.py`; the format has no corpus), TiLiA
   (Bruckner 5 scherzo), PerformancePrecision (the Chopin nocturne) and
   Rekordbox (the DJ-set specimen, skipped where absent) —
   `AlignmentBundle.from_dict(b.to_dict()).to_dict() == b.to_dict()`. The
   dictionary equality alone could hold for a reader that rebuilds the wrong
   object, so the rebuilt bundle must also answer every group's
   `get_timestamp_at` identically (value, type and unit of every cell) at the
   member's 0, a third of its length, half its length and its length, from
   every member. Claims are excluded: the MatchClaim table is not consulted. The
   rebuilt bundle must also hold exactly the same participants and the same
   group membership.
2. **Exact expected forms.** One direct construction pins the whole
   manifest in the complete form (`to_dict(minimal=False)`), which writes
   every key TimeToShare gives a default, the empty `resources`,
   `graphical.timestamps`, `graphical.matchstamps` and `graphical.elisions` included: two groups
   and an ungrouped root, a group member that is a
   child (its root is written, the child id is the member), the `group_N`
   ids, no `id` (the bundle's `bundle_1` is minted), no `support_policy`,
   and the member order as added. A bundle with a given id writes it; an
   anonymous bundle read back writes no id again.
3. **Identity.** A fresh bundle is `bundle_1`; `as_group` names the new group
   and its id is minted `group_1`, `group_2`; a child added to a group brings
   its root, ungrouped; a child added ungrouped registers only its root (an
   ungrouped participant is always a root), and the round trip restores
   exactly that participant set; three loader-minted `cpt_1` timelines join
   one bundle as `cpt_1`, `cpt_2`, `cpt_3`; a timeline already in one group
   may join another (see "A Timeline in Several Groups" below: the root
   brought along with its child and grouped later joins a third group by
   `as_group`, and a pre-built group holding a grouped member is admitted);
   the registered trees share one manifest, in which every id is unique
   whatever element holds it (timelines, conversion maps, regions and
   flow-control elements alike; the other kinds are proved in
   `tests/timelines/test_manifest_ids.py`); a given timeline id held by
   another timeline (of any tree in the bundle, descendants included)
   raises `ValueError` (`timeline 'holes' cannot join bundle 'bundle_1':
   its id 'holes' is already held by timeline 'holes'`) and leaves the
   bundle unchanged; a minted id held there is minted again. A refused newcomer keeps
   its ids: every collision check — the timeline ids and the group id —
   runs before any id is minted again, so a minted `clt_1` that admission
   would have re-minted is still `clt_1` after its group or its root was
   refused. A given group id refuses a timeline that carries it (`timeline
   'g' cannot join bundle 'bundle_1': its id 'g' is already held by group
   'g'`); a *minted*
   group id that a newcomer's given timeline id takes is minted again
   (`group_1` → `group_3` beside an existing `group_2`), and the members keep
   their membership under the new id. `from_bundles` merges two bundles'
   minted groups as `group_1` and `group_2`. A merge locks nothing and
   changes no source: a bundle read from a manifest whose group `g` holds a
   segment line `line` (segments 4 and 2 quarters) stating `locked: false`
   and a plain `plain`, merged with a bundle of one group of `c`, gives `line`
   `False`, `plain` and `c` `False`, written `false`; stated `true`, `line`
   stays `True` and is written `true`. Either way the source bundle's `line`
   keeps its lock and the source writes exactly the manifest it wrote before
   the merge.
4. **Nesting.** A parent and its child may sit in *different* groups of one
   bundle — each group converts within itself and the child's offset relates
   them. In *one* group they are refused: `grouped_with` naming a member
   nested with the newcomer raises `ValueError` naming both ids before
   anything changes, and so does a `Group` object in `from_dict` naming both.
5. **Refusals.** `from_dict` raises `ValueError` naming the problem for a
   missing marker or required key, an empty manifest id, a timeline written
   twice, a group id written twice or equal to a timeline id, a member naming
   no timeline, a group naming one member twice (`names a member twice`; two
   groups sharing a member are no error), a group naming a parent and its
   child, a key of `graphical.order` naming no group or timeline (`'graphical.order'
   names no group or timeline: ['group_7']`), and a group's order that is not a
   permutation of its members. A key the library does not know is not
   refused: it is kept and written back.
6. **Styling is carried, never read.** A bundle file's `graphical` may hold
   `styles` and `rules` beside `order`; they belong to the bundle, not
   to any of its groups, and the library keeps them on
   `AlignmentBundle.styling` without interpreting them. A styling block of a
   named style, a rule setting it by name and one setting a declaration
   directly survives `from_dict` → `to_dict` equal to the input, beside the
   unchanged per-group `order`, and the rebuilt bundle's `styling` shares no
   object with the input. The direct construction writes `graphical` as
   `{"order": {...}}` alone in the minimal form (item 2), so absent styling
   is never invented. Without `order`, every group joins in `members` order.
   Any other key of `graphical` (`frame`) is kept and written back, never
   read, by the one reader serving groups and bundles. The band orders of
   `graphical.order`, keyed by a timeline's id, are carried beside the
   styling; their tests, the bundle's included, are those of "Manifests" in
   `tests/timelines/README.md`.

---

## Note Onsets for Matching (`test_matching.py`)

### What We're Validating

`prepare_abc_notes_for_matching` keeps one row per sounding onset of an ms3 notes table.
ms3's `tied` column is `1` where a note is tied to the following one (a tie starts),
`-1` where it is tied to by the previous one (a tie ends), `0` where both hold (a tie
continues) and empty where the note is not tied. A note sounds anew only when it is not
tied or starts a tie; the continuation and end of a tie prolong a sound already started.
The function used to keep empty and `-1` and drop `1` and `0`, so every tied sound was
matched at its last note instead of its first.

### Proof

A hand-written frame of seven notes, each with an `id`, in row order:

| id | tied | reading |
|---|---|---|
| `a` | empty | untied |
| `b` | `1` | starts a chain |
| `c` | `0` | continues it |
| `d` | `0` | continues it |
| `e` | `-1` | ends it |
| `f` | `1` | starts a two-note tie |
| `g` | `-1` | ends it |

The kept ids must be exactly `["a", "b", "f"]`, in this order and re-indexed from 0. Keeping
the old reading gives `["a", "e", "g"]`; keeping `0` gives the chain three onsets; dropping
the empty cell loses the untied note, so each wrong reading fails. The frame is tested twice,
the tie cells once as floats with `NaN` (how `pandas.read_csv` reads an ms3 TSV) and once
as nullable `Int64` with `<NA>` (how a loaded timeline's event table arrives), with the same
result. The coordinate column holds the fraction strings ms3 writes; the kept rows read
`[0.0, 1.0, 707/2 = 353.5]` exactly, and the pitch column is renamed `pitch`. The frame is
built as pandas builds it, so under pandas 3 the strings arrive in a `str` column, not an
`object` one: the conversion used to test for `object` and left every coordinate a string.

Two more runs read the tie cells as text, as a TSV read with every column as `str` holds
them: `"1"`, `"0"`, `"-1"` as in the table, and for the untied note once `""` and once
`" "`. A text cell that is empty or holds only whitespace is an empty cell, so the kept ids
are again exactly `["a", "b", "f"]` in both runs. Parsing the text as numbers alone would
refuse the whitespace cell (`pd.to_numeric` cannot parse `" "`), and the untied note would
not be kept.

A tie cell outside ms3's values is not read as anything: a frame whose cells are `1` and `2`
raises `ValueError` with exactly `Column 'tied' holds tie values other than 1, 0, -1: [2.0]`,
so an unknown value is neither kept as an onset nor dropped silently.

The kept tie start carries its own notated duration, not the length of the chain: the
function selects onsets and promises no duration.

---

### Claims from matched notes

`match_notes_by_attributes` pairs rows of two frames greedily and states one MatchClaim per
pair. A claim joins two events, so the function takes the two timelines and, per side, an
optional id column naming each row's event on its timeline; a side without one states
positions only and each of its matched rows becomes a `Projection` added to its timeline.

The fixture is built by hand so every expected value is read off it: a performance timeline
in samples at 44100 Hz (with `SamplesToSeconds`), events `p1` `p2` `p3` at samples 22050,
44100, 66150, and a score timeline in quarters, events `n1` `n2` `n3` at `1/2`, `1`, `3/2`.
The performance frame holds the onsets in **seconds** (0.5, 1.0, 1.5) and the pitches
`C4 D4 E4`; the score frame the onsets as the fraction strings `"1/2"`, `"1"`, `"3/2"` and
the pitches `C4 E4 D4`, so a greedy pass pairs `p1–n1`, `p2–n3`, `p3–n2` (an order that
differs from row order, so a claim built from row positions instead of matched pairs fails).

* **Both sides named** (`source_id_column="id"`, `target_id_column="id"`): exactly the
  claims `(p1, n1)`, `(p2, n3)`, `(p3, n2)`, side A the performance, each anchor at the
  sample (22050, 44100, 66150: the seconds converted through the C-Map) and the exact
  quarter (`1/2`, `3/2`, `1`, read from the strings, never through a float); no event is
  added to either timeline.
* **The score side states positions only** (no `target_id_column`): the score timeline gains
  three Projections, `projection_1` … `projection_3`, at `1/2`, `3/2`, `1`, each
  `projection_of` naming the performance note it was matched to; the claims name
  `(p1, projection_1)`, `(p2, projection_2)`, `(p3, projection_3)`.
* **Neither side names events**: each timeline gains three Projections standing for no
  event (`projection_of` null), and the claims join them pairwise.
* **Matching again adds nothing.** The library has one projection operation (the bundle's
  `create_match_claims` and the loaders use it too), which reuses the Projection a timeline
  holds for the same event at the same position. Run twice on the same timelines, with the
  score side as positions and with neither side named, the second run's claims name the
  first run's Projections (`projection_1` … `projection_3`, in the same pairs) and both
  timelines hold as many events as after the first run (score 6, performance 3; then
  performance 6, score 6).
* **Refusals**, each leaving both timelines as they were: an id column naming an id the
  timeline does not hold (`KeyError`, `Pair 1: no event with id 'p9' on timeline 'dpt_1'
  or its children (3 events of its own)`); claims asked for without both timelines
  (`ValueError`). With `create_claims=False` no timeline is needed and the matched frame
  is unchanged.
* **Provenance** defaults to `Provenance(method="inferred", note="greedy sequential matching
  by pitch")`.
* **Matching before bundling.** Each timeline in a group of its own, neither side named, so
  each mints `projection_1` … `projection_3` alone; then a bundle takes both groups
  (`add_group`, one per group) and the claims (`add_match_claims`). Nothing refuses the
  shared IDs: the bundle holds both timelines in two groups (`[['dpt_1'], ['clt_1']]`) and
  the three claims `claim_1` … `claim_3`. Each claim resolves its two events by timeline and
  ID: side A `get_event` on `dpt_1` returns the Projection `projection_k` at sample 22050,
  44100, 66150, side B on `clt_1` the Projection `projection_k` at `1/2`, `3/2`, `1`.
  `get_matchstamp_at(44100, "dpt_1")` places `clt_1` at exactly `Coordinate(3/2, quarters)`,
  the anchor of the second claim. A join that refused the shared IDs fails at the first
  `add_group`.

### Matching across spellings

The library's loaders write pitch names with `♯`/`♭` (`Ms3Loader`, `EepNotesLoader`), a TSV
read with plain pandas holds ms3's ASCII names (`Eb5`). `prepare_eep_notes_for_matching` and
`prepare_abc_notes_for_matching` respell the column they compare on, so a caller matching
either kind of frame against the other loses no note with an accidental.

* **Hand-written.** A performance frame with the pitches `Eb5`, `F#4`, `C4` (ASCII, as a
  `.notes` file read with pandas) and an ms3 frame with `E♭5`, `F♯4`, `C4` (as the loader
  writes them), one staff, untied; and the reverse, ASCII on the score side. Each pairing
  matches exactly 3 notes and leaves none unmatched on either side, and the prepared `pitch`
  columns are `["E♭5", "F♯4", "C4"]` on both sides. Without the respelling only `C4` would
  match.
* **A categorical name column.** The ms3 frame's `name` column as a pandas categorical (what
  `read_parquet` gives for a dictionary-encoded column, or `read_csv(dtype="category")`)
  holding `Eb5`, `F#4`, `C4`, against the ASCII performance frame: the prepared `pitch` column
  is `["E♭5", "F♯4", "C4"]`, still categorical with the categories `C4`, `E♭5`, `F♯4`, and the
  match gives exactly 3 matched and none unmatched, as with an `object` column. A respelling
  that wrote new values into the column raises here, since a categorical refuses a value
  outside its categories.
* **The EEP recording against the score** (slow; `StringQuartetEEP_I_Normal` and
  `ABC/n04op18-4_04_unfolded.notes.tsv`, matched on `pitch` and `staff`). Three pairings
  give one count: the `EepNotesLoader` notes against the TSV read with pandas (ASCII), against
  the TSV read by `Ms3Loader` (`♯`/`♭`), and the `.notes` files read with pandas (ASCII)
  against the `Ms3Loader` notes. Each matches exactly 3741 notes, leaving 15 EEP notes and
  10 score notes unmatched, the counts of the two files matched with one spelling on both
  sides, so no pairing loses a note to its spelling.

## SUPRA Integration Tests (`test_supra_integration.py`)

### What We're Validating

The SUPRA (Stanford University Piano Roll Archive) tests validate **partial alignment** using real-world data from piano roll digitization: the musical region of the image is a child of the image at the first hole, and a bundle holding the child reaches the image through the child's exact offset.

### Data Source

| Parameter | Value | Description |
|-----------|-------|-------------|
| Roll | WM 990 | Welte-Mignon red roll, T-100 |
| DRUID | fd660zf8362 | Stanford Digital Repository ID |
| IMAGE_HEIGHT | 299,400 | Full image height in pixels |
| FIRST_HOLE | 15,343 | Pixel row of first musical hole |
| LAST_HOLE | 293,119 | Pixel row of last musical hole |
| MUSICAL_LENGTH | 277,776 | `last_hole - first_hole` |
| MUSICAL_HOLES | 30,092 | Individual hole punches |
| MUSICAL_NOTES | 8,718 | Notes after merging adjacent holes |

### Test Classes

| Class | Tests |
|-------|-------|
| `TestSUPRADataLoading` | `IIIFManifestLoader` dimensions, `ATONLoader` metadata (EXACT values) |
| `TestSUPRATimelineCreation` | Timeline lengths match loader data |
| `TestSUPRAAlignmentBundle` | Partial alignment as a child; the bundle registers the child and its root, and transfers through the offset |
| `TestSUPRAOrderIndependence` | Same alignment specifications produce same results regardless of add order |
| `TestSUPRASummary` | Bundle summary structure and determinism |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_iiif_dimensions_exact` | IIIF loader returns `width=4096, height=299400` (EXACT) |
| `test_aton_metadata_exact` | ATON loader returns EXACT counts from gold standard |
| `test_transfer_holes_to_image` | Holes coord 0 -> Image pixel 15343 (EXACT, no tolerance) |
| `test_transfer_image_to_holes` | Inverse transfer: Image 15343 -> Holes 0 (EXACT) |
| `test_three_timeline_same_partial_alignment` | The holes child grouped with MIDI transfers identically whichever is added first, and reaches the image through the offset |

### Alignment Diagram

```
DGT1 (Full Image: 0 - 299,400 px)
  |
  +-- [15,343 px] --- DGT1_holes (child; Musical Region: 0 - 277,776 px) --- [293,119 px]
                            |
                            | group: both span the same start and end
                            v
                      DLT1 (MIDI: 0 - 871,800 ticks)
```

### ZERO TOLERANCE Policy Compliance

Per the engineering standards:

1. **EXACT COUNTS REQUIRED**: All assertions use exact expected values from the gold standard
2. **NO TOLERANCE**: Boundary coordinates (0, 15343, 293119, 277776) are compared with `==`, not `pytest.approx()`
3. **DOCUMENTED ROOT CAUSE**: Interior point comparisons document why floating-point arithmetic is involved (irrational scale factors)

---

## Integration Tests

### Thoresen PoC Setup (`test_groups.py::TestGroupIntegration`)

```python
def test_thoresen_poc_setup(self):
    """
    DGT1 (2009): 5 equal segments, 4875 pixels total
    DGT2 (2010): 5 varying segments, 4328 pixels total
    Both map to 150 seconds of audio.
    """
```

This test validates that the Group infrastructure can model the Thoresen proof-of-concept discussed in the TISMIR article (https://doi.org/10.5334/tismir.296). It creates two independent groups (DGT1+audio, DGT2+audio) and verifies coordinate conversions match expected values. The `dgt1_timeline`, `dgt2_timeline`, and `audio_timeline` fixtures are provided by `conftest.py`.

**Why exact values**: The pixel counts (4875, 4328) and segment lengths come from the published Thoresen example (constants centralised in `conftest.py`). The test verifies that our implementation produces the same results described there.

### Thoresen Segment Claims (`test_anchors.py::TestClaimIntegration`)

```python
def test_thoresen_segment_claims(self):
    """Creates 5 interval MatchClaims for segment correspondence."""
    # Constants from conftest.py:
    # DGT1_SEGMENT_LENGTH = 967 (5 equal segments)
    # DGT2_SEGMENT_LENGTHS = [866, 867, 867, 864, 864]
```

This test validates that MatchClaims can represent the segment-to-segment correspondence needed for the Thoresen PoC. The `thoresen_segment_claims` fixture (defined in `conftest.py`) builds the claims from the centralised DGT1/DGT2 constants. It verifies:
- All 5 claims are intervals (not instants)
- All claims connect the same timeline pair
- Cumulative offsets are correct (first segment starts at 0, last ends at total length)

---

## MatchGraph Tests (`test_graph.py`)

### What We're Validating

The MatchGraph enforces the design principle that only synchronous claims produce graph edges, while non-synchronous claims are stored as metadata. The `extend_to_groups()` method infers edges through group membership and records each as the graph's own `InferredEdge` (the anchor, the group, the MatchClaim it traces back to), never as a MatchClaim: the extended graph's `claims` are exactly the claims it was given. The belief filter decides which claims a graph holds at all, and `filter()` holds exactly what a graph rebuilt from the claims it keeps holds.

### Local Fixtures (no conftest shadowing)

`test_graph.py` defines its own small group-test timelines with dimensions that
differ from the Thoresen fixtures in `conftest.py` (1000 px / 800 px / 100 s vs.
4835 px / 4328 px / 150 s). To avoid silently shadowing the conftest fixtures of
the same name, these local fixtures carry dimension-explicit names:

| Old (shadowed conftest) | New (local, dimension-explicit) | Used by |
|-------------------------|---------------------------------|---------|
| `dgt1_timeline` | `dgt1_1000px_timeline` | `dgt1_group` |
| `dgt2_timeline` | `dgt2_800px_timeline` | (defined for symmetry; no current user) |
| `audio_timeline` | `audio_100s_timeline` | `dgt1_group` |

Claims are built by three local helpers over the shared `match_claim`
fixture factory (`helpers.py`): `_claim` (a synchronous claim between the
fixture events `"<timeline>@<value>"`), `_nomatch` (one event, the other
side the NOMATCH sentinel) and `_conceptual`
(two section events, no anchors). A MatchGraph reads claims without their
timelines, so the timeline-free tests use dimensionless positions; every claim
touching a timeline of a group states its position on that timeline's own
axis — integer pixels on the graphical timelines (`dgt_1` at `500 pixels`,
`tl1` at `500 pixels`, `a` at `400 pixels`) — so a stamp read back holds
`Coordinate(500, pixels)` for `dgt_1`, exactly the claim's coordinate.

All coordinate assertions in this file are exact `==`: the stamp-coordinate
checks in `test_extend_creates_correct_coordinates` (50.0) and
`test_two_groups_three_inferred_edges` (250.0, 100.0, 200.0) are linear
group conversions at exactly representable ratios and carry no `pytest.approx`.

### Inferred edges

What `extend_to_groups` infers is the graph's own record, `InferredEdge`
(`anchor`, `group_id`, `source_claim`), listed by `inferred_edges` in the
order made and carried by the edge itself (edge attribute `inferred`; a
stated edge carries `claim`). It is never a MatchClaim, takes no id, and
never enters `claims`. `test_extension_infers_edges_not_claims` extends one
claim `dgt_1@500 px ↔ external@25` through the group `{dgt_1 1000 px, audio
100 s}`: the extended graph's `claims` is `[claim]` (`n_claims == 1`), the
original graph's `inferred_edges` stays empty, and the one inferred edge is
exactly `AlignmentAnchor(dgt_1 500 pixels ↔ audio 50.0 seconds)` via
`group1` from the claim object itself, each coordinate on its timeline's own
axis (an `int` pixel, a `float` second); 3 nodes, 2 edges.
`test_inferred_edge_names_its_source_claim` gives the claim the id `claim_1`
so `source_claim.id == "claim_1"` names it (an unset id would pass
vacuously) and pins the repr `InferredEdge(AlignmentAnchor(dgt_1@500 pixels
<-> audio@50 seconds) via group1, from claim_1)`. The two-group and
shared-member tests count the inferred edges (3, and 5 with their groups
`g1, g1, g2, g2, g2`) and assert the claims are still the one given.

### The belief filter

`MatchGraph(claims, believer=..., beliefs=...)` keeps exactly the claims the
belief filter keeps, and builds its nodes and edges from them alone. Five
claims at distinct positions on `tl_a`/`tl_b`, so that each of the four
states occurs: `claim_1` without provenance, `claim_2` believed `false` by
`ana`, `claim_3` believed `true` by `ana`, `claim_4` believed `false` by
`ben`, `claim_5` believed `unlikely` by `ana`. The kept ids, in order, and
`n_edges == len(kept)`, `n_nodes == 2·len(kept)`:

| believer | beliefs | kept |
|---|---|---|
| none | default / `["true"]` | all five (no believer filters nothing) |
| `ana` | default (all but `false`) | `claim_1`, `claim_3`, `claim_4`, `claim_5` (ben's belief is not hers) |
| `ana` | `["true"]` | `claim_3` |
| `ana` | `["undecided"]` | `claim_1`, `claim_4` |
| `ana` | `["unlikely"]` | `claim_5` |
| `ben` | default | `claim_1`, `claim_2`, `claim_3`, `claim_5` |
| `ben` | `["true"]` | none |
| `ben` | `["false"]` | `claim_4` |

A word that is no belief state (`["not_false"]`) is refused with `ValueError`.

### Filtering a graph by its claims

`MatchGraph.filter` takes the filter parameters every MatchClaim query takes,
under the same names and meanings as `AlignmentBundle.get_match_claims`, and
decides them through ONE implementation, `ClaimFilter.select` (the belief
filter asked once over the list, `believed` under the graph's source
bundle's `@context`, then the claim's own kind and timelines):

- `timeline_id`, `timeline_ids`, `id_pattern`: a claim with a matching
  timeline on EITHER side (any of the set, any id the pattern finds); a
  timeline the graph does not know gives no claim.
- `within`, `include_domains`, `include_units`: a claim whose BOTH
  timelines pass. Units and domains are read from the graph's source bundle
  (every timeline it knows), else from the units of the graph's anchors, so
  without a source a claim on a timeline no anchor names never passes them.
- `synchronous_only`: synchronous claims only (conceptual and NOMATCH claims
  excluded); `nomatch_only`: NOMATCH claims only (one event, the sentinel on
  the other side), never a conceptual claim.

The property that proves it is **equality with the graph rebuilt from the
kept claims**: `filter` must hold exactly what `MatchGraph(kept)` holds, and
for an extended graph what `MatchGraph(kept).extend_to_groups(...)` holds
through the same groups, the extension reaching only the timelines `within`
and the domain/unit filters admit (on top of the filters the extension was
made with); `stated_only` is the rebuilt graph not extended. Every test of
this section therefore builds the rebuilt graph by hand and compares,
through one helper `_assert_rebuilt(filtered, rebuilt)`:

1. the claims, as objects and in order (`is`, never ids: free claims may
   share or lack ids);
2. the edge sets (each edge the set of its two nodes);
3. the inferred edges in order, each by anchor, group and `source_claim`
   identity (a kept record naming a dropped claim fails here);
4. the stamps, by their rendered text (`str`), which carries every
   timeline, coordinate and `stated`/`inferred` tag.

A test that only counts nodes and edges would pass a filter that keeps the
right number of the wrong things, so each test also pins the exact claims,
the timelines and, where an extension is involved, the exact inferred
anchors, beside the comparison. The cases are chosen where a filter that
removes edges from the graph it was given (instead of rebuilding) gives a
different answer:

- `test_filter_keeps_the_believed_claims`: the nine belief cases of the
  table above, asked of `MatchGraph(claims).filter(believer=, beliefs=)`:
  the same kept ids in order, `n_edges == len(kept)`,
  `n_nodes == 2·len(kept)`, i.e. exactly the constructor's answer.
- `test_filter_drops_the_inferred_edges_of_a_disbelieved_claim`: in the
  group `{dgt_1 1000 px, audio 100 s}`, `claim_1` `dgt_1@500 px ↔
  external@25` believed `false` by `ana`, `claim_2` `dgt_1@200 px ↔
  external@10` with no belief. The extended graph holds 2 stated and 2
  inferred edges; `filter(believer="ana")` holds `[claim_2]`, 2 edges, 3
  nodes, timelines `{dgt_1, external, audio}`, one inferred edge `dgt_1@200
  pixels ↔ audio@20.0 seconds` via `group1` from `claim_2`, equal to the
  rebuilt graph.
- `test_filter_keeps_an_inferred_edge_another_kept_claim_grounds`: as
  above, but `claim_2` is `dgt_1@500 px ↔ score@7`: the node `dgt_1@500`
  is still stated, so the inferred edge to `audio@50.0` stays, now from
  `claim_2` (the extended graph's record named `claim_1`, the first claim
  naming the node); 2 edges, timelines `{dgt_1, score, audio}`.
- `test_filter_infers_again_an_anchor_a_dropped_claim_stated`: `claim_1` `dgt_1@500 px ↔ audio@50.0 s` (believed `false` by
  `ana`) states exactly the anchor the group gives, so the extended graph
  infers nothing (0 inferred edges, 2 stated). `filter(believer="ana")`
  keeps `[claim_2]` (`dgt_1@500 px ↔ external@25`), and the group infers
  `dgt_1@500 pixels ↔ audio@50.0 seconds` again, via `group1` from
  `claim_2`: 3 nodes, 2 edges, one stamp over `audio`, `dgt_1`, `external`
  — the timeline a node-removing filter loses.
- `test_inferred_edges_name_only_kept_claims`: `claim_3`
  `dgt_1@200 px ↔ external@10` (believed `false` by `ana`) and `claim_4`
  `dgt_1@200 px ↔ score@3`. The extended graph's one inferred edge names
  `claim_3`; filtered for `ana`, the same anchor `dgt_1@200 pixels ↔
  audio@20.0 seconds` names `claim_4`.
- `test_filter_keeps_an_anchor_a_kept_claim_also_states`: two claims state
  the same anchor `tl_a@1 ↔ tl_b@10`, `claim_2` without belief first,
  `claim_1` believed `false` by `ana` last. `filter(believer="ana")` keeps
  `[claim_2]`, 1 edge naming `claim_2` (`claim` the object, `claim_id ==
  "claim_2"`); the graph filtered still names `claim_1` on its own edge.
- `test_filter_by_timeline_id_ids_and_pattern`: claims `claim_1` `tl_a@100
  ↔ tl_b@50`, `claim_2` `tl_b@50 ↔ tl_c@25`. `timeline_ids={"tl_a"}` →
  `[claim_1]` (timelines `{tl_a, tl_b}`, 1 edge); `timeline_ids={"tl_a",
  "tl_b"}` → both claims (timelines `{tl_a, tl_b, tl_c}`, 2 edges, one
  stamp of three timelines); `timeline_id="tl_c"` → `[claim_2]`;
  `timeline_id="tl_x"` (unknown) → no claim, 0 nodes;
  `id_pattern=r"^tl_c$"` → `[claim_2]`; `id_pattern=r"^(?!tl_c$).*"` → both
  claims (each has a side the pattern finds); `timeline_ids={"tl_a",
  "tl_b"}` with `id_pattern=r"^tl_c$"` → `[claim_2]` (AND). Each equal to
  the rebuilt graph.
- `test_filter_within`: the same claims; `within={"tl_a", "tl_b"}` →
  `[claim_1]`, timelines `{tl_a, tl_b}`, 1 edge; `within={"tl_a"}` → no
  claim.
- `test_filter_within_on_an_extended_graph`: `dgt_1@500 px
  ↔ external@25` extended through the group. `within={"dgt_1", "audio"}` →
  no claim, 0 nodes, 0 edges, no inferred edge and no stamp (the node
  pruning kept an inferred edge nobody states); `within={"dgt_1",
  "external"}` → the claim, 2 nodes, 1 edge, no inferred edge, equal to
  `MatchGraph([claim]).extend_to_groups(groups, within={"dgt_1",
  "external"})`; `timeline_ids={"dgt_1", "audio"}` → the claim and the
  whole extended graph (3 nodes, 2 edges, one inferred edge).
- `test_filter_stated_only`: the extended graph of one claim
  `filter(stated_only=True)` → the claim, 1 edge, no `audio`, no inferred
  edge, equal to `MatchGraph([claim])`.
- `test_filter_nomatch_only`: claims `sync` (`tl_a@1 ↔ tl_b@10`),
  `nomatch_1` (`tl_a@2`, sentinel on `tl_b`, believed `false` by `ana`),
  `nomatch_2` (`tl_a@3`, sentinel on `tl_c`) and a conceptual `tl_b ↔ tl_c`
  (two events, no anchor: not a NOMATCH claim).
  `filter(nomatch_only=True)` → `[nomatch_1, nomatch_2]`, 0 nodes, 0 edges;
  with `timeline_ids={"tl_a", "tl_b"}` → `[nomatch_1, nomatch_2]` (both
  involve `tl_a`), with `within={"tl_a", "tl_b"}` → `[nomatch_1]`; with
  `believer="ana"` → `[nomatch_2]`; `filter(synchronous_only=True)` →
  `[sync]`, 1 edge; `filter()` → all four claims; `synchronous_only` with
  `nomatch_only` → `ValueError` `synchronous_only and nomatch_only are
  mutually exclusive`.
- `test_filter_timeline_ids_keeps_either_side`: `sync`
  (`tl_a@1 ↔ tl_b@10`), `nm` (`tl_c`, sentinel `tl_d`), `nm_ab` (`tl_a`,
  sentinel `tl_b`): `timeline_ids={"tl_a"}` → `[sync, nm_ab]`, the
  bundle's answer.
- `test_filter_nomatch_only_on_an_extended_graph`: `dgt_1@500 px ↔
  external@25` and a NOMATCH claim on `dgt_1@100 px` (sentinel on
  `external`), extended: `filter(nomatch_only=True)` → `[nomatch]`, 0
  edges, no inferred edge.

Domain and unit filters (`TestMatchGraphFilterByDomainAndUnit`):

- `test_filter_by_include_domains` / `test_filter_by_include_units`: one
  claim `dgt_1@500 px ↔ audio@50 s`; `include_domains={graphical}` → no
  claim, 0 nodes (the audio side fails); `include_domains={graphical,
  physical}` → the claim; `include_units={seconds}` → no claim;
  `include_units={pixels, seconds}` → the claim.
- `test_domain_and_unit_filters_read_the_source_bundle`: a bundle holding `tl_a`, `tl_b`, `tl_c` (seconds) and `tl_d`
  (pixels), claims `sync` (`tl_a@1 s ↔ tl_b@10 s`) and a NOMATCH `nm_cd`
  (`tl_c`, sentinel `tl_d`): with the bundle as source,
  `include_units={seconds}` → `[sync]`, `include_units={seconds, pixels}`
  → `[sync, nm_cd]`, `nomatch_only=True, include_domains={physical,
  graphical}` → `[nm_cd]`; the same graph without a source and
  `include_units={seconds, pixels}` → `[sync]` (no anchor names `tl_c` or
  `tl_d`, so their units are unknown to it: the documented limit).
- `test_filters_compose_without_a_source`: an extended graph of
  `dgt_1@500 px ↔ external@25` and a NOMATCH claim on `audio` (sentinel on
  `external`; `audio`'s unit known to the graph only through its inferred
  edge), no source. `filter(nomatch_only=True, include_units={seconds,
  number})`, `filter(nomatch_only=True).filter(include_units={seconds,
  number})` and the other order all give `[nm]`: a filtered graph keeps the
  units of the graph it was filtered from, so filters compose in any order.
- `test_canonical_id_filters_combine`: `timeline_ids={"tl_a", "tl_b"}`
  and `id_pattern=r"^tl_[ab]$"` over `tl_a ↔ tl_b`, `tl_b ↔ tl_c` keep both
  claims (each involves `tl_a` or `tl_b`, and each has a side the pattern
  finds); 3 nodes, 2 edges.
- `test_filter_keeps_non_synchronous_claims`: `within={"tl_a", "tl_b"}`
  keeps the synchronous and the NOMATCH claim between `tl_a` and `tl_b` and
  drops the NOMATCH claim on `tl_c`.

### A stamp renders the same in every process

`nx.connected_components` returns **sets**, and the stamp builder used to start
from whichever node came out of one first. Everything after the source was
already sorted, so the visible symptom was narrow and easy to dismiss: the
first entry of `present_timelines` moved between runs while the tail stayed
put, and two runs of the same notebook produced a four-line diff of swapped
stamp columns.

Per-process hash randomisation is the trigger, which is exactly why this
survived a green suite — a single pytest process picks one seed and every
assertion inside it agrees with itself. Determinism cannot be observed from
one run, so the tests are built around comparison:

| Test | Compares |
|---|---|
| `test_stamp_order_survives_claim_insertion_order` | The same claims added in several shuffled orders, asserting byte-identical `repr` and `present_timelines` |
| `test_stamp_order_is_the_source_then_lexical_order` | The concrete order against the retrieval spec: source first, then lexical |
| `test_stamp_order_survives_a_changed_hash_seed` (slow) | Byte-identical output from subprocesses run under different `PYTHONHASHSEED` values |

The hash-seed test is the only one that reproduces the original defect, since
shuffling insertion order does not by itself change set iteration for the same
seed. It is slow-marked because it spawns interpreters, and it asserts equality
between runs rather than against a stored string, so it cannot rot into a
snapshot test of whatever the ordering happens to be.

**Validity Rationale:** asserting that the first element equals one particular
timeline id would pass under the broken code for whichever seed the suite drew
that day. The property is that two runs agree, so the assertion has to be
between two runs. The spec-order test is the companion that pins *which*
deterministic order was chosen, so "stable but wrong" also fails.

### One spec, one map, for both stamps (`TestConversionMapChoice`)

A `TimeStamp` and a `MatchStamp` share one interface, so a `conversion_maps`
spec must choose the same map on both. The claim under test: among several
maps into one unit, `get_unit` converts through the one map the spec
enables, refuses (the timeline's ambiguity `ValueError`) when the spec
enables more than one, and finds none (`KeyError`) when it enables none.

The fixture is ONE timeline `clock` (seconds, length 100) holding two maps
into milliseconds that answer differently at every position, so the value
read names the map that produced it: `clock-ms`, a `TableMap` `[0, 100] s ->
[0, 100000] ms` (25 s → 25000 ms), and `clock-ms-offset`, a `LinearMap`
`1000·x + 5` (25 s → 25005 ms). The `TimeStamp` is `clock.get_timestamp(25.0,
conversion_maps=spec)`; the `MatchStamp` is the bundle's
`get_matchstamp_at(25.0, "clock", conversion_maps=spec)` with `clock` grouped
alone in a bundle. Each case is asserted on both stamps by one
parametrisation, so the two cannot drift apart:

| spec | `get_unit(milliseconds, format="float")` |
|---|---|
| `["clock-ms-offset"]` (selector) | `25005.0` |
| `[<the clock-ms-offset map>]` (the map object) | `25005.0` |
| `["clock-ms"]` | `25000.0` |
| `True` | `ValueError` `Timeline 'clock' has 2 conversion maps targeting 'milliseconds'; name the one you mean. Candidates: clock-ms (seconds_to_milliseconds), clock-ms-offset (seconds_to_milliseconds)` |
| `["clock-ms", "clock-ms-offset"]` | the same `ValueError` |
| `False` | `KeyError` `No eligible conversion to 'milliseconds' on stamp from 'clock'` (TimeStamp) / `... on MatchStamp` (each stamp names itself) |

The text of each stamp (`str()`) is asserted beside the value: for the three
naming cases its lines after the header are exactly the stamp's own row
(`clock 25 seconds`) and ONE conversion row (`milliseconds 25005
milliseconds`, or `25000`), the value `get_unit` returned, so the display
cannot name another map than the getter converts through.

**Validity rationale:** the two maps differ by 5 ms, so a stamp converting
through the wrong map, or through whichever was attached first, fails the
value assertion rather than passing by coincidence; and the ambiguity case
proves that naming one map is what removes the refusal, not a change that
picks a map silently. Before the fix the `MatchStamp` raised the ambiguity
for a spec naming one map.

### Key Test Classes

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestMatchStamp` | 15 | Stamp construction, retrieval, filtering, dictionary round trip, renderings; `filter_by_timelines` always keeps the stamp's source, the filter selecting the others: on a stamp of `tl_a` (the source), `tl_b`, `tl_c`, a filter passing none of them (`within={"zzz"}`, `within=set()`, `id_pattern=r"^zzz$"`) gives the stamp of `tl_a` alone, source `tl_a`, no pair; `within={"tl_b"}` gives `tl_a`, `tl_b` and the pair `(tl_a, tl_b)`; on the same stamp with source `tl_c`, `within={"tl_b"}` gives `tl_b`, `tl_c` (stamp order), source `tl_c`, the pair `(tl_b, tl_c)` |
| `TestMatchGraphBasic` | 8 | Node, edge and claim counts; node and coordinate retrieval; connectivity |
| `TestMatchGraphStamps` | 4 | Instant claim → one stamp, interval claim → two, components → stamps |
| `TestMatchGraphGroupExtension` | 4 | Inferred edges, converted coordinates, `include_inferred=False` |
| `TestMatchGraphBeliefFilter` | 2 (10 cases) | The graph holds exactly the claims the belief filter keeps; a word that is no state is refused |
| `TestMatchGraphFilterByClaims` | 13 (21 cases) | `filter()` holds what the graph rebuilt from the kept claims holds, for every filter parameter (above) |
| `TestMatchGraphSerialization` | 2 | `to_dict()`/`from_dict()` counts, repr |
| `TestMatchGraphThoresenIntegration` | 3 | Thoresen segment claims: 12 nodes, 6 edges, boundary coordinates and stamps |
| `TestMatchGraphNonSynchronousClaims` | 3 | Non-synchronous claims stored as metadata, no edges created |
| `TestMatchGraphGetStamps` | 4 | `get_stamps()` returns one stamp per connected component |
| `TestMatchGraphInferredEdges` | 3 | Inferred edges: the record, its source claim, two-group connectivity; the claims stay the ones given |
| `TestExtendToSeveralGroups` | 1 | A shared member carries group extension into its other group |
| `TestMatchGraphExtendToGroupsFilters` | 6 | `within`, `id_pattern`, `include_domains`, `include_units`; `MatchLine.from_claims` forwards them |
| `TestMatchGraphFilterByDomainAndUnit` | 6 | `filter()` by domain and unit, read from the source bundle; filters composed without a source; `timeline_ids` and `id_pattern` combined; non-synchronous claims `within` |
| `TestMatchStampGroupRetrieval` | 1 | Retrieves group members with the plural key getter and `timeline_ids` |
| `TestMatchStampDeterminism` | 3 | Stamp source and entry order are the same in every process |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_non_synchronous_claims_no_edges` | A NOMATCH claim beside a synchronous one: 2 claims, 2 nodes, 1 edge |
| `test_non_synchronous_claims_accessible_via_claims` | `synchronous_claims` and `non_synchronous_claims` partition `claims`; the NOMATCH claim is the object given |
| `test_get_stamps_one_per_component` | Two disconnected claims yield two stamps of two timelines each |
| `test_extension_infers_edges_not_claims` | The extended graph holds the given claim alone and one `InferredEdge`, exact (above) |
| `test_inferred_edge_names_its_source_claim` | The inferred edge traces back to the originating claim (`claim_1`); exact repr |
| `test_graph_keeps_the_believed_claims` | The belief filter decides the graph's claims, nodes and edges (table above) |
| `test_two_groups_three_inferred_edges` | One claim between TL1∈{TL1,TL4,TL5} and TL2∈{TL2,TL6} connects 5 timelines with exact interpolated coordinates through three inferred edges, `tl1→tl4`, `tl1→tl5`, `tl2→tl6` |
| `test_within_filter` | Only the timelines `within` names appear in the extension |
| `test_include_domains_filter` | Only timelines from specified domains appear |
| `test_filter_keeps_non_synchronous_claims` | `filter(within=)` keeps the non-synchronous claims whose two timelines are in the set |
| `test_group_coordinates` | Returns group-member coordinates in the group's declared order |

### The Two-Groups Connectivity Test

```python
def test_two_groups_three_inferred_edges(self):
    # Group A = {tl1 1000px, tl4 500px, tl5 200px}, Group B = {tl2 800px, tl6 400px}
    # One claim: tl1@500px <-> tl2@400px
    # After extend_to_groups():
    #   - tl4 at 250, tl5 at 100 via Group A from tl1@500
    #   - tl6 at 200 via Group B from tl2@400
    #   - All 5 timelines connected in one component, 3 inferred edges
    stamps = extended.get_stamps()
    assert len(stamps) == 1
    assert stamps[0].n_timelines == 5
```

This is the core group-extension test: it verifies that group extension infers edges with coordinates derived from the typed group retrieval API, producing a fully connected component from a single claim.

---

## MatchLine Tests (`test_matchline.py`)

### What We're Validating

`MatchLine` is an ordered sequence of `MatchStamp` objects for a given source timeline. It is the bridge between `MatchGraph` and `WarpMap`. A MatchLine collects stamps, sorts them by source coordinate, exposes typed `AlignmentAnchor` values, and keeps the float interpolation feed private.

Claims come from the local helpers `_claim` and `_nomatch` over the shared `match_claim` fixture factory: each claim names its fixture events, dimensionless positions where no timeline is involved and integer pixels on the graphical timelines (`dgt_1`, `dgt_2`), so every anchor value read back is the value on its timeline's axis.

### Test Classes

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestMatchLineBasic` | 6 | Construction, sorting, empty/single stamp, filtering of stamps missing source |
| `TestTargetTimelineIds` | 4 | `target_timeline_ids()` returns only timelines with >= 2 stamps |
| `TestGetAlignmentAnchors` | 6 | Extraction of typed alignment anchors, partial stamps, error on self-target |
| `TestFromClaims` | 5 | `from_claims()` with ordering, interval claims, group extension, non-synchronous exclusion |
| `TestFromGraphs` | 6 | `from_graphs()` merging, deduplication, Hendrix M6-M9 pattern |
| `TestMatchLineSerialization` | 4 | `to_dict()`/`from_dict()` round-trip, `__repr__` |
| `TestMatchLineIntegration` | 2 | Thoresen segment claims end-to-end, group extension coordinate pairs |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_stamps_sorted_by_source_coordinate` | Stamps are auto-sorted by source coordinate even if provided out of order |
| `test_stamps_without_source_are_dropped` | Stamps missing the source timeline are silently dropped (with log warning) |
| `test_target_timeline_ids_two_or_more` | `target_timeline_ids()` excludes timelines appearing in < 2 stamps (minimum for interpolation) |
| `test_same_as_source_raises` | `get_alignment_anchors()` raises ValueError when target == source |
| `test_from_claims_with_group_extension` | `from_claims()` with group parameters adds group member coordinates: audio pairs exactly `(0, 0.0)`, `(500, 50.0)`, `(1000, 100.0)`, the source values `int` (pixels) and the audio values `float` (seconds) |
| `test_from_claims_non_synchronous_excluded` | Non-synchronous claims (NOMATCH) do not produce stamps |
| `test_from_graphs_hendrix_pattern` | Four contiguous M-box graphs merged into 5 unique source coordinates (boundary deduplication) |
| `test_from_graphs_keeps_richer_stamp` | Deduplication keeps the stamp with more timelines |
| `test_thoresen_matchline` | Thoresen segment claims give exactly six stamps, target `{dgt_2}` only, and the boundary pairs `(0, 0)`, `(967, 866)`, `(1934, 1733)`, `(2901, 2600)`, `(3868, 3464)`, `(4835, 4328)` |

### The Hendrix Pattern Test

```python
def test_from_graphs_hendrix_pattern(self):
    # 4 contiguous M-boxes, each with 2 boundary claims
    # Boundaries: 0, 100, 200, 300, 400
    # Adjacent M-boxes share boundary coordinates -> deduplicated
    graphs = [MatchGraph([...]) for i in range(4)]
    line = MatchLine.from_graphs(graphs, source_timeline_id="score")
    assert line.n_stamps == 5  # 5 unique coordinates
    assert line.source_coordinates == [0.0, 100.0, 200.0, 300.0, 400.0]
```

This validates the Hendrix M6-M9 use case from the conceptual model: multiple MatchGraphs representing contiguous subsections can be merged into a single ordered MatchLine for WarpMap generation.

---

## A Timeline in Several Groups (`test_bundle.py::TestSeveralGroups`, `test_bundle_dict.py`, `test_graph.py::TestExtendToSeveralGroups`)

### What We're Validating

Group membership is a directed acyclic graph, not a partition: one timeline
object may be a member of several groups of a bundle. The groups are the only
statement of membership; `AlignmentBundle.timeline_to_groups` is read from
them, so no second record can disagree with them. Every consumer of
membership — transfer, the matchstamp closure, commensurability, group
extension of a `MatchGraph`, the grouped forms of `MatchStamp.to_dict`, the
diagram and the manifest — must treat a shared member as a member of each of
its groups.

### Specimen

Four quarters timelines with exact lengths, so every expected value is a
ratio: `t` (12), `a` (8), `u` (6), `b` (9). Group `g1` holds `t`, `a`, `u`;
group `g2` holds `t`, `u`, `b`. `t` and `u` share both groups; `a` lies in
`g1` only and `b` in `g2` only. The query position on `u` and on `a` is
`1/3`, chosen so that every converted coordinate is a non-integer ratio and a
float detour would show.

### What proves it

1. **One membership, read from the groups.** `timeline_to_groups` is exactly
   `{"t": ("g1", "g2"), "a": ("g1",), "u": ("g1", "g2"), "b": ("g2",)}`
   (group ids sorted); `get_groups_for_timeline("t")` returns the two group
   objects `g1`, `g2` in that order; the singular `get_group_for_timeline`
   answers `g1` for `a` and refuses `t` with the exact message
   `Timeline 't' is a member of several groups ['g1', 'g2']; use
   get_groups_for_timeline` rather than picking one. Removing `t` from `g1`
   (`TimelineGroup.remove_timeline`) leaves `("g2",)`, and from `g2` too
   leaves `t` out of the lookup while it stays registered — and the diagram
   then lists it as standalone. No stale record is possible because there is
   no record to go stale; these assertions are what would catch one.
2. **Within a group, any shared group answers alike, and the first by id is
   used.** From `u` at `1/3`, the matchstamp holds `t` at exactly `2/3`, `a`
   at `4/9` and `b` at `1/2`; the `t` value equals `g1.get_coordinate_at`
   from `u` at `1/3` alone, and `transfer` returns its float, `2/3` as a
   float. Two co-extensive groups cannot disagree, so the choice of group is
   unobservable through results; it is pinned on the private lookup instead:
   the group `u` and `t` share is `g1`, also when `g2` was added to the
   bundle first.
3. **A shared member carries a position from one group to the other.** From
   `a` at `1/3` the matchstamp holds `t` at `1/2`, `u` at `1/4` and `b` at
   `3/8` (`1/3 · 9/8`, through `t` or `u`, both exact); `transfer` from `a`
   to `b` returns `0.375`; `are_commensurable("a", "b")` is true. The control
   is the same topology without a shared member (`g1` = `a`, `x`; `g2` =
   `b`, `y`): the stamp from `a` holds only `a` and `x`, `transfer` returns
   `None` and `are_commensurable` is false — so it is the shared member, not
   group membership as such, that connects the groups.
4. **`grouped_with` never guesses.** Grouping a newcomer with `t` raises
   `ValueError` with exactly `Cannot group with 't': it is a member of several
   groups ['g1', 'g2'], so the group to join is ambiguous` and leaves the
   bundle's manifest and timelines unchanged; grouping it with `a` joins `g1`
   only. A grouped timeline joins a further group: `a` with
   `as_group="solo"` becomes `("g1", "group_1")`. A member is still refused
   a second place in one group: grouping `t` with `a` (both in `g1`) raises
   `Timeline 't' already in group 'g1'`.
5. **Grouped stamp forms list a shared member under each group.**
   `MatchStamp.to_dict(format="nested")` of the stamp from `u` has exactly
   the keys `g1` and `g2`, with `t (quarters)` and `u (quarters)` under both,
   `a (quarters)` under `g1` only and `b (quarters)` under `g2` only;
   `format="prefix"` writes `g1/t (quarters)` and `g2/t (quarters)`.
6. **The diagram lists a shared member under each group** (its rendered
   name occurs once per group) and lists no standalone section while every
   timeline is grouped. `summary()` writes each timeline's `"groups"`, the
   sorted list of its group ids: `["g1", "g2"]` for `t` and `u`, `["g1"]`
   for `a`, `["g2"]` for `b`, and `[]` for a standalone `s`; no singular
   `"group"` key remains.
7. **Manifest round trip** (`test_bundle_dict.py`). The bundle writes the
   groups `g1` with members `["a", "t", "u"]` and `g2` with
   `["b", "t", "u"]` (members are written sorted), their orders
   `["t", "a", "u"]` and `["t", "u", "b"]` under `graphical.order`, and
   every root once; read back, complete and `minimal=True` alike, it writes
   the identical dictionary, answers every group's cross-sections identically,
   has the same `timeline_to_groups`, and holds ONE object for `t`: the
   bundle's `t`, `g1`'s `t` and `g2`'s `t` are the same object, so the
   reader restores a shared member, not two copies of it.
8. **Group extension of a `MatchGraph` follows a shared member**
   (`test_graph.py`). The same topology in pixels (`t` 1200, `a` 800, `u`
   600, `b` 900) and one claim `a@400 ↔ e@100` (`a` at 400 integer
   pixels, its axis; `e` in no group),
   extended through `{g1, g2}`, yield one stamp over exactly `a`, `e`, `t`,
   `u`, `b`, with `t` at `600.0`, `u` at `300.0` and `b` at `450.0` — `b` is
   only reachable by extending the coordinates `g1` gave `t` and `u` through
   their second group — and exactly five inferred edges (`a`–`t`, `a`–`u`
   through `g1`; `t`–`u`, `t`–`b` and `u`–`b` through `g2`). A node reached
   through a group is not extended through that same group again, so the
   single-group expectations of the existing extension tests (node, edge and
   inferred-edge counts) are unchanged.

---

## AlignmentBundle Tests (`test_bundle.py`)

### What We're Validating

An `AlignmentBundle` holds its timelines, its groups and ONE MatchClaim table
(`match_claim_table`, a `MatchClaimField`). Every reader of the claims —
`get_match_claims`, the matchstamp getters, `transfer`, `are_commensurable`,
`get_matchstamp_table` — reads the rows of that table the belief filter keeps.
Cross-group transfer runs MatchClaim → MatchLine → WarpMap; the bundle builds
these lazily on first use and caches them until the table changes.

### Strategy

- **Claims on the timelines' own axes.** Every fixture claim names an event on
  each side (the fixture events `"<timeline>@<value>"` of
  `helpers.match_claim`) and states each coordinate in its timeline's unit and
  declared number type. The plain `Timeline` fixtures count seconds as floats,
  so `_make_linear_claims` states float seconds: score 0, 50, 100, 150, 200
  matches audio 0, 25, 50, 75, 100 (`audio = score · 0.5`). The bundle
  converts nothing when it reads, so every expected value is a claimed value
  or a linear interpolation between two claimed values, derived by hand.
- **Entry is checked, and atomic.** A claim naming a timeline the bundle does
  not hold raises `KeyError`; a coordinate off its timeline's axis — another
  unit (quarters on a seconds timeline) or another number type (a Fraction of
  seconds on a float axis) — raises `ValueError` naming the row and both axes.
  A valid claim given before the refused one in the same call is not added
  either: the table stays empty. A position in another unit enters only
  through `create_match_claims`, converted through the timeline's C-Map when
  the claim is made; the test pins the converted value exactly.
- **`create_match_claims` checks every pair before it adds a Projection.**
  A call whose first pair projects a position and whose second names an
  event id no timeline holds (`KeyError`), or states a dict with no `start`
  (`ValueError`), leaves the timelines without a Projection and the table
  empty: the evidence is the number of events, not the message alone.
- **A second call reuses every Projection.** The same pairs made twice — the
  held onset `o1` (audio 5.0) against score 10.0, and audio 20.0 against
  score 40.0 — name the same Projections both times (score `projection_1`
  standing for `o1`, score `projection_2` and audio `projection_3` standing
  for no event: a Projection's ID is unique in the bundle, and the score's
  are added first, the first pair giving it one), and the audio and score
  timelines hold as many events after
  the second call as after the first (2 each); the claims are new
  statements (`claim_3`, `claim_4`).
- **Equality includes the claims.** Two bundles built alike are equal; one
  claim more in one of them, or one claim's provenance differing (a belief
  added), makes them unequal; querying one of them (its caches filled)
  changes nothing.
- **One table.** Claims added one by one and the rows of a `MatchClaimField`
  join the same table: a claim read back equals the one added (another
  object, recording the bundle), every id is kept, a claim without one is
  minted `claim_N` above the highest id held.
- **The belief filter decides what is read.** One claim per belief state
  (believed true, believed false, no belief), with the false claim at a value
  (audio 60) the other two would never give (they interpolate to 50): which
  value a stamp holds shows which claims were read.
- **Actors compare as IRIs under the bundle's `@context`.** The same three
  claims with ana named by her full IRI `https://example.org/people/ana`,
  in a bundle declaring the prefix `people` for `https://example.org/people/`:
  a believer given compactly (`people:ana`) must select exactly what the full
  IRI selects, and a belief recorded compactly must be found by the full
  IRI, through every consumer of the filter — the table's one point
  (`get_match_claims`, the matchstamp getters), the list form given claims
  (`get_matchstamps(claims=...)`), `ClaimFilter.matches_claim` (a claim
  read from the bundle; a free copy has no bundle, hence no prefix) and a
  `MatchGraph` built with the bundle as its `source`. Before the prefix is
  declared `people:ana` names nobody and keeps all three; declaring it
  afterwards changes the answer, which proves the filtered table is not
  served from a cache made under the old context.
- **Private state only where nothing observable exists.** The cache tests
  count the work the cache saves (calls of `WarpMap.from_match_line`,
  through `monkeypatch`) and never read the cache; the ambiguity test places
  a MatchLine in the bundle's MatchLine cache, because MatchClaims cannot
  state the ambiguous line it needs.
- **Counts and derivations, never ranges or clocks.** A scale claim is
  proved by counted work: `MatchClaimField.from_columns` over 100,000 rows
  constructs no `MatchClaim` and no `AlignmentAnchor`, with no wall-clock
  threshold. A count proves this only if every way of building a pydantic
  model passes the counter, and `__init__` is not such a point:
  `model_validate` (behind both classes' `from_dict`) and `model_construct`
  build an instance without calling it. Every validating construction
  (`__init__`, `model_validate`, `model_validate_json`, `model_validate_strings`)
  runs the class's `__pydantic_validator__`, so the test replaces that
  validator on both classes by one counting each call, and counts
  `model_construct`, the one path that validates nothing. Two controls prove
  the counter sees what it guards: each of those five paths, called once on
  `AlignmentAnchor`, is counted once, and `MatchClaim.from_dict` of an
  instant MatchClaim counts its one `AlignmentAnchor` and one `MatchClaim`; and on the field the count stays empty through
  `from_columns` and `len`, then reading one row back (`field[50_000]`, the
  library's own row-to-object path) counts exactly one `AlignmentAnchor` and
  one `MatchClaim`. Copying an existing claim (`model_copy`, `copy`) builds
  nothing from a row and is not counted. Thoresen's
  H′ is `967 + 378/867 · 967 = 1203915/867` and `967 + 517/867 · 967 =
  1338328/867` (segment 2 of DGT1 spans `[967, 1934)`), computed in
  `Fraction`; its WarpMap transfer is the nearest pixel, 1389. A match file
  written back from the Chopin p01 context holds the original's eight
  `info(...)` lines exactly; the rich export of a context stating every
  field writes the format's version line, the five names and the two clock
  lines, in that order.

### Key Evidence

| Class | Tests | Validates |
|-------|-------|-----------|
| `TestCrossGroupTransfer` | 12 | score 100 → audio 50, 0 → 0, 200 → 100; 75 → 37.5 (25 + (75 − 50)/(100 − 50) · 25); audio 50 → score 100; image 200 → audio 50 (200/400 · 200 = 100 within the score group, then 100 · 0.5); score 100 → midi 50 through the recording group; no claims → `None`; a NOMATCH claim (no anchors) → `None`; one claim resolves its anchor (100 → 50) and nothing between (75 → `None`); `transfer_interval(50, 150)` → `(25, 75)`; a MatchLine whose source 50 has two targets makes `transfer` raise `ValueError` naming the source timeline and `get_matchstamp_at()` |
| `TestGetMatchstampAtCrossGroup` | 7 | the stamp at score 100 holds score 100 and image 200 (seconds) in the source group and audio 50 in the target group; at 75 its inferred edges are exactly `(score, image)`, `(score, audio)`, `(audio, midi)`; a graph the bundle caches gives score 50 / audio 25 with the timelines' units; nested format keys `group_1`, `group_2`; prefix keys exactly `group_1/image (seconds)` and `group_1/score (seconds)` (the stamp holds every member of the source group); without claims the source group only |
| `TestCommensurabilityWithClaims` | 4 | claims make score–audio and image–audio commensurable; without claims, or with a NOMATCH claim only, they are not |
| `TestCacheInvalidation` | 4 | the first transfer score → audio builds exactly one WarpMap (target `audio`) and answers 25; a second transfer answers 25 and builds none; after a new claim the next transfer builds one more; the rebuilt map gives score 100 → audio 50 and reaches the new anchor, 125 → 62.5 (50 + 25/50 · 25) |
| `TestAddMatchClaims` | 6 | chaining; three claims are three rows, `claim_1`–`claim_3`, read back equal to the claims added and recording the bundle; a fourth call adds one row minted `claim_4`; an unknown timeline (`['ghost'] that this bundle does not hold`) and an off-axis coordinate (`Row 1 start_anchor.coordinate_b is in quarters (fraction)` / `in seconds (fraction)`, `but timeline 'audio' counts in seconds (float)`) are refused with nothing added |
| `TestCreateMatchClaims` | 13 | an event ID names an event of the timeline or of a descendant, and one ID held twice there names none (an event ID is unique on its own timeline only): on `score` (quarters, 8) with children `c` at 0 and `d` at 4 (length 4 each), `c` holding `n1` at 1, `d` holding `n1` at 1 and `n3` at 2, the pair `n3` ↔ audio `o1` makes one claim at score 6 (the descendant's event, at `score`'s coordinate); the pair `n1` ↔ `o1`, after a projected pair in the same call, is refused (`ValueError`, `The event ID 'n1' is held by 2 events of descendants of timeline 'score' (on 'c', 'd') and not by the timeline itself, so it names no one event: name the timeline holding the event instead.`) and nothing is added (no claim, no Projection), and the advice works: `n1` on `c` resolves to `c`'s event at 1, on `d` to `d`'s at 1; the timeline's own event answers for its ID: with `n3` added to `score` itself (at 3), `n3` on `score` resolves to `score`'s own event at 3 (not `d`'s at 6), `n3` on `d` still to `d`'s at 2, and `n1` added to `score` (at 7) makes `n1` on `score` resolve at 7 where it was refused before; one level deeper (`score` (quarters, 8) ⊃ `c` (offset 2, length 4) ⊃ `g` (offset 1 in `c`, length 2)), where `c` reports its own events beside a descendant's: with `score` holding `n1` at 7, `c` `n1` at 1 and `g` `x9` at 1, `n1` on `score` resolves to `score`'s own event at 7 (a holder read from a null `source_timeline` alone would take `c`'s row for `score`'s and refuse, naming `'None'`); with `c` and `g` each holding `n1` at 1 and `score` none, `n1` on `score` is refused (`The event ID 'n1' is held by 2 events of descendants of timeline 'score' (on 'c', 'g') and not by the timeline itself, so it names no one event: name the timeline holding the event instead.`), nothing added, never resolved to `c`'s; a second call with the same pairs names the same Projections and adds no event (audio `o1` + `projection_3`, score `projection_1` of `o1` + `projection_2`, before and after: a Projection's ID is unique in the bundle), its claims `claim_3`, `claim_4`; a pair naming an unknown event id (`Pair 1: no event with id 'nope' on timeline 'audio' or its children (0 events of its own)`) and a dict without `start` (`Pair 1: the side on timeline 'audio' names no event of it and states no 'start': ...`) after a projected pair add nothing; the provenance given lands on the claim made and on the claim read back; `agent`, `agent_identifier` and `decision_criteria` are refused (`TypeError`), nothing added; a seconds position projected onto a samples timeline (C-Map samples → seconds at 1/44100): 2.0 s is sample 88200, so the claim between the note `o1` (2.5 s) and `projection_1` stores `Coordinate(88200, samples)` with an `int` value, read back equal, and the Projection's `projection_of` is `{"timeline_id": "performance", "event_id": "o1"}` |
| `TestBeliefFilter` | 2 | `claim_1` (0 → 0, ana true), `claim_2` (100 → 60, ana false), `claim_3` (200 → 100, no belief). `get_match_claims`: no believer → all three; `believer="ana"` → `claim_1`, `claim_3`; `beliefs=["true"]` → `claim_1`; `["undecided"]` → `claim_3`; `["true", "false"]` → `claim_1`, `claim_2`. `get_matchstamp_at(100, "score")`: no believer → `claim_2`'s anchor, audio and midi 60.0, not interpolated; ana → interpolated between `claim_1` and `claim_3`, audio and midi 50.0; ana, `["true"]` → one claim cannot interpolate: score 100 and image 200 alone |
| `TestBeliefFilterIris` | 6 | the three claims with ana's full IRI, prefix `people` declared: `believer="people:ana"` and the full IRI both → `claim_1`, `claim_3`, with `["true"]` → `claim_1`; beliefs recorded as `people:ana`, believer the full IRI → `claim_1`, `claim_3`; `get_matchstamp_at(100, "score", believer="people:ana")` interpolated, audio and midi 50.0; `get_matchstamps(claims=all three, believer="people:ana")` → two stamps, at score 0.0 and 200.0; `ClaimFilter(believer="people:ana").matches_claim` → `[True, False, True]` on the bundle's claims, `True` on a free copy of `claim_2`; `MatchGraph(all three, believer="people:ana", source=bundle)` holds `claim_1`, `claim_3`; before the declaration `people:ana` keeps all three, after it `claim_1`, `claim_3` |
| `TestAddMatchClaimField` | 6 | chaining; the five rows join the table, the bundle minting `claim_1`–`claim_5`, equal to the field's claims but for the ids, the field left as given (ids `None`); two `from_columns` fields (two loaders' fields) join one bundle as `claim_1`–`claim_5` and `claim_6`–`claim_10`; score 100 → audio 50 exactly; score 123.456 interpolates to audio and midi 61.728 (50 + 23.456 / 2) and image 246.912, flagged interpolated; a claim added after the field is minted `claim_6`, read as the sixth, and score 175 answers audio 87.5 exactly |
| `TestBundleEquality` | 3 | two bundles built alike over the same timelines are equal, also after one of them was queried (`transfer` 75 → 37.5 fills its caches); one claim more, or a belief added to one claim's provenance, makes them unequal |
| `TestBundleClaimFilters` | 7 | (`score` quarters as Fractions, `audio` and `other` seconds as floats; one claim at the origins per pair) `timeline_id="score"` → the first two; `timeline_ids={"audio"}` and `id_pattern="^audio$"` → the first and third; `between=("score", "audio")` → the first; `include_units={seconds}` → the third only (both its timelines count seconds), `{pixels}` → none; `within={"score", "audio"}` → the first alone (both its timelines in the set; `timeline_ids` of the same set keeps all three, each has a side in it), on `get_match_claims` and `get_claim_field` alike, and `within` meets `include_units={seconds}` (`within={"score", "audio"}` with it → none, `within={"audio", "other"}` with it → the third); a stamp filtered with `within={"score", "audio"}` holds exactly score `0` (quarters, Fraction) and audio `0.0` (seconds), and the stamp getters take no `timeline_ids` (`TypeError`): one name, one meaning. **The queried timeline is always kept; the narrowing selects the others** (`within`, `id_pattern`, `include_domains`, `include_units` alike): a narrowing passing none of the others (`within={"zzz"}`, `within=set()`, `include_units={pixels}`) gives the stamp of `score` alone (score `0`, quarters, Fraction; source `score`; no pair) on `get_matchstamp_at`, on both stamps of `get_matchstamps_at([0.0, 0.0])` and on the dispatcher `get_matchstamp`. On a warped fixture (`score` quarters, `audio` and `other` seconds, each its own group; claims `score` 0 ↔ 0.0 and 4 ↔ 2.0 on `audio`, 0 ↔ 0.0 and 4 ↔ 4.0 on `other`), `get_matchstamp_at(1, "score", within={"audio"})` has source `score` and holds exactly score `1` (Fraction) and audio `0.5`; its Interval form, `Interval(0, 1)` in quarters, is a `MatchIntervalStamp` with source `score`, the query's interval, and endpoint stamps holding exactly score `0`, audio `0.0` and score `1`, audio `0.5` (the form that raised `KeyError` when `within` dropped the queried timeline). A table's `within` stays a projection of its columns: `get_matchstamp_table([0, 1], "score", within={"audio"})` has the one column `audio`, its dataframe `audio (seconds)` `[0.0, 0.5]` |
| `TestAddGroup` | 3 | a pre-built group with its timelines; a duplicate group id refused; cross-group transfer through added groups (100 → 50) |
| `TestSeveralGroups` | 11 | a timeline in several groups (see "A Timeline in Several Groups") |
| Creation, timelines, groups, within-group transfer, a part as a child, order independence, commensurability, chaining | 32 | ids and names, registration and lookup, `grouped_with`/`as_group`, a child joining with its root (section 0 s = reference 25 s, section 100 s = reference 125 s), linear transfer within a group (tl1 50 → tl2 100), exact ratios within a group (1/3 of 3 quarters → 7/9 of 7), results independent of the order timelines are added |

### Lazy WarpMap cache

The bundle caches MatchLines per (belief filter, source timeline) and
WarpMaps per (belief filter, source timeline, target timeline), negative
results included. Every change of the MatchClaim table clears both, with
the MatchGraphs and the belief-filtered views of the table.

### Unexpected errors propagate (`TestUnexpectedErrorsPropagate`)

Two places read a failure of what they call as a negative answer, and may
only do so for the failures that mean it:

- a MatchLine is built from the MatchClaims naming its source, which were
  checked when they joined the bundle, so building it has no failure that
  means "no line" (a source no MatchClaim names has none, and that is decided
  before building). A `MatchLine.from_claims` replaced by one raising
  `RuntimeError("boom")` makes `transfer(50.0, "score", "audio")` raise it
  from `_get_or_build_match_line`;
  with the builder restored the same call answers `25.0`, so the failure was
  not cached as "no line" (the cache itself is private and not read).
- a conversion column of `get_matchstamp_table` holds a null where the map
  does not reach the row (its `ValueError`: outside its domain, or a value
  the target unit refuses). A `TableMap._evaluate` replaced by one raising
  `RuntimeError("boom")` makes `get_matchstamp_table([25.0], "clock",
  conversion_maps=True)` raise it from `_assemble_matchstamp_table` (the
  traceback names the frame, so the error is not one raised elsewhere).
  The documented case keeps its null: a seconds→milliseconds `TableMap`
  over `[0, 50]` with `extrapolate="error"` on a 100-second `clock` gives
  the rows `25.0` and `75.0` the cells `25000.0` and null.

The other six catches of the bundle's transfer paths take the same
narrowing (`ValueError`, and `KeyError` where a group conversion names a
member). Each is pinned twice: a callee replaced by one raising
`RuntimeError("boom")` makes the public call raise it, from the frame and
the statement of that catch (the traceback entry's statement is compared,
so the error cannot have come through a neighbouring catch); and the
documented error still gives the documented answer, with exact values that
differ from the answer without the error, so the test sees which way the
call went. The fixtures are the two groups of the cross-group tests
(`score` 200 and `image` 400 in one group, `audio` and `midi` 100 in the
other), with the claims named per case (`a ↔ b ×r`: `a` at 0, 50, 100
matches `b` at 0, 50·r, 100·r), and a third member where a case needs a
next one to fall back on (`image2` 400 beside `score`, `audio2` 100 beside
`audio`):

- `transfer`, the direct WarpMap (`score ↔ audio ×0.5`): `boom` from
  `WarpMap._interpolate_float` propagates from
  `return warp._interpolate_float(coord)`; a `ValueError` there answers
  `None` (no other path is tried), where `transfer(60.0, "score",
  "audio")` answers `30.0` without it.
- `transfer`, through a member of the source's group (`image ↔ audio
  ×0.5`, `image2 ↔ audio ×0.2`, so `score` has no WarpMap of its own):
  `boom` from `_interpolate_float` propagates from
  `return warp._interpolate_float(intermediate.value)`; the group refusing
  the conversion to `image` (`KeyError` and `ValueError` each) skips it, so
  `transfer(50.0, "score", "audio")` answers `20.0` through `image2`
  (`score` 50 → `image2` 100 → `audio` 20) where it answers `50.0` through
  `image` without it.
- `transfer`, through a member of the target's group (`image ↔ audio
  ×0.5`, `image ↔ audio2 ×0.2`; `transfer(50.0, "image", "midi")`, no
  WarpMap names `midi` or starts at `score`/`image2`): `boom` from
  `_interpolate_float` propagates from
  `warped = warp._interpolate_float(coord)`; the group refusing the
  conversion from `audio` (`KeyError` and `ValueError` each) skips it, so
  the answer is `10.0` through `audio2` where it is `25.0` through `audio`.
- the matchstamp closure's group conversion (`_get_group_timestamp`, no
  claims): `boom` from `TimelineGroup.get_timestamp_at` propagates from
  `ts = group.get_timestamp_at(`; a position outside the member (`300.0`
  on `score`, length 200: the group's own `ValueError`) and a `KeyError`
  from the group at `50.0` each give the stamp of `score` alone
  (`{"score": 300.0}`, `{"score": 50.0}`), where `50.0` reaches
  `image` `100.0` without the error.
- the closure's WarpMap into another group, directly (`_transfer_to_group`;
  `score ↔ audio ×0.5`, `score ↔ midi ×0.2`; `get_matchstamp_at(60.0,
  "score")`, not on an anchor): `boom` from
  `AlignmentBundle._transfer_coordinate` propagates from the statement
  passing `coordinate`; a `ValueError` for `audio` skips it, so `midi` is
  reached by its own WarpMap (`12.0`) and `audio` through the group
  (`12.0`), where both are `30.0` without it.
- the same, through a member of the source's group (`image ↔ audio ×0.5`,
  `image ↔ midi ×0.2`; `get_matchstamp_at(40.0, "score")`, `score` 40 is
  `image` 80, inside the claims' span `[0, 100]` on `image`, so no support
  policy decides): `boom` from `_transfer_coordinate` propagates from the
  statement passing `float(intermediate.value)`; a `ValueError` for
  `audio` skips it, so `midi` and `audio` read `16.0`, where both are
  `40.0` without it.

### `get_matchstamp_table` Conversion Columns

`get_matchstamp_table` accepts a keyword-only `conversion_maps` spec
and represents every timeline column as an `IdCoordinateField`. Numeric
conversion columns are `CoordinateField` values with the converted unit and
canonical number-type metadata; tests materialize their exact scalar values
through those semantic fields rather than comparing raw Arrow floats.
(default `False`, the opt-in default shared by every
matchstamp getter). When given, `_assemble_matchstamp_table` adds one
derived column per (timeline, enabled unit-conversion map) after the
timeline columns already in the table — **numeric unit conversions only**
(a map's `target_unit` must be set); label/structured maps surface in
`MatchStamp` display but never become table columns. A derived column's
name is the map's target-unit name, qualified `"<timeline_id>:<unit>"`
when two timelines would otherwise produce the same unit name — the same
collision rule `TimeStamp`/`MatchStamp` display uses.

Fixture: a single `clock` timeline (length 100 seconds, `uid="clock"`,
`as_group="clock-group"`) carrying the two exact-value `TableMap`s from
`test_stamp_interface.py` — seconds→milliseconds (`[0,100]->[0,100000]`)
and seconds→frames (`[0,100]->[0,5000]`).

| Test | Validates |
|------|-----------|
| `test_matchstamp_table_adds_conversion_columns` | `get_matchstamp_table(coordinates=[25.0], timeline_id="clock", conversion_maps=True)` returns columns `{"clock", "milliseconds", "frames"}`, one row, with exact values `clock=25.0`, `milliseconds=25000.0`, `frames=1250.0` |
| `test_matchstamp_table_no_conversion_columns_by_default` | Omitting `conversion_maps` yields `column_names == ["clock"]` — no derived columns |
| `test_matchstamp_table_conversion_column_collision_qualified` | Two timelines in different groups, each with its own seconds→milliseconds map, linked by one cross-group synchronous claim at 25 seconds: the table has one row (the claim's one anchor) and qualifies both derived columns as `"<timeline_id>:milliseconds"`, each `25000.0`, rather than colliding on a bare `"milliseconds"` |

### Coordinate-Type Parity (raw / Coordinate / IdCoordinate)

Every coordinate-accepting query method must accept a raw `int`/`float`/
`Fraction`, a `Coordinate`, or an `IdCoordinate` and resolve all three to the
same result when their units are native to the receiving timeline. A
unit-qualified coordinate is resolved through that timeline before graph or
group lookup: a C-Map may convert an afforded unit, while an unafforded unit
raises `ValueError`. For methods that pair a coordinate with the id of the
timeline it lives on (`get_matchstamp_at`, `get_timestamp_at`), an
`IdCoordinate` supplies that id, so the `timeline_id` argument becomes
optional; passing a non-Id coordinate without an explicit `timeline_id` is an
error. An explicit ID and an `IdCoordinate` ID must agree, including for the
named source endpoints of `transfer` and `transfer_interval`.

These tests reuse the `_make_cross_group_bundle()` + `_make_linear_claims()`
fixtures (no corpus). The claims map `score → audio` linearly
(`audio = score · 0.5`), so a query at score 100 resolves to audio 50 — an
exact value with no interpolation rounding. Parity is asserted by comparing
the whole result across all three call forms, plus a pinned concrete value.

| Test class | Validates |
|------------|-----------|
| `TestGetMatchstampAtCoordinateParity` | raw / native-unit Coordinate / IdCoordinate-alone agree (score 100, audio 50, image 200, midi 50); a millisecond query through the source's seconds → milliseconds C-Map (2500 ms) gives the stamp of the native 2.5 s; unavailable units and conflicting IDs ⇒ `ValueError`; non-coordinate type ⇒ `TypeError` |
| `TestGetMatchstampRenderingCoordinateParity` | the same three forms produce identical whole `to_dict(format="flat")` results; error paths identical |
| `TestTransferCoordinateParity` | `transfer`/`transfer_interval` resolve native-unit Coordinate/IdCoordinate inputs through the named source timeline; non-coordinate ⇒ `TypeError` |

The same parity is validated for `TimelineGroup.get_coordinate_at` in
`test_groups.py::TestCoordinateRetrieval` (`test_convert_accepts_coordinate_objects`,
`test_convert_rejects_unsupported_type`): 75 seconds on `audio` resolves to the
exact pixel `2438` on `dgt_1` for all three coordinate forms.

---

## WarpMap Tests (`test_warpmap.py`)

### What We're Validating

`WarpMap` is a standalone class that materialises warped timeline copies from alignment data. It wraps an `InterpolationMap` internally for O(log n) coordinate conversion and bridges the gap between `MatchLine` and `AlignmentBundle`. Public coordinate retrieval is bounded by the source-anchor support; bundle support-policy handling uses the private interpolation lane for explicit clamp or extrapolation behavior.

### Test Classes

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestWarpMapConstruction` | 5 | Basic init, `from_match_line()`, `from_coordinate_pairs()`, rejection of <2 points |
| `TestForwardInverse` | 5 | Linear mapping, identity, bounded-support errors, inverse round-trip |
| `TestMaterialise` | 7 | Event warping (instant/interval), child warping, region warping, event count, empty timeline |
| `TestMaterialiseTypeConversion` | 3 | CLT→CPT type conversion, unit propagation, region unit conversion |
| `TestSerialization` | 4 | `to_dict()`/`from_dict()` round-trip, repr |
| `TestMultiTarget` | 1 | Different WarpMaps from same MatchLine for different targets |
| `TestIntegrationWithClaims` | 1 | End-to-end: MatchClaim (fixture events, `match_claim`) → MatchLine → WarpMap |
| `TestEdgeCases` | 10 | Non-linear warping, single-point rejection, degenerate intervals, large datasets, overlapping regions |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_forward_inverse_roundtrip` | `inverse(forward(x)) ≈ x` for all x in domain |
| `test_materialise_warps_instant_events` | Instant event coordinates warped correctly (reads `start` struct dict, not `instant`) |
| `test_materialise_warps_interval_events` | Start/end/duration all warped; duration uses `forward(start+dur) - forward(start)` for non-linear correctness |
| `test_materialise_warps_children` | Child offsets converted, child count preserved |
| `test_materialise_warps_regions` | Region boundaries converted, region names preserved |
| `test_type_conversion_clt_to_cpt` | Source CLT (quarters) → target CPT (seconds): correct type and unit |
| `test_full_pipeline` | Three claims `score`↔`audio` at (0, 0), (100, 50), (200, 100) → MatchLine → WarpMap: forward 50 → 25.0, inverse 75 → 150.0 |

### EventData Struct Dict Discovery

A key implementation discovery documented in the tests: EventData converts `{"instant": 0.0}` to `{"start": {"value": 0.0, "numerator": None, "denominator": None}}` internally. The `instant` key is NOT preserved — it becomes `start`. The `temporal_type` field distinguishes instant vs interval events. WarpMap's `_warp_events()` handles both the struct dict format and plain floats.

### Floating-Point Tolerance (retained `pytest.approx`)

Every anchor, linear/non-linear forward map, inverse map, support boundary, chord
dedup, and every materialise assertion (lengths, instants, note
start/end/duration, region boundaries) resolves to an exactly representable
`float` and is asserted with `==`. The two fixed-input round-trip loops
(`test_round_trip_precision`, `test_round_trip_nonlinear`) also now use `==`:
each listed value recovers **bit-exactly** through `forward(inverse(x))` — the
earlier "accumulated sub-epsilon error" rationale was false for these specific
inputs. Only one assertion in this file retains `pytest.approx`, because it is
genuinely floating-point:

| Test | Tolerance | Why approx is required |
|------|-----------|------------------------|
| `test_proportional_warp_between_dgt_timelines` | `abs=1e-9` | The interior expected value `967 + (378/867) * 967` uses the non-dyadic rational scale `378/867`, which the map's interpolation and this test's recomputation evaluate via slightly different float operation orders (difference ~2.27e-13). The boundary-anchor loop in the same test maps anchors exactly and uses `==`. |

---

## Graphical Loader Tests (`test_graphical_loader.py`)

### What We're Validating

The graphical loader creates `GraphicalBundle` objects from images, mapping 2D pixel coordinates to 1D timeline coordinates.

### Key Components

| Component | Purpose |
|-----------|---------|
| `TimeAxisPath` | Abstract path mapping 1D -> 2D coordinates |
| `HorizontalLinePath` | Time axis as horizontal line (most common) |
| `ImageSource` | Unified image interface (files, PDFs) |
| `GraphicalSegment` | Source + path + timeline offset |
| `GraphicalBundle` | Complete timeline with coordinate conversion |
| `GraphicalLoader` | Factory for building bundles |

### Test Data

Test images are in `tests/data/thoresen/` (fetched via `pooch`; see `tests/data/README.md`):

| File | Description |
|------|-------------|
| `thoresen_2009_sound-objects_p312_page1_1.jpeg` | DGT1: single image, 5 horizontal systems |
| `thoresen_2010_form-building-patterns_p90-91_page*.jpeg` | DGT2: 5 separate images |

### Coordinate Data (from Applications.ipynb)

**DGT1 (2009):**
- Single image with 5 horizontal systems
- x-boundaries: (2, 969) for all systems = 967 pixels each
- y-positions: [18, 205, 396, 588, 785]
- Total width: 4835 pixels

**DGT2 (2010):**
- 5 separate images with varying dimensions
- Segment bounds (x0, x1, y): [(8,874,15), (7,874,18), (7,874,19), (8,872,15), (9,873,20)]
- Segment lengths: [866, 867, 867, 864, 864]
- Total width: 4328 pixels

**Event H (rect_h2):**
- Segment index: 1 (second segment)
- Local coordinates: [378, 517] (385-7 to 385-7+139)
- Global coordinates: [866+378, 866+517] = [1244, 1383]

### Why These Values Are Exact

The pixel coordinates come from:
1. Manual measurement in image editing software (x0, x1, y for each system)
2. Ground truth TSV files with annotated event locations
3. Cross-validation between Applications.ipynb calculations and test assertions

Any discrepancy between these sources indicates a bug that must be investigated--not tolerated

### Floating-Point Tolerance (retained `pytest.approx`)

Horizontal- and diagonal-path coordinate maps, all `GraphicalSegment`
to_image/from_image conversions, and every `GraphicalBundle`
timeline↔image conversion (including the DGT1/DGT2 event round-trips) land on
exactly representable values and are asserted with `==`. The only retained
`pytest.approx` assertion is the quarter-circle arc-length test, which is
genuinely floating-point:

| Test | Tolerance | Why approx is required |
|------|-----------|------------------------|
| `TestParametricPath::test_circle_quarter` | `rel=0.01` | Arc length of a quarter circle is computed by sampling 1000 points and summing chord lengths — a numerical approximation of `r·π/2`. |

The other two `ParametricPath` tests are now asserted with `==`:

- `test_straight_line_as_parametric` — every sampled segment of the
  `y=0` line contributes an exact chord length, so the accumulated arc length
  is exactly `100.0`, not a tolerance-bounded integral.
- `test_to_2d_endpoints` (4 asserts) — `ParametricPath._arc_to_t` does **not**
  search the sampled table at the endpoints: it clamps `arc <= 0` to `t_start`
  and `arc >= total_length` to `t_end` and returns the exact endpoint
  (`paths.py:450-455`). So `to_2d(0)` and `to_2d(length)` are bit-exact
  `(0, 0)` and `(100, 50)`.

---

## Running the Tests

```bash
cd timetoalign
python -m pytest tests/alignment/ -v
```

**Status**: Unified Stamp & Query API fully implemented. Full alignment suite: **561 passed** (alignment + core/ids). The skips are pre-existing stubs in `test_thoresen_poc.py` and graphical loader tests requiring PyMuPDF.

### Test Files

| File | Tests | Description |
|------|-------|-------------|
| `conftest.py` | — | Shared fixtures and constants: Thoresen timeline fixtures (`dgt1_timeline`, `dgt2_timeline`, `audio_timeline`, `thoresen_segment_claims`), DGT1/DGT2 coordinate constants, `autouse` ID reset fixture, graphical bundle fixtures |
| `test_bundle.py` | 131 | AlignmentBundle: linear alignment, a child joining with its root, cross-group transfer, timestamps, commensurability, caching, the MatchClaim table and its entry checks, Projections, the belief filter, coordinate-type parity (raw/Coordinate/IdCoordinate) |
| `test_bundle_dict.py` | — | AlignmentBundle dictionary form and identity: round trips of direct constructions and of every alignment loader's bundle, minted ids, collisions, a child bringing its root, refusals |
| `test_anchors.py` | 62 | AlignmentAnchor, MatchClaim (incl. NOMATCH coordinate preservation + repr), Provenance |
| `test_graph.py` | 93 | MatchGraph operations (inferred edges, filtering, the belief filter, stamps); imports Thoresen fixtures from conftest |
| Robustness coverage | — | Bundle graph stamp units/context, coordinate-bearing claim factories, Fraction serialization, inferred-edge completeness, and MatchStamp container isolation/axis handling |
| `test_matchline.py` | 33 | MatchLine construction, from_claims, from_graphs, coordinate pairs, serialization; imports Thoresen fixtures from conftest |
| `test_warpmap.py` | 39 | WarpMap construction, forward/inverse, materialise (events, children, regions, type conversion), serialization, end-to-end pipeline |
| `test_filters.py` | 62 | `ClaimFilter` dataclass: exact ID, set-of-IDs, regex, between, within, synchronous/nomatch, the belief filter, combined filters, timeline-level filtering, parity with `MatchClaimField.filter` |
| `test_stamp_query_api.py` | 72 | Unified Stamp & Query API: `get_match_claims()`, `get_matchstamp_at()`, position batches, `MatchGraph.get_matchstamp()`/`split_components()`, `MatchClaim.get_matchstamp()`, MatchGraph cache, display methods (`__str__`/`_repr_html_` incl. the affordance `Try` footer on `MatchStamp`/`MatchClaim`), the events a claim names, `transfer()` docstring, top-level exports |
| `test_supra_integration.py` | 13 | SUPRA piano roll workflow (partial alignment as a child) |
| `test_thoresen_poc.py` | 21 | Thoresen graphical analysis workflow; imports Thoresen fixtures from conftest |
| `../timelines/test_offset_arithmetic.py` | 11 | Parent–child offset arithmetic |

---

## Unified Stamp & Query API Tests

### ClaimFilter Tests (`test_filters.py`)

Tests for the `ClaimFilter` dataclass — the Unified Filter API. Covers all filter parameters individually and in combination:

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestClaimFilterCreation` | 5 | Empty filter, mutual exclusion, `from_kwargs`, repr |
| `TestClaimFilterExactId` | 4 | `timeline_id` filter on A-side, B-side, mismatch, timeline-level |
| `TestClaimFilterIdSet` | 3 | `timeline_ids` set filter |
| `TestClaimFilterRegex` | 4 | `id_pattern` regex: prefix, suffix, range, timeline-level |
| `TestClaimFilterBetween` | 3 | `between` pair filter: exact, reversed, mismatch |
| `TestClaimFilterWithin` | 3 | `within`: both timelines in the set, on claims and on timelines |
| `TestClaimFilterSynchronous` | 6 | `synchronous_only` and `nomatch_only`, a conceptual claim included |
| `TestClaimFilterCombined` | 5 | AND logic across multiple filters; a unit criterion rejects a timeline missing from the lookup |
| `TestClaimFilterBelief` | 5 (13 cases) | The belief filter (`believer`, `beliefs`) |
| `TestClaimFilterSelect` | 2 | `select`, the list form, keeps what `matches_claim` keeps, in order |
| `TestClaimFilterFieldParity` | 14 | `ClaimFilter.select` and `MatchClaimField.filter` keep the same claims, criterion by criterion |

The kinds: `synchronous_only` keeps a synchronous claim only, `nomatch_only` a NOMATCH claim only (one event, the sentinel on the other side). A conceptual claim (two events, not synchronous) is the case that tells "NOMATCH" from "not synchronous" apart, so `TestClaimFilterSynchronous` asks both criteria of one (`_conceptual`, `score:clt_1` ↔ `perf:dlt_5`): `nomatch_only` → `False`, `synchronous_only` → `False`; a filter equating NOMATCH with non-synchronous passes the first. `within={"score:clt_1", "perf:dlt_1"}` keeps the claim between those two and refuses `score:clt_1 ↔ perf:dlt_5` (one side out), and `matches_timeline` takes the same set (`score:clt_1` in, `perf:dlt_5` out). `select(claims, context=)` is the list form the MatchGraph filters with: over the four fixture claims plus a conceptual one, `ClaimFilter(nomatch_only=True).select(...)` is exactly `[claim_nomatch]`, and over the belief claims `ClaimFilter(timeline_id="score:clt_1", believer="ana").select(...)` equals `[c for c in claims if filter.matches_claim(c)]`, the same objects in order.

The claims name fixture events (`match_claim`; the NOMATCH claim one event and the sentinel). The belief tests use four claims on `score:clt_1`/`perf:dlt_1`, one per case: no provenance, believed `false` by `ana`, believed `true` by `ana`, believed `false` by `ben`. `matches_claim` keeps exactly: without a believer all four (for the default and for `["true"]`); `ana` with the default states (all but `false`) all but ana's false one (ben's belief is not hers); `ana`/`["true"]` only ana's true one; `ana`/`["undecided"]` the unstated claim and ben's (those ana holds no belief on); `ana`/`["false"]` only ana's false one; `ben` with the default all but ben's false one; `ben`/`["true"]` none; `ben`/`["true", "false"]` ben's false one. The states are held as a frozen set (`["false", "true", "false"]` → `{"true", "false"}`, repr `ClaimFilter(believer='ana', beliefs=['true', 'false'])` in the order of the four states; the default states are left out of the repr, and a filter given them in another order equals the default one). Combined with `timeline_id`, the belief filter is ANDed (ana's true claim kept; the unstated claim and a claim off the timeline dropped). `beliefs=["not_false"]` and the bare string `"unlikely"` are refused at construction (`ValueError`, the messages of `check_belief_filter`). `matches_timeline` ignores the belief filter, as it ignores every claim-level criterion.

**Two forms of one filter agree** (`TestClaimFilterFieldParity`). The claim criteria are decided twice: by `ClaimFilter._passes`, claim by claim (the MatchGraph filters with it), and by `MatchClaimField.filter`'s vectorized masks (the bundle filters its table with them). The library keeps both forms, the scalar one for claims held as objects and the columnar one for the table, so what proves them one filter is agreement on every criterion over every kind of claim. The claims: a synchronous event match `score:clt_1 ↔ perf:dlt_1`, a NOMATCH claim with its event on side A (`score:clt_1`, sentinel on `perf:dlt_5`), a NOMATCH claim with its event on side B (sentinel on `score:clt_1`, the event on `audio:cpt_1`), a conceptual claim `perf:dlt_1 ↔ perf:dlt_5`, and an interval claim `score:clt_1 ↔ audio:cpt_1`, believed `false` by `ana`. Each criterion is asked of both forms with the same arguments, and the kept claims' ids must be equal, in order, and equal to the ids written in the test: `timeline_id="perf:dlt_5"` → the side-A NOMATCH and the conceptual claim; `timeline_ids={"audio:cpt_1"}` → the side-B NOMATCH and the interval claim; `id_pattern=r"^perf:"` → the event match, the side-A NOMATCH and the conceptual claim; `between=("perf:dlt_1", "score:clt_1")` → the event match; `within={"score:clt_1", "perf:dlt_1", "perf:dlt_5"}` → the event match, the side-A NOMATCH and the conceptual claim; `synchronous_only` → the event match and the interval claim; `nomatch_only` → the two NOMATCH claims; `believer="ana"` → all but the interval claim; `believer="ana", beliefs=["false"]` → the interval claim; no criterion → all five; and four combinations (`timeline_id` with `nomatch_only`, `within` with `synchronous_only`, `id_pattern` with `believer`, `between` with `synchronous_only`).

### Stamp & Query API Tests (`test_stamp_query_api.py`)

Tests for the Unified Stamp & Query API, using a star-topology bundle (1 score + 3 performers, 5 coordinates each, plus 1 NOMATCH claim naming the score's event `orphan`). Every timeline counts in the dimensionless number unit (float axis), and every synchronous fixture claim names a fixture event on each side (`score:clt_1@50.0`, `helpers.match_claim`) with its coordinates on those axes, so the bundle admits them as given and every expected coordinate is the factor (2.0, 1.5, 1.8) times the score coordinate. The NOMATCH claim takes no part in a MatchGraph, so the counts split 15 synchronous + 1 NOMATCH.

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestGetMatchClaims` | 8 | `AlignmentBundle.get_match_claims()`: no filters, by timeline ID, by regex, synchronous_only, nomatch_only, between, combined |
| `TestMatchGraphGetMatchstamp` | 4 | `MatchGraph.get_matchstamp()`: single component, multi-component raises, empty graph, only-NOMATCH |
| `TestMatchGraphSplitComponents` | 4 | `split_components()`: single, two components, empty graph, `n_components` property |
| `TestMatchGraphStarTopology` | 2 | Star topology (1 score + N performers): single coordinate = one component, determinism |
| `TestMatchClaimGetMatchstamp` | 4 | `MatchClaim.get_matchstamp()`: reduced stamp, NOMATCH returns None, no-bundle raises, from-graph with bundle |
| `TestMatchGraphCache` | 5 | MatchGraph cache: hit, cross-key lookup, invalidation, no-claims raises, different coordinates |
| `TestGetMatchstampAt` | 6 | `AlignmentBundle.get_matchstamp_at()`: basic, non-zero coordinate, not-in-bundle, a position between anchors interpolated by the WarpMaps (37.5 → 75.0 / 56.25 / 67.5, `is_interpolated`, inferred edges only), regex filter, timeline_ids filter |
| `TestCoordinateBatch` | 7 | Position batches on `get_matchstamps_at()` / `get_matchstamp_table()`: order kept, one full cross-section per coordinate, IdCoordinate carries its timeline, parity with the singular getter, one table row per coordinate, `within` narrows columns, empty in → empty out, claims and positions together refused |
| `TestMatchStampDisplay` | 8 | `MatchStamp.__str__`/`_repr_html_`: header, entries, empty, integer formatting, valid HTML, bold anchors, greyed inferred, and typed retrieval affordance appended after the table |
| `TestMatchClaimDisplay` | 8 | `MatchClaim.__str__`/`_repr_html_`: instant, interval, NOMATCH, certainty and provenance lines (`Certainty:   0.95`, `Provenance:  method=inferred, note=…`), inferred, valid HTML, NOMATCH badge, affordance `Try` footer (`claim.get_matchstamp()`) appended after the table |
| `TestMatchClaimEventStorage` | 12 | The events a claim names: `from_events()` keeps both event IDs and names (a missing name is `None`; an event dict without an ID is refused, `event_a names no event id`); a position without an event is given a Projection by `create_match_claims` (`projection_1`, no name, `projection_of` naming the source event, its coordinate on the target axis); `nomatch()` names one event and `None` on the other side; IDs and names displayed in `__str__`/`_repr_html_` and round-tripped by `to_dict()`/`from_dict()`; direct construction |
| `TestTransferDocstring` | 1 | `transfer()` docstring no longer says "primary user-facing" |
| `TestTopLevelExports` | 4 | `MatchGraph`, `MatchStamp`, `ClaimFilter` importable from top-level |
