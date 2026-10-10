# Rekordbox Loader Tests - Validation Strategy

This document explains **how** the upcoming tests prove the Rekordbox loader
correct under the TimeToAlign! Zero Tolerance Validation Policy. Every anchor is
derived directly from the two ground-truth data files and stated as an exact value
with no range or approximation.

## Timeline-native measure structure

The synthetic single-track XML fixture already fixes the generated measure count
from its TEMPO grid. The measure map belongs to the track's quarter-anchored
structure, not to the seconds axis: the attached skeleton's section hierarchy
holds the one ``MeasureMap`` of exactly that count, with row IDs ``"1"`` to
``"4"``, while the seconds timeline carries none (``measure_map`` is ``None``)
and its document carries no measure map (the member is ``null``). Rekordbox TEMPO data states no score repeat,
jump, break, or target marker, so the bound flow-control registry must have
exactly zero breaks, zero jumps, and an empty marker mapping. A JSON round trip
of the seconds timeline must reproduce its payload exactly.

## Ground-truth data files

**`rekordbox.xml`** — a Rekordbox library export.

| Fact | Value |
|------|-------|
| Collection ``TRACK`` elements | **38** (the ``COLLECTION Entries`` attribute still reads 46, from before the export was curated — the elements are what counts) |
| ``TRACK`` elements carrying ``TEMPO`` tags | 38 (every collection track has ≥ 1) |
| Total ``TEMPO`` tags | **147** |
| The mix track ``TrackID`` | **147337955** |
| Mix ``Name`` | ``001-samuel_moriero-impact_halloween_xxl_2025_full_set`` |
| Mix ``TotalTime`` | **5277** (seconds) |
| Mix ``TEMPO`` tag count | **14** |
| Mix first ``TEMPO`` | ``Inizio 0.333``, ``Bpm 160.00``, ``Metro 4/4``, ``Battito 4`` |

**`tracks-to-mix.csv`** — an authored mapping of source tracks onto the mix.

| Fact | Value |
|------|-------|
| Data rows | **96** |
| Columns | ``track``, ``track_start_bar``, ``mix_start_bar``, ``track_end_bar``, ``mix_end_bar``, ``track_loop``, ``analysis`` |
| ``track`` labels | XML ``Name`` prefixed with ``"NN. "`` (a number, a dot, a space) |

---

## (a) `TEMPO` -> beat-grid semantics

**Ground truth.** Each ``TEMPO`` tag opens a **beat grid** on the track's physical
(seconds) timeline.

**Semantics a correct reader must encode.**

- A grid opens at ``Inizio`` seconds, carries tempo ``Bpm`` and meter ``Metro``,
  and its first beat (the beat *at* ``Inizio``) is beat number ``Battito`` of a
  measure. In ``4/4``, ``Battito`` ranges over 1..4.
- **``Battito`` fixes the first measure boundary.** ``Battito 1`` means ``Inizio``
  is itself a downbeat (a measure boundary). ``Battito B`` with ``B ≠ 1`` means the
  grid opens *mid-measure*; the next downbeat arrives ``(Metro − B + 1)`` beats
  later. For the mix's first grid (``Battito 4``, ``Bpm 160.00`` → beat 0.375 s),
  the first beat at ``0.333`` s is beat 4, and **the next beat begins a new
  measure** at ``0.333 + 0.375 = 0.708`` s. The bar containing ``Inizio`` is a
  truncated pickup (only beat 4 falls inside the grid).
- **Measure numbers rise monotonically across all grids of one track.** A pickup is
  the represented tail of nominal bar 0, and the first full downbeat is bar 1. The
  measure counter is continuous: a new grid does **not** reset it. Each downbeat
  increments the measure number; across grid boundaries the count only ever
  increases.
- A grid **ends** at the next grid's ``Inizio``, or at the track ``TotalTime`` for
  the last grid.
- **Boundary precedence: the half-beat rule.** A beat of one grid that falls
  within **half of that grid's beat duration** before the next grid's ``Inizio``
  **is** the next grid's anchor beat, displaced, and is not counted separately.
  At exactly half a beat the beat is kept. The rule applies at grid boundaries
  only, never at ``TotalTime``, where the last grid simply stops
  (``instant < TotalTime``).

  **Why exact-equality precedence is not enough.** ``Inizio`` is rounded to
  three decimals, and moving a grid's anchor to the right freezes the beats to
  its left without inserting a fill-in, so the displaced beat and the anchor
  are a hundredth of a second apart rather than equal. Counting both invents a
  bar and shifts every measure number after it.

  **The rule is a heuristic** and is documented as one: what the source program
  does for anchor displacements beyond half a beat has not been studied, so
  nothing is asserted there.

---

## (b) Per-grid bar-span derivation for the mix (14 grids)

**Ground truth.** The mix has 14 grids. In ``4/4`` a bar is 4 beats. At
``Bpm 160.00`` a beat is ``60 / 160 = 0.375`` s and a bar is ``4 × 0.375 = 1.5`` s.

**Derivation.** Each grid's span in seconds is ``next Inizio − this Inizio`` (the
final grid runs to ``TotalTime`` = 5277). Bars = span ÷ bar-length, where
bar-length uses **that grid's own bpm**: ``4 × 60 / Bpm``.

| # | Inizio (s) | Bpm | Battito | Span (s) | Bar (s) | Bars |
|---|-----------|-----|---------|----------|---------|------|
| 1 | 0.333 | 160.00 | 4 | 212.639 | 1.500000 | 141.7593 |
| 2 | 212.972 | 160.00 | 3 | 459.757 | 1.500000 | 306.5047 |
| 3 | 672.729 | 160.00 | 1 | 324.001 | 1.500000 | 216.0007 |
| 4 | 996.730 | 160.00 | 1 | 432.007 | 1.500000 | 288.0047 |
| 5 | 1428.737 | 160.00 | 1 | 669.011 | 1.500000 | 446.0073 |
| 6 | 2097.748 | **159.96** | 1 | 96.023 | 1.500375 | 63.9993 |
| 7 | 2193.771 | **159.96** | 1 | 60.014 | 1.500375 | 39.9993 |
| 8 | 2253.785 | 160.00 | 1 | 432.011 | 1.500000 | 288.0073 |
| 9 | 2685.796 | 160.00 | 1 | 726.008 | 1.500000 | 484.0053 |
| 10 | 3411.804 | 160.00 | 1 | 67.508 | 1.500000 | 45.0053 |
| 11 | 3479.312 | 160.00 | 1 | 670.494 | 1.500000 | 446.9960 |
| 12 | 4149.806 | 160.00 | 1 | 151.496 | 1.500000 | 100.9973 |
| 13 | 4301.302 | **160.38** | 1 | 134.679 | 1.496446 | 89.9992 |
| 14 | 4435.981 | 160.00 | 1 | 841.019 | 1.500000 | 560.6793 |

**Bpm deviations and why they matter.** Three grids do **not** carry ``160.00``:
grids 6 and 7 at ``159.96`` and grid 13 at ``160.38``. These are not noise — the
micro-adjusted bpm is exactly what makes each of those grids span a **whole**
number of bars:

- Grid 6: at ``159.96`` the bar is ``1.500375`` s and ``96.023 / 1.500375 =
  63.9993 ≈ 64`` bars; at a flat ``160.00`` the same 96.023 s would read
  ``64.015`` bars — off by a bar-and-a-half over the grid, so using the grid's own
  bpm is mandatory.
- Grid 7: ``60.014 / 1.500375 = 39.9993 ≈ 40`` bars.
- Grid 13: at ``160.38`` the bar is ``1.496446`` s and ``134.679 / 1.496446 =
  89.9992 ≈ 90`` bars; at ``160.00`` it would read ``89.786`` bars.

A test that computed grids 6/7/13 at a hard-coded ``160.00`` bar length must fail;
the loader must read ``Bpm`` per grid.

**Total mix bar count.** The whole mix, measured at the nominal ``160.00``/``4/4``
grid from ``t = 0`` to ``TotalTime``, is exactly

``5277 s ÷ 1.5 s/bar = 3518 bars`` (equivalently ``3518 × 4 = 14072`` beats).

This is the clean anchor: ``5277`` is an exact multiple of the ``1.5`` s bar. The
per-grid spans in the table sum to ``≈ 3517.97`` bars because grids 1 and 2 start
mid-measure (``Battito 4`` and ``3``) and the summation begins at the first grid
onset ``0.333`` rather than ``0``; the ``0.333`` s of pre-grid material plus the
truncated first pickup bar account for the remainder up to 3518.

**And it is the arithmetic that catches the phantom bars.** Under the half-beat
rule the mix generates exactly ``14072`` beats and ``3518`` downbeats — the
independent 3518 above. Exact-equality precedence generated ``3525``: seven
extra downbeats, one at each grid boundary whose predecessor ends 1–11 ms early.

| Boundary (next ``Inizio``) | Phantom downbeat | Gap (s) | Half a beat (s) |
|---|---|---|---|
| 672.729 | 672.722 | 0.007 | 0.1875 |
| 996.730 | 996.729 | 0.001 | 0.1875 |
| 1428.737 | 1428.730 | 0.007 | 0.1875 |
| 2097.748 | 2097.737 | 0.011 | 0.1875 |
| 2685.796 | 2685.785 | 0.011 | 0.1875 |
| 3411.804 | 3411.796 | 0.008 | 0.1875 |
| 3479.312 | 3479.304 | 0.008 | 0.1875 |

Every gap is far below ``0.1875`` s, so all seven are dropped
and the mix's measure records fall from 3526 to **3519** — 3518 grid bars plus
the ``Battito 4`` pickup as nominal bar 0.

**What the fix moves.** The bar that runs into the 672.729 boundary was a 7 ms
sliver (bar ``450``, ``7/1000 × 160/60 = 7/375`` quarters). It is gone; the bar
carrying that boundary is now ``measure_map[448]`` — id ``"449"``, an
``IrregularMeasure`` spanning ``671.222 → 672.729``, whose actual length is
``1507/1000 s × 160/60 = 1507/375`` quarters.

The seven phantoms also pushed every later measure number up by one per
phantom, which is the 7-bar drift the mix showed near its end. **Bar 2887,
derived from the grids alone:** 2867 downbeats fall in grids 1–12, so bar 2887
is the 20th downbeat of grid 13 (``Inizio 4301.302``, ``Bpm 160.38``, bar
``4 × 60 / 160.38 = 4000/2673`` s), that is

``2150651/500 + 19 × 4000/2673 = 5786690123/1336500 = 4329.734472876918`` s,

which is the ``72:09.7`` the source program displays for that bar. Under
exact-equality precedence the same query answered ``1924230041/445500 =
4319.259351290684`` s — the 2887th downbeat of a list holding seven beats that
are not downbeats, i.e. seven bars too early.

**Whole-collection effect.** Four of the 38 tracks generate phantom beats under
exact-equality precedence: the mix (−7), ``Fkn Raw`` (−1, at 156.839 before
156.840), ``HEROINE - Samuel Moriero Remix (FREE DL)`` (−2, at 60.070 and
63.070) and ``BLOW IT UP`` (−1, at 156.074 before 156.075). No track has a
downbeat exactly at ``TotalTime``, so the end-of-track branch is unaffected by
the change and is exercised synthetically.

---

## (c) Loader expectations: 38 physical-domain timelines

**Ground truth.** The 38 collection tracks.

**Expected.** The loader yields **38 timelines** in the physical domain (seconds),
one per collection track. Each timeline affords **floating-measure (``fm``)**
readings through its grids: given a seconds position, its ``fm`` value is derived
from the grid that contains it (its ``Inizio``, ``Bpm``, ``Metro``, ``Battito``),
with measure numbers continuous across that track's grids per section (a).
An anacrusis occupies its offset inside ``fm [0, 1)``; consequently, the first
full downbeat reads ``fm 1`` and bar ``b`` begins at ``fm b``. Every raw grid
instant is an interpolation anchor, including grid changes that occur mid-bar.

**Example tracks to pin (exact facts).**

- **``See Me Coming``** — single grid: ``Inizio 0.082``, ``Bpm 160.00``,
  ``Metro 4/4``, ``Battito 3``, ``TotalTime 256``. Because ``Battito 3``, the beat
  at ``0.082`` s is beat 3; the next downbeat (beat 1) is ``2 × 0.375 = 0.75`` s
  later at ``0.832`` s. Nominal bar 0 is a truncated pickup holding only beats 3
  and 4 inside the grid; the downbeat at ``0.832`` s reads ``fm 1``.
- **``Fkn Raw``** — four grids (mid-track grid changes), all ``Bpm 160.00``,
  ``Metro 4/4``: ``(0.087, Battito 3)``, ``(48.088, Battito 3)``,
  ``(102.464, Battito 4)``, ``(156.840, Battito 1)``; ``TotalTime 176``. The
  measure count must run continuously across all four grids (no reset at 48.088,
  102.464, or 156.840), and the ``Battito`` on each grid re-anchors where that
  grid's first downbeat falls.
- **``40. Brace For Impact``** — single grid: ``Inizio 0.145``, ``Bpm 150.00``,
  ``Metro 4/4``, ``Battito 2``, ``TotalTime 220``. Here the bpm is **not** 160:
  beat = ``60 / 150 = 0.4`` s, bar = ``1.6`` s. ``Battito 2`` puts the next
  downbeat ``3 × 0.4 = 1.2`` s after ``0.145`` s, at ``1.345`` s. This example
  guards against a loader that assumed the mix's 160/4/4 for every track.

---

## (d) `tracks-to-mix.csv` ingestion and name resolution

> **Sections (d), (e) and (f) specify a lane no loader implements yet.** They
> are the validation logic the alignment-CSV reader must satisfy when it is
> written — derived from the two data files, and to be asserted then. Nothing
> in `test_rekordbox_loader.py` currently covers them, so they must not be read
> as a description of existing behaviour. Sections (a)–(c), (g) and (h) do
> describe the implemented loader and are covered by tests.


**Ground truth.** 96 data rows. The ``track`` column carries the source file's
name stem — ``"NN. "`` prefix included (a number, an optional letter for split
entries such as ``41a.``/``41b.``, a dot, a space) — which per (g) is exactly
the collection timeline id.

**Name resolution.** A csv label resolves iff it **equals a timeline id** (the
URL-decoded ``Location`` stem): plain string equality, no prefix stripping, no
substring matching, no curated equivalences. Display-``Name`` matching is
forbidden — labels such as ``02. Tonight (Samuel Moriero Remix)`` differ from
their XML ``Name`` (``Habstrakt & Samplifire - Tonight (Samuel Moriero Remix)
FREE DL``) and resolve only by file stem.

**Expected — exactly which rows resolve.** Of the 96 rows, **90 resolve** and
**6 do not**: the six ``ID``-placeholder rows (``04.``, ``06.``, ``16.``,
``17.``, ``36.``, ``41a.``), whose audio was unavailable at analysis time and
which name no collection file. Unresolved rows are skipped. Of the 42 distinct
csv labels, 36 resolve; ``41b. Vielleicht Vielleicht`` is the one
``"NN. "``-prefixed source track never referenced by the csv. A test must
assert exactly this 90 / 6 split.

---

## (e) Loop semantics

**Ground truth.** A row with ``track_loop = N`` expresses that the mapped material
is looped: it must yield **N adjacent, equal-length intervals** on the mix
timeline, **all claiming the same** track interval ``[track_start_bar,
track_end_bar]``.

**Derivation that makes adjacency and equal length provable.** For a loop row with
mix interval ``[mix_start_bar, mix_end_bar]`` and ``N`` loops, the per-loop length
is

``L = (mix_end_bar − mix_start_bar) / N``.

The N intervals are ``[mix_start_bar + kL, mix_start_bar + (k+1)L]`` for
``k = 0 … N−1``. Adjacency holds because interval ``k`` ends exactly where
interval ``k+1`` begins; equal length holds because every interval has length
``L``; coverage holds because the union is ``[mix_start_bar, mix_end_bar]``. The
mapping is well-formed **iff** ``mix_end_bar − mix_start_bar = N × (track_end_bar −
track_start_bar)`` — i.e. ``L`` equals the track interval's own bar length, so each
mix loop is one full pass of the same track material.

**Every loop row in the CSV (all six resolve by name):**

| track | track [start,end] (bars) | mix [start,end] (bars) | mix span | N | L | check span = N × track span |
|-------|--------------------------|------------------------|----------|---|---|-----------------------------|
| ``10. The Sound`` | [64, 65] = 1 | [829, 841] | 12 | 12 | 1 | 12 = 12 × 1 ✓ |
| ``13. TURN IT UP!`` | [53, 61] = 8 | [1123, 1139] | 16 | 2 | 8 | 16 = 2 × 8 ✓ |
| ``28. Fkn Raw`` | [47, 51] = 4 | [2449, 2457] | 8 | 2 | 4 | 8 = 2 × 4 ✓ |
| ``32. GUCCI BAG - Revelation Live Edit`` | [76, 77] = 1 | [2868, 2880] | 12 | 12 | 1 | 12 = 12 × 1 ✓ |
| ``37. Annihilatiøn`` | [25, 26] = 1 | [3162, 3170] | 8 | 8 | 1 | 8 = 8 × 1 ✓ |
| ``37. Annihilatiøn`` | [59, 60] = 1 | [3166, 3170] | 4 | 4 | 1 | 4 = 4 × 1 ✓ |

**Expected claim count after loop expansion.** The 6 loop rows expand to
``12 + 2 + 2 + 12 + 8 + 4 = 40`` claims.

Total-claim accounting, all derived from the files:

| Category | Rows | Claims |
|----------|------|--------|
| Unresolved by name (skipped) | 6 | 0 |
| Resolved, non-loop, **complete** track interval (both bar endpoints present) | 78 | 78 |
| Resolved, loop rows (expanded) | 6 | 40 |
| Resolved, non-loop, **incomplete** track interval (a bar endpoint is missing) | 6 | 0 complete bidirectional claims |

The **6 incomplete rows** resolve by name but lack a full ``[track_start_bar,
track_end_bar]`` interval (``09. Rodeo`` ×1; ``11. LIKE ME`` ×2;
``14. Tunnel Vision - Junkie Kid Remix`` ×3). They cannot express a bar-unit
interval-on-track claim.

Therefore the exact expected count of **complete interval↔interval bar claims** is
``78 + 40 = 118``, with ``6`` rows skipped for name resolution and ``6`` resolved
rows carrying no complete track interval. (If a future loader instead emits one
mix-side placement per resolved row regardless of a missing track endpoint, the
count is ``84 + 40 = 124``; a test must state which policy it asserts. This
iteration pins **118** complete claims and calls the other 6 out explicitly rather
than folding them in silently.)

---

## (f) What a correct claim looks like

**Ground truth.** A claim maps an **interval-on-mix** to an **interval-on-track**,
both in bar units.

**Concrete non-loop row — ``01. See Me Coming`` (first entry).**
``track [1, 89]``, ``mix [3, 91]``. The claim records the mix interval
``[3, 91]`` (88 bars) ↔ the track interval ``[1, 89]`` (88 bars). Endpoint
arithmetic: mix span ``91 − 3 = 88``; track span ``89 − 1 = 88``; equal length,
so this is a 1:1 pass with no stretch, offset by two bars.

**Concrete non-loop row — ``01. See Me Coming`` (second entry).**
``track [121, 166]``, ``mix [91, 136]``. Claim: mix ``[91, 136]`` (45 bars) ↔
track ``[121, 166]`` (45 bars). Endpoint arithmetic: ``136 − 91 = 45`` and
``166 − 121 = 45`` — equal length again, but the track side jumps from bar 89 to
bar 121, i.e. this row skips 32 bars of the source before re-entering.

**Concrete loop row — ``13. TURN IT UP!``.** ``track [53, 61]``, ``mix [1123,
1139]``, ``track_loop = 2``. With ``L = (1139 − 1123) / 2 = 8`` (= the track span
``61 − 53``), the row expands to exactly two adjacent, equal-length claims, both
naming the **same** track interval:

- claim 1: mix ``[1123, 1131]`` ↔ track ``[53, 61]``;
- claim 2: mix ``[1131, 1139]`` ↔ track ``[53, 61]``.

They abut at bar 1131 and together cover ``[1123, 1139]``. A correct claim carries
both bar-unit intervals and their orientation (which axis is mix, which is track);
loop expansion must preserve the shared track interval across all N copies.

---

## (g) Timeline identity and conversion axes

### Identity comes from the file name, not the display name

**Ground truth.** The csv (and every artifact that cross-references the
collection) names tracks by their **file name** — ``"NN. "`` prefix included —
while the XML ``Name`` attribute is display metadata that often differs (artist
prefixes, label suffixes, remix tags). The stable identity of a track is
therefore the URL-decoded stem of its ``Location`` attribute:
``file://localhost/~/git/djmix-analysis/data/moriero/01.%20See%20Me%20Coming.mp3``
identifies the track ``01. See Me Coming``.

**Validation logic.** A synthetic track whose ``Location`` basename **differs**
from its ``Name`` separates the two lanes: the created timeline's **id** must be
the decoded ``Location`` stem and its **name** must remain the XML ``Name``. A
track carrying **no** ``Location`` cannot derive a file identity, so its id must
fall back to ``Name`` — never to an invented value. The proof requires the two
values to be observably different in the first case and identical in the second.

**Specimen anchors (exact).** All 38 collection tracks carry a ``Location``, so
the bundle's timeline ids are exactly the 38 decoded file stems. Pinned members:
``01. See Me Coming``, ``05. HUMBLE (Samuel Moriero REMIX)``,
``41b. Vielleicht Vielleicht``, and the mix
``001-samuel_moriero-impact_halloween_xxl_2025_full_set``. The six csv
placeholder labels (``04. ID``, ``06. ID``, ``16. ID``, ``17. ID``, ``36. ID``,
``41a. ID``) have **no** audio file in the collection and therefore must NOT
appear as timeline ids — a loader that minted ids from csv labels or ``Name``
collisions would fail this disjointness.

### SampleRate affords a seconds → samples axis; its absence affords nothing

**Ground truth.** A track's ``SampleRate`` attribute states the audio's discrete
resolution. ``SampleRate="48000"`` means second ``s`` is sample ``s × 48000``
exactly; samples are a discrete (integer) unit.

**Validation logic.** A track declaring ``SampleRate="48000"`` must expose a
samples conversion axis on its seconds timeline such that querying the position
``1.5`` seconds answers exactly ``72000`` samples — an integer on a discrete
axis, reached through the public retrieval surface (a ``Coordinate`` goes in, a
``Coordinate`` comes out; no bare floats). ``1.5 × 48000 = 72000`` is exact, so
no rounding ambiguity can hide an off-by-one. A track with **no** ``SampleRate``
attribute states nothing about its resolution, and the loader must attach **no**
samples axis — guessing a default sample rate would fabricate a fact the source
does not carry. All 38 specimen tracks carry a ``SampleRate`` (only ``44100``
and ``48000`` occur), so the absence branch has synthetic coverage only.

---

## (h) Synthetic grids: the boundary rule and mid-measure re-anchoring

The specimen proves the rule at scale; two synthetic collections prove it at
the places a specimen cannot isolate. Every value below is derived from the
``TEMPO`` attributes alone.

### Phantom suppression on both sides of the threshold

``TotalTime 30``; grids ``(0.5, 120, 4/4, 1)``, ``(10.51, 120, 4/4, 1)``,
``(20.31, 120, 4/4, 1)``. At 120 BPM a beat is ``0.5`` s, a bar ``2`` s and
half a beat ``0.25`` s.

- Grid 1's beats run ``0.5, 1.0, …``. Its 21st beat would fall at ``10.5``,
  ``0.01`` s before grid 2's anchor — **below** half a beat, so it is the
  displaced anchor and is dropped. Grid 1 therefore contributes 20 beats and
  its downbeats are ``0.5, 2.5, 4.5, 6.5, 8.5``.
- Grid 2's last beat falls at ``10.51 + 19 × 0.5 = 20.01``, ``0.30`` s before
  grid 3's anchor — **above** half a beat, so it is kept. The same rule
  therefore has one case dropping and one keeping in the same file.
- Grid 3 is the last grid: it stops at ``TotalTime``, keeping every beat
  strictly below ``30``; its last is ``20.31 + 19 × 0.5 = 29.81``.

Fifteen downbeats, so **15 measure records** and no pickup (``Battito 1``).
Three of them are irregular, one per grid boundary and one at the track end:

| Measure | Span (s) | Actual length (quarters) |
|---|---|---|
| 5 | ``[8.5, 10.51)`` | ``2.01 × 2 = 201/50`` (4.02) |
| 10 | ``[18.51, 20.31)`` | ``1.8 × 2 = 18/5`` (3.6) |
| 15 | ``[28.31, 30)`` | ``1.69 × 2 = 169/50`` (3.38) |

Floating-measure anchors: ``0 → 0.75`` (grid 1 opens ``0.5`` s = a quarter bar
after nominal bar 1, so ``1 − 0.5/2``), ``10.51 → 6.0`` (six downbeats at or
before it) and ``20.31 → 11.0``.

Grid cross-checks, which must agree because the measure records are generated
from the beat grid: ``seconds_at(6) == 10.51`` and ``position_at(10.505)``
answers the beat at ``10.0`` — measure 5, beat 4, segment 0 — because a
position between two beats floors to the earlier one, and 10.5 is not a beat.

### A grid that re-anchors mid-measure

``TotalTime 12``; grids ``(0, 60, 4/4, 1)`` and ``(6.01, 120, 4/4, 3)``. At 60
BPM a beat is ``1`` s (half a beat ``0.5``); at 120 BPM it is ``0.5`` s.

- Grid 1's beat at ``6.0`` is ``0.01`` s before grid 2's anchor, well inside
  half of grid 1's beat, so it is dropped; grid 1 keeps ``0 … 5.0``.
- Grid 2 anchors on ``Battito 3``, so its first two beats (``6.01``, ``6.51``)
  belong to the measure grid 1 opened at ``4.0``: set measure **2**, segment
  measure **0**. Its first downbeat is ``6.01 + 2 × 0.5 = 7.01``: set measure
  **3**, segment measure **1**. This is the case where the two numberings must
  differ.
- Measure 2 spans ``[4, 7.01)`` across the tempo change: ``2.01 s × 1 q/s +
  1.0 s × 2 q/s = 401/100`` quarters (4.01), a value neither tempo alone
  produces.

Grid cross-checks: ``seconds_at(3) == 7.01``; ``position_at(6.5)`` floors to
the anchor beat at ``6.01`` — segment 1, set measure 2, segment measure 0,
beat 3 — and ``position_at(6.6)`` reaches the next beat at ``6.51``, beat 4.

### The knife edge

The pickup/grid-change track (``(0, 60, 4/4, 4)``, ``(3.5, 120, 4/4, 2)``,
``(7, 120, 4/4, 1)``, ``TotalTime 8``) carries the boundary case: grid 1 runs
at 60 BPM, so half a beat is ``0.5`` s and its beat at ``3.0`` sits exactly
that far before grid 2's anchor. **It is kept.**

That beat is beat 3 of its measure, not a downbeat, so the measure records are
the same either way — the measure map cannot see the difference and this
track's measure expectations are unchanged by the rule. The beat lattice can:
grid 1 contributes four beats (``0, 1, 2, 3``) rather than three, and
``position_at(3.2)`` answers the beat at ``3.0`` rather than the one at
``2.0``. Both are asserted at grid level, in `tests/timelines/test_beatgrid.py`,
because a `<` / `<=` slip in the rule shows up nowhere else.
