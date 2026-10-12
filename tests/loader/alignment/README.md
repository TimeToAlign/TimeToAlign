# Alignment-loader tests

This directory holds the corpus-driven tests for loaders in
`timetoalign.loader.alignment` that need their own fixtures and their own
validation write-up.  The older alignment-loader tests still live one level up
in `tests/loader/` (`test_matchfile_loader.py`, `test_listen_here_loader.py`,
…); see `tests/loader/README.md` for those.

Corpus paths are resolved through `timetoalign.testdata.ensure_data("<corpus>")`
(see `tests/data/README.md`).  Hardcoded relative `Path("tests/data/...")`
constants are forbidden.  `conftest.py` calls `ensure_data("ieee1599")` at
module level so the corpus is materialised once, before collection.

| File | What it validates |
|------|-------------------|
| `test_ieee1599.py` | `Ieee1599Loader` against all six IEEE 1599 specimens — spine delta accumulation, the four projection layers, verbatim media references, and the spine-hub claim topology. Zero-tolerance counts and coordinates (see below). |
| `test_ieee1599_structural.py` | The `<structural>` layer of the gymnopédie specimen — the segment ↔ Petri-net-place resolution and the `external_references` rows it produces on the spine. Exact row counts and exact resolutions (see below). |

Fast lane: gymnopédie, animals, khomus.  The three large specimens
(pazzariello, serie, bach) carry `@pytest.mark.slow` and run under
`--runslow`.

## `Ieee1599Loader` validation logic

Corpus: `ieee1599` — six complete IEEE 1599 packages.

| specimen | XML path |
|---|---|
| gymnopédie | `SatiePetriNets/ieee1599/gymnopedie_01.xml` |
| animals | `Animals and their Sounds/animals_and_their_sounds.xml` |
| khomus | `Khorus Music/khomus.xml` |
| pazzariello | `Pazzariello Sparata/pazzariello_sparata.xml` |
| serie | `Serie in 9_8/serie_in_9_8.xml` |
| bach | `bach_artefuga_01.xml` (corpus root) |

All six declare `<ieee1599 version="1.0">` with a DOCTYPE and no namespaces;
three carry a UTF-8 BOM.  Nothing is fetched for the DTD and no media file is
ever opened, so the tests assert only what the XML states.

### Spine: relative deltas, cumulative coordinates

A spine `<event>`'s `timing` and `hpos` attributes are **deltas against the
preceding event**, not absolute positions.  The stored coordinate is therefore
the running sum:

```
vtu[0]  = timing[0]
vtu[i]  = vtu[i-1] + timing[i]
```

This is the single most consequential reading in the loader, so it is asserted
from both ends:

* **animals** is the minimal witness — eight events with deltas
  `0, 1, 1, 1, 1, 1, 1, 1` accumulating to coordinates `0 … 7`.  Read as
  absolute values the sequence would be `0, 1, 1, 1, 1, 1, 1, 1`, so the two
  readings differ on every event after the second and a wrong loader cannot
  pass.
* **khomus** shows the pattern with real note values: deltas
  `0, 0, 0, 0, 1024, 1024, 2048, 1024, 2048, 2048` accumulate to
  `0, 0, 0, 0, 1024, 2048, 4096, 5120, 7168, 9216`.  The four leading zeros are
  the time signature, key signature, clef and first rest, all at the same
  instant.
* **gymnopédie** shows the same for two simultaneous parts: its first nine
  events all carry `timing="0"` and therefore all sit at coordinate `0`; the
  tenth (`part_2_voice1_measure1_ev1`, delta `1024`) is the first to move.

Consequences asserted per specimen: the number of *distinct* coordinates is far
smaller than the number of events (gymnopédie 188 of 382, khomus 38 of 41,
animals 8 of 8), and the timeline `length` equals the total of all deltas
(gymnopédie 236544, khomus 48128, animals 7).

`hpos` is accumulated the same way and stored as one integer event field; the
deltas themselves are *not* stored, being recoverable by differencing.  Source
event ids are asserted verbatim (`Clef_part_1_1`, `event_cow`, …) — they are
what `event_ref` resolves against.

### Layers and timeline identity

| layer | timeline | unit |
|---|---|---|
| `<spine>` | `spine:dlt_1` (`DiscreteLogicalTimeline`) | ticks, int |
| `<los>` notes/rests/lyrics | `los:dlt_2` (`DiscreteLogicalTimeline`) | ticks, int |
| each `<graphic_instance_group>` | `<role>:dgt_<n>` (`SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]`) | pixels, int |
| each `<track>` | `<role>:cpt_<n>` (`ContinuousPhysicalTimeline`) | seconds, float |

Roles are asserted exactly, because they encode the sanitisation rule
(accent folding, lowercasing, every run outside `[a-z0-9_]` collapsing to one
`_`): the gymnopédie edition `"eng:Montréal: Les Éditions Outremontaises
(2006)"` becomes `eng_montreal_les_editions_outremontaises_2006`, and the track
`audio/ChaseColeman/satie-gymnopedie1-coleman.mp3` becomes
`satie_gymnopedie1_coleman`.  All timelines are standalone: the bundle holds no
`TimelineGroup`, since the claims already carry the connectivity.

A LOS event sits at the VTU coordinate of the spine event it references — that
reference *is* its temporal position.  A `<chord>` contributes one event per
`<notehead>` so that every event carries at most one pitch; the chord is
recovered by grouping on `event_ref` and ordering by `notehead_index`.  Hence
`los` event counts are notehead counts, not chord counts (gymnopédie: 469
noteheads from 288 chords).

Every `<graphic_instance>` declares `measurement_unit="pixels"`, so an edition
is measured in the `pixels` the document itself declares.  A pixel is
countable, so the boxes are rounded round-half-even onto
`DiscreteGraphicalTimeline` accolades; the verbatim box survives as
`source_bbox` wherever rounding would lose precision (`animals` has
`upper_left_x="992.96"`), and a test asserts both the rounded coordinate and
that fractional source box together.  The declared `measurement_unit` is
asserted to survive per page in `edition.meta["pages"]`.

An edition nests two levels, because a page is the unit that owns a pixel
origin: the edition line's segments are its pages (one per
`<graphic_instance>`, named `page_<n>`), and a page's segments are the
accolades engraved on it (named `accolade_<k>`, numbered within their page).
Positions are names, not ids: page and accolade ids are minted in the
edition's manifest, and the gymnopédie test pins both (pages `dgt_1`, `dgt_7`,
`dgt_12`, `dgt_18`; the first page's accolades `dgt_2` … `dgt_5`). The order
of minting follows the library's join rule — a page's subtree is planned as a
whole when it joins the edition — which is why the numbers are not
consecutive across pages.  Accolades are found by the half-span rule — within a page, a
drop in `upper_left_x` greater than half that page's observed x-span starts a
new accolade — and every page restarts the rule, so an accolade never straddles
two images.  The nesting is asserted three ways: the page/accolade counts of
all six specimens (gymnopédie: 4 pages of 4, 5, 5, 4 accolades per edition),
the fact that the first accolade of a page ends at the first half-span drop,
and each page child naming its own image file in `meta["page"]`.

Both levels concatenate their children, so the edition coordinate of an event
is `page_offset + accolade_offset + upper_left_x`; page-local x coordinates
overlap between pages and are disambiguated by that offset, never by the raw
box.  Claims therefore carry edition coordinates, and the edition's
`IntervalToConstantMap` named `file_name` maps any edition coordinate back to
the page image containing it.

Graphic events are interval events from `upper_left_x` to `lower_right_x`, with
the integer `bbox` struct, the page `file_name` and `position_in_group` as
fields, so a single event resolves to its image without walking the hierarchy.

Track events are instants at `start_time` seconds.

### Claim-count arithmetic

Every projected event contributes exactly one synchronous claim tying its own
coordinate to the referenced spine event's VTU coordinate, so

```
claims = los_events + graphic_events + track_events
los_events = noteheads + rests + lyric syllables
```

with no term for `<staff_list>` clefs / key signatures / time signatures (they
reference spine events but are not note/rest/lyric content) and none for the
`<structural>` layer (it states references into external resources, not
coordinates, so it produces `external_references` rows and no claim at all).

| specimen | spine | los = nh + rest + syl | graphic | track | claims |
|---|---|---|---|---|---|
| gymnopédie | 382 | 557 = 469 + 88 + 0 | 764 | 764 | **2085** |
| animals | 8 | 16 = 0 + 8 + 8 | 27 | 16 | **59** |
| khomus | 41 | 38 = 36 + 2 + 0 | 78 | 82 | **198** |
| pazzariello | 616 | 1070 = 810 + 131 + 129 | 1162 | 616 | **2848** |
| serie | 3509 | 3443 = 2480 + 963 + 0 | 3442 | 18395 | **25280** |
| bach | 1144 | 1132 = 1038 + 94 + 0 | 3397 | 4576 | **9105** |

The counts do not match the spine event count in either direction, which is
the point: gymnopédie's 382 spine events include 6 clef/key/time events with no
LOS counterpart but with graphic and track projections (hence 764 = 2 × 382
graphic and track events); khomus's two editions omit two spine events each
(39 boxes for 41 events) while both its tracks index all 41; serie's six tracks
are of unequal completeness (3509 × 4 + 3507 + 852 = 18395); and animals'
`Coloring page` and `Animal shapes` map only some animals, some of them twice
(8 + 6 + 13 = 27).

The claims are held **columnar** — one `MatchClaimField` for the whole
document, built with `MatchClaimField.from_columns`, never one `MatchClaim`
object per row.  The three layers are measured in ticks, pixels and seconds
respectively, which one field can hold because the coordinate storage carries a
unit *per row*; the tests materialise the first claim of each layer and assert
its `coordinate_a.unit` (ticks / pixels / seconds) against `coordinate_b.unit`
(always ticks, always `spine:dlt_1`).  Per-layer row counts are recovered by
counting the field's `timeline_a_id` column against the loader's `los_uid` /
`edition_uids` / `track_uids`, which is what the arithmetic above is asserted
through.  The bundle holds exactly one claim field and its per-claim Python
list stays empty (`cross_group_claims == []`) while `n_cross_group_claims`
reports the totals above.

### Structural layer → `external_references` on the spine

Only the gymnopédie specimen has a `<structural>` layer.  It states two
independent analyses of the same piece, each in two halves that have to be
joined to mean anything:

```xml
<analysis id="Analisi_1" author="Simone Delle Fave">
  <segmentation>
    <segment id="Analisi_1_L1_A">
      <segment_event event_ref="part_1_voice0_measure1_ev0"/>   <!-- ×32 -->
```
```xml
<petri_nets>
  <petri_net file_name="Analisi_1/L1.pnml">
    <place place_ref="p2" segment_ref="Analisi_1_L1_A"/>
```

A `<segment>` names the spine events it covers; a `<place>` names the segment
one node of one Petri net models.  Joining the two on the segment id answers
"which Petri-net node models this spine event", which is a reference *into* an
external resource rather than a timing statement — so it is carried as
`external_references` on the spine timeline, not as claims and not as events.

**Resolution rule.** For every `<segment_event>`, look up the `<place>` entries
whose `segment_ref` is the enclosing segment's id and emit one row per
*(segment_event, place)* pair:

| column | value |
|---|---|
| `event_id` | the `<segment_event>`'s `event_ref` (a spine event id) |
| `external_id` | the `<place>`'s `place_ref`, e.g. `p2` |
| `access_points` | `[{"uri": <the enclosing petri_net's file_name>, "kind": "relative_path"}]` |
| `comment` | the segment id, e.g. `Analisi_1_L1_A` |

A segment no `<place>` names still contributes one row per event, with
`external_id` = the segment id, `access_points` = `[]` and
`comment` = `"segment without petri-net node"`; the segmentation is stated by
the document and survives whole even where the Petri-net modelling does not
cover it.  The `.pnml` files are never opened and their paths are never
resolved against disk — the whole mapping is in the XML, and the `file_name` is
recorded verbatim (`Analisi_1/L3_RS_I.pnml`, `Analisi_2/L3-RS-C.pnml`: the two
analyses do not even agree on a separator).  The uri test pins the **complete**
set of distinct access-point uris — all **29** of them, as an exact sorted-list
equality (not a `<=` subset check) — so both the count and every verbatim,
unresolved path are held to zero tolerance (19 `Analisi_1/…` nets + 10
`Analisi_2/…` nets).

**Exact-count arithmetic.** The specimen states

| | Analisi_1 | Analisi_2 | total |
|---|---|---|---|
| `<segment>` | 49 | 46 | **95** |
| `<segment_event>` | 892 | 903 | **1795** |
| `<petri_net>` | 19 | 10 | **29** |
| `<place>` | 49 | 48 | **97** |

The 97 places carry 97 *distinct* `segment_ref` values, so no segment is named
by two places and the per-segment multiplier is never greater than one.  Of
those 97, 94 name a segment that exists; three do not and therefore resolve to
nothing at all:

* `Analisi_1_L3_RS__23` (place `p2` of `Analisi_1/L3_RS_I.pnml`) — a typo for
  the segment `Analisi_1_L3_RS_I_23`, whose `I` it drops;
* `Analisi_2_L3_RS_C_15` and `Analisi_2_L3_RS_C_16` (places `p9` / `p8` of
  `Analisi_2/L3-RS-C.pnml`) — segment ids the segmentation never defines.

That leaves exactly one of the 95 segments unmapped — `Analisi_1_L3_RS_I_23`,
the victim of the typo, which covers a single event — so the fallback path is
exercised by the specimen rather than only by construction.  With every
multiplier equal to one, the total is simply the number of `<segment_event>`
elements:

```
rows = Σ over segments (segment_events × max(places, 1))
     = 1794 × 1  (94 mapped segments)  +  1 × 1  (the unmapped one)
     = 1795      = 892 (Analisi_1) + 903 (Analisi_2)
```

Asserted exactly: **1795** rows, of which **exactly one** has empty
`access_points`; **32** rows for `Analisi_1_L1_A`, all with `external_id`
`p2` and access point `Analisi_1/L1.pnml`; **10** rows for
`Analisi_2_L2_RS_B`, all with `external_id` `p4` and access point
`Analisi_2/L2-RS.pnml`.  The rows name **376** distinct spine events — the 6
spine events missing from the 382 are the clef / key-signature / time-signature
events, which no segment covers.  Both analyses contribute (892 + 903), the
same rows appear in the loader's curated `structural` store table, and the
spine reaches them identically whether it is built by `create_timeline()` or by
`create_bundle()`.

`spine.to_dict(external_references=True)` renders all 1795 rows with their
nested access points; plain `to_dict()` omits the key entirely.  The five
specimens without a `<structural>` layer have no `structural` store table and
their spine's `external_references` is an empty table carrying the canonical
schema.

### Cross-section (`get_matchstamp_table(from_graph=True)`)

`from_graph=True` runs a union-find over the `(timeline_id, coordinate)` graph
the claims induce and emits one row per connected component.  Because every
claim has the spine on its B side, every component contains at least one spine
node, and the spine column is therefore non-null and unique — **at most one row
per spine coordinate**.

Rows are *fewer* than the distinct spine coordinates whenever two spine
coordinates share a projection node: two notes on different staves are engraved
at the same `upper_left_x`, and two spine events can be indexed at the same
`start_time` in a track.  That merging is a property of the specimens, not of
the loader, so the expected row counts are asserted exactly:

| specimen | distinct spine coordinates | cross-section rows |
|---|---|---|
| gymnopédie | 188 | **50** |
| animals | 8 | **8** |
| khomus | 38 | **36** |
| pazzariello | 300 | **208** |
| serie | 1156 | **30** |
| bach | 571 | **38** |

animals is the specimen where no merging occurs, so its whole cross-section is
asserted cell by cell: eight rows, one per animal, each carrying the LOS
coordinate, both audio timings and whichever pictures depict that animal
(`Coloring page` has no cat and no dog; `Animal shapes` has no pig and no
sheep).  Where one picture shows an animal more than once the collapse keeps
the smallest coordinate — the cow's two boxes at `x = 159.0` and `x = 14.25`
appear as `14.25`.

### Fidelity spot-checks

* **Durations** are kept as the verbatim `num` / `den` integer pair, not as a
  reduced `Fraction`: animals' whole-bar rests are `4/4`, and a test asserts
  `duration_num == 4 and duration_den == 4` (a `Fraction` would have collapsed
  them to `1`), while `Fraction(num, den)` is the exact notated value.
* **`<undefined/>` accidentals** (pazzariello, exactly 2) are stored as the
  string `"undefined"` on the note that carries them, never dropped and never
  inferred; both notes keep their `actual_accidental` (`double_sharp`,
  `natural`) alongside.
* **Ties** (bach, exactly 140) set `tie=True` on the notehead that carries the
  `<tie/>`.
* **Tuplets** (pazzariello, exactly 51) keep all four `tuplet_ratio` integers.
* **Lyrics** (animals 8, pazzariello 129) keep their text and their `hyphen`
  attribute.
* **Track descriptions** reach the timeline whole: a `<track_general>`'s
  `<performers>`, its `<recordings>` attributes and its free-text `<notes>` are
  kept verbatim in `meta`, and a key is absent when the document does not state
  it (gymnopédie's Coleman track notes read `Chase Coleman` and it has no
  `<recordings>`; pazzariello has no `<track_general>` at all).
* **Media references** are recorded verbatim and never resolved: khomus
  declares `file_format="video_avi"` for a `.mp4`, and every one of bach's
  media files is absent from disk — a test asserts both the recorded name and
  the file's absence, proving nothing was opened.
* **Column pruning**: a layer's table carries only the columns the specimen
  populates, so `text` is absent from the gymnopédie LOS table and
  `tuplet_enter_num` appears only for pazzariello.

### Loader contract

The loader accepts exactly one document (one spine, one bundle) and refuses a
second by whichever door it arrives at — `load()`, and equally the inherited
`load_string()` and `load_element()`, which would otherwise re-enter the parse
and suffix a second document's timelines onto the first's.  All three raise the
same `ValueError`, and a test drives each of them.  `create_timeline(uid)`
builds once and returns the cached timeline,
`create_timelines(id_pattern=...)` filters by uid regex, `create_group()`
returns `None`, and `get_field(MatchClaim)` follows the selector pattern: both
`MatchClaim` and `MatchClaimField` resolve to the document's single claim
field, anything else raises `TypeError`, and calling it before `load()` raises
`RuntimeError`.
