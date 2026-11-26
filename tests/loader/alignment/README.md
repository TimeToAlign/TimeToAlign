# Alignment-loader tests

This directory holds the tests for loaders in `timetoalign.loader.alignment`
that need their own fixtures and their own validation write-up, driven by a
corpus or by small files the tests write themselves.  The older alignment-loader tests still live one level up
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
| `test_trompa_maps.py` | `MapsLoader` on a MEI score and a MAPS file the tests write — one claim per listed `xml:id`, both sides of every claim, retrieval through a bundle on a samples and on a seconds performance, and every refusal (see below). |

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
| `<spine>` | `spine:dlt_…` (`DiscreteLogicalTimeline`) | ticks, int |
| `<los>` notes/rests/lyrics | `los:dlt_…` (`DiscreteLogicalTimeline`) | ticks, int |
| each `<graphic_instance_group>` | `<role>:dgt_…` (`SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]`) | pixels, int |
| each `<track>` | `<role>:cpt_…` (`ContinuousPhysicalTimeline`) | seconds, float |

Every id is minted from the loader's minter (seeded from the document's
bytes), so a uid is its role and prefix followed by 13 random base-36
characters. The tests compare a uid's *shape*, the uid without its suffix
(`spine:dlt`), after checking the suffix is 13 base-36 characters, and look a
timeline up by a regular expression matching its whole uid
(`create_timeline("spine:dlt_.*")`, which the loader resolves). A role is the
layer's label (an edition's `description`, a track's file stem); where layers
would share one, with each other or with `spine` and `los`, each takes a role
from more of what the source states (a track its whole `file_name`, an
edition its description and the `file_name` of its first
`<graphic_instance>`), so a lookup by role finds one timeline; there is no
counter. The test is a document with two tracks `a/take.mp3` and
`b/take.mp3`, a track `spine.wav` and two editions described `Score`, each
with its own first image: the roles are exactly `a_take_mp3`, `b_take_mp3`,
`spine_wav`, `score_p1_png`, `score_q1_png`, and `create_timeline` of each
`<role>:.*` returns one timeline. Roles are asserted exactly, because they encode the sanitisation rule
(accent folding, lowercasing, every run outside `[a-z0-9_]` collapsing to one
`_`): the gymnopédie edition `"eng:Montréal: Les Éditions Outremontaises
(2006)"` becomes `eng_montreal_les_editions_outremontaises_2006`, and the track
`audio/ChaseColeman/satie-gymnopedie1-coleman.mp3` becomes
`satie_gymnopedie1_coleman`.  The bundle holds every timeline in a group of
its own, since the claims already carry the connectivity. A group's name is
its timeline's role, what the document states, never a minted id: the
gymnopédie bundle's groups are named exactly `spine`, `los`,
`eng_montreal_les_editions_outremontaises_2006`, `eng_transcription_2012`,
`satie_gymnopedie1_coleman` and `satie_gymnopedie1_pfaul`, in timeline order.

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
Positions are names, not ids: page and accolade ids are minted from the
loader's minter when the edition is built, and the gymnopédie test asserts
that each is a minted `dgt_…` and that all are distinct.  Accolades are found by the half-span rule — within a page, a
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
the page image containing it; it ends at the edition's length, so that a
manifest can hold it (a `PiecewiseMap` of `ConstantMap`s).

Graphic events are interval events from `upper_left_x` to `lower_right_x`, with
the integer `bbox` struct, the page `file_name` and `position_in_group` as
fields, so a single event resolves to its image without walking the hierarchy.

Track events are instants at `start_time` seconds.

### Claim-count arithmetic

Every projected event contributes exactly one synchronous claim between itself,
at its own coordinate, and the spine event its `event_ref` names, at its VTU
coordinate, so

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
(always ticks, always the spine's uid).  Per-layer row counts are recovered by
counting the field's `timeline_a_id` column against the loader's `los_uid` /
`edition_uids` / `track_uids`, which is what the arithmetic above is asserted
through.  The bundle's MatchClaim table (`len(bundle.match_claim_table)`)
holds the totals above.

**Every claim names its two events.** The document names spine events only, so
the loader gives each layer event an id minted from its minter at the parse
(`{event type in lower case}_…`, one batch per type in document order:
loaded with `seed=0`, animals' LOS timeline holds the 3rd to 10th ids of a
fresh `IdMinter(0)` as `rest_…` and the 11th to 18th as `syllable_…`, the
spine's and the LOS's timeline ids being the 1st and 2nd).
For animals and khomus the test builds every timeline and checks, claim by
claim: `(timeline_a_id, event_a_id)` is exactly one layer event (the set of
claimed events equals the set of layer events, children included, so none is
named twice or missed); `event_b_id` is that event's own `event_ref`; and the
anchor equals `(timeline.get_coordinate_for(event_a_id),
spine.get_coordinate_for(event_b_id))`. A graphic event lives on an accolade
of its edition and the claim names it on the edition, at its edition-wide x:
the edition's own lookup through its pages and accolades agreeing with the
claim proves the running offsets the loader computes. Provenance is shared by
the field: `Provenance(method="imported", note="<creator>
ieee1599_event_ref")`.

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
and join a second document's timelines to the first's.  All three raise the
loaders' shared single-source `ValueError` (`tests/loader/README.md`,
"Single-source loaders"), whose message names the held file, and a test drives
each of them; two paths in one call are refused before either is parsed.  `create_timeline(uid)`
builds once and returns the cached timeline,
`create_timelines(id_pattern=...)` filters by uid regex, `create_group()`
returns `None`, and `get_field(MatchClaim)` follows the selector pattern: both
`MatchClaim` and `MatchClaimField` resolve to the document's single claim
field, anything else raises `TypeError`, and calling it before `load()` raises
`RuntimeError`.

## `MapsLoader` validation logic

No corpus: the tests write a two-measure MEI score and the MAPS JSON for it
into `tmp_path`, so every expected value below is read off the fixture by hand.
The score is loaded with `VerovioLoader` (the test module is skipped without
`verovio`).

### The score

Two staves, one 9/8 measure followed by a second, so a measure lasts 9/2
quarters and onsets fall on halves of a quarter: what proves that the score
side is the score's own exact position, not a float. Attacks, in quarters:

| quarter | staff 1 | staff 2 |
|---|---|---|
| 1/2 | chord `ch1a`, `ch1b`, `ch1c` | `b1` |
| 3/2 | `n2` | `b2` |
| 3 | `n3` (tie start) | `b3` |
| 9/2 | `n4` (tie end) | `b4` |
| 6 | `n5` | |
| 13/2 | `n6` | |
| 15/2 | `n7` | `b5` |

Both staves open with an eighth rest, so the first attack is at 1/2 and a
quarter before it exists on the score. `n4` continues `n3` and is not an
attack: the MAPS file has no row for it, and nothing in the loader needs one.

### The MAPS file

Seven rows, one per attack time, every onset a dyadic number of seconds whose
product with 44100 is a whole sample:

| obs_num | obs_mean_onset (s) | xml_id | sample at 44100 Hz |
|---|---|---|---|
| 1 | 0.5 | `ch1a`, `ch1b`, `ch1c`, `b1` | 22050 |
| 2 | 1.25 | `n2`, `b2` | 55125 |
| 3 | 2.0 | `n3`, `b3` | 88200 |
| 4 | 3.5 | `b4` | 154350 |
| 5 | 4.25 | `n5` | 187425 |
| 6 | 4.5 | `n6` | 198450 |
| 7 | 5 | `n7`, `b5` | 220500 |

Row 7's onset is written as the JSON integer `5`, the one way a JSON number
can differ from a float, to show it is accepted and read as 5.0 seconds.

### The file as trompa-align writes it

trompa-align's writers (`scripts/trompa_align.py`, `scripts/trompa-align.R` in
https://github.com/trompamusic/trompa-align) state a note on one side only in
two ways: a score note the performance misses gets onset `-1` (the
note-alignment file marks the missing performance note `*`, read as `-1`),
grouped like any other onset and so sorted first; a performed note not in the
score gets the id `trompa-align_inserted_` + its spelled pitch, `#` written
`s`, appended after the sorted rows; the Python writer gives that id as a bare
string. The second fixture follows that shape, eleven rows:

| obs_num | obs_mean_onset (s) | xml_id | what it states |
|---|---|---|---|
| 1 | -1 | `n4` | score note not performed |
| 2–8 | the seven rows above | | as above |
| 9 | 5.5 | `n7` | `n7` played a second time |
| 10 | 2.75 | `"trompa-align_inserted_Cs4"` (bare string) | performed note not in the score |
| 11 | 4.75 | `trompa-align_inserted_Cs4` (in a list) | a second one of that pitch |

`n4` is the tie end, which a performance never re-attacks: a natural note to
be missing. Row 9 repeats `n7` (a repeat played against an unrepeated
score); rows 10 and 11 share one id, as two inserted notes of one pitch do.
trompa-align itself keeps one row per score id (the closest candidate), so
row 9 stands for a file another tool wrote or edited; the loader keeps it as
stated all the same.

### What is asserted, and why it proves the loader

* **Rows verbatim.** `rows` holds the seven dicts as the file states them, and
  `len(loader) == 7`.
* **One claim per listed id.** 4 + 2 + 2 + 1 + 1 + 1 + 2 = 13 claims, in row
  order and within a row in the file's id order. A loader making one claim per
  row would give 7; one per chord or per distinct score position would give 7
  as well, so the count alone tells them apart.
* **Both sides of every claim are events.** The recording holds no event of
  its own, so each listed id of a row at a non-negative onset becomes a
  `Projection` added to the performance timeline, and the claim joins the
  score note to it. For each claim: the score timeline's id and the event's
  `xml:id` as `event_a_id`; the score coordinate, a `Fraction` in quarters,
  equal to the table above (`Fraction(1, 2)` for all four ids of row 1,
  `Fraction(15, 2)` for row 7); the performance timeline's id and
  the Projection's minted ID (`projection_…`) as `event_b_id`; the performance coordinate **on the performance timeline's
  own axis**: on the samples timeline the integer sample of the table (the
  onset converted through `SamplesToSeconds(44100)` before the claim
  exists), on the seconds timeline the onset as a float. A claim keeping
  seconds on a samples timeline, or naming no event, fails.
* **The Projections** (both timelines): the performance timeline holds
  exactly 13 events, one minted `projection_…` per claim, in claim order, each of
  event type `Projection`, its `projection_of` `{timeline_id: <score id>,
  event_id: <xml:id>}` of the score note it stands for, and its start the
  same sample (or seconds) as its claim.
* **Through the bundle, on a samples timeline** (`DiscretePhysicalTimeline`
  in samples with `SamplesToSeconds(44100)`; the bundle is `create_bundle`'s): `get_matchstamp_at(q)` on the
  score gives, on the performance, the integer sample of the table for each
  anchor quarter, with `is_interpolated` false. At quarter 15/4, halfway
  between the anchors at 3 (88200) and 9/2 (154350), the WarpMap gives 121275,
  with `is_interpolated` true.
* **Through the bundle, on a seconds timeline**: the same queries give the
  onsets themselves (0.5 … 5.0) and 2.75 halfway.
* **Before the first anchor**: quarter 1/4 lies on the score but before the
  first anchor (1/2); under the bundle's default `omit` policy the stamp holds
  the score and its one child alone, so the performance does not resolve. The
  child is the score's one child, the movement `mdiv1` (named `I`, so it is a child) holding the
  notes, at offset 0 over
  the whole score: it covers 1/4, so the score's offsets carry the query onto
  it at 1/4.
* **Every stated row, kept** (second fixture). `rows` is the eleven dicts
  verbatim, the bare string included. The claims are 17 = 1 + 13 + 1 + 2, in
  row order:
  * claim 1, a NOMATCH from the score: not synchronous, no anchor, timeline
    a the score and b the performance, `event_a_id` `n4`, `event_b_id` the
    NOMATCH sentinel `NOMATCH_SENTINEL` (the claim holds no
    coordinate: the note's position is its own on the score; a loader
    dropping the row would give 16 claims; one placing it at -1 s would give
    an anchor);
  * claims 2–14, the thirteen synchronous claims of the first fixture;
  * claim 15, synchronous, `n7` at quarter `Fraction(15, 2)` and 5.5 s: the
    repeated id gives its own claim, not a refusal and not a merge;
  * claims 16 and 17, NOMATCH from the performance: timeline a the
    performance and b the score, `event_a_id` two minted `projection_…`,
    `event_b_id` `NOMATCH_SENTINEL`,
    Projections standing for no score note (`projection_of`
    null) at 2.75 and 4.75 seconds (the Projections' `start`, as floats; the
    claims hold no coordinate). The file
    gives both notes one id, which can name neither event, so the timeline
    mints theirs and the claim keeps the file's id as `event_a_name`
    (`trompa-align_inserted_Cs4`). The performance holds 16 Projections.
* **A single inserted note keeps its id.** Adding one inserted row (2.75 s)
  to the first fixture on the samples timeline gives a last claim naming the
  Projection `trompa-align_inserted_Cs4` itself (no name), at sample 121275,
  `projection_of` null: an id the file gives once identifies one note.
* **The kinds are logged.** One INFO record names the file and the counts,
  14 score notes at a time of the recording, 1 not performed, 2 performed
  notes not in the score; one WARNING names `n7` in rows 8 and 9. The exact
  messages are asserted, so a wrong count shows.
* **`create_bundle`** builds what users would otherwise assemble by hand: the
  bundle is named after the file (`stated`), holds the score in group
  `score` and the performance in group `performance`, and carries the 17
  claims of `create_match_claims` (made again on the same recording: they
  state the same, see the next item). On a samples
  timeline it answers the same
  anchors as the hand-built bundle of the first fixture (quarter 3/2 →
  sample 55125, 15/4 → 121275 interpolated): the NOMATCH claims add no
  anchor, so the WarpMap is unchanged by them.
* **The bundle's package is written and read back.** The score first holds
  a Projection of its own, added while it belongs to no bundle (at quarter
  3/2, standing for no event). `create_bundle` on the second fixture with a
  samples timeline joins both timelines before it makes the claims; the
  performance's 16 Projections (claims 2–15 and the inserted notes' claims
  16 and 17) hold 16 distinct minted IDs, none the score's. `to_file` writes
  the package to `tmp_path`; its Projections table holds the 17 IDs, the
  score's own on the score and the 16 others on the performance;
  `AlignmentBundle.from_file` reads it back with the 17 claims stating the
  same events and anchors.
* **A second call reuses every Projection.** The loader makes its claims
  through the library's one projection operation, which keeps one Projection
  per score note and position on the performance timeline. Calling
  `create_match_claims` twice on the second fixture with one samples
  timeline gives claims stating the same events and anchors both times
  (the 16 Projections named alike), and the performance
  timeline holds the same 16 rows after the second call as after the first.
  The same holds with row 11 left out, so that the inserted id is given once
  and kept: 15 rows, and both calls' last claim names
  `trompa-align_inserted_Cs4` (the second time as the event the timeline
  holds). Before, a second call added every Projection again and refused the
  kept inserted id.
* **A second reading in one run mints the same IDs.** The claims and the
  Projections are minted from the loader's minter, seeded from the file
  read, never from the minters the process keeps for objects built in code,
  which count on from one build to the next. Two fresh `MapsLoader`s reading
  the second fixture one after the other, each onto the same score and a
  fresh seconds timeline, give equal claim IDs, in order, and equal
  Projection IDs on the two performances; the 17 claim IDs are distinct.

### Refusals

Each refusal raises before anything is created; each test changes one thing
in an otherwise valid fixture, so the refusal can only come from that change.

* **An id not on the score**: a row naming `zz9` → `KeyError` naming the id
  and the pair it is in (`Pair 8: no event with id 'zz9' on timeline
  '<score id>' or its children ...`: the pairs count the listed ids of rows
  1–4 from 0); the performance timeline holds no event afterwards, though
  rows 1–3 were read first (every lookup runs before a Projection is
  added).
* **No path to seconds**: a samples timeline without a C-Map, and a quarters
  timeline without one → `ValueError`.
* **Malformed rows** (`load`): `obs_num` not consecutive from 1 (`1, 2, 4`),
  a missing key, an onset that is a string or a boolean, an empty `xml_id`
  list, an empty id in a list or as a bare string, a top level that is not a
  list → `ValueError` naming the row. None of these is a row trompa-align
  writes.
* **An inserted id at a negative onset** → `ValueError` naming the row: it
  states a note on neither the score nor the recording, which no writer
  means.
* **A second source** → the loaders' shared single-source `ValueError`
  (`tests/loader/README.md`, "Single-source loaders"), naming the held file:
  one MAPS file aligns one performance. The held rows stay; after `clear()`
  the loader reads the other file.
