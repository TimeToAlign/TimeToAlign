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
| `thoresen_segment_claims` | `list[MatchClaim]` | 5 interval claims mapping DGT1↔DGT2 segments |
| `dgt1_bundle` / `dgt2_bundle` | `GraphicalBundle` | Full graphical bundles (requires PyMuPDF) |

### ID Reset (`autouse=True`)

The `reset_ids` fixture in `conftest.py` has `autouse=True` and resets the anchor and claim ID generators before every test. Group and bundle ids are minted statelessly (`group_N`, `bundle_N`, N the smallest free number in the scope), so they need no reset.

### Migration Note

Previously, `dgt1_timeline`, `dgt2_timeline`, `audio_timeline`, `thoresen_segment_claims`, and the DGT1/DGT2 coordinate constants were duplicated across `test_thoresen_poc.py`, `test_graph.py`, and `test_matchline.py`. These have been consolidated into `conftest.py` so that all test files share a single source of truth.

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
   timelines that are only reachable through a bridge timeline, and only after
   every derived-unit claim anchor is converted explicitly to its timeline's
   native unit before WarpMap construction.
2. A coordinate below the first alignment anchor produces **no negative
   coordinate under any policy**, and the three policies produce exact,
   distinct results.

### Synthetic Fixture Topology

Two bundles are built, then merged with `AlignmentBundle.from_bundles`, then
bridged with `create_match_claims`:

- **Bundle A — a star in the per-claim list.** `a1` (score, quarters, len 100),
  `a2` (seconds, len 100), `a3` (seconds, len 100), each in its own group.
  Cross-group `MatchClaim`s at `a1` coordinates `{0, 50, 100}`:
  `a2 = a1` and `a3 = a1 / 2`. A query at `a1 = 50` reaches `a2 = 50`,
  `a3 = 25` by exact anchor.
- **Bundle B — a WarpMap-able columnar `MatchClaimField`.** `b_bridge`
  (samples, len 1000) carrying a `ScalarMap(0.01, samples→seconds)` C-Map,
  plus `b1` (samples, len 3000) and `b2` (samples, len 3000), each in its own
  group. The field holds native-samples instant anchors:

  | `b_bridge` (samples) | `b1` (samples) | `b2` (samples) |
  |---|---|---|
  | 200 | 100 | 300 |
  | 400 | 900 | 600 |
  | 600 | 1700 | 900 |
  | 800 | 2500 | 1200 |

  The field's `b_bridge` hull is therefore `[200, 800]`.
- **Bridge A↔B.** A unit-bearing claim anchors `a1 = 50 quarters` to
  `b_bridge = 5 seconds`. The bridge C-Map converts the latter to exactly
  `500 samples` before its WarpMap is built. A separate mixed-unit assertion
  combines native-sample and seconds anchors on the same timeline and checks
  their exact normalized source coordinates; no range or magnitude inference
  participates. `TimeUnit.number` remains the explicit unitless form and is
  interpreted in the referenced timeline's native unit.
  Because `b_bridge`'s native unit is samples, the anchor is recorded as
  `b_bridge = 5` *samples* — a value that is really 5 seconds (the derived
  unit), the same "seconds-on-a-samples-timeline" shape as the specimen.

### Exact Expected Values

**Union query — `merged.get_matchstamp_at(50, "a1:clt_1")` (default `omit`):**

`is_interpolated is False` (the query carries an exact anchor); 6 timelines,
no negatives:

| timeline | coordinate | how |
|---|---|---|
| `a1:clt_1` | 50 | query |
| `a2:cpt_1` | 50 | exact anchor |
| `a3:cpt_2` | 25 | exact anchor |
| `b_bridge:dpt_1` | 5 | exact bridge anchor (unaltered) |
| `b1:dpt_2` | 1300 | reconcile 5 s → 500 samples, warp → 1300 |
| `b2:dpt_3` | 750 | reconcile 5 s → 500 samples, warp → 750 |

`b_bridge = 5` is out of the field hull `[200, 800]`; its C-Map inverse
reinterprets 5 seconds as 500 samples (in hull), and `np.interp(500, …)` gives
`b1 = 1300`, `b2 = 750`.

**Parity.** `b_bundle.get_matchstamp_at(500, "b_bridge:dpt_1")` (the reconciled
native coordinate) yields `b1 = 1300`, `b2 = 750` — identical to the union's B
portion for the transferred timelines.

**Out-of-support query — `merged.get_matchstamp_at(50, "b_bridge:dpt_1")`** (50
samples, below the hull; the query's own coordinate is never reconciled):

| policy | timelines | `b1:dpt_2` | `b2:dpt_3` |
|---|---|---|---|
| `omit` | 1 (`b_bridge` only) | — | — |
| `clamp` | 3 | 100 (`warp(200)`) | 300 (`warp(200)`) |
| `extrapolate` | 3 | 0 (`warp(50) = -500`, floored) | 75 (`warp(50)`, kept) |

`b_bridge = 50` stays present and unaltered under every policy. `a1/a2/a3` are
unreachable from `b_bridge` (the single bridge pair cannot form a WarpMap, which
needs at least two anchors), which keeps the out-of-support counts exact.

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

Every assertion uses exact `Coordinate`, `Interval`, `Fraction`, dictionary, and
ordering expectations; no tolerance or approximate comparison is used.

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

**Design:** AlignmentAnchor is a **pure coordinate pair** — a neutral record associating one coordinate on timeline A with one coordinate on timeline B. It contains no claim semantics (`is_synchronous`, `is_explicit`, `id` fields were removed). Claim semantics live exclusively on `MatchClaim`.

`AlignmentAnchor` is a **frozen pydantic v2 `BaseModel`** whose coordinates are
unit-bearing `Coordinate` values. Tests cover both the public scalar contract and
the claim-specific Arrow projection: coordinate values retain exact `Fraction`
representations, each side retains its per-row unit, and mixed-unit anchors survive
field storage and materialisation without changing their numeric values.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_basic_creation` | Anchor stores two timeline IDs and two required `Coordinate` scalars (no raw-float coercion or semantic flags) |
| `test_connects` / `test_connects_both` | Query methods correctly identify connected timelines |
| `test_get_coordinate_for` | Coordinate retrieval preserves the exact value and unit |
| `test_from_dict_roundtrip` | Nested coordinate dictionaries round-trip exactly (no legacy flat-float shape) |
| `test_fraction_roundtrip` / mixed-unit field tests | `Fraction(7, 3)` and independent quarters/seconds units survive `MatchClaimField` Arrow storage and materialisation |
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

**Design:** MatchClaim has `timeline_a_id` and `timeline_b_id` as **top-level fields** (not derived from anchors). Anchors are `Optional` — only synchronous claims have them. Four case-specific constructors:

| Constructor | Case | Synchronous | Anchors |
|-------------|------|-------------|---------|
| `from_events()` | Two timed events correspond | Yes (default) | Auto-built from event coordinates |
| `from_projection()` | Event projected onto timeline | Yes | Auto-built from event + target coord |
| `nomatch()` | Event has no equivalent | No | None |
| `implicit()` | Generated by MatchGraph extension | Yes | From given coordinates |

**Constructor history:** Legacy constructors `instant()` and `interval()` were removed. All 72 call sites across 6 test files and `table_schema.py` were migrated to direct `MatchClaim()` construction with explicit `AlignmentAnchor` objects.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_instant_creation` | Single anchor → `is_interval == False`, timeline IDs are top-level |
| `test_interval_creation` | Two anchors → `is_interval == True` |
| `test_mismatched_anchors_raises` | **Critical invariant**: Start and end anchors must connect same timeline pair |
| `test_synchronous_requires_anchor` | `__post_init__` rejects synchronous claims without anchors |
| `test_non_synchronous_rejects_anchors` | `__post_init__` rejects non-synchronous claims with anchors |
| `test_get_coordinates_for_interval` | Can retrieve both start and end coords for each timeline |
| `test_from_events_constructor` | `from_events()` builds anchor from event coordinates |
| `test_nomatch_constructor` | `nomatch()` has no anchors, `is_synchronous=False` |
| `test_implicit_constructor` | `implicit()` records `source_claim_id` for traceability |

### NOMATCH Coordinate Preservation & Repr

A NOMATCH claim records the *unmatched source-side coordinate* in a
`source_coordinate` field. Synchronous claims leave it `None` because their
coordinate already lives on the anchor (minimal-schema: the coordinate is
stored once, never duplicated). `nomatch()` extracts it from
`event["start"]`; `to_dict()`/`from_dict()` round-trip it so a serialised
NOMATCH claim reconstructs identically.

The `__repr__` of a NOMATCH claim shows that coordinate next to the source
timeline and tags the claim with the flag word `NOMATCH` (not the internal
field name "non-synchronous"). When no source coordinate is known the repr
falls back to the bare timeline id.

| Test | Validates |
|------|-----------|
| `test_repr_non_synchronous` | NOMATCH claim repr carries the `NOMATCH` flag and never the literal "non-synchronous" |
| `test_repr_nomatch_with_coordinate_exact` | Exact string `MatchClaim(score:clt_1@188.8 <-> perf:Chopin_Ashkenazy [NOMATCH])`; `source_coordinate.value == 188.8` with the source unit |
| `test_repr_nomatch_without_coordinate` | No `start` in the event ⇒ `source_coordinate is None` and repr drops the `@coord` segment |
| `test_nomatch_source_coordinate_roundtrip` | `from_dict(to_dict())` preserves `source_coordinate` and equals the original |
| `test_repr_synchronous_instant_unchanged` | Regression guard: a synchronous instant repr is byte-for-byte unchanged and its `source_coordinate is None` |

These assertions are exact strings (ZERO TOLERANCE), pinning both the new
NOMATCH form and the unchanged synchronous form.

### The Mismatch Test in Detail

```python
def test_mismatched_anchors_raises(self):
    start = AlignmentAnchor(timeline_a_id="tl1", ..., timeline_b_id="tl2", ...)
    end = AlignmentAnchor(timeline_a_id="tl1", ..., timeline_b_id="tl3", ...)  # Different!

    with pytest.raises(ValueError, match="must connect same timelines"):
        MatchClaim(timeline_a_id="tl1", timeline_b_id="tl2",
                   start_anchor=start, end_anchor=end)
```

This prevents creating semantically invalid claims. An interval match between `(tl1, tl2)` and `(tl1, tl3)` would represent... what? The constraint catches this at construction time.

---

## Agent & MatchMetadata Tests (`test_anchors.py::TestMatchMetadata`)

### What We're Validating

A match records who authored it and how confident that author is. The schema
was **slimmed**: `MatchMetadata` is now exactly `{agent: Agent, certainty:
float}`. The previous free-form fields — `decision_criteria`, `created_at`,
`notes`, `algorithm_params` — were **deleted entirely** (no compat shim).

`Agent` is a new frozen pydantic scalar `{name: str, type: AgentType,
identifier: str}`. `AgentType` is a two-member `FancyStrEnum`
(`human`, `software`); the `identifier` is a URI for a human and a version
string for software. Both `Agent` and `MatchMetadata` are frozen pydantic v2
models with `to_dict()` / `from_dict()` round-trips (the `type` member stores
as its string value and is coerced back to `AgentType` on read).

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_agent_type_members` | `AgentType` has exactly `human` and `software` |
| `test_basic_creation` | `MatchMetadata{agent, certainty}`; `certainty` defaults to `1.0` |
| `test_agent_roundtrip` | `Agent.from_dict(agent.to_dict())` reconstructs identically (`type` as string) |
| `test_metadata_roundtrip` | `MatchMetadata.from_dict(meta.to_dict())` reconstructs identically (nested `Agent`) |
| `test_certainty_validation` | Certainty must be in [0, 1] (message: `Certainty must be in [0, 1], got ...`) |
| `test_certainty_boundaries` | Boundary values (0.0, 1.0) are valid |
| `test_frozen_model` | Immutability enforced (frozen pydantic model) |

---

## MatchClaimField Tests (`test_match_claim_field.py`)

### What We're Validating

`MatchClaimField` is a **genuine `SemanticField[MatchClaim]`** — the Field
paired with the `MatchClaim` scalar. It holds a very large set of pairwise
alignment claims in a **single derived struct column** instead of one frozen
`MatchClaim` object per claim. Dense audio-to-audio alignments push this into
the millions of claims, so the store must build vectorized (never one Python
object per row) and materialise individual `MatchClaim` instances only on
demand.

Because `MatchClaim` is now a pydantic scalar, the field's struct schema is
**derived from the scalar** by `derive_arrow_struct(MatchClaim)` and cached on
the class as `MatchClaimField.pa_schema` — exactly like every other paired
`SemanticField` (`NoteField`, `CoordinateField`, …). The `MatchClaim` scalar
carries no `@data_shaped` methods, so the parity check requires no vectorized
mirrors.

Shared provenance is a single `MatchMetadata | None` held **once at field
level** (a Python attribute, `self._metadata`) and **injected on read**. This
mirrors how `CoordinateField` carries its `unit` outside the data: the struct
column's `metadata` sub-field is left null in the bulk path, and `__getitem__`
injects the field-level metadata into each materialised `MatchClaim`. This
keeps the store compact (one struct column, no per-row metadata) while the field
remains a real SemanticField. Each anchor coordinate is stored as a nested
`{value, numerator, denominator, unit}` struct so exact rational values and
per-row units survive materialisation.

### Scope (v1) — and what raises

The store holds **synchronous instant pairwise claims only**: each row has
`is_synchronous is True`, a `start_anchor`, and `end_anchor is None`. Two
categories are deliberately out of scope and `from_claims` raises `ValueError`
on either:

- **NOMATCH claims** (`is_synchronous is False`, no anchors).
- **Interval claims** (carry an `end_anchor`).

### Internal schema (the derived `MatchClaim` struct)

The backing table has **one** column, `match_claim`, whose type is exactly
`MatchClaimField.pa_schema == derive_arrow_struct(MatchClaim)`. The vectorized
`from_columns` fills the `timeline_a_id` / `timeline_b_id` top-level string
sub-fields and the `start_anchor` sub-struct
(`{timeline_a_id, coordinate_a, timeline_b_id, coordinate_b}`), sets
`is_synchronous` / `is_explicit` to `True`, and leaves every other sub-field
(`end_anchor`, `metadata`, event ids/names, `source_coordinate`,
`source_claim_id`, `id`) null. It uses `pa.StructArray.from_arrays` and never
materialises a Python `MatchClaim`. The `coordinate_a` / `coordinate_b` values
read back through the `start_anchor` sub-struct; `timeline_ids` reads the two
top-level id sub-fields vectorized.

### Canonical gold vector

```python
agent = Agent(name="test", type=AgentType.software, identifier="manual")
meta = MatchMetadata(agent=agent)
f = MatchClaimField.from_columns(
    timeline_a_ids=["A", "A", "B"],
    timeline_b_ids=["B", "C", "C"],
    coordinate_a=[0.0, 0.0, 1.0],
    coordinate_b=[10.0, 20.0, 21.0],
    unit_a=TimeUnit.quarters,
    unit_b=TimeUnit.seconds,
    metadata=meta,
)
```

### Key Evidence (exact, zero-tolerance)

Claim counts and coordinate values are **identical** to the prior layout; only
the metadata shape and the field's internal column layout changed.

| Test | Validates |
|------|-----------|
| `test_from_columns_length` | `len(f) == 3` |
| `test_is_semantic_field` | `issubclass(MatchClaimField, SemanticField)` |
| `test_pa_schema_is_derived` | `MatchClaimField.pa_schema == derive_arrow_struct(MatchClaim)` |
| `test_table_single_struct_column` | one column `match_claim` whose type equals `pa_schema` |
| `test_timeline_ids` | `f.timeline_ids == {"A", "B", "C"}` |
| `test_getitem_first_row` | `f[0]` is a `MatchClaim`, `A`↔`B`, synchronous instant, coords 0.0/10.0, metadata identity `is meta` |
| `test_getitem_negative_index` | `f[-1]` → `B`, `coordinate_b == 21.0` |
| `test_getitem_out_of_range` | `f[3]` / `f[-4]` raise `IndexError` |
| `test_connecting` | `f.connecting("C")` has `len == 2`; rows are exactly `{("A","C"), ("B","C")}` |
| `test_filter_timeline_ids` | `f.filter(timeline_ids={"A"})` has `len == 2` (rows 0 and 1) |
| `test_filter_timeline_id_equals_connecting` | `filter(timeline_id=...)` matches `connecting(...)` |
| `test_filter_both_none_copies` | `f.filter()` returns all rows |
| `test_filter_and_combination` | `timeline_id` AND `timeline_ids` combine with logical AND |
| `test_to_claims` | `len(f.to_claims()) == 3`, all synchronous instants |
| `test_iter` | iteration yields 3 synchronous-instant claims |
| `test_roundtrip_from_claims` | `from_claims(f.to_claims()).table` equals `f.table` (`.equals`) |
| `test_roundtrip_from_dict` | `from_dict(f.to_dict()).table` equals `f.table`; metadata preserved |
| `test_getitem_injects_metadata` | a materialised claim carries the field-level metadata (struct row is null there) |
| `test_from_claims_rejects_nomatch` | a `MatchClaim.nomatch(...)` raises `ValueError` |
| `test_from_claims_rejects_interval` | an interval claim (both anchors) raises `ValueError` |
| `test_from_claims_adopts_common_metadata` | shared per-claim metadata becomes field metadata when `metadata=None` |
| `test_from_claims_mixed_metadata_stays_none` | divergent per-claim metadata → field metadata stays `None` |
| `test_from_columns_length_mismatch_raises` | unequal column lengths raise `ValueError` |
| `test_empty_field` | zero-row field: `len == 0`, `timeline_ids == set()`, `to_claims() == []` |
| `test_repr` | `repr(f) == "MatchClaimField(claims=3, timelines=3)"` |
| `test_repr_html_contains_summary` | `_repr_html_()` is non-empty and reports the claim/timeline counts |
| `test_top_level_export` | `MatchClaimField` importable from `timetoalign` and `timetoalign.alignment` |
| `test_translator_strenum_to_string` | `derive_arrow_struct(MatchMetadata)` has a string `type` sub-field inside the `agent` struct |

### Vectorized query primitives

Every bundle-level claim query is composed from primitives that live **on the
field**, so the answer is computed over Arrow columns and no `MatchClaim` is
built unless the caller asks for one. These are the primitives, tested against
the canonical gold vector above (three rows, `A↔B`, `A↔C`, `B↔C`).

| Test | Validates |
|------|-----------|
| `test_at_matches_a_side` | `f.at("A", 0.0)` has `len == 2` (rows 0 and 1) |
| `test_at_matches_b_side` | `f.at("C", 20.0)` has `len == 1`, the `A↔C` row |
| `test_at_exact_equality_only` | `f.at("A", 0.0000001)` is empty — no tolerance, no nearest-value fallback |
| `test_at_unknown_timeline_empty` | `f.at("Z", 0.0)` is empty; metadata still carries over |
| `test_at_on_empty_field` | `at` on a zero-row field returns a zero-row field |
| `test_filter_id_pattern` | `filter(id_pattern=r"^A$")` has `len == 2` (rows 0 and 1) |
| `test_filter_id_pattern_no_match` | `filter(id_pattern=r"^Z")` is empty |
| `test_filter_between_order_independent` | `filter(between=("A","B"))` == `filter(between=("B","A"))`, `len == 1` |
| `test_filter_within_requires_both_sides` | `filter(within={"A","B"})` has `len == 1` — `A↔C` and `B↔C` are excluded because only one side is in the set |
| `test_filter_synchronous_only_is_noop` | `filter(synchronous_only=True)` has `len == 3` (class invariant) |
| `test_filter_nomatch_only_is_empty` | `filter(nomatch_only=True)` has `len == 0` (class invariant) |
| `test_filter_mutually_exclusive_raises` | both flags together raise `ValueError` |
| `test_filter_combines_with_and` | `filter(timeline_id="A", between=("A","C"))` has `len == 1` |
| `test_connects_groups_true` | `connects_groups({"A"}, {"C"})` is True |
| `test_connects_groups_order_independent` | `connects_groups({"C"}, {"A"})` is True |
| `test_connects_groups_false` | `connects_groups({"A"}, {"Z"})` is False |
| `test_connects_groups_empty_field` | False on a zero-row field |
| `test_max_coordinate_a_side` / `_b_side` | `max_coordinate("A") == 0.0`, `max_coordinate("C") == 21.0` |
| `test_max_coordinate_spans_both_sides` | `max_coordinate("B") == 10.0` (b-side row 0 = 10.0 beats a-side row 2 = 1.0) |
| `test_max_coordinate_unknown_timeline` | `max_coordinate("Z") is None` |
| `test_coordinate_pairs` | the four bulk lists are exactly `["A","A","B"]`, `["B","C","C"]`, `[0.0,0.0,1.0]`, `[10.0,20.0,21.0]` |
| `test_coordinate_pairs_empty` | four empty lists on a zero-row field |

`within` is the vectorized primitive behind domain- and unit-restricted
queries. `include_domains` / `include_units` cannot live on the field itself —
they need each timeline's domain and unit, which the store does not hold — so
the bundle resolves them into the set of timeline IDs that pass and hands that
set to `within`. The **both sides must be in the set** semantics is not a
choice: it is what `ClaimFilter.matches_claim` already applies to the Python
list, and parity demands the two agree.

---

## Claim-Store Parity (`test_claim_store_parity.py`)

### What We're Validating

`AlignmentBundle` has two claim stores — the per-claim Python list
`cross_group_claims` and the columnar `cross_group_claim_fields`. The contract
is that **which store a claim lives in is invisible to every reader**: a bundle
holding a set of claims in the list and a bundle holding the *same* claims in a
`MatchClaimField` must answer every public getter identically. Only cost
differs.

This matters because a loader may legitimately choose either store, and a
columnar loader must not silently lose query capability. A reader that consults
only one store is a bug, and this suite is what catches it.

### Fixture Topology

Three timelines, one group each (so every claim is cross-group), spanning two
domains so that domain/unit filters bite:

| Timeline ID | Class | Unit | Domain |
|------------|-------|------|--------|
| `score:clt_1` | `ContinuousLogicalTimeline` | `quarters` | logical |
| `perf:cpt_1` | `ContinuousPhysicalTimeline` | `seconds` | physical |
| `perf:cpt_2` | `ContinuousPhysicalTimeline` | `seconds` | physical |

The claims name these timeline IDs, the ids the bundle registers the
timelines under.

The claim topology is **complete pairwise at three aligned instants** — every
pair claims at every instant, 3 pairs × 3 instants = **9 synchronous instant
claims**:

| Instant | `score:clt_1` | `perf:cpt_1` | `perf:cpt_2` |
|---------|--------------|-------------|-------------|
| 0 | 0.0 | 1.0 | 2.0 |
| 1 | 4.0 | 3.0 | 5.0 |
| 2 | 8.0 | 6.5 | 9.0 |

Each instant is therefore one connected component spanning **all three**
timelines, so `from_graph=True` collapses 9 pairwise rows into 3 cross-section
rows — a non-trivial collapse, not an identity.

The same nine claims are built twice by a factory (fresh objects each time, so
the two bundles never share a claim's `set_bundle` association) and handed to:

- `bundle_list` — via `add_match_claims(claims)`
- `bundle_field` — via `add_match_claim_field(MatchClaimField.from_claims(claims))`

### Comparison Rule

Claim **identity** is not comparable across the two constructions: claims
materialised out of a field are rebuilt from the struct row, so their generated
`id` differs from the original objects', and the field carries provenance once
at field level rather than per claim. What must be identical is the *alignment
content*. Every claim comparison therefore runs on the normalised key

```python
(timeline_a_id, timeline_b_id, coordinate_a, coordinate_b, is_synchronous)
```

sorted, so ordering differences between the two stores also cannot mask a
difference in content. Coordinate values are compared exactly (no tolerance) —
both paths carry the same `float` through, so anything else is a defect.

### Key Evidence (exact, zero-tolerance)

| Test | Validates |
|------|-----------|
| `test_n_cross_group_claims` | both bundles report `9` |
| `test_store_placement` | `bundle_list` has 9 list claims / 0 fields; `bundle_field` has 0 list claims / 1 field of 9 rows (proves the fixture actually tests two different layouts) |
| `test_get_match_claims_unfiltered` | both return 9 claims with equal normalised keys |
| `test_get_match_claims_filtered[…]` | one case per filter kwarg — `timeline_id`, `timeline_ids`, `id_pattern`, `between`, `synchronous_only`, `nomatch_only`, `include_domains`, `include_units`, and an unknown-UID case — equal normalised keys for each, with the exact expected count pinned per case |
| `test_get_claim_fields_row_count_matches_claims` | for every filter case, `sum(len(f) for f in bundle_field.get_claim_fields(**kw))` equals `len(bundle_list.get_match_claims(**kw))` — the vectorized accessor and the materialising one agree |
| `test_get_claim_fields_empty_on_list_bundle` | a list-only bundle returns `[]` (nothing to serve columnar) |
| `test_get_claim_fields_drops_empty_fields` | a filter matching no row yields `[]`, not a zero-row field |
| `test_get_matchstamps` | equal sorted coordinate maps, 9 stamps each |
| `test_matchstamp_table_per_claim` | both: `num_rows == 9`, `column_names == ["perf:cpt_1", "perf:cpt_2", "score:clt_1"]`, and identical `to_pylist()` — each row exactly two non-null cells |
| `test_matchstamp_table_from_graph` | both: `num_rows == 3`, same 3 columns, all 9 cells filled, rows in the documented order (instant 0, 1, 2) with the exact coordinates of the topology table above |
| `test_matchstamp_table_timeline_filter` | `timeline_filter={"score:clt_1"}` yields a 1-column table in both modes for both bundles |
| `test_are_commensurable` | all three timeline pairs are commensurable in both bundles; a non-member ID is not |
| `test_get_matchstamp_at` | at each of the three instants, both bundles return a stamp spanning all 3 timelines with the exact coordinates of the topology table |
| `test_transfer_round_trip` | `transfer(4.0, "score:clt_1", "perf:cpt_1") == 3.0` in both bundles, and the reverse direction returns `4.0` — exercising the `MatchLine` → `WarpMap` path that was dead for columnar bundles |
| `test_diagram_reports_claim_count` | `bundle.diagram()` reports `MatchClaims: 9` for both |

### Collapsed-Row Ordering

`from_graph=True` must be deterministic or the table is not comparable at all.
Components are ordered by the coordinate on the lexicographically smallest
timeline ID present in the component, then by that ID. This is a total order:
two components can never share a `(timeline_id, coordinate)` node, since
sharing one would have unioned them. `test_matchstamp_table_from_graph` pins
the resulting row order explicitly rather than sorting the result, so a change
in the ordering rule fails the suite instead of passing silently.

### Scale sanity (vectorized path, no Python objects)

`test_scale_builds_columnar` builds a field of **100 000** rows via
`from_columns` with deterministic inputs (`coordinate_a = [float(i) for i in
range(100_000)]`), asserting `len == 100_000` and that `f[50_000]` materialises
to the correct coordinate (`50000.0`). Construction completes in well under a
second (≈30 ms locally) because `from_columns` builds Arrow arrays directly and
never instantiates a `MatchClaim`. This is the proof that the columnar path
scales to the dense-alignment workload.

---

## AlignmentBundle Dictionary Form and Identity (`test_bundle_dict.py`)

### What We're Validating

A bundle is its timelines and its groups. Its dictionary form writes every
root timeline once (each exactly `Timeline.to_dict()`, sorted by id), the
groups as `Group` objects naming their members by id (sorted by id), and every
group's member order under `graphical.order`. MatchClaims are not part of the
form and are excluded from every assertion here.

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
   every member. Claims are excluded: `cross_group_claims` is not consulted. The
   rebuilt bundle must also hold exactly the same participants and the same
   group membership.
2. **Exact expected forms.** One direct construction pins the whole
   dictionary: two groups and an ungrouped root, a group member that is a
   child (its root is written, the child id is the member), the `group_N`
   ids, `support_policy` as its string value, and the member order as added.
3. **Identity.** A fresh bundle is `bundle_1`; `as_group` names the new group
   and its id is minted `group_1`, `group_2`; a child added to a group brings
   its root, ungrouped; a child added ungrouped registers only its root (an
   ungrouped participant is always a root), and the round trip restores
   exactly that participant set; three loader-minted `cpt_1` timelines join
   one bundle as `cpt_1`, `cpt_2`, `cpt_3`; a timeline already in one group is refused by
   another; a given timeline id held by another timeline (of any tree in the
   bundle, descendants included) raises `ValueError` and leaves the bundle
   unchanged; a minted id held there is minted again. A refused newcomer keeps
   its ids: every collision check — the timeline ids and the group id —
   runs before any id is minted again, so a minted `clt_1` that admission
   would have re-minted is still `clt_1` after its group or its root was
   refused. A given group id refuses a timeline that carries it; a *minted*
   group id that a newcomer's given timeline id takes is minted again
   (`group_1` → `group_3` beside an existing `group_2`), and the members keep
   their membership under the new id.
4. **Nesting.** A parent and its child may sit in *different* groups of one
   bundle — each group converts within itself and the child's offset relates
   them. In *one* group they are refused: `grouped_with` naming a member
   nested with the newcomer raises `ValueError` naming both ids before
   anything changes, and so does a `Group` object in `from_dict` naming both.
5. **Refusals.** `from_dict` raises `ValueError` naming the problem for an
   unknown or missing key, an unknown `support_policy`, an empty bundle id, a
   timeline written twice, a group id written twice or equal to a timeline id,
   a member naming no timeline, a timeline in two groups, a group naming a
   parent and its child, and an order naming no group or not a permutation of
   its group's members.
6. **Styling is carried, never read.** A bundle file's `graphical` may hold
   `styles` and `statements` beside `order`; they belong to the bundle, not
   to any of its groups, and the library keeps them on
   `AlignmentBundle.styling` without interpreting them. A styling block of a
   named style, a statement setting it by name and one setting a declaration
   directly survives `from_dict` → `to_dict` equal to the input, beside the
   unchanged per-group `order`, and the rebuilt bundle's `styling` shares no
   object with the input. The direct construction writes `graphical` as
   `{"order": {...}}` alone (item 2), so absent styling is never invented.
   Without `order`, every group joins in `members` order. Any other key of
   `graphical` is refused with the exact message
   `'graphical' has unknown key(s) ['frame']`, the same reader serving
   groups and bundles.

---

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

## Offset Arithmetic Tests (`../timelines/test_offset_arithmetic.py`)

### What We're Validating

**Design:** Parent–child coordinate transfer uses exact offset arithmetic
(`child_coord = parent_coord - offset`), so no floating-point drift enters a
transfer and rational offsets compose exactly.

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_exact_parent_to_child` | `_get_child_coordinate()` returns the exact result |
| `test_exact_child_to_parent` | `_get_parent_coordinate_from_child()` returns the exact result |
| `test_out_of_bounds_returns_none` | `None` for a coordinate outside the child span |
| `test_unknown_child_returns_none` / `test_unknown_child_raises_keyerror` | Unknown ids keep their established return or exception |
| `test_recursive_offset_exact` | Offsets compose through a grandchild |
| `test_recursive_fraction_offset_exact` | Rational offsets compose without float coercion |
| `test_timestamp_uses_offset_arithmetic` | A `TimeStamp` resolves a child through its offset |
| `test_timestamp_out_of_bounds_returns_none` | A stamp outside the child span raises `KeyError` for the child |
| `test_timestamp_boundary_values` | Start inclusive, end exclusive |
| `test_no_interpolation_maps_attribute` | A timeline holds no interpolation maps |

---

## MatchGraph Tests (`test_graph.py`)

### What We're Validating

The MatchGraph enforces the design principle that only synchronous claims produce graph edges, while non-synchronous claims are stored as metadata. The `extend_to_groups()` method creates `MatchClaim.implicit()` objects with `source_claim_id` traceability, and filtering supports domain/unit/timeline constraints.

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

All coordinate assertions in this file are exact `==`: the stamp-coordinate
checks in `test_extend_creates_correct_coordinates` (50.0) and
`test_two_groups_five_implicit_claims` (250.0, 100.0, 200.0) are linear
group conversions at exactly representable ratios and carry no `pytest.approx`.

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

### Key Test Classes

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestMatchStampDeterminism` | 3 | Stamp source and entry order are the same in every process |
| `TestMatchGraphNonSynchronousClaims` | 4 | Non-synchronous claims stored as metadata, no edges created |
| `TestMatchGraphGetStamps` | 4 | `get_stamps()` returns one stamp per connected component |
| `TestMatchGraphExtendToGroupsImplicitClaims` | 5 | Implicit claims created with correct coordinates and traceability |
| `TestMatchGraphExtendToGroupsFilters` | 4 | `include_timelines`, `exclude_timelines`, `include_domains`, `include_units` |
| `TestMatchGraphFilter` | 3 | `filter()` preserves non-synchronous claims for remaining timelines |
| `TestMatchStampGroupRetrieval` | 2 | Retrieves group members with the plural key getter and `timeline_ids` |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_non_synchronous_claims_stored_as_metadata` | Non-synchronous claims don't create edges; graph has 0 edges |
| `test_synchronous_and_non_synchronous_separation` | `synchronous_claims` and `non_synchronous_claims` properties partition correctly |
| `test_get_stamps_returns_one_per_component` | Connected components yield separate stamps |
| `test_get_stamps_matches_legacy_for_single_claim` | Backward compatibility with `get_match_stamps()` |
| `test_implicit_claims_have_source_id` | Implicit claims trace back to originating explicit claim |
| `test_two_groups_full_connectivity` | Core test: 1 explicit claim between TL1∈{TL1,TL4,TL5} and TL2∈{TL2,TL6} produces 5 connected timelines with correct interpolated coordinates |
| `test_include_timelines_filter` | Only specified timelines appear in extension |
| `test_include_domains_filter` | Only timelines from specified domains appear |
| `test_filter_preserves_relevant_non_synchronous` | `filter()` keeps non-synchronous claims whose timelines survive filtering |
| `test_group_coordinates_basic` | Returns group-member coordinates via `TimelineGroup.get_coordinates_at()` |

### The Two-Groups Connectivity Test

```python
def test_two_groups_full_connectivity(self):
    # Group A = {TL1, TL4, TL5}, Group B = {TL2, TL6}
    # One explicit claim: TL1@100 <-> TL2@200
    # After extend_to_groups():
    #   - TL4, TL5 get coordinates via Group A interpolation from TL1@100
    #   - TL6 gets coordinate via Group B interpolation from TL2@200
    #   - All 5 timelines connected in one component
    stamps = graph.get_stamps()
    assert len(stamps) == 1
    assert len(stamps[0].coordinates) == 5  # All timelines present
```

This is the core group-extension test: it verifies that group extension creates implicit claims with coordinates derived from the typed group retrieval API, producing a fully connected component from a single explicit claim.

---

## MatchLine Tests (`test_matchline.py`)

### What We're Validating

`MatchLine` is an ordered sequence of `MatchStamp` objects for a given source timeline. It is the bridge between `MatchGraph` and `WarpMap`. A MatchLine collects stamps, sorts them by source coordinate, exposes typed `AlignmentAnchor` values, and keeps the float interpolation feed private.

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
| `test_from_claims_with_group_extension` | `from_claims()` with group parameters adds group member coordinates (audio mapped linearly) |
| `test_from_claims_non_synchronous_excluded` | Non-synchronous claims (NOMATCH) do not produce stamps |
| `test_from_graphs_hendrix_pattern` | Four contiguous M-box graphs merged into 5 unique source coordinates (boundary deduplication) |
| `test_from_graphs_keeps_richer_stamp` | Deduplication keeps the stamp with more timelines |
| `test_thoresen_matchline` | Thoresen segment claims produce correct boundary pairs: (0,0) to (4835,4328) |

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

## AlignmentBundle Tests (`test_bundle.py`)

### What We're Validating

`AlignmentBundle` uses the `MatchLine` → `WarpMap` pipeline. The bundle lazily builds `WarpMap` objects on first cross-group `transfer()` call and caches them, invalidating the cache when `add_match_claims()` is called.

### Key Test Classes

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestCrossGroupTransfer` | 6 | `transfer()` across groups via WarpMap (direct, boundary, interpolation, reverse, non-existent path) |
| `TestIndirectTransfer` | 2 | Within-group convert then cross-group warp (indirect path) |
| `TestGroupExtension` | 2 | Claims connect score→audio; transfer propagates to midi via group membership |
| `TestGetTimestampAtCrossGroup` | 3 | `get_timestamp_at()` propagation across groups (flat, nested, prefix formats) |
| `TestCommensurabilityWithClaims` | 3 | `are_commensurable()` returns True when claims connect groups (direct + via membership) |
| `TestCacheInvalidation` | 2 | WarpMap cache cleared on `add_match_claims()` |
| `TestEdgeCases` | 6 | No claims, non-synchronous claims, single claim insufficient for WarpMap |
| `TestAddMatchClaimsAPI` | 3 | Chaining, accumulation, no-arg validation |
| `TestAddGroupWithCrossGroupTransfer` | 4 | `add_group()` + cross-group claims + transfer end-to-end |

### Key Evidence

| Test | Validates |
|------|-----------|
| `test_cross_group_transfer_direct` | Score@0.0 → Audio@0.0 via WarpMap (exact boundary) |
| `test_cross_group_transfer_interpolation` | Interior coordinates interpolated correctly |
| `test_cross_group_transfer_reverse` | Bidirectional: Audio→Score as well as Score→Audio |
| `test_indirect_transfer_through_group` | MIDI→Audio via group convert, then Audio→Score via WarpMap |
| `test_group_extension_transfer` | Claims between Score and Audio propagate to MIDI via recording group membership |
| `test_get_timestamp_at_cross_group_flat` | `get_timestamp_at()` returns coordinates for timelines in both groups |
| `test_commensurable_via_claims` | `are_commensurable()` detects cross-group connectivity through claims |
| `test_cache_invalidated_on_add_claims` | New claims clear cached WarpMaps; subsequent transfer uses updated data |
| `test_no_cross_group_claims_returns_none` | `transfer()` returns None when no claims connect the groups |
| `test_non_synchronous_claims_no_transfer` | Non-synchronous claims (NOMATCH) don't produce WarpMaps |

### Design: Lazy WarpMap Cache

```python
# Bundle maintains:
_warp_map_cache: dict[tuple[str, str], WarpMap]  # (source_group, target_group) -> WarpMap
_cache_claims_hash: int  # Invalidation key

# On transfer():
warp = self._get_or_build_warp_map(source_group_id, target_group_id)
result = warp(source_coord)
```

The cache is keyed by `(source_group_id, target_group_id)` and invalidated whenever `add_match_claims()` is called. This avoids redundant `MatchLine.from_claims()` + `WarpMap.from_match_line()` computation for repeated queries.

### `get_matchstamp_table` Conversion Columns

`get_matchstamp_table` accepts a keyword-only `conversion_maps` spec
and represents every timeline column as an `IdCoordinateField`. Numeric
conversion columns are `CoordinateField` values with the converted unit and
canonical number-type metadata; tests materialize their exact scalar values
through those semantic fields rather than comparing raw Arrow floats.
(default `False`, matching the opt-in default now shared by every
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
| `test_matchstamp_table_conversion_column_collision_qualified` | Two timelines in different groups, each with its own seconds→milliseconds map, linked by one cross-group synchronous claim: requesting the table with `conversion_maps=True` qualifies both derived columns as `"<timeline_id>:milliseconds"` rather than colliding on a bare `"milliseconds"` |

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

These tests reuse the existing `_make_cross_group_bundle()` +
`_make_linear_claims()` fixtures (no new corpus). The claims map
`score → audio` linearly (`audio = score * 0.5`), so a query at score 100
resolves to audio 50 — an exact value with no interpolation rounding. Parity is
asserted by comparing the whole result across all three call forms, plus a
pinned concrete value.

| Test class | Validates |
|------------|-----------|
| `TestGetMatchstampAtCoordinateParity` | raw / native-unit Coordinate / IdCoordinate-alone agree; C-Map conversion agrees with a native query; unavailable units and conflicting IDs ⇒ `ValueError`; non-coordinate type ⇒ `TypeError` |
| `TestGetTimestampAtCoordinateParity` | same three forms produce identical whole-dict results; error paths identical |
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
| `TestIntegrationWithClaims` | 1 | End-to-end: MatchClaim → AlignmentAnchor → MatchLine → WarpMap |
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
| `test_full_pipeline` | MatchClaim → AlignmentAnchor → MatchLine → WarpMap → forward/inverse |

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
| `test_bundle.py` | 94 | AlignmentBundle: linear alignment, a child joining with its root, cross-group transfer, timestamps, commensurability, caching, edge cases, coordinate-type parity (raw/Coordinate/IdCoordinate) |
| `test_bundle_dict.py` | — | AlignmentBundle dictionary form and identity: round trips of direct constructions and of every alignment loader's bundle, minted ids, collisions, a child bringing its root, refusals |
| `test_anchors.py` | 62 | AlignmentAnchor, MatchClaim (incl. NOMATCH coordinate preservation + repr), MatchMetadata |
| `test_graph.py` | 54 | MatchGraph operations (implicit claims, filtering, stamps); imports Thoresen fixtures from conftest |
| Robustness coverage | — | Bundle graph stamp units/context, coordinate-bearing claim factories, Fraction serialization, inferred-edge completeness, and MatchStamp container isolation/axis handling |
| `test_matchline.py` | 33 | MatchLine construction, from_claims, from_graphs, coordinate pairs, serialization; imports Thoresen fixtures from conftest |
| `test_warpmap.py` | 36 | WarpMap construction, forward/inverse, materialise (events, children, regions, type conversion), serialization, end-to-end pipeline |
| `test_filters.py` | 27 | `ClaimFilter` dataclass: exact ID, set-of-IDs, regex, between, synchronous/nomatch, combined filters, timeline-level filtering |
| `test_stamp_query_api.py` | 54 | Unified Stamp & Query API: `get_match_claims()`, `get_matchstamp_at()`, `MatchGraph.get_matchstamp()`/`split_components()`, `MatchClaim.get_matchstamp()`, MatchGraph cache, display methods (`__str__`/`_repr_html_` incl. the affordance `Try` footer on `MatchStamp`/`MatchClaim`), `transfer()` docstring fix, top-level exports |
| `test_supra_integration.py` | 13 | SUPRA piano roll workflow (partial alignment as a child) |
| `test_thoresen_poc.py` | 35 | Thoresen graphical analysis workflow; imports Thoresen fixtures from conftest |
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
| `TestClaimFilterSynchronous` | 4 | `synchronous_only` and `nomatch_only` |
| `TestClaimFilterCombined` | 4 | AND logic across multiple filters |

### Stamp & Query API Tests (`test_stamp_query_api.py`)

Tests for the Unified Stamp & Query API, using a star-topology bundle (1 score + 3 performers, 5 coordinates each, plus 1 NOMATCH claim).

| Class | Tests | Purpose |
|-------|-------|---------|
| `TestGetMatchClaims` | 8 | `AlignmentBundle.get_match_claims()`: no filters, by timeline ID, by regex, synchronous_only, nomatch_only, between, combined |
| `TestMatchGraphGetMatchstamp` | 4 | `MatchGraph.get_matchstamp()`: single component, multi-component raises, empty graph, only-NOMATCH |
| `TestMatchGraphSplitComponents` | 4 | `split_components()`: single, two components, empty graph, `n_components` property |
| `TestMatchGraphStarTopology` | 2 | Star topology (1 score + N performers): single coordinate = one component, determinism |
| `TestMatchClaimGetMatchstamp` | 4 | `MatchClaim.get_matchstamp()`: reduced stamp, NOMATCH returns None, no-bundle raises, from-graph with bundle |
| `TestMatchGraphCache` | 5 | MatchGraph cache: hit, cross-key lookup, invalidation, no-claims raises, different coordinates |
| `TestGetMatchstampAt` | 6 | `AlignmentBundle.get_matchstamp_at()`: basic, non-zero coordinate, not-in-bundle, no claims, regex filter, timeline_ids filter |
| `TestMatchStampDisplay` | 8 | `MatchStamp.__str__`/`_repr_html_`: header, entries, empty, integer formatting, valid HTML, bold anchors, greyed inferred, and typed retrieval affordance appended after the table |
| `TestMatchClaimDisplay` | 9 | `MatchClaim.__str__`/`_repr_html_`: instant, interval, NOMATCH, metadata, inferred, valid HTML, NOMATCH badge, affordance `Try` footer (`claim.get_matchstamp()`) appended after the table |
| `TestTransferDocstring` | 1 | `transfer()` docstring no longer says "primary user-facing" |
| `TestTopLevelExports` | 4 | `MatchGraph`, `MatchStamp`, `ClaimFilter` importable from top-level |

---

## TimeSkeleton Validation

This section explains **how** the upcoming tests prove the ``TimeSkeleton`` correct
under the Zero Tolerance Validation Policy: every anchor below is derived from an
actual data file and stated as an exact value with no range or approximation.

A ``TimeSkeleton`` is **one** temporal structure shared by every timeline of a work.
It composes:

- a ``SectionHierarchy`` whose leaves carry immutable ``MeasureMap`` segments;
- a ``MetricHierarchy`` of ``BeatPolicy`` objects (a beat size, plus optional bpm
  and display name);
- **flows** — ordered traversals over the atomic sections, including authored
  flows and ``Gap`` markers for inserted material.

Internally the skeleton composes an **alignment bundle**: participants attach to a
flow and carry match claims; each flow that has attachments owns an unfolded
**reference timeline** measured in quarters as exact ``Fraction`` values.

Measures follow the Measure Map standard (``id``, ``count``, ``qstamp``,
``number``, ``name``, ``time_signature``, ``nominal_length``, ``actual_length``,
``start_repeat``, ``end_repeat``, ``next``; plus ``volta``). **A measure is never
an event; a measure address is a position.** Measure-address axes (measure id
``m<n>`` or printed number) index positions; floating-measure (``fm``) axes are
``float``; quarters are exact ``Fraction``.

### Ground-truth data file

`trois-gymnopedies.measures.tsv` (Satie, *Trois Gymnopédies*) is the reference
score. Facts verified directly from the file:

| Fact | Value |
|------|-------|
| Measure rows | 203 (``mc``/``mn`` both 1..203; ``mc == mn`` on every row) |
| Time signature | ``3/4`` on every measure (``act_dur`` ``3/4`` everywhere) |
| Quarters per measure | exactly 3 (``duration_qb`` == 3.0 on every row) |
| ``quarterbeats`` column | prefix sums 0, 3, 6, …, 606 (first = 0, last = 606) |
| Whole-score span | 606 + 3 = **609 quarters** |
| Section closers (``breaks`` contains ``"section"``) | measure ``mc 78`` and ``mc 143`` — the marker closes a section **after** that bar |
| ``repeats`` markers | ``firstMeasure`` on ``mc 1``, ``lastMeasure`` on ``mc 203`` (no interior repeats) |

The two section closers cut the 203 measures into **exactly three** leaf sections:

| Section | Measure ids | Measure count | Quarter span | First onset (quarters) | Last onset (quarters) |
|---------|-------------|---------------|--------------|------------------------|-----------------------|
| 1 | 1..78    | 78 | 78 × 3 = **234** | 0   | 231 |
| 2 | 79..143  | 65 | 65 × 3 = **195** | 234 | 426 |
| 3 | 144..203 | 60 | 60 × 3 = **180** | 429 | 606 |

Cross-checks a correct implementation must reproduce: 78 + 65 + 60 = 203;
234 + 195 + 180 = 609; each section's first onset equals the previous section's
last onset + 3 (231→234, 426→429); the last measure onset 606 + 3 = 609.

Each leaf ``MeasureMap`` segment is immutable and standard-conformant: measure ids
are the Measure Map row IDs ``"1"``..``"203"`` (``str(count)``, since the table
states no other); ``qstamp`` values are the exact ``Fraction`` prefix sums
(``Fraction(0)``, ``Fraction(3)``, …, ``Fraction(606)``); ``number`` == ``count``
here (1..203); ``time_signature`` ``3/4``; ``nominal_length`` == ``actual_length``
== 3 quarters; no ``start_repeat``/``end_repeat``/``volta``.

### Skeleton-from-load

**Ground truth.** Loading `trois-gymnopedies.measures.tsv`.

**Expected (default load).**

- Section count == **3**.
- Measure count == **203**.
- A flow named ``"source"`` is **always present** (count of flows named
  ``"source"`` == 1).
- Participant count == **1** (the score attaches to ``"source"`` by default).

**Expected (flatten opt-out).** Loading with flattening disabled creates **no
skeleton at all**: nothing is attached, and the timeline's ``skeleton`` accessor
raises (zero attachments). The opt-out suppresses structure creation entirely —
it does not produce a participant-less skeleton.

### (d) Attach semantics

**Ground truth.** Starting from the default-loaded skeleton (1 participant on
``"source"``).

**Derivation / expected.**

- Attaching a participant over a measure-id range **mints a one-step flow**: a flow
  whose traversal is exactly **1** atomic-section step (step count == 1) covering
  that range.
- Participant count increases by exactly 1: 1 → **2**.
- Flow count increases by exactly 1 (the minted flow joins ``"source"``): 1 → **2**.
- **Detach must be the exact inverse.** After detaching that participant:
  participant count returns to **1**; the minted flow — now with zero attachments —
  is removed, so flow count returns to **1** and only ``"source"`` remains; the
  ``SectionHierarchy`` (3 sections), measure count (203), and all quarter spans
  (234/195/180, total 609) are **unchanged**. Detach restores counts and the flow
  set; it never mutates the skeleton's temporal facts.

### (e) Authored flows

**Ground truth.** An authored **elision** flow that traverses ``1..41`` then
``44..78``, skipping ``42`` and ``43``.

**Derivation.**

- ``1..41`` = 41 measures; ``44..78`` = 35 measures; 41 + 35 = **76 measures**.
- Elided ``42, 43`` = 2 measures.
- Cross-check: the full ``1..78`` section is 78 measures; 78 − 2 = 76.
- Quarters: 76 × 3 = **228 quarters** (equivalently 123 + 105, or the section's 234
  minus the 6 quarters of the two elided bars).

**Expected.** The elision flow traverses exactly **76 measures = 228 quarters**.

A ``Gap`` step contributes **inserted material with no skeleton counterpart**: it
carries a quarter length on the flow's reference timeline but maps to **no**
measure id and **no** atomic section (its measure-address span is ``None``). A
correct flow store is distinguished by these facts, all of which a test must pin:

- the **ordered** list of steps (atomic-section references and ``Gap`` markers) in
  authored order;
- each step's measure-id span, or ``None`` for a ``Gap``;
- each step's quarter length;
- the flow's measure count (**76**, ``Gap`` excluded) and its reference-quarter
  span (228 plus any ``Gap`` quarters).

A store that merged the two non-adjacent ranges into a single ``1..78`` step, or
reordered steps, or counted a ``Gap`` toward the measure count, must fail.

### (f) Claim recording

**Ground truth.** A claim anchoring measure ``m1`` to **0.35 seconds** on a
recording participant.

**Expected — exactly two coordinate facts, on two axes.**

1. A **measure-address / reference-quarter** coordinate on the skeleton side:
   measure id ``m1``, whose onset qstamp is the exact ``Fraction(0)`` quarters.
   This is a **position** (a measure address), never an event.
2. A **physical-time** coordinate on the recording participant's seconds axis:
   ``0.35`` seconds.

A correct claim records both and only both; it must store ``m1`` on the
measure-address axis (equivalently ``Fraction(0)`` on the reference-quarter axis)
and ``0.35`` on the recording's physical (seconds) axis — not a floating-measure
value, and not an approximate quarter.

### (g) Reference-timeline materialization

**Ground truth.** A reordered "suite" flow that plays the sections in the order
**third, first, second** (``[s3, s1, s2]``).

**Derivation.** Section quarter spans in traversal order: 180, 234, 195. Cumulative
onsets on the unfolded reference axis:

| Position | Section | Onset (quarters) |
|----------|---------|------------------|
| start of s3 | 3 | 0 |
| start of s1 | 1 | 180 |
| start of s2 | 2 | 180 + 234 = 414 |
| flow end    | — | 414 + 195 = **609** |

**Expected.**

- Total span == **609 quarters**, with section spans **180 + 234 + 195** in that
  order (section boundaries at exact quarters 0, 180, 414, 609).
- The unfolded reference axis is **strictly monotonically increasing**: every
  measure onset is strictly greater than the previous. Concretely the 203 measure
  onsets run 0, 3, 6, … up to 606 (last measure), end 609 — no repeats, no
  decrease, even though the sections are reordered relative to the score.

### (h) Attach refusal and the single-attachment accessor

**Ground truth.** Participants attach to a **flow**, not to a timeline.

**Expected.**

- **Timeline-side attach refusal.** Attempting to attach a participant on the
  timeline side must be **refused (raise)**; only the flow side accepts attachments.
- **Single-attachment accessor.** The accessor that returns a flow's one attachment
  must **raise on zero** attachments and **raise on many** (≥ 2); it returns the
  attachment only when the flow has **exactly one**. For the default-loaded
  ``"source"`` flow with its single participant the accessor returns that
  participant; after a second attach it raises; after full detach it raises.

### (i) Structural skeleton equality

**Ground truth.** A ``TimeSkeleton`` *is* its authored temporal facts: the
section hierarchy, the metric hierarchy, and the authored flows (each flow's id
and its ordered step content, ``Gap`` markers included). Everything else —
generated uid, participants and their claims, materialized reference timelines,
display names — is identity or usage, not structure.

**Validation logic.** Equality must be decided by exactly the three structural
facts and nothing else. Each fact needs one positive and one negative proof:

1. Two skeletons built from structurally equal hierarchies compare **equal**
   even though their generated uids differ — so identity is excluded.
2. Hierarchies whose leaf measure counts differ (``(3, 2, 2)`` vs
   ``(4, 2, 2)``) break equality — so the section hierarchy is included, and
   deleting its clause cannot go unnoticed while the other facts agree.
3. Giving the two skeletons **different metric hierarchies** (policies differing
   in a metric fact such as bpm) breaks equality — so the metric hierarchy is
   included.
4. Authoring a custom flow on **one** side breaks equality — so flows are
   included; authoring the **same** flow (same id, same steps) on the other
   side restores it — so flow comparison is content-based, not
   object-identity-based.
5. Participants are excluded, proven on **both** attach lanes: attaching a
   participant to the always-present ``"source"`` flow leaves the two
   skeletons **equal**; and a measure-range attach mints its flow under a
   **content-derived** id — the range step itself (e.g. ``"m1-m5"``), never
   the participant's id — so two skeletons whose *different* participants
   attach over the *identical* range also stay **equal**. The same range
   attached twice on one skeleton **reuses** the flow (participant count
   rises, the flow list is unchanged); a range attach whose id collides with
   an authored flow holding different steps must **raise** rather than
   silently reuse it.

A non-skeleton operand must not compare equal: the equality hook returns
``NotImplemented`` and ``==`` answers ``False``. Instances are **unhashable by
contract** (``hash(skeleton)`` raises ``TypeError``): structural equality
without identity hashing would corrupt hash containers, so none may exist.
Section and metric hierarchies already define name-excluding structural
equality ((a) and (b) above); skeleton equality must **delegate** to them,
never re-derive their facts.

**Consequence that must be pinned separately: attachment bookkeeping is by
identity.** A timeline's skeleton-attachment store distinguishes attachments by
*object identity*, because two structurally equal skeletons are still two
distinct shared structures. The proof: attach two equal-but-distinct skeletons
to one timeline — the timeline must hold **both** (count 2, each retrievable as
the very object attached); detaching the **first** must remove exactly the
first, leaving the second in place. A store using equality-based membership
would alias the two and fail both halves.

### (j) Materialization round trip

**Ground truth.** A reference timeline materialized from a skeleton *carries*
that skeleton's structure, so harvesting structure back from the reference must
return the originating skeleton itself.

**Validation logic.** For a materialized reference ``ref`` of skeleton ``ts``
(both the abstract no-flow lane and the per-flow lane):

- ``ref.create_skeleton() is ts`` — the round trip returns the **same object**,
  not a reconstruction;
- ``ref.create_skeleton() == ts`` — and therefore also compares equal under (i);
- ``ts.materialize()`` (respectively ``ts.materialize(flow=...)``) called again
  returns ``ref`` **by identity** — materialization is cached, so repeated calls
  cannot mint drifting reference timelines with fresh generated ids.

Reference timelines are **not participants**: after materializing, the
participant count and the participant tuple are unchanged (no reference object
among them). Participation is attach()-enrollment only.
