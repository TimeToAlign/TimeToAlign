# Conversion Map Tests

This directory contains tests for the `timetoalign.maps` module, which implements coordinate conversion maps.

## Test Coverage Summary

| Module | Coverage | Status |
|--------|----------|--------|
| `maps/base.py` | 74% | Good |
| `maps/linear.py` | 64% | Good |
| `maps/table.py` | 66% | Good |
| `maps/composite.py` | 79% | Good |
| `maps/periodic.py` | 79% | Good |
| `maps/combination.py` | 81% | Good |

## Test Files

### `test_linear.py` - Linear Transformation Maps

**Purpose:** Validates affine transformations (`LinearMap`, `ScalarMap`, `ShiftMap`).

**Test Categories:**
- **Initialization**: Validates attributes and constraints (non-zero scalar).
- **Conversion**: Scalar and array conversions match mathematical expectations.
- **Inverse**: Inverse maps behave correctly ($f^{-1}(f(x)) = x$).
- **Composition**: Composing two linear maps results in a single linear map.
- **Fraction Support**: Validates high-precision arithmetic.

### `test_table.py` - Table-Based Maps

**Purpose:** Validates lookup and interpolation logic (`TableMap`).

**Test Categories:**
- **Interpolation**: Linear, Nearest, Previous (Step-Left), Next (Step-Right).
- **Extrapolation**: Extrapolate (linear), Constant (clamp), Error, NaN.
- **Inverse**: Inversion by swapping axes (requires monotonicity).
- **Factory Methods**: `from_tempo_changes` correctly builds maps from MIDI tempo data.

### `test_interpolation.py` - InterpolationMap

**Purpose:** Validates `InterpolationMap`, the anchor-pair engine used by
`TimelineGroup` and `WarpMap`. `InterpolationMap` is a `ConversionMap`
subclass like every other map in this package, so its tests exercise the
shared interface rather than a bespoke value-passing API.

**Test Categories:**
- **Conversion via `__call__`/`convert_array`**: scalar and array forward
  conversion, replacing the old `forward(values)` method form.
- **Inverse via `inverse()`**: the map-returning form (`inverse()(x)`),
  replacing the old `inverse(values)` method form. Covers increasing and
  decreasing target arrays, extrapolation, and the non-invertible error.
- **Inverse caching**: `inverse()` returns the same instance on repeated
  calls, and the returned map's own `inverse()` yields back the original
  instance (symmetric cache) — this matters because inverting is not free
  (it validates monotonicity and may reverse arrays), so callers that
  round-trip through `.inverse().inverse()` should not pay for it twice.
- **ConversionMap family membership**: `issubclass(InterpolationMap,
  ConversionMap)`; a `Coordinate` input is checked against the map's
  `source_unit` by the inherited `__call__`, so a mismatched unit raises
  `ValueError` without any InterpolationMap-specific code.
- **Serialization**: `to_dict()`/`from_dict()` round-trips arrays, ids, and
  units exactly; `ConversionMap.from_dict()` dispatches to
  `InterpolationMap` via the self-registering type registry.
- **Read-only coordinate arrays**: `source_coords`/`target_coords` are
  copied and frozen (`setflags(write=False)`) on construction, so mutating
  either raises `ValueError`; this also protects the cached inverse from
  desynchronizing, since the caller can no longer mutate one map's arrays
  in place and have it silently affect the other's converted values.
- **Selector matching**: `matches_selector()` (inherited from `ConversionMap`,
  overridden here) matches an `InterpolationMap`'s `source_id` in addition to
  its `id`/`name`, since group converter maps are addressed by the source
  timeline id in conversion-map specifications.
- **Exact assertions**: all numeric expectations use `==`/`np.array_equal`
  rather than `pytest.approx`/`assert_array_almost_equal` — the anchor
  values chosen throughout are exactly representable in binary floating
  point, so the computed conversions are exact.
- **Extraction factory removed**: `InterpolationMap` no longer offers a
  TableMap-extraction constructor that degrades a `TableMap`'s
  interpolation kind and extrapolation policy to plain linear
  interpolation/extrapolation — `Timeline.add_conversion_map` now stores
  `TableMap` instances directly (see `test_maps_integration.py`), so the
  factory and its tests are gone.

### `test_composite.py` - Composite Maps

**Purpose:** Validates maps composed of other maps (`ChainMap`, `PiecewiseMap`).

**Test Categories:**
- **ChainMap**:
    - Validates sequence execution ($f(g(x))$).
    - Unit compatibility checks between steps.
    - Inverse chain construction.
- **PiecewiseMap**:
    - Region selection based on break points.
    - Delegation to sub-maps.
    - Array conversion optimization.
    - Naive inversion support.

### `test_periodic.py` - Periodic Maps (RotationMap, FloorMap)

**Purpose:** Validates cyclic and floor-division transformations for metrical calculations.

**Test Categories:**

#### RotationMap
- **Initialization**: Validates period, scale, base, offset attributes.
- **Beat Rotation**: 4/4 time cycles through beats 1,2,3,4,1,2,3,4...
- **Beat Rotation 3/4**: 3/4 time cycles through beats 1,2,3,1,2,3...
- **Beat Rotation 6/8**: Compound meter with eighth-note beats.
- **Offset**: Handles non-zero starting positions (e.g., anacrusis).
- **Not Invertible**: Correctly reports `is_invertible = False` (many-to-one).
- **Serialization**: `to_dict()` / `from_dict()` roundtrip preserves all parameters.
- **Angle Normalization**: Handles negative inputs via modulo.

#### FloorMap
- **Initialization**: Validates divisor and base parameters.
- **Measure Numbers 4/4**: Quarters 0-3.99 -> Measure 1, 4-7.99 -> Measure 2, etc.
- **Measure Numbers 3/4**: 3 quarters per measure.
- **Zero-Indexed**: Supports base=0 for 0-indexed results.
- **Offset**: Handles non-zero divisor offsets.
- **Not Invertible**: Correctly reports `is_invertible = False` (many-to-one).
- **Page Numbers**: General use case for pagination.

#### Integration Tests
- **Measure/Beat Consistency**: FloorMap + RotationMap jointly cover all quarters.
- **SUPRA Compatibility**: Validates against SUPRA reference data (222 measures, 888 quarters).

**Validity Rationale:**

These maps implement the metrical coordinate transformations required by BeatGrid:
- `FloorMap`: quarters -> measure_number (integer division by quarters_per_measure)
- `RotationMap`: quarters -> beat_in_measure (cyclic modulo)

Both are explicitly NOT invertible because they are many-to-one mappings.

---

### `test_combination.py` - Multi-Output Maps (CombinationMap)

**Purpose:** Validates maps that produce multiple outputs from a single input.

**Test Categories:**
- **Initialization from Dict**: Maps specified as `{"name": map}` dictionary.
- **Initialization from Sequence**: Maps specified as `[("name", map), ...]` for ordering.
- **Empty Maps Error**: Raises ValueError for empty map collection.
- **Scalar Conversion**: Returns `{"name": value}` dict for scalar input.
- **Array Conversion**: Returns `{"name": array}` dict for array input.
- **Metrical Combination**: Combines FloorMap + RotationMap for (measure, beat) tuples.
- **XY Coordinates**: Combines two linear maps for (x, y) output.
- **Not Invertible**: Correctly reports `is_invertible = False`.
- **Get Map**: Can retrieve individual sub-maps by name.
- **Source Unit Validation**: All sub-maps must have compatible source units.
- **Source Unit Inheritance**: CombinationMap inherits source_unit from sub-maps.
- **Serialization**: `to_dict()` / `from_dict()` roundtrip.
- **SUPRA Metrical Positions**: Validates against SUPRA reference (measure 222, beat 4 at quarter 887).

**Validity Rationale:**

CombinationMap enables multi-valued lookups required by the TTA model:
- A single coordinate can map to (measure, beat, beat_type) tuples
- This is essential for the TimeStamp cross-section feature
- The map is NOT invertible because multiple outputs cannot uniquely determine input

---

### `test_meter.py` - Meter-Aware Maps (MetricMap, MetricalPositionMap)

**Purpose:** Validates `MetricMap.from_verovio_timemap` and the
`MetricalPositionMap` reverse lookup it feeds.

**Test Categories:**

#### MetricMap.from_verovio_timemap

A Verovio timemap is a JSON list whose measure-start entries carry a
`measureOn` xml:id and a `meterSig` of the form `"<measure_number>
<num>/<den>"`.  The factory collects each `measureOn` entry's `qstamp`
(absolute quarters) as a measure boundary, parses the measure-number token
as the label, and bounds the last measure by the timemap's final `qstamp`.

Validated against the Chopin Nocturne specimen (`performance_precision`
corpus) — zero-tolerance, exact values only:

- `n_measures == 38`
- `total_length == Fraction(425, 2)` (212.5 quarters)
- `starts[0] == Fraction(0)`, `starts[1] == Fraction(1, 2)` (0.5),
  `starts[2] == Fraction(13, 2)` (6.5), `starts[37] == Fraction(413, 2)`
  (206.5)
- `mns[0] == "1"`, `mns[-1] == "38"`; `mcs == list(range(1, 39))`
- last measure length == `Fraction(6)` (212.5 − 206.5)
- a timemap with no `measureOn` entries raises `ValueError`

#### MetricMap.quarters_at

`MetricMap.quarters_at(mc)` returns a measure's downbeat quarter position
directly — the reverse of `MetricMap.__call__` (quarters -> MC), and the
same value `get_measure_info(mc)["start"]` already exposed, now reachable
without building the `dict`. Validated against the same specimen boundaries
(`quarters_at(1)`, `quarters_at(2)`, `quarters_at(3)`); an unknown MC raises
`ValueError`.

#### MetricalPositionMap reverse lookup

`MetricalPositionMap(meter_map).quarters_at(mc, beat)` returns the
quarter position of a measure's beat, computed as
`meter_map.quarters_at(mc) + (beat - 1)`. Validated against the same specimen
boundaries (e.g. `quarters_at(2)` == `starts[1]` == the downbeat of MC 2),
and `mn_at(quarters)` returns the measure-number label at a position.

**Validity Rationale:**

The Verovio timemap supplies what a bare `.meter` file cannot: an upper
bound for the final measure.  A `.meter` file encodes only meter *changes*
(this specimen lists 4 rows across 38 measures), so the timemap's terminal
`qstamp` is required to close the last bar.  Exact-value assertions pin the
boundary arithmetic.

#### `ConversionMap.from_dict` registry dispatch

`MetricMap`, `BeatInMeasureMap`, and `MetricalPositionMap` round-trip
through `ConversionMap.from_dict()` using the self-registering class
registry (`__init_subclass__`), the same mechanism every other map type in
this package uses — there is no hand-written type-name table to keep in
sync as new map classes are added. `MetricalPositionMap.to_dict()` no
longer emits a redundant `map_type` key: the base `ConversionMap.to_dict()`
already writes `type` as the class name, which is exactly what the
registry dispatches on.

---

### `test_floating_measures.py` - The floating-measure conversion

**Purpose:** Validates `QuartersToFloatingMeasures.from_measure_map` — the one
map that answers the floating-measure (fm) convention — against the corpora
whose published annotations define it.

**What fm is, and why the two rules below are not cosmetic.** An fm value
reads a position as `<measure ordinal>.<how far into that bar>`, at three
decimals. Two properties decide every gold value here:

1. **The fractional part is anchored on the NOMINAL bar.** An incomplete bar's
   content sits where it is *notated*, not flush against its sounding start.
   The lattice therefore puts one knot per measure record at that bar's
   **virtual nominal downbeat** — sounding start minus the offset at which its
   content sits — and interpolates linearly between knots, so the slope inside
   a bar is exactly `1 / nominal_length`.
2. **Emission truncates, never rounds.**

**The specimen that separates truncation from rounding.** The Wagner Ring
Dataset's *Walküre* act III scene 3 opens with a 1/8 pickup in 9/8
(`Wagner_WWV086B-3.measures.tsv`, mc 1: `timesig 9/8`, `act_dur 1/8`,
`mc_offset 1`). In quarters: `nominal_length = 9/8 x 4 = 9/2`,
`actual_length = 1/8 x 4 = 1/2`, `nominal_offset = 1 x 4 = 4`. The pickup's
notated downbeat is therefore a virtual `0 - 4 = -4` quarters, and the onset at
quarter 0 sits

    fm(0) = 0 + (0 - (-4)) / (1/2 - (-4)) = 4 / (9/2) = 8/9 = 0.8888...

The published WRD tables read **`0.888`**. Truncation gives that;
`round(8/9, 3)` gives `0.889`, which is not the number the dataset states — so
the rule is `floor(value x 1000) / 1000`, and the same rule seen from the right
is what produces the `.999` an interval end shows. A rounding implementation
fails this one assertion and nothing else, which is exactly why it is pinned.

**The ordinal rule.** Ordinals count measure *records*, never printed labels.
The first record is `0` when it is an anacrusis (its content is offset and it is
shorter than its nominal bar) and `1` otherwise; every following record adds
one. Counting runs monotonically through voltas — Beethoven's WoO 71 has three
consecutive bars all printed `237` (mc 260, 261, 262: first ending, second
ending, and the bar after) and they get three consecutive ordinals — and never
resets. Deriving the ordinal from the label instead would give all three the
same fm, collapsing four quarters of music onto one point.

**The eroica pickup** is the second anacrusis shape: a 1/8 pickup in 2/4
(`nominal_length = 2`, `actual_length = 1/2`, `nominal_offset = 3/2`). Its
notated downbeat is `-3/2` and its onset reads
`(0 + 3/2) / (1/2 + 3/2) = 3/4`, i.e. **`0.750`** — the same rule, a different
metre.

**One map, one answer, however you ask.** Reading a column has to give what
reading its entries one at a time gives. That is not automatic: the knots are
ratios, so a lattice walked in floating point lands a few units below an exact
thousandth boundary about as often as above it, and truncating afterwards then
drops a whole thousandth — in the wrong direction half the time, so it is not
even a consistent offset. A bulk export would then disagree with a single
lookup on the same map and the same input.

The specimen that shows it is a bar of **7/3 quarters** — a septuplet-length
bar, whose boundaries no binary fraction reaches. Its slope is `3/7` fm per
quarter, so an exact thousandth boundary `k` sits at quarter `7k/3000`:

| quarter | derivation | fm |
|---|---|---|
| `0.007` | `1 + (3/7)(7/1000)` = `1 + 3/1000` | `1.003` |
| `0.014` | `1 + 6/1000` | `1.006` |
| `0.287` | `1 + 123/1000` | `1.122` |
| `2.331` | `1 + 999/1000` | `1.998` |

Both call shapes are asserted element for element across a sweep of the whole
three-bar grid, plus these four boundary readings. A float-interpolating column
reads `1.002` at quarter `0.007` and fails here.

**What a loaded score gets.** A score's fm conversion is derived from the
structure its measure rows describe — the measure map those rows build — never
from the printed labels in them. The Wagner *Walküre* file loaded through
`create_timeline()` therefore reads `fm(0) == 0.888`, the published value
derived above, and not the `1.0` that reading the label `"1"` off the pickup row
would produce. Where a score's structure cannot be expressed as an fm lattice —
WoO 71's split bars, below — the timeline is loaded with **no** floating-measure
conversion at all. Its absence is honest; numbers derived from a rule the
source does not support are not, and a silent fallback to label-derived
ordinals is the worst of the three, because nothing downstream can tell the two
kinds of fm apart.

**Split bars are a documented known issue, not a runtime warning.** Two measure
records sharing one notated downbeat — a bar divided across a repeat sign —
cannot both anchor the same ordinal, and no reachable input makes them
expressible, so the limitation belongs in the loader's and store's docstrings
(`Ms3Loader.create_timeline`, `ScoreStore.get_cmaps`) and here, where a reader
meets it before loading rather than after. Emitting a `UserWarning` on every
load of such a score told a caller nothing they could act on and landed in the
output of every notebook that loads one. **The validation logic is therefore
that the load succeeds with no warning at all**: the score loads, its measure
structure and every other conversion are intact, `TimeUnit.floating_measures`
is simply not among its units, and a test asserts the silence explicitly rather
than leaving it unobserved.

**The inverse** interpolates linearly over the same knots and performs no
truncation-reconstruction: a value that was truncated on the way out carries at
most one thousandth of a bar of error, and that is documented rather than
"corrected". `limit_denominator` and every other ratio-guessing device stay
banned. Inverting exactly-representable values is asserted instead: on the
eroica grid `0.750` inverts to `Coordinate(0)` quarters (the pickup's sounding
onset) and `2.0` to the start of measure 2.

**Zero tolerance:** every fm value is asserted as the exact float the
convention states; every quarters value as an exact `Fraction`.

---

### `test_properties.py` - Property-Based Invariants (Hypothesis)

**Purpose:** Verifies mathematical invariants (invertibility, composition
associativity, scaling, shifting, identity) that must hold for *all* valid
inputs, using Hypothesis to generate randomized examples across `LinearMap`,
`ScalarMap`, `ShiftMap`, the convenience maps (`TicksToQuarters` /
`QuartersToTicks`, `SamplesToSeconds` / `SecondsToSamples`), `TableMap`, and
`ChainMap`.

**Exact vs. retained `pytest.approx`.** The property tests split cleanly into
assertions whose two sides execute the *identical* sequence of float
operations (bit-exact — asserted with `==`) and assertions that round-trip an
arbitrary float through a lossy composition or compare a float map's output to
an independently-computed gold value (genuinely rounding — kept with
`pytest.approx`). 13 `approx` sites remain, and they belong to these classes:

- **Forward-then-inverse round-trips** — `inverse(forward(x)) == approx(x)`
  and `forward(inverse(x)) == approx(x)` for `LinearMap`, `ScalarMap`,
  `ShiftMap`, and `ChainMap` (`test_chain_inverse_roundtrip`). The inputs,
  scalars, and offsets are arbitrary Hypothesis-drawn floats, so IEEE-754
  rounding accumulates through the forward-then-inverse (and multi-step chain)
  composition. The tolerance absorbs that inherent loss, not any implementation
  defect (see the docstring on `test_chain_inverse_roundtrip`).
- **Interpolation / linearity at non-anchor points** — the `TableMap`
  mid-point round-trip (`test_inverse_roundtrip_monotonic`, the second loop)
  and the tempo-map inverse round-trip and linearity checks
  (`from_tempo_changes`). A mid-point is not a stored ordinate, so `np.interp`
  actually interpolates and the inverse re-interpolates; the result is exact
  only up to tolerance.
- **Reciprocal-scaled convenience maps** — `TicksToQuarters` /
  `QuartersToTicks` and `SamplesToSeconds` / `SecondsToSamples` are `ScalarMap`s
  whose scalar is a *rounded reciprocal* (`1/ppq`, `1/sr`). Both the int and
  float round-trips (`test_inverse_roundtrip_int` / `_float`) therefore pass
  through two rounding steps and are not bit-exact even for divisible integer
  inputs.
- **Fraction gold vs. float output** — `test_ticks_to_quarters_fraction_exact`
  compares the map's float output against `float(Fraction(ticks, ppq))`
  (`rel=1e-12`). The `Fraction` gold is exact but the map's float output is
  not, so equality only holds to tolerance.

**Sites converted to exact `==`.** Five assertions whose two sides run the
identical float arithmetic were hardened from `approx` to `==`:

- **Composition vs. sequential** (`test_composition_associativity`) —
  `ChainMap` calls its member maps in the same order as `m2(m1(x))`
  (`base.py:303-319`, `composite.py:101-122`), so both paths execute the same
  operations bit-for-bit.
- **Direct scaling** (`test_scaling_property`) — the test recomputes the exact
  product the implementation uses (`linear.py:237-239`).
- **Direct shifting** (`test_shift_property`) — the test recomputes the exact
  sum the implementation uses (`linear.py:338-340`).
- **Table anchor round-trips** (`test_inverse_roundtrip_monotonic`, the
  anchor loop) — `np.interp` returns the stored ordinate exactly at each anchor
  (`table.py:177-187`), so the inverse recovers the anchor abscissa exactly.
  (The mid-point loop in the same test keeps `approx` — see above.)
- **Nested chain associativity** (`test_chain_associativity`) — both groupings
  flatten to the same ordered `ChainMap` and run the identical operation
  sequence (`composite.py:101-122`).

The by-definition anchors that *are* bit-exact were already written with `==`
in this file (`t2q(ppq) == 1.0`, `q2t(1.0) == ppq`, `s2s(sr) == 1.0`,
`s2samp(1.0) == sr`) and the pure-`Fraction` `LinearMap` invariant asserts
exact `Fraction` equality (`result == expected`).

---

## Serialization

Every map's `to_dict()` is JSON-serializable: rational parameters (scalars,
offsets, boundaries, measure starts and lengths, table anchors) are emitted as
the **rational dictionary** `{value, numerator, denominator}` rather than as
`Fraction` objects or `"n/d"` strings, and `from_dict()` reads that shape and
nothing else. `to_dict()` also always emits `name`, and every subclass
`from_dict()` passes it to the constructor, so a custom map name round-trips.

The format, the fixpoint guarantee, and the cross-package JSON-safety sweep
are specified in `tests/core/README.md` (`test_wire_format.py`), which holds
the map name and `ConstantMap` `Fraction` round-trip tests.

## Integration Tests

Integration with the `Timeline` class is tested in `tests/timelines/test_maps_integration.py`.

## Validity Rationale

These tests verify the core mapping capabilities required by the TTA model:
- **Accuracy**: Mathematical transformations must be precise (using Fractions where possible).
- **Invertibility**: All maps must support inversion for bi-directional alignment.
- **Non-Invertibility**: Some maps (RotationMap, FloorMap, CombinationMap) are explicitly many-to-one.
- **Composition**: Maps must be composable to form complex chains.
- **Performance**: Array operations should use vectorized numpy implementations.
