# Conversion Map Tests

This directory contains tests for the `timetoalign.maps` module, which implements coordinate conversion maps.

## Manifest forms (`test_manifest_forms.py`)

A conversion map is written in a TimeToShare manifest as one of ten
abstract types — `LinearMap`, `TableMap`, `ConstantMap`, `PiecewiseMap`,
`ChainMap`, `CombinationMap` and the four measure views `MeasureRecordMap`,
`MeasureNumberMap`, `FloatingMeasureMap`, `MeasureBeatMap` — and every
library class lowers to one of them in `to_dict()`. A reader that only reads
what its own writer produced proves nothing about the format, so each
abstract type's form is pinned as a literal dictionary, and the proof that a
lowering is faithful is functional: the map read back from the written form
must convert every probe coordinate to exactly the result the written map
gives, value and Python type alike (`Fraction(5, 2)` is not `2.5`). Probes
lie inside every interval, on every anchor or boundary, and outside both ends
wherever the written map is defined there; a lowering that only agreed
inside its anchors, or only in value, fails.

- **Each abstract type is itself.** For the six general types and the four
  views, `ConversionMap.from_dict(m.to_dict())` (views given the measure maps
  they name) is of the same class, writes the same dictionary in both forms,
  and converts every probe identically.
- **Every library class lowers to one type, with exactly that type's
  fields.** `ScalarMap`, `ShiftMap`, `TicksToQuarters`, `QuartersToTicks`,
  `SamplesToSeconds` and `SecondsToSamples` write a `LinearMap` (the
  convenience keys `ppq` and `sample_rate` are gone); the minimal form drops
  `scalar` when it equals 1 and `offset` when it equals 0. TimeToShare's
  minimal form governs: it recognises a default by its exact value, and a
  map in the format carries no number type, so a `LinearMap` with the float
  scalar `1.0` writes no `scalar` key and is read back with the default
  scalar, the integer 1. **A float scalar equal to the default does not
  survive a minimal round trip as a float**: the written map multiplies the
  exact input `Fraction(1, 3)` into the float `0.3333333333333333`, the map
  read back answers `Fraction(1, 3)`. The complete form writes `1.0`.
  `InterpolationMap` writes a `TableMap`
  (`kind` linear, `extrapolate` linear), its `source_id` and `target_id`
  gone. A `TableMap` writes `extrapolate` `none` for the library's `error`
  and `nan`, `linear` for `extrapolate`, `constant` for `constant`; reading
  `none` gives `error`, so a probe outside the anchors raises after the
  round trip where `nan` answered NaN before it — the one difference the
  format rules, pinned as such. A `TableMap` of kind `previous` answers the
  stored anchor value itself (an exact `Fraction(1, 3)` stays exact), since
  no arithmetic is involved. An `IntervalToConstantMap` of numbers writes a
  `TableMap` of kind `previous` with `extrapolate` `constant` — exactly its
  clamping at both ends — or a `ConstantMap` when it has one boundary; one
  given an `end` writes a `PiecewiseMap` of `ConstantMap`s over its
  intervals, labels or numbers, and is then undefined outside
  `[first boundary, end)` before and after writing alike. A `CombinationMap`
  writes `maps` as an object keyed by output name and no `names`.
- **Measure-derived maps write a view over a measure map.**
  `QuartersToFloatingMeasures` writes a `FloatingMeasureMap`;
  `QuartersToMeasureNumber` and `MetricMap` write a `MeasureNumberMap`;
  `BeatInMeasureMap` and `MetricalPositionMap` write a `MeasureBeatMap`.
  Built outside a timeline, the view names the measure map built from the
  map's own data by the id the map minted for it when it was built (with
  `seed=7` and a given map id, `IdMinter(7).mint("measure_map")`; a measure
  map's own minted id has the prefix `measure_map`); within a timeline it names
  the timeline's measure map when the map derives from it, and otherwise a
  measure map of its own (the timeline tests pin which). Where the view's
  answer has another shape than the legacy map's, the test states the
  projection and pins it exactly: a `MetricMap` answers the measure count,
  which is the `number` of the rows it writes (`count` positional, `number`
  the count, `name` the label); a `QuartersToMeasureNumber` label `"3"`
  reads back as the number `3`; a `BeatInMeasureMap` answer is the `beat` of
  the read map's pair, and a `MetricalPositionMap`'s `{"mc": n, "beat": b}`
  is `{"number": n, "beat": b}`.
- **Measure starts read from a table keep its number type.**
  `QuartersToMeasureNumber.from_measure_data` reads each measure's `start`
  in the `MeasureData`'s declared number type, so the measure map its view
  writes carries the starts as the table holds them. Written and read back
  through a timeline, a fraction table with starts `0`, `1/3`, `4/3` gives
  the qstamps `Fraction(0)`, `Fraction(1, 3)`, `Fraction(4, 3)` (reading the
  float side gave the dyadic `6004799503160661/18014398509481984` for `1/3`,
  a ratio the table never stated); a float table with starts `0.0`, `0.5`,
  `1.25` gives `Fraction(0)`, `Fraction(1, 2)`, `Fraction(5, 4)`, the exact
  values of those doubles, as before. Numbers `1`, `2`, `3` in both. `1/3`
  is chosen because no double equals it, so only an exact read passes.
- **No lifting.** Reading builds the abstract type's class: a written
  `TicksToQuarters` reads back as a `LinearMap`, a `MetricMap` as a
  `MeasureNumberMap`. A manifest never says "ticks to quarters at 480 ppq",
  so restoring the specific class would infer what the file does not state.
  Equality is therefore that of class and state: a `ScalarMap` and the
  `LinearMap` it writes are not equal, though they write one dictionary and
  convert alike, and the round-trip tests compare dictionaries and
  conversions, never the two objects.
- **What has no form is refused, naming the map and the reason.** A
  `TableMap` of kind `nearest` or `next`, a `FloorMap`, a `RotationMap`, an
  `IntervalToConstantMap` of labels with more than one boundary and no
  `end`, a `QuartersToMeasureNumber` label that is no decimal integer
  (`"19a"`), a `MetricMap` whose measures do not start at 0 or leave a gap,
  and a `QuartersToFloatingMeasures` whose knots are no measure lattice
  (ordinals not consecutive from 0 or 1, a first knot that neither an
  anacrusis nor a downbeat at 0 explains) each raise `ValueError` with the
  exact message. Never an approximation: a lowering that is not lossless
  does not happen.
- **Reading refuses what the format does not define.** A former type name
  (`ScalarMap`, `InterpolationMap`, `MetricMap`, ...) is an unknown type,
  the message listing the ten; `kind` `nearest` and `extrapolate` `nan` are
  refused; a view naming a measure map the manifest does not hold is
  refused naming the id; a key the type writes is required unless it has a
  default (`name`, `scalar`, `offset`, `kind`, `extrapolate`).

### The measure views

The four views read a measure map's rows; their semantics are those of the
library's former measure maps, pinned with exact values on a measure map with
an anacrusis (offset 3 in a 4/4 bar), a regular bar, a 3/4 bar and an
irregular bar: the record lookup takes the last measure whose `qstamp` lies
at or before the position, clamped to the first and the last measure as
`MetricMap` clamps; `MeasureNumberMap` answers that record's `number` (the
count when a measure states none); `FloatingMeasureMap` answers what
`QuartersToFloatingMeasures.from_measure_map` answers, anchored on the
nominal bar and truncated to three decimals; `MeasureBeatMap` answers
`{"number": <number>, "beat": <position - qstamp + 1>}`, the beat of
`MetricMap.beat_in_measure`. `QuartersToFloatingMeasures.from_measure_map`
returns a `FloatingMeasureMap`, which remembers the measure map it reads, so
the score loaders' floating-measure map writes as a view without being
matched against anything.

## Content seeds of minted ids (`test_map_content_seeds.py`)

A map built without `uid` and without `seed=` mints its id from the content
minter of its class and every value that defines it, read once the
outermost constructor has run (a subclass sets its parameters after the
base initializer). What proves the seed holds a defining parameter is the
id's independence from what else was built: for every concrete
`ConversionMap` class, two maps differing in one defining parameter (`ppq`
480 against 960, one boundary moved, another interpolation kind, another
nested map, ...) are built in a fresh run in either order; each map's id is
the same whether it was built first or after the other, and the two ids
differ. A seed missing that parameter fails both: the second map built
takes the next id of a shared minter, so its id changes with the order.
Nested maps are given ids, so the pair differs only where the test says.
The parameter list covers every concrete class: a guard collects the
subclasses of `ConversionMap` the library defines and fails on one without
a case, so a new class cannot escape the check.

Two further properties, exact values throughout:

- **A NumPy scalar or array seeds alike in every environment.** A
  `TableMap` from `np.float64` lists gets the id the same map from Python
  floats gets (a NumPy scalar is read as its Python number, an array by
  its bytes), and from an array the id a list of the same floats gets.
- **`seed=` reaches the maps a constructor builds.** `MetricalPositionMap`
  and `IntervalToConstantMap` (given an `end`) mint their own id, then each
  nested map's, from one `IdMinter(seed)`: the ids are those `IdMinter(seed)`
  mints in that order, whatever ran before in the process.

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
- **Built from pairs** (`LinearMap.from_pairs`): the map is the line through
  the first and last pair by position, so the expected scalar and offset are
  computed by hand from those two pairs and asserted exactly, never fitted.
  Exact pairs (`Fraction`, `int`) give an exact `Fraction` scalar; float
  pairs give the float the two-point formula gives. The order of the pairs is
  irrelevant: a shuffled list builds a map equal to the sorted one.
- **The check of the inner pairs is a derived bound, measured on the
  positions.** In a drawn axis the labels are exact (0, 10, 20 … s) and the
  positions are numbers a program computed in doubles and printed, so the
  rounding lives in the positions. For each inner pair the test oracle is
  the position at which the line through the end pairs reaches the pair's
  value, computed exactly on the `Fraction` of every number as given, before
  any `origin` is subtracted. When every position and value is exact, the
  given position must be that position. Otherwise it may miss it by the
  bound a forward error analysis gives for a position computed as
  `x0 + w·(v − v_first)/(v_last − v_first)`, with the multiplication and
  the division in any order:
  - Write `t = (v − v_first)/(v_last − v_first)`, exact on the values as
    given, and `u = 2**-53`. Every operation rounds to the nearest double,
    so its result errs by at most half an ulp of itself, at most `u` of its
    magnitude; an operation whose exact result is a double does not round.
  - The product `w·t` passes through the multiplication, the division and
    each label subtraction whose exact difference is no double: `k` roundings
    (2 for integer labels), each a factor `(1 + δ)` with `|δ| ≤ u`, in
    whatever order, so it misses `w·t` by at most
    `γ_k·|t|·|w|`, `γ_k = k·u/(1 − k·u)`. The final addition of `x0` adds
    half an ulp of its result, the position itself. One position thus misses
    `x0 + w·t` by at most `E(p, t) = ½·ulp(p) + γ_k·|t|·|w|`.
  - The check draws its line through the end pairs as given, and they are
    computed positions too: the first at `t = 0`, where the product is
    exactly 0 (`E = ½·ulp(p_first)`), the last at `t = 1`. Their errors move
    the line at `t` by at most `|1 − t|·E(p_first, 0) + |t|·E(p_last, 1)`.
  - `w` is known only through the span as given, `S = |p_last − p_first|`,
    which misses `|w|` by the ends' errors:
    `|w| ≤ W = (S + ½·ulp(p_first) + ½·ulp(p_last)) / (1 − γ_k)`.
  - The bound is `E(p, t) + |1 − t|·E(p_first, 0) + |t|·E(p_last, 1)` with
    `W` for `|w|`, all in exact arithmetic. A position rounded once from its
    exact value misses it by `½·ulp(p)` alone, which the bound contains.

  The bound is probed at its edge on the line through (0.0, 0.0) and
  (2.0, 2.0), which reaches 1 at 1 with `t = ½` and `k = 2`. Just above 1 it
  is `2**-53 + γ_2·W` plus `½·(2**-52 + γ_2·W)`, with `W` a hair above 2:
  six `u`, three ulps of 1, plus terms of order `2**-105`. Just below 1, where
  the ulp halves, it is five and a half `u`. So 1 + 3·2**-52
  (1.0000000000000007) and 1 − 5·2**-53 (0.9999999999999994) pass, and
  1 + 4·2**-52 (1.0000000000000009) and 1 − 6·2**-53 (0.9999999999999993)
  are refused. Those values follow from the float grid and the derivation,
  not from the library. An inner value of 0.0 does not force an exact hit:
  on the line through (0, -1) and (2, 1), the value 0.0 at the position one
  ulp above 1 passes, where a bound in ulps of the value (5e-324) would
  refuse every rounded position.
- **Seeded random axes, rounded once and computed.** With the seed 20261118,
  2,000 axes are drawn: a start `x0` and a width `w` each a uniform number
  in [1, 10) times a decade from 0.1 to 1000 (`x0` of either sign), 3 to 13
  ticks, integer labels from a random first label by a random step. With
  `d = v − v_first` and `D = v_last − v_first`, each axis is placed four
  ways: every exact position `x0 + w·d/D` rounded once to a double, and
  computed in doubles as `x0 + (w·d)/D`, `x0 + w·(d/D)` and
  `x0 + (w/D)·d`. All 8,000 calibrations (12,193 inner pairs per
  placement) are accepted. The simulation is not vacuous: measured against the rounding alone,
  `½·ulp(p) + |1 − t|·½·ulp(p_first) + |t|·½·ulp(p_last)`, in exact
  arithmetic in the test, the axes rounded once all fit, while 339, 281 and
  227 of the computed axes (in the order above) have an inner position
  beyond it. And the bound still discriminates: each axis with one inner
  position moved by 1e-9 of its span is refused.
- **A drawn seconds axis with five labelled ticks.** Positions
  76.70371459980326, 290.5859487023929, 504.4681828049824,
  718.350416907572 and 932.2326510101616 pt against 0, 10, 20, 30 and 40 s,
  read from the first tick (`origin`), computed by a program in doubles. The
  inner ticks lie 0.9375, 0.375 and 0.15625 of their ulps off the exact
  positions (as the nearest floats: 290.58594870239284, 504.4681828049824
  and 718.350416907572), which is 0.373, 0.085 and 0.046 of the bound, so
  the calibration passes with the scalar 40 / 855.5289364103584 =
  0.04675470144567245 and offset 0.0. The 10 s tick's miss is beyond the
  rounding alone (1.11 of it), which is why the bound covers computed
  positions. The second tick moved up by two floats (290.585948702393) lies
  2.94 of its ulps off, 1.17 of the bound (2.51 of its ulps), and is
  refused, the message naming the exact position. The third tick moved by
  1e-9 of the span (8.555289364103584e-07 pt) is refused, and so is the last
  tick labelled 41 instead of 40, which moves the line's 10 s to
  285.3693088462321.
- **`origin` changes what the map reads, not what is checked.** The map
  reads `position - origin` in the numbers' own arithmetic: pairs at
  10, 50, 90 with `origin` 10 give the same scalar and offset (0.05, 0.0) as
  the pairs 0, 40, 80 without one. An `origin` given as a coordinate in
  another unit than the positions is refused.
- **Mixing exact and float numbers follows a stated rule, never a refusal.**
  Exactness is a property of the whole calibration: one float among the
  numbers gives every inner pair the position bound. The same inner value
  given as the float 1.0000000000000002 passes at the exact position 1
  (it lies within the bound), while its exact `Fraction` (every number
  then exact) is refused. Exact positions with float values ((0, 0.1),
  (1, 0.2), (2, 0.3)) pass: the line through the doubles 0.1 and 0.3 reaches
  the double 0.2 at 0.625 ulp of 1 from 1, and the map is the float line
  of scalar 0.09999999999999999 and offset 0.1. Exact labels with float
  positions are the drawn axis above.
- **The map declares the representation of its results.** Pairs holding a
  float give a map declaring `float`, which answers floats even into
  `quarters`, and its inverse declares `float` too. Exact pairs declare
  `fraction` where the value unit admits it (an exact line into `seconds`
  answers `Fraction(1, 2)` at 1); into a discrete unit, which admits
  integers only, the map declares nothing and writes whole ticks (480 at 1).
  The exact map refuses a float input with an `OutputNumberTypeError`
  naming the declaration (the points map through (0, 0) and (6, 1) answers
  `Fraction(1, 6)` at 1 and `Fraction(1, 12)` at `Fraction(1, 2)`, and
  refuses 1.0), as every map declared `fraction` does (below).
- **Values carry a unit.** Values are coordinates sharing one unit, which
  becomes the map's target unit, or numbers read in `value_unit`. A number
  without `value_unit` raises `TypeError` naming the keyword, and a
  `Coordinate` value in another unit than `value_unit` (or than the first
  coordinate value) is refused. Each refusal (fewer than two pairs, a
  repeated position, a value in another unit, an inner pair off the line)
  names the offending pair by its index in the given list; end pairs
  reading one value give a zero scalar, which `LinearMap` refuses. The base
  `ConversionMap.from_pairs` raises `NotImplementedError`, since a
  law that cannot be built from pairs must say so rather than guess.

### `test_table.py` - Table-Based Maps

**Purpose:** Validates lookup and interpolation logic (`TableMap`).

**Test Categories:**
- **Interpolation**: Linear, Nearest, Previous (Step-Left), Next (Step-Right).
- **Extrapolation**: Extrapolate (linear), Constant (clamp), Error, NaN.
- **Inverse**: Inversion by swapping axes (requires monotonicity).
- **Factory Methods**: `from_tempo_changes` correctly builds maps from MIDI tempo data.
- **Exact interpolation** (`TestTableMapExact`): when the input and every
  anchor are exact (`int` or `Fraction`), `linear` interpolation and linear
  extrapolation compute `y0 + (x - x0)·(y1 - y0)/(x1 - x0)` in `Fraction`
  arithmetic; any float among them keeps the float path.

**`is_invertible` is a Python `bool`** (`TestIsInvertibleIsABool`). Every map
declares the property `bool`; `TableMap` computed it from NumPy comparisons and
answered a `numpy.bool`, which equals `True` but is not `True` and which
`json.dumps` refuses. The test asserts identity (`is True` / `is False`), which
only a Python `bool` passes, on all three monotonicity cases: increasing
`[0, 1, 3]` and decreasing `[3, 1, 0]` (both `True`), and the non-monotonic
`[0, 2, 1]` (`False`); and `json.dumps` of the three answers gives
`[true, true, false]`.

**Why these values prove exactness.** A float path can only be told apart
from an exact one by a result that no binary float represents, so every
exact probe lands on a non-dyadic ratio, and the assertion checks the type
as well as the value (`Fraction(5, 4) == 1.25` holds, so value equality
alone would pass a float path wherever the result happens to be dyadic).
On anchors `[0, 4] → [0, 2]` given as `Fraction`s, `1/3` must give exactly
`Fraction(1, 6)` (a float path gives `0.16666666666666666`, which is not
equal to it), and `5` with `extrapolate="linear"` exactly `Fraction(5, 2)`.
On three anchors `[0, 3, 7] → [0, 1, 2]` (plain `int`s) the segment chosen
is checked as well: `1` gives `Fraction(1, 3)` from the first segment, `6`
gives `Fraction(7, 4)` from the second, `-3` extrapolates the first segment
to `Fraction(-1)` and `10` the last to `Fraction(11, 4)`; an anchor itself
answers its ordinate, `3` giving `Fraction(1)`. The map is built without
units, so the call returns the map's own result; on a `quarters` target the
public call keeps `Fraction(1, 6)`, since that axis writes fractions.

**Why the float path is pinned as well.** Exactness is a property of the
operands, never imposed on them: float anchors `[0.0, 4.0] → [0.0, 2.0]` at
`Fraction(1, 3)` give the float `0.16666666666666666`, exact anchors at the
float input `0.5` give the float `0.25`, and one float anchor among exact
ones (`[0, 4] → [0, 2.0]`) keeps the whole map on the float path. The array
lane is float64 by contract whatever the anchors: `convert_array` on the
`Fraction` anchors over `[0, 1, 2, 4]` gives a float64 array
`[0.0, 0.5, 1.0, 2.0]`. The step kinds (`previous`, `next`, `nearest`)
involve no arithmetic and keep their behaviour, which the tests above pin.

**A declared output number type** (`TestDeclaredOutputNumberType`). A map
given `output_number_type` at construction writes every public result in
it; without one it writes the target unit's default, as before. The probes
are exact anchors from quarters to seconds (`[0, 2, 4] → [0, 1/3, 1]`),
whose results 1/6, 2/3 and 4/3 no float represents, so type and value are
asserted together. Undeclared, the map answers the float
`0.16666666666666666` at 1, since seconds default to floats. Declared
`fraction`, it answers `Fraction(1, 3)` at the anchor 2, `Fraction(1, 6)` and
`Fraction(2, 3)` between anchors, and `Fraction(4, 3)` extrapolating past
the last. **A map declared `fraction` refuses a float input** with an
`OutputNumberTypeError` naming the declaration, a `ValueError` and never a
`TypeError`: the input's type is supported, its value is what the
declaration refuses. The map computes exactly, and a float input
would leave the float result rebuilt as a ratio (at 1.0 the dyadic
`Fraction(6004799503160661, 36028797018963968)`, a ratio nothing stated). The
refusal sits in the base's evaluation, so it holds for every map class and
for a `Coordinate` input declared `float`, and it is checked on `TableMap`,
on the exact `LinearMap` built from pairs, and on a `ChainMap`. An `int` and
a `Fraction` are computed exactly, and the float array lane is untouched (it
is float64 by contract). Declared `float` on a `quarters` target, whose
default is fractions, it answers the float. The inverse carries the
declaration where its target, the map's source unit, admits it: the
inverse of the fraction map answers `Fraction(3)` at 2/3, while the inverse
of a fraction map from `ticks` writes whole ticks (480 at 1/6) and declares
nothing. **The round trip drops the declaration, on purpose.** A TimeToShare
map is independent of any number type and the format has no key for one, so
the declaration is a setting of the library: the declared map writes the
same complete form as the undeclared one, and `ConversionMap.from_dict` of
its JSON text gives a map equal to the undeclared map (not to the declared
one) that answers the float `0.16666666666666666` at 1. A declaration the target unit does not
admit (`float` into `ticks`) is refused with the unit's own message.

**Every map with a target unit takes the declaration.** The keyword lives in
`ConversionMap.__init__`; `ChainMap`, `PiecewiseMap` and `InterpolationMap`
pass it through, as `LinearMap`, `ScalarMap`, `ShiftMap` and `TableMap` do.
Each is checked on its own (`test_composite.py`, `test_interpolation.py`), on
exact members from quarters to seconds whose result at 1 is 1/6, a value no
float represents. Undeclared, a `ChainMap` and a `PiecewiseMap` answer the
float `0.16666666666666666`, since seconds default to floats; declared
`fraction` they answer `Fraction(1, 6)`, refuse the float input 1.0, and
their inverses, into quarters, carry the declaration and answer
`Fraction(1)` at `Fraction(1, 6)`. `InterpolationMap` computes in doubles
over float64 anchors, so a `fraction` declaration would rebuild every result
from a float: it is refused at construction with a `ValueError` naming the
class. The declaration is a setting of the library for every
class: `to_dict` writes the same form with it and without it.

**A map that computes in doubles answers doubles.** Its output number type
is `float` wherever the target unit admits one, whatever the unit's default;
the rule lives in `ConversionMap.output_number_type`, once for every class.
Undeclared from seconds into quarters, whose default is fractions, an
`InterpolationMap` therefore declares `float` and answers the float `0.25`
at 0.5 (the value it computed, never `Fraction(1, 4)` rebuilt from it), as
it does declared `float`; its inverse, into seconds, declares `float` too.

**A map whose own parameters are floats computes in doubles**
(`TestFloatParameters` in `test_linear.py`, `test_table.py`,
`test_constant.py` and `test_interval.py`). Whether a
map computes in doubles is read from the instance: a float scalar or offset
of a `LinearMap`, `ScalarMap` or `ShiftMap`, or a float anchor of a
`TableMap`, makes it so, exactly as an `InterpolationMap` is. What would
expose the old behaviour is a float that is not dyadic-looking once
rebuilt, so the probes use 0.1: `LinearMap(scalar=0.1)` from quarters into
quarters answered `Fraction(3602879701896397, 36028797018963968)` at 1, the
double 0.1 rebuilt as a ratio nothing stated. Now:

- Declared `fraction`, `LinearMap(scalar=0.1)`, `LinearMap(offset=0.1)`,
  `ScalarMap(0.1)`, `ShiftMap(offset=0.1)` and a `TableMap` over
  `[0, 4] → [0, 0.1]` (one float anchor among integers) are refused at
  construction with the `ValueError` naming their class ("… computes in
  doubles and cannot be declared to write fractions.").
- Undeclared from quarters into quarters, whose default is fractions, each
  declares `float` and answers the float it computed: `0.1` at 1 for the
  scalars, `1.1` at 1 for the offsets, `0.025` at 1 for the table — type and
  value asserted together.
- Exact parameters stay exact: `LinearMap(scalar=Fraction(1, 3),
  offset=1)`, `ScalarMap()` (its default scalar is the integer 1),
  `ShiftMap(offset=Fraction(1, 3))` and the table `[0, 4] → [0, 2]` declare
  `fraction` and answer `Fraction(4, 3)`, `Fraction(1, 3)`,
  `Fraction(4, 3)` and `Fraction(1, 2)` at 1 (`Fraction(1, 3)` for the
  default scalar map, at `Fraction(1, 3)`). A rule that counted every
  `LinearMap` or `TableMap` as float arithmetic would answer floats here.
- A step function's parameters are its values: the one value of a
  `ConstantMap`, the value of each interval of an `IntervalToConstantMap`.
  Its boundaries only choose the interval and never enter the result, so
  they are no parameters, and float boundaries leave exact values exact.
  The probe uses float boundaries `[0.0, 1.5]` throughout. Undeclared from
  quarters into quarters, `ConstantMap(value=0.1)` answers the float `0.1`
  at 0, and the step function of values `[0.1, 0.5]` answers the float
  `0.1` at 1, with and without an `end` (3), and so does its
  `to_table_map()`: the two forms of one step function agree. Both
  answered `Fraction(3602879701896397, 36028797018963968)` before, the
  rebuilt double. With the exact values `Fraction(1, 10)` and
  `Fraction(1, 2)` (and `ConstantMap(value=Fraction(1, 10))`) they declare
  `fraction` and answer `Fraction(1, 10)` — which a rule counting the
  float boundaries would turn into a float. A label is no number: a
  `ConstantMap` or step function of strings answers the string as before.

**A constant on a target unit is a number or a string**
(`TestConstantOnATargetUnit` in `test_constant.py`). A target unit writes
the constant as that unit writes numbers; a string is a label the format
admits beside it (a manifest's `ConstantMap` holds a number or a string,
with any `target_unit`). Anything else can never be written: `None` raised
`TypeError` (`cannot read NoneType as a number`) and `True` (`Boolean
values are not valid number cells`) from every stamp display reading the
map. So `ConstantMap(value=…, target_unit="milliseconds")` is refused at
construction for `None`, `True`, `numpy.bool_(True)` and `{"a": 1}`, with
the exact `TypeError` naming the unit, the type and the value (`A
ConstantMap on the target unit 'milliseconds' holds a number or a string,
not NoneType: None`); `bool` is checked before the number test because it
is an `int`. Accepted on the same unit, each with the value it answers at 0:
`3` (`3.0`, milliseconds being float-canonical), `Fraction(1, 3)`
(`0.3333333333333333`) and the label `"x"` (`"x"`), each of exactly that
type. Without a target unit nothing is refused: `ConstantMap(value=None)`
answers `None`, as before (`test_scalar_conversion_none`). The manifest
reader still reads every `ConstantMap` the format admits: the string `"x"`
with `target_unit: milliseconds` round-trips through `to_dict` /
`ConversionMap.from_dict` to an equal map.

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
- **Serialization**: written as the `TableMap` it is (see *Manifest forms*).
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

**A composite computes in doubles when a member answers floats**
(`TestFloatArithmeticMembers`). Whether a map computes in doubles is a
property of the instance: a `ChainMap` or `PiecewiseMap` does when any
member does, or when a member states `float` as its own output number type.
What would expose a composite that ignores its members is a result no float
represents, so the probes are chosen around one:

- *A float-arithmetic member.* An `InterpolationMap` from quarters into
  quarters over `[0, 3] → [0, 1]`, followed in a chain by the exact scalar
  1/2, or alone in a piecewise map over `[0, 3)`. Declared `fraction`, both
  composites are refused at construction with the `ValueError` naming their
  class; before, the chain answered the float rebuilt as
  `Fraction(6004799503160661, 36028797018963968)` at `Fraction(1)`.
  Undeclared, both declare `float` (quarters default to fractions, so only
  the rule can give `float`) and answer `0.16666666666666666` (chain) and
  `0.3333333333333333` (piecewise) at `Fraction(1)`, type and value
  asserted together.
- *Exact members stay exact.* Two exact scalars 1/2 and 1/3 from quarters
  into quarters (chain), and the scalar 1/6 over `[0, 6)` (piecewise),
  undeclared: both declare `fraction`, the quarters default, and answer
  `Fraction(1, 6)` at 1. A composite that counted itself float-arithmetic
  regardless would answer a float.
- *A member's own declaration applies.* A member is called through its own
  evaluation, so its guard runs and its result is written in the
  representation it states itself (never in its target unit's default, an
  intermediate axis). A chain whose first member is the exact scalar 1/2
  declared `float` hands `0.5` to the scalar 1/3 and answers the float
  `0.16666666666666666`, declaring `float`; a piecewise map whose member is
  the scalar 1/3 declared `float` answers `0.3333333333333333`. Composites
  that bypassed the member would answer `Fraction(1, 6)` and
  `Fraction(1, 3)`. Declaring either composite `fraction` is refused.
- *A member's own guard applies.* A chain whose first member, the float
  scalar 0.5, hands `0.5` to a member declared `fraction` raises that
  member's `OutputNumberTypeError` ("… not the float 0.5."); a piecewise map whose
  member is declared `fraction` raises it for the float input 1.0. Before,
  both computed in floats.

**What a map states about its arithmetic is read once, at construction**
(`TestStatedOnceAtConstruction`). A map is immutable once built, so whether
it computes in doubles and the representation it states itself are fixed
when it is built; reading them again for every value walks every member,
recursively. What proves the reading happens once is a value converted after
the readers have been made to fail: both readers of `ConversionMap` are
replaced, after construction, by functions that raise, and a chain of a
float member (the scalar 0.5) and an exact one (1/3), and a piecewise map
over a chain of the exact scalars 1/2 and 1/3, still answer the float
`0.16666666666666666` and `Fraction(1, 6)` at 1, declaring `float` and
`fraction`. Building a map after the replacement raises, which shows the
readers are the ones construction calls.

**PiecewiseMap picks the interval in the input's own arithmetic.** An int or
a `Fraction` is compared exactly with the breaks as given; a float with the
breaks as floats. Breaks `[0, 3/10, 1]` with the exact scalars 1 and 2,
declared `fraction`: `3/10 − 10⁻²⁰` lies in the first interval and answers
`Fraction(29999999999999999999, 10**20)`, while `3/10` itself answers
`Fraction(3, 5)`. As a float `3/10 − 10⁻²⁰` is `0.3`, the break, so a
float comparison picks the second interval and answers
`Fraction(59999999999999999998, 10**20)`.

### `test_interval.py` - Step functions (`IntervalToConstantMap`)

**`to_table_map()` keeps the step function's numbers as given.** A map with
a float parameter computes in doubles, so a table map built from an exact
step function with float anchors would answer floats where the step function
answers exact values. The probe is the step function from quarters into
quarters with boundaries `0` and `3/2` and values `1/3` and `2/3`, none of
which (`1/3`, `2/3`) a float represents. Its table map has the anchors
`[0, 3/2, 5/2]` and `[1/3, 2/3, 2/3]` (the last boundary extended by one
unit, holding the last value), each of the type given, as its manifest form
shows (`[0, [3, 2], [5, 2]]`, an int and two Fraction pairs), declares `fraction`,
the quarters default, and answers `Fraction(2, 3)` at `Fraction(3, 2)`;
read linearly, it answers `Fraction(1, 2)` at `Fraction(3, 4)`. The float
anchors it was built with before answered `0.6666666666666666` and `0.5`. A
step function given float numbers keeps them: boundaries `[0.0, 1.5]` give
the anchors `[0.0, 1.5, 2.5]`, floats, and the table map declares `float`.

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
- **No manifest form**: writing a periodic map is refused (see *Manifest forms*).
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
- **Serialization**: see *Manifest forms*.
- **SUPRA Metrical Positions**: Validates against SUPRA reference (measure 222, beat 4 at quarter 887).

**Each output is its member's own evaluation** (`TestMemberEvaluation`).
A combination calls every member as a composite calls its members, so the
member's guard runs and its result is written in the representation the
member states itself. Only a result that differs between the two paths can
tell them apart, so the probes are built around one. A member declared
`fraction` (the exact scalar 1/3 from quarters into quarters) refuses the
float input 1.0 with its own `OutputNumberTypeError` ("… not the float 1.0."); a
combination bypassing it answered the float `0.3333333333333333` from a
map declared to write fractions. A member declared `float` (the same exact
scalar) answers `0.3333333333333333` at 1, where the bypass answered
`Fraction(1, 3)`. A float-arithmetic member (the float scalar 0.5 into
quarters) beside an exact one (the scalar 1/3) answers the float `0.5`
while its neighbour keeps `Fraction(1, 3)`: each output follows its own
member, type and value asserted together.

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
   The lattice therefore puts one knot per floating measure at its first
   record's **virtual nominal downbeat** — sounding start minus the offset at
   which its content sits — and interpolates linearly between knots, so the
   slope inside a bar that fills its nominal length is exactly
   `1 / nominal_length`.
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

**One notated bar, one floating measure.** A floating measure is a
**compound measure**: a maximal run of adjacent measure records that make
one notated bar. A record on its own is a compound of one. This is what
floating measures are for: a source that represents one notated bar in
several records (a bar divided by a repeat sign, by a metre change or by a
cadenza inside it) still reads as that one bar. The compound places ONE
knot, at its first record's nominal downbeat; its continuation records place
none, so inside a compound the reading spreads linearly over the compound's
whole extent, from its knot to the next compound's.

Write `end` for how far into its notated bar a compound reaches: its first
record's `offset_within_measure` plus the `actual_length` of all its
records. A record B joins the compound whose last record is A when A and B
do not both state a `volta` with different passes, and **either**

1. **by number**: A and B carry the same `number`, and it is not null; or
2. **by fit**: A and B both carry a number, different ones, and B
   completes the bar the compound leaves incomplete: `end` lies below A's
   `nominal_length`, B has that nominal length, B's
   `offset_within_measure` equals `end` exactly, and `end` plus B's
   `actual_length` equals the nominal length exactly.

The join by number reads nothing else: not A's `next`, not the two nominal
lengths, not B's `offset_within_measure`. Adjacent records with one number
are one notated bar by the numbering convention, whatever metre each part
is written in and wherever the first part jumps to. The join by fit is the
case where there is no doubt that two numbers name one bar: a source that
numbers its records by count (`mn` = `mc`) splits a bar at a repeat sign
into a first part ending the repeat and a second part, a pickup, that
states exactly the offset where the first part ends and fills exactly the
rest of the bar. Every one of the four facts is needed, and a split that
fails only the offset is warned about, not joined (below).

Why the conditions are needed, by the case each keeps apart: records
without a number say nothing about bars, so they join nothing; across
voltas, the endings carry one number by convention (`15a` and `15b`), yet
they are two bars, so the endings under different volta brackets count
through as separate floating measures. A record stating no volta joins a
record under a bracket: the rest of a second ending, written in a record of
its own, is still that ending's bar.

**The ordinal rule.** Ordinals count compound measures, never printed labels.
The first is `0` when it is an anacrusis (its first record's content is offset
and its records together are shorter than its nominal bar) and `1` otherwise;
every following compound adds one. Counting runs monotonically through voltas
and never resets. Deriving the ordinal from the label instead would give a
first and a second ending the same fm, collapsing their music onto one point.

**The refusal that remains.** Knots must increase. After joining, two knots can
still coincide or go backwards only where records that cannot join share a
nominal downbeat: records of different numbers that do not complete one bar
by fit, records without a number, or the endings under different voltas;
such a measure map has no lattice and the map raises, naming the two
floating measures. No approximate lattice is built in its place.

**Measures that do not add up: one warning.** Building a lattice emits ONE
`UserWarning` per measure map that lists, in score order and by measure ID
and number, every place where the measures do not add up to their bars, or
no warning at all. Each compound is measured against the nominal length of
its last record, the bar it ends in. Four kinds:

- **offset mismatch** — a record inside a compound that states an offset
  (`offset_within_measure` above 0) other than where the records before it
  in the compound end (the first record's offset plus their actual
  lengths). An offset of `0` states nothing: a continuation written from
  the start of its record is the numbering convention's business. A
  compound with a mismatch is reported for its mismatches only: its
  records say two different things about where it ends, so summing its
  lengths into an overlength or a short line would report a number the
  source does not state.
- **doubtful split** — a compound that ends short of its bar (`end` below
  A's nominal length), followed by a record B numbered apart (both
  numbers non-null, no different voltas) with the same nominal length
  whose `actual_length` completes the bar exactly from `end`, but whose
  offset is not `end` (no offset at all, i.e. `0`, or another one). They
  do not join: the numbers say two bars and the offset does not say
  otherwise. Both compounds of the pair are reported only under this kind,
  so the second is never also reported as overlength.
- **overlength** — a compound whose `end` exceeds its nominal length. It is
  kept as joined.
- **short** — a compound whose `end` lies below its nominal length, which is
  neither the first compound (an anacrusis) nor the last (whose rest the
  anacrusis supplies), and which is not completed by EVERY record its last
  record leads to. Its successors are its `next`, by default the following
  record; a successor completes it when it has the bar's nominal length,
  its offset is `end` and its actual length makes up the rest of the bar.
  This exemption is what a first ending looks like: a 3/4-of-a-bar first
  ending whose `next` names only the jump back to the start of the repeat,
  where a pickup at offset 3/4 carries the missing quarter, adds up across
  the jump, although the two records are not adjacent and never join. A
  bar that ends a repeat and is not a first ending names two successors,
  the jump back and the record after it; each pass plays the bar and then
  one successor, so each pass must add up, and a fresh bar after it does
  not complete the bar however well the jump target does. A `next` that
  names no record of the map leaves nothing to complete the bar.

The message reads
`Floating measures that do not add up to their bars:` followed by one line
per place, each `  <kind>: <detail>`, separated by `;` and a line break; a
group of records is written `measures 'a', 'b' (number 5)`, a single one
`measure 'a' (number 5)`, a record without a number `(no number)`. Lengths
are quarters as exact ratios (`35/4`). The three kinds read:

- `offset mismatch: measure 'B' (number n) states offset o, not e where
  'A' ends` (`A` the record before `B` in the compound, `e` where the
  compound ends up to `A`)
- `doubtful split, not joined: measure 'A' (number m) and measure 'B'
  (number n) together fill a bar of N quarters, but 'B' states offset o,
  not e where 'A' ends`
- `overlength: measures … reach E quarters into a bar of N` (`reaches`
  for a single record)
- `short: measures … end E quarters into a bar of N, and measure 'S'
  (number s), which they lead to, does not complete the bar` (`ends` /
  `it leads` for a single record; the successors that do not complete it
  are named in their `next` order, `measures 'S', 'T' (…)` and `do` for
  two; `…, and they lead to no measure of the map` where `next` names
  none of its records)

**Only a lattice that is built warns.** The knots are computed first. Where
they increase, the lattice is built and converts in all four cases, and the
warning is the only difference. Where they do not, the map raises the
refusal below and no warning is emitted; the refusal's message ends with
the same list (`. Floating measures that do not add up to their bars:` and
the lines), so the error says what in the measures led to it. Writing a
lattice to a manifest re-reads it without warning again.

**The warning names the user's line.** It is attributed to the first stack
frame outside the `timetoalign` package, however deep inside it was raised:
`warnings.catch_warnings(record=True)` records the calling test file as the
warning's `filename`, not a file of the package. Attributing a fixed number
of frames up would name a package file for every caller but one.

**A score without a lattice says so.** A score loader leaves the
floating-measure conversion out where the measures have no lattice (no
approximate lattice is built in its place). That absence is announced: the
score store emits one `UserWarning`, attributed like the above, reading
`The score '<source>' has no floating measures: <the refusal's message>`,
`<source>` being the file the loader read. A silent absence would leave a
loader's user with a timeline that lacks an axis and no word why.

**The specimens of the compound rule** are hand-derived below; every one of them
raised before records joined, because a continuation record's nominal downbeat
equals (or precedes) its first record's.

*A split bar after an anacrusis* — the opening of the second movement of
Beethoven's Op. 106 up to bar 47, written as the movement's own compressed
rows: count 1 is the anacrusis (`number 0`, 3/4, `qstamp 0`, `actual 1`,
`offset 2`); counts 2–46 are default successors, numbers 1–45, three quarters
each, so count 47 starts at `1 + 45 x 3 = 136`; counts 47 and 48 are both
`number 46` (`actual 2`, then `actual 1` at offset 2, `qstamp 138`); count 49
is bar 47 (`qstamp 139`, three quarters).
Ordinals, counted from the start: count 1 is an anacrusis (offset 2 > 0,
actual 1 < nominal 3), ordinal `0`; counts 2–46 are one compound each,
ordinals 1–45; counts 47 + 48 are one compound, ordinal `46`, knot at
`136 - 0 = 136` (count 48 would have put a second knot at `138 - 2 = 136`);
count 49 is ordinal `47`, knot `139`; the closing knot `142` carries `48`.
The slope inside bar 46 is `1/3`, as in any regular 3/4 bar:

| quarter | derivation | fm |
|---|---|---|
| `0` | `0 + (0 + 2)/3` | `0.666` |
| `136` | knot | `46.0` |
| `137` | `46 + 1/3` | `46.333` |
| `138` | `46 + 2/3` (count 48's sounding start) | `46.666` |
| `139` | knot | `47.0` |

*A three-part compound longer than its nominal bar* — the opening of the
fourth movement of Op. 106, rows 1–6, 4/4 throughout: bar 1 is counts 1
(`actual 23/4`) and 2 (`actual 3`, at `23/4`, offset 0), so it spans
`35/4` quarters; bar 2 is counts 3 (`actual 17/8`, at `35/4`), 4 (`actual 15/8`,
offset `17/8`, at `87/8`, nominal downbeat `87/8 - 17/8 = 35/4`) and 5
(`actual 6`, at `51/4`), ten quarters in all; count 6 is bar 3 at `75/4`, and
the closing knot lies at `91/4`. Count 1 starts at offset 0, so ordinal `1`;
knots `0 → 1`, `35/4 → 2`, `75/4 → 3`, `91/4 → 4`:

| quarter | derivation | fm |
|---|---|---|
| `0` | knot | `1.0` |
| `23/4` | `1 + (23/4)/(35/4) = 1 + 23/35 = 1.6571…` | `1.657` |
| `35/4` | knot | `2.0` |
| `87/8` | `2 + (17/8)/10 = 2 + 17/80 = 2.2125` | `2.212` |
| `51/4` | `2 + 4/10` | `2.4` |
| `75/4` | knot | `3.0` |

The inverse runs over the same knots: `2.0 → 35/4 = 8.75`,
`2.5 → 35/4 + 10/2 = 13.75`, `3.0 → 75/4 = 18.75`. Both call shapes read the
split bar of the second movement alike (`136, 137, 138, 139 → 46.0, 46.333,
46.666, 47.0`).

*A volta pair* — fourteen 4/4 bars, then `15a` (`number 15`, volta 1, end
repeat, `next` back to bar 1) and `15b` (`number 15`, volta 2). They state
volta 1 and volta 2, so they are two floating measures, `15.0` at quarter 56
and `16.0` at quarter 60, and the closing knot at 64 carries `17`. The same
two records with the default `next` (one played after the other) read the
same: the voltas alone keep them apart.

*Two adjacent records of one number* — `a` (`actual 1/2` of a 2/4 bar, at
`0`) and `b` (`actual 3/2`, at `1/2`), both numbered `5`, `next` the default:

- `b` a 2/4 bar at offset `1/2`: one bar, knots `0 → 1` and `2 → 2`, so
  `fm(0) = 1.0`, `fm(1) = 1.5`, `fm(2) = 2.0`;
- `b` a 3/4 bar (`nominal_length 3`) at offset `1/2`: the nominal lengths
  differ, the number joins them all the same, and the readings are those
  above;
- `b` a 2/4 bar at offset `0`: the offset does not continue `a`, the number
  joins them all the same. Knots `0 → 1` and `2 → 2` (no knot at `b`'s
  downbeat `1/2`), so `fm(1/2) = 1 + (1/2)/2 = 1.25`; apart, `b` would be a
  knot of its own and read `2.0`.

*Two adjacent records of two numbers that fit* — the same `a` and `b`,
numbered `5` and `6`, `b` a 2/4 bar at offset `1/2`: `a` ends `0 + 1/2 =
1/2` into its bar of `2`; `b` has the nominal length `2`, its offset is
`1/2`, and `1/2 + 3/2 = 2`. They join by fit: knots `0 → 1` and `2 → 2`,
`fm(0) = 1.0`, `fm(1) = 1.5`, `fm(2) = 2.0`, and no warning.

*Refusals* — `a` and `b`, `b` at offset `1/2`, both nominal downbeats `0`:

- numbered `5` and `6`, `b` sounding `1` quarter instead of `3/2`: `1/2 + 1
  = 3/2` does not complete the bar, so no fit; the numbers differ, the
  knots coincide, the map raises. Nothing in the measures fails to add up
  (`a` and `b` are the first and the last compound, and `b` does not
  complete the bar), so the message lists no places;
- both without a number: they never join, the map raises;
- numbered `5` and `5`, `a` under volta 1 and `b` under volta 2: the voltas
  differ, the map raises.

**A real score joined by fit: the `op18_no4_mov4_flow` skeleton**
(`op18_no4_mov4_flow.measures.tsv`, 226 records, 2/2 throughout, so
`nominal_length 4`). Its numbers run by count (`mn` = `mc`), so no two
records share a number, and five bars split at a repeat sign carry two
numbers each: mc 9/10, 18/19, 27/28 (3 + 1 quarters) and mc 77/78, 84/85
(`7/2 + 1/2`). Before the join by fit, mc 10's nominal downbeat `32 - 3 =
29` was mc 9's and the map raised.

- mc 1 is the anacrusis (`mn 1`, `act_dur 1/4` = 1 quarter, `mc_offset 3/4`
  = 3): offset above 0 and `1 < 4`, ordinal `0`, knot `-3`.
- mc 2–8 are full bars, ordinals 1–7, at `1, 5, …, 25`.
- mc 9 (`mn 9`, at `29`, `act_dur 3/4` = 3, end repeat, `next 1, 10`) and
  mc 10 (`mn 10`, at `32`, `act_dur 1/4` = 1, `mc_offset 3/4` = 3, start
  repeat): mc 9 ends `0 + 3 = 3 < 4` into its bar; mc 10 has the nominal
  length `4`, offset `3 = 3`, and `3 + 1 = 4`. One compound, ordinal `8`,
  knot `29`; mc 11 at `33` is ordinal `9`. Readings: `fm(29) = 8.0`,
  `fm(32) = 8 + 3/4 = 8.75`, `fm(33) = 9.0`.
- Likewise mc 18/19 (ordinal 16, knot 61) and mc 27/28 (ordinal 24, knot
  93), so mc 29 at `97` is ordinal `25`.
- mc 29–76 are 48 records, none joined (mc 44, `act_dur 3/4` under volta 1,
  and mc 45 under volta 2 stay apart), so mc 76 is ordinal `72` and mc 77
  starts at `97 + 47 x 4 + 3 = 288`. mc 77 (`mn 77`, `act_dur 7/8` = `7/2`)
  and mc 78 (`mn 78`, at `583/2`, `act_dur 1/8` = `1/2`, `mc_offset 7/8` =
  `7/2`): `7/2 < 4`, offset `7/2`, `7/2 + 1/2 = 4`. One compound, ordinal
  `73`, knot `288`; mc 79 at `292` is ordinal `74`. Readings: `fm(288) =
  73.0`, `fm(583/2) = 73 + (7/2)/4 = 73.875`, `fm(292) = 74.0`.
- mc 84/85 join alike (ordinal 79).

226 records, 5 joins: 221 floating measures, ordinals 0–220, and 222 knots
with the closing one carrying `221`. The skeleton warns about nothing: its
two first endings (mc 44, 3 quarters, `next 28`; mc 93, `7/2`, `next 85`)
are short, but each jumps back to a pickup (mc 28 at offset 3 with 1
quarter, mc 85 at offset `7/2` with `1/2`) that completes it.

**A real score: WoO 71.** Beethoven's WoO 71 (`WoO71.measures.tsv`) divides
bars at repeat signs throughout. Its rows, in DCML columns (`act_dur` and
`mc_offset` in whole notes, so times four in quarters):

- mc 1 is the anacrusis (`mn 0`, 2/4, `act_dur 1/4`, `mc_offset 1/4`):
  ordinal `0`. mc 2–10 are `mn 1`–`9`, two quarters each, ordinals 1–9;
  so mc 11 starts at `1 + 9 x 2 = 19`.
- mc 11 and 12 are both `mn 10` (`act_dur 1/4` each, mc 12 at `mc_offset 1/4`,
  i.e. at `20` with offset 1): one compound, ordinal `10`, knot `19`; bar 11
  (mc 13) at `21`. **`fm(20) = 10 + 1/2 = 10.5`.**
- mc 42 (`mn 38`, `act_dur 1/4`, at `75`, `next 33, 43`: the end of a
  repeat) and mc 43 (`mn 38`, `act_dur 1/8`, `mc_offset 3/8`, at `76`): mc 43's
  nominal downbeat `76 - 3/2 = 149/2` lies *before* mc 42's, which raised
  before; joined, the compound has ordinal `38` (mn 0–37 are one compound
  each) and its knot at `75`; mc 44 starts at `75 + 1 + 1/2 = 153/2`.
  **`fm(76) = 38 + 1/(3/2) = 38 + 2/3 → 38.666`.**
- mc 260 (`mn 237`, volta 1, `next 252`) and mc 261 (`mn 237`, volta 2) do not
  join (different voltas), and are ordinals `237` and `238` at the loaded
  map's qstamps `945/2` and `949/2`. mc 262 (`mn 237` again, the 6/8 rest of
  the second ending, at `957/2`, `act_dur 3/8`, `mc_offset 3/8`, no volta)
  carries 261's number and states no volta, so it joins 261 although 261 is
  a 2/4 bar (`nominal_length 2`, sounding `act_dur 1` = 4 quarters) and 262
  a 6/8 bar (`nominal_length 3`). The compound [261, 262] is ordinal `238`,
  its knot 261's downbeat `949/2 - 0 = 949/2`; its extent is `4 + 3/2 =
  11/2` quarters, and mc 263 (`mn 238`, offset 0) is the next knot, ordinal
  `239`, at `949/2 + 11/2 = 480`. Readings:

  | quarter | derivation | fm |
  |---|---|---|
  | `945/2` (mc 260) | knot | `237.0` |
  | `949/2` (mc 261) | knot | `238.0` |
  | `477` | `238 + (5/2)/(11/2) = 238 + 5/11 = 238.4545…` | `238.454` |
  | `957/2` (mc 262's qstamp) | `238 + 4/(11/2) = 238 + 8/11 = 238.7272…` | `238.727` |
  | `480` (mc 263) | knot | `239.0` |

The score's 397 measure records form 373 floating measures, ordinals 0–372;
with the closing knot, carrying `373`, the lattice has 374 knots.

**Bars split at a repeat sign: Couperin.** The DCML Couperin concerts split
bars at repeat signs whose first part jumps back.

- `c05n05_musete.measures.tsv` (6/8, `nominal_length 3`): mc 1 is the
  anacrusis (`mn 0`, `act_dur 3/8`, `mc_offset 3/8`: offset `3/2` above 0,
  actual `3/2` below 3, ordinal `0`). Every `mn` from 0 to 54 forms one
  compound: mn 4 (mc 5 + 6), mn 14 (mc 16 + 17) and mn 28 (mc 31 + 32) each
  join two adjacent records. So `mn` equals the ordinal throughout: 58
  records, 55 compounds, 56 knots, the closing one carrying `55`. mc 31 (`mn
  28`, at `165/2`, `act_dur 3/8` = `3/2` quarters, end repeat, `next 17, 6`)
  and mc 32 (`mn 28`, at `84`, `mc_offset 3/8` = `3/2` quarters) share a
  number and neither states a volta: one compound, ordinal `28`, knot
  `165/2`; mc 33 (`mn 29`) is the next knot at `171/2`. The slope is `1/3`:

  | quarter | derivation | fm |
  |---|---|---|
  | `165/2` | knot | `28.0` |
  | `84` (mc 32's qstamp) | `28 + (3/2)/3` | `28.5` |
  | `85` | `28 + (5/2)/3 = 28.8333…` | `28.833` |
  | `171/2` | knot | `29.0` |

  Apart, mc 32's nominal downbeat `84 - 3/2 = 165/2` would coincide with mc
  31's and the score would have no floating measures.
- `c11n08_Rondeau.measures.tsv` (3/8, `nominal_length 3/2`): mc 18 (`mn 16`,
  at `23`, `act_dur 1/4` = 1 quarter, `next 10, 1`) and mc 19 (`mn 16`, at
  `24`, `mc_offset 1/4` = 1 quarter, `act_dur 1/8`) join, ordinal `16`
  (mc 1 is the anacrusis `mn 0`, `act_dur 1/8`, `mc_offset 1/4`, ordinal `0`;
  mn 8, mc 9 + 10, joins as well; so `mn` is the ordinal), knot `23`;
  mc 20 (`mn 17`) at `49/2`. Readings: `fm(23) = 16.0`,
  `fm(24) = 16 + 1/(3/2) = 16.666`, `fm(49/2) = 17.0`. The movement warns
  once, about mc 27 (`mn 24`, `act_dur 1/4` = 1 quarter) and mc 28 (`mn
  24`, `act_dur 1/2` = 2 quarters, both at offset 0) in 3/8: `1 + 2 = 3`
  quarters in a bar of `3/2`, so `overlength: measures '27', '28' (number
  24) reach 3 quarters into a bar of 3/2`.

**Bars across a metre change: `sq8913219`.** OpenScore's string-quartet
movement (`OpenScoreSQ/sq8913219.measures.tsv`, qstamps from
`quarterbeats_all_endings`) shares a number across a metre change twice.

- mc 482 (`mn 480`, 3/8, `nominal_length 3/2`, `act_dur 3/8` = `3/2`
  quarters, at `1270`, `next 483`) and mc 483 (`mn 480`, 3/4, `nominal_length
  3`, `act_dur 1/4` = 1 quarter, `mc_offset 1/2` = 2 quarters, at `2543/2`):
  one number, so one compound although the nominal lengths differ. Its knot
  is 482's downbeat `1270`, its extent `3/2 + 1 = 5/2`, and mc 484 (`mn
  481`, offset 0) is the next knot at `1270 + 5/2 = 2545/2`. Ordinal: mc 1
  is the anacrusis (4/4, `act_dur 1/4`, `mc_offset 3/4`), ordinal `0`; mc
  2–481 are 480 records of which only mc 77 and 78 (`mn 76`) join, so 479
  compounds, ordinals 1–479; the compound [482, 483] is ordinal `480` —
  `mn` again.
- mc 583 (`mn 578`, 3/4, `act_dur 3/4`, at `3131/2`, `next 1`, the end
  before the *fine*) and mc 584 (`mn 578`, 2/2, `nominal_length 4`,
  `act_dur 1/4` = 1 quarter, `mc_offset 3/4` = 3 quarters, at `3137/2`):
  583's `next` jumps to the start and the nominal lengths are `3` and `4`,
  but they carry one number and no volta, so they are one compound. Up to mc
  582, the 582 records hold four joined pairs (mc 77/78, 482/483, 491/492
  and 533/534), so 578 compounds, ordinals 0–577, and [583, 584] is ordinal
  `578`, its knot 583's downbeat `3131/2`, its extent `3 + 1 = 4`; mc 585
  (`mn 579`, offset 0) is the next knot, ordinal `579`, at `3131/2 + 4 =
  3139/2`. Apart, the two would share the nominal downbeat `3137/2 - 3 =
  3131/2`, and the score had no lattice.

The whole score's 809 records hold 13 adjacent pairs of one number, none
under two different voltas (first records mc 77, 482, 491, 533, 583, 592,
601, 610, 626, 660, 667, 675, 684), so they form `809 - 13 = 796` compounds,
ordinals 0–795 (`mn` throughout: the last record, mc 809, is `mn 795`); with
the closing knot at mc 809's end `2444 + 3 = 2447`, carrying `796`, the
lattice has 797 knots. Readings:

| quarter | derivation | fm |
|---|---|---|
| `1270` | knot | `480.0` |
| `2543/2` (mc 483's qstamp) | `480 + (3/2)/(5/2) = 480 + 3/5` | `480.6` |
| `1272` | `480 + 2/(5/2) = 480 + 4/5` | `480.8` |
| `2545/2` (mc 484) | knot | `481.0` |
| `3131/2` (mc 583) | knot | `578.0` |
| `3137/2` (mc 584's qstamp) | `578 + 3/4` | `578.75` |
| `3139/2` (mc 585) | knot | `579.0` |
| `2446` | `795 + 2/3` (mc 809 is three quarters of a 2/2 bar at offset 0, so its compound runs from `2444` to the closing knot `2447`) | `795.666` |

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

**The warnings, by specimen.** Each kind is pinned with the whole
message (`pytest.warns(UserWarning, match=re.escape(...))` anchored on the
complete text), and with the readings the lattice still gives:

*Doubtful split* — `a` and `b` numbered `5` and `6`, `b` at offset `0`
(no offset stated): `a` ends at `1/2` in a bar of `2`, `b` has the nominal
length `2` and `1/2 + 3/2 = 2`, but its offset `0` is not `1/2`. Not
joined: `b` is a knot of its own at `1/2`, ordinal `2`; knots `0 → 1`,
`1/2 → 2`, `2 → 3`, so `fm(1/4) = 1.5`, `fm(1/2) = 2.0`, `fm(2) = 3.0`.
Message: `doubtful split, not joined: measure 'a' (number 5) and measure
'b' (number 6) together fill a bar of 2 quarters, but 'b' states offset 0,
not 1/2 where 'a' ends`. This is the fit case that must NOT join because
the offset is missing. The same pair with `b` at offset `1/4` reports
`states offset 1/4, not 1/2`; `b`'s knot `1/2 - 1/4 = 1/4` follows `a`'s,
knots `0 → 1`, `1/4 → 2`, `2 → 3`, so `fm(1/4) = 2.0`, `fm(1/2) = 2 +
(1/4)/(7/4) = 2 + 1/7 → 2.142`, `fm(2) = 3.0`.

*Overlength* — the Op. 106/IV opening above: bar 1 is records `1` and `2`
(number 1), `0 + 23/4 + 3 = 35/4 > 4`; bar 2 is records `3`, `4`, `5`
(number 2), `0 + 17/8 + 15/8 + 6 = 10 > 4`; bar 3 (record `6`) is the last
compound and fills its bar. Message lines: `overlength: measures '1', '2'
(number 1) reach 35/4 quarters into a bar of 4` and `overlength: measures
'3', '4', '5' (number 2) reach 10 quarters into a bar of 4`. The readings
are those of the table above.

*Short* — five 4/4 bars `m1`–`m5`, numbered 1–5, with `m3` sounding 3
quarters (offset 0) and the default `next`: `m3` is neither first nor last,
ends `3 < 4` into its bar, and its successor `m4` sits at offset `0`, not
`3`. Message: `short: measure 'm3' (number 3) ends 3 quarters into a bar of
4, and measure 'm4' (number 4), which it leads to, does not complete the
bar`. Knots `0, 4, 8, 11, 15, 19`
with ordinals 1–6: `fm(10) = 3 + 2/3 → 3.666`, `fm(11) = 4.0`.

*Not short: a first ending completed across the jump* — the anacrusis `m0`
(number 0, 4/4, offset 3, 1 quarter, at `0`), `m1` (number 1, a full bar
at `1`), `m2` (number 2, volta 1, 3 quarters at `5`, end repeat, `next`
naming `m0`), `m3` (number 2, volta 2, a full bar at `8`), `m4` (number 3,
a full bar at `12`). `m2` ends `3 < 4` into its bar, but its successor `m0`
has nominal length `4`, offset `3` and `3 + 1 = 4`: it completes the bar
across the jump, so no warning. `m2` and `m3` state different voltas and
stay apart. Knots `-3, 1, 5, 8, 12, 16` with ordinals 0–5: `fm(5) = 2.0`,
`fm(7) = 2 + 2/3 → 2.666`, `fm(8) = 3.0`. With `m2`'s `next` left to the
default, its successor is `m3` at offset `0`, and the same map warns
`short: measure 'm2' (number 2) ends 3 quarters into a bar of 4, and
measure 'm3' (number 2), which it leads to, does not complete the bar`.

*Short: a bar that ends a repeat leads both back and on* — the same
anacrusis `m0` and bar `m1`, then `m2` (number 2, no volta, 3 quarters at
`5`, end repeat, `next` naming `m0` and `m3`), `m3` (number 3, a full bar
at `8`), `m4` (number 4, a full bar at `12`). The first pass jumps back to
`m0`, which completes the bar (`3 + 1 = 4`); the second goes on to `m3`,
which starts a fresh bar at offset `0`. Not every pass adds up, so it is
short: `short: measure 'm2' (number 2) ends 3 quarters into a bar of 4, and
measure 'm3' (number 3), which it leads to, does not complete the bar`. An
exemption on ANY successor would let the jump target excuse the second
pass, and this map would be silent. Knots `-3, 1, 5, 8, 12, 16`, ordinals
0–5, `fm(7) = 2.666` as above. The real first endings stay silent under
the stricter rule, because a first ending's `next` names the jump target
alone: op18 flow mc 44 (`next 28`) and mc 93 (`next 85`), the same bars of
the ABC edition, and mc 126 of the first movement of Op. 106 (3.5 of 4
quarters, `next 1`, completed by the 1/2-quarter anacrusis mc 1).

*Offset mismatch* — `a` and `b` both numbered `5` (one compound by number),
`a` sounding `1/2` from offset 0, `b` stating offset `1` and sounding `1`
quarter at `1/2`. The compound ends at `1/2` up to `a`; `b` states `1`.
Message: `offset mismatch: measure 'b' (number 5) states offset 1, not 1/2
where 'a' ends`, and no other line (summed, `1/2 + 1 = 3/2 < 2` would be
short, but the compound is the first and the last anyway). The lattice is
built: one compound, knots `0 → 1`, `3/2 → 2`, so `fm(3/4) = 1 +
(3/4)/(3/2) = 1.5`.

*A doubtful split that has no lattice* — `a` and `b` numbered `5` and `6`,
`b` at offset `3/4`: `a` ends at `1/2`, `b` has the nominal length `2` and
`1/2 + 3/2 = 2`, so it is a doubtful split; `b`'s own reach `3/4 + 3/2 =
9/4 > 2` is not also reported as overlength. `b`'s nominal downbeat `1/2 -
3/4 = -1/4` precedes `a`'s `0`, so the map raises; no warning is emitted,
and the error reads, in full:
`The floating measures starting with measures 'a' and 'b' have nominal
downbeats 0 and -1/4, which do not increase; adjacent measures form one
floating measure only when they share a number or the second completes the
bar the first leaves incomplete, and endings under different volta brackets
count through as separate floating measures. Floating measures that do not
add up to their bars:` followed by the line `  doubtful split, not joined:
measure 'a' (number 5) and measure 'b' (number 6) together fill a bar of 2
quarters, but 'b' states offset 3/4, not 1/2 where 'a' ends`.

*The loader's word on a score without a lattice* — the same two bars as a
DCML measures table written to a temporary file: mc 1 (`mn 5`, 2/4,
`quarterbeats 0`, `act_dur 1/8` = `1/2` quarter, `mc_offset 0`) and mc 2
(`mn 6`, 2/4, `quarterbeats 1/2`, `act_dur 3/8` = `3/2`, `mc_offset 3/16`
= `3/4`). The loader names the records by their mc, `'1'` and `'2'`.
`create_timeline()` returns a timeline whose measure map holds both rows
and which has no floating-measure unit, and emits exactly one warning:
`The score '<path>' has no floating measures: ` followed by the refusal
above with `'1'` and `'2'` for `'a'` and `'b'`.

*Who the warning names* — `QuartersToFloatingMeasures.from_measure_map`
called from the test on the short-`m3` map, and `create_timeline()` of the
loader above, each record one warning whose `filename` is the test file.

**What a loaded score gets.** A score's fm conversion is derived from the
structure its measure rows describe — the measure map those rows build — never
from the printed labels in them. The Wagner *Walküre* file loaded through
`create_timeline()` therefore reads `fm(0) == 0.888`, the published value
derived above, and not the `1.0` that reading the label `"1"` off the pickup row
would produce. WoO 71, whose bars are split, loads with its floating-measure
conversion like any other score: `fm(20) == 10.5` on the loaded timeline, and
its measure map holds all `397` measure rows of the TSV. Its lattice warns
once, about three places that do not add up, all in the data:

- mc 42 + 43 (`mn 38`): mc 42 sounds 1 quarter from offset 0, so it ends at
  `0 + 1 = 1`; mc 43 (`act_dur 1/8`, `mc_offset 3/8`) states offset `3/2`.
  The two say different things about the bar: summed, the compound reaches
  `3/2` of a 2/4 bar (short); from mc 43's own offset, it reaches `3/2 +
  1/2 = 2` and fills it. The mismatch is the fact, so it is the line:
  `offset mismatch: measure '43' (number 38) states offset 3/2, not 1 where
  '42' ends`, and no short line;
- mc 261 + 262 (`mn 237`, the second ending): mc 261 sounds 4 quarters from
  offset 0 (`act_dur 1` in 2/4), ending at `4`; mc 262 (6/8, `act_dur
  3/8`, `mc_offset 3/8`) states offset `3/2`: `offset mismatch: measure
  '262' (number 237) states offset 3/2, not 4 where '261' ends`, and no
  overlength line (the sum `0 + 4 + 3/2 = 11/2` against the 6/8 bar's `3`
  is no fact the source states);
- mc 371 + 372 (`mn 346`, 6/8, `act_dur 3/4` then `3/8`, both at offset
  0): `3 + 3/2 = 9/2 > 3`: `overlength: measures '371', '372' (number 346)
  reach 9/2 quarters into a bar of 3`.

*sq8913219* warns once with seven lines, in this order: overlength
`'77', '78' (number 76)` 8 in 4 (mc 78 is the first ending, numbered like
the bar before it); short `'221' (number 219)` 3 in 4 (a bar cut short
before the change to 3/8; mc 221 states no `next`, so its successor is mc
222, `mn 220`, at offset 0: `short: measure '221' (number 219) ends 3
quarters into a bar of 4, and measure '222' (number 220), which it leads
to, does not complete the bar`); offset mismatch `'483' (number 480)`: mc
482 sounds `3/2` from offset 0, mc 483 states offset 2 (`mc_offset 1/2`),
so `offset mismatch: measure '483' (number 480) states offset 2, not 3/2
where '482' ends`; overlength `'533', '534' (number
529)` 5 in 3; overlength `'626', '627' (number 617)` 7 in 4; overlength
`'675', '676' (number 663)` `15/2` in 4; overlength `'684', '685' (number
671)` 8 in 4. The compound [583, 584] (3/4 bar before the *fine*, then the
2/2 pickup of one quarter at offset 3) reaches `0 + 3 + 1 = 4` into the
2/2 bar of its last record and adds up.
Where a score's structure still has no lattice (the refusal above: only
measures that cannot be joined and share a nominal downbeat), the timeline
carries no floating-measure conversion; its absence is honest, and a
silent fallback to label-derived ordinals would be the worst choice, because
nothing downstream could tell the two kinds of fm apart.

**The inverse** interpolates linearly over the same knots and performs no
truncation-reconstruction: a value that was truncated on the way out carries at
most one thousandth of a bar of error, and that is documented rather than
"corrected". `limit_denominator` and every other ratio-guessing device stay
banned. Inverting exactly-representable values is asserted instead: on the
eroica grid `0.750` inverts to `0.0` quarters (the pickup's sounding
onset) and `2.0` to `2.5`, the start of measure 2. Its knots on the fm side
are floats, so the inverse computes in doubles and answers floats even into
`quarters`, whose default is fractions: a `Fraction` there would be the
float rebuilt as a ratio (see *A map whose own parameters are floats
computes in doubles*). The Wagner residue is the float
`-0.0040000000000000036`.

**Zero tolerance:** every fm value, and every quarters value the inverse
answers, is asserted as the exact float, type and value together.

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
`pytest.approx`). 9 `approx` sites remain, and they belong to these classes:

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

`test_ticks_to_quarters_fraction_exact` is exact: an integer tick count into
quarters (a fraction-canonical target) answers `Fraction(ticks, ppq)`, of type
`Fraction`, by equality.

A `ScalarMap` inside a `CombinationMap` (`test_constant.py`) answers its
float target as `scalar × input`: `433.0 * (30.0 / 866)`, compared with `==`.

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

Every map's `to_dict()` is its manifest form (see *Manifest forms*):
rational parameters are exact Fraction pairs, integers JSON integers, floats
JSON numbers, and `from_dict()` reads that shape and nothing else. `name` is
written only when given, and every reader passes it to the constructor, so a
custom map name round-trips.

The format, the fixpoint guarantee, and the cross-package JSON-safety sweep
are specified in `tests/core/README.md` (`test_number_json.py`), which holds
the map name and `ConstantMap` `Fraction` round-trip tests.

## Integration Tests

Integration with the `Timeline` class is tested in `tests/timelines/test_maps_integration.py`.

## Validity Rationale

These tests verify the core mapping capabilities required by the TTA model:
- **Accuracy**: Mathematical transformations must be precise (using Fractions where possible).
- **Invertibility**: All maps must support inversion for bi-directional alignment.
- **Non-Invertibility**: Some maps (RotationMap, FloorMap, CombinationMap) are explicitly many-to-one.
- **Manifest fidelity**: A written map converts exactly as the map it was written from; what cannot be written losslessly is refused.
- **Composition**: Maps must be composable to form complex chains.
- **Performance**: Array operations should use vectorized numpy implementations.
