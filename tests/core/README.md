# Core Module Tests

This directory contains tests for the `timetoalign.core` module, which provides fundamental types and enums for the TimeToAlign! library.

## Test Coverage Summary

| Module | Status |
|--------|--------|
| `core/time.py` (TimeScalar hierarchy + paired Fields) | Complete |
| `core/events.py` (pitch / harmony / Note / Measure + paired Fields) | Complete |
| `core/fields.py` (SemanticField base + arrow translator/builder/metadata) | Complete |
| `core/enums.py` | Complete |
| `core/ids.py` | Complete |
| `core/protocols.py` | Complete |
| `core/timestamp.py` | Complete |

## Metrical and realization scalars

The scalar tests establish the arithmetic vocabulary on which anchored metric
readings rely. Exact time-signature specimens (``6/8``, ``4/4``, common time,
cut time, and an additive eighth-note signature) pin grouping, division,
derived rods, beat count, span, and source spelling. Re-expressing each as
counted divisions produces one unit group per division, preserves its exact
division and name, and is idempotent. This rules out a meter reader that
forgets source grouping or silently substitutes a default.

A policy authored with both ``division`` and ``beat_size`` is accepted only
when the two exact quantities agree in quarters. A contradictory quarter and
whole-note pair must fail validation, while the matching pair constructs and
survives JSON unchanged. Testing every time-signature shape through both
round-trip equality and a second serialization proves that completion cannot
turn a policy into a different value.

Tempo specimens cover integer, float, decimal-string, and ``Fraction`` rates,
including ramps. Their exact stored fractions and JSON Fraction pairs are
asserted directly; positive-rate validation is checked at both ramp ends. A
ramp mean must lie strictly inside its span and cannot exist without a target
rate. Construction rejects exact values outside signed 64-bit storage before
serialization, while the largest permitted denominator survives.
These cases rule out delayed serialization failures and invented ramp meaning.

Negative-level beats demonstrate groove below the pulse, while a zero beat
index remains invalid.
Measure-map and forest values are deliberately absent from the paired-field
registry: they are immutable timeline structure, not event columns. Tempo and
BeatPolicy remain event scalars, so this separation rules out attaching
semantic-field machinery to node levels, spans, reading ids, or expansion
membership.

## Test Files

### `test_number_json.py` - Exact JSON numbers

The JSON number forms are validated before their behavior is pinned. Integers
remain JSON integers, floats remain JSON numbers, and `Fraction` values become
`[numerator, denominator]`. Boolean values and invalid Fraction pairs are
refused. A declared number type decides the reading, never the lexical form:
a Fraction member written as an integral float (`[1, 8.0]`, as a JavaScript
writer may respell it) is read as that integer whatever the number type, and
`is_fraction_json` recognises such a pair, while a non-integral member
(`[1, 8.5]`), a boolean member and a denominator below 1 stay refused. The
tests name one case for each rule, asserting the value and its Python type.

### `test_fraction_field_arithmetic.py` - Exact Coordinate Arithmetic

**Purpose:** Validates that arithmetic over coordinate and duration fields runs
in the representation the field declares, and that both sides of every result
cell agree.

A storage cell carries its number twice — as a float64 and as an integer ratio
— and **both sides are populated on every non-null row**. Which side is
authoritative is not something a reader may infer from the row; it is declared
once per field, in `number_type` metadata. The tests therefore assert three
things per result: the materialised scalar, the ratio members, and that the
float member equals `float(numerator/denominator)` exactly. A cell whose two
sides disagree is a bug even when the side a caller happens to read is right.

Field-with-field arithmetic asserts that **the left operand decides** the
result's representation: a fraction-typed `CoordinateField` plus a float-typed
`DurationField` yields a fraction-typed `CoordinateField`, with the float side
mirrored from the exact result. Scalar arithmetic asserts the same rule, so
`Coordinate(Fraction(1, 3), quarters) + 0.5` is `Fraction(5, 6)` and not
`0.8333333333333333` — the float operand is converted into the left operand's
representation *before* the addition, which is the whole point of declaring one.

Scaling is asserted separately and deliberately, under the rule **quantize
the result, never the operand**. A multiplier is dimensionless, so converting
it into the field's representation first is a category error; instead the
arithmetic runs in the most exact representation the two sides afford, and
rounding happens once, at the end, on the canonical side. Two cases pin it,
and between them they separate every candidate implementation:

* `Coordinate(101, ticks) * 0.5` is `50` — exactly `101/2`, then one
  half-to-even rounding. Coercing the operand first would give `0`
  (`round(0.5)`), and skipping the final rounding would give `50.5`.
* `Coordinate(Fraction(1, 3), quarters) * 0.5` is `Fraction(1, 6)` — `0.5`
  *is* exactly `1/2`, so the exact lane stays exact. Doing this one in float
  would yield an ugly dyadic instead of a sixth.

Additive operands follow the opposite rule and are converted first, because
an added quantity is measured in the field's unit rather than standing
outside it.

Unit mismatch and cross-timeline-id mismatch are asserted to raise `TypeError`
once per call, from metadata, never per row.

**Validity Rationale:** Checking the ratio members directly catches precision
loss that value comparisons alone would hide, because the float side of an
exact cell is *supposed* to look right. Checking that the two sides mirror each
other catches the subtler failure: a cell that is internally inconsistent
answers different questions differently depending on which accessor a caller
reaches for.

### `test_exact_column_arithmetic.py` - Exact column arithmetic never overflows

**Purpose:** Proves that column arithmetic on an exact axis (`int`,
`fraction`) is exact whatever the size of its intermediates, refuses only a
result whose lowest terms do not fit the cell's int64 members, and gives
every cell the scalar engine gives.

**What would prove it.** `n1/d1 + n2/d2` is computed as
`(n1·d2 + n2·d1) / (d1·d2)` and then reduced. A float written into a
`fraction` axis becomes its exact dyadic ratio, with a denominator up to
`2⁵⁵` for a value like `0.1`; two such operands cross-multiply past `2⁶³`
although their sum, reduced, has a denominator of `2⁵⁵` again. An engine
that multiplies in int64 wraps silently or overflows on those rows; one that
detours through doubles returns a rounded sum. Each test therefore fixes
operands whose cross-products exceed int64 and asserts the exact reduced
result, written out as a literal ratio computed independently with Python's
`Fraction` (never by the engine under test), and the float side as the
correctly rounded double of that ratio:

- **Float-mirrored sums.** A quarters/`fraction` column of starts
  `[0.1, 3/2, 0.7, 1/3, -0.3]` plus durations `[0.2, 0.1, 0.2, 0.25, 0.1]`
  gives exactly `10808639105689191/2⁵⁵`, `57646075230342349/2⁵⁵`,
  `16212958658533785/2⁵⁴`, `7/12` and `-7205759403792793/2⁵⁵`, with floats
  `0.30000000000000004`, `1.6`, `0.8999999999999999`, `0.5833333333333334`,
  `-0.19999999999999998`. Rows 1 and 3 have small products, rows 0, 2 and 4
  do not, so one column mixes both ways of computing a row.
- **Subtraction** (`end - start`, a duration): `0.3 - 0.1` is
  `7205759403792793/2⁵⁵` (`0.19999999999999998`) and `7/2 - 0.7` is
  `12610078956637389/2⁵²` (`2.8`).
- **The refusal.** `1129/12 + 0.1` is, in lowest terms,
  `10179936597708269159/108086391056891904`: its numerator exceeds
  `2⁶³ − 1`, so no int64 cell holds it. The engine raises a `ValueError`
  whose whole message is asserted: it names the row (the position, or the
  caller's row number when one is passed), both operands, the reduced result
  and the member that does not fit. On an `int` axis `2⁶² + 2⁶²` is refused
  alike (`9223372036854775808`). The `int` lane computes on integers, never
  on doubles: `(2⁵³ + 4) - (2⁵³ + 1)` is exactly `3`, and `2⁵³ + 1` plus
  `0` stays `2⁵³ + 1`, which a double cannot hold.
- **Column and row agree.** A seeded table of 400 rows mixing small ratios,
  float-mirrored dyadics, negatives, integers, a few `null`s and value-only
  cells (a cell carrying only its float side) is combined under all five
  operations on both exact lanes, against a column operand and a scalar
  operand. Every output cell must equal
  `number_cell(combine_numbers(a, b, op, number_type), number_type)` of its
  row, all three members, the float compared bit for bit (`==` on a double
  that is the correctly rounded quotient, never a tolerance). The `int` lane
  also runs `multiply` and `divide` by a non-integral factor under each of
  the four rounding modes, so the rounding of a ratio is checked against
  Python's own on ties, negatives and exact quotients.

**Validity Rationale:** The expected ratios come from Python's
arbitrary-precision `Fraction`, which cannot overflow, so a test passes only
if the engine is exact on the rows where int64 would not be. The agreement
test closes the remaining gap: whichever way a row is computed, the scalar
engine is the reference, so a fast path that diverged from it on any cell
(a sign, a rounding tie, a mirror off by one ulp) fails.

### Display shows what the value carries

One formatter (`format_number`) sits behind every pretty rendering — scalar
`__str__`, stamp display, table cells. Having exactly one is the point rather
than a tidiness preference: a display that shows less than the value carries
is a lie the reader has no way to detect, and two formatters drift into
precisely that. Before unification, `str(Coordinate(Fraction(1, 3),
quarters))` read `0.333333` while the stamp rendering of the same position
read `1/3`.

Pinned, exactly:

| Value | Renders |
|---|---|
| `Fraction(1, 3)` quarters | `1/3 quarters` |
| `Fraction(19, 2)` quarters | `19/2 quarters` |
| `Fraction(10, 1)` quarters | `10 quarters` — integral, so no denominator |
| `2.0` seconds | `2 seconds` — integral, so no decimal point |
| `0.1` seconds | `0.1 seconds` |
| `1/3` (float) seconds | `0.3333333333333333 seconds` — every digit it round-trips with |
| `1234567.5` seconds | `1234567.5 seconds` — large values are not rounded away |
| `1e-7` seconds | `0.0000001 seconds` — never scientific, never `0` |
| `160` ticks | `160 ticks` |

The last two rows were live defects, not hypotheticals: the old scalar
formatter rendered `1234567.5` as `1234568` and `1e-7` as `0`. A test asserts
the scalar and stamp paths produce identical strings for the same value, so
they cannot drift apart again. `repr()` is the other lane and was already
exact — it is left alone.

`Interval` is the third rendering path and joins the same formatter.
`str(Interval(...))` shows the half-open span with the shared unit named
once, because the endpoints are validated to carry the same unit and
repeating it would only add noise:

| Endpoints | Renders |
|---|---|
| `Fraction(1, 2)` → `Fraction(3, 2)` quarters | `[1/2, 3/2) quarters` |
| `Fraction(10, 1)` → `Fraction(12, 1)` quarters | `[10, 12) quarters` — integral, so no denominator |
| `0.1` → `2.0` seconds | `[0.1, 2) seconds` |
| `160` → `480` ticks | `[160, 480) ticks` |

**Validity Rationale:** an interval is exactly the place a second formatter
would go unnoticed. Its endpoints are ordinary coordinates, so a hand-rolled
`f"[{start.value}, {end.value})"` renders plausibly for the common cases and
silently drops to `0.5` on a value the scalar path shows as `1/2` — the same
defect class the single formatter exists to remove, reintroduced one level up.
The tests therefore pin each endpoint kind rather than one representative
interval, and assert that each endpoint renders byte-identically to the bare
`Coordinate` carrying that value. `TimeIntervalStamp` heads its display with
the interval's own rendering rather than assembling brackets a second time,
so there is one bracket form in the library, not two.

#### A C-Map row shows what `get_unit` returns

A stamp renders each attached C-Map as a row, and a caller can ask the same
stamp for the same conversion through `get_unit`. These are two readings of
one number, so they must agree; when they did not, the display was the one
that was wrong, and wrong in two different ways at once.

The display built its row with `Coordinate(value, target_unit)` — the
*construction* path. Construction refuses an exact non-integral value on an
integer-valued unit (a deliberate ratio is not a rounding candidate), so a
conversion into pixels whose result was a ratio made `repr()` of an ordinary
stamp **raise**. A display raising is the worst available outcome: the object
becomes uninspectable exactly when a reader most wants to look at it. And
construction keeps an exact input on a float-canonical unit (the
never-degrade rule), so a seconds row printed `Fraction(3278347, 7350)` where
`get_unit("seconds")` returned `37.79265306122449`.

Both symptoms are one cause — a conversion result crossing into a unit is a
re-expression, not a construction — and the fix is one shared boundary
(`Stamp._on_unit`) that both lanes call. Pinned on a stamp whose maps cover
all three target classes, each with a non-integral exact reading:

| Target unit | Declares | Row and `get_unit` both give |
|---|---|---|
| `pixels` (from a `fraction` axis) | `int`, locked | `2` — rounded, because there is no fractional pixel |
| `seconds` (from an `int` axis) | `float` | `446.03360544217685` |
| `seconds` (from a `fraction` axis) | `float` | `446.03360544217685` — the same |
| `quarters` (from an `int` axis) | `fraction` | `Fraction(1, 3)` |
| `beats` (from an `int` axis) | `fraction` | `Fraction(3, 4)` |

**The target decides alone.** The two `seconds` rows are the pair worth
reading together: one conversion read off an exact axis and off an
integer-locked axis, reporting the same number. An earlier attempt let the
*source* axis's representation ride along wherever the target admitted it,
which looks reasonable — seconds do accept exact values — and is wrong twice
over. It makes one map answer differently depending on where it is read from,
so a caller comparing a score position against a scan position is comparing
two spellings; and on `floating_measures` it rendered ordinary readings as
forty-digit dyadics, because the exact quarters axis feeding them propagated
its own kind onto a float-canonical result. Carrying the source's
representation across a conversion is the provenance-in-the-type pattern the
boundary rule exists to remove. The last two rows are the reverse case, kept
so the rule cannot be mistaken for a preference for floats: a
fraction-canonical target stays exact from an integer-locked source.

**Validity Rationale:** asserting only that `repr()` no longer raises would
pass on a display that silently omitted the offending row — the existing
`except Exception: continue` guard would happily swallow it. The tests
therefore assert the rendered row's exact value against the `get_unit` value
for the same stamp and unit, so a skipped row fails as loudly as a wrong one.
Both `repr()` and `_repr_html_()` are exercised, since they are separate
callers of the same row set.

`get_conversion_for(key)` is the third reader of the same maps — it addresses
one by name rather than surfacing all of them — and it goes through the same
boundary, so naming a map cannot get a different number from displaying it.
Its contractual job stays label and structured maps (unit-valued conversions
belong to `get_unit`); it accepts a unit name as a convenience, and that
convenience is exactly where a fourth spelling of one value would have
appeared.

#### A map's own answer is written on its target axis

The same rule one level down: `ConversionMap.__call__` is public output, so
its result is written the way the target unit writes numbers — the map's
internal arithmetic representation is not the caller's business. A
`seconds`-target map answers `1.0`, not `Fraction(1, 1)`; a `quarters`-target
map answers `Fraction(3, 2)`, not `1.5`; a `ticks`-target map answers a whole
tick. Maps with no target unit (label and structured maps) pass through
untouched, and so does the array lane, which stays float64 by contract.

This has a consequence worth stating plainly rather than hiding in a
tolerance: **a float that is finer than a discrete axis does not survive a
round trip through it.** 1.234375 quarters at 96 ppq is 118.5 ticks; there is
no such tick, half-to-even names 118, and reading back gives `59/48`
quarters. The old property test asserted the original float came back within
1e-9 and passed only because the tick leg used to carry a fraction of a tick.
It is replaced by two exact assertions — tick-aligned values round-trip
exactly, off-tick values name the tick they land on — rather than by a wider
tolerance.

A map whose answer is an *ordinal* rather than a position on the target axis
declares that itself: `MetricMap` returns a measure count and `FloorMap` a
bucket index, both whole numbers however the nominal target unit writes
coordinates. They set `_declared_output_number_type`, which is the same
channel the rule already reserves for "the map carries one".

`ConversionMap.output_number_type` is the single expression of this rule —
declared output type, else the target unit's default, and nothing about the
source. Every reader consults it: `_conversion_rows`, `get_unit`,
`get_conversion_for`, timestamp-table C-Map column metadata, and
`__call__` itself. The preservation suite asserts all five together on one
map rather than each in isolation, because the property under test is that
they cannot disagree.

### Scalar construction: the unit chooses the representation

A scalar holds exactly one value, so something has to decide how it is
written — and that decision belongs to the unit, not to whichever Python
type the caller's literal happened to have. `Coordinate(2, quarters)` and
`Coordinate(1.5, quarters)` are both quarter positions and both come out
exact (`Fraction(2, 1)`, `Fraction(3, 2)`); they no longer differ because one
was typed with a decimal point. The tests pin each arm, because the failure
mode here is two adjacent literals on the same unit disagreeing about their
own type, which then propagates through arithmetic (the left operand decides)
into results that differ depending on operand order.

An explicit `number_type=` overrides the default and is validated against
what the unit admits — that is how a caller deliberately chooses the float
lane on a fractional unit.

Coercion toward the default never loses information, and each rule is pinned:

| Input | Unit default | Result | Why |
|---|---|---|---|
| `2` | `fraction` | `Fraction(2, 1)` | exact widening |
| `1.5` | `fraction` | `Fraction(3, 2)` | exact dyadic |
| `0.0001` | `fraction` | dyadic, denominator `2**66` | **no int64 ceiling here** — Python `Fraction` is arbitrary-precision, and the storage refusal belongs at the field boundary where the ceiling actually exists |
| `10` | `float` | `10.0` | widens, and raises if it would not survive the trip |
| `Fraction(1, 3)` | `float` | `Fraction(1, 3)` **kept** | degrading an exact input silently is the thing this scheme exists to prevent; it takes an explicit `number_type=float` |
| `120.7` | `int` (discrete) | `121` | a measurement read off a float clock |
| `Fraction(5, 24)` | `int` (discrete) | **raises** | an exact non-integral value is a unit mix-up, not a rounding candidate |

The 0.0001 row is worth reading twice: it is the case where the scalar path
and the storage path deliberately differ. The scalar accepts it and holds the
exact ratio; storing it in a *fraction-canonical field* raises, because that
is where int64 lives. Both are asserted.

**Everything the library hands back is expressed the same way.** The table
above governs the constructor, and the identical rule governs every other
route a value can take onto this axis — see "Number type is preserved
everywhere" below. `Coordinate(9.5, quarters)` and
`timeline.get_timestamp(9.5).get_coordinate_for("clt_1", format="coordinate")`
both give a `Coordinate` containing `Fraction(19, 2)`; there is one answer per
axis, not one per entry point.

#### Number type is preserved everywhere

**The declared type is preserved under all circumstances.** Whatever a value
was before and however it was obtained — typed by a caller, converted through
a C-Map, computed by offset arithmetic, interpolated between anchors, or read
back off a query — it is re-expressed in the declared `number_type` of the
axis it lands on. A fraction-canonical axis yields fractions, always; a
float-canonical axis yields floats, always. `express_as` is the one place
that happens.

```python
Coordinate(9.5, quarters)                          # Fraction(19, 2)
timeline.get_timestamp(9.5).get_coordinate_for("clt_1", format="coordinate")
```

The two agree, and that is the property worth having: a caller reasoning
about a fraction-canonical timeline never has to inspect each value's kind to
find out what it got.

**Estimation is not a type.** It is tempting to let an interpolated position
stay float on an exact axis so that its type "says" it was estimated — and
`tests/test_number_type_preservation.py` exists to reject exactly that. A
number's type says how it is written, not where it came from. Overloading it
with provenance makes two coordinates on one axis disagree about what they
are, and encodes a story nothing can query. Whether a position was estimated
is recorded where it can be asked about: `is_interpolated` and its siblings.

Re-expressing a float as its exact dyadic is not fabrication — the value is
numerically identical, digit for digit. Fabrication means inventing a
*tidier* ratio than the number really is, and that remains banned and
grep-enforced.

Internal machinery may still compute in float where that is what the
arithmetic affords — a WarpMap's vectorised lane, private numeric pair feeds.
The rule binds at the output boundary, not in the middle of a calculation.

**Every producer of a stamp, not just the one that was easiest to test.** A
timeline, a group and a bundle can all be asked for a position, and the axis
they report has to be written the same way, or the rule is a property of one
entry point rather than of the axis. Asserting it on `Timeline.get_timestamp`
alone is what let two live paths drift: the group inferred the axis
representation from the *argument's* Python type, and the bundle passed its
internal float query straight through. Both are the provenance-in-the-type
pattern this section rejects, and neither showed up as a failure because
nothing asked them the question.

The group case is asserted in **both directions**, because a one-directional
check passes for the wrong reason — a float argument on a fraction axis and a
Fraction argument on a float axis fail differently, and an implementation
that merely widens ints would satisfy the first while still reading the
caller's type:

| Producer | Query | Axis declares | Asserted axis |
|---|---|---|---|
| `Timeline.get_timestamp` | `9.5` | `fraction` (quarters) | `Fraction(19, 2)` |
| `TimelineGroup.get_timestamp_at` | `9.5` | `fraction` (quarters) | `Fraction(19, 2)` |
| `TimelineGroup.get_timestamp_at` | `10` | `float` (seconds) | `10.0` |
| `TimelineGroup.get_timestamp_at` | `Fraction(10, 1)` | `float` (seconds) | `10.0` |
| `AlignmentBundle.get_matchstamp_at` | `Fraction(79, 1)` | `fraction` (quarters) | `Fraction(79, 1)` |
| `AlignmentBundle.get_matchstamp_at` | `79.0` | `fraction` (quarters) | `Fraction(79, 1)` |

The bundle rows pin the same literal arriving as an exact ratio and as a
float: one answer per axis, not one per entry point. Each producer is asserted
on the value *and* its Python type, since `Fraction(10, 1) == 10.0` compares
equal and an equality-only check would pass on a wrong-typed axis.

**The claims lane is a producer too.** A `MatchClaim` answers coordinate
queries, and so does the `MatchStamp` built from it — the same position, twice.
They disagreed. An event value read out of storage arrives as the cell's exact
side (`{value: 2.5, numerator: 5, denominator: 2}`), the scalar
never-degrade rule kept `Fraction(5, 2)` on a float-canonical seconds axis, and
so `claim.get_coordinate_for("audio")` said `Fraction(5, 2)` while
`claim.get_matchstamp().get_coordinate_for("audio")` said `2.5`. It travelled:
anchor reprs read `@5/2 seconds` and `@23/4 seconds`, and
`WarpMap.from_match_line` inferred `target_number_type=fraction` for an axis
that declares float.

An anchor coordinate is not a free-standing number; it is a position on a named
timeline, so the axis decides, and it decides when the claim is made:
`MatchClaim.from_events` receives the timelines and places each event's
position on its timeline's axis (unit and declared number type). The fixture
claims are made that way from the storage cells, so the claim holds `2.5`, not
`Fraction(5, 2)`, before any reader sees it. Both directions are asserted, for
the reason given above:

| Reader | Axis declares | Asserted |
|---|---|---|
| `claim.get_coordinate_for` | `float` (seconds) | `2.5` |
| `claim.get_coordinate_for` | `fraction` (quarters) | `Fraction(4, 1)` |
| `claim.get_coordinates_for` / `get_coordinate` | both | same as above |
| `claim.get_matchstamp(from_graph=False)` | `float` (seconds) | `2.5` |
| `anchor.get_coordinate_for` / `repr(anchor)` | `float` (seconds) | `2.5` / `@2.5 seconds` |
| `MatchLine.get_alignment_anchors` | `float` (seconds) | `2.5`, `5.75` |
| `MatchLine.source_coordinates` | `fraction` (quarters) | `Fraction(4, 1)` |
| `WarpMap.from_match_line` | quarters → seconds | `fraction` → `float` |

**Validity Rationale:** the storage cell is the specimen that matters. Feeding
`from_events` a plain Python `2.5` never reproduced the defect, because the
never-degrade rule has nothing to keep; the exact side of a real event cell is
what turns a float axis rational. A test built from hand-written literals would
have gone green against the broken code, which is why the fixture reads its
values out of an actual `EventData` table.

A NOMATCH claim holds no coordinate (its event's position is the event's,
on its timeline), so the anchors are the only claim coordinates the rule
reaches.

**A claim is put on its axes when it is made, and only then.** A bundle reads
its claims as they are held and converts nothing, so the conversion has one
place: the claim's factory. A position in another unit goes through the
timeline's C-Map there, and the result is written in the axis's declared type.
Both directions of exactness are asserted, each on value and Python type
(`160 == 160.0`):

| Position given | Timeline (C-Map) | Claim holds |
|---|---|---|
| `Coordinate(1/3, quarters)` | ticks, `int` (`TicksToQuarters(480)`) | `160` (`1/3 × 480`) |
| `Coordinate(160, ticks)` | quarters, `fraction` (`QuartersToTicks(480)`) | `Fraction(1, 3)` |
| `Coordinate(1.5, seconds)` | samples, `int` (`SamplesToSeconds(44100)`) | `66150` (`1.5 × 44100`) |

End to end, `create_match_claims` projecting `1/3` quarters onto the ticks
timeline adds a Projection at `160` and the MatchClaim table's frame reads
`Fraction(1, 3)` in the quarters column and `160` in an `int64` ticks column:
the cells are the claim's coordinates as held.

The other half of the rule is the refusal: `add_match_claims` refuses a claim
whose coordinate is off its timeline's axis, whether in another unit (`1.5`
seconds on the samples timeline) or in the timeline's unit but another number
type (a float `4.0` on the fraction quarters axis), with a `ValueError` naming
the coordinate, the timeline and its axis. Each bad claim is sent in a batch
with a valid one, and the bundle holds no claim afterwards: every check
precedes any change. The valid claim is then added alone, so the refusal is
the bad claim's and not the batch's.

#### A published table writes the columns its axes declare

`group.get_timestamps_at(...)` and the group's frame present the same
positions, and they disagreed on dtype: the stamp getter gave `int64` /
`float64` / `object` per axis, while the frame gave `float64` for everything —
`12473.0` pixels beside a stamp's `12473`.

The cause is that a group's stored timestamps are a `float64` store by design:
it is the interpolation lane, and interpolation runs on doubles. Nothing was
wrong with the store; what was missing is that publishing is the boundary
where the declared type applies. It applies there now, driven by a
`number_type` entry on each field's metadata blob (the unit's default is the
fallback for older tables), so the store stays float and the published table
carries coordinate structs.

| Axis | stamp getter | published frame |
|---|---|---|
| `dgt_1` pixels (`int`) | `0`, `12473` | `0`, `12473` |
| `cpt_1` seconds (`float`) | `0.0`, `37.5` | `0.0`, `37.5` |
| `clt_1` quarters (`fraction`) | `Fraction(0, 1)`, `Fraction(19, 2)` | same |
| `pixels_to_beats` C-Map column | — | `Fraction(12473, 4)` |
| `quarters_to_seconds` C-Map column | — | `38.0` |

C-Map columns follow the same target-decides rule as C-Map rows: both read
`cmap.output_number_type`, so `quarters_to_seconds` is `float64` even though
the axis feeding it is exact, while `pixels_to_beats` stays exact because
beats are fraction-canonical. Timeline columns are a different question and
keep following their own axis — a `quarters` column is the timeline's own
coordinates, not a conversion of anything. Maps whose answer is an ordinal
(`MetricMap`, `FloorMap`) declare `int` and stay whole from every reader.

Nullable `Int64` appears only where a column actually has gaps — a group
member covering part of the span — so a dense integer axis reads `int64`,
exactly like the stamp getter.

**Validity Rationale:** the two lanes are asserted side by side in one test
rather than separately, because the property is *agreement*, and two
independently-pinned tables drift the moment one is updated. Both value and
dtype are asserted: values alone pass on a float column holding `12473.0`,
which is the defect. A group's fraction axis is included deliberately even
though it reconstructs the exact dyadic of a stored double rather than the
ratio that was authored — that lossiness is the interpolation store's, not the
table's, and pinning it here records which of the two lanes it belongs to.

### One stamp surface, one table surface

`test_stamp_retrieval_surface.py` (top level) asserts that the stamp lane and
the table lane are uniform across `Timeline`, `TimelineGroup` and
`AlignmentBundle`. Its subject is the shape of the API, not the arithmetic
behind it — every value it pins is one a reader can derive by hand from the
tiny fixtures it builds.

**The four precise questions, and one dispatcher over them.** Each receiver
answers a position query with `get_<stamp>_at` / `get_<stamp>s_at` and an
identity query with `get_<stamp>_for` / `get_<stamp>s_for`. The convenience
dispatcher `get_<stamp>(at=...)` chooses among them from the runtime form of
its argument alone. The matrix asserted per receiver is:

| `at` | selected | asserted result |
|---|---|---|
| scalar coordinate | `_at` | equals the precise call's result |
| coordinate collection | `_ats` | `list`, one entry per input, in order |
| `str` | `_for` | equals the precise call's result |
| key collection | `_fors` | `list`, one entry per input, in order |
| mixed keys and coordinates | — | `TypeError` |
| `bool` | — | `TypeError` |
| empty collection | `_ats` | `[]` |

The dispatcher result is compared against the precise getter's result rather
than re-derived, because the property under test is *selection*: a dispatcher
that computed the right numbers by a different route would still be wrong the
moment the precise method changed. `bool` is rejected even though it
subclasses `int`, matching the scalar layer; a mixed collection raises rather
than splitting element by element, because there is no honest answer to "some
of these are positions and some are names".

**Deleted names are gone, not aliased.** `get_timestamp_of`,
`get_timestamps_of` and `to_dataframe` are asserted absent on both `Timeline`
and `TimelineGroup`, and `as_fractions` is asserted not to be an accepted
keyword. Asserted rather than assumed because a re-export is invisible to
anything else in the suite: every migrated call site would keep passing.

**`get_timestamp` no longer shadows `get_timestamp_at`.** The two were once
the same function object, so an override of one silently changed the other and
a caller could not tell which name carried the behaviour. The test asserts
they are distinct functions and that the implementation sits on
`get_timestamp_at`.

**The table format vocabulary is closed.** `format="table"` gives a
`pa.Table`, `format="dataframe"` a `pd.DataFrame`, anything else a
`ValueError` naming both accepted values. The DataFrame-shaping options
(`fields`, `units`, `include_ids`) default to `None` rather than to their
effective values, so "not passed" is distinguishable from "passed the
default"; supplying any of them with `format="table"` raises a `ValueError`
naming exactly the offending options. Silently ignoring them would be the
alternative, and it hides a caller's mistaken belief that the Arrow result was
shaped.

**Cells are coordinate structs, and a frame cell is a scalar.** Every
timestamp table — a timeline's, a group's, a bundle's match-stamp table —
carries `IdCoordinateField` columns for timeline axes and `CoordinateField`
columns for derived conversions, each cell holding the number twice (a
float64 and an integer ratio) with `unit`, `number_type` and identity in field
metadata. The frame rendering decodes that to one scalar per cell.

The named regression is `test_authored_ratio_survives_the_frame`: a position
authored as `Fraction(5, 3)` quarters must read back as `Fraction(5, 3)`, not
as `7505999378950827/4503599627370496`. Both are the same number to sixteen
digits and only one of them is what the user wrote; a `double` column cannot
tell them apart, which is exactly why the column is a struct. The parquet
round trip asserts the same value survives a write and a read, since storage
is where the distinction would otherwise be lost for good.

**Every route to an axis is covered, not just the convenient one.** The three
ways a table's axis is sourced reach the column by different code, so each
gets its own exactness assertion: positions the caller passes (`at=[...]`),
positions collected from events (`at=None`, what a bare
`get_timestamp_table()` gives), and positions collected from timeline
boundaries (`include_boundaries=True`, where the ratio comes from an offset
and a length rather than from any event). Asserting only the first is what
let the event route ship reporting the dyadic while the stamp lane on the
same timeline reported the ratio. The MatchClaim table is covered over both
ways claims join a bundle, one MatchClaim at a time and a columnar
`MatchClaimField`, because both land in it and the table lane reads it column
by column in bulk.

**Known limitation — a group's stored rows are a float lane.** A
`TimelineGroup` keeps its timestamps as `float64` because interpolation
between members runs on doubles, so a boundary authored as an exact ratio is
rounded on the way into the store and reads back as that double's dyadic.
This is a property of the store, not of the table encoding, and it is narrow:
every *queried* position is exact — the stamp lane, the coordinate lane, and
`at=`-queried table rows all answer `5/3`.
`test_group_stored_boundary_rows_are_a_known_float_lane` pins both halves —
the loss on stored rows and the exactness of every query — so the limitation
cannot quietly widen, and so the day the store becomes typed the test fails
and names what changed.

**A batch answers completely or not at all — on both lanes.** A plural stamp
getter raises on the first element it cannot resolve and returns nothing: no
partial list, no NaN row, no silently dropped key. The table lane does the
same. An earlier draft let the table swallow an unresolvable query into a
null-filled row, which was wrong twice over: one input meant two things
depending on which exit the caller took, and a null became ambiguous — a
reader could not distinguish "this member does not reach this position" from
"your query position was invalid". Making the table raise leaves a null with
exactly one meaning, member not reachable, which is what makes nulls
readable. Both halves are pinned: the raise on an invalid query, and the
surviving null on a member that genuinely covers only part of the span.

### Stamp retrieval formats and canonical storage

A stamp stores one canonical `Coordinate` per present timeline. Retrieval uses
the six shared formats; the typed `IdCoordinate` is the default, and numeric
or Series projections are explicit.

| Result | Members | Why |
|---|---|---|
| **typed default** | `get_coordinate_for()`, dispatcher queries, `axis` | Carries the result timeline ID, canonical representation, and unit. |
| **explicit projection** | `format="coordinate"`, `"float"`, `"int"`, `"fraction"`, or `"series"` | Converts only when requested and applies the shared rounding contract. |

Rendered output (`__str__`, `_repr_html_`) reads from the exact lane and
formats it for a human: that same conversion displays as `1/3 quarters`.
Non-numeric conversion outputs — labels, mappings — are not numbers in either
currency and pass through both lanes untouched.

Collection retrieval validates every member before producing output. A pandas
Series input preserves its index; a list of event keys becomes the output
index; scalar Series results use a one-row index naming the queried axis or
key. Float and integer canonical axes use `float64` and `int64` Series dtypes,
while exact ratios and heterogeneous typed values use `object`. Empty results
retain the dtype implied by the selected axis.

**Interpolated values use the declared canonical number type.** Estimate
provenance is carried by `is_interpolated`; it does not change the coordinate's
number type. Exact anchors use the same canonical representation.

### `test_interval.py` - Typed interval scalar and field pairing

An `Interval` accepts equal endpoints and rejects reversed endpoints, mixed
units, or mixed canonical number types. `duration` preserves the shared unit
and number type exactly. `IntervalField` stores both coordinate structs under
one metadata triple; indexing reconstructs `Interval | None`, and its `start`,
`end`, and `duration` accessors return the paired semantic field types without
changing exact rational values.

### `test_projection.py` - The Projection event and its reference

**Purpose:** A `Projection` is an event standing on a timeline for an event
not held there: its ID (`None` until a timeline mints one), its position on
the timeline's axis (`start`, and `duration`, `None` for an instant) and
`projection_of`, an `EventReference` `{timeline_id, event_id}` or `None`. It
repeats nothing else of its source. These tests prove the scalar holds
exactly that and the paired `ProjectionField` stores it as TimeToShare's
Projections table does, without losing a value; adding Projections to a
timeline is tested in `tests/timelines/test_projections.py`.

**What proves it:**

- **The stored shape is the scalar's properties and nothing else, each
  position a coordinate cell.** `ProjectionField.pa_schema` is exactly
  `struct<id: string, start: COORDINATE_CELL_TYPE, duration:
  COORDINATE_CELL_TYPE, projection_of: struct<timeline_id: string,
  event_id: string>>` (the cell of `test_coordinate_cell.py`, the same as a
  MatchClaim's anchors), and `EventReferenceField.pa_schema` exactly
  `struct<timeline_id, event_id>`: a reference is an object of two values,
  never a joined string.
- **Construction.** An instant has `duration` and `end` `None`; an interval
  `start` 27/2 quarters with `duration` 1/3 has `end` exactly 83/6 quarters
  (a `Fraction`, not a float). A Projection without an ID has `id` `None`
  (`repr` `Projection(start=3 samples)`). `EventReference` and `Projection`
  refuse a key they do not have; both refuse an empty ID; `Projection`
  refuses a duration in another unit than its start and a duration written
  in another number type (both messages pinned, naming the Projection by its
  ID or as `A Projection` without one): start and duration are on one axis.
- **The event row.** `to_event_row()` is exactly `{event_type:
  "Projection", id, start, duration, projection_of}` with the scalars
  themselves; without an ID the row has no `id` key, so the timeline mints
  one.
- **Arrow round trip with exact values**, on three axes: quarters written as
  fractions (start 27/2, duration 1/3), samples as integers (850176,
  duration 22050), seconds as floats (0.1). Each field is written to a
  Parquet file in `tmp_path` and read back: every Projection read back
  equals the one written, its positions in the number type of the axis
  (`Fraction`, `int`, `float`), and the stored cells are exactly `{13.5,
  27, 2, "quarters", "fraction"}`, `{0.333…, 1, 3, "quarters",
  "fraction"}`, `{850176.0, 850176, 1, "samples", "int"}` and `{0.1,
  3602879701896397, 36028797018963968, "seconds", "float"}` (a float's
  exact ratio filled). The field metadata holds only the field type and the
  blob version: no axis is recorded per field.
- **`projection_of` set and `None`** survive the round trip alike: the
  first Projection keeps `EventReference(clt_6, x1azmn68)`, the second
  reads back `None` from a null cell.
- **Axes per cell.** One field holds Projections at samples (int), seconds
  (float), quarters written as float and quarters written as fraction,
  with a null row and a Projection without ID among them; after the round
  trip every Projection equals the one written and every cell names its own
  unit and number type. An empty field has the field's struct type.
- **TimeToShare's fixture Projections table**
  (`timetoshare.tables.fixture_path("projections.parquet")`, shipped with
  the installed package; never skipped), which is in the package form
  (each row's `unit` and `number_type` once, `start` and `duration` their
  number once): read by the package reader (`read_projection_table`
  against the fixture manifest), its eleven rows equal the eleven
  Projections spelled out in the test, in the fixture's row order, by
  timeline ID and each timeline's own order (one fraction on `clt_2`, the
  interval 7/2 quarters lasting 1/2; two floats on `cpt_1`, the instant
  10.0 seconds and the interval 12.5 seconds lasting 0.75; eight ints on
  `dpt_1`, one of them an interval; two without `projection_of`), each
  start and duration of its number type's Python type (`int`, `float`,
  `Fraction`), so each of the three number types is read
  from the member the package form writes it in. The fixture's `timeline_id`
  column is pinned in its row order (`clt_2`, `cpt_1` ×2, `dpt_1` ×8).
  `ProjectionField.from_projections` of those eleven, its cells written in
  the package form (`package_numbers` of each position's number by its
  number type, the row's `unit` and `number_type` from its start cell) and
  the fixture's `timeline_id` column added, in `PROJECTION_SCHEMA`'s order,
  equals the fixture table, and TimeToShare's `projection_problems` finds
  nothing.

### `test_number_storage.py` - One struct, one canonical side, one builder

**Purpose:** Pins the storage contract every coordinate in the library rests
on, and the rules by which a source value becomes a stored one.

**The struct.** `{value: float64, numerator: int64, denominator: int64}`. Both
sides carry the number on every non-null row; a null sub-field means the whole
row is null. Which side is authoritative is declared per field, in
`number_type` metadata, and never inferred from the row.

**What the builder must produce.** Expected values are exact, never ranges:

| Input | `number_type` | Canonical side | Mirror |
|---|---|---|---|
| `0.1` | `float` | `0.1` | `3602879701896397/36028797018963968` |
| `1/3.0` | `float` | `0.3333333333333333` | `6004799503160661/18014398509481984` |
| `2.0` | `float` | `2.0` | `2/1` |
| `1.5` | `float` | `1.5` | `3/2` |
| the same four | `fraction` | the same four ratios | `float(ratio)`, bit-for-bit |
| `[1.4, 2.5, -0.6]` | `int` | `[1, 2, -1]`, all `den = 1` | the ints as floats |

The float mirrors are the **exact dyadic** ratios, because every finite double
is exactly one rational with a power-of-two denominator. Nothing here searches
for a tidier ratio nearby: denominator-limiting approximation is absent from
the library, and the tests enforce that source-level invariant.

The `int` row settles ties on the even integer — `round(2.5) == 2`, Python's
own rule and PyArrow's `half_to_even`. This is asserted rather than assumed,
because the alternative (half-away-from-zero) drifts a column of `.5` values
consistently upwards, and a rounding rule that is only implicit is a rounding
rule nobody checks. Negative ties are pinned separately (`-2.5 → -2`,
`-1.5 → -2`, `-0.5 → 0`), since that is where the two conventions visibly
differ in sign and a positive-only probe would miss it.

All four modes are pinned, not just the default: for `[1.4, 2.5, -0.6]`,
`"floor"` gives `[1, 2, -1]`, `"ceil"` gives `[2, 3, 0]` and `"truncate"`
gives `[1, 2, 0]`. The same four names and the same `"round"` default apply
wherever the library makes a value integral, including `TimeScalar.to_int()`
— one vocabulary, one default, asserted in both places.

**The opt-in provenance flag.** `preserve_source_floats` keeps the incoming
floats on the float side of an `int` field instead of mirroring the stored
integer, so a cell can record what was measured beside what was stored:
`[1.4, 2.5]` stores numerators `[1, 2]` with `value` still `[1.4, 2.5]`.
It is construction-only — provenance floats do **not** survive arithmetic,
and a test asserts that adding zero rewrites them to exact mirrors
(`[1.0, 3.0]`). Otherwise a value that was never authoritative would
propagate as though it were.

**The vectorised path must equal the scalar path.** The builder takes a numpy
route for plain numeric columns and a per-value route for anything mixed,
textual or rational. The tests fuzz several thousand doubles — including the
sub-`2**-10` band where the exact dyadic denominator overflows int64 — and
assert the two routes agree cell for cell, and that the canonical float side
survives bit-for-bit in every case. Two implementations of one rule are two
chances to be wrong, so the equality is asserted rather than trusted.

**The vectorised path must also run where the developers do not.** Only the
vectorised route reaches numpy, so it alone can differ *by platform* — and a
rule the two routes share everywhere except on the platform nobody here runs is
not a rule the storage contract holds. The `platform_contract` fixture
(`tests/conftest.py`) therefore runs the route-agreement tests twice: once as
this machine offers numpy, and once with every C-typed ufunc argument narrowed
to 32 bits, as an LLP64 build (Windows) does. The difference is real: numpy
binds `ldexp`'s exponent to a C `int`/`long`, so an `int64` exponent finds a
loop where a long is 64 bits and finds none where it is 32, and numpy raises
rather than narrow it for you.

The builder answers that by never handing an exponent to a ufunc at all: the
denominator is a power of two, exactly representable as a double, so the mirror
is built by **multiplying** by it — exact for the same reason an exponent-taking
scale would be. Three tests hold that shape. The scale test asserts the same
exact ratios come out under both contracts. The width test guards the other
half: the exponent is a bit count, but what it shifts is not, and a denominator
accumulated in 32 bits would wrap past `1 << 31` silently and only for the
smallest values — so `2**-10` (denominator 1024) and the capped `0.0001`
(denominator `2**62`) are pinned. The reduction test pins the one-pass
reduction — the dyadic ratio's only common factor is a power of two, namely the
numerator's lowest set bit — with the bit to strip at the bottom (`2**-10`), in
the middle (`0.75`), absent (`0.1`), and with no bit to find at all (`0.0`,
which reduces to `0/1`). Each expected pair is spelled out, and each is checked
component-for-component against the value's own `Fraction`, which is in lowest
terms by construction.

**Where int64 runs out.** A double below roughly `2**-10` *may* need a
denominator past `2**62` — only the full-mantissa ones do, which is why the
fuzz test sees 362 of 10,009 rather than every sub-threshold value
(`2**-11` itself needs only 2048). Where it happens:

* the canonical float side always keeps the double untouched;
* the **mirror** falls back to `round(value * 2**62) / 2**62`, reduced.
  Worked example: `0.0001` is canonically `0.0001`, and its stored mirror is
  `461168601842739 / 4611686018427387904` — not `Fraction(0.0001)`, whose
  denominator is `2**66`;
* an **exact** field refuses the value outright rather than storing a
  degraded canonical side, and the error names `number_type float` as the
  place to put it. A canonical value is exact or it is an error; best effort
  belongs to mirrors only.

**The ceiling constrains storage, not retrieval.** Python `Fraction` is
arbitrary-precision, so `to_fraction()` and every as-fraction accessor
recompute the exact dyadic from the canonical float instead of reading the
stored mirror. Pinned directly: for `0.0001` the stored mirror is asserted to
be the approximation *and* `to_fraction()` is asserted to return the exact
`Fraction(0.0001)` anyway. Reading the mirror there would be a silent
exactness violation on precisely the values this limit identifies.

**Refusals.** An *exact* non-integral value entering an integer-valued field
raises — `Coordinate(Fraction(5, 24), TimeUnit.ticks)` is a unit mix-up, not a
rounding candidate, and rounding it to zero would bury the mistake somewhere
far from its cause. Inexact floats are made integral, because reading a tick
position off a float clock is exactly what rounding is for. `Duration` is
asserted alongside `Coordinate`: the rule lives on their shared base, so a gap
on one of them would be a design error rather than a missing branch.

**Conversion is the exception, deliberately.** Converting quarters to ticks at
a given resolution quantizes by definition, so `quantize_to_unit` rounds where
the constructor refuses. Tests assert both sides of that line, since a single
rule in both places would either forbid legitimate conversion or license the
silent rounding the refusal exists to prevent.

### `test_coordinate_cell.py` - A coordinate cell names its own axis

**Purpose:** A table whose rows lie on different timelines (the MatchClaim
table, the Projections table) cannot record one unit and number type in its
field metadata, so each cell carries them: `COORDINATE_CELL_TYPE` is the
number struct above followed by `unit` and `number_type`. In memory the cell
keeps both sides of its number; a package's tables write it in TimeToShare's
package form, the number once (below). These tests prove the cell is that
shape and holds exactly the number written.

**What proves it:**

- **The type.** `COORDINATE_CELL_TYPE` equals `struct<value: double,
  numerator: int64, denominator: int64, unit: string, number_type: string>`
  spelled out in the test.
- **Both sides on every non-null cell, a float included.** `coordinate_cells`
  of `[1/3 quarters (fraction), None, 0.1 seconds (float), 139776 samples
  (int), 2/3 quarters]` gives exactly, in that order: `{0.3333333333333333,
  1, 3, "quarters", "fraction"}`, a null cell, `{0.1, 3602879701896397,
  36028797018963968, "seconds", "float"}` (the double's exact ratio, never a
  guessed tidier one), `{139776.0, 139776, 1, "samples", "int"}`,
  `{0.6666666666666666, 2, 3, "quarters", "fraction"}`. Rows of one axis that
  are not adjacent keep their places. A `Duration` makes the same cell as a
  `Coordinate` of its value.
- **The column builder on one axis.** `axis_coordinate_cells([0, None,
  44100], unit="samples")` gives `{0.0, 0, 1, "samples", "int"}`, null,
  `{44100.0, 44100, 1, "samples", "int"}`; on a float axis `[0.25]` gives
  `{0.25, 1, 4, "seconds", "float"}`; without `number_type` the unit's
  default applies.
- **A column already of number structs is taken whole.** The `start` column
  of an event table is of the number struct type, and a builder that read it
  cell by cell would be correct and slow, so the proof has two halves. Same
  cells: the struct column `[{0.75, 3, 4}, null, {0.5, 2, 4}]` on a quarters
  axis gives `{0.75, 3, 4, "quarters", "fraction"}`, null, `{0.5, 1, 2,
  "quarters", "fraction"}` (a cell keeps its members, its ratio in lowest
  terms: memory holds one spelling of a value, so `2/4` is held as `1/2`),
  exactly what the per-row reader makes of the same cells given as dicts; a slice of a longer column, whose dropped first row and whose null
  cell hold numbers underneath, gives the cells of its own rows, and a
  chunked column the cells of all its chunks. No Python per row: with the
  per-row cell encoder replaced by one that fails, the struct column still
  builds the same cells. A non-null cell missing its ratio, `{0.5, null,
  null}`, is left to the per-row reader, which mirrors the float:
  `{0.5, 1, 2, "seconds", "float"}`.
- **One spelling of a value in memory** (`in_lowest_terms`). Every entry
  of whole number structs into memory reduces through it, so two spellings of
  one value never stand side by side to be compared member by member (the
  MatchClaim table's succession check, field equality and the Projection
  keys all compare members). What would show it broken: a ratio kept
  unreduced, a sign left on the denominator, another member touched, a
  null cell or a cell missing a side altered, a row moved. Evidence: the
  number column `[{0.5, 2, 4}, {-1.5, 6, -4}, {0.0, 0, 5}, {3.0, 3, 1},
  null, {0.5, null, null}]` gives exactly `[{0.5, 1, 2}, {-1.5, -3, 2},
  {0.0, 0, 1}, {3.0, 3, 1}, null, {0.5, null, null}]`; a coordinate cell
  column `[{0.5, 2, 4, "quarters", "fraction"}, {0.25, 1, 4, "seconds",
  "float"}]` gives `[{0.5, 1, 2, "quarters", "fraction"}, {0.25, 1, 4,
  "seconds", "float"}]` (its type kept, `unit` and `number_type` kept); a
  slice reduces its own rows. `number_cell` of the dict `{0.5, 2, 4}` gives
  `{0.5, 1, 2}` and of `{-1.5, 6, -4}` gives `{-1.5, -3, 2}`.
- **Reading back keeps the number type.** `cell_coordinate` of each built
  cell equals the coordinate written, and its value has the written Python
  type (`Fraction`, `float`, `int`); `scalar=Duration` gives a `Duration`; a
  null cell gives `None`; a cell without unit or number type is refused.
- **Empty input** gives an empty array of the cell type.

**The package form** (`TestPackageForm`). A package writes a number once:
the member its number type names holds it, the others are null — an int
`(n, null, null)`, a fraction `(n, d ≥ 1, null)`, a float
`(null, null, x)` with x finite — so a float is never also a ratio and an
int has no denominator. One encoder/decoder pair (`package_numbers` /
`numbers_from_package`, with `package_cells` / `cells_from_package` for a
cell that names its axis) converts at the file boundary; memory keeps both
sides. What would show it broken: a mirror left in the file, a side not
filled again on reading (an int read with a null denominator, a float
without its ratio), a malformed cell read as some number, an unreduced
fraction refused or read as another value than its reduced form, a
fraction rewritten on writing, a fraction's float not the one memory holds,
a wrong row named. Evidence:

- **The types.** `PACKAGE_NUMBER_TYPE` is `struct<numerator: int64,
  denominator: int64, value: double>` and `PACKAGE_COORDINATE_CELL_TYPE`
  that followed by `unit`, `number_type` (strings), each spelled out in
  the test and equal to TimeToShare's `NUMBER_TYPE` and `COORDINATE_TYPE`.
- **Written once.** `package_cells` of the five cells above gives exactly
  `{1, 3, null, "quarters", "fraction"}`, null, `{null, null, 0.1,
  "seconds", "float"}`, `{139776, null, null, "samples", "int"}`, `{2, 3,
  null, "quarters", "fraction"}`; a fraction cell holding `4/1` keeps its
  denominator `1` (it is a fraction). The encoder writes the ratio it is
  given and nothing else: a number column built by hand holding `2/4`
  (value `0.5`), `-6/4` and `0/5` as fractions (memory never holds them:
  every entry reduces) is written `(2, 4, null)`, `(-6, 4, null)`,
  `(0, 5, null)`, and its int and float rows are written as they stand.
- **Read back whole.** `cells_from_package` of that column equals the
  memory column built by `coordinate_cells` (`equals`, every member, the
  float's ratio `3602879701896397/36028797018963968` and the int's
  denominator `1` and float `139776.0` filled again), so a round trip
  loses nothing; a slice reads the cells of its own rows; members given in
  another order read alike (by name).
- **A cell not as its number type says is refused**, naming the column and
  row and what it holds, for each wrong pattern: an int with a
  denominator `(3, 1, null)`, an int with its float `(3, null, 3.0)`, a
  fraction without denominator `(1, null, null)`, a fraction with
  denominator `0`, a fraction with its float, a float with a ratio
  `(1, 2, 0.5)`, a float that is not finite (`inf`), a float without its
  value, a number type outside the three (`"decimal"`), a non-null cell
  without number type, and a cell without unit. The message of the first
  is pinned exactly (`Test row 1: the int cell (numerator, denominator,
  value) holds (3, 1, null), where an int is (n, null, null), a fraction
  (n, d >= 1, null) and a float (null, null, x) with x finite`); a null
  cell is no problem.
- **An unreduced fraction is read, as its reduced form.** Any fraction is
  a valid cell: `(2, 4, null)`, `(-2, 4, null)`, `(0, 2, null)` and
  `(-6, 4, null)` read without error as exactly the memory cells of
  `1/2`, `-1/2`, `0/1` and `-3/2` (numerator, denominator and float each
  equal to those of the reduced cells `(1, 2, null)`, `(-1, 2, null)`,
  `(0, 1, null)`, `(-3, 2, null)` read alike), so `cell_coordinate` gives
  `Coordinate(Fraction(1, 2), "quarters")` for both spellings and an
  unreduced cell compares equal to its reduced form wherever cells are
  compared; `(-1, 2, null)`, `(0, 1, null)` and `(4, 1, null)` read as
  `-1/2`, `0` and `4`.
- **A fraction's float is correctly rounded.** `(1152921504606847009, 3,
  null)`, whose numerator is past `2**53`, reads with value `3.843071682022824e+17`, which is `float(Fraction(n,
  3))` and which a float64 division of the two members does not give
  (`3.843071682022823e+17`); the ratio keeps the exact numerator. So a
  position read from a file and the same position built in memory share
  the float key that orders and deduplicates them.
- **Vectorized.** With the per-row cell encoder (`number_cell`) replaced
  by one that fails, `cells_from_package` and `package_cells` of 1,000
  cells still give the same cells.

### `test_numpy_portability_hygiene.py` - Integer widths the library chooses

**Purpose:** Keeps the storage guarantee ("an int64 numerator means 64 bits of
numerator wherever this runs") true by scanning the package for constructs that
hand the choice of width to the platform's C ABI instead.

**What is scanned, and why those.** The test parses every module with `ast` —
not a text grep, so prose and comments explaining a construct cannot trip it —
and reports `file:line` for:

| Construct | Why it is a decision, not a detail |
|---|---|
| `np.ldexp` | its exponent is a C `int`/`long`, so which loops exist differs by platform; an `int64` exponent binds on LP64 and raises on LLP64 |
| `np.int_`, `np.long`, `np.ulong`, `np.intc`, `np.uintc` | aliases for whatever integer the C ABI calls native, i.e. a different width on Windows than on Linux |
| `dtype=int`, `astype(int)` | the same question spelled with the builtin |

`np.frexp` is deliberately **not** on the list: its exponent is an *output*,
always a C int, and reading a 32-bit integer out is not a portability question.
Explicit widths (`np.int64(1) << shift`, `astype(np.int64)`, `dtype=np.int32`)
are exactly what the test asks for, and a companion case asserts they are left
alone — a scan that flags the recommended spelling would be uninstallable.

**Why the scan is itself tested.** A hygiene test that silently stops matching
is worse than none, since it reads as a guarantee. So every pattern is
exercised against a snippet containing it, and the recognised list is asserted
in full and in source order. The package passes with **zero** hits today; a new
use is not automatically wrong, but it must be argued — make the width
explicit, scale by a value rather than by an exponent, or record the exception
here and in `platform_contract` so the emulation still covers it.

### `test_coordinate_resolution.py` - Coordinate Input Resolution

**Purpose:** Validates the shared decomposition of public coordinate inputs and
the timeline-level policy that resolves timeline IDs and units.

The tests use exact values to cover raw integer, float, and `Fraction` inputs;
plain and timeline-qualified coordinates; conflicting IDs; unsupported input
types; exact preservation of native `Fraction` values; conversion through an
attached C-Map; missing conversion paths; unknown timeline IDs; and descendant
offset arithmetic. Descendant resolution covers two-level offset composition,
bare values qualified by `timeline_id`, and scalar and affine foreign-unit maps
registered on either the owning descendant or one of its ancestors. It verifies
that conversion precedes the exact upward offset composition. Public routing
coverage also verifies unknown timeline IDs through `to_dataframe()`, batch
group queries with independently qualified rows, missing IDs on raw batch
values, and conflicting embedded and explicit timeline IDs wherever both forms
are accepted. This separates the pure core decomposition contract from the
unit- and hierarchy-aware behavior owned by `Timeline.get_coordinate()`.

**Validity Rationale:** A single decomposition shape prevents callers from
silently discarding unit or timeline metadata, while timeline resolution makes
every public coordinate entry point apply the same conversion and child-offset
rules.

---

### `test_types.py` - Coordinate Type

**Purpose:** Validates the `Coordinate` dataclass, the fundamental unit for representing positions on timelines.

**Test Categories:**

1. **Creation Tests** (10 tests)
   - Integer, float, and Fraction values accepted
   - String units coerced to TimeUnit enum
   - Invalid types (str, None) rejected
   - Frozen model (immutable): attribute assignment raises `pydantic.ValidationError`
   - Hashable (usable in sets/dicts)

2. **Conversion Tests** (9 tests)
   - `to_float()`, `to_int()`, `to_fraction()` methods
   - Type preservation and truncation behavior

3. **Property Tests** (8 tests)
   - `number_type` property (INT, FLOAT, FRACTION)
   - `domain` property (physical, logical, graphical)
   - Boolean values rejected at construction by the `value` field validator,
     which raises `TypeError` — asserted as the concrete class, not a bare
     `Exception`. (The `TypeError` propagates unwrapped because pydantic only
     re-wraps `ValueError`/`AssertionError` raised inside validators.)

4. **Arithmetic Tests** (18 tests)
   - Addition/subtraction with same units
   - Different units raise TypeError
   - Scalar multiplication/division
   - Floor division
   - Zero division errors

5. **Comparison Tests** (8 tests)
   - `<`, `<=`, `>`, `>=` operators
   - Different units raise TypeError
   - Non-Coordinate comparison raises TypeError

6. **Utility Tests** (8 tests)
   - `__repr__`, `__str__` formatting
   - `is_zero()`, `is_positive()`, `is_negative()`
   - `with_value()`, `with_unit()` copy methods

**Validity Rationale:**

The Coordinate class is the atomic unit for all temporal positioning in TTA. Tests verify:
- Type safety (only valid number types accepted)
- Immutability (frozen dataclass prevents mutation bugs)
- Unit safety (operations between incompatible units are rejected)
- Mathematical correctness (arithmetic operations preserve semantics)

---

### `test_score_scalars.py` - Score Scalars

**Purpose:** Validates construction and protocol conformance for the score-level
scalars `MidiPitch`, `SpecificPitch`, `Note`, `Measure`, and `DcmlHarmony`.

**Test Categories:** construction, protocol conformance (`GenericPitchLike`,
`NoteLike`, `MeasureLike`, `HarmonyLabelLike`, `DcmlHarmonyLike`,
`SemanticTypeLike`), `semantic_type`, `metadata_dict`, and immutability.

**Concrete exceptions:** each scalar is a frozen pydantic v2 `BaseModel`, so
mutating a field raises `pydantic.ValidationError` (message contains
`"frozen"`). The `test_frozen_immutable` cases assert `pytest.raises(
ValidationError)` — the concrete class, not a bare `Exception`.

---

### `test_enums.py` - Domain and Unit Enums

**Purpose:** Validates the enumeration types that categorize domains, units, and event types.

**Test Categories:**

1. **FancyStrEnum Base** (3 tests)
   - Abbreviation lookup (dict and string formats)
   - `__repr__` and `__str__` formatting

2. **Domain Enum** (7 tests)
   - Values: `logical`, `physical`, `graphical`
   - String subclass (usable as strings)
   - Alias support (`lo`, `ph`, `gr`)
   - Invalid value handling

3. **TimeUnit Enum** (18 tests)
   - Physical units: `seconds`, `milliseconds`, `samples`, `frames`
   - Logical units: `quarters`, `floating_measures`, `ticks`, `number`
   - Graphical units: `pixels`, `points`, `inches`, `millimeters`
   - Alias support (`q`, `ms`, `px`, etc.)
   - `domain` property mapping
   - `is_discrete` property
   - The vector-graphics units `css_pixels` (the CSS pixel, 1/96 in) and
     `user_units` (a document's own coordinate system, such as an SVG
     viewBox): each is graphical and not discrete, defaults to `float` and
     admits exactly `float` and `fraction`, as `points` does. `uu` is the
     abbreviation of `user_units` (`get_abbreviations()` lists exactly `uu`
     for it, and `TimeUnit("uu")` and `TimeUnit.uu` are `TimeUnit.user_units`);
     `css_pixels` has none (`get_abbreviations()` lists none), and `px` stays
     the alias of the discrete `pixels` (`TimeUnit("px") is TimeUnit.pixels`):
     reading a CSS pixel as a discrete pixel would change the timeline class
     and the number type.

4. **NumberType Enum** (8 tests)
   - Values: `int`, `float`, `fraction`
   - `python_type` property
   - Construction from string, type, or instance

5. **EventType Enum** (5 tests)
   - Values: `instant`, `interval`
   - Alias support (`inst`, `intv`)

**Validity Rationale:**

The enum types enforce TimeToAlign's domain model:
- Three domains (Logical, Physical, Graphical)
- Each domain has discrete and continuous variants
- Units must be compatible within operations

---

### `test_addresses.py` - Typed measure and beat keys

**Purpose:** Validates the `Address` hierarchy — the typed keys that name a
measure or a beat rather than a position on a coordinate axis — and the
storage shape of the ones that are column payloads.

**Why these specimens.** Every value below is a real spelling from an
alignment or score source, chosen because it is exactly where a looser design
loses information:

* `"12+3/8"` is the performance-alignment spelling of "bar 12, three eighths
  in". The offset fraction counts **whole notes**, so the gold value is
  `3/8 × 4 = 3/2` quarters — a factor of four away from the reading that takes
  the fraction for quarters, which is why the conversion is pinned as an exact
  `Fraction` and not as `0.375`. The parsed key is a
  `MeasureNumberAddress(mn="12", at=Coordinate(Fraction(3, 2), quarters))`;
  a wrong answer here would place every note of a performance a dotted quarter
  off its bar.
* `"12a"` has no `+`, so it parses to a bare `MeasureNumber` and keeps its
  suffix. A parser that stripped the `a` would merge the two halves of a split
  bar into one address, which is the failure the label type exists to prevent.
* `MeasureId(value=13)` versus `MeasureId(value="m-0023")`: strictness is the
  whole point. `13` is a **position** — the thirteenth measure-like unit — and
  `"m-0023"` is an **identifier** looked up against the measures' own ids. A
  model that coerced `"13"` to `13` would silently turn an MEI `xml:id` into a
  count, so `MeasureId.count("13")` must raise rather than convert.
* `classify_dispatch_input(Address)` returning `"coordinate"` is the runtime
  half of the retrieval grid: an address names a structural position, so it
  belongs on the `_at` getters rather than the string-key `_for` getters.
* The Field round trip uses `"237b"` — the printed label of the second half of
  the split bar that is measure count 261 in Beethoven's WoO 71. The storage
  struct also carries optional rendition and section qualifiers;
  the round trip asserts the source facts because a
  struct that stored only an integer could not hold this label at all.

**Zero tolerance:** every offset is asserted as an exact `Fraction`, every
label as the exact string the source prints.

---

### `test_beat_policy.py` - How a bar is counted in beats

**Purpose:** Validates that a time signature is read into the two facts it
actually states — which note value is counted, and how those values group into
beats — and that every derived quantity follows from that pair.

**Why grouping + division rather than stored beat lengths.** A `6/8` bar is
not "six beats of an eighth"; it is two beats of three eighths. Storing the
rods directly would lose which encoding produced them, so `BeatPolicy`
stores `grouping` and `division` and derives the rest. The specimens:

* `from_time_signature("6/8")` → `grouping=(3, 3)`, `division=1/2` quarters,
  so `rods == (Fraction(3, 2), Fraction(3, 2))`. Reading `6/8` as six beats
  would give six rods of `1/2` — a different metrical claim, and one that puts
  "beat 3" in a different place.
* `from_time_signature("4/4")` → four rods of one quarter each. Simple meters
  count one division per beat, so the compound rule must not fire; `4 % 3 != 0`
  keeps it off, and the `n > 3 and d >= 8` guard keeps `3/8` simple.
* `from_time_signature("3/8+2/8+3/8")` → `grouping=(3, 2, 3)` over eighths, so
  `rods == (Fraction(3, 2), Fraction(1), Fraction(3, 2))` and
  `offset_for(3) == Fraction(3, 2) + Fraction(1) == Fraction(5, 2)`. This is
  the Don Giovanni-style uneven bar: any implementation that averages the
  grouping puts beat 3 at `5/3` instead of `5/2`.
* `uniform(Fraction(1, 2), 6)` is the override policy — counting a `3/4` bar in
  eighths — and gives six rods of `1/2` summing to the same three quarters.
  The same bar, counted two ways, must still span the same time.
* An unreadable signature raises. There is deliberately no silent fall back to
  `4/4`: choosing a default for a source that states nothing is a loader's
  decision, and a policy that guessed would report a metre the source never
  gave.

**Zero tolerance:** every rod, offset and span is an exact `Fraction`.

---

### `test_pitch_names.py` - Pitch names in one spelling

Every loader writes a pitch name's accidentals as `♯` and `♭`, so names from two sources
compare equal as strings. `canonical_pitch_name` (`core/events.py`) is the one respelling, its
column form `canonical_pitch_names` the one the loaders and the matching helpers call.

The proof is a table of cases, each with one exact expected string:

| Input | Output | What it shows |
|---|---|---|
| `Bb3` | `B♭3` | an ASCII flat |
| `F#4` | `F♯4` | an ASCII sharp |
| `F##4` | `F♯♯4` | a double sharp is two signs, as `SpecificPitch.get` writes it |
| `Bbb2` | `B♭♭2` | a double flat, the step `B` kept apart from the signs |
| `F#♯4` | `F♯♯4` | mixed spellings of one sign |
| `C#` | `C♯` | no octave |
| `C-1` | `C-1` | a negative octave is no flat |
| `Cb-1` | `C♭-1` | a flat before a negative octave |
| `eb5` | `e♭5` | the step is kept as written |
| `bb3` | `b♭3` | a step written `b` stays a step |
| `E♭5`, `C4` | unchanged | already canonical, no accidental |
| `""`, `rest` | unchanged | no pitch name |
| `Bb3\n`, ` Bb3`, `Bb3 ` | `B♭3\n`, ` B♭3`, `B♭3 ` | whitespace around a name is read past and kept |
| `G3, Bb4` | unchanged | two names are no pitch name |

A reading that counted `-` as a flat fails on `C-1`, one that respelled every `b` of the string
fails on `bb3`, and one that rewrote the step fails on `eb5`. The whitespace rows hold the
function to the reading of `SpecificPitch.from_string`, which strips the name and then matches
it whole: a match anchored by `$` alone reads `Bb3\n` and drops its newline, one without the
strip leaves ` Bb3` unrespelled although the parser reads it as B♭3, and one that matched a
prefix would respell `G3, Bb4` into a string that names neither pitch. Every output written
with an octave is read back by `SpecificPitch.from_string` to the pitch the input names, so the
spelling is the library's own. A value that is not a string raises `TypeError` with exactly
`A pitch name is a string, not int: 60`.

The column form is checked on a column holding a null, `rest`, a chord `G3,D4,Bb4,F#5` (with
`sep=","`: `G3,D4,B♭4,F♯5`; `G3, Bb4` with `sep=","`: `G3, B♭4`, the space kept) and a repeated
name, against the scalar function cell by cell, and the input column is left unchanged.

**Every column type a caller hands in.** A frame read from Parquet or CSV holds its names in
whichever type the reader chose, so the column form is run on the names `Bb3`, null, `F#4`,
`C4`, `Bb3` (index 10 to 14, column name `name`) in each type, and must give exactly
`B♭3`, null, `F♯4`, `C4`, `B♭3` with the index and the name kept and the input unchanged:

| Column type | Type returned | Why |
|---|---|---|
| `object` | `object` | the names are replaced in the column's own type |
| `str`, `string[python]`, `string[pyarrow]`, `ArrowDtype(string)` | the same | as `object`; the null stays the type's own null |
| `category` | `category`, categories `B♭3`, `C4`, `F♯4` | the categories are respelled, every row keeps its category |
| `ArrowDtype(dictionary(int32, string))` | the same | the dictionary is respelled, every row keeps its index |

A rewrite that writes new values into the column fails here twice: a categorical column
refuses a value outside its categories, and an Arrow dictionary column wrote another row's
name into the null. Two categories that respell to one name become one category:
`Bb3`, `B♭3`, null, `F#4` as an ordered categorical with the categories `F#4` < `Bb3` < `B♭3`
gives the values `B♭3`, `B♭3`, null, `F♯4` and the ordered categories `F♯4` < `B♭3`; as an
`ArrowDtype(dictionary(int8, string))` column it gives the dictionary `B♭3`, `F♯4` and the
indices 0, 0, null, 1. An Arrow column of two chunks with different dictionaries (`Bb3`, null
and `F#4`, `Bb3`) gives `B♭3`, null, `F♯4`, `B♭3`. The function takes a pandas column only;
a `pyarrow` array is not accepted.

### `test_ids.py` - Scoped ID System

**Purpose:** Validates the `ScopedId` class and the stateless generator of
manifest element ids. There is no process-wide id counter: `IdGenerator` is
deleted, so no test of one exists.

**Test Categories:**

1. **ScopedId Creation** (9 tests)
   - Basic creation with scope and local
   - Validation (no digits starting scope, no whitespace/colons in local)
   - Frozen and hashable

2. **ScopedId String Conversion** (3 tests)
   - `__str__` format: `scope:local` or just `local`
   - `__repr__` format

3. **ScopedId Parse** (4 tests)
   - Parsing from string with/without scope
   - Roundtrip: `parse(str(id)) == id`

4. **ScopedId Methods** (6 tests)
   - `with_scope()`, `with_local()` copy methods
   - `nested()` for hierarchical scopes
   - `is_scoped` property

5. **The manifest id generator** (`mint_id`, `validate_id`,
   `timeline_id_prefix`)
   - `mint_ids(prefix, taken, count)` is the library's one id generator
     (timelines, maps, manifest elements, MatchClaims, events): N continues
     above the highest number held under that prefix, so an id once given is
     never given again, a gap is never filled. `("claim", ["claim_2", "x"],
     3)` → `claim_3`, `claim_4`, `claim_5`; nulls in `taken` are ignored; a
     `count` of 0 answers an empty array; taken `claim_1`…`claim_5` with
     `count` 2 → `claim_6`, `claim_7`; ids that are not minted ones under the
     prefix do not count: `claim_0`, `claim_01`, `claimx_9`,
     `other:claim_9` and a number of more than 18 digits leave the next
     id `claim_1`; a role prefix is matched literally, `.` included
     (`("a.b", ["axb_4"])` → `a.b_1`). It must answer exactly what `mint_id`
     answers called `count` times with each answer added to the taken set
     (checked against that loop), and `taken` may be a `pa.ChunkedArray`.
   - `mint_id(prefix, taken)` mints one by the same rule: `("clt", ())` →
     `clt_1`; `("clt", {"clt_1", "clt_3"})` → `clt_4`; `("clt",
     {"clt_2"})` → `clt_3`; with a role, `("score:clt", {"score:clt_1"})` →
     `score:clt_2`. Called twice with the same arguments it answers the same
     id, because it keeps no state: an element held by no scope mints from
     an empty taken set, so every free `Region` is `region_1`.
   - An id is free: any non-empty string without a control character.
     `validate_id` returns unchanged `"performance"`, `"a"`, `"Bass Kick"`,
     `"score:clt_7"`, `"a/b"` (a slash), `"intro to part 2"` (spaces), the
     URI `"https://example.org/op35#m1"`, and words that name the parts
     and facets of a drawn timeline (`"start"`, `"card"`, `"markers_1"`,
     `"regions_12"`), since an address is an object naming its id, facet
     and part apart, never a string composed of ids. It raises exactly `Invalid id '<id>': an id
     must be non-empty and contain no control character` (the id spelled by
     `repr`) for `""`, `"a\tb"`, `"a\nb"`, `"\x00"` and `"\x1f"`, the last
     control character, and the same message for a non-string (`3`).
   - `timeline_id_prefix` maps one unit of each kind to its prefix:
     quarters `clt`, ticks `dlt`, seconds `cpt`, samples `dpt`, centimeters
     `cgt`, pixels `dgt`.

**Validity Rationale:**

The ID system ensures:
- Events and timelines have unique, traceable identifiers
- IDs can be hierarchically scoped (e.g., `midi:track1:note_42`)
- External IDs from source files are preserved

---

### `test_timestamp.py` - Unified TimeStamp Architecture

**Purpose:** Validates the unified `TimeStamp` and `TimeIntervalStamp` classes that provide cross-section views through timeline hierarchies.

**Test Categories:**

1. **TimeStampBasics** (3 tests)
   - `get_timestamp()` at specific coordinates
   - Coordinate object input
   - Source reference preservation

2. **TimeStampWithChildren** (6 tests)
   - Child coordinate resolution via `get_coordinate_for("child:id")`
   - Boundary handling: a child holds `[offset, offset+length)` and its end where
     no sibling starts there; a lone child's end is on the child, a boundary two
     children share belongs to the one starting at it
   - Out-of-range axes raise `KeyError` (no extrapolation)
   - Multiple children with staggered offsets
   - `to_dict()` materialization
   - `present_timelines` property

3. **TimeStampWithCMaps** (7 tests)
   - Unit conversion via `get_unit(TimeUnit)`
   - Creating timestamp in alternate unit
   - Unit access through `get_unit()`
   - Missing C-Map raises `KeyError`
   - Missing C-Map raises ValueError
    - `to_dict()` emits typed dictionaries only for stored timeline coordinates;
     conversion-map values are retrieved verbatim with `get_conversion_for()`
   - C-Map conversion values assert exact `==`, not `pytest.approx`: the
     `TableMap` anchors (e.g. `[0,960]->[0,2]`, `[0,20]->[0,10]`) are exactly
     representable, so linear interpolation is bit-exact (480 ticks -> `1.0` s,
     1.5 s -> `720.0` ticks, child 10 -> `5.0` s).

4. **TimeIntervalStamp** (10 tests)
   - Interval creation from start/end coordinates
   - `get_interval()` for child timelines
   - Typed `duration` and `get_duration_for()` calculation
   - `get_intervals()` for all timelines
   - `get_interval()` access
   - `__str__` basic format (header + aligned start/end columns)
   - `__str__` straddling children (`-` for out-of-range endpoint)
   - `__str__` omits fully out-of-range children

5. **TimeStampValidation** (1 test)
   - Start/end must be from same source

6. **TimeStampSource Protocol** (4 tests)
   - Timeline implements required methods
   - `_get_related_timeline_ids()` returns direct child IDs
   - `_get_descendant_timeline_ids()` returns the full subtree
   - `_get_available_units()` aggregates C-Map targets across the subtree

7. **TimeStampReprHtml** (2 tests)
   - `_repr_html_` still renders the coordinate cross-section table
   - `_repr_html_` appends an affordance `Try` footer after the table
     surfacing the real accessors (`ts.get_coordinate_for(<tl_id>)` / `ts.get_unit(<unit>)`)

8. **TimeStampCrossSectionConversions** — every C-Map surfaces, at any depth
   - A conversion map with **no `target_unit`** (a label or structured-value
     map such as `IntervalToConstantMap`) surfaces in `__str__`, `to_dict`,
     and `get_conversion_for` — not only `TimeUnit`-targeted maps. This
     is the core-contract requirement: a timestamp is a cross-section, so it
     exposes ALL attached conversions, not just the numeric-unit subset.
   - A C-Map registered on a **direct child** surfaces on the parent's
     timestamp, evaluated at the child's own coordinate.
   - A C-Map registered on a **grandchild** (deeper descendant) likewise
     surfaces, evaluated at the descendant coordinate reached by exact,
     composed offset arithmetic. A stop-at-direct-children design would still
     hide these — the whole subtree is walked.
   - **Non-numeric outputs render intact**: a mapping/label value appears as
     itself (e.g. `{'page': 2}`, `'B'`), never coerced through `float`.
   - **Map name/selector retrieval**: `get_conversion_for("<cmap-name>")`
     returns the raw C-Map output; an unknown key raises `KeyError`.
   - **Collision qualification**: when two present timelines expose the same
     label, both are qualified as `"{owner_id}:{label}"`.
   - **`conversion_maps` gating** still applies — `conversion_maps=False`
     suppresses every surfaced conversion; a selector list surfaces only the
     matching maps.

   the C-Map visibility rule: timestamps surface the full
   `_conversion_maps` set across the subtree via `_conversion_rows()`.
   `add_conversion_map` still indexes `TimeUnit`-targeted maps in `_unit_maps`
   for `convert_to`/`get_conversion_map`.

**Validity Rationale:**

The unified TimeStamp architecture enables:
- Identical coordinate resolution for Timeline and TimelineGroup
- O(log n) lookup via InterpolationMaps (no table scans)
- Seamless unit conversion through attached C-Maps
- Cross-timeline coordinate projection in hierarchies

**Key API Demonstrated:**

```python
# Timeline unified API
ts = timeline.get_timestamp(30.0)
ts.axis                              # Typed IdCoordinate
ts.get_coordinate_for("child:1")    # Typed coordinate; absent axes raise KeyError
ts.to_dict()                         # Typed coordinate dictionaries
print(ts)                  # Full cross-section display

# Interval stamps
interval = timeline.get_interval_stamp(20.0, 60.0)
interval.duration                    # Typed Duration
interval.get_interval("child:1")    # Typed Interval; absent axes raise KeyError
print(interval)            # Two-column (start, end) display with '-' for out-of-range
```

**Design Notes:**

- **Child bounds checking**: `TimeStamp.get_coordinate_for(child_id)` raises
  `KeyError` when the child does not hold the queried coordinate: outside its
  `[offset, offset+length)` span, or at its end where a sibling starts. The end
  of a child nothing follows is held, so a section ending before its parent
  still answers its own end; a shared boundary is held once, by the sibling
  starting there, so one parent position never names two siblings.
- **TimeIntervalStamp.__str__**: Shows a `-` when one endpoint is out of range for a
  child, making it easy to see events that straddle children.

### `test_stamp_interface.py` - Stamp Family Contract

**Purpose:** Pins the shared `Stamp` contract across `TimeStamp` and `MatchStamp`.
The exact coordinate values distinguish the typed default from explicit
numeric projections and make attached `TimeUnit` values observable. The cases
pin `present_timelines`, singular and plural retrieval, conversion-map gating, every
`MatchStamp.to_dict()` format, and frozen `MatchStamp` fields. Exact dictionary
shapes ensure neither grouped rendering nor legacy graph serialization drifts.

**`MatchStamp` conversion-row display:** `MatchStamp.__str__`/`_repr_html_`
surface every C-Map enabled by the stamp's `conversion_maps` spec, the same
row shape `TimeStamp` uses (`_conversion_rows()` — inherited from the shared
`Stamp` base). The fixture timeline (`clock`, length 100 seconds) carries two
exact-value `TableMap`s: seconds→milliseconds (`[0,100]->[0,100000]`) and
seconds→frames (`[0,100]->[0,5000]`), queried at 25.0 seconds:

- `conversion_maps=True` — `str(stamp)` and `_repr_html_()` contain
  `"milliseconds"`/`"25000"` and `"frames"`/`"1250"` (25.0×1000 and 25.0×50);
  the HTML rows are tagged `<em>cmap</em>`.
- `conversion_maps=False` (the getter default, see below) — neither unit name
  appears in either rendering; the source coordinate (`"clock"`/`"25"`) still
  does.

**`conversion_maps` is opt-in on `MatchStamp`:** the field default and every
public matchstamp getter (`AlignmentBundle.get_matchstamp_at`,
`get_matchstamps`, `get_matchstamp_table`, `MatchClaim.get_matchstamp`) now
default `conversion_maps` to `False` — a caller must ask for conversions
explicitly, matching the opt-in getter pattern elsewhere in the module. Every
existing exact-value assertion in this file passes `conversion_maps=True`
(or `False`) explicitly through `_matchstamp_with_maps`, so none depend on
the getter default.

---

### `test_field_scalar_parity.py` - Scalar `to(*)` vs Field `convert_to(*)` Parity

**Purpose:** Verifies that every `@data_shaped` conversion implemented at the `SemanticField` level produces the same scalar values as the per-row scalar dispatch. The field-level `convert_to(*)` MUST be a `pa.compute` expression over the underlying `pa.Array`; iterating over materialised scalars to call `scalar.to()` is forbidden by the vectorized conversion contract.

**Test Categories:**

1. **EnharmonicPitch conversions** (3 tests)
   - `EP → MidiPitch` (metadata-only retype)
   - `EP → EnharmonicPitchClass` (vectorized `pc.mod(midi_number, 12)`)
   - `EP → EP` (identity)
2. **SpecificPitch conversions** (4 parametrized tests)
   - Full conversion matrix: `SP → {SpecificPitch, MidiPitch, EnharmonicPitchClass, SpecificPitchClass}`
3. **SpecificPitchClass conversions** (2 tests)
   - `SPC → EnharmonicPitchClass`
   - `SPC → SPC` (identity)
4. **GenericPitch conversions** (1 test)
   - `GP → GenericPitchClass`
5. **Sliced-field parity** (2 tests)
   - `SP[20:70] → EnharmonicPitchClass` on a zero-copy slice
   - `EP[10:60] → EnharmonicPitchClass` on a zero-copy slice

**Validity Rationale:**

Scalar/Field parity is structurally enforced for data-shaped methods: every scalar method tagged `@data_shaped` MUST have a vectorized mirror on the paired `SemanticField` subclass. These tests are the runtime guarantee that the mirror produces semantically identical output to the per-row scalar path. Samples include nulls and boundary values (e.g. MIDI 0/1/127, step C/B with alter ±2, octave -1 and 8) to confirm that null propagation and edge cases are handled identically on both paths.

The sliced-field tests confirm the corpus-scale slicing constraint: SemanticField operations MUST work efficiently on zero-copy `pa.Array` slices, never round-tripping through Python loops over materialised scalars.

This file also carries the `__init_subclass__` parity-check failure-path test (`TestParityCheckEnforcement`) that exercises `SemanticField.__init_subclass__`'s `@data_shaped` enforcement on a synthetic bad subclass.

---

### `test_scalar_repr.py` - Uniform scalar `repr()` / `str()`

**Purpose:** Pins the one uniform representation rule across every scalar:
`repr()` is the SHORT typed form `ABBR(token)`, and `str()` is the PRETTY
human token. The pitch scalars consolidate this onto the shared
`TwelveTETPitchMixin` (a class attribute `_REPR_ABBR` + a single
`__str__` / `__repr__`); two pitch scalars override `__repr__` for a
dual-spelling or numeric form. The MIDI-event scalars share one rendered
`__repr__` driven by a `_repr_parts()` hook (no string surgery between the
base and the subclass).

**Exact expected strings (zero-tolerance; the canonical ♯/♭ characters):**

Pitch scalars — `repr()` then `str()`:

| Scalar | construction | `repr()` | `str()` |
|--------|--------------|----------|---------|
| `EnharmonicPitchClass` | `(0)` | `EPC(C)` | `C` |
| `EnharmonicPitchClass` | `(1)` | `EPC(C♯/D♭)` | `C♯/D♭` |
| `GenericPitchClass` | `(0)` | `GPC(C)` | `C` |
| `GenericPitch` | `(0, 4)` | `GP(C4)` | `C4` |
| `SpecificPitchClass` | `(step="C", alter=1)` | `SPC(C♯)` | `C♯` |
| `EnharmonicPitch` | `(56)` (black) | `EP(G♯/A♭3)` | `G♯/A♭3` |
| `EnharmonicPitch` | `(60)` (white) | `EP(C4)` | `C4` |
| `MidiPitch` | `(60)` | `MP(60)` | `60` |
| `SpecificPitch` | `(step="C", alter=1, octave=4)` | `SP(C♯4)` | `C♯4` |

Note: `EnharmonicPitchClass` is the only pitch scalar whose `get()` returns
the bare pitch-class integer (untouched — its Field-vectorized mirror depends
on it). To honour the str=pretty rule, `EnharmonicPitchClass.__str__` is an
explicit override returning the dual `_PC_TO_LABEL` label (mirroring the
repr's inner token); `repr()` likewise uses an explicit `_PC_TO_LABEL`
override.

Shallow scalars:

| Scalar | `repr()` | `str()` |
|--------|----------|---------|
| `MeasureNumber(value=16)` | `MeasureNumber(16)` | `16` |
| `Id(value="n0")` | `Id('n0')` | `n0` |

MIDI-event scalars (only non-None fields appear, in declaration order;
`_repr_parts()` is the single source — no `super().__repr__()` slicing):

| Scalar | `repr()` |
|--------|----------|
| `MidiEvent(pitch=EnharmonicPitch(60), velocity=80, channel=0)` | `MidiEvent(pitch=EP(C4), velocity=80, channel=0)` |
| `MidiEvent(control=64, value=127, channel=0)` | `MidiEvent(channel=0, control=64, value=127)` |
| `ScoreMidiEvent(pitch=EnharmonicPitch(60), velocity=80, voice=1, staff=2, part_id="P1")` | `ScoreMidiEvent(pitch=EP(C4), velocity=80, voice=1, staff=2, part_id='P1')` |

Harmony scalars — `str()` returns the bare `label`:

| Scalar | `str()` |
|--------|---------|
| `HarmonyLabel(label="V7", …)` | `V7` |
| `PitchBasedHarmony(label="Cmaj", …)` | `Cmaj` |
| `WesternTertianHarmony(label="Cmaj7", …)` | `Cmaj7` |
| `RomanNumeralHarmony(label="V7", …)` | `V7` |
| `DcmlHarmony(label="V(64)", …)` | `V(64)` |

Note / Measure — `str()` is a pretty one-liner; `repr()` unchanged:

| Scalar | `str()` |
|--------|---------|
| `Note(start=0q, duration=1q, pitch=EnharmonicPitch(61))` | `C♯/D♭4 @0 quarters+1 quarters` |
| `Note(start=0q, duration=1q, pitch=None)` (rest) | `rest @0 quarters+1 quarters` |
| `Measure(mn="16", …)` | `16` |

**Validity Rationale:**

The TimeStamp-uniformity philosophy extends to scalar
representation: a user must be able to trust that any scalar's `repr()` is a
short typed form and `str()` is the readable token. Consolidating the rule
onto `TwelveTETPitchMixin` means seven pitch
scalars share one implementation; the `_repr_parts()` hook makes the
`ScoreMidiEvent` extension a first-class override rather than fragile string
surgery over the base repr. The exact-string assertions guard against silent
drift, regression of the EnharmonicPitch dual-spelling repr, and accidental
re-introduction of the verbose `Object(field=…)` forms.

### `test_fields_base.py` - Raw fields and SemanticField delegation

A `SemanticField` composes its raw field and delegates every other attribute
to it through `__getattr__`. Delegation must not reach an instance whose
`__init__` has not run: `copy` and `pickle` build one with `__new__` and ask
it for `__setstate__`, and delegating that question asks for the missing
`_raw` again, without end (`RecursionError`). What proves the guard:

- On an instance made with `__new__` alone, `_raw` and a dunder name raise
  `AttributeError` naming the attribute, the answer `copy` expects.
- `copy.deepcopy` of a field gives a field of its class, equal row for row
  (`to_pylist()` and `field_names` the original's) and holding its own raw
  field (`_raw is not` the original's); `pickle` round-trips it the same way.
- `copy.deepcopy` of an `AlignmentBundle`, which holds its MatchClaim table
  as a `MatchClaimField`, gives a bundle equal to the original that is not
  the original, its MatchClaim table a `MatchClaimField` of zero rows that is
  not the original's. (`tests/alignment/test_claim_back_reference.py` copies
  a bundle holding 303 claims and checks that they follow the copy.)

### `test_field_repr_html.py` - Field affordance `_repr_html_`

Validates the rich-HTML affordance card every `DataField` / `SemanticField`
renders through the shared `affordance_html` helper.

**What we validate (exact presence of specific rows/snippets):**

- A live `CoordinateField` card titled `<h4>CoordinateField</h4>` shows:
  - a `Scalar type` row whose value is `Coordinate`;
  - a `Length` row matching the element count;
  - an `Arrow type` row (escaped `struct<…>`);
  - a `Sample` row containing the new scalar reprs (e.g. `Coordinate(0.0,`)
    — proving the sample uses `repr(field[i])`, the scalar reprs.
- The `Try` row lists exactly the three affordance snippets
  `field[i] -> <Scalar>`, `field.convert_to(<TargetScalar>)`,
  `field.get_raw()` (each as a `<code>` span, `<`/`>` escaped).
- A raw `StructField` card (the scalar-less base path) shows `Length`,
  `Arrow type`, `Sample` and the base `field[i]` / `field.get_raw()`
  affordances — but NO `Scalar type` row.
- A schema-only (blueprint) field's `Sample` value is `(schema-only)` and
  its `Length` is `0` — the card never raises.
- Head/tail truncation: a 6-element field's `Sample` shows the first 3 and
  last 2 element reprs separated by the `…` ellipsis.

### `test_eventdata_repr_html.py` - EventData affordance `_repr_html_`

**What we validate:**

- A `MidiEventData` card titled `<h4>MidiEventData</h4>` shows `Events`,
  `Unit`, `Number type`, and a `Fields` row listing each metadata-bearing
  semantic field as `name : ScalarType` inside an `<ul>` (the materialised
  `pitch` column reports `pitch : EnharmonicPitch`).
- The `Try` row lists `get_field(<Scalar>)`, `get_pitch_field()`,
  `get_raw('<col>')`.
- A plain `EventData` whose columns carry no `field_type` metadata renders a
  `Fields` row of `(none)` and does NOT raise (graceful skip per the
  "must not raise if a column has no paired field" contract).

---

### `test_parquet_metadata.py` - Versioned metadata blob, single ownership

**Purpose:** Pins the contract of the Arrow-metadata vocabulary that travels
with every TimeToAlign!-written `pa.Field` and table schema.
`core/fields.py` is its sole owner: it defines the metadata key, the blob
version, the two writers (`metadata_blob_from_dict`,
`metadata_blob_for_model`) and the one parser (`parse_metadata_blob`).

**What we validate:**

1. **Version injection.** Both writers stamp
   `"version": TIMETOALIGN_BLOB_VERSION` into every payload they encode —
   the dict writer even when the caller supplied no `version`, and the model
   writer alongside the model's JSONSchema. `parse_metadata_blob` returns the
   `version` entry as part of the payload, so a parsed dict can be re-encoded
   unchanged.
2. **Version-less rejection.** A hand-rolled blob with no `version` key, a
   non-integer `version`, or a `version` greater than
   `TIMETOALIGN_BLOB_VERSION` raises `ValueError`. There are no
   compatibility shims: every blob in circulation comes from one of the two
   writers. Absent (`None`) and empty blobs still parse to `{}`.
3. **Single ownership.** The byte string `b"timetoalign"` is spelled out in
   `timetoalign/core/fields.py` and nowhere else in the package — every other
   module imports `TIMETOALIGN_METADATA_KEY`. A test walks the package source
   to enforce this, in either quote style and including prose: an error
   message that spells the key out is a second source of truth too, so those
   name the constant instead.
4. **Hierarchy-wide stamping.** `SemanticField.to_field()` stamps the blob for
   every field in the hierarchy exactly once, deriving `field_type` from the
   class rather than from hand-written literals, and carries over metadata
   entries owned by anyone else untouched.
5. **Canonical-struct round-trip.** `rational_to_struct` /
   `struct_to_rational` are the row-wise pair for
   `RATIONAL_STRUCT_TYPE` (`{value, numerator, denominator}`), the library's
   only rational shape. Round-trips return the **exact** `Fraction`
   (`Fraction(3, 4)`, not `0.75`) because the parser reads the integer
   components and ignores the lossy float projection. Integer-valued float
   components from artifacts written by earlier releases are accepted and
   losslessly normalized to integers; fractional float components raise
   `ValueError` rather than being rounded.

**Validity Rationale:** A single owner plus a required version means a stored
table can always be interpreted, and a future payload change is a bump rather
than a guessing game. One rational struct means a reader never has to sniff
between two key spellings to recover an exact ratio.

---

### `test_number_json.py` - JSON numbers and coordinates

**Purpose:** Pins the TimeToShare JSON forms and their boundary with Arrow
storage cells. `number_to_json` and `number_from_json` round-trip each native
number exactly: an `int` is a JSON integer, a `float` is a JSON number, and a
`Fraction` is `[numerator, denominator]` with a strictly positive denominator.
The tests reject booleans, zero and negative denominators, and pairs of the
wrong length. `is_fraction_json` accepts only valid two-item pairs of
integral members.

Coordinates on another timeline's axis are `{ "unit", "number_type", "value"
}`; their number type selects the value representation, and a decoded
coordinate restores its declared Python type. A timeline's length and its
children's offsets lie on its own axis and are written as the bare value: the
tests prove that a written timeline is a JSON fixpoint whatever its number
type, that a length and an offset come back exactly in the axis's type, and
that a bare value represented otherwise than the axis selects is refused with
a message naming what to write. Arrow storage remains a separate, three-member
column representation and is not serialized as JSON.

Every conversion-map class that has a manifest form — all but the two
periodic maps, whose refusal `tests/maps/README.md` pins — contributes one
live instance (an `IntervalToConstantMap` of labels given its `end`, the
four measure views over one two-bar measure map): its `to_dict()` is
JSON-safe and keeps exact Fractions, its name is written only when given
(`null` in the complete form, left out of the minimal form it writes by
default), and a custom name round-trips through `ConversionMap.from_dict`, a view
given the measure map it names.

### Coordinate cell rendering

`Coordinate.to_dict()` and `Duration.to_dict()` emit Arrow storage cells, so
their three stored members remain available for table and Parquet use. Public
JSON payloads use the number and coordinate forms above instead. Tests compare
the complete dictionaries and verify that `Note.to_dict()` and `Measure.to_dict()`
embed storage cells unchanged.

---

### `test_provenance.py` - Paradata and beliefs, one entry

`Provenance` (`core/provenance.py`) is the one entry stating how an item was
made and who believes it; `Belief` is one actor's `true`, `false` or
`unlikely` at a date. Each is a frozen scalar followed by its paired Field.
What proves the entry is the format's and holds nothing else:

- **Keys.** Exactly `stated_by, date, method, service, basis, match, note,
  beliefs` (the derived struct's names, asserted as a literal list), a belief
  exactly `actor, belief, date, note`. An unknown key is refused, and so is an
  entry stating nothing (`A provenance entry states at least one key`).
- **Values.** A date written other than `YYYY-MM-DD` (`2026-1-5`) or not of
  the calendar (`2026-02-30`) is refused with its own message; `stated_by` and
  `basis` list at least one IRI, each once; `match` takes the five SKOS kinds
  only; `belief` the three values only. The vocabularies are TimeToShare's
  (`timetoshare.statements.METHODS`, `MATCHES`), never restated: `method` is
  one of the four method words (`manual`, `reconciled`, `imported`,
  `inferred`) or an IRI (a scheme and a colon, `https://example.org/m`); a
  phrase (`by hand`) is refused, and so is an empty `service`, as
  TimeToShare's table check refuses them.
- **One belief per actor.** An entry holding two beliefs of `ana` is refused;
  `with_belief` replaces `ana`'s belief in its place (her `unlikely` of
  2026-11-20 becomes `false` of 2026-11-21, `jo`'s `true` after it stays) and
  appends a new actor's, leaving the original unchanged (copy-on-write);
  `get_belief` answers the actor's belief or `None`.
- **One actor however spelled.** Actors compare as the IRIs they stand for
  (`actor_iri`, TimeToShare's `expand_iri`, the text as written where it
  expands to nothing), the comparison TimeToShare's `belief_actor` check
  makes. `actor_iri("ex:ana", _CONTEXT)` is
  `https://example.org/people/ana`; without a context `ex:ana` stays as
  written. On an entry where `https://example.org/people/ana` believes
  `true` and `jo` `unlikely`, `with_belief` of `ex:ana` `false` under the
  context replaces the absolute spelling's belief in its place (two beliefs:
  `ex:ana` `false`, then `jo`'s), and `get_belief("ex:ana", context=...)`
  answers it; without the context no prefix is declared, so `ex:ana` is
  another actor and is appended. Construction, which has no context,
  compares the same way: two beliefs of `https://example.org/people/ana`
  are refused.
- **Two spellings of one believer on one row.** Construction has no
  context, so an entry may hold `ex:ana` and
  `https://example.org/people/ana`, which the context makes one believer.
  The filter then reads the row as the most critical of the two beliefs
  (false, unlikely, true: the order in which a drawing colours an anchor
  stated by several MatchClaims), whatever the order they are written in,
  so the answer never depends on which came last. Two rows, `ex:ana`
  `true` + absolute `false` and absolute `false` + `ex:ana` `true`, read
  as `false` under the context for `ex:ana`: the default mask is `[False,
  False]`, `["false"]` `[True, True]`, `["true"]` `[False, False]`. Two
  more rows, `true` + `unlikely` in both orders, read as `unlikely`:
  `["unlikely"]` `[True, True]`, `["true"]` `[False, False]`. Without the
  context the two spellings are two actors, so `ex:ana` reads its own
  belief: `["true"]` `[True, True]` on the first pair.
- **One check of the states, TimeToShare's.** The four states, the default
  and the check are TimeToShare's (`timetoshare.BELIEF_STATES`,
  `DEFAULT_BELIEFS`, `check_belief_filter`): the library holds no copy, so
  `provenance.check_belief_filter is timetoshare.check_belief_filter` and
  `provenance.DEFAULT_BELIEFS is timetoshare.DEFAULT_BELIEFS`, and the
  states are `("true", "undecided", "unlikely", "false")`. The check's own
  cases are TimeToShare's tests; here `belief_mask` refuses through it
  with TimeToShare's messages: a single string, which would read as its
  characters (`"false"`: `the belief filter takes a collection of states,
  not the string 'false': write ['false']`), and a word that is no state
  (`["not_false"]`: `'not_false' is no belief state: write 'true',
  'undecided', 'unlikely' or 'false'`); so do `ClaimFilter` and `believed`
  (`test_claim_beliefs.py`).
- **Dictionary.** `to_dict` leaves absent keys out (an entry of `service`
  alone is `{"service": ...}`; a belief without a note has no `note` key),
  writes lists for `stated_by`, `basis` and `beliefs`, and `from_dict` gives
  back an equal entry; asserted on one whole literal.
- **The belief filter** is decided in one place, `belief_mask`, over a column
  of entries; `Provenance.passes_belief_filter` and
  `ProvenanceField.passes_belief_filter` ask it. Each row has one state for
  the believer — the believer's belief, or `undecided` where the believer
  holds none (no entry, an entry without beliefs, or only other actors'
  beliefs) — and is kept when that state is among `beliefs`. The column of
  five rows — no entry, an entry without beliefs, `jo` `false`, `jo` `true`,
  `jo` `unlikely` with `ana` `false` — has for `jo` the states `undecided,
  undecided, false, true, unlikely` and for `ana` `undecided` ×4, `false`.
  So for `jo`: the default (all but `false`) `[True, True, False, True,
  True]`; `["true"]` `[False, False, False, True, False]`; `["undecided"]`
  `[True, True, False, False, False]`; `["unlikely"]` `[False, False, False,
  False, True]`; `["false"]` `[False, False, True, False, False]`; all four
  states all `True`; `[]` all `False`. For `ana`: the default `[True, True,
  True, True, False]`, `["false"]` `[False, False, False, False, True]`. The
  four one-state masks of one believer partition the rows (each row in
  exactly one), which is what proves `undecided` is the complement of the
  three beliefs rather than a fourth value read from the data. For no
  believer every row is `True` whatever the states. An empty column gives
  an empty mask. The scalar answers equal the column's row for row, and the
  Field's equal the function's, so neither can drift from the one
  predicate; a tuple, a list and a set of the same states give one mask.
- **Actors compare as IRIs, in the predicate itself.** `belief_mask` expands
  the believer and every actor under its `context=` (TimeToShare's
  `expand_iri`), so statements and MatchClaims share one comparison. A column
  of four entries — `ex:ana` `false`, `https://example.org/people/ana`
  `false`, `https://example.org/people/ana` `true`, `wd:Q1` `false` — under
  the context declaring `ex` as `https://example.org/people/` gives for the
  believer `ex:ana` and for `https://example.org/people/ana` alike
  `[False, False, True, True]` (the default states), and for the absolute
  believer with `["true"]` `[False, False, True, False]`: the compact and the
  absolute spelling are one actor either way round. Without a context no
  prefix is declared, so the compact actor compares as written: `ex:ana`
  gives `[False, True, True, True]`, the absolute believer
  `[True, False, True, True]`. A prefix the context does not declare
  (`wd:Q1`) also compares as written, `[True, True, True, False]`.
- **Null beliefs.** A row of TimeToShare's provenance column states an entry
  without beliefs as a null `beliefs` cell (its fixture table does). Such a
  row reads as an entry whose `beliefs` is `()`, equal to one built without
  beliefs, and `provenance_array` writes an entry without beliefs back with a
  null `beliefs` cell, so a row read and written keeps its cell.
- **Field.** `ProvenanceField.from_provenances` of entries and `None` reads
  each row back equal (`None` where none), its type the scalar's derived
  struct.

## Running Tests

```bash
# Run all core tests
cd timetoalign
python -m pytest tests/core/ -v

# Run specific test file
python -m pytest tests/core/test_timestamp.py -v

# Run with coverage
python -m pytest tests/core/ --cov=timetoalign.core --cov-report=term-missing
```

---

## Validation Methodology

All tests follow the TimeToAlign! engineering standards:

1. **Exact Values**: Assertions use exact expected values, not ranges or tolerances
2. **Immutability**: Frozen pydantic models are tested for mutation rejection;
   assignment raises `pydantic.ValidationError`, pinned as the concrete class
   (never a bare `Exception`)
3. **Type Safety**: Invalid types are verified to raise the concrete error
   class (`TypeError` for bad `Coordinate` value/unit types, including bool)
4. **Roundtrip**: Serialization/parsing operations preserve data exactly
