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
index remains invalid. Address storage contains no skeleton identifier.
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
refused. The tests name one case for each rule.

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
`timeline.get_timestamp(9.5).get_coordinate_for("clt1", format="coordinate")`
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
timeline.get_timestamp(9.5).get_coordinate_for("clt1", format="coordinate")
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
timeline, so the axis decides. An anchor reaches its timelines only by id, so
what it can consult is the unit's own default (R3); a bundle holds the timeline
objects and refines that to each timeline's declared representation as the
claim is read in its context. Both directions are asserted, for the reason
given above:

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

**A NOMATCH claim keeps its position outside an anchor, and it counts.** The
anchored and interval lanes were brought onto their axes first and this one was
missed, because its coordinate lives in a different field (`source_coordinate`)
— which is exactly how one defect reappears in an area already believed fixed.
A section marker at `164.3` seconds rendered as
`@2890396167097549/17592186044416 seconds`. It gets the same rule for the same
reason, applied where the value is stored so that storage, `repr`, `__str__`
and `_repr_html_` cannot disagree. The test asserts the model field and all
three renderings, because an anchorless claim has no getter to go through:
`get_coordinate_for` raises on it, correctly, and is left alone.

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
| `dgt1` pixels (`int`) | `0`, `12473` | `0`, `12473` |
| `cpt1` seconds (`float`) | `0.0`, `37.5` | `0.0`, `37.5` |
| `clt1` quarters (`fraction`) | `Fraction(0, 1)`, `Fraction(19, 2)` | same |
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
same timeline reported the ratio. The claim table is covered over both claim
stores, because the columnar store reaches the table through a bulk
four-column read that the per-claim list does not use.

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

### `test_score_scalars.py` - Circle 1 Score Scalars

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
  struct also carries optional rendition, skeleton, and section qualifiers;
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

### `test_ids.py` - Scoped ID System

**Purpose:** Validates the `ScopedId` and `IdGenerator` classes for unique identification.

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

5. **IdGenerator Creation** (2 tests)
   - Creation with scope
   - Empty scope allowed

6. **IdGenerator get_or_create** (9 tests)
   - Wrapping external IDs
   - Auto-generation with type hints
   - Counter incrementation
   - Independent counters per type hint

7. **IdGenerator State** (5 tests)
   - `count` property
   - `has_seen()` lookup
   - `reset()` and `reset_counters()` methods

8. **IdGenerator Empty Scope** (2 tests)
   - Unscoped ID generation

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
   - Boundary handling: left-inclusive, right-exclusive `[offset, offset+length)`
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
  `KeyError` when the queried coordinate falls outside the child's
  `[offset, offset+length)` span, per the TTA
  left-inclusive, right-exclusive interval convention.
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

**Purpose:** Pins the TimeSkeleton JSON forms and their boundary with Arrow
storage cells. `number_to_json` and `number_from_json` round-trip each native
number exactly: an `int` is a JSON integer, a `float` is a JSON number, and a
`Fraction` is `[numerator, denominator]` with a strictly positive denominator.
The tests reject booleans, zero and negative denominators, and pairs of the
wrong length. `is_fraction_json` accepts only valid two-item integer pairs.

Coordinates are `{ "unit", "number_type", "value" }`; their number type
selects the value representation. The tests prove that a decoded coordinate
restores its declared Python type and that timeline lengths and child offsets
reject unit or number-type mismatches. Arrow storage remains a separate,
three-member column representation and is not serialized as JSON.

### Coordinate cell rendering

`Coordinate.to_dict()` and `Duration.to_dict()` emit Arrow storage cells, so
their three stored members remain available for table and Parquet use. Public
JSON payloads use the number and coordinate forms above instead. Tests compare
the complete dictionaries and verify that `Note.to_dict()` and `Measure.to_dict()`
embed storage cells unchanged.

---

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
