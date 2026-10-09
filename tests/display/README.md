# Display Module Tests

This directory contains tests for the `timetoalign.display` module, which provides
ASCII/Unicode terminal visualization and rich-HTML affordance cards for
TimeToAlign! objects.

## Test File: `test_html.py`

### Overview

Validates `timetoalign.display.html`, the single shared helper behind every
object's `_repr_html_` affordance card. Loaders, EventData, and Fields all
render through `affordance_html`, so this is the one canonical place the markup
shape is pinned.

### Validation Logic — `affordance_html` / `code` (deterministic, exact-string)

`affordance_html(title, rows, *, affordances=None)` and `code(s)` are pure
functions. Tests assert the EXACT full output string.

**`code(s)`** wraps text in an HTML-escaped `<code>` span:
- `code("y")` == `"<code>y</code>"`
- `code("y<z")` == `"<code>y&lt;z</code>"` (the value is escaped)

**`affordance_html`** renders `<h4>{escaped title}</h4>` + `<table>` with one
`<tr><td><b>{escaped label}</b></td><td>{verbatim value}</td></tr>` per row,
then — when `affordances` is non-empty — a final
`<tr><td><b>Try</b></td><td>{code-spans joined by ", "}</td></tr>`, then
`</table>`. Lines are joined by `"\n"`.

Pinned expectations:
- Title is HTML-escaped: a `"<X>"` title renders `<h4>&lt;X&gt;</h4>`.
- Row LABEL is escaped by the helper; row VALUE is passed through verbatim
  (so a caller-supplied `<code>` survives intact, an unescaped `<` in the
  value survives intact).
- `affordances=None` or `[]` emits NO `Try` row.
- A non-empty `affordances` list emits exactly one `Try` row whose value is the
  snippets each wrapped via `code()` and joined by `", "`.

The exact-string fixture is:

```
affordance_html("Demo", [("A", "x"), ("B", code("y<z"))], affordances=["f()", "g(1)"])
```

→

```
<h4>Demo</h4>
<table>
<tr><td><b>A</b></td><td>x</td></tr>
<tr><td><b>B</b></td><td><code>y&lt;z</code></td></tr>
<tr><td><b>Try</b></td><td><code>f()</code>, <code>g(1)</code></td></tr>
</table>
```

### Validation Logic — `affordance_line` (deterministic, exact-string)

`affordance_line(snippets)` is the standalone footer counterpart to
`affordance_html`'s `Try` row, for objects that own their own `<table>` (the
timestamp / claim coordinate cross-sections). It renders a free-standing
`<div>` listing the snippets as `<code>` spans under a `Try:` label, and shares
the snippet-join with `affordance_html` via the private `_join_snippets` helper
(no duplicated join logic).

Pinned expectations:
- A non-empty list renders
  `<div style='margin-top: 4px; color: #666; font-size: 0.85em;'>Try: {code-spans joined by ", "}</div>`.
- Each snippet is escaped via `code()` (a `<` in a snippet survives as `&lt;`).
- An empty list or `None` renders `""` (no element at all).

Exact-string fixtures:

```
affordance_line(["a()", "b(<x>)"])
```

→

```
<div style='margin-top: 4px; color: #666; font-size: 0.85em;'>Try: <code>a()</code>, <code>b(&lt;x&gt;)</code></div>
```

```
affordance_line([])  →  ""
affordance_line(None) →  ""
```

## Test File: `test_ascii.py`

### Overview

The test suite validates the pure-ASCII display functionality with **80 tests** covering:
- Character set definitions and completeness (timeline, region, flow)
- Helper function correctness (coordinate formatting, name elision, region rows)
- Timeline, group, and bundle diagram rendering
- Region display in timeline diagrams (`show` parameter)
- Flow control diagrams for ScoreFlowController
- Flow playthrough diagrams for Flow objects
- Flow comparison diagrams (side-by-side diffs)
- Unicode and ASCII fallback modes

### Validation Logic

#### 1. Character Set Tests (`TestCharacterSets`)

**What we validate:**
- All 6 core timeline type combinations have assigned characters
- Each character is exactly 1 character (for consistent column width)
- Core timeline characters are unique (no ambiguity in visual representation)
- Tree and box drawing character sets are complete

**Why this matters:**
The display module relies on consistent character mappings. If a timeline type lacks
a character, users would see `?` placeholders. The uniqueness test ensures users can
visually distinguish timeline types at a glance.

**Character mapping (NumberType, Domain) → Character:**
| NumberType | Domain    | Unicode | ASCII |
|------------|-----------|---------|-------|
| int        | graphical | `:`     | `:`   |
| float      | graphical | `=`     | `=`   |
| int        | physical  | `⋅`     | `.`   |
| float      | physical  | `~`     | `~`   |
| int        | logical   | `,`     | `,`   |
| float      | logical   | `_`     | `_`   |
| fraction   | (any)     | (same as float) | |

#### 2. Coordinate Formatting Tests (`TestFormatCoordinate`)

**What we validate:**
- Integer values display without decimal points (e.g., `100` not `100.0`)
- Float values display with 1 decimal place (e.g., `150.5`)
- Values that are mathematically integers format as integers

**Why this matters:**
Clean coordinate display improves readability. `0 :::: 4835 pixels` is clearer than
`0.0 :::: 4835.0 pixels`.

#### 3. Name Elision Tests (`TestElideName`)

**What we validate:**
- Names within the max width pass through unchanged
- Names exceeding max width get `...` suffix
- Edge cases (very short max_width) are handled gracefully

**Why this matters:**
Child timeline names must fit in fixed-width columns. Elision ensures long names
like `very_long_system_name_123` become `very_lon...` without breaking layout.

#### 4. Timeline Character Tests (`TestGetTimelineChar`)

**What we validate:**
- Each of the 6 concrete timeline types returns the correct character
- Unicode and ASCII modes return appropriate characters

**Why this matters:**
This is the core mapping that makes the display work. Each test creates an actual
timeline instance and verifies `_get_timeline_char()` returns the expected character.

#### 5. Child Truncation Tests (`TestGetChildrenToDisplay`)

**What we validate:**
- No truncation when children ≤ max_children
- Correct split into first/last groups when truncation is needed
- Accurate omitted count calculation

**Algorithm tested:**
```
Given: 10 children, max_children=6
Split: first_count = (6+1)//2 = 3, last_count = 6-3 = 3
Result: first 3 children, "... (4 more children)", last 3 children
```

**Why this matters:**
Without truncation, a timeline with 50 children would produce unreadable output.
The truncation shows first and last children with a clear count of omitted items.

#### 6. Child Row Building Tests (`TestBuildChildRow`)

**What we validate:**
- Row contains tree prefix (├─ or └─), name, coordinates, bar, and end coordinate
- Last child uses └─, others use ├─
- Bar characters are positioned proportionally on the parent scale

**Why this matters:**
The child row is the core visual element showing how children map to parent coordinates.
A child at offset 250 with length 500 on a parent of length 1000 should have its bar
positioned at 25% of the width, spanning 50% of the total.

#### 7. Timeline Diagram Tests (`TestTimelineDiagram`)

**What we validate:**
- Basic output includes class name, ID, and bar
- Children appear as properly formatted rows
- `show_children=False` hides child rows (but mentions count in header)
- ASCII mode produces output without Unicode characters
- `parent_id` annotation appears in header

**Example validated output:**
```
DiscreteGraphicalTimeline[page] (3 children)
0 :::::::::::::::::::::::::::::::::::: 1000 pixels
  ├─ System 1     0 :::::         300
  ├─ System 2   350       :::::   650
  └─ System 3   700            :::950
```

#### 7a. Recursive Child Diagram Tests (`TestTimelineDiagramRecursion`)

**What we validate:**
- `depth=True` renders descendants without a level limit
- `depth=False` and `depth=1` render direct children only with exactly the
  existing one-level output
- `depth=0` renders no child rows, while negative integer depths raise
  `ValueError`
- Descendant bars and entry/exit labels use absolute coordinates on the root
  timeline
- Ancestor continuation glyphs remain visible for descendants whose ancestor
  has following siblings
- `max_children` truncation applies independently at every rendered level and
  uses the existing `... (N more children)` row
- A `SegmentLine` child expands like any other child — `depth` alone decides
  how far the rendering descends, so `depth=True` reaches the segments of a
  segment line and `depth=1` stops at the segment line's own row
- Every rendered level puts its entry coordinate and bar in the **root's**
  columns: a descendant's tree glyphs come out of the name column, never out
  of the alignment (asserted structurally, not only as a golden string)
- The name column widens by exactly the tree glyphs the deepest rendered
  branch needs, so a one-level diagram is byte-identical to the default
- `group_diagram` forwards `depth` to every member timeline

**Depth rules tested:**
```
True  → all descendant levels
False → direct children only
0     → no child rows
1     → direct children only
2     → children and grandchildren
```

**Why this matters:**
Nested timelines must preserve their hierarchy while sharing one root coordinate
scale. Limiting depth keeps large structures readable and per-level truncation
bounds every branch — but the limit must be `depth` and nothing else. A
timeline type may not silently hide its own children: a nested segment line
(an IEEE 1599 edition is pages of accolades) would otherwise be undisplayable
at any depth, so `depth` governs uniformly and callers who want one row ask
for `depth=1`.

#### 8. Group Diagram Tests (`TestGroupDiagram`)

**What we validate:**
- Header shows group ID, timeline count, and timestamp count
- Box border is drawn with proper Unicode/ASCII characters
- All member timelines appear inside the box
- Footer shows timestamp count

#### 9. Bundle Diagram Tests (`TestBundleDiagram`)

**What we validate:**
- Header shows bundle ID
- Groups are rendered with indentation
- Match claims footer is present

#### 10. Integration Tests (`TestDiagramMethods`)

**What we validate:**
- `Timeline.diagram()` returns identical output to `timeline_diagram(tl)`
- `TimelineGroup.diagram()` returns identical output to `group_diagram(group)`
- `AlignmentBundle.diagram()` returns identical output to `bundle_diagram(bundle)`
- Parameters (width, unicode, etc.) are passed through correctly
- `depth` is forwarded by timeline, group, and bundle methods so callers can
  include or suppress grandchildren consistently

**Why this matters:**
Users interact with the `.diagram()` method, not the module functions. These tests
ensure the convenience methods are correctly connected to the underlying implementation.

## Flow Visualization Tests -- IMPLEMENTED

All flow visualization tests are implemented (42 tests).

### Step 1: Regions in `timeline_diagram()`

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestRegionCharSets` | 2 | Unicode and ASCII region char sets are complete |
| `TestBuildRegionRow` | 3 | Region row structure, proportional positioning, name elision |
| `TestTimelineDiagramWithRegions` | 5 | `show={"regions"}` shows regions; both regions+children; sorting; backwards compat; ASCII mode |
| `TestTimelineDiagramHeaderRegions` | 1 | Region count appears in timeline header |
| `TestDiagramMethodShowParam` | 2 | `Timeline.diagram(show=...)` passes through correctly |

**What we validate:**
- `show={"regions"}` renders region rows with `┄` prefix and `═` fill (distinct from child rows)
- `show={"regions", "children"}` renders both children (with `├─`/`└─`) and regions
- Regions are sorted by start coordinate regardless of insertion order
- `show=None` (default) preserves exact pre-existing behaviour (no regions displayed)
- `show_children=False` takes precedence over `show={"children"}`
- ASCII mode uses `~`, `[`, `]`, `=` for regions

### Step 2: `flow_control_diagram()` for ScoreFlowController

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestFlowCharSets` | 2 | Unicode and ASCII flow char sets complete |
| `TestFlowControlDiagram` | 11 | Header; MC ruler; section spans; repeat markers; volta brackets; legend; graph; show/hide toggle; ASCII mode; minimal controller |
| `TestFlowControllerDiagramMethod` | 2 | `.diagram()` delegates; `__str__` returns diagram |

**What we validate:**
- Header shows MC count, section count, flow event count
- MC ruler row lists all MC numbers
- Section span row shows section IDs with `├──ID──┤` delimiters
- Repeat barlines (`║:` and `:║`) aligned to correct MC columns
- Volta brackets (`┌1─`, `┌2─`) with closing `┐` between consecutive voltas
- Legend enumerates all flow control events per MC
- Section transition graph shows `ID → [targets]` per section
- `show_legend=False` and `show_graph=False` hide respective sections
- Minimal controllers with no flow events render cleanly

### Step 3: `flow_diagram()` for Flow

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestFlowDiagram` | 6 | Header; section rows with MC ranges; atomic sequence footer; reason annotations; show_reasons=False; show_mcs=True |
| `TestFlowDiagramMethod` | 3 | `Flow.diagram()` delegates; `__str__` returns diagram; `__repr__` unchanged |

**What we validate:**
- Header shows mode, folded/unfolded counts, ratio, section count
- Section table with `#`, `MCs` (right-open intervals), `Sections`, `Reason` columns
- Reason derivation: `start` for first; `→` for continuation; `repeat →` / `D.S.` / `D.C.` for jumps
- `show_mcs=True` expands MC sequences per section
- Footer shows complete atomic section sequence
- `__repr__` stays compact one-liner; `__str__` returns diagram

### Step 4: `flow_comparison_diagram()` for Flow Diffs

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestFlowComparisonDiagram` | 4 | Identical flows (all `=`); divergent flows (`≠` with explanation); summary footer; ASCII mode |
| `TestDiffDiagramMethod` | 1 | `flow.diff_diagram(other)` delegates correctly |

**What we validate:**
- Identical flows show `=` match markers on all rows
- Divergent flows show `≠` with diff explanation (mc_start, mc_end, sections)
- Summary footer shows section counts, unfolded lengths, match ratio
- `flow.diff_diagram(other)` produces identical output to `flow_comparison_diagram(flow, other)`

**Test data:** Tests use inline `MockMeasureData` with PyArrow tables (6 MCs with
repeats and voltas, and 3-MC minimal controller). This follows the established pattern
from `tests/timelines/test_flow.py`.

## Running Tests

```bash
# Run only display tests
cd timetoalign
python -m pytest tests/display/ -v

# Run with coverage
python -m pytest tests/display/ --cov=timetoalign.display --cov-report=term-missing
```

## Known Limitations

1. **Terminal width:** The `width` parameter controls output width but does not
   auto-detect terminal size. Users should pass an appropriate width.

2. **Very long coordinates:** Coordinate values over 6 digits may cause column
   misalignment. This is acceptable for the intended use case (music alignment).

3. **Deeply nested children:** Only immediate children are shown. Grandchildren
   are not displayed (by design - keeps output manageable).
