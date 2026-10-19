# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ToDo: This notebook has not been vetted yet and therefore not been integrated into the documentation.
#
# # How to Use Flow Control and FlowMaps
#
# A musical score is compact: a repeated section appears once on the page but is
# played two or more times in performance.  A {{< glossary Timeline >}} that models
# the printed page is **folded** — coordinates are shared across multiple playthrough
# instances.  A recording is **unfolded** — every event occupies a unique physical
# time point.
#
# {{< glossary Break >}} and {{< glossary Jump >}} are the two primitives that encode
# this structure.  A **FlowMap** is the bidirectional coordinate transformation that
# connects the folded and unfolded views.
#
# **Learning objectives:**
#
# 1. Construct {{< glossary Break >}} and {{< glossary Jump >}} objects by hand
# 2. Load a score and inspect its flow structure via `ScoreFlowController`
# 3. Compare all `FlowMode` variants and their effect on unfolded length
# 4. Use `FlowMap.unfold_coordinate()` and `FlowMap.fold()` for coordinate transformation
# 5. Attach a `FlowMap` to a {{< glossary Timeline >}} and query it by id
# 6. Build a custom `FlowMap` from explicit quarter-beat section boundaries
#
# **Prerequisites:** `how01_manual_timeline_construction`, `how01_coordinate_math`

# %% [markdown]
# ## Setup

# %%
from fractions import Fraction
from pathlib import Path

import pandas as pd

from timetoalign import Coordinate, Ms3Loader, SegmentNameGenerator, TimeUnit
from timetoalign.core.enums import FlowControlElement, FlowMode
from timetoalign.timelines.flow import FlowMap, compute_qb_sections
from timetoalign.timelines.flowcontrol import Break, Jump

DATA_DIR = Path("~/git/beethoven_eroica_variations_op35/measures").expanduser()
COUNT_THROUGH_TSV = (
    DATA_DIR
    / "15-variations-and-fugue-in-e-flat-major-op-35-eroica-ludwig-van-beethoven.measures.tsv"
)
PER_VARIATION_TSV = (
    DATA_DIR
    / "15-variations-and-fugue-in-e-flat-major-op-35-eroica-variations-ludwig-van-beethoven.measures.tsv"
)
NOTES_TSV = (
    DATA_DIR.parent
    / "notes"
    / "15-variations-and-fugue-in-e-flat-major-op-35-eroica-variations-ludwig-van-beethoven.notes.tsv"
)

# %%
count_through_loader = Ms3Loader.from_file(PER_VARIATION_TSV)
controller = count_through_loader.create_flow_controller()
controller

# %%
per_variation_loader = Ms3Loader.from_file(PER_VARIATION_TSV)
controller = per_variation_loader.create_flow_controller()
controller

# %% [markdown]
# ---
#
# ## Part 1: Break and Jump — The Two Primitives
#
# ### {{< glossary Contiguity >}} and how it is broken
#
# A {{< glossary Timeline >}} is contiguous by default: coordinates increase
# monotonically without gaps.  Two event types can change this:
#
# - **{{< glossary Break >}}** — voids {{< glossary Contiguity >}} at a specific
#   coordinate.  No {{< glossary TimeInterval >}} may span across a Break.
# - **{{< glossary Jump >}}** — makes two non-adjacent coordinates contiguous,
#   modelling repeat barlines, *da capo*, *dal segno*, and similar instructions.
#
# Both are **frozen dataclasses** — immutable value objects with no side effects.

# %%
# A Break at QB 20: ends a section, nothing can span across it
section_end = Break(
    coordinate=Coordinate(Fraction(20), TimeUnit.quarters),
    control_type=FlowControlElement.section_break,
    label="||",
)
section_end

# %%
{
    "position": float(section_end.position),
    "unit": str(section_end.unit),
    "is_break": FlowControlElement.section_break.is_break,
    "is_jump": FlowControlElement.section_break.is_jump,
}

# %% [markdown]
# ### Jump types
#
# A {{< glossary Jump >}} carries a `from_coordinate`, a `to_coordinate` and a
# `control_type`. The control type decides on which playthrough pass the jump
# fires (its `condition`): a `repeat_end` on its first `repeat_count` passes, a
# `to_coda` only after a D.C. or D.S., every other jump always.

# %%
# Standard repeat: jump from QB 20 back to QB 8 on the first pass only
repeat = Jump(
    from_coordinate=Coordinate(Fraction(20), TimeUnit.quarters),
    to_coordinate=Coordinate(Fraction(8), TimeUnit.quarters),
    control_type=FlowControlElement.repeat_end,
    repeat_count=1,
)
{
    "condition": repeat.condition.value,
    "from": float(repeat.from_position),
    "to": float(repeat.to_position),
    "is_backward": repeat.is_backward,
    "distance": float(repeat.distance),
    "active_on_pass_1": repeat.is_active(pass_number=1),
    "active_on_pass_2": repeat.is_active(pass_number=2),
}

# %%
# Da Capo: jump to the beginning whenever the traversal reaches it
dc = Jump(
    from_coordinate=Coordinate(Fraction(100), TimeUnit.quarters),
    to_coordinate=Coordinate(Fraction(0), TimeUnit.quarters),
    control_type=FlowControlElement.da_capo,
)
{
    "type": dc.control_type.value,
    "condition": dc.condition.value,
    "is_jump": FlowControlElement.da_capo.is_jump,
    "is_target": FlowControlElement.da_capo.is_target,
}

# %% [markdown]
# The full `FlowControlElement` vocabulary groups into six categories:
#
# - **Jump instructions** — `repeat_end`, `da_capo` (`dc`), `dal_segno` (`ds`),
#   `dal_segno_al_coda` (`dsac`), `dal_segno_al_fine` (`dsaf`),
#   `da_capo_al_coda` (`dcac`), `da_capo_al_fine` (`dcaf`), `to_coda`
# - **Target markers** — `repeat_start`, `segno`, `coda`
# - **Breaks** (void {{< glossary Contiguity >}}) — `section_break`, `fine`
# - **Structural markers** (do not void contiguity) — `double_barline`,
#   `final_barline`
# - **Boundary markers** — `first_measure`, `last_measure`
# - **Abstract roles** — `jump_from`, `jump_to` (super-categories emitted by
#   loaders alongside concrete types so consumers can filter jump origins or
#   destinations without enumerating every subtype)
#
# `FancyStrEnum` is iterable, so we can read the full taxonomy off the enum
# itself rather than restating it.

# %%
# is_jump / is_break / is_target / is_structural_marker at a glance
pd.DataFrame(
    [
        {
            "type": ct.value,
            "is_jump": ct.is_jump,
            "is_break": ct.is_break,
            "is_target": ct.is_target,
            "is_structural": ct.is_structural_marker,
        }
        for ct in FlowControlElement
    ]
)

# %% [markdown]
# ---
#
# ## Part 2: Loading Flow Control from a Score
#
# In practice you never construct {{< glossary Break >}} and {{< glossary Jump >}}
# objects by hand — they are parsed from score files (MuseScore, MusicXML, TSV).
# The `ScoreFlowController` reads the parsed measure data and exposes the full flow
# structure.
#
# The score used throughout this guide is Beethoven's **WoO 71** (15 Variations in
# A major), a concise piece with a clear repeat structure and mixed volta endings.

# %%
# from_file() handles both phases: load + internal state construction
per_variation_loader = Ms3Loader.from_file(PER_VARIATION_TSV)
controller = per_variation_loader.create_flow_controller()
controller

# %%
# ASCII diagram of the repeat/jump structure
controller.diagram()

# %%
# Quarter-beat coordinates where section breaks occur
boundaries = controller.get_section_boundary_coordinates()
{
    "n_atomic_sections": len(boundaries) + 1,
    "break_positions_qb": [float(b) for b in boundaries],
}

# %% [markdown]
# ---
#
# ## Part 2b: Naming the Atomic Sections
#
# Every atomic section carries a short **label** — the letters `A`, `B`, `C` …
# you see in `controller.diagram()` and in each section's `id`.  By default a
# *volta* (an alternative ending such as a *prima/seconda volta*) does **not**
# consume the next letter.  Instead it inherits the preceding section's letter
# plus a positional numeric suffix: a section `W` followed by two endings reads
# `W`, `W1`, `W2`.  Look at the right-hand end of the diagram above — the two
# volta slots render `┌1─W1─┌2─W2─`.
#
# The `┌N` number printed in the corner is the volta's **own ending number**
# (first ending → `1`, second ending → `2`).  It is read straight from the score
# and is independent of the label: the `1` and `2` in `W1`/`W2` happen to agree
# here only because `W` is the first volta group with exactly two endings.  The
# label suffix is *positional* (first volta after a base → `1`), so two
# independent volta groups read `W, W1, W2` then `X, X1, X2`, never `X3, X4`.

# %%
# The default labels — note the volta-suffixed W1, W2 at the tail
[s.id for s in controller.get_sections()]

# %% [markdown]
# ### Customising the labels
#
# The labelling strategy is a `SegmentNameGenerator`, passed to
# `create_flow_controller(name_generator=...)`.  Two policies are configurable.
#
# **`volta_suffix=False`** restores the historical behaviour in which a volta
# consumes the next letter in sequence — the two endings become fresh sections
# `X` and `Y`, and the diagram corner reads `┌1─X──┌2─Y──`:

# %%
controller_legacy = per_variation_loader.create_flow_controller(
    name_generator=SegmentNameGenerator(volta_suffix=False)
)
{
    "default_tail": [s.id for s in controller.get_sections()][-4:],
    "legacy_tail": [s.id for s in controller_legacy.get_sections()][-4:],
}

# %% [markdown]
# **`alphabet=`** accepts any sequence of symbols to drive the base labels.
# Here we label sections with Greek letters instead of Latin ones.  Once the
# alphabet is exhausted it repeats with a numeric suffix (`α2`, `β2`, …), and the
# volta-suffix rule still applies on top (so a volta of section `δ2` reads
# `δ21`, `δ22`):

# %%
greek = "αβγδεζηθικλμνξοπρστυφχψω"
controller_greek = per_variation_loader.create_flow_controller(
    name_generator=SegmentNameGenerator(alphabet=list(greek))
)
[s.id for s in controller_greek.get_sections()]

# %% [markdown]
# ---
#
# ## Part 3: FlowModes
#
# A `FlowMode` controls *how* the controller interprets the flow control events when
# computing a traversal path through the score.  The same printed score can yield
# very different unfolded lengths depending on the chosen mode.

# %%
rows = []
for mode in FlowMode:
    try:
        flow = controller.compute_flow(mode)
        rows.append(
            {
                "mode": mode.value,
                "folded_qb": flow.folded_length,
                "unfolded_qb": flow.unfolded_length,
                "has_repeats": flow.has_repeats,
            }
        )
    except Exception as exc:
        rows.append({"mode": mode.value, "error": str(exc)})

pd.DataFrame(rows)

# %% [markdown]
# Key modes and when to use them:
#
# | Mode | Semantics | When to use |
# |------|-----------|-------------|
# | `ATOMIC` | Finest segmentation; no jump resolution | Debugging, structural inspection |
# | `DEFAULT` | Repeats played twice; D.C./D.S. resolved | General-purpose alignment |
# | `PRINTED` | All repeats ignored; matches the page | Optical alignment to score images |
# | `SINGLE_PASS` | One linear pass; no jumps fire | First-pass annotation |
# | `MS3` | Ms3 software interpretation | Cross-validation with ms3 TSV exports |
# | `PARTITURA_MINIMAL` / `PARTITURA_MAXIMAL` | Partitura interpretation variants | Cross-validation with partitura |
# | `MUSIC21` | music21 interpretation | Cross-validation with music21 |

# %%
# DEFAULT vs PRINTED: the clearest contrast
flow_default = controller.compute_flow(FlowMode.default)
flow_printed = controller.compute_flow(FlowMode.printed)

{
    "default_unfolded_qb": flow_default.unfolded_length,
    "printed_unfolded_qb": flow_printed.unfolded_length,
    "extra_qb_from_repeats": flow_default.unfolded_length
    - flow_printed.unfolded_length,
}

# %% [markdown]
# ---
#
# ## Part 4: The FlowMap
#
# A `FlowMap` is the coordinate transformation derived from a `Flow`.  It maps
# between the **folded** (printed) coordinate space and the **unfolded**
# (performance) coordinate space.
#
# The mapping is asymmetric by design:
#
# - `unfold_coordinate(coord)` → `list[Fraction]` — a folded coordinate that falls
#   inside a repeated section appears at *multiple* unfolded positions
# - `fold(coord)` → `Fraction` — always returns a single value because the
#   unfolded timeline has no repeated positions

# %%
flow_map = controller.create_flow_map()  # DEFAULT flow by default
{
    "n_sections": flow_map.n_sections,
    "total_target_length_qb": float(flow_map.total_target_length),
}

# %%
# Pick a coordinate inside the repeated section
coord = boundaries[0] / 2 if boundaries else Fraction(5)
unfolded = flow_map.unfold_coordinate(coord)
{
    "source_coord": float(coord),
    "unfolded_positions": [float(u) for u in unfolded],
    "n_appearances": len(unfolded),
}

# %%
# Every unfolded position folds back to the same source coordinate
for target in unfolded:
    source = flow_map.fold(target)
    print(f"  unfolded {float(target):.3f}  →  folded {float(source):.3f}")

# %%
# A coordinate beyond all repeats appears exactly once
late_coord = flow_map.fold(flow_map.total_target_length - 1)
{
    "source_coord": float(late_coord),
    "n_appearances": len(flow_map.unfold_coordinate(late_coord)),
}

# %% [markdown]
# ### FlowMap.inverse()
#
# `inverse()` swaps source and target.  The result is useful for annotating an
# *unfolded* {{< glossary Timeline >}} with back-references into the printed score:
# given an unfolded position, fold it back to the score page.

# %%
inverse_map = flow_map.inverse()
{
    "original_n_sections": flow_map.n_sections,
    "inverse_n_sections": inverse_map.n_sections,
    "inverse_source_length": float(inverse_map.total_target_length),
}

# %% [markdown]
# ---
#
# ## Part 5: Attaching FlowMaps to a Timeline
#
# A `FlowMap` is most useful when attached to a {{< glossary Timeline >}}.
# Once attached, `timeline.unfold_coordinate()` and `timeline.fold()` delegate to it, keeping
# coordinate transformations co-located with the events they describe.
#
# Multiple `FlowMap` objects can coexist on the same timeline, each identified by a
# string key.  This lets you compare interpretations without duplicating the event
# data.

# %%
# Load notes + measures together
loader_full = Ms3Loader.from_file(NOTES_TSV, PER_VARIATION_TSV)
score_tl = loader_full.create_timeline()
{
    "id": score_tl.id,
    "n_events": score_tl.n_events,
    "unit": str(score_tl.unit),
    "length_qb": float(score_tl.length),
}

# %%
# Build the ScoreFlowController from the same loader's measure store
controller_full = loader_full.create_flow_controller()

# Create and attach the default FlowMap
fm_default = controller_full.create_flow_map()
score_tl.add_flow_map(fm_default)

score_tl.list_flow_maps()

# %%
# Unfold a coordinate from the repeated section
sample_coord = float(boundaries[0]) / 2 if boundaries else 5.0
positions = score_tl.unfold_coordinate(sample_coord)
{
    "folded_coord": sample_coord,
    "unfolded_positions": positions,
    "n_appearances": len(positions),
}

# %%
# Fold the second appearance back to the score
if len(positions) > 1:
    folded_back = score_tl.fold(positions[1])
    {"second_unfolded": positions[1], "folded_back": folded_back}

# %%
# Attach additional FlowMaps for other modes — no data duplication
for mode in [FlowMode.printed, FlowMode.single, FlowMode.atomic]:
    fm = controller_full.create_flow_map_for_mode(mode)
    score_tl.add_flow_map(fm, id=mode.value)

score_tl.list_flow_maps()

# %%
# The same folded coordinate behaves differently under each map
pd.DataFrame(
    [
        {
            "map_id": map_id,
            "n_appearances": len(score_tl.unfold_coordinate(sample_coord, id=map_id)),
        }
        for map_id in score_tl.list_flow_maps()
    ]
)

# %% [markdown]
# ---
#
# ## Part 6: Custom FlowMaps (Advanced)
#
# Two scenarios call for manual `FlowMap` construction:
#
# 1. **Correcting QB boundaries** — for scores with unusual measure durations or
#    mid-measure jumps, the automatic MC-to-QB conversion may need to be overridden.
# 2. **Modelling a non-standard interpretation** — a performance tradition that
#    differs from the printed repeat structure.
#
# Both go through `FlowMap.from_qb_sections(flow, qb_sections, id=...)`.  The
# `flow` carries the structural metadata (mode, folded length, source provenance);
# `qb_sections` is a `(qb_start, qb_end)` tuple **per section**, and its length
# must equal `len(flow.sections)`.
#
# `compute_qb_sections(flow, controller)` is the helper that derives these tuples
# from the controller's measure data — the same computation the default
# constructor uses internally.

# %%
# Inspect the QB section boundaries that `create_flow_map()` would use
flow_base = controller_full.compute_flow(FlowMode.default)
default_qb_sections = compute_qb_sections(flow_base, controller_full)

pd.DataFrame(
    [
        {"section": i, "qb_start": float(s), "qb_end": float(e)}
        for i, (s, e) in enumerate(default_qb_sections)
    ]
).head(8)

# %%
# Rebuild the FlowMap from those QB sections — equivalent to controller.create_flow_map()
manual_map = FlowMap.from_qb_sections(flow_base, default_qb_sections, id="manual")
{
    "default_n_sections": flow_map.n_sections,
    "manual_n_sections": manual_map.n_sections,
    "default_total_length": float(flow_map.total_target_length),
    "manual_total_length": float(manual_map.total_target_length),
}

# %% [markdown]
# A real custom use case: suppose the score notates a section that, by performance
# convention, is repeated **three** times rather than twice.  We can model this by
# duplicating that section's QB tuple before passing it to `from_qb_sections()`.
# Note that we must also rebuild the `Flow.sections` list to match the new count.

# %%
from timetoalign.timelines.flow import Flow

# Duplicate the first DEFAULT section to add an extra pass
extra_pass_qb = [default_qb_sections[0]] + list(default_qb_sections)
extra_pass_flow = Flow.from_sections(
    sections=[flow_base.sections[0]] + list(flow_base.sections),
    mode=FlowMode.custom,
    folded_length=flow_base.folded_length,
)

extra_pass_map = FlowMap.from_qb_sections(
    extra_pass_flow, extra_pass_qb, id="extra_pass"
)
score_tl.add_flow_map(extra_pass_map)

{
    "default_total_qb": float(flow_map.total_target_length),
    "extra_pass_total_qb": float(extra_pass_map.total_target_length),
    "added_qb": float(
        extra_pass_map.total_target_length - flow_map.total_target_length
    ),
    "appearances_in_default": len(
        score_tl.unfold_coordinate(sample_coord, id="default")
    ),
    "appearances_in_extra_pass": len(
        score_tl.unfold_coordinate(sample_coord, id="extra_pass")
    ),
}

# %% [markdown]
# ---
#
# ## Summary
#
# | Task | API |
# |------|-----|
# | Inspect a score's flow structure | `loader.create_flow_controller()` |
# | Visualise the repeat diagram | `controller.diagram()` |
# | List section break positions | `controller.get_section_boundary_coordinates()` |
# | Compute a traversal for a given mode | `controller.compute_flow(FlowMode.X)` |
# | Get the default `FlowMap` | `controller.create_flow_map()` |
# | Get a mode-specific `FlowMap` | `controller.create_flow_map_for_mode(FlowMode.X)` |
# | Map folded → unfolded (1 → N) | `flow_map.unfold_coordinate(coord)` · `timeline.unfold_coordinate(coord)` |
# | Map unfolded → folded (N → 1) | `flow_map.fold(coord)` · `timeline.fold(coord)` |
# | Reverse a `FlowMap` | `flow_map.inverse()` |
# | Attach to a {{< glossary Timeline >}} | `timeline.add_flow_map(flow_map, id=...)` |
# | List attached maps | `timeline.list_flow_maps()` |
# | Build from explicit QB sections | `FlowMap.from_qb_sections(flow, sections, id=...)` |
