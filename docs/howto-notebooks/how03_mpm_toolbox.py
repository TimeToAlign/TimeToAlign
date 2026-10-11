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
# # How to Load an MPM-Toolbox Project (Score, Modelled Performance, Observed Alignment)
#
# An MPM-Toolbox project is a triple of sibling XML files describing **one**
# musical work: a `.msm` notated score, a `.mpm` *modelled* performance overlay
# (tempo, dynamics, articulation, … as expressive markup), and a `.mpr` project
# file carrying an *observed* audio-to-score alignment. This guide loads such a
# project in a single call and arranges it as one multimodal
# {{< glossary AlignmentBundle >}} spanning the logical (score), physical
# (performance), and graphical (spectrogram) domains.
#
# The work is Beethoven's *Eroica* Variations, Op. 35 — Var. XIV — in a 1971
# Curzon recording. By the end we will have, in one bundle: the notated score in
# two logical units, the performance markup carried as
# {{< glossary Event >}}s, a modelled tempo curve mapping score quarters to
# seconds, the observed onsets tied back to the score note by note, and the
# recording's spectrogram as a graphical time axis.
#
# We **load** the modelled markup and the observed alignment exactly as they sit
# on disk. Nothing here runs an aligner, and the tempo model is read as written
# — we do not render a beat-accurate performance from it.
#
# The arc:
#
# 1. Load the whole project in one call.
# 2. Read the logical score in two units, linked by a
#    {{< glossary ConversionMap >}}.
# 3. Read the performance markup carried as {{< glossary Event >}}s on the score.
# 4. Read the modelled quarters→seconds tempo map.
# 5. Read the observed alignment — the physical performance and its cross-group
#    {{< glossary MatchClaim >}}s.
# 6. Read the recording's spectrogram as a graphical (pixels) time axis.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

from timetoalign.core import TimeUnit
from timetoalign.loader.alignment import MpmLoader
from timetoalign.testdata import ensure_data

ROOT = ensure_data("mpm_toolbox")
MPR = (
    ROOT
    / "MPRproject_1971Curzon_VariationXIV"
    / "Beethoven_op35_1971Curzon_Var14only.mpr"
)

# %% [markdown]
# ## 1. Load the project in one call
#
# {{< glossary MpmLoader >}} is given the `.mpr` project file; it resolves the
# sibling `.msm` and `.mpm` by the bare filenames the project names, parses the
# score, the selected performance's markup, and the observed alignment, and
# binds the recording's audio for its sample rate. `from_file()` is the one-line
# form of the standard two-phase loader pattern.

# %%
loader = MpmLoader.from_file(MPR)
loader

# %% [markdown]
# The `.mpm` may hold several `<performance>` blocks; by default the first is
# selected. The pulses-per-quarter grid (720) and the chosen performance name
# are read straight from the files. To select a different performance, pass
# `MpmLoader().load(MPR, performance="...")`.

# %%
{
    "pulses per quarter": loader.ppq,
    "performance": loader.performance_name,
}

# %% [markdown]
# `create_bundle()` assembles the {{< glossary AlignmentBundle >}}: four
# timelines arranged in two groups — a shared logical `"score"` group and a
# physical `"perf"` group. Everything below reads from this single bundle.

# %%
bundle = loader.create_bundle()
bundle

# %%
{
    "timelines": bundle.n_timelines,
    "groups": bundle.n_groups,
    "groups_listed": bundle.group_ids,
    "timelines_listed": bundle.timeline_ids,
}

# %% [markdown]
# ---
#
# ## 2. The logical score, in two units
#
# The score lives in the `"score"` group, in **two logical units** — the same
# notes, measured two ways:
#
# - `score:dlt_1` carries integer **tick** onsets (the `pulsesPerQuarter` grid the
#   score was notated on).
# - `score:clt_1` carries **quarter-note** onsets (an exact rational).

# %%
score_dlt = bundle.get_timeline("score:dlt_1")
score_dlt

# %%
score_clt = bundle.get_timeline("score:clt_1")
score_clt

# %% [markdown]
# Both hold the same 251 notes. The `.msm` records each note's MIDI pitch
# together with the spelling it was notated with (`pitchname` / `octave`); a look
# at the first few notes of `score:clt_1` shows what the score carried:

# %%
score_clt.get_events().table.slice(0, 5).to_pandas()[
    ["id", "pitch", "pitchname", "octave"]
]

# %% [markdown]
# ### The tick grid and the quarter grid are one conversion apart
#
# The two logical timelines are not independent: a `TicksToQuarters`
# {{< glossary ConversionMap >}} on `score:dlt_1` carries the tick grid to
# quarters — 720 ticks to the quarter. A {{< glossary TimeStamp >}} on
# `score:dlt_1` exposes the conversion at any coordinate, so asking for the
# quarter reading at a tick is the continuous↔discrete link of the logical
# domain made visible. The reading comes back typed: an `IdCoordinate` in
# `quarters`, still tagged with the tick timeline it converted from, because a
# unit conversion changes the representation and not which timeline you are
# standing on. Quarters are exact, so 360 ticks reads as `1/2` rather than
# `0.5`:

# %%
{
    "tick 360 -> quarters": score_dlt.get_timestamp(360).get_unit(TimeUnit.quarters),
    "tick 720 -> quarters": score_dlt.get_timestamp(720).get_unit(TimeUnit.quarters),
    "tick 1440 -> quarters": score_dlt.get_timestamp(1440).get_unit(TimeUnit.quarters),
}

# %% [markdown]
# ---
#
# ## 3. The performance markup, as events on the score
#
# A `.mpm` overlays the score with expressive markup — tempo, dynamics,
# articulation, asynchrony, and any other map type the project carries. The
# loader places every markup entry on `score:dlt_1` as an {{< glossary Event >}}
# at its tick onset, sitting alongside the `Note` events. A single logical
# timeline therefore carries both the notes and the modelled performance markup,
# read with the same event query used everywhere else — filter by `event_type`.
#
# The coordinate (`start`) comes back as a number; the remaining markup columns
# currently round-trip as strings, so we cast them as we read.

# %% [markdown]
# ### Tempo
#
# Each `Tempo` event carries a beats-per-minute reading. The MPM value may be an
# inline number or a **style name** declared in the performance's style block;
# either way the loader resolves it to a number, keeping the original token in a
# `*_label` column. Here the style name `"Meno mosso."` resolves to 100 BPM:

# %%
tempo_events = score_dlt.get_events().filter(event_type="Tempo").to_dataframe()

tempo_table = tempo_events[["start", "bpm_label", "bpm"]].copy()
tempo_table["tick"] = tempo_table.pop("start").astype(int)
tempo_table["bpm"] = tempo_table["bpm"].astype(float)

tempo_table[["tick", "bpm_label", "bpm"]].head(7)

# %% [markdown]
# ### Dynamics
#
# `Dynamics` events resolve the same way: the dynamic mark `"p"` is a style name
# that resolves to a MIDI volume of 48, with the mark preserved in
# `volume_label`:

# %%
dynamics_events = score_dlt.get_events().filter(event_type="Dynamics").to_dataframe()

dynamics_table = dynamics_events[["start", "volume_label", "volume"]].copy()
dynamics_table["tick"] = dynamics_table.pop("start").astype(int)
dynamics_table["volume"] = dynamics_table["volume"].astype(float)

dynamics_table[["tick", "volume_label", "volume"]].head(6)

# %% [markdown]
# ### Articulation
#
# `Articulation` events name an articulation (`staccato`, `tenuto`, …) and, where
# the performance declares a definition for it, resolve its numeric attributes.
# A `staccato` here resolves to an absolute played duration of 160 ms; the
# `noteid` it applies to is carried alongside (its leading `#` stripped):

# %%
articulation_events = (
    score_dlt.get_events().filter(event_type="Articulation").to_dataframe()
)

staccato = articulation_events[articulation_events["name"] == "staccato"].copy()
staccato["tick"] = staccato.pop("start").astype(int)
staccato["absolute_duration_ms"] = staccato["absolute_duration_ms"].astype(float)

staccato[["tick", "name", "absolute_duration_ms", "noteid"]].head(5)

# %% [markdown]
# The point of this section: tempo and dynamics style names resolve to numeric
# values, articulations resolve to played durations and velocities, and *every*
# one of them is an event on the score timeline, queried the same way the notes
# are. Any map type the loader does not model specially is still emitted, with
# its raw attributes carried verbatim, so nothing in a project is silently
# dropped.

# %% [markdown]
# ---
#
# ## 4. The modelled tempo, as quarters → seconds
#
# The performance's tempo markup also defines, segment by segment, how fast the
# score is taken. The loader integrates it into a modelled quarters→seconds
# {{< glossary ConversionMap >}} — a `TableMap` it exposes directly:

# %%
tempo_map = loader.tempo_map
tempo_map

# %% [markdown]
# It is anchored at the score's start (`0` quarters → `0` seconds) and walks
# forward at each tempo segment's pace. Converting a couple of quarter positions
# reads the modelled clock-time at which the score reaches them:

# %%
{
    "quarter 0 -> seconds": tempo_map(0),
    "quarter 39.5 -> seconds": tempo_map(39.5),
    "quarter 40.0 -> seconds": tempo_map(40.0),
}

# %% [markdown]
# This is a **constant-tempo-per-segment** model: each tempo entry sets a flat
# pace until the next. Accelerando / ritardando ramps (an entry's
# `transition.to`) are preserved as a `Tempo`-event attribute but are *not*
# rendered into the curve — the map reads what the project modelled, deliberately
# stopping short of synthesising a beat-accurate performance.

# %% [markdown]
# ---
#
# ## 5. The observed alignment
#
# The model above is one account of the performance. The `.mpr` carries another:
# an **observed** alignment, recording for every score note the moment it was
# actually played in the recording. The loader places these onsets in the
# physical `"perf"` group, in **two units** — `perf:cpt_1` in seconds and
# `perf:dpt_1` in samples, linked (as in any physical timeline) by a
# `SamplesToSeconds` {{< glossary ConversionMap >}} carrying the recording's
# sample rate.

# %%
perf_cpt = bundle.get_timeline("perf:cpt_1")
perf_cpt

# %% [markdown]
# The score group and the performance group are tied together by cross-group
# {{< glossary MatchClaim >}}s — one per score note. The `.mpr` alignment is a
# perfect bijection (every score note has exactly one observed onset and vice
# versa), so every claim is **synchronous**: there are no
# {{< glossary NOMATCH >}} gaps.

# %%
claims = bundle.cross_group_claims

{
    "total claims": len(claims),
    "synchronous (matched)": sum(1 for c in claims if c.is_synchronous),
    "NOMATCH": sum(1 for c in claims if not c.is_synchronous),
}

# %% [markdown]
# A single claim relates a score quarter to an observed performed second. This
# one anchors the note at score quarter 0.5 to the moment it was played, 0.8
# seconds into the recording:

# %%
quarter_half_claims = [
    c
    for c in claims
    if c.is_synchronous
    and c.start_anchor is not None
    and float(c.start_anchor.coordinate_a) == 0.5
]
quarter_half_claims[0]

# %% [markdown]
# ### One score position, read across the domains
#
# Because the performance is anchored back to the score, a single score
# coordinate resolves across the whole bundle. `get_matchstamp_at` takes a
# coordinate on `score:clt_1` (quarters) and returns the corresponding coordinate
# on every connected timeline — here score quarter 2.0 mapped to the second at
# which it was observed:

# %%
stamp = bundle.get_matchstamp_at(2.0, "score:clt_1")
stamp

# %% [markdown]
# Read across that {{< glossary MatchStamp >}}: the same notated quarter resolves
# to a logical position (2.0 quarters) and an observed physical position (≈ 2.18
# seconds), one query crossing from the score domain to the performance domain
# with no aligner ever run.

# %% [markdown]
# ---
#
# ## 6. The spectrogram, a graphical axis
#
# The project also ships a rendered **spectrogram** of the recording — a `.png`
# whose horizontal axis is time. The loader reads that axis as a third
# performance-group {{< glossary Timeline >}}, `perf:dgt_1`, measured in
# **pixels** (frame columns): one position per column of the image. This is the
# graphical domain — the same work, now reached visually.
#
# Unlike the other timelines, this one carries **no** {{< glossary Event >}}s. A
# spectrogram column is not a note; it is a tick mark on a picture's time axis.
# The timeline's job is to *be* that axis — its length is the image's width in
# frame columns:

# %%
perf_dgt = bundle.get_timeline("perf:dgt_1")
perf_dgt

# %%
{
    "unit": perf_dgt.unit.name,
    "length (frame columns)": perf_dgt.length.value,
    "events (it is an axis, not events)": perf_dgt.n_events,
}

# %% [markdown]
# A pixel column is only meaningful once tied to clock time. Each column
# advances the recording by a fixed hop of audio samples, so `perf:dgt_1` carries
# a px→seconds {{< glossary ConversionMap >}} — a `ScalarMap` whose scalar is
# `hopSize / sample_rate`. Pulling it off the timeline and reading a couple of
# columns places the picture on the same seconds clock as the performance:

# %%
px_to_seconds = perf_dgt.get_conversion_map(TimeUnit.seconds)

last_column = perf_dgt.length.value
{
    "scalar (seconds per column)": px_to_seconds.scalar,
    "column 0 -> seconds": perf_dgt.get_timestamp(0).get_unit(TimeUnit.seconds),
    f"column {last_column} -> seconds": perf_dgt.get_timestamp(last_column).get_unit(
        TimeUnit.seconds
    ),
}

# %% [markdown]
# Column 0 sits at the start of the recording; the rightmost column lands at the
# spectrogram's full time span — the audio duration. The picture's x-axis and
# the performance's seconds are now one conversion apart, exactly as ticks and
# quarters were on the score.

# %% [markdown]
# ## Recap
#
# | What the bundle expresses | How |
# |---|---|
# | Score, two logical units | `score:dlt_1` (ticks) + `score:clt_1` (quarters), one `TicksToQuarters` map |
# | Performance markup | `Tempo` / `Dynamics` / `Articulation` events on `score:dlt_1`, via `filter(event_type=...)` |
# | Modelled tempo | a quarters→seconds `TableMap` (`loader.tempo_map`), constant-tempo-per-segment |
# | Observed performance, two physical units | `perf:cpt_1` (s) + `perf:dpt_1` (samples), one `SamplesToSeconds` |
# | Spectrogram, a graphical axis | `perf:dgt_1` (pixels), no events, one px→seconds `ScalarMap` |
# | Score ↔ performance | one synchronous {{< glossary MatchClaim >}} per note (a perfect bijection) |
# | A score position read across domains | `bundle.get_matchstamp_at(coord, "score:clt_1")` |
#
# One {{< glossary AlignmentBundle >}} now expresses this single work across all
# three domains — logical (score ticks and quarters), physical (performance
# seconds and samples), and graphical (spectrogram pixels). Within each group the
# units are tied by {{< glossary ConversionMap >}}s; across the groups the score
# and performance are tied note by note by {{< glossary MatchClaim >}}s. A single
# object holds what was notated, how it was modelled, how it was actually played,
# and how it looks — the score, the model, the recording, and its picture, all on
# one set of linked clocks.
