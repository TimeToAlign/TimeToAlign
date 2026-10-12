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
# # How to Load a parangonada Note Alignment Across Several Performances
#
# *parangonar* aligns a symbolic score to a performance and exports the result
# as a *parangonada* triple of CSV files — the score notes, the performed notes,
# and the note-level correspondences between them. This guide loads such an
# export for **one** work performed **five** different times into a single
# multimodal {{< glossary AlignmentBundle >}}.
#
# We **load an existing alignment**; nothing here runs an aligner. The
# correspondences were computed once, offline, and written to disk; the loader's
# job is to read them faithfully and arrange them as timelines and
# {{< glossary MatchClaim >}}s.
#
# The work is Beethoven's *Eroica* Variations, Op. 35 — Var. XIV — in five
# recordings spanning six decades: Szegedi (1966), Gould (1970), Curzon (1971),
# Brendel (1985), and Hewitt (2023). One score, five performances, and the
# note-by-note alignment that ties each performance back to the score.
#
# The arc:
#
# 1. Load the whole export in one call.
# 2. Read the shared logical score, in two units, linked by a
#    {{< glossary ConversionMap >}}.
# 3. Read one performer's physical timelines, in two units, linked by a
#    {{< glossary ConversionMap >}}.
# 4. Inspect the cross-group {{< glossary MatchClaim >}}s — both the synchronous
#    matches and the {{< glossary NOMATCH >}} sentinels — and read one shared
#    score position across all five performances.
# 5. Read each performance's measured per-beat tempo and dynamics, carried as
#    events on its own timeline.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import pandas as pd

from timetoalign.core import TimeUnit
from timetoalign.loader.alignment import ParangonadaLoader
from timetoalign.testdata import ensure_data

root = ensure_data("parangonar")
dataset_dir = root / "Beethoven_Eroica_op35-cpjku"

# %% [markdown]
# ## 1. Load the export in one call
#
# `ParangonadaLoader` discovers every performance under the export's directory,
# parses the shared score once and each performance's notes and alignment, and
# binds each recording's audio for its sample rate. `from_file()` is the
# one-line form of the standard two-phase loader pattern.

# %%
bundle = ParangonadaLoader.from_file(dataset_dir).create_bundle()
bundle

# %% [markdown]
# Twelve timelines arranged in six groups: one shared `"score"` group, and one
# `"perf:<key>"` group per recording. Everything below reads from this single
# bundle.

# %%
{
    "timelines": bundle.n_timelines,
    "groups": bundle.n_groups,
    "groups_listed": bundle.group_ids,
}

# %% [markdown]
# ---
#
# ## 2. The shared logical score
#
# The score is identical across all five performances, so it is parsed once and
# placed in its own group. It appears in **two logical units** — the same notes,
# measured two ways:
#
# - `score:clt_1` carries quarter-note onsets (an exact rational).
# - `score:dlt_1` carries division-grid onsets (the integer tick grid the score
#   was notated on).

# %%
score_clt = bundle.get_timeline("score:clt_1")
score_clt

# %%
score_dlt = bundle.get_timeline("score:dlt_1")
score_dlt

# %% [markdown]
# Both hold the same 251 notes, each carrying its MIDI pitch and voice. A
# look at the first few notes shows the pitch and voice the export recorded
# (the first note sits at quarter −0.5, the score's anacrusis):

# %%
score_clt.get_events().table.slice(0, 5).to_pandas()[["id", "pitch", "voice"]]

# %% [markdown]
# ### The division grid and the quarter grid are one conversion apart
#
# The two logical timelines are not independent: a single
# {{< glossary ConversionMap >}} on `score:dlt_1` carries the division grid to
# quarters. It is a `LinearMap` — 32 divisions to the quarter, shifted by the
# half-quarter anacrusis — and it reproduces the quarter onsets exactly,
# because divisions are the discrete grid the quarters were notated on:

# %%
divs_to_quarters = score_dlt.get_conversion_map(TimeUnit.quarters)

{
    "div 16 -> quarters": divs_to_quarters(16),
    "div 48 -> quarters": divs_to_quarters(48),
    "div 1296 -> quarters": divs_to_quarters(1296),
}

# %% [markdown]
# The same conversion is what a {{< glossary TimeStamp >}} exposes. Querying
# `score:dlt_1` at division 48 and asking for the quarter reading reports `1.0`,
# the continuous↔discrete link of the logical domain made visible at a single
# coordinate:

# %%
score_dlt.get_timestamp(48).get_unit(TimeUnit.quarters)

# %% [markdown]
# ---
#
# ## 3. One performer's physical timelines
#
# Each performance lives in its own group, again in **two units** — the same
# performed notes, measured two ways:
#
# - `perf:<key>:cpt_1` carries onsets in seconds.
# - `perf:<key>:dpt_1` carries the same onsets as sample indices into the
#   recording's audio.
#
# Take Szegedi's 1966 recording:

# %%
szegedi_cpt = bundle.get_timeline("perf:1966_Szegedi:cpt_1")
szegedi_cpt

# %%
szegedi_dpt = bundle.get_timeline("perf:1966_Szegedi:dpt_1")
szegedi_dpt

# %% [markdown]
# The seconds grid and the samples grid are, again, one
# {{< glossary ConversionMap >}} apart — here a `SamplesToSeconds` map carrying
# the recording's 44.1 kHz sample rate. It is the physical-domain counterpart
# of the logical divs→quarters map: 44 100 samples is one second.

# %%
samples_to_seconds = szegedi_dpt.get_conversion_map(TimeUnit.seconds)

{
    "sample_rate (Hz)": samples_to_seconds.sample_rate,
    "44100 samples -> seconds": samples_to_seconds(44100),
    "88200 samples -> seconds": samples_to_seconds(88200),
}

# %% [markdown]
# ---
#
# ## 4. The cross-group alignment
#
# The score group and the five performance groups are tied together by
# cross-group {{< glossary MatchClaim >}}s — one per row of each performance's
# alignment file. A matched note becomes a **synchronous** claim relating a
# score quarter to a performed second; an unmatched note becomes a
# {{< glossary NOMATCH >}} claim, which records the dangling note rather than
# discarding it.

# %%
claims = bundle.cross_group_claims

{
    "total claims": len(claims),
    "synchronous (matched)": sum(1 for c in claims if c.is_synchronous),
    "NOMATCH (score-only or performance-only)": sum(
        1 for c in claims if not c.is_synchronous
    ),
}

# %% [markdown]
# A synchronous claim carries an anchor: a score quarter on one side, the
# performed second on the other. This is Szegedi's note at score quarter 40,
# played 38.18 seconds into the recording:

# %%
szegedi_synchronous = [
    c
    for c in claims
    if c.is_synchronous
    and c.connects("perf:1966_Szegedi:cpt_1")
    and c.start_anchor is not None
    and float(c.start_anchor.coordinate_a) == 40.0
]
szegedi_synchronous[0]

# %% [markdown]
# A {{< glossary NOMATCH >}} claim carries no anchor — there is no second
# coordinate to record — but it keeps the unmatched side's coordinate so the
# dangling note stays legible. Its text repr prints that coordinate inline
# (`...@<coord>`) with a `[NOMATCH]` flag:

# %%
szegedi_nomatch = [
    c for c in claims if not c.is_synchronous and c.connects("perf:1966_Szegedi:cpt_1")
]
szegedi_nomatch[0]

# %% [markdown]
# ### One score position across all five performances
#
# Because every performance is anchored back to the same score, a single score
# coordinate resolves across the whole bundle. `get_matchstamp_at` takes a
# coordinate on `score:clt_1` and returns the corresponding coordinate on every
# timeline connected to it — here, score quarter 40 mapped to the second at
# which each of the five pianists played it:

# %%
bundle.get_matchstamp_at(40.0, "score:clt_1")

# %% [markdown]
# Read across that {{< glossary MatchStamp >}}: the same notated moment falls at
# 37.8 s for Brendel and 38.2 s for Szegedi, but at 63.1 s for Gould — the raw
# material of a tempo comparison, drawn straight from the loaded alignment with
# no aligner ever run.

# %% [markdown]
# ---
#
# ## 5. Measured tempo and dynamics, as events on the performance
#
# The alignment ties each performance to the score note by note, but parangonar
# also measured, for every recording, a per-beat **tempo** and **dynamics**
# profile. These are not a model or a synthesised constant-tempo grid: they are
# the variable tempo and loudness actually measured beat by beat in each
# recording. The loader places them on the performance's own seconds timeline —
# `perf:<key>:cpt_1` — as `Beat` and `Dynamics`
# {{< glossary Event >}}s, sitting alongside the `Note` events at their measured
# onsets. A performance timeline therefore carries three kinds of event: the
# notes that were played, and the measured tempo and dynamics that describe how.
#
# We read them with the same event query used everywhere else — filter by
# `event_type`. The coordinate (`start`) comes back as a number; the remaining
# measured columns currently round-trip as strings, so we cast them as we read.

# %% [markdown]
# ### Tempo: the `Beat` events
#
# Each `Beat` event carries the measured beat-per-minute reading at its onset,
# together with the `measure_number` and `beat` it falls on. There are 63 of
# them — one per measured beat of the variation. Take Szegedi's:

# %%
szegedi_beats = szegedi_cpt.get_events().filter(event_type="Beat").to_dataframe()

szegedi_tempo = szegedi_beats[["start", "measure_number", "beat", "bpm"]].copy()
szegedi_tempo["onset_sec"] = szegedi_tempo.pop("start").astype(float)
szegedi_tempo["measure_number"] = szegedi_tempo["measure_number"].astype(int)
szegedi_tempo["beat"] = szegedi_tempo["beat"].astype(int)
szegedi_tempo["bpm"] = szegedi_tempo["bpm"].astype(float)

{
    "beat events": len(szegedi_tempo),
    "first onset (s)": szegedi_tempo["onset_sec"].iloc[0],
    "measured BPM: min / median / max": (
        round(szegedi_tempo["bpm"].min(), 1),
        round(szegedi_tempo["bpm"].median(), 1),
        round(szegedi_tempo["bpm"].max(), 1),
    ),
}

# %% [markdown]
# The tempo is anything but constant. Reading the first measures of the
# variation shows the measured BPM moving from beat to beat — the local rubato
# of the playing, recorded as data:

# %%
szegedi_tempo.head(8)

# %% [markdown]
# ### Dynamics: the `Dynamics` events
#
# At the same measured onsets sit 63 `Dynamics` events, each carrying the mean
# and peak MIDI velocity measured over its beat. The `.dyn` source rows have no
# onset of their own; the loader joined each to the `Beat` row sharing its
# `(measure_number, beat)` key, so the dynamics land at exactly the same seconds
# as the tempo readings:

# %%
szegedi_dyn = szegedi_cpt.get_events().filter(event_type="Dynamics").to_dataframe()

szegedi_dynamics = szegedi_dyn[
    ["start", "measure_number", "beat", "velocity_mean", "velocity_max"]
].copy()
szegedi_dynamics["onset_sec"] = szegedi_dynamics.pop("start").astype(float)
szegedi_dynamics["measure_number"] = szegedi_dynamics["measure_number"].astype(int)
szegedi_dynamics["beat"] = szegedi_dynamics["beat"].astype(int)
szegedi_dynamics["velocity_mean"] = szegedi_dynamics["velocity_mean"].astype(float)
szegedi_dynamics["velocity_max"] = szegedi_dynamics["velocity_max"].astype(float)

szegedi_dynamics.head(8)

# %% [markdown]
# ### Five interpretations, five tempo profiles
#
# Because every performance carries its own measured `Beat` events, the "five
# interpretations of one work" claim becomes concrete: the same 63 beats of the
# variation, measured in each recording. Summarising each performer's measured
# tempo shows how differently they take it — Gould stretches the variation across
# nearly twice the span of Brendel, at roughly half the median tempo:

# %%
tempo_by_performer = {}
for group in bundle.groups.values():
    key = group.name
    if not key.startswith("perf:"):
        continue
    cpt = bundle.get_timeline(f"{key}:cpt_1")
    beats = cpt.get_events().filter(event_type="Beat").to_dataframe()
    bpm = beats["bpm"].astype(float)
    onset = beats["start"].astype(float)
    tempo_by_performer[key.removeprefix("perf:")] = {
        "beats": len(beats),
        "median BPM": round(bpm.median(), 1),
        "span (s)": round(onset.max() - onset.min(), 1),
    }

pd.DataFrame(tempo_by_performer).T

# %% [markdown]
# These profiles are measured features carried on the performance timelines, not
# anything TimeToAlign! inferred. Loaded next to the notes and the cross-group
# {{< glossary MatchClaim >}}s, they let one bundle hold both *what* was played
# (the aligned notes) and *how* it was played (the measured tempo and dynamics),
# for every one of the five recordings.

# %% [markdown]
# ## Recap
#
# | What the bundle expresses | How |
# |---|---|
# | Score, two logical units | `score:clt_1` (quarters) + `score:dlt_1` (divs), one divs→quarters `LinearMap` |
# | Performance, two physical units | `perf:<key>:cpt_1` (s) + `perf:<key>:dpt_1` (samples), one `SamplesToSeconds` |
# | Score ↔ performance | a synchronous {{< glossary MatchClaim >}} per match; a {{< glossary NOMATCH >}} per gap |
# | A shared position read everywhere | `bundle.get_matchstamp_at(coord, "score:clt_1")` |
# | Measured tempo and dynamics | `Beat` / `Dynamics` events on `perf:<key>:cpt_1`, read via `filter(event_type=...)` |
#
# Two logical units for the score, two physical units per performance, each pair
# linked by a {{< glossary ConversionMap >}}, the whole tied together by
# cross-group {{< glossary MatchClaim >}}s, and each performance additionally
# carrying its measured per-beat tempo and dynamics as events — one
# {{< glossary AlignmentBundle >}} carrying an existing note alignment of one
# work across five performances, and a faithful record of how each was played.
