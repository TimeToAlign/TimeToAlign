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
# # How to Build Beat Grids
#
# You have a recording, you know its tempo, and you know the second at which a
# beat sounds. You want the beat and measure times, and you want them in a file
# an annotation program can open.
#
# A {{< glossary BeatGrid >}} is the arithmetic for that. It stores only what a
# source states — where an anchor beat sits, how fast the music runs, how beats
# group into bars, and which beat of a bar the anchor is — and generates every
# beat from those facts. It is not a {{< glossary Timeline >}}: nothing is
# stored per beat, so a grid can cover a whole DJ set as cheaply as a bar.

# %%
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from timetoalign import BeatGrid, BeatGridSegment, BeatPolicy, policy_for_metro

# %% [markdown]
# ## One tempo statement, one grid
#
# `BeatGrid.from_tempo()` states a tempo in beats per minute, the meter as the
# source spells it, the second at which the anchor beat sounds, and which beat
# of the bar that anchor is. `extent` bounds the grid — for a recording, its
# duration.
#
# Positions are read **exactly**. A decimal string spells the decimal it looks
# like; a float contributes the exact binary value it holds, which is a
# different and much longer rational. Pass a string or a `Fraction` wherever
# the decimal is the fact.

# %%
grid = BeatGrid.from_tempo(160, metro="4/4", start="0.092", extent="212.48")
grid

# %% [markdown]
# The grid holds one segment: the four stated facts, plus the `end` the grid
# gave it from `extent`.

# %%
segment = grid.segments[0]
segment

# %% [markdown]
# Beat length, bar length and the first downbeat are **derived** from those
# facts rather than stored beside them, so they cannot fall out of step with
# the tempo. `metro="4/4"` reads as four beats of one quarter each; the grid's
# lattice ticks once per counted value, so `"6/8"` would be six beats of an
# eighth, not two dotted ones.
#
# The derived values come back as exact rationals — `23/250` *is* 0.092, and
# `3/8` of a second *is* one beat at 160 BPM.

# %%
{
    "beat length (s)": segment.beat_seconds,
    "bar length (s)": segment.bar_seconds,
    "quarters per second": segment.quarters_per_second,
    "first downbeat (s)": segment.first_downbeat,
    "measures in the grid": grid.n_measures,
}

# %% [markdown]
# ## The beat table
#
# `get_beat_table()` renders every beat the grid generates, one row each. It is
# the only table the grid builds; the export formats below are renderings of
# it.

# %%
beats = grid.get_beat_table()
beats.head(8)

# %% [markdown]
# `seconds` is the beat's instant, `segment_seconds` the same instant measured
# from the start of its segment, `measure` and `beat` its label. **Downbeats
# are the rows whose `beat` is 1** — there is no separate downbeat lane,
# because there is nothing a second one could know.

# %%
downbeats = beats[beats["beat"] == 1]
downbeats.head(4)

# %%
{
    "beats": len(beats),
    "downbeats": len(downbeats),
    "measures the grid counts": grid.n_measures,
}

# %% [markdown]
# ## Asking in both directions
#
# `seconds_at(measure, beat)` goes from a label to an instant and returns a
# {{< glossary Coordinate >}} in seconds. A fractional beat interpolates
# between the two beats around it.

# %%
{
    "measure 2, beat 1": grid.seconds_at(2),
    "measure 2, beat 3": grid.seconds_at(2, 3),
    "measure 2, beat 2 1/2": grid.seconds_at(2, Fraction(5, 2)),
}

# %% [markdown]
# `position_at(seconds)` goes the other way and returns a `GridBeat`: the beat
# whose span contains the position, that is the last beat at or before it. It
# holds the exact ratio the grid computed on `seconds`.

# %%
here = grid.position_at(60)
here

# %%
{
    "queried second": 60,
    "beat sounding there": here.instant,
    "measure": here.measure,
    "beat": here.beat,
    "opens a measure": here.is_downbeat,
    "label back to seconds": grid.seconds_at(here.measure, here.beat),
    "the two directions agree": here.instant
    == grid.seconds_at(here.measure, here.beat),
}

# %% [markdown]
# The label round-trips: `position_at` names a beat, `seconds_at` on that
# beat's label returns the same {{< glossary Coordinate >}}, and the two
# compare equal. A beat keeps both readings and they are for different jobs:
# `instant` is the published seconds coordinate you display and compare, while
# `seconds` is the exact `Fraction` the grid computed with — reach for it when
# a position feeds further arithmetic.
#
# `quarters_between()` answers the notated question instead of the clock one —
# how many quarter notes sound between two instants:

# %%
grid.quarters_between("0.092", "212.48")

# %% [markdown]
# ## Counting the bar differently
#
# The grid counts in its own lattice. Pass a {{< glossary BeatPolicy >}} to
# read the same instant under another counting: the beat index becomes a
# quarter-note offset from the downbeat, converted back through the segments'
# tempi. The measure numbering is unaffected.
#
# A 4/4 lattice beat is one quarter, so the grid's beat 3 sits two quarters
# after the downbeat — which a counting in eighths calls beat 5. Both spellings
# name the same instant.
#
# `BeatPolicy.uniform(division, count)` counts `count` beats of `division`
# quarters each; the name is for display.

# %%
eighths = BeatPolicy.uniform(Fraction(1, 2), 8, name="eighths")
{
    "grid's own beat index at 60 s": grid.position_at(60).beat,
    "same instant counted in eighths": grid.position_at(60, policy=eighths).beat,
    "measure 2, beat 3 of the grid": grid.seconds_at(2, 3),
    "measure 2, eighth 5": grid.seconds_at(2, 5, policy=eighths),
}

# %% [markdown]
# ## A tempo change mid-recording
#
# One segment states one tempo. A recording that changes tempo — a live set, a
# DJ transition, an analyst re-anchoring a drifting grid — is several segments
# in **one** grid. Build each `BeatGridSegment` from what that stretch states
# and hand them to `BeatGrid` together.
#
# Each segment carries the policy its own stretch is counted under.
# `policy_for_metro()` builds one from a meter string, applying the same
# lattice reading `from_tempo(metro=…)` does. (`BeatPolicy.from_time_signature()`
# makes the other choice, reading `6/8` as two dotted beats; that is notated
# meter, not a grid's lattice.)

# %%
four_four = policy_for_metro("4/4")
opening = BeatGridSegment(start=0, bpm=120, policy=four_four, battito=1)
faster = BeatGridSegment(start=30, bpm=128, policy=four_four, battito=3)

live = BeatGrid([opening, faster], extent=60)
live

# %% [markdown]
# You state segments without an end; the grid orders them and bounds each one
# by the next one's start, the last by `extent`.

# %%
live.segments

# %% [markdown]
# ### Two readings of one lattice
#
# `numbering="set"` counts measures across the whole grid; `numbering="segment"`
# restarts the count at each segment. They are two labellings of the same
# beats, so they can be read side by side.

# %%
whole_grid = live.get_beat_table()
per_segment = live.get_beat_table(numbering="segment")
both = whole_grid.assign(segment_measure=per_segment["measure"])
both[(both["seconds"] >= 28.5) & (both["seconds"] <= 33)]

# %% [markdown]
# The second segment anchors at 30 s on **beat 3**. Under `"set"` numbering
# those two beats finish measure 15, the measure the first segment opened —
# the count never restarts. Under `"segment"` numbering they are that segment's
# measure **0**, the partial bar before its own first downbeat, and measure 1
# opens at 30.9375 s.
#
# A measure interrupted this way states some beat indices twice, once on either
# side of the re-anchor: measure 15 sounds a beat 3 at 29 s under the first
# segment and another at 30 s under the second. That label therefore names no
# single instant, and `seconds_at` refuses it rather than picking one — it
# names the candidates instead, so you can choose which count you meant.

# %%
try:
    live.seconds_at(15, 3)
except ValueError as error:
    ambiguous = str(error)
ambiguous

# %% [markdown]
# Read such a stretch from the table, where both rows are visible. The
# unambiguous labels around it answer as usual, and `segment=` restricts the
# table to one segment's beats:

# %%
live.get_beat_table(segment=1, numbering="segment").head(6)

# %% [markdown]
# ### Where one segment stops and the next begins
#
# `position_at` reports which segment is sounding, and `segment_seconds_at`
# gives an instant relative to its segment's start rather than to the
# recording:

# %%
{
    "just before the change": live.position_at(29.9),
    "at the change": live.position_at(30),
    "segment sounding at 45 s": live.segment_at(45),
    "measure 20 in the recording": live.seconds_at(20),
    "measure 20 within its segment": live.segment_seconds_at(20),
}

# %% [markdown]
# `quarters_between()` integrates each segment's own tempo rather than
# averaging them, so it stays exact across the change. The second before the
# change carries 2 quarters at 120 BPM and the second after carries 32/15 at
# 128 BPM — a total no single tempo produces:

# %%
{
    "29 s to 31 s, across the change": live.quarters_between(29, 31),
    "29 s to 30 s, at 120 BPM": live.quarters_between(29, 30),
    "30 s to 31 s, at 128 BPM": live.quarters_between(30, 31),
}

# %% [markdown]
# A generated beat that falls within **half a beat** before the next segment's
# start is that segment's anchor beat displaced, not a beat of its own, so the
# grid drops it. Exports that round an anchor to milliseconds would otherwise
# produce a phantom bar at every re-anchor. At exactly half a beat the beat is
# kept, and the rule never applies at the end of the grid.

# %%
nudged = BeatGridSegment(start="30.02", bpm=120, policy=four_four, battito=1)
snapped = BeatGrid([opening, nudged], extent=32).get_beat_table()
snapped[(snapped["seconds"] >= 29) & (snapped["seconds"] <= 31)]

# %% [markdown]
# The beat the first segment would have generated at 30.0 s is 0.02 s before
# the re-anchor, well within half of its 0.5 s beat, so measure 16 opens at
# 30.02 s and no bar is invented.

# %% [markdown]
# ## Exporting to an annotation program
#
# `export_to_csv()` writes the beat table in the shape a program expects and
# returns the number of rows written.
#
# | `format` | Opens in | Columns |
# |---|---|---|
# | `"sonic_visualiser"` | Sonic Visualiser, Audacity | `TIME`, `LABEL` |
# | `"tilia"` | TiLiA | `time`, `measure`, `beat`, `is_first_in_measure` |
#
# For the label track, `labels=` chooses what gets marked: every beat
# (`"beats"`, the default, labelled `M1B2`), only the downbeats (`"measures"`,
# labelled `M1`), or `"both"`.

# %%
with TemporaryDirectory() as export_dir:
    label_track = Path(export_dir) / "live_beats.csv"
    downbeat_track = Path(export_dir) / "live_downbeats.csv"
    tilia_track = Path(export_dir) / "live_tilia.csv"

    rows_written = {
        "every beat": live.export_to_csv(label_track, format="sonic_visualiser"),
        "downbeats only": live.export_to_csv(
            downbeat_track, format="sonic_visualiser", labels="measures"
        ),
        "TiLiA beat track": live.export_to_csv(tilia_track, format="tilia"),
    }
    label_lines = label_track.read_text(encoding="utf-8").splitlines()[:4]
    downbeat_lines = downbeat_track.read_text(encoding="utf-8").splitlines()[:4]
    tilia_lines = tilia_track.read_text(encoding="utf-8").splitlines()[:4]

{
    "rows written": rows_written,
    "label track": label_lines,
    "downbeat track": downbeat_lines,
    "TiLiA track": tilia_lines,
}

# %% [markdown]
# The files are written for real and read back here; the temporary directory
# leaves nothing behind.

# %% [markdown]
# ## A grid without an extent
#
# `extent` is optional. Left out, the grid generates beats without end — the
# case for a live feed, or for a tempo you want to project past the material
# you have. Consume it with `iter_beats(stop=…)`.

# %%
open_ended = BeatGrid.from_tempo(120, metro="4/4")
{
    "extent": open_ended.extent,
    "first six beats (s)": [beat.seconds for beat in open_ended.iter_beats(stop=3)],
}

# %% [markdown]
# Questions that need every beat — `get_beat_table()` and `n_measures` — cannot
# be answered without an end, and say so rather than guessing one:

# %%
try:
    open_ended.get_beat_table()
except ValueError as error:
    message = str(error)
message

# %% [markdown]
# ## Summary
#
# | Task | Call |
# |---|---|
# | Grid from one tempo statement | `BeatGrid.from_tempo(bpm, metro=…, start=…, extent=…)` |
# | Grid from several tempo statements | `BeatGrid([segment, …], extent=…)` |
# | One stated stretch | `BeatGridSegment(start=…, bpm=…, policy=…, battito=…)` |
# | A policy for the meter a source spells | `policy_for_metro("4/4")` |
# | Every beat as a table | `grid.get_beat_table(segment=…, numbering=…)` |
# | Downbeats | the table's rows where `beat == 1` |
# | Label to instant | `grid.seconds_at(measure, beat)` |
# | Label to instant within its segment | `grid.segment_seconds_at(measure, beat)` |
# | Instant to label | `grid.position_at(seconds)` |
# | Notated distance across tempo changes | `grid.quarters_between(a, b)` |
# | Another counting of the bar | the `policy=` argument, a `BeatPolicy` |
# | Beats of an unbounded grid | `grid.iter_beats(stop=…)` |
# | Annotation file | `grid.export_to_csv(path, format=…, labels=…)` |
#
# A grid does not have to be written by hand. Where a loader reads tempo
# statements out of a file, the `TimeSkeleton` it builds for that track carries
# the grid its measures came from, and `skeleton.create_beatgrid()` returns
# that grid — the same object, with the same surface as everything above.
