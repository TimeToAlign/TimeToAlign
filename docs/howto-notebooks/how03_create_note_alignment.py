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
# # How to Create a Note Alignment
#
# This notebook demonstrates the essential pattern for aligning a performance
# with a score using {{< glossary MatchClaim >}} objects in an
# {{< glossary AlignmentBundle >}}.
#
# **What you will learn:**
#
# 1. Match performance notes to score notes by shared attributes (pitch, staff),
#    stating each matched pair by its two positions
# 2. Create an `AlignmentBundle` with performance and score groups
# 3. Query coordinates across both using `get_matchstamp_at()`
# 4. Create `MatchLine` objects from both directions
# 5. Export a `MatchLine` to the Vienna `.match` format
# 6. Opt into converted coordinates in a MatchStamp with `conversion_maps=True`

# %% [markdown]
# ## TL;DR
#
# ```python
# result = match_notes_by_attributes(
#     perf_df, score_df, ["pitch", "staff"], ...,
#     source_timeline=performance, target_timeline=score,
# )
# bundle.add_match_claims(result.match_claims)
# stamp = bundle.get_matchstamp_at(78.0, "clt_1")  # quarterbeat 78 -> seconds
# ```

# %% [markdown]
# ## 1. Setup

# %%
import tempfile

import pandas as pd

from timetoalign import ContinuousLogicalTimeline, Ms3Loader, TimelineGroup, TimeUnit
from timetoalign.alignment import AlignmentBundle, MatchLine
from timetoalign.alignment.match_format import MatchFileContext
from timetoalign.alignment.matching import (
    match_notes_by_attributes,
    prepare_abc_notes_for_matching,
    prepare_eep_notes_for_matching,
)
from timetoalign.loader.physical.eep_notes import EepNotesLoader
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("score") / "beethoven_op18-4iv_multimodal"
NORMAL_DIR = DATA_DIR / "StringQuartetEEP_I_Normal"
ABC_DIR = DATA_DIR / "ABC"

# %% [markdown]
# ## 2. Load the Performance and the Score
#
# The performance notes come from `.notes` files (EEP format with timestamps
# in seconds); the loader turns them into the performance timeline. The score
# is the ABC edition played through. The loader reads the folded score with
# its repeats, and `get_flow()` plays them: the played-through score is as
# long as that flow. That is the axis on which the edition's unfolded TSV of
# score notes counts its `quarterbeats_playthrough`; the matching below places
# its score positions on a timeline of that length.

# %%
# Performance notes and timeline from the Normal recording
eep_loader = EepNotesLoader()
eep_loader.load(*sorted(NORMAL_DIR.glob("*_align_*.notes")))
eep_df = eep_loader.events.to_dataframe()
perf_tl = eep_loader.create_timeline(uid="cpt_1")

# The folded score, and the score played through (quarterbeats)
score_loader = Ms3Loader.from_file(
    ABC_DIR / "n04op18-4_04.notes.tsv",
    ABC_DIR / "n04op18-4_04.measures.tsv",
)
folded_score = score_loader.create_timeline(uid="abc_folded")
clt1 = ContinuousLogicalTimeline(
    length=folded_score.get_flow().length,
    uid="clt_1",
    name="ABC edition, played through",
)

# Score notes from the unfolded ABC edition
abc_df = pd.read_csv(ABC_DIR / "n04op18-4_04_unfolded.notes.tsv", sep="\t")

{
    "EEP notes": len(eep_df),
    "ABC notes": len(abc_df),
    "performance length": perf_tl.length,
    "folded score length": folded_score.length,
    "played-through score length": clt1.length,
}

# %% [markdown]
# ## 3. Prepare & Match Notes
#
# Before matching, we drop the rests of the performance notes and explode
# their chords into individual pitches. Of the score notes we keep those that
# sound anew: a note without a tie, or the first note of a tie. The notes a
# tie continues or ends prolong a sound already started, so a tied sound is
# matched at its first note.

# %%
eep_prepared = prepare_eep_notes_for_matching(eep_df)
abc_prepared = prepare_abc_notes_for_matching(abc_df)

{"EEP prepared": len(eep_prepared), "ABC prepared": len(abc_prepared)}

# %% [markdown]
# Now match by pitch name and staff number. A {{< glossary MatchClaim >}}
# joins two events, but neither table names its events uniquely: the four EEP
# files reuse their note IDs, one file per instrument, and a played-through
# score repeats a passage's notes under the same IDs. So both sides are given
# as positions (`source_coord_column`, `target_coord_column`, each on its
# timeline's axis). A synchronous claim between two positions names no event:
# each side is the sentinel CHUNK, whose coordinate the claim's anchor states,
# and no event is added to either timeline. The matcher returns a
# `MatchResult` containing the matched pairs and their `MatchClaim` objects.

# %%
match_result = match_notes_by_attributes(
    eep_prepared,
    abc_prepared,
    match_columns=["pitch", "staff"],
    source_coord_column="start",
    target_coord_column="quarterbeats_playthrough",
    source_timeline=perf_tl,  # performance timeline (seconds)
    target_timeline=clt1,  # score timeline (quarterbeats)
)

match_result.summary()

# %%
# The first claim, its CHUNK sides, and the Projections the matching added
first_claim = match_result.match_claims[0]
{
    "first claim": first_claim,
    "its CHUNK sides": first_claim.chunk_sides,
    "performance Projections": len(perf_tl.get_events(event_type="Projection")),
    "score Projections": len(clt1.get_events(event_type="Projection")),
}

# %% [markdown]
# ## 4. Create AlignmentBundle
#
# We need two {{< glossary TimelineGroup >}} objects: one for the performance,
# one for the score. The `AlignmentBundle` holds both and manages cross-group
# connections via {{< glossary MatchClaim >}} objects.

# %%
perf_group = TimelineGroup(
    id="performance",
    name="Normal Recording",
    timelines=[perf_tl],
)
perf_group

# %%
score_group = TimelineGroup(
    id="score",
    name="ABC Score",
    timelines=[clt1],
)
score_group

# %%
# Create the bundle and add the match claims; the bundle gives them their ids
bundle = AlignmentBundle(name="Beethoven Op.18/4 — Simple Alignment")
bundle.add_group(perf_group)
bundle.add_group(score_group)
bundle.add_match_claims(match_result.match_claims)

bundle

# %%
# Display an example MatchClaim (shows event IDs, timelines, coordinates)
match_result.match_claims[0]

# %% [markdown]
# ## 5. Query Coordinates via MatchStamp
#
# The `get_matchstamp_at()` method is the primary interface for cross-group
# coordinate transfer. Given a coordinate on one timeline, it returns
# the corresponding coordinates on all connected timelines.

# %%
# Query from the score side: quarterbeat 78 (a matched note onset)
stamp = bundle.get_matchstamp_at(78.0, "clt_1")
stamp

# %%
# The stamp shows coordinates on both timelines
{
    "score_qb": stamp.get_coordinate("clt_1"),
    "perf_seconds": stamp.get_coordinate("cpt_1"),
}

# %% [markdown]
# ### Reverse lookup: performance to score
#
# We can also query from the performance side. The claims store performance
# coordinates in seconds (native EEP format).

# %%
# Find the score position for a performance coordinate (~100 seconds)
# Using 100.3583 which is an exact matched coordinate
stamp_rev = bundle.get_matchstamp_at(100.3583, "cpt_1")

{
    "perf_seconds": stamp_rev.get_coordinate("cpt_1"),
    "score_qb": stamp_rev.get_coordinate("clt_1"),
}

# %% [markdown]
# ## 6. Create MatchLines
#
# A {{< glossary MatchLine >}} is an ordered sequence of coordinate pairs
# for a given source timeline. It is the input for WarpMap generation.
#
# The **direction matters**: the source timeline determines the ordering.

# %%
# Performance-to-score: source is performance, sorted by performance time
perf_to_score = MatchLine.from_claims(
    match_result.match_claims,
    source_timeline_id="cpt_1",
)
perf_to_score

# %%
# Score-to-performance: source is score, sorted by score position
score_to_perf = MatchLine.from_claims(
    match_result.match_claims,
    source_timeline_id="clt_1",
)
score_to_perf

# %% [markdown]
# **When to use which direction:**
#
# - `perf_to_score`: Use when you have a performance coordinate and want to
#   find the corresponding score position. Sorted by performance time.
# - `score_to_perf`: Use when you have a score coordinate and want to find
#   the corresponding performance time. Sorted by score position.
#
# Both contain the same number of stamps (one per matched note), but the
# ordering and lookup direction differ.

# %%
# Extract typed anchors — each carries both timeline IDs and both coordinates
anchors = score_to_perf.get_alignment_anchors("cpt_1")

{
    "n_anchors": len(anchors),
    "first_anchor": anchors[0],
    "last_anchor": anchors[-1],
}

# %% [markdown]
# ## 7. Export to .match Format
#
# A {{< glossary MatchLine >}} can be exported to the Vienna `.match` file
# format using `save_as()`.  The `.match` format is the standard interchange
# format for note-level alignments in MIR.
#
# To produce a rich `.match` file (with real pitch, duration, and staff
# data rather than placeholders), supply a `MatchFileContext` built from
# the same DataFrames used for matching.

# %%
ctx = MatchFileContext.from_dataframes(
    score_df=abc_prepared,
    perf_df=eep_prepared,
    match_result=match_result,
    piece="Beethoven Op.18/4-iv",
    composer="Ludwig van Beethoven",
    performer="StringQuartetEEP Normal",
)

with tempfile.TemporaryDirectory() as tmp:
    out_path = score_to_perf.save_as(f"{tmp}/alignment.match", context=ctx)
    text = out_path.read_text()

# Show the first 15 lines
for line in text.splitlines()[:15]:
    print(line)

# %% [markdown]
# The exported file is a valid `.match` file that can be loaded back with
# `MatchfileLoader` or any tool that reads the Vienna format.

# %% [markdown]
# ## 8. Conversion Maps in MatchStamp Display
#
# A conversion map can be added to any timeline at any time; loaders attach
# their own. Either way, `get_matchstamp_at()` leaves them out of the
# {{< glossary MatchStamp >}} by default — `conversion_maps` is opt-in.

# %%
# Add a conversion map to the performance timeline.
from timetoalign.maps import ScalarMap

perf_tl.add_conversion_map(
    ScalarMap(
        scalar=1000, source_unit=TimeUnit.seconds, target_unit=TimeUnit.milliseconds
    )
)

# %%
# Default: conversion_maps=False, so none of the enabled maps show up.
bundle.get_matchstamp_at(78.0, "clt_1")

# %%
# conversion_maps=True adds the enabled maps, each evaluated at its own
# timeline's coordinate: milliseconds from the map just added. The
# played-through score was built by hand and holds no conversion map.
bundle.get_matchstamp_at(78.0, "clt_1", conversion_maps=True)

# %% [markdown]
# The same flag adds a derived column to `get_matchstamp_table()` — one
# column per (timeline, enabled numeric-unit map), after the timeline
# columns. A label-valued map would still show in the stamp display above
# but never as a table column.

# %%
bundle.get_matchstamp_table([78.0], "clt_1", conversion_maps=True).column_names

# %% [markdown]
# ## Summary
#
# > *"MatchClaims connect timelines across groups. The AlignmentBundle
# > manages these connections and provides coordinate transfer via
# > MatchStamps. MatchLines order these stamps for WarpMap generation."*
#
# | Pattern | API |
# |---------|-----|
# | Match notes by attributes | `match_notes_by_attributes(..., source_timeline=, target_timeline=)` |
# | Add claims to bundle | `bundle.add_match_claims(claims)` |
# | Query by coordinate | `bundle.get_matchstamp_at(coord, tl_id)` |
# | Create MatchLine | `MatchLine.from_claims(claims, source_timeline_id)` |
# | Export to `.match` | `matchline.save_as("out.match", context=ctx)` |
# | Reveal converted coordinates | `bundle.get_matchstamp_at(coord, tl_id, conversion_maps=True)` |

# %%
