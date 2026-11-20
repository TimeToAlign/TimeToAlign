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
# # How to Load the Vienna 4x22 Corpus
#
# This guide demonstrates how to load score-to-performance alignment data
# from the **Vienna 4x22 Corpus** using the `MatchfileLoader`.
#
# ## About the Dataset
#
# The full **4x22 Vienna Corpus** (Grachten & Widmer, 2012) contains four
# classical piano pieces, each performed by 22 pianists, yielding 88
# score-to-performance alignments in the `.match` file format. Each `.match`
# file encodes three things simultaneously:
#
# - A **score representation** (note identities, pitch, metrical position,
#   duration in quarter beats).
# - A **performance representation** (MIDI pitch, onset/offset in ticks,
#   velocity).
# - An **alignment** linking each score note to its performed counterpart
#   (or marking it as a deletion when the pianist omitted it).
#
# **What we demonstrate here is a 1x22 sample** — one piece (Chopin Etude
# Op. 10 No. 3 in E major) performed by all 22 pianists. This is one quarter
# of the full dataset, and it is the specimen shipped with the TimeToAlign!
# test suite. The workflow generalises straightforwardly to the remaining
# three pieces.
#
# **What you will learn:**
#
# 1. Load all 22 `.match` files through a single `MatchfileLoader`
# 2. Inspect the resulting timelines and their TimeStamps
# 3. Assemble the `AlignmentBundle` and query its MatchClaims
# 4. Obtain MatchStamps — the cross-timeline coordinate cross-section

# %% [markdown]
# ## Setup

# %%
import warnings

from timetoalign import MatchfileLoader
from timetoalign.alignment import MatchGraph
from timetoalign.testdata import ensure_data

DATA_DIR = ensure_data("vienna_1x22")

match_files = sorted(DATA_DIR.glob("*.match"))
len(match_files)

# %% [markdown]
# 22 `.match` files — one per pianist — all sharing the same score
# (Chopin Op. 10 No. 3).

# %% [markdown]
# ---
#
# ## Step 1: Load All Match Files
#
# The `MatchfileLoader` processes **all** `.match` files for a given piece
# through a single instance. It builds a shared score timeline from the
# first file and verifies each subsequent file against it. Incompatible
# files are rejected with a warning; compatible files contribute their
# performance timeline and match claims.

# %%
loader = MatchfileLoader()
# partitura warns on the corpus's composite quarter durations, which it cannot
# express symbolically. That is a note from the score parser about its own
# symbolic-duration vocabulary, not a rejected file, so it is silenced here.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r"Quarter duration .*",
        category=UserWarning,
        module=r"partitura\.utils\.music",
    )
    loader.load(*match_files)
loader

# %% [markdown]
# All 22 files loaded successfully — zero rejected.

# %% [markdown]
# ---
#
# ## Step 2: Discover and Inspect Timelines
#
# Before assembling the bundle, we can access the timelines directly from
# the loader. This is useful for inspecting what was parsed.
#
# `create_timelines()` returns all timelines as a list (score first):

# %%
all_timelines = loader.create_timelines()
[(tl.id, tl.class_name, tl.n_events) for tl in all_timelines]

# %% [markdown]
# 23 timelines: 1 score + 22 performances. Each performance has a slightly
# different event count because some pianists omit notes (deletions).
#
# Individual timelines are accessed by role shorthand. The stored score UID is
# `score:clt_1`; performance UIDs are `perf:<stem>:dlt_1` (for example,
# `perf:Chopin_op10_no3_p01:dlt_1`). The shorthand remains convenient for
# lookup:

# %%
score = loader.create_timeline(uid="score")
score

# %% [markdown]
# The score is a `ContinuousLogicalTimeline` in quarter-beat coordinates,
# carrying two conversion maps: `raw_quarters` (a ShiftMap for
# anacrusis offset) and `quarters_to_divs` (a ScalarMap to MIDI
# divisions).

# %%
perf_01 = loader.create_timeline(uid="perf:p01")
perf_01

# %% [markdown]
# Each performance is a `DiscreteLogicalTimeline` in MIDI tick coordinates,
# with a `ticks_to_seconds` ScalarMap attached.

# %% [markdown]
# ---
#
# ## Step 3: TimeStamps — the Cross-Section View
#
# A TimeStamp is the **primary interface** for querying what happens at a
# given coordinate on a timeline. It returns the coordinate itself plus all
# conversion map results in a single cross-section.
#
# ### Score TimeStamp
#
# At quarter-beat 10.0, the score timestamp shows the coordinate in
# quarters, plus the raw (un-normalised) partitura value and the MIDI
# divisions equivalent:

# %%
score.get_timestamp(10.0)

# %% [markdown]
# ### Performance TimeStamp
#
# At tick 10000, the performance timestamp shows the tick coordinate plus
# the converted seconds value:

# %%
perf_01.get_timestamp(10000.0)

# %% [markdown]
# ---
#
# ## Step 4: Create the AlignmentBundle
#
# `create_bundle()` assembles an `AlignmentBundle` from the
# loaded data. The score goes into its own group; each performance is a
# standalone timeline; MatchClaims connect them.

# %%
bundle = loader.create_bundle()
bundle

# %% [markdown]
# The diagram shows the score group (with its conversion maps reflected
# in the timeline), 22 standalone performance timelines (with proportional
# bar widths reflecting their different lengths in ticks), and the total
# number of cross-group MatchClaims.

# %% [markdown]
# ---
#
# ## Step 5: Viewing and Querying MatchClaims
#
# MatchClaims are stored on the bundle as `cross_group_claims`. Each claim
# connects a score event to a performance event (synchronous match) or
# records a deletion (non-synchronous NOMATCH).

# %%
claims = bundle.cross_group_claims
len(claims)

# %% [markdown]
# 9,988 claims across 22 performers (22 x 454 snote records per file).
#
# Filtering claims for a specific performer:

# %%
p01_claims = [c for c in claims if c.connects(perf_01.id)]
synch_p01 = [c for c in p01_claims if c.is_synchronous]
nomatch_p01 = [c for c in p01_claims if not c.is_synchronous]

{
    "performer": perf_01.id,
    "total_claims": len(p01_claims),
    "synchronous (matched notes)": len(synch_p01),
    "nomatch (deletions)": len(nomatch_p01),
}

# %% [markdown]
# A single claim looks like this:

# %%
synch_p01[0]

# %% [markdown]
# The claim shows: interval match between score coordinate `[0.0, 0.5]`
# quarters and performance coordinate `[0, 261]` ticks. Both anchors
# (start and end) are present because this is a synchronous interval match.

# %% [markdown]
# ---
#
# ## Step 6: MatchStamps from Individual Claims
#
# A MatchStamp is the cross-timeline analogue of a TimeStamp. Where a
# TimeStamp shows coordinates within *one* timeline (plus C-Map
# conversions), a MatchStamp shows the synchronised coordinate across
# *multiple* timelines linked by MatchClaims.
#
# To obtain a MatchStamp, wrap one or more claims in a `MatchGraph` and
# call `get_stamps()`:

# %%
mg_single = MatchGraph(claims=[synch_p01[0]])
stamps = mg_single.get_stamps()
stamps[0]

# %% [markdown]
# The MatchStamp shows the score coordinate (0.0 quarters) and the
# corresponding performance coordinate (0.0 ticks) — the union of what
# both TimeStamps would show individually.
#
# For an interval claim, there are two stamps (start and end):

# %%
len(stamps)

# %%
stamps[1]

# %% [markdown]
# ---
#
# ## Step 7: MatchGraph Across All Performers
#
# The real power emerges when building a MatchGraph from *all* synchronous
# claims across all 22 performers. Each connected component in the graph
# produces a single MatchStamp spanning every timeline that shares that
# score coordinate.

# %%
all_synch = [c for c in claims if c.is_synchronous]
mg_all = MatchGraph(claims=all_synch)

{
    "synchronous_claims": mg_all.n_claims,
    "graph_nodes": mg_all.n_nodes,
    "graph_edges": mg_all.n_edges,
    "timelines_in_graph": len(mg_all.timeline_ids),
}

# %%
stamps_all = mg_all.get_stamps()
len(stamps_all)

# %% [markdown]
# Each stamp is a full cross-section. The first stamp (at score coordinate
# 0.0) spans all 23 timelines — the score plus all 22 performances that
# have a matched note at that position:

# %%
s0 = stamps_all[0]
s0.n_timelines

# %%
s0

# %% [markdown]
# This is the complete synchronised view: one score coordinate mapped to
# 22 different MIDI tick coordinates, each reflecting the expressive timing
# of a different pianist.

# %% [markdown]
# ---
#
# ## Summary
#
# The complete workflow:
#
# ```python
# from timetoalign import MatchfileLoader
# from timetoalign.alignment import MatchGraph
#
# # Load
# loader = MatchfileLoader()
# loader.load(*sorted(data_dir.glob("*.match")))
#
# # Inspect timelines and their TimeStamps
# score = loader.create_timeline(uid="score")
# score.get_timestamp(10.0)  # quarters + raw + divs
#
# # Assemble the bundle
# bundle = loader.create_bundle()
#
# # Query claims and build MatchStamps
# claims = bundle.cross_group_claims
# mg = MatchGraph(claims=[c for c in claims if c.is_synchronous])
# stamps = mg.get_stamps()  # full cross-section across all timelines
# ```
