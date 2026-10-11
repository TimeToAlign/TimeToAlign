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
# # How to Load an Audio-to-Audio Alignment (Listen Here!)
#
# *Listen Here!* aligns **several recordings of one work** to each other and
# exports the result as a single JSON file. Each recording is warped onto a
# shared, equidistant **reference grid**: per recording, the file stores a
# `times` array whose `i`-th entry is that recording's clock-time (in seconds)
# at reference-grid column `i`. Because every recording is sampled against the
# *same* grid, the arrays are parallel and equal length — together they form a
# dense alignment matrix of shape `(recordings × grid columns)`. This guide
# loads such an export into a single audio-to-audio
# {{< glossary AlignmentBundle >}} with `ListenHereLoader`.
#
# This is a different shape of data from the score↔performance loaders. There,
# each file relates **one** score to **one** performance, and a
# {{< glossary MatchGraph >}} only emerges once several such bundles are
# combined. A Listen Here! file instead encodes the alignment of *all*
# recordings against one another **directly** — one file already carries the
# whole graph. The natural reading is a **complete pairwise topology**: at every
# grid column, every unordered pair of recordings is related by a synchronous
# instant {{< glossary MatchClaim >}}.
#
# We **load an existing alignment**; nothing here runs an aligner. The warp was
# computed once, offline, and written to disk; the loader reads it faithfully.
#
# The arc:
#
# 1. Load the export in one call and read the reference recording.
# 2. Reach the columnar {{< glossary MatchClaimField >}} through the uniform
#    `get_field` API, and inspect a single claim.
# 3. Build the {{< glossary AlignmentBundle >}}.
# 4. The headline: place a point on one recording and read it on all the others.
# 5. A note on scale — why the claims live in a column, not a million objects.

# %% [markdown]
# ## Setup
#
# The example data is a real *Listen Here!* alignment export of six recordings
# of the whole of Beethoven's *Eroica Variations*, Op. 35 — six readings of the
# entire work warped onto one reference grid. The file is large (about 14 MB of
# parallel onset arrays), so it is read directly from its place on disk rather
# than fetched through the test-data helper. The cell below walks up from the
# notebook to the directory that holds the export.

# %%
from __future__ import annotations

import os
from pathlib import Path

from timetoalign.alignment.claims import MatchClaim
from timetoalign.loader.alignment import ListenHereLoader

# Locate the local Listen Here! alignment export by walking up the directory
# tree until an ancestor contains it. This reads a local export file directly;
# it is not a packaged test corpus.
_EXPORT_REL = Path("beethoven_eroica_variations_op35/variation_14/mdw/alignment.json")
_search_roots = [Path.cwd(), *Path.cwd().parents]
if "__file__" in globals():
    _search_roots += list(Path(__file__).resolve().parents)
_alignment_json = next(
    candidate for base in _search_roots if (candidate := base / _EXPORT_REL).exists()
)

# %% [markdown]
# ## 1. Load the export in one call
#
# `ListenHereLoader` parses the single alignment JSON file: it reads each
# recording's `times` array, checks that they all index the same reference grid
# (equal length), and assembles the complete pairwise claim set. `from_file()`
# is the one-line form of the standard two-phase loader pattern.

# %%
loader = ListenHereLoader.from_file(_alignment_json)
loader

# %% [markdown]
# The loader names the **reference recording** — the recording whose clock
# defines the grid origin. Reading the reference matters because it is the
# anchor against which every other recording was warped; it is, however, *just
# another recording* in the bundle, not a privileged hub. Every recording is
# related to every other directly, so the bundle can be read from any of them.

# %%
loader.reference

# %% [markdown]
# Six recordings, reported as sorted stems:

# %%
loader.recording_keys

# %% [markdown]
# ---
#
# ## 2. The whole pairwise claim set, in a column
#
# The alignment is reached through the uniform field API:
# `loader.get_field(MatchClaim)` returns a {{< glossary MatchClaimField >}} — a
# columnar, PyArrow-backed store that holds the entire set of pairwise
# {{< glossary MatchClaim >}}s as Arrow columns and materialises individual
# `MatchClaim` objects only on demand. Its length is the **complete topology**:
# for `R` recordings and `N` grid columns, every unordered pair at every column,
# i.e. `C(R, 2) × N` claims. Six recordings give `C(6, 2) = 15` pairs, and this
# 25-minute work is sampled at 76 376 grid columns — so the field holds well
# over a million claims:

# %%
field = loader.get_field(MatchClaim)

{
    "field type": type(field).__name__,
    "claims": len(field),
    "C(6,2) pairs": 15,
    "grid columns": len(field) // 15,
}

# %% [markdown]
# The field is held columnar precisely so that this many claims need not be a
# million Python objects. Indexing it materialises **one** {{< glossary MatchClaim >}}
# on demand — an ordinary pairwise claim relating two recordings at one grid
# column, carrying the shared provenance the export recorded (Listen Here!'s
# chroma-feature DTW). We look at a single claim to see the shape; we do **not**
# iterate or materialise the whole field.

# %%
field[0]

# %% [markdown]
# ---
#
# ## 3. Build the bundle
#
# `create_bundle()` assembles the {{< glossary AlignmentBundle >}}: one samples
# {{< glossary Timeline >}} per recording, each in its own group, and the
# complete pairwise claim set tying every recording to every other. Audio is
# carried on a {{< glossary Discrete >}} samples axis because a sample index is
# what a recording actually counts; each recording's verbatim JSON seconds are
# preserved beside it as a field on the grid-column events. All the alignment
# lives in the cross-group claim field, which the bundle keeps columnar rather
# than exploding into a million claim objects.

# %%
bundle = loader.create_bundle()

{
    "timelines": bundle.n_timelines,
    "groups": bundle.n_groups,
}

# %% [markdown]
# The bundle's diagram confirms the shape: six single-timeline groups and the
# full claim count, read straight from the columnar field (no claim is
# materialised to count it):

# %%
print(bundle.diagram())

# %% [markdown]
# ---
#
# ## 4. Place a point on one recording, read it on all the others
#
# Here is the promise that makes audio-to-audio alignment compelling: **place a
# point on one recording, and it is instantly placed on all the others.**
# `get_matchstamp_at` takes a coordinate on any one recording's timeline and
# returns the corresponding coordinate on *every* recording connected to it.
#
# The query coordinate must land on an exact grid column carried by the field,
# so we take a real one from the data rather than inventing a value: a claim
# from roughly the middle of the work, read off the reference recording's clock.

# %%
reference_uid = f"{os.path.splitext(loader.reference)[0]}:dpt_1"

reference_claims = field.connecting(reference_uid)
mid_claim = reference_claims[len(reference_claims) // 2]
query_coord = mid_claim.start_anchor.get_coordinate_for(reference_uid)

query_coord

# %% [markdown]
# That instant — about twelve and a half minutes into the reference reading —
# resolves to a {{< glossary MatchStamp >}} spanning all six recordings at once:

# %%
stamp = bundle.get_matchstamp_at(query_coord, reference_uid)
stamp

# %% [markdown]
# Read across that {{< glossary MatchStamp >}}: the same musical instant the
# reference reaches at this sample falls at a different sample in each of the
# other five recordings — a quicker reading reaches it earlier, a steadier one
# later. Every reading is an integer, because a samples axis is
# {{< glossary Discrete >}} and no sample index is ever half-counted. One
# coordinate, placed once, located in all six recordings in a single query:

# %%
{
    "timelines in the stamp": stamp.n_timelines,
    "samples per recording": {
        tl_id: stamp.get_coordinate_for(tl_id, format="int")
        for tl_id in sorted(stamp.coordinates)
    },
}

# %% [markdown]
# Because the topology is complete, the same query works from *any* recording as
# the anchor, not only the reference — every pair is directly related, so no
# recording is a required hub.

# %% [markdown]
# ---
#
# ## 5. A note on scale
#
# This single file holds **more than a million** pairwise claims: six recordings
# give `C(6, 2) = 15` pairs, and a 25-minute work sampled at 50 Hz runs to tens
# of thousands of grid columns. A whole-work export with more recordings would
# be larger still.
#
# This is exactly why the claims live in a {{< glossary MatchClaimField >}}
# rather than as a million individual `MatchClaim` objects. The field stores the
# whole set as Arrow columns (the two timeline ids dictionary-encoded, each
# coordinate as a struct of value, numerator, denominator and unit) and the
# loader builds it **vectorised** — never
# constructing a Python claim per row. A `MatchClaim` is materialised only when
# a single row is indexed. `get_matchstamp_at` likewise filters the column
# vectorised and materialises only the handful of claims at the queried
# coordinate. The columnar store is what lets one `AlignmentBundle` hold a
# whole-work, every-pair audio-to-audio graph without strain.

# %% [markdown]
# ## Recap
#
# | What the bundle expresses | How |
# |---|---|
# | One recording per group | `<stem>:dpt_1` samples timeline, one point event per grid column, recording-length |
# | The whole pairwise alignment | a columnar {{< glossary MatchClaimField >}} via `get_field(MatchClaim)` |
# | One claim, materialised | `field[i]` → a synchronous instant {{< glossary MatchClaim >}} |
# | The reference recording | `loader.reference` — the grid origin, but just another recording |
# | A point placed everywhere | `bundle.get_matchstamp_at(coord, "<stem>:dpt_1")` → every recording at once |
#
# A single file encodes an entire audio-to-audio {{< glossary MatchGraph >}} —
# every recording of one work warped onto a shared reference grid, every pair
# directly related. `ListenHereLoader` reads it into one
# {{< glossary AlignmentBundle >}} in which a coordinate placed on any one
# recording resolves, in a single query, across all the others; and the dense
# claim set that makes that possible is held in a column, not a million objects.
