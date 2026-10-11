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
# # How to Load an IEEE 1599 Document
#
# **IEEE 1599** is a multi-layer XML standard for encoding a single musical
# work once and relating every representation of it — the score, engraved
# page images, audio recordings, analytical annotations — back to a common
# axis. That axis is the ``<spine>``: a flat, ordered list of abstract
# events in *virtual time units* (VTU), unit-less by design. Every other
# layer states where in **its own** coordinate space one of those spine
# events falls, via an ``event_ref``. The spine is the hub; the layers are
# projections onto it; ``event_ref`` is the correspondence — exactly the
# shape of an alignment.
#
# `Ieee1599Loader` reads one such document into a single multimodal
# {{< glossary AlignmentBundle >}}: the spine as one {{< glossary Timeline >}}
# in ticks, the logically-organised-symbols (LOS) layer of notes, rests and
# lyrics on another, one graphical timeline per engraved edition, one
# physical timeline per audio track, and the whole set of projections as one
# columnar {{< glossary MatchClaimField >}}. The document's own analytical
# layer — a segmentation of the spine resolving to places of a Petri net — is
# carried too, not as timing but as an external reference.
#
# We **load an existing document**; nothing here runs an aligner. The
# correspondences are the document's own ``event_ref`` cross-references; the
# loader's job is to read them faithfully.
#
# The work is Erik Satie's *Gymnopédie No. 1*.
#
# The arc:
#
# 1. Load the document in one call and read its title.
# 2. The spine, in ticks — a cumulative virtual-time axis — and the LOS
#    layer of notes and rests sitting at spine coordinates.
# 3. The graphical editions as nested `SegmentLine`s of pages and accolades,
#    in pixels, and the audio tracks, in seconds.
# 4. The projections, reached through the uniform
#    {{< glossary MatchClaimField >}} API, and the cross-section they
#    describe over spine coordinates.
# 5. The structural layer: a Petri-net analysis carried as external
#    references on the spine.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import pyarrow.compute as pc

from timetoalign import IntervalToConstantMap
from timetoalign.alignment.claims import MatchClaim
from timetoalign.loader.alignment.ieee1599 import Ieee1599Loader
from timetoalign.testdata import ensure_data

base = ensure_data("ieee1599")
document_path = base / "SatiePetriNets" / "ieee1599" / "gymnopedie_01.xml"

# %% [markdown]
# ## 1. Load the document in one call
#
# `Ieee1599Loader` parses the whole document — spine, LOS, every graphical
# edition, every audio track, and the structural analysis — into curated
# tables, then assembles them into timelines and a claim field.
# `from_file()` is the one-line form of the standard two-step loader
# pattern: parse, then build.

# %%
loader = Ieee1599Loader.from_file(document_path)
loader

# %% [markdown]
# The document states its own bibliographic metadata, read straight off
# ``<general><description>``:

# %%
{
    "title": loader.file_metadata.get("title"),
    "work_title": loader.file_metadata.get("work_title"),
    "authors": loader.file_metadata.get("authors"),
}

# %% [markdown]
# `create_bundle()` builds the {{< glossary AlignmentBundle >}}: six
# standalone timelines — the spine, the LOS layer, two engraved editions and
# two audio tracks — tied together by the MatchClaims. Each timeline lives in
# its own {{< glossary TimelineGroup >}}; MatchClaims carry the connections
# between those groups.

# %%
bundle = loader.create_bundle()
print(bundle.diagram())

# %% [markdown]
# ---
#
# ## 2. The spine and the LOS layer
#
# ### The spine: a cumulative virtual-time axis
#
# The spine's ``timing`` attribute is a *relative* integer delta against the
# previous ``<event>``; the loader accumulates it into an absolute
# coordinate, so the stored value is the running sum — the cumulative VTU.
# Events notated as simultaneous carry ``timing="0"`` and land on the same
# coordinate. The spine is a `DiscreteLogicalTimeline` in ``ticks``:

# %%
spine = bundle.get_timeline(loader.spine_uid)
spine

# %%
spine.get_events().head(5)

# %% [markdown]
# ### The LOS layer: notes and rests at spine coordinates
#
# Every LOS event — a `Note`, a `Rest`, or a lyric `Syllable` — sits at the
# VTU coordinate of the spine event its ``event_ref`` names; that reference
# *is* its temporal position; the LOS layer carries no timing of its own.
# It is a second `DiscreteLogicalTimeline`, sharing the spine's unit and
# length but its own event set — 557 LOS events against 382 spine events,
# since a chord's several noteheads and a measure's clefs and key signatures
# do not correspond one-to-one.

# %%
los = bundle.get_timeline(loader.los_uid)
los

# %% [markdown]
# Notated durations are kept as the verbatim ``num``/``den`` pair rather
# than a reduced fraction, so a duration notated ``1/4`` never silently
# becomes something else; the exact value is
# ``Fraction(duration_num, duration_den)``. The opening melodic notes —
# each a quarter note (``duration_num=1``, ``duration_den=4``) — carry their
# pitch as a step, octave and accidental, e.g. the first is F♯6:

# %%
notes = los.get_events(event_type="Note").to_dataframe()
notes.head(5)

# %% [markdown]
# ---
#
# ## 3. Editions in pixels, tracks in seconds
#
# ### Per-edition graphical SegmentLines
#
# Each ``<graphic_instance_group>`` — one engraved edition of the score — is
# its own nested `SegmentLine`, in unit ``pixels``. An edition is printed on
# pages, and a page is one image file with its own pixel origin, so the
# edition's segments are its pages and a page's segments are the accolades
# engraved on it. Within a page, x coordinates zig-zag along the spine: when a
# new system begins beyond half of that page's x-span, the reset starts a new
# accolade. Both gymnopédie editions have four pages carrying 4 + 5 + 5 + 4
# accolades.

# %%
for uid in loader.edition_uids:
    edition = bundle.get_timeline(uid)
    print(f"{edition.id}: {edition.name!r}, {edition.length}")

# %%
edition = bundle.get_timeline(loader.edition_uids[0])
graphical_events = edition.get_events().to_dataframe()
graphical_events.head()

# %% [markdown]
# The edition's segments are its pages; each page's segments are its
# accolades. `iter_segments()` yields both levels in spine order, together
# with each child's offset in its parent's coordinates. A page carries its
# ``<graphic_instance>`` attributes — the image ``file_name`` above all — in
# ``meta["page"]``:

# %%
print(f"{edition.class_name}: {edition.n_segments} pages")
for _, page_offset, page in edition.iter_segments():
    print(f"{page.name} at x={page_offset.value}: {page.meta['page']['file_name']}")
    for _, offset, accolade in page.iter_segments():
        print(f"    {accolade.name} at x={offset.value}, length={accolade.length}")

# %% [markdown]
# An `IntervalToConstantMap` on the edition resolves any edition-wide
# x coordinate to the page-image ``file_name`` that contains it, without
# descending the hierarchy. The first graphical event provides one such
# coordinate:

# %%
page_image_map = next(
    cmap
    for cmap in edition._conversion_maps.values()
    if isinstance(cmap, IntervalToConstantMap)
)
first_graphical_event = graphical_events.iloc[0]
page_image_map(first_graphical_event["start"])

# %% [markdown]
# Graphical geometry is kept as one nested ``bbox`` struct rather than split
# coordinate columns. Its ``ul`` and ``lr`` members preserve the upper-left
# and lower-right pixel coordinates of the graphical event:

# %%
bbox = first_graphical_event["bbox"]
{
    "bbox": bbox,
    "ul": bbox["ul"],
    "lr": bbox["lr"],
}

# %% [markdown]
# ### Per-track audio timelines
#
# Each ``<track>`` — one recording — is its own `ContinuousPhysicalTimeline`
# in ``seconds``. The performers and the media file name the document states
# ride along in ``meta``; the media file itself is never opened:

# %%
for uid in loader.track_uids:
    track = bundle.get_timeline(uid)
    print(f"{track.id}: {track.meta['performers']}, {track.length}")

# %%
track = bundle.get_timeline(loader.track_uids[0])
track.get_events().head(3)

# %% [markdown]
# ---
#
# ## 4. The projections, as one columnar claim field
#
# Every LOS, graphical and audio event contributes one synchronous
# {{< glossary MatchClaim >}} tying its own coordinate to the spine event it
# references. All three layers go into **one** field — the alignment they
# express is one alignment, hub-and-spoke around the spine — reached through
# the uniform field API:

# %%
field = loader.get_field(MatchClaim)

{
    "field type": type(field).__name__,
    "claims": len(field),
}

# %% [markdown]
# The `MatchClaimField` remains columnar; its table has one struct column.
# Showing its head makes that stored representation visible before any
# individual `MatchClaim` is materialised:

# %%
field.table.to_pandas().head()

# %% [markdown]
# Indexing the field materialises one `MatchClaim` on demand. This
# mid-document row is more illustrative than the first row because it shows
# an ordinary in-document correspondence rather than the special opening
# coordinate:

# %%
field[len(field) // 2]

# %% [markdown]
# ### The cross-section over spine coordinates
#
# `get_matchstamp_table(from_graph=True)` collapses the claims into one row
# per connected component of the alignment graph — one row per spine
# coordinate that at least one layer reaches, every participating layer
# filled in the same row. It is the cross-section of all six timelines over
# the spine:

# %%
cross_section = bundle.get_matchstamp_table(from_graph=True, format="dataframe")

{"rows": len(cross_section), "columns": list(cross_section.columns)}

# %%
cross_section.head(5)

# %% [markdown]
# ---
#
# ## 5. The structural layer: an analysis resolved to a Petri net
#
# An ``<analysis>`` partitions the spine into ``<segment>`` elements, each
# listing the spine events it covers; a sibling ``<petri_nets>`` block names
# ``.pnml`` files and binds one place of one net to one segment. That is a
# reference *into* an external resource, not a timing statement, so it is
# carried as `spine.external_references` — one row per
# ``(segment_event, place)`` pair — rather than as events or claims:

# %%
refs = spine.external_references
refs.num_rows

# %% [markdown]
# Filtering to one segment shows the resolution end to end: every spine
# event of segment ``Analisi_1_L1_A`` resolves to place ``p2`` of the Petri
# net stored in ``Analisi_1/L1.pnml`` — read straight off the document, with
# the ``.pnml`` file itself never opened:

# %%
segment_a = refs.filter(pc.equal(refs["comment"], "Analisi_1_L1_A"))
segment_a.to_pandas()

# %% [markdown]
# A segment no ``<place>`` names keeps its row rather than being dropped:
# ``external_id`` falls back to the segment id, ``access_points`` is empty,
# and the comment records why. Exactly one segment of this analysis is
# unmapped:

# %%
unmapped = refs.filter(pc.equal(pc.list_value_length(refs["access_points"]), 0))
unmapped.to_pandas()

# %% [markdown]
# ## Recap
#
# | What the bundle expresses | How |
# |---|---|
# | The spine, a cumulative VTU axis | `spine:dlt_1`, `DiscreteLogicalTimeline`, ``ticks`` |
# | Notes, rests, lyrics at spine coordinates | `los:dlt_2`, verbatim `duration_num`/`duration_den` |
# | Engraved editions, page-image boxes | one nested `SegmentLine` per edition:
# |   4 page segments carrying 18 accolades, an `IntervalToConstantMap` to page images, ``pixels`` |
# | Audio recordings | one `ContinuousPhysicalTimeline` per track, ``seconds`` |
# | Every projection onto the spine | one columnar {{< glossary MatchClaimField >}} via `loader.get_field(MatchClaim)` |
# | The cross-section over spine coordinates | `bundle.get_matchstamp_table(from_graph=True)` |
# | The Petri-net analysis | `spine.external_references` — segment → place, resolved without opening any `.pnml` |
#
# One IEEE 1599 document — one spine, several projections, one analytical
# annotation layer — loaded into a single {{< glossary AlignmentBundle >}}
# in which every representation of the work stays reachable from the axis
# the document itself defines.
