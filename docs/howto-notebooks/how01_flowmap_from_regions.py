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
# # How to Build a FlowMap from Regions
#
# A printed score is **folded**: every coordinate on it is written once, even
# where the music is played several times or skipped entirely. A performance is
# **unfolded**: it plays a particular *path* through the page, in which some
# spans are repeated, some are omitted, and the rest follow in a single stream.
#
# A {{< glossary FlowMap >}} states such a path: an ordered list of stretches
# of one root timeline, each named by the ID of a {{< glossary Region >}}, a
# child timeline or a virtual timeline, with {{< glossary Gap >}}s between
# them where nothing of the timeline is played. Resolved against its timeline,
# it converts positions between the page and the path in both directions.
#
# This guide builds a FlowMap **from named spans you define yourself** rather
# than from the score's flow control. That is the right tool when the path is
# a performing decision (a cut, an omitted repeat, a bespoke concert ending)
# rather than something notated in the score. For the flows a score's own
# repeat signs give, see [How to Encode and Play Flow Control](how01_flow_control.ipynb).
#
# The worked example is a **cut**: a performance that plays measures 1–41,
# **skips measures 42 and 43**, then continues from measure 44 to the end.
#
# **Learning objectives:**
#
# 1. Mark the played spans as {{< glossary Region >}}s on the folded timeline
# 2. Build and attach a FlowMap in one call with `Timeline.create_flow_map()`
# 3. Map a folded coordinate to its performance position with `unfold()`,
#    and see a skipped coordinate return `[]`
# 4. `fold()` a performance coordinate back onto the score
# 5. State the performance as a {{< glossary "Virtual timeline" >}}
# 6. Give the chunks as IDs or as Region objects, and attach a FlowMap you
#    built yourself
# 7. Put the cut measures back as a gap, reading the performance against the
#    page
#
# **Prerequisites:** `how01_manual_timeline_construction`, `how01_coordinate_math`

# %% [markdown]
# ## Setup
#
# We work in quarterbeats (QB). The example score is 50 measures of 3/4, so each
# measure is 3 QB long and the whole score is 150 QB. `build_score()` returns a
# folded {{< glossary Timeline >}} with one downbeat marker per measure;
# `probe()` unfolds a handful of coordinates through a resolved map so we can
# compare maps side by side.

# %%
from timetoalign import Gap
from timetoalign.timelines import ContinuousLogicalTimeline, FlowMap
from timetoalign.timelines.flow import PlacedChunk

QB_PER_MEASURE = 3
N_MEASURES = 50


def build_score(
    n_measures: int = N_MEASURES, qb_per_measure: int = QB_PER_MEASURE
) -> ContinuousLogicalTimeline:
    """Return a folded score timeline with one downbeat marker per measure."""
    length = n_measures * qb_per_measure
    score = ContinuousLogicalTimeline(length=length, uid="score")
    score.add_events(
        [
            {
                "id": f"m{m}",
                "temporal_type": "instant",
                "event_type": "Downbeat",
                "instant": (m - 1) * qb_per_measure,
            }
            for m in range(1, n_measures + 1)
        ]
    )
    return score


def downbeat(measure: int, qb_per_measure: int = QB_PER_MEASURE) -> int:
    """Folded QB coordinate of a measure's downbeat."""
    return (measure - 1) * qb_per_measure


def probe(resolved, coords: list[int]) -> dict[int, list[float]]:
    """Unfold each probe coordinate; a skipped coordinate yields ``[]``."""
    return {c: [float(t.value) for t in resolved.unfold(c)] for c in coords}


# %%
score = build_score()
{
    "id": score.id,
    "unit": str(score.unit),
    "length_qb": float(score.length.value),
    "n_measures": score.n_events,
}

# %% [markdown]
# ## The played spans
#
# The performance omits measures 42 and 43. In QB space those two measures
# occupy `[123, 129)`: the downbeat of measure 42 is at QB 123 and the downbeat
# of measure 44 is at QB 129. Everything before and after is played, so the path
# is exactly two spans:
#
# - measures 1–41 → `[0, 123)`
# - measures 44–50 → `[129, 150)`
#
# We record each span as a {{< glossary Region >}} on the folded timeline, giving
# each an ID to refer to it by and a name to display.

# %%
CUT = (42, 43)
before_end = downbeat(CUT[0])  # 123: start of measure 42
after_start = downbeat(CUT[1] + 1)  # 129: start of measure 44

score.create_region(0, before_end, id="before_cut", name="Before the cut")
score.create_region(after_start, score.length, id="after_cut", name="After the cut")

score.list_regions()

# %% [markdown]
# ## Build and attach the FlowMap
#
# `Timeline.create_flow_map()` takes the played spans in the order played,
# builds the {{< glossary FlowMap >}}, attaches it to the timeline under the
# given ID, and returns it. The map names its chunks; it holds no interval.

# %%
cut_map = score.create_flow_map(["before_cut", "after_cut"], id="cut")
{"map": cut_map, "chunks": cut_map.chunks, "attached": score.list_flow_maps()}

# %% [markdown]
# `resolve()` reads the map against its timeline: each chunk becomes the
# stretch of the score it names, placed where the one before it ends. The
# spans concatenate, so the performance runs measures 1–41 straight into
# measures 44–50 and is 144 QB long, the 6 QB of the omitted measures removed
# from the folded 150.

# %%
cut = cut_map.resolve(score)
[(str(chunk.source), str(chunk.start), str(chunk.end)) for chunk in cut.chunks]

# %% [markdown]
# ## Unfold a coordinate: folded → performance
#
# `unfold()` reports where a folded coordinate lands in the performance.
# Because a folded coordinate could be played more than once (under a
# repeat), it always returns a **list** of positions, each a Coordinate in the
# score's unit. Here nothing repeats, so a played coordinate yields exactly
# one position.
#
# The first span is unshifted: measures 1–41 play at their score coordinates.

# %%
{"folded_qb": downbeat(11), "unfolded": cut.unfold(downbeat(11))}

# %% [markdown]
# The second span slides earlier by the 6 QB of removed music (two 3-QB
# measures). Measure 46's downbeat sits at QB 135 on the page but arrives at
# QB 129 in performance:

# %%
{"folded_qb": downbeat(46), "unfolded": cut.unfold(downbeat(46))}

# %% [markdown]
# ### A skipped coordinate returns `[]`
#
# The downbeats of the omitted measures 42 and 43 lie in neither span, so they
# are played **nowhere**. `unfold()` returns an empty list: the coordinate has
# no performance position at all.

# %%
probe(cut, [downbeat(42), downbeat(43)])

# %% [markdown]
# ## Fold: performance → folded
#
# `fold()` is the inverse. A position of the performance plays exactly one
# position of the score. Folding measure 46's arrival time (QB 129 in
# performance) recovers its score position (QB 135):

# %%
{"performance_qb": 129, "folded": cut.fold(129)}

# %% [markdown]
# ## The performance as a virtual timeline
#
# A {{< glossary "Virtual timeline" >}} states that a FlowMap was applied to
# its timeline. It has an ID, the score's unit and number type, and the
# performance's length, and holds nothing else: no events, no children.
# Positions convert through it in both directions, given the score.

# %%
performance = score.create_virtual_timeline("cut", id="performance")
{
    "virtual timeline": performance,
    "measure 46 is played at": performance.unfold(downbeat(46), score),
    "QB 129 plays": performance.fold(129, score),
}

# %% [markdown]
# ## Chunks as IDs or as objects
#
# A chunk is given by its ID or as the object itself, here the Region; the
# map holds IDs either way. A FlowMap can also be built on its own, naming its
# timeline by ID, and attached with `add_flow_map()`. All three describe the
# same path, so they unfold identically:

# %%
from_regions = score.create_flow_map(
    [score.get_region("before_cut"), score.get_region("after_cut")],
    id="cut_from_regions",
)
built_alone = score.add_flow_map(
    FlowMap("score", ["before_cut", "after_cut"], id="cut_built_alone")
)
probes = [downbeat(11), downbeat(42), downbeat(46)]
{
    "same chunks": from_regions.chunks == built_alone.chunks == cut_map.chunks,
    "from_ids": probe(cut, probes),
    "from_regions": probe(from_regions.resolve(score), probes),
    "built_alone": probe(built_alone.resolve(score), probes),
}

# %% [markdown]
# ## Putting the cut back as a gap
#
# Everything so far **concatenates**: each span starts where the previous one
# ended, which is how a performance runs. Reading the performance against the
# page needs the opposite: the second span has to return to measure 44, and
# the six removed quarters have to reopen as a hole. A {{< glossary Gap >}} of
# 6 QB between the two spans says so.
#
# Take the performance as the source this time: a 144 QB timeline whose two
# spans meet at QB 123.

# %%
played = build_score(n_measures=48)  # 144 QB: the cut playthrough
played.create_region(0, before_end, id="before_cut")
played.create_region(before_end, played.length, id="after_cut")

laid_out = played.create_flow_map(["before_cut", Gap(6), "after_cut"], id="laid_out")
page = laid_out.resolve(played)
{
    "length_qb": page.length,
    "stretches": [
        (
            str(chunk.source) if isinstance(chunk, PlacedChunk) else "gap",
            str(chunk.start),
            str(chunk.end),
        )
        for chunk in page.chunks
    ],
}

# %% [markdown]
# Measures 1–41 sit at QB 0 and measures 44–50 are back at QB 129, the
# downbeat of measure 44, with the two omitted measures empty between them;
# the whole is 150 QB, as the page is. A coordinate inside the gap plays
# nothing, so `fold()` returns None rather than inventing a position:

# %%
{
    "measure 44 on the page plays": page.fold(downbeat(44)),
    "measure 43 on the page plays": page.fold(downbeat(43)),
}

# %% [markdown]
# The omitted music itself does **not** come back: the performance never held
# it, and the gap only keeps its place.

# %% [markdown]
# ## Summary
#
# | Task | API |
# |------|-----|
# | Mark a played span | `timeline.create_region(start, end, id=..., name=...)` |
# | Build + attach a FlowMap from spans | `timeline.create_flow_map(chunks, id=...)` |
# | Build a FlowMap alone, then attach it | `FlowMap(timeline_id, chunks)`, `timeline.add_flow_map(flow_map)` |
# | Chunks | the ID or object of a Region, a child timeline or a virtual timeline, or a `Gap(length)` |
# | Read a map against its timeline | `flow_map.resolve(timeline)` |
# | Coordinate folded → performance (1 → N) | `resolved.unfold(coord)` |
# | A skipped coordinate | `unfold()` returns `[]` |
# | Coordinate performance → folded (N → 1) | `resolved.fold(coord)`, None inside a gap |
# | A virtual timeline | `timeline.create_virtual_timeline(flow_map)`, `unfold(x, timeline)`, `fold(x, timeline)` |
#
# The same {{< glossary FlowMap >}} also states *repeats*: a span listed twice
# plays twice, and `unfold()` then returns two positions.
