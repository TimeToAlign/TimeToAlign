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
# # Flow Control and Grids

# %% [markdown]
# ## What you will build
#
# You will build a folded {{< glossary Timeline >}}, describe its played order,
# and obtain an unfolded timeline that preserves the route back to the page.
# You will also build a {{< glossary BeatGrid >}} whose measures and beats can
# be queried in either direction, survive a tempo change, and be exported for
# annotation software.

# %% [markdown]
# ## Before you start
#
# Complete [Nesting and Timestamps](tut02_nesting_and_timestamps.ipynb),
# [Loading Real Data](tut04_loading_data.ipynb), and
# [Timeline Groups](tut05_timeline_groups.ipynb) first. This notebook follows
# [Alignment Bundles](tut06_alignment_bundles.ipynb) in reading order but does
# not depend on its bundles or claims.

# %%
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

from timetoalign import (
    BeatGrid,
    BeatGridSegment,
    BeatPolicy,
    ContinuousLogicalTimeline,
    Ms3Loader,
)
from timetoalign.testdata import ensure_data
from timetoalign.timelines import Flow, FlowMap, ScoreFlowController

score_data = ensure_data("score")

# %% [markdown]
# ## A page is not a performance
#
# A repeat sign or *Da Capo* changes traversal order without changing the page.
# The written timeline is **folded**; the route actually played is **unfolded**.

# %%
written_score = ContinuousLogicalTimeline.from_events(
    [
        {
            "id": "written_end",
            "temporal_type": "instant",
            "event_type": "Boundary",
            "instant": Fraction(12),
        }
    ],
    uid="written_score",
    name="Three written spans",
)
span_bounds = {
    "A": (Fraction(0), Fraction(4)),
    "B": (Fraction(4), Fraction(8)),
    "C": (Fraction(8), Fraction(12)),
}
written_order = list(span_bounds)
played_order = ["A", "A", "C"]
page_and_performance = {
    "written length": written_score.length,
    "written order": written_order,
    "played order": played_order,
}
page_and_performance

# %% [markdown]
# The length is a {{< glossary Coordinate >}} measured in quarters. The page
# contains A, B, and C once each, but the performance repeats A and skips B.

# %% [markdown]
# ## Naming the played spans
#
# A {{< glossary Region >}} gives a stable id to a span of the written score.
# We reuse `create_region` from the nesting tutorial, giving each span its letter
# as its id.

# %%
created_regions = [
    written_score.create_region(start, end, id=span_id)
    for span_id, (start, end) in span_bounds.items()
]
opening_region = created_regions[0]
closing_region = created_regions[2]
named_spans = written_score.list_regions()
region_overview = {
    "opening span": opening_region,
    "region ids": named_spans,
}
region_overview

# %% [markdown]
# Each region keeps its quarter-valued start and end. These ids now let the
# played route refer to legible musical spans rather than unexplained numbers.

# %% [markdown]
# ## Building the map
#
# A {{< glossary FlowMap >}} records how folded spans are placed in played
# order. `create_flow_map(intervals)` accepts a region id, a Region, a `(start, end)`
# pair, a Timeline, or an iterable mixing any of those shapes.

# %%
flow_intervals = [played_order[0], opening_region, closing_region]
performance_map = written_score.create_flow_map(
    flow_intervals,
    id="performance",
)
written_score.add_flow_map(performance_map, id="played")
retrieved_map = written_score.get_flow_map("played")
available_maps = written_score.list_flow_maps()
map_overview = {
    "is a FlowMap": isinstance(retrieved_map, FlowMap),
    "attached ids": available_maps,
    "selected map": retrieved_map,
}
map_overview

# %% [markdown]
# The single `flow_intervals` argument mixes a region name with Region objects.
# `add_flow_map` attaches an existing map under another id; `get_flow_map` and
# `list_flow_maps` retrieve and enumerate the attached maps.

# %% [markdown]
# ## Unfolding one position
#
# `unfold_coordinate(coord)` takes one folded position and returns a list,
# because repeated material can occur at several positions in performance.

# %%
repeated_folded = written_score.get_coordinate(Fraction(1))
skipped_folded = written_score.get_coordinate(Fraction(5))
repeated_unfolded = written_score.unfold_coordinate(repeated_folded, "played")
skipped_unfolded = written_score.unfold_coordinate(skipped_folded, "played")
unfold_report = {
    "folded position in A": repeated_folded,
    "played positions": repeated_unfolded,
    "folded position in B": skipped_folded,
    "played positions for skipped B": skipped_unfolded,
}
unfold_report

# %% [markdown]
# The position in A appears once in each rendition. The empty list for B is the
# correct answer, not a failure: that written span is absent from this route.
# This timeline convenience method returns plain numbers, so its played
# positions do not carry the written timeline's quarter unit.

# %% [markdown]
# ## Folding back
#
# `fold(coord)` goes in the other direction: many possible performance
# positions collapse to one position on the written page.

# %%
later_unfolded = repeated_unfolded[1]
folded_position = written_score.fold(later_unfolded, "played")
fold_report = {
    "later performance position": later_unfolded,
    "position on the page": folded_position,
    "matches the original value": folded_position == repeated_folded.value,
}
fold_report

# %% [markdown]
# The second rendition folds back to the original written value. Like
# `unfold_coordinate`, this convenience method returns a plain number rather
# than a unit-bearing Coordinate. Keep the directions distinct: unfolding is
# one-to-many; folding is many-to-one.

# %% [markdown]
# ## Assembling the whole unfolded timeline
#
# `apply_flow(id)` constructs the complete played timeline. It preserves the
# source's concrete type and makes one {{< glossary Child >}} timeline for every
# played span.

# %%
unfolded_score = written_score.apply_flow("played")
unfolded_children = unfolded_score.list_children()
unfolded_regions = {region.id: region.name for region in unfolded_score.iter_regions()}
reverse_map = unfolded_score.get_flow_map("source")
unfolded_overview = {
    "same concrete type": type(unfolded_score) is type(written_score),
    "children": unfolded_children,
    "regions": unfolded_regions,
    "reverse map": reverse_map,
}
unfolded_overview

# %% [markdown]
# Each child takes the id of the span it plays; the second visit to A is
# `A-rend2`, and further visits would use `-rend3` and so on. Matching regions
# cover the spans in unfolded coordinates: their ids are minted, and their names
# repeat the children's. Those named intervals are what make the route
# invertible through the `source` map.

# %% [markdown]
# ## Inspecting repeat events
#
# Score loaders retain repeat signs and non-sequential destinations as
# flow-control {{< glossary Event >}}s. Inspecting those parsed records first
# makes the controller's later route easier to understand.

# %%
score_dir = score_data / "beethoven_woo71"
score_loader = Ms3Loader.from_file(
    score_dir / "WoO71.notes.tsv",
    score_dir / "WoO71.measures.tsv",
)
measure_data = score_loader.store.measures
measure_table = measure_data.to_dataframe()
repeat_mask = measure_table["start_repeat"] | measure_table["end_repeat"]
repeat_columns = ["mc", "mn", "start_repeat", "end_repeat", "next"]
repeat_rows = measure_table.loc[
    repeat_mask,
    repeat_columns,
]
repeat_events = repeat_rows.head(6)
repeat_events

# %% [markdown]
# These six rows form three repeat-start and repeat-end pairs. The `next`
# column shows the encoded destinations; each repeat end has both a backward
# destination and the following written measure.

# %% [markdown]
# ## Ids and names of flow-control elements
#
# Every element of a timeline (a region, a break, a jump, a marker) has an `id`
# that identifies it, unique across everything you save together in one
# manifest (timelines, children, groups, maps and these elements alike), and a
# name or label for display. A loader gives ids where the source has none, with one
# prefix per kind of flow-control element: `flow_marker_1` for a break,
# `jump_from_1` for a jump and `jump_to_1` for a marker a jump lands on. A short
# score with a *Fine*, a segno and a coda shows all three kinds.

# %%
flow_dir = score_data / "flow_control" / "flow_only"
flow_score = Ms3Loader.from_file(
    flow_dir / "out_of_the_flow_experience-flow_only.measures.tsv",
).create_timeline(uid="flow_only")
flow_elements = flow_score.flow_control
element_ids = {
    "regions": {region.id: region.name for region in written_score.iter_regions()},
    "breaks": {brk.id: brk.name for brk in flow_elements.breaks},
    "jumps": {jump.id: jump.label for jump in flow_elements.jumps},
    "markers": {marker.id: marker.name for marker in flow_elements.markers},
    "jump targets": {jump.id: jump.target for jump in flow_elements.jumps},
}
element_ids

# %% [markdown]
# The written score's regions carry the ids you gave and no names. A jump
# that lands on a marker names that marker's id as its `target`; these repeats
# and the *Da Capo* land on no marker, so their targets are `None`, and each
# jump still carries its destination coordinate.

# %% [markdown]
# ## Computing and checking the route
#
# A `ScoreFlowController` turns the parsed measure records into a
# {{< glossary Flow >}} and its FlowMap. Diagnostics check the default
# unfolding route without interrupting the notebook.

# %%
controller = score_loader.create_flow_controller()
real_flow_map = controller.create_flow_map()
real_flow = real_flow_map.flow
diagnostics = controller.flow_diagnostics()
flow_report = {
    "controller type": isinstance(controller, ScoreFlowController),
    "flow type": isinstance(real_flow, Flow),
    "flow-map type": isinstance(real_flow_map, FlowMap),
    "folded measures": real_flow.folded_length,
    "played measures": real_flow.unfolded_length,
    "diagnostics": [item.kind for item in diagnostics] or "none",
}
flow_report

# %% [markdown]
# The default route visits 505 measures from a 397-measure folded score.
# `flow_diagnostics()` would return records for traversal or repeat-resolution
# problems; the displayed `none` means it found no such problem in this score.

# %% [markdown]
# ## Segmenting the folded score
#
# Controller boundaries mark changes in the score's flow structure. A
# {{< glossary SegmentLine >}} created from them turns the folded timeline into
# adjacent, inspectable sections.

# %%
score_timeline = score_loader.create_timeline(uid="woo71_folded")
section_boundaries = controller.get_section_boundary_coordinates()
boundary_coordinates = [
    score_timeline.start,
    *[score_timeline.make_coordinate(boundary) for boundary in section_boundaries],
    score_timeline.length,
]
score_segments = score_timeline.create_segment_line(boundary_coordinates)
segment_report = {
    "first boundaries": boundary_coordinates[:4],
    "boundary count": len(boundary_coordinates),
    "segment count": score_segments.n_children,
    "segment line": score_segments,
}
segment_report

# %% [markdown]
# The first boundary coordinates retain their quarter unit. Fourteen
# boundaries divide the score into thirteen adjacent segments, and the
# SegmentLine keeps those segments as children of one timeline.

# %% [markdown]
# ## Beats as a queryable grid
#
# A {{< glossary BeatGrid >}} answers where beats and measures fall in a
# recording. It is not a timeline: it stores one tempo statement — the anchor
# instant, the tempo, the meter, and which beat of a bar the anchor is — and
# generates every beat from it. `extent` bounds the grid, normally with the
# length of the audio.

# %%
grid = BeatGrid.from_tempo(120, metro="3/4", start=0, extent=12)
grid_overview = {
    "the grid": grid,
    "one stated segment": grid.segments[0],
    "beat length in seconds": grid.segments[0].beat_seconds,
    "measures generated": grid.n_measures,
}
grid_overview

# %% [markdown]
# At 120 BPM a beat lasts half a second, so a 3/4 bar lasts one and a half and
# twelve seconds hold eight measures. Nothing was stored per beat to say so.

# %% [markdown]
# ## Both directions
#
# `seconds_at(measure, beat)` turns a label into a
# {{< glossary Coordinate >}} in seconds; `position_at(seconds)` turns a
# position into the beat sounding there.

# %%
labelled_instant = grid.seconds_at(3, 2)
sounding_beat = grid.position_at(6.25)
direction_report = {
    "measure 3, beat 2": labelled_instant,
    "beat sounding at 6.25 seconds": sounding_beat,
    "that beat's own instant": sounding_beat.instant,
    "its measure": sounding_beat.measure,
    "it opens a measure": sounding_beat.is_downbeat,
    "quarters in the whole grid": grid.quarters_between(0, 12),
}
direction_report

# %% [markdown]
# `position_at` names the last beat at or before the position, so a query
# between two beats answers with the earlier one — 6.25 seconds sounds within
# the downbeat of measure 5. `quarters_between` measures notated length rather
# than clock time.
#
# A beat carries its position twice, for two jobs. `seconds` is the exact ratio
# the grid computed with, which is what the beat's display leads with; `instant`
# is that position as a seconds {{< glossary Coordinate >}} — the same value
# `seconds_at` returns for the beat's own label, so the two directions meet.

# %% [markdown]
# ## The beat table and its exports
#
# `get_beat_table()` renders every beat the grid states. It is the only table
# the grid builds, and the annotation-file exports are renderings of it.
# Downbeats are the rows whose `beat` is 1.

# %%
beat_table = grid.get_beat_table()
downbeat_rows = beat_table[beat_table["beat"] == 1]
with TemporaryDirectory() as temporary_dir:
    export_path = Path(temporary_dir) / "beats.csv"
    exported_rows = grid.export_to_csv(
        export_path,
        format="sonic_visualiser",
    )
    exported_text = export_path.read_text(encoding="utf-8")
    first_export_lines = exported_text.splitlines()[:4]
table_report = {
    "first beats": beat_table.head(4),
    "beats": len(beat_table),
    "downbeats": len(downbeat_rows),
    "rows written": exported_rows,
    "first CSV lines": first_export_lines,
}
table_report

# %% [markdown]
# The exported first column is measured in seconds, matching the annotation
# file's own convention. The temporary export is genuinely written and read
# back; it leaves no tutorial artefact behind.

# %% [markdown]
# ## When the tempo changes
#
# One segment states one tempo. A recording that changes tempo is several
# `BeatGridSegment`s in one grid: you state each segment without an end, and
# the grid orders them and bounds each by the next one's start. A
# `BeatPolicy` says how the lattice beats group into bars.

# %%
three_four = BeatPolicy.uniform(Fraction(1), 3, name="3/4")
opening = BeatGridSegment(start=0, bpm=120, policy=three_four, battito=1)
faster = BeatGridSegment(start=6, bpm=180, policy=three_four, battito=1)
changing = BeatGrid([opening, faster], extent=12)
across_the_change = changing.get_beat_table()
change_report = {
    "bounded segments": changing.segments,
    "measures generated": changing.n_measures,
    "around the change": across_the_change[
        (across_the_change["seconds"] >= 5) & (across_the_change["seconds"] <= 7)
    ],
    "quarters from 5 to 7 seconds": changing.quarters_between(5, 7),
}
change_report

# %% [markdown]
# The grid gave the first segment the second one's start as its end. Beats last
# half a second before six seconds and a third of a second after, so the two
# seconds around the change hold five quarters — a count no single tempo
# produces, because `quarters_between` integrates each segment's own tempo
# instead of averaging them.

# %% [markdown]
# ## Two numberings of one lattice
#
# `numbering="set"` counts measures across the whole grid; `numbering="segment"`
# restarts the count at each segment. They label the same beats, so you can read
# them side by side.

# %%
per_segment = changing.get_beat_table(numbering="segment")
both_numberings = across_the_change.assign(segment_measure=per_segment["measure"])
numbering_report = both_numberings[
    (both_numberings["seconds"] >= 5) & (both_numberings["seconds"] <= 7)
]
numbering_report

# %% [markdown]
# The faster segment opens measure 5 of the recording and measure 1 of its own
# count. Use whichever the receiving tool expects: a whole-recording annotation
# wants the running count, a per-section analysis the restarted one.

# %% [markdown]
# ## What you learned
#
# - You can distinguish a folded page from its unfolded performance order.
# - You can name written spans with regions.
# - You can build, attach, retrieve, and list polymorphic flow maps.
# - You can unfold one written position to zero, one, or several played positions.
# - You can fold a played position back to one place on the page.
# - You can assemble and inspect a complete unfolded timeline.
# - You can inspect repeat events retained by a score loader.
# - You can compute a score route and obtain non-raising flow diagnostics.
# - You can create a segment line from flow-section boundaries.
# - You can build a beat grid from one tempo statement and bound it with an extent.
# - You can turn a measure and beat into an instant, and an instant back into a beat.
# - You can render every beat as a table and export a real annotation CSV.
# - You can combine several tempo segments into one grid that changes tempo.
# - You can read a grid's measures counted across the whole grid or per segment.

# %% [markdown]
# ## Next
#
# Continue with [The Data Model](tut08_data_model.ipynb).
#
# ## Go deeper
#
# - [Flow control](../howto/how01_flow_control.ipynb)
# - [Flow maps from regions](../howto/how01_flowmap_from_regions.ipynb)
# - [Beat grids](../howto/how01_beat_grids.ipynb)
