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
# # How to Transfer Annotations Between Graphical Analyses
#
# This notebook demonstrates how to project graphical annotations from one
# image-based analytical diagram onto another using the TimeToAlign! library.
#
# The use case comes from Lasse Thoresen's spectromorphological analyses of
# *Objets Obscurs* by Manuella Blackburn.  Two published editions of the same
# analysis exist:
#
# - **DGT2** (2010): five separate images (strips), each covering roughly 30
#   seconds of music.
# - **DGT1** (2009): a single image with five horizontally stacked systems.
#
# Eleven annotated sound-object rectangles are placed on DGT2.  The goal is to
# project them onto DGT1 via a shared physical timeline (seconds), creating
# cross-domain `MatchClaims` that record the correspondence.
#
# ## Key Concepts Demonstrated
#
# - Building a `SegmentLine` by concatenation (DGT2: total length unknown in advance)
# - Building a plain `DiscreteGraphicalTimeline` with children (DGT1: known layout)
# - Attaching `ScalarMap` and `ConstantMap` to timeline segments
# - Adding interval events to child timelines
# - Querying `TimeIntervalStamp` via `get_timestamp_for()`
# - Creating `MatchClaim.from_projection()` for coordinate transfer
# - Querying `MatchStamp` from both a single claim and an `AlignmentBundle`
# - Transferring y-coordinates via per-segment `LinearMap`

# %% [markdown]
# ## Setup

# %%
import json

import pandas as pd

from timetoalign import (
    Agent,
    AlignmentBundle,
    ConstantMap,
    DiscreteGraphicalTimeline,
    LinearMap,
    MatchClaim,
    MatchMetadata,
    ScalarMap,
    TimeUnit,
)
from timetoalign.core import AgentType
from timetoalign.testdata import ensure_data  # noqa: E402
from timetoalign.timelines import SegmentLine

DATA_DIR = ensure_data("thoresen")

# %% [markdown]
# ## 1. Load the Annotation Data
#
# The file `thoresen_test.tsv` contains 11 annotated sound-object rectangles,
# each with pixel coordinates on one of the five DGT2 images and a start time
# and duration in seconds (from manual alignment with the audio).

# %%
events_df = pd.read_csv(DATA_DIR / "thoresen_test.tsv", sep="\t")

# Parse the JSON rectangle coordinates into separate columns
rect_data = events_df["rect_coords_json"].apply(json.loads).apply(pd.Series)
events_df = pd.concat([events_df, rect_data], axis=1)
events_df[
    [
        "graphical_element_id",
        "image_filename",
        "start_time_sec",
        "duration_sec",
        "x",
        "y",
        "width",
        "height",
    ]
]

# %% [markdown]
# ## 2. Build DGT2 (2010 Version, 5 Images)
#
# DGT2 is a `SegmentLine` of `DiscreteGraphicalTimeline` segments.  We do not
# know the total length in advance -- it accumulates as segments are appended.
#
# ### Coordinate constants
#
# From the source images, each segment has slightly different pixel boundaries:
#
# | Segment | Image          | x0 | x1  | Length |
# |---------|----------------|-----|-----|--------|
# | 1       | page1_1.jpeg   |  8  | 874 |   866  |
# | 2       | page1_2.jpeg   |  7  | 874 |   867  |
# | 3       | page1_3.jpeg   |  7  | 874 |   867  |
# | 4       | page1_4.jpeg   |  8  | 872 |   864  |
# | 5       | page2_1.jpeg   |  9  | 873 |   864  |

# %%
# Segment bounds: (x0, x1, y_axis) for each image
DGT2_BOUNDS = [
    (8, 874, 15),  # page1_1
    (7, 874, 18),  # page1_2
    (7, 874, 19),  # page1_3
    (8, 872, 15),  # page1_4
    (9, 873, 20),  # page2_1
]
DGT2_LENGTHS = [x1 - x0 for x0, x1, _ in DGT2_BOUNDS]

DGT2_IMAGE_FILES = [
    "thoresen_2010_form-building-patterns_p90-91_page1_1.jpeg",
    "thoresen_2010_form-building-patterns_p90-91_page1_2.jpeg",
    "thoresen_2010_form-building-patterns_p90-91_page1_3.jpeg",
    "thoresen_2010_form-building-patterns_p90-91_page1_4.jpeg",
    "thoresen_2010_form-building-patterns_p90-91_page2_1.jpeg",
]

# Audio spans 150 seconds total (5 segments x 30 seconds each)
AUDIO_DURATION = 150.0
SECONDS_PER_SEGMENT = AUDIO_DURATION / 5  # 30.0

# Build a lookup from image filename to segment index
IMAGE_TO_SEGMENT = {name: i for i, name in enumerate(DGT2_IMAGE_FILES)}

# %% [markdown]
# ### Create the SegmentLine by concatenation

# %%
dgt2 = SegmentLine[DiscreteGraphicalTimeline](
    length=0,
    unit="pixels",
    uid="dgt_2",
    name="Thoresen 2010",
)

dgt2_segments = []
for i, (seg_len, filename) in enumerate(zip(DGT2_LENGTHS, DGT2_IMAGE_FILES)):
    seg = DiscreteGraphicalTimeline(
        length=seg_len,
        unit="pixels",
        uid=f"dgt2_seg{i+1}",
        name=f"seg_{i+1}",
    )

    # Attach ScalarMap: pixels -> seconds
    px_to_sec = ScalarMap(
        scalar=SECONDS_PER_SEGMENT / seg_len,
        source_unit="pixels",
        target_unit="seconds",
        name=f"px_to_sec_{i+1}",
    )
    seg.add_conversion_map(px_to_sec)

    # Attach ConstantMap: every coordinate carries the source filename
    fname_map = ConstantMap(value=filename, name="filename")
    seg.add_conversion_map(fname_map)

    dgt2.append_segment(seg, name=f"seg_{i+1}")
    dgt2_segments.append(seg)

print(f"DGT2 total length: {dgt2.length} (expected: {sum(DGT2_LENGTHS)})")
print(f"Number of segments: {dgt2.n_segments}")

# %% [markdown]
# ### Add the 11 events to their respective segments
#
# Each event is placed on the correct DGT2 segment using the `image_filename`
# column from the TSV.  Coordinates are converted to *local* segment pixels
# (subtracting the segment's `x0` boundary).

# %%
for _, row in events_df.iterrows():
    seg_idx = IMAGE_TO_SEGMENT[row["image_filename"]]
    x0 = DGT2_BOUNDS[seg_idx][0]

    # Local coordinates within the segment
    local_start = row["x"] - x0
    local_end = local_start + row["width"]

    dgt2_segments[seg_idx].add_events(
        [
            {
                "id": row["graphical_element_id"],
                "event_type": "sound_object",
                "start": float(local_start),
                "end": float(local_end),
            }
        ]
    )

print("Events added to DGT2 segments")
for i, seg in enumerate(dgt2_segments):
    n = seg.n_events
    if n > 0:
        print(f"  seg_{i+1}: {n} events")

# %% [markdown]
# ### Validate: Event H TimeIntervalStamp
#
# Event H (`rect_h2`) is the specimen highlighted in the manuscript.  It sits
# on segment 2 (page1_2.jpeg) at raw pixel x=385, width=139.
#
# - Local x: 385 - 7 = 378 to 378 + 139 = 517
# - Global x: 866 + 378 = 1244 to 866 + 517 = 1383
# - Expected seconds: ~43.1 to ~47.9 (pixel-derived)
# - TSV ground truth: 43.5 s, duration 4.5 s

# %%
stamp_h = dgt2.get_timestamp_for("rect_h2")
print(stamp_h)

# %%
# Extract the key values for validation
h_start_global = stamp_h.start.axis
h_end_global = stamp_h.end.axis
print(f"Event H global pixels: [{h_start_global}, {h_end_global})")
print("  Expected: [1244, 1383)")


# %%
# Helper: convert a global DGT2 pixel coordinate to seconds.
# The ScalarMap on each segment converts local pixels to the *local* seconds
# within a 30-second window.  We need to know the segment index and offset
# to compute the absolute time.
def dgt2_global_px_to_sec(global_px: float) -> float:
    """Convert a global DGT2 pixel coordinate to absolute seconds."""
    offset = 0
    for i, seg_len in enumerate(DGT2_LENGTHS):
        if global_px < offset + seg_len:
            local_px = global_px - offset
            local_sec = local_px * (SECONDS_PER_SEGMENT / seg_len)
            return i * SECONDS_PER_SEGMENT + local_sec
        offset += seg_len
    # Edge: coordinate at the very end
    return AUDIO_DURATION


h_start_sec = dgt2_global_px_to_sec(h_start_global.value)
h_end_sec = dgt2_global_px_to_sec(h_end_global.value)
print(f"Event H seconds (pixel-derived): [{h_start_sec:.2f}, {h_end_sec:.2f})")
print("  TSV ground truth: [43.5, 48.0)")

# %% [markdown]
# The pixel-derived seconds differ slightly from the TSV values because the
# linear mapping assumes uniform time-per-pixel across each segment, whereas
# the TSV values come from a separate manual alignment.  The difference is
# small enough to demonstrate the coordinate-transfer workflow.

# %% [markdown]
# ## 3. Build DGT1 (2009 Version, Single Image)
#
# DGT1 is a plain `DiscreteGraphicalTimeline` with a known total length
# (5 systems x 967 pixels = 4835 pixels).  Each system becomes a child
# timeline created via `create_child()`.
#
# | System | Offset | Length |
# |--------|--------|--------|
# | 1      | 0      | 967    |
# | 2      | 967    | 967    |
# | 3      | 1934   | 967    |
# | 4      | 2901   | 967    |
# | 5      | 3868   | 967    |

# %%
DGT1_SEGMENT_LENGTH = 967  # x1 - x0 = 969 - 2
DGT1_TOTAL_WIDTH = DGT1_SEGMENT_LENGTH * 5  # 4835

dgt1 = DiscreteGraphicalTimeline(
    length=DGT1_TOTAL_WIDTH,
    unit="pixels",
    uid="dgt_1",
    name="Thoresen 2009",
)

# Attach a single ScalarMap on the parent: pixels -> seconds
dgt1_px_to_sec = ScalarMap(
    scalar=AUDIO_DURATION / DGT1_TOTAL_WIDTH,
    source_unit="pixels",
    target_unit="seconds",
    name="px_to_sec",
)
dgt1.add_conversion_map(dgt1_px_to_sec)

# Create 5 children (systems)
dgt1_children = []
for i in range(5):
    child = dgt1.create_child(
        length=DGT1_SEGMENT_LENGTH,
        offset=i * DGT1_SEGMENT_LENGTH,
        uid=f"dgt1_sys{i+1}",
        name=f"sys_{i+1}",
    )
    dgt1_children.append(child)

print(f"DGT1 total length: {dgt1.length}")
print(f"Number of children: {len(dgt1_children)}")

# %% [markdown]
# ## 4. Create AlignmentBundle
#
# Both timelines are added directly to the bundle as standalone timelines
# (no explicit `TimelineGroup` wrapping needed -- the bundle creates implicit
# singleton groups).

# %%
bundle = AlignmentBundle(name="Thoresen Annotation Transfer")
bundle.add_timeline(dgt2)
bundle.add_timeline(dgt1)
print(bundle)

# %% [markdown]
# ## 5. Create MatchClaims (Coordinate Projection)
#
# For each event on DGT2, we:
# 1. Get its `TimeIntervalStamp` to extract global pixel coordinates and seconds.
# 2. Project the seconds onto DGT1 via the inverse of DGT1's ScalarMap.
# 3. Create a `MatchClaim.from_projection()` recording the correspondence.

# %%
# Get the inverse map for DGT1: seconds -> pixels
dgt1_sec_to_px = dgt1_px_to_sec.inverse()

# Collect all event IDs from the TSV
event_ids = events_df["graphical_element_id"].tolist()

claims = []
projection_data = []

for event_id in event_ids:
    # Get the TimeIntervalStamp from DGT2
    stamp = dgt2.get_timestamp_for(event_id)

    # Extract global DGT2 coordinates (on the SegmentLine axis)
    dgt2_start = stamp.start.axis
    dgt2_end = stamp.end.axis

    # Convert global pixels to seconds via the segment-aware helper
    sec_start = dgt2_global_px_to_sec(dgt2_start.value)
    sec_end = dgt2_global_px_to_sec(dgt2_end.value)

    # Project seconds onto DGT1 pixels via the inverse of DGT1's ScalarMap
    dgt1_start = float(dgt1_sec_to_px(sec_start))
    dgt1_end = float(dgt1_sec_to_px(sec_end))

    # Create the MatchClaim
    claim = MatchClaim.from_projection(
        event={"id": event_id, "name": event_id, "start": dgt2_start, "end": dgt2_end},
        source_tl_id="dgt_2",
        target_tl_id="dgt_1",
        target_coord=dgt1_start,
        source_unit=TimeUnit.pixels,
        target_unit=TimeUnit.pixels,
        target_end_coord=dgt1_end,
        end_coord_key="end",
        metadata=MatchMetadata(
            agent=Agent(
                name="Thoresen analysis",
                type=AgentType.human,
                identifier="coordinate_projection_via_shared_physical_timeline",
            ),
            certainty=1.0,
        ),
    )
    claims.append(claim)

    # Track data for the summary table
    projection_data.append(
        {
            "event_id": event_id,
            "dgt2_start_px": dgt2_start,
            "dgt2_end_px": dgt2_end,
            "seconds_start": round(sec_start, 2),
            "seconds_end": round(sec_end, 2),
            "dgt1_start_px": round(dgt1_start, 1),
            "dgt1_end_px": round(dgt1_end, 1),
        }
    )

{"claims": len(claims)}

# %%
# Add claims to the bundle
bundle.add_match_claims(claims)

# %%
# Display an example claim (shows event ID, timelines, coordinates)
claims[0]

# %% [markdown]
# ## 6. Query MatchStamps
#
# ### Event H MatchStamp (from a single claim)
#
# Since each MatchClaim has two anchors (start and end of the interval),
# the claim-level `get_matchstamp(from_graph=False)` returns a reduced
# stamp for the start anchor, showing the coordinate on both DGT2 and DGT1.

# %%
# Find the claim for Event H (rect_h2)
claim_h = next(c for c in claims if c.event_a_id == "rect_h2")
matchstamp_h_start = claim_h.get_matchstamp(from_graph=False)
matchstamp_h_start

# %%
# The end anchor can be queried the same way by creating a fresh claim
# from just the end anchor, or we can read the coordinates directly:
{
    "start": {
        "DGT2 px": claim_h.start_anchor.coordinate_a.value,
        "DGT1 px": round(claim_h.start_anchor.coordinate_b.value, 1),
    },
    "end": {
        "DGT2 px": claim_h.end_anchor.coordinate_a.value,
        "DGT1 px": round(claim_h.end_anchor.coordinate_b.value, 1),
    },
}

# %% [markdown]
# ### All 11 MatchClaims as a DataFrame

# %%
projection_df = pd.DataFrame(projection_data)
projection_df = projection_df.set_index("event_id")
print(projection_df.to_string())

# %% [markdown]
# ### Validation: all DGT1 coordinates within bounds

# %%
assert projection_df["dgt1_start_px"].min() >= 0, "DGT1 start below zero"
assert (
    projection_df["dgt1_end_px"].max() <= DGT1_TOTAL_WIDTH
), "DGT1 end exceeds total width"
print(f"All 11 events project within DGT1 range [0, {DGT1_TOTAL_WIDTH})")

# %% [markdown]
# ---
# ## 7. Transfer Y-Coordinates (Part 2)
#
# The x-coordinate transfer above uses the shared physical timeline (seconds)
# as an intermediary.  The y-coordinate transfer works differently: it maps
# the *event space* (the vertical region where sound objects are drawn) from
# each DGT2 segment to the corresponding DGT1 system.
#
# ### Y-Range Definitions
#
# **DGT1 y-ranges** (from the single 935 px image):
#
# | System | y0 (time axis) | y1 (event bottom) | Height |
# |--------|----------------|-------------------|--------|
# | 1      | 18             | 165               | 147    |
# | 2      | 205            | 353               | 148    |
# | 3      | 396            | 544               | 148    |
# | 4      | 588            | 736               | 148    |
# | 5      | 785            | 933               | 148    |
#
# **DGT2 y-ranges** (per-image event space, excluding the form-analytical layer):
#
# | Segment | y0 (time axis) | y1 (event bottom) | Height |
# |---------|----------------|-------------------|--------|
# | 1       | 15             | 148               | 133    |
# | 2       | 18             | 150               | 132    |
# | 3       | 19             | 152               | 133    |
# | 4       | 15             | 147               | 132    |
# | 5       | 20             | 152               | 132    |

# %%
DGT1_Y_RANGES = [(18, 165), (205, 353), (396, 544), (588, 736), (785, 933)]
DGT2_Y_RANGES = [(15, 148), (18, 150), (19, 152), (15, 147), (20, 152)]

# %% [markdown]
# ### Create per-segment LinearMaps for y-transfer
#
# Each segment has a LinearMap that maps `[dgt2_y0, dgt2_y1]` to
# `[dgt1_y0, dgt1_y1]`.  The linear mapping is:
#
# $$y_{\text{DGT1}} = y_{0}^{\text{DGT1}} + \frac{y - y_{0}^{\text{DGT2}}}{h^{\text{DGT2}}} \cdot h^{\text{DGT1}}$$

# %%
y_maps = []
for i, ((d2_y0, d2_y1), (d1_y0, d1_y1)) in enumerate(zip(DGT2_Y_RANGES, DGT1_Y_RANGES)):
    d2_height = d2_y1 - d2_y0
    d1_height = d1_y1 - d1_y0
    scalar = d1_height / d2_height
    offset = d1_y0 - scalar * d2_y0

    lmap = LinearMap(
        scalar=scalar,
        offset=offset,
        name=f"y_transfer_seg{i+1}",
    )
    y_maps.append(lmap)
    print(
        f"Segment {i+1}: DGT2 [{d2_y0}, {d2_y1}] -> DGT1 [{d1_y0}, {d1_y1}], "
        f"scalar={scalar:.4f}, offset={offset:.2f}"
    )

# %% [markdown]
# ### Apply y-transfer to all events

# %%
full_transfer_data = []

for _, row in events_df.iterrows():
    event_id = row["graphical_element_id"]
    seg_idx = IMAGE_TO_SEGMENT[row["image_filename"]]
    x0 = DGT2_BOUNDS[seg_idx][0]

    # DGT2 local coordinates
    dgt2_x_start = row["x"] - x0
    dgt2_x_end = dgt2_x_start + row["width"]
    dgt2_y_top = row["y"]
    dgt2_y_bottom = row["y"] + row["height"]

    # Get the projection data we already computed
    proj = next(p for p in projection_data if p["event_id"] == event_id)

    # Apply y-transfer via the segment's LinearMap
    dgt1_y_top = float(y_maps[seg_idx](dgt2_y_top))
    dgt1_y_bottom = float(y_maps[seg_idx](dgt2_y_bottom))

    full_transfer_data.append(
        {
            "event_id": event_id,
            "segment": seg_idx + 1,
            "dgt2_x_start": dgt2_x_start,
            "dgt2_y_top": dgt2_y_top,
            "dgt2_x_end": dgt2_x_end,
            "dgt2_y_bottom": dgt2_y_bottom,
            "sec_start": proj["seconds_start"],
            "sec_end": proj["seconds_end"],
            "dgt1_x_start": proj["dgt1_start_px"],
            "dgt1_y_top": round(dgt1_y_top, 1),
            "dgt1_x_end": proj["dgt1_end_px"],
            "dgt1_y_bottom": round(dgt1_y_bottom, 1),
        }
    )

transfer_df = pd.DataFrame(full_transfer_data).set_index("event_id")

# %% [markdown]
# ### Full 2D transfer table

# %%
print(transfer_df.to_string())

# %% [markdown]
# ### Event H: detailed y-coordinate transfer
#
# Event H (`rect_h2`) is on DGT2 segment 2 at y=46, height=20.
# DGT2 segment 2 y-range: [18, 150], height=132.
# DGT1 system 2 y-range: [205, 353], height=148.
#
# Relative y-position: (46 - 18) / 132 = 0.212
# Mapped y_top: 205 + 0.212 * 148 ~ 236
# Mapped y_bottom: 205 + ((66 - 18) / 132) * 148 ~ 259

# %%
h_row = transfer_df.loc["rect_h2"]
print("Event H (rect_h2) full 2D transfer:")
print(
    f"  DGT2: x=[{h_row['dgt2_x_start']}, {h_row['dgt2_x_end']}), "
    f"y=[{h_row['dgt2_y_top']}, {h_row['dgt2_y_bottom']})"
)
print(f"  Seconds: [{h_row['sec_start']}, {h_row['sec_end']})")
print(
    f"  DGT1: x=[{h_row['dgt1_x_start']}, {h_row['dgt1_x_end']}), "
    f"y=[{h_row['dgt1_y_top']}, {h_row['dgt1_y_bottom']})"
)

# %% [markdown]
# ### Validation: y-coordinates within bounds

# %%
for _, row in pd.DataFrame(full_transfer_data).iterrows():
    seg_idx = int(row["segment"]) - 1
    d1_y0, d1_y1 = DGT1_Y_RANGES[seg_idx]
    assert (
        row["dgt1_y_top"] >= d1_y0
    ), f"{row['event_id']}: y_top {row['dgt1_y_top']} below system y0 {d1_y0}"
    assert (
        row["dgt1_y_bottom"] <= d1_y1
    ), f"{row['event_id']}: y_bottom {row['dgt1_y_bottom']} exceeds system y1 {d1_y1}"
print("All y-coordinates within their respective DGT1 system bounds")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the full annotation-transfer workflow:
#
# 1. **DGT2** was built as a `SegmentLine` by concatenation, with `ScalarMap`
#    (pixels to seconds) and `ConstantMap` (source filename) on each segment.
# 2. **DGT1** was built as a plain `DiscreteGraphicalTimeline` with five
#    children and a single `ScalarMap` on the parent.
# 3. **11 MatchClaims** were created by projecting DGT2 events through seconds
#    onto DGT1, using `MatchClaim.from_projection()`.
# 4. **MatchStamps** confirmed the coordinate correspondence from both the
#    claim and bundle perspectives.
# 5. **Y-coordinates** were transferred via per-segment `LinearMap`, mapping
#    each segment's event space to the corresponding DGT1 system.
#
# The two construction patterns -- SegmentLine concatenation (DGT2) vs. plain
# timeline with children (DGT1) -- illustrate how TimeToAlign! supports both
# "accumulate as you go" and "known layout" approaches.
