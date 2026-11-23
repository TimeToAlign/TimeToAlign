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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # How to Work with Graphical Timelines
#
# This notebook demonstrates how to load and work with graphical timelines
# using the `GraphicalLoader`.
#
# Graphical timelines map 2D pixel coordinates in images to 1D timeline
# coordinates. This is essential for:
# - Musical score analysis (staff systems as timelines)
# - Spectrograms (time on x-axis)
# - Graphical analytical diagrams (like Thoresen's morphological analyses)
#
# ## Key Concepts
#
# - **TimeAxisPath**: Abstract path defining how 1D timeline coordinates map to 2D pixels.
#   No image has a time axis until its path through the image is defined
# - **ImageSource**: Unified interface for images (files, PDFs)
# - **GraphicalSegment**: Source + path + timeline offset
# - **GraphicalBundle**: Collection of sources and segments
# - **GraphicalLoader**: Factory for building bundles
#

# %%
# Install dependencies if needed
# # !pip install "timetoalign[graphical]"

# %%
import math
from pathlib import Path

from timetoalign.loader.graphical import GraphicalLoader

# %% [markdown]
# ## Example 1: Single Image with Multiple Systems
#
# Load a musical score with multiple staff systems (horizontal lines).

# %%
# Create loader
loader = GraphicalLoader(metadata={"source": "Example Score"})

# Add the image
image_path = Path("path/to/score.png")
# idx = loader.add_image(image_path)

# For this example, we'll use mock data
# In real usage, uncomment the line above
idx = 0  # Mock source index

# Add horizontal segments for each staff system
# System 1: x from 50 to 800, y at 100
# loader.add_horizontal_segment(idx, x0=50, x1=800, y=100, name="system_1")

# System 2: x from 50 to 800, y at 250
# loader.add_horizontal_segment(idx, x0=50, x1=800, y=250, name="system_2")

# System 3: x from 50 to 800, y at 400
# loader.add_horizontal_segment(idx, x0=50, x1=800, y=400, name="system_3")

# Build the bundle
# bundle = loader.store

print("Bundle structure:")
# print(f"  Sources: {bundle.n_sources}")
# print(f"  Segments: {bundle.n_segments}")
# print(f"  Total length: {bundle.total_length} pixels")

# %% [markdown]
# ## Example 2: Coordinate Conversion
#
# Convert between 1D timeline coordinates and 2D image coordinates.

# %%
# Timeline coordinate -> Image coordinate
# coord = 400  # 400 pixels into the timeline
# source_idx, (x, y) = bundle.timeline_to_image(coord)

# print(f"Timeline coord {coord} maps to:")
# print(f"  Source image: {source_idx}")
# print(f"  Image position: ({x:.1f}, {y:.1f})")

# Image coordinate -> Timeline coordinate
# coord_back = bundle.image_to_timeline(source_idx, x, y)
# print(f"\nRound-trip: {coord_back:.1f} (should be {coord})")

# %% [markdown]
# ## Example 3: Multiple Images (Separate Pages)
#
# Load a multi-page analysis where each page is a separate image.

# %%
# Create loader for multi-page document
loader2 = GraphicalLoader(metadata={"source": "Multi-page Analysis"})

# Add images and segments
# image_paths = [
#     Path("analysis_page1.jpg"),
#     Path("analysis_page2.jpg"),
#     Path("analysis_page3.jpg"),
# ]

# Segment dimensions for each page (x0, x1, y)
# segments = [
#     (10, 850, 50),   # Page 1: 840 pixels
#     (10, 860, 50),   # Page 2: 850 pixels
#     (10, 840, 50),   # Page 3: 830 pixels
# ]

# for img_path, (x0, x1, y) in zip(image_paths, segments):
#     idx = loader2.add_image(img_path)
#     loader2.add_horizontal_segment(idx, x0=x0, x1=x1, y=y, name=f"page_{idx+1}")

# store2 = loader2.store
# print(f"Multi-page store: {store2.n_sources} sources, {store2.total_length:.0f} total pixels")

# %% [markdown]
# ## Example 4: Creating a Timeline
#
# Convert a GraphicalBundle into a DiscreteGraphicalTimeline.

# %%
# Create timeline from bundle
# timeline = bundle.create_timeline(uid="score_timeline", name="Example Score")

# print(f"Timeline: {timeline.id}")
# print(f"  Length: {timeline.length}")
# print(f"  Unit: {timeline.unit}")
# print(f"  Domain: {timeline.domain}")

# %% [markdown]
# ## Example 5: Visualization
#
# Draw paths and events on images.

# %%
# Draw all segment paths on the source image
# viz_img = bundle.draw_segments_on_source(source_index=0, color=(0, 255, 0), line_width=2)

# Save visualization
# viz_img.save(Path("score_with_paths.png"))

# Draw an event interval (e.g., a note or phrase)
# event_start = 100  # Timeline coordinate
# event_end = 300
# event_images = bundle.draw_interval(event_start, event_end, color=(255, 0, 0), line_width=3)

# print(f"Drew event on {len(event_images)} images")

# %% [markdown]
# ## Advanced: Custom Paths
#
# For non-horizontal timelines, use custom TimeAxisPath classes.
#

# %%
from timetoalign.loader.graphical import (  # noqa: E402
    DiagonalLinePath,
    ParametricPath,
    VerticalLinePath,
)

# Vertical timeline (top to bottom)
vertical_path = VerticalLinePath(x=400, y0=50, y1=600)
print(f"Vertical path length: {vertical_path.length} pixels")

# Diagonal timeline
diagonal_path = DiagonalLinePath(start=(100, 100), end=(700, 500))
print(f"Diagonal path length: {diagonal_path.length:.1f} pixels")

# Spiral timeline (parametric)
# r = a + b*t (Archimedean spiral)
a, b = 50, 10
spiral_path = ParametricPath(
    x_func=lambda t: 400 + (a + b * t) * math.cos(t),
    y_func=lambda t: 400 + (a + b * t) * math.sin(t),
    t_start=0,
    t_end=4 * math.pi,  # 2 full rotations
    samples=1000,
)
print(f"Spiral path length: {spiral_path.length:.1f} pixels")

# %% [markdown]
# ## Paths of Several Pieces
#
# A page of a score is read along its first accolade (system), then along
# the second, and so on. A `PiecewisePath` is such a path: consecutive
# pieces, each starting on the path where the ones before it end. Its
# numbers are the pieces' own, so pieces of integers give exact answers.

# %%
from timetoalign.loader.graphical import (  # noqa: E402
    HorizontalLinePath,
    PiecewisePath,
    TimeAxisPath,
)

# Two accolades of a page, 490 and 400 pixels long
page_path = PiecewisePath(
    (
        HorizontalLinePath(x0=10, x1=500, y=100, tolerance=40),
        HorizontalLinePath(x0=20, x1=420, y=300, tolerance=40),
    )
)
print(f"Offsets: {page_path.offsets}, length: {page_path.length} pixels")
print(f"Coordinate 489 is at {page_path.to_2d(489)}, 490 at {page_path.to_2d(490)}")
print(f"Image point (120, 320) is at {page_path.from_2d(120, 320)}")
print(f"Image point (120, 200) is at {page_path.from_2d(120, 200)}")

# %% [markdown]
# The second piece starts at 490 on the path, so the point (120, 320), 100
# pixels along it, is at 590; a point between the two strips is on no piece.
#
# `TimeAxisPath.from_boxes()` makes the path of a page from the areas drawn
# on it, given in reading order: one horizontal piece per accolade, from
# its boxes' least x to their greatest, at the middle of their y extent,
# with half that extent as `tolerance`. A box begins a new accolade when
# its left edge falls back by more than half the span of all left edges
# (or `accolades=` assigns the boxes). Below, three measures, the third on
# a second line.

# %%
from timetoalign import BoundingBox  # noqa: E402

measures = [
    BoundingBox.from_corners(100, 50, 300, 90, unit="pixels"),
    BoundingBox.from_corners(300, 52, 520, 95, unit="pixels"),
    BoundingBox.from_corners(90, 150, 400, 190, unit="pixels"),
]
measures_path = TimeAxisPath.from_boxes(measures)
for piece in measures_path.pieces:
    print(piece)
print(f"Offsets: {measures_path.offsets}, length: {measures_path.length} pixels")
print(f"The third measure lies on piece {measures_path.piece_of(measures[2])}")

# %% [markdown]
# The first piece spans x 100 to 520 (420 pixels) and y 50 to 95, whose
# middle is 145/2; the second spans x 90 to 400 (310 pixels), so the path is
# 730 pixels long. Pages built of such paths, with the areas as Box events,
# are the subject of
# [How to Represent Areas of Page Images as Boxes](how01_graphical_boxes.ipynb).

# %% [markdown]
# ## Summary
#
# The graphical loader provides:
#
# 1. **Flexible path definitions** - Horizontal, vertical, diagonal, parametric curves, or several pieces in a row
# 2. **Multiple image support** - Single or multi-page analyses
# 3. **Bidirectional conversion** - Timeline ↔ Image coordinates
# 4. **Visualization** - Draw paths and events on images
# 5. **Timeline integration** - Convert bundles to DiscreteGraphicalTimeline
#
# Next: See `how01_thoresen_annotation_transfer.ipynb` for the Thoresen alignment example using these tools.
