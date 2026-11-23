"""Graphical timeline loader module.

This module provides tools for loading and constructing graphical timelines
from images and PDFs. The key components are:

- **TimeAxisPath**: Abstract path defining how 1D timeline coordinates map to 2D pixels.
  Concrete implementations include HorizontalLinePath, VerticalLinePath,
  DiagonalLinePath, and ParametricPath (for curves/spirals). No image has a time
  axis until its path is defined.

- **PiecewisePath**: A path made of consecutive pieces, such as the accolades of a
  score page. ``TimeAxisPath.from_boxes`` makes one from the areas drawn on a page,
  and **group_into_accolades** is the rule that finds the accolades among them.

- **ImageSource**: Unified interface for image data from files, PDF pages, or
  embedded PDF images. Uses pymupdf as backend.

- **GraphicalSegment**: Combines a source reference with a path and timeline offset.

- **GraphicalStore**: Collection of sources and segments forming a complete
  graphical timeline with coordinate conversion and visualization methods.

- **GraphicalLoader**: Factory class for building stores with a fluent interface.

- **IIIFManifestLoader**: Load image metadata from IIIF Presentation API manifests.

- **ScoreWarpLoader**: Read the time axis drawn into a ScoreWarp SVG as a calibrated
  child of the drawing's timeline.

- **DmosLoader**: Read the pages, systems and measures of a DMOS JSON file as page
  timelines made of their systems' accolades, with the measures as Box events.

- **MeiFacsimileLoader**: Read the surfaces and zones of an MEI facsimile as page
  timelines holding Box events, and the ``@facs`` links to them.

Examples:
    >>> from timetoalign.loader.graphical import GraphicalLoader
    >>> loader = GraphicalLoader()
    >>> idx = loader.add_image("score.png")
    >>> loader.add_horizontal_segment(idx, x0=10, x1=500, y=100, name="system_1")
    >>> loader.add_horizontal_segment(idx, x0=10, x1=500, y=200, name="system_2")
    >>> store = loader.store
    >>> timeline = store.create_timeline(uid="score", name="Score")

    >>> # Coordinate conversion
    >>> source_idx, (x, y) = store.timeline_to_image(250.0)
    >>> coord = store.image_to_timeline(source_idx, x, y)

    >>> # Load IIIF manifest for image dimensions
    >>> from timetoalign.loader.graphical import IIIFManifestLoader
    >>> iiif = IIIFManifestLoader()
    >>> iiif.load("manifest.json")
    >>> iiif.dimensions
    {'width': 4096, 'height': 299400}
"""

from __future__ import annotations

# DMOS page layout
from .dmos import DmosLoader

# IIIF manifest loader
from .iiif import IIIFCanvasInfo, IIIFManifestInfo, IIIFManifestLoader

# Loader factory
from .loader import GraphicalLoader

# MEI facsimile surfaces and zones
from .mei_facsimile import MeiFacsimileLoader

# Path definitions
from .paths import (
    DiagonalLinePath,
    HorizontalLinePath,
    ParametricPath,
    PiecewisePath,
    TimeAxisPath,
    VerticalLinePath,
    group_into_accolades,
)

# ScoreWarp SVG time axis
from .scorewarp import ScoreWarpLoader

# Segment and bundle
from .segment import GraphicalSegment

# Image source
from .source import ImageMetadata, ImageSource
from .store import GraphicalStore

__all__ = [
    # Paths
    "TimeAxisPath",
    "HorizontalLinePath",
    "VerticalLinePath",
    "DiagonalLinePath",
    "ParametricPath",
    "PiecewisePath",
    "group_into_accolades",
    # Source
    "ImageSource",
    "ImageMetadata",
    # Segment and store
    "GraphicalSegment",
    "GraphicalStore",
    # Loaders
    "GraphicalLoader",
    "IIIFManifestLoader",
    "IIIFManifestInfo",
    "IIIFCanvasInfo",
    "ScoreWarpLoader",
    "DmosLoader",
    "MeiFacsimileLoader",
]
