"""Tests for canonical graphical segment bounding boxes."""

from __future__ import annotations

from timetoalign.core.events import BoundingBox
from timetoalign.loader.graphical.paths import HorizontalLinePath
from timetoalign.loader.graphical.segment import GraphicalSegment


def test_get_image_bounds_returns_bounding_box() -> None:
    """Path bounds are a box in whole pixels, rounded outward."""
    segment = GraphicalSegment(
        source_index=0,
        path=HorizontalLinePath(x0=10, x1=30, y=5),
        region=(100, 200, 500, 300),
    )

    assert segment.get_image_bounds() == BoundingBox.from_corners(
        110, 205, 130, 205, unit="pixels"
    )


def test_get_image_bounds_rounds_sampled_bounds_outward() -> None:
    """Fractional sampled bounds widen to whole pixels: floor below, ceiling above.

    The path runs from x 110.5 to 129.5 at y 205.25 in the image, so the
    smallest box of whole pixels holding every sampled point is x 110 to 130,
    y 205 to 206; rounding to the nearest pixel would cut the path off.
    """
    segment = GraphicalSegment(
        source_index=0,
        path=HorizontalLinePath(x0=10.5, x1=29.5, y=5.25),
        region=(100, 200, 500, 300),
    )

    assert segment.get_image_bounds() == BoundingBox.from_xywh(
        110, 205, 20, 1, unit="pixels"
    )
