"""GraphicalLoader for building graphical timelines.

This module provides a factory class for creating GraphicalStore objects
from images and PDFs. The loader handles:
- Adding image sources (files, PDF pages, embedded images)
- Defining segments with TimeAxisPath geometry
- Building the final store with all components

The loader is completely generic - test-case-specific configuration
belongs in test fixtures, not production code.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

from timetoalign.loader._sources import Source, SourceLike, as_source, describe_source
from timetoalign.loader.base import Loader

from .paths import HorizontalLinePath, TimeAxisPath, VerticalLinePath
from .segment import GraphicalSegment
from .source import ImageSource
from .store import GraphicalStore

if TYPE_CHECKING:
    import pymupdf

    from timetoalign.timelines import DiscreteGraphicalTimeline

module_logger = logging.getLogger(__name__)


class GraphicalLoader(Loader[GraphicalStore]):
    """Factory for building GraphicalStore objects.

    The loader accumulates sources and segments, then builds a store.
    This provides a fluent interface for constructing graphical timelines.

    Every addition is a door into the loader, like a ``load`` call: it seeds
    the loader's ID minter afresh from the previous seed and what it adds
    (an image file's bytes, an image's pixels, a segment's source, offset,
    name and geometry), so :meth:`create_timeline` mints an
    ID that follows the content (see ``Loader.load``).

    Examples:
        >>> loader = GraphicalLoader()
        >>> idx = loader.add_image("score.png")
        >>> loader.add_horizontal_segment(idx, x0=10, x1=500, y=100, name="system_1")
        >>> loader.add_horizontal_segment(idx, x0=10, x1=500, y=200, name="system_2")
        >>> store = loader.store

        >>> # From PDF
        >>> import pymupdf
        >>> doc = pymupdf.open("score.pdf")
        >>> loader = GraphicalLoader()
        >>> idx = loader.add_pdf_page(doc, 0)
        >>> loader.add_horizontal_segment(idx, x0=50, x1=550, y=100, name="page1")

    Attributes:
        metadata: Optional metadata dictionary attached to the store.
    """

    def __init__(self, metadata: dict | None = None, *, seed: int | None = None):
        """Initialize the loader.

        Args:
            metadata: Optional metadata to attach to the store.
            seed: The seed of the loader's ID minter (see ``Loader.load``).
        """
        super().__init__(seed=seed)
        self._sources: list[ImageSource] = []
        self._segments: list[GraphicalSegment] = []
        self._current_offset: float = 0.0
        self._metadata = metadata or {}
        self._logger = module_logger.getChild("GraphicalLoader")

    def _load_source(self, source: Source) -> GraphicalStore:
        """Reject file loading because graphical sources are structural."""
        raise NotImplementedError(
            "GraphicalLoader builds stores through add_* methods."
        )

    # --- Source Addition Methods ---

    def add_image(self, path: SourceLike) -> int:
        """Add an image file as a source.

        Args:
            path: Path to the image file, or its member in a ZIP archive.

        Returns:
            Index of the added source (for use in add_segment).

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If format is not supported.
        """
        source = ImageSource.from_image_file(path)
        self._start_minter(as_source(path))
        idx = len(self._sources)
        self._sources.append(source)
        self._logger.debug(f"Added image source {idx}: {describe_source(path)}")
        return idx

    def add_pdf_page(
        self,
        doc: "pymupdf.Document",
        page_index: int,
        dpi: int = 150,
    ) -> int:
        """Add a PDF page (rendered as image) as a source.

        Args:
            doc: An open pymupdf Document.
            page_index: Zero-based page index.
            dpi: Resolution for rendering.

        Returns:
            Index of the added source.
        """
        source = ImageSource.from_pdf_page(doc, page_index, dpi=dpi)
        self._start_minter(_pixels(source))
        idx = len(self._sources)
        self._sources.append(source)
        self._logger.debug(f"Added PDF page source {idx}: page {page_index}")
        return idx

    def add_pdf_embedded_image(
        self,
        doc: "pymupdf.Document",
        xref: int,
    ) -> int:
        """Add an embedded PDF image as a source.

        Args:
            doc: An open pymupdf Document.
            xref: Cross-reference number of the image.

        Returns:
            Index of the added source.
        """
        source = ImageSource.from_pdf_embedded_image(doc, xref)
        self._start_minter(_pixels(source))
        idx = len(self._sources)
        self._sources.append(source)
        self._logger.debug(f"Added PDF embedded image source {idx}: xref {xref}")
        return idx

    def add_image_source(self, source: ImageSource) -> int:
        """Add a pre-constructed ImageSource.

        Args:
            source: The ImageSource to add.

        Returns:
            Index of the added source.
        """
        self._start_minter(_pixels(source))
        idx = len(self._sources)
        self._sources.append(source)
        self._logger.debug(f"Added pre-constructed source {idx}")
        return idx

    # --- Segment Addition Methods ---

    def add_segment(
        self,
        source_index: int,
        path: TimeAxisPath,
        name: str | None = None,
        offset: float | None = None,
    ) -> GraphicalSegment:
        """Add a segment with arbitrary path.

        Args:
            source_index: Index of the source image.
            path: TimeAxisPath defining the time axis geometry.
            name: Optional human-readable name.
            offset: Timeline offset. If None, appends contiguously.

        Returns:
            The created GraphicalSegment.

        Raises:
            IndexError: If source_index is invalid.
        """
        if source_index < 0 or source_index >= len(self._sources):
            raise IndexError(
                f"Source index {source_index} out of range [0, {len(self._sources)})"
            )

        # Use provided offset or current accumulated offset
        timeline_offset = offset if offset is not None else self._current_offset

        segment = GraphicalSegment(
            source_index=source_index,
            path=path,
            timeline_offset=timeline_offset,
            name=name,
        )

        self._segments.append(segment)
        self._start_minter(_segment_part(segment))

        # Update current offset for contiguous addition
        if offset is None:
            self._current_offset += path.length

        self._logger.debug(
            f"Added segment {name or len(self._segments)-1}: "
            f"source={source_index}, offset={timeline_offset:.1f}, length={path.length:.1f}"
        )

        return segment

    def add_horizontal_segment(
        self,
        source_index: int,
        x0: float,
        x1: float,
        y: float,
        name: str | None = None,
        offset: float | None = None,
        tolerance: float = 10.0,
    ) -> GraphicalSegment:
        """Add a horizontal line segment (convenience method).

        This is the most common case for musical scores and spectrograms.

        Args:
            source_index: Index of the source image.
            x0: Starting x coordinate (left edge).
            x1: Ending x coordinate (right edge).
            y: Fixed y coordinate (vertical position).
            name: Optional human-readable name.
            offset: Timeline offset. If None, appends contiguously.
            tolerance: Distance tolerance for coordinate conversion.

        Returns:
            The created GraphicalSegment.
        """
        path = HorizontalLinePath(x0=x0, x1=x1, y=y, tolerance=tolerance)
        return self.add_segment(source_index, path, name=name, offset=offset)

    def add_vertical_segment(
        self,
        source_index: int,
        x: float,
        y0: float,
        y1: float,
        name: str | None = None,
        offset: float | None = None,
        tolerance: float = 10.0,
    ) -> GraphicalSegment:
        """Add a vertical line segment (convenience method).

        Args:
            source_index: Index of the source image.
            x: Fixed x coordinate.
            y0: Starting y coordinate (top).
            y1: Ending y coordinate (bottom).
            name: Optional human-readable name.
            offset: Timeline offset. If None, appends contiguously.
            tolerance: Distance tolerance for coordinate conversion.

        Returns:
            The created GraphicalSegment.
        """
        path = VerticalLinePath(x=x, y0=y0, y1=y1, tolerance=tolerance)
        return self.add_segment(source_index, path, name=name, offset=offset)

    # --- Builder Methods ---

    def reset_offset(self, offset: float = 0.0) -> None:
        """Reset the current offset for contiguous segment addition.

        Args:
            offset: New offset value (default 0.0).
        """
        self._current_offset = offset

    @property
    def current_offset(self) -> float:
        """Current offset for the next contiguous segment."""
        return self._current_offset

    @property
    def n_sources(self) -> int:
        """Number of sources added."""
        return len(self._sources)

    @property
    def n_segments(self) -> int:
        """Number of segments added."""
        return len(self._segments)

    @property
    def store(self) -> GraphicalStore:
        """Build and return a new ``GraphicalStore``.

        Each call creates an independent copy so that stores obtained
        at different points in the loader's state do not share mutable
        references.

        Returns:
            GraphicalStore containing copies of all sources and segments.
        """
        return GraphicalStore(
            sources=list(self._sources),
            segments=list(self._segments),
            metadata=dict(self._metadata),
        )

    def build(self) -> GraphicalStore:
        """Alias for :pyattr:`store`.

        Returns:
            GraphicalStore containing all sources and segments.
        """
        return self.store

    def create_timeline(
        self, uid: str | None = None, name: str | None = None, **kwargs: Any
    ) -> "DiscreteGraphicalTimeline":
        """Create the timeline of the store the loader builds (:attr:`store`).

        Args:
            uid: The timeline's ID; without one, it is minted by the
                loader's minter, seeded from what was added (see
                :meth:`add_image`).
            name: Optional timeline name.
            **kwargs: Not used; named so the loader's signature is the base's.

        Returns:
            A DiscreteGraphicalTimeline with length = total_length.
        """
        return self.store.create_timeline(uid=uid, name=name, minter=self._id_minter)

    def clear(self) -> None:
        """Clear all sources and segments; the next addition seeds a new minter."""
        super().clear()
        self._segments.clear()
        self._current_offset = 0.0
        self._logger.debug("Cleared loader state")

    def __repr__(self) -> str:
        return (
            f"GraphicalLoader(sources={len(self._sources)}, "
            f"segments={len(self._segments)}, "
            f"offset={self._current_offset:.1f})"
        )


def _pixels(source: ImageSource) -> bytes:
    """What an image contributes to the loader's seed: the SHA-256 digest of its pixels."""
    return hashlib.sha256(source.pixmap.samples).digest()


def _segment_part(segment: GraphicalSegment) -> str:
    """What a segment contributes to the loader's seed.

    Its source, offset and name, its path's class and length, and the image
    points at five evenly spaced positions along the path, which stand for
    the geometry of every path class alike.
    """
    path = segment.path
    length = path.length
    points = [path.to_2d(length * step / 4) for step in range(5)]
    return json.dumps(
        [
            segment.source_index,
            str(segment.timeline_offset),
            segment.name,
            type(path).__name__,
            str(length),
            [[str(x), str(y)] for x, y in points],
        ]
    )
