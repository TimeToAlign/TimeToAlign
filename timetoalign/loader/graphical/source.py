"""ImageSource for graphical timelines using pymupdf.

This module provides a unified interface for loading images from:
- Standalone image files (JPEG, PNG, etc.)
- PDF pages (rendered as images)
- Embedded images in PDFs

pymupdf (fitz) is used as the backend because it handles both
images and PDFs with excellent performance and a consistent API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from timetoalign.loader._optional import missing_dependency

module_logger = logging.getLogger(__name__)

# Lazy import pymupdf to allow module to load even if not installed
if TYPE_CHECKING:
    import PIL.Image
    import pymupdf

    from .paths import TimeAxisPath


def _get_pymupdf() -> "pymupdf":
    """Lazy import of pymupdf, naming the extra that installs it."""
    try:
        import pymupdf
    except ModuleNotFoundError as error:
        raise ImportError(
            missing_dependency("ImageSource", error.name, "graphical")
        ) from error
    return pymupdf


def _image_draw() -> ModuleType:
    """Lazy import of pillow's ``ImageDraw``, naming the extra that installs it."""
    try:
        from PIL import ImageDraw
    except ModuleNotFoundError as error:
        raise ImportError(
            missing_dependency("Drawing on an ImageSource", error.name, "graphical")
        ) from error
    return ImageDraw


# region ImageMetadata


@dataclass
class ImageMetadata:
    """Metadata for an image source.

    Attributes:
        width: Image width in pixels.
        height: Image height in pixels.
        source_type: Type of source ("file", "pdf_page", "pdf_embedded").
        source_path: Path to source file (if applicable).
        page_index: PDF page index (if from PDF).
        xref: PDF xref number (if embedded image).
        dpi: Resolution in dots per inch (if known).
        colorspace: Color space name (e.g., "RGB", "Gray").
        extra: Additional metadata.
    """

    width: int
    height: int
    source_type: str
    source_path: Path | None = None
    page_index: int | None = None
    xref: int | None = None
    dpi: tuple[float, float] | None = None
    colorspace: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# endregion


# region ImageSource


class ImageSource:
    """Image data source using pymupdf.

    Provides a unified interface for image data regardless of source:
    - Standalone images (JPEG, PNG, BMP, TIFF, etc.)
    - PDF pages rendered as images
    - Images embedded within PDFs

    The underlying pymupdf.Pixmap is stored and provides:
    - Width, height, colorspace information
    - Pixel data access
    - Conversion to PIL.Image for advanced operations
    - Drawing capabilities via Shape

    Examples:
        >>> # From image file
        >>> source = ImageSource.from_image_file(Path("diagram.png"))
        >>> print(source.width, source.height)

        >>> # From PDF page
        >>> import pymupdf
        >>> doc = pymupdf.open("document.pdf")
        >>> source = ImageSource.from_pdf_page(doc, page_index=0)

        >>> # Save modified image
        >>> source.save(Path("output.png"))
    """

    def __init__(
        self,
        pixmap: "pymupdf.Pixmap",
        metadata: ImageMetadata,
    ):
        """Initialize ImageSource.

        Args:
            pixmap: The pymupdf Pixmap containing image data.
            metadata: Metadata about the image source.
        """
        self._pixmap = pixmap
        self._metadata = metadata
        self._logger = module_logger.getChild("ImageSource")

    @classmethod
    def from_image_file(cls, path: Path | str) -> "ImageSource":
        """Load from a standalone image file.

        Supports: JPEG, PNG, BMP, TIFF, GIF, PNM, PAM, JXR, JPX.

        Args:
            path: Path to the image file.

        Returns:
            ImageSource wrapping the loaded image.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file format is not supported.
        """
        pymupdf = _get_pymupdf()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        try:
            pix = pymupdf.Pixmap(str(path))
        except Exception as e:
            raise ValueError(f"Failed to load image from {path}: {e}") from e

        metadata = ImageMetadata(
            width=pix.width,
            height=pix.height,
            source_type="file",
            source_path=path,
            dpi=(pix.xres, pix.yres) if pix.xres > 0 else None,
            colorspace=pix.colorspace.name if pix.colorspace else None,
        )

        return cls(pix, metadata)

    @classmethod
    def from_pdf_page(
        cls,
        doc: "pymupdf.Document",
        page_index: int,
        dpi: int = 150,
    ) -> "ImageSource":
        """Render a PDF page as an image.

        Args:
            doc: An open pymupdf Document.
            page_index: Zero-based page index.
            dpi: Resolution for rendering (default 150).

        Returns:
            ImageSource containing the rendered page.

        Raises:
            IndexError: If page_index is out of range.
        """
        pymupdf = _get_pymupdf()

        if page_index < 0 or page_index >= len(doc):
            raise IndexError(
                f"Page index {page_index} out of range [0, {len(doc) - 1}]"
            )

        page = doc[page_index]
        # Scale matrix: 72 dpi is the PDF default
        mat = pymupdf.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)

        metadata = ImageMetadata(
            width=pix.width,
            height=pix.height,
            source_type="pdf_page",
            source_path=Path(doc.name) if doc.name else None,
            page_index=page_index,
            dpi=(dpi, dpi),
            colorspace=pix.colorspace.name if pix.colorspace else None,
            extra={"page_rect": tuple(page.rect)},
        )

        return cls(pix, metadata)

    @classmethod
    def from_pdf_embedded_image(
        cls,
        doc: "pymupdf.Document",
        xref: int,
    ) -> "ImageSource":
        """Extract an embedded image from a PDF.

        Args:
            doc: An open pymupdf Document.
            xref: Cross-reference number of the image object.

        Returns:
            ImageSource containing the extracted image.

        Raises:
            ValueError: If xref doesn't refer to a valid image.
        """
        pymupdf = _get_pymupdf()

        try:
            pix = pymupdf.Pixmap(doc, xref)
        except Exception as e:
            raise ValueError(f"Failed to extract image at xref {xref}: {e}") from e

        metadata = ImageMetadata(
            width=pix.width,
            height=pix.height,
            source_type="pdf_embedded",
            source_path=Path(doc.name) if doc.name else None,
            xref=xref,
            dpi=(pix.xres, pix.yres) if pix.xres > 0 else None,
            colorspace=pix.colorspace.name if pix.colorspace else None,
        )

        return cls(pix, metadata)

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        source_name: str = "memory",
    ) -> "ImageSource":
        """Create from in-memory image data.

        Args:
            data: Raw image bytes (JPEG, PNG, etc.).
            source_name: Name for metadata.

        Returns:
            ImageSource wrapping the image.
        """
        pymupdf = _get_pymupdf()

        pix = pymupdf.Pixmap(data)

        metadata = ImageMetadata(
            width=pix.width,
            height=pix.height,
            source_type="bytes",
            extra={"source_name": source_name},
        )

        return cls(pix, metadata)

    # --- Properties ---

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self._pixmap.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self._pixmap.height

    @property
    def metadata(self) -> ImageMetadata:
        """Image metadata."""
        return self._metadata

    @property
    def pixmap(self) -> pymupdf.Pixmap:
        """Underlying pymupdf Pixmap."""
        return self._pixmap

    # --- Conversion Methods ---

    def to_pil(self) -> "PIL.Image.Image":
        """Convert to PIL Image for advanced operations.

        Requires pillow to be installed.

        Returns:
            PIL Image object.
        """
        return self._pixmap.pil_image()

    def to_bytes(self, format: str = "png") -> bytes:
        """Convert to bytes in specified format.

        Args:
            format: Output format ("png", "jpeg", "ppm", etc.).

        Returns:
            Image data as bytes.
        """
        return self._pixmap.tobytes(output=format)

    # --- I/O Methods ---

    def save(self, path: Path | str, format: str | None = None) -> None:
        """Save image to file.

        Args:
            path: Output file path.
            format: Output format. If None, inferred from extension.
        """
        path = Path(path)

        if format is None:
            ext = path.suffix.lower().lstrip(".")
            format = ext if ext else "png"

        self._pixmap.save(str(path), output=format)
        self._logger.debug(f"Saved image to {path}")

    # --- Visualization Methods ---

    def copy(self) -> "ImageSource":
        """Create a copy of this ImageSource.

        Returns:
            New ImageSource with copied pixmap.
        """
        pymupdf = _get_pymupdf()

        # Create new pixmap with same properties
        new_pix = pymupdf.Pixmap(self._pixmap)

        return ImageSource(new_pix, self._metadata)

    def draw_rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[int, int, int] = (255, 0, 0),
        line_width: int = 2,
        fill: bool = False,
    ) -> "ImageSource":
        """Return new ImageSource with rectangle drawn.

        Uses PIL for drawing since pymupdf Pixmap doesn't have
        direct drawing methods.

        Args:
            x: Left edge x coordinate.
            y: Top edge y coordinate.
            width: Rectangle width.
            height: Rectangle height.
            color: RGB color tuple.
            line_width: Stroke width in pixels.
            fill: If True, fill the rectangle (with alpha).

        Returns:
            New ImageSource with rectangle drawn.
        """
        ImageDraw = _image_draw()

        # Convert to PIL
        pil_img = self.to_pil()

        # Ensure we have an RGB or RGBA image
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        draw = ImageDraw.Draw(pil_img)

        x0, y0 = int(x), int(y)
        x1, y1 = int(x + width), int(y + height)

        if fill:
            # Semi-transparent fill
            fill_color = (*color, 64)
            if pil_img.mode != "RGBA":
                pil_img = pil_img.convert("RGBA")
                draw = ImageDraw.Draw(pil_img)
            draw.rectangle(
                [x0, y0, x1, y1], fill=fill_color, outline=color, width=line_width
            )
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

        # Convert back to pymupdf Pixmap
        pymupdf = _get_pymupdf()

        # Save to bytes and reload
        import io

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        new_pix = pymupdf.Pixmap(buf.getvalue())

        return ImageSource(new_pix, self._metadata)

    def draw_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int] = (0, 255, 0),
        line_width: int = 2,
    ) -> "ImageSource":
        """Return new ImageSource with line drawn.

        Args:
            x0, y0: Start point.
            x1, y1: End point.
            color: RGB color tuple.
            line_width: Line width in pixels.

        Returns:
            New ImageSource with line drawn.
        """
        ImageDraw = _image_draw()

        pil_img = self.to_pil()
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        draw = ImageDraw.Draw(pil_img)
        draw.line(
            [(int(x0), int(y0)), (int(x1), int(y1))],
            fill=color,
            width=line_width,
        )

        pymupdf = _get_pymupdf()
        import io

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        new_pix = pymupdf.Pixmap(buf.getvalue())

        return ImageSource(new_pix, self._metadata)

    def draw_path(
        self,
        path: "TimeAxisPath",
        color: tuple[int, int, int] = (0, 255, 0),
        line_width: int = 2,
        samples: int = 100,
    ) -> "ImageSource":
        """Return new ImageSource with a TimeAxisPath drawn.

        Args:
            path: The TimeAxisPath to draw.
            color: RGB color tuple.
            line_width: Line width in pixels.
            samples: Number of line segments for curves.

        Returns:
            New ImageSource with path drawn.
        """
        ImageDraw = _image_draw()

        pil_img = self.to_pil()
        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        draw = ImageDraw.Draw(pil_img)

        # Sample the path and draw line segments
        points = []
        for i in range(samples + 1):
            t = (i / samples) * path.length
            x, y = path.to_2d(t)
            points.append((int(x), int(y)))

        if len(points) >= 2:
            draw.line(points, fill=color, width=line_width)

        pymupdf = _get_pymupdf()
        import io

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)

        new_pix = pymupdf.Pixmap(buf.getvalue())

        return ImageSource(new_pix, self._metadata)

    def __repr__(self) -> str:
        return (
            f"ImageSource({self.width}x{self.height}, "
            f"source={self._metadata.source_type})"
        )


# endregion
