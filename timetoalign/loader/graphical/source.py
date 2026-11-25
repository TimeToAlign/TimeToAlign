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


def _get_pymupdf() -> ModuleType:
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


def _pixmap_to_image(pixmap: "pymupdf.Pixmap") -> "PIL.Image.Image":
    """Convert a whole pixmap to a new Pillow image in RGB or RGBA to draw on."""
    image = pixmap.pil_image()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image


def _image_to_pixmap(image: "PIL.Image.Image") -> "pymupdf.Pixmap":
    """Convert a whole RGB or RGBA Pillow image to a pixmap from its raw samples.

    A pixmap stores colour multiplied by alpha, as one read from a PNG file
    does, so the colour of an RGBA image is multiplied by its alpha here.
    """
    pymupdf = _get_pymupdf()
    alpha = image.mode == "RGBA"
    pixmap = pymupdf.Pixmap(
        pymupdf.csRGB, image.width, image.height, image.tobytes(), alpha
    )
    if alpha:
        pixmap.set_alpha(image.getchannel("A").tobytes(), premultiply=True)
    return pixmap


def _is_translucent(image: "PIL.Image.Image") -> bool:
    """Whether an image has pixels that are neither transparent nor opaque."""
    if image.mode != "RGBA":
        return False
    return any(image.getchannel("A").histogram()[1:255])


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
        self._pixmap: pymupdf.Pixmap | None = pixmap
        # A drawing whose pixmap is built only when it is needed.
        self._drawing: PIL.Image.Image | None = None
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

    @classmethod
    def _from_drawing(
        cls, image: "PIL.Image.Image", metadata: ImageMetadata, translucent: bool
    ) -> "ImageSource":
        """Wrap a finished drawing, which stays an image until its pixmap is needed.

        A ``translucent`` drawing (see :class:`_Canvas`) becomes its pixmap at
        once, so that drawing on it again starts, as it always did, from the
        pixmap's colour.
        """
        if translucent:
            return cls(_image_to_pixmap(image), metadata)
        source = cls.__new__(cls)
        source._pixmap = None
        source._drawing = image
        source._metadata = metadata
        source._logger = module_logger.getChild("ImageSource")
        return source

    # --- Properties ---

    @property
    def width(self) -> int:
        """Image width in pixels."""
        if self._pixmap is None:
            return self._pending_drawing().width
        return self._pixmap.width

    @property
    def height(self) -> int:
        """Image height in pixels."""
        if self._pixmap is None:
            return self._pending_drawing().height
        return self._pixmap.height

    @property
    def metadata(self) -> ImageMetadata:
        """Image metadata."""
        return self._metadata

    @property
    def pixmap(self) -> pymupdf.Pixmap:
        """Underlying pymupdf Pixmap."""
        pixmap = self._materialized()
        # The pixmap handed out may be changed in place: it is the image now.
        self._drawing = None
        return pixmap

    # --- Conversion Methods ---

    def _pending_drawing(self) -> "PIL.Image.Image":
        """The drawing of a source whose pixmap is not built yet.

        A source holds a pixmap, a drawing, or both: it is made from a pixmap
        or from a drawing, and loses its drawing only once a pixmap is built.
        """
        assert self._drawing is not None
        return self._drawing

    def _materialized(self) -> "pymupdf.Pixmap":
        """The pixmap, built once from a pending drawing."""
        if self._pixmap is None:
            self._pixmap = _image_to_pixmap(self._pending_drawing())
        return self._pixmap

    def _drawable(self) -> "PIL.Image.Image":
        """A new RGB or RGBA Pillow image of this source, to draw on."""
        if self._drawing is not None:
            return self._drawing.copy()
        return _pixmap_to_image(self._materialized())

    def to_pil(self) -> "PIL.Image.Image":
        """Convert to PIL Image for advanced operations.

        Requires pillow to be installed.

        Returns:
            PIL Image object.
        """
        return self._materialized().pil_image()

    def to_bytes(self, format: str = "png") -> bytes:
        """Convert to bytes in specified format.

        Args:
            format: Output format ("png", "jpeg", "ppm", etc.).

        Returns:
            Image data as bytes.
        """
        return self._materialized().tobytes(output=format)

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

        self._materialized().save(str(path), output=format)
        self._logger.debug(f"Saved image to {path}")

    # --- Visualization Methods ---

    def copy(self) -> "ImageSource":
        """Create a copy of this ImageSource.

        Returns:
            New ImageSource with copied pixmap.
        """
        pymupdf = _get_pymupdf()

        # Create new pixmap with same properties
        new_pix = pymupdf.Pixmap(self._materialized())

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
        canvas = _Canvas(self)
        canvas.rectangle(x, y, width, height, color, line_width, fill)
        return canvas.result()

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
        canvas = _Canvas(self)
        canvas.line(x0, y0, x1, y1, color, line_width)
        return canvas.result()

    def draw_path(
        self,
        path: "TimeAxisPath",
        color: tuple[int, int, int] = (0, 255, 0),
        line_width: int = 2,
        samples: int = 100,
    ) -> "ImageSource":
        """Return new ImageSource with a TimeAxisPath drawn.

        A :class:`~timetoalign.loader.graphical.paths.PiecewisePath` is
        drawn piece by piece: no line joins the end of one piece (an
        accolade) to the start of the next.

        Args:
            path: The TimeAxisPath to draw.
            color: RGB color tuple.
            line_width: Line width in pixels.
            samples: Number of line segments for curves (per piece of a
                ``PiecewisePath``).

        Returns:
            New ImageSource with path drawn.
        """
        canvas = _Canvas(self)
        canvas.path(path, color, line_width, samples)
        return canvas.result()

    def __repr__(self) -> str:
        return (
            f"ImageSource({self.width}x{self.height}, "
            f"source={self._metadata.source_type})"
        )


# endregion

# region Drawing


class _Canvas:
    """A copy of an ImageSource that any number of shapes are drawn on.

    The page is converted to a Pillow image once, before the first shape,
    and :meth:`result` wraps the drawing in a new ImageSource whose pixmap is
    built once, when it is needed; the source is never changed.

    A pixmap stores colour multiplied by alpha, so every conversion out and
    back in multiplies a semi-transparent pixel once more. On a translucent
    image (one with such pixels, or with a filled rectangle drawn) the image
    is therefore converted out and back in between two shapes: the pixels
    are those of drawing each shape on the pixmap the one before left. On
    any other image the conversion changes no pixel and is left out.
    """

    def __init__(self, source: ImageSource) -> None:
        self._image_draw = _image_draw()
        self._source = source
        self._image: PIL.Image.Image | None = None
        self._translucent = False

    def _next_image(self, color: tuple[int, ...]) -> "PIL.Image.Image":
        """The image the next shape, drawn in ``color``, is drawn on."""
        if self._image is None:
            self._image = self._source._drawable()
            self._translucent = _is_translucent(self._image)
        elif self._translucent:
            self._image = _pixmap_to_image(_image_to_pixmap(self._image))
        if len(color) > 3:
            self._translucent = True
        return self._image

    def _draw(self, color: tuple[int, ...]) -> Any:
        """The drawing context for the next shape, drawn in ``color``."""
        return self._image_draw.Draw(self._next_image(color))

    def rectangle(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[int, int, int],
        line_width: int,
        fill: bool,
    ) -> None:
        """Draw a rectangle outline, filled at alpha 64 when ``fill``."""
        image = self._next_image(color)
        draw = self._image_draw.Draw(image)
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + width), int(y + height)
        if fill:
            if image.mode != "RGBA":
                self._image = image.convert("RGBA")
                draw = self._image_draw.Draw(self._image)
            self._translucent = True
            draw.rectangle(
                [x0, y0, x1, y1], fill=(*color, 64), outline=color, width=line_width
            )
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: tuple[int, int, int],
        line_width: int,
    ) -> None:
        """Draw a straight line."""
        self._draw(color).line(
            [(int(x0), int(y0)), (int(x1), int(y1))], fill=color, width=line_width
        )

    def path(
        self,
        path: "TimeAxisPath",
        color: tuple[int, int, int],
        line_width: int,
        samples: int,
    ) -> None:
        """Draw a time path, a ``PiecewisePath`` piece by piece."""
        from .paths import PiecewisePath

        draw = self._draw(color)
        pieces = path.pieces if isinstance(path, PiecewisePath) else (path,)
        for piece in pieces:
            points = []
            for i in range(samples + 1):
                t = (i / samples) * piece.length
                x, y = piece.to_2d(t)
                points.append((int(x), int(y)))
            if len(points) >= 2:
                draw.line(points, fill=color, width=line_width)

    def result(self) -> ImageSource:
        """A new ImageSource of the drawing; a copy of the source if none was drawn."""
        if self._image is None:
            return self._source.copy()
        return ImageSource._from_drawing(
            self._image, self._source.metadata, self._translucent
        )


# endregion
