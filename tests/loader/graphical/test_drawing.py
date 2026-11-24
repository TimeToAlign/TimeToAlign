"""Drawing on page images: one conversion each way, the pixels of shape-by-shape drawing.

The validation logic is in ``README.md`` ("Drawing on a page image").
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

pymupdf = pytest.importorskip("pymupdf")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from timetoalign.loader.graphical import source as source_module  # noqa: E402
from timetoalign.loader.graphical.paths import (  # noqa: E402
    DiagonalLinePath,
    HorizontalLinePath,
    PiecewisePath,
    VerticalLinePath,
)
from timetoalign.loader.graphical.segment import GraphicalSegment  # noqa: E402
from timetoalign.loader.graphical.source import ImageSource  # noqa: E402
from timetoalign.loader.graphical.store import GraphicalStore  # noqa: E402
from timetoalign.testdata import ensure_data  # noqa: E402

IEEE1599 = ensure_data("ieee1599")
#: A greyscale scan (JPEG, 1248 x 1745 at 600 dpi).
SERIE_PAGE = IEEE1599 / "Serie in 9_8/score_files/serie_in_9_8/serie_9_8_2351_0011b.jpg"
#: An RGBA page (PNG, 475 x 638 at 300 dpi) with semi-transparent pixels.
SATIE_PAGE = (
    IEEE1599
    / "SatiePetriNets/ieee1599/score/Montreal/IMSLP01599-Satie_Gymnopedies-1.png"
)

#: The nine measure zones of a chorale page (ulx, uly, lrx, lry).
CHORALE_ZONES = [
    (490, 344, 973, 877),
    (944, 368, 1523, 864),
    (1481, 388, 2082, 912),
    (2034, 402, 2694, 884),
    (2669, 389, 3296, 916),
    (498, 858, 1496, 1337),
    (1465, 885, 2146, 1353),
    (2110, 886, 2748, 1357),
    (2704, 886, 3302, 1387),
]


# region The reference: one PNG round trip per shape


def _png_round_trip(image: Image.Image) -> pymupdf.Pixmap:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return pymupdf.Pixmap(buffer.getvalue())


def _reference_image(source: ImageSource) -> Image.Image:
    image = source.pixmap.pil_image()
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    return image


def reference_rectangle(
    source: ImageSource,
    x: float,
    y: float,
    width: float,
    height: float,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 2,
    fill: bool = False,
) -> ImageSource:
    image = _reference_image(source)
    draw = ImageDraw.Draw(image)
    x0, y0 = int(x), int(y)
    x1, y1 = int(x + width), int(y + height)
    if fill:
        if image.mode != "RGBA":
            image = image.convert("RGBA")
            draw = ImageDraw.Draw(image)
        draw.rectangle(
            [x0, y0, x1, y1], fill=(*color, 64), outline=color, width=line_width
        )
    else:
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
    return ImageSource(_png_round_trip(image), source.metadata)


def reference_line(
    source: ImageSource,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
) -> ImageSource:
    image = _reference_image(source)
    ImageDraw.Draw(image).line(
        [(int(x0), int(y0)), (int(x1), int(y1))], fill=color, width=line_width
    )
    return ImageSource(_png_round_trip(image), source.metadata)


def reference_path(
    source: ImageSource,
    path,
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
    samples: int = 100,
) -> ImageSource:
    image = _reference_image(source)
    draw = ImageDraw.Draw(image)
    pieces = path.pieces if isinstance(path, PiecewisePath) else (path,)
    for piece in pieces:
        points = []
        for i in range(samples + 1):
            x, y = piece.to_2d((i / samples) * piece.length)
            points.append((int(x), int(y)))
        draw.line(points, fill=color, width=line_width)
    return ImageSource(_png_round_trip(image), source.metadata)


def reference_segments(
    store: GraphicalStore,
    source_index: int,
    color: tuple[int, int, int] = (0, 255, 0),
    line_width: int = 2,
) -> ImageSource:
    result = store.sources[source_index].copy()
    for seg in store.get_segments_for_source(source_index):
        result = reference_path(result, seg.path, color=color, line_width=line_width)
    return result


def reference_interval(
    store: GraphicalStore,
    start: float,
    end: float,
    color: tuple[int, int, int] = (255, 0, 0),
    line_width: int = 3,
) -> list[ImageSource]:
    affected: dict[int, ImageSource] = {}
    for seg in store.segments:
        overlap_start = max(start, seg.timeline_offset)
        overlap_end = min(end, seg.timeline_end)
        if overlap_start >= overlap_end:
            continue
        index = seg.source_index
        if index not in affected:
            affected[index] = store.sources[index].copy()
        samples = max(10, int(overlap_end - overlap_start))
        points = []
        for i in range(samples + 1):
            t = overlap_start + (i / samples) * (overlap_end - overlap_start)
            points.append(seg.to_image(t)[1])
        image = affected[index]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            image = reference_line(image, x0, y0, x1, y1, color, line_width)
        affected[index] = image
    return [
        affected[i] if i in affected else source.copy()
        for i, source in enumerate(store.sources)
    ]


# endregion

# region Helpers


def _page(tmp_path: Path, mode: str, size: tuple[int, int], name: str) -> ImageSource:
    """A synthetic page: a background with a darker block, written as PNG."""
    background = {"L": 230, "RGB": (240, 235, 220)}[mode]
    block = {"L": 40, "RGB": (30, 40, 120)}[mode]
    image = Image.new(mode, size, background)
    ImageDraw.Draw(image).rectangle([5, 5, size[0] // 3, size[1] // 4], fill=block)
    file = tmp_path / f"{name}.png"
    image.save(file)
    return ImageSource.from_image_file(file)


def _pixels(source: ImageSource) -> tuple:
    pixmap = source.pixmap
    colorspace = pixmap.colorspace.name if pixmap.colorspace else None
    return (
        pixmap.width,
        pixmap.height,
        pixmap.n,
        pixmap.alpha,
        colorspace,
        pixmap.xres,
        pixmap.yres,
        pixmap.samples,
    )


class Spy:
    """Counts the whole-page conversions of the drawing module."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.into = 0
        self.out = 0
        to_image = source_module._pixmap_to_image
        to_pixmap = source_module._image_to_pixmap

        def counted_to_image(pixmap):
            self.into += 1
            return to_image(pixmap)

        def counted_to_pixmap(image):
            self.out += 1
            return to_pixmap(image)

        monkeypatch.setattr(source_module, "_pixmap_to_image", counted_to_image)
        monkeypatch.setattr(source_module, "_image_to_pixmap", counted_to_pixmap)

    @property
    def counts(self) -> tuple[int, int]:
        return (self.into, self.out)


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spy]:
    yield Spy(monkeypatch)


def _store(source: ImageSource, width: int, height: int) -> GraphicalStore:
    """Four segments on one page: three lines of music and a diagonal."""
    third = height // 4
    segments = [
        GraphicalSegment(0, HorizontalLinePath(x0=10, x1=width - 10, y=third)),
        GraphicalSegment(0, HorizontalLinePath(x0=12, x1=width - 14, y=2 * third)),
        GraphicalSegment(
            0,
            PiecewisePath(
                (
                    HorizontalLinePath(x0=20, x1=width // 2, y=3 * third),
                    HorizontalLinePath(
                        x0=width // 2 + 5, x1=width - 20, y=3 * third + 4
                    ),
                )
            ),
        ),
        GraphicalSegment(
            0, DiagonalLinePath(start=(3, 3), end=(width - 3, height - 3))
        ),
    ]
    offset = 0.0
    for seg in segments:
        seg.timeline_offset = offset
        offset += seg.length
    return GraphicalStore(sources=[source], segments=segments)


# endregion

# region Pixels equal to the reference

SHAPES: list[tuple[str, Callable, Callable]] = [
    (
        "rectangle",
        lambda s: s.draw_rectangle(7.6, 9.2, 40.5, 21.9),
        lambda s: reference_rectangle(s, 7.6, 9.2, 40.5, 21.9),
    ),
    (
        "filled rectangle",
        lambda s: s.draw_rectangle(15, 12, 50, 30, (0, 0, 255), 3, True),
        lambda s: reference_rectangle(s, 15, 12, 50, 30, (0, 0, 255), 3, True),
    ),
    (
        "line",
        lambda s: s.draw_line(2.7, 90.1, 97.9, 3.3, (10, 200, 30), 4),
        lambda s: reference_line(s, 2.7, 90.1, 97.9, 3.3, (10, 200, 30), 4),
    ),
    (
        "piecewise path",
        lambda s: s.draw_path(
            PiecewisePath(
                (
                    HorizontalLinePath(x0=10, x1=90, y=20),
                    VerticalLinePath(x=50, y0=30, y1=95),
                )
            ),
            line_width=1,
        ),
        lambda s: reference_path(
            s,
            PiecewisePath(
                (
                    HorizontalLinePath(x0=10, x1=90, y=20),
                    VerticalLinePath(x=50, y0=30, y1=95),
                )
            ),
            line_width=1,
        ),
    ),
    (
        "chain with a fill in the middle",
        lambda s: s.draw_line(0, 0, 99, 99)
        .draw_rectangle(20, 20, 40, 40, fill=True)
        .draw_line(0, 99, 99, 0, (0, 0, 0), 3)
        .draw_rectangle(30, 5, 10, 80),
        lambda s: reference_rectangle(
            reference_line(
                reference_rectangle(
                    reference_line(s, 0, 0, 99, 99), 20, 20, 40, 40, fill=True
                ),
                0,
                99,
                99,
                0,
                (0, 0, 0),
                3,
            ),
            30,
            5,
            10,
            80,
        ),
    ),
]


@pytest.mark.parametrize("mode", ["L", "RGB"])
@pytest.mark.parametrize(
    ("draw", "reference"),
    [(draw, reference) for _, draw, reference in SHAPES],
    ids=[name for name, _, _ in SHAPES],
)
def test_synthetic_page_pixels_equal_the_reference(
    tmp_path: Path, mode: str, draw: Callable, reference: Callable
) -> None:
    page = _page(tmp_path, mode, (100, 100), "page")
    assert _pixels(draw(page)) == _pixels(reference(page))


@pytest.mark.parametrize("file", [SERIE_PAGE, SATIE_PAGE], ids=["serie", "satie"])
def test_real_page_pixels_equal_the_reference(file: Path) -> None:
    page = ImageSource.from_image_file(file)
    drawn = (
        page.draw_rectangle(120.4, 300.7, 200, 150)
        .draw_line(40, 500, 400, 520, (0, 0, 255), 3)
        .draw_rectangle(60, 60, 90, 120, (0, 128, 0), 2, True)
        .draw_path(HorizontalLinePath(x0=30, x1=440, y=610))
    )
    expected = reference_path(
        reference_rectangle(
            reference_line(
                reference_rectangle(page, 120.4, 300.7, 200, 150),
                40,
                500,
                400,
                520,
                (0, 0, 255),
                3,
            ),
            60,
            60,
            90,
            120,
            (0, 128, 0),
            2,
            True,
        ),
        HorizontalLinePath(x0=30, x1=440, y=610),
    )
    assert _pixels(drawn) == _pixels(expected)


@pytest.mark.parametrize("file", [SERIE_PAGE, SATIE_PAGE], ids=["serie", "satie"])
def test_real_page_segments_equal_the_reference(file: Path) -> None:
    page = ImageSource.from_image_file(file)
    store = _store(page, 470, 630)
    assert _pixels(store.draw_segments_on_source(0)) == _pixels(
        reference_segments(store, 0)
    )


@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_synthetic_store_segments_equal_the_reference(
    tmp_path: Path, mode: str
) -> None:
    store = _store(_page(tmp_path, mode, (120, 90), "page"), 120, 90)
    drawn = store.draw_segments_on_source(0, color=(200, 0, 200), line_width=3)
    expected = reference_segments(store, 0, color=(200, 0, 200), line_width=3)
    assert _pixels(drawn) == _pixels(expected)


def test_interval_on_one_of_three_pages_equals_the_reference(tmp_path: Path) -> None:
    first = _page(tmp_path, "L", (80, 60), "first")
    second = _page(tmp_path, "RGB", (80, 60), "second")
    third = _page(tmp_path, "RGB", (80, 60), "third")
    store = GraphicalStore(
        sources=[first, second, third],
        segments=[
            GraphicalSegment(
                0, HorizontalLinePath(x0=5, x1=75, y=20), timeline_offset=0
            ),
            GraphicalSegment(
                0, HorizontalLinePath(x0=5, x1=75, y=45), timeline_offset=70
            ),
            GraphicalSegment(
                1, DiagonalLinePath(start=(0, 0), end=(60, 45)), timeline_offset=140
            ),
        ],
    )
    drawn = store.draw_interval(141.5, 190.25)
    expected = reference_interval(store, 141.5, 190.25)
    assert [_pixels(source) for source in drawn] == [
        _pixels(source) for source in expected
    ]


def test_interval_on_the_rgba_page_equals_the_reference() -> None:
    store = GraphicalStore(
        sources=[ImageSource.from_image_file(SATIE_PAGE)],
        segments=[GraphicalSegment(0, HorizontalLinePath(x0=50, x1=400, y=200))],
    )
    drawn = store.draw_interval(100.0, 112.0, line_width=2)
    expected = reference_interval(store, 100.0, 112.0, line_width=2)
    assert [_pixels(source) for source in drawn] == [
        _pixels(source) for source in expected
    ]


# endregion

# region Conversions counted


def test_one_shape_converts_once_each_way(tmp_path: Path, spy: Spy) -> None:
    drawn = _page(tmp_path, "RGB", (100, 100), "page").draw_line(0, 0, 50, 50)
    assert spy.counts == (1, 0)
    drawn.pixmap
    assert spy.counts == (1, 1)
    drawn.pixmap
    drawn.to_bytes()
    assert spy.counts == (1, 1)


def test_a_chain_converts_once_each_way(tmp_path: Path, spy: Spy) -> None:
    page = _page(tmp_path, "L", (3400, 1400), "chorale")
    drawn = page
    for ulx, uly, lrx, lry in CHORALE_ZONES:
        drawn = drawn.draw_rectangle(ulx, uly, lrx - ulx, lry - uly)
    for ulx, uly, lrx, lry in CHORALE_ZONES:
        drawn = drawn.draw_line(ulx, uly, lrx, lry).draw_line(ulx, lry, lrx, uly)
    drawn.save(tmp_path / "drawn.png")
    assert spy.counts == (1, 1)


def test_store_segments_convert_once_each_way(tmp_path: Path, spy: Spy) -> None:
    store = _store(_page(tmp_path, "RGB", (120, 90), "page"), 120, 90)
    store.draw_segments_on_source(0).pixmap
    assert spy.counts == (1, 1)


def test_store_interval_converts_once_each_way(tmp_path: Path, spy: Spy) -> None:
    pages = [_page(tmp_path, "RGB", (80, 60), name) for name in ("a", "b", "c")]
    store = GraphicalStore(
        sources=pages,
        segments=[
            GraphicalSegment(
                0, HorizontalLinePath(x0=5, x1=75, y=20), timeline_offset=0
            ),
            GraphicalSegment(
                0, HorizontalLinePath(x0=5, x1=75, y=45), timeline_offset=70
            ),
            GraphicalSegment(
                1, HorizontalLinePath(x0=5, x1=75, y=30), timeline_offset=140
            ),
        ],
    )
    drawn = store.draw_interval(72.0, 139.5)
    assert spy.counts == (1, 0)
    for source in drawn:
        source.pixmap
    assert spy.counts == (1, 1)


def test_an_rgba_page_converts_once_per_shape(spy: Spy) -> None:
    store = _store(ImageSource.from_image_file(SATIE_PAGE), 470, 630)
    store.draw_segments_on_source(0).pixmap
    assert spy.counts == (4, 4)


# endregion

# region Sources unchanged, nothing drawn


def test_the_source_is_unchanged_and_drawings_are_independent(tmp_path: Path) -> None:
    page = _page(tmp_path, "RGB", (100, 100), "page")
    before = _pixels(page)
    first = page.draw_line(0, 0, 99, 99)
    second = first.draw_line(0, 99, 99, 0)
    third = first.draw_rectangle(10, 10, 20, 20)
    assert _pixels(page) == before
    assert _pixels(first) == _pixels(reference_line(page, 0, 0, 99, 99))
    assert _pixels(second) == _pixels(reference_line(first, 0, 99, 99, 0))
    assert _pixels(third) == _pixels(reference_rectangle(first, 10, 10, 20, 20))
    assert len({first.pixmap.samples, second.pixmap.samples, third.pixmap.samples}) == 3


def test_a_source_without_segments_is_copied_unconverted(
    tmp_path: Path, spy: Spy
) -> None:
    grey = _page(tmp_path, "L", (60, 40), "grey")
    store = GraphicalStore(
        sources=[grey, _page(tmp_path, "RGB", (60, 40), "colour")],
        segments=[GraphicalSegment(1, HorizontalLinePath(x0=5, x1=55, y=20))],
    )
    copy = store.draw_segments_on_source(0)
    assert spy.counts == (0, 0)
    untouched = store.draw_interval(0.0, 20.0)[0]
    assert spy.counts == (1, 0)
    assert _pixels(copy) == _pixels(grey.copy())
    assert _pixels(untouched) == _pixels(grey.copy())
    assert (copy.pixmap.n, copy.pixmap.alpha) == (2, 1)
    assert copy is not grey and untouched is not grey


# endregion
