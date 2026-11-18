"""Tests for ScoreWarpLoader.

The fixture is a ScoreWarp-shaped SVG written by the test; negative cases are
mutated copies of its text. The validation logic is documented in
``tests/loader/README.md`` ("``ScoreWarpLoader`` validation logic"). All
expected values are exact.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign import ScoreWarpLoader
from timetoalign.core import Coordinate, TimeUnit
from timetoalign.timelines import ContinuousGraphicalTimeline

# The drawn axis: the ends of the horizontal line and the five major ticks.
AXIS_START = 76.70371459980326
AXIS_END = 932.2326510101616
MAJOR_TICKS = [
    76.70371459980326,
    290.5859487023929,
    504.4681828049824,
    718.350416907572,
    932.2326510101616,
]
LABELS = ["0", "10", "20", "30", "40"]


def _tick_xs() -> list[float]:
    """41 tick positions: the majors, and nine minors between each pair."""
    xs = [MAJOR_TICKS[0]]
    for left, right in zip(MAJOR_TICKS, MAJOR_TICKS[1:]):
        xs.extend(left + step * (right - left) / 10 for step in range(1, 10))
        xs.append(right)
    return xs


def _line(x1: float, x2: float, y1: int, y2: int, extra: str = "") -> str:
    return (
        f'<line x1="{x1!r}" x2="{x2!r}" y1="{y1}" y2="{y2}" stroke-width="1" '
        f'stroke="cornflowerblue"{extra}/>'
    )


def _text(x: float, y: int, text: str) -> str:
    return (
        f'<text text-anchor="middle" font-size="10.5" x="{x!r}" y="{y}">{text}</text>'
    )


def _svg() -> str:
    """The SVG text: a 792 x 175 root whose time axis ends past its width."""
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="792" height="175" '
        'viewBox="0 0 792 175" version="1.1" overflow="visible">',
        '<g class="page"><rect x="10" y="10" width="20" height="20"/></g>',
        '<g class="timeAxis">',
        _line(AXIS_START, AXIS_END, 155, 155, ' class="horizontalAxis"'),
        '<text text-anchor="left" font-size="10.5" x="1" y="151">Time (s)</text>',
    ]
    for x in _tick_xs():
        parts.append(_line(x, x, 155, 159 if x in MAJOR_TICKS else 157))
    for x, label in zip(MAJOR_TICKS, LABELS):
        parts.append(_text(x, 168, label))
    parts.append("</g></svg>")
    return "\n".join(parts)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "warped.svg"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def loader(tmp_path: Path) -> ScoreWarpLoader:
    """The loader with the fixture SVG loaded."""
    return ScoreWarpLoader.from_file(_write(tmp_path, _svg()))


@pytest.fixture
def drawing(loader: ScoreWarpLoader) -> ContinuousGraphicalTimeline:
    """The drawing's timeline."""
    return loader.create_timeline()


# region Reading the axis


def test_fixture_draws_41_ticks():
    assert len(_tick_xs()) == 41


def test_ticks_found(loader):
    assert len(loader.ticks) == 41
    assert loader.ticks[0] == Coordinate(AXIS_START, TimeUnit.css_pixels)
    assert loader.ticks[-1] == Coordinate(AXIS_END, TimeUnit.css_pixels)


def test_axis_segment(loader):
    start, end = loader.axis_segment
    assert start == Coordinate(76.70371459980326, TimeUnit.css_pixels)
    assert end == Coordinate(932.2326510101616, TimeUnit.css_pixels)


def test_calibration_pairs_read_from_labels(loader):
    pairs = loader.calibration
    assert len(pairs) == 5
    assert [position.value for position, _ in pairs] == MAJOR_TICKS
    assert [value.value for _, value in pairs] == [
        Fraction(0),
        Fraction(10),
        Fraction(20),
        Fraction(30),
        Fraction(40),
    ]
    assert all(type(value.value) is Fraction for _, value in pairs)
    assert {position.unit for position, _ in pairs} == {TimeUnit.css_pixels}
    assert {value.unit for _, value in pairs} == {TimeUnit.seconds}


def test_svg_size_verbatim(loader):
    assert loader.svg_size == {"width": "792", "height": "175"}


def test_px_size_kept_verbatim(tmp_path):
    text = _svg().replace('width="792" height="175"', 'width="792px" height="175px"')
    loader = ScoreWarpLoader.from_file(_write(tmp_path, text))
    assert loader.svg_size == {"width": "792px", "height": "175px"}


# endregion

# region The drawing's timeline


def test_timeline_length_is_content_extent(drawing):
    assert isinstance(drawing, ContinuousGraphicalTimeline)
    assert drawing.unit == TimeUnit.css_pixels
    assert drawing.length == Coordinate(932.2326510101616, TimeUnit.css_pixels)
    assert drawing.length != Coordinate(792.0, TimeUnit.css_pixels)


def test_timeline_meta_holds_size_strings(drawing):
    assert drawing.meta["svg_width"] == "792"
    assert drawing.meta["svg_height"] == "175"


def test_timeline_name_and_id(drawing, loader):
    assert drawing.name == "warped"
    assert drawing.id == "cgt_1"
    named = loader.create_timeline(uid="drawing", name="Drawing")
    assert (named.id, named.name) == ("drawing", "Drawing")


def test_axis_child(drawing):
    assert drawing.list_children() == ["cgt_2"]
    child = drawing.get_child("cgt_2")
    assert drawing.get_child_offset("cgt_2") == Coordinate(
        76.70371459980326, TimeUnit.css_pixels
    )
    assert child.length == Coordinate(855.5289364103584, TimeUnit.css_pixels)
    assert child.name == "Time (s)"
    assert child.unit == TimeUnit.css_pixels


def test_axis_map(drawing):
    cmap = drawing.get_child("cgt_2").get_conversion_map("seconds")
    assert cmap.source_unit == TimeUnit.css_pixels
    assert cmap.target_unit == TimeUnit.seconds
    assert cmap.scalar == 0.04675470144567245
    assert type(cmap.scalar) is float
    assert cmap.offset == 0.0
    assert type(cmap.offset) is float


def test_map_reads_the_ten_tick(drawing):
    child = drawing.get_child("cgt_2")
    cmap = child.get_conversion_map("seconds")
    stamp = drawing.get_timestamp_at(MAJOR_TICKS[1])
    position = stamp.get_coordinate("cgt_2")
    assert (position.value, position.unit) == (213.88223410258962, TimeUnit.css_pixels)

    # The line through the end ticks, evaluated in exact rational arithmetic on
    # the drawn doubles, reads the "10" tick just above 10; that real value
    # rounds to the double the map returns.
    start, ten, end = (
        Fraction(x) for x in (MAJOR_TICKS[0], MAJOR_TICKS[1], MAJOR_TICKS[4])
    )
    real = Fraction(40) * (ten - start) / (end - start)
    assert real != 10
    assert float(real) == 10.000000000000002

    assert cmap(position.value) == 10.000000000000002
    assert stamp.get_conversion_for("seconds") == 10.000000000000002


# endregion

# region Refusals


def test_refuses_no_time_axis(tmp_path):
    text = _svg().replace('class="timeAxis"', 'class="legend"')
    with pytest.raises(ValueError, match=r"0 <g class=\"timeAxis\"> elements"):
        ScoreWarpLoader.from_file(_write(tmp_path, text))


def test_refuses_transform_on_axis(tmp_path):
    text = _svg().replace(
        '<g class="timeAxis">', '<g class="timeAxis" transform="translate(10 0)">'
    )
    with pytest.raises(ValueError, match=r"transform 'translate\(10 0\)'"):
        ScoreWarpLoader.from_file(_write(tmp_path, text))


def test_refuses_transform_on_ancestor(tmp_path):
    text = (
        _svg()
        .replace('<g class="timeAxis">', '<g transform="scale(2)"><g class="timeAxis">')
        .replace("</g></svg>", "</g></g></svg>")
    )
    with pytest.raises(ValueError, match=r"transform 'scale\(2\)'"):
        ScoreWarpLoader.from_file(_write(tmp_path, text))


def test_refuses_contradicting_label(tmp_path):
    text = _svg().replace('y="168">40</text>', 'y="168">41</text>')
    loader = ScoreWarpLoader.from_file(_write(tmp_path, text))
    assert loader.calibration[-1][1] == Coordinate(Fraction(41), TimeUnit.seconds)
    with pytest.raises(
        ValueError, match=r"off the line through the first and last pair"
    ):
        loader.create_timeline()


def test_refuses_unknown_unit(tmp_path):
    text = _svg().replace(">Time (s)<", ">Time (fortnights)<")
    with pytest.raises(ValueError, match=r"unknown unit 'fortnights'"):
        ScoreWarpLoader.from_file(_write(tmp_path, text))


def test_refuses_a_second_svg(tmp_path, loader):
    first = loader.sources[0]
    second = tmp_path / "minutes.svg"
    second.write_text(
        _svg().replace(">Time (s)<", ">Time (minutes)<"), encoding="utf-8"
    )
    with pytest.raises(ValueError) as raised:
        loader.load(second)
    assert str(raised.value) == (
        f"ScoreWarpLoader reads one source and already holds {first}; use a new "
        f"loader for {second}, or call clear() first."
    )
    assert loader.sources == [first]
    assert loader.calibration[-1][1] == Coordinate(Fraction(40), TimeUnit.seconds)
    loader.clear()
    assert repr(loader) == "ScoreWarpLoader(not loaded)"
    loader.load(second)
    assert loader.sources == [second]
    assert loader.calibration[-1][1] == Coordinate(Fraction(40), TimeUnit.minutes)


def test_refuses_size_in_mm(tmp_path):
    text = _svg().replace('width="792"', 'width="209.55mm"')
    with pytest.raises(ValueError, match=r"width is '209.55mm'"):
        ScoreWarpLoader.from_file(_write(tmp_path, text))


# endregion
