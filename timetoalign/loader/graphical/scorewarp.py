"""ScoreWarpLoader for the time axis that ScoreWarp draws into an SVG.

ScoreWarp (https://github.com/iwk-digital/scorewarp) engraves an MEI score
with Verovio, shifts the engraved elements to the times of a performance and
appends a time axis to the SVG: a ``<g class="timeAxis">`` in the outer
viewport's CSS pixels, holding one ``<line class="horizontalAxis">``, a
vertical tick line per step (major ticks drawn longer than minor ones) and
text labels, one title such as "Time (s)" and a number at every major tick.

The loader reads that axis as drawn: the segment, the labelled ticks as
calibration pairs and the unit from the title. Nothing is re-derived from the
rule by which ScoreWarp chose its labels, so a label that contradicts the
others is refused by the calibration check. The positions of the other drawn
elements (noteheads and the transforms that place them) are not read.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, ClassVar

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.loader.base import Loader
from timetoalign.timelines import ContinuousGraphicalTimeline

module_logger = logging.getLogger(__name__)

# A root size: a number, unitless or in px (CSS pixels either way).
_CSS_PIXEL_LENGTH = re.compile(
    r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(px)?\s*"
)
# The axis title names its unit in parentheses: "Time (s)".
_TITLE_UNIT = re.compile(r"[^()]*\(([^()]+)\)\s*")


@dataclass(frozen=True)
class _ScoreWarpAxis:
    """The time axis of one ScoreWarp SVG, as drawn.

    Attributes:
        start: x of the horizontal axis line's start, in CSS pixels.
        end: x of its end.
        ticks: x of every tick line, in document order.
        calibration: ``(x, value)`` of every tick standing under a numeric
            label, the value read exactly from the label's decimal text.
        unit: The unit named by the title label.
        title: The title label's text.
        width: The root's ``width`` attribute, verbatim.
        height: The root's ``height`` attribute, verbatim.
        extent: The drawing's content extent: the viewport width or the end
            of the axis's farthest line, whichever is greater.
    """

    start: float
    end: float
    ticks: tuple[float, ...]
    calibration: tuple[tuple[float, Fraction], ...]
    unit: TimeUnit
    title: str
    width: str
    height: str
    extent: float


def _local_name(element: ET.Element) -> str:
    """Return *element*'s tag without its namespace."""
    return element.tag.rpartition("}")[2]


def _has_class(element: ET.Element, name: str) -> bool:
    """Return whether *name* is one of *element*'s classes."""
    return name in (element.get("class") or "").split()


def _css_pixels(element: ET.Element, attribute: str) -> float:
    """Read a root size attribute, refusing any unit but ``px``."""
    text = element.get(attribute)
    match = None if text is None else _CSS_PIXEL_LENGTH.fullmatch(text)
    if match is None:
        raise ValueError(
            f"The root <svg> {attribute} is {text!r}; only a number, unitless or "
            f"in px, places the drawing in CSS pixels."
        )
    return float(match.group(1))


def _coordinate(element: ET.Element, attribute: str) -> float:
    """Read a line coordinate attribute, which defaults to 0 in SVG."""
    return float(element.get(attribute, "0"))


def _decimal(text: str) -> Decimal | None:
    """Return *text* as a finite decimal, or None when it is no number."""
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _read_axis(source: Path) -> _ScoreWarpAxis:
    """Parse the ScoreWarp time axis of the SVG at *source*.

    Raises:
        ValueError: As :meth:`ScoreWarpLoader.load` states.
    """
    root = ET.parse(source).getroot()
    if _local_name(root) != "svg":
        raise ValueError(f"{source} is no SVG: its root is <{_local_name(root)}>.")
    viewport_width = _css_pixels(root, "width")
    _css_pixels(root, "height")

    axes = [
        element
        for element in root.iter()
        if _local_name(element) == "g" and _has_class(element, "timeAxis")
    ]
    if len(axes) != 1:
        raise ValueError(
            f'{source} holds {len(axes)} <g class="timeAxis"> elements, not one.'
        )
    axis = axes[0]
    parents = {child: parent for parent in root.iter() for child in parent}
    ancestors = []
    element = axis
    while element in parents:
        element = parents[element]
        ancestors.append(element)
    if any(_local_name(element) == "svg" for element in ancestors[:-1]):
        raise ValueError(
            "The time axis lies in a nested <svg> viewport, not in the outer "
            "viewport's CSS pixels."
        )
    for element in [*axis.iter(), *ancestors]:
        if element.get("transform") is not None:
            raise ValueError(
                f"A <{_local_name(element)}> on the time axis or around it has the "
                f"transform {element.get('transform')!r}; the axis is read in the "
                f"outer viewport's CSS pixels as drawn."
            )

    lines = [element for element in axis.iter() if _local_name(element) == "line"]
    segments = [line for line in lines if _has_class(line, "horizontalAxis")]
    if len(segments) != 1:
        raise ValueError(
            f'The time axis holds {len(segments)} <line class="horizontalAxis"> '
            f"elements, not one."
        )
    segment = segments[0]
    start, end = _coordinate(segment, "x1"), _coordinate(segment, "x2")
    if _coordinate(segment, "y1") != _coordinate(segment, "y2") or not start < end:
        raise ValueError(
            "The horizontal axis line must run from left to right at one y, "
            f"got x {start} to {end}, y {segment.get('y1')} to {segment.get('y2')}."
        )
    ticks = []
    for line in lines:
        if line is segment:
            continue
        x = _coordinate(line, "x1")
        if _coordinate(line, "x2") != x:
            raise ValueError(
                f"A tick line of the time axis is not vertical: x {x} to "
                f"{line.get('x2')}."
            )
        ticks.append(x)

    titles = []
    calibration = []
    for label in axis.iter():
        if _local_name(label) != "text":
            continue
        text = "".join(label.itertext()).strip()
        number = _decimal(text)
        if number is None:
            titles.append(text)
            continue
        x = float(label.get("x", "0"))
        if x not in ticks:
            raise ValueError(f"The label {text!r} at x {x} stands on no tick.")
        calibration.append((x, Fraction(number)))
    if len(titles) != 1:
        raise ValueError(
            f"The time axis holds {len(titles)} non-numeric labels {titles}, not one "
            f"title."
        )
    title = titles[0]
    match = _TITLE_UNIT.fullmatch(title)
    if match is None:
        raise ValueError(f"The axis title {title!r} names no unit in parentheses.")
    try:
        unit = TimeUnit(match.group(1).strip())
    except ValueError as error:
        raise ValueError(
            f"The axis title {title!r} names the unknown unit {match.group(1)!r}."
        ) from error

    return _ScoreWarpAxis(
        start=start,
        end=end,
        ticks=tuple(ticks),
        calibration=tuple(calibration),
        unit=unit,
        title=title,
        width=root.get("width"),
        height=root.get("height"),
        extent=max(
            viewport_width,
            *(max(_coordinate(line, "x1"), _coordinate(line, "x2")) for line in lines),
        ),
    )


class ScoreWarpLoader(Loader[_ScoreWarpAxis]):
    """Load the time axis drawn into a ScoreWarp SVG.

    The drawing becomes a :class:`ContinuousGraphicalTimeline` in
    ``css_pixels``, the unit of the SVG's outer viewport, and its time axis a
    child over the drawn segment, carrying the C-Map from ``css_pixels`` to
    the unit of the axis title, built through the labelled ticks (see
    :meth:`~timetoalign.timelines.Timeline.create_calibrated_child`).

    The drawing's length is its content extent, because a child lies within
    its parent: the viewport width, or the end of the axis's farthest line
    where the axis runs past the viewport (ScoreWarp sets
    ``overflow="visible"``).

    Known limits: the extent of text is not counted, since no font metrics
    are read, and nothing but the axis is read; the score's own elements and
    the transforms that place them are left alone.

    Examples:
        >>> loader = ScoreWarpLoader.from_file("warped.svg")
        >>> drawing = loader.create_timeline()
        >>> axis = drawing.get_child(drawing.list_children()[0])
        >>> cmap = axis.get_conversion_map("seconds")
    """

    #: One loader reads one SVG: one drawing, one axis.
    _single_source: ClassVar[bool] = True

    def __init__(self) -> None:
        """Initialize the loader."""
        super().__init__()
        self._axis: _ScoreWarpAxis | None = None
        self._logger = module_logger.getChild(self.__class__.__name__)

    def _load_source(self, source: Path) -> _ScoreWarpAxis:
        """Read the time axis of one ScoreWarp SVG.

        Args:
            source: Path to the SVG file.

        Returns:
            The axis as drawn.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the root is no ``<svg>`` or its ``width`` or
                ``height`` is in a unit other than ``px`` or none; if there is
                not exactly one ``g.timeAxis``, or it lies in a nested
                ``<svg>``, or a ``transform`` sits on it, inside it or on an
                ancestor; if there is not exactly one ``line.horizontalAxis``
                running left to right at one y, or a tick line is not
                vertical; if a numeric label stands on no tick; or if there
                is not exactly one title label, or its unit in parentheses is
                missing or unknown.
        """
        return _read_axis(source)

    def _accept_source(
        self, path: Path, source_meta: dict[str, Any], payload: _ScoreWarpAxis
    ) -> None:
        """Keep the axis of the one source."""
        super()._accept_source(path, source_meta, payload)
        self._axis = payload
        self._logger.debug(
            f"Loaded the time axis of {path}: {len(payload.ticks)} ticks, "
            f"{len(payload.calibration)} labelled"
        )

    def clear(self) -> None:
        """Forget the loaded SVG."""
        super().clear()
        self._axis = None

    def _require_axis(self) -> _ScoreWarpAxis:
        """Return the loaded axis, or raise when nothing is loaded."""
        if self._axis is None:
            raise RuntimeError("No SVG loaded. Call load() first.")
        return self._axis

    @property
    def axis_segment(self) -> tuple[Coordinate, Coordinate]:
        """Start and end of the drawn horizontal axis line, in ``css_pixels``."""
        axis = self._require_axis()
        return (
            Coordinate(axis.start, TimeUnit.css_pixels),
            Coordinate(axis.end, TimeUnit.css_pixels),
        )

    @property
    def ticks(self) -> list[Coordinate]:
        """The x of every tick line, in ``css_pixels`` and document order."""
        return [Coordinate(x, TimeUnit.css_pixels) for x in self._require_axis().ticks]

    @property
    def calibration(self) -> list[tuple[Coordinate, Coordinate]]:
        """``(position, value)`` of every labelled tick, in label order.

        The position is the tick's x in ``css_pixels`` as the file writes it,
        the value the label's decimal text read exactly, in the unit of the
        axis title.
        """
        axis = self._require_axis()
        return [
            (Coordinate(x, TimeUnit.css_pixels), Coordinate(value, axis.unit))
            for x, value in axis.calibration
        ]

    @property
    def svg_size(self) -> dict[str, str]:
        """The root's ``width`` and ``height`` attributes, verbatim."""
        axis = self._require_axis()
        return {"width": axis.width, "height": axis.height}

    def create_timeline(
        self,
        uid: str | None = None,
        name: str | None = None,
    ) -> ContinuousGraphicalTimeline:
        """Create the drawing's timeline, its time axis as a calibrated child.

        The timeline is in ``css_pixels``, its length the content extent, its
        ``meta`` the root's size strings (``svg_width``, ``svg_height``). Its
        one child is the axis: offset at the segment's start, length the
        segment's, named by the title label, with the C-Map from
        ``css_pixels`` to the title's unit through the labelled ticks.

        Both ids are minted ``cgt_N`` by this loader, the root first: its
        number is drawn even when *uid* names it, and a given *uid* is
        recorded among the loader's minted ids, so the child is never
        minted the root's id and successive timelines share none.

        Args:
            uid: Unique identifier for the timeline. ``None`` mints
                ``cgt_N``, a number this loader has not minted before.
            name: Human-readable name. Uses the source file's stem if None.

        Returns:
            The drawing's timeline.

        Raises:
            RuntimeError: If no SVG has been loaded.
            ValueError: If the labelled ticks do not lie on one line, as the
                calibration check states.
        """
        from timetoalign.timelines.factory import _flag_minted, _mint_timeline_id

        axis = self._require_axis()
        minted = uid is None
        root_id = _mint_timeline_id(self, TimeUnit.css_pixels)
        if minted:
            uid = root_id
        else:
            self._minted_timeline_ids.add(uid)
        timeline = _flag_minted(
            ContinuousGraphicalTimeline(
                length=axis.extent,
                unit=TimeUnit.css_pixels,
                uid=uid,
                name=name or self._sources[0].stem,
                meta={"svg_width": axis.width, "svg_height": axis.height},
            ),
            minted,
        )
        child = timeline.create_calibrated_child(
            axis.end - axis.start,
            axis.start,
            [(x, Coordinate(value, axis.unit)) for x, value in axis.calibration],
            uid=_mint_timeline_id(self, TimeUnit.css_pixels),
            name=axis.title,
        )
        _flag_minted(child, True)
        return timeline

    def __repr__(self) -> str:
        """Return a concise representation."""
        if self._axis is None:
            return "ScoreWarpLoader(not loaded)"
        return (
            f"ScoreWarpLoader(ticks={len(self._axis.ticks)}, "
            f"labelled={len(self._axis.calibration)}, unit={self._axis.unit.value})"
        )
