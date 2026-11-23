"""TimeAxisPath definitions for graphical timelines.

This module provides the abstraction for mapping 1D timeline coordinates
to 2D image coordinates (and vice versa).

A TimeAxisPath represents the geometric trajectory that the time axis
follows through an image: no image has a time axis until that path is
defined. Implementations range from simple (horizontal lines) to complex
(parametric curves, spirals); a :class:`PiecewisePath` is made of
consecutive pieces, such as the accolades of a score page, read one after
the other. :meth:`TimeAxisPath.from_boxes` makes the path of a page from
the areas drawn on it, and :func:`group_into_accolades` is the rule that
finds the accolades among them.

A path holds the numbers it is given and computes with them as they are: a
line of integers (or of ``Fraction`` values) has an exact length and gives
exact answers; nothing is turned into a float. Only a
:class:`DiagonalLinePath` and a :class:`ParametricPath` answer in floats
(their lengths are square roots).

The design is forward-compatible with video (3D paths through frame-time
and 2D pixel space).

Note:
    The TimeAxisPath classes in this module operate in path-local coordinates
    (starting at 0) and map directly to image coordinates.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from timetoalign.core import Box, Shape

module_logger = logging.getLogger(__name__)

#: A number of a path: kept as given.
Number = int | float | Fraction


def _half(number: Number) -> Number:
    """Half of *number*, exactly: an ``int`` where whole, else a ``Fraction``.

    A float is halved as a float (halving a float is exact).
    """
    if isinstance(number, float):
        return number / 2
    half = Fraction(number) / 2
    return half.numerator if half.denominator == 1 else half


def group_into_accolades(left_edges: Sequence[Number]) -> list[int]:
    """Assign the boxes of one page, given in reading order, to accolades.

    An accolade is one line of a page, read from left to right. Reading on,
    the left edges of the boxes grow within an accolade and fall back at the
    start of the next one. A box therefore begins a new accolade when its
    left edge lies left of the previous box's by more than half the span of
    all left edges (``max(left_edges) - min(left_edges)``). The comparison
    is ``2 * (previous - x) > span``, exact on integers and fractions.

    Args:
        left_edges: The left edge (least x) of every box of ONE page, in
            reading order.

    Returns:
        Per box, the index of its accolade, counted from 0. Empty for no
        box.

    Examples:
        Nine measures of a page whose sixth starts again at the left:

        >>> group_into_accolades([490, 944, 1481, 2034, 2669, 498, 1465, 2110, 2704])
        [0, 0, 0, 0, 0, 1, 1, 1, 1]
        >>> group_into_accolades([])
        []
    """
    if not left_edges:
        return []
    span = max(left_edges) - min(left_edges)
    accolades = []
    accolade = 0
    previous: Number | None = None
    for left_edge in left_edges:
        if previous is not None and 2 * (previous - left_edge) > span:
            accolade += 1
        accolades.append(accolade)
        previous = left_edge
    return accolades


# region TimeAxisPath ABC


class TimeAxisPath(ABC):
    """Abstract definition of a time axis path through 2D space.

    A TimeAxisPath maps 1D timeline coordinates to 2D image coordinates.
    The path has a defined length (in timeline units, typically pixels)
    and provides bidirectional coordinate conversion.

    Subclasses implement specific geometric shapes:
    - HorizontalLinePath: x = time, y = constant
    - DiagonalLinePath: arbitrary straight line
    - ParametricPath: x(t), y(t) functions for curves/spirals
    - PiecewisePath: consecutive pieces, such as the accolades of a page
    """

    @classmethod
    def from_boxes(
        cls,
        boxes: Iterable[Shape | Box],
        *,
        accolades: Sequence[int] | None = None,
    ) -> PiecewisePath:
        """Make the time path of one page from the areas drawn on it.

        Each accolade becomes one :class:`HorizontalLinePath` spanning its
        boxes exactly: ``x0`` is their least and ``x1`` their greatest x,
        ``y`` the middle of their y extent and ``tolerance`` half of that
        extent, so the line's strip is the accolade's y extent. The middle
        and the half are exact: an ``int`` where whole, else a ``Fraction``.

        Args:
            boxes: The areas of ONE page in reading order: shapes, or Box
                events (read through their shape), all of one unit and
                number type.
            accolades: Per box, a label of its accolade (one label per box);
                the accolades follow each other in the order of their first
                box. When omitted, :func:`group_into_accolades` assigns them
                from the boxes' left edges.

        Returns:
            The path, one piece per accolade (a ``PiecewisePath`` of one
            piece too).

        Raises:
            ValueError: If no box is given, the boxes lie on two units or
                number types, *accolades* does not hold one label per box,
                or an accolade has no x extent (the message names it by its
                position, counted from 1).

        Examples:
            Three measures, the third on a second line:

            >>> from timetoalign.core import BoundingBox
            >>> measures = [
            ...     BoundingBox.from_corners(100, 50, 300, 90, unit="pixels"),
            ...     BoundingBox.from_corners(300, 52, 520, 95, unit="pixels"),
            ...     BoundingBox.from_corners(90, 150, 400, 190, unit="pixels"),
            ... ]
            >>> path = TimeAxisPath.from_boxes(measures)
            >>> path.pieces[0]
            HorizontalLinePath(x0=100, x1=520, y=Fraction(145, 2), tolerance=Fraction(45, 2))
            >>> path.pieces[1]
            HorizontalLinePath(x0=90, x1=400, y=170, tolerance=20)
            >>> path.offsets, path.length
            ((0, 420), 730)
            >>> path.piece_of(measures[2])
            1
        """
        from timetoalign.core import Box

        shapes = [box.shape if isinstance(box, Box) else box for box in boxes]
        if not shapes:
            raise ValueError("A time path needs at least one box.")
        first = shapes[0]
        for position, shape in enumerate(shapes):
            if (shape.unit, shape.number_type) != (first.unit, first.number_type):
                raise ValueError(
                    "The boxes of a time path are in one unit and number type: "
                    f"box 0 is in {first.unit} as {first.number_type.name}, box "
                    f"{position} in {shape.unit} as {shape.number_type.name}."
                )
        if accolades is None:
            labels: Sequence[int] = group_into_accolades(
                [shape.x_min.value for shape in shapes]
            )
        else:
            labels = list(accolades)
            if len(labels) != len(shapes):
                raise ValueError(
                    f"accolades assigns {len(labels)} boxes; {len(shapes)} were given."
                )
        grouped: dict[int, list[Shape]] = {}
        for label, shape in zip(labels, shapes):
            grouped.setdefault(label, []).append(shape)
        pieces = []
        for position, members in enumerate(grouped.values(), start=1):
            top = min(shape.y_min.value for shape in members)
            bottom = max(shape.y_max.value for shape in members)
            try:
                pieces.append(
                    HorizontalLinePath(
                        x0=min(shape.x_min.value for shape in members),
                        x1=max(shape.x_max.value for shape in members),
                        y=_half(top + bottom),
                        tolerance=_half(bottom - top),
                    )
                )
            except ValueError as error:
                raise ValueError(f"accolade {position}: {error}") from error
        return PiecewisePath(tuple(pieces))

    @property
    @abstractmethod
    def length(self) -> Number:
        """Total length of this path in timeline units."""
        ...

    @abstractmethod
    def to_2d(self, coord: Number) -> tuple[Number, Number]:
        """Map timeline coordinate to image (x, y).

        Args:
            coord: Timeline coordinate in [0, length].

        Returns:
            (x, y) tuple in image pixel coordinates.
        """
        ...

    @abstractmethod
    def from_2d(self, x: Number, y: Number) -> Number | None:
        """Map image (x, y) to timeline coordinate.

        Args:
            x: X coordinate in image.
            y: Y coordinate in image.

        Returns:
            Timeline coordinate, or None if point is not on the path.
        """
        ...

    def distance_to_path(self, x: Number, y: Number) -> Number:
        """Compute perpendicular distance from point to path.

        Default implementation samples the path and finds minimum distance.
        Subclasses may override with analytical solutions.

        Args:
            x: X coordinate in image.
            y: Y coordinate in image.

        Returns:
            Distance in pixels.
        """
        # Sample path and find closest point
        min_dist = float("inf")
        samples = max(100, int(self.length))
        for i in range(samples + 1):
            t = (i / samples) * self.length
            px, py = self.to_2d(t)
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            min_dist = min(min_dist, dist)
        return min_dist

    def is_point_on_path(self, x: Number, y: Number, tolerance: Number = 5.0) -> bool:
        """Check if a point is close enough to the path.

        Args:
            x: X coordinate in image.
            y: Y coordinate in image.
            tolerance: Maximum distance in pixels.

        Returns:
            True if point is within tolerance of the path.
        """
        return self.distance_to_path(x, y) <= tolerance


# endregion


# region HorizontalLinePath


@dataclass(frozen=True)
class HorizontalLinePath(TimeAxisPath):
    """Time axis as a horizontal line at fixed y.

    The simplest case: x directly maps to time coordinate.
    This is the typical case for musical scores, spectrograms,
    and most graphical analyses.

    Attributes:
        x0: Starting x coordinate (left edge).
        x1: Ending x coordinate (right edge).
        y: Fixed y coordinate (vertical position of the line).
        tolerance: The half-breadth of the strip the line stands for: the
            strip spans ``y - tolerance`` to ``y + tolerance``, and
            :meth:`from_2d` answers for the points within it.

    Every number is kept as given: integers (and ``Fraction`` values) give
    exact answers.

    Examples:
        Pixel coordinates stay integral, so the geometry answers in whole
        pixels rather than widening to float:

        >>> path = HorizontalLinePath(x0=10, x1=500, y=100)
        >>> path.length
        490
        >>> path.to_2d(0)
        (10, 100)
        >>> path.to_2d(245)
        (255, 100)
        >>> path.from_2d(255, 100)
        245
    """

    x0: Number
    x1: Number
    y: Number
    tolerance: Number = 10.0

    def __post_init__(self) -> None:
        """Validate path parameters."""
        if self.x1 <= self.x0:
            raise ValueError(f"x1 ({self.x1}) must be greater than x0 ({self.x0})")

    @property
    def length(self) -> Number:
        """Length is simply x1 - x0."""
        return self.x1 - self.x0

    def to_2d(self, coord: Number) -> tuple[Number, Number]:
        """Map timeline coordinate to (x, y)."""
        return (self.x0 + coord, self.y)

    def from_2d(self, x: Number, y: Number) -> Number | None:
        """Map (x, y) to timeline coordinate.

        Returns None if:
        - y is outside the strip ``y ± tolerance``
        - x is outside the [x0, x1] range
        """
        if abs(y - self.y) > self.tolerance:
            return None
        if not (self.x0 <= x <= self.x1):
            return None
        return x - self.x0

    def distance_to_path(self, x: Number, y: Number) -> Number:
        """Distance is simply |y - self.y| if x is in range."""
        if x < self.x0:
            # Distance to left endpoint
            return math.sqrt((x - self.x0) ** 2 + (y - self.y) ** 2)
        if x > self.x1:
            # Distance to right endpoint
            return math.sqrt((x - self.x1) ** 2 + (y - self.y) ** 2)
        # x is in range, distance is vertical only
        return abs(y - self.y)


# endregion


# region VerticalLinePath


@dataclass(frozen=True)
class VerticalLinePath(TimeAxisPath):
    """Time axis as a vertical line at fixed x.

    Useful for vertical timelines where y maps to time.

    Attributes:
        x: Fixed x coordinate.
        y0: Starting y coordinate (top).
        y1: Ending y coordinate (bottom).
        tolerance: The half-breadth of the strip the line stands for: the
            strip spans ``x - tolerance`` to ``x + tolerance``, and
            :meth:`from_2d` answers for the points within it.

    Every number is kept as given: integers (and ``Fraction`` values) give
    exact answers.
    """

    x: Number
    y0: Number
    y1: Number
    tolerance: Number = 10.0

    def __post_init__(self) -> None:
        """Validate path parameters."""
        if self.y1 <= self.y0:
            raise ValueError(f"y1 ({self.y1}) must be greater than y0 ({self.y0})")

    @property
    def length(self) -> Number:
        """Length is y1 - y0."""
        return self.y1 - self.y0

    def to_2d(self, coord: Number) -> tuple[Number, Number]:
        """Map timeline coordinate to (x, y)."""
        return (self.x, self.y0 + coord)

    def from_2d(self, x: Number, y: Number) -> Number | None:
        """Map (x, y) to timeline coordinate (None outside the strip or the line)."""
        if abs(x - self.x) > self.tolerance:
            return None
        if not (self.y0 <= y <= self.y1):
            return None
        return y - self.y0

    def distance_to_path(self, x: Number, y: Number) -> Number:
        """Distance calculation for vertical line."""
        if y < self.y0:
            return math.sqrt((x - self.x) ** 2 + (y - self.y0) ** 2)
        if y > self.y1:
            return math.sqrt((x - self.x) ** 2 + (y - self.y1) ** 2)
        return abs(x - self.x)


# endregion


# region PiecewisePath


@dataclass(frozen=True)
class PiecewisePath(TimeAxisPath):
    """A time axis made of consecutive pieces, read one after the other.

    The path of a score page is piecewise: each accolade is one horizontal
    line, and the page's time runs along the first, then along the second,
    and so on. Each piece starts on the path where the pieces before it end
    (:attr:`offsets`); the path's length is the sum of its pieces' lengths.
    All numbers are the pieces' own: pieces of integers give an integer
    path.

    Attributes:
        pieces: The pieces in reading order, at least one, none of them a
            ``PiecewisePath`` itself (pieces are listed flat).

    Raises:
        ValueError: If there is no piece, or a piece is a ``PiecewisePath``.

    Examples:
        Two accolades of a page, 490 and 400 pixels long:

        >>> path = PiecewisePath(
        ...     (
        ...         HorizontalLinePath(x0=10, x1=500, y=100, tolerance=40),
        ...         HorizontalLinePath(x0=20, x1=420, y=300, tolerance=40),
        ...     )
        ... )
        >>> path.length
        890
        >>> path.offsets
        (0, 490)
        >>> path.to_2d(489)
        (499, 100)
        >>> path.to_2d(490)
        (20, 300)
        >>> path.from_2d(120, 320)
        590
        >>> path.from_2d(120, 200) is None
        True
    """

    pieces: tuple[TimeAxisPath, ...]

    def __post_init__(self) -> None:
        """Validate the pieces."""
        object.__setattr__(self, "pieces", tuple(self.pieces))
        if not self.pieces:
            raise ValueError("A PiecewisePath needs at least one piece.")
        for position, piece in enumerate(self.pieces):
            if isinstance(piece, PiecewisePath):
                raise ValueError(
                    f"Piece {position} of a PiecewisePath is itself a "
                    "PiecewisePath: the pieces are listed flat."
                )

    @property
    def offsets(self) -> tuple[Number, ...]:
        """Where each piece starts on the path: the lengths before it, summed."""
        offsets = []
        offset: Number = 0
        for piece in self.pieces:
            offsets.append(offset)
            offset = offset + piece.length
        return tuple(offsets)

    @property
    def length(self) -> Number:
        """The sum of the pieces' lengths."""
        return sum(piece.length for piece in self.pieces)

    def to_2d(self, coord: Number) -> tuple[Number, Number]:
        """Map a coordinate of the path to the image (x, y) of its piece.

        A piece holds the coordinates from its offset up to, not including,
        the next piece's offset; the last piece also holds the path's end.
        """
        index = 0
        for position, offset in enumerate(self.offsets):
            if coord >= offset:
                index = position
        return self.pieces[index].to_2d(coord - self.offsets[index])

    def from_2d(self, x: Number, y: Number) -> Number | None:
        """Map image (x, y) to the path's coordinate.

        Of the pieces that answer for the point, the one whose line is
        nearest decides (the earlier of two equally near); its answer is
        counted from the start of the path.

        Returns:
            The coordinate on the path, or None where no piece answers.
        """
        best: tuple[Number, Number] | None = None
        for piece, offset in zip(self.pieces, self.offsets):
            local = piece.from_2d(x, y)
            if local is None:
                continue
            distance = piece.distance_to_path(x, y)
            if best is None or distance < best[0]:
                best = (distance, local + offset)
        return None if best is None else best[1]

    def distance_to_path(self, x: Number, y: Number) -> Number:
        """The least distance from the point to a piece."""
        return min(piece.distance_to_path(x, y) for piece in self.pieces)

    def piece_of(self, shape: Shape) -> int:
        """The index of the horizontal piece an area of the image lies on.

        A shape lies on a :class:`HorizontalLinePath` piece when the piece's
        ``[x0, x1]`` holds the shape's x extent and the piece's strip
        (``y ± tolerance``) holds the shape's vertical centre, the exact
        middle of its y extent. Where several pieces hold it (strips may
        overlap), the one whose ``y`` is nearest the centre is taken.

        Args:
            shape: An area in the image's space.

        Returns:
            The piece's index in :attr:`pieces`.

        Raises:
            ValueError: If the shape lies on no piece, or two pieces holding
                it are equally near its centre.

        Examples:
            >>> from timetoalign.core import BoundingBox
            >>> path = PiecewisePath(
            ...     (
            ...         HorizontalLinePath(x0=10, x1=500, y=100, tolerance=40),
            ...         HorizontalLinePath(x0=20, x1=420, y=300, tolerance=40),
            ...     )
            ... )
            >>> path.piece_of(BoundingBox.from_xywh(30, 280, 100, 50, unit="pixels"))
            1
        """
        x_min, x_max = shape.x_min.value, shape.x_max.value
        centre = _half(shape.y_min.value + shape.y_max.value)
        where = f"The shape spanning x {x_min} to {x_max} with its centre at y {centre}"
        candidates = sorted(
            (abs(piece.y - centre), position)
            for position, piece in enumerate(self.pieces)
            if isinstance(piece, HorizontalLinePath)
            and piece.x0 <= x_min
            and x_max <= piece.x1
            and abs(piece.y - centre) <= piece.tolerance
        )
        if not candidates:
            raise ValueError(f"{where} lies on no piece of the path.")
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            raise ValueError(
                f"{where} is equally near pieces {candidates[0][1]} and "
                f"{candidates[1][1]} of the path."
            )
        return candidates[0][1]


# endregion


# region DiagonalLinePath


@dataclass(frozen=True)
class DiagonalLinePath(TimeAxisPath):
    """Time axis as a diagonal (arbitrary straight) line.

    Useful for rotated segments, skewed timelines, or any straight
    line that isn't perfectly horizontal or vertical.

    The timeline coordinate maps to arc length along the line,
    so length equals the Euclidean distance between endpoints.

    Attributes:
        start: (x, y) starting point.
        end: (x, y) ending point.
        tolerance: Distance tolerance for from_2d conversion.

    Examples:
        >>> path = DiagonalLinePath(start=(0, 0), end=(300, 400))
        >>> path.length  # sqrt(300^2 + 400^2) = 500
        500.0
        >>> path.to_2d(250)  # Halfway point
        (150.0, 200.0)
    """

    start: tuple[float, float]
    end: tuple[float, float]
    tolerance: float = 10.0

    def __post_init__(self) -> None:
        """Validate and compute derived values."""
        if self.start == self.end:
            raise ValueError("Start and end points must be different")

    @property
    def _dx(self) -> float:
        return self.end[0] - self.start[0]

    @property
    def _dy(self) -> float:
        return self.end[1] - self.start[1]

    @property
    def length(self) -> float:
        """Euclidean length of the line."""
        return math.sqrt(self._dx**2 + self._dy**2)

    def to_2d(self, coord: float) -> tuple[float, float]:
        """Map timeline coordinate (arc length) to (x, y)."""
        if self.length == 0:
            return self.start
        ratio = coord / self.length
        return (
            self.start[0] + ratio * self._dx,
            self.start[1] + ratio * self._dy,
        )

    def from_2d(self, x: float, y: float) -> float | None:
        """Map (x, y) to timeline coordinate.

        Projects the point onto the line and returns the arc length
        if the projection falls within the segment.
        """
        # Vector from start to point
        px = x - self.start[0]
        py = y - self.start[1]

        # Project onto line direction
        line_len_sq = self._dx**2 + self._dy**2
        if line_len_sq == 0:
            return None

        # Projection parameter t (0 = start, 1 = end)
        t = (px * self._dx + py * self._dy) / line_len_sq

        # Check if projection is within segment
        if not (0 <= t <= 1):
            return None

        # Check distance from line
        proj_x = self.start[0] + t * self._dx
        proj_y = self.start[1] + t * self._dy
        dist = math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)

        if dist > self.tolerance:
            return None

        # Return arc length
        return t * self.length

    def distance_to_path(self, x: float, y: float) -> float:
        """Perpendicular distance to line segment."""
        px = x - self.start[0]
        py = y - self.start[1]

        line_len_sq = self._dx**2 + self._dy**2
        if line_len_sq == 0:
            return math.sqrt(px**2 + py**2)

        t = max(0, min(1, (px * self._dx + py * self._dy) / line_len_sq))

        proj_x = self.start[0] + t * self._dx
        proj_y = self.start[1] + t * self._dy

        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


# endregion


# region ParametricPath


class ParametricPath(TimeAxisPath):
    """Time axis defined by parametric functions x(t), y(t).

    For complex paths like spirals, curves, Bezier paths, etc.
    The parameter t is NOT directly the timeline coordinate - it's
    mapped via arc length parameterization.

    This class pre-computes an arc length table for efficient
    coordinate conversion.

    Attributes:
        x_func: Function t -> x coordinate.
        y_func: Function t -> y coordinate.
        t_start: Starting parameter value.
        t_end: Ending parameter value.
        tolerance: Distance tolerance for from_2d conversion.

    Examples:
        >>> # Archimedean spiral: r = a + b*t
        >>> import math
        >>> a, b = 50, 10
        >>> x_func = lambda t: (a + b*t) * math.cos(t)
        >>> y_func = lambda t: (a + b*t) * math.sin(t)
        >>> path = ParametricPath(x_func, y_func, t_start=0, t_end=4*math.pi)
    """

    def __init__(
        self,
        x_func: Callable[[float], float],
        y_func: Callable[[float], float],
        t_start: float = 0.0,
        t_end: float = 1.0,
        samples: int = 1000,
        tolerance: float = 10.0,
    ):
        """Initialize parametric path.

        Args:
            x_func: Function mapping parameter t to x coordinate.
            y_func: Function mapping parameter t to y coordinate.
            t_start: Starting parameter value.
            t_end: Ending parameter value.
            samples: Number of samples for arc length computation.
            tolerance: Distance tolerance for from_2d.
        """
        self.x_func = x_func
        self.y_func = y_func
        self.t_start = t_start
        self.t_end = t_end
        self.tolerance = tolerance
        self._samples = samples

        # Pre-compute arc length table
        self._arc_lengths: list[float] = []
        self._t_values: list[float] = []
        self._build_arc_length_table()

    def _build_arc_length_table(self) -> None:
        """Build lookup table for arc length parameterization."""
        self._t_values = []
        self._arc_lengths = []

        total_length = 0.0
        prev_x = self.x_func(self.t_start)
        prev_y = self.y_func(self.t_start)

        self._t_values.append(self.t_start)
        self._arc_lengths.append(0.0)

        for i in range(1, self._samples + 1):
            t = self.t_start + (i / self._samples) * (self.t_end - self.t_start)
            x = self.x_func(t)
            y = self.y_func(t)

            segment_length = math.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            total_length += segment_length

            self._t_values.append(t)
            self._arc_lengths.append(total_length)

            prev_x, prev_y = x, y

        self._total_length = total_length

    @property
    def length(self) -> float:
        """Total arc length of the path."""
        return self._total_length

    def _arc_to_t(self, arc: float) -> float:
        """Convert arc length to parameter t using binary search."""
        if arc <= 0:
            return self.t_start
        if arc >= self._total_length:
            return self.t_end

        # Binary search in arc_lengths
        lo, hi = 0, len(self._arc_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self._arc_lengths[mid] < arc:
                lo = mid + 1
            else:
                hi = mid

        # Linear interpolation between samples
        if lo == 0:
            return self._t_values[0]

        arc_lo = self._arc_lengths[lo - 1]
        arc_hi = self._arc_lengths[lo]
        t_lo = self._t_values[lo - 1]
        t_hi = self._t_values[lo]

        if arc_hi == arc_lo:
            return t_lo

        ratio = (arc - arc_lo) / (arc_hi - arc_lo)
        return t_lo + ratio * (t_hi - t_lo)

    def to_2d(self, coord: float) -> tuple[float, float]:
        """Map timeline coordinate (arc length) to (x, y)."""
        t = self._arc_to_t(coord)
        return (self.x_func(t), self.y_func(t))

    def from_2d(self, x: float, y: float) -> float | None:
        """Map (x, y) to timeline coordinate.

        Finds the closest point on the path and returns its arc length.
        """
        min_dist = float("inf")
        best_arc = None

        for i, t in enumerate(self._t_values):
            px = self.x_func(t)
            py = self.y_func(t)
            dist = math.sqrt((x - px) ** 2 + (y - py) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_arc = self._arc_lengths[i]

        if min_dist > self.tolerance:
            return None

        return best_arc


# endregion
