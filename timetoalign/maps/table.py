"""Table-based mapping with interpolation.

This module provides TableMap, which uses explicit anchor points
with interpolation for values between them.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from fractions import Fraction
from numbers import Rational
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.enums import ExtrapolationPolicy, InterpolationKind, NumberType, TimeUnit
from ..core.time import (
    CoordinateValue,
    number_from_json,
    number_to_json,
)
from .base import ConversionMap, MeasureMapIds

if TYPE_CHECKING:
    from typing_extensions import Self


_KINDS = (InterpolationKind.linear, InterpolationKind.previous)
"""The interpolation kinds the format defines."""

_EXTRAPOLATION_WRITTEN = {
    ExtrapolationPolicy.error: "none",
    ExtrapolationPolicy.nan: "none",
    ExtrapolationPolicy.constant: "constant",
    ExtrapolationPolicy.extrapolate: "linear",
}
"""How each extrapolation policy is written: outside the anchors, an error
and a NaN are both undefined values."""

_EXTRAPOLATION_READ = {
    "none": ExtrapolationPolicy.error,
    "constant": ExtrapolationPolicy.constant,
    "linear": ExtrapolationPolicy.extrapolate,
}
"""The extrapolation policy each written value reads as."""


def _table_form(
    cmap: ConversionMap[Any],
    x_values: Sequence[Any],
    y_values: Sequence[Any],
    *,
    kind: InterpolationKind,
    extrapolate: ExtrapolationPolicy,
    minimal: bool,
) -> dict[str, Any]:
    """Write *cmap* as the format's ``TableMap`` over the given anchors."""
    return cmap._map_object(
        "TableMap",
        {
            "x_values": [number_to_json(v) for v in x_values],
            "y_values": [number_to_json(v) for v in y_values],
            "kind": kind.value,
            "extrapolate": _EXTRAPOLATION_WRITTEN[extrapolate],
        },
        minimal=minimal,
    )


def _is_exact(value: Any) -> bool:
    """Whether *value* is an exact number: an ``int`` or a ``Fraction``."""
    return isinstance(value, Rational)


class TableMap(ConversionMap[CoordinateValue]):
    """Lookup table with interpolation between anchor points.

    A TableMap defines a mapping using explicit (x, y) pairs. Values
    between pairs are interpolated according to the specified method.

    This is useful for:
    - Tempo-based time conversions (ticks to seconds with varying tempo)
    - Alignment anchors
    - Any non-linear monotonic mapping

    Attributes:
        x_values: The input coordinates (must be strictly increasing).
        y_values: The corresponding output values.
        kind: The interpolation method.
        extrapolate: How to handle out-of-bounds inputs.

    Examples:
        >>> # Simple tempo map: 0 ticks = 0 sec, 480 ticks = 0.5 sec, 960 ticks = 1.5 sec
        >>> tempo_map = TableMap(
        ...     x_values=[0, 480, 960],
        ...     y_values=[0.0, 0.5, 1.5],
        ...     source_unit="ticks",
        ...     target_unit="seconds",
        ... )
        >>> tempo_map(240)  # Interpolate: 0.25 sec
        0.25
        >>> tempo_map(720)  # Interpolate: 1.0 sec
        1.0

        >>> # Inverse map, landing on the ticks axis, which writes whole ticks
        >>> inverse = tempo_map.inverse()
        >>> inverse(1.0)
        720
    """

    _manifest_type: ClassVar[str | None] = "TableMap"

    def __init__(
        self,
        *,
        x_values: Sequence[CoordinateValue],
        y_values: Sequence[CoordinateValue],
        kind: InterpolationKind | str = InterpolationKind.linear,
        extrapolate: ExtrapolationPolicy | str = ExtrapolationPolicy.extrapolate,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
        output_number_type: NumberType | str | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize a TableMap.

        Args:
            x_values: Input coordinates. Must be strictly monotonically increasing.
            y_values: Output values. Must have same length as x_values.
                     For invertibility, should be strictly monotonic.
            kind: Interpolation method.
            extrapolate: How to handle out-of-bounds values.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates.
            uid: Optional explicit ID.
            seed: The seed of the minter of the id when *uid* is None.
            name: Human-readable name for this map. Defaults to the map's ID.
            output_number_type: The representation results are written in,
                among those the target unit admits; ``None`` for the target
                unit's default.

        Raises:
            ValueError: If x_values and y_values have different lengths.
            ValueError: If x_values are not strictly increasing.
            ValueError: If fewer than 2 points are provided.
            ValueError: If the target unit does not admit *output_number_type*.
            ValueError: If *output_number_type* is ``fraction`` and an
                anchor is a float.
        """
        # Validate inputs
        if len(x_values) != len(y_values):
            raise ValueError(
                f"x_values and y_values must have same length, "
                f"got {len(x_values)} and {len(y_values)}"
            )
        if len(x_values) < 2:
            raise ValueError("TableMap requires at least 2 anchor points")

        # Store original values for serialization (may have Fractions). Set
        # first: whether the map computes in doubles depends on them.
        self._x_original = list(x_values)
        self._y_original = list(y_values)
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            seed=seed,
            name=name,
            output_number_type=output_number_type,
        )

        # Convert to numpy arrays (float64 for interpolation)
        self._x = np.array([float(v) for v in x_values], dtype=np.float64)
        self._y = np.array([float(v) for v in y_values], dtype=np.float64)

        # Validate monotonicity
        x_diff = np.diff(self._x)
        if not np.all(x_diff > 0):
            raise ValueError("x_values must be strictly monotonically increasing")

        # Exact anchors (every one an int or a Fraction) interpolate exactly
        # for an exact input; one float among them keeps the float path.
        self._exact: tuple[list[Fraction], list[Fraction]] | None = None
        if all(_is_exact(v) for v in (*self._x_original, *self._y_original)):
            self._exact = (
                [Fraction(v) for v in self._x_original],
                [Fraction(v) for v in self._y_original],
            )

        # Parse enum values
        self._kind = InterpolationKind(kind) if isinstance(kind, str) else kind
        self._extrapolate = (
            ExtrapolationPolicy(extrapolate)
            if isinstance(extrapolate, str)
            else extrapolate
        )

        # Check if y is monotonic (for invertibility)
        y_diff = np.diff(self._y)
        self._y_increasing = bool(np.all(y_diff > 0))
        self._y_decreasing = bool(np.all(y_diff < 0))
        self._is_invertible = self._y_increasing or self._y_decreasing

    @property
    def x_values(self) -> NDArray[np.floating[Any]]:
        """The input anchor points."""
        return self._x

    @property
    def y_values(self) -> NDArray[np.floating[Any]]:
        """The output anchor points."""
        return self._y

    @property
    def kind(self) -> InterpolationKind:
        """The interpolation method."""
        return self._kind

    @property
    def extrapolation(self) -> ExtrapolationPolicy:
        """How out-of-bounds values are handled."""
        return self._extrapolate

    @property
    def is_invertible(self) -> bool:
        """Whether this map can be inverted (y values are strictly monotonic)."""
        return self._is_invertible

    @property
    def x_min(self) -> float:
        """Minimum x value."""
        return float(self._x[0])

    @property
    def x_max(self) -> float:
        """Maximum x value."""
        return float(self._x[-1])

    @property
    def y_min(self) -> float:
        """Minimum y value."""
        return float(np.min(self._y))

    @property
    def y_max(self) -> float:
        """Maximum y value."""
        return float(np.max(self._y))

    def _parameters(self) -> list[CoordinateValue]:
        return [*self._x_original, *self._y_original]

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Convert a single value using interpolation.

        Linear interpolation and linear extrapolation are exact when the input
        and every anchor are exact (``int`` or ``Fraction``): the result is the
        ``Fraction`` ``y0 + (x - x0) * (y1 - y0) / (x1 - x0)``. A float input
        or a float anchor computes in floats.
        """
        if self._exact is not None and _is_exact(value):
            exact = self._interpolate_exactly(Fraction(value), *self._exact)
            if exact is not None:
                return exact
        x = float(value)

        # Handle out of bounds
        if x < self._x[0] or x > self._x[-1]:
            return self._handle_out_of_bounds(x)

        # Handle interpolation based on kind
        if self._kind == InterpolationKind.linear:
            return float(np.interp(x, self._x, self._y))

        if self._kind == InterpolationKind.previous:
            # index of rightmost value <= x; no arithmetic is involved, so the
            # anchor value is answered as stored.
            idx = np.searchsorted(self._x, x, side="right") - 1
            idx = max(0, min(idx, len(self._x) - 1))
            return self._y_original[idx]

        if self._kind == InterpolationKind.next:
            # index of leftmost value >= x
            idx = np.searchsorted(self._x, x, side="left")
            idx = max(0, min(idx, len(self._x) - 1))
            return float(self._y[idx])

        if self._kind == InterpolationKind.nearest:
            # Binary search for interval
            # idx such that x[idx] <= x < x[idx+1]
            idx = np.searchsorted(self._x, x, side="right") - 1
            idx = max(0, min(idx, len(self._x) - 2))

            # Check distances
            dist_left = abs(x - self._x[idx])
            dist_right = abs(x - self._x[idx + 1])
            if dist_left <= dist_right:
                return float(self._y[idx])
            else:
                return float(self._y[idx + 1])

        # Fallback (should not be reached)
        return float(np.interp(x, self._x, self._y))

    def _interpolate_exactly(
        self, x: Fraction, xs: list[Fraction], ys: list[Fraction]
    ) -> Fraction | None:
        """The exact linear result at *x*, or ``None`` where no arithmetic applies.

        ``None`` leaves the step kinds and the policies other than linear
        extrapolation outside the anchors to the float path, which answers
        them without arithmetic on the anchors.
        """
        inside = xs[0] <= x <= xs[-1]
        if self._kind != InterpolationKind.linear or (
            not inside and self._extrapolate != ExtrapolationPolicy.extrapolate
        ):
            return None
        # the segment [xs[i], xs[i + 1]] holding x; outside the anchors, the
        # first or the last segment
        i = min(max(bisect_right(xs, x) - 1, 0), len(xs) - 2)
        x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert an array using vectorized interpolation."""
        x = values.astype(np.float64)

        # Handle interpolation based on kind
        if self._kind == InterpolationKind.linear:
            result = np.interp(x, self._x, self._y)
        elif self._kind == InterpolationKind.nearest:
            # Find nearest x value for each input
            indices = np.abs(x[:, None] - self._x[None, :]).argmin(axis=1)
            result = self._y[indices]
        elif self._kind == InterpolationKind.previous:
            # Find rightmost x <= input
            indices = np.searchsorted(self._x, x, side="right") - 1
            indices = np.clip(indices, 0, len(self._x) - 1)
            result = self._y[indices]
        elif self._kind == InterpolationKind.next:
            # Find leftmost x >= input
            indices = np.searchsorted(self._x, x, side="left")
            indices = np.clip(indices, 0, len(self._x) - 1)
            result = self._y[indices]
        else:
            # Fallback to linear
            result = np.interp(x, self._x, self._y)

        # Handle out of bounds based on policy
        if self._extrapolate != ExtrapolationPolicy.extrapolate:
            below_min = x < self._x[0]
            above_max = x > self._x[-1]

            if self._extrapolate == ExtrapolationPolicy.error:
                if np.any(below_min) or np.any(above_max):
                    raise ValueError(
                        f"Values outside table bounds [{self._x[0]}, {self._x[-1]}]"
                    )
            elif self._extrapolate == ExtrapolationPolicy.constant:
                result[below_min] = self._y[0]
                result[above_max] = self._y[-1]
            elif self._extrapolate == ExtrapolationPolicy.nan:
                result[below_min] = np.nan
                result[above_max] = np.nan

        return result

    def _handle_out_of_bounds(self, x: float) -> CoordinateValue:
        """Handle a single out-of-bounds value."""
        if self._extrapolate == ExtrapolationPolicy.error:
            raise ValueError(
                f"Value {x} is outside table bounds [{self._x[0]}, {self._x[-1]}]"
            )
        elif self._extrapolate == ExtrapolationPolicy.constant:
            if x < self._x[0]:
                return self._y_original[0]
            return self._y_original[-1]
        elif self._extrapolate == ExtrapolationPolicy.nan:
            return float("nan")
        else:
            # Extrapolate linearly using the slope of the first or last segment
            if x < self._x[0]:
                slope = (self._y[1] - self._y[0]) / (self._x[1] - self._x[0])
                return float(self._y[0] + slope * (x - self._x[0]))
            else:
                slope = (self._y[-1] - self._y[-2]) / (self._x[-1] - self._x[-2])
                return float(self._y[-1] + slope * (x - self._x[-1]))

    def inverse(self) -> Self:
        """Return the inverse map (swap x and y).

        Returns:
            A new TableMap with swapped coordinates.

        Raises:
            NotImplementedError: If y values are not strictly monotonic.
        """
        if not self._is_invertible:
            raise NotImplementedError(
                "Cannot invert TableMap: y values are not strictly monotonic"
            )

        # If y is decreasing, we need to reverse the arrays to maintain increasing x
        if self._y_decreasing:
            return self.__class__(
                x_values=list(reversed(self._y_original)),
                y_values=list(reversed(self._x_original)),
                kind=self._kind,
                extrapolate=self._extrapolate,
                source_unit=self._target_unit,
                target_unit=self._source_unit,
                output_number_type=self._inverse_output_number_type(),
            )
        else:
            return self.__class__(
                x_values=self._y_original,
                y_values=self._x_original,
                kind=self._kind,
                extrapolate=self._extrapolate,
                source_unit=self._target_unit,
                target_unit=self._source_unit,
                output_number_type=self._inverse_output_number_type(),
            )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``TableMap``; ``error`` and ``nan`` both write ``none``.

        Raises:
            ValueError: If the kind is ``nearest`` or ``next``, which the
                format does not define.
        """
        if self._kind not in _KINDS:
            raise self._no_form(
                f"the format defines no TableMap of kind {self._kind.value!r}"
            )
        return _table_form(
            self,
            self._x_original,
            self._y_original,
            kind=self._kind,
            extrapolate=self._extrapolate,
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> TableMap:
        """Read a ``TableMap`` from its complete form.

        Raises:
            ValueError: If ``kind`` or ``extrapolate`` names no value the
                format defines.
        """
        kind = data["kind"]
        if kind not in {member.value for member in _KINDS}:
            raise ValueError(f"a TableMap has an unknown kind {kind!r}")
        extrapolate = data["extrapolate"]
        if extrapolate not in _EXTRAPOLATION_READ:
            raise ValueError(f"a TableMap has an unknown extrapolate {extrapolate!r}")
        return cls(
            x_values=[number_from_json(v) for v in data["x_values"]],
            y_values=[number_from_json(v) for v in data["y_values"]],
            kind=kind,
            extrapolate=_EXTRAPOLATION_READ[extrapolate],
            source_unit=data.get("source_unit"),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        n = len(self._x)
        parts = [f"n_points={n}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        parts.append(f"kind={self._kind.name}")
        return f"TableMap({', '.join(parts)})"

    @classmethod
    def from_tempo_changes(
        cls,
        tick_positions: Sequence[int],
        tempos_bpm: Sequence[float],
        ticks_per_quarter: int = 480,
        source_unit: TimeUnit | str = "ticks",
        target_unit: TimeUnit | str = "seconds",
        *,
        seed: int | None = None,
    ) -> TableMap:
        """Create a TableMap from MIDI-style tempo changes.

        This is a convenience constructor for the common case of converting
        MIDI ticks to seconds based on tempo information.

        Args:
            tick_positions: Tick positions where tempo changes occur.
                           First should typically be 0.
            tempos_bpm: Tempo in BPM at each position.
            ticks_per_quarter: MIDI resolution (ticks per quarter note).
            source_unit: Source unit name.
            target_unit: Target unit name.
            seed: The seed of the minter of the map's id; not stored.

        Returns:
            A TableMap for tick-to-second conversion.

        Examples:
            >>> # Tempo starts at 120 BPM, changes to 60 BPM at tick 960
            >>> tempo_map = TableMap.from_tempo_changes(
            ...     tick_positions=[0, 960],
            ...     tempos_bpm=[120.0, 60.0],
            ...     ticks_per_quarter=480,
            ... )
        """
        if len(tick_positions) != len(tempos_bpm):
            raise ValueError("tick_positions and tempos_bpm must have same length")
        if len(tick_positions) == 0:
            raise ValueError("At least one tempo point is required")

        # Build cumulative time map
        x_values: list[int] = []
        y_values: list[float] = []

        current_time = 0.0

        for i, (tick, bpm) in enumerate(zip(tick_positions, tempos_bpm)):
            x_values.append(tick)
            y_values.append(current_time)

            # Calculate seconds per tick for this tempo region
            # BPM = beats per minute, so seconds per beat = 60 / BPM
            # seconds per tick = (60 / BPM) / ticks_per_quarter
            if i < len(tick_positions) - 1:
                next_tick = tick_positions[i + 1]
                seconds_per_tick = (60.0 / bpm) / ticks_per_quarter
                duration = (next_tick - tick) * seconds_per_tick
                current_time += duration

        # Add a final point to allow extrapolation
        # Use the last tempo to project forward
        # Always add this point to capture the last tempo slope
        final_tick = tick_positions[-1] + ticks_per_quarter * 4  # 4 beats ahead
        seconds_per_tick = (60.0 / tempos_bpm[-1]) / ticks_per_quarter
        x_values.append(final_tick)
        y_values.append(
            current_time + (final_tick - tick_positions[-1]) * seconds_per_tick
        )

        return cls(
            x_values=x_values,
            y_values=y_values,
            kind=InterpolationKind.linear,
            extrapolate=ExtrapolationPolicy.extrapolate,
            source_unit=source_unit,
            target_unit=target_unit,
            seed=seed,
        )
