"""Periodic and floor-based conversion maps.

This module provides maps for periodic/cyclic patterns and integer division:
- RotationMap: Periodic patterns via modular arithmetic (beat rotation, angles, etc.)
- FloorMap: Integer floor division (measure numbers, page numbers)

These are reusable building blocks for metrical grids and other cyclic patterns.
They have no TimeToShare form: a timeline holding one cannot be written to a
manifest.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core.enums import NumberType, TimeUnit
from ..core.time import CoordinateValue
from .base import ConversionMap

if TYPE_CHECKING:
    from typing_extensions import Self


# region RotationMap


class RotationMap(ConversionMap[float]):
    """Map that applies modular arithmetic for cyclic patterns.

    Output = ((input - offset) % period) * scale + base

    This is a many-to-one map (NOT invertible) that produces periodic outputs.
    Useful for any cyclic pattern where the output wraps around.

    Attributes:
        period: The period of the cycle (before scaling).
        scale: Multiplier applied after modulo operation.
        base: Value added to the result (output offset).
        offset: Value subtracted from input before modulo (input offset).

    Examples:
        >>> # Beat in 4/4 measure (1-indexed, quarter-note beats)
        >>> beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)
        >>> beat_map(0.0)   # quarter 0 -> beat 1
        1.0
        >>> beat_map(1.0)   # quarter 1 -> beat 2
        2.0
        >>> beat_map(4.0)   # quarter 4 -> beat 1 (wraps!)
        1.0
        >>> beat_map(4.5)   # quarter 4.5 -> beat 1.5
        1.5

        >>> # Beat in 3/4 measure
        >>> beat_map_3_4 = RotationMap(period=3.0, scale=1.0, base=1.0)
        >>> beat_map_3_4(3.0)  # wraps to beat 1
        1.0

        >>> # Angle normalization (0-360)
        >>> angle_map = RotationMap(period=360.0)
        >>> angle_map(450.0)
        90.0

        >>> # Day of week (0-6) from hours
        >>> day_map = RotationMap(period=168.0, scale=1/24)
        >>> day_map(48.0)  # 2 days
        2.0
    """

    def __init__(
        self,
        period: float,
        scale: float = 1.0,
        base: float = 0.0,
        offset: float = 0.0,
        *,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a RotationMap.

        Args:
            period: The period of the cycle (must be positive).
            scale: Multiplier applied after modulo operation. Default 1.0.
            base: Value added to the result. Default 0.0.
            offset: Value subtracted from input before modulo. Default 0.0.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.

        Raises:
            ValueError: If period is not positive.
        """
        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")

        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )
        self._period = float(period)
        self._scale = float(scale)
        self._base = float(base)
        self._offset = float(offset)

    @property
    def period(self) -> float:
        """The period of the cycle."""
        return self._period

    @property
    def scale(self) -> float:
        """The scaling factor applied after modulo."""
        return self._scale

    @property
    def base(self) -> float:
        """The base value added to output."""
        return self._base

    @property
    def offset(self) -> float:
        """The offset subtracted from input before modulo."""
        return self._offset

    @property
    def is_invertible(self) -> bool:
        """RotationMap is NOT invertible (many-to-one)."""
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> float:
        """Apply periodic transformation to a single value."""
        x = float(value)
        return ((x - self._offset) % self._period) * self._scale + self._base

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Apply periodic transformation to an array (vectorized)."""
        x = values.astype(np.float64)
        return ((x - self._offset) % self._period) * self._scale + self._base

    def inverse(self) -> Self:
        """RotationMap is not invertible.

        Raises:
            NotImplementedError: Always, as this map is many-to-one.
        """
        raise NotImplementedError(
            "RotationMap is not invertible: multiple inputs map to the same output"
        )

    def __repr__(self) -> str:
        parts = [f"period={self._period}"]
        if self._scale != 1.0:
            parts.append(f"scale={self._scale}")
        if self._base != 0.0:
            parts.append(f"base={self._base}")
        if self._offset != 0.0:
            parts.append(f"offset={self._offset}")
        return f"RotationMap({', '.join(parts)})"


# endregion


# region FloorMap


class FloorMap(ConversionMap[int]):
    """Map that applies floor division for integer results.

    Output = floor((input - offset) / divisor) + base

    This map produces integer outputs, useful for measure numbers,
    page numbers, or any discrete counting based on a continuous input.

    Attributes:
        divisor: The divisor for floor division (must be positive).
        base: The integer added to the result (typically 1 for 1-indexed).
        offset: Value subtracted from input before division.

    Examples:
        >>> # Measure number from quarters (4/4 time, 1-indexed)
        >>> measure_map = FloorMap(divisor=4.0, base=1)
        >>> measure_map(0.0)    # quarter 0 -> measure 1
        1
        >>> measure_map(3.99)   # quarter 3.99 -> measure 1
        1
        >>> measure_map(4.0)    # quarter 4 -> measure 2
        2
        >>> measure_map(7.5)    # quarter 7.5 -> measure 2
        2

        >>> # Page number from pixel position (1000 px/page, 1-indexed)
        >>> page_map = FloorMap(divisor=1000, base=1)
        >>> page_map(500)
        1
        >>> page_map(1500)
        2

        >>> # 0-indexed sections
        >>> section_map = FloorMap(divisor=10.0, base=0)
        >>> section_map(25.0)
        2
    """

    # Counting which bucket a value falls into gives a whole number by
    # definition, whatever unit is nominally attached to the buckets.
    _declared_output_number_type = NumberType.int

    def __init__(
        self,
        divisor: float,
        base: int = 0,
        offset: float = 0.0,
        *,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a FloorMap.

        Args:
            divisor: The divisor for floor division (must be positive).
            base: Integer added to the result. Default 0.
            offset: Value subtracted from input before division. Default 0.0.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.

        Raises:
            ValueError: If divisor is not positive.
        """
        if divisor <= 0:
            raise ValueError(f"Divisor must be positive, got {divisor}")

        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )
        self._divisor = float(divisor)
        self._base = int(base)
        self._offset = float(offset)

    @property
    def divisor(self) -> float:
        """The divisor for floor division."""
        return self._divisor

    @property
    def base(self) -> int:
        """The base value added to output."""
        return self._base

    @property
    def offset(self) -> float:
        """The offset subtracted from input."""
        return self._offset

    @property
    def is_invertible(self) -> bool:
        """FloorMap is NOT invertible (many-to-one)."""
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> int:
        """Apply floor division to a single value."""
        x = float(value)
        return math.floor((x - self._offset) / self._divisor) + self._base

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Apply floor division to an array (vectorized)."""
        x = values.astype(np.float64)
        result = np.floor((x - self._offset) / self._divisor) + self._base
        return result.astype(np.int64)

    def inverse(self) -> Self:
        """FloorMap is not invertible.

        Raises:
            NotImplementedError: Always, as this map is many-to-one.
        """
        raise NotImplementedError(
            "FloorMap is not invertible: multiple inputs map to the same output"
        )

    def __repr__(self) -> str:
        parts = [f"divisor={self._divisor}"]
        if self._base != 0:
            parts.append(f"base={self._base}")
        if self._offset != 0.0:
            parts.append(f"offset={self._offset}")
        return f"FloorMap({', '.join(parts)})"


# endregion
