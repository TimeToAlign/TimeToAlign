"""Composite conversion maps.

This module provides maps that are composed of other maps:
- ChainMap: Applies a sequence of maps (f(g(x)))
- PiecewiseMap: Applies different maps based on the input coordinate
"""

from __future__ import annotations

import bisect
from typing import TYPE_CHECKING, Any, ClassVar, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.enums import TimeUnit
from ..core.time import (
    CoordinateValue,
    number_from_json,
    number_to_json,
)
from .base import ConversionMap

if TYPE_CHECKING:
    from typing_extensions import Self


# region ChainMap


class ChainMap(ConversionMap[CoordinateValue]):
    """Composition of multiple maps: f(g(h(x))).

    The maps are applied in sequence from first to last.
    The output unit of one map must match the input unit of the next
    (if specified).

    Attributes:
        maps: The sequence of maps to apply.

    Examples:
        >>> # Ticks -> Seconds -> Milliseconds
        >>> ticks_to_secs = ScalarMap(scalar=1/480, source_unit="ticks", target_unit="seconds")
        >>> secs_to_ms = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        >>> chain = ChainMap([ticks_to_secs, secs_to_ms])
        >>> chain(480)
        1000.0
    """

    def __init__(
        self,
        maps: Sequence[ConversionMap[Any]],
        *,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ChainMap.

        Args:
            maps: Sequence of maps. Must not be empty.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.

        Raises:
            ValueError: If maps is empty.
            ValueError: If units of adjacent maps are incompatible.
        """
        if not maps:
            raise ValueError("ChainMap requires at least one map")

        # Validate unit compatibility
        for i in range(len(maps) - 1):
            curr_map = maps[i]
            next_map = maps[i + 1]
            if (
                curr_map.target_unit is not None
                and next_map.source_unit is not None
                and curr_map.target_unit != next_map.source_unit
            ):
                raise ValueError(
                    f"Incompatible units at index {i}: "
                    f"Map {curr_map.id} outputs {curr_map.target_unit}, "
                    f"but Map {next_map.id} expects {next_map.source_unit}"
                )

        super().__init__(
            source_unit=maps[0].source_unit,
            target_unit=maps[-1].target_unit,
            uid=uid,
            name=name,
        )
        self._maps = list(maps)

    @property
    def maps(self) -> list[ConversionMap[Any]]:
        """The sequence of maps."""
        return list(self._maps)

    @property
    def is_invertible(self) -> bool:
        """Whether the chain is invertible (all sub-maps must be invertible)."""
        return all(m.is_invertible for m in self._maps)

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Apply maps in sequence."""
        result = value
        for m in self._maps:
            # We explicitly handle scalar/coordinate conversion in the base __call__
            # so we can use the internal _convert_scalar or __call__ of sub-maps.
            # Since maps can return T or NDArray, and we want scalar here,
            # we rely on the map's behavior.
            # Note: sub-maps might be flexible, so we pass kwargs.
            if isinstance(m, ConversionMap):
                # We're inside _convert_scalar, so we have a raw value.
                # Sub-maps expect raw value or Coordinate.
                # We pass the raw value.
                if hasattr(m, "_convert_scalar"):
                    result = m._convert_scalar(result, **kwargs)
                else:
                    # Fallback to public interface if not our subclass (unlikely)
                    result = m(result, **kwargs)
            else:
                # Should not happen given typing, but safe fallback
                result = m(result)
        return result

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Apply maps in sequence to array."""
        result = values
        for m in self._maps:
            if hasattr(m, "_convert_array"):
                result = m._convert_array(result, **kwargs)
            else:
                result = m(result, **kwargs)
        return result

    def inverse(self) -> Self:
        """Return the inverse chain.

        This reverses the order of maps and inverts each one.
        (f(g(x)))^-1 = g^-1(f^-1(x))
        """
        if not self.is_invertible:
            raise NotImplementedError("ChainMap is not invertible")

        inv_maps = [m.inverse() for m in reversed(self._maps)]
        return self.__class__(inv_maps, uid=f"inv_{self.id}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d["maps"] = [m.to_dict() for m in self._maps]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChainMap:
        """Deserialize from dictionary."""
        from .base import ConversionMap

        maps_data = data.get("maps", [])
        maps = [ConversionMap.from_dict(m_data) for m_data in maps_data]

        return cls(
            maps=maps,
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"ChainMap(steps={len(self._maps)})"


# endregion

# region PiecewiseMap


class PiecewiseMap(ConversionMap[CoordinateValue]):
    """Map defined by different maps on different intervals.

    Also known as ConcatenationMap.
    Defined by a set of break points and the maps to use between them.

    Intervals are [start, end).

    Attributes:
        breaks: Sorted list of break points. N breaks define N+1 regions (or N regions).
                Here we strictly define regions between breaks.
                If we have breaks [0, 10, 20], we have regions [0, 10) and [10, 20).
                Anything outside is handled by 'extrapolate' or explicit outer maps.
        maps: Maps for each interval. len(maps) must be len(breaks) - 1.
    """

    _id_prefix: ClassVar[str] = "pmap"

    def __init__(
        self,
        *,
        breaks: Sequence[CoordinateValue],
        maps: Sequence[ConversionMap[Any]],
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a PiecewiseMap.

        Args:
            breaks: Sorted list of interval boundaries.
            maps: List of maps, one for each interval between breaks.
            source_unit: Input unit.
            target_unit: Output unit.
            uid: Explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.

        Raises:
            ValueError: If lengths mismatch or breaks are not sorted.
        """
        if len(breaks) < 2:
            raise ValueError("PiecewiseMap requires at least 2 break points")
        if len(maps) != len(breaks) - 1:
            raise ValueError(
                f"Number of maps ({len(maps)}) must be len(breaks)-1 ({len(breaks)-1})"
            )

        # Verify sorting
        breaks_float = [float(b) for b in breaks]
        if not all(x < y for x, y in zip(breaks_float, breaks_float[1:])):
            raise ValueError("Breaks must be strictly increasing")

        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )

        self._breaks = np.array(breaks_float, dtype=np.float64)
        self._breaks_orig = list(breaks)
        self._maps = list(maps)

    @property
    def breaks(self) -> NDArray[np.floating[Any]]:
        """The interval boundaries."""
        return self._breaks

    @property
    def maps(self) -> list[ConversionMap[Any]]:
        """The maps for each interval."""
        return list(self._maps)

    def _get_map_index(self, x: float) -> int:
        """Find the interval index for x."""
        # bisect_right returns insertion point to maintain order.
        # breaks=[0, 10, 20]
        # x=5 -> idx=1 -> map 0 (intervals are 0-10, 10-20)
        # x=10 -> idx=2 -> map 1
        # x=20 -> idx=3 -> out of bounds (unless we handle inclusive end)

        # Using searchsorted equivalent logic
        # We want index i such that breaks[i] <= x < breaks[i+1]

        if x < self._breaks[0] or x >= self._breaks[-1]:
            raise ValueError(
                f"Value {x} out of bounds [{self._breaks[0]}, {self._breaks[-1]})"
            )

        idx = bisect.bisect_right(self._breaks, x) - 1
        return max(0, min(idx, len(self._maps) - 1))

    def _convert_scalar(
        self, value: CoordinateValue, allow_upper_bound: bool = False, **kwargs: Any
    ) -> CoordinateValue:
        """Convert scalar by selecting the appropriate map."""
        x = float(value)

        # Check bounds
        if x < self._breaks[0]:
            raise ValueError(
                f"Value {x} out of bounds [{self._breaks[0]}, {self._breaks[-1]})"
            )

        # Special case for upper bound if allowed (used for inversion)
        if allow_upper_bound and x == self._breaks[-1]:
            idx = len(self._maps) - 1
        elif x >= self._breaks[-1]:
            raise ValueError(
                f"Value {x} out of bounds [{self._breaks[0]}, {self._breaks[-1]})"
            )
        else:
            idx = self._get_map_index(x)

        m = self._maps[idx]

        if hasattr(m, "_convert_scalar"):
            return m._convert_scalar(value, **kwargs)
        return m(value, **kwargs)

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert array by processing chunks."""
        x = values.astype(np.float64)
        result = np.empty_like(x)

        # Optimize using searchsorted
        indices = np.searchsorted(self._breaks, x, side="right") - 1

        # Clip indices to valid range for map lookup, but mask out-of-bounds later
        map_indices = np.clip(indices, 0, len(self._maps) - 1)

        # Identify valid range
        valid_mask = (x >= self._breaks[0]) & (x < self._breaks[-1])
        if not np.all(valid_mask):
            # Simple policy: raise error for now, could support extrapolation
            raise ValueError("Values outside map bounds")

        # Iterate over unique maps involved
        unique_indices = np.unique(map_indices)

        for idx in unique_indices:
            mask = (map_indices == idx) & valid_mask
            if not np.any(mask):
                continue

            subset = values[mask]
            m = self._maps[idx]

            if hasattr(m, "_convert_array"):
                converted = m._convert_array(subset, **kwargs)
            else:
                converted = m(subset, **kwargs)

            result[mask] = converted

        return result

    def inverse(self) -> Self:
        """Return the inverse map.

        Requires all sub-maps to be invertible and the resulting target intervals to be contiguous.
        This is non-trivial if the target intervals don't line up.
        For now, we implement the naive inversion:
        Convert breaks using forward maps to get new breaks, invert maps.
        """
        # Convert breaks. Note: allow mapping the final break point.
        new_breaks = []
        for b in self._breaks_orig:
            new_breaks.append(self._convert_scalar(b, allow_upper_bound=True))

        # Note: Depending on map direction (increasing/decreasing), new_breaks might not be sorted.
        # We assume strictly increasing monotonic maps for simple inversion.

        # Check monotonicity
        if not all(x < y for x, y in zip(new_breaks, new_breaks[1:])):
            raise NotImplementedError("Cannot invert non-monotonic PiecewiseMap")

        inv_maps = [m.inverse() for m in self._maps]

        return self.__class__(
            breaks=new_breaks,
            maps=inv_maps,
            source_unit=self._target_unit,
            target_unit=self._source_unit,
            uid=f"inv_{self.id}",
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d = super().to_dict()
        d["breaks"] = [number_to_json(b) for b in self._breaks_orig]
        d["maps"] = [m.to_dict() for m in self._maps]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PiecewiseMap:
        """Deserialize from dictionary."""
        from .base import ConversionMap

        maps_data = data.get("maps", [])
        maps = [ConversionMap.from_dict(m_data) for m_data in maps_data]

        return cls(
            breaks=[number_from_json(b) for b in data.get("breaks", [])],
            maps=maps,
            source_unit=data.get("source_unit"),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        return f"PiecewiseMap(regions={len(self._maps)})"


# endregion
