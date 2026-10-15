"""InterpolationMap: bidirectional anchor-pair coordinate conversion.

This module provides InterpolationMap, a ConversionMap that performs
O(log n) bidirectional coordinate conversion using numpy.interp between
explicit source/target anchor points.

InterpolationMap is the internal engine of WarpMap alignment warping.
Parent-child coordinate conversion in Timeline uses exact offset arithmetic
instead, and the members of a TimelineGroup convert linearly and exactly
between their common start and end.

InterpolationMap is a full member of the ConversionMap family: it supports
the shared __call__/convert_array interface, inverse(), composition, and
to_dict/from_dict serialization, while remaining a lightweight building
block optimized for performance. Its coordinate arrays (source_coords,
target_coords) are read-only after construction, and inverse() caches the
computed inverse map so repeated calls return the same instance.

Design rationale:
- No table lookups for coordinate conversion - direct np.interp calls
- Precomputed sorted arrays enable O(log n) binary search
- Bidirectional: forward (source -> target) and inverse (target -> source)
- Coordinate arrays are read-only: safe to share between threads/contexts
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ..core.enums import TimeUnit
from .base import ConversionMap

module_logger = logging.getLogger(__name__)


class InterpolationMap(ConversionMap[float]):
    """Bidirectional coordinate mapping using numpy.interp.

    Provides O(log n) coordinate conversion without table lookup.
    Used for:
    - Timeline <-> Group relationships
    - WarpMap alignment warping

    The map is bidirectional: calling the map converts source -> target,
    and ``inverse()`` returns a new map that converts target -> source.

    Attributes:
        source_coords: Sorted source axis coordinates (float64). Read-only:
            the array is copied on construction and marked non-writable, so
            in-place mutation raises ``ValueError`` instead of silently
            desynchronizing the cached inverse.
        target_coords: Corresponding target values (float64). Read-only, for
            the same reason as source_coords.
        source_id: Timeline/C-Map ID for source.
        target_id: Timeline/C-Map ID for target.

    Examples:
        >>> # Simple offset relationship: child at offset 10 in parent
        >>> imap = InterpolationMap(
        ...     source_coords=np.array([0.0, 100.0]),  # child coords
        ...     target_coords=np.array([10.0, 110.0]),  # parent coords
        ...     source_id="child:1",
        ...     target_id="parent:1",
        ... )
        >>> imap(50.0)  # child 50 -> parent 60
        60.0
        >>> imap.inverse()(60.0)  # parent 60 -> child 50
        50.0

        >>> # Tempo map: ticks to seconds
        >>> imap = InterpolationMap(
        ...     source_coords=np.array([0.0, 480.0, 960.0]),
        ...     target_coords=np.array([0.0, 0.5, 1.5]),
        ...     source_id="ticks",
        ...     target_id="seconds",
        ... )
        >>> imap(240.0)  # 240 ticks -> 0.25 seconds
        0.25
    """

    def __init__(
        self,
        *,
        source_coords: NDArray[np.floating[Any]] | Any,
        target_coords: NDArray[np.floating[Any]] | Any,
        source_id: str,
        target_id: str,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize an InterpolationMap.

        Args:
            source_coords: Source axis coordinates. Must be strictly
                monotonically increasing.
            target_coords: Corresponding target values. Same length as
                source_coords.
            source_id: Timeline/C-Map ID for source.
            target_id: Timeline/C-Map ID for target.
            source_unit: The unit of source coordinates (optional).
            target_unit: The unit of target coordinates (optional).
            uid: Optional explicit ID.
            name: Optional human-readable name.

        Raises:
            ValueError: If source_coords and target_coords have different
                lengths, if fewer than 2 anchor points are given, or if
                source_coords is not strictly monotonically increasing.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )

        self.source_coords: NDArray[np.floating[Any]] = np.array(
            source_coords, dtype=np.float64, copy=True
        )
        self.source_coords.setflags(write=False)
        self.target_coords: NDArray[np.floating[Any]] = np.array(
            target_coords, dtype=np.float64, copy=True
        )
        self.target_coords.setflags(write=False)
        self.source_id = source_id
        self.target_id = target_id

        if len(self.source_coords) != len(self.target_coords):
            raise ValueError(
                f"source_coords and target_coords must have same length, "
                f"got {len(self.source_coords)} and {len(self.target_coords)}"
            )
        if len(self.source_coords) < 2:
            raise ValueError("InterpolationMap requires at least 2 anchor points")

        if not np.all(np.diff(self.source_coords) > 0):
            raise ValueError("source_coords must be strictly monotonically increasing")

        self._inverse: InterpolationMap | None = None

    @property
    def n_anchors(self) -> int:
        """Number of anchor points."""
        return len(self.source_coords)

    @property
    def source_min(self) -> float:
        """Minimum source coordinate."""
        return float(self.source_coords[0])

    @property
    def source_max(self) -> float:
        """Maximum source coordinate."""
        return float(self.source_coords[-1])

    @property
    def target_min(self) -> float:
        """Minimum target coordinate."""
        return float(np.min(self.target_coords))

    @property
    def target_max(self) -> float:
        """Maximum target coordinate."""
        return float(np.max(self.target_coords))

    @property
    def is_invertible(self) -> bool:
        """Whether this map can be inverted.

        True if target_coords are strictly monotonic (increasing or decreasing).
        """
        diff = np.diff(self.target_coords)
        return bool(np.all(diff > 0) or np.all(diff < 0))

    # region Conversion

    @staticmethod
    def _interp_with_extrapolation(
        values: float | NDArray[np.floating[Any]],
        xp: NDArray[np.floating[Any]],
        fp: NDArray[np.floating[Any]],
    ) -> float | NDArray[np.floating[Any]]:
        """Interpolate with linear extrapolation outside bounds.

        numpy.interp clips values to the boundary - this method extends
        linearly using the slope at the boundary.

        Args:
            values: Values to interpolate.
            xp: X coordinates (must be increasing).
            fp: Y coordinates.

        Returns:
            Interpolated/extrapolated values.
        """
        values_arr = np.atleast_1d(values)
        result = np.interp(values_arr, xp, fp)

        # Extrapolate below minimum
        below_mask = values_arr < xp[0]
        if np.any(below_mask):
            slope_left = (fp[1] - fp[0]) / (xp[1] - xp[0])
            result[below_mask] = fp[0] + slope_left * (values_arr[below_mask] - xp[0])

        # Extrapolate above maximum
        above_mask = values_arr > xp[-1]
        if np.any(above_mask):
            slope_right = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
            result[above_mask] = fp[-1] + slope_right * (
                values_arr[above_mask] - xp[-1]
            )

        # Return scalar if input was scalar
        if np.isscalar(values):
            return float(result[0])
        return result

    def _convert_scalar(self, value: Any, **kwargs: Any) -> float:
        """Convert a single source value to target via interpolation."""
        return float(
            self._interp_with_extrapolation(
                float(value), self.source_coords, self.target_coords
            )
        )

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Convert an array of source values to target via interpolation."""
        return self._interp_with_extrapolation(
            values, self.source_coords, self.target_coords
        )

    def matches_selector(self, key: str) -> bool:
        """Return whether a user conversion-map selector addresses this map.

        In addition to the id/name match provided by the family base class,
        group converter maps are addressed by their source timeline id in
        conversion-map specifications, since InterpolationMap instances are
        auto-generated and have no meaningful user-facing id or name.

        Args:
            key: A selector string from a conversion-map specification.

        Returns:
            True if key equals this map's id, name, or source_id.
        """
        return super().matches_selector(key) or key == self.source_id

    # endregion

    # region Inverse

    def inverse(self) -> Self:
        """Return the inverse map (target -> source).

        The inverse is cached: repeated calls return the same instance,
        and the returned map's own ``inverse()`` returns back the original.

        Returns:
            A new InterpolationMap with source and target swapped.

        Raises:
            ValueError: If target_coords are not strictly monotonic.
        """
        if self._inverse is not None:
            return self._inverse

        if not self.is_invertible:
            raise ValueError("Cannot invert: target_coords are not strictly monotonic")

        if self.target_coords[0] > self.target_coords[-1]:
            new_source = self.target_coords[::-1].copy()
            new_target = self.source_coords[::-1].copy()
        else:
            new_source = self.target_coords.copy()
            new_target = self.source_coords.copy()

        inv = InterpolationMap(
            source_coords=new_source,
            target_coords=new_target,
            source_id=self.target_id,
            target_id=self.source_id,
            source_unit=self._target_unit,
            target_unit=self._source_unit,
        )
        inv._inverse = self
        self._inverse = inv
        return inv

    # endregion

    # region Factory Methods

    @classmethod
    def identity(
        cls,
        start: float = 0.0,
        end: float = 1.0,
        timeline_id: str = "identity",
    ) -> InterpolationMap:
        """Create an identity map (output = input).

        Useful for testing or placeholder mappings.

        Args:
            start: Start coordinate.
            end: End coordinate.
            timeline_id: ID to use for both source and target.

        Returns:
            InterpolationMap where map(x) == x.
        """
        coords = np.array([start, end], dtype=np.float64)
        return cls(
            source_coords=coords,
            target_coords=coords.copy(),
            source_id=timeline_id,
            target_id=timeline_id,
        )

    # endregion

    # region Serialization

    def to_dict(self, *, minimal: bool = False) -> dict[str, Any]:
        """Serialize the map to a dictionary.

        Returns:
            Dictionary representation of the map.
        """
        d = super().to_dict(minimal=minimal)
        d["source_coords"] = [float(v) for v in self.source_coords]
        d["target_coords"] = [float(v) for v in self.target_coords]
        d["source_id"] = self.source_id
        d["target_id"] = self.target_id
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterpolationMap:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            A new InterpolationMap instance.
        """
        return cls(
            source_coords=np.array(data["source_coords"], dtype=np.float64),
            target_coords=np.array(data["target_coords"], dtype=np.float64),
            source_id=data["source_id"],
            target_id=data["target_id"],
            source_unit=data.get("source_unit"),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    # endregion

    # region Display

    def __repr__(self) -> str:
        return (
            f"InterpolationMap(source={self.source_id!r}, target={self.target_id!r}, "
            f"n_anchors={self.n_anchors})"
        )

    def __str__(self) -> str:
        return f"InterpolationMap({self.source_id} -> {self.target_id})"

    # endregion
