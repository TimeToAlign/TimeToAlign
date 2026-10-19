"""Linear transformation maps.

This module provides maps that perform affine transformations:
- LinearMap: y = ax + b (scaling and offset)
- ScalarMap: y = ax (scaling only)
- ShiftMap: y = x + b (offset only)
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING, Any, ClassVar

from numpy.typing import NDArray

from ..core.enums import TimeUnit
from ..core.time import (
    CoordinateValue,
    number_from_json,
    number_to_json,
)
from .base import ConversionMap, MeasureMapIds

if TYPE_CHECKING:
    from typing_extensions import Self


def _reciprocal(value: CoordinateValue) -> CoordinateValue:
    """Return an exact reciprocal when the input is rational."""
    if isinstance(value, (int, Fraction)):
        return Fraction(1, 1) / value
    return 1 / value


def _linear_form(
    cmap: ConversionMap[Any],
    scalar: CoordinateValue,
    offset: CoordinateValue,
    *,
    minimal: bool,
) -> dict[str, Any]:
    """Write *cmap* as the format's ``LinearMap`` of *scalar* and *offset*."""
    return cmap._map_object(
        "LinearMap",
        {"scalar": number_to_json(scalar), "offset": number_to_json(offset)},
        minimal=minimal,
        defaults=LinearMap._dict_defaults,
    )


# region LinearMap


class LinearMap(ConversionMap[CoordinateValue]):
    """Affine transformation: y = scalar * x + offset.

    This is the most general linear map, combining scaling and offset.

    The format's ``LinearMap`` type, which every affine map of the library
    is written as.

    Attributes:
        scalar: The multiplicative factor (default 1).
        offset: The additive offset (default 0).

    Examples:
        >>> # Convert ticks to quarters: y = x / 480
        >>> tick_to_quarter = LinearMap(scalar=1/480, source_unit="ticks", target_unit="quarters")
        >>> tick_to_quarter(480)
        1.0

        >>> # Temperature conversion: Celsius to Fahrenheit: y = 1.8x + 32
        >>> c_to_f = LinearMap(scalar=1.8, offset=32)
        >>> c_to_f(0)
        32.0
        >>> c_to_f(100)
        212.0

        >>> # Inverse map
        >>> f_to_c = c_to_f.inverse()
        >>> f_to_c(212.0)
        100.0
    """

    _id_prefix: ClassVar[str] = "lmap"
    _manifest_type: ClassVar[str | None] = "LinearMap"
    # The integer defaults leave an exact input exact: 1.0 * x is a float.
    _dict_defaults: ClassVar[dict[str, Any]] = {"scalar": 1, "offset": 0}

    def __init__(
        self,
        *,
        scalar: CoordinateValue = 1,
        offset: CoordinateValue = 0,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a LinearMap.

        Args:
            scalar: The multiplicative factor. Must be non-zero for invertibility.
            offset: The additive offset applied after scaling.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Used as field header in
                timestamp tables. Defaults to "source_to_target".

        Raises:
            ValueError: If scalar is zero (map would not be invertible).
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )
        if scalar == 0:
            raise ValueError("LinearMap scalar cannot be zero (not invertible)")
        self._scalar = scalar
        self._offset = offset

    @property
    def scalar(self) -> CoordinateValue:
        """The multiplicative factor."""
        return self._scalar

    @property
    def offset(self) -> CoordinateValue:
        """The additive offset."""
        return self._offset

    @property
    def is_identity(self) -> bool:
        """Whether this map is the identity transformation."""
        return self._scalar == 1 and self._offset == 0

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Apply the linear transformation: y = ax + b."""
        return self._scalar * value + self._offset

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Optimized array conversion."""
        return self._scalar * values + self._offset

    def inverse(self) -> Self:
        """Return the inverse map: y = (x - b) / a."""
        # Inverse of y = ax + b is x = (y - b) / a = (1/a)y - b/a
        inv_scalar = _reciprocal(self._scalar)
        inv_offset = -self._offset * inv_scalar
        return self.__class__(
            scalar=inv_scalar,
            offset=inv_offset,
            source_unit=self._target_unit,
            target_unit=self._source_unit,
        )

    def compose_with(self, other: LinearMap) -> LinearMap:
        """Compose with another LinearMap: (self then other).

        For y1 = a1*x + b1 and y2 = a2*y1 + b2:
        y2 = a2*(a1*x + b1) + b2 = (a1*a2)*x + (a2*b1 + b2)

        Args:
            other: The map to apply after this one.

        Returns:
            A new LinearMap equivalent to applying both in sequence.
        """
        new_scalar = self._scalar * other._scalar
        new_offset = other._scalar * self._offset + other._offset
        return LinearMap(
            scalar=new_scalar,
            offset=new_offset,
            source_unit=self._source_unit,
            target_unit=other._target_unit,
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``LinearMap`` with this map's scalar and offset."""
        return _linear_form(self, self._scalar, self._offset, minimal=minimal)

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> LinearMap:
        """Read a ``LinearMap``, ``scalar`` 1 and ``offset`` 0 when left out."""
        return cls(
            scalar=number_from_json(data.get("scalar", 1)),
            offset=number_from_json(data.get("offset", 0)),
            source_unit=data.get("source_unit"),
            target_unit=data.get("target_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        parts = [
            f"scalar={self._scalar!r}",
            f"offset={self._offset!r}",
        ]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        return f"LinearMap({', '.join(parts)})"


# endregion

# region ScalarMap


class ScalarMap(ConversionMap[CoordinateValue]):
    """Pure scaling transformation: y = scalar * x.

    A simplified version of LinearMap with no offset.

    Examples:
        >>> # Seconds to milliseconds
        >>> s_to_ms = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        >>> s_to_ms(1.5)
        1500.0

        >>> # Samples to seconds at 44100 Hz
        >>> samples_to_sec = ScalarMap(scalar=1/44100, source_unit="samples", target_unit="seconds")
        >>> samples_to_sec(44100)
        1.0
    """

    def __init__(
        self,
        scalar: CoordinateValue = 1.0,
        *,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ScalarMap.

        Args:
            scalar: The multiplicative factor. Must be non-zero.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Used as field header in
                timestamp tables. Defaults to "source_to_target" (e.g.,
                "pixels_to_inches").

        Raises:
            ValueError: If scalar is zero.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )
        if scalar == 0:
            raise ValueError("ScalarMap scalar cannot be zero")
        self._scalar = scalar

    @property
    def scalar(self) -> CoordinateValue:
        """The multiplicative factor."""
        return self._scalar

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Apply the scaling: y = ax."""
        return self._scalar * value

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Optimized array conversion."""
        return self._scalar * values

    def inverse(self) -> Self:
        """Return the inverse map: y = x / a."""
        return self.__class__(
            scalar=_reciprocal(self._scalar),
            source_unit=self._target_unit,
            target_unit=self._source_unit,
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``LinearMap`` with this map's scalar and offset 0."""
        return _linear_form(self, self._scalar, 0, minimal=minimal)

    def __repr__(self) -> str:
        parts = [f"scalar={self._scalar!r}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        return f"ScalarMap({', '.join(parts)})"


# endregion

# region ShiftMap


class ShiftMap(ConversionMap[CoordinateValue]):
    """Pure offset transformation: y = x + offset.

    A simplified version of LinearMap with scalar=1.

    Examples:
        >>> # Add an offset to timestamps
        >>> shift = ShiftMap(offset=10.0)
        >>> shift(5.0)
        15.0

        >>> # Can be used to align timelines with different origins
        >>> align = ShiftMap(offset=-100)  # Subtract 100 from all coordinates
        >>> align(150)
        50
    """

    def __init__(
        self,
        *,
        offset: CoordinateValue = 0,
        source_unit: TimeUnit | str | None = None,
        target_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ShiftMap.

        Args:
            offset: The additive offset.
            source_unit: The unit of input coordinates.
            target_unit: The unit of output coordinates (typically same as source).
            uid: Optional explicit ID.
            name: Human-readable name for this map. Used as field header in
                timestamp tables. Defaults to "source_to_target".
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=target_unit,
            uid=uid,
            name=name,
        )
        self._offset = offset

    @property
    def offset(self) -> CoordinateValue:
        """The additive offset."""
        return self._offset

    @property
    def is_identity(self) -> bool:
        """Whether this map is the identity transformation."""
        return self._offset == 0

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> CoordinateValue:
        """Apply the shift: y = x + b."""
        return value + self._offset

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Optimized array conversion."""
        return values + self._offset

    def inverse(self) -> Self:
        """Return the inverse map: y = x - b."""
        return self.__class__(
            offset=-self._offset,
            source_unit=self._target_unit,
            target_unit=self._source_unit,
        )

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``LinearMap`` with scalar 1 and this map's offset."""
        return _linear_form(self, 1, self._offset, minimal=minimal)

    def __repr__(self) -> str:
        parts = [f"offset={self._offset!r}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._target_unit:
            parts.append(f"target_unit={self._target_unit}")
        return f"ShiftMap({', '.join(parts)})"


# endregion
