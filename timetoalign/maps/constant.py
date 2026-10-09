"""Constant-value conversion map.

This module provides ConstantMap, a ConversionMap that returns a fixed value
for any input coordinate. Useful for attaching metadata (e.g., filenames)
to every coordinate on a timeline, typically inside a CombinationMap.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from ..core.enums import TimeUnit
from ..core.time import (
    CoordinateValue,
    is_rational_dict,
    rational_from_dict,
    rational_to_dict,
)
from .base import ConversionMap

if TYPE_CHECKING:
    from typing_extensions import Self


# region ConstantMap


class ConstantMap(ConversionMap[Any]):
    """A ConversionMap that returns a constant value for any input.

    Used to associate a fixed label (e.g., a filename) with every coordinate
    on a timeline. Typically used inside a `timetoalign.CombinationMap` to
    pair a numeric conversion (pixels to seconds) with a string label
    (source image filename).

    The constant value can be of any type, though strings are the most
    common use case. The map has no source or target unit by default.

    Attributes:
        value: The constant value returned for every input.

    Examples:
        >>> cmap = ConstantMap(value="page1.jpeg", name="filename")
        >>> cmap(42.0)
        'page1.jpeg'
        >>> cmap(np.array([1.0, 2.0, 3.0]))
        array(['page1.jpeg', 'page1.jpeg', 'page1.jpeg'], dtype=object)

    See Also:
        timetoalign.CombinationMap
    """

    def __init__(
        self,
        *,
        value: Any,
        source_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a ConstantMap.

        Args:
            value: The constant value to return for every input coordinate.
            source_unit: The unit of input coordinates (optional). Typically
                left as None so that CombinationMap validation does not
                conflict with sibling maps.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.
        """
        super().__init__(
            source_unit=source_unit,
            target_unit=None,
            uid=uid,
            name=name,
        )
        self._value = value

    @property
    def value(self) -> Any:
        """The constant value returned for every input."""
        return self._value

    @property
    def is_invertible(self) -> bool:
        """ConstantMap is not invertible (many-to-one mapping)."""
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> Any:
        """Return the constant value regardless of input."""
        return self._value

    def _convert_array(self, values: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Return an array filled with the constant value.

        Args:
            values: Input array (used only for its shape).

        Returns:
            Object-type array filled with the constant value.
        """
        result = np.empty(values.shape, dtype=object)
        result[:] = self._value
        return result

    def inverse(self) -> Self:
        """ConstantMap cannot be inverted.

        Raises:
            NotImplementedError: Always, as a constant function has no
                well-defined inverse.
        """
        raise NotImplementedError(
            "ConstantMap cannot be inverted: a constant function "
            "is many-to-one and has no unique inverse"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        A ``Fraction`` constant is encoded as the canonical rational dictionary
        dict; every other constant (the common case is a string label)
        is emitted as-is and must therefore already be JSON-native.
        """
        d = super().to_dict()
        d["value"] = (
            rational_to_dict(self._value)
            if isinstance(self._value, Fraction)
            else self._value
        )
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstantMap:
        """Deserialize from dictionary.

        Args:
            data: Dictionary representation with at least a 'value' key.

        Returns:
            A ConstantMap instance.
        """
        raw = data["value"]
        return cls(
            value=rational_from_dict(raw) if is_rational_dict(raw) else raw,
            source_unit=data.get("source_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def __repr__(self) -> str:
        parts = [f"value={self._value!r}"]
        if self._source_unit:
            parts.append(f"source_unit={self._source_unit}")
        if self._name and self._name != self._id:
            parts.append(f"name={self._name!r}")
        return f"ConstantMap({', '.join(parts)})"


# endregion
