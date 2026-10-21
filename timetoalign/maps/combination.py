"""Combination map for multi-output conversions.

This module provides CombinationMap, which yields multiple outputs from
multiple C-Maps applied to the same input.

A CombinationMap yields outputs from multiple C-maps at once, such as
(x, y) coordinate pairs and resulting two-field matrices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Mapping, Sequence

from numpy.typing import NDArray

from ..core.enums import TimeUnit
from ..core.time import CoordinateValue
from .base import ConversionMap, MeasureMapIds

if TYPE_CHECKING:
    from typing_extensions import Self


class CombinationMap(ConversionMap[dict[str, Any]]):
    """Map that combines multiple C-Maps to yield dict outputs.

    A CombinationMap applies multiple C-Maps to the same input coordinate
    and returns a dictionary with named outputs. This is useful for:
    - (x, y) coordinate pairs for graphical mapping
    - (measure_number, beat_in_measure) for metrical grids
    - (filename, pixel_position) for multi-image graphical timelines

    The map is invertible only if ALL sub-maps are invertible, and even
    then, inversion is only meaningful for bijective sub-maps.

    The format's ``CombinationMap`` type: ``maps`` keyed by output name.
    The library keeps the order the outputs were given in for display; the
    format's outputs carry no order, so a map read back lists them in the
    manifest's key order.

    Attributes:
        maps: Dictionary of name -> ConversionMap.
        names: List of output names in order.

    Examples:
        >>> from timetoalign.maps import FloorMap, RotationMap
        >>>
        >>> # (measure, beat) from quarters
        >>> measure_map = FloorMap(divisor=4.0, base=1)
        >>> beat_map = RotationMap(period=4.0, base=1.0)
        >>> combo = CombinationMap(
        ...     maps={"measure": measure_map, "beat": beat_map},
        ... )
        >>> combo(7.5)
        {'measure': 2, 'beat': 4.5}
        >>>
        >>> # Array conversion returns dict of arrays
        >>> import numpy as np
        >>> result = combo(np.array([0.0, 4.0, 7.5]))
        >>> result["measure"]
        array([1, 2, 2])
        >>> result["beat"]
        array([1. , 1. , 4.5])

        >>> # (x, y) coordinates
        >>> from timetoalign.maps import LinearMap
        >>> x_map = LinearMap(scalar=10.0, offset=100.0)
        >>> y_map = LinearMap(scalar=5.0, offset=50.0)
        >>> coord_map = CombinationMap(maps={"x": x_map, "y": y_map})
        >>> coord_map(1.0)
        {'x': 110.0, 'y': 55.0}
    """

    _manifest_type: ClassVar[str | None] = "CombinationMap"

    def __init__(
        self,
        maps: (
            Mapping[str, ConversionMap[Any]] | Sequence[tuple[str, ConversionMap[Any]]]
        ),
        *,
        source_unit: TimeUnit | str | None = None,
        uid: str | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize a CombinationMap.

        Args:
            maps: Either a dict of name->map, or a sequence of (name, map) tuples.
                  The latter preserves insertion order explicitly.
            source_unit: The unit of input coordinates. If not specified,
                        uses the source_unit of the first sub-map.
            uid: Optional explicit ID.
            name: Human-readable name for this map. Defaults to the map's ID.

        Raises:
            ValueError: If maps is empty.
            ValueError: If sub-maps have conflicting source units.
        """
        if isinstance(maps, Mapping):
            self._maps: dict[str, ConversionMap[Any]] = dict(maps)
            self._names: list[str] = list(maps.keys())
        else:
            self._maps = {name: m for name, m in maps}
            self._names = [name for name, _ in maps]

        if not self._maps:
            raise ValueError("CombinationMap requires at least one sub-map")

        # Determine source unit
        if source_unit is not None:
            resolved_source_unit = source_unit
        else:
            # Use first sub-map's source unit
            first_map = next(iter(self._maps.values()))
            resolved_source_unit = first_map.source_unit

        # Validate source units are compatible
        for sub_name, m in self._maps.items():
            if (
                m.source_unit is not None
                and resolved_source_unit is not None
                and m.source_unit != TimeUnit(resolved_source_unit)
            ):
                raise ValueError(
                    f"Sub-map '{sub_name}' has source_unit {m.source_unit}, "
                    f"expected {resolved_source_unit}"
                )

        # CombinationMap has no single target_unit (multiple outputs)
        super().__init__(
            source_unit=resolved_source_unit,
            target_unit=None,
            uid=uid,
            name=name,
        )
        self._admit_ids()

    @property
    def maps(self) -> dict[str, ConversionMap[Any]]:
        """The sub-maps by name."""
        return dict(self._maps)

    @property
    def names(self) -> list[str]:
        """The output names in order."""
        return list(self._names)

    @property
    def is_invertible(self) -> bool:
        """CombinationMap is generally not invertible.

        Even if all sub-maps are invertible, the combination as a whole
        cannot be inverted to produce a single input from multiple outputs.
        """
        return False

    def _convert_scalar(self, value: CoordinateValue, **kwargs: Any) -> dict[str, Any]:
        """Apply all sub-maps to a single value."""
        return {
            name: self._maps[name]._convert_scalar(value, **kwargs)
            for name in self._names
        }

    def _convert_array(
        self, values: NDArray[Any], **kwargs: Any
    ) -> dict[str, NDArray[Any]]:
        """Apply all sub-maps to an array.

        Returns a dict of arrays, one per output name.
        """
        return {
            name: self._maps[name]._convert_array(values, **kwargs)
            for name in self._names
        }

    def inverse(self) -> Self:
        """CombinationMap cannot be inverted.

        Raises:
            NotImplementedError: Always, as combining outputs cannot be
                uniquely reversed to a single input.
        """
        raise NotImplementedError(
            "CombinationMap cannot be inverted: multiple outputs cannot "
            "be uniquely reversed to a single input"
        )

    def get_map(self, name: str) -> ConversionMap[Any]:
        """Get a sub-map by name.

        Args:
            name: The output name.

        Returns:
            The corresponding sub-map.

        Raises:
            KeyError: If name is not found.
        """
        return self._maps[name]

    def _to_manifest(self, *, minimal: bool, entries: MeasureMapIds) -> dict[str, Any]:
        """Write the ``CombinationMap``: each output's map under its name."""
        return self._map_object(
            "CombinationMap",
            {
                "maps": {
                    name: self._maps[name]._to_manifest(
                        minimal=minimal, entries=entries
                    )
                    for name in self._names
                }
            },
            minimal=minimal,
        )

    @classmethod
    def _from_manifest(
        cls, data: dict[str, Any], measure_maps: Mapping[str, Any]
    ) -> CombinationMap:
        """Read a ``CombinationMap``, its outputs in the manifest's key order.

        Raises:
            ValueError: If ``maps`` is not an object.
        """
        maps_data = data.get("maps")
        if not isinstance(maps_data, dict):
            raise ValueError("a CombinationMap's 'maps' must be an object")
        return cls(
            maps=[
                (name, ConversionMap.from_dict(item, measure_maps=measure_maps))
                for name, item in maps_data.items()
            ],
            source_unit=data.get("source_unit"),
            uid=data.get("id"),
            name=data.get("name"),
        )

    def _nested_maps(self) -> list[ConversionMap[Any]]:
        return [self._maps[name] for name in self._names]

    def __repr__(self) -> str:
        return f"CombinationMap(outputs={self._names})"

    def __getitem__(self, name: str) -> ConversionMap[Any]:
        """Get a sub-map by name using indexing syntax."""
        return self._maps[name]
