"""Tests for CombinationMap (multi-output maps).

CombinationMap yields multiple outputs from multiple C-Maps applied to the
same input.
"""

import numpy as np
import pytest

from timetoalign.core.enums import TimeUnit
from timetoalign.maps import LinearMap
from timetoalign.maps.combination import CombinationMap
from timetoalign.maps.periodic import FloorMap, RotationMap


class TestCombinationMap:
    """Tests for CombinationMap."""

    def test_initialization_from_dict(self):
        """Test initialization from a dictionary of maps."""
        m1 = LinearMap(scalar=2.0, offset=0.0)
        m2 = LinearMap(scalar=0.5, offset=1.0)

        combo = CombinationMap(maps={"double": m1, "half_plus_one": m2})

        assert "double" in combo.names
        assert "half_plus_one" in combo.names
        assert len(combo.maps) == 2

    def test_initialization_from_sequence(self):
        """Test initialization from sequence of (name, map) tuples."""
        m1 = LinearMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5)

        combo = CombinationMap(maps=[("first", m1), ("second", m2)])

        # Order should be preserved
        assert combo.names == ["first", "second"]

    def test_empty_maps_error(self):
        """Cannot create CombinationMap with no sub-maps."""
        with pytest.raises(ValueError, match="at least one"):
            CombinationMap(maps={})

    def test_scalar_conversion(self):
        """Test conversion of a single value."""
        m1 = LinearMap(scalar=2.0, offset=0.0)
        m2 = LinearMap(scalar=0.5, offset=1.0)

        combo = CombinationMap(maps={"double": m1, "half_plus_one": m2})

        result = combo(10.0)
        assert isinstance(result, dict)
        assert result["double"] == 20.0
        assert result["half_plus_one"] == 6.0

    def test_array_conversion(self):
        """Test conversion of an array."""
        m1 = LinearMap(scalar=2.0, offset=0.0)
        m2 = LinearMap(scalar=0.5, offset=1.0)

        combo = CombinationMap(maps={"double": m1, "half_plus_one": m2})

        arr = np.array([1.0, 2.0, 3.0])
        result = combo(arr)

        assert isinstance(result, dict)
        np.testing.assert_array_equal(result["double"], np.array([2.0, 4.0, 6.0]))
        np.testing.assert_array_equal(
            result["half_plus_one"], np.array([1.5, 2.0, 2.5])
        )

    def test_metrical_combination(self):
        """Test combining measure and beat maps (primary use case)."""
        measure_map = FloorMap(divisor=4.0, base=1)
        beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)

        combo = CombinationMap(
            maps={"measure": measure_map, "beat": beat_map},
            source_unit="quarters",
        )

        # Test at various positions
        result = combo(0.0)
        assert result["measure"] == 1
        assert result["beat"] == 1.0

        result = combo(7.5)
        assert result["measure"] == 2
        assert result["beat"] == 4.5

        result = combo(100.0)
        assert result["measure"] == 26
        assert result["beat"] == 1.0

    def test_xy_coordinates(self):
        """Test combining x and y coordinate maps."""
        x_map = LinearMap(scalar=10.0, offset=100.0)
        y_map = LinearMap(scalar=5.0, offset=50.0)

        combo = CombinationMap(maps={"x": x_map, "y": y_map})

        result = combo(1.0)
        assert result["x"] == 110.0
        assert result["y"] == 55.0

    def test_not_invertible(self):
        """CombinationMap cannot be inverted."""
        m1 = LinearMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5)
        combo = CombinationMap(maps={"a": m1, "b": m2})

        with pytest.raises(NotImplementedError, match="cannot be inverted"):
            combo.inverse()

    def test_get_map(self):
        """Test retrieving individual sub-maps."""
        m1 = LinearMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5)
        combo = CombinationMap(maps={"a": m1, "b": m2})

        assert combo.get_map("a") is m1
        assert combo["b"] is m2  # Also via indexing

        with pytest.raises(KeyError):
            combo.get_map("nonexistent")

    def test_source_unit_validation(self):
        """Sub-maps should have compatible source units."""
        m1 = LinearMap(scalar=2.0, source_unit="quarters")
        m2 = LinearMap(scalar=0.5, source_unit="seconds")

        with pytest.raises(ValueError, match="source_unit"):
            CombinationMap(maps={"a": m1, "b": m2})

    def test_source_unit_inheritance(self):
        """CombinationMap inherits source_unit from first sub-map."""
        m1 = LinearMap(scalar=2.0, source_unit="quarters")
        m2 = LinearMap(scalar=0.5)  # No unit specified

        combo = CombinationMap(maps={"a": m1, "b": m2})
        assert combo.source_unit == TimeUnit.quarters

    def test_supra_metrical_positions(self):
        """Test metrical position calculation for SUPRA data.

        SUPRA: 888 quarters, 222 measures in 4/4 time.
        """
        measure_map = FloorMap(divisor=4.0, base=1)
        beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)

        combo = CombinationMap(
            maps={"measure": measure_map, "beat": beat_map},
            source_unit="quarters",
        )

        # Test boundaries
        assert combo(0) == {"measure": 1, "beat": 1.0}
        assert combo(3) == {"measure": 1, "beat": 4.0}
        assert combo(4) == {"measure": 2, "beat": 1.0}
        assert combo(884) == {"measure": 222, "beat": 1.0}
        assert combo(887) == {"measure": 222, "beat": 4.0}

        # Array test
        quarters = np.array([0, 100, 500, 884, 887])
        result = combo(quarters)

        expected_measures = np.array([1, 26, 126, 222, 222])
        expected_beats = np.array([1.0, 1.0, 1.0, 1.0, 4.0])

        np.testing.assert_array_equal(result["measure"], expected_measures)
        # Beat values are exact modular arithmetic results (quarters % 4 + 1)
        np.testing.assert_array_equal(result["beat"], expected_beats)
