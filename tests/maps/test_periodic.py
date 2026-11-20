"""Tests for periodic and floor-based conversion maps.

This module tests RotationMap and FloorMap which are essential building blocks
for metrical grids (beat rotation, measure numbering).
"""

from __future__ import annotations

import numpy as np
import pytest

from timetoalign.maps.periodic import FloorMap, RotationMap


class TestRotationMap:
    """Tests for RotationMap (periodic/cyclic patterns)."""

    def test_initialization(self):
        """Test basic initialization and properties."""
        rot = RotationMap(period=4.0, scale=1.0, base=1.0, offset=0.0)
        assert rot.period == 4.0
        assert rot.scale == 1.0
        assert rot.base == 1.0
        assert rot.offset == 0.0
        assert not rot.is_invertible  # Many-to-one

    def test_invalid_period(self):
        """Period must be positive."""
        with pytest.raises(ValueError, match="positive"):
            RotationMap(period=0.0)
        with pytest.raises(ValueError, match="positive"):
            RotationMap(period=-1.0)

    def test_beat_rotation_4_4(self):
        """Test beat rotation in 4/4 time (1, 2, 3, 4, 1, 2, 3, 4...)."""
        # For 4/4 with quarter-note beats:
        # period = 4 quarters per measure
        # scale = 1 (1 quarter = 1 beat)
        # base = 1 (1-indexed beats)
        beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)

        # Within first measure
        assert beat_map(0.0) == 1.0  # Quarter 0 -> Beat 1
        assert beat_map(1.0) == 2.0  # Quarter 1 -> Beat 2
        assert beat_map(2.0) == 3.0  # Quarter 2 -> Beat 3
        assert beat_map(3.0) == 4.0  # Quarter 3 -> Beat 4

        # Wraps to next measure
        assert beat_map(4.0) == 1.0  # Quarter 4 -> Beat 1 (wraps!)
        assert beat_map(5.0) == 2.0  # Quarter 5 -> Beat 2

        # Fractional beats
        assert beat_map(0.5) == 1.5  # Quarter 0.5 -> Beat 1.5
        assert beat_map(4.5) == 1.5  # Quarter 4.5 -> Beat 1.5

    def test_beat_rotation_3_4(self):
        """Test beat rotation in 3/4 time."""
        beat_map = RotationMap(period=3.0, scale=1.0, base=1.0)

        assert beat_map(0.0) == 1.0
        assert beat_map(1.0) == 2.0
        assert beat_map(2.0) == 3.0
        assert beat_map(3.0) == 1.0  # Wraps
        assert beat_map(6.0) == 1.0  # Wraps twice

    def test_beat_rotation_6_8(self):
        """Test beat rotation in 6/8 time (eighth-note beats).

        In 6/8: 6 beats per measure, each beat = 1/2 quarter.
        So period = 3 quarters (6 * 0.5), scale = 2 (1 quarter = 2 beats)
        """
        beat_map = RotationMap(period=3.0, scale=2.0, base=1.0)

        assert beat_map(0.0) == 1.0  # Quarter 0 -> Beat 1
        assert beat_map(0.5) == 2.0  # Quarter 0.5 -> Beat 2
        assert beat_map(1.0) == 3.0  # Quarter 1 -> Beat 3
        assert beat_map(1.5) == 4.0  # Quarter 1.5 -> Beat 4
        assert beat_map(2.0) == 5.0  # Quarter 2 -> Beat 5
        assert beat_map(2.5) == 6.0  # Quarter 2.5 -> Beat 6
        assert beat_map(3.0) == 1.0  # Quarter 3 -> Beat 1 (wraps!)

    def test_offset(self):
        """Test input offset (shifts the cycle start)."""
        # Offset of 1 means the cycle starts at input 1
        rot = RotationMap(period=4.0, scale=1.0, base=0.0, offset=1.0)

        assert rot(1.0) == 0.0  # (1 - 1) % 4 = 0
        assert rot(2.0) == 1.0  # (2 - 1) % 4 = 1
        assert rot(5.0) == 0.0  # (5 - 1) % 4 = 0

    def test_array_conversion(self):
        """Test vectorized array conversion."""
        beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)
        arr = np.array([0.0, 1.0, 4.0, 4.5, 7.5])
        expected = np.array([1.0, 2.0, 1.0, 1.5, 4.5])
        np.testing.assert_array_almost_equal(beat_map(arr), expected)

    def test_not_invertible(self):
        """RotationMap cannot be inverted."""
        rot = RotationMap(period=4.0)
        with pytest.raises(NotImplementedError, match="not invertible"):
            rot.inverse()

    def test_angle_normalization(self):
        """Test angle normalization (0-360) as a general use case."""
        angle_map = RotationMap(period=360.0)

        assert angle_map(0.0) == 0.0
        assert angle_map(90.0) == 90.0
        assert angle_map(360.0) == 0.0
        assert angle_map(450.0) == 90.0
        assert angle_map(-90.0) == 270.0


class TestFloorMap:
    """Tests for FloorMap (integer floor division)."""

    def test_initialization(self):
        """Test basic initialization and properties."""
        fm = FloorMap(divisor=4.0, base=1, offset=0.0)
        assert fm.divisor == 4.0
        assert fm.base == 1
        assert fm.offset == 0.0
        assert not fm.is_invertible  # Many-to-one

    def test_invalid_divisor(self):
        """Divisor must be positive."""
        with pytest.raises(ValueError, match="positive"):
            FloorMap(divisor=0.0)
        with pytest.raises(ValueError, match="positive"):
            FloorMap(divisor=-4.0)

    def test_measure_numbers_4_4(self):
        """Test measure numbering in 4/4 time (4 quarters per measure)."""
        measure_map = FloorMap(divisor=4.0, base=1)

        # Measure 1: quarters 0-3.99
        assert measure_map(0.0) == 1
        assert measure_map(1.0) == 1
        assert measure_map(3.0) == 1
        assert measure_map(3.99) == 1

        # Measure 2: quarters 4-7.99
        assert measure_map(4.0) == 2
        assert measure_map(7.5) == 2

        # Measure 3: quarters 8-11.99
        assert measure_map(8.0) == 3

        # Large values
        assert measure_map(100.0) == 26  # floor(100/4) + 1 = 26

    def test_measure_numbers_3_4(self):
        """Test measure numbering in 3/4 time (3 quarters per measure)."""
        measure_map = FloorMap(divisor=3.0, base=1)

        assert measure_map(0.0) == 1
        assert measure_map(2.99) == 1
        assert measure_map(3.0) == 2
        assert measure_map(6.0) == 3

    def test_zero_indexed(self):
        """Test 0-indexed output with base=0."""
        measure_map = FloorMap(divisor=4.0, base=0)

        assert measure_map(0.0) == 0
        assert measure_map(4.0) == 1
        assert measure_map(8.0) == 2

    def test_offset(self):
        """Test input offset (for pickup measures, etc.)."""
        # If the first measure starts at quarter 2 (pickup of 2 quarters)
        measure_map = FloorMap(divisor=4.0, base=1, offset=2.0)

        # At quarter 2, we're at the start of measure 1
        assert measure_map(2.0) == 1
        assert measure_map(5.99) == 1
        assert measure_map(6.0) == 2

    def test_array_conversion(self):
        """Test vectorized array conversion."""
        measure_map = FloorMap(divisor=4.0, base=1)
        arr = np.array([0.0, 4.0, 7.5, 12.0, 100.0])
        expected = np.array([1, 2, 2, 4, 26])
        np.testing.assert_array_equal(measure_map(arr), expected)

    def test_not_invertible(self):
        """FloorMap cannot be inverted."""
        fm = FloorMap(divisor=4.0)
        with pytest.raises(NotImplementedError, match="not invertible"):
            fm.inverse()

    def test_page_numbers(self):
        """Test page numbering as a general use case."""
        # 1000 pixels per page, 1-indexed
        page_map = FloorMap(divisor=1000, base=1)

        assert page_map(0) == 1
        assert page_map(999) == 1
        assert page_map(1000) == 2
        assert page_map(2500) == 3


class TestPeriodicMapIntegration:
    """Integration tests combining RotationMap and FloorMap."""

    def test_measure_and_beat_consistency(self):
        """Test that measure and beat maps are consistent with each other."""
        # 4/4 time: 4 quarters per measure
        measure_map = FloorMap(divisor=4.0, base=1)
        beat_map = RotationMap(period=4.0, scale=1.0, base=1.0)

        # Test at various positions
        test_quarters = [0.0, 0.5, 1.0, 3.99, 4.0, 7.5, 100.0, 887.5]

        for q in test_quarters:
            measure = measure_map(q)
            beat = beat_map(q)

            # Verify: measure should increment every 4 quarters
            expected_measure = int(q // 4) + 1
            assert measure == expected_measure, f"At quarter {q}"

            # Verify: beat should equal (q % 4) + 1 exactly
            # (modular arithmetic on exact floating-point values produces exact results)
            beat_in_measure = (q % 4.0) + 1.0
            assert beat == beat_in_measure, f"At quarter {q}, beat={beat}"

    def test_supra_compatibility(self):
        """Test that maps work correctly for SUPRA data dimensions.

        SUPRA reference values:
        - 888 quarter notes total
        - 222 measures in 4/4 time
        - Measures numbered 1-222
        """
        TOTAL_QUARTERS = 888
        TOTAL_MEASURES = 222
        QUARTERS_PER_MEASURE = 4

        measure_map = FloorMap(
            divisor=QUARTERS_PER_MEASURE,
            base=1,
            source_unit="quarters",
            target_unit="number",
        )
        beat_map = RotationMap(
            period=QUARTERS_PER_MEASURE,
            scale=1.0,
            base=1.0,
            source_unit="quarters",
            target_unit="number",
        )

        # First quarter -> measure 1, beat 1
        assert measure_map(0) == 1
        assert beat_map(0) == 1.0

        # Last quarter of first measure
        assert measure_map(3) == 1
        assert beat_map(3) == 4.0

        # First quarter of second measure
        assert measure_map(4) == 2
        assert beat_map(4) == 1.0

        # Last complete measure boundary (measure 222 starts at quarter 884)
        assert measure_map(884) == 222
        assert beat_map(884) == 1.0

        # End of piece (quarter 887 = last quarter of measure 222)
        assert measure_map(887) == 222
        assert beat_map(887) == 4.0

        # Verify total measure count
        # Measures 1-222 span quarters 0-887 (888 quarters total)
        last_measure = measure_map(TOTAL_QUARTERS - 1)
        assert last_measure == TOTAL_MEASURES

        # Array test for all measure boundaries
        measure_starts = np.arange(0, TOTAL_QUARTERS, QUARTERS_PER_MEASURE)
        measures = measure_map(measure_starts)
        expected_measures = np.arange(1, TOTAL_MEASURES + 1)
        np.testing.assert_array_equal(measures, expected_measures)
