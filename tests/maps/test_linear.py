"""Tests for linear conversion maps."""

import re
from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.time import Coordinate
from timetoalign.maps.linear import LinearMap, ScalarMap, ShiftMap

# A seconds axis drawn in points: the positions of its five labelled ticks
# (0, 10, 20, 30, 40 s), as a program computed and printed them.
TICKS = [
    76.70371459980326,
    290.5859487023929,
    504.4681828049824,
    718.350416907572,
    932.2326510101616,
]


class TestLinearMap:
    def test_initialization(self):
        m = LinearMap(scalar=2.0, offset=10.0)
        assert m.scalar == 2.0
        assert m.offset == 10.0
        assert not m.is_identity

        m_ident = LinearMap(scalar=1.0, offset=0.0)
        assert m_ident.is_identity

        with pytest.raises(ValueError, match="not invertible"):
            LinearMap(scalar=0.0)

    def test_conversion_scalar(self):
        m = LinearMap(scalar=2.0, offset=10.0)
        # y = 2x + 10
        assert m(0) == 10.0
        assert m(5) == 20.0
        assert m(-5) == 0.0

        # With coordinate
        c = Coordinate(5, TimeUnit.ticks)
        # Should return raw value
        assert m(c) == 20.0

    def test_conversion_array(self):
        m = LinearMap(scalar=2.0, offset=10.0)
        arr = np.array([0, 5, -5])
        expected = np.array([10.0, 20.0, 0.0])
        np.testing.assert_array_equal(m(arr), expected)

    def test_inverse(self):
        m = LinearMap(scalar=2.0, offset=10.0)
        inv = m.inverse()

        # Original: y = 2x + 10
        # Inverse: x = (y - 10) / 2 = 0.5y - 5
        assert isinstance(inv, LinearMap)
        assert inv.scalar == 0.5
        assert inv.offset == -5.0

        # Round trip
        val = 42.0
        assert inv(m(val)) == val

    def test_composition(self):
        m1 = LinearMap(scalar=2.0, offset=10.0)
        m2 = LinearMap(scalar=0.5, offset=5.0)

        # m1: y = 2x + 10
        # m2: z = 0.5y + 5 = 0.5(2x + 10) + 5 = x + 5 + 5 = x + 10

        comp = m1.compose_with(m2)
        assert isinstance(comp, LinearMap)
        assert comp.scalar == 1.0
        assert comp.offset == 10.0

        assert comp(5) == 15.0  # 5 -> 20 -> 15

    def test_fraction_support(self):
        m = LinearMap(scalar=Fraction(1, 3), offset=Fraction(2, 3))
        res = m(Fraction(1, 1))
        # 1/3 * 1 + 2/3 = 1
        assert res == Fraction(1, 1)


class TestLinearMapFromPairs:
    def test_exact_pairs_give_an_exact_line(self):
        m = LinearMap.from_pairs(
            [(Fraction(1, 2), 1), (Fraction(7, 2), 2), (Fraction(13, 2), 3)],
            source_unit="points",
            value_unit="quarters",
        )
        # (3 - 1) / (13/2 - 1/2) = 1/3; 1 - 1/3 * 1/2 = 5/6
        assert m.scalar == Fraction(1, 3)
        assert type(m.scalar) is Fraction
        assert m.offset == Fraction(5, 6)
        assert m.source_unit == TimeUnit.points
        assert m.target_unit == TimeUnit.quarters

    def test_float_pairs_give_the_two_point_line(self):
        m = LinearMap.from_pairs(
            [(0.0, 0.0), (213.88223410258962, 10.0), (855.5289364103584, 40.0)],
            source_unit="points",
            value_unit="seconds",
        )
        assert m.scalar == 0.04675470144567245
        assert m.offset == 0.0

    def test_order_is_irrelevant(self):
        pairs = [(2, 5), (0, 1), (1, 3)]
        assert LinearMap.from_pairs(
            pairs, source_unit=None, value_unit="quarters"
        ) == LinearMap.from_pairs(
            sorted(pairs), source_unit=None, value_unit="quarters"
        )

    def test_coordinate_values_set_the_target_unit(self):
        m = LinearMap.from_pairs(
            [
                (Coordinate(0, TimeUnit.ticks), Coordinate(0, TimeUnit.quarters)),
                (Coordinate(960, TimeUnit.ticks), Coordinate(2, TimeUnit.quarters)),
            ],
            source_unit="ticks",
        )
        assert m.scalar == Fraction(1, 480)
        assert m.target_unit == TimeUnit.quarters

    def test_numbers_are_read_in_value_unit(self):
        m = LinearMap.from_pairs(
            [(0, 0), (960, Coordinate(2, TimeUnit.quarters))],
            source_unit="ticks",
            value_unit="quarters",
        )
        assert m.scalar == Fraction(1, 480)
        assert m.target_unit == TimeUnit.quarters

    def test_a_number_needs_value_unit(self):
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Calibration pair 0 has a value of type int, not a Coordinate "
                "carrying its unit; give value_unit to read numbers."
            ),
        ):
            LinearMap.from_pairs(
                [(0, 0), (960, Coordinate(2, TimeUnit.quarters))], source_unit="ticks"
            )

    def test_values_must_share_their_unit(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 has a value in seconds, not in quarters."
            ),
        ):
            LinearMap.from_pairs(
                [(0, 0), (960, Coordinate(2, TimeUnit.seconds))],
                source_unit="ticks",
                value_unit="quarters",
            )
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 has a value in seconds, not in quarters."
            ),
        ):
            LinearMap.from_pairs(
                [
                    (0, Coordinate(0, TimeUnit.quarters)),
                    (960, Coordinate(2, TimeUnit.seconds)),
                ],
                source_unit="ticks",
            )

    @pytest.mark.parametrize("position", [1.0000000000000002, 1.0, 0.9999999999999999])
    def test_float_bound_is_one_ulp_of_the_position(self, position):
        # The exact line through (0, 0) and (2, 2) reaches 1 at 1.
        m = LinearMap.from_pairs(
            [(0.0, 0.0), (position, 1.0), (2.0, 2.0)],
            source_unit=None,
            value_unit="seconds",
        )
        assert m.scalar == 1.0

    @pytest.mark.parametrize("position", [1.0000000000000004, 0.9999999999999998])
    def test_two_ulps_of_the_position_are_refused(self, position):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Calibration pair 1 ({position}, 1.0) lies off the line through "
                "the first and last pair, which reaches 1.0 at 1.0."
            ),
        ):
            LinearMap.from_pairs(
                [(0.0, 0.0), (position, 1.0), (2.0, 2.0)],
                source_unit=None,
                value_unit="seconds",
            )

    def test_an_inner_value_of_zero_takes_the_position_bound(self):
        # The line through (0, -1) and (2, 1) reaches 0 at 1, and the
        # position one ulp above 1 is within the bound. Measured on the
        # value, the bound would be one ulp of 0.0 (5e-324), which forces an
        # exact hit on a rounded position.
        m = LinearMap.from_pairs(
            [(0.0, -1.0), (1.0000000000000002, 0.0), (2.0, 1.0)],
            source_unit=None,
            value_unit="seconds",
        )
        assert m.scalar == 1.0
        assert m.offset == -1.0

    def test_drawn_axis_ticks_within_the_bound(self):
        # A seconds axis drawn in points with five labelled ticks, positions
        # as a program printed them, read from the first tick.
        m = LinearMap.from_pairs(
            list(zip(TICKS, [0, 10, 20, 30, 40])),
            source_unit="points",
            value_unit="seconds",
            origin=TICKS[0],
        )
        # 40 / (932.2326510101616 - 76.70371459980326), and 0 - scalar * 0.0
        assert m.scalar == 0.04675470144567245
        assert m.offset == 0.0
        assert m.output_number_type is NumberType.float

    def test_drawn_axis_tick_two_floats_up_is_refused(self):
        # The second tick moved up by two floats lies 2.94 of its ulps above
        # the exact position at which the line reaches 10.
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (290.585948702393, 10) lies off the line through "
                "the first and last pair, which reaches 10 at 290.58594870239284."
            ),
        ):
            LinearMap.from_pairs(
                list(
                    zip([TICKS[0], 290.585948702393, *TICKS[2:]], [0, 10, 20, 30, 40])
                ),
                source_unit="points",
                value_unit="seconds",
                origin=TICKS[0],
            )

    def test_origin_is_subtracted_after_the_check(self):
        from_child = LinearMap.from_pairs(
            [(0.0, 0.0), (40.0, 2.0), (80.0, 4.0)],
            source_unit="points",
            value_unit="seconds",
        )
        from_parent = LinearMap.from_pairs(
            [(10.0, 0.0), (50.0, 2.0), (90.0, 4.0)],
            source_unit="points",
            value_unit="seconds",
            origin=Coordinate(10.0, TimeUnit.points),
        )
        assert (from_parent.scalar, from_parent.offset) == (0.05, 0.0)
        assert (from_child.scalar, from_child.offset) == (0.05, 0.0)

    def test_origin_in_another_unit_is_refused(self):
        with pytest.raises(
            ValueError,
            match=re.escape("The calibration origin is in pixels, not in points."),
        ):
            LinearMap.from_pairs(
                [(10.0, 0.0), (90.0, 4.0)],
                source_unit="points",
                value_unit="seconds",
                origin=Coordinate(10, TimeUnit.pixels),
            )

    def test_exact_inner_pair_must_lie_on_the_line(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 2 (1, 1000001/1000000) lies off the line through "
                "the first and last pair, which reaches 1000001/1000000 at "
                "1000001/1000000."
            ),
        ):
            LinearMap.from_pairs(
                [(0, 0), (2, 2), (1, Fraction(1000001, 1000000))],
                source_unit=None,
                value_unit="quarters",
            )

    def test_one_float_gives_the_whole_calibration_the_position_bound(self):
        # The same value, once as a float and once as its exact Fraction:
        # with the float, the calibration is not exact and the position 1 may
        # miss the line's 1 + 2**-52 by one ulp of 1; with the Fraction,
        # every number is exact and the pair must lie on the line.
        m = LinearMap.from_pairs(
            [(0, 0), (1, 1.0000000000000002), (2, 2)],
            source_unit=None,
            value_unit="seconds",
        )
        assert m.scalar == 1
        assert m.output_number_type is NumberType.float
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (1, 4503599627370497/4503599627370496) lies off "
                "the line through the first and last pair, which reaches "
                "4503599627370497/4503599627370496 at "
                "4503599627370497/4503599627370496."
            ),
        ):
            LinearMap.from_pairs(
                [(0, 0), (1, Fraction(1.0000000000000002)), (2, 2)],
                source_unit=None,
                value_unit="seconds",
            )

    def test_exact_positions_with_float_values(self):
        # The line through (0, 0.1) and (2, 0.3) reaches the double 0.2 at
        # 1 + 1.39e-16 (exactly on the three doubles): 0.625 ulp of 1.
        m = LinearMap.from_pairs(
            [(0, 0.1), (1, 0.2), (2, 0.3)], source_unit=None, value_unit="seconds"
        )
        # (0.3 - 0.1) / 2 in floats
        assert m.scalar == 0.09999999999999999
        assert m.offset == 0.1

    def test_declared_output_number_type(self):
        floats = LinearMap.from_pairs(
            [(0.0, 0.0), (2.0, 1.0)], source_unit="points", value_unit="quarters"
        )
        assert floats.output_number_type is NumberType.float
        assert type(floats(1.0)) is float
        assert floats.inverse().output_number_type is NumberType.float
        exact = LinearMap.from_pairs(
            [(0, 0), (2, 1)], source_unit="points", value_unit="seconds"
        )
        assert exact.output_number_type is NumberType.fraction
        assert exact(1) == Fraction(1, 2)
        assert type(exact(1)) is Fraction
        # Ticks admit integers only: the map writes whole ticks.
        ticks = LinearMap.from_pairs(
            [(0, 0), (2, 960)], source_unit="points", value_unit="ticks"
        )
        assert ticks.output_number_type is NumberType.int
        assert ticks(1) == 480
        assert type(ticks(1)) is int

    def test_refusals(self):
        with pytest.raises(
            ValueError,
            match=re.escape("A calibration needs at least two pairs, got 0."),
        ):
            LinearMap.from_pairs([], source_unit=None, value_unit="quarters")
        with pytest.raises(
            ValueError,
            match=re.escape("Calibration pairs 0 and 2 share the position 1."),
        ):
            LinearMap.from_pairs(
                [(1, 0), (2, 1), (1, 2)], source_unit=None, value_unit="quarters"
            )

    def test_end_pairs_reading_one_value_are_refused(self):
        with pytest.raises(
            ValueError, match=re.escape("LinearMap scalar cannot be zero")
        ):
            LinearMap.from_pairs(
                [(0, 1), (1, 1), (2, 1)], source_unit=None, value_unit="quarters"
            )

    def test_other_maps_are_no_law(self):
        with pytest.raises(
            NotImplementedError,
            match=re.escape("ShiftMap is built from no calibration pairs"),
        ):
            ShiftMap.from_pairs(
                [(0, 0), (1, 1)], source_unit=None, value_unit="quarters"
            )


class TestScalarMap:
    def test_basic(self):
        m = ScalarMap(scalar=10.0)
        assert m.scalar == 10.0
        assert m(5) == 50.0

        inv = m.inverse()
        assert inv.scalar == 0.1
        assert inv(50) == 5.0

    def test_units(self):
        m = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        assert m.source_unit == TimeUnit.seconds
        assert m.target_unit == TimeUnit.milliseconds

        c = Coordinate(1.5, TimeUnit.seconds)
        assert m(c) == 1500.0

        # Wrong unit
        c_wrong = Coordinate(1.5, TimeUnit.ticks)
        with pytest.raises(ValueError, match="does not match"):
            m(c_wrong)


class TestShiftMap:
    def test_basic(self):
        m = ShiftMap(offset=100)
        assert m.offset == 100
        assert m(50) == 150

        inv = m.inverse()
        assert inv.offset == -100
        assert inv(150) == 50

    def test_identity(self):
        m = ShiftMap(offset=0)
        assert m.is_identity


# region Public convert_array() API Tests


class TestConvertArrayPublicAPI:
    """Tests for the public convert_array() method on all linear map types.

    These tests verify the integration pattern used by the timestamp system.
    """

    def test_linear_map_convert_array_returns_ndarray(self):
        """convert_array() returns numpy array."""
        m = LinearMap(scalar=2.0, offset=1.0)
        values = np.array([0.0, 1.0, 2.0, 3.0])

        result = m.convert_array(values)

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape

    def test_linear_map_convert_array_values(self):
        """convert_array() computes correct values."""
        m = LinearMap(scalar=2.0, offset=1.0)
        values = np.array([0.0, 1.0, 2.0, 3.0])

        result = m.convert_array(values)

        np.testing.assert_array_equal(result, [1.0, 3.0, 5.0, 7.0])

    def test_scalar_map_convert_array(self):
        """ScalarMap.convert_array() works correctly."""
        m = ScalarMap(scalar=10.0)
        values = np.array([1.0, 2.0, 3.0])

        result = m.convert_array(values)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_shift_map_convert_array(self):
        """ShiftMap.convert_array() works correctly."""
        m = ShiftMap(offset=5.0)
        values = np.array([0.0, 10.0, 20.0])

        result = m.convert_array(values)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [5.0, 15.0, 25.0])

    def test_convert_array_empty_input(self):
        """convert_array() handles empty arrays."""
        m = LinearMap(scalar=2.0, offset=1.0)
        values = np.array([])

        result = m.convert_array(values)

        assert isinstance(result, np.ndarray)
        assert len(result) == 0

    def test_convert_array_single_element(self):
        """convert_array() handles single-element arrays."""
        m = LinearMap(scalar=2.0, offset=1.0)
        values = np.array([5.0])

        result = m.convert_array(values)

        assert len(result) == 1
        assert result[0] == 11.0

    def test_convert_array_large_array(self):
        """convert_array() handles large arrays efficiently."""
        m = LinearMap(scalar=2.0, offset=1.0)
        values = np.arange(100000, dtype=np.float64)

        result = m.convert_array(values)

        assert len(result) == 100000
        # Spot check some values
        assert result[0] == 1.0
        assert result[50000] == 100001.0

    def test_convert_array_matches_scalar_conversion(self):
        """convert_array() results match element-wise scalar conversion."""
        m = LinearMap(scalar=1.5, offset=-3.0)
        values = np.array([0.0, 1.0, -1.0, 100.0, -100.0])

        array_result = m.convert_array(values)
        scalar_results = np.array([m(v) for v in values])

        # Exact same linear arithmetic (f(x) = 1.5x - 3.0), no floating-point
        # divergence between array and scalar paths
        np.testing.assert_array_equal(array_result, scalar_results)


# endregion
