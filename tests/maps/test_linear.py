"""Tests for linear conversion maps."""

import math
import random
import re
from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.core.time import Coordinate
from timetoalign.maps.base import OutputNumberTypeError
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

# Seeded random axes: a start and a width each a uniform number in [1, 10)
# times a decade, 3 to 13 ticks, integer labels from a random first label by
# a random step.
SEED = 20261118
DECADES = [0.1, 1, 10, 100, 1000]


def random_axes(n: int) -> list[tuple[float, float, list[int]]]:
    rng = random.Random(SEED)
    axes = []
    for _ in range(n):
        x0 = rng.uniform(1, 10) * rng.choice(DECADES) * rng.choice([1, -1])
        w = rng.uniform(1, 10) * rng.choice(DECADES)
        first, step = rng.randint(0, 100), rng.randint(1, 20)
        axes.append((x0, w, [first + j * step for j in range(rng.randint(3, 13))]))
    return axes


AXES = random_axes(2000)

# Each axis placed four ways: every exact position rounded once, and computed
# in doubles with the multiplication and the division in each order.
PLACEMENTS = {
    "rounded once": lambda x0, w, d, D: float(Fraction(x0) + Fraction(w) * d / D),
    "(w*d)/D": lambda x0, w, d, D: x0 + (w * d) / D,
    "w*(d/D)": lambda x0, w, d, D: x0 + w * (d / D),
    "(w/D)*d": lambda x0, w, d, D: x0 + (w / D) * d,
}


def placed(placement: str) -> list[list[tuple[float, int]]]:
    place = PLACEMENTS[placement]
    calibrations = []
    for x0, w, labels in AXES:
        span = labels[-1] - labels[0]
        positions = [place(x0, w, v - labels[0], span) for v in labels]
        calibrations.append(list(zip(positions, labels)))
    return calibrations


def beyond_rounding_alone(pairs: list[tuple[float, int]]) -> bool:
    """Whether an inner position misses the line by more than the rounding of
    itself and of the end positions, in exact arithmetic."""
    (p0, v0), (pn, vn) = pairs[0], pairs[-1]

    def half_ulp(x: float) -> Fraction:
        return Fraction(math.ulp(x)) / 2

    for p, v in pairs[1:-1]:
        t = Fraction(v - v0, vn - v0)
        miss = abs(Fraction(p) - (Fraction(p0) + t * (Fraction(pn) - Fraction(p0))))
        rounding = half_ulp(p) + (1 - t) * half_ulp(p0) + t * half_ulp(pn)
        if miss > rounding:
            return True
    return False


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
            pairs, source_unit=None, value_unit="quarters", seed=1
        ) == LinearMap.from_pairs(
            sorted(pairs), source_unit=None, value_unit="quarters", seed=1
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

    @pytest.mark.parametrize(
        "position",
        [1.0000000000000007, 1.0000000000000002, 1.0, 0.9999999999999994],
    )
    def test_float_bound_edge_passes(self, position):
        # The exact line through (0, 0) and (2, 2) reaches 1 at 1. With t = 1/2
        # the bound is six 2**-53 above 1 (three ulps of 1) and five and a half
        # below, plus terms of order 2**-105: 1 + 3 * 2**-52 and 1 - 5 * 2**-53
        # are its last floats.
        m = LinearMap.from_pairs(
            [(0.0, 0.0), (position, 1.0), (2.0, 2.0)],
            source_unit=None,
            value_unit="seconds",
        )
        assert m.scalar == 1.0

    @pytest.mark.parametrize("position", [1.0000000000000009, 0.9999999999999993])
    def test_float_bound_edge_refuses(self, position):
        # 1 + 4 * 2**-52 and 1 - 6 * 2**-53: the first floats past the bound.
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
        # the exact position at which the line reaches 10, 1.17 of the bound.
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

    def test_drawn_axis_tick_moved_by_a_billionth_of_the_span_is_refused(self):
        moved = TICKS[2] + 1e-9 * (TICKS[4] - TICKS[0])
        assert moved == 504.46818366051133
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 2 (504.46818366051133, 20) lies off the line "
                "through the first and last pair, which reaches 20 at "
                "504.4681828049824."
            ),
        ):
            LinearMap.from_pairs(
                list(zip([*TICKS[:2], moved, *TICKS[3:]], [0, 10, 20, 30, 40])),
                source_unit="points",
                value_unit="seconds",
                origin=TICKS[0],
            )

    def test_drawn_axis_mislabelled_tick_is_refused(self):
        # Labelled 41, the last tick moves the line's 10 s to
        # 76.70371459980326 + 10/41 * 855.5289364103584.
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Calibration pair 1 (290.5859487023929, 10) lies off the line "
                "through the first and last pair, which reaches 10 at "
                "285.3693088462321."
            ),
        ):
            LinearMap.from_pairs(
                list(zip(TICKS, [0, 10, 20, 30, 41])),
                source_unit="points",
                value_unit="seconds",
                origin=TICKS[0],
            )

    @pytest.mark.parametrize(
        ("placement", "beyond"),
        [("rounded once", 0), ("(w*d)/D", 339), ("w*(d/D)", 281), ("(w/D)*d", 227)],
    )
    def test_random_axes_are_accepted(self, placement, beyond):
        calibrations = placed(placement)
        # Not vacuous: computed positions miss by more than rounding alone.
        assert sum(map(beyond_rounding_alone, calibrations)) == beyond
        for pairs in calibrations:
            LinearMap.from_pairs(pairs, source_unit="points", value_unit="seconds")

    @pytest.mark.parametrize("placement", list(PLACEMENTS))
    def test_random_axes_moved_by_a_billionth_of_the_span_are_refused(self, placement):
        for pairs in placed(placement):
            (p0, _), (pn, _) = pairs[0], pairs[-1]
            middle = len(pairs) // 2
            position, value = pairs[middle]
            moved = list(pairs)
            moved[middle] = (position + 1e-9 * (pn - p0), value)
            with pytest.raises(ValueError, match=f"Calibration pair {middle} "):
                LinearMap.from_pairs(moved, source_unit="points", value_unit="seconds")

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

    def test_exact_map_refuses_a_float(self):
        exact = LinearMap.from_pairs(
            [(0, 0), (6, 1)], source_unit="points", value_unit="seconds"
        )
        assert exact(1) == Fraction(1, 6)
        assert exact(Fraction(1, 2)) == Fraction(1, 12)
        with pytest.raises(
            OutputNumberTypeError,
            match=re.escape(
                f"Map {exact.id} is declared to write fractions and computes "
                "exactly, so it takes an int or a Fraction, not the float 1.0."
            ),
        ):
            exact(1.0)

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


QUARTERS = {"source_unit": "quarters", "target_unit": "quarters"}


class TestFloatParameters:
    """A float scalar or offset makes the map compute in doubles."""

    FLOAT_MAPS = [
        (lambda **k: LinearMap(scalar=0.1, **QUARTERS, **k), "LinearMap", 0.1),
        (lambda **k: LinearMap(offset=0.1, **QUARTERS, **k), "LinearMap", 1.1),
        (lambda **k: ScalarMap(0.1, **QUARTERS, **k), "ScalarMap", 0.1),
        (lambda **k: ShiftMap(offset=0.1, **QUARTERS, **k), "ShiftMap", 1.1),
    ]

    @pytest.mark.parametrize("build, name, _", FLOAT_MAPS)
    def test_fraction_declaration_is_refused(self, build, name, _):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{name} computes in doubles and cannot be declared to write "
                "fractions."
            ),
        ):
            build(output_number_type="fraction")

    @pytest.mark.parametrize("build, _, expected", FLOAT_MAPS)
    def test_undeclared_map_answers_its_float(self, build, _, expected):
        cmap = build()
        assert cmap.output_number_type is NumberType.float
        assert cmap(1) == expected
        assert type(cmap(1)) is float

    @pytest.mark.parametrize(
        "cmap, at, expected",
        [
            (LinearMap(scalar=Fraction(1, 3), offset=1, **QUARTERS), 1, Fraction(4, 3)),
            (ScalarMap(**QUARTERS), Fraction(1, 3), Fraction(1, 3)),
            (ShiftMap(offset=Fraction(1, 3), **QUARTERS), 1, Fraction(4, 3)),
        ],
    )
    def test_exact_parameters_stay_exact(self, cmap, at, expected):
        assert cmap.output_number_type is NumberType.fraction
        assert cmap(at) == expected
        assert type(cmap(at)) is Fraction


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
