"""Tests for composite maps (ChainMap, PiecewiseMap)."""

from __future__ import annotations

import re
from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core.enums import NumberType, TimeUnit
from timetoalign.maps.base import ConversionMap
from timetoalign.maps.composite import ChainMap, PiecewiseMap
from timetoalign.maps.interpolation import InterpolationMap
from timetoalign.maps.linear import LinearMap, ScalarMap


class TestChainMap:
    def test_initialization(self):
        m1 = ScalarMap(scalar=2.0, source_unit="quarters", target_unit="seconds")
        m2 = ScalarMap(scalar=10.0, source_unit="seconds", target_unit="milliseconds")

        chain = ChainMap([m1, m2])
        assert chain.source_unit == TimeUnit.quarters
        assert chain.target_unit == TimeUnit.milliseconds
        assert len(chain.maps) == 2

        # Test unit validation
        m3 = ScalarMap(scalar=1.0, source_unit="ticks")  # Incompatible with seconds
        with pytest.raises(ValueError, match="Incompatible units"):
            ChainMap([m1, m3])

    def test_conversion(self):
        m1 = ScalarMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5, offset=5.0)

        chain = ChainMap([m1, m2])

        # 10 -> 20 -> 0.5*20 + 5 = 15
        assert chain(10) == 15.0

        arr = np.array([10, 20])
        # [10, 20] -> [20, 40] -> [15, 25]
        expected = np.array([15.0, 25.0])
        np.testing.assert_array_equal(chain(arr), expected)

    def test_inverse(self):
        m1 = ScalarMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5, offset=5.0)
        chain = ChainMap([m1, m2])

        inv = chain.inverse()
        assert isinstance(inv, ChainMap)

        # Inverse: reverse order, inverse maps
        # m2_inv: y -> 2(y-5)
        # m1_inv: z -> 0.5z
        # Chain: y -> 2(y-5) -> 0.5 * 2(y-5) = y-5

        assert inv(15.0) == 10.0  # (15-5) = 10

        # Round trip
        val = 42.0
        assert inv(chain(val)) == val


class TestPiecewiseMap:
    def test_initialization(self):
        m1 = ScalarMap(scalar=1.0)
        m2 = ScalarMap(scalar=2.0)

        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        assert len(pm.maps) == 2

        with pytest.raises(ValueError, match="Number of maps"):
            PiecewiseMap(breaks=[0, 10], maps=[m1, m2])

        with pytest.raises(ValueError, match="strictly increasing"):
            PiecewiseMap(breaks=[0, 0, 10], maps=[m1, m2])

    def test_conversion(self):
        # Map 1: y = x (0 <= x < 10)
        # Map 2: y = 2x (10 <= x < 20)
        m1 = ScalarMap(scalar=1.0)
        m2 = ScalarMap(scalar=2.0)

        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        # First interval
        assert pm(5) == 5.0
        assert pm(0) == 0.0

        # Second interval
        assert pm(10) == 20.0
        assert pm(15) == 30.0

        # Out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            pm(20)
        with pytest.raises(ValueError, match="out of bounds"):
            pm(-1)

    def test_array_conversion(self):
        m1 = ScalarMap(scalar=1.0)
        m2 = ScalarMap(scalar=2.0)
        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        # [5, 15] -> [5, 30]
        arr = np.array([5, 15])
        expected = np.array([5.0, 30.0])
        np.testing.assert_array_equal(pm(arr), expected)

        # Out of bounds
        with pytest.raises(ValueError):
            pm(np.array([5, 25]))

    def test_inverse(self):
        # m1: y = x [0, 10) -> [0, 10)
        # m2: y = 2x [10, 20) -> [20, 40)
        # Note: Gaps in target! [10, 20) is missing.
        # This PiecewiseMap is technically invertible only on its image.

        # Let's try a continuous one
        # m1: y = x [0, 10) -> [0, 10)
        # m2: y = x [10, 20) -> [10, 20)

        m1 = ScalarMap(scalar=1.0)
        m2 = ScalarMap(scalar=1.0)  # Identity
        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        inv = pm.inverse()
        # Breaks should be mapped: 0->0, 10->10, 20->20
        # Maps inverted

        assert inv(5) == 5.0
        assert inv(15) == 15.0

        # Now with scaling
        # m1: y = 2x [0, 10) -> [0, 20)
        # m2: y = x + 10 [10, 20) -> [20, 30) (Continuous at 10: 2*10=20, 10+10=20)
        m1 = ScalarMap(scalar=2.0)
        m2 = LinearMap(scalar=1.0, offset=10.0)

        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])
        inv = pm.inverse()

        # New breaks: 0->0, 10->20, 20->30
        np.testing.assert_array_equal(inv.breaks, [0, 20, 30])

        # Test inverse values
        # 10 (in first interval) -> 5
        assert inv(10) == 5.0
        # 25 (in second interval) -> 15
        assert inv(25) == 15.0


class TestChainMapConvertArray:
    """Tests for ChainMap.convert_array() public API."""

    def test_convert_array_returns_ndarray(self):
        """convert_array() returns numpy array."""
        m1 = ScalarMap(scalar=2.0)
        m2 = LinearMap(scalar=0.5, offset=5.0)
        chain = ChainMap([m1, m2])

        values = np.array([0.0, 10.0, 20.0])
        result = chain.convert_array(values)

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape

    def test_convert_array_correct_values(self):
        """convert_array() computes correct values through chain."""
        m1 = ScalarMap(scalar=2.0)  # x -> 2x
        m2 = LinearMap(scalar=0.5, offset=5.0)  # x -> 0.5x + 5
        chain = ChainMap([m1, m2])

        values = np.array([0.0, 10.0, 20.0])
        # 0 -> 0 -> 5
        # 10 -> 20 -> 15
        # 20 -> 40 -> 25

        result = chain.convert_array(values)

        np.testing.assert_array_equal(result, [5.0, 15.0, 25.0])

    def test_convert_array_matches_scalar(self):
        """convert_array() results match element-wise scalar conversion."""
        m1 = LinearMap(scalar=1.5, offset=-3.0)
        m2 = ScalarMap(scalar=0.1)
        chain = ChainMap([m1, m2])

        values = np.array([0.0, 10.0, -10.0, 100.0])

        array_result = chain.convert_array(values)
        scalar_results = np.array([chain(v) for v in values])

        # Array and scalar paths perform identical IEEE 754 operations
        np.testing.assert_array_equal(array_result, scalar_results)


class TestPiecewiseMapConvertArray:
    """Tests for PiecewiseMap.convert_array() public API."""

    def test_convert_array_returns_ndarray(self):
        """convert_array() returns numpy array."""
        m1 = ScalarMap(scalar=1.0)
        m2 = ScalarMap(scalar=2.0)
        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        values = np.array([5.0, 15.0])
        result = pm.convert_array(values)

        assert isinstance(result, np.ndarray)
        assert result.shape == values.shape

    def test_convert_array_correct_values(self):
        """convert_array() applies correct map per region."""
        m1 = ScalarMap(scalar=1.0)  # [0, 10)
        m2 = ScalarMap(scalar=2.0)  # [10, 20)
        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        values = np.array([0.0, 5.0, 10.0, 15.0, 19.9])
        # 0 -> 0, 5 -> 5, 10 -> 20, 15 -> 30, 19.9 -> 39.8

        result = pm.convert_array(values)

        # All operands are exactly representable; scalar multiplication is exact
        np.testing.assert_array_equal(result, [0.0, 5.0, 20.0, 30.0, 39.8])

    def test_convert_array_matches_scalar(self):
        """convert_array() results match element-wise scalar conversion."""
        m1 = LinearMap(scalar=2.0, offset=0.0)  # [0, 10)
        m2 = LinearMap(scalar=1.0, offset=10.0)  # [10, 20)
        pm = PiecewiseMap(breaks=[0, 10, 20], maps=[m1, m2])

        values = np.array([0.0, 5.0, 9.9, 10.0, 15.0, 19.9])

        array_result = pm.convert_array(values)
        scalar_results = np.array([pm(v) for v in values])

        # Array and scalar paths perform identical IEEE 754 operations
        np.testing.assert_array_equal(array_result, scalar_results)


def exact_members() -> list[ScalarMap]:
    # quarters -> quarters -> seconds, 1/6 s per quarter: no float is 1/6.
    return [
        ScalarMap(
            scalar=Fraction(1, 2), source_unit="quarters", target_unit="quarters"
        ),
        ScalarMap(scalar=Fraction(1, 3), source_unit="quarters", target_unit="seconds"),
    ]


def exact_chain(**kwargs) -> ChainMap:
    return ChainMap(exact_members(), **kwargs)


def exact_piecewise(**kwargs) -> PiecewiseMap:
    return PiecewiseMap(
        breaks=[0, 6],
        maps=[
            ScalarMap(
                scalar=Fraction(1, 6), source_unit="quarters", target_unit="seconds"
            )
        ],
        source_unit="quarters",
        target_unit="seconds",
        **kwargs,
    )


@pytest.mark.parametrize("build", [exact_chain, exact_piecewise])
class TestDeclaredOutputNumberType:
    def test_undeclared_map_writes_the_target_units_default(self, build):
        cmap = build()
        assert cmap.output_number_type is NumberType.float
        assert cmap(1) == 0.16666666666666666
        assert type(cmap(1)) is float

    def test_fraction_map_answers_exactly(self, build):
        cmap = build(output_number_type="fraction")
        assert cmap.output_number_type is NumberType.fraction
        assert cmap(1) == Fraction(1, 6)
        assert type(cmap(1)) is Fraction

    def test_fraction_map_refuses_a_float(self, build):
        cmap = build(output_number_type="fraction", uid="exact")
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Map exact is declared to write fractions and computes exactly, "
                "so it takes an int or a Fraction, not the float 1.0."
            ),
        ):
            cmap(1.0)

    def test_inverse_carries_the_declaration(self, build):
        inverse = build(output_number_type="fraction").inverse()
        assert inverse.output_number_type is NumberType.fraction
        assert inverse(Fraction(1, 6)) == Fraction(1)
        assert type(inverse(Fraction(1, 6))) is Fraction

    def test_declaration_is_not_written(self, build):
        assert (
            build(output_number_type="fraction", uid="m").to_dict()
            == build(uid="m").to_dict()
        )


def quarters_scalar(scalar, **kwargs) -> ScalarMap:
    return ScalarMap(
        scalar=scalar, source_unit="quarters", target_unit="quarters", **kwargs
    )


def interpolation_member() -> InterpolationMap:
    # quarters [0, 3] -> [0, 1]: 1/3 at 1, which no float represents.
    return InterpolationMap(
        source_coords=[0.0, 3.0],
        target_coords=[0.0, 1.0],
        source_id="a",
        target_id="b",
        source_unit="quarters",
        target_unit="quarters",
    )


def quarters_piecewise(member: ConversionMap, end: int = 3, **kwargs) -> PiecewiseMap:
    return PiecewiseMap(
        breaks=[0, end],
        maps=[member],
        source_unit="quarters",
        target_unit="quarters",
        **kwargs,
    )


class TestFloatArithmeticMembers:
    @pytest.mark.parametrize(
        "build, name",
        [
            (
                lambda **k: ChainMap(
                    [interpolation_member(), quarters_scalar(Fraction(1, 2))], **k
                ),
                "ChainMap",
            ),
            (
                lambda **k: quarters_piecewise(interpolation_member(), **k),
                "PiecewiseMap",
            ),
        ],
    )
    def test_interpolation_member_refuses_fraction(self, build, name):
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"{name} computes in doubles and cannot be declared to write "
                "fractions."
            ),
        ):
            build(output_number_type="fraction")

    def test_chain_with_interpolation_member_answers_the_float(self):
        chain = ChainMap([interpolation_member(), quarters_scalar(Fraction(1, 2))])
        assert chain.output_number_type is NumberType.float
        assert chain(Fraction(1)) == 0.16666666666666666
        assert type(chain(Fraction(1))) is float

    def test_piecewise_with_interpolation_member_answers_the_float(self):
        piecewise = quarters_piecewise(interpolation_member())
        assert piecewise.output_number_type is NumberType.float
        assert piecewise(Fraction(1)) == 0.3333333333333333
        assert type(piecewise(Fraction(1))) is float

    def test_exact_members_stay_exact(self):
        chain = ChainMap(
            [quarters_scalar(Fraction(1, 2)), quarters_scalar(Fraction(1, 3))]
        )
        piecewise = quarters_piecewise(quarters_scalar(Fraction(1, 6)), end=6)
        for cmap in (chain, piecewise):
            assert cmap.output_number_type is NumberType.fraction
            assert cmap(1) == Fraction(1, 6)
            assert type(cmap(1)) is Fraction

    def test_chain_member_declaration_applies(self):
        def members() -> list[ScalarMap]:
            return [
                quarters_scalar(Fraction(1, 2), output_number_type="float"),
                quarters_scalar(Fraction(1, 3)),
            ]

        chain = ChainMap(members())
        assert chain.output_number_type is NumberType.float
        assert chain(1) == 0.16666666666666666
        assert type(chain(1)) is float
        with pytest.raises(ValueError, match="ChainMap computes in doubles"):
            ChainMap(members(), output_number_type="fraction")

    def test_piecewise_member_declaration_applies(self):
        def member() -> ScalarMap:
            return quarters_scalar(Fraction(1, 3), output_number_type="float")

        piecewise = quarters_piecewise(member())
        assert piecewise.output_number_type is NumberType.float
        assert piecewise(1) == 0.3333333333333333
        assert type(piecewise(1)) is float
        with pytest.raises(ValueError, match="PiecewiseMap computes in doubles"):
            quarters_piecewise(member(), output_number_type="fraction")

    def test_chain_member_guard_applies(self):
        chain = ChainMap(
            [
                quarters_scalar(0.5),
                quarters_scalar(
                    Fraction(1, 3), output_number_type="fraction", uid="exact"
                ),
            ]
        )
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Map exact is declared to write fractions and computes exactly, "
                "so it takes an int or a Fraction, not the float 0.5."
            ),
        ):
            chain(1)

    def test_piecewise_member_guard_applies(self):
        piecewise = quarters_piecewise(
            quarters_scalar(Fraction(1, 3), output_number_type="fraction", uid="exact")
        )
        with pytest.raises(
            TypeError,
            match=re.escape(
                "Map exact is declared to write fractions and computes exactly, "
                "so it takes an int or a Fraction, not the float 1.0."
            ),
        ):
            piecewise(1.0)


class TestPiecewiseExactIntervalSelection:
    def test_exact_input_is_compared_exactly(self):
        piecewise = PiecewiseMap(
            breaks=[0, Fraction(3, 10), 1],
            maps=[quarters_scalar(1), quarters_scalar(2)],
            source_unit="quarters",
            target_unit="quarters",
            output_number_type="fraction",
        )
        below = Fraction(3, 10) - Fraction(1, 10**20)
        assert float(below) == 0.3
        assert piecewise(below) == Fraction(29999999999999999999, 10**20)
        assert piecewise(Fraction(3, 10)) == Fraction(3, 5)
