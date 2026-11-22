"""Property-based tests for ConversionMaps using Hypothesis.

These tests verify mathematical invariants that must hold for all valid inputs:
- Invertibility: inverse(forward(x)) == x
- Composition: (f >> g)(x) == g(f(x))
- Identity: identity(x) == x
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from timetoalign.maps.composite import ChainMap
from timetoalign.maps.convenience import (
    QuartersToTicks,
    SamplesToSeconds,
    SecondsToSamples,
    TicksToQuarters,
)
from timetoalign.maps.linear import LinearMap, ScalarMap, ShiftMap
from timetoalign.maps.table import TableMap

# region Strategies


# Finite floats with reasonable range for numerical stability
# Using smaller range to avoid precision loss in extreme cases
finite_floats = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

# Positive scalars with reasonable range for numerical stability
positive_scalars = st.floats(
    min_value=1e-3,  # Not too small to avoid precision issues
    max_value=1e3,  # Not too large to avoid overflow
    allow_nan=False,
    allow_infinity=False,
)

# Non-zero scalars (positive or negative) with reasonable range
nonzero_scalars = st.one_of(
    st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e3, max_value=-1e-3, allow_nan=False, allow_infinity=False),
)

# PPQ values (typical MIDI ticks per quarter)
ppq_values = st.sampled_from([96, 120, 240, 480, 960, 1920])

# Sample rates
sample_rates = st.sampled_from([22050, 44100, 48000, 96000])

# endregion


# region LinearMap Properties


class TestLinearMapProperties:
    """Property-based tests for LinearMap."""

    @given(scalar=nonzero_scalars, offset=finite_floats, x=finite_floats)
    @settings(max_examples=200)
    def test_inverse_roundtrip(self, scalar: float, offset: float, x: float):
        """inverse(forward(x)) == x for all valid x.

        This is the fundamental invertibility property.
        """
        m = LinearMap(scalar=scalar, offset=offset)
        inv = m.inverse()

        forward = m(x)
        back = inv(forward)

        # Use reasonable tolerance for floating point operations
        # With extreme scale differences, some precision loss is expected
        assert back == pytest.approx(x, rel=1e-6, abs=1e-6)

    @given(scalar=nonzero_scalars, offset=finite_floats, x=finite_floats)
    @settings(max_examples=200)
    def test_forward_inverse_roundtrip(self, scalar: float, offset: float, x: float):
        """forward(inverse(x)) == x for all valid x.

        Tests the other direction of invertibility.
        """
        m = LinearMap(scalar=scalar, offset=offset)
        inv = m.inverse()

        back = inv(x)
        forward = m(back)

        assert forward == pytest.approx(x, rel=1e-6, abs=1e-6)

    @given(
        s1=nonzero_scalars,
        o1=finite_floats,
        s2=nonzero_scalars,
        o2=finite_floats,
        x=finite_floats,
    )
    @settings(max_examples=200)
    def test_composition_associativity(
        self, s1: float, o1: float, s2: float, o2: float, x: float
    ):
        """(f >> g)(x) == g(f(x)).

        Composition via >> operator matches sequential application.
        """
        m1 = LinearMap(scalar=s1, offset=o1)
        m2 = LinearMap(scalar=s2, offset=o2)

        chain = m1 >> m2
        sequential = m2(m1(x))

        # ChainMap applies the maps in the same order as sequential application
        # (base.py:303-319, composite.py:101-122), so the two paths execute the
        # identical sequence of float operations and agree bit-for-bit.
        assert chain(x) == sequential

    @given(x=finite_floats)
    def test_identity_map(self, x: float):
        """Identity map returns input unchanged."""
        m = LinearMap(scalar=1.0, offset=0.0)
        assert m.is_identity
        assert m(x) == x


# endregion


# region ScalarMap Properties


class TestScalarMapProperties:
    """Property-based tests for ScalarMap."""

    @given(scalar=positive_scalars, x=finite_floats)
    @settings(max_examples=200)
    def test_inverse_roundtrip(self, scalar: float, x: float):
        """inverse(forward(x)) == x."""
        m = ScalarMap(scalar=scalar)
        inv = m.inverse()

        assert inv(m(x)) == pytest.approx(x, rel=1e-6, abs=1e-6)

    @given(scalar=positive_scalars, x=finite_floats)
    def test_scaling_property(self, scalar: float, x: float):
        """ScalarMap(a)(x) == a * x."""
        m = ScalarMap(scalar=scalar)
        # The test computes the exact same product the implementation does
        # (linear.py:237-239), so the results are bit-identical.
        assert m(x) == scalar * x


# endregion


# region ShiftMap Properties


class TestShiftMapProperties:
    """Property-based tests for ShiftMap."""

    @given(offset=finite_floats, x=finite_floats)
    @settings(max_examples=200)
    def test_inverse_roundtrip(self, offset: float, x: float):
        """inverse(forward(x)) == x."""
        m = ShiftMap(offset=offset)
        inv = m.inverse()

        assert inv(m(x)) == pytest.approx(x, rel=1e-6, abs=1e-6)

    @given(offset=finite_floats, x=finite_floats)
    def test_shift_property(self, offset: float, x: float):
        """ShiftMap(b)(x) == x + b."""
        m = ShiftMap(offset=offset)
        # The test computes the exact same sum the implementation does
        # (linear.py:338-340), so the results are bit-identical.
        assert m(x) == x + offset

    @given(x=finite_floats)
    def test_zero_shift_is_identity(self, x: float):
        """ShiftMap(0) is identity."""
        m = ShiftMap(offset=0.0)
        assert m.is_identity
        assert m(x) == x


# endregion


# region Convenience Map Properties


class TestTicksToQuartersProperties:
    """Property-based tests for TicksToQuarters / QuartersToTicks."""

    @given(ppq=ppq_values, ticks=st.integers(min_value=0, max_value=10_000_000))
    def test_inverse_roundtrip_int(self, ppq: int, ticks: int):
        """Round-trip for integer tick values."""
        t2q = TicksToQuarters(ppq)
        q2t = QuartersToTicks(ppq)

        quarters = t2q(ticks)
        back = q2t(quarters)

        # Integer ticks round-trip exactly: the quarters leg is an exact
        # ratio and the tick leg is whole, so nothing is left to lose.
        assert back == ticks
        assert isinstance(back, int)

    def test_off_tick_quarters_name_the_tick_they_land_on(self) -> None:
        """A quarter position between ticks resolves to one whole tick.

        The tick axis holds whole numbers, so the outward leg quantizes --
        1.234375 quarters at 96 ppq is 118.5 ticks, which is not a tick.
        Half-to-even names 118, and the return leg reads that tick exactly.
        The original float does not come back, and saying so is the point:
        a position stated finer than the axis can hold is a position the
        axis has to choose for.
        """
        t2q = TicksToQuarters(96)
        q2t = QuartersToTicks(96)

        assert q2t(1.234375) == 118
        assert t2q(118) == Fraction(59, 48)
        assert t2q(119) == Fraction(119, 96)

    @given(ppq=ppq_values)
    def test_one_quarter_equals_ppq_ticks(self, ppq: int):
        """1 quarter note == ppq ticks (by definition)."""
        t2q = TicksToQuarters(ppq)
        q2t = QuartersToTicks(ppq)

        assert t2q(ppq) == 1.0
        assert q2t(1.0) == ppq


class TestSamplesToSecondsProperties:
    """Property-based tests for SamplesToSeconds / SecondsToSamples."""

    @given(sr=sample_rates, samples=st.integers(min_value=0, max_value=100_000_000))
    def test_inverse_roundtrip_int(self, sr: int, samples: int):
        """Round-trip for integer sample values."""
        s2s = SamplesToSeconds(sr)
        s2samp = SecondsToSamples(sr)

        seconds = s2s(samples)
        back = s2samp(seconds)

        assert back == pytest.approx(samples, abs=1e-6)

    def test_sub_sample_seconds_name_the_sample_they_land_on(self) -> None:
        """A second position finer than one sample resolves to one sample.

        The samples axis holds whole numbers, so a time between two of them
        names one; 0.0200005 s at 48 kHz falls inside sample 960, and the
        seconds read back off that sample are the sample's own time, not the
        time that was asked for.
        """
        s2s = SamplesToSeconds(48000)
        s2samp = SecondsToSamples(48000)

        assert s2samp(0.0200005) == 960
        assert s2s(960) == 0.02
        assert s2s(961) == 0.020020833333333335

    @given(sr=sample_rates)
    def test_one_second_equals_sample_rate(self, sr: int):
        """1 second == sample_rate samples (by definition)."""
        s2s = SamplesToSeconds(sr)
        s2samp = SecondsToSamples(sr)

        assert s2s(sr) == 1.0
        assert s2samp(1.0) == sr


# endregion


# region TableMap Properties


class TestTableMapProperties:
    """Property-based tests for TableMap."""

    @given(
        x_start=st.floats(min_value=0, max_value=100),
        x_step=st.floats(min_value=0.1, max_value=10),
        y_start=st.floats(min_value=0, max_value=100),
        y_step=st.floats(min_value=0.1, max_value=10),
    )
    @settings(max_examples=100)
    def test_inverse_roundtrip_monotonic(
        self, x_start: float, x_step: float, y_start: float, y_step: float
    ):
        """inverse(forward(x)) == x for monotonically increasing maps."""
        assume(not any(np.isnan([x_start, x_step, y_start, y_step])))
        assume(not any(np.isinf([x_start, x_step, y_start, y_step])))

        # Create strictly monotonic table
        x_values = [x_start, x_start + x_step, x_start + 2 * x_step]
        y_values = [y_start, y_start + y_step, y_start + 2 * y_step]

        m = TableMap(x_values=x_values, y_values=y_values)
        assert m.is_invertible

        inv = m.inverse()

        # Test at table points (anchors). np.interp returns the stored ordinate
        # exactly at each anchor (table.py:177-187), so the round trip through
        # the inverse recovers the anchor abscissa bit-for-bit.
        for x in x_values:
            y = m(x)
            back = inv(y)
            assert back == x

        # Test at midpoints
        for i in range(len(x_values) - 1):
            x_mid = (x_values[i] + x_values[i + 1]) / 2
            y = m(x_mid)
            back = inv(y)
            assert back == pytest.approx(x_mid, rel=1e-9, abs=1e-9)


class TestTableMapTempoProperties:
    """Property-based tests for TableMap.from_tempo_changes."""

    @given(
        tempo1=st.floats(min_value=20, max_value=300),
        tempo2=st.floats(min_value=20, max_value=300),
        ppq=ppq_values,
    )
    @settings(max_examples=100)
    def test_tempo_map_inverse_roundtrip(self, tempo1: float, tempo2: float, ppq: int):
        """Tempo maps are invertible for valid tick values."""
        assume(not np.isnan(tempo1) and not np.isnan(tempo2))

        m = TableMap.from_tempo_changes(
            tick_positions=[0, ppq * 4],  # Change at measure 2
            tempos_bpm=[tempo1, tempo2],
            ticks_per_quarter=ppq,
        )

        # Test at key points
        test_ticks = [0, ppq, ppq * 2, ppq * 4, ppq * 8]
        inv = m.inverse()

        for ticks in test_ticks:
            seconds = m(ticks)
            back = inv(seconds)
            assert back == pytest.approx(ticks, rel=1e-9, abs=1e-6)

    @given(tempo=st.floats(min_value=20, max_value=300), ppq=ppq_values)
    def test_constant_tempo_linear_relationship(self, tempo: float, ppq: int):
        """With constant tempo, ticks and seconds have linear relationship."""
        assume(not np.isnan(tempo))

        m = TableMap.from_tempo_changes(
            tick_positions=[0],
            tempos_bpm=[tempo],
            ticks_per_quarter=ppq,
        )

        # Seconds per tick = 60 / (tempo * ppq)
        sec_per_tick = 60.0 / (tempo * ppq)

        # Test linearity
        t1, t2 = 0, ppq * 4
        s1, s2 = m(t1), m(t2)

        expected_duration = (t2 - t1) * sec_per_tick
        assert (s2 - s1) == pytest.approx(expected_duration, rel=1e-9)


# endregion


# region ChainMap Properties


class TestChainMapProperties:
    """Property-based tests for ChainMap."""

    @given(
        s1=st.floats(
            min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False
        ),
        o1=st.floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        s2=st.floats(
            min_value=0.1, max_value=10, allow_nan=False, allow_infinity=False
        ),
        o2=st.floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
        x=st.floats(
            min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_chain_inverse_roundtrip(
        self, s1: float, o1: float, s2: float, o2: float, x: float
    ):
        """ChainMap inverse works correctly.

        Note: Value ranges are constrained to avoid numerical instability:
        - Scalars: [0.1, 10] to avoid extreme compression/expansion
        - Offsets: [-1e3, 1e3] to avoid precision loss with large intermediate values
        - Inputs: [-1e3, 1e3] for same reason

        With extreme parameters (e.g., scalar=0.01 with offset=8000),
        floating-point precision loss during forward/inverse operations
        can exceed tight tolerances. This is inherent to IEEE 754 floats,
        not a bug in the ChainMap implementation.

        Tolerance is 1e-6 to account for compounded errors in chained operations.
        """
        m1 = LinearMap(scalar=s1, offset=o1)
        m2 = LinearMap(scalar=s2, offset=o2)

        chain = ChainMap([m1, m2])
        inv = chain.inverse()

        y = chain(x)
        back = inv(y)

        # Tolerance of 1e-6 is appropriate for chained floating-point operations
        assert back == pytest.approx(x, rel=1e-6, abs=1e-6)

    @given(
        s1=nonzero_scalars,
        o1=finite_floats,
        s2=nonzero_scalars,
        o2=finite_floats,
        s3=nonzero_scalars,
        o3=finite_floats,
        x=finite_floats,
    )
    @settings(max_examples=100)
    def test_chain_associativity(
        self,
        s1: float,
        o1: float,
        s2: float,
        o2: float,
        s3: float,
        o3: float,
        x: float,
    ):
        """(f >> g) >> h == f >> (g >> h)."""
        m1 = LinearMap(scalar=s1, offset=o1)
        m2 = LinearMap(scalar=s2, offset=o2)
        m3 = LinearMap(scalar=s3, offset=o3)

        left = (m1 >> m2) >> m3
        right = m1 >> (m2 >> m3)

        # Both groupings flatten to the same ordered ChainMap and therefore
        # execute the identical sequence of float operations (composite.py:101-122),
        # so associativity holds bit-for-bit.
        assert left(x) == right(x)


# endregion


# region Fraction Support Properties


class TestFractionProperties:
    """Property-based tests for Fraction support in maps."""

    @given(
        num=st.integers(min_value=1, max_value=100),
        den=st.integers(min_value=1, max_value=100),
        x_num=st.integers(min_value=-1000, max_value=1000),
        x_den=st.integers(min_value=1, max_value=100),
    )
    def test_linear_map_fraction_exact(
        self, num: int, den: int, x_num: int, x_den: int
    ):
        """LinearMap preserves Fraction exactness."""
        scalar = Fraction(num, den)
        offset = Fraction(1, den)
        x = Fraction(x_num, x_den)

        m = LinearMap(scalar=scalar, offset=offset)
        result = m(x)

        # Result should be exact Fraction
        expected = scalar * x + offset
        assert result == expected

    @given(
        ppq=ppq_values,
        beats_num=st.integers(min_value=0, max_value=100),
        beats_den=st.sampled_from([1, 2, 4, 8, 16]),
    )
    def test_ticks_to_quarters_fraction_exact(
        self, ppq: int, beats_num: int, beats_den: int
    ):
        """TicksToQuarters preserves exactness for integer tick inputs."""
        ticks = ppq * beats_num // beats_den * beats_den  # Ensure divisible

        t2q = TicksToQuarters(ppq)
        quarters = t2q(ticks)

        assert quarters == Fraction(ticks, ppq)
        assert type(quarters) is Fraction


# endregion
