"""Tests for InterpolationMap.

InterpolationMap is a full ConversionMap family member providing
bidirectional anchor-pair coordinate conversion for TimelineGroup and
WarpMap.
"""

from __future__ import annotations

import numpy as np
import pytest

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.maps import ConversionMap, InterpolationMap, LinearMap


class TestInterpolationMapBasics:
    """Basic functionality tests."""

    def test_create_simple(self):
        """Create a simple interpolation map."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child:1",
            target_id="parent:1",
        )
        assert imap.source_id == "child:1"
        assert imap.target_id == "parent:1"
        assert imap.n_anchors == 2

    def test_forward_conversion(self):
        """Calling the map converts source -> target."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        # child 0 -> parent 10
        assert imap(0.0) == 10.0
        # child 50 -> parent 60
        assert imap(50.0) == 60.0
        # child 100 -> parent 110
        assert imap(100.0) == 110.0

    def test_inverse_conversion(self):
        """inverse() returns a map that converts target -> source."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        inv = imap.inverse()
        # parent 10 -> child 0
        assert inv(10.0) == 0.0
        # parent 60 -> child 50
        assert inv(60.0) == 50.0
        # parent 110 -> child 100
        assert inv(110.0) == 100.0

    def test_forward_array(self):
        """Calling the map with an array input."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([0.0, 200.0]),  # 2x scaling
            source_id="source",
            target_id="target",
        )
        values = np.array([0.0, 25.0, 50.0, 100.0])
        result = imap.convert_array(values)
        expected = np.array([0.0, 50.0, 100.0, 200.0])
        assert np.array_equal(result, expected)

    def test_inverse_array(self):
        """inverse() map with array input."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([0.0, 200.0]),
            source_id="source",
            target_id="target",
        )
        values = np.array([0.0, 50.0, 100.0, 200.0])
        result = imap.inverse().convert_array(values)
        expected = np.array([0.0, 25.0, 50.0, 100.0])
        assert np.array_equal(result, expected)


class TestInterpolationMapIsConversionMap:
    """InterpolationMap is a ConversionMap family member."""

    def test_is_subclass(self):
        """InterpolationMap subclasses ConversionMap."""
        assert issubclass(InterpolationMap, ConversionMap)

    def test_coordinate_input_matching_unit(self):
        """A Coordinate matching the map's source unit converts normally."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 480.0]),
            target_coords=np.array([0.0, 1.0]),
            source_id="ticks",
            target_id="seconds",
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        result = imap(Coordinate(240.0, TimeUnit.ticks))
        assert result == 0.5

    def test_coordinate_input_wrong_unit_raises(self):
        """A Coordinate with an incompatible unit raises via the base __call__."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 480.0]),
            target_coords=np.array([0.0, 1.0]),
            source_id="ticks",
            target_id="seconds",
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        with pytest.raises(ValueError, match="does not match"):
            imap(Coordinate(240.0, TimeUnit.seconds))


class TestInterpolationMapWithUnits:
    """Tests with TimeUnit annotations."""

    def test_units_stored(self):
        """TimeUnits are stored correctly."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 480.0]),
            target_coords=np.array([0.0, 1.0]),
            source_id="ticks",
            target_id="seconds",
            source_unit=TimeUnit.ticks,
            target_unit=TimeUnit.seconds,
        )
        assert imap.source_unit == TimeUnit.ticks
        assert imap.target_unit == TimeUnit.seconds


class TestInterpolationMapTempoConversion:
    """Tests for tempo-based conversion (non-linear mapping)."""

    def test_tempo_change(self):
        """Conversion with tempo change in the middle."""
        # 0-480 ticks at 120 BPM = 0.5 sec
        # 480-960 ticks at 60 BPM = 1.0 sec (total = 1.5 sec at tick 960)
        imap = InterpolationMap(
            source_coords=np.array([0.0, 480.0, 960.0]),
            target_coords=np.array([0.0, 0.5, 1.5]),
            source_id="ticks",
            target_id="seconds",
        )

        # Check anchor points
        assert imap(0.0) == 0.0
        assert imap(480.0) == 0.5
        assert imap(960.0) == 1.5

        # Check interpolation within first segment (faster tempo)
        assert imap(240.0) == 0.25

        # Check interpolation within second segment (slower tempo)
        assert imap(720.0) == 1.0

    def test_tempo_inverse(self):
        """Inverse conversion with tempo changes."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 480.0, 960.0]),
            target_coords=np.array([0.0, 0.5, 1.5]),
            source_id="ticks",
            target_id="seconds",
        )
        inv = imap.inverse()

        # Check inverse at anchor points
        assert inv(0.0) == 0.0
        assert inv(0.5) == 480.0
        assert inv(1.5) == 960.0

        # Check interpolation
        assert inv(0.25) == 240.0
        assert inv(1.0) == 720.0


class TestInterpolationMapExtrapolation:
    """Tests for extrapolation beyond anchor points."""

    def test_forward_extrapolation(self):
        """Forward extrapolation extends linearly."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([0.0, 200.0]),  # 2x scaling
            source_id="source",
            target_id="target",
        )
        # Beyond max: extrapolate
        assert imap(150.0) == 300.0
        # Below min: extrapolate
        assert imap(-50.0) == -100.0

    def test_inverse_extrapolation(self):
        """Inverse extrapolation extends linearly."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([0.0, 200.0]),
            source_id="source",
            target_id="target",
        )
        inv = imap.inverse()
        # Beyond max: extrapolate
        assert inv(300.0) == 150.0
        # Below min: extrapolate
        assert inv(-100.0) == -50.0


class TestInterpolationMapValidation:
    """Tests for validation and error handling."""

    def test_requires_two_points(self):
        """Must have at least 2 anchor points."""
        with pytest.raises(ValueError, match="at least 2 anchor"):
            InterpolationMap(
                source_coords=np.array([0.0]),
                target_coords=np.array([0.0]),
                source_id="a",
                target_id="b",
            )

    def test_length_mismatch(self):
        """Source and target must have same length."""
        with pytest.raises(ValueError, match="same length"):
            InterpolationMap(
                source_coords=np.array([0.0, 100.0]),
                target_coords=np.array([0.0, 50.0, 100.0]),
                source_id="a",
                target_id="b",
            )

    def test_source_must_be_increasing(self):
        """Source coords must be strictly monotonically increasing."""
        with pytest.raises(ValueError, match="monotonically increasing"):
            InterpolationMap(
                source_coords=np.array([0.0, 50.0, 30.0]),
                target_coords=np.array([0.0, 100.0, 60.0]),
                source_id="a",
                target_id="b",
            )

    def test_non_invertible_target(self):
        """Non-monotonic target prevents inversion."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 50.0, 100.0]),
            target_coords=np.array([0.0, 100.0, 50.0]),  # Not monotonic
            source_id="a",
            target_id="b",
        )
        assert not imap.is_invertible
        with pytest.raises(ValueError, match="not strictly monotonic"):
            imap.inverse()


class TestInterpolationMapInverseCaching:
    """Tests for inverse() caching."""

    def test_inverse_is_cached(self):
        """Repeated inverse() calls return the same instance."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        inv1 = imap.inverse()
        inv2 = imap.inverse()
        assert inv1 is inv2

    def test_inverse_cache_is_symmetric(self):
        """The inverse's own inverse() returns the original instance."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        inv = imap.inverse()
        assert inv.inverse() is imap


class TestInterpolationMapDecreasingTarget:
    """Tests for maps with decreasing target values."""

    def test_decreasing_target_forward(self):
        """Forward works with decreasing target."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([100.0, 0.0]),  # Decreasing
            source_id="source",
            target_id="target",
        )
        assert imap.is_invertible
        assert imap(0.0) == 100.0
        assert imap(100.0) == 0.0
        assert imap(50.0) == 50.0

    def test_decreasing_target_inverse(self):
        """Inverse works with decreasing target."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([100.0, 0.0]),
            source_id="source",
            target_id="target",
        )
        inv = imap.inverse()
        # Note: inverse reverses the arrays internally
        assert inv(100.0) == 0.0
        assert inv(0.0) == 100.0
        assert inv(50.0) == 50.0


class TestInterpolationMapFactories:
    """Tests for factory methods."""

    def test_identity(self):
        """Identity map: output equals input."""
        imap = InterpolationMap.identity(0.0, 100.0, "test")
        assert imap(50.0) == 50.0
        assert imap.inverse()(50.0) == 50.0


class TestInterpolationMapProperties:
    """Tests for property accessors."""

    def test_min_max(self):
        """Min/max properties work correctly."""
        imap = InterpolationMap(
            source_coords=np.array([10.0, 50.0, 100.0]),
            target_coords=np.array([0.0, 25.0, 50.0]),
            source_id="a",
            target_id="b",
        )
        assert imap.source_min == 10.0
        assert imap.source_max == 100.0
        assert imap.target_min == 0.0
        assert imap.target_max == 50.0

    def test_repr(self):
        """Repr shows useful info."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([0.0, 200.0]),
            source_id="child:1",
            target_id="parent:1",
        )
        r = repr(imap)
        assert "child:1" in r
        assert "parent:1" in r
        assert "2" in r  # n_anchors


class TestInterpolationMapReadOnlyArrays:
    """Tests for read-only coordinate arrays and inverse cache integrity."""

    def test_source_coords_read_only(self):
        """Mutating source_coords raises ValueError."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        with pytest.raises(ValueError):
            imap.source_coords[0] = 5.0

    def test_target_coords_read_only(self):
        """Mutating target_coords raises ValueError."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        with pytest.raises(ValueError):
            imap.target_coords[0] = 5.0

    def test_conversion_unaffected_by_input_array_mutation(self):
        """Mutating the caller's input array after construction has no effect."""
        source = np.array([0.0, 100.0])
        target = np.array([10.0, 110.0])
        imap = InterpolationMap(
            source_coords=source,
            target_coords=target,
            source_id="child",
            target_id="parent",
        )
        source[0] = 999.0
        target[0] = 999.0
        assert imap(50.0) == 60.0

    def test_inverse_coords_read_only(self):
        """The cached inverse's coordinate arrays are also read-only."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        inv = imap.inverse()
        with pytest.raises(ValueError):
            inv.source_coords[0] = 5.0
        with pytest.raises(ValueError):
            inv.target_coords[0] = 5.0

    def test_inverse_cache_identity_unaffected(self):
        """Since arrays cannot be mutated, the cached inverse identity holds."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child",
            target_id="parent",
        )
        inv = imap.inverse()
        assert imap(50.0) == 60.0
        assert inv(60.0) == 50.0
        assert inv.inverse() is imap


class TestMatchesSelector:
    """Tests for ConversionMap.matches_selector and its InterpolationMap override."""

    def test_linear_map_matches_id(self):
        """A LinearMap matches its own id."""
        lm = LinearMap(scalar=2.0, source_unit="quarters", target_unit="seconds")
        assert lm.matches_selector(lm.id)

    def test_linear_map_matches_name(self):
        """A LinearMap matches its own name."""
        lm = LinearMap(
            scalar=2.0, source_unit="quarters", target_unit="seconds", name="q2s"
        )
        assert lm.matches_selector("q2s")

    def test_linear_map_negative(self):
        """A LinearMap does not match an unrelated selector."""
        lm = LinearMap(
            scalar=2.0, source_unit="quarters", target_unit="seconds", name="q2s"
        )
        assert not lm.matches_selector("nope")

    def test_interpolation_map_matches_source_id(self):
        """An InterpolationMap matches its source_id."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child:1",
            target_id="parent:1",
        )
        assert imap.matches_selector("child:1")

    def test_interpolation_map_matches_id_and_name(self):
        """An InterpolationMap still matches its own id/name too."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child:1",
            target_id="parent:1",
        )
        assert imap.matches_selector(imap.id)

    def test_interpolation_map_negative(self):
        """An InterpolationMap does not match an unrelated selector."""
        imap = InterpolationMap(
            source_coords=np.array([0.0, 100.0]),
            target_coords=np.array([10.0, 110.0]),
            source_id="child:1",
            target_id="parent:1",
        )
        assert not imap.matches_selector("nope")
