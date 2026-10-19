"""Tests for interval-to-constant and quarters-to-measures maps."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core import Measure, MeasureConstituent
from timetoalign.core.enums import InterpolationKind
from timetoalign.maps.interval import (
    IntervalToConstantMap,
    QuartersToFloatingMeasures,
    QuartersToMeasureNumber,
)
from timetoalign.maps.meter import MetricMap
from timetoalign.timelines import MeasureMap


class TestIntervalToConstantMap:
    """Tests for IntervalToConstantMap base class."""

    def test_initialization(self):
        """Basic initialization with string values."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8, 12],
            values=["Intro", "Verse", "Chorus", "Outro"],
        )
        assert len(m.boundaries) == 4
        assert len(m.values) == 4
        assert m.is_invertible is False

    def test_initialization_validation(self):
        """Validation of input parameters."""
        # Different lengths
        with pytest.raises(ValueError, match="same length"):
            IntervalToConstantMap(boundaries=[0, 4, 8], values=["A", "B"])

        # Empty boundaries
        with pytest.raises(ValueError, match="at least 1"):
            IntervalToConstantMap(boundaries=[], values=[])

        # Non-monotonic boundaries
        with pytest.raises(ValueError, match="monotonically increasing"):
            IntervalToConstantMap(boundaries=[4, 2, 8], values=["A", "B", "C"])

    def test_scalar_lookup(self):
        """Lookup returns correct value for each interval."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8, 12],
            values=["A", "B", "C", "D"],
        )

        # At boundaries
        assert m(0) == "A"
        assert m(4) == "B"
        assert m(8) == "C"
        assert m(12) == "D"

        # Within intervals
        assert m(2) == "A"
        assert m(6) == "B"
        assert m(10) == "C"
        assert m(14) == "D"  # Beyond last boundary

    def test_scalar_lookup_edge_cases(self):
        """Edge cases: before first boundary, exactly at boundaries."""
        m = IntervalToConstantMap(
            boundaries=[4, 8, 12],
            values=["A", "B", "C"],
        )

        # Before first boundary
        assert m(0) == "A"
        assert m(3.99) == "A"

        # At boundaries (left-inclusive)
        assert m(4) == "A"
        assert m(7.99) == "A"
        assert m(8) == "B"

    def test_array_lookup(self):
        """Vectorized lookup."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8],
            values=["A", "B", "C"],
        )
        values = np.array([0, 2, 4, 6, 8, 10])

        result = m(values)

        assert list(result) == ["A", "A", "B", "B", "C", "C"]

    def test_numeric_values(self):
        """IntervalToConstantMap with numeric values."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8],
            values=[10, 20, 30],
        )

        assert m(2) == 10
        assert m(6) == 20
        assert m(10) == 30

    def test_to_table_map(self):
        """Convert to TableMap for numeric values."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8],
            values=[1, 2, 3],
        )

        table_map = m.to_table_map(kind=InterpolationKind.previous)

        # Should produce step function
        assert table_map(0) == 1
        assert table_map(2) == 1
        assert table_map(4) == 2
        assert table_map(6) == 2

    def test_to_table_map_non_numeric_raises(self):
        """to_table_map raises TypeError for non-numeric values."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8],
            values=["A", "B", "C"],
        )

        with pytest.raises(TypeError, match="numeric values"):
            m.to_table_map()

    def test_inverse_raises(self):
        """inverse() raises NotImplementedError."""
        m = IntervalToConstantMap(
            boundaries=[0, 4, 8],
            values=["A", "B", "C"],
        )

        with pytest.raises(NotImplementedError):
            m.inverse()


class TestQuartersToMeasureNumber:
    """Tests for QuartersToMeasureNumber."""

    def test_initialization(self):
        """Basic initialization."""
        m = QuartersToMeasureNumber(
            boundaries=[0, 4, 8, 12],
            mns=["1", "2", "3", "4"],
        )
        assert m.mns == ["1", "2", "3", "4"]
        assert m.source_unit.value == "quarters"
        assert m.target_unit is None  # Not a coordinate map - returns labels

    def test_lookup_standard_measures(self):
        """Lookup with standard 4/4 measures."""
        m = QuartersToMeasureNumber(
            boundaries=[0, 4, 8, 12],
            mns=["1", "2", "3", "4"],
        )

        assert m(0) == "1"
        assert m(3.5) == "1"
        assert m(4) == "2"
        assert m(7.99) == "2"
        assert m(8) == "3"

    def test_lookup_anacrusis(self):
        """Lookup with anacrusis (pickup measure)."""
        # 1-beat anacrusis followed by 4/4 measures
        m = QuartersToMeasureNumber(
            boundaries=[0, 1, 5, 9],
            mns=["0", "1", "2", "3"],
        )

        assert m(0) == "0"  # In anacrusis
        assert m(0.5) == "0"
        assert m(1) == "1"  # Start of M1
        assert m(5) == "2"  # Start of M2

    def test_lookup_split_bars(self):
        """Lookup with split bars (e.g., "19a", "19b")."""
        m = QuartersToMeasureNumber(
            boundaries=[0, 2, 4],
            mns=["19a", "19b", "20"],
        )

        assert m(0) == "19a"
        assert m(1) == "19a"
        assert m(2) == "19b"
        assert m(4) == "20"

    def test_from_metric_map(self):
        """Create from MetricMap."""
        meter = MetricMap.from_uniform(
            n_measures=4,
            quarters_per_measure=Fraction(4, 1),
            start_mc=1,
            start_mn="1",
        )

        m = QuartersToMeasureNumber.from_metric_map(meter)

        assert m(0) == "1"
        assert m(4) == "2"
        assert m(8) == "3"
        assert m(12) == "4"

    def test_from_metric_map_anacrusis(self):
        """Create from MetricMap with anacrusis."""
        meter = MetricMap.from_uniform(
            n_measures=4,
            quarters_per_measure=Fraction(4, 1),
            anacrusis_quarters=Fraction(1, 1),
            start_mc=1,
            start_mn="0",
        )

        m = QuartersToMeasureNumber.from_metric_map(meter)

        assert m(0) == "0"
        assert m(1) == "1"
        assert m(5) == "2"


class TestQuartersToFloatingMeasures:
    """Tests for QuartersToFloatingMeasures."""

    def test_initialization(self):
        """Basic initialization."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert m.source_unit.value == "quarters"
        assert (
            m.target_unit.value == "floating_measures"
        )  # Returns coordinates in measures

    def test_linear_interpolation(self):
        """Interpolation within measures."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        # At boundaries
        assert m(0) == 1.0
        assert m(4) == 2.0
        assert m(8) == 3.0

        # Within measures
        assert m(2) == 1.5  # Halfway through M1
        assert m(6) == 2.5  # Halfway through M2
        assert m(10) == 3.5

    def test_irregular_measures(self):
        """Interpolation with irregular measure lengths."""
        # M1: 4 quarters, M2: 3 quarters (3/4 time), M3: 4 quarters
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 7, 11, 15],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        assert m(0) == 1.0
        assert m(2) == 1.5  # Halfway through M1 (4 quarters)
        assert m(4) == 2.0
        assert m(5.5) == 2.5  # Halfway through M2 (3 quarters)
        assert m(7) == 3.0

    def test_extrapolation(self):
        """Extrapolation beyond bounds."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12],
            y_values=[1.0, 2.0, 3.0, 4.0],
        )

        # Beyond last measure
        assert m(14) == 4.5  # Extrapolates linearly

        # Before first measure (negative)
        assert m(-4) == 0.0

    def test_from_measure_map(self):
        """Build the lattice from exact measure lengths."""
        measure_map = MeasureMap(
            Measure(
                id=f"m{count}",
                number=count,
                nominal_length=Fraction(4),
                actual_length=Fraction(4),
            )
            for count in range(1, 5)
        )

        m = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert m(0) == 1.0
        assert m(2) == 1.5
        assert m(4) == 2.0
        assert m(8) == 3.0

    def test_from_measure_map_anacrusis(self):
        """A pickup is measure 0, and its content sits where it is notated.

        The pickup is one quarter of a four-quarter bar, so it occupies the
        LAST quarter of the nominal bar 0: its notated downbeat is a virtual
        -3 quarters, and its onset reads three quarters of the way through
        bar 0.
        """
        measure_map = MeasureMap(
            [
                MeasureConstituent(
                    id="m1",
                    number=0,
                    nominal_length=Fraction(4),
                    actual_length=Fraction(1),
                    offset_within_measure=Fraction(3),
                ),
                *[
                    Measure(
                        id=f"m{count}",
                        number=count - 1,
                        nominal_length=Fraction(4),
                        actual_length=Fraction(4),
                    )
                    for count in range(2, 5)
                ],
            ]
        )

        m = QuartersToFloatingMeasures.from_measure_map(measure_map)

        assert m(0) == 0.75
        assert m(0.5) == 0.875  # Halfway through the sounding pickup
        assert m(1) == 1.0  # Start of M1
        assert m(5) == 2.0  # Start of M2

    def test_to_measure_number_map(self):
        """Convert to QuartersToMeasureNumber."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        mn_map = m.to_measure_number_map()

        assert mn_map(0) == "1"
        assert mn_map(3) == "1"
        assert mn_map(4) == "2"
        assert mn_map(8) == "3"

    def test_to_measure_number_map_custom_mns(self):
        """Convert with custom MN labels."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        mn_map = m.to_measure_number_map(mns=["A", "B", "C", "D"])

        assert mn_map(0) == "A"
        assert mn_map(4) == "B"
        assert mn_map(8) == "C"

    def test_inverse(self):
        """Inverse map (floating measures -> quarters)."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        inv = m.inverse()

        # Round-trip
        assert inv(1.0) == 0.0
        assert inv(2.0) == 4.0
        assert inv(1.5) == 2.0
        assert inv(2.5) == 6.0

    def test_array_conversion(self):
        """Vectorized conversion."""
        m = QuartersToFloatingMeasures(
            x_values=[0, 4, 8, 12, 16],
            y_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )

        values = np.array([0, 2, 4, 6, 8])
        result = m.convert_array(values)

        np.testing.assert_array_equal(result, [1.0, 1.5, 2.0, 2.5, 3.0])


class TestScoreStoreIntegration:
    """Tests for ScoreStore.get_cmaps() integration."""

    def test_measures_in_cmaps(self):
        """ScoreStore.get_cmaps() includes measures map when measures present."""
        from timetoalign.loader.score.store import ScoreStore
        from timetoalign.loader.score.stores.measures import MeasureData

        # Create a ScoreStore with measures
        measures = MeasureData.from_dicts(
            [
                {
                    "mc": 1,
                    "mn": "1",
                    "start": 0,
                    "duration": 4,
                    "event_type": "Measure",
                },
                {
                    "mc": 2,
                    "mn": "2",
                    "start": 4,
                    "duration": 4,
                    "event_type": "Measure",
                },
                {
                    "mc": 3,
                    "mn": "3",
                    "start": 8,
                    "duration": 4,
                    "event_type": "Measure",
                },
            ]
        )

        store = ScoreStore.empty()
        store.measures = measures

        cmaps = store.get_cmaps()

        assert "measures" in cmaps
        fm = cmaps["measures"]
        assert isinstance(fm, QuartersToFloatingMeasures)

        # Verify correct interpolation
        assert fm(0) == 1.0
        assert fm(2) == 1.5
        assert fm(4) == 2.0

    def test_no_measures_when_empty(self):
        """ScoreStore.get_cmaps() does not include measures when no measure data."""
        from timetoalign.loader.score.store import ScoreStore

        store = ScoreStore.empty()

        cmaps = store.get_cmaps()

        assert "measures" not in cmaps
        assert "ticks" in cmaps  # But ticks should always be present
