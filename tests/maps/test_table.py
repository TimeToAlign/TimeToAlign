"""Tests for table-based maps."""

import json
import re
from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core.enums import ExtrapolationPolicy, InterpolationKind, NumberType
from timetoalign.maps.base import ConversionMap
from timetoalign.maps.table import TableMap


class TestTableMap:
    def test_initialization(self):
        m = TableMap(x_values=[0, 10], y_values=[0, 20])
        assert m.x_min == 0
        assert m.x_max == 10
        assert m.kind == InterpolationKind.linear

        with pytest.raises(ValueError, match="same length"):
            TableMap(x_values=[0, 10], y_values=[0])

        with pytest.raises(ValueError, match="monotonically increasing"):
            TableMap(x_values=[10, 0], y_values=[0, 20])

        with pytest.raises(ValueError, match="at least 2"):
            TableMap(x_values=[0], y_values=[0])

    def test_linear_interpolation(self):
        m = TableMap(x_values=[0, 10, 20], y_values=[0, 100, 150])

        assert m(0) == 0.0
        assert m(10) == 100.0
        assert m(20) == 150.0

        # Midpoints
        assert m(5) == 50.0  # (0,0) to (10,100) -> slope 10
        assert m(15) == 125.0  # (10,100) to (20,150) -> slope 5

    def test_extrapolation_policies(self):
        # Default: extrapolate
        m = TableMap(x_values=[0, 10], y_values=[0, 20])
        assert m(20) == 40.0
        assert m(-10) == -20.0

        # Constant (clamp)
        m_const = TableMap(
            x_values=[0, 10], y_values=[0, 20], extrapolate=ExtrapolationPolicy.constant
        )
        assert m_const(20) == 20.0
        assert m_const(-10) == 0.0

        # Error
        m_err = TableMap(
            x_values=[0, 10], y_values=[0, 20], extrapolate=ExtrapolationPolicy.error
        )
        with pytest.raises(ValueError, match="outside table bounds"):
            m_err(20)

    def test_other_interpolations(self):
        x = [0, 10]
        y = [0, 20]

        # Nearest
        m_near = TableMap(x_values=x, y_values=y, kind=InterpolationKind.nearest)
        assert m_near(4) == 0.0  # Closer to 0
        assert m_near(6) == 20.0  # Closer to 10

        # Previous (Step Left)
        m_prev = TableMap(x_values=x, y_values=y, kind=InterpolationKind.previous)
        assert m_prev(9.9) == 0.0
        assert m_prev(10) == 20.0

        # Next (Step Right)
        m_next = TableMap(x_values=x, y_values=y, kind=InterpolationKind.next)
        assert m_next(0.1) == 20.0
        assert m_next(0) == 0.0

    def test_inverse(self):
        m = TableMap(x_values=[0, 10, 20], y_values=[0, 100, 300])
        assert m.is_invertible

        inv = m.inverse()
        assert np.array_equal(inv.x_values, [0, 100, 300])
        assert np.array_equal(inv.y_values, [0, 10, 20])

        # Round trip
        val = 15.0
        mapped = m(val)  # 10->100, 20->300. Slope 20. 15 -> 100 + 5*20 = 200
        assert mapped == 200.0
        assert inv(mapped) == 15.0

    def test_inverse_decreasing(self):
        # y values decreasing
        m = TableMap(x_values=[0, 10], y_values=[100, 0])
        assert m.is_invertible

        inv = m.inverse()
        # Should reverse so x is increasing
        assert np.array_equal(inv.x_values, [0, 100])
        assert np.array_equal(inv.y_values, [10, 0])

        assert m(5) == 50.0
        assert inv(50) == 5.0

    def test_from_tempo(self):
        # 120 BPM = 0.5 sec/beat = 0.5 sec/480 ticks = 1/960 sec/tick
        # 60 BPM = 1.0 sec/beat = 1.0 sec/480 ticks = 1/480 sec/tick

        m = TableMap.from_tempo_changes(
            tick_positions=[0, 960], tempos_bpm=[120.0, 60.0], ticks_per_quarter=480
        )

        # 0 -> 0 sec
        assert m(0) == 0.0

        # 480 ticks (1 quarter) @ 120 bpm -> 0.5 sec
        assert m(480) == 0.5

        # 960 ticks (2 quarters) @ 120 bpm -> 1.0 sec
        assert m(960) == 1.0

        # 960 + 480 ticks (1 more quarter) @ 60 bpm -> 1.0 + 1.0 = 2.0 sec
        assert m(1440) == 2.0


class TestTableMapConvertArray:
    """Tests for TableMap.convert_array() public API."""

    def test_convert_array_linear_interpolation(self):
        """convert_array() with linear interpolation."""
        m = TableMap(x_values=[0, 10, 20], y_values=[0, 100, 150])
        values = np.array([0, 5, 10, 15, 20])

        result = m.convert_array(values)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0.0, 50.0, 100.0, 125.0, 150.0])

    def test_convert_array_from_tempo(self):
        """convert_array() with tempo-based table map."""
        m = TableMap.from_tempo_changes(
            tick_positions=[0, 960], tempos_bpm=[120.0, 60.0], ticks_per_quarter=480
        )
        values = np.array([0, 480, 960, 1440])

        result = m.convert_array(values)

        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0, 2.0])

    def test_convert_array_matches_scalar(self):
        """convert_array() results match element-wise scalar conversion."""
        m = TableMap(x_values=[0, 5, 10, 15, 20], y_values=[0, 2, 8, 12, 20])
        values = np.array([0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20])

        array_result = m.convert_array(values)
        scalar_results = np.array([m(v) for v in values])

        np.testing.assert_array_almost_equal(array_result, scalar_results)

    def test_convert_array_extrapolation(self):
        """convert_array() respects extrapolation policy."""
        m = TableMap(
            x_values=[0, 10],
            y_values=[0, 20],
            extrapolate=ExtrapolationPolicy.constant,
        )
        values = np.array([-5, 0, 5, 10, 15])

        result = m.convert_array(values)

        np.testing.assert_array_equal(result, [0.0, 0.0, 10.0, 20.0, 20.0])


class TestTableMapExact:
    """Exact operands interpolate and extrapolate exactly; any float keeps the float path."""

    @staticmethod
    def _exact_map(**kwargs) -> TableMap:
        return TableMap(
            x_values=[Fraction(0), Fraction(4)],
            y_values=[Fraction(0), Fraction(2)],
            **kwargs,
        )

    @staticmethod
    def _assert_fraction(result, expected: Fraction) -> None:
        assert type(result) is Fraction
        assert result == expected

    def test_linear_interpolation_is_exact(self):
        self._assert_fraction(self._exact_map()(Fraction(1, 3)), Fraction(1, 6))

    def test_linear_extrapolation_is_exact(self):
        m = self._exact_map(extrapolate="extrapolate")
        self._assert_fraction(m(5), Fraction(5, 2))

    def test_segments_and_both_ends(self):
        m = TableMap(x_values=[0, 3, 7], y_values=[0, 1, 2])
        self._assert_fraction(m(1), Fraction(1, 3))
        self._assert_fraction(m(6), Fraction(7, 4))
        self._assert_fraction(m(-3), Fraction(-1))
        self._assert_fraction(m(10), Fraction(11, 4))
        self._assert_fraction(m(3), Fraction(1))

    def test_fraction_axis_keeps_the_exact_result(self):
        m = self._exact_map(source_unit="quarters", target_unit="quarters")
        self._assert_fraction(m(Fraction(1, 3)), Fraction(1, 6))

    def test_float_anchors_stay_float(self):
        m = TableMap(x_values=[0.0, 4.0], y_values=[0.0, 2.0])
        result = m(Fraction(1, 3))
        assert type(result) is float
        assert result == 0.16666666666666666

    def test_float_input_stays_float(self):
        result = self._exact_map()(0.5)
        assert type(result) is float
        assert result == 0.25

    def test_one_float_anchor_keeps_the_float_path(self):
        m = TableMap(x_values=[0, 4], y_values=[0, 2.0])
        result = m(Fraction(1, 3))
        assert type(result) is float
        assert result == 0.16666666666666666

    def test_array_lane_stays_float64(self):
        result = self._exact_map().convert_array(np.array([0, 1, 2, 4]))
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [0.0, 0.5, 1.0, 2.0])


def exact_tempo_map(**kwargs) -> TableMap:
    """Quarters to seconds through exact anchors: 1/6 s per quarter, then 1/3."""
    return TableMap(
        x_values=[0, 2, 4],
        y_values=[Fraction(0), Fraction(1, 3), Fraction(1)],
        source_unit="quarters",
        target_unit="seconds",
        **kwargs,
    )


class TestDeclaredOutputNumberType:
    def test_undeclared_map_writes_the_target_units_default(self):
        cmap = exact_tempo_map()
        assert cmap.output_number_type is NumberType.float
        assert cmap(1) == 1 / 6
        assert type(cmap(1)) is float

    def test_fraction_map_answers_exactly(self):
        cmap = exact_tempo_map(output_number_type="fraction")
        assert cmap.output_number_type is NumberType.fraction
        # at an anchor, between anchors, and past the last anchor
        for position, expected in [
            (2, Fraction(1, 3)),
            (1, Fraction(1, 6)),
            (Fraction(3), Fraction(2, 3)),
            (5, Fraction(4, 3)),
        ]:
            result = cmap(position)
            assert result == expected
            assert type(result) is Fraction

    def test_fraction_map_never_answers_a_float(self):
        result = exact_tempo_map(output_number_type="fraction")(1.0)
        # the float path's 1/6, re-expressed as its exact dyadic
        assert result == Fraction(1 / 6)
        assert type(result) is Fraction

    def test_float_map_never_answers_a_fraction(self):
        cmap = TableMap(
            x_values=[0, 2],
            y_values=[Fraction(0), Fraction(1, 3)],
            source_unit="seconds",
            target_unit="quarters",
            output_number_type=NumberType.float,
        )
        assert cmap(1) == 1 / 6
        assert type(cmap(1)) is float

    def test_inverse_answers_exactly(self):
        inverse = exact_tempo_map(output_number_type="fraction").inverse()
        assert inverse.output_number_type is NumberType.fraction
        result = inverse(Fraction(2, 3))
        assert result == Fraction(3)
        assert type(result) is Fraction

    def test_inverse_into_a_discrete_unit_writes_its_integers(self):
        cmap = TableMap(
            x_values=[0, 960],
            y_values=[Fraction(0), Fraction(1, 3)],
            source_unit="ticks",
            target_unit="seconds",
            output_number_type="fraction",
        )
        inverse = cmap.inverse()
        assert inverse.output_number_type is NumberType.int
        assert inverse(Fraction(1, 6)) == 480

    def test_round_trip_drops_the_declaration(self):
        # The manifest states no number type for a map, so the declaration
        # is not written and the map read back answers as its target unit
        # does by default.
        cmap = exact_tempo_map(output_number_type="fraction", uid="tempo")
        data = cmap.to_dict(minimal=False)
        assert data == exact_tempo_map(uid="tempo").to_dict(minimal=False)
        restored = ConversionMap.from_dict(json.loads(json.dumps(data)))
        assert restored == exact_tempo_map(uid="tempo")
        assert restored != cmap
        assert restored.output_number_type is NumberType.float
        assert restored(1) == 1 / 6
        assert type(restored(1)) is float

    def test_target_unit_must_admit_the_declaration(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Unit 'ticks' does not admit number_type 'float'. "
                "Allowed number types: int"
            ),
        ):
            TableMap(
                x_values=[0, 1],
                y_values=[0, 480],
                source_unit="quarters",
                target_unit="ticks",
                output_number_type="float",
            )
