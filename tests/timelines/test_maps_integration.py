"""Tests for Timeline conversion map integration."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from timetoalign.core import Coordinate, TimeUnit
from timetoalign.maps.linear import ScalarMap
from timetoalign.maps.meter import MetricMap
from timetoalign.maps.table import TableMap
from timetoalign.timelines import Timeline


class TestMapIntegration:
    def test_add_conversion_map(self):
        tl = Timeline(unit=TimeUnit.seconds)
        cmap = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")

        tl.add_conversion_map(cmap)

        # Verify it's stored (private access for testing)
        assert cmap.id in tl._conversion_maps
        assert tl._conversion_maps[cmap.id] is cmap

    def test_add_conversion_map_validation(self):
        tl = Timeline(unit=TimeUnit.seconds)
        cmap = ScalarMap(scalar=1.0, source_unit="ticks")  # Mismatch

        with pytest.raises(ValueError) as refused:
            tl.add_conversion_map(cmap)
        assert str(refused.value) == (
            "Map source unit 'ticks' does not match timeline unit 'seconds'"
        )
        assert tl.n_conversion_maps == 0

    def test_a_map_without_source_unit_attaches_to_any_unit(self):
        for unit in (TimeUnit.seconds, TimeUnit.ticks, TimeUnit.quarters):
            tl = Timeline(unit=unit)
            cmap = ScalarMap(scalar=2)
            tl.add_conversion_map(cmap)
            assert tl.get_conversion_map(cmap.id) is cmap

    def test_get_conversion_map(self):
        tl = Timeline(unit=TimeUnit.seconds)
        cmap1 = ScalarMap(
            scalar=1000, source_unit="seconds", target_unit="milliseconds"
        )
        cmap2 = ScalarMap(scalar=1 / 60, source_unit="seconds", target_unit="minutes")

        tl.add_conversion_map(cmap1)
        tl.add_conversion_map(cmap2)

        assert tl.get_conversion_map("milliseconds") is cmap1
        assert tl.get_conversion_map("minutes") is cmap2
        assert tl.get_conversion_map("ticks") is None

    def test_convert_to(self):
        tl = Timeline(unit=TimeUnit.seconds)
        cmap = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        tl.add_conversion_map(cmap)

        # Scalar returns Coordinate object
        val = tl.convert_to(1.5, "milliseconds")
        assert isinstance(val, Coordinate)
        assert val.value == 1500.0
        assert val.unit == TimeUnit.milliseconds

        # Array returns array (not Coordinate)
        arr = np.array([1.0, 2.0])
        conv = tl.convert_to(arr, "milliseconds")
        assert isinstance(conv, np.ndarray)
        np.testing.assert_array_equal(conv, np.array([1000.0, 2000.0]))

        # Error if no map
        with pytest.raises(ValueError, match="No conversion map found"):
            tl.convert_to(1.0, "ticks")

    def test_convert_coordinate_object(self):
        tl = Timeline(unit=TimeUnit.seconds)
        cmap = ScalarMap(scalar=1000, source_unit="seconds", target_unit="milliseconds")
        tl.add_conversion_map(cmap)

        # Coordinate input returns Coordinate output
        c = Coordinate(1.5, TimeUnit.seconds)
        val = tl.convert_to(c, "milliseconds")
        assert isinstance(val, Coordinate)
        assert val.value == 1500.0
        assert val.unit == TimeUnit.milliseconds

    def test_serialization_with_maps(self):
        tl = Timeline(unit=TimeUnit.seconds, uid="tl1")
        cmap = ScalarMap(
            scalar=1000, source_unit="seconds", target_unit="milliseconds", uid="cmap1"
        )
        tl.add_conversion_map(cmap)

        data = tl.to_dict()
        (timeline_data,) = data["timelines"]
        assert len(timeline_data["conversion_maps"]) == 1
        assert timeline_data["conversion_maps"][0]["id"] == "cmap1"

        # Round trip
        restored = Timeline.from_dict(data)
        assert restored.id == "tl1"
        restored_cmap = restored.get_conversion_map("milliseconds")
        assert restored_cmap is not None
        assert restored_cmap.id == "cmap1"
        assert restored_cmap(1.0) == 1000.0


class TestTableMapHonesty:
    """A TableMap attached via add_conversion_map is stored as-is: its
    interpolation kind and extrapolation policy are honored exactly at the
    TimeStamp level, not silently converted to plain linear interpolation.
    """

    def test_kind_previous_honored_at_timestamp(self):
        tl = Timeline(length=10, unit=TimeUnit.quarters)
        tmap = TableMap(
            x_values=[0, 4, 8],
            y_values=[0.0, 10.0, 20.0],
            kind="previous",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tmap)

        assert tl._unit_maps[TimeUnit.seconds] == [tmap]

        ts = tl.get_timestamp(2)
        assert ts.get_unit(TimeUnit.seconds, format="float") == 0.0

    def test_extrapolate_error_honored_at_timestamp(self):
        tl = Timeline(length=20, unit=TimeUnit.quarters)
        tmap = TableMap(
            x_values=[0, 4, 8],
            y_values=[0.0, 10.0, 20.0],
            extrapolate="error",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tmap)

        ts = tl.get_timestamp(12)
        with pytest.raises(ValueError, match="outside table bounds"):
            ts.get_unit(TimeUnit.seconds)

    def test_extrapolate_constant_honored_at_timestamp(self):
        tl = Timeline(length=20, unit=TimeUnit.quarters)
        tmap = TableMap(
            x_values=[0, 4, 8],
            y_values=[0.0, 10.0, 20.0],
            extrapolate="constant",
            source_unit=TimeUnit.quarters,
            target_unit=TimeUnit.seconds,
        )
        tl.add_conversion_map(tmap)

        ts = tl.get_timestamp(20)
        assert ts.get_unit(TimeUnit.seconds, format="float") == 20.0


class TestTimelineSerializationWithMeterMap:
    """A MetricMap attached to a Timeline round-trips through to_dict/from_dict."""

    def test_round_trip_with_metric_map(self):
        tl = Timeline(length=20, unit=TimeUnit.quarters, uid="tl1")
        meter = MetricMap.from_uniform(4, Fraction(4, 1), uid="meter1")
        tl.add_conversion_map(meter)

        data = tl.to_dict()
        restored = Timeline.from_dict(data)

        restored_meter = restored.get_conversion_map("meter1")
        assert restored_meter is not None
        assert restored_meter(4.0) == 2
