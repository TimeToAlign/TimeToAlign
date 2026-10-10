"""Tests for meter-aware maps (MetricMap, MetricalPositionMap).

This module validates ``MetricMap.from_verovio_timemap`` — the factory
that builds a measure-boundary map from a Verovio timemap JSON file — and
the ``MetricalPositionMap`` reverse lookup it feeds.

All counts and coordinates are exact per the Zero Tolerance Validation
Policy.  Validation logic and expected values are documented in
``tests/maps/README.md``.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from timetoalign.maps.base import ConversionMap
from timetoalign.maps.meter import BeatInMeasureMap, MetricalPositionMap, MetricMap
from timetoalign.testdata import ensure_data

SPECIMEN_DIR = ensure_data("performance_precision")
TIMEMAP_PATH = SPECIMEN_DIR / "Chopin Nocturne Op. 9 No. 2.json"


# region MetricMap.from_verovio_timemap


class TestFromVerovioTimemap:
    """Validate the Verovio-timemap factory against the Chopin specimen."""

    def test_n_measures(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm.n_measures == 38

    def test_total_length(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm.total_length == Fraction(425, 2)

    def test_measure_starts(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm._starts_frac[0] == Fraction(0)
        assert mm._starts_frac[1] == Fraction(1, 2)
        assert mm._starts_frac[2] == Fraction(13, 2)
        assert mm._starts_frac[37] == Fraction(413, 2)

    def test_measure_number_labels(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm._mns[0] == "1"
        assert mm._mns[-1] == "38"

    def test_measure_counts(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert list(mm._mcs) == list(range(1, 39))

    def test_last_measure_length(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm._lengths_frac[-1] == Fraction(6)

    def test_starts_are_fractions(self) -> None:
        # Fraction(str(qstamp)) keeps 0.5 exact as 1/2 (not a float artifact).
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert all(isinstance(s, Fraction) for s in mm._starts_frac)

    def test_no_boundaries_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "no_measures.json"
        empty.write_text(json.dumps([{"qstamp": 0.0}, {"qstamp": 4.0}]))
        with pytest.raises(ValueError, match="no measure boundaries"):
            MetricMap.from_verovio_timemap(empty)

    def test_accepts_str_path(self) -> None:
        mm = MetricMap.from_verovio_timemap(str(TIMEMAP_PATH))
        assert mm.n_measures == 38


# endregion


# region MetricMap.quarters_at


class TestMetricMapQuartersAt:
    """Validate the MC -> downbeat quarters reverse lookup on MetricMap itself."""

    def test_quarters_at_downbeats(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        assert mm.quarters_at(1) == Fraction(0)
        assert mm.quarters_at(2) == Fraction(1, 2)
        assert mm.quarters_at(3) == Fraction(13, 2)

    def test_unknown_mc_raises(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        with pytest.raises(ValueError, match="not found"):
            mm.quarters_at(999)


# endregion


# region MetricalPositionMap reverse lookup


class TestMetricalPositionMap:
    """Validate quarters_at / mn_at on a map built from the timemap."""

    def test_quarters_at_downbeats(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        pos = MetricalPositionMap(mm)
        # MC 1 downbeat is the first boundary (quarter 0).
        assert pos.quarters_at(1) == Fraction(0)
        # MC 2 downbeat is the second boundary (quarter 0.5).
        assert pos.quarters_at(2) == Fraction(1, 2)
        # MC 3 downbeat is the third boundary (quarter 6.5).
        assert pos.quarters_at(3) == Fraction(13, 2)

    def test_quarters_at_beat_offset(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        pos = MetricalPositionMap(mm)
        # MC 3 starts at 6.5; beat 2 is one quarter later.
        assert pos.quarters_at(3, beat=Fraction(2, 1)) == Fraction(15, 2)

    def test_mn_at(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        pos = MetricalPositionMap(mm)
        # A position inside MC 3 (6.5 .. 12.5) maps to measure label "3".
        assert pos.mn_at(Fraction(7)) == "3"

    def test_unknown_mc_raises(self) -> None:
        mm = MetricMap.from_verovio_timemap(TIMEMAP_PATH)
        pos = MetricalPositionMap(mm)
        with pytest.raises(ValueError, match="not found"):
            pos.quarters_at(999)


# endregion


# region ConversionMap.from_dict registry dispatch


class TestRegistryDispatch:
    """Validate that meter maps round-trip through the shared registry."""

    def test_metric_map_round_trip(self) -> None:
        mm = MetricMap.from_uniform(4, Fraction(4, 1))
        restored = ConversionMap.from_dict(mm.to_dict())
        assert isinstance(restored, MetricMap)
        assert restored(0.0) == 1
        assert restored(4.0) == 2
        assert restored.n_measures == 4

    def test_beat_in_measure_map_round_trip(self) -> None:
        mm = MetricMap.from_uniform(4, Fraction(4, 1))
        beat_map = BeatInMeasureMap(mm)
        restored = ConversionMap.from_dict(beat_map.to_dict())
        assert isinstance(restored, BeatInMeasureMap)
        assert restored(0) == Fraction(1, 1)
        assert restored(Fraction(3, 2)) == Fraction(5, 2)

    def test_metrical_position_map_round_trip(self) -> None:
        mm = MetricMap.from_uniform(10, Fraction(4, 1))
        pos = MetricalPositionMap(mm)
        restored = ConversionMap.from_dict(pos.to_dict())
        assert isinstance(restored, MetricalPositionMap)
        assert restored(7.5) == {"mc": 2, "beat": Fraction(9, 2)}
        assert restored.quarters_at(2, beat=Fraction(9, 2)) == Fraction(15, 2)

    def test_metrical_position_map_dictionary_round_trip_is_exact(self) -> None:
        source = MetricalPositionMap(
            MetricMap.from_uniform(2, Fraction(4, 1))
        ).to_dict()
        restored = MetricalPositionMap.from_dict(source).to_dict()
        assert restored == source

    def test_metrical_position_map_to_dict_has_no_map_type_key(self) -> None:
        mm = MetricMap.from_uniform(4, Fraction(4, 1))
        pos = MetricalPositionMap(mm)
        assert "map_type" not in pos.to_dict()
        assert pos.to_dict()["type"] == "MetricalPositionMap"


# endregion
