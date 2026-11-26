"""Tests for the shared Stamp contract implemented by TimeStamp and MatchStamp."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.timestamp import Stamp, TimeStamp
from timetoalign.maps import LinearMap, TableMap
from timetoalign.timelines import Timeline, TimelineGroup


def _milliseconds(uid: str) -> TableMap:
    """A seconds-to-milliseconds map, ``[0, 100] -> [0, 100000]``."""
    return TableMap(
        x_values=[0.0, 100.0],
        y_values=[0.0, 100000.0],
        source_unit=TimeUnit.seconds,
        target_unit=TimeUnit.milliseconds,
        uid=uid,
    )


def _timeline_with_maps() -> Timeline:
    """Build a timeline with exact-value conversion maps."""
    timeline = Timeline(length=100, unit=TimeUnit.seconds, uid="clock")
    timeline.add_conversion_map(_milliseconds("clock-ms"))
    timeline.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 5000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.frames,
            uid="clock-frames",
        )
    )
    return timeline


def _matchstamp_with_maps(*, conversion_maps: bool = True) -> MatchStamp:
    """Build a bundle-backed MatchStamp with exact-value conversion maps."""
    timeline = _timeline_with_maps()
    bundle = AlignmentBundle(id="stamp-contract")
    bundle.add_group(TimelineGroup(id="clock-group", timelines=[timeline]))
    return bundle.get_matchstamp_at(
        25.0,
        "clock",
        conversion_maps=conversion_maps,
    )


def test_timestamp_implements_stamp_contract() -> None:
    """TimeStamp exposes every Stamp member with the documented types."""
    timeline = _timeline_with_maps()
    stamp = timeline.get_timestamp(25.0)

    assert isinstance(stamp, Stamp)
    assert isinstance(stamp, TimeStamp)
    assert isinstance(stamp.axis, IdCoordinate)
    assert stamp.source is timeline
    assert isinstance(stamp.source_id, str)
    assert isinstance(stamp.present_timelines, list)
    assert isinstance(stamp.is_interpolated, bool)
    assert isinstance(stamp.get_coordinate_for(timeline.id, format="float"), float)
    assert isinstance(stamp.get_unit(TimeUnit.milliseconds, format="float"), float)
    assert isinstance(stamp.to_dict(), dict)
    assert stamp._unit_for(timeline.id) == TimeUnit.seconds


def test_coordinate_and_precise_getter_contract() -> None:
    """Coordinate values retain their unit across precise getters."""
    timeline = _timeline_with_maps()
    stamp = timeline.get_timestamp(25.0)

    coordinate = stamp.get_coordinate_for(timeline.id, format="coordinate")
    assert isinstance(coordinate, Coordinate)
    assert coordinate.value == 25.0
    assert coordinate.unit == TimeUnit.seconds
    assert stamp.axis.value == 25.0
    assert stamp.axis.unit == TimeUnit.seconds

    assert stamp.get_coordinate_for(timeline.id, format="float") == 25.0
    assert stamp.get_unit(TimeUnit.milliseconds, format="float") == 25000.0
    with pytest.raises(KeyError):
        stamp.get_coordinate_for("not-a-timeline-or-unit")


def test_multi_axis_stamp_series_is_object_typed() -> None:
    """Different result axes force object dtype even when both are floats."""
    stamp = TimeStamp(
        coordinates={
            "audio": Coordinate(1.25, TimeUnit.seconds),
            "video": Coordinate(2.5, TimeUnit.seconds),
        },
        source_id="audio",
    )

    result = stamp.get_coordinates_for(["audio", "video"], format="series")

    assert result.name == "coordinate"
    assert result.index.tolist() == ["audio", "video"]
    assert result.dtype == object
    assert result.tolist() == [1.25, 2.5]


def test_conversion_maps_false_disables_unit_resolution() -> None:
    """False disables both unit conversion and its unit-name subscript path."""
    timeline = _timeline_with_maps()
    stamp = timeline.get_timestamp(25.0, conversion_maps=False)

    with pytest.raises(KeyError):
        stamp.get_unit(TimeUnit.milliseconds)


def test_conversion_maps_allowed_set_restricts_units() -> None:
    """An allowed set exposes only the named conversion maps."""
    timeline = _timeline_with_maps()
    stamp = timeline.get_timestamp(25.0, conversion_maps=["clock-ms"])

    assert stamp.get_unit(TimeUnit.milliseconds, format="float") == 25000.0
    with pytest.raises(KeyError):
        stamp.get_unit(TimeUnit.frames)


def test_to_dict_shape_is_stable() -> None:
    """to_dict returns the source coordinate and explicitly requested unit."""
    timeline = _timeline_with_maps()
    stamp = timeline.get_timestamp(25.0)

    assert stamp.to_dict() == {
        "clock": {
            "unit": "seconds",
            "number_type": "float",
            "value": 25.0,
        },
    }


def test_matchstamp_implements_stamp_contract() -> None:
    """MatchStamp exposes the shared members with exact values and types."""
    stamp = _matchstamp_with_maps()

    assert isinstance(stamp, Stamp)
    assert stamp.axis == IdCoordinate(25.0, TimeUnit.seconds, "clock")
    assert stamp.source is not None
    assert stamp.source_id == "clock"
    assert stamp.present_timelines == ["clock"]
    assert stamp.is_interpolated is True
    assert stamp.get_coordinate_for("clock", format="float") == 25.0
    assert stamp.get_unit(TimeUnit.milliseconds, format="float") == 25000.0
    assert stamp._unit_for("clock") == TimeUnit.seconds


def test_matchstamp_coordinate_and_precise_getter_contract() -> None:
    """MatchStamp retains units across timeline and unit getters."""
    stamp = _matchstamp_with_maps()

    assert stamp.get_coordinate_for("clock", format="coordinate") == Coordinate(
        25.0, TimeUnit.seconds
    )
    assert stamp.get_coordinate_for("clock", format="float") == 25.0
    assert stamp.get_unit(TimeUnit.milliseconds, format="float") == 25000.0
    with pytest.raises(KeyError):
        stamp.get_coordinate_for("not-a-timeline-or-unit")


def test_matchstamp_to_dict_formats_are_exact() -> None:
    """MatchStamp materializes every supported format exactly."""
    stamp = _matchstamp_with_maps()

    data = {
        "unit": "seconds",
        "number_type": "float",
        "value": 25.0,
    }
    assert stamp.to_dict(format="flat") == {"clock (seconds)": data}
    assert stamp.to_dict(format="prefix") == {"clock-group/clock (seconds)": data}
    assert stamp.to_dict(format="nested") == {"clock-group": {"clock (seconds)": data}}
    assert stamp.to_dict(format="graph") == {
        "coordinates": {"clock": data},
        "stated_pairs": [],
        "inferred_pairs": [],
    }


def test_matchstamp_is_frozen() -> None:
    """MatchStamp fields cannot be rebound after construction."""
    stamp = _matchstamp_with_maps()

    with pytest.raises(FrozenInstanceError):
        stamp.axis = 50.0  # type: ignore[misc]


def test_matchstamp_conversion_maps_false_disables_unit_subscript() -> None:
    """MatchStamp threads disabled conversion maps to group machinery."""
    stamp = _matchstamp_with_maps(conversion_maps=False)

    with pytest.raises(KeyError):
        stamp.get_unit(TimeUnit.milliseconds)


def test_matchstamp_str_surfaces_conversion_rows() -> None:
    """Enabled conversion maps appear as rows in __str__, with exact values."""
    stamp = _matchstamp_with_maps(conversion_maps=True)
    s = str(stamp)

    assert "milliseconds" in s
    assert "25000" in s
    assert "frames" in s
    assert "1250" in s


def test_matchstamp_str_omits_conversions_when_disabled() -> None:
    """Disabled conversion maps surface neither row while the source coordinate remains."""
    stamp = _matchstamp_with_maps(conversion_maps=False)
    s = str(stamp)

    assert "milliseconds" not in s
    assert "frames" not in s
    assert "clock" in s
    assert "25" in s


def test_matchstamp_repr_html_surfaces_conversion_rows() -> None:
    """_repr_html_ mirrors __str__'s conversion-row gating, tagged as cmap rows."""
    enabled = _matchstamp_with_maps(conversion_maps=True)._repr_html_()
    assert "milliseconds" in enabled
    assert "25000" in enabled
    assert "frames" in enabled
    assert "1250" in enabled
    assert "cmap" in enabled

    disabled = _matchstamp_with_maps(conversion_maps=False)._repr_html_()
    assert "milliseconds" not in disabled
    assert "frames" not in disabled


class TestOneSpecOneAnswer:
    """``conversion_maps`` enables maps one by one, for every reader alike."""

    @staticmethod
    def _pair() -> tuple[TimelineGroup, TableMap]:
        left = Timeline(length=100, unit=TimeUnit.seconds, uid="left")
        left_ms = _milliseconds("left-ms")
        left.add_conversion_map(left_ms)
        right = Timeline(length=100, unit=TimeUnit.seconds, uid="right")
        right.add_conversion_map(_milliseconds("right-ms"))
        return TimelineGroup(id="pair", timelines=[left, right]), left_ms

    @pytest.mark.parametrize("by", ["map", "selector"])
    def test_a_map_the_spec_leaves_out_converts_nothing(self, by: str) -> None:
        group, left_ms = self._pair()
        spec = [left_ms] if by == "map" else ["left-ms"]
        stamp = group.get_timestamp_at(10.0, "left", conversion_maps=spec)

        assert (
            stamp.get_unit(TimeUnit.milliseconds, timeline_id="left", format="float")
            == 10000.0
        )
        with pytest.raises(KeyError) as unit_refusal:
            stamp.get_unit(TimeUnit.milliseconds, timeline_id="right")
        assert unit_refusal.value.args[0] == (
            "No eligible conversion to 'milliseconds' on stamp from 'left'"
        )
        with pytest.raises(KeyError) as selector_refusal:
            stamp.get_conversion_for("right-ms")
        assert (
            selector_refusal.value.args[0] == "Unknown conversion selector 'right-ms'"
        )
        assert str(stamp) == (
            "TimeStamp @10 seconds\n"
            "  left          10 seconds\n"
            "  right         10 seconds\n"
            "  milliseconds  10000 milliseconds"
        )

    @staticmethod
    def _clock_with_two_millisecond_maps() -> Timeline:
        timeline = _timeline_with_maps()
        timeline.add_conversion_map(
            LinearMap(
                scalar=1000,
                offset=5,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.milliseconds,
                uid="clock-ms-offset",
            )
        )
        return timeline

    def test_a_spec_naming_one_of_two_maps_converts_through_it(self) -> None:
        stamp = self._clock_with_two_millisecond_maps().get_timestamp(
            25.0, conversion_maps=["clock-ms-offset"]
        )

        assert stamp.get_unit(TimeUnit.milliseconds, format="float") == 25005.0

    def test_two_enabled_maps_are_ambiguous(self) -> None:
        stamp = self._clock_with_two_millisecond_maps().get_timestamp(25.0)

        with pytest.raises(ValueError) as refusal:
            stamp.get_unit(TimeUnit.milliseconds)
        assert str(refusal.value) == (
            "Timeline 'clock' has 2 conversion maps targeting 'milliseconds'; name"
            " the one you mean. Candidates: clock-ms (seconds_to_milliseconds),"
            " clock-ms-offset (seconds_to_milliseconds)"
        )

    def test_no_enabled_map_asks_the_source_to_choose_nothing(self) -> None:
        stamp = self._clock_with_two_millisecond_maps().get_timestamp(
            25.0, conversion_maps=False
        )

        with pytest.raises(KeyError) as refusal:
            stamp.get_unit(TimeUnit.milliseconds)
        assert refusal.value.args[0] == (
            "No eligible conversion to 'milliseconds' on stamp from 'clock'"
        )
