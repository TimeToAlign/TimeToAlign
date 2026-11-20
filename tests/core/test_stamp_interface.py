"""Tests for the shared Stamp contract implemented by TimeStamp and MatchStamp."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.alignment.graph import MatchStamp
from timetoalign.core import Coordinate, IdCoordinate, TimeUnit
from timetoalign.core.timestamp import Stamp, TimeStamp
from timetoalign.maps import TableMap
from timetoalign.timelines import Timeline


def _timeline_with_maps() -> Timeline:
    """Build a timeline with exact-value conversion maps."""
    timeline = Timeline(length=100, unit=TimeUnit.seconds, uid="clock")
    timeline.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 100000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="clock-ms",
        )
    )
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
    bundle.add_timeline(timeline, as_group="clock-group")
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
    assert stamp.to_dict(format="prefix") == {"group_1/clock (seconds)": data}
    assert stamp.to_dict(format="nested") == {"group_1": {"clock (seconds)": data}}
    assert stamp.to_dict(format="graph") == {
        "coordinates": {"clock": data},
        "anchor_edges": [],
        "inferred_edges": [],
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
