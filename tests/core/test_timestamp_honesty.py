"""Timestamp metadata and renderings describe their coordinates honestly."""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.maps import SecondsToSamples
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    TimelineGroup,
)


def _sample_timeline() -> ContinuousPhysicalTimeline:
    """Create a seconds timeline with a sample conversion map."""
    timeline = ContinuousPhysicalTimeline(length=30.0, uid="audio")
    timeline.add_conversion_map(SecondsToSamples(sample_rate=44100))
    return timeline


def test_direct_axis_query_is_not_interpolated() -> None:
    """A direct coordinate on the source axis is exact."""
    timeline = ContinuousLogicalTimeline(length=Fraction(12), uid="score")

    stamp = timeline.get_timestamp(Fraction(4))

    assert stamp.is_interpolated is False
    assert "interpolated" not in repr(stamp)


def test_group_coordinate_between_anchors_is_interpolated() -> None:
    """A coordinate transferred between group anchors remains estimated."""
    audio = ContinuousPhysicalTimeline(length=10.0, uid="audio")
    guide = ContinuousPhysicalTimeline(length=20.0, uid="guide")
    group = TimelineGroup(id="paired", timelines=[audio, guide])

    stamp = group.get_timestamp_at(2.5, "audio")

    assert stamp.is_interpolated is True
    assert "interpolated" in repr(stamp)


def test_conversion_values_are_separate_from_typed_coordinate_entries() -> None:
    """Typed dictionaries stay structural and conversions remain verbatim."""
    stamp = _sample_timeline().get_timestamp(2.5)

    assert stamp.to_dict() == {
        "audio": {
            "unit": "seconds",
            "number_type": "float",
            "value": 2.5,
        }
    }
    assert stamp.get_conversion_for("samples") == 110250.0


def test_disabled_conversion_units_are_absent_from_access_and_rendering() -> None:
    """Disabled conversion maps stay unavailable and absent from displays."""
    stamp = _sample_timeline().get_timestamp(2.5, conversion_maps=False)

    assert stamp.to_dict() == {
        "audio": {
            "unit": "seconds",
            "number_type": "float",
            "value": 2.5,
        }
    }
    with pytest.raises(KeyError):
        stamp.get_conversion_for("samples")
    assert "samples" not in repr(stamp)
    assert "samples" not in stamp._repr_html_()


def test_conversion_units_appear_in_text_and_html_renderings() -> None:
    """Notebook-facing renderings expose enabled conversion units."""
    stamp = _sample_timeline().get_timestamp(2.5)

    assert "samples=110250" in repr(stamp)
    assert "110250 samples" in stamp._repr_html_()
