"""Projections added to a timeline through ``add_events``.

Strategy: ``tests/timelines/README.md``, section ``test_projections.py``.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.core import (
    Coordinate,
    Duration,
    EventReference,
    Projection,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.maps import LinearMap
from timetoalign.timelines import ContinuousLogicalTimeline, DiscretePhysicalTimeline


def _samples() -> DiscretePhysicalTimeline:
    """A samples timeline (int) converting into seconds at 44.1 kHz."""
    timeline = DiscretePhysicalTimeline(length=1_000_000, uid="dpt_1")
    timeline.add_conversion_map(
        LinearMap(
            source_unit="samples",
            target_unit="seconds",
            scalar=Fraction(1, 44100),
            uid="s2s",
        )
    )
    return timeline


def _quarters() -> ContinuousLogicalTimeline:
    """A quarters timeline (fraction) converting exactly into seconds."""
    timeline = ContinuousLogicalTimeline(length=Fraction(8), uid="clt_6")
    timeline.add_conversion_map(
        LinearMap(
            source_unit="quarters",
            target_unit="seconds",
            scalar=Fraction(1, 2),
            output_number_type="fraction",
            uid="q2s",
        )
    )
    return timeline


def _row(timeline, event_id: str) -> dict:
    """The stored row of *event_id*, cells as stored."""
    rows = [row for row in timeline.events.table.to_pylist() if row["id"] == event_id]
    assert len(rows) == 1
    return rows[0]


class TestPositionsOnTheTimelineAxis:
    def test_instant_in_seconds_is_stored_in_samples(self) -> None:
        timeline = _samples()
        ids = timeline.add_events(
            [
                {
                    "event_type": "Projection",
                    "id": "half",
                    "start": Coordinate(Fraction(1, 2), TimeUnit.seconds),
                },
                {
                    "event_type": "Projection",
                    "id": "layerit",
                    "start": Coordinate(19.278367347, TimeUnit.seconds),
                },
            ]
        )
        assert ids == ["half", "layerit"]
        half = _row(timeline, "half")
        assert half["event_type"] == "Projection"
        assert half["temporal_type"] == "instant"
        assert half["start"] == {"value": 22050.0, "numerator": 22050, "denominator": 1}
        assert half["end"] is None
        assert half["duration"] is None
        assert _row(timeline, "layerit")["start"] == {
            "value": 850176.0,
            "numerator": 850176,
            "denominator": 1,
        }

    def test_instant_in_seconds_is_stored_exactly_in_quarters(self) -> None:
        timeline = _quarters()
        timeline.add_events(
            [
                {
                    "event_type": "Projection",
                    "id": "p",
                    "start": Coordinate(Fraction(3, 4), TimeUnit.seconds),
                }
            ]
        )
        assert _row(timeline, "p")["start"] == {
            "value": 1.5,
            "numerator": 3,
            "denominator": 2,
        }
        assert timeline.events.column_values("start") == [Fraction(3, 2)]

    def test_interval_in_seconds_is_stored_in_samples(self) -> None:
        timeline = _samples()
        timeline.add_events(
            [
                {
                    "event_type": "Projection",
                    "id": "p",
                    "start": Coordinate(Fraction(1), TimeUnit.seconds),
                    "duration": Duration(Fraction(1, 2), TimeUnit.seconds),
                }
            ]
        )
        row = _row(timeline, "p")
        assert row["temporal_type"] == "interval"
        assert row["start"] == {"value": 44100.0, "numerator": 44100, "denominator": 1}
        assert row["duration"] == {
            "value": 22050.0,
            "numerator": 22050,
            "denominator": 1,
        }
        assert row["end"] == {"value": 66150.0, "numerator": 66150, "denominator": 1}


class TestTheReferenceIsAnObject:
    def test_dict_reference_and_none(self) -> None:
        timeline = _samples()
        timeline.add_events(
            [
                {
                    "event_type": "Projection",
                    "id": "a",
                    "start": 10,
                    "projection_of": {"timeline_id": "clt_6", "event_id": "x1azmn68"},
                },
                {
                    "event_type": "Projection",
                    "id": "b",
                    "start": 20,
                    "projection_of": EventReference(timeline_id="clt_6", event_id="x2"),
                },
                {"event_type": "Projection", "id": "c", "start": 30},
            ]
        )
        column = timeline.events.schema.field("projection_of")
        assert str(column.type) == "struct<timeline_id: string, event_id: string>"
        assert field_metadata(column)["field_type"] == "EventReferenceField"
        assert timeline.events.column_values("projection_of") == [
            {"timeline_id": "clt_6", "event_id": "x1azmn68"},
            {"timeline_id": "clt_6", "event_id": "x2"},
            None,
        ]

    def test_batch_without_references_types_the_column(self) -> None:
        timeline = _samples()
        timeline.add_events([{"event_type": "Projection", "id": "a", "start": 10}])
        timeline.add_events(
            [
                {
                    "event_type": "Projection",
                    "id": "b",
                    "start": 20,
                    "projection_of": {"timeline_id": "clt_6", "event_id": "x1azmn68"},
                }
            ]
        )
        assert timeline.events.column_values("projection_of") == [
            None,
            {"timeline_id": "clt_6", "event_id": "x1azmn68"},
        ]


class TestIds:
    def test_a_source_id_is_kept(self) -> None:
        timeline = _samples()
        assert timeline.add_events(
            [{"event_type": "Projection", "id": "x1azmn68", "start": 5}]
        ) == ["x1azmn68"]

    def test_minted_ids_follow_the_id_scheme(self) -> None:
        timeline = _samples()
        assert timeline.add_events(
            [
                {"event_type": "Projection", "start": 5},
                {"event_type": "Projection", "start": 6},
            ]
        ) == ["projection_1", "projection_2"]

    def test_minted_ids_continue_above_every_event_type(self) -> None:
        timeline = _samples()
        timeline.add_events([{"event_type": "Note", "id": "projection_2", "start": 1}])
        assert timeline.add_events(
            [
                {"event_type": "Projection", "start": 5},
                {"event_type": "Projection", "start": 6},
            ]
        ) == ["projection_3", "projection_4"]


class TestProjectionObjects:
    def test_an_instant_without_id(self) -> None:
        timeline = _samples()
        ids = timeline.add_events(
            Projection(
                start=Coordinate(Fraction(1, 2), TimeUnit.seconds),
                projection_of=EventReference(timeline_id="clt_6", event_id="x1azmn68"),
            )
        )
        assert ids == ["projection_1"]
        row = _row(timeline, "projection_1")
        assert row["event_type"] == "Projection"
        assert row["start"] == {"value": 22050.0, "numerator": 22050, "denominator": 1}
        assert row["duration"] is None
        assert row["projection_of"] == {"timeline_id": "clt_6", "event_id": "x1azmn68"}

    def test_an_interval_beside_a_dict(self) -> None:
        timeline = _samples()
        ids = timeline.add_events(
            [
                Projection(
                    id="given",
                    start=Coordinate(1.0, TimeUnit.seconds),
                    duration=Duration(0.5, TimeUnit.seconds),
                ),
                {"event_type": "Note", "start": 3},
            ]
        )
        assert ids == ["given", "note_1"]
        row = _row(timeline, "given")
        assert row["start"] == {"value": 44100.0, "numerator": 44100, "denominator": 1}
        assert row["duration"] == {
            "value": 22050.0,
            "numerator": 22050,
            "denominator": 1,
        }
        assert row["projection_of"] is None


class TestRefusals:
    def test_an_id_held_by_another_event_type(self) -> None:
        timeline = _samples()
        timeline.add_events([{"event_type": "Note", "id": "n1", "start": 1}])
        with pytest.raises(ValueError) as refused:
            timeline.add_events([{"event_type": "Projection", "id": "n1", "start": 2}])
        assert str(refused.value) == (
            "Event id 'n1' is already held by an event of timeline 'dpt_1'"
        )
        assert len(timeline.events) == 1

    def test_a_property_the_projection_does_not_have(self) -> None:
        timeline = _samples()
        with pytest.raises(ValueError) as refused:
            timeline.add_events(
                [{"event_type": "Projection", "id": "p", "start": 2, "pitch": 60}]
            )
        assert str(refused.value) == (
            "A Projection holds its ID, its position and projection_of, nothing "
            "else of the event it stands for; the row also gives ['pitch']"
        )
        assert len(timeline.events) == 0

    def test_a_joined_string_is_not_a_reference(self) -> None:
        timeline = _samples()
        with pytest.raises(ValueError) as refused:
            timeline.add_events(
                [
                    {
                        "event_type": "Projection",
                        "id": "p",
                        "start": 2,
                        "projection_of": "clt_6:x1azmn68",
                    }
                ]
            )
        assert str(refused.value) == (
            "A Projection's projection_of is {'timeline_id': ..., 'event_id': ...} "
            "or None, not 'clt_6:x1azmn68'"
        )
        assert len(timeline.events) == 0

    def test_a_reference_to_its_own_timeline(self) -> None:
        timeline = _samples()
        with pytest.raises(ValueError) as refused:
            timeline.add_events(
                Projection(
                    start=Coordinate(2, TimeUnit.samples),
                    projection_of=EventReference(timeline_id="dpt_1", event_id="n1"),
                )
            )
        assert str(refused.value) == (
            "A Projection added to timeline 'dpt_1' stands for an event of another "
            "timeline; its projection_of names 'dpt_1' itself"
        )
        assert len(timeline.events) == 0
