"""Projections added to a timeline through ``add_events``.

Strategy: ``tests/timelines/README.md``, section ``test_projections.py``.
"""

from __future__ import annotations

from fractions import Fraction

import pyarrow as pa
import pytest

from timetoalign.alignment import AlignmentBundle
from timetoalign.core import (
    Coordinate,
    Duration,
    EventReference,
    Projection,
    TimeUnit,
)
from timetoalign.core.fields import field_metadata
from timetoalign.core.ids import IdMinter
from timetoalign.core.time import RATIONAL_STRUCT_TYPE
from timetoalign.maps import LinearMap
from timetoalign.storage import EventData
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
        minter = IdMinter(7)
        assert timeline.add_events(
            [
                {"event_type": "Projection", "start": 5},
                {"event_type": "Projection", "start": 6},
            ],
            seed=7,
        ) == [minter.mint("projection"), minter.mint("projection")]
        assert timeline.events.table.column("id").to_pylist() == [
            "projection_1hvif5xc2r1fv",
            "projection_0g2wfq87pa7jx",
        ]

    def test_a_minted_id_held_by_another_event_type_is_refused(self) -> None:
        timeline = _samples()
        held = IdMinter(7).mint("projection")
        timeline.add_events([{"event_type": "Note", "id": held, "start": 1}])
        with pytest.raises(ValueError) as refused:
            timeline.add_events([{"event_type": "Projection", "start": 5}], seed=7)
        assert str(refused.value) == (
            f"Event id '{held}' is already held by an event of timeline 'dpt_1'"
        )
        assert timeline.events.table.column("id").to_pylist() == [held]


class TestIdsAcrossTheManifest:
    """A Projection's ID is unique among all Projections of its manifest."""

    @staticmethod
    def _bundle() -> AlignmentBundle:
        """A bundle of ``dpt_1`` and ``clt_6``; it lives as long as the test
        holds it, since a timeline records the bundles holding it weakly."""
        bundle = AlignmentBundle()
        bundle.add_timeline(_samples())
        bundle.add_timeline(_quarters())
        return bundle

    @staticmethod
    def _shared(id: str, other: str) -> str:
        return (
            f"Projection ID '{id}' names a Projection on the timeline 'dpt_1' and "
            f"one on '{other}': a Projection's ID is unique among all Projections "
            "of a manifest and its package"
        )

    def test_minted_ids_are_distinct_across_the_bundle(self) -> None:
        bundle = self._bundle()
        samples, quarters = bundle.timelines["dpt_1"], bundle.timelines["clt_6"]
        ids = samples.add_projections(
            [
                Projection(start=Coordinate(22050, TimeUnit.samples)),
                Projection(start=Coordinate(44100, TimeUnit.samples)),
            ]
        )
        ids += quarters.add_events([{"event_type": "Projection", "start": Fraction(1)}])
        assert len(set(ids)) == 3

    def test_a_minted_id_another_timeline_of_the_bundle_holds_is_refused(
        self,
    ) -> None:
        bundle = self._bundle()
        samples, quarters = bundle.timelines["dpt_1"], bundle.timelines["clt_6"]
        (held,) = samples.add_projections(
            [Projection(start=Coordinate(22050, TimeUnit.samples))], seed=7
        )
        assert held == IdMinter(7).mint("projection")
        with pytest.raises(ValueError) as raised:
            quarters.add_events(
                [{"event_type": "Projection", "start": Fraction(1)}], seed=7
            )
        assert str(raised.value) == self._shared(held, "clt_6")
        with pytest.raises(ValueError) as raised:
            quarters.add_projections(
                [Projection(start=Coordinate(Fraction(1), TimeUnit.quarters))], seed=7
            )
        assert str(raised.value) == self._shared(held, "clt_6")
        assert len(quarters.events) == 0

    def test_a_minted_id_another_timeline_of_the_tree_holds_is_refused(self) -> None:
        samples = _samples()
        part = DiscretePhysicalTimeline(length=1000, uid="part")
        samples.add_child(part, 0)
        (held,) = samples.add_events([{"event_type": "Projection", "start": 5}], seed=7)
        with pytest.raises(ValueError) as raised:
            part.add_events([{"event_type": "Projection", "start": 5}], seed=7)
        assert str(raised.value) == self._shared(held, "part")
        assert len(part.events) == 0

    def test_a_timeline_of_no_shared_manifest_is_not_checked_against_it(
        self,
    ) -> None:
        bundle = self._bundle()
        samples = bundle.timelines["dpt_1"]
        (held,) = samples.add_events([{"event_type": "Projection", "start": 5}], seed=7)
        free = _quarters()
        assert free.add_events(
            [{"event_type": "Projection", "start": Fraction(1)}], seed=7
        ) == [held]

    def test_a_given_id_another_timeline_holds_is_refused(self) -> None:
        bundle = self._bundle()
        samples, quarters = bundle.timelines["dpt_1"], bundle.timelines["clt_6"]
        (held,) = samples.add_events([{"event_type": "Projection", "start": 5}])
        message = self._shared(held, "clt_6")
        with pytest.raises(ValueError) as raised:
            quarters.add_events(
                [{"event_type": "Projection", "id": held, "start": Fraction(1)}]
            )
        assert str(raised.value) == message
        with pytest.raises(ValueError) as raised:
            quarters.add_projections(
                Projection(id=held, start=Coordinate(Fraction(1), TimeUnit.quarters))
            )
        assert str(raised.value) == message
        assert len(quarters.events) == 0

    def test_a_child_sharing_a_projection_id_joins(self) -> None:
        parent = _samples()
        (shared,) = parent.add_events(
            [{"event_type": "Projection", "start": 5}], seed=7
        )
        part = DiscretePhysicalTimeline(length=1000, uid="part")
        # Minted in the child's own manifest from the same seed: the same ID.
        assert part.add_events([{"event_type": "Projection", "start": 7}], seed=7) == [
            shared
        ]
        parent.add_child(part, 0)
        assert parent.n_children == 1
        assert part._parent is parent
        assert part.events.table.column("id").to_pylist() == [shared]

        # The shared ID refuses nothing on creation: a missing ID is minted
        # apart from every Projection of the manifest, a free one is kept ...
        (minted,) = parent.add_events([{"event_type": "Projection", "start": 9}])
        assert minted != shared
        assert part.add_events(
            [{"event_type": "Projection", "id": "onset_a", "start": 8}]
        ) == ["onset_a"]
        # ... and one another timeline of the manifest holds is refused.
        with pytest.raises(ValueError) as raised:
            part.add_events([{"event_type": "Projection", "id": minted, "start": 9}])
        assert str(raised.value) == self._shared(minted, "part")
        assert part.events.table.column("id").to_pylist() == [shared, "onset_a"]

    def test_a_child_sharing_a_projection_id_of_the_bundle_joins(self) -> None:
        bundle = self._bundle()
        samples, quarters = bundle.timelines["dpt_1"], bundle.timelines["clt_6"]
        (shared,) = quarters.add_events(
            [{"event_type": "Projection", "start": Fraction(1)}], seed=7
        )
        part = DiscretePhysicalTimeline(length=1000, uid="part")
        part.add_events([{"event_type": "Projection", "start": 7}], seed=7)
        samples.add_child(part, 0)
        assert samples.n_children == 1
        assert part._parent is samples
        assert part.events.table.column("id").to_pylist() == [shared]
        assert quarters.events.table.column("id").to_pylist() == [shared]


class TestProjectionObjects:
    def test_an_instant_without_id(self) -> None:
        timeline = _samples()
        ids = timeline.add_events(
            Projection(
                start=Coordinate(Fraction(1, 2), TimeUnit.seconds),
                projection_of=EventReference(timeline_id="clt_6", event_id="x1azmn68"),
            )
        )
        (minted,) = ids
        assert minted.startswith("projection_")
        row = _row(timeline, minted)
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
            ],
            seed=7,
        )
        assert ids == ["given", IdMinter(7).mint("note")]
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


class TestGetOrAdd:
    """``add_projections`` keeps one Projection per thing it stands for."""

    @staticmethod
    def _projections() -> list[Projection]:
        n1 = EventReference(timeline_id="clt_6", event_id="n1")
        n2 = EventReference(timeline_id="clt_6", event_id="n2")
        half = Coordinate(Fraction(1, 2), TimeUnit.seconds)
        one = Coordinate(Fraction(1), TimeUnit.seconds)
        return [
            Projection(start=half, projection_of=n1),
            Projection(start=half, projection_of=n2),
            Projection(start=one),
            Projection(start=one, duration=Duration(Fraction(1, 2), TimeUnit.seconds)),
        ]

    def test_a_second_call_reuses_every_projection(self) -> None:
        timeline = _samples()
        expected = timeline.add_projections(self._projections())
        assert len(set(expected)) == 4
        assert timeline.add_projections(self._projections()) == expected
        assert len(timeline.events) == 4
        first, second, third, fourth = expected
        assert [
            (row["id"], row["start"]["numerator"], row["duration"])
            for row in timeline.events.table.to_pylist()
        ] == [
            (first, 22050, None),
            (second, 22050, None),
            (third, 44100, None),
            (
                fourth,
                44100,
                {"value": 22050.0, "numerator": 22050, "denominator": 1},
            ),
        ]

    def test_the_position_is_compared_on_the_axis(self) -> None:
        timeline = _samples()
        first = timeline.add_projections(self._projections())[0]
        n1 = EventReference(timeline_id="clt_6", event_id="n1")
        assert timeline.add_projections(
            [{"event_type": "Projection", "start": 22050, "projection_of": n1}]
        ) == [first]
        assert len(timeline.events) == 4

    def test_one_value_is_one_key_however_the_cell_is_spelled(self) -> None:
        timeline = _quarters()
        unreduced = {"value": 0.5, "numerator": 2, "denominator": 4}
        (added,) = timeline.add_projections(
            [{"event_type": "Projection", "start": unreduced}]
        )
        assert _row(timeline, added)["start"] == {
            "value": 0.5,
            "numerator": 1,
            "denominator": 2,
        }
        assert timeline.add_projections(
            [Projection(start=Coordinate(Fraction(1, 2), TimeUnit.quarters))]
        ) == [added]
        assert len(timeline.events) == 1

    def test_an_event_table_given_whole_is_held_in_lowest_terms(self) -> None:
        start = pa.array(
            [
                {"value": 0.5, "numerator": 2, "denominator": 4},
                {"value": 0.75, "numerator": 3, "denominator": 4},
            ],
            RATIONAL_STRUCT_TYPE,
        )
        events = EventData.from_arrays(
            {"id": ["a", "b"], "start": start}, TimeUnit.quarters
        )
        assert events.table.column("start").to_pylist() == [
            {"value": 0.5, "numerator": 1, "denominator": 2},
            {"value": 0.75, "numerator": 3, "denominator": 4},
        ]

    def test_equal_rows_of_one_call_are_one_projection(self) -> None:
        timeline = _samples()
        rows = [Projection(start=Coordinate(22050, TimeUnit.samples))] * 3 + [
            Projection(start=Coordinate(44100, TimeUnit.samples))
        ]
        ids = timeline.add_projections(rows)
        assert ids == [ids[0], ids[0], ids[0], ids[3]]
        assert ids[0] != ids[3]
        assert len(timeline.events) == 2

    def test_what_tells_projections_apart(self) -> None:
        timeline = _samples()
        n1 = EventReference(timeline_id="clt_6", event_id="n1")
        at = Coordinate(22050, TimeUnit.samples)
        later = Coordinate(66150, TimeUnit.samples)
        ids = timeline.add_projections(
            [
                Projection(start=at, projection_of=n1),
                Projection(start=at),
                Projection(start=later, projection_of=n1),
                Projection(start=at, duration=Duration(100, TimeUnit.samples)),
            ]
        )
        assert len(set(ids)) == len(ids) == 4
        assert len(timeline.events) == 4

    def test_a_given_id_is_kept_and_reused(self) -> None:
        timeline = _samples()
        given = Projection(id="x1azmn68", start=Coordinate(22050, TimeUnit.samples))
        assert timeline.add_projections(given) == ["x1azmn68"]
        assert timeline.add_projections([given, given]) == ["x1azmn68", "x1azmn68"]
        assert len(timeline.events) == 1

    def test_refusals_change_nothing(self) -> None:
        timeline = _samples()
        timeline.add_events([{"id": "n9", "event_type": "Note", "instant": 0}])
        timeline.add_projections(
            Projection(id="x1azmn68", start=Coordinate(22050, TimeUnit.samples))
        )
        at = Coordinate(44100, TimeUnit.samples)
        cases = [
            (
                [Projection(id="n9", start=at)],
                "Event id 'n9' is already held by another event of timeline 'dpt_1'",
            ),
            (
                [Projection(id="x1azmn68", start=at)],
                "Event id 'x1azmn68' is already held by another event of timeline "
                "'dpt_1'",
            ),
            (
                [
                    Projection(id="p", start=at),
                    Projection(id="p", start=Coordinate(66150, TimeUnit.samples)),
                ],
                "Projection id 'p' is given twice for different positions in one "
                "call on timeline 'dpt_1'",
            ),
            (
                [Projection(start=at), {"event_type": "Note", "instant": 3}],
                "add_projections adds Projections, not a row of type 'Note'",
            ),
        ]
        for rows, message in cases:
            with pytest.raises(ValueError) as raised:
                timeline.add_projections(rows)
            assert str(raised.value) == message
            assert len(timeline.events) == 2
