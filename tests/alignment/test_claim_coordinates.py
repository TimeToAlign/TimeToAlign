"""Regression tests for coordinates recorded by TimeSkeleton claims."""

from fractions import Fraction

import pytest

from timetoalign import Coordinate, Measure, SectionHierarchy, TimeSkeleton
from timetoalign.core import (
    IdCoordinate,
    MeasureId,
    MeasureIdAddress,
    NumberType,
    TimeUnit,
)
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    DiscretePhysicalTimeline,
)


def test_coordinate_keeps_its_value_unit_and_number_type() -> None:
    measures = [
        Measure(
            id=f"coordinate-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = DiscretePhysicalTimeline(length=1_000_000, uid="coordinate-samples")
    skeleton.attach(participant)

    claim = skeleton.create_match_claim(
        participant.id,
        at=MeasureId(1),
        coordinate=Coordinate(0.35, TimeUnit.seconds),
    )
    coordinate = claim.start_anchor.coordinate_a

    assert type(coordinate) is Coordinate
    assert coordinate.value == 0.35
    assert type(coordinate.value) is float
    assert coordinate.unit is TimeUnit.seconds
    assert coordinate.number_type is NumberType.float


def test_raw_int_uses_participant_unit_and_number_type() -> None:
    measures = [
        Measure(
            id=f"raw-int-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = DiscretePhysicalTimeline(length=1_000_000, uid="raw-int-samples")
    skeleton.attach(participant)

    claim = skeleton.create_match_claim(
        participant.id, at=MeasureId(1), coordinate=52_000
    )
    coordinate = claim.start_anchor.coordinate_a

    assert coordinate.value == 52_000
    assert type(coordinate.value) is int
    assert coordinate.unit is TimeUnit.samples
    assert coordinate.number_type is NumberType.int


def test_raw_fraction_uses_participant_unit_and_number_type() -> None:
    measures = [
        Measure(
            id=f"raw-fraction-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = ContinuousLogicalTimeline(
        length=Fraction(9), uid="raw-fraction-quarters"
    )
    skeleton.attach(participant)

    claim = skeleton.create_match_claim(
        participant.id, at=MeasureId(1), coordinate=Fraction(7, 3)
    )
    coordinate = claim.start_anchor.coordinate_a

    assert coordinate.value == Fraction(7, 3)
    assert type(coordinate.value) is Fraction
    assert coordinate.unit is TimeUnit.quarters
    assert coordinate.number_type is NumberType.fraction


def test_matching_id_coordinate_keeps_its_own_coordinate_data() -> None:
    measures = [
        Measure(
            id=f"matching-id-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = DiscretePhysicalTimeline(length=1_000_000, uid="matching-id-samples")
    skeleton.attach(participant)

    claim = skeleton.create_match_claim(
        participant.id,
        at=MeasureId(1),
        coordinate=IdCoordinate(
            0.35,
            TimeUnit.seconds,
            participant.id,
            number_type=NumberType.float,
        ),
    )
    coordinate = claim.start_anchor.coordinate_a

    assert type(coordinate) is Coordinate
    assert coordinate.value == 0.35
    assert type(coordinate.value) is float
    assert coordinate.unit is TimeUnit.seconds
    assert coordinate.number_type is NumberType.float


def test_mismatched_id_coordinate_names_both_timeline_ids() -> None:
    measures = [
        Measure(
            id=f"mismatched-id-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = DiscretePhysicalTimeline(
        length=1_000_000, uid="mismatched-id-samples"
    )
    skeleton.attach(participant)

    with pytest.raises(ValueError) as exc_info:
        skeleton.create_match_claim(
            participant.id,
            at=MeasureId(1),
            coordinate=IdCoordinate(0.35, TimeUnit.seconds, "different-timeline"),
        )

    message = str(exc_info.value)
    assert "different-timeline" in message
    assert participant.id in message


def test_attach_returns_skeleton_and_supports_chaining() -> None:
    measures = [
        Measure(
            id=f"attach-m{i}",
            time_signature="3/4",
            actual_length=Fraction(3),
            nominal_length=Fraction(3),
        )
        for i in range(1, 4)
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    samples = DiscretePhysicalTimeline(length=1_000_000, uid="attach-samples")
    quarters = ContinuousLogicalTimeline(length=Fraction(9), uid="attach-quarters")

    assert skeleton.attach(samples).attach(quarters) is skeleton
    assert skeleton.attach(samples) is skeleton
    assert skeleton.n_participants == 2


def test_reference_coordinate_rejects_seconds_unit() -> None:
    measures = [
        Measure(id="direct-unit-m1", actual_length=Fraction(3), time_signature="3/4")
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = ContinuousLogicalTimeline(length=Fraction(3), uid="direct-unit")
    skeleton.attach(participant)

    with pytest.raises(ValueError, match="seconds.*quarters.*reference axis"):
        skeleton.create_match_claim(
            participant.id,
            at=Coordinate(2, TimeUnit.seconds),
            coordinate=Fraction(0),
        )


def test_within_measure_reference_coordinate_rejects_seconds_unit() -> None:
    measures = [
        Measure(id="within-unit-m1", actual_length=Fraction(3), time_signature="3/4")
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = ContinuousLogicalTimeline(length=Fraction(3), uid="within-unit")
    skeleton.attach(participant)

    with pytest.raises(ValueError, match="seconds.*quarters.*reference axis"):
        skeleton.create_match_claim(
            participant.id,
            at=MeasureIdAddress(
                "within-unit-m1",
                at=Coordinate(Fraction(3, 2), TimeUnit.seconds),
            ),
            coordinate=Fraction(0),
        )


def test_rendition_qualified_address_is_not_resolved() -> None:
    measures = [
        Measure(id="rendition-m1", actual_length=Fraction(3), time_signature="3/4")
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = ContinuousLogicalTimeline(length=Fraction(3), uid="rendition")
    skeleton.attach(participant)

    with pytest.raises(
        NotImplementedError,
        match="rendition.*occurrence-qualified resolution is not available yet",
    ):
        skeleton.create_match_claim(
            participant.id,
            at=MeasureId("rendition-m1", rendition=2),
            coordinate=Fraction(0),
        )


def test_first_source_measure_records_exact_fraction_zero() -> None:
    measures = [
        Measure(id="source-first-m1", actual_length=Fraction(3), time_signature="3/4"),
        Measure(id="source-first-m2", actual_length=Fraction(3), time_signature="3/4"),
    ]
    skeleton = TimeSkeleton(SectionHierarchy.from_measures(measures))
    participant = ContinuousLogicalTimeline(length=Fraction(6), uid="source-first")
    skeleton.attach(participant)

    claim = skeleton.create_match_claim(
        participant.id,
        at=MeasureId("source-first-m1"),
        coordinate=Fraction(0),
    )
    reference_coordinate = claim.start_anchor.coordinate_b

    assert reference_coordinate == Coordinate(Fraction(0), TimeUnit.quarters)
    assert type(reference_coordinate.value) is Fraction
    assert reference_coordinate.unit is TimeUnit.quarters
    assert reference_coordinate.number_type is NumberType.fraction
