"""Tests for the 6 Timeline subclasses.

This module tests:
- ContinuousLogicalTimeline, DiscreteLogicalTimeline
- ContinuousPhysicalTimeline, DiscretePhysicalTimeline
- ContinuousGraphicalTimeline, DiscreteGraphicalTimeline

Validity Rationale:
    The TTA model defines 6 timeline types across 3 domains and 2 modalities.
    Each type has:
    1. Restricted allowed units (domain-specific)
    2. Appropriate default unit and number_type
    3. Consistent behavior with base Timeline
    These tests verify type constraints are enforced correctly.
"""

from __future__ import annotations

import pickle
from fractions import Fraction
from typing import cast

import pytest

from timetoalign.core import Domain, NumberType, TimeUnit, number_to_json
from timetoalign.maps.linear import ScalarMap
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    GraphicalTimeline,
    LogicalTimeline,
    PhysicalTimeline,
    SegmentLine,
    Timeline,
)

# region Parametrized Type Tests


class TestAllTimelineTypes:
    """Parametrized tests for all 6 timeline types."""

    @pytest.mark.parametrize(
        ("timeline_class", "legal_unit", "legal_length", "illegal_unit"),
        [
            (
                ContinuousLogicalTimeline,
                TimeUnit.quarters,
                Fraction(4, 1),
                TimeUnit.ticks,
            ),
            (DiscreteLogicalTimeline, TimeUnit.ticks, 480, TimeUnit.seconds),
            (ContinuousPhysicalTimeline, TimeUnit.seconds, 4.0, TimeUnit.samples),
            (DiscretePhysicalTimeline, TimeUnit.samples, 4, TimeUnit.seconds),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.centimeters,
                4.0,
                TimeUnit.pixels,
            ),
            (DiscreteGraphicalTimeline, TimeUnit.pixels, 4, TimeUnit.seconds),
        ],
    )
    def test_leaf_unit_contracts(
        self,
        timeline_class,
        legal_unit,
        legal_length,
        illegal_unit,
    ):
        """Each leaf type accepts its unit and rejects a forbidden one."""
        timeline = timeline_class(length=legal_length, unit=legal_unit)
        assert timeline.unit is legal_unit

        with pytest.raises(ValueError) as error:
            timeline_class(length=legal_length, unit=illegal_unit)

        assert type(error.value) is ValueError
        message = str(error.value)
        assert timeline_class.__name__ in message
        assert str(illegal_unit) in message
        assert "Allowed units:" in message

    @pytest.mark.parametrize(
        ("timeline_class", "unit", "number_type", "length"),
        [
            (ContinuousLogicalTimeline, TimeUnit.whole_note, NumberType.float, 4.0),
            (
                ContinuousLogicalTimeline,
                TimeUnit.whole_note,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (ContinuousLogicalTimeline, TimeUnit.quarters, NumberType.float, 4.0),
            (
                ContinuousLogicalTimeline,
                TimeUnit.quarters,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (
                ContinuousLogicalTimeline,
                TimeUnit.floating_measures,
                NumberType.float,
                4.0,
            ),
            (
                ContinuousLogicalTimeline,
                TimeUnit.floating_measures,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (ContinuousLogicalTimeline, TimeUnit.number, NumberType.float, 4.0),
            (
                ContinuousLogicalTimeline,
                TimeUnit.number,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (DiscreteLogicalTimeline, TimeUnit.ticks, NumberType.int, 4),
            (ContinuousPhysicalTimeline, TimeUnit.seconds, NumberType.float, 4.0),
            (
                ContinuousPhysicalTimeline,
                TimeUnit.seconds,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (
                ContinuousPhysicalTimeline,
                TimeUnit.milliseconds,
                NumberType.float,
                4.0,
            ),
            (
                ContinuousPhysicalTimeline,
                TimeUnit.milliseconds,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (ContinuousPhysicalTimeline, TimeUnit.minutes, NumberType.float, 4.0),
            (
                ContinuousPhysicalTimeline,
                TimeUnit.minutes,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (DiscretePhysicalTimeline, TimeUnit.samples, NumberType.int, 4),
            (DiscretePhysicalTimeline, TimeUnit.frames, NumberType.int, 4),
            (ContinuousGraphicalTimeline, TimeUnit.meters, NumberType.float, 4.0),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.meters,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.centimeters,
                NumberType.float,
                4.0,
            ),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.centimeters,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.millimeters,
                NumberType.float,
                4.0,
            ),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.millimeters,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (ContinuousGraphicalTimeline, TimeUnit.inches, NumberType.float, 4.0),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.inches,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (ContinuousGraphicalTimeline, TimeUnit.points, NumberType.float, 4.0),
            (
                ContinuousGraphicalTimeline,
                TimeUnit.points,
                NumberType.fraction,
                Fraction(4, 1),
            ),
            (DiscreteGraphicalTimeline, TimeUnit.pixels, NumberType.int, 4),
        ],
    )
    def test_leaf_legal_combinations_round_trip(
        self,
        timeline_class,
        unit,
        number_type,
        length,
    ):
        """Every legal leaf unit and number type survives a round trip."""
        original = timeline_class(
            length=length,
            unit=unit,
            number_type=number_type,
            uid="typed",
        )

        restored = Timeline.from_dict(original.to_dict())

        assert type(restored) is timeline_class
        assert restored.unit is unit
        assert restored.number_type is number_type
        assert restored.length.value == length

    def test_default_unit_is_valid(self, timeline_type_fixture):
        """Default unit is valid for the timeline type."""
        TimelineClass, default_unit, default_number_type, sample_length = (
            timeline_type_fixture
        )

        tl = TimelineClass(length=sample_length)
        assert tl.unit == default_unit

    @pytest.mark.parametrize(
        "timeline_class",
        [
            ContinuousLogicalTimeline,
            DiscreteLogicalTimeline,
            ContinuousPhysicalTimeline,
            DiscretePhysicalTimeline,
            ContinuousGraphicalTimeline,
            DiscreteGraphicalTimeline,
        ],
    )
    def test_base_from_dict_preserves_concrete_type(self, timeline_class):
        """The serialized class tag dispatches through the base registry."""
        original = timeline_class(length=8, uid="typed")
        restored = Timeline.from_dict(original.to_dict())
        assert type(restored) is timeline_class

    def test_default_number_type_is_valid(self, timeline_type_fixture):
        """Default number_type is valid for the timeline type."""
        TimelineClass, default_unit, default_number_type, sample_length = (
            timeline_type_fixture
        )

        tl = TimelineClass(length=sample_length)
        assert tl.number_type == default_number_type

    def test_timeline_has_correct_domain(self, timeline_type_fixture):
        """Timeline domain is derived from unit correctly."""
        TimelineClass, default_unit, default_number_type, sample_length = (
            timeline_type_fixture
        )

        tl = TimelineClass(length=sample_length)
        expected_domain = default_unit.domain
        assert tl.domain == expected_domain

    def test_timeline_can_hold_events(self, timeline_type_fixture):
        """All timeline types can hold events."""
        TimelineClass, default_unit, default_number_type, sample_length = (
            timeline_type_fixture
        )

        tl = TimelineClass(length=sample_length)

        # Create events with appropriate coordinate type
        if default_number_type == NumberType.int:
            coords = [0, int(sample_length) // 4, int(sample_length) // 2]
        elif default_number_type == NumberType.fraction:
            coords = [Fraction(0), sample_length / 4, sample_length / 2]
        else:
            coords = [0.0, float(sample_length) / 4, float(sample_length) / 2]

        events = [
            {
                "id": f"e_{i}",
                "temporal_type": "instant",
                "event_type": "Test",
                "instant": c,
            }
            for i, c in enumerate(coords)
        ]

        tl.add_events(events)
        assert tl.n_events == 3


# endregion


# region Timeline Subclass Resolution Tests


class TestTimelineSubclassResolution:
    """Subclass lookup uses unit specificity and number type."""

    def test_fractional_logical_unit_resolves_continuous_logical(self) -> None:
        """Fractional quarters select the continuous logical timeline."""
        result = Timeline.resolve_subclass(TimeUnit.quarters, NumberType.fraction)

        assert result is ContinuousLogicalTimeline

    def test_discrete_graphical_unit_resolves_discrete_graphical(self) -> None:
        """Integer pixels select the discrete graphical timeline."""
        result = Timeline.resolve_subclass(TimeUnit.pixels, NumberType.int)

        assert result is DiscreteGraphicalTimeline

    def test_string_unit_resolves_continuous_physical(self) -> None:
        """A string unit is normalized before selecting its canonical timeline."""
        result = Timeline.resolve_subclass("seconds")

        assert result is ContinuousPhysicalTimeline

    def test_unclaimed_unit_falls_back_to_base_timeline(self) -> None:
        """An unmatched unit returns the documented base Timeline fallback."""
        result = Timeline.resolve_subclass(cast(TimeUnit, object()))

        assert result is Timeline


# endregion


# region Serialized Type Parameters


class TestSerializedTypeParameters:
    """Parameterized timeline serialization tests."""

    @pytest.mark.parametrize(
        ("segment_class", "length", "event_instant", "target_unit"),
        [
            (DiscreteGraphicalTimeline, 10, 3, TimeUnit.points),
            (
                ContinuousLogicalTimeline,
                Fraction(4, 1),
                Fraction(1, 1),
                TimeUnit.whole_note,
            ),
        ],
    )
    def test_parameterized_segment_line_round_trip(
        self,
        segment_class,
        length,
        event_instant,
        target_unit,
    ):
        """SegmentLine tags restore their registered segment type hierarchy."""
        original = SegmentLine[segment_class](
            length=0,
            unit=segment_class._default_unit,
            number_type=segment_class._default_unit.default_number_type,
            uid="segments",
        )
        for index in range(2):
            segment = segment_class(length=length, uid=f"segment_{index}")
            segment.add_events(
                [
                    {
                        "id": f"event_{index}",
                        "temporal_type": "instant",
                        "event_type": "Marker",
                        "instant": event_instant,
                    }
                ]
            )
            segment.add_conversion_map(
                ScalarMap(
                    scalar=2,
                    source_unit=segment.unit,
                    target_unit=target_unit,
                    uid=f"map_{index}",
                )
            )
            original.append_segment(segment)

        data = original.to_dict(events=True)
        restored = Timeline.from_dict(data)

        assert data["class"] == f"SegmentLine[{segment_class.__name__}]"
        assert isinstance(restored, SegmentLine)
        assert restored.segment_type is segment_class
        assert restored.n_segments == original.n_segments == 2
        assert [type(segment) for _, _, segment in restored.iter_segments()] == [
            segment_class,
            segment_class,
        ]
        assert [segment.n_events for _, _, segment in restored.iter_segments()] == [
            1,
            1,
        ]
        assert [
            segment.to_dict(events=True)["events"][0]["id"]
            for _, _, segment in restored.iter_segments()
        ] == ["event_0", "event_1"]
        assert [
            segment.to_dict(events=True)["events"][0]["start"]
            for _, _, segment in restored.iter_segments()
        ] == [number_to_json(event_instant), number_to_json(event_instant)]
        assert [
            segment.get_conversion_map(target_unit).id
            for _, _, segment in restored.iter_segments()
        ] == ["map_0", "map_1"]

    def test_parameterized_segment_line_rejects_unregistered_inner_class(self):
        """An unknown segment type identifies both the type and full tag."""
        data = SegmentLine[DiscreteGraphicalTimeline](length=0).to_dict()
        data["class"] = "SegmentLine[UnregisteredSegmentTimeline]"

        with pytest.raises(ValueError) as error:
            Timeline.from_dict(data)

        assert str(error.value) == (
            "Unknown serialized timeline class 'UnregisteredSegmentTimeline' "
            "in parameterized tag 'SegmentLine[UnregisteredSegmentTimeline]'"
        )

    def test_nested_parameterized_segment_line_round_trip(self):
        """Nested SegmentLine tags restore both levels of segment type."""
        system_class = SegmentLine[DiscreteGraphicalTimeline]
        original = SegmentLine[system_class](length=0, uid="systems")
        system = system_class(length=0, uid="system_0")
        segment = DiscreteGraphicalTimeline(length=10, uid="measure_0")
        segment.add_events(
            [
                {
                    "id": "marker_0",
                    "temporal_type": "instant",
                    "event_type": "Marker",
                    "instant": 3,
                }
            ]
        )
        segment.add_conversion_map(
            ScalarMap(
                scalar=2,
                source_unit=TimeUnit.pixels,
                target_unit=TimeUnit.points,
                uid="pixels_to_points",
            )
        )
        system.append_segment(segment)
        original.append_segment(system)

        restored = Timeline.from_dict(original.to_dict(events=True))

        assert restored.class_name == (
            "SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]"
        )
        assert restored.segment_type is system_class
        assert restored.n_segments == 1
        _, restored_system = restored.get_segment_by_index(0)
        assert isinstance(restored_system, SegmentLine)
        assert restored_system.segment_type is DiscreteGraphicalTimeline
        assert restored_system.n_segments == 1
        _, restored_segment = restored_system.get_segment_by_index(0)
        assert restored_segment.n_events == 1
        assert (
            restored_segment.get_conversion_map(TimeUnit.points).id
            == "pixels_to_points"
        )

    def test_parameterized_class_cache_and_inheritance(self):
        """Dynamic classes are cached and inherit both public API surfaces."""
        line_class = SegmentLine[ContinuousPhysicalTimeline]

        assert line_class is SegmentLine[ContinuousPhysicalTimeline]
        assert issubclass(line_class, SegmentLine)
        assert issubclass(line_class, ContinuousPhysicalTimeline)

        line = line_class(length=0)
        assert isinstance(line, SegmentLine)
        assert isinstance(line, ContinuousPhysicalTimeline)
        assert line.unit is TimeUnit.seconds

    @pytest.mark.parametrize(
        "parameter",
        [Timeline, SegmentLine, 42],
    )
    def test_parameterization_rejects_non_strict_parameters(self, parameter):
        """Timeline and bare SegmentLine are not valid parameters."""
        with pytest.raises(TypeError):
            SegmentLine[parameter]

    def test_parameterized_class_cannot_be_parameterized_again(self):
        """Only the bare SegmentLine class accepts subscription."""
        with pytest.raises(TypeError):
            SegmentLine[ContinuousPhysicalTimeline][DiscreteGraphicalTimeline]

    def test_parameterized_segment_line_pickle_round_trip(self):
        """Pickle restores a dynamic class through its module-level name."""
        line_class = SegmentLine[ContinuousPhysicalTimeline]
        original = line_class(length=0, uid="pickled")
        original.append_segment(ContinuousPhysicalTimeline(length=2.5, uid="physical"))

        restored = pickle.loads(pickle.dumps(original))

        assert type(restored) is line_class
        assert restored.id == "pickled"
        assert restored.n_segments == 1

    def test_nested_payload_materializes_dynamic_classes(self):
        """A hand-captured nested payload materializes both dynamic classes."""
        zero = {"unit": "pixels", "number_type": "int", "value": 0}
        ten = {"unit": "pixels", "number_type": "int", "value": 10}
        payload = {
            "id": "systems",
            "name": None,
            "class": "SegmentLine[SegmentLine[DiscreteGraphicalTimeline]]",
            "unit": "pixels",
            "number_type": "int",
            "length": ten,
            "locked": False,
            "meta": {},
            "children": {
                "system": {
                    "offset": zero,
                    "timeline": {
                        "id": "system",
                        "name": None,
                        "class": "SegmentLine[DiscreteGraphicalTimeline]",
                        "unit": "pixels",
                        "number_type": "int",
                        "length": ten,
                        "locked": True,
                        "meta": {},
                        "children": {
                            "staff": {
                                "offset": zero,
                                "timeline": {
                                    "id": "staff",
                                    "name": None,
                                    "class": "DiscreteGraphicalTimeline",
                                    "unit": "pixels",
                                    "number_type": "int",
                                    "length": ten,
                                    "locked": True,
                                    "meta": {},
                                    "children": {},
                                    "conversion_maps": [],
                                },
                            }
                        },
                        "conversion_maps": [],
                    },
                }
            },
            "conversion_maps": [],
        }

        restored = Timeline.from_dict(payload)
        inner_class = SegmentLine[DiscreteGraphicalTimeline]

        assert type(restored) is SegmentLine[inner_class]
        assert type(restored.get_segment_by_index(0)[1]) is inner_class

    def test_continuous_graphical_pixels_legacy_payload_error(self):
        """Legacy continuous graphical pixel payloads name their valid replacement."""
        data = DiscreteGraphicalTimeline(length=16, uid="legacy").to_dict()
        data["class"] = "ContinuousGraphicalTimeline"

        with pytest.raises(ValueError) as error:
            Timeline.from_dict(data)

        assert str(error.value) == (
            "Serialized ContinuousGraphicalTimeline payload uses unit 'pixels', "
            "which is discrete-only. This payload predates unit enforcement; "
            "use DiscreteGraphicalTimeline instead."
        )


# endregion


# region Logical Timeline Tests


class TestLogicalTimeline:
    """Tests for logical (musical) timelines."""

    def test_logical_timeline_allowed_units(self):
        """LogicalTimeline accepts musical units only."""
        # Should work
        LogicalTimeline(length=4.0, unit=TimeUnit.quarters)
        LogicalTimeline(length=4.0, unit=TimeUnit.whole_note)
        LogicalTimeline(length=4.0, unit=TimeUnit.floating_measures)
        LogicalTimeline(length=480, unit=TimeUnit.ticks)

        # Should fail
        with pytest.raises(ValueError, match="does not allow unit"):
            LogicalTimeline(length=10.0, unit=TimeUnit.seconds)

    def test_continuous_logical_default_fraction(self):
        """ContinuousLogicalTimeline defaults to Fraction coordinates."""
        tl = ContinuousLogicalTimeline(length=Fraction(4, 1))
        assert tl.number_type == NumberType.fraction
        assert tl.unit == TimeUnit.quarters

    def test_continuous_logical_accepts_float(self):
        """ContinuousLogicalTimeline also accepts float number_type."""
        tl = ContinuousLogicalTimeline(length=4.0, number_type=NumberType.float)
        assert tl.number_type == NumberType.float

    def test_continuous_logical_rejects_int(self):
        """ContinuousLogicalTimeline rejects int number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            ContinuousLogicalTimeline(length=4, number_type=NumberType.int)

    def test_continuous_logical_rejects_ticks(self):
        """ContinuousLogicalTimeline rejects ticks unit."""
        with pytest.raises(ValueError, match="does not allow unit"):
            ContinuousLogicalTimeline(length=480, unit=TimeUnit.ticks)

    def test_discrete_logical_default_int(self):
        """DiscreteLogicalTimeline defaults to int coordinates."""
        tl = DiscreteLogicalTimeline(length=1920)
        assert tl.number_type == NumberType.int
        assert tl.unit == TimeUnit.ticks

    def test_discrete_logical_rejects_float(self):
        """DiscreteLogicalTimeline rejects float number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            DiscreteLogicalTimeline(length=1920, number_type=NumberType.float)

    def test_discrete_logical_rejects_quarters(self):
        """DiscreteLogicalTimeline rejects continuous units."""
        with pytest.raises(ValueError, match="does not allow unit"):
            DiscreteLogicalTimeline(length=4, unit=TimeUnit.quarters)


# endregion


# region Physical Timeline Tests


class TestPhysicalTimeline:
    """Tests for physical (acoustic) timelines."""

    def test_physical_timeline_allowed_units(self):
        """PhysicalTimeline accepts physical units only."""
        # Should work
        PhysicalTimeline(length=10.0, unit=TimeUnit.seconds)
        PhysicalTimeline(length=1000.0, unit=TimeUnit.milliseconds)
        PhysicalTimeline(length=5.0, unit=TimeUnit.minutes)
        PhysicalTimeline(length=44100, unit=TimeUnit.samples)

        # Should fail
        with pytest.raises(ValueError, match="does not allow unit"):
            PhysicalTimeline(length=4.0, unit=TimeUnit.quarters)

    def test_continuous_physical_default_float(self):
        """ContinuousPhysicalTimeline defaults to float coordinates."""
        tl = ContinuousPhysicalTimeline(length=10.0)
        assert tl.number_type == NumberType.float
        assert tl.unit == TimeUnit.seconds

    def test_continuous_physical_accepts_fraction(self):
        """ContinuousPhysicalTimeline also accepts Fraction."""
        tl = ContinuousPhysicalTimeline(
            length=Fraction(10, 1), number_type=NumberType.fraction
        )
        assert tl.number_type == NumberType.fraction

    def test_continuous_physical_rejects_int(self):
        """ContinuousPhysicalTimeline rejects int number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            ContinuousPhysicalTimeline(length=10, number_type=NumberType.int)

    def test_continuous_physical_rejects_samples(self):
        """ContinuousPhysicalTimeline rejects discrete units."""
        with pytest.raises(ValueError, match="does not allow unit"):
            ContinuousPhysicalTimeline(length=44100, unit=TimeUnit.samples)

    def test_discrete_physical_default_int(self):
        """DiscretePhysicalTimeline defaults to int coordinates."""
        tl = DiscretePhysicalTimeline(length=44100)
        assert tl.number_type == NumberType.int
        assert tl.unit == TimeUnit.samples

    def test_discrete_physical_rejects_float(self):
        """DiscretePhysicalTimeline rejects float number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            DiscretePhysicalTimeline(length=44100, number_type=NumberType.float)

    def test_discrete_physical_rejects_seconds(self):
        """DiscretePhysicalTimeline rejects continuous units."""
        with pytest.raises(ValueError, match="does not allow unit"):
            DiscretePhysicalTimeline(length=10, unit=TimeUnit.seconds)


# endregion


# region Graphical Timeline Tests


class TestGraphicalTimeline:
    """Tests for graphical (visual) timelines."""

    def test_graphical_timeline_allowed_units(self):
        """GraphicalTimeline accepts graphical units only."""
        # Should work
        GraphicalTimeline(length=1920, unit=TimeUnit.pixels)
        GraphicalTimeline(length=21.0, unit=TimeUnit.centimeters)
        GraphicalTimeline(length=0.21, unit=TimeUnit.meters)
        GraphicalTimeline(length=8.5, unit=TimeUnit.inches)

        # Should fail
        with pytest.raises(ValueError, match="does not allow unit"):
            GraphicalTimeline(length=10.0, unit=TimeUnit.seconds)

    def test_continuous_graphical_default_float(self):
        """ContinuousGraphicalTimeline defaults to float coordinates."""
        tl = ContinuousGraphicalTimeline(length=100.0)
        assert tl.number_type == NumberType.float
        assert tl.unit == TimeUnit.centimeters

    def test_continuous_graphical_accepts_fraction(self):
        """ContinuousGraphicalTimeline also accepts Fraction."""
        tl = ContinuousGraphicalTimeline(
            length=Fraction(100, 1), number_type=NumberType.fraction
        )
        assert tl.number_type == NumberType.fraction

    def test_continuous_graphical_rejects_int(self):
        """ContinuousGraphicalTimeline rejects int number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            ContinuousGraphicalTimeline(length=100, number_type=NumberType.int)

    def test_continuous_graphical_rejects_pixels(self):
        """Pixels belong exclusively to discrete graphical timelines."""
        with pytest.raises(ValueError, match="does not allow unit"):
            ContinuousGraphicalTimeline(length=1206.4, unit=TimeUnit.pixels)

    def test_continuous_graphical_rejects_non_graphical_units(self):
        """The domain still holds: seconds are not a graphical unit."""
        with pytest.raises(ValueError, match="does not allow unit"):
            ContinuousGraphicalTimeline(length=10.0, unit=TimeUnit.seconds)

    def test_discrete_graphical_default_int(self):
        """DiscreteGraphicalTimeline defaults to int coordinates."""
        tl = DiscreteGraphicalTimeline(length=1920)
        assert tl.number_type == NumberType.int
        assert tl.unit == TimeUnit.pixels

    def test_discrete_graphical_rejects_float(self):
        """DiscreteGraphicalTimeline rejects float number_type."""
        with pytest.raises(ValueError, match="does not admit number_type"):
            DiscreteGraphicalTimeline(length=1920, number_type=NumberType.float)

    def test_discrete_graphical_rejects_centimeters(self):
        """DiscreteGraphicalTimeline rejects continuous units."""
        with pytest.raises(ValueError, match="does not allow unit"):
            DiscreteGraphicalTimeline(length=100, unit=TimeUnit.centimeters)


# endregion


# region Domain Property Tests


class TestDomainProperties:
    """Test domain-related properties."""

    def test_logical_timeline_domain_is_logical(self):
        """Logical timelines have Domain.logical."""
        tl = ContinuousLogicalTimeline(length=Fraction(4, 1))
        assert tl.domain == Domain.logical

    def test_physical_timeline_domain_is_physical(self):
        """Physical timelines have Domain.physical."""
        tl = ContinuousPhysicalTimeline(length=10.0)
        assert tl.domain == Domain.physical

    def test_graphical_timeline_domain_is_graphical(self):
        """Graphical timelines have Domain.graphical."""
        tl = ContinuousGraphicalTimeline(length=100.0)
        assert tl.domain == Domain.graphical


# endregion


# region Inheritance Tests


class TestInheritance:
    """Test class hierarchy relationships."""

    def test_continuous_logical_is_logical(self):
        """ContinuousLogicalTimeline is a LogicalTimeline."""
        tl = ContinuousLogicalTimeline(length=Fraction(4, 1))
        assert isinstance(tl, LogicalTimeline)
        assert isinstance(tl, Timeline)

    def test_discrete_logical_is_logical(self):
        """DiscreteLogicalTimeline is a LogicalTimeline."""
        tl = DiscreteLogicalTimeline(length=1920)
        assert isinstance(tl, LogicalTimeline)
        assert isinstance(tl, Timeline)

    def test_continuous_physical_is_physical(self):
        """ContinuousPhysicalTimeline is a PhysicalTimeline."""
        tl = ContinuousPhysicalTimeline(length=10.0)
        assert isinstance(tl, PhysicalTimeline)
        assert isinstance(tl, Timeline)

    def test_discrete_physical_is_physical(self):
        """DiscretePhysicalTimeline is a PhysicalTimeline."""
        tl = DiscretePhysicalTimeline(length=44100)
        assert isinstance(tl, PhysicalTimeline)
        assert isinstance(tl, Timeline)

    def test_continuous_graphical_is_graphical(self):
        """ContinuousGraphicalTimeline is a GraphicalTimeline."""
        tl = ContinuousGraphicalTimeline(length=100.0)
        assert isinstance(tl, GraphicalTimeline)
        assert isinstance(tl, Timeline)

    def test_discrete_graphical_is_graphical(self):
        """DiscreteGraphicalTimeline is a GraphicalTimeline."""
        tl = DiscreteGraphicalTimeline(length=1920)
        assert isinstance(tl, GraphicalTimeline)
        assert isinstance(tl, Timeline)


# endregion


# region Cross-Domain Compatibility Tests


class TestDiscreteAndContinuousPredicates:
    """Test is_discrete and is_continuous convenience properties."""

    def test_continuous_logical_is_continuous(self):
        """ContinuousLogicalTimeline reports is_continuous=True."""
        tl = ContinuousLogicalTimeline(length=Fraction(4, 1))
        assert tl.is_continuous is True
        assert tl.is_discrete is False

    def test_discrete_logical_is_discrete(self):
        """DiscreteLogicalTimeline reports is_discrete=True."""
        tl = DiscreteLogicalTimeline(length=1920)
        assert tl.is_discrete is True
        assert tl.is_continuous is False

    def test_continuous_physical_is_continuous(self):
        """ContinuousPhysicalTimeline reports is_continuous=True."""
        tl = ContinuousPhysicalTimeline(length=10.0)
        assert tl.is_continuous is True
        assert tl.is_discrete is False

    def test_discrete_physical_is_discrete(self):
        """DiscretePhysicalTimeline reports is_discrete=True."""
        tl = DiscretePhysicalTimeline(length=44100)
        assert tl.is_discrete is True
        assert tl.is_continuous is False

    def test_continuous_graphical_is_continuous(self):
        """ContinuousGraphicalTimeline reports is_continuous=True."""
        tl = ContinuousGraphicalTimeline(length=100.0)
        assert tl.is_continuous is True
        assert tl.is_discrete is False

    def test_discrete_graphical_is_discrete(self):
        """DiscreteGraphicalTimeline reports is_discrete=True."""
        tl = DiscreteGraphicalTimeline(length=1920)
        assert tl.is_discrete is True
        assert tl.is_continuous is False

    def test_base_timeline_seconds_is_continuous(self):
        """Base Timeline with seconds unit is continuous."""
        tl = Timeline(length=10.0, unit=TimeUnit.seconds)
        assert tl.is_continuous is True
        assert tl.is_discrete is False

    def test_base_timeline_ticks_is_discrete(self):
        """Base Timeline with ticks unit is discrete."""
        tl = Timeline(length=480, unit=TimeUnit.ticks)
        assert tl.is_discrete is True
        assert tl.is_continuous is False

    def test_predicates_are_complementary(self, timeline_type_fixture):
        """is_discrete and is_continuous are always complementary."""
        TimelineClass, default_unit, default_number_type, sample_length = (
            timeline_type_fixture
        )
        tl = TimelineClass(length=sample_length)
        assert tl.is_discrete != tl.is_continuous


# endregion


# region To Typed Tests


class TestToTyped:
    """Test the to_typed() method for upgrading base Timeline to subtypes."""

    def test_base_timeline_seconds_becomes_continuous_physical(self):
        """Base Timeline with seconds upgrades to ContinuousPhysicalTimeline."""
        tl = Timeline(length=10.0, unit=TimeUnit.seconds)
        typed = tl.to_typed()
        assert type(typed) is ContinuousPhysicalTimeline
        assert typed.length.value == 10.0
        assert typed.unit == TimeUnit.seconds

    def test_base_timeline_ticks_becomes_discrete_logical(self):
        """Base Timeline with ticks upgrades to DiscreteLogicalTimeline."""
        tl = Timeline(length=1920, unit=TimeUnit.ticks, number_type=NumberType.int)
        typed = tl.to_typed()
        assert type(typed) is DiscreteLogicalTimeline
        assert typed.length.value == 1920

    def test_base_timeline_pixels_becomes_discrete_graphical(self):
        """Base Timeline with pixels upgrades to DiscreteGraphicalTimeline."""
        tl = Timeline(length=1920, unit=TimeUnit.pixels, number_type=NumberType.int)
        typed = tl.to_typed()
        assert type(typed) is DiscreteGraphicalTimeline

    def test_base_timeline_quarters_becomes_continuous_logical(self):
        """Base Timeline with quarters upgrades to ContinuousLogicalTimeline."""
        tl = Timeline(length=4.0, unit=TimeUnit.quarters)
        typed = tl.to_typed()
        assert type(typed) is ContinuousLogicalTimeline

    def test_base_timeline_samples_becomes_discrete_physical(self):
        """Base Timeline with samples upgrades to DiscretePhysicalTimeline."""
        tl = Timeline(length=44100, unit=TimeUnit.samples, number_type=NumberType.int)
        typed = tl.to_typed()
        assert type(typed) is DiscretePhysicalTimeline

    def test_base_timeline_centimeters_becomes_continuous_graphical(self):
        """Base Timeline with centimeters upgrades to ContinuousGraphicalTimeline."""
        tl = Timeline(length=100.0, unit=TimeUnit.centimeters)
        typed = tl.to_typed()
        assert type(typed) is ContinuousGraphicalTimeline

    def test_already_typed_returns_self(self):
        """to_typed on already-typed timeline returns self."""
        tl = ContinuousPhysicalTimeline(length=10.0)
        typed = tl.to_typed()
        assert typed is tl

    def test_to_typed_preserves_id(self):
        """to_typed preserves the timeline's unique ID."""
        tl = Timeline(length=10.0, unit=TimeUnit.seconds, uid="my_id")
        typed = tl.to_typed()
        assert typed.id == "my_id"

    def test_to_typed_preserves_name(self):
        """to_typed preserves the timeline's name."""
        tl = Timeline(length=10.0, unit=TimeUnit.seconds, name="audio_track")
        typed = tl.to_typed()
        assert typed.name == "audio_track"

    def test_to_typed_preserves_meta(self):
        """to_typed preserves metadata."""
        tl = Timeline(
            length=10.0,
            unit=TimeUnit.seconds,
            meta={"source": "test.wav"},
        )
        typed = tl.to_typed()
        assert typed.meta == {"source": "test.wav"}

    def test_to_typed_preserves_events(self):
        """to_typed preserves events."""
        tl = Timeline(length=10.0, unit=TimeUnit.seconds)
        tl.add_events(
            [
                {
                    "id": "e1",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 1.0,
                },
                {
                    "id": "e2",
                    "temporal_type": "instant",
                    "event_type": "Beat",
                    "instant": 2.0,
                },
            ]
        )
        typed = tl.to_typed()
        assert typed.n_events == 2

    def test_to_typed_preserves_regions(self):
        """to_typed preserves regions."""
        tl = Timeline(length=100.0, unit=TimeUnit.seconds)
        tl.add_region("Chorus", start=30.0, end=60.0)
        typed = tl.to_typed()
        assert "Chorus" in typed
        region = typed.get_region("Chorus")
        assert region.start.value == 30.0
        assert region.end.value == 60.0

    def test_to_typed_recursively_types_segment_line_hierarchy(self):
        """Every hierarchy level receives its exact concrete class."""
        grandchild = Timeline(length=1.0, unit=TimeUnit.seconds, uid="grandchild")
        child = Timeline(length=2.0, unit=TimeUnit.seconds, uid="child")
        child.add_child(grandchild, offset=0.0)
        line = SegmentLine(length=0, unit=TimeUnit.seconds, uid="line")
        line.append_segment(child)

        typed = line.to_typed()
        typed_child = typed.get_segment_by_index(0)[1]
        typed_grandchild = typed_child.get_child("grandchild")

        assert type(typed) is SegmentLine[ContinuousPhysicalTimeline]
        assert type(typed_child) is ContinuousPhysicalTimeline
        assert type(typed_grandchild) is ContinuousPhysicalTimeline


# endregion


# region Cross-Domain Compatibility Tests


class TestCrossDomainCompatibility:
    """Test that cross-domain nesting is prevented."""

    def test_logical_cannot_contain_physical(self):
        """Logical timeline rejects physical children."""
        logical = ContinuousLogicalTimeline(length=Fraction(8, 1))
        physical = ContinuousPhysicalTimeline(length=5.0)

        with pytest.raises(ValueError, match="does not match"):
            logical.add_child(physical, offset=Fraction(0))

    def test_physical_cannot_contain_graphical(self):
        """Physical timeline rejects graphical children."""
        physical = ContinuousPhysicalTimeline(length=20.0)
        graphical = ContinuousGraphicalTimeline(length=100.0)

        with pytest.raises(ValueError, match="does not match"):
            physical.add_child(graphical, offset=0.0)

    def test_graphical_cannot_contain_logical(self):
        """Graphical timeline rejects logical children."""
        graphical = ContinuousGraphicalTimeline(length=200.0)
        logical = ContinuousLogicalTimeline(length=Fraction(4, 1))

        with pytest.raises(ValueError, match="does not match"):
            graphical.add_child(logical, offset=0.0)


class TestTimestampTableEdgeCases:
    """Empty timelines and structured conversion maps stay exportable."""

    def test_empty_timeline_dataframe_keeps_its_columns(self):
        """No events still means a schema, so callers can read df.columns."""
        timeline = ContinuousLogicalTimeline(length=16, uid="clt_empty")

        df = timeline.get_timestamp_table(format="dataframe")

        assert len(df) == 0
        assert list(df.columns) == ["clt_empty (quarters)"]

    def test_structured_conversion_map_is_not_a_column(self):
        """A map answering {"mc": ..., "beat": ...} has no column type."""
        from timetoalign.maps import LinearMap
        from timetoalign.maps.meter import MetricalPositionMap, MetricMap

        grid = ContinuousLogicalTimeline(length=Fraction(120), uid="clt_metrical")
        grid.add_conversion_map(
            LinearMap(
                scalar=0.5,
                offset=0.0,
                source_unit=TimeUnit.quarters,
                target_unit=TimeUnit.seconds,
                uid="quarters_to_seconds",
            )
        )
        grid.add_conversion_map(
            MetricalPositionMap(
                MetricMap.from_uniform(n_measures=30, quarters_per_measure=Fraction(4)),
                uid="grid_metrical_map",
            )
        )

        names = grid.get_timestamp_table().column_names

        # The numeric seconds conversion becomes a column; neither typed beat
        # addresses nor the structured MetricalPositionMap do.
        assert "quarters_to_seconds" in names
        assert not any(name.endswith("_metrical_map") for name in names)


# endregion
