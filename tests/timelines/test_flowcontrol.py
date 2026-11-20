"""Exact tests for axis-bound timeline flow-control storage and its dictionary form."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from timetoalign.core import (
    Coordinate,
    FlowControlElement,
    IdCoordinate,
    NumberType,
    TimeUnit,
)
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
    DiscreteGraphicalTimeline,
    DiscreteLogicalTimeline,
    DiscretePhysicalTimeline,
    SegmentLine,
    Timeline,
)
from timetoalign.timelines.flowcontrol import Break, FlowControlRegistry, Jump, Marker


def _break(unit: TimeUnit = TimeUnit.quarters) -> Break:
    return Break(
        Coordinate(Fraction(3, 2), unit),
        control_type=FlowControlElement.fine,
        label="fine label",
        name="fine2",
        meta={"source": "fixture"},
    )


def _jump(unit: TimeUnit = TimeUnit.quarters, target: str | None = None) -> Jump:
    return Jump(
        Coordinate(Fraction(7, 3), unit),
        Coordinate(Fraction(1, 3), unit),
        control_type=FlowControlElement.dal_segno_al_coda,
        target=target,
        name="ds",
        label="D.S. al Coda",
        meta={"source": "fixture"},
    )


SEGNO = FlowControlElement.segno


def _segno(unit: TimeUnit = TimeUnit.quarters) -> Marker:
    return Marker(Coordinate(Fraction(1, 3), unit), id="segno2", control_type=SEGNO)


class TestBoundRegistry:
    """A bound registry validates units and stores the axis number type."""

    @pytest.mark.parametrize("kind", ["break", "jump", "marker"])
    def test_unit_mismatch_names_both_units(self, kind: str) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.seconds, number_type=NumberType.float
        )
        with pytest.raises(
            ValueError,
            match="coordinate unit quarters does not match registry unit seconds",
        ):
            if kind == "break":
                registry.add_break(_break())
            elif kind == "jump":
                registry.add_jump(_jump())
            else:
                registry.add_marker(
                    Marker(
                        Coordinate(1, TimeUnit.quarters),
                        name="segno",
                        control_type=FlowControlElement.segno,
                    )
                )

    def test_fraction_axis_reexpresses_float_coordinate_exactly(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        registry.add_marker(
            Marker(
                Coordinate(1.5, TimeUnit.quarters, number_type=NumberType.float),
                name="segno",
                control_type=SEGNO,
            )
        )
        assert registry.markers[0].coordinate == Coordinate(
            Fraction(3, 2), TimeUnit.quarters
        )
        assert registry.markers[0].coordinate.number_type is NumberType.fraction

    def test_float_axis_reexpresses_fraction_coordinate_as_python_float(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.seconds, number_type=NumberType.float
        )
        registry.add_marker(
            Marker(
                Coordinate(Fraction(1, 3), TimeUnit.seconds),
                name="coda",
                control_type=FlowControlElement.coda,
            )
        )
        assert registry.markers[0].coordinate.value == 0.3333333333333333
        assert registry.markers[0].coordinate.number_type is NumberType.float

    def test_fraction_axis_reexpresses_float_break_and_jump_exactly(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        authored = Coordinate(1.5, TimeUnit.quarters, number_type=NumberType.float)
        registry.add_break(Break(authored))
        registry.add_jump(Jump(authored, authored))
        stored = (
            registry.breaks[0].coordinate,
            registry.jumps[0].from_coordinate,
            registry.jumps[0].to_coordinate,
        )
        for coordinate in stored:
            assert coordinate == Coordinate(Fraction(3, 2), TimeUnit.quarters)
            assert type(coordinate.value) is Fraction
            assert coordinate.value == Fraction(3, 2)
            assert coordinate.number_type is NumberType.fraction

    def test_float_axis_reexpresses_fraction_break_and_jump_as_python_float(
        self,
    ) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.seconds, number_type=NumberType.float
        )
        authored = Coordinate(Fraction(1, 3), TimeUnit.seconds)
        registry.add_break(Break(authored))
        registry.add_jump(Jump(authored, authored))
        stored = (
            registry.breaks[0].coordinate,
            registry.jumps[0].from_coordinate,
            registry.jumps[0].to_coordinate,
        )
        for coordinate in stored:
            assert type(coordinate.value) is float
            assert coordinate.value == 0.3333333333333333
            assert coordinate.number_type is NumberType.float

    def test_query_coordinate_unit_mismatch_names_both_units(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        registry.add_break(_break())
        with pytest.raises(
            ValueError,
            match="Query coordinate unit seconds does not match registry unit quarters",
        ):
            registry.breaks_at(Coordinate(Fraction(3, 2), TimeUnit.seconds))
        assert registry.breaks_at(Fraction(3, 2)) == registry.breaks
        assert registry.breaks_at(Coordinate(Fraction(3, 2), TimeUnit.quarters)) == (
            registry.breaks
        )

    def test_unbound_registry_keeps_input_objects_unchanged(self) -> None:
        registry = FlowControlRegistry()
        brk = _break()
        jump = _jump()
        marker = Marker(
            Coordinate(Fraction(5, 3), TimeUnit.quarters),
            name="segno",
            control_type=SEGNO,
        )
        registry.add_break(brk)
        registry.add_jump(jump)
        registry.add_marker(marker)
        assert registry.breaks[0] is brk
        assert registry.jumps[0] is jump
        assert registry.markers[0] is marker
        assert registry.has_break_at(Coordinate(Fraction(3, 2), TimeUnit.seconds))

    def test_unbound_marker_keeps_an_id_coordinate_identical(self) -> None:
        registry = FlowControlRegistry()
        marker = IdCoordinate(Fraction(5, 3), TimeUnit.quarters, timeline_id="score")
        registry.add_marker(
            Marker(marker, name="segno", control_type=FlowControlElement.segno)
        )
        assert registry.markers[0].coordinate is marker
        assert type(registry.markers[0].coordinate) is IdCoordinate
        with pytest.raises(
            ValueError, match="A raw marker coordinate requires a bound registry unit"
        ):
            registry.add_marker(
                Marker(
                    Fraction(5, 3), name="coda", control_type=FlowControlElement.coda
                )
            )

    def test_constructor_content_does_not_alias_or_reorder_caller_containers(
        self,
    ) -> None:
        late = Break(Coordinate(5, TimeUnit.quarters))
        early = Break(Coordinate(1, TimeUnit.quarters))
        given_breaks = [late, early]
        given_jumps = [_jump()]
        given_markers = [_segno()]
        registry = FlowControlRegistry(
            breaks=given_breaks, jumps=given_jumps, markers=given_markers
        )
        assert registry.breaks is not given_breaks
        assert registry.jumps is not given_jumps
        assert registry.markers is not given_markers
        assert len(given_breaks) == 2
        assert given_breaks[0] is late
        assert given_breaks[1] is early
        assert len(registry.breaks) == 2
        # Both breaks minted flow_marker_1; the second one added is minted again.
        assert registry.breaks[0].id == "flow_marker_2"
        assert registry.breaks[0].coordinate is early.coordinate
        assert registry.breaks[1] is late


class TestFlowControlJson:
    """All fields and exact typed coordinates survive JSON serialization."""

    def test_break_round_trip(self) -> None:
        brk = _break()
        payload = json.loads(json.dumps(brk.to_dict()))
        restored = Break.from_dict(
            payload, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        assert restored == brk
        assert restored.to_dict() == payload

    def test_jump_round_trip(self) -> None:
        segno = _segno()
        jump = _jump(target=segno.id)
        payload = json.loads(json.dumps(jump.to_dict()))
        restored = Jump.from_dict(
            payload,
            unit=TimeUnit.quarters,
            number_type=NumberType.fraction,
            markers={segno.id: segno},
        )
        assert restored == jump
        assert restored.to_dict() == payload

    def test_registry_round_trip(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        registry.add_break(_break())
        registry.add_marker(_segno())
        registry.add_jump(_jump(target="segno2"))
        payload = json.loads(json.dumps(registry.to_dict()))
        restored = FlowControlRegistry.from_dict(
            payload, unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        assert restored.breaks == registry.breaks
        assert restored.jumps == registry.jumps
        assert restored.markers == registry.markers
        assert restored.to_dict() == payload

    def test_copy_is_independent(self) -> None:
        registry = FlowControlRegistry(
            unit=TimeUnit.quarters, number_type=NumberType.fraction
        )
        registry.add_break(_break())
        copied = registry.copy()
        copied.add_marker(
            Marker(
                Coordinate(Fraction(1, 3), TimeUnit.quarters),
                name="segno",
                control_type=SEGNO,
            )
        )
        registry.add_marker(_segno())
        registry.add_jump(_jump(target="segno2"))
        assert copied is not registry
        assert copied.unit is TimeUnit.quarters
        assert copied.number_type is NumberType.fraction
        assert copied.breaks == registry.breaks
        assert copied.jumps == []
        assert [marker.name for marker in copied.markers] == ["segno"]
        assert [marker.id for marker in registry.markers] == ["segno2"]


@pytest.mark.parametrize(
    "timeline",
    [
        Timeline(length=1),
        ContinuousLogicalTimeline(length=1),
        DiscreteLogicalTimeline(length=1),
        ContinuousPhysicalTimeline(length=1),
        DiscretePhysicalTimeline(length=1),
        ContinuousGraphicalTimeline(length=1),
        DiscreteGraphicalTimeline(length=1),
        SegmentLine[ContinuousPhysicalTimeline](),
    ],
    ids=(
        "timeline",
        "continuous-logical",
        "discrete-logical",
        "continuous-physical",
        "discrete-physical",
        "continuous-graphical",
        "discrete-graphical",
        "segment-line",
    ),
)
def test_every_timeline_has_fresh_axis_bound_storage(timeline: object) -> None:
    assert timeline.flow_control.unit is timeline.unit
    assert timeline.flow_control.number_type is timeline.number_type
    assert timeline.measure_map is None
    assert timeline.metric_hierarchy is None
