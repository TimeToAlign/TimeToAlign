"""Tests for shared coordinate decomposition and timeline resolution."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from typing import Literal

import pandas as pd
import pytest

from timetoalign import (
    AlignmentBundle,
    Coordinate,
    IdCoordinate,
    NumberType,
    ResolvedCoordinate,
    Timeline,
    TimelineGroup,
    TimeUnit,
    resolve_coordinate_spec,
)
from timetoalign.maps import LinearMap, ScalarMap
from timetoalign.timelines import ContinuousPhysicalTimeline


@pytest.mark.parametrize("value", [7, 2.5, Fraction(3, 4)])
def test_raw_coordinate_spec_passthrough(value: int | float | Fraction) -> None:
    """Raw numeric inputs retain their exact value and optional timeline ID."""
    assert resolve_coordinate_spec(value, timeline_id="axis") == ResolvedCoordinate(
        value, "axis", None
    )


def test_coordinate_decomposition() -> None:
    """A Coordinate contributes its value and unit."""
    coordinate = Coordinate(Fraction(5, 3), TimeUnit.quarters)
    assert resolve_coordinate_spec(coordinate) == ResolvedCoordinate(
        Fraction(5, 3), None, TimeUnit.quarters
    )


def test_id_coordinate_decomposition() -> None:
    """An IdCoordinate contributes its own timeline ID."""
    coordinate = IdCoordinate(11, TimeUnit.ticks, "notes")
    assert resolve_coordinate_spec(coordinate) == ResolvedCoordinate(
        11, "notes", TimeUnit.ticks
    )


def test_id_coordinate_conflict() -> None:
    """Conflicting explicit and embedded timeline IDs are rejected."""
    coordinate = IdCoordinate(11, TimeUnit.ticks, "notes")
    with pytest.raises(ValueError, match="axis.*notes"):
        resolve_coordinate_spec(coordinate, timeline_id="axis")


def test_coordinate_spec_rejects_unsupported_type() -> None:
    """Unsupported input reports its concrete type."""
    with pytest.raises(TypeError, match="object"):
        resolve_coordinate_spec(object())  # type: ignore[arg-type]


def test_timeline_resolves_native_fraction_exactly() -> None:
    """Native-unit Fraction values are preserved without float coercion."""
    timeline = Timeline(length=8, unit=TimeUnit.quarters, uid="score")
    resolved = timeline.get_coordinate_at(
        Coordinate(Fraction(7, 4), TimeUnit.quarters), format="coordinate"
    )
    assert resolved == Coordinate(Fraction(7, 4), TimeUnit.quarters)
    assert isinstance(resolved.value, Fraction)


@pytest.mark.parametrize(
    ("rounding", "expected"),
    [("round", 2), ("floor", 2), ("ceil", 3), ("truncate", 2)],
)
def test_timeline_int_projection_rounds_exact_fractions(
    rounding: Literal["round", "floor", "ceil", "truncate"], expected: int
) -> None:
    """Integral projection applies the selected mode to exact ratios too."""
    timeline = Timeline(length=8, unit=TimeUnit.quarters, uid="score")

    assert (
        timeline.get_coordinate_at(Fraction(5, 2), format="int", rounding=rounding)
        == expected
    )


def test_timeline_positional_series_preserves_input_index() -> None:
    """A pandas input index survives canonical Series projection unchanged."""
    timeline = Timeline(length=8, unit=TimeUnit.seconds, uid="audio")
    query = pd.Series([1.25, 2.5], index=pd.Index(["left", "right"], name="side"))

    result = timeline.get_coordinates_at(query, format="series")

    assert result.name == "audio"
    assert result.index.equals(query.index)
    assert result.dtype == "float64"
    assert result.tolist() == [1.25, 2.5]


def test_timeline_empty_series_uses_selected_axis_dtype() -> None:
    """An empty Series retains the selected canonical axis representation."""
    timeline = Timeline(
        length=8,
        unit=TimeUnit.ticks,
        number_type=NumberType.int,
        uid="clock",
    )

    result = timeline.get_coordinates_at([], format="series")

    assert result.name == "clock"
    assert result.dtype == "int64"
    assert result.empty


def test_timeline_resolves_foreign_unit_through_cmap() -> None:
    """A foreign-unit coordinate is inverted through the attached C-Map."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")
    timeline.add_conversion_map(
        ScalarMap(
            scalar=1000,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )
    assert timeline.get_coordinate_at(
        Coordinate(2500, TimeUnit.milliseconds), format="coordinate"
    ) == Coordinate(2.5, TimeUnit.seconds)


def test_timeline_rejects_missing_conversion_path() -> None:
    """A missing C-Map reports both units and the timeline ID."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")
    with pytest.raises(ValueError, match="quarters.*seconds.*audio"):
        timeline.get_coordinate_at(Coordinate(2, TimeUnit.quarters))


def test_timeline_rejects_unknown_timeline_id() -> None:
    """A non-descendant timeline ID is never treated as a native coordinate."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")
    with pytest.raises(KeyError, match="missing.*audio"):
        timeline.get_coordinate_at(IdCoordinate(2, TimeUnit.seconds, "missing"))


def test_timeline_resolves_direct_child_offset_exactly() -> None:
    """A direct child's local coordinate receives the exact stored offset."""
    parent = Timeline(length=20, unit=TimeUnit.quarters, uid="score")
    child = Timeline(length=5, unit=TimeUnit.quarters, uid="measure")
    parent.add_child(child, offset=Fraction(9, 2))

    resolved = parent.get_coordinate_at(
        IdCoordinate(Fraction(3, 2), TimeUnit.quarters, "measure"),
        format="coordinate",
    )
    assert resolved == Coordinate(Fraction(6), TimeUnit.quarters)
    assert isinstance(resolved.value, Fraction)


def test_timeline_converts_scalar_child_value_before_native_offset() -> None:
    """A foreign child-local value is converted before its native offset is added."""
    parent = Timeline(length=20, unit=TimeUnit.seconds, uid="audio")
    child = Timeline(length=5, unit=TimeUnit.seconds, uid="child")
    parent.add_child(child, offset=10)
    parent.add_conversion_map(
        ScalarMap(
            scalar=1000,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )
    child_coordinate = IdCoordinate(2000, TimeUnit.milliseconds, "child")

    assert parent.get_coordinate_at(
        child_coordinate, format="coordinate"
    ) == Coordinate(12.0, TimeUnit.seconds)
    dataframe = parent.get_timestamp_table(
        [child_coordinate], conversion_maps=False, format="dataframe"
    )
    assert dataframe["audio (seconds)"].tolist() == [12.0]


def test_timeline_converts_affine_child_value_before_native_offset() -> None:
    """Affine inverse conversion also precedes the native child offset."""
    parent = Timeline(length=20, unit=TimeUnit.seconds, uid="audio")
    child = Timeline(length=5, unit=TimeUnit.seconds, uid="child")
    parent.add_child(child, offset=10)
    parent.add_conversion_map(
        LinearMap(
            scalar=1000,
            offset=500,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )

    assert parent.get_coordinate_at(
        IdCoordinate(2500, TimeUnit.milliseconds, "child"), format="coordinate"
    ) == Coordinate(12.0, TimeUnit.seconds)


def test_timeline_resolves_grandchild_offset_exactly() -> None:
    """A grandchild coordinate receives both ancestor offsets."""
    parent = ContinuousPhysicalTimeline(length=20.0, uid="parent")
    child = ContinuousPhysicalTimeline(length=10.0, uid="child")
    grandchild = ContinuousPhysicalTimeline(length=5.0, uid="grandchild")
    child.add_child(grandchild, offset=5.0)
    parent.add_child(child, offset=10.0)

    assert parent.get_coordinate_at(
        IdCoordinate(2.0, TimeUnit.seconds, "grandchild"), format="coordinate"
    ) == Coordinate(17.0, TimeUnit.seconds)


def test_timeline_composes_grandchild_fraction_offsets_exactly() -> None:
    """Two rational descendant offsets remain exact when composed upward."""
    parent = Timeline(length=10, unit=TimeUnit.quarters, uid="parent")
    child = Timeline(length=4, unit=TimeUnit.quarters, uid="child")
    grandchild = Timeline(length=2, unit=TimeUnit.quarters, uid="grandchild")
    child.add_child(grandchild, offset=Fraction(3, 2))
    parent.add_child(child, offset=Fraction(9, 2))

    resolved = parent.get_coordinate_at(
        IdCoordinate(Fraction(0), TimeUnit.quarters, "grandchild"),
        format="coordinate",
    )

    assert resolved == Coordinate(Fraction(6), TimeUnit.quarters)
    assert isinstance(resolved.value, Fraction)


def test_timeline_id_keyword_validates_the_result_axis() -> None:
    """A Timeline result-axis validator must identify the receiver."""
    parent = Timeline(length=20, unit=TimeUnit.seconds, uid="parent")
    child = Timeline(length=5, unit=TimeUnit.seconds, uid="child")
    parent.add_child(child, offset=10)

    with pytest.raises(KeyError, match="child.*parent"):
        parent.get_coordinate_at(2, timeline_id="child")
    assert parent.get_coordinate_at(
        IdCoordinate(2, TimeUnit.seconds, "child"), format="coordinate"
    ) == Coordinate(12.0, TimeUnit.seconds)


def test_timeline_rejects_conflicting_coordinate_and_keyword_ids() -> None:
    """An explicit timeline ID cannot conflict with an embedded timeline ID."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")

    with pytest.raises(KeyError, match="other.*audio"):
        timeline.get_coordinate_at(
            IdCoordinate(2, TimeUnit.seconds, "other"), timeline_id="audio"
        )


def test_timeline_uses_grandparent_cmap_for_grandchild_coordinate() -> None:
    """A grandparent C-Map converts a foreign-unit grandchild value."""
    parent = Timeline(length=20, unit=TimeUnit.seconds, uid="parent")
    child = Timeline(length=10, unit=TimeUnit.seconds, uid="child")
    grandchild = Timeline(length=5, unit=TimeUnit.seconds, uid="grandchild")
    child.add_child(grandchild, offset=5)
    parent.add_child(child, offset=10)
    parent.add_conversion_map(
        ScalarMap(
            scalar=1000,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )

    assert parent.get_coordinate_at(
        IdCoordinate(2000, TimeUnit.milliseconds, "grandchild"), format="coordinate"
    ) == Coordinate(17.0, TimeUnit.seconds)


def test_timeline_uses_intermediate_cmap_for_grandchild_coordinate() -> None:
    """An intermediate C-Map takes precedence over a grandparent C-Map."""
    parent = Timeline(length=20, unit=TimeUnit.seconds, uid="parent")
    child = Timeline(length=10, unit=TimeUnit.seconds, uid="child")
    grandchild = Timeline(length=5, unit=TimeUnit.seconds, uid="grandchild")
    child.add_child(grandchild, offset=5)
    parent.add_child(child, offset=10)
    parent.add_conversion_map(
        ScalarMap(
            scalar=100,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )
    child.add_conversion_map(
        ScalarMap(
            scalar=1000,
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
        )
    )

    assert parent.get_coordinate_at(
        IdCoordinate(2000, TimeUnit.milliseconds, "grandchild"), format="coordinate"
    ) == Coordinate(17.0, TimeUnit.seconds)


def test_timestamp_table_rejects_unknown_timeline_id() -> None:
    """Timestamp tables reject coordinates qualified by an unknown timeline."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")

    with pytest.raises(ValueError) as exc_info:
        timeline.get_timestamp_table(
            [IdCoordinate(2, TimeUnit.seconds, "missing")],
            conversion_maps=False,
        )

    assert str(exc_info.value) == (
        "Timeline ID 'missing' is not this timeline 'audio' or one of its descendants"
    )


GroupCoordinateQuery = Callable[[TimelineGroup, IdCoordinate], object]


def _group_get_timestamp(group: TimelineGroup, coord: IdCoordinate) -> object:
    return group.get_timestamp_at(coord, "audio")


def _group_get_timestamps(group: TimelineGroup, coord: IdCoordinate) -> object:
    return group.get_timestamps_at([coord], "audio")


def _group_get_coordinate(group: TimelineGroup, coord: IdCoordinate) -> object:
    return group.get_coordinate_at(coord, timeline_id="audio")


@pytest.mark.parametrize(
    "query",
    [_group_get_timestamp, _group_get_timestamps],
)
def test_timeline_group_rejects_conflicting_coordinate_id(
    query: GroupCoordinateQuery,
) -> None:
    """Group queries reject an embedded ID that conflicts with the explicit ID."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")
    group = TimelineGroup(id="group", timelines=[timeline])
    coord = IdCoordinate(2, TimeUnit.seconds, "other")

    with pytest.raises(ValueError) as exc_info:
        query(group, coord)

    assert str(exc_info.value) == (
        "Timeline ID 'audio' conflicts with coordinate timeline ID 'other'"
    )


def test_timeline_group_retrieval_rejects_unknown_source_axis() -> None:
    """Typed group retrieval treats the embedded ID as the source axis."""
    timeline = Timeline(length=10, unit=TimeUnit.seconds, uid="audio")
    group = TimelineGroup(id="group", timelines=[timeline])

    with pytest.raises(KeyError, match="other"):
        _group_get_coordinate(group, IdCoordinate(2, TimeUnit.seconds, "other"))


BundleCoordinateQuery = Callable[[AlignmentBundle, IdCoordinate], object]


def _bundle_get_matchstamp(bundle: AlignmentBundle, coord: IdCoordinate) -> object:
    return bundle.get_matchstamp_at(coord, "source")


def _bundle_transfer(bundle: AlignmentBundle, coord: IdCoordinate) -> object:
    return bundle.transfer(coord, "source", "target")


def _bundle_transfer_interval(bundle: AlignmentBundle, coord: IdCoordinate) -> object:
    return bundle.transfer_interval(coord, 4, "source", "target")


@pytest.mark.parametrize(
    "query",
    [_bundle_get_matchstamp, _bundle_transfer, _bundle_transfer_interval],
)
def test_alignment_bundle_rejects_conflicting_coordinate_id(
    query: BundleCoordinateQuery,
) -> None:
    """Bundle queries reject an embedded ID that conflicts with the explicit ID."""
    bundle = AlignmentBundle(id="bundle")
    bundle.add_timeline(Timeline(length=10, unit=TimeUnit.seconds, uid="source"))
    bundle.add_timeline(Timeline(length=10, unit=TimeUnit.seconds, uid="target"))
    coord = IdCoordinate(2, TimeUnit.seconds, "other")

    with pytest.raises(ValueError) as exc_info:
        query(bundle, coord)

    assert str(exc_info.value) == (
        "Timeline ID 'source' conflicts with coordinate timeline ID 'other'"
    )
