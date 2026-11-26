"""Calls given ``seed=`` mint from one minter (see README, "Calls given seed=")."""

from __future__ import annotations

import pytest

from timetoalign.core.ids import IdMinter
from timetoalign.maps import LinearMap
from timetoalign.timelines import (
    ContinuousGraphicalTimeline,
    ContinuousLogicalTimeline,
    MeasureMap,
)
from timetoalign.timelines.types import ContinuousPhysicalTimeline


def _expected(seed: int, *prefixes: str) -> list[str]:
    minter = IdMinter(seed)
    return [minter.mint(prefix) for prefix in prefixes]


def _built_before() -> None:
    """Content-minted objects a content seed would count."""
    ContinuousLogicalTimeline(length=8)
    LinearMap(scalar=3)


def _score() -> ContinuousLogicalTimeline:
    score = ContinuousLogicalTimeline(length=8, uid="score")
    score.add_conversion_map(
        LinearMap(scalar=2, source_unit="quarters", target_unit="seconds", uid="tempo")
    )
    score.add_events(
        [
            {"id": "b1", "event_type": "Break", "instant": 2, "breaks": "section"},
            {"id": "b2", "event_type": "Break", "instant": 6, "breaks": "section"},
        ]
    )
    return score


def test_get_slice_mints_the_slice_then_each_copied_map() -> None:
    score = _score()
    for _ in range(2):
        _built_before()
        sliced = score.get_slice(2, 6, seed=3)
        assert [sliced.id, *sliced._conversion_maps] == _expected(3, "clt", "cmap")


def test_create_regions_by_splitting_mints_in_coordinate_order() -> None:
    for _ in range(2):
        score = _score()
        _built_before()
        regions = score.create_regions_by_splitting("breaks", seed=4)
        assert [region.id for region in regions] == _expected(
            4, "region", "region", "region"
        )
        assert [region.name for region in regions] == [
            "section_1",
            "section_2",
            "section_3",
        ]


def test_create_segment_line_mints_the_line_then_each_segment() -> None:
    score = _score()
    for _ in range(2):
        _built_before()
        line = score.create_segment_line([0, 4, 8], copy_events=False, seed=5)
        assert [line.id, *line.list_children()] == _expected(5, "clt", "clt", "clt")


def test_derive_mints_the_timeline_then_its_inverse_map() -> None:
    score = _score()
    for _ in range(2):
        _built_before()
        derived = score.derive("seconds", seed=6)
        assert isinstance(derived, ContinuousPhysicalTimeline)
        assert [derived.id, *derived._conversion_maps] == _expected(6, "cpt", "cmap")
    named = score.derive("seconds", uid="audio", seed=6)
    assert [named.id, *named._conversion_maps] == ["audio", *_expected(6, "cmap")]


def test_measure_map_get_slice_mints_from_the_seed() -> None:
    bars = MeasureMap.from_rows([{"count": 1, "time_signature": "3/4"}, {"count": 4}])
    for _ in range(2):
        _built_before()
        assert bars.get_slice(3, 9, seed=8).id == IdMinter(8).mint("measure_map")


def test_a_refused_appended_child_keeps_its_id_and_name() -> None:
    parent = ContinuousLogicalTimeline(length=0, uid="parent")
    parent.create_region(0, 0, id="taken")
    child = ContinuousLogicalTimeline(length=4, uid="child", name="Child")
    with pytest.raises(ValueError, match="'taken' is already held by region 'taken'"):
        parent.append_child(child, uid="taken", name="Renamed")
    assert (child.id, child.name) == ("child", "Child")
    assert parent.list_children() == []


def test_create_calibrated_child_mints_the_map_then_the_child() -> None:
    for _ in range(2):
        page = ContinuousGraphicalTimeline(length=100, unit="points", uid="page")
        _built_before()
        child = page.create_calibrated_child(
            80, 10, [(10, 0), (90, 4)], value_unit="seconds", seed=8
        )
        assert [*child._conversion_maps, child.id] == _expected(8, "cmap", "cgt")
    page = ContinuousGraphicalTimeline(length=100, unit="points", uid="page")
    named = page.create_calibrated_child(
        80, 10, [(10, 0), (90, 4)], value_unit="seconds", uid="axis", seed=8
    )
    assert [*named._conversion_maps, named.id] == [*_expected(8, "cmap"), "axis"]
