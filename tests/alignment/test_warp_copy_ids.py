"""Ids of a warped copy and of a new group (see README)."""

from __future__ import annotations

import pytest

from timetoalign.alignment import AlignmentBundle, MatchLine, WarpMap
from timetoalign.core.ids import IdMinter, content_minter, reset_content_minters
from timetoalign.maps import LinearMap
from timetoalign.timelines.types import ContinuousLogicalTimeline

from .helpers import make_match_stamp as MatchStamp


def _warp() -> WarpMap:
    """Score quarters to audio seconds at 2:1."""
    stamps = [
        MatchStamp(
            coordinates={"score": float(q), "audio": q / 2},
            coordinate_units={"score": "quarters", "audio": "seconds"},
            stated_pairs=[("score", "audio")],
        )
        for q in (0, 100, 200)
    ]
    return WarpMap.from_match_line(
        MatchLine(source_timeline_id="score", stamps=stamps), "audio"
    )


def _score() -> ContinuousLogicalTimeline:
    score = ContinuousLogicalTimeline(length=200, uid="score")
    score.create_region(0, 100, name="Verse", id="verse")
    score.create_region(100, 200, name="Chorus", id="chorus")
    return score


def test_a_warped_copy_takes_region_ids_of_its_own() -> None:
    score = _score()
    reset_content_minters()
    warped = _warp().materialise(score)
    expected = {}
    for source_id in ("verse", "chorus"):
        reset_content_minters()
        expected[source_id] = content_minter("region", source_id, warped.id).mint(
            "region"
        )
    assert list(warped._regions) == [expected["verse"], expected["chorus"]]
    assert [
        (region.name, region.start.value, region.end.value)
        for region in warped._regions.values()
    ] == [("Verse", 0.0, 50.0), ("Chorus", 50.0, 100.0)]
    bundle = AlignmentBundle()
    bundle.add_timeline(score)
    bundle.add_timeline(warped)
    assert list(bundle.timelines) == ["score", warped.id]


def test_a_seeded_warp_mints_from_the_seed_in_order() -> None:
    minter = IdMinter(7)
    expected = [minter.mint("cpt"), minter.mint("region"), minter.mint("region")]
    first = _warp().materialise(_score(), seed=7)
    LinearMap(scalar=3)  # something built in between
    second = _warp().materialise(_score(), seed=7)
    for warped in (first, second):
        assert [warped.id, *warped._regions] == expected


def test_a_new_group_id_held_in_the_bundle_refuses_before_anything_joins() -> None:
    held_id = IdMinter(5).mint("group")
    bundle = AlignmentBundle()
    bundle.add_timeline(ContinuousLogicalTimeline(length=4, uid=held_id))
    newcomer = ContinuousLogicalTimeline(length=4, uid="newcomer")
    with pytest.raises(ValueError) as refused:
        bundle.add_timeline(newcomer, as_group="g", seed=5)
    assert str(refused.value) == (
        f"group '{held_id}' cannot join bundle '{bundle.id}': its id '{held_id}' "
        f"is already held by timeline '{held_id}'"
    )
    assert list(bundle.timelines) == [held_id]
    assert bundle.groups == {}
