"""Transitive cross-group union and support-policy tests.

These tests prove two properties of ``AlignmentBundle.get_matchstamp_at`` on a
purely synthetic fixture (no corpus data):

1. A query assembles the transitive cross-group union — it reaches every
   timeline of both merged bundles, including timelines only reachable through
   a bridge timeline whose unit-bearing anchor is converted to its native unit
   before warping.
2. A coordinate below the first alignment anchor is governed by
   ``support_policy`` (``omit`` / ``clamp`` / ``extrapolate``) and never yields
   a negative coordinate under any policy.

The scenario and every expected value are documented in ``README.md`` under
"Transitive Cross-Group Union & Support Policy".
"""

from __future__ import annotations

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import (
    AlignmentAnchor,
    MatchClaim,
    MatchClaimField,
)
from timetoalign.core import Coordinate, SupportPolicy, TimeUnit
from timetoalign.maps import ScalarMap
from timetoalign.timelines import Timeline

# region Fixture


def _star_claim(a_coord: float, b_id: str, b_coord: float) -> MatchClaim:
    """One synchronous instant claim a1:clt_1 <-> ``b_id``."""
    return MatchClaim(
        timeline_a_id="a1:clt_1",
        timeline_b_id=b_id,
        start_anchor=AlignmentAnchor(
            timeline_a_id="a1:clt_1",
            coordinate_a=Coordinate(a_coord, TimeUnit.quarters),
            timeline_b_id=b_id,
            coordinate_b=Coordinate(
                b_coord,
                TimeUnit.seconds,
            ),
        ),
        is_synchronous=True,
    )


def _make_bundles() -> tuple[AlignmentBundle, AlignmentBundle]:
    """Build the merged bundle and the standalone B bundle.

    Returns:
        ``(merged, b_bundle)``. ``merged`` is A ∪ B bridged at a1=50 ↔
        b_bridge=5 seconds (a derived-unit anchor). ``b_bundle`` is B alone, for the
        parity check.
    """
    # Bundle A: a per-claim-list star, each timeline in its own group.
    a1 = Timeline(length=100, unit=TimeUnit.quarters, uid="a1:clt_1")
    a2 = Timeline(length=100, unit=TimeUnit.seconds, uid="a2:cpt_1")
    a3 = Timeline(length=100, unit=TimeUnit.seconds, uid="a3:cpt_2")
    bundle_a = AlignmentBundle(id="bundle_a")
    bundle_a.add_timeline(a1, uid="a1:clt_1", as_group="ga1")
    bundle_a.add_timeline(a2, uid="a2:cpt_1", as_group="ga2")
    bundle_a.add_timeline(a3, uid="a3:cpt_2", as_group="ga3")
    star: list[MatchClaim] = []
    for a_coord in (0.0, 50.0, 100.0):
        star.append(_star_claim(a_coord, "a2:cpt_1", a_coord))  # a2 = a1
        star.append(_star_claim(a_coord, "a3:cpt_2", a_coord / 2))  # a3 = a1 / 2
    bundle_a.add_match_claims(star)

    # Bundle B: a WarpMap-able columnar field, b_bridge carries a derived-unit
    # C-Map (samples -> seconds). Each timeline in its own group.
    b_bridge = Timeline(length=1000, unit=TimeUnit.samples, uid="b_bridge:dpt_1")
    b_bridge.add_conversion_map(
        ScalarMap(
            scalar=0.01,
            source_unit=TimeUnit.samples,
            target_unit=TimeUnit.seconds,
        )
    )
    b1 = Timeline(length=3000, unit=TimeUnit.samples, uid="b1:dpt_2")
    b2 = Timeline(length=3000, unit=TimeUnit.samples, uid="b2:dpt_3")
    bundle_b = AlignmentBundle(id="bundle_b")
    bundle_b.add_timeline(b_bridge, uid="b_bridge:dpt_1", as_group="gb0")
    bundle_b.add_timeline(b1, uid="b1:dpt_2", as_group="gb1")
    bundle_b.add_timeline(b2, uid="b2:dpt_3", as_group="gb2")
    field = MatchClaimField.from_columns(
        timeline_a_ids=["b_bridge:dpt_1"] * 8,
        timeline_b_ids=["b1:dpt_2"] * 4 + ["b2:dpt_3"] * 4,
        coordinate_a=[200.0, 400.0, 600.0, 800.0, 200.0, 400.0, 600.0, 800.0],
        coordinate_b=[100.0, 900.0, 1700.0, 2500.0, 300.0, 600.0, 900.0, 1200.0],
        unit_a=TimeUnit.samples,
        unit_b=TimeUnit.samples,
    )
    bundle_b.add_match_claim_field(field)

    merged = AlignmentBundle.from_bundles([bundle_a, bundle_b], name="merged")
    merged.add_match_claims(
        [
            MatchClaim(
                timeline_a_id="a1:clt_1",
                timeline_b_id="b_bridge:dpt_1",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="a1:clt_1",
                    coordinate_a=Coordinate(50.0, TimeUnit.quarters),
                    timeline_b_id="b_bridge:dpt_1",
                    coordinate_b=Coordinate(5.0, TimeUnit.seconds),
                ),
            )
        ]
    )
    return merged, bundle_b


# endregion


# region Transitive union


class TestTransitiveUnion:
    """The union spans every timeline of both merged bundles."""

    def test_union_spans_all_timelines_with_exact_values(self) -> None:
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(50, "a1:clt_1")

        assert stamp.is_interpolated is False
        assert stamp.n_timelines == 6
        assert {
            timeline_id: stamp.get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamp.present_timelines
        } == {
            "a1:clt_1": 50.0,
            "a2:cpt_1": 50.0,
            "a3:cpt_2": 25.0,
            "b_bridge:dpt_1": 500.0,
            "b1:dpt_2": 1300.0,
            "b2:dpt_3": 750.0,
        }

    def test_union_has_no_negative_coordinates(self) -> None:
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(50, "a1:clt_1")
        assert all(coordinate.value >= 0 for coordinate in stamp.coordinates.values())

    def test_b_portion_parity_with_b_bundle_at_reconciled_coordinate(self) -> None:
        # The bridge anchor 5 seconds converts to 500 native samples; B's own
        # query there must agree on the transferred
        # timelines b1/b2.
        merged, b_bundle = _make_bundles()
        union = merged.get_matchstamp_at(50, "a1:clt_1")
        b_only = b_bundle.get_matchstamp_at(500, "b_bridge:dpt_1")

        assert b_only.get_coordinate_for("b1:dpt_2", format="float") == 1300.0
        assert b_only.get_coordinate_for("b2:dpt_3", format="float") == 750.0
        assert union.get_coordinate_for("b1:dpt_2") == b_only.get_coordinate_for(
            "b1:dpt_2"
        )
        assert union.get_coordinate_for("b2:dpt_3") == b_only.get_coordinate_for(
            "b2:dpt_3"
        )

    def test_mixed_claim_units_normalize_each_anchor(self) -> None:
        source = Timeline(length=1000, unit=TimeUnit.samples, uid="source")
        source.add_conversion_map(
            ScalarMap(
                scalar=0.01,
                source_unit=TimeUnit.samples,
                target_unit=TimeUnit.seconds,
            )
        )
        target = Timeline(length=100, unit=TimeUnit.seconds, uid="target")
        bundle = AlignmentBundle()
        bundle.add_timeline(source, uid="source", as_group="source_group")
        bundle.add_timeline(target, uid="target", as_group="target_group")
        source_coordinates = [
            Coordinate(2.0, TimeUnit.seconds),
            Coordinate(400.0, TimeUnit.samples),
            Coordinate(6.0, TimeUnit.seconds),
        ]
        claims = [
            MatchClaim(
                timeline_a_id="source",
                timeline_b_id="target",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="source",
                    coordinate_a=source_coordinate,
                    timeline_b_id="target",
                    coordinate_b=Coordinate(target_coordinate, TimeUnit.seconds),
                ),
            )
            for source_coordinate, target_coordinate in zip(
                source_coordinates, [20.0, 40.0, 60.0], strict=True
            )
        ]
        bundle.add_match_claims(claims)

        warp = bundle._get_or_build_warp_map("source", "target")

        assert warp is not None
        assert warp._source_float_array.tolist() == [200.0, 400.0, 600.0]
        assert bundle.transfer(300.0, "source", "target") == 30.0


# endregion


# region Support policy


class TestSupportPolicy:
    """An out-of-support query is governed by support_policy."""

    def test_default_is_omit(self) -> None:
        assert AlignmentBundle().support_policy is SupportPolicy.omit

    def test_omit_drops_out_of_support_timelines(self) -> None:
        # b_bridge=50 samples is below the field hull [200, 800]; the query's
        # own coordinate is native and is never reconciled, so b1/b2 are out of
        # support and dropped. Only b_bridge remains.
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(50, "b_bridge:dpt_1")
        assert stamp.is_interpolated is True
        assert stamp.get_coordinate_for("b_bridge:dpt_1", format="float") == 50.0
        assert stamp.present_timelines == ["b_bridge:dpt_1"]

    def test_clamp_yields_first_anchor_targets(self) -> None:
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(50, "b_bridge:dpt_1", support_policy="clamp")
        assert {
            timeline_id: stamp.get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamp.present_timelines
        } == {
            "b_bridge:dpt_1": 50.0,
            "b1:dpt_2": 100.0,  # warp(200), the first anchor
            "b2:dpt_3": 300.0,
        }

    def test_extrapolate_floors_negatives_and_keeps_positives(self) -> None:
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(
            50, "b_bridge:dpt_1", support_policy=SupportPolicy.extrapolate
        )
        assert {
            timeline_id: stamp.get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamp.present_timelines
        } == {
            "b_bridge:dpt_1": 50.0,
            "b1:dpt_2": 0.0,  # warp(50) = -500, floored to 0
            "b2:dpt_3": 75.0,  # warp(50) = 75, kept
        }

    @pytest.mark.parametrize("policy", ["omit", "clamp", "extrapolate"])
    def test_no_policy_emits_a_negative_coordinate(self, policy: str) -> None:
        merged, _ = _make_bundles()
        stamp = merged.get_matchstamp_at(50, "b_bridge:dpt_1", support_policy=policy)
        assert all(coordinate.value >= 0 for coordinate in stamp.coordinates.values())

    def test_query_coordinate_never_altered(self) -> None:
        merged, _ = _make_bundles()
        for policy in ("omit", "clamp", "extrapolate"):
            stamp = merged.get_matchstamp_at(
                50, "b_bridge:dpt_1", support_policy=policy
            )
            assert stamp.get_coordinate_for("b_bridge:dpt_1", format="float") == 50.0

    def test_instance_default_governs_omitted_argument(self) -> None:
        # Setting the bundle-level policy changes the no-argument behaviour.
        merged, _ = _make_bundles()
        merged.support_policy = SupportPolicy.clamp
        stamp = merged.get_matchstamp_at(50, "b_bridge:dpt_1")
        assert {
            timeline_id: stamp.get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamp.present_timelines
        } == {
            "b_bridge:dpt_1": 50.0,
            "b1:dpt_2": 100.0,
            "b2:dpt_3": 300.0,
        }


# endregion
