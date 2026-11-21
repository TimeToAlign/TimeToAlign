"""Transitive cross-group union and support-policy tests.

These tests prove two properties of ``AlignmentBundle.get_matchstamp_at`` on a
purely synthetic fixture (no corpus data):

1. A query assembles the transitive cross-group union — it reaches every
   timeline of both merged bundles, including timelines only reachable through
   a bridge timeline whose position, stated in a derived unit, is placed on
   the bridge's own axis when the bridging claim is made.
2. A coordinate below the first alignment anchor is governed by
   ``support_policy`` (``omit`` / ``clamp`` / ``extrapolate``) and never yields
   a negative coordinate under any policy.

The scenario and every expected value are documented in ``README.md`` under
"Transitive Cross-Group Union & Support Policy".
"""

from __future__ import annotations

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import MatchClaim, MatchClaimField
from timetoalign.core import Coordinate, SupportPolicy, TimeUnit
from timetoalign.maps import ScalarMap
from timetoalign.timelines import Timeline

from .helpers import match_claim

# region Fixture

B_BRIDGE = [200, 400, 600, 800, 200, 400, 600, 800]
B_TARGETS = [100, 900, 1700, 2500, 300, 600, 900, 1200]


def _star_claim(a_coord: int, b_id: str, b_coord: float) -> MatchClaim:
    """One synchronous instant claim a1:clt_1 <-> ``b_id``."""
    return match_claim(
        "a1:clt_1",
        Coordinate(a_coord, TimeUnit.quarters),
        b_id,
        Coordinate(b_coord, TimeUnit.seconds),
    )


def _make_bundles() -> tuple[AlignmentBundle, AlignmentBundle]:
    """Build the merged bundle and the standalone B bundle.

    Returns:
        ``(merged, b_bundle)``. ``merged`` is A ∪ B bridged at a1=50 ↔
        b_bridge=5 seconds (a derived-unit position, placed at 500 samples).
        ``b_bundle`` is B alone, for the parity check.
    """
    # Bundle A: a star of single claims, each timeline in its own group.
    a1 = Timeline(length=100, unit=TimeUnit.quarters, uid="a1:clt_1")
    a2 = Timeline(length=100, unit=TimeUnit.seconds, uid="a2:cpt_1")
    a3 = Timeline(length=100, unit=TimeUnit.seconds, uid="a3:cpt_2")
    bundle_a = AlignmentBundle(id="bundle_a")
    bundle_a.add_timeline(a1, as_group="ga1")
    bundle_a.add_timeline(a2, as_group="ga2")
    bundle_a.add_timeline(a3, as_group="ga3")
    star: list[MatchClaim] = []
    for a_coord in (0, 50, 100):
        star.append(_star_claim(a_coord, "a2:cpt_1", float(a_coord)))  # a2 = a1
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
    bundle_b.add_timeline(b_bridge, as_group="gb0")
    bundle_b.add_timeline(b1, as_group="gb1")
    bundle_b.add_timeline(b2, as_group="gb2")
    targets = ["b1:dpt_2"] * 4 + ["b2:dpt_3"] * 4
    field = MatchClaimField.from_columns(
        timeline_a_ids=["b_bridge:dpt_1"] * 8,
        timeline_b_ids=targets,
        event_a_ids=[f"bridge@{value}" for value in B_BRIDGE],
        event_b_ids=[
            f"{target}@{value}"
            for target, value in zip(targets, B_TARGETS, strict=True)
        ],
        coordinate_a=B_BRIDGE,
        coordinate_b=B_TARGETS,
        timelines=[b_bridge, b1, b2],
    )
    bundle_b.add_match_claim_field(field)

    merged = AlignmentBundle.from_bundles([bundle_a, bundle_b], name="merged")
    merged.create_match_claims(
        [
            (
                {"start": 50},
                "a1:clt_1",
                {"start": Coordinate(5.0, TimeUnit.seconds)},
                "b_bridge:dpt_1",
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

    def test_merge_keeps_every_claim_id_in_its_scope(self) -> None:
        merged, _ = _make_bundles()
        assert merged.claim_scopes == (merged.id, "bundle_a", "bundle_b")
        table = merged.claim_table
        assert table.column("id").to_pylist() == (
            [f"claim_{n}" for n in range(1, 7)]
            + [f"claim_{n}" for n in range(1, 9)]
            + ["claim_1"]
        )
        assert table.column("scope_key").to_pylist() == [1] * 6 + [2] * 8 + [0]

    def test_bridge_position_is_placed_on_the_bridge_axis(self) -> None:
        merged, _ = _make_bundles()
        (bridge,) = merged.get_match_claims(between=("a1:clt_1", "b_bridge:dpt_1"))
        assert (bridge.event_a_id, bridge.event_b_id) == (
            "projection_1",
            "projection_1",
        )
        assert bridge.start_anchor.coordinate_b == Coordinate(500, TimeUnit.samples)

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
        bundle.add_timeline(source, as_group="source_group")
        bundle.add_timeline(target, as_group="target_group")
        source_positions = [
            Coordinate(2.0, TimeUnit.seconds),
            Coordinate(400, TimeUnit.samples),
            Coordinate(6.0, TimeUnit.seconds),
        ]
        claims = bundle.create_match_claims(
            [
                ({"start": position}, "source", {"start": target_position}, "target")
                for position, target_position in zip(
                    source_positions, [20.0, 40.0, 60.0], strict=True
                )
            ]
        )

        assert [claim.start_anchor.coordinate_a for claim in claims] == [
            Coordinate(200, TimeUnit.samples),
            Coordinate(400, TimeUnit.samples),
            Coordinate(600, TimeUnit.samples),
        ]
        assert [claim.event_a_id for claim in claims] == [
            "projection_1",
            "projection_2",
            "projection_3",
        ]
        assert [event["id"] for event in source.events] == [
            "projection_1",
            "projection_2",
            "projection_3",
        ]
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
