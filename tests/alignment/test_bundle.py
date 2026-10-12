"""Tests for AlignmentBundle.

Tests cover core functionality:
- Timeline registration and lookup
- Group management
- Coordinate transfer within groups
- Order-independence (critical invariant)
- Cross-group transfer via MatchClaim -> MatchLine -> WarpMap pipeline
- get_matchstamp_at() propagation across groups
- are_commensurable() with claims
- WarpMap cache invalidation
- Indirect transfer (within-group convert + cross-group warp)
- Edge cases (insufficient claims, non-synchronous claims)

Per ZERO TOLERANCE policy, all assertions use exact expected values.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from timetoalign.alignment import (
    AlignmentAnchor,
    AlignmentBundle,
    MatchClaim,
    MatchClaimField,
)
from timetoalign.alignment.matchline import MatchLine
from timetoalign.core import (
    Coordinate,
    CoordinateField,
    IdCoordinate,
    IdCoordinateField,
    TimeUnit,
)
from timetoalign.maps import ScalarMap, TableMap
from timetoalign.timelines import Timeline, TimelineGroup

from .helpers import make_match_stamp as MatchStamp

# region Fixtures


@pytest.fixture
def simple_timeline() -> Timeline:
    """A simple 100-unit timeline."""
    return Timeline(length=100, uid="tl1", name="Simple Timeline")


@pytest.fixture
def second_timeline() -> Timeline:
    """A second 200-unit timeline."""
    return Timeline(length=200, uid="tl2", name="Second Timeline")


@pytest.fixture
def third_timeline() -> Timeline:
    """A third 50-unit timeline."""
    return Timeline(length=50, uid="tl3", name="Third Timeline")


# endregion


# region Test: Bundle Creation


class TestBundleCreation:
    """Tests for AlignmentBundle initialization."""

    def test_create_empty_bundle(self) -> None:
        """Empty bundle can be created."""
        bundle = AlignmentBundle()

        assert bundle.n_timelines == 0
        assert bundle.n_groups == 0
        assert bundle.id == "bundle_1"

    def test_create_bundle_with_name(self) -> None:
        """Bundle can be created with a name."""
        bundle = AlignmentBundle(name="Test Bundle")

        assert bundle.name == "Test Bundle"
        assert "Test Bundle" in repr(bundle)

    def test_create_bundle_with_explicit_id(self) -> None:
        """Bundle can be created with explicit ID."""
        bundle = AlignmentBundle(id="my_bundle")

        assert bundle.id == "my_bundle"


# endregion


# region Test: Timeline Management


class TestTimelineManagement:
    """Tests for adding and retrieving timelines."""

    def test_add_timeline_simple(self, simple_timeline: Timeline) -> None:
        """Can add a simple timeline."""
        bundle = AlignmentBundle()
        result = bundle.add_timeline(simple_timeline)

        # Returns self for chaining
        assert result is bundle

        # Timeline is stored
        assert bundle.n_timelines == 1
        assert "tl1" in bundle.timeline_ids

    def test_add_timeline_uses_timeline_id(self, simple_timeline: Timeline) -> None:
        """A timeline is registered under its own id."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        assert simple_timeline.id in bundle.timeline_ids

    def test_add_duplicate_timeline_raises(self, simple_timeline: Timeline) -> None:
        """Adding the same timeline twice raises ValueError."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        with pytest.raises(ValueError, match="already exists"):
            bundle.add_timeline(simple_timeline)

    def test_get_timeline(self, simple_timeline: Timeline) -> None:
        """Can retrieve timeline by ID."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        retrieved = bundle.get_timeline("tl1")
        assert retrieved is simple_timeline

    def test_get_timeline_not_found_raises(self) -> None:
        """Getting non-existent timeline raises KeyError."""
        bundle = AlignmentBundle()

        with pytest.raises(KeyError, match="No ID matches pattern"):
            bundle.get_timeline("nonexistent")

    def test_timeline_ids_property(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """timeline_ids returns all timeline IDs."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline)

        assert set(bundle.timeline_ids) == {"tl1", "tl2"}


# endregion


# region Test: Group Management


class TestGroupManagement:
    """Tests for creating and managing timeline groups."""

    def test_add_timeline_grouped_with_creates_group(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Adding timeline with grouped_with creates a group."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        assert bundle.n_groups == 1
        assert bundle.n_timelines == 2

    def test_group_contains_both_timelines(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Group contains both source and grouped timeline."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        group = bundle.default_group
        assert group is not None
        assert simple_timeline.id in group
        assert second_timeline.id in group

    def test_reference_timeline_is_first(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Group reference is the first timeline added."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        group = bundle.default_group
        assert group is not None
        assert simple_timeline.id in group.timeline_ids

    def test_add_timeline_to_nonexistent_raises(
        self, simple_timeline: Timeline
    ) -> None:
        """Aligning to non-existent timeline raises KeyError."""
        bundle = AlignmentBundle()

        with pytest.raises(KeyError, match="not in bundle"):
            bundle.add_timeline(simple_timeline, grouped_with="nonexistent")

    def test_add_timeline_as_group(self, simple_timeline: Timeline) -> None:
        """Can create timeline as reference of a named group."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline, as_group="my_group")

        assert bundle.group_ids == ["group_1"]
        group = bundle.get_group("group_1")
        assert group.name == "my_group"
        assert simple_timeline.id in group.timeline_ids

    def test_get_group_for_timeline(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Can get the group containing a timeline."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline, as_group="grp1")
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        group = bundle.get_group_for_timeline("tl2")
        assert group is not None
        assert group.id == "group_1"
        assert group.name == "grp1"

    def test_standalone_timeline_has_no_group(self, simple_timeline: Timeline) -> None:
        """Standalone timeline returns None for get_group_for_timeline."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        assert bundle.get_group_for_timeline("tl1") is None

    def test_add_child_timeline_brings_its_root(self) -> None:
        """A part of a timeline joins as a child, and its root joins with it.

        The section covers ``[25, 125]`` seconds of the reference: it is the
        reference's child at offset 25, so its 0 is the reference's 25 and its
        length 100 the reference's 125.
        """
        reference = Timeline(length=150.0, unit=TimeUnit.seconds, uid="reference")
        section = Timeline(length=100.0, unit=TimeUnit.seconds, uid="section")
        reference.add_child(section, offset=25.0)
        bundle = AlignmentBundle()
        bundle.add_timeline(section, as_group="part")

        assert bundle.timeline_ids == ["section", "reference"]
        assert bundle.timeline_to_group == {"section": "group_1"}
        assert bundle.get_matchstamp_at(0.0, "section").coordinates == {
            "section": Coordinate(0.0, TimeUnit.seconds),
            "reference": Coordinate(25.0, TimeUnit.seconds),
        }
        assert bundle.get_matchstamp_at(125.0, "reference").coordinates == {
            "reference": Coordinate(125.0, TimeUnit.seconds),
            "section": Coordinate(100.0, TimeUnit.seconds),
        }


# endregion


# region Test: Coordinate Transfer


class TestCoordinateTransfer:
    """Tests for coordinate transfer between timelines."""

    def test_transfer_same_timeline_returns_input(
        self, simple_timeline: Timeline
    ) -> None:
        """Transferring to same timeline returns input coordinate."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline, as_group="grp")

        result = bundle.transfer(50.0, "tl1", "tl1")
        assert result == 50.0

    def test_transfer_within_group_linear(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Transfer within group uses linear interpolation."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)  # length=100
        bundle.add_timeline(second_timeline, grouped_with="tl1")  # length=200

        # 50/100 = 0.5, so in tl2: 0.5 * 200 = 100
        result = bundle.transfer(50.0, "tl1", "tl2")
        assert result == 100.0

        # Reverse: 100/200 = 0.5, so in tl1: 0.5 * 100 = 50
        result_reverse = bundle.transfer(100.0, "tl2", "tl1")
        assert result_reverse == 50.0

    def test_transfer_boundary_values(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Transfer works correctly at boundaries."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)  # length=100
        bundle.add_timeline(second_timeline, grouped_with="tl1")  # length=200

        # Start: 0 -> 0
        assert bundle.transfer(0.0, "tl1", "tl2") == 0.0

        # End: 100 -> 200
        assert bundle.transfer(100.0, "tl1", "tl2") == 200.0

    def test_transfer_not_in_bundle_raises(self, simple_timeline: Timeline) -> None:
        """Transfer with non-existent timeline raises KeyError."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        with pytest.raises(KeyError, match="not in bundle"):
            bundle.transfer(50.0, "nonexistent", "tl1")

        with pytest.raises(KeyError, match="not in bundle"):
            bundle.transfer(50.0, "tl1", "nonexistent")

    def test_transfer_not_in_same_group_returns_none(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Transfer between different groups returns None."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline, as_group="grp1")
        bundle.add_timeline(second_timeline, as_group="grp2")

        result = bundle.transfer(50.0, "tl1", "tl2")
        assert result is None

    def test_transfer_interval(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Can transfer an interval."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)  # length=100
        bundle.add_timeline(second_timeline, grouped_with="tl1")  # length=200

        result = bundle.transfer_interval(25.0, 75.0, "tl1", "tl2")
        assert result is not None
        assert result == (50.0, 150.0)


# endregion


# region Test: Exact within-group transfer


def test_matchstamp_carries_group_members_exactly() -> None:
    """A bundle stamp converts within a group on exact ratios, not floats.

    Members of lengths 3 and 7 quarters: quarter 1/3 on the first is 7/9 on
    the second. A float lane would answer the dyadic of 0.777….
    """
    from timetoalign.timelines.types import ContinuousLogicalTimeline

    bundle = AlignmentBundle()
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=Fraction(3), uid="a"), as_group="pair"
    )
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=Fraction(7), uid="b"), grouped_with="a"
    )

    stamp = bundle.get_matchstamp_at(Fraction(1, 3), "a")

    assert stamp.get_coordinate_for("b", format="coordinate") == Coordinate(
        Fraction(7, 9), TimeUnit.quarters
    )
    assert type(stamp.get_coordinate_for("b").value) is Fraction


# endregion


# region Test: Custom Alignment


class TestCustomAlignment:
    """Tests for custom alignment specifications.

    A timeline that corresponds to part of another takes part as a child of
    it: the child carries the exact offset, the group relates whole members.
    """

    def test_partial_alignment_via_child(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """A part of a timeline is a child, and the child is the group member."""
        # second_timeline's full extent (0-200) corresponds to simple_timeline's
        # 0-50, which is the child covering that part.
        part = Timeline(length=50.0, unit=simple_timeline.unit, uid="tl1_part")
        simple_timeline.add_child(part, offset=0)
        group = TimelineGroup(id="test_group", timelines=[part, second_timeline])

        # At tl2 coord 50: 25% through tl2's range (0-200)
        # Should map to 25% through the part (0-50) = 12.5, which the part's
        # offset 0 places at 12.5 on tl1
        stamp = group.get_timestamp_at(
            IdCoordinate(50.0, second_timeline.unit, second_timeline.id)
        )
        assert stamp.get_coordinate_for(part.id, format="coordinate") == Coordinate(
            12.5, simple_timeline.unit
        )
        assert (
            simple_timeline._get_parent_coordinate_from_child(
                part.id, stamp.get_coordinate_for(part.id, format="float")
            )
            == 12.5
        )


# endregion


# region Test: Order Independence (CRITICAL)


class TestOrderIndependence:
    """Tests verifying order-independence of timeline addition.

    This is a CRITICAL invariant: the resulting bundle structure and
    coordinate transfer results must be identical regardless of the
    order in which timelines are added.
    """

    def test_two_timelines_order_1(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Adding tl1 first, then tl2 grouped with tl1."""
        bundle = AlignmentBundle(id="test_bundle")
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        # Transfer should work
        result = bundle.transfer(50.0, "tl1", "tl2")
        assert result == 100.0

    def test_three_timelines_any_order_same_result(
        self,
        simple_timeline: Timeline,
        second_timeline: Timeline,
        third_timeline: Timeline,
    ) -> None:
        """Three timelines added in different orders produce same transfer results."""
        # Order 1: tl1, tl2, tl3
        b1 = AlignmentBundle(id="b1")
        tl1_1 = Timeline(length=100, uid="tl1")
        tl2_1 = Timeline(length=200, uid="tl2")
        tl3_1 = Timeline(length=50, uid="tl3")
        b1.add_timeline(tl1_1)
        b1.add_timeline(tl2_1, grouped_with="tl1")
        b1.add_timeline(tl3_1, grouped_with="tl1")

        # Order 2: tl2, tl1, tl3 (tl2 first, then tl1 grouped with tl2)
        # Note: This changes the reference, so let's test with same reference
        b2 = AlignmentBundle(id="b2")
        tl1_2 = Timeline(length=100, uid="tl1")
        tl2_2 = Timeline(length=200, uid="tl2")
        tl3_2 = Timeline(length=50, uid="tl3")
        b2.add_timeline(tl1_2)
        b2.add_timeline(tl3_2, grouped_with="tl1")
        b2.add_timeline(tl2_2, grouped_with="tl1")

        # Both should give same transfer results
        # tl1 -> tl2: 50/100 * 200 = 100
        assert (
            b1.transfer(50.0, "tl1", "tl2") == b2.transfer(50.0, "tl1", "tl2") == 100.0
        )

        # tl1 -> tl3: 50/100 * 50 = 25
        assert (
            b1.transfer(50.0, "tl1", "tl3") == b2.transfer(50.0, "tl1", "tl3") == 25.0
        )

        # tl2 -> tl3: 100/200 = 0.5 * 50 = 25
        assert (
            b1.transfer(100.0, "tl2", "tl3") == b2.transfer(100.0, "tl2", "tl3") == 25.0
        )

    def test_summary_is_deterministic(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Summary output is deterministic (sorted keys)."""
        bundle = AlignmentBundle(id="test_bundle", name="Test")
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        summary = bundle.summary()

        # Check structure
        assert summary["id"] == "test_bundle"
        assert summary["name"] == "Test"
        assert summary["n_timelines"] == 2
        assert summary["n_groups"] == 1

        # Timeline keys should be sorted
        assert list(summary["timelines"].keys()) == ["tl1", "tl2"]


# endregion


# region Test: Commensurability


class TestCommensurability:
    """Tests for checking if timelines can be connected."""

    def test_same_timeline_is_commensurable(self, simple_timeline: Timeline) -> None:
        """A timeline is commensurable with itself."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)

        assert bundle.are_commensurable("tl1", "tl1") is True

    def test_same_group_is_commensurable(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Timelines in same group are commensurable."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline)
        bundle.add_timeline(second_timeline, grouped_with="tl1")

        assert bundle.are_commensurable("tl1", "tl2") is True

    def test_different_groups_not_commensurable(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """Timelines in different groups are not commensurable."""
        bundle = AlignmentBundle()
        bundle.add_timeline(simple_timeline, as_group="grp1")
        bundle.add_timeline(second_timeline, as_group="grp2")

        assert bundle.are_commensurable("tl1", "tl2") is False


# endregion


# region Test: Method Chaining


class TestMethodChaining:
    """Tests for fluent API (method chaining)."""

    def test_chained_add_timeline(
        self, simple_timeline: Timeline, second_timeline: Timeline
    ) -> None:
        """add_timeline returns self for chaining."""
        tl1 = Timeline(length=100, uid="tl1")
        tl2 = Timeline(length=200, uid="tl2")
        tl3 = Timeline(length=50, uid="tl3")

        bundle = (
            AlignmentBundle()
            .add_timeline(tl1)
            .add_timeline(tl2, grouped_with="tl1")
            .add_timeline(tl3, grouped_with="tl1")
        )

        assert bundle.n_timelines == 3
        assert bundle.n_groups == 1


# endregion


# region Helpers for Cross-Group Tests


def _make_cross_group_bundle() -> (
    tuple[AlignmentBundle, Timeline, Timeline, Timeline, Timeline]
):
    """Build a 2-group bundle with 2 timelines per group.

    Group A ("score_group"):
        - score_tl (length=200)   [id: "score"]
        - image_tl (length=400)   [id: "image", grouped with "score"]

    Group B ("recording_group"):
        - audio_tl (length=100)   [id: "audio"]
        - midi_tl  (length=100)   [id: "midi", grouped with "audio"]

    Returns:
        (bundle, score_tl, image_tl, audio_tl, midi_tl)
    """
    score_tl = Timeline(length=200, uid="score", name="Score")
    image_tl = Timeline(length=400, uid="image", name="Image")
    audio_tl = Timeline(length=100, uid="audio", name="Audio")
    midi_tl = Timeline(length=100, uid="midi", name="MIDI")

    bundle = AlignmentBundle(id="xgroup_test")
    bundle.add_timeline(score_tl, as_group="score_group")
    bundle.add_timeline(image_tl, grouped_with="score")
    bundle.add_timeline(audio_tl, as_group="recording_group")
    bundle.add_timeline(midi_tl, grouped_with="audio")

    return bundle, score_tl, image_tl, audio_tl, midi_tl


def _make_linear_claims(
    tl_a_id: str, tl_b_id: str, n_points: int = 5
) -> list[MatchClaim]:
    """Create n_points instant MatchClaims mapping [0..n-1] linearly.

    The mapping is: coord_b = coord_a * 0.5
    So tl_a range [0, n-1] maps to tl_b range [0, (n-1)*0.5].

    Uses the timeline IDs.
    """
    claims = []
    for i in range(n_points):
        coord_a = float(i * 50)  # 0, 50, 100, 150, 200
        coord_b = float(i * 25)  # 0, 25, 50, 75, 100
        claims.append(
            MatchClaim(
                timeline_a_id=tl_a_id,
                timeline_b_id=tl_b_id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=tl_a_id,
                    coordinate_a=Coordinate(coord_a, TimeUnit.number),
                    timeline_b_id=tl_b_id,
                    coordinate_b=Coordinate(coord_b, TimeUnit.number),
                ),
            )
        )
    return claims


def _make_claim_filter_bundle() -> tuple[
    AlignmentBundle,
    list[MatchClaim],
]:
    """Build a three-group bundle with one claim between each pair of groups."""
    score = Timeline(
        length=100.0,
        unit=TimeUnit.quarters,
        uid="score",
    )
    audio = Timeline(
        length=100.0,
        unit=TimeUnit.seconds,
        uid="audio",
    )
    other = Timeline(
        length=100.0,
        unit=TimeUnit.seconds,
        uid="other",
    )
    bundle = AlignmentBundle(id="claim_filters")
    bundle.add_timeline(score, as_group="score-group")
    bundle.add_timeline(audio, as_group="audio-group")
    bundle.add_timeline(other, as_group="other-group")

    def claim(timeline_a: Timeline, timeline_b: Timeline) -> MatchClaim:
        return MatchClaim(
            timeline_a_id=timeline_a.id,
            timeline_b_id=timeline_b.id,
            start_anchor=AlignmentAnchor(
                timeline_a_id=timeline_a.id,
                coordinate_a=Coordinate(0.0, timeline_a.unit),
                timeline_b_id=timeline_b.id,
                coordinate_b=Coordinate(0.0, timeline_b.unit),
            ),
        )

    claims = [claim(score, audio), claim(score, other), claim(audio, other)]
    bundle.add_match_claims(claims)
    return bundle, claims


# endregion


# region Test: Claim filters


class TestBundleClaimFilters:
    """Claim and stamp filters select by the bundle's timeline IDs."""

    def test_claim_selectors_select_by_timeline_id(self) -> None:
        """Every ID selector returns the claims pinned by timeline IDs."""
        bundle, claims = _make_claim_filter_bundle()

        assert bundle.get_match_claims(timeline_id="score") == claims[:2]
        assert bundle.get_match_claims(timeline_ids={"audio"}) == [
            claims[0],
            claims[2],
        ]
        assert bundle.get_match_claims(id_pattern=r"^audio$") == [
            claims[0],
            claims[2],
        ]
        assert bundle.get_match_claims(between=("score", "audio")) == [claims[0]]

    def test_claim_unit_filter_uses_timeline_metadata_lookup(self) -> None:
        """Unit filters resolve claim timeline IDs and reject impossible units."""
        bundle, claims = _make_claim_filter_bundle()

        assert bundle.get_match_claims(include_units={TimeUnit.seconds}) == [claims[2]]
        assert bundle.get_match_claims(include_units={TimeUnit.pixels}) == []

    def test_matchstamp_timeline_ids_filter_stamp_keys(self) -> None:
        """Post-hoc stamp filtering keeps exactly the selected timeline IDs."""
        bundle, _ = _make_claim_filter_bundle()

        stamp = bundle.get_matchstamp_at(
            0.0,
            "score",
            timeline_ids={"score", "audio"},
        )

        assert stamp.coordinates == {
            "score": Coordinate(Fraction(0, 1), TimeUnit.quarters),
            "audio": Coordinate(0.0, TimeUnit.seconds),
        }


# endregion


# region Test: Cross-Group Transfer


class TestCrossGroupTransfer:
    """Tests for cross-group transfer via MatchClaim -> MatchLine -> WarpMap."""

    def test_direct_cross_group_transfer(self) -> None:
        """Transfer between directly-claimed timelines in different groups."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        # Claims map score_tl.id -> audio_tl.id linearly: coord_b = coord_a * 0.5
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # score coord 100 -> audio coord 50 (linear interpolation)
        result = bundle.transfer(100.0, "score", "audio")
        assert result == 50.0

    def test_transfer_surfaces_ambiguous_warp_error(self) -> None:
        """Ambiguous source coordinates are not converted to a false value."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        line = MatchLine(
            source_timeline_id=score_tl.id,
            stamps=[
                MatchStamp(
                    coordinates={score_tl.id: 0.0, audio_tl.id: 0.0},
                    anchor_edges=[(score_tl.id, audio_tl.id)],
                ),
                MatchStamp(
                    coordinates={score_tl.id: 50.0, audio_tl.id: 0.0},
                    anchor_edges=[(score_tl.id, audio_tl.id)],
                ),
                MatchStamp(
                    coordinates={score_tl.id: 50.0, audio_tl.id: 22272.0},
                    anchor_edges=[(score_tl.id, audio_tl.id)],
                ),
            ],
        )
        bundle._matchline_cache[score_tl.id] = line
        bundle._cache_claims_hash = id(bundle.cross_group_claims) + 1
        bundle._warp_map_cache.clear()
        bundle.cross_group_claims.append(
            MatchClaim(
                timeline_a_id=score_tl.id,
                timeline_b_id=audio_tl.id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=score_tl.id,
                    coordinate_a=Coordinate(0.0, TimeUnit.number),
                    timeline_b_id=audio_tl.id,
                    coordinate_b=Coordinate(0.0, TimeUnit.number),
                ),
            )
        )

        with pytest.raises(
            ValueError,
            match=r"source timeline 'score'.*get_matchstamp_at\(\)",
        ):
            bundle.transfer(50.0, "score", "audio")

    def test_direct_cross_group_transfer_boundary_zero(self) -> None:
        """Transfer at coordinate 0 (boundary)."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        result = bundle.transfer(0.0, "score", "audio")
        assert result == 0.0

    def test_direct_cross_group_transfer_boundary_max(self) -> None:
        """Transfer at the maximum anchor coordinate."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # coord_a=200 -> coord_b=100
        result = bundle.transfer(200.0, "score", "audio")
        assert result == 100.0

    def test_cross_group_interpolation_midpoint(self) -> None:
        """Transfer at a coordinate between anchors uses linear interpolation."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # coord 75 is between anchors at 50 and 100 (mapped to 25 and 50)
        # Linear interpolation: 25 + (75-50)/(100-50) * (50-25) = 25 + 12.5 = 37.5
        result = bundle.transfer(75.0, "score", "audio")
        assert result == 37.5

    def test_cross_group_reverse_direction(self) -> None:
        """Transfer works in reverse (target -> source) via separate WarpMap."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # audio coord 50 -> score coord 100
        result = bundle.transfer(50.0, "audio", "score")
        assert result == 100.0

    def test_indirect_cross_group_transfer(self) -> None:
        """Transfer via indirect path: within-group convert then cross-group warp.

        Scenario: "image" is in same group as "score" (image = score * 2).
        Claims connect score.id -> audio.id. So image -> audio requires:
        1. Convert image coord in score_group to score coord (200/400 = 0.5 ratio)
        2. Warp score coord to audio coord via WarpMap (score * 0.5)
        """
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # image coord 200 -> score coord 100 (within group, 200/400 * 200 = 100)
        # score coord 100 -> audio coord 50 (warp, 100 * 0.5 = 50)
        result = bundle.transfer(200.0, "image", "audio")
        assert result == 50.0

    def test_indirect_cross_group_transfer_via_group_extension(self) -> None:
        """Transfer to a non-claimed target timeline via group extension.

        Scenario: Claims connect score.id -> audio.id. "midi" is in same
        group as "audio". MatchLine.from_claims() with group info extends
        the claims to include midi via the recording_group.
        """
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        # score coord 100 -> audio coord 50 (warp) -> midi coord 50
        # (midi is same length as audio, 1:1 within the recording_group)
        result = bundle.transfer(100.0, "score", "midi")
        assert result == 50.0

    def test_cross_group_no_claims_returns_none(self) -> None:
        """Transfer between unconnected groups returns None."""
        bundle, _, _, _, _ = _make_cross_group_bundle()

        # No claims added
        result = bundle.transfer(50.0, "score", "audio")
        assert result is None

    def test_cross_group_non_synchronous_claims_return_none(self) -> None:
        """Non-synchronous claims do not enable cross-group transfer."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        # Add non-synchronous claims (no anchors -> no WarpMap)
        claims = [
            MatchClaim.nomatch(
                event={"start": 50.0},
                source_tl_id=score_tl.id,
                target_tl_id=audio_tl.id,
                unit=TimeUnit.number,
            )
        ]
        bundle.add_match_claims(claims)

        result = bundle.transfer(50.0, "score", "audio")
        assert result is None

    def test_cross_group_single_claim_resolves_only_exact_anchor(self) -> None:
        """One claim resolves its exact edge but cannot interpolate a WarpMap."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        claims = [
            MatchClaim(
                timeline_a_id=score_tl.id,
                timeline_b_id=audio_tl.id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=score_tl.id,
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id=audio_tl.id,
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            )
        ]
        bundle.add_match_claims(claims)

        assert bundle.transfer(100.0, "score", "audio") == 50.0
        assert bundle.transfer(75.0, "score", "audio") is None

    def test_transfer_interval_cross_group(self) -> None:
        """transfer_interval works across groups."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        result = bundle.transfer_interval(50.0, 150.0, "score", "audio")
        assert result is not None
        assert result == (25.0, 75.0)


# endregion


# region Test: get_matchstamp_at Cross-Group


class TestGetMatchstampAtCrossGroup:
    """Tests for get_matchstamp_at() propagation across groups."""

    def test_timestamp_includes_source_group(self) -> None:
        """get_matchstamp_at returns source group timelines."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        stamp = bundle.get_matchstamp_at(100.0, "score")
        ts = stamp.to_dict(format="flat")

        # Source group timelines present (score + image)
        score_key = next(k for k in ts if k.startswith("score"))
        image_key = next(k for k in ts if k.startswith("image"))
        assert ts[score_key]["value"] == 100.0
        assert ts[score_key]["unit"] == "seconds"
        # image = score * 2 (200 length mapped to 400)
        assert ts[image_key]["value"] == 200.0
        assert ts[image_key]["unit"] == "seconds"

    def test_timestamp_includes_target_group(self) -> None:
        """get_matchstamp_at propagates to connected groups."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        stamp = bundle.get_matchstamp_at(100.0, "score")
        ts = stamp.to_dict(format="flat")

        # Target group timelines should be present
        audio_key = next((k for k in ts if k.startswith("audio")), None)
        assert audio_key is not None
        assert ts[audio_key]["value"] == 50.0

    def test_interpolated_stamp_records_all_group_edges(self) -> None:
        """Interpolation records every materialized group relationship."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claims(_make_linear_claims(score_tl.id, audio_tl.id))

        stamp = bundle.get_matchstamp_at(75.0, "score")

        assert stamp.inferred_edges == [
            ("score", "image"),
            ("score", "audio"),
            ("audio", "midi"),
        ]

    def test_cached_graph_matchstamp_carries_bundle_units(self) -> None:
        """Stamps returned directly by a bundle-owned graph carry units."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claims(_make_linear_claims(score_tl.id, audio_tl.id))

        graph = bundle._get_or_build_matchgraph(score_tl.id, 50.0)
        stamp = graph.get_matchstamp()

        assert stamp.get_coordinate(score_tl.id, format="coordinate") == Coordinate(
            50.0, score_tl.unit
        )
        assert stamp.get_coordinate(audio_tl.id, format="coordinate") == Coordinate(
            25.0, audio_tl.unit
        )

    def test_timestamp_nested_format(self) -> None:
        """MatchStamp nested format groups by group_id."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        ts = bundle.get_matchstamp_at(100.0, "score").to_dict(format="nested")

        # Must have source group; groups are keyed by their minted ids
        assert "group_1" in ts
        assert "group_2" in ts
        recording_vals = ts["group_2"]
        audio_key = next(k for k in recording_vals if k.startswith("audio"))
        assert recording_vals[audio_key]["value"] == 50.0

    def test_timestamp_prefix_format(self) -> None:
        """MatchStamp prefix format uses group/timeline keys."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=5)
        bundle.add_match_claims(claims)

        ts = bundle.get_matchstamp_at(100.0, "score").to_dict(format="prefix")

        # Keys should be "group_id/timeline_id" format
        score_keys = [k for k in ts if k.startswith("group_1/")]
        assert len(score_keys) >= 1  # at least "score"

    def test_timestamp_no_claims_source_only(self) -> None:
        """Without claims, get_matchstamp_at returns only source group."""
        bundle, _, _, _, _ = _make_cross_group_bundle()

        ts = bundle.get_matchstamp_at(100.0, "score").to_dict(format="nested")

        assert "group_1" in ts
        assert "group_2" not in ts


# endregion


# region Test: are_commensurable with Claims


class TestCommensurabilityWithClaims:
    """Tests for are_commensurable() when cross-group claims exist."""

    def test_commensurable_with_direct_claims(self) -> None:
        """Two timelines connected by direct claims are commensurable."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        assert bundle.are_commensurable("score", "audio") is True

    def test_commensurable_via_group_membership(self) -> None:
        """Timeline in same group as a claimed timeline is commensurable.

        Claims connect score.id <-> audio.id. "image" is in score's group.
        The claim's anchor touches score_tl.id which is in score_group,
        and audio_tl.id which is in recording_group. So "image" (in score_group)
        should be commensurable with "audio" (in recording_group).
        """
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        # image is in score_group, audio is in recording_group
        # Claims connect score_tl.id (in score_group) to audio_tl.id (in recording_group)
        assert bundle.are_commensurable("image", "audio") is True

    def test_not_commensurable_without_claims(self) -> None:
        """Without claims, timelines in different groups are not commensurable."""
        bundle, _, _, _, _ = _make_cross_group_bundle()

        assert bundle.are_commensurable("score", "audio") is False

    def test_not_commensurable_non_synchronous_only(self) -> None:
        """Non-synchronous claims don't make timelines commensurable."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        claims = [
            MatchClaim.nomatch(
                event={"start": 50.0},
                source_tl_id=score_tl.id,
                target_tl_id=audio_tl.id,
                unit=TimeUnit.number,
            )
        ]
        bundle.add_match_claims(claims)

        assert bundle.are_commensurable("score", "audio") is False


# endregion


# region Test: Cache Invalidation


class TestCacheInvalidation:
    """Tests for WarpMap cache invalidation on add_match_claims()."""

    def test_cache_populated_on_first_transfer(self) -> None:
        """First transfer builds WarpMap and populates cache."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        assert len(bundle._warp_map_cache) == 0  # No cache yet

        bundle.transfer(50.0, "score", "audio")

        assert len(bundle._warp_map_cache) >= 1  # Cache populated

    def test_cache_reused_on_second_transfer(self) -> None:
        """Second transfer reuses cached WarpMap (no rebuild)."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        result1 = bundle.transfer(50.0, "score", "audio")
        cache_after_first = dict(bundle._warp_map_cache)

        result2 = bundle.transfer(50.0, "score", "audio")

        assert result1 == result2
        # Same WarpMap objects should be in cache
        for key in cache_after_first:
            if key in bundle._warp_map_cache:
                assert bundle._warp_map_cache[key] is cache_after_first[key]

    def test_cache_invalidated_on_new_claims(self) -> None:
        """Adding new claims invalidates the cache."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        bundle.transfer(50.0, "score", "audio")
        assert len(bundle._warp_map_cache) >= 1

        # Add more claims -> invalidates cache
        more_claims = [
            MatchClaim(
                timeline_a_id=score_tl.id,
                timeline_b_id=audio_tl.id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=score_tl.id,
                    coordinate_a=Coordinate(175.0, TimeUnit.number),
                    timeline_b_id=audio_tl.id,
                    coordinate_b=Coordinate(87.5, TimeUnit.number),
                ),
            )
        ]
        bundle.add_match_claims(more_claims)

        assert len(bundle._warp_map_cache) == 0  # Cache cleared

    def test_transfer_after_invalidation_rebuilds(self) -> None:
        """Transfer after cache invalidation builds a new WarpMap."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        # Start with 3 claims: linear mapping coord_b = coord_a * 0.5
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        result1 = bundle.transfer(100.0, "score", "audio")
        assert result1 == 50.0

        # Add an additional claim that is consistent with the linear mapping
        more_claims = [
            MatchClaim(
                timeline_a_id=score_tl.id,
                timeline_b_id=audio_tl.id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=score_tl.id,
                    coordinate_a=Coordinate(150.0, TimeUnit.number),
                    timeline_b_id=audio_tl.id,
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            )
        ]
        bundle.add_match_claims(more_claims)

        # Transfer should still work after rebuild
        result2 = bundle.transfer(100.0, "score", "audio")
        assert result2 == 50.0


# endregion


# region Test: add_match_claims API


class TestAddMatchClaims:
    """Tests for the add_match_claims() method."""

    def test_returns_self_for_chaining(self) -> None:
        """add_match_claims returns self for method chaining."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)

        result = bundle.add_match_claims(claims)
        assert result is bundle

    def test_claims_stored_in_cross_group_claims(self) -> None:
        """Claims are stored in cross_group_claims list."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims)

        assert len(bundle.cross_group_claims) == 3

    def test_multiple_add_claims_accumulate(self) -> None:
        """Multiple add_match_claims calls accumulate claims."""
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()

        claims1 = _make_linear_claims(score_tl.id, audio_tl.id, n_points=3)
        bundle.add_match_claims(claims1)
        assert len(bundle.cross_group_claims) == 3

        claims2 = [
            MatchClaim(
                timeline_a_id=score_tl.id,
                timeline_b_id=audio_tl.id,
                start_anchor=AlignmentAnchor(
                    timeline_a_id=score_tl.id,
                    coordinate_a=Coordinate(150.0, TimeUnit.number),
                    timeline_b_id=audio_tl.id,
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            )
        ]
        bundle.add_match_claims(claims2)
        assert len(bundle.cross_group_claims) == 4


# endregion


# region Test: create_match_claims agent_identifier


class TestCreateMatchClaims:
    """Tests for create_match_claims() and the agent_identifier keyword."""

    def test_agent_identifier_lands_in_agent_identifier(self) -> None:
        # The public keyword is agent_identifier (renamed from
        # decision_criteria); its value lands in Agent.identifier.
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        claims = bundle.create_match_claims(
            [({"start": 0.0}, score_tl.id, {"start": 10.0}, audio_tl.id)],
            agent="dtw",
            agent_identifier="dynamic_time_warping",
        )
        assert len(claims) == 1
        meta = claims[0].metadata
        assert meta is not None
        assert meta.agent.name == "dtw"
        assert meta.agent.identifier == "dynamic_time_warping"

    def test_no_decision_criteria_keyword(self) -> None:
        # The old keyword is gone — no compat shim.
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        with pytest.raises(TypeError):
            bundle.create_match_claims(
                [({"start": 0.0}, score_tl.id, {"start": 10.0}, audio_tl.id)],
                decision_criteria="manual",
            )


# endregion


# region Test: add_match_claim_field (columnar) API


class TestAddMatchClaimField:
    """Tests for the columnar add_match_claim_field() path."""

    def _field(self, tl_a_id: str, tl_b_id: str) -> MatchClaimField:
        """A 5-row instant claim field: coord_b = coord_a * 0.5."""
        return MatchClaimField.from_columns(
            timeline_a_ids=[tl_a_id] * 5,
            timeline_b_ids=[tl_b_id] * 5,
            coordinate_a=[0.0, 50.0, 100.0, 150.0, 200.0],
            coordinate_b=[0.0, 25.0, 50.0, 75.0, 100.0],
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

    def test_returns_self_for_chaining(self) -> None:
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        field = self._field(score_tl.id, audio_tl.id)
        assert bundle.add_match_claim_field(field) is bundle

    def test_field_stored_columnar_not_in_list(self) -> None:
        # The field goes to the columnar store; the per-claim Python list is
        # left untouched (never exploded).
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claim_field(self._field(score_tl.id, audio_tl.id))
        assert len(bundle.cross_group_claims) == 0
        assert len(bundle.cross_group_claim_fields) == 1
        assert len(bundle.cross_group_claim_fields[0]) == 5

    def test_matchstamp_via_columnar_field(self) -> None:
        # get_matchstamp_at answers from the columnar field at an exact grid
        # coordinate, without touching the Python claim list.
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claim_field(self._field(score_tl.id, audio_tl.id))
        stamp = bundle.get_matchstamp_at(100.0, score_tl.id)
        assert stamp.get_coordinate_for("score", format="float") == 100.0
        assert stamp.get_coordinate_for("audio", format="float") == 50.0
        assert len(bundle.cross_group_claims) == 0

    def test_inexact_coordinate_returns_interpolated_source_stamp(self) -> None:
        # Columnar claims are exact-match queried; without a list-backed
        # MatchLine, the fallback still returns the source-group cross-section.
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claim_field(self._field(score_tl.id, audio_tl.id))
        stamp = bundle.get_matchstamp_at(123.456, score_tl.id)

        assert stamp.is_interpolated is True
        assert stamp.get_coordinate_for("score", format="float") == 123.456

    def test_list_and_field_stores_coexist(self) -> None:
        # A bundle may hold a Python-list claim AND a columnar field at once;
        # a query reads both stores.
        bundle, score_tl, _, audio_tl, _ = _make_cross_group_bundle()
        bundle.add_match_claims(_make_linear_claims(score_tl.id, audio_tl.id, 5))
        bundle.add_match_claim_field(self._field(score_tl.id, audio_tl.id))
        assert len(bundle.cross_group_claims) == 5
        assert len(bundle.cross_group_claim_fields) == 1
        stamp = bundle.get_matchstamp_at(50.0, score_tl.id)
        assert stamp.get_coordinate_for("audio", format="float") == 25.0


# endregion


# region Test: add_group API


class TestAddGroup:
    """Tests for the add_group() method."""

    def test_add_prebuilt_group(self) -> None:
        """Can add a pre-built TimelineGroup with all its timelines."""
        tl1 = Timeline(length=100, uid="g1")
        tl2 = Timeline(length=200, uid="g2")

        group = TimelineGroup(id="my_group", timelines=[tl1, tl2])

        bundle = AlignmentBundle(id="test")
        bundle.add_group(group)

        assert bundle.n_groups == 1
        assert bundle.n_timelines == 2
        assert tl1.id in bundle.timeline_ids
        assert tl2.id in bundle.timeline_ids

    def test_add_group_duplicate_id_raises(self) -> None:
        """Adding group with duplicate ID raises ValueError."""
        tl1 = Timeline(length=100, uid="g1")
        group = TimelineGroup(id="my_group", timelines=[tl1])

        bundle = AlignmentBundle(id="test")
        bundle.add_group(group)

        tl2 = Timeline(length=200, uid="g2")
        group2 = TimelineGroup(id="my_group", timelines=[tl2])
        with pytest.raises(ValueError, match="already exists"):
            bundle.add_group(group2)

    def test_add_group_cross_group_transfer(self) -> None:
        """Cross-group transfer works with add_group API."""
        tl_score = Timeline(length=200, uid="s1")
        tl_audio = Timeline(length=100, uid="a1")

        grp_score = TimelineGroup(id="score", timelines=[tl_score])
        grp_audio = TimelineGroup(id="audio", timelines=[tl_audio])

        bundle = AlignmentBundle(id="test")
        bundle.add_group(grp_score)
        bundle.add_group(grp_audio)

        claims = _make_linear_claims(tl_score.id, tl_audio.id, n_points=5)
        bundle.add_match_claims(claims)

        result = bundle.transfer(100.0, tl_score.id, tl_audio.id)
        assert result == 50.0


# endregion


# region Test: Coordinate-type parity (raw / Coordinate / IdCoordinate)


def _make_xgroup_bundle_with_claims() -> tuple[AlignmentBundle, str, str]:
    """Cross-group bundle with linear claims; returns (bundle, score_id, audio_id).

    Reuses the module-level ``_make_cross_group_bundle`` + ``_make_linear_claims``
    helpers. Claims map score -> audio linearly (audio = score * 0.5). The
    score and audio timelines are seconds.

    The returned ``score_id`` / ``audio_id`` are the timeline IDs accepted as
    source selectors and used as stamp result axes.
    """
    bundle, score_tl, _image_tl, audio_tl, _midi_tl = _make_cross_group_bundle()
    bundle.add_match_claims(_make_linear_claims(score_tl.id, audio_tl.id, n_points=5))
    return bundle, score_tl.id, audio_tl.id


class TestGetMatchstampAtCoordinateParity:
    """get_matchstamp_at accepts raw value, Coordinate, and IdCoordinate."""

    def test_raw_float_form(self) -> None:
        """Raw float with explicit timeline_id returns the known stamp."""
        bundle, score_id, audio_id = _make_xgroup_bundle_with_claims()
        stamp = bundle.get_matchstamp_at(100.0, score_id)
        # The stamp is the transitive cross-group union: the exact anchor pair
        # (score, audio) plus each reached group's cross-section
        # (image in score's group, midi in audio's group).
        assert stamp.n_timelines == 4
        assert stamp.get_coordinate_for("score", format="float") == 100.0
        assert stamp.get_coordinate_for("audio", format="float") == 50.0
        assert stamp.get_coordinate_for("image", format="float") == 200.0
        assert stamp.get_coordinate_for("midi", format="float") == 50.0

    def test_coordinate_form_equals_raw(self) -> None:
        """A Coordinate with explicit timeline_id matches the raw-float result."""
        bundle, score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import Coordinate
        from timetoalign.core.enums import TimeUnit

        raw = bundle.get_matchstamp_at(100.0, score_id)
        from_coord = bundle.get_matchstamp_at(
            Coordinate(100.0, TimeUnit.seconds), score_id
        )
        assert from_coord.coordinates == raw.coordinates

    def test_idcoordinate_alone_equals_raw(self) -> None:
        """An IdCoordinate alone (timeline_id omitted) matches the raw result."""
        bundle, score_id, audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import IdCoordinate
        from timetoalign.core.enums import TimeUnit

        raw = bundle.get_matchstamp_at(100.0, score_id)
        from_id = bundle.get_matchstamp_at(
            IdCoordinate(100.0, TimeUnit.seconds, score_id)
        )
        assert from_id.coordinates == raw.coordinates
        assert from_id.get_coordinate_for("score", format="float") == 100.0
        assert from_id.get_coordinate_for("audio", format="float") == 50.0

    def test_missing_timeline_id_raises_value_error(self) -> None:
        """Raw float without timeline_id raises ValueError."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        with pytest.raises(ValueError, match="timeline_id is required"):
            bundle.get_matchstamp_at(100.0)

    def test_unsupported_type_raises_type_error(self) -> None:
        """A non-coordinate type raises TypeError."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        with pytest.raises(TypeError, match="Unsupported coordinate specification"):
            bundle.get_matchstamp_at("not-a-coordinate")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Unsupported coordinate specification"):
            bundle.get_matchstamp_at([1.0])  # type: ignore[arg-type]

    def test_foreign_unit_without_cmap_raises_value_error(self) -> None:
        """A unit without a source-timeline C-Map is never dropped."""
        bundle, score_id, _audio_id = _make_xgroup_bundle_with_claims()

        with pytest.raises(ValueError, match="quarters.*seconds"):
            bundle.get_matchstamp_at(Coordinate(100.0, TimeUnit.quarters), score_id)

    def test_foreign_unit_with_cmap_matches_native_stamp(self) -> None:
        """A C-Map-backed coordinate resolves before the graph lookup."""
        source = Timeline(length=10, unit=TimeUnit.seconds, uid="source")
        target = Timeline(length=10, unit=TimeUnit.seconds, uid="target")
        source.add_conversion_map(
            ScalarMap(
                scalar=1000,
                source_unit=TimeUnit.seconds,
                target_unit=TimeUnit.milliseconds,
            )
        )
        bundle = AlignmentBundle(id="cmap-query")
        bundle.add_timeline(source, as_group="source-group")
        bundle.add_timeline(target, as_group="target-group")
        bundle.add_match_claims(
            [
                MatchClaim(
                    timeline_a_id=source.id,
                    timeline_b_id=target.id,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id=source.id,
                        coordinate_a=Coordinate(2.5, TimeUnit.seconds),
                        timeline_b_id=target.id,
                        coordinate_b=Coordinate(5.0, TimeUnit.seconds),
                    ),
                )
            ]
        )

        native = bundle.get_matchstamp_at(2.5, "source")
        converted = bundle.get_matchstamp_at(
            Coordinate(2500, TimeUnit.milliseconds), "source"
        )
        assert converted.coordinates == native.coordinates

    def test_idcoordinate_conflicting_timeline_id_raises_value_error(self) -> None:
        """An explicit timeline ID cannot conflict with an IdCoordinate."""
        bundle, score_id, audio_id = _make_xgroup_bundle_with_claims()

        with pytest.raises(ValueError, match="conflicts"):
            bundle.get_matchstamp_at(
                IdCoordinate(100.0, TimeUnit.seconds, audio_id), score_id
            )


class TestGetMatchstampRenderingCoordinateParity:
    """get_matchstamp_at accepts raw value, Coordinate, and IdCoordinate.

    ``get_matchstamp_at`` is keyed on the timeline ID (``"score"``), so these
    tests pass IDs and compare whole-dict results across the three forms.
    """

    def test_raw_float_form(self) -> None:
        """Raw float with explicit timeline_id returns the known timestamp."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        stamp = bundle.get_matchstamp_at(100.0, "score")
        ts = stamp.to_dict(format="flat")
        score_key = next(k for k in ts if k.startswith("score"))
        audio_key = next(k for k in ts if k.startswith("audio"))
        assert ts[score_key]["value"] == 100.0
        assert ts[audio_key]["value"] == 50.0

    def test_coordinate_form_equals_raw(self) -> None:
        """A Coordinate with explicit timeline_id matches the raw-float result."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import Coordinate
        from timetoalign.core.enums import TimeUnit

        raw = bundle.get_matchstamp_at(100.0, "score").to_dict(format="flat")
        from_coord = bundle.get_matchstamp_at(
            Coordinate(100.0, TimeUnit.seconds), "score"
        ).to_dict(format="flat")
        assert from_coord == raw

    def test_idcoordinate_alone_equals_raw(self) -> None:
        """An IdCoordinate alone (timeline_id omitted) matches the raw result.

        The IdCoordinate carries the timeline ID ``"score"`` so the omitted
        ``timeline_id`` is resolved from it.
        """
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import IdCoordinate
        from timetoalign.core.enums import TimeUnit

        raw = bundle.get_matchstamp_at(100.0, "score").to_dict(format="flat")
        from_id = bundle.get_matchstamp_at(
            IdCoordinate(100.0, TimeUnit.seconds, "score")
        ).to_dict(format="flat")
        assert from_id == raw

    def test_missing_timeline_id_raises_value_error(self) -> None:
        """Raw float without timeline_id raises ValueError."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        with pytest.raises(ValueError, match="timeline_id is required"):
            bundle.get_matchstamp_at(100.0).to_dict(format="flat")

    def test_unsupported_type_raises_type_error(self) -> None:
        """A non-coordinate type raises TypeError."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        with pytest.raises(TypeError, match="Unsupported coordinate specification"):
            bundle.get_matchstamp_at({"start": 1.0}).to_dict(  # type: ignore[arg-type]
                format="flat"
            )


class TestTransferCoordinateParity:
    """transfer / transfer_interval accept raw value, Coordinate, IdCoordinate.

    ``transfer`` is keyed on the timeline IDs (``"score"`` / ``"audio"``).
    """

    def test_transfer_accepts_coordinate(self) -> None:
        """transfer resolves a Coordinate in the source timeline's native unit."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import Coordinate
        from timetoalign.core.enums import TimeUnit

        assert bundle.transfer(100.0, "score", "audio") == 50.0
        assert (
            bundle.transfer(Coordinate(100.0, TimeUnit.seconds), "score", "audio")
            == 50.0
        )

    def test_transfer_accepts_idcoordinate_value_only(self) -> None:
        """transfer resolves an IdCoordinate in the named source timeline."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import IdCoordinate
        from timetoalign.core.enums import TimeUnit

        result = bundle.transfer(
            IdCoordinate(100.0, TimeUnit.seconds, "score"), "score", "audio"
        )
        assert result == 50.0

    def test_transfer_unsupported_type_raises(self) -> None:
        """transfer rejects non-coordinate types."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        with pytest.raises(TypeError, match="Unsupported coordinate specification"):
            bundle.transfer("x", "score", "audio")  # type: ignore[arg-type]

    def test_transfer_interval_accepts_coordinates(self) -> None:
        """transfer_interval resolves Coordinate endpoints before transfer."""
        bundle, _score_id, _audio_id = _make_xgroup_bundle_with_claims()
        from timetoalign.core import Coordinate
        from timetoalign.core.enums import TimeUnit

        raw = bundle.transfer_interval(0.0, 200.0, "score", "audio")
        assert raw == (0.0, 100.0)
        from_coord = bundle.transfer_interval(
            Coordinate(0.0, TimeUnit.seconds),
            Coordinate(200.0, TimeUnit.seconds),
            "score",
            "audio",
        )
        assert from_coord == (0.0, 100.0)


# endregion


# region Test: get_matchstamp_table Conversion Columns


def _clock_bundle_with_maps() -> AlignmentBundle:
    """A single ``clock`` timeline with the same exact-value maps used to pin
    ``MatchStamp`` conversion-row display in ``tests/core/test_stamp_interface.py``.
    """
    timeline = Timeline(length=100, unit=TimeUnit.seconds, uid="clock")
    timeline.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 100000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="clock-ms",
        )
    )
    timeline.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 5000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.frames,
            uid="clock-frames",
        )
    )
    bundle = AlignmentBundle(id="matchstamp-table-conversions")
    bundle.add_timeline(timeline, as_group="clock-group")
    return bundle


def _cross_group_millisecond_bundle() -> AlignmentBundle:
    """Two single-timeline groups, each with its own seconds->milliseconds
    map, linked by one cross-group synchronous claim at 25 seconds.
    """
    tl_a = Timeline(length=100, unit=TimeUnit.seconds, uid="clock_a")
    tl_a.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 100000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="clock_a-ms",
        )
    )
    tl_b = Timeline(length=100, unit=TimeUnit.seconds, uid="clock_b")
    tl_b.add_conversion_map(
        TableMap(
            x_values=[0.0, 100.0],
            y_values=[0.0, 100000.0],
            source_unit=TimeUnit.seconds,
            target_unit=TimeUnit.milliseconds,
            uid="clock_b-ms",
        )
    )
    bundle = AlignmentBundle(id="matchstamp-table-collision")
    bundle.add_timeline(tl_a, as_group="group-a")
    bundle.add_timeline(tl_b, as_group="group-b")
    bundle.add_match_claims(
        [
            MatchClaim(
                timeline_a_id="clock_a",
                timeline_b_id="clock_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="clock_a",
                    coordinate_a=Coordinate(25.0, TimeUnit.seconds),
                    timeline_b_id="clock_b",
                    coordinate_b=Coordinate(25.0, TimeUnit.seconds),
                ),
            )
        ]
    )
    return bundle


class TestGetMatchstampTableConversionColumns:
    """``get_matchstamp_table(conversion_maps=...)`` derived unit columns."""

    def test_matchstamp_table_adds_conversion_columns(self) -> None:
        """Enabling conversion_maps adds one column per numeric unit map."""
        bundle = _clock_bundle_with_maps()

        table = bundle.get_matchstamp_table([25.0], "clock", conversion_maps=True)

        assert set(table.column_names) == {"clock", "milliseconds", "frames"}
        assert table.num_rows == 1
        assert IdCoordinateField.from_table(table, "clock")[0] == IdCoordinate(
            25.0, TimeUnit.seconds, "clock"
        )
        assert CoordinateField.from_table(table, "milliseconds")[0] == Coordinate(
            25000.0, TimeUnit.milliseconds
        )
        assert CoordinateField.from_table(table, "frames")[0] == Coordinate(
            1250.0, TimeUnit.frames
        )

    def test_matchstamp_table_no_conversion_columns_by_default(self) -> None:
        """Without conversion_maps, only the timeline column is present."""
        bundle = _clock_bundle_with_maps()

        table = bundle.get_matchstamp_table([25.0], "clock")

        assert table.column_names == ["clock"]

    def test_matchstamp_table_conversion_column_collision_qualified(self) -> None:
        """Two timelines converting to the same unit get qualified column names."""
        bundle = _cross_group_millisecond_bundle()

        table = bundle.get_matchstamp_table(conversion_maps=True)

        assert table.num_rows == 1
        assert "milliseconds" not in table.column_names
        assert "clock_a:milliseconds" in table.column_names
        assert "clock_b:milliseconds" in table.column_names
        assert CoordinateField.from_table(table, "clock_a:milliseconds")[
            0
        ] == Coordinate(25000.0, TimeUnit.milliseconds)
        assert CoordinateField.from_table(table, "clock_b:milliseconds")[
            0
        ] == Coordinate(25000.0, TimeUnit.milliseconds)


# endregion
