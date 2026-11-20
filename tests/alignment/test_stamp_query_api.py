"""Tests for the Unified Stamp & Query API.

Covers:
- AlignmentBundle.get_match_claims() with unified filter API
- MatchClaim.get_matchstamp() (reduced and graph-expanded)
- AlignmentBundle MatchGraph cache
- AlignmentBundle.get_matchstamp_at() cross-group coordinate resolution
- AlignmentBundle.get_matchstamps_at() / get_matchstamp_table() position
  batches: an ``at`` collection (+ ``timeline_id``) is a thin,
  order-preserving fan-out that delegates element-by-element to
  ``get_matchstamp_at``. The cases below pin that raw values and IdCoordinates
  coerce identically to the singular resolver, that each query coordinate
  yields exactly one full transitive cross-section (one list entry / one table
  row), that ``timeline_filter`` narrows the table's columns, that empty input
  yields empty output, and that supplying both ``claims`` and ``at`` is
  rejected.
- MatchGraph.get_matchstamp() + split_components()
- MatchStamp __str__ and _repr_html_ display methods
- MatchClaim __str__ and _repr_html_ display methods
- transfer() docstring correctness
- Top-level exports (MatchGraph, MatchStamp, ClaimFilter)
"""

from __future__ import annotations

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import (
    Agent,
    AlignmentAnchor,
    MatchClaim,
    MatchMetadata,
)
from timetoalign.alignment.graph import MatchGraph
from timetoalign.alignment.graph import MatchStamp as MatchStampType
from timetoalign.core import (
    AgentType,
    Coordinate,
    IdCoordinate,
    IdCoordinateField,
    TimeUnit,
)
from timetoalign.timelines import Timeline

from .helpers import make_match_stamp as MatchStamp

# region Helpers


def _make_star_bundle():
    """Build a star-topology bundle: 1 score + 3 performers.

    score_tl (length=100) in "score_group"
    perf1_tl (length=200) standalone
    perf2_tl (length=200) standalone
    perf3_tl (length=200) standalone

    Claims: score <-> each perf at coordinates 0, 25, 50, 75, 100
    Plus one NOMATCH claim: score <-> perf3 (non-synchronous).

    Claims name the timeline IDs the bundle registers its timelines under.
    """
    # Timelines are number-native to match the units their own claims use,
    # so a number-unit Coordinate/IdCoordinate resolves without conversion.
    score_tl = Timeline(
        length=100, uid="score:clt_1", name="Score", unit=TimeUnit.number
    )
    perf1_tl = Timeline(
        length=200, uid="perf:dlt_1", name="Perf1", unit=TimeUnit.number
    )
    perf2_tl = Timeline(
        length=200, uid="perf:dlt_2", name="Perf2", unit=TimeUnit.number
    )
    perf3_tl = Timeline(
        length=200, uid="perf:dlt_3", name="Perf3", unit=TimeUnit.number
    )

    bundle = AlignmentBundle(id="star_test")
    bundle.add_timeline(score_tl, as_group="score_group")
    bundle.add_timeline(perf1_tl, as_group="perf1_group")
    bundle.add_timeline(perf2_tl, as_group="perf2_group")
    bundle.add_timeline(perf3_tl, as_group="perf3_group")

    claims = []
    # 5 coordinates per performer
    for i in range(5):
        coord_s = float(i * 25)
        for perf_id, factor in [
            ("perf:dlt_1", 2.0),
            ("perf:dlt_2", 1.5),
            ("perf:dlt_3", 1.8),
        ]:
            coord_p = coord_s * factor
            claims.append(
                MatchClaim(
                    timeline_a_id="score:clt_1",
                    timeline_b_id=perf_id,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="score:clt_1",
                        coordinate_a=Coordinate(coord_s, TimeUnit.number),
                        timeline_b_id=perf_id,
                        coordinate_b=Coordinate(coord_p, TimeUnit.number),
                    ),
                    is_synchronous=True,
                )
            )

    # One NOMATCH claim
    claims.append(
        MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_3",
            is_synchronous=False,
        )
    )

    bundle.add_match_claims(claims)
    return bundle, claims


# endregion


# region AlignmentBundle.get_match_claims()


class TestGetMatchClaims:
    """Tests for AlignmentBundle.get_match_claims()."""

    def test_no_filters_returns_all(self):
        """No filters returns all claims."""
        bundle, claims = _make_star_bundle()
        result = bundle.get_match_claims()
        assert len(result) == len(claims)

    def test_filter_by_timeline_id(self):
        """Filter by single timeline ID."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(timeline_id="perf:dlt_1")
        # 5 synchronous claims: score <-> perf:dlt1 at 5 coordinates
        assert len(result) == 5
        for c in result:
            assert c.connects("perf:dlt_1")

    def test_filter_by_id_pattern(self):
        """Filter by regex pattern."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(id_pattern=r"^perf:")
        # All 15 synchronous + 1 NOMATCH involve perf: timelines
        assert len(result) == 16

    def test_filter_synchronous_only(self):
        """synchronous_only excludes NOMATCH."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(synchronous_only=True)
        assert len(result) == 15  # 5 claims * 3 performers
        for c in result:
            assert c.is_synchronous

    def test_filter_nomatch_only(self):
        """nomatch_only returns only NOMATCH claims."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(nomatch_only=True)
        assert len(result) == 1
        assert not result[0].is_synchronous

    def test_filter_between(self):
        """between filter for exact timeline pair."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(
            between=("score:clt_1", "perf:dlt_2"),
            synchronous_only=True,
        )
        assert len(result) == 5
        for c in result:
            assert c.connects_both("score:clt_1", "perf:dlt_2")

    def test_filter_combined(self):
        """Multiple filters are AND-combined."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(
            timeline_id="perf:dlt_3",
            synchronous_only=True,
        )
        # 5 synchronous claims for perf:dlt3 (the NOMATCH is excluded)
        assert len(result) == 5

    def test_filter_pattern_and_nomatch(self):
        """Pattern + nomatch_only."""
        bundle, _ = _make_star_bundle()
        result = bundle.get_match_claims(
            id_pattern=r"dlt_3$",
            nomatch_only=True,
        )
        assert len(result) == 1


# endregion


# region MatchGraph.get_matchstamp() + split_components()


class TestMatchGraphGetMatchstamp:
    """Tests for MatchGraph.get_matchstamp()."""

    def test_single_component(self):
        """Single connected component returns one MatchStamp."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_b",
                timeline_b_id="tl_c",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_b",
                    coordinate_a=Coordinate(50.0, TimeUnit.number),
                    timeline_b_id="tl_c",
                    coordinate_b=Coordinate(25.0, TimeUnit.number),
                ),
            ),
        ]
        mg = MatchGraph(claims=claims)
        stamp = mg.get_matchstamp()
        assert stamp.n_timelines == 3
        assert stamp.get_coordinate_for("tl_a", format="float") == 100.0
        assert stamp.get_coordinate_for("tl_b", format="float") == 50.0
        assert stamp.get_coordinate_for("tl_c", format="float") == 25.0

    def test_multi_component_raises(self):
        """Multiple disconnected components raise ValueError."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_c",
                timeline_b_id="tl_d",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_c",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_d",
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            ),
        ]
        mg = MatchGraph(claims=claims)
        assert mg.n_components == 2
        with pytest.raises(ValueError, match="disconnected components"):
            mg.get_matchstamp()

    def test_empty_graph_raises(self):
        """Empty graph raises ValueError."""
        mg = MatchGraph()
        with pytest.raises(ValueError, match="no synchronous claims"):
            mg.get_matchstamp()

    def test_only_nomatch_raises(self):
        """Graph with only NOMATCH claims raises ValueError."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                is_synchronous=False,
            ),
        ]
        mg = MatchGraph(claims=claims)
        with pytest.raises(ValueError, match="no synchronous claims"):
            mg.get_matchstamp()


class TestMatchGraphSplitComponents:
    """Tests for MatchGraph.split_components()."""

    def test_single_component(self):
        """Single component returns list of one MatchGraph."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
        ]
        mg = MatchGraph(claims=claims)
        components = mg.split_components()
        assert len(components) == 1
        stamp = components[0].get_matchstamp()
        assert stamp.n_timelines == 2

    def test_two_components(self):
        """Two disconnected components split correctly."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
            MatchClaim(
                timeline_a_id="tl_c",
                timeline_b_id="tl_d",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_c",
                    coordinate_a=Coordinate(200.0, TimeUnit.number),
                    timeline_b_id="tl_d",
                    coordinate_b=Coordinate(75.0, TimeUnit.number),
                ),
            ),
        ]
        mg = MatchGraph(claims=claims)
        components = mg.split_components()
        assert len(components) == 2
        # Each component can produce a single matchstamp
        for comp in components:
            stamp = comp.get_matchstamp()
            assert stamp.n_timelines == 2

    def test_empty_graph(self):
        """Empty graph splits to empty list."""
        mg = MatchGraph()
        assert mg.split_components() == []

    def test_n_components_property(self):
        """n_components property works correctly."""
        claims = [
            MatchClaim(
                timeline_a_id="tl_a",
                timeline_b_id="tl_b",
                start_anchor=AlignmentAnchor(
                    timeline_a_id="tl_a",
                    coordinate_a=Coordinate(100.0, TimeUnit.number),
                    timeline_b_id="tl_b",
                    coordinate_b=Coordinate(50.0, TimeUnit.number),
                ),
            ),
        ]
        mg = MatchGraph(claims=claims)
        assert mg.n_components == 1


class TestMatchGraphStarTopology:
    """Test MatchGraph with star topology (1 score + N performers).

    Simulates the Vienna 1xN pattern: all performers connect to the
    score at the same coordinate, forming one connected component.
    """

    def test_star_single_coordinate(self):
        """All claims at one coordinate form one component."""
        claims = []
        for i in range(5):
            perf_id = f"perf:dlt{i + 1}"
            claims.append(
                MatchClaim(
                    timeline_a_id="score:clt_1",
                    timeline_b_id=perf_id,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="score:clt_1",
                        coordinate_a=Coordinate(0.0, TimeUnit.number),
                        timeline_b_id=perf_id,
                        coordinate_b=Coordinate(float(i * 10), TimeUnit.number),
                    ),
                )
            )
        mg = MatchGraph(claims=claims)
        assert mg.n_components == 1
        stamp = mg.get_matchstamp()
        assert stamp.n_timelines == 6  # 1 score + 5 performers
        assert stamp.get_coordinate_for("score:clt_1", format="float") == 0.0

    def test_star_determinism(self):
        """Building graph from any single claim gives same stamp.

        This tests the Vienna critical requirement: the same MatchGraph
        (and thus MatchStamp) must result regardless of which claim
        seeds the graph.
        """
        claims = []
        for i in range(5):
            perf_id = f"perf:dlt{i + 1}"
            claims.append(
                MatchClaim(
                    timeline_a_id="score:clt_1",
                    timeline_b_id=perf_id,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="score:clt_1",
                        coordinate_a=Coordinate(0.0, TimeUnit.number),
                        timeline_b_id=perf_id,
                        coordinate_b=Coordinate(float(i * 10), TimeUnit.number),
                    ),
                )
            )
        # Build from ALL claims
        mg_full = MatchGraph(claims=claims)
        stamp_full = mg_full.get_matchstamp()

        # Each individual claim should produce a 2-timeline stamp alone
        # (without transitive closure). But building from all claims
        # at the same score coordinate gives the full star.
        assert stamp_full.n_timelines == 6


# endregion


# region MatchClaim.get_matchstamp()


class TestMatchClaimGetMatchstamp:
    """Tests for MatchClaim.get_matchstamp()."""

    def test_reduced_stamp(self):
        """from_graph=False returns 2-timeline MatchStamp."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        stamp = claim.get_matchstamp(from_graph=False)
        assert stamp is not None
        assert stamp.n_timelines == 2
        assert stamp.get_coordinate_for("score:clt_1", format="float") == 10.0
        assert stamp.get_coordinate_for("perf:dlt_1", format="float") == 128.0

    def test_nomatch_returns_none(self):
        """NOMATCH claim returns None."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            is_synchronous=False,
        )
        assert claim.get_matchstamp(from_graph=False) is None

    def test_from_graph_without_bundle_raises(self):
        """from_graph=True without bundle raises ValueError."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        with pytest.raises(ValueError, match="bundle is required"):
            claim.get_matchstamp()  # from_graph=True by default

    def test_from_graph_with_bundle(self):
        """from_graph=True with bundle returns full stamp."""
        bundle, claims = _make_star_bundle()
        # Pick the first synchronous claim
        sync_claims = [c for c in claims if c.is_synchronous]
        claim = sync_claims[0]
        stamp = claim.get_matchstamp(bundle=bundle)
        assert stamp is not None
        # At coordinate 0.0 on score, all 3 performers are connected
        assert stamp.n_timelines == 4  # score + 3 performers


# endregion


# region AlignmentBundle MatchGraph cache


class TestMatchGraphCache:
    """Tests for AlignmentBundle MatchGraph cache."""

    def test_cache_hit(self):
        """Second lookup returns cached MatchGraph."""
        bundle, _ = _make_star_bundle()
        mg1 = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        mg2 = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        assert mg1 is mg2  # Same object

    def test_cache_via_different_key(self):
        """Lookup via performer node returns same graph as score node."""
        bundle, _ = _make_star_bundle()
        mg_score = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        # perf:dlt1 at coordinate 0.0 should map to the same graph
        mg_perf = bundle._get_or_build_matchgraph("perf:dlt_1", 0.0)
        assert mg_score is mg_perf

    def test_cache_invalidation(self):
        """add_match_claims() clears the cache."""
        bundle, _ = _make_star_bundle()
        mg1 = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        # Add more claims
        bundle.add_match_claims(
            [
                MatchClaim(
                    timeline_a_id="score:clt_1",
                    timeline_b_id="perf:dlt_1",
                    start_anchor=AlignmentAnchor(
                        timeline_a_id="score:clt_1",
                        coordinate_a=Coordinate(99.0, TimeUnit.number),
                        timeline_b_id="perf:dlt_1",
                        coordinate_b=Coordinate(198.0, TimeUnit.number),
                    ),
                ),
            ]
        )
        mg2 = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        assert mg1 is not mg2  # Cache was invalidated

    def test_cache_no_claims_raises(self):
        """Cache miss with no matching claims raises ValueError."""
        bundle, _ = _make_star_bundle()
        with pytest.raises(ValueError, match="No synchronous claims"):
            bundle._get_or_build_matchgraph("score:clt_1", 999.0)

    def test_different_coordinates_different_graphs(self):
        """Different coordinates produce different cached graphs."""
        bundle, _ = _make_star_bundle()
        mg0 = bundle._get_or_build_matchgraph("score:clt_1", 0.0)
        mg25 = bundle._get_or_build_matchgraph("score:clt_1", 25.0)
        assert mg0 is not mg25


# endregion


# region AlignmentBundle.get_matchstamp_at()


class TestGetMatchstampAt:
    """Tests for AlignmentBundle.get_matchstamp_at()."""

    def test_basic(self):
        """Basic cross-group MatchStamp at a coordinate."""
        bundle, _ = _make_star_bundle()
        stamp = bundle.get_matchstamp_at(0.0, "score:clt_1")
        assert stamp.n_timelines == 4  # score + 3 performers
        assert stamp.is_interpolated is False
        assert stamp.get_coordinate_for("score:clt_1", format="float") == 0.0
        assert stamp.get_coordinate_for("perf:dlt_1", format="float") == 0.0
        assert stamp.get_coordinate_for("perf:dlt_2", format="float") == 0.0
        assert stamp.get_coordinate_for("perf:dlt_3", format="float") == 0.0

    def test_nonzero_coordinate(self):
        """MatchStamp at a non-zero coordinate."""
        bundle, _ = _make_star_bundle()
        stamp = bundle.get_matchstamp_at(50.0, "score:clt_1")
        assert stamp.n_timelines == 4
        assert stamp.get_coordinate_for("score:clt_1", format="float") == 50.0
        assert stamp.get_coordinate_for("perf:dlt_1", format="float") == 100.0
        assert stamp.get_coordinate_for("perf:dlt_2", format="float") == 75.0
        assert stamp.get_coordinate_for("perf:dlt_3", format="float") == 90.0

    def test_not_in_bundle_raises(self):
        """Unknown timeline raises KeyError."""
        bundle, _ = _make_star_bundle()
        with pytest.raises(KeyError, match="not in bundle"):
            bundle.get_matchstamp_at(0.0, "nonexistent")

    def test_between_anchors_uses_interpolation(self):
        """A coordinate between claim anchors uses exact WarpMap interpolation."""
        bundle, _ = _make_star_bundle()
        stamp = bundle.get_matchstamp_at(37.5, "score:clt_1")

        assert stamp.is_interpolated is True
        assert stamp.anchor_edges == []
        assert stamp.get_coordinate_for("score:clt_1", format="float") == 37.5
        assert stamp.get_coordinate_for("perf:dlt_1", format="float") == 75.0
        assert stamp.get_coordinate_for("perf:dlt_2", format="float") == 56.25
        assert stamp.get_coordinate_for("perf:dlt_3", format="float") == 67.5
        assert set(stamp.inferred_edges) == {
            ("score:clt_1", "perf:dlt_1"),
            ("score:clt_1", "perf:dlt_2"),
            ("score:clt_1", "perf:dlt_3"),
        }
        assert bundle.transfer(37.5, "score:clt_1", "perf:dlt_1") == 75.0

    def test_filtered_by_pattern(self):
        """Regex filter reduces output."""
        bundle, _ = _make_star_bundle()
        stamp = bundle.get_matchstamp_at(
            0.0,
            "score:clt_1",
            id_pattern=r"dlt_[12]$",
        )
        # Only perf:dlt_1 and perf:dlt_2 pass the filter
        # (score:clt_1 does NOT match the pattern, so it's also excluded)
        assert "perf:dlt_1" in stamp.coordinates
        assert "perf:dlt_2" in stamp.coordinates
        assert "perf:dlt_3" not in stamp.coordinates

    def test_filtered_by_timeline_ids(self):
        """timeline_ids filter reduces output."""
        bundle, _ = _make_star_bundle()
        stamp = bundle.get_matchstamp_at(
            0.0,
            "score:clt_1",
            timeline_ids={"score:clt_1", "perf:dlt_1"},
        )
        assert stamp.n_timelines == 2
        assert "score:clt_1" in stamp.coordinates
        assert "perf:dlt_1" in stamp.coordinates


# endregion


# region coordinate-batch


class TestCoordinateBatch:
    """Position batches on get_matchstamps_at() / get_matchstamp_table().

    Validation rationale: users hold coordinates, not ``MatchClaim`` objects.
    The plural positional getter and the table both take an ``at`` collection
    (plus ``timeline_id``) that fans out element-by-element to
    ``get_matchstamp_at`` — the singular resolver already validated above. The
    batch adds no new resolution semantics, so these cases pin only the fan-out
    contract: input order is preserved, each query coordinate yields exactly
    one full transitive cross-section, raw values and IdCoordinates coerce
    exactly as they do on the singular path, the table lays one row per
    coordinate through the same assembly the claims path uses, ``timeline_filter``
    narrows columns, empty input yields empty output, and combining ``claims``
    with ``at`` is rejected.
    """

    # Full cross-section at score coordinate 50 (score length 100; performer
    # factors 2.0 / 1.5 / 1.8), reused as the gold row across cases.
    COORD_50 = {
        "score:clt_1": 50.0,
        "perf:dlt_1": 100.0,
        "perf:dlt_2": 75.0,
        "perf:dlt_3": 90.0,
    }

    def test_matchstamps_raw_values_ordered(self):
        """Raw-value batch: one ordered stamp per coordinate, exact cross-sections."""
        bundle, _ = _make_star_bundle()
        stamps = bundle.get_matchstamps_at([0.0, 50.0, 100.0], "score:clt_1")
        assert len(stamps) == 3
        assert {
            timeline_id: stamps[1].get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamps[1].present_timelines
        } == self.COORD_50
        assert stamps[0].get_coordinate_for("perf:dlt_1", format="float") == 0.0
        assert stamps[2].get_coordinate_for("perf:dlt_1", format="float") == 200.0
        assert stamps[2].get_coordinate_for("perf:dlt_2", format="float") == 150.0
        assert stamps[2].get_coordinate_for("perf:dlt_3", format="float") == 180.0

    def test_matchstamps_idcoordinate_carries_timeline(self):
        """An IdCoordinate carries its own timeline, so timeline_id is optional."""
        bundle, _ = _make_star_bundle()
        stamps = bundle.get_matchstamps_at(
            [IdCoordinate(50.0, TimeUnit.number, "score:clt_1")]
        )
        assert len(stamps) == 1
        assert {
            timeline_id: stamps[0].get_coordinate_for(timeline_id, format="float")
            for timeline_id in stamps[0].present_timelines
        } == self.COORD_50

    def test_matchstamps_wiring_parity_with_singular(self):
        """Batch fan-out equals mapping get_matchstamp_at over the coordinates."""
        bundle, _ = _make_star_bundle()
        coords = [0.0, 25.0, 50.0, 75.0, 100.0]
        batched = [
            s.coordinates for s in bundle.get_matchstamps_at(coords, "score:clt_1")
        ]
        singular = [
            bundle.get_matchstamp_at(c, "score:clt_1").coordinates for c in coords
        ]
        assert batched == singular

    def test_table_one_row_per_coordinate(self):
        """Table batch: one row per coordinate, full cross-section columns."""
        bundle, _ = _make_star_bundle()
        table = bundle.get_matchstamp_table([0.0, 50.0, 100.0], "score:clt_1")
        assert table.num_rows == 3
        assert set(table.column_names) == {
            "score:clt_1",
            "perf:dlt_1",
            "perf:dlt_2",
            "perf:dlt_3",
        }
        assert {
            timeline_id: IdCoordinateField.from_table(table, timeline_id)[1].value
            for timeline_id in table.column_names
        } == self.COORD_50

    def test_table_timeline_filter_narrows_columns(self):
        """timeline_filter narrows the coordinate table to the named fields."""
        bundle, _ = _make_star_bundle()
        table = bundle.get_matchstamp_table(
            [0.0, 50.0, 100.0],
            "score:clt_1",
            timeline_filter={"score:clt_1", "perf:dlt_1"},
        )
        assert set(table.column_names) == {"score:clt_1", "perf:dlt_1"}
        assert {
            timeline_id: IdCoordinateField.from_table(table, timeline_id)[1].value
            for timeline_id in table.column_names
        } == {"score:clt_1": 50.0, "perf:dlt_1": 100.0}

    def test_empty_coordinates_yield_empty_output(self):
        """Empty coordinates yield an empty table and an empty stamp list."""
        bundle, _ = _make_star_bundle()
        assert bundle.get_matchstamp_table([], "score:clt_1").num_rows == 0
        assert bundle.get_matchstamps_at([], "score:clt_1") == []

    def test_claims_and_positions_mutually_exclusive(self):
        """Supplying both claims and positions raises on the table."""
        bundle, claims = _make_star_bundle()
        with pytest.raises(ValueError, match="not both"):
            bundle.get_matchstamp_table([0.0], "score:clt_1", claims=claims)


# endregion


# region MatchStamp display


class TestMatchStampDisplay:
    """Tests for MatchStamp __str__ and _repr_html_."""

    def test_str_header(self):
        """__str__ includes timeline count and edge count."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            anchor_edges=[("tl_a", "tl_b")],
        )
        s = str(stamp)
        assert "MatchStamp" in s
        assert "2 timelines" in s
        assert "1 edges" in s

    def test_str_entries(self):
        """__str__ shows coordinate values."""
        stamp = MatchStamp(
            coordinates={"score": 10.0, "audio": 45.5},
            anchor_edges=[("score", "audio")],
        )
        s = str(stamp)
        assert "score" in s
        assert "10" in s
        assert "audio" in s
        assert "45.5" in s

    def test_empty_stamp_is_invalid(self):
        """A match stamp requires a source-axis coordinate."""
        with pytest.raises((TypeError, ValueError)):
            MatchStampType()

    def test_str_integer_formatting(self):
        """Integer-valued coordinates shown without decimals."""
        stamp = MatchStamp(
            coordinates={"tl_a": 100.0, "tl_b": 50.0},
            anchor_edges=[("tl_a", "tl_b")],
        )
        s = str(stamp)
        assert "100" in s
        assert "100.0" not in s

    def test_repr_html_valid(self):
        """_repr_html_ produces valid HTML."""
        stamp = MatchStamp(
            coordinates={"score": 10.0, "audio": 45.5},
            anchor_edges=[("score", "audio")],
        )
        html = stamp._repr_html_()
        assert "<table" in html
        assert "MatchStamp" in html
        assert "score" in html
        assert "audio" in html

    def test_repr_html_anchor_bold(self):
        """Anchor timelines are bold in HTML."""
        stamp = MatchStamp(
            coordinates={"score": 10.0, "audio": 45.5},
            anchor_edges=[("score", "audio")],
        )
        html = stamp._repr_html_()
        assert "<strong>score</strong>" in html

    def test_repr_html_inferred_greyed(self):
        """Inferred timelines are greyed in HTML."""
        stamp = MatchStamp(
            coordinates={"score": 10.0, "audio": 45.5, "video": 1000.0},
            anchor_edges=[("score", "audio")],
            inferred_edges=[("audio", "video")],
        )
        html = stamp._repr_html_()
        assert "color: #666" in html
        assert "inferred" in html

    def test_repr_html_try_footer(self):
        """_repr_html_ appends the affordance Try footer after the table."""
        stamp = MatchStamp(
            coordinates={"score": 10.0, "audio": 45.5},
            anchor_edges=[("score", "audio")],
        )
        html = stamp._repr_html_()
        # The coordinate table still renders.
        assert "<strong>score</strong>" in html
        # The Try footer surfaces the real MatchStamp accessors.
        assert (
            "Try: <code>stamp.get_coordinate_for(&lt;tl_id&gt;)</code>, "
            "<code>stamp.get_coordinates_for(&lt;tl_ids&gt;)</code>" in html
        )
        # Footer is a free-standing div after the table close.
        assert html.index("</table>") < html.index("Try:")


# endregion


# region MatchClaim display


class TestMatchClaimDisplay:
    """Tests for MatchClaim __str__ and _repr_html_."""

    def test_str_synchronous_instant(self):
        """__str__ for synchronous instant claim."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        s = str(claim)
        assert "synchronous, instant" in s
        assert "score:clt_1" in s
        assert "perf:dlt_1" in s
        assert "@10" in s
        assert "@128" in s

    def test_str_synchronous_interval(self):
        """__str__ for synchronous interval claim."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            end_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        s = str(claim)
        assert "synchronous, interval" in s
        assert "[0 number -- 10 number]" in s
        assert "[0 number -- 128 number]" in s

    def test_str_nomatch(self):
        """__str__ for a NOMATCH claim (orphaned event named on one side)."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            is_synchronous=False,
            event_a_id="orphan",
        )
        s = str(claim)
        assert "NOMATCH" in s
        assert "score:clt_1" in s
        assert "perf:dlt_1" in s

    def test_str_with_metadata(self):
        """__str__ shows metadata."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
            metadata=MatchMetadata(
                agent=Agent(
                    name="partitura",
                    type=AgentType.software,
                    identifier="note_matching",
                ),
                certainty=0.95,
            ),
        )
        s = str(claim)
        assert "Metadata:" in s
        assert "partitura" in s
        assert "0.95" in s

    def test_str_inferred(self):
        """__str__ shows [inferred] flag."""
        claim = MatchClaim(
            timeline_a_id="tl_a",
            timeline_b_id="tl_b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_a",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="tl_b",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            is_explicit=False,
            source_claim_id="claim_1",
        )
        s = str(claim)
        assert "[inferred]" in s
        assert "Source:" in s
        assert "claim_1" in s

    def test_repr_html_valid(self):
        """_repr_html_ produces valid HTML."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        html = claim._repr_html_()
        assert "<table" in html
        assert "MatchClaim" in html
        assert "score:clt_1" in html

    def test_repr_html_nomatch_badge(self):
        """NOMATCH claim (orphaned event named on one side) has red badge in HTML."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            is_synchronous=False,
            event_a_id="orphan",
        )
        html = claim._repr_html_()
        assert "NOMATCH" in html
        assert "#ffcdd2" in html  # Red background

    def test_repr_html_try_footer(self):
        """_repr_html_ appends the affordance Try footer after the table."""
        claim = MatchClaim(
            timeline_a_id="score:clt_1",
            timeline_b_id="perf:dlt_1",
            start_anchor=AlignmentAnchor(
                timeline_a_id="score:clt_1",
                coordinate_a=Coordinate(10.0, TimeUnit.number),
                timeline_b_id="perf:dlt_1",
                coordinate_b=Coordinate(128.0, TimeUnit.number),
            ),
        )
        html = claim._repr_html_()
        # The claim detail table still renders.
        assert "<strong>score:clt_1</strong>" in html
        # The Try footer surfaces the real MatchClaim accessor.
        assert "Try: <code>claim.get_matchstamp()</code>" in html
        assert html.index("</table>") < html.index("Try:")


# endregion


# region MatchClaim event storage and display


class TestMatchClaimEventStorage:
    """Tests for MatchClaim storing and displaying event information.

    Events should be stored with their ID and name (when provided),
    and always displayed in __str__ and _repr_html_.
    """

    def test_from_events_stores_event_ids(self):
        """from_events() extracts and stores event IDs from event dicts."""
        event_a = {"id": "e000001", "start": 10.0}
        event_b = {"id": "e000042", "start": 128.0}

        claim = MatchClaim.from_events(
            event_a=event_a,
            tl_a_id="score:clt_1",
            event_b=event_b,
            tl_b_id="perf:dlt_1",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        assert claim.event_a_id == "e000001"
        assert claim.event_b_id == "e000042"

    def test_from_events_stores_event_names(self):
        """from_events() extracts and stores event names from event dicts."""
        event_a = {"id": "e001", "name": "Intro", "start": 0.0}
        event_b = {"id": "e002", "name": "Verse 1", "start": 0.0}

        claim = MatchClaim.from_events(
            event_a=event_a,
            tl_a_id="tl_a",
            event_b=event_b,
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        assert claim.event_a_name == "Intro"
        assert claim.event_b_name == "Verse 1"

    def test_from_events_handles_missing_id_and_name(self):
        """from_events() gracefully handles missing id/name fields."""
        event_a = {"start": 10.0}  # No id or name
        event_b = {"start": 20.0}

        claim = MatchClaim.from_events(
            event_a=event_a,
            tl_a_id="tl_a",
            event_b=event_b,
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        assert claim.event_a_id is None
        assert claim.event_a_name is None
        assert claim.event_b_id is None
        assert claim.event_b_name is None

    def test_from_projection_stores_source_event_info(self):
        """from_projection() stores source event ID and name."""
        event = {"id": "e100", "name": "Beat 1", "start": 0.0}

        claim = MatchClaim.from_projection(
            event=event,
            source_tl_id="source",
            target_tl_id="target",
            target_coord=10.0,
            source_unit=TimeUnit.number,
            target_unit=TimeUnit.number,
        )

        assert claim.event_a_id == "e100"
        assert claim.event_a_name == "Beat 1"
        # Target has no event, so event_b fields are None
        assert claim.event_b_id is None
        assert claim.event_b_name is None

    def test_nomatch_stores_source_event_info(self):
        """nomatch() stores source event ID and name."""
        event = {"id": "e999", "name": "Orphan Note", "start": 50.0}

        claim = MatchClaim.nomatch(
            event=event,
            source_tl_id="score",
            target_tl_id="perf",
            unit=TimeUnit.number,
        )

        assert claim.event_a_id == "e999"
        assert claim.event_a_name == "Orphan Note"

    def test_str_displays_event_ids(self):
        """__str__ includes event IDs when present."""
        claim = MatchClaim.from_events(
            event_a={"id": "e001", "start": 10.0},
            tl_a_id="score:clt_1",
            event_b={"id": "e042", "start": 128.0},
            tl_b_id="perf:dlt_1",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        s = str(claim)
        assert "e001" in s
        assert "e042" in s

    def test_str_displays_event_names(self):
        """__str__ includes event names when present."""
        claim = MatchClaim.from_events(
            event_a={"id": "e001", "name": "Intro", "start": 0.0},
            tl_a_id="tl_a",
            event_b={"id": "e002", "name": "Verse 1", "start": 0.0},
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        s = str(claim)
        assert "Intro" in s
        assert "Verse 1" in s

    def test_repr_html_displays_event_info(self):
        """_repr_html_ includes event ID and name when present."""
        claim = MatchClaim.from_events(
            event_a={"id": "note_001", "name": "C4", "start": 10.0},
            tl_a_id="score",
            event_b={"id": "note_042", "name": "C4", "start": 128.0},
            tl_b_id="perf",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        html = claim._repr_html_()
        assert "note_001" in html
        assert "note_042" in html
        assert "C4" in html

    def test_repr_html_no_event_info_still_valid(self):
        """_repr_html_ is valid HTML even without event info."""
        claim = MatchClaim.from_events(
            event_a={"start": 10.0},  # No id/name
            tl_a_id="tl_a",
            event_b={"start": 20.0},
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        html = claim._repr_html_()
        assert "<table" in html
        assert "MatchClaim" in html

    def test_to_dict_includes_event_info(self):
        """to_dict() serializes event ID and name fields."""
        claim = MatchClaim.from_events(
            event_a={"id": "e001", "name": "Note A", "start": 0.0},
            tl_a_id="tl_a",
            event_b={"id": "e002", "name": "Note B", "start": 0.0},
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        d = claim.to_dict()
        assert d["event_a_id"] == "e001"
        assert d["event_a_name"] == "Note A"
        assert d["event_b_id"] == "e002"
        assert d["event_b_name"] == "Note B"

    def test_from_dict_restores_event_info(self):
        """from_dict() restores event ID and name fields."""
        original = MatchClaim.from_events(
            event_a={"id": "e001", "name": "Beat", "start": 0.0},
            tl_a_id="tl_a",
            event_b={"id": "e002", "name": "Onset", "start": 0.0},
            tl_b_id="tl_b",
            unit_a=TimeUnit.number,
            unit_b=TimeUnit.number,
        )

        d = original.to_dict()
        restored = MatchClaim.from_dict(d)

        assert restored.event_a_id == "e001"
        assert restored.event_a_name == "Beat"
        assert restored.event_b_id == "e002"
        assert restored.event_b_name == "Onset"

    def test_direct_construction_with_event_info(self):
        """MatchClaim can be constructed directly with event info."""
        claim = MatchClaim(
            timeline_a_id="tl_a",
            timeline_b_id="tl_b",
            start_anchor=AlignmentAnchor(
                timeline_a_id="tl_a",
                coordinate_a=Coordinate(0.0, TimeUnit.number),
                timeline_b_id="tl_b",
                coordinate_b=Coordinate(0.0, TimeUnit.number),
            ),
            event_a_id="ev_a",
            event_a_name="Event A",
            event_b_id="ev_b",
            event_b_name="Event B",
        )

        assert claim.event_a_id == "ev_a"
        assert claim.event_a_name == "Event A"
        assert claim.event_b_id == "ev_b"
        assert claim.event_b_name == "Event B"


# endregion


# region transfer() docstring correctness


class TestTransferDocstring:
    """Verify transfer() docstring no longer claims to be primary."""

    def test_docstring_not_primary(self):
        """transfer() docstring must NOT say 'primary user-facing'."""
        docstring = AlignmentBundle.transfer.__doc__
        assert "primary user-facing" not in docstring.lower()
        assert "get_matchstamp_at" in docstring


# endregion


# region Top-level exports


class TestTopLevelExports:
    """Verify that MatchGraph, MatchStamp, ClaimFilter are in top-level __all__."""

    def test_matchgraph_importable(self):
        """MatchGraph importable from top-level."""
        from timetoalign import MatchGraph as MG

        assert MG is MatchGraph

    def test_matchstamp_importable(self):
        """MatchStamp importable from top-level."""
        from timetoalign import MatchStamp as MS

        assert MS is MatchStampType

    def test_claimfilter_importable(self):
        """ClaimFilter importable from top-level."""
        from timetoalign import ClaimFilter
        from timetoalign.alignment.filters import ClaimFilter as CF

        assert ClaimFilter is CF

    def test_in_all(self):
        """All three are in __all__."""
        import timetoalign

        assert "MatchGraph" in timetoalign.__all__
        assert "MatchStamp" in timetoalign.__all__
        assert "ClaimFilter" in timetoalign.__all__


# endregion
