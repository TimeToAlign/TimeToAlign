"""Parity between the per-claim and columnar cross-group claim stores.

``AlignmentBundle`` keeps cross-group claims in two places: the per-claim
Python list ``cross_group_claims`` and the columnar
``cross_group_claim_fields``.  Which one a loader chooses must be invisible to
callers — a bundle holding a set of claims in the list and a bundle holding
the *same* claims in a ``MatchClaimField`` must answer every public getter
identically.  Only the cost of the answer differs.

This module builds the same nine synchronous instant claims twice, once into
each store, and asserts the two bundles agree on every reader.  Validation
logic is documented in ``tests/alignment/README.md``.
"""

from __future__ import annotations

import pytest

from timetoalign.alignment.bundle import AlignmentBundle
from timetoalign.alignment.claims import (
    Agent,
    AlignmentAnchor,
    MatchClaim,
    MatchClaimField,
    MatchMetadata,
)
from timetoalign.core import AgentType, Coordinate, Domain, IdCoordinateField, TimeUnit
from timetoalign.timelines import (
    ContinuousLogicalTimeline,
    ContinuousPhysicalTimeline,
)

# region Fixture topology

SCORE = "score:clt_1"
PERF_1 = "perf:cpt_1"
PERF_2 = "perf:cpt_2"

#: Complete pairwise topology at three aligned instants.  Each instant is one
#: connected component spanning all three timelines, so ``from_graph=True``
#: collapses nine pairwise rows into three cross-section rows.
INSTANTS: list[dict[str, float]] = [
    {SCORE: 0.0, PERF_1: 1.0, PERF_2: 2.0},
    {SCORE: 4.0, PERF_1: 3.0, PERF_2: 5.0},
    {SCORE: 8.0, PERF_1: 6.5, PERF_2: 9.0},
]

_UNITS = {
    SCORE: TimeUnit.quarters,
    PERF_1: TimeUnit.seconds,
    PERF_2: TimeUnit.seconds,
}
_PAIRS = [(SCORE, PERF_1), (SCORE, PERF_2), (PERF_1, PERF_2)]


def _make_claims() -> list[MatchClaim]:
    """Build the nine claims fresh (never shared between two bundles)."""
    metadata = MatchMetadata(
        agent=Agent(name="parity", type=AgentType.software, identifier="fixture"),
    )
    claims = []
    for instant in INSTANTS:
        for tl_a, tl_b in _PAIRS:
            claims.append(
                MatchClaim(
                    timeline_a_id=tl_a,
                    timeline_b_id=tl_b,
                    start_anchor=AlignmentAnchor(
                        timeline_a_id=tl_a,
                        coordinate_a=Coordinate(instant[tl_a], _UNITS[tl_a]),
                        timeline_b_id=tl_b,
                        coordinate_b=Coordinate(instant[tl_b], _UNITS[tl_b]),
                    ),
                    is_synchronous=True,
                    metadata=metadata,
                )
            )
    return claims


def _make_bundle(bundle_id: str) -> AlignmentBundle:
    """Build the three-timeline, three-group bundle without any claims."""
    bundle = AlignmentBundle(id=bundle_id)
    bundle.add_timeline(
        ContinuousLogicalTimeline(length=12, uid=SCORE, unit=TimeUnit.quarters),
        uid=SCORE,
        as_group="score_group",
    )
    bundle.add_timeline(
        ContinuousPhysicalTimeline(length=12, uid=PERF_1, unit=TimeUnit.seconds),
        uid=PERF_1,
        as_group="perf1_group",
    )
    bundle.add_timeline(
        ContinuousPhysicalTimeline(length=12, uid=PERF_2, unit=TimeUnit.seconds),
        uid=PERF_2,
        as_group="perf2_group",
    )
    return bundle


@pytest.fixture()
def bundle_list() -> AlignmentBundle:
    """Bundle whose claims live in the per-claim Python list."""
    bundle = _make_bundle("parity_list")
    bundle.add_match_claims(_make_claims())
    return bundle


@pytest.fixture()
def bundle_field() -> AlignmentBundle:
    """Bundle whose identical claims live in a columnar MatchClaimField."""
    bundle = _make_bundle("parity_field")
    bundle.add_match_claim_field(MatchClaimField.from_claims(_make_claims()))
    return bundle


def _keys(claims: list[MatchClaim]) -> list[tuple]:
    """Sorted normalised keys — alignment content, free of generated ids."""
    keys = []
    for claim in claims:
        anchor = claim.start_anchor
        keys.append(
            (
                claim.timeline_a_id,
                claim.timeline_b_id,
                None if anchor is None else float(anchor.coordinate_a.value),
                None if anchor is None else float(anchor.coordinate_b.value),
                claim.is_synchronous,
            )
        )
    return sorted(keys)


#: One case per filter kwarg, with the exact number of claims it must select.
FILTER_CASES: list[tuple[str, dict, int]] = [
    ("timeline_id", {"timeline_id": PERF_1}, 6),
    ("timeline_ids", {"timeline_ids": {PERF_1, PERF_2}}, 9),
    ("id_pattern", {"id_pattern": r"^score:"}, 6),
    ("between", {"between": (PERF_2, PERF_1)}, 3),
    ("synchronous_only", {"synchronous_only": True}, 9),
    ("nomatch_only", {"nomatch_only": True}, 0),
    ("include_domains", {"include_domains": {Domain.physical}}, 3),
    ("include_units", {"include_units": {TimeUnit.seconds}}, 3),
    ("unknown_uid", {"timeline_id": "absent:cpt_9"}, 0),
]

# endregion


# region Store placement


def test_n_cross_group_claims(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    assert bundle_list.n_cross_group_claims == 9
    assert bundle_field.n_cross_group_claims == 9


def test_store_placement(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    """The two fixtures really do use two different layouts."""
    assert len(bundle_list.cross_group_claims) == 9
    assert bundle_list.cross_group_claim_fields == []
    assert bundle_field.cross_group_claims == []
    assert len(bundle_field.cross_group_claim_fields) == 1
    assert len(bundle_field.cross_group_claim_fields[0]) == 9


# endregion


# region get_match_claims / get_claim_fields


def test_get_match_claims_unfiltered(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    from_list = bundle_list.get_match_claims()
    from_field = bundle_field.get_match_claims()
    assert len(from_list) == 9
    assert len(from_field) == 9
    assert _keys(from_list) == _keys(from_field)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [(kwargs, expected) for _, kwargs, expected in FILTER_CASES],
    ids=[name for name, _, _ in FILTER_CASES],
)
def test_get_match_claims_filtered(
    bundle_list: AlignmentBundle,
    bundle_field: AlignmentBundle,
    kwargs: dict,
    expected: int,
) -> None:
    from_list = bundle_list.get_match_claims(**kwargs)
    from_field = bundle_field.get_match_claims(**kwargs)
    assert len(from_list) == expected
    assert _keys(from_list) == _keys(from_field)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [(kwargs, expected) for _, kwargs, expected in FILTER_CASES],
    ids=[name for name, _, _ in FILTER_CASES],
)
def test_get_claim_fields_row_count_matches_claims(
    bundle_field: AlignmentBundle,
    kwargs: dict,
    expected: int,
) -> None:
    """The vectorized accessor selects exactly what the materialising one does."""
    fields = bundle_field.get_claim_fields(**kwargs)
    assert sum(len(claim_field) for claim_field in fields) == expected


def test_get_claim_fields_empty_on_list_bundle(bundle_list: AlignmentBundle) -> None:
    assert bundle_list.get_claim_fields() == []


def test_get_claim_fields_drops_empty_fields(bundle_field: AlignmentBundle) -> None:
    assert bundle_field.get_claim_fields(nomatch_only=True) == []
    assert bundle_field.get_claim_fields(timeline_id="absent:cpt_9") == []


# endregion


# region Stamps and tables


def test_get_matchstamps(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    def coordinate_maps(bundle: AlignmentBundle) -> list[tuple]:
        return sorted(
            tuple(sorted(stamp.coordinates.items()))
            for stamp in bundle.get_matchstamps()
        )

    from_list = coordinate_maps(bundle_list)
    assert len(from_list) == 9
    assert from_list == coordinate_maps(bundle_field)


def test_matchstamp_table_per_claim(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    expected = [
        {
            PERF_1: coordinates.get(PERF_1),
            PERF_2: coordinates.get(PERF_2),
            SCORE: coordinates.get(SCORE),
        }
        for instant in INSTANTS
        for coordinates in (
            {SCORE: instant[SCORE], PERF_1: instant[PERF_1]},
            {SCORE: instant[SCORE], PERF_2: instant[PERF_2]},
            {PERF_1: instant[PERF_1], PERF_2: instant[PERF_2]},
        )
    ]
    for bundle in (bundle_list, bundle_field):
        table = bundle.get_matchstamp_table()
        assert table.num_rows == 9
        assert table.column_names == [PERF_1, PERF_2, SCORE]
        # Each pairwise claim fills exactly two cells and leaves one null.
        assert _coordinate_rows(table) == expected


def test_matchstamp_table_from_graph(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    expected = [
        {PERF_1: 1.0, PERF_2: 2.0, SCORE: 0.0},
        {PERF_1: 3.0, PERF_2: 5.0, SCORE: 4.0},
        {PERF_1: 6.5, PERF_2: 9.0, SCORE: 8.0},
    ]
    for bundle in (bundle_list, bundle_field):
        table = bundle.get_matchstamp_table(from_graph=True)
        assert table.num_rows == 3
        assert table.column_names == [PERF_1, PERF_2, SCORE]
        assert _coordinate_rows(table) == expected


def test_matchstamp_table_timeline_filter(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    for bundle in (bundle_list, bundle_field):
        per_claim = bundle.get_matchstamp_table(timeline_filter={SCORE})
        assert per_claim.column_names == [SCORE]
        assert per_claim.num_rows == 9
        collapsed = bundle.get_matchstamp_table(
            timeline_filter={SCORE}, from_graph=True
        )
        assert collapsed.column_names == [SCORE]
        assert _coordinate_rows(collapsed) == [
            {SCORE: 0.0},
            {SCORE: 4.0},
            {SCORE: 8.0},
        ]


def test_get_matchstamp_at(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    for bundle in (bundle_list, bundle_field):
        for instant in INSTANTS:
            stamp = bundle.get_matchstamp_at(instant[SCORE], SCORE)
            assert stamp.n_timelines == 3
            for timeline_id, coordinate in instant.items():
                assert stamp.get_coordinate(timeline_id).value == coordinate


# endregion


# region Commensurability and transfer


def test_are_commensurable(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    for bundle in (bundle_list, bundle_field):
        assert bundle.are_commensurable(SCORE, PERF_1) is True
        assert bundle.are_commensurable(SCORE, PERF_2) is True
        assert bundle.are_commensurable(PERF_1, PERF_2) is True
        assert bundle.are_commensurable(SCORE, "absent:cpt_9") is False


def test_transfer_round_trip(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    """The MatchLine -> WarpMap path, dead for columnar bundles before."""
    for bundle in (bundle_list, bundle_field):
        assert bundle.transfer(4.0, SCORE, PERF_1) == 3.0
        assert bundle.transfer(3.0, PERF_1, SCORE) == 4.0


def test_diagram_reports_claim_count(
    bundle_list: AlignmentBundle, bundle_field: AlignmentBundle
) -> None:
    assert "MatchClaims: 9" in str(bundle_list.diagram())
    assert "MatchClaims: 9" in str(bundle_field.diagram())


# endregion
def _coordinate_rows(table) -> list[dict[str, float | None]]:
    """Materialize semantic coordinate columns as exact numeric rows."""
    columns = {
        name: IdCoordinateField.from_table(table, name) for name in table.column_names
    }
    return [
        {
            name: None if field[position] is None else field[position].value
            for name, field in columns.items()
        }
        for position in range(table.num_rows)
    ]
